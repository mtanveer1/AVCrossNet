#NEW STUFF
import math
from pathlib import Path
import random
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
import librosa
import numpy as np
import torch
import torchaudio
from einops.layers.torch import EinMix
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from utils.dnn import BasicBlock, ResNet, Swish, cal_si_snr
from huggingface_hub import hf_hub_download
from torch.nn import MultiheadAttention
from timm.models.layers import trunc_normal_, DropPath

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global pooling, output is (B, C, 1, 1)

        # Two fully connected layers with a reduction in channel dimensionality
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, H, W = x.size()

        # Global average pooling
        y = self.global_avg_pool(x).view(batch_size, channels)  # (B, C)

        # Fully connected layers
        y = self.fc1(y)  # (B, C // r)
        y = self.relu(y)
        y = self.fc2(y)  # (B, C)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)  # (B, C, 1, 1)

        # Scale the input by the learned weights
        return x * y  # Element-wise multiplication with input feature map



# ConvNeXt Block with SE integration
class Block(nn.Module):
    def __init__(self, dim, drop_path=0., reduction=16):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # SE block
        self.se_block = SEBlock(dim, reduction)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        # Pass through SE block
        x = self.se_block(x)

        # Add the residual connection with stochastic depth (DropPath)
        x = input + self.drop_path(x)
        return x


class CustomConvNeXtV2(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[48, 96, 192, 384],
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)

            # Check if the layer has a bias term
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x.mean([-2,-1])) # global average pooling, (N, C, H, W) -> (N, C)
        x = self.head(x)
        return x
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        return x


#MAIN AUDIO MODEL
@torch.compiler.disable
def fft_conv(equation, input, kernel, *args):
    input, kernel = input.float(), kernel.float()
    args = tuple(arg.cfloat() for arg in args)
    n = input.shape[-1]

    kernel_f = torch.fft.rfft(kernel, 2 * n)
    input_f = torch.fft.rfft(input, 2 * n)
    output_f = torch.einsum(equation, input_f, kernel_f, *args)
    output = torch.fft.irfft(output_f, 2 * n)

    return output[..., :n]


def ssm_basis_kernels(A, B, log_dt, length):
    log_A_real, A_imag = A.T  # (2, num_coeffs)
    lrange = torch.arange(length, device=A.device)
    dt = log_dt.exp()

    dtA_real, dtA_imag = -dt * F.softplus(log_A_real), dt * A_imag
    return (dtA_real[:, None] * lrange).exp() * torch.cos(dtA_imag[:, None] * lrange), B * dt[:, None]


def opt_ssm_forward(input, K, B_hat, C):
    """SSM ops with einsum contractions
    """
    batch, c_in, _ = input.shape
    c_out, coeffs = C.shape

    if (1 / c_in + 1 / c_out) > (1 / batch + 1 / coeffs):
        if c_in * c_out <= coeffs:
            kernel = torch.einsum('dn,nc,nl->dcl', C, B_hat, K)
            return fft_conv('bcl,dcl->bdl', input, kernel)
    else:
        if coeffs <= c_in:
            x = torch.einsum('bcl,nc->bnl', input, B_hat)
            x = fft_conv('bnl,nl->bnl', x, K)
            return torch.einsum('bnl,dn->bdl', x, C)

    return fft_conv('bcl,nl,nc,dn->bdl', input, K, B_hat, C)


class SSMLayer(nn.Module):
    def __init__(self,
                 num_coeffs: int,
                 in_channels: int,
                 out_channels: int,
                 repeat: int):
        from torch.backends import opt_einsum
        assert opt_einsum.is_available()
        opt_einsum.strategy = 'optimal'

        super().__init__()

        init_parameter = lambda mat: Parameter(torch.tensor(mat, dtype=torch.float))
        normal_parameter = lambda fan_in, shape: Parameter(torch.randn(*shape) * math.sqrt(2 / fan_in))

        A_real, A_imag = 0.5 * np.ones(num_coeffs), math.pi * np.arange(num_coeffs)
        log_A_real = np.log(np.exp(A_real) - 1)  # inv softplus
        B = np.ones(num_coeffs)
        A = np.stack([log_A_real, A_imag], -1)
        log_dt = np.linspace(np.log(0.001), np.log(0.1), repeat)

        A = np.tile(A, (repeat, 1))
        B = np.tile(B[:, None], (repeat, in_channels)) / math.sqrt(in_channels)
        log_dt = np.repeat(log_dt, num_coeffs)

        self.log_dt, self.A, self.B = init_parameter(log_dt), init_parameter(A), init_parameter(B)
        self.C = normal_parameter(num_coeffs * repeat, (out_channels, num_coeffs * repeat))

    def forward(self, input):
        K, B_hat = ssm_basis_kernels(self.A, self.B, self.log_dt, input.shape[-1])
        return opt_ssm_forward(input, K, B_hat, self.C)


class LayerNormFeature(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.layer_norm = nn.LayerNorm(features)

    def forward(self, input):
        return self.layer_norm(input.moveaxis(-1, -2)).moveaxis(-1, -2)


class Denoiser(nn.Module):
    def __init__(self,
                 CCA,
                 UDA,
                 in_channels=1,
                 channels=[16, 32, 64, 96, 128, 256],
                 num_coeffs=16,
                 repeat=16,
                 resample_factors=[4, 4, 2, 2, 2, 2],
                 pre_conv=True):
        super().__init__()
        self.CCA = CCA
        self.UDA = UDA(CCA)
        depth = len(channels)
        self.depth = depth
        self.channels = [in_channels] + channels
        self.num_coeffs = num_coeffs
        self.repeat = repeat
        self.pre_conv = pre_conv
        self.mha = MultiheadAttention(embed_dim=100, num_heads=4, batch_first=True)
        self.down_linear = nn.Linear(100, 1)

        self.down_ssms = nn.ModuleList([
            self.ssm_pool(c_in, c_out, r, downsample=True) for (c_in, c_out, r) in zip(self.channels[:-1], self.channels[1:], resample_factors)
        ])
        self.up_ssms = nn.ModuleList([
            self.ssm_pool(c_in, c_out, r, downsample=False) for (c_in, c_out, r) in zip(self.channels[1:], self.channels[:-1], resample_factors)
        ])
        self.hid_ssms = nn.Sequential(
            self.ssm_block(self.channels[-1], True), self.ssm_block(self.channels[-1], True),
        )
        self.last_ssms = nn.Sequential(
            self.ssm_block(self.channels[0], True), self.ssm_block(self.channels[0], False),
        )

    def ssm_pool(self, in_channels, out_channels, resample_factor, downsample=True):
        if downsample:
            return nn.Sequential(
                self.ssm_block(in_channels, use_activation=True),
                EinMix('b c (t r) -> b d t', weight_shape='c d r', c=in_channels, d=out_channels, r=resample_factor),
            )
        else:
            return nn.Sequential(
                EinMix('b c t -> b d (t r)', weight_shape='c d r', c=in_channels, d=out_channels, r=resample_factor),
                self.ssm_block(out_channels, use_activation=True),
            )

    def ssm_block(self, channels, use_activation=False):
        block = nn.Sequential()
        if channels > 1 and self.pre_conv:
            block.append(nn.Conv1d(channels, channels, 3, 1, 1, groups=channels))
        block.append(SSMLayer(self.num_coeffs, channels, channels, self.repeat))
        if use_activation:
            if channels > 1:
                block.append(LayerNormFeature(channels))
            block.append(nn.SiLU())

        return block

    def upscale_tensor(self,input_tensor):
        # Reshape the input tensor to (batch, 256*k, 1)
        # print("ishape ", input_tensor.shape) #(175,batch,256)
        batch = input_tensor.shape[1]
        reshaped_tensor = input_tensor.reshape(batch, 256 * input_tensor.shape[0], 1)
        # print(reshaped_tensor.shape)
        # Apply a linear layer to transform the tensor to (256*k, 100)
        # linear_layer = nn.Linear(reshaped_tensor.shape[-1], 100).to('cuda:1')
        # output_tensor = linear_layer(reshaped_tensor)
        return reshaped_tensor #shape(batch, 44800, 1), as UDA downsample converts to (batch,112,100)

    def downscale_tensor(self,x, k, batch):
        # Step 2: Reshape the tensor to the desired shape (k, batch, 256)
        x = x.view(k, batch, 256)  # Reshape to (k, batch, 256)
        return x

    def forward(self, input,vf):
        x, skips = input, []

        # encoder
        for ssm in self.down_ssms:
            skips.append(x)
            x = ssm(x)

        # neck
        x = x.permute(2, 0, 1)
        k,batch,_ = x.shape
        x = self.upscale_tensor(x)
        self.UDA = self.UDA
        x = self.UDA(x,vf)
        x = self.downscale_tensor(x,k,batch)
        x = x.permute(1, 2, 0)
        x = self.hid_ssms(x)
        # decoder

        for (ssm, skip) in zip(self.up_ssms[::-1], skips[::-1]):
            x = ssm[0](x)
            x = x + skip
            x = ssm[1](x)

        return self.last_ssms(x)

    def denoise_single(self, noisy):
        assert noisy.ndim == 2, f"noisy input should be shaped (samples, length)"
        noisy = noisy[:, None, :]  # unsqueeze channel dim

        padding = 256 - noisy.shape[-1] % 256
        noisy = F.pad(noisy, (0, padding))
        denoised = self.forward(noisy)

        return denoised.squeeze(1)[..., :-padding]

    def denoise_multiple(self, noisy_samples):
        audio_lens = [noisy.shape[-1] for noisy in noisy_samples]
        max_len = max(audio_lens)
        noisy_samples = torch.stack([F.pad(noisy, (0, max_len - noisy.shape[-1])) for noisy in noisy_samples])
        denoised_samples = self.denoise_single(noisy_samples)

        return [denoised[..., :audio_len] for (denoised, audio_len) in zip(denoised_samples, audio_lens)]

    def denoise(self, noisy_dir, denoised_dir=None):
        noisy_dir = Path(noisy_dir)
        denoised_dir = None if denoised_dir is None else Path(denoised_dir)

        noisy_files = [fn for fn in noisy_dir.glob('*.wav')]
        noisy_samples = [torch.tensor(librosa.load(wav_file, sr=16000)[0]) for wav_file in noisy_files]
        print("denoising...")
        denoised_samples = self.denoise_multiple(noisy_samples)

        if denoised_dir is not None:
            print("saving audio files...")
            for (denoised, noisy_fn) in zip(denoised_samples, noisy_files):
                torchaudio.save(denoised_dir / f"{noisy_fn.stem}.wav", denoised[None, :], 16000)

        return denoised_samples

    def from_pretrained(self, repo_id):
        print(f"loading weights from {repo_id}...")
        model_weights_path = hf_hub_download(repo_id=repo_id, filename="weights.pt")
        self.load_state_dict(torch.load(model_weights_path), strict = False)
        print("Denoiser loaded from pretrained!")

##CUSTOM ATTENTION
from collections.abc import ValuesView
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CustomCrossAttention(nn.Module):
    def __init__(self, da, dv):
        super(CustomCrossAttention, self).__init__()
        self.W_A_g = nn.Parameter(torch.randn(1, da + dv))
        self.W_V_g = nn.Parameter(torch.randn(1, dv + da))
        self.b_A_g = nn.Parameter(torch.randn(1))
        self.b_V_g = nn.Parameter(torch.randn(1))

    def showgrads(self):
        print(self.W_A_g.grad)
        print(self.W_V_g.grad)
        print(self.b_A_g.grad)
        print(self.b_V_g.grad)

    def forward(self, A, V):
        batch_size, T, da = A.shape
        _, _, dv = V.shape
        # print("dv: ",dv)
        # Mean and covariance for visual features
        mu_v = torch.mean(V, axis=1).to(A.device)  # (batch_size, dv)
        Sigma_v = torch.stack([torch.cov(V[b].permute(1, 0)) for b in range(batch_size)]).to(A.device)  # (batch_size, dv, dv)

        # Sample visual features from Gaussian
        visual_features_gaussian = torch.stack([
            torch.tensor(np.random.multivariate_normal(mu_v[b].detach().cpu(), Sigma_v[b].detach().cpu(), T)).to(A.device)
            for b in range(batch_size)
        ])  # (batch_size, T, dv)

        # Query and key matrices for cross-attention
        Q_a = A.to(A.device)  # (batch_size, T, da)
        K_v = visual_features_gaussian.to(A.device)  # (batch_size, T, dv)

        # Compute scaled dot-product attention
        scaled_dot_product_av = torch.matmul(Q_a.float(), K_v.permute(0, 2, 1).float()) / torch.sqrt(torch.tensor(dv, dtype=torch.float32))

        attention_weights_av = F.softmax(scaled_dot_product_av, dim=-1)  # (batch_size, T, T)

        weighted_visual_values_av = torch.matmul(attention_weights_av, V)  # (batch_size, T, dv)

        # Normalize by Z_a (Z_a is 1 in this case)
        attention_result_av = weighted_visual_values_av

        # Mean and covariance for audio features
        mu_a = torch.mean(A, axis=1)  # (batch_size, da)
        Sigma_a = torch.stack([torch.cov(A[b].permute(1, 0)) for b in range(batch_size)])  # (batch_size, da, da)

        # Sample audio features from Gaussian
        audio_features_gaussian = torch.stack([
            torch.tensor(np.random.multivariate_normal(mu_a[b].detach().cpu(), Sigma_a[b].detach().cpu(), T))
            for b in range(batch_size)
        ])  # (batch_size, T, da)

        # Query and key matrices for cross-attention (visual-to-audio)
        Q_v = V  # (batch_size, T, dv)
        K_a = audio_features_gaussian  # (batch_size, T, da)

        # Compute scaled dot-product attention
        scaled_dot_product_va = torch.matmul(Q_v.float().to(A.device), K_a.permute(0, 2, 1).float().to(A.device)) / torch.sqrt(torch.tensor(da, dtype=torch.float32).to(A.device))  # (batch_size, T, T)
        attention_weights_va = F.softmax(scaled_dot_product_va, dim=-1).to(A.device)  # (batch_size, T, T)

        weighted_audio_values_va = torch.matmul(attention_weights_va, A).to(A.device)  # (batch_size, T, da)

        # Normalize by Z_v (Z_v is 1 in this case)
        attention_result_va = weighted_audio_values_va.to(A.device)

        # Gated cross-attention fusion
        fused_audio = []
        fused_visual = []
        combined_features = []

        for b in range(batch_size):
            fused_audio_b = []
            fused_visual_b = []
            combined_features_b = []

            for t in range(T):
                # Original and attended features
                a_t = A[b, t].to(A.device)
                v_t = V[b, t].to(A.device)
                a_t_att = attention_result_va[b, t].to(A.device)
                v_t_att = attention_result_av[b, t].to(A.device)

                # Gates
                concat_a = torch.cat((a_t, v_t_att)).to(A.device)  # (da + dv)
                g_A_t = torch.sigmoid(torch.matmul(self.W_A_g, concat_a) + self.b_A_g).to(A.device)  # Gate for audio
                concat_v = torch.cat((v_t, a_t_att)).to(A.device)  # (dv + da)
                g_V_t = torch.sigmoid(torch.matmul(self.W_V_g, concat_v) + self.b_V_g).to(A.device) # Gate for visual

                # Fused audio and visual
                a_fused_t = g_A_t * a_t + (1 - g_A_t) * v_t_att
                v_fused_t = g_V_t * v_t + (1 - g_V_t) * a_t_att

                fused_audio_b.append(a_fused_t)
                fused_visual_b.append(v_fused_t)

                # Cosine similarity and influence coefficients
                S_at_vt = F.cosine_similarity(a_t, v_t, dim=0)
                alpha_t = torch.exp(S_at_vt) / (torch.exp(S_at_vt) + 1)
                beta_t = 1 / (torch.exp(S_at_vt) + 1)

                # Combined feature mean
                mu_f = alpha_t * a_fused_t + beta_t * v_fused_t

                # Combine features
                combined_features_b.append(mu_f)

            fused_audio.append(torch.stack(fused_audio_b))
            fused_visual.append(torch.stack(fused_visual_b))
            combined_features.append(torch.stack(combined_features_b))

        fused_audio = torch.stack(fused_audio)
        fused_visual = torch.stack(fused_visual)
        combined_features = torch.stack(combined_features)

        return combined_features

##CCA - UDAModule

class UDA(nn.Module):
    def __init__(self, blackbox):
        super(UDA, self).__init__()
        self.blackbox = blackbox(100,100)

    def upsample(self, input_tensor, len_audio, len_video):
        """
        Upsamples the input tensor from shape (batch, len_video, 100) to (batch, len_audio, 1)
        by first reshaping and then interpolating.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch, len_video, 100)
            len_audio (int): Target length for the audio dimension (len_audio)
            len_video (int): Original length of the video dimension (len_video)

        Returns:
            torch.Tensor: Upsampled tensor of shape (batch, len_audio, 1)
        """

        reshaped_tensor = input_tensor.view(-1, 100 * len_video)
        upsampled_tensor = F.interpolate(reshaped_tensor.unsqueeze(1), size=(len_audio,), mode='linear', align_corners=False)
        final_output = upsampled_tensor.unsqueeze(-1)

        return final_output[:,0,...]


    def downsample(self, input_tensor, len_audio, len_video):
        """
        Downsamples the input tensor from shape (batch, len_audio, 1) to (batch, len_video, 100)
        by interpolating to a higher resolution and reshaping.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch, len_audio, 1)
            len_audio (int): Original length of the audio dimension
            len_video (int): Target length for the video dimension

        Returns:
            torch.Tensor: Downsampled tensor of shape (batch, len_video, 100)
        """

        input_tensor = input_tensor.squeeze(-1)
        resized_tensor = F.interpolate(input_tensor.unsqueeze(1), size=(100 * len_video,), mode='linear', align_corners=False)
        reshaped_tensor = resized_tensor.view(-1, len_video, 100)
        return reshaped_tensor

    def forward(self, input_vec, input_vec_small):
        B, L , _ = input_vec.shape
        B, vf, _ = input_vec_small.shape
        downscaled_input = self.downsample(input_vec, L, vf)
        blackbox_output = self.blackbox(downscaled_input,input_vec_small)
        final_output = input_vec + self.upsample(blackbox_output, L, vf)
        return final_output

##VFE - VideoFrameEncoder

class VideoFrameEncoder(nn.Module):
    def __init__(self, image_encoder, num_classes):
        super(VideoFrameEncoder, self).__init__()
        self.image_encoder = image_encoder(in_chans=1, num_classes=num_classes, depths=[3, 3, 6, 3], dims=[48, 96, 192, 384], drop_path_rate=0.)  # Your image encoder

    def forward(self, x):
        B, N, C, H, W = x.size()  # B=batch size, N=number of frames, C=channels, H=height, W=width
        # Reshape to (B * N, C, H, W), treat each frame as a separate image
        x = x.view(B * N, C, H, W)
        # Encode each frame using the image encoder
        encoded_frames = self.image_encoder(x)  # Shape: (B * N, Encoded_Dim)
        # Reshape back to (B, N, Encoded_Dim) to match video structure
        encoded_frames = encoded_frames.view(B, N, -1)
        return encoded_frames  # Shape: (B, N, Encoded_Dim)

#NEW STUFF ENDS
#Modified this class according to new model
class AVSE(nn.Module):
    def __init__(self):
        super(AVSE, self).__init__()
        self.UDA = UDA
        self.CCA = CustomCrossAttention
        self.VFE = VideoFrameEncoder(CustomConvNeXtV2,100)
        self.Denoiser = Denoiser(self.CCA,self.UDA)
        
    def load_denoiser_og(self):
        self.Denoiser.from_pretrained("PeaBrane/aTENNuate")

    def forward(self, input):
        a0 = input["noisy_audio"]
        video_frames = input["video_frames"]
        v0 = video_frames.permute(0,2,1,3,4)
        self.VFE = self.VFE
        v1 = self.VFE(v0)
        v1 = v1
        self.Denoiser = self.Denoiser
        a1 = self.Denoiser(a0.unsqueeze(1),v1)
        return a1[:,0,...]


class AVSEModule(LightningModule):
    def __init__(self, lr=0.001, val_dataset=None):
        super(AVSEModule, self).__init__()
        self.lr = lr
        self.val_dataset = val_dataset
        self.loss = cal_si_snr
        self.model = AVSE()

    def load_denoiser_og(self):
        self.model.load_denoiser_og()

    def forward(self, data):
        """ Processes the input tensor x and returns an output tensor."""
        est_source = self.model(data)
        return est_source

    def training_step(self, batch_inp, batch_idx):
        loss = self.cal_loss(batch_inp)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch_inp, batch_idx):
        loss = self.cal_loss(batch_inp)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def enhance(self, data):
        inputs = dict(noisy_audio=torch.tensor(data["noisy_audio"][None, ...]).to(self.device),
                      video_frames=torch.tensor(data["video_frames"][None, ...]).to(self.device))
        estimated_audio = self(inputs).cpu().numpy()
        estimated_audio /= np.max(np.abs(estimated_audio))
        return estimated_audio

    def on_training_epoch_end(self, outputs):
        if self.val_dataset is not None:
            with torch.no_grad():
                tensorboard = self.logger.experiment
                for index in range(5):
                    rand_int = random.randint(0, len(self.val_dataset))
                    data = self.val_dataset[rand_int]
                    estimated_audio = self.enhance(data)
                    tensorboard.add_audio("{}/{}_clean".format(self.current_epoch, index),
                                          data["clean"][np.newaxis, ...],
                                          sample_rate=16000)
                    tensorboard.add_audio("{}/{}_noisy".format(self.current_epoch, index),
                                          data["noisy_audio"][np.newaxis, ...],
                                          sample_rate=16000)
                    tensorboard.add_audio("{}/{}_enhanced".format(self.current_epoch, index),
                                          estimated_audio.reshape(-1)[np.newaxis, ...],
                                          sample_rate=16000)

    def cal_loss(self, batch_inp):
        mask = batch_inp["clean"].T
        pred_mask = self(batch_inp).T.reshape(mask.shape)
        loss = self.loss(pred_mask.unsqueeze(2), mask.unsqueeze(2))
        loss[loss < -30] = -30
        return torch.mean(loss)

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.8, patience=5),
                "monitor": "val_loss_epoch",
            },
        }
