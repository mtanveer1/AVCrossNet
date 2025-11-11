import pandas as pd
import numpy as np

def compute_metrics_by_snr_range(metrics_file, snr_file, output_file='snr_range_metrics_demucs.csv'):
    """
    Compute mean of STOI, PESQ, and SISDR metrics across different SNR ranges.

    Parameters:
    -----------
    metrics_file : str
        Path to CSV file with format [scene_name, stoi, pesq, sisdr]
    snr_file : str
        Path to CSV file with format [scene_name, snr]
    output_file : str
        Path to save the output results
    """

    # Define SNR ranges
    snr_ranges = [
        (-30, -24),
        (-24, -18),
        (-18, -12),
        (-12, -6),
        (-6, 0),
        (0, 6),
        (6, 12),
        (12, 18)
    ]

    # Read the CSV files without headers
    metrics_df = pd.read_csv(metrics_file, header=None,names=['scene_name', 'stoi', 'pesq', 'sisdr'])
    snr_df = pd.read_csv(snr_file, header=None, names=['scene_name', 'snr'])

    # Convert numeric columns to float (handles string conversion issues)
    metrics_df['stoi'] = pd.to_numeric(metrics_df['stoi'], errors='coerce')
    metrics_df['pesq'] = pd.to_numeric(metrics_df['pesq'], errors='coerce')
    metrics_df['sisdr'] = pd.to_numeric(metrics_df['sisdr'], errors='coerce')
    snr_df['snr'] = pd.to_numeric(snr_df['snr'], errors='coerce')

    # Remove any rows with NaN values after conversion
    metrics_df = metrics_df.dropna()
    snr_df = snr_df.dropna()

    print(f"Loaded {len(metrics_df)} rows from metrics file")
    print(f"Loaded {len(snr_df)} rows from SNR file")

    # Merge the two dataframes on scene_name
    merged_df = pd.merge(metrics_df, snr_df, on='scene_name', how='inner')
    print(f"\nMerged dataframe has {len(merged_df)} rows")

    # Display first few rows
    print("\nFirst 5 rows of merged data:")
    print(merged_df.head())

    # Display SNR statistics
    print(f"\nSNR statistics:")
    print(f"Min SNR: {merged_df['snr'].min():.2f}")
    print(f"Max SNR: {merged_df['snr'].max():.2f}")
    print(f"Mean SNR: {merged_df['snr'].mean():.2f}")

    # Compute metrics for each SNR range
    results = []

    for snr_min, snr_max in snr_ranges:
        # Filter data within the SNR range (inclusive lower bound, exclusive upper bound)
        mask = (merged_df['snr'] >= snr_min) & (merged_df['snr'] < snr_max)
        range_data = merged_df[mask]

        count = len(range_data)

        if count > 0:
            mean_stoi = range_data['stoi'].mean()
            mean_pesq = range_data['pesq'].mean()
            mean_sisdr = range_data['sisdr'].mean()

            std_stoi = range_data['stoi'].std()
            std_pesq = range_data['pesq'].std()
            std_sisdr = range_data['sisdr'].std()
        else:
            mean_stoi = mean_pesq = mean_sisdr = np.nan
            std_stoi = std_pesq = std_sisdr = np.nan

        results.append({
            'snr_range': f'[{snr_min}, {snr_max})',
            'snr_min': snr_min,
            'snr_max': snr_max,
            'count': count,
            'mean_stoi': mean_stoi,
            'std_stoi': std_stoi,
            'mean_pesq': mean_pesq,
            'std_pesq': std_pesq,
            'mean_sisdr': mean_sisdr,
            'std_sisdr': std_sisdr
        })

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Display results
    print("\n" + "="*80)
    print("RESULTS: Mean Metrics by SNR Range")
    print("="*80)
    print(f"\n{'SNR Range':<15} {'Count':<8} {'STOI':<12} {'PESQ':<12} {'SISDR':<12}")
    print("-"*80)

    for _, row in results_df.iterrows():
        if row['count'] > 0:
            print(f"{row['snr_range']:<15} {row['count']:<8} "
                  f"{row['mean_stoi']:<12.4f} {row['mean_pesq']:<12.4f} {row['mean_sisdr']:<12.4f}")
        else:
            print(f"{row['snr_range']:<15} {row['count']:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

    print("="*80)

    # Save results to CSV
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Also create a simple summary table
    summary_df = results_df[['snr_range', 'count', 'mean_stoi', 'mean_pesq', 'mean_sisdr']].copy()
    summary_file = output_file.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

    return results_df, merged_df


if __name__ == "__main__":
    # Example usage:
    # Replace these filenames with your actual file paths
    metrics_file = "metrics_demucs.csv"  # File with [scene_name, stoi, pesq, sisdr]
    snr_file = "snr_results.csv"           # File with [scene_name, snr]

    # Run the analysis
    results_df, merged_df = compute_metrics_by_snr_range(metrics_file, snr_file)

    # Optional: Print additional statistics
    print("\nOverall Statistics:")
    print("-"*80)
    print(f"Total scenes processed: {len(merged_df)}")
    print(f"Overall mean STOI: {merged_df['stoi'].mean():.4f}")
    print(f"Overall mean PESQ: {merged_df['pesq'].mean():.4f}")
    print(f"Overall mean SISDR: {merged_df['sisdr'].mean():.4f}")*
