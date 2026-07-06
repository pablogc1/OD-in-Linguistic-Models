#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log Histogram Worker (V6 - SOD Only)

Generates log-binned histogram data for corpus comparisons.
V6 UPDATE: Only processes SOD metrics (WOD removed).
"""
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate log-binned histogram data (V6 - SOD only).")
    parser.add_argument("--comparison_key", type=str, required=True)
    args = parser.parse_args()

    # --- Configuration ---
    input_dir = "pairwise_diff_results"
    output_dir = "temp_loghist_parts"
    chunk_size = 5_000_000
    
    os.makedirs(output_dir, exist_ok=True)
    input_file = os.path.join(input_dir, f"diffs_{args.comparison_key}.csv")
    output_file = os.path.join(output_dir, f"loghist_{args.comparison_key}.csv")
    
    if not os.path.exists(input_file):
        print(f"INFO: Input file not found, skipping: {input_file}")
        return

    print(f"Reading {input_file} in chunks...")
    
    # V6: Only SOD metrics (no WOD)
    metrics = ['sod', 'sod_tl']
    # Use a dictionary of counters to store binned data for all metrics
    # Using collections.Counter to accumulate counts per bin
    from collections import Counter
    binned_data_collector = {f'diff_{m}': Counter() for m in metrics}

    try:
        reader = pd.read_csv(input_file, sep=',', chunksize=chunk_size)

        for chunk in tqdm(reader, desc=f"Processing {args.comparison_key}"):
            for metric in metrics:
                col = f'diff_{metric}'
                if col not in chunk.columns: continue

                abs_diffs = chunk[col].abs()
                log_abs_diffs = np.log10(abs_diffs + 1)
                
                # Dynamically bin this chunk's data
                max_val = int(np.ceil(log_abs_diffs.max())) if not log_abs_diffs.empty else 0
                bins = np.arange(0, max_val + 2)
                
                if len(bins) > 1:
                    binned_series = pd.cut(log_abs_diffs, bins=bins, right=False, labels=bins[:-1])
                    hist_counts = binned_series.value_counts()
                    # Add chunk's counts to the overall counter
                    binned_data_collector[col].update(hist_counts.to_dict())
    
    except Exception as e:
        print(f"ERROR: Could not read or process {input_file}. Error: {e}")
        return

    # Now, format the final histogram data
    all_final_hist_data = []
    for metric in metrics:
        col = f'diff_{metric}'
        if not binned_data_collector[col]: continue

        print(f"Finalizing histogram for: {col}")
        
        # Sort bins and add to final list
        for bin_start in sorted(binned_data_collector[col].keys()):
            count = binned_data_collector[col][bin_start]
            if count > 0:
                all_final_hist_data.append({
                    'comparison': args.comparison_key,
                    'metric': col,
                    'log10_bin': bin_start,
                    'count': count
                })

    if all_final_hist_data:
        df_out = pd.DataFrame(all_final_hist_data)
        df_out.to_csv(output_file, index=False)
        print(f"Successfully wrote log-histogram data to {output_file}")

if __name__ == "__main__":
    main()
