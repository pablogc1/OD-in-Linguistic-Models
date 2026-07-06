#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Summary Report Generator (V6 - SOD Only)

Generates comprehensive analysis outputs:
1. summary_report.txt with distributional statistics
2. log_binned_histogram_data.csv for plots

V6 UPDATE: Only processes SOD metrics (WOD removed).
"""
import gc
import pandas as pd
import itertools
import os
from tqdm import tqdm
import numpy as np

# --- Configuration ---
EOD_RESULTS_FILE = "eod_results_final.csv"
PAIRWISE_DIFF_DIR = "pairwise_diff_results"
COMMON_VOCAB_FILE = "curated_corpora/common_vocabulary.txt"
FINAL_REPORT_FILE = "summary_report.txt"
LOG_HISTOGRAM_FILE = "log_binned_histogram_data.csv"

CORPORA_KEYS = ["ground_filtered", "null_model", "random_removal", "targeted_removal", "ai_generated", "merriam_webster"]
CHUNK_SIZE = 5_000_000

def main():
    """
    Main function to generate all final analysis outputs.
    """
    # --- Pre-flight Checks ---
    if not all(os.path.exists(f) for f in [EOD_RESULTS_FILE, PAIRWISE_DIFF_DIR, COMMON_VOCAB_FILE]):
        print("FATAL: One or more required input files or directories are missing.")
        return

    all_log_hist_data = []

    with open(FINAL_REPORT_FILE, "w") as f_report:
        print("--- Generating Final, All-Inclusive Summary Report ---")

        # =============================================================
        # Section 1: Global Analysis Statistics
        # =============================================================
        print("Calculating Section 1: Global Stats...")
        f_report.write("="*80 + "\nSECTION 1: GLOBAL ANALYSIS STATISTICS\n" + "="*80 + "\n\n")
        with open(COMMON_VOCAB_FILE, "r") as vocab_f:
            num_common_words = len(vocab_f.readlines())
        total_possible_pairs = (num_common_words * (num_common_words - 1)) // 2
        f_report.write(f"{'Total Common Words:':<35} {num_common_words:,}\n")
        f_report.write(f"{'Total Possible Word Pairs:':<35} {total_possible_pairs:,}\n\n")

        # =============================================================
        # Section 2: Eigen Ontological Differentiation (EOD) Summary
        # =============================================================
        print("Calculating Section 2: EOD Summary...")
        f_report.write("="*80 + "\nSECTION 2: EIGEN ONTOLOGICAL DIFFERENTIATION (EOD) SUMMARY\n" + "="*80 + "\n")
        try:
            df_eod = pd.read_csv(EOD_RESULTS_FILE)
            eod_summary_data = []
            for key_A, key_B in itertools.combinations(CORPORA_KEYS, 2):
                combo_key = f"{key_A}_vs_{key_B}"
                score_col, tl_col = f"eod_score_{combo_key}", f"eod_tl_{combo_key}"
                valid_runs = df_eod[df_eod[score_col] > -1]
                eod_summary_data.append({
                    "Corpus Combination": combo_key, "Valid Words": f"{len(valid_runs):,}",
                    "Mean EOD Score": valid_runs[score_col].mean(), "Median EOD Score": valid_runs[score_col].median(),
                    "Mean EOD TL": valid_runs[tl_col].mean(), "Median EOD TL": valid_runs[tl_col].median()
                })
            df_eod_summary = pd.DataFrame(eod_summary_data)
            f_report.write(df_eod_summary.to_string(index=False, float_format="%.2f") + "\n")
            del df_eod, df_eod_summary, eod_summary_data
            gc.collect()
        except Exception as e:
            f_report.write(f"Could not process EOD results due to an error: {e}\n")

        # =============================================================
        # Section 3: Inter-Corpus Pairwise Difference Summary
        # =============================================================
        print("\nCalculating Section 3: Full Pairwise Difference Summary...")
        f_report.write("\n\n" + "="*80 + "\nSECTION 3: INTER-CORPUS PAIRWISE DIFFERENCE SUMMARY\n" + "="*80 + "\n")
        
        # V6: Only SOD metrics (no WOD)
        metrics = ['sod', 'sod_tl']
        
        for key_A, key_B in itertools.combinations(CORPORA_KEYS, 2):
            combo_key = f"{key_A}_vs_{key_B}"
            f_report.write("\n\n" + "-"*80 + f"\nComparison: {combo_key.upper()}\n" + "-"*80 + "\n")
            
            diff_file = os.path.join(PAIRWISE_DIFF_DIR, f"diffs_{combo_key}.csv")
            if not os.path.exists(diff_file):
                f_report.write("  -> Difference file not found. Skipping.\n")
                continue

            # --- Pass 1: Memory-efficient Mean calculation ---
            print(f"  ({combo_key}) Pass 1/2: Calculating Mean via chunking...")
            net_sum = pd.Series(0.0, index=[f'diff_{m}' for m in metrics])
            total_count = 0
            try:
                reader = pd.read_csv(diff_file, sep=',', chunksize=CHUNK_SIZE)
                for chunk in tqdm(reader, desc=f"  Mean ({combo_key})"):
                    net_sum += chunk[[f'diff_{m}' for m in metrics]].sum()
                    total_count += len(chunk)
                net_mean = net_sum / total_count if total_count > 0 else net_sum
                coverage = (total_count / total_possible_pairs) * 100 if total_possible_pairs > 0 else 0
                f_report.write(f"  Pairs Compared: {total_count:,} ({coverage:.2f}% coverage)\n\n")
            except Exception as e:
                f_report.write(f"  -> ERROR during mean calculation: {e}. Skipping.\n")
                continue

            # --- Pass 2: Exact calculations, one metric at a time, freeing memory between each ---
            print(f"  ({combo_key}) Pass 2/2: Calculating Median, IQR, and Outliers...")
            for metric in metrics:
                col = f'diff_{metric}'
                f_report.write(f"  --- Metric: {metric.upper()} ---\n")
                try:
                    data_series = pd.read_csv(diff_file, sep=',', usecols=[col]).squeeze("columns")
                    
                    net_median = data_series.median()
                    abs_series = data_series.abs()
                    del data_series
                    gc.collect()
                    
                    abs_mean = abs_series.mean()
                    abs_median = abs_series.median()
                    
                    log_abs_series = np.log10(abs_series.values + 1)
                    del abs_series
                    gc.collect()
                    
                    q1 = np.quantile(log_abs_series, 0.25)
                    q3 = np.quantile(log_abs_series, 0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outlier_mask = (log_abs_series < lower_bound) | (log_abs_series > upper_bound)
                    num_outliers = int(outlier_mask.sum())
                    del outlier_mask
                    
                    f_report.write(f"    Net Mean:       {net_mean[col]:,.2f}\n")
                    f_report.write(f"    Net Median:     {net_median:,.2f}\n")
                    f_report.write(f"    Absolute Mean:  {abs_mean:,.2f}\n")
                    f_report.write(f"    Absolute Median:{abs_median:,.2f}\n")
                    f_report.write(f"    Log10 IQR:      {iqr:.4f}\n")
                    f_report.write(f"    Extreme Outliers: {num_outliers:,} ({num_outliers/total_count:.4%})\n")

                    max_log = int(np.ceil(log_abs_series.max()))
                    bins = np.arange(0, max_log + 2)
                    hist_counts, _ = np.histogram(log_abs_series, bins=bins)
                    del log_abs_series
                    gc.collect()
                    
                    for i, count in enumerate(hist_counts):
                        all_log_hist_data.append({
                            'comparison': combo_key, 'metric': col,
                            'log10_bin_start': bins[i], 'count': int(count)
                        })

                except Exception as e:
                    f_report.write(f"    -> ERROR processing column {col}: {e}\n")

    # --- Write the separate log histogram data file ---
    if all_log_hist_data:
        print(f"\nWriting log histogram data to '{LOG_HISTOGRAM_FILE}'...")
        df_log_hist = pd.DataFrame(all_log_hist_data)
        df_log_hist.to_csv(LOG_HISTOGRAM_FILE, index=False)
        print("Log histogram data file created successfully.")

    print("\n--- Definitive Summary Report Generation Complete! ---")

if __name__ == "__main__":
    main()
