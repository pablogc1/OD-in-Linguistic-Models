"""
Aggregate Diff Summary (V6 - SOD Only)

Aggregates semantic difference results by mutual SOD level.
V6 UPDATE: Only processes SOD metrics (WOD removed).

Input format (V6): idx_A,idx_B,mutual_sod_level,avg_sod_diff,avg_sod_term_diff
"""
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- Configuration ---
INPUT_CSV = "combined_semantic_diff_results.csv"
FINAL_OUTPUT_FILE = "semantic_difference_summary.txt"
CHUNK_SIZE = 1_000_000

def aggregate_summaries(input_filepath):
    """
    Reads the large CSV in chunks and aggregates statistics by SOD level.
    V6: Only SOD (no WOD).
    """
    if not os.path.exists(input_filepath):
        print(f"FATAL ERROR: Input file '{input_filepath}' not found.")
        return None

    sod_level_stats = {}
    
    # V6 columns to average
    cols_to_average = ['avg_sod_diff', 'avg_sod_term_diff']

    print("Processing large CSV in chunks...")
    try:
        total_size = os.path.getsize(input_filepath)
        total_chunks = (total_size // (CHUNK_SIZE * 50)) + 1
    except FileNotFoundError:
        total_chunks = None

    reader = pd.read_csv(input_filepath, chunksize=CHUNK_SIZE, low_memory=False)

    for chunk in tqdm(reader, total=total_chunks, desc="Reading chunks"):
        # Filter out invalid mutual levels (<= 0)
        chunk = chunk[chunk['mutual_sod_level'] > 0]
        if chunk.empty:
            continue

        # Aggregate by SOD level
        sod_grouped = chunk.groupby('mutual_sod_level')[cols_to_average].agg(['sum', 'count'])
        for level, group_data in sod_grouped.iterrows():
            if level not in sod_level_stats:
                sod_level_stats[level] = {col: {'sum': 0, 'count': 0} for col in cols_to_average}
            for col in cols_to_average:
                sod_level_stats[level][col]['sum'] += group_data[(col, 'sum')]
                sod_level_stats[level][col]['count'] += group_data[(col, 'count')]

    print("Finished reading all chunks. Finalizing statistics...")

    # Finalize SOD Summary
    sod_summary_records = []
    for level, data in sorted(sod_level_stats.items()):
        record = {'Mutual SOD Level': level}
        record['Pair Count'] = data[cols_to_average[0]]['count']
        for col in cols_to_average:
            total_sum = data[col]['sum']
            total_count = data[col]['count']
            mean_val = total_sum / total_count if total_count > 0 else 0
            record[f'Mean {col}'] = mean_val
        sod_summary_records.append(record)
    
    return pd.DataFrame(sod_summary_records)

def format_and_save_summary(sod_summary, output_filepath):
    """Formats the summary DataFrame into a text report."""
    float_format = "{:.4f}".format
    
    with open(output_filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Semantic Difference Analysis Summary (V6 - SOD Only)".center(80) + "\n")
        f.write("Grouped by Mutual SOD Level".center(80) + "\n")
        f.write("=" * 80 + "\n\n")
        
        if sod_summary is None or sod_summary.empty:
            f.write("No data was aggregated.\n")
        else:
            sod_summary.columns = [col.replace('_', ' ').title() for col in sod_summary.columns]
            report_str = sod_summary.to_string(index=False, float_format=float_format)
            f.write(report_str)
            
    print(f"Final summary report saved to '{output_filepath}'.")

if __name__ == "__main__":
    sod_summary = aggregate_summaries(INPUT_CSV)
    format_and_save_summary(sod_summary, FINAL_OUTPUT_FILE)
    
    if sod_summary is not None and not sod_summary.empty:
        print("\n--- Preview of Final SOD Summary ---")
        sod_summary.columns = [col.replace('_', ' ').title() for col in sod_summary.columns]
        print(sod_summary.to_string(index=False, float_format="{:.4f}".format))
