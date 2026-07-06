"""
Aggregate Reports (V6 - SOD Only)

Aggregates level coincidence results from the agreement analysis.
V6 UPDATE: Only processes SOD (WOD removed).
"""
import pandas as pd
from tqdm import tqdm
import re
import os

# --- Configuration ---
FINAL_REPORT_FILE = "detailed_agreement_report.txt"
SUMMARY_OUTPUT_FILE = "aggregated_summary_by_level.txt"

def parse_and_aggregate_reports(report_filepath):
    """
    Reads the multi-report file and aggregates SOD statistics.
    V6: Only SOD (no WOD).
    """
    sod_aggregator = {}
    
    print(f"Starting aggregation process for '{report_filepath}'...")
    print("Reading file and aggregating SOD statistics...")

    try:
        total_size = os.path.getsize(report_filepath)
    except FileNotFoundError:
        print(f"Error: Input file '{report_filepath}' not found. Aborting.")
        return {}

    with open(report_filepath, 'r', encoding='utf-8') as f:
        current_mutual_level = None
        is_in_table = False

        pbar = tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc="Aggregating")
        
        for line in f:
            pbar.update(len(line.encode('utf-8')))
            line = line.strip()

            if not line:
                continue

            # V6: Only SOD analysis marker
            if "--- ANALYSIS: SOD ---" in line:
                is_in_table = False
                continue

            if "Mutual Termination Level:" in line:
                try:
                    level_match = re.search(r'(-?\d+)', line)
                    current_mutual_level = int(level_match.group(1)) if level_match else -1
                except (AttributeError, ValueError):
                    current_mutual_level = -1
                continue
            
            if "--- Agreement Analysis Results ---" in line:
                is_in_table = True
                header_line = next(f, '')
                pbar.update(len(header_line.encode('utf-8')))
                continue

            if "----------------------------------" in line:
                is_in_table = False
                continue

            if is_in_table and current_mutual_level is not None:
                try:
                    parts = line.split()
                    if len(parts) < 3: 
                        continue

                    partner_level = int(parts[0])
                    total_pairs = int(parts[1])
                    matching_pairs = int(parts[2])
                    
                    if current_mutual_level not in sod_aggregator:
                        sod_aggregator[current_mutual_level] = {}
                    if partner_level not in sod_aggregator[current_mutual_level]:
                        sod_aggregator[current_mutual_level][partner_level] = {'total': 0, 'match': 0}
                        
                    sod_aggregator[current_mutual_level][partner_level]['total'] += total_pairs
                    sod_aggregator[current_mutual_level][partner_level]['match'] += matching_pairs

                except (ValueError, IndexError):
                    continue
    
    pbar.close()
    print("\nAggregation complete.")
    return sod_aggregator

def format_and_save_summary(aggregator, output_file_handle):
    """
    Formats aggregated results into tables and writes to file.
    """
    output_file_handle.write("="*80 + "\n")
    output_file_handle.write("AGGREGATED SUMMARY FOR: SOD (V6)\n")
    output_file_handle.write("="*80 + "\n\n")

    if not aggregator:
        output_file_handle.write("No data was aggregated.\n\n")
        return

    for mutual_level in sorted(aggregator.keys()):
        if mutual_level < 0: 
            continue

        output_file_handle.write(f"--- AGGREGATED REPORT FOR MUTUAL LEVEL {mutual_level} ---\n")
        level_data = aggregator[mutual_level]
        records = []
        for partner_level in sorted(level_data.keys()):
            total = level_data[partner_level]['total']
            match = level_data[partner_level]['match']
            avg_pct = (match / total) * 100 if total > 0 else 0
            records.append({
                'Termination Level': partner_level,
                'Total Partner Pairs': total,
                'Total Matching Partner Pairs': match,
                'Avg Match Percentage (%)': avg_pct
            })
        df = pd.DataFrame(records)
        output_file_handle.write(df.to_string(index=False, float_format="%.2f"))
        output_file_handle.write("\n--------------------------------------------\n\n")

if __name__ == "__main__":
    sod_results = parse_and_aggregate_reports(FINAL_REPORT_FILE)
    
    with open(SUMMARY_OUTPUT_FILE, 'w') as f:
        format_and_save_summary(sod_results, f)
        
    print(f"Summary saved to '{SUMMARY_OUTPUT_FILE}'.")
    
    if os.path.exists(SUMMARY_OUTPUT_FILE):
        print("\n" + "="*80)
        print("              FINAL AGGREGATED SUMMARY (SOD)")
        print("="*80)
        with open(SUMMARY_OUTPUT_FILE, 'r') as f:
            print(f.read())
