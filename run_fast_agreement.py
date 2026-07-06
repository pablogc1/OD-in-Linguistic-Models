"""
Fast Agreement Analysis (V6 - SOD Only)

Analyzes level coincidence between word pairs using pre-built indices.
V6 UPDATE: Only processes SOD (WOD removed).

MEMORY FIX: Writes results incrementally to a temp file, then renames.
This avoids buffering millions of results in memory.

Index format (V6): partner_idx,sod_score,sod_term_level
"""
import os
import argparse
import itertools
from multiprocessing import Pool
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- Globals for worker processes ---
worker_word_map = None
worker_index_dir = None

def init_worker(word_map, index_dir):
    """Initializes global variables for each worker process."""
    global worker_word_map, worker_index_dir
    worker_word_map = word_map
    worker_index_dir = index_dir

def generate_detailed_report(df_merged, word_a, word_b):
    """Takes the merged dataframe and computes the detailed results table."""
    if df_merged.empty: 
        return pd.DataFrame()
    confusion_matrix = pd.crosstab(df_merged['level_A'], df_merged['level_B'])
    total_counts = confusion_matrix.sum(axis=1)
    if total_counts.empty: 
        return pd.DataFrame()

    all_levels = sorted(list(set(confusion_matrix.index) | set(confusion_matrix.columns)))
    cm_sq = confusion_matrix.reindex(index=all_levels, columns=all_levels, fill_value=0)
    match_counts = pd.Series(np.diag(cm_sq), index=cm_sq.index)
    match_counts = match_counts.reindex(total_counts.index).fillna(0).astype(int)
    agreement_pct = (match_counts / total_counts) * 100
    
    return pd.DataFrame({
        'Termination Level': total_counts.index,
        f'Total Pairs for {word_a}': total_counts.values,
        f'Matching Pairs with {word_b}': match_counts.values,
        'Match Percentage (%)': agreement_pct.values
    }).reset_index(drop=True)

def worker_process_fast_agreement(pair):
    """Worker function using PRE-BUILT INDEX. V6: SOD only."""
    word_a, word_b = pair
    idx_a = worker_word_map.get(word_a)
    idx_b = worker_word_map.get(word_b)
    final_report = f"--- PAIR REPORT: {word_a} ({idx_a}) vs {word_b} ({idx_b}) ---\n"
    
    final_report += "--- ANALYSIS: SOD ---\n"
    try:
        path_a = os.path.join(worker_index_dir, f"{idx_a}.csv")
        path_b = os.path.join(worker_index_dir, f"{idx_b}.csv")
        
        df_a = pd.read_csv(path_a, header=None, names=['partner_idx', 'sod_score', 'level_A']).set_index('partner_idx')
        df_b = pd.read_csv(path_b, header=None, names=['partner_idx', 'sod_score', 'level_B']).set_index('partner_idx')

        mutual_level = df_a.loc[idx_b, 'level_A'] if idx_b in df_a.index else -1
        final_report += f"Mutual Termination Level: {mutual_level}\n"
        
        df_merged = df_a[['level_A']].join(df_b[['level_B']], how='inner')
        detailed_df = generate_detailed_report(df_merged, word_a, word_b)

        if detailed_df.empty:
            final_report += "Agreement Analysis: FAILED (No common partner words).\n"
        else:
            final_report += "--- Agreement Analysis Results ---\n"
            final_report += detailed_df.to_string(index=False, float_format="%.2f") + "\n"
            final_report += "----------------------------------\n"

    except FileNotFoundError:
        final_report += "Agreement Analysis: FAILED (Index file not found for one or both words).\n"
    except Exception as e:
        final_report += f"Agreement Analysis: FAILED (An unexpected error occurred: {e}).\n"

    final_report += f"--- END REPORT: {word_a} vs {word_b} ---\n\n"
    return final_report

def main():
    parser = argparse.ArgumentParser(description="Run FAST agreement analysis (V6 - SOD only).")
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--total_jobs", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--index_dir", type=str, required=True)
    args = parser.parse_args()

    # Load vocab
    ordered_words = []
    with open(args.vocab_file, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                head = line.split(":", 1)[0].strip().lower()
                if head not in ordered_words: 
                    ordered_words.append(head)
    word_to_index_map = {word: i + 1 for i, word in enumerate(ordered_words)}
    
    # Generate all pairs dynamically using an iterator to save memory
    all_pairs_iter = itertools.combinations(ordered_words, 2)
    total_pairs = len(ordered_words) * (len(ordered_words) - 1) // 2
    
    # Distribute Work
    chunk_size = (total_pairs + args.total_jobs - 1) // args.total_jobs
    start = (args.job_id - 1) * chunk_size
    end = min(total_pairs, start + chunk_size)
    
    print(f"Skipping to pair {start:,}...")
    items_for_this_job = list(itertools.islice(all_pairs_iter, start, end))

    output_file = f"fast_agreement_results_job_{args.job_id}.txt"
    temp_file = f".{output_file}.tmp"
    sentinel_file = f"{output_file}.done"
    
    print(f"Job {args.job_id}/{args.total_jobs}: Processing {len(items_for_this_job):,} pairs using {args.num_workers} workers...")
    
    # MEMORY FIX: Write incrementally to temp file instead of buffering
    count = 0
    init_args = (word_to_index_map, args.index_dir)
    
    # Use maxtasksperchild to prevent pandas memory leaks in workers over long runs
    with Pool(processes=args.num_workers, initializer=init_worker, initargs=init_args) as pool:
        with open(temp_file, "w") as f:
            results_iterator = pool.imap_unordered(worker_process_fast_agreement, items_for_this_job)
            for result in tqdm(results_iterator, total=len(items_for_this_job), desc=f"Job {args.job_id}"):
                f.write(result)
                count += 1
    
    # Atomic rename: temp file -> final file (only complete file gets the final name)
    os.rename(temp_file, output_file)
    
    # Write sentinel file ONLY after output file is complete
    with open(sentinel_file, "w") as f:
        f.write(f"completed: {count} pairs\n")
    
    print(f"Job {args.job_id}: Analysis complete. Results in {output_file}")

if __name__ == '__main__':
    main()
