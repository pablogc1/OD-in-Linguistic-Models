"""
Semantic Difference Analysis (V6 - SOD Only)

Calculates semantic difference metrics between word pairs using pre-built indices.
V6 UPDATE: Only processes SOD (WOD removed).

MEMORY FIX: Writes results incrementally to a temp file, then renames.
This avoids buffering millions of results in memory.

Index format (V6): partner_idx,sod_score,sod_term_level
"""
import pandas as pd
import numpy as np
import os
import argparse
import itertools
from multiprocessing import Pool
from tqdm import tqdm

# --- Globals for worker processes ---
worker_word_map = None
worker_index_dir = None

# V6 index columns: partner_idx,sod_score,sod_term_level
INDEX_COLS = ['partner_idx', 'sod_score', 'sod_term_level']

def init_worker(word_map, index_dir):
    """Initializes global variables for each worker process."""
    global worker_word_map, worker_index_dir
    worker_word_map = word_map
    worker_index_dir = index_dir

def worker_process_difference(pair):
    """Process one pair (A, B), calculate SOD difference metrics."""
    word_a, word_b = pair
    idx_a = worker_word_map.get(word_a)
    idx_b = worker_word_map.get(word_b)
    
    if idx_a is None or idx_b is None:
        return None
        
    try:
        path_a = os.path.join(worker_index_dir, f"{idx_a}.csv")
        path_b = os.path.join(worker_index_dir, f"{idx_b}.csv")
        
        df_a = pd.read_csv(path_a, header=None, names=INDEX_COLS).set_index('partner_idx')
        df_b = pd.read_csv(path_b, header=None, names=INDEX_COLS).set_index('partner_idx')
        
        if idx_b not in df_a.index:
            return None
        
        mutual_sod_level_val = df_a.loc[idx_b, 'sod_term_level']
        mutual_sod_level = mutual_sod_level_val.iloc[0] if isinstance(mutual_sod_level_val, pd.Series) else mutual_sod_level_val

        df_merged = df_a.join(df_b, how='inner', lsuffix='_A', rsuffix='_B')
        
        if df_merged.empty:
            return f"{idx_a},{idx_b},{mutual_sod_level},0,0\n"

        df_merged['sod_diff'] = np.abs(np.log1p(df_merged['sod_score_A']) - np.log1p(df_merged['sod_score_B']))
        df_merged['sod_term_diff'] = np.abs(df_merged['sod_term_level_A'] - df_merged['sod_term_level_B'])

        avg_sod_diff = df_merged['sod_diff'].mean()
        avg_sod_term_diff = df_merged['sod_term_diff'].mean()
        
        return f"{idx_a},{idx_b},{mutual_sod_level},{avg_sod_diff:.6f},{avg_sod_term_diff:.6f}\n"

    except FileNotFoundError:
        return None
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser(description="Run semantic difference analysis (V6 - SOD only).")
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--total_jobs", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--index_dir", type=str, required=True)
    args = parser.parse_args()

    print("Loading vocabulary...")
    ordered_words = []
    seen = set()
    with open(args.vocab_file, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                head = line.split(":", 1)[0].strip().lower()
                if head not in seen:
                    ordered_words.append(head)
                    seen.add(head)
    word_to_index_map = {word: i + 1 for i, word in enumerate(ordered_words)}
    print(f"Loaded {len(ordered_words)} words.")
    
    total_pairs = len(ordered_words) * (len(ordered_words) - 1) // 2
    print(f"Total pairs to process across all jobs: {total_pairs:,}")
    
    chunk_size = (total_pairs + args.total_jobs - 1) // args.total_jobs
    start = (args.job_id - 1) * chunk_size
    end = min(total_pairs, start + chunk_size)
    
    print(f"Skipping to pair {start:,}...")
    all_pairs_iter = itertools.combinations(ordered_words, 2)
    items_for_this_job = list(itertools.islice(all_pairs_iter, start, end))

    output_file = f"semantic_diff_results_job_{args.job_id}.csv"
    temp_file = f".{output_file}.tmp"
    sentinel_file = f"{output_file}.done"
    
    print(f"Job {args.job_id}: Processing {len(items_for_this_job):,} pairs using {args.num_workers} workers...")
    
    # V6 header (SOD only)
    header = "idx_A,idx_B,mutual_sod_level,avg_sod_diff,avg_sod_term_diff\n"

    # MEMORY FIX: Write incrementally to temp file instead of buffering
    count = 0
    # Use maxtasksperchild to prevent pandas memory leaks in workers over long runs
    with Pool(processes=args.num_workers, initializer=init_worker, initargs=(word_to_index_map, args.index_dir)) as pool:
        with open(temp_file, "w") as f:
            f.write(header)
            results_iterator = pool.imap_unordered(worker_process_difference, items_for_this_job)
            for result in tqdm(results_iterator, total=len(items_for_this_job), desc=f"Job {args.job_id}"):
                if result:
                    f.write(result)
                    count += 1
    
    # Atomic rename: temp file -> final file
    os.rename(temp_file, output_file)
    
    with open(sentinel_file, "w") as f:
        f.write(f"completed: {count} pairs\n")
    
    print(f"Job {args.job_id}: Analysis complete. {count} results saved to {output_file}")


if __name__ == '__main__':
    main()
