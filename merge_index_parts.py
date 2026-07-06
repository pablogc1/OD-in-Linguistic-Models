#!/usr/bin/env python3
"""
Fast Index Merger (Parallel Processing)

Merges per-job index files into final per-word index files.
Uses parallel processing for ~10-20x speedup over bash loop.

Usage:
    python3 merge_index_parts.py --temp_dir temp_index_parts_CORPUS --output_dir results_CORPUS/indexed_pairs_data
"""
import os
import argparse
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil


def find_all_word_indices(temp_dir):
    """Find all unique word indices across all job directories."""
    word_indices = set()
    job_dirs = [d for d in os.listdir(temp_dir) if d.startswith('job_') and os.path.isdir(os.path.join(temp_dir, d))]
    
    print(f"  Scanning {len(job_dirs)} job directories...")
    for job_dir in job_dirs:
        job_path = os.path.join(temp_dir, job_dir)
        for filename in os.listdir(job_path):
            if filename.endswith('.csv'):
                word_idx = filename[:-4]  # Remove .csv
                word_indices.add(word_idx)
    
    return sorted(word_indices, key=lambda x: int(x) if x.isdigit() else x)


def merge_single_word(args):
    """Merge all job parts for a single word index."""
    word_idx, temp_dir, output_dir, job_dirs = args
    
    output_file = os.path.join(output_dir, f"{word_idx}.csv")
    lines = []
    
    # Collect data from all job directories
    for job_dir in job_dirs:
        source_file = os.path.join(temp_dir, job_dir, f"{word_idx}.csv")
        if os.path.exists(source_file):
            with open(source_file, 'r') as f:
                lines.extend(f.readlines())
    
    # Write merged file
    if lines:
        with open(output_file, 'w') as f:
            f.writelines(lines)
        return len(lines)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Merge index parts in parallel.")
    parser.add_argument("--temp_dir", required=True, help="Temp directory with job_* subdirs")
    parser.add_argument("--output_dir", required=True, help="Output directory for merged indices")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--cleanup", action="store_true", help="Delete temp_dir after successful merge")
    args = parser.parse_args()
    
    if not os.path.exists(args.temp_dir):
        print(f"ERROR: Temp directory not found: {args.temp_dir}")
        return 1
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all job directories
    job_dirs = [d for d in os.listdir(args.temp_dir) 
                if d.startswith('job_') and os.path.isdir(os.path.join(args.temp_dir, d))]
    
    if not job_dirs:
        print(f"ERROR: No job_* directories found in {args.temp_dir}")
        return 1
    
    print(f"  Found {len(job_dirs)} job directories")
    
    # Find all unique word indices
    word_indices = find_all_word_indices(args.temp_dir)
    print(f"  Found {len(word_indices):,} unique word indices to merge")
    
    # Prepare arguments for parallel processing
    merge_args = [(idx, args.temp_dir, args.output_dir, job_dirs) for idx in word_indices]
    
    # Determine number of workers
    num_workers = args.workers or min(cpu_count(), 16)
    print(f"  Merging with {num_workers} parallel workers...")
    
    # Parallel merge
    total_lines = 0
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(merge_single_word, merge_args),
            total=len(word_indices),
            desc="  Merging"
        ))
        total_lines = sum(results)
    
    print(f"  ✓ Merged {total_lines:,} total index entries into {len(word_indices):,} files")

    sentinel_file = os.path.join(args.output_dir, ".merge_complete.done")
    with open(sentinel_file, "w") as f:
        f.write(f"completed: {len(word_indices)} words, {total_lines} entries\n")

    # Cleanup if requested
    if args.cleanup:
        print(f"  Cleaning up {args.temp_dir}...")
        shutil.rmtree(args.temp_dir)
        print(f"  ✓ Deleted temp directory")
    
    return 0


if __name__ == "__main__":
    exit(main())
