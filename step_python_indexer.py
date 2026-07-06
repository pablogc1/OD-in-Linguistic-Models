"""
Index Builder for OD Results (V6 - SOD Only)

Creates per-word index files from pairs_results.txt for fast lookup.
V6 UPDATE: Only processes SOD (WOD removed).

MEMORY FIX: Uses simple line-by-line processing (like step1_synchronize_indices.py)
instead of pd.read_csv() which loads entire file into memory.

Input format (V6): idx1 idx2 sod_score sod_term_level omega_god
Output format: partner_idx,sod_score,sod_term_level (per word index file)
"""
import os
import argparse
from collections import defaultdict
from tqdm import tqdm

# How many entries to buffer before flushing to disk
BUFFER_FLUSH_SIZE = 5_000_000

def main():
    parser = argparse.ArgumentParser(description="Create per-word index from pairs results.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--corpus_key", type=str, required=True)
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--total_jobs", type=int, default=40)
    args = parser.parse_args()

    # Each job writes to its own directory
    temp_output_dir = f"temp_index_parts_{args.corpus_key}/job_{args.job_id}"
    os.makedirs(temp_output_dir, exist_ok=True)
    
    if not os.path.exists(args.input_file):
        print(f"INFO: Job {args.job_id} - Input file '{args.input_file}' not found.")
        write_sentinel(args.corpus_key, args.job_id, 0, "file not found")
        return

    # Count total lines for work distribution
    print(f"Job {args.job_id}: Counting lines in {args.input_file}...")
    total_lines = sum(1 for _ in open(args.input_file, 'r'))
    print(f"Job {args.job_id}: Found {total_lines:,} total pairs")
    
    # Calculate which lines this job should process
    lines_per_job = (total_lines + args.total_jobs - 1) // args.total_jobs
    start_line = (args.job_id - 1) * lines_per_job
    end_line = min(total_lines, start_line + lines_per_job)
    lines_to_process = end_line - start_line
    
    print(f"Job {args.job_id}: Processing lines {start_line:,} to {end_line:,} ({lines_to_process:,} lines)")
    
    # Process file line-by-line (memory efficient, like step1_synchronize_indices.py)
    data_buffer = defaultdict(list)
    buffer_size = 0
    total_processed = 0
    skipped_invalid = 0
    
    with open(args.input_file, 'r') as f:
        # Skip to our starting line
        for _ in range(start_line):
            next(f, None)
        
        # Process only our assigned lines
        pbar = tqdm(total=lines_to_process, desc=f"Job {args.job_id} ({args.corpus_key})")
        
        for i, line in enumerate(f):
            if i >= lines_to_process:
                break
                
            parts = line.strip().split()
            if len(parts) < 5:
                skipped_invalid += 1
                pbar.update(1)
                continue
            
            try:
                idx1 = int(parts[0])
                idx2 = int(parts[1])
                sod_score = int(parts[2])
                sod_tl = int(parts[3])
                # parts[4] is omega_god, we don't need it for the index
                
                # Skip invalid results (sod == -1 means computation failed)
                if sod_score == -1:
                    skipped_invalid += 1
                    pbar.update(1)
                    continue
                
                # V6 index format: partner_idx,sod_score,sod_term_level
                line_content = f"{sod_score},{sod_tl}\n"
                data_buffer[idx1].append(f"{idx2},{line_content}")
                data_buffer[idx2].append(f"{idx1},{line_content}")
                buffer_size += 2
                total_processed += 1
                
            except (ValueError, IndexError) as e:
                skipped_invalid += 1
            
            pbar.update(1)
            
            # Flush buffer periodically to avoid memory buildup
            if buffer_size >= BUFFER_FLUSH_SIZE:
                print(f"\nJob {args.job_id}: Flushing {buffer_size:,} entries to disk...")
                flush_buffer(data_buffer, temp_output_dir)
                data_buffer.clear()
                buffer_size = 0
        
        pbar.close()
    
    # Final flush
    if data_buffer:
        print(f"Job {args.job_id}: Final flush of {buffer_size:,} entries...")
        flush_buffer(data_buffer, temp_output_dir)
    
    # Write sentinel file
    write_sentinel(args.corpus_key, args.job_id, total_processed, "success")
    
    print(f"Job {args.job_id}: Finished. Processed {total_processed:,} pairs, skipped {skipped_invalid:,} invalid.")


def flush_buffer(data_buffer, output_dir):
    """Write buffered data to per-word index files."""
    for idx, lines in data_buffer.items():
        filepath = os.path.join(output_dir, f"{idx}.csv")
        with open(filepath, "a") as f:
            f.writelines(lines)


def write_sentinel(corpus_key, job_id, count, status):
    """Write sentinel file to signal job completion."""
    sentinel_dir = f"temp_index_parts_{corpus_key}"
    os.makedirs(sentinel_dir, exist_ok=True)
    sentinel_file = os.path.join(sentinel_dir, f"job_{job_id}.done")
    with open(sentinel_file, "w") as f:
        f.write(f"completed: {count} pairs ({status})\n")


if __name__ == "__main__":
    main()
