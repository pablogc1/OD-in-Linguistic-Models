#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ontological Differentiation (OD) Large-Scale Analysis Script (V6)

This script performs OD analyses, designed for parallel execution on a
SLURM cluster and leveraging multiprocessing within each job.

VERSION 6 CHANGES:
- Updated scoring to count UNCANCELLATIONS instead of cancellations
  (uncancelled elements = unique/differentiated elements = more intuitive)
- Removed WOD variant (SOD only)
- Enhanced canary test with basic validation

Modes:
1.  '--mode pairs': Calculates SOD, termination level, and pairwise GOD 
    for all unique pairs in the vocabulary slice. This single run produces 
    all data needed for both 'all-vs-all' and 'one-vs-all' post-analysis. 
    It also triggers a one-off canary test.
2.  '--mode single_god': Calculates the GOD for each individual word.
"""

import os
import sys
import argparse
from collections import Counter
import itertools
from tqdm import tqdm
from multiprocessing import Pool

# ===============================================
#   CORE ONTOLOGICAL DIFFERENTIATION LOGIC
# ===============================================

def process_GOD(words_to_expand, sets_dict, max_level=100):
    """
    Calculate Global Ontological Differentiation (GOD).
    Returns the level at which all definitions converge.
    """
    all_elements_seen = set(words_to_expand)
    elements_to_open_now = set(words_to_expand)
    for level in range(1, max_level + 1):
        elements_for_next_level = set()
        for word in elements_to_open_now:
            elements_for_next_level.update(sets_dict.get(word, []))
        new_elements = elements_for_next_level - all_elements_seen
        if not new_elements:
            return level
        all_elements_seen.update(new_elements)
        elements_to_open_now = new_elements
    return max_level


def process_SOD(seed_a, seed_b, sets_dict, omega_god, verbose=False):
    """
    Calculate Strong Ontological Differentiation (SOD).
    
    V6 UPDATE: Score is now based on UNCANCELLATIONS.
    - At each level, we count how many elements are NOT cancelled
    - Uncancelled elements represent unique/differentiated semantic content
    - Score = Σ(uncancelled_count × level) for all levels
    
    Returns: (score, termination_level) or (score, termination_level, log) if verbose
    """
    E, U, R = {}, {}, {}
    log_steps = []
    
    # Initialize level 0
    E[(0, 1)], E[(0, 2)] = Counter([seed_a]), Counter([seed_b])
    
    # Global counters for tracking all elements seen
    global_E_side1 = E[(0, 1)].copy()
    global_E_side2 = E[(0, 2)].copy()

    for side in [1, 2]:
        U[(0, side)], R[(0, side)] = E[(0, side)].copy(), Counter()
    
    if verbose:
        log_steps.append(f"Level 0:")
        log_steps.append(f"  U_1: {dict(U[(0,1)])}, R_1: {dict(R[(0,1)])}")
        log_steps.append(f"  U_2: {dict(U[(0,2)])}, R_2: {dict(R[(0,2)])}")

    for level in range(1, omega_god + 2):
        # Expand definitions for this level
        for side in [1, 2]:
            E[(level, side)] = Counter()
            for elem, count in E.get((level - 1, side), {}).items():
                for e in sets_dict.get(elem, []):
                    E[(level, side)][e] += count
            U[(level, side)], R[(level, side)] = E[(level, side)].copy(), Counter()

        # Update global counters
        global_E_side1 += E.get((level, 1), Counter())
        global_E_side2 += E.get((level, 2), Counter())

        # Cancellation logic (SOD: cancel if element appears in opposite side globally)
        for m in range(level + 1):
            for side in [1, 2]:
                check_set = global_E_side2 if side == 1 else global_E_side1
                words_to_cancel = []
                for u_word in U.get((m, side), {}):
                    if check_set[u_word] > 0:
                        words_to_cancel.append(u_word)
                
                # Move cancelled words from U to R
                for u_word in words_to_cancel:
                    count = U[(m, side)].pop(u_word)
                    R[(m, side)][u_word] += count
        
        if verbose:
            log_steps.append(f"\nLevel {level}:")
            for m in range(level + 1):
                u1_str = str(dict(U.get((m,1),{})))
                r1_str = str(dict(R.get((m,1),{})))
                u2_str = str(dict(U.get((m,2),{})))
                r2_str = str(dict(R.get((m,2),{})))
                log_steps.append(f"  (m={m}) U_1: {u1_str}, R_1: {r1_str}")
                log_steps.append(f"  (m={m}) U_2: {u2_str}, R_2: {r2_str}")
            log_steps.append("-" * 40)

        # Check termination: any U set becomes empty
        if any(len(U.get(key, {})) == 0 for key in U.keys()):
            # V6: Calculate score based on UNCANCELLATIONS (U)
            score = sum(
                count * lvl 
                for (lvl, side), u_set in U.items() 
                for word, count in u_set.items()
            )
            
            if verbose:
                # Also show cancellation score for comparison
                score_R = sum(
                    count * lvl 
                    for (lvl, side), r_set in R.items() 
                    for word, count in r_set.items()
                )
                log_steps.append(f"\nTERMINATED at level {level}")
                log_steps.append(f"Final SOD Score (UNCANCELLATIONS): {score}")
                log_steps.append(f"  (Old cancellation score would be: {score_R})")
                return score, level, "\n".join(log_steps)
            return score, level
        
        # GOD rule: if we exceed omega_god, mark as invalid
        if level > omega_god:
            if verbose:
                log_steps.append(f"\nTERMINATION: Invalid run (exceeded GOD level {omega_god})")
                return -1, level, "\n".join(log_steps)
            return -1, level
    
    # Fallback (should rarely happen)
    return -1, omega_god + 1


# ===============================================
#   WORKER FUNCTIONS FOR MULTIPROCESSING
# ===============================================
worker_sets_dict = None
worker_word_map = None

def init_worker(sets_dict, word_map):
    global worker_sets_dict, worker_word_map
    worker_sets_dict = sets_dict
    worker_word_map = word_map

def worker_process_pair(pair):
    word_a, word_b = pair
    omega_god = process_GOD([word_a, word_b], worker_sets_dict)
    
    sod_score, sod_term_level = process_SOD(word_a, word_b, worker_sets_dict, omega_god)
    
    idx1 = worker_word_map[word_a]
    idx2 = worker_word_map[word_b]
    
    # Output format: idx1 idx2 sod_score sod_term_level omega_god
    return f"{idx1} {idx2} {sod_score} {sod_term_level} {omega_god}\n"

def worker_process_single_god(word):
    god_score = process_GOD([word], worker_sets_dict)
    idx = worker_word_map[word]
    return f"{idx} {god_score}\n"
    
# ===============================================
#   UTILITY AND I/O FUNCTIONS
# ===============================================

def read_definitions(file_path):
    definitions, ordered_words = {}, []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            head, def_text = line.split(":", 1)
            head = head.strip().lower()
            tokens = def_text.strip().split()
            if head not in definitions:
                ordered_words.append(head)
            definitions[head] = tokens
    return definitions, ordered_words

def run_canary_test(sets_dict):
    """
    Runs a canary test with verbose output and basic validation.
    Tests 'money' vs 'business' as a representative pair.
    
    Validation checks:
    1. SOD terminates successfully (score != -1)
    2. Termination level is reasonable (> 0)
    
    Returns True if canary passes, False otherwise.
    """
    print("\n" + "="*60)
    print("           CANARY TEST: 'money' vs 'business'")
    print("="*60)
    
    word_a, word_b = "money", "business"
    log_filename = "canary_test_od_log.txt"
    
    # Check words exist
    if word_a not in sets_dict or word_b not in sets_dict:
        print(f"CANARY FAILED: Test words not in vocabulary!")
        print(f"  '{word_a}' exists: {word_a in sets_dict}")
        print(f"  '{word_b}' exists: {word_b in sets_dict}")
        return False

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"CANARY TEST: {word_a} vs {word_b}\n")
        f.write("="*60 + "\n\n")
        f.write("V6: Scoring based on UNCANCELLATIONS\n")
        f.write("(Higher score = more differentiated/unique semantic content)\n\n")

        f.write("--- GOD Calculation ---\n")
        omega_god = process_GOD([word_a, word_b], sets_dict)
        f.write(f"Result: ω_GOD = {omega_god}\n\n")

        f.write("--- SOD Calculation (verbose) ---\n")
        sod_score, sod_term_level, sod_log = process_SOD(
            word_a, word_b, sets_dict, omega_god, verbose=True
        )
        f.write(sod_log)
        f.write(f"\n\nFinal SOD Score: {sod_score}")
        f.write(f"\nTerminated at level: {sod_term_level}\n")

    # Validation
    print(f"\nResults:")
    print(f"  GOD (ω):           {omega_god}")
    print(f"  SOD Score:         {sod_score}")
    print(f"  Termination Level: {sod_term_level}")
    print(f"\nDetailed log saved to: {log_filename}")
    
    # Basic validation
    if sod_score == -1:
        print("\n[CANARY FAILED] SOD did not terminate correctly!")
        return False
    
    if sod_term_level <= 0:
        print("\n[CANARY FAILED] Invalid termination level!")
        return False
    
    print("\n[CANARY PASSED] SOD terminated correctly.")
    print("="*60 + "\n")
    return True


# ===============================================
#   MAIN EXECUTION BLOCK
# ===============================================

def main():
    parser = argparse.ArgumentParser(
        description="Run large-scale Ontological Differentiation analysis (V6 - Uncancellation Scoring)."
    )
    parser.add_argument("--job_id", type=int, required=True, 
                        help="Current job index (1-based).")
    parser.add_argument("--total_jobs", type=int, required=True, 
                        help="Total number of parallel jobs.")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=['pairs', 'single_god'], 
                        help="The analysis mode to run.")
    parser.add_argument("--num_workers", type=int, default=1, 
                        help="Number of CPU cores to use within this job.")
    parser.add_argument("--input_file", type=str, default="extracted_definitions.txt",
                        help="Input definitions file.")
    parser.add_argument("--start_idx", type=int, default=1,
                        help="Start index for vocabulary slice (1-based).")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index for vocabulary slice (inclusive). Default: all words.")
    args = parser.parse_args()

    # --- Data Loading ---
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    print(f"Loading definitions from '{args.input_file}'...")
    definitions, ordered_words = read_definitions(args.input_file)
    word_to_index_map = {word: i + 1 for i, word in enumerate(ordered_words)}
    print(f"Loaded {len(definitions)} definitions.")
    
    # Determine vocabulary slice
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx else len(ordered_words)
    analysis_vocab = ordered_words[start_idx - 1 : end_idx]
    print(f"Analysis vocabulary: {len(analysis_vocab)} words (indices {start_idx} to {end_idx}).")

    # --- Canary Test (job 1 only, pairs mode) ---
    if args.job_id == 1 and args.mode == 'pairs':
        if not run_canary_test(definitions):
            print("Aborting due to canary test failure.")
            sys.exit(1)

    # --- Mode Dispatch ---
    if args.mode == 'pairs':
        output_file = f"pairs_results_job_{args.job_id}.txt"
        
        all_pairs_iter = itertools.combinations(analysis_vocab, 2)
        total_pairs = len(analysis_vocab) * (len(analysis_vocab) - 1) // 2
        
        chunk_size = (total_pairs + args.total_jobs - 1) // args.total_jobs
        start = (args.job_id - 1) * chunk_size
        end = min(total_pairs, start + chunk_size)
        
        print(f"Skipping to pair {start:,}...")
        items_for_this_job = list(itertools.islice(all_pairs_iter, start, end))
        
        worker_func = worker_process_pair
        print(f"\nJob {args.job_id}/{args.total_jobs}: Processing {len(items_for_this_job)} pairs")
        print(f"  (Pairs {start+1} to {end} of {total_pairs} total)")

    elif args.mode == 'single_god':
        output_file = f"single_god_results_job_{args.job_id}.txt"
        total_words = len(analysis_vocab)
        chunk_size = (total_words + args.total_jobs - 1) // args.total_jobs
        start = (args.job_id - 1) * chunk_size
        end = min(total_words, start + chunk_size)
        items_for_this_job = analysis_vocab[start:end]
        worker_func = worker_process_single_god
        print(f"\nJob {args.job_id}/{args.total_jobs}: Processing {len(items_for_this_job)} words")
        print(f"  (Words {start+1} to {end} of {total_words} total)")
    
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        sys.exit(1)

    # Sentinel file to signal completion (for watch_stage.py)
    sentinel_file = f"{output_file}.done"
    
    if len(items_for_this_job) == 0:
        print(f"Job {args.job_id}: No items to process. Creating empty output file.")
        open(output_file, "w").close()
        # Write sentinel file to signal completion
        with open(sentinel_file, "w") as f:
            f.write(f"completed: 0 items\n")
        return

    # --- Multiprocessing Execution ---
    # MEMORY FIX: Write incrementally to temp file, then rename.
    # Uses temp file to avoid watch_stage.py seeing partial results.
    temp_file = f".{output_file}.tmp"
    print(f"Starting processing with {args.num_workers} workers...")
    
    count = 0
    with Pool(processes=args.num_workers, 
              initializer=init_worker, 
              initargs=(definitions, word_to_index_map),
              maxtasksperchild=1000) as pool:
        with open(temp_file, "w") as f:
            for result in tqdm(
                pool.imap_unordered(worker_func, items_for_this_job), 
                total=len(items_for_this_job), 
                desc=f"Job {args.job_id} ({args.mode})"
            ):
                f.write(result)
                count += 1
    
    # Atomic rename: temp file -> final file
    os.rename(temp_file, output_file)
    
    # Write sentinel file ONLY after output file is fully written
    with open(sentinel_file, "w") as f:
        f.write(f"completed: {count} items\n")
    
    print(f"\nJob {args.job_id}: Complete. {count} results saved to '{output_file}'")


if __name__ == '__main__':
    main()
