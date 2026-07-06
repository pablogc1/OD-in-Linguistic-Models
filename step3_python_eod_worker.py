#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EOD Calculation Worker (Step 3 - Corrected)

This version fixes the CSV writing bug by using pandas to generate the output,
which correctly handles special characters in words (e.g., commas), thus
preventing downstream ParserErrors.
"""

import os
import argparse
import itertools
from collections import Counter
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd  # <-- Import pandas

DEFINITIONS_FILES = {
    "ground_filtered": "curated_corpora/curated_ground_filtered.txt",
    "null_model": "curated_corpora/curated_null_model.txt",
    "random_removal": "curated_corpora/curated_random_removal.txt",
    "targeted_removal": "curated_corpora/curated_targeted_removal.txt",
    "ai_generated": "curated_corpora/curated_ai_generated.txt",
    "merriam_webster": "curated_corpora/curated_merriam_webster.txt"
}
COMMON_VOCAB_FILE = "curated_corpora/common_vocabulary.txt"
OUTPUT_DIR = "eod_results"

# --- Globals for worker processes ---
worker_sets_dicts = None
worker_word_map = None

# ===============================================
#   EOD LOGIC (This part remains unchanged)
# ===============================================

def process_GOD(words_to_expand, sets_dict_A, sets_dict_B, max_level=100):
    all_elements_seen = set(words_to_expand)
    elements_to_open_now = set(words_to_expand)
    for level in range(1, max_level + 1):
        elements_for_next_level = set()
        for word in elements_to_open_now:
            elements_for_next_level.update(sets_dict_A.get(word, []))
            elements_for_next_level.update(sets_dict_B.get(word, []))
        new_elements = elements_for_next_level - all_elements_seen
        if not new_elements: return level
        all_elements_seen.update(new_elements)
        elements_to_open_now = new_elements
    return max_level

def run_EOD_engine(seed_word, sets_dict_A, sets_dict_B, omega_god, key_A="Side 1", key_B="Side 2", verbose=False):
    E, U, R = {}, {}, {}
    log_steps = []
    
    E[(0, 1)], E[(0, 2)] = Counter([seed_word]), Counter([seed_word])
    global_E_side1, global_E_side2 = E[(0, 1)].copy(), E[(0, 2)].copy()

    for side in [1, 2]:
        U[(0, side)], R[(0, side)] = E[(0, side)].copy(), Counter()
    
    for level in range(1, omega_god + 2):
        if verbose: log_steps.append(f"\n--- Level {level} ---")
        
        E[(level, 1)] = Counter(e for elem in E.get((level - 1, 1), {}) for e in sets_dict_A.get(elem, []))
        E[(level, 2)] = Counter(e for elem in E.get((level - 1, 2), {}) for e in sets_dict_B.get(elem, []))
        if verbose:
            log_steps.append(f"  Expansion ({key_A}): {dict(E[(level, 1)])}")
            log_steps.append(f"  Expansion ({key_B}): {dict(E[(level, 2)])}")
            
        U[(level, 1)], R[(level, 1)] = E[(level, 1)].copy(), Counter()
        U[(level, 2)], R[(level, 2)] = E[(level, 2)].copy(), Counter()

        global_E_side1 += E.get((level, 1), Counter())
        global_E_side2 += E.get((level, 2), Counter())

        for m in range(level + 1):
            for side in [1, 2]:
                words_to_cancel = []
                check_set = global_E_side2 if side == 1 else global_E_side1
                for u_word in list(U.get((m, side), {})):
                    if check_set[u_word] > 0:
                        words_to_cancel.append(u_word)
                if words_to_cancel and verbose:
                    log_steps.append(f"  Cancellation: At level m={m} on side {side}, words {words_to_cancel} are canceled.")
                for u_word in words_to_cancel:
                    count = U[(m, side)].pop(u_word)
                    R[(m, side)][u_word] += count
        
        termination_triggered = False
        for m_term, s_term in U.keys():
            if m_term > 0 and not U[(m_term, s_term)]:
                termination_triggered = True
                break

        if termination_triggered:
            if verbose: log_steps.append("\nTERMINATION: An uncanceled set U (at level > 0) became empty.")
            
            # V6: Calculate EOD score based on UNCANCELLATIONS (U)
            score = sum(
                count * lvl 
                for (lvl, side), u_set in U.items() 
                for word, count in u_set.items()
            )
            
            if verbose:
                log_steps.append("\n--- Final Score Calculation (Uncancelled Elements) ---")
                for (lvl, side), u_set in sorted(U.items()):
                    if not u_set: continue
                    for word, count in u_set.items():
                        log_steps.append(f"  Level {lvl} ({key_A if side == 1 else key_B}): {count} uncancelled '{word}' -> +{count * lvl} to score")
                log_steps.append("---------------------------------")
                log_steps.append(f"  Total EOD Score = {score}")
                return score, level, "\n".join(log_steps)
            return score, level
            
        if level > omega_god:
            if verbose: return -1, level, "\n".join(log_steps) + "\nTERMINATION: GOD rule."
            return -1, level
            
    final_level = omega_god + 1
    return -1, final_level

# ===============================================
#       UTILITY AND I/O FUNCTIONS
# ===============================================

def read_definitions(file_path):
    definitions = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                head, def_text = line.split(":", 1)
                definitions[head.strip().lower()] = def_text.strip().split()
    return definitions

def run_canary_test(sets_dicts):
    print("--- Running Comprehensive EOD Canary Test for 'money' ---")
    word_to_test = "money"
    log_filename = "canary_test_eod_log.txt"
    corpus_combinations = list(itertools.combinations(DEFINITIONS_FILES.keys(), 2))

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"Comprehensive EOD Canary Test for word: '{word_to_test}'\n")
        
        for key_A, key_B in corpus_combinations:
            f.write("\n\n" + "="*80 + "\n")
            f.write(f"   EOD TEST: {key_A} vs {key_B}\n")
            f.write("="*80 + "\n\n")

            dict_A, dict_B = sets_dicts[key_A], sets_dicts[key_B]
            if word_to_test not in dict_A or word_to_test not in dict_B:
                f.write(f"SKIPPED: '{word_to_test}' not present in both corpora definitions.\n")
                continue

            omega_god = process_GOD([word_to_test], dict_A, dict_B)
            f.write(f"--- GOD Calculation ---\nResult: ω_GOD = {omega_god}\n\n")

            f.write("--- EOD Calculation (verbose) ---\n")
            eod_score, term_level, eod_log = run_EOD_engine(word_to_test, dict_A, dict_B, omega_god, key_A=key_A, key_B=key_B, verbose=True)
            f.write(eod_log)
            f.write(f"\n\n--- SUMMARY for {key_A} vs {key_B} ---\n")
            f.write(f"Final EOD Score: {eod_score}\n")
            f.write(f"Terminated at level: {term_level}\n")
            f.write("="*80)

    print(f"--- Canary test complete. Detailed log saved to '{log_filename}' ---")

def init_worker(sets_dicts, word_map):
    global worker_sets_dicts, worker_word_map
    worker_sets_dicts, worker_word_map = sets_dicts, word_map

# =================================================================
#               *** FIX IS IN THIS FUNCTION ***
# =================================================================
def worker_process_eod(word):
    """
    FIX: This function now returns a dictionary, which is safer for
    data with special characters. Pandas will handle converting this
    dictionary to a correctly formatted CSV row.
    """
    word_idx = worker_word_map[word]
    result_dict = {"master_idx": word_idx, "word": word}
    corpus_combinations = list(itertools.combinations(DEFINITIONS_FILES.keys(), 2))

    for key_A, key_B in corpus_combinations:
        dict_A = worker_sets_dicts[key_A]
        dict_B = worker_sets_dicts[key_B]
        
        # EOD can only run if the word exists in both definition sets for that pair
        if word not in dict_A or word not in dict_B:
            eod_score, term_level = -2, -2 # Use a special code for "word not found"
        else:
            omega_god = process_GOD([word], dict_A, dict_B)
            eod_score, term_level = run_EOD_engine(word, dict_A, dict_B, omega_god)
        
        result_dict[f"eod_score_{key_A}_vs_{key_B}"] = eod_score
        result_dict[f"eod_tl_{key_A}_vs_{key_B}"] = term_level
        
    return result_dict

# =================================================================
#               *** FIX IS IN THIS FUNCTION ***
# =================================================================
def main():
    parser = argparse.ArgumentParser(description="Run Eigen Ontological Differentiation analysis.")
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--total_jobs", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()
    
    print("Loading all definition files...")
    all_sets_dicts = {key: read_definitions(fp) for key, fp in DEFINITIONS_FILES.items()}
    
    with open(COMMON_VOCAB_FILE, "r") as f:
        common_words = [line.strip() for line in f]
    master_word_to_idx = {word: i for i, word in enumerate(common_words)}
    
    if args.job_id == 1:
        run_canary_test(all_sets_dicts)

    chunk_size = (len(common_words) + args.total_jobs - 1) // args.total_jobs
    start = (args.job_id - 1) * chunk_size
    end = min(len(common_words), start + chunk_size)
    words_for_this_job = common_words[start:end]

    output_file = os.path.join(OUTPUT_DIR, f"eod_results_part_{args.job_id}.csv")
    print(f"Job {args.job_id}: Processing {len(words_for_this_job)} words using {args.num_workers} workers...")
    
    # --- FIX: Use Pandas to write the CSV correctly ---
    results_list = []
    init_args = (all_sets_dicts, master_word_to_idx)
    with Pool(processes=args.num_workers, initializer=init_worker, initargs=init_args) as pool:
        results_iterator = pool.imap_unordered(worker_process_eod, words_for_this_job)
        for result in tqdm(results_iterator, total=len(words_for_this_job), desc=f"Job {args.job_id}"):
            if result:
                results_list.append(result)

    if results_list:
        df_results = pd.DataFrame(results_list)
        
        # Define header columns to ensure a consistent order
        header_cols = ["master_idx", "word"]
        for key_A, key_B in itertools.combinations(DEFINITIONS_FILES.keys(), 2):
            header_cols.append(f"eod_score_{key_A}_vs_{key_B}")
            header_cols.append(f"eod_tl_{key_A}_vs_{key_B}")
        
        # Reorder DataFrame columns and save to CSV
        df_results = df_results[header_cols]
        df_results.to_csv(output_file, index=False)
    
    # V2 FIX: Write sentinel file ONLY after output file is fully written
    sentinel_file = os.path.join(OUTPUT_DIR, f"eod_job_{args.job_id}.done")
    with open(sentinel_file, "w") as f:
        f.write(f"completed: {len(results_list)} words\n")

    print(f"Job {args.job_id}: Analysis complete. Results saved to {output_file}")

if __name__ == '__main__':
    main()

