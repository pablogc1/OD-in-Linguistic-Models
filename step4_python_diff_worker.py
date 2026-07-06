#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pairwise Difference Worker (V6 - SOD Only)

Calculates inter-corpus SOD differences for all common-vocabulary words.
V6 UPDATE: Only processes SOD (WOD removed).

CRITICAL FIX: Each corpus has its own 1-based word indices. This script
builds per-corpus word-to-index mappings and remaps partner indices to
common word names before joining, ensuring correct cross-corpus comparison.
"""
import os
import argparse
import itertools
import pandas as pd
from tqdm import tqdm

COMMON_VOCAB_FILE = "curated_corpora/common_vocabulary.txt"
OUTPUT_DIR = "pairwise_diff_results"

CORPORA_CONFIG = {
    "ground_filtered": {
        "corpus_file": "curated_corpora/curated_ground_filtered.txt",
        "index_dir": "results_curated_ground_filtered/indexed_pairs_data"
    },
    "null_model": {
        "corpus_file": "curated_corpora/curated_null_model.txt",
        "index_dir": "results_curated_null_model/indexed_pairs_data"
    },
    "random_removal": {
        "corpus_file": "curated_corpora/curated_random_removal.txt",
        "index_dir": "results_curated_random_removal/indexed_pairs_data"
    },
    "targeted_removal": {
        "corpus_file": "curated_corpora/curated_targeted_removal.txt",
        "index_dir": "results_curated_targeted_removal/indexed_pairs_data"
    },
    "ai_generated": {
        "corpus_file": "curated_corpora/curated_ai_generated.txt",
        "index_dir": "results_curated_ai_generated/indexed_pairs_data"
    },
    "merriam_webster": {
        "corpus_file": "curated_corpora/curated_merriam_webster.txt",
        "index_dir": "results_curated_merriam_webster/indexed_pairs_data"
    },
}

CORPORA_KEYS = list(CORPORA_CONFIG.keys())


def build_corpus_mappings():
    """Build word<->index mappings for each corpus (1-based, matching run_od_analysis.py)."""
    word_to_idx = {}
    idx_to_word = {}

    for key, cfg in CORPORA_CONFIG.items():
        w2i = {}
        i2w = {}
        idx = 1
        with open(cfg["corpus_file"], "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    head = line.split(":", 1)[0].strip().lower()
                    if head not in w2i:
                        w2i[head] = idx
                        i2w[idx] = head
                        idx += 1
        word_to_idx[key] = w2i
        idx_to_word[key] = i2w

    return word_to_idx, idx_to_word


def main():
    parser = argparse.ArgumentParser(description="Calculate pairwise SOD differences (V6).")
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--total_jobs", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Job {args.job_id}: Loading common vocabulary and corpus mappings...")
    with open(COMMON_VOCAB_FILE, "r") as f:
        common_words = [line.strip() for line in f if line.strip()]

    word_to_master = {word: i for i, word in enumerate(common_words)}
    corpus_w2i, corpus_i2w = build_corpus_mappings()

    chunk_size = (len(common_words) + args.total_jobs - 1) // args.total_jobs
    start = (args.job_id - 1) * chunk_size
    end = min(len(common_words), start + chunk_size)
    words_for_this_job = common_words[start:end]

    print(f"Job {args.job_id}: Processing {len(words_for_this_job)} words (master idx {start} to {end - 1}).")

    corpus_combinations = list(itertools.combinations(CORPORA_KEYS, 2))

    file_handles = {}
    for key_A, key_B in corpus_combinations:
        output_key = f"{key_A}_vs_{key_B}"
        temp_path = os.path.join(OUTPUT_DIR, f".diffs_{output_key}_part_{args.job_id}.csv.tmp")
        file_handles[output_key] = open(temp_path, "w")

    for word in tqdm(words_for_this_job, desc=f"Job {args.job_id}"):
        master_idx = word_to_master[word]

        for key_A, key_B in corpus_combinations:
            try:
                idx_A = corpus_w2i[key_A].get(word)
                idx_B = corpus_w2i[key_B].get(word)
                if idx_A is None or idx_B is None:
                    continue

                path_A = os.path.join(CORPORA_CONFIG[key_A]["index_dir"], f"{idx_A}.csv")
                path_B = os.path.join(CORPORA_CONFIG[key_B]["index_dir"], f"{idx_B}.csv")

                df_A = pd.read_csv(path_A, header=None, names=['partner_idx', 'sod_A', 'sod_tl_A'])
                df_B = pd.read_csv(path_B, header=None, names=['partner_idx', 'sod_B', 'sod_tl_B'])

                # Remap corpus-specific partner indices to word names
                i2w_A = corpus_i2w[key_A]
                i2w_B = corpus_i2w[key_B]
                df_A['partner_word'] = df_A['partner_idx'].map(i2w_A)
                df_B['partner_word'] = df_B['partner_idx'].map(i2w_B)

                df_A = df_A.dropna(subset=['partner_word']).set_index('partner_word')
                df_B = df_B.dropna(subset=['partner_word']).set_index('partner_word')

                df_merged = df_A[['sod_A', 'sod_tl_A']].join(df_B[['sod_B', 'sod_tl_B']], how='inner')
                if df_merged.empty:
                    continue

                # Upper-triangle filter: only keep partners with master_idx > current word
                df_merged['partner_master'] = df_merged.index.map(word_to_master)
                df_merged = df_merged.dropna(subset=['partner_master'])
                df_merged = df_merged[df_merged['partner_master'] > master_idx]
                if df_merged.empty:
                    continue

                df_merged['diff_sod'] = df_merged['sod_A'] - df_merged['sod_B']
                df_merged['diff_sod_tl'] = df_merged['sod_tl_A'] - df_merged['sod_tl_B']
                df_merged['master_idx1'] = master_idx
                df_merged['master_idx2'] = df_merged['partner_master'].astype(int)

                output_df = df_merged[['master_idx1', 'master_idx2', 'diff_sod', 'diff_sod_tl']]

                output_key = f"{key_A}_vs_{key_B}"
                csv_string = output_df.to_csv(header=False, index=False, sep=',')
                file_handles[output_key].write(csv_string)

            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error processing word '{word}' for {key_A} vs {key_B}: {e}")

    print(f"Job {args.job_id}: Closing and renaming output files...")
    for key_A, key_B in corpus_combinations:
        output_key = f"{key_A}_vs_{key_B}"
        temp_path = os.path.join(OUTPUT_DIR, f".diffs_{output_key}_part_{args.job_id}.csv.tmp")
        final_path = os.path.join(OUTPUT_DIR, f"diffs_{output_key}_part_{args.job_id}.csv")

        file_handles[output_key].close()
        os.rename(temp_path, final_path)

    sentinel_file = os.path.join(OUTPUT_DIR, f"job_{args.job_id}.done")
    with open(sentinel_file, "w") as f:
        f.write(f"completed: {len(words_for_this_job)} words\n")

    print(f"Job {args.job_id}: Finished.")


if __name__ == "__main__":
    main()
