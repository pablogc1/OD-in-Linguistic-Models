#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ontological Dynamics Analysis (Phase 6) — V4: Equal-Count Percentile Walks
with On-the-Fly Aggregation + Distribution Histograms

Studies the stability of semantic proximity under random walks guided by
each word's SOD spectrum, using true equal-count percentile bins.

For each word, all pairwise SOD scores are sorted and divided into 100
equal-count bins (true percentiles: each bin contains ~1% of partners).

Starting pairs (A, B) are selected at 21 different percentile conditions:
  1st, 5th, 10th, 15th, ..., 95th, 100th
meaning A is in B's k-th percentile AND B is in A's k-th percentile.
ALL mutual pairs are used (no cap).

Walk variants (all non-repeating):
  1. CHAIN: At each step, current word picks from its 1st percentile.
     Run for ALL 21 starting conditions.
  2. SWEEP: Ascending sweep through percentiles 1->100 of the ORIGINAL A,B.
     Run only for 1st-percentile starting pairs.

Statistics aggregated on-the-fly: sums, sums-of-squares, and histograms
for SOD, TL, and percentile rank at every step.

Output: two compact CSVs per job:
  - *_stats.csv:  per-step sums/counts
  - *_hist.csv:   per-step histograms (TL, pctl rank, SOD)

Usage:
    python3 step6_dynamics_worker.py --corpus_key curated_ground_filtered \
        --job_id 1 --total_jobs 40 --num_trajectories 100 --max_steps 100
"""
import os
import sys
import random
import argparse
import numpy as np
from tqdm import tqdm

OUTPUT_DIR = "dynamics_results"

STARTING_PERCENTILES = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                        50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

SOD_HIST_EDGES = np.concatenate([
    [0], np.logspace(0, 8, 81)
])
N_SOD_BINS = len(SOD_HIST_EDGES) - 1
MAX_TL = 25
N_PCTL = 100


# ==============================================================================
# Data loading
# ==============================================================================

def load_vocabulary(corpus_file):
    word_to_idx = {}
    idx_to_word = {}
    idx = 1
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                head = line.split(":", 1)[0].strip().lower()
                if head not in word_to_idx:
                    word_to_idx[head] = idx
                    idx_to_word[idx] = head
                    idx += 1
    return word_to_idx, idx_to_word


def load_pair_data(index_dir, idx_to_word, num_words):
    max_idx = num_words + 2
    pair_sod = {}
    pair_tl = {}

    for idx in tqdm(idx_to_word, desc="  Loading pair data"):
        filepath = os.path.join(index_dir, f"{idx}.csv")
        if not os.path.exists(filepath):
            continue

        sod_arr = np.full(max_idx, -1, dtype=np.int64)
        tl_arr = np.full(max_idx, -1, dtype=np.int8)

        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                partner_idx = int(parts[0])
                sod = int(parts[1])
                sod_tl = int(parts[2])
                if partner_idx < max_idx:
                    sod_arr[partner_idx] = sod
                    tl_arr[partner_idx] = sod_tl

        pair_sod[idx] = sod_arr
        pair_tl[idx] = tl_arr

    return pair_sod, pair_tl


# ==============================================================================
# Percentile bin computation (equal-count / true percentiles)
# ==============================================================================

def build_percentile_bins(pair_sod, idx_to_word):
    max_idx = max(idx_to_word.keys()) + 1
    has_bin = np.zeros((max_idx, 100), dtype=np.bool_)
    bins_arrs = {}
    EMPTY = np.array([], dtype=np.int16)
    num_bins = 100

    for idx in tqdm(idx_to_word, desc="  Building 100-bin spectra"):
        sod_arr = pair_sod.get(idx)
        if sod_arr is None:
            continue

        valid_mask = sod_arr > 0
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            bins_arrs[idx] = tuple(EMPTY for _ in range(num_bins))
            continue

        valid_sods = sod_arr[valid_indices]
        sorted_order = np.argsort(valid_sods)
        sorted_indices = valid_indices[sorted_order]

        n = len(sorted_indices)
        chunk_size = n // num_bins
        remainder = n % num_bins

        result = []
        pos = 0
        for k in range(num_bins):
            size = chunk_size + (1 if k < remainder else 0)
            if size > 0:
                has_bin[idx, k] = True
                result.append(sorted_indices[pos:pos + size].astype(np.int16))
            else:
                result.append(EMPTY)
            pos += size
        bins_arrs[idx] = tuple(result)

    return bins_arrs, has_bin


def precompute_bin0_sets(bins_arrs, idx_to_word):
    """Pre-cache bin 0 (1st percentile) as Python sets for all words."""
    bin0 = {}
    for idx in idx_to_word:
        if idx in bins_arrs and len(bins_arrs[idx][0]) > 0:
            bin0[idx] = set(bins_arrs[idx][0].astype(np.int32).tolist())
        else:
            bin0[idx] = set()
    return bin0


# ==============================================================================
# Sorted SOD arrays for percentile rank lookup
# ==============================================================================

def build_sorted_sod(pair_sod, idx_to_word):
    sorted_sods = {}
    for idx in tqdm(idx_to_word, desc="  Building sorted SOD arrays"):
        sod_arr = pair_sod.get(idx)
        if sod_arr is None:
            continue
        valid_mask = sod_arr > 0
        valid_sods = sod_arr[valid_mask]
        if len(valid_sods) == 0:
            continue
        sorted_sods[idx] = np.sort(valid_sods)
    return sorted_sods


def compute_pctl_rank(sorted_arr, sod_value):
    if sorted_arr is None or len(sorted_arr) == 0:
        return -1
    rank = np.searchsorted(sorted_arr, sod_value, side="right")
    pctl = int(np.ceil(100.0 * rank / len(sorted_arr)))
    return max(1, min(100, pctl))


# ==============================================================================
# Helper
# ==============================================================================

def _bin_to_set(arr):
    if len(arr) == 0:
        return set()
    return set(arr.astype(np.int32).tolist())


# ==============================================================================
# Starting pair selection
# ==============================================================================

def find_starting_pairs_for_percentile(bins_arrs, idx_to_word, target_bin,
                                       seed=42):
    rng = random.Random(seed)

    bin_sets = {}
    for idx in idx_to_word:
        if idx in bins_arrs and len(bins_arrs[idx][target_bin]) > 0:
            bin_sets[idx] = _bin_to_set(bins_arrs[idx][target_bin])

    valid_pairs = []
    for idx_a in sorted(bin_sets.keys()):
        for idx_b in bin_sets[idx_a]:
            if idx_b <= idx_a:
                continue
            if idx_b in bin_sets and idx_a in bin_sets[idx_b]:
                valid_pairs.append((idx_a, idx_b))

    rng.shuffle(valid_pairs)
    return valid_pairs


# ==============================================================================
# SOD/TL lookup
# ==============================================================================

def lookup_pair(pair_sod, pair_tl, idx_a, idx_b):
    sod_arr = pair_sod.get(idx_a)
    if sod_arr is None:
        return None, None
    sod = int(sod_arr[idx_b])
    if sod == -1:
        return None, None
    tl = int(pair_tl[idx_a][idx_b])
    return sod, tl


# ==============================================================================
# On-the-fly statistics accumulator with histograms
# ==============================================================================

class StepAccumulator:
    def __init__(self, max_step=100):
        self.max_step = max_step
        n = max_step + 1
        self.count = np.zeros(n, dtype=np.int64)
        self.sod_sum = np.zeros(n, dtype=np.float64)
        self.sod_sq_sum = np.zeros(n, dtype=np.float64)
        self.tl_sum = np.zeros(n, dtype=np.float64)
        self.tl_sq_sum = np.zeros(n, dtype=np.float64)
        self.pctl_a_sum = np.zeros(n, dtype=np.float64)
        self.pctl_a_sq_sum = np.zeros(n, dtype=np.float64)
        self.pctl_b_sum = np.zeros(n, dtype=np.float64)
        self.pctl_b_sq_sum = np.zeros(n, dtype=np.float64)

        # Cross-products for correlations
        self.sod_tl_sum = np.zeros(n, dtype=np.float64)
        self.pctl_ab_sum = np.zeros(n, dtype=np.float64)

        # Log SOD for geometric mean
        self.log_sod_sum = np.zeros(n, dtype=np.float64)
        self.log_sod_sq_sum = np.zeros(n, dtype=np.float64)

        # Step-to-step SOD delta (accumulated per trajectory via add_delta)
        self.delta_sod_sum = np.zeros(n, dtype=np.float64)
        self.delta_sod_sq_sum = np.zeros(n, dtype=np.float64)
        self.delta_sod_count = np.zeros(n, dtype=np.int64)

        # Histograms
        self.tl_hist = np.zeros((n, MAX_TL + 1), dtype=np.int64)
        self.pctl_a_hist = np.zeros((n, N_PCTL + 1), dtype=np.int64)
        self.pctl_b_hist = np.zeros((n, N_PCTL + 1), dtype=np.int64)
        self.sod_hist = np.zeros((n, N_SOD_BINS), dtype=np.int64)

        # Walk length histogram (exact counts for each possible length 0..max_step+1)
        self.walk_len_hist = np.zeros(n + 1, dtype=np.int64)

        # Walk death cause counters (only at the step where the walk died)
        self.death_pool_a_empty = np.zeros(n, dtype=np.int64)
        self.death_pool_b_empty = np.zeros(n, dtype=np.int64)
        self.death_both_empty = np.zeros(n, dtype=np.int64)
        self.death_no_bin = np.zeros(n, dtype=np.int64)

    def add_step(self, step, sod, tl, pctl_a, pctl_b):
        if step > self.max_step:
            return
        self.count[step] += 1
        if sod is not None:
            fsod = float(sod)
            self.sod_sum[step] += fsod
            self.sod_sq_sum[step] += fsod * fsod
            if sod > 0:
                ls = np.log(fsod)
                self.log_sod_sum[step] += ls
                self.log_sod_sq_sum[step] += ls * ls
            b = np.searchsorted(SOD_HIST_EDGES, sod, side="right") - 1
            self.sod_hist[step, min(b, N_SOD_BINS - 1)] += 1
        if tl is not None:
            self.tl_sum[step] += tl
            self.tl_sq_sum[step] += tl * tl
            self.tl_hist[step, min(tl, MAX_TL)] += 1
        if sod is not None and tl is not None:
            self.sod_tl_sum[step] += float(sod) * tl
        if pctl_a > 0:
            self.pctl_a_sum[step] += pctl_a
            self.pctl_a_sq_sum[step] += pctl_a * pctl_a
            self.pctl_a_hist[step, min(pctl_a, N_PCTL)] += 1
        if pctl_b > 0:
            self.pctl_b_sum[step] += pctl_b
            self.pctl_b_sq_sum[step] += pctl_b * pctl_b
            self.pctl_b_hist[step, min(pctl_b, N_PCTL)] += 1
        if pctl_a > 0 and pctl_b > 0:
            self.pctl_ab_sum[step] += pctl_a * pctl_b

    def add_delta(self, step, prev_sod, cur_sod):
        if step > self.max_step or step < 1:
            return
        if prev_sod is not None and cur_sod is not None:
            delta = float(cur_sod) - float(prev_sod)
            self.delta_sod_sum[step] += delta
            self.delta_sod_sq_sum[step] += delta * delta
            self.delta_sod_count[step] += 1

    def record_walk_length(self, length):
        idx = min(length, self.max_step + 1)
        self.walk_len_hist[idx] += 1

    def record_death(self, step, cause):
        if step > self.max_step:
            return
        if cause == 'pool_a':
            self.death_pool_a_empty[step] += 1
        elif cause == 'pool_b':
            self.death_pool_b_empty[step] += 1
        elif cause == 'both':
            self.death_both_empty[step] += 1
        elif cause == 'no_bin':
            self.death_no_bin[step] += 1


# ==============================================================================
# Walk variants (optimized: use pre-cached bin0 sets)
# ==============================================================================

def run_sweep_accum(idx_a, idx_b, bins_arrs, pair_sod, pair_tl,
                    sorted_sods, rng, accum):
    used_a = set()
    used_b = set()
    sa = bins_arrs.get(idx_a)
    sb = bins_arrs.get(idx_b)
    if sa is None or sb is None:
        accum.record_walk_length(0)
        return

    last_step = -1
    prev_sod = None
    for step in range(len(sa)):
        avail_a = list(_bin_to_set(sa[step]) - used_b)
        avail_b = list(_bin_to_set(sb[step]) - used_a)

        if not avail_a or not avail_b:
            a_empty = len(avail_a) == 0
            b_empty = len(avail_b) == 0
            if a_empty and b_empty:
                accum.record_death(step, 'both')
            elif a_empty:
                accum.record_death(step, 'pool_a')
            else:
                accum.record_death(step, 'pool_b')
            break

        pa = rng.choice(avail_a)
        pb = rng.choice(avail_b)
        used_a.add(pa)
        used_b.add(pb)

        sod, tl = lookup_pair(pair_sod, pair_tl, pa, pb)
        pctl_a = compute_pctl_rank(sorted_sods.get(pa), sod) if sod is not None else -1
        pctl_b = compute_pctl_rank(sorted_sods.get(pb), sod) if sod is not None else -1

        accum.add_step(step, sod, tl, pctl_a, pctl_b)
        if step > 0:
            accum.add_delta(step, prev_sod, sod)
        prev_sod = sod
        last_step = step

    accum.record_walk_length(last_step + 1)


def run_chain_accum(idx_a, idx_b, bin0_sets, has_bin, pair_sod, pair_tl,
                    sorted_sods, max_steps, rng, accum):
    """Chained walk using pre-cached bin0 sets."""
    cur_a, cur_b = idx_a, idx_b
    used_a = {idx_a}
    used_b = {idx_b}

    prev_sod, tl = lookup_pair(pair_sod, pair_tl, cur_a, cur_b)
    pctl_a = compute_pctl_rank(sorted_sods.get(cur_a), prev_sod) if prev_sod is not None else -1
    pctl_b = compute_pctl_rank(sorted_sods.get(cur_b), prev_sod) if prev_sod is not None else -1
    accum.add_step(0, prev_sod, tl, pctl_a, pctl_b)

    last_step = 0
    for step in range(1, max_steps + 1):
        set_a = bin0_sets.get(cur_a)
        set_b = bin0_sets.get(cur_b)
        if not set_a or not set_b:
            accum.record_death(step, 'no_bin')
            break

        pool_a = set_a - used_b
        pool_b = set_b - used_a

        candidates_a = [p for p in pool_a if has_bin[p, 0]]
        candidates_b = [p for p in pool_b if has_bin[p, 0]]

        if not candidates_a:
            candidates_a = list(pool_a)
        if not candidates_b:
            candidates_b = list(pool_b)

        if not candidates_a or not candidates_b:
            a_empty = len(candidates_a) == 0
            b_empty = len(candidates_b) == 0
            if a_empty and b_empty:
                accum.record_death(step, 'both')
            elif a_empty:
                accum.record_death(step, 'pool_a')
            else:
                accum.record_death(step, 'pool_b')
            break

        cur_a = rng.choice(candidates_a)
        cur_b = rng.choice(candidates_b)
        used_a.add(cur_a)
        used_b.add(cur_b)

        sod, tl = lookup_pair(pair_sod, pair_tl, cur_a, cur_b)
        pctl_a = compute_pctl_rank(sorted_sods.get(cur_a), sod) if sod is not None else -1
        pctl_b = compute_pctl_rank(sorted_sods.get(cur_b), sod) if sod is not None else -1
        accum.add_step(step, sod, tl, pctl_a, pctl_b)
        accum.add_delta(step, prev_sod, sod)
        prev_sod = sod
        last_step = step

    accum.record_walk_length(last_step + 1)


# ==============================================================================
# Diagnostics
# ==============================================================================

def print_diagnostics(bins_arrs, idx_to_word, all_starting_pairs, pair_sod, pair_tl):
    print("\n" + "=" * 60)
    print("DIAGNOSTICS")
    print("=" * 60)

    p1_sizes = []
    for idx in idx_to_word:
        if idx in bins_arrs and len(bins_arrs[idx][0]) > 0:
            p1_sizes.append(len(bins_arrs[idx][0]))

    if p1_sizes:
        p1_arr = np.array(p1_sizes)
        print(f"\n  1st percentile bin sizes (per word):")
        print(f"    Mean: {p1_arr.mean():.1f}, Median: {np.median(p1_arr):.0f}, "
              f"Min: {p1_arr.min()}, Max: {p1_arr.max()}, Std: {p1_arr.std():.1f}")
        print(f"    Words with non-empty 1st pctl: {len(p1_sizes)} / {len(idx_to_word)}")

    print(f"\n  Starting pairs per percentile:")
    for pctl, pairs in sorted(all_starting_pairs.items()):
        if pairs:
            sods, tls = [], []
            for idx_a, idx_b in pairs[:500]:
                s, t = lookup_pair(pair_sod, pair_tl, idx_a, idx_b)
                if s is not None:
                    sods.append(s)
                    tls.append(t)
            if sods:
                print(f"    P{pctl:3d}: {len(pairs):6d} pairs | "
                      f"SOD mean={np.mean(sods):.0f}, median={np.median(sods):.0f} | "
                      f"TL mean={np.mean(tls):.1f}, median={np.median(tls):.0f}")
            else:
                print(f"    P{pctl:3d}: {len(pairs):6d} pairs | no valid SOD/TL data")
        else:
            print(f"    P{pctl:3d}:      0 pairs")

    print("=" * 60 + "\n")


# ==============================================================================
# Output
# ==============================================================================

def write_stats_csv(filepath, corpus_key, accumulators):
    with open(filepath, "w") as f:
        f.write("corpus,start_pctl,variant,step,"
                "count,sod_sum,sod_sq_sum,tl_sum,tl_sq_sum,"
                "pctl_a_sum,pctl_a_sq_sum,pctl_b_sum,pctl_b_sq_sum,"
                "sod_tl_sum,pctl_ab_sum,"
                "log_sod_sum,log_sod_sq_sum,"
                "delta_sod_sum,delta_sod_sq_sum,delta_sod_count,"
                "death_pool_a,death_pool_b,death_both,death_no_bin\n")

        for (sp, variant), accum in sorted(accumulators.items()):
            for step in range(accum.max_step + 1):
                if accum.count[step] == 0 and accum.death_pool_a_empty[step] == 0 \
                        and accum.death_pool_b_empty[step] == 0 \
                        and accum.death_both_empty[step] == 0 \
                        and accum.death_no_bin[step] == 0:
                    continue
                f.write(f"{corpus_key},{sp},{variant},{step},"
                        f"{accum.count[step]},"
                        f"{accum.sod_sum[step]:.0f},{accum.sod_sq_sum[step]:.0f},"
                        f"{accum.tl_sum[step]:.0f},{accum.tl_sq_sum[step]:.0f},"
                        f"{accum.pctl_a_sum[step]:.0f},{accum.pctl_a_sq_sum[step]:.0f},"
                        f"{accum.pctl_b_sum[step]:.0f},{accum.pctl_b_sq_sum[step]:.0f},"
                        f"{accum.sod_tl_sum[step]:.0f},{accum.pctl_ab_sum[step]:.0f},"
                        f"{accum.log_sod_sum[step]:.6f},{accum.log_sod_sq_sum[step]:.6f},"
                        f"{accum.delta_sod_sum[step]:.0f},{accum.delta_sod_sq_sum[step]:.0f},"
                        f"{accum.delta_sod_count[step]},"
                        f"{accum.death_pool_a_empty[step]},"
                        f"{accum.death_pool_b_empty[step]},"
                        f"{accum.death_both_empty[step]},"
                        f"{accum.death_no_bin[step]}\n")


def write_hist_csv(filepath, corpus_key, accumulators):
    with open(filepath, "w") as f:
        f.write("corpus,start_pctl,variant,step,hist_type,bin_idx,bin_count\n")

        for (sp, variant), accum in sorted(accumulators.items()):
            # Walk length histogram (step column = -1 as sentinel)
            for length in range(accum.max_step + 2):
                if accum.walk_len_hist[length] > 0:
                    f.write(f"{corpus_key},{sp},{variant},-1,"
                            f"walk_len,{length},{accum.walk_len_hist[length]}\n")

            # Per-step histograms
            for step in range(accum.max_step + 1):
                if accum.count[step] == 0:
                    continue

                for b in range(MAX_TL + 1):
                    if accum.tl_hist[step, b] > 0:
                        f.write(f"{corpus_key},{sp},{variant},{step},"
                                f"tl,{b},{accum.tl_hist[step, b]}\n")

                for b in range(1, N_PCTL + 1):
                    if accum.pctl_a_hist[step, b] > 0:
                        f.write(f"{corpus_key},{sp},{variant},{step},"
                                f"pctl_a,{b},{accum.pctl_a_hist[step, b]}\n")

                for b in range(1, N_PCTL + 1):
                    if accum.pctl_b_hist[step, b] > 0:
                        f.write(f"{corpus_key},{sp},{variant},{step},"
                                f"pctl_b,{b},{accum.pctl_b_hist[step, b]}\n")

                for b in range(N_SOD_BINS):
                    if accum.sod_hist[step, b] > 0:
                        f.write(f"{corpus_key},{sp},{variant},{step},"
                                f"sod,{b},{accum.sod_hist[step, b]}\n")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ontological dynamics analysis (Phase 6 V4 — aggregated walks)."
    )
    parser.add_argument("--corpus_key", type=str, required=True)
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--total_jobs", type=int, required=True)
    parser.add_argument("--num_trajectories", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    corpus_file = f"curated_corpora/{args.corpus_key}.txt"
    index_dir = f"results_{args.corpus_key}/indexed_pairs_data"

    if not os.path.exists(corpus_file):
        print(f"ERROR: Corpus file not found: {corpus_file}")
        return

    # --- Load data ---
    print(f"Loading corpus: {corpus_file}")
    word_to_idx, idx_to_word = load_vocabulary(corpus_file)
    num_words = len(word_to_idx)
    print(f"  Vocabulary: {num_words} words")

    print("Loading pair data...")
    pair_sod, pair_tl = load_pair_data(index_dir, idx_to_word, num_words)
    print(f"  Loaded data for {len(pair_sod)} words")

    # --- Build percentile bins ---
    print("Building spectral bins...")
    bins_arrs, has_bin = build_percentile_bins(pair_sod, idx_to_word)

    # --- Pre-cache bin 0 sets for speed ---
    print("Pre-caching 1st percentile sets...")
    bin0_sets = precompute_bin0_sets(bins_arrs, idx_to_word)

    # --- Build sorted SOD arrays for percentile rank lookups ---
    print("Building sorted SOD arrays for rank computation...")
    sorted_sods = build_sorted_sod(pair_sod, idx_to_word)

    # --- Find starting pairs for each percentile condition ---
    print("Finding mutual starting pairs for each percentile...")
    all_starting_pairs = {}
    for pctl in STARTING_PERCENTILES:
        target_bin = pctl - 1
        pairs = find_starting_pairs_for_percentile(
            bins_arrs, idx_to_word, target_bin, seed=args.seed
        )
        all_starting_pairs[pctl] = pairs
        print(f"  P{pctl:3d}: {len(pairs)} mutual pairs found")

    total_pairs = sum(len(p) for p in all_starting_pairs.values())
    if total_pairs == 0:
        print("ERROR: No valid starting pairs found for any percentile. Exiting.")
        return

    # --- Diagnostics (job 1 only) ---
    if args.job_id == 1:
        print_diagnostics(bins_arrs, idx_to_word, all_starting_pairs,
                          pair_sod, pair_tl)

    # --- Build flat work list ---
    work_items = []
    global_pair_id = 0
    for pctl in STARTING_PERCENTILES:
        for local_i, (idx_a, idx_b) in enumerate(all_starting_pairs[pctl]):
            do_sweep = (pctl == 1)
            work_items.append((pctl, global_pair_id, idx_a, idx_b, do_sweep))
            global_pair_id += 1

    # --- Shard work across jobs ---
    chunk_size = (len(work_items) + args.total_jobs - 1) // args.total_jobs
    start = (args.job_id - 1) * chunk_size
    end = min(len(work_items), start + chunk_size)
    my_items = work_items[start:end]

    n_sweep = sum(1 for w in my_items if w[4])
    print(f"Job {args.job_id}/{args.total_jobs}: {len(my_items)} pair-conditions "
          f"({len(my_items)} chains + {n_sweep} sweeps) "
          f"x {args.num_trajectories} trajectories")
    print(f"  Total pairs across all percentiles: {total_pairs}")

    # --- Initialize accumulators ---
    accumulators = {}
    for pctl in STARTING_PERCENTILES:
        accumulators[(pctl, 'chain')] = StepAccumulator(args.max_steps)
    accumulators[(1, 'sweep')] = StepAccumulator(args.max_steps)

    # --- Run walks with on-the-fly aggregation ---
    for item_i, (pctl, pair_id, idx_a, idx_b, do_sweep) in enumerate(
            tqdm(my_items, desc=f"Job {args.job_id}")):

        chain_accum = accumulators[(pctl, 'chain')]

        for traj in range(args.num_trajectories):
            traj_seed = args.seed + pair_id * 10000 + traj

            traj_rng = random.Random(traj_seed)
            run_chain_accum(idx_a, idx_b, bin0_sets, has_bin,
                            pair_sod, pair_tl, sorted_sods,
                            args.max_steps, traj_rng, chain_accum)

            if do_sweep:
                sweep_accum = accumulators[(1, 'sweep')]
                traj_rng = random.Random(traj_seed + 500000)
                run_sweep_accum(idx_a, idx_b, bins_arrs,
                                pair_sod, pair_tl, sorted_sods,
                                traj_rng, sweep_accum)

    # --- Write output ---
    stats_file = os.path.join(OUTPUT_DIR,
                              f"dynamics_{args.corpus_key}_job_{args.job_id}_stats.csv")
    hist_file = os.path.join(OUTPUT_DIR,
                             f"dynamics_{args.corpus_key}_job_{args.job_id}_hist.csv")

    write_stats_csv(stats_file, args.corpus_key, accumulators)
    write_hist_csv(hist_file, args.corpus_key, accumulators)

    sentinel_file = os.path.join(OUTPUT_DIR,
                                 f"dynamics_{args.corpus_key}_job_{args.job_id}.done")
    with open(sentinel_file, "w") as f:
        f.write(f"completed: {len(my_items)} pair-conditions, "
                f"{args.num_trajectories} trajectories (aggregated + histograms)\n")

    print(f"Job {args.job_id}: Finished.")
    print(f"  Stats: {stats_file}")
    print(f"  Histograms: {hist_file}")


if __name__ == "__main__":
    main()
