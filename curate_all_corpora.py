#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corpus Curation Script

APPROACH:
1. Load 3 base corpora: GF, AI, MW
2. Generate derived corpora from GF: RR, TR, NM
3. Make each corpus SELF-CONTAINED independently (the only curation requirement)
4. Save common_vocabulary.txt = intersection of all headwords (for EOD only)

IMPORTANT: Each corpus keeps its own vocabulary. We do NOT force all corpora
to have identical entries. TR removes words, so it will have fewer entries than GF.

Usage:
    python curate_all_corpora.py
"""

import os
import random
import pathlib
from collections import Counter

# --- Configuration ---
PROJECT_DIR = pathlib.Path(__file__).parent.resolve()

BASE_CORPORA = {
    'ground_filtered': PROJECT_DIR / 'raw_corpora/extracted_definitions_ground_filtered.txt',
    'ai_generated': PROJECT_DIR / 'raw_corpora/extracted_definitions_ai_generated.txt',
    'merriam_webster': PROJECT_DIR / 'raw_corpora/extracted_definitions_merriam_webster.txt',
}

OUTPUT_DIR = PROJECT_DIR / 'curated_corpora'

# Derived corpora settings
RANDOM_REMOVAL_PCT = 20  # % of multi-token definitions that get one token removed
TARGETED_REMOVAL_TOP_N = 20  # Remove top N most frequent words
RANDOM_SEED = 42


def load_corpus(filepath):
    """Load a corpus file into a dictionary."""
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found, skipping.")
        return None, None
    
    definitions = {}
    ordered_words = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            head, def_text = line.split(':', 1)
            head = head.strip().lower()
            tokens = def_text.strip().split()
            if head not in definitions:
                ordered_words.append(head)
            definitions[head] = tokens
    
    return definitions, ordered_words


def save_corpus(definitions, ordered_words, filepath):
    """Save a corpus to file."""
    count = 0
    with open(filepath, 'w', encoding='utf-8') as f:
        for word in ordered_words:
            if word in definitions and definitions[word]:
                f.write(f"{word}: {' '.join(definitions[word])}\n")
                count += 1
    return count


def make_self_contained(definitions, max_iterations=100):
    """
    Make a corpus self-contained: all tokens in definitions must be headwords.
    
    Iteratively removes:
    1. Tokens from definitions that reference non-existent headwords
    2. Entries whose definitions become empty
    
    Returns the self-contained corpus.
    """
    defs = {k: v[:] for k, v in definitions.items()}  # Deep copy
    
    for iteration in range(max_iterations):
        headwords = set(defs.keys())
        changes = False
        
        # Filter definitions to only include tokens that are headwords
        new_defs = {}
        for word, tokens in defs.items():
            filtered = [t for t in tokens if t in headwords]
            if filtered:
                if filtered != tokens:
                    changes = True
                new_defs[word] = filtered
            else:
                changes = True  # Entry removed
        
        defs = new_defs
        
        if not changes:
            break
    
    return defs


def create_random_removal(definitions, removal_pct, seed):
    """
    Create Random Removal corpus.
    
    For each definition with >1 token, with probability removal_pct/100,
    remove one random token.
    """
    random.seed(seed)
    
    new_defs = {}
    
    for word, tokens in definitions.items():
        new_tokens = tokens[:]
        
        if len(new_tokens) > 1 and random.random() < removal_pct / 100:
            idx = random.randint(0, len(new_tokens) - 1)
            new_tokens.pop(idx)
        
        if new_tokens:
            new_defs[word] = new_tokens
    
    return new_defs


def create_targeted_removal(definitions, top_n, seed):
    """
    Create Targeted Removal corpus - remove top N most frequent words entirely.
    
    These words are removed as:
    1. Headwords (entries deleted)
    2. Tokens in other definitions
    """
    random.seed(seed)
    
    # Count word frequencies in definitions
    freq = Counter()
    for tokens in definitions.values():
        freq.update(tokens)
    
    # Get most frequent words that are also headwords
    headwords = set(definitions.keys())
    most_frequent = [w for w, c in freq.most_common() if w in headwords]
    
    # Remove exactly top_n words
    words_to_remove = set(most_frequent[:top_n])
    
    print(f"    Removing {len(words_to_remove)} words: {', '.join(list(words_to_remove)[:10])}...")
    
    new_defs = {}
    for word, tokens in definitions.items():
        if word not in words_to_remove:
            new_tokens = [t for t in tokens if t not in words_to_remove]
            if new_tokens:
                new_defs[word] = new_tokens
    
    return new_defs


def create_null_model(definitions, seed):
    """
    Create Null Model corpus - randomize definitions.
    
    For each entry, replace the definition with random headwords
    (same length as original definition).
    """
    random.seed(seed)
    
    words = list(definitions.keys())
    new_defs = {}
    
    for word, tokens in definitions.items():
        num_tokens = len(tokens)
        if num_tokens > 0:
            new_defs[word] = random.choices(words, k=num_tokens)
    
    return new_defs


def main():
    print("=" * 70)
    print("  CORPUS CURATION")
    print("  Goal: Self-contained corpora (each corpus independent)")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load base corpora
    # =========================================================================
    print("\n[1] Loading base corpora...")
    
    all_corpora = {}
    all_orders = {}
    
    for name, filepath in BASE_CORPORA.items():
        defs, order = load_corpus(filepath)
        if defs is not None:
            all_corpora[name] = defs
            all_orders[name] = order
            print(f"  {name}: {len(defs)} words")
    
    # =========================================================================
    # STEP 2: Generate derived corpora from GF
    # =========================================================================
    print("\n[2] Generating derived corpora from Ground Filtered...")
    
    gf_defs = all_corpora['ground_filtered']
    gf_order = all_orders['ground_filtered']
    
    print(f"  Random Removal ({RANDOM_REMOVAL_PCT}% of multi-token definitions)...")
    all_corpora['random_removal'] = create_random_removal(gf_defs, RANDOM_REMOVAL_PCT, RANDOM_SEED)
    all_orders['random_removal'] = gf_order
    print(f"    -> {len(all_corpora['random_removal'])} entries")
    
    print(f"  Targeted Removal (top {TARGETED_REMOVAL_TOP_N} words)...")
    all_corpora['targeted_removal'] = create_targeted_removal(gf_defs, TARGETED_REMOVAL_TOP_N, RANDOM_SEED)
    all_orders['targeted_removal'] = gf_order
    print(f"    -> {len(all_corpora['targeted_removal'])} entries")
    
    print(f"  Null Model (randomized definitions)...")
    all_corpora['null_model'] = create_null_model(gf_defs, RANDOM_SEED)
    all_orders['null_model'] = gf_order
    print(f"    -> {len(all_corpora['null_model'])} entries")
    
    # =========================================================================
    # STEP 3: Make each corpus self-contained INDEPENDENTLY
    # =========================================================================
    print("\n[3] Making each corpus self-contained (independent curation)...")
    
    curated = {}
    for name in ['ground_filtered', 'ai_generated', 'merriam_webster', 
                 'random_removal', 'targeted_removal', 'null_model']:
        if name not in all_corpora:
            continue
        
        original_size = len(all_corpora[name])
        curated[name] = make_self_contained(all_corpora[name])
        final_size = len(curated[name])
        
        print(f"  {name}: {original_size} -> {final_size} entries")
    
    # =========================================================================
    # STEP 4: Save all curated corpora
    # =========================================================================
    print("\n[4] Saving curated corpora...")
    
    for name, defs in curated.items():
        order = all_orders[name]
        path = os.path.join(OUTPUT_DIR, f"curated_{name}.txt")
        count = save_corpus(defs, order, path)
        print(f"  {path}: {count} entries")
    
    # =========================================================================
    # STEP 5: Find and save common vocabulary (intersection for EOD)
    # =========================================================================
    print("\n[5] Computing common vocabulary (intersection of all headwords)...")
    
    vocab_sets = [set(defs.keys()) for defs in curated.values()]
    common_vocab = set.intersection(*vocab_sets)
    
    vocab_path = os.path.join(OUTPUT_DIR, "common_vocabulary.txt")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for word in sorted(common_vocab):
            f.write(f"{word}\n")
    
    print(f"  Common vocabulary: {len(common_vocab)} words")
    print(f"  Saved to: {vocab_path}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("  CURATION COMPLETE")
    print("=" * 70)
    print("\n  Corpus sizes (each self-contained independently):")
    for name, defs in curated.items():
        print(f"    {name}: {len(defs)} entries")
    print(f"\n  Common vocabulary (for EOD): {len(common_vocab)} words")
    print(f"  Output directory: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
