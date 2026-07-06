#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corpus Generation Script for Ontological Differentiation Analysis

This script generates all derived corpora from a base corpus (Ground Filtered).

Corpora generated:
1. Random Removal   - Randomly remove X% of words
2. Targeted Removal - Remove the most frequent words
3. Null Model       - Randomize definitions (preserve degree distribution)

Usage:
    python generate_corpora.py --base <ground_filtered.txt> --output-dir <dir>
    
Options:
    --random-pct N      Percentage of words to remove for Random Removal (default: 20)
    --targeted-pct N    Percentage of words to remove for Targeted Removal (default: 20)
    --seed N            Random seed for reproducibility (default: 42)
    --only <corpus>     Generate only specific corpus: random, targeted, null, or all
"""

import os
import sys
import argparse
import random
from collections import Counter
from tqdm import tqdm


def read_definitions(filepath):
    """Reads definitions file and returns ordered dict."""
    if not os.path.exists(filepath):
        print(f"ERROR: Input file not found at '{filepath}'")
        return None, None
    
    definitions = {}
    ordered_words = []
    
    with open(filepath, "r", encoding="utf-8") as f:
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


def save_definitions(definitions, ordered_words, filepath):
    """Saves definitions to file, preserving order."""
    with open(filepath, "w", encoding="utf-8") as f:
        for word in ordered_words:
            if word in definitions:
                f.write(f"{word}: {' '.join(definitions[word])}\n")
    print(f"  Saved: {filepath} ({len(definitions)} words)")


def clean_corpus(definitions, ordered_words):
    """
    Iteratively prune dangling words until corpus is self-contained.
    Returns cleaned definitions and ordered words.
    """
    print("  Cleaning corpus (removing dangling words)...")
    
    definitions = {k: list(v) for k, v in definitions.items()}  # Deep copy
    
    iteration = 0
    while True:
        iteration += 1
        valid_headwords = set(definitions.keys())
        
        # Prune dangling words from definitions
        for headword in list(definitions.keys()):
            definitions[headword] = [t for t in definitions[headword] if t in valid_headwords]
        
        # Remove empty definitions
        to_remove = [h for h, d in definitions.items() if not d]
        if not to_remove:
            break
        
        for h in to_remove:
            del definitions[h]
    
    # Update ordered words
    cleaned_ordered = [w for w in ordered_words if w in definitions]
    
    print(f"    Iterations: {iteration}, Final size: {len(definitions)} words")
    return definitions, cleaned_ordered


def create_random_removal(definitions, ordered_words, removal_pct, seed):
    """
    Create Random Removal corpus by randomly removing X% of words.
    """
    print(f"\n[1/3] Creating Random Removal corpus ({removal_pct}% removal)...")
    
    random.seed(seed)
    
    # Calculate how many words to remove
    num_words = len(ordered_words)
    num_to_remove = int(num_words * removal_pct / 100)
    
    print(f"  Removing {num_to_remove} of {num_words} words...")
    
    # Randomly select words to remove
    words_to_remove = set(random.sample(ordered_words, num_to_remove))
    
    # Create new corpus without removed words
    new_definitions = {}
    new_ordered = []
    
    for word in ordered_words:
        if word not in words_to_remove:
            # Keep word, but also remove references to deleted words in definition
            new_def = [t for t in definitions[word] if t not in words_to_remove]
            new_definitions[word] = new_def
            new_ordered.append(word)
    
    # Clean the corpus (will remove words with empty definitions)
    return clean_corpus(new_definitions, new_ordered)


def create_targeted_removal(definitions, ordered_words, removal_pct, seed):
    """
    Create Targeted Removal corpus by removing the most frequent words.
    Frequency = how often a word appears in OTHER words' definitions.
    """
    print(f"\n[2/3] Creating Targeted Removal corpus ({removal_pct}% removal)...")
    
    random.seed(seed)  # For any tie-breaking
    
    # Count word frequencies (how often each word is used in definitions)
    word_freq = Counter()
    for word, def_tokens in definitions.items():
        word_freq.update(def_tokens)
    
    # Calculate how many to remove
    num_words = len(ordered_words)
    num_to_remove = int(num_words * removal_pct / 100)
    
    # Get the most frequent words
    most_frequent = [word for word, count in word_freq.most_common()]
    
    # Only remove words that are actually headwords
    words_to_remove = set()
    headword_set = set(ordered_words)
    for word in most_frequent:
        if word in headword_set:
            words_to_remove.add(word)
        if len(words_to_remove) >= num_to_remove:
            break
    
    print(f"  Removing {len(words_to_remove)} most frequent words...")
    
    # Show top 10 removed words
    removed_list = list(words_to_remove)[:10]
    print(f"  Top removed: {', '.join(removed_list)}...")
    
    # Create new corpus without removed words
    new_definitions = {}
    new_ordered = []
    
    for word in ordered_words:
        if word not in words_to_remove:
            new_def = [t for t in definitions[word] if t not in words_to_remove]
            new_definitions[word] = new_def
            new_ordered.append(word)
    
    # Clean the corpus
    return clean_corpus(new_definitions, new_ordered)


def create_null_model(definitions, ordered_words, seed):
    """
    Create Null Model corpus by randomizing definitions.
    Preserves out-degree (definition length) but randomizes which words are used.
    """
    print(f"\n[3/3] Creating Null Model corpus (randomized definitions)...")
    
    random.seed(seed)
    
    all_words = list(definitions.keys())
    null_definitions = {}
    
    for word in tqdm(ordered_words, desc="  Rewiring"):
        def_tokens = definitions[word]
        num_tokens = len(def_tokens)
        
        if num_tokens > 0:
            # Replace with random words from vocabulary
            null_definitions[word] = random.choices(all_words, k=num_tokens)
        else:
            null_definitions[word] = []
    
    # Note: Null model doesn't need cleaning - all words are valid headwords
    return null_definitions, ordered_words


def main():
    parser = argparse.ArgumentParser(
        description="Generate derived corpora for OD analysis."
    )
    parser.add_argument("--base", required=True,
                        help="Path to base corpus (Ground Filtered)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for generated corpora")
    parser.add_argument("--random-pct", type=float, default=20,
                        help="Percentage to remove for Random Removal (default: 20)")
    parser.add_argument("--targeted-pct", type=float, default=20,
                        help="Percentage to remove for Targeted Removal (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--only", choices=['random', 'targeted', 'null', 'all'],
                        default='all', help="Which corpus to generate (default: all)")
    args = parser.parse_args()
    
    print("="*60)
    print("  CORPUS GENERATION")
    print("="*60)
    print(f"  Base corpus:    {args.base}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Random seed:    {args.seed}")
    print(f"  Generate:       {args.only}")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load base corpus
    print(f"\nLoading base corpus: {args.base}")
    definitions, ordered_words = read_definitions(args.base)
    
    if definitions is None:
        sys.exit(1)
    
    print(f"  Loaded {len(definitions)} words")
    
    # Generate corpora
    if args.only in ['random', 'all']:
        rand_defs, rand_order = create_random_removal(
            definitions, ordered_words, args.random_pct, args.seed
        )
        save_definitions(
            rand_defs, rand_order,
            os.path.join(args.output_dir, "extracted_definitions_random_removal.txt")
        )
    
    if args.only in ['targeted', 'all']:
        targ_defs, targ_order = create_targeted_removal(
            definitions, ordered_words, args.targeted_pct, args.seed
        )
        save_definitions(
            targ_defs, targ_order,
            os.path.join(args.output_dir, "extracted_definitions_targeted_removal.txt")
        )
    
    if args.only in ['null', 'all']:
        null_defs, null_order = create_null_model(
            definitions, ordered_words, args.seed
        )
        save_definitions(
            null_defs, null_order,
            os.path.join(args.output_dir, "extracted_definitions_null_model.txt")
        )
    
    print("\n" + "="*60)
    print("  CORPUS GENERATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
