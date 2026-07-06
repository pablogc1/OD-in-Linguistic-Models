#!/usr/bin/env python3
"""
Build a set of valid vocabulary words from the Wiktionary corpus.
These words have already been validated as nouns/verbs/adjectives.
Output: valid_vocabulary.txt (one word per line)
"""

import sys

WIKTIONARY_FILE = "extracted_definitions.txt"
OUTPUT_FILE = "valid_vocabulary.txt"

def main():
    print("Building valid vocabulary from Wiktionary corpus...")
    
    valid_words = set()
    
    with open(WIKTIONARY_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' not in line:
                continue
            
            # Get the headword (before the colon)
            headword = line.split(':', 1)[0].strip().lower()
            if headword and headword.isalpha() and len(headword) >= 2:
                valid_words.add(headword)
            
            # Get all words in the definition (after the colon)
            definition = line.split(':', 1)[1].strip()
            for word in definition.split():
                word = word.strip().lower()
                if word and word.isalpha() and len(word) >= 2:
                    valid_words.add(word)
    
    # Sort and write
    sorted_words = sorted(valid_words)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for word in sorted_words:
            f.write(word + '\n')
    
    print(f"✓ Extracted {len(sorted_words)} valid words")
    print(f"✓ Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
