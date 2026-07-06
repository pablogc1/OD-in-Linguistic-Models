#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BATCH-FOCUSED version of the Merriam-Webster Scraper.
This script implements a hybrid strategy for speed and safety:
1. It processes all entries from scratch in batches of a defined size.
2. It uses multiple threads for fast processing within a batch.
3. It enforces a mandatory cool-down period between batches.
4. It includes a robust retry mechanism with backoff if a block is detected.

Validation: Uses get_final_form + is_allowed_category from wiktionary_validator
to match the exact same validation used for the Wiktionary corpus extraction.
Each word in a definition is lemmatized and checked against Simple English Wiktionary.

RECOMMENDED: Run this locally (not on CESVIMA) for reliable network access,
then upload the result to CESVIMA.
"""

import os
import re
import time
import random
import requests
from tqdm import tqdm
import concurrent.futures
import pathlib
from wiktionary_validator import get_final_form, is_allowed_category

# --- Configuration ---
PROJECT_DIR = pathlib.Path(__file__).parent.resolve()
MASTER_LIST_FILE = PROJECT_DIR / "raw_corpora/extracted_definitions_ground_filtered.txt"
RAW_OUTPUT_FILE = PROJECT_DIR / "raw_corpora/merriam_webster_raw_definitions.txt"
VALIDATED_OUTPUT_FILE = PROJECT_DIR / "raw_corpora/extracted_definitions_merriam_webster.txt"

# --- Batch & Politeness Settings ---
# Reduced parallelism because validation also makes network calls to Wiktionary
BATCH_SIZE = 100
MAX_WORKERS = 4
COOLDOWN_PERIOD = 30 # Seconds to wait between batches

# --- Retry settings for when a block is detected ---
RETRY_ATTEMPTS = 3
RETRY_INITIAL_DELAY = 60 # If we get a 403, wait 1 minute immediately.

def get_mw_definition(html_source):
    """Extracts the definition from the Merriam-Webster HTML source.
    
    Updated April 2026: MW changed their HTML structure.
    Now using <span class="dtText"> which contains the actual definition.
    """
    try:
        # Method 1: Try dtText span (current MW structure as of 2026)
        start_marker = '<span class="dtText">'
        start_index = html_source.find(start_marker)
        if start_index != -1:
            content_start = start_index + len(start_marker)
            end_index = html_source.find('</span>', content_start)
            if end_index != -1:
                definition = html_source[content_start:end_index]
                # Clean up HTML tags and entities
                definition = re.sub(r'<[^>]+>', '', definition)
                definition = definition.replace('&nbsp;', ' ')
                definition = definition.replace('&amp;', '&')
                # Remove leading colon if present
                definition = definition.strip()
                if definition.startswith(':'):
                    definition = definition[1:].strip()
                return definition
        
        # Method 2: Fallback to meta description
        meta_marker = '<meta name="description" content="'
        meta_index = html_source.find(meta_marker)
        if meta_index != -1:
            content_start = meta_index + len(meta_marker)
            end_index = html_source.find('"', content_start)
            if end_index != -1:
                definition = html_source[content_start:end_index]
                # Extract just the definition part (after "The meaning of X is")
                if ' is ' in definition:
                    definition = definition.split(' is ', 1)[1]
                # Remove trailing "How to use..." if present
                if '. How to use' in definition:
                    definition = definition.split('. How to use')[0]
                return definition
        
        return None
    except Exception:
        return None

def process_entry_batch(entry):
    """Processes a single entry with a retry mechanism for blocks."""
    # URL-encode the entry for special characters
    from urllib.parse import quote
    encoded_entry = quote(entry, safe='')
    url = f"https://www.merriam-webster.com/dictionary/{encoded_entry}"
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
    })

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = session.get(url, timeout=20)
            if response.status_code == 200:
                break # Success
            elif response.status_code == 403: # Blocked!
                # This is our emergency brake.
                wait_time = RETRY_INITIAL_DELAY * (attempt + 1)
                time.sleep(wait_time) # Wait for one minute (or more on repeated fails)
                continue
            else: # 404 Not Found, or other error
                return entry, None, None # Give up on this word
        except requests.RequestException:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(20) # Wait 20s on connection error
                continue
            else:
                return entry, None, None # Give up after final attempt

    if response.status_code != 200:
        return entry, None, None # Failed all retries

    raw_definition = get_mw_definition(response.text)
    if not raw_definition: return entry, None, None

    # Validate: same approach as Wiktionary extraction
    # Each word is lemmatized and checked against Simple English Wiktionary
    validated_words = []
    tokens = [re.sub(r'^[^\w\s]+|[^\w\s]+$', '', word) for word in raw_definition.split()]
    for token in tokens:
        if len(token) < 2 or not token.isalpha():
            continue
        # Get lemma form and check if it's a noun/verb/adjective
        final_form = get_final_form(token)
        if is_allowed_category(final_form):
            validated_words.append(token.lower())
    
    # Remove duplicates while preserving order and remove self-references
    seen = set()
    unique_validated_words = []
    for x in validated_words:
        if x != entry and x not in seen:
            seen.add(x)
            unique_validated_words.append(x)
            
    validated_definition = " ".join(unique_validated_words)

    # Empty definition check after cleaning
    if not validated_definition:
        return entry, raw_definition, None

    return entry, raw_definition, validated_definition

def main():
    print("--- Starting BATCH-FOCUSED Merriam-Webster Scraper ---")
    print("NOTE: Using wiktionary_validator for lemmatization + category checking.")
    print("This requires network access to Simple English Wiktionary.")
    print("")
    
    RAW_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    VALIDATED_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load the master list of all entries
    with open(MASTER_LIST_FILE, "r", encoding="utf-8") as f:
        all_entries = [line.split(":", 1)[0].strip() for line in f if ":" in line]
    
    if not all_entries:
        print("Master list is empty. Exiting.")
        return

    # 2. Check for previously processed words to support resuming
    processed_words = set()
    if os.path.exists(RAW_OUTPUT_FILE):
        with open(RAW_OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    processed_words.add(line.split(":", 1)[0].strip())
        print(f"Found {len(processed_words)} words already processed in raw output file. Skipping them.")
    
    entries_to_process = [w for w in all_entries if w not in processed_words]
    
    if not entries_to_process:
        print("All entries have already been processed. Exiting.")
        return

    # 3. Create batches
    batches = [entries_to_process[i:i + BATCH_SIZE] for i in range(0, len(entries_to_process), BATCH_SIZE)]
    total_batches = len(batches)
    print(f"Divided {len(entries_to_process)} remaining entries into {total_batches} batches of up to {BATCH_SIZE} each.")

    # 4. Process each batch
    for i, batch_entries in enumerate(batches):
        print(f"\n--- Processing Batch {i+1} of {total_batches} ---")
        
        # Open files in append mode for this batch
        with open(RAW_OUTPUT_FILE, "a", encoding="utf-8") as raw_f, \
             open(VALIDATED_OUTPUT_FILE, "a", encoding="utf-8") as validated_f:

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_entry = {executor.submit(process_entry_batch, entry): entry for entry in batch_entries}
                
                for future in tqdm(concurrent.futures.as_completed(future_to_entry), total=len(batch_entries), desc=f"Batch {i+1}/{total_batches}"):
                    try:
                        entry, raw_def, validated_def = future.result()
                        if raw_def:
                            raw_f.write(f"{entry}: {raw_def}\n")
                        if validated_def:
                            validated_f.write(f"{entry}: {validated_def}\n")
                    except Exception as e:
                        entry = future_to_entry[future]
                        print(f"\nError on entry '{entry}': {e}")
        
        # 5. Mandatory cool-down period (unless it's the very last batch)
        if i < total_batches - 1:
            print(f"Batch {i+1} complete. Cooling down for {COOLDOWN_PERIOD} seconds...")
            time.sleep(COOLDOWN_PERIOD)

    print("\n--- All batches have been processed. ---")
    print(f"Final results are in '{RAW_OUTPUT_FILE}' and '{VALIDATED_OUTPUT_FILE}'.")
    print("It is recommended to run the 'merge_and_sort.slurm' job to ensure results are correctly ordered.")

if __name__ == "__main__":
    main()
