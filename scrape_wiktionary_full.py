#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Wiktionary Scraper - Single Script

This script performs the full extraction pipeline:
1. Fetches all entries from Simple English Wiktionary's "All Pages"
2. Extracts definitions for each entry (parallel processing)
3. Saves the final cleaned definitions

Usage:
    python scrape_wiktionary_full.py [--output OUTPUT_FILE] [--workers N]

Output:
    - wiktionary_entries.txt (intermediate - word list)
    - extracted_definitions.txt (final - Ground Filtered corpus)
"""

import os
import re
import time
import random
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
import concurrent.futures

# ==============================================================================
#                           CONFIGURATION
# ==============================================================================

# Delay between page requests (seconds) for entry fetching
ENTRY_FETCH_DELAY = 1.0

# Number of parallel workers for definition extraction
# Reduced to avoid rate limiting issues with Wiktionary
DEFAULT_WORKERS = 2

# Debug mode
DEBUG = False

# HTTP Headers to avoid 403 blocks
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

# ==============================================================================
#                           CACHES
# ==============================================================================

final_form_cache = {}
allowed_category_cache = {}

# ==============================================================================
#                      PART 1: FETCH ALL ENTRIES
# ==============================================================================

def fetch_all_entries(start_url="https://simple.wiktionary.org/w/index.php?title=Special:AllPages&from=%21"):
    """
    Fetches all entries from Simple Wiktionary's All Pages list.
    Returns a sorted list of unique entries.
    """
    print("="*60)
    print("  PART 1: Fetching Wiktionary Entry List")
    print("="*60)
    
    all_entries = set()
    current_url = start_url
    page_counter = 0
    
    while current_url:
        page_counter += 1
        print(f"  Page {page_counter}: fetching...", end=" ", flush=True)
        
        try:
            response = requests.get(current_url, headers=HEADERS, timeout=30)
            if response.status_code != 200:
                print(f"FAILED (status {response.status_code})")
                break
        except requests.RequestException as e:
            print(f"ERROR: {e}")
            break
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract entries from the page
        body = soup.find("div", id="mw-allpages-body")
        if not body:
            body = soup
        
        entries_on_page = []
        for li in body.find_all("li"):
            a = li.find("a")
            if a and a.get("href", "").startswith("/wiki/"):
                entry = a.get_text(strip=True)
                entries_on_page.append(entry)
                all_entries.add(entry)
        
        print(f"found {len(entries_on_page)} entries (total: {len(all_entries)})")
        
        # Find next page link
        nav_div = soup.find("div", class_="mw-allpages-nav")
        next_url = None
        if nav_div:
            next_link = nav_div.find("a", string=lambda t: t and t.startswith("Next page"))
            if next_link and next_link.get("href"):
                next_url = urljoin(current_url, next_link["href"])
        
        if next_url:
            current_url = next_url
            time.sleep(ENTRY_FETCH_DELAY)
        else:
            print("\n  No more pages. Entry fetching complete.")
            break
    
    sorted_entries = sorted(all_entries)
    print(f"\n  Total unique entries: {len(sorted_entries)}")
    
    return sorted_entries


# ==============================================================================
#                   PART 2: EXTRACT DEFINITIONS
# ==============================================================================

def debug_log(message):
    if DEBUG:
        print(message)

def extract_main_definition_html(element):
    """Extract HTML up to definition end markers."""
    html_str = str(element)
    candidates = []
    
    for marker in ["<dl", "<ul"]:
        idx = html_str.find(marker)
        if idx != -1:
            candidates.append(idx)
    
    lower_html = html_str.lower()
    for marker in ["synonyms:", "antonyms:"]:
        idx = lower_html.find(marker)
        if idx != -1:
            candidates.append(idx)
    
    if candidates:
        return html_str[:min(candidates)]
    return html_str


def make_request_with_retry(url, max_retries=5):
    """Helper function to handle rate limits gracefully."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            if response.status_code == 429:
                # Rate limited. Sleep longer and retry.
                sleep_time = (2 ** attempt) + random.uniform(1.0, 3.0)
                time.sleep(sleep_time)
                continue
            return response
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(1.0)
    return None

def get_final_form(word):
    """Get the lemma form of a word."""
    if word in final_form_cache:
        return final_form_cache[word]
    
    time.sleep(random.uniform(0.1, 0.3))
    url = f"https://simple.wiktionary.org/wiki/{word}"
    
    try:
        response = make_request_with_retry(url)
        if not response or response.status_code != 200:
            final_form_cache[word] = word
            return word
        
        soup = BeautifulSoup(response.text, "html.parser")
        ol = soup.find("ol")
        if not ol:
            final_form_cache[word] = word
            return word
        
        li = ol.find("li")
        element = li if li else ol
        main_html = extract_main_definition_html(element)
        cleaned_html_str = re.sub(r'\([^)]*\)', '', main_html)
        cleaned_soup = BeautifulSoup(cleaned_html_str, "html.parser")
        plain_text = cleaned_soup.get_text(separator=" ", strip=True)
        
        running_length = 0
        stop_limit = len(plain_text)
        
        for elem in cleaned_soup.recursiveChildGenerator():
            if isinstance(elem, str):
                running_length += len(elem)
                if running_length >= stop_limit:
                    break
            elif elem.name == "i":
                if running_length < stop_limit:
                    candidate = elem.get_text(strip=True)
                    if "?" not in candidate:
                        final_form_cache[word] = candidate
                        return candidate
        
        final_form_cache[word] = word
        return word
    except Exception:
        # DON'T cache failures - might be temporary network issue
        return word


def is_allowed_category(word):
    """Check if word belongs to allowed category (noun, verb, adj, proper noun)."""
    if word in allowed_category_cache:
        return allowed_category_cache[word]
    
    time.sleep(random.uniform(0.1, 0.3))
    url = f"https://simple.wiktionary.org/wiki/{word}"
    
    try:
        response = make_request_with_retry(url)
        if not response or response.status_code != 200:
            # Don't cache 404s as False permanently, just return False
            return False
        
        html = response.text.lower()
        
        # Check for numbers
        number_markers = ['title="number"', 'title="ordinal number"', 'title="cardinal number"']
        if any(m in html for m in number_markers):
            allowed_category_cache[word] = False
            return False

        # TOC markers (lowercased from toc-Noun, toc-Verb, etc.)
        # NOTE: Proper nouns excluded (toc-proper_noun, toc-proper noun)
        toc_markers = ["toc-verb", "toc-adjective", "toc-noun"]
        
        # Check if it's a proper noun - if so, reject it
        proper_noun_markers = ["toc-proper_noun", "toc-proper noun", "/wiki/category:proper_nouns", 'id="proper_noun"', 'category:proper nouns']
        if any(m in html for m in proper_noun_markers):
            allowed_category_cache[word] = False
            return False
        
        # Category markers - updated for current Wiktionary format
        # NOTE: Proper nouns excluded from category markers
        cat_markers = ['/wiki/category:nouns', '/wiki/category:verbs', 
                       '/wiki/category:adjectives',
                       'category:nouns', 'category:verbs', 'category:adjectives']
        
        if any(m in html for m in toc_markers) or any(cm in html for cm in cat_markers):
            allowed_category_cache[word] = True
            return True
        
        allowed_category_cache[word] = False
        return False
    except Exception:
        # DON'T cache failures - might be temporary network issue
        # Return False but don't cache, so it can be retried
        return False


def extract_definition(word):
    """Extract and clean the definition for a word."""
    url = f"https://simple.wiktionary.org/wiki/{word}"
    
    try:
        response = make_request_with_retry(url)
    except Exception:
        return ""
    
    if not response or response.status_code != 200:
        return ""
    
    soup = BeautifulSoup(response.text, "html.parser")
    ol = soup.find("ol")
    if not ol:
        return ""
    
    li = ol.find("li")
    element = li if li else ol
    
    # Remove math elements
    for span in element.find_all("span", class_="mwe-math-mathml-inline"):
        span.decompose()
    for img in element.find_all("img", class_="mwe-math-fallback-image-inline"):
        alt_text = img.get("alt", "")
        if alt_text.startswith("{\\displaystyle ") and alt_text.endswith("}"):
            clean_text = alt_text[len("{\\displaystyle "):-1]
        else:
            clean_text = alt_text
        img.replace_with(clean_text)
    
    main_html = extract_main_definition_html(element)
    cleaned_html_str = re.sub(r"\([^)]*\)", "", main_html)
    cleaned_soup = BeautifulSoup(cleaned_html_str, "html.parser")
    
    # Check for "form of X" patterns where italic indicates the lemma
    # Examples: "The past tense of hanker." or "Plural of dog."
    # In these cases, the italic word is the REAL entry, not this one
    full_text = cleaned_soup.get_text(separator=" ", strip=True)
    lower_text = full_text.lower()
    
    # Patterns that indicate this is just a grammatical form, not a real entry
    form_patterns = [
        'past tense of', 'past participle of', 'present participle of',
        'plural of', 'singular of', 'comparative of', 'superlative of',
        'third person singular of', 'third-person singular of',
        'feminine of', 'masculine of',
        'alternative spelling of', 'archaic spelling of', 'misspelling of',
        'singular form of', 'plural form of', 'past form of',
        'form of'  # Generic catch-all for "X form of Y" patterns
    ]
    
    is_form_redirect = any(pattern in lower_text for pattern in form_patterns)
    
    if is_form_redirect:
        # This entry is just a grammatical form - skip it
        # The real definition is under the lemma (the italic word)
        return ""
    cleaned_text = re.sub(r"[^\w\s.]", "", full_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    tokens = cleaned_text.split()
    
    # Filter tokens
    prelim_filtered = []
    for token in tokens:
        token = token.strip(".")
        if len(token) <= 1 or any(ch.isdigit() for ch in token):
            continue
        prelim_filtered.append(token)
    
    # Validate words
    allowed_words = []
    for token in prelim_filtered:
        final = get_final_form(token)
        if is_allowed_category(final):
            allowed_words.append(final)
    
    # Remove duplicates and disallowed words
    disallowed = {"be", "to", "if", word.lower()}
    unique_allowed = []
    seen = set()
    for token in allowed_words:
        lw = token.lower()
        if lw in disallowed or lw in seen:
            continue
        seen.add(lw)
        unique_allowed.append(lw)
    
    return " ".join(unique_allowed)


def is_strict_lowercase_word(word):
    """Check if the entry is a single lowercase English word (no spaces, hyphens, caps)."""
    return bool(re.match(r'^[a-z]+$', word))

def process_entry(entry):
    """Process a single entry. Returns (entry, definition) or (entry, None)."""
    try:
        if not is_strict_lowercase_word(entry):
            return entry, None, "invalid_format"
            
        allowed = is_allowed_category(entry)
        if not allowed:
            return entry, None, "not_allowed_category"
        
        definition = extract_definition(entry)
        if not definition:
            return entry, None, "empty_definition"
            
        if re.search(r'\bsuffix\b', definition.lower()):
            return entry, None, "is_suffix"
        
        return entry, definition, "success"
    except Exception as e:
        return entry, None, f"error: {e}"


def extract_all_definitions(entries, num_workers=DEFAULT_WORKERS):
    """Extract definitions for all entries using parallel processing."""
    print("\n" + "="*60)
    print("  PART 2: Extracting Definitions")
    print("="*60)
    print(f"  Entries to process: {len(entries)}")
    print(f"  Parallel workers: {num_workers}")
    print("")
    
    extracted_definitions = {}
    discard_reasons = {
        "invalid_format": 0,
        "not_allowed_category": 0,
        "empty_definition": 0,
        "is_suffix": 0,
        "error": 0,
        "success": 0
    }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_entry = {executor.submit(process_entry, entry): entry for entry in entries}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_entry), 
                          total=len(entries), desc="  Extracting"):
            entry = future_to_entry[future]
            try:
                entry_result, definition_result, reason = future.result()
                if reason == "success":
                    extracted_definitions[entry_result] = definition_result
                    discard_reasons["success"] += 1
                elif reason == "invalid_format":
                    discard_reasons["invalid_format"] += 1
                elif reason == "not_allowed_category":
                    discard_reasons["not_allowed_category"] += 1
                elif reason == "empty_definition":
                    discard_reasons["empty_definition"] += 1
                elif reason == "is_suffix":
                    discard_reasons["is_suffix"] += 1
                else:
                    discard_reasons["error"] += 1
                    if discard_reasons["error"] <= 5:
                        print(f"\n  Error on '{entry}': {reason}")
            except Exception as e:
                print(f"\n  Exception processing '{entry}': {e}")
                discard_reasons["error"] += 1
    
    print(f"\n  Results breakdown:")
    print(f"    Success:                 {discard_reasons['success']}")
    print(f"    Invalid format/caps/multi: {discard_reasons['invalid_format']}")
    print(f"    Not allowed category:    {discard_reasons['not_allowed_category']}")
    print(f"    Empty definition:        {discard_reasons['empty_definition']}")
    print(f"    Is suffix:               {discard_reasons['is_suffix']}")
    print(f"    Errors:                  {discard_reasons['error']}")
    
    return extracted_definitions


# ==============================================================================
#                              MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Complete Wiktionary scraper")
    parser.add_argument("--output", default="extracted_definitions.txt",
                        help="Output file for definitions (default: extracted_definitions.txt)")
    parser.add_argument("--entries-file", default="wiktionary_entries.txt",
                        help="File to save/load entries list (default: wiktionary_entries.txt)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip fetching entries, use existing entries file")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    args = parser.parse_args()
    
    global DEBUG
    DEBUG = args.debug
    
    print("\n" + "="*60)
    print("  WIKTIONARY SCRAPER - COMPLETE PIPELINE")
    print("="*60)
    print(f"  Output file: {args.output}")
    print(f"  Workers: {args.workers}")
    print("="*60)
    
    # PART 1: Get entries
    if args.skip_fetch and os.path.exists(args.entries_file):
        print(f"\n  Loading existing entries from '{args.entries_file}'...")
        with open(args.entries_file, "r", encoding="utf-8") as f:
            entries = sorted(set(line.strip() for line in f if line.strip()))
        print(f"  Loaded {len(entries)} entries.")
    else:
        entries = fetch_all_entries()
        
        # Save entries list
        with open(args.entries_file, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(f"{entry}\n")
        print(f"\n  Entries saved to '{args.entries_file}'")
    
    # PART 2: Extract definitions
    definitions = extract_all_definitions(entries, num_workers=args.workers)
    
    # Save definitions
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output, "w", encoding="utf-8") as f:
        for word in sorted(definitions.keys()):
            f.write(f"{word}: {definitions[word]}\n")
    
    print("\n" + "="*60)
    print("  COMPLETE")
    print("="*60)
    print(f"  Definitions saved to: {args.output}")
    print(f"  Total words: {len(definitions)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
