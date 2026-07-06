#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper module containing functions to validate words using the Simple English Wiktionary.
This includes finding a word's lemma and checking its grammatical category.
Functions are intended to be imported by other scripts.
"""

import re
import requests
import time
import random
from bs4 import BeautifulSoup

# HTTP Headers to avoid 403 blocks
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

# Caches to avoid redundant network calls.
final_form_cache = {}
allowed_category_cache = {}

def make_request_with_retry(url, max_retries=5):
    """Helper function to handle rate limits gracefully."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            if response.status_code == 429:
                sleep_time = (2 ** attempt) + random.uniform(1.0, 3.0)
                time.sleep(sleep_time)
                continue
            return response
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(1.0)
    return None

def extract_main_definition_html(element):
    """
    Extracts the HTML of the main definition from a Wiktionary list item element.
    Stops at common section markers like <dl>, <ul>, or synonym/antonym lists.
    """
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
        return False

def is_strict_lowercase_word(word):
    """Check if the entry is a single lowercase English word (no spaces, hyphens, caps)."""
    return bool(re.match(r'^[a-z]+$', word))