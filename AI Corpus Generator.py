#!/usr/bin/env python3
# generate_ai_definitions_v5.py
#
# Distribution-controlled generation: each batch gets a fixed length constraint.
# Words are shuffled and assigned to length bins matching the average GF/MW
# distribution, then processed in batches with "EXACTLY N words" instructions.

import pathlib
import json
import time
import re
import sys
import random
from openai import OpenAI, RateLimitError, APIError
from tqdm import tqdm
import tiktoken

# ==============================================================================
#                      Configuration
# ==============================================================================

API_KEY      = "X"
MODEL = "gpt-5.4"
TEMP = 0.0
MAX_RETRY = 3
BACKOFF = 2
BATCH_SIZE = 100
RANDOM_SEED = 42

PROJECT_DIR = pathlib.Path("/Volumes/PortableSSD/Escritorio/Programar/New Semantics Project")
SOURCE_DEFINITIONS_FILE = PROJECT_DIR / "cesvima_codes/project_backup/curated_corpora/common_vocabulary.txt"
OUTPUT_AI_DEFINITIONS_FILE = PROJECT_DIR / "cesvima_codes/project_backup/raw_corpora/extracted_definitions_ai_generated.txt"

# Target length distribution (average of GF and MW)
# Format: (definition_length, percentage)
LENGTH_DISTRIBUTION = [
    (1,  6.0),
    (2,  15.0),
    (3,  20.0),
    (4,  18.0),
    (5,  13.0),
    (6,  9.0),
    (7,  6.0),
    (8,  4.0),
    (9,  3.0),
    (10, 2.0),
    (12, 2.0),
    (15, 1.0),
    (20, 0.7),
    (25, 0.2),
    (30, 0.1),
]

# ==============================================================================
#                      Prompt template
# ==============================================================================

def build_system_prompt(target_length):
    """Build a system prompt with a specific length constraint."""
    return f"""You are a lexicographer creating a dataset for a computational linguistics project. Your task is to provide a definition for each word you receive.
Follow these rules with ABSOLUTE STRICTNESS. There are NO exceptions.
--- RULES ---
1.  **ALLOWED WORD TYPES:** The definition MUST be a space-separated list containing ONLY these three types of words: Nouns, Adjectives, Verbs in their base infinitive form.
2.  **FORBIDDEN WORD TYPES:** You are ABSOLUTELY FORBIDDEN from using: Adverbs (e.g. 'not', 'very', 'often'), Prepositions (e.g. 'by', 'with', 'in', 'of', 'to'), Conjunctions (e.g. 'and', 'or', 'but'), Articles (e.g. 'a', 'the', 'an'), Pronouns (e.g. 'it', 'one', 'that'), or any conjugated verb forms (e.g. 'is', 'has', 'used').
3.  **FORMAT:** You MUST return a single, raw JSON array. Each object must contain the "word" and its "definition".
4.  **DEFINITION LENGTH:** Each definition MUST be EXACTLY {target_length} words long.
Produce only the final JSON array and no other text."""


# ==============================================================================
#                      Helper functions
# ==============================================================================

def assign_lengths(words, seed=RANDOM_SEED):
    """Assign a target length to each word based on the distribution."""
    rng = random.Random(seed)
    shuffled = words[:]
    rng.shuffle(shuffled)

    total_pct = sum(pct for _, pct in LENGTH_DISTRIBUTION)
    n = len(shuffled)

    assignments = []
    idx = 0
    for length, pct in LENGTH_DISTRIBUTION:
        count = round(n * pct / total_pct)
        for _ in range(count):
            if idx < n:
                assignments.append((shuffled[idx], length))
                idx += 1
        if idx >= n:
            break

    fallback_length = 4
    while idx < n:
        assignments.append((shuffled[idx], fallback_length))
        idx += 1

    return assignments


def group_by_length(assignments):
    """Group word assignments by target length."""
    groups = {}
    for word, length in assignments:
        groups.setdefault(length, []).append(word)
    return groups


def estimate_cost(length_groups):
    """Estimate API cost for all batches."""
    print("\n--- Calculating Cost Estimate ---")

    PRICE_INPUT_PER_MILLION = 5.00
    PRICE_OUTPUT_PER_MILLION = 15.00

    try:
        encoding = tiktoken.encoding_for_model(MODEL)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")

    total_input_tokens = 0
    total_output_tokens = 0
    total_batches = 0
    total_words = 0

    for length in sorted(length_groups.keys()):
        words = length_groups[length]
        prompt = build_system_prompt(length)
        prompt_tokens = len(encoding.encode(prompt))

        for i in range(0, len(words), BATCH_SIZE):
            batch = words[i : i + BATCH_SIZE]
            user_content = json.dumps(batch)
            total_input_tokens += prompt_tokens + len(encoding.encode(user_content))
            total_output_tokens += len(batch) * (length + 15)
            total_batches += 1
            total_words += len(batch)

    input_cost = (total_input_tokens / 1_000_000) * PRICE_INPUT_PER_MILLION
    output_cost = (total_output_tokens / 1_000_000) * PRICE_OUTPUT_PER_MILLION
    total_cost = input_cost + output_cost

    print(f"Model: {MODEL}")
    print(f"Words to process: {total_words:,}")
    print(f"Batches to run: {total_batches:,}")
    print()
    print("Length distribution plan:")
    for length in sorted(length_groups.keys()):
        n = len(length_groups[length])
        print(f"  {length:>2} tokens: {n:>5} words ({100*n/total_words:.1f}%)")
    print("-" * 35)
    print(f"Estimated INPUT tokens: {total_input_tokens:,} (~${input_cost:.4f})")
    print(f"Estimated OUTPUT tokens: {total_output_tokens:,} (~${output_cost:.4f})")
    print("-" * 35)
    print(f"ESTIMATED TOTAL COST: ${total_cost:.2f}")
    print("-" * 35)

    return total_cost


def parse_json_array_from_response(text):
    """Safely extracts a JSON array from a string."""
    match = re.search(r'\[.*\]', text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


# ==============================================================================
#                           Main Execution
# ==============================================================================

def main():
    if "YOUR_API_KEY" in API_KEY:
        print("--> FATAL ERROR: Please replace the API key.")
        sys.exit(1)

    if not SOURCE_DEFINITIONS_FILE.exists():
        print(f"--> FATAL ERROR: Source file '{SOURCE_DEFINITIONS_FILE}' not found.")
        sys.exit(1)

    print(f"--> Loading headwords from '{SOURCE_DEFINITIONS_FILE}'...")
    all_headwords = []
    with open(SOURCE_DEFINITIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            word = line.split(":", 1)[0].strip() if ":" in line else line.strip()
            if word:
                all_headwords.append(word)
    print(f"    {len(all_headwords):,} headwords loaded.")

    processed_words = set()
    if OUTPUT_AI_DEFINITIONS_FILE.exists():
        with open(OUTPUT_AI_DEFINITIONS_FILE, "r", encoding="utf-8") as f_read:
            processed_words = {line.split(":", 1)[0].strip() for line in f_read if ":" in line}
        print(f"--> Found {len(processed_words)} words already processed.")

    words_to_process = [w for w in all_headwords if w not in processed_words]
    if not words_to_process:
        print("\n--> All words have already been processed. Nothing to do.")
        return

    print(f"--> Assigning target lengths to {len(words_to_process):,} words...")
    assignments = assign_lengths(words_to_process)
    length_groups = group_by_length(assignments)

    estimate_cost(length_groups)

    proceed = input("\n--> Do you want to proceed with this job? (yes/no): ").lower()
    if proceed != 'yes':
        print("--> Aborting job as requested.")
        return

    print("\n--> Starting API calls...")
    client = OpenAI(api_key=API_KEY)

    batch_list = []
    for length in sorted(length_groups.keys()):
        words = length_groups[length]
        for i in range(0, len(words), BATCH_SIZE):
            batch = words[i : i + BATCH_SIZE]
            batch_list.append((length, batch))

    with open(OUTPUT_AI_DEFINITIONS_FILE, "a", encoding="utf-8") as f_out:
        for length, batch in tqdm(batch_list, desc="Processing Batches"):
            user_content = json.dumps(batch)
            prompt = build_system_prompt(length)
            messages = [{"role": "system", "content": prompt},
                        {"role": "user", "content": user_content}]

            tries = 0
            response_text = None
            while True:
                try:
                    response = client.chat.completions.create(
                        model=MODEL, temperature=TEMP, messages=messages)
                    response_text = response.choices[0].message.content
                    break
                except (RateLimitError, APIError) as e:
                    tries += 1
                    if tries > MAX_RETRY:
                        print(f"\n--> API failure for batch (len={length}, "
                              f"first='{batch[0]}'). Skipping. Error: {e}")
                        break
                    time.sleep(BACKOFF ** tries)

            if response_text is None:
                continue

            results_list = parse_json_array_from_response(response_text)

            if results_list:
                for item in results_list:
                    if isinstance(item, dict) and "word" in item and "definition" in item:
                        f_out.write(f"{item['word']}: {item['definition'].strip()}\n")
                f_out.flush()
            else:
                print(f"\n--> Warning: Could not parse JSON for batch "
                      f"(len={length}, first='{batch[0]}'). Skipping.")

    print("\n--> Process finished successfully.")
    print(f"--> AI-generated lexicon saved to '{OUTPUT_AI_DEFINITIONS_FILE}'")


if __name__ == '__main__':
    main()
