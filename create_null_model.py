# create_null_model.py

import random
from tqdm import tqdm
import os

# --- Configuration ---
INPUT_DEFINITIONS_FILE = "extracted_definitions.txt"
OUTPUT_NULL_MODEL_FILE = "null_model_definitions.txt"

def read_definitions(filepath):
    """Reads definitions and returns a dictionary."""
    if not os.path.exists(filepath):
        print(f"ERROR: Input file not found at '{filepath}'")
        return None
        
    definitions = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                head, def_text = line.split(":", 1)
                head = head.strip().lower()
                tokens = def_text.strip().split()
                definitions[head] = tokens
    return definitions

def create_and_save_null_model(definitions, output_filepath):
    """
    Creates a randomly rewired dictionary and saves it to a file.
    The rewiring preserves the out-degree of each node (word).
    """
    print("Starting creation of the Null Model (Randomly Rewired Dictionary)...")
    
    null_definitions = {}
    all_words = list(definitions.keys()) # The pool of all possible words to choose from
    
    # Set a fixed seed for reproducibility. Anyone running this script will get
    # the exact same null model. This is crucial for scientific replication.
    random.seed(42)

    for word, def_tokens in tqdm(definitions.items(), desc="Rewiring definitions"):
        # Preserve the out-degree (the number of words in the definition)
        num_tokens = len(def_tokens)
        
        # Replace the original definition with a random sample of words
        # drawn from the entire vocabulary. `random.choices` allows for replacement,
        # which is a standard assumption in this type of null model.
        if num_tokens > 0:
            null_definitions[word] = random.choices(all_words, k=num_tokens)
        else:
            null_definitions[word] = [] # Preserve empty definitions

    print(f"\nNull model created. Saving to '{output_filepath}'...")
    
    # Save the new null model to the output file
    with open(output_filepath, "w", encoding="utf-8") as f:
        for head, tokens in null_definitions.items():
            f.write(f"{head}: {' '.join(tokens)}\n")
            
    print("Save complete.")


if __name__ == '__main__':
    print(f"Reading original definitions from '{INPUT_DEFINITIONS_FILE}'...")
    original_definitions = read_definitions(INPUT_DEFINITIONS_FILE)
    
    if original_definitions:
        create_and_save_null_model(original_definitions, OUTPUT_NULL_MODEL_FILE)
        print("\nProcess finished successfully.")
