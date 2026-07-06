#!/bin/bash
# ==============================================================================
#           ONTOLOGICAL DIFFERENTIATION PIPELINE - MASTER ORCHESTRATOR
# ==============================================================================
#
# This script automates the complete OD analysis pipeline:
#   Stage 0: Clean definitions (remove dangling words)
#   Stage 1: Run SOD pairs analysis (all word pairs)
#   Stage 2: Run single GOD analysis (individual words)
#   Stage 3: Aggregate results
#
# Usage:
#   ./master_run.sh <definitions_file> [options]
#
# Example:
#   ./master_run.sh extracted_definitions_ai_generated.txt --jobs 40 --cores 16
#
# ==============================================================================

set -e
set -o pipefail

# ==============================================================================
#                           CONFIGURATION
# ==============================================================================

# Default values (can be overridden via command line)
NUM_JOBS=${NUM_JOBS:-40}           # Number of parallel array jobs
CORES_PER_JOB=${CORES_PER_JOB:-16} # CPU cores per job
SLURM_PARTITION=${SLURM_PARTITION:-"standard"}
SLURM_TIME=${SLURM_TIME:-"100:00:00"}
SLURM_MEM=${SLURM_MEM:-"24G"}

# Corpus generation defaults
RANDOM_REMOVAL_PCT=${RANDOM_REMOVAL_PCT:-20}
TARGETED_REMOVAL_PCT=${TARGETED_REMOVAL_PCT:-20}
CORPUS_SEED=${CORPUS_SEED:-42}

# Derived paths (set after parsing arguments)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=""
LOG_DIR=""

# ==============================================================================
#                           HELPER FUNCTIONS
# ==============================================================================

print_header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║  $1"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_stage() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STAGE $1: $2"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

cleanup_on_exit() {
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [ERROR] Pipeline failed with exit code: $EXIT_CODE"
        echo "  Check logs in: ${LOG_DIR:-'(not set)'}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    fi
    exit $EXIT_CODE
}

trap cleanup_on_exit EXIT INT TERM

usage() {
    echo "Usage: $0 <definitions_file> [options]"
    echo ""
    echo "Arguments:"
    echo "  definitions_file    Input definitions file (e.g., extracted_definitions_ground_filtered.txt)"
    echo ""
    echo "Pipeline Options:"
    echo "  --jobs N            Number of parallel SLURM array jobs (default: $NUM_JOBS)"
    echo "  --cores N           CPU cores per job (default: $CORES_PER_JOB)"
    echo "  --partition NAME    SLURM partition (default: $SLURM_PARTITION)"
    echo "  --time HH:MM:SS     SLURM time limit (default: $SLURM_TIME)"
    echo "  --mem SIZE          Memory per job (default: $SLURM_MEM)"
    echo "  --skip-clean        Skip Stage 0 (use pre-cleaned definitions)"
    echo "  --start-idx N       Start index for vocabulary (default: 1)"
    echo "  --end-idx N         End index for vocabulary (default: all words)"
    echo ""
    echo "Corpus Generation Options:"
    echo "  --generate-corpora  Generate derived corpora (Random, Targeted, Null) from base"
    echo "  --random-pct N      Percentage for Random Removal (default: $RANDOM_REMOVAL_PCT)"
    echo "  --targeted-pct N    Percentage for Targeted Removal (default: $TARGETED_REMOVAL_PCT)"
    echo "  --corpus-seed N     Random seed for corpus generation (default: $CORPUS_SEED)"
    echo "  --run-all-corpora   Run OD analysis on all generated corpora (not just base)"
    echo ""
    echo "Web Scraping Options (SLOW - requires network):"
    echo "  --scrape-wiktionary Scrape Simple Wiktionary for Ground Filtered corpus"
    echo "  --scrape-merriam    Scrape Merriam-Webster for Complex corpus"
    echo "  --scrape-all        Run both scrapers (equivalent to --scrape-wiktionary --scrape-merriam)"
    echo ""
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Run pipeline on a single corpus:"
    echo "  $0 extracted_definitions_ai_generated.txt --jobs 40"
    echo ""
    echo "  # Generate derived corpora from Ground Filtered and run on all:"
    echo "  $0 extracted_definitions_ground_filtered.txt --generate-corpora --run-all-corpora"
    echo ""
    echo "  # Full regeneration: scrape, generate corpora, run OD on all:"
    echo "  $0 --scrape-all --generate-corpora --run-all-corpora --jobs 40"
    echo ""
    exit 1
}

# ==============================================================================
#                           PARSE ARGUMENTS
# ==============================================================================

if [ $# -lt 1 ]; then
    usage
fi

INPUT_DEFINITIONS=""
SKIP_CLEAN=false
START_IDX=1
END_IDX=""
GENERATE_CORPORA=false
RUN_ALL_CORPORA=false
SCRAPE_WIKTIONARY=false
SCRAPE_MERRIAM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        --cores)
            CORES_PER_JOB="$2"
            shift 2
            ;;
        --partition)
            SLURM_PARTITION="$2"
            shift 2
            ;;
        --time)
            SLURM_TIME="$2"
            shift 2
            ;;
        --mem)
            SLURM_MEM="$2"
            shift 2
            ;;
        --skip-clean)
            SKIP_CLEAN=true
            shift
            ;;
        --start-idx)
            START_IDX="$2"
            shift 2
            ;;
        --end-idx)
            END_IDX="$2"
            shift 2
            ;;
        --generate-corpora)
            GENERATE_CORPORA=true
            shift
            ;;
        --random-pct)
            RANDOM_REMOVAL_PCT="$2"
            shift 2
            ;;
        --targeted-pct)
            TARGETED_REMOVAL_PCT="$2"
            shift 2
            ;;
        --corpus-seed)
            CORPUS_SEED="$2"
            shift 2
            ;;
        --run-all-corpora)
            RUN_ALL_CORPORA=true
            shift
            ;;
        --scrape-wiktionary)
            SCRAPE_WIKTIONARY=true
            shift
            ;;
        --scrape-merriam)
            SCRAPE_MERRIAM=true
            shift
            ;;
        --scrape-all)
            SCRAPE_WIKTIONARY=true
            SCRAPE_MERRIAM=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            if [ -z "$INPUT_DEFINITIONS" ]; then
                INPUT_DEFINITIONS="$1"
            else
                echo "Unexpected argument: $1"
                usage
            fi
            shift
            ;;
    esac
done

# If scraping, input file is optional (will be created by scraper)
if [ "$SCRAPE_WIKTIONARY" = true ]; then
    # Input will be generated by scraper
    if [ -z "$INPUT_DEFINITIONS" ]; then
        INPUT_DEFINITIONS="${SCRIPT_DIR}/extracted_definitions.txt"
        echo "[INFO] Will use scraped Wiktionary as base: $INPUT_DEFINITIONS"
    fi
elif [ -z "$INPUT_DEFINITIONS" ]; then
    echo "Error: No definitions file specified."
    echo "       Either provide a definitions file or use --scrape-wiktionary"
    usage
elif [ ! -f "$INPUT_DEFINITIONS" ]; then
    echo "Error: Definitions file not found: $INPUT_DEFINITIONS"
    exit 1
fi

# ==============================================================================
#                           SETUP ENVIRONMENT
# ==============================================================================

print_header "ONTOLOGICAL DIFFERENTIATION PIPELINE"

echo "Configuration:"
echo "  Input file:        $INPUT_DEFINITIONS"
echo "  Parallel jobs:     $NUM_JOBS"
echo "  Cores per job:     $CORES_PER_JOB"
echo "  SLURM partition:   $SLURM_PARTITION"
echo "  Time limit:        $SLURM_TIME"
echo "  Memory:            $SLURM_MEM"
echo "  Skip cleaning:     $SKIP_CLEAN"
echo "  Generate corpora:  $GENERATE_CORPORA"
echo "  Run all corpora:   $RUN_ALL_CORPORA"
echo "  Scrape Wiktionary: $SCRAPE_WIKTIONARY"
echo "  Scrape Merriam:    $SCRAPE_MERRIAM"
if [ -n "$END_IDX" ]; then
    echo "  Vocab range:       $START_IDX to $END_IDX"
else
    echo "  Vocab range:       $START_IDX to end"
fi

# Create output directories
BASENAME=$(basename "$INPUT_DEFINITIONS" .txt)
OUTPUT_DIR="${SCRIPT_DIR}/results_${BASENAME}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo ""
echo "Output directories:"
echo "  Results: $OUTPUT_DIR"
echo "  Logs:    $LOG_DIR"

# Load modules
echo ""
echo "Loading modules..."
module --force purge > /dev/null 2>&1 || true
module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || {
    echo "Warning: Could not load Python module. Using system Python."
}

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Export variables for SLURM jobs
export INPUT_DEFINITIONS
export OUTPUT_DIR
export LOG_DIR
export NUM_JOBS
export CORES_PER_JOB
export START_IDX
export END_IDX

# ==============================================================================
#                   STAGE -3: SCRAPE WIKTIONARY (OPTIONAL)
# ==============================================================================

if [ "$SCRAPE_WIKTIONARY" = true ]; then
    print_stage "-3" "Scrape Simple Wiktionary"
    
    echo "WARNING: This stage scrapes the web and may take several hours!"
    echo ""
    
    # Step 1: Get all Wiktionary entries
    ENTRIES_FILE="${SCRIPT_DIR}/wiktionary_entries.txt"
    
    if [ -f "$ENTRIES_FILE" ] && [ -s "$ENTRIES_FILE" ]; then
        echo "Using existing entries file: $ENTRIES_FILE"
        echo "  ($(wc -l < "$ENTRIES_FILE") entries)"
    else
        echo "Step 1: Fetching Wiktionary entry list..."
        python3 "${SCRIPT_DIR}/wiktionary_entries.py"
        
        if [ ! -f "$ENTRIES_FILE" ]; then
            echo "ERROR: Failed to create wiktionary_entries.txt"
            exit 1
        fi
    fi
    
    # Step 2: Extract definitions
    echo ""
    echo "Step 2: Extracting definitions (this may take hours)..."
    cd "$SCRIPT_DIR"
    python3 "${SCRIPT_DIR}/wiktionary_definitions.py"
    cd - > /dev/null
    
    INPUT_DEFINITIONS="${SCRIPT_DIR}/extracted_definitions.txt"
    
    if [ ! -f "$INPUT_DEFINITIONS" ]; then
        echo "ERROR: Failed to create extracted_definitions.txt"
        exit 1
    fi
    
    echo ""
    echo "Wiktionary scraping complete."
    echo "  Output: $INPUT_DEFINITIONS"
    echo "  Words:  $(wc -l < "$INPUT_DEFINITIONS")"
else
    echo ""
    echo "[INFO] Wiktionary scraping skipped (use --scrape-wiktionary to enable)"
fi

# ==============================================================================
#                   STAGE -2: SCRAPE MERRIAM-WEBSTER (OPTIONAL)
# ==============================================================================

if [ "$SCRAPE_MERRIAM" = true ]; then
    print_stage "-2" "Scrape Merriam-Webster"
    
    echo "WARNING: This stage scrapes the web with rate limiting."
    echo "         Expect this to take many hours!"
    echo ""
    
    # MW scraper needs extracted_definitions.txt as input
    if [ ! -f "${SCRIPT_DIR}/extracted_definitions.txt" ]; then
        echo "ERROR: Merriam-Webster scraper requires extracted_definitions.txt"
        echo "       Run with --scrape-wiktionary first, or provide the file"
        exit 1
    fi
    
    cd "$SCRIPT_DIR"
    python3 "${SCRIPT_DIR}/merriam_webster_scraper_batch.py"
    cd - > /dev/null
    
    MW_OUTPUT="${SCRIPT_DIR}/merriam_webster_validated_definitions.txt"
    
    if [ -f "$MW_OUTPUT" ]; then
        echo ""
        echo "Merriam-Webster scraping complete."
        echo "  Output: $MW_OUTPUT"
        echo "  Words:  $(wc -l < "$MW_OUTPUT")"
    else
        echo "WARNING: Merriam-Webster scraper did not produce output."
    fi
else
    echo ""
    echo "[INFO] Merriam-Webster scraping skipped (use --scrape-merriam to enable)"
fi

# ==============================================================================
#                   STAGE -1: GENERATE DERIVED CORPORA (OPTIONAL)
# ==============================================================================

if [ "$GENERATE_CORPORA" = true ]; then
    print_stage "-1" "Generate Derived Corpora"
    
    CORPORA_DIR="${SCRIPT_DIR}/generated_corpora"
    mkdir -p "$CORPORA_DIR"
    
    echo "Generating derived corpora from: $INPUT_DEFINITIONS"
    echo "  Output directory: $CORPORA_DIR"
    echo "  Random Removal:   ${RANDOM_REMOVAL_PCT}%"
    echo "  Targeted Removal: ${TARGETED_REMOVAL_PCT}%"
    echo "  Seed:             $CORPUS_SEED"
    echo ""
    
    python3 "${SCRIPT_DIR}/generate_corpora.py" \
        --base "$INPUT_DEFINITIONS" \
        --output-dir "$CORPORA_DIR" \
        --random-pct "$RANDOM_REMOVAL_PCT" \
        --targeted-pct "$TARGETED_REMOVAL_PCT" \
        --seed "$CORPUS_SEED"
    
    CORPUS_EXIT_CODE=$?
    
    if [ $CORPUS_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "ERROR: Corpus generation failed!"
        exit 1
    fi
    
    echo ""
    echo "Derived corpora created:"
    ls -la "$CORPORA_DIR"
    echo ""
    
    # Store list of corpora for --run-all-corpora
    CORPORA_LIST=(
        "$INPUT_DEFINITIONS"
        "${CORPORA_DIR}/extracted_definitions_random_removal.txt"
        "${CORPORA_DIR}/extracted_definitions_targeted_removal.txt"
        "${CORPORA_DIR}/extracted_definitions_null_model.txt"
    )
    
    # Add Merriam-Webster corpus if it exists
    MW_CORPUS="${SCRIPT_DIR}/merriam_webster_validated_definitions.txt"
    if [ -f "$MW_CORPUS" ]; then
        CORPORA_LIST+=("$MW_CORPUS")
        echo "  Including Merriam-Webster corpus in analysis"
    fi
else
    echo ""
    echo "[INFO] Corpus generation skipped (use --generate-corpora to enable)"
    CORPORA_LIST=("$INPUT_DEFINITIONS")
    
    # Still add MW corpus if available
    MW_CORPUS="${SCRIPT_DIR}/merriam_webster_validated_definitions.txt"
    if [ -f "$MW_CORPUS" ] && [ "$RUN_ALL_CORPORA" = true ]; then
        CORPORA_LIST+=("$MW_CORPUS")
    fi
fi

# ==============================================================================
#                    MAIN PIPELINE FUNCTION (for each corpus)
# ==============================================================================

run_pipeline_for_corpus() {
    local CORPUS_FILE="$1"
    local CORPUS_NAME=$(basename "$CORPUS_FILE" .txt)
    local CORPUS_OUTPUT_DIR="${SCRIPT_DIR}/results_${CORPUS_NAME}"
    local CORPUS_LOG_DIR="${CORPUS_OUTPUT_DIR}/logs"
    
    print_header "Processing: ${CORPUS_NAME}"
    
    mkdir -p "$CORPUS_OUTPUT_DIR" "$CORPUS_LOG_DIR"
    
    echo "  Results: $CORPUS_OUTPUT_DIR"
    echo "  Logs:    $CORPUS_LOG_DIR"
    echo ""
    
    # Export for use in stages
    local CURRENT_OUTPUT_DIR="$CORPUS_OUTPUT_DIR"
    local CURRENT_LOG_DIR="$CORPUS_LOG_DIR"
    local CURRENT_INPUT="$CORPUS_FILE"
    local CLEANED_DEFINITIONS=""

# ==============================================================================
#                           STAGE 0: CLEAN DEFINITIONS
# ==============================================================================

if [ "$SKIP_CLEAN" = true ]; then
    print_stage "0" "Clean Definitions (SKIPPED)"
    CLEANED_DEFINITIONS="$CURRENT_INPUT"
    echo "Using input file directly: $CLEANED_DEFINITIONS"
else
    print_stage "0" "Clean Definitions"
    
    CLEANED_DEFINITIONS="${CURRENT_OUTPUT_DIR}/extracted_definitions_cleaned.txt"
    
    echo "Running definition cleaner..."
    echo "  Input:  $CURRENT_INPUT"
    echo "  Output: $CLEANED_DEFINITIONS"
    
    # Create a temporary cleaner script that uses our output paths
    python3 << EOF
import os
from tqdm import tqdm
from collections import Counter

INPUT_FILE = "${CURRENT_INPUT}"
OUTPUT_FILE = "${CLEANED_DEFINITIONS}"
DANGLING_LOG = "${CURRENT_OUTPUT_DIR}/dangling_words_log.txt"
REMOVED_LOG = "${CURRENT_OUTPUT_DIR}/removed_headwords_log.txt"

print(f"Loading definitions from '{INPUT_FILE}'...")
definitions = {}
original_order = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or ":" not in line:
            continue
        head, def_text = line.split(":", 1)
        head = head.strip().lower()
        tokens = def_text.strip().split()
        if head not in definitions:
            original_order.append(head)
        definitions[head] = tokens

initial_count = len(definitions)
print(f"Loaded {initial_count} definitions.")

all_dangling = Counter()
all_removed = set()

print("Starting iterative pruning...")
iteration = 0
while True:
    iteration += 1
    valid_headwords = set(definitions.keys())
    
    # Prune dangling words
    for headword in list(definitions.keys()):
        dangling = [t for t in definitions[headword] if t not in valid_headwords]
        all_dangling.update(dangling)
        definitions[headword] = [t for t in definitions[headword] if t in valid_headwords]
    
    # Remove empty definitions
    to_remove = [h for h, d in definitions.items() if not d]
    if not to_remove:
        print(f"  Iteration {iteration}: Stable. No more empty definitions.")
        break
    
    print(f"  Iteration {iteration}: Removing {len(to_remove)} empty definitions.")
    all_removed.update(to_remove)
    for h in to_remove:
        del definitions[h]

final_count = len(definitions)
print(f"\nCleaning complete:")
print(f"  Initial: {initial_count} words")
print(f"  Final:   {final_count} words")
print(f"  Removed: {initial_count - final_count} words")

# Save outputs
with open(DANGLING_LOG, "w", encoding="utf-8") as f:
    f.write("# Dangling words (count)\n")
    for word, count in all_dangling.most_common():
        f.write(f"{word}, {count}\n")

with open(REMOVED_LOG, "w", encoding="utf-8") as f:
    f.write("# Removed headwords\n")
    for word in sorted(all_removed):
        f.write(f"{word}\n")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for head in original_order:
        if head in definitions:
            f.write(f"{head}: {' '.join(definitions[head])}\n")

print(f"\nSaved cleaned definitions to: {OUTPUT_FILE}")
print(f"Saved dangling words log to: {DANGLING_LOG}")
print(f"Saved removed headwords log to: {REMOVED_LOG}")
EOF

    if [ ! -f "$CLEANED_DEFINITIONS" ]; then
        echo "Error: Cleaning failed - output file not created"
        exit 1
    fi
    
    echo ""
    echo "Stage 0 complete."
fi

# ==============================================================================
#                           STAGE 1: SOD PAIRS ANALYSIS
# ==============================================================================

print_stage "1" "SOD Pairs Analysis"

# Calculate expected number of output files
EXPECTED_PAIRS_FILES=$NUM_JOBS

echo "Submitting SLURM array job..."
echo "  Array size: $NUM_JOBS jobs"
echo "  Expected output files: $EXPECTED_PAIRS_FILES"

# Build end-idx argument if specified
END_IDX_ARG=""
if [ -n "$END_IDX" ]; then
    END_IDX_ARG="--end_idx $END_IDX"
fi

# Submit the pairs analysis job
JID_PAIRS=$(sbatch --parsable \
    --job-name="od_pairs_${CORPUS_NAME}" \
    --partition="$SLURM_PARTITION" \
    --time="$SLURM_TIME" \
    --mem="$SLURM_MEM" \
    --cpus-per-task="$CORES_PER_JOB" \
    --array="1-${NUM_JOBS}" \
    --output="${CURRENT_LOG_DIR}/pairs_%A_%a.out" \
    --error="${CURRENT_LOG_DIR}/pairs_%A_%a.err" \
    --wrap="cd ${CURRENT_OUTPUT_DIR} && python3 ${SCRIPT_DIR}/run_od_analysis.py \
        --job_id \${SLURM_ARRAY_TASK_ID} \
        --total_jobs ${NUM_JOBS} \
        --mode pairs \
        --num_workers ${CORES_PER_JOB} \
        --input_file ${CLEANED_DEFINITIONS} \
        --start_idx ${START_IDX} \
        ${END_IDX_ARG}")

echo "  Submitted Job ID: $JID_PAIRS"
echo ""

# Monitor progress
echo "Monitoring progress..."
python3 "${SCRIPT_DIR}/watch_stage.py" \
    --job_id "$JID_PAIRS" \
    --pattern "${CURRENT_OUTPUT_DIR}/pairs_results_job_*.txt" \
    --total "$EXPECTED_PAIRS_FILES" \
    --name "Stage 1 (Pairs)"

echo ""
echo "Stage 1 complete."

# ==============================================================================
#                           STAGE 2: SINGLE GOD ANALYSIS
# ==============================================================================

print_stage "2" "Single GOD Analysis"

EXPECTED_GOD_FILES=$NUM_JOBS

echo "Submitting SLURM array job..."
echo "  Array size: $NUM_JOBS jobs"
echo "  Expected output files: $EXPECTED_GOD_FILES"

JID_GOD=$(sbatch --parsable \
    --job-name="od_god_${CORPUS_NAME}" \
    --partition="$SLURM_PARTITION" \
    --time="04:00:00" \
    --mem="8G" \
    --cpus-per-task="$CORES_PER_JOB" \
    --array="1-${NUM_JOBS}" \
    --output="${CURRENT_LOG_DIR}/god_%A_%a.out" \
    --error="${CURRENT_LOG_DIR}/god_%A_%a.err" \
    --wrap="cd ${CURRENT_OUTPUT_DIR} && python3 ${SCRIPT_DIR}/run_od_analysis.py \
        --job_id \${SLURM_ARRAY_TASK_ID} \
        --total_jobs ${NUM_JOBS} \
        --mode single_god \
        --num_workers ${CORES_PER_JOB} \
        --input_file ${CLEANED_DEFINITIONS} \
        --start_idx ${START_IDX} \
        ${END_IDX_ARG}")

echo "  Submitted Job ID: $JID_GOD"
echo ""

# Monitor progress
python3 "${SCRIPT_DIR}/watch_stage.py" \
    --job_id "$JID_GOD" \
    --pattern "${CURRENT_OUTPUT_DIR}/single_god_results_job_*.txt" \
    --total "$EXPECTED_GOD_FILES" \
    --name "Stage 2 (GOD)"

echo ""
echo "Stage 2 complete."

# ==============================================================================
#                           STAGE 3: AGGREGATE RESULTS
# ==============================================================================

print_stage "3" "Aggregate Results"

echo "Merging pairs results..."
cat "${CURRENT_OUTPUT_DIR}"/pairs_results_job_*.txt > "${CURRENT_OUTPUT_DIR}/pairs_results.txt" 2>/dev/null || {
    echo "Warning: No pairs result files found to merge"
}

PAIRS_COUNT=$(wc -l < "${CURRENT_OUTPUT_DIR}/pairs_results.txt" 2>/dev/null || echo "0")
echo "  Total pairs: $PAIRS_COUNT"

echo ""
echo "Merging GOD results..."
cat "${CURRENT_OUTPUT_DIR}"/single_god_results_job_*.txt > "${CURRENT_OUTPUT_DIR}/single_god_results.txt" 2>/dev/null || {
    echo "Warning: No GOD result files found to merge"
}

GOD_COUNT=$(wc -l < "${CURRENT_OUTPUT_DIR}/single_god_results.txt" 2>/dev/null || echo "0")
echo "  Total GOD scores: $GOD_COUNT"

echo ""
echo "Cleaning up temporary files..."
rm -f "${CURRENT_OUTPUT_DIR}"/pairs_results_job_*.txt 2>/dev/null || true
rm -f "${CURRENT_OUTPUT_DIR}"/single_god_results_job_*.txt 2>/dev/null || true

echo ""
echo "Stage 3 complete."

# Summary for this corpus
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  CORPUS COMPLETE: ${CORPUS_NAME}"
echo "  Results: ${CURRENT_OUTPUT_DIR}"
echo "  Pairs: ${PAIRS_COUNT} | GOD: ${GOD_COUNT}"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

}  # End of run_pipeline_for_corpus function

# ==============================================================================
#                           RUN PIPELINE FOR CORPORA
# ==============================================================================

if [ "$RUN_ALL_CORPORA" = true ] && [ "$GENERATE_CORPORA" = true ]; then
    print_header "RUNNING PIPELINE FOR ALL CORPORA"
    
    echo "Corpora to process:"
    for corpus in "${CORPORA_LIST[@]}"; do
        echo "  - $(basename "$corpus")"
    done
    echo ""
    
    PROCESSED=0
    TOTAL_CORPORA=${#CORPORA_LIST[@]}
    
    for corpus in "${CORPORA_LIST[@]}"; do
        PROCESSED=$((PROCESSED + 1))
        echo ""
        echo "==============================================================================="
        echo "  CORPUS $PROCESSED / $TOTAL_CORPORA"
        echo "==============================================================================="
        run_pipeline_for_corpus "$corpus"
    done
else
    # Run on just the primary input file
    run_pipeline_for_corpus "$INPUT_DEFINITIONS"
fi

# ==============================================================================
#                           FINAL SUMMARY
# ==============================================================================

print_header "ALL PIPELINES COMPLETE"

if [ "$RUN_ALL_CORPORA" = true ] && [ "$GENERATE_CORPORA" = true ]; then
    echo "Processed ${TOTAL_CORPORA} corpora:"
    for corpus in "${CORPORA_LIST[@]}"; do
        cname=$(basename "$corpus" .txt)
        echo "  - results_${cname}/"
    done
else
    echo "Processed: results_$(basename "$INPUT_DEFINITIONS" .txt)/"
fi

echo ""
echo "Generated corpora location: ${SCRIPT_DIR}/generated_corpora/"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"

exit 0
