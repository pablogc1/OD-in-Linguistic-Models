#!/bin/bash
# ==============================================================================
# Post-Processing Script for Ontological Differentiation Results
#
# This script:
# 1. Merges the per-job output files into final result files.
# 2. Extracts "one-vs-all" results for a target word (optional).
#
# Usage:
#   bash aggregate_and_filter.sh [options]
#
# Options:
#   --results-dir DIR    Directory containing result files (default: current dir)
#   --target-word WORD   Extract one-vs-all results for this word
#   --definitions FILE   Definitions file (for finding word index)
#   --cleanup            Remove per-job files after merging
#
# ==============================================================================

set -e

# Default values
RESULTS_DIR="."
TARGET_WORD=""
DEFINITIONS_FILE=""
CLEANUP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --target-word)
            TARGET_WORD="$2"
            shift 2
            ;;
        --definitions)
            DEFINITIONS_FILE="$2"
            shift 2
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        -h|--help)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "  OD Results Aggregation & Filtering"
echo "=============================================="
echo ""
echo "Results directory: $RESULTS_DIR"

# --- Task 1: Merge Pairs Results ---
echo ""
echo "[Task 1] Merging 'pairs' results..."

PAIRS_PATTERN="${RESULTS_DIR}/pairs_results_job_*.txt"
PAIRS_OUTPUT="${RESULTS_DIR}/pairs_results.txt"

if ls $PAIRS_PATTERN 1> /dev/null 2>&1; then
    cat ${RESULTS_DIR}/pairs_results_job_*.txt > "$PAIRS_OUTPUT"
    PAIRS_COUNT=$(wc -l < "$PAIRS_OUTPUT")
    echo "  -> Merged into pairs_results.txt ($PAIRS_COUNT pairs)"
    
    if [ "$CLEANUP" = true ]; then
        rm -f ${RESULTS_DIR}/pairs_results_job_*.txt
        echo "  -> Removed temporary files"
    fi
else
    echo "  -> No pairs_results_job_*.txt files found. Skipping."
fi

# --- Task 2: Merge Single GOD Results ---
echo ""
echo "[Task 2] Merging 'single_god' results..."

GOD_PATTERN="${RESULTS_DIR}/single_god_results_job_*.txt"
GOD_OUTPUT="${RESULTS_DIR}/single_god_results.txt"

if ls $GOD_PATTERN 1> /dev/null 2>&1; then
    cat ${RESULTS_DIR}/single_god_results_job_*.txt > "$GOD_OUTPUT"
    GOD_COUNT=$(wc -l < "$GOD_OUTPUT")
    echo "  -> Merged into single_god_results.txt ($GOD_COUNT words)"
    
    if [ "$CLEANUP" = true ]; then
        rm -f ${RESULTS_DIR}/single_god_results_job_*.txt
        echo "  -> Removed temporary files"
    fi
else
    echo "  -> No single_god_results_job_*.txt files found. Skipping."
fi

# --- Task 3: Extract One-vs-All Results (Optional) ---
if [ -n "$TARGET_WORD" ]; then
    echo ""
    echo "[Task 3] Extracting one-vs-all results for: '${TARGET_WORD}'"
    
    if [ -z "$DEFINITIONS_FILE" ]; then
        # Try to find a definitions file
        DEFINITIONS_FILE="${RESULTS_DIR}/extracted_definitions_cleaned.txt"
        if [ ! -f "$DEFINITIONS_FILE" ]; then
            DEFINITIONS_FILE="${RESULTS_DIR}/extracted_definitions.txt"
        fi
    fi
    
    if [ ! -f "$DEFINITIONS_FILE" ]; then
        echo "  -> ERROR: Definitions file not found. Cannot determine word index."
        echo "     Specify with --definitions <file>"
    else
        # Find the word index (1-based line number)
        WORD_INDEX=$(grep -n "^${TARGET_WORD}:" "${DEFINITIONS_FILE}" | head -1 | cut -d: -f1)
        
        if [ -z "${WORD_INDEX}" ]; then
            echo "  -> ERROR: Word '${TARGET_WORD}' not found in definitions file."
        else
            echo "  -> Word index: ${WORD_INDEX}"
            
            OUTPUT_FILE="${RESULTS_DIR}/one_vs_all_${TARGET_WORD}.txt"
            
            # Extract all pairs containing this word index
            # Format: idx1 idx2 sod_score sod_term_level omega_god
            grep -E "(^${WORD_INDEX} |^[0-9]+ ${WORD_INDEX} )" "$PAIRS_OUTPUT" > "$OUTPUT_FILE" || true
            
            ONE_VS_ALL_COUNT=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo "0")
            echo "  -> Extracted ${ONE_VS_ALL_COUNT} pairs to: one_vs_all_${TARGET_WORD}.txt"
        fi
    fi
fi

# --- Summary ---
echo ""
echo "=============================================="
echo "  Aggregation Complete"
echo "=============================================="
echo ""

if [ -f "$PAIRS_OUTPUT" ]; then
    echo "  pairs_results.txt:       $(wc -l < "$PAIRS_OUTPUT") pairs"
fi
if [ -f "$GOD_OUTPUT" ]; then
    echo "  single_god_results.txt:  $(wc -l < "$GOD_OUTPUT") words"
fi
if [ -n "$TARGET_WORD" ] && [ -f "${RESULTS_DIR}/one_vs_all_${TARGET_WORD}.txt" ]; then
    echo "  one_vs_all_${TARGET_WORD}.txt: $(wc -l < "${RESULTS_DIR}/one_vs_all_${TARGET_WORD}.txt") pairs"
fi

echo ""
