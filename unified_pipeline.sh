#!/bin/bash
# ==============================================================================
#           ONTOLOGICAL DIFFERENTIATION - UNIFIED PIPELINE
# ==============================================================================
#
# A live orchestrator for the OD project. Stays running on a compute node
# and actively manages job submissions, respecting Cesvima's resource limits.
#
# ARCHITECTURE: Instead of submitting everything as a DAG upfront, this script
# submits one batch at a time, polls squeue for completion, then submits the
# next batch. This avoids MaxSubmitJobsPerAccount errors and keeps the cluster
# saturated.
#
# PHASES:
#   Phase 1: Data Generation (optional, usually done locally)
#   Phase 2: Data Curation (optional, usually done locally)
#   Phase 3: OD Calculations (SOD + GOD for all corpora)
#   Phase 4: Intra-Corpus Analysis (index, level coincidence, semantic diff)
#   Phase 5: Inter-Corpus Analysis (EOD + pairwise diffs + summary)
#   Phase 6: Dynamics Analysis (stability of TL=1 walks)
#
# RESOURCE LIMITS (Cesvima standard partition):
#   - Max 600 cores simultaneously
#   - Max ~160 submitted jobs (empirical MaxSubmitJobsPerAccount)
#   - Max 160 hours walltime (with reduced cores)
#
# ==============================================================================

set -o pipefail

# ==============================================================================
#                           CONFIGURATION
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_JOBS=${NUM_JOBS:-40}
CORES_PER_JOB=${CORES_PER_JOB:-16}
SLURM_PARTITION=${SLURM_PARTITION:-"standard"}
SLURM_TIME=${SLURM_TIME:-"48:00:00"}
SLURM_MEM=${SLURM_MEM:-"24G"}
SLURM_EXCLUDE=${SLURM_EXCLUDE:-"r1n8,r3n17,r1n46,r1n16"}
POLL_INTERVAL=${POLL_INTERVAL:-30}

CURATED_DIR="curated_corpora"

declare -a ALL_CORPORA=(
    "${CURATED_DIR}/curated_ground_filtered.txt"
    "${CURATED_DIR}/curated_ai_generated.txt"
    "${CURATED_DIR}/curated_merriam_webster.txt"
    "${CURATED_DIR}/curated_random_removal.txt"
    "${CURATED_DIR}/curated_targeted_removal.txt"
    "${CURATED_DIR}/curated_null_model.txt"
)

# ==============================================================================
#                           HELPER FUNCTIONS
# ==============================================================================

print_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    printf "║  %-76s ║\n" "$1"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_phase() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  PHASE $1: $2"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

print_step() {
    echo ""
    echo "  ┌─────────────────────────────────────────────────────────────────────────────┐"
    echo "  │ Step $1: $2"
    echo "  └─────────────────────────────────────────────────────────────────────────────┘"
}

format_elapsed() {
    local seconds=$1
    printf '%02d:%02d:%02d' $((seconds/3600)) $(((seconds%3600)/60)) $((seconds%60))
}

# Wait for ALL tasks of a Slurm job to complete.
# Polls squeue until no tasks remain for the given job ID.
# Returns 0 on success, 1 if any tasks failed.
wait_for_slurm_job() {
    local JOB_ID="$1"
    local LABEL="$2"
    local EXPECTED="${3:-0}"
    local SENTINEL_PATTERN="${4:-}"
    local START_TIME=$(date +%s)

    echo "    Monitoring: ${LABEL} (Job ${JOB_ID})"

    while true; do
        local REMAINING=$(squeue -j "$JOB_ID" -h 2>/dev/null | wc -l | tr -d ' ')

        if [ "$REMAINING" -eq 0 ] 2>/dev/null; then
            local ELAPSED=$(( $(date +%s) - START_TIME ))
            echo "    ✓ ${LABEL} complete | Elapsed: $(format_elapsed $ELAPSED)"

            # If sentinel pattern provided, verify completions
            if [ -n "$SENTINEL_PATTERN" ] && [ "$EXPECTED" -gt 0 ]; then
                local DONE_COUNT=$(ls ${SENTINEL_PATTERN} 2>/dev/null | wc -l | tr -d ' ')
                if [ "$DONE_COUNT" -lt "$EXPECTED" ]; then
                    echo "    ⚠ WARNING: Only ${DONE_COUNT}/${EXPECTED} sentinel files found."
                    echo "      Pattern: ${SENTINEL_PATTERN}"
                    echo "      Some tasks may have FAILED. Check .err log files."
                    return 1
                fi
                echo "    ✓ Verified: ${DONE_COUNT}/${EXPECTED} sentinels present"
            fi
            return 0
        fi

        local ELAPSED=$(( $(date +%s) - START_TIME ))
        # Progress every 5 minutes
        if [ $((ELAPSED % 300)) -lt "$POLL_INTERVAL" ] && [ "$ELAPSED" -gt 0 ]; then
            local DONE_COUNT=0
            if [ -n "$SENTINEL_PATTERN" ]; then
                DONE_COUNT=$(ls ${SENTINEL_PATTERN} 2>/dev/null | wc -l | tr -d ' ')
            fi
            echo "    [${LABEL}] ${REMAINING} tasks remaining | ${DONE_COUNT}/${EXPECTED} done | Elapsed: $(format_elapsed $ELAPSED)"
        fi

        sleep "$POLL_INTERVAL"
    done
}

check_file() {
    if [ -f "$1" ]; then
        local lines=$(wc -l < "$1")
        echo "    ✓ $1 ($lines entries)"
        return 0
    else
        echo "    ✗ $1 (not found)"
        return 1
    fi
}

# ==============================================================================
#                           PARSE ARGUMENTS
# ==============================================================================

usage() {
    cat << EOF
Usage: $0 [options]

PHASE 3 - OD Calculations:
  --run-od             Run SOD analysis on all available corpora

PHASE 4 - Intra-Corpus Analysis (per corpus):
  --run-analysis       Run level coincidence + semantic difference analysis

PHASE 5 - Inter-Corpus Analysis:
  --run-eod            Run EOD + pairwise diffs + summary reports

PHASE 6 - Dynamics Analysis:
  --run-dynamics       Run ontological dynamics (SOD percentile walks V4)

SLURM Options:
  --jobs N             Number of parallel SLURM array jobs (default: $NUM_JOBS)
  --cores N            CPU cores per job (default: $CORES_PER_JOB)
  --partition NAME     SLURM partition (default: $SLURM_PARTITION)
  --time HH:MM:SS      SLURM time limit (default: $SLURM_TIME)
  --mem SIZE           Memory per job (default: $SLURM_MEM)

Other:
  --status             Show current status of all corpora and results
  --yes, -y            Non-interactive mode (skip confirmation prompts)
  -h, --help           Show this help message

EOF
    exit 0
}

DO_RUN_OD=false
DO_RUN_ANALYSIS=false
DO_RUN_EOD=false
DO_RUN_DYNAMICS=true
DO_STATUS=false
NON_INTERACTIVE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --run-od)           DO_RUN_OD=true; shift ;;
        --run-analysis)     DO_RUN_ANALYSIS=true; shift ;;
        --run-eod)          DO_RUN_EOD=true; shift ;;
        --run-dynamics)     DO_RUN_DYNAMICS=true; shift ;;
        --jobs)             NUM_JOBS="$2"; shift 2 ;;
        --cores)            CORES_PER_JOB="$2"; shift 2 ;;
        --partition)        SLURM_PARTITION="$2"; shift 2 ;;
        --time)             SLURM_TIME="$2"; shift 2 ;;
        --mem)              SLURM_MEM="$2"; shift 2 ;;
        --status)           DO_STATUS=true; shift ;;
        --yes|-y)           NON_INTERACTIVE=true; shift ;;
        -h|--help)          usage ;;
        *)                  echo "Unknown option: $1"; usage ;;
    esac
done

if ! $DO_RUN_OD && ! $DO_RUN_ANALYSIS && ! $DO_RUN_EOD && ! $DO_RUN_DYNAMICS && ! $DO_STATUS; then
    echo "No phases selected and defaults are all off. Use --help for options."
    exit 1
fi

# ==============================================================================
#                           STATUS DISPLAY
# ==============================================================================

if $DO_STATUS; then
    print_banner "PIPELINE STATUS"

    echo "CURATED CORPORA:"
    for corpus in "${ALL_CORPORA[@]}"; do
        check_file "$corpus" || true
    done
    check_file "${CURATED_DIR}/common_vocabulary.txt" || true

    echo ""
    echo "OD RESULTS:"
    for corpus in "${ALL_CORPORA[@]}"; do
        basename=$(basename "$corpus" .txt)
        results_dir="results_${basename}"
        if [ -d "$results_dir" ]; then
            pairs_file="${results_dir}/pairs_results.txt"
            god_file="${results_dir}/single_god_results.txt"
            if [ -f "$pairs_file" ] && [ -f "$god_file" ]; then
                pairs_count=$(wc -l < "$pairs_file")
                god_count=$(wc -l < "$god_file")
                echo "    ✓ $basename: ${pairs_count} pairs, ${god_count} GOD scores"
            else
                echo "    ◐ $basename: incomplete"
            fi
        else
            echo "    ✗ $basename: not processed"
        fi
    done

    echo ""
    echo "INTRA-CORPUS ANALYSIS:"
    for corpus in "${ALL_CORPORA[@]}"; do
        basename=$(basename "$corpus" .txt)
        results_dir="results_${basename}"
        if [ -d "$results_dir" ]; then
            agree_file="${results_dir}/aggregated_summary_by_level.txt"
            semdiff_file="${results_dir}/semantic_difference_summary.txt"
            has_index=$(ls ${results_dir}/indexed_pairs_data/*.csv 2>/dev/null | head -1)
            status=""
            [ -n "$has_index" ] && status="index"
            [ -f "$agree_file" ] && status="${status}+agree"
            [ -f "$semdiff_file" ] && status="${status}+semdiff"
            if [ -n "$status" ]; then
                echo "    ◐ $basename: $status"
            else
                echo "    ✗ $basename: no analysis"
            fi
        fi
    done

    echo ""
    echo "INTER-CORPUS ANALYSIS:"
    check_file "eod_results_final.csv" || true
    check_file "summary_report.txt" || true
    if [ -d "pairwise_diff_results" ]; then
        diff_count=$(ls pairwise_diff_results/diffs_*.csv 2>/dev/null | wc -l)
        echo "    Pairwise diff files: $diff_count / 15"
    fi

    exit 0
fi

# ==============================================================================
#                           EXECUTION PLAN
# ==============================================================================

print_banner "ONTOLOGICAL DIFFERENTIATION - UNIFIED PIPELINE"

echo "EXECUTION PLAN:"
echo "  Phase 3 - OD Calculations:"
echo "    • Run SOD analysis:       $(if $DO_RUN_OD; then echo 'YES (all corpora)'; else echo 'skip'; fi)"
echo ""
echo "  Phase 4 - Intra-Corpus Analysis:"
echo "    • Level coincidence:      $(if $DO_RUN_ANALYSIS; then echo 'YES'; else echo 'skip'; fi)"
echo "    • Semantic difference:    $(if $DO_RUN_ANALYSIS; then echo 'YES'; else echo 'skip'; fi)"
echo ""
echo "  Phase 5 - Inter-Corpus Analysis:"
echo "    • EOD + Pairwise diffs:   $(if $DO_RUN_EOD; then echo 'YES'; else echo 'skip'; fi)"
echo ""
echo "  Phase 6 - Dynamics Analysis:"
    echo "    • SOD percentile walks:   $(if $DO_RUN_DYNAMICS; then echo 'YES (all corpora)'; else echo 'skip'; fi)"
echo ""
echo "ARCHITECTURE: Live orchestrator (sequential batches, cluster-saturating)"
echo "  • Jobs per batch: ${NUM_JOBS}"
echo "  • Cores per job:  ${CORES_PER_JOB}"
echo "  • Poll interval:  ${POLL_INTERVAL}s"
echo ""

if ! $NON_INTERACTIVE; then
    echo "Press ENTER to continue or Ctrl+C to abort..."
    read -r
fi

echo "Loading modules..."
module --force purge > /dev/null 2>&1 || true
module load apps/2021 Python/3.10.8-GCCcore-12.2.0 2>/dev/null || {
    echo "Warning: Could not load Python module. Using system Python."
}

PIPELINE_START=$(date +%s)
PIPELINE_ERRORS=0

# ==============================================================================
#                     PHASE 3: OD CALCULATIONS
# ==============================================================================

if $DO_RUN_OD; then
    print_phase "3" "OD CALCULATIONS (SOD Analysis)"

    CORPORA_TO_PROCESS=()
    for corpus in "${ALL_CORPORA[@]}"; do
        if [ -f "$corpus" ]; then
            CORPORA_TO_PROCESS+=("$corpus")
        fi
    done

    echo "Corpora to process: ${#CORPORA_TO_PROCESS[@]}"
    for corpus in "${CORPORA_TO_PROCESS[@]}"; do
        echo "  - $(basename "$corpus")"
    done

    CORPUS_NUM=0
    TOTAL_CORPORA=${#CORPORA_TO_PROCESS[@]}

    for CORPUS_FILE in "${CORPORA_TO_PROCESS[@]}"; do
        CORPUS_NUM=$((CORPUS_NUM + 1))
        CORPUS_NAME=$(basename "$CORPUS_FILE" .txt)
        OUTPUT_DIR="results_${CORPUS_NAME}"
        LOG_DIR="${OUTPUT_DIR}/logs"

        print_step "3.${CORPUS_NUM}" "Processing ${CORPUS_NAME} ($CORPUS_NUM/$TOTAL_CORPORA)"

        mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

        # Resume logic: skip if final results already exist
        if [ -f "${OUTPUT_DIR}/pairs_results.txt" ] && [ -f "${OUTPUT_DIR}/single_god_results.txt" ]; then
            PAIRS_COUNT=$(wc -l < "${OUTPUT_DIR}/pairs_results.txt")
            GOD_COUNT=$(wc -l < "${OUTPUT_DIR}/single_god_results.txt")
            echo "  ✓ SKIP (already complete): ${PAIRS_COUNT} pairs, ${GOD_COUNT} GOD scores"
            continue
        fi

        WORD_COUNT=$(wc -l < "${CORPUS_FILE}")
        echo "  Corpus: ${CORPUS_FILE} (${WORD_COUNT} words)"

        # --- Submit SOD pairs (40 array tasks) ---
        echo "  Submitting SOD pairs analysis ($NUM_JOBS jobs)..."
        JID_PAIRS=$(sbatch --parsable \
            --job-name="sod_${CORPUS_NAME}" \
            --partition="$SLURM_PARTITION" \
            --exclude="$SLURM_EXCLUDE" \
            --time="$SLURM_TIME" \
            --mem="$SLURM_MEM" \
            --cpus-per-task="$CORES_PER_JOB" \
            --array="1-${NUM_JOBS}" \
            --output="${LOG_DIR}/pairs_%A_%a.out" \
            --error="${LOG_DIR}/pairs_%A_%a.err" \
            --wrap="module load apps/2021 Python/3.10.8-GCCcore-12.2.0 && cd ${OUTPUT_DIR} && python3 ${SCRIPT_DIR}/run_od_analysis.py \
                --job_id \${SLURM_ARRAY_TASK_ID} \
                --total_jobs ${NUM_JOBS} \
                --mode pairs \
                --num_workers ${CORES_PER_JOB} \
                --input_file ../${CORPUS_FILE}")

        if [ $? -ne 0 ]; then
            echo "  ✗ FAILED to submit SOD pairs for ${CORPUS_NAME}"
            PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
            continue
        fi
        echo "    Pairs Job ID: $JID_PAIRS"

        # --- Submit GOD (40 array tasks) ---
        echo "  Submitting single GOD analysis ($NUM_JOBS jobs)..."
        JID_GOD=$(sbatch --parsable \
            --job-name="god_${CORPUS_NAME}" \
            --partition="$SLURM_PARTITION" \
            --exclude="$SLURM_EXCLUDE" \
            --time="01:00:00" \
            --mem="8G" \
            --cpus-per-task="$CORES_PER_JOB" \
            --array="1-${NUM_JOBS}" \
            --output="${LOG_DIR}/god_%A_%a.out" \
            --error="${LOG_DIR}/god_%A_%a.err" \
            --wrap="module load apps/2021 Python/3.10.8-GCCcore-12.2.0 && cd ${OUTPUT_DIR} && python3 ${SCRIPT_DIR}/run_od_analysis.py \
                --job_id \${SLURM_ARRAY_TASK_ID} \
                --total_jobs ${NUM_JOBS} \
                --mode single_god \
                --num_workers ${CORES_PER_JOB} \
                --input_file ../${CORPUS_FILE}")

        if [ $? -ne 0 ]; then
            echo "  ✗ FAILED to submit GOD for ${CORPUS_NAME}"
            PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
            continue
        fi
        echo "    GOD Job ID: $JID_GOD"

        # --- Wait for both to complete ---
        wait_for_slurm_job "$JID_PAIRS" "SOD pairs for ${CORPUS_NAME}" "$NUM_JOBS" "${OUTPUT_DIR}/pairs_results_job_*.txt.done"
        PAIRS_OK=$?

        wait_for_slurm_job "$JID_GOD" "GOD for ${CORPUS_NAME}" "$NUM_JOBS" "${OUTPUT_DIR}/single_god_results_job_*.txt.done"
        GOD_OK=$?

        if [ "$PAIRS_OK" -ne 0 ] || [ "$GOD_OK" -ne 0 ]; then
            echo "  ⚠ WARNING: Some jobs may have failed for ${CORPUS_NAME}. Continuing anyway."
            PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
        fi

        # --- Aggregate results inline ---
        echo "  Aggregating results..."
        cat ${OUTPUT_DIR}/pairs_results_job_*.txt > ${OUTPUT_DIR}/pairs_results.txt 2>/dev/null || true
        cat ${OUTPUT_DIR}/single_god_results_job_*.txt > ${OUTPUT_DIR}/single_god_results.txt 2>/dev/null || true

        PAIRS_COUNT=$(wc -l < ${OUTPUT_DIR}/pairs_results.txt 2>/dev/null || echo 0)
        GOD_COUNT=$(wc -l < ${OUTPUT_DIR}/single_god_results.txt 2>/dev/null || echo 0)
        echo "  ✓ ${CORPUS_NAME} complete: ${PAIRS_COUNT} pairs, ${GOD_COUNT} GOD scores"

        # Clean up job part files
        rm -f ${OUTPUT_DIR}/pairs_results_job_*.txt ${OUTPUT_DIR}/pairs_results_job_*.txt.done 2>/dev/null || true
        rm -f ${OUTPUT_DIR}/single_god_results_job_*.txt ${OUTPUT_DIR}/single_god_results_job_*.txt.done 2>/dev/null || true
        echo ""
    done

    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  PHASE 3 COMPLETE"
    echo "═══════════════════════════════════════════════════════════════════════════════"
fi

# ==============================================================================
#                     PHASE 4: INTRA-CORPUS ANALYSIS
# ==============================================================================

if $DO_RUN_ANALYSIS; then
    print_phase "4" "INTRA-CORPUS ANALYSIS (Level Coincidence + Semantic Difference)"

    ANALYSIS_CORPORA=()
    for corpus in "${ALL_CORPORA[@]}"; do
        if [ -f "$corpus" ]; then
            ANALYSIS_CORPORA+=("$corpus")
        fi
    done

    if [ ${#ANALYSIS_CORPORA[@]} -eq 0 ]; then
        echo "ERROR: No curated corpora found."
        exit 1
    fi

    echo "Processing ${#ANALYSIS_CORPORA[@]} corpora..."

    CORPUS_NUM=0
    for CORPUS_FILE in "${ANALYSIS_CORPORA[@]}"; do
        CORPUS_NUM=$((CORPUS_NUM + 1))
        CORPUS_NAME=$(basename "$CORPUS_FILE" .txt)
        RESULTS_DIR="results_${CORPUS_NAME}"
        PAIRS_FILE="${RESULTS_DIR}/pairs_results.txt"
        INDEX_DIR="${RESULTS_DIR}/indexed_pairs_data"
        LOG_DIR="${RESULTS_DIR}/logs"
        TEMP_INDEX_DIR="temp_index_parts_${CORPUS_NAME}"

        print_step "4.${CORPUS_NUM}" "Processing ${CORPUS_NAME} ($CORPUS_NUM/${#ANALYSIS_CORPORA[@]})"

        # Verify pairs_results.txt exists
        if [ ! -f "$PAIRS_FILE" ]; then
            echo "  ✗ ${PAIRS_FILE} not found. Skipping this corpus."
            echo "    (Run Phase 3 first or check for errors.)"
            PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
            continue
        fi

        mkdir -p "$INDEX_DIR" "$LOG_DIR"

        # Resume logic: check if this corpus is fully done
        AGREE_DONE=false
        SEMDIFF_DONE=false
        INDEX_DONE=false
        [ -f "${RESULTS_DIR}/aggregated_summary_by_level.txt" ] && AGREE_DONE=true
        [ -f "${RESULTS_DIR}/semantic_difference_summary.txt" ] && SEMDIFF_DONE=true

        if [ -n "$(ls ${INDEX_DIR}/*.csv 2>/dev/null | head -1)" ]; then
            INDEX_DONE=true
        fi

        if $AGREE_DONE && $SEMDIFF_DONE; then
            echo "  ✓ SKIP (already complete): agreement + semantic diff done"
            continue
        fi

        # ===== Step 4a: Build per-word index =====
        if $INDEX_DONE; then
            echo "  [4a] SKIP index building (index files already exist)"
        else
            rm -rf "${TEMP_INDEX_DIR}" 2>/dev/null || true

            echo "  [4a] Submitting per-word index building ($NUM_JOBS jobs)..."

            JID_INDEX=$(sbatch --parsable \
                --job-name="index_${CORPUS_NAME}" \
                --partition="$SLURM_PARTITION" \
                --exclude="$SLURM_EXCLUDE" \
                --time="04:00:00" \
                --mem="16G" \
                --cpus-per-task="1" \
                --array="1-${NUM_JOBS}" \
                --output="${LOG_DIR}/index_%A_%a.out" \
                --error="${LOG_DIR}/index_%A_%a.err" \
                --wrap="module load apps/2021 Python/3.10.8-GCCcore-12.2.0 && python3 ${SCRIPT_DIR}/step_python_indexer.py \
                    --input_file ${PAIRS_FILE} \
                    --corpus_key ${CORPUS_NAME} \
                    --job_id \${SLURM_ARRAY_TASK_ID} \
                    --total_jobs ${NUM_JOBS}")

            if [ $? -ne 0 ]; then
                echo "  ✗ FAILED to submit index building for ${CORPUS_NAME}"
                PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
                continue
            fi
            echo "    Index Job ID: $JID_INDEX"

            wait_for_slurm_job "$JID_INDEX" "Index building for ${CORPUS_NAME}" "$NUM_JOBS" "${TEMP_INDEX_DIR}/job_*.done"
            if [ $? -ne 0 ]; then
                echo "  ⚠ WARNING: Index building incomplete for ${CORPUS_NAME}."
                PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
            fi

            # ===== Step 4a-merge: Merge index parts =====
            echo "  [4a-merge] Submitting index merge (16 cores)..."

            JID_MERGE=$(sbatch --parsable \
                --job-name="idxmrg_${CORPUS_NAME}" \
                --partition="$SLURM_PARTITION" \
                --exclude="$SLURM_EXCLUDE" \
                --time="02:00:00" \
                --mem="16G" \
                --cpus-per-task="16" \
                --output="${LOG_DIR}/idxmrg_%j.out" \
                --error="${LOG_DIR}/idxmrg_%j.err" \
                --wrap="module load apps/2021 Python/3.10.8-GCCcore-12.2.0 && python3 ${SCRIPT_DIR}/merge_index_parts.py \
                    --temp_dir ${TEMP_INDEX_DIR} \
                    --output_dir ${INDEX_DIR} \
                    --cleanup")

            if [ $? -ne 0 ]; then
                echo "  ✗ FAILED to submit index merge for ${CORPUS_NAME}"
                PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
                continue
            fi
            echo "    Merge Job ID: $JID_MERGE"

            wait_for_slurm_job "$JID_MERGE" "Index merge for ${CORPUS_NAME}"
            if [ $? -ne 0 ]; then
                echo "  ⚠ WARNING: Index merge may have failed for ${CORPUS_NAME}."
                PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
            fi
        fi

        # ===== Step 4b: Level Coincidence (Agreement) =====
        JID_AGREE=""
        if $AGREE_DONE; then
            echo "  [4b] SKIP level coincidence (aggregated_summary_by_level.txt exists)"
        else
            rm -f ${RESULTS_DIR}/fast_agreement_results_job_*.txt ${RESULTS_DIR}/fast_agreement_results_job_*.txt.done 2>/dev/null || true

            echo "  [4b] Submitting level coincidence analysis ($NUM_JOBS jobs)..."

            JID_AGREE=$(sbatch --parsable \
                --job-name="agree_${CORPUS_NAME}" \
                --partition="$SLURM_PARTITION" \
                --exclude="$SLURM_EXCLUDE" \
                --time="$SLURM_TIME" \
                --mem="$SLURM_MEM" \
                --cpus-per-task="$CORES_PER_JOB" \
                --array="1-${NUM_JOBS}" \
                --output="${LOG_DIR}/agree_%A_%a.out" \
                --error="${LOG_DIR}/agree_%A_%a.err" \
                --wrap="module load apps/2021 Python/3.10.8-GCCcore-12.2.0 && cd ${RESULTS_DIR} && python3 ${SCRIPT_DIR}/run_fast_agreement.py \
                    --job_id \${SLURM_ARRAY_TASK_ID} \
                    --total_jobs ${NUM_JOBS} \
                    --num_workers ${CORES_PER_JOB} \
                    --vocab_file ../${CORPUS_FILE} \
                    --index_dir indexed_pairs_data")

            if [ $? -ne 0 ]; then
                echo "  ✗ FAILED to submit agreement for ${CORPUS_NAME}"
                PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
            else
                echo "    Agreement Job ID: $JID_AGREE"
            fi
        fi

        # ===== Step 4c: Semantic Difference =====
        JID_SEMDIFF=""
        if $SEMDIFF_DONE; then
            echo "  [4c] SKIP semantic difference (semantic_difference_summary.txt exists)"
        else
            rm -f ${RESULTS_DIR}/semantic_diff_results_job_*.csv ${RESULTS_DIR}/semantic_diff_results_job_*.csv.done 2>/dev/null || true

            echo "  [4c] Submitting semantic difference analysis ($NUM_JOBS jobs)..."

            JID_SEMDIFF=$(sbatch --parsable \
                --job-name="semdiff_${CORPUS_NAME}" \
                --partition="$SLURM_PARTITION" \
                --exclude="$SLURM_EXCLUDE" \
                --time="$SLURM_TIME" \
                --mem="$SLURM_MEM" \
                --cpus-per-task="$CORES_PER_JOB" \
                --array="1-${NUM_JOBS}" \
                --output="${LOG_DIR}/semdiff_%A_%a.out" \
                --error="${LOG_DIR}/semdiff_%A_%a.err" \
                --wrap="module load apps/2021 Python/3.10.8-GCCcore-12.2.0 && cd ${RESULTS_DIR} && python3 ${SCRIPT_DIR}/run_semantic_difference.py \
                    --job_id \${SLURM_ARRAY_TASK_ID} \
                    --total_jobs ${NUM_JOBS} \
                    --num_workers ${CORES_PER_JOB} \
                    --vocab_file ../${CORPUS_FILE} \
                    --index_dir indexed_pairs_data")

            if [ $? -ne 0 ]; then
                echo "  ✗ FAILED to submit semantic diff for ${CORPUS_NAME}"
                PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
            else
                echo "    Semantic Diff Job ID: $JID_SEMDIFF"
            fi
        fi

        # --- Wait for agreement and semantic diff ---
        if [ -n "$JID_AGREE" ]; then
            wait_for_slurm_job "$JID_AGREE" "Level coincidence for ${CORPUS_NAME}" "$NUM_JOBS" "${RESULTS_DIR}/fast_agreement_results_job_*.txt.done"
            if [ $? -ne 0 ]; then
                PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
            fi

            echo "  Aggregating agreement results..."
            AGREEMENT_FILE="${RESULTS_DIR}/detailed_agreement_report.txt"
            cat ${RESULTS_DIR}/fast_agreement_results_job_*.txt > "${AGREEMENT_FILE}" 2>/dev/null || true
            rm -f ${RESULTS_DIR}/fast_agreement_results_job_*.txt ${RESULTS_DIR}/fast_agreement_results_job_*.txt.done 2>/dev/null || true

            echo "  Running agreement summary..."
            (cd "${RESULTS_DIR}" && python3 "${SCRIPT_DIR}/aggregate_reports.py") || {
                echo "  ⚠ WARNING: Agreement aggregation failed."
                PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
            }

            if [ -f "${RESULTS_DIR}/aggregated_summary_by_level.txt" ]; then
                echo "  Cleaning up intermediate agreement file..."
                rm -f "${RESULTS_DIR}/detailed_agreement_report.txt"
            fi
        fi

        if [ -n "$JID_SEMDIFF" ]; then
            wait_for_slurm_job "$JID_SEMDIFF" "Semantic diff for ${CORPUS_NAME}" "$NUM_JOBS" "${RESULTS_DIR}/semantic_diff_results_job_*.csv.done"
            if [ $? -ne 0 ]; then
                PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
            fi

            echo "  Aggregating semantic diff results..."
            FIRST_SD=$(ls ${RESULTS_DIR}/semantic_diff_results_job_*.csv 2>/dev/null | head -1)
            if [ -n "$FIRST_SD" ]; then
                head -1 "$FIRST_SD" > "${RESULTS_DIR}/combined_semantic_diff_results.csv"
                tail -q -n +2 ${RESULTS_DIR}/semantic_diff_results_job_*.csv >> "${RESULTS_DIR}/combined_semantic_diff_results.csv"
            fi
            rm -f ${RESULTS_DIR}/semantic_diff_results_job_*.csv ${RESULTS_DIR}/semantic_diff_results_job_*.csv.done 2>/dev/null || true

            echo "  Running semantic diff summary..."
            (cd "${RESULTS_DIR}" && python3 "${SCRIPT_DIR}/aggregate_diff_summary.py") || {
                echo "  ⚠ WARNING: Semantic diff aggregation failed."
                PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
            }

            if [ -f "${RESULTS_DIR}/semantic_difference_summary.txt" ]; then
                echo "  Cleaning up intermediate semantic diff file..."
                rm -f "${RESULTS_DIR}/combined_semantic_diff_results.csv"
            fi
        fi

        echo "  ✓ ${CORPUS_NAME} Phase 4 complete"
        echo ""
    done

    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  PHASE 4 COMPLETE"
    echo "═══════════════════════════════════════════════════════════════════════════════"
fi

# ==============================================================================
#                     PHASE 5: INTER-CORPUS ANALYSIS (EOD)
# ==============================================================================

if $DO_RUN_EOD; then
    print_phase "5" "INTER-CORPUS ANALYSIS (EOD + Pairwise Diffs)"

    if [ ! -f "${CURATED_DIR}/common_vocabulary.txt" ]; then
        echo "ERROR: ${CURATED_DIR}/common_vocabulary.txt not found."
        echo "       Run curate_all_corpora.py first."
        exit 1
    fi

    EOD_DIR="eod_results"
    mkdir -p "$EOD_DIR"

    COMMON_VOCAB_SIZE=$(wc -l < "${CURATED_DIR}/common_vocabulary.txt")
    echo "  Common vocabulary: $COMMON_VOCAB_SIZE words"
    echo "  Corpus pairs: 15 (C(6,2) combinations)"
    echo ""

    # ===== Step 5a: EOD Calculation =====
    print_step "5a" "EOD Analysis ($NUM_JOBS jobs)"

    JID_EOD=$(sbatch --parsable \
        --job-name="eod_analysis" \
        --partition="$SLURM_PARTITION" \
        --exclude="$SLURM_EXCLUDE" \
        --time="24:00:00" \
        --mem="16G" \
        --cpus-per-task="$CORES_PER_JOB" \
        --array="1-${NUM_JOBS}" \
        --output="${EOD_DIR}/eod_%A_%a.out" \
        --error="${EOD_DIR}/eod_%A_%a.err" \
        --wrap="module load apps/2021 Python/3.10.8-GCCcore-12.2.0 && python3 ${SCRIPT_DIR}/step3_python_eod_worker.py \
            --job_id \${SLURM_ARRAY_TASK_ID} \
            --total_jobs ${NUM_JOBS} \
            --num_workers ${CORES_PER_JOB}")

    if [ $? -ne 0 ]; then
        echo "  ✗ FAILED to submit EOD analysis"
        PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
    else
        echo "    EOD Job ID: $JID_EOD"

        wait_for_slurm_job "$JID_EOD" "EOD analysis" "$NUM_JOBS" "${EOD_DIR}/eod_job_*.done"
        if [ $? -ne 0 ]; then
            PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
        fi

        # Merge EOD results inline
        echo "  Merging EOD results..."
        FINAL_EOD_FILE="eod_results_final.csv"
        FIRST_PART=$(ls ${EOD_DIR}/eod_results_part_*.csv 2>/dev/null | head -1)
        if [ -n "$FIRST_PART" ]; then
            head -1 "$FIRST_PART" > "${FINAL_EOD_FILE}"
            for part_file in ${EOD_DIR}/eod_results_part_*.csv; do
                tail -n +2 "$part_file" >> "${FINAL_EOD_FILE}"
            done
            EOD_LINES=$(wc -l < "${FINAL_EOD_FILE}")
            echo "  ✓ EOD merge complete: ${EOD_LINES} lines in ${FINAL_EOD_FILE}"
        else
            echo "  ⚠ WARNING: No EOD result parts found."
            PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
        fi
        rm -f ${EOD_DIR}/eod_job_*.done 2>/dev/null || true
    fi

    # ===== Step 5b: Pairwise Differences =====
    print_step "5b" "Inter-Corpus Pairwise Differences (15 jobs)"

    DIFF_DIR="pairwise_diff_results"
    mkdir -p "$DIFF_DIR"

    # Verify index directories exist for all corpora
    ALL_INDICES_OK=true
    for corpus in "${ALL_CORPORA[@]}"; do
        cname=$(basename "$corpus" .txt)
        idir="results_${cname}/indexed_pairs_data"
        if [ ! -d "$idir" ] || [ -z "$(ls ${idir}/*.csv 2>/dev/null | head -1)" ]; then
            echo "  ⚠ WARNING: Index directory missing or empty: ${idir}"
            ALL_INDICES_OK=false
        fi
    done

    if $ALL_INDICES_OK; then
        JID_DIFFS=$(sbatch --parsable \
            --job-name="pairwise_diffs" \
            --partition="$SLURM_PARTITION" \
            --exclude="$SLURM_EXCLUDE" \
            --time="24:00:00" \
            --mem="32G" \
            --cpus-per-task="1" \
            --array="1-15" \
            --output="${DIFF_DIR}/diff_%A_%a.out" \
            --error="${DIFF_DIR}/diff_%A_%a.err" \
            --wrap="module load apps/2021 Python/3.10.8-GCCcore-12.2.0 && python3 ${SCRIPT_DIR}/step4_python_diff_worker.py \
                --job_id \${SLURM_ARRAY_TASK_ID} \
                --total_jobs 15")

        if [ $? -ne 0 ]; then
            echo "  ✗ FAILED to submit pairwise diffs"
            PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
        else
            echo "    Pairwise Diff Job ID: $JID_DIFFS"

            wait_for_slurm_job "$JID_DIFFS" "Pairwise differences" "15" "${DIFF_DIR}/job_*.done"
            if [ $? -ne 0 ]; then
                PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
            fi

            # Merge diff results inline
            echo "  Merging pairwise diff results..."
            for combo_file in ${DIFF_DIR}/diffs_*_part_1.csv; do
                if [ -f "$combo_file" ]; then
                    combo=$(basename "$combo_file" | sed 's/diffs_\(.*\)_part_1\.csv/\1/')
                    echo "master_idx1,master_idx2,diff_sod,diff_sod_tl" > ${DIFF_DIR}/diffs_${combo}.csv
                    cat ${DIFF_DIR}/diffs_${combo}_part_*.csv >> ${DIFF_DIR}/diffs_${combo}.csv
                    rm -f ${DIFF_DIR}/diffs_${combo}_part_*.csv
                fi
            done
            rm -f ${DIFF_DIR}/job_*.done 2>/dev/null || true

            DIFF_COUNT=$(ls ${DIFF_DIR}/diffs_*.csv 2>/dev/null | wc -l)
            echo "  ✓ Pairwise diff merge complete: ${DIFF_COUNT} comparison files"
        fi
    else
        echo "  ✗ Skipping pairwise diffs: index directories incomplete."
        echo "    Run Phase 4 (--run-analysis) first."
        PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
    fi

    # ===== Step 5c: Final Summary Report =====
    print_step "5c" "Final Summary Report"

    if [ -f "eod_results_final.csv" ] && [ -d "$DIFF_DIR" ]; then
        echo "  Generating summary report and log histograms..."
        python3 "${SCRIPT_DIR}/step5_generate_summary.py" || {
            echo "  ⚠ WARNING: Summary generation failed."
            PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
        }

        if [ -f "summary_report.txt" ]; then
            echo "  ✓ Summary report: summary_report.txt"
        fi
        if [ -f "log_binned_histogram_data.csv" ]; then
            echo "  ✓ Histogram data: log_binned_histogram_data.csv"
        fi
    else
        echo "  ✗ Skipping summary: prerequisites missing."
        PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  PHASE 5 COMPLETE"
    echo "═══════════════════════════════════════════════════════════════════════════════"
fi

# ==============================================================================
#                    PHASE 6: DYNAMICS ANALYSIS
# ==============================================================================

if $DO_RUN_DYNAMICS; then
    print_phase "6" "DYNAMICS ANALYSIS (SOD Percentile Walks — V4: Equal-Count Bins)"

    DYNAMICS_DIR="dynamics_results"
    mkdir -p "$DYNAMICS_DIR"

    DYNAMICS_JOBS=40
    DYNAMICS_TRAJECTORIES=100
    DYNAMICS_MAX_STEPS=100

    echo "Configuration:"
    echo "  Jobs per corpus:    ${DYNAMICS_JOBS}"
    echo "  Starting pairs:     ALL mutual pairs (no cap)"
    echo "  Starting pctls:     1,5,10,15,...,95,100 (21 conditions, true equal-count bins)"
    echo "  Trajectories/pair:  ${DYNAMICS_TRAJECTORIES}"
    echo "  Max steps:          ${DYNAMICS_MAX_STEPS}"
    echo "  Walk variants:      chain (1st pctl walk, all conditions) + sweep (1st pctl start only)"
    echo ""

    CORPUS_NUM=0
    for CORPUS_FILE in "${ALL_CORPORA[@]}"; do
        CORPUS_NUM=$((CORPUS_NUM + 1))

        if [ ! -f "$CORPUS_FILE" ]; then
            echo "  ⚠ Corpus not found: $CORPUS_FILE — skipping."
            continue
        fi

        CORPUS_NAME=$(basename "$CORPUS_FILE" .txt)
        RESULTS_DIR="results_${CORPUS_NAME}"
        INDEX_DIR="${RESULTS_DIR}/indexed_pairs_data"
        LOG_DIR="${RESULTS_DIR}/logs"
        STATS_FINAL_CHECK="${DYNAMICS_DIR}/dynamics_${CORPUS_NAME}_stats_final.csv"

        print_step "6.${CORPUS_NUM}" "Dynamics for ${CORPUS_NAME} ($CORPUS_NUM/${#ALL_CORPORA[@]})"

        if [ -f "$STATS_FINAL_CHECK" ]; then
            echo "  ✓ SKIP (already complete): ${STATS_FINAL_CHECK} exists"
            continue
        fi

        if [ ! -d "$INDEX_DIR" ] || [ -z "$(ls ${INDEX_DIR}/*.csv 2>/dev/null | head -1)" ]; then
            echo "  ✗ Index directory missing or empty: ${INDEX_DIR}"
            echo "    Run Phase 4 (--run-analysis) first."
            PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
            continue
        fi

        mkdir -p "$LOG_DIR"

        rm -f ${DYNAMICS_DIR}/dynamics_${CORPUS_NAME}_job_*.csv \
              ${DYNAMICS_DIR}/dynamics_${CORPUS_NAME}_job_*.done 2>/dev/null || true

        echo "  Submitting ${DYNAMICS_JOBS} dynamics jobs..."

        JID_DYN=$(sbatch --parsable \
            --job-name="dyn_${CORPUS_NAME}" \
            --partition="$SLURM_PARTITION" \
            --exclude="$SLURM_EXCLUDE" \
            --time="48:00:00" \
            --mem="32G" \
            --cpus-per-task="1" \
            --array="1-${DYNAMICS_JOBS}" \
            --output="${LOG_DIR}/dyn_%A_%a.out" \
            --error="${LOG_DIR}/dyn_%A_%a.err" \
            --wrap="module load apps/2021 Python/3.10.8-GCCcore-12.2.0 && cd $(pwd) && python3 ${SCRIPT_DIR}/step6_dynamics_worker.py \
                --corpus_key ${CORPUS_NAME} \
                --job_id \${SLURM_ARRAY_TASK_ID} \
                --total_jobs ${DYNAMICS_JOBS} \
                --num_trajectories ${DYNAMICS_TRAJECTORIES} \
                --max_steps ${DYNAMICS_MAX_STEPS}")

        if [ $? -ne 0 ]; then
            echo "  ✗ FAILED to submit dynamics for ${CORPUS_NAME}"
            PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
            continue
        fi
        echo "    Dynamics Job ID: $JID_DYN"

        wait_for_slurm_job "$JID_DYN" "Dynamics for ${CORPUS_NAME}" "$DYNAMICS_JOBS" \
            "${DYNAMICS_DIR}/dynamics_${CORPUS_NAME}_job_*.done"
        if [ $? -ne 0 ]; then
            echo "  ⚠ WARNING: Some dynamics jobs may have failed for ${CORPUS_NAME}."
            PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
        fi

        echo "  Merging aggregated dynamics results..."
        STATS_FINAL="${DYNAMICS_DIR}/dynamics_${CORPUS_NAME}_stats_final.csv"
        HIST_FINAL="${DYNAMICS_DIR}/dynamics_${CORPUS_NAME}_hist_final.csv"

        FIRST_STATS=$(ls ${DYNAMICS_DIR}/dynamics_${CORPUS_NAME}_job_*_stats.csv 2>/dev/null | head -1)
        FIRST_HIST=$(ls ${DYNAMICS_DIR}/dynamics_${CORPUS_NAME}_job_*_hist.csv 2>/dev/null | head -1)

        if [ -n "$FIRST_STATS" ]; then
            head -1 "$FIRST_STATS" > "$STATS_FINAL"
            tail -q -n +2 ${DYNAMICS_DIR}/dynamics_${CORPUS_NAME}_job_*_stats.csv >> "$STATS_FINAL"
            STATS_LINES=$(($(wc -l < "$STATS_FINAL") - 1))
            echo "  ✓ Stats merged: ${STATS_LINES} rows in ${STATS_FINAL}"

            head -1 "$FIRST_HIST" > "$HIST_FINAL"
            tail -q -n +2 ${DYNAMICS_DIR}/dynamics_${CORPUS_NAME}_job_*_hist.csv >> "$HIST_FINAL"
            HIST_LINES=$(($(wc -l < "$HIST_FINAL") - 1))
            echo "  ✓ Histograms merged: ${HIST_LINES} rows in ${HIST_FINAL}"

            rm -f ${DYNAMICS_DIR}/dynamics_${CORPUS_NAME}_job_*_stats.csv \
                  ${DYNAMICS_DIR}/dynamics_${CORPUS_NAME}_job_*_hist.csv \
                  ${DYNAMICS_DIR}/dynamics_${CORPUS_NAME}_job_*.done 2>/dev/null || true
        else
            echo "  ✗ No dynamics output files found for ${CORPUS_NAME}."
            PIPELINE_ERRORS=$((PIPELINE_ERRORS + 1))
        fi

        echo ""
    done

    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  PHASE 6 COMPLETE"
    echo "═══════════════════════════════════════════════════════════════════════════════"
fi

# ==============================================================================
#                           FINAL SUMMARY
# ==============================================================================

PIPELINE_END=$(date +%s)
PIPELINE_ELAPSED=$((PIPELINE_END - PIPELINE_START))

print_banner "PIPELINE COMPLETE"

echo "Total elapsed time: $(format_elapsed $PIPELINE_ELAPSED)"
echo "Errors encountered: ${PIPELINE_ERRORS}"
echo ""

echo "Results:"
for corpus in "${ALL_CORPORA[@]}"; do
    if [ -f "$corpus" ]; then
        cname=$(basename "$corpus" .txt)
        rdir="results_${cname}"
        if [ -f "${rdir}/pairs_results.txt" ]; then
            pairs=$(wc -l < "${rdir}/pairs_results.txt")
            god=$(wc -l < "${rdir}/single_god_results.txt" 2>/dev/null || echo 0)
            echo "  ${cname}: ${pairs} pairs, ${god} GOD scores"
        else
            echo "  ${cname}: no results"
        fi
    fi
done
echo ""

if [ -f "eod_results_final.csv" ]; then
    eod_lines=$(wc -l < "eod_results_final.csv")
    echo "  EOD: ${eod_lines} lines"
fi
if [ -d "pairwise_diff_results" ]; then
    diff_count=$(ls pairwise_diff_results/diffs_*.csv 2>/dev/null | wc -l)
    echo "  Pairwise diffs: ${diff_count}/15 files"
fi
if [ -d "dynamics_results" ]; then
    dyn_stats=$(ls dynamics_results/dynamics_*_stats_final.csv 2>/dev/null | wc -l)
    dyn_hist=$(ls dynamics_results/dynamics_*_hist_final.csv 2>/dev/null | wc -l)
    echo "  Dynamics: ${dyn_stats}/6 stats files, ${dyn_hist}/6 histogram files"
fi

echo ""
echo "To download results for local analysis:"
echo "  scp -r w007104@magerit.cesvima.upm.es:~/od_semantics_project/results_* ."
echo "  scp w007104@magerit.cesvima.upm.es:~/od_semantics_project/eod_results_final.csv ."
echo "  scp -r w007104@magerit.cesvima.upm.es:~/od_semantics_project/pairwise_diff_results ."
echo "  scp -r w007104@magerit.cesvima.upm.es:~/od_semantics_project/dynamics_results ."
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"

exit $PIPELINE_ERRORS
