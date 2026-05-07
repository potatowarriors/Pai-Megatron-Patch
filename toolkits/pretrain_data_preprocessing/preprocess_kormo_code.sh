#!/bin/bash
#
# KORMo Code Dataset Preprocessing Script
# Converts Nemotron-Pretraining-Code-v2 JSONL files to Megatron binary format (.bin/.idx)
# Each subset is processed independently for blended training.
#
# Usage:
#   bash preprocess_kormo_code.sh [workers] [subsets...]
#
# Examples:
#   bash preprocess_kormo_code.sh              # Process all 5 subsets with 64 workers
#   bash preprocess_kormo_code.sh 32           # Process all with 32 workers
#   bash preprocess_kormo_code.sh 64 code_review question_answering  # Specific subsets only
#
# Output structure:
#   qwen3_code/
#   ├── code_review/
#   │   ├── code_review_content_document.bin
#   │   └── code_review_content_document.idx
#   ├── question_answering/
#   │   ├── question_answering_content_document.bin
#   │   └── question_answering_content_document.idx
#   ├── rewriting/
#   │   ├── rewriting_content_document.bin
#   │   └── rewriting_content_document.idx
#   ├── student_teacher/
#   │   ├── student_teacher_content_document.bin
#   │   └── student_teacher_content_document.idx
#   └── transpilation/
#       ├── transpilation_content_document.bin
#       └── transpilation_content_document.idx
#

set -e

# ==================== Configuration ====================

START_TIME=$SECONDS

# Paths
MEGATRON_PATCH_PATH="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch"
MEGATRON_PATH="${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250624"
TOKENIZER_PATH="${MEGATRON_PATCH_PATH}/examples/alpha/tokenizer_v5"

# Data paths
INPUT_BASE="${MEGATRON_PATCH_PATH}/datasets/raw/KORMo/pretraining/stage2/code"
OUTPUT_BASE="${MEGATRON_PATCH_PATH}/datasets/processed/qwen3_code"

# Subset mapping: output_name -> input_directory_name
declare -A SUBSET_MAP=(
    ["code_review"]="synthetic-code-review"
    ["question_answering"]="synthetic-question-answering"
    ["rewriting"]="synthetic-rewriting"
    ["student_teacher"]="synthetic-student-teacher"
    ["transpilation"]="synthetic-transpilation"
)

# All subset names (processing order: smallest first for quick validation)
ALL_SUBSETS=(transpilation student_teacher code_review rewriting question_answering)

# Parameters
WORKERS=${1:-64}

# Parse subset arguments (if provided after workers)
if [ $# -gt 1 ]; then
    shift  # Remove workers arg
    SUBSETS=("$@")
else
    SUBSETS=("${ALL_SUBSETS[@]}")
fi

# ==================== Validation ====================

echo "================================================================================"
echo "KORMo Code Dataset Preprocessing"
echo "================================================================================"
echo "Input base:  ${INPUT_BASE}"
echo "Output base: ${OUTPUT_BASE}"
echo "Tokenizer:   ${TOKENIZER_PATH}"
echo "Workers:     ${WORKERS}"
echo "Subsets:     ${SUBSETS[*]}"
echo "================================================================================"
echo

# Validate paths
if [ ! -d "${INPUT_BASE}" ]; then
    echo "Error: Input directory not found: ${INPUT_BASE}"
    exit 1
fi

if [ ! -d "${TOKENIZER_PATH}" ]; then
    echo "Error: Tokenizer not found: ${TOKENIZER_PATH}"
    exit 1
fi

if [ ! -d "${MEGATRON_PATH}" ]; then
    echo "Error: Megatron path not found: ${MEGATRON_PATH}"
    exit 1
fi

# Validate subset names
for subset in "${SUBSETS[@]}"; do
    if [ -z "${SUBSET_MAP[$subset]}" ]; then
        echo "Error: Unknown subset '${subset}'"
        echo "Valid subsets: ${!SUBSET_MAP[*]}"
        exit 1
    fi
    INPUT_DIR="${INPUT_BASE}/${SUBSET_MAP[$subset]}"
    if [ ! -d "${INPUT_DIR}" ]; then
        echo "Error: Input directory not found: ${INPUT_DIR}"
        exit 1
    fi
done

# Set Python path
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
export NUMEXPR_MAX_THREADS=${WORKERS}

cd "${MEGATRON_PATCH_PATH}/toolkits/pretrain_data_preprocessing"

# ==================== Process Each Subset ====================

TOTAL_SUBSETS=${#SUBSETS[@]}
CURRENT=0
RESULTS=()

for subset in "${SUBSETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    SUBSET_START=$SECONDS

    INPUT_DIR_NAME="${SUBSET_MAP[$subset]}"
    INPUT_DIR="${INPUT_BASE}/${INPUT_DIR_NAME}"
    OUTPUT_DIR="${OUTPUT_BASE}/${subset}"
    OUTPUT_PREFIX="${OUTPUT_DIR}/${subset}"

    FILE_COUNT=$(ls "${INPUT_DIR}"/*.jsonl 2>/dev/null | wc -l)

    echo
    echo "================================================================================"
    echo "[${CURRENT}/${TOTAL_SUBSETS}] Processing: ${subset}"
    echo "================================================================================"
    echo "  Input:  ${INPUT_DIR} (${FILE_COUNT} files)"
    echo "  Output: ${OUTPUT_PREFIX}_content_document.{bin,idx}"
    echo

    # Check if output already exists
    if [ -f "${OUTPUT_PREFIX}_content_document.bin" ] && [ -f "${OUTPUT_PREFIX}_content_document.idx" ]; then
        BIN_SIZE=$(du -h "${OUTPUT_PREFIX}_content_document.bin" | cut -f1)
        echo "  Already exists (${BIN_SIZE}). Skipping."
        echo "  (Use --force or delete output files to reprocess)"
        SUBSET_TIME=$((SECONDS - SUBSET_START))
        RESULTS+=("${subset}: SKIPPED (already exists, ${BIN_SIZE})")
        continue
    fi

    # Create output directory
    mkdir -p "${OUTPUT_DIR}"

    # Run preprocessing
    echo "  Starting tokenization..."
    python preprocess_data.py \
      --input "${INPUT_DIR}" \
      --output-prefix ${OUTPUT_PREFIX} \
      --dataset-impl mmap \
      --patch-tokenizer-type Qwen2Tokenizer \
      --load ${TOKENIZER_PATH} \
      --workers ${WORKERS} \
      --append-eod \
      --log-interval 10000 \
      --extra-vocab-size 0

    SUBSET_TIME=$((SECONDS - SUBSET_START))

    # Verify output
    if [ -f "${OUTPUT_PREFIX}_content_document.bin" ] && [ -f "${OUTPUT_PREFIX}_content_document.idx" ]; then
        BIN_SIZE=$(du -h "${OUTPUT_PREFIX}_content_document.bin" | cut -f1)
        IDX_SIZE=$(du -h "${OUTPUT_PREFIX}_content_document.idx" | cut -f1)
        echo
        echo "  Done: ${BIN_SIZE} bin, ${IDX_SIZE} idx ($(($SUBSET_TIME/60))m $(($SUBSET_TIME%60))s)"
        RESULTS+=("${subset}: OK (${BIN_SIZE}, $(($SUBSET_TIME/60))m $(($SUBSET_TIME%60))s)")
    else
        echo
        echo "  ERROR: Output files not created!"
        RESULTS+=("${subset}: FAILED")
    fi
done

TOTAL_TIME=$((SECONDS - START_TIME))

# ==================== Summary ====================

echo
echo "================================================================================"
echo "Preprocessing Complete!"
echo "================================================================================"
echo "Total time: $(($TOTAL_TIME/3600))h $(($TOTAL_TIME%3600/60))m $(($TOTAL_TIME%60))s"
echo
echo "Results:"
for result in "${RESULTS[@]}"; do
    echo "  ${result}"
done

echo
echo "Output directory:"
if [ -d "${OUTPUT_BASE}" ]; then
    for subset in "${ALL_SUBSETS[@]}"; do
        BIN_FILE="${OUTPUT_BASE}/${subset}/${subset}_content_document.bin"
        if [ -f "$BIN_FILE" ]; then
            SIZE=$(du -h "$BIN_FILE" | cut -f1)
            echo "  ${subset}: ${SIZE}"
        else
            echo "  ${subset}: (not processed)"
        fi
    done
fi

echo
echo "================================================================================"
echo "Usage in training (data config YAML):"
echo "================================================================================"
echo "data:"
echo "  blend:"
for subset in "${ALL_SUBSETS[@]}"; do
    echo "    - path: \"${OUTPUT_BASE}/${subset}/${subset}_content_document\""
done
echo "  dataset_impl: \"MMAP\""
echo "  split: [99, 1, 0]"
echo "================================================================================"
