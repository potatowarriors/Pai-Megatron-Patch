#!/bin/bash
#
# Korean Web Half2 Preprocessing Script
# Tokenizes the second half of korean_web.jsonl (~20M lines, ~55B tokens)
# for use as an independent mmap dataset in stage2 blended training.
#
# Prerequisites:
#   korean_web_half2.jsonl must already exist (created via tail from korean_web.jsonl)
#
# Usage:
#   bash preprocess_koreanweb_half.sh [workers]
#
# Output:
#   /home/work/Datasets/LL_preprocessed/mmap/koreanweb_half/
#   ├── korean_web_half_text_document.bin
#   └── korean_web_half_text_document.idx
#

set -e

# ==================== Configuration ====================

START_TIME=$SECONDS

# Paths
MEGATRON_PATCH_PATH="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch"
MEGATRON_PATH="${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-251125"
TOKENIZER_PATH="${MEGATRON_PATCH_PATH}/examples/alpha/tokenizer_v5"

# Data paths
INPUT_FILE="/home/work/Datasets/LL_datasets/pretraining/stage1/korean_web_half2.jsonl"
OUTPUT_DIR="/home/work/Datasets/LL_preprocessed/mmap/koreanweb_half"
OUTPUT_PREFIX="${OUTPUT_DIR}/korean_web_half"

# Parameters
WORKERS=${1:-64}

# ==================== Validation ====================

echo "================================================================================"
echo "Korean Web Half2 Preprocessing"
echo "================================================================================"
echo "Input:     ${INPUT_FILE}"
echo "Output:    ${OUTPUT_PREFIX}_text_document.{bin,idx}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Megatron:  ${MEGATRON_PATH}"
echo "Workers:   ${WORKERS}"
echo "================================================================================"
echo

if [ ! -f "${INPUT_FILE}" ]; then
    echo "Error: Input file not found: ${INPUT_FILE}"
    echo "Create it first with:"
    echo "  tail -n 20149029 /home/work/Datasets/LL_datasets/pretraining/stage1/korean_web.jsonl > ${INPUT_FILE}"
    exit 1
fi

if [ ! -d "${TOKENIZER_PATH}" ]; then
    echo "Error: Tokenizer not found: ${TOKENIZER_PATH}"
    exit 1
fi

if [ ! -d "${MEGATRON_PATH}" ]; then
    echo "Error: Megatron backend not found: ${MEGATRON_PATH}"
    exit 1
fi

INPUT_SIZE=$(du -h "${INPUT_FILE}" | cut -f1)
echo "Input file size: ${INPUT_SIZE}"
echo

# Set Python path
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
export NUMEXPR_MAX_THREADS=${WORKERS}

cd "${MEGATRON_PATCH_PATH}/toolkits/pretrain_data_preprocessing"

# ==================== Check Existing Output ====================

if [ -f "${OUTPUT_PREFIX}_text_document.bin" ] && [ -f "${OUTPUT_PREFIX}_text_document.idx" ]; then
    BIN_SIZE=$(du -h "${OUTPUT_PREFIX}_text_document.bin" | cut -f1)
    echo "Warning: Output already exists (${BIN_SIZE}). Skipping."
    echo "Delete existing files to reprocess."
    exit 0
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# ==================== Run Preprocessing ====================

echo "Starting tokenization..."
echo

python preprocess_data_megatron.py \
  --input "${INPUT_FILE}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --json-keys text \
  --patch-tokenizer-type Qwen3Tokenizer \
  --load "${TOKENIZER_PATH}" \
  --workers ${WORKERS} \
  --append-eod

TOTAL_TIME=$((SECONDS - START_TIME))

# ==================== Summary ====================

echo
echo "================================================================================"
echo "Preprocessing Complete!"
echo "================================================================================"
echo "Total time: $(($TOTAL_TIME/3600))h $(($TOTAL_TIME%3600/60))m $(($TOTAL_TIME%60))s"
echo

if [ -f "${OUTPUT_PREFIX}_text_document.bin" ] && [ -f "${OUTPUT_PREFIX}_text_document.idx" ]; then
    BIN_SIZE=$(du -h "${OUTPUT_PREFIX}_text_document.bin" | cut -f1)
    IDX_SIZE=$(du -h "${OUTPUT_PREFIX}_text_document.idx" | cut -f1)
    BIN_BYTES=$(stat --format=%s "${OUTPUT_PREFIX}_text_document.bin")
    TOKEN_COUNT=$((BIN_BYTES / 2))

    echo "Output files:"
    echo "  ${OUTPUT_PREFIX}_text_document.bin (${BIN_SIZE})"
    echo "  ${OUTPUT_PREFIX}_text_document.idx (${IDX_SIZE})"
    echo
    echo "Estimated tokens: ${TOKEN_COUNT} (~$((TOKEN_COUNT / 1000000000))B)"
    echo
    echo "================================================================================"
    echo "Usage in stage2 data config YAML:"
    echo "================================================================================"
    echo "  - path: \"${OUTPUT_DIR}/korean_web_half_text_document\""
    echo "    weight: <blend_weight>"
    echo "================================================================================"
else
    echo "ERROR: Output files not created!"
    exit 1
fi
