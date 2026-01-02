#!/bin/bash
#
# KORMo 50% Blended Dataset Preprocessing Script
# Creates separate mmap files for DCLM and Korean Web for blended training
#
# Usage:
#   bash preprocess_kormo_50pct_blended.sh [workers]
#
# Output structure:
#   qwen3_50pct/
#   ├── dclm/
#   │   ├── dclm_content_document.bin
#   │   └── dclm_content_document.idx
#   └── korean_web/
#       ├── korean_web_content_document.bin
#       └── korean_web_content_document.idx
#

set -e

# ==================== Configuration ====================

START_TIME=$SECONDS

# Paths
MEGATRON_PATCH_PATH="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch"
MEGATRON_PATH="${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250624"
TOKENIZER_PATH="${MEGATRON_PATCH_PATH}/models/Qwen3-Next-tokenizer"

# Data paths
DATA_BASE="/home/work/Datasets/KORMo_processed"
INPUT_DIR="${DATA_BASE}/intermediate/subset_50pct"
OUTPUT_BASE="${DATA_BASE}/mmap/qwen3_50pct"

# Parameters
WORKERS=${1:-64}

# ==================== Validation ====================

echo "================================================================================"
echo "KORMo 50% Blended Dataset Preprocessing"
echo "================================================================================"
echo "Input directory: ${INPUT_DIR}"
echo "Output base: ${OUTPUT_BASE}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Workers: ${WORKERS}"
echo "================================================================================"
echo

# Check directories
if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: Input directory not found: ${INPUT_DIR}"
    echo "Please run create_50pct_subset.sh first."
    exit 1
fi

if [ ! -d "${TOKENIZER_PATH}" ]; then
    echo "Error: Tokenizer not found: ${TOKENIZER_PATH}"
    exit 1
fi

# Set Python path
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
export NUMEXPR_MAX_THREADS=${WORKERS}

cd "${MEGATRON_PATCH_PATH}/toolkits/pretrain_data_preprocessing"

# ==================== Process DCLM ====================

echo
echo "================================================================================"
echo "[1/2] Processing DCLM (8 shards)"
echo "================================================================================"

DCLM_INPUT_DIR="${INPUT_DIR}/dclm_temp"
DCLM_OUTPUT_DIR="${OUTPUT_BASE}/dclm"
DCLM_OUTPUT_PREFIX="${DCLM_OUTPUT_DIR}/dclm"

# Create temp directory with only DCLM files
mkdir -p "${DCLM_INPUT_DIR}"
rm -f "${DCLM_INPUT_DIR}"/*.jsonl 2>/dev/null || true

echo "Creating DCLM symlinks..."
for f in "${INPUT_DIR}"/dclm_shard_*.jsonl; do
    ln -sf "$f" "${DCLM_INPUT_DIR}/"
done

DCLM_COUNT=$(ls "${DCLM_INPUT_DIR}"/*.jsonl 2>/dev/null | wc -l)
echo "Found ${DCLM_COUNT} DCLM shard files"

# Create output directory
mkdir -p "${DCLM_OUTPUT_DIR}"

# Check if output already exists
if [ -f "${DCLM_OUTPUT_PREFIX}_content_document.bin" ]; then
    echo "Warning: DCLM output already exists. Skipping..."
else
    echo "Starting DCLM preprocessing..."
    python preprocess_data.py \
      --input "${DCLM_INPUT_DIR}" \
      --output-prefix ${DCLM_OUTPUT_PREFIX} \
      --dataset-impl mmap \
      --patch-tokenizer-type Qwen2Tokenizer \
      --load ${TOKENIZER_PATH} \
      --workers ${WORKERS} \
      --append-eod \
      --log-interval 10000 \
      --extra-vocab-size 0
fi

# Cleanup temp directory
rm -rf "${DCLM_INPUT_DIR}"

DCLM_TIME=$(($SECONDS - $START_TIME))
echo "DCLM processing time: $(($DCLM_TIME/60)) min $(($DCLM_TIME%60)) sec"

# ==================== Process Korean Web ====================

echo
echo "================================================================================"
echo "[2/2] Processing Korean Web"
echo "================================================================================"

KW_INPUT_DIR="${INPUT_DIR}/korean_web_temp"
KW_OUTPUT_DIR="${OUTPUT_BASE}/korean_web"
KW_OUTPUT_PREFIX="${KW_OUTPUT_DIR}/korean_web"

# Create temp directory with only Korean Web file
mkdir -p "${KW_INPUT_DIR}"
rm -f "${KW_INPUT_DIR}"/*.jsonl 2>/dev/null || true

if [ -f "${INPUT_DIR}/korean_web.jsonl" ]; then
    ln -sf "${INPUT_DIR}/korean_web.jsonl" "${KW_INPUT_DIR}/"
    echo "Found korean_web.jsonl"
else
    echo "Error: korean_web.jsonl not found in ${INPUT_DIR}"
    exit 1
fi

# Create output directory
mkdir -p "${KW_OUTPUT_DIR}"

# Check if output already exists
if [ -f "${KW_OUTPUT_PREFIX}_content_document.bin" ]; then
    echo "Warning: Korean Web output already exists. Skipping..."
else
    echo "Starting Korean Web preprocessing..."
    KW_START=$SECONDS
    python preprocess_data.py \
      --input "${KW_INPUT_DIR}" \
      --output-prefix ${KW_OUTPUT_PREFIX} \
      --dataset-impl mmap \
      --patch-tokenizer-type Qwen2Tokenizer \
      --load ${TOKENIZER_PATH} \
      --workers ${WORKERS} \
      --append-eod \
      --log-interval 10000 \
      --extra-vocab-size 0
fi

# Cleanup temp directory
rm -rf "${KW_INPUT_DIR}"

TOTAL_TIME=$(($SECONDS - $START_TIME))

# ==================== Summary ====================

echo
echo "================================================================================"
echo "Preprocessing Complete!"
echo "================================================================================"
echo "Total time: $(($TOTAL_TIME/60)) min $(($TOTAL_TIME%60)) sec"
echo
echo "Output files:"
echo
echo "DCLM:"
if [ -f "${DCLM_OUTPUT_PREFIX}_content_document.bin" ]; then
    DCLM_BIN_SIZE=$(du -h "${DCLM_OUTPUT_PREFIX}_content_document.bin" | cut -f1)
    echo "  ${DCLM_OUTPUT_PREFIX}_content_document.bin (${DCLM_BIN_SIZE})"
    echo "  ${DCLM_OUTPUT_PREFIX}_content_document.idx"
else
    echo "  (not found)"
fi
echo
echo "Korean Web:"
if [ -f "${KW_OUTPUT_PREFIX}_content_document.bin" ]; then
    KW_BIN_SIZE=$(du -h "${KW_OUTPUT_PREFIX}_content_document.bin" | cut -f1)
    echo "  ${KW_OUTPUT_PREFIX}_content_document.bin (${KW_BIN_SIZE})"
    echo "  ${KW_OUTPUT_PREFIX}_content_document.idx"
else
    echo "  (not found)"
fi
echo
echo "================================================================================"
echo "Usage in training:"
echo "================================================================================"
echo "--data-path \\"
echo "  0.92 ${DCLM_OUTPUT_DIR}/dclm_content_document \\"
echo "  0.08 ${KW_OUTPUT_DIR}/korean_web_content_document"
echo "================================================================================"
