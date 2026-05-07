#!/bin/bash
#
# Nemotron-CC-HQ-diverse_qa_pairs Preprocessing Script
# Converts zstd-compressed JSONL files to Megatron binary format (.bin/.idx)
#
# Dataset Info:
#   - Files: 2,564 .jsonl.zstd files
#   - Size: ~814 GB compressed (~2.3 TB uncompressed)
#   - JSON field: "text"
#
# Usage:
#   bash preprocess_nemotron_qa_pairs.sh [workers]
#
# Examples:
#   bash preprocess_nemotron_qa_pairs.sh          # Default: 64 workers
#   bash preprocess_nemotron_qa_pairs.sh 32       # 32 workers
#
# Note: Uses preprocess_data.py with lm_dataformat which natively supports
#       .jsonl.zstd files (streaming decompression)

set -e  # Exit on error

# ==================== Configuration ====================

START_TIME=$SECONDS

# Paths
MEGATRON_PATCH_PATH="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch"
MEGATRON_PATH="${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250624"
TOKENIZER_PATH="${MEGATRON_PATCH_PATH}/examples/alpha/tokenizer_v5"

# Input/Output paths
INPUT_DIR="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/datasets/raw/KORMo/pretraining/stage2/eng/Nemotron-CC-HQ-diverse_qa_pairs"
OUTPUT_DIR="/home/work/Datasets/KORMo_processed/mmap/nemotron_cc_hq/qa_pairs"
OUTPUT_PREFIX="${OUTPUT_DIR}/nemotron_qa_pairs"

# Parameters
WORKERS=${1:-64}      # Default: 64 workers

# ==================== Validation ====================

echo "================================================================================"
echo "Nemotron-CC-HQ-diverse_qa_pairs Preprocessing"
echo "================================================================================"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Workers: ${WORKERS}"
echo "Megatron version: Megatron-LM-250624"
echo "Preprocessing script: preprocess_data.py (lm_dataformat)"
echo "Input format: .jsonl.zstd (zstd compressed, native support)"
echo "================================================================================"
echo

# Check if input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: Input directory not found: ${INPUT_DIR}"
    exit 1
fi

# Check if tokenizer exists
if [ ! -d "${TOKENIZER_PATH}" ]; then
    echo "Error: Tokenizer not found: ${TOKENIZER_PATH}"
    exit 1
fi

# Check if Megatron path exists
if [ ! -d "${MEGATRON_PATH}" ]; then
    echo "Error: Megatron path not found: ${MEGATRON_PATH}"
    exit 1
fi

# Count input files (support both .jsonl.zstd and .jsonl.zst)
INPUT_FILE_COUNT=$(find "${INPUT_DIR}" \( -name "*.jsonl.zstd" -o -name "*.jsonl.zst" \) -type f 2>/dev/null | wc -l)
if [ ${INPUT_FILE_COUNT} -eq 0 ]; then
    echo "Error: No zstd-compressed JSONL files found in ${INPUT_DIR}"
    echo "Expected: *.jsonl.zstd or *.jsonl.zst files"
    exit 1
fi

echo "Found ${INPUT_FILE_COUNT} zstd-compressed JSONL files in input directory"
echo

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check if output already exists
if [ -f "${OUTPUT_PREFIX}_text_document.bin" ] && [ -f "${OUTPUT_PREFIX}_text_document.idx" ]; then
    echo "Warning: Output files already exist:"
    echo "  - ${OUTPUT_PREFIX}_text_document.bin"
    echo "  - ${OUTPUT_PREFIX}_text_document.idx"
    echo
    read -p "Do you want to overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    echo "Removing existing files..."
    rm -f "${OUTPUT_PREFIX}_text_document.bin"
    rm -f "${OUTPUT_PREFIX}_text_document.idx"
fi

# ==================== Preprocessing ====================

echo
echo "================================================================================"
echo "Starting preprocessing..."
echo "Start time: $(date)"
echo "Note: zstd files will be decompressed on-the-fly (streaming)"
echo "================================================================================"
echo

# Set Python path
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH

# Set NUMEXPR_MAX_THREADS to avoid warning
export NUMEXPR_MAX_THREADS=${WORKERS}

# Change to preprocessing directory
cd "${MEGATRON_PATCH_PATH}/toolkits/pretrain_data_preprocessing"

# Show sample input files
echo "Sample input files:"
find "${INPUT_DIR}" \( -name "*.jsonl.zstd" -o -name "*.jsonl.zst" \) -type f | sort | head -5
if [ ${INPUT_FILE_COUNT} -gt 5 ]; then
    echo "... and $(($INPUT_FILE_COUNT - 5)) more files"
fi
echo

# Run preprocessing with lm_dataformat (native zstd support, document-level parallelism)
python preprocess_data.py \
  --input "${INPUT_DIR}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --dataset-impl mmap \
  --patch-tokenizer-type Qwen2Tokenizer \
  --load "${TOKENIZER_PATH}" \
  --workers ${WORKERS} \
  --jsonl-keys text \
  --append-eod \
  --log-interval 10000 \
  --extra-vocab-size 0

ELAPSED_TIME=$(($SECONDS - $START_TIME))

# ==================== Summary ====================

echo
echo "================================================================================"
echo "Preprocessing Complete!"
echo "================================================================================"
echo "End time: $(date)"
echo "Time elapsed: $(($ELAPSED_TIME/3600))h $(($ELAPSED_TIME%3600/60))m $(($ELAPSED_TIME%60))s"
echo
echo "Output files:"
if [ -f "${OUTPUT_PREFIX}_text_document.bin" ]; then
    BIN_SIZE=$(du -h "${OUTPUT_PREFIX}_text_document.bin" | cut -f1)
    echo "  ${OUTPUT_PREFIX}_text_document.bin (${BIN_SIZE})"
else
    echo "  ${OUTPUT_PREFIX}_text_document.bin (NOT FOUND)"
fi

if [ -f "${OUTPUT_PREFIX}_text_document.idx" ]; then
    IDX_SIZE=$(du -h "${OUTPUT_PREFIX}_text_document.idx" | cut -f1)
    echo "  ${OUTPUT_PREFIX}_text_document.idx (${IDX_SIZE})"
else
    echo "  ${OUTPUT_PREFIX}_text_document.idx (NOT FOUND)"
fi
echo "================================================================================"
echo

# Return to original directory
cd "${MEGATRON_PATCH_PATH}"
