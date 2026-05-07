#!/bin/bash
#
# KORMo Subset Preprocessing Script
# Converts JSONL subsets to Megatron binary format (.bin/.idx)
#
# Usage:
#   bash preprocess_kormo_subset.sh <subset_percent> [workers]
#
# Examples:
#   bash preprocess_kormo_subset.sh 1     # Process 1% subset with 64 workers
#   bash preprocess_kormo_subset.sh 10    # Process 10% subset with 64 workers
#   bash preprocess_kormo_subset.sh 1 32  # Process 1% subset with 32 workers
#

set -e  # Exit on error

# ==================== Configuration ====================

START_TIME=$SECONDS

# Paths
MEGATRON_PATCH_PATH="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch"
MEGATRON_PATH="${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250624"
TOKENIZER_PATH="${MEGATRON_PATCH_PATH}/examples/alpha/tokenizer_v5"

# Data paths
DATA_BASE="/home/work/Datasets/KORMo_processed"
INPUT_BASE="${DATA_BASE}/intermediate"
OUTPUT_BASE="${DATA_BASE}/mmap"

# Parameters
SUBSET_PCT=${1:-1}  # Default: 1%
WORKERS=${2:-64}    # Default: 64 workers

# Derived paths
SUBSET_NAME="subset_${SUBSET_PCT}pct"
INPUT_DIR="${INPUT_BASE}/${SUBSET_NAME}"
OUTPUT_DIR="${OUTPUT_BASE}/qwen3_${SUBSET_PCT}pct"
OUTPUT_PREFIX="${OUTPUT_DIR}/kormo"

# ==================== Validation ====================

echo "================================================================================"
echo "KORMo Subset Preprocessing"
echo "================================================================================"
echo "Subset: ${SUBSET_PCT}%"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Workers: ${WORKERS}"
echo "Megatron version: Megatron-LM-250624"
echo "================================================================================"
echo

# Check if input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "❌ Error: Input directory not found: ${INPUT_DIR}"
    echo "Please run subset generation first."
    exit 1
fi

# Check if tokenizer exists
if [ ! -d "${TOKENIZER_PATH}" ]; then
    echo "❌ Error: Tokenizer not found: ${TOKENIZER_PATH}"
    echo "Please run: python3 download_qwen3_tokenizer.py"
    exit 1
fi

# Check if Megatron path exists
if [ ! -d "${MEGATRON_PATH}" ]; then
    echo "❌ Error: Megatron path not found: ${MEGATRON_PATH}"
    exit 1
fi

# Count input files
INPUT_FILE_COUNT=$(find "${INPUT_DIR}" -name "*.jsonl" -type f | wc -l)
if [ ${INPUT_FILE_COUNT} -eq 0 ]; then
    echo "❌ Error: No JSONL files found in ${INPUT_DIR}"
    exit 1
fi

echo "✓ Found ${INPUT_FILE_COUNT} JSONL files in input directory"
echo

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check if output already exists
if [ -f "${OUTPUT_PREFIX}_text_document.bin" ] && [ -f "${OUTPUT_PREFIX}_text_document.idx" ]; then
    echo "⚠️  Warning: Output files already exist:"
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
echo "================================================================================"
echo

# Set Python path
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH

# Set NUMEXPR_MAX_THREADS to avoid warning
export NUMEXPR_MAX_THREADS=${WORKERS}

# Change to preprocessing directory
cd "${MEGATRON_PATCH_PATH}/toolkits/pretrain_data_preprocessing"

# Show input files for verification
echo "Input directory: ${INPUT_DIR}"
echo "Input files in directory:"
find "${INPUT_DIR}" -name "*.jsonl" -type f | sort | head -5
if [ ${INPUT_FILE_COUNT} -gt 5 ]; then
    echo "... and $(($INPUT_FILE_COUNT - 5)) more files"
fi
echo

# Run preprocessing
# Note: preprocess_data.py expects a DIRECTORY path, not individual files
# It will automatically read all JSONL files from the directory
# --extra-vocab-size 0: Use base vocabulary only (no padding needed for preprocessing)
# Padding to 151936 will be applied during training for GPU optimization
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

ELAPSED_TIME=$(($SECONDS - $START_TIME))

# ==================== Summary ====================

echo
echo "================================================================================"
echo "✅ Preprocessing Complete!"
echo "================================================================================"
echo "Time elapsed: $(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
echo
echo "Output files:"
if [ -f "${OUTPUT_PREFIX}_content_document.bin" ]; then
    BIN_SIZE=$(du -h "${OUTPUT_PREFIX}_content_document.bin" | cut -f1)
    echo "  ✓ ${OUTPUT_PREFIX}_content_document.bin (${BIN_SIZE})"
else
    echo "  ❌ ${OUTPUT_PREFIX}_content_document.bin (NOT FOUND)"
fi

if [ -f "${OUTPUT_PREFIX}_content_document.idx" ]; then
    IDX_SIZE=$(du -h "${OUTPUT_PREFIX}_content_document.idx" | cut -f1)
    echo "  ✓ ${OUTPUT_PREFIX}_content_document.idx (${IDX_SIZE})"
else
    echo "  ❌ ${OUTPUT_PREFIX}_content_document.idx (NOT FOUND)"
fi
echo "================================================================================"
echo

# Return to original directory
cd "${MEGATRON_PATCH_PATH}"
