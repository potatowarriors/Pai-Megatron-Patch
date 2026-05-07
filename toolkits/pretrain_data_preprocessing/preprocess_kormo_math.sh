#!/bin/bash
#
# KORMo Math Dataset Preprocessing Script
# Combines 3 subsets (3, 4plus, 4plus_MIND) into single .bin/.idx
#
# Dataset Info:
#   - Subsets: 3 (57 files, ~283GB), 4plus (46 files, ~173GB), 4plus_MIND (90 files, ~262GB)
#   - Total: 193 .jsonl files (~718GB)
#   - JSON field: "text"
#
# Problem: preprocess_data.py only scans 1 level deep (os.listdir),
#          so we create a flat symlink directory combining all subsets.
#
# Usage:
#   bash preprocess_kormo_math.sh [workers]
#
# Examples:
#   bash preprocess_kormo_math.sh          # Default: 64 workers
#   bash preprocess_kormo_math.sh 32       # 32 workers

set -e  # Exit on error

# ==================== Configuration ====================

START_TIME=$SECONDS

# Paths
MEGATRON_PATCH_PATH="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch"
MEGATRON_PATH="${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250624"
TOKENIZER_PATH="${MEGATRON_PATCH_PATH}/examples/alpha/tokenizer_v5"

# Input/Output paths
INPUT_BASE="/home/work/Datasets/KORMo/pretraining/stage2/math"
SUBSETS=("3" "4plus" "4plus_MIND")
COMBINED_DIR="/tmp/kormo_math_combined"
OUTPUT_DIR="/home/work/Datasets/KORMo_processed/mmap/math"
OUTPUT_PREFIX="${OUTPUT_DIR}/math"

# Parameters
WORKERS=${1:-64}      # Default: 64 workers

# ==================== Validation ====================

echo "================================================================================"
echo "KORMo Math Dataset Preprocessing"
echo "================================================================================"
echo "Input base: ${INPUT_BASE}"
echo "Subsets: ${SUBSETS[*]}"
echo "Combined symlink dir: ${COMBINED_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "Tokenizer: ${TOKENIZER_PATH}"
echo "Workers: ${WORKERS}"
echo "Megatron version: Megatron-LM-250624"
echo "Preprocessing script: preprocess_data.py (lm_dataformat)"
echo "================================================================================"
echo

# Check if input base directory exists
if [ ! -d "${INPUT_BASE}" ]; then
    echo "Error: Input base directory not found: ${INPUT_BASE}"
    exit 1
fi

# Check each subset directory
for subset in "${SUBSETS[@]}"; do
    if [ ! -d "${INPUT_BASE}/${subset}" ]; then
        echo "Error: Subset directory not found: ${INPUT_BASE}/${subset}"
        exit 1
    fi
done

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

# ==================== Create Combined Symlink Directory ====================

echo "Creating combined symlink directory: ${COMBINED_DIR}"
rm -rf "${COMBINED_DIR}"
mkdir -p "${COMBINED_DIR}"

TOTAL_LINKS=0
for subset in "${SUBSETS[@]}"; do
    SUBSET_COUNT=0
    for f in "${INPUT_BASE}/${subset}"/*.jsonl; do
        [ -f "$f" ] || continue
        basename=$(basename "$f")
        ln -s "$f" "${COMBINED_DIR}/${subset}__${basename}"
        SUBSET_COUNT=$((SUBSET_COUNT + 1))
    done
    echo "  ${subset}: ${SUBSET_COUNT} files linked"
    TOTAL_LINKS=$((TOTAL_LINKS + SUBSET_COUNT))
done

echo "Total symlinks created: ${TOTAL_LINKS}"
echo

if [ ${TOTAL_LINKS} -eq 0 ]; then
    echo "Error: No JSONL files found across subsets"
    rm -rf "${COMBINED_DIR}"
    exit 1
fi

# ==================== Preprocessing ====================

echo "================================================================================"
echo "Starting preprocessing..."
echo "Start time: $(date)"
echo "================================================================================"
echo

# Set Python path
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH

# Set NUMEXPR_MAX_THREADS to avoid warning
export NUMEXPR_MAX_THREADS=${WORKERS}

# Change to preprocessing directory
cd "${MEGATRON_PATCH_PATH}/toolkits/pretrain_data_preprocessing"

# Show sample input files
echo "Sample symlinked files:"
ls "${COMBINED_DIR}" | sort | head -5
if [ ${TOTAL_LINKS} -gt 5 ]; then
    echo "... and $(($TOTAL_LINKS - 5)) more files"
fi
echo

# Run preprocessing with lm_dataformat (efficient document-level parallelism)
python preprocess_data.py \
  --input "${COMBINED_DIR}" \
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

# ==================== Cleanup ====================

echo
echo "Cleaning up symlink directory: ${COMBINED_DIR}"
rm -rf "${COMBINED_DIR}"

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
