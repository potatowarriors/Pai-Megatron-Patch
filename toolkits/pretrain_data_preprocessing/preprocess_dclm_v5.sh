#!/bin/bash
#
# DCLM v5 Pre-tokenization Script
# Tokenizes 7 DCLM shards (~1.9 TB raw, ~446B tokens / 0.45T) with the alpha v5
# tokenizer (effective vocab 163,860 / padded 163,968).
# Selected shards: dclm_shard_{00, 01, 02, 03, 04, 11, 13}.jsonl
# (subset-consistent random.seed=42 — see /home/work/Datasets/LL_datasets/minio_download_dclm.py)
#
# Output: /home/work/Datasets/LL_preprocessed/v5/stage1/dclm/data_text_document.{bin,idx}
#
# Usage:
#   bash preprocess_dclm_v5.sh [workers] [partitions]
# Defaults: workers=64, partitions=8
#
# Multi-partition mode: preprocess_data_megatron.py round-robins lines from the
# input directory across N partition .jsonl files, tokenizes each in parallel,
# then merges per-partition .bin/.idx via IndexedDatasetBuilder.add_index().
#

set -e

START_TIME=$SECONDS

# ==================== Configuration ====================

MEGATRON_PATCH_PATH="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch"
MEGATRON_PATH="${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-251125"
TOKENIZER_PATH="${MEGATRON_PATCH_PATH}/examples/alpha/tokenizer_v5"

INPUT_DIR="/home/work/Datasets/LL_datasets/pretraining/stage1/dclm"
OUTPUT_DIR="/home/work/Datasets/LL_preprocessed/v5/stage1/dclm"
OUTPUT_PREFIX="${OUTPUT_DIR}/data"   # → data_text_document.{bin,idx}

WORKERS=${1:-128}    # 8480+ has 224 logical cores; 128 = 57% utilization, leaves OS/NFS headroom
PARTITIONS=${2:-8}   # WORKERS must be a multiple of PARTITIONS

# ==================== Validation ====================

echo "================================================================================"
echo "DCLM v5 Pre-tokenization"
echo "================================================================================"
echo "Input dir:   ${INPUT_DIR}"
echo "Output:      ${OUTPUT_PREFIX}_text_document.{bin,idx}"
echo "Tokenizer:   ${TOKENIZER_PATH}"
echo "Megatron:    ${MEGATRON_PATH}"
echo "Workers:     ${WORKERS}"
echo "Partitions:  ${PARTITIONS}"
echo "================================================================================"
echo

if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: input dir not found: ${INPUT_DIR}"
    exit 1
fi

SHARD_COUNT=$(ls -1 "${INPUT_DIR}"/dclm_shard_*.jsonl 2>/dev/null | wc -l)
if [ "${SHARD_COUNT}" -ne 7 ]; then
    echo "Error: expected exactly 7 dclm_shard_*.jsonl files in ${INPUT_DIR}, found ${SHARD_COUNT}"
    exit 1
fi

if [ ! -d "${TOKENIZER_PATH}" ]; then
    echo "Error: tokenizer not found: ${TOKENIZER_PATH}"
    exit 1
fi

if [ ! -d "${MEGATRON_PATH}" ]; then
    echo "Error: Megatron backend not found: ${MEGATRON_PATH}"
    exit 1
fi

INPUT_SIZE=$(du -sh "${INPUT_DIR}" | cut -f1)
echo "Input dir size: ${INPUT_SIZE}"
echo

# Skip if final output already exists
if [ -f "${OUTPUT_PREFIX}_text_document.bin" ] && [ -f "${OUTPUT_PREFIX}_text_document.idx" ]; then
    BIN_SIZE=$(du -h "${OUTPUT_PREFIX}_text_document.bin" | cut -f1)
    echo "Warning: output already exists (${BIN_SIZE}). Skipping."
    echo "Delete ${OUTPUT_PREFIX}_text_document.{bin,idx} to reprocess."
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
export NUMEXPR_MAX_THREADS=${WORKERS}

cd "${MEGATRON_PATCH_PATH}/toolkits/pretrain_data_preprocessing"

# ==================== Run Preprocessing ====================

echo "Starting tokenization..."
echo

python preprocess_data_megatron.py \
  --input "${INPUT_DIR}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --json-keys text \
  --patch-tokenizer-type AlphaTokenizer \
  --load "${TOKENIZER_PATH}" \
  --workers ${WORKERS} \
  --partitions ${PARTITIONS} \
  --append-eod

TOTAL_TIME=$((SECONDS - START_TIME))

# ==================== Summary ====================

echo
echo "================================================================================"
echo "Preprocessing Complete!"
echo "================================================================================"
echo "Total time: $((TOTAL_TIME/3600))h $((TOTAL_TIME%3600/60))m $((TOTAL_TIME%60))s"
echo

if [ -f "${OUTPUT_PREFIX}_text_document.bin" ] && [ -f "${OUTPUT_PREFIX}_text_document.idx" ]; then
    BIN_SIZE=$(du -h "${OUTPUT_PREFIX}_text_document.bin" | cut -f1)
    IDX_SIZE=$(du -h "${OUTPUT_PREFIX}_text_document.idx" | cut -f1)
    BIN_BYTES=$(stat --format=%s "${OUTPUT_PREFIX}_text_document.bin")
    # vocab > 65k → uint32 (4 bytes/token)
    TOKEN_COUNT=$((BIN_BYTES / 4))

    echo "Output files:"
    echo "  ${OUTPUT_PREFIX}_text_document.bin (${BIN_SIZE})"
    echo "  ${OUTPUT_PREFIX}_text_document.idx (${IDX_SIZE})"
    echo
    echo "Estimated tokens: ${TOKEN_COUNT} (~$((TOKEN_COUNT / 1000000000))B)"
    echo
    echo "Note: per-partition scratch files remain at the sibling level of INPUT_DIR"
    echo "(preprocess_data_megatron.py uses os.path.splitext on args.input, which has"
    echo "no extension, so files are named '<input_dirname>_<i>' without extension):"
    echo "  $(dirname ${INPUT_DIR})/$(basename ${INPUT_DIR})_[0-$((PARTITIONS-1))]    (round-robin partitions, no .jsonl suffix)"
    echo "  ${OUTPUT_DIR}/data_[0-$((PARTITIONS-1))]_text_document.{bin,idx}    (per-partition outputs)"
    echo "Delete after verifying the merged output, e.g.:"
    echo "  rm $(dirname ${INPUT_DIR})/$(basename ${INPUT_DIR})_[0-9]"
    echo "  rm ${OUTPUT_DIR}/data_[0-9]_text_document.{bin,idx}"
else
    echo "ERROR: output files not created!"
    exit 1
fi
