#!/bin/bash
#
# preprocess_stage2_v5.sh — generic v5 multi-shard pre-tokenization driver.
#
# Drives the FAST path (fast_tokenize_v5.py, Rust encode_batch) across P parallel
# processes — one per round-robin partition of a multi-file jsonl dataset — then
# concatenates the P parts with merge_indices.py into a single mmap dataset.
# This is the stage-2 analogue of preprocess_dclm_v5.sh, but using the fast
# tokenizer (~15-19M tok/s here) instead of the slow per-doc preprocess_data_megatron.py.
#
# Output: <output_prefix>_text_document.{bin,idx}   (alpha v5, vocab 163,968, int32, EOD=id 0)
#
# Usage:
#   bash preprocess_stage2_v5.sh <input_jsonl_dir> <output_prefix> [procs] [text_key]
#
# Defaults: procs=12 (×8 rayon threads ≈ 96 of the 110-core cgroup), text_key=text
# Env: AUTO_CLEAN_PARTS=1 to delete per-partition parts after a verified merge.
#
# Example:
#   bash preprocess_stage2_v5.sh \
#     /home/work/Datasets/LL_datasets/pretraining/stage2/math \
#     /home/work/Datasets/LL_preprocessed/v5/stage2/math/data
#
# Notes:
#   - --append-eod appends EOS = <|endoftext|> (id 0). No remap_eod needed (the v5
#     tokenizer_config already designates id 0 as EOS), unlike Stage 1.
#   - Round-robin balances by file COUNT; with many similar-sized shards (math 193,
#     code 64/subset, CC-HQ ~2.7k) byte imbalance is small. For few huge shards
#     prefer more partitions to avoid a slow-partition tail.

set -euo pipefail

START_TIME=$SECONDS

# ==================== Configuration ====================

MEGATRON_PATCH_PATH="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch"
MEGATRON_PATH="${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-251125"
TOKENIZER_PATH="${MEGATRON_PATCH_PATH}/examples/alpha/tokenizer_v5"
PREPROC_DIR="${MEGATRON_PATCH_PATH}/toolkits/pretrain_data_preprocessing"

INPUT_DIR="${1:-}"
OUTPUT_PREFIX="${2:-}"          # e.g. /.../v5/stage2/math/data -> data_text_document.{bin,idx}
PROCS="${3:-12}"
TEXT_KEY="${4:-text}"
RAYON_THREADS="${RAYON_THREADS:-8}"
BATCH_SIZE="${BATCH_SIZE:-5000}"
AUTO_CLEAN_PARTS="${AUTO_CLEAN_PARTS:-0}"

if [ -z "${INPUT_DIR}" ] || [ -z "${OUTPUT_PREFIX}" ]; then
    echo "Usage: bash preprocess_stage2_v5.sh <input_jsonl_dir> <output_prefix> [procs] [text_key]"
    exit 1
fi

OUTPUT_DIR="$(dirname "${OUTPUT_PREFIX}")"
WORKDIR="${OUTPUT_DIR}/_parts"

echo "================================================================================"
echo "Stage-2 v5 Pre-tokenization (fast path)"
echo "================================================================================"
echo "Input dir:    ${INPUT_DIR}"
echo "Output:       ${OUTPUT_PREFIX}_text_document.{bin,idx}"
echo "Workdir:      ${WORKDIR}"
echo "Tokenizer:    ${TOKENIZER_PATH}"
echo "Megatron:     ${MEGATRON_PATH}"
echo "Procs:        ${PROCS}  (x ${RAYON_THREADS} rayon threads)"
echo "Text key:     ${TEXT_KEY}"
echo "================================================================================"
echo

# ==================== Validation ====================

[ -d "${INPUT_DIR}" ]     || { echo "Error: input dir not found: ${INPUT_DIR}"; exit 1; }
[ -d "${TOKENIZER_PATH}" ]|| { echo "Error: tokenizer not found: ${TOKENIZER_PATH}"; exit 1; }
[ -d "${MEGATRON_PATH}" ] || { echo "Error: Megatron backend not found: ${MEGATRON_PATH}"; exit 1; }

# Skip if final output already exists
if [ -f "${OUTPUT_PREFIX}_text_document.bin" ] && [ -f "${OUTPUT_PREFIX}_text_document.idx" ]; then
    BIN_SIZE=$(du -h "${OUTPUT_PREFIX}_text_document.bin" | cut -f1)
    echo "Warning: output already exists (${BIN_SIZE}). Skipping."
    echo "Delete ${OUTPUT_PREFIX}_text_document.{bin,idx} to reprocess."
    exit 0
fi

mkdir -p "${WORKDIR}"

# Enumerate input files deterministically (sorted)
ALL_LIST="${WORKDIR}/all_files.txt"
find "${INPUT_DIR}" -type f -name '*.jsonl' | LC_ALL=C sort > "${ALL_LIST}"
NUM_FILES=$(wc -l < "${ALL_LIST}")
if [ "${NUM_FILES}" -eq 0 ]; then
    echo "Error: no *.jsonl files found under ${INPUT_DIR}"
    echo "(qa_pairs is .jsonl.zstd — decompress with eng/decompress_zstd.py first)"
    exit 1
fi
echo "Found ${NUM_FILES} jsonl files."

# Don't spawn more partitions than files
if [ "${PROCS}" -gt "${NUM_FILES}" ]; then
    PROCS="${NUM_FILES}"
    echo "Fewer files than procs → using PROCS=${PROCS}"
fi

# Guard: confirm the text key exists on the first doc (fail fast vs silent empty output)
FIRST_FILE=$(head -1 "${ALL_LIST}")
HAS_KEY=$(head -1 "${FIRST_FILE}" | python3 -c "import sys,json;
try:
    print(int('${TEXT_KEY}' in json.loads(sys.stdin.readline())))
except Exception:
    print(0)")
if [ "${HAS_KEY}" != "1" ]; then
    echo "Error: text key '${TEXT_KEY}' not found in first doc of ${FIRST_FILE}"
    echo "First-doc keys:"; head -1 "${FIRST_FILE}" | python3 -c "import sys,json; print(list(json.loads(sys.stdin.readline()).keys()))" 2>/dev/null
    exit 1
fi
echo "Text key '${TEXT_KEY}' present. OK."
echo

# ==================== Round-robin partition the file list ====================

# filelist_i gets files where (line_no-1) % PROCS == i  (sorted input → deterministic)
rm -f "${WORKDIR}"/filelist_*.txt
awk -v P="${PROCS}" -v d="${WORKDIR}" '{ print > (d "/filelist_" ((NR-1)%P) ".txt") }' "${ALL_LIST}"

echo "Partition file counts:"
for i in $(seq 0 $((PROCS-1))); do
    c=$(wc -l < "${WORKDIR}/filelist_${i}.txt" 2>/dev/null || echo 0)
    echo "  part ${i}: ${c} files"
done
echo

# ==================== Launch P parallel fast_tokenize_v5 processes ====================

export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:${PYTHONPATH:-}
cd "${PREPROC_DIR}"

echo "Launching ${PROCS} tokenizer processes..."
PIDS=()
for i in $(seq 0 $((PROCS-1))); do
    RAYON_NUM_THREADS=${RAYON_THREADS} python3 fast_tokenize_v5.py \
        --mode jsonl-chunk \
        --input-list-file "${WORKDIR}/filelist_${i}.txt" \
        --output-prefix "${WORKDIR}/part_${i}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --text-key "${TEXT_KEY}" \
        --rayon-threads "${RAYON_THREADS}" \
        --batch-size "${BATCH_SIZE}" \
        --append-eod \
        --megatron-path "${MEGATRON_PATH}" \
        > "${WORKDIR}/log_${i}.txt" 2>&1 &
    PIDS+=($!)
done

# Wait for all; collect failures
FAIL=0
for i in $(seq 0 $((PROCS-1))); do
    if ! wait "${PIDS[$i]}"; then
        echo "ERROR: partition ${i} failed (see ${WORKDIR}/log_${i}.txt):"
        tail -5 "${WORKDIR}/log_${i}.txt" || true
        FAIL=1
    fi
done
if [ "${FAIL}" -ne 0 ]; then
    echo "Aborting: one or more partitions failed. Parts left in ${WORKDIR} for inspection."
    exit 1
fi
echo "All ${PROCS} partitions tokenized."
echo

# ==================== Merge parts ====================

PART_PREFIXES=()
for i in $(seq 0 $((PROCS-1))); do
    PART_PREFIXES+=("${WORKDIR}/part_${i}")
done

echo "Merging ${PROCS} parts → ${OUTPUT_PREFIX}_text_document.{bin,idx}"
python3 merge_indices.py \
    --output "${OUTPUT_PREFIX}" \
    --parts "${PART_PREFIXES[@]}" \
    --dtype int32 \
    --megatron-path "${MEGATRON_PATH}"

TOTAL_TIME=$((SECONDS - START_TIME))

# ==================== Summary + cleanup ====================

echo
echo "================================================================================"
echo "Done in $((TOTAL_TIME/3600))h $((TOTAL_TIME%3600/60))m $((TOTAL_TIME%60))s"
if [ -f "${OUTPUT_PREFIX}_text_document.bin" ]; then
    BIN_BYTES=$(stat --format=%s "${OUTPUT_PREFIX}_text_document.bin")
    echo "Output: ${OUTPUT_PREFIX}_text_document.bin ($(du -h "${OUTPUT_PREFIX}_text_document.bin" | cut -f1))"
    echo "        ~$((BIN_BYTES / 4)) tokens (~$((BIN_BYTES / 4 / 1000000000))B, int32)"
fi
echo "================================================================================"

if [ "${AUTO_CLEAN_PARTS}" = "1" ]; then
    echo "AUTO_CLEAN_PARTS=1 → removing parts + file lists"
    rm -f "${WORKDIR}"/part_*_text_document.{bin,idx} "${WORKDIR}"/filelist_*.txt "${WORKDIR}"/all_files.txt "${WORKDIR}"/log_*.txt
    rmdir "${WORKDIR}" 2>/dev/null || true
else
    echo "Per-partition parts kept in ${WORKDIR} (merge verified doc+token counts)."
    echo "Reclaim disk after the merged output is validated:"
    echo "  rm -rf ${WORKDIR}"
fi
