#!/bin/bash
# run_cc_code_v5.sh — tokenize + best-fit pack Nemotron-CC-Code-v1 (427.9B tokens,
# real Common-Crawl code pages; Ultra's "nemotron-cc-code" category) for the
# stage2 P2b blend switch at iter 18,000. One subset -> flat driver, idempotent
# per step (skips convert/tokenize/pack whose outputs already exist).
#
#   NCORES=96 nice -n 10 ionice -c2 -n6 bash run_cc_code_v5.sh
set -uo pipefail

MP="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch"
PRE="${MP}/toolkits/pretrain_data_preprocessing"
SRC="${MP}/datasets/raw/Nemotron-CC-Code-v1/data"
JSONL="/home/work/Datasets/LL_datasets/pretraining/stage3/_jsonl/Nemotron-CC-Code-v1/CC-Code"
UNPACKED="/home/work/Datasets/LL_preprocessed/v5/stage2/cc_code/data"
PACKED_DIR="/home/work/Datasets/LL_preprocessed/v5/stage2_packed/cc_code"
NCORES="${NCORES:-96}"
export AUTO_CLEAN_PARTS=1

have() { [ -f "$1_text_document.bin" ] && [ -f "$1_text_document.idx" ]; }

if have "${PACKED_DIR}/data"; then
  echo "== packed already exists — nothing to do =="; exit 0
fi

if ! have "${UNPACKED}"; then
  n_parq=$(find "${SRC}" -maxdepth 1 -name '*.parquet' | wc -l)
  n_json=$(find "${JSONL}" -maxdepth 1 -name '*.jsonl' 2>/dev/null | wc -l)
  if [ "${n_json}" -lt "${n_parq}" ]; then
    w="${NCORES}"; [ "${n_parq}" -lt "${w}" ] && w="${n_parq}"
    echo "== convert: ${n_parq} parquet -> jsonl (workers=${w}) =="
    python3 "${PRE}/convert_parquet_to_jsonl.py" \
      --input-dir "${SRC}" --output-dir "${JSONL}" --workers "${w}" --batch-size 10000 || exit 1
  else
    echo "== convert: already done (${n_json}/${n_parq}) =="
  fi
  procs=12; rayon=$(( NCORES / procs ))
  echo "== tokenize: ${procs} procs x ${rayon} rayon =="
  RAYON_NUM_THREADS="${rayon}" bash "${PRE}/preprocess_stage2_v5.sh" \
    "${JSONL}" "${UNPACKED}" "${procs}" text || exit 1
  echo "== clean jsonl =="
  rm -rf "${JSONL}"
else
  echo "== unpacked already exists — skip convert/tokenize =="
fi

mkdir -p "${PACKED_DIR}"
echo "== pack (seq=4096 eod=0) =="
python3 "${PRE}/bestfit_pack.py" \
  --input "${UNPACKED}" \
  --output "${PACKED_DIR}/data" \
  --seq-length 4096 --eod 0 || exit 1

echo "== drop unpacked (packed-only policy) =="
rm -rf "$(dirname "${UNPACKED}")"
echo "== DONE: ${PACKED_DIR}/data_text_document.{bin,idx} =="
