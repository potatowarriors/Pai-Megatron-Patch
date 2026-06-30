#!/bin/bash
#
# run_stage2_v5.sh — canonical recipe to re-tokenize the legacy Stage-2 blend with
# the alpha v5 tokenizer (vocab 163,968). Reproduces examples/alpha/configs/data/
# arxive/stage2.yaml (9 datasets) in v5, using the FAST path (preprocess_stage2_v5.sh).
#
# Legacy blend members and how each is handled here:
#   - korean_web (2nd half)  : already v5-tokenized (v5/stage2/korean_web) — NOT re-done here
#   - math                   : LOCAL jsonl → tokenize
#   - code (5 synthetic subsets): LOCAL jsonl → tokenize (each a separate .bin, as in legacy)
#   - nemotron_cc_hq actual  : restore from MinIO (jsonl)        → tokenize
#   - nemotron_cc_hq qa_pairs: restore from MinIO (jsonl.zstd)   → decompress → tokenize
#
# Sub-targets:
#   bash run_stage2_v5.sh restore   # MinIO restore of both CC-HQ subsets + decompress qa_pairs
#   bash run_stage2_v5.sh local     # tokenize math + 5 code subsets (smallest first)
#   bash run_stage2_v5.sh cchq      # tokenize CC-HQ actual + qa_pairs (needs `restore` done)
#   bash run_stage2_v5.sh all       # restore → local → cchq
#   bash run_stage2_v5.sh <name>    # one dataset by key (math|code_review|...|cchq_actual|cchq_qa)
#   bash run_stage2_v5.sh pack          # Best-fit Packing of ALL blend members (writes stage2_packed/)
#   bash run_stage2_v5.sh pack-dry      # same, --dry-run (report truncation drop / fill, no write)
#   bash run_stage2_v5.sh pack <member> # one member, e.g. `pack code/transpilation` or `pack math`
#
# Best-fit Packing (arXiv 2404.10830) minimizes document truncation by bin-packing
# whole documents into seq_length bins offline; train-time concat-and-chunk then
# slices on document boundaries. Run PER DATASET (Megatron blends whole sequences,
# so per-dataset packing is preserved). Output mirrors the input tree under
# OUT_PACKED; point a packed blend yaml at it. See bestfit_pack.py for details.
#
# Env: PROCS (default 12), AUTO_CLEAN_PARTS=1 to delete parts after each verified merge.
#      SEQLEN (default 4096) and EOD (default 0) for the `pack` targets.

set -uo pipefail

MP="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch"
PRE="${MP}/toolkits/pretrain_data_preprocessing"
RAW="/home/work/Datasets/LL_datasets/pretraining/stage2"
OUT="/home/work/Datasets/LL_preprocessed/v5/stage2"
OUT_PACKED="${OUT_PACKED:-/home/work/Datasets/LL_preprocessed/v5/stage2_packed}"
PROCS="${PROCS:-12}"
SEQLEN="${SEQLEN:-4096}"
EOD="${EOD:-0}"

ENG="${RAW}/eng"
CCHQ_ACTUAL_DIR="${ENG}/Nemotron-CC-HQ-actual"
CCHQ_QA_DIR="${ENG}/Nemotron-CC-HQ-diverse_qa_pairs"
CCHQ_ACTUAL_PREFIX="Opensource-data/Text/LLM/pretraining/stage2/eng/Nemotron-CC-HQ-actual"
CCHQ_QA_PREFIX="Opensource-data/Text/LLM/pretraining/stage2/eng/Nemotron-CC-HQ-diverse_qa_pairs"

# dataset key -> "input_dir|output_prefix"
declare -A DS=(
  [math]="${RAW}/math|${OUT}/math/data"
  [code_review]="${RAW}/code/synthetic-code-review|${OUT}/code/code_review/data"
  [question_answering]="${RAW}/code/synthetic-question-answering|${OUT}/code/question_answering/data"
  [rewriting]="${RAW}/code/synthetic-rewriting|${OUT}/code/rewriting/data"
  [student_teacher]="${RAW}/code/synthetic-student-teacher|${OUT}/code/student_teacher/data"
  [transpilation]="${RAW}/code/synthetic-transpilation|${OUT}/code/transpilation/data"
  [cchq_actual]="${CCHQ_ACTUAL_DIR}|${OUT}/nemotron_cc_hq/actual/data"
  [cchq_qa]="${CCHQ_QA_DIR}|${OUT}/nemotron_cc_hq/qa_pairs/data"
)
# local-only set, smallest first (by raw size) for fast validation
LOCAL_ORDER=(math transpilation student_teacher code_review rewriting question_answering)
CCHQ_ORDER=(cchq_actual cchq_qa)

# Best-fit Packing members = the 10 stage2_v5_blend.yaml datasets, as relative
# subtrees under OUT (smallest first for fast validation). korean_web/fineweb2hq
# are reused v5 shards (already EOD-remapped to id 0), so they pack like the rest.
PACK_MEMBERS=(
  fineweb2hq
  korean_web
  code/transpilation
  code/student_teacher
  code/code_review
  code/rewriting
  math
  code/question_answering
  nemotron_cc_hq/actual
  nemotron_cc_hq/qa_pairs
)

pack_one() {
  local member="$1"; local extra="${2:-}"
  local inpre="${OUT}/${member}/data"
  local outpre="${OUT_PACKED}/${member}/data"
  echo; echo "########## best-fit pack: ${member} ${extra} ##########"
  echo "  in:  ${inpre}_text_document.{bin,idx}"
  echo "  out: ${outpre}_text_document.{bin,idx}"
  # Idempotent in WRITE mode: skip if output already exists (resumable across
  # interruptions of the multi-hour CC-HQ jobs). --dry-run always runs (no output).
  if [[ "${extra}" != *--dry-run* ]] \
     && [ -f "${outpre}_text_document.bin" ] && [ -f "${outpre}_text_document.idx" ]; then
    echo "  SKIP — output already exists (delete to re-pack)"
    return 0
  fi
  python3 "${PRE}/bestfit_pack.py" --input "${inpre}" --output "${outpre}" \
    --seq-length "${SEQLEN}" --eod "${EOD}" ${extra}
}

do_pack() {
  local extra="${1:-}"; local only="${2:-}"
  if [ -n "${only}" ]; then pack_one "${only}" "${extra}"; return $?; fi
  for m in "${PACK_MEMBERS[@]}"; do pack_one "${m}" "${extra}" || return 1; done
}

tokenize() {
  local key="$1"; local spec="${DS[$key]}"
  local indir="${spec%%|*}"; local outpre="${spec##*|}"
  echo; echo "########## tokenize: ${key} ##########"
  echo "  in:  ${indir}"; echo "  out: ${outpre}_text_document.{bin,idx}"
  bash "${PRE}/preprocess_stage2_v5.sh" "${indir}" "${outpre}" "${PROCS}" text
}

do_restore() {
  echo "########## MinIO restore: CC-HQ actual + qa_pairs ##########"
  python3 "${PRE}/minio_restore.py" --workers 16 \
    --prefix "${CCHQ_ACTUAL_PREFIX}" --dest "${CCHQ_ACTUAL_DIR}" --pattern "*.jsonl" || return 1
  python3 "${PRE}/minio_restore.py" --workers 16 \
    --prefix "${CCHQ_QA_PREFIX}" --dest "${CCHQ_QA_DIR}" --pattern "*.jsonl.zstd" || return 1
  echo "########## decompress qa_pairs (.jsonl.zstd → .jsonl) ##########"
  python3 "${ENG}/decompress_zstd.py" "${CCHQ_QA_DIR}" --workers 16 || return 1
  echo "restore + decompress done."
  echo "  actual jsonl:   $(find "${CCHQ_ACTUAL_DIR}" -name '*.jsonl' | wc -l) (expect 2755)"
  echo "  qa_pairs jsonl: $(find "${CCHQ_QA_DIR}" -name '*.jsonl' | wc -l) (expect 2564)"
}

target="${1:-all}"
case "${target}" in
  restore) do_restore ;;
  local)   for k in "${LOCAL_ORDER[@]}"; do tokenize "$k" || exit 1; done ;;
  cchq)    for k in "${CCHQ_ORDER[@]}";  do tokenize "$k" || exit 1; done ;;
  all)     do_restore && { for k in "${LOCAL_ORDER[@]}"; do tokenize "$k" || exit 1; done; }                                 && { for k in "${CCHQ_ORDER[@]}"; do tokenize "$k" || exit 1; done; } ;;
  pack)     do_pack "" "${2:-}" ;;
  pack-dry) do_pack "--dry-run" "${2:-}" ;;
  *)       if [ -n "${DS[$target]:-}" ]; then tokenize "${target}"; else echo "unknown target: ${target}"; echo "valid: restore local cchq all pack pack-dry ${!DS[*]}"; exit 1; fi ;;
esac
