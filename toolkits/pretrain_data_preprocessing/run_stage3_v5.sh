#!/bin/bash
#
# run_stage3_v5.sh — pre-tokenize + best-fit pack the three Nemotron-Pretraining-
# Specialized datasets (v1, v1.1, v1.2) with the alpha v5 tokenizer
# (vocab 163,968, int32, EOD = id 0), reusing the fast stage-2 path.
#
# Per subset, idempotent state machine:
#   1) packed output already exists            -> SKIP
#   2) unpacked mmap exists but not packed      -> pack -> (drop unpacked)
#   3) neither exists                           -> parquet->jsonl -> tokenize
#                                                  -> pack -> (drop jsonl + unpacked)
#
#   parquet->jsonl : convert_parquet_to_jsonl.py   (one jsonl per parquet)
#   jsonl->mmap    : preprocess_stage2_v5.sh        (fast_tokenize_v5 + merge)
#   mmap->packed   : bestfit_pack.py --seq-length 4096 --eod 0
#
# Subsets are auto-discovered (any dir holding *.parquet) under each version's raw
# dir, so Multiple-Choice appearing in both v1.1 and v1.2 stays separated by version.
#
# Layout (all on shared NFS):
#   raw parquet : $RAW/<Version>/<Subset>/*.parquet
#   jsonl (temp): $RAW/_jsonl/<Version>/<Subset>/*.jsonl      (deleted after tokenize)
#   unpacked    : $OUT/<ver>/<key>/data_text_document.{bin,idx}  (deleted after pack)
#   PACKED      : $OUT_PACKED/<ver>/<key>/data_text_document.{bin,idx}  <-- final, for training
#
# Parallelism scales to NCORES (default: all visible cores). On a busy training node
# set NCORES below nproc to leave headroom, e.g.  NCORES=64 nice -n 10 bash run_stage3_v5.sh all
#   tokenize: PROCS=min(NCORES,nfiles), RAYON=max(1,NCORES/PROCS)  (total ~= NCORES)
#
# Usage:
#   NCORES=64 bash run_stage3_v5.sh all       # convert+tokenize+pack every subset, all 3 versions
#   NCORES=64 bash run_stage3_v5.sh v1|v1.1|v1.2
#   bash run_stage3_v5.sh convert  [ver]      # only parquet->jsonl
#   bash run_stage3_v5.sh tokenize [ver]      # only jsonl->unpacked mmap
#   bash run_stage3_v5.sh pack     [ver]      # only unpacked->packed (+drop unpacked)
#   bash run_stage3_v5.sh status              # print per-subset done/pending table
#
# Env:
#   NCORES          total worker threads (default: $(nproc))
#   SEQLEN=4096     pack bin capacity = model seq-length (MUST match train seq-length)
#   EOD=0           end-of-document token id
#   KEEP_UNPACKED=0 1 = keep $OUT unpacked mmaps too (default 0 = packed-only, matches stage2)
#   CLEAN_JSONL=1   delete a subset's jsonl after its mmap is built
#   AUTO_CLEAN_PARTS=1  delete tokenizer partition parts after a verified merge

set -uo pipefail

MP="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch"
PRE="${MP}/toolkits/pretrain_data_preprocessing"
RAW="/home/work/Datasets/LL_datasets/pretraining/stage3"
JSONL_ROOT="${RAW}/_jsonl"
# 2026-07-29: outputs live under the stage2 tree — these datasets are consumed by
# the (final) stage2 run, there is no stage3. Raw parquet stays under .../stage3/.
OUT="/home/work/Datasets/LL_preprocessed/v5/stage2/specialized"
OUT_PACKED="/home/work/Datasets/LL_preprocessed/v5/stage2_packed/specialized"

NCORES="${NCORES:-$(nproc)}"
SEQLEN="${SEQLEN:-4096}"
EOD="${EOD:-0}"
KEEP_UNPACKED="${KEEP_UNPACKED:-0}"
CLEAN_JSONL="${CLEAN_JSONL:-1}"
export AUTO_CLEAN_PARTS="${AUTO_CLEAN_PARTS:-1}"

declare -A VER=(
  [v1]="Nemotron-Pretraining-Specialized-v1"
  [v1.1]="Nemotron-Pretraining-Specialized-v1.1"
  [v1.2]="Nemotron-Pretraining-Specialized-v1.2"
)
# v1.1 first (small, quick real-data validation), then big v1, then v1.2.
VER_ORDER=(v1.1 v1 v1.2)

subset_key() { echo "${1#Nemotron-Pretraining-}" | tr 'A-Z-' 'a-z_'; }

list_subsets() {  # echo subset dirs (those containing *.parquet) for a version
  local ver="$1" vroot="${RAW}/${VER[$ver]}"
  [ -d "${vroot}" ] || { echo "  [warn] raw dir missing: ${vroot}" >&2; return 0; }
  find "${vroot}" -mindepth 1 -maxdepth 1 -type d | LC_ALL=C sort | while read -r d; do
    ls "${d}"/*.parquet >/dev/null 2>&1 && echo "${d}"
  done
}

have_mmap() { [ -f "${1}_text_document.bin" ] && [ -f "${1}_text_document.idx" ]; }

convert_subset() {
  local ver="$1" subdir="$2" name key jdir workers nfiles
  name="$(basename "${subdir}")"; key="$(subset_key "${name}")"
  jdir="${JSONL_ROOT}/${VER[$ver]}/${name}"
  nfiles=$(find "${subdir}" -maxdepth 1 -name '*.parquet' | wc -l)
  workers="${NCORES}"; [ "${nfiles}" -lt "${workers}" ] && workers="${nfiles}"
  echo; echo "########## convert [${ver}] ${name} -> ${key}  (workers=${workers}) ##########"
  python3 "${PRE}/convert_parquet_to_jsonl.py" \
      --input-dir "${subdir}" --output-dir "${jdir}" --workers "${workers}" --batch-size 10000
}

tokenize_subset() {
  local ver="$1" subdir="$2" name key jdir outpre nfiles procs rayon
  name="$(basename "${subdir}")"; key="$(subset_key "${name}")"
  jdir="${JSONL_ROOT}/${VER[$ver]}/${name}"
  outpre="${OUT}/${ver}/${key}/data"
  nfiles=$(find "${jdir}" -maxdepth 1 -name '*.jsonl' 2>/dev/null | wc -l)
  [ "${nfiles}" -eq 0 ] && { echo "  [ERR] no jsonl in ${jdir} (run convert first)"; return 1; }
  procs="${NCORES}"; [ "${nfiles}" -lt "${procs}" ] && procs="${nfiles}"
  rayon=$(( NCORES / procs )); [ "${rayon}" -lt 1 ] && rayon=1
  echo; echo "########## tokenize [${ver}] ${name} (${nfiles} files, PROCS=${procs} RAYON=${rayon}) ##########"
  mkdir -p "$(dirname "${outpre}")"
  RAYON_THREADS="${rayon}" bash "${PRE}/preprocess_stage2_v5.sh" "${jdir}" "${outpre}" "${procs}" text
}

pack_subset() {
  local ver="$1" subdir="$2" name key inpre outpre emit
  name="$(basename "${subdir}")"; key="$(subset_key "${name}")"
  inpre="${OUT}/${ver}/${key}/data"
  outpre="${OUT_PACKED}/${ver}/${key}/data"
  have_mmap "${inpre}" || { echo "  [ERR] unpacked mmap missing for ${key}: ${inpre}"; return 1; }
  emit="${NCORES}"; [ "${emit}" -gt 48 ] && emit=48
  echo; echo "########## pack [${ver}] ${key}  (seq=${SEQLEN} eod=${EOD}) ##########"
  mkdir -p "$(dirname "${outpre}")"
  python3 "${PRE}/bestfit_pack.py" --input "${inpre}" --output "${outpre}" \
      --seq-length "${SEQLEN}" --eod "${EOD}" --emit-threads "${emit}"
}

process_subset() {  # full idempotent flow for one subset
  local ver="$1" subdir="$2" name key packpre unpre jdir
  name="$(basename "${subdir}")"; key="$(subset_key "${name}")"
  packpre="${OUT_PACKED}/${ver}/${key}/data"
  unpre="${OUT}/${ver}/${key}/data"
  jdir="${JSONL_ROOT}/${VER[$ver]}/${name}"

  if have_mmap "${packpre}"; then
    echo "########## SKIP [${ver}] ${key} — packed output exists ##########"; return 0
  fi
  if ! have_mmap "${unpre}"; then
    convert_subset  "${ver}" "${subdir}" || return 1
    tokenize_subset "${ver}" "${subdir}" || return 1
    [ "${CLEAN_JSONL}" = "1" ] && { echo "  rm jsonl ${jdir}"; rm -rf "${jdir}"; }
  else
    echo "########## reuse existing unpacked mmap for [${ver}] ${key} ##########"
  fi
  pack_subset "${ver}" "${subdir}" || return 1
  if [ "${KEEP_UNPACKED}" != "1" ]; then
    echo "  packed-only -> rm unpacked ${OUT}/${ver}/${key}"; rm -rf "${OUT}/${ver}/${key}"
  fi
  return 0
}

do_version() {  # $1=ver  $2=step(process|convert|tokenize|pack)
  local ver="$1" step="${2:-process}" any=0
  echo "==================================================================="
  echo "VERSION ${ver}  (${VER[$ver]})  step=${step}  NCORES=${NCORES}"
  echo "==================================================================="
  while read -r subdir; do
    [ -z "${subdir}" ] && continue
    any=1
    case "${step}" in
      process)  process_subset  "${ver}" "${subdir}" || return 1 ;;
      convert)  convert_subset  "${ver}" "${subdir}" || return 1 ;;
      tokenize) tokenize_subset "${ver}" "${subdir}" || return 1 ;;
      pack)     pack_subset     "${ver}" "${subdir}" || return 1
                [ "${KEEP_UNPACKED}" != "1" ] && rm -rf "${OUT}/${ver}/$(subset_key "$(basename "${subdir}")")" ;;
    esac
  done < <(list_subsets "${ver}")
  [ "${any}" -eq 0 ] && echo "  (no subsets found for ${ver} — is it downloaded?)"
  return 0
}

print_status() {
  printf "%-6s %-28s %-8s %-8s\n" VER SUBSET UNPACKED PACKED
  for ver in "${VER_ORDER[@]}"; do
    while read -r subdir; do
      [ -z "${subdir}" ] && continue
      local key; key="$(subset_key "$(basename "${subdir}")")"
      local u="-" p="-"
      have_mmap "${OUT}/${ver}/${key}/data" && u="yes"
      have_mmap "${OUT_PACKED}/${ver}/${key}/data" && p="YES"
      printf "%-6s %-28s %-8s %-8s\n" "${ver}" "${key}" "${u}" "${p}"
    done < <(list_subsets "${ver}")
  done
}

target="${1:-all}"
case "${target}" in
  all)       for v in "${VER_ORDER[@]}"; do do_version "$v" process  || exit 1; done ;;
  convert)   if [ -n "${2:-}" ]; then do_version "$2" convert;  else for v in "${VER_ORDER[@]}"; do do_version "$v" convert  || exit 1; done; fi ;;
  tokenize)  if [ -n "${2:-}" ]; then do_version "$2" tokenize; else for v in "${VER_ORDER[@]}"; do do_version "$v" tokenize || exit 1; done; fi ;;
  pack)      if [ -n "${2:-}" ]; then do_version "$2" pack;     else for v in "${VER_ORDER[@]}"; do do_version "$v" pack     || exit 1; done; fi ;;
  status)    print_status; exit 0 ;;
  v1|v1.1|v1.2) do_version "${target}" process ;;
  *) echo "usage: bash run_stage3_v5.sh {all|v1|v1.1|v1.2|convert [ver]|tokenize [ver]|pack [ver]|status}"; exit 1 ;;
esac

echo; echo "run_stage3_v5.sh: '${target}' complete."
print_status
