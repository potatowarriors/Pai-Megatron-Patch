#!/usr/bin/env bash
# alpha SFT phase-2 트리 — phase-1 128k 혼합 트리를 symlink 로 미러하고 수정분 셋만 재변환 (2026-09-01).
#   docs/SFT_PHASE2_PLAN.md §4·§5 (G-P1~G-P3). 집계·플래그는 convert_sft_128k_mixed.sh 와 동일.
#   opencode_fixed : Nemotron-SFT-OpenCode-v1 — normalize_row 가 tool-result list 를 평문화 (KNOWN_ISSUES 2026-09-01 ①)
#   identity_v2    : alpha-SFT-Identity-v2/data/train_x12.jsonl — 카드 1.2 슬라이스 교체 후 (G-P2, 디렉터리 있을 때만)
#   chat_v3_chat_restored : F3 복원분만 담은 jsonl (G-P3, 파일 있을 때만)
# 실행 (유휴 노드 CPU): NCORES=96 bash convert_sft_128k_mixed_p2.sh
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SFT=/home/work/Datasets/LL_datasets/posttraining/SFT
P1=/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_mixed_pad16
OUT=${OUT:-/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_mixed_p2_pad16}
NCORES=${NCORES:-96}
TOK=$REPO/examples/alpha/tokenizer_v5
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$REPO:$REPO/backends/megatron/Megatron-LM-251125${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT"

# 미변경 24 셋은 phase-1 트리로 symlink (identity_v1·opencode_v1 제외)
for d in "$P1"/*/; do
  n=$(basename "$d")
  case "$n" in identity_v1|opencode_v1) continue;; esac
  [ -e "$OUT/$n" ] || ln -s "$P1/$n" "$OUT/$n"
done
echo "symlinks: $(find "$OUT" -maxdepth 1 -type l | wc -l)  ($(date +%H:%M:%S))"

run() {
  local name=$1 input=$2
  shift 2
  if [ -f "$OUT/$name/data_text_document.idx" ]; then
    echo "== $name: SKIP (idx 존재)"; return
  fi
  echo "== $name: $input  ($(date +%H:%M:%S))"
  mkdir -p "$OUT/$name"
  nice -n 10 python3 "$REPO/toolkits/sft_data_preprocessing/build_alpha_sft_idxmap.py" \
    --input "$input" --tokenizer "$TOK" \
    --output-prefix "$OUT/$name/data" \
    --seq-length 131072 --pad-doc-multiple 16 --workers "$NCORES" "$@" \
    || echo "!! $name FAILED (exit $?)"
}

run opencode_fixed "$SFT/Nemotron-SFT-OpenCode-v1"
[ -f "$SFT/alpha-SFT-Identity-v2/data/train_x12.jsonl" ] \
  && run identity_v2 "$SFT/alpha-SFT-Identity-v2/data/train_x12.jsonl" \
  || echo "-- identity_v2: 입력 없음 (G-P2 대기)"
CHAT=$SFT/Nemotron-SFT-Instruction-Following-Chat-v3/data
[ -f "$CHAT/chat.restored_only.jsonl" ] \
  && run chat_v3_chat_restored "$CHAT/chat.restored_only.jsonl" \
  || echo "-- chat_v3_chat_restored: 입력 없음 (G-P3 대기)"
echo "done ($(date +%H:%M:%S))"
