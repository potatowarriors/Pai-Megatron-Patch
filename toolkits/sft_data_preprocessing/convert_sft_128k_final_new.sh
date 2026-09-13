#!/bin/bash
# 최종 단일 SFT 신규 멤버 변환 (2026-09-13, 사용자 결정: phase-1/2 폐기 → Ultra 비율 단일 SFT)
#   chat_v2_on_scrub  : Chat-v2 reasoning_on p2scrub 판(자기귀속 1,823행 드롭) + --fanout-implicit-turns (구 chat_v2_on_fanout 은 스크럽 전 원본이었음)
#   chat_v2_off_scrub : Chat-v2 reasoning_off p2scrub 판(829행 드롭), no-think 규약
#   ifchat_v1_chat_if : Nemotron-Instruction-Following-Chat-v1 chat_if (426k행, 멀티턴 83%·reasoning 36%·train_turns 없음) → --fanout-implicit-turns. Ultra 공개 블렌드 최상위 가중치 셋
#   ifchat_v1_structured : 〃 structured_outputs (4,969행, XML/JSON 스키마 준수, reasoning 100%)
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SFT=/home/work/Datasets/LL_datasets/posttraining/SFT
OUT=${OUT:-/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_final_pad16}
NCORES=${NCORES:-96}
TOK=$REPO/examples/alpha/tokenizer_v5
export TOKENIZERS_PARALLELISM=false NUMEXPR_MAX_THREADS=64
export PYTHONPATH=$REPO:$REPO/backends/megatron/Megatron-LM-251125${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT"
run() {
  local name=$1 input=$2; shift 2
  if [ -f "$OUT/$name/data_text_document.idx" ]; then echo "== $name: SKIP (idx 존재)"; return; fi
  echo "== $name: $input  ($(date +%H:%M:%S))"; mkdir -p "$OUT/$name"
  nice -n 10 python3 "$REPO/toolkits/sft_data_preprocessing/build_alpha_sft_idxmap.py" --input "$input" --tokenizer "$TOK" --output-prefix "$OUT/$name/data" \
    --seq-length 131072 --pad-doc-multiple 16 --workers "$NCORES" "$@" 2>&1 | grep -E "^\[(encode|pack|done)\]|Error|Traceback" || echo "!! $name FAILED"
}
run chat_v2_off_scrub    "$SFT/Nemotron-SFT-Instruction-Following-Chat-v2/data/reasoning_off.p2scrub.jsonl"
run ifchat_v1_structured "$SFT/Nemotron-Instruction-Following-Chat-v1/data/structured_outputs.jsonl"
run ifchat_v1_chat_if    "$SFT/Nemotron-Instruction-Following-Chat-v1/data/chat_if.jsonl" --fanout-implicit-turns
run chat_v2_on_scrub     "$SFT/Nemotron-SFT-Instruction-Following-Chat-v2/data/reasoning_on.p2scrub.jsonl" --fanout-implicit-turns
echo "== ALL DONE ($(date +%H:%M:%S)) -> $OUT"
