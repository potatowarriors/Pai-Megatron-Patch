#!/usr/bin/env bash
# alpha SFT 128k **혼합** 버킷 변환 — 단일 128k 런 실측용 (2026-08-28).
#
# 투버킷(64k 본 런 + 128k 꼬리 런) 대안으로 "전 데이터를 128k 한 번에" 학습할 때의
# 실 처리량·fill 을 재기 위해, sft_40b_blend.yaml 의 26 멤버 전부를 seq 131072 로
# 재변환한다. --min-tokens 없음 → <=128k 샘플 전량 수용, 단문이 장문 꼬리를 채워
# fill ~99% (투버킷 128k 의 65% 와 대조). 입력·플래그는 64k 산출물 stats.json 의
# 기록을 그대로 미러 (keepthink = 현행 템플릿, fanout/me/budget = 동일 플래그).
#
# identity: 64k 는 ×6(114 bins) 로 valid 0-doc 을 피했으나 128k 는 bins 가 절반이라
#   ×12(train_x12.jsonl) 사용 — 학습 스트림 등가, 가중치 불변.
# 실행 (유휴 노드): NCORES=160 bash convert_sft_128k_mixed.sh
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SFT=/home/work/Datasets/LL_datasets/posttraining/SFT
OUT=${OUT:-/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_mixed_pad16}
NCORES=${NCORES:-96}
TOK=$REPO/examples/alpha/tokenizer_v5
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$REPO:$REPO/backends/megatron/Megatron-LM-251125${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT"

run() {
  local name=$1 input=$2
  shift 2
  if [ -f "$OUT/$name/data_text_document.idx" ]; then
    echo "== $name: SKIP (idx 존재)"; return
  fi
  echo "== $name: $input  ($(date +%H:%M:%S))"
  mkdir -p "$OUT/$name"
  nice -n 10 python "$REPO/toolkits/sft_data_preprocessing/build_alpha_sft_idxmap.py" \
    --input "$input" --tokenizer "$TOK" \
    --output-prefix "$OUT/$name/data" \
    --seq-length 131072 --pad-doc-multiple 16 --workers "$NCORES" "$@" \
    || echo "!! $name FAILED (exit $?)"
}

CHAT=$SFT/Nemotron-SFT-Instruction-Following-Chat-v3/data
KO=$SFT/alpha-SFT-KoChat-v2

# 소형 → 대형
run identity_v1        "$SFT/alpha-SFT-Identity-v1/data/train_x12.jsonl"
run cuda_v1            "$SFT/Nemotron-SFT-CUDA-v1"
run safety_v2          "$SFT/Nemotron-SFT-Safety-v2"
run kochat_b_fanout_t2 "$KO/trackB.jsonl" --fanout-train-turns
run budget_trunc_v1_math "$SFT/Nemotron-SFT-Math-v4" --truncate-reasoning-budget --row-stride 20
run budget_trunc_v1_if "$CHAT/instruction_following.jsonl" \
  --fanout-train-turns --truncate-reasoning-budget --row-stride 4
run kochat_if_fanout_me_t2 "$KO/trackA_if.jsonl" --fanout-train-turns --medium-effort
run kochat_chat_t2     "$KO/trackA_chat.jsonl"
run chat_v3_if_fanout_me "$CHAT/instruction_following.jsonl" --fanout-train-turns --medium-effort
run chat_v3_chat       "$CHAT/chat.with_prompts.jsonl"
for f in "$SFT"/Nemotron-SFT-Multilingual-v2/*_ko_* \
         "$SFT"/Nemotron-SFT-Multilingual-v2/*_ja_* \
         "$SFT"/Nemotron-SFT-Multilingual-v2/*_pt_*; do
  [ -f "$f" ] || continue
  run "ml_$(basename "$f" .jsonl)" "$f"
done
run math_proofs_v2     "$SFT/Nemotron-Math-Proofs-v2"
run swe_v3_keepthink   "$SFT/Nemotron-SFT-SWE-v3/data"
run arc_agi_v1_keepthink "$SFT/Nemotron-SFT-ARC-AGI-v1"
run math_v4            "$SFT/Nemotron-SFT-Math-v4"
run opencode_v1        "$SFT/Nemotron-SFT-OpenCode-v1"
run science_v2         "$SFT/Nemotron-SFT-Science-v2"
run cp_v2              "$SFT/Nemotron-SFT-Competitive-Programming-v2"

echo "== ALL DONE ($(date +%H:%M:%S)) -> $OUT"
