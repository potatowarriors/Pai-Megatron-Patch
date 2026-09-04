#!/usr/bin/env bash
# alpha SFT phase-2 "능력 추가" 멤버 변환 — 미사용 SFT 셋을 phase-2 트리(p2)에 추가 (2026-09-04).
#   설계·epoch·리플레이 규칙: docs/SFT_PHASE2_PLAN.md §11 (절충안 M). 집계·플래그는 convert_sft_128k_mixed.sh 와 동일.
#   미변경 24 셋 symlink·opencode_fixed·identity_v2 는 convert_sft_128k_mixed_p2.sh 가 이미 만든 상태를 전제한다.
#
#   agentic_v2_search   : Nemotron-SFT-Agentic-v2 search split, held-out 300행 제외본 (splits/search_train.jsonl)
#                         — Ultra 기술보고서 §Search Capabilities 가 retain 한 Super 검색 트라젝토리
#   agentic_v2_ia       : Agentic-v2 interactive_agent.jsonl (고객응대 838 도메인, DeepSeek-V3.2, reasoning 93%)
#   agentic_v2_tc       : Agentic-v2 tool_calling.jsonl (707,052행 — 2026-09-04 재다운로드 정본; 절단본은 .truncated_* 로 보존)
#   chat_v2_on / _off   : SFT-Instruction-Following-Chat-v2 reasoning_on / reasoning_off (off = 빈 <think></think> no-think 규약)
#   swe_v2_openhands    : SFT-SWE-v2 swe.jsonl (openhands, reasoning 0% — no-think 에이전틱, opencode 선례)
#   swe_v2_agentless    : SFT-SWE-v2 agentless.jsonl (텍스트 localization/repair, reasoning 100%)
#   swe_v1_r2e          : Nemotron-SWE-v1 r2e_gym.jsonl (reasoning 0%)
#   science_v1          : Nemotron-Science-v1 (MCQ + RQA)
#   finance_v1          : SpecializedDomains-Finance-v1 (trainable 4.9% — 긴 문서 프롬프트)
#   math_proofs_v1_lean : Math-Proofs-v1 lean.jsonl (messages=="[]" 인 행 ≈33% 는 bad_row 드롭 — 정상)
#   ml_super-v3_*       : SFT-Multilingual-v1 12 파일 (de/es/fr/it/ja/zh × code/math) — ALPHA_LANGS 대조 후 편입
# 실행 (유휴 노드 CPU, 예: sub1 220 core): NCORES=180 nohup bash convert_sft_128k_mixed_p2b.sh > /path/log 2>&1 &
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SFT=/home/work/Datasets/LL_datasets/posttraining/SFT
OUT=${OUT:-/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_mixed_p2_pad16}
NCORES=${NCORES:-96}
TOK=$REPO/examples/alpha/tokenizer_v5
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$REPO:$REPO/backends/megatron/Megatron-LM-251125${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT"
[ -L "$OUT/cp_v2" ] || { echo "!! $OUT 에 phase-1 symlink 가 없다 — convert_sft_128k_mixed_p2.sh 먼저"; exit 1; }

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

# 소형 → 대형
A2=$SFT/Nemotron-SFT-Agentic-v2
[ -f "$A2/splits/search_train.jsonl" ] || { echo "!! search_train.jsonl 없음 — held-out 분리 먼저"; exit 1; }
run agentic_v2_search   "$A2/splits/search_train.jsonl"
run science_v1          "$SFT/Nemotron-Science-v1/data"
run agentic_v2_ia       "$A2/data/interactive_agent.jsonl"
run swe_v1_r2e          "$SFT/Nemotron-SWE-v1/data/r2e_gym.jsonl"
run swe_v2_agentless    "$SFT/Nemotron-SFT-SWE-v2/data/agentless.jsonl"
run swe_v2_openhands    "$SFT/Nemotron-SFT-SWE-v2/data/swe.jsonl"
run chat_v2_off         "$SFT/Nemotron-SFT-Instruction-Following-Chat-v2/data/reasoning_off.jsonl"
run chat_v2_on          "$SFT/Nemotron-SFT-Instruction-Following-Chat-v2/data/reasoning_on.jsonl"
run agentic_v2_tc       "$A2/data/tool_calling.jsonl"
run finance_v1          "$SFT/Nemotron-SpecializedDomains-Finance-v1/data/train.jsonl"
run math_proofs_v1_lean "$SFT/Nemotron-Math-Proofs-v1/data/lean.jsonl"
if [ "${SKIP_ML_V1:-0}" != "1" ]; then
  for f in "$SFT"/Nemotron-SFT-Multilingual-v1/data/super-v3_*_translated_final.jsonl; do
    [ -f "$f" ] || continue
    run "ml_$(basename "$f" .jsonl)" "$f"
  done
fi
echo "done ($(date +%H:%M:%S)) -> $OUT"
