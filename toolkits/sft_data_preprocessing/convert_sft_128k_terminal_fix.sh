#!/usr/bin/env bash
# alpha SFT phase-3 데이터 교정 멤버 변환 (2026-09-09, SFT 데이터 일관성 검토 결과 — docs/KNOWN_ISSUES.md 2026-09-09).
#   P3 트리에 교정 멤버 4종을 실제 디렉터리로 만든다. 구 멤버(symlink)는 대조·롤백용으로 보존.
#
#   swe_v3_tools_keepthink : SWE-v3 + 도구 선언 주입(--tools-sidecar, build_swe_v3_tools_sidecar.py 산출) — 검토 #1
#   opencode_tools         : OpenCode-v1, normalize_tool_schema 가 MCP 형(id/inputSchema)을 name/parameters 로 — 검토 #2
#   chat_v2_on_fanout      : Chat-v2 reasoning_on, --fanout-implicit-turns (train_turns 없는 멀티 user 행 전개) — 검토 #3
#   identity_v2_fanout     : Identity-v2 ×12, --fanout-train-turns (multi-True 23.8% 인데 미적용이었음) — 검토 #4
# 실행 (유휴 노드 CPU, 예: sub1): NCORES=160 nohup bash convert_sft_128k_terminal_fix.sh > <log> 2>&1 < /dev/null &
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SFT=/home/work/Datasets/LL_datasets/posttraining/SFT
OUT=${OUT:-/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_terminal_pad16}
NCORES=${NCORES:-96}
TOK=$REPO/examples/alpha/tokenizer_v5
SIDECAR=$SFT/Nemotron-SFT-SWE-v3/alpha_tools_sidecar.json
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
  nice -n 5 python3 "$REPO/toolkits/sft_data_preprocessing/build_alpha_sft_idxmap.py" \
    --input "$input" --tokenizer "$TOK" \
    --output-prefix "$OUT/$name/data" \
    --seq-length 131072 --pad-doc-multiple 16 --workers "$NCORES" "$@" \
    || echo "!! $name FAILED (exit $?)"
}

run identity_v2_fanout "$SFT/alpha-SFT-Identity-v2/data/train_x12.jsonl" --fanout-train-turns
run chat_v2_on_fanout  "$SFT/Nemotron-SFT-Instruction-Following-Chat-v2/data/reasoning_on.jsonl" --fanout-implicit-turns
run opencode_tools     "$SFT/Nemotron-SFT-OpenCode-v1"
for i in $(seq 1 60); do [ -f "$SIDECAR" ] && break; echo "-- sidecar 대기 ($i) $SIDECAR"; sleep 30; done
[ -f "$SIDECAR" ] || { echo "!! sidecar 없음 — swe_v3_tools_keepthink 생략"; exit 1; }
run swe_v3_tools_keepthink "$SFT/Nemotron-SFT-SWE-v3/data" --tools-sidecar "$SIDECAR"
echo "done ($(date +%H:%M:%S)) -> $OUT"
