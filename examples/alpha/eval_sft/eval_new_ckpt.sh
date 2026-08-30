#!/bin/bash
# eval_new_ckpt.sh — 새 체크포인트 하나를 변환부터 전 티어 벤치까지 자동으로 돌린다.
#
#   MG ckpt → (G1 내장 변환) → run_suite.sh (T1·T3·에이전틱·T2) → 판정·집계
#
# 변환은 GPU 를 쓰므로 **sub1 fleet 가 내려간 상태**에서 시작해야 한다. run_suite.sh 가
# 각 단계마다 fleet 를 갈아끼우므로, 이 스크립트는 변환 → run_suite 순서만 보장한다.
#
# 사용: bash eval_sft/eval_new_ckpt.sh <RUN_DIR> <ITER> [STAGES]
#   RUN_DIR : outputs/<sft_run>            (checkpoints/ 를 품은 디렉토리)
#   ITER    : 600                          (checkpoints/iter_0000600)
#   STAGES  : t1,t3,agentic,t2 (기본 전부)
# 환경변수: GPUS (기본 0~7), SWE_W, TERM_W
set -uo pipefail
RUN_DIR="${1:?run dir (outputs/<sft_run>)}"; ITER="${2:?iteration (예: 600)}"
STAGES="${3:-t1,t3,agentic,t2}"
HERE="$(cd "$(dirname "$0")" && pwd)"; ALPHA="$(dirname "$HERE")"
REPO="$(dirname "$(dirname "$ALPHA")")"
ITERPAD=$(printf "%07d" "$ITER")
HFDIR="$RUN_DIR/hfmodel_$ITERPAD"
RUN_TAG="$(basename "$RUN_DIR")_iter$ITERPAD"

echo "[new-ckpt] run=$RUN_DIR iter=$ITER tag=$RUN_TAG stages=$STAGES"

# ---- 1) 변환 (이미 있으면 재사용) ----
if [ -f "$HFDIR/generation_config.json" ] && [ -f "$HFDIR/config.json" ]; then
  echo "[new-ckpt] HF 변환본 재사용: $HFDIR"
else
  echo "[new-ckpt] MG→HF 변환 시작 (G1 게이트 내장)"
  # fleet 가 GPU 를 물고 있으면 변환이 OOM 난다 — 먼저 내린다.
  bash "$HERE/stop_fleet.sh" "${GPUS:-0,1,2,3,4,5,6,7}" >/dev/null 2>&1 || true
  sleep 5
  bash "$REPO/toolkits/distributed_checkpoints_convertor/scripts/alpha/run_convert.sh" \
      baseline_48L "$RUN_DIR" "auto:$ITER" true true bf16 || {
    echo "[new-ckpt] ❌ 변환 실패 — 중단"; exit 1; }
fi

# ---- 2) G1 재확인 (변환 경로를 안 탔을 수도 있으므로) ----
python3 "$ALPHA/tools/emit_generation_config.py" "$HFDIR" --check || {
  echo "[new-ckpt] ❌ G1 실패 — 벤치 중단"; exit 1; }

# ---- 3) 전 티어 ----
bash "$HERE/run_suite.sh" "$HFDIR" "$RUN_TAG" "$STAGES"
rc=$?
echo "[new-ckpt] 완료 (rc=$rc): $HERE/results/$RUN_TAG"
exit $rc
