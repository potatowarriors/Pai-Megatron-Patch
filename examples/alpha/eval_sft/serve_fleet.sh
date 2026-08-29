#!/bin/bash
# serve_fleet.sh — 단일 GPU vLLM 서버 N개 + 라운드로빈 프록시.
# vLLM data-parallel 이 이 환경에서 munmap 크래시 → DP1 서버 N개로 처리량 확보.
# 사용: bash eval_sft/serve_fleet.sh <HF_CKPT> [MAX_LEN] [N_GPUS] [PROXY_PORT]
#   프록시: http://localhost:<PROXY_PORT>/v1  (기본 8100), 백엔드 800..
set -uo pipefail
CKPT="${1:?HF checkpoint required}"; MAX_LEN="${2:-49152}"; N="${3:-8}"; PROXY_PORT="${4:-8100}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGD=/home/work/vidsearch/tools/fleet_logs; mkdir -p "$LOGD"
# GPUS 환경변수(예 "1,2,3,4,5,6,7")로 GPU 지정 가능. 기본은 0..N-1.
if [ -n "${GPUS:-}" ]; then IFS=',' read -ra GPULIST <<< "$GPUS"; else GPULIST=($(seq 0 $((N-1)))); fi
BACKENDS=""
for idx in "${!GPULIST[@]}"; do
  g="${GPULIST[$idx]}"
  PORT=$((8000+g))
  BACKENDS="${BACKENDS:+$BACKENDS,}$PORT"
  CUDA_VISIBLE_DEVICES=$g setsid bash "$HERE/serve_alpha.sh" "$CKPT" "$MAX_LEN" 1 "$PORT" \
      > "$LOGD/serve_$PORT.log" 2>&1 < /dev/null &
  disown 2>/dev/null || true
  echo "[fleet] GPU$g -> :$PORT"
done
echo "[fleet] backends=$BACKENDS  proxy=:$PROXY_PORT"
# 프록시 (포그라운드로 두면 이 스크립트가 fleet 수명을 잡음; 백그라운드 실행 권장)
setsid python3 "$HERE/lb_proxy.py" --port "$PROXY_PORT" --backends "$BACKENDS" \
    > "$LOGD/proxy.log" 2>&1 < /dev/null &
disown 2>/dev/null || true
echo "[fleet] 프록시 기동. 준비 대기: 각 :800x /v1/models 200 확인 후 http://localhost:$PROXY_PORT/v1 사용"
