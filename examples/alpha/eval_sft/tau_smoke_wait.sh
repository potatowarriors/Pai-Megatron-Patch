#!/bin/bash
# tau_smoke_wait.sh — sub1 이 비면(벤치 스위트·vLLM 없음) tau_smoke.sh 를 돌린다. 기한(기본 12h) 넘기면 포기.
# 사용(sub1): setsid nohup bash eval_sft/tau_smoke_wait.sh <HF_CKPT> [TAG] > /home/work/vidsearch/tools/bench_logs/tau_smoke_wait.log 2>&1 < /dev/null &
set -uo pipefail
CKPT="${1:?HF ckpt}"; TAG="${2:-tau_smoke}"; DEADLINE="${DEADLINE:-43200}"
HERE="$(cd "$(dirname "$0")" && pwd)"
end=$((SECONDS + DEADLINE))
echo "[wait] $(date) 시작 — 스위트·vLLM 유휴 대기 (기한 ${DEADLINE}s)"
while bash "$HERE/suite_running.sh" || pgrep -f "alpha_serve_venv/bin/[v]llm" >/dev/null; do
  [ "$SECONDS" -gt "$end" ] && { echo "[wait] $(date) 기한 초과 — 포기"; exit 1; }
  sleep 120
done
echo "[wait] $(date) sub1 유휴 확인 — 60s 후 스모크"
sleep 60
exec bash "$HERE/tau_smoke.sh" "$CKPT" "$TAG"
