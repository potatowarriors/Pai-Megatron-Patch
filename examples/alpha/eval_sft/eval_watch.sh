#!/bin/bash
# eval_watch.sh — SFT 학습이 만드는 새 체크포인트를 감시해 자동 반복 평가.
#   학습이 진행되며 checkpoints/iter_* 가 늘어나면, 아직 평가 안 한 것을 eval_ckpt 로 돌린다.
#
# 사용: bash eval_sft/eval_watch.sh <RUN_DIR> [TIERS] [POLL_SEC]
#   RUN_DIR : outputs/<sft_run>  (checkpoints/latest_checkpointed_iteration.txt 감시)
#   TIERS   : 기본 t1
#   POLL_SEC: 감시 주기 (기본 600초)
# 환경: GPUS(기본 1..7), MAXLEN. Ctrl-C 또는 STOP 파일로 종료.
#   중단: touch <RUN_DIR>/.eval_watch_stop
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="${1:?RUN_DIR 필요}"; TIERS="${2:-t1}"; POLL="${3:-600}"
STOP="$RUN_DIR/.eval_watch_stop"; rm -f "$STOP"
echo "[watch] $RUN_DIR 감시 시작 (tiers=$TIERS, poll=${POLL}s). 중단: touch $STOP"
declare -A DONE
while true; do
    [ -f "$STOP" ] && { echo "[watch] STOP 감지 → 종료"; break; }
    LATEST=$(tr -d '[:space:]' < "$RUN_DIR/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null)
    if [ -n "$LATEST" ]; then
        IP=$(printf '%07d' $((10#$LATEST)))
        TAG="$(basename "$RUN_DIR")_iter$IP"
        if [ -z "${DONE[$IP]:-}" ] && [ ! -f "$HERE/results/$TAG/.done" ]; then
            echo "[watch] 새 체크포인트 iter=$LATEST → 평가 시작 $(date +%H:%M:%S)"
            bash "$HERE/eval_ckpt.sh" "$RUN_DIR" "$LATEST" "$TIERS" && DONE[$IP]=1 \
                || echo "[watch] iter=$LATEST 평가 실패 (다음 주기 재시도)"
        fi
    fi
    for i in $(seq 1 $((POLL/10))); do [ -f "$STOP" ] && break; sleep 10; done
done
