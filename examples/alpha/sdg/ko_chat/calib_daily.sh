#!/bin/bash
# calib_daily.sh — 매일 UTC 00:05(리셋 직후) OxAlpha 심판 캘리브레이션 900건 실행.
# 무인 루프. 시작: nohup bash calib_daily.sh > out/calib_daily.log 2>&1 &
CHAIN_NAME=CALIB
source "$(dirname "$0")/chain_common.sh"

while true; do
  now=$(date -u +%s)
  next=$(( (now / 86400 + 1) * 86400 + 300 ))   # 다음 UTC 00:05
  say "다음 실행까지 $(( (next - now) / 60 ))분 대기"
  sleep $(( next - now ))
  say "캘리브레이션 실행 (900건, effort high)"
  PYTHONPATH="$K" "$VENV" "$K/calibrate_judge.py" --n 900 --concurrency 8 --effort high \
    >> "$K/out/calib_runs.log" 2>&1
  say "실행 종료 — 누적 요약은 calibrate_judge.py --report"
done
