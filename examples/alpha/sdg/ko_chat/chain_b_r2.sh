#!/bin/bash
# chain_b_r2.sh — 트랙 B 무인 체인 (사용자 사전 승인 2026-08-23).
# r1(20k, pid $1) 종료 대기 → 미완료면 --resume 재시도(최대 3회) →
# 새 시드 100k 생성 → r2 투입 (크래시 시 --resume 재시도, DD tranche1 전례).
CHAIN_NAME=B
source "$(dirname "$0")/chain_common.sh"
B_PID=${1:?usage: chain_b_r2.sh <r1_pid>}

say "B r1 (pid $B_PID) 종료 대기"
while kill -0 "$B_PID" 2>/dev/null; do sleep 60; done
say "B r1 종료 감지"

for i in $(seq 1 3); do
  grep -q "생성 완료" "$K/out/r1_b.log" && break
  say "B r1 미완료 — resume 재시도 $i (admission 리셋 겸)"
  wait_server || { say "CHAIN HALT: 서버 복구 실패"; exit 1; }
  "$VENV" "$K/ko_chat_sdg.py" --vllm-endpoint "$BASE" --model gemma-4-31b \
    --num-records 20000 --dataset-name ko_chat_b_r1 --max-parallel 64 \
    --no-tui --resume >> "$K/out/r1_b.log" 2>&1
done
if ! grep -q "생성 완료" "$K/out/r1_b.log"; then
  say "CHAIN HALT: B r1 3회 재시도 실패 — 수동 검토 필요"
  touch "$K/out/CHAIN_HALT_B"
  exit 1
fi
say "B r1 완료 — r2 시드 100k 생성 (seed 20260824)"

"$VENV" "$K/prepare_ko_seed.py" --num-records 100000 --seed 20260824 \
  --out "$K/ko_seed_r2.parquet" >> "$K/out/r2_b.log" 2>&1 \
  || { say "CHAIN HALT: r2 시드 생성 실패"; exit 1; }
say "트랙 B r2 가동 (100k, max-parallel 64)"

for i in $(seq 1 6); do
  wait_server || { say "CHAIN HALT: 서버 복구 실패"; exit 1; }
  RESUME=""
  [ "$i" -gt 1 ] && RESUME="--resume"
  say "B r2 실행 round $i $RESUME"
  "$VENV" "$K/ko_chat_sdg.py" --vllm-endpoint "$BASE" --model gemma-4-31b \
    --seed-path "$K/ko_seed_r2.parquet" --num-records 100000 \
    --dataset-name ko_chat_b_r2 --max-parallel 96 --no-tui $RESUME \
    >> "$K/out/r2_b.log" 2>&1
  if grep -q "생성 완료 → .*ko_chat_b_r2" "$K/out/r2_b.log"; then
    say "트랙 B r2 완료"
    break
  fi
done
say "B 체인 종료"
