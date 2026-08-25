#!/bin/bash
# chain_a_r2.sh — 트랙 A 무인 체인 (사용자 사전 승인 2026-08-23).
# r1(30k, pid $1) 종료 대기 → 자동 게이트 → 잔여 적격 풀 전체(~745k) r2 투입.
# 라운드 반복은 멱등 재개(uuid 스킵)라 에러분 자동 회수. 세션과 독립적으로 동작.
CHAIN_NAME=A
source "$(dirname "$0")/chain_common.sh"
A_PID=${1:?usage: chain_a_r2.sh <r1_pid>}

say "A r1 (pid $A_PID) 종료 대기"
while kill -0 "$A_PID" 2>/dev/null; do sleep 60; done
say "A r1 종료 감지 — 게이트 판정"

if ! python3 "$K/auto_gate_a.py" "$K/out/r1_a"; then
  say "CHAIN HALT: r1_a 게이트 실패 — r2 투입 보류 (수동 검토 필요)"
  touch "$K/out/CHAIN_HALT_A"
  exit 1
fi
say "r1_a GATE PASS — r2 시드 추출 (잔여 풀 전체)"

if [ -s "$K/seeds_r2.jsonl" ]; then
  say "r2 시드 기존 파일 재사용 (재추출 생략 — 결정적 시드라 동일)"
else
  cat "$K/seeds_pilot.jsonl" "$K/seeds_r1.jsonl" > "$K/out/seeds_used.jsonl"
  if ! python3 "$K/extract_sources.py" --num-if 250000 --num-chat 650000 \
      --out "$K/seeds_r2.jsonl" --exclude "$K/out/seeds_used.jsonl" --seed 20260824 \
      >> "$K/out/extract_r2.log" 2>&1; then
    say "CHAIN HALT: r2 시드 추출 실패 (extract_r2.log 확인)"
    exit 1
  fi
fi
N_SEEDS=$(wc -l < "$K/seeds_r2.jsonl")
# 로컬 Gemma 96 + OpenRouter OxAlpha 재생성 25% (재배분 2026-08-25: OR 파이의 몫을
# 한국 맥락 트랙 B 생성으로 이전 — 0.4 였을 때 A 단독으로 429 경계 3% 폴백)
say "r2 시드 $N_SEEDS 행 — 트랙 A r2 가동 (workers 128, openrouter regen 25%)"

for i in $(seq 1 8); do
  wait_server || { say "CHAIN HALT: 서버 복구 실패"; exit 1; }
  say "A r2 실행 round $i"
  python3 "$K/translate_regen.py" --seeds "$K/seeds_r2.jsonl" --out "$K/out/r2_a" \
    --base-url "$BASE" --workers 128 --openrouter-frac 0.25 >> "$K/out/r2_a.log" 2>&1
  DONE=$(( $(cat "$K/out/r2_a/results.jsonl" 2>/dev/null | wc -l) \
        + $(cat "$K/out/r2_a/rejects.jsonl" 2>/dev/null | wc -l) ))
  say "round $i 종료: done=$DONE / $N_SEEDS"
  if [ "$DONE" -ge "$N_SEEDS" ]; then
    say "트랙 A r2 완료 (done=$DONE)"
    break
  fi
done
say "A 체인 종료"
