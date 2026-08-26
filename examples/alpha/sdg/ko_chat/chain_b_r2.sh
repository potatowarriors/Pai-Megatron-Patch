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
# OxAlpha 는 계정당 **1,000 요청/일** 상한(free-models-per-day-stealth, 2026-08-25 실측,
# 리셋 UTC 00:00). 100k 규모 생성·심판에는 쓸 수 없어 기본 off — 소규모 프리미엄
# 슬라이스/감사용으로만 env 로 켠다. B_OR_SLICE=N (0=없음), B_JUDGE_BACKEND=local|openrouter
B_OR_SLICE=${B_OR_SLICE:-0}
B_JUDGE_BACKEND=${B_JUDGE_BACKEND:-local}
say "B r1 완료 — r2 시드 100k 생성 (seed 20260824) → OR 슬라이스 $B_OR_SLICE + Gemma 나머지 (judge=$B_JUDGE_BACKEND)"

"$VENV" "$K/prepare_ko_seed.py" --num-records 100000 --seed 20260824 \
  --out "$K/ko_seed_r2_all.parquet" >> "$K/out/r2_b.log" 2>&1 \
  || { say "CHAIN HALT: r2 시드 생성 실패"; exit 1; }
"$VENV" - << PYEOF >> "$K/out/r2_b.log" 2>&1 || { say "CHAIN HALT: 시드 분할 실패"; exit 1; }
import pandas as pd
n_or = int("$B_OR_SLICE")
df = pd.read_parquet("$K/ko_seed_r2_all.parquet")
df.iloc[:n_or].reset_index(drop=True).to_parquet("$K/ko_seed_r2_or.parquet", index=False)
df.iloc[n_or:].reset_index(drop=True).to_parquet("$K/ko_seed_r2.parquet", index=False)
print("split: or=", n_or, "gemma=", len(df) - n_or)
PYEOF
N_GEMMA=$((100000 - B_OR_SLICE))

run_dd() {  # $1=seed parquet $2=num $3=dataset $4=max-parallel $5=extra flags $6=log
  local seed=$1 num=$2 name=$3 par=$4 extra=$5 log=$6
  for i in $(seq 1 6); do
    wait_server || { say "CHAIN HALT: 서버 복구 실패"; return 1; }
    local RESUME=""
    [ "$i" -gt 1 ] && RESUME="--resume"
    say "DD $name round $i $RESUME"
    "$VENV" "$K/ko_chat_sdg.py" --vllm-endpoint "$BASE" --model gemma-4-31b \
      --seed-path "$seed" --num-records "$num" --dataset-name "$name" \
      --max-parallel "$par" --no-tui $RESUME --judge-backend "$B_JUDGE_BACKEND" $extra \
      >> "$log" 2>&1
    if grep -q "생성 완료 → .*$name\$" "$log"; then
      say "DD $name 완료"; return 0
    fi
  done
  say "DD $name 6회 후 미완료"; return 2
}

say "트랙 B r2 가동 — Gemma 슬라이스($N_GEMMA, par 96)$([ "$B_OR_SLICE" -gt 0 ] && echo " ∥ OR 슬라이스($B_OR_SLICE, gen OxAlpha, par 8)")"
PID_OR=""
# --strict-facts: 캘리브레이션 실측(08-26) 기반 사실성 규칙 — r2 부터 (r1 지문 보존)
if [ "$B_OR_SLICE" -gt 0 ]; then
  run_dd "$K/ko_seed_r2_or.parquet" "$B_OR_SLICE" ko_chat_b_r2_or 8 "--gen-backend openrouter --strict-facts" "$K/out/r2_b_or.log" &
  PID_OR=$!
fi
run_dd "$K/ko_seed_r2.parquet" "$N_GEMMA" ko_chat_b_r2 96 "--strict-facts" "$K/out/r2_b.log"
[ -n "$PID_OR" ] && wait $PID_OR

# OR 슬라이스 잔여 → Gemma 로 회수 (시드 단위 폴백)
if [ "$B_OR_SLICE" -gt 0 ] && "$VENV" "$K/b_remainder.py" --seeds "$K/ko_seed_r2_or.parquet" \
     --artifacts "$K/artifacts/ko_chat_b_r2_or/**/*.parquet" \
     --out "$K/ko_seed_r2_or_rem.parquet" >> "$K/out/r2_b_or.log" 2>&1; then
  N_REM=$("$VENV" -c "import pandas as pd; print(len(pd.read_parquet('$K/ko_seed_r2_or_rem.parquet')))")
  say "OR 슬라이스 잔여 $N_REM 건 → Gemma 회수 런"
  run_dd "$K/ko_seed_r2_or_rem.parquet" "$N_REM" ko_chat_b_r2_or_rem 96 "" "$K/out/r2_b.log"
fi
say "B 체인 종료"
