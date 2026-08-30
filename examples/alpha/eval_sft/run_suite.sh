#!/bin/bash
# run_suite.sh — 한 체크포인트에 대해 전 티어를 **정해진 순서로** 돌린다.
#
# 티어마다 서빙 요구가 달라서 fleet 를 갈아끼워야 한다. 그 순서를 사람이 매번 기억하는
# 대신 여기에 굳힌다 (2026-08-30: 잘못된 fleet 로 에이전틱을 돌려 전량 0점이 나왔고,
# 그 0점이 모델 실패와 구분되지 않았다).
#
# | 단계 | 서빙 | 게이트 |
# |---|---|---|
# | T1 코어 | 표준 fleet 40960 | G1·G2·G3 |
# | T3 판정 | 표준 fleet (동일) | G1·G2·G3 |
# | 에이전틱 | **TOOLS=1** fleet 40960 + 역터널 | A1·A2·A3 |
# | T2 롱 | **롱 fleet 139264** | G1·G2·G3 |
#
# 사용: bash eval_sft/run_suite.sh <HF_CKPT> <RUN_TAG> [STAGES]
#   STAGES: 쉼표 목록 (t1,t3,agentic,t2). 기본 전부.
# 환경변수:
#   GPUS       서빙에 쓸 GPU (기본 0~7 전부). GPU0 은 2026-08-29 좀비 누수로 한시 제외했다가
#              2026-08-30 회수 확인 후 복귀 (78.6GiB 여유, bf16 matmul 222 TFLOP/s).
#   SWE_N      SWE 부분 표본 수 (0=전량, 기본 0)
#   TERM_N     Terminal 부분 표본 수 (0=전량, 기본 0)
set -uo pipefail
CKPT="${1:?HF checkpoint dir}"; RUN_TAG="${2:?run tag}"; STAGES="${3:-t1,t3,agentic,t2}"
HERE="$(cd "$(dirname "$0")" && pwd)"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
PROXY=8100
BURL="http://localhost:$PROXY/v1"
LOGD=/home/work/vidsearch/tools/bench_logs; mkdir -p "$LOGD"

has() { case ",$STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }

fleet_up() {  # $1=max_len  $2=tools(0/1)
  echo "[suite] fleet 기동 (max_len=$1 TOOLS=$2 GPUS=$GPUS)"
  bash "$HERE/stop_fleet.sh" "$GPUS" >/dev/null 2>&1 || true
  sleep 5
  ( export PIP_CONSTRAINT= TOOLS="$2" GPUS="$GPUS"
    setsid bash "$HERE/serve_fleet.sh" "$CKPT" "$1" "$NGPU" "$PROXY" \
      > "$LOGD/fleet_${RUN_TAG}.log" 2>&1 < /dev/null & )
  for i in $(seq 1 60); do
    c=0
    for g in $(echo "$GPUS" | tr ',' ' '); do
      [ "$(curl -s -o /dev/null -w %{http_code} --max-time 3 http://localhost:$((8000+g))/v1/models)" = "200" ] && c=$((c+1))
    done
    [ "$c" -eq "$NGPU" ] && { echo "[suite] fleet 준비 $c/$NGPU"; return 0; }
    sleep 20
  done
  echo "[suite] ❌ fleet 준비 실패"; return 1
}

gate_core() {
  python3 "$HERE/check_gates.py" --hf-dir "$CKPT" --base-url "$BURL" \
    || { echo "[suite] ❌ G1~G3 실패 — 중단"; return 1; }
}

rc=0

# ── T1 + T3 : 표준 fleet ──────────────────────────────────────────────
if has t1 || has t3; then
  fleet_up 40960 0 || exit 1
  gate_core || exit 1
  has t1 && { echo "[suite] === T1 ==="; bash "$HERE/run_tier1.sh" "$BURL" "$RUN_TAG" || rc=1; }
  if has t3; then
    echo "[suite] === T3 SimpleQA ==="
    python3 "$HERE/runners/run_simpleqa.py" --base-url "$BURL" --run-name "$RUN_TAG" || rc=1
    echo "[suite] === T3 LogicKor ==="
    python3 "$HERE/runners/run_logickor.py" --base-url "$BURL" --run-name "$RUN_TAG" || rc=1
  fi
fi

# ── 에이전틱 : TOOLS=1 fleet + 역터널 ─────────────────────────────────
if has agentic; then
  fleet_up 40960 1 || exit 1
  echo "[suite] 역터널 기동"
  bash /home/work/vidsearch/tools/start_swe_tunnel.sh; sleep 10
  python3 "$HERE/check_agentic_gates.py" --base-url "$BURL" || { echo "[suite] ❌ A1~A3 실패 — 에이전틱 건너뜀"; rc=1; }
  if [ "$rc" -eq 0 ] || [ "${FORCE_AGENTIC:-0}" = "1" ]; then
    echo "[suite] === SWE-bench ==="
    SKIP_GATES=1 BASE_URL="$BURL" bash "$HERE/run_swe.sh" "$RUN_TAG" "${SWE_N:-0}" "${SWE_W:-6}" || rc=1
    echo "[suite] === Terminal-Bench ==="
    SKIP_GATES=1 BASE_URL="$BURL" bash "$HERE/run_terminal.sh" "$RUN_TAG" "${TERM_N:-0}" "${TERM_W:-4}" || rc=1
  fi
fi

# ── T2 : 롱 fleet ─────────────────────────────────────────────────────
if has t2; then
  fleet_up 139264 0 || exit 1
  gate_core || exit 1
  echo "[suite] === T2 RULER (Reasoning-Off) ==="
  bash "$HERE/run_tier2.sh" "$BURL" "$RUN_TAG" || rc=1
fi

echo "[suite] fleet 종료"
bash "$HERE/stop_fleet.sh" "$GPUS" >/dev/null 2>&1 || true

echo "[suite] === 판정 ==="
python3 "$HERE/summarize.py" "$HERE/results/$RUN_TAG"
echo "[suite] === 집계 ==="
python3 "$HERE/aggregate_results.py" --results-dir "$HERE/results" --out "$HERE/results/TRACKING.md"
echo "[suite] 완료 (rc=$rc): $HERE/results/$RUN_TAG"
exit $rc
