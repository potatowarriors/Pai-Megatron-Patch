#!/bin/bash
# probe_ckpt.sh — 체크포인트 1개의 **경량 게이트 2종** (SFT 최종 런 체인, docs/SFT_FINAL_PLAN.md §4.3). GPU 1~2장, ≈20분.
#   ① identity 프로브 (eval_sft/identity_probe.py): 제작자 ≥95% · 누출 0 — thinking OFF/ON 두 번
#   ② 유령 호출 재생 (results/reasoning_probe/bestcase_replay.py): 도구 25종 주입(tools25) 조건 33 샘플 중 유령 호출 ≤ 1
#   T1 벤치는 eval_ckpt.sh(8 GPU fleet) 가 따로 돈다 — 이 스크립트는 학습 중 노드의 남는 GPU 로도 돌 만큼 가볍게 유지한다.
#
# 사용: GPUS=6,7 PORT=8011 bash eval_sft/probe_ckpt.sh <RUN_DIR|HF_DIR> [ITER|latest]
#   RUN_DIR 이면 evaluate.sh 로 MG→HF 변환(forward_sanity·eos 게이트 포함, GPUS 전부 사용) 후 첫 GPU 1장에 vLLM(DP=1, TOOLS=1) 을 띄운다.
#   결과: eval_sft/results/probe/<RUN_TAG>.txt (+ .identity*.json, .ghost.json, .serve.log), 추이 results/probe/TRACKING.md. 멱등(.done).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ALPHA="$(dirname "$HERE")"
INPUT="${1:?RUN_DIR 또는 hfmodel 경로 필요}"; ITER="${2:-latest}"
GPUS="${GPUS:-6,7}"; PORT="${PORT:-8011}"; MAXLEN="${MAXLEN:-32768}"
N=$(( $(echo "$GPUS" | tr ',' '\n' | wc -l) )); G0="${GPUS%%,*}"
RP="$HERE/results/reasoning_probe"; OUTD="$HERE/results/probe"; mkdir -p "$OUTD"
log() { echo "[probe $(date '+%H:%M:%S')] $*"; }

# ---- 1) HF 모델 확정 (eval_ckpt.sh 와 동일 규약) ----
if [ -f "$INPUT/config.json" ]; then
    HF_OUT="$INPUT"; RUN_TAG="$(basename "$(dirname "$INPUT")")_$(basename "$INPUT")"
else
    RUN_DIR="$INPUT"
    [ "$ITER" = "latest" ] && ITER=$(tr -d '[:space:]' < "$RUN_DIR/checkpoints/latest_checkpointed_iteration.txt")
    IP=$(printf '%07d' $((10#$ITER))); HF_OUT="$RUN_DIR/hfmodel_$IP"; RUN_TAG="$(basename "$RUN_DIR")_iter$IP"
    if [ ! -f "$HF_OUT/config.json" ]; then
        log "변환 MG→HF iter=$ITER (GPUS=$GPUS)"
        CUDA_VISIBLE_DEVICES="$GPUS" GPUS="$N" bash "$ALPHA/evaluate.sh" "$RUN_DIR" --iter "$ITER" --gpus "$N" || { log "변환/게이트 실패"; exit 1; }
    fi
fi
SUM="$OUTD/$RUN_TAG.txt"
[ -f "$OUTD/$RUN_TAG.done" ] && { log "이미 완료(.done) → skip: $RUN_TAG"; exit 0; }
: > "$SUM"; echo "# probe $RUN_TAG  ($(date '+%F %T'))  hf=$HF_OUT" >> "$SUM"

# ---- 2) vLLM 1장 (TOOLS=1: tool_choice auto + qwen3_xml 파서 — 재생의 tool_calls 판정에 필요) ----
export HF_TOKEN=$(grep -E '^HF_TOKEN=' "$ALPHA/.env" 2>/dev/null | cut -d= -f2)
CUDA_VISIBLE_DEVICES="$G0" TOOLS=1 setsid bash "$HERE/serve_alpha.sh" "$HF_OUT" "$MAXLEN" 1 "$PORT" > "$OUTD/$RUN_TAG.serve.log" 2>&1 < /dev/null &
SPID=$!; sleep 2; PGID=$(ps -o pgid= -p "$SPID" 2>/dev/null | tr -d ' '); PGID=${PGID:-$SPID}
stop_server() { kill -TERM -- "-$PGID" 2>/dev/null; sleep 15; kill -KILL -- "-$PGID" 2>/dev/null; }
trap stop_server EXIT
log "vLLM 기동 GPU $G0 port $PORT (pgid $PGID)"; ready=0
for i in $(seq 1 90); do
    [ "$(curl -s -o /dev/null -w %{http_code} --max-time 3 http://localhost:$PORT/v1/models 2>/dev/null)" = "200" ] && { ready=1; break; }
    kill -0 "$SPID" 2>/dev/null || break; sleep 20
done
[ "$ready" = 1 ] || { log "서버 준비 실패 → $OUTD/$RUN_TAG.serve.log"; echo "SERVER_FAIL" >> "$SUM"; exit 1; }

# ---- 3) 프로브 ----
BURL="http://localhost:$PORT/v1"; rc=0
log "identity 프로브 (thinking OFF)"; python3 "$HERE/identity_probe.py" --base-url "$BURL" --out "$OUTD/$RUN_TAG.identity_off.json" 2>&1 | tee -a "$SUM" | tail -2
log "identity 프로브 (thinking ON)";  python3 "$HERE/identity_probe.py" --base-url "$BURL" --thinking --max-tokens 2048 --out "$OUTD/$RUN_TAG.identity_on.json" 2>&1 | tee -a "$SUM" | tail -2
log "유령 호출 재생 (tools25 / none)"; python3 "$RP/bestcase_replay.py" "$BURL" "$RUN_TAG" "$RP/owui_builtin_tools_25.json" "$RP/owui_chats.json" "$OUTD/$RUN_TAG.ghost.json" 2>&1 | tee -a "$SUM" | tail -6

# ---- 4) 판정 ----
ID_FAIL=$(grep -c "FAIL" "$SUM" || true)
GHOST=$(python3 - "$OUTD/$RUN_TAG.ghost.json" <<'PY'
import json,sys
try: rows=json.load(open(sys.argv[1]))["rows"]
except Exception: print("NA NA"); sys.exit()
t=[r for r in rows if r["cond"]=="tools25"]; print(sum(1 for r in t if r.get("ghost_call")), len(t))
PY
)
G_N=${GHOST% *}; G_T=${GHOST#* }
G_OK=$([ "$G_N" != "NA" ] && [ "$G_N" -le 1 ] && echo PASS || echo FAIL)
I_OK=$([ "$ID_FAIL" -eq 0 ] && echo PASS || echo FAIL)
{ echo "## 판정: identity=$I_OK (FAIL 줄 $ID_FAIL) · ghost=$G_OK ($G_N/$G_T, 기준 ≤1)"; } | tee -a "$SUM"
printf '| %s | %s | %s | %s/%s | %s |\n' "$(date '+%F %H:%M')" "$RUN_TAG" "$I_OK" "$G_N" "$G_T" "$G_OK" >> "$OUTD/TRACKING.md"
[ "$I_OK" = PASS ] && [ "$G_OK" = PASS ] && touch "$OUTD/$RUN_TAG.done" || rc=2
log "완료 rc=$rc → $SUM"; exit $rc
