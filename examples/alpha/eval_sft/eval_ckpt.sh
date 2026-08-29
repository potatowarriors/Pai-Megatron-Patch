#!/bin/bash
# eval_ckpt.sh — 한 체크포인트의 전체 벤치 파이프라인 (반복 실행용, 멱등).
#   변환(MG→HF, 필요시) → fleet 기동 → 티어 실행 → 깨끗한 종료 → 결과 집계
#
# 사용:
#   bash eval_sft/eval_ckpt.sh <RUN_DIR> [ITER|latest] [TIERS]
#     RUN_DIR : outputs/<run>  (checkpoints/ 포함) 또는 이미 변환된 hfmodel_* 경로
#     ITER    : 반복번호 또는 latest (기본 latest). hfmodel_* 를 직접 주면 무시.
#     TIERS   : 콤마구분 (기본 "t1"). 현재 t1 지원, t2/t3 는 러너 추가 시.
# 환경:
#   GPUS   : fleet·변환에 쓸 GPU 목록 (기본 8장 0~7)
#   MAXLEN : 서버 max-model-len (기본 49152)
#   WANDB  : 1이면 결과 wandb 업로드(있으면)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ALPHA="$(dirname "$HERE")"
ROOT="$(cd "$ALPHA/../.." && pwd)"
INPUT="${1:?RUN_DIR 또는 hfmodel 경로 필요}"; ITER="${2:-latest}"; TIERS="${3:-t1}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"; MAXLEN="${MAXLEN:-49152}"
N=$(( $(echo "$GPUS" | tr ',' '\n' | wc -l) )); PROXY_PORT=8100
export HF_TOKEN=$(grep -E '^HF_TOKEN=' "$ALPHA/.env" 2>/dev/null | cut -d= -f2)

# ---- 1) HF 모델 확정 (변환 필요시) ----
if [ -f "$INPUT/config.json" ]; then
    HF_OUT="$INPUT"; RUN_TAG="$(basename "$(dirname "$INPUT")")_$(basename "$INPUT")"
else
    RUN_DIR="$INPUT"
    if [ "$ITER" = "latest" ]; then
        ITER=$(tr -d '[:space:]' < "$RUN_DIR/checkpoints/latest_checkpointed_iteration.txt")
    fi
    IP=$(printf '%07d' $((10#$ITER)))
    HF_OUT="$RUN_DIR/hfmodel_$IP"; RUN_TAG="$(basename "$RUN_DIR")_iter$IP"
    if [ ! -f "$HF_OUT/config.json" ]; then
        echo "[eval] 변환 MG→HF iter=$ITER (GPUS=$GPUS)"
        CUDA_VISIBLE_DEVICES="$GPUS" GPUS="$N" bash "$ALPHA/evaluate.sh" "$RUN_DIR" --iter "$ITER" --gpus "$N" \
            || { echo "[eval] 변환/게이트 실패"; exit 1; }
    else
        echo "[eval] HF 이미 존재 → 변환 skip: $HF_OUT"
    fi
fi

OUTD="$HERE/results/$RUN_TAG"; mkdir -p "$OUTD"
if [ -f "$OUTD/.done" ]; then echo "[eval] 이미 평가됨(.done) → skip: $RUN_TAG"; exit 0; fi

# ---- 2) fleet 기동 ----
echo "[eval] fleet 기동 (GPUS=$GPUS, maxlen=$MAXLEN)"
GPUS="$GPUS" bash "$HERE/serve_fleet.sh" "$HF_OUT" "$MAXLEN" "$N" "$PROXY_PORT"
# 준비 대기 (각 백엔드 200)
echo -n "[eval] 백엔드 준비 대기"
for i in $(seq 1 120); do
    c=0; for g in $(echo "$GPUS" | tr ',' ' '); do
        [ "$(curl -s -o /dev/null -w %{http_code} --max-time 3 http://localhost:$((8000+g))/v1/models 2>/dev/null)" = "200" ] && c=$((c+1))
    done
    [ "$c" -eq "$N" ] && { echo " → $N/$N ready"; break; }
    echo -n "."; sleep 10
done
[ "$c" -ne "$N" ] && { echo " → 준비 실패 ($c/$N)"; bash "$HERE/stop_fleet.sh" "$GPUS"; exit 1; }

# ---- 3) 티어 실행 ----
rc=0
BURL="http://localhost:$PROXY_PORT/v1"
case ",$TIERS," in *,t1,*) echo "[eval] T1 실행"; bash "$HERE/run_tier1.sh" "$BURL" "$RUN_TAG" || rc=1 ;; esac
case ",$TIERS," in *,t3,*)
  echo "[eval] T3 SimpleQA 실행"; python3 "$HERE/runners/run_simpleqa.py" --base-url "$BURL" --run-name "$RUN_TAG" || rc=1
  echo "[eval] T3 LogicKor 실행"; python3 "$HERE/runners/run_logickor.py" --base-url "$BURL" --run-name "$RUN_TAG" || rc=1
esac
# (t2 RULER 는 별도 롱컨텍스트 fleet 필요 — 여기 미포함)

# ---- 4) 깨끗한 종료 ----
echo "[eval] fleet 종료"
bash "$HERE/stop_fleet.sh" "$GPUS" || echo "[eval] ⚠️ 종료 시 GPU 회수 경고 (aggregate 는 계속)"

# ---- 5) 결과 집계 ----
python3 "$HERE/aggregate_results.py" --results-dir "$HERE/results" --out "$HERE/results/TRACKING.md" || true
# wandb 로깅 (alpha-post-eval). WANDB=0 이면 skip.
if [ "${WANDB:-1}" != "0" ]; then
    ( source "$ALPHA/scripts/setup_wandb.sh" >/dev/null 2>&1; export WANDB_SILENT=true
      python3 "$HERE/log_eval_wandb.py" --results-dir "$HERE/results" --run-tag "$RUN_TAG" )       || echo "[eval] ⚠️ wandb 로깅 실패 (비치명)"
fi
[ "$rc" -eq 0 ] && touch "$OUTD/.done"
echo "[eval] 완료: $RUN_TAG (rc=$rc). 결과: $OUTD, 추이표: $HERE/results/TRACKING.md"
exit $rc
