#!/bin/bash
# LC-B 자동 연계 런처 — LC-A 완주를 기다렸다가 스모크 게이트 통과 시 본 런을 개시한다
# (2026-08-25; 레시피·근거는 configs/training/lc_b.yaml 헤더, 게이트는 STATUS.md LC-B 행).
#
#   nohup bash scripts/launch_lc_b_after_lc_a.sh > outputs/lc_b_launcher.log 2>&1 &
#
# 단계:
#   ① LC-A 완주 대기 — ckpt latest == 1113. 프로세스 사망 + ckpt<1113이면 연계 중단
#      (죽은 런에 이어붙이면 안 된다 — 사람이 LC-A resume을 판단해야 함).
#   ② GPU 유휴 대기 (잔류 torchrun EADDRINUSE 함정 — 프로세스 소멸까지 대기)
#   ③ lc_b_preflight.py --deep --require-iter 1113
#   ④ 10-iter 스모크: THD×128K×CP8×offload 첫 실데이터 조합 (S5는 dense+mock).
#      WANDB_MODE=disabled + --train-samples 960 --no-save-optim (본 preset 그대로,
#      예산만 축소 — 별도 smoke yaml을 만들면 레시피가 드리프트한다).
#   ⑤ 스모크 판정: 10/10 완주 · loss 유한 · max-alloc ≤ SMOKE_MAX_ALLOC_MB ·
#      오프로더 배너 존재. 실패 시 HOLD (본 런 안 나감 — LC-A run1 iter2 OOM 교훈:
#      스모크 최대치 ≠ 프로덕션 최대치이므로 여유폭 있는 게이트만 자동 통과).
#   ⑥ 본 런 launch → ⑦ 첫 iteration 확인 (128k 콜드 캐시 — 최대 3.5h 허용)
#
# 모든 실패는 outputs/LC_B_CHAIN_ALERT.txt에 사유를 남기고 exit 1 (재시도 없음).
set -u
ALPHA=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha
LCA_CKPT=$ALPHA/outputs/alpha_baseline_48L_lc_a_resume_20260823_070651/checkpoints
LCA_TOTAL_ITERS=1113
SMOKE_MAX_ALLOC_MB=66560   # 65GB: S5 포락선 54.9~58.8GB + 실데이터 여유. 초과 = HOLD
SMOKE_TIMEOUT_S=18000      # 5h: 콜드 캐시(~80분+) + 10 iters(~1h) + 종료 valid/save
FIRST_ITER_TIMEOUT_S=12600 # 3.5h: 본 런 캐시 재빌드(train-samples가 달라 인덱스 재생성)
cd "$ALPHA"

log() { echo "[$(date '+%F %T')] $*"; }
alert() {
    log "ALERT: $*"
    printf '%s\n%s\n' "$(date '+%F %T')" "$*" > "$ALPHA/outputs/LC_B_CHAIN_ALERT.txt"
    exit 1
}

# ① LC-A 완주 대기 — 판정은 ckpt 기준 (완주 저장은 training.py 종료 시 무조건 수행)
log "waiting for LC-A completion (ckpt $LCA_CKPT -> $LCA_TOTAL_ITERS)"
while true; do
    it=$(cat "$LCA_CKPT/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    [ "$it" = "$LCA_TOTAL_ITERS" ] && break
    if ! pgrep -f "[p]retrain_alpha.py" > /dev/null; then
        alert "LC-A process gone but ckpt=$it < $LCA_TOTAL_ITERS — died mid-run; NOT chaining (LC-A resume 필요)"
    fi
    sleep 600
done
log "LC-A COMPLETE (ckpt $it)"

# ② GPU 유휴 대기
while true; do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    if [ "$busy" -lt 1000 ] && ! pgrep -f "[p]retrain_alpha.py" > /dev/null; then
        break
    fi
    log "GPUs busy (sum ${busy}MiB) — waiting"
    sleep 120
done
sleep 60
log "GPUs idle"

# ③ 프리플라이트
if ! python3 scripts/lc_b_preflight.py --deep --require-iter "$LCA_TOTAL_ITERS" --skip-gpu; then
    alert "LC-B preflight failed — 사람 확인 필요"
fi

# ④ 스모크 launch
runs_before=$(ls -d "$ALPHA"/outputs/alpha_baseline_48L_lc_b_* 2>/dev/null | tr '\n' ' ')
WANDB_MODE=disabled nohup bash train.sh baseline_48L lc_b lc_b_128k_blend \
    --train-samples 960 --no-save-optim \
    > "$ALPHA/outputs/lc_b_smoke_stdout.log" 2>&1 &
SMOKE_PID=$!
log "smoke launched (pid $SMOKE_PID)"
SMOKE_DIR=""
for i in $(seq 1 20); do
    sleep 30
    for d in $(ls -td "$ALPHA"/outputs/alpha_baseline_48L_lc_b_* 2>/dev/null); do
        case " $runs_before " in *" $d "*) ;; *) SMOKE_DIR=$d ;; esac
    done
    [ -n "$SMOKE_DIR" ] && break
done
[ -n "$SMOKE_DIR" ] || alert "smoke run dir not created in 10min — check outputs/lc_b_smoke_stdout.log"
log "smoke run dir: $SMOKE_DIR"

# ⑤ 스모크 완료 대기 + 판정
waited=0
while kill -0 "$SMOKE_PID" 2>/dev/null; do
    sleep 120
    waited=$((waited + 120))
    if [ "$waited" -ge "$SMOKE_TIMEOUT_S" ]; then
        kill "$SMOKE_PID" 2>/dev/null
        alert "smoke exceeded ${SMOKE_TIMEOUT_S}s — killed; $SMOKE_DIR 확인 필요"
    fi
done
SLOG=$(ls "$SMOKE_DIR"/logs/train_*.log 2>/dev/null | head -1)
[ -n "$SLOG" ] || alert "smoke log missing in $SMOKE_DIR"

grep -qE "iteration +10/" "$SLOG" || alert "smoke did not reach iteration 10 — $SLOG"
if grep -E "lm loss:" "$SLOG" | grep -qiE "nan|inf"; then
    alert "smoke loss non-finite — $SLOG"
fi
if grep -qi "CUDA out of memory" "$SLOG"; then
    alert "smoke OOM — $SLOG"
fi
# ckpt 로드 실패(무작위 초기화)면 loss ~11 — LC-A 종반(~1.45) 연속성 게이트
first_loss=$(grep -m1 -oE "lm loss: [0-9.eE+-]+" "$SLOG" | awk '{print $3}')
[ -n "$first_loss" ] || alert "smoke has no lm loss line — $SLOG"
if ! awk -v l="$first_loss" 'BEGIN {exit !(l+0 < 3.0)}'; then
    alert "smoke first loss $first_loss >= 3.0 — ckpt 미로드 의심, HOLD"
fi
maxalloc=$(grep -oE "max allocated: [0-9.]+" "$SLOG" | awk '{if ($3>m) m=$3} END {printf "%.0f", m}')
[ -n "$maxalloc" ] || alert "smoke has no memory report — $SLOG"
if [ "$maxalloc" -gt "$SMOKE_MAX_ALLOC_MB" ]; then
    alert "smoke max-alloc ${maxalloc}MB > gate ${SMOKE_MAX_ALLOC_MB}MB — HOLD (프로덕션 꼬리 OOM 위험, 사람 판단 필요)"
fi
offl=$(grep -c "chunked optimizer state offload: enabled" "$SLOG" || true)
[ "$offl" -ge 1 ] || alert "offloader banner absent — offload 미활성 의심, HOLD"
log "SMOKE PASS: 10/10 iters, max-alloc ${maxalloc}MB (gate ${SMOKE_MAX_ALLOC_MB}), offload banner x${offl}"
grep -m3 -E "iteration +(8|9|10)/" "$SLOG" || true

# ⑥ GPU 유휴 재확인 후 본 런
while true; do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    [ "$busy" -lt 1000 ] && break
    log "GPUs draining after smoke (sum ${busy}MiB)"
    sleep 60
done
sleep 60
runs_before=$(ls -d "$ALPHA"/outputs/alpha_baseline_48L_lc_b_* 2>/dev/null | tr '\n' ' ')
nohup bash train.sh baseline_48L lc_b lc_b_128k_blend \
    > "$ALPHA/outputs/lc_b_train_stdout.log" 2>&1 &
TRAIN_PID=$!
log "LC-B main run launched (pid $TRAIN_PID)"

# ⑦ 첫 iteration 확인
MAIN_DIR=""
waited=0
while [ "$waited" -lt "$FIRST_ITER_TIMEOUT_S" ]; do
    sleep 60
    waited=$((waited + 60))
    if [ -z "$MAIN_DIR" ]; then
        for d in $(ls -td "$ALPHA"/outputs/alpha_baseline_48L_lc_b_* 2>/dev/null); do
            case " $runs_before " in *" $d "*) ;; *) MAIN_DIR=$d ;; esac
        done
    fi
    L=$(ls "$MAIN_DIR"/logs/train_*.log 2>/dev/null | head -1)
    if [ -n "$L" ] && grep -qE "iteration +1/" "$L"; then
        log "LC-B TRAINING RUNNING: $MAIN_DIR"
        grep -m2 -E "iteration +[12]/" "$L"
        exit 0
    fi
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        alert "LC-B main run exited before first iteration — check outputs/lc_b_train_stdout.log"
    fi
done
alert "no first iteration within ${FIRST_ITER_TIMEOUT_S}s — investigate $MAIN_DIR"
