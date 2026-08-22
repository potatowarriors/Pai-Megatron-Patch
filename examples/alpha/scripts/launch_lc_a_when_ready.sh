#!/bin/bash
# LC-A 자동 런처 — filler specialized 잔여분 도착을 기다렸다가 프리플라이트 통과 시
# 본 학습을 시작한다 (2026-08-22; 근거·구성은 configs/training/lc_a.yaml 헤더).
#
#   nohup bash scripts/launch_lc_a_when_ready.sh > outputs/lc_a_launcher.log 2>&1 &
#
# 단계: ① 블렌드 전 경로 .bin+.idx 등장 대기 → ② 쓰기 안정화 확인(연속 2회 크기
# 불변) → ③ preflight(--deep specialized: %16 표본검사 포함) → ④ GPU 유휴 대기 →
# ⑤ train.sh 본 런치(wandb ON). 프리플라이트 실패 시 즉시 중단(재시도 없음 —
# 데이터 문제는 사람이 봐야 한다).
set -u
ALPHA=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha
BLEND=$ALPHA/configs/data/lc_a_32k_blend.yaml
cd "$ALPHA"

log() { echo "[$(date '+%F %T')] $*"; }

paths=$(python3 -c "
import re
m = re.search(r'data-path: \"([^\"]+)\"', open('$BLEND').read())
t = m.group(1).split()
print('\n'.join(t[i+1] for i in range(0, len(t), 2)))")

# ① 전 경로 등장 대기
while true; do
    missing=0
    for p in $paths; do
        [ -f "$p.bin" ] && [ -f "$p.idx" ] || missing=$((missing+1))
    done
    [ "$missing" -eq 0 ] && break
    log "waiting for data: $missing prefixes incomplete"
    sleep 120
done
log "all blend paths present"

# ② 쓰기 안정화: 전체 크기 합이 2회 연속(3분 간격) 동일해야 통과
prev=-1
while true; do
    cur=$(for p in $paths; do stat -c %s "$p.bin" "$p.idx" 2>/dev/null; done | awk '{s+=$1} END {print s}')
    [ "$cur" = "$prev" ] && break
    prev=$cur
    log "sizes still settling (sum=$cur)"
    sleep 180
done
log "sizes stable"

# ③ 프리플라이트 (GPU 검사는 ④에서 별도)
if ! python3 scripts/lc_a_preflight.py --deep specialized --skip-gpu; then
    log "PREFLIGHT FAILED — aborting (사람 확인 필요)"
    exit 1
fi

# ④ GPU 유휴 대기
while true; do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    [ "$busy" -lt 1000 ] && break
    log "GPUs busy (sum ${busy}MiB) — waiting"
    sleep 120
done
log "GPUs idle — launching LC-A"

# ⑤ 본 런치 (wandb는 train.sh가 키 파일로 활성화; 세션과 독립 생존)
nohup bash train.sh baseline_48L lc_a lc_a_32k_blend > "$ALPHA/outputs/lc_a_train_stdout.log" 2>&1 &
TRAIN_PID=$!
log "train.sh started (pid $TRAIN_PID)"

# 기동 확인: 30분 내 첫 iteration 로그가 찍히는지
for i in $(seq 1 60); do
    sleep 30
    R=$(ls -td "$ALPHA"/outputs/alpha_baseline_48L_lc_a_* 2>/dev/null | head -1)
    L=$(ls "$R"/logs/train_*.log 2>/dev/null | head -1)
    if [ -n "$L" ] && grep -qE "iteration +1/" "$L"; then
        log "LC-A TRAINING RUNNING: $R"
        grep -m2 -E "iteration +[12]/" "$L"
        exit 0
    fi
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
        log "train.sh exited before first iteration — check $ALPHA/outputs/lc_a_train_stdout.log"
        exit 1
    fi
done
log "no first-iteration log within 30min — investigate ($R)"
exit 1
