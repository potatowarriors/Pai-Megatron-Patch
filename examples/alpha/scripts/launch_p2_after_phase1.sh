#!/bin/bash
# SFT phase-2 자동 연계 런처 — phase-1(교체 재개 런) 2448 완주를 기다렸다가 phase-2 본 런을 개시한다
# (2026-09-04; 설계·게이트는 docs/SFT_PHASE2_PLAN.md §11, 선례는 launch_lc_b_after_lc_a.sh).
#
#   nohup bash scripts/launch_p2_after_phase1.sh > outputs/p2_launcher.log 2>&1 &
#   해제: pkill -f "[l]aunch_p2_after_phase1.sh"   (본 런이 이미 떴으면 본 런은 별도 종료)
#
# 단계:
#   ① phase-1 완주 대기 — ckpt latest == 2448. 프로세스 사망 + ckpt<2448 이면 연계 중단 (사람이 resume 판단).
#   ② GPU 유휴 대기 (잔류 torchrun EADDRINUSE 함정 — 프로세스 소멸까지 대기)
#   ③ sanity — 블렌드 49 경로 idx·합 1.0, preset load 가 완주 ckpt 디렉터리, latest 2448.
#   ④ 본 런 launch (G-P5 스모크는 2026-09-04 sub1 에서 iter 1500 으로 선행 — 캐시 프리빌드 포함, §11.4)
#   ⑤ 첫 iteration 확인 (캐시 재사용이면 수 분, 콜드면 ≤ FIRST_ITER_TIMEOUT_S) · loss 유한 · Traceback 없음
# 모든 실패는 outputs/P2_CHAIN_ALERT.txt 에 사유를 남기고 exit 1 (재시도 없음).
set -u
ALPHA=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha
P1_CKPT=$ALPHA/outputs/alpha_baseline_48L_sft_128k_full_swap_20260901_101523/checkpoints
P1_TOTAL_ITERS=2448
FIRST_ITER_TIMEOUT_S=10800   # 3h: 콜드 캐시 49 멤버(≈45분 추정) + 로드 + 첫 스텝. 프리빌드 캐시면 훨씬 빠름
cd "$ALPHA"

log() { echo "[$(date '+%F %T')] $*"; }
alert() {
    log "ALERT: $*"
    printf '%s\n%s\n' "$(date '+%F %T')" "$*" > "$ALPHA/outputs/P2_CHAIN_ALERT.txt"
    exit 1
}

# ① phase-1 완주 대기
log "waiting for phase-1 completion (ckpt $P1_CKPT -> $P1_TOTAL_ITERS)"
while true; do
    it=$(cat "$P1_CKPT/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
    [ "$it" = "$P1_TOTAL_ITERS" ] && break
    if ! pgrep -f "[p]retrain_alpha.py" > /dev/null; then
        alert "phase-1 process gone but ckpt=$it < $P1_TOTAL_ITERS — died mid-run; NOT chaining (resume 필요)"
    fi
    sleep 600
done
log "phase-1 COMPLETE (ckpt $it)"

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

# ③ sanity
python3 - "$P1_CKPT" "$P1_TOTAL_ITERS" <<'EOF' || alert "phase-2 sanity failed — 사람 확인 필요"
import os, re, sys, yaml
ckpt, total = sys.argv[1], int(sys.argv[2])
p = yaml.safe_load(open("configs/training/sft_128k_full_p2.yaml"))
assert p["load"] == ckpt, ("preset load", p["load"])
assert p.get("finetune") is True and p.get("no-load-optim") is True, "stage 전환 키"
assert os.path.isdir(os.path.join(ckpt, f"iter_{total:07d}")), "완주 ckpt 디렉터리 없음"
d = yaml.safe_load(open("configs/data/sft_128k_mixed_blend_p2.yaml"))
toks = d["data-path"].split(); ws = [float(x) for x in toks[0::2]]; paths = toks[1::2]
assert len(paths) == 49, len(paths)
assert abs(sum(ws) - 1) < 1e-5, sum(ws)
missing = [x for x in paths if not os.path.exists(x + ".idx")]
assert not missing, missing
print(f"sanity OK: 49 members, sum {sum(ws):.6f}, train-samples {p['train-samples']}, load {ckpt}")
EOF

# ④ 본 런 launch
runs_before=$(ls -d "$ALPHA"/outputs/alpha_baseline_48L_sft_128k_full_p2_* 2>/dev/null | tr '\n' ' ')
nohup bash train.sh baseline_48L sft_128k_full_p2 sft_128k_mixed_blend_p2 \
    > "$ALPHA/outputs/p2_main_stdout.log" 2>&1 &
MAIN_PID=$!
log "phase-2 main run launched (pid $MAIN_PID)"
RUN_DIR=""
for i in $(seq 1 20); do
    sleep 30
    for d in $(ls -td "$ALPHA"/outputs/alpha_baseline_48L_sft_128k_full_p2_* 2>/dev/null); do
        case " $runs_before " in *" $d "*) ;; *) RUN_DIR=$d ;; esac
    done
    [ -n "$RUN_DIR" ] && break
done
[ -n "$RUN_DIR" ] || alert "phase-2 run dir not created — see outputs/p2_main_stdout.log"
log "run dir: $RUN_DIR"

# ⑤ 첫 iteration 확인
deadline=$(( $(date +%s) + FIRST_ITER_TIMEOUT_S ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    LOGF=$(ls -t "$RUN_DIR"/logs/*.log 2>/dev/null | head -1)
    if [ -n "$LOGF" ]; then
        if grep -q -E "Traceback|CUDA out of memory" "$LOGF" "$ALPHA/outputs/p2_main_stdout.log" 2>/dev/null; then
            alert "phase-2 first steps raised Traceback/OOM — $LOGF"
        fi
        line=$(grep -E "iteration +[0-9]+/ +[0-9]+" "$LOGF" | head -1)
        if [ -n "$line" ]; then
            loss=$(echo "$line" | grep -o "lm loss: [0-9.eE+-]*" | awk '{print $3}')
            log "first iteration: $line"
            case "$loss" in ""|nan|NaN|inf) alert "first-iteration loss not finite: '$loss'";; esac
            log "phase-2 CHAIN OK — monitor with: tail -f $LOGF"
            exit 0
        fi
    fi
    if ! kill -0 "$MAIN_PID" 2>/dev/null && ! pgrep -f "[p]retrain_alpha.py" > /dev/null; then
        alert "phase-2 process exited before first iteration — see $ALPHA/outputs/p2_main_stdout.log"
    fi
    sleep 60
done
alert "phase-2 first iteration not seen within ${FIRST_ITER_TIMEOUT_S}s"
