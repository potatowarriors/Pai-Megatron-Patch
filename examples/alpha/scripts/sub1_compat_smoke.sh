#!/bin/bash
# sub1 정식 G-P5 스모크 — CUDA compat 를 570 으로 잠시 되돌려 phase-2 프리셋·블렌드를 2 iter 돌리고 595 로 복원한다.
# (2026-09-04; 배경은 docs/KNOWN_ISSUES.md 2026-09-04 "sub1 은 Megatron 학습을 못 돌린다". 사용자 지시: sub1 fleet 은 09-05 종료,
#  그 뒤에 테스트.)
#
#   sub1 에서:  nohup bash scripts/sub1_compat_smoke.sh --wait-after <epoch> > outputs/sub1_compat_smoke_watcher.log 2>&1 &
#   즉시 실행:  bash scripts/sub1_compat_smoke.sh --now
#
# 단계:
#   ① (--wait-after) 지정 시각 이후, GPU 유휴(합 <1000MiB) + vllm/lb_proxy/pretrain 프로세스 없음이 IDLE_MIN 분 연속이면 시작
#   ② 현재 compat symlink 기록 → libcuda.so.1 → 570.124.06 (sudo -n) → nvidia-smi CUDA Version 12.8 확인
#   ③ 스모크: train.sh baseline_48L sft_128k_full_p2 sft_128k_mixed_blend_p2 --exit-interval 2 --save "" (WANDB 차단, 캐시 재사용)
#   ④ 판정: iteration 1·2 존재 · lm loss 유한 · munmap/Traceback 없음 → outputs/smoke_p2_sub1_compat570_<ts>.summary.txt
#   ⑤ 복원: 어떤 경로로 끝나든 trap 으로 symlink 를 원래(595)로 되돌리고 CUDA Version 13.2 확인
# G-P5 기준(§11.4): loss 유한·traceback 0. 절대값은 신규 도메인 때문에 phase-1 최종(≈0.70)보다 높을 수 있어 기록만 한다.
set -u
ALPHA=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha
REAL=/usr/local/cuda/compat/lib.real
TARGET=libcuda.so.570.124.06
IDLE_MIN=${IDLE_MIN:-20}
SMOKE_TIMEOUT_S=${SMOKE_TIMEOUT_S:-3600}
MODE=""; WAIT_AFTER=0
while [ $# -gt 0 ]; do
  case "$1" in
    --now) MODE=now; shift;;
    --wait-after) MODE=wait; WAIT_AFTER=$2; shift 2;;
    *) echo "usage: $0 --now | --wait-after <epoch>"; exit 2;;
  esac
done
[ -n "$MODE" ] || { echo "usage: $0 --now | --wait-after <epoch>"; exit 2; }
[ "$(hostname)" = sub1 ] || { echo "이 스크립트는 sub1 에서만 실행한다 (hostname=$(hostname))"; exit 2; }
cd "$ALPHA"
log() { echo "[$(date '+%F %T')] $*"; }

busy_procs() { pgrep -f "[v]llm|[l]b_proxy|[p]retrain_alpha|[a]pi_server" | wc -l; }
gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}'; }

# ① 대기
if [ "$MODE" = wait ]; then
  log "waiting until epoch $WAIT_AFTER ($(date -d @"$WAIT_AFTER" '+%F %T')) then ${IDLE_MIN}min idle (no vllm/lb_proxy/pretrain, GPU <1000MiB)"
  while [ "$(date +%s)" -lt "$WAIT_AFTER" ]; do sleep 300; done
  idle_since=""
  while true; do
    if [ "$(busy_procs)" -eq 0 ] && [ "$(gpu_mib)" -lt 1000 ]; then
      [ -n "$idle_since" ] || idle_since=$(date +%s)
      if [ $(( $(date +%s) - idle_since )) -ge $(( IDLE_MIN * 60 )) ]; then break; fi
    else
      idle_since=""
    fi
    sleep 60
  done
  log "sub1 idle for ${IDLE_MIN}min — starting"
fi

# ② compat 570
ORIG=$(readlink "$REAL/libcuda.so.1")
[ -f "$REAL/$TARGET" ] || { log "ALERT: $REAL/$TARGET 없음 — 중단"; exit 1; }
restore() {
  sudo -n ln -sf "$ORIG" "$REAL/libcuda.so.1"
  log "compat restored -> $(readlink "$REAL/libcuda.so.1") (nvidia-smi: $(nvidia-smi -q | grep -i 'cuda version' | awk -F: '{print $2}' | xargs))"
}
trap restore EXIT
log "compat before: $ORIG"
sudo -n ln -sf "$TARGET" "$REAL/libcuda.so.1" || { log "ALERT: sudo ln 실패"; exit 1; }
CV=$(nvidia-smi -q | grep -i 'cuda version' | awk -F: '{print $2}' | xargs)
log "compat now: $(readlink "$REAL/libcuda.so.1") (nvidia-smi CUDA Version $CV)"
case "$CV" in 12.*) ;; *) log "ALERT: CUDA Version 이 12.x 가 아님($CV) — 스모크 중단"; exit 1;; esac

# ③ 스모크
TS=$(date +%Y%m%d_%H%M%S)
LOG="$ALPHA/outputs/smoke_p2_sub1_compat570_${TS}.log"
SUM="$ALPHA/outputs/smoke_p2_sub1_compat570_${TS}.summary.txt"
log "smoke start -> $LOG"
WANDB_MODE=disabled PYTHONFAULTHANDLER=1 timeout "$SMOKE_TIMEOUT_S" \
  bash train.sh baseline_48L sft_128k_full_p2 sft_128k_mixed_blend_p2 --exit-interval 2 --save "" > "$LOG" 2>&1
RC=$?

# ④ 판정
it1=$(grep -E "iteration +1/ " "$LOG" | head -1); it2=$(grep -E "iteration +2/ " "$LOG" | head -1)
l1=$(echo "$it1" | grep -o "lm loss: [0-9.eE+-]*" | awk '{print $3}'); l2=$(echo "$it2" | grep -o "lm loss: [0-9.eE+-]*" | awk '{print $3}')
bad=$(grep -c -E "munmap_chunk|Traceback|CUDA out of memory|Fatal Python error" "$LOG")
mem=$(grep -o "max allocated: [0-9.]*" "$LOG" | tail -1)
verdict=FAIL
if [ -n "$it1" ] && [ -n "$it2" ] && [ "$bad" -eq 0 ]; then
  case "$l1$l2" in *nan*|*inf*|*NaN*) verdict=FAIL;; *) verdict=PASS;; esac
fi
{
  echo "sub1 compat-570 G-P5 smoke  $TS  verdict=$verdict  (train.sh rc=$RC, timeout ${SMOKE_TIMEOUT_S}s)"
  echo "compat during smoke: $TARGET ; restored to: $ORIG"
  echo "iter1: ${it1:-<none>}"; echo "iter2: ${it2:-<none>}"
  echo "lm loss: ${l1:-?} -> ${l2:-?} ; ${mem:-max allocated: ?} ; error lines: $bad"
  echo "log: $LOG"
} | tee "$SUM"
log "done verdict=$verdict"
[ "$verdict" = PASS ]
