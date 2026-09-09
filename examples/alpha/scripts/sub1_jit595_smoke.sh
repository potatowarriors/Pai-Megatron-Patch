#!/bin/bash
# sub1_jit595_smoke.sh — sub1 에서 프리셋·블렌드 2-iter 스모크(G-P5). sudo·symlink 변경 없이 595 JIT 링크 디렉토리를
# LD_LIBRARY_PATH 앞에 두고 train.sh 를 돌린다 (2026-09-09; 기본값은 phase-3 터미널 보정 스테이지).
#
# 배경: sub1 은 compat libcuda 595 + JIT(ptxjitcompiler·nvvm) 570 혼합 상태라 학습 첫 스텝에서 munmap_chunk 로 죽는다
#   (docs/KNOWN_ISSUES.md 2026-09-04). 09-07 NCCL 재현에서 같은 혼합이 원인으로 확정됐고 jit595 디렉토리가 해결했다(〃 09-07).
#   학습 스택에도 같은 처방이 통하는지는 미검증 → 이 스크립트가 그 검증이자 phase-3 G-P5 스모크다.
#   대안 sub1_compat_smoke.sh 는 sudo 로 symlink 를 570 으로 되돌려야 해 root 가 필요하다.
#
#   sub1 에서: nohup bash scripts/sub1_jit595_smoke.sh [training] [data] > outputs/sub1_jit595_smoke.log 2>&1 < /dev/null &
#   main1 에서: ssh sub1 'bash -lc "nohup bash <abs>/scripts/sub1_jit595_smoke.sh > <abs>/outputs/sub1_jit595_smoke.log 2>&1 < /dev/null &"'
#
# 단계: ① sub1·GPU 유휴·jit595 존재 확인 ② train.sh <training> <data> --exit-interval 2 --save "" (WANDB 차단)
#       ③ 판정: iteration 1·2 존재 · lm loss 유한 · munmap/Traceback/OOM 0 → outputs/smoke_<data>_sub1_jit595_<ts>.summary.txt
# 부수효과: configs/data/.cache/<data> 인덱스 캐시가 NFS 에 만들어져 main1 본 런이 재사용한다 (콜드 캐시 80분+ 절감).
# 주의: --save "" 라 ckpt 는 쓰지 않는다. 런 디렉터리 outputs/alpha_<model>_<training>_<ts>/ 는 로그용으로 남는다.
set -u
ALPHA=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha
JIT595=/home/work/vidsearch/tools/cuda_compat13/jit595
TRAIN=${1:-sft_128k_terminal_p3}; DATA=${2:-sft_128k_terminal_blend_p3}; MODEL=${MODEL:-baseline_48L}
SMOKE_TIMEOUT_S=${SMOKE_TIMEOUT_S:-10800}
cd "$ALPHA"
log() { echo "[$(date '+%F %T')] $*"; }
[ "$(hostname)" = sub1 ] || { log "이 스크립트는 sub1 에서만 실행한다 (hostname=$(hostname))"; exit 2; }
[ -e "$JIT595/libnvidia-ptxjitcompiler.so.1" ] && [ -e "$JIT595/libnvidia-nvvm.so.4" ] || { log "ALERT: $JIT595 링크 없음"; exit 1; }
busy=$(pgrep -f "[v]llm|[l]b_proxy|[p]retrain_alpha|[a]pi_server" | wc -l)
mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
[ "$busy" -eq 0 ] && [ "$mib" -lt 1000 ] || { log "ALERT: sub1 busy (procs=$busy gpu=${mib}MiB) — 중단"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
LOG="$ALPHA/outputs/smoke_${DATA}_sub1_jit595_${TS}.log"
SUM="$ALPHA/outputs/smoke_${DATA}_sub1_jit595_${TS}.summary.txt"
log "libcuda: $(readlink /usr/local/cuda/compat/lib.real/libcuda.so.1 2>/dev/null || echo '?') ; JIT: $(readlink "$JIT595/libnvidia-ptxjitcompiler.so.1")"
log "smoke start: train.sh $MODEL $TRAIN $DATA --exit-interval 2 --save '' (timeout ${SMOKE_TIMEOUT_S}s) -> $LOG"
LD_LIBRARY_PATH="$JIT595${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" WANDB_MODE=disabled PYTHONFAULTHANDLER=1 \
  timeout -k 120 "$SMOKE_TIMEOUT_S" bash train.sh "$MODEL" "$TRAIN" "$DATA" --exit-interval 2 --save "" > "$LOG" 2>&1
RC=$?
if [ "$RC" -eq 124 ]; then log "timeout — 잔류 학습 프로세스 정리"; pkill -TERM -f "[p]retrain_alpha.py"; sleep 20; fi

# ③ 판정 (sub1_compat_smoke.sh 와 동일 기준)
it1=$(grep -E "iteration +1/ " "$LOG" | head -1); it2=$(grep -E "iteration +2/ " "$LOG" | head -1)
l1=$(echo "$it1" | grep -o "lm loss: [0-9.eE+-]*" | awk '{print $3}'); l2=$(echo "$it2" | grep -o "lm loss: [0-9.eE+-]*" | awk '{print $3}')
bad=$(grep -c -E "munmap_chunk|Traceback|CUDA out of memory|Fatal Python error" "$LOG")
mem=$(grep -o "max allocated: [0-9.]*" "$LOG" | tail -1)
ncache=$(ls "configs/data/.cache/$DATA" 2>/dev/null | wc -l)
verdict=FAIL
if [ -n "$it1" ] && [ -n "$it2" ] && [ "$bad" -eq 0 ]; then
  case "$l1$l2" in *nan*|*inf*|*NaN*) verdict=FAIL;; *) verdict=PASS;; esac
fi
{
  echo "sub1 jit595 G-P5 smoke  $TS  $MODEL/$TRAIN/$DATA  verdict=$verdict  (train.sh rc=$RC, timeout ${SMOKE_TIMEOUT_S}s)"
  echo "libcuda: $(readlink /usr/local/cuda/compat/lib.real/libcuda.so.1 2>/dev/null || echo '?') ; LD_LIBRARY_PATH 선두: $JIT595"
  echo "iter1: ${it1:-<none>}"; echo "iter2: ${it2:-<none>}"
  echo "lm loss: ${l1:-?} -> ${l2:-?} ; ${mem:-max allocated: ?} ; error lines: $bad ; data cache files: $ncache"
  echo "log: $LOG"
} | tee "$SUM"
log "done verdict=$verdict"
[ "$verdict" = PASS ]
