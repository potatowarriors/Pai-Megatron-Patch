#!/bin/bash
# sft_final_chain.sh — SFT 최종 런 A/B 체인 (docs/SFT_FINAL_PLAN.md §4.2, 사용자 승인 2026-09-13 "진행해").
#   ① A(main1 단일 150 iters) 의 iter 100 ckpt 가 나오면 sub1(유휴) 에서 경량 게이트 probe_ckpt.sh 실행 (GPU 0,1 · jit595)
#   ② A 가 iter 150 에서 종료하고 양 노드 GPU 가 비면 B(DiLoCo 2노드 150 iters) 기동 — launch_diloco.sh 는 node0 를 전경에서 돌리므로 이 스크립트가 끝까지 붙어 있는다.
# 사용: nohup bash scripts/sft_final_chain.sh <A_RUN_DIR> > outputs/sft_final_chain.log 2>&1 < /dev/null &
set -u
ALPHA=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha; cd "$ALPHA"
A_RUN="${1:?A run dir}"; A_LOG="${A_LOG:-$HOME/run_sft_final_A_main1.log}"
JIT595=/home/work/vidsearch/tools/cuda_compat13/jit595
log() { echo "[chain $(date '+%F %T')] $*"; }
latest() { { tr -d '[:space:]' < "$A_RUN/checkpoints/latest_checkpointed_iteration.txt"; } 2>/dev/null || echo 0; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}'; }
sub1_gpu_used() { ssh -o ConnectTimeout=15 sub1 "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=\$1} END {print s+0}'" 2>/dev/null || echo 999999; }

# ① iter 100 probe on sub1
log "A=$A_RUN — iter 100 ckpt 대기"
while [ "$(latest)" -lt 100 ]; do grep -q "exiting program\|Traceback" "$A_LOG" 2>/dev/null && break; sleep 120; done
if [ "$(latest)" -ge 100 ]; then
  log "iter 100 ckpt 확인 → sub1 probe 시작"
  ssh -o ConnectTimeout=15 sub1 "cd $ALPHA && LD_LIBRARY_PATH=$JIT595\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH} GPUS=0,1 PORT=8011 bash eval_sft/probe_ckpt.sh $A_RUN 100" > "$ALPHA/outputs/probe_A_iter100.log" 2>&1
  log "probe rc=$? → outputs/probe_A_iter100.log"; tail -3 "$ALPHA/outputs/probe_A_iter100.log"
fi

# ② A 종료 대기 → B
log "A 종료 대기 (exiting program at iteration 150)"
while ! grep -q "exiting program at iteration 150" "$A_LOG" 2>/dev/null; do
  grep -q "Traceback" "$A_LOG" 2>/dev/null && { log "A 에 Traceback — 체인 중단(사람 확인)"; exit 1; }
  sleep 120
done
log "A 종료 감지 (latest=$(latest)). GPU 회수 대기"
for i in $(seq 1 60); do [ "$(gpu_used)" -lt 2000 ] && [ "$(sub1_gpu_used)" -lt 2000 ] && break; sleep 20; done
[ "$(gpu_used)" -lt 2000 ] && [ "$(sub1_gpu_used)" -lt 2000 ] || { log "GPU 미회수(main1 $(gpu_used) / sub1 $(sub1_gpu_used) MiB) — 체인 중단"; exit 1; }
[ -d "$A_RUN/checkpoints/iter_0000150" ] || log "경고: A iter_0000150 ckpt 디렉터리 없음"
log "B 기동: launch_diloco.sh sft_final (H=30 τ=2 SHARD_BLOCK=160, NODE1_ENV jit595, --exit-interval 150)"
NODE1_ENV="LD_LIBRARY_PATH=$JIT595" DILOCO_DATA_SHARD=1 DILOCO_SHARD_BLOCK=160 DILOCO_H=30 DILOCO_TAU=2 \
  bash launch_diloco.sh sft_final baseline_48L sft_128k_final_diloco sft_128k_final_blend --exit-interval 150
log "B(launch_diloco) 종료 rc=$? — 로그 ~/run_diloco_sft_final_node0.log (main1) / node1.log (sub1 \$HOME)"
