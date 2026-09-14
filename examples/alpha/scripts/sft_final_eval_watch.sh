#!/bin/bash
# sft_final_eval_watch.sh — SFT 최종 본 런(main1 단일)의 새 ckpt 를 sub1 에서 평가하는 감시 루프 (docs/SFT_FINAL_PLAN.md §4.3~4.4).
#   100 iters 마다: probe_ckpt.sh (MG→HF 변환 sub1 8 GPU + identity 프로브 + 유령 호출 재생) → eval_sft/results/probe/
#   300 iters 마다: eval_ckpt.sh t1 (8 GPU fleet: MMLU-Pro·GPQA-D·IFEval·AIME·HMMT) → eval_sft/results/
#   변환은 sub1 2 GPU 에서 SIGSEGV 가 났으므로 8 GPU 로 한다(09-13). vLLM 은 main1 드라이버가 낡아 sub1 전용. jit595 LD_LIBRARY_PATH 필수.
# 사용(sub1): nohup bash scripts/sft_final_eval_watch.sh <RUN_DIR> [POLL_SEC] > outputs/sft_final_eval_watch.log 2>&1 < /dev/null &
#   중단: touch <RUN_DIR>/.eval_watch_stop
set -u
ALPHA=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha; cd "$ALPHA"
RUN_DIR="${1:?RUN_DIR}"; POLL="${2:-600}"; STOP="$RUN_DIR/.eval_watch_stop"; rm -f "$STOP"
export LD_LIBRARY_PATH=/home/work/vidsearch/tools/cuda_compat13/jit595${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
log() { echo "[eval-watch $(date '+%F %T')] $*"; }
[ "$(hostname)" = sub1 ] || { log "sub1 에서만 실행 (hostname=$(hostname))"; exit 2; }
log "감시 시작 $RUN_DIR (poll ${POLL}s)"
while true; do
  [ -f "$STOP" ] && { log "STOP → 종료"; break; }
  for d in $(ls -d "$RUN_DIR"/checkpoints/iter_* 2>/dev/null | sort); do
    it=$((10#$(basename "$d" | sed 's/iter_//'))); ip=$(printf '%07d' "$it"); tag="$(basename "$RUN_DIR")_iter$ip"
    [ -f "$RUN_DIR/checkpoints/latest_checkpointed_iteration.txt" ] && [ "$(tr -d '[:space:]' < "$RUN_DIR/checkpoints/latest_checkpointed_iteration.txt")" -ge "$it" ] || continue   # 저장 완료된 것만
    if [ $((it % 100)) -eq 0 ] && [ ! -f "eval_sft/results/probe/$tag.done" ] && [ ! -f "eval_sft/results/probe/$tag.tried" ]; then
      log "probe iter $it"; touch "eval_sft/results/probe/$tag.tried"
      GPUS=0,1,2,3,4,5,6,7 PORT=8011 bash eval_sft/probe_ckpt.sh "$RUN_DIR" "$it" > "outputs/probe_${tag}.log" 2>&1; log "probe iter $it rc=$? → $(grep '판정' "eval_sft/results/probe/$tag.txt" 2>/dev/null | tail -1)"
    fi
    if [ $((it % 300)) -eq 0 ] && [ ! -f "eval_sft/results/$tag/.done" ] && [ ! -f "eval_sft/results/$tag.t1_tried" ]; then
      log "T1 iter $it"; touch "eval_sft/results/$tag.t1_tried"
      bash eval_sft/eval_ckpt.sh "$RUN_DIR" "$it" t1 > "outputs/t1_${tag}.log" 2>&1; log "T1 iter $it rc=$?"
    fi
  done
  for i in $(seq 1 $((POLL/10))); do [ -f "$STOP" ] && break; sleep 10; done
done
