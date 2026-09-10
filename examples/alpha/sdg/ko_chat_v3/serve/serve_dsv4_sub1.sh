#!/bin/bash
# serve_dsv4_sub1.sh — sub1 DeepSeek-V4-Flash-0731 (공식 FP8 릴리스) 교사 서빙. ko_chat v3 2번째 생성 교사 (사용자 결정 2026-09-10).
#
# 근거: FP4 indexer cache·NVFP4 경로는 Blackwell 전용이지만, all-FP8 0731 체크포인트는 Hopper 에서 Triton MoE 백엔드로 서빙
#   (LMCache/vLLM 레시피: "on H100 you must run the FP8-quantized release", "serve with the Triton MoE backend"; H200 은 4 GPU/레플리카).
#   우리 glm_serve_venv(0.28.1rc dev485)는 DeepseekV4ForCausalLM·deepseek_v4 reasoning parser·MLA 백엔드를 등록하고 있다 —
#   dev 브랜치가 V4-Flash 에 깨졌다는 보고가 있어(레시피 "use latest release") 실서빙으로 검증하고, 실패 시 릴리스 venv 를 새로 만든다.
# 플래그: --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4, KV fp8, TP8+EP,
#   Hopper 는 FP4 indexer cache 비활성(--attention_config.use_fp4_indexer_cache False), --moe-backend triton.
# 실행: nohup bash serve_dsv4_sub1.sh > /home/work/vidsearch/tools/glm53/kochat_v3/serve_dsv4_<ts>.log 2>&1 &
set -uo pipefail
PORT="${PORT:-8300}"
MAX_LEN="${MAX_LEN:-131072}"
MODEL="${MODEL:-/home/work/vidsearch/models/DeepSeek-V4-Flash-0731}"
MAX_SEQS="${MAX_SEQS:-128}"
GPU_UTIL="${GPU_UTIL:-0.90}"
VENV="${VENV:-/home/work/vidsearch/tools/glm_serve_venv}"

export PIP_CONSTRAINT=
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export VLLM_ENGINE_READY_TIMEOUT_S=3600
JIT595=/home/work/vidsearch/tools/cuda_compat13/jit595
[ -e "$JIT595/libnvidia-ptxjitcompiler.so.1" ] && export LD_LIBRARY_PATH="$JIT595:${LD_LIBRARY_PATH:-}"

echo "[serve-dsv4] model=$MODEL TP8+EP kv=fp8 moe=triton seqs=$MAX_SEQS max_len=$MAX_LEN port=$PORT venv=$VENV"
exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name dsv4-flash \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --moe-backend "${MOE_BACKEND:-marlin}" \
  --kv-cache-dtype fp8 \
  --attention_config.use_fp4_indexer_cache False \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-prefix-caching \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  ${EXTRA:-} \
  --host 0.0.0.0 --port "$PORT"
