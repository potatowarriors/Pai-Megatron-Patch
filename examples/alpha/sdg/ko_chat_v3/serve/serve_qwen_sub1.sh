#!/bin/bash
# serve_qwen_sub1.sh — sub1 Qwen3.8-Flash-Next-FP8 교사 서빙 (ko_chat v3 2번째 교사, 다양성·교차검증).
#
# arch = Qwen4ExpForConditionalGeneration (model_type qwen4_exp), FP8. glm_serve_venv(cu130, 0.28.1rc)이
# 레지스트리에 Qwen4Exp 를 등록한다(alpha_serve_venv 0.25.1 은 미등록). sub1 compat 우회(jit595)는 GLM 판과 동일.
# H100 51B PLE 임베딩 → TP8 로 8-way 샤딩 + PLE CPU offload 안전판. reasoning 파서는 REASONING_PARSER 로 재시도 가능.
#
# 실행: nohup bash serve_qwen_sub1.sh > /home/work/vidsearch/tools/glm53/kochat_v3/serve_qwen_<ts>.log 2>&1 &
set -uo pipefail
PORT="${PORT:-8300}"
MAX_LEN="${MAX_LEN:-131072}"
MODEL="${MODEL:-/home/work/vidsearch/models/Qwen3.8-Flash-Next-FP8}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
# vLLM 공식 레시피(recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next, 8×Hopper) 정합 (2026-09-10):
#   --max-num-seqs 256 ("keep 256" — Mamba-cache 오류 회피 + 처리량), --enable-prefix-caching (identity 시스템
#   프롬프트가 전 요청 공통 접두), --moe-backend triton (레시피 Hopper 권장; 자동 선택은 FLASHINFER_CUTLASS 였음 —
#   MOE_BACKEND= 로 비우면 자동), --tool-call-parser qwen3_xml. MTP 투기 디코딩은 레시피가 H100 에서 8~36% 느리다고
#   명시 → 미사용. 기준선(TP8+EP, seqs 64, cutlass): 동시 64 에서 2,135~2,367 tok/s.
MAX_SEQS="${MAX_SEQS:-256}"
MOE_BACKEND="${MOE_BACKEND-triton}"
GPU_UTIL="${GPU_UTIL:-0.90}"
VENV=/home/work/vidsearch/tools/glm_serve_venv

export PIP_CONSTRAINT=
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_PLE_CPU_OFFLOAD=1
# sub1 compat 우회 (serve_glm53.sh 와 동일 근거, KNOWN_ISSUES 09-07)
JIT595=/home/work/vidsearch/tools/cuda_compat13/jit595
[ -e "$JIT595/libnvidia-ptxjitcompiler.so.1" ] && export LD_LIBRARY_PATH="$JIT595:${LD_LIBRARY_PATH:-}"

MOE_FLAG=""; [ -n "$MOE_BACKEND" ] && MOE_FLAG="--moe-backend $MOE_BACKEND"
echo "[serve-qwen] model=$MODEL TP8+EP max_len=$MAX_LEN seqs=$MAX_SEQS moe=${MOE_BACKEND:-auto} reasoning=$REASONING_PARSER port=$PORT"
# TP8 단독은 전문가 gate/up 차원(80)이 8 로 안 나뉘어 실패 → 전문가는 EP 로 통째 분산(GLM 판과 동일 원리),
# attention/임베딩은 TP8 샤딩. EXTRA 로 추가 플래그 주입 가능.
exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name qwen38-flash-next \
  --tensor-parallel-size 8 --enable-expert-parallel \
  $MOE_FLAG \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --trust-remote-code \
  --reasoning-parser "$REASONING_PARSER" \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  ${EXTRA:-} \
  --host 0.0.0.0 --port "$PORT"
