#!/bin/bash
# serve_glm53_main1.sh — main1 GLM-5.3-Flash 교사 서빙 (ko_chat v3 합성 트랙)
#
# sub1 판(sdg/terminal/serve/serve_glm53.sh)과의 차이:
#   - venv: glm_serve_venv_cu129 (main1 드라이버 535 + cu129 torch 로 CUDA 정상 — 실측 matmul OK).
#   - jit595 우회 없음: main1 은 sub1 의 compat 반영 사고(595/570 혼합)와 무관하다.
#   - 포트 8000, all-8 GPU EP+DP8.
# 핵심 플래그는 동일: --reasoning-parser glm45 (네이티브 <think> 를 reasoning 필드로 분리 = 이 트랙의 목적),
#   --tool-call-parser glm47 --enable-auto-tool-choice (도구 비상관 슬라이스용).
#
# 실행: nohup bash serve_glm53_main1.sh > /home/work/vidsearch/tools/glm53/kochat_v3/serve_<ts>.log 2>&1 &
# 중지: pkill -f 'glm_serve_venv_cu129/bin/vllm serve .*GLM-5.3-Flas[h]'
set -uo pipefail
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-131072}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
MAX_SEQS="${MAX_SEQS:-96}"
MODEL="${MODEL:-/home/work/vidsearch/models/GLM-5.3-Flash}"
VENV=/home/work/vidsearch/tools/glm_serve_venv_cu129

export PIP_CONSTRAINT=
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
export VLLM_ENGINE_READY_TIMEOUT_S=3600

echo "[serve-main1] model=$MODEL EP+DP8 max_len=$MAX_LEN batched=$MAX_BATCHED_TOKENS seqs=$MAX_SEQS kv=${KV_DTYPE:-fp8} prefix=${PREFIX_CACHE:---enable-prefix-caching} port=$PORT"
exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name glm53-flash \
  --enable-expert-parallel --data-parallel-size 8 \
  --max-model-len "$MAX_LEN" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_SEQS" \
  --kv-cache-dtype "${KV_DTYPE:-fp8}" \
  ${PREFIX_CACHE:---enable-prefix-caching} \
  --gpu-memory-utilization 0.90 \
  --no-enable-flashinfer-autotune \
  --tool-call-parser glm47 --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --host 0.0.0.0 --port "$PORT"
