#!/bin/bash
# serve_alpha.sh — sub1 alpha vLLM 서빙 (SFT 벤치 평가용)
# vllm 0.25.1 + vllm_alpha_plugin (AlphaForCausalLM 등록). 상세: docs/SFT_BENCHMARKS.md §3
#
# 사용: bash eval_sft/serve_alpha.sh <HF_CKPT_DIR> [MAX_LEN] [DP] [PORT]
#   MAX_LEN 기본 32768 (T1/T3). 롱컨텍스트(T2)는 131072 등으로.
#   DP = data-parallel 레플리카 수(각 TP=1). 기본 8 (8×H100 전부).
set -euo pipefail
CKPT="${1:?HF checkpoint dir required}"
MAX_LEN="${2:-32768}"
DP="${3:-8}"
PORT="${4:-8000}"
VENV=/home/work/vidsearch/tools/alpha_serve_venv
PLUGIN=/home/work/vidsearch/repos/project_s/NeMo-RL/examples/configs/alpha/vllm_alpha_plugin

# NGC 전역 PIP_CONSTRAINT가 venv torch를 오염시키지 않게 차단
export PIP_CONSTRAINT=
# alpha GDN은 TP>1 미지원 → TP=1, throughput은 DP 레플리카로
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# IB admin-disabled 클러스터 (ko_chat serve 전례)
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0

echo "[serve] ckpt=$CKPT max_len=$MAX_LEN DP=$DP port=$PORT"
exec $VENV/bin/vllm serve "$CKPT" \
  --served-model-name alpha \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --data-parallel-size "$DP" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port "$PORT"
