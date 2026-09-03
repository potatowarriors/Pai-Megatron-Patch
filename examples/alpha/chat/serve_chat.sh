#!/bin/bash
# serve_chat.sh — alpha SFT 체크포인트를 사람이 대화할 수 있게 띄운다 (OpenWebUI 백엔드).
#
# eval_sft/serve_alpha.sh 와의 차이 — 목적이 다르므로 설정도 다르다:
#   1. --reasoning-parser nemotron_v3  : <think> 를 reasoning 필드로 분리해 UI 가 접어서 표시.
#      (벤치는 파서 off 로 텍스트를 통째로 파싱한다. 그래서 G2 게이트는 여기서 의도적으로 FAIL 한다.)
#   1b. --enable-auto-tool-choice --tool-call-parser qwen3_xml
#      OpenWebUI 는 tool_choice="auto" 를 보낸다. 파서가 없으면 vLLM 이 요청을 거절한다:
#        '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
#      파서는 반드시 **XML** 계열이어야 한다 — 우리 chat template 은 도구 호출을
#      <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call> 로 지시한다.
#      hermes 계열은 <tool_call>{"name":...} JSON 을 기대해 어긋난다 (KNOWN_ISSUES 2026-08-30).
#   2. 단일 GPU · DP=1               : 1인 사용. 벤치의 8-레플리카 fleet 불필요.
#   3. 보수적 메모리 예산            : 40GB A100 슬라이스에 30GB 가중치가 들어간다.
#
# 사용: bash chat/serve_chat.sh [HF_CKPT] [MAX_LEN] [PORT] [GPU]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ALPHA="$(cd "$HERE/.." && pwd)"
CKPT="${1:-$ALPHA/outputs/alpha_baseline_48L_sft_128k_full_swap_20260901_101523/hfmodel_0001200}"
MAX_LEN="${2:-131072}"
PORT="${3:-8001}"
GPU="${4:-3}"
NAME="${SERVED_NAME:-alpha-v2-sft}"   # 별칭 `alpha` 도 함께 노출 — eval_sft/check_gates.py 가
                                      # 모델명 "alpha" 를 하드코딩하므로 게이트를 그대로 쓰려면 필요
VENV=/home/work/vidsearch/tools/alpha_serve_venv
LOGD=/home/work/vidsearch/tools/chat_logs; mkdir -p "$LOGD"

# NGC 전역 PIP_CONSTRAINT 가 venv torch 를 오염시키지 않게 차단
export PIP_CONSTRAINT=
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0

echo "[chat] ckpt=$CKPT"
echo "[chat] gpu=$GPU port=$PORT max_len=$MAX_LEN name=$NAME"
exec $VENV/bin/vllm serve "$CKPT" \
  --served-model-name "$NAME" alpha \
  --trust-remote-code \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-qwen3_xml}" \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 \
  --host 0.0.0.0 --port "$PORT"
