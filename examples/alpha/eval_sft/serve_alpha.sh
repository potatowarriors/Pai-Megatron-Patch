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

# vLLM 컴파일 캐시(torch_compile_cache 등)의 기본 위치는 ~/.cache/vllm 이다. sub1 의 HOME(/home/work)은 **49GB 루프
# 볼륨**이라 uv·pip·HF 캐시와 나눠 쓰다 가득 찼고, 2026-09-14 iter2448 fleet 8대가 profile_run 의 컴파일 결과 저장에서
# `OSError: [Errno 28] No space left on device` 로 전부 기동 실패했다(스위트는 준비 대기 20분 뒤에야 끝난다).
# 기본을 로컬 오버레이 /tmp(수백 GB)로 둔다. 컨테이너 재시작 시 사라지지만 캐시라 재컴파일로 복구된다.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache}"
mkdir -p "$VLLM_CACHE_ROOT"
CACHE_AVAIL_GB=$(df -BG --output=avail "$VLLM_CACHE_ROOT" | tail -1 | tr -dc '0-9')
if [ "${CACHE_AVAIL_GB:-0}" -lt "${MIN_CACHE_GB:-20}" ]; then
  echo "[serve] ❌ VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT 여유 ${CACHE_AVAIL_GB}GB < ${MIN_CACHE_GB:-20}GB — 컴파일 캐시 저장이 ENOSPC 로 죽는다. 공간 확보 또는 VLLM_CACHE_ROOT 변경" >&2
  exit 1
fi

TOOL_FLAGS=""
if [ "${TOOLS:-0}" = "1" ]; then
  # mini-swe-agent/litellm 이 tool_choice=auto 를 보냄 → vLLM 이 수용하도록.
  # (에이전트는 텍스트 파싱이라 파서 종류 무관; hermes 로 요청만 통과시킴)
  TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser ${TOOL_PARSER:-qwen3_xml}"
fi
REASON_FLAGS=""
if [ -n "${REASONING_PARSER:-}" ]; then
  # think 를 reasoning 필드로 분리. vLLM 0.25.1 parser engine 은 tool 파서(TOOLS=1)가 켜지면 도구 선언 여부와 무관하게
  # </think> 토큰을 소비해, reasoning 파서 없이는 content 가 think+답변이 마커 없이 붙는다 (2026-09-14 실측, §3.13·§3.14).
  # 에이전틱·τ³ fleet 는 nemotron_v3 로 켠다. T1/T3/T2 fleet 는 끈다 — G2 게이트와 T1 채점이 content 의 </think> 를 본다.
  REASON_FLAGS="--reasoning-parser $REASONING_PARSER"
fi
echo "[serve] ckpt=$CKPT max_len=$MAX_LEN DP=$DP port=$PORT tools=${TOOLS:-0} reasoning=${REASONING_PARSER:-off} cache=$VLLM_CACHE_ROOT(${CACHE_AVAIL_GB}GB free)"
exec $VENV/bin/vllm serve "$CKPT" \
  $TOOL_FLAGS $REASON_FLAGS \
  --served-model-name alpha \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --data-parallel-size "$DP" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port "$PORT"
