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
# 하이브리드(GDN) prefix caching (2026-09-17, 사용자 결정). vLLM 0.25.1 은 하이브리드에 기본 OFF("experimental") 이지만
# Qwen3-Next 경로가 `--mamba-cache-mode align` 을 구현하고 alpha 플러그인이 그대로 상속한다(`all` 은 플러그인이 거부).
# align 은 chunked prefill(기본 ON) 필수. 에이전틱은 턴마다 5.7만 토큰 접두사를 다시 보내므로 prefix caching 없이는
# 처리 토큰의 99% 가 재-prefill 이었다. 세션 고정 라우팅(lb_proxy)과 함께 써야 적중한다.
#   PREFIX_CACHE=1        → --enable-prefix-caching --mamba-cache-mode align
#   MAMBA_BLOCK=<N>       → --mamba-block-size N (8 의 배수). 기본은 attention 블록(544). 블록마다 GDN 상태(18 레이어 ≈ 18 MiB)를
#                           스냅샷하므로 작을수록 재사용은 촘촘하고 KV 풀 소모는 크다 — 선택 근거는 SFT_BENCHMARKS.md §2.5.
#   MAX_BATCHED_TOKENS=N  → --max-num-batched-tokens N (기본 8192; 5.7만 프롬프트 prefill 스텝 수를 줄인다)
# 정확도 게이트: eval_sft/prefix_cache_check.py (ON/OFF 로짓 대조 + 적중 확인) 를 통과한 뒤에만 fleet 에 켠다.
PC_FLAGS=""
if [ "${PREFIX_CACHE:-0}" = "1" ]; then
  PC_FLAGS="--enable-prefix-caching --mamba-cache-mode align"
  [ -n "${MAMBA_BLOCK:-}" ] && PC_FLAGS="$PC_FLAGS --mamba-block-size $MAMBA_BLOCK"
fi
[ -n "${MAX_BATCHED_TOKENS:-}" ] && PC_FLAGS="$PC_FLAGS --max-num-batched-tokens $MAX_BATCHED_TOKENS"
echo "[serve] ckpt=$CKPT max_len=$MAX_LEN DP=$DP port=$PORT tools=${TOOLS:-0} reasoning=${REASONING_PARSER:-off} prefix_cache=${PREFIX_CACHE:-0}${MAMBA_BLOCK:+/block$MAMBA_BLOCK}${MAX_BATCHED_TOKENS:+ batched=$MAX_BATCHED_TOKENS} cache=$VLLM_CACHE_ROOT(${CACHE_AVAIL_GB}GB free)"
exec $VENV/bin/vllm serve "$CKPT" \
  $TOOL_FLAGS $REASON_FLAGS $PC_FLAGS \
  --served-model-name alpha \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --data-parallel-size "$DP" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port "$PORT"
