#!/bin/bash
# serve_glm53.sh — sub1 전용 GLM-5.3-Flash 교사 서빙 (터미널 에이전트 SDG 트랙)
#
# 왜 venv 인가: sub1(Backend.AI)은 docker·apptainer 가 없어 사용자가 준
#   `docker run vllm/vllm-openai:glm53-flash …` 를 그대로 못 쓴다. 플래그만 승계하고
#   vLLM nightly cu130 휠(0.28.1rc1.dev485, torch 2.13.0+cu130)을 venv 에 설치했다
#   (정식 릴리스 0.28.0 은 GLM-5.3-Flash 미지원, 0.29 요구). 설치 기록: README §P0.
#
# 승계한 플래그 (사용자 제공 docker 명령, 2026-09-07):
#   --enable-expert-parallel --data-parallel-size 8 --no-enable-flashinfer-autotune
#   --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45
# 추가한 플래그:
#   --max-model-len 131072   학습 시퀀스 128k 와 정합 (초과 트라젝토리는 어차피 드롭)
#   --limit-mm-per-prompt 0  텍스트 전용 — 멀티모달 래퍼(Glm5NextForConditionalGeneration)의
#                            비전 인코더를 띄우지 않는다
#   KV 는 BF16 — Hopper 는 이 모델의 FP8 KV 미지원 (vLLM 레시피)
#
# MODE=dp8 (기본, 레시피) | tp8 (DP 크래시 폴백 — 이 환경에서 vLLM DP munmap 크래시 전례,
#   eval_sft/serve_fleet.sh 주석)
#
# 실행 (sub1):
#   nohup bash serve_glm53.sh > /home/work/vidsearch/tools/glm53/serve_<ts>.log 2>&1 &
# 중지:
#   pkill -f 'vllm serve .*GLM-5.3-Flas[h]'   # [브래킷] = ssh 자기매치 방지
set -uo pipefail
MODE="${MODE:-dp8}"
PORT="${PORT:-8300}"
MAX_LEN="${MAX_LEN:-131072}"
# 프로파일링 OOM 대응 (2026-09-07 실측): 기본 8192 청크에서 EP+DP8 은 랭크 8개 토큰이 한 랭크의
# 전문가로 모여(8×8192) FlashInfer CUTLASS FP8 MoE 작업공간이 잔여 28 GiB 를 넘겼다. 청크 4096·
# 랭크당 동시 시퀀스 64(8 랭크 = 512)로 제한 — 트라젝토리 수집은 디코드 위주라 처리량 영향 작음.
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
MAX_SEQS="${MAX_SEQS:-64}"
MODEL="${MODEL:-/home/work/vidsearch/models/GLM-5.3-Flash}"
VENV=/home/work/vidsearch/tools/glm_serve_venv

# NGC 전역 PIP_CONSTRAINT 가 venv 를 오염시키지 않게 차단 (serve_alpha.sh 전례)
export PIP_CONSTRAINT=
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# IB HCA 는 보이지만 admin-Disabled 인 클러스터 — NCCL 이 잡지 않게 명시 차단
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
# 306 GiB 를 NFS 에서 읽으므로 엔진 준비 대기를 길게 (사용자 docker 명령 승계)
export VLLM_ENGINE_READY_TIMEOUT_S=3600
# sub1 compat 불일치 우회 (2026-09-07 실측, README §P0 / KNOWN_ISSUES):
#   /usr/local/cuda/compat/lib.real 은 libcuda.so.1 만 595 이고 libnvidia-ptxjitcompiler.so.1·
#   libnvidia-nvvm.so.4 는 570 을 가리킨다(08-29 스왑이 절반만 적용). NCCL 2.29 커널 로드가
#   PTX JIT 를 타면 570 JIT 가 595 libcuda 힙을 깨뜨려 전 rank munmap_chunk SIGABRT.
#   595 JIT·NVVM 심볼릭 링크만 담은 사용자 디렉토리를 LD_LIBRARY_PATH 앞에 둔다 — libcuda 가
#   soname 으로 dlopen 하므로 이것만으로 595 가 잡힌다 (sudo 불필요, 프로세스 범위).
JIT595=/home/work/vidsearch/tools/cuda_compat13/jit595
[ -e "$JIT595/libnvidia-ptxjitcompiler.so.1" ] || { echo "[serve] ❌ $JIT595 없음 — README §P0 의 링크 생성 절차 참조"; exit 3; }
export LD_LIBRARY_PATH="$JIT595:${LD_LIBRARY_PATH:-}"

case "$MODE" in
  dp8) PAR="--enable-expert-parallel --data-parallel-size 8" ;;
  tp8) PAR="--tensor-parallel-size 8" ;;
  *) echo "[serve] unknown MODE=$MODE (dp8|tp8)"; exit 2 ;;
esac
echo "[serve] model=$MODEL mode=$MODE max_len=$MAX_LEN batched=$MAX_BATCHED_TOKENS seqs=$MAX_SEQS port=$PORT"
exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name glm53-flash \
  $PAR \
  --max-model-len "$MAX_LEN" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_SEQS" \
  --gpu-memory-utilization 0.90 \
  --no-enable-flashinfer-autotune \
  --tool-call-parser glm47 --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --host 0.0.0.0 --port "$PORT"
