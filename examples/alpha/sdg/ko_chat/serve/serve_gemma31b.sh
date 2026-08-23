#!/bin/bash
# serve_gemma31b.sh — sub1 전용 vLLM 서버 (한국어 chat SDG 트랙)
#
# gemma-4-31B-it을 TP2 × DP4 (8 GPU 전체, 레플리카 4개)로 서빙한다.
# 단일 엔드포인트(:8000)라 DataDesigner의 provider 설정이 그대로 붙는다.
# 전례: syn_data/configs/models.yaml 주석의 TP4 단일 인스턴스 명령
# (LC 게이트 이전, 4 GPU 시절). 지금은 8 GPU 유휴라 DP4로 확장.
#
# 실행 (sub1):
#   nohup bash serve_gemma31b.sh > logs/serve_<ts>.log 2>&1 &
# 중지:
#   pkill -f 'vllm serve google/gemma-4-31[B]'   # [브래킷] = ssh 자기매치 방지

export HF_HOME=/home/work/vidsearch/.cache/huggingface
export HF_HUB_OFFLINE=1
# IB HCA는 보이지만 admin-Disabled인 클러스터 — NCCL이 잡지 않게 명시 차단
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

exec /home/work/vidsearch/repos/translate/.venv/bin/vllm serve \
  google/gemma-4-31B-it \
  --served-model-name gemma-4-31b \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000
