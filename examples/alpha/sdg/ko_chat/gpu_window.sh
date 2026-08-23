#!/bin/bash
# gpu_window.sh — sub1 GPU 창구 (SFT 스모크 등 타 작업에 8×GPU 임시 양도).
# 합성 트랙을 유실 없이 동결하는 절차 (2026-08-23, f1 세션 스모크 조율용):
#   open : 체인·드라이버 SIGSTOP(레코드 유실 0) → vLLM 종료 → GPU 메모리 드레인 확인
#   close: vLLM 재기동·ready 대기 → SIGCONT (동결 중 in-flight 요청은 재시도가 회수)
# close 는 반드시 서버 ready 이후 CONT — 순서 뒤집으면 재시도 예산을 죽은 서버에 소진.
set -u
K=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/sdg/ko_chat
BASE=http://127.0.0.1:8000/v1

# ssh 자기매치 방지용 [브래킷] 패턴
PATTERNS=('chain_[ab]_r2.sh' 'translate_rege[n].py' 'ko_chat_sd[g].py')

sig_all() {  # $1 = -STOP | -CONT
  for p in "${PATTERNS[@]}"; do
    pkill "$1" -f "$p" 2>/dev/null && echo "  $1 $p"
  done
  return 0
}

case "${1:?usage: gpu_window.sh open|close}" in
  open)
    echo "[$(date '+%H:%M:%S')] 창구 OPEN 시작 — 합성 동결"
    sig_all -STOP
    pkill -f 'vllm serve google/gemma-4-31[B]' 2>/dev/null
    for _ in $(seq 1 60); do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
      [ "$used" -lt 2000 ] && break
      sleep 5
    done
    echo "[$(date '+%H:%M:%S')] WINDOW OPEN — GPU max used ${used}MiB. 스모크 진행하세요."
    ;;
  close)
    echo "[$(date '+%H:%M:%S')] 창구 CLOSE 시작 — 서버 재기동"
    nohup bash "$K/serve/serve_gemma31b.sh" \
      > "$K/serve/logs/serve_window_$(date +%m%d_%H%M).log" 2>&1 < /dev/null &
    for _ in $(seq 1 90); do
      curl -s -m 5 $BASE/models 2>/dev/null | grep -q gemma && break
      sleep 10
    done
    if ! curl -s -m 5 $BASE/models 2>/dev/null | grep -q gemma; then
      echo "[$(date '+%H:%M:%S')] 서버 재기동 실패 — 드라이버 동결 유지, 수동 확인 필요"
      exit 1
    fi
    sig_all -CONT
    echo "[$(date '+%H:%M:%S')] WINDOW CLOSED — 합성 재개 완료"
    ;;
  *)
    echo "usage: gpu_window.sh open|close"; exit 2 ;;
esac
