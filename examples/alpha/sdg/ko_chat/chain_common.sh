#!/bin/bash
# chain_common.sh — 무인 체인 러너 공용 헬퍼 (sub1 전용).
# 사용자 사전 승인(2026-08-23): r1 완료 시 게이트 통과 후 r2 무인 투입.
K=/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/sdg/ko_chat
VENV=/home/work/vidsearch/repos/project_s/syn_data/.venv/bin/python
BASE=http://127.0.0.1:8000/v1

say() { echo "[$(date '+%m-%d %H:%M:%S')] [$CHAIN_NAME] $*"; }

# 서버 헬스체크 + 자기치유: 죽어 있으면 락 잡고 재기동. 최대 ~40분 대기.
wait_server() {
  for _ in $(seq 1 70); do
    if curl -s -m 5 $BASE/models 2>/dev/null | grep -q gemma; then
      return 0
    fi
    if ! pgrep -f 'vllm serve google/gemma-4-31[B]' >/dev/null; then
      if mkdir "$K/out/.server_lock" 2>/dev/null; then
        say "서버 다운 감지 — 자동 재기동"
        nohup bash "$K/serve/serve_gemma31b.sh" \
          > "$K/serve/logs/serve_auto_$(date +%m%d_%H%M).log" 2>&1 < /dev/null &
        sleep 240
        rmdir "$K/out/.server_lock" 2>/dev/null
      fi
    fi
    sleep 30
  done
  say "서버 복구 실패 (40분 초과)"
  return 1
}
