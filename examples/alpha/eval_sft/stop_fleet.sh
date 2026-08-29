#!/bin/bash
# stop_fleet.sh — fleet(vLLM 서버들 + 프록시) 깨끗한 종료.
# GPU0 누수 교훈: hard-kill(-9) 전에 SIGTERM으로 CUDA 컨텍스트를 정상 정리시키고,
# GPU 메모리 반납을 확인한 뒤에야 다음 단계로 넘어간다.
# 사용: bash eval_sft/stop_fleet.sh [GPUS]   # GPUS 예 "1,2,3,4,5,6,7" (메모리 회수 검증 대상)
set -uo pipefail
GPUS="${1:-1,2,3,4,5,6,7}"

# 1) 프록시 먼저 (새 요청 차단)
pkill -TERM -f "eval_sft/lb_proxy.py" 2>/dev/null || true
# 2) vLLM 서버들에 SIGTERM (setsid 프로세스그룹째) — 정상 종료 유도
for pid in $(pgrep -f "alpha_serve_venv/bin/vllm"); do
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$pgid" ] && kill -TERM -- "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
done
# 3) 최대 40초 정상 종료 대기
for i in $(seq 1 20); do
    n=$(pgrep -f "alpha_serve_venv/bin/vllm" | wc -l)
    [ "$n" -eq 0 ] && break
    sleep 2
done
# 4) 남으면 SIGKILL
if [ "$(pgrep -f 'alpha_serve_venv/bin/vllm' | wc -l)" -gt 0 ]; then
    echo "[stop] SIGTERM 후 잔류 → SIGKILL"
    pkill -9 -f "alpha_serve_venv/bin/vllm" 2>/dev/null || true
    sleep 3
fi
# 5) GPU 메모리 회수 확인 (누수 조기 감지)
IFS=',' read -ra GL <<< "$GPUS"
leaked=""
for g in "${GL[@]}"; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
    [ "${m:-0}" -gt 2000 ] 2>/dev/null && leaked="$leaked $g(${m}MiB)"
done
if [ -n "$leaked" ]; then
    echo "[stop] ⚠️ GPU 메모리 미회수:$leaked — 좀비 할당 의심(리셋 필요할 수 있음)"
    exit 2
fi
echo "[stop] fleet 정상 종료, GPU 메모리 회수 확인"
