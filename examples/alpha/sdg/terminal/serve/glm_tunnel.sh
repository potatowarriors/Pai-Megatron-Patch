#!/bin/bash
# glm_tunnel.sh — sub1 → gpu06 alpha-eval 컨테이너 역터널 (컨테이너:8299 → sub1:8300).
#
# Harbor 의 Terminus-2 는 컨테이너 프로세스에서 돌며 교사 엔드포인트를 localhost 로 본다.
# fleet 용 8199→8100 터널(tools/swe_tunnel_keep.sh)과 같은 구조, 포트만 다르다 —
# 두 터널이 공존해야 벤치(alpha)와 합성(GLM)이 동시에 돌 수 있다.
#
# 사용 (sub1):  bash glm_tunnel.sh start | stop | status
#   컨테이너에서 확인: curl -s localhost:8299/v1/models
set -uo pipefail
SSHC=/home/work/vidsearch/.ssh-keys/config
LOG=/home/work/vidsearch/tools/glm53/tunnel.log
REMOTE_PORT="${REMOTE_PORT:-8299}"
LOCAL_PORT="${LOCAL_PORT:-8300}"
TAG="glm_tunnel_keepalive"

keep() {
  while true; do
    ssh -F "$SSHC" -o BatchMode=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes -N -R "$REMOTE_PORT:localhost:$LOCAL_PORT" alpha-eval
    echo "[glm-tunnel] down $(date +%H:%M:%S), reconnect"; sleep 4
  done
}

case "${1:-status}" in
  keep) keep ;;
  start)
    pkill -f "$TAG" 2>/dev/null; sleep 1
    mkdir -p "$(dirname "$LOG")"
    setsid bash -c "exec -a $TAG bash '$0' keep" < /dev/null > "$LOG" 2>&1 &
    echo "[glm-tunnel] started (container:$REMOTE_PORT -> sub1:$LOCAL_PORT), log $LOG" ;;
  stop) pkill -f "$TAG" && echo "[glm-tunnel] stopped" ;;
  status)
    pgrep -f "$TAG" > /dev/null && echo "[glm-tunnel] keepalive running" || echo "[glm-tunnel] not running"
    ssh -F "$SSHC" -o BatchMode=yes -o ConnectTimeout=10 alpha-eval \
      "curl -s -o /dev/null -w 'container localhost:$REMOTE_PORT -> %{http_code}\n' localhost:$REMOTE_PORT/v1/models" ;;
  *) echo "usage: $0 start|stop|status"; exit 2 ;;
esac
