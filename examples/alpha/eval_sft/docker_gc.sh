#!/bin/bash
# docker_gc.sh — 에이전틱 실행 후 컨테이너 호스트 디스크 회수.
#
# 무엇을 지우고 무엇을 남기는가 (2026-08-31 실측 근거):
#   지움: build cache, 미사용 볼륨, 정지 컨테이너, dangling 이미지
#   남김: **sweb.eval 태스크 이미지** (498개 = 479.7GB). SWE-bench Verified 인스턴스별
#         전용 이미지로 다음 체크포인트에서 그대로 재사용된다. 지우면 500 × 4.77GB 를
#         Docker Hub 에서 다시 받아야 하고 그 시간이 디스크보다 비싸다.
#
# 누수는 없다 — 실측에서 정지 컨테이너 0개였다(mini-swe-agent 가 정리한다).
# 늘어나는 것은 이미지 캐시와 build cache 뿐이다.
#
# 사용: bash eval_sft/docker_gc.sh [--images]   # --images 는 sweb.eval 까지 삭제(주의)
set -uo pipefail
SSHC=/home/work/vidsearch/.ssh-keys/config
CONTAINER=alpha-eval
PURGE_IMAGES=0
[ "${1:-}" = "--images" ] && PURGE_IMAGES=1

before=$(ssh -F "$SSHC" -o BatchMode=yes "$CONTAINER" "df -BG --output=avail /var/lib/docker | tail -1 | tr -d 'G '" 2>/dev/null)
echo "[gc] 시작 — 여유 ${before}GB"

ssh -F "$SSHC" -o BatchMode=yes "$CONTAINER" '
  docker container prune -f >/dev/null 2>&1
  docker builder prune -f 2>&1 | tail -1
  docker volume prune -f  2>&1 | tail -1
  docker image prune -f   2>&1 | tail -1
' 2>/dev/null

if [ "$PURGE_IMAGES" = "1" ]; then
  echo "[gc] ⚠️ sweb.eval 이미지까지 삭제 (다음 SWE 실행에서 재다운로드 필요)"
  ssh -F "$SSHC" -o BatchMode=yes "$CONTAINER" \
    'docker images --format "{{.Repository}}:{{.Tag}}" | grep "^sweb.eval" | xargs -r docker rmi -f >/dev/null 2>&1; echo done' 2>/dev/null
fi

after=$(ssh -F "$SSHC" -o BatchMode=yes "$CONTAINER" "df -BG --output=avail /var/lib/docker | tail -1 | tr -d 'G '" 2>/dev/null)
echo "[gc] 완료 — 여유 ${before}GB → ${after}GB (회수 $((after - before))GB)"
