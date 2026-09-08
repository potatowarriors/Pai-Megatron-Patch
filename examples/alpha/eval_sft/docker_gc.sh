#!/bin/bash
# docker_gc.sh — 에이전틱 실행 후 컨테이너 호스트 디스크 회수.
#
# 무엇을 지우고 무엇을 남기는가 (2026-08-31 실측 근거):
#   지움: build cache, 미사용 볼륨, 정지 컨테이너, dangling 이미지
#   남김: **sweb.eval 태스크 이미지** (500개 = **2,386GB** — 2026-09-08 `docker images` 합산 실측; 08-31 의 479.7GB 는
#         공유 레이어 가정의 과소 집계였다). SWE-bench Verified 인스턴스별
#         전용 이미지로 다음 체크포인트에서 그대로 재사용된다. 지우면 500 × 4.77GB 를
#         Docker Hub 에서 다시 받아야 하고 그 시간이 디스크보다 비싸다.
#
# **누수는 있었다 (2026-09-07 정정).** 위 문단은 docker 객체만 보고 쓴 것이다. 실제로 쌓이는
# 것은 컨테이너 안 `/opt` 의 **에이전트 산출물**이다 — SWE 예측·궤적, Terminal 세션 기록.
# docker 명령에 보이지 않아 이 스크립트가 매번 30GB 를 회수하는 동안 조용히 늘었다:
#   /opt 합계  09-02 42GB → 09-07 108GB (5일 +66GB), 여유 592GB(08-31) → 290GB(09-07)
# 회수 로그만 보면 관리되는 것처럼 보이는 것이 이 누수가 닷새간 안 보인 이유다.
#
# 그래서 산출물 **회전**을 추가한다: 체크포인트 실행별 디렉토리를 최근 N개만 남긴다.
# 점수·리포트는 리포지토리 `results/<tag>/` 에 복사돼 있으므로 궤적을 지워도 수치는 남는다.
# 회전 대상은 실행별 디렉토리뿐이다 — `oracle_healthcheck` 같은 기준선은 건드리지 않는다.
#
# 사용: bash eval_sft/docker_gc.sh [--images] [--dry-run]
#   --images   sweb.eval 태스크 이미지까지 삭제 (주의: 500개 재다운로드)
#   --dry-run  지울 목록만 출력
#   --rotate-only  산출물 회전만 (docker prune 생략 — 에이전틱 실행 중에 안전)
# 환경변수: GC_KEEP_RUNS(기본 2) · GC_SAFE_MINUTES(기본 120, 최근 수정분은 진행 중으로 보고 보존)
set -uo pipefail
SSHC=/home/work/vidsearch/.ssh-keys/config
CONTAINER=alpha-eval
PURGE_IMAGES=0
DRY=0
ROTATE_ONLY=0
for a in "$@"; do
  [ "$a" = "--images" ] && PURGE_IMAGES=1
  [ "$a" = "--dry-run" ] && DRY=1
  # 실행 중에는 docker prune 을 건너뛰고 산출물 회전만 한다 — builder prune 이
  # 진행 중인 이미지 빌드의 캐시를 건드릴 수 있다.
  [ "$a" = "--rotate-only" ] && ROTATE_ONLY=1
done
KEEP="${GC_KEEP_RUNS:-2}"
SAFE="${GC_SAFE_MINUTES:-120}"

before=$(ssh -F "$SSHC" -o BatchMode=yes "$CONTAINER" "df -BG --output=avail /var/lib/docker | tail -1 | tr -d 'G '" 2>/dev/null)
echo "[gc] 시작 — 여유 ${before}GB"

# ── 에이전트 산출물 회전 (docker 가 못 보는 부분) ───────────────────
# 실행별 디렉토리만 대상: SWE 는 preds_*_iter<숫자>, Terminal 은 tb<해시>.
# mtime 내림차순으로 최근 KEEP 개를 남기고, SAFE 분 내 수정된 것은 진행 중으로 보고 건너뛴다.
echo "[gc] 산출물 회전 (최근 ${KEEP}개 유지, 최근 ${SAFE}분 수정분 보존)"
# 원격 스크립트는 **stdin 으로 넘긴다**. ssh "..." 안에 넣으면 따옴표가 3중으로 중첩돼
# $d 가 로컬에서 먼저 먹힌다 (2026-09-07 에 실제로 그래서 빈 경로를 지울 뻔했다).
ssh -F "$SSHC" -o BatchMode=yes "$CONTAINER" "KEEP=$KEEP SAFE=$SAFE DRY=$DRY bash -s" <<'REMOTE' 2>/dev/null
rotate() {
  ls -dt $1 2>/dev/null | tail -n +$((KEEP+1)) | while read -r d; do
    [ -d "$d" ] || continue
    if [ -n "$(find "$d" -maxdepth 0 -mmin -$SAFE 2>/dev/null)" ]; then
      echo "  보존(진행중) $(basename "$d")"; continue
    fi
    sz=$(du -sBM "$d" 2>/dev/null | cut -f1 | tr -d M)
    if [ "$DRY" = 1 ]; then echo "  [dry] ${sz}MB $(basename "$d")"
    else rm -rf "$d" && echo "  회수 ${sz}MB $(basename "$d")"; fi
  done
}
rotate '/opt/swebench/preds_*_iter[0-9]*'
rotate '/opt/terminalbench/runs/tb[0-9a-f]*'
REMOTE

[ "$DRY" = 1 ] && { echo "[gc] --dry-run — docker prune 은 건너뜀"; exit 0; }
if [ "$ROTATE_ONLY" = 1 ]; then
  after=$(ssh -F "$SSHC" -o BatchMode=yes "$CONTAINER" "df -BG --output=avail /var/lib/docker | tail -1 | tr -d 'G '" 2>/dev/null)
  echo "[gc] --rotate-only 완료 — 여유 ${before}GB → ${after}GB (회수 $((after - before))GB)"
  exit 0
fi

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
