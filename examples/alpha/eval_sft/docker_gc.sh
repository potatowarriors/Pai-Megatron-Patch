#!/bin/bash
# docker_gc.sh — 에이전틱 실행 후 컨테이너 호스트 디스크 회수.
#
# 무엇을 지우고 무엇을 남기는가 (2026-08-31 실측 근거):
#   지움: build cache, 미사용 볼륨, 정지 컨테이너, dangling 이미지
#   지움(기본): **sweb.eval 인스턴스 이미지** — SWE-bench 5.0.2 는 Docker Hub 에서 인스턴스별
#         이미지를 받아 쓰고 **정리하지 않는다**(리포트의 `unremoved_images: 500` 이 그 증거).
#         구 하니스의 `--cache_level {none,base,env,instance}` 는 5.0.2 에 없다 — 보존 정책을
#         쓸 수 없으니 우리가 지운다. 이미지는 재취득 가능하고, 인스턴스 이미지는 3층
#         (base → environment → instance) 중 맨 위 얇은 층이다(실측: 10개 레이어 중 9개가
#         같은 repo 인스턴스끼리 공유, 고유는 1개).
#         **정정(2026-09-08)**: 이 자리에 "지우면 500×4.77GB 를 다시 받아야 하니 디스크보다
#         비싸다" 고 적어 두었던 것은 과장이었다. 공유 레이어를 감안하지 않은 계산이었고,
#         그 근거로 500개를 무기한 쌓아 A3 임계를 반복해 밑돌았다. `docker images` 합산(509GB)은
#         레이어를 이미지마다 중복 집계한 값이고 실제 점유는 ≈430GB 였다(2026-09-08
#         `du /var/lib/docker/containerd`). 정책은 사용자 결정(09-08)대로 **실행 후 정리**다.
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
# 사용: bash eval_sft/docker_gc.sh [--keep-images] [--dry-run] [--rotate-only]
#   --keep-images  sweb.eval 인스턴스 이미지를 남긴다 (재취득 시간을 아껴야 할 때만)
#   --dry-run  지울 목록만 출력
#   --rotate-only  산출물 회전만 (docker prune 생략 — 에이전틱 실행 중에 안전)
# 환경변수: GC_KEEP_RUNS(기본 2) · GC_SAFE_MINUTES(기본 120, 최근 수정분은 진행 중으로 보고 보존)
set -uo pipefail
SSHC=/home/work/vidsearch/.ssh-keys/config
CONTAINER=alpha-eval
PURGE_IMAGES=1   # 기본값 = 표준(정리한다). --keep-images 로 끈다.
DRY=0
ROTATE_ONLY=0
# 모르는 인자는 **아무것도 지우기 전에** 거부한다. 2026-09-14 `--help` 를 넘겼더니 무시되고 정리가 그대로 실행돼
# 빌드 캐시 대부분이 지워졌다(정지 컨테이너 prune 포함, 이미지·볼륨·산출물은 도달 전 중단). 파괴적 스크립트는
# 오타(`--dryrun` 등)에도 실행되면 안 된다.
usage() { sed -n '/^# 사용:/,/^# 환경변수:/p' "$0" | sed 's/^# \{0,1\}//'; }
for a in "$@"; do
  case "$a" in
    --keep-images) PURGE_IMAGES=0 ;;
    --images)      PURGE_IMAGES=1 ;;   # 구 플래그 (하위호환, 기본이 이미 1)
    --dry-run)     DRY=1 ;;
    # 실행 중에는 docker prune 을 건너뛰고 산출물 회전만 한다 — builder prune 이
    # 진행 중인 이미지 빌드의 캐시를 건드릴 수 있다.
    --rotate-only) ROTATE_ONLY=1 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "[gc] ❌ 모르는 인자: $a — 아무것도 하지 않고 종료" >&2; usage >&2; exit 2 ;;
  esac
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

# 컨테이너에 keep 마커(/opt/swebench/.keep_images_*)가 있으면 sweb.eval 이미지를 보존한다 — Docker Hub 미인증 IP 는
# **시간당 100 pull** 제한이라 500 이미지 재취득에 5시간이 든다(2026-09-25 iter2300 SWE 500/500 환경 실패, KNOWN_ISSUES).
# pre-pull 을 마친 이미지를 SWE 단계 직전 gc 가 지우면 안 된다. 마커는 SWE 가 끝난 뒤 사람이 지운다.
if [ "$PURGE_IMAGES" = "1" ] && ssh -F "$SSHC" -o BatchMode=yes "$CONTAINER" 'ls /opt/swebench/.keep_images_* >/dev/null 2>&1' 2>/dev/null; then
  echo "[gc] keep 마커 있음(/opt/swebench/.keep_images_*) — sweb.eval 이미지 보존"; PURGE_IMAGES=0
fi
if [ "$PURGE_IMAGES" = "1" ]; then
  echo "[gc] sweb.eval 인스턴스 이미지 정리 (다음 SWE 실행에서 재취득)"
  ssh -F "$SSHC" -o BatchMode=yes "$CONTAINER" \
    'docker images --format "{{.Repository}}:{{.Tag}}" | grep "^swebench/sweb.eval" | xargs -r docker rmi -f >/dev/null 2>&1; echo done' 2>/dev/null
fi

after=$(ssh -F "$SSHC" -o BatchMode=yes "$CONTAINER" "df -BG --output=avail /var/lib/docker | tail -1 | tr -d 'G '" 2>/dev/null)
echo "[gc] 완료 — 여유 ${before}GB → ${after}GB (회수 $((after - before))GB)"
