#!/bin/bash
# progress.sh — 벤치 진행 상황 한눈에.
#
# 왜 필요한가: 에이전틱 러너는 ssh 출력을 `tail -N` 으로 파이프하므로 **실행 중에는
# 로그에 아무것도 안 나온다**(tail 은 스트림 종료 후 출력). 로그 크기로 정체를 판단하면
# 오판한다 — 진행은 산출물 개수로 세야 한다.
#
# 사용: bash eval_sft/progress.sh [RUN_TAG]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SSHC=/home/work/vidsearch/.ssh-keys/config
TAG="${1:-}"

# 프로세스·포트는 **스위트가 도는 노드**에서만 보인다(로그는 NFS 라 어디서나). 다른 노드에서 돌리면
# "프로세스 없음 / 백엔드 0/8" 로 읽힌다 — 2026-09-01 에 정상 실행 중인 iter900 스위트를 중단된 줄 알았다.
# 벤치 노드는 시기마다 다르다(09월 sub1 → 10-01 부터 main1, 이름은 역할일 뿐 — CLAUDE.md 환경 불변량).
# 기본은 이 스크립트를 돌리는 노드. 다른 노드를 보려면 BENCH_HOST=<노드>.
BENCH_HOST="${BENCH_HOST:-$(hostname)}"
if [ "$(hostname)" = "$BENCH_HOST" ]; then
  ON() { bash -c "$1"; }
else
  ON() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$BENCH_HOST" "$1" 2>/dev/null; }
fi

echo "── 프로세스 ($BENCH_HOST) ───────────────────"
# 판정은 suite_running.sh 한 곳에만 둔다 — pgrep -f 문자열 매칭은 자기 자신과
# "이름만 언급한" 남의 셸을 모두 잡는다 (그 서사는 suite_running.sh 주석).
bash "$HERE/suite_running.sh" -v | sed 's/^/  /' || echo "  (없음)"

echo "── fleet ($BENCH_HOST) ──────────────────────"
ON 'c=0; for p in 8000 8001 8002 8003 8004 8005 8006 8007; do
      [ "$(curl -s -o /dev/null -w %{http_code} --max-time 2 http://localhost:$p/v1/models)" = "200" ] && c=$((c+1))
    done
    echo "  백엔드 $c/8   프록시 $(curl -s -o /dev/null -w %{http_code} --max-time 2 http://localhost:8100/v1/models)"'

echo "── lm_eval (T1/T2) ──────────────────────────"
for L in /home/work/vidsearch/tools/bench_logs/t1_*.log /home/work/vidsearch/tools/bench_logs/t2_*.log \
         /home/work/vidsearch/tools/bench_logs/iter*_full.log /home/work/vidsearch/tools/bench_logs/*_suite.log /home/work/vidsearch/tools/bench_logs/ruler_v3_*.log; do
  [ -f "$L" ] || continue
  # tqdm 진행바는 두 개의 '|' 사이에 블록 문자가 들어간다 — 단순 정규식은 걸린다.
  # 카운터("13111/18904 [2:52:07<2:55:58")만 뽑는다.
  p=$(tr '\r' '\n' < "$L" 2>/dev/null | grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:?]+" | tail -1)
  [ -n "$p" ] && echo "  $(basename "$L"): $p"
done

echo "── 에이전틱 (컨테이너 산출물로 계수) ────────"
if [ -n "$TAG" ]; then
  # Terminal 실행 디렉토리는 태그에서 결정된다 (run_terminal.sh 와 같은 산식).
  # `ls -dt | head -1` 로 최신 디렉토리를 잡으면 **다른 체크포인트의 중단된 실행**을
  # 현재 진행률로 보고한다 — 2026-09-01 에 iter900 을 물었는데 iter300 의 42/80
  # (중단분)이 나왔다. 태그로 지목해야 한다.
  # echo 의 개행까지 해싱한다 — 러너가 `echo "$RUN_NAME" | md5sum` 이므로
  # printf '%s' 를 쓰면 해시가 달라져 없는 디렉토리를 가리킨다.
  H8="$(echo "$TAG" | md5sum | cut -c1-8)"
  # TB-1(run_terminal.sh, /opt/terminalbench/runs/tb<h8>) 은 09-07 에 TB-2 로 대체됐다. TB-2 는 harbor job
  # /opt/harbor/jobs/tb2<h8>(2.0) · tb21<h8>(2.1, 2026-10-01~) — 트라이얼 = 89 과제 × 8 = 712, 끝난 것은 result.json 으로 센다.
  # 2026-10-01 까지 이 절이 TB-1 경로만 봐서 돌고 있는 TB-2 를 "미시작"으로 찍었다.
  ssh -F "$SSHC" -o BatchMode=yes -o ConnectTimeout=10 alpha-eval "
    D=/opt/swebench/preds_$TAG
    [ -d \$D ] && echo \"  SWE: \$(ls -d \$D/*/ 2>/dev/null | wc -l)/500  (로그 \$(stat -c %y \$D/minisweagent.log 2>/dev/null | cut -c12-19))\"
    found=0
    for J in /opt/harbor/jobs/tb21$H8 /opt/harbor/jobs/tb2$H8; do
      [ -d \"\$J\" ] || continue; found=1
      case \${J##*/} in tb21*) V=2.1;; *) V=2.0;; esac
      echo \"  Terminal-Bench \$V: 완료 \$(find \$J -mindepth 2 -maxdepth 2 -name result.json | wc -l)/712 · 시작된 트라이얼 \$(ls -d \$J/*/ 2>/dev/null | wc -l)  (\${J##*/})\"
    done
    [ \$found = 1 ] || echo \"  Terminal-Bench: 미시작 (tb21$H8 / tb2$H8)\"
    R=/opt/terminalbench/runs/tb$H8
    [ -d \"\$R\" ] && echo \"  Terminal(TB-1, 구): \$(ls -d \$R/*/ 2>/dev/null | wc -l)/80  (tb$H8)\"
    echo \"  실행 중 컨테이너: \$(docker ps -q | wc -l)\"" 2>/dev/null || echo "  (컨테이너 접속 실패)"
else
  echo "  RUN_TAG 를 인자로 주면 계수한다"
fi

echo "── 결과 ─────────────────────────────────────"
ls -1 "$HERE/results" 2>/dev/null | grep -v TRACKING | sed 's/^/  /' || echo "  (없음)"
