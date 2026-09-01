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

# 스위트와 fleet 는 **sub1** 에서 돈다. main1 에서 이 스크립트를 돌리면 로그(NFS 공유)는
# 보이지만 프로세스·포트는 안 보인다 — 2026-09-01 에 이것 때문에 정상 실행 중인 iter900
# 스위트를 "프로세스 없음 / 백엔드 0/8" 로 읽어 중단된 줄 알았다. 산출물로 세라고 만든
# 스크립트가 정작 자기 진단은 로컬 가정으로 했다.
BENCH_HOST="${BENCH_HOST:-sub1}"
if [ "$(hostname)" = "$BENCH_HOST" ]; then
  ON() { bash -c "$1"; }
else
  ON() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$BENCH_HOST" "$1" 2>/dev/null; }
fi

echo "── 프로세스 ($BENCH_HOST) ───────────────────"
# run_suite.sh 는 오케스트레이터다 — 이게 빠져 있으면 스위트 전체가 안 보인다.
procs=$(ON "pgrep -af 'run_suite.sh|run_tier1.sh|run_tier2.sh|run_swe.sh|run_terminal.sh|run_simpleqa|run_logickor|chain_iter600|run_convert|lm_eval' | grep -v pgrep | cut -c1-110")
[ -n "$procs" ] && echo "$procs" | sed 's/^/  /' || echo "  (없음)"

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
  ssh -F "$SSHC" -o BatchMode=yes -o ConnectTimeout=10 alpha-eval "
    D=/opt/swebench/preds_$TAG
    [ -d \$D ] && echo \"  SWE: \$(ls -d \$D/*/ 2>/dev/null | wc -l)/500  (로그 \$(stat -c %y \$D/minisweagent.log 2>/dev/null | cut -c12-19))\"
    R=\$(ls -dt /opt/terminalbench/runs/tb* 2>/dev/null | head -1)
    [ -n \"\$R\" ] && echo \"  Terminal: \$(ls -d \$R/*/ 2>/dev/null | wc -l)/80  (\$(basename \$R))\"
    echo \"  실행 중 컨테이너: \$(docker ps -q | wc -l)\"" 2>/dev/null || echo "  (컨테이너 접속 실패)"
else
  echo "  RUN_TAG 를 인자로 주면 계수한다"
fi

echo "── 결과 ─────────────────────────────────────"
ls -1 "$HERE/results" 2>/dev/null | grep -v TRACKING | sed 's/^/  /' || echo "  (없음)"
