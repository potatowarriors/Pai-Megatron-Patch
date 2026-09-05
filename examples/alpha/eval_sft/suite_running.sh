#!/bin/bash
# suite_running.sh — 벤치 스위트가 BENCH_HOST 에서 도는가.
#   돌면 exit 0, 안 돌면 exit 1. `-v` 면 매치한 프로세스도 찍는다.
#
# ⚠️ **왜 이 판정을 한 곳에 모았나 (2026-09-05)**
#
# `pgrep -f <패턴>` 은 명령줄 **문자열**을 본다. 이 방식은 두 번 배신한다.
#
# ① 자기 자신 매치. ssh 로 보내면 원격에 `bash -c "pgrep -f 'run_suite.sh'"` 가
#    뜨고 그 명령줄에 패턴이 들어 있어 **아무것도 안 돌아도 참**이 된다.
#      $ ssh sub1 "pgrep -af 'run_suite.sh'"
#      3733803 bash -c pgrep -af 'run_suite.sh'      ← 자기 자신
#    이것 때문에 iter1200 종료를 기다리던 체인이 **24시간 헛돌았다**(09-04 08:29
#    종료 → 09-05 08:52 까지 iter1500 미시작).
#
# ② 남의 명령줄 매치. 브래킷(`[r]un_suite.sh`)으로 ①을 막아도, 스크립트 이름을
#    **언급만 한** 다른 셸(로그 grep, 다른 세션의 편집 명령)이 그대로 걸린다.
#    실제로 main1 에서 브래킷 형태가 Claude 세션 셸 2개를 잡았다.
#
# 그래서 문자열이 아니라 **argv 구조**로 본다: argv0 이 셸이고 argv1 이 우리
# 스크립트 파일일 때만 "실행 중"이다. `bash -c "... run_suite.sh ..."` 는
# argv1 이 `-c` 라 걸리지 않는다.
#
# 새 감시 스크립트는 직접 pgrep 하지 말고 **이 스크립트를 호출**한다.
#   while bash eval_sft/suite_running.sh; do sleep 300; done   # 종료 대기
set -uo pipefail
BENCH_HOST="${BENCH_HOST:-sub1}"

read -r -d '' PROBE <<'AWK'
ps -eo args --no-headers | awk '
  # 셸이 우리 스크립트를 직접 실행 중인가 (argv1 이 파일 경로)
  $1 ~ /(^|\/)(ba)?sh$/ && $2 ~ /(run_suite|eval_new_ckpt|run_tier1|run_tier2|run_swe|run_terminal|run_convert)\.sh$/ { print; next }
  # 파이썬 러너 · lm_eval (argv0 이 python 일 때만)
  $1 ~ /(^|\/)python3?$/ && $0 ~ /(run_simpleqa\.py|run_logickor\.py|lm_eval)/ { print; next }
'
AWK

if [ "$(hostname)" = "$BENCH_HOST" ]; then
  out=$(bash -c "$PROBE")
else
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$BENCH_HOST" "$PROBE" 2>/dev/null)
fi

[ "${1:-}" = "-v" ] && [ -n "$out" ] && echo "$out" | cut -c1-110
[ -n "$out" ]
