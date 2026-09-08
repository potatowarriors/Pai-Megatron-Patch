#!/bin/bash
# precheck_setup.sh — 시나리오 과제의 setup.sh 를 **자원 상한이 걸린 샌드박스**에서 미리 실행해 폭주를 걸러낸다.
#
# 왜: 2026-09-08 교사가 만든 과제 하나가 setup.sh 에서 `fallocate -l $(df 기반) /app/vault.bin` 으로 2.36 TB 를 써서
#     gpu06 디스크를 가득 채웠다 (KNOWN_ISSUES 09-08). 이미지 빌드(RUN setup.sh)에는 디스크 상한이 없으므로
#     빌드 전에 tmpfs 512 MB + fsize 200 MB + 120 초 상한으로 한 번 돌려 보고, 실패하면 격리한다.
#
# 사용: bash precheck_setup.sh <task_root> [P=8]
#   gpu06 alpha-eval 컨테이너에서 실행된다 (task_root 는 tar 로 전송). 결과: <task_root>/PRECHECK_FAIL.txt,
#   실패 과제는 <task_root>_quarantine/ 로 이동. setup.sh 가 없는 과제는 검사 없이 통과.
set -uo pipefail
ROOT="${1:?task_root}"; P="${2:-8}"
SSHC=/home/work/vidsearch/.ssh-keys/config
TAG="pre-$(basename "$ROOT")-$$"
REMOTE=/opt/harbor/precheck/$TAG
Q="${ROOT}_quarantine"; mkdir -p "$Q"
LIST=$(for d in "$ROOT"/*/; do [ -f "$d/environment/app/setup.sh" ] && basename "$d"; done)
N=$(echo "$LIST" | grep -c . || true)
echo "[precheck] $N tasks with setup.sh → sandbox on alpha-eval (P=$P)"
[ "$N" -eq 0 ] && { : > "$ROOT/PRECHECK_FAIL.txt"; exit 0; }
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "mkdir -p $REMOTE"
echo "$LIST" | tar -C "$ROOT" -czf - -T - | ssh -F "$SSHC" -o BatchMode=yes alpha-eval "tar -C $REMOTE -xzf -"
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "
  cd $REMOTE
  run_one() {
    t=\$1
    # tmpfs /app 512MB: 넘치면 ENOSPC 로 실패. fsize 200MB: 단일 파일 상한. 120초. 네트워크 없음. root 지만 호스트와 격리.
    out=\$(timeout 150 docker run --rm --network none --tmpfs /app:size=512m,exec --ulimit fsize=209715200 \
        --memory 2g --cpus 1 -v $REMOTE/\$t/environment/app:/src:ro alpha-terminal-base:1 \
        bash -c 'cp -r /src/. /app/ && cd /app && timeout 120 bash ./setup.sh >/dev/null 2>&1; rc=\$?; du -sm /app | cut -f1; exit \$rc' 2>/dev/null)
    rc=\$?; size=\$(echo \"\$out\" | tail -1)
    if [ \$rc -ne 0 ] || [ \"\${size:-999}\" -gt 300 ]; then echo \"\$t FAIL rc=\$rc size_mb=\${size:-?}\"; else echo \"\$t OK size_mb=\$size\"; fi
  }
  export -f run_one; export REMOTE=$REMOTE
  ls | xargs -P $P -I{} bash -c 'run_one {}'
  rm -rf $REMOTE
" > "$ROOT/.precheck.txt"
grep " FAIL " "$ROOT/.precheck.txt" | awk '{print $1, $3, $4}' > "$ROOT/PRECHECK_FAIL.txt"
while read -r t _; do [ -n "$t" ] && [ -d "$ROOT/$t" ] && mv "$ROOT/$t" "$Q/"; done < "$ROOT/PRECHECK_FAIL.txt"
echo "[precheck] fail=$(wc -l < "$ROOT/PRECHECK_FAIL.txt") ok=$(grep -c " OK " "$ROOT/.precheck.txt") → 격리 $Q"
