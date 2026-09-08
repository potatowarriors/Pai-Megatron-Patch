#!/bin/bash
# validate_tasks.sh — 합성 과제 배치를 gpu06 Harbor 로 검증: oracle(solve.sh) → reward 1, nop → reward 0 인 과제만 채택.
#
# 사용: bash validate_tasks.sh <task_root> <tag> [W=16] [--oracle-only]
#   task_root: 과제 디렉토리들이 든 로컬(NFS) 디렉토리. 컨테이너 /opt/harbor/synth/<tag>/ 로 복사돼 실행된다.
# 산출: <task_root>/VALID.txt (채택 과제 이름), <task_root>/INVALID.txt (사유), 요약 표준출력.
#   채택 과제는 <task_root>/../<tag>_valid/ 에 심볼릭 링크가 아닌 복사본으로 모인다 (Harbor -p 가 읽을 디렉토리).
set -uo pipefail
ROOT="${1:?task_root}"; TAG="${2:?tag}"; W="${3:-16}"; MODE="${4:-}"
SSHC=/home/work/vidsearch/.ssh-keys/config
REMOTE=/opt/harbor/synth/$TAG
N=$(ls -d "$ROOT"/*/ 2>/dev/null | wc -l)
echo "[validate] $N tasks → alpha-eval:$REMOTE (W=$W)"
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "rm -rf $REMOTE /opt/harbor/jobs/val-$TAG-oracle /opt/harbor/jobs/val-$TAG-nop; mkdir -p /opt/harbor/synth"
tar -C "$ROOT" --exclude='GEN_MANIFEST.jsonl' --exclude='*.txt' -czf - . | ssh -F "$SSHC" -o BatchMode=yes alpha-eval "mkdir -p $REMOTE && tar -C $REMOTE -xzf -"

run_agent() {  # $1 = agent name
  ssh -F "$SSHC" -o BatchMode=yes alpha-eval "
    export HOME=/opt/harbor; cd /opt/harbor
    ./venv/bin/harbor run -p $REMOTE -a $1 -n $W -k 1 -o /opt/harbor/jobs --job-name val-$TAG-$1 -y -q 2>&1 | tail -3
    for d in /opt/harbor/jobs/val-$TAG-$1/*__*/; do
      t=\$(basename \$d); t=\${t%%__*}; r=\$(cat \$d/verifier/reward.txt 2>/dev/null || echo NA)
      e=\$(python3 -c \"import json,sys; print((json.load(open('\$d/result.json')).get('exception_info') or {}).get('exception_type') or '')\" 2>/dev/null)
      echo \"\$t \$r \$e\"
    done"
}
echo "[validate] oracle 실행"; run_agent oracle > "$ROOT/.oracle.txt"
if [ "$MODE" = "--oracle-only" ]; then
  awk '$2=="1"{print $1}' "$ROOT/.oracle.txt" > "$ROOT/VALID.txt"
  awk '$2!="1"{print $1, "oracle_reward=" $2, $3}' "$ROOT/.oracle.txt" > "$ROOT/INVALID.txt"
else
  echo "[validate] nop 실행"; run_agent nop > "$ROOT/.nop.txt"
  python3 - "$ROOT" <<'PY'
import sys, os
root = sys.argv[1]
orc = {l.split()[0]: l.split()[1:] for l in open(os.path.join(root, ".oracle.txt")) if l.strip()}
nop = {l.split()[0]: l.split()[1:] for l in open(os.path.join(root, ".nop.txt")) if l.strip()}
valid, invalid = [], []
for t in sorted(set(orc) | set(nop)):
    o = orc.get(t, ["NA"])[0]; n = nop.get(t, ["NA"])[0]
    if o == "1" and n == "0": valid.append(t)
    else: invalid.append(f"{t} oracle={o} nop={n} {' '.join(orc.get(t, [])[1:])}")
open(os.path.join(root, "VALID.txt"), "w").write("\n".join(valid) + ("\n" if valid else ""))
open(os.path.join(root, "INVALID.txt"), "w").write("\n".join(invalid) + ("\n" if invalid else ""))
print(f"[validate] valid={len(valid)} invalid={len(invalid)}")
PY
fi
VALID_DIR="$(dirname "$ROOT")/${TAG}_valid"; mkdir -p "$VALID_DIR"
while read -r t; do [ -n "$t" ] && cp -r "$ROOT/$t" "$VALID_DIR/" ; done < "$ROOT/VALID.txt"
echo "[validate] 채택 $(wc -l < "$ROOT/VALID.txt")/$N → $VALID_DIR ; 사유는 $ROOT/INVALID.txt"
