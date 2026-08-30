#!/bin/bash
# run_terminal.sh — Terminal-Bench (에이전틱). tb(gpu06 컨테이너) terminus 에이전트로 우리 모델 평가.
# 전제: TOOLS=1 서빙 fleet + 역터널(컨테이너:8199→fleet).
# 사용: bash eval_sft/run_terminal.sh <RUN_NAME> [N_TASKS] [WORKERS]
set -uo pipefail
RUN_NAME="${1:?run name}"; N="${2:-10}"; W="${3:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"; SSHC="/home/work/vidsearch/.ssh-keys/config"
OUT="$HERE/results/$RUN_NAME"; mkdir -p "$OUT"
# 터널 확인
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "curl -s -o /dev/null -w '%{http_code}' --max-time 6 http://localhost:8199/v1/models" 2>/dev/null | grep -q 200 \
  || { echo "[term] 터널 기동"; bash /home/work/vidsearch/tools/start_swe_tunnel.sh; sleep 8; }
RID="tb$(echo "$RUN_NAME" | md5sum | cut -c1-8)"
echo "[term] tb run (terminus, N=$N tasks, W=$W)"
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "rm -rf /opt/terminalbench/runs/$RID" 2>/dev/null || true
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "
  export OPENAI_API_KEY=dummy OPENAI_API_BASE=http://localhost:8199/v1
  cd /opt/terminalbench
  ./venv/bin/tb run --agent terminus --model openai/alpha \
    -k api_base=http://localhost:8199/v1 -k temperature=0.0 -k max_tokens=8192 \
    --dataset terminal-bench-core==0.1.1 --n-tasks $N --n-concurrent $W \
    --run-id $RID --output-path /opt/terminalbench/runs 2>&1 | tail -12
"
# 결과 회수 (accuracy)
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "cat /opt/terminalbench/runs/$RID/results.json 2>/dev/null || find /opt/terminalbench/runs/$RID -name 'results.json' -exec cat {} \; 2>/dev/null" > "$OUT/terminal_raw.json" 2>/dev/null || true
python3 - "$OUT" "$N" <<'PY'
import json, sys, os
outd, n = sys.argv[1], int(sys.argv[2])
raw=os.path.join(outd,"terminal_raw.json"); acc=0.0; resolved=total=0
try:
    d=json.load(open(raw))
    acc=d.get("accuracy", d.get("resolved_rate", 0.0)) or 0.0
    resolved=d.get("n_resolved", d.get("resolved", 0)); total=d.get("n_tasks", d.get("total", n)) or n
    if acc==0 and total: acc=resolved/total
except Exception: total=n
json.dump({"results":{"terminal_bench":{"resolved,none":acc}},
           "terminal_detail":{"resolved":resolved,"total":total}},
          open(os.path.join(outd,"results_terminal.json"),"w"), indent=2)
print(f"[term] accuracy {acc*100:.1f}% ({resolved}/{total})")
PY
echo "== Terminal 완료: $OUT =="
