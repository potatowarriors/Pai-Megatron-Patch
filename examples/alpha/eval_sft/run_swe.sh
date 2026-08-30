#!/bin/bash
# run_swe.sh — SWE-bench Verified 배치 (에이전틱). mini-swe-agent(gpu06 컨테이너)로 예측 생성 → swebench 채점.
# 전제: TOOLS=1 서빙 fleet + sub1→컨테이너 역터널(start_swe_tunnel.sh, 컨테이너:8199→fleet).
# 사용: bash eval_sft/run_swe.sh <RUN_NAME> [N_INSTANCES] [WORKERS]
set -uo pipefail
RUN_NAME="${1:?run name}"; N="${2:-50}"; W="${3:-6}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SSHC="/home/work/vidsearch/.ssh-keys/config"
RID="alpha_${RUN_NAME}"
OUT="$HERE/results/$RUN_NAME"; mkdir -p "$OUT"
HFTOK=$(grep -E '^HF_TOKEN=' "$HERE/../.env" 2>/dev/null | cut -d= -f2)

# 터널 살아있는지 확인, 없으면 기동
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "curl -s -o /dev/null -w '%{http_code}' --max-time 6 http://localhost:8199/v1/models" 2>/dev/null | grep -q 200 \
  || { echo "[swe] 터널 없음 → 기동"; bash /home/work/vidsearch/tools/start_swe_tunnel.sh; sleep 8; }

echo "[swe] 예측 생성 (mini-swe-agent, N=$N insts, W=$W workers)"
# 컨테이너에서 배치 예측 생성 → preds.jsonl
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "
  export HF_TOKEN=$HFTOK OPENAI_API_KEY=dummy OPENAI_API_BASE=http://localhost:8199/v1
  cd /opt/swebench
  ./venv/bin/mini-extra swebench --subset SWE-bench/SWE-bench_Verified --split test \
    --shuffle --slice 0:$N --workers $W --redo-existing \
    -m openai/alpha -c swebench.yaml \
    -c model.model_kwargs.api_base=http://localhost:8199/v1 \
    -c model.model_kwargs.temperature=0.0 \
    -c model.model_kwargs.max_tokens=8192 \
    -o /opt/swebench/preds_${RUN_NAME} 2>&1 | tail -5
"
echo "[swe] 채점 (swebench eval)"
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "
  export HF_TOKEN=$HFTOK; cd /opt/swebench
  PREDS=\$(ls -t preds_${RUN_NAME}/preds.jsonl preds_${RUN_NAME}*.jsonl 2>/dev/null | head -1)
  ./venv/bin/swebench eval SWE-bench/SWE-bench_Verified -p \"\$PREDS\" --run-id $RID -j $W 2>&1 | tail -8
"
# 리포트 회수 → 결과 JSON (집계기 호환)
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "cat /opt/swebench/*$RID*.json 2>/dev/null || cat /opt/swebench/logs/run_evaluation/$RID/*/report.json 2>/dev/null" > "$OUT/swe_report_raw.json" 2>/dev/null || true
python3 - "$OUT" "$N" <<'PY'
import json, sys, glob, os
outd, n = sys.argv[1], int(sys.argv[2])
raw=os.path.join(outd,"swe_report_raw.json")
resolved=total=0
try:
    d=json.load(open(raw))
    resolved=d.get("resolved_instances", d.get("resolved", 0))
    total=d.get("total_instances", d.get("submitted_instances", n)) or n
except Exception: total=n
acc=resolved/total if total else 0.0
json.dump({"results":{"swe_bench_verified":{"resolved,none":acc}},
           "swe_detail":{"resolved":resolved,"total":total}},
          open(os.path.join(outd,"results_swe.json"),"w"), indent=2)
print(f"[swe] resolved {resolved}/{total} = {acc*100:.1f}%")
PY
echo "== SWE 완료: $OUT =="
