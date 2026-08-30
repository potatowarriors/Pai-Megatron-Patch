#!/bin/bash
# run_terminal.sh — Terminal-Bench (에이전틱). tb(gpu06 컨테이너) terminus 에이전트.
#
# 규약 (docs/SFT_BENCHMARKS.md §3.4·§7):
#   - 생성 temp 1.0 / top_p 0.95 (Nemotron 3 Ultra 동일).
#   - **전량 실행이 기본** (terminal-bench-core 0.1.1 = 80 tasks). N 을 주면 부분 표본이
#     되고 결과에 subsampled=true 가 박혀 집계에서 무효 처리된다.
#   - 투입 전 A1~A3 게이트 통과 필수.
#
# 전제: **TOOLS=1 로 서빙된 fleet** + 역터널(컨테이너:8199 → sub1:8100).
#
# 사용: bash eval_sft/run_terminal.sh <RUN_NAME> [N_TASKS] [WORKERS]
#   N_TASKS: 0 또는 미지정 = 전량(80). 양수면 부분 표본(무효 표시).
set -uo pipefail
RUN_NAME="${1:?run name}"; N="${2:-0}"; W="${3:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"; SSHC="/home/work/vidsearch/.ssh-keys/config"
BASE_URL="${BASE_URL:-http://localhost:8100/v1}"
OUT="$HERE/results/$RUN_NAME"; mkdir -p "$OUT"
# compose 프로젝트명 길이 제한 때문에 run-id 는 짧게 (2026-08-30 사고, 71a1d84)
RID="tb$(echo "$RUN_NAME" | md5sum | cut -c1-8)"

if [ "${SKIP_GATES:-0}" != "1" ]; then
  python3 "$HERE/check_agentic_gates.py" --base-url "$BASE_URL" --min-disk-gb "${MIN_DISK_GB:-150}" || {
    echo "[term] ❌ 게이트 실패 — 중단."; exit 1; }
fi

NTASKS=""; SUBSAMPLED=false
if [ "$N" -gt 0 ] 2>/dev/null; then
  NTASKS="--n-tasks $N"; SUBSAMPLED=true
  echo "[term] ⚠️ 부분 표본 $N 태스크 — 결과에 subsampled=true (집계 무효)"
else
  echo "[term] 전량 (terminal-bench-core 0.1.1, 80 tasks)"
fi

echo "[term] tb run (terminus, W=$W, temp 1.0)"
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "rm -rf /opt/terminalbench/runs/$RID" 2>/dev/null || true
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "
  export OPENAI_API_KEY=dummy OPENAI_API_BASE=http://localhost:8199/v1
  # terminus 도 litellm 을 쓴다 — 미등록 모델 비용 계산 실패 방지 (run_swe.sh 와 동일 사유)
  export LITELLM_MODEL_REGISTRY_PATH=/opt/terminalbench/alpha_model_registry.json
  export MSWEA_COST_TRACKING=ignore_errors
  cd /opt/terminalbench
  ./venv/bin/tb run --agent terminus --model openai/alpha \
    -k api_base=http://localhost:8199/v1 -k temperature=1.0 -k top_p=0.95 -k max_tokens=16384 \
    --dataset terminal-bench-core==0.1.1 $NTASKS --n-concurrent $W \
    --run-id $RID --output-path /opt/terminalbench/runs 2>&1 | tail -14
"
ssh -F "$SSHC" -o BatchMode=yes alpha-eval \
  "cat /opt/terminalbench/runs/$RID/results.json 2>/dev/null || find /opt/terminalbench/runs/$RID -name 'results.json' -exec cat {} \; 2>/dev/null" \
  > "$OUT/terminal_raw.json" 2>/dev/null || true

python3 - "$OUT" "$SUBSAMPLED" <<'PY'
import json, sys, os
outd, sub = sys.argv[1], sys.argv[2] == "true"
raw = os.path.join(outd, "terminal_raw.json")
acc = 0.0; resolved = total = 0
try:
    d = json.load(open(raw))
    acc = d.get("accuracy", d.get("resolved_rate", 0.0)) or 0.0
    resolved = d.get("n_resolved", d.get("resolved", 0)) or 0
    total = d.get("n_tasks", d.get("total", 0)) or 0
    if acc == 0 and total:
        acc = resolved / total
except Exception:
    pass
res = {"resolved,none": acc}
if sub or total == 0:
    res["no_answer,none"] = 1.0   # 부분 표본/결과 없음 → 집계 무효
json.dump({"results": {"terminal_bench": res},
           "terminal_detail": {"resolved": resolved, "total": total, "subsampled": sub}},
          open(os.path.join(outd, "results_terminal.json"), "w"), indent=2)
mark = "  [부분표본 → 무효]" if sub else ("  [결과 없음 → 무효]" if total == 0 else "")
print(f"[term] accuracy {acc*100:.1f}% ({resolved}/{total}){mark}")
PY
echo "== Terminal 완료: $OUT =="
