#!/bin/bash
# run_swe.sh — SWE-bench Verified (에이전틱). mini-swe-agent(gpu06 컨테이너) → swebench 채점.
#
# 규약 (docs/SFT_BENCHMARKS.md §3.4·§7):
#   - 생성 temp 1.0 / top_p 0.95 (Nemotron 3 Ultra 동일). skip_special_tokens false.
#   - **전량 실행이 기본** — 부분 표본 결과는 프론티어 규약상 보고 대상이 아니다
#     (NVIDIA 재현 문서: "Never report sub-sampled / limited runs").
#     N 을 지정해 줄이면 결과 JSON 에 subsampled=true 가 박혀 집계에서 무효 처리된다.
#   - 투입 전 A1~A3 게이트 통과 필수 (check_agentic_gates.py).
#
# 전제: **TOOLS=1 로 서빙된 fleet** (mini-swe-agent litellm 이 tool_choice=auto 전송) +
#       sub1→컨테이너 역터널(컨테이너:8199 → sub1:8100).
#
# 사용: bash eval_sft/run_swe.sh <RUN_NAME> [N_INSTANCES] [WORKERS]
#   N_INSTANCES: 0 또는 미지정 = 전량(500). 양수면 부분 표본(무효 표시).
set -uo pipefail
RUN_NAME="${1:?run name}"; N="${2:-0}"; W="${3:-6}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SSHC="/home/work/vidsearch/.ssh-keys/config"
BASE_URL="${BASE_URL:-http://localhost:8100/v1}"
RID="alpha_$(echo "$RUN_NAME" | md5sum | cut -c1-10)"
OUT="$HERE/results/$RUN_NAME"; mkdir -p "$OUT"
HFTOK=$(grep -E '^HF_TOKEN=' "$HERE/../.env" 2>/dev/null | cut -d= -f2 || true)

# ---- 게이트 (SKIP_GATES=1 은 이미 통과한 실행의 재개 전용) ----
if [ "${SKIP_GATES:-0}" != "1" ]; then
  python3 "$HERE/check_agentic_gates.py" --base-url "$BASE_URL" --min-disk-gb "${MIN_DISK_GB:-300}" || {
    echo "[swe] ❌ 게이트 실패 — 중단. 이 상태의 0점은 모델 실패와 구분되지 않는다."; exit 1; }
fi

SLICE=""; SUBSAMPLED=false
if [ "$N" -gt 0 ] 2>/dev/null; then
  SLICE="--shuffle --slice 0:$N"; SUBSAMPLED=true
  echo "[swe] ⚠️ 부분 표본 $N 건 — 결과에 subsampled=true 가 박힌다(집계 무효)"
else
  echo "[swe] 전량(SWE-bench Verified 500)"
fi

echo "[swe] 예측 생성 (mini-swe-agent, W=$W workers, temp 1.0)"
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "
  export HF_TOKEN=$HFTOK OPENAI_API_KEY=dummy OPENAI_API_BASE=http://localhost:8199/v1
  cd /opt/swebench
  ./venv/bin/mini-extra swebench --subset SWE-bench/SWE-bench_Verified --split test \
    $SLICE --workers $W --redo-existing \
    -m openai/alpha -c swebench.yaml \
    -c model.model_kwargs.api_base=http://localhost:8199/v1 \
    -c model.model_kwargs.temperature=1.0 \
    -c model.model_kwargs.top_p=0.95 \
    -c model.model_kwargs.max_tokens=8192 \
    -o /opt/swebench/preds_${RUN_NAME} 2>&1 | tail -8
"
echo "[swe] 채점 (swebench eval)"
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "
  export HF_TOKEN=$HFTOK; cd /opt/swebench
  PREDS=\$(ls -t preds_${RUN_NAME}/preds.jsonl preds_${RUN_NAME}*.jsonl 2>/dev/null | head -1)
  ./venv/bin/swebench eval SWE-bench/SWE-bench_Verified -p \"\$PREDS\" --run-id $RID -j $W 2>&1 | tail -10
"
ssh -F "$SSHC" -o BatchMode=yes alpha-eval \
  "cat /opt/swebench/*$RID*.json 2>/dev/null || cat /opt/swebench/logs/run_evaluation/$RID/*/report.json 2>/dev/null" \
  > "$OUT/swe_report_raw.json" 2>/dev/null || true

python3 - "$OUT" "$SUBSAMPLED" <<'PY'
import json, sys, os
outd, sub = sys.argv[1], sys.argv[2] == "true"
raw = os.path.join(outd, "swe_report_raw.json")
resolved = total = 0
try:
    d = json.load(open(raw))
    resolved = d.get("resolved_instances", d.get("resolved", 0)) or 0
    total = d.get("total_instances", d.get("submitted_instances", 0)) or 0
except Exception:
    pass
acc = resolved / total if total else 0.0
res = {"resolved,none": acc}
# 부분 표본은 no_answer 를 1.0 으로 박아 집계기가 무효로 판정하게 한다 —
# 프론티어 규약상 부분 표본 결과는 보고 대상이 아니다.
if sub or total == 0:
    res["no_answer,none"] = 1.0
json.dump({"results": {"swe_bench_verified": res},
           "swe_detail": {"resolved": resolved, "total": total, "subsampled": sub}},
          open(os.path.join(outd, "results_swe.json"), "w"), indent=2)
mark = "  [부분표본 → 무효]" if sub else ("  [리포트 없음 → 무효]" if total == 0 else "")
print(f"[swe] resolved {resolved}/{total} = {acc*100:.1f}%{mark}")
PY
echo "== SWE 완료: $OUT =="
