#!/bin/bash
# run_terminal.sh — Terminal-Bench (에이전틱). tb(gpu06 컨테이너) terminus 에이전트.
#
# 규약 (docs/SFT_BENCHMARKS.md §3.4·§7):
#   - 생성 temp 1.0 / top_p 0.95 (Nemotron 3 Ultra 동일).
#   - **전량 실행이 기본** (terminal-bench-core 0.1.1 = 80 tasks). N 을 주면 부분 표본이
#     되고 결과에 subsampled=true 가 박혀 집계에서 무효 처리된다.
#   - 투입 전 A1~A4 게이트 통과 필수.
#   - **max_tokens 는 넉넉해야 한다** (기본 32768, TERM_MAX_TOKENS 로 조절).
#     terminus 는 finish_reason == "length" 를 받으면 OutputLengthExceededError 를 던지고
#     **태스크를 즉시 중단한다** — 재시도가 없다 (lite_llm.py:175). mini-swe-agent 가
#     RepeatedFormatError 로 몇 번 더 시도하는 것과 다르다.
#     2026-08-31 실측: 16384 로 돌린 80건 중 44건(55%)이 unknown_agent_error 로 죽었고,
#     원인은 전부 "hit max_tokens limit". 0/80 은 모델 실력이 아니라 설정 문제였다.
#
# 전제: **TOOLS=1 로 서빙된 fleet** + 역터널(컨테이너:8199 → sub1:8100).
#
# 사용: bash eval_sft/run_terminal.sh <RUN_NAME> [N_TASKS] [WORKERS]
#   N_TASKS: 0 또는 미지정 = 전량(80). 양수면 부분 표본(무효 표시).
set -uo pipefail
RUN_NAME="${1:?run name}"; N="${2:-0}"; W="${3:-8}"
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
    -k api_base=http://localhost:8199/v1 -k temperature=1.0 -k top_p=0.95 \
    -k max_tokens=${TERM_MAX_TOKENS:-32768} \
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
    # terminal-bench results.json 스키마 (0.2.x): 최상위에 n_resolved / n_unresolved /
    # accuracy 가 있고, results 는 trial 리스트다. n_tasks/total 은 없다 —
    # 그것을 찾다 total=0 이 되어 정상 결과가 무효로 찍혔다 (2026-08-31 수정).
    resolved = int(d.get("n_resolved", 0) or 0)
    unresolved = int(d.get("n_unresolved", 0) or 0)
    total = resolved + unresolved or len(d.get("results") or [])
    acc = d.get("accuracy")
    if acc is None:
        acc = resolved / total if total else 0.0
    acc = float(acc)
except Exception:
    pass
res = {"resolved,none": acc}
if sub or total == 0:
    res["no_answer,none"] = 1.0   # 부분 표본/결과 없음 → 집계 무효
import collections
modes = collections.Counter(str(r.get("failure_mode")) for r in (d.get("results") or []))
json.dump({"results": {"terminal_bench": res},
           "terminal_detail": {"resolved": resolved, "total": total, "subsampled": sub,
                               "failure_modes": dict(modes.most_common())}},
          open(os.path.join(outd, "results_terminal.json"), "w"), indent=2)
if modes:
    print(f"[term] 실패 원인: {dict(modes.most_common(4))}")
mark = "  [부분표본 → 무효]" if sub else ("  [결과 없음 → 무효]" if total == 0 else "")
print(f"[term] accuracy {acc*100:.1f}% ({resolved}/{total}){mark}")
PY
echo "== Terminal 완료: $OUT =="
