#!/bin/bash
# run_terminal_tb2.sh — Terminal-Bench **2.0** (Harbor + Terminus-2).
#
# 왜 TB-1 을 대체하는가 (사용자 결정 2026-09-07, `SFT_RL_DATASETS.md` §2.9):
#   학습 데이터의 Terminus 행은 **Terminus-2 스키마**
#   (`analysis/plan/commands{keystrokes,duration}/task_complete`)인데, 구 하니스는
#   TB-1 core 0.1.1 + terminus **v1**(4필드 `state_analysis/explanation/commands/
#   is_task_complete`)이었다. 스키마는 하니스 에이전트가 정하므로 v1 하니스에 v2 형식을
#   끼워 넣을 수 없다. Ultra 공개 수치도 TB 2.0/2.1 기준이라 비교 조건이 맞지 않았다.
#   TB-1 수치는 참고치로 남기고(유효 계열이 짧아 연속성 손실 없음) 이쪽을 정본으로 쓴다.
#
# 실행 규약은 TB-1 에서 **승계**한다: 반복 8 · temp 1.0 / top_p 0.95 ·
# max_tokens 65,536 · A1~A4 게이트 · 전량 실행(부분 표본은 무효 표시).
#
# TB-1 → TB-2 플래그 대응 (2026-09-07 `harbor run --help` 실측):
#   tb run                → harbor run
#   --agent terminus      → -a terminus-2
#   -k api_base=…         → --ak api_base=…      (--ak 값은 JSON 리터럴로 파싱된다)
#   --dataset core==0.1.1 → -d terminal-bench@2.0   (80 tasks → **89 tasks**)
#   --n-attempts K        → -k K                  ⚠️ TB-1 의 -k(agent kwarg)와 의미가 다르다
#   --n-concurrent W      → -n W
#   --n-tasks N           → -l N
#   --run-id / --output-path → --job-name / -o
#
# parser_name 은 **json** 이다 — 학습 데이터가 Terminus-2 JSON 스키마다.
# (terminus-2 는 json/xml 두 파서를 갖고 각각 프롬프트 템플릿이 다르다.)
#
# 사용: bash eval_sft/run_terminal_tb2.sh <RUN_NAME> [N_TASKS] [W]
#   N_TASKS: 0/미지정 = 전량(89). 양수면 부분 표본(무효 표시).
set -uo pipefail
RUN_NAME="${1:?run name}"; N="${2:-0}"; W="${3:-8}"
HERE="$(cd "$(dirname "$0")" && pwd)"; SSHC="/home/work/vidsearch/.ssh-keys/config"
BASE_URL="${BASE_URL:-http://localhost:8100/v1}"
OUT="$HERE/results/$RUN_NAME"; mkdir -p "$OUT"
# compose 프로젝트명 길이 제한 때문에 job 이름은 짧게 (2026-08-30 사고, 71a1d84)
RID="tb2$(echo "$RUN_NAME" | md5sum | cut -c1-8)"

if [ "${SKIP_GATES:-0}" != "1" ]; then
  python3 "$HERE/check_agentic_gates.py" --base-url "$BASE_URL" --min-disk-gb "${MIN_DISK_GB:-150}" || {
    echo "[term2] ❌ 게이트 실패 — 중단."; exit 1; }
fi

NTASKS=""; SUBSAMPLED=false
if [ "$N" -gt 0 ] 2>/dev/null; then
  NTASKS="-l $N"; SUBSAMPLED=true
  echo "[term2] ⚠️ 부분 표본 $N 태스크 — 결과에 subsampled=true (집계 무효)"
else
  echo "[term2] 전량 (terminal-bench@2.0, 89 tasks)"
fi

echo "[term2] harbor run (terminus-2, parser=json, W=$W, k=${TERM_REPEATS:-8}, temp 1.0)"
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "rm -rf /opt/harbor/jobs/$RID" 2>/dev/null || true
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "
  export HOME=/opt/harbor
  export OPENAI_API_KEY=dummy OPENAI_API_BASE=http://localhost:8199/v1
  # terminus-2 도 litellm 을 쓴다 — 미등록 모델 비용 계산 실패 방지 (run_swe.sh 와 동일 사유)
  export LITELLM_MODEL_REGISTRY_PATH=/opt/terminalbench/alpha_model_registry.json
  cd /opt/harbor
  ./venv/bin/harbor run -d terminal-bench@2.0 -a terminus-2 -m openai/alpha \
    --ak api_base=http://localhost:8199/v1 \
    --ak temperature=1.0 \
    --ak parser_name=json \
    --ak 'llm_call_kwargs={\"top_p\":0.95,\"max_tokens\":${TERM_MAX_TOKENS:-65536}}' \
    $NTASKS -n $W -k ${TERM_REPEATS:-8} \
    -o /opt/harbor/jobs --job-name $RID -y -q 2>&1 | tail -20
"
ssh -F "$SSHC" -o BatchMode=yes alpha-eval \
  "cat /opt/harbor/jobs/$RID/result.json 2>/dev/null" > "$OUT/terminal_raw.json" 2>/dev/null || true

python3 - "$OUT" "$SUBSAMPLED" <<'PY'
import json, sys, os
outd, sub = sys.argv[1], sys.argv[2] == "true"
raw = os.path.join(outd, "terminal_raw.json")
acc = 0.0; ntr = nerr = 0; rewards = {}; exc = {}
try:
    d = json.load(open(raw))
    ev = (d.get("stats") or {}).get("evals") or {}
    # 키는 "<agent>__<dataset>" — 하나뿐이다.
    for _, v in ev.items():
        ntr = int(v.get("n_trials") or 0)
        nerr = int(v.get("n_errors") or 0)
        m = v.get("metrics") or [{}]
        acc = float(m[0].get("mean") or 0.0)
        rewards = {k: len(x) for k, x in ((v.get("reward_stats") or {}).get("reward") or {}).items()}
        exc = {k: (len(x) if hasattr(x, "__len__") else x) for k, x in (v.get("exception_stats") or {}).items()}
        break
except Exception as e:  # noqa: BLE001
    print(f"[term2] ⚠️ 결과 파싱 실패: {e}")

res = {"results": {"terminal_bench_2": {"resolved,none": acc}},
       "terminal_detail": {"harness": "terminal-bench@2.0 + harbor + terminus-2",
                           "n_trials": ntr, "n_errors": nerr,
                           "reward_counts": rewards, "exception_stats": exc,
                           "subsampled": sub}}
json.dump(res, open(os.path.join(outd, "results_terminal.json"), "w"), ensure_ascii=False, indent=2)
print(f"[term2] accuracy {acc*100:.1f}%  trials={ntr} errors={nerr}")
if exc: print(f"[term2] 예외 분포: {exc}")
print(f"[term2] → {outd}/results_terminal.json")
PY
