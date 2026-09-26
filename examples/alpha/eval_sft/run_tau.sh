#!/bin/bash
# run_tau.sh — τ³-bench (tau2-bench v1.0.1, Sierra). tool + agent + user 3자 상호작용에서 도구 사용 능력 측정.
#
# 구성 (docs/SFT_BENCHMARKS.md §3.13):
#   tau2 run (sub1) ─ agent: litellm openai/alpha ─▶ tau_proxy(:8110) ─▶ lb_proxy(:8100) ─▶ TOOLS=1 fleet
#                   └ user : litellm $TAU_USER_LLM  ─▶ 외부 엔드포인트 (기본 gemma-4-12B-it, 비용 0)
#   - docker 불요(도구는 JSON DB 위의 순수 파이썬) → gpu06 컨테이너·역터널 없이 sub1 에서 직접.
#   - fleet 는 **TOOLS=1 + REASONING_PARSER=nemotron_v3** 로 뜬다(run_suite τ 단계가 재기동; 게이트 T1b). vLLM 0.25.1
#     parser engine 은 tool 파서만 켜면 </think> 를 소비해 think 가 답변에 붙으므로 reasoning 필드로 받는다.
#     tau2 는 reasoning 을 재전송하지 않으므로 tau_proxy 가 think 를 캐시해 다음 요청의 히스토리에 복원한다(규칙 5).
#   - 기본 도메인 retail(114)+airline(50), base split. telecom(114, dual-control) 은 상대역이 도구를 써야 하는데
#     기본 상대역 엔드포인트가 tools 요청을 400 으로 거절한다 → preflight T2 가 판정해 건너뛰고 사유를 기록.
#   - 상대역이 12B 라 리더보드(gpt-5.2 user-sim)·보고서 수치와 직접 비교 불가. ckpt 간 추이가 목적이며
#     상대역이 고정이라 추이는 유효하다. 외부 비교치는 TAU_USER_LLM/TAU_USER_ARGS 로 별도 런.
#   - 생성 규약: temp 1.0 / top_p 0.95 / max_tokens 32768 / skip_special_tokens=false (runners/gen_common.py).
#     통신 프로토콜(문장·도구호출 동시 금지)은 강제하지 않는다 — 템플릿이 도구 호출 앞 자연어를 허용한다.
#   - 결과: results/<RUN>/results_tau.json — tau_<domain>·tau_bench 의 pass^1..pass^K (리더보드 주지표 pass^1),
#     진단 no_answer(하니스 측 실패율)·think_closed. 무효 조건이면 no_answer=1.0 (집계에서 자동 무효).
#
# 사용: bash eval_sft/run_tau.sh <RUN_NAME> [N_TASKS=0] [TRIALS=4] [W=8]
#   N_TASKS: 0=전량. 양수면 도메인당 앞 N 과제(부분 표본, 무효 표시).  W: 동시 시뮬레이션 수.
# 환경변수: TAU_JUDGE_LLM(gemini/gemini-3.7-flash; NL 판정기, 2026-09-27) TAU_DOMAINS("retail airline") TAU_SPLIT_<dom>(base) TAU_USER_LLM TAU_USER_ARGS TAU_MAX_TOKENS(32768)
#   TAU_REATTACH(1; 0=프록시 복원 끔, differential 용) TAU_FRESH(0; 1=기존 시뮬레이션 삭제 후 처음부터)
#   TAU_TASK_IDS("" ; 지정 시 --task-ids) TAU_DOMAIN_TIMEOUT(12h) TAU_SIM_TIMEOUT(7200) BASE_URL SKIP_GATES
set -uo pipefail
RUN_NAME="${1:?run name}"; N="${2:-0}"; TRIALS="${3:-4}"; W="${4:-8}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE_URL="${BASE_URL:-http://localhost:8100/v1}"
TAU2_HOME="${TAU2_HOME:-/home/work/vidsearch/tools/tau2-bench}"
TAU2="$TAU2_HOME/.venv/bin/tau2"; PY="$TAU2_HOME/.venv/bin/python"
DATA_DIR="${TAU2_DATA_DIR:-$TAU2_HOME/data}"
PROXY_PORT="${TAU_PROXY_PORT:-8110}"
TAU_DOMAINS="${TAU_DOMAINS:-retail airline}"
TAU_AGENT_LLM="${TAU_AGENT_LLM:-openai/alpha}"
TAU_USER_LLM="${TAU_USER_LLM:-openai/gemma-4-12B-it}"
TAU_USER_ARGS="${TAU_USER_ARGS:-{\"api_base\":\"https://gemma4.withai.cj.net:10206/v1\",\"api_key\":\"dummy\",\"temperature\":0.0,\"max_tokens\":1024\}}"
TAU_MAX_TOKENS="${TAU_MAX_TOKENS:-32768}"
TAU_REATTACH="${TAU_REATTACH:-1}"
TAU_TASK_IDS="${TAU_TASK_IDS:-}"
OUT="$HERE/results/$RUN_NAME"; RAW="$OUT/tau_raw"; mkdir -p "$RAW"
RID="tau_$(echo "$RUN_NAME" | md5sum | cut -c1-8)"
[ -x "$TAU2" ] || { echo "[tau] ❌ tau2-bench 미설치: bash eval_sft/install_tau2.sh"; exit 1; }

export PIP_CONSTRAINT=
export LITELLM_MODEL_REGISTRY_PATH="$HERE/configs/alpha_model_registry.json"
export OPENAI_API_KEY=dummy OPENAI_API_BASE="http://localhost:$PROXY_PORT/v1"
# 외부 상대역용 키 — .env 는 source 하지 않는다(3번째 줄이 이름 없는 값). 필요한 줄만 뽑는다.
for K in GEMINI_API_KEY OPENROUTER_API_KEY; do
  if [ -z "${!K:-}" ]; then v=$(grep -E "^$K=" "$HERE/../.env" 2>/dev/null | cut -d= -f2- | tr -d "\"'" || true); [ -n "$v" ] && export "$K=$v"; fi
done
# NL-assertion 판정기(retail 40/114 과제의 채점 단계) — 사용자 결정 2026-09-27: gemini-3.7-flash(LogicKor 판정기와 동일 계열).
# upstream 은 gpt-4.1 하드코딩 → vendored config.py 에 env 오버라이드(tau2_judge_patch.py, install_tau2.sh 가 적용).
# Gemini 는 response_format=json_object 없이는 ```json 펜스를 씌워 json.loads 가 실패한다(2026-09-27 스모크) → EXTRA_ARGS 로 강제.
TAU_JUDGE_LLM="${TAU_JUDGE_LLM:-gemini/gemini-3.7-flash}"
export TAU2_LLM_NL_ASSERTIONS="$TAU_JUDGE_LLM"
case "$TAU_JUDGE_LLM" in gemini/*) export TAU2_LLM_NL_ASSERTIONS_EXTRA_ARGS='{"response_format":{"type":"json_object"}}';; esac
AGENT_ARGS=$(printf '{"api_base":"http://localhost:%s/v1","api_key":"dummy","temperature":1.0,"top_p":0.95,"max_tokens":%s,"extra_body":{"skip_special_tokens":false}}' "$PROXY_PORT" "$TAU_MAX_TOKENS")

# ── 게이트 A1·A4 (SKIP_GATES=1 은 스위트가 이미 통과시킨 경우) ────────────────────
if [ "${SKIP_GATES:-0}" != "1" ]; then
  python3 "$HERE/check_agentic_gates.py" --base-url "$BASE_URL" --skip-container || {
    echo "[tau] ❌ 게이트 실패 — 중단. 이 상태의 0점은 모델 실패와 구분되지 않는다."; exit 1; }
fi

# ── T1: tau_proxy ─────────────────────────────────────────────────────────────
if curl -s -m 3 -o /dev/null "http://localhost:$PROXY_PORT/stats"; then
  echo "[tau] ❌ :$PROXY_PORT 이미 사용 중 — 다른 τ 런이 진행 중이다(한 번에 하나)."; exit 1
fi
PFLAGS=""; [ "$TAU_REATTACH" = "0" ] && PFLAGS="--no-reattach"
setsid python3 "$HERE/tau_proxy.py" --port "$PROXY_PORT" --upstream "${BASE_URL%/v1}" \
  --stats-file "$RAW/proxy_stats.json" --dump-dir "$RAW" $PFLAGS > "$RAW/proxy.log" 2>&1 < /dev/null &
PROXY_PID=$!
trap 'kill -TERM $PROXY_PID 2>/dev/null; sleep 1; kill -9 $PROXY_PID 2>/dev/null; true' EXIT
for i in $(seq 1 15); do curl -s -m 2 -o /dev/null "http://localhost:$PROXY_PORT/stats" && break; sleep 1; done
curl -s -m 2 -o /dev/null "http://localhost:$PROXY_PORT/stats" || { echo "[tau] ❌ tau_proxy 기동 실패: $RAW/proxy.log"; exit 1; }
[ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://localhost:$PROXY_PORT/v1/models")" = "200" ] \
  || { echo "[tau] ❌ 프록시 → fleet(${BASE_URL}) 통과 실패"; exit 1; }
echo "[tau] T1 프록시 :$PROXY_PORT → ${BASE_URL%/v1} (reattach=$TAU_REATTACH)"
# T1b: fleet 가 think 를 분리해 주는가 — reasoning 파서(nemotron_v3)가 켜져 있어야 한다. tool 파서만 켜진 fleet 는
# 도구 턴에서 </think> 가 사라져 think 가 답변에 붙는다(2026-09-14 스모크: 복원 miss 57%, 상대역이 think 를 읽음).
python3 - "$BASE_URL" <<'PY' || { echo "[tau] ❌ T1b: fleet 응답에 reasoning 필드가 없다 — REASONING_PARSER=nemotron_v3 TOOLS=1 로 fleet 를 재기동할 것"; exit 1; }
import json, sys, urllib.request
body = {"model": "alpha", "messages": [{"role": "user", "content": "Reply with exactly: OK"}], "max_tokens": 1024,
        "temperature": 1.0, "top_p": 0.95, "skip_special_tokens": False}
req = urllib.request.Request(sys.argv[1].rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
m = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]
field = m.get("reasoning_content") or m.get("reasoning")
ok = isinstance(field, str) and field.strip() and "</think>" not in (m.get("content") or "")
print(f"[tau] T1b fleet reasoning 필드 {'OK' if ok else '없음'} (content={repr((m.get('content') or '')[:60])})")
sys.exit(0 if ok else 1)
PY

# ── T2: 상대역 preflight (chat 1회 + tools 1회) ───────────────────────────────
T2_OUT=$("$PY" - "$TAU_USER_LLM" "$TAU_USER_ARGS" <<'PY' 2>"$RAW/user_preflight.err"
import json, sys, logging
logging.disable(logging.CRITICAL)
import litellm
litellm.suppress_debug_info = True
model, args = sys.argv[1], json.loads(sys.argv[2])
a = {k: v for k, v in args.items() if k != "max_tokens"}
r = litellm.completion(model=model, messages=[{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=64, **a)
txt = (r.choices[0].message.content or "").strip()
if not txt:
    print("chat returned empty content", file=sys.stderr); sys.exit(2)
tools = [{"type": "function", "function": {"name": "noop", "description": "no-op", "parameters": {"type": "object", "properties": {}}}}]
try:
    litellm.completion(model=model, messages=[{"role": "user", "content": "Say hi."}], tools=tools, tool_choice="auto", max_tokens=32, **a)
    print("USER_TOOLS=1")
except Exception as e:  # noqa: BLE001
    print(f"tools rejected: {str(e)[:160]}", file=sys.stderr); print("USER_TOOLS=0")
PY
)
rc=$?
USER_TOOLS=$(printf '%s\n' "$T2_OUT" | grep -oE '^USER_TOOLS=[01]$' | tail -1 | cut -d= -f2)
[ -z "$USER_TOOLS" ] && rc=1
if [ "$rc" -ne 0 ]; then echo "[tau] ❌ T2 상대역($TAU_USER_LLM) 응답 실패: $(head -c 300 "$RAW/user_preflight.err")"; exit 1; fi
echo "[tau] T2 상대역 $TAU_USER_LLM OK (tools=$USER_TOOLS)"
# T3: NL-assertion 판정기 1회 호출(tau_judge_preflight.py) — retail 이 있을 때만. 실패 = retail 채점 전부 infra_error 이므로 중단.
if echo " $TAU_DOMAINS " | grep -q " retail "; then
  (cd "$TAU2_HOME" && PIP_CONSTRAINT= "$PY" "$HERE/tau_judge_preflight.py") \
    || { echo "[tau] ❌ T3 판정기($TAU_JUDGE_LLM) 실패 — retail 채점 불가, 중단"; exit 1; }
fi

# ── 표본·도메인 ────────────────────────────────────────────────────────────────
NTASKS=""; SUBSAMPLED=false
if [ "$N" -gt 0 ] 2>/dev/null; then NTASKS="--num-tasks $N"; SUBSAMPLED=true
  echo "[tau] ⚠️ 부분 표본 도메인당 $N 과제 — 결과에 subsampled=true (집계 무효)"; fi
if [ -n "$TAU_TASK_IDS" ]; then NTASKS="--task-ids $TAU_TASK_IDS"; SUBSAMPLED=true
  echo "[tau] ⚠️ task-ids 지정($TAU_TASK_IDS) — 부분 표본(집계 무효)"; fi
SKIPPED_JSON="{}"
for d in $TAU_DOMAINS; do
  if [ "$d" = "telecom" ] && [ "$USER_TOOLS" != "1" ]; then
    echo "[tau] ⏭ telecom 건너뜀 — 상대역이 tools 요청을 거절한다(dual-control 불가). TAU_USER_LLM 을 도구 가능 모델로."
    SKIPPED_JSON=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); d['telecom']='user endpoint rejects tools'; print(json.dumps(d))" "$SKIPPED_JSON")
    continue
  fi
  sv="TAU_SPLIT_$d"; SPLIT="${!sv:-base}"
  [ "${TAU_FRESH:-0}" = "1" ] && rm -rf "$DATA_DIR/simulations/${RID}_$d"
  curl -s "http://localhost:$PROXY_PORT/stats" > "$RAW/proxy_${d}_before.json"
  echo "[tau] === $d (split=$SPLIT, trials=$TRIALS, W=$W) → $DATA_DIR/simulations/${RID}_$d ==="
  ( cd "$TAU2_HOME" && timeout "${TAU_DOMAIN_TIMEOUT:-12h}" "$TAU2" run --domain "$d" \
      --agent llm_agent --agent-llm "$TAU_AGENT_LLM" --agent-llm-args "$AGENT_ARGS" \
      --user user_simulator --user-llm "$TAU_USER_LLM" --user-llm-args "$TAU_USER_ARGS" \
      --task-split-name "$SPLIT" --num-trials "$TRIALS" $NTASKS \
      --max-steps 200 --max-errors 10 --max-concurrency "$W" --seed 300 --timeout "${TAU_SIM_TIMEOUT:-7200}" \
      --save-to "${RID}_$d" --auto-resume --log-level WARNING ) > "$RAW/tau2_${d}.log" 2>&1
  echo "[tau] tau2 run rc=$? — $(grep -cE 'Traceback|Error' "$RAW/tau2_${d}.log") error lines (로그 $RAW/tau2_${d}.log)"
  curl -s "http://localhost:$PROXY_PORT/stats" > "$RAW/proxy_${d}_after.json"
  rm -rf "$RAW/$d"; cp -r "$DATA_DIR/simulations/${RID}_$d" "$RAW/$d" 2>/dev/null || echo "[tau] ⚠️ $d 결과 없음"
done

# ── 합산 → results_tau.json (tau_combine.py; tests/test_tau_combine.py 로 검증) ─────────
"$PY" "$HERE/tau_combine.py" --out "$OUT" --raw "$RAW" --trials "$TRIALS" --domains "$TAU_DOMAINS" \
  --skipped-json "$SKIPPED_JSON" --user-llm "$TAU_USER_LLM" --user-args "$TAU_USER_ARGS" --agent-args "$AGENT_ARGS" --judge-llm "$TAU_JUDGE_LLM" \
  --reattach "$TAU_REATTACH" --home "$TAU2_HOME" $( [ "$SUBSAMPLED" = true ] && echo --subsampled )
