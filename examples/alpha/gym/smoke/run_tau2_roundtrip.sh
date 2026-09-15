#!/bin/bash
# 2단계 게이트 — alpha fleet(TOOLS=1 REASONING_PARSER=nemotron_v3) 에 Gym vllm_model 을 붙여 τ² 를 몇 문항 돌리고
# 추론 왕복(파서 분리 + 이전 턴 restore)을 smoke/roundtrip_gate.py 로 판정한다. 09-14 tau_proxy 스모크와 같은 판정 기준.
#
# 사용 (fleet 가 떠 있는 상태에서, GPU 불필요, sub1/main1 어디서나 — 단 Ray 가 뜨는 노드여야 한다):
#   FLEET_URL=http://sub1:8100/v1 bash examples/alpha/gym/smoke/run_tau2_roundtrip.sh [LIMIT=5] [REPEATS=2]
# 산출: examples/alpha/gym/results/tau2_roundtrip_<ts>/{env_start.log, vllm_transport.jsonl, rollouts*.jsonl, gate.txt, gate.json}
# 주의: 첫 실행은 tau2 에이전트 venv(tau2-bench git 설치)를 만들어 수 분 걸린다. 시뮬레이터는 Gemini(.env GEMINI_API_KEY).
set -euo pipefail
LIMIT="${1:-5}"; REPEATS="${2:-2}"
GYM_ROOT="${GYM_ROOT:-/home/work/vidsearch/tools/Gym}"
ALPHA_GYM="$(cd "$(dirname "$0")/.." && pwd)"
ALPHA_ROOT="$(cd "$ALPHA_GYM/.." && pwd)"
GATE_PY="${GATE_PY:-/home/work/vidsearch/tools/alpha_serve_venv/bin/python}"   # tokenizers 가 있는 파이썬
FLEET_URL="${FLEET_URL:?FLEET_URL=http://<host>:<port>/v1 필요 (serve_alpha.sh TOOLS=1 REASONING_PARSER=nemotron_v3)}"
OUT="${OUT:-$ALPHA_GYM/results/tau2_roundtrip_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

set -a; source "$ALPHA_ROOT/.env"; set +a
: "${GEMINI_API_KEY:?examples/alpha/.env 에 GEMINI_API_KEY 필요}"
export PATH=/home/work/vidsearch/tools/bin:$PATH
export UV_CACHE_DIR=/home/work/vidsearch/tools/uv_cache UV_PYTHON_INSTALL_DIR=/home/work/vidsearch/tools/uv_python RAY_TMPDIR=/tmp
export NEMO_GYM_VLLM_TRANSPORT_LOG="$OUT/vllm_transport.jsonl"
cd "$GYM_ROOT"

# T1b 상당: fleet 가 reasoning 을 분리해 주는지 먼저 1회 확인 (파서 없는 fleet 에 붙이면 Gym 이 assert 로 죽어 원인이 흐려진다)
python3 - "$FLEET_URL" <<'PY'
import json, sys, urllib.request
url = sys.argv[1].rstrip("/") + "/chat/completions"
body = {"model": "alpha", "messages": [{"role": "user", "content": "1+1=? 한 단어로."}], "max_tokens": 64,
        "tools": [{"type": "function", "function": {"name": "noop", "description": "no-op", "parameters": {"type": "object", "properties": {}}}}]}
req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"})
msg = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]
ok = bool(msg.get("reasoning") or msg.get("reasoning_content"))
print(f"[preflight] reasoning field={'yes' if ok else 'NO'} content={str(msg.get('content'))[:60]!r}")
sys.exit(0 if ok else 1)
PY

nohup .venv/bin/gym env start \
  --config "$ALPHA_GYM/configs/tau2_alpha.yaml" \
  --config "$ALPHA_GYM/configs/alpha_vllm_model_gate.yaml" \
  ++policy_base_url="$FLEET_URL" ++policy_api_key=dummy ++policy_model_name=alpha \
  ++user_sim_api_key="$GEMINI_API_KEY" ++user_sim_model_name="${USER_SIM_MODEL:-gemini-2.5-flash}" \
  > "$OUT/env_start.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true; sleep 3; .venv/bin/ray stop --force >/dev/null 2>&1 || true' EXIT
for i in $(seq 1 240); do
  n=$(grep -c "Uvicorn running on" "$OUT/env_start.log" 2>/dev/null || true); n=${n:-0}
  [ "$n" -ge 3 ] && break
  kill -0 $SRV 2>/dev/null || { echo "[gate] ❌ gym env start 가 죽었다 — $OUT/env_start.log"; exit 1; }
  sleep 5
done
[ "$(grep -c 'Uvicorn running on' "$OUT/env_start.log")" -ge 3 ] || { echo "[gate] ❌ 서버 3종 기동 대기 초과"; exit 1; }
echo "[gate] servers up (~$((i*5))s)"

.venv/bin/gym eval run --no-serve --agent tau2_alpha_agent \
  --input benchmarks/tau2/data/tau2_benchmark.jsonl \
  --output "$OUT/rollouts.jsonl" --limit "$LIMIT" --num-repeats "$REPEATS" --concurrency "${CONCURRENCY:-4}" \
  2>&1 | tee "$OUT/eval_run.log" | grep -v "^\s*$" | tail -20

"$GATE_PY" "$ALPHA_GYM/smoke/roundtrip_gate.py" --transport "$OUT/vllm_transport.jsonl" \
  --rollouts "$OUT/rollouts.jsonl" --tokenizer "$ALPHA_ROOT/tokenizer_v5" --expect restore --json "$OUT/gate.json" | tee "$OUT/gate.txt"
