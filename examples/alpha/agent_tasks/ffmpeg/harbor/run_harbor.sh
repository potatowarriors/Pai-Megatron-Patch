#!/bin/bash
# run_harbor.sh — score a model on the ffmpeg tasks with Harbor + Terminus-2 (same agent as TB-2).
#
# Prerequisites (README "gpu06 실행"):  tasks synced to /opt/ffbench/<DATASET> · image alpha-ffmpeg-base:1 ·
# oracle 50/50 and nop 0/50 on that dataset · harbor/tunnel.sh running for this model.
#
#   bash harbor/run_harbor.sh <RUN_NAME> <SERVED_MODEL_NAME> [K=4] [W=12]
#   env: DATASET=pilot_v0  RPORT=8299  MAX_TOKENS=8192  INCLUDE=<glob>  (INCLUDE = smoke subset, not a valid score)
#
# Alpha needs the reasoning-restore path of eval_sft/run_terminal_tb2.sh (tau_proxy + interleaved_thinking);
# this runner talks to the endpoint directly, which is right for models without a reasoning parser.
set -uo pipefail
RUN_NAME="${1:?run name}"; MODEL="${2:?served model name}"; K="${3:-4}"; W="${4:-12}"
DATASET="${DATASET:-pilot_v0}"; RPORT="${RPORT:-8299}"; MAX_TOKENS="${MAX_TOKENS:-8192}"
HERE="$(cd "$(dirname "$0")" && pwd)"; SSHC="/home/work/vidsearch/.ssh-keys/config"
OUT="$HERE/../results/$RUN_NAME"; mkdir -p "$OUT"
# compose project names are length-limited — keep the job name short (eval_sft 2026-08-30)
RID="ffb$(echo "$RUN_NAME" | md5sum | cut -c1-8)"
FILTER=""; [ -n "${INCLUDE:-}" ] && FILTER="-i ${INCLUDE}" && echo "[ffb] ⚠️ subset '${INCLUDE}' — not a valid score"

ssh -F "$SSHC" -o BatchMode=yes alpha-eval 'bash -s' <<EOF
  export HOME=/opt/harbor OPENAI_API_KEY=dummy OPENAI_API_BASE=http://localhost:$RPORT/v1
  curl -s -m 10 -o /dev/null http://localhost:$RPORT/v1/models || { echo "[ffb] ❌ no endpoint on container:$RPORT — start harbor/tunnel.sh"; exit 1; }
  # litellm dies computing cost for an unregistered model (eval_sft 2026-08-30)
  cat > /opt/ffbench/model_registry.json <<JSON
{"openai/$MODEL": {"max_tokens": $MAX_TOKENS, "max_input_tokens": 262144, "max_output_tokens": $MAX_TOKENS,
  "input_cost_per_token": 0, "output_cost_per_token": 0, "litellm_provider": "openai", "mode": "chat"}}
JSON
  export LITELLM_MODEL_REGISTRY_PATH=/opt/ffbench/model_registry.json
  cd /opt/harbor && rm -rf /opt/ffbench/jobs/$RID
  ./venv/bin/harbor run -p /opt/ffbench/$DATASET -a terminus-2 -m openai/$MODEL \
    --ak api_base=http://localhost:$RPORT/v1 --ak temperature=1.0 --ak parser_name=json \
    --ak 'llm_call_kwargs={"top_p":0.95,"max_tokens":$MAX_TOKENS}' \
    $FILTER -n $W -k $K -o /opt/ffbench/jobs --job-name $RID -y -q 2>&1 | tail -25
EOF

# per-trial rewards + failed checks -> local results
ssh -F "$SSHC" -o BatchMode=yes alpha-eval "JOB=/opt/ffbench/jobs/$RID python3 -" > "$OUT/trials.json" <<'PY'
import glob, json, os
rows = []
for f in sorted(glob.glob(os.path.join(os.environ["JOB"], "*", "result.json"))):
    d = json.load(open(f)); t = os.path.dirname(f)
    det = os.path.join(t, "verifier", "details.json")
    failed = [f"{r['type']}: {r['detail']}" for r in json.load(open(det))["results"] if not r["ok"]] if os.path.exists(det) else None
    rows.append({"task": d.get("task_name"), "trial": os.path.basename(t),
                 "reward": ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
                 "exception": (d.get("exception_info") or {}).get("exception_type"), "failed_checks": failed})
print(json.dumps(rows, ensure_ascii=False, indent=1))
PY
python3 - "$OUT/trials.json" <<'PY'
import json, sys, collections
rows = json.load(open(sys.argv[1])); by = collections.defaultdict(list)
for r in rows:
    by[r["task"].split("-")[0]].append(1 if r["reward"] == 1 else 0)
for lv in sorted(by):
    print(f"  {lv}: {sum(by[lv])}/{len(by[lv])} = {100 * sum(by[lv]) / len(by[lv]):.1f}%")
n = sum(map(len, by.values())); s = sum(map(sum, by.values()))
exc = collections.Counter(r["exception"] for r in rows if r["exception"])
print(f"  all: {s}/{n} = {100 * s / max(n, 1):.1f}%   exceptions: {dict(exc)}")
PY
