#!/bin/bash
# tau_smoke.sh — τ³-bench 온보딩 스모크 + 복원 ON/OFF differential (docs/SFT_BENCHMARKS.md §3.13 검증 (b)·(c)).
#   sub1 에서 실행. TOOLS=1 fleet 를 직접 띄우고(기본 GPU 0~3, 262144) 끝나면 내린다.
#   1) A1·A4 게이트 → 2) airline N_TASKS×1 trial W=2 스모크 → 3) 결과 검사(도구 호출 파싱·think 없는 발화·프록시
#   miss 0·reward 존재·렌더 보존) → 4) airline 앞 DIFF_N 과제를 복원 ON/OFF 로 각 1 trial → differential.json
#   모든 런은 부분 표본이라 결과는 무효 표기된다(집계 대상 아님).
# 사용: bash eval_sft/tau_smoke.sh <HF_CKPT> [TAG=tau_smoke]   env: GPUS(0,1,2,3) MAX_LEN(262144) N_TASKS(2) DIFF_N(5)
set -uo pipefail
CKPT="${1:?HF ckpt}"; TAG="${2:-tau_smoke}"
HERE="$(cd "$(dirname "$0")" && pwd)"
GPUS="${GPUS:-0,1,2,3}"; NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l); PROXY=8100; BURL="http://localhost:$PROXY/v1"
LOGD=/home/work/vidsearch/tools/bench_logs; mkdir -p "$LOGD"
TAU2_HOME="${TAU2_HOME:-/home/work/vidsearch/tools/tau2-bench}"; PY="$TAU2_HOME/.venv/bin/python"
export PIP_CONSTRAINT=
log() { echo "[tau-smoke $(date +%H:%M:%S)] $*"; }

bash "$HERE/suite_running.sh" && { log "❌ 벤치 스위트가 돌고 있다 — 중단"; exit 1; }
pgrep -f "alpha_serve_venv/bin/[v]llm" >/dev/null && { log "❌ vLLM 이 이미 떠 있다 — 중단"; exit 1; }

log "fleet 기동 TOOLS=1 GPUS=$GPUS max_len=${MAX_LEN:-262144} ckpt=$CKPT"
( export TOOLS=1 GPUS="$GPUS"; setsid bash "$HERE/serve_fleet.sh" "$CKPT" "${MAX_LEN:-262144}" "$NGPU" "$PROXY" \
    > "$LOGD/fleet_${TAG}.log" 2>&1 < /dev/null & )
trap 'log "fleet 종료"; bash "$HERE/stop_fleet.sh" "$GPUS" >/dev/null 2>&1 || true' EXIT
ready=0
for i in $(seq 1 60); do
  c=0; for g in $(echo "$GPUS" | tr ',' ' '); do
    [ "$(curl -s -o /dev/null -w %{http_code} --max-time 3 http://localhost:$((8000+g))/v1/models)" = "200" ] && c=$((c+1)); done
  [ "$c" -eq "$NGPU" ] && { ready=1; break; }; sleep 20
done
[ "$ready" = 1 ] || { log "❌ fleet 준비 실패 ($LOGD/fleet_${TAG}.log)"; exit 1; }
log "fleet 준비 $NGPU/$NGPU"
python3 "$HERE/check_agentic_gates.py" --base-url "$BURL" --skip-container || { log "❌ A1/A4 실패"; exit 1; }

# ── 1) 스모크 ──────────────────────────────────────────────────────────────────
log "=== 스모크: airline ${N_TASKS:-2} tasks × 1 trial, W=2 ==="
TAU_DOMAINS=airline SKIP_GATES=1 BASE_URL="$BURL" TAU_FRESH=1 bash "$HERE/run_tau.sh" "$TAG" "${N_TASKS:-2}" 1 2
RAW="$HERE/results/$TAG/tau_raw"
log "=== 스모크 검사 ==="
python3 - "$RAW" <<'PY'
import json, os, sys
raw = sys.argv[1]; fails = []
def chk(cond, msg): print(("  ✅ " if cond else "  ❌ ") + msg); (None if cond else fails.append(msg))
R = json.load(open(os.path.join(raw, "airline", "results.json")))
sims = R["simulations"]; msgs = [m for s in sims for m in (s.get("messages") or [])]
asst = [m for m in msgs if m.get("role") == "assistant"]
n_tc = sum(1 for m in asst if m.get("tool_calls"))
chk(len(sims) > 0, f"시뮬레이션 {len(sims)}개")
chk(n_tc > 0, f"assistant 도구 호출 턴 {n_tc}개 (qwen3_xml 파싱·프록시 통과)")
leak = [m for m in asst if isinstance(m.get("content"), str) and ("<think>" in m["content"] or "</think>" in m["content"])]
chk(not leak, f"assistant content 에 <think> 누출 0 (실제 {len(leak)})")
rw = [s for s in sims if (s.get("reward_info") or {}).get("reward") is not None]
chk(len(rw) > 0, f"reward 채점된 sim {len(rw)}/{len(sims)}; 값 {[ (s['reward_info']['reward']) for s in rw ]}")
term = {}; [term.__setitem__(s["termination_reason"], term.get(s["termination_reason"], 0) + 1) for s in sims]
chk(all(k in ("user_stop", "agent_stop", "max_steps") for k in term), f"종료 사유 {term}")
a = json.load(open(os.path.join(raw, "proxy_airline_before.json"))); b = json.load(open(os.path.join(raw, "proxy_airline_after.json")))
d = {k: b.get(k, 0) - a.get(k, 0) for k in ("requests", "reinlined", "miss", "miss_first_assistant", "think_stripped", "think_unclosed", "tool_calls", "mixed_content_and_tools", "upstream_errors", "seed_stripped", "sst_forced")}
print("  proxy delta:", d)
chk(d["upstream_errors"] == 0, "프록시 upstream 오류 0")
chk(d["think_stripped"] > 0, "think 분리 발생 (skip_special_tokens=false 도달)")
chk(d["miss"] == 0, f"히스토리 복원 miss 0 (reinlined {d['reinlined']})")
chk(d["miss_first_assistant"] == d["requests"], "합성 인사 = 요청당 1회")
detail = json.load(open(os.path.join(raw, "..", "results_tau.json")))["tau_detail"]
chk(detail["subsampled"] is True and detail["skipped_domains"] == {}, "부분 표본 무효 표기, 스킵 도메인 없음(airline)")
print("SMOKE_FAILS=" + str(len(fails)))
PY
if [ -f "$RAW/last_request.json" ]; then python3 "$HERE/tau_render_check.py" "$RAW/last_request.json" || log "⚠️ 렌더 검사 실패"; else log "⚠️ last_request.json 없음 (assistant<3 대화)"; fi

# ── 2) differential: 복원 ON vs OFF ────────────────────────────────────────────
IDS=$(PIP_CONSTRAINT= "$PY" - "${DIFF_N:-5}" <<'PY' 2>/dev/null | tail -1
import sys, logging; logging.disable(logging.CRITICAL)
from loguru import logger; logger.remove()
from tau2.registry import registry
print(" ".join(t.id for t in registry.get_tasks_loader("airline")("base")[:int(sys.argv[1])]))
PY
)
log "=== differential: airline task-ids [$IDS] × 1 trial, W=${DIFF_N:-5} ==="
for mode in on off; do
  r=1; [ "$mode" = off ] && r=0
  TAU_DOMAINS=airline TAU_TASK_IDS="$IDS" TAU_REATTACH=$r SKIP_GATES=1 BASE_URL="$BURL" TAU_FRESH=1 \
    bash "$HERE/run_tau.sh" "${TAG}_$mode" 0 1 "${DIFF_N:-5}"
done
python3 - "$HERE/results" "$TAG" <<'PY'
import json, os, sys
root, tag = sys.argv[1], sys.argv[2]; out = {}
for mode in ("on", "off"):
    p = os.path.join(root, f"{tag}_{mode}", "tau_raw", "airline", "results.json")
    R = json.load(open(p)); sims = R["simulations"]
    ptok = ctok = 0; turns = []; rewards = []; term = {}
    for s in sims:
        ms = s.get("messages") or []
        turns.append(sum(1 for m in ms if m.get("role") == "assistant"))
        for m in ms:
            u = m.get("usage") or {}
            if m.get("role") == "assistant" and u:
                ptok += int(u.get("prompt_tokens") or 0); ctok += int(u.get("completion_tokens") or 0)
        rewards.append((s.get("reward_info") or {}).get("reward")); term[s["termination_reason"]] = term.get(s["termination_reason"], 0) + 1
    pr = json.load(open(os.path.join(root, f"{tag}_{mode}", "tau_raw", "proxy_airline_after.json")))
    out[mode] = {"n_sims": len(sims), "agent_prompt_tokens": ptok, "agent_completion_tokens": ctok,
                 "avg_agent_turns": sum(turns) / len(turns) if turns else None, "rewards": rewards, "term": term,
                 "proxy": {k: pr.get(k) for k in ("reinlined", "miss", "think_stripped", "think_unclosed", "mixed_content_and_tools", "tool_calls")}}
json.dump(out, open(os.path.join(root, tag, "differential.json"), "w"), indent=2)
for m, d in out.items():
    print(f"  [{m:3s}] sims={d['n_sims']} prompt_tok={d['agent_prompt_tokens']} compl_tok={d['agent_completion_tokens']} turns={d['avg_agent_turns']} "
          f"rewards={d['rewards']} term={d['term']} proxy={d['proxy']}")
print(f"  → {os.path.join(root, tag, 'differential.json')}")
PY
log "완료 — results/$TAG (스모크), results/${TAG}_on|_off (differential)"
