#!/usr/bin/env python3
"""τ³ preflight T3 — NL-assertion 판정기 1회 호출: 모델 도달 + JSON 파싱(펜스 없음)까지 확인 (2026-09-27).
실패하면 retail NL 과제 40개의 채점이 전부 infrastructure_error 가 되므로 run_tau.sh 가 여기서 중단한다.
tau2 venv python 으로, TAU2_HOME 에서 실행 (run_tau.sh 가 호출). exit 0 = OK."""
import json, sys
from tau2.config import DEFAULT_LLM_NL_ASSERTIONS as M, DEFAULT_LLM_NL_ASSERTIONS_ARGS as A
from tau2.utils.llm_utils import generate
from tau2.data_model.message import SystemMessage, UserMessage
msgs = [SystemMessage(role="system", content='You are a judge. Return ONLY JSON: {"results":[{"expectedOutcome":"...","reasoning":"...","metExpectation":true}]}'),
        UserMessage(role="user", content="conversation:\nuser: cancel order 123\nassistant: Order 123 is cancelled.\n\nexpectedOutcomes:\n- Agent confirms the cancellation.\n- Agent offers a discount coupon.")]
r = generate(model=M, messages=msgs, call_name="preflight_judge", **A)
c = r.content or ""
try:
    d = json.loads(c)
except Exception as e:
    print(f"[tau] ❌ T3 판정기 {M}: JSON 파싱 실패 ({e}) — 응답 머리: {c[:160]!r}"); sys.exit(1)
res = d.get("results", [])
if len(res) != 2 or not isinstance(res[0].get("metExpectation"), bool):
    print(f"[tau] ❌ T3 판정기 {M}: 형식 불일치 — {json.dumps(d)[:200]}"); sys.exit(1)
print(f"[tau] T3 판정기 {M} OK (args={A}, met={[x['metExpectation'] for x in res]})")
