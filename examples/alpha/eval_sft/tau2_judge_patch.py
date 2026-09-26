#!/usr/bin/env python3
"""tau2-bench vendored 패치 — NL-assertion 판정기 env 오버라이드 (2026-09-27, 사용자 결정 gemini-3.7-flash).

upstream tau2 v1.0.1 은 `DEFAULT_LLM_NL_ASSERTIONS = "gpt-4.1-2025-04-14"` 를 config.py 에 하드코딩해 CLI/env 로 바꿀 수 없다.
우리 환경은 OPENAI_API_BASE 가 tau_proxy 라 그 요청이 vLLM 에 가서 404 → retail NL 과제 40개가 전부 infrastructure_error
(KNOWN_ISSUES 2026-09-18). 이 스크립트가 config.py 에 두 env 를 읽는 코드를 넣는다(멱등, 마커 확인):
  TAU2_LLM_NL_ASSERTIONS            litellm 모델 id (run_tau.sh 기본 gemini/gemini-3.7-flash)
  TAU2_LLM_NL_ASSERTIONS_EXTRA_ARGS JSON dict → 판정 호출 kwargs 에 병합. Gemini 는 response_format=json_object 가 없으면
                                    ```json 펜스를 씌워 json.loads 가 실패한다(2026-09-27 스모크 실측).
사용: python3 tau2_judge_patch.py [<tau2-bench 루트>]   (install_tau2.sh 가 checkout 직후 호출)
"""
import sys, os
root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
p = os.path.join(root, "src/tau2/config.py")
MARK = "# [alpha-patch] NL-assertion judge override"
ANCHOR = 'DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE}\n'
ADD = ANCHOR + MARK + " (examples/alpha/eval_sft/tau2_judge_patch.py 가 적용; 근거는 그 파일 docstring)\n" + \
      "import json as _json, os as _os\n" + \
      'DEFAULT_LLM_NL_ASSERTIONS = _os.environ.get("TAU2_LLM_NL_ASSERTIONS", DEFAULT_LLM_NL_ASSERTIONS)\n' + \
      "DEFAULT_LLM_NL_ASSERTIONS_ARGS = {**DEFAULT_LLM_NL_ASSERTIONS_ARGS,\n" + \
      '                                  **_json.loads(_os.environ.get("TAU2_LLM_NL_ASSERTIONS_EXTRA_ARGS", "{}"))}\n'
s = open(p).read()
if MARK in s:
    print(f"   {p}: 판정기 오버라이드 이미 적용"); sys.exit(0)
if s.count(ANCHOR) != 1:
    print(f"   ❌ {p}: 패치 앵커를 못 찾음 — upstream 변경 확인"); sys.exit(1)
open(p, "w").write(s.replace(ANCHOR, ADD, 1)); print(f"   {p}: 판정기 오버라이드 적용")
