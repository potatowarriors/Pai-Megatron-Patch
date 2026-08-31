#!/bin/bash
# smoke_chat.sh — 채팅 fleet 검증. G1·G3 와 별개로 "UI 가 쓸 수 있는 형태인가"를 본다.
#   1. /v1/models 200 · 모델명 노출
#   2. 한 턴 대화가 finish_reason=stop 으로 끝나고 content 가 비어있지 않다
#   3. reasoning 파서가 <think> 를 `reasoning` 필드로 분리했다 (본문에 태그 잔류 없음)
#   4. 멀티턴에서 히스토리 렌더가 깨지지 않는다
# 사용: bash chat/smoke_chat.sh [BASE_URL]
set -uo pipefail
BASE="${1:-http://localhost:8001/v1}"
python3 - "$BASE" <<'PY'
import json, sys, urllib.request

base = sys.argv[1]
PASS = FAIL = 0

def check(name, ok, detail=""):
    global PASS, FAIL
    print(f"  {'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""))
    if ok: PASS += 1
    else:  FAIL += 1

def post(path, body, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

print("── 1. 엔드포인트 " + "─" * 45)
models = json.load(urllib.request.urlopen(base + "/models", timeout=10))
ids = [m["id"] for m in models["data"]]
check("/v1/models 응답", bool(ids), f"served={ids}")
MODEL = ids[0]

print("── 2. 한 턴 대화 " + "─" * 45)
r = post("/chat/completions", {
    "model": MODEL,
    "messages": [{"role": "user", "content": "자기소개를 두 문장으로 해줘."}],
    "max_tokens": 2048, "temperature": 0.6, "top_p": 0.95,
})
ch = r["choices"][0]
msg, finish = ch["message"], ch["finish_reason"]
content = (msg.get("content") or "")
# vLLM 0.25.1 은 `reasoning` 으로 내보낸다 (구버전 이름은 reasoning_content).
reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "")
check("finish_reason=stop", finish == "stop", f"finish={finish}")
check("content 비어있지 않음", len(content.strip()) > 0, f"{len(content)}자")
print(f"\n     [reasoning] {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")
print(f"     [content]   {content[:300]}{'...' if len(content) > 300 else ''}\n")

print("── 3. reasoning 분리 " + "─" * 41)
check("reasoning 필드 분리", len(reasoning.strip()) > 0, f"{len(reasoning)}자")
check("본문에 <think> 태그 잔류 없음",
      "<think>" not in content and "</think>" not in content)

print("── 4. 멀티턴 " + "─" * 49)
r2 = post("/chat/completions", {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "내 이름은 준호야. 기억해."},
        {"role": "assistant", "content": "네, 준호님이라고 기억하겠습니다."},
        {"role": "user", "content": "내 이름이 뭐라고 했지?"},
    ],
    "max_tokens": 1024, "temperature": 0.6,
})
c2 = (r2["choices"][0]["message"].get("content") or "")
check("멀티턴 응답 생성", len(c2.strip()) > 0, f"{len(c2)}자")
check("히스토리 참조 성공", "준호" in c2, c2[:120].replace("\n", " "))

print("\n" + "=" * 60)
print(f"  smoke: {PASS} PASS / {FAIL} FAIL")
sys.exit(1 if FAIL else 0)
PY
