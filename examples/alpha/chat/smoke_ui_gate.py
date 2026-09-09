#!/usr/bin/env python3
"""smoke_ui_gate.py — UI 경유 요청이 학습 분포를 지키는지 검사한다 (smoke_chat.sh §6).

무엇을 보는가: 채팅 UI(LibreChat)를 거쳐 vLLM 에 도착한 프롬프트의 **토큰 수**.
  "안녕?" 한 턴의 학습 렌더는 17 토큰이다. UI 가 도구·시스템 프롬프트를 몰래 붙이면 이 수가
  수천으로 뛴다 (OpenWebUI 0.11 사고: 5,440 토큰, docs/KNOWN_ISSUES.md 2026-09-09).
어떻게 재는가: vLLM /metrics 의 `vllm:prompt_tokens_total` 을 요청 전후로 읽어 차이를 본다.
  UI 가 무엇을 보내는지 UI 를 믿지 않고 서버 쪽 카운터로 확인한다. 1인 사용 중에 돌릴 것 —
  동시에 다른 대화가 있으면 차이가 오염된다.
함께 보는 것: 응답이 think 파트와 text 파트로 분리됐는가 (reasoningKey 설정 검증), 본문에
  <think> 잔류가 없는가.
§7 도구 경로 (2026-09-09): 웹검색 토글을 켠 대화에서 ① 프롬프트가 커진다(도구 명세가 선언됨),
  ② vLLM 파서가 XML 호출을 `tool_call` 파트로 구조화하고 LibreChat 이 실행해 결과를 돌려준다,
  ③ 그 뒤 본문이 온다. 세 단계가 모두 지나야 alpha 규약(XML tool_call → <tool_response>)이 산 것이다.
  모델이 도구를 안 부르면 FAIL — 프롬프트가 검색을 명시하므로 안 부르는 것도 이상이다.

사용: python3 chat/smoke_ui_gate.py [LC_URL] [VLLM_BASE]   (기본 http://localhost:8080, http://localhost:8001/v1)
  스모크 계정은 $LC_DATA/smoke_credentials 에 만들어 재사용한다 (LibreChat 은 로그인 필수).
  채팅 라우트는 브라우저 User-Agent 만 받으므로 헤더를 흉내 낸다.
"""
import json
import os
import re
import secrets
import sys
import time
import urllib.error
import urllib.request
import uuid

LC = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080").rstrip("/")
VLLM = (sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8001/v1").rstrip("/")
METRICS = re.sub(r"/v1$", "", VLLM) + "/metrics"
LC_DATA = os.environ.get("LC_DATA", "/home/work/vidsearch/tools/librechat_data")
CRED = os.path.join(LC_DATA, "smoke_credentials")
PROMPT = "안녕?"
TOOL_PROMPT = "web_search 도구로 오늘 코스피(KOSPI) 지수를 검색해서 숫자와 출처를 알려줘."
MAX_PROMPT_TOKENS = 64        # 학습 렌더 17 + 여유. 도구 주입이면 수천.
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    print(f"  {'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""))
    PASS += ok
    FAIL += (not ok)


def http(url, body=None, token=None, stream=False, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json", "User-Agent": UA,
         "Accept": "text/event-stream" if stream else "application/json"}
    if token:
        h["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, headers=h, method="POST" if body is not None else "GET")
    return urllib.request.urlopen(req, timeout=timeout)


def prompt_tokens_total():
    total = 0.0
    for line in http(METRICS, timeout=10).read().decode().splitlines():
        if line.startswith("vllm:prompt_tokens_total"):
            total += float(line.rsplit(" ", 1)[1])
    return total


def credentials():
    if os.path.exists(CRED):
        email, pw = open(CRED).read().split()[:2]
        return email, pw
    email, pw = "smoke@alpha.local", "Smoke-" + secrets.token_hex(12)
    os.makedirs(LC_DATA, exist_ok=True)
    fd = os.open(CRED, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"{email} {pw}\n")
    return email, pw


def login():
    email, pw = credentials()
    try:
        http(LC + "/api/auth/register", {"name": "smoke", "username": "smoke", "email": email,
                                         "password": pw, "confirm_password": pw}).read()
    except urllib.error.HTTPError as e:   # 이미 가입됨(409/400) 은 정상
        if e.code not in (400, 409):
            raise
    return json.load(http(LC + "/api/auth/login", {"email": email, "password": pw}))["token"]


def run_chat(tok, endpoint, model, text, web_search):
    """한 턴을 보내고 (final 이벤트, vLLM prompt_tokens 증분) 을 돌려준다. 실패 시 (None, 0)."""
    before = prompt_tokens_total()
    body = {"text": text, "messageId": str(uuid.uuid4()), "parentMessageId": "00000000-0000-0000-0000-000000000000",
            "conversationId": None, "isCreatedByUser": True, "sender": "User", "endpoint": endpoint,
            "endpointType": "custom", "model": model, "isTemporary": True,
            "ephemeralAgent": {"execute_code": False, "web_search": web_search, "mcp": []}, "timezone": "Asia/Seoul"}
    start = json.load(http(f"{LC}/api/agents/chat/{endpoint}", body, token=tok))
    sid = start.get("streamId")
    check("채팅 요청 수락", bool(sid), f"status={start.get('status')} web_search={web_search}")
    if not sid:
        return None, 0
    final = None
    t0 = time.time()
    for raw in http(f"{LC}/api/agents/chat/stream/{sid}", token=tok, stream=True, timeout=900):
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data:"):
            try:
                d = json.loads(line[5:].strip())
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("final"):
                final = d
    check("final 이벤트 수신", final is not None, f"{time.time() - t0:.0f}s")
    if final is None:
        return None, 0
    return final, prompt_tokens_total() - before


def main():
    print("── 6. UI 경유 프롬프트 게이트 (LibreChat → vLLM) " + "─" * 16)
    try:
        cfg = json.load(http(LC + "/api/config", timeout=10))
    except Exception as e:
        check("LibreChat 응답", False, f"{LC}: {e}")
        return
    check("LibreChat 응답", True, f"{LC} appTitle={cfg.get('appTitle')}")
    tok = login()
    # /api/models 는 미설정 기본 제공자(openAI 등)까지 정적 목록으로 돌려준다 — 설정된 엔드포인트는 /api/endpoints 로 고른다
    endpoints = json.load(http(LC + "/api/endpoints", token=tok))
    endpoint = next((k for k, v in endpoints.items() if isinstance(v, dict) and v.get("type") == "custom"), None)
    models = json.load(http(LC + "/api/models", token=tok))
    model = (models.get(endpoint) or [None])[0] if endpoint else None
    check("커스텀 엔드포인트·모델 노출", bool(model), f"endpoint={endpoint} model={model}")
    if not model:
        return

    final, delta = run_chat(tok, endpoint, model, PROMPT, web_search=False)
    if final is None:
        return
    check(f"프롬프트 토큰 ≤ {MAX_PROMPT_TOKENS} (도구·시스템 프롬프트 미주입)", delta <= MAX_PROMPT_TOKENS,
          f"vllm:prompt_tokens_total 증분 {delta:.0f} (학습 렌더 17)")
    parts = final["responseMessage"].get("content") or []
    think = "".join(p.get("think", "") for p in parts if p.get("type") == "think")
    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    check("think / text 파트 분리", bool(think.strip()) and bool(text.strip()),
          f"parts={[p.get('type') for p in parts]} think={len(think)}자 text={len(text)}자")
    check("본문에 <think> 태그 잔류 없음", "<think>" not in text and "</think>" not in text)
    print(f"\n     [text] {text[:200]!r}")

    print("── 7. 도구 경로 (웹검색 토글 ON → XML tool_call → 실행 → 답변) " + "─" * 4)
    cfg_ws = (json.load(http(LC + "/api/config", token=tok)) or {}).get("webSearch") or {}   # 인증된 요청에만 노출
    check("웹검색 구성 노출 (/api/config webSearch)", bool(cfg_ws), f"{ {k: cfg_ws[k] for k in cfg_ws if 'Provider' in k or 'Type' in k} }")
    final, delta = run_chat(tok, endpoint, model, TOOL_PROMPT, web_search=True)
    if final is None:
        return
    check("프롬프트가 도구 명세만큼 커짐 (tool 시나리오 렌더)", delta > MAX_PROMPT_TOKENS,
          f"증분 {delta:.0f} 토큰 (누적: 선언 + 호출 후 재호출)")
    parts = final["responseMessage"].get("content") or []
    calls = [p for p in parts if p.get("type") == "tool_call"]
    names = [(p.get("tool_call") or {}).get("name") for p in calls]
    outputs = [(p.get("tool_call") or {}).get("output") for p in calls]
    check("tool_call 파트 구조화 (vLLM qwen3_xml 파서)", bool(calls), f"parts={[p.get('type') for p in parts]} names={names}")
    check("도구 실행 결과 존재 (LibreChat 실행 → Tavily)", any(o for o in outputs), f"output_len={[len(str(o or '')) for o in outputs]}")
    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    check("도구 결과 뒤 본문 생성", bool(text.strip()), f"text={len(text)}자")
    check("본문에 XML 원문 누수 없음", "<tool_call>" not in text and "<function=" not in text and "<tool_response>" not in text)
    args = (calls[0].get("tool_call") or {}).get("args") if calls else None
    print(f"\n     [tool args] {str(args)[:160]}\n     [text] {text[:220]!r}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # 네트워크·인증 실패도 FAIL 로 센다 — 침묵 스킵 금지
        check("게이트 실행", False, repr(e)[:300])
    print(f"  ui-gate: {PASS} PASS / {FAIL} FAIL")
    sys.exit(1 if FAIL else 0)
