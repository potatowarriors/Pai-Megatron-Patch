"""T3 판정 러너 공용 생성 헬퍼 — AA/Nemotron 규약 (2026-08-30).

`docs/SFT_BENCHMARKS.md` §3.4 와 **같은 파라미터**를 쓴다. 러너마다 값을 따로 박아두면
T1 과 조건이 어긋나 비교가 깨진다 (2026-08-30 사고: 러너·태스크·CLI 세 곳이 서로 다른
값을 갖고 있어 어느 설정이 적용됐는지 사후 추적이 불가능했다).
"""

from __future__ import annotations

import json
import time
import urllib.request

# §3.4 정본. 온도 1.0 = Nemotron 3 Ultra (사용자 확정 2026-08-30).
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_TOKENS = 32768
SKIP_SPECIAL_TOKENS = False   # </think> 가 출력에 살아남아야 답변부를 가를 수 있다


def chat(base_url: str, messages: list[dict], max_tokens: int = MAX_TOKENS,
         timeout: int = 1800, retries: int = 3) -> str:
    """OpenAI 호환 chat/completions 1회. 실패 시 지수 백오프."""
    body = json.dumps({
        "model": "alpha",
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": max_tokens,
        "seed": None,                              # 고정 시드는 반복을 동일 표본으로 만든다
        "skip_special_tokens": SKIP_SPECIAL_TOKENS,
    }).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["choices"][0]["message"].get("content") or ""
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                return ""
            time.sleep(2 ** attempt)
    return ""


def split_think(text: str) -> tuple[str, bool]:
    """(답변부, 사고를 닫았는지). 심판에는 답변부만 넘긴다.

    `</think>` 가 없다 = 모델이 답변에 도달하지 못했다. 그 비율(`think_closed`)을
    함께 보고해야 "모델이 틀렸다"와 "측정이 성립하지 않았다"를 구분할 수 있다.
    """
    if not text:
        return "", False
    i = text.rfind("</think>")
    if i < 0:
        return text.strip(), False
    return text[i + len("</think>"):].strip(), True
