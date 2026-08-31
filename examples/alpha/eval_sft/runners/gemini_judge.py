"""Gemini judge (provider-agnostic surface) for SFT benchmarks.

Judge model: gemini-3.7-flash (Google Generative Language API v1beta).
Key: examples/alpha/.env  GEMINI_API_KEY  (gitignored).

Kept deliberately small and dependency-free (urllib only) so it runs in any
python on sub1/main1 without extra installs. Concurrency via threads.
"""
from __future__ import annotations
import json, os, time, urllib.request, urllib.error, concurrent.futures as cf
from pathlib import Path

DEFAULT_MODEL = "gemini-3.7-flash"
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


def load_key(env_path: str | None = None) -> str:
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"].strip()
    p = Path(env_path or Path(__file__).resolve().parents[2] / ".env")
    for line in p.read_text().splitlines():
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("GEMINI_API_KEY not found (env or examples/alpha/.env)")


# judge 는 **추론 모델**이다(gemini-3.7-flash). 토큰 예산이 작으면 내부 사고에 전부
# 소모하고 텍스트를 하나도 내놓지 않는다 — 2026-08-31 실측: 8토큰 → '', 64토큰 → 'A'.
# 그 빈 응답이 호출부에서 등급으로 흡수되어 SimpleQA 1000건 중 761건이 NOT_ATTEMPTED 로
# 잘못 찍혔다. 호출부가 작은 값을 넘겨도 여기서 바닥을 강제한다.
MIN_OUTPUT_TOKENS = 64


def judge_one(prompt: str, key: str, model: str = DEFAULT_MODEL,
              temperature: float = 0.0, max_tokens: int = 4000,
              retries: int = 4) -> str:
    if max_tokens < MIN_OUTPUT_TOKENS:
        max_tokens = MIN_OUTPUT_TOKENS
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }).encode()
    url = API.format(model=model, key=key)
    last = ""
    for a in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read())
            cand = d.get("candidates", [{}])[0]
            parts = cand.get("content", {}).get("parts", [{}])
            return "".join(p.get("text", "") for p in parts).strip()
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
            if e.code in (429, 500, 503):
                time.sleep(2 ** a); continue
            return f"__JUDGE_ERROR__ {last}"
        except Exception as e:  # noqa: BLE001
            last = str(e); time.sleep(2 ** a)
    return f"__JUDGE_ERROR__ {last}"


def judge_batch(prompts: list[str], key: str | None = None, model: str = DEFAULT_MODEL,
                workers: int = 8, **kw) -> list[str]:
    key = key or load_key()
    out: list[str] = [""] * len(prompts)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(judge_one, p, key, model, **kw): i for i, p in enumerate(prompts)}
        for f in cf.as_completed(futs):
            out[futs[f]] = f.result()
    return out


if __name__ == "__main__":
    k = load_key()
    r = judge_batch(["Reply with exactly: A", "What is 2+2? Reply with only the number."], k, workers=2)
    print("SELFTEST:", r)
