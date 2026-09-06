r"""벤치 투입 전 게이트 G1~G3 — `docs/SFT_BENCHMARKS.md` §7.

2026-08-30 사고(벤치 전량 무효)의 재발 방지. **셋 다 통과해야 수치를 기록한다.**

| 게이트 | 무엇을 막는가 |
|---|---|
| G1 변환 산출물 | `generation_config.json` 부재 → eos 에 `<\|im_end\|>` 가 없어 턴 종료 시 안 멈춤 |
| G2 태그 관측 | `</think>` 가 출력에서 사라짐 → 사고/답변 구간을 가를 수 없음 |
| G3 서빙 스모크 | 실제 생성이 `finish_reason=stop` 으로 끝나고 답변이 비어있지 않은지 |

G2 는 체크포인트의 tokenizer 를 고쳐서 푸는 문제가 아니다. NVIDIA 공식 설정
(Nemotron 3 Ultra eval yaml)은 요청에 `skip_special_tokens: false` 를 넣어 해결한다 —
우리 태스크 yaml 도 같은 방식이다. 여기서는 그 요청이 실제로 먹히는지 **실측**한다.

사용:
    python3 check_gates.py --hf-dir <HF_CKPT>                      # G1 만
    python3 check_gates.py --base-url http://localhost:8100/v1     # G2·G3 만
    python3 check_gates.py --hf-dir <HF_CKPT> --base-url <URL>     # 전부
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# 쉬운 질문 — 반드시 짧게 끝나야 한다. 여기서 length 로 끝나면 종료가 깨진 것.
EASY = "What is 17*23? Answer briefly."
# 어려운 질문 — 모델 성숙도 진단용. 실패해도 게이트를 막지는 않는다.
HARD = (
    "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b$.\n\n"
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def post(base_url: str, body: dict, timeout: int = 1800) -> dict:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# G2 표본 수. 태그가 살아 있으면 보통 1회에 끝나고, 안 보일 때만 더 뽑는다.
G2_SAMPLES = 6


def _gen(base_url: str, prompt: str, max_tokens: int) -> dict:
    """태스크 yaml 과 **동일한** 파라미터로 생성한다 — 게이트가 실제 조건을 재야 한다."""
    return post(base_url, {
        "model": "alpha",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "seed": None,
        "skip_special_tokens": False,
    })


def gate_g1(hf_dir: Path) -> tuple[bool, list[str]]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from emit_generation_config import _added_tokens, check  # noqa: PLC0415

    problems = check(hf_dir)
    if problems:
        return False, problems
    gen = json.loads((hf_dir / "generation_config.json").read_text())
    named = {v: k for k, v in _added_tokens(hf_dir).items()}
    eos = gen["eos_token_id"]
    pretty = ", ".join(f"{i}={named.get(i, '?')}" for i in eos)
    return True, [f"eos: {pretty}"]


def gate_g23(base_url: str, max_tokens: int) -> tuple[bool, bool, list[str]]:
    """(G2 통과, G3 통과, 로그)."""
    log: list[str] = []

    t0 = time.time()
    d = _gen(base_url, EASY, max_tokens)
    ch = d["choices"][0]
    msg = ch["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    full = reasoning + content
    finish = ch["finish_reason"]
    used = d["usage"]["completion_tokens"]
    log.append(
        f"쉬운 질문: finish={finish} tokens={used} t={time.time() - t0:.0f}s "
        f"content={len(content)}자 reasoning={len(reasoning)}자"
    )
    log.append(f"  content: {content.strip()[:160]!r}")

    # G2 — </think> 가 응답 어딘가에 살아있는가.
    #      reasoning parser 를 켰다면 파서가 이미 소비했을 수 있으므로,
    #      reasoning_content 가 채워진 것도 관측 성공으로 본다.
    #
    # **표본을 여러 번 뽑는다 (2026-09-06).** 온도 1.0 에서 모델은 쉬운 질문에
    # 가끔 태그 없이 바로 답한다 — 실측 19/20(95%). 단일 표본으로 판정하면 5% 가
    # 오탐이고, 스위트당 fleet 재기동이 3회이므로 한 번이라도 헛걸릴 확률이 14% 다.
    # 실제로 iter1500 이 이 오탐으로 T2 직전에 중단됐고 후속 체인까지 막혔다.
    #
    # 이 게이트가 잡아야 하는 진짜 고장은 태그가 **아예 사라지는** 것이다
    # (tokenizer/서빙 오설정 → 0%). 그래서 표본 k 개 중 하나라도 관측되면 통과로
    # 본다 — 정상(95%)에서 k=6 이면 오탐 확률 0.95^-... ≈ 1.6e-8, 고장(0%)이면 항상 잡힌다.
    seen = 1 if (("</think>" in full) or bool(reasoning)) else 0
    tried = 1
    while seen == 0 and tried < G2_SAMPLES:
        tried += 1
        dx = _gen(base_url, EASY, max_tokens)
        mx = dx["choices"][0]["message"]
        if ("</think>" in ((mx.get("reasoning_content") or "") + (mx.get("content") or ""))
                or mx.get("reasoning_content")):
            seen = 1
    g2 = seen > 0
    log.append(
        f"G2 태그 관측: {'OK' if g2 else 'FAIL'} "
        f"({tried}회 표본 중 {seen}회 관측, reasoning_content={'있음' if reasoning else '없음'})"
    )

    # G3 — 종료·비어있지 않은 답변
    g3 = finish == "stop" and bool(content.strip())
    if finish != "stop":
        log.append(f"G3 실패 원인: finish_reason={finish} (쉬운 질문이 예산을 소진 — 종료 미작동)")
    elif not content.strip():
        log.append("G3 실패 원인: content 가 비어 있음 (파서가 답변을 통째로 가져갔을 수 있음)")

    # 진단(게이트 아님) — 어려운 질문에서 사고를 닫는가.
    try:
        t0 = time.time()
        d2 = _gen(base_url, HARD, max_tokens)
        c2 = d2["choices"][0]
        m2 = c2["message"]
        cont2 = m2.get("content") or ""
        rsn2 = m2.get("reasoning_content") or ""
        log.append(
            f"[진단] 어려운 질문: finish={c2['finish_reason']} "
            f"tokens={d2['usage']['completion_tokens']} t={time.time() - t0:.0f}s "
            f"content={len(cont2)}자 reasoning={len(rsn2)}자"
        )
        if c2["finish_reason"] == "length" and not cont2.strip():
            log.append(
                "[진단] 어려운 문제에서 답변에 도달하지 못했다 — 설정이 아니라 "
                "**모델 미성숙** 신호. 벤치는 돌릴 수 있으나 낮은 점수를 예상할 것."
            )
    except Exception as e:  # noqa: BLE001
        log.append(f"[진단] 어려운 질문 실패(게이트 아님): {type(e).__name__}: {e}")

    return g2, g3, log


def main() -> int:
    ap = argparse.ArgumentParser(description="벤치 투입 전 게이트 G1~G3")
    ap.add_argument("--hf-dir", type=Path, help="G1: 변환된 HF 체크포인트 디렉토리")
    ap.add_argument("--base-url", help="G2·G3: OpenAI 호환 엔드포인트 (예: http://localhost:8100/v1)")
    ap.add_argument("--max-tokens", type=int, default=32768)
    a = ap.parse_args()

    if not a.hf_dir and not a.base_url:
        ap.error("--hf-dir 또는 --base-url 중 하나는 필요하다")

    results: dict[str, bool] = {}

    if a.hf_dir:
        print("── G1: 변환 산출물 eos 정합성 " + "─" * 34)
        ok, log = gate_g1(a.hf_dir)
        for line in log:
            print(f"   {line}")
        print(f"   → {'PASS' if ok else 'FAIL'}\n")
        results["G1"] = ok

    if a.base_url:
        print("── G2·G3: 서빙 스모크 " + "─" * 40)
        try:
            g2, g3, log = gate_g23(a.base_url, a.max_tokens)
        except (urllib.error.URLError, OSError, KeyError) as e:
            print(f"   ❌ 엔드포인트 호출 실패: {type(e).__name__}: {e}")
            return 1
        for line in log:
            print(f"   {line}")
        print(f"   → G2 {'PASS' if g2 else 'FAIL'} / G3 {'PASS' if g3 else 'FAIL'}\n")
        results["G2"], results["G3"] = g2, g3

    bad = [k for k, v in results.items() if not v]
    checked = "·".join(sorted(results))
    if bad:
        print(f"❌ 게이트 실패: {', '.join(sorted(bad))} — 벤치를 돌리지 말 것. "
              "수치를 TRACKING.md 에 기록하는 것도 금지.")
        return 1
    print(f"✅ 게이트 통과 ({checked}) — 벤치 투입 가능")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
