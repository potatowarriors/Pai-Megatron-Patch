#!/usr/bin/env python3
"""tau_render_check.py — 프록시가 복원한 요청을 tokenizer_v5 템플릿으로 렌더해 히스토리 think 보존을 검사한다 (게이트 T1 의 렌더 판).

입력: tau_proxy --dump-dir 의 last_request.json (assistant ≥3 인 최신 요청, tools 포함).
검사: 히스토리 assistant 턴이 `<think>\n…</think>` 로 렌더되는가(preserved). 첫 assistant(tau2 합성 인사)는 think 가
없으니 empty-think 가 정상. 생성 프롬프트가 `<|im_start|>assistant\n<think>\n` 로 끝나는가. 복원 OFF 대비 토큰 차(=think 분량).

사용: python3 eval_sft/tau_render_check.py <last_request.json> [--tokenizer examples/alpha/tokenizer_v5]
종료 코드 1 = 보존된 턴이 하나도 없거나 'other' 형태가 있음.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tau_proxy import split_think  # noqa: E402


def classify(text: str) -> list[str]:
    asst = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, flags=re.S)
    out = []
    for a in asst:
        if a.startswith("<think>\n") and "</think>" in a:
            out.append("preserved")
        elif a.startswith("<think></think>"):
            out.append("empty-think")
        else:
            out.append("other")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("request_json")
    ap.add_argument("--tokenizer", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tokenizer_v5"))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    body = json.load(open(a.request_json))
    msgs, tools = body["messages"], body.get("tools")

    def render(ms):
        return tok.apply_chat_template(ms, tools=tools, tokenize=False, add_generation_prompt=True)

    def strip(m):
        if m.get("role") != "assistant" or not isinstance(m.get("content"), str):
            return m
        return dict(m, content=split_think(m["content"])["answer"].strip())

    on, off = render(msgs), render([strip(m) for m in msgs])
    kinds = classify(on)
    n_on, n_off = (len(tok(t, add_special_tokens=False)["input_ids"]) for t in (on, off))
    ok_tail = on.endswith("<|im_start|>assistant\n<think>\n")
    n_pres = kinds.count("preserved")
    print(f"[render] tools={'yes' if tools else 'no'} history_assistant={kinds} preserved={n_pres}/{len(kinds)} "
          f"tokens ON={n_on} OFF={n_off} diff={n_on - n_off} gen_prompt_ok={ok_tail}")
    bad = ("other" in kinds) or n_pres == 0 or not ok_tail
    if bad:
        print("[render] ❌ 히스토리 think 보존 실패 — 프록시 복원 또는 템플릿 경로를 의심")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
