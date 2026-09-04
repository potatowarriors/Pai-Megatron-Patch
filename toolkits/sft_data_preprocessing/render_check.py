#!/usr/bin/env python3
# Copyright (c) 2026 alpha team. Apache-2.0.
"""SFT bins 렌더 육안 확인 (INTERLEAVED_THINKING.md §7 규칙 9) — 변환 산출물에서 문서를 골라 디코드하고
첫 <tool_response> 블록(없으면 첫 assistant 턴)을 뽑아 RENDER_CHECK.md 로 남긴다.

opencode_v1 사고(KNOWN_ISSUES 2026-09-01 ①: tool 결과가 Python repr `[{'type': 'tool-result', …}]` + 리터럴 `\\n` 으로
렌더)를 자동 탐지한다. 통과 기준은 사람이 본문을 읽고 판단하는 것이며, 이 스크립트는 그 근거를 고정 형식으로 기록한다.

Usage:
  python render_check.py --member /path/tree/<set> [--docs 0,1000,-1] [--write]
  python render_check.py --tree /path/tree --only a,b,c --write     # 여러 멤버
bins 계약(문서 = [tokens(S) | labels(S)], 음수 마커 = 세그먼트 끝, 복원 = -x-1)은 verify_sft_bins.py 와 동일.
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np

IGNORE = -100
PAD_ID = 1
# repr 봉투의 결정적 흔적만 자동 플래그. JSON 문자열 안의 이스케이프 `\n` 은 정상(str tool 결과가 JSON 인 셋)이라 플래그하지
# 않고 개수만 보고한다 — opencode_v1 사고의 리터럴 `\n` 은 repr 봉투와 함께 나타났으므로 위 패턴으로 잡힌다.
ENVELOPE_PATTERNS = ("[{'type'", "{'type':", "[{\"type\": \"tool-result\"", "'tool-result'")


def restore(tokens: np.ndarray) -> np.ndarray:
    t = tokens.astype(np.int64).copy()
    neg = t < 0
    t[neg] = -t[neg] - 1
    return t


def first_sample(doc: np.ndarray):
    """문서의 첫 세그먼트(샘플) 토큰과 라벨을 돌려준다."""
    S = doc.size // 2
    tokens = doc[:S].astype(np.int64)
    labels = doc[S:].astype(np.int64)
    markers = np.nonzero(tokens < 0)[0]
    end = int(markers[0]) + 1 if markers.size else S
    return restore(tokens[:end]), labels[:end]


def inspect_doc(tok, doc: np.ndarray, width: int):
    ids, labels = first_sample(doc)
    text = tok.decode(ids.tolist())
    out = {
        "tokens": int(ids.size),
        "trainable": int((labels != IGNORE).sum()),
        "assistant_turns": text.count("<|im_start|>assistant"),
        "think_open": text.count("<think>"),
        "think_close": text.count("</think>"),
        "tool_call": text.count("<tool_call>"),
        "tool_response": text.count("<tool_response>"),
    }
    i = text.find("<tool_response>")
    if i >= 0:
        block = text[i:i + width]
        out["kind"] = "tool_response"
    else:
        j = text.find("<|im_start|>assistant")
        block = text[j:j + width] if j >= 0 else text[:width]
        out["kind"] = "assistant_head"
    out["block"] = block
    out["envelope"] = [p for p in ENVELOPE_PATTERNS if p in block]
    out["literal_newlines"] = block.count("\\n")
    li = np.nonzero(labels != IGNORE)[0]
    if li.size:
        s = int(li[0]); e = s
        while e < labels.size and labels[e] != IGNORE:
            e += 1
        # labels 는 다음 토큰을 가리키므로 학습 스팬 텍스트는 ids[s+1:e+1]
        out["first_trainable"] = tok.decode(ids[s + 1:e + 1].tolist())[:240]
    else:
        out["first_trainable"] = ""
    return out


def check_member(member_dir: str, tok, docs, width: int, write: bool) -> bool:
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    prefix = os.path.join(member_dir, "data_text_document")
    if not os.path.exists(prefix + ".idx"):
        print(f"[skip] {member_dir}: idx 없음")
        return False
    ds = IndexedDataset(prefix)
    n = len(ds)
    picks = []
    for d in docs:
        d = n - 1 if d < 0 else d
        if 0 <= d < n and d not in picks:
            picks.append(d)
    name = os.path.basename(member_dir.rstrip("/"))
    lines = [f"# {name} 렌더 육안 확인 (규칙 9) — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             f"docs={n:,}. 각 문서의 첫 샘플을 디코드해 첫 `<tool_response>` 블록(없으면 첫 assistant 턴 머리)을 기록한다.",
             "봉투 흔적 = Python repr(`[{'type': …`)·리터럴 `\\n` 탐지. 판정은 본문을 읽고 사람이 한다.", ""]
    flagged = False
    for d in picks:
        r = inspect_doc(tok, np.asarray(ds[d]), width)
        env = ", ".join(r["envelope"]) if r["envelope"] else "없음"
        flagged |= bool(r["envelope"])
        lines += [f"## doc {d}  ({r['kind']})  봉투 흔적: {env}",
                  f"tokens {r['tokens']:,} · trainable {r['trainable']:,} · assistant 턴 {r['assistant_turns']} · "
                  f"`<think>` {r['think_open']}/{r['think_close']} · tool_call {r['tool_call']} · tool_response {r['tool_response']}"
                  f" · 블록 내 이스케이프 `\\n` {r['literal_newlines']}개(JSON 문자열이면 정상)",
                  "```", r["block"].rstrip(), "```",
                  "첫 학습 스팬:", "```", r["first_trainable"].rstrip(), "```", ""]
    body = "\n".join(lines)
    print(body)
    if write:
        with open(os.path.join(member_dir, "RENDER_CHECK.md"), "w", encoding="utf-8") as f:
            f.write(body)
        print(f"[write] {os.path.join(member_dir, 'RENDER_CHECK.md')}")
    return not flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", help="단일 멤버 디렉터리")
    ap.add_argument("--tree", help="트리 (여러 멤버)")
    ap.add_argument("--only", help="--tree 에서 검사할 멤버 이름 콤마 목록 (기본: symlink 아닌 전부)")
    ap.add_argument("--docs", default="0,1000,-1", help="검사 문서 인덱스 (음수 = 뒤에서)")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--width", type=int, default=700, help="블록 길이(문자)")
    ap.add_argument("--write", action="store_true", help="멤버 디렉터리에 RENDER_CHECK.md 기록")
    a = ap.parse_args()
    if not a.member and not a.tree:
        sys.exit("--member 또는 --tree 필요")
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tok_path = a.tokenizer or os.path.join(repo, "examples", "alpha", "tokenizer_v5")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path)
    docs = [int(x) for x in a.docs.split(",") if x.strip()]
    members = []
    if a.member:
        members.append(a.member)
    if a.tree:
        names = a.only.split(",") if a.only else sorted(
            n for n in os.listdir(a.tree) if not os.path.islink(os.path.join(a.tree, n)) and os.path.isdir(os.path.join(a.tree, n)))
        members += [os.path.join(a.tree, n) for n in names]
    ok = True
    for m in members:
        ok &= check_member(m, tok, docs, a.width, a.write)
    print("\n[RESULT]", "clean (봉투 흔적 없음)" if ok else "FLAGGED — 본문 확인 필요")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
