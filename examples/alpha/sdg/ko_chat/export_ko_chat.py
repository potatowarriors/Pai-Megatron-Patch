#!/usr/bin/env python3
"""export_ko_chat.py — 트랙 B 산출물(parquet) → Nemotron messages jsonl.

identity/export_sft.py 전례를 따른다:
  - 필터: ko_check 규칙 게이트 + judge 점수 임계 (심판은 관대하므로 약한 게이트,
    기본 korean_naturalness≥4, helpfulness≥3, factuality≥4, coherence≥4)
  - 중복 제거: 정규화 완전중복 + (task_type, domain, 접두 40자) 및 **후미 40자**
    버킷 상한 (후미 수렴이 접두보다 심각하다 — identity 실측)
  - train_turns: 모든 assistant 턴 True (전 턴이 네이티브 생성이므로)

사용:
  python3 export_ko_chat.py --dataset 'artifacts/ko_chat_b_r1/**/*.parquet' \
      --out out/trackB_r1.jsonl
"""
import argparse
import glob
import hashlib
import json
import re
import uuid as uuidlib
from collections import defaultdict
from pathlib import Path

import pandas as pd

# special-token 리터럴 방어 (LC-A iter170 사고 반영 — 상세 extract_sources.py 주석).
# 이미 생성된 parquet(가드 이전 런)에도 소급 적용되는 최종 게이트.
SPECIAL_TOKEN_RE = re.compile(r"<\|[A-Za-z_]+\|>|</?(?:tool_call|tool_response|think)>")

GENERIC_SYSTEMS = [
    "당신은 유능하고 친절한 AI 어시스턴트입니다.",
    "당신은 도움이 되는 AI 어시스턴트입니다. 정확하고 친절하게 답변합니다.",
    "You are a helpful assistant.",
    "당신은 한국어에 능숙한 AI 어시스턴트입니다.",
]


def _as_dict(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip().startswith("{"):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return {}
    return {}


def _judge_score(judge, name):
    j = _as_dict(judge)
    s = j.get(name)
    if isinstance(s, dict):
        s = s.get("score")
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _norm(t):
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def _assistant_msg(turn):
    """identity/export_sft 전례: reasoning 있으면 reasoning_content 로 부착."""
    msg = {"role": "assistant", "content": turn["content"].strip()}
    reasoning = (turn.get("reasoning") or "").strip()
    if reasoning:
        msg["reasoning_content"] = reasoning
    return msg


def build_record(row):
    uc = _as_dict(row.get("user_turn"))
    ac = _as_dict(row.get("assistant_turn"))
    if not uc.get("message") or not ac.get("content"):
        return None
    messages = []
    if row.get("system_variant") == "generic_helpful":
        h = int(hashlib.sha256(str(row.get("record_uuid")).encode()).hexdigest(), 16)
        messages.append({"role": "system", "content": GENERIC_SYSTEMS[h % len(GENERIC_SYSTEMS)]})
    messages.append({"role": "user", "content": uc["message"].strip()})
    messages.append(_assistant_msg(ac))
    fu, a2 = _as_dict(row.get("followup_user")), _as_dict(row.get("assistant_turn_2"))
    if fu.get("message") and a2.get("content"):
        messages.append({"role": "user", "content": fu["message"].strip()})
        messages.append(_assistant_msg(a2))
        fu2, a3 = _as_dict(row.get("followup_user_2")), _as_dict(row.get("assistant_turn_3"))
        if fu2.get("message") and a3.get("content"):
            messages.append({"role": "user", "content": fu2["message"].strip()})
            messages.append(_assistant_msg(a3))
    return {
        "messages": messages,
        "used_in": ["sft"],
        "uuid": str(row.get("record_uuid") or uuidlib.uuid4()),
        "metadata": {
            "seed_dataset": "ko_chat_sdg_native",
            "model": "google/gemma-4-31B-it",
            "ko_synthesis": {"method": "native_generation", "date": "2026-08-23"},
            "axes": {k: row.get(k) for k in
                     ("task_type", "domain", "persona", "user_style",
                      "turn_shape", "specificity", "length_style")},
            "train_turns": [m["role"] == "assistant" for m in messages],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="parquet glob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-naturalness", type=int, default=4)
    ap.add_argument("--min-helpfulness", type=int, default=3)
    ap.add_argument("--min-factuality", type=int, default=4)
    ap.add_argument("--min-coherence", type=int, default=4)
    ap.add_argument("--bucket-cap", type=int, default=3, help="접두/후미 버킷당 최대 행")
    args = ap.parse_args()

    files = sorted(glob.glob(args.dataset, recursive=True))
    if not files:
        raise SystemExit(f"parquet 없음: {args.dataset}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"rows loaded: {len(df)} from {len(files)} files")

    drops = defaultdict(int)
    kept, seen_exact = [], set()
    prefix_buckets, suffix_buckets = defaultdict(int), defaultdict(int)

    for _, row in df.iterrows():
        chk = _as_dict(row.get("ko_check"))
        if not chk.get("is_valid", False):
            drops[f"rule:{chk.get('invalid_reason','?')}"] += 1
            continue
        scores = {
            "korean_naturalness": args.min_naturalness,
            "helpfulness": args.min_helpfulness,
            "factuality": args.min_factuality,
            "coherence": args.min_coherence,
        }
        bad = False
        for name, minv in scores.items():
            s = _judge_score(row.get("judge"), name)
            if s is not None and s < minv:
                drops[f"judge:{name}<{minv}"] += 1
                bad = True
                break
        if bad:
            continue
        rec = build_record(row)
        if rec is None:
            drops["missing_turn"] += 1
            continue
        if any(SPECIAL_TOKEN_RE.search(m.get(f) or "")
               for m in rec["messages"] for f in ("content", "reasoning_content")):
            drops["special_token_literal"] += 1
            continue
        a_texts = [m["content"] for m in rec["messages"] if m["role"] == "assistant"]
        norm_all = _norm(" ".join(a_texts))
        h = hashlib.sha256(norm_all.encode()).hexdigest()
        if h in seen_exact:
            drops["dup_exact"] += 1
            continue
        seen_exact.add(h)
        key = (row.get("task_type"), row.get("domain"))
        pb = (key, norm_all[:40])
        sb = (key, norm_all[-40:])
        if prefix_buckets[pb] >= args.bucket_cap or suffix_buckets[sb] >= args.bucket_cap:
            drops["dup_bucket"] += 1
            continue
        prefix_buckets[pb] += 1
        suffix_buckets[sb] += 1
        kept.append(rec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    total_chars = sum(len(m["content"]) for r in kept for m in r["messages"])
    print(f"kept {len(kept)}/{len(df)} → {out}  (~{total_chars/1e6:.1f}M chars)")
    for k, v in sorted(drops.items(), key=lambda x: -x[1]):
        print(f"  drop {k}: {v}")


if __name__ == "__main__":
    main()
