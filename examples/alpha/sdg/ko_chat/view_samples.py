#!/usr/bin/env python3
"""view_samples.py — 합성 chat jsonl 을 사람이 읽는 형태로 출력 (검수용).

사용:
  python3 view_samples.py <jsonl> [-n 5] [--seed 0] [--filter key=val ...] [--full]
  예) python3 view_samples.py trackA_chat.jsonl -n 3
      python3 view_samples.py trackB.jsonl -n 5 --filter run=r2 --filter domain=부동산·주거
      python3 view_samples.py trackA_if.jsonl -n 2 --full      # 턴 전문 (기본은 800자 절단)
필터 키는 metadata.ko_synthesis.* / metadata.axes.* / metadata.source_split 을 평탄화해 매칭.
"""
import argparse
import json
import random
import sys


def flat_meta(r):
    m = r.get("metadata") or {}
    out = {"source_split": m.get("source_split"), "model": r.get("model") or m.get("model")}
    for k, v in (m.get("ko_synthesis") or {}).items():
        out[k] = v
    for k, v in (m.get("axes") or {}).items():
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filter", action="append", default=[])
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    filt = dict(f.split("=", 1) for f in args.filter)

    # reservoir sampling — 큰 파일도 전체를 한 번만 훑음
    rng = random.Random(args.seed)
    picked, seen = [], 0
    with open(args.path) as f:
        for line in f:
            r = json.loads(line)
            fm = flat_meta(r)
            if any(str(fm.get(k)) != v for k, v in filt.items()):
                continue
            seen += 1
            if len(picked) < args.n:
                picked.append(r)
            else:
                j = rng.randrange(seen)
                if j < args.n:
                    picked[j] = r
    print(f"# {args.path}: matched={seen}, showing={len(picked)}\n")
    lim = None if args.full else 800
    for i, r in enumerate(picked, 1):
        fm = flat_meta(r)
        tt = (r.get("metadata") or {}).get("train_turns")
        print("=" * 100)
        print(f"[{i}] uuid={r.get('uuid')} | " + " ".join(f"{k}={v}" for k, v in fm.items() if v is not None))
        print(f"    train_turns={tt}")
        for j, m in enumerate(r["messages"]):
            tag = "★학습" if (tt and j < len(tt) and tt[j]) else "  "
            print(f"\n--- [{j}] {m['role'].upper()} {tag}")
            if m.get("reasoning_content"):
                rc = m["reasoning_content"]
                print(f"<사고> {rc[:lim] if lim else rc}{' …' if lim and len(rc) > lim else ''}")
            c = m.get("content") or ""
            print(c[:lim] if lim else c, "…" if lim and len(c) > lim else "")
        print()


if __name__ == "__main__":
    main()
