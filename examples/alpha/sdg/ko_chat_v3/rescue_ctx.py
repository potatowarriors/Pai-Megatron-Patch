#!/usr/bin/env python3
"""rescue_ctx.py — context_prompts.py 리젝(전문 보존) 을 현재 게이트로 재판정해 통과분을 별도 시드 파일로(merge_seeds.py 가 seeds_*.jsonl 을 모두 읽음).
사용: python3 rescue_ctx.py --rejects out/p1/seeds_ctx_glm.rejects.jsonl,out/p1/seeds_ctx_dsv4.rejects.jsonl --out out/p1/seeds_ctx_rescued.jsonl
"""
import argparse, json, hashlib, re, collections
from context_prompts import gate
ap = argparse.ArgumentParser(); ap.add_argument("--rejects", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
n = collections.Counter(); seen = set()
with open(a.out, "w") as o:
    for f in a.rejects.split(","):
        wn = "glm53-flash" if "glm" in f else "dsv4-flash"
        for l in open(f):
            r = json.loads(l); p = r.get("prompt"); why = gate(p) if p else "no_prompt"
            k = hashlib.sha1(re.sub(r"\s+", " ", p or "")[:160].encode()).hexdigest()
            if not why and k in seen: why = "dup"
            n[why or "rescued"] += 1
            if why: continue
            seen.add(k); cid = "ctx:" + hashlib.sha1(json.dumps(r["combo"], ensure_ascii=False).encode() + p[:80].encode()).hexdigest()[:16]
            o.write(json.dumps({"source": "ko_context_grid", "conv_id": cid, "language": "Korean", "turns": [{"role": "user", "content": p}], "first_user": p, "n_user_turns": 1,
                                "seed_origin": {"dataset": "ko_context_grid", "writer": wn, "rescued": True, **r["combo"]}}, ensure_ascii=False) + "\n")
print(dict(n))
