#!/usr/bin/env python3
"""terminal_corpus_sample.py — Nemotron-Terminal-Corpus(parquet, conversations) → 표본 jsonl(messages 스키마).
채택 검증용(2026-09-10): assistant content 의 인라인 <think>…</think> 를 reasoning_content 로 분리(변환기 special-token
가드와 정합), 첫 user 턴이 Terminus 시스템 프롬프트를 겸하므로 그 프리픽스 md5 로 계열(SWE-v3 fa616539 / Harbor 7665e733) 판별.
사용: python3 terminal_corpus_sample.py <corpus_dir> --per-file 7000 --out <sample.jsonl>
"""
import argparse, glob, json, hashlib, re, collections, os
import pyarrow.parquet as pq
THINK = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*", re.S)
ap = argparse.ArgumentParser(); ap.add_argument("corpus"); ap.add_argument("--per-file", type=int, default=7000); ap.add_argument("--out", required=True)
a = ap.parse_args()
stat = collections.Counter(); md5s = collections.Counter(); think_turns = tot_turns = 0
with open(a.out, "w") as out:
    for f in sorted(glob.glob(os.path.join(a.corpus, "**", "*.parquet"), recursive=True)):
        pf = pq.ParquetFile(f); n = 0; stat[f"rows_total:{os.path.basename(f)}"] = pf.metadata.num_rows
        for b in pf.iter_batches(batch_size=500):
            for row in b.to_pylist():
                msgs = row.get("conversations") or row.get("messages") or []
                outm = []
                for m in msgs:
                    c = m.get("content") or ""; r = m.get("role")
                    if r == "assistant":
                        tot_turns += 1; mt = THINK.match(c)
                        if mt: think_turns += 1; outm.append({"role": r, "content": c[mt.end():], "reasoning_content": mt.group(1)})
                        else: outm.append({"role": r, "content": c})
                    else: outm.append({"role": r, "content": c})
                if outm and outm[0]["role"] == "user":
                    md5s[hashlib.md5(outm[0]["content"][:2836].encode()).hexdigest()[:8]] += 1
                out.write(json.dumps({"messages": outm, "source": os.path.basename(f), "task": row.get("task"), "model": row.get("model"),
                                      "agent": row.get("agent"), "enable_thinking": row.get("enable_thinking")}, ensure_ascii=False) + "\n")
                n += 1; stat[f"sampled:{os.path.basename(f)}"] += 1
                if n >= a.per_file: break
            if n >= a.per_file: break
print(json.dumps({"stat": dict(stat), "assistant_turns": tot_turns, "think_turns": think_turns,
                  "think_ratio": round(think_turns / max(tot_turns, 1), 3), "first_user_prefix_md5_top": md5s.most_common(5)}, ensure_ascii=False, indent=1))
