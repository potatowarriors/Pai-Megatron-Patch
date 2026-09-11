#!/usr/bin/env python3
"""p3_report.py — P3 통합 검수: 생성 산출 전부(A 판정본·A1/A2 judged·구제·B) 를 교사×트랙으로 집계해 NVIDIA Chat-v3 와 대조하고 게이트·심판을 요약한다.
사용: python3 p3_report.py --inputs out/p1/gen_A.jsonl,out/p1/gen_A1.judged.jsonl,out/p1/gen_A2.judged.jsonl,out/p1/gen_A_salvage.jsonl,out/p1/gen_B.jsonl --out out/p1/P3_REPORT.md [--sample 6000]
"""
import argparse, json, re, random, collections, statistics as st, os
from transformers import AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "..", "..", "tokenizer_v5")); nt = lambda s: len(tok(s or "", add_special_tokens=False).input_ids)
H = re.compile(r"[가-힣]"); L = re.compile(r"[A-Za-z]"); J = re.compile(r"[一-鿿]")
def hr(s): h, l, j = len(H.findall(s)), len(L.findall(s)), len(J.findall(s)); t = h + l + j; return (h / t if t else 0, j / t if t else 0)
def track(r):
    so = r.get("seed_origin") or {}; d = so.get("dataset") or ""; s = r.get("source") or ""
    if d == "ko_context_grid": return "S4 한국 맥락"
    if d.startswith("Nemotron"): return "S1 현지화"
    if "lmsys" in d or "WildChat" in d or s in ("lmsys", "wildchat1m"): return "S3 실사용자"
    return "S2 기사-" + (r.get("mode") or "?")
def q(x, p): x = sorted(x); return x[int(p * (len(x) - 1))] if x else 0
ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--sample", type=int, default=6000); a = ap.parse_args()
rows = []; seen = set(); dup = 0; byfile = collections.Counter()
for f in a.inputs.split(","):
    for l in open(f):
        r = json.loads(l)
        if r["conv_id"] in seen: dup += 1; continue
        seen.add(r["conv_id"]); rows.append(r); byfile[os.path.basename(f)] += 1
ver = collections.Counter((r["teacher"], r["ko_synthesis"].get("verdict")) for r in rows)
pend = sum(1 for r in rows if r["ko_synthesis"].get("judge_pending"))
selfattr = sum(1 for r in rows if re.search(r"(저는|나는|제가)\s*[^.\n]{0,25}?(GLM|Zhipu|智谱|DeepSeek|딥시크|OpenAI|Google|구글|Gemini|Claude|Qwen)", r["messages"][-1]["content"]))
multi = sum(1 for r in rows if sum(1 for m in r["messages"] if m["role"] == "user") >= 2)
random.seed(7); samp = random.sample(rows, min(a.sample, len(rows)))
g = collections.defaultdict(list)
for r in samp: g[(r["teacher"], track(r))].append(r)
doc = ["# P3 통합 검수 (kor-chat-v3 P1 트랜치)", "", f"- 행 {len(rows):,} (파일별 {dict(byfile)}; conv_id 중복 제거 {dup})", f"- 교사별: {dict(collections.Counter(r['teacher'] for r in rows))}", f"- 트랙별: {dict(collections.Counter(track(r) for r in rows))}",
       f"- 심판 판정(교사, 판정): {dict(ver)} · 심판 보류 잔여 {pend}", f"- 자기귀속 재검(채택 답변) {selfattr} · 멀티턴 {multi} ({multi/len(rows):.1%})", "",
       f"## 교사 × 트랙 (표본 {len(samp):,}행 토큰화) — 대조 NVIDIA Chat-v3 사고 1,256 (970/1,606) · 답변 878", "",
       "| 교사 | 트랙 | 표본 | 사고 tok p50 (p25/p75) | p90 | 답변 tok p50 | 사고 한글비 | 한자 | 답변 한글비 |", "|---|---|---|---|---|---|---|---|---|"]
for (t, k), rs in sorted(g.items()):
    rt = [nt(r["messages"][-1]["reasoning_content"]) for r in rs]; ct = [nt(r["messages"][-1]["content"]) for r in rs]; rh = [hr(r["messages"][-1]["reasoning_content"]) for r in rs]; ch = [hr(r["messages"][-1]["content"])[0] for r in rs]
    doc.append(f"| {t} | {k} | {len(rs)} | **{q(rt,.5)}** ({q(rt,.25)}/{q(rt,.75)}) | {q(rt,.9)} | {q(ct,.5)} | {st.mean(x[0] for x in rh):.2f} | {st.mean(x[1] for x in rh):.4f} | {st.mean(ch):.2f} |")
allr = [nt(r["messages"][-1]["reasoning_content"]) for r in samp]; allc = [nt(r["messages"][-1]["content"]) for r in samp]
doc += ["", f"**전체(교사 혼합)**: 사고 p50 **{q(allr,.5)}** (p25 {q(allr,.25)} / p75 {q(allr,.75)} / p90 {q(allr,.9)}) · 답변 p50 {q(allc,.5)} — NVIDIA 1,256 / 878", ""]
open(a.out, "w").write("\n".join(doc) + "\n"); print("\n".join(doc))
