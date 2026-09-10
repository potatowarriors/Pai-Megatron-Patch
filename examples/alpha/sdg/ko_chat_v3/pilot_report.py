#!/usr/bin/env python3
"""pilot_report.py — P0 파일럿 표본 보고서: 통계(NVIDIA Chat-v3 대조) + 모드/소스별 실제 표본(프롬프트·사고·답변).
사용: python3 pilot_report.py --news out/pilot_gen_news.jsonl --loc out/pilot_gen_loc.jsonl --out out/P0_PILOT_REPORT.md --samples 4
"""
import argparse, json, re, random, statistics as st, collections, os
from transformers import AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__))
tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "..", "..", "tokenizer_v5")); ntok = lambda s: len(tok(s or "", add_special_tokens=False).input_ids)
H = re.compile(r"[가-힣]"); L = re.compile(r"[A-Za-z]"); J = re.compile(r"[一-鿿]")
def hr(s): h, l, j = len(H.findall(s)), len(L.findall(s)), len(J.findall(s)); t = h + l + j; return (h / t if t else 0, j / t if t else 0)
def q(x, p): x = sorted(x); return x[int(p * (len(x) - 1))] if x else 0

def load(p):
    rows = [json.loads(l) for l in open(p)] if p and os.path.exists(p) else []
    rej = p.replace(".jsonl", "") + ".rejects.jsonl"
    nrej = sum(1 for _ in open(rej)) if os.path.exists(rej) else 0
    return rows, nrej

def stats(name, rows, nrej, key=None):
    groups = collections.defaultdict(list)
    for r in rows: groups[(key(r) if key else "all")].append(r)
    lines = [f"### {name}: 채택 {len(rows)} / 양샘플 리젝 {nrej} (채택률 {len(rows)/max(len(rows)+nrej,1):.1%})", "",
             "| 그룹 | 행 | 사고 tok 중앙값 (p25/p75) | 답변 tok 중앙값 | 사고 한글비 | 한자 | 답변 한글비 | 심판 A/B/없음 | 리젝 사유 |", "|---|---|---|---|---|---|---|---|---|"]
    for g, rs in sorted(groups.items()):
        rt = [ntok(r["messages"][-1]["reasoning_content"]) for r in rs]; ct = [ntok(r["messages"][-1]["content"]) for r in rs]
        rh = [hr(r["messages"][-1]["reasoning_content"]) for r in rs]; ch = [hr(r["messages"][-1]["content"])[0] for r in rs]
        v = collections.Counter(r["ko_synthesis"]["verdict"] for r in rs); rej = collections.Counter(x["why"].split(":")[0] for r in rs for x in r["ko_synthesis"]["rejects"])
        lines.append(f"| {g} | {len(rs)} | **{q(rt,.5)}** ({q(rt,.25)}/{q(rt,.75)}) | {q(ct,.5)} | {st.mean(x[0] for x in rh):.2f} | {st.mean(x[1] for x in rh):.3f} | {st.mean(ch):.2f} | {v.get('A',0)}/{v.get('B',0)}/{v.get(None,0)} | {dict(rej) or '-'} |")
    return lines

def samples(name, rows, n, key=None, seed=1):
    random.seed(seed); out = [f"### {name} 표본"]
    groups = collections.defaultdict(list)
    for r in rows: groups[(key(r) if key else "all")].append(r)
    for g, rs in sorted(groups.items()):
        for r in random.sample(rs, min(n, len(rs))):
            a = r["messages"][-1]; u = r["messages"][-2]["content"]
            if "\n\n[기사]" in u: u = u.split("\n\n[기사]")[0] + "\n\n[기사 … 본문 생략 …]"
            meta = r.get("seed_origin", {}); m = r["ko_synthesis"]
            out += [f"\n#### [{g}] {meta.get('topic') or meta.get('seed_dataset','')} · {r.get('task_type','')} · {r.get('persona','')}",
                    f"- 사고 {ntok(a['reasoning_content'])} tok · 답변 {ntok(a['content'])} tok · 심판 {m['verdict']} · 교사 {r['teacher']}",
                    "", "**프롬프트**", "", "> " + u[:700].replace("\n", "\n> "), "", "**사고(앞부분)**", "", "> " + a["reasoning_content"][:500].replace("\n", "\n> ") + " …",
                    "", "**답변(앞부분)**", "", "> " + a["content"][:900].replace("\n", "\n> ") + (" …" if len(a["content"]) > 900 else "")]
    return out

ap = argparse.ArgumentParser(); ap.add_argument("--news"); ap.add_argument("--loc"); ap.add_argument("--out", required=True); ap.add_argument("--samples", type=int, default=4)
ap.add_argument("--dsv4-news"); ap.add_argument("--dsv4-loc"); ap.add_argument("--dsv4-samples", type=int, default=2)
a = ap.parse_args()
news, nrej = load(a.news); loc, lrej = load(a.loc)
doc = ["# P0 파일럿 표본 보고서 (kor-chat-v3)", "", "대조 기준: NVIDIA Chat-v3 chat(GLM-5 교사) 사고 토큰 중앙값 **1,256**, 답변 878 · Chat-v2 reasoning_on 604. 폐기 v1/v2 ≈ 60 tok(204자).", ""]
if news: doc += stats("S2 기사 기반 (모드 A=문서 제공 / B=영감형) — 교사 GLM-5.3-Flash", news, nrej, key=lambda r: "A 문서제공" if r.get("mode") == "A" else "B 영감형") + [""]
if loc: doc += stats("S1 현지화 (Chat-v3 영어 프롬프트 → 한국어, thinking-on 재작성) — 교사 GLM-5.3-Flash", loc, lrej) + [""]
# 교사 비교: 같은 시드 부분집합을 DSV4 로 생성한 결과 (심판 = GLM low-effort)
dn, dnrej = load(a.dsv4_news) if a.dsv4_news else ([], 0); dl, dlrej = load(a.dsv4_loc) if a.dsv4_loc else ([], 0)
if dn or dl:
    doc += ["## 교사 비교 — 같은 시드, GLM vs DSV4-Flash-0731", ""]
    if dn:
        ids = {r["conv_id"] for r in dn}; g = [r for r in news if r["conv_id"] in ids]
        doc += stats("기사 시드 부분집합 — GLM", g, 0, key=lambda r: "A 문서제공" if r.get("mode") == "A" else "B 영감형")
        doc += stats("기사 시드 부분집합 — DSV4", dn, dnrej, key=lambda r: "A 문서제공" if r.get("mode") == "A" else "B 영감형") + [""]
    if dl:
        ids = {r["conv_id"] for r in dl}; g = [r for r in loc if r["conv_id"] in ids]
        doc += stats("현지화 시드 부분집합 — GLM", g, 0) + stats("현지화 시드 부분집합 — DSV4", dl, dlrej) + [""]
if news: doc += samples("S2 기사 기반 (GLM)", news, a.samples, key=lambda r: "A 문서제공" if r.get("mode") == "A" else "B 영감형") + [""]
if loc: doc += samples("S1 현지화 (GLM)", loc, a.samples) + [""]
if dn: doc += samples("S2 기사 기반 (DSV4)", dn, a.dsv4_samples, key=lambda r: "A 문서제공" if r.get("mode") == "A" else "B 영감형") + [""]
if dl: doc += samples("S1 현지화 (DSV4)", dl, a.dsv4_samples) + [""]
open(a.out, "w").write("\n".join(doc) + "\n"); print("\n".join(l for l in doc if l.startswith(("###", "|")))[:4000])
