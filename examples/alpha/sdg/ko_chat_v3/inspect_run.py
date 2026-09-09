#!/usr/bin/env python3
"""inspect_run.py — generate_v3.py 산출 검수 요약 (채택률·교사별 사고 길이/언어·심판 분포·리젝 사유·누출 재검).
사용: python3 inspect_run.py out/v3_r1_lmsys.jsonl [--md out/R1_INSPECT.md]
"""
import argparse, json, re, collections, statistics as st
H = re.compile(r"[가-힣]"); L = re.compile(r"[A-Za-z]"); HJ = re.compile(r"[一-鿿]")
def rat(s):
    h, l, j = len(H.findall(s)), len(L.findall(s)), len(HJ.findall(s)); t = h + l + j
    return (h / t, j / t) if t else (0, 0)
ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--md"); a = ap.parse_args()
rows = [json.loads(l) for l in open(a.path)]
rej_path = a.path.replace(".jsonl", ".rejects.jsonl")
try: n_rej_rows = sum(1 for _ in open(rej_path))
except FileNotFoundError: n_rej_rows = 0
by_t = collections.defaultdict(list); verd = collections.Counter(); rej = collections.Counter(); judge_fail = 0
for r in rows:
    a_ = r["messages"][-1]; m = r["ko_synthesis"]
    by_t[r["teacher"]].append((len(a_["reasoning_content"]), len(a_["content"]), rat(a_["reasoning_content"]), rat(a_["content"]), m["ctoks"]))
    verd[m["verdict"]] += 1
    if m["kept"] >= 2 and m["verdict"] is None: judge_fail += 1
    for x in m["rejects"]: rej[x["why"].split(":")[0]] += 1
lines = [f"# r1 검수 — {a.path}", "", f"- 채택 행 {len(rows)} · 양 샘플 리젝 행 {n_rej_rows} → 채택률 {len(rows)/max(len(rows)+n_rej_rows,1):.1%}",
         f"- 심판 판정 분포: {dict(verd)} · 심판 실패(2샘플인데 판정 없음) {judge_fail}",
         f"- 샘플 단위 리젝 사유: {dict(rej.most_common())}", "", "| 교사 | 행 | 사고 중앙값(자) | 사고 p25/p75 | 사고 한글비 | 사고 한자비 | 답변 중앙값(자) | 답변 한글비 | 평균 completion tok |", "|---|---|---|---|---|---|---|---|---|"]
for t, v in by_t.items():
    rc = sorted(x[0] for x in v); cc = [x[1] for x in v]
    lines.append(f"| {t} | {len(v)} | {int(st.median(rc))} | {rc[len(rc)//4]}/{rc[3*len(rc)//4]} | {st.mean(x[2][0] for x in v):.2f} | {st.mean(x[2][1] for x in v):.3f} | {int(st.median(cc))} | {st.mean(x[3][0] for x in v):.2f} | {st.mean(x[4] for x in v):.0f} |")
# 누출 재검(자기귀속 없어야 함) + 다중턴 비율
VEND = re.compile(r"(저는|나는|I am|I'm)\s*[^.\n]{0,25}?(Qwen|GLM|Google|구글|OpenAI|Claude|DeepSeek|Gemini|Alibaba)", re.I)
leak = sum(1 for r in rows if VEND.search(r["messages"][-1]["content"]))
multi = sum(1 for r in rows if sum(1 for m in r["messages"] if m["role"] == "user") >= 2)
lines += ["", f"- 채택 행 답변의 자기귀속 재검: {leak} 건 (0 이어야 함)", f"- 멀티턴(user≥2) 행: {multi} ({multi/max(len(rows),1):.1%})"]
out = "\n".join(lines); print(out)
if a.md: open(a.md, "w").write(out + "\n")
