#!/usr/bin/env python3
"""pilot_generate.py — ko_chat v3 P1 파일럿: 교사 네이티브 사고 검증 + reasoning 언어 A/B + 처리량 + 누출/정체성.

v1/v2 실패의 핵심은 "비-reasoning 교사의 가짜 사고". 이 파일럿의 통과 조건은:
  1. 네이티브 사고가 실제로 길고 실질적인가 (중앙값 길이가 v1/v2 의 204자를 크게 상회)
  2. reasoning 언어: native(교사 자연 언어) vs ko(한국어 사고 지시) A/B — 한국어 프롬프트에서 품질 유지하며 한글 사고 가능한가
  3. 누출 0 (교사 벤더 자기귀속), 정체성 정상 (alpha-banana / CJ)
  4. 처리량 실측 (동시 요청 집계 tok/s)

사용: python3 pilot_generate.py --endpoint http://localhost:8300/v1 --model glm53-flash \
        --seeds out/seeds_ko_pilot.jsonl --n 60 --concurrency 16 --out out/pilot_<tag>.json
"""
import argparse, json, re, time, urllib.request, statistics as st, os
from concurrent.futures import ThreadPoolExecutor
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CARD = yaml.safe_load(open(os.path.join(HERE, "..", "identity", "identity_card.yaml")))
NAME = CARD["name"]["canonical"]; ORG = CARD["organization"]["ko"]
SYS_IDENTITY = (f"당신은 {ORG}에서 개발한 AI 어시스턴트 {NAME}입니다. "
                f"사용자에게 정확하고 도움이 되는 답변을 한국어로 제공합니다.")
SYS_KO_THINK = SYS_IDENTITY + " 답변 전 사고 과정(reasoning)도 한국어로 작성합니다."

H = re.compile(r"[가-힣]"); L = re.compile(r"[A-Za-z]"); HANJA = re.compile(r"[一-鿿]")
VENDOR = re.compile(r"\b(Google|Gemini|Gemma|OpenAI|GPT-|ChatGPT|Anthropic|Claude|Zhipu|智谱|GLM|Qwen|Alibaba|DeepSeek|Moonshot|Kimi|Mistral|Llama|Meta)\b", re.I)

def ratios(s):
    h, l, hj = len(H.findall(s)), len(L.findall(s)), len(HANJA.findall(s))
    tot = h + l + hj
    return (h / tot if tot else None, hj / tot if tot else None)

def chat(ep, model, system, user, timeout=600):
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 8192, "temperature": 1.0, "top_p": 0.95}
    req = urllib.request.Request(ep.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    ch = d["choices"][0]; m = ch["message"]
    return {"reasoning": m.get("reasoning") or m.get("reasoning_content") or "", "content": m.get("content") or "",
            "finish": ch["finish_reason"], "ctoks": d.get("usage", {}).get("completion_tokens", 0)}

def run_cond(ep, model, seeds, system, cond, workers):
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda s: chat(ep, model, system, s["first_user"]), seeds))
    wall = time.time() - t0
    rows = []
    for s, r in zip(seeds, res):
        rhr, rhj = ratios(r["reasoning"]); chr_, chj = ratios(r["content"])
        rows.append({"cond": cond, "q": s["first_user"][:80], "r_chars": len(r["reasoning"]), "r_hangul": rhr,
                     "r_hanja": rhj, "c_chars": len(r["content"]), "c_hangul": chr_, "finish": r["finish"],
                     "ctoks": r["ctoks"], "vendor_hits": VENDOR.findall((r["reasoning"] + " " + r["content"])),
                     "reasoning": r["reasoning"][:500], "content": r["content"][:300]})
    tot_toks = sum(r["ctoks"] for r in rows)
    return rows, wall, tot_toks

def summ(rows):
    rc = [r["r_chars"] for r in rows]; rh = [r["r_hangul"] for r in rows if r["r_hangul"] is not None]
    empty = sum(r["r_chars"] == 0 for r in rows); leak = sum(bool(r["vendor_hits"]) for r in rows)
    return dict(n=len(rows), r_chars_med=int(st.median(rc)) if rc else 0, r_chars_p25=sorted(rc)[len(rc)//4] if rc else 0,
                r_hangul_mean=round(st.mean(rh), 2) if rh else None, r_empty=empty, leak=leak,
                c_hangul_mean=round(st.mean([r["c_hangul"] for r in rows if r["c_hangul"] is not None]), 2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True); ap.add_argument("--model", default="glm53-flash")
    ap.add_argument("--seeds", required=True); ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=16); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [json.loads(l) for l in open(a.seeds)][:a.n]
    out = {"endpoint": a.endpoint, "model": a.model, "n": len(seeds), "conditions": {}}
    for cond, system in (("native", SYS_IDENTITY), ("ko_think", SYS_KO_THINK)):
        rows, wall, toks = run_cond(a.endpoint, a.model, seeds, system, cond, a.concurrency)
        s = summ(rows); s["wall_s"] = round(wall, 1); s["throughput_toks_s"] = round(toks / wall, 1)
        out["conditions"][cond] = {"summary": s, "rows": rows}
        print(f"[{cond}] {s}", flush=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("saved", a.out, flush=True)

if __name__ == "__main__":
    main()
