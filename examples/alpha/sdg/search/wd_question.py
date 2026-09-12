#!/usr/bin/env python3
"""wd_question.py — Wikidata 연쇄(chains.jsonl) → 교사가 한국어 다중 홉 질문 1개로 자연어화 (NVIDIA `question_obfuscated` 대응).
규칙: 시작 개체(또는 그 묘사)와 관계만 드러내고 중간 개체·정답은 절대 언급 금지, 유일하게 답이 정해지도록, 자연스러운 한 문장(또는 두 문장) 질문.
검증: 정답 라벨·중간 개체 라벨이 질문에 포함되면 리젝(누출). 교사 = GLM(기본, v2 지시문 불필요 → GEN_THOROUGH=0), 사고 예산 4k.
사용: GEN_THOROUGH=0 python3 wd_question.py --chains out/d1/chains.jsonl --out out/d1/questions.jsonl --workers 96 [--teacher glm53-flash]
"""
import argparse, json, os, sys, re, threading, time, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ko_chat_v3")); import generate_v3 as g

PROMPT = """아래는 지식 그래프에서 뽑은 관계 연쇄입니다. 이 연쇄를 따라가야만 풀 수 있는 **한국어 질문 하나**를 만드세요.

연쇄:
{chain}

요구사항:
- 질문에는 시작 개체 「{seed}」(또는 그것을 가리키는 짧은 묘사)와 관계들만 드러내고, 중간 개체와 최종 정답 「{answer}」의 이름은 절대 쓰지 마세요.
- 각 관계를 자연스러운 한국어로 풀어 쓰되(예: "행정구역" → "~가 속한 광역자치단체", "출신 학교" → "~가 졸업한 학교"), 답이 하나로 정해지게 하세요.
- 한 문장 또는 두 문장. 정답은 개체명 하나가 되도록 "~은 무엇인가요?/~는 어디인가요?/~는 누구인가요?" 형태로 끝내세요.
- 출력은 JSON 하나만: {{"question": "..."}}"""

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--chains", required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--teacher", default="glm53-flash"); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    t = g.ALL_TEACHERS[a.teacher]; chains = [json.loads(l) for l in open(a.chains)]
    if a.limit: chains = chains[:a.limit]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out): done.add(json.loads(l)["chain_id"])
    # 누출 등으로 리젝된 연쇄는 최대 2회까지만 재시도(라운드마다 무한 재시도로 GLM 낭비 — 2026-09-12 실측 3,723 리젝 중 재시도 653)
    rp = a.out.replace(".jsonl", "") + ".rejects.jsonl"
    if os.path.exists(rp):
        cnt = collections.Counter(json.loads(l)["chain_id"] for l in open(rp))
        done |= {k for k, v in cnt.items() if v >= 2}
    GENERIC = {"P17", "P37", "P38", "P30", "P36", "P1376", "P361", "P131"}; BL = {"대한민국", "미국", "일본", "중국", "영국", "프랑스", "독일", "러시아", "서울특별시", "도쿄", "워싱턴 D.C.", "런던", "파리", "베이징", "한국어", "영어", "일본어", "중국어", "프랑스어", "독일어", "스페인어", "미국 달러", "대한민국 원", "유로", "엔", "아시아", "유럽", "북아메리카", "지구"}
    chains = [c for c in chains if c["hops"][-1]["prop"] not in GENERIC and c["answer"] not in BL and sum(1 for h in c["hops"] if h["prop"] in GENERIC) <= 1]
    todo = [c for c in chains if c["chain_id"] not in done]; print(f"chains={len(chains)} done={len(done)} todo={len(todo)} teacher={t['name']}", flush=True)
    st = collections.Counter(); lock = threading.Lock(); t0 = time.time()
    def work(c):
        chain = " → ".join([c["seed"]["label"]] + [f"[{h['prop_label']}] {h['to_label']}" for h in c["hops"]])
        body = {"model": t["model"], "messages": [{"role": "user", "content": PROMPT.format(chain=chain, seed=c["seed"]["label"], answer=c["answer"])}], "max_tokens": 4096, "temperature": 0.8, "top_p": 0.95}
        if t["name"] == "glm53-flash": body["chat_template_kwargs"] = {"reasoning_effort": "low"}
        try:
            d = g.post(t["endpoint"], body, timeout=900); content = (d["choices"][0]["message"].get("content") or "")
        except Exception as e: return c, None, f"error:{e!r}"[:100]
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m: return c, None, "no_json"
        try: q = (json.loads(m.group(0)).get("question") or "").strip()
        except Exception: return c, None, "bad_json"
        if len(q) < 15: return c, None, "short"
        leak = [h["to_label"] for h in c["hops"] if h["to_label"] and h["to_label"] in q]
        if leak: return c, q, "leak:" + leak[0][:20]
        if not re.search(r"[?？]\s*$", q): return c, q, "no_qmark"
        return c, q, None
    with open(a.out, "a") as out, open(a.out.replace(".jsonl", "") + ".rejects.jsonl", "a") as rej, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, c) for c in todo]
        for n, f in enumerate(as_completed(futs), 1):
            c, q, why = f.result()
            with lock:
                if why: st["rej_" + why.split(":")[0]] += 1; rej.write(json.dumps({"chain_id": c["chain_id"], "why": why, "question": q}, ensure_ascii=False) + "\n")
                else:
                    st["ok"] += 1
                    out.write(json.dumps({"conv_id": "d1:" + c["chain_id"], "question": q, "answer": c["answer"], "answer_aliases": c.get("answer_aliases", []), "n_hops": c["n_hops"], "chain_id": c["chain_id"],
                                          "seed_origin": {"dataset": "wikidata_chain_ko", "seed": c["seed"], "hops": [(h["prop"], h["to"]) for h in c["hops"]]}}, ensure_ascii=False) + "\n"); out.flush()
                if n % 200 == 0 or n == len(futs): print(f"[{n}/{len(futs)}] {time.time()-t0:.0f}s {dict(st)}", flush=True)
    print("DONE", dict(st), flush=True)

if __name__ == "__main__":
    main()
