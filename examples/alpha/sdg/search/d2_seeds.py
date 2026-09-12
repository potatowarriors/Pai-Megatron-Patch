#!/usr/bin/env python3
"""d2_seeds.py — 트랙 D2 시드: NIKL 기사(2024, 9주제)에서 **여러 출처를 대조해야 답할 수 있는 조사형 요청**을 GLM 이 한국어로 작성.
로컬 코퍼스(위키+2024 기사)로 답할 수 있어야 하므로 주제는 기사에서 뽑고, 요청은 배경·쟁점·이해관계·시간순·비교 같은 다면적 조사를 요구하도록 한다.
사용: python3 d2_seeds.py --n 1200 --out out/d2/requests.jsonl --workers 64
"""
import argparse, json, glob, os, re, sys, random, threading, time, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ko_chat_v3")); import generate_v3 as g
NIKL = "/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/datasets/raw/Korea_NewsPaper_Corpus/NIKL_NEWSPAPER_2025_v1.0/*.json"
KINDS = ["배경과 경과를 시간순으로 정리한 보고서", "찬반·이해관계자별 입장을 비교한 보고서", "관련 제도·정책의 현황과 쟁점 정리", "국내외 유사 사례 비교", "원인·영향·전망 분석", "핵심 용어·개념 설명과 관련 현황 조사"]
PERSONAS = ["대학생 과제", "기업 기획팀 실무자", "지자체 공무원", "기자 지망생", "투자자", "시민단체 활동가", "교사 수업 자료"]
PROMPT = """아래 기사 제목·요지에서 출발해, AI 리서치 어시스턴트에게 맡길 **조사형 요청** 하나를 한국어로 쓰세요.
- 요청자: {persona}
- 원하는 산출물: {kind}
- 요청은 한 번의 검색으로 답할 수 없고 여러 출처(기사·백과)를 대조해야 하는 것이어야 합니다. 특정 기사 자체를 언급하지 말고 주제로 요청하세요.
- 조사 범위(기간·대상·관점)를 구체적으로 지정하고, 출처 표시를 요구하세요. 2~4문장.

[기사]
제목: {title}
요지: {lead}

출력은 JSON 하나만: {{"request": "..."}}"""

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1200); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=64); ap.add_argument("--seed", type=int, default=3); a = ap.parse_args()
    rng = random.Random(a.seed); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    files = sorted(glob.glob(NIKL)); rng.shuffle(files); arts = []
    for fp in files[:40]:
        d = json.load(open(fp, encoding="utf-8"))
        for doc in d.get("document", []):
            md = doc.get("metadata") or {}; paras = [re.sub(r"</?p>", "", x.get("form") or "").strip() for x in doc.get("paragraph", [])]
            body = " ".join(p for p in paras if p)
            if len(body) < 600: continue
            arts.append({"id": doc.get("id"), "title": md.get("title"), "topic": md.get("topic"), "lead": body[:400]})
    rng.shuffle(arts); by = collections.defaultdict(list)
    for x in arts: by[x["topic"]].append(x)
    pick = []; per = max(a.n // max(len(by), 1), 1)
    for t, xs in by.items(): pick += xs[:per + 1]
    pick = pick[:a.n]; print(f"articles {len(arts)} topics {len(by)} picked {len(pick)}", flush=True)
    t = g.ALL_TEACHERS["glm53-flash"]; st = collections.Counter(); lock = threading.Lock(); t0 = time.time()
    def work(x):
        body = {"model": t["model"], "messages": [{"role": "user", "content": PROMPT.format(persona=rng.choice(PERSONAS), kind=rng.choice(KINDS), title=x["title"], lead=x["lead"])}], "max_tokens": 3000, "temperature": 0.9, "top_p": 0.95, "chat_template_kwargs": {"reasoning_effort": "low"}}
        try: d = g.post(t["endpoint"], body, timeout=600); c = d["choices"][0]["message"].get("content") or ""
        except Exception as e: return x, None, f"error:{e!r}"[:80]
        m = re.search(r"\{.*\}", c, flags=re.S)
        if not m: return x, None, "no_json"
        try: q = (json.loads(m.group(0)).get("request") or "").strip()
        except Exception: return x, None, "bad_json"
        if len(q) < 40 or len(re.findall(r"[가-힣]", q)) < 30: return x, None, "bad_request"
        return x, q, None
    with open(a.out, "w") as out, ThreadPoolExecutor(a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, x) for x in pick]), 1):
            x, q, why = f.result()
            with lock:
                if why: st["rej_" + why.split(":")[0]] += 1
                else: st["ok"] += 1; out.write(json.dumps({"conv_id": f"d2:{x['id']}", "question": q, "seed_origin": {"dataset": "nikl_news", "doc_id": x["id"], "topic": x["topic"], "title": x["title"]}}, ensure_ascii=False) + "\n")
                if n % 200 == 0 or n == len(pick): print(f"[{n}/{len(pick)}] {time.time()-t0:.0f}s {dict(st)}", flush=True)
    print("DONE", dict(st), flush=True)

if __name__ == "__main__":
    main()
