#!/usr/bin/env python3
"""news_prompts.py — S2 기사 기반 프롬프트 생성 (kor-chat-v3, 2026-09-10 계획 §3).

NIKL 신문 말뭉치 2025(974k 기사, 2024년) → 주제·월 층화 샘플 → reasoning 교사(GLM/DSV4, **thinking 켠 채**)가
"한국 사용자가 실제로 할 법한 요청"을 씀. 파서가 사고를 분리하므로 content(JSON)만 사용 — no-think 불필요(사용자 지적 2026-09-10).

모드 A(문서 제공형): 요청 + [기사 본문] 을 프롬프트에 포함. 요약·수치 표·근거 QA·제목/리드·반대 논평·독자별 재서술·영문 요약 등.
모드 B(영감형):  기사 없이 독립 요청. 기사가 다룬 **일반 주제**의 설명·비교·절차·의견·작문. 2024 특정 사건·인명·수치 언급 금지
               (학생 지식 밖 → 환각 학습 방지) — 사건 특정성 게이트로 걸러 A 로 강등하거나 폐기.

출력: generate_v3.py 시드 스키마(turns/first_user/conv_id + seed_origin + article_ref/mode/task_type/persona).
사용: python3 news_prompts.py --articles 200 --per-article 2 --out out/seeds_news_pilot.jsonl --workers 48
"""
import argparse, glob, json, os, random, re, time, threading, urllib.request, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

CORPUS = "/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/datasets/raw/Korea_NewsPaper_Corpus/NIKL_NEWSPAPER_2025_v1.0"
WRITERS = {
    "glm53-flash": {"endpoint": os.environ.get("GLM_EP", "http://localhost:8000/v1"), "sampling": {"temperature": 0.8, "top_p": 0.95}},
    "dsv4-flash": {"endpoint": os.environ.get("DSV4_EP", "http://sub1:8300/v1"), "sampling": {"temperature": 0.8, "top_p": 0.95}},
}
H = re.compile(r"[가-힣]"); L = re.compile(r"[A-Za-z]"); HANJA = re.compile(r"[一-鿿]")
SPECIAL = re.compile(r"<\|[A-Za-z_]+\|>|</?(?:tool_call|tool_response|think)>")
TAG = re.compile(r"</?p>")
# 모드 B 사건 특정성: 연도·날짜·'이날/지난달/오늘' 류 시점 지시·기사 언급
SPECIFIC_B = re.compile(r"(20\d\d년|20\d\d\.|\d{1,2}월 \d{1,2}일|이날|지난달|지난주|오늘|어제|최근 기사|해당 기사|위 기사|아래 기사|본문)")

TASKS_A = ["세 줄 요약", "한 문장 요약", "핵심 수치·인물·일정을 표로 정리", "본문에 근거한 사실 질문 1~2개에 답하기",
           "제목과 리드(첫 문단) 다시 쓰기", "반대 관점에서 비판적 논평", "초등학생 눈높이로 다시 설명", "외국인 독자를 위한 영문 요약",
           "후속 취재 질문 목록", "팩트 체크가 필요한 주장 골라내기", "SNS 게시용 요약문 작성", "이해관계자별 영향 분석"]
TASKS_B = ["관련 제도·용어의 배경 설명", "비슷한 선택지 비교와 장단점", "일반인이 따라 할 절차·방법 안내", "찬반 의견과 근거 정리",
           "관련 공문·안내문·이메일 초안 작성", "관련 수치를 계산·추정하는 방법 설명", "아이디어 브레인스토밍", "전문가에게 상담하듯 묻는 질문"]
PERSONAS = ["대학생", "직장인", "자영업자", "주부", "공무원", "기자 지망생", "은퇴자", "한국어를 배우는 외국인 유학생", "중학생", "스타트업 대표", "교사", "간호사"]

WRITER_INSTR = """당신은 한국어 AI 어시스턴트 학습용 데이터를 만드는 작가입니다. 아래 신문 기사를 읽고, 지정된 조건에 맞는 **한국 사용자가 실제로 AI 에게 보낼 법한 요청문 하나**를 쓰세요.

조건
- 화자: {persona}
- 모드: {mode_desc}
- 과업 유형: {task}
- 요청은 자연스러운 구어체/문어체 한국어. 인위적 표시("다음 조건에 따라" 등) 없이 사람이 쓴 것처럼.
- 길이는 상황에 맞게(짧으면 1~2문장, 필요하면 배경 설명 포함 몇 문장).
{mode_rules}

출력은 아래 JSON 하나만 (설명·코드펜스 금지):
{{"prompt": "<요청문>", "task_type": "{task}", "persona": "{persona}"}}

[기사 제목] {title}
[매체/날짜/주제] {publisher} / {date} / {topic}
[기사 본문]
{body}"""

MODE_DESC = {"A": "문서 제공형 — 사용자가 이 기사를 AI 에게 붙여 넣고 요청한다. 요청문 안에서 기사를 '아래 기사' 로 지칭한다(본문은 시스템이 자동으로 뒤에 붙이므로 요청문에 본문을 복사하지 말 것).",
             "B": "영감형 — 기사는 주제를 고르기 위한 참고일 뿐이고, 사용자는 기사를 본 적이 없다. 기사 없이도 답할 수 있는 **일반적인** 질문·요청이어야 한다."}
MODE_RULES = {"A": "- 요청문에 기사 본문을 복사하지 말 것. '아래 기사를 …' 형태로 지칭.",
              "B": "- 2024년의 특정 사건·인물·기관명·수치·날짜를 언급하지 말 것(사용자는 기사를 모른다). '이날/지난달' 같은 시점 표현 금지.\n- 제도·개념·방법·의견·작문처럼 시간에 덜 묶인 요청으로."}

def load_articles(n, seed=7):
    rng = random.Random(seed); pool = {}
    for f in sorted(glob.glob(os.path.join(CORPUS, "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for doc in d.get("document") or []:
            m = doc.get("metadata") or {}; paras = [TAG.sub("", p.get("form") or "").strip() for p in (doc.get("paragraph") or [])]
            paras = [p for p in paras if p]
            if len(paras) < 3: continue
            title, body = paras[0], "\n".join(paras[1:])
            if not (500 <= len(body) <= 1600): continue
            if re.search(r"(무단 전재|재배포 금지|ⓒ|저작권자)", body[-120:]): body = re.sub(r"\s*(ⓒ|저작권자|무단 전재)[^\n]*$", "", body)
            pool.setdefault(m.get("topic") or "기타", []).append({"doc_id": doc.get("id"), "title": title, "body": body, "publisher": m.get("publisher"),
                                                                   "date": str(m.get("date") or ""), "topic": m.get("topic"), "original_topic": m.get("original_topic")})
    topics = sorted(pool); per = max(1, n // len(topics)); out = []
    for t in topics:
        rng.shuffle(pool[t]); out.extend(pool[t][:per])
    rng.shuffle(out); return out[:n]

def post(ep, body, timeout=900):
    req = urllib.request.Request(ep.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def write_prompt(writer_name, art, mode, task, persona):
    w = WRITERS[writer_name]
    instr = WRITER_INSTR.format(persona=persona, mode_desc=MODE_DESC[mode], task=task, mode_rules=MODE_RULES[mode], title=art["title"],
                                publisher=art["publisher"], date=art["date"], topic=art["topic"], body=art["body"])
    d = post(w["endpoint"], {"model": writer_name, "messages": [{"role": "user", "content": instr}], "max_tokens": 8192, **w["sampling"]})
    m = d["choices"][0]["message"]; c = (m.get("content") or "").strip()
    j = re.search(r"\{.*\}", c, flags=re.S)
    if not j: return None, "no_json", c[:200]
    try: obj = json.loads(j.group(0))
    except Exception: return None, "bad_json", c[:200]
    p = (obj.get("prompt") or "").strip()
    return p, None, c[:200]

def gate(p, mode, body):
    if not p or len(p) < 12: return "short"
    if SPECIAL.search(p): return "special_token"
    h, l, j = len(H.findall(p)), len(L.findall(p)), len(HANJA.findall(p)); t = h + l + j
    if t and j / t > 0.01: return "hanja"
    if t and h / t < 0.5: return "low_hangul"
    if mode == "A" and len(p) > 1500: return "too_long_A"
    if mode == "A" and body[:80] in p: return "body_copied"
    if mode == "B" and SPECIFIC_B.search(p): return "specific_B"
    if mode == "B" and len(p) > 1200: return "too_long_B"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", type=int, default=200); ap.add_argument("--per-article", type=int, default=2)
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--writers", default=os.environ.get("WRITERS", "glm53-flash")); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(); writers = [w.strip() for w in a.writers.split(",") if w.strip()]
    arts = load_articles(a.articles, a.seed); rng = random.Random(a.seed)
    print(f"articles={len(arts)} writers={writers}", flush=True)
    jobs = []
    for i, art in enumerate(arts):
        modes = ["A", "B"] if a.per_article >= 2 else [rng.choice(["A", "B"])]
        for k, mode in enumerate(modes[:a.per_article]):
            task = rng.choice(TASKS_A if mode == "A" else TASKS_B); persona = rng.choice(PERSONAS)
            jobs.append((writers[(i * a.per_article + k) % len(writers)], art, mode, task, persona))
    stats = {"ok": 0}; lock = threading.Lock(); t0 = time.time(); seen = set()
    rej = open(a.out.replace(".jsonl", "") + ".rejects.jsonl", "w")
    def work(wn, art, mode, task, persona):
        try: p, why, raw = write_prompt(wn, art, mode, task, persona)
        except Exception as e: return wn, art, mode, task, persona, None, f"error:{e!r}"[:150], ""
        return wn, art, mode, task, persona, p, (why or gate(p, mode, art["body"])), raw
    with open(a.out, "w") as out, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, *j) for j in jobs]
        for n, fu in enumerate(as_completed(futs), 1):
            wn, art, mode, task, persona, p, why, raw = fu.result()
            with lock:
                if not why:
                    hkey = hashlib.sha1(re.sub(r"\s+", " ", p)[:200].encode()).hexdigest()
                    if hkey in seen: why = "dup"
                    else: seen.add(hkey)
                if why:
                    stats[why.split(":")[0]] = stats.get(why.split(":")[0], 0) + 1
                    rej.write(json.dumps({"doc_id": art["doc_id"], "mode": mode, "task": task, "why": why, "prompt": p, "raw": raw}, ensure_ascii=False) + "\n")
                else:
                    stats["ok"] += 1
                    final = p + f"\n\n[기사]\n제목: {art['title']}\n{art['body']}" if mode == "A" else p
                    cid = f"nikl:{art['doc_id']}:{mode}"
                    out.write(json.dumps({"source": "nikl_news", "conv_id": cid, "language": "Korean", "mode": mode, "task_type": task, "persona": persona,
                                          "turns": [{"role": "user", "content": final}], "first_user": final, "n_user_turns": 1,
                                          "seed_origin": {"dataset": "NIKL_NEWSPAPER_2025_v1.0", "doc_id": art["doc_id"], "publisher": art["publisher"], "date": art["date"],
                                                          "topic": art["topic"], "original_topic": art["original_topic"], "writer": wn, "mode": mode},
                                          "article_ref": {"title": art["title"], "body": art["body"]}}, ensure_ascii=False) + "\n"); out.flush()
                if n % 50 == 0 or n == len(futs): print(f"[{n}/{len(futs)}] {time.time()-t0:.0f}s {stats}", flush=True)
    print("DONE", json.dumps(stats, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
