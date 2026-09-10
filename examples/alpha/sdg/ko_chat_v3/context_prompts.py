#!/usr/bin/env python3
"""context_prompts.py — S4 한국 맥락 샘플러 (kor-chat-v3 계획 §2, 17k). 폐기한 트랙 B 의 격자 설계만 계승, 교사는 reasoning 모델(thinking 켠 채).

기사 코퍼스가 못 덮는 **생활·행정·직장·금융·교육·건강** 의 실제 요청을 만든다: 도메인 × 상황 × 페르소나 × 과업 유형 × 어투 격자에서
샘플한 조합을 교사에게 주고 "그 사람이 실제로 AI 에게 보낼 요청" 을 쓰게 한다(실존 인물·특정 사건·날짜 금지 → 제도·절차 일반).
출력: generate_v3.py 시드 스키마 + seed_origin(dataset="ko_context_grid", 격자 값).
사용: python3 context_prompts.py --n 17000 --out out/p1/seeds_ctx.jsonl --workers 96 --writers glm53-flash,dsv4-flash
"""
import argparse, hashlib, json, os, random, re, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

WRITERS = {"glm53-flash": os.environ.get("GLM_EP", "http://localhost:8000/v1"), "dsv4-flash": os.environ.get("DSV4_EP", "http://sub1:8300/v1")}
H = re.compile(r"[가-힣]"); L = re.compile(r"[A-Za-z]"); HANJA = re.compile(r"[一-鿿]"); SPECIAL = re.compile(r"<\|[A-Za-z_]+\|>|</?(?:tool_call|tool_response|think)>")
SPECIFIC = re.compile(r"(20\d\d년 \d{1,2}월|이날|어제 뉴스|최근 기사|오늘 뉴스|이번 사건)")   # "지난주/지난달" 은 개인 맥락(지난주 진료·지난달 계약)이라 제외(P1 리젝 42/42 전부 정당)

DOMAINS = {
    "주거": ["전세·월세 계약", "청약·분양", "전세사기 예방", "이사·입주", "관리비·층간소음", "주택담보대출"],
    "금융·세무": ["연말정산", "종합소득세", "ISA·연금저축·IRP", "적금·예금 비교", "신용점수·대출", "가상자산 세금", "부가세 신고"],
    "행정·민원": ["주민센터 서류", "정부24·홈택스 이용", "여권·비자", "병역·예비군", "출생신고·가족관계", "교통 과태료·범칙금", "실업급여"],
    "직장": ["이직·퇴사 절차", "연차·휴가·수당", "근로계약서 검토", "보고서·기획안 작성", "회의록·이메일", "성과 평가·면담", "동료 갈등"],
    "교육·입시": ["수시·정시 전략", "학원·과외 선택", "자녀 학습 습관", "대학원 진학", "자격증 준비", "영어·한국어 공부법"],
    "건강·의료": ["건강검진 결과 해석", "실손보험 청구", "병원 진료과 선택", "식단·운동 계획", "육아·수유", "노부모 돌봄"],
    "생활·소비": ["가전·자동차 구매 비교", "여행 계획(국내)", "명절·경조사 예절", "요리·레시피", "반려동물", "중고거래 분쟁"],
    "창업·사업": ["사업자등록·세금", "소상공인 지원사업", "온라인 쇼핑몰 운영", "프랜차이즈 검토", "마케팅 문구"],
    "법률·권리": ["임대차 분쟁", "층간소음·이웃 분쟁", "소비자 환불", "교통사고 처리", "근로기준법 위반 신고"],
    "기술·IT 활용": ["엑셀·스프레드시트", "파이썬 자동화", "AI 도구 활용", "스마트폰·PC 문제 해결", "개인정보 보호"],
}
PERSONAS = ["20대 사회초년생", "30대 맞벌이 부모", "40대 자영업자", "50대 직장인", "60대 은퇴자", "대학생", "고등학생 자녀를 둔 학부모",
            "지방 소도시 거주자", "1인 가구", "한국에 정착한 외국인(한국어 능숙)", "프리랜서", "공무원", "간호사", "초등교사", "스타트업 초기 팀원"]
TASKS = ["절차·방법을 단계별로 안내", "선택지 비교와 추천", "서류·이메일·문자 초안 작성", "내 상황이 맞는지 판단 요청", "주의할 점·실수 목록",
         "비용·기간 추정", "체크리스트 작성", "간단한 표로 정리", "쉬운 말로 개념 설명", "협상·대화 스크립트"]
TONES = ["반말 짧게", "존댓말 정중하게", "급한 상황 설명 후 질문", "배경을 길게 설명한 뒤 구체적 질문", "여러 질문을 한 번에"]

INSTR = """당신은 한국어 AI 어시스턴트 학습용 데이터를 만드는 작가입니다. 아래 조건의 사람이 **실제로 AI 어시스턴트에게 보낼 법한 한국어 요청문 하나**를 쓰세요.

- 화자: {persona}
- 도메인/상황: {domain} — {situation}
- 과업 유형: {task}
- 어투/형식: {tone}

규칙
- 실제 사람이 쓴 것처럼 구체적인 자기 상황(숫자·조건·제약)을 담되, 실존 인물·특정 사건·특정 날짜의 뉴스는 넣지 말 것.
- 인위적 표시("다음 조건에 따라")나 메타 설명 없이 요청문만.
- 길이는 어투에 맞게. 여러 질문이면 자연스럽게 나열.

출력은 아래 JSON 하나만 (코드펜스 금지):
{{"prompt": "<요청문>"}}"""

def post(ep, body, timeout=900):
    req = urllib.request.Request(ep.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def write(wn, combo):
    d = post(WRITERS[wn], {"model": wn, "messages": [{"role": "user", "content": INSTR.format(**combo)}], "max_tokens": 6144, "temperature": 0.9, "top_p": 0.95})
    c = (d["choices"][0]["message"].get("content") or "").strip(); j = re.search(r"\{.*\}", c, flags=re.S)
    if not j: return None, "no_json"
    try: p = (json.loads(j.group(0)).get("prompt") or "").strip()
    except Exception: return None, "bad_json"
    return p, None

def gate(p):
    if not p or len(p) < 12: return "short"
    if SPECIAL.search(p): return "special_token"
    h, l, j = len(H.findall(p)), len(L.findall(p)), len(HANJA.findall(p)); t = h + l + j
    if t and j / t > 0.01: return "hanja"
    if t and h / t < 0.6: return "low_hangul"
    if SPECIFIC.search(p): return "specific"
    if len(p) > 1500: return "too_long"
    return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=17000); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=96); ap.add_argument("--writers", default="glm53-flash,dsv4-flash"); ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args(); writers = [w for w in a.writers.split(",") if w]; rng = random.Random(a.seed)
    combos = []
    for i in range(a.n):
        dom = rng.choice(list(DOMAINS)); combos.append({"persona": rng.choice(PERSONAS), "domain": dom, "situation": rng.choice(DOMAINS[dom]), "task": rng.choice(TASKS), "tone": rng.choice(TONES)})
    stats = {"ok": 0}; lock = threading.Lock(); t0 = time.time(); seen = set()
    rej = open(a.out.replace(".jsonl", "") + ".rejects.jsonl", "w")
    def work(i, combo):
        wn = writers[i % len(writers)]
        try: p, why = write(wn, combo)
        except Exception as e: return wn, combo, None, f"error:{e!r}"[:120]
        return wn, combo, p, (why or gate(p))
    with open(a.out, "w") as out, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, i, c) for i, c in enumerate(combos)]
        for n, fu in enumerate(as_completed(futs), 1):
            wn, combo, p, why = fu.result()
            with lock:
                if not why:
                    k = hashlib.sha1(re.sub(r"\s+", " ", p)[:160].encode()).hexdigest()
                    if k in seen: why = "dup"
                    else: seen.add(k)
                if why:
                    stats[why.split(":")[0]] = stats.get(why.split(":")[0], 0) + 1; rej.write(json.dumps({"combo": combo, "why": why, "prompt": p}, ensure_ascii=False) + "\n")
                else:
                    stats["ok"] += 1; cid = "ctx:" + hashlib.sha1(json.dumps(combo, ensure_ascii=False).encode() + p[:80].encode()).hexdigest()[:16]
                    out.write(json.dumps({"source": "ko_context_grid", "conv_id": cid, "language": "Korean", "turns": [{"role": "user", "content": p}], "first_user": p, "n_user_turns": 1,
                                          "seed_origin": {"dataset": "ko_context_grid", "writer": wn, **combo}}, ensure_ascii=False) + "\n"); out.flush()
                if n % 500 == 0 or n == len(futs): print(f"[{n}/{len(futs)}] {time.time()-t0:.0f}s {stats}", flush=True)
    print("DONE", json.dumps(stats, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
