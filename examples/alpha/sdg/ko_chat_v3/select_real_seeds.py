#!/usr/bin/env python3
"""select_real_seeds.py — S3 실사용자 한국어 시드 선별 (kor-chat-v3 P1, 목표 3k). 입력: extract_ko_seeds.py 산출(lmsys 2,561 · WildChat-1M 2,833).
왜: 실사용자 행은 절반이 10토큰 이하 단문(r1 이 단순해진 원인) → 복잡도 필터로 걸러 "실제 사용자 분포의 어려운 꼬리"만 남긴다.
게이트: 길이(기본 40자) · 한글비 ≥0.5(코드펜스 예외) · 한자 ≤1% · 정체성/인사 제외 · special token · 근사 중복. lmsys 유래는 사내 연구 전용(재배포 금지) 태그.
사용: python3 select_real_seeds.py --out out/p1/seeds_real.jsonl [--min-chars 40] [--cap 3000]
"""
import argparse, json, re, hashlib, collections, random
H = re.compile(r"[가-힣]"); L = re.compile(r"[A-Za-z]"); HANJA = re.compile(r"[一-鿿]"); FENCE = re.compile(r"```")
SPECIAL = re.compile(r"<\|[A-Za-z_]+\|>|</?(?:tool_call|tool_response|think)>")
IDENT = re.compile(r"(누가 만들|누가 개발|너는 누구|넌 누구|당신은 누구|이름이 뭐|chatgpt|gpt-?4|클로드|claude|제미나이|gemini|바드|bard)", re.I)
VENDOR_HIST = re.compile(r"(chatgpt|openai|gpt-?[34]|claude|anthropic|gemini|bard|구글|google|오픈에이아이|챗지피티)", re.I)   # 히스토리 assistant 턴의 타 벤더 자기귀속(원 모델 답) 제외
TRIVIAL = re.compile(r"^(안녕|하이|헬로|ㅎㅇ|고마워|감사|ok|test|테스트|ㅋ+|ㅎ+|\.+)\W*$", re.I)
SRC = {"lmsys": ("lmsys/lmsys-chat-1m", "internal-research-only"), "wildchat1m": ("allenai/WildChat-1M", "ODC-By")}
def gate(p, a):
    if not p or len(p) < a.min_chars: return "short"
    if len(p) > a.max_chars: return "long"
    if SPECIAL.search(p): return "special_token"
    if TRIVIAL.match(p.strip()) or IDENT.search(p): return "trivial_or_identity"
    body = re.sub(r"```.*?```", "", p, flags=re.S); h, l, j = len(H.findall(body)), len(L.findall(body)), len(HANJA.findall(body)); t = h + l + j
    if t and j / t > 0.01: return "hanja"
    if t and h / t < 0.5 and not FENCE.search(p): return "low_hangul"
    return None
def hist_gate(r):
    return "vendor_in_history" if any(t["role"] == "assistant" and VENDOR_HIST.search(t.get("content") or "") for t in r["turns"]) else None
ap = argparse.ArgumentParser(); ap.add_argument("--inputs", default="out/seeds_ko_pilot.jsonl,out/seeds_ko_wildchat1m.jsonl"); ap.add_argument("--out", required=True)
ap.add_argument("--min-chars", type=int, default=25); ap.add_argument("--max-chars", type=int, default=6000); ap.add_argument("--cap", type=int, default=3000); ap.add_argument("--seed", type=int, default=3)
a = ap.parse_args(); stats = collections.Counter(); seen = set(); keep = []
for path in a.inputs.split(","):
    for line in open(path):
        r = json.loads(line); p = r["first_user"]; why = gate(p, a) or hist_gate(r)
        if not why:
            k = hashlib.sha1(re.sub(r"\s+", " ", p.lower())[:160].encode()).hexdigest()
            why = "dup" if k in seen else None; seen.add(k)
        stats[(r["source"], why or "ok")] += 1
        if why: continue
        ds, lic = SRC[r["source"]]
        keep.append({**r, "seed_origin": {"dataset": ds, "license_note": lic, "orig_conv_id": r["conv_id"]}})
random.Random(a.seed).shuffle(keep)
if len(keep) > a.cap: keep = keep[:a.cap]
with open(a.out, "w") as f:
    for r in keep: f.write(json.dumps(r, ensure_ascii=False) + "\n")
by = collections.Counter(r["source"] for r in keep); L2 = sorted(len(r["first_user"]) for r in keep)
print("kept", len(keep), dict(by), "chars p50", L2[len(L2)//2], "| stats", {f"{s}/{w}": n for (s, w), n in sorted(stats.items())})
