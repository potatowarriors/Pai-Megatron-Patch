#!/usr/bin/env python3
"""build_seeds_localized.py — ko_chat v3 주력 시드: NVIDIA Chat-v3 복원 영어 프롬프트(637k)를 한국인 사용자의 한국어 요청으로 현지화.

왜: 실사용자 한국어 대화는 lmsys+WildChat 을 합쳐도 ≈4k 행이고 절반이 10토큰 이하 단문 → r1 이 단순해진 원인.
    NVIDIA 가 쓴 영어 인간 프롬프트를 "번역"이 아니라 "한국 사용자가 이렇게 물었을" 요청으로 재작성해 시드로 쓴다
    (2026-09-10 실험: 같은 프롬프트라도 한국어면 교사 사고가 1/3 → 생성 단계에서 "철저 사고" 지시로 회복).

파이프라인: 복잡도 필터(짧은 인사·단답·정체성 질문 제외) → 중복 제거 → GLM no-think 현지화 → 게이트(한글비·코드펜스 보존·
special-token·벤더명 유입·길이 비율) → seeds jsonl (generate_v3.py 입력 스키마: turns/first_user/conv_id + seed_origin 출처 태그).

단일턴 행만 1차 대상(Chat-v3 의 ≈27%). 멀티턴 현지화(히스토리 재생성)는 2단계.
사용: python3 build_seeds_localized.py --n 120000 --out out/seeds_ko_cv3loc.jsonl --workers 64
"""
import argparse, json, re, hashlib, os, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

SRC = "/home/work/Datasets/LL_datasets/posttraining/SFT/Nemotron-SFT-Instruction-Following-Chat-v3/data/chat.with_prompts.jsonl"
# 현지화 교사(2026-09-10 스모크 실측): GLM 은 chat_template_kwargs enable_thinking=False 를 무시하고 사고를 계속해 사고 텍스트가
# content 로 새거나 예산이 잘림(0/200 통과). Qwen3.8-Flash-Next 는 enable_thinking=False 가 문서화·동작 → 기본 현지화 교사.
# LOCALIZER=glm 이면 GLM 을 사고 허용 상태로 쓰고(파서가 분리) max_tokens 를 키운다.
LOCALIZERS = {
    "qwen": {"model": "qwen38-flash-next", "endpoint": "http://sub1:8300/v1", "max_tokens": 2048,
             "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 0.0},   # Qwen non-thinking best practice
             "extra": {"chat_template_kwargs": {"enable_thinking": False}}},
    "glm": {"model": "glm53-flash", "endpoint": "http://localhost:8000/v1", "max_tokens": 6144,
            "sampling": {"temperature": 0.3, "top_p": 0.9}, "extra": {}},
}
LOC = LOCALIZERS[os.environ.get("LOCALIZER", "qwen")]
H = re.compile(r"[가-힣]"); L = re.compile(r"[A-Za-z]"); HANJA = re.compile(r"[一-鿿]")
SPECIAL = re.compile(r"<\|[A-Za-z_]+\|>|</?(?:tool_call|tool_response|think)>")
FENCE = re.compile(r"```")
TRIVIAL = re.compile(r"^(hi|hello|hey|thanks?|thank you|ok|okay|yes|no|test|hello there|good morning|who are you|what are you|what is your name|are you (chatgpt|gpt|an ai))\W*$", re.I)
IDENT = re.compile(r"\b(who (made|created|built|developed) you|your name|are you (chatgpt|gpt-?4|claude|gemini|bard|an ai))\b", re.I)
INSTR = ("아래 사용자 요청(어떤 언어로 쓰였든)을, 한국인 사용자가 한국어 AI 어시스턴트에게 자연스럽게 썼을 한국어 요청으로 다시 써라.\n"
         "- 의미·요구사항·제약·난이도는 그대로 유지한다. 번역투가 아니라 한국 사용자의 실제 말투로.\n"
         "- 인물·지명·통화·서비스·법제도 등 맥락은 필요할 때만 한국 맥락으로 바꾼다(문제의 본질이 바뀌지 않는 범위).\n"
         "- 코드·수식·고유 기술용어·프로그래밍 언어명·영문 약어는 원문 유지. 코드 블록 개수를 유지한다.\n"
         "- 설명·따옴표·머리말 없이 재작성한 한국어 요청만 출력한다.\n\n[원문 요청]\n")

def ratios(s):
    h, l, j = len(H.findall(s)), len(L.findall(s)), len(HANJA.findall(s)); t = h + l + j
    return (h / t if t else 0.0, j / t if t else 0.0)

def strip_code(s): return re.sub(r"```.*?```", "", s, flags=re.S)

def localize(prompt):
    body = {"model": LOC["model"], "messages": [{"role": "user", "content": INSTR + prompt}], "max_tokens": LOC["max_tokens"],
            **LOC["sampling"], **LOC["extra"]}
    req = urllib.request.Request(LOC["endpoint"] + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600)); m = d["choices"][0]["message"]
    return (m.get("content") or "").strip()    # reasoning 필드는 버림(파서 분리) — content 만 시드

def gate(en, ko):
    if not ko or len(ko) < 8: return "empty"
    if SPECIAL.search(ko): return "special_token"
    hk, hj = ratios(strip_code(ko))
    if hj > 0.01: return "hanja"
    if hk < 0.4 and not FENCE.search(en): return "low_hangul"
    if len(FENCE.findall(ko)) != len(FENCE.findall(en)): return "codefence_mismatch"
    r = len(ko) / max(len(en), 1)
    if r < 0.25 or r > 4.0: return "length_ratio"
    if re.search(r"(다시 써|재작성|영어 요청|\[영어)", ko): return "instruction_leak"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120000); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64); ap.add_argument("--min-chars", type=int, default=60)
    ap.add_argument("--max-chars", type=int, default=4000); ap.add_argument("--multi-turn", action="store_true", help="멀티턴 행도 포함(마지막 user 만 현지화·히스토리는 영어 유지 — 2단계 전용)")
    ap.add_argument("--exclude-origin", default="", help="쉼표 구분 seed_dataset 접두(예: lmsys) — 해당 유래 행 제외. lmsys 유래는 파생 셋 재배포 금지라 사용자 확정 전 제외 가능")
    a = ap.parse_args()
    excl = [x.strip().lower() for x in a.exclude_origin.split(",") if x.strip()]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: done.add(json.loads(l)["conv_id"])
            except Exception: pass
    cands, seen = [], set()
    with open(SRC) as f:
        for line in f:
            d = json.loads(line); m = [x for x in d["messages"] if x["role"] in ("user", "assistant")]
            if not m or m[0]["role"] != "user": continue
            if not a.multi_turn and len(m) != 2: continue
            p = m[0]["content"]
            if not isinstance(p, str): continue
            if not (a.min_chars <= len(p) <= a.max_chars): continue
            if TRIVIAL.match(p.strip()) or IDENT.search(p) or SPECIAL.search(p): continue
            sd = ((d.get("metadata") or {}).get("seed_dataset") or "").lower()
            if excl and any(sd.startswith(x) for x in excl): continue
            if not L.search(p): continue                      # 영어 원문이어야 현지화 의미가 있음
            h = hashlib.sha1(re.sub(r"\s+", " ", p.lower())[:500].encode()).hexdigest()
            if h in seen: continue
            seen.add(h)
            cid = "cv3loc:" + h[:16]
            if cid in done: continue
            cands.append((cid, p, d))
            if len(cands) >= a.n - len(done): break
    print(f"candidates={len(cands)} (done={len(done)})", flush=True)
    stats = {"ok": 0}; lock = threading.Lock(); t0 = time.time()
    rej_f = open(a.out.replace(".jsonl", "") + ".rejects.jsonl", "a")
    def work(cid, p, d):
        try: ko = localize(p)
        except Exception as e: return cid, None, f"error:{e!r}"[:120], p, d
        return cid, ko, gate(p, ko), p, d
    with open(a.out, "a") as out, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, *c) for c in cands]
        for i, fu in enumerate(as_completed(futs), 1):
            cid, ko, why, p, d = fu.result()
            with lock:
                if why:
                    stats[why.split(":")[0]] = stats.get(why.split(":")[0], 0) + 1
                    rej_f.write(json.dumps({"conv_id": cid, "why": why, "en": p[:500], "ko": (ko or "")[:500]}, ensure_ascii=False) + "\n")
                else:
                    stats["ok"] += 1
                    out.write(json.dumps({"source": "chat_v3_localized", "conv_id": cid, "language": "Korean",
                                          "turns": [{"role": "user", "content": ko}], "first_user": ko, "n_user_turns": 1,
                                          # 출처 태그: Chat-v3 metadata.seed_dataset (lmsys/lmsys-chat-1m · allenai/WildChat-1M · lmarena-ai/…) — lmsys 유래는 파생 셋 재배포 금지(라이선스)라 행 단위로 분리 가능해야 한다
                                          "seed_origin": {"dataset": "Nemotron-SFT-Instruction-Following-Chat-v3", "seed_dataset": (d.get("metadata") or {}).get("seed_dataset") or "unknown",
                                                          "seed_source": (d.get("metadata") or {}).get("seed_source"), "uuid": d.get("uuid"), "en_sha1": cid[7:]},
                                          "en_prompt": p}, ensure_ascii=False) + "\n"); out.flush()
                if i % 500 == 0 or i == len(futs):
                    el = time.time() - t0; print(f"[{i}/{len(futs)}] {el:.0f}s {stats}", flush=True)
    print("DONE", json.dumps(stats), flush=True)

if __name__ == "__main__":
    main()
