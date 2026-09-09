#!/usr/bin/env python3
"""generate_v3.py — ko_chat v3 본 생성기 (NVIDIA Chat-v3 레시피 + 이번 사고 교정).

시드(실사용자 한국어 대화) → 마지막 assistant 턴을 reasoning 교사가 네이티브 사고와 함께 재생성.
  - 교사 2종 라운드로빈(GLM-5.3-Flash @main1, Qwen3.8-Flash-Next @sub1). 생성 N=2 → **상대 교사가 쌍대 심판**(교차검증).
  - identity 카드 시스템 주입(생성 시에만; 학습 행에는 넣지 않음 — identity 셋이 별도).
  - 게이트: 빈 답변/미완결(finish≠stop) 리젝 · **언어 게이트(사고·답변의 한자/중국어 혼입 리젝; 영어·영한 사고 허용)** ·
    벤더 **자기귀속** 누출 리젝(블랭킷 벤더명 아님 — "클로드 마켈렐레"·"OpenAI CLIP" 오탐 방지) · special-token 리터럴 리젝.
  - 재개 가능: 출력 jsonl 의 conv_id 를 읽어 건너뜀. 리젝은 rejects jsonl 에 사유와 함께 보존.
출력 행 = Chat-v3 스키마(messages, 마지막 assistant 에 reasoning_content) + ko_synthesis 메타. 마지막 턴만 학습(변환기 규약).

사용: python3 generate_v3.py --seeds out/seeds_ko_pilot.jsonl --out out/v3_r1.jsonl --workers 96 [--limit N]
"""
import argparse, json, os, re, sys, time, threading, random, urllib.request, yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
CARD = yaml.safe_load(open(os.path.join(HERE, "..", "identity", "identity_card.yaml")))
NAME, ORG = CARD["name"]["canonical"], CARD["organization"]["ko"]
SYS_IDENTITY = (f"당신은 {ORG}에서 개발한 AI 어시스턴트 {NAME}입니다. 사용자에게 정확하고 도움이 되는 답변을 "
                f"한국어로 제공합니다. 다른 회사나 다른 모델의 이름으로 자신을 소개하지 않습니다.")

TEACHERS = [
    {"name": "glm53-flash", "endpoint": "http://localhost:8000/v1", "model": "glm53-flash",
     "sampling": {"temperature": 1.0, "top_p": 0.95}},
    {"name": "qwen38-flash-next", "endpoint": "http://sub1:8300/v1", "model": "qwen38-flash-next",
     "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}},  # HF best practice (thinking)
]

H = re.compile(r"[가-힣]"); L = re.compile(r"[A-Za-z]"); HANJA = re.compile(r"[一-鿿㐀-䶿]")
SPECIAL = re.compile(r"<\|[A-Za-z_]+\|>|</?(?:tool_call|tool_response|think)>")
VENDORS = r"(Qwen|통이치엔원|GLM|Zhipu|智谱|Alibaba|알리바바|Google|구글|Gemini|제미나이|Gemma|OpenAI|오픈AI|ChatGPT|GPT-4|GPT-5|Anthropic|앤트로픽|Claude|클로드|DeepSeek|딥시크|Moonshot|Kimi|Mistral|Llama|Meta AI)"
SELF_ATTR = re.compile(
    r"(저는|나는|제가|I am|I'm|I’m|as an?|this is)\s*[^.\n]{0,25}?" + VENDORS + r"|"
    r"(developed|trained|created|made|built)\s+by\s+" + VENDORS + r"|"
    + VENDORS + r"\s*(이|가|에서|에 의해|팀이)\s*(개발|훈련|만든|만들|제작)", re.I)

def ratios(s):
    h, l, j = len(H.findall(s)), len(L.findall(s)), len(HANJA.findall(s)); t = h + l + j
    return (h / t if t else 0.0, j / t if t else 0.0)

def post(ep, body, timeout=900):
    req = urllib.request.Request(ep.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def gen(teacher, messages, max_tokens):
    body = {"model": teacher["model"], "messages": messages, "max_tokens": max_tokens, **teacher["sampling"]}
    d = post(teacher["endpoint"], body); ch = d["choices"][0]; m = ch["message"]
    return {"reasoning": m.get("reasoning") or m.get("reasoning_content") or "", "content": m.get("content") or "",
            "finish": ch["finish_reason"], "ctoks": d.get("usage", {}).get("completion_tokens", 0)}

def gate(s):
    """리젝 사유 문자열 or None."""
    r, c = s["reasoning"], s["content"]
    if not c.strip(): return "empty_content"
    if s["finish"] != "stop": return f"finish_{s['finish']}"
    if not r.strip(): return "empty_reasoning"
    if SPECIAL.search(r) or SPECIAL.search(c): return "special_token"
    _, rj = ratios(r); _, cj = ratios(c)
    if rj > 0.02: return "reasoning_chinese"      # 중국어 혼입 사고 (사용자 규칙: 영어·영한만 허용)
    if cj > 0.01: return "content_chinese"
    if SELF_ATTR.search(r) or SELF_ATTR.search(c): return "vendor_self_attribution"
    if degenerate(r): return "reasoning_degenerate"   # "We need call now. Let's call." 류 반복 퇴행 (스모크 실측)
    return None

def degenerate(text):
    """짧은 문장/구의 과도 반복 = 퇴행 사고. 4-gram 이 5회 이상 반복되거나 고유 문장 비율이 40% 미만이면 True."""
    toks = text.split()
    if len(toks) >= 40:
        grams = {}
        for i in range(len(toks) - 3):
            g = " ".join(toks[i:i + 4]); grams[g] = grams.get(g, 0) + 1
        if max(grams.values()) >= 5: return True
    sents = [s.strip() for s in re.split(r"[.!?。\n]+", text) if len(s.strip()) > 3]
    if len(sents) >= 8 and len(set(sents)) / len(sents) < 0.4: return True
    return False

JUDGE_PROMPT = """다음은 한국어 사용자 대화와 두 개의 후보 답변(A, B)입니다. 정확성·도움됨·요청 준수·한국어 자연스러움을 기준으로 더 나은 답변 하나를 고르세요.
마지막 줄에 반드시 "판정: A" 또는 "판정: B" 만 쓰세요.

[대화]
{conv}

[답변 A]
{a}

[답변 B]
{b}"""

def judge(judge_teacher, conv_text, a, b):
    body = {"model": judge_teacher["model"], "max_tokens": 4096, "temperature": 0.0,
            "messages": [{"role": "user", "content": JUDGE_PROMPT.format(conv=conv_text[:6000], a=a[:6000], b=b[:6000])}]}
    try:
        d = post(judge_teacher["endpoint"], body); c = (d["choices"][0]["message"].get("content") or "")
        m = re.findall(r"판정\s*[:：]\s*([AB])", c)
        return (m[-1] if m else None), c[-200:]
    except Exception as e:
        return None, f"judge_error:{e!r}"[:200]

def build_messages(seed):
    turns = [t for t in seed["turns"] if t["role"] in ("user", "assistant")]
    while turns and turns[-1]["role"] == "assistant": turns.pop()      # 마지막 assistant 는 재생성 대상
    if not turns or turns[-1]["role"] != "user": return None
    # 히스토리 assistant 턴 중 빈/특수토큰 행 방어
    for t in turns:
        if SPECIAL.search(t["content"] or ""): return None
    return turns

def process(idx, seed, args, stats, lock, out_f, rej_f):
    turns = build_messages(seed)
    if turns is None:
        with lock: stats["skip_seed"] += 1
        return
    gi = idx % 2; teacher, judge_t = TEACHERS[gi], TEACHERS[1 - gi]
    msgs = [{"role": "system", "content": SYS_IDENTITY}] + turns
    samples, rejects = [], []
    for k in range(args.n):
        try: s = gen(teacher, msgs, args.max_tokens)
        except Exception as e:
            rejects.append({"k": k, "why": f"gen_error:{e!r}"[:200]}); continue
        why = gate(s)
        if why: rejects.append({"k": k, "why": why, "reasoning": s["reasoning"][:300], "content": s["content"][:300]})
        else: samples.append(s)
    rec_base = {"source": seed["source"], "conv_id": seed["conv_id"], "teacher": teacher["name"]}
    if not samples:
        with lock:
            stats["reject_all"] += 1
            rej_f.write(json.dumps({**rec_base, "rejects": rejects}, ensure_ascii=False) + "\n"); rej_f.flush()
        return
    verdict, jtail = None, ""
    if len(samples) >= 2:
        conv_text = "\n".join(f"{t['role']}: {t['content'][:1500]}" for t in turns)
        verdict, jtail = judge(judge_t, conv_text, samples[0]["content"], samples[1]["content"])
    best = samples[1] if verdict == "B" else samples[0]
    loser = (samples[0] if verdict == "B" else samples[1]) if len(samples) >= 2 else None
    row = {"messages": turns + [{"role": "assistant", "content": best["content"], "reasoning_content": best["reasoning"]}],
           **rec_base,
           "ko_synthesis": {"pipeline": "ko_chat_v3", "think": True, "n_samples": args.n, "kept": len(samples),
                            "judge": judge_t["name"] if len(samples) >= 2 else None, "verdict": verdict,
                            "judge_tail": jtail, "loser_content": (loser["content"][:2000] if loser else None),
                            "rejects": [{"k": r["k"], "why": r["why"]} for r in rejects],
                            "r_hangul": round(ratios(best["reasoning"])[0], 2), "ctoks": best["ctoks"]}}
    with lock:
        stats["ok"] += 1; stats["toks"] += sum(s["ctoks"] for s in samples)
        if verdict is None and len(samples) >= 2: stats["judge_fail"] += 1
        for r in rejects: stats["rej_" + r["why"].split(":")[0]] = stats.get("rej_" + r["why"].split(":")[0], 0) + 1
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n"); out_f.flush()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=96); ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=12288); ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try: done.add(json.loads(l)["conv_id"])
            except Exception: pass
    seeds = [json.loads(l) for l in open(args.seeds)]
    if args.limit: seeds = seeds[:args.limit]
    todo = [(i, s) for i, s in enumerate(seeds) if s["conv_id"] not in done]
    print(f"seeds={len(seeds)} done={len(done)} todo={len(todo)} workers={args.workers} n={args.n}", flush=True)
    stats = {"ok": 0, "reject_all": 0, "skip_seed": 0, "judge_fail": 0, "toks": 0}; lock = threading.Lock(); t0 = time.time()
    rej_path = args.out.replace(".jsonl", "") + ".rejects.jsonl"
    with open(args.out, "a") as out_f, open(rej_path, "a") as rej_f, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process, i, s, args, stats, lock, out_f, rej_f) for i, s in todo]
        for n, f in enumerate(as_completed(futs), 1):
            try: f.result()
            except Exception as e:
                with lock: stats["worker_error"] = stats.get("worker_error", 0) + 1
                print("worker_error", repr(e)[:200], flush=True)
            if n % 50 == 0 or n == len(futs):
                el = time.time() - t0
                print(f"[{n}/{len(futs)}] {el:.0f}s ok={stats['ok']} reject_all={stats['reject_all']} judge_fail={stats['judge_fail']} "
                      f"tok/s={stats['toks']/max(el,1):.0f} | " + " ".join(f"{k}={v}" for k, v in stats.items() if k.startswith('rej_')), flush=True)
    print("DONE", json.dumps(stats, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
