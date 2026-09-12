#!/usr/bin/env python3
"""u_generate.py — 트랙 U(구조화 출력·사용성) 생성기. EN 80 / KO 20 (사용자 결정 2026-09-12).

흐름: 시드(EN: Chat-v3 복원 영어 프롬프트 / KO: P1 한국어 시드) → 제약 템플릿 결합(u_templates) → 교사 라운드로빈(GLM v2 지시문·DSV4) n=2
      → **프로그램 검증기** 통과 샘플만 후보 → 후보 2개면 상대 교사 심판, 1개면 그대로 → 게이트(generate_v3.gate: 중국어·자기귀속·special·퇴행)
      → 행(Chat-v3 스키마 + u_meta). 후속턴(followup) 은 1턴 답변을 먼저 만들고 변환 지시를 2턴으로 붙여 마지막 턴만 학습.
사용: GEN_THOROUGH=2 python3 u_generate.py --n 100 --out out/p0/u_p0.jsonl --workers 96 [--en-ratio 0.8] [--seed 1]
"""
import argparse, json, os, sys, re, random, threading, time, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "ko_chat_v3"))
import generate_v3 as g
from u_templates import pick_template, t_clarify, lang_ratio

CV3 = "/home/work/Datasets/LL_datasets/posttraining/SFT/Nemotron-SFT-Instruction-Following-Chat-v3/data/chat.with_prompts.jsonl"
KO_SEEDS = [os.path.join(HERE, "..", "ko_chat_v3", "out", "p1", f) for f in ("seeds_p1_A.jsonl", "seeds_p1_B.jsonl")]
NAME, ORG_EN = g.NAME, g.CARD["organization"]["en"]
SYS_EN = (f"You are {NAME}, an AI assistant developed by {ORG_EN}. Give accurate, helpful answers and never present yourself under another company's or model's name."
          + (" Before answering, re-state the request in your reasoning, consider what the user actually needs from several angles, weigh alternatives and edge cases, and verify key facts. Do not pre-write the final answer inside your reasoning; keep reasoning to points, structure, and checks."
             if os.environ.get("GEN_THOROUGH", "2") != "0" else ""))
SYS_KO = g.SYS_GEN
JUDGE_U = """Below is a conversation and two candidate final responses (A and B). Both were checked against the explicit format/length constraints; judge which one is better overall on: correctness, helpfulness, faithfulness to every instruction in the request, and natural, fluent writing in the language the response is supposed to be in. Ignore surface length unless the request constrains it.
On the last line write exactly "판정: A" or "판정: B".

[Conversation]
{conv}

[Response A]
{a}

[Response B]
{b}"""
IDENT = re.compile(r"\b(who (made|created|built|developed) you|your name|are you (chatgpt|gpt-?4|claude|gemini|bard|an ai))\b", re.I)
SPECIAL = g.SPECIAL

def load_en(n, rng, short=False):
    out = []
    with open(CV3) as f:
        for line in f:
            d = json.loads(line); m = [x for x in d["messages"] if x["role"] in ("user", "assistant")]
            if len(m) != 2 or m[0]["role"] != "user": continue
            p = m[0]["content"]
            if not isinstance(p, str) or SPECIAL.search(p) or IDENT.search(p): continue
            h, l = lang_ratio(p)
            if l < 0.9: continue
            L = len(p)
            if short and not (15 <= L <= 90): continue
            if not short and not (40 <= L <= 1500): continue
            out.append({"conv_id": "u_en:" + hashlib.sha1(p.encode()).hexdigest()[:16], "prompt": p, "seed_origin": {"dataset": "Nemotron-SFT-Instruction-Following-Chat-v3", "seed_dataset": (d.get("metadata") or {}).get("seed_dataset"), "uuid": d.get("uuid")}})
            if len(out) >= n * 8: break
    rng.shuffle(out); return out[:n]

def load_ko(n, rng, short=False):
    out = []
    for f in KO_SEEDS:
        for line in open(f):
            s = json.loads(line); p = s["first_user"]; so = s.get("seed_origin") or {}
            if s.get("n_user_turns", 1) != 1: continue
            if so.get("mode") == "A" and rng.random() > 0.2: continue      # 기사 첨부형은 20% 만
            L = len(p)
            if short and not (8 <= L <= 40): continue
            if not short and not (20 <= L <= 1200): continue
            out.append({"conv_id": "u_ko:" + s["conv_id"], "prompt": p, "seed_origin": so})
    rng.shuffle(out); return out[:n]

def build(spec, lang):
    sys_ = SYS_KO if lang == "ko" else SYS_EN
    return [{"role": "system", "content": sys_}, {"role": "user", "content": spec["user"]}]

def one_sample(teacher, msgs, max_tokens):
    s = g.gen(teacher, msgs, max_tokens); why = g.gate(s)
    return s, why

def judge_u(judge_t, conv_text, a, b):
    swap = random.random() < 0.5; x, y = (b, a) if swap else (a, b)
    kw, budget = g.JUDGE_KW.get(judge_t["name"], ({}, 4096))
    body = {"model": judge_t["model"], "max_tokens": budget, "temperature": 0.0, **kw, "messages": [{"role": "user", "content": JUDGE_U.format(conv=conv_text[:6000], a=x[:6000], b=y[:6000])}]}
    try:
        d = g.post(judge_t["endpoint"], body); c = (d["choices"][0]["message"].get("content") or "")
        m = re.findall(r"판정\s*[:：]\s*([AB])", c)
        if not m: return None, ("empty" if not c.strip() else c[-160:])
        v = m[-1]; v = ("B" if v == "A" else "A") if swap else v
        return v, c[-160:]
    except Exception as e:
        return None, f"judge_error:{e!r}"[:160]

def process(idx, seed, lang, spec, args, stats, lock, out_f, rej_f):
    teacher, judge_t = g.TEACHERS[idx % len(g.TEACHERS)], g.pick_judge(idx)
    msgs = build(spec, lang); history = []; prev_answer = None
    if spec["kind"] == "followup":
        # 1턴: 원 요청에 대한 답(검증 없음, 게이트만) → 히스토리; 2턴 = 변환 지시
        try: s1, why1 = one_sample(teacher, msgs, args.max_tokens)
        except Exception as e: why1, s1 = f"gen_error:{e!r}"[:120], None
        if why1:
            with lock: stats["reject_all"] += 1; stats["rej_t1_" + why1.split(":")[0]] = stats.get("rej_t1_" + why1.split(":")[0], 0) + 1
            rej_f.write(json.dumps({"conv_id": seed["conv_id"], "why": "turn1:" + why1, "meta": spec["meta"]}, ensure_ascii=False) + "\n"); return
        prev_answer = s1["content"]
        history = [{"role": "user", "content": spec["user"]}, {"role": "assistant", "content": s1["content"]}, {"role": "user", "content": spec["followup"]}]
        msgs = [msgs[0]] + history
        validate = lambda c: spec["validate2"](c, prev_answer)
    else:
        history = [{"role": "user", "content": spec["user"]}]; validate = spec["validate"]
    samples, rejects = [], []
    for k in range(args.n_samples):
        try: s, why = one_sample(teacher, msgs, args.max_tokens)
        except Exception as e: rejects.append({"k": k, "why": f"gen_error:{e!r}"[:120]}); continue
        if not why:
            ok, vwhy = validate(s["content"]); why = None if ok else "validator:" + str(vwhy)
        if why: rejects.append({"k": k, "why": why, "content": s["content"][:400]})
        else: samples.append(s)
    base = {"conv_id": seed["conv_id"], "lang": lang, "teacher": teacher["name"], "seed_origin": seed.get("seed_origin"), "u_meta": spec["meta"]}
    if not samples:
        with lock:
            stats["reject_all"] += 1
            for r in rejects: k = "rej_" + r["why"].split(":")[0]; stats[k] = stats.get(k, 0) + 1
            rej_f.write(json.dumps({**base, "rejects": rejects}, ensure_ascii=False) + "\n"); rej_f.flush()
        return
    verdict, jtail = None, ""
    if len(samples) >= 2:
        conv_text = "\n".join(f"{t['role']}: {t['content'][:1500]}" for t in history)
        verdict, jtail = judge_u(judge_t, conv_text, samples[0]["content"], samples[1]["content"])
    best = samples[1] if verdict == "B" else samples[0]; loser = (samples[0] if verdict == "B" else samples[1]) if len(samples) >= 2 else None
    row = {"messages": history + [{"role": "assistant", "content": best["content"], "reasoning_content": best["reasoning"]}], **base, "source": "usab_v1",
           "ko_synthesis": {"pipeline": "usab_v1", "think": True, "n_samples": args.n_samples, "kept": len(samples), "judge": judge_t["name"] if len(samples) >= 2 else None, "verdict": verdict, "judge_tail": jtail,
                            "loser_content": loser["content"][:2000] if loser else None, "rejects": [{"k": r["k"], "why": r["why"]} for r in rejects], "ctoks": best["ctoks"], "validator": "pass"}}
    with lock:
        stats["ok"] += 1; stats["toks"] += sum(s["ctoks"] for s in samples)
        if verdict is None and len(samples) >= 2: stats["judge_fail"] += 1
        for r in rejects: k = "rej_" + r["why"].split(":")[0]; stats[k] = stats.get(k, 0) + 1
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n"); out_f.flush()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=96); ap.add_argument("--n-samples", type=int, default=2); ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--en-ratio", type=float, default=0.8); ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(); rng = random.Random(args.seed); os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try: done.add(json.loads(l)["conv_id"])
            except Exception: pass
    n_en = int(args.n * args.en_ratio); n_ko = args.n - n_en
    # 되묻기 템플릿용 짧은 시드는 별도 풀
    en, ko = load_en(n_en, rng), load_ko(n_ko, rng); en_s, ko_s = load_en(max(n_en // 8, 20), rng, short=True), load_ko(max(n_ko // 8, 10), rng, short=True)
    jobs = []
    for lang, pool, spool in (("en", en, en_s), ("ko", ko, ko_s)):
        for s in pool:
            t = pick_template(rng)
            if t is t_clarify and spool: s = spool[rng.randrange(len(spool))]; s = {**s, "conv_id": s["conv_id"] + ":c"}
            spec = t(s["prompt"], lang, rng); spec["meta"]["lang"] = lang
            jobs.append((s, lang, spec))
    rng.shuffle(jobs); jobs = [j for j in jobs if j[0]["conv_id"] not in done]
    print(f"jobs={len(jobs)} (done={len(done)}) en={sum(1 for j in jobs if j[1]=='en')} ko={sum(1 for j in jobs if j[1]=='ko')} teachers={[t['name'] for t in g.TEACHERS]} thorough={os.environ.get('GEN_THOROUGH','2')}", flush=True)
    stats = {"ok": 0, "reject_all": 0, "judge_fail": 0, "toks": 0}; lock = threading.Lock(); t0 = time.time()
    with open(args.out, "a") as out_f, open(args.out.replace(".jsonl", "") + ".rejects.jsonl", "a") as rej_f, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process, i, s, lang, spec, args, stats, lock, out_f, rej_f) for i, (s, lang, spec) in enumerate(jobs)]
        for n, f in enumerate(as_completed(futs), 1):
            try: f.result()
            except Exception as e:
                with lock: stats["worker_error"] = stats.get("worker_error", 0) + 1
                print("worker_error", repr(e)[:200], flush=True)
            if n % 50 == 0 or n == len(futs):
                el = time.time() - t0
                print(f"[{n}/{len(futs)}] {el:.0f}s ok={stats['ok']} reject_all={stats['reject_all']} judge_fail={stats['judge_fail']} tok/s={stats['toks']/max(el,1):.0f} | " + " ".join(f"{k}={v}" for k, v in stats.items() if k.startswith("rej_")), flush=True)
    print("DONE", json.dumps(stats, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
