#!/usr/bin/env python3
"""t_generate.py — 트랙 T: 한국어 도구 호출/미호출 (When2Call 형, 호출:미호출 = 1:2). phase-2 회귀(도구 선언 시 유령 호출) 교정용 (2026-09-12).

1) 도구 세트: Nemotron-SFT-Agentic-v2 tool_calling 행의 `tools`(실제 API 스키마, 1~8개) 를 그대로 선언.
2) 사례: call(1/3) · direct(1/3: 도구와 무관·도구 없이 답 가능) · clarify(1/6: 도구 대상이나 필수 인자 누락) · infeasible(1/6: 선언 도구로 불가능한 작업).
   요청문은 GLM(저효율 사고) 이 사례 조건에 맞춰 한국어로 작성.
3) 정책 응답: 교사 라운드로빈(GLM v2 지시문·DSV4), tools 선언 + tool_choice auto, n=2 → **검증기**(call: tool_calls 있음·선언된 이름·필수 인자 충족 / 미호출: tool_calls 없음, clarify: 질문형, infeasible: 한계 인정) 통과분 → 상대 교사 심판.
   call 사례는 도구 결과를 DSV4 무사고로 모의 생성 → 최종 답변 턴까지 붙인다(Agentic-v2 형).
출력 행: {messages(system 없음), tools, conv_id, source: kotool_v1, teacher, metadata{case,...}}. 변환기는 tools 선언 → tool 시나리오 분기.
사용: GEN_THOROUGH=2 python3 t_generate.py --n 60 --out out/p0/t_p0.jsonl --workers 96
"""
import argparse, json, os, sys, re, random, threading, time, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "ko_chat_v3")); import generate_v3 as g

SRC = "/home/work/Datasets/LL_datasets/posttraining/SFT/Nemotron-SFT-Agentic-v2/data/tool_calling.jsonl"
H = re.compile(r"[가-힣]"); L = re.compile(r"[A-Za-z]")
CASES = [("call", 2), ("direct", 2), ("clarify", 2), ("infeasible", 1)]   # P0: clarify 수율 0/7 → 규칙 강화 + 비중 상향
WRITER = {
 "call": "선언된 도구 중 「{target}」 를 실제로 호출해야만 처리할 수 있는 요청을 쓰세요. 그 도구의 필수 인자를 채울 수 있도록 구체적 정보(이름·날짜·수량 등)를 요청문에 자연스럽게 포함하세요. 도구 이름을 직접 언급하지는 마세요.",
 "direct": "선언된 도구들과 전혀 무관하거나 도구 없이도 충분히 답할 수 있는 일반 요청(지식 질문·설명·글쓰기·조언 등)을 쓰세요. 도구를 써야 하는 것처럼 보이지 않게 하세요.",
 "clarify": "선언된 도구 「{target}」 가 필요한 요청이지만, 되묻지 않고는 호출이 **불가능**하도록 핵심 정보를 빠뜨리세요: 필수 인자 중 기본값으로 대신할 수 없는 것(대상 식별자·이름·도시·날짜·계좌·상품 등)을 하나 이상 비우고, 요청 자체는 짧고 막연하게(예: '예약 좀 해줘', '그 사람 정보 찾아줘'). 도구 없이 답할 수 있는 질문이면 안 됩니다. 도구 이름은 언급하지 마세요.",
 "infeasible": "선언된 도구들이 제공하지 않는 능력(예: 결제·전화·물리적 행동·다른 서비스 조작)이 필요한 요청을 쓰세요. 선언된 도구로는 해결할 수 없어야 합니다.",
}
WRITE_PROMPT = """당신은 AI 어시스턴트 학습 데이터를 만드는 작가입니다. 아래 도구들이 어시스턴트에게 선언되어 있습니다.

[선언된 도구]
{tools}

과제: {case_rule}

형식: 실제 한국인 사용자가 채팅창에 쓸 법한 자연스러운 한국어 요청 1개(1~3문장). 메타 설명 없이 JSON 하나만 출력: {{"request": "..."}}"""
SIM_PROMPT = """다음 도구 호출에 대한 그럴듯한 실행 결과를 JSON 하나로만 출력하세요(설명 금지). 스키마에 맞는 현실적인 값(한국 맥락)으로 채우고 에러가 아닌 정상 결과로 만드세요.
도구: {name}
설명: {desc}
호출 인자: {args}"""

def lang_ok(s): h, l = len(H.findall(s)), len(L.findall(s)); return h >= 10 and h / max(h + l, 1) >= 0.6

def load_toolsets(n, rng):
    sets = []
    with open(SRC) as f:
        for i, l in enumerate(f):
            if i > 120000: break
            r = json.loads(l); ts = r.get("tools") or []
            if not ts or len(ts) > 8: continue
            if any(not isinstance(t, dict) or "function" not in t or not t["function"].get("name") for t in ts): continue
            sets.append(ts)
    rng.shuffle(sets); return sets[:n]

def tools_brief(ts):
    out = []
    for t in ts:
        fn = t["function"]; req = (fn.get("parameters") or {}).get("required") or []; props = list(((fn.get("parameters") or {}).get("properties") or {}).keys())
        out.append(f"- {fn['name']}: {(fn.get('description') or '')[:200]} | 인자 {props[:8]} | 필수 {req}")
    return "\n".join(out)

def write_request(case, ts, rng):
    target = rng.choice(ts)["function"]["name"] if case in ("call", "clarify") else None
    body = {"model": "glm53-flash", "messages": [{"role": "user", "content": WRITE_PROMPT.format(tools=tools_brief(ts), case_rule=WRITER[case].format(target=target))}], "max_tokens": 3000, "temperature": 0.9, "top_p": 0.95, "chat_template_kwargs": {"reasoning_effort": "low"}}
    d = g.post(g.ALL_TEACHERS["glm53-flash"]["endpoint"], body, timeout=600); c = d["choices"][0]["message"].get("content") or ""
    m = re.search(r"\{.*\}", c, flags=re.S)
    if not m: return None, None, "no_json"
    try: q = (json.loads(m.group(0)).get("request") or "").strip()
    except Exception: return None, None, "bad_json"
    if not lang_ok(q) or len(q) < 12: return None, None, "bad_request"
    return q, target, None

def validate(case, m, ts, target):
    tcs = m.get("tool_calls") or []; content = m.get("content") or ""
    names = {t["function"]["name"] for t in ts}
    if case == "call":
        if not tcs: return "no_call"
        tc = tcs[0]; fn = tc["function"]["name"]
        if fn not in names: return "unknown_tool"
        try: args = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
        except Exception: return "bad_args"
        spec = next(t for t in ts if t["function"]["name"] == fn); req = (spec["function"].get("parameters") or {}).get("required") or []
        if any(k not in args or args[k] in ("", None) for k in req): return "missing_required"
        return None
    if tcs: return "ghost_call"
    if not content.strip(): return "empty"
    if case == "clarify":
        last = [l for l in content.strip().splitlines() if l.strip()][-1]
        if "?" not in last and "？" not in last: return "no_question"
        if len(content) > 300: return "too_long"
    if case == "infeasible" and not re.search(r"(없습니다|없어요|할 수 없|불가능|지원하지 않|권한이 없|제공되지 않|도구가 없|기능이 없)", content): return "no_limitation_ack"
    return None

def simulate_tool(tc, ts):
    fn = tc["function"]["name"]; spec = next(t for t in ts if t["function"]["name"] == fn)
    body = {"model": "dsv4-flash", "messages": [{"role": "user", "content": SIM_PROMPT.format(name=fn, desc=(spec["function"].get("description") or "")[:300], args=tc["function"]["arguments"])}], "max_tokens": 800, "temperature": 0.7, "chat_template_kwargs": {"thinking": False}}
    d = g.post(g.ALL_TEACHERS["dsv4-flash"]["endpoint"], body, timeout=600); c = d["choices"][0]["message"].get("content") or ""
    m = re.search(r"\{.*\}", c, flags=re.S); return m.group(0) if m else json.dumps({"status": "ok", "result": c[:300]}, ensure_ascii=False)

def policy(teacher, msgs, ts, max_tokens):
    body = {"model": teacher["model"], "messages": msgs, "tools": ts, "tool_choice": "auto", "max_tokens": max_tokens, **teacher["sampling"]}
    d = g.post(teacher["endpoint"], body, timeout=int(os.environ.get("GEN_TIMEOUT", "1800"))); ch = d["choices"][0]; m = ch["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or m.get("reasoning") or "", "tool_calls": m.get("tool_calls") or [], "finish": ch["finish_reason"], "ctoks": d.get("usage", {}).get("completion_tokens", 0)}

def process(idx, ts, case, args, stats, lock, out_f, rej_f, rng):
    teacher, judge_t = g.TEACHERS[idx % len(g.TEACHERS)], g.pick_judge(idx)
    try: q, target, why = write_request(case, ts, rng)
    except Exception as e: q, target, why = None, None, f"write_error:{e!r}"[:100]
    cid = f"kotool:{idx + args.id_offset:06d}"
    if why:
        with lock: stats["rej_" + why] = stats.get("rej_" + why, 0) + 1; rej_f.write(json.dumps({"conv_id": cid, "case": case, "why": why}, ensure_ascii=False) + "\n")
        return
    # 생성 시점 시스템 규칙(학습 행에는 미포함): 유령 호출 억제·되묻기 유도. P1 실측: 규칙 없이는 clarify 수율 0, ghost_call 리젝 30%
    msgs = [{"role": "system", "content": g.SYS_GEN + " 도구는 요청 처리에 꼭 필요할 때만 호출하세요. 도구 호출에 필요한 핵심 정보가 요청에 없으면 추측해서 호출하지 말고 가장 중요한 것 하나만 짧게 되물으세요. 선언된 도구로 할 수 없는 일이면 그 한계를 솔직히 밝히고 대안을 제시하세요."}, {"role": "user", "content": q}]
    samples, rejects = [], []
    for k in range(args.n_samples):
        try: s = policy(teacher, msgs, ts, args.max_tokens)
        except Exception as e: rejects.append({"k": k, "why": f"gen_error:{e!r}"[:100]}); continue
        v = validate(case, s, ts, target)
        if v is None and not s["tool_calls"]:
            gw = g.gate({"reasoning": s["reasoning"], "content": s["content"], "finish": s["finish"], "ctoks": s["ctoks"]}); v = gw
        if v is None and s["tool_calls"] and (g.HANJA.search(s["reasoning"]) and g.ratios(s["reasoning"])[1] > 0.02): v = "reasoning_chinese"
        if v: rejects.append({"k": k, "why": v})
        else: samples.append(s)
    base = {"conv_id": cid, "case": case, "teacher": teacher["name"], "target": target}
    if not samples:
        with lock:
            stats["reject_all"] += 1
            for r in rejects: stats["rej_" + r["why"].split(":")[0]] = stats.get("rej_" + r["why"].split(":")[0], 0) + 1
            rej_f.write(json.dumps({**base, "request": q, "rejects": rejects}, ensure_ascii=False) + "\n"); rej_f.flush()
        return
    verdict, jtail = None, ""
    if len(samples) >= 2:
        A = samples[0]["content"] or json.dumps(samples[0]["tool_calls"], ensure_ascii=False); B = samples[1]["content"] or json.dumps(samples[1]["tool_calls"], ensure_ascii=False)
        verdict, jtail = g.judge(judge_t, f"[선언된 도구]\n{tools_brief(ts)}\n\nuser: {q}", A, B)
    best = samples[1] if verdict == "B" else samples[0]
    am = {"role": "assistant", "content": best["content"], "reasoning_content": best["reasoning"]}
    convo = [{"role": "user", "content": q}]
    if best["tool_calls"]:
        tcs = [{"id": tc.get("id") or f"call_{idx}", "type": "function", "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}} for tc in best["tool_calls"]]
        am["tool_calls"] = tcs; convo.append(am)
        try:
            for tc in tcs: convo.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["function"]["name"], "content": simulate_tool(tc, ts)})
            fin = policy(teacher, [msgs[0]] + convo, ts, args.max_tokens)
            if fin["tool_calls"] or not fin["content"].strip(): raise RuntimeError("final_turn_invalid")
            convo.append({"role": "assistant", "content": fin["content"], "reasoning_content": fin["reasoning"]})
        except Exception as e:
            with lock: stats["reject_all"] += 1; stats["rej_final_turn"] = stats.get("rej_final_turn", 0) + 1; rej_f.write(json.dumps({**base, "request": q, "why": f"final:{e!r}"[:100]}, ensure_ascii=False) + "\n")
            return
    else: convo.append(am)
    row = {"messages": convo, "tools": ts, **base, "source": "kotool_v1",
           "metadata": {"case": case, "n_tools": len(ts), "judge": judge_t["name"] if len(samples) >= 2 else None, "verdict": verdict, "judge_tail": jtail, "rejects": [{"k": r["k"], "why": r["why"]} for r in rejects], "ctoks": best["ctoks"]}}
    with lock:
        stats["ok"] += 1; stats["ok_" + case] = stats.get("ok_" + case, 0) + 1
        for r in rejects: stats["rej_" + r["why"].split(":")[0]] = stats.get("rej_" + r["why"].split(":")[0], 0) + 1
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n"); out_f.flush()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--n-samples", type=int, default=2); ap.add_argument("--max-tokens", type=int, default=6144); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--id-offset", type=int, default=0)
    args = ap.parse_args(); rng = random.Random(args.seed); os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sets = load_toolsets(args.n, rng); print(f"toolsets={len(sets)} teachers={[t['name'] for t in g.TEACHERS]}", flush=True)
    cases = [c for c, w in CASES for _ in range(w)]
    stats = {"ok": 0, "reject_all": 0}; lock = threading.Lock(); t0 = time.time()
    with open(args.out, "a") as out_f, open(args.out.replace(".jsonl", "") + ".rejects.jsonl", "a") as rej_f, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process, i, ts, rng.choice(cases), args, stats, lock, out_f, rej_f, random.Random(i)) for i, ts in enumerate(sets)]
        for n, f in enumerate(as_completed(futs), 1):
            try: f.result()
            except Exception as e:
                with lock: stats["worker_error"] = stats.get("worker_error", 0) + 1
                print("worker_error", repr(e)[:200], flush=True)
            if n % 50 == 0 or n == len(futs): print(f"[{n}/{len(futs)}] {time.time()-t0:.0f}s " + json.dumps(stats, ensure_ascii=False), flush=True)
    print("DONE", json.dumps(stats, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
