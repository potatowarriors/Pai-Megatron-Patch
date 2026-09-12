#!/usr/bin/env python3
"""agent_run.py — 트랙 D 에이전트 루프: 정책 교사(DSV4 기본)가 `web-search` 도구를 반복 호출해 답하고, D1 은 정답 일치만 채택.
행 형식 = NVIDIA Agentic-v2 search 와 동일(messages: system/user/assistant(tool_calls, reasoning_content)/tool…, tools 선언) → 변환기 tool 시나리오 분기 그대로.

사용: python3 agent_run.py --questions out/d1/questions.jsonl --out out/d1/traj.jsonl --search http://localhost:8600 --workers 160 [--max-calls 20] [--mode d1|d2]
  d1: question/answer/answer_aliases 필요, 최종 답 == 정답(정규화·별칭·포함) 만 채택, 나머지는 rejects.
  d2: 요청만 있음. 검색 ≥ min_calls 이면 채택(심판은 별도 judge_d2.py).
"""
import argparse, json, os, re, sys, time, threading, random, urllib.request, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ko_chat_v3")); import generate_v3 as g

TOOLS = [{"type": "function", "function": {"name": "web-search", "description": "Search the web for a query.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query."}}, "required": ["query"]}}}]
SYS_D1 = ("당신은 전문 검색 에이전트입니다. 검증 가능한 출처를 근거로 사용자의 질문에 정확하게 답하는 것이 목표입니다.\n\n실행 규칙:\n"
          "1. **검색 결과** — 검색 한 번에 관련 문서 최대 10건이 반환됩니다. 필요한 만큼 검색하되 같은 질의를 반복하지 마세요.\n"
          "2. **계획** — 먼저 질문을 풀기 위한 하위 질문들로 쪼개세요. 여러 단계를 거쳐야 하는 질문이면 한 단계씩 확인하며 진행하세요.\n"
          "3. **검증** — 각 단계의 사실은 검색 결과 본문으로 확인하고, 확인되지 않은 추측으로 다음 단계를 진행하지 마세요. 결과가 없으면 질의를 바꿔 다시 검색하세요.\n"
          "4. **최종 답변** — 모든 단계가 확인되면 도구 호출 없이 최종 답을 한국어로 쓰세요. 마지막 줄에 `최종 답: <개체명>` 형식으로 답만 적으세요.")
SYS_D2 = ("당신은 리서치 에이전트입니다. `web-search` 도구로 조사한 뒤 출처를 인용한 한국어 보고서를 작성합니다.\n\n규칙:\n"
          "1. 조사 계획을 세우고 서로 다른 관점·하위 주제로 최소 5회 이상 검색하세요. 같은 질의 반복 금지.\n"
          "2. 보고서의 모든 사실 주장은 검색 결과에 근거해야 하며, 문장 끝에 [번호] 로 출처를 표시하고 마지막에 `출처` 목록을 붙이세요(번호, 매체 또는 사이트, 제목, 날짜, url 은 있을 때만; 없는 url 을 지어내지 마세요).\n"
          "3. 검색 결과에 없는 내용은 추정임을 명시하거나 쓰지 마세요.\n4. 구조: 요약 → 본문(소제목) → 한계 → 출처.")

def norm(s):
    s = re.sub(r"\(.*?\)", "", s or ""); s = re.sub(r"[\s·\-_.,'\"「」『』《》〈〉()\[\]]", "", s.lower())
    return s

def match_answer(final, answer, aliases):
    m = re.search(r"최종\s*답\s*[:：]\s*(.+)", final)
    cand = m.group(1).strip() if m else final.strip()[-200:]
    c = norm(cand); targets = [norm(answer)] + [norm(x) for x in aliases if x]
    targets = [t for t in targets if len(t) >= 2]
    ok = any(t == c or t in c or (len(t) >= 4 and c in t) for t in targets)
    return ok, cand[:120]

def search(ep, query, k=10):
    """검색 서버 호출 — 일시 장애(서버 재기동·연결 오류)에 5회 백오프 재시도. 2026-09-12 실측: 재기동 1분 동안 진행 중 트라젝토리 238건이 통째로 폐기됐다."""
    req = urllib.request.Request(ep.rstrip("/") + "/search", data=json.dumps({"query": query, "max_results": k}).encode(), headers={"Content-Type": "application/json"})
    for i in range(6):
        try: return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception as e:
            err = e; time.sleep(min(60, 5 * (i + 1)))
    raise err

def llm_post(teacher, body):
    for i in range(4):
        try: return g.post(teacher["endpoint"], body, timeout=int(os.environ.get("GEN_TIMEOUT", "1800")))
        except Exception as e:
            err = e; time.sleep(min(60, 10 * (i + 1)))
    raise err

def run_one(teacher, ep, question, args):
    sys_ = SYS_D1 if args.mode == "d1" else SYS_D2
    msgs = [{"role": "system", "content": sys_}, {"role": "user", "content": question}]
    n_calls, turns_tok, finish = 0, [], []
    for step in range(args.max_calls + 1):
        body = {"model": teacher["model"], "messages": msgs, "tools": TOOLS, "tool_choice": "auto", "max_tokens": args.max_tokens, **teacher["sampling"]}
        d = llm_post(teacher, body); ch = d["choices"][0]; m = ch["message"]
        rc = m.get("reasoning_content") or m.get("reasoning") or ""; content = m.get("content") or ""; tcs = m.get("tool_calls") or []
        finish.append(ch["finish_reason"]); turns_tok.append(d.get("usage", {}).get("completion_tokens", 0))
        am = {"role": "assistant", "content": content, "reasoning_content": rc}
        if tcs:
            am["tool_calls"] = [{"id": tc.get("id") or f"call_{step}", "type": "function", "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}} for tc in tcs]
            msgs.append(am)
            for tc in am["tool_calls"]:
                try: q = json.loads(tc["function"]["arguments"]).get("query", "")
                except Exception: q = ""
                n_calls += 1
                res = search(ep, q) if q else {"query": q, "results": [], "answer": None, "follow_up_questions": None, "images": [], "response_time": 0}
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "name": "web-search", "content": json.dumps(res, ensure_ascii=False)})
            if n_calls >= args.max_calls:
                msgs.append({"role": "user", "content": "검색 한도에 도달했습니다. 지금까지 확인한 내용으로 최종 답을 쓰세요."})
            continue
        msgs.append(am)
        return msgs, n_calls, finish, turns_tok, content
    return msgs, n_calls, finish, turns_tok, None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--questions", required=True); ap.add_argument("--out", required=True); ap.add_argument("--search", default="http://127.0.0.1:8600")   # localhost 는 IPv6 ::1 폴백으로 오류 99 를 낸다(컨테이너에 IPv6 없음)
    ap.add_argument("--workers", type=int, default=160); ap.add_argument("--max-calls", type=int, default=20); ap.add_argument("--max-tokens", type=int, default=6144); ap.add_argument("--mode", choices=["d1", "d2"], default="d1")
    ap.add_argument("--min-calls", type=int, default=5); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--teacher", default=os.environ.get("AGENT_TEACHER", "dsv4-flash"))
    args = ap.parse_args(); teacher = g.ALL_TEACHERS[args.teacher]
    qs = [json.loads(l) for l in open(args.questions)]
    if args.limit: qs = qs[:args.limit]
    done = set()
    for p in (args.out, args.out.replace(".jsonl", "") + ".rejects.jsonl"):
        if os.path.exists(p):
            for l in open(p):
                try:
                    r = json.loads(l)
                    if str(r.get("why", "")).startswith("error:"): continue      # 일시 오류(서버 교체 등)는 재시도 대상
                    done.add(r["conv_id"])
                except Exception: pass
    todo = [q for q in qs if q["conv_id"] not in done]
    print(f"questions={len(qs)} done={len(done)} todo={len(todo)} teacher={teacher['name']} mode={args.mode} search={args.search}", flush=True)
    st = collections.Counter(); lock = threading.Lock(); t0 = time.time(); calls_hist = []
    def work(q):
        try: msgs, n_calls, finish, toks, final = run_one(teacher, args.search, q["question"], args)
        except Exception as e: return q, None, f"error:{e!r}"[:160], 0, None
        if final is None: return q, msgs, "no_final", n_calls, None
        if g.SPECIAL.search(final) or g.SELF_ATTR.search(final): return q, msgs, "gate", n_calls, None
        if args.mode == "d1":
            ok, cand = match_answer(final, q["answer"], q.get("answer_aliases", []))
            return q, msgs, (None if ok else "wrong_answer"), n_calls, cand
        if n_calls < args.min_calls: return q, msgs, "too_few_searches", n_calls, None
        return q, msgs, None, n_calls, None
    with open(args.out, "a") as out, open(args.out.replace(".jsonl", "") + ".rejects.jsonl", "a") as rej, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, q) for q in todo]
        for n, f in enumerate(as_completed(futs), 1):
            q, msgs, why, n_calls, cand = f.result()
            with lock:
                if why:
                    st["rej_" + why.split(":")[0]] += 1
                    rej.write(json.dumps({"conv_id": q["conv_id"], "why": why, "n_calls": n_calls, "final_cand": cand, "answer": q.get("answer"), "messages": msgs}, ensure_ascii=False) + "\n"); rej.flush()
                else:
                    st["ok"] += 1; calls_hist.append(n_calls)
                    row = {"messages": msgs, "tools": TOOLS, "conv_id": q["conv_id"], "source": "search_ko_v1" if args.mode == "d1" else "research_ko_v1", "teacher": teacher["name"],
                           "metadata": {"num_tool_calls": n_calls, "ground_truth": q.get("answer"), "final_answer_cand": cand, "n_hops": q.get("n_hops"), "chain_id": q.get("chain_id"), "seed_origin": q.get("seed_origin"), "mode": args.mode}}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                if n % 25 == 0 or n == len(futs):
                    el = time.time() - t0; ch = sorted(calls_hist)
                    print(f"[{n}/{len(futs)}] {el:.0f}s ok={st['ok']} calls_median={ch[len(ch)//2] if ch else 0} | " + " ".join(f"{k}={v}" for k, v in st.items() if k.startswith("rej_")), flush=True)
    print("DONE", json.dumps(st, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
