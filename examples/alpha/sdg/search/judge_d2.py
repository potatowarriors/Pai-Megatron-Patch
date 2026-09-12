#!/usr/bin/env python3
"""judge_d2.py — 트랙 D2 보고서 심판: 최종 보고서의 인용 주장이 수집된 검색 결과로 뒷받침되는지(인용 충실성) + 구조·검색 횟수.
심판 = GLM(저효율) 기본. 채택 기준(P0 재보정 2026-09-12): supported ≥ 0.7, 허위 출처 ≤ 1, 구조 충족, 검색 ≥ min_calls. 심판에는 수집 passage 전문(700자)·최대 120k자를 넘긴다(발췌 300자로는 지지 판정이 과소).
사용: python3 judge_d2.py --inp out/d2/traj.jsonl --out out/d2/traj.judged.jsonl --workers 96
"""
import argparse, json, os, sys, re, threading, time, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ko_chat_v3")); import generate_v3 as g

PROMPT = """아래는 리서치 에이전트가 검색으로 수집한 출처들과, 그 출처만 근거로 써야 하는 한국어 보고서입니다. 보고서를 평가하세요.

[수집된 출처] (번호는 보고서의 인용 번호와 무관할 수 있음; 제목·URL·본문 발췌)
{sources}

[보고서]
{report}

평가 항목:
1. 보고서에서 사실 주장 문장을 최대 10개 고르고, 각 주장이 수집된 출처 어딘가에서 뒷받침되는지 판정(supported / unsupported).
2. 구조: 요약·본문(소제목)·한계·출처 목록이 모두 있는가.
3. 출처 목록의 항목이 실제 수집된 출처와 대응하는가(지어낸 출처 여부).

JSON 하나만 출력:
{{"claims": [{{"claim": "...", "supported": true}}, ...], "structure_ok": true, "fabricated_sources": 0, "notes": "..."}}"""

def collect_sources(msgs):
    seen, out = set(), []
    for m in msgs:
        if m["role"] != "tool": continue
        try: res = json.loads(m["content"]).get("results") or []
        except Exception: continue
        for r in res:
            key = r.get("url") or (str(r.get("title")) + str(r.get("published_date")))
            if key in seen: continue
            seen.add(key); out.append(f"- {r.get('title')} | {r.get('url') or ((r.get('publisher') or '') + ' ' + (r.get('published_date') or ''))}\n  {(r.get('content') or '')[:700]}")
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--judge", default="glm53-flash"); ap.add_argument("--min-supported", type=float, default=0.7); ap.add_argument("--max-fabricated", type=int, default=1); ap.add_argument("--min-calls", type=int, default=5); a = ap.parse_args()
    jt = g.ALL_TEACHERS[a.judge]; kw, budget = g.JUDGE_KW.get(jt["name"], ({}, 4096))
    rows = [json.loads(l) for l in open(a.inp)]; done = set()
    if os.path.exists(a.out):
        for l in open(a.out): done.add(json.loads(l)["conv_id"])
    todo = [r for r in rows if r["conv_id"] not in done]; print(f"rows={len(rows)} done={len(done)} todo={len(todo)} judge={jt['name']}", flush=True)
    st = collections.Counter(); lock = threading.Lock(); t0 = time.time()
    def work(r):
        report = r["messages"][-1]["content"]; srcs = collect_sources(r["messages"])
        body = {"model": jt["model"], "max_tokens": max(budget, 6000), "temperature": 0.0, **kw, "messages": [{"role": "user", "content": PROMPT.format(sources="\n".join(srcs)[:120000], report=report[:12000])}]}
        try: d = g.post(jt["endpoint"], body, timeout=900); c = d["choices"][0]["message"].get("content") or ""
        except Exception as e: return r, None, f"judge_error:{e!r}"[:100]
        m = re.search(r"\{.*\}", c, flags=re.S)
        if not m: return r, None, "judge_no_json"
        try: j = json.loads(m.group(0))
        except Exception: return r, None, "judge_bad_json"
        claims = j.get("claims") or []; sup = sum(1 for x in claims if x.get("supported")) / max(len(claims), 1)
        verdict = {"supported_ratio": round(sup, 2), "n_claims": len(claims), "structure_ok": bool(j.get("structure_ok")), "fabricated_sources": int(j.get("fabricated_sources") or 0), "notes": (j.get("notes") or "")[:300]}
        n_calls = r["metadata"].get("num_tool_calls", 0)
        why = None
        if n_calls < a.min_calls: why = "too_few_searches"
        elif len(claims) < 3: why = "too_few_claims"
        elif sup < a.min_supported: why = "unsupported"
        elif not verdict["structure_ok"]: why = "structure"
        elif verdict["fabricated_sources"] > a.max_fabricated: why = "fabricated_sources"
        return r, verdict, why
    with open(a.out, "a") as out, open(a.out.replace(".jsonl", "") + ".rejects.jsonl", "a") as rej, ThreadPoolExecutor(a.workers) as ex:
        for n, f in enumerate(as_completed([ex.submit(work, r) for r in todo]), 1):
            r, v, why = f.result()
            with lock:
                if v: r["metadata"]["d2_judge"] = v
                if why: st["rej_" + why.split(":")[0]] += 1; rej.write(json.dumps({**r, "why": why}, ensure_ascii=False) + "\n")
                else: st["ok"] += 1; out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush()
                if n % 50 == 0 or n == len(todo): print(f"[{n}/{len(todo)}] {time.time()-t0:.0f}s {dict(st)}", flush=True)
    print("DONE", dict(st), flush=True)

if __name__ == "__main__":
    main()
