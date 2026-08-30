"""LogicKor (KoChat, T3) — 한국어 멀티턴 chat 품질, gemini judge 1~10점.

42문항 × 2턴(싱글턴 + 후속턴). ko_chat SFT 데이터가 가르치는 한국어 능력 측정.
지표: single_turn(1턴 평균), multi_turn(2턴 평균), overall(전체 평균) — 각 /10.
사용: python3 eval_sft/runners/run_logickor.py --base-url http://localhost:8100/v1 --run-name baseline_lcb_iter320
"""
from __future__ import annotations
import argparse, ast, json, re, sys, time, urllib.request, concurrent.futures as cf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemini_judge import load_key, judge_batch  # noqa: E402

JUDGE_TMPL = """당신은 한국어 언어모델 답변을 평가하는 채점자입니다. 아래 질문에 대한 모델 답변의 품질을
1점(매우 나쁨)에서 10점(매우 우수)으로 평가하세요. 정확성·논리성·한국어 유창성·지시 이행을 종합합니다.
{ref_block}
[질문]
{question}

[모델 답변]
{answer}

먼저 한 문장으로 평가 근거를 쓰고, 마지막 줄에 반드시 `[[점수]]` 형식으로 정수 점수만 쓰세요. 예: [[7]]"""

def chat(base_url, messages, max_tokens, timeout):
    body=json.dumps({"model":"alpha","messages":messages,"temperature":0.6,"top_p":0.95,"max_tokens":max_tokens}).encode()
    req=urllib.request.Request(base_url+"/chat/completions", data=body, headers={"Content-Type":"application/json"})
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["choices"][0]["message"].get("content") or ""
        except Exception: time.sleep(2**a)
    return ""

def strip_think(t):
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()

def run_item(base_url, qs, max_tokens, timeout):
    a1_raw=chat(base_url, [{"role":"user","content":qs[0]}], max_tokens, timeout)
    a1=strip_think(a1_raw)
    a2=""
    if len(qs)>1 and qs[1]:
        a2_raw=chat(base_url, [{"role":"user","content":qs[0]},{"role":"assistant","content":a1},
                               {"role":"user","content":qs[1]}], max_tokens, timeout)
        a2=strip_think(a2_raw)
    return a1, a2

def parse_score(t):
    m=re.findall(r"\[\[\s*(\d+(?:\.\d+)?)\s*\]\]", t or "")
    if m:
        try: return max(0.0, min(10.0, float(m[-1])))
        except: pass
    m2=re.findall(r"(\d+(?:\.\d+)?)\s*/\s*10", t or "")
    return max(0.0, min(10.0, float(m2[-1]))) if m2 else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True); ap.add_argument("--run-name", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=32768); ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1]/"results"))
    a=ap.parse_args()
    import os; os.environ.setdefault("HF_HOME","/home/work/Datasets/benchmarks")
    from datasets import load_dataset
    rows=list(load_dataset("maywell/LogicKor")["train"])
    if a.limit: rows=rows[:a.limit]
    def parse_list(x): return x if isinstance(x, list) else ast.literal_eval(x)
    Q=[parse_list(r["questions"]) for r in rows]
    R=[parse_list(r["references"]) for r in rows]
    print(f"[logickor] {len(rows)} items × 2턴, 생성…", flush=True)
    ans=[None]*len(rows)
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs={ex.submit(run_item, a.base_url, q, a.max_tokens, a.timeout):i for i,q in enumerate(Q)}
        for f in cf.as_completed(futs): ans[futs[f]]=f.result()
    # judge: 턴별 채점
    print("[logickor] 채점…", flush=True)
    jp=[]; meta=[]
    for i,(q,r,(a1,a2)) in enumerate(zip(Q,R,ans)):
        rb=lambda ref: f"[참고 답안]\n{ref}\n" if ref else ""
        jp.append(JUDGE_TMPL.format(ref_block=rb(r[0] if r else None), question=q[0], answer=a1)); meta.append((i,1))
        if len(q)>1 and q[1]:
            jp.append(JUDGE_TMPL.format(ref_block=rb(r[1] if len(r)>1 else None), question=q[1], answer=a2)); meta.append((i,2))
    grades=judge_batch(jp, load_key(), workers=12, max_tokens=512)
    s1=[]; s2=[]; cat_scores={}
    for (i,turn),g in zip(meta,grades):
        sc=parse_score(g)
        if sc is None: continue
        (s1 if turn==1 else s2).append(sc)
        cat_scores.setdefault(rows[i]["category"], []).append(sc)
    mean=lambda xs: sum(xs)/len(xs) if xs else 0.0
    metrics={"single_turn":mean(s1),"multi_turn":mean(s2),"overall":mean(s1+s2),
             "by_category":{k:round(mean(v),2) for k,v in cat_scores.items()},"n":len(rows)}
    outd=Path(a.out_dir)/a.run_name; outd.mkdir(parents=True, exist_ok=True)
    # 집계기 호환: 10점 만점을 100분율로 환산해 저장
    (outd/"results_logickor.json").write_text(json.dumps(
        {"results":{"logickor":{"score,none":metrics["overall"]/10.0,
                    "single_turn,none":metrics["single_turn"]/10.0,
                    "multi_turn,none":metrics["multi_turn"]/10.0}}, "logickor_detail":metrics}, indent=2, ensure_ascii=False))
    (outd/"logickor_samples.jsonl").write_text("\n".join(
        json.dumps({"cat":rows[i]["category"],"q1":Q[i][0][:200],"a1":ans[i][0][:400],
                    "a2":ans[i][1][:400]}, ensure_ascii=False) for i in range(len(rows))))
    print(f"[logickor] overall={metrics['overall']:.2f}/10  single={metrics['single_turn']:.2f}  multi={metrics['multi_turn']:.2f}", flush=True)
    print(f"[logickor] by_cat={metrics['by_category']}", flush=True)
    print(f"[logickor] → {outd}", flush=True)

if __name__=="__main__": main()
