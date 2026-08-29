"""SimpleQA-Verified (T3) — 사실성/환각 벤치, gemini judge 채점.

모델 답(chat/thinking)을 프록시로 생성 → google 공식 SimpleQA 채점 프롬프트로
judge(gemini-3.7-flash)가 CORRECT/INCORRECT/NOT_ATTEMPTED 분류.
지표: accuracy = correct/total, attempted, correct_given_attempted (= DSV4 Pass@1 근사).

사용: python3 eval_sft/runners/run_simpleqa.py --base-url http://localhost:8100/v1 \
        --run-name baseline_lcb_iter320 [--limit N]
"""
from __future__ import annotations
import argparse, json, re, sys, time, urllib.request, concurrent.futures as cf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemini_judge import load_key, judge_batch  # noqa: E402

GRADER_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].

Question: {question}
Gold target: {target}
Predicted answer: {predicted}

Grade the predicted answer as one of:
A: CORRECT — the predicted answer fully contains the gold target without contradicting it (semantic match; ignore minor formatting, hedging that doesn't contradict).
B: INCORRECT — the predicted answer contradicts the gold target.
C: NOT_ATTEMPTED — the answer neither confirms nor contradicts the gold target (e.g. "I don't know", refusal, or no clear answer).

Reply with a SINGLE letter: A, B, or C. Nothing else."""

def gen_one(base_url, question, max_tokens, timeout):
    body=json.dumps({"model":"alpha","messages":[{"role":"user","content":question}],
                     "temperature":0.0,"max_tokens":max_tokens}).encode()
    req=urllib.request.Request(base_url+"/chat/completions", data=body, headers={"Content-Type":"application/json"})
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d=json.loads(r.read())
            return d["choices"][0]["message"].get("content") or ""
        except Exception: time.sleep(2**a)
    return ""

def strip_think(t):  # thinking 태그 제거 → 최종 답만 judge 에 전달
    if "</think>" in t: t=t.split("</think>")[-1]
    return t.strip()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True); ap.add_argument("--run-name", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=8192); ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1]/"results"))
    a=ap.parse_args()
    import os; os.environ.setdefault("HF_HOME","/home/work/Datasets/benchmarks")
    from datasets import load_dataset
    ds=load_dataset("google/simpleqa-verified")["eval"]
    rows=list(ds)[: a.limit] if a.limit else list(ds)
    print(f"[simpleqa] {len(rows)} questions, generating…", flush=True)

    # 1) 모델 답 생성 (병렬)
    preds=[""]*len(rows)
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs={ex.submit(gen_one, a.base_url, r["problem"], a.max_tokens, a.timeout):i for i,r in enumerate(rows)}
        done=0
        for f in cf.as_completed(futs):
            preds[futs[f]]=f.result(); done+=1
            if done%100==0: print(f"  gen {done}/{len(rows)}", flush=True)

    # 2) judge 채점 (병렬)
    print("[simpleqa] judging…", flush=True)
    prompts=[GRADER_TEMPLATE.format(question=r["problem"], target=r["answer"], predicted=strip_think(p))
             for r,p in zip(rows,preds)]
    grades=judge_batch(prompts, load_key(), workers=16, max_tokens=8)
    def cls(g):
        g=(g or "").strip().upper()
        m=re.search(r"[ABC]", g)
        return {"A":"CORRECT","B":"INCORRECT","C":"NOT_ATTEMPTED"}.get(m.group(0) if m else "", "NOT_ATTEMPTED")
    labels=[cls(g) for g in grades]
    n=len(labels); nc=labels.count("CORRECT"); ni=labels.count("INCORRECT"); na=labels.count("NOT_ATTEMPTED")
    attempted=nc+ni
    metrics={"n":n,"correct":nc,"incorrect":ni,"not_attempted":na,
             "accuracy":nc/n if n else 0.0,
             "attempted_rate":attempted/n if n else 0.0,
             "correct_given_attempted":nc/attempted if attempted else 0.0,
             "f1": (2*(nc/n)*(nc/attempted)/((nc/n)+(nc/attempted))) if (n and attempted and nc) else 0.0}

    outd=Path(a.out_dir)/a.run_name; outd.mkdir(parents=True, exist_ok=True)
    # lm_eval 호환 형태로도 저장 (집계기가 읽도록 results.json 안에 results.simpleqa_verified)
    (outd/"results_simpleqa.json").write_text(json.dumps(
        {"results":{"simpleqa_verified":{"accuracy,none":metrics["accuracy"],
                     "correct_given_attempted,none":metrics["correct_given_attempted"],
                     "f1,none":metrics["f1"]}}, "simpleqa_detail":metrics}, indent=2))
    (outd/"simpleqa_samples.jsonl").write_text("\n".join(
        json.dumps({"q":r["problem"],"target":r["answer"],"pred":strip_think(p)[:500],"grade":l}, ensure_ascii=False)
        for r,p,l in zip(rows,preds,labels)))
    print(f"[simpleqa] accuracy={metrics['accuracy']*100:.1f}  attempted={metrics['attempted_rate']*100:.1f}  "
          f"correct|attempted={metrics['correct_given_attempted']*100:.1f}  F1={metrics['f1']*100:.1f}", flush=True)
    print(f"[simpleqa] → {outd}", flush=True)

if __name__=="__main__": main()
