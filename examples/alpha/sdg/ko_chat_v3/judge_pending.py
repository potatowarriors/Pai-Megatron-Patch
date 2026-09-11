#!/usr/bin/env python3
"""judge_pending.py — --defer-judge 로 생성된 행(judge_pending=true, pending_sample 보존)을 심판 모델로 판정해 best 를 교체한다.
사용: JUDGE=dsv4-flash python3 judge_pending.py --inp out/p1/gen_A1.jsonl --out out/p1/gen_A1.judged.jsonl --workers 224
판정 B 면 messages[-1] 을 pending_sample 로 교체(원 best 는 loser_content 로), verdict/judge/judge_tail 기록, pending 필드 제거. 재개 가능(out 의 conv_id).
"""
import argparse,json,os,sys,threading,time
from concurrent.futures import ThreadPoolExecutor,as_completed
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import generate_v3 as g
ap=argparse.ArgumentParser(); ap.add_argument("--inp",required=True); ap.add_argument("--out",required=True); ap.add_argument("--workers",type=int,default=224); a=ap.parse_args()
judge_t=g.ALL_TEACHERS[os.environ.get("JUDGE","dsv4-flash")]
done=set()
if os.path.exists(a.out):
    for l in open(a.out):
        try: done.add(json.loads(l)["conv_id"])
        except Exception: pass
rows=[json.loads(l) for l in open(a.inp)]; todo=[r for r in rows if r["conv_id"] not in done]
print(f"rows={len(rows)} done={len(done)} todo={len(todo)} judge={judge_t['name']}",flush=True)
st={"judged":0,"swapped":0,"fail":0,"passthrough":0}; lock=threading.Lock(); t0=time.time()
def work(r):
    ks=r["ko_synthesis"]
    if not ks.get("judge_pending") or not ks.get("pending_sample"):
        return r,"passthrough"
    turns=r["messages"][:-1]; a_=r["messages"][-1]; b_=ks["pending_sample"]
    conv="\n".join(f"{t['role']}: {t['content'][:1500]}" for t in turns)
    v,tail=g.judge(judge_t,conv,a_["content"],b_["content"])
    if v=="B":
        r["messages"][-1]={"role":"assistant","content":b_["content"],"reasoning_content":b_["reasoning"]}
        ks["loser_content"]=a_["content"][:2000]; ks["ctoks"]=b_["ctoks"]; ks["r_hangul"]=round(g.ratios(b_["reasoning"])[0],2)
    else: ks["loser_content"]=b_["content"][:2000]
    ks.update({"judge":judge_t["name"],"verdict":v,"judge_tail":tail,"judge_pending":False}); ks.pop("pending_sample",None)
    return r,("fail" if v is None else ("swapped" if v=="B" else "judged"))
with open(a.out,"a") as o, ThreadPoolExecutor(a.workers) as ex:
    futs=[ex.submit(work,r) for r in todo]
    for n,f in enumerate(as_completed(futs),1):
        r,k=f.result()
        with lock:
            st[k]+=1; 
            if k=="swapped": st["judged"]+=1
            o.write(json.dumps(r,ensure_ascii=False)+"\n")
            if n%500==0 or n==len(futs): print(f"[{n}/{len(futs)}] {time.time()-t0:.0f}s {st}",flush=True)
print("DONE",json.dumps(st),flush=True)
