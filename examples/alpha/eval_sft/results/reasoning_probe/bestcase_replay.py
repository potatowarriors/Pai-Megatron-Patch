#!/usr/bin/env python3
"""OpenWebUI 베스트 케이스(09-07) 대화를 턴 단위로 재생. 히스토리는 그때 저장된 assistant 답변을 그대로 쓴다(모델이 그때 본 문맥과 동일).
조건: tools25(OpenWebUI 주입 재현) / none(LibreChat 경로). 샘플링: greedy 1 + 기본(1.0/0.95) 2."""
import json, sys, re, urllib.request, statistics as st
from concurrent.futures import ThreadPoolExecutor
base, tag, tools_path, chats_path, out = sys.argv[1:6]
TOOLS = json.load(open(tools_path)); CH = json.load(open(chats_path))
CHATS = ["ad2ab411", "18ae6a99", "b67e5e9d"]          # 09-07 identity / coding / knowledge
H=re.compile(r"[가-힣]"); L=re.compile(r"[A-Za-z]")
def hr(s):
    h=len(H.findall(s)); l=len(L.findall(s)); return h/(h+l) if h+l else None
def ask(msgs, tools, samp):
    body={"model":"alpha","messages":msgs,"max_tokens":6144,"chat_template_kwargs":{"enable_thinking":True},**samp}
    if tools: body["tools"]=tools
    r=urllib.request.Request(base+"/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(r,timeout=1200)); ch=d["choices"][0]; m=ch["message"]
    return dict(reasoning=(m.get("reasoning") or m.get("reasoning_content") or ""), content=(m.get("content") or ""), finish=ch["finish_reason"], tool_calls=[tc["function"]["name"] for tc in (m.get("tool_calls") or [])], usage=d.get("usage",{}))
jobs=[]
for cid in CHATS:
    rows=[r for r in CH if r["chat"]==cid]
    hist=[]
    for r in rows:
        if r["role"]=="user":
            hist.append({"role":"user","content":r["content"]})
            for cond,tools in (("tools25",TOOLS),("none",None)):
                for sname,samp,n in (("greedy",{"temperature":0.0},1),("default",{},2)):
                    for i in range(n): jobs.append(dict(chat=cid,turn=len([h for h in hist if h["role"]=="user"]),q=r["content"],cond=cond,samp=sname,i=i,msgs=list(hist),tools=tools,sp=samp))
        else:
            hist.append({"role":"assistant","content":r["content"]})
with ThreadPoolExecutor(8) as ex: res=list(ex.map(lambda j: ask(j["msgs"],j["tools"],j["sp"]), jobs))
recs=[]
for j,r in zip(jobs,res):
    recs.append(dict(chat=j["chat"],turn=j["turn"],q=j["q"][:80],cond=j["cond"],samp=j["samp"],i=j["i"],r_chars=len(r["reasoning"]),r_hangul=hr(r["reasoning"]),c_chars=len(r["content"]),c_hangul=hr(r["content"]),finish=r["finish"],tool_calls=r["tool_calls"],ghost_call=bool(r["tool_calls"]) or ("<tool_call>" in r["content"]),reasoning=r["reasoning"][:800],content=r["content"][:600]))
json.dump(dict(tag=tag,rows=recs),open(out,"w"),ensure_ascii=False,indent=1)
print(f"### {tag}")
print(f"{'cond':8s} {'samp':8s} {'n':>3s} {'R med':>6s} {'R en%':>6s} {'R ko%':>6s} {'C hangul':>8s} {'ghost':>5s} {'empty':>5s} {'cut':>4s}")
for cond in ("tools25","none"):
    for sname in ("greedy","default"):
        sub=[x for x in recs if x["cond"]==cond and x["samp"]==sname]
        rh=[x["r_hangul"] for x in sub if x["r_hangul"] is not None]
        print(f"{cond:8s} {sname:8s} {len(sub):3d} {int(st.median([x['r_chars'] for x in sub])):6d} {100*sum(h<0.3 for h in rh)/len(rh):5.0f}% {100*sum(h>=0.7 for h in rh)/len(rh):5.0f}% {st.mean([x['c_hangul'] for x in sub if x['c_hangul'] is not None]):8.2f} {sum(x['ghost_call'] for x in sub):5d} {sum(x['c_chars']==0 for x in sub):5d} {sum(x['finish']=='length' for x in sub):4d}")
