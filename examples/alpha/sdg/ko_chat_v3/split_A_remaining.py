#!/usr/bin/env python3
"""split_A_remaining.py — P2-A 잔여 시드를 GLM 2대(main1/sub1)로 나눈다 (2026-09-11 사용자 승인: sub1 GLM 병행·심판 보류).
잔여 = seeds_p1_A − gen_A 채택 − gen_A 양샘플 리젝(구제 루프가 맡음) → conv_id sha1 홀짝으로 A1(main1)/A2(sub1). 재실행 시 gen_A1/gen_A2 완료분은 generate_v3 의 재개가 건너뜀.
"""
import json,hashlib,os
seeds=[json.loads(l) for l in open("out/p1/seeds_p1_A.jsonl")]
done=set()
for f in ("out/p1/gen_A.jsonl","out/p1/gen_A.rejects.jsonl"):
    for l in open(f):
        try: done.add(json.loads(l)["conv_id"])
        except Exception: pass
rem=[s for s in seeds if s["conv_id"] not in done]
a1=[s for s in rem if int(hashlib.sha1(s["conv_id"].encode()).hexdigest(),16)%4 in (0,1)]; a2=[s for s in rem if s not in a1]
a2=[s for s in rem if int(hashlib.sha1(s["conv_id"].encode()).hexdigest(),16)%4 in (2,3)]
for name,rs in (("A1",a1),("A2",a2)):
    rs.sort(key=lambda r: (r.get("seed_origin", {}).get("dataset") == "ko_context_grid"))   # S4 후순위 유지
    open(f"out/p1/seeds_p1_{name}.jsonl","w").write("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rs))
print(f"A seeds {len(seeds)} done+rejected {len(done)} remaining {len(rem)} → A1(main1) {len(a1)} A2(sub1) {len(a2)}")
