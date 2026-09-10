#!/usr/bin/env python3
"""rescue_rejects.py — 게이트 재보정 후 리젝 파일(사고·답변 전문 보존)을 새 gate() 로 재판정해 통과 샘플을 채택 행으로 구제.
통과 1개면 그대로 채택(verdict None), 2개면 심판(GLM low-effort 등 JUDGE 환경변수) 호출.
사용: python3 rescue_rejects.py --seeds <seeds.jsonl> --rejects <run.rejects.jsonl> --out <run.rescued.jsonl>
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_v3 as G

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--rejects", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
seeds = {json.loads(l)["conv_id"]: json.loads(l) for l in open(a.seeds)}
stats = {"rows": 0, "rescued": 0, "still_rejected": 0, "judged": 0}; why_after = {}
with open(a.out, "w") as out:
    for line in open(a.rejects):
        r = json.loads(line); stats["rows"] += 1
        samples = [x for x in r["rejects"] if x.get("reasoning") is not None and x.get("content") is not None]
        passed = []
        for x in samples:
            s = {"reasoning": x["reasoning"], "content": x["content"], "finish": "stop" if not x["why"].startswith("finish_") else x["why"][7:], "ctoks": 0}
            why = G.gate(s)
            if why: why_after[why] = why_after.get(why, 0) + 1
            else: passed.append(s)
        if not passed: stats["still_rejected"] += 1; continue
        seed = seeds.get(r["conv_id"]); turns = G.build_messages(seed) if seed else None
        if not turns: stats["still_rejected"] += 1; continue
        verdict = None; judge_name = None
        if len(passed) >= 2:
            jt = G.pick_judge(1) if G.TEACHERS[0]["name"] == r["teacher"] else G.pick_judge(0)
            conv_text = "\n".join(f"{t['role']}: {t['content'][:1500]}" for t in turns)
            verdict, _ = G.judge(jt, conv_text, passed[0]["content"], passed[1]["content"]); judge_name = jt["name"]; stats["judged"] += 1
        best = passed[1] if verdict == "B" else passed[0]
        row = {"messages": turns + [{"role": "assistant", "content": best["content"], "reasoning_content": best["reasoning"]}],
               "source": r["source"], "conv_id": r["conv_id"], "teacher": r["teacher"],
               "ko_synthesis": {"pipeline": "ko_chat_v3", "think": True, "rescued": True, "n_samples": len(samples), "kept": len(passed),
                                "judge": judge_name, "verdict": verdict, "rejects": [], "r_hangul": round(G.ratios(best["reasoning"])[0], 2), "ctoks": 0}}
        for k in ("mode", "task_type", "persona", "seed_origin"):
            if seed and k in seed: row[k] = seed[k]
        out.write(json.dumps(row, ensure_ascii=False) + "\n"); stats["rescued"] += 1
print(json.dumps({"stats": stats, "why_after": why_after}, ensure_ascii=False))
