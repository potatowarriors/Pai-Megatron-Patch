#!/usr/bin/env python3
"""export_sft.py — P3: generate_v3.py 산출(gen_A/gen_B) → 변환기 입력(Chat-v3 스키마) 2종: thinking / no-think 파생.

- 마지막 턴만 학습: `metadata.train_turns` = [False]*(n-1)+[True] (Chat-v3 동일 규약; 없으면 변환기가 전 턴 학습 → 실사용자 멀티턴의
  원 모델 히스토리 답을 배우게 된다). 시스템 프롬프트(identity·철저 사고)는 생성 시에만 쓰였고 행에 없다(identity 셋 별도).
- no-think 파생: 행의 결정적 해시로 `--nothink-ratio`(기본 0.30) 몫을 골라 reasoning_content 를 제거(NVIDIA 방식: 사고로 만든 답을 무사고 타깃으로).
  `--nothink-mode disjoint`(기본: 70/30 서로소) | `duplicate`(전량 thinking + 30% 무사고 복제).
- 최종 게이트 재검: special token·`<|endoftext|>` 리터럴·한자·벤더 자기귀속·빈 답변. 리젝은 `<out>.rejects.jsonl`.
- 출처 태그: metadata.seed_dataset / seed_file / license_note(lmsys 유래 = 사내 연구 전용) / model(생성 교사) / judge.
사용: python3 export_sft.py --inputs out/p1/gen_A.jsonl,out/p1/gen_B.jsonl --out-prefix out/p1/export/kochat_v3
"""
import argparse, json, re, hashlib, collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_v3 import SELF_ATTR, SPECIAL, HANJA, ratios
EOD = re.compile(r"<\|endoftext\|>|<\|im_end\|>|<\|im_start\|>")

def final_gate(msgs):
    a = msgs[-1]; c = a.get("content") or ""; r = a.get("reasoning_content") or ""
    if not c.strip(): return "empty_content"
    for m in msgs:
        t = (m.get("content") or "") + (m.get("reasoning_content") or "")
        if SPECIAL.search(t) or EOD.search(t): return "special_token"
    if ratios(r)[1] > 0.02 or ratios(c)[1] > 0.01: return "hanja"
    if SELF_ATTR.search(c) or SELF_ATTR.search(r): return "self_attribution"
    return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True); ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--nothink-ratio", type=float, default=0.30); ap.add_argument("--nothink-mode", choices=["disjoint", "duplicate"], default="disjoint")
    ap.add_argument("--exclude-origin", default="", help="쉼표 구분 seed_dataset 접두(예: lmsys) 제외")
    a = ap.parse_args(); os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)
    excl = [x for x in a.exclude_origin.split(",") if x]
    st = collections.Counter(); seen = set()
    ft = open(a.out_prefix + "_think.jsonl", "w"); fn = open(a.out_prefix + "_nothink.jsonl", "w"); fr = open(a.out_prefix + ".rejects.jsonl", "w")
    for path in a.inputs.split(","):
        for line in open(path):
            r = json.loads(line); cid = r["conv_id"]
            if cid in seen: st["dup_conv_id"] += 1; continue
            seen.add(cid)
            so = r.get("seed_origin") or {}; sd = so.get("seed_dataset") or so.get("dataset") or r.get("source") or "unknown"
            if excl and any(sd.lower().startswith(x.lower()) for x in excl): st["excluded_origin"] += 1; continue
            msgs = [{k: v for k, v in m.items() if k in ("role", "content", "reasoning_content")} for m in r["messages"]]
            why = final_gate(msgs)
            if why: st["rej_" + why] += 1; fr.write(json.dumps({"conv_id": cid, "why": why, "content": msgs[-1]["content"][:300]}, ensure_ascii=False) + "\n"); continue
            lic = so.get("license_note") or ("internal-research-only" if "lmsys" in sd.lower() else None)
            meta = {"train_turns": [False] * (len(msgs) - 1) + [True], "model": r["teacher"], "judge": r["ko_synthesis"].get("judge"), "verdict": r["ko_synthesis"].get("verdict"),
                    "pipeline": "ko_chat_v3", "seed_dataset": sd, "seed_file": r.get("seed_file"), "task_type": r.get("task_type"), "mode": r.get("mode"), "license_note": lic}
            uuid = "kochat_v3:" + hashlib.sha1(cid.encode()).hexdigest()[:24]
            h = int(hashlib.sha1(("nothink|" + cid).encode()).hexdigest(), 16) % 10000 < a.nothink_ratio * 10000
            think_row = {"messages": msgs, "uuid": uuid, "metadata": {**meta, "think": True}}
            nt_msgs = msgs[:-1] + [{"role": "assistant", "content": msgs[-1]["content"]}]
            nothink_row = {"messages": nt_msgs, "uuid": uuid + ":nt", "metadata": {**meta, "think": False, "derived_from": uuid}}
            if a.nothink_mode == "duplicate" or not h: ft.write(json.dumps(think_row, ensure_ascii=False) + "\n"); st["think"] += 1
            if h: fn.write(json.dumps(nothink_row, ensure_ascii=False) + "\n"); st["nothink"] += 1
            st["multi_turn" if len(msgs) > 2 else "single_turn"] += 1; st["teacher_" + r["teacher"]] += 1
    print("EXPORT", json.dumps(st, ensure_ascii=False))

if __name__ == "__main__":
    main()
