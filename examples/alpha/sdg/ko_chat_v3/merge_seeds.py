#!/usr/bin/env python3
"""merge_seeds.py — P1 시드 4종(현지화·기사·실사용자·한국맥락, 교사별 파일) 병합 → 생성 교사별 A/B 두 파일.
배정은 conv_id 의 sha1 홀짝(결정적) → 시드 파일이 자라도(작업 진행 중 재병합) 이미 생성된 행의 배정이 바뀌지 않는다.
  A → GLM 생성·DSV4 심판, B → DSV4 생성·GLM 심판 (사용자 결정 1:1, 2026-09-10). 프롬프트 작가(GLM/DSV4)와 생성 교사는 독립 → 4 조합 모두 등장.
사용: python3 merge_seeds.py --out-prefix out/p1/seeds_p1   → out/p1/seeds_p1_A.jsonl, out/p1/seeds_p1_B.jsonl
"""
import argparse, glob, hashlib, json, collections, os
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="out/p1"); ap.add_argument("--out-prefix", default="out/p1/seeds_p1"); a = ap.parse_args()
files = sorted(f for f in glob.glob(os.path.join(a.dir, "seeds_*.jsonl")) if not f.endswith(".rejects.jsonl") and "seeds_p1_" not in f and "smoke" not in f)
seen, rows, cnt = set(), [], collections.Counter()
for f in files:
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if r["conv_id"] in seen: continue
        seen.add(r["conv_id"]); r["seed_file"] = os.path.basename(f); rows.append(r); cnt[os.path.basename(f)] += 1
split = {"A": [], "B": []}
for r in rows: split["A" if int(hashlib.sha1(r["conv_id"].encode()).hexdigest(), 16) % 2 == 0 else "B"].append(r)
for k, rs in split.items():
    with open(f"{a.out_prefix}_{k}.jsonl", "w") as o:
        for r in rs: o.write(json.dumps(r, ensure_ascii=False) + "\n")
print("total", len(rows), "A", len(split["A"]), "B", len(split["B"]), "| by file", dict(cnt))
