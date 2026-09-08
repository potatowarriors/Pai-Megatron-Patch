#!/usr/bin/env python3
"""gen_openmath_tasks.py — OpenMathReasoning(문제 + expected_answer) → Harbor 터미널 과제 (P2 경로 (a) 수학 재포맷).

교사 호출 없음: 정답이 있으므로 채점은 정답 비교, oracle 은 정답을 그대로 기록. 정수·소수·분수형 정답만 채택
(LaTeX 식 정답은 채점이 불안정). 과제: /app/problem.md 를 읽고 최종 답을 /app/answer.txt 에 한 줄로 쓴다 —
지시문이 python3/sympy 로 계산·검산하도록 유도해 터미널을 계산기로 쓰는 행동을 학습시킨다.

사용: python3 gen_openmath_tasks.py --out <root>/<batch> --n 2000 [--offset 0] [--shard-files 0,1,2,3]
"""
import argparse, glob, hashlib, json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import short_hash, slug, write_task  # noqa: E402

SEED_GLOB = "/home/work/Datasets/LL_datasets/midtraining/OpenMathReasoning/**/*.parquet"
ANS_RE = re.compile(r"^-?\d+(\.\d+)?$|^-?\d+/\d+$")

INSTR_TEMPLATES = [
    """The file /app/problem.md contains a math problem. Solve it and write the final answer, and nothing else, as a single line in /app/answer.txt.

Use the tools available in the shell to compute and verify — for example `python3 -c '...'` with the standard library, or `python3` with `sympy`, which is installed. Answers are graded as exact values: give integers as plain integers (e.g. `42`), rational numbers as reduced fractions (e.g. `3/4`) or decimals with at least 6 significant digits (e.g. `0.333333`). Do not include units, text, or LaTeX in the answer file.""",
    """Read the problem in /app/problem.md and produce the final numeric answer in /app/answer.txt (one line, the value only).

You may reason on paper, but verify the result numerically with `python3` (sympy and numpy are available) before writing the file. Format: plain integer, reduced fraction like `7/12`, or a decimal with 6+ significant digits. No units, no words, no LaTeX.""",
    """Solve the math problem stated in /app/problem.md.

Write only the final answer to /app/answer.txt as one line: an integer, a reduced fraction (`a/b`), or a decimal accurate to 1e-6. Use python3 (sympy is installed) in the terminal to compute and double-check — grading compares the value exactly (fractions/integers) or within a 1e-6 relative tolerance (decimals).""",
]

TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
if [ ! -f /app/answer.txt ]; then echo "answer.txt missing"; echo 0 > /logs/verifier/reward.txt; exit 0; fi
if python3 /tests/check.py; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""

CHECK_PY = r'''import re, sys
from fractions import Fraction
exp = open("/tests/expected.ans").read().strip()
got = open("/app/answer.txt").read().strip().splitlines()
got = got[-1].strip() if got else ""
got = got.replace("$", "").replace("\\boxed{", "").rstrip("}").replace(",", "").strip()
def parse(s):
    s = s.strip()
    if re.fullmatch(r"-?\d+/\d+", s): return Fraction(s), "frac"
    if re.fullmatch(r"-?\d+", s): return Fraction(int(s)), "int"
    if re.fullmatch(r"-?\d*\.\d+(e-?\d+)?|-?\d+e-?\d+", s, re.I): return float(s), "dec"
    return None, None
e, et = parse(exp); g, gt = parse(got)
if g is None:
    print(f"unparseable answer: {got!r}"); sys.exit(1)
if et in ("frac", "int") and gt in ("frac", "int"):
    ok = (e == g)
else:
    ef, gf = float(e), float(g)
    ok = abs(ef - gf) <= 1e-6 * max(1.0, abs(ef))
print(f"expected={exp} got={got} -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
'''


def load_problems(shard_files, offset, need, seed=0):
    import pyarrow.parquet as pq
    fs = sorted(glob.glob(SEED_GLOB, recursive=True))
    fs = [fs[i] for i in shard_files if i < len(fs)]
    seen, out = set(), []
    for f in fs:
        t = pq.read_table(f, columns=["problem", "expected_answer", "problem_source", "problem_type"])
        for row in t.to_pylist():
            p, a = (row["problem"] or "").strip(), str(row["expected_answer"] or "").strip()
            if not (80 <= len(p) <= 2000) or not ANS_RE.match(a):
                continue
            if re.search(r"https?://|\.png|\.jpg|\[asy\]", p, re.I):
                continue
            if re.search(r"\bprove\b|\bshow that\b", p, re.I):
                continue
            h = hashlib.md5(p.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            out.append({"hash": h, "problem": p, "answer": a, "source": row["problem_source"] or "", "type": row["problem_type"] or ""})
    random.Random(seed).shuffle(out)
    return out[offset:offset + need]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--offset", type=int, default=0); ap.add_argument("--shard-files", default="0,1,2,3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    probs = load_problems([int(x) for x in a.shard_files.split(",")], a.offset, a.n)
    t0 = time.time(); n = 0
    with open(os.path.join(a.out, "GEN_MANIFEST.jsonl"), "a") as mf:
        for prob in probs:
            name = f"om-{slug(prob['problem'][:60], 22)}-{prob['hash'][:6]}"
            instr = random.Random(prob["hash"]).choice(INSTR_TEMPLATES)
            delim = "ANS_" + short_hash(prob["hash"], 6)
            solve = f"#!/bin/bash\nmkdir -p /app\ncat > /app/answer.txt <<'{delim}'\n{prob['answer']}\n{delim}\n"
            write_task(a.out, name, instr, {"problem.md": prob["problem"].rstrip() + "\n"}, TEST_SH,
                       {"check.py": CHECK_PY, "expected.ans": prob["answer"] + "\n"}, solve,
                       category="math", difficulty="medium", agent_timeout=600,
                       tags=["openmath-reasoning", "answer-file", prob["source"]],
                       meta={"seed": "OpenMathReasoning", "seed_source": prob["source"], "seed_type": prob["type"]})
            mf.write(json.dumps({"task": name, "hash": prob["hash"], "source": prob["source"]}) + "\n"); n += 1
    print(f"[openmath] tasks={n}/{len(probs)} → {a.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
