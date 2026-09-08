#!/usr/bin/env python3
"""gen_opencode_tasks.py — OpenCodeReasoning(문제 + 정답 파이썬) → Harbor 터미널 과제 (P2 경로 (a) 재포맷).

과제 형태: /app/problem.md 를 읽고 /app/solution.py (stdin→stdout) 를 작성. /app/samples/ 에 공개 예제 2개,
/tests/cases/ 에 숨은 테스트 ≥5개. 테스트 입력은 교사가 쓴 입력 생성기(gen.py)로 만들고, 기대 출력은 **정답 코드를
실제로 실행**해 얻는다 (정답 코드가 실패하는 입력은 버림 → 생성기 오류가 과제를 오염시키지 않는다).
oracle(solve.sh) = 정답 코드를 /app/solution.py 로 기록 → reward 1 이어야 하고, nop → solution.py 없음 → reward 0.

사용:
  TEACHER_BASE=http://sub1:8300/v1 python3 gen_opencode_tasks.py --out <root>/<batch> --n 1000 [--offset 0] [--workers 32]
    [--effort high] [--shard-files 0,1,2,3]
산출: <out>/<task-name>/… + <out>/GEN_MANIFEST.jsonl (과제별 통계) + 표준출력 요약
"""
import argparse, glob, hashlib, json, os, random, re, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import extract_code_block, short_hash, slug, teacher, write_task  # noqa: E402

SEED_GLOB = "/home/work/Datasets/LL_datasets/midtraining/OpenCodeReasoning/**/*.parquet"
CASE_SEP = "===CASE==="

GEN_PROMPT = """You are preparing hidden test cases for a competitive-programming problem.

Problem statement:
<problem>
{problem}
</problem>

A correct reference solution (Python 3, reads stdin, writes stdout):
<solution>
{solution}
</solution>

Write a Python 3 script `gen.py` that prints {k} diverse, VALID test inputs for this problem, separated by a line containing exactly `{sep}`.
Requirements:
- Every input must satisfy the constraints in the statement exactly (format, ranges, counts, trailing newline).
- Include small edge cases (minimum sizes, boundary values) and a few random medium-sized cases. Keep each input under 20 KB and fast to solve (< 1 second).
- Use only the standard library and `random` with a fixed seed. Print inputs only, nothing else.
Return only the script in a single ```python code block."""

INSTR_TEMPLATES = [
    """The file /app/problem.md describes a programming problem. Write a Python 3 program at /app/solution.py that reads the input from standard input and writes the answer to standard output exactly in the format the problem specifies.

Sample inputs and expected outputs are provided in /app/samples/ as `N.in` / `N.out`. Verify your program against them, e.g. `python3 /app/solution.py < /app/samples/1.in`. Hidden tests use additional inputs within the stated constraints. Do not modify the files in /app/samples/.""",
    """Solve the programming task described in /app/problem.md.

Deliverable: /app/solution.py (Python 3). It must read from stdin and print to stdout in the exact output format of the problem. Two sample cases live in /app/samples/ (`1.in`/`1.out`, `2.in`/`2.out`) — make sure `python3 /app/solution.py < /app/samples/1.in` reproduces `1.out`. Additional hidden inputs within the constraints will be used for grading. Leave /app/samples/ untouched.""",
    """You are in /app. Read problem.md and implement a solution.

Write your program to /app/solution.py using Python 3; it reads standard input and writes standard output as specified. Check it against the provided samples in /app/samples/ (diff your output with the `.out` files). Grading runs hidden test inputs that follow the same constraints. Do not edit anything under /app/samples/.""",
]

TEST_SH = """#!/bin/bash
# 숨은 테스트: /tests/cases/*.in 을 /app/solution.py 에 넣어 .out 과 비교 (공백 정규화). 전부 통과 = 1.
mkdir -p /logs/verifier
cd /app || { echo 0 > /logs/verifier/reward.txt; exit 0; }
if [ ! -f /app/solution.py ]; then echo "solution.py missing"; echo 0 > /logs/verifier/reward.txt; exit 0; fi
if python3 /tests/check.py; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""

CHECK_PY = """import glob, os, subprocess, sys
def norm(s):
    return [line.rstrip() for line in s.replace("\\r\\n", "\\n").strip().split("\\n")]
ok = fail = 0
for inp in sorted(glob.glob("/tests/cases/*.in")):
    exp = open(inp[:-3] + ".out").read()
    try:
        r = subprocess.run([sys.executable, "/app/solution.py"], input=open(inp).read(), capture_output=True, text=True, timeout=10)
        got = r.stdout
        passed = (r.returncode == 0) and (norm(got) == norm(exp))
    except subprocess.TimeoutExpired:
        passed = False; got = "<timeout>"
    ok += passed; fail += (not passed)
    print(("PASS " if passed else "FAIL ") + os.path.basename(inp))
print(f"{ok} passed, {fail} failed")
sys.exit(0 if fail == 0 else 1)
"""


def load_problems(shard_files, offset, need, seed=0, exclude_prefixes=None):
    import pyarrow.parquet as pq
    fs = sorted(glob.glob(SEED_GLOB, recursive=True))
    fs = [fs[i] for i in shard_files if i < len(fs)]
    seen, out = set(), []
    excl = exclude_prefixes or set()   # 이미 과제로 만든 문제의 해시 앞 6자 (과제명 끝 6자) — 배치 간 중복 방지
    for f in fs:
        t = pq.read_table(f, columns=["id", "input", "solution", "difficulty", "source", "dataset", "license"])
        for row in t.to_pylist():
            p, s = (row["input"] or "").strip(), (row["solution"] or "").strip()
            h = hashlib.md5(p.encode()).hexdigest()
            if h in seen or h[:6] in excl:
                continue
            seen.add(h)
            if not (300 <= len(p) <= 4000 and 30 <= len(s) <= 4000):
                continue
            if not ("input(" in s or "stdin" in s):
                continue
            # 배치 1 실측(2026-09-08): HARD/VERY_HARD 는 성공 32%, 성공도 15분 상한 직전이라 핸드셰이크 없이 끝남 → 제외
            if (row["difficulty"] or "") in ("HARD", "VERY_HARD"):
                continue
            if re.search(r"https?://|\.png|\.jpg|<image", p, re.I):
                continue
            out.append({"hash": h, "problem": p, "solution": s, "difficulty": row["difficulty"] or "UNKNOWN",
                        "source": row["source"], "dataset": row["dataset"], "license": row["license"], "id": row["id"]})
    random.Random(seed).shuffle(out)
    return out[offset:offset + need]


def run_py(code, stdin_text, timeout):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "prog.py")
        open(p, "w").write(code)
        try:
            r = subprocess.run([sys.executable, p], input=stdin_text, capture_output=True, text=True, timeout=timeout, cwd=td)
            return r.returncode, r.stdout, r.stderr[-300:]
        except subprocess.TimeoutExpired:
            return -9, "", "timeout"


def build_one(prob, out_root, k_cases, effort, min_cases):
    t0 = time.time()
    rec = {"hash": prob["hash"], "difficulty": prob["difficulty"], "source": prob["source"]}
    try:
        content, _rc, usage = teacher([{"role": "user", "content": GEN_PROMPT.format(
            problem=prob["problem"], solution=prob["solution"], k=k_cases, sep=CASE_SEP)}],
            max_tokens=6144, reasoning_effort=effort)
        rec["teacher_tokens"] = (usage.get("completion_tokens") or 0)
        gen = extract_code_block(content, "python")
        if not gen:
            rec["drop"] = "no_gen_code"; return rec
        rc, out, err = run_py(gen, "", 30)
        if rc != 0 or not out.strip():
            rec["drop"] = "gen_failed"; rec["err"] = err; return rec
        raw_cases = [c.strip("\n") for c in out.split(CASE_SEP)]
        raw_cases = [c + "\n" for c in raw_cases if c.strip() and len(c) < 20000]
        cases = []
        seen = set()
        for c in raw_cases:
            if c in seen:
                continue
            seen.add(c)
            rc2, o2, _e = run_py(prob["solution"], c, 5)
            if rc2 == 0 and o2.strip():
                cases.append((c, o2))
        if len(cases) < min_cases:
            rec["drop"] = "too_few_valid_cases"; rec["valid_cases"] = len(cases); return rec
        samples, hidden = cases[:2], cases[2:]
        name = f"oc-{slug(prob['problem'][:60], 22)}-{prob['hash'][:6]}"   # ≤32자 (Harbor 트라이얼 이름 절단 대비)
        files = {"problem.md": prob["problem"].rstrip() + "\n"}
        for i, (ci, co) in enumerate(samples, 1):
            files[f"samples/{i}.in"] = ci; files[f"samples/{i}.out"] = co
        test_files = {"check.py": CHECK_PY}
        for i, (ci, co) in enumerate(hidden, 1):
            test_files[f"cases/{i}.in"] = ci; test_files[f"cases/{i}.out"] = co
        delim = "PYSOL_" + short_hash(prob["hash"], 6)
        solve = f"#!/bin/bash\nmkdir -p /app\ncat > /app/solution.py <<'{delim}'\n{prob['solution'].rstrip()}\n{delim}\n"
        instr = random.Random(prob["hash"]).choice(INSTR_TEMPLATES)
        diff = {"EASY": "easy", "MEDIUM": "medium", "MEDIUM_HARD": "medium", "HARD": "hard", "VERY_HARD": "hard"}.get(
            prob["difficulty"], "medium")
        write_task(out_root, name, instr, files, TEST_SH, test_files, solve, category="programming",
                   difficulty=diff, agent_timeout=600, tags=["opencode-reasoning", "stdin-stdout", prob["source"] or ""],
                   meta={"seed": "OpenCodeReasoning", "seed_id": prob["id"], "seed_license": prob["license"],
                         "hidden_cases": len(hidden), "generator_tokens": rec["teacher_tokens"]})
        rec.update({"task": name, "hidden_cases": len(hidden), "sec": round(time.time() - t0, 1)})
        return rec
    except Exception as e:  # noqa: BLE001
        rec["drop"] = "exception:" + type(e).__name__; rec["err"] = str(e)[:200]; return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--shard-files", default="0,1,2,3")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--k-cases", type=int, default=10)
    ap.add_argument("--min-cases", type=int, default=6)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--exclude-task-roots", default="", help="콤마 목록: 이 디렉토리들의 oc-* 과제명 끝 6자 해시를 제외")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    excl = set()
    for root in [r for r in a.exclude_task_roots.split(",") if r]:
        excl |= {d[-6:] for d in os.listdir(root) if d.startswith("oc-") and os.path.isdir(os.path.join(root, d))}
    probs = load_problems([int(x) for x in a.shard_files.split(",")], a.offset, a.n, exclude_prefixes=excl)
    print(f"[gen] excluded hashes: {len(excl)}", flush=True)
    print(f"[gen] problems loaded: {len(probs)} (offset {a.offset}) → {a.out}", flush=True)
    t0 = time.time(); recs = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, rec in enumerate(ex.map(lambda p: build_one(p, a.out, a.k_cases, a.effort, a.min_cases), probs), 1):
            recs.append(rec)
            if i % 25 == 0:
                ok = sum(1 for r in recs if "task" in r)
                print(f"[gen] {i}/{len(probs)} tasks={ok} ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(a.out, "GEN_MANIFEST.jsonl"), "a") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in recs if "task" in r]
    from collections import Counter
    drops = Counter(r.get("drop") for r in recs if "drop" in r)
    toks = sum(r.get("teacher_tokens", 0) for r in recs)
    print(f"[gen] done: tasks={len(ok)}/{len(probs)} drops={dict(drops)} teacher_completion_tokens={toks:,} "
          f"hidden_cases_mean={sum(r['hidden_cases'] for r in ok)/max(len(ok),1):.1f} wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
