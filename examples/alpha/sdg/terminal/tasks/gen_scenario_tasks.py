#!/usr/bin/env python3
"""gen_scenario_tasks.py — 교사(GLM-5.3-Flash)가 터미널 시나리오 과제를 통째로 합성 (P2 경로 (b)).

카테고리 카드 × 소주제 × 난이도를 무작위로 뽑아 교사에게 JSON 과제 명세를 받는다:
  instruction(에이전트가 보는 과제) · files(/app 에 놓을 자료, setup.sh 는 빌드 시 실행) · dockerfile_extra ·
  test_sh + test_files(/tests, reward 0/1 기록) · solve_sh(정답 절차)
유효성은 validate_tasks.sh (Harbor oracle → 1, nop → 0) 로 판정한다. 여기서는 구문 검사·금지 항목만 거른다.

사용: TEACHER_BASE=http://sub1:8300/v1 python3 gen_scenario_tasks.py --out <root>/<batch> --n 300 [--workers 32] [--seed 0]
"""
import argparse, json, os, random, re, subprocess, sys, tempfile, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import extract_json_block, short_hash, slug, teacher, write_task  # noqa: E402

CARDS = {
    "log-analysis": ["web server access logs: extract top clients/paths/error rates into a report file",
                     "application logs with mixed formats: find the root cause window and summarize",
                     "multi-file rotated logs (gz): count events per hour into CSV",
                     "syslog-like auth logs: detect brute-force patterns and write offending IPs"],
    "data-wrangling": ["messy CSV cleanup (quotes, encodings, duplicates) into a normalized CSV",
                       "join two datasets (CSV + JSON) and compute aggregates with jq/awk/pandas",
                       "convert nested JSON to flat TSV with specific columns",
                       "sqlite3 database: write SQL to answer questions, export results"],
    "build-debug": ["a small C program with a Makefile that fails to build — fix and produce the binary",
                    "a Python package whose tests fail due to 2-3 bugs — make pytest pass",
                    "a shell script with quoting/globbing bugs — fix so a checker passes",
                    "a broken pip-installable package layout (missing __init__, wrong entry point) — fix"],
    "sysadmin": ["cron-style schedule file: parse and produce the next-run listing",
                 "disk usage investigation: find large/duplicate files and write a cleanup plan script",
                 "user/group/permission setup on a directory tree per a spec (no sudo needed: run as root)",
                 "environment/config drift: compare two config trees and produce a patch"],
    "git-local": ["repository with a bad commit: bisect using a test script, revert it (local only)",
                  "merge two branches with conflicts and make the tests pass",
                  "rewrite history locally: squash WIP commits, fix a commit message, keep tree identical",
                  "recover a deleted file from history and restore it"],
    "shell-scripting": ["write a bash script that processes files per rules (rename/organize by pattern)",
                        "implement a small CLI in bash/python with flags, tested by a checker",
                        "text transformation pipeline with sed/awk producing an exact output file",
                        "batch-rename with regex and collision handling"],
    "config-debug": ["broken YAML/INI configs: validate and fix so a loader script passes",
                     "environment variable and dotenv precedence bug in a startup script",
                     "nginx-like config text: rewrite rules to satisfy a checker (no daemon)",
                     "Makefile variables and pattern rules producing wrong targets"],
    "testing-qa": ["write tests that expose a bug in a module, then fix the bug",
                   "a flaky test depending on ordering/time — make it deterministic",
                   "generate a coverage report and add tests until a threshold is met"],
    "packaging-archive": ["reconstruct a directory from mixed archives (tar/zip/gz) preserving structure",
                          "create a reproducible tarball with specific content and checksums",
                          "split/merge large files and verify integrity with sha256"],
    "text-processing": ["regex extraction across many files into a structured report",
                        "encoding repair (mojibake) and normalization of text files",
                        "markdown table generation from raw data with sorting rules"],
}
DIFF = ["easy", "medium", "medium", "hard"]

PROMPT = """Design ONE self-contained Linux terminal task for evaluating an AI agent that works only through a shell.

Category: {category}
Theme: {theme}
Target difficulty: {difficulty} ({steps} agent steps of work for a competent engineer)
Twist to include: {twist}

Environment facts: Docker container based on python:3.12-slim with bash, coreutils, findutils, grep, sed, gawk, git, jq, curl, make, gcc, sqlite3, tmux, tree, vim-tiny, and Python packages pytest/numpy/pandas/pyyaml. The agent runs as root in /app. There is NO network access. Tests run after the agent finishes by executing /tests/test.sh inside the same container; it must write `1` (pass) or `0` (fail) to /logs/verifier/reward.txt.

Return ONLY a JSON object with these fields:
- "name": short kebab-case name (<= 40 chars)
- "instruction": the task text the agent sees. Must be fully self-contained: exact input locations under /app, exact deliverable paths and formats, and any rules. 80-250 words. Do not mention tests or /tests.
- "files": object mapping relative paths under /app to full text contents (the starting materials: data files, buggy code, configs, etc.). Keep total size under 60 KB. If materials must be generated procedurally (large logs, git history), include a "setup.sh" that creates them deterministically (it runs once at image build time as root, in /app, and is deleted afterwards).
- "dockerfile_extra": optional extra Dockerfile lines (e.g. apt-get/pip installs); usually "".
- "test_sh": bash script content for /tests/test.sh. It must be robust: check deliverables precisely (exact values, file formats, exit codes), be deterministic, finish within 60 seconds, and ALWAYS write /logs/verifier/reward.txt. Put helper checkers in "test_files" and call them (e.g. python3 /tests/check.py).
- "test_files": object mapping relative paths under /tests to contents (may be empty object).
- "solve_sh": bash script content that a correct solver would run from /app to produce the deliverables so that test_sh passes. It must be complete and non-interactive.
Rules: no network, no interactive prompts, no destructive system changes, no reliance on the current date/time unless files provide it, no git push/pull/fetch/clone. The task must be genuinely solvable by reading /app and must not be solvable by doing nothing (tests must fail on the untouched environment)."""

TWISTS = ["an edge case hidden in the data that a naive solution misses", "a second deliverable that summarizes what was done",
          "output must be sorted deterministically with a tie-breaker", "one of the provided files is a red herring",
          "the deliverable must be an executable script that is itself tested with new inputs",
          "strict output format (exact whitespace and header)", "a size/performance constraint that rules out brute force",
          "the starting state contains a partially completed, wrong attempt that must be corrected", "none"]

FORBIDDEN = re.compile(r"\b(git\s+(push|pull|fetch|clone)|curl\s+https?://|wget\s+https?://|pip\s+install\s+[^-]|apt-get\s+install)\b")


def bash_ok(script):
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script); p = f.name
    try:
        return subprocess.run(["bash", "-n", p], capture_output=True, text=True, timeout=10).returncode == 0
    finally:
        os.unlink(p)


def build_one(spec_seed, out_root, effort):
    cat, theme, diff, twist, rnd = spec_seed
    steps = {"easy": "5-8", "medium": "8-15", "hard": "15-25"}[diff]
    rec = {"category": cat, "theme": theme, "difficulty": diff}
    t0 = time.time()
    try:
        # effort high 는 reasoning 이 12k 토큰을 먹어 본문이 잘렸다(2026-09-08 실측: reasoning 11,752/12,000).
        # 명세 생성은 low + JSON 강제(response_format)로: 완성 ≈2k 토큰, 파싱 100% (probe 기록 README §5 P2).
        content, _r, usage = teacher([{"role": "user", "content": PROMPT.format(
            category=cat, theme=theme, difficulty=diff, steps=steps, twist=twist)}], max_tokens=16000,
            reasoning_effort=effort, temperature=1.0, response_format={"type": "json_object"})
        rec["teacher_tokens"] = usage.get("completion_tokens") or 0
        spec = extract_json_block(content)
        if not spec or not all(k in spec for k in ("name", "instruction", "files", "test_sh", "solve_sh")):
            rec["drop"] = "bad_json"; return rec
        files = spec.get("files") or {}; test_files = spec.get("test_files") or {}
        if not isinstance(files, dict) or not isinstance(test_files, dict):
            rec["drop"] = "bad_files"; return rec
        if any(not isinstance(v, str) for v in list(files.values()) + list(test_files.values())):
            rec["drop"] = "nonstring_file"; return rec
        total = sum(len(v) for v in files.values()) + sum(len(v) for v in test_files.values())
        if total > 200_000:
            rec["drop"] = "too_large"; return rec
        blob = spec["instruction"] + spec["test_sh"] + spec["solve_sh"] + "".join(files.values())
        if FORBIDDEN.search(blob) or "BENCHMARK DATA SHOULD NEVER" in blob:
            rec["drop"] = "forbidden"; return rec
        if not bash_ok(spec["test_sh"]) or not bash_ok(spec["solve_sh"]):
            rec["drop"] = "bash_syntax"; return rec
        if "reward.txt" not in spec["test_sh"]:
            rec["drop"] = "no_reward_write"; return rec
        extra = spec.get("dockerfile_extra") or ""
        if "setup.sh" in files:
            extra = extra.rstrip() + "\nCOPY app/ /app/\nRUN bash /app/setup.sh && rm -f /app/setup.sh\n"
            # write_task 는 files 가 있으면 마지막에 COPY 를 또 넣는다 — setup 후 재복사돼도 setup.sh 만 되살아나므로 지운다
            extra += "COPY app/ /app/\nRUN rm -f /app/setup.sh\n"
        name = f"sc-{slug(cat, 14)}-{slug(spec['name'], 30)}-{short_hash(str(rnd) + spec['instruction'], 6)}"
        write_task(out_root, name, spec["instruction"], files, spec["test_sh"], test_files, spec["solve_sh"],
                   category=cat, difficulty=diff, agent_timeout=1200, dockerfile_extra=extra,
                   tags=["scenario", cat, diff], meta={"seed": "teacher-scenario", "theme": theme, "twist": twist,
                                                       "generator_tokens": rec["teacher_tokens"]})
        rec.update({"task": name, "files": len(files), "sec": round(time.time() - t0, 1)})
        return rec
    except Exception as e:  # noqa: BLE001
        rec["drop"] = "exception:" + type(e).__name__; rec["err"] = str(e)[:200]; return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effort", default="low")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    cats = list(CARDS)
    seeds = []
    for i in range(a.n):
        cat = cats[i % len(cats)]; theme = rng.choice(CARDS[cat]); diff = rng.choice(DIFF); twist = rng.choice(TWISTS)
        seeds.append((cat, theme, diff, twist, rng.random()))
    print(f"[scen] generating {a.n} specs → {a.out}", flush=True)
    t0 = time.time(); recs = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, rec in enumerate(ex.map(lambda s: build_one(s, a.out, a.effort), seeds), 1):
            recs.append(rec)
            if i % 25 == 0:
                print(f"[scen] {i}/{a.n} tasks={sum(1 for r in recs if 'task' in r)} ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(a.out, "GEN_MANIFEST.jsonl"), "a") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in recs if "task" in r]
    print(f"[scen] done: tasks={len(ok)}/{a.n} drops={dict(Counter(r.get('drop') for r in recs if 'drop' in r))} "
          f"teacher_tokens={sum(r.get('teacher_tokens', 0) for r in recs):,} wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
