#!/usr/bin/env python3
"""repair_scenario_tasks.py — 검증에 떨어진 시나리오 과제를 실패 증거와 함께 교사에게 되먹여 고친다 (1회 수리 루프).

입력: validate_tasks.sh 를 돌린 과제 루트(INVALID.txt)와 그 검증 job 이름 접두(val-<tag>).
증거: 빌드 실패 → exception_message + trial.log 의 오류 줄, oracle=0 → verifier/test-stdout.txt 꼬리,
      nop=1 → "손대지 않은 환경에서 테스트 통과", oracle=0·nop=1 → "테스트가 뒤집힘".
출력: <out_root>/<같은 과제명>/ 에 수정본 (원본은 보존). 이어서 validate_tasks.sh <out_root> <tag>-r 로 재검증.

사용: TEACHER_BASE=… python3 repair_scenario_tasks.py --root <task_root> --tag <tag> --out <out_root> [--workers 16]
"""
import argparse, json, os, subprocess, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import extract_json_block, teacher, write_task  # noqa: E402
from gen_scenario_tasks import FORBIDDEN, bash_ok  # noqa: E402

SSHC = "/home/work/vidsearch/.ssh-keys/config"

REPAIR_PROMPT = """You previously designed a Linux terminal task for an AI agent. It FAILED validation. Fix it.

How validation works: the task's own reference solution (solve_sh) is executed in a fresh container built from the Dockerfile (python:3.12-slim base + your files under /app; if files contain setup.sh it runs once at build time and is then deleted), then /tests/test.sh runs and must write 1 to /logs/verifier/reward.txt. Separately, test.sh is run on an untouched container and must write 0.

Failure evidence:
<evidence>
{evidence}
</evidence>

Current task specification (JSON):
<spec>
{spec}
</spec>

Return the COMPLETE corrected specification as a JSON object with exactly the same fields ("name", "instruction", "files", "dockerfile_extra", "test_sh", "test_files", "solve_sh"). Keep the task idea, but make solve_sh, test_sh and the files mutually consistent: every file that setup.sh or the solution reads must exist in "files" (or be created by setup.sh), expected values in tests must be derived from the same data and rules the instruction states, tests must fail on the untouched environment and pass after solve_sh. No network, no interactive prompts, no git push/pull/fetch/clone."""


def read_task(d):
    spec = {"name": os.path.basename(d), "instruction": open(os.path.join(d, "instruction.md")).read(), "files": {},
            "dockerfile_extra": "", "test_sh": open(os.path.join(d, "tests", "test.sh")).read(), "test_files": {},
            "solve_sh": open(os.path.join(d, "solution", "solve.sh")).read()}
    app = os.path.join(d, "environment", "app")
    for root, _dirs, files in os.walk(app):
        for f in files:
            p = os.path.join(root, f); spec["files"][os.path.relpath(p, app)] = open(p, errors="replace").read()
    tests = os.path.join(d, "tests")
    for root, _dirs, files in os.walk(tests):
        for f in files:
            if f == "test.sh":
                continue
            p = os.path.join(root, f); spec["test_files"][os.path.relpath(p, tests)] = open(p, errors="replace").read()
    df = open(os.path.join(d, "environment", "Dockerfile")).read().split("WORKDIR /app\n", 1)
    extra = df[1] if len(df) > 1 else ""
    spec["dockerfile_extra"] = "\n".join(l for l in extra.splitlines() if not l.startswith("COPY app/") and "rm -f /app/setup.sh" not in l)
    meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
    return spec, meta


def fetch_evidence(tag, prefixes):
    """alpha-eval 의 val-<tag>-oracle / -nop job 에서 증거 수집 → {prefix: text}"""
    script = f"""
import json, glob, os
out = {{}}
for kind in ("oracle", "nop"):
    for d in glob.glob("/opt/harbor/jobs/val-{tag}-" + kind + "/*__*/"):
        name = os.path.basename(d.rstrip("/")).split("__")[0]
        ev = out.setdefault(name, {{}})
        r = json.load(open(d + "result.json")) if os.path.exists(d + "result.json") else {{}}
        e = r.get("exception_info") or {{}}
        if e.get("exception_type"):
            ev[kind + "_exception"] = (e.get("exception_message") or "")[-1500:]
            try:
                lines = [l for l in open(d + "trial.log", errors="replace").read().splitlines() if "ERROR" in l or "error" in l.lower()]
                ev[kind + "_log"] = "\\n".join(lines[-12:])[-1500:]
            except Exception: pass
        p = d + "verifier/test-stdout.txt"
        if os.path.exists(p):
            ev[kind + "_test_stdout"] = open(p, errors="replace").read()[-1800:]
        rp = d + "verifier/reward.txt"
        ev[kind + "_reward"] = open(rp).read().strip() if os.path.exists(rp) else "NA"
print(json.dumps(out))
"""
    r = subprocess.run(["ssh", "-F", SSHC, "-o", "BatchMode=yes", "alpha-eval", "python3", "-"], input=script,
                       capture_output=True, text=True, timeout=180)
    return json.loads(r.stdout.strip() or "{}")


def evidence_text(ev):
    parts = []
    o, n = ev.get("oracle_reward", "NA"), ev.get("nop_reward", "NA")
    parts.append(f"oracle (reference solution then tests) reward = {o}; untouched environment reward = {n}")
    if ev.get("oracle_exception"):
        parts.append("Environment/build failure during oracle run:\n" + ev["oracle_exception"])
        if ev.get("oracle_log"):
            parts.append("Build log errors:\n" + ev["oracle_log"])
    if o == "0" and ev.get("oracle_test_stdout"):
        parts.append("Test output after running the reference solution:\n" + ev["oracle_test_stdout"])
    if n == "1":
        parts.append("PROBLEM: tests pass on the untouched environment — the task is solvable by doing nothing, or the checks are inverted.")
        if ev.get("nop_test_stdout"):
            parts.append("Test output on untouched environment:\n" + ev["nop_test_stdout"])
    return "\n\n".join(parts)


def repair_one(d, ev, out_root, meta):
    name = os.path.basename(d); rec = {"task": name}
    try:
        spec, meta0 = read_task(d)
        spec_for_prompt = dict(spec)
        spec_for_prompt["files"] = {k: (v if len(v) < 6000 else v[:6000] + "\n…[truncated]") for k, v in spec["files"].items()}
        content, _r, usage = teacher([{"role": "user", "content": REPAIR_PROMPT.format(
            evidence=evidence_text(ev), spec=json.dumps(spec_for_prompt, ensure_ascii=False, indent=1))}],
            max_tokens=16000, reasoning_effort="low", temperature=1.0, response_format={"type": "json_object"})
        rec["teacher_tokens"] = usage.get("completion_tokens") or 0
        new = extract_json_block(content)
        if not new or not all(k in new for k in ("instruction", "files", "test_sh", "solve_sh")):
            rec["drop"] = "bad_json"; return rec
        files = new.get("files") or {}; test_files = new.get("test_files") or {}
        if any(not isinstance(v, str) for v in list(files.values()) + list(test_files.values())):
            rec["drop"] = "nonstring_file"; return rec
        blob = new["instruction"] + new["test_sh"] + new["solve_sh"] + "".join(files.values())
        if FORBIDDEN.search(blob) or not bash_ok(new["test_sh"]) or not bash_ok(new["solve_sh"]) or "reward.txt" not in new["test_sh"]:
            rec["drop"] = "forbidden_or_syntax"; return rec
        extra = new.get("dockerfile_extra") or ""
        if "setup.sh" in files:
            extra = extra.rstrip() + "\nCOPY app/ /app/\nRUN bash /app/setup.sh && rm -f /app/setup.sh\nCOPY app/ /app/\nRUN rm -f /app/setup.sh\n"
        write_task(out_root, name, new["instruction"], files, new["test_sh"], test_files, new["solve_sh"],
                   category=meta0.get("category", "scenario") if meta0 else "scenario",
                   difficulty=meta0.get("difficulty", "medium") if meta0 else "medium", agent_timeout=1200,
                   dockerfile_extra=extra, tags=["scenario", "repaired"], meta={**(meta0 or {}), "repaired": True,
                                                                              "repair_tokens": rec["teacher_tokens"]})
        rec["ok"] = True; return rec
    except Exception as e:  # noqa: BLE001
        rec["drop"] = "exception:" + type(e).__name__; rec["err"] = str(e)[:200]; return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    invalid = [l.split()[0] for l in open(os.path.join(a.root, "INVALID.txt")) if l.strip() and l.split()[0].startswith("sc-")]
    dirs = {}
    for pre in invalid:
        m = [d for d in os.listdir(a.root) if d.startswith(pre) and os.path.isdir(os.path.join(a.root, d))]
        if m:
            dirs[pre] = os.path.join(a.root, m[0])
    ev = fetch_evidence(a.tag, list(dirs))
    print(f"[repair] invalid={len(invalid)} matched_dirs={len(dirs)} evidence={len(ev)}", flush=True)
    t0 = time.time(); recs = []
    with ThreadPoolExecutor(a.workers) as ex:
        for rec in ex.map(lambda kv: repair_one(kv[1], ev.get(kv[0], {}), a.out, None), dirs.items()):
            recs.append(rec)
    ok = sum(1 for r in recs if r.get("ok"))
    print(f"[repair] rewritten={ok}/{len(recs)} drops={dict(Counter(r.get('drop') for r in recs if 'drop' in r))} "
          f"teacher_tokens={sum(r.get('teacher_tokens', 0) for r in recs):,} wall={time.time()-t0:.0f}s → {a.out}")
    with open(os.path.join(a.out, "REPAIR_MANIFEST.jsonl"), "a") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
