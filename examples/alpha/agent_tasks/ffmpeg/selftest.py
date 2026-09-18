"""Verifier self-test: the gate that must pass before any model is scored.

For every task:  reference -> PASS · each alt -> PASS · each negative -> FAIL · empty out/ -> FAIL.
A reference/alt failure is a false negative (verifier too strict or task broken);
a negative passing is a false positive (verifier hole). Either one fails the gate.

  python selftest.py [--set pilot|v0] [--only ID ...] [--workers 4] [--report selftest_report.json]
"""
import argparse
import json
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from checks import verify  # noqa: E402
from env import prepare, reset_output, run_commands  # noqa: E402
import importlib  # noqa: E402


def _psnrs(result):
    """(psnr, check_ok) for every frames_match check that produced a measurement."""
    return [(float(m.group(1)), r["ok"]) for r in result["results"] if r["type"] == "frames_match"
            for m in [re.search(r"psnr=([0-9.]+|inf)", r["detail"])] if m]


def run_task(task):
    cases = []
    with tempfile.TemporaryDirectory(prefix=f"ffb_{task['id']}_") as root:
        try:
            work, ref = prepare(task, root)
        except Exception as e:
            return {"id": task["id"], "level": task["level"], "error": str(e)[-600:], "cases": []}
        plan = [("reference", "reference", task["reference"], True)]
        plan += [("alt", a["why"], a["cmds"], True) for a in task["alts"]]
        plan += [("negative", n["why"], n["cmds"], False) for n in task["negatives"]]
        plan += [("negative", "no output at all", [], False)]
        for kind, why, cmds, want in plan:
            reset_output(work)
            ran, log = run_commands(cmds, work)
            res = verify(task, work, ref)
            cases.append({"kind": kind, "why": why, "want_pass": want, "got_pass": res["passed"],
                          "cmds_ok": ran, "psnr": _psnrs(res),
                          "failed_checks": [f"{r['type']}: {r['detail']}" for r in res["results"] if not r["ok"]],
                          "cmd_error": None if ran else log[-1]["stderr"][-300:]})
    return {"id": task["id"], "level": task["level"], "cases": cases}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="pilot", help="task module suffix: pilot | v0")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report", default=str(Path(__file__).parent / "selftest_report.json"))
    args = ap.parse_args()
    all_tasks = importlib.import_module(f"tasks_{args.set}").TASKS
    tasks = [t for t in all_tasks if not args.only or t["id"] in args.only]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        reports = list(ex.map(run_task, tasks))

    bad, counts, pass_psnr, fail_psnr = [], {"reference": [0, 0], "alt": [0, 0], "negative": [0, 0]}, [], []
    for rep in reports:
        if "error" in rep:
            bad.append(f"{rep['id']}: PREPARE ERROR {rep['error']}")
        for c in rep["cases"]:
            ok = c["got_pass"] == c["want_pass"]
            counts[c["kind"]][0] += ok
            counts[c["kind"]][1] += 1
            for p, check_ok in c["psnr"]:
                if p != float("inf"):
                    (pass_psnr if check_ok else fail_psnr).append(p)
            if not ok:
                label = "FALSE NEGATIVE" if c["want_pass"] else "FALSE POSITIVE"
                bad.append(f"{rep['id']} [{c['kind']}: {c['why']}] {label} — {c['failed_checks'] or c['cmd_error'] or 'all checks passed'}")

    Path(args.report).write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"tasks {len(reports)}")
    for kind, (ok, n) in counts.items():
        print(f"  {kind:9s} {ok}/{n} as expected")
    if pass_psnr and fail_psnr:
        print(f"  frames_match: accepted min {min(pass_psnr):.1f} dB | rejected max {max(fail_psnr):.1f} dB")
    for line in bad:
        print("  !!", line)
    print("SELFTEST", "PASS" if not bad else f"FAIL ({len(bad)})")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
