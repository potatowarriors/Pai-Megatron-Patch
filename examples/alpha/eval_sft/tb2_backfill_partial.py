"""TB-2 부분 통과율 백필 — 이미 끝난 harbor job(gpu06 `/opt/harbor/jobs/tb2*`)에서 verifier/ctrf.json 을 세어
`results/<tag>/results_terminal.json` 에 `partial_pass`·`any_pass`·`timeout_rate` 와 보조 키(`*_partial`, `*_anypass`)를 넣는다.

run_terminal_tb2.sh 가 2026-10-01 부터 같은 지표를 추출 단계에서 직접 내므로, 이 도구는 그 전에 측정된 체크포인트
(general iter600·2300·2862, agentic iter200·1200)용이다. job ↔ 결과 디렉토리 대응은 `terminal_raw.json` 의 `id`
= 트라이얼 `result.json` 의 `config.job_id` 로 한다(job 이름 md5 는 RUN_NAME 규약이 바뀌어 믿을 수 없다).

사용: python3 eval_sft/tb2_backfill_partial.py [--dry-run] [--results-dir eval_sft/results]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SSHC = "/home/work/vidsearch/.ssh-keys/config"
REMOTE = r'''
import json, glob, os
out = {}
for J in glob.glob("/opt/harbor/jobs/tb2*"):
    files = glob.glob(J + "/*/result.json")
    if len(files) < 100:
        continue
    jid = None; n_tr = n_to = n_any = n_full = n_noctrf = 0; fr, fr_ok, fr_to = [], [], []
    for f in files:
        try:
            r = json.load(open(f))
        except Exception:
            continue
        jid = jid or (r.get("config") or {}).get("job_id")
        n_tr += 1
        to = ((r.get("exception_info") or {}).get("exception_type") == "AgentTimeoutError"); n_to += to
        try:
            sm = json.load(open(os.path.join(os.path.dirname(f), "verifier", "ctrf.json")))["results"]["summary"]
            p, t = int(sm.get("passed") or 0), int(sm.get("tests") or 0)
        except Exception:
            n_noctrf += 1; p, t = 0, 0
        x = (p / t) if t else 0.0
        fr.append(x); (fr_to if to else fr_ok).append(x)
        n_any += (p > 0); n_full += (t > 0 and p == t)
    mean = lambda v: (sum(v) / len(v)) if v else 0.0
    out[jid] = {"job": os.path.basename(J), "trials_scored": n_tr, "timeout_rate": n_to / n_tr if n_tr else 0.0,
                "partial_pass_mean": mean(fr), "any_pass_rate": n_any / n_tr if n_tr else 0.0, "full_pass_trials": n_full,
                "partial_pass_mean_completed": mean(fr_ok), "n_completed": len(fr_ok),
                "partial_pass_mean_timeout": mean(fr_to), "no_ctrf": n_noctrf}
print(json.dumps(out))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    raw = subprocess.run(["ssh", "-F", SSHC, "-o", "BatchMode=yes", "alpha-eval", "python3", "-"],
                         input=REMOTE, capture_output=True, text=True, timeout=600)
    if raw.returncode != 0:
        print("[backfill] ❌ gpu06 집계 실패:", raw.stderr[-400:]); return 1
    by_job = json.loads(raw.stdout)
    print(f"[backfill] gpu06 job {len(by_job)}개 집계")
    n = 0
    for rawf in sorted(Path(a.results_dir).glob("*/terminal_raw.json")):
        tag = rawf.parent.name
        try:
            jid = json.load(open(rawf)).get("id")
        except Exception:
            continue
        m = by_job.get(jid)
        if not m:
            continue
        resf = rawf.parent / "results_terminal.json"
        if not resf.exists():
            continue
        res = json.load(open(resf))
        key = next((k for k in res.get("results", {}) if k.startswith("terminal_bench_2") and not k.endswith(("_partial", "_anypass"))), None)
        if key is None:
            continue
        main_res = res["results"][key]
        if m["no_ctrf"] > 0.5 * m["trials_scored"]:
            print(f"[backfill] ⚠️ {tag}: ctrf 없는 트라이얼 {m['no_ctrf']}/{m['trials_scored']} — 중단된 job, 건너뜀 ({m['job']})"); continue
        invalid = main_res.get("no_answer,none") == 1.0
        main_res.update({"partial_pass,none": m["partial_pass_mean"], "any_pass,none": m["any_pass_rate"],
                         "timeout_rate,none": m["timeout_rate"]})
        res["results"][key + "_partial"] = {"partial_pass,none": m["partial_pass_mean"]}
        res["results"][key + "_anypass"] = {"any_pass,none": m["any_pass_rate"]}
        if invalid:
            for k in (key + "_partial", key + "_anypass"):
                res["results"][k]["no_answer,none"] = 1.0
        det = res.setdefault("terminal_detail", {})
        det.update({k: v for k, v in m.items() if k != "job"})
        det["partial_pass_backfilled_from"] = m["job"]
        print(f"[backfill] {tag}: job {m['job']} trials {m['trials_scored']} timeout {m['timeout_rate']*100:.1f}% "
              f"partial {m['partial_pass_mean']*100:.2f}% any1 {m['any_pass_rate']*100:.1f}% full {m['full_pass_trials']}"
              + (" (dry-run)" if a.dry_run else ""))
        if not a.dry_run:
            json.dump(res, open(resf, "w"), ensure_ascii=False, indent=2)
        n += 1
    print(f"[backfill] 갱신 {n}개{' (dry-run)' if a.dry_run else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
