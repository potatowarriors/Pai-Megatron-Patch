"""SFT 벤치 결과 → wandb (프로젝트 alpha-post-eval). post-train 과 별도 프로젝트.

results/<run_tag>/ 의 lm_eval(T1)·simpleqa·logickor(T3) JSON 을 읽어, 학습 run 별
wandb run 에 iter 를 step 으로 로깅한다. 체크포인트마다 resume 하여 곡선이 누적된다.
같은 프로젝트에 baseline_lcb 와 sft_128k_full_* 가 각각 run 으로 올라가 오버레이 비교된다.

사용: python3 eval_sft/log_eval_wandb.py --results-dir eval_sft/results --run-tag <run>_iter<N> \
        [--project alpha-post-eval]
"""
from __future__ import annotations
import argparse, json, re, os, hashlib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_registry import HEADLINE, DIAGNOSTIC, get, invalid_reasons  # noqa: E402

def parse_tag(tag: str):
    m = re.search(r"_iter(\d+)$", tag)
    return (tag[:m.start()] if m else tag, int(m.group(1)) if m else 0)

def collect(tagdir: Path) -> tuple[dict, list[str]]:
    """(로깅할 지표, 무효 판정된 태스크 목록).

    **무효 셀은 올리지 않는다.** 측정이 성립하지 않은 값을 wandb 곡선에 넣으면
    나중에 되돌릴 수 없는 잘못된 이력이 된다 (2026-08-30 사고 교훈 — 무효 수치가
    클라우드에 올라가 자격증명 부재로 삭제하지 못했다).
    진단 지표(no_answer 등)는 `diag/` 네임스페이스로 함께 올려 원인을 남긴다.
    """
    out, invalid = {}, []
    for jf in sorted(tagdir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime):
        try:
            res = json.loads(jf.read_text()).get("results", {})
        except Exception:
            continue
        for task, tr in res.items():
            spec = HEADLINE.get(task)
            if not spec:
                continue
            metric, wkey = spec
            bad = invalid_reasons(tr)
            if bad:
                invalid.append(f"{task}: {', '.join(bad)}")
                continue
            v = get(tr, metric)
            if v is not None:
                out[f"bench/{wkey}"] = v
            for d in DIAGNOSTIC:
                dv = get(tr, d)
                if dv is not None:
                    out[f"diag/{wkey}_{d}"] = dv
    return out, invalid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--project", default="alpha-post-eval")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tagdir = Path(a.results_dir) / a.run_tag
    if not tagdir.is_dir():
        raise SystemExit(f"[wandb] 결과 디렉토리 없음: {tagdir}")
    run_name, it = parse_tag(a.run_tag)
    metrics, invalid = collect(tagdir)
    for msg in invalid:
        print(f"[wandb] ⛔ 무효 — 업로드 제외: {msg}")
    if not metrics:
        raise SystemExit(f"[wandb] {tagdir} 에 로깅할 지표 없음")
    # 비율 지표만 100분율. gen_chars/samples_k 같은 절대값은 그대로 둔다.
    RAW = ("gen_chars", "samples_k")
    metrics_pct = {k: (v if any(k.endswith(r) for r in RAW) else (v * 100.0 if v <= 1.0 else v))
                   for k, v in metrics.items()}
    print(f"[wandb] run={run_name} iter={it} project={a.project}")
    for k, v in sorted(metrics_pct.items()): print(f"   {k} = {v:.2f}")
    if a.dry_run:
        print("[wandb] dry-run — 업로드 안 함"); return
    import wandb
    # 학습 run 당 결정적 id (resume 로 곡선 누적). wandb id 는 [a-z0-9_-], <=64.
    rid = "eval-" + re.sub(r"[^a-z0-9_-]", "-", run_name.lower())[:56]
    run = wandb.init(project=a.project, name=run_name, id=rid, resume="allow",
                     config={"training_run": run_name}, reinit=True)
    run.log({**metrics_pct, "iteration": it}, step=it)
    run.summary.update({f"latest/{k.split('/')[-1]}": v for k, v in metrics_pct.items()})
    run.summary["latest_iter"] = it
    wandb.finish()
    print(f"[wandb] ✅ 로깅 완료: {a.project}/{run_name} @ iter {it}")

if __name__ == "__main__":
    main()
