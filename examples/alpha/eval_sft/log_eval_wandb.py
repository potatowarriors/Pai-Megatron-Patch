"""SFT 벤치 결과 → wandb (프로젝트 alpha-post-eval). post-train 과 별도 프로젝트.

**전 지표를 그대로 올린다.** 대표 지표 하나를 골라 올리지 않는다 — 어느 지표를 볼지는
사람이 정한다. lm_eval 이 내는 모든 숫자(예: IFEval 의 4지표, RULER 의 구간별 점수,
`exact_match,strict-match` 와 `exact_match,flexible-extract` 같은 필터별 변형)를
`bench/<task>/<metric>` 으로 전부 기록한다.

네임스페이스 규약:
  bench/<task>/<metric>        점수·지표 원값 (비율은 100분율로 환산, 절대값은 그대로)
  bench/<task>/<metric>__err   stderr
  diag/<task>/valid            1=유효 0=무효 (summarize 와 같은 임계). **기록은 막지 않는다** —
                               무효 여부를 플래그로 남기고 판단은 사람이 한다.
  n/<task>                     sample_len

사용: python3 eval_sft/log_eval_wandb.py --results-dir eval_sft/results --run-tag <run>_iter<N>
      [--project alpha-post-eval] [--dry-run] [--all-tags]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_registry import invalid_reasons  # noqa: E402

# 100분율로 바꾸지 않는 절대값 지표 (토큰 수·글자 수·샘플 수 등)
RAW_SUFFIX = ("gen_chars", "samples_k", "sample_len", "n_judged", "avg_tokens")


def parse_tag(tag: str) -> tuple[str, int]:
    m = re.search(r"_iter(\d+)$", tag)
    return (tag[: m.start()] if m else tag, int(m.group(1)) if m else 0)


def _clean(name: str) -> str:
    """wandb 키로 안전한 이름. `metric,filter` → `metric__filter`."""
    return re.sub(r"[^\w.\-/]", "_", name.replace(",", "__"))


def collect(tagdir: Path) -> tuple[dict, list[str]]:
    """(wandb 지표, 태스크별 무효 사유 메모).

    **아무것도 버리지 않는다.** 무효 판정은 `diag/<task>/valid` 플래그로만 남긴다.
    """
    out: dict[str, float] = {}
    notes: list[str] = []
    files = sorted(tagdir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    files += sorted(tagdir.rglob("results.json"), key=lambda p: p.stat().st_mtime)

    # 같은 태스크가 여러 JSON 에 있으면 **가장 최근 실행이 이긴다**. 태스크 단위로 먼저
    # 골라낸 뒤 기록한다 — 파일 단위로 순회하며 덮어쓰면 지표는 신본, 무효 플래그는
    # 구본이 남는 뒤섞임이 생긴다 (2026-09-01: RULER 구본의 no_answer=1.0 이 살아남았다).
    latest: dict[str, dict] = {}
    for jf in files:
        try:
            blob = json.loads(jf.read_text())
        except Exception:  # noqa: BLE001
            continue
        for task, tr in (blob.get("results") or {}).items():
            if isinstance(tr, dict):
                latest[task] = tr

    for task, tr in latest.items():
        bad = invalid_reasons(tr)
        out[f"diag/{task}/valid"] = 0.0 if bad else 1.0
        if bad:
            notes.append(f"{task}: {', '.join(bad)}")
        n = tr.get("sample_len")
        if isinstance(n, (int, float)):
            out[f"n/{task}"] = float(n)
        for k, v in tr.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            if k in ("alias", "sample_len"):
                continue
            base = k.split(",")[0]
            key = _clean(k)
            if "stderr" in base:
                out[f"bench/{task}/{key}"] = float(v)
                continue
            raw = any(base.endswith(sfx) for sfx in RAW_SUFFIX)
            # 비율 지표만 100분율. 절대값(토큰 수 등)은 그대로.
            out[f"bench/{task}/{key}"] = float(v) if raw else float(v) * 100.0
    return out, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--run-tag", help="단일 태그. --all-tags 와 배타")
    ap.add_argument("--all-tags", action="store_true", help="results-dir 의 모든 <run>_iter<N> 을 백필")
    ap.add_argument("--project", default="alpha-post-eval")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rd = Path(a.results_dir)
    if a.all_tags:
        tags = sorted(d.name for d in rd.iterdir() if d.is_dir() and re.search(r"_iter\d+$", d.name))
    elif a.run_tag:
        tags = [a.run_tag]
    else:
        ap.error("--run-tag 또는 --all-tags 가 필요하다")

    if not tags:
        print("[wandb] 대상 태그 없음")
        return 0

    for tag in tags:
        tagdir = rd / tag
        if not tagdir.is_dir():
            print(f"[wandb] 건너뜀 (디렉토리 없음): {tag}")
            continue
        run_name, it = parse_tag(tag)
        metrics, notes = collect(tagdir)
        if not metrics:
            print(f"[wandb] 건너뜀 (지표 없음): {tag}")
            continue

        n_bench = sum(1 for k in metrics if k.startswith("bench/"))
        print(f"[wandb] {run_name} iter={it} — 지표 {n_bench}개 (+진단 {len(metrics) - n_bench})")
        for msg in notes:
            print(f"   ⚠️ 무효 플래그: {msg}")
        if a.dry_run:
            for k in sorted(metrics):
                if k.startswith("bench/"):
                    print(f"      {k} = {metrics[k]:.4g}")
            continue

        import wandb  # noqa: PLC0415

        rid = "eval-" + re.sub(r"[^a-z0-9_-]", "-", run_name.lower())[:56]
        run = wandb.init(project=a.project, name=run_name, id=rid, resume="allow",
                         config={"training_run": run_name}, reinit=True)
        run.log({**metrics, "iteration": it}, step=it)
        run.summary.update({f"latest/{k}": v for k, v in metrics.items()})
        run.summary["latest_iter"] = it
        wandb.finish()
        print(f"[wandb] ✅ {a.project}/{run_name} @ iter {it}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
