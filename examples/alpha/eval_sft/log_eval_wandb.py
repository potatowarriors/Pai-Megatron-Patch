"""SFT 벤치 결과 → wandb (프로젝트 alpha-post-eval). post-train 과 별도 프로젝트.

**키 규약은 `<task>/<metric>`** — 기존 alpha-evals 프로젝트와 같다. wandb 는 첫 `/` 앞을
패널 섹션으로 잡으므로 이렇게 해야 **벤치마크별로 묶인다**. `bench/` 같은 공통 접두사를
붙이면 전부 한 섹션에 뭉쳐 읽을 수 없다 (2026-09-01 수정).

lm_eval 의 필터 접미사는 **원본 그대로** 둔다:
    gsm8k/exact_match,flexible-extract
    gsm8k/exact_match_stderr,strict-match

**평가 결과만 올린다.** 실패율·진단 지표(`no_answer`, `think_closed`, `judge_fail`,
`empty`, `gen_chars`, `samples_k`)는 올리지 않는다 — 측정 성립 여부는 `summarize.py` 와
결과 JSON 에서 본다. wandb 는 점수를 보는 곳이다.

사용: python3 eval_sft/log_eval_wandb.py --results-dir eval_sft/results --run-tag <run>_iter<N>
      [--project alpha-post-eval] [--dry-run] [--all-tags]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# 진단 지표 — wandb 에 올리지 않는다. 평가 결과가 아니라 "측정이 성립했나" 를 보는 값이다.
# 판정은 summarize.py 와 결과 JSON 에서 한다.
DIAGNOSTIC_PREFIX = (
    "no_answer", "think_closed", "judge_fail", "empty",
    "gen_chars", "samples_k", "n_judged", "avg_tokens",
)


# 학습 런 디렉토리가 갈려도 **계보가 이어지면 한 곡선**으로 본다.
# 2026-09-01: iter900 에서 데이터 블렌드를 교체하고 새 출력 디렉토리로 재개했다
# (`…_full_20260828_081911` → `…_full_swap_20260901_101523`). 재개 게이트에서 LR 1.901e-5 가
# cosine 연속값과 일치하고 loss 도 이어졌으며, 블렌드 가중치는 비트 동일 · 경로 2개
# (opencode_fixed, identity_v2)만 치환됐다 — 같은 모델의 연속이다. 런 이름이 다르다는 이유로
# 곡선이 iter900 에서 끊기면 추이를 볼 수 없으므로 정본 이름으로 합친다.
# 교체 지점은 run config 의 `blend_swap_at_iter` 로 남긴다 — 합쳤다는 사실을 숨기지 않는다.
RUN_ALIASES = {
    "alpha_baseline_48L_sft_128k_full_swap_20260901_101523":
        "alpha_baseline_48L_sft_128k_full_20260828_081911",
}
BLEND_SWAP_AT_ITER = 900


def parse_tag(tag: str) -> tuple[str, int]:
    m = re.search(r"_iter(\d+)$", tag)
    return (tag[: m.start()] if m else tag, int(m.group(1)) if m else 0)


def _clean(name: str) -> str:
    """wandb 키로 안전한 이름. `metric,filter` → `metric__filter`."""
    return re.sub(r"[^\w.\-/]", "_", name.replace(",", "__"))


def collect(tagdir: Path) -> dict:
    """`<task>/<metric>` 형태의 평가 결과 지표.

    같은 태스크가 여러 결과 JSON 에 있으면 **가장 최근 실행이 이긴다**.
    """
    latest: dict[str, dict] = {}
    files = sorted(tagdir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    files += sorted(tagdir.rglob("results.json"), key=lambda p: p.stat().st_mtime)
    for jf in files:
        try:
            blob = json.loads(jf.read_text())
        except Exception:  # noqa: BLE001
            continue
        for task, tr in (blob.get("results") or {}).items():
            if isinstance(tr, dict):
                latest[task] = tr

    out: dict[str, float] = {}
    for task, tr in latest.items():
        for k, v in tr.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            base = k.split(",")[0]
            if base in ("alias", "sample_len"):
                continue
            if any(base.startswith(d) for d in DIAGNOSTIC_PREFIX):
                continue
            # 필터 접미사(`,strict-match` 등)는 원본 그대로. lm_eval 관행이다.
            out[f"{task}/{k}"] = float(v) * 100.0
    return out


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
        src_run, it = parse_tag(tag)
        run_name = RUN_ALIASES.get(src_run, src_run)
        metrics = collect(tagdir)
        if not metrics:
            print(f"[wandb] 건너뜀 (지표 없음): {tag}")
            continue

        tasks = sorted({k.split("/")[0] for k in metrics})
        merged = f"  (← {src_run})" if src_run != run_name else ""
        print(f"[wandb] {run_name} iter={it} — 태스크 {len(tasks)}개, 지표 {len(metrics)}개{merged}")
        if a.dry_run:
            for k in sorted(metrics):
                print(f"      {k} = {metrics[k]:.4g}")
            continue

        import wandb  # noqa: PLC0415

        # run id 에 키 규약 버전을 넣는다. wandb 는 기록된 키를 지울 수 없으므로,
        # 규약이 바뀌면 **새 run 으로 시작**해야 구 키가 섞이지 않는다
        # (2026-09-01: 한 run 에 bench/<task>, bench/<task>/<metric>, diag/ 3세대가 누적됐다).
        rid = "eval-v3-" + re.sub(r"[^a-z0-9_-]", "-", run_name.lower())[:48]
        cfg = {"training_run": run_name, "key_schema": "task/metric"}
        if src_run != run_name:
            # 어느 학습 런이 이 점을 만들었는지 config 에 남긴다.
            cfg["source_run"] = src_run
            cfg["blend_swap_at_iter"] = BLEND_SWAP_AT_ITER
        run = wandb.init(project=a.project, name=run_name, id=rid, resume="allow",
                         config=cfg, reinit=True)
        run.log({**metrics, "iteration": it}, step=it)
        # summary 에 `latest/` 사본을 만들지 않는다 — 같은 값이 두 벌 생겨
        # 패널 목록이 두 배로 늘고 섹션이 어지러워진다. 최신값은 wandb 가 자동으로
        # summary 에 넣는다 (run.log 의 마지막 값).
        run.summary["latest_iter"] = it
        wandb.finish()
        print(f"[wandb] ✅ {a.project}/{run_name} @ iter {it}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
