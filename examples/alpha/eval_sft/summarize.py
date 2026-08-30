"""T1 결과 판정 — 점수와 함께 **측정이 성립했는지**를 본다.

`exact_match` 만 보면 측정 실패가 오답으로 위장된다. 2026-08-30 사고에서 MMLU-Pro
36.9 는 추출 실패 34% 를 오답으로 흡수한 값이었고, AIME 0.0 은 30/30 이 답에 도달하기
전에 잘린 결과였다. 둘 다 "모델이 약하다"로 읽혔지만 측정 자체가 성립하지 않았다.

NeMo-Skills 가 `no_answer` 를 점수 옆 열로 함께 내는 이유가 이것이다. 여기서도 같은
규약을 쓰고, 임계를 넘으면 **결과를 무효로 판정한다**.

사용:
    python3 summarize.py <RESULT_DIR> [--strict]
      --strict : 무효 판정이 하나라도 있으면 exit 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 무효 판정 임계. 프론티어는 부분 표본·고실패 결과를 발표하지 않는다
# (NVIDIA 재현 문서: "Never report sub-sampled / limited runs").
NO_ANSWER_MAX = 0.10   # 추출 실패 10% 초과 → 하니스/서빙 결함 의심
THINK_CLOSED_MIN = 0.50  # 사고 마감 50% 미만 → 예산 부족 또는 종료 결함

# 태스크별 대표 점수 (표 첫 열)
HEADLINE = {
    "mmlu_pro_aa": "exact_match",
    "gpqa_diamond_aa": "exact_match",
    "aime25_aa": "exact_match",
    "hmmt_feb_2025_aa": "exact_match",
    "ifeval_aa": "inst_level_loose_acc",
}


def _get(res: dict, name: str) -> float | None:
    """`metric,filter` 형태 키에서 metric 이름으로 값을 찾는다."""
    for k, v in res.items():
        if not isinstance(v, (int, float)) or "stderr" in k:
            continue
        if k == name or k.split(",")[0] == name:
            return float(v)
    return None


def load(result_dir: Path) -> dict[str, dict]:
    """같은 디렉토리의 결과 JSON 을 mtime 순으로 병합 (나중 실행이 이김)."""
    out: dict[str, dict] = {}
    files = sorted(result_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    for f in files:
        try:
            res = json.loads(f.read_text()).get("results", {})
        except Exception:  # noqa: BLE001
            continue
        for task, tr in res.items():
            out[task] = tr
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", type=Path)
    ap.add_argument("--strict", action="store_true", help="무효 판정 시 exit 1")
    a = ap.parse_args()

    results = load(a.result_dir)
    if not results:
        print(f"결과 JSON 없음: {a.result_dir}")
        return 1

    rows = []
    invalid = []
    for task in sorted(results):
        tr = results[task]
        head = HEADLINE.get(task)
        score = _get(tr, head) if head else None
        if score is None:  # 미지 태스크 — 첫 숫자 메트릭
            score = next(
                (v for k, v in tr.items()
                 if isinstance(v, (int, float)) and "stderr" not in k and k != "alias"),
                None,
            )
        na = _get(tr, "no_answer")
        tc = _get(tr, "think_closed")
        k = _get(tr, "samples_k")
        n = tr.get("sample_len")

        flags = []
        if na is not None and na > NO_ANSWER_MAX:
            flags.append(f"추출실패 {na * 100:.0f}%>{NO_ANSWER_MAX * 100:.0f}%")
        if tc is not None and tc < THINK_CLOSED_MIN:
            flags.append(f"사고마감 {tc * 100:.0f}%<{THINK_CLOSED_MIN * 100:.0f}%")
        if flags:
            invalid.append((task, flags))

        rows.append({
            "task": task,
            "score": f"{score * 100:.1f}" if isinstance(score, float) else "—",
            "metric": head or "?",
            "n": str(n) if n is not None else "—",
            "k": f"{k:.0f}" if k else "—",
            "no_answer": f"{na * 100:.1f}%" if na is not None else "—",
            "think_closed": f"{tc * 100:.1f}%" if tc is not None else "—",
            "gen_chars": f"{_get(tr, 'gen_chars'):,.0f}" if _get(tr, "gen_chars") else "—",
            "verdict": "무효" if flags else "유효",
        })

    cols = ["task", "score", "metric", "n", "k", "no_answer", "think_closed", "gen_chars", "verdict"]
    width = {c: max(len(c), *(len(r[c]) for r in rows)) for c in cols}
    print("  ".join(c.ljust(width[c]) for c in cols))
    print("  ".join("-" * width[c] for c in cols))
    for r in rows:
        print("  ".join(r[c].ljust(width[c]) for c in cols))

    print()
    if invalid:
        print("❌ 무효 판정 — TRACKING.md 에 기록하지 말 것:")
        for task, flags in invalid:
            print(f"   {task}: {', '.join(flags)}")
        print("   추출실패가 높으면 하니스(프롬프트·정규식), 사고마감이 낮으면 "
              "생성 예산 또는 모델 성숙도를 의심한다.")
        return 1 if a.strict else 0

    print("✅ 전 태스크 유효 — 기록 가능")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
