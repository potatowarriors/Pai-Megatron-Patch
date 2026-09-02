"""RULER 능력 스위트 요약·판정 — outputs/ruler_cap_eval/<cell>/results_*.json → SUMMARY.md.

판정 규약 (study/lc_512k_eval.md §4): RULER 논문의 유효 컨텍스트 정의를 따른다 —
**12태스크 구간 평균 ≥ 85 인 최대 길이 = "지원" 길이**. 12태스크가 전부 유효할 때만
그 구간의 평균을 공식 수치로 삼는다(누락·무효는 표기).

사용: python3 scripts/ruler_cap_summarize.py outputs/ruler_cap_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval_sft"))
from bench_registry import get  # noqa: E402

LENGTHS = ("131072", "258048", "393216", "520192")
TASKS = [f"ruler_cap_{n}" for n in (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue", "vt", "cwe", "fwe", "qa_hotpot",
)]
THRESHOLD = 85.0


def load_cell(cell: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(cell.rglob("results_*.json"), key=lambda p: p.stat().st_mtime):
        try:
            out.update(json.loads(f.read_text()).get("results", {}))
        except Exception:  # noqa: BLE001
            continue
    return out


def pct(v: float | None) -> str:
    return "—" if v is None or v < 0 else f"{100 * v:.1f}"


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/ruler_cap_eval")
    cells = {d.name: load_cell(d) for d in sorted(root.iterdir())
             if d.is_dir() and any(d.rglob("results_*.json"))}
    if not cells:
        print("결과 없음:", root)
        return 1

    lines = [f"# RULER 능력 스위트 요약 (`{root}`)\n",
             "RULER-13 중 12태스크(qa_squad 제외 — SQuAD 풀 ~0.27M tok 로 393K+ 미충족), n=50/구간, "
             "Reasoning-Off. **판정 = 구간 평균 ≥85 (RULER 논문 유효 컨텍스트 규약)**.\n"]
    for name, res in cells.items():
        lines.append(f"## {name}\n")
        lines.append("| task | " + " | ".join(LENGTHS) + " | no_ans |")
        lines.append("|---|" + "---:|" * (len(LENGTHS) + 1))
        sums = {L: [] for L in LENGTHS}
        missing = []
        for t in TASKS:
            tr = res.get(t)
            if not tr:
                missing.append(t)
                lines.append(f"| {t.removeprefix('ruler_cap_')} | " + " | ".join("—" for _ in LENGTHS) + " | — |")
                continue
            row = []
            for L in LENGTHS:
                v = get(tr, L)
                row.append(pct(v))
                if v is not None and v >= 0:
                    sums[L].append(v)
            na = get(tr, "no_answer")
            lines.append(f"| {t.removeprefix('ruler_cap_')} | " + " | ".join(row)
                         + f" | {pct(na)}{'%' if na is not None else ''} |")
        avg_row, verdict = [], []
        for L in LENGTHS:
            if len(sums[L]) == len(TASKS):
                a = 100 * sum(sums[L]) / len(sums[L])
                avg_row.append(f"**{a:.1f}**")
                verdict.append(f"{int(L):,}: {'✅ 지원' if a >= THRESHOLD else '❌ 미달'} ({a:.1f})")
            else:
                avg_row.append(f"({len(sums[L])}/{len(TASKS)})")
                verdict.append(f"{int(L):,}: ⚠️ 불완전 ({len(sums[L])}/{len(TASKS)} 태스크)")
        lines.append("| **평균 (12/12일 때만)** | " + " | ".join(avg_row) + " | |")
        lines.append("")
        lines.append("판정: " + " · ".join(verdict))
        if missing:
            lines.append(f"누락 태스크: {', '.join(m.removeprefix('ruler_cap_') for m in missing)}")
        lines.append("")
    text = "\n".join(lines) + "\n"
    (root / "SUMMARY.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
