"""512K 그리드 요약 — outputs/lc512k_eval/<model>_<profile>/results_*.json → SUMMARY.md.

RULER 4태스크 × 4구간(131072/258048/393216/520192)을 셀(모델×프로파일)별로 표로 만들고,
SFT 셀의 단문 회귀 표본(ifeval prompt-strict, gpqa exact_match)을 덧붙인다.
판정 규칙은 `study/lc_512k_eval.md` §4. 수치 산출만 하고 판정은 사람이 한다.

사용: python3 scripts/lc512k_summarize.py outputs/lc512k_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval_sft"))
from bench_registry import get  # noqa: E402

LENGTHS = ("131072", "258048", "393216", "520192")
RULER = {
    "ruler_niah_single_1_512k": "single_1",
    "ruler_niah_single_2_512k": "single_2",
    "ruler_niah_multikey_1_512k": "multikey_1",
    "ruler_niah_multivalue_512k": "multivalue",
}
SHORT = {"ifeval_aa": "prompt_level_strict_acc", "gpqa_diamond_aa": "exact_match"}
ORDER = ["sft_yarn2", "sft_ext", "sft_yarn4", "base_yarn2", "base_ext", "base_yarn4"]


def load_cell(cell: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(cell.rglob("results_*.json"), key=lambda p: p.stat().st_mtime):
        try:
            res = json.loads(f.read_text()).get("results", {})
        except Exception:  # noqa: BLE001
            continue
        out.update(res)
    return out


def pct(v: float | None) -> str:
    return "—" if v is None or v < 0 else f"{100 * v:.0f}"


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/lc512k_eval")
    cells = {d.name: load_cell(d) for d in root.iterdir() if d.is_dir() and d.name in ORDER}
    names = [n for n in ORDER if n in cells] + sorted(n for n in cells if n not in ORDER)
    if not names:
        print("결과 없음:", root)
        return 1

    lines = [f"# 512K 그리드 요약 (`{root}`)\n",
             "행 = 모델×프로파일 (yarn2: s=2/orig 262144 · ext: 순수 외삽 · yarn4: s=4/orig 131072). "
             "값 = 정확도 %, n=20/구간. `—` = 미측정. `no_ans` = 추출 실패율(>10% 면 측정 무효).\n"]
    for task, short in RULER.items():
        lines.append(f"## RULER {short}\n")
        lines.append("| cell | " + " | ".join(LENGTHS) + " | no_ans |")
        lines.append("|---|" + "---:|" * (len(LENGTHS) + 1))
        for n in names:
            tr = cells[n].get(task)
            if not tr:
                lines.append(f"| {n} | " + " | ".join("—" for _ in LENGTHS) + " | — |")
                continue
            na = get(tr, "no_answer")
            lines.append(f"| {n} | " + " | ".join(pct(get(tr, L)) for L in LENGTHS)
                         + f" | {pct(na)}{'%' if na is not None else ''} |")
        lines.append("")

    lines.append("## 단문 회귀 표본 (SFT 셀, ifeval·gpqa 각 100문항 — 프로파일 fleet 그대로)\n")
    lines.append("| cell | ifeval prompt-strict | gpqa exact_match |")
    lines.append("|---|---:|---:|")
    for n in names:
        vals = []
        for task, metric in SHORT.items():
            tr = cells[n].get(task)
            vals.append(pct(get(tr, metric)) if tr else "—")
        lines.append(f"| {n} | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("판정 규칙: `study/lc_512k_eval.md` §4 (520192 single_1 ≥ 90 · 어려운 태스크 128K 대비 −10pp 이내 · "
                 "393216 회복 · 단문 표본 Δ 노이즈 이내). 이 표는 수치만 낸다.")
    text = "\n".join(lines) + "\n"
    (root / "SUMMARY.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
