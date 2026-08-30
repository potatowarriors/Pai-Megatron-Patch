"""벤치 결과 집계 — results/<run>_iter<N>/ 아래 JSON 들을 iter별 추이표(TRACKING.md)로.

태스크↔지표 매핑은 `bench_registry.py` 정본을 쓴다.

**무효 판정을 통과한 결과만 기록한다.** 추출 실패율(`no_answer`)이 높거나 사고 마감률
(`think_closed`)이 낮으면 측정이 성립하지 않은 것이고, 그런 값을 표에 넣으면 "모델이
약하다"로 읽히는 측정 실패가 이력에 남는다 — 2026-08-30 사고가 정확히 그랬다.
무효 셀은 `무효`로 표기하고 점수를 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_registry import (  # noqa: E402
    TASK_ORDER, display_name, headline, invalid_reasons,
)


def parse_tag(tag: str) -> tuple[str, int]:
    m = re.search(r"_iter(\d+)$", tag)
    return (tag[: m.start()] if m else tag, int(m.group(1)) if m else -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows: dict[tuple[str, int], dict[str, tuple[float | None, list[str]]]] = {}
    seen: set[str] = set()

    for tagdir in sorted(Path(a.results_dir).glob("*")):
        if not tagdir.is_dir():
            continue
        jsons = list(tagdir.rglob("results_*.json")) + list(tagdir.rglob("results.json"))
        if not jsons:
            continue
        run, it = parse_tag(tagdir.name)
        cell = rows.setdefault((run, it), {})
        # mtime 오름차순 병합 — 나중 실행이 같은 태스크를 덮어쓴다
        for jf in sorted(jsons, key=lambda p: p.stat().st_mtime):
            try:
                res = json.loads(jf.read_text()).get("results", {})
            except Exception:  # noqa: BLE001
                continue
            for task, tr in res.items():
                if task not in TASK_ORDER:
                    continue  # 하위 카테고리(mmlu_pro_biology 등) 제외
                cell[task] = (headline(task, tr), invalid_reasons(tr))
                seen.add(task)

    tasks = [t for t in TASK_ORDER if t in seen]
    if not tasks:
        Path(a.out).write_text(
            "# 벤치 추이 (eval_ckpt 집계)\n\n**유효 수치 없음.**\n"
            "게이트(`check_gates.py`)와 판정(`summarize.py`)을 통과한 결과만 여기 들어온다.\n"
        )
        print("[aggregate] 기록할 태스크 없음")
        return 0

    hdr = [display_name(t) for t in tasks]
    lines = [
        "# 벤치 추이 (eval_ckpt 집계)\n",
        "각 체크포인트별 대표 점수(100분율). 매핑 정본은 `bench_registry.py`.\n",
        "`무효` = 추출 실패율/사고 마감률이 임계를 벗어나 측정이 성립하지 않은 셀 "
        "(판정: `summarize.py`).\n",
        "| run | iter | " + " | ".join(hdr) + " |",
        "|---|---|" + "|".join("---" for _ in hdr) + "|",
    ]
    for run, it in sorted(rows, key=lambda x: (x[0], x[1])):
        cell = rows[(run, it)]
        vals = []
        for t in tasks:
            got = cell.get(t)
            if got is None:
                vals.append("—")
                continue
            score, bad = got
            if bad:
                vals.append("무효")
            elif isinstance(score, float):
                vals.append(f"{score * 100:.1f}")
            else:
                vals.append("—")
        lines.append(f"| {run} | {it} | " + " | ".join(vals) + " |")

    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"[aggregate] {len(rows)} 체크포인트 × {len(tasks)} 태스크 → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
