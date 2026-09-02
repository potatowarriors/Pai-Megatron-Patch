"""RULER 능력 스위트 사전 빌드 검증 — 본 실행(ruler_cap_run.sh) 전 게이트.

5개 생성 경로(essay-niah / needle-niah / vt / cwe / fwe / qa_hotpot)를 실제 512k 구간
집합에서 n=1 로 빌드해 ① seq_set 주입 ② max_length 라벨 ③ length ≤ max_length
④ fill(실길이/라벨) 을 확인한다. 520K 하이스택 토크나이즈 때문에 수십 분 걸린다 —
`ruler_cap_run.sh` 앞에 체인으로 실행 (`&&`). 실패 시 본 실행이 시작되지 않는다.

사용: python3 scripts/ruler_cap_validate.py   (exit 0 = PASS)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ALPHA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ALPHA / "eval_sft" / "tasks"))

SEQS = [131072, 258048, 393216, 520192]
TOK = str(ALPHA / "tokenizer_v5")
# 카테고리 대표: essay-niah(single_3=uuid), needle-niah(multikey_3), vt, fwe, qa_hotpot
# (cwe 는 2026-09-02 이 게이트가 ~130K 포화를 검출해 스위트에서 제외 — cap_utils docstring)
TARGETS = ("niah_single_3", "niah_multikey_3", "vt", "fwe", "qa_hotpot")
# fill 하한 — 라벨 대비 실길이. qa/vt 는 문서·noise 단위 granularity 로 여유를 둔다.
MIN_FILL = 0.85


def main() -> int:
    import ruler_cap_utils as C
    failures = []
    for name in TARGETS:
        t0 = time.time()
        try:
            ds = getattr(C, name)(max_seq_lengths=list(SEQS), num_samples=1, tokenizer=TOK)["test"]
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: 빌드 예외 {type(e).__name__}: {e}")
            print(f"[validate] {name}: ❌ {type(e).__name__}: {e}", flush=True)
            continue
        rows = sorted(ds, key=lambda r: r["max_length"])
        Ls = [r["max_length"] for r in rows]
        if Ls != SEQS:
            failures.append(f"{name}: max_length 집합 {Ls} ≠ {SEQS}")
        for r in rows:
            if r.get("seq_set") != SEQS:
                failures.append(f"{name}@{r['max_length']}: seq_set 미주입")
            if r["length"] > r["max_length"]:
                failures.append(f"{name}@{r['max_length']}: length {r['length']} 초과")
        fills = [round(r["length"] / r["max_length"], 3) for r in rows]
        low = [f"{L}:{f}" for L, f in zip(Ls, fills) if f < MIN_FILL]
        if low:
            failures.append(f"{name}: fill 부족 {low} (< {MIN_FILL} — 라벨만 긴 측정 위험)")
        print(f"[validate] {name}: {time.time()-t0:.0f}s, fill={fills}"
              + (" ❌" if low else " ✅"), flush=True)
    if failures:
        print("[validate] FAIL:", *[f"  - {f}" for f in failures], sep="\n", flush=True)
        return 1
    print("[validate] ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
