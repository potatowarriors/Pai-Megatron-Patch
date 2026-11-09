"""IFBench 체크포인트 비교 보고 (2026-11-09, 적대적 검토 권고 반영) — lm_eval 표본(`--log_samples`)의 생성 원문을 다시 채점한다.

ifbench_aa 의 대표 점수(prompt-level loose)만으로는 체크포인트를 고를 수 없다:
  - 검사기의 빈 참 때문에 단답만으로 통과하는 프롬프트가 많다 → **비자명 부분** (TRIVIAL_ANSWERS 중 어느 것으로도 loose 통과가
    안 되는 프롬프트) 점수를 같이 본다.
  - General 학습 블렌드와 템플릿이 같거나 비슷한 유형이 있다 → **OOD 부분** (SAME_TEMPLATE_TYPES · SIMILAR_TYPES 가 없는 프롬프트).
  - 사고를 닫지 못한 응답은 빈 답으로 채점된다 → **닫힌 응답만의 점수**와 think_closed 를 나란히, 짝 비교에서 Δthink_closed > 2pp 표시.
  - 두 체크포인트 차이는 프롬프트로 짝지은 표준오차로 판단한다 (각자의 lm_eval stderr 를 따로 보면 지나치게 보수적이다):
      a_i, b_i = 프롬프트 i 의 k 회 평균, d_i = a_i − b_i, Δ = mean(d), SE = sqrt(Σ(d_i − Δ)² / (n(n−1)))
      재생성만의 잡음 SE_samp² = Σ[a_i(1−a_i) + b_i(1−b_i)] / ((k−1) n²) — SE ≈ SE_samp 이면 k 를, SE ≫ SE_samp 이면 프롬프트 수가 한계다.
    95% 구간은 Δ ± z·SE (z = 1.96, 비교 m 쌍이면 Bonferroni z), 프롬프트 단위 부트스트랩으로 교차 확인한다.

재채점은 운영 채점과 같은 함수(`tasks/ifbench_utils.score_answer`)를 쓰고, 문서별 재채점 값이 lm_eval 이 기록한 값과 같은지 먼저 확인한다.

사용: python3 eval_sft/ifbench_compare.py <결과 디렉토리> [<결과 디렉토리> ...]   (먼저 준 것이 기준, 이웃한 쌍과 기준 대비 쌍을 비교)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tasks"))
import ifbench_utils as U  # noqa: E402  (import 때 사전 점검이 돈다)
from aa_utils import split_think  # noqa: E402


def load_samples(result_dir: Path) -> dict[str, list[str]]:
    files = sorted(result_dir.rglob("samples_ifbench_aa_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no samples_ifbench_aa_*.jsonl under {result_dir}")
    out, logged = {}, {}
    for line in files[-1].read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        key = str(d["doc"]["key"])
        resp = d["filtered_resps"][0]
        out[key] = [str(x) for x in resp] if isinstance(resp, list) else [str(resp)]
        logged[key] = d["prompt_level_loose_acc"]
    return out, logged, files[-1]


def rescore(resps: dict[str, list[str]], raw: dict[str, dict]) -> dict[str, dict]:
    per = {}
    for key, texts in resps.items():
        row = raw[key]
        rec = {"strict": [], "loose": [], "closed": [], "words": [], "chars": [], "err": 0}
        for t in texts:
            a, c = split_think(t)
            a = a if c else ""
            s, lo, err = U.score_answer(row, a)
            rec["strict"].append(float(all(s)))
            rec["loose"].append(float(all(lo)))
            rec["closed"].append(c)
            rec["err"] += err > 0
            if c:
                rec["words"].append(len(a.split()))
                rec["chars"].append(len(a))
        per[key] = rec
    return per


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summary(per: dict[str, dict], keys: list[str]) -> dict:
    k = len(next(iter(per.values()))["loose"])
    closed_loose = [lo for key in keys for lo, c in zip(per[key]["loose"], per[key]["closed"]) if c]
    words = [w for key in keys for w in per[key]["words"]]
    chars = [x for key in keys for x in per[key]["chars"]]
    n_resp = len(keys) * k
    return {
        "loose": 100 * mean([mean(per[key]["loose"]) for key in keys]),
        "strict": 100 * mean([mean(per[key]["strict"]) for key in keys]),
        "closed_only_loose": 100 * mean(closed_loose),
        "think_closed": 100 * sum(c for key in keys for c in per[key]["closed"]) / n_resp,
        "short_answer": 100 * sum(1 for w in words if w < U.SHORT_ANSWER_WORDS) / n_resp,
        "empty_answer": 100 * sum(1 for w in words if w == 0) / n_resp,
        "checker_error": 100 * sum(per[key]["err"] for key in keys) / n_resp,
        "answer_chars_median": statistics.median(chars) if chars else float("nan"),
        "n_prompts": len(keys),
    }


def paired(pa: dict, pb: dict, keys: list[str], z: float, boot: int = 10000, seed: int = 20261109) -> dict:
    k = len(next(iter(pa.values()))["loose"])
    a = [mean(pa[key]["loose"]) for key in keys]
    b = [mean(pb[key]["loose"]) for key in keys]
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    delta = mean(d)
    se = math.sqrt(sum((x - delta) ** 2 for x in d) / (n * (n - 1)))
    se_samp = math.sqrt(sum(x * (1 - x) + y * (1 - y) for x, y in zip(a, b)) / ((k - 1) * n * n))
    rng = random.Random(seed)
    bs = sorted(mean([d[rng.randrange(n)] for _ in range(n)]) for _ in range(boot))
    lo_q, hi_q = bs[int(0.025 * boot)], bs[int(0.975 * boot) - 1]
    return {"delta": 100 * delta, "se": 100 * se, "ci_lo": 100 * (delta - z * se), "ci_hi": 100 * (delta + z * se),
            "boot_lo": 100 * lo_q, "boot_hi": 100 * hi_q, "se_samp": 100 * se_samp, "n": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dirs", nargs="+")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    raw = U._raw_rows()

    trivial = {key for key, row in raw.items() if any(all(U.score_answer(row, t)[1]) for t in U.TRIVIAL_ANSWERS)}
    overlap = U.SAME_TEMPLATE_TYPES | U.SIMILAR_TYPES
    subsets = {
        "all": sorted(raw),
        "nontrivial": sorted(k for k in raw if k not in trivial),
        "ood": sorted(k for k in raw if not set(raw[k]["instruction_id_list"]) & overlap),
    }
    print(f"subsets: all {len(subsets['all'])} · nontrivial {len(subsets['nontrivial'])} (trivially passable {len(trivial)}) · "
          f"ood {len(subsets['ood'])} (excluded types {sorted(overlap)})")

    names, pers = [], []
    for rd in a.result_dirs:
        rd = Path(rd)
        resps, logged, f = load_samples(rd)
        if set(resps) != set(raw):
            raise SystemExit(f"{rd}: samples cover {len(resps)} keys, expected {len(raw)}")
        per = rescore(resps, raw)
        mism = sum(abs(mean(per[k]["loose"]) - logged[k]) > 1e-9 for k in raw)
        if mism:
            raise SystemExit(f"{rd}: re-score differs from logged per-doc loose on {mism} prompts")
        names.append(rd.name)
        pers.append(per)
        print(f"{rd.name}: {f.name} · re-score = logged on {len(raw)}/{len(raw)} prompts")

    out = {"subsets": {k: len(v) for k, v in subsets.items()}, "checkpoints": {}, "pairs": []}
    cols = ["loose", "strict", "closed_only_loose", "think_closed", "short_answer", "empty_answer", "checker_error", "answer_chars_median"]
    print("\n| checkpoint | subset | n | " + " | ".join(cols) + " |")
    print("|---|---|---:|" + "---:|" * len(cols))
    for name, per in zip(names, pers):
        out["checkpoints"][name] = {}
        for sub, keys in subsets.items():
            s = summary(per, keys)
            out["checkpoints"][name][sub] = s
            print(f"| {name} | {sub} | {s['n_prompts']} | " + " | ".join(f"{s[c]:.1f}" for c in cols) + " |")

    pairs = [(i + 1, i) for i in range(len(names) - 1)] + [(j, 0) for j in range(2, len(names))]
    m = len(pairs) * len(subsets)
    z = 1.96 if m <= 1 else statistics.NormalDist().inv_cdf(1 - 0.025 / m)
    print(f"\npaired loose Δ (later − earlier), z = {z:.2f} (Bonferroni over {m} comparisons)")
    print("| pair | subset | n | Δ | paired SE | CI | bootstrap 95% | SE_samp | Δthink_closed | verdict |")
    print("|---|---|---:|---:|---:|---|---|---:|---:|---|")
    for j, i in pairs:
        for sub, keys in subsets.items():
            p = paired(pers[j], pers[i], keys, z)
            dtc = out["checkpoints"][names[j]][sub]["think_closed"] - out["checkpoints"][names[i]][sub]["think_closed"]
            sig = "구분됨" if p["ci_lo"] > 0 or p["ci_hi"] < 0 else "구분 불가"
            flag = " ⚠" if abs(dtc) > 2 else ""
            print(f"| {names[j]} − {names[i]} | {sub} | {p['n']} | {p['delta']:+.1f} | {p['se']:.2f} | [{p['ci_lo']:+.1f}, {p['ci_hi']:+.1f}] | "
                  f"[{p['boot_lo']:+.1f}, {p['boot_hi']:+.1f}] | {p['se_samp']:.2f} | {dtc:+.1f}{flag} | {sig} |")
            out["pairs"].append({"a": names[j], "b": names[i], "subset": sub, **p, "d_think_closed": dtc, "verdict": sig})
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
