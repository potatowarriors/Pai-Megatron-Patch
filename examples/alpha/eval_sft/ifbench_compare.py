"""IFBench 체크포인트 비교 보고 (2026-11-09, 적대적 검토 1·2차 권고 반영) — lm_eval 표본(`--log_samples`)의 생성 원문을 다시 채점한다.

**주 비교는 미리 고정한다**: 전체 294 프롬프트 · prompt-level loose · 체크포인트 쌍마다 프롬프트로 짝지은 차.
  a_i, b_i = 프롬프트 i 의 k 회 평균, d_i = a_i − b_i, Δ = mean(d), SE = sqrt(Σ(d_i − Δ)² / (n(n−1)))
  구간은 Δ ± z·SE, z 는 **쌍 수 m 에만** 건 Bonferroni (m=3 이면 2.39). 부트스트랩도 같은 수준(분위 0.025/m)으로 교차 확인하고,
  두 구간이 모두 0 을 빼야 "구분됨" 이다. 재생성만의 잡음 SE_samp² = Σ[a_i(1−a_i) + b_i(1−b_i)] / ((k−1) n²) 를 함께 낸다
  (SE ≈ SE_samp 이면 k 를, SE ≫ SE_samp 이면 프롬프트 수가 한계다).

**민감도 분석** (판정 없음 — Δ · SE · 주 비교와 부호 일치만):
  - nontrivial: TRIVIAL_ANSWERS 중 어느 것으로도 loose 통과가 안 되는 프롬프트 (검사기 빈 참 제거)
  - ood: General 학습 블렌드와 템플릿이 같거나 비슷한 8종이 없는 프롬프트, ood∩nontrivial
  - both_closed: 두 체크포인트 모두 k 개 응답의 사고를 다 닫은 프롬프트 (사고 마감 차이의 선택 편향 제거)

**유효성**: 체크포인트마다 결과 JSON 의 `bench_registry.invalid_reasons` 와, 5단어 이상 답의 검사기 예외 비율 > 1% (실행 중 검사기 고장)를
본다. 무효 체크포인트가 낀 쌍은 판정이 "무효" 다. 퇴화 표시(판정 아님): 프롬프트 사이 동일 답 비율 · 고유 단어 비율 · nontrivial/all 점수비.

재채점은 운영 채점과 같은 함수(`tasks/ifbench_utils.score_answer`)를 쓰고, 문서별 재채점 값이 lm_eval 이 기록한 값과 같은지 먼저 확인한다.

사용: python3 eval_sft/ifbench_compare.py <결과 디렉토리> [<결과 디렉토리> ...] [--json-out 파일]
      (먼저 준 것이 기준. 쌍 = 이웃한 쌍 + 기준 대비 쌍)
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tasks"))
import ifbench_utils as U  # noqa: E402  (import 때 사전 점검이 돈다)
from aa_utils import split_think  # noqa: E402
from bench_registry import invalid_reasons  # noqa: E402

LONG_ANSWER_CHECKER_ERROR_MAX = 0.01  # 적대적 검토 2차: 정상 출력(5단어 이상)의 검사기 예외는 0 — 실행 중 검사기 고장 신호
BOOT = 20000


def load_run(result_dir: Path):
    files = sorted(result_dir.rglob("samples_ifbench_aa_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"no samples_ifbench_aa_*.jsonl under {result_dir}")
    resps, logged = {}, {}
    for line in files[-1].read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        key = str(d["doc"]["key"])
        r = d["filtered_resps"][0]
        resps[key] = [str(x) for x in r] if isinstance(r, list) else [str(r)]
        logged[key] = d["prompt_level_loose_acc"]
    # 같은 런의 결과 JSON 을 파일 이름의 타임스탬프로 짝짓는다 (mtime 으로 따로 고르면 끊긴 런의 파일이 섞일 수 있다)
    ts = files[-1].name[len("samples_ifbench_aa_"):-len(".jsonl")]
    jf = files[-1].with_name(f"results_{ts}.json")
    if not jf.exists():
        raise SystemExit(f"{files[-1]}: matching {jf.name} not found")
    res = json.loads(jf.read_text()).get("results", {})
    if "ifbench_aa" not in res:
        raise SystemExit(f"{jf}: no ifbench_aa result")
    return resps, logged, res["ifbench_aa"], files[-1]


def run_name(result_dir: Path) -> str:
    """lm_eval 은 results/<RUN_TAG>/alpha/ 에 쓴다 — `.../alpha` 를 받으면 run tag 를 이름으로 쓴다."""
    result_dir = result_dir.resolve()
    return result_dir.parent.name if result_dir.name == "alpha" else result_dir.name


def nondeterministic_keys(raw: dict[str, dict]) -> set[str]:
    """공식 검사기가 무작위 값으로 지시를 만드는 행 — 같은 kwargs 로 시드만 바꿔 build_description 이 달라지면 채점이 실행마다 달라진다.

    알려진 사례: key 286 `repeat:repeat_span` 은 n_start=0 을 `if not n_start` 가 없음으로 보아 무작위 시작 위치를 쓴다 (ifbench_utils 머리말).
    이 행은 lm_eval 기록값과 재채점 값이 다를 수 있어 일치 검사에서 허용하고 수를 보고한다. 나머지 행은 정확히 같아야 한다.
    """
    reg = U._ifbench().instructions_registry
    state = random.getstate()
    out = set()
    for key, row in raw.items():
        for index, iid in enumerate(row["instruction_id_list"]):
            kw = {k: v for k, v in row["kwargs"][index].items() if v is not None}
            descs = set()
            for seed in range(4):
                random.seed(seed)
                descs.add(str(reg.INSTRUCTION_DICT[iid](iid).build_description(**kw)))
            if len(descs) > 1:
                out.add(key)
    random.setstate(state)
    return out


def rescore(resps: dict[str, list[str]], raw: dict[str, dict]) -> dict[str, dict]:
    per = {}
    for key, texts in resps.items():
        row = raw[key]
        rec = {"strict": [], "loose": [], "closed": [], "answers": [], "long_err": 0, "long_n": 0}
        for t in texts:
            a, c = split_think(t)
            a = a if c else ""
            s, lo, err = U.score_answer(row, a)
            rec["strict"].append(float(all(s)))
            rec["loose"].append(float(all(lo)))
            rec["closed"].append(c)
            if c:
                rec["answers"].append(a)
                if len(a.split()) >= U.SHORT_ANSWER_WORDS:
                    rec["long_n"] += 1
                    rec["long_err"] += err > 0
        per[key] = rec
    return per


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def score(per, keys, metric="loose"):
    return 100 * mean([mean(per[k][metric]) for k in keys])


def checkpoint_row(per, keys_all, keys_nontrivial, k):
    answers = [a for key in keys_all for a in per[key]["answers"]]
    n_resp = len(keys_all) * k
    long_n = sum(per[key]["long_n"] for key in keys_all)
    long_err = sum(per[key]["long_err"] for key in keys_all)
    norm = [" ".join(a.lower().split()) for a in answers if a.strip()]
    top = collections.Counter(norm).most_common(1)
    loose_all = score(per, keys_all)
    return {
        "loose": loose_all,
        "strict": score(per, keys_all, "strict"),
        "think_closed": 100 * sum(c for key in keys_all for c in per[key]["closed"]) / n_resp,
        "short_answer": 100 * sum(1 for a in answers if len(a.split()) < U.SHORT_ANSWER_WORDS) / n_resp,
        "empty_answer": 100 * sum(1 for a in answers if not a.strip()) / n_resp,
        "long_checker_error": 100 * long_err / long_n if long_n else 0.0,
        "answer_chars_median": statistics.median([len(a) for a in answers]) if answers else float("nan"),
        "dup_answer": 100 * top[0][1] / len(norm) if top else 0.0,
        "unique_word_ratio": mean([len(set(a.lower().split())) / len(a.split()) for a in answers if a.split()]),
        "nontrivial_over_all": score(per, keys_nontrivial) / loose_all if loose_all else float("nan"),
    }


def paired(pa, pb, keys, z, q, seed=20261109):
    k = len(next(iter(pa.values()))["loose"])
    a = [mean(pa[key]["loose"]) for key in keys]
    b = [mean(pb[key]["loose"]) for key in keys]
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return None
    delta = mean(d)
    se = math.sqrt(sum((x - delta) ** 2 for x in d) / (n * (n - 1)))
    se_samp = math.sqrt(sum(x * (1 - x) + y * (1 - y) for x, y in zip(a, b)) / ((k - 1) * n * n))
    rng = random.Random(seed)
    bs = sorted(mean([d[rng.randrange(n)] for _ in range(n)]) for _ in range(BOOT))
    return {"n": n, "delta": 100 * delta, "se": 100 * se, "ci_lo": 100 * (delta - z * se), "ci_hi": 100 * (delta + z * se),
            "boot_lo": 100 * bs[int(q * BOOT)], "boot_hi": 100 * bs[BOOT - 1 - int(q * BOOT)], "se_samp": 100 * se_samp}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dirs", nargs="+")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    raw = U._raw_rows()

    trivial = {key for key, row in raw.items() if any(all(U.score_answer(row, t)[1]) for t in U.TRIVIAL_ANSWERS)}
    overlap = U.SAME_TEMPLATE_TYPES | U.SIMILAR_TYPES
    subsets = {"all": sorted(raw), "nontrivial": sorted(k for k in raw if k not in trivial)}
    subsets["ood"] = sorted(k for k in raw if not set(raw[k]["instruction_id_list"]) & overlap)
    subsets["ood_nontrivial"] = sorted(set(subsets["ood"]) & set(subsets["nontrivial"]))
    print("subsets: " + " · ".join(f"{k} {len(v)}" for k, v in subsets.items()) + f"  (ood excludes {sorted(overlap)})")
    nondet = nondeterministic_keys(raw)
    print(f"nondeterministic official checker rows (match check tolerated, re-scored value used): {sorted(nondet)}")

    names, pers, valid = [], [], {}
    for rd in map(Path, a.result_dirs):
        resps, logged, task_res, f = load_run(rd)
        if set(resps) != set(raw):
            raise SystemExit(f"{rd}: samples cover {len(resps)} keys, expected {len(raw)}")
        per = rescore(resps, raw)
        mism = [k for k in raw if abs(mean(per[k]["loose"]) - logged[k]) > 1e-9]
        hard = [k for k in mism if k not in nondet]
        if hard:
            raise SystemExit(f"{rd}: re-score differs from logged per-doc loose on {len(hard)} deterministic prompts {hard[:10]}")
        name = run_name(rd)
        if name in valid:
            raise SystemExit(f"duplicate checkpoint name {name!r} ({rd}) — give distinct run directories")
        names.append(name)
        pers.append(per)
        valid[name] = invalid_reasons(task_res)
        print(f"{name}: {f.name} · re-score = logged on {len(raw) - len(mism)}/{len(raw)} prompts"
              + (f" (differs only on nondeterministic rows {mism}: " + ", ".join(f"logged {logged[k]:.3f} vs re-scored {mean(per[k]['loose']):.3f}" for k in mism) + ")" if mism else ""))

    k = len(next(iter(pers[0].values()))["loose"])
    out = {"subsets": {s: len(v) for s, v in subsets.items()}, "checkpoints": {}, "primary": [], "sensitivity": []}
    cols = ["loose", "strict", "think_closed", "short_answer", "empty_answer", "long_checker_error", "answer_chars_median",
            "dup_answer", "unique_word_ratio", "nontrivial_over_all"]
    print("\n## 체크포인트 (전체 294)\n")
    print("| checkpoint | 판정 | " + " | ".join(cols) + " |")
    print("|---|---|" + "---:|" * len(cols))
    for name, per in zip(names, pers):
        row = checkpoint_row(per, subsets["all"], subsets["nontrivial"], k)
        if row["long_checker_error"] > 100 * LONG_ANSWER_CHECKER_ERROR_MAX:
            valid[name] = valid[name] + [f"긴 답 검사기 예외 {row['long_checker_error']:.1f}%>{100 * LONG_ANSWER_CHECKER_ERROR_MAX:.0f}%"]
        flags = [c for c, bad in (("dup_answer", row["dup_answer"] > 10), ("nontrivial_over_all", row["nontrivial_over_all"] < 0.5)) if bad]
        verdict = ("무효: " + ", ".join(valid[name])) if valid[name] else "유효"
        verdict += (" ⚠퇴화 " + ",".join(flags)) if flags else ""
        out["checkpoints"][name] = {"verdict": verdict, "degenerate": bool(flags), "all": row,
                                    "subsets": {s: {"loose": score(per, keys), "strict": score(per, keys, "strict")} for s, keys in subsets.items()}}
        print(f"| {name} | {verdict} | " + " | ".join(f"{row[c]:.3f}" if c == "unique_word_ratio" else f"{row[c]:.2f}" if c == "nontrivial_over_all" else f"{row[c]:.1f}" for c in cols) + " |")
    print("\n| checkpoint | " + " | ".join(f"{s} loose / strict" for s in subsets) + " |")
    print("|---|" + "---:|" * len(subsets))
    for name in names:
        ss = out["checkpoints"][name]["subsets"]
        print(f"| {name} | " + " | ".join(f"{ss[s]['loose']:.1f} / {ss[s]['strict']:.1f}" for s in subsets) + " |")

    pairs = [(i + 1, i) for i in range(len(names) - 1)] + [(j, 0) for j in range(2, len(names))]
    m = len(pairs)
    q = 0.025 / m
    z = statistics.NormalDist().inv_cdf(1 - q)
    print(f"\n## 주 비교 — 전체 294 · loose · 짝 차 (뒤 − 앞), Bonferroni m={m} → z={z:.2f}, 부트스트랩 분위 {q:.4f}/{1 - q:.4f} × {BOOT}\n")
    print("| pair | n | Δ | 짝 SE | 구간 | 부트스트랩 | SE_samp | Δthink_closed | 판정 |")
    print("|---|---:|---:|---:|---|---|---:|---:|---|")
    for j, i in pairs:
        p = paired(pers[j], pers[i], subsets["all"], z, q)
        dtc = out["checkpoints"][names[j]]["all"]["think_closed"] - out["checkpoints"][names[i]]["all"]["think_closed"]
        if valid[names[j]] or valid[names[i]]:
            verdict = "무효 (무효 체크포인트 포함)"
        elif (p["ci_lo"] > 0 and p["boot_lo"] > 0) or (p["ci_hi"] < 0 and p["boot_hi"] < 0):
            verdict = "구분됨"
        else:
            verdict = "구분 불가"
        flag = " ⚠" if abs(dtc) > 2 else ""
        if out["checkpoints"][names[j]]["degenerate"] or out["checkpoints"][names[i]]["degenerate"]:
            verdict += " ⚠퇴화 체크포인트 포함"
        print(f"| {names[j]} − {names[i]} | {p['n']} | {p['delta']:+.1f} | {p['se']:.2f} | [{p['ci_lo']:+.1f}, {p['ci_hi']:+.1f}] | "
              f"[{p['boot_lo']:+.1f}, {p['boot_hi']:+.1f}] | {p['se_samp']:.2f} | {dtc:+.1f}{flag} | {verdict} |")
        out["primary"].append({"a": names[j], "b": names[i], **p, "d_think_closed": dtc, "verdict": verdict})

    print("\n읽기 규칙: \"구분 불가\" 는 \"같다\" 가 아니다 (80% 검정력의 최소 검출 차 약 3.5pp, 적대적 검토 모의실험). "
          "Δthink_closed ⚠ 인 쌍은 주 비교의 차이에 사고 마감 실패분이 들어 있으므로 아래 both_closed 의 부호와 함께 읽는다.")
    print("\n## 민감도 분석 — 판정 없음 (Δ · 짝 SE · 주 비교와 부호 일치)\n")
    print("| pair | subset | n | Δ | 짝 SE | 부호 일치 |")
    print("|---|---|---:|---:|---:|---|")
    for j, i in pairs:
        primary = next(x for x in out["primary"] if x["a"] == names[j] and x["b"] == names[i])
        both_closed = sorted(key for key in raw if all(pers[j][key]["closed"]) and all(pers[i][key]["closed"]))
        for sub, keys in [(s, subsets[s]) for s in ("nontrivial", "ood", "ood_nontrivial")] + [("both_closed", both_closed)]:
            p = paired(pers[j], pers[i], keys, z, q)
            if p is None:
                print(f"| {names[j]} − {names[i]} | {sub} | {len(keys)} | — | — | — |")
                continue
            same = "예" if (p["delta"] >= 0) == (primary["delta"] >= 0) else "아니오"
            print(f"| {names[j]} − {names[i]} | {sub} | {p['n']} | {p['delta']:+.1f} | {p['se']:.2f} | {same} |")
            out["sensitivity"].append({"a": names[j], "b": names[i], "subset": sub, "n": p["n"], "delta": p["delta"], "se": p["se"], "sign_agrees": same})
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
