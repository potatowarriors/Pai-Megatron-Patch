#!/usr/bin/env python3
"""tau_combine.py — τ³-bench 도메인별 results.json → eval_sft 결과 규약(results_tau.json).

run_tau.sh 의 마지막 단계. 도메인마다 tau2-bench 의 `simulations/<rid>_<domain>/results.json` 을 읽어
  - pass^k (k=1..trials): 과제별 comb(c,k)/comb(n,k) 의 평균. tau2.metrics 와 같이 INFRASTRUCTURE_ERROR 는 제외.
    tau2 자체 compute_metrics 가 import 되면 pass^1 을 대조한다(불일치 = 무효).
  - no_answer: 하니스 측 실패 비율 (infrastructure/unexpected/user_error/timeout/agent_error) — 모델 실패(max_steps,
    too_many_errors, context_window_exceeded)는 점수에 반영되고 여기 안 들어간다.
  - think_closed: 프록시 카운터((think_stripped+think_from_field) / (…+unclosed)).
무효 조건(→ 모든 tau 셀 no_answer=1.0, bench_registry.invalid_reasons 가 자동 무효 처리):
  부분 표본 · 도메인 results.json 부재 · sims < tasks×trials · reattach 켠 채 miss_rate>0.05 · 프록시가 think 를 하나도 못 봄 ·
  pass^1 자체계산 ≠ tau2 계산.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os

HARNESS_FAIL = {"infrastructure_error", "unexpected_error", "user_error", "timeout", "agent_error"}
PROXY_KEYS = ("requests", "reinlined", "restored", "miss", "miss_first_assistant", "think_stripped", "think_from_field", "think_absent", "think_unclosed",
              "tool_calls", "mixed_content_and_tools", "finish_length", "upstream_errors", "seed_stripped")


def pass_k(c: int, n: int, k: int) -> float:
    return math.comb(c, k) / math.comb(n, k) if n >= k else float("nan")


def proxy_delta(raw: str, d: str) -> dict:
    try:
        a = json.load(open(os.path.join(raw, f"proxy_{d}_before.json")))
        b = json.load(open(os.path.join(raw, f"proxy_{d}_after.json")))
    except Exception:  # noqa: BLE001
        return {}
    dl = {k: int(b.get(k, 0)) - int(a.get(k, 0)) for k in PROXY_KEYS}
    dl["reasoning_field_inlined"] = int(b.get("reasoning_field_inlined", 0)) - int(a.get("reasoning_field_inlined", 0))
    if "restored" not in b:
        # a834e48 이전 통계: restored 키가 없다. 모드별 대체 — restore 는 캐시 적중(reinlined)+필드 인라인,
        # strip 은 필드 인라인만(strip 의 reinlined 는 적중일 뿐 적용이 아니므로 합산 금지; 77040d3 이전 strip 의
        # 필드 인라인은 실제 누수라 그대로 restored 로 친다). 동료 세션 run_swe.sh 파서와 같은 규칙.
        reattach = bool(b.get("reattach", True))
        dl["restored"] = (dl["reinlined"] if reattach else 0) + dl["reasoning_field_inlined"]
        dl["restored_derived"] = True
    hm = dl["reinlined"] + dl["miss"]
    dl["miss_rate"] = (dl["miss"] / hm) if hm else 0.0
    return dl


def tau2_xcheck(path: str):
    """tau2 자체 메트릭으로 pass^k 를 계산 (venv 에서만 가능; 실패 시 사유 문자열)."""
    try:
        from tau2.data_model.simulation import Results
        from tau2.metrics.agent_metrics import compute_metrics
        res = Results.load(path) if hasattr(Results, "load") else Results(**json.load(open(path)))
        m = compute_metrics(res)
        return {int(k): float(v) for k, v in (m.pass_hat_ks or {}).items()}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}


def summarize_domain(raw: str, d: str, K: int, reattach: bool, xcheck: bool = True):
    p = os.path.join(raw, d, "results.json")
    if not os.path.exists(p):
        return None, None, [f"{d}: results.json 없음"]
    R = json.load(open(p))
    sims = R.get("simulations") or []
    tasks = R.get("tasks") or []
    per: dict[str, list[bool]] = {}
    term: dict[str, int] = {}
    turns, dur, invalid = [], [], []
    n_infra = 0
    for s in sims:
        tr = s.get("termination_reason")
        term[tr] = term.get(tr, 0) + 1
        if tr == "infrastructure_error":
            n_infra += 1
            continue
        rw = (s.get("reward_info") or {}).get("reward")
        ok = rw is not None and abs(float(rw) - 1.0) <= 1e-6
        per.setdefault(s["task_id"], []).append(ok)
        msgs = s.get("messages") or []
        turns.append(sum(1 for m in msgs if m.get("role") == "assistant"))
        dur.append(float(s.get("duration") or 0))
    n_sims = len(sims)
    pk = {}
    for k in range(1, K + 1):
        vals = [pass_k(sum(v), len(v), k) for v in per.values() if len(v) >= k]
        pk[k] = (sum(vals) / len(vals)) if vals else float("nan")
    xc = tau2_xcheck(p) if xcheck else None
    if isinstance(xc, dict) and 1 in xc and not math.isnan(pk[1]) and abs(xc[1] - pk[1]) > 1e-6:
        invalid.append(f"{d}: pass^1 자체계산 {pk[1]:.4f} ≠ tau2 {xc[1]:.4f}")
    harness_fail = sum(v for k, v in term.items() if k in HARNESS_FAIL)
    no_answer = (harness_fail / n_sims) if n_sims else 1.0
    pd = proxy_delta(raw, d)
    st, un = pd.get("think_stripped", 0) + pd.get("think_from_field", 0), pd.get("think_unclosed", 0)
    think_closed = (st / (st + un)) if (st + un) else 0.0
    if n_sims < len(tasks) * K:
        invalid.append(f"{d}: sims {n_sims} < {len(tasks)}×{K}")
    if reattach and pd.get("miss_rate", 0) > 0.05:
        invalid.append(f"{d}: 프록시 miss_rate {pd['miss_rate']:.3f} > 0.05")
    if pd and (st + un) == 0:
        invalid.append(f"{d}: 프록시가 think 를 하나도 못 봤다 (경로 우회 또는 reasoning 파서 ON?)")
    res = {f"pass{k},none": pk[k] for k in pk}
    res["no_answer,none"] = no_answer
    res["think_closed,none"] = think_closed
    detail = {"n_tasks": len(tasks), "n_tasks_with_sims": len(per), "n_sims": n_sims, "trials": K,
              "task_split": ((R.get("info") or {}).get("environment_info") or {}).get("task_split_name"),
              "termination_reasons": term, "n_infra_excluded": n_infra,
              "avg_agent_turns": (sum(turns) / len(turns)) if turns else None,
              "avg_duration_s": (sum(dur) / len(dur)) if dur else None,
              "pass_k_tau2_xcheck": xc, "proxy": pd}
    return res, detail, invalid


def combine(out: str, raw: str, K: int, domains: list[str], skipped: dict, user_llm: str, user_args: dict,
            agent_args: dict, reattach: bool, home: str, subsampled: bool, xcheck: bool = True) -> dict:
    try:
        commit = open(os.path.join(home, ".pinned_commit")).read().strip()
    except Exception:  # noqa: BLE001
        commit = None
    results, detail_doms, invalid = {}, {}, []
    for d in domains:
        if d in skipped:
            continue
        res, det, inv = summarize_domain(raw, d, K, reattach, xcheck)
        invalid += inv
        if res is not None:
            results[f"tau_{d}"] = res
            detail_doms[d] = det
    if results:
        ks = [k for k in next(iter(results.values())) if k.startswith("pass")]
        agg = {k: sum(r[k] for r in results.values()) / len(results) for k in ks}
        agg["no_answer,none"] = max(r["no_answer,none"] for r in results.values())
        agg["think_closed,none"] = min(r["think_closed,none"] for r in results.values())
        results["tau_bench"] = agg
    if subsampled:
        invalid.append("부분 표본(subsampled)")
    if invalid:
        for r in results.values():
            r["no_answer,none"] = 1.0
    out_json = {"results": results,
                "tau_detail": {"harness": "tau2-bench v1.0.1 (τ³)", "harness_commit": commit, "domains_requested": domains,
                               "skipped_domains": skipped, "user_llm": user_llm, "user_llm_args": user_args,
                               "agent_llm_args": agent_args, "trials": K, "reattach": reattach,
                               "domains": detail_doms, "invalid_reasons": invalid, "subsampled": subsampled}}
    os.makedirs(out, exist_ok=True)
    json.dump(out_json, open(os.path.join(out, "results_tau.json"), "w"), ensure_ascii=False, indent=2)
    return out_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--raw", required=True)
    ap.add_argument("--trials", type=int, required=True); ap.add_argument("--domains", required=True)
    ap.add_argument("--skipped-json", default="{}"); ap.add_argument("--user-llm", default="")
    ap.add_argument("--user-args", default="{}"); ap.add_argument("--agent-args", default="{}")
    ap.add_argument("--reattach", default="1"); ap.add_argument("--home", default="")
    ap.add_argument("--subsampled", action="store_true"); ap.add_argument("--no-xcheck", action="store_true")
    a = ap.parse_args()
    logging.disable(logging.CRITICAL)
    o = combine(a.out, a.raw, a.trials, a.domains.split(), json.loads(a.skipped_json), a.user_llm,
                json.loads(a.user_args), json.loads(a.agent_args), a.reattach == "1", a.home, a.subsampled,
                xcheck=not a.no_xcheck)
    K = a.trials
    for d, dd in o["tau_detail"]["domains"].items():
        r = o["results"][f"tau_{d}"]
        pks = " ".join(f"pass^{k} {r[f'pass{k},none'] * 100:.1f}" for k in range(1, K + 1))
        print(f"[tau] {d:8s} {pks}  n={dd['n_tasks']}×{K} sims={dd['n_sims']} harness_fail={r['no_answer,none'] * 100:.1f}% "
              f"miss={dd['proxy'].get('miss_rate', 0) * 100:.1f}% mixed={dd['proxy'].get('mixed_content_and_tools', 0)} "
              f"term={dd['termination_reasons']}")
    if "tau_bench" in o["results"]:
        print(f"[tau] tau_bench pass^1 {o['results']['tau_bench']['pass1,none'] * 100:.1f} (도메인 평균)")
    if o["tau_detail"]["invalid_reasons"]:
        print(f"[tau] ⚠️ 무효 사유: {o['tau_detail']['invalid_reasons']} → no_answer=1.0 (집계 무효)")
    print(f"[tau] → {a.out}/results_tau.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
