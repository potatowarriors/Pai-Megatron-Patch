"""Unit tests for examples/alpha/eval_sft/tau_combine.py (τ³-bench 결과 합산 → results_tau.json).

Run from the repo root:
    python3 -m pytest tests/test_tau_combine.py -v
Synthetic tau2-bench results.json + proxy counters; no harness install needed (xcheck disabled).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples/alpha/eval_sft"))

import tau_combine as tc  # noqa: E402


def _sim(task, trial, reward, term="user_stop", n_asst=5):
    return {"id": f"{task}-{trial}", "task_id": task, "trial": trial, "termination_reason": term, "duration": 60.0,
            "reward_info": None if reward is None else {"reward": reward},
            "messages": [{"role": "assistant", "content": "x"}] * n_asst + [{"role": "user", "content": "y"}]}


def _write_domain(raw: Path, d: str, sims, tasks, proxy_before, proxy_after):
    (raw / d).mkdir(parents=True)
    json.dump({"info": {"environment_info": {"task_split_name": "base"}}, "tasks": [{"id": t} for t in tasks],
               "simulations": sims}, open(raw / d / "results.json", "w"))
    json.dump(proxy_before, open(raw / f"proxy_{d}_before.json", "w"))
    json.dump(proxy_after, open(raw / f"proxy_{d}_after.json", "w"))


P0 = {k: 0 for k in tc.PROXY_KEYS}


def test_pass_k_formula():
    assert tc.pass_k(4, 4, 4) == 1.0 and tc.pass_k(2, 4, 1) == 0.5
    assert tc.pass_k(2, 4, 2) == math.comb(2, 2) / math.comb(4, 2)
    assert math.isnan(tc.pass_k(1, 1, 2))


def test_combine_two_domains(tmp_path):
    raw = tmp_path / "raw"; out = tmp_path / "out"
    # retail: 2 tasks × 2 trials — t1 passes both, t2 passes once
    sims = [_sim("t1", 0, 1.0), _sim("t1", 1, 1.0), _sim("t2", 0, 1.0), _sim("t2", 1, 0.0)]
    pa = dict(P0, requests=8, reinlined=6, miss=0, miss_first_assistant=8, think_stripped=8, think_unclosed=0, tool_calls=3)
    _write_domain(raw, "retail", sims, ["t1", "t2"], P0, pa)
    # airline: 1 task × 2 trials — one infra error (excluded), one fail; plus a user_error (harness fail)
    sims = [_sim("a1", 0, None, term="infrastructure_error"), _sim("a1", 1, 0.0, term="user_error")]
    pb = dict(P0, requests=3, reinlined=1, miss=0, miss_first_assistant=3, think_stripped=2, think_unclosed=1)
    _write_domain(raw, "airline", sims, ["a1"], P0, pb)
    o = tc.combine(str(out), str(raw), 2, ["retail", "airline", "telecom"], {"telecom": "user endpoint rejects tools"},
                   "openai/gemma", {"temperature": 0.0}, {"temperature": 1.0}, True, str(tmp_path), False, xcheck=False)
    r = o["results"]
    assert r["tau_retail"]["pass1,none"] == 0.75                 # mean(1.0, 0.5)
    assert r["tau_retail"]["pass2,none"] == 0.5                  # mean(1.0, 0.0)
    assert r["tau_retail"]["no_answer,none"] == 0.0 and r["tau_retail"]["think_closed,none"] == 1.0
    assert r["tau_airline"]["pass1,none"] == 0.0                 # infra sim excluded, remaining fails
    assert r["tau_airline"]["no_answer,none"] == 1.0             # infra(제외됐어도 하니스 실패) + user_error = 2 of 2 sims
    assert abs(r["tau_airline"]["think_closed,none"] - 2 / 3) < 1e-9
    assert "tau_telecom" not in r and o["tau_detail"]["skipped_domains"] == {"telecom": "user endpoint rejects tools"}
    assert r["tau_bench"]["pass1,none"] == 0.375 and r["tau_bench"]["no_answer,none"] == 1.0   # max over domains
    d = o["tau_detail"]["domains"]["retail"]
    assert d["n_sims"] == 4 and d["proxy"]["miss_rate"] == 0.0 and d["avg_agent_turns"] == 5.0
    assert o["tau_detail"]["invalid_reasons"] == []              # complete run, valid
    assert json.load(open(out / "results_tau.json"))["results"]["tau_bench"]["pass1,none"] == 0.375


def test_invalidation_rules(tmp_path):
    raw = tmp_path / "raw"; out = tmp_path / "out"
    sims = [_sim("t1", 0, 1.0)]                                 # trials=2 requested but only 1 sim -> incomplete
    pa = dict(P0, requests=2, reinlined=1, miss=1, think_stripped=2)  # miss_rate 0.5 > 0.05
    _write_domain(raw, "retail", sims, ["t1"], P0, pa)
    o = tc.combine(str(out), str(raw), 2, ["retail", "airline"], {}, "u", {}, {}, True, str(tmp_path), False, xcheck=False)
    inv = o["tau_detail"]["invalid_reasons"]
    assert any("sims 1 < 1×2" in s for s in inv) and any("miss_rate" in s for s in inv)
    assert any("airline: results.json 없음" in s for s in inv)
    assert o["results"]["tau_retail"]["no_answer,none"] == 1.0 and o["results"]["tau_bench"]["no_answer,none"] == 1.0
    assert o["results"]["tau_retail"]["pass1,none"] == 1.0      # score kept, cell marked invalid via no_answer


def test_subsampled_and_no_think_seen(tmp_path):
    raw = tmp_path / "raw"; out = tmp_path / "out"
    _write_domain(raw, "retail", [_sim("t1", 0, 1.0)], ["t1"], P0, dict(P0, requests=1))  # proxy saw no think
    o = tc.combine(str(out), str(raw), 1, ["retail"], {}, "u", {}, {}, True, str(tmp_path), True, xcheck=False)
    inv = o["tau_detail"]["invalid_reasons"]
    assert any("부분 표본" in s for s in inv) and any("think 를 하나도" in s for s in inv)
    assert o["tau_detail"]["subsampled"] is True


def test_no_reattach_ignores_miss_rate(tmp_path):
    raw = tmp_path / "raw"; out = tmp_path / "out"
    _write_domain(raw, "retail", [_sim("t1", 0, 1.0)], ["t1"], P0, dict(P0, requests=2, reinlined=0, miss=2, think_stripped=2))
    o = tc.combine(str(out), str(raw), 1, ["retail"], {}, "u", {}, {}, False, str(tmp_path), False, xcheck=False)
    assert o["tau_detail"]["invalid_reasons"] == [] and o["tau_detail"]["reattach"] is False


def test_restored_fallback_for_pre_a834e48_stats(tmp_path):
    """restored 키가 없는 과거 통계: restore 는 reinlined+필드 인라인, strip 은 필드 인라인만 (reinlined 합산 금지)."""
    raw = tmp_path / "raw"
    old_restore = dict(P0, requests=3, reinlined=5, reasoning_field_inlined=2, think_stripped=3, reattach=True); old_restore.pop("restored")
    _write_domain(raw, "retail", [_sim("t1", 0, 1.0)], ["t1"], P0, old_restore)
    d = tc.proxy_delta(str(raw), "retail")
    assert d["restored"] == 7 and d["restored_derived"] is True
    old_strip = dict(P0, requests=3, reinlined=66, reasoning_field_inlined=0, think_stripped=3, reattach=False); old_strip.pop("restored")
    _write_domain(raw, "airline", [_sim("a1", 0, 1.0)], ["a1"], P0, old_strip)
    d = tc.proxy_delta(str(raw), "airline")
    assert d["restored"] == 0                      # 정상 strip: 적중 66 이어도 적용 0
    new_stats = dict(P0, requests=3, reinlined=4, restored=4, think_stripped=3)
    _write_domain(raw, "mock", [_sim("m1", 0, 1.0)], ["m1"], P0, new_stats)
    d = tc.proxy_delta(str(raw), "mock")
    assert d["restored"] == 4 and "restored_derived" not in d
