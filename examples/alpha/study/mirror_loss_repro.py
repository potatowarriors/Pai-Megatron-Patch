#!/usr/bin/env python3
"""Reproduction script for the 2-node DiLoCo mirror-loss ("거울상 시소") root cause.

Claim being verified (full report: study/mirror_loss_aliasing.md):
  The anti-correlated per-node lm-loss oscillation that appeared at the P2b
  blend switch is fully explained by DILOCO_DATA_SHARD's even/odd split of the
  deterministic BlendedDataset index sequence, whose parity assignment
  PRECESSES because the 6-decimal blend weights sum to 1 +/- 1e-6 and
  normalize() shifts every weight off its exact rational. Period law:
  one parity flip cycle = 2/|sum(w)-1| global samples (= 325.5 iters at
  GBS 3072 x 2 nodes for residual 1e-6).

Reproduces, from raw tensorboard logs + Megatron's own compiled C++ helper:
  1. zero-sum anti-correlation of the two nodes' lm/seq-bal loss (band-passed)
  2. per-iteration composition deltas that predict the measured gap
     (R^2 ~ 0.83 raw / 0.98 smoothed; collapses to ~0.4 at +/-1 iter shift)
  3. counterfactual period scaling: residual 1e-6 -> ~313 it, 3e-6 -> ~108 it,
     0 -> static offset (P2 regime)
  4. out-of-sample replication on the 08/07 window (R^2 ~ 0.84)

Usage (needs the Megatron submodule importable and tensorboard installed):
  python mirror_loss_repro.py
Constants below pin the run layout of the 2026-08 stage2 pair; adjust RUNS /
CONSUMED_AT_FIRST_ITER for other windows.
"""
import math
import re
import sys
from pathlib import Path

import numpy as np

ALPHA = Path(__file__).resolve().parents[1]
REPO = ALPHA.parents[1]
sys.path.insert(0, str(REPO / "backends/megatron/Megatron-LM-251125"))
from megatron.core.datasets import helpers  # compiled C++ blending, the real thing

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT = ALPHA / "outputs"
CFG = ALPHA / "configs/data"
GBS = 3072            # per-node global batch size in the analyzed window
WORLD = 2             # DiLoCo pair

# main1 = node0 (DILOCO_RANK=0), sub1 = node1. 08/11 bias-fix restart pair.
RUNS = {
    "node0": OUT / "alpha_baseline_48L_stage2_20260811_122136/tensorboard",
    "node1": OUT / "alpha_baseline_48L_stage2_20260811_121528/tensorboard",
}
# 08/07 pre-restart P2b pair, used as the out-of-sample window.
RUNS_OOS = {
    "node0": OUT / "alpha_baseline_48L_stage2_20260807_154327/tensorboard",
    "node1": OUT / "alpha_baseline_48L_stage2_20260807_153725/tensorboard",
}


def load_scalars(tb_dir, tags):
    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for t in tags:
        ev = ea.Scalars(t)
        out[t] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def load_weights(yaml_path):
    toks = re.search(r'data-path:\s*"([^"]+)"', open(yaml_path).read()).group(1).split()
    w = np.array([float(toks[i]) for i in range(0, len(toks), 2)], dtype=np.float64)
    names = [toks[i + 1].split("stage2_packed/")[1].rsplit("/data_text_document", 1)[0]
             for i in range(0, len(toks), 2)]
    return w, names


def build_blend_index(weights, size):
    """Exactly what BlendedDataset does: normalize + greedy max-error (helpers.cpp)."""
    w = weights / weights.sum()
    ds_idx = np.zeros(size, dtype=np.int16)
    ds_smp = np.zeros(size, dtype=np.int64)
    helpers.build_blending_indices(ds_idx, ds_smp, w, len(w), size, False)
    return ds_idx


def window_deltas(ds_idx, consumed_after, gbs, ndat):
    """Per-iteration (node0 - node1) composition counts.

    Iteration with post-increment consumed count c covers LOCAL samples
    [c-gbs, c) on each node -> GLOBAL blend positions [2(c-gbs), 2c);
    even positions are node0's (shard map: global = 2*local + rank)."""
    delta = np.zeros((len(consumed_after), ndat), dtype=np.int32)
    tot = np.zeros_like(delta)
    for t, c in enumerate(consumed_after):
        win = ds_idx[2 * (int(c) - gbs): 2 * int(c)]
        be = np.bincount(win[0::2], minlength=ndat)
        bo = np.bincount(win[1::2], minlength=ndat)
        delta[t], tot[t] = be - bo, be + bo
    return delta, tot


def ols_r2(X, y):
    X1 = np.column_stack([X, np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return 1 - np.var(y - X1 @ beta) / np.var(y)


def smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode="same")


def dominant_period(x, min_period_guard=3000):
    x = smooth(x, 21)[30:-30]
    x = x - x.mean()
    if x.std() < 1e-9:
        return math.inf
    F = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    fr = np.fft.rfftfreq(len(x), 1.0)
    m = fr > 1.0 / min_period_guard
    return 1.0 / fr[m][np.argmax(F[m])]


def main():
    tags = ["lm loss", "seq_load_balancing_loss", "lm loss vs samples"]
    d0 = load_scalars(RUNS["node0"], tags)
    d1 = load_scalars(RUNS["node1"], tags)
    it = d0["lm loss"][0]
    assert np.array_equal(it, d1["lm loss"][0])
    lm_gap = d0["lm loss"][1] - d1["lm loss"][1]
    sb_gap = d0["seq_load_balancing_loss"][1] - d1["seq_load_balancing_loss"][1]
    consumed = d0["lm loss vs samples"][0].astype(np.int64)  # post-increment counter

    print(f"window: iters {it[0]}..{it[-1]}  consumed {consumed[0]}..{consumed[-1]}")

    # 1. zero-sum anti-correlation
    for name, gap, a, b in [("lm", lm_gap, d0["lm loss"][1], d1["lm loss"][1]),
                            ("seq_bal", sb_gap, d0["seq_load_balancing_loss"][1],
                             d1["seq_load_balancing_loss"][1])]:
        W, w = 201, 21
        r0 = smooth(a - smooth(a, W), w)[W:-W]
        r1 = smooth(b - smooth(b, W), w)[W:-W]
        c = np.corrcoef(r0, r1)[0, 1]
        ratio = np.var((r0 - r1) / 2) / max(np.var((r0 + r1) / 2), 1e-30)
        print(f"  {name}: band corr {c:+.3f}, var(anti)/var(common) = {ratio:.0f}x, "
              f"gap period ~ {dominant_period(gap):.0f} it")

    # 2. simulation vs measurement (+ alignment shift test)
    w2b, names = load_weights(CFG / "stage2_v5_blend_packed_p2b.yaml")
    print(f"p2b raw weight sum = {w2b.sum():.9f} (residual {w2b.sum()-1:+.1e}) "
          f"-> theory period {2/abs(w2b.sum()-1)/(WORLD*GBS):.1f} it")
    idx = build_blend_index(w2b, 2 * int(consumed[-1]) + 8 * WORLD * GBS)
    for shift in (-1, 0, 1):
        delta, tot = window_deltas(idx, consumed + shift * GBS, GBS, len(w2b))
        r2 = ols_r2(delta / GBS, lm_gap)
        mark = "  <-- exact alignment" if shift == 0 else ""
        print(f"  shift {shift:+d}: lm-gap R^2 = {r2:.3f}{mark}")
    delta, tot = window_deltas(idx, consumed, GBS, len(w2b))
    print(f"  seq-bal R^2 = {ols_r2(delta / GBS, sb_gap):.3f}")
    print(f"  pair-union max deviation from exact blend = "
          f"{np.abs(tot - WORLD * GBS * (w2b / w2b.sum())[None, :]).max():.1f} samples")

    # 3. counterfactual period law (from position 0; start phase is irrelevant)
    icc = names.index("cc_code")
    NW = 3500
    for val, label in [(0.080000, "residual +1e-6"), (0.080002, "residual +3e-6"),
                       (0.079999, "residual 0 (sum=1)")]:
        wv = w2b.copy()
        wv[icc] = val
        di = build_blend_index(wv, NW * WORLD * GBS + 10)
        dd, _ = window_deltas(di, np.arange(1, NW + 1) * GBS, GBS, len(wv))
        top = int(np.argmax(dd.std(axis=0)))
        print(f"  CF {label}: top mover {names[top]} period "
              f"{dominant_period(dd[:, top] / GBS):.0f} it, "
              f"osc std {dd[:, top].std():.0f}, static mean {dd[:, top].mean():+.0f}")

    # 4. out-of-sample window
    o0 = load_scalars(RUNS_OOS["node0"], ["lm loss", "lm loss vs samples"])
    o1 = load_scalars(RUNS_OOS["node1"], ["lm loss"])
    common, i0, i1 = np.intersect1d(o0["lm loss"][0], o1["lm loss"][0],
                                    return_indices=True)
    gap_oos = o0["lm loss"][1][i0] - o1["lm loss"][1][i1]
    cons_oos = o0["lm loss vs samples"][0][i0].astype(np.int64)
    d_oos, _ = window_deltas(idx, cons_oos, GBS, len(w2b))
    print(f"  out-of-sample (iters {common[0]}..{common[-1]}): "
          f"R^2 = {ols_r2(d_oos / GBS, gap_oos):.3f}")


if __name__ == "__main__":
    main()
