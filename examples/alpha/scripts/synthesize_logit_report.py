#!/usr/bin/env python3
"""
Synthesis for the v1-vs-v2 attention-logit study: joins the weight-trajectory (gamma)
and logit-trajectory (measured) JSONs into

  1) a twin-axis "co-explosion" plot — QK-norm gamma (left) + measured max logit (right)
     vs consumed tokens, showing v1 gamma+logit rise together while v2 stays bounded, and
  2) a machine-readable summary (printed as a markdown table) of first/last values per run.

v2-stage1 and v2-resume are merged into one "v2" timeline (sorted by consumed tokens).

Usage:
  python synthesize_logit_report.py \
    --weight outputs/analysis/v1_vs_v2_logit_explosion/weight_trajectory.json \
    --logit  outputs/analysis/v1_vs_v2_logit_explosion/logit_trajectory.json \
    --plot   outputs/analysis/v1_vs_v2_logit_explosion/plots
"""

import argparse
import json
import os


def fam(group: str) -> str:
    return "v1" if group == "v1" else "v2"


def gamma_series(weight_json):
    """family -> sorted [(tokens, qk_absmax, final_absmax)]."""
    out = {}
    for run in weight_json["runs"]:
        f = fam(run["group"])
        for r in run["recs"]:
            qk = max((v["abs_max"] for v in r["qk_gamma"].values()), default=float("nan"))
            final = next((v["abs_max"] for k, v in r["rms_gamma"].items() if "final" in k), float("nan"))
            out.setdefault(f, []).append((r["consumed_tokens"], qk, final))
    for f in out:
        out[f].sort()
    return out


def logit_series(logit_json):
    """family -> sorted [(tokens, random_max, real_max)]."""
    out = {}
    for run in logit_json["runs"]:
        f = fam(run["group"])
        for p in run["points"]:
            res = p.get("result")
            if not res:
                continue
            rnd = res.get("random", {}).get("global", {}).get("global_max")
            real = res.get("real", {}).get("global", {}).get("global_max")
            out.setdefault(f, []).append((p["consumed_tokens"], rnd, real))
    for f in out:
        out[f].sort()
    return out


FAM_COLOR = {"v1": "#d62728", "v2": "#2ca02c"}


def twin_axis_plot(gser, lser, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)

    fig, axL = plt.subplots(figsize=(11, 6))
    axR = axL.twinx()
    for f in ("v1", "v2"):
        c = FAM_COLOR[f]
        if f in gser:
            xs = [t / 1e9 for t, _, _ in gser[f]]
            axL.plot(xs, [g for _, g, _ in gser[f]], color=c, marker="o", ls="-",
                     label=f"{f} QK |γ|max (left)")
        if f in lser:
            xs = [t / 1e9 for t, _, _ in lser[f]]
            axR.plot(xs, [r for _, r, _ in lser[f]], color=c, marker="x", ls="--",
                     label=f"{f} max logit (right)")
    axL.axhline(8.0, ls=":", c="orange", lw=1)
    axR.set_yscale("log")
    axL.set_xlabel("consumed tokens (B)")
    axL.set_ylabel("QK-norm effective |γ| max (linear)")
    axR.set_ylabel("measured max attention logit (log)")
    axL.set_title("v1 co-explosion vs v2 joint stability — QK gamma (solid) & max logit (dashed)")
    lL, labL = axL.get_legend_handles_labels()
    lR, labR = axR.get_legend_handles_labels()
    axL.legend(lL + lR, labL + labR, fontsize=8, loc="upper left")
    axL.grid(alpha=0.3)
    p = os.path.join(plot_dir, "twin_axis_gamma_logit.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    return p


def md_table(gser, lser):
    lines = ["| family | tokens(B) | QK |γ|max | final |γ|max | max logit (rand) | max logit (real) |",
             "|---|---|---|---|---|---|---|"]
    # build token-aligned lookups
    for f in ("v1", "v2"):
        g = {t: (qk, fin) for t, qk, fin in gser.get(f, [])}
        l = {t: (rnd, real) for t, rnd, real in lser.get(f, [])}
        toks = sorted(set(g) | set(l))
        for t in toks:
            qk, fin = g.get(t, (None, None))
            rnd, real = l.get(t, (None, None))
            def fmt(x, p=2):
                return f"{x:.{p}f}" if isinstance(x, (int, float)) else "—"
            lines.append(f"| {f} | {t/1e9:.1f} | {fmt(qk,3)} | {fmt(fin,3)} | {fmt(rnd,1)} | {fmt(real,1)} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--logit", required=True)
    ap.add_argument("--plot", required=True)
    args = ap.parse_args()

    wj = json.load(open(args.weight))
    lj = json.load(open(args.logit))
    gser = gamma_series(wj)
    lser = logit_series(lj)
    p = twin_axis_plot(gser, lser, args.plot)
    print(f"Wrote twin-axis plot: {p}\n")
    print(md_table(gser, lser))


if __name__ == "__main__":
    main()
