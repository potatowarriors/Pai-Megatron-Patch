#!/usr/bin/env python3
"""
Weight-trajectory analysis for Alpha checkpoints — v1 (logit-explosion) vs v2 (stable) study.

Reads Megatron **torch_dist** checkpoints directly (the raw `iter_NNNNNNN/` dirs with
`.distcp` shards + `common.pt`) — NO HuggingFace conversion, NO forward pass, NO training
launch. Everything is computed from static weights, so it is cheap to run on CPU at every
save interval, across many checkpoints, for several runs at once.

Primary question: how do the QK-LayerNorm gammas (and other RMSNorm gammas) evolve over
training, and does v1 reproduce the known Stage-1 gamma blowup (|γ| -> ~12) while v2 with
`apply_wd_to_qk_layernorm` stays bounded (~1-2)?

KEY CORRECTNESS DETAIL — zero-centered gamma convention (per-run, from common.pt):
  * apply_layernorm_1p=True  (v1): stored weight w has EFFECTIVE gamma = 1 + w  (init w=0)
  * apply_layernorm_1p=False (v2): EFFECTIVE gamma = w                          (init w=1)
Raw `.weight` tensors from v1 and v2 are NOT directly comparable; we read each run's
`apply_layernorm_1p` from common.pt and convert to effective gamma before any statistic.
(This is the same trap that caused the "AlphaRMSNorm 1p" eval bug — see examples/alpha/CLAUDE.md.)

X-axis fairness: v1 GBS=256, v2 GBS=1536 → iteration is not comparable. We plot against
consumed TOKENS = consumed_train_samples * seq_length (preferred from common.pt; falls back
to iteration*GBS*seq_len), so the two runs overlay on a fair budget axis.

Usage:
  export PYTHONPATH=<repo>:<repo>/backends/megatron/Megatron-LM-251125:$PYTHONPATH
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
  python analyze_weight_trajectory.py \
    --runs outputs/alpha_v1/alpha_baseline_48L_20251219_095156 \
           outputs/alpha_baseline_48L_stage1_20260512_170157 \
           outputs/alpha_baseline_48L_stage1_resume_20260526_194537 \
    --output outputs/analysis/v1_vs_v2_logit_explosion/weight_trajectory.json \
    --plot   outputs/analysis/v1_vs_v2_logit_explosion/plots
"""

import argparse
import json
import os
import re
import sys

import torch

# Reuse the battle-tested torch_dist loaders from the sibling analysis script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_expert_balance import (  # noqa: E402
    load_named_tensors,
    load_common_args,
    discover_layers,
)

# Effective-gamma histogram bins (covers healthy ~1 and the ~12 blowup, plus negatives).
HIST_EDGES = [round(-4.0 + 0.25 * i, 4) for i in range(81)]  # [-4, 16], 80 bins


# ----------------------------------------------------------------------------
# Run / checkpoint discovery
# ----------------------------------------------------------------------------

def find_checkpoints(run_dir: str):
    """Return sorted list of iter_NNNNNNN/ torch_dist dirs under a run (or ckpt) dir."""
    run_dir = run_dir.rstrip("/")
    # accept either a run dir (has checkpoints/) or a checkpoints/ dir directly
    cand = os.path.join(run_dir, "checkpoints")
    base = cand if os.path.isdir(cand) else run_dir
    iters = []
    for name in os.listdir(base):
        m = re.fullmatch(r"iter_(\d+)", name)
        full = os.path.join(base, name)
        if m and os.path.isfile(os.path.join(full, "common.pt")):
            iters.append((int(m.group(1)), full))
    return [p for _, p in sorted(iters)]


def derive_group(run_dir: str) -> str:
    """Auto-label a run as v1 / v2 family for plot coloring."""
    s = run_dir.lower()
    if "alpha_v1" in s or "_v1" in s:
        return "v1"
    if "resume" in s:
        return "v2-resume"
    if "stage1" in s:
        return "v2-stage1"
    return os.path.basename(run_dir.rstrip("/"))[:24]


def consumed_tokens(cargs, iteration: int):
    """(tokens, samples, source) using consumed_train_samples when available."""
    seq = int(getattr(cargs, "seq_length", 0) or 0)
    cs = getattr(cargs, "consumed_train_samples", None)
    src = "consumed_train_samples"
    if not cs:
        gbs = int(getattr(cargs, "global_batch_size", 0) or 0)
        cs = iteration * gbs
        src = f"iter*GBS({gbs})"
    return int(cs) * seq, int(cs), src


# ----------------------------------------------------------------------------
# Gamma statistics (effective-gamma, convention-aware)
# ----------------------------------------------------------------------------

def gamma_stats(weight: torch.Tensor, zero_centered: bool, keep_vector: bool):
    """Effective-gamma summary for one norm tensor.

    zero_centered=True applies the v1 (apply_layernorm_1p) transform γ = 1 + w.
    keep_vector=True stores the full per-channel γ vector (only for small QK / final norms).
    """
    w = weight.detach().float().cpu().flatten()
    g = (1.0 + w) if zero_centered else w
    ag = g.abs()
    n = g.numel()
    counts = torch.histogram(g, bins=torch.tensor(HIST_EDGES)).hist.to(torch.int64).tolist()
    out = {
        "numel": int(n),
        "mean": g.mean().item(),
        "std": g.std().item(),
        "min": g.min().item(),
        "max": g.max().item(),
        "abs_max": ag.max().item(),
        "l2": g.norm().item(),
        "p99_abs": torch.quantile(ag, 0.99).item(),
        "frac_gt_2": (ag > 2.0).float().mean().item(),
        "frac_lt_0p5": (g < 0.5).float().mean().item(),
        "hist_counts": counts,
        "argmax_channel": int(ag.argmax().item()),
    }
    if keep_vector:
        out["gamma_vector"] = g.tolist()
    return out


def tensor_health(weight: torch.Tensor, per_row: bool = False):
    """Scalar weight-health stats for a generic (large) parameter, computed then freed."""
    w = weight.detach().float()
    flat = w.flatten()
    out = {
        "shape": list(weight.shape),
        "frobenius": w.norm().item(),
        "mean_abs": flat.abs().mean().item(),
        "std": flat.std().item(),
        "abs_max": flat.abs().max().item(),
        "rms": flat.pow(2).mean().sqrt().item(),
    }
    if per_row and w.dim() >= 2:
        row = w.flatten(1).norm(dim=1)  # (out_features,)
        out["row_norm_mean"] = row.mean().item()
        out["row_norm_abs_max"] = row.abs().max().item()
        out["row_norm_cov"] = (row.std() / row.mean().abs()).item() if row.mean().abs() > 1e-12 else float("nan")
    return out


# ----------------------------------------------------------------------------
# Per-checkpoint analysis
# ----------------------------------------------------------------------------

# Non-QK norm gammas. All follow the apply_layernorm_1p convention (effective γ = 1+w
# when on) EXCEPT the Mamba GDN gated norm `mixer.norm.weight` (AlphaRMSNormGated), which
# is ALWAYS standard (effective γ = w) regardless of apply_layernorm_1p — applying 1+w to
# it in v1 would fabricate a false explosion signal. (mirrors check_layernorm_states.py's
# STANDARD_NORMS = {linear_attn.norm}.)
RMS_NORM_SUFFIXES = (
    "self_attention.linear_qgkv.layer_norm_weight",  # attn input norm (TENorm-fused)
    "mixer.in_proj.layer_norm_weight",               # mamba input norm (TENorm-fused)
    "pre_mlp_layernorm.weight",                      # MLP input norm
    "mixer.norm.weight",                             # mamba GDN gated norm — STANDARD always
)
GATED_NORM_SUFFIX = "mixer.norm.weight"
FINAL_NORM_KEYS = ("decoder.final_norm.weight", "decoder.final_layernorm.weight")


def _is_gated(key: str) -> bool:
    return key.endswith(GATED_NORM_SUFFIX)


def analyze_checkpoint(ckpt: str, moe_stride: int, skip_health: bool):
    cargs, iteration = load_common_args(ckpt)
    one_p = bool(getattr(cargs, "apply_layernorm_1p", False))
    tokens, samples, tok_src = consumed_tokens(cargs, iteration)
    layers = discover_layers(ckpt)
    all_keys = set(layers["all_keys"])
    attn = layers["attn"]

    # --- assemble norm keys present in this checkpoint ---
    qk_keys = []
    for n in attn:
        for which in ("q_layernorm", "k_layernorm"):
            k = f"decoder.layers.{n}.self_attention.{which}.weight"
            if k in all_keys:
                qk_keys.append(k)
    rms_keys = sorted(k for k in all_keys if any(k.endswith(s) for s in RMS_NORM_SUFFIXES))
    final_key = next((k for k in FINAL_NORM_KEYS if k in all_keys), None)

    load_keys = list(qk_keys) + list(rms_keys) + ([final_key] if final_key else [])
    tensors = load_named_tensors(ckpt, load_keys)

    rec = {
        "checkpoint": ckpt,
        "iteration": iteration,
        "consumed_samples": samples,
        "consumed_tokens": tokens,
        "tokens_source": tok_src,
        "apply_layernorm_1p": one_p,
        "seq_length": int(getattr(cargs, "seq_length", 0) or 0),
        "global_batch_size": int(getattr(cargs, "global_batch_size", 0) or 0),
        "num_experts": int(getattr(cargs, "num_experts", 0) or 0),
        "attn_layers": attn,
        "qk_gamma": {},     # key -> stats (full vector kept)
        "rms_gamma": {},    # key -> stats (final norm keeps vector)
        "weight_health": {},
    }

    for k in qk_keys:
        rec["qk_gamma"][k] = gamma_stats(tensors[k], one_p, keep_vector=True)
    for k in rms_keys:
        # gated GDN norm is standard regardless of apply_layernorm_1p
        zc = one_p and not _is_gated(k)
        rec["rms_gamma"][k] = gamma_stats(tensors[k], zc, keep_vector=False)
    if final_key:
        rec["rms_gamma"][final_key] = gamma_stats(tensors[final_key], one_p, keep_vector=True)

    if not skip_health:
        rec["weight_health"] = weight_health_sweep(ckpt, attn, all_keys, moe_stride)

    return rec


def weight_health_sweep(ckpt: str, attn, all_keys, moe_stride: int):
    """Load big params one family at a time, compute scalar stats, free immediately."""
    health = {}

    def add(keys, per_row=False):
        keys = [k for k in keys if k in all_keys]
        if not keys:
            return
        loaded = load_named_tensors(ckpt, keys)
        for k in keys:
            health[k] = tensor_health(loaded[k], per_row=per_row)
            loaded[k] = None  # free
        del loaded

    # attention projections (per-row catches single-channel blowups in Q|Gate|K|V)
    for n in attn:
        add([f"decoder.layers.{n}.self_attention.linear_qgkv.weight"], per_row=True)
        add([f"decoder.layers.{n}.self_attention.linear_proj.weight"], per_row=True)

    # mamba mixers (subsample: first few + last few mamba layers via regex over keys)
    mamba_layers = sorted({int(re.search(r"layers\.(\d+)\.", k).group(1))
                           for k in all_keys if ".mixer.in_proj.weight" in k})
    for n in mamba_layers:
        add([f"decoder.layers.{n}.mixer.in_proj.weight",
             f"decoder.layers.{n}.mixer.out_proj.weight"])
        for extra in ("A_log", "dt_bias", "conv1d.weight", "norm.weight"):
            add([f"decoder.layers.{n}.mixer.{extra}"])

    # MoE (expert FFN is the heaviest; subsample layers by stride)
    moe_layers = sorted({int(re.search(r"layers\.(\d+)\.", k).group(1))
                         for k in all_keys if ".mlp.router.weight" in k})
    for i, n in enumerate(moe_layers):
        add([f"decoder.layers.{n}.mlp.router.weight"])
        if i % max(1, moe_stride) == 0:
            add([f"decoder.layers.{n}.mlp.experts.experts.linear_fc1.weight"], per_row=True)
            add([f"decoder.layers.{n}.mlp.experts.experts.linear_fc2.weight"], per_row=True)

    # global embeddings
    add(["embedding.word_embeddings.weight"], per_row=False)
    add(["output_layer.weight"], per_row=False)
    return health


# ----------------------------------------------------------------------------
# Stdout summary
# ----------------------------------------------------------------------------

def print_run_summary(label: str, recs):
    print()
    print("=" * 100)
    print(f"  RUN: {label}   ({len(recs)} checkpoints)")
    print("=" * 100)
    hdr = (f"{'iter':>8}  {'tokens(B)':>10}  {'1p':>3}  "
           f"{'qk|γ|max':>9}  {'qk_layer':>9}  {'qk_ch':>6}  "
           f"{'rms|γ|max':>10}  {'final|γ|max':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        qk = r["qk_gamma"]
        if qk:
            kbest = max(qk, key=lambda k: qk[k]["abs_max"])
            qk_max = qk[kbest]["abs_max"]
            qk_layer = int(re.search(r"layers\.(\d+)\.", kbest).group(1))
            qk_ch = qk[kbest]["argmax_channel"]
        else:
            qk_max, qk_layer, qk_ch = float("nan"), -1, -1
        rms = r["rms_gamma"]
        non_final = [v["abs_max"] for k, v in rms.items() if "final" not in k]
        rms_max = max(non_final) if non_final else float("nan")
        final_max = next((v["abs_max"] for k, v in rms.items() if "final" in k), float("nan"))
        print(f"{r['iteration']:>8d}  {r['consumed_tokens']/1e9:>10.1f}  {int(r['apply_layernorm_1p']):>3d}  "
              f"{qk_max:>9.3f}  {qk_layer:>9d}  {qk_ch:>6d}  "
              f"{rms_max:>10.3f}  {final_max:>11.3f}")
    print("-" * len(hdr))
    print("  Reference: healthy effective |γ| ~1.97; the v1 Stage-1 blowup hit 11.9-12.9.")


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

GROUP_STYLE = {
    "v1": dict(color="#d62728", marker="o"),
    "v2-stage1": dict(color="#1f77b4", marker="s"),
    "v2-resume": dict(color="#2ca02c", marker="^"),
}


def write_plots(plot_dir: str, runs):
    """runs: list of dicts {label, group, recs}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    paths = []

    def style(group):
        return GROUP_STYLE.get(group, dict(color="#7f7f7f", marker="x"))

    # 1) max QK-norm |γ| (worst over all attn layers) vs tokens
    fig, ax = plt.subplots(figsize=(11, 6))
    for run in runs:
        xs, ys = [], []
        for r in run["recs"]:
            qk = r["qk_gamma"]
            if not qk:
                continue
            xs.append(r["consumed_tokens"] / 1e9)
            ys.append(max(v["abs_max"] for v in qk.values()))
        if xs:
            ax.plot(xs, ys, label=run["label"], **style(run["group"]))
    ax.axhline(2.0, ls=":", c="gray", lw=1, label="healthy ~2")
    ax.axhline(8.0, ls="--", c="orange", lw=1, label="explosion onset ~8")
    ax.set_xlabel("consumed tokens (B)")
    ax.set_ylabel("max QK-norm effective |γ| (worst attn layer)")
    ax.set_title("QK-LayerNorm gamma trajectory — v1 (explode) vs v2 (WD-controlled)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = os.path.join(plot_dir, "qk_gamma_absmax_vs_tokens.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    # 2) per-attn-layer QK |γ| spaghetti (last checkpoint of each run as bars not ideal;
    #    instead: each attn layer's |γ| vs tokens, one subplot per run-group)
    fig, ax = plt.subplots(figsize=(11, 6))
    for run in runs:
        st = style(run["group"])
        # collect per-layer series
        layer_series = {}
        for r in run["recs"]:
            for k, v in r["qk_gamma"].items():
                L = int(re.search(r"layers\.(\d+)\.", k).group(1))
                layer_series.setdefault(L, ([], []))
                layer_series[L][0].append(r["consumed_tokens"] / 1e9)
                layer_series[L][1].append(v["abs_max"])
        for L, (xs, ys) in sorted(layer_series.items()):
            alpha = 0.35 + 0.6 * (L / 47.0)  # deeper layers more opaque
            ax.plot(xs, ys, color=st["color"], alpha=alpha, lw=1.2)
        ax.plot([], [], color=st["color"], label=run["label"])  # legend proxy
    ax.set_xlabel("consumed tokens (B)")
    ax.set_ylabel("QK-norm effective |γ| (per attn layer; opacity = depth)")
    ax.set_title("Per-attention-layer QK gamma (deeper layers explode first in v1)")
    ax.axhline(8.0, ls="--", c="orange", lw=1)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    p = os.path.join(plot_dir, "qk_gamma_per_layer_vs_tokens.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    # 3) RMSNorm (final + non-final max) vs tokens
    fig, ax = plt.subplots(figsize=(11, 6))
    for run in runs:
        st = style(run["group"])
        xs, yf, yr = [], [], []
        for r in run["recs"]:
            rms = r["rms_gamma"]
            if not rms:
                continue
            xs.append(r["consumed_tokens"] / 1e9)
            yf.append(next((v["abs_max"] for k, v in rms.items() if "final" in k), float("nan")))
            nf = [v["abs_max"] for k, v in rms.items() if "final" not in k]
            yr.append(max(nf) if nf else float("nan"))
        if xs:
            ax.plot(xs, yf, color=st["color"], marker=st["marker"], ls="-", label=f"{run['label']} final")
            ax.plot(xs, yr, color=st["color"], marker=st["marker"], ls="--", alpha=0.6, label=f"{run['label']} other-RMS max")
    ax.set_xlabel("consumed tokens (B)")
    ax.set_ylabel("RMSNorm effective |γ| max")
    ax.set_title("Non-QK RMSNorm gamma — does it co-explode? (user observation)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    p = os.path.join(plot_dir, "rmsnorm_gamma_absmax_vs_tokens.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    # 4) per-channel γ of the deepest attn-layer q_norm: first vs last checkpoint
    fig, ax = plt.subplots(figsize=(11, 6))
    for run in runs:
        st = style(run["group"])
        recs = run["recs"]
        if not recs:
            continue
        def deepest_qnorm(r):
            cands = {k: v for k, v in r["qk_gamma"].items() if "q_layernorm" in k}
            if not cands:
                return None
            kk = max(cands, key=lambda k: int(re.search(r"layers\.(\d+)\.", k).group(1)))
            return cands[kk].get("gamma_vector")
        last = deepest_qnorm(recs[-1])
        if last:
            ax.plot(sorted(last, reverse=True), color=st["color"], label=f"{run['label']} (final ckpt)")
    ax.set_xlabel("channel rank (sorted desc)")
    ax.set_ylabel("effective γ of deepest q_norm")
    ax.set_title("Per-channel QK gamma at final checkpoint (which channels blow up)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    p = os.path.join(plot_dir, "qk_gamma_per_channel_final.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    return paths


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Weight-trajectory analysis across Alpha checkpoints (v1 vs v2).")
    p.add_argument("--runs", nargs="+", required=True,
                   help="Run dirs (each containing checkpoints/iter_*) or checkpoints dirs.")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Optional labels matching --runs (default auto-derived).")
    p.add_argument("--output", default=None, help="Write combined JSON report here.")
    p.add_argument("--csv", default=None, help="Write a flat per-checkpoint CSV here.")
    p.add_argument("--plot", default=None, help="Write PNG plots to this directory.")
    p.add_argument("--moe-layer-stride", type=int, default=4,
                   help="Compute heavy expert-FFN norms every Nth MoE layer (default 4).")
    p.add_argument("--skip-weight-health", action="store_true",
                   help="Only gamma analysis (skip the heavy weight-norm sweep).")
    p.add_argument("--max-checkpoints", type=int, default=0,
                   help="If >0, analyze at most this many (evenly spaced) checkpoints per run.")
    return p.parse_args()


def subsample(items, k):
    if k <= 0 or len(items) <= k:
        return items
    if k == 1:
        return [items[-1]]  # latest checkpoint
    idx = [round(i * (len(items) - 1) / (k - 1)) for i in range(k)]
    return [items[i] for i in sorted(set(idx))]


def main():
    args = parse_args()
    runs_out = []
    for i, run_dir in enumerate(args.runs):
        label = args.labels[i] if args.labels and i < len(args.labels) else None
        group = derive_group(run_dir)
        label = label or group
        ckpts = subsample(find_checkpoints(run_dir), args.max_checkpoints)
        print(f"\n### {label}  ({group})  — {len(ckpts)} checkpoints under {run_dir}")
        recs = []
        for ckpt in ckpts:
            print(f"  analyzing {os.path.basename(ckpt)} ...", flush=True)
            recs.append(analyze_checkpoint(ckpt, args.moe_layer_stride, args.skip_weight_health))
        runs_out.append({"run_dir": run_dir, "label": label, "group": group, "recs": recs})
        print_run_summary(label, recs)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"runs": runs_out}, f)
        print(f"\nWrote JSON report: {args.output}")

    if args.csv:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "group", "iteration", "consumed_tokens", "apply_1p",
                        "qk_absmax", "qk_layer", "rms_nonfinal_absmax", "final_absmax"])
            for run in runs_out:
                for r in run["recs"]:
                    qk = r["qk_gamma"]
                    kbest = max(qk, key=lambda k: qk[k]["abs_max"]) if qk else None
                    rms = r["rms_gamma"]
                    nf = [v["abs_max"] for k, v in rms.items() if "final" not in k]
                    w.writerow([run["label"], run["group"], r["iteration"], r["consumed_tokens"],
                                int(r["apply_layernorm_1p"]),
                                f"{qk[kbest]['abs_max']:.4f}" if kbest else "",
                                int(re.search(r'layers\.(\d+)\.', kbest).group(1)) if kbest else "",
                                f"{max(nf):.4f}" if nf else "",
                                next((f"{v['abs_max']:.4f}" for k, v in rms.items() if "final" in k), "")])
        print(f"Wrote CSV: {args.csv}")

    if args.plot:
        paths = write_plots(args.plot, runs_out)
        print(f"Wrote {len(paths)} plots to {args.plot}:")
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
