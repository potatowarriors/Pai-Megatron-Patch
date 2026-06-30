#!/usr/bin/env python3
"""
MoE expert load-balance + training-tendency analysis for Alpha checkpoints.

Reads a Megatron **torch_dist** checkpoint directly (the raw `iter_NNNNNNN/`
directory with `.distcp` shards + `common.pt`) — NO HuggingFace conversion, NO
forward pass, NO training launch. Everything below is computed from static
weights / buffers, so it is cheap to run at every save interval.

Primary question: are the MoE experts being trained EVENLY (load-balanced)?

The strongest static proxy is the per-expert **router.expert_bias** buffer used by
DeepSeek-V3-style aux-loss-free routing. Each step the bias is nudged by
`+update_rate` for under-used experts and `-update_rate` for over-used ones
(see `get_updated_expert_bias` in megatron/core/transformer/moe/moe_utils.py).
So at iteration N the bias lives in an envelope of `±(N * update_rate)`, and its
spread / one-sidedness directly measures chronic imbalance. We corroborate this
with per-expert router-row norms and per-expert FFN (grouped-GEMM) Frobenius
norms — under-used experts tend to stay closer to their init norm.

A "broader tendencies" section (same checkpoint, no extra cost) reports QK-norm
gamma health (the known Stage-1 gamma-explosion failure mode) and a depthwise
weight-norm sanity sweep.

NOTE: config (num_experts, update_rate, iteration, ...) is read from `common.pt`
at runtime and never hardcoded — the alpha CLAUDE.md drifted from the live config
(it says 184 experts / `none` balancing; the actual run uses 192 / seq_aux_loss).

Usage:
  export PYTHONPATH=<repo>:<repo>/backends/megatron/Megatron-LM-251125:$PYTHONPATH
  python check_expert_balance.py \\
    --checkpoint outputs/alpha_baseline_48L_stage1_.../checkpoints/iter_0010000 \\
    --output /tmp/expert_balance_10k.json \\
    --plot /tmp/expert_balance_plots
"""

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import torch


# ----------------------------------------------------------------------------
# Checkpoint access (torch_dist)
# ----------------------------------------------------------------------------

def _ensure_process_group():
    """torch_dist `load` of tensor *data* needs a process group; a single-process
    gloo group is enough for standalone analysis. `load_tensors_metadata` does
    not need it, but `load` does."""
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29577")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)


def load_common_args(ckpt_dir: str):
    """Read the argparse.Namespace stored in common.pt (training config)."""
    common = torch.load(os.path.join(ckpt_dir, "common.pt"),
                        map_location="cpu", weights_only=False)
    return common.get("args"), int(common.get("iteration", 0))


def load_named_tensors(ckpt_dir: str, keys: List[str]) -> Dict[str, torch.Tensor]:
    """Load a subset of tensors by key from a torch_dist checkpoint, consolidating
    EP/TP shards transparently. Returns key -> CPU tensor."""
    from megatron.core.dist_checkpointing.serialization import (
        load_tensors_metadata, load,
    )
    meta = load_tensors_metadata(ckpt_dir)
    missing = [k for k in keys if k not in meta]
    if missing:
        raise KeyError(f"{len(missing)} keys absent from checkpoint, e.g. {missing[:3]}")
    template = {k: meta[k] for k in keys}
    _ensure_process_group()
    loaded = load(template, ckpt_dir, validate_access_integrity=False)
    return {k: loaded[k].cpu() for k in keys}


def discover_layers(ckpt_dir: str):
    """Enumerate checkpoint keys and group layer indices by role (no hardcoding)."""
    from megatron.core.dist_checkpointing.serialization import load_tensors_metadata
    keys = list(load_tensors_metadata(ckpt_dir).keys())

    def idxs(pat):
        out = set()
        for k in keys:
            if re.search(pat, k):
                m = re.search(r"layers\.(\d+)\.", k)
                if m:
                    out.add(int(m.group(1)))
        return sorted(out)

    return {
        "all_keys": keys,
        "moe": idxs(r"\.mlp\.router\.expert_bias$"),
        "attn": idxs(r"self_attention\.q_layernorm\.weight$"),
    }


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

@dataclass
class ExpertLayerStats:
    """Per-MoE-layer expert load-balance statistics (over the routed experts).

    NOTE: top-k routing is invariant to a uniform shift of expert_bias, so all
    load-balance metrics use the MEAN-CENTERED bias (deviation from the layer
    mean). `bias_dc` is the raw per-layer mean, kept only as an FYI."""
    layer: int
    bias_dc: float               # raw mean(bias) — routing-irrelevant, FYI only
    bias_std: float              # std(bias) == std of centered bias
    spread_score: float          # std(bias)/envelope; 0 = perfectly balanced
    max_underuse_steps: float    # (max-mean)/rate ~ net #steps the worst expert was under-used
    max_overuse_steps: float     # (mean-min)/rate ~ net #steps the worst expert was over-used
    n_cold: int                  # experts > +sigma*std from mean (relatively under-used)
    n_hot: int                   # experts < -sigma*std from mean (relatively over-used)
    # router-row norm (per-expert L2 of router.weight rows)
    router_norm_cov: float
    # expert FFN per-expert Frobenius norm (grouped GEMM, experts-first)
    ffn_norm_cov: float
    n_low_norm: int              # FFN-norm outliers below mean-2*std (bottom-tail, ~expected 2%)
    # cross-signal corroboration (meaningful only when spread is non-trivial)
    corr_bias_ffn: float         # pearson(centered bias, ffn_norm)
    corr_bias_router: float


@dataclass
class AttnGammaStats:
    layer: int
    q_mean: float
    q_abs_max: float
    k_mean: float
    k_abs_max: float


@dataclass
class Report:
    checkpoint: str
    iteration: int
    num_experts: int
    moe_topk: int
    bias_update_rate: float
    envelope: float
    outlier_sigma: float
    moe_layers: List[int]
    attn_layers: List[int]
    per_layer: List[dict] = field(default_factory=list)
    attn_gamma: List[dict] = field(default_factory=list)
    # per-expert-index bias averaged across all MoE layers (length num_experts)
    global_expert_bias_profile: List[float] = field(default_factory=list)
    globals: dict = field(default_factory=dict)


def _cov(x: torch.Tensor) -> float:
    """Coefficient of variation (std/|mean|); guarded against ~0 mean."""
    m = x.mean().item()
    s = x.std().item()
    return s / abs(m) if abs(m) > 1e-12 else float("nan")


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return (a @ b / denom).item() if denom > 1e-12 else float("nan")


def per_expert_ffn_norm(fc1: torch.Tensor, fc2: torch.Tensor) -> torch.Tensor:
    """Combined per-expert Frobenius norm. fc1/fc2 are experts-first
    (num_experts, ...); norm over all non-expert dims."""
    n1 = fc1.float().flatten(1).norm(dim=1)
    n2 = fc2.float().flatten(1).norm(dim=1)
    return torch.sqrt(n1 ** 2 + n2 ** 2)


def compute_expert_stats(layer, bias, router_w, ffn_norm,
                         envelope, rate, sigma) -> ExpertLayerStats:
    bias = bias.float()
    dc = bias.mean().item()
    centered = bias - dc                          # routing-relevant deviation
    std = bias.std().item()
    router_norm = router_w.float().norm(dim=1)    # (num_experts,)
    return ExpertLayerStats(
        layer=layer,
        bias_dc=dc,
        bias_std=std,
        spread_score=(std / envelope) if envelope > 0 else float("nan"),
        max_underuse_steps=((bias.max().item() - dc) / rate) if rate > 0 else float("nan"),
        max_overuse_steps=((dc - bias.min().item()) / rate) if rate > 0 else float("nan"),
        n_cold=int((centered > sigma * std).sum().item()),
        n_hot=int((centered < -sigma * std).sum().item()),
        router_norm_cov=_cov(router_norm),
        ffn_norm_cov=_cov(ffn_norm),
        n_low_norm=int((ffn_norm < ffn_norm.mean() - 2 * ffn_norm.std()).sum().item()),
        corr_bias_ffn=_pearson(centered, ffn_norm),
        corr_bias_router=_pearson(centered, router_norm),
    )


# ----------------------------------------------------------------------------
# Output: stdout tables + verdict
# ----------------------------------------------------------------------------

def section(title: str):
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)


def print_expert_table(stats: List[ExpertLayerStats]):
    section("Section A — Per-MoE-layer expert load balance (mean-centered, over routed experts)")
    hdr = (f"{'Layer':>5}  {'bias_std':>9}  {'spread':>7}  {'under_st':>9}  {'over_st':>8}  "
           f"{'cold':>5}  {'hot':>5}  {'router_cov':>10}  {'ffn_cov':>8}  {'ffnOut':>6}  {'r(b,ffn)':>9}")
    print(hdr)
    print("-" * len(hdr))
    for s in stats:
        print(f"{s.layer:>5d}  {s.bias_std:>9.4f}  {s.spread_score:>7.4f}  "
              f"{s.max_underuse_steps:>9.0f}  {s.max_overuse_steps:>8.0f}  "
              f"{s.n_cold:>5d}  {s.n_hot:>5d}  "
              f"{s.router_norm_cov:>10.4f}  {s.ffn_norm_cov:>8.4f}  {s.n_low_norm:>6d}  "
              f"{s.corr_bias_ffn:>+9.3f}")
    print("-" * len(hdr))
    print("  spread = std(bias)/envelope (0=perfectly balanced; metrics use mean-centered bias)")
    print("  under_st/over_st = net #steps the worst under-/over-used expert drifted (max ~= envelope/rate)")
    print("  cold/hot = #experts beyond +-sigma*std from layer mean (relative outliers)")
    print("  ffnOut = #experts with FFN norm < mean-2sd (bottom tail; ~2% expected even if healthy)")
    print("  r(b,ffn) = pearson(centered bias, ffn norm); only meaningful when spread is non-trivial")


def print_attn_gamma(gammas: List[AttnGammaStats]):
    section("Section B — QK-norm gamma health (attention layers; known explosion mode)")
    hdr = f"{'Layer':>5}  {'q_mean':>8}  {'q_abs_max':>10}  {'k_mean':>8}  {'k_abs_max':>10}  {'flag':<14}"
    print(hdr)
    print("-" * len(hdr))
    for g in gammas:
        worst = max(g.q_abs_max, g.k_abs_max)
        flag = "[EXPLODING]" if worst > 8.0 else ("[elevated]" if worst > 4.0 else "")
        print(f"{g.layer:>5d}  {g.q_mean:>8.3f}  {g.q_abs_max:>10.3f}  "
              f"{g.k_mean:>8.3f}  {g.k_abs_max:>10.3f}  {flag:<14}")
    print("-" * len(hdr))
    print("  Reference: healthy gamma |max| ~1.97; the Stage-1 blowup bug hit 11.9-12.9.")


def print_verdict(report: Report, stats: List[ExpertLayerStats],
                  gammas: List[AttnGammaStats]):
    section("Section C — Interpretation / verdict")

    n = len(stats)
    mean_spread = sum(s.spread_score for s in stats) / n
    max_spread = max(s.spread_score for s in stats)
    total_cold = sum(s.n_cold for s in stats)
    total_hot = sum(s.n_hot for s in stats)
    total_low = sum(s.n_low_norm for s in stats)
    mean_corr = sum(s.corr_bias_ffn for s in stats
                    if not math.isnan(s.corr_bias_ffn)) / n

    print(f"  MoE layers analyzed:        {n}  (experts/layer = {report.num_experts}, top-{report.moe_topk})")
    print(f"  iteration / bias envelope:  {report.iteration} / +-{report.envelope:.2f}")
    print(f"  mean spread_score:          {mean_spread:.3f}   (0 = perfectly balanced)")
    print(f"  worst-layer spread_score:   {max_spread:.3f}")
    print(f"  chronically cold experts:   {total_cold}  (over all {n} layers)")
    print(f"  chronically hot experts:    {total_hot}")
    print(f"  FFN-norm bottom-tail outliers: {total_low}  (~2% expected even if healthy)")
    corr_note = "" if max_spread >= 0.05 else "  (spread negligible -> corr is ~noise, ignore sign)"
    print(f"  mean corr(bias, ffn_norm):  {mean_corr:+.3f}{corr_note}")
    print()

    # Verdict thresholds: spread_score is std(bias) as a fraction of the ±envelope.
    # spread 0.05 ~ std 0.5 ~ worst expert net under-used ~500/10000 steps.
    if mean_spread < 0.05 and max_spread < 0.10:
        verdict = "BALANCED"
        msg = ("Expert usage is well spread. The bias is making only small corrections "
               "around each layer mean — no expert is being chronically starved.")
    elif mean_spread < 0.15:
        verdict = "MILD IMBALANCE"
        msg = ("Some layers show a widening bias spread. Often normal early in training; "
               "re-check at the next checkpoints that spread is not growing monotonically.")
    else:
        verdict = "SIGNIFICANT IMBALANCE"
        msg = ("Bias spread is large — a subset of experts is being persistently under- or "
               "over-used. Inspect the per-layer table and the global expert profile; "
               "consider router-init / aux-loss-coeff / group-routing settings.")
    print(f"  >>> EXPERT BALANCE: {verdict}")
    print(f"      {msg}")

    worst_gamma = max((max(g.q_abs_max, g.k_abs_max) for g in gammas), default=0.0)
    if worst_gamma > 8.0:
        print(f"  >>> QK-NORM: WARNING — gamma |max| {worst_gamma:.2f} approaching the known "
              f"explosion regime. Verify apply_wd_to_qk_layernorm is active.")
    elif worst_gamma > 4.0:
        print(f"  >>> QK-NORM: elevated (|max| {worst_gamma:.2f}) — watch over next checkpoints.")
    else:
        print(f"  >>> QK-NORM: healthy (|max| {worst_gamma:.2f}).")


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def write_plots(plot_dir: str, moe_layers, bias_mat, ffn_mat, expert_profile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    # 1. expert_bias heatmap [layers x experts]
    fig, ax = plt.subplots(figsize=(14, 6))
    vmax = max(abs(bias_mat.min()), abs(bias_mat.max()))
    im = ax.imshow(bias_mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(moe_layers)))
    ax.set_yticklabels(moe_layers)
    ax.set_xlabel("expert index")
    ax.set_ylabel("MoE layer")
    ax.set_title("router.expert_bias (mean-centered per layer)  (red = under-used / cold)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    p1 = os.path.join(plot_dir, "expert_bias_heatmap.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)

    # 2. per-expert FFN norm heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(ffn_mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(moe_layers)))
    ax.set_yticklabels(moe_layers)
    ax.set_xlabel("expert index")
    ax.set_ylabel("MoE layer")
    ax.set_title("per-expert FFN Frobenius norm (dark = near-init / under-trained)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    p2 = os.path.join(plot_dir, "expert_ffn_norm_heatmap.png")
    fig.savefig(p2, dpi=120)
    plt.close(fig)

    # 3. global per-expert-index bias profile (averaged across layers)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(expert_profile)), expert_profile, width=1.0)
    ax.set_xlabel("expert index")
    ax.set_ylabel("mean expert_bias across MoE layers")
    ax.set_title("Globally cold (positive) / hot (negative) expert slots")
    fig.tight_layout()
    p3 = os.path.join(plot_dir, "global_expert_bias_profile.png")
    fig.savefig(p3, dpi=120)
    plt.close(fig)

    return [p1, p2, p3]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="MoE expert load-balance + tendency analysis for an Alpha torch_dist checkpoint")
    p.add_argument("--checkpoint", required=True,
                   help="Path to an iter_NNNNNNN/ torch_dist checkpoint directory")
    p.add_argument("--outlier-sigma", type=float, default=3.0,
                   help="experts beyond +-sigma*std from the layer mean are flagged "
                        "cold/hot (default 3.0)")
    p.add_argument("--top-k", type=int, default=20,
                   help="how many extreme global expert slots to print (default 20)")
    p.add_argument("--output", default=None, help="write JSON report to this path")
    p.add_argument("--plot", default=None, help="write PNG heatmaps to this directory")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    ckpt = args.checkpoint.rstrip("/")

    print("=" * 100)
    print("  Alpha MoE expert load-balance + tendency analysis")
    print("=" * 100)
    print(f"  checkpoint: {ckpt}")

    if not os.path.isfile(os.path.join(ckpt, "common.pt")):
        print(f"ERROR: no common.pt under {ckpt} — is this a torch_dist iter_* directory?")
        sys.exit(1)

    # --- config from common.pt (never hardcoded) ---
    cargs, iteration = load_common_args(ckpt)
    num_experts = int(getattr(cargs, "num_experts", 0))
    moe_topk = int(getattr(cargs, "moe_router_topk", 0))
    update_rate = float(getattr(cargs, "moe_router_bias_update_rate", 0.0))
    envelope = iteration * update_rate
    if envelope <= 0:
        print("WARN: envelope (iteration*update_rate) <= 0; spread metric disabled.")
        envelope = float("nan")
    print(f"  config (from common.pt): num_experts={num_experts}, top-{moe_topk}, "
          f"bias_update_rate={update_rate}, iteration={iteration}")
    print(f"  bias envelope = +-{envelope:.3f}; outlier flag = +-{args.outlier_sigma}*std from layer mean")

    # --- discover layers + load tensors ---
    layers = discover_layers(ckpt)
    moe_layers = layers["moe"]
    attn_layers = layers["attn"]
    print(f"  MoE layers ({len(moe_layers)}): {moe_layers}")
    print(f"  Attention layers ({len(attn_layers)}): {attn_layers}")

    keys = []
    for i in moe_layers:
        keys += [
            f"decoder.layers.{i}.mlp.router.expert_bias",
            f"decoder.layers.{i}.mlp.router.weight",
            f"decoder.layers.{i}.mlp.experts.experts.linear_fc1.weight",
            f"decoder.layers.{i}.mlp.experts.experts.linear_fc2.weight",
        ]
    for i in attn_layers:
        keys += [
            f"decoder.layers.{i}.self_attention.q_layernorm.weight",
            f"decoder.layers.{i}.self_attention.k_layernorm.weight",
        ]
    print(f"  loading {len(keys)} tensors from torch_dist shards ...")
    tensors = load_named_tensors(ckpt, keys)

    # --- compute per-layer expert stats ---
    stats: List[ExpertLayerStats] = []
    bias_mat = []
    ffn_mat = []
    for i in moe_layers:
        bias = tensors[f"decoder.layers.{i}.mlp.router.expert_bias"]
        router_w = tensors[f"decoder.layers.{i}.mlp.router.weight"]
        fc1 = tensors[f"decoder.layers.{i}.mlp.experts.experts.linear_fc1.weight"]
        fc2 = tensors[f"decoder.layers.{i}.mlp.experts.experts.linear_fc2.weight"]
        ffn_norm = per_expert_ffn_norm(fc1, fc2)
        stats.append(compute_expert_stats(i, bias, router_w, ffn_norm,
                                          envelope, update_rate, args.outlier_sigma))
        # store MEAN-CENTERED bias (routing-relevant) for the global profile/heatmap
        b = bias.float()
        bias_mat.append((b - b.mean()).tolist())
        ffn_mat.append(ffn_norm.tolist())

    # --- attention gamma ---
    gammas: List[AttnGammaStats] = []
    for i in attn_layers:
        q = tensors[f"decoder.layers.{i}.self_attention.q_layernorm.weight"].float()
        k = tensors[f"decoder.layers.{i}.self_attention.k_layernorm.weight"].float()
        gammas.append(AttnGammaStats(
            layer=i, q_mean=q.mean().item(), q_abs_max=q.abs().max().item(),
            k_mean=k.mean().item(), k_abs_max=k.abs().max().item()))

    # --- global per-expert-index bias profile (avg across MoE layers) ---
    bias_tensor = torch.tensor(bias_mat)            # (n_moe, num_experts)
    expert_profile = bias_tensor.mean(dim=0)        # (num_experts,)

    # --- print ---
    print_expert_table(stats)
    print_attn_gamma(gammas)

    # top-k globally cold / hot expert slots (on mean-CENTERED bias averaged across layers:
    # reveals expert SLOTS that are systematically under/over-used regardless of layer DC offset)
    section(f"Section B' — Top {args.top_k} systematically cold / hot expert slots "
            f"(mean-centered bias avg across {len(moe_layers)} layers)")
    order = torch.argsort(expert_profile, descending=True)
    print("  COLD (systematically under-used): " +
          ", ".join(f"#{int(e)}({expert_profile[e]:+.3f})" for e in order[:args.top_k]))
    print("  HOT  (systematically over-used):  " +
          ", ".join(f"#{int(e)}({expert_profile[e]:+.3f})" for e in order[-args.top_k:].flip(0)))
    print("  (values are deviation from layer mean in bias units; near 0 => no global slot preference)")

    print_verdict(Report("", iteration, num_experts, moe_topk, update_rate,
                         envelope, args.outlier_sigma, moe_layers, attn_layers),
                  stats, gammas)
    print()

    # --- JSON ---
    if args.output:
        report = Report(
            checkpoint=ckpt, iteration=iteration, num_experts=num_experts,
            moe_topk=moe_topk, bias_update_rate=update_rate, envelope=envelope,
            outlier_sigma=args.outlier_sigma,
            moe_layers=moe_layers, attn_layers=attn_layers,
            per_layer=[asdict(s) for s in stats],
            attn_gamma=[asdict(g) for g in gammas],
            global_expert_bias_profile=expert_profile.tolist(),
            globals={
                "mean_spread_score": sum(s.spread_score for s in stats) / len(stats),
                "max_spread_score": max(s.spread_score for s in stats),
                "total_cold": sum(s.n_cold for s in stats),
                "total_hot": sum(s.n_hot for s in stats),
                "total_low_norm": sum(s.n_low_norm for s in stats),
            },
        )
        with open(args.output, "w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"  JSON report written: {args.output}")

    # --- plots ---
    if args.plot:
        import numpy as np
        paths = write_plots(args.plot, moe_layers,
                            np.array(bias_mat), np.array(ffn_mat),
                            expert_profile.tolist())
        print(f"  plots written: {', '.join(paths)}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
