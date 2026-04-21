#!/usr/bin/env python3
"""
LayerNorm state drift analysis for Alpha model checkpoints.

Compares LayerNorm gamma parameters between two HuggingFace checkpoints
(e.g., pre-Stage2 vs. post-spike) to detect systematic drift that could
indicate a Muon-LayerNorm velocity mismatch during stage transitions.

Background: Alpha trains most parameters with Muon while LayerNorm 1D params
use Adam. If Muon's update scale changes abruptly (e.g., Stage2 0.2x),
LayerNorm gammas remain tuned for the old Muon regime and must re-adapt.
This script quantifies that drift per layer / per norm type.

Two LayerNorm classes exist in modeling_alpha.py and are handled differently:
  - AlphaRMSNorm (zero-centered gamma): effective = 1 + w
  - AlphaRMSNormGated (standard gamma): effective = w

Usage:
  python check_layernorm_states.py \\
    --model-path outputs/.../hfmodel_0480000 \\
    --baseline-path outputs/.../hfmodel_0440000
"""

import argparse
import sys
from dataclasses import dataclass

import torch


# Norm name → whether the module uses zero-centered gamma (AlphaRMSNorm).
# AlphaRMSNormGated (linear_attn.norm) uses standard gamma.
ZERO_CENTERED_NORMS = {
    "input_layernorm",
    "post_attention_layernorm",
    "q_norm",
    "k_norm",
    "model.norm",
}
STANDARD_NORMS = {
    "linear_attn.norm",
}


@dataclass
class GammaStats:
    """Summary statistics for one norm's effective gamma tensor."""
    mean: float
    std: float
    max: float
    min: float
    abs_max: float
    frac_gt_2: float       # fraction of elements with effective gamma > 2.0
    frac_lt_0p5: float     # fraction with effective gamma < 0.5
    weight_tensor: torch.Tensor  # raw stored weight (not effective) — used for L2 diff


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare LayerNorm gamma states between two Alpha HF checkpoints"
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to HF checkpoint to analyze (e.g., post-Stage2 spike)",
    )
    parser.add_argument(
        "--baseline-path", required=True,
        help="Path to baseline HF checkpoint for comparison (e.g., pre-Stage2)",
    )
    parser.add_argument(
        "--top-k-drift", type=int, default=20,
        help="Print top-K norms by relative L2 drift (default: 20)",
    )
    return parser.parse_args()


def load_model_cpu(path: str):
    """Load an Alpha HF checkpoint onto CPU (no forward pass needed)."""
    # Guard against SGLang native checkpoints (incompatible with AutoModel)
    if path.rstrip("/").endswith("_sglang_native"):
        orig = path.rstrip("/").rsplit("_sglang_native", 1)[0]
        print(f"ERROR: SGLang native checkpoints cannot be loaded with AutoModel.")
        print(f"  Use the original HF checkpoint instead: {orig}")
        sys.exit(1)

    from transformers import AutoModelForCausalLM

    print(f"Loading: {path}")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    return model


def collect_norm_weights(model):
    """
    Walk the model and collect raw .weight tensors for every LayerNorm.

    Returns a dict keyed by (layer_idx, layer_type, norm_name) → weight tensor.
    layer_idx == -1 is reserved for the final model.norm.
    """
    config = model.config
    layer_types = config.layer_types
    records = {}

    for i in range(config.num_hidden_layers):
        layer = model.model.layers[i]
        ltype = layer_types[i]

        records[(i, ltype, "input_layernorm")] = layer.input_layernorm.weight.detach()
        records[(i, ltype, "post_attention_layernorm")] = (
            layer.post_attention_layernorm.weight.detach()
        )

        if ltype == "full_attention":
            records[(i, ltype, "q_norm")] = layer.self_attn.q_norm.weight.detach()
            records[(i, ltype, "k_norm")] = layer.self_attn.k_norm.weight.detach()
        elif ltype == "linear_attention":
            records[(i, ltype, "linear_attn.norm")] = (
                layer.linear_attn.norm.weight.detach()
            )

    records[(-1, "final", "model.norm")] = model.model.norm.weight.detach()
    return records


def compute_gamma_stats(weight: torch.Tensor, norm_name: str) -> GammaStats:
    """Compute effective-gamma statistics, accounting for zero-centered init."""
    w = weight.float().cpu()
    if norm_name in ZERO_CENTERED_NORMS:
        gamma = 1.0 + w
    else:
        gamma = w

    numel = gamma.numel()
    return GammaStats(
        mean=gamma.mean().item(),
        std=gamma.std().item(),
        max=gamma.max().item(),
        min=gamma.min().item(),
        abs_max=gamma.abs().max().item(),
        frac_gt_2=(gamma > 2.0).sum().item() / numel,
        frac_lt_0p5=(gamma < 0.5).sum().item() / numel,
        weight_tensor=w,
    )


def build_all_stats(model):
    """Collect weights and compute stats for every norm in the model."""
    weights = collect_norm_weights(model)
    return {key: compute_gamma_stats(w, key[2]) for key, w in weights.items()}


def section_header(title: str):
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)


def print_per_layer_table(baseline_stats, current_stats):
    """Section A: effective-gamma mean + L2 drift, one row per norm."""
    section_header(
        "Section A — Per-norm summary (effective gamma = 1+w for AlphaRMSNorm, w for AlphaRMSNormGated)"
    )
    header = (
        f"{'Layer':>5}  {'Type':<18}  {'Norm':<26}  "
        f"{'μ_base':>8}  {'μ_cur':>8}  {'Δμ':>8}  "
        f"{'|γ|max_b':>9}  {'|γ|max_c':>9}  "
        f"{'L2_drift':>10}  {'rel_drift':>10}"
    )
    print(header)
    print("-" * len(header))

    keys = sorted(
        current_stats.keys(),
        key=lambda k: (k[0] if k[0] >= 0 else 10_000, k[2]),
    )
    rows = []

    for key in keys:
        layer_idx, ltype, nname = key
        cur = current_stats[key]
        base = baseline_stats.get(key)
        if base is None:
            print(f"  WARN: baseline missing key {key}; skipping")
            continue

        l2_drift = (cur.weight_tensor - base.weight_tensor).norm().item()
        base_norm = base.weight_tensor.norm().item()
        # For zero-centered (all-zero init) the baseline can have tiny norm
        # early in training; fall back to ||current|| to avoid div-by-zero.
        denom = base_norm if base_norm > 1e-8 else max(cur.weight_tensor.norm().item(), 1e-8)
        rel_drift = l2_drift / denom
        dmu = cur.mean - base.mean

        layer_label = f"{layer_idx:>5d}" if layer_idx >= 0 else f"{'final':>5}"
        print(
            f"{layer_label}  {ltype:<18}  {nname:<26}  "
            f"{base.mean:>8.3f}  {cur.mean:>8.3f}  {dmu:>+8.3f}  "
            f"{base.abs_max:>9.3f}  {cur.abs_max:>9.3f}  "
            f"{l2_drift:>10.4f}  {rel_drift * 100:>9.2f}%"
        )
        rows.append((key, rel_drift, dmu, l2_drift))

    return rows


def print_top_k_drift(rows, k: int):
    """Section B: top-K norms by relative L2 drift."""
    section_header(f"Section B — Top {k} norms by relative L2 drift")
    print(
        f"{'Rank':>4}  {'Layer':>5}  {'Type':<18}  {'Norm':<26}  "
        f"{'rel_drift':>10}  {'Δμ':>8}  {'L2_drift':>10}  {'Tag':<15}"
    )
    print("-" * 100)

    sorted_rows = sorted(rows, key=lambda r: r[1], reverse=True)[:k]
    for rank, (key, rel_drift, dmu, l2_drift) in enumerate(sorted_rows, 1):
        layer_idx, ltype, nname = key
        if rel_drift > 0.10:
            tag = "[LARGE DRIFT]"
        elif rel_drift > 0.05:
            tag = "[MODERATE]"
        else:
            tag = ""
        layer_label = f"{layer_idx:>5d}" if layer_idx >= 0 else f"{'final':>5}"
        print(
            f"{rank:>4d}  {layer_label}  {ltype:<18}  {nname:<26}  "
            f"{rel_drift * 100:>9.2f}%  {dmu:>+8.3f}  {l2_drift:>10.4f}  {tag:<15}"
        )


def print_diagnosis(rows):
    """
    Section C: Heuristic interpretation of drift distribution.

    Hypothesis under test: Stage2's 0.2x Muon scaling leaves Adam-trained
    LayerNorm gammas mis-adapted, so drift should appear GLOBALLY across
    many layers / norm types, not just at the known q_norm/k_norm hotspots.
    """
    section_header("Section C — Interpretation")

    total = len(rows)
    qk_rows = [r for r in rows if r[0][2] in ("q_norm", "k_norm")]
    non_qk_rows = [r for r in rows if r[0][2] not in ("q_norm", "k_norm")]

    large_drift = [r for r in rows if r[1] > 0.10]
    moderate_drift = [r for r in rows if 0.05 < r[1] <= 0.10]
    small_drift = [r for r in rows if r[1] <= 0.01]

    qk_large = [r for r in qk_rows if r[1] > 0.10]
    non_qk_large = [r for r in non_qk_rows if r[1] > 0.10]
    non_qk_moderate = [r for r in non_qk_rows if r[1] > 0.05]

    print(f"  Total norms analyzed:   {total}")
    print(f"  Large drift (>10%):     {len(large_drift)}")
    print(f"    ↳ q_norm/k_norm:      {len(qk_large)}")
    print(f"    ↳ other norms:        {len(non_qk_large)}")
    print(f"  Moderate drift (5-10%): {len(moderate_drift)}")
    print(f"  Small drift (<1%):      {len(small_drift)} / {total}")
    print()

    if len(small_drift) / total > 0.9 and len(non_qk_large) == 0:
        print("  → LayerNorm drift is NEGLIGIBLE across the model.")
        print("    The Muon-LayerNorm velocity-mismatch hypothesis is NOT supported")
        print("    by parameter drift evidence. Spike is likely due to a different")
        print("    mechanism (data regime shift, LR schedule, optimizer state reset).")
    elif non_qk_large or non_qk_moderate:
        print("  → Drift is WIDESPREAD beyond q_norm/k_norm.")
        print(f"    {len(non_qk_moderate)} non-QK norms drifted >5%, {len(non_qk_large)} drifted >10%.")
        print("    This is consistent with the Stage2 Muon-scaling hypothesis:")
        print("    LayerNorm gammas are re-adapting to the new equilibrium.")
    elif qk_large and not non_qk_large:
        print("  → Drift is LOCALIZED to q_norm/k_norm only.")
        print("    This matches the known QK LayerNorm gamma blowup bug")
        print("    (see CLAUDE.md). Global Muon-mismatch hypothesis is NOT evidenced.")
    else:
        print("  → Drift pattern is mixed. Inspect Section B for the specific norms")
        print("    that moved most and correlate with training logs.")


def main():
    args = parse_args()

    print("=" * 100)
    print("  Alpha LayerNorm state drift analysis")
    print("=" * 100)

    baseline_model = load_model_cpu(args.baseline_path)
    current_model = load_model_cpu(args.model_path)

    # Sanity: both checkpoints should have the same layer_types
    if baseline_model.config.layer_types != current_model.config.layer_types:
        print("ERROR: layer_types differ between checkpoints — architecture mismatch.")
        sys.exit(1)

    print(
        f"  Checkpoints loaded. num_hidden_layers="
        f"{current_model.config.num_hidden_layers}"
    )

    baseline_stats = build_all_stats(baseline_model)
    current_stats = build_all_stats(current_model)

    rows = print_per_layer_table(baseline_stats, current_stats)
    print_top_k_drift(rows, args.top_k_drift)
    print_diagnosis(rows)
    print()


if __name__ == "__main__":
    main()
