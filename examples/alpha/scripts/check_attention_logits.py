#!/usr/bin/env python3
"""
Attention logit analysis for Alpha model checkpoints.

Measures pre-softmax attention logit magnitudes (Q @ K^T / sqrt(d_k))
to determine whether QK-Clip is needed for training stability.

QK-Clip (qk_clip=true) prevents softmax overflow but disables Flash/Fused
Attention kernels, falling back to O(N^2) memory unfused attention.
This script checks if the logit magnitudes actually warrant QK-Clip.

Thresholds:
  max |logit| < 20  : QK-Clip unnecessary (safe to disable)
  max |logit| 20-30 : Optional (depends on precision/task)
  max |logit| > 30  : QK-Clip strongly recommended

Usage:
  python check_attention_logits.py \\
    --model-path outputs/alpha_baseline_48L_cooldown_20260209_200711/hfmodel_0440000 \\
    --seq-len 2048 --num-samples 4
"""

import argparse
import sys
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class LayerLogitStats:
    """Accumulated logit statistics for one attention layer."""
    max_vals: list = field(default_factory=list)
    min_vals: list = field(default_factory=list)
    mean_vals: list = field(default_factory=list)
    std_vals: list = field(default_factory=list)
    abs_max_vals: list = field(default_factory=list)
    per_head_max: list = field(default_factory=list)   # list of (num_heads,) lists


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze pre-softmax attention logit magnitudes"
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to HuggingFace model checkpoint",
    )
    parser.add_argument(
        "--seq-len", type=int, default=2048,
        help="Sequence length for analysis (default: 2048)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=4,
        help="Number of random input samples to average over (default: 4)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def find_modeling_module():
    """Find the dynamically-loaded modeling_alpha module."""
    for name, mod in sys.modules.items():
        if "modeling_alpha" in name and hasattr(mod, "eager_attention_forward"):
            return mod
    return None


def make_patched_forward(original_fn, repeat_kv_fn, logit_records):
    """Create a monkey-patched eager_attention_forward that captures logit stats."""

    def patched_eager_attention_forward(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        with torch.no_grad():
            # Replicate GQA key expansion (same as original)
            key_expanded = repeat_kv_fn(key, module.num_key_value_groups)

            # Pre-softmax logits: Q @ K^T * (1/sqrt(d_k))
            # Shape: (batch, num_heads, seq_len, seq_len)
            attn_logits = torch.matmul(
                query, key_expanded.transpose(2, 3)
            ) * scaling

            seq_len = attn_logits.shape[-1]

            # Causal mask: only lower-triangular positions are valid
            causal = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool,
                           device=attn_logits.device)
            )
            valid_mask = causal.unsqueeze(0).unsqueeze(0)  # (1,1,S,S)

            logits_f32 = attn_logits.float()

            # --- Per-head max (over valid positions) ---
            masked_for_max = logits_f32.masked_fill(~valid_mask, float("-inf"))
            per_head_max = masked_for_max.amax(dim=(-2, -1))  # (B, H)

            # --- Global stats over all valid positions ---
            valid_elements = logits_f32[valid_mask.expand_as(logits_f32)]

            layer_idx = module.layer_idx
            if layer_idx not in logit_records:
                logit_records[layer_idx] = LayerLogitStats()

            stats = logit_records[layer_idx]
            stats.max_vals.append(valid_elements.max().item())
            stats.min_vals.append(valid_elements.min().item())
            stats.mean_vals.append(valid_elements.mean().item())
            stats.std_vals.append(valid_elements.std().item())
            stats.abs_max_vals.append(valid_elements.abs().max().item())
            stats.per_head_max.append(per_head_max.squeeze(0).tolist())

        # Delegate to original implementation for correct model output
        return original_fn(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs
        )

    return patched_eager_attention_forward


def print_report(logit_records, num_heads_display=None):
    """Print formatted analysis report."""
    print()
    print("=" * 70)
    print("  Attention Logit Analysis (pre-softmax Q @ K^T / sqrt(d_k))")
    print("=" * 70)
    print(
        f"{'Layer':>7}  {'Max':>8}  {'Min':>8}  {'Mean':>8}  "
        f"{'Std':>8}  {'|Max|':>8}"
    )
    print("-" * 70)

    global_max = float("-inf")
    global_max_layer = -1
    global_max_head = -1

    for layer_idx in sorted(logit_records.keys()):
        stats = logit_records[layer_idx]

        agg_max = max(stats.max_vals)
        agg_min = min(stats.min_vals)
        agg_mean = sum(stats.mean_vals) / len(stats.mean_vals)
        agg_std = max(stats.std_vals)
        agg_abs_max = max(stats.abs_max_vals)

        # Per-head max across all samples
        num_heads = len(stats.per_head_max[0])
        for h in range(num_heads):
            head_max = max(sample[h] for sample in stats.per_head_max)
            if head_max > global_max:
                global_max = head_max
                global_max_layer = layer_idx
                global_max_head = h

        print(
            f"{layer_idx:>7d}  {agg_max:>8.1f}  {agg_min:>8.1f}  "
            f"{agg_mean:>8.2f}  {agg_std:>8.1f}  {agg_abs_max:>8.1f}"
        )

    print("-" * 70)
    print(
        f"\n  Global max logit: {global_max:.1f} "
        f"(Layer {global_max_layer}, Head {global_max_head})"
    )

    # Per-head breakdown for the worst layer
    worst_stats = logit_records[global_max_layer]
    num_heads = len(worst_stats.per_head_max[0])
    print(f"\n  Per-head max for Layer {global_max_layer}:")
    for h in range(num_heads):
        head_max = max(sample[h] for sample in worst_stats.per_head_max)
        marker = " <<<" if h == global_max_head else ""
        print(f"    Head {h:>2d}: {head_max:>8.1f}{marker}")

    # Recommendation
    print()
    print("=" * 70)
    if global_max < 20:
        print("  RESULT: QK-Clip is NOT needed (max logit < 20)")
        print()
        print("  Safe to set qk_clip=false and use Flash/Fused Attention.")
        print("  This avoids the O(N^2) memory overhead of unfused attention.")
    elif global_max < 30:
        print("  RESULT: QK-Clip is OPTIONAL (max logit 20-30)")
        print()
        print("  Logit magnitudes are moderate. Consider disabling QK-Clip")
        print("  if OOM is a concern. Monitor loss stability during training.")
    else:
        print("  RESULT: QK-Clip is STRONGLY RECOMMENDED (max logit > 30)")
        print()
        print("  High logit values indicate risk of softmax numerical issues.")
        print("  Keep qk_clip=true despite the memory cost.")
    print("=" * 70)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # --- Detect SGLang native checkpoint (incompatible with AutoModel) ---
    if args.model_path.rstrip("/").endswith("_sglang_native"):
        orig = args.model_path.rstrip("/").rsplit("_sglang_native", 1)[0]
        print("ERROR: SGLang native checkpoints cannot be loaded with AutoModel.")
        print(f"  Use the original HF checkpoint instead: {orig}")
        sys.exit(1)

    # --- Load model with eager attention ---
    print(f"Loading model from: {args.model_path}")
    print("  attn_implementation='eager' (required for logit capture)")

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    # Verify eager mode is active
    assert model.config._attn_implementation == "eager", (
        f"Expected eager attention, got '{model.config._attn_implementation}'"
    )

    # --- Identify full attention layers ---
    config = model.config
    layer_types = config.layer_types
    attn_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]

    print(f"  Full attention layers: {attn_layers} ({len(attn_layers)} layers)")
    print(
        f"  GQA config: {config.num_attention_heads} Q-heads, "
        f"{config.num_key_value_heads} KV-heads, head_dim={config.head_dim}"
    )
    print(f"  Scaling factor: 1/sqrt({config.head_dim}) = {config.head_dim**-0.5:.6f}")
    print(f"  Sequence length: {args.seq_len}, Samples: {args.num_samples}")

    # --- Monkey-patch eager_attention_forward ---
    modeling_module = find_modeling_module()
    if modeling_module is None:
        print("\nERROR: Could not find modeling_alpha module in sys.modules")
        sys.exit(1)

    original_fn = modeling_module.eager_attention_forward
    repeat_kv_fn = modeling_module.repeat_kv
    logit_records = {}

    modeling_module.eager_attention_forward = make_patched_forward(
        original_fn, repeat_kv_fn, logit_records
    )

    # --- Forward passes with random token inputs ---
    print()
    vocab_size = config.vocab_size
    device = next(model.parameters()).device

    for i in range(args.num_samples):
        input_ids = torch.randint(
            0, vocab_size, (1, args.seq_len), device=device
        )
        print(f"  Sample {i + 1}/{args.num_samples} ...", end=" ", flush=True)
        with torch.no_grad():
            model(input_ids)
        print("done")

    # --- Restore original function ---
    modeling_module.eager_attention_forward = original_fn

    # --- Report ---
    if not logit_records:
        print("\nERROR: No logit data captured. Check model architecture.")
        sys.exit(1)

    print_report(logit_records)


if __name__ == "__main__":
    main()
