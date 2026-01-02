#!/usr/bin/env python3
# Copyright (c) 2025 Alpha Project Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Alpha Model Parameter Counter
==============================
Accurately count Total and Activated parameters by analyzing model architecture.

This script does NOT require GPU - it calculates parameters based on model config.

Usage:
    python count_model_params.py
    python count_model_params.py --config configs/model/baseline_48L.yaml
"""

import argparse
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class ModelParams:
    """Container for parameter counts"""
    total: int = 0
    active: int = 0

    def add(self, total: int, active: int = None):
        """Add parameters. If active is None, use total."""
        self.total += total
        self.active += active if active is not None else total


class AlphaParameterCounter:
    """
    Accurate parameter counter for Alpha model.

    Based on actual implementation in:
    - megatron_patch/model/qwen3_next/gated_deltanet.py
    - megatron_patch/model/qwen3_next/layer_specs.py
    """

    def __init__(self, config: Dict):
        self.config = config['model']
        self.moe = config['model']['moe']
        self.hybrid = config['model']['hybrid']

        # Basic architecture
        self.num_layers = self.config['num_layers']
        self.hidden_size = self.config['hidden_size']
        self.ffn_hidden_size = self.config['ffn_hidden_size']
        self.vocab_size = self.config['padded_vocab_size']

        # Attention config
        self.num_attention_heads = self.config['num_attention_heads']
        self.kv_channels = self.config['kv_channels']
        self.num_query_groups = self.config['num_query_groups']
        self.qk_layernorm = self.config.get('qk_layernorm', True)

        # MoE config
        self.num_experts = self.moe['num_experts']
        self.moe_ffn_hidden_size = self.moe['moe_ffn_hidden_size']
        self.router_topk = self.moe['router_topk']
        self.shared_expert_size = self.moe['shared_expert_intermediate_size']

        # GatedDeltaNet (Mamba) config
        self.mamba_state_dim = self.hybrid['mamba_state_dim']  # d_state
        self.mamba_head_dim = self.hybrid['mamba_head_dim']    # head_v_dim
        self.mamba_num_groups = self.hybrid['mamba_num_groups']  # ngroups
        self.mamba_num_heads = self.hybrid['mamba_num_heads']   # nheads

        # Derived values for GatedDeltaNet
        self.d_inner = self.mamba_num_heads * self.mamba_head_dim  # nheads * head_v_dim
        self.d_conv = 4  # Fixed in implementation

        # Parse pattern
        self.pattern = self.hybrid['override_pattern']
        self._parse_pattern()

    def _parse_pattern(self):
        """Parse hybrid pattern to count each layer type"""
        self.num_mamba_layers = self.pattern.count('M')
        self.num_attention_layers = self.pattern.count('*')
        self.num_mlp_layers = self.pattern.count('-')

        total = self.num_mamba_layers + self.num_attention_layers + self.num_mlp_layers
        assert total == self.num_layers, \
            f"Pattern tokens ({total}) != num_layers ({self.num_layers})"

    def count_embedding_params(self) -> Tuple[int, int]:
        """
        Embedding layers (input + output if untied)

        - input_embedding: vocab_size × hidden_size
        - output_embedding: vocab_size × hidden_size (if untied)
        """
        input_embed = self.vocab_size * self.hidden_size

        if self.config.get('untie_embeddings_and_output_weights', True):
            output_embed = self.vocab_size * self.hidden_size
        else:
            output_embed = 0

        total = input_embed + output_embed
        return total, total  # Always active

    def count_gated_deltanet_layer_params(self) -> Tuple[int, int]:
        """
        Single GatedDeltaNet layer parameters.

        Based on gated_deltanet.py implementation:
        - in_proj: h → (d_inner*2 + 2*ngroups*d_state + nheads*2)
        - conv1d: (d_inner + 2*ngroups*d_state) channels × kernel_size
        - A_log: (nheads,)
        - dt_bias: (nheads,)
        - norm (RMSNormGated): (head_v_dim,) weight only
        - out_proj: d_inner → h
        """
        h = self.hidden_size
        d_state = self.mamba_state_dim
        ngroups = self.mamba_num_groups
        nheads = self.mamba_num_heads
        head_v_dim = self.mamba_head_dim
        d_inner = self.d_inner
        d_conv = self.d_conv

        # 1. in_proj: TELayerNormColumnParallelLinear
        # Includes LayerNorm (h) + Linear (h → out_dim)
        # LayerNorm: weight (h) + bias (h) for RMSNorm just weight (h)
        in_proj_out_dim = d_inner * 2 + 2 * ngroups * d_state + nheads * 2
        in_proj_ln = h  # RMSNorm weight only
        in_proj_linear = h * in_proj_out_dim  # No bias (disable_bias_linear=True)
        in_proj = in_proj_ln + in_proj_linear

        # 2. Conv1D: depthwise convolution
        # Weight: (conv_channels, 1, kernel_size) → conv_channels * kernel_size
        conv_channels = d_inner + 2 * ngroups * d_state
        conv1d = conv_channels * d_conv  # No bias

        # 3. A_log: (nheads,)
        a_log = nheads

        # 4. dt_bias: (nheads,)
        dt_bias = nheads

        # 5. RMSNormGated: (head_v_dim,) weight only
        # Note: This is per-head normalization
        norm = head_v_dim

        # 6. out_proj: TERowParallelLinear
        # Linear: d_inner → h (no bias)
        out_proj = d_inner * h

        total = in_proj + conv1d + a_log + dt_bias + norm + out_proj
        return total, total  # All active

    def count_attention_layer_params(self) -> Tuple[int, int]:
        """
        Single Attention layer parameters (GatedSoftmaxAttention).

        Based on layer_specs.py:
        - linear_qkv: TELayerNormColumnParallelLinear
        - core_attention: TEDotProductAttention (no learnable params)
        - linear_proj: TERowParallelLinear
        - q_layernorm, k_layernorm: TENorm (if qk_layernorm=True)
        """
        h = self.hidden_size
        num_heads = self.num_attention_heads
        kv_channels = self.kv_channels
        num_kv_heads = self.num_query_groups

        # 1. linear_qkv: TELayerNormColumnParallelLinear
        # LayerNorm: weight (h)
        # Linear: h → (num_heads * kv_channels + 2 * num_kv_heads * kv_channels)
        qkv_ln = h  # RMSNorm weight
        q_dim = num_heads * kv_channels
        kv_dim = num_kv_heads * kv_channels * 2  # K and V
        qkv_linear = h * (q_dim + kv_dim)
        linear_qkv = qkv_ln + qkv_linear

        # 2. core_attention: No learnable parameters

        # 3. linear_proj: TERowParallelLinear
        # Linear: num_heads * kv_channels → h
        linear_proj = num_heads * kv_channels * h

        # 4. QK LayerNorm (if enabled)
        qk_norm = 0
        if self.qk_layernorm:
            # q_layernorm: weight (kv_channels)
            # k_layernorm: weight (kv_channels)
            qk_norm = kv_channels * 2  # RMSNorm weight only

        total = linear_qkv + linear_proj + qk_norm
        return total, total  # All active

    def count_moe_layer_params(self) -> Tuple[int, int]:
        """
        Single MoE layer parameters.

        Based on layer_specs.py mlp_layer:
        - pre_mlp_layernorm: TENorm
        - moe: MoELayer
          - router: h → num_experts
          - experts: GroupedMLP (SwiGLU per expert)
          - shared_experts: SharedExpertMLP
        """
        h = self.hidden_size
        num_experts = self.num_experts
        expert_ffn = self.moe_ffn_hidden_size
        topk = self.router_topk
        shared_ffn = self.shared_expert_size

        # 1. pre_mlp_layernorm: TENorm
        # RMSNorm: weight (h)
        pre_norm = h

        # 2. Router: Linear h → num_experts
        router = h * num_experts

        # 3. Expert FFN: SwiGLU per expert
        # Gate: h → expert_ffn
        # Up: h → expert_ffn
        # Down: expert_ffn → h
        expert_gate = h * expert_ffn
        expert_up = h * expert_ffn
        expert_down = expert_ffn * h
        per_expert = expert_gate + expert_up + expert_down
        all_experts = per_expert * num_experts
        active_experts = per_expert * topk

        # 4. Shared Expert: SwiGLU (always active)
        shared_gate = h * shared_ffn
        shared_up = h * shared_ffn
        shared_down = shared_ffn * h
        shared_expert = shared_gate + shared_up + shared_down

        total = pre_norm + router + all_experts + shared_expert
        active = pre_norm + router + active_experts + shared_expert

        return total, active

    def count_all_params(self) -> Dict[str, ModelParams]:
        """Count all parameters in the model"""
        results = {}

        # 1. Embedding
        embed_total, embed_active = self.count_embedding_params()
        results['embedding'] = ModelParams(embed_total, embed_active)

        # 2. GatedDeltaNet layers (M in pattern)
        gdn_total, gdn_active = self.count_gated_deltanet_layer_params()
        results['gated_deltanet'] = ModelParams(
            gdn_total * self.num_mamba_layers,
            gdn_active * self.num_mamba_layers
        )
        results['gated_deltanet_per_layer'] = ModelParams(gdn_total, gdn_active)

        # 3. Attention layers (* in pattern)
        attn_total, attn_active = self.count_attention_layer_params()
        results['attention'] = ModelParams(
            attn_total * self.num_attention_layers,
            attn_active * self.num_attention_layers
        )
        results['attention_per_layer'] = ModelParams(attn_total, attn_active)

        # 4. MoE layers (- in pattern)
        moe_total, moe_active = self.count_moe_layer_params()
        results['moe'] = ModelParams(
            moe_total * self.num_mlp_layers,
            moe_active * self.num_mlp_layers
        )
        results['moe_per_layer'] = ModelParams(moe_total, moe_active)

        # 5. Final LayerNorm (before output)
        final_norm = self.hidden_size  # RMSNorm weight
        results['final_norm'] = ModelParams(final_norm, final_norm)

        # 6. Total
        total = ModelParams()
        for key in ['embedding', 'gated_deltanet', 'attention', 'moe', 'final_norm']:
            total.add(results[key].total, results[key].active)
        results['total'] = total

        return results

    def format_number(self, num: int) -> str:
        """Format number with B/M/K suffix"""
        if num >= 1e9:
            return f"{num / 1e9:.4f}B"
        elif num >= 1e6:
            return f"{num / 1e6:.2f}M"
        elif num >= 1e3:
            return f"{num / 1e3:.2f}K"
        return str(num)

    def print_report(self, results: Dict[str, ModelParams]):
        """Print detailed parameter report"""
        print("\n" + "=" * 80)
        print(f"Alpha Model Parameter Report: {self.config['name']}")
        print("=" * 80)

        print(f"\n[Configuration]")
        print(f"  Pattern: {self.pattern}")
        print(f"  Layers: {self.num_layers} total")
        print(f"    - GatedDeltaNet (M): {self.num_mamba_layers}")
        print(f"    - Attention (*): {self.num_attention_layers}")
        print(f"    - MoE (-): {self.num_mlp_layers}")
        print(f"  Hidden Size: {self.hidden_size}")
        print(f"  Vocab Size: {self.vocab_size:,}")
        print(f"  MoE: {self.num_experts} experts, Top-{self.router_topk}")

        print(f"\n[Per-Layer Parameters]")
        print("-" * 80)
        print(f"  {'Component':<25} {'Total':>20} {'Active':>20}")
        print("-" * 80)

        gdn = results['gated_deltanet_per_layer']
        print(f"  {'GatedDeltaNet':<25} {self.format_number(gdn.total):>20} {self.format_number(gdn.active):>20}")

        attn = results['attention_per_layer']
        print(f"  {'Attention':<25} {self.format_number(attn.total):>20} {self.format_number(attn.active):>20}")

        moe = results['moe_per_layer']
        print(f"  {'MoE':<25} {self.format_number(moe.total):>20} {self.format_number(moe.active):>20}")

        print(f"\n[Total Parameters by Component]")
        print("-" * 80)
        print(f"  {'Component':<25} {'Count':>6} {'Total':>20} {'Active':>20}")
        print("-" * 80)

        embed = results['embedding']
        print(f"  {'Embedding':<25} {'1':>6} {self.format_number(embed.total):>20} {self.format_number(embed.active):>20}")

        gdn_all = results['gated_deltanet']
        print(f"  {'GatedDeltaNet':<25} {self.num_mamba_layers:>6} {self.format_number(gdn_all.total):>20} {self.format_number(gdn_all.active):>20}")

        attn_all = results['attention']
        print(f"  {'Attention':<25} {self.num_attention_layers:>6} {self.format_number(attn_all.total):>20} {self.format_number(attn_all.active):>20}")

        moe_all = results['moe']
        print(f"  {'MoE':<25} {self.num_mlp_layers:>6} {self.format_number(moe_all.total):>20} {self.format_number(moe_all.active):>20}")

        final = results['final_norm']
        print(f"  {'Final LayerNorm':<25} {'1':>6} {self.format_number(final.total):>20} {self.format_number(final.active):>20}")

        print("-" * 80)

        total = results['total']
        print(f"\n[SUMMARY]")
        print("=" * 80)
        print(f"  {'Total Parameters:':<30} {self.format_number(total.total):>20}")
        print(f"  {'Activated Parameters:':<30} {self.format_number(total.active):>20}")
        print(f"  {'Activation Ratio:':<30} {total.active / total.total * 100:>19.2f}%")
        print("=" * 80)

        # Additional breakdown
        print(f"\n[Detailed Breakdown]")
        print(f"  MoE Contribution:")
        print(f"    - Total MoE: {self.format_number(moe_all.total)} ({moe_all.total / total.total * 100:.1f}% of total)")
        print(f"    - Active MoE: {self.format_number(moe_all.active)} (Top-{self.router_topk} experts)")
        print(f"    - Expert Utilization: {self.router_topk / self.num_experts * 100:.2f}%")

        non_moe = total.total - moe_all.total
        print(f"\n  Non-MoE Parameters:")
        print(f"    - {self.format_number(non_moe)} ({non_moe / total.total * 100:.1f}% of total)")

        print("\n")


def main():
    parser = argparse.ArgumentParser(description="Alpha Model Parameter Counter")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model/baseline_48L.yaml',
        help='Path to model config YAML file'
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Count parameters
    counter = AlphaParameterCounter(config)
    results = counter.count_all_params()
    counter.print_report(results)

    # Return for programmatic use
    return results


if __name__ == '__main__':
    main()
