# Copyright (c) 2025 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common utilities for Alpha model checkpoint conversion.

This module contains shared functionality between MG2HF and HF2MG synchronizers
to reduce code duplication.
"""
import logging
from typing import Dict, Set

# Pattern characters for hybrid model layers
CHAR_MAMBA = 'M'      # Mamba (GatedDeltaNet) layer
CHAR_ATTENTION = '*'  # Full Attention layer
CHAR_MLP = '-'        # MoE MLP layer
CHAR_DENSE_MLP = 'D'  # Dense MLP layer (standard FFN with SwiGLU)

VALID_PATTERN_CHARS: Set[str] = {CHAR_MAMBA, CHAR_ATTENTION, CHAR_MLP, CHAR_DENSE_MLP}


def validate_hybrid_pattern(
    layout: str,
    num_layers: int,
    hybrid_attention_ratio: float,
    rank: int = 0
) -> None:
    """Validate Alpha hybrid model pattern configuration.

    Args:
        layout: The hybrid override pattern string (e.g., "M-M-M-*-M-M-M-*-...")
        num_layers: Expected number of Megatron layers
        hybrid_attention_ratio: Expected ratio of attention layers
        rank: Process rank for logging (only rank 0 logs info)

    Raises:
        ValueError: If pattern length or characters are invalid
    """
    # 1. Validate pattern length
    if len(layout) != num_layers:
        raise ValueError(
            f"\n{'='*70}\n"
            f"Pattern length mismatch!\n"
            f"{'='*70}\n"
            f"  Expected: {num_layers} (--num-layers={num_layers})\n"
            f"  Actual:   {len(layout)}\n"
            f"  Pattern:  '{layout}'\n"
            f"\n"
            f"Hint: Each HF layer = 2 MG layers.\n"
            f"      Pattern should have exactly num_layers tokens.\n"
            f"      For {num_layers} MG layers → {num_layers//2} HF layers\n"
            f"{'='*70}"
        )

    # 2. Validate pattern characters
    invalid_chars = set(layout) - VALID_PATTERN_CHARS
    if invalid_chars:
        raise ValueError(
            f"\n{'='*70}\n"
            f"Invalid characters in hybrid_override_pattern!\n"
            f"{'='*70}\n"
            f"  Invalid: {invalid_chars}\n"
            f"  Valid:   {VALID_PATTERN_CHARS}\n"
            f"  Pattern: '{layout}'\n"
            f"\n"
            f"Legend:\n"
            f"  {CHAR_MAMBA} = Mamba layer\n"
            f"  {CHAR_ATTENTION} = Full Attention layer\n"
            f"  {CHAR_MLP} = MLP layer\n"
            f"{'='*70}"
        )

    # 3. Validate attention ratio consistency (warning only)
    attention_count = layout.count(CHAR_ATTENTION)
    expected_attention = int(num_layers * hybrid_attention_ratio)

    if attention_count != expected_attention:
        logging.warning(
            f"\n{'='*70}\n"
            f"Attention layer count mismatch (non-fatal):\n"
            f"{'='*70}\n"
            f"  Expected: {expected_attention} layers ({hybrid_attention_ratio*100:.1f}%)\n"
            f"  Actual:   {attention_count} layers ({attention_count/num_layers*100:.1f}%)\n"
            f"\n"
            f"This may be intentional if using a custom pattern.\n"
            f"Continuing with pattern: '{layout}'\n"
            f"{'='*70}"
        )


def log_conversion_summary(
    direction: str,
    layout: str,
    args,
    tp_size: int,
    pp_size: int,
    ep_size: int,
    rank: int = 0,
    dp_info: str = None
) -> None:
    """Log conversion configuration summary.

    Args:
        direction: Conversion direction ("MG2HF" or "HF2MG")
        layout: The hybrid override pattern string
        args: Megatron arguments namespace
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        ep_size: Expert parallel size
        rank: Process rank (only rank 0 logs)
        dp_info: Optional data parallel info string
    """
    if rank != 0:
        return

    mamba_count = layout.count(CHAR_MAMBA)
    attention_count = layout.count(CHAR_ATTENTION)
    mlp_count = layout.count(CHAR_MLP)
    num_layers = args.num_layers

    dp_line = f"  DP: {dp_info}\n" if dp_info else ""

    logging.info(
        f"\n{'='*70}\n"
        f"Alpha {direction} Conversion Configuration\n"
        f"{'='*70}\n"
        f"Model Architecture:\n"
        f"  MG Layers:        {num_layers}\n"
        f"  HF Layers:        {num_layers // 2}\n"
        f"  Mapping Ratio:    2:1 (MG → HF)\n"
        f"\n"
        f"Hybrid Pattern:\n"
        f"  Pattern:          '{layout}'\n"
        f"  Mamba layers:     {mamba_count} ({mamba_count/num_layers*100:.1f}%)\n"
        f"  Attention layers: {attention_count} ({attention_count/num_layers*100:.1f}%)\n"
        f"  MLP layers:       {mlp_count} ({mlp_count/num_layers*100:.1f}%)\n"
        f"\n"
        f"MoE Configuration:\n"
        f"  Num Experts:      {args.num_experts}\n"
        f"  Router TopK:      {args.moe_router_topk}\n"
        f"  Expert FFN Size:  {args.moe_ffn_hidden_size}\n"
        f"\n"
        f"Parallelism:\n"
        f"  TP: {tp_size}, PP: {pp_size}, EP: {ep_size}\n"
        f"{dp_line}"
        f"{'='*70}\n"
    )


def build_pipeline_parallel_mapping(
    num_layers: int,
    pp_size: int,
    pp_rank: int
) -> Dict[int, int]:
    """Build mapping from local layer index to global layer index.

    Args:
        num_layers: Total number of Megatron layers
        pp_size: Pipeline parallel size
        pp_rank: Current pipeline parallel rank

    Returns:
        Dictionary mapping local layer indices to global layer indices
    """
    pp_layers_per_stage = [num_layers // pp_size] * pp_size

    return {
        i: v for i, v in enumerate(
            range(
                sum(pp_layers_per_stage[:pp_rank]),
                sum(pp_layers_per_stage[:pp_rank + 1])
            )
        )
    }
