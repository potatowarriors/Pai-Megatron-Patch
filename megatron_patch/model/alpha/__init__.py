# Copyright (c) 2025, Alibaba PAI. All rights reserved.
#
# Alpha model module for Pai-Megatron-Patch
#
# Alpha is a Qwen3-Next based Mamba Hybrid architecture with:
# - GatedDeltaNet (linear attention with gated delta rule)
# - Multi-Head Attention
# - Mixed Dense/MoE MLP layers
#
# This module extends Qwen3-Next with support for Dense MLP layers
# (first layer) following the DeepSeek-V3 design pattern.

from megatron_patch.model.alpha.layer_specs import (
    get_alpha_layer_spec,
    get_dense_mlp_module_spec,
    get_moe_module_spec,
)

__all__ = [
    "get_alpha_layer_spec",
    "get_dense_mlp_module_spec",
    "get_moe_module_spec",
]
