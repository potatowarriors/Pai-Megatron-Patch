# Copyright (c) 2025, Alibaba PAI. All rights reserved.
#
# SSM (State Space Model) module overrides for Pai-Megatron-Patch
#
# This module extends Megatron-LM's SSM implementation to support:
# - Dense MLP layers (D symbol) in addition to MoE MLP layers (-)
# - Hybrid architectures with mixed Dense/MoE MLP layers

from megatron_patch.ssm.mamba_block import MambaStack, MambaStackSubmodules
from megatron_patch.ssm.mamba_hybrid_layer_allocation import (
    Symbols,
    allocate_layers,
    get_layer_maps_from_layer_type_list,
)

__all__ = [
    "MambaStack",
    "MambaStackSubmodules",
    "Symbols",
    "allocate_layers",
    "get_layer_maps_from_layer_type_list",
]
