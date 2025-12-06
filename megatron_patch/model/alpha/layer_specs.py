# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Alibaba PAI. All rights reserved.
#
# Alpha model layer specifications
#
# This module defines layer specs for the Alpha model, which extends
# Qwen3-Next with support for Dense MLP layers (first layer) in addition
# to MoE MLP layers (remaining layers).
#
# Layer pattern symbols:
# - M: Mamba (GatedDeltaNet) layer
# - *: Attention layer
# - -: MoE MLP layer (Mixture of Experts with Shared Expert)
# - D: Dense MLP layer (standard FFN with SwiGLU)

from typing import Optional

from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
    TENorm
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixerSubmodules
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    HAVE_TE = True
except ImportError:
    HAVE_TE = False

# Use our extended MambaStack with Dense MLP support
from megatron_patch.ssm.mamba_block import MambaStack, MambaStackSubmodules

# Reuse components from qwen3_next
from megatron_patch.model.qwen3_next.gated_attention import GatedSoftmaxAttention
from megatron_patch.model.qwen3_next.gated_deltanet import GatedDeltaNetMixer


def get_moe_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE MLP layer.

    This creates a Mixture of Experts layer with:
    - Routed experts: num_experts experts, top-k routing
    - Shared expert: Always-active expert for common knowledge

    Args:
        use_te: Whether to use Transformer Engine
        num_experts: Number of routed experts
        moe_grouped_gemm: Whether to use grouped GEMM for experts
        moe_use_legacy_grouped_gemm: Whether to use legacy grouped GEMM

    Returns:
        ModuleSpec for MoE layer
    """
    if use_te is not None and use_te:
        backend: BackendSpecProvider = TESpecProvider()
    else:
        backend = LocalSpecProvider()
    return get_moe_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )


def get_moe_module_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    use_te_activation_func: bool = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    assert num_experts is not None

    linear_fc1 = backend.column_parallel_linear()
    linear_fc2 = backend.row_parallel_linear()
    activation_func = backend.activation_func()

    mlp = MLPSubmodules(
        linear_fc1=linear_fc1, linear_fc2=linear_fc2, activation_func=activation_func
    )

    expert_module, expert_submodule = backend.grouped_mlp_modules(
        moe_grouped_gemm is not None and moe_grouped_gemm,
        moe_use_legacy_grouped_gemm is not None and moe_use_legacy_grouped_gemm,
    )
    if expert_submodule is not None:
        expert_submodule.activation_func = activation_func

    experts = ModuleSpec(module=expert_module, submodules=expert_submodule)

    # shared experts spec
    shared_experts = ModuleSpec(module=SharedExpertMLP, submodules=mlp)

    # MoE module spec
    moe_module_spec = ModuleSpec(
        module=MoELayer, submodules=MoESubmodules(experts=experts, shared_experts=shared_experts)
    )
    return moe_module_spec


def get_dense_mlp_module_spec(
    use_te: Optional[bool] = True,
) -> ModuleSpec:
    """Helper function to get module spec for Dense MLP layer.

    This creates a standard FFN layer with SwiGLU activation:
    - Input: hidden_size
    - FC1: hidden_size -> ffn_hidden_size * 2 (for GLU)
    - Activation: SwiGLU (silu(gate) * linear)
    - FC2: ffn_hidden_size -> hidden_size

    The ffn_hidden_size is taken from the model config.

    Args:
        use_te: Whether to use Transformer Engine

    Returns:
        ModuleSpec for Dense MLP layer
    """
    if use_te is not None and use_te:
        backend: BackendSpecProvider = TESpecProvider()
    else:
        backend = LocalSpecProvider()

    linear_fc1 = backend.column_parallel_linear()
    linear_fc2 = backend.row_parallel_linear()
    activation_func = backend.activation_func()

    mlp_submodules = MLPSubmodules(
        linear_fc1=linear_fc1,
        linear_fc2=linear_fc2,
        activation_func=activation_func
    )

    # Standard MLP module spec
    dense_mlp_spec = ModuleSpec(
        module=MLP,
        submodules=mlp_submodules
    )

    return dense_mlp_spec


def get_alpha_layer_spec(args):
    """Get the layer specification for Alpha model.

    Alpha is a Mamba-Attention-MoE hybrid model with optional Dense MLP support.
    The model supports the following layer types via hybrid_override_pattern:
    - M: Mamba (GatedDeltaNet) - linear attention with gated delta rule
    - *: Attention - standard multi-head attention with RoPE
    - -: MoE MLP - Mixture of Experts with shared expert
    - D: Dense MLP - standard FFN with SwiGLU (NEW)

    Example patterns:
    - "M-M-M-*-M-M-M-*-M-M-M-*-" : Original (all MoE)
    - "MDM-M-*-M-M-*-M-M-*-M-M-*-" : First MLP is Dense, rest are MoE

    Args:
        args: Model arguments containing:
            - num_experts: Number of routed experts for MoE layers
            - moe_grouped_gemm: Whether to use grouped GEMM

    Returns:
        ModuleSpec for the Alpha model stack
    """
    # Mamba layer spec (GatedDeltaNet)
    mamba_layer_spec = ModuleSpec(
        module=MambaLayer,
        submodules=MambaLayerSubmodules(
            mixer=ModuleSpec(
                module=GatedDeltaNetMixer,
                submodules=MambaMixerSubmodules(
                    in_proj=TELayerNormColumnParallelLinear,
                    out_proj=TERowParallelLinear
                ),
            ),
            mamba_bda=get_bias_dropout_add,
        ),
    )

    # Attention layer spec (GatedSoftmaxAttention)
    attention_layer_spec = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=GatedSoftmaxAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
        ),
    )

    # MoE MLP layer spec (symbol: -)
    moe_mlp_layer_spec = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            pre_mlp_layernorm=TENorm,
            mlp=get_moe_module_spec(
                num_experts=args.num_experts,
                moe_grouped_gemm=args.moe_grouped_gemm,
            ),
            mlp_bda=get_bias_dropout_add
        ),
    )

    # Dense MLP layer spec (symbol: D)
    # Uses ffn_hidden_size from config for standard FFN
    dense_mlp_layer_spec = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            pre_mlp_layernorm=TENorm,
            mlp=get_dense_mlp_module_spec(),
            mlp_bda=get_bias_dropout_add
        ),
    )

    return ModuleSpec(
        module=MambaStack,
        submodules=MambaStackSubmodules(
            mamba_layer=mamba_layer_spec,
            attention_layer=attention_layer_spec,
            mlp_layer=moe_mlp_layer_spec,           # MoE MLP (-)
            dense_mlp_layer=dense_mlp_layer_spec,   # Dense MLP (D)
        ),
    )
