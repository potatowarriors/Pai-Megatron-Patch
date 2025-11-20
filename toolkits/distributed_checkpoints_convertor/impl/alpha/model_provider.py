# Copyright (c) 2025 Alibaba PAI and Nvidia Megatron-LM Team.
# Copyright (c) 2025 Alpha Project Team.
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

"""
Alpha Model Provider for Checkpoint Conversion
==============================================
Defines the Alpha model architecture for HF ↔ Megatron conversion.

Base: Qwen3-Next Mamba Hybrid Architecture
Modifications: 24 layers, 256 experts, 32 heads
"""

from functools import partial
import torch
import torch._dynamo

from model_provider import model_provider as mcore_model_provider # Megatron-LM-250908/model_provider.py

from megatron.training.arguments import core_transformer_config_from_args

from megatron.training import print_rank_0

torch._dynamo.config.suppress_errors = True


from model_provider import count_parameters_in_layer
from megatron.core.models.mamba import MambaModel
from megatron.core.transformer import TransformerConfig
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from megatron_patch.model.qwen3_next.layer_specs import get_qwen3_next_layer_spec
from megatron_patch.model.qwen3_next.transformer_config import Qwen3NextTransformerConfig


def _validate_alpha_args(args):
    """
    Validate Alpha-specific arguments before model building.

    Raises:
        ValueError: If required arguments are missing or invalid
    """
    # 1. Check required hybrid arguments
    required_hybrid_args = {
        'hybrid_attention_ratio': 'attention ratio in hybrid model',
        'hybrid_mlp_ratio': 'MLP ratio in hybrid model',
        'hybrid_override_pattern': 'layer pattern (M-M-M-*-...)'
    }

    missing_args = []
    for arg_name, description in required_hybrid_args.items():
        if not hasattr(args, arg_name) or getattr(args, arg_name) is None:
            missing_args.append(f"  --{arg_name.replace('_', '-')}: {description}")

    if missing_args:
        raise ValueError(
            f"\n{'='*70}\n"
            f"Missing required Alpha arguments!\n"
            f"{'='*70}\n"
            f"Alpha model requires hybrid configuration:\n"
            + '\n'.join(missing_args) + '\n'
            f"\n"
            f"Example:\n"
            f"  --hybrid-attention-ratio 0.125\n"
            f"  --hybrid-mlp-ratio 0.5\n"
            f"  --hybrid-override-pattern M-M-M-*-M-M-M-*-M-M-M-*-\n"
            f"{'='*70}"
        )

    # 2. Check TP constraint (Mamba limitation)
    if args.tensor_model_parallel_size != 1:
        raise ValueError(
            f"\n{'='*70}\n"
            f"Tensor Parallelism constraint violation!\n"
            f"{'='*70}\n"
            f"  Current TP: {args.tensor_model_parallel_size}\n"
            f"  Required:   1\n"
            f"\n"
            f"Reason: Alpha uses Mamba layers which currently do not support TP > 1.\n"
            f"\n"
            f"Solution: Set --tensor-model-parallel-size 1\n"
            f"{'='*70}"
        )

    # 3. Check pattern format
    pattern = args.hybrid_override_pattern
    valid_chars = {'M', '*', '-'}
    invalid_chars = set(pattern) - valid_chars

    if invalid_chars:
        raise ValueError(
            f"\n{'='*70}\n"
            f"Invalid hybrid_override_pattern!\n"
            f"{'='*70}\n"
            f"  Pattern: '{pattern}'\n"
            f"  Invalid characters: {invalid_chars}\n"
            f"\n"
            f"Valid characters:\n"
            f"  M = Mamba layer (Linear Attention SSM)\n"
            f"  * = Full Attention layer (Multi-Head Attention)\n"
            f"  - = MLP layer (Feed-Forward Network)\n"
            f"\n"
            f"Example valid pattern: M-M-M-*-M-M-M-*-M-M-M-*-\n"
            f"{'='*70}"
        )

    # 4. Check pattern length matches num_layers
    if len(pattern) != args.num_layers:
        raise ValueError(
            f"\n{'='*70}\n"
            f"Pattern length mismatch!\n"
            f"{'='*70}\n"
            f"  Pattern length: {len(pattern)}\n"
            f"  num_layers:     {args.num_layers}\n"
            f"\n"
            f"Pattern must have exactly num_layers characters.\n"
            f"Each character represents one Megatron layer.\n"
            f"\n"
            f"Current pattern: '{pattern}'\n"
            f"Expected length: {args.num_layers}\n"
            f"{'='*70}"
        )

    print_rank_0("✓ Alpha arguments validation passed")


def mamba_builder(args, pre_process, post_process, vp_stage=None, config=None):
    """
    Build Alpha Mamba model for checkpoint conversion.

    Args:
        args: Megatron arguments
        pre_process: Include embedding layer
        post_process: Include output layer
        vp_stage: Virtual pipeline stage (unused)
        config: Transformer config (auto-generated if None)

    Returns:
        MambaModel: Alpha model instance
    """
    print_rank_0('building Alpha MAMBA model for conversion ...')

    # Validate Alpha-specific arguments before building model
    _validate_alpha_args(args)

    if config is None:
        config = core_transformer_config_from_args(args, Qwen3NextTransformerConfig)
    assert args.use_legacy_models is False, "Mamba only supported in Mcore!"

    model = MambaModel(
        config=config,
        mamba_stack_spec=get_qwen3_next_layer_spec(args),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        hybrid_attention_ratio=args.hybrid_attention_ratio,
        hybrid_mlp_ratio=args.hybrid_mlp_ratio,
        hybrid_override_pattern=args.hybrid_override_pattern,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
    )

    return model

model_provider = partial(mcore_model_provider, mamba_builder)
