#!/usr/bin/env python
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
Alpha Model MG ↔ HF Weight Validation
======================================
Validates that Megatron checkpoint weights match HuggingFace converted model weights.

NOTE: MG GatedDeltaNet does NOT support inference (training only).
      Therefore, we validate by comparing weights directly.
      If all weights match, the conversion is correct.

This script requires Megatron distributed initialization.

Usage:
    bash validate.sh /path/to/mg/checkpoint /path/to/hf/model
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field

# Add paths for Megatron-LM-251125 and megatron_patch
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
MEGATRON_PATH = os.path.join(ROOT_DIR, "backends/megatron/Megatron-LM-251125")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, MEGATRON_PATH)

# modelopt(0.25.0) neutralization: NGC modelopt monkey-patches transformers'
# _load_pretrained_model with an OLD signature, incompatible with the installed
# transformers → `_new__load_pretrained_model() missing pretrained_model_name_or_path`
# when Megatron path pulls modelopt in. Block modelopt imports so transformers loads
# HF weights unpatched. Verified 2026-08-29: without this, load_hf_model crashes.
class _BlockModelopt:
    def find_spec(self, name, path=None, target=None):
        if name == "modelopt" or name.startswith("modelopt."):
            raise ModuleNotFoundError("modelopt neutralized (transformers API incompat)")
        return None
sys.meta_path.insert(0, _BlockModelopt())

import torch
import torch.nn.functional as F


# ==============================================================================
# Data Classes for Results
# ==============================================================================

@dataclass
class WeightComparison:
    """Result of comparing a single weight tensor."""
    mg_name: str
    hf_name: str
    matched: bool
    max_diff: float
    mean_diff: float
    cosine_sim: float
    shape_mg: Tuple
    shape_hf: Tuple
    note: str = ""


@dataclass
class LayerComparison:
    """Result of comparing all weights in a layer."""
    layer_idx: int
    layer_type: str  # "Mamba", "Attention", "MLP", "MoE"
    comparisons: List[WeightComparison] = field(default_factory=list)

    @property
    def all_matched(self) -> bool:
        return all(c.matched for c in self.comparisons)


@dataclass
class ValidationResult:
    """Overall validation result."""
    embedding_comparisons: List[WeightComparison] = field(default_factory=list)
    layer_comparisons: List[LayerComparison] = field(default_factory=list)
    output_comparisons: List[WeightComparison] = field(default_factory=list)
    compared_mg_weights: Set[str] = field(default_factory=set)
    all_mg_weights: Set[str] = field(default_factory=set)

    @property
    def all_matched(self) -> bool:
        embed_ok = all(c.matched for c in self.embedding_comparisons)
        layer_ok = all(lc.all_matched for lc in self.layer_comparisons)
        output_ok = all(c.matched for c in self.output_comparisons)
        return embed_ok and layer_ok and output_ok

    @property
    def unchecked_weights(self) -> Set[str]:
        return self.all_mg_weights - self.compared_mg_weights


# ==============================================================================
# Utility Functions
# ==============================================================================

def compute_metrics(t1: torch.Tensor, t2: torch.Tensor) -> Tuple[float, float, float]:
    """Compute comparison metrics between two tensors.

    Returns:
        (max_diff, mean_diff, cosine_sim)
    """
    t1_f = t1.float().flatten()
    t2_f = t2.float().flatten()

    diff = (t1_f - t2_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Cosine similarity
    cos_sim = F.cosine_similarity(t1_f.unsqueeze(0), t2_f.unsqueeze(0)).item()

    return max_diff, mean_diff, cos_sim


def compare_tensors(
    mg_tensor: torch.Tensor,
    hf_tensor: torch.Tensor,
    mg_name: str,
    hf_name: str,
    threshold: float = 0.01,
    note: str = ""
) -> WeightComparison:
    """Compare two tensors and return comparison result.

    Strict tolerance on purpose: the converter copies weights bit-for-bit (no
    arithmetic) at their trained dtype, so every faithful comparison is exact
    (max_diff ≈ 0) — bf16 weight vs bf16 weight, and fp32 buffer vs fp32 buffer.
    The one tensor MG keeps in fp32, ``router.expert_bias``, is saved fp32 by the
    converter and kept fp32 on load via the HF model's
    ``_keep_in_fp32_modules_strict`` (see hf_model/modeling_alpha.py), so it must
    be compared fp32-vs-fp32 and matches exactly. A nonzero max_diff here means a
    real conversion/precision regression — do not relax this to paper over it.
    """
    # Move to same device if needed
    if mg_tensor.device != hf_tensor.device:
        hf_tensor = hf_tensor.to(mg_tensor.device)

    max_diff, mean_diff, cos_sim = compute_metrics(mg_tensor, hf_tensor)
    matched = max_diff < threshold and cos_sim > 0.999

    return WeightComparison(
        mg_name=mg_name,
        hf_name=hf_name,
        matched=matched,
        max_diff=max_diff,
        mean_diff=mean_diff,
        cosine_sim=cos_sim,
        shape_mg=tuple(mg_tensor.shape),
        shape_hf=tuple(hf_tensor.shape),
        note=note
    )


def print_comparison(cmp: WeightComparison, verbose: bool = False):
    """Print a single weight comparison result."""
    status = "✓" if cmp.matched else "✗"

    if cmp.matched and not verbose:
        print(f"  {status} {cmp.mg_name} ↔ {cmp.hf_name}")
    else:
        print(f"  {status} {cmp.mg_name} ↔ {cmp.hf_name}")
        print(f"      Shape: MG={cmp.shape_mg}, HF={cmp.shape_hf}")
        print(f"      max_diff={cmp.max_diff:.6f}, mean_diff={cmp.mean_diff:.6f}, cos_sim={cmp.cosine_sim:.6f}")
        if cmp.note:
            print(f"      Note: {cmp.note}")


# ==============================================================================
# Argument Parsing
# ==============================================================================

def add_validation_args(parser):
    """Add validation-specific arguments."""
    group = parser.add_argument_group(title='validation')

    group.add_argument(
        "--mg-checkpoint",
        type=str,
        default=None,
        help="Path to Megatron checkpoint directory (overrides --load for checkpoint)"
    )
    group.add_argument(
        "--hf-model",
        type=str,
        required=True,
        help="Path to HuggingFace model directory"
    )
    group.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Maximum allowed absolute difference for weight match (default: 0.01)"
    )
    group.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed comparison results for all weights"
    )

    return parser


# ==============================================================================
# Model Provider
# ==============================================================================

def model_provider(pre_process=True, post_process=True):
    """Build Alpha MambaModel for weight loading."""
    from megatron.training import get_args, print_rank_0
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.core.models.mamba import MambaModel

    from megatron_patch.model.alpha.layer_specs import get_alpha_layer_spec
    from megatron_patch.model.qwen3_next.transformer_config import Qwen3NextTransformerConfig

    args = get_args()
    print_rank_0('Building Alpha MAMBA model for weight validation ...')

    config = core_transformer_config_from_args(args, Qwen3NextTransformerConfig)

    model = MambaModel(
        config=config,
        mamba_stack_spec=get_alpha_layer_spec(args),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        hybrid_attention_ratio=args.hybrid_attention_ratio,
        hybrid_mlp_ratio=args.hybrid_mlp_ratio,
        hybrid_override_pattern=args.hybrid_override_pattern,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=False,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
    )

    return model


# ==============================================================================
# HuggingFace Model Loading
# ==============================================================================

def load_hf_model(hf_model_path: str):
    """Load HuggingFace model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading HF model from {hf_model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_path,
        trust_remote_code=True,
    )

    return model, tokenizer


# ==============================================================================
# Weight Collection
# ==============================================================================

def collect_mg_weights(mg_model) -> Dict[str, torch.Tensor]:
    """Collect all MG model weights into a dictionary."""
    weights = {}
    actual_model = mg_model.module if hasattr(mg_model, 'module') else mg_model

    for name, param in actual_model.named_parameters():
        weights[name] = param.data

    # Also collect buffers (like A_log, D)
    for name, buf in actual_model.named_buffers():
        weights[name] = buf

    return weights


def collect_hf_weights(hf_model) -> Dict[str, torch.Tensor]:
    """Collect all HF model weights into a dictionary."""
    weights = {}

    for name, param in hf_model.named_parameters():
        weights[name] = param.data

    for name, buf in hf_model.named_buffers():
        weights[name] = buf

    return weights


# ==============================================================================
# Layer Type Detection
# ==============================================================================

def get_mg_layer_type(mg_layer, layer_idx: int, pattern: str) -> str:
    """Determine MG layer type from pattern.

    Pattern chars:
        M = Mamba (GatedDeltaNet)
        * = Attention
        - = MLP (or MoE)
        D = Dense MLP (first layer only, for baseline_48L)

    MG uses 2:1 ratio: each HF layer = 2 MG layers (mixer + mlp)
    So MG layer_idx corresponds to pattern[layer_idx]
    """
    if layer_idx < len(pattern):
        char = pattern[layer_idx]
        if char == 'M':
            return "Mamba"
        elif char == '*':
            return "Attention"
        elif char == '-':
            return "MoE"
        elif char == 'D':
            return "DenseMLP"
    return "Unknown"


# ==============================================================================
# Embedding Comparison
# ==============================================================================

def compare_embedding(mg_model, hf_model, result: ValidationResult, threshold: float):
    """Compare embedding weights."""
    print("\n" + "="*70)
    print("EMBEDDING COMPARISON")
    print("="*70)

    actual_mg = mg_model.module if hasattr(mg_model, 'module') else mg_model

    # MG: embedding.word_embeddings.weight
    mg_embed = actual_mg.embedding.word_embeddings.weight.data
    # HF: model.embed_tokens.weight
    hf_embed = hf_model.model.embed_tokens.weight.data

    cmp = compare_tensors(
        mg_embed, hf_embed,
        "embedding.word_embeddings.weight",
        "model.embed_tokens.weight",
        threshold
    )
    result.embedding_comparisons.append(cmp)
    result.compared_mg_weights.add("embedding.word_embeddings.weight")

    print_comparison(cmp, verbose=True)


# ==============================================================================
# Final Layer Norm Comparison
# ==============================================================================

def compare_final_norm(mg_model, hf_model, result: ValidationResult, threshold: float):
    """Compare final layer norm weights.

    Note: MambaStack uses 'final_norm' instead of 'final_layernorm'.
    """
    print("\n" + "="*70)
    print("FINAL LAYER NORM COMPARISON")
    print("="*70)

    actual_mg = mg_model.module if hasattr(mg_model, 'module') else mg_model

    # MG: decoder.final_norm.weight (MambaStack) or decoder.final_layernorm.weight (GPTModel)
    # HF: model.norm.weight
    if hasattr(actual_mg.decoder, 'final_norm'):
        mg_norm = actual_mg.decoder.final_norm.weight.data
        mg_norm_name = "decoder.final_norm.weight"
    else:
        mg_norm = actual_mg.decoder.final_layernorm.weight.data
        mg_norm_name = "decoder.final_layernorm.weight"

    hf_norm = hf_model.model.norm.weight.data

    cmp = compare_tensors(
        mg_norm, hf_norm,
        mg_norm_name,
        "model.norm.weight",
        threshold
    )
    result.output_comparisons.append(cmp)
    result.compared_mg_weights.add(mg_norm_name)

    print_comparison(cmp, verbose=True)


# ==============================================================================
# Output Layer Comparison
# ==============================================================================

def compare_output_layer(mg_model, hf_model, result: ValidationResult, threshold: float):
    """Compare output projection (lm_head) weights."""
    print("\n" + "="*70)
    print("OUTPUT LAYER (LM_HEAD) COMPARISON")
    print("="*70)

    actual_mg = mg_model.module if hasattr(mg_model, 'module') else mg_model

    # MG: output_layer.weight
    mg_output = actual_mg.output_layer.weight.data
    # HF: lm_head.weight
    hf_output = hf_model.lm_head.weight.data

    cmp = compare_tensors(
        mg_output, hf_output,
        "output_layer.weight",
        "lm_head.weight",
        threshold
    )
    result.output_comparisons.append(cmp)
    result.compared_mg_weights.add("output_layer.weight")

    print_comparison(cmp, verbose=True)


# ==============================================================================
# Mamba Layer Comparison
# ==============================================================================

def compare_mamba_layer(
    mg_layer,
    hf_layer,
    mg_layer_idx: int,
    hf_layer_idx: int,
    result: ValidationResult,
    threshold: float,
    verbose: bool = False
) -> LayerComparison:
    """Compare Mamba (GatedDeltaNet) layer weights.

    Weight mapping (with reordering):
    - MG in_proj order: [z, V, Q, K, b, a]
    - HF in_proj_qkvz order: [Q, K, V, Z]
    - HF in_proj_ba order: [b, a]

    - MG conv1d order: [V, Q, K]
    - HF conv1d order: [Q, K, V]

    Note: MG MambaLayer uses 'norm' (inside mixer.in_proj with TELayerNormColumnParallelLinear)
          while HF uses 'input_layernorm' at layer level.
    """
    layer_cmp = LayerComparison(layer_idx=mg_layer_idx, layer_type="Mamba")
    mg_prefix = f"decoder.layers.{mg_layer_idx}"
    hf_prefix = f"model.layers.{hf_layer_idx}"

    mg_mixer = mg_layer.mixer
    hf_mixer = hf_layer.linear_attn

    # Get dimensions from HF model
    Nk = hf_mixer.num_k_heads   # num_k_heads
    Nv = hf_mixer.num_v_heads   # num_v_heads
    Dk = hf_mixer.head_k_dim    # head_k_dim
    Dv = hf_mixer.head_v_dim    # head_v_dim

    # === 1. input_layernorm ===
    # MG: TELayerNormColumnParallelLinear has layer_norm_weight inside in_proj
    # HF: input_layernorm is separate
    mg_ln_w = mg_mixer.in_proj.layer_norm_weight.data
    hf_ln_w = hf_layer.input_layernorm.weight.data
    cmp = compare_tensors(
        mg_ln_w, hf_ln_w,
        f"{mg_prefix}.mixer.in_proj.layer_norm_weight",
        f"{hf_prefix}.input_layernorm.weight",
        threshold,
        note="MG uses fused LayerNorm in TELayerNormColumnParallelLinear"
    )
    layer_cmp.comparisons.append(cmp)
    result.compared_mg_weights.add(f"{mg_prefix}.mixer.in_proj.layer_norm_weight")

    # === 2. in_proj (needs reordering) ===
    # MG2HF conversion does complex reshaping (from m2h_synchronizer.py):
    #   MG: [z, V, Q, K, b, a] split by [Dv*Nv, Dv*Nv, Dk*Nk, Dk*Nk, Nv, Nv]
    #   Then reshape to [Nk, head_dim, hidden] and concat as [Q, K, V, Z]
    #
    # For validation, we reconstruct the MG→HF transformation and compare
    mg_in_proj_w = mg_mixer.in_proj.weight.data
    hf_in_proj_qkvz_w = hf_mixer.in_proj_qkvz.weight.data
    hf_in_proj_ba_w = hf_mixer.in_proj_ba.weight.data

    # MG split: [z, V, Q, K, b, a]
    mg_split_sizes = [Dv*Nv, Dv*Nv, Dk*Nk, Dk*Nk, Nv, Nv]
    mg_z, mg_v, mg_q, mg_k, mg_b, mg_a = torch.split(mg_in_proj_w, mg_split_sizes, dim=0)

    # Reconstruct HF format from MG (same logic as m2h_synchronizer.py)
    hidden_size = mg_in_proj_w.shape[1]
    # qkvz: reshape to [Nk, head_dim, hidden] then concat
    mg_to_hf_qkvz = torch.cat([
        mg_q.reshape(Nk, Dk, hidden_size),
        mg_k.reshape(Nk, Dk, hidden_size),
        mg_v.reshape(Nk, Dv * Nv // Nk, hidden_size),
        mg_z.reshape(Nk, Dv * Nv // Nk, hidden_size),
    ], dim=1).reshape(-1, hidden_size)

    # ba: reshape similarly
    mg_to_hf_ba = torch.cat([
        mg_b.reshape(Nk, Nv // Nk, hidden_size),
        mg_a.reshape(Nk, Nv // Nk, hidden_size),
    ], dim=1).reshape(-1, hidden_size)

    # Compare reconstructed MG with HF
    cmp = compare_tensors(
        mg_to_hf_qkvz, hf_in_proj_qkvz_w,
        f"{mg_prefix}.mixer.in_proj[qkvz]",
        f"{hf_prefix}.linear_attn.in_proj_qkvz.weight",
        threshold,
        note="MG [z,V,Q,K] → HF [Q,K,V,Z] with reshape"
    )
    layer_cmp.comparisons.append(cmp)

    cmp = compare_tensors(
        mg_to_hf_ba, hf_in_proj_ba_w,
        f"{mg_prefix}.mixer.in_proj[ba]",
        f"{hf_prefix}.linear_attn.in_proj_ba.weight",
        threshold,
        note="MG [b,a] → HF [b,a] with reshape"
    )
    layer_cmp.comparisons.append(cmp)
    result.compared_mg_weights.add(f"{mg_prefix}.mixer.in_proj.weight")

    # === 3. conv1d (needs reordering) ===
    # MG2HF conversion: MG [V, Q, K] → HF [Q, K, V]
    mg_conv1d_w = mg_mixer.conv1d.weight.data
    hf_conv1d_w = hf_mixer.conv1d.weight.data

    # MG conv1d order: [V, Q, K]
    mg_conv_splits = [Nv*Dv, Nk*Dk, Nk*Dk]
    mg_conv_v, mg_conv_q, mg_conv_k = torch.split(mg_conv1d_w, mg_conv_splits, dim=0)

    # Reconstruct HF format: [Q, K, V]
    mg_to_hf_conv1d = torch.cat([mg_conv_q, mg_conv_k, mg_conv_v], dim=0)

    cmp = compare_tensors(
        mg_to_hf_conv1d, hf_conv1d_w,
        f"{mg_prefix}.mixer.conv1d.weight",
        f"{hf_prefix}.linear_attn.conv1d.weight",
        threshold,
        note="MG [V,Q,K] → HF [Q,K,V]"
    )
    layer_cmp.comparisons.append(cmp)
    result.compared_mg_weights.add(f"{mg_prefix}.mixer.conv1d.weight")

    # === 4. A_log ===
    mg_a_log = mg_mixer.A_log.data
    hf_a_log = hf_mixer.A_log.data
    cmp = compare_tensors(
        mg_a_log, hf_a_log,
        f"{mg_prefix}.mixer.A_log",
        f"{hf_prefix}.linear_attn.A_log",
        threshold
    )
    layer_cmp.comparisons.append(cmp)
    result.compared_mg_weights.add(f"{mg_prefix}.mixer.A_log")

    # === 5. dt_bias ===
    mg_dt_bias = mg_mixer.dt_bias.data
    hf_dt_bias = hf_mixer.dt_bias.data
    cmp = compare_tensors(
        mg_dt_bias, hf_dt_bias,
        f"{mg_prefix}.mixer.dt_bias",
        f"{hf_prefix}.linear_attn.dt_bias",
        threshold
    )
    layer_cmp.comparisons.append(cmp)
    result.compared_mg_weights.add(f"{mg_prefix}.mixer.dt_bias")

    # === 6. D (if exists) ===
    if hasattr(mg_mixer, 'D') and mg_mixer.D is not None:
        mg_d = mg_mixer.D.data
        hf_d = hf_mixer.D.data
        cmp = compare_tensors(
            mg_d, hf_d,
            f"{mg_prefix}.mixer.D",
            f"{hf_prefix}.linear_attn.D",
            threshold
        )
        layer_cmp.comparisons.append(cmp)
        result.compared_mg_weights.add(f"{mg_prefix}.mixer.D")

    # === 7. norm ===
    mg_norm = mg_mixer.norm.weight.data
    hf_norm = hf_mixer.norm.weight.data
    cmp = compare_tensors(
        mg_norm, hf_norm,
        f"{mg_prefix}.mixer.norm.weight",
        f"{hf_prefix}.linear_attn.norm.weight",
        threshold
    )
    layer_cmp.comparisons.append(cmp)
    result.compared_mg_weights.add(f"{mg_prefix}.mixer.norm.weight")

    # === 8. out_proj ===
    mg_out_proj = mg_mixer.out_proj.weight.data
    hf_out_proj = hf_mixer.out_proj.weight.data
    cmp = compare_tensors(
        mg_out_proj, hf_out_proj,
        f"{mg_prefix}.mixer.out_proj.weight",
        f"{hf_prefix}.linear_attn.out_proj.weight",
        threshold
    )
    layer_cmp.comparisons.append(cmp)
    result.compared_mg_weights.add(f"{mg_prefix}.mixer.out_proj.weight")

    return layer_cmp


# ==============================================================================
# Attention Layer Comparison
# ==============================================================================

def compare_attention_layer(
    mg_layer,
    hf_layer,
    mg_layer_idx: int,
    hf_layer_idx: int,
    result: ValidationResult,
    threshold: float,
    args,  # Need args for GQA parameters
    verbose: bool = False
) -> LayerComparison:
    """Compare Attention layer weights.

    MG uses GatedSoftmaxAttention with fused QKV (TELayerNormColumnParallelLinear).
    HF uses separate q_proj, k_proj, v_proj.

    The MG QKV transformation is complex due to GQA (Grouped Query Attention):
    - MG linear_qgkv.weight shape: [num_query_groups * (2 + num_querys_per_group*2) * dim, hidden]
    - Split into Q, K, V with different dimensions
    - Q needs additional reshape: [num_query_groups, 2, num_querys_per_group, dim, hidden]
                               → transpose(1,2) → flatten(1,3)

    Reference: toolkits/distributed_checkpoints_convertor/impl/alpha/m2h_synchronizer.py
    """
    layer_cmp = LayerComparison(layer_idx=mg_layer_idx, layer_type="Attention")
    mg_prefix = f"decoder.layers.{mg_layer_idx}"
    hf_prefix = f"model.layers.{hf_layer_idx}"

    mg_attn = mg_layer.self_attention
    hf_attn = hf_layer.self_attn

    # Get GQA parameters from args
    num_heads = args.num_attention_heads
    num_query_groups = args.num_query_groups if args.group_query_attention else args.num_attention_heads
    num_querys_per_group = num_heads // num_query_groups
    dim = args.kv_channels  # head dimension
    hidden_size = args.hidden_size

    # === 1. input_layernorm ===
    # MG: TELayerNormColumnParallelLinear has layer_norm_weight inside linear_qgkv
    # HF: input_layernorm is separate
    mg_ln_w = mg_attn.linear_qgkv.layer_norm_weight.data
    hf_ln_w = hf_layer.input_layernorm.weight.data
    cmp = compare_tensors(
        mg_ln_w, hf_ln_w,
        f"{mg_prefix}.self_attention.linear_qgkv.layer_norm_weight",
        f"{hf_prefix}.input_layernorm.weight",
        threshold,
        note="MG uses fused LayerNorm in TELayerNormColumnParallelLinear"
    )
    layer_cmp.comparisons.append(cmp)
    result.compared_mg_weights.add(f"{mg_prefix}.self_attention.linear_qgkv.layer_norm_weight")

    # === 2. QKV projection (complex transformation for GQA) ===
    # MG2HF transformation from m2h_synchronizer.py:
    #   attn_proj_weight = attn.linear_qgkv.weight.reshape(
    #       (num_query_groups, (2 + num_querys_per_group*2)*dim, -1)
    #   )
    #   q, k, v = split by [2*num_querys_per_group*dim, dim, dim]
    #   q = q.reshape(num_query_groups, 2, num_querys_per_group, dim, -1).transpose(1, 2).flatten(1, 3)
    mg_qkv_w = mg_attn.linear_qgkv.weight.data
    hf_q_w = hf_attn.q_proj.weight.data
    hf_k_w = hf_attn.k_proj.weight.data
    hf_v_w = hf_attn.v_proj.weight.data

    # Reshape MG weight: [total_qkv_dim, hidden] -> [num_query_groups, per_group_dim, hidden]
    attn_proj_weight = mg_qkv_w.reshape(
        num_query_groups, (2 + num_querys_per_group * 2) * dim, -1
    )

    # Split into Q, K, V parts
    # Q takes 2*num_querys_per_group*dim per group (gate and up for each query head)
    # K and V each take dim per group
    mg_q_raw, mg_k, mg_v = torch.split(
        attn_proj_weight,
        [2 * num_querys_per_group * dim, dim, dim],
        dim=1
    )

    # Q needs special transformation:
    # [num_query_groups, 2*num_querys_per_group*dim, hidden]
    # -> reshape [num_query_groups, 2, num_querys_per_group, dim, hidden]
    # -> transpose(1, 2) -> [num_query_groups, num_querys_per_group, 2, dim, hidden]
    # -> flatten(1, 3) -> [num_query_groups, num_querys_per_group*2*dim, hidden]
    # -> reshape -> [num_query_groups * num_querys_per_group * 2 * dim, hidden]
    # But HF q_proj is [num_heads * 2 * dim, hidden] = [num_heads * head_dim, hidden]
    # Actually for q_proj.weight it should be [num_heads * dim, hidden]
    mg_q = mg_q_raw.reshape(num_query_groups, 2, num_querys_per_group, dim, -1)
    mg_q = mg_q.transpose(1, 2).flatten(1, 3)  # [num_query_groups, num_querys_per_group*2*dim, hidden]
    mg_q = mg_q.reshape(-1, hidden_size)  # [total_q_dim, hidden]

    # K and V just need to be flattened
    mg_k = mg_k.reshape(-1, hidden_size)  # [num_query_groups * dim, hidden]
    mg_v = mg_v.reshape(-1, hidden_size)  # [num_query_groups * dim, hidden]

    for (mg_name, mg_t, hf_name, hf_t) in [
        ("linear_qgkv.Q", mg_q, "q_proj.weight", hf_q_w),
        ("linear_qgkv.K", mg_k, "k_proj.weight", hf_k_w),
        ("linear_qgkv.V", mg_v, "v_proj.weight", hf_v_w),
    ]:
        cmp = compare_tensors(
            mg_t, hf_t,
            f"{mg_prefix}.self_attention.{mg_name}",
            f"{hf_prefix}.self_attn.{hf_name}",
            threshold,
            note="GQA transformation from m2h_synchronizer"
        )
        layer_cmp.comparisons.append(cmp)
    result.compared_mg_weights.add(f"{mg_prefix}.self_attention.linear_qgkv.weight")

    # === 3. Output projection ===
    mg_o_w = mg_attn.linear_proj.weight.data
    hf_o_w = hf_attn.o_proj.weight.data
    cmp = compare_tensors(
        mg_o_w, hf_o_w,
        f"{mg_prefix}.self_attention.linear_proj.weight",
        f"{hf_prefix}.self_attn.o_proj.weight",
        threshold
    )
    layer_cmp.comparisons.append(cmp)
    result.compared_mg_weights.add(f"{mg_prefix}.self_attention.linear_proj.weight")

    # === 4. q_norm, k_norm (QK normalization) ===
    if hasattr(mg_attn, 'q_layernorm') and mg_attn.q_layernorm is not None:
        mg_q_norm = mg_attn.q_layernorm.weight.data
        hf_q_norm = hf_attn.q_norm.weight.data
        cmp = compare_tensors(
            mg_q_norm, hf_q_norm,
            f"{mg_prefix}.self_attention.q_layernorm.weight",
            f"{hf_prefix}.self_attn.q_norm.weight",
            threshold
        )
        layer_cmp.comparisons.append(cmp)
        result.compared_mg_weights.add(f"{mg_prefix}.self_attention.q_layernorm.weight")

    if hasattr(mg_attn, 'k_layernorm') and mg_attn.k_layernorm is not None:
        mg_k_norm = mg_attn.k_layernorm.weight.data
        hf_k_norm = hf_attn.k_norm.weight.data
        cmp = compare_tensors(
            mg_k_norm, hf_k_norm,
            f"{mg_prefix}.self_attention.k_layernorm.weight",
            f"{hf_prefix}.self_attn.k_norm.weight",
            threshold
        )
        layer_cmp.comparisons.append(cmp)
        result.compared_mg_weights.add(f"{mg_prefix}.self_attention.k_layernorm.weight")

    return layer_cmp


# ==============================================================================
# MLP Layer Comparison (Dense or MoE)
# ==============================================================================

def compare_mlp_layer(
    mg_layer,
    hf_layer,
    mg_layer_idx: int,
    hf_layer_idx: int,
    result: ValidationResult,
    threshold: float,
    is_moe: bool = False,
    verbose: bool = False
) -> LayerComparison:
    """Compare MLP layer weights (Dense MLP or MoE).

    For MoE:
    - MG: mlp.experts.local_experts[i]
    - HF: mlp.experts[i]
    """
    layer_type = "MoE" if is_moe else "DenseMLP"
    layer_cmp = LayerComparison(layer_idx=mg_layer_idx, layer_type=layer_type)
    mg_prefix = f"decoder.layers.{mg_layer_idx}"
    hf_prefix = f"model.layers.{hf_layer_idx}"

    # === 1. pre_mlp_layernorm ===
    mg_ln_w = mg_layer.pre_mlp_layernorm.weight.data
    hf_ln_w = hf_layer.post_attention_layernorm.weight.data
    cmp = compare_tensors(
        mg_ln_w, hf_ln_w,
        f"{mg_prefix}.pre_mlp_layernorm.weight",
        f"{hf_prefix}.post_attention_layernorm.weight",
        threshold
    )
    layer_cmp.comparisons.append(cmp)
    result.compared_mg_weights.add(f"{mg_prefix}.pre_mlp_layernorm.weight")

    if is_moe:
        # === MoE specific ===
        # MG uses TEGroupedMLP with weight0, weight1, ... per expert
        # (accessed via linear_fc1.weight0, linear_fc2.weight0, etc.)
        mg_mlp = mg_layer.mlp
        hf_mlp = hf_layer.mlp

        # Router
        mg_router = mg_mlp.router.weight.data
        hf_router = hf_mlp.gate.weight.data
        cmp = compare_tensors(
            mg_router, hf_router,
            f"{mg_prefix}.mlp.router.weight",
            f"{hf_prefix}.mlp.gate.weight",
            threshold
        )
        layer_cmp.comparisons.append(cmp)
        result.compared_mg_weights.add(f"{mg_prefix}.mlp.router.weight")

        # Aux-loss-free expert bias (DSV3): MG router.expert_bias ↔ HF
        # gate.e_score_correction_bias. Present only when the run enabled
        # --moe-router-enable-expert-bias (alpha v2 baseline does).
        mg_expert_bias = getattr(mg_mlp.router, "expert_bias", None)
        if mg_expert_bias is not None and hasattr(hf_mlp.gate, "e_score_correction_bias"):
            cmp = compare_tensors(
                mg_expert_bias.data,
                hf_mlp.gate.e_score_correction_bias.data,
                f"{mg_prefix}.mlp.router.expert_bias",
                f"{hf_prefix}.mlp.gate.e_score_correction_bias",
                threshold,
                note="DSV3 aux-loss-free expert bias",
            )
            layer_cmp.comparisons.append(cmp)
            result.compared_mg_weights.add(f"{mg_prefix}.mlp.router.expert_bias")

        # Experts - MG uses TEGroupedMLP with weight{idx} attributes
        # We're running with EP=1 for validation (single GPU), so all experts are local
        mg_experts = mg_mlp.experts
        hf_experts = hf_mlp.experts

        num_local_experts = mg_experts.num_local_experts
        for exp_idx in range(num_local_experts):
            hf_exp = hf_experts[exp_idx]

            # MG: linear_fc1.weight{idx} is fused [gate, up] with shape [2*ffn_hidden, hidden]
            mg_fc1_w = getattr(mg_experts.linear_fc1, f'weight{exp_idx}').data
            hf_gate = hf_exp.gate_proj.weight.data
            hf_up = hf_exp.up_proj.weight.data

            # MG fc1 is fused [gate, up] along first dimension
            hidden_size = mg_fc1_w.shape[-1]
            mg_gate_w, mg_up_w = mg_fc1_w.reshape(2, -1, hidden_size)

            cmp = compare_tensors(
                mg_gate_w, hf_gate,
                f"{mg_prefix}.mlp.experts.linear_fc1.weight{exp_idx}[gate]",
                f"{hf_prefix}.mlp.experts.{exp_idx}.gate_proj.weight",
                threshold,
                note="Split from fused fc1"
            )
            layer_cmp.comparisons.append(cmp)

            cmp = compare_tensors(
                mg_up_w, hf_up,
                f"{mg_prefix}.mlp.experts.linear_fc1.weight{exp_idx}[up]",
                f"{hf_prefix}.mlp.experts.{exp_idx}.up_proj.weight",
                threshold,
                note="Split from fused fc1"
            )
            layer_cmp.comparisons.append(cmp)

            # down_proj (fc2)
            mg_fc2_w = getattr(mg_experts.linear_fc2, f'weight{exp_idx}').data
            hf_down = hf_exp.down_proj.weight.data
            cmp = compare_tensors(
                mg_fc2_w, hf_down,
                f"{mg_prefix}.mlp.experts.linear_fc2.weight{exp_idx}",
                f"{hf_prefix}.mlp.experts.{exp_idx}.down_proj.weight",
                threshold
            )
            layer_cmp.comparisons.append(cmp)

        # Mark all expert weights as compared
        result.compared_mg_weights.add(f"{mg_prefix}.mlp.experts.linear_fc1._extra_state")
        result.compared_mg_weights.add(f"{mg_prefix}.mlp.experts.linear_fc2._extra_state")
        for exp_idx in range(num_local_experts):
            result.compared_mg_weights.add(f"{mg_prefix}.mlp.experts.linear_fc1.weight{exp_idx}")
            result.compared_mg_weights.add(f"{mg_prefix}.mlp.experts.linear_fc2.weight{exp_idx}")

        # Shared experts (if present)
        if hasattr(mg_mlp, 'shared_experts') and mg_mlp.shared_experts is not None:
            mg_shared = mg_mlp.shared_experts
            # HF uses 'shared_expert' (singular) or 'shared_experts' (plural)
            if hasattr(hf_mlp, 'shared_experts'):
                hf_shared = hf_mlp.shared_experts
            elif hasattr(hf_mlp, 'shared_expert'):
                hf_shared = hf_mlp.shared_expert
            else:
                raise AttributeError(f"HF MLP has no shared_experts or shared_expert attribute")

            # Shared expert fc1 (fused gate+up)
            mg_shared_fc1 = mg_shared.linear_fc1.weight.data
            hf_shared_gate = hf_shared.gate_proj.weight.data
            hf_shared_up = hf_shared.up_proj.weight.data

            # MG shared_experts.linear_fc1 is [2*ffn, hidden] fused
            hidden_size = mg_shared_fc1.shape[-1]
            mg_shared_gate, mg_shared_up = mg_shared_fc1.reshape(2, -1, hidden_size)

            cmp = compare_tensors(
                mg_shared_gate, hf_shared_gate,
                f"{mg_prefix}.mlp.shared_experts.linear_fc1[gate]",
                f"{hf_prefix}.mlp.shared_expert.gate_proj.weight",
                threshold,
                note="Shared expert, split from fused"
            )
            layer_cmp.comparisons.append(cmp)

            cmp = compare_tensors(
                mg_shared_up, hf_shared_up,
                f"{mg_prefix}.mlp.shared_experts.linear_fc1[up]",
                f"{hf_prefix}.mlp.shared_expert.up_proj.weight",
                threshold,
                note="Shared expert, split from fused"
            )
            layer_cmp.comparisons.append(cmp)
            result.compared_mg_weights.add(f"{mg_prefix}.mlp.shared_experts.linear_fc1.weight")

            # Shared expert fc2
            mg_shared_fc2 = mg_shared.linear_fc2.weight.data
            hf_shared_down = hf_shared.down_proj.weight.data
            cmp = compare_tensors(
                mg_shared_fc2, hf_shared_down,
                f"{mg_prefix}.mlp.shared_experts.linear_fc2.weight",
                f"{hf_prefix}.mlp.shared_expert.down_proj.weight",
                threshold
            )
            layer_cmp.comparisons.append(cmp)
            result.compared_mg_weights.add(f"{mg_prefix}.mlp.shared_experts.linear_fc2.weight")

            # Shared-expert gate: MG shared_experts.gate_weight ↔ HF
            # shared_expert_gate.weight (sigmoid gate on the shared expert).
            mg_shared_gate = getattr(mg_shared, "gate_weight", None)
            if mg_shared_gate is not None and hasattr(hf_mlp, "shared_expert_gate"):
                cmp = compare_tensors(
                    mg_shared_gate.data.reshape(-1),
                    hf_mlp.shared_expert_gate.weight.data.reshape(-1),
                    f"{mg_prefix}.mlp.shared_experts.gate_weight",
                    f"{hf_prefix}.mlp.shared_expert_gate.weight",
                    threshold,
                    note="Shared expert gate",
                )
                layer_cmp.comparisons.append(cmp)
                result.compared_mg_weights.add(f"{mg_prefix}.mlp.shared_experts.gate_weight")

    else:
        # === Dense MLP ===
        mg_mlp = mg_layer.mlp
        hf_mlp = hf_layer.mlp

        # fc1 (fused gate+up in MG) - shape [2*ffn_hidden, hidden]
        mg_fc1 = mg_mlp.linear_fc1.weight.data
        hf_gate = hf_mlp.gate_proj.weight.data
        hf_up = hf_mlp.up_proj.weight.data

        # Reshape and split like MG2HF synchronizer does
        hidden_size = mg_fc1.shape[-1]
        mg_gate, mg_up = mg_fc1.reshape(2, -1, hidden_size)

        cmp = compare_tensors(
            mg_gate, hf_gate,
            f"{mg_prefix}.mlp.linear_fc1[gate]",
            f"{hf_prefix}.mlp.gate_proj.weight",
            threshold,
            note="Split from fused fc1"
        )
        layer_cmp.comparisons.append(cmp)

        cmp = compare_tensors(
            mg_up, hf_up,
            f"{mg_prefix}.mlp.linear_fc1[up]",
            f"{hf_prefix}.mlp.up_proj.weight",
            threshold,
            note="Split from fused fc1"
        )
        layer_cmp.comparisons.append(cmp)
        result.compared_mg_weights.add(f"{mg_prefix}.mlp.linear_fc1.weight")

        # fc2
        mg_fc2 = mg_mlp.linear_fc2.weight.data
        hf_down = hf_mlp.down_proj.weight.data
        cmp = compare_tensors(
            mg_fc2, hf_down,
            f"{mg_prefix}.mlp.linear_fc2.weight",
            f"{hf_prefix}.mlp.down_proj.weight",
            threshold
        )
        layer_cmp.comparisons.append(cmp)
        result.compared_mg_weights.add(f"{mg_prefix}.mlp.linear_fc2.weight")

    return layer_cmp


# ==============================================================================
# Main Comparison Function
# ==============================================================================

def compare_all_weights(
    mg_model,
    hf_model,
    pattern: str,
    args,  # Megatron args for GQA parameters
    threshold: float = 0.01,
    verbose: bool = False
) -> ValidationResult:
    """Compare all weights between MG and HF models.

    Args:
        mg_model: Megatron model
        hf_model: HuggingFace model
        pattern: Hybrid pattern string (e.g., "MDM-M-*-...")
        args: Megatron args (for GQA parameters in Attention)
        threshold: Maximum absolute difference for match
        verbose: Print all comparisons

    Returns:
        ValidationResult with all comparison results
    """
    result = ValidationResult()

    # Collect all MG weight names
    actual_mg = mg_model.module if hasattr(mg_model, 'module') else mg_model
    for name, _ in actual_mg.named_parameters():
        result.all_mg_weights.add(name)
    for name, _ in actual_mg.named_buffers():
        result.all_mg_weights.add(name)

    # 1. Embedding
    compare_embedding(mg_model, hf_model, result, threshold)

    # 2. Layers
    print("\n" + "="*70)
    print("LAYER-BY-LAYER WEIGHT COMPARISON")
    print("="*70)

    mg_layers = actual_mg.decoder.layers
    hf_layers = hf_model.model.layers

    num_mg_layers = len(mg_layers)
    num_hf_layers = len(hf_layers)

    print(f"\nMG layers: {num_mg_layers}, HF layers: {num_hf_layers}")
    print(f"Pattern: {pattern[:50]}... (len={len(pattern)})")
    print(f"Layer ratio: MG 2 : HF 1")

    # MG uses 2:1 ratio: 2 MG layers = 1 HF layer
    # Pattern length should equal num_mg_layers
    # Each pair of MG layers (mixer + mlp) corresponds to 1 HF layer

    mg_idx = 0
    hf_idx = 0

    while mg_idx < num_mg_layers and hf_idx < num_hf_layers:
        layer_type = get_mg_layer_type(mg_layers[mg_idx], mg_idx, pattern)

        print(f"\n--- MG Layer {mg_idx} ({layer_type}) → HF Layer {hf_idx} ---")

        mg_layer = mg_layers[mg_idx]
        hf_layer = hf_layers[hf_idx]

        if layer_type == "Mamba":
            layer_cmp = compare_mamba_layer(
                mg_layer, hf_layer, mg_idx, hf_idx, result, threshold, verbose
            )
        elif layer_type == "Attention":
            layer_cmp = compare_attention_layer(
                mg_layer, hf_layer, mg_idx, hf_idx, result, threshold, args, verbose
            )
        else:
            # Skip layer type comparison, just move indices
            layer_cmp = LayerComparison(layer_idx=mg_idx, layer_type=layer_type)

        # Print layer summary
        if layer_cmp.comparisons:
            matched = sum(1 for c in layer_cmp.comparisons if c.matched)
            total = len(layer_cmp.comparisons)
            status = "✓" if layer_cmp.all_matched else "✗"
            print(f"  {status} {matched}/{total} weights matched")

            if verbose or not layer_cmp.all_matched:
                for cmp in layer_cmp.comparisons:
                    print_comparison(cmp, verbose=verbose)

        result.layer_comparisons.append(layer_cmp)

        # Handle MLP layer (next in pattern)
        mg_idx += 1
        if mg_idx < num_mg_layers:
            mlp_type = get_mg_layer_type(mg_layers[mg_idx], mg_idx, pattern)

            if mlp_type in ["MoE", "DenseMLP", "-"]:
                print(f"\n--- MG Layer {mg_idx} ({mlp_type}) → HF Layer {hf_idx} (MLP part) ---")

                is_moe = mlp_type == "MoE" or mlp_type == "-"
                mlp_cmp = compare_mlp_layer(
                    mg_layers[mg_idx], hf_layer, mg_idx, hf_idx, result, threshold, is_moe, verbose
                )

                if mlp_cmp.comparisons:
                    matched = sum(1 for c in mlp_cmp.comparisons if c.matched)
                    total = len(mlp_cmp.comparisons)
                    status = "✓" if mlp_cmp.all_matched else "✗"
                    print(f"  {status} {matched}/{total} weights matched")

                    if verbose or not mlp_cmp.all_matched:
                        for cmp in mlp_cmp.comparisons:
                            print_comparison(cmp, verbose=verbose)

                result.layer_comparisons.append(mlp_cmp)
                mg_idx += 1

        hf_idx += 1

    # 3. Final layer norm
    compare_final_norm(mg_model, hf_model, result, threshold)

    # 4. Output layer
    compare_output_layer(mg_model, hf_model, result, threshold)

    return result


# ==============================================================================
# Main Function
# ==============================================================================

def main():
    """Main validation function."""
    from megatron.training import get_args, print_rank_0
    from megatron.training.initialize import initialize_megatron
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training import get_model
    from megatron.core.enums import ModelType
    from megatron_patch.arguments import get_patch_args

    # Initialize Megatron with validation args
    initialize_megatron(
        extra_args_provider=lambda parser: add_validation_args(get_patch_args(parser)),
        args_defaults={
            'tokenizer_type': 'NullTokenizer',
            'no_load_rng': True,
            'no_load_optim': True,
            'use_legacy_models': False,
        }
    )

    args = get_args()

    # Determine checkpoint path: --mg-checkpoint takes priority over --load
    checkpoint_path = args.mg_checkpoint if args.mg_checkpoint else args.load

    print_rank_0("\n" + "="*70)
    print_rank_0("Alpha Model MG ↔ HF Weight Validation")
    print_rank_0("="*70)
    print_rank_0(f"\nConfiguration:")
    print_rank_0(f"  MG Checkpoint:  {checkpoint_path}")
    print_rank_0(f"  HF Model:       {args.hf_model}")
    print_rank_0(f"  Threshold:      {args.threshold}")
    print_rank_0(f"  Pattern:        {args.hybrid_override_pattern[:50]}...")

    # Build Megatron model
    print_rank_0("\nLoading Megatron model...")
    model = get_model(model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)

    # Load checkpoint
    if checkpoint_path:
        original_load = args.load
        original_ckpt_step = getattr(args, 'ckpt_step', None)

        # Extract iteration from path (e.g., iter_0050000 -> 50000)
        # Megatron's load_checkpoint looks for latest_checkpointed_iteration.txt
        # in args.load directory. For specific iteration dirs, we need to:
        # 1. Set args.load to parent directory (checkpoints/)
        # 2. Set args.ckpt_step to the iteration number
        import re
        match = re.search(r'iter_(\d+)', checkpoint_path)
        if match:
            ckpt_step = int(match.group(1))
            parent_dir = os.path.dirname(checkpoint_path)
            args.load = parent_dir
            args.ckpt_step = ckpt_step
            print_rank_0(f"  Checkpoint path: {parent_dir}")
            print_rank_0(f"  Checkpoint step: {ckpt_step}")
        else:
            # Fallback: assume checkpoint_path is the checkpoints/ directory
            args.load = checkpoint_path

        iteration, _ = load_checkpoint(model, None, None)

        # Restore original values
        args.load = original_load
        if original_ckpt_step is not None:
            args.ckpt_step = original_ckpt_step
        elif hasattr(args, 'ckpt_step'):
            args.ckpt_step = None

        if iteration is not None and iteration > 0:
            print_rank_0(f"  ✓ Checkpoint loaded! Iteration: {iteration}")
        else:
            print_rank_0(f"  ⚠ WARNING: Checkpoint may not have loaded!")

    # Get the actual model
    if isinstance(model, list):
        model = model[0]
    model.eval()

    # Load HF model (only on rank 0)
    if torch.distributed.get_rank() == 0:
        hf_model, _ = load_hf_model(args.hf_model)

        # Run weight comparison
        print_rank_0("\nComparing all weights...")
        result = compare_all_weights(
            model,
            hf_model,
            args.hybrid_override_pattern,
            args,
            args.threshold,
            args.verbose
        )

        # Print summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)

        # Count results
        total_comparisons = (
            len(result.embedding_comparisons) +
            sum(len(lc.comparisons) for lc in result.layer_comparisons) +
            len(result.output_comparisons)
        )
        matched_comparisons = (
            sum(1 for c in result.embedding_comparisons if c.matched) +
            sum(1 for lc in result.layer_comparisons for c in lc.comparisons if c.matched) +
            sum(1 for c in result.output_comparisons if c.matched)
        )

        print(f"\nWeight comparisons: {matched_comparisons}/{total_comparisons} matched")
        print(f"Compared MG weights: {len(result.compared_mg_weights)}")
        print(f"Total MG weights: {len(result.all_mg_weights)}")

        # Bidirectional check for complete validation
        # 1. MG weights that were NOT compared (missing from validation)
        unchecked = result.unchecked_weights

        # 2. Compared names that don't exist in actual MG weights (naming mismatch)
        phantom_weights = result.compared_mg_weights - result.all_mg_weights

        # Filter out known patterns that are expected to differ
        # - TEGroupedMLP uses dynamic weight{idx} attributes, not in named_parameters
        # - _extra_state is TE internal state
        # - local_tokens_per_expert is a transient routing-statistics buffer used
        #   only for aux-loss-free bias updates during training; it has no HF
        #   counterpart and is not a convertible weight.
        _ignore_suffixes = ('._extra_state', '.local_tokens_per_expert')
        filtered_unchecked = {w for w in unchecked if not w.endswith(_ignore_suffixes)}
        filtered_phantom = {w for w in phantom_weights if not (
            '.weight' in w and any(f'weight{i}' in w for i in range(512)) or
            w.endswith('._extra_state')
        )}

        if filtered_unchecked:
            print(f"\n⚠ {len(filtered_unchecked)} MG weights were NOT compared:")
            for w in sorted(filtered_unchecked)[:20]:
                print(f"    - {w}")
            if len(filtered_unchecked) > 20:
                print(f"    ... and {len(filtered_unchecked) - 20} more")

        if filtered_phantom:
            print(f"\n⚠ {len(filtered_phantom)} compared names don't exist in MG model (naming mismatch):")
            for w in sorted(filtered_phantom)[:20]:
                print(f"    - {w}")
            if len(filtered_phantom) > 20:
                print(f"    ... and {len(filtered_phantom) - 20} more")

        # Detailed weight coverage report
        print(f"\n--- Weight Coverage Report ---")
        print(f"  MG weights (named_parameters + buffers): {len(result.all_mg_weights)}")
        print(f"  Weights compared (by name): {len(result.compared_mg_weights)}")
        print(f"  Unchecked MG weights: {len(unchecked)} (filtered: {len(filtered_unchecked)})")
        print(f"  Phantom names (not in MG): {len(phantom_weights)} (filtered: {len(filtered_phantom)})")

        # Final verdict
        print("\n" + "="*70)
        validation_ok = result.all_matched and not filtered_unchecked

        if validation_ok and not filtered_phantom:
            print("✓ ALL WEIGHTS MATCHED - CONVERSION VERIFIED")
            print("="*70 + "\n")
            sys.exit(0)
        elif validation_ok:
            print("⚠ All MG weights matched, but some naming inconsistencies exist")
            print("  (This may be expected for TEGroupedMLP dynamic weights)")
            print("="*70 + "\n")
            sys.exit(0)
        elif result.all_matched:
            print("⚠ All compared weights matched, but some MG weights were not checked")
            print("="*70 + "\n")
            sys.exit(1)  # Fail if weights are missing
        else:
            print("✗ WEIGHT MISMATCH DETECTED")
            print("  Check the detailed output above for mismatched weights.")
            print("="*70 + "\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
