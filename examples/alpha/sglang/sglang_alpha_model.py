"""SGLang model adapter for Alpha (GatedDeltaNet + Attention + MoE hybrid).

Alpha is based on Qwen3-Next architecture with one key extension: mlp_only_layers.
This adapter extends SGLang's Qwen3NextForCausalLM to support per-layer Dense MLP
selection, while reusing all existing hybrid optimizations (MambaRadixCache,
dual memory pool, HybridLinearAttnBackend).

Installation:
    cp sglang_alpha_model.py $(python -c 'import sglang; print(sglang.__path__[0])')/srt/models/alpha.py

The model is auto-registered via EntryClass for architectures: ["AlphaForCausalLM"].
"""

from typing import Iterable, Optional, Set, Tuple

import torch

from sglang.srt.configs.qwen3_next import Qwen3NextConfig
from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.srt.layers.quantization import QuantizationConfig


def _alpha_config_to_qwen3next(hf_config) -> Qwen3NextConfig:
    """Convert Alpha HF config to SGLang Qwen3NextConfig.

    SGLang's Qwen3NextConfig has dynamic properties (layers_block_type,
    hybrid_gdn_params, mamba_cache_per_req) that AlphaConfig lacks.
    This function creates a proper Qwen3NextConfig with all Alpha values.
    """
    # Extract all relevant fields from the HF config
    kwargs = {}
    fields = [
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "hidden_act",
        "max_position_embeddings", "initializer_range", "rms_norm_eps",
        "use_cache", "tie_word_embeddings", "rope_theta", "rope_scaling",
        "partial_rotary_factor", "attention_bias", "attention_dropout",
        # Linear attention (GatedDeltaNet)
        "linear_conv_kernel_dim", "linear_key_head_dim", "linear_value_head_dim",
        "linear_num_key_heads", "linear_num_value_heads",
        # MoE
        "decoder_sparse_step", "moe_intermediate_size",
        "shared_expert_intermediate_size", "num_experts_per_tok",
        "num_experts", "norm_topk_prob", "output_router_logits",
        "router_aux_loss_coef",
    ]

    for field in fields:
        if hasattr(hf_config, field):
            kwargs[field] = getattr(hf_config, field)

    # full_attention_interval: derive from layer_types or use default
    if hasattr(hf_config, "full_attention_interval"):
        kwargs["full_attention_interval"] = hf_config.full_attention_interval
    elif hasattr(hf_config, "layer_types"):
        # Count interval from layer_types
        layer_types = hf_config.layer_types
        attn_indices = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
        if len(attn_indices) >= 2:
            kwargs["full_attention_interval"] = attn_indices[1] - attn_indices[0]
        elif len(attn_indices) == 1:
            kwargs["full_attention_interval"] = attn_indices[0] + 1
        else:
            kwargs["full_attention_interval"] = kwargs.get("num_hidden_layers", 24)

    qwen3_config = Qwen3NextConfig(**kwargs)

    # Preserve mlp_only_layers on the config
    qwen3_config.mlp_only_layers = getattr(hf_config, "mlp_only_layers", [])

    # Preserve architectures for model registry lookup
    qwen3_config.architectures = getattr(hf_config, "architectures", ["AlphaForCausalLM"])
    # NOTE: SGLang's is_hybrid_gdn check needs "AlphaForCausalLM" in its allowlist.
    # This is done by patching model_runner.py (see deploy.sh).

    return qwen3_config


class AlphaForCausalLM(Qwen3NextForCausalLM):
    """Alpha model for SGLang — extends Qwen3-Next with mlp_only_layers support.

    After base Qwen3-Next initialization, replaces MoE MLP modules with
    Dense MLP (Qwen2MoeMLP) for layers listed in config.mlp_only_layers.
    All other functionality (GatedDeltaNet, RadixAttention, hybrid memory,
    prefix caching) is inherited unchanged.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Convert Alpha HF config to SGLang Qwen3NextConfig
        mlp_only = getattr(config, "mlp_only_layers", [])
        qwen3_config = _alpha_config_to_qwen3next(config)

        # Initialize as Qwen3-Next (all layers get MoE MLP by default)
        super().__init__(qwen3_config, quant_config, prefix)

        # Store original config reference for load_weights
        self.config = qwen3_config

        # Replace MoE MLP with Dense MLP for mlp_only_layers
        if mlp_only:
            for layer_idx in mlp_only:
                if layer_idx < len(self.model.layers):
                    layer = self.model.layers[layer_idx]
                    layer.mlp = Qwen2MoeMLP(
                        hidden_size=qwen3_config.hidden_size,
                        intermediate_size=qwen3_config.intermediate_size,
                        hidden_act=qwen3_config.hidden_act,
                        quant_config=quant_config,
                        prefix=f"model.layers.{layer_idx}.mlp",
                    )
                    layer.is_layer_sparse = False

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> Set[str]:
        """Load weights with mlp_only_layers Dense MLP support.

        For Dense MLP layers (mlp_only_layers), weight names use the standard
        gate_proj/up_proj/down_proj format (no experts/shared_experts/gate).
        For MoE layers, delegates to Qwen3-Next's standard loading.
        """
        mlp_only = set(getattr(self.config, "mlp_only_layers", []))

        if not mlp_only:
            return super().load_weights(weights, is_mtp)

        # Separate Dense MLP weights from everything else
        dense_weights = []
        other_weights = []

        for name, loaded_weight in weights:
            is_dense = False
            if ".mlp." in name:
                # Dense MLP layers have gate_proj/up_proj/down_proj directly
                # (no .experts. or .shared_experts. or .gate.weight)
                has_experts = ".experts." in name or ".shared_experts." in name
                is_gate_weight = name.endswith(".mlp.gate.weight")
                if not has_experts and not is_gate_weight:
                    parts = name.split(".")
                    for i, p in enumerate(parts):
                        if p == "layers" and i + 1 < len(parts):
                            try:
                                layer_idx = int(parts[i + 1])
                                if layer_idx in mlp_only:
                                    is_dense = True
                            except ValueError:
                                pass
                            break

            if is_dense:
                dense_weights.append((name, loaded_weight))
            else:
                other_weights.append((name, loaded_weight))

        # Load MoE + non-MLP weights via parent
        loaded = super().load_weights(iter(other_weights), is_mtp)

        # Load Dense MLP weights (gate_proj + up_proj → gate_up_proj fusion)
        params_dict = dict(self.named_parameters())
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        stacked_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        for name, loaded_weight in dense_weights:
            handled = False
            for param_name, weight_name, shard_id in stacked_mapping:
                if weight_name not in name:
                    continue
                new_name = name.replace(weight_name, param_name)
                if new_name in params_dict:
                    param = params_dict[new_name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight, shard_id)
                    loaded.add(new_name)
                    handled = True
                    break

            if not handled and name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)

        return loaded


# SGLang Model Registry: auto-registered for architectures: ["AlphaForCausalLM"]
EntryClass = AlphaForCausalLM
