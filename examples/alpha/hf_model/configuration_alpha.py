# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Copyright 2025 Alpha Project Contributors
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
"""Alpha model configuration

Alpha is based on Qwen3-Next Mamba Hybrid architecture (GatedDeltaNet + Full Attention).
This configuration file is derived from Qwen3NextConfig for local version control
and customization independence from the transformers library.
"""

from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)


class AlphaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`AlphaModel`]. It is used to instantiate an
    Alpha model according to the specified arguments, defining the model architecture.

    Alpha is based on Qwen3-Next Mamba Hybrid architecture with the following key features:
    - GatedDeltaNet (Linear Attention) + Full Multi-Head Attention Hybrid
    - Mixture-of-Experts (MoE) with Shared Expert
    - 2:1 layer mapping (Megatron layers to HuggingFace layers)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 163968):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `inputs_ids`. Alpha v5 BBPE: effective 163,860 + 108 pad slots → 163,968 (multiple of 128).
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimension of the dense MLP representations (matches `ffn-hidden-size` in baseline_48L.yaml).
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 262144):
            The maximum sequence length that this model might ever be used with. Training context is 4096
            (`seq-length`); 262144 reflects the long-context envelope supported by RoPE θ=10M.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000000.0):
            The base period of the RoPE embeddings. Alpha uses θ=10M (frontier-LLM standard for long-context).
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
        partial_rotary_factor (`float`, *optional*, defaults to 0.25):
            Percentage of the query and keys which will have rotary embedding.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        head_dim (`int`, *optional*, defaults to 256):
            Projection weights dimension in multi-head attention.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Kernel size of the convolution used in linear attention layers.
        linear_key_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each key head in linear attention.
        linear_value_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each value head in linear attention.
        linear_num_key_heads (`int`, *optional*, defaults to 16):
            Number of key heads used in linear attention layers.
        linear_num_value_heads (`int`, *optional*, defaults to 32):
            Number of value heads used in linear attention layers.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer.
        moe_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of the routed expert.
        shared_expert_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of the shared expert.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of selected experts per token (matches `moe-router-topk` in baseline_48L.yaml).
        num_experts (`int`, *optional*, defaults to 192):
            Number of routed experts (DSV3-tuned for ~15B; 24 experts/GPU at EP=8).
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the topk probabilities.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.0001):
            The aux loss factor for the total loss (DSV3 — matches `moe-aux-loss-coeff` in baseline_48L.yaml).
        mlp_only_layers (`list[int]`, *optional*, defaults to `[]`):
            Indicate which layers use AlphaMLP rather than AlphaSparseMoeBlock.
        layer_types (`list[str]`, *optional*):
            Types of each layer (attention or linear).

    ```python
    >>> from transformers import AutoModel, AutoConfig

    >>> # Load Alpha model with trust_remote_code
    >>> config = AutoConfig.from_pretrained("path/to/alpha", trust_remote_code=True)
    >>> model = AutoModel.from_pretrained("path/to/alpha", trust_remote_code=True)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "alpha"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "colwise",
        "layers.*.mlp.experts.*.up_proj": "colwise",
        "layers.*.mlp.experts.*.down_proj": "rowwise",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=163968,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000000.0,
        rope_scaling=None,
        partial_rotary_factor=0.25,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        decoder_sparse_step=1,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=8,
        num_experts=192,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=1.0e-4,
        # DSV3-style routing (matches baseline_48L training): sigmoid score +
        # group-limited routing + aux-loss-free expert bias + topk scaling.
        scoring_func="sigmoid",
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        mlp_only_layers=[],
        layer_types=None,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            interval_pattern = kwargs.get("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        # layer_type_validation signature changed between transformers versions:
        #   4.57.0+: layer_type_validation(layer_types, num_hidden_layers)
        #   4.56.x:  layer_type_validation(layer_types)
        import inspect
        sig = inspect.signature(layer_type_validation)
        if len(sig.parameters) >= 2:
            layer_type_validation(self.layer_types, self.num_hidden_layers)
        else:
            layer_type_validation(self.layer_types)

        # linear attention part
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.scoring_func = scoring_func
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.mlp_only_layers = mlp_only_layers

    # ── SGLang compatibility properties ──────────────────────────
    # These are required by SGLang's model_runner.py for hybrid GDN models.
    # They mirror Qwen3NextConfig's properties from sglang.srt.configs.qwen3_next.

    @property
    def full_attention_interval(self):
        """Derive interval from layer_types or stored value."""
        if hasattr(self, "_full_attention_interval"):
            return self._full_attention_interval
        if hasattr(self, "layer_types") and self.layer_types:
            attn_ids = [i for i, lt in enumerate(self.layer_types) if lt == "full_attention"]
            if len(attn_ids) >= 2:
                return attn_ids[1] - attn_ids[0]
            elif len(attn_ids) == 1:
                return attn_ids[0] + 1
        return 4

    @full_attention_interval.setter
    def full_attention_interval(self, value):
        self._full_attention_interval = value

    @property
    def layers_block_type(self):
        interval = self.full_attention_interval
        return [
            "attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(self.num_hidden_layers)
        ]

    @property
    def full_attention_layer_ids(self):
        return [i for i, t in enumerate(self.layers_block_type) if t == "attention"]

    @property
    def linear_layer_ids(self):
        return [i for i, t in enumerate(self.layers_block_type) if t == "linear_attention"]

    @property
    def hybrid_gdn_params(self):
        """GatedDeltaNet state shapes for hybrid memory pool allocation."""
        try:
            from sglang.srt.distributed import get_attention_tp_size
            world_size = get_attention_tp_size()
        except (ImportError, RuntimeError):
            world_size = 1

        def _divide(a, b):
            assert a % b == 0, f"{a} not divisible by {b}"
            return a // b

        import torch, os
        conv_dim = (
            self.linear_key_head_dim * self.linear_num_key_heads * 2
            + self.linear_value_head_dim * self.linear_num_value_heads
        )
        conv_state_shape = (_divide(conv_dim, world_size), self.linear_conv_kernel_dim - 1)
        temporal_state_shape = (
            _divide(self.linear_num_value_heads, world_size),
            self.linear_key_head_dim,
            self.linear_value_head_dim,
        )
        conv_dtype = torch.bfloat16
        dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
        ssm_dtype = dtype_map.get(os.environ.get("SGLANG_MAMBA_SSM_DTYPE", "float32"), torch.float32)
        return conv_state_shape, temporal_state_shape, conv_dtype, ssm_dtype, self.linear_layer_ids

    @property
    def mamba_cache_per_req(self):
        """Memory per request for Mamba state pool sizing."""
        import numpy as np
        conv_state_shape, temporal_state_shape, conv_dtype, ssm_dtype, mamba_layers = (
            self.hybrid_gdn_params
        )
        return (
            int(np.prod(conv_state_shape)) * conv_dtype.itemsize
            + int(np.prod(temporal_state_shape)) * ssm_dtype.itemsize
        ) * len(mamba_layers)


__all__ = ["AlphaConfig"]
