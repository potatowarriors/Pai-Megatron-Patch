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
import logging

import torch
from typing import Dict
from general.m2h_synchronizer import MG2HFSynchronizer as _MG2HFSynchronizer
from general.synchronizer import ParamType
from .common import (
    validate_hybrid_pattern,
    log_conversion_summary,
    build_pipeline_parallel_mapping,
    CHAR_MAMBA, CHAR_ATTENTION, CHAR_MLP
)

class MG2HFSynchronizer(_MG2HFSynchronizer):
    # TODO: to be refactored to Hybrid Model convertor
    def __init__(self, load_dir, model_provider_func=None):
        super().__init__(load_dir, model_provider_func)
        self.layout = self.get_hybrid_layout()
        assert self.tp_size == 1, "Currently MCore2HF conversion for Alpha is only available with TP 1."

        # Validate conversion configuration before starting
        self._validate_conversion_config()

    def get_hybrid_layout(self) -> str:
        assert self.args.hybrid_override_pattern is not None, \
            "hybrid_override_pattern is required for Alpha conversion"
        return self.args.hybrid_override_pattern

    def _validate_conversion_config(self):
        """Validate Alpha conversion configuration to catch errors early."""
        # Use common validation function
        validate_hybrid_pattern(
            layout=self.layout,
            num_layers=self.args.num_layers,
            hybrid_attention_ratio=self.args.hybrid_attention_ratio,
            rank=self.rank
        )

        # Log conversion summary
        dp_size = torch.distributed.get_world_size() // self.tp_size // self.pp_size // self.ep_size
        log_conversion_summary(
            direction="MG2HF",
            layout=self.layout,
            args=self.args,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            ep_size=self.ep_size,
            rank=self.rank,
            dp_info=f"{self.dp_rank}/{dp_size}"
        )

    def set_preprocess_state(self, mg_model, hf_model):
        '''Override to use hf_model.model.embed_tokens for Qwen3Next structure.'''
        self.copy(
            mg_model.embedding.word_embeddings.weight,
            hf_model.model.embed_tokens.weight,
            param_type=ParamType.COLUMN
        )

    def _copy_impl(self, src_tensor, dst_tensor, param_type: ParamType=ParamType.UNIQUE):
        '''Debug override to log tensor types and detect dict issues.'''
        param_id = self._hf_params_to_id[dst_tensor]
        hf_key = self._id_to_hf_params_key[param_id]

        # Log what we're copying
        if self.rank == 0 or (param_type in [ParamType.MOE_COLUMN, ParamType.MOE_ROW, ParamType.MOE_GATE_UP, ParamType.MOE_DOWN] and self.ep_rank < 2):
            src_type = type(src_tensor).__name__
            src_info = f"shape={src_tensor.shape}" if hasattr(src_tensor, 'shape') else f"type={src_type}"
            logging.info(f"[RANK{self.rank}] _copy_impl: {hf_key} | param_type={param_type.name} | src={src_info}")

            # Alert if src_tensor is dict
            if isinstance(src_tensor, dict):
                logging.error(f"[RANK{self.rank}] ERROR: src_tensor is dict for {hf_key}! Keys: {list(src_tensor.keys())[:5]}")

        # Call parent implementation
        return super()._copy_impl(src_tensor, dst_tensor, param_type=param_type)

    def sync_params(self, mg_model = None, hf_model = None):
        # assume TE backend
        if self.args.transformer_impl != "transformer_engine":
            raise NotImplementedError("Currently only TE model is implemented.")
        
        if mg_model is None:
            mg_model = self._mgmodel
        if hf_model is None:
            hf_model = self._hfmodel

        if mg_model.pre_process:
            self.set_preprocess_state(mg_model=mg_model, hf_model=hf_model)
        
        if mg_model.post_process:
            self.set_postprocess_state(mg_model=mg_model, hf_model=hf_model, is_mamba=True)

        for mg_layer_id, global_mg_layer_id in self._build_pipeline_parallel_mapping().items():
            # Alpha: 2:1 mapping (24 MG layers → 12 HF layers)
            # Qwen3-Next: 2:1 mapping (96 MG layers → 48 HF layers)
            # Use parent class (qwen3_next) implementation: 2:1 mapping
            hf_layer_id  = global_mg_layer_id // 2
            if (
                self.tp_rank == 0 and
                self.ep_rank == 0 and
                self.etp_rank == 0
            ):
                logging.info(f"Converting layer {hf_layer_id}")

            layer = mg_model.decoder.layers[mg_layer_id]
            hf_layer = hf_model.model.layers[hf_layer_id]

            if self.layout[global_mg_layer_id] == 'M':
                # Mamba layer
                self.set_mamba_layer_state(layer.mixer, hf_layer.linear_attn)
                self.copy(layer.mixer.in_proj.layer_norm_weight, hf_layer.input_layernorm.weight)
            elif self.layout[global_mg_layer_id] == '-':
                # transformer_layer of MLP (MoE)
                self.set_moe_layer_state(layer.mlp, hf_layer.mlp)
                self.copy(layer.pre_mlp_layernorm.weight, hf_layer.post_attention_layernorm.weight)
            elif self.layout[global_mg_layer_id] == 'D':
                # Dense MLP layer (standard FFN with SwiGLU)
                # Uses inherited set_mlp_state() from base class
                self.set_mlp_state(layer.mlp, hf_layer.mlp)
                self.copy(layer.pre_mlp_layernorm.weight, hf_layer.post_attention_layernorm.weight)
            elif self.layout[global_mg_layer_id] == '*':
                # transformer_layer with Full Attention only (no MLP!)
                # NOTE: Megatron's '*' layers use IdentityOp for MLP (no actual MLP weights)
                # NOTE: HF creates self.self_attn for full_attention layers (not linear_attn)
                self.set_gated_selfattn_state(layer.self_attention, hf_layer.self_attn)
                self.copy(layer.self_attention.linear_qgkv.layer_norm_weight, hf_layer.input_layernorm.weight)
            else:
                raise ValueError(f"Unrecognized layer type {self.layout[global_mg_layer_id]} in {self.layout}")

    def set_mamba_layer_state(self, mixer, hf_mixer):
        Nk, Nv, Dk, Dv = (
            hf_mixer.num_k_heads,
            hf_mixer.num_v_heads,
            hf_mixer.head_k_dim,
            hf_mixer.head_v_dim
        )
        split_size_list = [
            Dv * Nv // self.tp_size, 
            Dv * Nv // self.tp_size, 
            Dk * Nk // self.tp_size, 
            Dk * Nk // self.tp_size, 
            Nv // self.tp_size, 
            Nv // self.tp_size
        ]
        z, v, q, k, b, a = torch.split(
            mixer.in_proj.weight,
            split_size_list,
            dim=0
        )
        in_proj_qkvz_weight = torch.cat([
            q.reshape(Nk // self.tp_size, Dk, -1), 
            k.reshape(Nk // self.tp_size, Dk, -1), 
            v.reshape(Nk // self.tp_size, Dv * Nv // Nk, -1), 
            z.reshape(Nk // self.tp_size, Dv * Nv // Nk, -1), 
        ], dim=1)
        self.copy(in_proj_qkvz_weight, hf_mixer.in_proj_qkvz.weight, param_type=ParamType.QKV_W)

        in_proj_ba_weight = torch.cat([
            b.reshape(Nk // self.tp_size, Nv // Nk, -1), 
            a.reshape(Nk // self.tp_size, Nv // Nk, -1), 
        ], dim=1)
        self.copy(in_proj_ba_weight, hf_mixer.in_proj_ba.weight, param_type=ParamType.QKV_W)

        self.copy(mixer.dt_bias, hf_mixer.dt_bias, param_type=ParamType.COLUMN)
        self.copy(mixer.A_log, hf_mixer.A_log, param_type=ParamType.COLUMN)

        # TODO: support TP > 1 if needed
        split_size_list = [
            Nv * Dv, 
            Nk * Dk, 
            Nk * Dk, 
        ]
        conv_v, conv_q, conv_k = torch.split(
            mixer.conv1d.weight, 
            split_size_or_sections=split_size_list, 
            dim=0
        )
        conv1d_weight = torch.cat([
            conv_q, 
            conv_k, 
            conv_v, 
        ], dim=0)
        self.copy(conv1d_weight, hf_mixer.conv1d.weight, param_type=ParamType.UNIQUE)
        
        self.copy(mixer.norm.weight, hf_mixer.norm.weight, param_type=ParamType.UNIQUE)
        self.copy(mixer.out_proj.weight, hf_mixer.out_proj.weight, param_type=ParamType.ROW)

    def _build_pipeline_parallel_mapping(self) -> Dict[int, int]:
        return build_pipeline_parallel_mapping(
            num_layers=self.args.num_layers,
            pp_size=self.pp_size,
            pp_rank=self.pp_rank
        )

    def set_gated_selfattn_state(self, attn, hf_attn):
        '''Set gated self-attention params.'''
        # Reshape loaded weights.
        tp = self.tp_size
        num_heads = self.args.num_attention_heads
        num_query_groups = (self.args.num_query_groups if self.args.group_query_attention else self.args.num_attention_heads)
        num_querys_per_group = num_heads // num_query_groups
        dim = self.args.kv_channels
        assert num_heads % num_querys_per_group == 0
        # copy qk norm if indeed.
        if self.args.qk_layernorm:
            self.copy(attn.q_layernorm.weight, hf_attn.q_norm.weight)
            self.copy(attn.k_layernorm.weight, hf_attn.k_norm.weight)

        # Copy weights (re-order dimensions for Megatron).
        attn_proj_weight = attn.linear_qgkv.weight.reshape(
            (num_query_groups // tp, (2 + num_querys_per_group*2)*dim, -1)
        )
        (
            q_proj_weight, 
            k_proj_weight, 
            v_proj_weight
        ) = torch.split(attn_proj_weight, [2*num_querys_per_group*dim, dim, dim], dim=1)

        q_proj_weight = q_proj_weight.reshape(num_query_groups // tp, 2, num_querys_per_group, dim, -1).transpose(1, 2).flatten(1, 3)
        self.copy(q_proj_weight, hf_attn.q_proj.weight, param_type=ParamType.QKV_W)
        self.copy(k_proj_weight, hf_attn.k_proj.weight, param_type=ParamType.QKV_W)
        self.copy(v_proj_weight, hf_attn.v_proj.weight, param_type=ParamType.QKV_W)

        self.copy(
            attn.linear_proj.weight,
            hf_attn.o_proj.weight,
            param_type=ParamType.ROW
        )

        # Copy bias
        # NOTE: dormant for Alpha (attention_bias=False / add_qkv_bias unset), so this
        # path is currently never taken. Two fixes vs the original code:
        #   1. read from `linear_qgkv` (4-way Q|Gate|K|V), not `linear_qkv` — the
        #      latter does not exist on gated attention and would AttributeError.
        #   2. the Q bias must get the same gate-interleave transpose the weight path
        #      applies above (reshape→transpose(1,2)→flatten) before splitting into
        #      [Q,Gate] per head; the plain split below is INCOMPLETE. Apply the
        #      weight-path reshape to `q_proj_bias` before enabling add_qkv_bias.
        if self.args.add_qkv_bias:
            attn_proj_bias = attn.linear_qgkv.bias.reshape(
                (num_query_groups // tp, (2 + num_querys_per_group * 2)*dim, -1)
            )
            q_proj_bias, k_proj_bias, v_proj_bias = torch.split(
                attn_proj_bias,
                [2*num_querys_per_group*dim, dim, dim],
                dim=1
            )
            self.copy(q_proj_bias, hf_attn.q_proj.bias, param_type=ParamType.QKV_B)
            self.copy(k_proj_bias, hf_attn.k_proj.bias, param_type=ParamType.QKV_B)
            self.copy(v_proj_bias, hf_attn.v_proj.bias, param_type=ParamType.QKV_B)