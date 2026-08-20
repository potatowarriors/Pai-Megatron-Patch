# Copyright (c) 2026 Alibaba PAI Team.
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

from typing import Optional

import torch
from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.utils import deprecate_inference_params


class VarlenMambaLayer(MambaLayer):
    """MambaLayer that forwards PackedSeqParams (THD/varlen packing) to its mixer.

    Upstream MambaLayer.forward() has no packed_seq_params parameter, so packed
    sequences (SFT packing, LC document isolation) can never reach the mixer.
    This subclass adds the parameter and passes it through only when set, so
    behavior with packed_seq_params=None is identical to upstream.
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,  # Not used in MambaLayer
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,  # Not used in MambaLayer
        packed_seq_params: Optional[PackedSeqParams] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        inference_context = deprecate_inference_params(inference_context, inference_params)

        residual = hidden_states
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = hidden_states.to(dtype=self.config.params_dtype)
        hidden_states = self.norm(hidden_states)

        # Pass packed_seq_params only when set: mixers without varlen support
        # (upstream MambaMixer) keep working as long as packing stays off, and
        # fail loudly (TypeError) instead of silently ignoring it when on.
        mixer_kwargs = {}
        if packed_seq_params is not None:
            mixer_kwargs["packed_seq_params"] = packed_seq_params
        mixer_out_with_bias = self.mixer(
            hidden_states, inference_context=inference_context, **mixer_kwargs
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mamba_bda(
                training=self.training, fused=self.config.bias_dropout_fusion
            )(mixer_out_with_bias, residual, self.hidden_dropout)

        return hidden_states
