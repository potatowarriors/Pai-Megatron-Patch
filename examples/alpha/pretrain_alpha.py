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
Alpha Project - Pre-training Script
====================================
Qwen3-Next Mamba 기반 Alpha 모델 학습을 위한 메인 스크립트.

이 스크립트는 다음을 수행합니다:
1. Qwen3-Next의 Mamba 아키텍처 기반 모델 빌드
2. Hybrid Attention/MLP 패턴 지원
3. 분산 학습 환경에서 안전한 종료 처리
4. YAML 기반 설정과 통합

Base: Qwen3-Next (Pai-Megatron-Patch)
Architecture: Mamba + Hybrid Attention
"""

from functools import partial
from typing import Optional
import torch
import torch._dynamo
from torch import Tensor

from megatron.core.enums import ModelType
from megatron.core import mpu
from model_provider import model_provider as base_model_provider  # Megatron-LM-250908/model_provider.py

from megatron.training.arguments import core_transformer_config_from_args
from megatron_patch.arguments import get_patch_args
from megatron_patch.data import train_valid_test_datasets_provider
from megatron.training import pretrain, print_rank_0

torch._dynamo.config.suppress_errors = True

from megatron.core.models.mamba import MambaModel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

# Alpha model with Dense MLP support (D symbol in pattern)
from megatron_patch.model.alpha.layer_specs import get_alpha_layer_spec
from megatron_patch.model.qwen3_next.transformer_config import Qwen3NextTransformerConfig


def mamba_builder(args, pre_process, post_process, vp_stage=None, config=None):
    """
    Alpha 모델 빌더 (Mamba 기반)

    Args:
        args: 학습 인자 (megatron + patch args)
        pre_process: 임베딩 레이어 포함 여부
        post_process: 최종 레이어 포함 여부
        vp_stage: Virtual pipeline stage (pipeline parallelism)
        config: Transformer 설정 (None이면 args에서 생성)

    Returns:
        MambaModel: Alpha 모델 인스턴스
    """
    print_rank_0('building Alpha MAMBA model ...')
    if config is None:
        config = core_transformer_config_from_args(args, Qwen3NextTransformerConfig)
    assert args.use_legacy_models is False, "Mamba only supported in Mcore!"

    model = MambaModel(
        config=config,
        mamba_stack_spec=get_alpha_layer_spec(args),
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


# Partial application으로 base_model_provider와 통합
model_provider = partial(base_model_provider, mamba_builder)


if __name__ == "__main__":
    from megatron_patch.template.helper import forward_step

    # ── QK-Clip hybrid fix ──────────────────────────────────────────────
    # Upstream clip_qk() assumes all decoder layers have self_attention
    # (TransformerLayer). In hybrid models, MambaLayer has no self_attention
    # attribute → AttributeError. Fix: skip layers that lack self_attention.
    import megatron.training.training as _training_mod

    def _hybrid_clip_qk(model, log_max_only=False):
        """clip_qk patched for hybrid Mamba+Attention models."""
        with torch.no_grad():
            log_max_attention_logit = 0
            for model_chunk in model:
                for layer in model_chunk.module.module.decoder.layers:
                    if not hasattr(layer, 'self_attention'):
                        continue
                    if hasattr(layer.self_attention, 'clip_qk'):
                        torch.distributed.all_reduce(
                            layer.self_attention.core_attention.current_max_attn_logits,
                            op=torch.distributed.ReduceOp.MAX,
                            group=mpu.get_data_parallel_group(with_context_parallel=True),
                        )
                        log_max_attention_logit = max(
                            log_max_attention_logit,
                            torch.max(
                                layer.self_attention.core_attention.current_max_attn_logits
                            ).item(),
                        )
                        if not log_max_only:
                            layer.self_attention.clip_qk()
        return log_max_attention_logit

    _training_mod.clip_qk = _hybrid_clip_qk
    # ─────────────────────────────────────────────────────────────────────

    # 분산 데이터 로딩 활성화
    train_valid_test_datasets_provider.is_distributed = True

    # Megatron pretrain 실행
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=get_patch_args,
    )
