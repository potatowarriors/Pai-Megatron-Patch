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

    # ── FP8 GDN-exclusion: keep Mamba/GatedDeltaNet layers in bf16 ─────────
    # A global --fp8-format autocast quantizes EVERY TE linear GEMM — including
    # the GatedDeltaNet in_proj/out_proj (the gateway GEMMs feeding the delta-rule
    # recurrence). Alpha runs the Pai *custom* GDN which, unlike upstream
    # megatron/core/ssm/gated_delta_net.py, has NO fp8 alignment guard → SSM fp8 is
    # unproven. Low-risk path: FP8 only the attention ('*') and MoE ('-') layers;
    # force the GDN 'M' layers to bf16.
    #
    # Hook: get_fp8_context() returns nullcontext (bf16) whenever
    # is_first_last_bf16_layer(config, layer_no) is True. We extend that predicate
    # to also return True for 'M' positions of hybrid_override_pattern. Per-layer
    # bf16 requires a NON-delayed recipe (blockwise/tensorwise/mxfp8) — upstream
    # asserts delayed+per-layer-bf16 is unsupported, so pair this with
    # --fp8-recipe blockwise. When fp8 is OFF, get_fp8_context short-circuits and
    # never calls this predicate → strict no-op for bf16 runs.
    import megatron.core.fp8_utils as _fp8_utils

    _orig_is_first_last_bf16_layer = _fp8_utils.is_first_last_bf16_layer
    _gdn_bf16_state = {"mset": None, "logged": False}

    def _gdn_bf16_layer_set():
        """Global layer indices that must stay bf16 = 'M' (GDN) positions."""
        if _gdn_bf16_state["mset"] is None:
            try:
                from megatron.training import get_args
                pattern = getattr(get_args(), "hybrid_override_pattern", None) or ""
            except Exception:
                pattern = ""
            _gdn_bf16_state["mset"] = {i for i, c in enumerate(pattern) if c == "M"}
        return _gdn_bf16_state["mset"]

    def _alpha_is_first_last_bf16_layer(config, layer_no):
        # Preserve upstream first/last-layers-bf16 behavior.
        if _orig_is_first_last_bf16_layer(config, layer_no):
            return True
        if layer_no is None or layer_no < 0:
            return False
        mset = _gdn_bf16_layer_set()
        if mset and not _gdn_bf16_state["logged"]:
            print_rank_0(
                f"[alpha-fp8] GDN/Mamba 'M' layers forced to bf16 under FP8: "
                f"{len(mset)}/{getattr(config, 'num_layers', '?')} "
                f"(attention '*' + MoE '-' layers run FP8)"
            )
            _gdn_bf16_state["logged"] = True
        return layer_no in mset

    _fp8_utils.is_first_last_bf16_layer = _alpha_is_first_last_bf16_layer
    # ─────────────────────────────────────────────────────────────────────

    # ── LayerNorm WD: apply_wd_to_all_layernorm ────────────────────────
    # Upstream get_no_weight_decay_cond only supports 'apply_wd_to_qk_layernorm'
    # (q/k_layernorm only). For Stage 2+ ablation we want WD on ALL standard
    # LayerNorms (input, post_attention, q/k, final) to keep zero-centered
    # gamma close to identity (raw w → 0 ⇔ effective γ = 1+w → 1).
    #
    # Mamba's linear_attn.norm uses RMSNormGated (init=ones, NOT zero-centered)
    # so standard WD would push γ → 0 and destroy normalization. Excluded
    # explicitly. See plan: floofy-noodling-coral.md
    _orig_get_no_weight_decay_cond = _training_mod.get_no_weight_decay_cond

    def _patched_get_no_weight_decay_cond(
        no_weight_decay_cond_type, default_skip_embedding_weight_decay
    ):
        if no_weight_decay_cond_type != "apply_wd_to_all_layernorm":
            return _orig_get_no_weight_decay_cond(
                no_weight_decay_cond_type, default_skip_embedding_weight_decay
            )

        def apply_wd_to_all_layernorm_fn(name, param):
            lname = name.lower()
            # === EXCLUSIONS (Mamba RMSNormGated — NOT zero-centered) ===
            # Order matters: must check exclusions BEFORE the .norm.weight
            # inclusion below, since mixer.norm.weight would match both.
            if "mixer.norm" in lname:
                return True
            if "linear_attn" in lname and "norm" in lname:
                return True

            # === INCLUSIONS (zero-centered LayerNorms — apply WD) ===
            # 1. Standard Megatron names: input_layernorm, pre_mlp_layernorm,
            #    post_attention_layernorm, q_layernorm, k_layernorm, final_layernorm.
            if "layernorm" in lname or "layer_norm" in lname:
                return False
            # 2. Mamba pre-mixer norm: decoder.layers.X.norm.weight
            #    (functionally equivalent to input_layernorm; zero-centered via 1p).
            if lname.endswith(".norm.weight"):
                return False

            # === DEFAULT Megatron exclusions ===
            return (
                name.endswith(".bias")
                or len(param.shape) == 1
                or (default_skip_embedding_weight_decay and "embedding" in name)
            )

        return apply_wd_to_all_layernorm_fn

    _training_mod.get_no_weight_decay_cond = _patched_get_no_weight_decay_cond

    # Extend --no-weight-decay-cond-type argparse choices to accept the new value.
    # Upstream restricts choices=['apply_wd_to_qk_layernorm']; without this wrapper,
    # passing 'apply_wd_to_all_layernorm' would fail at argparse.
    import argparse as _argparse

    def _alpha_extra_args_provider(parser):
        parser = get_patch_args(parser)
        for action in parser._actions:
            if isinstance(action, _argparse._StoreAction) and (
                "--no-weight-decay-cond-type" in action.option_strings
            ):
                if "apply_wd_to_all_layernorm" not in action.choices:
                    action.choices.append("apply_wd_to_all_layernorm")
                break
        return parser
    # ─────────────────────────────────────────────────────────────────────

    # 분산 데이터 로딩 활성화
    train_valid_test_datasets_provider.is_distributed = True

    # Megatron pretrain 실행
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=_alpha_extra_args_provider,
    )
