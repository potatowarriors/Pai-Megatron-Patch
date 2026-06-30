#!/bin/bash
# Copyright (c) 2025 Alibaba PAI Team.
# Copyright (c) 2025 Alpha Project Team.
#
# Alpha alpha Configuration
# =================================
# 48 Megatron layers -> 24 HF layers (2:1 mapping)
# Auto-generated from: examples/alpha/configs/model/alpha.yaml

GPT_MODEL_ARGS+=(
    # Basic Architecture
    --num-layers 48
    --hidden-size 2048
    --ffn-hidden-size 8192
    --num-attention-heads 16
    --kv-channels 256
    --num-query-groups 2

    # Hybrid Model Pattern
    # Pattern: M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-
    # Attention ratio: 0.125 (6 layers)
    # MLP ratio: 0.5 (24 layers)
    --hybrid-attention-ratio 0.125
    --hybrid-mlp-ratio 0.5
    --hybrid-override-pattern M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-
    --is-hybrid-model

    # MoE Configuration
    --num-experts 192
    --moe-router-topk 8
    --moe-ffn-hidden-size 512
    --moe-shared-expert-intermediate-size 512
    --moe-shared-expert-gate
    --moe-grouped-gemm
    --moe-router-score-function sigmoid
    --moe-token-dispatcher-type alltoall

    # RoPE Settings
    --rotary-base 10000000
    --rotary-percent 0.25

    # Output Layer
    --untie-embeddings-and-output-weights
)

# Parallelization Strategy
# NOTE: TP=1 required (Mamba layer constraint)
if [ -z "$MODEL_PARALLEL_ARGS" ]; then
    MODEL_PARALLEL_ARGS=(
        --tensor-model-parallel-size 1
        --pipeline-model-parallel-size 1
        --expert-model-parallel-size 8      # 192 experts / 8 GPUs = 24 per GPU
    )
fi

# Vocabulary
VOCAB_SIZE=163968
