#!/bin/bash
# Copyright (c) 2025 Alibaba PAI Team.
# Copyright (c) 2025 Alpha Project Team.
#
# Alpha baseline_24L Configuration
# =================================
# 24 Megatron layers → 12 HF layers (2:1 mapping)
# Based on: examples/alpha/configs/model/baseline_24L.yaml

GPT_MODEL_ARGS+=(
    # Basic Architecture
    --num-layers 24
    --hidden-size 2048
    --ffn-hidden-size 5120
    --num-attention-heads 32
    --kv-channels 128
    --num-query-groups 2

    # Hybrid Model Pattern
    # 24 layers = 6 groups × (M-M-M-*) = 18 Mamba + 6 Attention
    # Attention ratio: 3/24 = 0.125 (12.5%)
    # MLP ratio: 12/24 = 0.5 (50%)
    --hybrid-attention-ratio 0.125
    --hybrid-mlp-ratio 0.5
    --hybrid-override-pattern M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-
    --is-hybrid-model

    # MoE Configuration
    --num-experts 256
    --moe-router-topk 8
    --moe-ffn-hidden-size 768
    --moe-shared-expert-intermediate-size 768
    --moe-grouped-gemm
    --moe-router-score-function softmax
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
        --expert-model-parallel-size 8      # 256 experts / 8 GPUs = 32 per GPU
    )
fi

# Vocabulary
VOCAB_SIZE=151936
