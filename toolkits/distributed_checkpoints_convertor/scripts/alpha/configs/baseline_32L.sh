#!/bin/bash
# Copyright (c) 2025 Alibaba PAI Team.
# Copyright (c) 2025 Alpha Project Team.
#
# Alpha baseline_32L Configuration (Example)
# ===========================================
# 32 Megatron layers → 16 HF layers (2:1 mapping)
# This is a template for future 32-layer experiments

GPT_MODEL_ARGS+=(
    # Basic Architecture
    --num-layers 32                         # 24 → 32
    --hidden-size 2048
    --ffn-hidden-size 5120
    --num-attention-heads 32
    --kv-channels 128
    --num-query-groups 2

    # Hybrid Model Pattern
    # 32 layers = 8 groups × (M-M-M-*) = 24 Mamba + 8 Attention
    # Attention ratio: 4/32 = 0.125 (12.5%)
    # MLP ratio: 16/32 = 0.5 (50%)
    --hybrid-attention-ratio 0.125
    --hybrid-mlp-ratio 0.5
    --hybrid-override-pattern M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-
    --is-hybrid-model

    # MoE Configuration (same as 24L)
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
if [ -z "$MODEL_PARALLEL_ARGS" ]; then
    MODEL_PARALLEL_ARGS=(
        --tensor-model-parallel-size 1
        --pipeline-model-parallel-size 1
        --expert-model-parallel-size 8
    )
fi

VOCAB_SIZE=151936
