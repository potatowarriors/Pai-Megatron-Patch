#!/bin/bash
# Copyright (c) 2025 Alibaba PAI Team.
# Copyright (c) 2025 Alpha Project Team.
#
# Alpha baseline_48L Configuration (Example)
# ===========================================
# 48 Megatron layers → 24 HF layers (2:1 mapping)
# This is a template for future 48-layer experiments

GPT_MODEL_ARGS+=(
    # Basic Architecture
    --num-layers 48                         # 24 → 48
    --hidden-size 2048
    --ffn-hidden-size 8192
    --num-attention-heads 32
    --kv-channels 128
    --num-query-groups 2

    # Hybrid Model Pattern
    # Pattern: MDM-M-*-M-M-M-*-... (48 chars total)
    # M=Mamba(18), D=Dense MLP(1), -=MoE MLP(23), *=Attention(6)
    # Attention ratio: 6/48 = 0.125 (12.5%)
    # MLP ratio: 24/48 = 0.5 (50%)
    --hybrid-attention-ratio 0.125
    --hybrid-mlp-ratio 0.5
    --hybrid-override-pattern MDM-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-
    --is-hybrid-model

    # MoE Configuration
    # Option 1: Keep 256 experts (same density as 24L)
    --num-experts 128
    --moe-router-topk 8

    # Option 2: Scale to 512 experts (double for 2x layers)
    # --num-experts 512
    # --moe-router-topk 10
    # --expert-model-parallel-size 8  # 512/8 = 64 per GPU

    --moe-ffn-hidden-size 768
    --moe-shared-expert-intermediate-size 768
    --moe-grouped-gemm
    --moe-router-score-function softmax
    --moe-token-dispatcher-type alltoall
    --moe-shared-expert-gate

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
        --expert-model-parallel-size 8      # Adjust if using 512 experts
    )
fi

VOCAB_SIZE=151936
