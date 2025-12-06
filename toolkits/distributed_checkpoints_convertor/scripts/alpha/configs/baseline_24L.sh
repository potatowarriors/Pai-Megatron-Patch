#!/bin/bash
# Copyright (c) 2025 Alibaba PAI Team.
# Copyright (c) 2025 Alpha Project Team.
#
# Alpha alpha-baseline-24L Configuration
# =================================
# 24 Megatron layers -> 12 HF layers (2:1 mapping)
# Auto-generated from: examples/alpha/configs/model/baseline-24L.yaml

GPT_MODEL_ARGS+=(
    # Basic Architecture
    --num-layers 24
    --hidden-size 2048
    --ffn-hidden-size 5120
    --num-attention-heads 32
    --kv-channels 128
    --num-query-groups 2

    # Hybrid Model Pattern
    # Pattern: MDM-M-*-M-M-M-*-M-M-M-*- (24 chars total)
    # M=Mamba(9), D=Dense MLP(1), -=MoE MLP(11), *=Attention(3)
    # Attention ratio: 0.125 (3 layers)
    # MLP ratio: 0.5 (12 layers)
    --hybrid-attention-ratio 0.125
    --hybrid-mlp-ratio 0.5
    --hybrid-override-pattern MDM-M-*-M-M-M-*-M-M-M-*-
    --is-hybrid-model

    # MoE Configuration
    --num-experts 256
    --moe-router-topk 8
    --moe-ffn-hidden-size 768
    --moe-shared-expert-intermediate-size 768
    --moe-shared-expert-gate
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
