#!/bin/bash
# Copyright (c) 2025 Alibaba PAI and Nvidia Megatron-LM Team.
# Copyright (c) 2025 Alpha Project Team.
#
# Alpha Model Checkpoint Conversion Script
# =========================================
# HuggingFace ↔ Megatron checkpoint conversion for Alpha model
#
# Usage:
#   bash run_8xH20.sh <MODEL_SIZE> <LOAD_DIR> <SAVE_DIR> <MG2HF> <USE_CUDA> <PRECISION> [HF_DIR]
#
# Arguments:
#   MODEL_SIZE  : Model configuration (baseline_48L)
#   LOAD_DIR    : Input checkpoint directory (or training output directory for auto mode)
#   SAVE_DIR    : Output checkpoint directory, or:
#                 - "auto"       : Auto-detect latest checkpoint, save to hfmodel_{iter}
#                 - "auto:50000" : Use specific iteration, save to hfmodel_0050000
#   MG2HF       : Conversion direction (true: MG→HF, false: HF→MG)
#   USE_CUDA    : Use GPU for conversion (true/false)
#   PRECISION   : bf16/fp16/fp32
#   HF_DIR      : HuggingFace model directory (optional for MG→HF)
#                 If not provided, config.json will be auto-generated from unified config
#
# Examples:
#   # HF → Megatron
#   bash run_8xH20.sh baseline_48L /path/to/hf /path/to/mcore false true bf16
#
#   # Megatron → HF (with existing HF reference)
#   bash run_8xH20.sh baseline_48L /path/to/mcore /path/to/hf true true bf16 /path/to/hf-orig
#
#   # Megatron → HF (auto-generate HF config from unified config)
#   bash run_8xH20.sh baseline_48L /path/to/mcore /path/to/hf true true bf16
#
#   # Megatron → HF (AUTO MODE - latest checkpoint)
#   bash run_8xH20.sh baseline_48L /path/to/outputs auto true true bf16
#
#   # Megatron → HF (AUTO MODE - specific iteration)
#   bash run_8xH20.sh baseline_48L /path/to/outputs auto:50000 true true bf16

set -e
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
CONVERTOR_DIR=$( dirname $( dirname ${CURRENT_DIR}))
MEGATRON_PATCH_PATH=$( dirname $( dirname ${CONVERTOR_DIR}))
export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-251125:${CONVERTOR_DIR}/impl:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6


NUM_NODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}

MODEL_SIZE=$1
LOAD_DIR=$2
SAVE_DIR=$3
MG2HF=$4
USE_CUDA=$5
PR=$6
HF_DIR=$7

# ============================================================================
# Auto Mode Detection
# ============================================================================
# When SAVE_DIR is "auto" or "auto:ITERATION", automatically detect paths:
#   - "auto"       : Use latest checkpoint from latest_checkpointed_iteration.txt
#   - "auto:50000" : Use specific iteration
#
# Example:
#   bash run_8xH20.sh baseline_48L /path/to/outputs auto true true bf16
#   bash run_8xH20.sh baseline_48L /path/to/outputs auto:50000 true true bf16
# ============================================================================
if [[ "${SAVE_DIR}" == "auto"* ]]; then
    OUTPUT_DIR="${LOAD_DIR}"

    # Check if this is a training output directory
    if [ ! -d "${OUTPUT_DIR}/checkpoints" ]; then
        echo "❌ Error: ${OUTPUT_DIR}/checkpoints not found"
        echo "   Auto mode requires training output directory with checkpoints/"
        exit 1
    fi

    # Parse iteration from "auto" or "auto:ITERATION"
    if [[ "${SAVE_DIR}" == "auto:"* ]]; then
        ITERATION="${SAVE_DIR#auto:}"
        echo "📍 Using specified iteration: ${ITERATION}"
    else
        # Read latest iteration
        LATEST_FILE="${OUTPUT_DIR}/checkpoints/latest_checkpointed_iteration.txt"
        if [ ! -f "${LATEST_FILE}" ]; then
            echo "❌ Error: ${LATEST_FILE} not found"
            exit 1
        fi
        ITERATION=$(cat "${LATEST_FILE}" | tr -d '[:space:]')
        echo "📍 Using latest iteration: ${ITERATION}"
    fi

    # Format iteration with leading zeros (7 digits)
    ITERATION_PADDED=$(printf "%07d" ${ITERATION})

    # Set actual paths
    LOAD_DIR="${OUTPUT_DIR}/checkpoints/iter_${ITERATION_PADDED}"
    SAVE_DIR="${OUTPUT_DIR}/hfmodel_${ITERATION_PADDED}"

    echo ""
    echo "🔄 Auto mode enabled:"
    echo "   Input:  ${LOAD_DIR}"
    echo "   Output: ${SAVE_DIR}"
    echo ""

    # Validate checkpoint exists
    if [ ! -d "${LOAD_DIR}" ]; then
        echo "❌ Error: Checkpoint not found: ${LOAD_DIR}"
        echo "   Available checkpoints:"
        ls -1 "${OUTPUT_DIR}/checkpoints/" | grep "iter_" | sed 's/^/     /'
        exit 1
    fi
fi

# ============================================================================
# Checkpoint Path Fix for Megatron Distributed Checkpoints
# ============================================================================
# Megatron's load_checkpoint() looks for latest_checkpointed_iteration.txt
# in the --load-dir path. When iter_NNNNNN is passed directly, it fails.
#
# Fix: Extract iteration from path and use parent directory with --ckpt-step
# ============================================================================
CKPT_STEP=""
if [[ "${LOAD_DIR}" =~ iter_([0-9]+) ]]; then
    ITERATION="${BASH_REMATCH[1]}"
    PARENT_DIR=$(dirname "${LOAD_DIR}")

    # Check if parent directory has latest_checkpointed_iteration.txt
    if [ -f "${PARENT_DIR}/latest_checkpointed_iteration.txt" ]; then
        echo "📍 Detected iter_${ITERATION} path, adjusting for Megatron checkpoint loading:"
        echo "   Original: ${LOAD_DIR}"
        echo "   Adjusted: ${PARENT_DIR} with --ckpt-step ${ITERATION}"
        LOAD_DIR="${PARENT_DIR}"
        CKPT_STEP="${ITERATION}"
    fi
fi

# Alpha config tool path
ALPHA_DIR="${MEGATRON_PATCH_PATH}/examples/alpha"
ALPHA_CONFIG_TOOL="${ALPHA_DIR}/tools/alpha_config.py"

# Fixed tokenizer path (Alpha uses Qwen3-Next tokenizer)
TOKENIZER_PATH="${ALPHA_DIR}/tokenizer"

OTHER_ARGS=()
if [ ${MG2HF} = true ]; then
    mkdir -p ${SAVE_DIR}

    if [ -z "${HF_DIR}" ] || [ ! -d "${HF_DIR}" ]; then
        # Auto-generate HF config from unified config
        echo "🔧 Auto-generating HF config from unified config..."

        # Generate config.json
        python3 ${ALPHA_CONFIG_TOOL} generate-hf-config ${MODEL_SIZE} --output ${SAVE_DIR}/config.json

        # Copy tokenizer files from fixed tokenizer path (exclude config.json to preserve generated one)
        echo "📋 Copying tokenizer files from ${TOKENIZER_PATH}..."
        find -L ${TOKENIZER_PATH} -maxdepth 1 -type f -name "tokenizer*.json" -print0 2>/dev/null | xargs -0 -r cp -t ${SAVE_DIR} || true
        find -L ${TOKENIZER_PATH} -maxdepth 1 -type f -name "vocab.json" -print0 2>/dev/null | xargs -0 -r cp -t ${SAVE_DIR} || true
        find -L ${TOKENIZER_PATH} -maxdepth 1 -type f -name "*.txt" -print0 2>/dev/null | xargs -0 -r cp -t ${SAVE_DIR} || true
        find -L ${TOKENIZER_PATH} -maxdepth 1 -type f -name "*.model" -print0 2>/dev/null | xargs -0 -r cp -t ${SAVE_DIR} || true

        # Copy Alpha HF modeling files (required for auto_map in config.json)
        ALPHA_HF_MODEL_DIR="${MEGATRON_PATCH_PATH}/examples/alpha/hf_model"
        if [ -d "${ALPHA_HF_MODEL_DIR}" ]; then
            echo "📋 Copying Alpha HF modeling files..."
            cp ${ALPHA_HF_MODEL_DIR}/*.py ${SAVE_DIR}/
        fi

        HF_DIR=${SAVE_DIR}
        echo "✅ HF config auto-generated at ${SAVE_DIR}/config.json"
    else
        # Use existing HF reference
        echo "📋 Copying HF config from ${HF_DIR}..."
        find -L ${HF_DIR} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVE_DIR}
        find -L ${HF_DIR} -maxdepth 1 -type f -name "merges.txt" -print0 2>/dev/null | xargs -0 -r cp -t ${SAVE_DIR} || true
    fi

    OTHER_ARGS+=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model ${TOKENIZER_PATH}
        --hf-dir ${HF_DIR}
        --mcore2hf
    )
else
    # HF → Megatron conversion
    OTHER_ARGS+=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model ${LOAD_DIR}
    )
    mkdir -p ${SAVE_DIR}
    find -L ${LOAD_DIR} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVE_DIR}
    find -L ${LOAD_DIR} -maxdepth 1 -type f -name "merges.txt" -print0 2>/dev/null | xargs -0 -r cp -t ${SAVE_DIR} || true
fi

if [ ${USE_CUDA} = true ]; then
    OTHER_ARGS+=(
        --use-gpu
    )
fi

if [ ${PR} = fp16 ]; then
    OTHER_ARGS+=(
        --fp16
    )
elif [ ${PR} = bf16 ]; then
    OTHER_ARGS+=(
        --bf16
    )
fi

if [ -z ${NUM_NODES} ]; then
    echo "Please Provide WORLD_SIZE"
    exit
fi

if [ -z ${NODE_RANK} ]; then
    echo "Please Provide RANK"
    exit
fi

if [ -z ${MASTER_ADDR} ]; then
    echo "Please Provide MASTER_ADDR"
    exit
fi

if [ -z ${MASTER_PORT} ]; then
    echo "Please Provide MASTER_PORT"
    exit
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --normalization RMSNorm
    --swiglu
    --disable-bias-linear
    --seq-length 1
    --max-position-embeddings 4096
    --attention-backend auto
    --position-embedding-type rope
    --kv-channels 128
    --qk-layernorm
    --group-query-attention
)

# Load model-specific configuration
CONFIG_FILE="${CURRENT_DIR}/configs/${MODEL_SIZE}.sh"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Model configuration not found: ${CONFIG_FILE}"
    echo "Available configurations:"
    ls -1 "${CURRENT_DIR}/configs/"*.sh 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "Loading configuration: ${CONFIG_FILE}"
source "${CONFIG_FILE}"

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1024
    --train-iters 500000
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --bf16
    --lr 6.0e-5
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --lr-warmup-fraction .001
    --lr-decay-iters 430000
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 10
)

CONVERT_ARGS=(
    --model-type GPT
    --load-dir ${LOAD_DIR}
    --save-dir ${SAVE_DIR}

    --padded-vocab-size ${VOCAB_SIZE}
    --no-load-optim
    --no-load-rng
    --logging-level 1

    --synchronizer alpha
    --pretrain-script alpha.model_provider
    --auto-detect-ckpt-format    # Support both torch and torch_dist checkpoint formats
    # --debug  # Disabled for Alpha EP-only conversion
)

# Add --ckpt-step if detected from iter_NNNNNN path
if [ -n "${CKPT_STEP}" ]; then
    CONVERT_ARGS+=(--ckpt-step ${CKPT_STEP})
fi

# Change to CONVERTOR_DIR to use relative path impl/convert.py
cd ${CONVERTOR_DIR}

cmd="torchrun ${DISTRIBUTED_ARGS[@]} impl/convert.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${CONVERT_ARGS[@]} \
    ${OTHER_ARGS[@]}"

echo $cmd
eval $cmd

# Post-conversion: Verify Alpha HF modeling files exist
if [ ${MG2HF} = true ]; then
    if [ -f "${SAVE_DIR}/configuration_alpha.py" ] && [ -f "${SAVE_DIR}/modeling_alpha.py" ]; then
        echo "✅ Alpha HF model conversion complete: ${SAVE_DIR}"
    else
        echo "⚠️ Warning: Alpha HF modeling files missing in ${SAVE_DIR}"
    fi
fi
