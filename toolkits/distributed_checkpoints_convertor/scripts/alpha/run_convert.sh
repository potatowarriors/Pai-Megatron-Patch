#!/bin/bash
# Copyright (c) 2025 Alibaba PAI and Nvidia Megatron-LM Team.
# Copyright (c) 2025 Alpha Project Team.
#
# Alpha Model Checkpoint Conversion (GPU-count agnostic)
# ======================================================
# Single parameterized converter replacing run_8xH20.sh / run_4xGPU.sh. The GPU
# count is detected (or overridden via GPUS=) and the expert-parallel size is
# derived from it (EP = #GPU, TP=PP=1). Megatron's torch_dist resharding loads a
# checkpoint trained at any EP (e.g. EP=8) under a different EP (e.g. EP=4), so
# the conversion topology is independent of the training topology.
#
# Model-skeleton flags are NOT hardcoded here: for MG→HF they are derived from
# the checkpoint's own common.pt (ground truth) via
#   examples/alpha/tools/alpha_config.py emit-megatron-flags --from-checkpoint ...
# eliminating the historical drift between training YAML and a stale convert .sh.
#
# Usage:
#   [GPUS=N] bash run_convert.sh <MODEL_SIZE> <LOAD_DIR> <SAVE_DIR> <MG2HF> <USE_CUDA> <PRECISION> [HF_DIR]
#
# Arguments:
#   MODEL_SIZE  : Model configuration name (e.g. baseline_48L) — used for HF→MG
#                 and to locate the alpha synchronizer. For MG→HF the structural
#                 flags come from the checkpoint, not this name.
#   LOAD_DIR    : Input checkpoint dir (or training output dir for auto mode)
#   SAVE_DIR    : Output dir, or "auto" / "auto:ITERATION"
#   MG2HF       : true = MG→HF, false = HF→MG
#   USE_CUDA    : true/false
#   PRECISION   : bf16/fp16/fp32
#   HF_DIR      : (optional) HF reference for config.json (MG→HF)
#
# Env:
#   GPUS        : number of GPUs to convert with (default: auto-detect). EP=GPUS.
#
# Examples:
#   bash run_convert.sh baseline_48L /path/to/outputs auto true true bf16          # auto GPUs
#   GPUS=4 bash run_convert.sh baseline_48L /path/to/outputs auto true true bf16   # force 4 GPUs

set -e
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
CONVERTOR_DIR=$( dirname $( dirname ${CURRENT_DIR}))
MEGATRON_PATCH_PATH=$( dirname $( dirname ${CONVERTOR_DIR}))
export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-251125:${CONVERTOR_DIR}/impl:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6

ALPHA_DIR="${MEGATRON_PATCH_PATH}/examples/alpha"
ALPHA_CONFIG_TOOL="${ALPHA_DIR}/tools/alpha_config.py"

NUM_NODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}

# GPU count: explicit GPUS= > k8s resource hint > nvidia-smi autodetect > 1
if [ -z "${GPUS}" ]; then
    GPUS=${KUBERNETES_CONTAINER_RESOURCE_GPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
fi
if [ -z "${GPUS}" ] || [ "${GPUS}" -lt 1 ] 2>/dev/null; then
    GPUS=1
fi
GPUS_PER_NODE=${GPUS}

MODEL_SIZE=$1
LOAD_DIR=$2
SAVE_DIR=$3
MG2HF=$4
USE_CUDA=$5
PR=$6
HF_DIR=$7

# ============================================================================
# Auto Mode Detection (SAVE_DIR == "auto" or "auto:ITERATION")
# ============================================================================
if [[ "${SAVE_DIR}" == "auto"* ]]; then
    OUTPUT_DIR="${LOAD_DIR}"
    if [ ! -d "${OUTPUT_DIR}/checkpoints" ]; then
        echo "❌ Error: ${OUTPUT_DIR}/checkpoints not found (auto mode needs a training output dir)"
        exit 1
    fi
    if [[ "${SAVE_DIR}" == "auto:"* ]]; then
        ITERATION="${SAVE_DIR#auto:}"
        echo "📍 Using specified iteration: ${ITERATION}"
    else
        LATEST_FILE="${OUTPUT_DIR}/checkpoints/latest_checkpointed_iteration.txt"
        if [ ! -f "${LATEST_FILE}" ]; then
            echo "❌ Error: ${LATEST_FILE} not found"
            exit 1
        fi
        ITERATION=$(cat "${LATEST_FILE}" | tr -d '[:space:]')
        echo "📍 Using latest iteration: ${ITERATION}"
    fi
    ITERATION_PADDED=$(printf "%07d" $((10#${ITERATION})))
    LOAD_DIR="${OUTPUT_DIR}/checkpoints/iter_${ITERATION_PADDED}"
    SAVE_DIR="${OUTPUT_DIR}/hfmodel_${ITERATION_PADDED}"
    echo ""
    echo "🔄 Auto mode (GPUS=${GPUS} → EP=${GPUS}, torch_dist resharding):"
    echo "   Input:  ${LOAD_DIR}"
    echo "   Output: ${SAVE_DIR}"
    echo ""
    if [ ! -d "${LOAD_DIR}" ]; then
        echo "❌ Error: Checkpoint not found: ${LOAD_DIR}"
        ls -1 "${OUTPUT_DIR}/checkpoints/" | grep "iter_" | sed 's/^/     /'
        exit 1
    fi
fi

# ============================================================================
# Checkpoint Path Fix: Megatron's load_checkpoint() wants the parent dir +
# --ckpt-step, not the iter_NNNNNN dir directly.
# ============================================================================
CKPT_STEP=""
EMIT_CKPT_DIR="${LOAD_DIR}"   # dir that actually contains common.pt
if [[ "${LOAD_DIR}" =~ iter_([0-9]+) ]]; then
    ITERATION="${BASH_REMATCH[1]}"
    PARENT_DIR=$(dirname "${LOAD_DIR}")
    if [ -f "${PARENT_DIR}/latest_checkpointed_iteration.txt" ]; then
        echo "📍 Detected iter_${ITERATION}; loading from ${PARENT_DIR} with --ckpt-step ${ITERATION}"
        EMIT_CKPT_DIR="${LOAD_DIR}"     # keep pointing at the iter dir for common.pt
        LOAD_DIR="${PARENT_DIR}"
        CKPT_STEP="${ITERATION}"
    fi
fi

# ============================================================================
# Derive the Megatron model-skeleton flags (single source of truth)
#   MG→HF: from the checkpoint's common.pt (ground truth)
#   HF→MG: from the named YAML (no checkpoint to read)
# Read one-token-per-line into an array to avoid word-splitting / globbing the
# '*' in the hybrid-override-pattern.
# ============================================================================
MODEL_ARGS=()
if [ "${MG2HF}" = true ]; then
    readarray -t MODEL_ARGS < <(python3 "${ALPHA_CONFIG_TOOL}" emit-megatron-flags --from-checkpoint "${EMIT_CKPT_DIR}")
else
    readarray -t MODEL_ARGS < <(python3 "${ALPHA_CONFIG_TOOL}" emit-megatron-flags "${MODEL_SIZE}")
fi
if [ ${#MODEL_ARGS[@]} -eq 0 ]; then
    echo "❌ Error: failed to derive model flags (emit-megatron-flags returned nothing)"
    exit 1
fi

# Extract values we need from the derived flags (num-experts, tokenizer-model).
NUM_EXPERTS=""
TOKENIZER_PATH=""
for ((i=0; i<${#MODEL_ARGS[@]}; i++)); do
    case "${MODEL_ARGS[$i]}" in
        --num-experts)     NUM_EXPERTS="${MODEL_ARGS[$((i+1))]}" ;;
        --tokenizer-model) TOKENIZER_PATH="${MODEL_ARGS[$((i+1))]}" ;;
    esac
done

# ============================================================================
# Parallelism: EP = #GPU, TP=PP=1 (alpha constraint). Validate divisibility.
# ============================================================================
if [ -n "${NUM_EXPERTS}" ] && [ "${NUM_EXPERTS}" -gt 0 ] 2>/dev/null; then
    if [ $(( NUM_EXPERTS % GPUS )) -ne 0 ]; then
        echo "❌ Error: num_experts (${NUM_EXPERTS}) is not divisible by GPUS (${GPUS})."
        echo "   Pick a GPU count that divides ${NUM_EXPERTS} (e.g. 1, 2, 4, 8)."
        exit 1
    fi
fi
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size ${GPUS}
)

# ============================================================================
# Output config.json + tokenizer + HF modeling files (MG→HF), or copy HF json (HF→MG)
# ============================================================================
OTHER_ARGS=()
if [ "${MG2HF}" = true ]; then
    mkdir -p ${SAVE_DIR}

    if [ -z "${HF_DIR}" ] || [ ! -d "${HF_DIR}" ]; then
        echo "🔧 Auto-generating HF config.json from checkpoint (ground truth)..."
        python3 "${ALPHA_CONFIG_TOOL}" generate-hf-config --from-checkpoint "${EMIT_CKPT_DIR}" --output "${SAVE_DIR}/config.json"

        echo "📋 Copying tokenizer files from ${TOKENIZER_PATH}..."
        find -L ${TOKENIZER_PATH} -maxdepth 1 -type f -name "tokenizer*.json" -print0 2>/dev/null | xargs -0 -r cp -t ${SAVE_DIR} || true
        find -L ${TOKENIZER_PATH} -maxdepth 1 -type f -name "special_tokens_map.json" -print0 2>/dev/null | xargs -0 -r cp -t ${SAVE_DIR} || true
        find -L ${TOKENIZER_PATH} -maxdepth 1 -type f -name "vocab.json" -print0 2>/dev/null | xargs -0 -r cp -t ${SAVE_DIR} || true
        find -L ${TOKENIZER_PATH} -maxdepth 1 -type f -name "merges.txt" -print0 2>/dev/null | xargs -0 -r cp -t ${SAVE_DIR} || true
        find -L ${TOKENIZER_PATH} -maxdepth 1 -type f -name "*.model" -print0 2>/dev/null | xargs -0 -r cp -t ${SAVE_DIR} || true

        ALPHA_HF_MODEL_DIR="${ALPHA_DIR}/hf_model"
        if [ -d "${ALPHA_HF_MODEL_DIR}" ]; then
            echo "📋 Copying Alpha HF modeling files (for trust_remote_code)..."
            cp ${ALPHA_HF_MODEL_DIR}/*.py ${SAVE_DIR}/
        fi

        HF_DIR=${SAVE_DIR}
        echo "✅ HF config auto-generated at ${SAVE_DIR}/config.json"
    else
        echo "📋 Copying HF config from ${HF_DIR}..."
        find -L ${HF_DIR} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVE_DIR}
        find -L ${HF_DIR} -maxdepth 1 -type f -name "merges.txt" -print0 2>/dev/null | xargs -0 -r cp -t ${SAVE_DIR} || true
    fi

    # ------------------------------------------------------------------ G1 게이트
    # generation_config.json 을 쓰고, eos 가 챗 종료 토큰(<|im_end|>)을 담는지 검사한다.
    # 이것이 없으면 서버가 턴 종료를 인식하지 못해 max_tokens 까지 생성하고, 벤치가
    # 모델이 아니라 서빙 설정을 측정하게 된다 — 2026-08-30 전량 무효 사고의 원인.
    # 판정 기준·나머지 게이트: examples/alpha/docs/SFT_BENCHMARKS.md §7.
    echo "🔒 G1: generation_config.json 생성·eos 정합성 검사..."
    if ! python3 "${ALPHA_DIR}/tools/emit_generation_config.py" "${SAVE_DIR}"; then
        echo "❌ G1 실패 — 이 체크포인트로 벤치를 돌리지 말 것."
        exit 1
    fi

    OTHER_ARGS+=(
        --hf-dir ${HF_DIR}
        --mcore2hf
    )
else
    # HF → Megatron: override tokenizer-model to the HF source dir (last wins).
    OTHER_ARGS+=(
        --tokenizer-model ${LOAD_DIR}
    )
    mkdir -p ${SAVE_DIR}
    find -L ${LOAD_DIR} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVE_DIR}
    find -L ${LOAD_DIR} -maxdepth 1 -type f -name "merges.txt" -print0 2>/dev/null | xargs -0 -r cp -t ${SAVE_DIR} || true
fi

[ "${USE_CUDA}" = true ] && OTHER_ARGS+=(--use-gpu)
if [ "${PR}" = fp16 ]; then OTHER_ARGS+=(--fp16); elif [ "${PR}" = bf16 ]; then OTHER_ARGS+=(--bf16); fi

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# Runtime knobs that are not part of the model skeleton (kept minimal).
RUNTIME_ARGS=(
    --attention-backend auto
    --micro-batch-size 1
    --bf16
)

CONVERT_ARGS=(
    --model-type GPT
    --load-dir ${LOAD_DIR}
    --save-dir ${SAVE_DIR}
    --no-load-optim
    --no-load-rng
    --logging-level 1
    --synchronizer alpha
    --pretrain-script alpha.model_provider
    --auto-detect-ckpt-format
)
[ -n "${CKPT_STEP}" ] && CONVERT_ARGS+=(--ckpt-step ${CKPT_STEP})

cd ${CONVERTOR_DIR}

echo "🚀 Converting with EP=${GPUS} (TP=PP=1), precision=${PR}"
echo "   num_experts=${NUM_EXPERTS:-?}, tokenizer=${TOKENIZER_PATH:-?}"

# Direct quoted-array invocation (no eval) so the '*' in the hybrid pattern is
# passed verbatim and never glob-expands.
torchrun "${DISTRIBUTED_ARGS[@]}" impl/convert.py \
    "${MODEL_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${RUNTIME_ARGS[@]}" \
    "${CONVERT_ARGS[@]}" \
    "${OTHER_ARGS[@]}"

if [ "${MG2HF}" = true ]; then
    if [ -f "${SAVE_DIR}/configuration_alpha.py" ] && [ -f "${SAVE_DIR}/modeling_alpha.py" ]; then
        echo "✅ Alpha HF model conversion complete: ${SAVE_DIR}"
    else
        echo "⚠️ Warning: Alpha HF modeling files missing in ${SAVE_DIR}"
    fi
fi
