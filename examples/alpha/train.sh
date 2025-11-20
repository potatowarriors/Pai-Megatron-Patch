#!/bin/bash
#
# Alpha 프로젝트 통합 학습 스크립트
# YAML 기반 설정으로 Qwen3-Next Mamba 모델 학습
#
# 사용법:
#   bash train.sh [model_config] [training_config] [infra_config] [data_config]
#
# 예시:
#   bash train.sh  # 기본 설정 사용
#   bash train.sh baseline_24L pretrain h100x8 kormo_1pct
#
# 설정 파일 위치:
#   configs/model/*.yaml
#   configs/training/*.yaml
#   configs/data/*.yaml
#   configs/env.yaml

set -e

#==============================================================================
# WANDB 환경 자동 로드 (Optional)
#==============================================================================

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"

# WANDB 설정 스크립트가 있으면 자동으로 source
WANDB_SETUP_SCRIPT="${CURRENT_DIR}/scripts/setup_wandb.sh"
if [ -f "$WANDB_SETUP_SCRIPT" ]; then
    source "$WANDB_SETUP_SCRIPT" 2>/dev/null || true
fi

#==============================================================================
# 경로 및 기본 설정
#==============================================================================

MEGATRON_PATCH_PATH=$( dirname $( dirname ${CURRENT_DIR}))
CONFIG_DIR="${CURRENT_DIR}/configs"

# 설정 파일 인자 (기본값)
MODEL_CONFIG=${1:-"baseline_24L"}
TRAINING_CONFIG=${2:-"pretrain"}
INFRA_CONFIG=${3:-"h100x8"}
DATA_CONFIG=${4:-"kormo_1pct"}

echo "=============================================="
echo "Alpha 프로젝트 학습 설정"
echo "=============================================="
echo "모델 설정: ${MODEL_CONFIG}"
echo "학습 설정: ${TRAINING_CONFIG}"
echo "인프라 설정: ${INFRA_CONFIG}"
echo "데이터 설정: ${DATA_CONFIG}"
echo "=============================================="
echo ""

#==============================================================================
# YAML 파싱 헬퍼 함수
#==============================================================================

# yaml_get: YAML 파일에서 값 추출
# 사용: yaml_get <file> <key_path>
# 예: yaml_get configs/model/baseline_24L.yaml "model.num_layers"
yaml_get() {
    local file=$1
    local key=$2
    python3 -c "
import yaml
import sys
with open('$file', 'r') as f:
    data = yaml.safe_load(f)
keys = '$key'.split('.')
value = data
for k in keys:
    value = value.get(k, '')
print(value if value != '' else '')
" 2>/dev/null || echo ""
}

#==============================================================================
# 환경 변수 설정 (env.yaml에서 로드)
#==============================================================================

ENV_CONFIG="${CONFIG_DIR}/env.yaml"

if [ -f "$ENV_CONFIG" ]; then
    echo "환경 변수 설정 중..."

    # Megatron 경로
    MEGATRON_VERSION=$(yaml_get $ENV_CONFIG "environment.megatron_version")
    export PYTHONPATH=${MEGATRON_PATCH_PATH}:${MEGATRON_PATCH_PATH}/backends/megatron/${MEGATRON_VERSION}:$PYTHONPATH

    # CUDA 설정
    export CUDA_DEVICE_MAX_CONNECTIONS=$(yaml_get $ENV_CONFIG "environment.cuda.device_max_connections")
    ALLOC_CONF=$(yaml_get $ENV_CONFIG "environment.cuda.alloc_conf")
    export PYTORCH_CUDA_ALLOC_CONF=$ALLOC_CONF

    # PyTorch 설정
    TORCH_FORCE=$(yaml_get $ENV_CONFIG "environment.pytorch.force_no_weights_only_load")
    if [ "$TORCH_FORCE" = "True" ] || [ "$TORCH_FORCE" = "true" ]; then
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
    fi

    # TransformerEngine
    export NVTE_NORM_FWD_USE_CUDNN=$(yaml_get $ENV_CONFIG "environment.transformer_engine.norm_fwd_use_cudnn")
    export NVTE_NORM_BWD_USE_CUDNN=$(yaml_get $ENV_CONFIG "environment.transformer_engine.norm_bwd_use_cudnn")
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=$(yaml_get $ENV_CONFIG "environment.transformer_engine.allow_nondeterministic_algo")
    export NVTE_FLASH_ATTN=$(yaml_get $ENV_CONFIG "environment.transformer_engine.flash_attn")
    export NVTE_FUSED_ATTN=$(yaml_get $ENV_CONFIG "environment.transformer_engine.fused_attn")

    # NCCL
    export NCCL_DEBUG=$(yaml_get $ENV_CONFIG "environment.nccl.debug")

    # CPU 스레드
    export OMP_NUM_THREADS=$(yaml_get $ENV_CONFIG "environment.threading.omp_num_threads")
    export NUMEXPR_MAX_THREADS=$(yaml_get $ENV_CONFIG "environment.threading.numexpr_max_threads")
    export MKL_NUM_THREADS=$(yaml_get $ENV_CONFIG "environment.threading.mkl_num_threads")

    # 프로파일링 및 모드
    export PROFILE=$(yaml_get $ENV_CONFIG "environment.profiling.enabled" | python3 -c "import sys; print(1 if sys.stdin.read().strip().lower() == 'true' else 0)")
    export PRETRAIN=$(yaml_get $ENV_CONFIG "environment.mode.pretrain" | python3 -c "import sys; print(1 if sys.stdin.read().strip().lower() == 'true' else 0)")
    export MOE_GROUPED_GEMM=$(yaml_get $ENV_CONFIG "environment.mode.moe_grouped_gemm" | python3 -c "import sys; print('true' if sys.stdin.read().strip().lower() == 'true' else 'false')")

    echo "✅ 환경 변수 설정 완료"
else
    echo "❌ 환경 설정 파일을 찾을 수 없습니다: $ENV_CONFIG"
    exit 1
fi

echo ""

#==============================================================================
# 분산 학습 설정
#==============================================================================

NUM_NODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU:-$(python3 -c "import torch; print(torch.cuda.device_count())")}
[ -z "$MASTER_ADDR" ] && export MASTER_ADDR=localhost
[ -z "$MASTER_PORT" ] && export MASTER_PORT=${MASTER_PORT:-$(shuf -n 1 -i 10000-65535)}

echo "=============================================="
echo "분산 학습 설정"
echo "=============================================="
echo "노드 수: ${NUM_NODES}"
echo "노드 순위: ${NODE_RANK}"
echo "노드당 GPU: ${GPUS_PER_NODE}"
echo "Master 주소: ${MASTER_ADDR}"
echo "Master 포트: ${MASTER_PORT}"
echo "=============================================="
echo ""

DISTRIBUTED_ARGS=(
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --nproc_per_node $GPUS_PER_NODE
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

#==============================================================================
# 인프라 설정 로드 (h100x8.yaml)
#==============================================================================

INFRA_CONFIG_FILE="${CONFIG_DIR}/training/${INFRA_CONFIG}.yaml"

if [ ! -f "$INFRA_CONFIG_FILE" ]; then
    echo "❌ 인프라 설정 파일을 찾을 수 없습니다: $INFRA_CONFIG_FILE"
    exit 1
fi

echo "인프라 설정 로드 중: ${INFRA_CONFIG}.yaml"

# 병렬화 설정
TP=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.parallelism.tensor_parallel")
PP=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.parallelism.pipeline_parallel")
EP=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.parallelism.expert_parallel")
ETP=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.parallelism.expert_tensor_parallel")
CP=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.parallelism.context_parallel")

# 배치 설정
MBS=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.batch.micro_batch_size")
GBS=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.batch.global_batch_size")
SEQ_LEN=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.batch.seq_length")
MAX_POS_EMB=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.batch.max_position_embeddings")

echo "✅ 인프라 설정 완료"
echo "  - 병렬화: TP=${TP}, PP=${PP}, EP=${EP}, ETP=${ETP}, CP=${CP}"
echo "  - 배치: MBS=${MBS}, GBS=${GBS}, SEQ=${SEQ_LEN}"
echo ""

#==============================================================================
# 학습 설정 로드 (pretrain.yaml)
#==============================================================================

TRAINING_CONFIG_FILE="${CONFIG_DIR}/training/${TRAINING_CONFIG}.yaml"

if [ ! -f "$TRAINING_CONFIG_FILE" ]; then
    echo "❌ 학습 설정 파일을 찾을 수 없습니다: $TRAINING_CONFIG_FILE"
    exit 1
fi

echo "학습 설정 로드 중: ${TRAINING_CONFIG}.yaml"

# 토큰 및 Iteration 계산
TRAIN_TOKENS=$(yaml_get $TRAINING_CONFIG_FILE "training.train_tokens")
WARMUP_TOKENS=$(yaml_get $TRAINING_CONFIG_FILE "training.warmup_tokens")

TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GBS} / ${SEQ_LEN} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS} / ${GBS} / ${SEQ_LEN} ))
LR_DECAY_ITERS=${TRAIN_ITERS}

# Learning Rate
LR=$(yaml_get $TRAINING_CONFIG_FILE "training.lr")
MIN_LR=$(yaml_get $TRAINING_CONFIG_FILE "training.min_lr")
LR_DECAY_STYLE=$(yaml_get $TRAINING_CONFIG_FILE "training.lr_decay_style")

# Optimizer
OPTIMIZER=$(yaml_get $TRAINING_CONFIG_FILE "training.optimizer")
WEIGHT_DECAY=$(yaml_get $TRAINING_CONFIG_FILE "training.weight_decay")
ADAM_BETA1=$(yaml_get $TRAINING_CONFIG_FILE "training.adam_beta1")
ADAM_BETA2=$(yaml_get $TRAINING_CONFIG_FILE "training.adam_beta2")
INIT_STD=$(yaml_get $TRAINING_CONFIG_FILE "training.init_method_std")
CLIP_GRAD=$(yaml_get $TRAINING_CONFIG_FILE "training.clip_grad")

# 체크포인트 및 로깅
SAVE_INTERVAL=$(yaml_get $TRAINING_CONFIG_FILE "training.save_interval")
EVAL_ITERS=$(yaml_get $TRAINING_CONFIG_FILE "training.eval_iters")
EVAL_INTERVAL=$(yaml_get $TRAINING_CONFIG_FILE "training.eval_interval")
LOG_INTERVAL=$(yaml_get $TRAINING_CONFIG_FILE "training.log_interval")
DIST_TIMEOUT=$(yaml_get $TRAINING_CONFIG_FILE "training.distributed_timeout_minutes")
MANUAL_GC_INTERVAL=$(yaml_get $TRAINING_CONFIG_FILE "training.manual_gc_interval")

echo "✅ 학습 설정 완료"
echo "  - 학습 토큰: ${TRAIN_TOKENS}, Iterations: ${TRAIN_ITERS}"
echo "  - Learning Rate: ${LR} → ${MIN_LR}"
echo ""

#==============================================================================
# WANDB 설정 로드 (Optional)
#==============================================================================

WANDB_ENABLED=$(yaml_get $TRAINING_CONFIG_FILE "training.wandb.enabled")

if [ "$WANDB_ENABLED" = "True" ] || [ "$WANDB_ENABLED" = "true" ]; then
    echo "WANDB 설정 로드 중..."

    WANDB_PROJECT=$(yaml_get $TRAINING_CONFIG_FILE "training.wandb.project")
    WANDB_ENTITY=$(yaml_get $TRAINING_CONFIG_FILE "training.wandb.entity")
    WANDB_SAVE_DIR=$(yaml_get $TRAINING_CONFIG_FILE "training.wandb.save_dir")

    # Experiment name: 자동 생성 (모델_데이터_시간)
    WANDB_EXP_NAME="${MODEL_CONFIG}_${DATA_CONFIG}_${TIMESTAMP}"

    # API 키 확인
    if [ -z "$WANDB_API_KEY" ]; then
        echo "⚠️  경고: WANDB_API_KEY가 설정되지 않았습니다."
        echo "   WANDB 로깅을 건너뜁니다. API 키 설정:"
        echo "   export WANDB_API_KEY='your_api_key'"
        WANDB_ENABLED="false"
    else
        echo "✅ WANDB 설정 완료"
        echo "  - Project: ${WANDB_PROJECT}"
        echo "  - Run Name: ${WANDB_EXP_NAME}"
        [ ! -z "$WANDB_ENTITY" ] && echo "  - Entity: ${WANDB_ENTITY}"
    fi
else
    echo "WANDB: 비활성화됨"
    WANDB_ENABLED="false"
fi

echo ""

#==============================================================================
# 데이터 설정 로드 (kormo_1pct.yaml)
#==============================================================================

DATA_CONFIG_FILE="${CONFIG_DIR}/data/${DATA_CONFIG}.yaml"

if [ ! -f "$DATA_CONFIG_FILE" ]; then
    echo "❌ 데이터 설정 파일을 찾을 수 없습니다: $DATA_CONFIG_FILE"
    exit 1
fi

echo "데이터 설정 로드 중: ${DATA_CONFIG}.yaml"

DATA_PATH=$(yaml_get $DATA_CONFIG_FILE "data.train_path")
DATASET_IMPL=$(yaml_get $DATA_CONFIG_FILE "data.dataset_impl")
NUM_WORKERS=$(yaml_get $DATA_CONFIG_FILE "data.num_workers")
DATA_SPLIT=$(yaml_get $DATA_CONFIG_FILE "data.split")

# Split 배열 파싱 [99, 1, 0] → "99,1,0"
DATA_SPLIT=$(echo $DATA_SPLIT | tr -d '[]' | tr ' ' ',')

# 데이터 존재 확인
if [ ! -f "${DATA_PATH}.bin" ] || [ ! -f "${DATA_PATH}.idx" ]; then
    echo "❌ 오류: 데이터셋 파일을 찾을 수 없습니다!"
    echo "   필요: ${DATA_PATH}.bin"
    echo "   필요: ${DATA_PATH}.idx"
    echo ""
    echo "데이터 전처리를 먼저 실행하세요:"
    echo "  cd toolkits/pretrain_data_preprocessing/"
    echo "  bash preprocess_kormo_subset.sh 1"
    exit 1
fi

echo "✅ 데이터 설정 완료"
echo "  - 경로: ${DATA_PATH}"
echo "  - Split: ${DATA_SPLIT}"
echo ""

#==============================================================================
# 모델 설정 로드 (baseline_24L.yaml)
#==============================================================================

MODEL_CONFIG_FILE="${CONFIG_DIR}/model/${MODEL_CONFIG}.yaml"

if [ ! -f "$MODEL_CONFIG_FILE" ]; then
    echo "❌ 모델 설정 파일을 찾을 수 없습니다: $MODEL_CONFIG_FILE"
    exit 1
fi

echo "모델 설정 로드 중: ${MODEL_CONFIG}.yaml"

# 기본 아키텍처
NUM_LAYERS=$(yaml_get $MODEL_CONFIG_FILE "model.num_layers")
HIDDEN_SIZE=$(yaml_get $MODEL_CONFIG_FILE "model.hidden_size")
FFN_HIDDEN_SIZE=$(yaml_get $MODEL_CONFIG_FILE "model.ffn_hidden_size")
NUM_ATTN_HEADS=$(yaml_get $MODEL_CONFIG_FILE "model.num_attention_heads")
KV_CHANNELS=$(yaml_get $MODEL_CONFIG_FILE "model.kv_channels")
NUM_QUERY_GROUPS=$(yaml_get $MODEL_CONFIG_FILE "model.num_query_groups")

# MoE
MOE_FFN_HIDDEN=$(yaml_get $MODEL_CONFIG_FILE "model.moe.moe_ffn_hidden_size")
NUM_EXPERTS=$(yaml_get $MODEL_CONFIG_FILE "model.moe.num_experts")
ROUTER_TOPK=$(yaml_get $MODEL_CONFIG_FILE "model.moe.router_topk")
MOE_AUX_LOSS=$(yaml_get $MODEL_CONFIG_FILE "model.moe.aux_loss_coeff")
SHARED_EXPERT_SIZE=$(yaml_get $MODEL_CONFIG_FILE "model.moe.shared_expert_intermediate_size")

# Hybrid
HYBRID_ATTN_RATIO=$(yaml_get $MODEL_CONFIG_FILE "model.hybrid.attention_ratio")
HYBRID_MLP_RATIO=$(yaml_get $MODEL_CONFIG_FILE "model.hybrid.mlp_ratio")
HYBRID_PATTERN=$(yaml_get $MODEL_CONFIG_FILE "model.hybrid.override_pattern")
MAMBA_STATE_DIM=$(yaml_get $MODEL_CONFIG_FILE "model.hybrid.mamba_state_dim")
MAMBA_HEAD_DIM=$(yaml_get $MODEL_CONFIG_FILE "model.hybrid.mamba_head_dim")
MAMBA_NUM_GROUPS=$(yaml_get $MODEL_CONFIG_FILE "model.hybrid.mamba_num_groups")
MAMBA_NUM_HEADS=$(yaml_get $MODEL_CONFIG_FILE "model.hybrid.mamba_num_heads")

# RoPE
ROTARY_BASE=$(yaml_get $MODEL_CONFIG_FILE "model.rotary_base")
ROTARY_PERCENT=$(yaml_get $MODEL_CONFIG_FILE "model.rotary_percent")

# Tokenizer
PADDED_VOCAB_SIZE=$(yaml_get $MODEL_CONFIG_FILE "model.padded_vocab_size")
TOKENIZER_TYPE=$(yaml_get $MODEL_CONFIG_FILE "model.tokenizer_type")
TOKENIZER_PATH=$(yaml_get $MODEL_CONFIG_FILE "model.tokenizer_path")

echo "✅ 모델 설정 완료"
echo "  - 아키텍처: ${NUM_LAYERS} layers, ${HIDDEN_SIZE} hidden, ${NUM_EXPERTS} experts"
echo "  - Hybrid: ${HYBRID_ATTN_RATIO} attention ratio, pattern: ${HYBRID_PATTERN}"
echo ""

#==============================================================================
# 출력 디렉토리 설정
#==============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${CURRENT_DIR}/outputs/alpha_${MODEL_CONFIG}_${TIMESTAMP}
TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard
CHECKPOINT_DIR=${OUTPUT_DIR}/checkpoints
LOG_DIR=${OUTPUT_DIR}/logs

mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${LOG_DIR}

echo "=============================================="
echo "출력 경로"
echo "=============================================="
echo "출력 디렉토리: ${OUTPUT_DIR}"
echo "TensorBoard: ${TENSORBOARD_DIR}"
echo "체크포인트: ${CHECKPOINT_DIR}"
echo "로그: ${LOG_DIR}"
echo "=============================================="
echo ""

#==============================================================================
# Megatron 인자 구성
#==============================================================================

MODEL_ARGS=(
    --use-mcore-models
    --transformer-impl transformer_engine

    # 기본 아키텍처
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --ffn-hidden-size ${FFN_HIDDEN_SIZE}
    --num-attention-heads ${NUM_ATTN_HEADS}
    --kv-channels ${KV_CHANNELS}

    # Group Query Attention
    --group-query-attention
    --num-query-groups ${NUM_QUERY_GROUPS}

    # MoE 설정
    --moe-ffn-hidden-size ${MOE_FFN_HIDDEN}
    --num-experts ${NUM_EXPERTS}
    --moe-router-topk ${ROUTER_TOPK}
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff ${MOE_AUX_LOSS}
    --moe-router-score-function softmax
    --moe-router-dtype fp32
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-router-fusion
    --moe-shared-expert-intermediate-size ${SHARED_EXPERT_SIZE}

    # Hybrid Model
    --hybrid-attention-ratio ${HYBRID_ATTN_RATIO}
    --hybrid-mlp-ratio ${HYBRID_MLP_RATIO}
    --hybrid-override-pattern ${HYBRID_PATTERN}
    --is-hybrid-model
    --mamba-state-dim ${MAMBA_STATE_DIM}
    --mamba-head-dim ${MAMBA_HEAD_DIM}
    --mamba-num-groups ${MAMBA_NUM_GROUPS}
    --mamba-num-heads ${MAMBA_NUM_HEADS}

    # Normalization
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --qk-layernorm
    --apply-layernorm-1p

    # Activation & Dropout
    --swiglu
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --disable-bias-linear

    # Positional Encoding (RoPE)
    --use-rotary-position-embeddings
    --rotary-base ${ROTARY_BASE}
    --rotary-percent ${ROTARY_PERCENT}
    --position-embedding-type rope
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${MAX_POS_EMB}

    # Tokenizer & Vocab
    --untie-embeddings-and-output-weights
    --padded-vocab-size ${PADDED_VOCAB_SIZE}
    --patch-tokenizer-type ${TOKENIZER_TYPE}
)

TRAINING_ARGS=(
    # 데이터
    --data-path ${DATA_PATH}
    --split ${DATA_SPLIT}
    --dataset ${DATASET_IMPL}
    --num-workers ${NUM_WORKERS}

    # 배치 및 Iteration
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --train-iters ${TRAIN_ITERS}

    # 토크나이저 로드
    --load ${TOKENIZER_PATH}
    --no-load-optim
    --no-load-rng

    # Optimizer
    --optimizer ${OPTIMIZER}
    --weight-decay ${WEIGHT_DECAY}
    --adam-beta1 ${ADAM_BETA1}
    --adam-beta2 ${ADAM_BETA2}
    --init-method-std ${INIT_STD}
    --clip-grad ${CLIP_GRAD}

    # Learning Rate
    --lr ${LR}
    --lr-decay-style ${LR_DECAY_STYLE}
    --min-lr ${MIN_LR}
    --lr-decay-iters ${LR_DECAY_ITERS}
    --lr-warmup-iters ${LR_WARMUP_ITERS}

    # Precision
    --bf16

    # 체크포인트 및 로깅
    --save ${CHECKPOINT_DIR}
    --save-interval ${SAVE_INTERVAL}
    --no-save-optim
    --ckpt-format torch_dist

    # 평가
    --eval-iters ${EVAL_ITERS}
    --eval-interval ${EVAL_INTERVAL}

    # TensorBoard
    --tensorboard-dir ${TENSORBOARD_DIR}
    --tensorboard-queue-size 1
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-throughput
    --log-interval ${LOG_INTERVAL}

    # 기타
    --distributed-timeout-minutes ${DIST_TIMEOUT}
    --manual-gc
    --manual-gc-interval ${MANUAL_GC_INTERVAL}
)

#==============================================================================
# WANDB Arguments (조건부 추가)
#==============================================================================

if [ "$WANDB_ENABLED" = "True" ] || [ "$WANDB_ENABLED" = "true" ]; then
    echo "📊 WANDB 인자 추가 중..."

    TRAINING_ARGS+=(
        --wandb-project "${WANDB_PROJECT}"
        --wandb-exp-name "${WANDB_EXP_NAME}"
    )

    # Optional arguments
    if [ ! -z "$WANDB_ENTITY" ]; then
        TRAINING_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
    fi

    if [ ! -z "$WANDB_SAVE_DIR" ]; then
        TRAINING_ARGS+=(--wandb-save-dir "${WANDB_SAVE_DIR}")
    fi

    echo "  ✅ WANDB 인자 추가 완료"
fi

# Activation Checkpointing (YAML에서 로드)
AC_GRANULARITY=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.activation_checkpointing.granularity")
if [ ! -z "$AC_GRANULARITY" ]; then
    INFRA_ARGS+=(--recompute-granularity ${AC_GRANULARITY})

    # Recompute modules
    RECOMPUTE_MODULES=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.activation_checkpointing.recompute_modules")
    if [ ! -z "$RECOMPUTE_MODULES" ]; then
        # YAML 배열을 공백으로 구분된 문자열로 변환
        MODULES=$(echo $RECOMPUTE_MODULES | tr -d '[],' | tr "'" ' ')
        INFRA_ARGS+=(--recompute-modules ${MODULES})
    fi
fi

INFRA_ARGS=(
    # 병렬화
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --expert-model-parallel-size ${EP}
    --expert-tensor-parallel-size ${ETP}
    --context-parallel-size ${CP}

    # Activation Checkpointing
    --recompute-granularity ${AC_GRANULARITY}
    --recompute-modules layernorm moe_act shared_experts

    # 최적화
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather

    # Attention Backend (Flash Attention for H100)
    --attention-backend flash

    # Loss Fusion
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te

    # MoE Token Dispatcher
    --moe-token-dispatcher-type alltoall
)

# TP > 1일 때만 통신 최적화 활성화
if [ $TP -gt 1 ]; then
    echo "  - TP > 1: Sequence Parallel 및 TP Comm Overlap 활성화"
    INFRA_ARGS+=(--sequence-parallel)
    INFRA_ARGS+=(--tp-comm-overlap)
fi

# Optimizer CPU Offload (YAML에서 로드, optional)
OPT_CPU_OFFLOAD=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.optimizations.optimizer_cpu_offload")
if [ "$OPT_CPU_OFFLOAD" = "True" ] || [ "$OPT_CPU_OFFLOAD" = "true" ]; then
    echo "  - Optimizer CPU Offload: 활성화"
    INFRA_ARGS+=(--use-cpu-optimizer-offload)
    INFRA_ARGS+=(--use-precision-aware-optimizer)

    # Offload fraction (optional)
    OFFLOAD_FRAC=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.optimizations.optimizer_offload_fraction")
    if [ ! -z "$OFFLOAD_FRAC" ]; then
        INFRA_ARGS+=(--optimizer-offload-fraction ${OFFLOAD_FRAC})
    fi
fi

#==============================================================================
# 학습 실행
#==============================================================================

echo "=============================================="
echo "🚀 Alpha 프로젝트 학습 시작!"
echo "=============================================="
echo ""

cmd="torchrun ${DISTRIBUTED_ARGS[@]} ${CURRENT_DIR}/pretrain_alpha.py \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INFRA_ARGS[@]}"

echo "실행 명령:"
echo "$cmd"
echo ""
echo "로그 파일: ${LOG_DIR}/train.log"
echo ""

# 설정 스냅샷 저장
CONFIG_SNAPSHOT="${LOG_DIR}/config_snapshot_${TIMESTAMP}.txt"
cat > ${CONFIG_SNAPSHOT} <<EOF
============================================
Alpha 프로젝트 학습 설정 스냅샷
생성 시각: $(date)
============================================

[설정 파일]
- 모델: ${MODEL_CONFIG}.yaml
- 학습: ${TRAINING_CONFIG}.yaml
- 인프라: ${INFRA_CONFIG}.yaml
- 데이터: ${DATA_CONFIG}.yaml

[병렬화]
- Tensor Parallel: ${TP}
- Pipeline Parallel: ${PP}
- Expert Parallel: ${EP}
- Expert Tensor Parallel: ${ETP}
- Context Parallel: ${CP}
- Data Parallel: $((GPUS_PER_NODE / (TP * PP * EP * CP)))

[모델 아키텍처]
- Layers: ${NUM_LAYERS}
- Hidden Size: ${HIDDEN_SIZE}
- Attention Heads: ${NUM_ATTN_HEADS}
- KV Channels: ${KV_CHANNELS}
- Query Groups: ${NUM_QUERY_GROUPS}
- Experts: ${NUM_EXPERTS}
- Router TopK: ${ROUTER_TOPK}
- Hybrid Pattern: ${HYBRID_PATTERN}

[학습 설정]
- Global Batch Size: ${GBS}
- Micro Batch Size: ${MBS}
- Sequence Length: ${SEQ_LEN}
- Total Iterations: ${TRAIN_ITERS}
- Learning Rate: ${LR} → ${MIN_LR}
- Optimizer: ${OPTIMIZER}

[데이터]
- Path: ${DATA_PATH}
- Split: ${DATA_SPLIT}
- Workers: ${NUM_WORKERS}

[출력]
- Checkpoints: ${CHECKPOINT_DIR}
- TensorBoard: ${TENSORBOARD_DIR}
- Logs: ${LOG_DIR}

============================================
EOF

echo "✅ 설정 스냅샷 저장: ${CONFIG_SNAPSHOT}"
echo ""

# 실행 및 로그 저장
eval $cmd 2>&1 | tee ${LOG_DIR}/train.log

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✅ 학습 완료!"
    echo "=============================================="
    echo "체크포인트: ${CHECKPOINT_DIR}"
    echo "TensorBoard: ${TENSORBOARD_DIR}"
    echo "로그: ${LOG_DIR}"
    echo ""
    echo "TensorBoard 실행:"
    echo "  tensorboard --logdir ${TENSORBOARD_DIR} --port 6006"
else
    echo ""
    echo "=============================================="
    echo "❌ 학습 실패 (Exit Code: ${EXIT_CODE})"
    echo "=============================================="
    echo "로그 확인: ${LOG_DIR}/train.log"
fi

exit $EXIT_CODE
