#!/bin/bash
#
# Alpha 프로젝트 Stage 2-3 학습 스크립트
# Stage 2-2 (375k에서 조기 중단) 체크포인트에서 800k까지 추가 425k iter 학습
#
# train_stage2_2.sh 대비 변경사항:
#   - LR 4× 상향: 1e-4 → 4e-4 (다른 technical report 참조)
#   - Min LR: 1e-5 → 4e-5 (10% of max)
#   - Warmup: 500 → 2000 (큰 LR jump 안정화)
#   - LayerNorm WD 신기능 도입 (apply_wd_to_all_layernorm — pretrain_alpha.py monkey-patch)
#   - Output dir: alpha_<model>_stage2_<TS> → alpha_<model>_stage2_3_<TS>
#
# 사용법:
#   bash train_stage2_3.sh [model_config] [training_config] [infra_config] [data_config]
#
# 예시:
#   bash train_stage2_3.sh baseline_48L stage2_3 h100x8 stage2_2
#   (data config은 stage2_2 그대로 — consumed_samples 이어서)

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

# 설정 파일 인자 (기본값) — stage2_3 defaults
MODEL_CONFIG=${1:-"baseline_48L"}
TRAINING_CONFIG=${2:-"stage2_3"}
INFRA_CONFIG=${3:-"h100x8"}
DATA_CONFIG=${4:-"stage2_2"}

echo "=============================================="
echo "Alpha 프로젝트 Stage 2-3 학습 설정"
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
# 예: yaml_get configs/model/baseline_48L.yaml "model.num_layers"
yaml_get() {
    local file="$1"
    local key="$2"
    python3 -c "
import yaml
import sys
with open('${file}', 'r') as f:
    data = yaml.safe_load(f)
keys = '${key}'.split('.')
value = data
for k in keys:
    if isinstance(value, dict):
        value = value.get(k, '')
    else:
        value = ''
        break
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
    # NVTE_FLASH_ATTN / NVTE_FUSED_ATTN: --attention-backend가 관리
    # 수동 export 시 Megatron _set_attention_backend() assert 충돌 발생

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
# 인프라 설정 로드 (h100x8_deepep.yaml)
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
# 학습 설정 로드 (stage2 YAML에서 직접 iteration 읽기)
#==============================================================================

TRAINING_CONFIG_FILE="${CONFIG_DIR}/training/${TRAINING_CONFIG}.yaml"

if [ ! -f "$TRAINING_CONFIG_FILE" ]; then
    echo "❌ 학습 설정 파일을 찾을 수 없습니다: $TRAINING_CONFIG_FILE"
    exit 1
fi

echo "학습 설정 로드 중: ${TRAINING_CONFIG}.yaml"

# Iteration (YAML에서 직접 읽기 — token 변환 없음)
TRAIN_ITERS=$(yaml_get $TRAINING_CONFIG_FILE "training.train_iters")
LR_WARMUP_ITERS=$(yaml_get $TRAINING_CONFIG_FILE "training.lr_warmup_iters")
LR_DECAY_ITERS=$(yaml_get $TRAINING_CONFIG_FILE "training.lr_decay_iters")

if [ -z "$TRAIN_ITERS" ] || [ "$TRAIN_ITERS" = "" ]; then
    echo "❌ train_iters가 YAML에 정의되지 않았습니다."
    echo "   Stage 2 config에는 train_iters, lr_warmup_iters, lr_decay_iters가 필요합니다."
    exit 1
fi

echo "📊 Stage 2 iterations: train=${TRAIN_ITERS}, warmup=${LR_WARMUP_ITERS}, decay=${LR_DECAY_ITERS}"

# Learning Rate
LR=$(yaml_get $TRAINING_CONFIG_FILE "training.lr")
MIN_LR=$(yaml_get $TRAINING_CONFIG_FILE "training.min_lr")
LR_DECAY_STYLE=$(yaml_get $TRAINING_CONFIG_FILE "training.lr_decay_style")

# WSD Scheduler (변경 B: 추가 변수)
LR_WSD_DECAY_STYLE=$(yaml_get $TRAINING_CONFIG_FILE "training.lr_wsd_decay_style")
LR_WSD_DECAY_ITERS=$(yaml_get $TRAINING_CONFIG_FILE "training.lr_wsd_decay_iters")

# QK-Clip
QK_CLIP=$(yaml_get $TRAINING_CONFIG_FILE "training.qk_clip")
QK_CLIP_ALPHA=$(yaml_get $TRAINING_CONFIG_FILE "training.qk_clip_alpha")
QK_CLIP_THRESHOLD=$(yaml_get $TRAINING_CONFIG_FILE "training.qk_clip_threshold")
NO_WEIGHT_DECAY_COND_TYPE=$(yaml_get $TRAINING_CONFIG_FILE "training.no_weight_decay_cond_type")

# Optimizer
OPTIMIZER=$(yaml_get $TRAINING_CONFIG_FILE "training.optimizer")
WEIGHT_DECAY=$(yaml_get $TRAINING_CONFIG_FILE "training.weight_decay")
ADAM_BETA1=$(yaml_get $TRAINING_CONFIG_FILE "training.adam_beta1")
ADAM_BETA2=$(yaml_get $TRAINING_CONFIG_FILE "training.adam_beta2")
INIT_STD=$(yaml_get $TRAINING_CONFIG_FILE "training.init_method_std")
CLIP_GRAD=$(yaml_get $TRAINING_CONFIG_FILE "training.clip_grad")

# Muon hyperparameters
MUON_MOMENTUM=$(yaml_get $TRAINING_CONFIG_FILE "training.muon_momentum")
MUON_USE_NESTEROV=$(yaml_get $TRAINING_CONFIG_FILE "training.muon_use_nesterov")
MUON_NUM_NS_STEPS=$(yaml_get $TRAINING_CONFIG_FILE "training.muon_num_ns_steps")
MUON_SCALE_MODE=$(yaml_get $TRAINING_CONFIG_FILE "training.muon_scale_mode")
MUON_FP32_MATMUL_PREC=$(yaml_get $TRAINING_CONFIG_FILE "training.muon_fp32_matmul_prec")
MUON_TP_MODE=$(yaml_get $TRAINING_CONFIG_FILE "training.muon_tp_mode")
MUON_EXTRA_SCALE_FACTOR=$(yaml_get $TRAINING_CONFIG_FILE "training.muon_extra_scale_factor")

# 체크포인트 및 로깅
SAVE_INTERVAL=$(yaml_get $TRAINING_CONFIG_FILE "training.save_interval")
SAVE_OPTIM=$(yaml_get $TRAINING_CONFIG_FILE "training.save_optim")
EVAL_ITERS=$(yaml_get $TRAINING_CONFIG_FILE "training.eval_iters")
EVAL_INTERVAL=$(yaml_get $TRAINING_CONFIG_FILE "training.eval_interval")
LOG_INTERVAL=$(yaml_get $TRAINING_CONFIG_FILE "training.log_interval")
DIST_TIMEOUT=$(yaml_get $TRAINING_CONFIG_FILE "training.distributed_timeout_minutes")
MANUAL_GC_INTERVAL=$(yaml_get $TRAINING_CONFIG_FILE "training.manual_gc_interval")

#==============================================================================
# Resume 설정 로드 및 검증
#==============================================================================

RESUME_ENABLED=$(yaml_get $TRAINING_CONFIG_FILE "training.resume.enabled")
LOAD_CHECKPOINT_PATH=$(yaml_get $TRAINING_CONFIG_FILE "training.resume.load_checkpoint_path")
FINETUNE=$(yaml_get $TRAINING_CONFIG_FILE "training.resume.finetune")
NO_LOAD_OPTIM=$(yaml_get $TRAINING_CONFIG_FILE "training.resume.no_load_optim")

if [ "$RESUME_ENABLED" != "true" ] && [ "$RESUME_ENABLED" != "True" ]; then
    echo "❌ resume.enabled가 true가 아닙니다."
    echo "   Stage 2 스크립트는 resume 설정이 필수입니다."
    exit 1
fi

if [ -z "$LOAD_CHECKPOINT_PATH" ] || [ "$LOAD_CHECKPOINT_PATH" = "" ] || [[ "$LOAD_CHECKPOINT_PATH" == *"<"* ]]; then
    echo "❌ resume.load_checkpoint_path가 설정되지 않았거나 placeholder입니다."
    echo "   현재 값: ${LOAD_CHECKPOINT_PATH}"
    echo "   Cooldown 체크포인트 절대 경로를 설정하세요."
    exit 1
fi

if [ ! -d "$LOAD_CHECKPOINT_PATH" ]; then
    echo "❌ 체크포인트 경로가 존재하지 않습니다: ${LOAD_CHECKPOINT_PATH}"
    exit 1
fi

echo ""
echo "=============================================="
echo "Resume 설정"
echo "=============================================="
echo "체크포인트 경로: ${LOAD_CHECKPOINT_PATH}"
echo "Finetune (fresh start): ${FINETUNE}"
echo "No Load Optim: ${NO_LOAD_OPTIM}"
echo "Optimizer 저장: ${SAVE_OPTIM}"
echo "=============================================="

echo "✅ 학습 설정 완료"
echo "  - Iterations: ${TRAIN_ITERS} (warmup=${LR_WARMUP_ITERS}, decay=${LR_DECAY_ITERS})"
echo "  - Learning Rate: ${LR} → ${MIN_LR} (${LR_DECAY_STYLE})"
if [[ "${LR_DECAY_STYLE}" == "WSD" ]]; then
    echo "  - WSD: decay_style=${LR_WSD_DECAY_STYLE}, decay_iters=${LR_WSD_DECAY_ITERS}"
fi
echo ""

#==============================================================================
# 타임스탬프 설정 (WANDB 및 출력 디렉토리에서 사용)
#==============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

#==============================================================================
# WANDB 설정 로드 (Optional)
#==============================================================================

WANDB_ENABLED=$(yaml_get $TRAINING_CONFIG_FILE "training.wandb.enabled")

if [ "$WANDB_ENABLED" = "True" ] || [ "$WANDB_ENABLED" = "true" ]; then
    echo "WANDB 설정 로드 중..."

    WANDB_PROJECT=$(yaml_get $TRAINING_CONFIG_FILE "training.wandb.project")
    WANDB_ENTITY=$(yaml_get $TRAINING_CONFIG_FILE "training.wandb.entity")
    WANDB_SAVE_DIR=$(yaml_get $TRAINING_CONFIG_FILE "training.wandb.save_dir")

    # Experiment name: stage2_3 표시
    WANDB_EXP_NAME="${MODEL_CONFIG}_${DATA_CONFIG}_stage2_3_${TIMESTAMP}"

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
# 데이터 설정 로드
#==============================================================================

DATA_CONFIG_FILE="${CONFIG_DIR}/data/${DATA_CONFIG}.yaml"

if [ ! -f "$DATA_CONFIG_FILE" ]; then
    echo "❌ 데이터 설정 파일을 찾을 수 없습니다: $DATA_CONFIG_FILE"
    exit 1
fi

echo "데이터 설정 로드 중: ${DATA_CONFIG}.yaml"

DATASET_IMPL=$(yaml_get $DATA_CONFIG_FILE "data.dataset_impl")
NUM_WORKERS=$(yaml_get $DATA_CONFIG_FILE "data.num_workers")
DATA_SPLIT=$(yaml_get $DATA_CONFIG_FILE "data.split")

# Split 배열 파싱 [99, 1, 0] → "99,1,0"
DATA_SPLIT=$(echo $DATA_SPLIT | tr -d '[]' | tr ' ' ',')

# Blended dataset 또는 단일 dataset 파싱
BLEND_CONFIG=$(yaml_get $DATA_CONFIG_FILE "data.blend")
if [ ! -z "$BLEND_CONFIG" ] && [ "$BLEND_CONFIG" != "None" ] && [ "$BLEND_CONFIG" != "" ]; then
    echo "  Blended dataset 설정 감지..."
    DATA_PATH=$(python3 -c "
import yaml
import os
with open('${DATA_CONFIG_FILE}') as f:
    data = yaml.safe_load(f)
blend = data.get('data', {}).get('blend', [])
parts = []
for item in blend:
    path = os.path.expandvars(item['path'])
    # 가중치가 있으면 추가
    if 'weight' in item:
        parts.append(str(item['weight']))
    parts.append(path)
print(' '.join(parts))
")
    echo "  - Blend 경로: ${DATA_PATH}"
else
    DATA_PATH=$(yaml_get $DATA_CONFIG_FILE "data.train_path")
    # 단일 데이터셋 존재 확인
    if [ ! -f "${DATA_PATH}.bin" ] || [ ! -f "${DATA_PATH}.idx" ]; then
        echo "❌ 오류: 데이터셋 파일을 찾을 수 없습니다!"
        echo "   필요: ${DATA_PATH}.bin"
        echo "   필요: ${DATA_PATH}.idx"
        exit 1
    fi
    echo "  - 경로: ${DATA_PATH}"
fi

echo "✅ 데이터 설정 완료"
echo "  - Split: ${DATA_SPLIT}"

# 데이터셋 캐시 경로 (데이터셋별 공유)
DATA_CACHE_PATH="${CONFIG_DIR}/data/.cache/${DATA_CONFIG}"
mkdir -p ${DATA_CACHE_PATH}
echo "  - Cache: ${DATA_CACHE_PATH}"
echo ""

#==============================================================================
# 모델 설정 로드 (baseline_48L.yaml)
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
# NoPE for Full Attention (Kimi-Linear style)
NO_ROPE_FREQ=$(yaml_get $MODEL_CONFIG_FILE "model.no_rope_freq")

# Tokenizer (Megatron-LM standard)
PADDED_VOCAB_SIZE=$(yaml_get $MODEL_CONFIG_FILE "model.padded_vocab_size")
TOKENIZER_TYPE=$(yaml_get $MODEL_CONFIG_FILE "model.tokenizer_type")
TOKENIZER_MODEL=$(yaml_get $MODEL_CONFIG_FILE "model.tokenizer_model")
# Fixed tokenizer path (Alpha uses Qwen3-Next tokenizer)
TOKENIZER_PATH="${CURRENT_DIR}/${TOKENIZER_MODEL}"

echo "✅ 모델 설정 완료"
echo "  - 아키텍처: ${NUM_LAYERS} layers, ${HIDDEN_SIZE} hidden, ${NUM_EXPERTS} experts"
echo "  - Hybrid: ${HYBRID_ATTN_RATIO} attention ratio, pattern: ${HYBRID_PATTERN}"
echo ""

#==============================================================================
# 출력 디렉토리 설정 (stage2_3 표시)
#==============================================================================

OUTPUT_DIR=${CURRENT_DIR}/outputs/alpha_${MODEL_CONFIG}_stage2_3_${TIMESTAMP}
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
    --moe-shared-expert-gate

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

    # Positional Encoding (RoPE with NoPE for Full Attention)
    --use-rotary-position-embeddings
    --rotary-base ${ROTARY_BASE}
    --rotary-percent ${ROTARY_PERCENT}
    --position-embedding-type rope
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${MAX_POS_EMB}

    # Tokenizer & Vocab (Megatron-LM standard)
    --untie-embeddings-and-output-weights
    --padded-vocab-size ${PADDED_VOCAB_SIZE}
    --tokenizer-type ${TOKENIZER_TYPE}
    --tokenizer-model ${TOKENIZER_PATH}
)

TRAINING_ARGS=(
    # 데이터
    --data-path ${DATA_PATH}
    --data-cache-path ${DATA_CACHE_PATH}
    --split ${DATA_SPLIT}
    --dataset ${DATASET_IMPL}
    --num-workers ${NUM_WORKERS}

    # 배치 및 Iteration
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --train-iters ${TRAIN_ITERS}

    # Resume: 체크포인트 로드
    --load ${LOAD_CHECKPOINT_PATH}

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

# 변경 D: save_optim 조건부 (YAML 기반)
if [ "${SAVE_OPTIM}" != "true" ] && [ "${SAVE_OPTIM}" != "True" ]; then
    TRAINING_ARGS+=(--no-save-optim)
    echo "📌 --no-save-optim 활성화 (optimizer 저장하지 않음)"
else
    echo "📌 Optimizer 저장 활성화 (확장 시 scheduler resume 가능)"
fi

# --finetune 추가 (조건부: iteration=0, consumed_samples=0, optimizer/RNG 새로 생성)
if [ "$FINETUNE" = "true" ] || [ "$FINETUNE" = "True" ]; then
    TRAINING_ARGS+=(--finetune)
    echo "📌 --finetune 활성화 (iteration=0, 새 dataset 처음부터 학습)"
fi

# --no-load-optim 추가 (조건부: 같은 dataset 이어서 학습, optimizer/scheduler만 리셋)
if [ "$NO_LOAD_OPTIM" = "true" ] || [ "$NO_LOAD_OPTIM" = "True" ]; then
    TRAINING_ARGS+=(--no-load-optim)
    echo "📌 --no-load-optim 활성화 (optimizer/scheduler 리셋, data position 유지)"
fi

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

# 변경 C: WSD Scheduler Arguments (조건부 추가)
if [[ "${LR_DECAY_STYLE}" == "WSD" ]]; then
    echo "📈 WSD scheduler 인자 추가 중..."

    TRAINING_ARGS+=(
        --lr-wsd-decay-style ${LR_WSD_DECAY_STYLE}
        --lr-wsd-decay-iters ${LR_WSD_DECAY_ITERS}
    )

    echo "  - WSD Decay Style: ${LR_WSD_DECAY_STYLE}"
    echo "  - WSD Decay Iters: ${LR_WSD_DECAY_ITERS}"
    echo "  ✅ WSD 인자 추가 완료"
fi

# QK-Clip Arguments (조건부 추가, TE >= 2.9.0 필요)
if [[ "${QK_CLIP}" == "true" || "${QK_CLIP}" == "True" ]]; then
    echo "🔒 QK-Clip 인자 추가 중..."

    TRAINING_ARGS+=(
        --qk-clip
        --qk-clip-alpha ${QK_CLIP_ALPHA}
        --qk-clip-threshold ${QK_CLIP_THRESHOLD}
        --log-max-attention-logit
    )

    echo "  - QK-Clip Alpha: ${QK_CLIP_ALPHA}"
    echo "  - QK-Clip Threshold: ${QK_CLIP_THRESHOLD}"
    echo "  ✅ QK-Clip 인자 추가 완료"
fi

# No Weight Decay Cond Type (QK LayerNorm에 weight decay 적용)
if [ ! -z "$NO_WEIGHT_DECAY_COND_TYPE" ] && [ "$NO_WEIGHT_DECAY_COND_TYPE" != "" ]; then
    echo "⚖️  No Weight Decay Cond Type 인자 추가 중..."
    TRAINING_ARGS+=(
        --no-weight-decay-cond-type ${NO_WEIGHT_DECAY_COND_TYPE}
    )
    echo "  - Type: ${NO_WEIGHT_DECAY_COND_TYPE}"
    echo "  ✅ No Weight Decay Cond Type 인자 추가 완료"
fi

# Muon Optimizer Arguments (조건부 추가)
if [[ "${OPTIMIZER}" == "muon" || "${OPTIMIZER}" == "dist_muon" ]]; then
    echo "🚀 Muon optimizer 인자 추가 중..."

    TRAINING_ARGS+=(
        --muon-momentum ${MUON_MOMENTUM}
        --muon-num-ns-steps ${MUON_NUM_NS_STEPS}
        --muon-scale-mode ${MUON_SCALE_MODE}
        --muon-fp32-matmul-prec ${MUON_FP32_MATMUL_PREC}
        --muon-tp-mode ${MUON_TP_MODE}
        --muon-extra-scale-factor ${MUON_EXTRA_SCALE_FACTOR}
    )

    # Nesterov flag 처리 (true일 때 --muon-use-nesterov 명시적 전달)
    # 버그 수정: argparse store_true는 default=False이므로 true일 때 플래그를 전달해야 함
    if [[ "${MUON_USE_NESTEROV}" == "true" || "${MUON_USE_NESTEROV}" == "True" ]]; then
        TRAINING_ARGS+=(--muon-use-nesterov)
    fi

    echo "  ✅ Muon 인자 추가 완료"
fi

# NoPE Arguments (조건부 추가 - Kimi-Linear style)
if [ ! -z "$NO_ROPE_FREQ" ] && [ "$NO_ROPE_FREQ" != "None" ] && [ "$NO_ROPE_FREQ" != "" ]; then
    echo "🎯 NoPE 인자 추가 중... (Full Attention에서 RoPE 비활성화)"

    MODEL_ARGS+=(
        --no-rope-freq "${NO_ROPE_FREQ}"
    )

    echo "  - Pattern: ${NO_ROPE_FREQ}"
    echo "  ✅ NoPE 인자 추가 완료"
fi

# INFRA_ARGS 기본 정의 (병렬화 및 최적화 설정)
INFRA_ARGS=(
    # 병렬화
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --expert-model-parallel-size ${EP}
    --expert-tensor-parallel-size ${ETP}
    --context-parallel-size ${CP}

    # 최적화
    # --use-distributed-optimizer  # Muon uses LayerWise distributed optimizer
    --overlap-grad-reduce
    # --overlap-param-gather  # Requires distributed optimizer (disabled for Muon)

    # Attention Backend: auto → TE가 최적 백엔드 자동 선택
    # QK-Clip(return_max_logit=True) 시 flash 자동 비활성화, fused 우선, unfused fallback
    --attention-backend auto

    # Loss Fusion
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
)

# MoE 통신 최적화 (YAML에서 로드, TransformerEngine #2438 권장)
MOE_PERMUTE_FUSION=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.optimizations.moe_permute_fusion")
if [ "$MOE_PERMUTE_FUSION" = "True" ] || [ "$MOE_PERMUTE_FUSION" = "true" ]; then
    # 이미 MODEL_ARGS에 --moe-permute-fusion 있음, 중복 체크
    if [[ ! " ${MODEL_ARGS[*]} " =~ " --moe-permute-fusion " ]]; then
        INFRA_ARGS+=(--moe-permute-fusion)
    fi
fi

OVERLAP_MOE_EP_COMM=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.optimizations.overlap_moe_expert_parallel_comm")
if [ "$OVERLAP_MOE_EP_COMM" = "True" ] || [ "$OVERLAP_MOE_EP_COMM" = "true" ]; then
    echo "  - MoE EP 통신 오버랩: 활성화"
    INFRA_ARGS+=(--overlap-moe-expert-parallel-comm)
fi

DELAY_WGRAD=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.optimizations.delay_wgrad_compute")
if [ "$DELAY_WGRAD" = "True" ] || [ "$DELAY_WGRAD" = "true" ]; then
    echo "  - Delay wgrad compute: 활성화"
    INFRA_ARGS+=(--delay-wgrad-compute)
fi

# MoE Token Dispatcher Type (alltoall or flex)
MOE_DISPATCHER_TYPE=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.optimizations.moe_token_dispatcher_type")
if [ -z "$MOE_DISPATCHER_TYPE" ]; then
    MOE_DISPATCHER_TYPE="alltoall"  # 기본값
fi
INFRA_ARGS+=(--moe-token-dispatcher-type ${MOE_DISPATCHER_TYPE})
echo "  - MoE Token Dispatcher: ${MOE_DISPATCHER_TYPE}"

# DeepEP Settings (when dispatcher_type is "flex")
if [ "$MOE_DISPATCHER_TYPE" = "flex" ]; then
    echo "  🚀 DeepEP Flex Dispatcher 설정 중..."

    # Flex Dispatcher Backend (deepep, deepep_low_latency, 등)
    FLEX_BACKEND=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.optimizations.moe_flex_dispatcher_backend")
    if [ ! -z "$FLEX_BACKEND" ] && [ "$FLEX_BACKEND" != "" ]; then
        INFRA_ARGS+=(--moe-flex-dispatcher-backend ${FLEX_BACKEND})
        echo "    - Backend: ${FLEX_BACKEND}"
    fi

    # DeepEP Num SMs (튜닝 파라미터, H100: 132 SMs)
    DEEPEP_NUM_SMS=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.optimizations.moe_deepep_num_sms")
    if [ ! -z "$DEEPEP_NUM_SMS" ] && [ "$DEEPEP_NUM_SMS" != "" ]; then
        INFRA_ARGS+=(--moe-deepep-num-sms ${DEEPEP_NUM_SMS})
        echo "    - Num SMs: ${DEEPEP_NUM_SMS}"
    fi

    echo "  ✅ DeepEP 설정 완료"
fi

# Fine-Grained Activation Offloading (YAML에서 로드, Megatron-LM-251125+)
FINE_GRAINED_OFFLOAD=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.optimizations.fine_grained_activation_offloading")
if [ "$FINE_GRAINED_OFFLOAD" = "True" ] || [ "$FINE_GRAINED_OFFLOAD" = "true" ]; then
    echo "  - Fine-Grained Activation Offloading: 활성화"
    INFRA_ARGS+=(--fine-grained-activation-offloading)

    # Offload modules
    OFFLOAD_MODULES=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.optimizations.offload_modules")
    if [ ! -z "$OFFLOAD_MODULES" ]; then
        # 공백과 따옴표 정리
        OFFLOAD_MODULES=$(echo $OFFLOAD_MODULES | tr -d '"' | tr ',' ' ')
        INFRA_ARGS+=(--offload-modules ${OFFLOAD_MODULES})
        echo "    - Modules: ${OFFLOAD_MODULES}"
    fi
fi

# Activation Checkpointing (YAML에서 로드, INFRA_ARGS에 추가)
AC_GRANULARITY=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.activation_checkpointing.granularity")
if [ ! -z "$AC_GRANULARITY" ]; then
    INFRA_ARGS+=(--recompute-granularity ${AC_GRANULARITY})

    # Recompute modules
    RECOMPUTE_MODULES=$(yaml_get $INFRA_CONFIG_FILE "infrastructure.activation_checkpointing.recompute_modules")
    if [ ! -z "$RECOMPUTE_MODULES" ]; then
        # YAML 배열을 공백으로 구분된 문자열로 변환
        MODULES=$(echo $RECOMPUTE_MODULES | tr -d '[],' | tr "'" ' ')
        INFRA_ARGS+=(--recompute-modules ${MODULES})
    else
        # YAML에 없으면 기본값 사용
        INFRA_ARGS+=(--recompute-modules layernorm moe_act shared_experts)
    fi
else
    # Activation checkpointing 비활성화 (기본값)
    echo "  ℹ️  Activation checkpointing 설정이 없습니다. 기본값 사용."
fi

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
echo "🚀 Alpha 프로젝트 Stage 2-3 학습 시작!"
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
Alpha 프로젝트 Stage 2 학습 설정 스냅샷
생성 시각: $(date)
============================================

[Resume]
- Checkpoint: ${LOAD_CHECKPOINT_PATH}
- Finetune (fresh start): ${FINETUNE}
- Save Optim: ${SAVE_OPTIM}

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

[학습 설정 (Stage 2)]
- Global Batch Size: ${GBS}
- Micro Batch Size: ${MBS}
- Sequence Length: ${SEQ_LEN}
- Total Iterations: ${TRAIN_ITERS}
- Warmup Iterations: ${LR_WARMUP_ITERS}
- Decay Iterations: ${LR_DECAY_ITERS}
- Learning Rate: ${LR} → ${MIN_LR}
- LR Decay Style: ${LR_DECAY_STYLE}
- WSD Decay Style: ${LR_WSD_DECAY_STYLE}
- WSD Decay Iters: ${LR_WSD_DECAY_ITERS}
- Optimizer: ${OPTIMIZER}
- Save Optim: ${SAVE_OPTIM}
- QK-Clip: ${QK_CLIP} (alpha=${QK_CLIP_ALPHA}, threshold=${QK_CLIP_THRESHOLD})
- MoE Dispatcher: ${MOE_DISPATCHER_TYPE}

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
    echo "✅ Stage 2-3 학습 완료!"
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
    echo "❌ Stage 2-3 학습 실패 (Exit Code: ${EXIT_CODE})"
    echo "=============================================="
    echo "로그 확인: ${LOG_DIR}/train.log"
fi

exit $EXIT_CODE
