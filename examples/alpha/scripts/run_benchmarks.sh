#!/bin/bash
#==============================================================================
# Alpha 모델 벤치마크 평가 스크립트
# LM-Evaluation-Harness를 사용한 평가 + WandB 로깅
#==============================================================================

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

#==============================================================================
# WANDB 환경 자동 로드
#==============================================================================
WANDB_SETUP_SCRIPT="${SCRIPT_DIR}/setup_wandb.sh"
if [ -f "$WANDB_SETUP_SCRIPT" ]; then
    source "$WANDB_SETUP_SCRIPT" 2>/dev/null || true
fi

#==============================================================================
# 인자 처리
#==============================================================================
MODEL_PATH=$1

# Default benchmark suite for consistent evaluation
DEFAULT_TASKS="mmlu,hellaswag,arc_easy,arc_challenge,winogrande,boolq,piqa,gsm8k,kmmlu"
TASKS=${2:-$DEFAULT_TASKS}

# Replace 'siqa' with 'social_iqa' as expected by lm-eval
TASKS=${TASKS//siqa/social_iqa}
BATCH_SIZE=${3:-auto}
DEVICE=${4:-cuda:0}

# WandB settings (학습과 별도 프로젝트로 분리)
WANDB_PROJECT=${WANDB_PROJECT:-"alpha-evals"}
WANDB_ENABLED=${WANDB_ENABLED:-true}

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <MODEL_PATH> [TASKS] [BATCH_SIZE] [DEVICE]"
    echo "Example: $0 outputs/alpha_baseline_24L_*/hf_converted hellaswag auto cuda:0"
    echo ""
    echo "Environment variables:"
    echo "  WANDB_PROJECT  - WandB project name (default: alpha-evals)"
    echo "  WANDB_ENABLED  - Enable WandB logging (default: true)"
    exit 1
fi

# Extract run name from model path
# e.g., outputs/alpha_baseline_24L_20251129_231218/hf_converted -> alpha_baseline_24L_20251129_231218
RUN_NAME=$(basename $(dirname $MODEL_PATH))
if [ "$RUN_NAME" == "." ] || [ -z "$RUN_NAME" ] || [ "$RUN_NAME" == "outputs" ]; then
    RUN_NAME=$(basename $MODEL_PATH)
fi

echo "================================================================"
echo "Alpha 모델 벤치마크 평가"
echo "================================================================"
echo "Model: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Batch Size: $BATCH_SIZE"
echo "----------------------------------------------------------------"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run Name: ${RUN_NAME}_eval"
echo "WandB Group: $RUN_NAME"
echo "WandB Enabled: $WANDB_ENABLED"
echo "================================================================"

#==============================================================================
# 환경 설정
#==============================================================================

# Set HF Cache paths
export HF_HOME=/home/work/Datasets/benchmarks
export HF_DATASETS_CACHE=/home/work/Datasets/benchmarks
export HF_TOKEN="${HF_TOKEN:-}"
mkdir -p $HF_DATASETS_CACHE

# Check WandB API key
if [ "$WANDB_ENABLED" == "true" ] && [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  WARNING: WANDB_API_KEY not set. WandB logging disabled."
    echo "   Run: source scripts/setup_wandb.sh"
    WANDB_ENABLED=false
fi

# Create output directory for results
OUTPUT_DIR="${MODEL_PATH}/eval_results"
mkdir -p $OUTPUT_DIR

echo "Results will be saved to: $OUTPUT_DIR"
echo "================================================================"

#==============================================================================
# 벤치마크 실행
#==============================================================================

# Build wandb args if enabled
# - project: 평가 전용 프로젝트 (alpha-evals)
# - name: 모델 run 이름
# - job_type: eval
WANDB_ARGS=""
if [ "$WANDB_ENABLED" == "true" ]; then
    WANDB_ARGS="--wandb_args project=$WANDB_PROJECT,name=${RUN_NAME},job_type=eval"
    echo "📊 WandB logging enabled (project: $WANDB_PROJECT)"
fi

# Use accelerate for multi-GPU evaluation
# Note: --log_samples 제거됨 (최종 결과만 저장, 개별 샘플 로그 불필요)
accelerate launch --multi_gpu --num_processes=8 -m lm_eval \
    --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True,dtype=bfloat16 \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --output_path $OUTPUT_DIR \
    $WANDB_ARGS
