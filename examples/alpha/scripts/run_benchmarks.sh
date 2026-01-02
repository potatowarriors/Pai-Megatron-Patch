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

# Default benchmark suite with standard n-shot settings
# Following Qwen paper style:
#   5-shot: mmlu, hellaswag, arc_easy, arc_challenge, winogrande, boolq, piqa, kmmlu
#   8-shot: gsm8k (with chain-of-thought)
#
DEFAULT_TASKS="standard"
TASKS=${2:-$DEFAULT_TASKS}

# Replace 'siqa' with 'social_iqa' as expected by lm-eval
TASKS=${TASKS//siqa/social_iqa}
BATCH_SIZE=${3:-auto}
DEVICE=${4:-cuda:0}

# WandB settings (학습과 별도 프로젝트로 분리)
WANDB_PROJECT=${WANDB_PROJECT:-"alpha-evals"}
WANDB_ENABLED=${WANDB_ENABLED:-true}

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <MODEL_PATH> [TASKS|standard] [BATCH_SIZE] [DEVICE]"
    echo ""
    echo "Examples:"
    echo "  # Run standard benchmark suite (recommended)"
    echo "  $0 outputs/alpha_baseline_48L_*/hfmodel_0050000 standard"
    echo ""
    echo "  # Run specific tasks (all 0-shot)"
    echo "  $0 outputs/alpha_baseline_48L_*/hfmodel_0050000 mmlu,hellaswag"
    echo ""
    echo "Standard benchmark suite (Qwen style n-shot):"
    echo "  5-shot: mmlu, hellaswag, arc_easy, arc_challenge, winogrande, boolq, piqa, kmmlu"
    echo "  8-shot: gsm8k"
    echo ""
    echo "Environment variables:"
    echo "  WANDB_PROJECT  - WandB project name (default: alpha-evals)"
    echo "  WANDB_ENABLED  - Enable WandB logging (default: true)"
    exit 1
fi

# Extract run name from model path
# e.g., outputs/alpha_baseline_48L_20251129_231218/hfmodel_0050000 -> alpha_baseline_48L_20251129_231218
RUN_NAME=$(basename $(dirname $MODEL_PATH))
if [ "$RUN_NAME" == "." ] || [ -z "$RUN_NAME" ] || [ "$RUN_NAME" == "outputs" ]; then
    RUN_NAME=$(basename $MODEL_PATH)
fi

# Extract iteration from hfmodel_XXXXXX pattern (e.g., hfmodel_0050000 -> 0050000)
MODEL_BASENAME=$(basename $MODEL_PATH)
if [[ "$MODEL_BASENAME" =~ ^hfmodel_([0-9]+)$ ]]; then
    ITERATION="${BASH_REMATCH[1]}"
    RUN_NAME="${RUN_NAME}_iter${ITERATION}"
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
    # Multi-GPU에서 중복 run 방지
    export WANDB_START_METHOD=thread
    WANDB_ARGS="--wandb_args project=$WANDB_PROJECT,name=${RUN_NAME},job_type=eval"
    echo "📊 WandB logging enabled (project: $WANDB_PROJECT)"
fi

# Function to run evaluation with specific n-shot
run_eval() {
    local tasks=$1
    local num_fewshot=$2
    local desc=$3

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔄 Running: $desc"
    echo "   Tasks: $tasks"
    echo "   N-shot: $num_fewshot"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    accelerate launch --multi_gpu --num_processes=8 --main_process_port 29501 -m lm_eval \
        --model hf \
        --model_args pretrained=$MODEL_PATH,trust_remote_code=True,dtype=bfloat16 \
        --tasks $tasks \
        --num_fewshot $num_fewshot \
        --batch_size $BATCH_SIZE \
        --output_path $OUTPUT_DIR \
        $WANDB_ARGS
}

# Run benchmarks
if [ "$TASKS" == "standard" ]; then
    echo ""
    echo "================================================================"
    echo "🚀 Running STANDARD benchmark suite (Qwen style n-shot)"
    echo "================================================================"

    # 5-shot: all benchmarks except gsm8k
    run_eval "mmlu,hellaswag,arc_easy,arc_challenge,winogrande,boolq,piqa,kmmlu" 5 \
        "5-shot benchmarks (MMLU, HellaSwag, ARC, Winogrande, BoolQ, PIQA, KMMLU)"

    # 8-shot: gsm8k (chain-of-thought)
    run_eval "gsm8k" 8 "8-shot benchmark (GSM8K)"

    echo ""
    echo "================================================================"
    echo "✅ Standard benchmark suite completed!"
    echo "================================================================"
else
    # Custom tasks: run with 0-shot (legacy behavior)
    echo ""
    echo "================================================================"
    echo "🔄 Running custom tasks with 0-shot (specify n-shot manually if needed)"
    echo "================================================================"

    accelerate launch --multi_gpu --num_processes=8 --main_process_port 29501 -m lm_eval \
        --model hf \
        --model_args pretrained=$MODEL_PATH,trust_remote_code=True,dtype=bfloat16 \
        --tasks $TASKS \
        --batch_size $BATCH_SIZE \
        --output_path $OUTPUT_DIR \
        $WANDB_ARGS
fi
