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
# Supports multiple model paths:
#   $0 <MODEL_PATH1> [MODEL_PATH2 ...] [--tasks TASKS] [--batch-size BS]
# Legacy single-model syntax also supported:
#   $0 <MODEL_PATH> [TASKS] [BATCH_SIZE] [DEVICE]

# Default settings
DEFAULT_TASKS="standard"
BATCH_SIZE="auto"
DEVICE="cuda:0"

# GPU 수 자동 감지 또는 환경변수로 지정
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

# WandB settings (학습과 별도 프로젝트로 분리)
WANDB_PROJECT=${WANDB_PROJECT:-"alpha-evals"}
WANDB_ENABLED=${WANDB_ENABLED:-true}

# Parse arguments: collect model paths and options
MODEL_PATHS=()
TASKS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)
            TASKS="$2"; shift 2 ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        --help|-h)
            MODEL_PATHS=(); break ;;
        --*)
            echo "Unknown option: $1"; exit 1 ;;
        *)
            # Heuristic: if it looks like a path (contains / or starts with . or is a directory),
            # treat it as a model path. Otherwise treat as legacy positional arg.
            if [ -d "$1" ] || [[ "$1" == */* ]] || [[ "$1" == .* ]]; then
                MODEL_PATHS+=("$1")
            elif [ ${#MODEL_PATHS[@]} -eq 0 ]; then
                # No model path yet — this can't be a task name
                MODEL_PATHS+=("$1")
            elif [ -z "$TASKS" ]; then
                # Legacy: second positional arg = TASKS
                TASKS="$1"
            elif [ "$BATCH_SIZE" == "auto" ]; then
                # Legacy: third positional arg = BATCH_SIZE
                BATCH_SIZE="$1"
            else
                # Legacy: fourth positional arg = DEVICE
                DEVICE="$1"
            fi
            shift ;;
    esac
done

TASKS=${TASKS:-$DEFAULT_TASKS}

# Replace 'siqa' with 'social_iqa' as expected by lm-eval
TASKS=${TASKS//siqa/social_iqa}

if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    echo "Usage: $0 <MODEL_PATH> [MODEL_PATH2 ...] [--tasks TASKS] [--batch-size BS]"
    echo ""
    echo "Examples:"
    echo "  # Single model (legacy syntax)"
    echo "  $0 outputs/alpha_baseline_48L_*/hfmodel_0050000 standard"
    echo ""
    echo "  # Multiple models"
    echo "  $0 path/to/hfmodel_0050000 path/to/hfmodel_0100000 --tasks standard"
    echo ""
    echo "  # Glob pattern (shell expands to multiple paths)"
    echo "  $0 outputs/*/hfmodel_0050000 --tasks standard"
    echo ""
    echo "  # Run specific tasks"
    echo "  $0 path/to/hfmodel --tasks mmlu,hellaswag"
    echo ""
    echo "Standard benchmark suite (Qwen style n-shot):"
    echo "  5-shot: mmlu, hellaswag, arc_easy, arc_challenge, winogrande, boolq, piqa, kmmlu"
    echo "  4-shot: gsm8k"
    echo "  3-shot: mbpp (pass@1)"
    echo "  0-shot: humaneval (pass@1)"
    echo ""
    echo "Environment variables:"
    echo "  WANDB_PROJECT  - WandB project name (default: alpha-evals)"
    echo "  WANDB_ENABLED  - Enable WandB logging (default: true)"
    echo "  NUM_GPUS       - Number of GPUs (default: auto-detect)"
    exit 1
fi

echo "================================================================"
echo "Alpha 모델 벤치마크 평가"
echo "================================================================"
echo "Models: ${#MODEL_PATHS[@]}개"
for mp in "${MODEL_PATHS[@]}"; do
    echo "  - $mp"
done
echo "Tasks: $TASKS"
echo "Batch Size: $BATCH_SIZE"
echo "Num GPUs: $NUM_GPUS"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Enabled: $WANDB_ENABLED"
echo "================================================================"

#==============================================================================
# 환경 설정
#==============================================================================

# Set HF Cache paths
export HF_HOME=/home/work/Datasets/benchmarks
export HF_DATASETS_CACHE=/home/work/Datasets/benchmarks
export HF_TOKEN="${HF_TOKEN:-}"

# Allow code execution for HumanEval/MBPP (code generation benchmarks)
export HF_ALLOW_CODE_EVAL=1
mkdir -p $HF_DATASETS_CACHE

# Check WandB API key
if [ "$WANDB_ENABLED" == "true" ] && [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  WARNING: WANDB_API_KEY not set. WandB logging disabled."
    echo "   Run: source scripts/setup_wandb.sh"
    WANDB_ENABLED=false
fi

#==============================================================================
# 모델별 벤치마크 실행 함수
#==============================================================================

# Function to run evaluation with specific n-shot for a given model
# Usage: run_eval <model_path> <output_dir> <wandb_args> <tasks> <num_fewshot> <description>
run_eval() {
    local model_path=$1
    local output_dir=$2
    local wandb_args=$3
    local tasks=$4
    local num_fewshot=$5
    local desc=$6

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔄 Running: $desc"
    echo "   Tasks: $tasks"
    echo "   N-shot: $num_fewshot"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # --multi_gpu는 2+ 프로세스 전용 — 단일 GPU(NUM_GPUS=1)에서는 빼야 함
    local MULTI_GPU_FLAG=""
    if [ "$NUM_GPUS" -gt 1 ]; then
        MULTI_GPU_FLAG="--multi_gpu"
    fi

    accelerate launch $MULTI_GPU_FLAG --num_processes=$NUM_GPUS --main_process_port 29501 \
        ${SCRIPT_DIR}/eval_wrapper.py \
        --model hf \
        --model_args pretrained=$model_path,trust_remote_code=True,dtype=bfloat16 \
        --tasks $tasks \
        --num_fewshot $num_fewshot \
        --batch_size $BATCH_SIZE \
        --output_path $output_dir \
        --confirm_run_unsafe_code \
        $wandb_args
}

# Function to evaluate a single model
eval_model() {
    local MODEL_PATH=$1
    local MODEL_IDX=$2
    local MODEL_TOTAL=$3

    # Extract run name from model path
    # e.g., outputs/alpha_baseline_48L_20251129_231218/hfmodel_0050000 -> alpha_baseline_48L_20251129_231218
    local RUN_NAME=$(basename $(dirname $MODEL_PATH))
    if [ "$RUN_NAME" == "." ] || [ -z "$RUN_NAME" ] || [ "$RUN_NAME" == "outputs" ]; then
        RUN_NAME=$(basename $MODEL_PATH)
    fi

    # Extract iteration from hfmodel_XXXXXX pattern (e.g., hfmodel_0050000 -> 0050000)
    local MODEL_BASENAME=$(basename $MODEL_PATH)
    if [[ "$MODEL_BASENAME" =~ ^hfmodel_([0-9]+)$ ]]; then
        local ITERATION="${BASH_REMATCH[1]}"
        RUN_NAME="${RUN_NAME}_iter${ITERATION}"
    fi

    # Create output directory for results
    local OUTPUT_DIR="${MODEL_PATH}/eval_results"
    mkdir -p $OUTPUT_DIR

    # Build wandb args if enabled
    local WANDB_ARGS=""
    if [ "$WANDB_ENABLED" == "true" ]; then
        export WANDB_START_METHOD=thread
        WANDB_ARGS="--wandb_args project=$WANDB_PROJECT,name=${RUN_NAME},job_type=eval"
    fi

    echo ""
    echo "================================================================"
    echo "📋 Model [$MODEL_IDX/$MODEL_TOTAL]: $MODEL_PATH"
    echo "----------------------------------------------------------------"
    echo "Run Name: ${RUN_NAME}"
    echo "Results: $OUTPUT_DIR"
    if [ -n "$WANDB_ARGS" ]; then
        echo "WandB Run: ${RUN_NAME}"
    fi
    echo "================================================================"

    # Run benchmarks
    if [ "$TASKS" == "standard" ]; then
        run_eval "$MODEL_PATH" "$OUTPUT_DIR" "$WANDB_ARGS" \
            "mmlu,hellaswag,arc_easy,arc_challenge,winogrande,boolq,piqa,kmmlu" 5 \
            "5-shot benchmarks (MMLU, HellaSwag, ARC, Winogrande, BoolQ, PIQA, KMMLU)"

        run_eval "$MODEL_PATH" "$OUTPUT_DIR" "$WANDB_ARGS" \
            "gsm8k" 4 "4-shot benchmark (GSM8K)"

        run_eval "$MODEL_PATH" "$OUTPUT_DIR" "$WANDB_ARGS" \
            "humaneval" 0 "0-shot code benchmark (HumanEval pass@1)"

        run_eval "$MODEL_PATH" "$OUTPUT_DIR" "$WANDB_ARGS" \
            "mbpp" 3 "3-shot code benchmark (MBPP pass@1)"
    else
        # Custom tasks: auto-group by recommended n-shot setting
        # Split comma-separated tasks into groups by n-shot
        local tasks_5shot=""
        local tasks_4shot=""
        local tasks_3shot=""
        local tasks_0shot=""

        IFS=',' read -ra TASK_ARRAY <<< "$TASKS"
        for task in "${TASK_ARRAY[@]}"; do
            case "$task" in
                mmlu|hellaswag|arc_easy|arc_challenge|winogrande|boolq|piqa|kmmlu)
                    tasks_5shot="${tasks_5shot:+$tasks_5shot,}$task" ;;
                gsm8k|minerva_math|hendrycks_math)
                    tasks_4shot="${tasks_4shot:+$tasks_4shot,}$task" ;;
                mbpp)
                    tasks_3shot="${tasks_3shot:+$tasks_3shot,}$task" ;;
                *)
                    tasks_0shot="${tasks_0shot:+$tasks_0shot,}$task" ;;
            esac
        done

        [ -n "$tasks_5shot" ] && run_eval "$MODEL_PATH" "$OUTPUT_DIR" "$WANDB_ARGS" \
            "$tasks_5shot" 5 "5-shot: $tasks_5shot"
        [ -n "$tasks_4shot" ] && run_eval "$MODEL_PATH" "$OUTPUT_DIR" "$WANDB_ARGS" \
            "$tasks_4shot" 4 "4-shot: $tasks_4shot"
        [ -n "$tasks_3shot" ] && run_eval "$MODEL_PATH" "$OUTPUT_DIR" "$WANDB_ARGS" \
            "$tasks_3shot" 3 "3-shot: $tasks_3shot"
        [ -n "$tasks_0shot" ] && run_eval "$MODEL_PATH" "$OUTPUT_DIR" "$WANDB_ARGS" \
            "$tasks_0shot" 0 "0-shot: $tasks_0shot"
    fi

    # Upload summary metrics to WandB benchmarks project
    if [ "$WANDB_ENABLED" == "true" ]; then
        echo "📊 Uploading summary to WandB..."
        unset WANDB_START_METHOD  # Clean up env from accelerate eval
        python3 ${SCRIPT_DIR}/upload_benchmarks_to_wandb.py \
            --model-path "$MODEL_PATH" \
            --project "${WANDB_PROJECT_BENCHMARKS:-alpha-benchmarks}" || \
            echo "⚠️  WandB summary upload failed (non-fatal)"
    fi

    echo ""
    echo "✅ Model [$MODEL_IDX/$MODEL_TOTAL] completed: $MODEL_PATH"
}

#==============================================================================
# 메인 실행: 모든 모델에 대해 순차 실행
#==============================================================================

TOTAL_MODELS=${#MODEL_PATHS[@]}
CURRENT=0
FAILED_MODELS=()

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    CURRENT=$((CURRENT + 1))

    if [ ! -d "$MODEL_PATH" ]; then
        echo "⚠️  WARNING: Model path not found, skipping: $MODEL_PATH"
        FAILED_MODELS+=("$MODEL_PATH (not found)")
        continue
    fi

    eval_model "$MODEL_PATH" "$CURRENT" "$TOTAL_MODELS"

    if [ $? -ne 0 ]; then
        echo "⚠️  WARNING: Evaluation failed for: $MODEL_PATH"
        FAILED_MODELS+=("$MODEL_PATH (eval failed)")
    fi
done

# Summary
echo ""
echo "================================================================"
echo "🏁 전체 벤치마크 완료: $TOTAL_MODELS개 모델"
echo "================================================================"
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "⚠️  실패한 모델:"
    for fm in "${FAILED_MODELS[@]}"; do
        echo "  - $fm"
    done
fi
echo "================================================================"
