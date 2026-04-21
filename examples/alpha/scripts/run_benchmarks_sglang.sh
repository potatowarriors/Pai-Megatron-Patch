#!/bin/bash
#==============================================================================
# Alpha 모델 SGLang 벤치마크 평가 스크립트
# LM-Evaluation-Harness + SGLang offline Engine 기반 평가
#
# 기존 run_benchmarks.sh와 동일한 태스크/n-shot 구성을 사용하되,
# HF transformers 대신 SGLang Engine으로 추론하여 벤치마크 속도를 대폭 향상.
#
# Usage:
#   bash run_benchmarks_sglang.sh <MODEL_PATH> [--tasks TASKS] [--batch-size BS]
#   bash run_benchmarks_sglang.sh <MODEL_PATH1> <MODEL_PATH2> --tasks standard
#
# Environment:
#   SGLang venv가 활성화된 상태에서 실행하거나, 스크립트가 자동 활성화.
#   PYTHONPATH에 로컬 SGLang 백엔드 경로가 포함되어야 함.
#==============================================================================

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
ALPHA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ALPHA_DIR/../.." && pwd)"

#==============================================================================
# SGLang 환경 설정
#==============================================================================
SGLANG_VERSION="${SGLANG_VERSION:-v0.5.2}"
SGLANG_BACKEND="${REPO_ROOT}/backends/sglang/sglang-${SGLANG_VERSION}"
SGLANG_VENV="${REPO_ROOT}/backends/sglang/.venv"

# SGLang venv 활성화 (아직 활성화되지 않은 경우)
if [ -z "$VIRTUAL_ENV" ] || [[ "$VIRTUAL_ENV" != *"sglang"* ]]; then
    if [ -f "$SGLANG_VENV/bin/activate" ]; then
        echo "Activating SGLang venv: $SGLANG_VENV"
        source "$SGLANG_VENV/bin/activate"
    else
        echo "ERROR: SGLang venv not found at $SGLANG_VENV"
        echo "  Run setup_pai_megatron_env_A100.sh or create the venv manually."
        exit 1
    fi
fi

# 로컬 SGLang 백엔드 (패치된 Alpha adapter 포함)를 PYTHONPATH에 추가
export PYTHONPATH="${SGLANG_BACKEND}/python:${PYTHONPATH:-}"

# SGLang import 확인
python -c "import sglang" 2>/dev/null || {
    echo "ERROR: Cannot import sglang. Check SGLang venv and PYTHONPATH."
    exit 1
}

#==============================================================================
# WANDB 환경 자동 로드
#==============================================================================
WANDB_SETUP_SCRIPT="${SCRIPT_DIR}/setup_wandb.sh"
if [ -f "$WANDB_SETUP_SCRIPT" ]; then
    source "$WANDB_SETUP_SCRIPT" 2>/dev/null || true
fi

#==============================================================================
# 인자 처리 (기존 run_benchmarks.sh와 동일)
#==============================================================================
DEFAULT_TASKS="standard"
BATCH_SIZE="auto"

# SGLang model args
TP_SIZE=${TP_SIZE:-1}       # Alpha Mamba layers require TP=1
DTYPE=${DTYPE:-bfloat16}
MEM_FRACTION=${MEM_FRACTION:-0.7}  # Reserve more memory for logprobs computation

# WandB settings
WANDB_PROJECT=${WANDB_PROJECT:-"alpha-evals"}
WANDB_ENABLED=${WANDB_ENABLED:-true}

# Parse arguments
MODEL_PATHS=()
TASKS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)
            TASKS="$2"; shift 2 ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --tp-size)
            TP_SIZE="$2"; shift 2 ;;
        --dtype)
            DTYPE="$2"; shift 2 ;;
        --help|-h)
            MODEL_PATHS=(); break ;;
        --*)
            echo "Unknown option: $1"; exit 1 ;;
        *)
            if [ -d "$1" ] || [[ "$1" == */* ]] || [[ "$1" == .* ]]; then
                MODEL_PATHS+=("$1")
            elif [ ${#MODEL_PATHS[@]} -eq 0 ]; then
                MODEL_PATHS+=("$1")
            elif [ -z "$TASKS" ]; then
                TASKS="$1"
            elif [ "$BATCH_SIZE" == "auto" ]; then
                BATCH_SIZE="$1"
            fi
            shift ;;
    esac
done

TASKS=${TASKS:-$DEFAULT_TASKS}
TASKS=${TASKS//siqa/social_iqa}

if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    echo "Usage: $0 <MODEL_PATH> [MODEL_PATH2 ...] [--tasks TASKS] [--batch-size BS]"
    echo ""
    echo "Examples:"
    echo "  # Single model"
    echo "  $0 outputs/alpha_baseline_48L_*/hfmodel_0050000 --tasks standard"
    echo ""
    echo "  # Quick validation"
    echo "  $0 path/to/hfmodel --tasks hellaswag"
    echo ""
    echo "  # Multiple models"
    echo "  $0 path/to/hfmodel_0050000 path/to/hfmodel_0100000 --tasks standard"
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
    echo "  TP_SIZE        - Tensor parallel size (default: 1, must be 1 for Alpha)"
    echo "  DTYPE          - Data type (default: bfloat16)"
    exit 1
fi

echo "================================================================"
echo "Alpha 모델 SGLang 벤치마크 평가"
echo "================================================================"
echo "Engine: SGLang (offline)"
echo "Models: ${#MODEL_PATHS[@]}개"
for mp in "${MODEL_PATHS[@]}"; do
    echo "  - $mp"
done
echo "Tasks: $TASKS"
echo "Batch Size: $BATCH_SIZE"
echo "TP Size: $TP_SIZE"
echo "Dtype: $DTYPE"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Enabled: $WANDB_ENABLED"
echo "================================================================"

#==============================================================================
# 환경 설정
#==============================================================================
export HF_HOME=/home/work/Datasets/benchmarks
export HF_DATASETS_CACHE=/home/work/Datasets/benchmarks
export HF_TOKEN="${HF_TOKEN:-}"
export HF_ALLOW_CODE_EVAL=1
mkdir -p $HF_DATASETS_CACHE

if [ "$WANDB_ENABLED" == "true" ] && [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. WandB logging disabled."
    echo "   Run: source scripts/setup_wandb.sh"
    WANDB_ENABLED=false
fi

#==============================================================================
# 모델별 벤치마크 실행 함수
#==============================================================================

# SGLang 기반 evaluation 실행
# Usage: run_eval <model_path> <output_dir> <wandb_args> <tasks> <num_fewshot> <description>
run_eval() {
    local model_path=$1
    local output_dir=$2
    local wandb_args=$3
    local tasks=$4
    local num_fewshot=$5
    local desc=$6

    echo ""
    echo "------------------------------------------------------------"
    echo "Running: $desc"
    echo "   Tasks: $tasks"
    echo "   N-shot: $num_fewshot"
    echo "------------------------------------------------------------"

    python -m lm_eval --model sglang \
        --model_args pretrained=$model_path,dtype=$DTYPE,tp_size=$TP_SIZE,mem_fraction_static=$MEM_FRACTION \
        --tasks $tasks \
        --num_fewshot $num_fewshot \
        --batch_size $BATCH_SIZE \
        --output_path $output_dir \
        --confirm_run_unsafe_code \
        $wandb_args
}

# 단일 모델 평가
eval_model() {
    local MODEL_PATH=$1
    local MODEL_IDX=$2
    local MODEL_TOTAL=$3

    # Run name 추출 (기존 run_benchmarks.sh와 동일 로직)
    local RUN_NAME=$(basename $(dirname $MODEL_PATH))
    if [ "$RUN_NAME" == "." ] || [ -z "$RUN_NAME" ] || [ "$RUN_NAME" == "outputs" ]; then
        RUN_NAME=$(basename $MODEL_PATH)
    fi

    local MODEL_BASENAME=$(basename $MODEL_PATH)
    if [[ "$MODEL_BASENAME" =~ ^hfmodel_([0-9]+)$ ]]; then
        local ITERATION="${BASH_REMATCH[1]}"
        RUN_NAME="${RUN_NAME}_iter${ITERATION}"
    fi

    local OUTPUT_DIR="${MODEL_PATH}/eval_results"
    mkdir -p $OUTPUT_DIR

    local WANDB_ARGS=""
    if [ "$WANDB_ENABLED" == "true" ]; then
        export WANDB_START_METHOD=thread
        WANDB_ARGS="--wandb_args project=$WANDB_PROJECT,name=${RUN_NAME}_sglang,job_type=eval"
    fi

    echo ""
    echo "================================================================"
    echo "Model [$MODEL_IDX/$MODEL_TOTAL]: $MODEL_PATH"
    echo "----------------------------------------------------------------"
    echo "Run Name: ${RUN_NAME}"
    echo "Results: $OUTPUT_DIR"
    if [ -n "$WANDB_ARGS" ]; then
        echo "WandB Run: ${RUN_NAME}_sglang"
    fi
    echo "================================================================"

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
        # Custom tasks: n-shot 자동 그룹핑
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

    # WandB summary 업로드
    if [ "$WANDB_ENABLED" == "true" ]; then
        echo "Uploading summary to WandB..."
        unset WANDB_START_METHOD
        python3 ${SCRIPT_DIR}/upload_benchmarks_to_wandb.py \
            --model-path "$MODEL_PATH" \
            --project "${WANDB_PROJECT_BENCHMARKS:-alpha-benchmarks}" || \
            echo "WARNING: WandB summary upload failed (non-fatal)"
    fi

    echo ""
    echo "Model [$MODEL_IDX/$MODEL_TOTAL] completed: $MODEL_PATH"
}

#==============================================================================
# 메인 실행
#==============================================================================

TOTAL_MODELS=${#MODEL_PATHS[@]}
CURRENT=0
FAILED_MODELS=()

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    CURRENT=$((CURRENT + 1))

    if [ ! -d "$MODEL_PATH" ]; then
        echo "WARNING: Model path not found, skipping: $MODEL_PATH"
        FAILED_MODELS+=("$MODEL_PATH (not found)")
        continue
    fi

    eval_model "$MODEL_PATH" "$CURRENT" "$TOTAL_MODELS"

    if [ $? -ne 0 ]; then
        echo "WARNING: Evaluation failed for: $MODEL_PATH"
        FAILED_MODELS+=("$MODEL_PATH (eval failed)")
    fi
done

# Summary
echo ""
echo "================================================================"
echo "SGLang 벤치마크 완료: $TOTAL_MODELS개 모델"
echo "================================================================"
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "  Failed:"
    for fm in "${FAILED_MODELS[@]}"; do
        echo "  - $fm"
    done
fi
echo "================================================================"
