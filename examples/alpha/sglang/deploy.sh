#!/bin/bash
# Alpha Model SGLang Deployment Script
#
# Uses the SGLang venv (backends/sglang/.venv) created by setup_pai_megatron_env_A100.sh.
# SGLang runs in a separate venv from Megatron to avoid PyTorch ABI conflicts.
#
# Usage:
#   bash deploy.sh <HF_MODEL_PATH> [OPTIONS]
#
# Modes:
#   Option A (HF Fallback):  bash deploy.sh /path/to/alpha-hf --mode fallback
#   Option B (Qwen3-Next):   bash deploy.sh /path/to/alpha-hf --mode native
#
# The --mode native option first converts the checkpoint to Qwen3-Next format,
# then launches SGLang with native hybrid model optimizations (MambaRadixCache,
# dual memory pool, elastic memory).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALPHA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MEGATRON_PATCH_PATH="$(cd "$ALPHA_DIR/../.." && pwd)"

# ─── SGLang Backend Configuration ──────────────────────────
SGLANG_VERSION="${SGLANG_VERSION:-v0.5.2}"
SGLANG_BACKEND="${MEGATRON_PATCH_PATH}/backends/sglang/sglang-${SGLANG_VERSION}"
SGLANG_SETUP="${MEGATRON_PATCH_PATH}/backends/sglang/setup.sh"
SGLANG_VENV="${MEGATRON_PATCH_PATH}/backends/sglang/.venv"

# ─── Defaults ───────────────────────────────────────────────
MODE="fallback"          # fallback | native
PORT=30000
HOST="0.0.0.0"
TP=1                     # Tensor Parallel (Alpha Mamba layers require TP=1)
DP=1                     # Data Parallel
EP=1                     # Expert Parallel
MAX_MAMBA_CACHE=""       # --max-mamba-cache-size (max concurrent requests with Mamba state)
DTYPE="bfloat16"
MAX_MODEL_LEN=4096
EXTRA_ARGS=""

# ─── Parse Arguments ────────────────────────────────────────
usage() {
    cat <<EOF
Usage: bash deploy.sh <HF_MODEL_PATH> [OPTIONS]

Arguments:
  HF_MODEL_PATH          Path to Alpha HuggingFace checkpoint directory

Options:
  --mode MODE             Deployment mode: fallback (default) or native
  --port PORT             Server port (default: 30000)
  --host HOST             Server host (default: 0.0.0.0)
  --tp TP                 Tensor parallel size (default: 1, must be 1 for Mamba)
  --dp DP                 Data parallel size (default: 1)
  --ep EP                 Expert parallel size (default: 1)
  --max-mamba-cache N     Max Mamba cache entries (default: auto, native mode only)
  --dtype DTYPE           Data type: bfloat16, float16, float32 (default: bfloat16)
  --max-model-len LEN     Maximum sequence length (default: 4096)
  --extra-args "ARGS"     Additional SGLang arguments

Environment:
  SGLANG_VERSION          SGLang backend version (default: v0.5.2)

Examples:
  # Quick test with HF fallback (no SGLang modification needed)
  bash deploy.sh /data/models/alpha-hf-50000

  # Native mode with dual memory pool optimization
  bash deploy.sh /data/models/alpha-hf-50000 --mode native --ep 8

  # Multi-GPU data parallel
  bash deploy.sh /data/models/alpha-hf-50000 --dp 4 --port 30000
EOF
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

MODEL_PATH="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2";          shift 2 ;;
        --port)       PORT="$2";          shift 2 ;;
        --host)       HOST="$2";          shift 2 ;;
        --tp)         TP="$2";            shift 2 ;;
        --dp)         DP="$2";            shift 2 ;;
        --ep)         EP="$2";            shift 2 ;;
        --max-mamba-cache) MAX_MAMBA_CACHE="$2"; shift 2 ;;
        --dtype)      DTYPE="$2";         shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --extra-args) EXTRA_ARGS="$2";    shift 2 ;;
        -h|--help)    usage ;;
        *)            echo "Unknown option: $1"; usage ;;
    esac
done

# ─── Validation ─────────────────────────────────────────────
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: config.json not found in $MODEL_PATH"
    echo "  Make sure this is a valid HuggingFace checkpoint directory."
    exit 1
fi

if [ "$TP" -gt 1 ]; then
    echo "WARNING: Alpha's Mamba (GatedDeltaNet) layers do not support TP > 1."
    echo "  Setting TP=1. Use --dp or --ep for multi-GPU serving."
    TP=1
fi

# ─── SGLang Backend Setup ──────────────────────────────────
# Use local backend from backends/sglang/ (not pip-installed)
if [ ! -d "$SGLANG_BACKEND/python/sglang" ]; then
    echo "SGLang backend not found. Running setup..."
    bash "$SGLANG_SETUP" "$SGLANG_VERSION"
elif [ ! -f "$SGLANG_BACKEND/python/sglang/srt/models/alpha.py" ]; then
    echo "Alpha model adapter not installed. Running setup..."
    bash "$SGLANG_SETUP" "$SGLANG_VERSION"
fi

# ─── SGLang venv 활성화 ──────────────────────────────────
# SGLang은 Megatron과 별도 venv에서 실행 (ABI 충돌 방지)
if [ -f "$SGLANG_VENV/bin/activate" ]; then
    echo "Activating SGLang venv: $SGLANG_VENV"
    source "$SGLANG_VENV/bin/activate"
else
    echo "WARNING: SGLang venv not found at $SGLANG_VENV"
    echo "  Run setup_pai_megatron_env_A100.sh or create the venv manually."
    echo "  Falling back to system Python (sgl-kernel/flashinfer may not work)."
    # Fallback: PYTHONPATH 방식
    export PYTHONPATH="${SGLANG_BACKEND}/python:${PYTHONPATH:-}"
fi

echo "═══════════════════════════════════════════════════════"
echo "  Alpha Model SGLang Deployment"
echo "═══════════════════════════════════════════════════════"
echo "  Mode:       $MODE"
echo "  Backend:    $SGLANG_BACKEND"
echo "  Model:      $MODEL_PATH"
echo "  Port:       $PORT"
echo "  TP/DP/EP:   $TP / $DP / $EP"
echo "  Dtype:      $DTYPE"
echo "  Max Len:    $MAX_MODEL_LEN"
if [ "$MODE" = "native" ] && [ -n "$MAX_MAMBA_CACHE" ]; then
    echo "  Mamba Cache: $MAX_MAMBA_CACHE"
fi
echo "═══════════════════════════════════════════════════════"

# ─── Option A: HF Fallback Mode ────────────────────────────
if [ "$MODE" = "fallback" ]; then
    echo ""
    echo "[Option A] Launching with HuggingFace Transformers fallback..."
    echo "  - Uses Alpha's modeling_alpha.py via --trust-remote-code"
    echo "  - No hybrid optimizations (MambaRadixCache, dual pool disabled)"
    echo "  - Good for quick testing and validation"
    echo ""

    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --trust-remote-code \
        --port "$PORT" \
        --host "$HOST" \
        --tp-size "$TP" \
        --dp-size "$DP" \
        --dtype "$DTYPE" \
        --context-length "$MAX_MODEL_LEN" \
        $EXTRA_ARGS

# ─── Option B: Native Qwen3-Next Mode ──────────────────────
elif [ "$MODE" = "native" ]; then
    echo ""
    echo "[Option B] Launching with native Qwen3-Next hybrid optimizations..."
    echo "  - MambaRadixCache: dual prefix caching (SSM state + KV cache)"
    echo "  - Dual Memory Pool: elastic Mamba/KV allocation"
    echo "  - HybridLinearAttnBackend: optimized layer dispatch"
    echo ""

    # Step 1: Convert config to Qwen3-Next format
    CONVERTED_PATH="${MODEL_PATH}_sglang_native"
    if [ ! -f "$CONVERTED_PATH/config.json" ]; then
        echo "[Step 1] Converting Alpha config to Qwen3-Next format..."
        python "$SCRIPT_DIR/convert_config_for_sglang.py" \
            --input-path "$MODEL_PATH" \
            --output-path "$CONVERTED_PATH" \
            --mode native
        echo "  Converted checkpoint: $CONVERTED_PATH"
    else
        echo "[Step 1] Using existing converted checkpoint: $CONVERTED_PATH"
    fi

    # Step 2: Launch SGLang with native hybrid support
    echo "[Step 2] Launching SGLang server..."

    NATIVE_ARGS=""
    # Expert parallel for MoE distribution
    if [ "$EP" -gt 1 ]; then
        NATIVE_ARGS="$NATIVE_ARGS --ep-size $EP"
    fi

    # Mamba cache size (optional)
    if [ -n "$MAX_MAMBA_CACHE" ]; then
        NATIVE_ARGS="$NATIVE_ARGS --max-mamba-cache-size $MAX_MAMBA_CACHE"
    fi

    python -m sglang.launch_server \
        --model-path "$CONVERTED_PATH" \
        --trust-remote-code \
        --port "$PORT" \
        --host "$HOST" \
        --tp-size "$TP" \
        --dp-size "$DP" \
        --dtype "$DTYPE" \
        --context-length "$MAX_MODEL_LEN" \
        $NATIVE_ARGS \
        $EXTRA_ARGS

else
    echo "ERROR: Unknown mode '$MODE'. Use 'fallback' or 'native'."
    exit 1
fi
