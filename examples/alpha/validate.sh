#!/bin/bash
# Alpha Model MG ↔ HF Weight Validation Script
# ==============================================
#
# Validates that Megatron checkpoint weights match the converted HuggingFace
# model weights. MG GatedDeltaNet does NOT support inference, so this is a
# direct weight-by-weight comparison (no generation).
#
# Model-skeleton flags are derived from the checkpoint's own common.pt (ground
# truth) via alpha_config.py — no hardcoded vocab/expert/tokenizer values, no
# YAML parsing that can drift from what was actually trained.
#
# Usage:
#   bash validate.sh <MG_CHECKPOINT> <HF_MODEL> [CONFIG_NAME] [--verbose]
#
# Arguments:
#   MG_CHECKPOINT : Megatron checkpoint dir (iter_NNNNNNN, checkpoints/, or run dir)
#   HF_MODEL      : Converted HuggingFace model directory
#   CONFIG_NAME   : (ignored; kept for back-compat — config now comes from the checkpoint)
#   --verbose     : Print per-weight comparison

set -e

MG_CHECKPOINT=${1:-""}
HF_MODEL=${2:-""}
VERBOSE_FLAG=""
for arg in "$@"; do
    [ "$arg" == "--verbose" ] && VERBOSE_FLAG="--verbose"
done

if [ -z "$MG_CHECKPOINT" ] || [ -z "$HF_MODEL" ]; then
    echo "Usage: bash validate.sh <MG_CHECKPOINT> <HF_MODEL> [CONFIG_NAME] [--verbose]"
    echo ""
    echo "Validates that Megatron checkpoint weights match HuggingFace model weights."
    echo "Model config is derived from the checkpoint's common.pt (ground truth)."
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
MEGATRON_PATH="$ROOT_DIR/backends/megatron/Megatron-LM-251125"
ALPHA_CONFIG_TOOL="$SCRIPT_DIR/tools/alpha_config.py"

export PYTHONPATH="$ROOT_DIR:$MEGATRON_PATH:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

echo "=============================================================="
echo "Alpha Model MG ↔ HF Weight Validation"
echo "=============================================================="
echo "MG Checkpoint: $MG_CHECKPOINT"
echo "HF Model:      $HF_MODEL"
echo "Verbose:       ${VERBOSE_FLAG:-no}"
echo "=============================================================="

# Derive the model-skeleton flags from the checkpoint (one token per line so the
# '*' in the hybrid pattern is never glob-expanded or word-split).
readarray -t MODEL_ARGS < <(python3 "$ALPHA_CONFIG_TOOL" emit-megatron-flags --from-checkpoint "$MG_CHECKPOINT")
if [ ${#MODEL_ARGS[@]} -eq 0 ]; then
    echo "Error: failed to derive model flags from checkpoint (common.pt not found?)"
    exit 1
fi

echo "Derived ${#MODEL_ARGS[@]} model-flag tokens from checkpoint common.pt."

# Single-GPU validation: load the EP=8 checkpoint under EP=1 (torch_dist reshards).
torchrun --nproc_per_node=1 \
    "$SCRIPT_DIR/validate_mg_hf_full.py" \
    --mg-checkpoint "$MG_CHECKPOINT" \
    --hf-model "$HF_MODEL" \
    "${MODEL_ARGS[@]}" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --bf16 \
    --micro-batch-size 1 \
    --no-load-optim \
    --no-load-rng \
    --ckpt-format torch_dist \
    --threshold 0.01 \
    $VERBOSE_FLAG

echo ""
echo "Validation completed."
