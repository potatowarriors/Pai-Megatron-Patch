#!/bin/bash
# Alpha Model MG ↔ HF Weight Validation Script
# ==============================================
#
# Validates that Megatron checkpoint weights match HuggingFace model weights.
# NOTE: MG GatedDeltaNet does NOT support inference, so we compare weights directly.
#
# Usage:
#   bash validate.sh <MG_CHECKPOINT> <HF_MODEL> [CONFIG_NAME] [--verbose]
#
# Example:
#   bash validate.sh /path/to/mg/checkpoint /path/to/hf/model baseline_48L
#   bash validate.sh /path/to/mg/checkpoint /path/to/hf/model baseline_48L --verbose
#
# Arguments:
#   MG_CHECKPOINT: Path to Megatron checkpoint directory
#   HF_MODEL:      Path to HuggingFace model directory
#   CONFIG_NAME:   Config name (default: baseline_48L)
#   --verbose:     Print detailed comparison for all weights

set -e

# Arguments
MG_CHECKPOINT=${1:-""}
HF_MODEL=${2:-""}
CONFIG_NAME=${3:-"baseline_48L"}
VERBOSE_FLAG=""

# Check for --verbose in any position
for arg in "$@"; do
    if [ "$arg" == "--verbose" ]; then
        VERBOSE_FLAG="--verbose"
    fi
done

# Validate arguments
if [ -z "$MG_CHECKPOINT" ] || [ -z "$HF_MODEL" ]; then
    echo "Usage: bash validate.sh <MG_CHECKPOINT> <HF_MODEL> [CONFIG_NAME] [--verbose]"
    echo ""
    echo "Validates that Megatron checkpoint weights match HuggingFace model weights."
    echo "NOTE: MG GatedDeltaNet does NOT support inference, so we compare weights directly."
    echo ""
    echo "Arguments:"
    echo "  MG_CHECKPOINT: Path to Megatron checkpoint directory"
    echo "  HF_MODEL:      Path to HuggingFace model directory"
    echo "  CONFIG_NAME:   Config name (default: baseline_48L)"
    echo "  --verbose:     Print detailed comparison for all weights"
    exit 1
fi

# Paths
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
MEGATRON_PATH="$ROOT_DIR/backends/megatron/Megatron-LM-251125"

# Set PYTHONPATH
export PYTHONPATH="$ROOT_DIR:$MEGATRON_PATH:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "=============================================================="
echo "Alpha Model MG ↔ HF Weight Validation"
echo "=============================================================="
echo "MG Checkpoint: $MG_CHECKPOINT"
echo "HF Model:      $HF_MODEL"
echo "Config:        $CONFIG_NAME"
echo "Verbose:       ${VERBOSE_FLAG:-no}"
echo "=============================================================="

# Load config
CONFIG_FILE="$SCRIPT_DIR/configs/model/${CONFIG_NAME}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Loading config from: $CONFIG_FILE"

# Parse config using Python
read_yaml_value() {
    python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    config = yaml.safe_load(f)
def get_nested(d, keys):
    for k in keys.split('.'):
        if isinstance(d, dict):
            d = d.get(k, None)
        else:
            return None
    return d
value = get_nested(config, '$1')
if value is not None:
    print(value)
"
}

# Extract model config
NUM_LAYERS=$(read_yaml_value "model.num_layers")
HIDDEN_SIZE=$(read_yaml_value "model.hidden_size")
FFN_HIDDEN_SIZE=$(read_yaml_value "model.ffn_hidden_size")
NUM_ATTENTION_HEADS=$(read_yaml_value "model.num_attention_heads")
KV_CHANNELS=$(read_yaml_value "model.kv_channels")
NUM_QUERY_GROUPS=$(read_yaml_value "model.num_query_groups")

# MoE config
NUM_EXPERTS=$(read_yaml_value "model.moe.num_experts")
MOE_FFN_HIDDEN_SIZE=$(read_yaml_value "model.moe.moe_ffn_hidden_size")
ROUTER_TOPK=$(read_yaml_value "model.moe.router_topk")
SHARED_EXPERT_SIZE=$(read_yaml_value "model.moe.shared_expert_intermediate_size")

# Hybrid config
ATTENTION_RATIO=$(read_yaml_value "model.hybrid.attention_ratio")
MLP_RATIO=$(read_yaml_value "model.hybrid.mlp_ratio")
OVERRIDE_PATTERN=$(read_yaml_value "model.hybrid.override_pattern")
MAMBA_STATE_DIM=$(read_yaml_value "model.hybrid.mamba_state_dim")
MAMBA_HEAD_DIM=$(read_yaml_value "model.hybrid.mamba_head_dim")
MAMBA_NUM_HEADS=$(read_yaml_value "model.hybrid.mamba_num_heads")

# Vocab
PADDED_VOCAB_SIZE=$(read_yaml_value "model.padded_vocab_size")
TOKENIZER_PATH="$SCRIPT_DIR/tokenizer"

# Other settings
ROTARY_BASE=$(read_yaml_value "model.rotary_base")
ROTARY_PERCENT=$(read_yaml_value "model.rotary_percent")
NORM_EPSILON=$(read_yaml_value "model.norm_epsilon")

echo ""
echo "Model Config:"
echo "  Layers: $NUM_LAYERS"
echo "  Hidden: $HIDDEN_SIZE"
echo "  Heads:  $NUM_ATTENTION_HEADS"
echo "  Experts: $NUM_EXPERTS"
echo "  Pattern: ${OVERRIDE_PATTERN:0:50}..."
echo ""

# Run validation with torchrun (single GPU for validation)
torchrun --nproc_per_node=1 \
    "$SCRIPT_DIR/validate_mg_hf_full.py" \
    --mg-checkpoint "$MG_CHECKPOINT" \
    --hf-model "$HF_MODEL" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers "$NUM_LAYERS" \
    --hidden-size "$HIDDEN_SIZE" \
    --ffn-hidden-size "$FFN_HIDDEN_SIZE" \
    --num-attention-heads "$NUM_ATTENTION_HEADS" \
    --kv-channels "$KV_CHANNELS" \
    --group-query-attention \
    --num-query-groups "$NUM_QUERY_GROUPS" \
    --num-experts "$NUM_EXPERTS" \
    --moe-ffn-hidden-size "$MOE_FFN_HIDDEN_SIZE" \
    --moe-router-topk "$ROUTER_TOPK" \
    --moe-shared-expert-intermediate-size "$SHARED_EXPERT_SIZE" \
    --moe-router-load-balancing-type "aux_loss" \
    --moe-aux-loss-coeff 0.001 \
    --moe-grouped-gemm \
    --hybrid-attention-ratio "$ATTENTION_RATIO" \
    --hybrid-mlp-ratio "$MLP_RATIO" \
    --hybrid-override-pattern "$OVERRIDE_PATTERN" \
    --mamba-state-dim "$MAMBA_STATE_DIM" \
    --mamba-head-dim "$MAMBA_HEAD_DIM" \
    --mamba-num-heads "$MAMBA_NUM_HEADS" \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --padded-vocab-size "$PADDED_VOCAB_SIZE" \
    --tokenizer-type NullTokenizer \
    --vocab-size 151936 \
    --patch-tokenizer-type Qwen3Tokenizer \
    --load "$TOKENIZER_PATH" \
    --extra-vocab-size 0 \
    --position-embedding-type rope \
    --rotary-base "$ROTARY_BASE" \
    --rotary-percent "$ROTARY_PERCENT" \
    --normalization RMSNorm \
    --norm-epsilon "$NORM_EPSILON" \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --bf16 \
    --micro-batch-size 1 \
    --no-load-optim \
    --no-load-rng \
    --ckpt-format torch_dist \
    --threshold 0.01 \
    $VERBOSE_FLAG

echo ""
echo "Validation completed."
