#!/usr/bin/env bash
# YAML-driven launcher for alpha pretraining.
#
# Usage:
#     bash train.sh <model_preset> <training_preset> <data_preset> [extra-megatron-args...]
#
# Examples:
#     bash train.sh baseline_48L stage1 stage1_v5_blend  # current Stage 1 recipe
#     bash train.sh baseline_48L stage2_3 stage2_2
#     bash train.sh smoke smoke mock                     # 2-iter smoke test
#
# Each preset name resolves to configs/<group>/<preset>.yaml. Top-level keys
# in those YAMLs become Megatron CLI flags:
#     foo: bar     ->  --foo bar
#     foo: true    ->  --foo                (store_true)
#     foo: false   ->  (omitted)
#     foo: [a,b]   ->  --foo a,b            (comma-joined; for nargs='*' use a
#                                            pre-flattened space-separated string instead)
#
# Anything passed after the three presets is forwarded verbatim to Megatron's
# argument parser, so you can override any flag from the command line, e.g.:
#     bash train.sh baseline_48L stage2_3 stage2_2 --lr 5e-4 --train-iters 1000
#
# The launcher itself derives a few values that depend on the run's identity
# (timestamp / preset names) and passes them as flags:
#     --save               outputs/alpha_<MODEL>_<TRAINING>_<TIMESTAMP>/checkpoints
#     --tensorboard-dir    outputs/alpha_<MODEL>_<TRAINING>_<TIMESTAMP>/tensorboard
#     --data-cache-path    configs/data/.cache/<DATA>
#     --wandb-exp-name     <MODEL>_<DATA>_<TIMESTAMP>   (only if WANDB_API_KEY is set)

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $(basename "$0") <model_preset> <training_preset> <data_preset> [extra-args...]" >&2
    exit 2
fi

MODEL_PRESET="$1"
TRAINING_PRESET="$2"
DATA_PRESET="$3"
shift 3

ALPHA_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$ALPHA_DIR")")"
CONFIG_DIR="$ALPHA_DIR/configs"

# ── PYTHONPATH ─────────────────────────────────────────────────────────────
# Megatron-LM-251125 is the dev branch with Muon optimizer support.
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/backends/megatron/Megatron-LM-251125:${PYTHONPATH:-}"

# ── WANDB credentials (optional) ───────────────────────────────────────────
WANDB_SETUP_SCRIPT="$ALPHA_DIR/scripts/setup_wandb.sh"
if [[ -f "$WANDB_SETUP_SCRIPT" ]]; then
    # shellcheck disable=SC1090
    source "$WANDB_SETUP_SCRIPT" 2>/dev/null || true
fi

# ── Smoke preset auto-detection (skip wandb for smoke runs) ───────────────
# Trigger if any of: model/training/data preset is "smoke", or data preset
# is "mock" (mock data is always smoke verification).
SMOKE_RUN=false
if [[ "$MODEL_PRESET" == "smoke" ]] || [[ "$TRAINING_PRESET" == "smoke" ]] \
    || [[ "$DATA_PRESET" == "smoke" ]] || [[ "$DATA_PRESET" == "mock" ]]; then
    SMOKE_RUN=true
fi
if $SMOKE_RUN; then
    export WANDB_MODE=disabled
fi

# ── Megatron-required env ──────────────────────────────────────────────────
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

# ── PyTorch / NCCL tuning ──────────────────────────────────────────────────
# expandable_segments avoids fragmentation under CUDA graph capture.
# NCCL_GRAPH_REGISTER=0 is required by Megatron's validate_args when
# expandable_segments is on AND CUDA graphs are enabled.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_GRAPH_REGISTER=0
export NCCL_DEBUG="${NCCL_DEBUG:-VERSION}"

# ── Transformer Engine (cuDNN-accelerated norms; deterministic off) ───────
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
# NOTE: NVTE_FLASH_ATTN / NVTE_FUSED_ATTN intentionally NOT exported.
#   --attention-backend=auto in YAML lets TE pick the right backend
#   (fused for QK-Clip return_max_logit, flash otherwise). Globally pinning
#   them here would break --attention-backend=auto's selection logic.

# ── Threading (220-core host, 32 dataloader workers × 8 OMP) ──────────────
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-256}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

# Quiet HF tokenizers fork-after-parallelism warning. Pre-tokenized .bin/.idx
# data means the tokenizer is only consulted for vocab metadata.
export TOKENIZERS_PARALLELISM=false

# ── Distributed (multi-node aware: k8s WORLD_SIZE/RANK, single-node fallback) ─
NUM_NODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU:-$(python3 -c 'import torch; print(torch.cuda.device_count())')}
[ -z "${MASTER_ADDR:-}" ] && export MASTER_ADDR=localhost
[ -z "${MASTER_PORT:-}" ] && export MASTER_PORT=$(shuf -n 1 -i 10000-65535)

# ── Output directory ───────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="alpha_${MODEL_PRESET}_${TRAINING_PRESET}_${TIMESTAMP}"
OUTPUT_DIR="$ALPHA_DIR/outputs/$RUN_NAME"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard"
LOG_DIR="$OUTPUT_DIR/logs"
DATA_CACHE_DIR="$CONFIG_DIR/data/.cache/$DATA_PRESET"
mkdir -p "$CHECKPOINT_DIR" "$TENSORBOARD_DIR" "$LOG_DIR" "$DATA_CACHE_DIR"

# ── YAML → flags helper ────────────────────────────────────────────────────
# Top-level keys become --key flags. Booleans: emit when true, omit when
# false. Lists become --key a,b (comma). Strings with embedded spaces (e.g.
# pre-flattened nargs='*' values like "0.3 /foo 0.7 /bar") survive by being
# emitted as one whitespace-delimited line; xargs+read split them correctly
# into argparse's expected position arguments.
yaml_to_flags() {
    local yaml_path="$1"
    if [[ ! -f "$yaml_path" ]]; then
        echo "preset config not found: $yaml_path" >&2
        exit 1
    fi
    python3 - "$yaml_path" <<'PY'
import sys, yaml
with open(sys.argv[1]) as fh:
    cfg = yaml.safe_load(fh) or {}
for key, value in cfg.items():
    flag = f"--{key}"
    if isinstance(value, bool):
        if value:
            print(flag)
    elif isinstance(value, list):
        print(flag, ",".join(str(v) for v in value))
    elif value is None:
        continue
    else:
        print(flag, str(value))
PY
}

MODEL_YAML="$CONFIG_DIR/model/${MODEL_PRESET}.yaml"
TRAINING_YAML="$CONFIG_DIR/training/${TRAINING_PRESET}.yaml"
DATA_YAML="$CONFIG_DIR/data/${DATA_PRESET}.yaml"

# read -a + xargs splits on any whitespace, so pre-flattened "weight path
# weight path" strings in --data-path / --recompute-modules / etc. pass
# through as separate argv tokens (matching argparse nargs='*' expectations).
read -r -a MODEL_FLAGS    < <(yaml_to_flags "$MODEL_YAML"    | xargs)
read -r -a TRAINING_FLAGS < <(yaml_to_flags "$TRAINING_YAML" | xargs)
read -r -a DATA_FLAGS     < <(yaml_to_flags "$DATA_YAML"     | xargs)

# ── WANDB derived flags ───────────────────────────────────────────────────
# Megatron requires --wandb-exp-name when --wandb-project is in training preset
# (which stage1.yaml has). For smoke runs we still emit a dummy name
# so argparse validation passes, but WANDB_MODE=disabled (set above) blocks
# actual logging.
WANDB_FLAGS=()
if $SMOKE_RUN; then
    WANDB_FLAGS=(--wandb-exp-name "smoke_${TIMESTAMP}")
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_FLAGS=(--wandb-exp-name "${MODEL_PRESET}_${DATA_PRESET}_${TIMESTAMP}")
fi

# ── Banner ─────────────────────────────────────────────────────────────────
echo "=============================================="
echo "Alpha train.sh"
echo "=============================================="
echo "  model:          $MODEL_YAML"
echo "  training:       $TRAINING_YAML"
echo "  data:           $DATA_YAML"
echo "  run dir:        $OUTPUT_DIR"
echo "  GPUs/node:      $GPUS_PER_NODE   (nodes: $NUM_NODES, rank: $NODE_RANK)"
echo "  master:         $MASTER_ADDR:$MASTER_PORT"
echo "  python:         $(which python3)"
echo "  Megatron:       backends/megatron/Megatron-LM-251125"
if $SMOKE_RUN; then
    echo "  wandb:          DISABLED (smoke preset detected)"
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "  wandb:          online (project: alpha-pretraining)"
else
    echo "  wandb:          off (no WANDB_API_KEY)"
fi
echo "=============================================="

# Persist a config snapshot for reproducibility.
{
    echo "# alpha train.sh snapshot — $(date)"
    echo "# Run: $RUN_NAME"
    echo "# Args: $MODEL_PRESET $TRAINING_PRESET $DATA_PRESET $*"
    echo
    echo "## model: $MODEL_YAML"
    cat "$MODEL_YAML"
    echo
    echo "## training: $TRAINING_YAML"
    cat "$TRAINING_YAML"
    echo
    echo "## data: $DATA_YAML"
    cat "$DATA_YAML"
} > "$LOG_DIR/config_snapshot.yaml"

LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

# ── Launch ─────────────────────────────────────────────────────────────────
# Run with `python -m torch.distributed.run` rather than the `torchrun` shim
# (system shims often hardcode #!/usr/bin/python and bypass our venv).
exec python3 -m torch.distributed.run \
    --nnodes "$NUM_NODES" \
    --node_rank "$NODE_RANK" \
    --nproc_per_node "$GPUS_PER_NODE" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    "$ALPHA_DIR/pretrain_alpha.py" \
    "${MODEL_FLAGS[@]}" \
    "${TRAINING_FLAGS[@]}" \
    "${DATA_FLAGS[@]}" \
    --save "$CHECKPOINT_DIR" \
    --tensorboard-dir "$TENSORBOARD_DIR" \
    --data-cache-path "$DATA_CACHE_DIR" \
    "${WANDB_FLAGS[@]}" \
    "$@" \
    2>&1 | tee "$LOG_FILE"
