#!/bin/bash
# Alpha v2 — Unified evaluation pipeline
# ======================================
# One entrypoint for the previously-manual MG→HF → validate → benchmark flow.
# Everything is bound to the checkpoint's own common.pt (ground truth), so the
# 184-vs-192 / mamba-dim / tokenizer drift that plagued the split scripts can't
# recur. GPU count is parameterized (EP = #GPU; torch_dist reshards on load).
#
# Stages:
#   0   preflight  — derive expected HF config from common.pt, sanity-check it
#   1   convert    — MG→HF (run_convert.sh, EP=#GPU)
#   1.5 verify     — produced config.json ↔ common.pt + tokenizer round-trip
#   2   validate   — MG↔HF weight equivalence (validate.sh)
#   3   benchmark  — lm-eval suite (opt-in, --benchmark)
#
# Usage:
#   bash evaluate.sh <RUN_DIR|CKPT_DIR> [options]
#
# Options:
#   --iter N|latest   Iteration to evaluate (default: latest)
#   --gpus N          GPUs for conversion/benchmark (default: auto-detect). EP=N.
#   --out DIR         HF output dir (default: <run>/hfmodel_<iter>)
#   --benchmark       Run the lm-eval benchmark suite (multi-GPU, long)
#   --tasks T         Benchmark tasks (default: standard)
#   --skip-convert    Reuse an existing HF dir (skip Stage 1)
#   --skip-validate   Skip Stage 2 weight validation
#
# Examples:
#   bash evaluate.sh outputs/alpha_baseline_48L_stage1_20260512_170157 --gpus 4
#   bash evaluate.sh outputs/<run> --gpus 4 --benchmark --tasks standard

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
MEGATRON_PATH="$ROOT_DIR/backends/megatron/Megatron-LM-251125"
export PYTHONPATH="$ROOT_DIR:$MEGATRON_PATH:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

ALPHA_CONFIG_TOOL="$SCRIPT_DIR/tools/alpha_config.py"
VERIFY_TOOL="$SCRIPT_DIR/tools/verify_pipeline.py"
CONVERT_SCRIPT="$ROOT_DIR/toolkits/distributed_checkpoints_convertor/scripts/alpha/run_convert.sh"

# ---- arg parsing ----
INPUT=""
ITER="latest"
GPUS=""
OUT=""
DO_BENCHMARK=false
TASKS="standard"
SKIP_CONVERT=false
SKIP_VALIDATE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iter) ITER="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --benchmark) DO_BENCHMARK=true; shift ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --skip-convert) SKIP_CONVERT=true; shift ;;
        --skip-validate) SKIP_VALIDATE=true; shift ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        --*) echo "Unknown option: $1"; exit 1 ;;
        *) INPUT="$1"; shift ;;
    esac
done

if [ -z "$INPUT" ]; then
    echo "Usage: bash evaluate.sh <RUN_DIR|CKPT_DIR> [--iter N] [--gpus N] [--out DIR] [--benchmark] [--tasks T] [--skip-convert] [--skip-validate]"
    exit 1
fi

# ---- GPU count (EP) ----
if [ -z "$GPUS" ]; then
    GPUS=${KUBERNETES_CONTAINER_RESOURCE_GPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
fi
if [ -z "$GPUS" ] || [ "$GPUS" -lt 1 ] 2>/dev/null; then GPUS=1; fi

# ---- resolve RUN_DIR + ITER + checkpoint iter dir ----
if [[ "$INPUT" =~ /iter_([0-9]+)/?$ ]]; then
    CKPT_ITER_DIR=$(cd "$INPUT" && pwd)
    ITER_NUM="${BASH_REMATCH[1]}"
    RUN_DIR=$(dirname "$(dirname "$CKPT_ITER_DIR")")
else
    RUN_DIR=$(cd "$INPUT" && pwd)
    if [ ! -d "$RUN_DIR/checkpoints" ]; then
        echo "❌ Error: $RUN_DIR has no checkpoints/ — pass a run dir or an iter_NNNNNNN dir"
        exit 1
    fi
    if [ "$ITER" == "latest" ]; then
        ITER_NUM=$(cat "$RUN_DIR/checkpoints/latest_checkpointed_iteration.txt" | tr -d '[:space:]')
    else
        ITER_NUM="$ITER"
    fi
    CKPT_ITER_DIR="$RUN_DIR/checkpoints/iter_$(printf '%07d' $((10#${ITER_NUM})))"
fi
ITER_PADDED=$(printf '%07d' $((10#${ITER_NUM})))
HF_OUT=${OUT:-"$RUN_DIR/hfmodel_${ITER_PADDED}"}

echo "================================================================"
echo "Alpha v2 Evaluation Pipeline"
echo "================================================================"
echo "  Run dir:     $RUN_DIR"
echo "  Checkpoint:  $CKPT_ITER_DIR"
echo "  HF output:   $HF_OUT"
echo "  GPUs (EP):   $GPUS"
echo "  Benchmark:   $DO_BENCHMARK  (tasks: $TASKS)"
echo "================================================================"

if [ ! -d "$CKPT_ITER_DIR" ]; then
    echo "❌ Error: checkpoint not found: $CKPT_ITER_DIR"
    exit 1
fi

# ---- Stage 0: preflight gate ----
echo ""; echo "▶ Stage 0: preflight (checkpoint ground truth)"
python3 "$VERIFY_TOOL" preflight --from-checkpoint "$CKPT_ITER_DIR" --gpus "$GPUS"

# ---- Stage 1: convert MG→HF ----
if [ "$SKIP_CONVERT" = true ]; then
    echo ""; echo "▶ Stage 1: convert — SKIPPED (reusing $HF_OUT)"
else
    echo ""; echo "▶ Stage 1: convert MG→HF (EP=$GPUS)"
    GPUS="$GPUS" bash "$CONVERT_SCRIPT" baseline_48L "$CKPT_ITER_DIR" "$HF_OUT" true true bf16
fi

if [ ! -f "$HF_OUT/config.json" ]; then
    echo "❌ Error: $HF_OUT/config.json not found after convert"
    exit 1
fi

# ---- Stage 1.5: post-convert verification gate ----
echo ""; echo "▶ Stage 1.5: verify config.json ↔ checkpoint + tokenizer round-trip"
python3 "$VERIFY_TOOL" compare-config --from-checkpoint "$CKPT_ITER_DIR" --hf "$HF_OUT"
python3 "$VERIFY_TOOL" tokenizer-roundtrip --hf "$HF_OUT"

# ---- Stage 2: weight validation ----
if [ "$SKIP_VALIDATE" = true ]; then
    echo ""; echo "▶ Stage 2: validate — SKIPPED"
else
    echo ""; echo "▶ Stage 2: MG↔HF weight validation"
    bash "$SCRIPT_DIR/validate.sh" "$CKPT_ITER_DIR" "$HF_OUT"
fi

# ---- Stage 2.5: forward sanity gate ----
# Weight validation (Stage 2) compares tensors only and shares the converter's
# reshape, so it CANNOT catch a forward-pass mismatch (e.g. RMSNorm 1p-vs-standard)
# that copies weights correctly but interprets them wrong — which silently makes
# every benchmark random. This one-forward perplexity check closes that hole.
if [ "$SKIP_VALIDATE" = true ]; then
    echo ""; echo "▶ Stage 2.5: forward sanity — SKIPPED"
else
    echo ""; echo "▶ Stage 2.5: forward sanity (perplexity gate)"
    python3 "$SCRIPT_DIR/forward_sanity.py" --hf "$HF_OUT"
fi

# ---- Stage 3: benchmark (opt-in) ----
if [ "$DO_BENCHMARK" = true ]; then
    echo ""; echo "▶ Stage 3: benchmark (lm-eval, NUM_GPUS=$GPUS)"
    NUM_GPUS="$GPUS" bash "$SCRIPT_DIR/scripts/run_benchmarks.sh" "$HF_OUT" --tasks "$TASKS"
else
    echo ""; echo "▶ Stage 3: benchmark — SKIPPED (pass --benchmark to run)"
fi

echo ""
echo "================================================================"
echo "✅ Pipeline complete: $HF_OUT"
echo "================================================================"
