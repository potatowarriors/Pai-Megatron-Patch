#!/bin/bash
# SGLang Backend Setup Script
#
# Applies Alpha-specific patches to the SGLang submodule and installs
# the model adapter. Safe to run multiple times (idempotent).
#
# Usage:
#   bash backends/sglang/setup.sh [SGLANG_VERSION]
#   bash backends/sglang/setup.sh              # defaults to v0.5.2
#   bash backends/sglang/setup.sh v0.5.2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SGLANG_VERSION="${1:-v0.5.2}"
SGLANG_DIR="${SCRIPT_DIR}/sglang-${SGLANG_VERSION}"
PATCHES_DIR="${SCRIPT_DIR}/patches"
ALPHA_MODEL_ADAPTER="${REPO_ROOT}/examples/alpha/sglang/sglang_alpha_model.py"

echo "═══════════════════════════════════════════════"
echo "  SGLang Backend Setup (${SGLANG_VERSION})"
echo "═══════════════════════════════════════════════"

# ─── 1. Verify submodule ───────────────────────────────────
if [ ! -d "$SGLANG_DIR/python/sglang" ]; then
    echo "[1/4] SGLang submodule not found. Initializing..."
    cd "$REPO_ROOT"
    git submodule update --init "backends/sglang/sglang-${SGLANG_VERSION}"
    cd "$SCRIPT_DIR"
    if [ ! -d "$SGLANG_DIR/python/sglang" ]; then
        echo "ERROR: Failed to initialize submodule at $SGLANG_DIR"
        exit 1
    fi
else
    echo "[1/4] SGLang submodule OK: $SGLANG_DIR"
fi

# ─── 2. Apply patches ─────────────────────────────────────
echo "[2/4] Checking patches..."
if [ -d "$PATCHES_DIR" ]; then
    cd "$SGLANG_DIR"
    applied=0
    skipped=0
    for patch_file in "$PATCHES_DIR"/*.patch; do
        [ -f "$patch_file" ] || continue
        patch_name=$(basename "$patch_file")
        if git apply --check "$patch_file" 2>/dev/null; then
            git apply "$patch_file"
            echo "  Applied: $patch_name"
            applied=$((applied + 1))
        else
            echo "  Skipped (already applied): $patch_name"
            skipped=$((skipped + 1))
        fi
    done
    cd "$SCRIPT_DIR"
    echo "  Total: ${applied} applied, ${skipped} skipped"
else
    echo "  No patches directory found"
fi

# ─── 3. Install Alpha model adapter ───────────────────────
echo "[3/4] Installing Alpha model adapter..."
MODELS_DIR="${SGLANG_DIR}/python/sglang/srt/models"
if [ -f "$ALPHA_MODEL_ADAPTER" ]; then
    cp "$ALPHA_MODEL_ADAPTER" "${MODELS_DIR}/alpha.py"
    echo "  Installed: sglang/srt/models/alpha.py"
else
    echo "  WARNING: Model adapter not found: $ALPHA_MODEL_ADAPTER"
    echo "  Native mode (Option B) will not work without it."
fi

# ─── 4. Verify compiled dependencies ──────────────────────
echo "[4/4] Checking compiled dependencies..."
SGLANG_VENV="${SCRIPT_DIR}/.venv"

if [ -f "$SGLANG_VENV/bin/activate" ]; then
    echo "  Checking in SGLang venv: $SGLANG_VENV"
    source "$SGLANG_VENV/bin/activate"
    missing=()
    python -c "import sgl_kernel" 2>/dev/null || missing+=("sgl-kernel")
    python -c "import flashinfer" 2>/dev/null || missing+=("flashinfer")
    deactivate
else
    echo "  SGLang venv not found, checking system Python..."
    missing=()
    python -c "import sgl_kernel" 2>/dev/null || missing+=("sgl-kernel")
    python -c "import flashinfer" 2>/dev/null || missing+=("flashinfer")
fi

if [ ${#missing[@]} -eq 0 ]; then
    echo "  All compiled dependencies OK"
else
    echo "  WARNING: Missing compiled packages: ${missing[*]}"
    echo "  Run setup_pai_megatron_env_A100.sh Step 14 to create SGLang venv,"
    echo "  or install manually: pip install ${missing[*]}"
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "  Setup complete!"
echo "═══════════════════════════════════════════════"
echo "  SGLang source: $SGLANG_DIR/python/"
if [ -f "$SGLANG_VENV/bin/activate" ]; then
    echo "  SGLang venv:   $SGLANG_VENV"
fi
echo ""
echo "  To deploy Alpha model:"
echo "    bash examples/alpha/sglang/deploy.sh /path/to/alpha-hf --mode native"
