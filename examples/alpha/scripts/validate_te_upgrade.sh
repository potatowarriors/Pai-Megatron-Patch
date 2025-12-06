#!/bin/bash
# TransformerEngine 2.9 업그레이드 검증 스크립트
# Usage: bash validate_te_upgrade.sh

set -e

# 색상 정의
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo "=============================================="
echo "TransformerEngine 2.9 Upgrade Validation"
echo "=============================================="
echo ""

# 1. TE 버전 확인
log_info "Checking TransformerEngine version..."
TE_VERSION=$(python -c "import transformer_engine; print(transformer_engine.__version__)" 2>/dev/null || echo "NOT_INSTALLED")

if [ "$TE_VERSION" == "NOT_INSTALLED" ]; then
    log_error "TransformerEngine is not installed!"
    exit 1
fi

echo "  Version: $TE_VERSION"

if [[ "$TE_VERSION" == 2.9* ]]; then
    log_info "✓ TE 2.9.x detected - QK-Clip support available"
elif [[ "$TE_VERSION" == 2.8* ]]; then
    log_error "✗ TE 2.8.x detected - QK-Clip NOT supported"
    log_warn "Upgrade to TE 2.9+ required for QK-Clip"
    log_info "Run: bash /path/to/setup_pai_megatron_env.sh"
    exit 1
else
    log_warn "⚠ Unknown TE version: $TE_VERSION"
fi

echo ""

# 2. QK-Clip 지원 확인
log_info "Checking QK-Clip support (return_max_logit feature)..."
python -c "
from transformer_engine.pytorch.attention import DotProductAttention
import inspect

# Check if DotProductAttention.__init__ has return_max_logit parameter
sig = inspect.signature(DotProductAttention.__init__)
if 'return_max_logit' in sig.parameters:
    print('✓ return_max_logit parameter found in DotProductAttention')
else:
    print('✗ return_max_logit parameter NOT found (TE may be too old)')
    exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    log_info "✓ QK-Clip feature verified"
else
    log_error "✗ QK-Clip feature NOT available"
    exit 1
fi

echo ""

# 3. Muon Optimizer 확인
log_info "Checking Muon optimizer availability..."
python -c "
from emerging_optimizers.orthogonalized_optimizers import OrthogonalizedOptimizer
print('✓ Emerging-Optimizers (Muon) import successful')
" 2>/dev/null

if [ $? -eq 0 ]; then
    log_info "✓ Muon optimizer available"
else
    log_error "✗ Muon optimizer NOT available"
    log_warn "Install: pip install emerging-optimizers"
    exit 1
fi

echo ""

# 4. GatedDeltaNet 확인 (Alpha-specific)
log_info "Checking GatedDeltaNet (Alpha model architecture)..."
python -c "
import sys
sys.path.insert(0, '/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch')
from megatron_patch.model.qwen3_next.gated_deltanet import GatedDeltaNet
print('✓ GatedDeltaNet import successful')
" 2>/dev/null

if [ $? -eq 0 ]; then
    log_info "✓ GatedDeltaNet available"
else
    log_warn "⚠ GatedDeltaNet import failed (check PYTHONPATH)"
fi

echo ""

# 5. Megatron-LM-251125 호환성 확인
log_info "Checking Megatron-LM-251125 compatibility..."
python -c "
import sys
sys.path.insert(0, '/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/backends/megatron/Megatron-LM-251125')
from megatron.core.optimizer.muon import Muon
from megatron.core.optimizer.qk_clip import clip_qk
print('✓ Megatron-LM-251125 Muon and QK-Clip modules import successful')
" 2>/dev/null

if [ $? -eq 0 ]; then
    log_info "✓ Megatron-LM-251125 compatible"
else
    log_warn "⚠ Megatron-LM-251125 modules import failed (check backend)"
fi

echo ""

# 6. 종합 결과
echo "=============================================="
echo "Validation Summary"
echo "=============================================="

if [[ "$TE_VERSION" == 2.9* ]]; then
    log_info "✓ TransformerEngine: $TE_VERSION (QK-Clip ready)"
    log_info "✓ All core components verified"
    log_info ""
    log_info "You can now use QK-Clip in training configs:"
    echo "  qk_clip: true"
    echo "  qk_clip_alpha: 0.5"
    echo "  qk_clip_threshold: 100"
    exit 0
else
    log_error "✗ TransformerEngine upgrade required"
    log_info "Current: $TE_VERSION"
    log_info "Required: >= 2.9.0"
    exit 1
fi
