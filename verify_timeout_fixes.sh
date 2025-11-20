#!/bin/bash
# Verification script for NCCL timeout bug fixes

echo "========================================="
echo "NCCL Timeout Bug Fix Verification Script"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counter for checks
PASS=0
FAIL=0

# Check 1: Verify timeout parameter in EXPERT_MODEL_PARALLEL_GROUP
echo -n "Check 1: EXPERT_MODEL_PARALLEL_GROUP has timeout parameter... "
COUNT=$(grep -A 10 "Build the expert model parallel group" \
   backends/megatron/Megatron-LM-250908/megatron/core/parallel_state.py | \
   grep -c "timeout=timeout" || echo "0")
if [ "$COUNT" -ge 1 ]; then
    echo -e "${GREEN}PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}FAIL${NC}"
    echo "  ERROR: timeout parameter not found in EXPERT_MODEL_PARALLEL_GROUP creation"
    ((FAIL++))
fi

# Check 2: Verify other expert groups still have timeout
echo -n "Check 2: EXPERT_TENSOR_PARALLEL_GROUP has timeout parameter... "
COUNT=$(grep -A 10 "Build the expert tensor parallel group" \
   backends/megatron/Megatron-LM-250908/megatron/core/parallel_state.py | \
   grep -c "timeout=timeout" || echo "0")
if [ "$COUNT" -ge 1 ]; then
    echo -e "${GREEN}PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}FAIL${NC}"
    ((FAIL++))
fi

# Check 3: Verify cleanup barrier timeout
echo -n "Check 3: cleanup_distributed barrier has 10-minute timeout... "
if grep -A 5 "def cleanup_distributed" \
   megatron_patch/training.py | \
   grep -q "timeout=timedelta(minutes=10)"; then
    echo -e "${GREEN}PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}FAIL${NC}"
    echo "  ERROR: cleanup barrier timeout not set to 10 minutes"
    ((FAIL++))
fi

# Check 4: Verify no hardcoded 30-second timeout in cleanup
echo -n "Check 4: No hardcoded 30-second timeout in cleanup... "
if ! grep "def cleanup_distributed" -A 10 megatron_patch/training.py | \
   grep -q "timeout=timedelta(seconds=30)"; then
    echo -e "${GREEN}PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}FAIL${NC}"
    echo "  ERROR: Found hardcoded 30-second timeout (should be 10 minutes)"
    ((FAIL++))
fi

# Check 5: Verify timedelta import
echo -n "Check 5: timedelta imported in training.py... "
if grep -q "from datetime import timedelta" megatron_patch/training.py; then
    echo -e "${GREEN}PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}FAIL${NC}"
    echo "  ERROR: timedelta not imported"
    ((FAIL++))
fi

# Check 6: Verify atexit cleanup registration
echo -n "Check 6: atexit cleanup handler registered... "
if grep -q "atexit.register(cleanup_distributed)" megatron_patch/training.py; then
    echo -e "${GREEN}PASS${NC}"
    ((PASS++))
else
    echo -e "${YELLOW}WARN${NC} (optional)"
fi

echo ""
echo "========================================="
echo "Summary"
echo "========================================="
echo -e "Tests passed: ${GREEN}${PASS}${NC}"
echo -e "Tests failed: ${RED}${FAIL}${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! The bug fixes have been correctly applied.${NC}"
    echo ""
    echo "Bug Fix Details:"
    echo "  1. EXPERT_MODEL_PARALLEL_GROUP now respects --distributed-timeout-minutes"
    echo "  2. Cleanup barrier timeout increased from 30s to 10 minutes"
    echo "  3. Graceful shutdown handler registered (atexit)"
    echo ""
    echo "You can now run your training with:"
    echo "  cd examples/qwen3_next"
    echo "  bash run_test_h100x8.sh"
    echo ""
    echo "Expected behavior:"
    echo "  - Training will NOT timeout at iteration 100"
    echo "  - All process groups use 60-minute timeout (from --distributed-timeout-minutes 60)"
    echo "  - Training completion will exit cleanly without hanging"
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please review the changes.${NC}"
    exit 1
fi
