#!/bin/bash
#==============================================================================
# WANDB Integration Test
#==============================================================================
# This script tests WANDB integration by parsing configs and checking arguments
#==============================================================================

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ALPHA_DIR=$(dirname "$SCRIPT_DIR")

cd "$ALPHA_DIR"

echo "=========================================="
echo "WANDB Integration Test"
echo "=========================================="
echo ""

# Source WANDB setup
echo "1️⃣  Setting up WANDB environment..."
source ./scripts/setup_wandb.sh
echo ""

# Load YAML parser
echo "2️⃣  Loading YAML parser..."
source ./scripts/load_config.sh
echo ""

# Test YAML parsing
echo "3️⃣  Testing WANDB configuration parsing..."
TRAINING_CONFIG_FILE="./configs/training/pretrain.yaml"

WANDB_ENABLED=$(yaml_get $TRAINING_CONFIG_FILE "training.wandb.enabled")
WANDB_PROJECT=$(yaml_get $TRAINING_CONFIG_FILE "training.wandb.project")
WANDB_ENTITY=$(yaml_get $TRAINING_CONFIG_FILE "training.wandb.entity")
WANDB_SAVE_DIR=$(yaml_get $TRAINING_CONFIG_FILE "training.wandb.save_dir")

echo "  - enabled: ${WANDB_ENABLED}"
echo "  - project: ${WANDB_PROJECT}"
echo "  - entity: ${WANDB_ENTITY}"
echo "  - save_dir: ${WANDB_SAVE_DIR}"
echo ""

# Test WANDB Python API
echo "4️⃣  Testing WANDB Python API..."
python - <<EOF
import wandb
import os

# Check environment
api_key = os.environ.get('WANDB_API_KEY')
if api_key:
    print(f"  ✅ WANDB_API_KEY is set (length: {len(api_key)})")
else:
    print("  ❌ WANDB_API_KEY is NOT set")
    exit(1)

# Check login
try:
    wandb.login(key=api_key)
    print("  ✅ WANDB login successful")
except Exception as e:
    print(f"  ❌ WANDB login failed: {e}")
    exit(1)

# Test initialization (dry run mode)
try:
    run = wandb.init(
        project="alpha-pretraining-test",
        name="wandb-integration-test",
        mode="offline",  # Offline mode for testing
        config={
            "test": True,
            "model": "alpha-24L",
            "batch_size": 2
        }
    )
    print("  ✅ WANDB init successful (offline mode)")

    # Log a test metric
    wandb.log({"test_metric": 42})
    print("  ✅ WANDB log successful")

    # Finish run
    wandb.finish()
    print("  ✅ WANDB finish successful")
except Exception as e:
    print(f"  ❌ WANDB init failed: {e}")
    exit(1)

print("")
print("🎉 All WANDB tests passed!")
EOF

echo ""
echo "=========================================="
echo "✅ WANDB Integration Test Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review the test results above"
echo "  2. Check WANDB dashboard: https://wandb.ai/kide004/alpha-pretraining-test"
echo "  3. Run actual training with WANDB enabled"
echo ""
