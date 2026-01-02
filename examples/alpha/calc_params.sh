#!/bin/bash
# Quick parameter calculation script for Alpha models

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default config
CONFIG="${1:-configs/model/baseline_48L.yaml}"
DETAILED="${2:-}"

# Run calculator
if [ "$DETAILED" = "--detailed" ] || [ "$DETAILED" = "-d" ]; then
    python "${SCRIPT_DIR}/calculate_parameters.py" --config "$CONFIG" --detailed
else
    python "${SCRIPT_DIR}/calculate_parameters.py" --config "$CONFIG"
fi
