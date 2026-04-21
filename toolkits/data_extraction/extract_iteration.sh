#!/bin/bash
#
# Convenience wrapper for extract_training_samples.py
#
# Usage:
#   ./extract_iteration.sh 324065                    # Extract iteration 324065
#   ./extract_iteration.sh 324065 --analyze          # With anomaly analysis
#   ./extract_iteration.sh 324065 --output-dir ./samples/  # Save individual files
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Default values
ITERATION="${1:-324065}"
shift 2>/dev/null || true

# Create output directory
OUTPUT_DIR="${SCRIPT_DIR}/outputs/iter_${ITERATION}"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Training Sample Extraction"
echo "=============================================="
echo "Iteration: $ITERATION"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run extraction
python3 "$SCRIPT_DIR/extract_training_samples.py" \
    --iteration "$ITERATION" \
    --output "$OUTPUT_DIR/samples.json" \
    --output-text "$OUTPUT_DIR/samples_readable.txt" \
    --analyze \
    "$@"

echo ""
echo "=============================================="
echo "Output Files"
echo "=============================================="
echo "JSON: $OUTPUT_DIR/samples.json"
echo "Text: $OUTPUT_DIR/samples_readable.txt"
echo ""
echo "Quick inspection:"
echo "  python -c \"import json; d=json.load(open('$OUTPUT_DIR/samples.json')); print(json.dumps(d['statistics'], indent=2))\""
echo ""
