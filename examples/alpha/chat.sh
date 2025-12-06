#!/bin/bash

# Default model path
MODEL_PATH="${1:-outputs/alpha_baseline_24L_20251117_234326/hfmodel}"

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path '$MODEL_PATH' does not exist"
    echo "Usage: bash chat.sh [model_path]"
    exit 1
fi

echo "Starting chat with model at: $MODEL_PATH"
echo ""

# Run the chat script
python chat_with_model.py \
    --model-path "$MODEL_PATH" \
    --max-new-tokens 512 \
    --temperature 0.7 \
    --top-p 0.9 \
    --top-k 50
