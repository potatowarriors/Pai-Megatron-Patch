#!/usr/bin/env python
"""
Quick inference script for testing HuggingFace models with a single prompt.
Usage: python quick_inference.py --model-path <path> --prompt "Your question here"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Quick inference with HuggingFace model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the HuggingFace model directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello! Tell me about yourself.",
        help="Input prompt"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from {args.model_path}...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model.eval()

    # Prepare input
    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode and print
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response:\n{response}")


if __name__ == "__main__":
    main()
