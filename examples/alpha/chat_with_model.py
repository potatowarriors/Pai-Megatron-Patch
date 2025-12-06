#!/usr/bin/env python
"""
Simple chat interface for HuggingFace models.
Usage: python chat_with_model.py --model-path <path_to_hf_model>
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with your trained HuggingFace model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the HuggingFace model directory"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda/cpu)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from {args.model_path}...")
    print(f"Using device: {args.device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device == "cuda" else None
    )

    if args.device == "cpu":
        model = model.to(args.device)

    model.eval()
    print("\nModel loaded successfully!")
    print("=" * 50)
    print("Chat Interface - Type 'quit' or 'exit' to end")
    print("=" * 50)

    conversation_history = []

    while True:
        try:
            # Get user input
            user_input = input("\n[You]: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Add user message to history
            conversation_history.append({"role": "user", "content": user_input})

            # Format conversation using chat template if available
            try:
                # Try using chat template
                text = tokenizer.apply_chat_template(
                    conversation_history,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                # Fallback to simple format
                print(f"Note: Chat template not available, using simple format")
                text = user_input

            # Tokenize
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            # Generate response
            print("\n[Model]: ", end="", flush=True)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            # Decode response
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(response)

            # Add assistant response to history
            conversation_history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Continuing...")


if __name__ == "__main__":
    main()
