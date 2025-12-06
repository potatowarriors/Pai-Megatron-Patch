#!/usr/bin/env python
"""
Chat with Megatron checkpoint directly (without HF conversion).
Usage: python chat_megatron.py --checkpoint-dir <path_to_megatron_checkpoint>
"""

import argparse
import sys
import os
import torch

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "../..")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "backends/megatron/Megatron-LM-250908"))

from megatron.training import get_args
from megatron.training import get_tokenizer
from megatron.training.initialize import initialize_megatron
from megatron.core import mpu
from megatron_patch.model.qwen3_next.layer_specs import get_qwen3_next_layer_spec
from megatron.core.models.mamba import MambaModel


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()

    layer_spec = get_qwen3_next_layer_spec(args)

    model = MambaModel(
        config=args,
        transformer_layer_spec=layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        hybrid_attention_ratio=args.hybrid_attention_ratio,
        hybrid_mlp_ratio=args.hybrid_mlp_ratio,
        hybrid_override_pattern=args.hybrid_override_pattern
    )

    return model


def add_text_generate_args(parser):
    """Add arguments for text generation."""
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Path to Megatron checkpoint directory")
    group.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    group.add_argument("--top-p", type=float, default=0.9,
                       help="Nucleus sampling top-p")
    group.add_argument("--top-k", type=int, default=50,
                       help="Top-k sampling")
    group.add_argument("--max-new-tokens", type=int, default=512,
                       help="Maximum new tokens to generate")

    return parser


def generate_text(model, tokenizer, prompt, args):
    """Generate text from prompt."""
    # Tokenize input
    tokens = tokenizer.tokenize(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=torch.cuda.current_device())

    model.eval()

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            # Forward pass
            logits = model(input_ids)

            # Get next token logits
            next_token_logits = logits[:, -1, :] / args.temperature

            # Apply top-k filtering
            if args.top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, args.top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if args.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > args.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = float('-inf')

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check for EOS
            if next_token.item() == tokenizer.eos_id:
                break

            # Append to input
            input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode output
    output_tokens = input_ids[0].tolist()
    response = tokenizer.detokenize(output_tokens[len(tokens):])

    return response


def main():
    """Main function."""
    # Initialize Megatron
    initialize_megatron(
        extra_args_provider=add_text_generate_args,
        args_defaults={'tokenizer_type': 'Qwen2Tokenizer',
                       'no_load_rng': True,
                       'no_load_optim': True}
    )

    args = get_args()

    # Load checkpoint path
    if os.path.isdir(args.checkpoint_dir):
        checkpoint_path = args.checkpoint_dir
    else:
        print(f"Error: {args.checkpoint_dir} is not a directory")
        return

    # Load model
    print(f"Loading model from {checkpoint_path}...")

    # Build model
    model = model_provider()

    # Load checkpoint
    from megatron.training.checkpointing import load_checkpoint
    iteration = load_checkpoint(model, None, None)

    print(f"Loaded checkpoint from iteration {iteration}")

    # Get tokenizer
    tokenizer = get_tokenizer()

    if torch.distributed.get_rank() == 0:
        print("\n" + "=" * 50)
        print("Chat Interface - Type 'quit' or 'exit' to end")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n[You]: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                print("\n[Model]: ", end="", flush=True)

                response = generate_text(model, tokenizer, user_input, args)
                print(response)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing...")


if __name__ == "__main__":
    main()
