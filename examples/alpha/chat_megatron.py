#!/usr/bin/env python
# Copyright (c) 2025 Alpha Project Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Chat with Alpha Megatron checkpoint directly (without HF conversion).

This script allows interactive text generation using a Megatron checkpoint.
Note: This is a simple implementation without KV-cache optimization.

Usage:
    # With shell script (recommended):
    bash chat.sh /path/to/checkpoint baseline_48L

    # Or directly with torchrun:
    torchrun --nproc_per_node=1 chat_megatron.py \
        --load /path/to/checkpoint \
        [Megatron model args...]
"""

import argparse
import sys
import os
import torch

# Add paths for Megatron-LM-251125 and megatron_patch
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
MEGATRON_PATH = os.path.join(ROOT_DIR, "backends/megatron/Megatron-LM-251125")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, MEGATRON_PATH)

from megatron.training import get_args, print_rank_0
from megatron_patch.tokenizer import build_tokenizer
from megatron.training.initialize import initialize_megatron
from megatron.training.checkpointing import load_checkpoint
from megatron.training import get_model
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.enums import ModelType
from megatron.core.models.mamba import MambaModel

from megatron_patch.arguments import get_patch_args
from megatron_patch.model.alpha.layer_specs import get_alpha_layer_spec
from megatron_patch.model.qwen3_next.transformer_config import Qwen3NextTransformerConfig


def model_provider(pre_process=True, post_process=True):
    """Build the Alpha MambaModel."""
    args = get_args()
    print_rank_0('Building Alpha MAMBA model for inference...')

    config = core_transformer_config_from_args(args, Qwen3NextTransformerConfig)

    model = MambaModel(
        config=config,
        mamba_stack_spec=get_alpha_layer_spec(args),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        hybrid_attention_ratio=args.hybrid_attention_ratio,
        hybrid_mlp_ratio=args.hybrid_mlp_ratio,
        hybrid_override_pattern=args.hybrid_override_pattern,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=False,  # False for inference to get full vocab logits
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
    )

    return model


def add_text_generate_args(parser):
    """Add arguments for text generation."""
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--mg-checkpoint", type=str, default=None,
                       help="Path to Megatron checkpoint directory (overrides --load for checkpoint)")
    group.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (default: 0.7)")
    group.add_argument("--top-p", type=float, default=0.9,
                       help="Nucleus sampling top-p (default: 0.9)")
    group.add_argument("--top-k", type=int, default=50,
                       help="Top-k sampling (default: 50)")
    group.add_argument("--max-new-tokens", type=int, default=256,
                       help="Maximum new tokens to generate (default: 256)")
    group.add_argument("--repetition-penalty", type=float, default=1.0,
                       help="Repetition penalty (default: 1.0, no penalty)")

    return parser


def generate_text(model, tokenizer, prompt, args):
    """Generate text from prompt using simple autoregressive decoding."""
    device = torch.cuda.current_device()

    # Tokenize input
    tokens = tokenizer.tokenize(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    model.eval()
    generated_tokens = []

    with torch.no_grad():
        for step in range(args.max_new_tokens):
            seq_len = input_ids.size(1)

            # Build attention mask (causal)
            attention_mask = torch.ones(1, 1, seq_len, seq_len, device=device, dtype=torch.bool)
            attention_mask = torch.tril(attention_mask)

            # Build position ids
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            # Forward pass
            logits = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

            # Get next token logits
            next_token_logits = logits[:, -1, :].float()

            # Apply repetition penalty
            if args.repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= args.repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= args.repetition_penalty

            # Apply temperature
            next_token_logits = next_token_logits / args.temperature

            # Apply top-k filtering
            if args.top_k > 0:
                top_k_values, _ = torch.topk(next_token_logits, min(args.top_k, next_token_logits.size(-1)))
                threshold = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < threshold,
                    torch.full_like(next_token_logits, float('-inf')),
                    next_token_logits
                )

            # Apply top-p (nucleus) filtering
            if args.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > args.top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                # Scatter back to original indices
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check for EOS
            if hasattr(tokenizer, 'eos_id') and next_token.item() == tokenizer.eos_id:
                break
            if hasattr(tokenizer, 'eod') and next_token.item() == tokenizer.eod:
                break

            # Append to generated tokens and input
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Print token as it's generated (streaming)
            token_str = tokenizer.detokenize([next_token.item()])
            print(token_str, end="", flush=True)

    print()  # Newline after generation
    return tokenizer.detokenize(generated_tokens)


def main():
    """Main function."""
    # Initialize Megatron with patch args and generation args
    def extra_args_provider(parser):
        parser = get_patch_args(parser)
        parser = add_text_generate_args(parser)
        return parser

    initialize_megatron(
        extra_args_provider=extra_args_provider,
        args_defaults={
            'tokenizer_type': 'NullTokenizer',
            'no_load_rng': True,
            'no_load_optim': True,
            'use_legacy_models': False,
        }
    )

    args = get_args()

    # Determine checkpoint path: --mg-checkpoint takes priority over --load
    checkpoint_path = args.mg_checkpoint if args.mg_checkpoint else args.load

    # Build and load model
    print_rank_0(f"\nLoading model from {checkpoint_path}...")

    model = get_model(model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)

    # Load checkpoint
    if checkpoint_path:
        # Temporarily set args.load to checkpoint path for load_checkpoint
        original_load = args.load
        args.load = checkpoint_path
        load_checkpoint(model, None, None)
        args.load = original_load  # Restore original (tokenizer path)
        print_rank_0("Checkpoint loaded successfully.")

    # Unwrap model if needed
    if isinstance(model, list):
        model = model[0]
    model.eval()

    # Get tokenizer (using megatron_patch tokenizer)
    tokenizer = build_tokenizer(args)

    # Interactive chat loop (only on rank 0)
    if torch.distributed.get_rank() == 0:
        print("\n" + "=" * 60)
        print("Alpha Chat Interface")
        print("=" * 60)
        print(f"Model: {checkpoint_path}")
        print(f"Temperature: {args.temperature}")
        print(f"Top-p: {args.top_p}")
        print(f"Top-k: {args.top_k}")
        print(f"Max new tokens: {args.max_new_tokens}")
        print("-" * 60)
        print("Type 'quit', 'exit', or 'q' to end")
        print("Type 'config' to show current settings")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n[You]: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if user_input.lower() == 'config':
                    print(f"\nCurrent settings:")
                    print(f"  Temperature: {args.temperature}")
                    print(f"  Top-p: {args.top_p}")
                    print(f"  Top-k: {args.top_k}")
                    print(f"  Max new tokens: {args.max_new_tokens}")
                    print(f"  Repetition penalty: {args.repetition_penalty}")
                    continue

                print("\n[Alpha]: ", end="", flush=True)
                generate_text(model, tokenizer, user_input, args)

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
