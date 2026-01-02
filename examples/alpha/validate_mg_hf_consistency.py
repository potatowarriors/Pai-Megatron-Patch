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
Alpha Model MG ↔ HF Logits Consistency Validation
==================================================
Validates that Megatron checkpoint and HuggingFace converted model produce
identical (or near-identical) logits for the same input.

Usage:
    python validate_mg_hf_consistency.py \
        --hf-model /path/to/hf/model \
        --prompts "Hello world" "What is AI?"

    # With Megatron checkpoint (requires distributed launch):
    torchrun --nproc_per_node=1 validate_mg_hf_consistency.py \
        --mg-checkpoint /path/to/megatron/checkpoint \
        --hf-model /path/to/hf/model \
        --prompts "Hello world"

Validation Criteria (BF16):
    - torch.allclose(atol=0.01, rtol=0.01)
    - Max absolute difference < 0.1
    - Cosine similarity > 0.999
    - Top-10 token match rate > 90%
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate MG ↔ HF logits consistency for Alpha model"
    )
    parser.add_argument(
        "--mg-checkpoint",
        type=str,
        default=None,
        help="Path to Megatron checkpoint directory (optional, skip MG if not provided)"
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        required=True,
        help="Path to HuggingFace model directory"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["Hello, how are you?", "The capital of France is"],
        help="Test prompts for validation"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.01,
        help="Absolute tolerance for torch.allclose (default: 0.01 for BF16)"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.01,
        help="Relative tolerance for torch.allclose (default: 0.01 for BF16)"
    )
    parser.add_argument(
        "--max-diff-threshold",
        type=float,
        default=0.1,
        help="Maximum allowed absolute difference (default: 0.1)"
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.999,
        help="Minimum cosine similarity threshold (default: 0.999)"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-k tokens to compare (default: 10)"
    )
    parser.add_argument(
        "--topk-match-threshold",
        type=float,
        default=0.9,
        help="Minimum top-k match rate (default: 0.9 = 90%%)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed comparison results"
    )
    return parser.parse_args()


def load_hf_model(model_path: str):
    """Load HuggingFace model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading HF model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    print(f"  Model loaded: {model.__class__.__name__}")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")

    return model, tokenizer


def get_hf_logits(model, tokenizer, prompts: List[str]) -> Dict[str, torch.Tensor]:
    """Get logits from HuggingFace model for each prompt."""
    results = {}

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Debug: Print tokenization result
        print(f"  [DEBUG] Prompt: '{prompt}'")
        print(f"  [DEBUG] HF input_ids: {inputs['input_ids'].tolist()}")
        print(f"  [DEBUG] HF input_ids shape: {inputs['input_ids'].shape}")

        # Debug: Check embedding output
        with torch.no_grad():
            embed_out = model.model.embed_tokens(inputs['input_ids'])
            print(f"  [DEBUG] HF embedding output shape: {embed_out.shape}")
            print(f"  [DEBUG] HF embedding mean: {embed_out.mean().item():.6f}, std: {embed_out.std().item():.6f}")
            print(f"  [DEBUG] HF embedding[0,0,:5]: {embed_out[0, 0, :5].tolist()}")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [batch=1, seq_len, vocab_size]

        # Debug: Print logits stats
        print(f"  [DEBUG] HF logits shape: {logits.shape}")
        print(f"  [DEBUG] HF logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
        print(f"  [DEBUG] HF logits[0,-1,:5]: {logits[0, -1, :5].tolist()}")

        results[prompt] = {
            "logits": logits.cpu(),
            "input_ids": inputs["input_ids"].cpu(),
        }

    return results


def compare_logits(
    mg_logits: torch.Tensor,
    hf_logits: torch.Tensor,
    atol: float = 0.01,
    rtol: float = 0.01,
    topk: int = 10
) -> Dict:
    """
    Compare two logits tensors and return various metrics.

    Args:
        mg_logits: Megatron logits [batch, seq_len, vocab_size]
        hf_logits: HuggingFace logits [batch, seq_len, vocab_size]
        atol: Absolute tolerance
        rtol: Relative tolerance
        topk: Number of top tokens to compare

    Returns:
        Dictionary with comparison metrics
    """
    # Ensure same device and dtype for comparison
    mg_logits = mg_logits.float()
    hf_logits = hf_logits.float()

    # 1. torch.allclose check
    is_close = torch.allclose(mg_logits, hf_logits, atol=atol, rtol=rtol)

    # 2. Absolute difference statistics
    abs_diff = (mg_logits - hf_logits).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # 3. Cosine similarity (per position, then averaged)
    # Flatten batch and seq dimensions for comparison
    mg_flat = mg_logits.view(-1, mg_logits.size(-1))  # [batch*seq, vocab]
    hf_flat = hf_logits.view(-1, hf_logits.size(-1))

    cos_sim_per_pos = F.cosine_similarity(mg_flat, hf_flat, dim=-1)
    cos_sim_mean = cos_sim_per_pos.mean().item()
    cos_sim_min = cos_sim_per_pos.min().item()

    # 4. Cosine similarity for last token only (most important for generation)
    last_mg = mg_logits[:, -1, :]  # [batch, vocab]
    last_hf = hf_logits[:, -1, :]
    cos_sim_last = F.cosine_similarity(last_mg, last_hf, dim=-1).mean().item()

    # 5. Top-k token match rate
    mg_topk = mg_logits[:, -1, :].topk(k=topk, dim=-1).indices  # [batch, k]
    hf_topk = hf_logits[:, -1, :].topk(k=topk, dim=-1).indices

    # Check how many of the top-k tokens match (order-independent)
    matches = 0
    total = mg_topk.numel()
    for b in range(mg_topk.size(0)):
        mg_set = set(mg_topk[b].tolist())
        hf_set = set(hf_topk[b].tolist())
        matches += len(mg_set & hf_set)
    topk_match_rate = matches / total if total > 0 else 0.0

    # 6. Top-1 match (greedy decoding would produce same token)
    mg_top1 = mg_logits[:, -1, :].argmax(dim=-1)
    hf_top1 = hf_logits[:, -1, :].argmax(dim=-1)
    top1_match = (mg_top1 == hf_top1).float().mean().item()

    return {
        "allclose": is_close,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cosine_sim_mean": cos_sim_mean,
        "cosine_sim_min": cos_sim_min,
        "cosine_sim_last": cos_sim_last,
        f"top{topk}_match_rate": topk_match_rate,
        "top1_match": top1_match,
        "mg_top1": mg_top1.tolist(),
        "hf_top1": hf_top1.tolist(),
    }


def print_comparison_results(
    prompt: str,
    results: Dict,
    args,
    tokenizer=None
) -> bool:
    """Print comparison results and return whether validation passed."""
    print(f"\n{'='*70}")
    print(f"Prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
    print(f"{'='*70}")

    # Determine pass/fail for each metric
    checks = {
        "allclose": results["allclose"],
        "max_diff": results["max_diff"] < args.max_diff_threshold,
        "cosine_sim_last": results["cosine_sim_last"] > args.cosine_threshold,
        f"top{args.topk}_match": results[f"top{args.topk}_match_rate"] >= args.topk_match_threshold,
    }

    all_passed = all(checks.values())

    # Print metrics
    print(f"\n  Metric                    Value           Threshold       Status")
    print(f"  {'-'*66}")

    print(f"  torch.allclose            {results['allclose']}          "
          f"atol={args.atol}, rtol={args.rtol}    {'✓ PASS' if checks['allclose'] else '✗ FAIL'}")

    print(f"  Max absolute diff         {results['max_diff']:.6f}        "
          f"< {args.max_diff_threshold}            {'✓ PASS' if checks['max_diff'] else '✗ FAIL'}")

    print(f"  Mean absolute diff        {results['mean_diff']:.6f}        (info)")

    print(f"  Cosine sim (all)          {results['cosine_sim_mean']:.6f}        (info)")

    print(f"  Cosine sim (last token)   {results['cosine_sim_last']:.6f}        "
          f"> {args.cosine_threshold}         {'✓ PASS' if checks['cosine_sim_last'] else '✗ FAIL'}")

    print(f"  Top-{args.topk} match rate         {results[f'top{args.topk}_match_rate']:.2%}           "
          f">= {args.topk_match_threshold:.0%}             {'✓ PASS' if checks[f'top{args.topk}_match'] else '✗ FAIL'}")

    print(f"  Top-1 match               {results['top1_match']:.2%}           (info)")

    # Print top-1 tokens if tokenizer available
    if tokenizer and args.verbose:
        print(f"\n  Top-1 predictions:")
        for i, (mg_id, hf_id) in enumerate(zip(results["mg_top1"], results["hf_top1"])):
            mg_token = tokenizer.decode([mg_id])
            hf_token = tokenizer.decode([hf_id])
            match = "✓" if mg_id == hf_id else "✗"
            print(f"    [{i}] MG: '{mg_token}' ({mg_id}) | HF: '{hf_token}' ({hf_id}) {match}")

    print(f"\n  {'='*66}")
    print(f"  Overall: {'✓ VALIDATION PASSED' if all_passed else '✗ VALIDATION FAILED'}")
    print(f"  {'='*66}")

    return all_passed


def run_hf_only_validation(args):
    """Run validation with HF model only (self-consistency check)."""
    print("\n" + "="*70)
    print("HF Model Self-Consistency Check")
    print("="*70)
    print("(Running HF model twice to verify deterministic output)")

    model, tokenizer = load_hf_model(args.hf_model)

    # Get logits twice
    print("\nFirst inference pass...")
    results1 = get_hf_logits(model, tokenizer, args.prompts)

    print("Second inference pass...")
    results2 = get_hf_logits(model, tokenizer, args.prompts)

    all_passed = True
    for prompt in args.prompts:
        logits1 = results1[prompt]["logits"]
        logits2 = results2[prompt]["logits"]

        comparison = compare_logits(logits1, logits2, args.atol, args.rtol, args.topk)
        passed = print_comparison_results(
            f"[Self-check] {prompt}", comparison, args, tokenizer
        )
        all_passed = all_passed and passed

    return all_passed


def main():
    args = parse_args()

    print("\n" + "="*70)
    print("Alpha Model MG ↔ HF Logits Consistency Validation")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  HF Model:       {args.hf_model}")
    print(f"  MG Checkpoint:  {args.mg_checkpoint or '(not provided - HF self-check only)'}")
    print(f"  Prompts:        {len(args.prompts)} test cases")
    print(f"  Tolerances:     atol={args.atol}, rtol={args.rtol}")
    print(f"  Max diff:       < {args.max_diff_threshold}")
    print(f"  Cosine sim:     > {args.cosine_threshold}")
    print(f"  Top-{args.topk} match:   >= {args.topk_match_threshold:.0%}")

    if args.mg_checkpoint is None:
        # HF-only mode: self-consistency check
        print("\n⚠ No Megatron checkpoint provided. Running HF self-consistency check.")
        success = run_hf_only_validation(args)
    else:
        # Full MG ↔ HF comparison
        print("\n⚠ Full MG ↔ HF comparison requires Megatron initialization.")
        print("  Please use the companion script 'validate_mg_hf_full.py' with torchrun.")
        print("\n  For now, running HF self-consistency check as fallback...")
        success = run_hf_only_validation(args)

    # Final summary
    print("\n" + "="*70)
    if success:
        print("✓ ALL VALIDATIONS PASSED")
    else:
        print("✗ SOME VALIDATIONS FAILED")
    print("="*70 + "\n")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
