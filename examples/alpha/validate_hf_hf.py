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
Alpha HF Model ↔ HF Model Validation
====================================
Validates that two HuggingFace models produce identical results.

Use Cases:
    1. Compare old HF model (qwen3_next) vs new HF model (alpha)
    2. Verify model conversion produces identical weights
    3. Validate model behavior consistency

Validation Methods:
    1. Weight comparison (fastest, most reliable)
    2. Forward pass comparison (validates computation)
    3. Generation comparison (optional, probabilistic)

Usage:
    python validate_hf_hf.py --model1 /path/to/hfmodel --model2 /path/to/hfmodel_newconv

    # With forward pass validation
    python validate_hf_hf.py --model1 /path/to/hfmodel --model2 /path/to/hfmodel_newconv --forward-pass

    # Verbose output
    python validate_hf_hf.py --model1 /path/to/hfmodel --model2 /path/to/hfmodel_newconv --verbose
"""

import argparse
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class WeightComparison:
    """Result of comparing a single weight tensor."""
    name1: str
    name2: str
    matched: bool
    max_diff: float
    mean_diff: float
    cosine_sim: float
    shape1: Tuple
    shape2: Tuple
    note: str = ""


@dataclass
class ForwardPassResult:
    """Result of forward pass comparison."""
    seq_length: int
    logits_matched: bool
    max_diff: float
    mean_diff: float
    cosine_sim: float
    top1_match_rate: float
    top5_match_rate: float


@dataclass
class ValidationResult:
    """Overall validation result."""
    weight_comparisons: List[WeightComparison] = field(default_factory=list)
    forward_pass_results: List[ForwardPassResult] = field(default_factory=list)
    model1_only_weights: List[str] = field(default_factory=list)
    model2_only_weights: List[str] = field(default_factory=list)

    @property
    def weights_matched(self) -> bool:
        return all(c.matched for c in self.weight_comparisons)

    @property
    def forward_matched(self) -> bool:
        if not self.forward_pass_results:
            return True
        return all(r.logits_matched for r in self.forward_pass_results)


# ==============================================================================
# Utility Functions
# ==============================================================================

def compute_metrics(t1: torch.Tensor, t2: torch.Tensor) -> Tuple[float, float, float]:
    """Compute comparison metrics between two tensors.

    Returns:
        (max_diff, mean_diff, cosine_sim)
    """
    t1_f = t1.float().flatten()
    t2_f = t2.float().flatten()

    diff = (t1_f - t2_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Cosine similarity
    cos_sim = F.cosine_similarity(t1_f.unsqueeze(0), t2_f.unsqueeze(0)).item()

    return max_diff, mean_diff, cos_sim


def compare_tensors(
    t1: torch.Tensor,
    t2: torch.Tensor,
    name1: str,
    name2: str,
    threshold: float = 0.01,
    note: str = ""
) -> WeightComparison:
    """Compare two tensors and return comparison result."""
    # Move to same device
    if t1.device != t2.device:
        t2 = t2.to(t1.device)

    # Handle shape mismatch
    if t1.shape != t2.shape:
        return WeightComparison(
            name1=name1,
            name2=name2,
            matched=False,
            max_diff=float('inf'),
            mean_diff=float('inf'),
            cosine_sim=0.0,
            shape1=tuple(t1.shape),
            shape2=tuple(t2.shape),
            note=f"Shape mismatch: {t1.shape} vs {t2.shape}"
        )

    max_diff, mean_diff, cos_sim = compute_metrics(t1, t2)
    matched = max_diff < threshold and cos_sim > 0.999

    return WeightComparison(
        name1=name1,
        name2=name2,
        matched=matched,
        max_diff=max_diff,
        mean_diff=mean_diff,
        cosine_sim=cos_sim,
        shape1=tuple(t1.shape),
        shape2=tuple(t2.shape),
        note=note
    )


def print_comparison(cmp: WeightComparison, verbose: bool = False):
    """Print a single weight comparison result."""
    status = "✓" if cmp.matched else "✗"

    if cmp.matched and not verbose:
        return  # Skip printing matched weights in non-verbose mode

    print(f"  {status} {cmp.name1}")
    if cmp.name1 != cmp.name2:
        print(f"      ↔ {cmp.name2}")
    print(f"      Shape: {cmp.shape1} vs {cmp.shape2}")
    print(f"      max_diff={cmp.max_diff:.6f}, mean_diff={cmp.mean_diff:.6f}, cos_sim={cmp.cosine_sim:.6f}")
    if cmp.note:
        print(f"      Note: {cmp.note}")


# ==============================================================================
# Model Loading
# ==============================================================================

def load_hf_model(model_path: str, device: str = "cuda:0"):
    """Load HuggingFace model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    print(f"Loading model from {model_path}...")

    # Load config to show model type
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"  Model type: {config.model_type}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    return model, tokenizer, config


# ==============================================================================
# Weight Comparison
# ==============================================================================

def get_weight_mapping(model1_keys: List[str], model2_keys: List[str]) -> Dict[str, str]:
    """Create mapping between model1 and model2 weight names.

    Handles naming differences between qwen3_next and alpha models.
    """
    mapping = {}

    # Common patterns to map
    # qwen3_next: model.layers.X.linear_attn.* -> alpha: model.layers.X.linear_attn.*
    # These should be identical for most cases

    set1 = set(model1_keys)
    set2 = set(model2_keys)

    # First pass: exact matches
    common = set1 & set2
    for key in common:
        mapping[key] = key

    # Remaining keys
    remaining1 = set1 - common
    remaining2 = set2 - common

    # For now, assume no complex mapping needed
    # If there are remaining keys, they'll be reported as model-only

    return mapping, list(remaining1), list(remaining2)


def compare_state_dicts(
    model1,
    model2,
    threshold: float = 0.01,
    verbose: bool = False
) -> ValidationResult:
    """Compare state_dict of two models."""
    result = ValidationResult()

    # Collect state dicts
    state1 = {k: v for k, v in model1.named_parameters()}
    state1.update({k: v for k, v in model1.named_buffers()})

    state2 = {k: v for k, v in model2.named_parameters()}
    state2.update({k: v for k, v in model2.named_buffers()})

    print(f"\nModel 1 weights: {len(state1)}")
    print(f"Model 2 weights: {len(state2)}")

    # Get mapping
    mapping, only1, only2 = get_weight_mapping(list(state1.keys()), list(state2.keys()))
    result.model1_only_weights = only1
    result.model2_only_weights = only2

    print(f"Mapped weights: {len(mapping)}")
    print(f"Model 1 only: {len(only1)}")
    print(f"Model 2 only: {len(only2)}")

    # Compare mapped weights
    print("\n" + "="*70)
    print("WEIGHT COMPARISON")
    print("="*70)

    for name1, name2 in sorted(mapping.items()):
        t1 = state1[name1]
        t2 = state2[name2]

        cmp = compare_tensors(t1, t2, name1, name2, threshold)
        result.weight_comparisons.append(cmp)

        if verbose or not cmp.matched:
            print_comparison(cmp, verbose)

    return result


# ==============================================================================
# Forward Pass Comparison
# ==============================================================================

def compare_forward_pass(
    model1,
    model2,
    tokenizer,
    seq_lengths: List[int] = [5, 64, 256],
    threshold: float = 0.05,
    device: str = "cuda:0"
) -> List[ForwardPassResult]:
    """Compare forward pass outputs of two models."""
    results = []

    print("\n" + "="*70)
    print("FORWARD PASS COMPARISON")
    print("="*70)

    for seq_len in seq_lengths:
        print(f"\n--- Sequence length: {seq_len} ---")

        # Generate random input
        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

        with torch.no_grad():
            # Forward pass
            out1 = model1(input_ids=input_ids)
            out2 = model2(input_ids=input_ids)

            logits1 = out1.logits
            logits2 = out2.logits

        # Compute metrics
        max_diff, mean_diff, cos_sim = compute_metrics(logits1, logits2)

        # Top-k matching
        top1_1 = logits1.argmax(dim=-1)
        top1_2 = logits2.argmax(dim=-1)
        top1_match = (top1_1 == top1_2).float().mean().item()

        top5_1 = logits1.topk(5, dim=-1).indices
        top5_2 = logits2.topk(5, dim=-1).indices
        # Check if top-1 of model1 is in top-5 of model2 and vice versa
        top5_match = 0.0
        for i in range(seq_len):
            if top1_1[0, i] in top5_2[0, i] and top1_2[0, i] in top5_1[0, i]:
                top5_match += 1
        top5_match /= seq_len

        logits_matched = max_diff < threshold and cos_sim > 0.995

        result = ForwardPassResult(
            seq_length=seq_len,
            logits_matched=logits_matched,
            max_diff=max_diff,
            mean_diff=mean_diff,
            cosine_sim=cos_sim,
            top1_match_rate=top1_match,
            top5_match_rate=top5_match
        )
        results.append(result)

        status = "✓" if logits_matched else "✗"
        print(f"  {status} max_diff={max_diff:.6f}, cos_sim={cos_sim:.6f}")
        print(f"      top1_match={top1_match*100:.1f}%, top5_match={top5_match*100:.1f}%")

    return results


# ==============================================================================
# Main Function
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate two HuggingFace models produce identical results"
    )

    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="Path to first HF model (e.g., old qwen3_next model)"
    )
    parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help="Path to second HF model (e.g., new alpha model)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Maximum allowed absolute difference for weight match (default: 0.01)"
    )
    parser.add_argument(
        "--forward-pass",
        action="store_true",
        help="Also run forward pass comparison"
    )
    parser.add_argument(
        "--seq-lengths",
        type=str,
        default="5,64,256",
        help="Comma-separated sequence lengths for forward pass (default: 5,64,256)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to load models (default: cuda:0)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all weight comparisons (not just mismatches)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*70)
    print("Alpha HF Model ↔ HF Model Validation")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model 1:    {args.model1}")
    print(f"  Model 2:    {args.model2}")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Forward:    {args.forward_pass}")
    print(f"  Device:     {args.device}")

    # Load models
    print("\n" + "="*70)
    print("LOADING MODELS")
    print("="*70)

    model1, tokenizer1, config1 = load_hf_model(args.model1, args.device)
    model2, tokenizer2, config2 = load_hf_model(args.model2, args.device)

    print(f"\nModel 1 type: {config1.model_type}")
    print(f"Model 2 type: {config2.model_type}")

    # Weight comparison
    result = compare_state_dicts(model1, model2, args.threshold, args.verbose)

    # Forward pass comparison (optional)
    if args.forward_pass:
        seq_lengths = [int(x) for x in args.seq_lengths.split(",")]
        forward_results = compare_forward_pass(
            model1, model2, tokenizer1, seq_lengths, args.threshold * 5, args.device
        )
        result.forward_pass_results = forward_results

    # Print summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    # Weight summary
    matched = sum(1 for c in result.weight_comparisons if c.matched)
    total = len(result.weight_comparisons)
    print(f"\nWeight comparisons: {matched}/{total} matched")

    if result.model1_only_weights:
        print(f"\n⚠ {len(result.model1_only_weights)} weights only in Model 1:")
        for w in sorted(result.model1_only_weights)[:10]:
            print(f"    - {w}")
        if len(result.model1_only_weights) > 10:
            print(f"    ... and {len(result.model1_only_weights) - 10} more")

    if result.model2_only_weights:
        print(f"\n⚠ {len(result.model2_only_weights)} weights only in Model 2:")
        for w in sorted(result.model2_only_weights)[:10]:
            print(f"    - {w}")
        if len(result.model2_only_weights) > 10:
            print(f"    ... and {len(result.model2_only_weights) - 10} more")

    # Worst comparisons
    if result.weight_comparisons:
        worst = sorted(result.weight_comparisons, key=lambda x: -x.max_diff)[:5]
        print(f"\nTop 5 worst weight differences:")
        for cmp in worst:
            status = "✓" if cmp.matched else "✗"
            print(f"  {status} {cmp.name1}: max_diff={cmp.max_diff:.6f}, cos_sim={cmp.cosine_sim:.6f}")

    # Forward pass summary
    if result.forward_pass_results:
        fwd_matched = sum(1 for r in result.forward_pass_results if r.logits_matched)
        fwd_total = len(result.forward_pass_results)
        print(f"\nForward pass: {fwd_matched}/{fwd_total} matched")

        for r in result.forward_pass_results:
            status = "✓" if r.logits_matched else "✗"
            print(f"  {status} seq_len={r.seq_length}: max_diff={r.max_diff:.6f}, "
                  f"top1={r.top1_match_rate*100:.1f}%")

    # Final verdict
    print("\n" + "="*70)

    weights_ok = result.weights_matched
    forward_ok = result.forward_matched
    no_missing = not result.model1_only_weights and not result.model2_only_weights

    if weights_ok and forward_ok and no_missing:
        print("✓ ALL VALIDATIONS PASSED - MODELS ARE IDENTICAL")
        print("="*70 + "\n")
        sys.exit(0)
    elif weights_ok and forward_ok:
        print("⚠ Weights matched but some weight names differ between models")
        print("  This may be expected if model architectures have different naming")
        print("="*70 + "\n")
        sys.exit(0)
    elif weights_ok:
        print("⚠ Weights matched but forward pass differs")
        print("  Check for numerical precision or model state issues")
        print("="*70 + "\n")
        sys.exit(1)
    else:
        print("✗ VALIDATION FAILED - WEIGHTS DO NOT MATCH")
        print("  Check the detailed output above for mismatched weights.")
        print("="*70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
