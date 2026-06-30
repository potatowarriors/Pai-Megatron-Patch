#!/usr/bin/env python3
"""Forward-sanity gate for a converted Alpha HF checkpoint.

Why this exists
---------------
`validate_mg_hf_full.py` compares MG↔HF **weights** only, and its attention
comparison even reuses the converter's own reshape — so it cannot detect a
*forward-pass* mismatch where the tensors are copied correctly but the HF model
**interprets** them differently than Megatron does. That blind spot let a real
bug ship: `AlphaRMSNorm` applied the zero-centered `(1 + gamma)` form inherited
from Qwen3-Next, while Alpha v2 trains with `apply-layernorm-1p` OFF (standard
`x * gamma`). Every norm output was scaled ~1.7-2.5x, the residual stream got
corrupted across all layers, and *all* benchmarks collapsed to chance
(ARC-easy = 25%). Weight validation passed the whole time.

This gate closes that hole with the cheapest possible forward check: load the
converted HF model and measure perplexity on a fixed, easy English snippet. A
faithfully-converted model — even early in training — scores low perplexity; a
globally-corrupted forward scores near-uniform (ppl ≈ vocab_size). It needs no
Megatron, no distributed setup, and one forward pass.

Usage
-----
    python forward_sanity.py --hf <hfmodel_dir> [--threshold 100] [--device cuda:0]

Exit code 0 if perplexity < threshold and logits are finite, else 1.
"""
import argparse
import math
import sys

import torch

# A few short, unambiguous factual sentences. A trained LM (any size, any stage)
# assigns these high likelihood; a broken forward does not.
SANITY_TEXT = (
    "The capital of France is Paris. Water is made of hydrogen and oxygen. "
    "The sun rises in the east and sets in the west. Two plus two equals four. "
    "Monday comes before Tuesday, and winter is colder than summer."
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf", required=True, help="Converted HF model directory")
    ap.add_argument("--threshold", type=float, default=100.0,
                    help="Max allowed perplexity (default 100; random ≈ vocab_size).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--text", default=SANITY_TEXT)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[forward-sanity] loading {args.hf} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf, dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
    ).eval()

    ids = tok(args.text, return_tensors="pt").input_ids.to(args.device)
    with torch.no_grad():
        out = model(ids, labels=ids)
    loss = out.loss.item()
    ppl = math.exp(loss)
    finite = bool(torch.isfinite(out.logits).all().item())

    print(f"[forward-sanity] loss={loss:.4f}  perplexity={ppl:.2f}  "
          f"logits_finite={finite}  threshold={args.threshold}", flush=True)

    ok = finite and ppl < args.threshold
    if ok:
        print("[forward-sanity] ✅ PASS — forward looks faithful.", flush=True)
        sys.exit(0)
    print("[forward-sanity] ❌ FAIL — perplexity near-random or non-finite logits.\n"
          "   The converted model's forward is corrupted even though weights may match.\n"
          "   Suspect a forward/interpretation mismatch (RMSNorm 1p vs standard, RoPE\n"
          "   partial-rotary, MoE routing, gate ordering). See examples/alpha/CLAUDE.md.",
          flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
