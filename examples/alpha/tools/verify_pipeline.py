#!/usr/bin/env python3
# Copyright (c) 2025 Alpha Project Team.
#
# Alpha v2 evaluation-pipeline verification gates
# ===============================================
# Cheap, deterministic checks that bind the conversion to the checkpoint's
# ground truth (common.pt). Used by evaluate.sh (Stage 0 pre-convert and Stage
# 1.5 post-convert) and by tests/test_alpha_pipeline_config.py.
#
#   preflight       --from-checkpoint <ckpt> [--gpus N]
#       Derive the expected HF config from common.pt and sanity-check it
#       (tokenizer dir exists & is v5, num_experts % gpus == 0, even #layers,
#        eos=0 / no BOS). Fails fast before any expensive conversion.
#
#   compare-config  --from-checkpoint <ckpt> --hf <hfmodel_dir>
#       Assert the produced config.json matches the common.pt-derived values on
#       every structural field (catches 184-vs-192-style drift).
#
#   tokenizer-roundtrip --hf <hfmodel_dir>
#       Load the converted tokenizer (trust_remote_code) and assert EOS id == 0
#       plus a byte-level encode/decode round-trip on EN/KO/code samples.

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.absolute()))
import alpha_config as ac  # noqa: E402

# config.json fields that must match the checkpoint exactly.
STRUCT_FIELDS = [
    "num_experts",
    "num_experts_per_tok",
    "vocab_size",
    "head_dim",
    "num_hidden_layers",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "rope_theta",
    "partial_rotary_factor",
    "max_position_embeddings",
    "num_attention_heads",
    "num_key_value_heads",
    "eos_token_id",
    "full_attention_interval",
    # DSV3 routing — must match training or the converted model routes wrongly.
    "scoring_func",
    "n_group",
    "topk_group",
    "routed_scaling_factor",
]


def _expected(ckpt):
    cfg = ac.load_config_from_checkpoint(ckpt)
    return ac.generate_hf_config(cfg), cfg


def cmd_preflight(args):
    hf, cfg = _expected(args.from_checkpoint)
    problems, warnings = [], []

    tp = cfg.tokenizer_path
    if not tp or not Path(tp).is_dir():
        problems.append(f"tokenizer dir not found: {tp!r}")
    elif "tokenizer_v5" not in tp:
        warnings.append(f"tokenizer path is not tokenizer_v5: {tp!r}")

    if args.gpus:
        if cfg.moe.num_experts % args.gpus != 0:
            problems.append(
                f"num_experts ({cfg.moe.num_experts}) not divisible by gpus ({args.gpus})"
            )

    if cfg.num_layers % 2 != 0:
        problems.append(f"num_layers ({cfg.num_layers}) must be even for 2:1 HF mapping")
    if hf.get("eos_token_id") != 0:
        problems.append(f"eos_token_id expected 0, got {hf.get('eos_token_id')}")
    if "bos_token_id" in hf:
        problems.append(f"bos_token_id should be absent (alpha has no BOS); got {hf['bos_token_id']}")

    print("── Stage 0 preflight (checkpoint ground truth) ──")
    for k in STRUCT_FIELDS:
        print(f"  {k:32s} = {hf.get(k)}")
    print(f"  {'tokenizer_model':32s} = {tp}")
    for w in warnings:
        print(f"  ⚠️  {w}")
    if problems:
        print("❌ preflight FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("✅ preflight passed")
    return 0


def cmd_compare_config(args):
    exp, _ = _expected(args.from_checkpoint)
    cfg_path = Path(args.hf) / "config.json"
    if not cfg_path.exists():
        print(f"❌ {cfg_path} not found")
        return 1
    actual = json.loads(cfg_path.read_text())

    mismatches = []
    for k in STRUCT_FIELDS:
        if exp.get(k) != actual.get(k):
            mismatches.append((k, exp.get(k), actual.get(k)))
    # BOS must be absent (or explicitly None) in the produced config too.
    if actual.get("bos_token_id") is not None:
        mismatches.append(("bos_token_id", "absent/None", actual.get("bos_token_id")))

    print("── Stage 1.5 config.json ↔ checkpoint ──")
    print(f"  {'field':32s} {'expected':>14s} {'actual':>14s}")
    for k in STRUCT_FIELDS:
        flag = "" if exp.get(k) == actual.get(k) else "  <-- MISMATCH"
        print(f"  {k:32s} {str(exp.get(k)):>14s} {str(actual.get(k)):>14s}{flag}")
    if mismatches:
        print("❌ config.json does not match checkpoint:")
        for k, e, a in mismatches:
            print(f"  - {k}: expected {e!r}, got {a!r}")
        return 1
    print("✅ config.json matches checkpoint ground truth")
    return 0


def cmd_tokenizer_roundtrip(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "안녕하세요. 오늘 날씨가 정말 좋네요!",
        "def add(a, b):\n    return a + b\n",
    ]
    problems = []
    print("── tokenizer round-trip ──")
    print(f"  eos_token_id = {tok.eos_token_id}   vocab_size = {tok.vocab_size}   len = {len(tok)}")
    if tok.eos_token_id != 0:
        problems.append(f"eos_token_id expected 0, got {tok.eos_token_id}")
    for s in samples:
        ids = tok.encode(s, add_special_tokens=False)
        dec = tok.decode(ids)
        ok = dec.strip() == s.strip()
        print(f"  {'OK ' if ok else 'DIFF'} {len(ids):3d} tok  {s[:40]!r}")
        if not ok:
            problems.append(f"round-trip differs for {s[:30]!r}: decoded {dec[:40]!r}")
    if problems:
        print("❌ tokenizer checks FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("✅ tokenizer round-trip ok (EOS id 0)")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Alpha v2 pipeline verification gates")
    sub = parser.add_subparsers(dest="command")

    p0 = sub.add_parser("preflight", help="Pre-convert sanity gate (checkpoint)")
    p0.add_argument("--from-checkpoint", required=True)
    p0.add_argument("--gpus", type=int, default=None)
    p0.set_defaults(func=cmd_preflight)

    p1 = sub.add_parser("compare-config", help="config.json ↔ checkpoint")
    p1.add_argument("--from-checkpoint", required=True)
    p1.add_argument("--hf", required=True)
    p1.set_defaults(func=cmd_compare_config)

    p2 = sub.add_parser("tokenizer-roundtrip", help="EOS id + encode/decode round-trip")
    p2.add_argument("--hf", required=True)
    p2.set_defaults(func=cmd_tokenizer_roundtrip)

    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
