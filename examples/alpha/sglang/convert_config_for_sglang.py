#!/usr/bin/env python3
"""Convert Alpha HuggingFace checkpoint for SGLang deployment.

Two modes:
  - validate: Check if the Alpha HF checkpoint is ready for --trust-remote-code fallback
  - native:   Create a Qwen3-Next compatible checkpoint directory for native SGLang serving

Usage:
  # Validate for Option A (HF Fallback)
  python convert_config_for_sglang.py --input-path /path/to/alpha-hf --mode validate

  # Convert for Option B (Native Qwen3-Next)
  python convert_config_for_sglang.py --input-path /path/to/alpha-hf --output-path /path/to/output --mode native
"""

import argparse
import json
import os
import sys
from pathlib import Path


# Fields that map 1:1 between Alpha and Qwen3-Next configs
SHARED_FIELDS = [
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "max_position_embeddings",
    "initializer_range",
    "rms_norm_eps",
    "use_cache",
    "tie_word_embeddings",
    "rope_theta",
    "rope_scaling",
    "partial_rotary_factor",
    "attention_bias",
    "attention_dropout",
    "vocab_size",
    # Linear attention (GatedDeltaNet)
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    # MoE
    "decoder_sparse_step",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "num_experts_per_tok",
    "num_experts",
    "norm_topk_prob",
    "output_router_logits",
    "router_aux_loss_coef",
    # Hybrid
    "full_attention_interval",
    # Misc
    "bos_token_id",
    "eos_token_id",
    "torch_dtype",
    "use_sliding_window",
]


def load_config(path: str) -> dict:
    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        print(f"ERROR: config.json not found in {path}")
        sys.exit(1)
    with open(config_path) as f:
        return json.load(f)


def validate_alpha_config(config: dict) -> list[str]:
    """Validate Alpha config for SGLang compatibility. Returns list of warnings."""
    warnings = []

    # Check model_type
    if config.get("model_type") != "alpha":
        warnings.append(f"Unexpected model_type: {config.get('model_type')} (expected 'alpha')")

    # Check architectures
    archs = config.get("architectures", [])
    if "AlphaForCausalLM" not in archs:
        warnings.append(f"AlphaForCausalLM not in architectures: {archs}")

    # Check required HF model files
    required_files = ["modeling_alpha.py", "configuration_alpha.py"]
    # These will be checked at path level

    # Check hybrid parameters
    interval = config.get("full_attention_interval")
    num_layers = config.get("num_hidden_layers", 0)
    if interval and num_layers:
        num_attn = num_layers // interval
        pct = num_attn / num_layers * 100
        print(f"  Layer structure: {num_layers} HF layers, {num_attn} attention ({pct:.1f}%), "
              f"{num_layers - num_attn} linear ({100-pct:.1f}%)")

    # Check mlp_only_layers
    mlp_only = config.get("mlp_only_layers", [])
    if mlp_only:
        print(f"  Dense MLP layers (mlp_only_layers): {mlp_only}")

    # Check MoE config
    num_experts = config.get("num_experts", 0)
    topk = config.get("num_experts_per_tok", 0)
    if num_experts > 0:
        print(f"  MoE: {num_experts} experts, top-{topk}")

    # Check GatedDeltaNet config
    key_heads = config.get("linear_num_key_heads")
    val_heads = config.get("linear_num_value_heads")
    key_dim = config.get("linear_key_head_dim")
    val_dim = config.get("linear_value_head_dim")
    if key_heads and val_heads:
        print(f"  GatedDeltaNet: {key_heads} key heads ({key_dim}d), {val_heads} value heads ({val_dim}d)")

    return warnings


def convert_to_qwen3_next(alpha_config: dict) -> dict:
    """Convert Alpha config to Qwen3-Next format for native SGLang serving.

    The converted config uses model_type='alpha' so our custom SGLang model
    adapter (sglang_alpha_model.py) handles it. This adapter extends Qwen3-Next
    with mlp_only_layers support.
    """
    native_config = {}

    # Copy shared fields
    for field in SHARED_FIELDS:
        if field in alpha_config:
            native_config[field] = alpha_config[field]

    # Set architecture for SGLang model adapter registration
    native_config["architectures"] = ["AlphaForCausalLM"]
    native_config["model_type"] = "alpha"

    # auto_map for config class only (needed for --trust-remote-code config loading)
    # Points to configuration_alpha.py which will be copied to the output dir.
    # NOTE: no AutoModelForCausalLM mapping — SGLang uses its own model adapter.
    native_config["auto_map"] = {
        "AutoConfig": "configuration_alpha.AlphaConfig",
    }

    # Keep mlp_only_layers - our SGLang model adapter handles this
    native_config["mlp_only_layers"] = alpha_config.get("mlp_only_layers", [])

    # Generate layer_types for explicit layer type specification
    num_layers = alpha_config.get("num_hidden_layers", 24)
    interval = alpha_config.get("full_attention_interval", 4)
    layer_types = []
    for i in range(num_layers):
        if (i + 1) % interval == 0:
            layer_types.append("full_attention")
        else:
            layer_types.append("linear_attention")
    native_config["layer_types"] = layer_types

    # Remove AutoModelForCausalLM from auto_map (SGLang uses its own model adapter)
    # but keep AutoConfig (needed for --trust-remote-code config loading)
    if "auto_map" in native_config:
        native_config["auto_map"].pop("AutoModelForCausalLM", None)

    # Ensure transformers_version is present
    if "transformers_version" not in native_config:
        native_config["transformers_version"] = alpha_config.get("transformers_version", "4.57.0.dev0")

    return native_config


def create_native_checkpoint(input_path: str, output_path: str, config: dict):
    """Create a native SGLang checkpoint directory.

    Symlinks weight files (no copy) and writes converted config.json.
    """
    os.makedirs(output_path, exist_ok=True)
    input_path = os.path.abspath(input_path)

    # Write converted config
    config_out = os.path.join(output_path, "config.json")
    with open(config_out, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Written: {config_out}")

    # Symlink weight files (safetensors or bin)
    weight_patterns = ["*.safetensors", "*.bin", "model.safetensors.index.json", "model.bin.index.json"]
    linked = 0
    for pattern in weight_patterns:
        for src in Path(input_path).glob(pattern):
            dst = os.path.join(output_path, src.name)
            if not os.path.exists(dst):
                os.symlink(str(src), dst)
                linked += 1
    print(f"  Symlinked {linked} weight files")

    # Symlink tokenizer files
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "added_tokens.json",
    ]
    for fname in tokenizer_files:
        src = os.path.join(input_path, fname)
        dst = os.path.join(output_path, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    # Copy configuration_alpha.py (needed for --trust-remote-code config loading)
    # but DO NOT copy modeling_alpha.py (SGLang uses its own model adapter)
    config_py_src = os.path.join(input_path, "configuration_alpha.py")
    config_py_dst = os.path.join(output_path, "configuration_alpha.py")
    init_py_src = os.path.join(input_path, "__init__.py")
    init_py_dst = os.path.join(output_path, "__init__.py")
    if os.path.exists(config_py_src) and not os.path.exists(config_py_dst):
        os.symlink(config_py_src, config_py_dst)
        print(f"  Symlinked: configuration_alpha.py (for config parsing)")
    if os.path.exists(init_py_src) and not os.path.exists(init_py_dst):
        os.symlink(init_py_src, init_py_dst)
    print(f"  Note: modeling_alpha.py NOT copied — native mode uses SGLang adapter")


def main():
    parser = argparse.ArgumentParser(description="Convert Alpha HF checkpoint for SGLang")
    parser.add_argument("--input-path", required=True, help="Path to Alpha HF checkpoint")
    parser.add_argument("--output-path", help="Output path for converted checkpoint (native mode)")
    parser.add_argument("--mode", choices=["validate", "native"], default="validate",
                        help="validate: check compatibility, native: convert for SGLang")
    parser.add_argument("--print-config", action="store_true", help="Print converted config.json")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Alpha → SGLang Config Converter")
    print(f"{'='*60}")
    print(f"  Input:  {args.input_path}")
    print(f"  Mode:   {args.mode}")
    if args.output_path:
        print(f"  Output: {args.output_path}")
    print()

    # Load and validate Alpha config
    config = load_config(args.input_path)
    print("[1] Validating Alpha config...")
    warnings = validate_alpha_config(config)

    if warnings:
        print(f"\n  Warnings:")
        for w in warnings:
            print(f"    ⚠ {w}")

    # Check required model files for fallback mode
    hf_files = ["modeling_alpha.py", "configuration_alpha.py", "__init__.py"]
    missing = [f for f in hf_files if not os.path.exists(os.path.join(args.input_path, f))]
    if missing and args.mode == "validate":
        print(f"\n  Missing HF model files for --trust-remote-code: {missing}")
        print(f"  Copy from: examples/alpha/hf_model/")

    if args.mode == "validate":
        print(f"\n[Result] Config is {'valid' if not warnings else 'valid with warnings'} for SGLang fallback mode.")
        print(f"  Deploy command:")
        print(f"    python -m sglang.launch_server --model-path {args.input_path} --trust-remote-code --port 30000")
        return

    # Native mode: convert config
    if args.mode == "native":
        if not args.output_path:
            args.output_path = args.input_path + "_sglang_native"

        print(f"\n[2] Converting to native SGLang format...")
        native_config = convert_to_qwen3_next(config)

        if args.print_config:
            print(f"\n  Converted config.json:")
            print(json.dumps(native_config, indent=2))

        print(f"\n[3] Creating native checkpoint at: {args.output_path}")
        create_native_checkpoint(args.input_path, args.output_path, native_config)

        # Summary
        print(f"\n{'='*60}")
        print(f"  Conversion complete!")
        print(f"{'='*60}")
        print(f"  Native checkpoint: {args.output_path}")
        print(f"")
        print(f"  Deploy with local SGLang backend:")
        print(f"    bash examples/alpha/sglang/deploy.sh {args.output_path} --mode native")


if __name__ == "__main__":
    main()
