#!/usr/bin/env python3
"""
Quick test: Compare MG and HF in_proj/conv1d weights to verify ordering.
This tests ONLY the weight ordering without running full inference.
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, ROOT_DIR)

import torch
import argparse
from safetensors import safe_open

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", required=True, help="Path to HF model")
    parser.add_argument("--mg-checkpoint", required=True, help="Path to MG checkpoint (unused, we load from HF)")
    args = parser.parse_args()

    print("="*70)
    print("Weight Ordering Test: HF in_proj structure analysis")
    print("="*70)

    # Load HF weights directly from safetensors
    print("\n[1] Loading HF weights from safetensors...")
    hf_dir = args.hf_model

    # Find layer 0 linear_attn weights
    layer0_weights = {}
    for i in range(1, 9):
        st_file = os.path.join(hf_dir, f"model-0000{i}-of-00008.safetensors")
        if os.path.exists(st_file):
            with safe_open(st_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if "layers.0.linear_attn" in key:
                        layer0_weights[key] = f.get_tensor(key)

    print(f"  Found {len(layer0_weights)} layer 0 linear_attn weights:")
    for k, v in layer0_weights.items():
        print(f"    {k}: {v.shape}")

    # Get dimensions from config
    import json
    config_path = os.path.join(hf_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    print(f"\n[2] Config dimensions:")
    # For Qwen3-Next GatedDeltaNet, use linear_* keys:
    # linear_num_key_heads, linear_num_value_heads
    # linear_key_head_dim, linear_value_head_dim

    Nk = config.get("linear_num_key_heads", 16)  # num_k_heads
    Nv = config.get("linear_num_value_heads", 32)  # num_v_heads
    Dk = config.get("linear_key_head_dim", 128)  # head_k_dim
    Dv = config.get("linear_value_head_dim", 128)  # head_v_dim
    hidden_size = config.get("hidden_size", 2048)

    print(f"  Nk (num_k_heads) = {Nk}")
    print(f"  Nv (num_v_heads) = {Nv}")
    print(f"  Dk (head_k_dim) = {Dk}")
    print(f"  Dv (head_v_dim) = {Dv}")
    print(f"  Q dim = Dk * Nk = {Dk * Nk}")
    print(f"  K dim = Dk * Nk = {Dk * Nk}")
    print(f"  V dim = Dv * Nv = {Dv * Nv}")
    print(f"  Z dim = Dv * Nv = {Dv * Nv}")

    # HF in_proj_qkvz: [Q, K, V, Z] order
    # HF in_proj_ba: [b, a] order
    in_proj_qkvz_key = "model.layers.0.linear_attn.in_proj_qkvz.weight"
    in_proj_ba_key = "model.layers.0.linear_attn.in_proj_ba.weight"
    conv1d_key = "model.layers.0.linear_attn.conv1d.weight"

    if in_proj_qkvz_key not in layer0_weights:
        print(f"\n  ERROR: {in_proj_qkvz_key} not found!")
        return

    hf_in_proj_qkvz = layer0_weights[in_proj_qkvz_key]
    hf_in_proj_ba = layer0_weights[in_proj_ba_key]
    hf_conv1d = layer0_weights[conv1d_key]

    print(f"\n[3] HF weight shapes:")
    print(f"  in_proj_qkvz: {hf_in_proj_qkvz.shape}")
    print(f"  in_proj_ba: {hf_in_proj_ba.shape}")
    print(f"  conv1d: {hf_conv1d.shape}")

    # Split HF weights
    hf_qkvz_splits = [Dk*Nk, Dk*Nk, Dv*Nv, Dv*Nv]
    print(f"\n[4] HF in_proj_qkvz split sizes [Q, K, V, Z]: {hf_qkvz_splits}")
    hf_Q_w, hf_K_w, hf_V_w, hf_Z_w = torch.split(hf_in_proj_qkvz, hf_qkvz_splits, dim=0)

    hf_ba_splits = [Nv, Nv]
    print(f"  HF in_proj_ba split sizes [b, a]: {hf_ba_splits}")
    hf_b_w, hf_a_w = torch.split(hf_in_proj_ba, hf_ba_splits, dim=0)

    print(f"\n  Split results:")
    print(f"    Q: {hf_Q_w.shape}, mean={hf_Q_w.float().mean().item():.6f}")
    print(f"    K: {hf_K_w.shape}, mean={hf_K_w.float().mean().item():.6f}")
    print(f"    V: {hf_V_w.shape}, mean={hf_V_w.float().mean().item():.6f}")
    print(f"    Z: {hf_Z_w.shape}, mean={hf_Z_w.float().mean().item():.6f}")
    print(f"    b: {hf_b_w.shape}, mean={hf_b_w.float().mean().item():.6f}")
    print(f"    a: {hf_a_w.shape}, mean={hf_a_w.float().mean().item():.6f}")

    # HF conv1d order: [Q, K, V]
    hf_conv_splits = [Dk*Nk, Dk*Nk, Dv*Nv]
    print(f"\n[5] HF conv1d split sizes [Q, K, V]: {hf_conv_splits}")
    hf_conv_Q, hf_conv_K, hf_conv_V = torch.split(hf_conv1d.squeeze(1), hf_conv_splits, dim=0)

    print(f"  Split results:")
    print(f"    conv_Q: {hf_conv_Q.shape}, mean={hf_conv_Q.float().mean().item():.6f}")
    print(f"    conv_K: {hf_conv_K.shape}, mean={hf_conv_K.float().mean().item():.6f}")
    print(f"    conv_V: {hf_conv_V.shape}, mean={hf_conv_V.float().mean().item():.6f}")

    # Now reconstruct what MG should have
    print("\n" + "="*70)
    print("[6] Reconstructing MG format from HF weights")
    print("="*70)

    # MG in_proj order: [z, V, Q, K, b, a]
    mg_in_proj_reconstructed = torch.cat([hf_Z_w, hf_V_w, hf_Q_w, hf_K_w, hf_b_w, hf_a_w], dim=0)
    print(f"  MG in_proj (reconstructed): {mg_in_proj_reconstructed.shape}")
    print(f"  Expected MG split sizes [z, V, Q, K, b, a]: [{Dv*Nv}, {Dv*Nv}, {Dk*Nk}, {Dk*Nk}, {Nv}, {Nv}]")

    # MG conv1d order: [V, Q, K]
    mg_conv_reconstructed = torch.cat([hf_conv_V, hf_conv_Q, hf_conv_K], dim=0)
    print(f"  MG conv1d (reconstructed): {mg_conv_reconstructed.shape}")
    print(f"  Expected MG split sizes [V, Q, K]: [{Dv*Nv}, {Dk*Nk}, {Dk*Nk}]")

    print("\n" + "="*70)
    print("SUMMARY: Forward split order comparison")
    print("="*70)
    print("""
MG forward (gated_deltanet.py line 290-320):
  1. in_proj output: [z, VQK, ba] split
     - z: d_inner (=Dv*Nv)
     - VQK: d_inner + 2*ngroups*d_state (=Dv*Nv + Dk*Nk + Dk*Nk)
     - ba: nheads*2 (=Nv + Nv)
  2. VQK → conv1d → [V, Q, K] split
     - V: d_inner (=Dv*Nv)
     - Q: ngroups*d_state (=Dk*Nk)
     - K: ngroups*d_state (=Dk*Nk)

HF forward (modeling_qwen3_next.py):
  1. in_proj_qkvz output: [Q, K, V, Z] split
  2. in_proj_ba output: [b, a] split
  3. QKV → conv1d → [Q, K, V] split

KEY INSIGHT:
  - MG forward expects weights in [z, V, Q, K, b, a] order
  - MG sharded_state_dict stores with labels ["z", "V", "Q", "K", "b", "a"]
  - MG2HF converter reorders to HF format [Q, K, V, Z], [b, a]
  - HF2MG converter should reorder back to MG format

  If MG loads weights correctly and forward uses correct order,
  the inference should match HF.

  PROBLEM: The intermediate values don't match!
  → Check if MG forward is actually using the correct order
  → Or if there's something else different (e.g., normalization, FLA kernel)
""")

    # Check if there's a bias
    print("\n[7] Checking for biases:")
    for k in layer0_weights:
        if "bias" in k.lower():
            print(f"  {k}: {layer0_weights[k].shape}")


if __name__ == "__main__":
    main()
