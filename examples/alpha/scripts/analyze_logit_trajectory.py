#!/usr/bin/env python3
"""
Attention-logit TRAJECTORY analysis for Alpha — v1 (explosion) vs v2 (stable).

Extends scripts/check_attention_logits.py from a single checkpoint to a full time-series
across each run's `hfmodel_NNNNNNN/` exports, measuring the actual pre-softmax attention
logits (Q @ K^T / sqrt(d_k)) per checkpoint with BOTH input modes:
  * random tokens (fixed seed) — apples-to-apples with the original "max logit < 20" claim
  * real-text sample (each model's own tokenizer) — faithful to training-time logits

CONVENTION CORRECTNESS (critical for v1):
  q_norm/k_norm are AlphaRMSNorm modules. v1 trained with apply_layernorm_1p=True
  (effective γ = 1 + w); v2 with False (effective γ = w). The HF export's bundled
  modeling_alpha.py may have been overwritten with the standard-γ version — loading v1
  under standard-γ would make q_norm output ≈ 0 and the measured logits meaningless.
  We therefore OVERRIDE AlphaRMSNorm.forward per run with the convention read from the
  matching torch_dist common.pt (apply_layernorm_1p). AlphaRMSNormGated (Mamba GDN norm,
  always standard) is left untouched, so only the QK path is corrected.

X-axis: consumed TOKENS (from the matching torch_dist common.pt), so v1 (GBS 256) and
v2 (GBS 1536) overlay on a fair budget axis.

Usage:
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
  echo "The capital of France is Paris. 인공지능 ..." > /tmp/probe.txt
  CUDA_VISIBLE_DEVICES=0 python analyze_logit_trajectory.py \
    --runs outputs/alpha_v1/alpha_baseline_48L_20251219_095156 \
           outputs/alpha_baseline_48L_stage1_20260512_170157 \
           outputs/alpha_baseline_48L_stage1_resume_20260526_194537 \
    --real-text /tmp/probe.txt --seq-len 2048 --num-samples 2 \
    --output outputs/analysis/v1_vs_v2_logit_explosion/logit_trajectory.json \
    --plot   outputs/analysis/v1_vs_v2_logit_explosion/plots
"""

import argparse
import gc
import os
import re
import sys
import json
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_expert_balance import load_common_args  # pure torch.load(common.pt), no megatron
from analyze_weight_trajectory import consumed_tokens, derive_group, GROUP_STYLE


# ----------------------------------------------------------------------------
# Discovery + config
# ----------------------------------------------------------------------------

def find_hf_exports(run_dir: str):
    """Return sorted [(iter, hfmodel_dir)] under a run dir."""
    run_dir = run_dir.rstrip("/")
    out = []
    for name in os.listdir(run_dir):
        m = re.fullmatch(r"hfmodel_(\d+)", name)
        full = os.path.join(run_dir, name)
        if m and os.path.isdir(full) and os.path.isfile(os.path.join(full, "config.json")):
            out.append((int(m.group(1)), full))
    return sorted(out)


def checkpoint_meta(run_dir: str, it: int, group: str):
    """Read apply_layernorm_1p + consumed tokens from the matching torch_dist common.pt."""
    ckpt = os.path.join(run_dir.rstrip("/"), "checkpoints", f"iter_{it:07d}")
    if os.path.isfile(os.path.join(ckpt, "common.pt")):
        cargs, iteration = load_common_args(ckpt)
        one_p = bool(getattr(cargs, "apply_layernorm_1p", False))
        tokens, samples, _ = consumed_tokens(cargs, iteration)
        return one_p, tokens
    # fallback: infer from group (v1 used 1p; v2 did not)
    one_p = group == "v1"
    return one_p, 0


# ----------------------------------------------------------------------------
# Logit capture (monkey-patch eager_attention_forward)
# ----------------------------------------------------------------------------

def make_patched_forward(original_fn, repeat_kv_fn, records):
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        with torch.no_grad():
            key_e = repeat_kv_fn(key, module.num_key_value_groups)
            logits = (torch.matmul(query, key_e.transpose(2, 3)) * scaling).float()
            S = logits.shape[-1]
            causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=logits.device))
            mask = causal.unsqueeze(0).unsqueeze(0)
            per_head_max = logits.masked_fill(~mask, float("-inf")).amax(dim=(-2, -1))  # (B,H)
            valid = logits[mask.expand_as(logits)]
            li = module.layer_idx
            r = records.setdefault(li, {"global_max": [], "per_head_max": []})
            r["global_max"].append(valid.max().item())
            r["per_head_max"].append(per_head_max.squeeze(0).tolist())
        return original_fn(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    return patched


def find_modeling_module():
    for name, mod in sys.modules.items():
        if "modeling_alpha" in name and hasattr(mod, "eager_attention_forward"):
            return mod
    return None


def patch_rmsnorm_convention(modeling_mod, one_p: bool):
    """Force AlphaRMSNorm.forward to the run's convention (γ = 1+w if one_p else w).
    Leaves AlphaRMSNormGated (always standard) untouched. Returns (cls, original)."""
    cls = getattr(modeling_mod, "AlphaRMSNorm", None)
    if cls is None:
        return None, None
    original = cls.forward

    def forward(self, hidden_states):
        in_dtype = hidden_states.dtype
        eps = getattr(self, "eps", None)
        if eps is None:
            eps = getattr(self, "variance_epsilon", 1e-6)
        x = hidden_states.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + eps)
        w = self.weight.float()
        g = (1.0 + w) if one_p else w
        return (g * x).to(in_dtype)

    cls.forward = forward
    return cls, original


# ----------------------------------------------------------------------------
# Per-checkpoint measurement
# ----------------------------------------------------------------------------

def reduce_records(records):
    """records: layer_idx -> {global_max:[...], per_head_max:[[...]]}.  Reduce over samples."""
    out = {}
    g_max, g_layer, g_head = float("-inf"), -1, -1
    for li in sorted(records):
        r = records[li]
        gmax = max(r["global_max"])
        nh = len(r["per_head_max"][0])
        head_max = [max(s[h] for s in r["per_head_max"]) for h in range(nh)]
        out[li] = {"global_max": gmax, "per_head_max": head_max}
        lmax = max(head_max)
        if lmax > g_max:
            g_max, g_layer, g_head = lmax, li, head_max.index(lmax)
    return out, {"global_max": g_max, "layer": g_layer, "head": g_head}


def measure_checkpoint(hf_dir, one_p, seq_len, num_samples, seed, real_ids, device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        hf_dir, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device).eval()
    assert model.config._attn_implementation == "eager"

    # IMPORTANT: each hfmodel_* dir gets its OWN dynamic module
    # (transformers_modules.hfmodel_XXXX.modeling_alpha). Derive the module from THIS
    # model instance — a global sys.modules search returns a stale module after the first
    # checkpoint, so the monkey-patch would miss and capture nothing (-inf).
    mod = sys.modules.get(type(model).__module__)
    if mod is None or not hasattr(mod, "eager_attention_forward"):
        mod = find_modeling_module()
    if mod is None:
        raise RuntimeError("modeling_alpha module not found for this model")

    norm_cls, norm_orig = patch_rmsnorm_convention(mod, one_p)
    orig_fn, repeat_kv = mod.eager_attention_forward, mod.repeat_kv
    vocab = model.config.vocab_size

    result = {}
    try:
        # --- random tokens ---
        rand_rec = {}
        mod.eager_attention_forward = make_patched_forward(orig_fn, repeat_kv, rand_rec)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for _ in range(num_samples):
            ids = torch.randint(0, vocab, (1, seq_len), generator=gen).to(device)
            with torch.no_grad():
                model(ids)
        result["random"] = dict(zip(("per_layer", "global"), reduce_records(rand_rec)))

        # --- real text (tiled/truncated to seq_len) ---
        if real_ids is not None:
            ids = real_ids[:seq_len]
            if len(ids) < seq_len:
                reps = (seq_len + len(ids) - 1) // len(ids)
                ids = (ids * reps)[:seq_len]
            ids_t = torch.tensor([ids], device=device)
            ids_t = torch.clamp(ids_t, 0, vocab - 1)
            real_rec = {}
            mod.eager_attention_forward = make_patched_forward(orig_fn, repeat_kv, real_rec)
            with torch.no_grad():
                model(ids_t)
            result["real"] = dict(zip(("per_layer", "global"), reduce_records(real_rec)))
    finally:
        mod.eager_attention_forward = orig_fn
        if norm_cls is not None:
            norm_cls.forward = norm_orig
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return result


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def write_plots(plot_dir, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)
    paths = []

    def style(g):
        return GROUP_STYLE.get(g, dict(color="#7f7f7f", marker="x"))

    # 1) global max logit vs tokens (random solid, real dashed)
    fig, ax = plt.subplots(figsize=(11, 6))
    for run in runs:
        st = style(run["group"])
        pts = [p for p in run["points"] if p.get("result")]
        if not pts:
            continue
        xs = [p["consumed_tokens"] / 1e9 for p in pts]
        for mode, ls in (("random", "-"), ("real", "--")):
            ys = [p["result"].get(mode, {}).get("global", {}).get("global_max") for p in pts]
            if any(y is not None for y in ys):
                ax.plot(xs, ys, ls=ls, color=st["color"], marker=st["marker"],
                        label=f"{run['label']} ({mode})")
    ax.axhline(20, ls=":", c="green", lw=1, label="QK-Clip unnecessary (<20)")
    ax.axhline(30, ls="--", c="red", lw=1, label="QK-Clip recommended (>30)")
    ax.set_xlabel("consumed tokens (B)")
    ax.set_ylabel("global max pre-softmax attention logit")
    ax.set_title("Measured max attention logit — v1 (explode) vs v2 (stable)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    p = os.path.join(plot_dir, "max_logit_vs_tokens.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    # 2) per-head max for the deepest attn layer at the final checkpoint of each run
    fig, ax = plt.subplots(figsize=(11, 5))
    for run in runs:
        st = style(run["group"])
        pts = [p for p in run["points"] if p.get("result", {}).get("random")]
        if not pts:
            continue
        last = pts[-1]["result"]["random"]["per_layer"]
        deepest = max(last.keys())
        heads = last[deepest]["per_head_max"]
        ax.plot(range(len(heads)), sorted(heads, reverse=True), color=st["color"],
                marker=st["marker"], label=f"{run['label']} L{deepest} (final)")
    ax.axhline(20, ls=":", c="green", lw=1); ax.axhline(30, ls="--", c="red", lw=1)
    ax.set_xlabel("head rank (sorted desc)")
    ax.set_ylabel("per-head max logit")
    ax.set_title("Per-head max logit, deepest attn layer, final checkpoint")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    p = os.path.join(plot_dir, "max_logit_per_head_final.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)
    return paths


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Attention-logit trajectory across Alpha checkpoints (v1 vs v2).")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--real-text", default=None, help="Path to a fixed real-text probe file.")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--num-samples", type=int, default=2, help="random-token samples per checkpoint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default=None)
    p.add_argument("--plot", default=None)
    p.add_argument("--max-checkpoints", type=int, default=0)
    return p.parse_args()


def subsample(items, k):
    if k <= 0 or len(items) <= k:
        return items
    if k == 1:
        return [items[-1]]
    idx = [round(i * (len(items) - 1) / (k - 1)) for i in range(k)]
    return [items[i] for i in sorted(set(idx))]


def main():
    args = parse_args()
    real_text = None
    if args.real_text and os.path.isfile(args.real_text):
        with open(args.real_text) as f:
            real_text = f.read()

    runs_out = []
    for i, run_dir in enumerate(args.runs):
        group = derive_group(run_dir)
        label = (args.labels[i] if args.labels and i < len(args.labels) else None) or group
        exports = subsample(find_hf_exports(run_dir), args.max_checkpoints)
        print(f"\n### {label} ({group}) — {len(exports)} HF exports under {run_dir}")
        points = []
        # tokenizer per run (real text encoded once per run; vocab differs across runs)
        real_ids = None
        if real_text is not None and exports:
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(exports[0][1], trust_remote_code=True)
                real_ids = tok(real_text, add_special_tokens=False)["input_ids"]
                print(f"  real-text tokens: {len(real_ids)} (tokenizer from {os.path.basename(exports[0][1])})")
            except Exception as e:
                print(f"  WARN: tokenizer load failed ({e}); skipping real-text mode")
        for it, hf_dir in exports:
            one_p, tokens = checkpoint_meta(run_dir, it, group)
            print(f"  iter {it} (1p={int(one_p)}, {tokens/1e9:.1f}B) loading {os.path.basename(hf_dir)} ...",
                  flush=True)
            pt = {"iter": it, "hf_dir": hf_dir, "apply_layernorm_1p": one_p,
                  "consumed_tokens": tokens, "result": None}
            try:
                pt["result"] = measure_checkpoint(
                    hf_dir, one_p, args.seq_len, args.num_samples, args.seed, real_ids, args.device)
                g = pt["result"]["random"]["global"]
                rl = pt["result"].get("real", {}).get("global", {})
                print(f"      random max={g['global_max']:.2f} (L{g['layer']} H{g['head']})"
                      + (f" | real max={rl.get('global_max', float('nan')):.2f}" if rl else ""))
            except Exception as e:
                pt["error"] = f"{type(e).__name__}: {e}"
                print(f"      ERROR: {pt['error']}")
                traceback.print_exc()
            points.append(pt)
        runs_out.append({"run_dir": run_dir, "label": label, "group": group, "points": points})

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"runs": runs_out,
                       "config": {"seq_len": args.seq_len, "num_samples": args.num_samples,
                                  "seed": args.seed, "real_text": args.real_text}}, f, indent=1)
        print(f"\nWrote JSON: {args.output}")

    if args.plot:
        paths = write_plots(args.plot, runs_out)
        print(f"Wrote {len(paths)} plots to {args.plot}")
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
