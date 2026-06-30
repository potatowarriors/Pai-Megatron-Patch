#!/usr/bin/env python3
"""
Attention-COLLAPSE analysis — the functional consequence of the v1 logit explosion.

The logit-trajectory study showed v1's deepest attention layer reaches max pre-softmax
logit ~1664. This script measures what that DID to the attention distribution: it captures
the POST-softmax attention probabilities per head and reports

  * mean max attention probability  (1.0 = fully collapsed onto a single key)
  * mean attention entropy (nats)    (0 = collapsed; high = diffuse)

A head whose logit exploded should show max-prob ≈ 1 and entropy ≈ 0 — i.e. it attends to
essentially one token and has stopped doing useful mixing ("silent degeneration": no NaN,
loss stays ~2.2, but the head is dead).

Measurement care:
  * causal mask makes EARLY query positions trivially low-entropy (few valid keys), so we
    average ONLY over the LATTER half of query positions (>= seq_len//2), where a high
    max-prob genuinely indicates collapse.
  * per-run AlphaRMSNorm convention patch (v1: 1+w, v2: w) and per-model dynamic-module
    derivation, identical to analyze_logit_trajectory.py.

Usage:
  python analyze_attention_collapse.py \
    --runs outputs/alpha_v1/alpha_baseline_48L_20251219_095156 \
           outputs/alpha_baseline_48L_stage1_resume_20260526_194537 \
    --iters 50000,200000,400000:45000 --seq-len 2048 \
    --output outputs/analysis/v1_vs_v2_logit_explosion/attention_collapse.json \
    --plot   outputs/analysis/v1_vs_v2_logit_explosion/plots
"""

import argparse
import gc
import os
import sys
import json
import math

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_logit_trajectory import (  # reuse convention patch + discovery + meta
    patch_rmsnorm_convention, find_hf_exports, checkpoint_meta,
)
from analyze_weight_trajectory import derive_group, GROUP_STYLE


def make_collapse_forward(original_fn, repeat_kv_fn, records, tail_frac=0.5):
    """Patched eager attention: record per-head post-softmax max-prob + entropy,
    averaged over the LATTER `tail_frac` of query positions (avoids causal-mask bias)."""
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        with torch.no_grad():
            key_e = repeat_kv_fn(key, module.num_key_value_groups)
            logits = (torch.matmul(query, key_e.transpose(2, 3)) * scaling).float()
            S = logits.shape[-1]
            causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=logits.device))
            logits = logits.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))
            probs = torch.softmax(logits, dim=-1)               # (B,H,Sq,Sk)
            start = int(S * (1.0 - tail_frac))
            p = probs[:, :, start:, :]                          # latter query positions
            maxp = p.amax(dim=-1)                               # (B,H,Sq')
            ent = -(p * torch.clamp(p, min=1e-12).log()).sum(-1)  # (B,H,Sq')
            li = module.layer_idx
            r = records.setdefault(li, {"max_prob": [], "entropy": []})
            r["max_prob"].append(maxp.mean(dim=(0, 2)).tolist())   # per-head mean
            r["entropy"].append(ent.mean(dim=(0, 2)).tolist())
        return original_fn(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    return patched


def _reduce(rec):
    out = {}
    for li in sorted(rec):
        nh = len(rec[li]["max_prob"][0])
        mp = [sum(s[h] for s in rec[li]["max_prob"]) / len(rec[li]["max_prob"]) for h in range(nh)]
        en = [sum(s[h] for s in rec[li]["entropy"]) / len(rec[li]["entropy"]) for h in range(nh)]
        out[li] = {"max_prob": mp, "entropy": en}
    return out


def measure(hf_dir, one_p, seq_len, num_samples, seed, device, tail_frac, real_ids=None):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        hf_dir, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device).eval()
    mod = sys.modules.get(type(model).__module__)
    norm_cls, norm_orig = patch_rmsnorm_convention(mod, one_p)
    orig_fn, repeat_kv = mod.eager_attention_forward, mod.repeat_kv
    vocab = model.config.vocab_size
    result = {}
    try:
        # random tokens
        rec = {}
        mod.eager_attention_forward = make_collapse_forward(orig_fn, repeat_kv, rec, tail_frac)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for _ in range(num_samples):
            ids = torch.randint(0, vocab, (1, seq_len), generator=gen).to(device)
            with torch.no_grad():
                model(ids)
        result["random"] = _reduce(rec)
        # real text (tiled/truncated to seq_len)
        if real_ids is not None:
            ids = real_ids[:seq_len]
            if len(ids) < seq_len:
                ids = (ids * ((seq_len + len(ids) - 1) // len(ids)))[:seq_len]
            ids_t = torch.clamp(torch.tensor([ids], device=device), 0, vocab - 1)
            rec2 = {}
            mod.eager_attention_forward = make_collapse_forward(orig_fn, repeat_kv, rec2, tail_frac)
            with torch.no_grad():
                model(ids_t)
            result["real"] = _reduce(rec2)
    finally:
        mod.eager_attention_forward = orig_fn
        if norm_cls is not None:
            norm_cls.forward = norm_orig
        del model
        gc.collect(); torch.cuda.empty_cache()
    return result


def parse_iters(spec):
    """'50000,200000,400000:45000' -> {run_idx: [iters]} keyed positionally by ':' groups."""
    groups = spec.split(":")
    return [[int(x) for x in g.split(",") if x] for g in groups]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--iters", default=None,
                    help="per-run iter lists separated by ':' e.g. 50000,200000,400000:45000")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--num-samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tail-frac", type=float, default=0.5)
    ap.add_argument("--real-text", default=None, help="fixed real-text probe file")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=None)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    real_text = open(args.real_text).read() if (args.real_text and os.path.isfile(args.real_text)) else None
    iter_groups = parse_iters(args.iters) if args.iters else [None] * len(args.runs)
    runs_out = []
    for i, run_dir in enumerate(args.runs):
        group = derive_group(run_dir)
        exports = dict(find_hf_exports(run_dir))
        want = iter_groups[i] if i < len(iter_groups) and iter_groups[i] else sorted(exports)
        print(f"\n### {group} — iters {want}")
        real_ids = None
        if real_text is not None and want and want[0] in exports:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(exports[want[0]], trust_remote_code=True)
            real_ids = tok(real_text, add_special_tokens=False)["input_ids"]
        pts = []
        for it in want:
            if it not in exports:
                print(f"  skip iter {it} (no hfmodel)"); continue
            one_p, tokens = checkpoint_meta(run_dir, it, group)
            print(f"  iter {it} (1p={int(one_p)}, {tokens/1e9:.1f}B) ...", flush=True)
            res = measure(exports[it], one_p, args.seq_len, args.num_samples, args.seed,
                          args.device, args.tail_frac, real_ids=real_ids)
            rnd = res["random"]; deepest = max(rnd)
            def summ(tag, perlayer):
                mp = perlayer[deepest]["max_prob"]; en = perlayer[deepest]["entropy"]
                n = len(mp)
                print(f"      [{tag}] L{deepest}: collapsed(>0.9)={sum(1 for v in mp if v>0.9)}/{n} "
                      f"meanMP={sum(mp)/n:.3f} maxMP={max(mp):.3f} meanEnt={sum(en)/n:.2f}")
            summ("random", rnd)
            if "real" in res:
                summ("real", res["real"])
            pts.append({"iter": it, "consumed_tokens": tokens, "deepest_layer": deepest,
                        "random": rnd, "real": res.get("real")})
        runs_out.append({"run_dir": run_dir, "group": group, "points": pts})

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        json.dump({"runs": runs_out}, open(args.output, "w"), indent=1)
        print(f"\nWrote {args.output}")

    if args.plot:
        plot(runs_out, args.plot)


def plot(runs, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(plot_dir, exist_ok=True)

    def mode_of(pt):
        return pt["real"] if pt.get("real") else pt["random"]

    # per-head max-prob (deepest layer) at the LAST point of each run, sorted desc
    fig, ax = plt.subplots(figsize=(11, 5))
    for run in runs:
        if not run["points"]:
            continue
        st = GROUP_STYLE.get(run["group"], dict(color="#7f7f7f", marker="x"))
        last = run["points"][-1]
        pl = mode_of(last)[last["deepest_layer"]]
        mp = sorted(pl["max_prob"], reverse=True)
        ax.plot(range(len(mp)), mp, color=st["color"], marker=st["marker"],
                label=f"{run['group']} L{last['deepest_layer']} ({last['consumed_tokens']/1e9:.0f}B)")
    ax.axhline(0.9, ls="--", c="red", lw=1, label="collapse (max-prob>0.9)")
    ax.set_xlabel("head rank (sorted desc)"); ax.set_ylabel("mean max attention prob (latter half)")
    ax.set_title("Attention collapse: per-head max softmax prob, deepest layer, final ckpt (real-text)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, 1.02)
    p = os.path.join(plot_dir, "attention_collapse_per_head.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)

    # collapse trajectory: max max-prob (deepest layer) vs tokens, random vs real
    fig, ax = plt.subplots(figsize=(11, 5))
    for run in runs:
        st = GROUP_STYLE.get(run["group"], dict(color="#7f7f7f", marker="x"))
        for key, ls in (("random", "--"), ("real", "-")):
            xs, ys = [], []
            for pt in run["points"]:
                src = pt.get(key)
                if not src:
                    continue
                xs.append(pt["consumed_tokens"] / 1e9)
                ys.append(max(src[pt["deepest_layer"]]["max_prob"]))
            if xs:
                ax.plot(xs, ys, color=st["color"], ls=ls, marker=st["marker"],
                        label=f"{run['group']} ({key})")
    ax.axhline(0.9, ls=":", c="red", lw=1)
    ax.set_xlabel("consumed tokens (B)"); ax.set_ylabel("max over heads of mean max-prob (deepest L)")
    ax.set_title("Collapse onset: peak per-head attention concentration vs tokens")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, 1.02)
    p2 = os.path.join(plot_dir, "attention_collapse_onset.png")
    fig.tight_layout(); fig.savefig(p2, dpi=120); plt.close(fig)
    print(f"Wrote plots: {p}, {p2}")


if __name__ == "__main__":
    main()
