#!/usr/bin/env python3
"""
Attention-PATTERN analysis — tests the "attention sink" hypothesis for the v1 explosion.

Open puzzles from the causal study: (1) why does ONLY the deepest attention layer (L23)
explode, and (2) why is it functionally benign (entropy unchanged, loss ~2.2)? A single
mechanism would explain both: if the exploded heads are **attention sinks** (nearly all
query positions dump attention onto a fixed "junk" key, typically the first token), then
the sink key aligns with every query → huge logits, yet the head is ~no-op → benign, and
sinks characteristically form in deeper layers.

This script captures, per attention head (averaged over the latter half of query positions
to avoid causal-mask bias):
  * sink_mass     = mean attention prob on key position 0      (the canonical sink)
  * sink_mass4    = mean prob on keys 0..3
  * frac_amax_sink= fraction of queries whose ARGMAX key is position 0
  * local_mass    = mean prob within a width-W diagonal window [q-W+1 .. q]  (induction/local)
  * self_mass     = mean prob on the query's own position (key == q)
  * entropy       = mean attention entropy (nats)

Per-run AlphaRMSNorm convention patch + per-model dynamic-module derivation (see
analyze_logit_trajectory.py).

Usage:
  python analyze_attention_pattern.py \
    --runs outputs/alpha_v1/alpha_baseline_48L_20251219_095156 \
           outputs/alpha_baseline_48L_stage1_resume_20260526_194537 \
    --iters "200000,400000:45000" --real-text /tmp/probe.txt --seq-len 2048 \
    --layers 11,23 \
    --output outputs/analysis/v1_vs_v2_logit_explosion/attention_pattern.json
"""

import argparse
import gc
import os
import sys
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_logit_trajectory import patch_rmsnorm_convention, find_hf_exports, checkpoint_meta
from analyze_attention_collapse import parse_iters
from analyze_weight_trajectory import derive_group


def make_pattern_forward(original_fn, repeat_kv_fn, records, tail_frac, window):
    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        with torch.no_grad():
            key_e = repeat_kv_fn(key, module.num_key_value_groups)
            logits = (torch.matmul(query, key_e.transpose(2, 3)) * scaling).float()
            S = logits.shape[-1]
            dev = logits.device
            causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=dev))
            logits = logits.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))
            probs = torch.softmax(logits, dim=-1)            # (B,H,Sq,Sk)
            start = int(S * (1.0 - tail_frac))
            qs = torch.arange(start, S, device=dev)          # absolute query indices
            p = probs[:, :, start:, :]                       # (B,H,Sq',Sk)
            B, H, Sq, Sk = p.shape

            sink = p[:, :, :, 0].mean(dim=(0, 2))            # (H,)
            sink4 = p[:, :, :, :4].sum(-1).mean(dim=(0, 2))
            amax = p.argmax(-1)                              # (B,H,Sq')
            amax_sink = (amax == 0).float().mean(dim=(0, 2))
            # self (diagonal) mass
            selfmass = p.gather(-1, qs.view(1, 1, -1, 1).expand(B, H, Sq, 1)).mean(dim=(0, 2, 3))
            # local window band [q-W+1, q]
            kidx = torch.arange(Sk, device=dev).view(1, -1)
            band = ((kidx <= qs.view(-1, 1)) & (kidx > qs.view(-1, 1) - window)).float()  # (Sq',Sk)
            local = (p * band.view(1, 1, Sq, Sk)).sum(-1).mean(dim=(0, 2))
            ent = -(p * torch.clamp(p, min=1e-12).log()).sum(-1).mean(dim=(0, 2))

            li = module.layer_idx
            rec = records.setdefault(li, {k: [] for k in
                  ("sink", "sink4", "amax_sink", "self", "local", "entropy")})
            rec["sink"].append(sink.tolist())
            rec["sink4"].append(sink4.tolist())
            rec["amax_sink"].append(amax_sink.tolist())
            rec["self"].append(selfmass.tolist())
            rec["local"].append(local.tolist())
            rec["entropy"].append(ent.tolist())
        return original_fn(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    return patched


def _avg(rec):
    out = {}
    for li, d in rec.items():
        nh = len(d["sink"][0])
        out[li] = {k: [sum(s[h] for s in d[k]) / len(d[k]) for h in range(nh)] for k in d}
    return out


def measure(hf_dir, one_p, seq_len, num_samples, seed, device, tail_frac, window, real_ids):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        hf_dir, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", low_cpu_mem_usage=True).to(device).eval()
    mod = sys.modules.get(type(model).__module__)
    norm_cls, norm_orig = patch_rmsnorm_convention(mod, one_p)
    orig_fn, repeat_kv = mod.eager_attention_forward, mod.repeat_kv
    vocab = model.config.vocab_size
    result = {}
    try:
        if real_ids is not None:
            ids = real_ids[:seq_len]
            if len(ids) < seq_len:
                ids = (ids * ((seq_len + len(ids) - 1) // len(ids)))[:seq_len]
            ids_t = torch.clamp(torch.tensor([ids], device=device), 0, vocab - 1)
            rec = {}
            mod.eager_attention_forward = make_pattern_forward(orig_fn, repeat_kv, rec, tail_frac, window)
            with torch.no_grad():
                model(ids_t)
            result["real"] = _avg(rec)
        rec2 = {}
        mod.eager_attention_forward = make_pattern_forward(orig_fn, repeat_kv, rec2, tail_frac, window)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for _ in range(num_samples):
            ids = torch.randint(0, vocab, (1, seq_len), generator=gen).to(device)
            with torch.no_grad():
                model(ids)
        result["random"] = _avg(rec2)
    finally:
        mod.eager_attention_forward = orig_fn
        if norm_cls is not None:
            norm_cls.forward = norm_orig
        del model
        gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--iters", default=None)
    ap.add_argument("--layers", default=None, help="comma-sep HF layer indices to print (default all attn)")
    ap.add_argument("--real-text", default=None)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tail-frac", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    real_text = open(args.real_text).read() if (args.real_text and os.path.isfile(args.real_text)) else None
    want_layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    iter_groups = parse_iters(args.iters) if args.iters else [None] * len(args.runs)

    runs_out = []
    for i, run_dir in enumerate(args.runs):
        group = derive_group(run_dir)
        exports = dict(find_hf_exports(run_dir))
        want = iter_groups[i] if i < len(iter_groups) and iter_groups[i] else sorted(exports)
        real_ids = None
        if real_text is not None and want and want[0] in exports:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(exports[want[0]], trust_remote_code=True)
            real_ids = tok(real_text, add_special_tokens=False)["input_ids"]
        print(f"\n### {group} — iters {want}")
        pts = []
        for it in want:
            if it not in exports:
                continue
            one_p, tokens = checkpoint_meta(run_dir, it, group)
            print(f"  iter {it} (1p={int(one_p)}, {tokens/1e9:.1f}B) ...", flush=True)
            res = measure(exports[it], one_p, args.seq_len, args.num_samples, args.seed,
                          args.device, args.tail_frac, args.window, real_ids)
            mode = "real" if "real" in res else "random"
            data = res[mode]
            layers = want_layers or sorted(data)
            for L in layers:
                if L not in data:
                    continue
                d = data[L]
                nh = len(d["sink"])
                print(f"    [{mode}] L{L} ({nh} heads):  "
                      f"sink_mass μ={sum(d['sink'])/nh:.2f} max={max(d['sink']):.2f} | "
                      f"argmax@0 μ={sum(d['amax_sink'])/nh:.2f} | "
                      f"local(W{args.window}) μ={sum(d['local'])/nh:.2f} | "
                      f"ent μ={sum(d['entropy'])/nh:.2f}")
                if nh == 32:  # v1 GQA group split
                    for tag, sl in (("g0[0-15]", slice(0, 16)), ("g1[16-31]", slice(16, 32))):
                        print(f"        {tag}: sink μ={sum(d['sink'][sl])/16:.2f} "
                              f"argmax@0 μ={sum(d['amax_sink'][sl])/16:.2f} "
                              f"local μ={sum(d['local'][sl])/16:.2f} ent μ={sum(d['entropy'][sl])/16:.2f}")
            pts.append({"iter": it, "consumed_tokens": tokens, "mode": mode, "pattern": data})
        runs_out.append({"run_dir": run_dir, "group": group, "points": pts})

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        json.dump({"runs": runs_out, "window": args.window}, open(args.output, "w"), indent=1)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
