#!/usr/bin/env python
"""FlashQLA vs fla(Triton) vs torch-oracle benchmark for Alpha's GDN kernel.

Compares three backends of chunk_gated_delta_rule on Alpha's exact shapes:
  - fla (flash-linear-attention, Triton)  — current production backend
  - flash_qla (FlashQLA, TileLang)        — candidate backend
  - torch_chunk_gated_delta_rule          — fp32 correctness oracle
    (backends/megatron/Megatron-LM-251125/megatron/core/ssm/gated_delta_net.py)

Alpha GDN facts mirrored here (megatron_patch/model/qwen3_next/gated_deltanet.py:355):
  num_k_heads=16, num_v_heads=32 (GQA 2:1), head_k_dim=head_v_dim=128,
  q/k/v bf16, g fp32, beta bf16, use_qk_l2norm_in_kernel=True, no initial_state.
  Production call repeat_interleaves q/k to 32 heads before the kernel.

Scenarios cover P3 today (seq 4096, b=3) and the LC phases with the gdn-cp
head-split design (each CP rank sees the FULL sequence and heads/cp heads).

Safety: refuses to run on a GPU that is already in use (live training guard).

Usage (from repo root, needs a FREE H100):
  PYTHONPATH=/home/work/vidsearch/envs/flashqla_poc/pylibs \
    python examples/alpha/study/flashqla_bench.py --device cuda:0
  # subset:
  ... --scenarios p3_today lc_a_cp8 --skip-oracle
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MEGATRON_DIR = os.path.join(REPO_ROOT, "backends", "megatron", "Megatron-LM-251125")
for p in (REPO_ROOT, MEGATRON_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- backends

def load_backends(flashqla_path):
    if flashqla_path and flashqla_path not in sys.path:
        sys.path.insert(0, flashqla_path)

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_cgdr
    import fla as fla_pkg

    from flash_qla import chunk_gated_delta_rule as qla_cgdr
    import flash_qla as qla_pkg

    oracle = None
    try:
        from megatron.core.ssm.gated_delta_net import torch_chunk_gated_delta_rule as oracle
    except Exception as e:  # noqa: BLE001
        print(f"[warn] oracle import failed ({type(e).__name__}: {e}); "
              f"correctness checks vs oracle will be skipped")
    return fla_pkg, fla_cgdr, qla_pkg, qla_cgdr, oracle


# ---------------------------------------------------------------- scenarios

@dataclass
class Scenario:
    name: str
    seq: int
    batch: int
    num_v_heads: int          # HV seen by this (possibly CP-sharded) rank
    num_k_heads: int          # H  seen by this rank (before GQA expansion)
    note: str = ""
    check_oracle: bool = False   # oracle is O(seq * chunk^2) python-loop; keep short


HEAD_K_DIM = 128
HEAD_V_DIM = 128

SCENARIOS = [
    # P3 today: seq 4096, MBS=3, full heads.
    Scenario("p3_today", 4096, 3, 32, 16, "current stage2/P3 shape", check_oracle=True),
    # LC-A 32K single-node CP variants (gdn-cp head-split: full seq, heads/cp).
    Scenario("lc_a_cp1", 32768, 1, 32, 16, "32K no CP"),
    Scenario("lc_a_cp4", 32768, 1, 8, 4, "32K, CP=4 rank-local"),
    Scenario("lc_a_cp8", 32768, 1, 4, 2, "32K, CP=8 rank-local"),
    # LC-B 128K.
    Scenario("lc_b_cp8", 131072, 1, 4, 2, "128K, CP=8 rank-local"),
    # Small correctness-focused case (fast oracle, incl. gradients).
    Scenario("corr_small", 2048, 2, 32, 16, "correctness focus", check_oracle=True),
]


def make_inputs(sc: Scenario, device, seed=1234, near_zero_gate=False):
    """Alpha-realistic inputs. q/k/v bf16, g fp32 negative, beta bf16 in (0,1)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    b, s = sc.batch, sc.seq
    kw = dict(device=device, generator=gen)
    q = torch.randn(b, s, sc.num_k_heads, HEAD_K_DIM, dtype=torch.float32, **kw).to(torch.bfloat16)
    k = torch.randn(b, s, sc.num_k_heads, HEAD_K_DIM, dtype=torch.float32, **kw).to(torch.bfloat16)
    v = (torch.randn(b, s, sc.num_v_heads, HEAD_V_DIM, dtype=torch.float32, **kw) * 0.5).to(torch.bfloat16)
    beta = torch.rand(b, s, sc.num_v_heads, dtype=torch.float32, **kw).clamp(0.05, 0.95).to(torch.bfloat16)
    if near_zero_gate:
        # worst case for AutoCP's decay-based warmup: almost no forgetting
        g = -1e-4 * torch.rand(b, s, sc.num_v_heads, dtype=torch.float32, **kw)
    else:
        # matches alpha's g = -exp(A_log)*softplus(...): negative, O(1) magnitude
        g = -F.softplus(torch.randn(b, s, sc.num_v_heads, dtype=torch.float32, **kw))
    return q, k, v, g, beta


def expand_gqa(q, k, hv):
    """Mirror production repeat_interleave (gated_deltanet.py:351-353)."""
    rep = hv // q.shape[2]
    if rep > 1:
        q = q.repeat_interleave(rep, dim=2)
        k = k.repeat_interleave(rep, dim=2)
    return q, k


# ---------------------------------------------------------------- timing

def _sync(dev):
    torch.cuda.synchronize(dev)


def time_fn(fn, dev, warmup=5, iters=20):
    """Median ms via CUDA events. Returns (median_ms, first_call_s)."""
    t0 = time.perf_counter()
    fn()
    _sync(dev)
    first_call_s = time.perf_counter() - t0        # includes TileLang JIT compile
    for _ in range(warmup):
        fn()
    _sync(dev)
    times = []
    for _ in range(iters):
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        fn()
        ev1.record()
        _sync(dev)
        times.append(ev0.elapsed_time(ev1))
    times.sort()
    return times[len(times) // 2], first_call_s


def bench_call(call, mk_inputs, dev, mode, warmup, iters):
    """mode: 'fwd' (no_grad) or 'fwdbwd' (autograd through q,k,v,g,beta)."""
    torch.cuda.reset_peak_memory_stats(dev)
    if mode == "fwd":
        inputs = mk_inputs()
        def fn():
            with torch.no_grad():
                call(*inputs)
    else:
        inputs = tuple(t.detach().requires_grad_(True) for t in mk_inputs())
        def fn():
            o = call(*inputs)
            o.backward(torch.ones_like(o))
            for t in inputs:
                t.grad = None
    ms, first_s = time_fn(fn, dev, warmup, iters)
    peak_gb = torch.cuda.max_memory_allocated(dev) / 2**30
    return {"ms": round(ms, 3), "first_call_s": round(first_s, 2), "peak_gb": round(peak_gb, 2)}


# ---------------------------------------------------------------- correctness

def max_err(a, b):
    d = (a.float() - b.float()).abs()
    denom = b.float().abs().clamp_min(1e-3)
    return d.max().item(), (d / denom).max().item()


def correctness(sc, dev, fla_cgdr, qla_cgdr, oracle, near_zero_gate=False):
    """fwd + grad comparison on expanded-GQA layout (oracle needs equal heads)."""
    tag = sc.name + ("+g~0" if near_zero_gate else "")
    q0, k0, v0, g0, b0 = make_inputs(sc, dev, near_zero_gate=near_zero_gate)
    qx, kx = expand_gqa(q0, k0, sc.num_v_heads)
    out = {"case": tag}

    def run(call, **kw):
        ins = tuple(t.detach().clone().requires_grad_(True) for t in (qx, kx, v0, g0, b0))
        o = call(ins[0], ins[1], ins[2], g=ins[3], beta=ins[4],
                 initial_state=None, output_final_state=False,
                 use_qk_l2norm_in_kernel=True, **kw)
        if isinstance(o, tuple):
            o = o[0]
        o.backward(torch.ones_like(o))
        return o.detach(), [t.grad.detach().clone() for t in ins]

    o_fla, g_fla = run(fla_cgdr)
    o_qla, g_qla = run(qla_cgdr)
    out["qla_vs_fla_fwd"] = max_err(o_qla, o_fla)
    out["qla_vs_fla_grads"] = [max_err(a, b) for a, b in zip(g_qla, g_fla)]

    if oracle is not None and sc.check_oracle:
        ins = tuple(t.detach().clone().float().requires_grad_(True) for t in (qx, kx, v0, g0, b0))
        o_ref = oracle(ins[0], ins[1], ins[2], ins[3], ins[4],
                       chunk_size=64, use_qk_l2norm_in_kernel=True)
        if isinstance(o_ref, tuple):
            o_ref = o_ref[0]
        o_ref.backward(torch.ones_like(o_ref))
        g_ref = [t.grad.detach().clone() for t in ins]
        out["fla_vs_oracle_fwd"] = max_err(o_fla, o_ref)
        out["qla_vs_oracle_fwd"] = max_err(o_qla, o_ref)
        out["fla_vs_oracle_grads"] = [max_err(a, b) for a, b in zip(g_fla, g_ref)]
        out["qla_vs_oracle_grads"] = [max_err(a, b) for a, b in zip(g_qla, g_ref)]
    return out


# ---------------------------------------------------------------- safety

def assert_gpu_free(dev, force):
    free_b, total_b = torch.cuda.mem_get_info(dev)
    used_gb = (total_b - free_b) / 2**30
    if used_gb > 2.0 and not force:
        raise SystemExit(
            f"[abort] {dev} already has {used_gb:.1f} GiB in use — a training job may be live. "
            f"Pick a free GPU with --device or pass --force if you are sure."
        )


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--flashqla-path", default="/home/work/vidsearch/envs/flashqla_poc/pylibs")
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help=f"subset of {[s.name for s in SCENARIOS]}")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--skip-bench", action="store_true", help="correctness only")
    ap.add_argument("--force", action="store_true", help="run even if GPU busy")
    ap.add_argument("--out", default=None, help="results json path")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    assert_gpu_free(dev, args.force)

    fla_pkg, fla_cgdr, qla_pkg, qla_cgdr, oracle = load_backends(args.flashqla_path)
    if args.skip_oracle:
        oracle = None
    print(f"fla {fla_pkg.__version__} | flash_qla {qla_pkg.__version__} | "
          f"torch {torch.__version__} | {torch.cuda.get_device_name(dev)}")

    chosen = [s for s in SCENARIOS if args.scenarios is None or s.name in args.scenarios]
    results = {"env": {"torch": torch.__version__, "fla": fla_pkg.__version__,
                       "flash_qla": qla_pkg.__version__,
                       "gpu": torch.cuda.get_device_name(dev)},
               "correctness": [], "bench": []}

    # -------- correctness (small cases + AutoCP worst-case gate)
    for sc in chosen:
        if not sc.check_oracle and oracle is not None:
            continue
        for nzg in (False, True):
            if sc.seq > 8192:
                continue
            r = correctness(sc, dev, fla_cgdr, qla_cgdr, oracle, near_zero_gate=nzg)
            results["correctness"].append(r)
            print(f"[corr] {r['case']}: qla_vs_fla fwd maxabs/rel={r['qla_vs_fla_fwd']}")
            for kkey in ("fla_vs_oracle_fwd", "qla_vs_oracle_fwd"):
                if kkey in r:
                    print(f"       {kkey}: maxabs/rel={r[kkey]}")

    # -------- performance
    if not args.skip_bench:
        for sc in chosen:
            q0, k0, v0, g0, b0 = make_inputs(sc, dev)
            qx, kx = expand_gqa(q0, k0, sc.num_v_heads)

            variants = {
                # production-equivalent: expanded GQA, fla defaults
                "fla_expanded": (fla_cgdr, (qx, kx, v0, g0, b0), {}),
                "qla_expanded": (qla_cgdr, (qx, kx, v0, g0, b0), {}),
                # FlashQLA-native GVA (skip q/k head replication entirely)
                "qla_gva": (qla_cgdr, (q0, k0, v0, g0, b0), {}),
                # isolate AutoCP contribution
                "qla_expanded_nocp": (qla_cgdr, (qx, kx, v0, g0, b0), {"auto_cp": False}),
            }
            # fla GVA support is version-dependent — probe once
            try:
                with torch.no_grad():
                    fla_cgdr(q0[:, :256], k0[:, :256], v0[:, :256], g=g0[:, :256],
                             beta=b0[:, :256], use_qk_l2norm_in_kernel=True)
                variants["fla_gva"] = (fla_cgdr, (q0, k0, v0, g0, b0), {})
            except Exception:
                pass

            for vname, (call, tensors, extra) in variants.items():
                def mk():
                    return tuple(t.detach().clone() for t in tensors)
                def wrapped(q_, k_, v_, g_, b_):
                    return call(q_, k_, v_, g=g_, beta=b_, initial_state=None,
                                output_final_state=False,
                                use_qk_l2norm_in_kernel=True, **extra)[0]
                row = {"scenario": sc.name, "seq": sc.seq, "batch": sc.batch,
                       "hv": sc.num_v_heads, "hk": sc.num_k_heads, "variant": vname}
                try:
                    row["fwd"] = bench_call(wrapped, mk, dev, "fwd", args.warmup, args.iters)
                    row["fwdbwd"] = bench_call(wrapped, mk, dev, "fwdbwd", args.warmup, args.iters)
                except Exception as e:  # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {e}"
                results["bench"].append(row)
                msg = row.get("error") or (f"fwd {row['fwd']['ms']}ms  "
                                           f"fwd+bwd {row['fwdbwd']['ms']}ms  "
                                           f"peak {row['fwdbwd']['peak_gb']}GB")
                print(f"[bench] {sc.name:10s} {vname:18s} {msg}")
                torch.cuda.empty_cache()

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"flashqla_bench_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults written to {out_path}")


if __name__ == "__main__":
    main()
