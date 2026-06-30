#!/usr/bin/env python3
"""Analyze an nsys trace for training-throughput optimization.

Computes, from a single captured step, the quantities you actually need to
decide *what is optimizable* — things the Nsight Systems GUI makes you eyeball:

  1. Wall-time coverage (sweepline): idle / compute-only / comm-only(EXPOSED) /
     hidden(comm+compute). "comm-only" is the recoverable wall time if comm were
     perfectly overlapped.
  2. Communication exposure by NCCL op type (SendRecv / AllReduce / AllGather):
     how much of each is exposed vs hidden behind compute.
  3. Kernel-size distribution: the launch-bound tail (many tiny kernels).
  4. Optimizer NVTX range (e.g. Muon Newton-Schulz): span, kernel count, and
     in-range GPU idle (launch-bound waste).

Usage:
    python tools/analyze_nsys_trace.py <trace.nsys-rep | trace.sqlite> \
        [--optimizer-nvtx TensorParallelMuon]

If given a .nsys-rep it auto-exports to .sqlite next to it (needs `nsys` on PATH).
Classification of comm vs compute is by kernel name ('nccl*' => comm), so it is
robust to stream/context-id quirks (per-stream sums are NOT trustworthy because
CUPTI streamId collides across contexts).

See docs/throughput_optimization.md for how to read the output and which levers
each number points to.
"""
import argparse, bisect, os, subprocess, sqlite3, sys
from collections import defaultdict


def ensure_sqlite(path: str) -> str:
    if path.endswith(".sqlite"):
        return path
    if not path.endswith(".nsys-rep"):
        sys.exit(f"expected .nsys-rep or .sqlite, got {path}")
    sq = path[: -len(".nsys-rep")] + ".sqlite"
    if not os.path.exists(sq):
        print(f"[exporting {os.path.basename(path)} -> sqlite ...]", file=sys.stderr)
        subprocess.run(
            ["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
             "-o", sq, path], check=True)
    return sq


def comm_group(name: str):
    n = name.lower()
    if "sendrecv" in n: return "SendRecv (EP all-to-all)"
    if "allgather" in n: return "AllGather (param/optim)"
    if "allreduce" in n: return "AllReduce (grad sync)"
    if "reducescatter" in n: return "ReduceScatter (grad sync)"
    if "nccl" in n: return "other-nccl"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="path to .nsys-rep or .sqlite")
    ap.add_argument("--optimizer-nvtx", default="TensorParallelMuon",
                    help="substring of the optimizer-step NVTX range to profile")
    args = ap.parse_args()

    c = sqlite3.connect(ensure_sqlite(args.trace))
    cur = c.cursor()
    sid = {i: v for i, v in cur.execute("SELECT id,value FROM StringIds")}
    def nm(x):
        v = sid.get(x);
        return v if (v and not str(v).isdigit()) else str(x)

    rows = cur.execute(
        "SELECT start,end,demangledName,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchall()
    if not rows:
        sys.exit("no kernels in trace (was --profile-ranks capturing this rank?)")

    def kname(r):
        n = nm(r[2])
        return n if (n and not n.isdigit()) else nm(r[3])

    comp, comm = [], []
    for s, e, dn, sn in rows:
        g = comm_group(kname((s, e, dn, sn)))
        (comm if g else comp).append((s, e, g) if g else (s, e))

    span = max(r[1] for r in rows) - min(r[0] for r in rows)
    comm_sum = sum(e - s for s, e, _ in comm)
    comp_sum = sum(e - s for s, e in comp)

    # ---- sweepline: exact wall-time coverage ----
    ev = []
    for s, e, _g in comm: ev += [(s, 1, 1), (e, -1, 1)]
    for s, e in comp:     ev += [(s, 1, 0), (e, -1, 0)]
    ev.sort()
    ncomp = ncomm = 0; prev = None
    idle = comp_only = comm_only = both = 0
    for t, d, k in ev:
        if prev is not None and t > prev:
            seg = t - prev
            if ncomp and ncomm: both += seg
            elif ncomp: comp_only += seg
            elif ncomm: comm_only += seg
            else: idle += seg
        if k == 0: ncomp += d
        else: ncomm += d
        prev = t
    tot = idle + comp_only + comm_only + both

    def line(lbl, v): print(f"  {lbl:<26}{v/1e6:9.1f} ms  ({100*v/tot:5.1f}%)")
    print(f"\n# Wall-time coverage  (span = {span/1e6:.1f} ms, {len(rows):,} kernels)")
    line("GPU idle (bubble)", idle)
    line("compute only", comp_only)
    line("comm ONLY (EXPOSED)", comm_only)
    line("comm+compute (hidden)", both)
    line("GPU busy (any)", tot - idle)
    if comm_only + both:
        print(f"  -> {100*comm_only/(comm_only+both):.0f}% of comm wall is EXPOSED")

    # ---- per-comm-type exposure (vs merged compute union) ----
    comp.sort()
    union = []
    for s, e in comp:
        if union and s <= union[-1][1]:
            if e > union[-1][1]: union[-1] = (union[-1][0], e)
        else: union.append((s, e))
    ust = [u[0] for u in union]; uen = [u[1] for u in union]
    def exposed(s, e):
        ov = 0; j = bisect.bisect_right(ust, e) - 1
        while j >= 0 and uen[j] > s:
            ov += max(0, min(e, uen[j]) - max(s, ust[j])); j -= 1
        return (e - s) - ov
    tt = defaultdict(float); ee = defaultdict(float); cc = defaultdict(int)
    for s, e, g in comm:
        tt[g] += e - s; ee[g] += exposed(s, e); cc[g] += 1
    print(f"\n# Communication by type (exposed = kernel-time not overlapped by compute)")
    print(f"  {'type':<28}{'count':>8}{'total ms':>11}{'exposed ms':>12}{'exp%':>7}")
    for g in sorted(tt, key=lambda k: -ee[k]):
        print(f"  {g:<28}{cc[g]:>8,}{tt[g]/1e6:>11.1f}{ee[g]/1e6:>12.1f}{100*ee[g]/tt[g]:>6.0f}%")
    print(f"  (comm kernel-time {comm_sum/1e6:.0f} ms | compute kernel-time {comp_sum/1e6:.0f} ms)")

    # ---- kernel-size distribution (launch-bound tail) ----
    durs = sorted(e - s for s, e, *_ in rows)
    n = len(durs); allt = sum(durs)
    print(f"\n# Kernel-size distribution (launch-bound tail)")
    for th in (2000, 5000, 10000):
        k = bisect.bisect_right(durs, th)
        print(f"  < {th//1000:>2} us: {k:>8,} kernels ({100*k/n:2.0f}%)  "
              f"{sum(durs[:k])/1e6:7.0f} ms ({100*sum(durs[:k])/allt:2.0f}% of time)")
    print(f"  median {durs[n//2]/1000:.1f} us | mean {allt/n/1000:.1f} us")

    # ---- optimizer NVTX range (launch-bound waste) ----
    cols = [r[1] for r in cur.execute("PRAGMA table_info(NVTX_EVENTS)")]
    if "text" in cols:
        ev = [(s, e) for s, e in cur.execute(
            "SELECT start,end FROM NVTX_EVENTS WHERE text LIKE ?",
            (f"%{args.optimizer_nvtx}%",)) if e and s and e > s]
        if ev:
            s0, e0 = max(ev, key=lambda x: x[1] - x[0]); rng = e0 - s0
            kin = sorted((s, e) for s, e, *_ in rows if s >= s0 and e <= e0)
            u = []
            for s, e in kin:
                if u and s <= u[-1][1]: u[-1][1] = max(u[-1][1], e)
                else: u.append([s, e])
            busy = sum(b - a for a, b in u)
            print(f"\n# Optimizer NVTX range  (match: '{args.optimizer_nvtx}')")
            print(f"  span {rng/1e6:.0f} ms | {len(kin):,} kernels | "
                  f"GPU busy {busy/1e6:.0f} ms | idle {(rng-busy)/1e6:.0f} ms "
                  f"({100*(rng-busy)/rng:.0f}% launch-bound)")
    print()


if __name__ == "__main__":
    main()
