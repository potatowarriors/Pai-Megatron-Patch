#!/usr/bin/env python3
"""
bestfit_pack.py — Offline Best-fit Packing of a tokenized Megatron IndexedDataset
to MINIMIZE document truncation.

Implements "Best-fit Packing" (BFP) from the paper
    "Fewer Truncations Improve Language Modeling" (Ding et al., ICML 2024,
     arXiv:2404.10830).

Background — what truncation is and why this helps
--------------------------------------------------
Megatron pretraining uses *concatenate-and-chunk*: at train time `GPTDataset`
glues the whole (shuffled) document stream end-to-end and slices it into fixed
`seq_length` windows (`helpers.cpp::build_sample_idx`). Any document that
straddles a window boundary is split across two training samples — a truncation.
The paper shows this hurts (hallucinated/undefined references in code, broken
facts, lost long-range structure) and fixes it by *bin-packing* whole documents
into sequences instead.

This tool is the OFFLINE realization of that fix for Megatron. It does NOT touch
Megatron core. Given a tokenized `<prefix>_text_document.{bin,idx}` it:
  1. reads the per-document length array from the `.idx` (cheap; no `.bin` read),
  2. runs Best-Fit-Decreasing (BFD) bin packing with bin capacity L = seq_length,
     accelerated by a segment tree (O(N log L)),
  3. re-emits a NEW `.bin/.idx` where every "document" is exactly one packed bin
     of L tokens (real docs concatenated, then EOD-padded up to L).

Because every emitted document is exactly L tokens and the model trains with
`--seq-length L`, Megatron's concat-and-chunk then slices EXACTLY on bin
boundaries — so a training sample's input is one whole bin and documents are
no longer split (except genuinely-long docs > L, which any method must split).

Why padding with EOD (not a distinct pad id)
--------------------------------------------
Alpha trains with `--eod-mask-loss` + `--reset-attention-mask`, EOD id = 0.
Padding each bin up to L with EOD (id 0) means:
  - `--eod-mask-loss` masks the entire pad tail from the loss
    (`loss_mask[data==eod]=0`, gpt_dataset.py),
  - `--reset-attention-mask` isolates each pad position (data==eod is a boundary).
A distinct pad id would NOT be masked (Megatron's own _pad_token_id net keys off
its own id, not ours) → the model would train to predict pad tokens. So EOD-pad
is the correct (and only) choice for this flag set.

Critical invariant
------------------
Each bin is written with `add_document(bin_arr, [L])` — a SINGLE-element length
list. `GPTDataset` packs at the *sequence* level and ignores `document_indices`;
passing the per-doc length list would make each doc its own sequence again and
silently undo all packing. This is asserted at emit time.

Scope
-----
Run PER DATASET (never across datasets) — Megatron's BlendedDataset samples whole
sequences from one constituent, so per-dataset packing is preserved through the
blend; cross-dataset packing would corrupt blend ratios.

Usage
-----
    # dry-run: report packing stats (truncation drop, fill ratio) without writing
    python bestfit_pack.py --input  /path/<prefix> --dry-run

    # write the packed dataset
    python bestfit_pack.py --input  /path/in_prefix \\
                           --output /path/out_prefix \\
                           --seq-length 4096 --eod 0

`--input` / `--output` are prefixes BEFORE `_text_document.{bin,idx}` (same
convention as merge_indices.py). For the over-length-doc note and the
`--strict-eod` option see the argument help and examples/alpha/CLAUDE.md.
"""

import argparse
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Segment tree: successor query over remaining-capacity values [0..cap].
# Supports point add/remove of "a bin with remaining capacity v" and
# find_ge(s) = smallest remaining capacity >= s that has at least one bin.
# Iterative (canonical-cover walk + descend) — no per-item recursion.
# ---------------------------------------------------------------------------
class _SegTreeSuccessor:
    __slots__ = ("cap", "S", "t")

    def __init__(self, cap: int):
        self.cap = cap
        S = 1
        while S < cap + 1:
            S <<= 1
        self.S = S
        self.t = [0] * (2 * S)

    def add(self, v: int, delta: int) -> None:
        t = self.t
        k = self.S + v
        t[k] += delta
        k >>= 1
        while k:
            t[k] = t[2 * k] + t[2 * k + 1]
            k >>= 1

    def find_ge(self, s: int) -> int:
        """Smallest value v in [s, cap] with count > 0, else -1."""
        cap = self.cap
        if s > cap:
            return -1
        if s < 0:
            s = 0
        t = self.t
        S = self.S
        # Canonical cover of [s, cap], gathered left-to-right.
        l = s + S
        r = cap + 1 + S
        left = []
        right = []
        while l < r:
            if l & 1:
                left.append(l)
                l += 1
            if r & 1:
                r -= 1
                right.append(r)
            l >>= 1
            r >>= 1
        for node in left + right[::-1]:
            if t[node] > 0:
                # Descend to the leftmost positive leaf under this node.
                k = node
                while k < S:
                    k = 2 * k if t[2 * k] > 0 else 2 * k + 1
                return k - S
        return -1


# ---------------------------------------------------------------------------
# Best-Fit-Decreasing over pool items (each item length in [1, cap-1]).
# Returns item_bin (bin id per item), num_bins, and per-bin remaining capacity.
# ---------------------------------------------------------------------------
def bestfit_decreasing(item_lens: np.ndarray, cap: int, log_every: int = 5_000_000):
    n = int(item_lens.shape[0])
    item_bin = np.empty(n, dtype=np.int64)
    if n == 0:
        return item_bin, 0, []

    # Descending by length; stable so ties keep input order (reproducible).
    order = np.argsort(item_lens, kind="stable")[::-1]

    seg = _SegTreeSuccessor(cap)
    bins_at = [[] for _ in range(cap + 1)]  # bins_at[r] = bin ids with remaining r
    bin_remaining = []                       # indexed by bin id
    num_bins = 0
    t0 = time.time()

    for count, it in enumerate(order):
        s = int(item_lens[it])
        r = seg.find_ge(s)
        if r == -1:
            b = num_bins
            num_bins += 1
            bin_remaining.append(cap - s)
            item_bin[it] = b
            nr = cap - s
            if nr > 0:
                bins_at[nr].append(b)
                seg.add(nr, 1)
        else:
            b = bins_at[r].pop()
            seg.add(r, -1)
            item_bin[it] = b
            nr = r - s
            bin_remaining[b] = nr
            if nr > 0:
                bins_at[nr].append(b)
                seg.add(nr, 1)
        if log_every and (count + 1) % log_every == 0:
            el = time.time() - t0
            print(f"    BFD {count + 1:,}/{n:,} items  {num_bins:,} bins  "
                  f"{(count + 1) / el / 1e6:.2f}M items/s", flush=True)

    return item_bin, num_bins, bin_remaining


# ---------------------------------------------------------------------------
# Build the list of "pieces" (doc_idx, offset, length) from raw doc lengths.
#   - small docs (0 < len < cap)  -> one pool piece (the whole doc)
#   - full docs (len == cap)      -> one full-bin piece (ends in its own EOD)
#   - long docs (len > cap)       -> default: ceil(len/cap) full-bin head pieces
#                                     + one tail pool piece (carries doc EOD).
#                                     strict_eod: chunk into <= (cap-1) pool pieces
#                                     so every emitted bin ends in EOD (leak-free,
#                                     at the cost of a false mid-doc EOD-prediction).
# Returns numpy arrays for pool pieces and full-bin pieces + leak count.
# ---------------------------------------------------------------------------
def build_pieces(lengths: np.ndarray, cap: int, strict_eod: bool):
    lengths = np.asarray(lengths)
    small_mask = (lengths > 0) & (lengths < cap)
    full_mask = lengths == cap
    long_mask = lengths > cap
    n_empty = int((lengths <= 0).sum())

    small_idx = np.where(small_mask)[0]
    pool_doc = [small_idx.astype(np.int64)]
    pool_off = [np.zeros(small_idx.shape[0], dtype=np.int64)]
    pool_len = [lengths[small_idx].astype(np.int64)]

    full_idx = np.where(full_mask)[0]
    full_doc = [full_idx.astype(np.int64)]
    full_off = [np.zeros(full_idx.shape[0], dtype=np.int64)]
    full_len = [np.full(full_idx.shape[0], cap, dtype=np.int64)]
    full_ends_eod = [np.ones(full_idx.shape[0], dtype=bool)]  # len==cap doc ends in real EOD

    # Long docs: Python loop (rare). Accumulate then concatenate.
    long_pool, long_full, long_full_eod = [], [], []
    for i in np.where(long_mask)[0]:
        L = int(lengths[i])
        if strict_eod:
            off = 0
            while off < L:
                ln = min(cap - 1, L - off)
                long_pool.append((i, off, ln))
                off += ln
        else:
            num_full = L // cap
            rem = L % cap
            for k in range(num_full):
                is_last = (rem == 0 and k == num_full - 1)  # last full chunk == doc end
                long_full.append((i, k * cap, cap))
                long_full_eod.append(is_last)
            if rem > 0:
                long_pool.append((i, num_full * cap, rem))  # tail carries the doc EOD

    if long_pool:
        lp = np.array(long_pool, dtype=np.int64)
        pool_doc.append(lp[:, 0]); pool_off.append(lp[:, 1]); pool_len.append(lp[:, 2])
    if long_full:
        lf = np.array(long_full, dtype=np.int64)
        full_doc.append(lf[:, 0]); full_off.append(lf[:, 1]); full_len.append(lf[:, 2])
        full_ends_eod.append(np.array(long_full_eod, dtype=bool))

    pool = dict(
        doc=np.concatenate(pool_doc), off=np.concatenate(pool_off), len=np.concatenate(pool_len),
    )
    full = dict(
        doc=np.concatenate(full_doc), off=np.concatenate(full_off), len=np.concatenate(full_len),
        ends_eod=np.concatenate(full_ends_eod),
    )
    leak_chunks = int((~full["ends_eod"]).sum())  # full bins that do NOT end in EOD
    return pool, full, leak_chunks, n_empty


def pad_lengths(lengths: np.ndarray, multiple: int) -> np.ndarray:
    """Round each piece length up to the next multiple (int64; multiple<=1 = no-op).

    Used by --pad-doc-multiple: the emit step then pads each piece with EOD to
    this rounded length, so every packed segment satisfies the THD+CP
    divisibility requirement (TE's thd_get_partitioned_indices and the GDN
    a2a permutation both need per-segment lengths % (2*cp_size) == 0).
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    if multiple <= 1:
        return lengths
    return -(-lengths // multiple) * multiple


def estimate_baseline_truncations(lengths: np.ndarray, cap: int) -> int:
    """Doc-splits that plain concat-and-chunk would cause in the FILE order:
    a cut at every multiple of cap splits a doc unless it lands on a doc boundary.
    Order-dependent, but representative — a fair before/after comparison number."""
    lengths = np.asarray(lengths)
    boundaries = np.cumsum(lengths.astype(np.int64))  # cumulative doc-end positions
    total = int(boundaries[-1]) if boundaries.size else 0
    num_cuts = total // cap
    if num_cuts == 0:
        return 0
    cut_positions = np.arange(1, num_cuts + 1, dtype=np.int64) * cap
    # A cut is "aligned" (harmless) iff it equals a cumulative doc boundary.
    aligned = np.intersect1d(cut_positions, boundaries, assume_unique=False).shape[0]
    return int(num_cuts - aligned)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="Input prefix BEFORE _text_document.{bin,idx} (e.g. .../data)")
    p.add_argument("--output",
                   help="Output prefix (writes <output>_text_document.{bin,idx}); "
                        "required unless --dry-run")
    p.add_argument("--seq-length", type=int, default=4096,
                   help="Bin capacity L = model --seq-length (default 4096). "
                        "MUST equal the seq-length used at train time.")
    p.add_argument("--eod", type=int, default=0,
                   help="EOD token id, used as the pad token (default 0 = alpha v5).")
    p.add_argument("--strict-eod", action="store_true",
                   help="Make EVERY emitted bin end in EOD (no unmasked 1-token leak from "
                        "full-L chunks of over-length docs), at the cost of a false mid-doc "
                        "EOD-prediction baseline does not have. Default: accept+report the leak.")
    p.add_argument("--dry-run", action="store_true",
                   help="Pack and report stats; do not write any output.")
    p.add_argument("--verify-samples", type=int, default=20,
                   help="Number of emitted bins to round-trip verify against the source "
                        "(0 to skip). Default 20.")
    p.add_argument("--emit-threads", type=int, default=48,
                   help="Concurrent reader threads for the emit phase. Emit is NFS "
                        "random-read latency-bound, so parallel os.pread hides it (big "
                        "speedup on large shards). Writes stay ordered on the main thread "
                        "so output is byte-identical to --emit-threads 1. Default 48.")
    p.add_argument("--pad-doc-multiple", type=int, default=1,
                   help="Pad every packed piece with EOD so its emitted length is a multiple "
                        "of this value. Required for THD+CP document isolation: TE's "
                        "thd_get_partitioned_indices needs each segment divisible by "
                        "2*cp_size, so use 16 to cover CP<=8. Default 1 = no per-doc "
                        "padding (output byte-identical to previous behavior).")
    p.add_argument("--megatron-path",
                   default="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/"
                           "backends/megatron/Megatron-LM-251125",
                   help="Path containing megatron.core")
    args = p.parse_args()

    if not args.dry_run and not args.output:
        p.error("--output is required unless --dry-run")
    if args.megatron_path not in sys.path:
        sys.path.insert(0, args.megatron_path)
    from megatron.core.datasets import indexed_dataset

    cap = args.seq_length
    eod = args.eod
    pad_mult = args.pad_doc_multiple
    if pad_mult > 1 and cap % pad_mult != 0:
        p.error(f"--seq-length {cap} must be divisible by --pad-doc-multiple {pad_mult}")
    in_prefix = args.input + "_text_document"

    t_all = time.time()
    print(f"[bestfit_pack] input        = {in_prefix}.{{bin,idx}}")
    print(f"[bestfit_pack] seq-length L  = {cap}")
    print(f"[bestfit_pack] eod / pad id  = {eod}")
    print(f"[bestfit_pack] mode          = {'DRY-RUN' if args.dry_run else 'WRITE'}"
          f"{' (strict-eod)' if args.strict_eod else ''}"
          f"{f' (pad-doc-multiple={pad_mult})' if pad_mult > 1 else ''}")

    if not indexed_dataset.IndexedDataset.exists(in_prefix):
        print(f"ERROR: not found: {in_prefix}.{{bin,idx}}", file=sys.stderr)
        sys.exit(1)
    ds = indexed_dataset.IndexedDataset(in_prefix)
    dtype = ds.index.dtype
    lengths = np.asarray(ds.sequence_lengths)
    N = int(lengths.shape[0])
    real_tokens = int(lengths.sum())
    print(f"[bestfit_pack]   sequences   = {N:,}")
    print(f"[bestfit_pack]   real tokens = {real_tokens:,} ({real_tokens/1e9:.2f}B)")
    print(f"[bestfit_pack]   dtype       = {np.dtype(dtype).name}")

    # Sanity: do documents end in EOD? (they should, from --append-eod)
    if args.verify_samples > 0 and N > 0:
        rng = np.random.default_rng(7)
        sn = min(2000, N)
        sidx = rng.choice(N, size=sn, replace=False)
        ends_eod = 0
        for j in sidx:
            tok = ds.get(int(j))
            if tok.shape[0] > 0 and int(tok[-1]) == eod:
                ends_eod += 1
        frac = ends_eod / sn
        print(f"[bestfit_pack]   doc-end==EOD on {sn} samples: {frac*100:.1f}%"
              + ("" if frac > 0.99 else "  ⚠️  expected ~100% — wrong --eod or no --append-eod?"))

    # 1) pieces
    print(f"[bestfit_pack] building pieces ...", flush=True)
    pool, full, leak_chunks, n_empty = build_pieces(lengths, cap, args.strict_eod)
    n_long = int((lengths > cap).sum())
    if n_empty:
        print(f"[bestfit_pack]   ⚠️  {n_empty:,} empty (len<=0) sequences skipped")
    print(f"[bestfit_pack]   pool pieces = {pool['len'].shape[0]:,}  "
          f"full-bin pieces = {full['len'].shape[0]:,}  (long docs > L = {n_long:,})")

    # 1.5) per-piece padded lengths (--pad-doc-multiple; pad_mult==1 -> plen==len)
    pool_plen = pad_lengths(pool["len"], pad_mult)
    doc_pad_tokens = int((pool_plen - pool["len"]).sum())
    if pad_mult > 1:
        print(f"[bestfit_pack]   per-doc pad to %{pad_mult}: +{doc_pad_tokens:,} tokens "
              f"({doc_pad_tokens/max(real_tokens,1)*100:.3f}% of real)")

    # 2) BFD pack the pool (by padded lengths, so bins hold whole padded pieces)
    print(f"[bestfit_pack] best-fit-decreasing packing ...", flush=True)
    t0 = time.time()
    item_bin, n_pool_bins, bin_remaining = bestfit_decreasing(pool_plen, cap)
    n_full_bins = full["len"].shape[0]
    total_bins = n_pool_bins + n_full_bins
    print(f"[bestfit_pack]   BFD done in {time.time()-t0:.1f}s  "
          f"pool bins = {n_pool_bins:,}  full bins = {n_full_bins:,}  total = {total_bins:,}")

    # 3) stats
    tail_pad = int(np.asarray(bin_remaining, dtype=np.int64).sum()) if bin_remaining else 0
    pad_tokens = tail_pad + doc_pad_tokens
    emitted = total_bins * cap
    # invariant: emitted == real + pad (doc-level + bin-tail)
    assert emitted == real_tokens + pad_tokens, (emitted, real_tokens, doc_pad_tokens, tail_pad)
    fill = real_tokens / emitted if emitted else 1.0
    base_trunc = estimate_baseline_truncations(lengths, cap)
    # BFP only splits docs strictly longer than L: a doc of length L_i becomes
    # ceil(L_i/cap) pieces => ceil(L_i/cap)-1 internal cuts. Order-independent.
    if n_long:
        long_lens = lengths[lengths > cap].astype(np.int64)
        bfp_trunc = int((-(-long_lens // cap) - 1).sum())  # ceil division then -1
    else:
        bfp_trunc = 0
    print(f"[bestfit_pack] --- packing stats ---")
    print(f"[bestfit_pack]   bins (= packed seqs)      = {total_bins:,}")
    print(f"[bestfit_pack]   fill ratio                = {fill*100:.3f}%  "
          f"(pad overhead {(1-fill)*100:.3f}%, {pad_tokens:,} pad tokens)")
    print(f"[bestfit_pack]   BFP truncations (unavoidable long-doc cuts) = {bfp_trunc:,}")
    print(f"[bestfit_pack]   baseline concat-chunk doc-splits (file order, est.) = {base_trunc:,}")
    if base_trunc:
        print(f"[bestfit_pack]   => truncation reduction ≈ {(1 - bfp_trunc/base_trunc)*100:.1f}%")
    if not args.strict_eod and leak_chunks:
        print(f"[bestfit_pack]   note: {leak_chunks:,} full-L chunks of over-length docs end in "
              f"content (1 eod-unmasked label each, ~{leak_chunks/emitted*100:.4f}% of positions). "
              f"Use --strict-eod to eliminate.")

    if args.dry_run:
        print(f"[bestfit_pack] DRY-RUN — no output written. total {time.time()-t_all:.1f}s")
        return

    # 4) emit
    out_prefix = args.output + "_text_document"
    out_bin, out_idx = out_prefix + ".bin", out_prefix + ".idx"
    if os.path.exists(out_bin) or os.path.exists(out_idx):
        print(f"ERROR: output exists ({out_bin} or {out_idx}); refusing to overwrite",
              file=sys.stderr)
        sys.exit(1)
    Path(out_bin).parent.mkdir(parents=True, exist_ok=True)
    print(f"[bestfit_pack] emitting {total_bins:,} bins -> {out_prefix}.{{bin,idx}}", flush=True)
    builder = indexed_dataset.IndexedDatasetBuilder(out_bin, dtype=dtype)

    # Emit is NFS random-read latency-bound. Read each bin's member docs via
    # os.pread (releases the GIL -> true parallel reads) on a shared read-only fd,
    # assemble+pad in worker threads, and write ORDERED on the main thread so the
    # output is byte-identical to a serial emit. Byte offsets come from the .idx
    # sequence_pointers, exactly mirroring IndexedDataset.get().
    src_fd = os.open(in_prefix + ".bin", os.O_RDONLY)
    seq_ptr = np.asarray(ds.index.sequence_pointers)   # int64 byte offsets
    itemsize = np.dtype(dtype).itemsize
    pdoc, poff, plen = pool["doc"], pool["off"], pool["len"]
    pplen = pool_plen
    fdoc, foff, flen = full["doc"], full["off"], full["len"]

    def _read(doc, off, length):
        nbytes = int(length) * itemsize
        buf = os.pread(src_fd, nbytes, int(seq_ptr[int(doc)]) + int(off) * itemsize)
        return np.frombuffer(buf, dtype=dtype)

    def build_pool_bin(members):
        parts = []
        for m in members:
            part = _read(pdoc[m], poff[m], plen[m])
            target = int(pplen[m])
            if target > part.shape[0]:
                # per-doc EOD pad (--pad-doc-multiple; masked by --eod-mask-loss)
                part = np.concatenate([part, np.full(target - part.shape[0], eod, dtype=dtype)])
            parts.append(part)
        arr = np.concatenate(parts) if len(parts) > 1 else parts[0]
        fill_n = arr.shape[0]
        assert fill_n <= cap, f"bin overflow {fill_n} > {cap}"
        if fill_n < cap:
            # pad to exactly cap with EOD (masked by --eod-mask-loss)
            arr = np.concatenate([arr, np.full(cap - fill_n, eod, dtype=dtype)])
        return arr

    def build_full_bin(j):
        return _read(fdoc[j], foff[j], flen[j])   # exactly cap tokens, no pad

    def ordered_results(build_fn, arg_iter, window, ex):
        """Yield build_fn(arg) in submission order, keeping `window` reads in flight."""
        futs = deque()
        for _ in range(window):
            try:
                futs.append(ex.submit(build_fn, next(arg_iter)))
            except StopIteration:
                break
        while futs:
            r = futs.popleft().result()
            try:
                futs.append(ex.submit(build_fn, next(arg_iter)))
            except StopIteration:
                pass
            yield r

    n_threads = max(1, args.emit_threads)
    window = n_threads * 4
    t0 = time.time()
    written = 0
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        # pool bins: group pool pieces by bin id (stable argsort -> contiguous groups)
        if item_bin.shape[0] > 0:
            sort_idx = np.argsort(item_bin, kind="stable")
            sorted_bin = item_bin[sort_idx]
            starts = np.concatenate(([0], np.where(np.diff(sorted_bin) != 0)[0] + 1))
            ends = np.concatenate((starts[1:], [sort_idx.shape[0]]))
            pool_args = (sort_idx[starts[gi]:ends[gi]] for gi in range(starts.shape[0]))
            for arr in ordered_results(build_pool_bin, pool_args, window, ex):
                assert arr.shape[0] == cap
                # CRITICAL invariant: single-element length list so GPTDataset treats
                # the whole bin as one sequence (it ignores document_indices).
                builder.add_document(arr, [cap])
                written += 1
                if written % 2_000_000 == 0:
                    print(f"    emitted {written:,}/{total_bins:,}  "
                          f"{written/(time.time()-t0)/1e3:.0f}K bins/s", flush=True)
        # full bins: one piece each (already cap-long, no pad)
        if n_full_bins > 0:
            for arr in ordered_results(build_full_bin, iter(range(n_full_bins)), window, ex):
                assert arr.shape[0] == cap
                builder.add_document(arr, [cap])
                written += 1

    os.close(src_fd)
    builder.finalize(out_idx)
    print(f"[bestfit_pack]   emitted {written:,} bins in {time.time()-t0:.1f}s")
    assert written == total_bins, (written, total_bins)

    # 5) post-verify
    print(f"[bestfit_pack] verifying output ...", flush=True)
    out_ds = indexed_dataset.IndexedDataset(out_prefix)
    out_n = len(out_ds)
    out_lens = np.asarray(out_ds.sequence_lengths)
    if out_n != total_bins:
        print(f"ERROR: bin count mismatch out={out_n} expected={total_bins}", file=sys.stderr)
        sys.exit(2)
    if not (out_lens == cap).all():
        bad = int((out_lens != cap).sum())
        print(f"ERROR: {bad} output sequences are not exactly L={cap}", file=sys.stderr)
        sys.exit(2)
    out_total = int(out_lens.sum())
    if out_total != emitted:
        print(f"ERROR: token total mismatch out={out_total} expected={emitted}", file=sys.stderr)
        sys.exit(2)
    print(f"[bestfit_pack]   OK: {out_n:,} bins x {cap} = {out_total:,} tokens "
          f"(real {real_tokens:,} + pad {pad_tokens:,})")

    # round-trip: a few pool bins reconstructed from source must match output
    if args.verify_samples > 0 and item_bin.shape[0] > 0:
        rng = np.random.default_rng(11)
        # rebuild bin -> member pieces for sampled bins
        sample_bins = rng.choice(n_pool_bins, size=min(args.verify_samples, n_pool_bins),
                                 replace=False)
        sample_set = set(int(b) for b in sample_bins)
        members_of = {b: [] for b in sample_set}
        for it in range(item_bin.shape[0]):
            b = int(item_bin[it])
            if b in sample_set:
                members_of[b].append(it)
        ok = 0
        for b in sample_set:
            exp_parts = []
            for m in members_of[b]:
                part = ds.get(int(pool["doc"][m]), int(pool["off"][m]), int(pool["len"][m]))
                target = int(pool_plen[m])
                if target > part.shape[0]:
                    part = np.concatenate(
                        [part, np.full(target - part.shape[0], eod, dtype=dtype)])
                exp_parts.append(part)
            expected = np.concatenate(exp_parts)
            fn = expected.shape[0]
            if fn < cap:
                expected = np.concatenate([expected, np.full(cap - fn, eod, dtype=dtype)])
            got = np.asarray(out_ds[b])
            if got.shape[0] == cap and np.array_equal(got, expected):
                ok += 1
        print(f"[bestfit_pack]   round-trip OK on {ok}/{len(sample_set)} sampled bins")
        if ok != len(sample_set):
            print(f"ERROR: round-trip verification FAILED", file=sys.stderr)
            sys.exit(3)

    print(f"[bestfit_pack] DONE. output: {out_bin} "
          f"({os.path.getsize(out_bin)/1024**3:.2f} GB)  total {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
