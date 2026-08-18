#!/usr/bin/env python3
"""
split_by_doclen.py — partition an unpacked (document-level) Megatron
IndexedDataset into two datasets by document token length.

Why: LC stage allocation (examples/alpha/docs/LC_DATASETS.md §5.1). Docs >= the
threshold (default 64k tokens) are the only source that can teach >32k
dependencies, and there are only ~4.8B such tokens in the whole LC pool — so
the 32k stage trains on the `lt` side and the `ge` side is RESERVED fresh for
the 128k stage. Run BEFORE bestfit_pack.py, on the unpacked mmap (never on
packed output — packed docs are all exactly L and carry pad EODs).

  unpacked _text_document ──split_by_doclen──> <out-lt>_text_document  (< T)
                                            └> <out-ge>_text_document  (>= T)

Each output document stays ONE sequence (add_document(arr, [len])), preserving
GPTDataset doc semantics; dtype is preserved; doc order is preserved within
each side. Empty (len<=0) docs are dropped and reported.

Usage:
  # report the split without writing
  python3 split_by_doclen.py --input /path/prefix --threshold 65536 --dry-run

  # write both sides
  python3 split_by_doclen.py --input /path/prefix \
      --output-lt /path/lt64k/data --output-ge /path/ge64k/data

`--input/--output-*` are prefixes BEFORE `_text_document.{bin,idx}` (same
convention as bestfit_pack.py / merge_indices.py). Either output side may be
omitted to write only the other.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="input prefix BEFORE _text_document.{bin,idx}")
    p.add_argument("--output-lt", help="output prefix for docs < threshold")
    p.add_argument("--output-ge", help="output prefix for docs >= threshold")
    p.add_argument("--threshold", type=int, default=65536,
                   help="token length threshold T (default 65536 = 64k); "
                        "docs < T -> lt side, docs >= T -> ge side")
    p.add_argument("--dry-run", action="store_true",
                   help="report the partition; write nothing")
    p.add_argument("--log-every", type=int, default=500_000)
    p.add_argument("--megatron-path",
                   default="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/"
                           "backends/megatron/Megatron-LM-251125")
    args = p.parse_args()

    if not args.dry_run and not (args.output_lt or args.output_ge):
        p.error("--output-lt and/or --output-ge required unless --dry-run")
    if args.megatron_path not in sys.path:
        sys.path.insert(0, args.megatron_path)
    from megatron.core.datasets import indexed_dataset

    in_prefix = args.input + "_text_document"
    if not indexed_dataset.IndexedDataset.exists(in_prefix):
        print(f"ERROR: not found: {in_prefix}.{{bin,idx}}", file=sys.stderr)
        sys.exit(1)
    ds = indexed_dataset.IndexedDataset(in_prefix)
    dtype = ds.index.dtype
    lengths = np.asarray(ds.sequence_lengths, dtype=np.int64)
    T = args.threshold
    lt_mask = (lengths > 0) & (lengths < T)
    ge_mask = lengths >= T
    n_empty = int((lengths <= 0).sum())
    stats = {
        "lt": (int(lt_mask.sum()), int(lengths[lt_mask].sum())),
        "ge": (int(ge_mask.sum()), int(lengths[ge_mask].sum())),
    }
    total_tokens = int(lengths.sum())
    print(f"[split_by_doclen] input   = {in_prefix}  "
          f"({lengths.size:,} docs / {total_tokens/1e9:.3f}B tokens, "
          f"dtype {np.dtype(dtype).name})")
    print(f"[split_by_doclen] T       = {T:,} tokens")
    for side in ("lt", "ge"):
        n, tok = stats[side]
        print(f"[split_by_doclen]   {side} : {n:,} docs / {tok/1e9:.3f}B tokens "
              f"({tok/total_tokens*100 if total_tokens else 0:.1f}%)")
    if n_empty:
        print(f"[split_by_doclen]   dropped {n_empty:,} empty docs")
    if args.dry_run:
        print("[split_by_doclen] DRY-RUN — nothing written")
        return

    sides = []
    if args.output_lt:
        sides.append(("lt", lt_mask, args.output_lt))
    if args.output_ge:
        sides.append(("ge", ge_mask, args.output_ge))
    for side, mask, out in sides:
        out_prefix = out + "_text_document"
        out_bin, out_idx = out_prefix + ".bin", out_prefix + ".idx"
        if os.path.exists(out_bin) or os.path.exists(out_idx):
            print(f"ERROR: output exists ({out_bin}); refusing to overwrite",
                  file=sys.stderr)
            sys.exit(1)
        Path(out_bin).parent.mkdir(parents=True, exist_ok=True)
        idxs = np.where(mask)[0]
        print(f"[split_by_doclen] writing {side}: {idxs.size:,} docs -> "
              f"{out_prefix}.{{bin,idx}}", flush=True)
        builder = indexed_dataset.IndexedDatasetBuilder(out_bin, dtype=dtype)
        t0 = time.time()
        for k, i in enumerate(idxs):
            arr = ds.get(int(i))
            builder.add_document(arr, [arr.shape[0]])
            if args.log_every and (k + 1) % args.log_every == 0:
                print(f"    {side} {k+1:,}/{idxs.size:,} "
                      f"({(k+1)/(time.time()-t0):.0f} docs/s)", flush=True)
        builder.finalize(out_idx)

        # verify: doc count and token totals must match the plan exactly
        out_ds = indexed_dataset.IndexedDataset(out_prefix)
        out_lens = np.asarray(out_ds.sequence_lengths, dtype=np.int64)
        n_exp, tok_exp = stats[side]
        if len(out_ds) != n_exp or int(out_lens.sum()) != tok_exp:
            print(f"ERROR: {side} verify failed: docs {len(out_ds)} vs {n_exp}, "
                  f"tokens {int(out_lens.sum())} vs {tok_exp}", file=sys.stderr)
            sys.exit(2)
        # spot round-trip
        if idxs.size:
            rng = np.random.default_rng(5)
            for j in rng.choice(idxs.size, min(10, idxs.size), replace=False):
                if not np.array_equal(out_ds.get(int(j)), ds.get(int(idxs[j]))):
                    print(f"ERROR: {side} round-trip mismatch at {j}",
                          file=sys.stderr)
                    sys.exit(3)
        print(f"[split_by_doclen]   {side} OK: {n_exp:,} docs / "
              f"{tok_exp/1e9:.3f}B tokens verified "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("[split_by_doclen] DONE")


if __name__ == "__main__":
    main()
