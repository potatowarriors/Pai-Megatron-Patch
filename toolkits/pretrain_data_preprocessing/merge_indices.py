#!/usr/bin/env python3
"""
merge_indices.py — Merge N per-process IndexedDataset parts into one final mmap.

Reads each part (`<prefix>_text_document.{bin,idx}`) and concatenates them via
`IndexedDatasetBuilder.add_index(prefix)` + `finalize(idx_path)`.

The order of parts on the CLI defines the final document order. For
correctness, list parts in deterministic numerical order (e.g. part0, part1, ...).

Usage:
    python merge_indices.py \\
        --output /path/to/final/<output_prefix> \\
        --parts /path/to/part0 /path/to/part1 /path/to/part2 ...
        [--dtype int32]
        [--megatron-path <path>]

Each `--parts` entry should be the prefix BEFORE `_text_document.{bin,idx}`.
"""

import argparse
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True,
                        help="Output prefix (writes <output>_text_document.{bin,idx})")
    parser.add_argument("--parts", required=True, nargs="+",
                        help="Part prefixes in MERGE ORDER (each is <prefix>_text_document.{bin,idx})")
    parser.add_argument("--dtype", default="int32", choices=["int32", "uint16"],
                        help="dtype of part .bin (must match the parts; default int32 for vocab > 65k)")
    parser.add_argument("--megatron-path",
                        default="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/backends/megatron/Megatron-LM-251125",
                        help="Path containing megatron.core (default points to 251125 backend)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Don't print recommended part cleanup commands at the end")
    args = parser.parse_args()

    if args.megatron_path not in sys.path:
        sys.path.insert(0, args.megatron_path)

    import numpy as np
    from megatron.core.datasets import indexed_dataset

    dtype_map = {"int32": np.int32, "uint16": np.uint16}
    dtype = dtype_map[args.dtype]

    # Verify all parts exist before starting
    missing = []
    total_bin_bytes = 0
    for prefix in args.parts:
        bin_path = f"{prefix}_text_document.bin"
        idx_path = f"{prefix}_text_document.idx"
        if not os.path.exists(bin_path):
            missing.append(bin_path)
        if not os.path.exists(idx_path):
            missing.append(idx_path)
        if os.path.exists(bin_path):
            total_bin_bytes += os.path.getsize(bin_path)
    if missing:
        print(f"ERROR: missing files:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)

    # Verify each part loads + check dtype matches
    print(f"verifying {len(args.parts)} parts...", flush=True)
    total_docs = 0
    total_tokens = 0
    for prefix in args.parts:
        ds = indexed_dataset.IndexedDataset(prefix + "_text_document")
        if ds.index.dtype != dtype:
            print(f"ERROR: dtype mismatch on {prefix}: got {ds.index.dtype}, expected {dtype.__name__}",
                  file=sys.stderr)
            sys.exit(1)
        part_docs = len(ds)
        part_tokens = int(ds.sequence_lengths.sum())
        print(f"  {os.path.basename(prefix)}: {part_docs:,} docs, "
              f"{part_tokens/1e9:.2f}B tokens, "
              f"{os.path.getsize(prefix + '_text_document.bin')/1024**3:.2f} GB",
              flush=True)
        total_docs += part_docs
        total_tokens += part_tokens
        del ds

    print(f"\ntotal: {total_docs:,} docs, {total_tokens:,} tokens ({total_tokens/1e9:.2f}B), "
          f"{total_bin_bytes/1024**3:.2f} GB", flush=True)

    # Open output builder
    output_bin = f"{args.output}_text_document.bin"
    output_idx = f"{args.output}_text_document.idx"
    Path(output_bin).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(output_bin) or os.path.exists(output_idx):
        print(f"ERROR: output already exists ({output_bin} or {output_idx}); refusing to overwrite",
              file=sys.stderr)
        sys.exit(1)

    print(f"\nmerging into {args.output}_text_document.{{bin,idx}}", flush=True)
    builder = indexed_dataset.IndexedDatasetBuilder(output_bin, dtype=dtype)
    t0 = time.time()
    for i, prefix in enumerate(args.parts):
        t_part = time.time()
        builder.add_index(prefix + "_text_document")
        dt = time.time() - t_part
        print(f"  [{i+1}/{len(args.parts)}] added {os.path.basename(prefix)} in {dt:.1f}s", flush=True)
    builder.finalize(output_idx)
    total_dt = time.time() - t0

    # Verify
    print(f"\nverifying merged output...", flush=True)
    ds = indexed_dataset.IndexedDataset(args.output + "_text_document")
    merged_docs = len(ds)
    merged_tokens = int(ds.sequence_lengths.sum())
    if merged_docs != total_docs:
        print(f"ERROR: doc count mismatch — parts={total_docs}, merged={merged_docs}", file=sys.stderr)
        sys.exit(1)
    if merged_tokens != total_tokens:
        print(f"ERROR: token count mismatch — parts={total_tokens}, merged={merged_tokens}", file=sys.stderr)
        sys.exit(1)
    print(f"  OK: {merged_docs:,} docs, {merged_tokens:,} tokens", flush=True)

    print(f"\nDONE merge in {total_dt:.1f}s", flush=True)
    print(f"  output:  {output_bin} ({os.path.getsize(output_bin)/1024**3:.2f} GB)", flush=True)
    print(f"           {output_idx} ({os.path.getsize(output_idx)/1024**2:.1f} MB)", flush=True)

    if not args.no_cleanup:
        print(f"\nTo clean up parts (after verifying training works):", flush=True)
        for prefix in args.parts:
            print(f"  rm {prefix}_text_document.bin {prefix}_text_document.idx", flush=True)


if __name__ == "__main__":
    main()
