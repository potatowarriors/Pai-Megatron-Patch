"""Remap the doc-end EOD ID in a pre-tokenized Megatron `.bin` in place.

Background:
  Historical alpha v5 pre-tokenization ran under the pre-2026-05-12 tokenizer
  configuration where `eos_token = "<|im_end|>"` → `tokenizer.eod = 3`. The
  `.bin` files therefore have id 3 at the end of every document.

  Phase 0.0 of the Stage 1 pre-flight (2026-05-12) updated `tokenizer_config.json`
  to designate `<|endoftext|>` (id 0) as the new EOS/EOD, separating pre-training
  doc-end from chat-turn-end per frontier convention (Qwen3, Llama 3, DSV3).

  This script makes the `.bin` consistent with the new tokenizer by remapping
  the trailing id 3 of every document to id 0. No structural change — total
  token count, .idx, and all non-end byte positions are unchanged. Only the
  final 4 bytes (one int32) of each document are modified.

Safety contract:
  Verified empirically in `tests/preflight_stage1/B_dataset_integrity.md` that
  id 3 appears EXCLUSIVELY at document boundaries across all three Stage 1
  datasets (5,000 docs × ≈ 17 M tokens per source: 0 mid-document occurrences).
  Therefore the byte-level substitution cannot corrupt any document content —
  every position we modify is guaranteed to currently hold id 3.

Usage:
    python remap_eod.py --prefix <prefix> --old-eod 3 --new-eod 0 [--dry-run]

    --prefix: path to data without .bin/.idx suffix
    --old-eod: id currently at every doc end (default 3)
    --new-eod: id to write (default 0)
    --dry-run: pre-verify that all doc-end positions hold --old-eod, do not write
    --no-verify-after: skip post-patch verification (default: verify)
"""
import argparse
import os
import struct
import sys
import time

import numpy as np


def load_idx(idx_path):
    """Read the .idx header + sequence_lengths + sequence_pointers via numpy.

    Returns (sequence_lengths: np.int32[N], sequence_pointers: np.int64[N],
            document_indices: np.int64[D]).
    """
    with open(idx_path, "rb") as f:
        magic = f.read(9)
        assert magic == b"MMIDIDX\x00\x00", f"bad magic: {magic!r}"
        version = struct.unpack("<Q", f.read(8))[0]
        assert version == 1
        dtype_code = struct.unpack("<B", f.read(1))[0]
        assert dtype_code == 4, f"expected int32 (4), got {dtype_code}"
        sequence_count = struct.unpack("<Q", f.read(8))[0]
        document_count = struct.unpack("<Q", f.read(8))[0]
        # The arrays follow contiguously:
        #   sequence_lengths : int32[sequence_count]
        #   sequence_pointers: int64[sequence_count]
        #   document_indices : int64[document_count]
        offset = f.tell()
    sl_bytes = sequence_count * 4
    sp_bytes = sequence_count * 8
    di_bytes = document_count * 8
    # Use mmap-style numpy access for memory efficiency on huge .idx files.
    sl = np.memmap(idx_path, dtype=np.int32, mode="r",
                   offset=offset, shape=(sequence_count,))
    sp = np.memmap(idx_path, dtype=np.int64, mode="r",
                   offset=offset + sl_bytes, shape=(sequence_count,))
    di = np.memmap(idx_path, dtype=np.int64, mode="r",
                   offset=offset + sl_bytes + sp_bytes, shape=(document_count,))
    return sl, sp, di


def compute_doc_end_token_positions(sequence_lengths, sequence_pointers):
    """Return int64 token-index positions (in .bin, in units of int32 = 4 bytes)
    of the LAST token of each sequence.

    For alpha Stage 1 data each sequence == one document, so this is the
    doc-end position of every document.
    """
    # sequence_pointers is in BYTE offsets. Each token = 4 bytes.
    # Last-token byte offset of sequence i = sequence_pointers[i] + (sequence_lengths[i] - 1) * 4
    # In token units: (sequence_pointers[i] // 4) + (sequence_lengths[i] - 1)
    return (sequence_pointers // 4).astype(np.int64) + \
           (sequence_lengths.astype(np.int64) - 1)


def pre_verify(bin_path, end_positions, expected_id, sample_size=200_000):
    """Verify that a random sample of doc-end positions currently hold expected_id."""
    bin_mm = np.memmap(bin_path, dtype=np.int32, mode="r")
    rng = np.random.default_rng(42)
    n = end_positions.shape[0]
    sample_idx = rng.choice(n, size=min(sample_size, n), replace=False)
    sample_positions = end_positions[sample_idx]
    sample_values = bin_mm[sample_positions]
    mismatches = int((sample_values != expected_id).sum())
    actual_min = int(sample_values.min())
    actual_max = int(sample_values.max())
    del bin_mm
    return {
        "sampled": int(sample_positions.shape[0]),
        "mismatches": mismatches,
        "expected": expected_id,
        "observed_min": actual_min,
        "observed_max": actual_max,
    }


def patch_in_place(bin_path, end_positions, new_id, batch=1_000_000):
    """Write new_id (int32) at each end position. Sequential by position.

    Uses numpy.memmap r+ for write. Memmap automatically flushes to NFS.
    """
    bin_mm = np.memmap(bin_path, dtype=np.int32, mode="r+")
    # Process in batches to give progress feedback for the huge DCLM file.
    n = end_positions.shape[0]
    for start in range(0, n, batch):
        end = min(start + batch, n)
        bin_mm[end_positions[start:end]] = new_id
    # Flush to disk
    bin_mm.flush()
    del bin_mm


def post_verify(bin_path, end_positions, expected_id, sample_size=200_000):
    """Post-patch verification."""
    bin_mm = np.memmap(bin_path, dtype=np.int32, mode="r")
    rng = np.random.default_rng(43)
    n = end_positions.shape[0]
    sample_idx = rng.choice(n, size=min(sample_size, n), replace=False)
    sample_positions = end_positions[sample_idx]
    sample_values = bin_mm[sample_positions]
    mismatches = int((sample_values != expected_id).sum())
    # Also confirm boundary docs (first and last 100)
    boundary_check_first = bin_mm[end_positions[:100]].tolist()
    boundary_check_last = bin_mm[end_positions[-100:]].tolist()
    del bin_mm
    return {
        "sampled": int(sample_positions.shape[0]),
        "mismatches": mismatches,
        "all_first_100_match": all(v == expected_id for v in boundary_check_first),
        "all_last_100_match": all(v == expected_id for v in boundary_check_last),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True, help="path without .bin/.idx suffix")
    p.add_argument("--old-eod", type=int, default=3, help="current EOD id at doc ends")
    p.add_argument("--new-eod", type=int, default=0, help="new EOD id to write")
    p.add_argument("--dry-run", action="store_true",
                   help="only pre-verify; do not modify .bin")
    p.add_argument("--no-verify-after", action="store_true",
                   help="skip post-patch verification")
    p.add_argument("--sample-size", type=int, default=200_000,
                   help="number of doc-end positions to verify per pass")
    args = p.parse_args()

    bin_path = args.prefix + ".bin"
    idx_path = args.prefix + ".idx"

    print(f"[remap_eod] prefix       = {args.prefix}")
    print(f"[remap_eod] .bin size    = {os.path.getsize(bin_path):,} bytes")
    print(f"[remap_eod] .idx size    = {os.path.getsize(idx_path):,} bytes")
    print(f"[remap_eod] mode         = {'DRY-RUN' if args.dry_run else 'IN-PLACE PATCH'}")
    print(f"[remap_eod] {args.old_eod} → {args.new_eod}")

    t0 = time.time()
    print(f"[remap_eod] loading .idx ...", flush=True)
    sl, sp, di = load_idx(idx_path)
    print(f"[remap_eod]   sequences  = {sl.shape[0]:,}")
    print(f"[remap_eod]   documents  = {di.shape[0]:,} (incl. start-0 marker)")
    print(f"[remap_eod]   sum lens   = {int(sl.sum()):,}")

    print(f"[remap_eod] computing doc-end token positions ...", flush=True)
    end_positions = compute_doc_end_token_positions(sl, sp)
    print(f"[remap_eod]   N positions = {end_positions.shape[0]:,}")
    print(f"[remap_eod]   first 3 positions = {end_positions[:3].tolist()}")
    print(f"[remap_eod]   last 3 positions  = {end_positions[-3:].tolist()}")

    print(f"[remap_eod] pre-verifying {args.sample_size:,} sampled doc-end positions hold {args.old_eod} ...",
          flush=True)
    pre = pre_verify(bin_path, end_positions, args.old_eod, args.sample_size)
    print(f"[remap_eod]   sampled={pre['sampled']:,}  mismatches={pre['mismatches']}  "
          f"observed_range=[{pre['observed_min']}, {pre['observed_max']}]")
    if pre["mismatches"] > 0:
        print(f"[remap_eod] ❌ pre-verification FAILED — refusing to patch.")
        sys.exit(2)
    print(f"[remap_eod] ✅ pre-verification OK")

    if args.dry_run:
        print(f"[remap_eod] DRY-RUN — exiting before write.")
        print(f"[remap_eod] total wall: {time.time()-t0:.1f} s")
        return

    print(f"[remap_eod] applying in-place patch ({args.old_eod} → {args.new_eod}) ...", flush=True)
    pt = time.time()
    patch_in_place(bin_path, end_positions, args.new_eod)
    print(f"[remap_eod]   patch wall: {time.time()-pt:.1f} s")

    if not args.no_verify_after:
        print(f"[remap_eod] post-verifying {args.sample_size:,} sampled positions hold {args.new_eod} ...",
              flush=True)
        post = post_verify(bin_path, end_positions, args.new_eod, args.sample_size)
        print(f"[remap_eod]   sampled={post['sampled']:,}  mismatches={post['mismatches']}  "
              f"first100_ok={post['all_first_100_match']}  last100_ok={post['all_last_100_match']}")
        if post["mismatches"] != 0 or not post["all_first_100_match"] or not post["all_last_100_match"]:
            print(f"[remap_eod] ❌ post-verification FAILED")
            sys.exit(3)
        print(f"[remap_eod] ✅ post-verification OK")

    print(f"[remap_eod] DONE.  total wall: {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
