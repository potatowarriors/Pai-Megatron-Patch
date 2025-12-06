#!/usr/bin/env python
"""
Count total tokens in MMAP dataset
"""
import struct
import numpy as np
import sys

def count_mmap_tokens(idx_path):
    """
    Read MMAP .idx file and calculate total tokens

    .idx file format:
    - Header: b"MMIDIDX\x00\x00" (9 bytes)
    - Version: uint64 (8 bytes)
    - Dtype code: uint8 (1 byte)
    - Sequence count: uint64 (8 bytes)
    - Document count: uint64 (8 bytes)
    - Sequence lengths: int32 array [sequence_count]
    - Sequence pointers: int64 array [sequence_count]
    - Document indices: int64 array [document_count]
    """
    print(f"Reading: {idx_path}")

    with open(idx_path, "rb") as f:
        # Read header
        header = f.read(9)
        assert header == b"MMIDIDX\x00\x00", "Invalid header"

        # Read version
        version = struct.unpack("<Q", f.read(8))[0]
        assert version == 1, "Unsupported version"

        # Read dtype code
        dtype_code = struct.unpack("<B", f.read(1))[0]
        dtype_map = {
            1: np.uint8,
            2: np.int8,
            3: np.int16,
            4: np.int32,
            5: np.int64,
            6: np.float64,
            7: np.float32,
            8: np.uint16,
        }
        dtype = dtype_map[dtype_code]
        dtype_size = dtype().itemsize

        # Read counts
        sequence_count = struct.unpack("<Q", f.read(8))[0]
        document_count = struct.unpack("<Q", f.read(8))[0]

        print(f"Sequence count: {sequence_count:,}")
        print(f"Document count: {document_count:,}")
        print(f"Dtype: {dtype.__name__} ({dtype_size} bytes)")

        # Read sequence lengths
        offset = f.tell()

    # Use memmap for efficient reading
    sequence_lengths = np.memmap(
        idx_path,
        dtype=np.int32,
        mode='r',
        offset=offset,
        shape=(sequence_count,)
    )

    # Calculate total tokens
    total_tokens = np.sum(sequence_lengths, dtype=np.int64)

    print(f"\nTotal tokens: {total_tokens:,}")
    print(f"Total tokens (scientific): {total_tokens:.2e}")
    print(f"Avg tokens per sequence: {total_tokens / sequence_count:.1f}")
    print(f"Avg tokens per document: {total_tokens / document_count:.1f}")

    # Verify with .bin file size
    bin_path = idx_path.replace('.idx', '.bin')
    try:
        import os
        bin_size = os.path.getsize(bin_path)
        expected_tokens = bin_size // dtype_size
        print(f"\nVerification:")
        print(f".bin file size: {bin_size:,} bytes ({bin_size / (1024**3):.2f} GB)")
        print(f"Expected tokens from file size: {expected_tokens:,}")
        print(f"Match: {'✓' if expected_tokens == total_tokens else '✗ MISMATCH'}")
    except Exception as e:
        print(f"\nCannot verify with .bin file: {e}")

    return total_tokens, sequence_count, document_count

if __name__ == "__main__":
    if len(sys.argv) > 1:
        idx_path = sys.argv[1]
    else:
        # Default to 1% subset
        idx_path = "/home/work/Datasets/KORMo_processed/mmap/qwen3_1pct/kormo_content_document.idx"

    total_tokens, seq_count, doc_count = count_mmap_tokens(idx_path)

    print(f"\n{'='*60}")
    print(f"Summary: {total_tokens:,} tokens across {doc_count:,} documents")
    print(f"{'='*60}")
