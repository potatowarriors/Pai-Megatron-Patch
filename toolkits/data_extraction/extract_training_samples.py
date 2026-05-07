#!/usr/bin/env python3
"""
Extract training samples from a specific iteration for grad norm spike analysis.

This script extracts the exact samples used during a specific training iteration
by reconstructing the data pipeline from cached indices.

Data Flow:
    global_sample_idx (consumed_samples range)
        ↓ BlendedDataset: dataset_index[idx] → dataset_id
        ↓ BlendedDataset: dataset_sample_index[idx] → within_dataset_idx
        ↓ GPTDataset: shuffle_index[within_dataset_idx] → shuffled_idx
        ↓ GPTDataset: sample_index[shuffled_idx] → (doc_idx, offset)
        ↓ GPTDataset: document_index[doc_idx] → actual_doc_id
        ↓ IndexedDataset: read tokens from .bin file

Usage:
    python extract_training_samples.py \
        --iteration 324065 \
        --output ./iter_324065_samples.json \
        --output-text ./iter_324065_readable.txt
"""

import argparse
import json
import os
import struct
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExtractionConfig:
    """Configuration for sample extraction."""
    # Iteration parameters
    iteration: int = 324065
    consumed_samples: int = 82_960_640
    global_batch_size: int = 256
    sequence_length: int = 4096
    random_seed: int = 1234

    # Cache hashes (from training run)
    blended_hash: str = "7a70c9f2e20b95da0f51ba7d24be187f"
    dclm_hash: str = "8efc4b85c35229927727698046e57a21"
    korean_hash: str = "9a9c17d8c2970cae8e1e80c4b00d94d1"

    # Paths
    cache_dir: str = "/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/configs/data/.cache/kormo_50pct"
    dclm_data_path: str = "/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/datasets/processed/qwen3_50pct/dclm/dclm_content_document"
    korean_data_path: str = "/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/datasets/processed/qwen3_50pct/korean_web/korean_web_content_document"
    tokenizer_path: str = "/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/tokenizer_v5"

    # Dataset info
    dataset_names: Tuple[str, ...] = ("DCLM", "Korean Web")
    dataset_weights: Tuple[float, ...] = (0.9024118057814148, 0.09758819421858526)

    @property
    def sample_start(self) -> int:
        """First sample index in the batch."""
        return self.consumed_samples - self.global_batch_size

    @property
    def sample_end(self) -> int:
        """Last sample index in the batch (exclusive)."""
        return self.consumed_samples


@dataclass
class SampleInfo:
    """Information about a single extracted sample."""
    global_idx: int
    dataset_id: int
    dataset_name: str
    within_dataset_idx: int
    shuffled_idx: int
    doc_indices: List[int]
    doc_offsets: List[Tuple[int, int]]  # (start_offset, end_offset) for each doc
    token_count: int
    tokens: List[int]
    text: str
    text_preview: str  # First 500 chars


# ============================================================================
# Index File Readers
# ============================================================================

class IndexedDatasetReader:
    """
    Reads Megatron IndexedDataset (.bin/.idx) files.

    The .idx file structure:
        - Header: "MMIDIDX\x00\x00" (9 bytes)
        - Version: uint64 (8 bytes)
        - dtype code: uint8 (1 byte)
        - sequence_count: uint64 (8 bytes)
        - document_count: uint64 (8 bytes)
        - sequence_lengths: int32[sequence_count]
        - sequence_pointers: int64[sequence_count]
        - document_indices: int64[document_count]
    """

    _INDEX_HEADER = b"MMIDIDX\x00\x00"
    _DTYPE_MAP = {
        1: np.uint8,
        2: np.int8,
        3: np.int16,
        4: np.int32,
        5: np.int64,
        6: np.float64,
        7: np.float32,
        8: np.uint16,
    }

    def __init__(self, path_prefix: str):
        self.path_prefix = path_prefix
        self.idx_path = path_prefix + ".idx"
        self.bin_path = path_prefix + ".bin"

        # Read index file
        self._read_index()

        # Memory-map the binary file
        self._bin_mmap = np.memmap(self.bin_path, mode='r', order='C')

    def _read_index(self):
        """Read the .idx file."""
        with open(self.idx_path, 'rb') as f:
            # Verify header
            header = f.read(9)
            if header != self._INDEX_HEADER:
                raise ValueError(f"Invalid header in {self.idx_path}")

            # Read version
            version = struct.unpack('<Q', f.read(8))[0]
            if version != 1:
                raise ValueError(f"Unsupported version {version} in {self.idx_path}")

            # Read dtype
            dtype_code = struct.unpack('<B', f.read(1))[0]
            self.dtype = self._DTYPE_MAP[dtype_code]
            self.dtype_size = np.dtype(self.dtype).itemsize

            # Read counts
            self.sequence_count = struct.unpack('<Q', f.read(8))[0]
            self.document_count = struct.unpack('<Q', f.read(8))[0]

            offset = f.tell()

        # Memory-map the rest using numpy
        idx_mmap = np.memmap(self.idx_path, mode='r', order='C')

        # Extract arrays
        self.sequence_lengths = np.frombuffer(
            idx_mmap, dtype=np.int32, count=self.sequence_count, offset=offset
        )

        self.sequence_pointers = np.frombuffer(
            idx_mmap, dtype=np.int64, count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes
        )

        self.document_indices = np.frombuffer(
            idx_mmap, dtype=np.int64, count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes
        )

    def get_sequence(self, idx: int, offset: int = 0, length: Optional[int] = None) -> np.ndarray:
        """
        Get a sequence (or part of it) from the dataset.

        Args:
            idx: Sequence index
            offset: Token offset within the sequence
            length: Number of tokens to read (None = to end)

        Returns:
            Token array
        """
        if length is None:
            length = self.sequence_lengths[idx] - offset

        ptr = self.sequence_pointers[idx] + offset * self.dtype_size
        return np.frombuffer(
            self._bin_mmap, dtype=self.dtype, count=length, offset=ptr
        ).copy()

    def __len__(self) -> int:
        return self.sequence_count


class GPTDatasetIndexReader:
    """Reads GPTDataset cached indices."""

    def __init__(self, cache_dir: str, hash_prefix: str):
        self.cache_dir = cache_dir
        self.hash_prefix = hash_prefix

        base = f"{hash_prefix}-GPTDataset-train"

        self.document_index = np.load(
            os.path.join(cache_dir, f"{base}-document_index.npy"),
            allow_pickle=True, mmap_mode='r'
        )
        self.sample_index = np.load(
            os.path.join(cache_dir, f"{base}-sample_index.npy"),
            allow_pickle=True, mmap_mode='r'
        )
        self.shuffle_index = np.load(
            os.path.join(cache_dir, f"{base}-shuffle_index.npy"),
            allow_pickle=True, mmap_mode='r'
        )

    def get_sample_range(self, idx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Get the document range and offsets for a sample.

        Returns:
            ((doc_idx_beg, doc_idx_beg_offset), (doc_idx_end, doc_idx_end_offset))
        """
        beg = self.sample_index[idx]
        end = self.sample_index[idx + 1]
        return (int(beg[0]), int(beg[1])), (int(end[0]), int(end[1]))

    def get_actual_doc_id(self, doc_idx: int) -> int:
        """Get the actual document ID from the document index."""
        return int(self.document_index[doc_idx])

    def __len__(self) -> int:
        return len(self.shuffle_index)


class BlendedDatasetIndexReader:
    """Reads BlendedDataset cached indices."""

    def __init__(self, cache_dir: str, hash_prefix: str):
        self.cache_dir = cache_dir
        self.hash_prefix = hash_prefix

        base = f"{hash_prefix}-BlendedDataset-train"

        self.dataset_index = np.load(
            os.path.join(cache_dir, f"{base}-dataset_index.npy"),
            allow_pickle=True, mmap_mode='r'
        )
        self.dataset_sample_index = np.load(
            os.path.join(cache_dir, f"{base}-dataset_sample_index.npy"),
            allow_pickle=True, mmap_mode='r'
        )

    def get_mapping(self, global_idx: int) -> Tuple[int, int]:
        """
        Map a global sample index to (dataset_id, within_dataset_idx).
        """
        return int(self.dataset_index[global_idx]), int(self.dataset_sample_index[global_idx])

    def __len__(self) -> int:
        return len(self.dataset_index)


# ============================================================================
# Tokenizer
# ============================================================================

def load_tokenizer(tokenizer_path: str):
    """Load the Qwen3 tokenizer."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        return tokenizer
    except ImportError:
        print("Warning: transformers not available, text decoding will be skipped")
        return None


# ============================================================================
# Sample Extraction
# ============================================================================

class SampleExtractor:
    """Extracts training samples from a specific iteration."""

    def __init__(self, config: ExtractionConfig):
        self.config = config

        print(f"Loading indices and datasets...")

        # Load BlendedDataset indices
        print(f"  Loading BlendedDataset indices (hash: {config.blended_hash})...")
        self.blended_reader = BlendedDatasetIndexReader(
            config.cache_dir, config.blended_hash
        )

        # Load GPTDataset indices for each dataset
        print(f"  Loading DCLM GPTDataset indices (hash: {config.dclm_hash})...")
        self.gpt_readers = [
            GPTDatasetIndexReader(config.cache_dir, config.dclm_hash),
            GPTDatasetIndexReader(config.cache_dir, config.korean_hash),
        ]
        print(f"  Loading Korean Web GPTDataset indices (hash: {config.korean_hash})...")

        # Load IndexedDatasets
        print(f"  Loading DCLM IndexedDataset...")
        self.indexed_datasets = [
            IndexedDatasetReader(config.dclm_data_path),
            IndexedDatasetReader(config.korean_data_path),
        ]
        print(f"  Loading Korean Web IndexedDataset...")

        # Load tokenizer
        print(f"  Loading tokenizer...")
        self.tokenizer = load_tokenizer(config.tokenizer_path)

        print(f"Initialization complete.\n")

    def extract_sample(self, global_idx: int) -> SampleInfo:
        """
        Extract a single sample given its global index.

        Data flow:
            global_idx
                → BlendedDataset: dataset_id, within_dataset_idx
                → GPTDataset: shuffled_idx via shuffle_index
                → GPTDataset: (doc_idx_beg, offset_beg), (doc_idx_end, offset_end) via sample_index
                → GPTDataset: actual_doc_ids via document_index
                → IndexedDataset: tokens via .bin file
        """
        # Step 1: BlendedDataset mapping
        dataset_id, within_dataset_idx = self.blended_reader.get_mapping(global_idx)

        # Step 2: Get the GPT reader and indexed dataset for this dataset
        gpt_reader = self.gpt_readers[dataset_id]
        indexed_dataset = self.indexed_datasets[dataset_id]

        # Step 3: Apply shuffle index
        shuffled_idx = int(gpt_reader.shuffle_index[within_dataset_idx])

        # Step 4: Get sample range from sample_index
        (doc_idx_beg, offset_beg), (doc_idx_end, offset_end) = gpt_reader.get_sample_range(shuffled_idx)

        # Step 5: Collect tokens from documents
        tokens = []
        doc_indices = []
        doc_offsets = []

        if doc_idx_beg == doc_idx_end:
            # Sample is within a single document
            actual_doc_id = gpt_reader.get_actual_doc_id(doc_idx_beg)
            doc_indices.append(actual_doc_id)

            # Read tokens (add 1 for the extra token used in training)
            length = offset_end - offset_beg + 1
            doc_tokens = indexed_dataset.get_sequence(actual_doc_id, offset_beg, length)
            tokens.extend(doc_tokens.tolist())
            doc_offsets.append((offset_beg, offset_beg + length))
        else:
            # Sample spans multiple documents
            for i in range(doc_idx_beg, doc_idx_end + 1):
                actual_doc_id = gpt_reader.get_actual_doc_id(i)
                doc_indices.append(actual_doc_id)

                if i == doc_idx_beg:
                    # First document: from offset_beg to end
                    offset = offset_beg
                    length = None
                elif i == doc_idx_end:
                    # Last document: from start to offset_end + 1
                    offset = 0
                    length = offset_end + 1
                else:
                    # Middle documents: entire document
                    offset = 0
                    length = None

                doc_tokens = indexed_dataset.get_sequence(actual_doc_id, offset, length)
                tokens.extend(doc_tokens.tolist())

                actual_length = len(doc_tokens)
                doc_offsets.append((offset, offset + actual_length))

        # Decode text
        if self.tokenizer is not None:
            text = self.tokenizer.decode(tokens, skip_special_tokens=False)
        else:
            text = f"[{len(tokens)} tokens - tokenizer not available]"

        text_preview = text[:500] + "..." if len(text) > 500 else text

        return SampleInfo(
            global_idx=global_idx,
            dataset_id=dataset_id,
            dataset_name=self.config.dataset_names[dataset_id],
            within_dataset_idx=within_dataset_idx,
            shuffled_idx=shuffled_idx,
            doc_indices=doc_indices,
            doc_offsets=doc_offsets,
            token_count=len(tokens),
            tokens=tokens,
            text=text,
            text_preview=text_preview,
        )

    def extract_iteration(self, iteration: Optional[int] = None) -> Tuple[List[SampleInfo], Dict[str, Any]]:
        """
        Extract all samples from an iteration.

        Returns:
            (samples, statistics)
        """
        config = self.config

        if iteration is not None:
            # Recalculate consumed_samples based on iteration
            consumed_samples = iteration * config.global_batch_size
            sample_start = consumed_samples - config.global_batch_size
            sample_end = consumed_samples
        else:
            sample_start = config.sample_start
            sample_end = config.sample_end
            iteration = config.iteration

        print(f"Extracting samples for iteration {iteration}")
        print(f"  Sample range: [{sample_start}, {sample_end})")
        print(f"  Global batch size: {config.global_batch_size}")
        print()

        samples = []
        dataset_counts = {name: 0 for name in config.dataset_names}
        total_tokens = 0

        for i, global_idx in enumerate(range(sample_start, sample_end)):
            if i % 50 == 0:
                print(f"  Extracting sample {i+1}/{config.global_batch_size}...")

            sample = self.extract_sample(global_idx)
            samples.append(sample)

            dataset_counts[sample.dataset_name] += 1
            total_tokens += sample.token_count

        print(f"\nExtraction complete.")

        # Compute statistics
        statistics = {
            "iteration": iteration,
            "sample_range": [sample_start, sample_end],
            "total_samples": len(samples),
            "dataset_distribution": dataset_counts,
            "expected_distribution": {
                name: round(weight * config.global_batch_size)
                for name, weight in zip(config.dataset_names, config.dataset_weights)
            },
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": total_tokens / len(samples) if samples else 0,
            "config": {
                "global_batch_size": config.global_batch_size,
                "sequence_length": config.sequence_length,
                "random_seed": config.random_seed,
            }
        }

        return samples, statistics


# ============================================================================
# Output Formatters
# ============================================================================

def save_json(samples: List[SampleInfo], statistics: Dict[str, Any], output_path: str):
    """Save samples to JSON format."""
    output = {
        "statistics": statistics,
        "samples": [
            {
                "global_idx": s.global_idx,
                "dataset_id": s.dataset_id,
                "dataset_name": s.dataset_name,
                "within_dataset_idx": s.within_dataset_idx,
                "shuffled_idx": s.shuffled_idx,
                "doc_indices": s.doc_indices,
                "doc_offsets": s.doc_offsets,
                "token_count": s.token_count,
                "tokens": s.tokens,
                "text_preview": s.text_preview,
            }
            for s in samples
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved JSON to {output_path}")


def save_text(samples: List[SampleInfo], statistics: Dict[str, Any], output_path: str):
    """Save samples to human-readable text format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("TRAINING SAMPLES EXTRACTION REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Statistics
        f.write("STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Iteration: {statistics['iteration']}\n")
        f.write(f"Sample Range: {statistics['sample_range']}\n")
        f.write(f"Total Samples: {statistics['total_samples']}\n")
        f.write(f"Total Tokens: {statistics['total_tokens']}\n")
        f.write(f"Avg Tokens/Sample: {statistics['avg_tokens_per_sample']:.1f}\n")
        f.write(f"\nDataset Distribution:\n")
        for name, count in statistics['dataset_distribution'].items():
            expected = statistics['expected_distribution'][name]
            f.write(f"  {name}: {count} (expected: ~{expected})\n")
        f.write("\n")

        # Samples
        f.write("=" * 80 + "\n")
        f.write("SAMPLES\n")
        f.write("=" * 80 + "\n\n")

        for i, s in enumerate(samples):
            f.write(f"--- Sample {i+1} (global_idx={s.global_idx}) ---\n")
            f.write(f"Dataset: {s.dataset_name} (id={s.dataset_id})\n")
            f.write(f"Within-dataset idx: {s.within_dataset_idx}\n")
            f.write(f"Shuffled idx: {s.shuffled_idx}\n")
            f.write(f"Documents: {s.doc_indices}\n")
            f.write(f"Token count: {s.token_count}\n")
            f.write(f"\nText:\n")
            f.write("-" * 40 + "\n")
            f.write(s.text + "\n")
            f.write("-" * 40 + "\n\n")

    print(f"Saved text to {output_path}")


def save_full_text(samples: List[SampleInfo], output_dir: str):
    """Save full text for each sample to individual files."""
    os.makedirs(output_dir, exist_ok=True)

    for s in samples:
        filename = f"sample_{s.global_idx:010d}_{s.dataset_name.replace(' ', '_')}.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Global Index: {s.global_idx}\n")
            f.write(f"Dataset: {s.dataset_name}\n")
            f.write(f"Token Count: {s.token_count}\n")
            f.write(f"Documents: {s.doc_indices}\n")
            f.write("=" * 80 + "\n\n")
            f.write(s.text)

    print(f"Saved {len(samples)} individual text files to {output_dir}/")


# ============================================================================
# Analysis Helpers
# ============================================================================

def analyze_samples(samples: List[SampleInfo]) -> Dict[str, Any]:
    """Perform analysis on extracted samples to find anomalies."""
    analysis = {
        "token_length_stats": {},
        "multi_doc_samples": [],
        "short_samples": [],
        "potential_issues": [],
    }

    token_counts = [s.token_count for s in samples]
    analysis["token_length_stats"] = {
        "min": min(token_counts),
        "max": max(token_counts),
        "mean": sum(token_counts) / len(token_counts),
        "samples_under_4096": sum(1 for c in token_counts if c < 4096),
        "samples_over_4096": sum(1 for c in token_counts if c > 4096),
    }

    for s in samples:
        # Multi-document samples
        if len(s.doc_indices) > 1:
            analysis["multi_doc_samples"].append({
                "global_idx": s.global_idx,
                "num_docs": len(s.doc_indices),
                "doc_indices": s.doc_indices,
            })

        # Short samples
        if s.token_count < 4096:
            analysis["short_samples"].append({
                "global_idx": s.global_idx,
                "token_count": s.token_count,
            })

        # Check for potential issues
        issues = []

        # Check for repeated tokens
        if len(s.tokens) > 100:
            for i in range(len(s.tokens) - 100):
                chunk = s.tokens[i:i+50]
                if chunk == s.tokens[i+50:i+100]:
                    issues.append("Repeated 50-token pattern detected")
                    break

        # Check for unusual characters in decoded text
        if s.text:
            null_count = s.text.count('\x00')
            if null_count > 10:
                issues.append(f"High null character count: {null_count}")

        if issues:
            analysis["potential_issues"].append({
                "global_idx": s.global_idx,
                "dataset_name": s.dataset_name,
                "issues": issues,
            })

    return analysis


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract training samples from a specific iteration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract samples from iteration 324065 (default)
    python extract_training_samples.py

    # Extract with custom output paths
    python extract_training_samples.py \\
        --iteration 324065 \\
        --output ./iter_324065_samples.json \\
        --output-text ./iter_324065_readable.txt

    # Extract with individual sample files
    python extract_training_samples.py \\
        --iteration 324065 \\
        --output-dir ./samples/
        """
    )

    parser.add_argument(
        "--iteration", type=int, default=324065,
        help="Training iteration to extract (default: 324065)"
    )
    parser.add_argument(
        "--consumed-samples", type=int, default=None,
        help="Override consumed samples (default: iteration * global_batch_size)"
    )
    parser.add_argument(
        "--global-batch-size", type=int, default=256,
        help="Global batch size (default: 256)"
    )
    parser.add_argument(
        "--output", type=str, default="./extracted_samples.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--output-text", type=str, default=None,
        help="Output human-readable text file path"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for individual sample text files"
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Perform anomaly analysis on extracted samples"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Override cache directory path"
    )

    args = parser.parse_args()

    # Build config
    config = ExtractionConfig(
        iteration=args.iteration,
        global_batch_size=args.global_batch_size,
    )

    if args.consumed_samples is not None:
        config.consumed_samples = args.consumed_samples
    else:
        config.consumed_samples = args.iteration * args.global_batch_size

    if args.cache_dir is not None:
        config.cache_dir = args.cache_dir

    # Create extractor
    extractor = SampleExtractor(config)

    # Extract samples
    samples, statistics = extractor.extract_iteration()

    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Iteration: {statistics['iteration']}")
    print(f"Total samples: {statistics['total_samples']}")
    print(f"Total tokens: {statistics['total_tokens']}")
    print(f"Dataset distribution:")
    for name, count in statistics['dataset_distribution'].items():
        expected = statistics['expected_distribution'][name]
        pct = count / statistics['total_samples'] * 100
        print(f"  {name}: {count} ({pct:.1f}%, expected: ~{expected})")

    # Perform analysis if requested
    if args.analyze:
        print("\n" + "=" * 60)
        print("ANOMALY ANALYSIS")
        print("=" * 60)
        analysis = analyze_samples(samples)

        stats = analysis["token_length_stats"]
        print(f"Token length: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.1f}")
        print(f"Samples under 4096 tokens: {stats['samples_under_4096']}")
        print(f"Multi-document samples: {len(analysis['multi_doc_samples'])}")

        if analysis["potential_issues"]:
            print(f"\nPotential issues found: {len(analysis['potential_issues'])}")
            for issue in analysis["potential_issues"][:5]:
                print(f"  Sample {issue['global_idx']} ({issue['dataset_name']}): {issue['issues']}")
        else:
            print("\nNo obvious anomalies detected in token patterns.")

        statistics["analysis"] = analysis

    # Save outputs
    save_json(samples, statistics, args.output)

    if args.output_text:
        save_text(samples, statistics, args.output_text)

    if args.output_dir:
        save_full_text(samples, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
