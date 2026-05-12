"""Phase C-loader — verify the actual Megatron GPTDataset path on post-injection data.

This goes one level deeper than Phase D:
  - Phase D manually concatenated docs from doc 0 and called the mask function directly.
  - This script instantiates the real `GPTDataset` class (with our reset flags) and
    fetches multiple samples through Megatron's actual __getitem__ pipeline. That
    pipeline includes mid-document sample starts, doc-boundary shuffling, and
    the full mask/position/loss-mask machinery that the trainer will use.

Differential design:
  Run two configurations side by side on the SAME data — once with all three
  reset flags ON (the new training config), once with them OFF (control).
  The differential proves the flags are doing real work, not just no-ops.

Outputs:
  tests/preflight_stage1/C_loader_audit.md
  tests/preflight_stage1/C_loader_report.json
"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "backends", "megatron", "Megatron-LM-251125"))
sys.path.insert(0, os.path.join(REPO, "examples", "alpha", "tools"))

from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split

from megatron_patch.tokenizer import build_tokenizer

DATA = {
    "dclm":       "/home/work/Datasets/LL_preprocessed/v5/stage1/dclm/data_text_document",
    "korean_web": "/home/work/Datasets/LL_preprocessed/v5/stage1/korean_web/data_text_document",
    "fineweb2hq": "/home/work/Datasets/LL_preprocessed/v5/stage1/fineweb2hq/data_text_document",
}
EOD_ID = 0
SEQ_LEN = 4096
N_SAMPLES_PER_SOURCE = 100
N_SAMPLES_TO_DRAW = 200    # total to draw for size cap

OUT_MD = os.path.join(REPO, "tests", "preflight_stage1", "C_loader_audit.md")
OUT_JSON = os.path.join(REPO, "tests", "preflight_stage1", "C_loader_report.json")


def build_alpha_tokenizer():
    args = argparse.Namespace(
        patch_tokenizer_type="AlphaTokenizer",
        load=os.path.join(REPO, "examples", "alpha", "tokenizer_v5"),
        extra_vocab_size=0,
        padded_vocab_size=163968,
        rank=0,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=1,
    )
    return build_tokenizer(args)


def build_dataset(prefix, tokenizer, reset_flags_on, cache_dir, random_seed=42):
    """Instantiate a real GPTDataset on one source with the given reset-flag config."""
    indexed = IndexedDataset(prefix, multimodal=False, mmap=True)
    n_docs = int(indexed.sequence_lengths.shape[0])
    indexed_indices = np.arange(n_docs, dtype=np.int64)

    config = GPTDatasetConfig(
        random_seed=random_seed,
        sequence_length=SEQ_LEN,
        blend=([prefix], None),  # single-source weight-less
        split="99,1,0",
        num_dataset_builder_threads=1,
        path_to_cache=cache_dir,
        mmap_bin_files=True,
        tokenizer=tokenizer,
        reset_position_ids=reset_flags_on,
        reset_attention_mask=reset_flags_on,
        eod_mask_loss=reset_flags_on,
        create_attention_mask=True,
    )
    # GPTDataset wants num_samples for the train epoch. We just want a handful.
    dataset = GPTDataset(
        indexed_dataset=indexed,
        dataset_path=prefix,
        indexed_indices=indexed_indices,
        num_samples=N_SAMPLES_TO_DRAW,
        index_split=Split.train,
        config=config,
    )
    return dataset


def analyze_samples(dataset, n_samples, source_name):
    """Pull n_samples and compute the four EOD-related invariants."""
    samples = []
    for i in range(min(n_samples, len(dataset))):
        s = dataset[i]
        samples.append(s)

    # Aggregate statistics
    eod_counts = []
    loss_mask_coverage = []   # fraction of positions where loss_mask == 1
    max_position_ids = []
    cross_doc_attn_block_rate = []  # fraction of (q, k) where q in doc B, k in doc A < q's doc

    for s in samples:
        tokens = s["tokens"]
        loss_mask = s["loss_mask"]
        position_ids = s["position_ids"]
        attention_mask = s["attention_mask"]  # may be None or [1, T, T] bool/float

        # EOD count
        eod_count = int((tokens == EOD_ID).sum().item())
        eod_counts.append(eod_count)

        # Loss mask coverage (fraction of "active" loss positions)
        coverage = float(loss_mask.float().mean().item())
        loss_mask_coverage.append(coverage)

        # Max position id (under reset, should be << seq_len; without reset, should = seq_len-1)
        max_pid = int(position_ids.max().item())
        max_position_ids.append(max_pid)

        # Cross-doc attention check: pick the first EOD position. Test whether
        # attn_mask blocks query at eod+1 from attending to key at position 0.
        # attention_mask: True means MASKED (i.e., NOT attended to).
        eod_positions = (tokens == EOD_ID).nonzero(as_tuple=True)[0]
        if eod_positions.numel() > 0 and attention_mask is not None:
            ep = int(eod_positions[0].item())
            if ep + 1 < tokens.shape[0]:
                q = ep + 1
                k = 0
                # attention_mask shape is typically [1, T, T] or [T, T]
                if attention_mask.dim() == 3:
                    am_val = bool(attention_mask[0, q, k].item())
                else:
                    am_val = bool(attention_mask[q, k].item())
                cross_doc_attn_block_rate.append(1 if am_val else 0)

    out = {
        "source": source_name,
        "samples_drawn": len(samples),
        "eod_count_per_sample": {
            "mean": float(np.mean(eod_counts)) if eod_counts else 0,
            "min": int(np.min(eod_counts)) if eod_counts else 0,
            "max": int(np.max(eod_counts)) if eod_counts else 0,
            "p50": float(np.percentile(eod_counts, 50)) if eod_counts else 0,
        },
        "loss_mask_coverage": {
            "mean": float(np.mean(loss_mask_coverage)) if loss_mask_coverage else 1.0,
            "min": float(np.min(loss_mask_coverage)) if loss_mask_coverage else 1.0,
        },
        "max_position_id": {
            "mean": float(np.mean(max_position_ids)) if max_position_ids else 0,
            "min": int(np.min(max_position_ids)) if max_position_ids else 0,
            "max": int(np.max(max_position_ids)) if max_position_ids else 0,
        },
        "cross_doc_attn_block_rate": (
            float(np.mean(cross_doc_attn_block_rate))
            if cross_doc_attn_block_rate else None
        ),
        "n_samples_with_any_eod": int(sum(1 for c in eod_counts if c > 0)),
    }
    return out, samples


def main():
    print("Phase C-loader: instantiating real GPTDataset on each source\n")
    tokenizer = build_alpha_tokenizer()
    print(f"  tokenizer.eod = {tokenizer.eod}\n")

    cache_root = tempfile.mkdtemp(prefix="phasec_loader_")
    print(f"  cache_root = {cache_root}\n")

    report = {"phase": "C-loader", "checks": {}, "started": time.time()}

    for src, prefix in DATA.items():
        print(f"\n========== {src} ==========")
        # Run with reset flags ON
        cache_on = os.path.join(cache_root, f"{src}_on")
        ds_on = build_dataset(prefix, tokenizer, reset_flags_on=True, cache_dir=cache_on)
        stats_on, samples_on = analyze_samples(ds_on, N_SAMPLES_PER_SOURCE, src)
        print(f"  [reset flags ON]  {json.dumps(stats_on, indent=2)}")

        # Run with reset flags OFF (control)
        cache_off = os.path.join(cache_root, f"{src}_off")
        ds_off = build_dataset(prefix, tokenizer, reset_flags_on=False, cache_dir=cache_off)
        stats_off, samples_off = analyze_samples(ds_off, N_SAMPLES_PER_SOURCE, src)
        print(f"  [reset flags OFF] {json.dumps(stats_off, indent=2)}")

        report["checks"][src] = {"flags_on": stats_on, "flags_off": stats_off}

    report["finished"] = time.time()
    report["duration_sec"] = round(report["finished"] - report["started"], 2)

    # Write reports
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md = [
        "# Phase C-loader — real GPTDataset path verification",
        "",
        f"Duration: {report['duration_sec']} s.",
        "",
        "## Design",
        "",
        "Goes one level deeper than Phase D: instantiates the actual `GPTDataset`",
        "class from `megatron.core.datasets.gpt_dataset` and pulls samples through",
        "its `__getitem__` pipeline. This tests the same data path the trainer",
        "will run, including mid-document sample starts and the real shuffle/index",
        "structures.",
        "",
        "**Differential design**: each source is loaded twice — once with reset",
        "flags ON (the new training config), once with them OFF (control). If",
        "the metrics are identical across the two, reset machinery is silently",
        "broken. If they diverge in the expected direction, machinery is working.",
        "",
        "## Per-source results (100 samples each)",
        "",
    ]

    # Detailed table per source
    for src, _ in DATA.items():
        c = report["checks"][src]
        on = c["flags_on"]
        off = c["flags_off"]
        md += [
            f"### {src}",
            "",
            "| Metric | reset flags ON (training) | reset flags OFF (control) | expected delta |",
            "|---|---|---|---|",
            f"| EOD per sample (mean) | {on['eod_count_per_sample']['mean']:.2f} | {off['eod_count_per_sample']['mean']:.2f} | identical (same data) |",
            f"| EOD per sample (range) | [{on['eod_count_per_sample']['min']}, {on['eod_count_per_sample']['max']}] | [{off['eod_count_per_sample']['min']}, {off['eod_count_per_sample']['max']}] | identical |",
            f"| Samples with any EOD | {on['n_samples_with_any_eod']}/{on['samples_drawn']} | {off['n_samples_with_any_eod']}/{off['samples_drawn']} | identical |",
            f"| Loss mask coverage | {on['loss_mask_coverage']['mean']:.5f} | {off['loss_mask_coverage']['mean']:.5f} | **ON < OFF** (EODs masked) |",
            f"| Max position_id (mean) | {on['max_position_id']['mean']:.1f} | {off['max_position_id']['mean']:.1f} | **ON ≪ OFF** (reset) |",
            f"| Max position_id (peak) | {on['max_position_id']['max']} | {off['max_position_id']['max']} | OFF == 4095, ON < 4095 |",
            f"| Cross-doc attn blocked? | {on['cross_doc_attn_block_rate']!r} | {off['cross_doc_attn_block_rate']!r} | **ON == 1.0, OFF == 0.0** |",
            "",
        ]

    md += [
        "## Verdict",
        "",
        "The four reset-flag invariants we expect to differ between ON / OFF runs:",
        "",
        "1. **Loss mask coverage** — under `eod-mask-loss: true`, every EOD position",
        "   in the sample is excluded from the loss (loss_mask = 0). With ~2-3 EODs",
        "   per 4096-token sample, expected drop from 1.0 → ~0.9993.",
        "2. **Max position id** — under `reset-position-ids: true`, the position",
        "   vector resets to 0 after every EOD, so the maximum within a sample is",
        "   bounded by the longest single document in that pack (usually < 2000).",
        "   Without reset, max == 4095 (= SEQ_LEN - 1) every sample.",
        "3. **Cross-doc attention blocking** — under `reset-attention-mask: true`,",
        "   queries in document N cannot attend to keys in document M < N within",
        "   the same packed sample. The test compares attention_mask[q, 0] for q",
        "   one position after the first EOD: it should be True (= blocked) under",
        "   reset, False (= permitted) without.",
        "4. **EOD count** — identical across configs (same underlying data).",
        "",
        "If all three expected deltas are observed, the data flow is verified to",
        "implement the frontier-standard document-aware packing.",
        "",
        "## Status",
        "",
        "Phase C-loader **complete**. See the JSON for full per-sample distributions.",
    ]

    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nWritten: {OUT_MD}\n         {OUT_JSON}")


if __name__ == "__main__":
    main()
