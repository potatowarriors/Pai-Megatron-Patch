#!/usr/bin/env python3
"""
fast_tokenize_v5.py — optimized pre-tokenization for alpha v5 tokenizer.

Replaces `preprocess_data_megatron.py` for large inputs (>100 GB) on the
v5 (fast-only) tokenizer. Uses `tokenizers.Tokenizer.from_file` + `encode_batch`
+ `IndexedDatasetBuilder` directly — no multiprocessing.Pool per-doc IPC.

See `examples/alpha/CLAUDE.md` § "Pre-tokenization Performance" for the
architecture rules and measured throughput table that motivated this script.

Two modes:
  - `jsonl-chunk`     : tokenize one JSONL file (--input) OR many JSONL files
                       (--input-list-file, one path per line) → single .bin/.idx.
                       Multi-file is byte-identical to tokenizing the files
                       concatenated in list order, and is how the stage-2 driver
                       feeds one process a partition of a multi-file dataset.
  - `parquet-subset`  : tokenize a list of parquet files, shuffle rows per
                       parquet with seed, write first half → stage1,
                       second half → stage2 (both .bin/.idx in one pass)

Output is a per-process part. Merge N parts via `merge_indices.py` to get
the final mmap.

Throughput budget (Intel Xeon 8480+, 224 cores, v5 tokenizer):
  ~2M tok/s per process @ RAYON_NUM_THREADS=8.
  Run N such processes in parallel → aggregate up to ~32M tok/s.

Byte-perfect parity with `preprocess_data_megatron.py --patch-tokenizer-type
AlphaTokenizer` is REQUIRED — verify with cmp on a 100-doc sample before
any production run.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _set_thread_env(rayon_threads: int) -> None:
    """Must be called BEFORE importing tokenizers (and ideally pyarrow)."""
    os.environ["RAYON_NUM_THREADS"] = str(rayon_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _load_tokenizer(tokenizer_dir: str):
    """Load Rust tokenizer + derive EOS token id."""
    from tokenizers import Tokenizer
    tok_path = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"tokenizer.json not found at {tok_path}")
    tok = Tokenizer.from_file(tok_path)
    # Read EOS from tokenizer_config.json
    cfg_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    eos_str = cfg.get("eos_token", None)
    if eos_str is None:
        raise RuntimeError(f"eos_token missing in {cfg_path}")
    if isinstance(eos_str, dict):
        eos_str = eos_str["content"]
    eos_id = tok.token_to_id(eos_str)
    if eos_id is None:
        raise RuntimeError(f"EOS token {eos_str!r} not in tokenizer vocab")
    vocab_size = tok.get_vocab_size()
    return tok, eos_id, vocab_size


def _open_builder(prefix: str, dtype):
    """Open IndexedDatasetBuilder at <prefix>_text_document.bin."""
    from megatron.core.datasets import indexed_dataset
    bin_path = f"{prefix}_text_document.bin"
    Path(bin_path).parent.mkdir(parents=True, exist_ok=True)
    return indexed_dataset.IndexedDatasetBuilder(bin_path, dtype=dtype)


def _finalize_builder(builder, prefix: str) -> None:
    idx_path = f"{prefix}_text_document.idx"
    builder.finalize(idx_path)


def _flush_batch(tok, builder, batch_texts, eos_id, append_eod, dtype, stats):
    """Encode a batch of texts and write each doc to builder."""
    import numpy as np
    if not batch_texts:
        return
    encs = tok.encode_batch(batch_texts, add_special_tokens=False)
    for enc in encs:
        ids = enc.ids
        if not ids:
            stats["empty_docs"] += 1
            continue
        if append_eod:
            ids = ids + [eos_id]  # avoid in-place to keep enc immutable
        arr = np.asarray(ids, dtype=dtype)
        builder.add_document(arr, [len(arr)])
        stats["docs"] += 1
        stats["tokens"] += len(arr)


def _progress_log(stats, t0):
    elapsed = time.time() - t0
    docs_s = stats["docs"] / elapsed if elapsed > 0 else 0
    tok_s = stats["tokens"] / elapsed if elapsed > 0 else 0
    print(
        f"  [{elapsed:6.0f}s] docs={stats['docs']:>12,} "
        f"tokens={stats['tokens']/1e9:6.2f}B  "
        f"rate={docs_s/1e3:5.1f}K docs/s  {tok_s/1e6:5.2f}M tok/s",
        flush=True,
    )


# ============================================================
# Mode: jsonl-chunk
# ============================================================
def _resolve_jsonl_inputs(args):
    """Return the ordered list of jsonl files for jsonl-chunk mode.

    Either a single --input file or many files via --input-list-file (one path
    per line). List order defines document order, so it must be deterministic
    (the driver sorts before partitioning)."""
    if args.input_list_file:
        with open(args.input_list_file, "r") as f:
            files = [ln.strip() for ln in f if ln.strip()]
    else:
        files = [args.input]
    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} input jsonl file(s) not found, e.g. {missing[0]}")
    return files


def run_jsonl_chunk(args):
    import numpy as np
    tok, eos_id, vocab_size = _load_tokenizer(args.tokenizer)
    from megatron.core.datasets.indexed_dataset import DType
    dtype = DType.optimal_dtype(vocab_size)
    print(f"tokenizer vocab_size={vocab_size}, dtype={dtype.__name__}, eos_id={eos_id}", flush=True)

    input_files = _resolve_jsonl_inputs(args)
    print(f"jsonl-chunk: {len(input_files)} input file(s) → {args.output_prefix}_text_document.{{bin,idx}}", flush=True)

    builder = _open_builder(args.output_prefix, dtype)
    stats = {"docs": 0, "tokens": 0, "empty_docs": 0, "parse_errors": 0}
    batch = []
    t0 = time.time()
    last_log = t0
    log_every_s = 60

    try:
        for fi, path in enumerate(input_files):
            with open(path, "r") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        text = obj.get(args.text_key, None)
                    except json.JSONDecodeError:
                        stats["parse_errors"] += 1
                        continue
                    if not text:
                        stats["empty_docs"] += 1
                        continue
                    batch.append(text)
                    if len(batch) >= args.batch_size:
                        _flush_batch(tok, builder, batch, eos_id, args.append_eod, dtype, stats)
                        batch = []
                        now = time.time()
                        if now - last_log >= log_every_s:
                            print(f"  [file {fi+1}/{len(input_files)}]", end=" ")
                            _progress_log(stats, t0)
                            last_log = now
        _flush_batch(tok, builder, batch, eos_id, args.append_eod, dtype, stats)
    finally:
        _finalize_builder(builder, args.output_prefix)

    elapsed = time.time() - t0
    print(f"\nDONE jsonl-chunk ({len(input_files)} files) in {elapsed:.1f}s", flush=True)
    print(f"  docs:          {stats['docs']:,}", flush=True)
    print(f"  tokens:        {stats['tokens']:,} ({stats['tokens']/1e9:.2f}B)", flush=True)
    print(f"  empty_docs:    {stats['empty_docs']:,}", flush=True)
    print(f"  parse_errors:  {stats['parse_errors']:,}", flush=True)
    print(f"  throughput:    {stats['tokens']/elapsed/1e6:.2f}M tok/s", flush=True)
    print(f"  output:        {args.output_prefix}_text_document.{{bin,idx}}", flush=True)


# ============================================================
# Mode: parquet-subset
# ============================================================
def run_parquet_subset(args):
    import numpy as np
    import pyarrow.parquet as pq

    tok, eos_id, vocab_size = _load_tokenizer(args.tokenizer)
    from megatron.core.datasets.indexed_dataset import DType
    dtype = DType.optimal_dtype(vocab_size)
    print(f"tokenizer vocab_size={vocab_size}, dtype={dtype.__name__}, eos_id={eos_id}", flush=True)

    builder1 = _open_builder(args.output_prefix_stage1, dtype)
    builder2 = _open_builder(args.output_prefix_stage2, dtype)

    parquet_paths = []
    if args.input_list:
        parquet_paths = [p.strip() for p in args.input_list.split(",") if p.strip()]
    elif args.input_list_file:
        with open(args.input_list_file, "r") as f:
            parquet_paths = [ln.strip() for ln in f if ln.strip()]
    else:
        raise ValueError("--input-list or --input-list-file required for parquet-subset mode")

    print(f"processing {len(parquet_paths)} parquet files", flush=True)

    stats1 = {"docs": 0, "tokens": 0, "empty_docs": 0, "parse_errors": 0}
    stats2 = {"docs": 0, "tokens": 0, "empty_docs": 0, "parse_errors": 0}
    batch1, batch2 = [], []
    t0 = time.time()
    last_log = t0
    log_every_s = 60

    try:
        for pq_idx, pq_path in enumerate(parquet_paths):
            if not os.path.exists(pq_path):
                print(f"  [SKIP] missing parquet: {pq_path}", flush=True)
                continue
            try:
                table = pq.read_table(pq_path, columns=[args.text_key])
            except Exception as e:
                print(f"  [ERROR] reading {pq_path}: {e}", flush=True)
                continue
            schema_names = [f.name for f in table.schema]
            if args.text_key not in schema_names:
                print(f"  [ERROR] {pq_path}: column {args.text_key!r} not in schema {schema_names}", flush=True)
                continue
            texts = table.column(args.text_key).to_pylist()
            n = len(texts)
            # Fresh RNG per parquet — independent of file iteration order
            rng = np.random.default_rng(args.shuffle_seed)
            perm = rng.permutation(n)
            half = n // 2

            print(f"  [{pq_idx+1}/{len(parquet_paths)}] {os.path.basename(pq_path)}: "
                  f"{n:,} rows → s1={half:,}, s2={n-half:,}", flush=True)

            # Stage 1: first half of shuffled rows
            for i in perm[:half]:
                text = texts[i]
                if not text:
                    stats1["empty_docs"] += 1
                    continue
                batch1.append(text)
                if len(batch1) >= args.batch_size:
                    _flush_batch(tok, builder1, batch1, eos_id, args.append_eod, dtype, stats1)
                    batch1 = []

            # Stage 2: second half
            for i in perm[half:]:
                text = texts[i]
                if not text:
                    stats2["empty_docs"] += 1
                    continue
                batch2.append(text)
                if len(batch2) >= args.batch_size:
                    _flush_batch(tok, builder2, batch2, eos_id, args.append_eod, dtype, stats2)
                    batch2 = []

            # Release parquet memory
            del texts, table

            # Periodic progress log
            now = time.time()
            if now - last_log >= log_every_s:
                print("  [STAGE1]", end=" "); _progress_log(stats1, t0)
                print("  [STAGE2]", end=" "); _progress_log(stats2, t0)
                last_log = now

        # Final flush
        _flush_batch(tok, builder1, batch1, eos_id, args.append_eod, dtype, stats1)
        _flush_batch(tok, builder2, batch2, eos_id, args.append_eod, dtype, stats2)
    finally:
        _finalize_builder(builder1, args.output_prefix_stage1)
        _finalize_builder(builder2, args.output_prefix_stage2)

    elapsed = time.time() - t0
    print(f"\nDONE parquet-subset in {elapsed:.1f}s", flush=True)
    for stage_name, s, prefix in [
        ("stage1", stats1, args.output_prefix_stage1),
        ("stage2", stats2, args.output_prefix_stage2),
    ]:
        print(f"  [{stage_name}]", flush=True)
        print(f"    docs:         {s['docs']:,}", flush=True)
        print(f"    tokens:       {s['tokens']:,} ({s['tokens']/1e9:.2f}B)", flush=True)
        print(f"    empty_docs:   {s['empty_docs']:,}", flush=True)
        print(f"    output:       {prefix}_text_document.{{bin,idx}}", flush=True)
    total_tokens = stats1["tokens"] + stats2["tokens"]
    print(f"  throughput:   {total_tokens/elapsed/1e6:.2f}M tok/s aggregate", flush=True)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["jsonl-chunk", "parquet-subset"])
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer dir containing tokenizer.json + tokenizer_config.json")
    parser.add_argument("--text-key", default="text", help="JSON / parquet column key for document text (default: text)")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for encode_batch (default: 5000)")
    parser.add_argument("--rayon-threads", type=int, default=8, help="RAYON_NUM_THREADS (default: 8, the empirical sweet spot)")
    parser.add_argument("--append-eod", action="store_true", help="Append EOS token id at end of each non-empty doc")
    parser.add_argument("--megatron-path", default="/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/backends/megatron/Megatron-LM-251125",
                        help="Megatron-LM-* path (for IndexedDatasetBuilder import)")

    # jsonl-chunk mode args
    parser.add_argument("--input", help="[jsonl-chunk] single input jsonl file path")
    parser.add_argument("--output-prefix", help="[jsonl-chunk] output prefix (writes <prefix>_text_document.{bin,idx})")

    # parquet-subset mode args
    parser.add_argument("--input-list", help="[parquet-subset] comma-separated parquet paths")
    # --input-list-file is shared: jsonl-chunk (many jsonl files) + parquet-subset (parquet paths)
    parser.add_argument("--input-list-file", help="file of input paths, one per line "
                        "([jsonl-chunk] jsonl files, or [parquet-subset] parquet files)")
    parser.add_argument("--output-prefix-stage1", help="[parquet-subset] stage1 output prefix")
    parser.add_argument("--output-prefix-stage2", help="[parquet-subset] stage2 output prefix")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="[parquet-subset] per-parquet row shuffle seed")

    args = parser.parse_args()

    # Set thread env BEFORE importing tokenizers (must be first)
    _set_thread_env(args.rayon_threads)

    # Add megatron path so IndexedDatasetBuilder is importable
    if args.megatron_path not in sys.path:
        sys.path.insert(0, args.megatron_path)

    # Validate mode-specific args
    if args.mode == "jsonl-chunk":
        if not (args.input or args.input_list_file) or not args.output_prefix:
            parser.error("jsonl-chunk requires (--input or --input-list-file) and --output-prefix")
        if args.input and args.input_list_file:
            parser.error("jsonl-chunk: pass only one of --input / --input-list-file")
        run_jsonl_chunk(args)
    elif args.mode == "parquet-subset":
        if not (args.input_list or args.input_list_file):
            parser.error("parquet-subset requires --input-list or --input-list-file")
        if not args.output_prefix_stage1 or not args.output_prefix_stage2:
            parser.error("parquet-subset requires --output-prefix-stage1 and --output-prefix-stage2")
        run_parquet_subset(args)


if __name__ == "__main__":
    main()
