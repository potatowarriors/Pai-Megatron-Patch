#!/usr/bin/env python3
"""
convert_parquet_to_jsonl.py — parallel Parquet -> JSONL for v5 pre-tokenization.

Writes ONE .jsonl per input .parquet (same basename) under --output-dir, each line
a single JSON object {"text": <document>}. One-jsonl-per-parquet keeps good file-count
parallelism for the downstream round-robin tokenizer driver (preprocess_stage2_v5.sh).

- Streams each parquet in row-batches (bounded memory; RQA/STEM shards are large).
- Skips rows whose text is None/empty.
- Idempotent: an output .jsonl is considered done only if a sidecar .jsonl.done marker
  exists; partial files from an interrupted run are overwritten.

Usage:
  python3 convert_parquet_to_jsonl.py --input-dir <subset_dir> --output-dir <jsonl_dir> \
      [--text-key text] [--workers 8] [--batch-size 10000]
"""
import os
import io
import json
import glob
import argparse
from multiprocessing import Pool

import pyarrow.parquet as pq


def convert_one(task):
    pq_path, out_path, text_key, batch_size = task
    done_marker = out_path + ".done"
    if os.path.exists(done_marker) and os.path.exists(out_path):
        return (pq_path, -1, None)  # already done
    tmp_path = out_path + ".partial"
    n = 0
    try:
        pf = pq.ParquetFile(pq_path)
        if text_key not in pf.schema_arrow.names:
            return (pq_path, 0, f"column {text_key!r} not in {pf.schema_arrow.names}")
        with io.open(tmp_path, "w", encoding="utf-8") as f:
            for batch in pf.iter_batches(batch_size=batch_size, columns=[text_key]):
                for t in batch.column(0).to_pylist():
                    if t:
                        f.write(json.dumps({"text": t}, ensure_ascii=False))
                        f.write("\n")
                        n += 1
        os.replace(tmp_path, out_path)
        with io.open(done_marker, "w") as f:
            f.write(str(n))
        return (pq_path, n, None)
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return (pq_path, 0, str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="dir containing *.parquet")
    ap.add_argument("--output-dir", required=True, help="dir to write *.jsonl")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=10000)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    parquets = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
    if not parquets:
        raise SystemExit(f"No parquet files under {args.input_dir}")

    tasks = []
    for p in parquets:
        base = os.path.splitext(os.path.basename(p))[0]
        tasks.append((p, os.path.join(args.output_dir, base + ".jsonl"),
                      args.text_key, args.batch_size))

    print(f"convert: {len(tasks)} parquet -> {args.output_dir}  (workers={args.workers})",
          flush=True)
    total_rows, done, failed = 0, 0, 0
    with Pool(processes=args.workers) as pool:
        for pq_path, n, err in pool.imap_unordered(convert_one, tasks):
            if err:
                failed += 1
                print(f"  [FAIL] {os.path.basename(pq_path)}: {err}", flush=True)
            elif n < 0:
                done += 1
                print(f"  [skip] {os.path.basename(pq_path)} (already done)", flush=True)
            else:
                done += 1
                total_rows += n
                print(f"  [ok]   {os.path.basename(pq_path)}: {n} rows", flush=True)
    print(f"DONE convert: {done}/{len(tasks)} files, {total_rows} new rows, {failed} failed",
          flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
