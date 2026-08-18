#!/usr/bin/env python3
"""
reconstruct_longblocks_ib.py — restore LongBlocks Institutional-Books documents
and emit LC doc-QA JSONL, without mirroring the 946GB source corpus.

LongBlocks does not redistribute Institutional-Books-1.0 documents (licensing);
those rows carry document=null and `id` = the source book barcode. This script
adapts the official reconstruction snippet from the LongBlocks dataset card:

  1. Build barcode -> [(question, answer), ...] from the LOCAL LongBlocks
     parquet copy. Institutional rows are en/de/fr — all three are alpha
     blend languages (fineweb2hq includes de/fr), so the default keeps all
     107,817 rows; --languages can narrow this.
  2. STREAM institutional/institutional-books-1.0 (gated auto; the HF account
     must have accepted access on the dataset page once), matching barcodes.
     Column projection keeps transfer bounded; nothing is mirrored to disk.
  3. For each matched book pick the better OCR text (ocr_score_src vs _gen,
     same rule as the official snippet), join pages, and emit
     {"text": "<document>\n\nQuestion: <q>\n\nAnswer: <a>"} — the same format
     as convert_longblocks_to_jsonl.py, one sample per QA row, single logical
     document (EOD only at tokenize time; LC_DATASETS.md §5.1 invariant).

Output: <output-dir>/institutional-<i>-of-<N>.jsonl (one per worker) with
.done sidecars, alongside the 37 files from convert_longblocks_to_jsonl.py so
the round-robin tokenizer driver picks all of them up together.

Interrupted runs restart whole worker shards (streaming has no cheap resume);
finished shards are skipped via their .done marker.

Usage:
  python3 reconstruct_longblocks_ib.py \
      --longblocks-dir /home/work/Datasets/LL_datasets/longcontext/en/LongBlocks/data \
      --output-dir     /home/work/Datasets/LL_datasets/longcontext/en/LongBlocks/_jsonl \
      [--languages en,ko] [--num-proc 8] [--plain]
"""
import argparse
import glob
import io
import json
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import pyarrow.parquet as pq

IB_SOURCE = "Institutional-Books-1.0"
IB_REPO = "institutional/institutional-books-1.0"
IB_COLUMNS = ["barcode_src", "text_by_page_src", "text_by_page_gen",
              "ocr_score_src", "ocr_score_gen"]


def build_qa_map(longblocks_dir, languages):
    qa_by_barcode = defaultdict(list)
    skipped_lang = skipped_empty = 0
    cols = ["id", "source", "language", "question", "answer"]
    for p in sorted(glob.glob(os.path.join(longblocks_dir, "*.parquet"))):
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=2000, columns=cols):
            ids, srcs, langs, qs, ans = (batch.column(i).to_pylist()
                                         for i in range(5))
            for i, s, lg, q, a in zip(ids, srcs, langs, qs, ans):
                if s != IB_SOURCE or not i:
                    continue
                if languages is not None and lg not in languages:
                    skipped_lang += 1
                    continue
                if not q or not a:
                    skipped_empty += 1
                    continue
                qa_by_barcode[i].append((q, a))
    return qa_by_barcode, skipped_lang, skipped_empty


# Worker globals (fork-inherited; the qa map is read-only)
_G = {}


def _init(qa_by_barcode, num_shards, out_dir, plain):
    _G.update(qa=qa_by_barcode, num_shards=num_shards, out_dir=out_dir,
              plain=plain)


def stream_shard(index):
    import datasets
    qa, num_shards = _G["qa"], _G["num_shards"]
    out_path = os.path.join(_G["out_dir"],
                            f"institutional-{index:04d}-of-{num_shards:04d}.jsonl")
    done_marker = out_path + ".done"
    if os.path.exists(done_marker) and os.path.exists(out_path):
        return (index, -1, 0, None)
    tmp = out_path + ".partial"
    n_rows = n_books = 0
    t0 = time.time()
    try:
        ds = datasets.load_dataset(IB_REPO, split="train", streaming=True)
        ds = ds.select_columns(IB_COLUMNS).shard(num_shards=num_shards, index=index)
        with io.open(tmp, "w", encoding="utf-8") as f:
            for book in ds:
                rows = qa.get(book["barcode_src"])
                if not rows:
                    continue
                use_src = (book["ocr_score_src"] or 0) >= (book["ocr_score_gen"] or 0)
                pages = book["text_by_page_src"] if use_src else book["text_by_page_gen"]
                document = "".join(pages).strip()
                if not document:
                    continue
                n_books += 1
                for q, a in rows:
                    if _G["plain"]:
                        text = f"{document}\n\n{q}\n\n{a}"
                    else:
                        text = f"{document}\n\nQuestion: {q}\n\nAnswer: {a}"
                    f.write(json.dumps({"text": text}, ensure_ascii=False))
                    f.write("\n")
                    n_rows += 1
        os.replace(tmp, out_path)
        with io.open(done_marker, "w") as f:
            f.write(f"{n_rows} rows from {n_books} books, "
                    f"{(time.time()-t0)/60:.1f} min")
        return (index, n_rows, n_books, None)
    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return (index, 0, 0, f"{type(e).__name__}: {str(e)[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--longblocks-dir", required=True,
                    help="local LongBlocks parquet dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--languages", default="en,de,fr",
                    help='comma list to keep ("all" disables). Institutional '
                         'rows are en/de/fr — all alpha blend languages '
                         '(fineweb2hq covers de/fr), so default keeps all.')
    ap.add_argument("--num-proc", type=int, default=8)
    ap.add_argument("--plain", action="store_true")
    args = ap.parse_args()
    languages = None if args.languages.strip() == "all" else \
        set(x.strip() for x in args.languages.split(",") if x.strip())

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[1/2] building barcode->QA map from {args.longblocks_dir} "
          f"(languages={sorted(languages) if languages else 'all'})", flush=True)
    qa, skipped_lang, skipped_empty = build_qa_map(args.longblocks_dir, languages)
    n_rows = sum(len(v) for v in qa.values())
    print(f"      {len(qa):,} barcodes / {n_rows:,} QA rows to reconstruct "
          f"({skipped_lang:,} lang-filtered, {skipped_empty:,} empty-qa)", flush=True)

    print(f"[2/2] streaming {IB_REPO} with {args.num_proc} shard workers ...",
          flush=True)
    t0 = time.time()
    total_rows = total_books = failed = 0
    with Pool(processes=args.num_proc, initializer=_init,
              initargs=(dict(qa), args.num_proc, args.output_dir,
                        args.plain)) as pool:
        for index, n, books, err in pool.imap_unordered(
                stream_shard, range(args.num_proc)):
            if err:
                failed += 1
                print(f"  [FAIL] shard {index}: {err}", flush=True)
            elif n < 0:
                print(f"  [skip] shard {index} (already done)", flush=True)
            else:
                total_rows += n
                total_books += books
                print(f"  [ok]   shard {index}: {n:,} rows / {books:,} books "
                      f"({(time.time()-t0)/60:.0f} min elapsed)", flush=True)
    print(f"DONE reconstruct: {total_rows:,} rows from {total_books:,} books, "
          f"{failed} shard failures (expected rows ~= map size {n_rows:,}; "
          f"missing barcodes = books absent upstream)", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
