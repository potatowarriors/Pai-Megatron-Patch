#!/usr/bin/env python3
"""
convert_longblocks_to_jsonl.py — LongBlocks doc-QA -> JSONL for LC pre-tokenization.

Emits ONE .jsonl per input .parquet (same basename), each line
{"text": "<document>\n\nQuestion: <question>\n\nAnswer: <answer>"} — the LC CPT
doc-QA format (Nemotron 3 LC-Phase recipe; see examples/alpha/docs/LC_DATASETS.md
§5.1). One sample stays ONE logical document: downstream --append-eod puts the
only EOD at the end, so cross-document attention isolation
(--reset-attention-mask) sees the doc+QA pair as a single unit.

Row filtering (counted and reported per file):
  - document is null  -> skipped. These are the Institutional-Books-1.0 rows
    (~107.8k of 193.9k; not redistributed upstream for licensing reasons). They
    need a separate reconstruction pass (reconstruct_longblocks_ib.py, streaming
    barcode-id join against institutional/institutional-books-1.0).
  - question/answer empty -> skipped (a handful of rows have answer == "").
  - --languages (default "alpha"): keep only rows whose `language` is in the
    set; the special token "code" keeps all Stack-Edu rows (their language
    labels are programming languages). "alpha" = en + ko + code + the 20
    FineWeb2-HQ languages actually present in the v5 stage1/stage2 blend
    (verified 2026-08-01 by language-detecting 1.5k decoded docs from the
    tokenized fineweb2hq .bin: fr de es cs nl tr el it pt pl id vi da hu ar
    ja ru sv fa zh, each ~2.5-7.7%). Languages alpha never trained on
    (uk hi ca ro fi sk sl mt hr no ga gl lt in LongBlocks) are dropped.
    Pass --languages all to disable, or a custom comma list.

Teacher-response columns (response_*) are intentionally dropped — CPT uses only
grounded doc+Q+A; distillation targets belong to the later SFT stage.

Usage:
  python3 convert_longblocks_to_jsonl.py \
      --input-dir  /home/work/Datasets/LL_datasets/longcontext/en/LongBlocks/data \
      --output-dir /home/work/Datasets/LL_datasets/longcontext/en/LongBlocks/_jsonl \
      [--plain] [--workers 8] [--batch-size 2000]

--plain drops the English "Question:/Answer:" labels and joins with blank lines
only (data is 47-language; default keeps labels for an unambiguous QA boundary).
"""
import argparse
import glob
import io
import json
import os
from multiprocessing import Pool

import pyarrow.parquet as pq

COLS = ["document", "question", "answer", "language", "source"]

# en + ko + all 20 FineWeb2-HQ languages in the alpha v5 blend + Stack-Edu code
ALPHA_LANGS = {
    "en", "ko", "code",
    "fr", "de", "es", "cs", "nl", "tr", "el", "it", "pt", "pl",
    "id", "vi", "da", "hu", "ar", "ja", "ru", "sv", "fa", "zh",
}


def keep_language(lang, source, languages):
    if languages is None:
        return True
    if "code" in languages and source == "Stack-Edu":
        return True
    return lang in languages


def convert_one(task):
    pq_path, out_path, plain, batch_size, languages = task
    done_marker = out_path + ".done"
    if os.path.exists(done_marker) and os.path.exists(out_path):
        return (pq_path, -1, 0, None)
    tmp_path = out_path + ".partial"
    n = skipped_null = skipped_empty = skipped_lang = 0
    try:
        pf = pq.ParquetFile(pq_path)
        missing = [c for c in COLS if c not in pf.schema_arrow.names]
        if missing:
            return (pq_path, 0, 0, f"missing columns {missing}")
        with io.open(tmp_path, "w", encoding="utf-8") as f:
            for batch in pf.iter_batches(batch_size=batch_size, columns=COLS):
                docs, qs, ans, langs, srcs = (batch.column(i).to_pylist()
                                              for i in range(5))
                for d, q, a, lg, sc in zip(docs, qs, ans, langs, srcs):
                    if not keep_language(lg, sc, languages):
                        skipped_lang += 1
                        continue
                    if not d:
                        skipped_null += 1
                        continue
                    if not q or not a:
                        skipped_empty += 1
                        continue
                    if plain:
                        text = f"{d}\n\n{q}\n\n{a}"
                    else:
                        text = f"{d}\n\nQuestion: {q}\n\nAnswer: {a}"
                    f.write(json.dumps({"text": text}, ensure_ascii=False))
                    f.write("\n")
                    n += 1
        os.replace(tmp_path, out_path)
        with io.open(done_marker, "w") as f:
            f.write(f"{n} kept, {skipped_null} null-doc, {skipped_empty} empty-qa, "
                    f"{skipped_lang} lang-filtered")
        return (pq_path, n, skipped_null + skipped_empty + skipped_lang, None)
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return (pq_path, 0, 0, str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="dir containing *.parquet")
    ap.add_argument("--output-dir", required=True, help="dir to write *.jsonl")
    ap.add_argument("--plain", action="store_true",
                    help="no Question:/Answer: labels, blank-line joins only")
    ap.add_argument("--languages", default="alpha",
                    help='"alpha" (default) = en+ko+code+20 FineWeb2-HQ langs; '
                         'or a comma list of codes ("code" keeps Stack-Edu rows); '
                         '"all" disables filtering')
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2000)
    args = ap.parse_args()
    if args.languages.strip() == "all":
        languages = None
    elif args.languages.strip() == "alpha":
        languages = set(ALPHA_LANGS)
    else:
        languages = set(x.strip() for x in args.languages.split(",") if x.strip())

    os.makedirs(args.output_dir, exist_ok=True)
    parquets = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
    if not parquets:
        raise SystemExit(f"No parquet files under {args.input_dir}")

    tasks = []
    for p in parquets:
        base = os.path.splitext(os.path.basename(p))[0]
        tasks.append((p, os.path.join(args.output_dir, base + ".jsonl"),
                      args.plain, args.batch_size, languages))

    print(f"convert: {len(tasks)} parquet -> {args.output_dir}  "
          f"(workers={args.workers}, plain={args.plain}, "
          f"languages={sorted(languages) if languages else 'all'})", flush=True)
    total_rows = total_skipped = done = failed = 0
    with Pool(processes=args.workers) as pool:
        for pq_path, n, skipped, err in pool.imap_unordered(convert_one, tasks):
            if err:
                failed += 1
                print(f"  [FAIL] {os.path.basename(pq_path)}: {err}", flush=True)
            elif n < 0:
                done += 1
                print(f"  [skip] {os.path.basename(pq_path)} (already done)", flush=True)
            else:
                done += 1
                total_rows += n
                total_skipped += skipped
                print(f"  [ok]   {os.path.basename(pq_path)}: {n} kept, "
                      f"{skipped} skipped", flush=True)
    print(f"DONE convert: {done}/{len(tasks)} files, {total_rows} rows kept, "
          f"{total_skipped} skipped (null-doc Institutional rows need the "
          f"reconstruction pass), {failed} failed", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
