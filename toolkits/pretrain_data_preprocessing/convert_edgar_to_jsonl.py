#!/usr/bin/env python3
"""
convert_edgar_to_jsonl.py — EDGAR-corpus(섹션 분리 10-K) -> LC용 JSONL.

EDGAR-corpus 레코드는 본문이 section_1 ... section_15 필드로 쪼개져 있다
(eloukas/edgar-corpus). 사용 시점 문서 = 섹션들을 필드 순서 그대로 "\n\n"으로
결합한 것 (LC_DATASETS.md §3.2 — 전수 분석도 같은 규칙으로 측정했음).

- 입력: <input-root>/<year>/{train,validate,test}.jsonl (1993..2020)
- 출력: <output-dir>/<year>_<split>.jsonl, 각 줄 {"text": ...}
  (연도×스플릿당 1파일 → 다운스트림 라운드로빈 토크나이저 병렬성 유지.
   CPT 용도이므로 train/validate/test 전 스플릿 포함 — LC_DATASETS.md §3.2 기록 참조)
- 빈 섹션은 건너뛰고, 결합 결과가 빈 문서면 스킵(카운트 보고).
- 멱등: .done 마커 + .partial -> os.replace (convert_parquet_to_jsonl.py 컨벤션).

Usage:
  python3 convert_edgar_to_jsonl.py \
      --input-root /home/work/Datasets/LL_datasets/longcontext/en/edgar-corpus \
      --output-dir /home/work/Datasets/LL_datasets/longcontext/en/edgar-corpus/_jsonl \
      [--workers 6]
"""
import argparse
import glob
import io
import json
import os
from multiprocessing import Pool

import orjson


def convert_one(task):
    in_path, out_path = task
    done_marker = out_path + ".done"
    if os.path.exists(done_marker) and os.path.exists(out_path):
        return (in_path, -1, 0, None)
    tmp = out_path + ".partial"
    n = skipped = 0
    try:
        with open(in_path, "rb") as fin, io.open(tmp, "w", encoding="utf-8") as fout:
            for line in fin:
                d = orjson.loads(line)
                parts = [v for k, v in d.items()
                         if k.startswith("section") and isinstance(v, str) and v]
                text = "\n\n".join(parts)
                if not text:
                    skipped += 1
                    continue
                fout.write(json.dumps({"text": text}, ensure_ascii=False))
                fout.write("\n")
                n += 1
        os.replace(tmp, out_path)
        with io.open(done_marker, "w") as f:
            f.write(f"{n} kept, {skipped} empty")
        return (in_path, n, skipped, None)
    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return (in_path, 0, 0, str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = []
    for yd in sorted(glob.glob(os.path.join(args.input_root, "[12][0-9][0-9][0-9]"))):
        year = os.path.basename(yd)
        for split in ("train", "validate", "test"):
            p = os.path.join(yd, f"{split}.jsonl")
            if os.path.exists(p):
                tasks.append((p, os.path.join(args.output_dir,
                                              f"{year}_{split}.jsonl")))
    if not tasks:
        raise SystemExit(f"no year/split jsonl under {args.input_root}")
    print(f"convert edgar: {len(tasks)} files -> {args.output_dir}", flush=True)
    total = failed = 0
    with Pool(args.workers) as pool:
        for in_path, n, skipped, err in pool.imap_unordered(convert_one, tasks):
            name = os.path.basename(os.path.dirname(in_path)) + "/" + \
                os.path.basename(in_path)
            if err:
                failed += 1
                print(f"  [FAIL] {name}: {err}", flush=True)
            elif n < 0:
                print(f"  [skip] {name} (done)", flush=True)
            else:
                total += n
                print(f"  [ok]   {name}: {n} docs ({skipped} empty)", flush=True)
    print(f"DONE edgar convert: {total} docs, {failed} failed", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
