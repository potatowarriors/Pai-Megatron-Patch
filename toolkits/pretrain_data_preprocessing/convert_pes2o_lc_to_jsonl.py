#!/usr/bin/env python3
"""
convert_pes2o_lc_to_jsonl.py — peS2o v3에서 LC 유효분만 추출 -> JSONL.

peS2o v3 train은 s2ag(초록, ~8.8B tok — LC 무관) + s2orc(본문, 63.5B tok — 그러나
mean 5.8k tok으로 대부분 짧음)이다. LC 블렌드가 쓰는 것은 s2orc의 ≥16k tok 꼬리
(≈3.4B tok, 158k docs — LC_DATASETS.md §3.3/§5.1)뿐이므로, 전체 307GB를 토크나이즈
하는 대신 여기서 소스·길이로 필터링해 LC 유효분만 emit한다.

필터: source == */s2orc AND len(text) >= --min-chars
  기본 77000 chars ≈ 16k tokens (실측 s2orc 4.844 chars/tok, LC_DATASETS.md §2).

Filler 티어 (--filler-rate > 0): 주 필터만 쓰면 모든 문서가 16k~32k tok이라
bestfit_pack(32k)이 bin당 1문서 + 평균 ~10k tok 패딩으로 fill 77%에 그친다
(2026-08-06 실측: pad 1.19B tok). 갭을 채우도록 [--filler-min-chars,
--filler-max-chars) 구간(기본 20k~68k chars ≈ 4k~14k tok) 문서를
--filler-rate 확률로 추가 채택한다. 샘플링은 doc id의 crc32 기반 결정적
해시(재현 가능). rate 기본 0.03 ≈ 갭 질량(~1.2B tok)에 맞춘 값.

- 입력: <input-dir>/train-*.zst (zstd jsonl). valid-* 는 제외(held-out 보존).
- 출력: <output-dir>/<shard>.jsonl — 비어 있지 않은 샤드만 파일 생성
  (초반 샤드는 s2ag 위주라 0건; .done 마커는 전 샤드에 기록).
- 멱등: .done + .partial -> os.replace 컨벤션.

Usage:
  python3 convert_pes2o_lc_to_jsonl.py \
      --input-dir  /home/work/Datasets/LL_datasets/longcontext/en/peS2o/data/v3 \
      --output-dir /home/work/Datasets/LL_datasets/longcontext/en/peS2o/_jsonl_lc \
      [--min-chars 77000] [--workers 8]
"""
import argparse
import glob
import io
import json
import os
from multiprocessing import Pool

import orjson
import zstandard

_MIN = 77000


def convert_one(task):
    in_path, out_path, min_chars, f_min, f_max, f_rate = task
    done_marker = out_path + ".done"
    if os.path.exists(done_marker):
        return (in_path, -1, 0, None)
    tmp = out_path + ".partial"
    n = seen = 0
    try:
        import zlib
        dctx = zstandard.ZstdDecompressor()
        with open(in_path, "rb") as fin, io.open(tmp, "w", encoding="utf-8") as fout:
            reader = io.BufferedReader(dctx.stream_reader(fin), buffer_size=1 << 24)
            for line in reader:
                d = orjson.loads(line)
                seen += 1
                if not d.get("source", "").endswith("s2orc"):
                    continue
                t = d.get("text") or ""
                L = len(t)
                keep = L >= min_chars
                if not keep and f_rate > 0 and f_min <= L < f_max:
                    h = zlib.crc32(str(d.get("id", L)).encode()) & 0xFFFFFFFF
                    keep = (h / 0x100000000) < f_rate
                if not keep:
                    continue
                fout.write(json.dumps({"text": t}, ensure_ascii=False))
                fout.write("\n")
                n += 1
        if n > 0:
            os.replace(tmp, out_path)
        else:
            os.remove(tmp)
        with io.open(done_marker, "w") as f:
            f.write(f"{n} kept of {seen}")
        return (in_path, n, seen, None)
    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return (in_path, 0, 0, str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-chars", type=int, default=_MIN)
    ap.add_argument("--filler-min-chars", type=int, default=20000)
    ap.add_argument("--filler-max-chars", type=int, default=68000)
    ap.add_argument("--filler-rate", type=float, default=0.0,
                    help="0=off. 4k~14k tok 문서를 이 확률로 filler 채택 "
                         "(32k 패킹 갭 충전용; 권장 0.03)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(args.input_dir, "train-*.zst")))
    if not shards:
        raise SystemExit(f"no train-*.zst under {args.input_dir}")
    tasks = [(s, os.path.join(args.output_dir,
                              os.path.basename(s).replace(".zst", ".jsonl")),
              args.min_chars, args.filler_min_chars, args.filler_max_chars,
              args.filler_rate) for s in shards]
    print(f"convert pes2o-lc: {len(tasks)} shards -> {args.output_dir} "
          f"(min-chars {args.min_chars}, filler rate {args.filler_rate} "
          f"[{args.filler_min_chars},{args.filler_max_chars}))", flush=True)
    total = failed = done_cnt = 0
    with Pool(args.workers) as pool:
        for in_path, n, seen, err in pool.imap_unordered(convert_one, tasks):
            done_cnt += 1
            if err:
                failed += 1
                print(f"  [FAIL] {os.path.basename(in_path)}: {err}", flush=True)
            elif n < 0:
                pass  # already done
            else:
                total += n
                if n or done_cnt % 20 == 0:
                    print(f"  [{done_cnt}/{len(tasks)}] "
                          f"{os.path.basename(in_path)}: {n} kept / {seen}",
                          flush=True)
    print(f"DONE pes2o-lc convert: {total} docs kept, {failed} failed", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
