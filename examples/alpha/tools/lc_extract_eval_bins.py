#!/usr/bin/env python3
"""Extract single-document 32k bins from the validation tail of LC packed datasets.

LC-A 조기 검증(위치별 NLL·NIAH)의 데이터 준비 단계. 모델 프로세스가 megatron을
import하지 않도록(HF 로드와의 modelopt 충돌 회피 + 재사용) 여기서 미리 .npy로 뽑는다.

Selection:
  - split "99,1,0"에서 train은 문서 인덱스 앞 99%. 뒤쪽 tail-frac(기본 0.8%)만
    사용해 학습이 본 적 없는 bin만 뽑는다 (경계 오프바이원 여유 포함).
  - pad16 패킹의 pad·문서경계는 모두 EOD(id 0)이므로, EOD가 하나도 없는 bin
    = 단일 문서가 32768을 꽉 채운 순수 청크. 위치별 NLL의 "위치 = 실제 문맥
    길이"가 성립하는 표본이다.

Usage:
  python tools/lc_extract_eval_bins.py \
    --dataset pg19:/path/pg19/data_text_document \
    --dataset edgar:/path/edgar/data_text_document \
    --per-dataset 32 --out outputs/lc_a_early_eval/eval_bins
"""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "backends", "megatron", "Megatron-LM-251125"))

from megatron.core.datasets import indexed_dataset  # noqa: E402

SEQ_LEN = 32768
EOD = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True,
                    help="name:prefix (repeatable)")
    ap.add_argument("--per-dataset", type=int, default=32)
    ap.add_argument("--tail-frac", type=float, default=0.008,
                    help="fraction of trailing docs to draw from (inside the 1%% valid split)")
    ap.add_argument("--out", required=True, help="output prefix (.npy/.json)")
    args = ap.parse_args()

    rows, meta = [], []
    for spec in args.dataset:
        name, prefix = spec.split(":", 1)
        ds = indexed_dataset.IndexedDataset(prefix)
        n = len(ds.sequence_lengths)
        start = int(n * (1.0 - args.tail_frac))
        picked = 0
        scanned = 0
        for j in range(n - 1, start - 1, -1):  # 뒤에서부터: split 경계에서 가장 먼 표본 우선
            if ds.sequence_lengths[j] != SEQ_LEN:
                continue
            tok = ds.get(j)
            scanned += 1
            if (tok == EOD).any():
                continue
            # IndexedDataset.get()은 memmap 뷰를 반환 — dataset 객체가 닫히면
            # 뷰가 무효화되므로(다음 dataset으로 넘어갈 때) 반드시 즉시 복사.
            rows.append(np.array(tok, dtype=np.int32, copy=True))
            meta.append({"source": name, "doc_idx": int(j)})
            picked += 1
            if picked >= args.per_dataset:
                break
        print(f"{name}: total_docs={n} tail_start={start} scanned={scanned} "
              f"picked={picked}/{args.per_dataset}")

    if not rows:
        sys.exit("no qualifying bins found")
    arr = np.stack(rows)
    np.save(args.out + ".npy", arr)
    with open(args.out + ".json", "w") as f:
        json.dump({"seq_len": SEQ_LEN, "rows": meta}, f, indent=1)
    print(f"saved {arr.shape} -> {args.out}.npy")


if __name__ == "__main__":
    main()
