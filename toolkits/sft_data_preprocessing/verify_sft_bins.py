#!/usr/bin/env python3
# Copyright (c) 2026 alpha team. Apache-2.0.
"""SFT idxmap bin 검증 게이트 — 변환 산출물의 소비 계약 일괄 검사.

LC 러너북의 %16 표본검사·scan_internal_eod 게이트와 동형 역할 (SFT 판).
검사 항목 (문서별):
  1. 문서 길이 == 2S 정확
  2. 마커(음수) 존재, 모든 세그먼트 경계(마커+1)가 %grid == 0
     — helper 의 snap_cu_seqlens_to_grid(CP>=2)가 no-op 임을 보장하는 조건
  3. 토큰 반쪽의 EOD(id 0) == 0 — SFT 경로 리셋은 마커 기반이라 EOD 는
     불필요하며, 존재 시 잡탕 EOD(합성 원문 리터럴) 오염 신호 (exit 1)
  4. 라벨 시프트: labels[i] != -100 이면 labels[i] == restore(tokens[i+1])
  5. 복원 토큰이 vocab 범위 내, 꼬리 pad 는 마커 없음

Usage:
  python verify_sft_bins.py --tree /path/sft_packed_64k_pad16 \
      --seq-length 65536 [--bins-per-set 50] [--grid 16]
오염/위반 발견 시 exit 1.
"""
import argparse
import glob
import os
import sys

import numpy as np

IGNORE = -100
EOD_ID = 0
PAD_ID = 1


def check_doc(doc: np.ndarray, S: int, grid: int, vocab: int) -> list:
    errs = []
    if doc.size != 2 * S:
        return [f"doc length {doc.size} != 2S"]
    tokens = doc[:S].astype(np.int64)
    labels = doc[S:].astype(np.int64)

    markers = np.nonzero(tokens < 0)[0]
    if markers.size == 0:
        errs.append("no negative marker")
        return errs
    boundaries = markers + 1
    off_grid = boundaries[boundaries % grid != 0]
    if off_grid.size:
        errs.append(f"off-grid boundaries: {off_grid[:5].tolist()}")

    restored = tokens.copy()
    neg = restored < 0
    restored[neg] = -restored[neg] - 1
    if restored.min() < 0 or restored.max() >= vocab:
        errs.append(f"restored token out of vocab: [{restored.min()}, {restored.max()}]")

    n_eod = int((restored == EOD_ID).sum())
    if n_eod:
        errs.append(f"EOD(id 0) x{n_eod} in token half (잡탕 EOD 오염 의심)")

    tail = tokens[markers[-1] + 1:]
    if tail.size and not (tail == PAD_ID).all():
        errs.append("tail after last marker is not all-pad")

    li = np.nonzero(labels != IGNORE)[0]
    if li.size == 0:
        errs.append("no trainable label in doc")
    else:
        if li.max() + 1 >= S:
            errs.append("trainable label at last position")
        bad = li[labels[li] != restored[li + 1]]
        if bad.size:
            errs.append(f"label-shift mismatch x{bad.size} (first at {bad[0]})")
        if (labels[markers] != IGNORE).any():
            errs.append("non-ignore label at marker position")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True,
                    help="sft_packed_* 트리 (하위 <set>/data_text_document.*)")
    ap.add_argument("--seq-length", type=int, required=True)
    ap.add_argument("--bins-per-set", type=int, default=50)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--vocab-size", type=int, default=163968)
    args = ap.parse_args()

    from megatron.core.datasets.indexed_dataset import IndexedDataset

    prefixes = sorted(glob.glob(os.path.join(args.tree, "*", "data_text_document.idx")))
    if not prefixes:
        print(f"[FAIL] no datasets under {args.tree}")
        sys.exit(1)

    total_bad = 0
    for idx_path in prefixes:
        name = os.path.basename(os.path.dirname(idx_path))
        ds = IndexedDataset(idx_path[:-4])
        n = len(ds)
        picks = sorted(set(np.linspace(0, n - 1, min(args.bins_per_set, n), dtype=int).tolist()))
        bad = 0
        first_err = None
        for d in picks:
            errs = check_doc(np.asarray(ds[d]), args.seq_length, args.grid, args.vocab_size)
            if errs:
                bad += 1
                if first_err is None:
                    first_err = (d, errs[:3])
        status = "OK " if bad == 0 else "BAD"
        print(f"[{status}] {name:46s} docs={n:>7,d} sampled={len(picks):>3d} bad={bad}"
              + (f"  first: doc{first_err[0]} {first_err[1]}" if first_err else ""))
        total_bad += bad

    if total_bad:
        print(f"\n[FAIL] {total_bad} bad docs — 오염/계약 위반. 해당 세트 재변환 필요.")
        sys.exit(1)
    print("\n[PASS] all sampled docs satisfy the SFT bin contract.")


if __name__ == "__main__":
    main()
