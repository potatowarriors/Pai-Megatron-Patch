#!/usr/bin/env python3
"""Per-position NLL over 32k single-document sequences (LC-A early verification).

측정 원리: 위치 p의 loss는 "문맥 p토큰을 쓸 수 있을 때의 예측 성능". LC 학습이
작동하면 긴 위치 구간의 NLL이 기준선(P3) 대비 내려가고, 격차가 위치와 함께
벌어져야 한다. 도메인 적응 효과(전 위치 균일 하락)와 long-context 효과(위치에
따라 커지는 하락)를 구분하려면 두 모델의 격차-위치 곡선을 볼 것.

megatron import 없음 — 입력은 lc_extract_eval_bins.py가 뽑은 .npy.

Usage:
  python tools/lc_position_nll.py --model <hf_dir> \
    --bins outputs/lc_a_early_eval/eval_bins.npy \
    --meta outputs/lc_a_early_eval/eval_bins.json \
    --out  outputs/lc_a_early_eval/nll_<tag>.json [--device cuda:0]
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

BUCKET = 2048  # 위치 버킷 (2048 → 32k에서 16버킷; 4k 단위는 사후 집계 가능)


def load_model(model_dir, device):
    from transformers import AutoModelForCausalLM
    for impl in ("flash_attention_2", "sdpa"):
        try:
            m = AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16,
                trust_remote_code=True, attn_implementation=impl)
            print(f"loaded with attn_implementation={impl}")
            return m.to(device).eval()
        except Exception as e:  # noqa: BLE001
            print(f"attn_implementation={impl} failed: {type(e).__name__}: {e}")
    raise RuntimeError("could not load model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bins", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="use first N seqs only (smoke)")
    args = ap.parse_args()

    arr = np.load(args.bins)
    meta = json.load(open(args.meta))["rows"]
    if args.limit:
        arr, meta = arr[: args.limit], meta[: args.limit]
    L = arr.shape[1]
    nb = (L + BUCKET - 1) // BUCKET

    model = load_model(args.model, args.device)

    # sums[src][b], counts[src][b]
    sums, counts = {}, {}
    t0 = time.time()
    for i in range(arr.shape[0]):
        ids = torch.from_numpy(arr[i].astype(np.int64))[None].to(args.device)
        src = meta[i]["source"]
        s = sums.setdefault(src, np.zeros(nb))
        c = counts.setdefault(src, np.zeros(nb))
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=False)
            logits = out.logits[0]  # [L, V] bf16
            for lo in range(0, L - 1, 4096):
                hi = min(lo + 4096, L - 1)
                lg = logits[lo:hi].float()
                tg = ids[0, lo + 1: hi + 1]
                loss = F.cross_entropy(lg, tg, reduction="none")  # 목표 위치 lo+1..hi
                pos = np.arange(lo + 1, hi + 1)
                np.add.at(s, pos // BUCKET, loss.float().cpu().numpy())
                np.add.at(c, pos // BUCKET, 1)
            del out, logits
        if (i + 1) % 8 == 0:
            dt = (time.time() - t0) / (i + 1)
            print(f"[{i+1}/{arr.shape[0]}] {dt:.1f}s/seq", flush=True)

    result = {"model": args.model, "n_seqs": int(arr.shape[0]),
              "bucket": BUCKET, "per_source": {}, "overall": {}}
    tot_s, tot_c = np.zeros(nb), np.zeros(nb)
    for src in sums:
        result["per_source"][src] = {
            "bucket_nll": (sums[src] / np.maximum(counts[src], 1)).round(4).tolist(),
            "tokens": counts[src].astype(int).tolist(),
        }
        tot_s += sums[src]
        tot_c += counts[src]
    result["overall"]["bucket_nll"] = (tot_s / np.maximum(tot_c, 1)).round(4).tolist()
    result["overall"]["mean_nll"] = float(tot_s.sum() / tot_c.sum())

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"mean NLL {result['overall']['mean_nll']:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
