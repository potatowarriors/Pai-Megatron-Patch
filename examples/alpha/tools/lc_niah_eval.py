#!/usr/bin/env python3
"""Needle-in-a-haystack retrieval eval for base checkpoints (LC-A early verification).

base 모델용 completion-style NIAH: 실문서 haystack(pg19/edgar 검증 tail 청크) 속에
"magic number" 문장을 깊이별로 심고 greedy 생성으로 회수율을 잰다. 합성이라
외부 데이터·chat template 불필요. 조립은 전부 토큰 공간에서 수행(재토큰화 드리프트
차단).

Usage:
  python tools/lc_niah_eval.py --model <hf_dir> \
    --hay outputs/lc_a_early_eval/eval_bins.npy \
    --out outputs/lc_a_early_eval/niah_<tag>.json \
    [--lengths 4096,8192,16384,32768] [--depths 0.1,0.25,0.5,0.75,0.9] [--trials 8]
"""

import argparse
import json
import time

import numpy as np
import torch

PREAMBLE = "The following is a long document.\n\n"
NEEDLE_T = "\nThe special magic number mentioned in this document is {num}.\n"
QUERY = ("\n\nQuestion: What is the special magic number mentioned in this "
         "document?\nAnswer: The special magic number mentioned in this document is")


def load(model_dir, device, max_gpu_mem=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    for impl in ("flash_attention_2", "sdpa"):
        try:
            # device="auto": 레이어를 가시 GPU 전체에 분산 (512K처럼 단일 GPU
            # 활성값 한계를 넘는 길이용). 입력은 cuda:0(첫 레이어 쪽)에 두면
            # accelerate 훅이 레이어 경계에서 넘겨준다.
            # 주의: max_memory 없이는 auto가 GPU0에 전량 탐욕 적재해 분산이 안 됨
            # (512K GDN 커널의 일시 작업공간 ~40GB가 못 들어가 OOM — 08-27 실측).
            # --max-gpu-mem "18GiB"로 가중치 상한을 걸어 강제 분산.
            kw = dict(torch_dtype=torch.bfloat16, trust_remote_code=True,
                      attn_implementation=impl)
            if device == "auto":
                mm = None
                if max_gpu_mem:
                    mm = {i: max_gpu_mem for i in range(torch.cuda.device_count())}
                m = AutoModelForCausalLM.from_pretrained(
                    model_dir, device_map="auto", max_memory=mm, **kw)
                print(f"loaded with attn_implementation={impl}, device_map=auto, "
                      f"max_memory={mm}")
                print("hf_device_map:", {k: v for k, v in list(m.hf_device_map.items())[:3]},
                      "...", {k: v for k, v in list(m.hf_device_map.items())[-2:]})
                return tok, m.eval()
            m = AutoModelForCausalLM.from_pretrained(model_dir, **kw)
            print(f"loaded with attn_implementation={impl}")
            return tok, m.to(device).eval()
        except Exception as e:  # noqa: BLE001
            print(f"attn_implementation={impl} failed: {type(e).__name__}: {e}")
    raise RuntimeError("could not load model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hay", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lengths", default="4096,8192,16384,32768")
    ap.add_argument("--depths", default="0.1,0.25,0.5,0.75,0.9")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--max-gpu-mem", default=None,
                    help='--device auto 시 GPU당 가중치 상한 (예 "18GiB") — 분산 강제')
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [float(x) for x in args.depths.split(",")]
    hay = np.load(args.hay)
    tok, model = load(args.model, args.device, args.max_gpu_mem)
    if args.device == "auto":
        args.device = "cuda:0"  # 입력은 첫 레이어 쪽 GPU에 (accelerate 훅이 이관)
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731

    pre = enc(PREAMBLE)
    query = enc(QUERY)
    rng = np.random.default_rng(args.seed)
    results, t0, done = {}, time.time(), 0
    total = len(lengths) * len(depths) * args.trials
    for L in lengths:
        for d in depths:
            hits = []
            for _ in range(args.trials):
                num = int(rng.integers(100000, 1000000))
                needle = enc(NEEDLE_T.format(num=num))
                budget = L - len(pre) - len(needle) - len(query)
                row = hay[rng.integers(0, hay.shape[0])].astype(np.int64)
                h = row[: budget].tolist()
                k = int(d * len(h))
                ids = pre + h[:k] + needle + h[k:] + query
                x = torch.tensor(ids, device=args.device)[None]
                with torch.inference_mode():
                    g = model.generate(x, max_new_tokens=args.max_new,
                                       do_sample=False,
                                       pad_token_id=tok.pad_token_id or 1)
                text = tok.decode(g[0, x.shape[1]:].tolist())
                hits.append(str(num) in text)
                done += 1
                if done % 20 == 0:
                    print(f"[{done}/{total}] {(time.time()-t0)/done:.1f}s/trial",
                          flush=True)
            results[f"L{L}_d{d}"] = {"acc": float(np.mean(hits)),
                                     "hits": int(np.sum(hits)),
                                     "trials": args.trials}

    # 길이별 집계
    by_len = {str(L): float(np.mean([results[f"L{L}_d{d}"]["acc"] for d in depths]))
              for L in lengths}
    out = {"model": args.model, "lengths": lengths, "depths": depths,
           "trials": args.trials, "cells": results, "acc_by_length": by_len}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("acc_by_length:", by_len)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
