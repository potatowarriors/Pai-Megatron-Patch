#!/usr/bin/env python3
# Copyright (c) 2026 alpha team. Apache-2.0.
"""64k 본 블렌드 → 128k 혼합 블렌드 파생 (epoch 압력 보존).

단일 128k 런 설계용. 기준 yaml(sft_40b_blend.yaml)의 멤버별 가중치가 담고 있는
설계 결정(SWE 1-pass 앵커, 카테고리 비율, KoChat 동일 epoch, budget 1% 등)을
**멤버별 epoch 수**로 환원한 뒤, 같은 epoch 수를 128k 혼합 트리(<=128k 전량)의
real-token 에 적용해 소비량을 다시 계산한다:

    epochs_m = B64 * w64_m / real64_m
    consume128_m = epochs_m * real128_m
    w128_m = consume128_m / sum(consume128)
    B128 = sum(consume128)  (bin-tokens; fill 은 두 트리 모두 ~99% 라 근사 등가)

→ 64k 에서 too_long 드롭된 >64k 꼬리가 각 멤버의 epoch 압력 그대로 편입되고,
  SWE 는 64k분+꼬리 합쳐 정확히 1 epoch 이 된다.

Usage:
  python gen_mixed_blend.py --base-yaml configs/data/sft_40b_blend.yaml \
      --tree-64k .../sft_packed_64k_pad16 --tree-128k .../sft_packed_128k_mixed_pad16 \
      --budget-64k 40e9 --gbs 96 --out configs/data/sft_128k_mixed_blend.yaml
"""
import argparse
import json
import os
import re


def load_stats(tree, name):
    with open(os.path.join(tree, name, "data.stats.json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-yaml", required=True)
    ap.add_argument("--tree-64k", required=True)
    ap.add_argument("--tree-128k", required=True)
    ap.add_argument("--budget-64k", type=float, default=40e9,
                    help="기준 yaml 이 전제한 64k 런 bin-token 예산")
    ap.add_argument("--seq-length", type=int, default=131072)
    ap.add_argument("--gbs", type=int, default=96)
    ap.add_argument("--fixed-consume", nargs="*", default=["identity_v1"],
                    help="128k 트리의 real_tokens 가 입력 복제(×k) 차이로 부풀어 "
                         "epoch 환산이 무효인 멤버 — 64k 소비량을 그대로 유지")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    text = open(args.base_yaml).read()
    m = re.search(r'data-path:\s*"([^"]+)"', text)
    toks = m.group(1).split()
    pairs = [(float(toks[i]), toks[i + 1]) for i in range(0, len(toks), 2)]

    rows = []
    for w64, path in pairs:
        name = path.split("/")[-2]
        r64 = load_stats(args.tree_64k, name)["real_tokens"]
        s128 = load_stats(args.tree_128k, name)
        r128 = s128["real_tokens"]
        epochs = args.budget_64k * w64 / r64
        consume = epochs * r128
        if name in args.fixed_consume:
            consume = args.budget_64k * w64   # 동일 데이터 — 소비량 고정
            epochs = float("nan")
        rows.append(dict(name=name, w64=w64, r64=r64, r128=r128,
                         epochs=epochs, consume=consume,
                         bins128=s128["n_bins"], fill128=s128["fill_rate"]))

    B128 = sum(r["consume"] for r in rows)
    for r in rows:
        r["w128"] = r["consume"] / B128
    samples = int(round(B128 / args.seq_length / args.gbs)) * args.gbs
    iters = samples // args.gbs

    dp = " ".join(
        f'{r["w128"]:.6f} {args.tree_128k}/{r["name"]}/data_text_document' for r in rows)
    hdr = f"""# SFT 128k 혼합 블렌드 — 단일 128k 런용 (gen_mixed_blend.py 산출물, 수정 금지)
#
# 기준: {os.path.basename(args.base_yaml)} 의 멤버별 epoch 압력을 보존한 채 풀을
#   <=128k 전량으로 확장 (64k 에서 too_long 드롭된 꼬리가 같은 epoch 로 편입).
# 예산: {B128/1e9:.2f}B bin-tokens = {samples:,} samples = {iters:,} iters @ GBS {args.gbs}
#   (64k 기준 {args.budget_64k/1e9:.0f}B 대비 +{(B128/args.budget_64k-1)*100:.1f}% — 꼬리 편입분)
# 학습 preset: seq 131072 · CP8+offload · GBS 96 (12.58M tok/iter 상수 관례)
data-path: "{dp}"
split: "99,1,0"
dataset: MMAP
num-workers: 8
"""
    with open(args.out, "w") as f:
        f.write(hdr)

    print(f"{'member':42s} {'real64':>8s} {'real128':>8s} {'ep':>5s} {'w64':>7s} {'w128':>7s} {'bins':>7s} {'fill':>5s}")
    for r in sorted(rows, key=lambda r: -r["w128"]):
        print(f"{r['name']:42s} {r['r64']/1e9:>7.2f}B {r['r128']/1e9:>7.2f}B "
              f"{r['epochs']:>5.2f} {r['w64']*100:>6.2f}% {r['w128']*100:>6.2f}% "
              f"{r['bins128']:>7,d} {r['fill128']*100:>4.0f}%")
    print(f"\nB128 = {B128/1e9:.2f}B bin-tokens → train-samples {samples:,} = {iters:,} iters @ GBS {args.gbs}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
