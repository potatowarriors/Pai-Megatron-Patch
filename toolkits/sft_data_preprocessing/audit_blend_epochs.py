#!/usr/bin/env python3
"""블렌드 YAML 의 멤버별 실제 에폭을 재계산한다 (생성기 주석과 독립, GPU 불필요).

로더(BlendableDataset)는 가중치를 샘플(= bin) 비율로 해석한다. 따라서
  samples = w/Σw × train_samples,  epoch = samples / (bins × train_split)
bins 는 로더가 실제로 읽는 .idx 헤더에서 읽고 data.stats.json 의 n_bins 와 대조한다(불일치 시 exit 1).
생성기(gen_sft_128k_final_blend.py)의 ep 는 실토큰 기준이라 실제 에폭 = 생성기 ep × bin 충전율이다.

Usage:
  python audit_blend_epochs.py --blend ../../examples/alpha/configs/data/sft_128k_final_blend.yaml \
      --train-samples 457920 --spec sft_128k_final_spec.tsv [--md]
"""
import argparse, json, os, re, struct, sys
from collections import defaultdict


def idx_bins(prefix):
    # MMIDIDX 헤더: magic 9 · version <Q · dtype <B · len <Q · doc_count <Q
    with open(prefix + ".idx", "rb") as f:
        if f.read(9) != b"MMIDIDX\x00\x00": sys.exit(f"{prefix}.idx: MMIDIDX 아님")
        f.read(9); return struct.unpack("<Q", f.read(8))[0]


def load_cats(path):
    cats = {}
    for line in open(path):
        if line.strip() and not line.startswith("#"):
            p = line.split(); cats[p[0]] = p[1]
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True); ap.add_argument("--train-samples", type=int, required=True)
    ap.add_argument("--seq-length", type=int, default=131072); ap.add_argument("--train-split", type=float, default=0.99)
    ap.add_argument("--spec", help="member→category TSV (없으면 카테고리 '-')"); ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    dp = re.search(r'data-path:\s*"([^"]+)"', open(a.blend).read()).group(1).split()
    pairs = [(float(dp[i]), dp[i + 1]) for i in range(0, len(dp), 2)]
    cats = load_cats(a.spec) if a.spec else {}
    wsum = sum(w for w, _ in pairs); rows = []; bad = []
    for w, prefix in pairs:
        name = os.path.basename(os.path.dirname(prefix)); bins = idx_bins(prefix)
        s = json.load(open(os.path.join(os.path.dirname(prefix), "data.stats.json")))
        if s["n_bins"] != bins: bad.append(f"{name}: idx {bins} != stats {s['n_bins']}")
        samples = w / wsum * a.train_samples
        rows.append(dict(name=name, cat=cats.get(name, "-"), bins=bins, samples=samples, tok=samples * a.seq_length,
                         ep=samples / (bins * a.train_split), fill=s["real_tokens"] / (bins * a.seq_length),
                         train=s["trainable_tokens"] / s["real_tokens"]))
    if bad: sys.exit("bins 불일치:\n" + "\n".join(bad))
    total = sum(r["tok"] for r in rows); ttrain = sum(r["tok"] * r["fill"] * r["train"] for r in rows)
    by = defaultdict(list)
    for r in rows: by[r["cat"]].append(r)
    order = sorted(by, key=lambda c: -sum(r["tok"] for r in by[c]))
    sep = " | " if a.md else "  "
    def emit(cells): print(("| " + " | ".join(cells) + " |") if a.md else sep.join(cells))
    print(f"members {len(rows)} · weight sum {wsum:.6f} · bins idx↔stats {len(rows)}/{len(rows)} 일치 · "
          f"total {total/1e9:.2f}B bin-tok · trainable {ttrain/1e9:.2f}B\n")
    emit(["category", "members", "bin-tok(B)", "share%", "trainable(B)", "trainable share%", "epoch min~max"])
    if a.md: emit(["---"] * 7)
    for c in order:
        rs = by[c]; t = sum(r["tok"] for r in rs); tt = sum(r["tok"] * r["fill"] * r["train"] for r in rs)
        emit([c, str(len(rs)), f"{t/1e9:.2f}", f"{100*t/total:.2f}", f"{tt/1e9:.2f}", f"{100*tt/ttrain:.2f}",
              f"{min(r['ep'] for r in rs):.3f} ~ {max(r['ep'] for r in rs):.3f}"])
    print()
    emit(["member", "category", "bins", "samples", "bin-tok(B)", "share%", "epoch", "fill%", "train%", "unseen(B)"])
    if a.md: emit(["---"] * 10)
    for c in order:
        for r in sorted(by[c], key=lambda r: -r["tok"]):
            unseen = max(0.0, 1 - r["ep"]) * r["bins"] * a.train_split * a.seq_length
            emit([f"`{r['name']}`" if a.md else r["name"], r["cat"], str(r["bins"]), f"{r['samples']:.0f}", f"{r['tok']/1e9:.3f}",
                  f"{100*r['tok']/total:.3f}", f"{r['ep']:.3f}", f"{100*r['fill']:.1f}", f"{100*r['train']:.1f}", f"{unseen/1e9:.2f}"])


if __name__ == "__main__":
    main()
