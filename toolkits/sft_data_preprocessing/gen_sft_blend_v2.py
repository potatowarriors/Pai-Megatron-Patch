#!/usr/bin/env python3
# Copyright (c) 2026 alpha team. Apache-2.0.
"""SFT 블렌드 생성기 v2 — bin-토큰 기준, 2단 목표(버킷 → 하위) (2026-09-22).

gen_sft_128k_final_blend.py 를 대체한다. 그 생성기는 consume = ep × real_tokens(패딩 제외)로 가중치를 냈지만 로더(BlendableDataset)는
가중치를 샘플(= bin) 비율로 소비하므로 실제 에폭이 표기 × fill 만큼 낮았다(docs/SFT_FINAL_PLAN.md §1.1). 여기서는
  samples_m = ep_m × bins_m × train_split,   w_m = samples_m / Σ samples,   Σ samples = iters × gbs
로 정의해 표기 ep 가 곧 학습 split 기준 실제 에폭이 되게 한다(audit_blend_epochs.py 와 같은 정의).

스펙 TSV 열: member  bucket  sub  base_ep  fixed
  bucket : 최상위 버킷(agent / reasoning / chat / etc …) — --targets 로 bin 비중(%)을 준다.
  sub    : 버킷 안의 하위(예: agent 의 code / terminal / tool-call / search). --sub-targets 로 버킷 안 비중(%)을 주면 그 비율로,
           주지 않으면 버킷 전체를 base_ep 비율로 채운다. '-' 는 하위 없음.
  fixed  : 1 이면 base_ep 를 에폭으로 고정하고 목표 배분에서 제외(소형 셋의 4~8 에폭 고정용). 버킷 목표에서 fixed 소비를 먼저 뺀다.
  fixed=0 멤버는 같은 (bucket, sub) 안에서 base_ep 상대 비율을 유지한 채 공통 배율 k 로 목표를 채운다: ep = k × base_ep.

사용:
  python3 gen_sft_blend_v2.py --tree <bins 트리> --spec sft_128k_agentic_spec.tsv --out <yaml> --budget-tokens 60e9 \
      --targets agent=58,reasoning=22,chat=15,etc=5 --sub-targets agent:code=45,terminal=25,tool-call=25,search=5
출력 yaml 은 data-path / split / dataset / num-workers 4 키 + 주석 표(버킷·하위·멤버, ep·bin-tok·학습 tok·gradient 비중).
"""
import argparse, datetime, json, math, os, sys
from collections import defaultdict


def load_spec(path):
    rows = []
    for l in open(path):
        l = l.rstrip("\n")
        if not l.strip() or l.startswith("#"): continue
        m, b, s, ep, fx = l.split("\t")
        rows.append(dict(name=m.strip(), bucket=b.strip(), sub=s.strip(), base=float(ep), fixed=fx.strip() == "1"))
    return rows


def parse_targets(s):
    out = {}
    for kv in s.split(","):
        k, v = kv.split("="); out[k.strip()] = float(v)
    return out


def parse_sub_targets(s):
    out = {}
    if not s: return out
    for grp in s.split(";"):
        b, rest = grp.split(":"); out[b.strip()] = parse_targets(rest)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True); ap.add_argument("--spec", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--budget-tokens", type=float, default=60e9); ap.add_argument("--gbs", type=int, default=160)
    ap.add_argument("--seq-length", type=int, default=131072); ap.add_argument("--train-split", type=float, default=0.99)
    ap.add_argument("--targets", required=True, help="버킷 bin 비중 %, 예 agent=58,reasoning=22,chat=15,etc=5 (합 100)")
    ap.add_argument("--sub-targets", default="", help="버킷 안 하위 비중 %, 예 'agent:code=45,terminal=25,tool-call=25,search=5' (;로 버킷 구분)")
    ap.add_argument("--title", default="SFT 블렌드")
    a = ap.parse_args()
    per_iter = a.gbs * a.seq_length
    iters = math.ceil(a.budget_tokens / per_iter); N = iters * a.gbs; B = N * a.seq_length
    target = parse_targets(a.targets); subt = parse_sub_targets(a.sub_targets)
    if abs(sum(target.values()) - 100) > 1e-6: sys.exit(f"--targets 합 {sum(target.values())} ≠ 100")
    for b, st in subt.items():
        if abs(sum(st.values()) - 100) > 1e-6: sys.exit(f"--sub-targets {b} 합 {sum(st.values())} ≠ 100")
    spec = load_spec(a.spec)
    for m in spec:
        p = os.path.join(a.tree, m["name"], "data.stats.json")
        if not os.path.exists(p): sys.exit(f"{m['name']}: {p} 없음")
        s = json.load(open(p)); m["bins"] = s["n_bins"]; m["real"] = s["real_tokens"]; m["train"] = s["trainable_tokens"]
        m["path"] = os.path.join(a.tree, m["name"], "data_text_document")
        if m["bins"] < 100: sys.exit(f"{m['name']}: bins {m['bins']} < 100 (valid 0-doc 무한대기, rules/sft-data.md)")
    buckets = sorted({m["bucket"] for m in spec})
    unknown = set(buckets) - set(target)
    if unknown: sys.exit(f"목표 없는 버킷: {unknown}")
    # 샘플 단위 배분. samples(ep) = ep × bins × train_split
    sm = lambda m, ep: ep * m["bins"] * a.train_split
    for b in buckets:
        mem = [m for m in spec if m["bucket"] == b]; tgt = target[b] / 100 * N
        fixed_s = sum(sm(m, m["base"]) for m in mem if m["fixed"])
        if fixed_s > tgt: sys.exit(f"{b}: 고정 소비 {fixed_s:.0f} samples > 목표 {tgt:.0f}")
        free = [m for m in mem if not m["fixed"]]
        groups = defaultdict(list)
        for m in free: groups[m["sub"]].append(m)
        if b in subt:
            missing = set(subt[b]) - set(groups); extra = set(groups) - set(subt[b])
            if missing or extra: sys.exit(f"{b}: sub 불일치 — 목표에만 {missing}, 스펙에만 {extra}")
            share = {s: subt[b][s] / 100 for s in groups}
        else:
            base_all = sum(sm(m, m["base"]) for m in free)
            share = {s: sum(sm(m, m["base"]) for m in g) / base_all for s, g in groups.items()}
        for s, g in groups.items():
            tgt_s = (tgt - fixed_s) * share[s]; base_s = sum(sm(m, m["base"]) for m in g)
            k = tgt_s / base_s
            for m in g: m["ep"] = k * m["base"]; m["k"] = k
        for m in mem:
            if m["fixed"]: m["ep"] = m["base"]; m["k"] = 1.0
            m["samples"] = sm(m, m["ep"]); m["bintok"] = m["samples"] * a.seq_length
            m["fill"] = m["real"] / (m["bins"] * a.seq_length); m["trainpct"] = m["train"] / m["real"]
            m["traintok"] = m["bintok"] * m["fill"] * m["trainpct"]
    tot_s = sum(m["samples"] for m in spec)
    assert abs(tot_s - N) / N < 1e-6, (tot_s, N)
    for m in spec: m["w"] = m["samples"] / tot_s
    T = sum(m["traintok"] for m in spec)
    # 출력
    ts = datetime.date.today().isoformat()
    L = [f"# {a.title} — gen_sft_blend_v2.py 산출물, 수정 금지 ({ts})",
         f"# 예산 {iters} iters × GBS {a.gbs} × {a.seq_length} = {B/1e9:.2f}B bin-tok = {N:,} samples · 학습 tok {T/1e9:.2f}B",
         f"# 가중치 = 샘플(bin) 비율: samples = ep × bins × {a.train_split}. 표기 ep = 학습 split 기준 실제 에폭 (audit_blend_epochs.py 와 동일 정의).",
         f"# --targets {a.targets}" + (f"  --sub-targets '{a.sub_targets}'" if a.sub_targets else ""),
         "#", f"# {'bucket/sub':22s} {'bin%':>6s} {'bin-tok(B)':>10s} {'train(B)':>9s} {'grad%':>6s} {'members':>7s} {'ep min~max':>14s}"]
    def row(label, mem):
        bt = sum(m["bintok"] for m in mem); tt = sum(m["traintok"] for m in mem)
        L.append(f"# {label:22s} {100*bt/B:6.2f} {bt/1e9:10.2f} {tt/1e9:9.2f} {100*tt/T:6.2f} {len(mem):7d} {min(m['ep'] for m in mem):6.3f}~{max(m['ep'] for m in mem):6.3f}")
    for b in sorted(buckets, key=lambda b: -target[b]):
        mem = [m for m in spec if m["bucket"] == b]; row(b, mem)
        subs = sorted({m["sub"] for m in mem if m["sub"] != "-"})
        for s in subs: row(f"  {b}/{s}", [m for m in mem if m["sub"] == s])
        fx = [m for m in mem if m["fixed"]]
        if fx: row(f"  {b}/(fixed)", fx)
    L += ["#", f"# {'member':46s} {'bucket':10s} {'sub':10s} {'base_ep':>7s} {'k':>6s} {'ep':>6s} {'bin-tok(B)':>10s} {'w':>9s} {'fill%':>5s} {'train%':>6s} {'bins':>6s}"]
    for m in sorted(spec, key=lambda x: -x["w"]):
        L.append(f"# {m['name']:46s} {m['bucket']:10s} {m['sub']:10s} {m['base']:7.3f} {m['k']:6.3f} {m['ep']:6.3f} {m['bintok']/1e9:10.3f} {m['w']:9.6f} {100*m['fill']:5.1f} {100*m['trainpct']:6.1f} {m['bins']:6d}")
    dp = " ".join(f"{m['w']:.6f} {m['path']}" for m in sorted(spec, key=lambda x: -x["w"]))
    L += [f'data-path: "{dp}"', 'split: "99,1,0"', "dataset: MMAP", "num-workers: 8"]
    with open(a.out, "w") as f: f.write("\n".join(L) + "\n")
    print("\n".join(l for l in L if l.startswith("#") and not l.startswith("# ml_") and "translated" not in l)[:6000])
    print(f"iters={iters} samples={N} members={len(spec)} Σw={sum(m['w'] for m in spec):.6f} -> {a.out}")


if __name__ == "__main__":
    main()
