#!/usr/bin/env python3
# Copyright (c) 2026 alpha team. Apache-2.0.
"""최종 단일 SFT 블렌드 생성 — Nemotron 3 Ultra 공개 SFT 블렌드의 카테고리 비중을 토큰 기준으로 그대로 적용 (사용자 결정 2026-09-13).

근거: NVIDIA-NeMo/Nemotron `recipes/ultra3/stage1_sft/config/data_prep/data_blend_raw.json` — 공개 12 항목 58.3 + 내부 13 범주 47.5 = 105.8
(비정규화, 런타임 정규화). 정규화 비중: chat/IF/구조화 27.03 · 코드 27.13 · 수학 15.88 · 과학 12.10 · 다국어 6.99 · 에이전틱 3.78 · 금융 2.84 ·
장문맥 1.89 · 저노력 1.89 · safety 0.47. 장문맥은 LC-A/B 에서 이미 학습해 제외, identity 0.5% 는 alpha 추가 → 나머지를 99.5% 로 재정규화.
Ultra 가중치 단위는 샘플(NeMo 관례)이지만 우리는 bin 1표 집계라 토큰 비중 = gradient 비중이므로 토큰 기준으로 적용한다.

스펙(sft_128k_final_spec.tsv): member / category / base_ep / fixed.
  fixed=1 : 에폭 고정(구조화 8ep·usab 4ep·cuda 4ep — 셋이 작아 카테고리 비중을 채우면 수십~수백 회 반복이 되는 멤버).
  fixed=0 : 카테고리 목표 토큰에서 fixed 소비를 뺀 잔여를 base_ep 비율(멤버 간 상대 에폭)을 유지한 채 공통 배율 k 로 채운다: ep = k × base_ep.
예산: --budget-tokens (기본 60B = Ultra 1단계 294,912×64×204,800 ≈ 60B 미러) → iters = ceil(B / (gbs × seq)).
사용: python3 gen_sft_128k_final_blend.py --tree <bins 트리> --spec sft_128k_final_spec.tsv --out <yaml>
"""
import argparse, json, math, os, sys, datetime

CATEGORY_TARGET_ULTRA = {  # Ultra 정규화 비중(%), 장문맥 제외 전
    "chat": 27.03, "code": 27.13, "math": 15.88, "science": 12.10, "multilingual": 6.99,
    "agentic": 3.78, "finance": 2.84, "low_effort": 1.89, "safety": 0.47}
IDENTITY_SHARE = 0.5  # alpha 추가(%)

def load_spec(path):
    rows = []
    for l in open(path):
        l = l.strip()
        if not l or l.startswith("#"): continue
        m, c, ep, fx = l.split("\t")
        rows.append(dict(name=m, cat=c, base=float(ep), fixed=fx.strip() == "1"))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True); ap.add_argument("--spec", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--budget-tokens", type=float, default=60e9); ap.add_argument("--gbs", type=int, default=160); ap.add_argument("--seq-length", type=int, default=131072)
    a = ap.parse_args()
    per_iter = a.gbs * a.seq_length
    iters = math.ceil(a.budget_tokens / per_iter); B = iters * per_iter
    spec = load_spec(a.spec)
    for m in spec:
        p = os.path.join(a.tree, m["name"], "data.stats.json")
        if not os.path.exists(p): sys.exit(f"{m['name']}: {p} 없음")
        s = json.load(open(p)); m["real"] = s["real_tokens"]; m["train"] = s["trainable_tokens"]; m["bins"] = s["n_bins"]
        m["path"] = os.path.join(a.tree, m["name"], "data_text_document")
    # 카테고리 목표(%) 재정규화: Ultra 비중 합(장문맥 제외) → 100 − identity
    ultra_sum = sum(CATEGORY_TARGET_ULTRA.values())
    target = {c: v / ultra_sum * (100 - IDENTITY_SHARE) for c, v in CATEGORY_TARGET_ULTRA.items()}; target["identity"] = IDENTITY_SHARE
    cats = sorted({m["cat"] for m in spec})
    unknown = set(cats) - set(target)
    if unknown: sys.exit(f"목표 없는 카테고리: {unknown}")
    for c in cats:
        mem = [m for m in spec if m["cat"] == c]; tgt = target[c] / 100 * B
        fixed_tok = sum(m["base"] * m["real"] for m in mem if m["fixed"])
        free = [m for m in mem if not m["fixed"]]
        if fixed_tok > tgt: sys.exit(f"{c}: 고정 소비 {fixed_tok/1e9:.2f}B > 목표 {tgt/1e9:.2f}B")
        base_tok = sum(m["base"] * m["real"] for m in free)
        k = (tgt - fixed_tok) / base_tok if free else 0.0
        for m in mem:
            m["ep"] = m["base"] if m["fixed"] else k * m["base"]; m["consume"] = m["ep"] * m["real"]; m["k"] = 1.0 if m["fixed"] else k
    total = sum(m["consume"] for m in spec)
    for m in spec: m["w"] = m["consume"] / total
    assert abs(total - B) / B < 1e-6, (total, B)
    # 출력
    ts = datetime.date.today().isoformat()
    lines = [f"# 최종 단일 SFT 블렌드 — gen_sft_128k_final_blend.py 산출물, 수정 금지 ({ts})",
             f"# 예산 {iters} iters × GBS {a.gbs} × {a.seq_length} = {B/1e9:.2f}B bin-tok = {iters*a.gbs:,} samples (Ultra 1단계 ≈60B 미러)",
             "# 비중 = Nemotron 3 Ultra 공개 SFT 블렌드(data_blend_raw.json, 합 105.8) 카테고리 비중을 토큰 기준으로 적용. 장문맥 1.89% 제외(LC-A/B 기학습), identity 0.5% 추가(alpha), 나머지 99.5% 재정규화.",
             "# 카테고리 안에서는 스펙의 base_ep 상대 비율을 유지한 채 공통 배율 k 로 목표를 채움(fixed=1 멤버는 에폭 고정). 집계 bin 1표라 토큰 비중 = gradient 비중.",
             "#", "# category      target%   tokens(B)   members"]
    for c in cats:
        mem = [m for m in spec if m["cat"] == c]
        lines.append(f"# {c:13s} {target[c]:6.2f}  {sum(m['consume'] for m in mem)/1e9:8.2f}   {len(mem)}")
    lines += ["#", f"# {'member':46s} {'category':12s} {'base_ep':>7s} {'k':>6s} {'ep':>6s} {'consume(B)':>10s} {'w':>9s} {'real(B)':>8s} {'train%':>6s} {'bins':>6s}"]
    for m in sorted(spec, key=lambda x: -x["w"]):
        lines.append(f"# {m['name']:46s} {m['cat']:12s} {m['base']:7.3f} {m['k']:6.3f} {m['ep']:6.3f} {m['consume']/1e9:10.3f} {m['w']:9.6f} {m['real']/1e9:8.3f} {100*m['train']/m['real']:6.1f} {m['bins']:6d}")
    dp = " ".join(f"{m['w']:.6f} {m['path']}" for m in sorted(spec, key=lambda x: -x["w"]))
    lines += [f'data-path: "{dp}"', 'split: "99,1,0"', "dataset: MMAP", "num-workers: 8"]
    with open(a.out, "w") as f: f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:len(cats) + 7]))
    print(f"iters={iters} samples={iters*a.gbs} members={len(spec)} Σw={sum(m['w'] for m in spec):.6f} -> {a.out}")

if __name__ == "__main__":
    main()
