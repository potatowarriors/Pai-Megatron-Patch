#!/usr/bin/env python3
# Copyright (c) 2026 alpha team. Apache-2.0.
"""SFT phase-2(연속 학습) 블렌드 생성 — phase-1 블렌드 리플레이 + 수정분 멤버.

설계(docs/SFT_PHASE2_PLAN.md §1): phase-2 예산 B2 = iters × gbs × seq 안에서
  - 고정 멤버: --ep NAME=E (phase-2 에폭 지정, consume = E × real128)
              --share NAME=S (예산 비중 지정, consume = S × B2)
    → opencode_fixed 부스트(총 0.25ep), identity_v2 0.6%, chat_restored 1.9ep, safety 상한 등
  - 리플레이 멤버(나머지): phase-1 상대 가중치 그대로 잔여 예산을 배분
  - --map OLD=NEW: phase-1 멤버 OLD 를 NEW 디렉터리로 교체(opencode_v1 → opencode_fixed)
  - --drop NAME: phase-1 멤버 제외(identity_v1)
  - --add NAME: phase-1 에 없던 멤버 추가(identity_v2, chat_v3_chat_restored) — 반드시 --ep/--share 와 함께
집계는 bin 1표(calculate_per_token_loss=False)라 토큰 비중 = gradient 비중 (KNOWN_ISSUES 2026-09-01 ②).
NEW 디렉터리에 data.stats.json 이 없으면 OLD 의 stats 로 대신 계산하고 DRAFT 표시 (G-P1 후 재실행).

카테고리별 리플레이 (2026-09-04, docs/SFT_PHASE2_PLAN.md §11 절충안 M):
  - --scale NAME=f : phase-1 멤버의 phase-2 비중을 phase-1 비중의 f 배로 고정 (consume = f × w1 × B2).
        형제 셋이 있는 카테고리 0.15, 대체재 없는 대형(cp·math) 0.5, 대체재 없는 소형(한국어·IF·ml·identity) 1.0.
  - --solve-iters  : --iters 대신 예산을 푼다. B2 = Σ(ep 고정 소비) / (1 − Σ share − Σ f·w1), iters = ceil(B2/(gbs·seq)).
        올림으로 생긴 잔여는 scale·replay 멤버에 w1 비례로 배분 (ep 고정 멤버의 epoch 는 불변).
  - phase-1 stats 는 --phase1-tree 에 없으면 yaml 경로의 자기 디렉터리에서 읽는다 (교체 재개 yaml 의 opencode_fixed·identity_v2).
"""
import argparse, json, math, os, re, sys
from datetime import date

SEQ_DEFAULT = 131072


def parse_yaml_blend(path):
    s = open(path, encoding="utf-8").read()
    toks = re.search(r'data-path:\s*"([^"]+)"', s).group(1).split()
    return [(toks[i + 1], float(toks[i])) for i in range(0, len(toks), 2)]


def load_stats(tree, name):
    p = os.path.join(tree, name, "data.stats.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def kv_list(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-yaml", required=True)
    ap.add_argument("--phase1-tree", required=True, help="phase-1 bins 트리 (ep_p1 계산용)")
    ap.add_argument("--tree", required=True, help="phase-2 bins 트리 (미변경 셋은 symlink)")
    ap.add_argument("--phase1-iters", type=int, default=2448)
    ap.add_argument("--phase1-consumed-iters", type=int, default=None,
                    help="phase-1 을 중간(iter N)에서 끊고 교체할 때 — ep_p1 을 N/phase1_iters 로 환산해 cum 을 맞춘다")
    ap.add_argument("--iters", type=int, default=550)
    ap.add_argument("--gbs", type=int, default=160)
    ap.add_argument("--seq-length", type=int, default=SEQ_DEFAULT)
    ap.add_argument("--map", nargs="*", default=[], help="OLD=NEW")
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--add", nargs="*", default=[])
    ap.add_argument("--ep", nargs="*", default=[], help="NAME=E  phase-2 에폭 고정")
    ap.add_argument("--share", nargs="*", default=[], help="NAME=S  phase-2 예산 비중 고정")
    ap.add_argument("--scale", nargs="*", default=[], help="NAME=f  phase-1 비중의 f 배로 고정 (consume = f·w1·B2)")
    ap.add_argument("--solve-iters", action="store_true",
                    help="--iters 대신 예산을 푼다: B2 = Σ(ep 고정 소비)/(1 − Σshare − Σf·w1), iters 올림")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    B1 = a.phase1_iters * a.gbs * a.seq_length
    consumed_frac = (a.phase1_consumed_iters / a.phase1_iters) if a.phase1_consumed_iters else 1.0
    per_iter = a.gbs * a.seq_length
    B2 = a.iters * per_iter
    mapping = kv_list(a.map)
    ep_fix = {k: float(v) for k, v in kv_list(a.ep).items()}
    share_fix = {k: float(v) for k, v in kv_list(a.share).items()}
    scale_fix = {k: float(v) for k, v in kv_list(a.scale).items()}

    members = []  # dict(name, path, w1, real1, real2, ep1, kind)
    draft = False
    for path, w1 in parse_yaml_blend(a.phase1_yaml):
        old = path.split("/")[-2]
        if old in a.drop:
            continue
        new = mapping.get(old, old)
        s1 = load_stats(a.phase1_tree, old)
        if s1 is None:  # 교체 재개 yaml 의 멤버(opencode_fixed·identity_v2)는 자기 디렉터리에서
            own = os.path.dirname(path)
            s1 = json.load(open(os.path.join(own, "data.stats.json"))) if os.path.exists(os.path.join(own, "data.stats.json")) else None
            if s1 is None:
                sys.exit(f"{old}: phase-1 stats 없음 ({a.phase1_tree} 및 {own})")
        s2 = load_stats(a.tree, new)
        if s2 is None:
            print(f"[warn] {new}: stats 없음 → {old} 의 phase-1 stats 로 대체 (DRAFT)")
            s2, draft = s1, True
        ep1 = w1 * B1 / s1["real_tokens"] * consumed_frac
        members.append(dict(name=new, old=old, w1=w1, real2=s2["real_tokens"], ep1=ep1,
                            path=os.path.join(a.tree, new, "data_text_document"),
                            bins=s2["n_bins"]))
    for name in a.add:
        s2 = load_stats(a.tree, name)
        if s2 is None:
            if name not in ep_fix and name not in share_fix:
                sys.exit(f"--add {name}: --ep 또는 --share 필요")
            print(f"[warn] {name}: stats 없음 → 예산 비중만으로 배치 (DRAFT, real 미상)")
            s2, draft = {"real_tokens": None, "n_bins": None}, True
        members.append(dict(name=name, old=None, w1=0.0, real2=s2["real_tokens"], ep1=0.0,
                            path=os.path.join(a.tree, name, "data_text_document"), bins=s2["n_bins"]))

    # 멤버 종류 판정: ep 고정(절대 토큰) / share·scale 고정(예산 비례) / replay(잔여)
    ep_abs = 0.0; rel_share = 0.0
    for m in members:
        n = m["name"]
        if n in share_fix:
            m["kind"] = f"share={share_fix[n]}"; rel_share += share_fix[n]
        elif n in ep_fix:
            if m["real2"] is None:
                sys.exit(f"{n}: real_tokens 미상 — --share 로 지정하거나 stats 를 먼저 만들 것")
            m["kind"] = f"ep={ep_fix[n]}"; ep_abs += ep_fix[n] * m["real2"]
        elif n in scale_fix:
            m["kind"] = f"scale={scale_fix[n]}"; rel_share += scale_fix[n] * m["w1"]
        else:
            m["kind"] = "replay"
    unknown = (set(ep_fix) | set(share_fix) | set(scale_fix)) - {m["name"] for m in members}
    if unknown:
        sys.exit(f"블렌드에 없는 멤버 지정: {sorted(unknown)}")
    if a.solve_iters:
        denom = 1.0 - rel_share
        if denom <= 0 or ep_abs <= 0:
            sys.exit(f"--solve-iters: 1−Σ비중 = {denom:.4f}, ep 고정 소비 {ep_abs/1e9:.2f}B — 풀 수 없음")
        a.iters = math.ceil(ep_abs / denom / per_iter)
        B2 = a.iters * per_iter
    # 소비량
    fixed = 0.0
    for m in members:
        n = m["name"]
        if n in share_fix:
            m["consume"] = share_fix[n] * B2
        elif n in ep_fix:
            m["consume"] = ep_fix[n] * m["real2"]
        elif n in scale_fix:
            m["consume"] = scale_fix[n] * m["w1"] * B2
        else:
            m["consume"] = None
        if m["consume"] is not None:
            fixed += m["consume"]
    if fixed > B2 + 1e-6:
        sys.exit(f"고정 멤버 소비 {fixed/1e9:.2f}B > 예산 {B2/1e9:.2f}B")
    pool = [m for m in members if m["consume"] is None]
    R = B2 - fixed
    if pool:
        wsum = sum(m["w1"] for m in pool)
        for m in pool:
            m["consume"] = R * m["w1"] / wsum
    elif R > 0:  # --solve-iters 올림 잔여 → scale 멤버에 w1 비례 가산 (ep 고정 멤버 불변)
        sc = [m for m in members if m["name"] in scale_fix]
        wsum = sum(m["w1"] for m in sc)
        for m in sc:
            m["consume"] += R * m["w1"] / wsum
    for m in members:
        m["w2"] = m["consume"] / B2
        m["ep2"] = (m["consume"] / m["real2"]) if m["real2"] else float("nan")
        m["cum"] = m["ep1"] + m["ep2"]

    members.sort(key=lambda m: -m["w2"])
    hdr = [f"# SFT phase-2 연속학습 블렌드 — gen_phase2_blend.py 산출물{' (DRAFT: 일부 stats 대체)' if draft else ''}, 수정 금지",
           f"# {date.today().isoformat()} | 예산 {a.iters} iters × GBS {a.gbs} × {a.seq_length} = {B2/1e9:.2f}B bin-tok = {a.iters*a.gbs:,} samples",
           f"# 기준 {a.phase1_yaml} (phase-1 {a.phase1_iters} iters = {B1/1e9:.2f}B" + (f", 실소비 {a.phase1_consumed_iters} iters 후 교체 → ep_p1 은 {consumed_frac:.3f} 배" if a.phase1_consumed_iters else "") + f"). 리플레이 {R/B2*100:.1f}% / 고정 {fixed/B2*100:.1f}%",
           "# 설계: docs/SFT_PHASE2_PLAN.md §1 — 집계 bin 1표라 토큰 비중 = gradient 비중"]
    if scale_fix or a.solve_iters:
        hdr.append("# 카테고리별 리플레이(§11 절충안 M): scale=f 는 phase-1 비중의 f 배, ep=E 는 절대 소비(E×real), "
                   + ("iters 는 --solve-iters 로 풂(올림 잔여는 scale 멤버 w1 비례)" if a.solve_iters else f"iters 고정 {a.iters}"))
    hdr += ["#",
            f"# {'member':<36}{'kind':<14}{'w2':>10}{'consume(B)':>11}{'ep_p1':>7}{'ep_p2':>7}{'cum':>7}"]
    for m in members:
        hdr.append(f"# {m['name']:<36}{m['kind']:<14}{m['w2']:>10.6f}{m['consume']/1e9:>11.3f}"
                   f"{m['ep1']:>7.2f}{m['ep2']:>7.2f}{m['cum']:>7.2f}")
    dp = " ".join(f"{m['w2']:.6f} {m['path']}" for m in members)
    body = "\n".join(hdr) + f'\ndata-path: "{dp}"\nsplit: "99,1,0"\ndataset: MMAP\nnum-workers: 8\n'
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(body)
    print("\n".join(hdr[5:])); print(f"\n합 {sum(m['w2'] for m in members):.6f} → {a.out}")


if __name__ == "__main__":
    main()
