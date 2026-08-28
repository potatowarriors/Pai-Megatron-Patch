#!/usr/bin/env python3
"""update_blend_kochat.py — sft_40b_blend.yaml 의 chat 슬롯을 KoChat 세트 포함 동일-epoch 로 재비례.

규칙 (사용자 확정 2026-08-26): 한국어 chat 과 영어 chat_v3 가 **동일 epoch** — chat 슬롯
가중치 합(W)은 불변, 슬롯 내 각 세트 가중치 = W × real_tokens_i / Σ real_tokens.
기존 kochat_* 엔트리는 제거하고 --ko-sets 로 준 세트로 교체 (트랜치 교체).

사용:
  python3 update_blend_kochat.py --ko-sets kochat_if_fanout_me_t2 kochat_chat_t2 kochat_b_t2 \
      [--yaml <path>] [--write]      # --write 없으면 미리보기만
"""
import argparse
import json
import re
from pathlib import Path

D = Path("/home/work/Datasets/LL_preprocessed/v5/sft_packed_64k_pad16")
Y = Path("/home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/configs/data/sft_40b_blend.yaml")
EN_CHAT = ["chat_v3_if_fanout_me", "chat_v3_chat"]
TOTAL_BIN_TOK = 610368 * 65536


def real(name):
    return json.load(open(D / name / "data.stats.json"))["real_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ko-sets", nargs="+", required=True)
    ap.add_argument("--yaml", default=str(Y))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    txt = open(args.yaml).read()
    m = re.search(r'^data-path: "(.*)"$', txt, re.M)
    toks = m.group(1).split()
    pairs = [(float(toks[i]), toks[i + 1]) for i in range(0, len(toks), 2)]

    def setname(p):
        return p.split("/")[-2]

    chat_members = set(EN_CHAT) | {n for n in (setname(p) for _, p in pairs) if n.startswith("kochat")}
    W = sum(w for w, p in pairs if setname(p) in chat_members)
    reals = {s: real(s) for s in EN_CHAT + args.ko_sets}
    tot = sum(reals.values())
    ep = W * TOTAL_BIN_TOK / tot
    ko_real = sum(reals[s] for s in args.ko_sets)

    new = [(w, p) for w, p in pairs if setname(p) not in chat_members]
    for s in EN_CHAT + args.ko_sets:
        new.append((W * reals[s] / tot, str(D / s / "data_text_document")))
    # 원래 위치(swe 다음) 유지: swe 엔트리 뒤에 chat 블록 삽입
    swe = [e for e in new if "swe_v3" in e[1]]
    rest = [e for e in new if "swe_v3" not in e[1] and setname(e[1]) not in (EN_CHAT + args.ko_sets)]
    chat = [e for e in new if setname(e[1]) in (EN_CHAT + args.ko_sets)]
    ordered = swe + chat + rest

    print(f"chat slot W={W:.6f} ({W * TOTAL_BIN_TOK / 1e9:.2f}B learned)")
    print("real tokens (B):", {k: round(v / 1e9, 4) for k, v in reals.items()}, "sum", round(tot / 1e9, 3))
    print(f"common epoch={ep:.3f} | korean learned={ko_real * ep / 1e9:.3f}B = {ko_real / tot:.1%} of chat slot")
    print("chat weights:", {setname(p): round(w, 6) for w, p in chat})
    print(f"sum all={sum(w for w, _ in ordered):.6f}")
    if not args.write:
        return
    line = 'data-path: "' + " ".join(f"{w:.6f} {p}" for w, p in ordered) + '"'
    new_txt = txt[: m.start()] + line + txt[m.end():]
    open(args.yaml, "w").write(new_txt)
    print(f"wrote {args.yaml}")


if __name__ == "__main__":
    main()
