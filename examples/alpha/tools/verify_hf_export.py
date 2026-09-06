"""HF 변환 산출물이 쓸 수 있는 상태인가 — 변환기 종료 코드와 **독립적으로** 판정한다.

왜 필요한가 (2026-09-05/06):
    sub1 은 compat libcuda 570→595 스왑 이후 학습·변환에서 SIGSEGV 가 난다
    (`examples/alpha/CLAUDE.md` 함정 표 09-04). 그런데 두 번 모두 **작업은 끝나고
    teardown 에서** 죽었다 — 8개 랭크 전부가 `destroy_process_group() was not called
    before program exit` 경고를 냈다(그 경고는 프로그램이 끝까지 도달해야 나온다).
    iter1200 은 `.partial` 로, iter1500 은 종료코드 -11 로 남았지만 산출물은 둘 다
    정상이었고 G1·G2·G3 게이트를 통과해 벤치가 정상적으로 돌았다.

    그래서 "종료 코드가 실패면 무조건 폐기"도, "종료 코드를 무시하고 진행"도 틀렸다.
    **산출물 자체를 재는** 절차가 필요하다. 이 스크립트가 그 절차다.

검사:
    1. 필수 파일 존재 (config / generation_config / index / tokenizer)
    2. index 가 가리키는 샤드가 모두 존재
    3. 파일 크기 합 ≥ index total_size, 초과분(헤더+패딩)이 총량의 0.1% 미만
    4. 샤드에서 실제로 읽히는 텐서 수 == weight_map 항목 수
    5. NaN / Inf 스캔 (기본 표본, --full 이면 전량)
    6. generation_config.eos_token_id 가 비어 있지 않음
    7. --reference 를 주면 텐서 이름 집합이 정확히 같은지 대조

사용:
    python3 verify_hf_export.py <hfdir> [--reference <정상본>] [--full] [--sample N]
    exit 0 = 통과, 1 = 실패
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

REQUIRED = ("config.json", "generation_config.json", "model.safetensors.index.json",
            "tokenizer.json", "tokenizer_config.json")


def fail(msg: str) -> None:
    print(f"   ❌ {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hfdir")
    ap.add_argument("--reference", help="정상 변환본 — 텐서 이름 집합 대조")
    ap.add_argument("--full", action="store_true", help="모든 텐서를 NaN/Inf 스캔 (느림)")
    ap.add_argument("--sample", type=int, default=200, help="스캔할 텐서 수 (기본 200)")
    a = ap.parse_args()
    d = a.hfdir
    bad = []

    print(f"── HF 변환본 검증: {d}")

    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(d, f))]
    if missing:
        fail(f"필수 파일 누락: {missing}")
        bad.append("files")
    else:
        print(f"   필수 파일 {len(REQUIRED)}종 존재")

    ipath = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(ipath):
        fail("index 가 없어 이후 검사를 할 수 없다")
        return 1
    idx = json.load(open(ipath))
    wmap = idx["weight_map"]
    shards = sorted(set(wmap.values()))

    gone = [s for s in shards if not os.path.exists(os.path.join(d, s))]
    if gone:
        fail(f"샤드 누락 {len(gone)}개: {gone[:3]}")
        bad.append("shards")
    else:
        print(f"   샤드 {len(shards)}개 전부 존재")

    if not gone:
        total = sum(os.path.getsize(os.path.join(d, s)) for s in shards)
        declared = idx["metadata"]["total_size"]
        over = total - declared
        if over < 0:
            fail(f"파일 합({total:,}) < index total_size({declared:,}) — 잘렸다")
            bad.append("size")
        elif over > declared * 0.001:
            fail(f"초과분 {over:,} 바이트가 총량의 0.1% 초과 — 구조 이상")
            bad.append("size")
        else:
            print(f"   크기 {total/2**30:.2f} GiB, 헤더+패딩 {over:,} 바이트 (정상 범위)")

    from safetensors import safe_open  # noqa: PLC0415

    names: list[str] = []
    if not gone:
        for s in shards:
            with safe_open(os.path.join(d, s), framework="pt") as f:
                names.extend(f.keys())
        if len(names) != len(wmap):
            fail(f"텐서 수 불일치: 샤드 {len(names)} vs weight_map {len(wmap)}")
            bad.append("count")
        else:
            print(f"   텐서 {len(names):,} == weight_map {len(wmap):,}")

    if a.reference and names:
        ref = json.load(open(os.path.join(a.reference, "model.safetensors.index.json")))
        diff = set(wmap) ^ set(ref["weight_map"])
        if diff:
            fail(f"정상본과 텐서 이름 {len(diff)}개 차이: {sorted(diff)[:3]}")
            bad.append("names")
        else:
            print(f"   텐서 이름 집합이 정상본과 동일 ({os.path.basename(a.reference)})")

    if names:
        import torch  # noqa: PLC0415
        key = [n for n in names if "embed_tokens" in n or "lm_head" in n]
        rest = [n for n in names if n not in key]
        pick = names if a.full else key + random.Random(0).sample(rest, min(a.sample, len(rest)))
        nan = inf = 0
        by_shard: dict[str, list[str]] = {}
        for n in pick:
            by_shard.setdefault(wmap[n], []).append(n)
        for s, ns in by_shard.items():
            with safe_open(os.path.join(d, s), framework="pt") as f:
                for n in ns:
                    t = f.get_tensor(n).float()
                    nan += int(torch.isnan(t).any())
                    inf += int(torch.isinf(t).any())
        if nan or inf:
            fail(f"NaN {nan}개 / Inf {inf}개 텐서 — 가중치 훼손")
            bad.append("nan")
        else:
            print(f"   NaN/Inf 0 ({len(pick):,}개 텐서 스캔{' · 전량' if a.full else ' · 표본'})")

    gc = os.path.join(d, "generation_config.json")
    if os.path.exists(gc):
        eos = json.load(open(gc)).get("eos_token_id")
        if not eos:
            fail("generation_config.eos_token_id 가 비어 있다 (G1 실패)")
            bad.append("eos")
        else:
            print(f"   eos_token_id = {eos}")

    if bad:
        print(f"❌ 검증 실패: {', '.join(bad)} — 이 산출물로 벤치를 돌리지 말 것")
        return 1
    print("✅ 변환본 검증 통과 — 변환기 종료 코드와 무관하게 사용 가능")
    return 0


if __name__ == "__main__":
    sys.exit(main())
