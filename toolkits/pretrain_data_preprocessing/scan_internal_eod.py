#!/usr/bin/env python3
"""packed pad16 데이터의 '문서 내부 잡탕 EOD' 스캔.

배경 (2026-08-23, LC-A iter 170 크래시): 합성 원문에 리터럴 `<|endoftext|>` 문자열이
남아 있으면 HF tokenizers가 added special token을 본문에서도 그대로 매칭해 id 0이
문서 중간에 박힌다. `--reset-position-ids`는 EOD마다 position을 리셋하므로 그 문서가
격자(%pad-doc-multiple) 비정렬 위치에서 분열되고, THD+CP a2a의 %2cp 가드가 학습을
정지시킨다. 런타임은 `snap_cu_seqlens_to_grid`(megatron_patch/data/utils.py)가
복원하지만, **데이터 생산·검증 단계에서 이 스캔으로 잡는 것이 1차 방어선**이다.

판정: bestfit --pad-doc-multiple G 산출물에서 정상 EOD run(문서 종결 + pad)은
반드시 G의 배수 위치에서 끝난다. run 끝이 G 비정렬인 bin = 내부 잡탕 EOD 보유.

사용:
  # 단일 prefix 표본 스캔 (기본 2000 bins)
  python scan_internal_eod.py --prefix <.../data_text_document>
  # blend yaml 전 멤버 스캔
  python scan_internal_eod.py --blend-yaml <.../lc_a_32k_blend.yaml>
  # 전수 스캔(전 bin) + 오염 bin 인덱스 출력
  python scan_internal_eod.py --prefix ... --full --list-bad

exit code: 오염 발견 시 1 (CI/러너북 게이트용).
2026-08-23 표본 실측(2000 bins/set): longblocks 1/2008 · code_review 1/2058 ·
rewriting 1/2064 오염, 나머지 29멤버(신규 specialized 15종 포함) 청정.
"""
import argparse
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "backends/megatron/Megatron-LM-251125"))

from megatron.core.datasets.indexed_dataset import IndexedDataset  # noqa: E402


def scan_prefix(prefix, grid, eod, sample, full, list_bad):
    ds = IndexedDataset(prefix)
    n = len(ds)
    step = 1 if full else max(1, n // sample)
    bad = []
    checked = 0
    for k in range(0, n, step):
        seq = np.asarray(ds.get(k))
        is_eod = seq == eod
        run_end = np.where(is_eod[:-1] & ~is_eod[1:])[0] + 1
        checked += 1
        if (run_end % grid != 0).any():
            bad.append(k)
    name = prefix
    flag = "  <-- 오염" if bad else ""
    print(f"{name}: bad {len(bad)}/{checked}{flag}", flush=True)
    if bad and list_bad:
        print(f"  bad bin indices: {bad[:200]}{' ...' if len(bad) > 200 else ''}")
    del ds
    return len(bad)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prefix", help="IndexedDataset prefix (…/data_text_document)")
    g.add_argument("--blend-yaml", help="data preset yaml — data-path 전 멤버 스캔")
    ap.add_argument("--grid", type=int, default=16, help="pad-doc-multiple (기본 16)")
    ap.add_argument("--eod", type=int, default=0, help="EOD token id (alpha v5: 0)")
    ap.add_argument("--sample", type=int, default=2000, help="표본 bins/set (기본 2000)")
    ap.add_argument("--full", action="store_true", help="전수 스캔")
    ap.add_argument("--list-bad", action="store_true", help="오염 bin 인덱스 출력")
    args = ap.parse_args()

    if args.prefix:
        prefixes = [args.prefix]
    else:
        m = re.search(r'data-path: "([^"]+)"', open(args.blend_yaml).read())
        toks = m.group(1).split()
        prefixes = [toks[i + 1] for i in range(0, len(toks), 2)]

    total_bad = 0
    for p in prefixes:
        total_bad += scan_prefix(p, args.grid, args.eod, args.sample, args.full, args.list_bad)
    print("CLEAN" if total_bad == 0 else f"CONTAMINATED ({total_bad} bins)")
    sys.exit(1 if total_bad else 0)


if __name__ == "__main__":
    main()
