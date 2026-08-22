#!/usr/bin/env python3
"""LC-A 런치 프리플라이트 — 통과(exit 0)해야 학습을 시작한다.

검사 항목:
  1. lc_a_32k_blend.yaml의 전 데이터 경로 .bin+.idx 존재
  2. 각 prefix가 IndexedDataset으로 실제 로드되고 첫/끝 bin 길이 == 32768
     (NFS 동기화 중 반쯤 쓰인 파일 방지 — 존재 확인만으로는 부족)
  3. --deep <prefix substr>: 해당 셋 표본 bins의 EOD-run 경계 %16 검사
     (늦게 도착한 specialized용; 기본 프리플라이트에선 생략)
  4. P3 체크포인트 latest == 26832
  5. GPU 전체 유휴 (--skip-gpu로 생략 가능)

사용:  python scripts/lc_a_preflight.py [--deep specialized] [--skip-gpu]
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/alpha
REPO = os.path.dirname(os.path.dirname(ROOT))
sys.path.insert(0, os.path.join(REPO, "backends/megatron/Megatron-LM-251125"))

BLEND = os.path.join(ROOT, "configs/data/lc_a_32k_blend.yaml")
CKPT = os.path.join(ROOT, "outputs/alpha_baseline_48L_stage2_20260822_123916/checkpoints")
SEQ = 32768
PAD_MULT = 16
EOD = 0


def fail(msg):
    print(f"[PREFLIGHT FAIL] {msg}")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", default=None, help="substring: 표본 %16 검사 대상 prefix")
    ap.add_argument("--skip-gpu", action="store_true")
    args = ap.parse_args()

    # 1+2) 블렌드 경로 전수 + 로드 검사
    m = re.search(r'data-path: "([^"]+)"', open(BLEND).read())
    toks = m.group(1).split()
    prefixes = [toks[i + 1] for i in range(0, len(toks), 2)]
    print(f"blend members: {len(prefixes)}")
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    for p in prefixes:
        for ext in (".bin", ".idx"):
            if not os.path.exists(p + ext):
                fail(f"missing {p + ext}")
        try:
            ds = IndexedDataset(p)
        except Exception as e:  # noqa: BLE001
            fail(f"unreadable {p}: {e}")
        n = len(ds)
        if n == 0:
            fail(f"empty dataset {p}")
        for k in (0, n - 1):
            if len(ds.get(k)) != SEQ:
                fail(f"{p} bin {k} length {len(ds.get(k))} != {SEQ} (반쯤 쓰인 파일?)")
        del ds
    print("[ok] all blend paths load, first/last bins == 32768")

    # 3) 심층 %16 표본 검사
    if args.deep:
        import numpy as np

        targets = [p for p in prefixes if args.deep in p]
        print(f"deep-check targets: {len(targets)}")
        for p in targets:
            ds = IndexedDataset(p)
            n = len(ds)
            step = max(1, n // 50)
            bad = 0
            for k in range(0, n, step):
                seq = np.asarray(ds.get(k))
                is_eod = seq == EOD
                # EOD-run의 끝 위치(다음 토큰이 비-EOD가 되는 경계)가 %16이어야 함
                run_end = np.where(is_eod[:-1] & ~is_eod[1:])[0] + 1
                if (run_end % PAD_MULT != 0).any():
                    bad += 1
            if bad:
                fail(f"{p}: %{PAD_MULT} misaligned bins {bad} (pad16 재패킹 안 된 산출물?)")
            del ds
        print(f"[ok] deep %{PAD_MULT} check passed")

    # 4) 체크포인트
    lat = os.path.join(CKPT, "latest_checkpointed_iteration.txt")
    if not os.path.exists(lat):
        fail(f"checkpoint missing: {lat}")
    it = open(lat).read().strip()
    if it != "26832":
        fail(f"unexpected latest iteration {it} (expected 26832)")
    print(f"[ok] P3 checkpoint present (iter {it})")

    # 5) GPU 유휴
    if not args.skip_gpu:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        )
        used = sum(int(x) for x in out.split())
        if used > 1000:
            fail(f"GPUs busy (sum used {used} MiB)")
        print("[ok] GPUs idle")

    print("PREFLIGHT PASS")


if __name__ == "__main__":
    main()
