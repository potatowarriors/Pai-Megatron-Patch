#!/usr/bin/env python3
"""LC-B 런치 프리플라이트 — 통과(exit 0)해야 학습을 시작한다.

lc_a_preflight.py의 LC-B 판. 차이점:
  - 혼합 bin 길이: cpt_lc_packed_128k_pad16 멤버는 131072, lc_filler_packed_32k_pad16
    멤버는 32768 (32k 팩 재사용 — 131072 %% 32768 == 0이라 concat-and-chunk가
    항상 bin 경계에서 잘림; lc_b_128k_blend.yaml 헤더 참조).
  - 체크포인트 게이트: lc_b.yaml의 load: 경로에서 latest == --require-iter
    (LC-A 완주 최종 ckpt 1113). 경로를 yaml에서 읽으므로 preset과 드리프트 없음.
  - deep %16 검사는 128k LC 멤버 대상. longblocks 2/40,876 오염(0.005%)은 기지·
    수용(런타임 snap 수리)이므로 표본 중 소수 발견은 경고, 구조적(>10%)만 실패.
  - 블렌드 가중치 합 == 1 ±1e-6, 디스크 여유 검사 (optim 동봉 세이브 ~수백GB).

사용:  python scripts/lc_b_preflight.py [--deep] [--require-iter 1113] [--skip-gpu]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/alpha
REPO = os.path.dirname(os.path.dirname(ROOT))
sys.path.insert(0, os.path.join(REPO, "backends/megatron/Megatron-LM-251125"))

BLEND = os.path.join(ROOT, "configs/data/lc_b_128k_blend.yaml")
TRAIN_YAML = os.path.join(ROOT, "configs/training/lc_b.yaml")
SEQ_LC = 131072
SEQ_FILLER = 32768
PAD_MULT = 16
EOD = 0
MIN_FREE_TB = 2.0  # optimizer 동봉 세이브 ~7회 × 수백GB 대비


def fail(msg):
    print(f"[PREFLIGHT FAIL] {msg}")
    sys.exit(1)


def expected_len(prefix):
    if "cpt_lc_packed_128k_pad16" in prefix:
        return SEQ_LC
    if "lc_filler_packed_32k_pad16" in prefix:
        return SEQ_FILLER
    fail(f"unknown pack family for {prefix} (128k/32k 판별 불가)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true", help="128k LC 멤버 표본 %16 검사")
    ap.add_argument("--require-iter", default="1113", help="load ckpt latest 요구값")
    ap.add_argument("--skip-gpu", action="store_true")
    args = ap.parse_args()

    # 1) 블렌드 파싱 + 가중치 합
    m = re.search(r'data-path: "([^"]+)"', open(BLEND).read())
    toks = m.group(1).split()
    weights = [float(toks[i]) for i in range(0, len(toks), 2)]
    prefixes = [toks[i + 1] for i in range(0, len(toks), 2)]
    print(f"blend members: {len(prefixes)}")
    wsum = sum(weights)
    if abs(wsum - 1.0) > 1e-6:
        fail(f"blend weights sum {wsum:.8f} != 1 (±1e-6)")
    print(f"[ok] weights sum {wsum:.6f}")

    # 2) 전 경로 로드 + 첫/끝 bin 길이 (반쯤 쓰인 NFS 파일 방지 — 존재 확인만으론 부족)
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
        want = expected_len(p)
        for k in (0, n - 1):
            got = len(ds.get(k))
            if got != want:
                fail(f"{p} bin {k} length {got} != {want}")
        del ds
    print(f"[ok] all {len(prefixes)} members load; bins == 131072(LC) / 32768(filler)")

    # 3) deep: 128k LC 멤버 표본 %16 검사 (기지 오염 소수는 경고, 구조적만 실패)
    if args.deep:
        import numpy as np

        targets = [p for p in prefixes if "cpt_lc_packed_128k_pad16" in p]
        print(f"deep-check targets: {len(targets)} (128k LC members)")
        for p in targets:
            ds = IndexedDataset(p)
            n = len(ds)
            step = max(1, n // 50)
            checked = bad = 0
            for k in range(0, n, step):
                seq = np.asarray(ds.get(k))
                is_eod = seq == EOD
                run_end = np.where(is_eod[:-1] & ~is_eod[1:])[0] + 1
                checked += 1
                if (run_end % PAD_MULT != 0).any():
                    bad += 1
            if bad > max(2, checked // 10):
                fail(f"{p}: %{PAD_MULT} misaligned {bad}/{checked} (pad16 팩 아님?)")
            if bad:
                print(f"  [warn] {p}: {bad}/{checked} bins with off-grid EOD "
                      "(기지 오염 범위 — 런타임 snap이 수리)")
            del ds
        print(f"[ok] deep %{PAD_MULT} check passed")

    # 4) 체크포인트 — lc_b.yaml의 load: 경로에서 latest 확인
    ty = open(TRAIN_YAML).read()
    lm = re.search(r"^load:\s*(\S+)", ty, re.M)
    if not lm:
        fail(f"no load: in {TRAIN_YAML}")
    ckpt = lm.group(1)
    lat = os.path.join(ckpt, "latest_checkpointed_iteration.txt")
    if not os.path.exists(lat):
        fail(f"checkpoint missing: {lat}")
    it = open(lat).read().strip()
    if it != str(args.require_iter):
        fail(f"latest iteration {it} != required {args.require_iter} (LC-A 미완주?)")
    it_dir = os.path.join(ckpt, f"iter_{int(it):07d}")
    if not os.path.isdir(it_dir):
        fail(f"checkpoint dir missing: {it_dir}")
    print(f"[ok] LC-A checkpoint present (iter {it})")

    # 5) 디스크 여유 (outputs 볼륨)
    free_tb = shutil.disk_usage(os.path.join(ROOT, "outputs")).free / 1e12
    if free_tb < MIN_FREE_TB:
        fail(f"outputs free {free_tb:.1f}TB < {MIN_FREE_TB}TB (optim 동봉 세이브 대비)")
    print(f"[ok] disk free {free_tb:.1f}TB")

    # 6) GPU 유휴
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
