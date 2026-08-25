#!/usr/bin/env python3
"""b_remainder.py — DD 슬라이스 런에서 산출 안 된 시드만 골라 잔여 parquet 생성.

OxAlpha 생성 슬라이스(트랙 B r2, 25k)는 레코드별 폴백이 없다 — 제공자 장애나
파싱 실패로 빠진 시드는 이 스크립트로 추려 Gemma 슬라이스 런에 넘긴다(시드 단위 폴백).

사용:
  python3 b_remainder.py --seeds ko_seed_r2_or.parquet \
      --artifacts 'artifacts/ko_chat_b_r2_or/**/*.parquet' --out ko_seed_r2_or_rem.parquet
종료 코드: 0 = 잔여 있음(파일 생성), 3 = 잔여 없음(파일 미생성).
"""
import argparse
import glob
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--artifacts", required=True, help="산출 parquet glob")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seeds = pd.read_parquet(args.seeds)
    files = sorted(glob.glob(args.artifacts, recursive=True))
    done = set()
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["seed_uuid", "assistant_turn"])
        except Exception:  # noqa: BLE001 — 컬럼 없는 중간 산출물
            continue
        ok = df["assistant_turn"].notna()
        done.update(df.loc[ok, "seed_uuid"].astype(str))
    rem = seeds[~seeds["seed_uuid"].astype(str).isin(done)].reset_index(drop=True)
    print(f"seeds={len(seeds)} done={len(done)} remainder={len(rem)} (files={len(files)})")
    if len(rem) == 0:
        sys.exit(3)
    rem.to_parquet(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
