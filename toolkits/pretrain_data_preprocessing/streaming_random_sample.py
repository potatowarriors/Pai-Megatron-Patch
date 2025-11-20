#!/usr/bin/env python3
"""
스트리밍 무작위 샘플링 스크립트
메모리 효율적으로 대용량 JSONL 파일에서 무작위 샘플 추출

특징:
- Reservoir Sampling: 메모리 사용량 일정
- Seed 고정: 재현 가능한 샘플링
- 진행률 표시: 실시간 진행 상황
- 통계 정보: 샘플링 비율 및 결과 요약
"""

import os
import random
import argparse
from pathlib import Path
from tqdm import tqdm


def count_lines(file_path: str) -> int:
    """
    파일의 총 라인 수 계산

    Args:
        file_path: 파일 경로

    Returns:
        총 라인 수
    """
    print(f"파일 라인 수 계산 중: {file_path}")

    line_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in tqdm(f, desc="라인 계산", unit=" lines"):
            line_count += 1

    return line_count


def reservoir_sample(
    input_file: str,
    output_file: str,
    sample_rate: float,
    seed: int = 42
):
    """
    Reservoir Sampling으로 무작위 샘플 추출
    메모리 효율적 (전체 파일을 메모리에 로드하지 않음)

    Args:
        input_file: 입력 JSONL 파일
        output_file: 출력 JSONL 파일
        sample_rate: 샘플링 비율 (0.01 = 1%, 0.1 = 10%)
        seed: 랜덤 시드
    """
    random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Reservoir Sampling (메모리 효율적)")
    print(f"{'='*60}")
    print(f"입력: {input_file}")
    print(f"출력: {output_file}")
    print(f"샘플링 비율: {sample_rate * 100:.2f}%")
    print(f"Random seed: {seed}")
    print(f"{'='*60}\n")

    # 출력 디렉토리 생성
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    sampled_lines = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        # 파일 크기 확인 (진행률 표시용)
        f_in.seek(0, 2)  # 파일 끝으로
        file_size = f_in.tell()
        f_in.seek(0)  # 처음으로

        with tqdm(total=file_size, unit='B', unit_scale=True, desc="샘플링") as pbar:
            for line in f_in:
                total_lines += 1

                # Reservoir sampling: 일정 확률로 샘플 선택
                if random.random() < sample_rate:
                    f_out.write(line)
                    sampled_lines += 1

                # 진행률 업데이트
                pbar.update(len(line.encode('utf-8')))

    # 결과 통계
    actual_rate = sampled_lines / total_lines if total_lines > 0 else 0

    print(f"\n{'='*60}")
    print(f"✅ 샘플링 완료!")
    print(f"{'='*60}")
    print(f"총 라인 수: {total_lines:,}")
    print(f"샘플 라인 수: {sampled_lines:,}")
    print(f"실제 샘플링 비율: {actual_rate * 100:.4f}%")
    print(f"목표 샘플링 비율: {sample_rate * 100:.2f}%")
    print(f"오차: {abs(actual_rate - sample_rate) * 100:.4f}%")

    # 출력 파일 크기
    output_size_gb = Path(output_file).stat().st_size / (1024**3)
    print(f"출력 파일 크기: {output_size_gb:.2f} GB")
    print(f"{'='*60}\n")


def indexed_sample(
    input_file: str,
    output_file: str,
    sample_rate: float,
    seed: int = 42
):
    """
    인덱스 기반 무작위 샘플 추출
    정확한 샘플링 비율, 하지만 2번 파일 읽기 필요

    Args:
        input_file: 입력 JSONL 파일
        output_file: 출력 JSONL 파일
        sample_rate: 샘플링 비율 (0.01 = 1%, 0.1 = 10%)
        seed: 랜덤 시드
    """
    random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Indexed Sampling (정확한 비율)")
    print(f"{'='*60}")
    print(f"입력: {input_file}")
    print(f"출력: {output_file}")
    print(f"샘플링 비율: {sample_rate * 100:.2f}%")
    print(f"Random seed: {seed}")
    print(f"{'='*60}\n")

    # 출력 디렉토리 생성
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Step 1: 총 라인 수 계산
    total_lines = count_lines(input_file)
    sample_size = int(total_lines * sample_rate)

    print(f"\n총 라인 수: {total_lines:,}")
    print(f"샘플 크기: {sample_size:,}")

    # Step 2: 무작위 인덱스 선택
    print(f"\n무작위 인덱스 생성 중...")
    selected_indices = set(random.sample(range(total_lines), sample_size))
    print(f"✅ {len(selected_indices):,}개 인덱스 선택 완료")

    # Step 3: 선택된 라인만 추출
    print(f"\n샘플 라인 추출 중...")
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for idx, line in enumerate(tqdm(f_in, total=total_lines, desc="추출")):
            if idx in selected_indices:
                f_out.write(line)

    # 결과 통계
    actual_sample_size = sum(1 for _ in open(output_file, 'r'))
    actual_rate = actual_sample_size / total_lines

    print(f"\n{'='*60}")
    print(f"✅ 샘플링 완료!")
    print(f"{'='*60}")
    print(f"총 라인 수: {total_lines:,}")
    print(f"샘플 라인 수: {actual_sample_size:,}")
    print(f"실제 샘플링 비율: {actual_rate * 100:.4f}%")
    print(f"목표 샘플링 비율: {sample_rate * 100:.2f}%")

    # 출력 파일 크기
    output_size_gb = Path(output_file).stat().st_size / (1024**3)
    print(f"출력 파일 크기: {output_size_gb:.2f} GB")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="스트리밍 무작위 샘플링 (메모리 효율적)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 JSONL 파일"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="출력 JSONL 파일"
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        required=True,
        help="샘플링 비율 (0.01 = 1%%, 0.1 = 10%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (재현성, 기본: 42)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["reservoir", "indexed"],
        default="reservoir",
        help="샘플링 방법 (reservoir: 빠름, indexed: 정확)"
    )

    args = parser.parse_args()

    # 입력 파일 존재 확인
    if not Path(args.input).exists():
        raise FileNotFoundError(f"입력 파일이 존재하지 않음: {args.input}")

    # 샘플링 비율 검증
    if not 0 < args.sample_rate <= 1:
        raise ValueError(f"샘플링 비율은 0과 1 사이여야 함: {args.sample_rate}")

    # 샘플링 실행
    if args.method == "reservoir":
        reservoir_sample(
            input_file=args.input,
            output_file=args.output,
            sample_rate=args.sample_rate,
            seed=args.seed
        )
    else:  # indexed
        indexed_sample(
            input_file=args.input,
            output_file=args.output,
            sample_rate=args.sample_rate,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
