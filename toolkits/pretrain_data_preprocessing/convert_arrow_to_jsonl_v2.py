#!/usr/bin/env python3
"""
고성능 Arrow → JSONL 변환 스크립트 (파일 레벨 병렬화)
224 CPU 코어를 최대한 활용하여 4.5TB 데이터를 빠르게 처리

특징:
- 파일 레벨 병렬 처리: 모든 Arrow 파일을 병렬로 처리
- 메모리 효율적: 스트리밍 방식
- 진행률 추적: 실시간 진행 상황 표시
- 디스크 공간 체크: 처리 전 공간 확인
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from datasets import Dataset


def check_disk_space(path: str, required_gb: float):
    """디스크 공간 확인"""
    stat = os.statvfs(path)
    available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

    print(f"\n=== 디스크 공간 확인 ===")
    print(f"경로: {path}")
    print(f"필요 공간: {required_gb:.1f} GB")
    print(f"사용 가능: {available_gb:.1f} GB")

    if available_gb < required_gb:
        raise RuntimeError(
            f"❌ 디스크 공간 부족!\n"
            f"   필요: {required_gb:.1f} GB\n"
            f"   사용 가능: {available_gb:.1f} GB"
        )
    print(f"✅ 충분한 공간 확보됨 ({available_gb - required_gb:.1f} GB 여유)\n")


def process_single_arrow_file(args_tuple):
    """
    단일 Arrow 파일 처리 (파일 레벨 병렬화)

    Returns:
        (파일경로, 처리된행수, JSONL라인들, 에러메시지)
    """
    arrow_file, text_column = args_tuple

    try:
        # HuggingFace datasets 포맷으로 읽기
        dataset = Dataset.from_file(str(arrow_file))

        # JSONL 라인 생성
        lines = []
        for sample in dataset:
            if text_column in sample and sample[text_column]:
                line = json.dumps({"text": sample[text_column]}, ensure_ascii=False)
                lines.append(line)

        return (str(arrow_file), len(lines), lines, None)

    except Exception as e:
        return (str(arrow_file), 0, [], str(e))


def convert_arrow_to_jsonl(
    input_dir: str,
    output_file: str,
    text_column: str = "text",
    num_workers: Optional[int] = None
):
    """
    Arrow 포맷을 JSONL로 변환 (파일 레벨 병렬 처리)

    Args:
        input_dir: Arrow 파일이 있는 디렉토리
        output_file: 출력 JSONL 파일 경로
        text_column: 텍스트 컬럼명
        num_workers: 병렬 worker 수 (None이면 CPU 코어 수)
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Arrow 파일 찾기 (shard 구조 또는 flat 구조 모두 지원)
    arrow_files = sorted(input_path.rglob("*.arrow"))

    if not arrow_files:
        raise ValueError(f"No arrow files found in {input_dir}")

    print(f"\n{'='*60}")
    print(f"Arrow → JSONL 변환 시작 (파일 레벨 병렬화)")
    print(f"{'='*60}")
    print(f"입력: {input_dir}")
    print(f"출력: {output_file}")
    print(f"Arrow 파일 수: {len(arrow_files):,}")
    print(f"텍스트 컬럼: {text_column}")
    print(f"CPU 코어: {cpu_count()}")

    # Worker 수 결정
    if num_workers is None:
        num_workers = cpu_count()

    print(f"병렬 worker: {num_workers}")
    print(f"{'='*60}\n")

    # 디스크 공간 체크 (예상 크기: 입력의 1.2배)
    input_size_gb = sum(f.stat().st_size for f in arrow_files) / (1024**3)
    required_space_gb = input_size_gb * 1.2
    check_disk_space(str(output_path.parent), required_space_gb)

    # 파일 레벨 병렬 처리
    args_list = [(arrow_file, text_column) for arrow_file in arrow_files]

    print(f"파일 레벨 병렬 처리 시작 ({num_workers} workers)...\n")

    total_lines = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        with Pool(processes=num_workers) as pool:
            for result in tqdm(
                pool.imap(process_single_arrow_file, args_list),
                total=len(args_list),
                desc="변환 진행",
                unit=" 파일"
            ):
                file_path, lines_count, lines, error = result

                if error:
                    print(f"  ⚠ {Path(file_path).name} 처리 실패: {error}")
                    continue

                # 결과를 바로 파일에 쓰기
                for line in lines:
                    f_out.write(line + '\n')
                    total_lines += 1

    # 최종 통계
    output_size_gb = output_path.stat().st_size / (1024**3)

    print(f"\n{'='*60}")
    print(f"✅ 변환 완료!")
    print(f"{'='*60}")
    print(f"출력 파일: {output_path}")
    print(f"총 라인 수: {total_lines:,}")
    print(f"파일 크기: {output_size_gb:.2f} GB")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="고성능 Arrow → JSONL 변환 (파일 레벨 병렬화)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Arrow 파일이 있는 디렉토리"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="출력 JSONL 파일 경로"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="텍스트 컬럼명 (기본: text)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="병렬 worker 수 (기본: CPU 코어 수)"
    )

    args = parser.parse_args()

    convert_arrow_to_jsonl(
        input_dir=args.input_dir,
        output_file=args.output_file,
        text_column=args.text_column,
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()
