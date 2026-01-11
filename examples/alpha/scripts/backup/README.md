# Alpha Checkpoint Backup System

Megatron 분산 체크포인트를 MinIO 오브젝트 스토리지로 자동 백업하는 시스템입니다.

## 기능

- **자동 백업**: Cron을 통해 매 시간 새로운 체크포인트 감지 및 백업
- **중복 방지**: 상태 추적으로 이미 백업된 체크포인트 스킵
- **대용량 지원**: 멀티파트 업로드로 29GB+ 체크포인트 처리
- **무결성 검증**: 업로드 후 파일 크기 검증
- **병렬 업로드**: 여러 파일 동시 업로드로 속도 최적화

## 빠른 시작

### 1. 환경 설정

```bash
cd examples/alpha/scripts/backup/

# 환경변수 파일 생성
cp .env.example .env

# .env 파일 편집 (MinIO 접속 정보 입력)
vim .env
```

### 2. 현재 상태 확인

```bash
# 백업 대기 중인 체크포인트 확인
./checkpoint_backup.sh --list
```

### 3. 테스트 실행 (Dry Run)

```bash
# 실제 업로드 없이 시뮬레이션
./checkpoint_backup.sh --dry-run
```

### 4. 실제 백업 실행

```bash
# 백업 실행
./checkpoint_backup.sh
```

### 5. Cron 설정 (자동화)

```bash
# crontab 편집
crontab -e

# 매 시간 정각에 실행 (아래 줄 추가)
0 * * * * /home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/scripts/backup/checkpoint_backup.sh >> /home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha/scripts/backup/logs/cron.log 2>&1
```

## 사용법

```bash
# 기본 백업 실행
./checkpoint_backup.sh

# Dry run (시뮬레이션)
./checkpoint_backup.sh --dry-run

# 모든 체크포인트 강제 재백업
./checkpoint_backup.sh --force

# 현재 상태 확인
./checkpoint_backup.sh --list

# Python 스크립트 직접 실행
python checkpoint_backup.py --help
```

## 환경 변수

| 변수 | 필수 | 설명 | 예시 |
|------|------|------|------|
| `MINIO_ENDPOINT` | O | MinIO 서버 주소 | `http://minio.example.com:9000` |
| `MINIO_ACCESS_KEY` | O | Access Key | `minioadmin` |
| `MINIO_SECRET_KEY` | O | Secret Key | `minioadmin123` |
| `MINIO_SECURE` | X | HTTPS 사용 여부 | `false` (기본값) |
| `MINIO_REGION` | X | Region | `us-east-1` (기본값) |

## 디렉토리 구조

```
scripts/backup/
├── checkpoint_backup.py    # 메인 Python 스크립트
├── checkpoint_backup.sh    # Wrapper 스크립트 (cron용)
├── config.yaml             # 설정 파일
├── .env                    # 환경변수 (gitignore됨)
├── .env.example            # 환경변수 템플릿
├── README.md               # 이 문서
├── state/                  # 상태 추적
│   └── backup_state.json   # 백업 상태 (자동 생성)
└── logs/                   # 로그 파일
    └── backup_*.log        # 일별 로그 (30일 보관)
```

## MinIO 버킷 구조

```
alpha-checkpoints/                          # 버킷
└── megatron-checkpoints/                   # Prefix
    └── alpha_baseline_48L_20251219_095156/ # 실험명
        ├── iter_0050000/                   # 체크포인트
        │   ├── __0_0.distcp
        │   ├── __0_1.distcp
        │   ├── ... (16 shard files)
        │   ├── common.pt
        │   ├── .metadata
        │   └── metadata.json
        ├── iter_0100000/
        └── iter_0150000/
```

## 설정 파일 (config.yaml)

주요 설정 항목:

```yaml
backup:
  source:
    base_path: "/path/to/outputs"           # 체크포인트 기본 경로
    experiment_pattern: "alpha_baseline_*"   # 실험 디렉토리 패턴

  destination:
    bucket: "alpha-checkpoints"              # MinIO 버킷명
    prefix: "megatron-checkpoints"           # S3 prefix

  upload:
    multipart_chunksize: 104857600           # 100MB 청크
    max_concurrency: 8                       # 파일당 병렬 스레드
    max_parallel_files: 4                    # 동시 업로드 파일 수
```

## 복원 방법

MinIO에서 체크포인트 복원:

```bash
# mc (MinIO Client) 설치
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
./mc alias set myminio http://minio.example.com:9000 ACCESS_KEY SECRET_KEY

# 체크포인트 다운로드
./mc cp -r myminio/alpha-checkpoints/megatron-checkpoints/alpha_baseline_48L_20251219_095156/iter_0100000 ./restored_checkpoint/
```

## 문제 해결

### "Another backup process is running"
```bash
# 잠금 파일 확인
cat state/backup.lock

# 필요시 잠금 해제 (프로세스가 없는 경우만)
rm state/backup.lock
```

### MinIO 연결 실패
```bash
# 연결 테스트
curl -I http://your-minio:9000/minio/health/live

# 환경변수 확인
echo $MINIO_ENDPOINT
```

### 업로드 실패
- 로그 확인: `tail -f logs/backup_$(date +%Y%m%d).log`
- 네트워크 대역폭 확인
- MinIO 서버 디스크 공간 확인

## 의존성

- Python 3.8+
- boto3 (S3 클라이언트)
- PyYAML

```bash
pip install boto3 pyyaml
```
