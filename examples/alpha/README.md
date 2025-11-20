# Alpha 프로젝트

Qwen3-Next Mamba 기반의 실험적 언어 모델 학습 프로젝트

---

## 개요

Alpha는 Qwen3-Next-80B-A3B 아키텍처를 기반으로 한 독립적인 연구 프로젝트입니다. Hybrid Mamba-Attention 모델과 Mixture-of-Experts (MoE)를 결합하여 효율적인 대규모 언어 모델 학습을 목표로 합니다.

### 주요 특징

- **Hybrid Architecture**: Mamba SSM + Multi-Head Attention 조합
- **Mixture-of-Experts**: 256 experts with 8 active per token
- **YAML 기반 설정**: 모듈화되고 읽기 쉬운 설정 관리
- **실험 추적**: 체계적인 실험 기록 및 재현성
- **H100 최적화**: 8-GPU 환경 최적화 설정

---

## 빠른 시작

### 1. 환경 검증

```bash
cd examples/alpha
bash scripts/validate_environment.sh
```

### 2. 데이터 전처리 (필요시)

```bash
cd toolkits/pretrain_data_preprocessing/
bash preprocess_kormo_subset.sh 1  # 1% subset
```

### 3. 학습 시작

```bash
cd examples/alpha
bash train.sh
```

기본 설정으로 학습이 시작됩니다:
- 모델: `baseline_24L`
- 학습: `pretrain`
- 인프라: `h100x8`
- 데이터: `kormo_1pct`

### 4. 다른 설정 사용

```bash
bash train.sh [model] [training] [infra] [data]

# 예시
bash train.sh baseline_24L pretrain h100x8 kormo_1pct
```

---

## 프로젝트 구조

```
examples/alpha/
├── README.md                    # 이 파일
├── pretrain_alpha.py            # 메인 학습 스크립트
├── train.sh                     # YAML 기반 통합 학습 스크립트
│
├── configs/                     # YAML 설정 파일
│   ├── model/
│   │   └── baseline_24L.yaml   # 모델 아키텍처 설정
│   ├── training/
│   │   ├── pretrain.yaml       # 학습 하이퍼파라미터
│   │   └── h100x8.yaml         # H100 8-GPU 인프라 설정
│   ├── data/
│   │   └── kormo_1pct.yaml     # 데이터셋 설정
│   └── env.yaml                # 환경 변수 설정
│
├── scripts/                     # 유틸리티 스크립트
│   ├── validate_environment.sh # 환경 검증
│   ├── verify_fixes.sh         # 버그 수정 확인
│   └── load_config.sh          # YAML 로드 헬퍼
│
├── experiments/                 # 실험 기록
│   └── 20250117_baseline_24L/
│       ├── run_original.sh     # 원본 테스트 스크립트
│       ├── config_snapshot.yaml # 설정 스냅샷
│       └── notes.md            # 실험 노트
│
├── docs/                        # 문서
│   ├── ARCHITECTURE.md         # 모델 아키텍처 상세
│   ├── EXPERIMENTS.md          # 실험 로그
│   └── SETUP.md                # 환경 세팅 가이드
│
└── outputs/                     # 학습 결과 (자동 생성)
    └── alpha_baseline_24L_*/
        ├── checkpoints/
        ├── tensorboard/
        └── logs/
```

---

## 설정 파일

### 모델 설정 (`configs/model/*.yaml`)

모델 아키텍처를 정의합니다:
- 레이어 수, Hidden size
- Attention 설정 (heads, GQA)
- MoE 설정 (experts, routing)
- Hybrid 패턴 (Mamba + Attention)

### 학습 설정 (`configs/training/*.yaml`)

학습 하이퍼파라미터를 정의합니다:
- Learning rate, Optimizer
- 학습 토큰 수
- 체크포인트 저장 간격

### 인프라 설정 (`configs/training/h100x8.yaml`)

분산 학습 및 최적화 전략:
- 병렬화 (TP, PP, EP, CP)
- 배치 크기
- Activation checkpointing

### 데이터 설정 (`configs/data/*.yaml`)

데이터셋 경로 및 로딩 설정:
- 데이터 경로
- Train/Valid split
- DataLoader workers

---

## 주요 명령

### 환경 검증
```bash
bash scripts/validate_environment.sh
```

### 설정 테스트
```bash
bash scripts/load_config.sh
```

### 학습 시작
```bash
bash train.sh
```

### TensorBoard
```bash
tensorboard --logdir outputs/alpha_*/tensorboard --port 6006
```

---

## Baseline 24L 모델

최초 실험 모델 설정:

| 항목 | 값 |
|------|-----|
| **레이어 수** | 24 |
| **Hidden Size** | 2048 |
| **Attention Heads** | 32 |
| **Query Groups** | 2 (GQA) |
| **Experts** | 256 |
| **Router TopK** | 8 |
| **Hybrid Ratio** | 12.5% (3/24 layers) |

**메모리 절약 전략**:
- 레이어 75% 감소 (96 → 24)
- Expert 50% 감소 (512 → 256), 크기 증가로 보상
- Attention head_dim 4배 감소 (256 → 64)

---

## 실험 추적

각 실험은 `experiments/YYYYMMDD_name/` 디렉토리에 기록됩니다:

- `run_original.sh`: 원본 스크립트
- `config_snapshot.yaml`: 설정 스냅샷
- `notes.md`: 실험 노트 (목표, 결과, 관찰)

학습 실행 시 자동으로 생성되는 항목:
- `outputs/alpha_*/checkpoints/`: 모델 체크포인트
- `outputs/alpha_*/tensorboard/`: TensorBoard 로그
- `outputs/alpha_*/logs/train.log`: 학습 로그

---

## 문서

- [**ARCHITECTURE.md**](docs/ARCHITECTURE.md): 모델 아키텍처 상세 설명
- [**EXPERIMENTS.md**](docs/EXPERIMENTS.md): 실험 로그 및 결과
- [**SETUP.md**](docs/SETUP.md): 환경 세팅 상세 가이드

---

## 트러블슈팅

### OOM (Out of Memory)

1. `configs/training/h100x8.yaml`에서 `micro_batch_size` 감소
2. `pipeline_parallel` 증가 (1 → 2 또는 4)
3. Activation checkpointing 강화

### 낮은 Throughput

1. `micro_batch_size` 증가
2. `num_workers` 조정
3. Flash Attention 3 활성화 확인

### 설정 오류

```bash
# YAML 문법 확인
python3 -c "import yaml; yaml.safe_load(open('configs/model/baseline_24L.yaml'))"

# 환경 검증
bash scripts/validate_environment.sh
```

---

## 참고 자료

- **Pai-Megatron-Patch**: [메인 README](../../README.md)
- **Qwen3 Best Practice**: https://github.com/yanring/Megatron-MoE-ModelZoo/tree/main/best_practice/Qwen3
- **Megatron-LM**: https://github.com/NVIDIA/Megatron-LM

---

## 라이선스

Apache License 2.0 (Pai-Megatron-Patch 상속)

---

## 기여

Alpha는 실험적 프로젝트입니다. 실험 결과 및 개선 사항은 `experiments/` 디렉토리에 기록해주세요.
