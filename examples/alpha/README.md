# Alpha 프로젝트

Qwen3-Next Mamba 기반의 실험적 언어 모델 학습 프로젝트

---

## 개요

Alpha는 Qwen3-Next-80B-A3B 아키텍처를 기반으로 한 독립적인 연구 프로젝트입니다. Hybrid Mamba-Attention 모델과 Mixture-of-Experts (MoE)를 결합하여 효율적인 대규모 언어 모델 학습을 목표로 합니다.

### 주요 특징

- **Hybrid Architecture**: Mamba SSM + Multi-Head Attention 조합
- **Mixture-of-Experts**: 128 experts with 8 active per token
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
- 모델: `baseline_48L`
- 학습: `pretrain`
- 인프라: `h100x8`
- 데이터: `kormo_1pct`

### 4. 다른 설정 사용

```bash
bash train.sh [model] [training] [infra] [data]

# 예시
bash train.sh baseline_48L pretrain h100x8 kormo_1pct
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
│   │   └── baseline_48L.yaml   # 모델 아키텍처 설정
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
│   └── 20250117_baseline_48L/
│       ├── run_original.sh     # 원본 테스트 스크립트
│       ├── config_snapshot.yaml # 설정 스냅샷
│       └── notes.md            # 실험 노트
│
├── docs/                        # 문서
│   ├── ARCHITECTURE.md         # 모델 아키텍처 상세
│   ├── CONVERSION.md           # HuggingFace 변환 가이드
│   ├── DEBUGGING.md            # VSCode 디버깅 가이드
│   ├── EVALUATION.md           # 벤치마크 평가 가이드
│   ├── EXPERIMENTS.md          # 실험 로그
│   ├── MUON.md                 # Muon optimizer 가이드
│   ├── PARAMETERS.md           # 파라미터 분석 가이드
│   └── SETUP.md                # 환경 세팅 가이드
│
└── outputs/                     # 학습 결과 (자동 생성)
    └── alpha_baseline_48L_*/
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

### 파라미터 계산
```bash
# 기본 요약
bash calc_params.sh

# 상세 분석
bash calc_params.sh configs/model/baseline_48L.yaml --detailed

# 또는 직접 실행
python calculate_parameters.py --config configs/model/baseline_48L.yaml --detailed
```

### 학습 시작
```bash
bash train.sh
```

### 디버깅 (VSCode)
```bash
# VSCode에서 F5 누르기
# 드롭다운에서 선택:
# 1. "Alpha: Single GPU Debug (Minimal)" ⭐ 권장
# 2. "Alpha: Multi-GPU Debug (8 GPUs)"
# 3. "Alpha: Model Init Only"
# 4. "Alpha: Data Loading Debug"

# 상세 가이드
cat docs/DEBUGGING.md
```

### TensorBoard
```bash
tensorboard --logdir outputs/alpha_*/tensorboard --port 6006
```

---

## Baseline 48L 모델

기본 실험 모델 설정:

| 항목 | 값 |
|------|-----|
| **레이어 수** | 48 |
| **Hidden Size** | 2048 |
| **FFN Hidden Size** | 8192 |
| **Attention Heads** | 32 |
| **Query Groups** | 2 (GQA) |
| **Experts** | 128 |
| **Router TopK** | 8 |
| **Hybrid Ratio** | 12.5% (6/48 layers) |

**아키텍처 특징**:
- 48 Megatron layers → 24 HuggingFace layers (2:1 mapping)
- Mamba (GatedDeltaNet) + Attention Hybrid
- MoE + Shared Expert 구조
- Top-8 expert 선택으로 효율적인 활성화

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

## 모델 평가

학습 완료 후 벤치마크로 모델 성능을 평가할 수 있습니다.

### 사전 준비: HuggingFace 토큰 설정

벤치마크 데이터셋을 다운로드하려면 HuggingFace 토큰이 필요합니다:

```bash
# 방법 1: 환경 변수로 설정 (권장)
export HF_TOKEN="hf_your_token_here"

# 방법 2: huggingface-cli 로그인
huggingface-cli login
```

토큰 생성: https://huggingface.co/settings/tokens

> **Note**: 토큰은 이미 `run_benchmarks.sh`에 설정되어 있으나, 보안을 위해 환경 변수 사용을 권장합니다.

### 빠른 시작: HuggingFace 모델로 평가

**변환된 HF 모델이 있는 경우**:
```bash
# 기본 벤치마크 세트 (추천)
bash scripts/run_benchmarks.sh \
  outputs/alpha_baseline_48L_*/hfmodel \
  "mmlu,hellaswag,arc_easy,arc_challenge,winogrande"

# 한 줄로 실행
bash scripts/run_benchmarks.sh outputs/alpha_baseline_48L_*/hfmodel
```

**지원 벤치마크 태스크**:
```bash
# 영어 표준 벤치마크
bash scripts/run_benchmarks.sh MODEL_PATH \
  "mmlu,hellaswag,arc_easy,arc_challenge,winogrande,boolq,piqa,social_iqa,openbookqa"

# 수학 & 추론
bash scripts/run_benchmarks.sh MODEL_PATH \
  "gsm8k,mathqa"

# 한국어 벤치마크
bash scripts/run_benchmarks.sh MODEL_PATH \
  "kmmlu"
```

### 지원 벤치마크

**영어 (English)**:
- **MMLU**: 대학 수준 지식 평가 (57 과목)
- **HellaSwag**: 상식 추론 (이야기 완성)
- **ARC-Easy/Challenge**: 초/중등 과학 질문
- **Winogrande**: 문장 이해 (대명사 해결)
- **BoolQ**: Yes/No 질문 답변
- **PIQA**: 물리적 상식 추론
- **Social IQA**: 사회적 상황 이해
- **OpenBookQA**: 과학 지식 추론
- **GSM8K**: 초등 수학 문제

**한국어 (Korean)**:
- **KMMLU**: Korean MMLU (한국형 지식 평가)

> **주의**: GPQA는 gated dataset으로 별도 접근 권한 필요

### 결과 해석

평가 완료 후 결과는 JSON 형식으로 출력됩니다:

```json
{
  "results": {
    "mmlu": {
      "acc": 0.45,
      "acc_stderr": 0.01
    },
    "hellaswag": {
      "acc_norm": 0.62
    }
  }
}
```

**참고 점수 (Qwen2.5 기준)**:
- MMLU: 70-85% (모델 크기에 따라)
- HellaSwag: 75-85%
- ARC-Challenge: 85-95%

### 고급: Megatron 체크포인트 변환 후 평가

**1. Megatron → HuggingFace 변환**
```bash
cd /home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor

bash scripts/alpha/run_8xH20.sh \
  baseline_48L \
  ../../examples/alpha/outputs/alpha_*/checkpoints/iter_0010000 \
  ../../examples/alpha/outputs/alpha_*/hfmodel \
  true true bf16
```

**2. 벤치마크 실행**
```bash
cd /home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/examples/alpha

bash scripts/run_benchmarks.sh \
  outputs/alpha_baseline_48L_*/hfmodel \
  "mmlu,hellaswag"
```

자세한 변환 가이드는 [**CONVERSION.md**](docs/CONVERSION.md)를 참고하세요.

### 성능 최적화

**배치 크기 조정**:
```bash
# 기본 (auto)
bash scripts/run_benchmarks.sh MODEL_PATH TASKS auto

# 수동 지정 (메모리 부족 시)
bash scripts/run_benchmarks.sh MODEL_PATH TASKS 1
```

**멀티 GPU 설정**:
`run_benchmarks.sh`는 기본적으로 8 GPU를 사용합니다. 수정하려면:
```bash
# scripts/run_benchmarks.sh 편집
accelerate launch --multi_gpu --num_processes=4 -m lm_eval ...
```

---

## 문서

- [**ARCHITECTURE.md**](docs/ARCHITECTURE.md): 모델 아키텍처 상세 설명
- [**CONVERSION.md**](docs/CONVERSION.md): HuggingFace 변환 및 평가 가이드
- [**DEBUGGING.md**](docs/DEBUGGING.md): VSCode 디버거 사용 가이드 🐛
- [**EVALUATION.md**](docs/EVALUATION.md): 벤치마크 평가 완전 가이드 ⭐
- [**EXPERIMENTS.md**](docs/EXPERIMENTS.md): 실험 로그 및 결과
- [**MUON.md**](docs/MUON.md): Muon optimizer 사용 가이드 🚀
- [**PARAMETERS.md**](docs/PARAMETERS.md): 파라미터 분석 가이드 📊
- [**SETUP.md**](docs/SETUP.md): 환경 세팅 상세 가이드

---

## 트러블슈팅

### 학습 관련

#### OOM (Out of Memory)
1. `configs/training/h100x8.yaml`에서 `micro_batch_size` 감소
2. `pipeline_parallel` 증가 (1 → 2 또는 4)
3. Activation checkpointing 강화

#### 낮은 Throughput
1. `micro_batch_size` 증가
2. `num_workers` 조정
3. Flash Attention 3 활성화 확인

#### 설정 오류
```bash
# YAML 문법 확인
python3 -c "import yaml; yaml.safe_load(open('configs/model/baseline_48L.yaml'))"

# 환경 검증
bash scripts/validate_environment.sh
```

### 평가 관련

#### HuggingFace Rate Limit
**에러**: `429 Client Error: Too Many Requests`

**해결**:
```bash
# HF 토큰 설정
export HF_TOKEN="hf_your_token_here"

# 또는 CLI 로그인
huggingface-cli login
```

#### Gated Dataset 접근 불가
**에러**: `Dataset 'xxx' is a gated dataset`

**해결**:
- 해당 데이터셋 페이지에서 접근 권한 요청
- 또는 벤치마크 실행 시 해당 태스크 제외

#### 모델 경로 에러
**에러**: `Unrecognized model in .../checkpoints`

**해결**: HuggingFace 포맷 모델 경로 사용
```bash
# 올바름 ✅
bash scripts/run_benchmarks.sh outputs/alpha_*/hfmodel

# 잘못됨 ❌
# bash scripts/run_benchmarks.sh outputs/alpha_*/checkpoints
```

자세한 내용은 [**EVALUATION.md**](docs/EVALUATION.md)를 참고하세요.

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
