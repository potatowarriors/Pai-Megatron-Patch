# Pai-Megatron-Patch 한국어 가이드

## 목차
- [소개](#소개)
- [핵심 개념](#핵심-개념)
- [시작하기](#시작하기)
- [주요 작업 흐름](#주요-작업-흐름)
- [모델 학습 실행하기](#모델-학습-실행하기)
- [디버깅 가이드](#디버깅-가이드)
- [지원 모델](#지원-모델)

## 소개

Pai-Megatron-Patch는 NVIDIA의 Megatron 프레임워크를 사용하여 대규모 언어 모델(LLM)과 비전-언어 모델(VLM)을 쉽게 학습할 수 있게 해주는 도구입니다.

### 이 도구를 사용하는 이유

- **높은 학습 효율**: 100억 파라미터 이상의 모델을 효율적으로 학습 가능
- **HuggingFace 호환**: HuggingFace 모델을 쉽게 변환하여 사용 가능
- **비침습적 설계**: Megatron 원본 코드를 수정하지 않고 패치 방식으로 확장
- **풍부한 예제**: 30개 이상의 모델에 대한 학습 예제 제공

### 핵심 기능

- ✅ Qwen, LLaMA, DeepSeek, Mistral 등 30+ 모델 지원
- ✅ HuggingFace ↔ Megatron 가중치 변환
- ✅ FP8, BF16, FP16 혼합 정밀도 학습
- ✅ Flash Attention 2/3 지원
- ✅ 강화학습(RLHF) 파이프라인 제공
- ✅ 분산 학습 최적화 (TP, PP, DP, SP, CP, EP)

## 핵심 개념

### 디렉토리 구조

```
Pai-Megatron-Patch/
├── megatron_patch/          # 핵심 라이브러리 (225개 파일)
│   ├── model/               # 모델 구현 (32개 이상의 모델)
│   ├── data/                # 데이터 로딩
│   ├── training.py          # 메인 학습 코드 (820줄)
│   └── arguments.py         # 인자 파싱 (573줄)
│
├── examples/                # 모델별 학습 스크립트 (104개 파일)
│   ├── qwen3/               # Qwen3 학습 예제
│   ├── llama3/              # LLaMA3 학습 예제
│   ├── deepseek_v3/         # DeepSeek-V3 학습 예제
│   └── ...                  # 기타 모델들
│
├── toolkits/                # 유틸리티 도구
│   ├── model_checkpoints_convertor/     # 모델 체크포인트 변환
│   ├── pretrain_data_preprocessing/     # 사전학습 데이터 전처리
│   ├── sft_data_preprocessing/          # 파인튜닝 데이터 전처리
│   └── auto_configurator/               # 자동 설정 도구
│
└── backends/                # Git 서브모듈
    ├── megatron/            # Megatron-LM 여러 버전
    │   ├── Megatron-LM-250908/  (최신)
    │   └── ...              # 8개 버전
    └── rl/                  # 강화학습 프레임워크
        ├── ChatLearn/       # Alibaba RL 프레임워크
        └── verl/            # 대안 RL 프레임워크
```

### 병렬화 전략 이해하기

대규모 모델 학습을 위한 핵심 개념들:

| 전략 | 약어 | 설명 | 언제 사용? |
|------|------|------|------------|
| **Tensor Parallel** | TP | 모델 가중치를 여러 GPU에 분할 | 대규모 모델 (TP=4, 8) |
| **Pipeline Parallel** | PP | 레이어를 여러 GPU에 분할 | 메모리 부족 시 (PP=2, 4, 8) |
| **Data Parallel** | DP | 데이터를 여러 GPU에 분할 | 자동 계산됨 |
| **Sequence Parallel** | SP | 시퀀스를 분할 (TP와 함께 사용) | TP 사용 시 활성화 |
| **Context Parallel** | CP | 초장문 컨텍스트 처리 | 32K+ 토큰 처리 시 |
| **Expert Parallel** | EP | MoE 모델의 expert 분할 | MoE 모델에만 사용 |

**예시**:
- 8개 GPU로 70B 모델 학습: `TP=8, PP=1`
- 32개 GPU로 70B 모델 학습: `TP=4, PP=2, DP=4`

## 시작하기

### 1. 설치

```bash
# 저장소 클론 (서브모듈 포함)
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch

# 서브모듈 업데이트
git submodule update --init --recursive

# 환경변수 설정 (사용할 Megatron 버전 선택)
export PYTHONPATH=$(pwd):$(pwd)/backends/megatron/Megatron-LM-250624:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

### 2. 의존성 설치

```bash
# PyTorch 설치 (2.0 이상)
pip install torch>=2.0

# 기타 의존성
pip install transformers datasets packaging modelscope
```

## 주요 작업 흐름

### 워크플로우 1: 사전학습 (Pre-training)

```bash
# Step 1: 데이터 전처리
cd toolkits/pretrain_data_preprocessing/
bash run_make_pretraining_dataset.sh \
  <vocab_파일> \
  <입력_jsonl> \
  <출력_prefix> \
  <워커_수>

# 결과: .bin과 .idx 파일 생성

# Step 2: HuggingFace 모델을 Megatron 형식으로 변환
cd ../../toolkits/model_checkpoints_convertor/qwen3/
bash hf2mcore_qwen3_convertor.sh \
  8B \                    # 모델 크기
  /path/to/hf/model \     # HuggingFace 모델 경로
  /path/to/output \       # 출력 경로
  4 \                     # TP (Tensor Parallel)
  1                       # PP (Pipeline Parallel)

# Step 3: 학습 실행
cd ../../examples/qwen3/
bash run_mcore_qwen3.sh \
  dsw \                   # 환경 (dsw=단일노드, dlc=다중노드)
  8B \                    # 모델 크기
  1 \                     # 배치 크기
  128 \                   # 글로벌 배치 크기
  1e-5 \                  # 학습률
  1e-6 \                  # 최소 학습률
  2048 \                  # 시퀀스 길이
  2048 \                  # 패딩 길이
  bf16 \                  # 정밀도 (bf16/fp16)
  4 \                     # TP
  1 \                     # PP
  1 \                     # CP
  1 \                     # ETP
  1 \                     # EP
  true \                  # SP 활성화
  true \                  # Distributed Optimizer
  true \                  # Flash Attention
  false \                 # SFT 모드 (사전학습이므로 false)
  1 \                     # Activation Checkpointing
  false \                 # Optimizer Offload
  2000 \                  # 체크포인트 저장 간격
  /data/train.bin \       # 학습 데이터
  /data/valid.bin \       # 검증 데이터
  "" \                    # 사전학습 체크포인트 (없으면 빈 문자열)
  100000000 \             # 학습 토큰 수
  1000000 \               # Warmup 토큰 수
  /outputs                # 출력 디렉토리
```

### 워크플로우 2: 파인튜닝 (Supervised Fine-Tuning)

```bash
# Step 1: SFT 데이터 전처리
cd toolkits/sft_data_preprocessing/
bash convert_sft_dataset.sh \
  <입력_jsonl> \
  <출력_디렉토리> \
  <토크나이저_경로>

# JSONL 형식 예시:
# {"messages": [
#   {"role": "user", "content": "안녕하세요"},
#   {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}
# ]}

# Step 2: 학습 실행 (사전학습과 거의 동일, SFT=true로 설정)
cd ../../examples/qwen3/
bash run_mcore_qwen3.sh \
  dsw 8B 1 128 1e-5 1e-6 2048 2048 bf16 \
  4 1 1 1 1 true true false \
  true \                  # SFT 모드 활성화!
  1 false 2000 \
  /data/sft_train.json \  # SFT 데이터 (JSONL)
  /data/sft_valid.json \
  /path/to/pretrain/ckpt \  # 사전학습된 체크포인트
  "" "" \                 # 토큰 수는 SFT에서 사용 안 함
  /outputs
```

### 워크플로우 3: 모델 평가

```bash
cd examples/qwen3/

# LM Evaluation Harness를 사용한 벤치마크 평가
python evaluate_megatron_qwen3.py \
  --model-path /path/to/checkpoint \
  --tasks mmlu,hellaswag,arc_challenge \
  --tp 4 \
  --pp 1

# 텍스트 생성 테스트
python generate_text.py \
  --checkpoint /path/to/checkpoint \
  --prompt "안녕하세요, 저는" \
  --max-tokens 100
```

### 워크플로우 4: Megatron → HuggingFace 변환

```bash
# 학습이 끝난 후 HuggingFace 형식으로 변환하여 배포
cd toolkits/model_checkpoints_convertor/qwen3/
bash mcore2hf_qwen3_convertor.sh \
  8B \                       # 모델 크기
  /path/to/megatron/ckpt \   # Megatron 체크포인트
  /path/to/hf/output \       # HuggingFace 출력 경로
  4 \                        # TP (학습 시와 동일하게)
  1                          # PP (학습 시와 동일하게)

# 이제 HuggingFace Transformers로 모델 사용 가능!
```

### 워크플로우 5: 강화학습 (RLHF)

```bash
# 준비물:
# 1. SFT로 학습된 모델
# 2. Reward 모델
# 3. RL 데이터셋

# ChatLearn을 사용한 GRPO 학습
cd examples/qwen3/
bash run_mcore_qwen3_chatlearn.sh \
  <환경> <모델크기> <배치크기> ... \
  <SFT_모델_경로> \
  <Reward_모델_경로> \
  <RL_데이터_경로>

# 또는 Verl 사용
bash run_mcore_qwen3_verl.sh ...
```

## 모델 학습 실행하기

### 간단한 예제: Qwen3 8B 학습

```bash
#!/bin/bash
# 8개 GPU에서 Qwen3 8B 파인튜닝

cd examples/qwen3/

# 모든 파라미터를 한 줄로
bash run_mcore_qwen3.sh \
  dsw \              # 단일 노드
  8B \               # 8B 모델
  1 \                # 마이크로 배치=1
  128 \              # 글로벌 배치=128 (gradient accumulation 128회)
  1e-5 \             # 학습률
  1e-6 \             # 최소 학습률
  2048 \             # 시퀀스 길이
  2048 \             # 패딩 길이
  bf16 \             # BFloat16 정밀도
  4 \                # TP=4 (8 GPU / 4 TP = 2 DP)
  1 \                # PP=1
  1 \                # CP=1
  1 \                # ETP=1
  1 \                # EP=1
  true \             # Sequence Parallel ON
  true \             # Distributed Optimizer ON
  true \             # Flash Attention ON
  true \             # SFT 모드 ON
  1 \                # Activation Checkpointing level 1
  false \            # Optimizer Offload OFF
  500 \              # 500 iteration마다 체크포인트 저장
  /data/train.json \ # 학습 데이터
  /data/valid.json \ # 검증 데이터
  /data/ckpt/iter_0001000 \ # 초기 체크포인트
  "" "" \            # 토큰 수 (SFT에서 미사용)
  /outputs/qwen3-8b-sft     # 출력 디렉토리

# 학습 진행 상황 모니터링
tensorboard --logdir /outputs/qwen3-8b-sft/tensorboard/
```

### 주요 파라미터 설명

| 파라미터 | 설명 | 일반적인 값 |
|---------|------|------------|
| `ENV` | 실행 환경 | `dsw` (단일노드) 또는 `dlc` (다중노드) |
| `MODEL_SIZE` | 모델 크기 | `0.6B`, `1.7B`, `4B`, `8B`, `14B`, `32B`, `72B` |
| `BATCH_SIZE` | GPU당 마이크로 배치 크기 | `1`, `2`, `4` (메모리에 따라) |
| `GLOBAL_BATCH_SIZE` | 총 배치 크기 | `128`, `256`, `512` |
| `LR` | 학습률 | 사전학습: `3e-4`, SFT: `1e-5` |
| `SEQ_LEN` | 시퀀스 길이 | `2048`, `4096`, `8192` |
| `PR` | 정밀도 | `bf16` (권장), `fp16`, `fp32` |
| `TP` | Tensor Parallel 크기 | `1`, `2`, `4`, `8` |
| `PP` | Pipeline Parallel 크기 | `1`, `2`, `4` |
| `SP` | Sequence Parallel | `true` (TP>1일 때 권장) |
| `FL` | Flash Attention | `true` (권장, 더 빠름) |
| `SFT` | SFT 모드 | `true` (파인튜닝), `false` (사전학습) |
| `AC` | Activation Checkpointing | `0` (없음), `1` (레이어), `2` (서브레이어) |

## 디버깅 가이드

### 문제 1: GPU 메모리 부족 (OOM)

**증상**: `RuntimeError: CUDA out of memory`

**해결방법**:
```bash
# 방법 1: 마이크로 배치 크기 줄이기
BATCH_SIZE=1  # 2 → 1로 변경

# 방법 2: Gradient Accumulation 늘리기
GLOBAL_BATCH_SIZE=256  # 128 → 256 (동일 효과, 더 느림)

# 방법 3: Activation Checkpointing 활성화
AC=2  # 0 → 2로 변경 (메모리 절약, 약간 느려짐)

# 방법 4: Optimizer Offloading (CPU로 옵티마이저 이동)
OPTIMIZER_OFFLOAD=true

# 방법 5: Pipeline Parallel 증가
PP=2  # 1 → 2로 변경 (레이어를 더 많은 GPU에 분산)
```

### 문제 2: 학습이 너무 느림

**증상**: 초당 처리 샘플 수가 낮음

**해결방법**:
```bash
# 방법 1: Flash Attention 활성화
FL=true
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0

# 방법 2: Sequence Parallel 활성화 (TP 사용 시)
SP=true

# 방법 3: 환경변수 설정
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 방법 4: 통신-연산 오버랩 활성화
# run_mcore_*.sh 스크립트에 추가:
# --overlap-grad-reduce --overlap-param-gather
```

### 문제 3: 체크포인트 로딩 실패

**증상**: `Checkpoint loading error` 또는 shape mismatch

**해결방법**:
```bash
# 원인 1: TP/PP 값이 학습 시와 다름
# ✅ 해결: 체크포인트 학습 시와 동일한 TP/PP 사용

# 원인 2: Megatron 버전 불일치
# ✅ 해결: PYTHONPATH에서 동일한 Megatron 버전 사용
export PYTHONPATH=.../Megatron-LM-250624:$PYTHONPATH  # 학습 시와 동일 버전

# 원인 3: torch_dist 체크포인트 사용
# ✅ 해결: --use-dist-ckpt 플래그 추가
```

### 문제 4: 데이터 로딩 오류

**증상**: `FileNotFoundError` 또는 데이터셋 오류

**해결방법**:
```bash
# 원인 1: .bin과 .idx 파일이 함께 없음
# ✅ 해결: 두 파일 모두 있는지 확인
ls /data/train.bin
ls /data/train.idx

# 원인 2: 경로에 .bin 확장자 포함
# ❌ 잘못: DATASET_PATH=/data/train.bin
# ✅ 올바름: DATASET_PATH=/data/train

# 원인 3: 토크나이저 불일치
# ✅ 해결: 데이터 전처리 시 사용한 토크나이저와 동일한 것 사용
```

### 문제 5: 학습이 발산함 (Loss가 NaN)

**해결방법**:
```bash
# 방법 1: 학습률 낮추기
LR=1e-6  # 1e-5 → 1e-6

# 방법 2: Gradient Clipping 확인
# 스크립트에 --clip-grad 1.0 있는지 확인

# 방법 3: 혼합 정밀도 변경
PR=fp32  # bf16 → fp32 (더 안정적이지만 느림)

# 방법 4: Warmup 증가
WARMUP_TOKENS=10000000  # 1000000 → 10000000
```

### 로그 확인 방법

```bash
# TensorBoard로 학습 모니터링
tensorboard --logdir /outputs/qwen3-8b/tensorboard/ --port 6006

# 체크포인트 확인
ls -lh /outputs/qwen3-8b/checkpoints/

# 로그 파일 확인
tail -f /outputs/qwen3-8b/logs/train.log

# GPU 사용률 모니터링
watch -n 1 nvidia-smi
```

## 지원 모델

### 밀집 모델 (Dense Models)

| 모델 | 크기 | 예제 경로 | 특징 |
|------|------|-----------|------|
| **Qwen3** | 0.6B ~ 235B | `examples/qwen3/` | 최신 Qwen, GQA, 긴 컨텍스트 |
| **Qwen2.5** | 0.5B ~ 72B | `examples/qwen2_5/` | 다국어, 코드 생성 |
| **LLaMA3.1** | 8B ~ 70B | `examples/llama3_1/` | 128K 컨텍스트, RoPE |
| **LLaMA3** | 8B ~ 70B | `examples/llama3/` | Meta의 오픈소스 모델 |
| **DeepSeek** | 7B ~ 67B | `examples/deepseek/` | 코드 특화 |
| **Mistral** | 7B | `examples/mistral/` | 효율적인 7B 모델 |
| **Baichuan2** | 7B ~ 13B | `examples/baichuan2/` | 중국어 특화 |

### 희소 모델 (MoE - Mixture of Experts)

| 모델 | 전체 파라미터 | 활성 파라미터 | 예제 경로 |
|------|--------------|--------------|-----------|
| **Qwen3-MoE** | 80B (A3B) | ~8B | `examples/qwen3/` |
| **Qwen2-MoE** | 57B (A14B) | ~14B | `examples/qwen2_moe/` |
| **DeepSeek-V3** | 671B | ~37B | `examples/deepseek_v3/` |
| **Mixtral** | 8x7B (47B) | ~13B | `examples/mixtral/` |

### 비전-언어 모델 (VLM - Vision Language Models)

| 모델 | 예제 경로 | 특징 |
|------|-----------|------|
| **Qwen3-VL** | `examples/qwen3_vl/` | 최신 멀티모달, 비디오 지원 |
| **Qwen2.5-VL** | `examples/qwen2_5_vl/` | 이미지+텍스트 |
| **Qwen2-VL** | `examples/qwen2_vl/` | 이전 버전 |
| **LLaVA** | `examples/llava/` | 오픈소스 VLM |

### 특수 모델

| 모델 | 용도 | 예제 경로 |
|------|------|-----------|
| **QwQ** | 추론 특화 | `examples/qwq/` |
| **Moonlight-16B** | Kimi 모델 | `examples/moonlight/` |
| **CodeLlama** | 코드 생성 | `examples/codellama/` |

## 빠른 참조

### 일반적인 명령어 모음

```bash
# 1. 저장소 설정
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
export PYTHONPATH=$(pwd):$(pwd)/backends/megatron/Megatron-LM-250624:$PYTHONPATH

# 2. 데이터 전처리 (사전학습)
cd toolkits/pretrain_data_preprocessing/
bash run_make_pretraining_dataset.sh vocab.json input.jsonl output_prefix 32

# 3. 데이터 전처리 (SFT)
cd toolkits/sft_data_preprocessing/
bash convert_sft_dataset.sh input.jsonl output_dir tokenizer_path

# 4. HF → Megatron 변환
cd toolkits/model_checkpoints_convertor/qwen3/
bash hf2mcore_qwen3_convertor.sh 8B /hf/path /megatron/path 4 1

# 5. 학습 실행 (간단한 버전)
cd examples/qwen3/
bash run_mcore_qwen3.sh dsw 8B 1 128 1e-5 1e-6 2048 2048 bf16 \
  4 1 1 1 1 true true false true 1 false 500 \
  /data/train.json /data/valid.json /ckpt "" "" /outputs

# 6. Megatron → HF 변환
cd toolkits/model_checkpoints_convertor/qwen3/
bash mcore2hf_qwen3_convertor.sh 8B /megatron/path /hf/path 4 1

# 7. 평가
cd examples/qwen3/
python evaluate_megatron_qwen3.py --model-path /ckpt --tasks mmlu --tp 4

# 8. TensorBoard
tensorboard --logdir /outputs/tensorboard/
```

### 추천 하드웨어 구성

| 모델 크기 | GPU 개수 | TP | PP | 메모리 (GPU당) | 예상 학습 시간 |
|----------|---------|----|----|---------------|--------------|
| 1B | 1-2 | 1 | 1 | 24GB | 빠름 |
| 7-8B | 4-8 | 4 | 1 | 40GB+ | 보통 |
| 14B | 8 | 4-8 | 1 | 40GB+ | 보통 |
| 30-32B | 8-16 | 4-8 | 2 | 80GB | 느림 |
| 70B | 16-32 | 8 | 2-4 | 80GB | 매우 느림 |
| 100B+ | 32+ | 8 | 4-8 | 80GB | 매우 매우 느림 |

### 추천 설정 (모델 크기별)

```bash
# Qwen3-8B (8 x H20 GPU, 96GB)
bash run_mcore_qwen3.sh dsw 8B 1 128 1e-5 1e-6 2048 2048 bf16 \
  4 1 1 1 1 true true false true 1 false 500 ...

# Qwen3-14B (8 x H20 GPU, 96GB)
bash run_mcore_qwen3.sh dsw 14B 1 128 1e-5 1e-6 2048 2048 bf16 \
  8 1 1 1 1 true true false true 2 false 500 ...

# Qwen3-32B (16 x A100 GPU, 80GB)
bash run_mcore_qwen3.sh dlc 32B 1 256 1e-5 1e-6 2048 2048 bf16 \
  8 2 1 1 1 true true false true 2 true 500 ...

# Qwen3-72B (32 x A100 GPU, 80GB)
bash run_mcore_qwen3.sh dlc 72B 1 512 1e-5 1e-6 2048 2048 bf16 \
  8 4 1 1 1 true true false true 2 true 500 ...
```

## Qwen3-Next Scratch 학습 완전 가이드

Qwen3-Next 모델을 처음부터 (from scratch) 학습하는 전체 워크플로우입니다.

### 1단계: 대용량 데이터 전처리 (Multi-TB)

Qwen3-Next를 scratch부터 학습하려면 수 TB의 데이터가 필요합니다. 최적화된 전처리 파이프라인:

```bash
cd toolkits/pretrain_data_preprocessing/

# Step 1: Arrow → JSONL 변환 (DCLM, FineWeb 등)
# 4.5TB 데이터를 224 CPU 코어로 약 8분 처리
python convert_arrow_to_jsonl_v2.py \
  --input-dir /data/dclm/arrow/ \
  --output-file /data/dclm/dclm_full.jsonl \
  --text-column text \
  --workers 224

# Step 2 (선택): 테스트용 서브셋 생성
# 전체 데이터의 1%만 무작위 샘플링 (재현 가능)
python streaming_random_sample.py \
  --input /data/dclm/dclm_full.jsonl \
  --output /data/dclm/dclm_1pct.jsonl \
  --sample-rate 0.01 \
  --seed 42 \
  --method reservoir

# Step 3: Megatron 바이너리 포맷 변환
# 서브셋으로 테스트 (빠름)
bash preprocess_kormo_subset.sh 1

# 또는 전체 데이터 처리 (느림, 수 시간 소요)
python preprocess_data.py \
  --input /data/dclm/dclm_full.jsonl \
  --output-prefix /data/mmap/dclm \
  --dataset-impl mmap \
  --patch-tokenizer-type Qwen2Tokenizer \
  --load /path/to/Qwen3-Next-tokenizer \
  --workers 64 \
  --append-eod \
  --extra-vocab-size 0
```

**성능 지표**:
- Arrow → JSONL: 4.5TB를 8분 (기존 대비 40배 속도 향상)
- 1% 샘플링: 수백 GB를 수 분 (메모리 효율적 reservoir sampling)
- 바이너리 변환: 수 시간 (CPU 코어 수에 비례)

### 2단계: 모델 초기화 (선택 사항)

**옵션 A: 랜덤 초기화로 완전 scratch 학습**
```bash
# 학습 스크립트에서 PRETRAIN_CHECKPOINT_PATH를 빈 문자열로 설정
PRETRAIN_CHECKPOINT_PATH=""
```

**옵션 B: 기존 체크포인트에서 계속 학습**
```bash
# HuggingFace → Megatron 변환
cd toolkits/model_checkpoints_convertor/qwen3_next/
bash hf2mcore_qwen3_next_convertor.sh \
  A3B \  # 80B-A3B 모델
  /path/to/Qwen3-Next-80B-A3B \
  /path/to/mcore/qwen3next \
  8 1  # TP=8, PP=1 (8 GPU 기준)
```

### 3단계: 학습 실행

**소규모 테스트 (1% 서브셋, 8 GPU):**
```bash
cd examples/qwen3_next/

bash run_mcore_qwen3_next.sh \
  dsw \              # 단일 노드
  A3B \              # 80B-A3B 모델
  1 \                # 마이크로 배치 = 1
  128 \              # 글로벌 배치 = 128
  3e-4 \             # 학습률 (scratch: 높게, continue: 낮게)
  3e-5 \             # 최소 학습률
  2048 \             # 시퀀스 길이
  2048 \             # 패딩 길이
  bf16 \             # BFloat16
  8 \                # TP=8 (8 GPU에서 MoE 모델)
  1 \                # PP=1
  1 \                # CP=1
  1 \                # ETP=1
  1 \                # EP=1
  true \             # Sequence Parallel
  true \             # Distributed Optimizer
  true \             # Flash Attention 3
  false \            # SFT 모드 OFF (사전학습)
  1 \                # Activation Checkpointing
  false \            # Optimizer Offload
  500 \              # 500 iter마다 체크포인트 저장
  /data/mmap/dclm_1pct_content_document \  # 학습 데이터
  /data/mmap/dclm_1pct_content_document \  # 검증 데이터
  "" \               # 초기 체크포인트 (빈 문자열 = scratch)
  100000000 \        # 학습 토큰 수 (1% 테스트: ~1B 토큰)
  1000000 \          # Warmup 토큰
  /outputs/qwen3next_test
```

**대규모 학습 (전체 데이터, 다중 노드):**
```bash
# 64 GPU (8노드 x 8GPU) 예시
bash run_mcore_qwen3_next.sh \
  dlc \              # 다중 노드
  A3B \
  1 \
  1024 \             # 글로벌 배치 = 1024 (많은 GPU)
  3e-4 \
  3e-5 \
  2048 \
  2048 \
  bf16 \
  8 \                # TP=8
  1 \                # PP=1
  1 \
  1 \
  1 \
  true \
  true \
  true \
  false \
  2 \                # AC=2 (메모리 절약)
  false \
  2000 \
  /data/mmap/dclm_full_content_document \
  /data/mmap/valid_content_document \
  "" \
  1000000000000 \    # 1T 토큰 학습
  10000000000 \      # 10B 토큰 warmup
  /outputs/qwen3next_1T
```

### 4단계: 학습 모니터링

```bash
# TensorBoard로 실시간 모니터링
tensorboard --logdir /outputs/qwen3next_test/tensorboard/ --port 6006

# GPU 사용률 확인
watch -n 1 nvidia-smi

# 로그 확인
tail -f /outputs/qwen3next_test/logs/train.log

# 체크포인트 확인
ls -lh /outputs/qwen3next_test/checkpoints/
```

### 5단계: 학습 완료 후 평가 & 배포

```bash
# Megatron → HuggingFace 변환
cd toolkits/model_checkpoints_convertor/qwen3_next/
bash mcore2hf_qwen3_next_convertor.sh \
  A3B \
  /outputs/qwen3next_test/checkpoints/iter_0010000 \
  /outputs/hf_model \
  8 1

# 벤치마크 평가 (MMLU, HellaSwag 등)
cd examples/qwen3_next/
python evaluate_megatron_qwen3_next.py \
  --model-path /outputs/hf_model \
  --tasks mmlu,hellaswag,arc_challenge \
  --tp 8 --pp 1

# HuggingFace Hub에 업로드
huggingface-cli upload my-org/my-qwen3-next /outputs/hf_model
```

### 6단계: 준비 상태 체크

학습 전 모든 것이 준비되었는지 확인:

```bash
cd examples/qwen3_next/
bash final_check.sh
```

출력 예시:
```
[1] 소프트웨어 환경 ✅
  ✓ Flash Attention 3: 사용 가능
  ✓ Transformer Engine PyTorch: 사용 가능
  ✓ Megatron-Core (250908): 사용 가능
  ✓ GPU: 8개 사용 가능

[2] 데이터 준비 상태
  ✓ Megatron 변환 모델: 156G
  ✓ 사전학습 데이터: 23G

[3] 학습 실행 가능 여부
  🚀 준비 완료! 학습을 시작할 수 있습니다.
```

### 주요 파라미터 최적화 팁

| 파라미터 | Scratch 학습 | Continue 학습 |
|---------|-------------|--------------|
| **학습률 (LR)** | `3e-4` ~ `1e-4` | `1e-5` ~ `3e-6` |
| **Warmup 토큰** | 전체의 1-5% | 전체의 0.1-1% |
| **배치 크기** | 크게 (512-2048) | 중간 (128-512) |
| **체크포인트 빈도** | 자주 (1000-2000 iter) | 보통 (5000-10000 iter) |

### 문제 해결

**메모리 부족 (OOM)**:
```bash
# AC=2로 변경 (activation checkpointing 강화)
AC=2

# 마이크로 배치 줄이기
BATCH_SIZE=1

# Optimizer Offload 활성화
OPTIMIZER_OFFLOAD=true
```

**학습이 너무 느림**:
```bash
# Flash Attention 3 활성화 확인
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0

# Sequence Parallel 활성화
SP=true

# 통신-연산 오버랩 활성화 (스크립트 수정)
--overlap-grad-reduce --overlap-param-gather
```

**Loss가 발산 (NaN)**:
```bash
# 학습률 낮추기
LR=1e-4  # 3e-4 → 1e-4

# Warmup 증가
WARMUP_TOKENS=20000000  # 10B → 20B

# Gradient Clipping 확인 (스크립트에 있어야 함)
--clip-grad 1.0
```

## 추가 자료

- **원본 README**: [README.md](README.md) (중국어)
- **영문 가이드**: [CLAUDE.md](CLAUDE.md) (기술 세부사항)
- **Megatron-LM 공식 문서**: https://github.com/NVIDIA/Megatron-LM
- **HuggingFace 모델 허브**: https://huggingface.co/Qwen
- **Qwen3-Next 예제**: [examples/qwen3_next/README.md](examples/qwen3_next/README.md)

## 문의 및 지원

- **GitHub Issues**: https://github.com/alibaba/Pai-Megatron-Patch/issues
- **DingTalk 그룹**: README.md의 QR 코드 참조

---

**팁**: 처음 시작할 때는 작은 모델(1B-7B)로 시작하여 전체 파이프라인을 이해한 후, 큰 모델로 확장하는 것을 추천합니다!
