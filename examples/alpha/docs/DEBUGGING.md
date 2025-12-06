# Alpha Model Debugging Guide

VSCode Python Debugger를 사용한 Alpha 모델 디버깅 완전 가이드

---

## 빠른 시작

### 1. VSCode 디버거 실행

1. VSCode에서 `F5` 누르기 (또는 Run → Start Debugging)
2. 드롭다운에서 원하는 시나리오 선택
3. 디버깅 시작!

### 2. 추천 시나리오 (첫 디버깅)

**"Alpha: Single GPU Debug (Minimal)"** ⭐ 선택
- 가장 빠르고 간단
- 단일 GPU로 전체 학습 파이프라인 확인
- ~30초 안에 시작

---

## 디버그 설정 (4가지 시나리오)

### ⭐ Scenario 1: Single GPU Debug (Minimal)

**용도**: 모델 구조, forward/backward pass, 데이터 로딩 디버깅

**설정**:
- GPU: 1개 (CUDA_VISIBLE_DEVICES=0)
- Expert Parallel: EP=1 (모든 256 experts를 1개 GPU에)
- Batch Size: Micro=1, Global=1
- Iterations: 100 (빠른 테스트)

**메모리**: ~20GB GPU RAM

**시작 시간**: ~30초

**추천 Breakpoint**:
```python
# 모델 생성
examples/alpha/pretrain_alpha.py:69  # mamba_builder function

# Layer spec 정의
megatron_patch/model/qwen3_next/layer_specs.py:50  # get_qwen3_next_layer_spec

# 데이터 로딩
megatron_patch/data/pretrain_dataset.py:100  # GPTDataset __init__

# Forward pass
megatron_patch/template/helper.py:30  # forward_step
```

**사용 예시**:
```python
# Breakpoint 설정: pretrain_alpha.py:69
# F5로 디버깅 시작 → Breakpoint에서 멈춤
# 변수 확인:
print(config)  # Transformer config
print(args.num_layers)  # 24
print(args.num_experts)  # 256
```

---

### 🔀 Scenario 2: Multi-GPU Debug (8 GPUs, torchrun)

**용도**: 분산 학습, Expert Parallelism 디버깅

**설정**:
- GPU: 8개
- Expert Parallel: EP=8 (각 GPU에 32 experts)
- Batch Size: Micro=2, Global=256
- Iterations: 100
- Torchrun 사용

**메모리**: ~15GB/GPU

**시작 시간**: ~1분

**주의사항**:
- Rank 0 프로세스만 디버거 attach
- 다른 rank의 로그는 터미널에서 확인
- NCCL 통신 디버깅 가능

**사용 예시**:
```python
# Breakpoint: megatron/core/parallel_state.py
import torch.distributed as dist
print(f"Rank: {dist.get_rank()}")  # Current rank
print(f"World Size: {dist.get_world_size()}")  # 8
print(f"EP Group: {get_expert_parallel_group()}")
```

---

### 📊 Scenario 3: Model Init Only (Parameter Calculator)

**용도**: 파라미터 계산 로직 검증, 설정 파일 테스트

**설정**:
- Program: `calculate_parameters.py`
- GPU 불필요 (CPU만 사용)
- YAML 파싱 및 파라미터 계산만 수행

**시작 시간**: ~5초

**추천 Breakpoint**:
```python
# examples/alpha/calculate_parameters.py:377
calculator = AlphaParameterCalculator(config)
results = calculator.calculate()

# 각 계산 함수 내부
calculate_parameters.py:90  # calc_embedding_params
calculate_parameters.py:103  # calc_mamba_layer_params
calculate_parameters.py:155  # calc_attention_layer_params
calculate_parameters.py:188  # calc_moe_layer_params
```

**사용 예시**:
```python
# Breakpoint: calculate_parameters.py:377
print(config['model']['moe']['num_experts'])  # 256
print(config['model']['moe']['moe_ffn_hidden_size'])  # 768

# Step into calc_moe_layer_params
# 각 expert의 파라미터 수 확인
expert_params = (gate + up + down) * num_experts
print(f"Total MoE params: {expert_params:,}")
```

---

### 🗂️ Scenario 4: Data Loading Debug (Tiny Model)

**용도**: Tokenizer, Dataset, DataLoader 파이프라인 디버깅

**설정**:
- 매우 작은 모델 (2 layers, 256 hidden, 8 experts)
- Sequence length: 512 (짧음)
- Workers: 2
- Iterations: 10

**메모리**: ~5GB GPU RAM

**시작 시간**: ~10초

**추천 Breakpoint**:
```python
# 데이터셋 초기화
megatron_patch/data/pretrain_dataset.py:100  # GPTDataset.__init__
megatron_patch/data/pretrain_dataset.py:150  # __getitem__

# Tokenizer
megatron_patch/tokenizer/tokenizer.py:50  # Tokenizer loading

# DataLoader
megatron_patch/data/data_utils.py:80  # build_train_valid_test_data_iterators
```

**사용 예시**:
```python
# Breakpoint: pretrain_dataset.py:150 (__getitem__)
def __getitem__(self, idx):
    # 여기서 실제 데이터 확인
    sample = self.dataset[idx]
    print(f"Sample {idx}: {sample[:10]}")  # 첫 10 토큰
    print(f"Sample shape: {sample.shape}")  # [seq_len]
    print(f"Token range: {sample.min()} - {sample.max()}")
    return sample
```

---

## 주요 Breakpoint 위치

### 1. 모델 생성 (Model Construction)

| 파일 | 라인 | 설명 |
|------|------|------|
| `examples/alpha/pretrain_alpha.py` | 69 | `mamba_builder` 함수 시작 |
| `megatron_patch/model/qwen3_next/layer_specs.py` | 50 | Layer spec 생성 |
| `backends/.../megatron/core/models/mamba/mamba_model.py` | 100 | MambaModel `__init__` |
| `megatron_patch/model/qwen3_next/gated_deltanet.py` | 149 | Mamba layer 초기화 |

### 2. 데이터 로딩 (Data Loading)

| 파일 | 라인 | 설명 |
|------|------|------|
| `megatron_patch/data/pretrain_dataset.py` | 100 | GPTDataset 초기화 |
| `megatron_patch/data/pretrain_dataset.py` | 150 | `__getitem__` (샘플 가져오기) |
| `examples/alpha/pretrain_alpha.py` | 118 | Data provider 설정 |

### 3. 학습 루프 (Training Loop)

| 파일 | 라인 | 설명 |
|------|------|------|
| `megatron_patch/training.py` | 300 | Main training loop |
| `megatron_patch/template/helper.py` | 30 | Forward step |
| `backends/.../megatron/training/training.py` | 500 | Backward pass |

### 4. 인자 파싱 (Argument Parsing)

| 파일 | 라인 | 설명 |
|------|------|------|
| `megatron_patch/arguments.py` | 50 | `get_patch_args` |
| `backends/.../megatron/training/arguments.py` | 100 | Megatron args 파싱 |

---

## 디버깅 팁

### 1. justMyCode = false

모든 설정에 `"justMyCode": false`가 설정되어 있어 Megatron-LM 내부 코드까지 step into 가능합니다.

```json
"justMyCode": false  // Megatron 코드 내부 진입 가능
```

### 2. 조건부 Breakpoint

특정 조건에서만 멈추고 싶을 때:

```python
# VSCode에서 Breakpoint 우클릭 → Edit Breakpoint → Condition
# 예시:
rank == 0  # Rank 0에서만 멈춤
idx > 100  # 100번째 iteration 이후만
```

### 3. Logpoint

Breakpoint 대신 로그만 출력:

```python
# Breakpoint 우클릭 → Edit Breakpoint → Logpoint
# 예시:
"Iteration {iteration}, Loss: {loss}"
```

### 4. 변수 감시 (Watch)

자주 확인하는 변수를 Watch에 추가:

```python
# Debug sidebar → Watch → Add Expression
args.num_experts  # 256
config.hidden_size  # 2048
torch.cuda.memory_allocated()  # GPU 메모리
```

### 5. 호출 스택 (Call Stack)

현재 함수 호출 경로 확인:
- Debug sidebar → Call Stack
- 각 프레임 클릭하여 상위 함수로 이동

---

## 환경 변수 설명

모든 디버그 설정에 포함된 주요 환경 변수:

```bash
# PYTHONPATH (Megatron 경로)
PYTHONPATH=${workspaceFolder}:${workspaceFolder}/backends/megatron/Megatron-LM-250908

# CUDA 설정
CUDA_DEVICE_MAX_CONNECTIONS=1  # 단일 CUDA stream
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 메모리 최적화

# TransformerEngine 설정
NVTE_FLASH_ATTN=1  # Flash Attention 사용
NVTE_FUSED_ATTN=0  # Fused Attention 비활성화
NVTE_NORM_FWD_USE_CUDNN=1  # cuDNN normalization

# Threading 설정
OMP_NUM_THREADS=8  # OpenMP threads
NUMEXPR_MAX_THREADS=256  # NumExpr threads
```

---

## 트러블슈팅

### 문제 1: OOM (Out of Memory)

**증상**: CUDA out of memory 에러

**해결**:
1. Scenario 4 (Tiny Model) 사용
2. `--micro-batch-size` 감소 (1 → 1 이미 최소)
3. `--seq-length` 감소 (4096 → 2048 → 1024)
4. `--num-experts` 감소 (256 → 128 → 64)

**launch.json 수정**:
```json
"--seq-length", "1024",  // 4096 → 1024
"--num-experts", "128",  // 256 → 128
```

### 문제 2: 데이터 경로 오류

**증상**: FileNotFoundError: data path not found

**해결**:
```bash
# 데이터 전처리 실행
cd toolkits/pretrain_data_preprocessing/
bash preprocess_kormo_subset.sh 1

# 경로 확인
ls /home/work/Datasets/KORMo_processed/mmap/qwen3_1pct/
```

### 문제 3: Tokenizer 로드 실패

**증상**: Tokenizer not found

**해결**:
```bash
# Tokenizer 경로 확인
ls /home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/models/Qwen3-Next-tokenizer/

# launch.json 수정
"--load", "/path/to/your/tokenizer"
```

### 문제 4: Multi-GPU 디버깅 실패

**증상**: torchrun이 제대로 시작되지 않음

**해결**:
1. Single GPU 시나리오로 먼저 테스트
2. NCCL 버전 확인: `python -c "import torch; print(torch.cuda.nccl.version())"`
3. 포트 충돌 확인: `lsof -i :29500`

### 문제 5: Breakpoint가 작동하지 않음

**증상**: Breakpoint가 회색으로 표시되고 멈추지 않음

**해결**:
1. `justMyCode: false` 확인
2. PYTHONPATH 정확한지 확인
3. 파일 경로가 절대 경로인지 확인

---

## 성능 최적화

### 빠른 iteration을 위한 설정

```json
// launch.json에서 다음 값 조정
"--train-iters", "10",        // 100 → 10
"--seq-length", "512",        // 4096 → 512
"--num-workers", "2",         // 4 → 2
"--eval-interval", "5",       // 50 → 5
"--log-interval", "1"         // 매 iteration마다 로그
```

### 메모리 절약 설정

```json
"--num-layers", "2",          // 24 → 2
"--hidden-size", "512",       // 2048 → 512
"--num-experts", "32",        // 256 → 32
"--moe-router-topk", "2",     // 8 → 2
```

---

## 유용한 단축키

| 단축키 | 기능 |
|--------|------|
| `F5` | 디버깅 시작 / 계속 |
| `F9` | Breakpoint 토글 |
| `F10` | Step Over (다음 줄) |
| `F11` | Step Into (함수 내부) |
| `Shift+F11` | Step Out (함수 밖으로) |
| `Ctrl+Shift+F5` | 디버깅 재시작 |
| `Shift+F5` | 디버깅 중지 |

---

## 추가 리소스

- **VSCode Python 디버깅 가이드**: https://code.visualstudio.com/docs/python/debugging
- **Alpha 프로젝트 README**: [../README.md](../README.md)
- **모델 아키텍처 가이드**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **파라미터 분석 가이드**: [PARAMETERS.md](PARAMETERS.md)

---

**마지막 업데이트**: 2025-01-21
**VSCode 최소 버전**: 1.80+
**Python 확장 버전**: 2023.10+
