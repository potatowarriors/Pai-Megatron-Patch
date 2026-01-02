# Alpha Model Checkpoint Converter

Alpha 모델의 HuggingFace ↔ Megatron 체크포인트 변환 도구입니다.

## 개요

Alpha는 Qwen3-Next Mamba Hybrid 아키텍처 기반의 모델입니다:

- **48 Megatron layers → 24 HF layers** (2:1 mapping)
- **128 experts** with Top-8 routing
- **Hybrid pattern**: GatedDeltaNet + Full Attention
- **TP=1 constraint**: Mamba layers require Tensor Parallelism = 1

> **상세 가이드**: 아키텍처 변경, 트러블슈팅, 내부 동작 원리는 [examples/alpha/docs/CONVERSION.md](../../../../examples/alpha/docs/CONVERSION.md) 참조

---

## Quick Start

### Prerequisites

```bash
# CUDA 설정 (스크립트에서 자동 설정되지만 확인용)
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
```

### HuggingFace → Megatron

```bash
cd /path/to/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor

bash scripts/alpha/run_8xH20.sh \
  baseline_48L \
  /path/to/alpha-hf-checkpoint \
  /path/to/alpha-mcore-output \
  false \
  true \
  bf16
```

### Megatron → HuggingFace

```bash
bash scripts/alpha/run_8xH20.sh \
  baseline_48L \
  /path/to/alpha-mcore-checkpoint \
  /path/to/alpha-hf-output \
  true \
  true \
  bf16 \
  /path/to/alpha-hf-reference
```

### 🚀 Auto Mode (권장)

훈련 출력 디렉토리를 직접 지정하면 자동으로 체크포인트를 찾아 변환합니다:

```bash
# 최신 체크포인트 자동 변환
bash scripts/alpha/run_8xH20.sh \
  baseline_48L \
  /path/to/outputs/alpha_baseline_48L_20251219_095156 \
  auto \
  true \
  true \
  bf16

# 특정 iteration 지정
bash scripts/alpha/run_8xH20.sh \
  baseline_48L \
  /path/to/outputs/alpha_baseline_48L_20251219_095156 \
  auto:50000 \
  true \
  true \
  bf16
```

**Auto Mode 동작:**
- `auto`: `checkpoints/latest_checkpointed_iteration.txt`에서 최신 iteration 자동 감지
- `auto:50000`: 지정된 iteration 사용
- 출력 경로 자동 생성: `{OUTPUT_DIR}/hfmodel_{ITERATION:07d}`

```
📍 Using latest iteration: 100000

🔄 Auto mode enabled:
   Input:  .../checkpoints/iter_0100000
   Output: .../hfmodel_0100000
```

---

## Script Arguments

| 위치 | 인자 | 설명 | 예시 |
|------|------|------|------|
| 1 | `MODEL_SIZE` | 모델 설정 | `baseline_48L`, `baseline_32L` |
| 2 | `LOAD_DIR` | 입력 체크포인트 또는 훈련 출력 경로 (auto mode) | `/data/ckpts/alpha-hf` |
| 3 | `SAVE_DIR` | 출력 경로, 또는 `auto`/`auto:ITER` | `auto`, `auto:50000` |
| 4 | `MG2HF` | 변환 방향 | `true` (MG→HF), `false` (HF→MG) |
| 5 | `USE_CUDA` | GPU 사용 | `true` (권장), `false` (CPU) |
| 6 | `PRECISION` | 정밀도 | `bf16`, `fp16`, `fp32` |
| 7 | `HF_DIR` | 원본 HF 모델 (MG→HF 시, optional) | `/data/alpha-hf-orig` |

---

## Model Configurations

### baseline_48L (현재 지원)

```yaml
# Architecture
num_layers: 48 (MG) → 24 (HF)
hidden_size: 2048
ffn_hidden_size: 8192
num_attention_heads: 32
num_query_groups: 2

# MoE
num_experts: 128
router_topk: 8
moe_ffn_hidden_size: 768

# Hybrid Pattern
hybrid_attention_ratio: 0.125    # 12.5% (48 layers 중 6개)
hybrid_mlp_ratio: 0.5            # 50% (48 layers 중 24개)
hybrid_override_pattern: "MDM-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-"

# Parallelism
TP: 1 (필수), PP: 1, EP: 8, DP: 1
```

> **새 모델 크기 추가**: `run_8xH20.sh`의 MODEL_SIZE 분기에 설정 추가. 상세 방법은 [CONVERSION.md](../../../../examples/alpha/docs/CONVERSION.md#아키텍처-변경-시-업데이트) 참조.

---

## Validation

### 변환 성공 확인

```bash
# 1. 파일 존재 확인
ls /path/to/alpha-hf-output/
# → config.json, model-*.safetensors, tokenizer.json

# 2. HF 모델 로드 테스트
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('/path/to/alpha-hf-output', device_map='auto')
print(f'✓ Loaded {len(model.model.layers)} layers')
"
# Expected: 24 layers (HF format)
```

### 변환 설정 검증

변환 전/후 자동 검증이 실행됩니다:

```
✓ Alpha arguments validation passed
✓ Pattern length matches num_layers: 48
✓ Valid pattern characters: M, *, -, D
✓ Attention ratio: 6/48 = 0.125 (matches config)
✓ MG Layers: 48 → HF Layers: 24 (2:1 mapping)
```

---

## Troubleshooting

### Common Issues

| 에러 | 원인 | 해결 |
|------|------|------|
| `assert self.tp_size == 1` | TP > 1 사용 | TP=1로 변환 (Mamba 제약) |
| `Pattern length mismatch` | 잘못된 pattern 길이 | 24 tokens 확인 (각 MG layer당 1개) |
| `Invalid characters in pattern` | M, *, - 외 문자 | Pattern 수정 |
| `OOM during conversion` | GPU 메모리 부족 | `USE_CUDA=false` 또는 GPU 추가 |
| `Tokenizer not found` | tokenizer 파일 누락 | HF_DIR에서 자동 복사됨 확인 |

> **상세 트러블슈팅**: [CONVERSION.md](../../../../examples/alpha/docs/CONVERSION.md#트러블슈팅) 참조

---

## Advanced Usage

### 다중 iteration 배치 변환

```bash
# Auto mode로 간편하게 특정 iteration들 변환
for iter in 50000 100000 150000; do
  bash scripts/alpha/run_8xH20.sh \
    baseline_48L \
    /path/to/outputs/alpha_baseline_48L_20251219_095156 \
    auto:${iter} \
    true true bf16
done
# 결과: hfmodel_0050000, hfmodel_0100000, hfmodel_0150000
```

### Custom Expert Parallelism

```bash
# EP=4로 변환 (128 experts / 4 = 32 per GPU)
# run_8xH20.sh에서 EXPERT_PARALLEL_SIZE 수정 또는:
export EXPERT_PARALLEL_SIZE=4
bash scripts/alpha/run_8xH20.sh baseline_48L /load /save true true bf16 /hf-ref
```

---

## Documentation

- **새 모델 추가 가이드**: [docs/ADDING_NEW_MODELS.md](docs/ADDING_NEW_MODELS.md) ⭐
  - Config 파일 생성 방법
  - Pattern 생성 규칙 및 자동 생성 스크립트
  - MoE 스케일링 전략
  - 체크리스트 및 트러블슈팅

- **상세 변환 가이드**: [examples/alpha/docs/CONVERSION.md](../../../../examples/alpha/docs/CONVERSION.md)
  - 아키텍처 변경 시 업데이트 방법
  - 내부 동작 원리 (weight mapping, synchronizer)
  - FAQ 및 성능 최적화

- **관련 파일**:
  - 모델 설정: [examples/alpha/configs/model/baseline_48L.yaml](../../../../examples/alpha/configs/model/baseline_48L.yaml)
  - 변환 구현: [toolkits/.../impl/alpha/m2h_synchronizer.py](../../impl/alpha/m2h_synchronizer.py)
  - 모델 정의: [toolkits/.../impl/alpha/model_provider.py](../../impl/alpha/model_provider.py)

---

## Requirements

- **Megatron-LM**: 250908 (latest)
- **PyTorch**: 2.3+
- **CUDA**: 12.1+
- **Transformer Engine**: 2.8+
