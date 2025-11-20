# Alpha 새 모델 크기 추가 가이드

Alpha 프로젝트에서 새로운 모델 변형(예: 32L, 40L, 48L 등)을 추가하고 MG→HF 변환하는 방법을 설명합니다.

---

## Quick Start

### 1단계: Config 파일 생성

기존 템플릿을 복사하여 새 모델 설정을 만듭니다:

```bash
cd toolkits/distributed_checkpoints_convertor/scripts/alpha/configs

# 예: baseline_40L 추가
cp baseline_24L.sh baseline_40L.sh
```

### 2단계: Config 파일 수정

`baseline_40L.sh` 파일을 열어 다음 값들을 수정:

```bash
#!/bin/bash
# Alpha baseline_40L Configuration

GPT_MODEL_ARGS+=(
    # 1. 레이어 수 변경
    --num-layers 40                      # ← 24에서 40으로 변경

    # 2. Hybrid Pattern 업데이트 (길이 = num_layers)
    --hybrid-attention-ratio 0.125       # 12.5% 유지
    --hybrid-mlp-ratio 0.5               # 50% 유지
    --hybrid-override-pattern M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-
    #                         ↑ 40개 토큰 (10 groups × 4)

    # 3. MoE 설정 (필요시 변경)
    --num-experts 256                    # 또는 512로 스케일업
    --moe-router-topk 8                  # 또는 10으로 증가
    --moe-ffn-hidden-size 768

    # 나머지는 동일...
    --hidden-size 2048
    --ffn-hidden-size 5120
    --num-attention-heads 32
)

# 4. Parallelism 설정
if [ -z "$MODEL_PARALLEL_ARGS" ]; then
    MODEL_PARALLEL_ARGS=(
        --tensor-model-parallel-size 1
        --pipeline-model-parallel-size 1
        --expert-model-parallel-size 8   # 256 experts / 8 = 32 per GPU
    )
fi

VOCAB_SIZE=151936
```

### 3단계: 변환 실행

학습 완료 후 checkpoint를 변환:

```bash
cd /path/to/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor

bash scripts/alpha/run_8xH20.sh \
  baseline_40L \
  /path/to/outputs/alpha_40L/checkpoints/iter_0010000 \
  /path/to/hf_models/alpha_40L_iter10k \
  true \
  true \
  bf16 \
  /path/to/models/Alpha-baseline-24L-config
```

### 4단계: 자동 검증 확인

스크립트가 자동으로 설정을 검증합니다:

```
Loading configuration: .../configs/baseline_40L.sh
✓ Alpha arguments validation passed
✓ Pattern length matches num_layers: 40
✓ Valid pattern characters: M, *, -
✓ Attention ratio: 5/40 = 0.125 (matches config)

======================================================================
Alpha MG2HF Conversion Configuration
======================================================================
Model Architecture:
  MG Layers:        40
  HF Layers:        20
  Mapping Ratio:    2:1 (MG → HF)

Hybrid Pattern:
  Pattern:          'M-M-M-*-M-M-M-*-...'
  Mamba layers:     30 (75.0%)
  Attention layers: 5 (12.5%)
  MLP layers:       20 (50.0%)

MoE Configuration:
  Num Experts:      256
  Router TopK:      8
  Expert FFN Size:  768

Parallelism:
  TP: 1, PP: 1, EP: 8
  DP: 0/1
======================================================================
```

### 5단계: 변환 결과 검증

```bash
# HF 모델 로드 테스트
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    '/path/to/hf_models/alpha_40L_iter10k',
    device_map='auto'
)
print(f'✓ Loaded {len(model.model.layers)} HF layers')  # Expected: 20
print(f'✓ Total params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B')
"
```

---

## 상세 설정 가이드

### Pattern 생성 규칙

**기본 원칙:**
- Pattern 길이 = `--num-layers` (정확히 일치해야 함)
- 각 토큰(M, *, -)이 1개의 Megatron layer를 나타냄
- 기본 그룹 패턴: `M-M-M-*-` (4 layers per group)

**다양한 레이어 수별 패턴:**

| 모델 크기 | num_layers | 그룹 수 | Pattern | Attention 개수 |
|-----------|-----------|---------|---------|---------------|
| baseline_24L | 24 | 6 | `(M-M-M-*-)×6` | 3 (12.5%) |
| baseline_32L | 32 | 8 | `(M-M-M-*-)×8` | 4 (12.5%) |
| baseline_40L | 40 | 10 | `(M-M-M-*-)×10` | 5 (12.5%) |
| baseline_48L | 48 | 12 | `(M-M-M-*-)×12` | 6 (12.5%) |

**Python으로 자동 생성:**

```python
def generate_pattern(num_layers, attention_ratio=0.125):
    """
    Alpha 모델 hybrid pattern 자동 생성

    Args:
        num_layers: Megatron 레이어 수
        attention_ratio: Attention 비율 (기본: 0.125 = 12.5%)

    Returns:
        Pattern string (예: "M-M-M-*-M-M-M-*-...")
    """
    # 기본 그룹 패턴
    group = "M-M-M-*-"
    num_groups = num_layers // 4
    pattern = group * num_groups

    # 검증
    assert len(pattern) == num_layers, \
        f"Pattern length {len(pattern)} != {num_layers}"

    expected_attn = int(num_layers * attention_ratio)
    actual_attn = pattern.count('*')
    assert actual_attn == expected_attn, \
        f"Attention count {actual_attn} != {expected_attn}"

    return pattern

# 사용 예시
for layers in [24, 32, 40, 48]:
    pattern = generate_pattern(layers)
    print(f"{layers}L: {pattern[:20]}... (length={len(pattern)})")
```

### MoE 스케일링 전략

모델 크기에 따라 expert 수를 조정할 수 있습니다:

**Option 1: Expert 수 고정 (권장)**
```bash
# 모든 크기에서 256 experts 유지
--num-experts 256
--moe-router-topk 8
--expert-model-parallel-size 8  # 256/8 = 32 per GPU
```

**장점:** 변환 일관성, 메모리 사용량 예측 가능
**단점:** 큰 모델에서 expert 활용도 낮을 수 있음

**Option 2: 레이어 비례 스케일링**
```bash
# 24L: 256 experts
# 48L: 512 experts (2배)

--num-experts 512
--moe-router-topk 10               # topk도 증가
--expert-model-parallel-size 8     # 512/8 = 64 per GPU
```

**장점:** 큰 모델에서 표현력 증가
**단점:** 메모리 사용량 증가, 변환 복잡도 증가

### Attention Ratio 조정

기본값 0.125(12.5%)를 유지하는 것을 권장하지만, 필요시 조정 가능:

```bash
# 더 많은 Full Attention (25%)
--hybrid-attention-ratio 0.25
--hybrid-override-pattern M-M-*-M-M-*-M-M-*-...  # 2 Mamba + 1 Attention

# 더 적은 Full Attention (6.25%)
--hybrid-attention-ratio 0.0625
--hybrid-override-pattern M-M-M-M-M-M-M-*-M-...  # 7 Mamba + 1 Attention
```

**주의:** Pattern의 실제 `*` 개수가 `num_layers × attention_ratio`와 일치해야 합니다.

---

## 체크리스트

새 모델 추가 시 반드시 확인할 사항:

### Config 파일 (`configs/<model_size>.sh`)

- [ ] `--num-layers` 설정 (예: 40)
- [ ] `--hybrid-override-pattern` 길이 = num_layers
- [ ] Pattern의 `*` 개수 = `num_layers × attention_ratio`
- [ ] Pattern의 `-` 개수 = `num_layers × mlp_ratio`
- [ ] `--num-experts` 설정
- [ ] `--expert-model-parallel-size` × GPU 수 ≥ num_experts
- [ ] `VOCAB_SIZE` 설정 (통상 151936)

### 변환 실행

- [ ] `MODEL_SIZE` 파라미터에 새 config 이름 사용
- [ ] `HF_DIR`에 reference tokenizer 경로 지정
- [ ] 자동 검증 메시지 확인 (에러 없음)
- [ ] Conversion summary에서 레이어 수 확인 (MG layers / 2 = HF layers)

### 검증

- [ ] HF 모델 로딩 성공
- [ ] `len(model.model.layers)` = num_layers / 2
- [ ] Config 확인: `num_hidden_layers`, `num_experts` 일치
- [ ] Parameter 수 합리적 (비슷한 크기 모델과 비교)

---

## 일반적인 모델 크기별 설정

### Small (24L ~ 32L)

```bash
--num-layers 24-32
--num-experts 256
--moe-router-topk 8
--hidden-size 2048
--num-attention-heads 32
```

**용도:** 빠른 실험, 프로토타이핑
**메모리:** ~10-15GB (bf16)

### Medium (40L ~ 64L)

```bash
--num-layers 40-64
--num-experts 256-512
--moe-router-topk 8-10
--hidden-size 2048-3072
--num-attention-heads 32-48
```

**용도:** 본격적인 학습, 벤치마크
**메모리:** ~20-40GB (bf16)

### Large (96L+)

```bash
--num-layers 96
--num-experts 512
--moe-router-topk 10
--hidden-size 4096
--num-attention-heads 64
```

**용도:** Production 모델
**메모리:** ~80GB+ (bf16), PP=2 이상 권장

---

## 트러블슈팅

### 에러: Pattern length mismatch

```
Pattern length mismatch!
  Pattern length: 24
  num_layers:     40
```

**원인:** Pattern을 업데이트하지 않음
**해결:** Pattern 길이를 40으로 수정

```bash
# Before (24 tokens)
M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-

# After (40 tokens)
M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-
```

### 에러: Attention ratio mismatch

```
Attention layer count mismatch (non-fatal):
  Expected: 5 layers (12.5%)
  Actual:   3 layers (7.5%)
```

**원인:** Pattern의 `*` 개수가 ratio와 불일치
**해결:** Pattern 또는 ratio 수정

```bash
# Option 1: Pattern 수정 (더 많은 *)
# 3개 → 5개로 증가

# Option 2: Ratio 수정
--hybrid-attention-ratio 0.075  # 3/40 = 7.5%
```

### 에러: Model configuration not found

```
Error: Model configuration not found: .../configs/baseline_40L.sh
Available configurations:
  baseline_24L.sh
  baseline_32L.sh
```

**원인:** Config 파일이 없음
**해결:** Config 파일 생성 후 재실행

### 에러: OOM during conversion

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**해결 방법:**

1. **CPU 모드 사용** (느리지만 안정적)
   ```bash
   bash run_8xH20.sh baseline_40L /load /save true false bf16 /hf-ref
   #                                              ↑ USE_CUDA=false
   ```

2. **EP 증가** (더 많은 GPU 활용)
   ```bash
   # configs/baseline_40L.sh에서
   --expert-model-parallel-size 16  # 8 → 16으로 증가
   ```

3. **Batch 변환** (여러 iteration 나눠서)
   ```bash
   # 전체 checkpoint 대신 일부만 변환
   ```

---

## 고급: 커스텀 Pattern

기본 `M-M-M-*-` 패턴 외에 다른 조합도 가능합니다:

### Dense Attention (더 많은 Attention)

```bash
# 50% Attention
--hybrid-attention-ratio 0.5
--hybrid-override-pattern M-*-M-*-M-*-M-*-...
```

### Sparse Attention (더 적은 Attention)

```bash
# 6.25% Attention (16 layers 중 1개)
--hybrid-attention-ratio 0.0625
--hybrid-override-pattern M-M-M-M-M-M-M-*-M-M-M-M-M-M-M-*-...
```

### Hybrid MLP Pattern

```bash
# MLP 비율 조정
--hybrid-mlp-ratio 0.25  # 50% → 25%
```

**주의:** 커스텀 패턴 사용 시 반드시 학습 config(`examples/alpha/configs/model/*.yaml`)도 동일하게 설정해야 합니다.

---

## 참고 자료

- **Converter README**: [scripts/alpha/README.md](../README.md)
- **상세 변환 가이드**: [examples/alpha/docs/CONVERSION.md](../../../../../examples/alpha/docs/CONVERSION.md)
- **기존 config 예시**: [configs/baseline_24L.sh](../configs/baseline_24L.sh)
- **학습 config**: [examples/alpha/configs/model/](../../../../../examples/alpha/configs/model/)

---

## 요약

1. **Config 복사**: `cp baseline_24L.sh baseline_<N>L.sh`
2. **수정**: `--num-layers`, `--hybrid-override-pattern` 업데이트
3. **검증**: Pattern 길이 = num_layers, `*` 개수 일치
4. **변환**: `bash run_8xH20.sh baseline_<N>L ...`
5. **확인**: 자동 검증 메시지 + HF 모델 로딩 테스트

질문이나 이슈 발생 시 자동 검증 메시지가 정확한 원인과 해결 방법을 제공합니다!
