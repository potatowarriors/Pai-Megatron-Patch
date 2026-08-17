# Alpha Model Parameter Analysis

이 문서는 Alpha MoE 모델의 파라미터 구성과 계산 방법을 설명합니다.

---

## 빠른 시작

### 파라미터 계산

```bash
# 기본 요약
bash calc_params.sh

# 상세 분석
bash calc_params.sh configs/model/baseline_48L.yaml --detailed

# 또는 Python 직접 실행
python calculate_parameters.py --config configs/model/baseline_48L.yaml --detailed
```

### 출력 예시

```
================================================================================
Alpha Model Parameter Summary: alpha-baseline-24L
================================================================================

Model Configuration:
  Total Layers: 24
    - Mamba Layers: 9
    - Attention Layers: 3
  Hidden Size: 2048
  Vocab Size: 151,936

MoE Configuration:
  Total Experts: 256
  Active Experts (Top-K): 8
  Expert Utilization: 3.1%

--------------------------------------------------------------------------------
Parameter Counts:
--------------------------------------------------------------------------------

Total Parameters:                       15.31B (15,314,630,656)
Active Parameters:                       1.27B (1,272,100,864)
Activation Ratio:                         8.3%
```

---

## Baseline 24L 모델 파라미터 분석

### 전체 구성

| 컴포넌트 | 전체 파라미터 | 활성화 파라미터 | 비율 |
|----------|--------------|----------------|------|
| **Embedding** | 622.33M | 622.33M | 4.1% |
| **Mamba Layers (9)** | 80.33M | 80.33M | 0.5% |
| **Attention Layers (3)** | 53.49M | 53.49M | 0.3% |
| **MoE Layers (12)** | 14.56B | 515.95M | 95.1% |
| **총합** | **15.31B** | **1.27B** | **100%** |

### 핵심 지표

- **전체 파라미터**: 15.31B (15,314,630,656)
- **활성화 파라미터**: 1.27B (1,272,100,864)
- **활성화 비율**: 8.3%
- **Expert 활용률**: 3.1% (8/256 experts)

### 해석

1. **MoE 지배적 구조**
   - MoE가 전체 파라미터의 95.1%를 차지
   - 256개 expert 중 8개만 활성화 (Top-K routing)
   - 대부분의 파라미터는 선택적으로 사용됨

2. **효율적인 추론**
   - 15.31B 모델이지만 실제로는 1.27B처럼 동작
   - 추론 시 메모리 효율성이 높음
   - GPU 메모리에는 활성화 파라미터만 로드 필요

3. **학습 vs 추론 비용**
   - **학습**: 15.31B 전체 파라미터 업데이트 필요 (높은 비용)
   - **추론**: 1.27B 활성화 파라미터만 사용 (낮은 비용)
   - 학습은 비싸지만, 추론은 효율적

---

## 레이어별 상세 분석

### 1. Embedding Layers (622.33M)

```
Input Embedding:  vocab_size × hidden_size
                  151,936 × 2,048 = 311.17M

Output Embedding: vocab_size × hidden_size (unshared)
                  151,936 × 2,048 = 311.17M

Total:            622.33M (항상 활성화)
```

**특징**:
- 입출력 임베딩이 분리됨 (`untie_embeddings_and_output_weights: true`)
- 모든 토큰에 대해 항상 활성화

### 2. Mamba Layers (9 layers × 8.93M = 80.33M)

**단일 레이어 구성**:
```
Input Projection:   hidden_size → (state_dim + 2 × num_groups × head_dim)
                    2,048 → (128 + 2 × 16 × 64) = 2,048 → 2,176

Conv1D:             state_dim × kernel_size
                    128 × 4 = 512

SSM Parameters:     A, B, C, D, dt matrices
                    ~1.5M params

Output Projection:  (num_heads × head_dim) → hidden_size
                    (32 × 64) → 2,048

Total per layer:    8.93M
```

**특징**:
- Linear Attention SSM (상태 공간 모델)
- Conv1D를 통한 시간적 정보 처리
- 모든 Mamba 레이어는 항상 활성화

### 3. Attention Layers (3 layers × 17.83M = 53.49M)

**단일 레이어 구성**:
```
Q Projection:       hidden_size → (num_heads × kv_channels)
                    2,048 → (32 × 128) = 2,048 → 4,096

K, V Projection:    hidden_size → (num_kv_heads × kv_channels) each
                    2,048 → (2 × 128) = 2,048 → 256 each (GQA)

Output Projection:  (num_heads × kv_channels) → hidden_size
                    4,096 → 2,048

QK LayerNorm:       ~1K params (if enabled)

Total per layer:    17.83M
```

**특징**:
- Grouped Query Attention (GQA) 사용 (32 heads, 2 groups)
- Q는 32 heads, K/V는 2 heads만 (메모리 절약)
- 모든 Attention 레이어는 항상 활성화

### 4. MoE Layers (12 layers × 1.21B = 14.56B)

**단일 레이어 구성**:
```
Router:             hidden_size → num_experts
                    2,048 → 256 = 0.52M

Expert FFN:         (gate + up + down) × num_experts
                    (2,048 × 768 + 2,048 × 768 + 768 × 2,048) × 256
                    = 5.24M × 256 = 1.34B

Shared Expert:      gate + up + down
                    (2,048 × 768) × 3 = 15.73M

Total per layer:    1.21B

Active per layer:   Router + (Expert FFN × topk) + Shared Expert
                    0.52M + (5.24M × 8) + 15.73M = 43.00M
```

**특징**:
- 256 experts, Top-8 routing (3.1% 활용률)
- Shared expert는 항상 활성화 (기본 지식)
- 전체 파라미터의 대부분을 차지하지만, 활성화는 극히 일부

---

## 하이브리드 패턴 분석

### Pattern: `M-M-M-*-M-M-M-*-M-M-M-*-` (24 layers)

```
Layer Type Distribution:
- M (Mamba):     9 layers (37.5%)
- * (Attention): 3 layers (12.5%)
- - (MLP):      12 layers (50%)

Pattern Breakdown:
M-M-M-*-  (4 layers) × 6 groups = 24 layers

Each group:
  3 Mamba layers + 1 Attention layer
```

**설계 의도**:
1. **Mamba 중심**: 대부분의 sequential processing을 Mamba가 담당
2. **Sparse Attention**: 핵심 정보만 Multi-Head Attention으로 처리
3. **MoE everywhere**: 모든 레이어에 MoE 적용 (파라미터 효율성)

---

## MoE 파라미터 효율성 분석

### Expert Utilization

```
Total Experts:        256
Active per token:     8 (Top-K)
Utilization:          3.1%

Memory Requirements:
- Training:           15.31B (모든 expert 저장)
- Inference:          1.27B (활성 expert만 로드)

Efficiency Ratio:     15.31B / 1.27B = 12.1x
```

### 메모리 비교 (vs Dense Model)

| 모델 | 전체 파라미터 | 활성 파라미터 | 추론 메모리 (BF16) |
|------|--------------|--------------|-------------------|
| **Alpha (MoE)** | 15.31B | 1.27B | ~2.5 GB |
| **Dense 15B** | 15.00B | 15.00B | ~30 GB |
| **Dense 1.3B** | 1.30B | 1.30B | ~2.6 GB |

**결론**:
- Alpha는 15B 모델의 capacity를 가지면서
- 1.3B 모델과 유사한 추론 비용만 발생
- **Best of both worlds**: 큰 모델 성능 + 작은 모델 효율성

---

## 파라미터 스케일링 시뮬레이션

### 다른 설정으로 확장 시

#### Option 1: More Layers (24L → 48L)
```
Mamba:     9 → 18 layers   (+80.33M)
Attention: 3 → 6 layers    (+53.49M)
MoE:      12 → 24 layers   (+14.56B)

Total:    15.31B → 29.90B  (거의 2배)
Active:    1.27B → 2.40B   (거의 2배)
```

#### Option 2: More Experts (256 → 512)
```
Expert params: 1.34B → 2.68B per layer
Total:        15.31B → 29.72B  (거의 2배)
Active:        1.27B → 1.54B   (1.2배, Top-K=8 유지)
```

#### Option 3: Larger Hidden Size (2048 → 4096)
```
대부분의 파라미터가 비례적으로 증가
Total:        15.31B → ~60B   (4배)
Active:        1.27B → ~5B    (4배)
```

**권장 스케일링 전략**:
1. **Capacity 중심**: More experts (추론 비용 적게 증가)
2. **Quality 중심**: More layers (균형있는 증가)
3. **Universal**: Larger hidden size (모든 면에서 증가)

---

## 계산 방법론

### 계산 공식

파라미터 계산기는 다음 공식을 사용합니다:

```python
# Embedding
input_embed = vocab_size × hidden_size
output_embed = vocab_size × hidden_size (if unshared)

# Mamba Layer
in_proj = hidden_size × (state_dim + 2 × num_groups × head_dim)
conv1d = state_dim × kernel_size
ssm = num_heads × head_dim × state_dim + ...
out_proj = (num_heads × head_dim) × hidden_size

# Attention Layer
q_proj = hidden_size × (num_heads × kv_channels)
k_proj = hidden_size × (num_kv_heads × kv_channels)
v_proj = hidden_size × (num_kv_heads × kv_channels)
out_proj = (num_heads × kv_channels) × hidden_size

# MoE Layer
router = hidden_size × num_experts
expert_ffn = (gate + up + down) × num_experts
  where gate = hidden_size × expert_ffn_size
        up = hidden_size × expert_ffn_size
        down = expert_ffn_size × hidden_size
shared_expert = (gate + up + down)

# Active Parameters
active_expert_ffn = (gate + up + down) × topk
active_moe = router + active_expert_ffn + shared_expert
```

### 검증 방법

실제 모델 파라미터와 비교:
```bash
# 1. 모델 빌드 후 파라미터 카운트
python -c "
from pretrain_alpha import model_provider
model = model_provider(...)
print(sum(p.numel() for p in model.parameters()))
"

# 2. 계산기 결과와 비교
python calculate_parameters.py --config configs/model/baseline_48L.yaml
```

일반적으로 1-2% 오차 내에서 일치합니다.

---

## 사용 예시

### 1. 기본 계산
```bash
$ bash calc_params.sh

Total Parameters:    15.31B
Active Parameters:   1.27B
Activation Ratio:    8.3%
```

### 2. 상세 분석
```bash
$ bash calc_params.sh configs/model/baseline_48L.yaml --detailed

# 컴포넌트별 상세 분석 출력
# 결과는 configs/outputs/alpha-baseline-24L_params.txt에 저장
```

### 3. 다른 모델 설정 분석
```bash
# 새로운 모델 설정 생성
cp configs/model/baseline_48L.yaml configs/model/my_model.yaml
# my_model.yaml 편집...

# 파라미터 계산
bash calc_params.sh configs/model/my_model.yaml --detailed
```

### 4. Python API 사용
```python
from calculate_parameters import AlphaParameterCalculator
import yaml

# 설정 로드
with open('configs/model/baseline_48L.yaml') as f:
    config = yaml.safe_load(f)

# 계산
calc = AlphaParameterCalculator(config)
results = calc.calculate()

# 결과 출력
calc.print_summary(results, detailed=True)

# 개별 값 접근
print(f"Total: {results['total_params']:,}")
print(f"Active: {results['active_params']:,}")
print(f"Ratio: {results['active_params']/results['total_params']:.1%}")
```

---

## 참고 자료

- **모델 아키텍처**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **설정 가이드**: [../configs/model/baseline_48L.yaml](../configs/model/baseline_48L.yaml)
- **Megatron-LM MoE**: https://github.com/NVIDIA/Megatron-LM/blob/main/docs/moe.md

---

## FAQ

### Q1: 왜 활성화 파라미터가 8.3%밖에 안 되나요?

A: MoE 구조에서 256개 expert 중 8개만 사용하기 때문입니다 (3.1% utilization). MoE가 전체 파라미터의 95.1%를 차지하므로, 전체 활성화 비율도 낮아집니다.

### Q2: 학습 시 메모리는 얼마나 필요한가요?

A: 학습 시에는 **모든 파라미터** (15.31B)를 메모리에 로드해야 합니다. Optimizer states까지 고려하면:
- Model params: 15.31B × 2 bytes (BF16) = 30.62 GB
- Optimizer states (Adam): 15.31B × 8 bytes = 122.48 GB
- Total: ~153 GB (단일 GPU 기준)

분산 학습 (TP=1, PP=1, EP=8)로 나누면:
- Per GPU: ~19 GB (8 GPUs)

### Q3: 추론 시 메모리는 얼마나 필요한가요?

A: 추론 시에는 **활성화 파라미터** (1.27B)만 필요합니다:
- Model params: 1.27B × 2 bytes (BF16) = 2.54 GB
- KV Cache (seq_len=2048, batch=1): ~0.5 GB
- Total: ~3 GB (단일 GPU로 가능)

### Q4: Dense 모델과 비교하면 어떤가요?

A:
- **Capacity**: 15.31B MoE ≈ 5-7B Dense (경험적)
- **Efficiency**: 15.31B MoE 추론 비용 = 1.27B Dense
- **Trade-off**: 학습은 비싸지만, 추론은 매우 효율적

### Q5: Expert 수를 늘리면 성능이 좋아지나요?

A: 일반적으로 그렇습니다만, trade-off가 있습니다:
- **장점**: 더 많은 specialization, 더 높은 capacity
- **단점**: 학습 메모리 증가, expert 활용률 감소
- **최적값**: 256-512 experts가 일반적 (Top-K=8 기준)

---

**마지막 업데이트**: 2025-01-21
**계산기 버전**: 1.0.0
