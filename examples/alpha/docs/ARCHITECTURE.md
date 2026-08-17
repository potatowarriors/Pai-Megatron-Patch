# Alpha 모델 아키텍처

Alpha 프로젝트의 모델 아키텍처 상세 문서

---

## 목차

1. [개요](#개요)
2. [Base: Qwen3-Next-80B-A3B](#base-qwen3-next-80b-a3b)
3. [Alpha Baseline 24L](#alpha-baseline-24l)
4. [Hybrid Architecture](#hybrid-architecture)
5. [Mixture-of-Experts](#mixture-of-experts)
6. [Attention Mechanism](#attention-mechanism)
7. [Positional Encoding](#positional-encoding)
8. [메모리 최적화](#메모리-최적화)

---

## 개요

Alpha는 **Qwen3-Next-80B-A3B** 아키텍처를 기반으로 하며, 다음 핵심 컴포넌트를 결합합니다:

1. **Mamba State Space Model (SSM)**: 효율적인 시퀀스 모델링
2. **Multi-Head Attention**: Selective attention for critical layers
3. **Mixture-of-Experts (MoE)**: Sparse activation for scalability
4. **Group Query Attention (GQA)**: Memory-efficient KV caching

### 주요 설계 원칙

- **Hybrid Sparsity**: Mamba (dense SSM) + Attention (selective focus)
- **Expert Specialization**: 256 experts with top-8 routing
- **Kimi-Linear Alignment**: 더 많은 heads, 작은 head_dim
- **Memory Efficiency**: 75% layer reduction with capacity compensation

---

## Base: Qwen3-Next-80B-A3B

Qwen3-Next-80B-A3B는 Alibaba의 최신 MoE 기반 언어 모델입니다.

### 원본 아키텍처

```
총 파라미터: ~80B (Active: ~3B per token)
├─ Dense Parameters: ~32B
│  ├─ Embeddings: ~2B
│  ├─ Mamba Layers: ~24B
│  └─ Attention Layers: ~6B
└─ MoE Parameters: ~48B
   ├─ 512 Experts × ~94M each
   └─ Top-10 routing (Active: ~940M per token)
```

### 주요 특징

- **96 Layers**: Hybrid Mamba (87.5%) + Attention (12.5%)
- **Hidden Size**: 2048
- **512 Experts**: Top-10 routing
- **GQA**: 16 heads, 2 query groups
- **RoPE**: Base 10M, 25% rotary

---

## Alpha Baseline 24L

### 아키텍처 다이어그램

```
Alpha Baseline 24L Model
├─ Input Embeddings (151936 vocab)
│
├─ 24 Transformer Layers
│  │
│  ├─ Group 1 (Layers 0-3)
│  │  ├─ Layer 0: Mamba Block
│  │  ├─ Layer 1: Mamba Block
│  │  ├─ Layer 2: Mamba Block
│  │  └─ Layer 3: Attention Block ⭐
│  │
│  ├─ Group 2 (Layers 4-7)
│  │  ├─ Layer 4: Mamba Block
│  │  ├─ Layer 5: Mamba Block
│  │  ├─ Layer 6: Mamba Block
│  │  └─ Layer 7: Attention Block ⭐
│  │
│  ├─ ... (Groups 3-5 동일 패턴)
│  │
│  └─ Group 6 (Layers 20-23)
│     ├─ Layer 20: Mamba Block
│     ├─ Layer 21: Mamba Block
│     ├─ Layer 22: Mamba Block
│     └─ Layer 23: Attention Block ⭐
│
└─ Output Head (LM Head, untied)

Pattern: M-M-M-*-M-M-M-*-M-M-M-*- (repeat 2x)
Total: 21 Mamba + 3 Attention = 12.5% attention ratio
```

### 파라미터 분포

```
총 파라미터: ~X.XB (추정)
├─ Dense Parameters: ~X.XB
│  ├─ Embeddings: ~0.3B (2048 × 151936)
│  ├─ Mamba Layers (21): ~XB
│  └─ Attention Layers (3): ~XB
└─ MoE Parameters: ~XX.XB
   ├─ 256 Experts × 768 hidden
   ├─ Shared Expert: 768 hidden
   └─ Active per token: 8 experts
```

### 모델 설정 비교

| 항목 | Qwen3-Next-80B-A3B | Alpha Baseline 24L | 변경 이유 |
|------|-------------------|-------------------|----------|
| **Layers** | 96 | 24 | **메모리 절감** (75% 감소) |
| **Hidden Size** | 2048 | 2048 | 유지 |
| **FFN Hidden** | 5120 | 5120 | 유지 (2.5x ratio) |
| **Attention Heads** | 16 | 32 | **Diversity 증가** (Kimi-Linear) |
| **KV Channels** | 256 | 128 | **메모리 절감** (head_dim 64) |
| **Query Groups** | 2 | 2 | 유지 (GQA) |
| **Experts** | 512 | 256 | **메모리 절감** (50% 감소) |
| **MoE FFN Hidden** | 512 | 768 | **Capacity 보상** (50% 증가) |
| **Router TopK** | 10 | 8 | **효율성** (20% 감소) |
| **Shared Expert** | 512 | 768 | **정보 공유** (증가) |
| **Hybrid Ratio** | 12.5% | 12.5% | 유지 |

---

## Hybrid Architecture

### Mamba State Space Model

**Mamba Block 구조**:

```python
MambaBlock(
    input: [batch, seq, hidden=2048]
) -> output: [batch, seq, hidden=2048]

Components:
├─ Input Layernorm (RMSNorm, ε=1e-6)
├─ Mamba Mixer
│  ├─ State Dimension: 128
│  ├─ Head Dimension: 64
│  ├─ Num Groups: 16
│  ├─ Num Heads: 32
│  └─ Conv Kernel: 4
└─ Residual Connection
```

**Mamba vs Attention**:

| 특성 | Mamba SSM | Multi-Head Attention |
|------|-----------|---------------------|
| **복잡도** | O(N) | O(N²) |
| **장기 의존성** | State 기반 | Direct attention |
| **메모리** | O(N) | O(N²) for KV cache |
| **학습 안정성** | 높음 | 중간 (QK layernorm 필요) |
| **사용 비율** | 87.5% (21/24) | 12.5% (3/24) |

### Hybrid Pattern

```
Pattern: M-M-M-*-M-M-M-*-M-M-M-*-

M: Mamba Block (21개)
*: Attention Block (3개)

그룹별 배치:
- Group 1 (0-3): M M M A
- Group 2 (4-7): M M M A
- Group 3 (8-11): M M M A
- Group 4 (12-15): M M M A
- Group 5 (16-19): M M M A
- Group 6 (20-23): M M M A

각 그룹마다 마지막 레이어가 Attention
→ 주기적으로 global context aggregation
```

### Hybrid 장점

1. **효율성**: 대부분 O(N) Mamba로 처리
2. **표현력**: Critical layers에서 full attention
3. **메모리**: Attention 12.5%만 O(N²) 메모리 사용
4. **학습 안정성**: Attention이 global signal 제공

---

## Mixture-of-Experts

### MoE 구조

```
MoE Layer
├─ Router (Top-8 Routing)
│  ├─ Input: [batch, seq, hidden=2048]
│  ├─ Linear: [2048 → 256]
│  ├─ Softmax (FP32)
│  └─ Top-8 Selection
│
├─ 256 Experts
│  ├─ Expert FFN
│  │  ├─ W_gate: [2048 → 768]
│  │  ├─ W_up: [2048 → 768]
│  │  ├─ SwiGLU Activation
│  │  └─ W_down: [768 → 2048]
│  └─ Expert Parallel Size: 8
│     → Each GPU handles 32 experts
│
└─ Shared Expert
   ├─ Always Active
   ├─ FFN Hidden: 768
   └─ 정보 공유 및 안정성 향상
```

### Router 메커니즘

**Top-8 Routing**:

1. **Scoring**: `scores = softmax(Wx)` (FP32 for precision)
2. **Selection**: Top-8 highest scores
3. **Load Balancing**: Auxiliary loss (coeff=0.001)
   ```
   L_aux = α × ∑(f_i × P_i)
   f_i: fraction of tokens to expert i
   P_i: average router probability to expert i
   ```

**Expert Specialization**:
- 256 experts → 다양한 패턴 학습 가능
- Top-8 → 적당한 sparsity (3.1% active)
- Shared expert → 공통 지식 anchor

### Expert Parallel

```
8 GPUs × 32 experts each = 256 total experts

GPU 0: Experts [0-31]
GPU 1: Experts [32-63]
GPU 2: Experts [64-95]
...
GPU 7: Experts [224-255]

All-to-All Communication:
- Token routing: 각 token을 담당 GPU로 전송
- Expert computation: 병렬 처리
- Gather: 결과 수집
```

---

## Attention Mechanism

### Group Query Attention (GQA)

```
Multi-Query vs GQA vs MHA

MHA (Multi-Head Attention):
  Q: 32 heads × 64 dim = 2048
  K: 32 heads × 64 dim = 2048
  V: 32 heads × 64 dim = 2048

GQA (Group Query Attention): ⭐ Alpha 사용
  Q: 32 heads × 64 dim = 2048
  K: 2 groups × 128 dim = 256  (16x compression!)
  V: 2 groups × 128 dim = 256

MQA (Multi-Query Attention):
  Q: 32 heads × 64 dim = 2048
  K: 1 head × 2048 dim = 2048
  V: 1 head × 2048 dim = 2048
```

### GQA 장점

1. **KV Cache 메모리**: 16배 감소 (32 heads → 2 groups)
2. **표현력 유지**: MQA보다 높은 품질
3. **추론 속도**: Smaller KV cache → faster decoding

### Attention 상세

```python
AttentionBlock(
    input: [batch, seq, hidden=2048]
) -> output: [batch, seq, hidden=2048]

Components:
├─ Input Layernorm (RMSNorm)
├─ QKV Projection
│  ├─ Q: [2048 → 32 heads × 64 dim]
│  ├─ K: [2048 → 2 groups × 128 dim]
│  └─ V: [2048 → 2 groups × 128 dim]
├─ RoPE (Rotary Position Embedding)
│  ├─ Base: 10,000,000
│  ├─ Percent: 25% (first 16 dims)
│  └─ Applied to Q, K
├─ QK Layernorm (stability)
├─ Scaled Dot-Product Attention
│  ├─ Flash Attention 3 (H100 optimized)
│  ├─ Scale: 1/sqrt(64)
│  └─ Causal masking
├─ Output Projection
└─ Residual Connection
```

### QK Layernorm

Attention 학습 안정성을 위한 추가 정규화:

```python
Q_norm = LayerNorm(Q)
K_norm = LayerNorm(K)
attn = softmax(Q_norm @ K_norm^T / sqrt(d_k))
```

**효과**:
- Pre-softmax값 안정화
- Gradient explosion 방지
- 더 높은 learning rate 가능

---

## Positional Encoding

### Rotary Position Embedding (RoPE)

**설정**:
- **Base**: 10,000,000 (long context 지원)
- **Percent**: 0.25 (첫 25% dimensions에만 적용)
- **Dimensions**: 64 (head_dim) × 0.25 = 16 dims rotated

**수식**:

```
θ_i = base^(-2i/d), i ∈ [0, d/4)

RoPE(x, m) = [
  x[0:d/4] ⊙ cos(m·θ) - x[d/4:d/2] ⊙ sin(m·θ),
  x[d/4:d/2] ⊙ sin(m·θ) + x[0:d/4] ⊙ cos(m·θ),
  x[d/2:]  (unchanged)
]

m: position index
d: head dimension (64)
```

**특징**:
1. **Relative positioning**: 상대적 위치 encoding
2. **Extrapolation**: 학습보다 긴 시퀀스 추론 가능
3. **Efficiency**: 일부 dims만 rotate (25%)

---

## 메모리 최적화

### 1. 레이어 감소 (96 → 24)

**메모리 절감**:
```
Parameters per layer: ~X GB
96 layers → 24 layers: 75% 감소
절감량: ~XX GB
```

**Capacity 보상**:
- Expert size 증가 (512 → 768)
- Shared expert 강화
- 더 긴 학습 (토큰 수 증가)

### 2. Attention Head Dimension 감소

**원본**: 16 heads × 256 dim = 4096 total
**Alpha**: 32 heads × 64 dim = 2048 total

**KV Cache 메모리 (per token)**:
```
원본: 2 groups × 256 dim × 2 (K,V) × 2 bytes (FP16) = 2 KB
Alpha: 2 groups × 128 dim × 2 (K,V) × 2 bytes (FP16) = 1 KB

Batch 256, Seq 4096:
원본: 2 KB × 256 × 4096 = 2 GB
Alpha: 1 KB × 256 × 4096 = 1 GB
절감: 50%
```

### 3. Expert 감소 (512 → 256)

**파라미터 절감**:
```
Expert size: 768 (increased from 512)
원본: 512 experts × 512 hidden × layers
Alpha: 256 experts × 768 hidden × layers

Net reduction: ~XX% (expert 수 감소 > size 증가)
```

### 4. Activation Checkpointing

**Selective Recomputation**:
```
Recompute modules:
- layernorm (cheap to recompute)
- moe_act (SwiGLU activation)
- shared_experts (always active)

Not recomputed:
- Router (expensive softmax)
- Expert outputs (alltoall communication)
```

**메모리 절감**: ~60-70%
**연산 오버헤드**: ~15-20%

### 5. Distributed Optimizer

**ZeRO Stage 1** (optimizer state sharding):
```
8 GPUs:
- Optimizer states: 각 GPU 1/8 저장
- Gradients: All-reduce
- Parameters: Replicated

메모리 절감: Optimizer state × 7/8
```

---

## 성능 특성

### 이론적 FLOPs (per token)

```
Mamba Blocks (21):
  - SSM: ~X TFLOPs
  - Linear: ~X TFLOPs
  Total: ~XX TFLOPs

Attention Blocks (3):
  - QKV projection: ~X TFLOPs
  - Attention: ~X TFLOPs (O(N²))
  - Output: ~X TFLOPs
  Total: ~X TFLOPs

MoE:
  - Router: ~X TFLOPs
  - 8 Experts: ~X TFLOPs
  - Shared Expert: ~X TFLOPs
  Total: ~X TFLOPs

Grand Total: ~XXX TFLOPs per token
```

### 메모리 사용 (예상)

**모델 파라미터**: ~X.X GB (FP16)
**Optimizer States**: ~XX GB (Adam: 2× gradients + 2× params)
**Activations**: ~X GB (batch 256, seq 4096, selective checkpointing)
**KV Cache**: ~X GB (inference only)

**Total (training)**: ~XX GB per GPU (8 GPUs)

---

## 참고 자료

- **Qwen3 Technical Report**: [Link]
- **Mamba Paper**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- **GQA Paper**: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
- **RoPE Paper**: "RoFormer: Enhanced Transformer with Rotary Position Embedding"

---

**마지막 업데이트**: 2025-01-17
