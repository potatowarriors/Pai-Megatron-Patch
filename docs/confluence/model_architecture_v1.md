# Alpha 모델 아키텍처 v1

> **Confluence 페이지용**: 이 문서는 Confluence "Model Architecture v1" 페이지에 업로드할 내용입니다.
> - 스페이스: alpha banana (AB)
> - 페이지 ID: 720912

---

## 1. 개요

**Alpha**는 Qwen3-Next-80B-A3B 아키텍처를 기반으로 한 **Mamba-Attention-MoE 하이브리드** 모델입니다.

### 핵심 특징

| 특징 | 설명 |
|------|------|
| **GatedDeltaNet (ICLR 2025)** | O(n) 선형 복잡도의 Linear Attention |
| **하이브리드 아키텍처** | Mamba SSM (87.5%) + Multi-Head Attention (12.5%) |
| **Mixture-of-Experts** | 128개 전문가, Top-8 라우팅 |
| **Group Query Attention** | KV 캐시 16배 감소 |

---

## 2. 전체 아키텍처 다이어그램

```
Alpha Model Architecture
├─ 입력 임베딩 (vocab=151,936)
│
├─ Transformer Stack (48 Megatron 레이어 → 24 HuggingFace 레이어)
│  │
│  ├─ 그룹 1-6: 각 그룹은 [M - M - M - *] 패턴
│  │   ├─ M: GatedDeltaNet (Mamba) 레이어
│  │   ├─ *: Full Attention 레이어
│  │   └─ -: MoE MLP 레이어 (암묵적 포함)
│  │
│  └─ 레이어별 구성:
│      ├─ Pre-LayerNorm (RMSNorm)
│      ├─ Mixer (Mamba 또는 Attention)
│      ├─ Post-LayerNorm (RMSNorm)
│      └─ MLP (MoE 또는 Dense)
│
├─ Final LayerNorm (RMSNorm)
│
└─ 출력 헤드 (LM Head, untied embeddings)
```

---

## 3. 레이어 패턴 상세

### 3.1 패턴 문자 설명

| 문자 | 의미 | 설명 |
|------|------|------|
| **M** | GatedDeltaNet (Mamba) | O(n) 선형 복잡도 Linear Attention |
| **\*** | Full Attention | O(n²) Multi-Head Attention (Flash Attention) |
| **-** | MoE MLP | Mixture-of-Experts FFN 레이어 |
| **D** | Dense MLP | 표준 FFN 레이어 (MoE 대신) |

### 3.2 Baseline 24L 패턴

```
패턴: M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-

구성:
  - Mamba (M): 18개 (75%)
  - Attention (*): 6개 (25%)
  - MoE (-): 모든 레이어에 포함

그룹별 배치 (6개 그룹, 각 4레이어):
  Group 1: M M M *
  Group 2: M M M *
  Group 3: M M M *
  Group 4: M M M *
  Group 5: M M M *
  Group 6: M M M *
```

### 3.3 Baseline 48L 패턴

```
패턴: MDM-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-

구성:
  - Mamba (M): 36개 (75%)
  - Attention (*): 12개 (25%)
  - Dense (D): 1개 (첫 번째 MLP만)
  - MoE (-): 나머지 MLP
```

---

## 4. 주요 하이퍼파라미터

### 4.1 모델 기본 설정

| 파라미터 | Baseline 24L | Baseline 48L |
|---------|--------------|--------------|
| **Megatron 레이어 수** | 24 | 48 |
| **HuggingFace 레이어 수** | 12 | 24 |
| **Hidden Size** | 2048 | 2048 |
| **FFN Hidden Size** | 8192 (4x) | 8192 |
| **Vocab Size** | 151,936 | 151,936 |

### 4.2 Attention 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **Attention Heads** | 32 | 더 많은 헤드, 작은 차원 |
| **KV Channels** | 128 | head_dim = 64 |
| **Query Groups (GQA)** | 2 | KV 캐시 16배 감소 |
| **QK LayerNorm** | 활성화 | 안정성 향상 |

### 4.3 GatedDeltaNet (Mamba) 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **State Dimension** | 128 | SSM 상태 크기 |
| **Head Dimension** | 128 | GatedDeltaNet 헤드 |
| **Num Groups** | 16 | 모노헤드 그룹 |
| **Num Heads** | 32 | Mamba 헤드 |
| **Conv Kernel** | 4 | Causal Conv1d |

### 4.4 MoE 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **Num Experts** | 128 | 라우팅 전문가 수 |
| **Router TopK** | 8 | 토큰당 활성 전문가 |
| **Expert FFN Size** | 768 | 전문가 중간 크기 |
| **Shared Expert Size** | 768 | 공유 전문가 |
| **Load Balancing** | aux_loss (0.001) | 로드 밸런싱 손실 |

### 4.5 RoPE 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **RoPE Base** | 10,000,000 | 장문맥 지원 |
| **RoPE Percent** | 0.25 | 25% 차원에만 적용 |

---

## 5. Megatron ↔ HuggingFace 레이어 매핑

### 5.1 2:1 매핑 원칙

Megatron의 2개 레이어가 HuggingFace의 1개 레이어로 매핑됩니다.

```
Megatron (48 레이어)       HuggingFace (24 레이어)
─────────────────────────────────────────────────
MG Layer 0 (Mamba)    ─┐
MG Layer 1 (MoE)      ─┴─→  HF Layer 0
MG Layer 2 (Mamba)    ─┐
MG Layer 3 (MoE)      ─┴─→  HF Layer 1
...
MG Layer 46 (Mamba)   ─┐
MG Layer 47 (MoE)     ─┴─→  HF Layer 23
```

### 5.2 가중치 레이아웃 차이

**Megatron 구조:**
```
transformer.mamba_stack.layers.0.mixer.*  (Mamba L0)
transformer.mamba_stack.layers.1.mlp.*    (MoE L1)
```

**HuggingFace 구조:**
```
model.layers.0.linear_attention.*  (from MG-0)
model.layers.0.mlp.*               (from MG-1)
```

---

## 6. 주요 컴포넌트 상세

### 6.1 GatedDeltaNet 레이어

```
GatedDeltaNetMixer
├─ Input Projection (in_proj)
│   └─ hidden_size → [z, V, Q, K, b, a] 투영
├─ Causal Conv1d (커널 크기: 4)
├─ Gated Delta Rule
│   ├─ Beta = sigmoid(b)
│   ├─ Gamma = -A × softplus(a + dt_bias)
│   └─ Attention 점수: e^(-Γ·δ)
└─ Output Projection (out_proj)

복잡도: O(n) - 선형 시간
```

### 6.2 GatedSoftmaxAttention 레이어

```
GatedSoftmaxAttention
├─ QKV Projection (linear_qgkv)
│   ├─ Query: 32 heads × 64 dim
│   ├─ Gate: 32 heads
│   └─ Key/Value: 2 groups × 128 dim (GQA)
├─ Q,K LayerNorm (안정성)
├─ RoPE (Rotary Position Embedding)
├─ Flash Attention (Scaled Dot-Product)
├─ Gate 적용: Output × sigmoid(Gate)
└─ Output Projection (linear_proj)

복잡도: O(n²) - 이차 시간
```

### 6.3 MoE 레이어

```
MoE Layer
├─ Router
│   ├─ Linear: 2048 → 128
│   ├─ Softmax (FP32)
│   └─ Top-8 선택
├─ 128 Routed Experts
│   └─ Expert FFN: Gate(2048→768) × Up(2048→768) → Down(768→2048)
├─ Shared Expert (항상 활성화)
└─ Load Balancing (aux_loss)

활성화 비율: 6.25% (8/128)
```

---

## 7. 파라미터 분석

### Baseline 48L 파라미터 분포

| 컴포넌트 | 레이어 수 | 레이어당 크기 | 총 크기 |
|---------|----------|-------------|--------|
| **Mamba** | 36 | ~8.9M | ~321M |
| **Attention** | 12 | ~17.8M | ~214M |
| **MoE** | 24 | ~1.2B | ~29B |
| **Embeddings** | 1 | ~311M | ~311M |
| **총합** | - | - | **~30B** |

---

## 8. 제약사항 및 호환성

### 8.1 TP=1 필수

> **중요**: Mamba 레이어의 순차적 상태 관리 특성으로 인해 **Tensor Parallelism > 1을 지원하지 않습니다**.

```yaml
# 올바른 설정
tensor_parallel: 1
expert_parallel: 8  # EP로 병렬화 보완
```

### 8.2 Pattern 검증 규칙

- 패턴 길이 = num_layers와 정확히 일치
- 유효한 문자: M, *, -, D만 허용
- Attention 비율 = 설정된 hybrid_attention_ratio와 일치

### 8.3 환경 요구사항

| 요구사항 | 버전 |
|---------|------|
| **Megatron-LM** | 250908 또는 251125 (Muon용) |
| **PyTorch** | ≥2.0 (2.3+ 권장) |
| **Transformer Engine** | ≥2.9.0 |
| **Flash Attention** | 2.x 또는 3.x |

---

## 9. 관련 파일 참조

### 핵심 구현 파일

| 파일 | 설명 |
|------|------|
| `examples/alpha/configs/model/baseline_48L.yaml` | 통합 모델 설정 |
| `megatron_patch/model/qwen3_next/gated_deltanet.py` | GatedDeltaNet 구현 |
| `megatron_patch/model/qwen3_next/gated_attention.py` | Attention 구현 |
| `examples/alpha/pretrain_alpha.py` | 학습 진입점 |

### 변환 도구

| 파일 | 설명 |
|------|------|
| `toolkits/distributed_checkpoints_convertor/impl/alpha/` | MG ↔ HF 변환기 |
| `examples/alpha/tools/alpha_config.py` | Config 자동 생성 도구 |

---

*마지막 업데이트: 2026-01-06*
