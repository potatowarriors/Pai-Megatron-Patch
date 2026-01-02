# Alpha Config Tool

통합 설정 파일에서 학습/변환/HF 설정을 자동 생성하는 도구입니다.

## 개요

이 도구는 **단일 YAML 파일**에서 모든 설정을 관리할 수 있게 해주어, 모델 구조 변경 시 수정해야 할 파일 수를 5-7개에서 **1개**로 줄입니다.

### 문제점 (기존)

| 파일 | 용도 | 수정 필요 시점 |
|------|------|---------------|
| `configs/model/baseline_48L.yaml` | 학습 설정 | 모델 구조 변경 시 |
| `toolkits/.../configs/baseline_48L.sh` | 변환 설정 | 모델 구조 변경 시 |
| `hfmodel/config.json` | HF 모델 설정 | 모델 구조 변경 시 |

### 해결책 (통합 Config)

```
examples/alpha/configs/model/baseline_48L.yaml  (Single Source of Truth)
                    ↓
           python alpha_config.py sync baseline_48L
                    ↓
    ┌───────────────┴───────────────┐
    ↓                               ↓
변환 설정 자동 생성           HF config 자동 생성
(baseline_48L.sh)            (config.json)
```

## 사용법

### 1. 설정 검증

```bash
python tools/alpha_config.py validate baseline_48L
```

출력:
```
✅ Validation PASSED for baseline_48L

📊 Model Summary:
  - Megatron layers: 48
  - HuggingFace layers: 24
  - Pattern: MDM-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-
  - Mamba layers: 18 (37.5%)
  - Attention layers: 6 (12.5%)
  - MLP layers: 24 (50.0%)
  - Experts: 128 (top-8)
```

### 2. 변환 설정 동기화

```bash
python tools/alpha_config.py sync baseline_48L
```

이 명령은 `configs/model/baseline_48L.yaml`에서 읽어 `toolkits/.../configs/baseline_48L.sh`를 자동 생성합니다.

### 3. HF config.json 생성

```bash
# 표준 출력
python tools/alpha_config.py generate-hf-config baseline_48L

# 파일로 저장
python tools/alpha_config.py generate-hf-config baseline_48L --output /path/to/config.json
```

### 4. 학습 인자 생성

```bash
python tools/alpha_config.py generate-train-args baseline_48L
```

## 새 모델 크기 추가하기

### Step 1: YAML 파일 생성

`examples/alpha/configs/model/` 디렉토리에 새 YAML 파일을 생성합니다:

```yaml
# examples/alpha/configs/model/baseline_48L.yaml
model:
  name: "alpha-baseline-48L"
  architecture: "qwen3_next_mamba_hybrid"

  # 핵심 아키텍처
  num_layers: 48  # 48 Megatron layers → 24 HF layers
  hidden_size: 4096
  ffn_hidden_size: 10240

  # Attention
  num_attention_heads: 64
  kv_channels: 128
  group_query_attention: true
  num_query_groups: 4

  # Hybrid 패턴
  hybrid:
    attention_ratio: 0.125  # 12.5% = 6 attention layers
    mlp_ratio: 0.5          # 50% = 24 MLP layers
    override_pattern: null  # 자동 생성 (또는 명시적 패턴)
    mamba_state_dim: 128
    mamba_head_dim: 128
    mamba_num_groups: 16
    mamba_num_heads: 32

  # MoE
  moe:
    num_experts: 512
    moe_ffn_hidden_size: 1024
    router_topk: 8
    ...
```

### Step 2: 검증 및 동기화

```bash
# 검증
python tools/alpha_config.py validate baseline_48L

# 동기화 (변환 스크립트 자동 생성)
python tools/alpha_config.py sync baseline_48L
```

### Step 3: MG2HF 변환 (HF config 자동 생성)

```bash
# HF_DIR 없이 실행하면 config.json이 자동 생성됨
bash run_8xH20.sh baseline_48L /path/to/mcore /path/to/hf true true bf16
```

## 패턴 문법

| 문자 | 의미 | 설명 |
|------|------|------|
| `M` | Mamba | GatedDeltaNet (Linear Attention) 레이어 |
| `*` | Attention | Full Multi-Head Attention 레이어 |
| `-` | MLP | MLP-only 레이어 |

### 예시

```
M-M-M-*-M-M-M-*-M-M-M-*-  (24 layers)
│ │ │ │ │ │ │ │ │ │ │ │
│ │ │ └─Attention
│ │ └─MLP
│ └─MLP
└─Mamba
```

### 자동 생성 vs 명시적 패턴

```yaml
# 자동 생성 (attention_ratio, mlp_ratio에서 계산)
hybrid:
  attention_ratio: 0.125
  mlp_ratio: 0.5
  override_pattern: null  # 또는 생략

# 명시적 패턴 (정확한 제어 필요 시)
hybrid:
  attention_ratio: 0.125  # 검증용
  mlp_ratio: 0.5          # 검증용
  override_pattern: "M-M-M-*-M-M-M-*-M-M-M-*-"
```

## 명령어 레퍼런스

| 명령어 | 설명 |
|--------|------|
| `validate <config>` | 설정 검증 및 요약 표시 |
| `sync <config>` | 변환 스크립트 자동 생성 |
| `generate-train-args <config>` | Megatron 학습 인자 생성 |
| `generate-convert-args <config>` | 변환 인자 생성 |
| `generate-hf-config <config>` | HuggingFace config.json 생성 |
| `generate-convert-script <config>` | 변환 bash 스크립트 생성 |

## 예상 효과

| 지표 | 기존 | 개선 후 |
|------|------|---------|
| 모델 크기 추가 시 수정 파일 | 5-7개 | **1개** |
| 설정 불일치 에러 | 자주 | 거의 없음 |
| 새 모델 추가 시간 | 2-3시간 | **15분** |
| 변환 전 검증 | 불가능 | 자동화 |
