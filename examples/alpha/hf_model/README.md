# Alpha HuggingFace Model

Alpha 모델의 HuggingFace 호환 구현체입니다.

## 개요

이 디렉토리는 Alpha 모델의 로컬 HuggingFace 구현을 포함합니다. `transformers` 라이브러리에 의존하지 않고 버전 관리와 커스터마이징이 가능합니다.

### 왜 로컬 구현이 필요한가?

1. **버전 독립성**: `transformers` 업그레이드 시 호환성 문제 방지
2. **커스터마이징**: Alpha 전용 기능 추가 가능
3. **오프라인 사용**: `transformers`에 등록되지 않은 모델도 로드 가능
4. **Git 추적**: 모델 정의 변경 이력 관리

## 파일 구조

```
hf_model/
├── __init__.py               # HF 동적 로딩 설정
├── configuration_alpha.py    # AlphaConfig 클래스
├── modeling_alpha.py         # AlphaForCausalLM 클래스
└── README.md                 # 이 파일
```

## 사용법

### 변환된 모델 로드

```python
from transformers import AutoModelForCausalLM, AutoConfig

# trust_remote_code=True 필수
model = AutoModelForCausalLM.from_pretrained(
    "path/to/alpha/hfmodel",
    trust_remote_code=True,
    device_map="auto"
)

config = AutoConfig.from_pretrained(
    "path/to/alpha/hfmodel",
    trust_remote_code=True
)
```

### 직접 클래스 사용

```python
import sys
sys.path.append("/path/to/Pai-Megatron-Patch/examples/alpha")

from hf_model import AlphaConfig, AlphaForCausalLM

config = AlphaConfig(
    hidden_size=2048,
    num_hidden_layers=24,
    num_attention_heads=32,
    # ...
)

model = AlphaForCausalLM(config)
```

## 아키텍처

Alpha는 Qwen3-Next Mamba Hybrid 아키텍처 기반입니다:

- **GatedDeltaNet**: ICLR 2025 Linear Attention (O(n) complexity)
- **Full Attention**: 표준 Multi-Head Attention (매 4번째 레이어)
- **MoE**: Mixture-of-Experts with Shared Expert
- **2:1 레이어 매핑**: 48 MG layers → 24 HF layers

### 주요 클래스

| 클래스 | 설명 |
|--------|------|
| `AlphaConfig` | 모델 설정 클래스 |
| `AlphaForCausalLM` | Causal LM 모델 |
| `AlphaModel` | 기본 트랜스포머 모델 |
| `AlphaGatedDeltaNet` | Linear Attention 레이어 |
| `AlphaAttention` | Full Attention 레이어 |
| `AlphaSparseMoeBlock` | MoE 레이어 |

## config.json 구조

변환 시 자동 생성되는 `config.json`에는 다음 필드가 포함됩니다:

```json
{
    "architectures": ["AlphaForCausalLM"],
    "model_type": "alpha",
    "auto_map": {
        "AutoConfig": "configuration_alpha.AlphaConfig",
        "AutoModelForCausalLM": "modeling_alpha.AlphaForCausalLM"
    },
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 32,
    "num_experts": 128,
    ...
}
```

`auto_map` 필드는 `trust_remote_code=True`로 로드 시 로컬 Python 파일에서 클래스를 찾도록 합니다.

## 변환 워크플로우

```
Megatron 체크포인트
        │
        ↓
run_8xH20.sh (MG→HF 변환)
        │
        ├─ 1. torchrun convert.py (가중치 변환)
        │
        ├─ 2. alpha_config.py generate-hf-config (config.json 생성)
        │
        └─ 3. cp hf_model/*.py ${SAVE_DIR}/ (모델링 파일 복사)
        │
        ↓
HuggingFace 모델 디렉토리
├── config.json
├── __init__.py
├── configuration_alpha.py
├── modeling_alpha.py
├── model-*.safetensors
└── tokenizer*.json
```

## 관련 파일

- **통합 설정**: `examples/alpha/configs/model/baseline_48L.yaml`
- **변환 스크립트**: `toolkits/.../scripts/alpha/run_8xH20.sh`
- **Config 도구**: `examples/alpha/tools/alpha_config.py`

## 주의사항

1. **trust_remote_code**: HF 모델 로드 시 반드시 `trust_remote_code=True` 필요
2. **Python 파일 복사**: 변환 후 출력 디렉토리에 `.py` 파일이 있어야 함
3. **버전 관리**: 이 디렉토리의 파일을 수정하면 변환된 모델에도 반영됨

## 라이선스

Apache License 2.0
