# Alpha 코드 리팩토링 가이드

이 문서는 2025-11-29에 수행된 Alpha 프로젝트 코드 리팩토링의 변경 사항을 정리합니다.

## 요약

| Phase | 항목 | 심각도 | 상태 |
|-------|------|--------|------|
| 1.1 | train.sh TIMESTAMP 버그 | High | 완료 |
| 1.2 | train.sh INFRA_ARGS 버그 | High | 완료 |
| 1.3 | baseline_24L.yaml 절대 경로 | Medium | 완료 |
| 1.4 | kormo_1pct.yaml 절대 경로 | Medium | 완료 |
| 1.5 | h2m_synchronizer.py 검증 누락 | Medium | 완료 |
| 1.6 | gated_attention.py bare except | Medium | 완료 |
| 2.1 | yaml_get 함수 변수 인용 | Low | 완료 |
| 2.2 | Synchronizer 공통 코드 추출 | Low | 완료 |
| 2.3 | alpha_config.py 패턴 생성 통합 | Low | 완료 |
| 2.4 | alpha_config.py 토큰 ID dataclass | Low | 완료 |

---

## Phase 1: 즉시 수정 (High/Medium Severity)

### 1.1 train.sh TIMESTAMP 정의 순서 버그

**문제**: TIMESTAMP 변수가 Line 384에서 정의되기 전에 Line 270 (WANDB 섹션)에서 사용됨

**수정**: TIMESTAMP 정의를 Line 260으로 이동 (WANDB 섹션 전)

```bash
# 수정 전 (Line 270)
RUN_NAME="alpha_${MODEL_CONFIG}_${TIMESTAMP}"  # TIMESTAMP 미정의!

# 수정 후 (Line 260)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# WANDB 설정 (Line 266+)
if [ "$WANDB_ENABLED" = "True" ] || [ "$WANDB_ENABLED" = "true" ]; then
    RUN_NAME="alpha_${MODEL_CONFIG}_${TIMESTAMP}"  # 이제 정상 동작
```

**파일**: [examples/alpha/train.sh](../train.sh)

---

### 1.2 train.sh INFRA_ARGS 재정의 버그

**문제**: Line 627에서 `INFRA_ARGS=(...)`가 이전에 설정된 값을 덮어씀

**수정**: 기본 정의를 먼저 하고, activation checkpointing 등은 `+=` 패턴 사용

```bash
# 수정 전 (Line 627)
INFRA_ARGS=(...)  # 이전 설정 덮어쓰기!

# 수정 후 (Line 613-636)
# 기본 정의
INFRA_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    ...
)

# Activation Checkpointing 추가 (Line 638+)
if [ ! -z "$AC_GRANULARITY" ]; then
    INFRA_ARGS+=(--recompute-granularity ${AC_GRANULARITY})  # += 패턴
    ...
fi
```

**파일**: [examples/alpha/train.sh](../train.sh)

---

### 1.3 baseline_24L.yaml 절대 경로 제거

**문제**: tokenizer_path가 하드코딩된 절대 경로 사용

**수정**: 환경 변수 기반 경로로 변경

```yaml
# 수정 전
tokenizer_path: "/home/work/models/Qwen3-Next-tokenizer"

# 수정 후
tokenizer_path: "${ALPHA_TOKENIZER_PATH:-${MEGATRON_PATCH_PATH}/models/Qwen3-Next-tokenizer}"
```

**환경 변수 설정**:
```bash
export ALPHA_TOKENIZER_PATH=/your/path/to/tokenizer
# 또는
export MEGATRON_PATCH_PATH=/your/path/to/Pai-Megatron-Patch
```

**파일**: [examples/alpha/configs/model/baseline_24L.yaml](../configs/model/baseline_24L.yaml)

---

### 1.4 kormo_1pct.yaml 절대 경로 제거

**문제**: train_path가 하드코딩된 절대 경로 사용

**수정**: 환경 변수 기반 경로로 변경

```yaml
# 수정 전
train_path: "/home/work/Datasets/KORMo_processed/mmap/qwen3_1pct/kormo_content_document"

# 수정 후
train_path: "${ALPHA_DATA_PATH:-/home/work/Datasets/KORMo_processed/mmap/qwen3_1pct}/kormo_content_document"
```

**환경 변수 설정**:
```bash
export ALPHA_DATA_PATH=/your/path/to/data
```

**파일**: [examples/alpha/configs/data/kormo_1pct.yaml](../configs/data/kormo_1pct.yaml)

---

### 1.5 h2m_synchronizer.py 검증 메서드 누락

**문제**: HF2MG 변환기에 `_validate_conversion_config()` 메서드 누락 (MG2HF에는 있음)

**수정**: m2h_synchronizer.py와 동일한 검증 로직 추가

```python
# 추가된 메서드
def _validate_conversion_config(self):
    """Validate Alpha conversion configuration to catch errors early."""
    validate_hybrid_pattern(
        layout=self.layout,
        num_layers=self.args.num_layers,
        hybrid_attention_ratio=self.args.hybrid_attention_ratio,
        rank=self.rank
    )

    log_conversion_summary(
        direction="HF2MG",
        layout=self.layout,
        args=self.args,
        tp_size=self.tp_size,
        pp_size=self.pp_size,
        ep_size=self.ep_size,
        rank=self.rank
    )
```

**파일**: [toolkits/distributed_checkpoints_convertor/impl/alpha/h2m_synchronizer.py](../../../../toolkits/distributed_checkpoints_convertor/impl/alpha/h2m_synchronizer.py)

---

### 1.6 gated_attention.py bare except 수정

**문제**: Line 56에서 `except:` (bare except) 사용

**수정**: 구체적인 예외 타입 지정

```python
# 수정 전
try:
    from flash_attn_interface import flash_attn_func as fa3
    HAVE_FA3 = True
except:
    HAVE_FA3 = False

# 수정 후
try:
    from flash_attn_interface import flash_attn_func as fa3
    HAVE_FA3 = True
except (ImportError, ModuleNotFoundError):
    HAVE_FA3 = False
```

**파일**: [megatron_patch/model/qwen3_next/gated_attention.py](../../../../megatron_patch/model/qwen3_next/gated_attention.py)

---

## Phase 2: 단기 개선 (Low Severity)

### 2.1 yaml_get 함수 변수 인용 추가

**문제**: 변수 인용 없이 사용, dict 타입 체크 누락

**수정**: 변수 인용 추가 및 isinstance 체크

```bash
# 수정 전
yaml_get() {
    python3 -c "
...
for k in '$2'.split('.'):
    value = value.get(k, '')
...
"
}

# 수정 후
yaml_get() {
    local file="$1"
    local key="$2"
    python3 -c "
import yaml
with open('${file}', 'r') as f:
    data = yaml.safe_load(f)
keys = '${key}'.split('.')
value = data
for k in keys:
    if isinstance(value, dict):
        value = value.get(k, '')
    else:
        value = ''
        break
print(value if value != '' else '')
" 2>/dev/null || echo ""
}
```

**파일**: [examples/alpha/train.sh](../train.sh)

---

### 2.2 Synchronizer 공통 코드 추출

**문제**: m2h_synchronizer.py와 h2m_synchronizer.py에 중복 코드 존재

**수정**: 공통 모듈 `common.py` 생성

**새 파일**: [toolkits/distributed_checkpoints_convertor/impl/alpha/common.py](../../../../toolkits/distributed_checkpoints_convertor/impl/alpha/common.py)

```python
# 패턴 문자 상수
CHAR_MAMBA = 'M'      # GatedDeltaNet layer
CHAR_ATTENTION = '*'  # Full Attention layer
CHAR_MLP = '-'        # MLP-only layer

# 공통 함수
def validate_hybrid_pattern(layout, num_layers, hybrid_attention_ratio, rank=0):
    """Validate Alpha hybrid model pattern configuration."""
    ...

def log_conversion_summary(direction, layout, args, tp_size, pp_size, ep_size, rank=0, dp_info=None):
    """Log conversion configuration summary."""
    ...

def build_pipeline_parallel_mapping(num_layers, pp_size, pp_rank):
    """Build mapping from local layer index to global layer index."""
    ...
```

**사용법** (두 synchronizer에서):
```python
from .common import (
    validate_hybrid_pattern,
    log_conversion_summary,
    build_pipeline_parallel_mapping,
    CHAR_MAMBA, CHAR_ATTENTION, CHAR_MLP
)
```

---

### 2.3 alpha_config.py 패턴 생성 함수 통합

**문제**: 5개 이상의 함수에서 동일한 패턴 생성 코드 중복

**수정**: `ModelConfig.get_pattern()` 메서드 추가 (캐싱 포함)

```python
@dataclass
class ModelConfig:
    ...
    # Cached pattern (computed lazily)
    _pattern_cache: Optional[str] = field(default=None, repr=False)

    def get_pattern(self) -> str:
        """Get the hybrid override pattern (cached)."""
        if self._pattern_cache is None:
            if self.hybrid.override_pattern:
                self._pattern_cache = self.hybrid.override_pattern
            else:
                self._pattern_cache = generate_pattern(
                    self.num_layers, self.hybrid.attention_ratio, self.hybrid.mlp_ratio
                )
        return self._pattern_cache
```

**업데이트된 함수들**:
- `validate_config()` - `pattern = config.get_pattern()`
- `generate_train_args()` - `pattern = config.get_pattern()`
- `generate_convert_args()` - `pattern = config.get_pattern()`
- `generate_hf_config()` - `pattern = config.get_pattern()`
- `generate_convert_script()` - `pattern = config.get_pattern()`
- `cmd_validate()` - `pattern = config.get_pattern()`

**파일**: [examples/alpha/tools/alpha_config.py](../tools/alpha_config.py)

---

### 2.4 alpha_config.py 토큰 ID dataclass

**문제**: bos_token_id, eos_token_id가 여러 곳에 하드코딩 (151643, 151645)

**수정**: TokenConfig dataclass 및 상수 추가

```python
# 상수 정의
DEFAULT_BOS_TOKEN_ID = 151643
DEFAULT_EOS_TOKEN_ID = 151645

# TokenConfig dataclass
@dataclass
class TokenConfig:
    """Token ID configuration for tokenizer."""
    bos_token_id: int = DEFAULT_BOS_TOKEN_ID
    eos_token_id: int = DEFAULT_EOS_TOKEN_ID
    pad_token_id: Optional[int] = None

# ModelConfig에 추가
@dataclass
class ModelConfig:
    ...
    tokens: TokenConfig = field(default_factory=TokenConfig)
```

**사용법** (generate_hf_config에서):
```python
# 수정 전
"bos_token_id": 151643,
"eos_token_id": 151645,

# 수정 후
"bos_token_id": config.tokens.bos_token_id,
"eos_token_id": config.tokens.eos_token_id,
```

**YAML 설정** (선택적):
```yaml
model:
  tokens:
    bos_token_id: 151643
    eos_token_id: 151645
    pad_token_id: 151645  # optional
```

**파일**: [examples/alpha/tools/alpha_config.py](../tools/alpha_config.py)

---

## 파일 변경 요약

| 파일 | 변경 유형 | Phase |
|------|-----------|-------|
| `examples/alpha/train.sh` | 수정 | 1.1, 1.2, 2.1 |
| `examples/alpha/configs/model/baseline_24L.yaml` | 수정 | 1.3 |
| `examples/alpha/configs/data/kormo_1pct.yaml` | 수정 | 1.4 |
| `toolkits/.../impl/alpha/h2m_synchronizer.py` | 수정 | 1.5, 2.2 |
| `toolkits/.../impl/alpha/m2h_synchronizer.py` | 수정 | 2.2 |
| `toolkits/.../impl/alpha/common.py` | 신규 | 2.2 |
| `megatron_patch/model/qwen3_next/gated_attention.py` | 수정 | 1.6 |
| `examples/alpha/tools/alpha_config.py` | 수정 | 2.3, 2.4 |

---

## 환경 변수 요약

리팩토링 후 사용 가능한 환경 변수:

| 환경 변수 | 용도 | 기본값 |
|-----------|------|--------|
| `ALPHA_TOKENIZER_PATH` | 토크나이저 경로 | `${MEGATRON_PATCH_PATH}/models/Qwen3-Next-tokenizer` |
| `ALPHA_DATA_PATH` | 데이터셋 경로 | `/home/work/Datasets/KORMo_processed/mmap/qwen3_1pct` |
| `MEGATRON_PATCH_PATH` | Pai-Megatron-Patch 루트 | (필수 설정) |

**설정 예시**:
```bash
export MEGATRON_PATCH_PATH=/home/work/Pai-Megatron-Patch
export ALPHA_TOKENIZER_PATH=/data/models/Qwen3-Next-tokenizer
export ALPHA_DATA_PATH=/data/datasets/kormo
```

---

## 테스트 방법

### alpha_config.py 검증
```bash
cd examples/alpha/tools
python alpha_config.py validate baseline_24L
```

### HF config 생성 테스트
```bash
python alpha_config.py generate-hf-config baseline_24L
```

### 변환 스크립트 테스트
```bash
cd toolkits/distributed_checkpoints_convertor/scripts/alpha
bash run_8xH20.sh baseline_24L /path/to/mcore /path/to/hf true true bf16
```

---

## 관련 문서

- [CONVERSION.md](CONVERSION.md) - 체크포인트 변환 상세 가이드
- [ARCHITECTURE.md](ARCHITECTURE.md) - Alpha 모델 아키텍처
- [SETUP.md](SETUP.md) - 환경 설정 가이드
