# Alpha 모델 변환 가이드

Megatron checkpoint를 HuggingFace 형식으로 변환하여 LM-Evaluation-Harness로 벤치마크 평가하는 가이드입니다.

---

## 목차

1. [개요](#개요)
2. [변환 프로세스](#변환-프로세스)
3. [사용 방법](#사용-방법)
4. [아키텍처 변경 시 업데이트](#아키텍처-변경-시-업데이트)
5. [트러블슈팅](#트러블슈팅)
6. [내부 동작 원리](#내부-동작-원리)

---

## 개요

### 변환이 필요한 이유

1. **벤치마크 평가**: LM-Evaluation-Harness는 HuggingFace 모델 형식을 요구
2. **생태계 호환성**: HF 생태계의 다양한 도구 활용 가능 (vLLM, TGI 등)
3. **배포 편의성**: 변환된 모델은 `transformers` 라이브러리로 바로 로딩

### 제한사항

- **TP=1 필수**: Qwen3-Next 변환은 현재 Tensor Parallelism 1만 지원
- **EP=8 권장**: Expert Parallelism은 8 유지 (256 experts / 8 = 32 experts per GPU)
- **Architecture consistency**: config.json과 weight shape이 정확히 일치해야 함

---

## 변환 프로세스

### 전체 워크플로우

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Megatron Checkpoint (학습 완료)                         │
│    outputs/alpha_*/checkpoints/iter_XXXXXX/                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 변환 스크립트 실행                                       │
│    bash scripts/convert_to_hf.sh <CKPT_DIR> <OUT_DIR>      │
│                                                             │
│    - Megatron args로 weight mapping                        │
│    - Reference config.json 복사                            │
│    - Weight 변환 및 reshaping                               │
│    - Safetensors 형식으로 저장                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Config 검증 및 수정                                      │
│    python scripts/validate_conversion_config.py \           │
│      <OUT_DIR> --fix                                        │
│                                                             │
│    - config.json을 Alpha 아키텍처에 맞게 자동 수정          │
│    - num_layers, num_experts, layer_types 등 업데이트       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. HuggingFace Model (변환 완료)                           │
│    hf_models/alpha_*/                                       │
│    ├── config.json                                          │
│    ├── model-00001-of-00008.safetensors                     │
│    ├── ...                                                  │
│    ├── tokenizer.json                                       │
│    └── tokenizer_config.json                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. 벤치마크 평가                                            │
│    bash scripts/run_benchmarks.sh <HF_MODEL_DIR>           │
└─────────────────────────────────────────────────────────────┘
```

---

## 사용 방법

### 1. 기본 사용법

```bash
cd examples/alpha

# Step 1: Megatron → HuggingFace 변환
bash scripts/convert_to_hf.sh \
  outputs/alpha_baseline_48L_20250117/checkpoints/iter_0010000 \
  hf_models/alpha_baseline_48L_iter10k

# Step 2: config.json 검증 및 수정
python scripts/validate_conversion_config.py \
  hf_models/alpha_baseline_48L_iter10k \
  --fix

# Step 3: HF 모델 로딩 테스트
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('hf_models/alpha_baseline_48L_iter10k')
print('✓ Model loaded successfully!')
"

# Step 4: 벤치마크 평가
bash scripts/run_benchmarks.sh hf_models/alpha_baseline_48L_iter10k
```

### 2. 통합 파이프라인 (원스텝)

```bash
# Megatron checkpoint → HF → 벤치마크 평가 (자동)
bash scripts/evaluate_checkpoint.sh \
  outputs/alpha_baseline_48L_20250117/checkpoints/iter_0010000
```

### 3. 커스텀 설정

```bash
# HF tokenizer 경로 직접 지정
bash scripts/convert_to_hf.sh \
  <MEGATRON_CKPT> \
  <HF_OUTPUT> \
  /custom/path/to/Qwen3-Next-tokenizer

# YAML config로부터 architecture 로드
python scripts/validate_conversion_config.py \
  hf_models/alpha_baseline_48L \
  --from-yaml configs/model/baseline_48L.yaml \
  --fix
```

---

## 아키텍처 변경 시 업데이트

Alpha 모델의 아키텍처가 변경될 때마다 다음 파일들을 업데이트해야 합니다.

### 1. 학습 Config 업데이트

**파일**: `examples/alpha/configs/model/*.yaml`

```yaml
model:
  num_layers: 32           # 24 → 32로 변경 예시
  hidden_size: 2048
  moe:
    num_experts: 512       # 256 → 512로 변경 예시
    router_topk: 10
  # ... 기타 설정
```

### 2. 변환 스크립트 업데이트

**파일**: `examples/alpha/scripts/convert_to_hf.sh`

스크립트 상단의 Configuration 섹션 수정:

```bash
#####################################
# Alpha Architecture Configuration
#####################################

ALPHA_NUM_LAYERS=32              # 업데이트!
ALPHA_NUM_EXPERTS=512            # 업데이트!
ALPHA_ROUTER_TOPK=10             # 업데이트!
# ... 기타 파라미터
```

### 3. Config Validation 스크립트 업데이트

**파일**: `examples/alpha/scripts/validate_conversion_config.py`

`ALPHA_ARCHITECTURE` 딕셔너리 수정:

```python
ALPHA_ARCHITECTURE = {
    "num_hidden_layers": 32,     # 업데이트!
    "num_experts": 512,          # 업데이트!
    "num_experts_per_tok": 10,   # 업데이트!
    # ... 기타 파라미터
}
```

### 4. Reference Config 업데이트

**파일**: `examples/alpha/reference_model/config.json`

```json
{
  "num_hidden_layers": 32,
  "num_experts": 512,
  "num_experts_per_tok": 10,
  ...
}
```

### 업데이트 체크리스트

변경 후 다음을 확인하세요:

- [ ] `configs/model/*.yaml` 업데이트
- [ ] `scripts/convert_to_hf.sh` Configuration 섹션 업데이트
- [ ] `scripts/validate_conversion_config.py` ALPHA_ARCHITECTURE 업데이트
- [ ] `reference_model/config.json` 업데이트
- [ ] Hybrid pattern 변경 시 `layer_types` 배열 길이 확인
- [ ] 변환 테스트 실행: `bash scripts/convert_to_hf.sh ...`
- [ ] Config 검증: `python scripts/validate_conversion_config.py ... --fix`

---

## 트러블슈팅

### Issue 1: TP > 1 에러

**증상**:
```
AssertionError: Currently MCore2HF conversion for Qwen3-Next is ONLY available with TP=1
```

**원인**: Qwen3-Next synchronizer가 TP > 1을 지원하지 않음

**해결**:
1. Alpha를 TP=1로 재학습, 또는
2. TP > 1 checkpoint를 TP=1로 재분할 (checkpoint resharding), 또는
3. Synchronizer 수정 (고급)

### Issue 2: Config Mismatch

**증상**:
```
RuntimeError: Error(s) in loading state_dict for Qwen3NextForCausalLM:
    size mismatch for model.layers.0.mlp.experts.0.gate_proj.weight
```

**원인**: config.json의 `num_experts`가 실제 weight와 불일치

**해결**:
```bash
python scripts/validate_conversion_config.py hf_models/alpha_* --fix
```

### Issue 3: Layer Types Mismatch

**증상**:
```
IndexError: list index out of range in layer_types
```

**원인**: `layer_types` 배열 길이가 `num_hidden_layers`와 불일치

**해결**:
1. Hybrid pattern 확인: `M-M-M-*-...` 토큰 개수 = num_layers
2. `validate_conversion_config.py`가 자동으로 수정

### Issue 4: Missing Tokenizer Files

**증상**:
```
FileNotFoundError: tokenizer.json not found
```

**원인**: HF reference directory에 tokenizer 파일 누락

**해결**:
```bash
# Tokenizer 파일 확인
ls /home/work/vidsearch/repos/project_s/Pai-Megatron-Patch/models/Qwen3-Next-tokenizer/

# 필요 파일: tokenizer.json, tokenizer_config.json, vocab.json, merges.txt
```

### Issue 5: CUDA OOM During Conversion

**증상**:
```
torch.cuda.OutOfMemoryError
```

**해결**:
1. GPU 개수 증가: `export KUBERNETES_CONTAINER_RESOURCE_GPU=8`
2. EP 증가: MODEL_PARALLEL_ARGS에서 `--expert-model-parallel-size 8` 확인
3. CPU 모드 사용: `convert_to_hf.sh` 수정 (`--use-gpu` 제거)

---

## 내부 동작 원리

### Weight Mapping 과정

Megatron과 HuggingFace의 weight layout이 다르므로 reshaping이 필요합니다:

**1. Embedding Layer**
```
Megatron: embedding.word_embeddings.weight [V, H]
     ↓ (COLUMN parallel 수집)
HF:      model.embed_tokens.weight [V, H]
```

**2. Attention Layer (GQA)**
```
Megatron:
  self_attention.linear_qkv.weight [H, (Q+K+V)*H_head]
     ↓ (QKV 분리 + GQA reshaping)
HF:
  self_attn.q_proj.weight [H, num_heads * head_dim]
  self_attn.k_proj.weight [H, num_kv_heads * head_dim]
  self_attn.v_proj.weight [H, num_kv_heads * head_dim]
```

**3. MoE Layer**
```
Megatron (EP=8):
  mlp.experts.local_experts[0...31].linear_fc1.weight  (각 GPU에 32 experts)
     ↓ (Expert parallel 수집 + reshaping)
HF:
  mlp.experts[0...255].gate_proj.weight  (전체 256 experts)
  mlp.experts[0...255].up_proj.weight
  mlp.experts[0...255].down_proj.weight
```

**4. Mamba Layer (Linear Attention)**
```
Megatron:
  linear_attn.{in_proj, conv1d, x_proj, dt_proj, out_proj}.weight
     ↓ (Direct mapping, Mamba-specific reshape)
HF:
  linear_attn.{in_proj, conv1d, x_proj, dt_proj, out_proj}.weight
```

### Config.json 생성 전략

**현재 방식 (Manual)**:
1. Reference config.json 복사 (Qwen3-Next tokenizer에서)
2. 사용자가 `validate_conversion_config.py`로 수동 수정

**Why?**
- Megatron checkpoint에는 config metadata가 부족
- `args`는 저장되지만 쉽게 접근 불가
- HF config는 더 많은 필드 필요 (layer_types, decoder_sparse_step 등)

**Alternative (자동화, 미구현)**:
1. Megatron checkpoint metadata에서 args 읽기
2. Megatron args → HF config 자동 변환 스크립트
3. Hybrid pattern → layer_types 자동 변환

### Synchronizer 역할

**`qwen3_next` Synchronizer** (`impl/qwen3_next/m2h_synchronizer.py`):

- `sync_params()`: 레이어별 반복하며 weight 복사
- `set_layer_state()`: Hybrid pattern 파싱하여 Mamba vs Attention 구분
- `set_mamba_layer_state()`: Mamba weight 특수 처리
- `set_selfattn_state()`: QKV reshaping (GQA, QK-layernorm 포함)
- `set_moe_layer_state()`: Expert weight EP aware 수집

Alpha는 이 synchronizer를 **재사용** (Qwen3-Next 기반이므로 weight layout 동일)

---

## 공식 Alpha Converter (신규)

**위치**: `toolkits/distributed_checkpoints_convertor/scripts/alpha/`

Alpha 전용 converter가 새로 추가되었습니다. Qwen3-Next converter를 기반으로 Alpha baseline_48L 설정에 맞게 최적화되었습니다.

### 빠른 시작

```bash
cd toolkits/distributed_checkpoints_convertor

# Megatron → HuggingFace
bash scripts/alpha/run_8xH20.sh \
  baseline_48L \
  /path/to/alpha-mcore-checkpoint \
  /path/to/alpha-hf-output \
  true \
  true \
  bf16 \
  /path/to/alpha-hf-reference
```

### 주요 특징

- **Alpha 전용 설정**: baseline_48L 하이퍼파라미터 내장
- **자동 config 복사**: HF reference에서 tokenizer 자동 복사
- **병렬화 최적화**: TP=1, EP=8 기본 설정
- **상세 문서**: `toolkits/distributed_checkpoints_convertor/scripts/alpha/README.md`

### 기존 방식과의 차이

| 항목 | 기존 (examples/alpha/scripts) | 신규 (toolkits/alpha) |
|------|-------------------------------|----------------------|
| 위치 | `examples/alpha/scripts/convert_to_hf.sh` | `toolkits/.../scripts/alpha/run_8xH20.sh` |
| 설정 | 스크립트 내 하드코딩 | MODEL_SIZE 분기 (확장 가능) |
| 검증 | 별도 validate 스크립트 필요 | `--debug` 플래그로 내장 |
| 문서 | CONVERSION.md | alpha/README.md |

### 권장 사항

- **신규 프로젝트**: `toolkits/alpha` converter 사용 (표준화됨)
- **기존 워크플로우**: 현재 스크립트 유지 (검증됨)

상세 사용법은 `toolkits/distributed_checkpoints_convertor/scripts/alpha/README.md` 참조.

---

## 참고 자료

- **Pai-Megatron-Patch**: [../../README.md](../../README.md)
- **Alpha Converter (신규)**: `/toolkits/distributed_checkpoints_convertor/scripts/alpha/`
- **Qwen3-Next 변환 스크립트**: `/toolkits/distributed_checkpoints_convertor/scripts/qwen3_next/`
- **HuggingFace Transformers**: https://github.com/huggingface/transformers
- **LM-Evaluation-Harness**: https://github.com/EleutherAI/lm-evaluation-harness

---

## FAQ

**Q: 변환 시간은 얼마나 걸리나요?**

A: 모델 크기와 GPU 수에 따라 다릅니다:
- Alpha 24L (256 experts, EP=8): ~10-20분 (8 GPUs)
- 96L (512 experts): ~30-60분

**Q: 변환된 모델 크기는?**

A: Alpha 24L (bf16):
- Megatron checkpoint (torch_dist): ~8GB
- HuggingFace (safetensors): ~8-10GB (비슷함)

**Q: 변환 후 다시 Megatron으로 되돌릴 수 있나요?**

A: 네, `convert_to_hf.sh`에서 `MG2HF=false`로 실행하면 HF → Megatron 변환 가능합니다.

**Q: 여러 iteration checkpoint를 한 번에 변환할 수 있나요?**

A: 스크립트는 한 번에 하나만 지원합니다. 여러 개는 loop로:
```bash
for iter in 1000 2000 5000 10000; do
  bash scripts/convert_to_hf.sh \
    outputs/alpha_*/checkpoints/iter_$(printf "%07d" $iter) \
    hf_models/alpha_iter${iter}
done
```

**Q: TP=1이 성능에 영향을 주나요?**

A: Alpha는 작은 모델(24L, 2048 hidden)이므로 TP=1로도 충분히 빠릅니다. 평가(inference)는 학습보다 memory bound가 적습니다.
