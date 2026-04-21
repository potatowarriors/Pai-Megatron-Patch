# Muon CLIP (QK-Clip) 활성화를 위한 환경 분석 보고서

> 작성일: 2026-02-10 (2026-02-19 업데이트: cuDNN 단일화 완료)
> 대상: Alpha Stage 2 학습에서 Muon CLIP 활성화 가능성 분석

## Context

Alpha Stage 2 학습에서 Muon CLIP (QK-Clip)을 활성화하려면 TE의 `return_max_logit` 기능이 필요합니다.
[PR #2195 (NVIDIA/TransformerEngine)](https://github.com/NVIDIA/TransformerEngine/pull/2195)가 이 기능을 TE 2.9.0에 추가했지만, 현재 환경에서는 cuDNN 버전 부족으로 실제 사용이 불가능한 상태입니다.

---

## 1. 현재 환경 상태

| 구성 요소 | 현재 버전 | Muon CLIP 요구 | 상태 |
|-----------|-----------|---------------|------|
| TransformerEngine | 2.9.0+70f53666 | >= 2.9.0 | OK |
| cuDNN | **9.19.0** (pip only) | >= 9.15.1 (SDPA 최적화) | **OK** |
| cudnn-frontend | **1.18.0** | >= 1.16 | **OK** |
| CUDA | 12.9 | 12.x | OK |
| PyTorch | 2.8.0a0 | >= 2.0 | OK |
| Megatron-LM | 251125 | 251125 | OK |
| Flash Attention | 2.7.3 (v3) | - | OK (FusedAttn으로 전환 예정) |

### 패키지 설치 경로
- cuDNN: `nvidia-cudnn-cu12` (9.19.0, pip) — 시스템 apt cuDNN (9.10.1) 제거 완료
- cudnn-frontend: `nvidia-cudnn-frontend` (1.18.0, pip)
- linker 설정: `/etc/ld.so.conf.d/cudnn-pip.conf` → pip cuDNN lib 경로 등록

---

## 2. PR #2195 분석: "[PyTorch] Add max_logit support for MuonClip"

- **상태**: 2025-10-25 머지됨 (TE 2.9.0 타겟)
- **목적**: Attention forward pass에서 `max(Q * K^T * scale + bias)` 값을 head별로 반환
- **지원 Backend**: FusedAttention (cuDNN), UnfusedAttention
- **미지원**: Flash Attention (명시적으로 제거됨: "remove logic for flash-attn")
- **cuDNN 요구**: >= 9.15.1 (SDPA 성능 최적화) + cudnn-frontend >= 1.16

### Muon CLIP 동작 원리

```
[1] qk_clip=true 설정
    |
[2] TEDotProductAttention.__init__() -> extra_kwargs["return_max_logit"] = True
    (backends/megatron/Megatron-LM-251125/megatron/core/extensions/transformer_engine.py:1028)
    |
[3] Forward pass: TE attention이 (output, max_logit) 튜플 반환
    (transformer_engine.py:1125)
    |
[4] max_logit을 current_max_attn_logits에 누적 (배치 내 최대값)
    (transformer_engine.py:1128-1133)
    |
[5] Optimizer step 후, clip_qk() 호출
    (backends/megatron/Megatron-LM-251125/megatron/core/optimizer/qk_clip.py:8-39)
    |
[6] DP group 간 all_reduce로 글로벌 max 계산
    (qk_clip.py:25-29)
    |
[7] threshold 초과 시 Q/K weight 스케일링
    eta = threshold / max_logit
    weight_q *= eta^alpha,  weight_k *= eta^(1-alpha)
    (backends/megatron/Megatron-LM-251125/megatron/core/transformer/attention.py:1280-1299)
```

---

## 3. 백엔드 선택 로직 실험 결과 (실제 테스트)

### 3-1. FusedAttention 지원 범위 (cuDNN 9.10.1)

| head_dim | seq_len | FusedAttention | sub-backend |
|----------|---------|:--------------:|-------------|
| 64 | 512 | O | F16_max512_seqlen |
| 64 | 4096 | X | - |
| 96 | 512 | X | - |
| 128 | 512 | X | - |
| 128 | 4096 | X | - |

**결론**: cuDNN 9.10.1에서는 `F16_max512_seqlen` sub-backend만 사용 가능 (head<=64, seq<=512).
Alpha의 head_dim=128, seq_len=4096에서는 **FusedAttention 자체가 작동하지 않음**.

### 3-2. return_max_logit 지원 (cuDNN 9.10.1)

| 설정 | Flash | Fused | Unfused |
|------|:-----:|:-----:|:-------:|
| `return_max_logit=False` | **선택됨** | X | 가능 |
| `return_max_logit=True` | X (강제 비활성) | X | **선택됨** |

**결론**: `return_max_logit=True` 시 Flash는 강제 비활성, Fused도 cuDNN 부족으로 불가 -> **Unfused만 사용 가능 (극심한 성능 저하)**

### 3-3. cuDNN 업그레이드 후 예상 (>= 9.15.1, SDPA 최적화 포함)

| 설정 | Flash | Fused | Unfused |
|------|:-----:|:-----:|:-------:|
| `return_max_logit=False` | 가능 | **F16_arbitrary_seqlen** | 가능 |
| `return_max_logit=True` | X (강제 비활성) | **F16_arbitrary_seqlen** | 가능 |

cuDNN >= 9.15.1이면 `F16_arbitrary_seqlen` sub-backend가 활성화되어 head_dim=128 + return_max_logit 조합이 가능해짐. 9.15.1에서 SDPA 커널 성능이 크게 최적화되어 FA3에 근접한 throughput 달성.

### 3-4. 백엔드 선택 핵심 코드

TE의 `get_attention_backend()` 함수에서 `return_max_logit=True` 시:
- FlashAttention: **무조건 비활성** (TE 소스 line 511-513)
- FusedAttention (THD format): 비활성 (line 515-516)
- FusedAttention (BSHD format): `nvte_get_fused_attn_backend()`에 `return_max_logit` 전달 -> cuDNN이 지원하면 활성화
- Hopper(sm90)에서는 FusedAttention이 FlashAttention보다 우선 선택됨 (line 1119-1124)

---

## 4. Alpha 하이브리드 아키텍처와의 호환성

### FusedAttention은 GatedDeltaNet과 충돌하는가?

**아니요.** FusedAttention은 표준 Attention 레이어(`*`)에만 적용됩니다.

```
M D M - M - * - M M M - * - M M M - * - M M M - M M M - * - M M M - * - M M M - * -
                ^                ^                ^                    ^                ^                ^
          이 6개 레이어만 FusedAttention 영향받음
```

- **GatedDeltaNet (M)**: SSM 기반 linear attention -> TE attention 사용 안함
- **MoE MLP (-)**: FFN 레이어 -> attention과 무관
- **Dense MLP (D)**: 일반 FFN -> attention과 무관
- **Attention (*)**: 48개 중 6개 -> 이것만 FA3/FusedAttn/Unfused 선택에 영향받음

따라서 cuDNN 업그레이드 후 `NVTE_FUSED_ATTN=1, NVTE_FLASH_ATTN=0`으로 전환하면
6개 attention 레이어만 FusedAttention으로 전환되고, 나머지 42개 레이어는 변경 없음.

---

## 5. cuDNN 단일화 방법 (적용 완료: 2026-02-19)

### 문제: apt/pip cuDNN 공존 시 TE blocklist 적용

시스템에 apt cuDNN(9.10.1)과 pip cuDNN(9.19.0)이 공존하면, TE의 C++ 코드가
시스템 linker를 통해 apt 버전을 로드합니다. cuDNN 9.10.0/9.10.1은 TE `fused_attn.cpp`에서
명시적으로 blocklist 되어 FusedAttention이 비활성화됩니다.

### 해결: apt cuDNN 제거 + pip cuDNN 시스템 등록

```bash
# 1. 시스템 cuDNN 제거
sudo apt remove --purge libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-headers-cuda-12

# 2. pip cuDNN을 시스템 linker에 등록
echo "/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib" | sudo tee /etc/ld.so.conf.d/cudnn-pip.conf

# 3. CUDA 디렉토리의 dangling symlink 교체 (TE _load_cudnn()이 이 경로를 시도)
sudo ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 /usr/local/cuda/lib64/libcudnn.so
sudo ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/libcudnn.so.9 /usr/local/cuda/targets/x86_64-linux/lib/libcudnn.so

# 4. linker 캐시 갱신
sudo ldconfig

# 5. 검증
python3 -c "import torch; print(f'cuDNN: {torch.backends.cudnn.version()}')"   # 91900
ldconfig -p | grep cudnn   # pip 경로만 표시
```

### 롤백 방법
```bash
sudo apt install libcudnn9-cuda-12
sudo rm /etc/ld.so.conf.d/cudnn-pip.conf
sudo ldconfig
```

### 설치 스크립트에 자동화 반영
`setup_pai_megatron_env_with_deepep.sh`의 Step 4에서 자동으로 수행됩니다:
- apt cuDNN 버전이 blocklist(9.10.0/9.10.1)인 경우만 조건부 제거
- pip cuDNN 경로를 `/etc/ld.so.conf.d/cudnn-pip.conf`에 등록
- CUDA 디렉토리 symlink 교체 + `ldconfig`

### 검증 코드
```python
import os
os.environ["NVTE_FUSED_ATTN"] = "1"
os.environ["NVTE_FLASH_ATTN"] = "0"
from transformer_engine.pytorch.attention.dot_product_attention.utils import get_attention_backend, AttentionParams
import torch

params = AttentionParams(
    qkv_type=torch.Tensor, qkv_dtype=torch.bfloat16,
    qkv_layout="bshd_bshd_bshd", batch_size=1,
    num_heads=32, num_gqa_groups=2,
    max_seqlen_q=4096, max_seqlen_kv=4096,
    head_dim_qk=128, head_dim_v=128,
    attn_mask_type="causal", window_size=(-1, -1),
    alibi_slopes_shape=None,
    core_attention_bias_type="no_bias", core_attention_bias_shape=None,
    core_attention_bias_requires_grad=False,
    attention_dropout=0.0, is_training=True, pad_between_seqs=False,
    context_parallel=False, deterministic=False,
    fp8=False, fp8_meta=None, inference_params=None,
    softmax_type="vanilla", return_max_logit=True,
)
result = get_attention_backend(params)
_, _, use_fused, fused_backend, _, _ = result
assert use_fused, f"FusedAttention not available! backend={fused_backend}"
print(f"SUCCESS: FusedAttention + return_max_logit works! backend={fused_backend}")
```

---

## 6. FA3 max_logit 이슈 현황

**이슈**: [Dao-AILab/flash-attention#1876](https://github.com/Dao-AILab/flash-attention/issues/1876) - "Per Head Max-Attention-Logits"

| 항목 | 상태 |
|------|------|
| 이슈 상태 | **Open** (2025-09-09 생성) |
| 구현 PR | 없음 |
| 타임라인 | 없음 |

**Tri Dao의 답변** (2025-09-10):
- Flash Attention의 forward pass 마지막에 이미 `softmax.row_max`와 `row_sum`이 계산됨
- 이 `row_max`를 LSE(Log-Sum-Exp)와 같은 방식으로 global memory에 write하면 됨
- Hopper 커널의 line 444 참조
- 단, 이 값은 `softmax_scale` (1/sqrt(d)) 적용 **전**의 값임에 주의

**해석**: 기술적 경로는 제시되었으나 공식 구현 계획은 없음. FA3에서 max_logit이 지원되면
FA3 + Muon CLIP 조합이 가능해져 FusedAttention 전환 없이도 사용 가능.

---

## 7. 종합 결론

### 완료 상태 (2026-02-19)

cuDNN 단일화가 완료되어 Muon CLIP 활성화의 모든 기술적 조건이 충족되었습니다.

```
cuDNN 9.19.0 (pip only, apt 제거 완료, >= 9.15.1 SDPA 최적화 포함)
  -> F16_arbitrary_seqlen sub-backend 활성화 ✓ (검증 완료)
    -> FusedAttention + return_max_logit=True 조합 가능 ✓
      -> SDPA 성능 최적화 (9.15.1+) ✓
        -> qk_clip: true 설정 가능 ✓
          -> Muon CLIP 활성화 준비 완료
```

### Stage 2 Muon CLIP 활성화 체크리스트:

1. ~~**cuDNN 업그레이드**~~ → 9.19.0 (pip, >= 9.15.1 SDPA 최적화 포함), apt cuDNN 제거 완료
2. ~~**cudnn-frontend 업그레이드**~~ → 1.18.0 (>= 1.16) 완료
3. **설정 변경** (Stage 2 config에 적용 필요):
   - `qk_clip: true` (`examples/alpha/configs/training/stage2.yaml`)
   - `NVTE_FUSED_ATTN=1`, `NVTE_FLASH_ATTN=0` (`examples/alpha/configs/env.yaml`)
4. ~~**검증**~~ → `F16_arbitrary_seqlen` + `return_max_logit=True` 확인 완료

### 예상 영향:
- 6개 attention 레이어가 FA3 -> FusedAttention(cuDNN)으로 전환
- 42개 비-attention 레이어는 변경 없음
- attention 연산 성능은 FA3 대비 약간 다를 수 있으나, 전체 throughput에서 attention 비중이 12.5%이므로 영향 제한적
- Muon CLIP으로 학습 안정성 향상 기대 (QK attention logit 폭발 방지)

### 관련 파일:
- `examples/alpha/configs/training/pretrain.yaml` (line 40-46: qk_clip 설정)
- `examples/alpha/configs/env.yaml` (line 25-26: attention backend 설정)
- `backends/megatron/Megatron-LM-251125/megatron/core/extensions/transformer_engine.py` (line 1023-1133: TE attention 연동)
- `backends/megatron/Megatron-LM-251125/megatron/core/optimizer/qk_clip.py` (clip_qk 함수)
- `backends/megatron/Megatron-LM-251125/megatron/core/transformer/attention.py` (line 1255-1302: clip_qk 구현)
