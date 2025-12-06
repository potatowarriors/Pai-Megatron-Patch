# Muon Optimizer 사용 가이드

## 개요

Muon (Momentum Orthogonalized by Newton-schulz) optimizer는 NVIDIA Megatron-LM에서 제공하는 고급 optimizer로, Adam 대비 약 **2배의 computational efficiency**를 제공합니다.

Alpha 프로젝트에서는 Megatron-LM dev branch (Megatron-LM-251125)를 새 백엔드로 추가하여 Muon optimizer를 사용할 수 있습니다.

## Muon의 특징

### 장점
- **2배 빠른 수렴**: Compute-optimal training에서 AdamW 대비 ~2× computational efficiency
- **Tensor Parallel 지원**: TP 환경에서 Newton-Schulz orthogonalization 지원
- **MoE 최적화**: Expert Parallel (EP)과 호환
- **검증된 성능**: Moonlight 3B/16B MoE 모델로 5.7T tokens 학습 성공

### 제약사항
- ⚠️ **Megatron Distributed Optimizer와 비호환**: `--use-distributed-optimizer` 사용 불가
- ⚠️ **torch checkpoint 형식만 지원**: `torch_dist` 형식 미지원
- ⚠️ **2D 파라미터 전용**: Linear/Conv 가중치에만 적용 (embedding은 Adam 권장)
- ⚠️ **FSDP 미지원**: Torch-FSDP2, Megatron-FSDP 사용 불가

### 백엔드 버전
Alpha 프로젝트는 **Megatron-LM-251125** (dev branch)를 사용합니다. 이 버전은:
- ✅ Muon optimizer가 공식적으로 통합됨
- ✅ emerging-optimizers 0.2.0과 호환 (NEW API 사용)
- ✅ LayerWiseDistributedOptimizer 포함
- ✅ 수동 패치 불필요 (코드베이스 깨끗함)

## Muon의 두 가지 모드

### 1. `muon` (비분산)
```bash
--optimizer muon
```
- 각 GPU가 **전체 optimizer state 유지**
- 메모리 사용량: 높음
- 단일 노드 또는 작은 모델에 적합

### 2. `dist_muon` (LayerWise 분산) ⭐ 권장
```bash
--optimizer dist_muon
```
- **LayerWiseDistributedOptimizer** 자동 활성화
- 각 DP rank가 **일부 레이어의 optimizer state만 유지**
- 레이어 단위로 파라미터 분산
- 업데이트 후 broadcast로 동기화
- 메모리 사용량: 중간 (muon보다 낮음, Adam+ZeRO-1보다 높음)
- **멀티 노드 Data Parallel에 최적**

### `dist_muon` vs `--use-distributed-optimizer`

**중요**: 이 둘은 **서로 다른 구현**이며 **충돌합니다!**

| Feature | `--use-distributed-optimizer` (Adam) | `dist_muon` (LayerWise) |
|---------|--------------------------------------|-------------------------|
| **파라미터 분산 방식** | Flatten buffer, 균등 분할 (ZeRO-1) | **레이어 단위 분할** |
| **그래디언트 처리** | AllReduce 후 sharded update | AllReduce 후 **레이어별** update |
| **동기화 방식** | AllGather (contiguous buffer) | **Broadcast loop** (레이어별) |
| **메모리 효율성** | 높음 (ZeRO-1 최적화) | 중간 (레이어 단위 오버헤드) |
| **호환성** | Adam, SGD | Muon only |

```bash
# ✅ 올바른 사용법
--optimizer dist_muon
# (--use-distributed-optimizer는 사용하지 않음)

# ❌ 잘못된 사용법
--optimizer dist_muon --use-distributed-optimizer
# (assertion error 발생)
```

## Alpha 프로젝트에서 Muon 사용하기

### 0. 백엔드 버전 확인

Alpha 프로젝트는 이미 Muon을 지원하는 Megatron-LM-251125 백엔드를 사용하도록 설정되어 있습니다.

#### `configs/env.yaml`
```yaml
environment:
  megatron_version: "Megatron-LM-251125"  # Dev branch with Muon optimizer support
```

**변경 불필요**: 이미 올바른 백엔드로 설정되어 있습니다 ✅

### 1. 설정 파일 수정

#### `configs/training/pretrain.yaml`
```yaml
training:
  # Optimizer
  optimizer: "dist_muon"  # Adam에서 dist_muon으로 변경
  weight_decay: 0.01

  # Muon hyperparameters
  muon_momentum: 0.95
  muon_use_nesterov: true
  muon_num_ns_steps: 5
  muon_scale_mode: "spectral"  # spectral, unit_rms_norm, shape_scaling
  muon_fp32_matmul_prec: "medium"  # low, medium, high
  muon_tp_mode: "blockwise"  # blockwise, duplicated, distributed
  muon_extra_scale_factor: 1.0
```

#### `configs/training/h100x8.yaml`
```yaml
infrastructure:
  optimizations:
    use_distributed_optimizer: false  # ⚠️ Muon과 비호환이므로 false
```

### 2. 학습 실행

```bash
cd examples/alpha
bash train.sh baseline_24L
```

자동으로 Muon optimizer가 활성화됩니다. `env.yaml`에서 설정한 `Megatron-LM-251125` 백엔드가 사용됩니다.

### 3. 학습 로그 확인

```
🚀 Muon optimizer 인자 추가 중...
  ✅ Muon 인자 추가 완료

[INFO] Initializing Muon optimizer with LayerWise distribution...
```

## 하이퍼파라미터 가이드

### `muon_momentum` (default: 0.95)
- 모멘텀 계수
- Adam의 beta1과 유사한 역할
- 권장 범위: 0.9 ~ 0.95

### `muon_use_nesterov` (default: true)
- Nesterov momentum 사용 여부
- 일반적으로 true 권장

### `muon_num_ns_steps` (default: 5)
- Newton-Schulz iteration 단계 수
- 더 많은 단계 = 더 정확한 orthogonalization, 더 느린 속도
- 권장 범위: 3 ~ 7

### `muon_scale_mode` (default: spectral)
- **spectral**: Spectral norm 기반 스케일링 (권장)
- **unit_rms_norm**: Unit RMS normalization
- **shape_scaling**: Shape-based scaling

### `muon_fp32_matmul_prec` (default: medium)
- Newton-Schulz iteration의 FP32 matmul 정밀도
- **low**: 빠름, 낮은 정밀도
- **medium**: 균형 (권장)
- **high**: 느림, 높은 정밀도

### `muon_tp_mode` (default: blockwise)
- Tensor Parallel weights의 Newton-Schulz 계산 방식
- **blockwise**: TP 환경에서 block 단위 처리 (권장)
- **duplicated**: 중복 계산
- **distributed**: 분산 계산

### `muon_extra_scale_factor` (default: 1.0)
- 추가 스케일 팩터
- Learning rate 조정에 사용
- 일반적으로 1.0 유지

## Learning Rate 튜닝

Muon은 Adam과 다른 update 스케일을 가지므로 LR 조정이 필요할 수 있습니다:

```yaml
training:
  lr: 2.0e-4      # Adam baseline
  # vs
  lr: 3.0e-4      # Muon (실험 필요)
```

**권장 실험 순서**:
1. Adam baseline LR로 시작
2. Loss가 발산하면 LR을 0.5배 감소
3. Loss가 너무 느리게 감소하면 LR을 1.5배 증가

## 메모리 사용량 비교

### H100 80GB 기준 (Alpha baseline_24L, BS=256, seq=4096)

| Optimizer | Distributed Opt | 예상 메모리 | 비고 |
|-----------|----------------|------------|------|
| Adam | ZeRO-1 (true) | ~65GB | Baseline |
| dist_muon | LayerWise (자동) | ~70GB | 약 +5GB |
| muon | false | ~75GB | 전체 state 유지 |

**결론**: H100 80GB로 충분히 사용 가능 ✅

## 트러블슈팅

### 1. `ImportError: No module named 'emerging_optimizers'`
```
ImportError: No module named 'emerging_optimizers'
```

**원인**: emerging-optimizers 패키지 미설치

**해결**:
```bash
bash /home/work/vidsearch/repos/project_s/setup_pai_megatron_env.sh
# 또는 수동 설치:
cd /tmp
git clone https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git
cd Emerging-Optimizers
pip install .
```

### 2. `Muon optimizer does not support distributed optimizer`
```
AssertionError: Muon optimizer does not support distributed optimizer for now.
```

**원인**: `--use-distributed-optimizer` 플래그가 활성화됨

**해결**:
- `h100x8.yaml`에서 `use_distributed_optimizer: false` 설정 (이미 적용됨 ✅)
- `train.sh`에서 `--use-distributed-optimizer` 주석 처리됨

### 3. ~~`Muon optimizer only supports torch checkpoint format`~~ (해결됨)
```
AssertionError: Muon optimizer only supports torch checkpoint format for now.
```

**상태**: ✅ **Megatron-LM-251125에서 해결됨**

최신 Megatron-LM-251125에서는 Muon 옵티마이저가 `torch`와 `torch_dist` 형식을 **모두 지원**합니다:
```python
# megatron/training/arguments.py:1202
assert args.ckpt_format in ["torch", "torch_dist"], "Muon optimizer supports torch and torch_dist checkpoint format."
```

**권장 설정**:
- `--ckpt-format torch_dist` 사용 (MG2HF 변환 호환성 향상)

### 4. Loss 발산 (NaN)
**원인**: Learning rate가 너무 높음

**해결**:
- `pretrain.yaml`에서 `lr`을 0.5배 감소
- `muon_num_ns_steps`를 증가 (5 → 7)
- `clip_grad`를 감소 (1.0 → 0.5)

### 5. 학습 속도가 느림
**원인**: Newton-Schulz iteration 오버헤드

**해결**:
- `muon_num_ns_steps`를 감소 (5 → 3)
- `muon_fp32_matmul_prec`를 low로 변경
- 충분한 `global_batch_size` 사용 (256 이상 권장)

## 성능 벤치마크

### Alpha baseline_24L (예상치)

| Metric | Adam (ZeRO-1) | dist_muon |
|--------|---------------|-----------|
| Tokens to target loss | 10.7B | ~5.4B (2x 효율) |
| Throughput (tokens/sec) | 100K | ~95K (-5%) |
| Memory usage (GB) | 65 | 70 (+7.7%) |
| Wall-clock time to target | 30h | ~15h (2x 빠름) |

**실제 벤치마크는 학습 후 업데이트 예정**

## 참고 자료

- **Muon Paper**: "Muon is Scalable for LLM Training" (arXiv:2502.16982)
- **Original Muon**: https://kellerjordan.github.io/posts/muon/
- **PyTorch Docs**: https://pytorch.org/docs/stable/generated/torch.optim.Muon.html
- **Megatron-LM Muon**: https://github.com/NVIDIA/Megatron-LM (dev branch)
- **Emerging Optimizers**: https://github.com/NVIDIA-NeMo/Emerging-Optimizers

## FAQ

**Q: Muon을 Adam과 혼합해서 사용할 수 있나요?**
A: 네, 가능합니다. Muon은 2D 파라미터(Linear/Conv)에만 적용하고, embedding/classifier는 Adam을 사용하는 것이 권장됩니다. 하지만 현재 Alpha 구현은 전체 모델에 단일 optimizer를 사용합니다.

**Q: Checkpoint를 Adam에서 Muon으로 전환할 수 있나요?**
A: Checkpoint를 로드할 수 있지만 optimizer state는 호환되지 않습니다. `--no-load-optim` 플래그를 사용하여 optimizer state 없이 로드하고 새로 학습하세요.

**Q: Muon이 모든 모델에 더 좋나요?**
A: 아닙니다. Muon은 대규모 transformer 모델에 최적화되어 있습니다. 작은 모델이나 특수한 아키텍처에서는 Adam이 더 나을 수 있습니다. 실험을 통해 확인하세요.

**Q: 왜 `dist_muon`을 사용해야 하나요?**
A: `muon`은 전체 optimizer state를 각 GPU에 저장하므로 메모리 부담이 큽니다. `dist_muon`은 LayerWise 분산으로 메모리를 절약하면서도 성능을 유지합니다.

**Q: 왜 Megatron-LM-251125 백엔드를 사용하나요?**
A: Megatron-LM dev branch는 Muon optimizer를 공식적으로 지원하며, emerging-optimizers 0.2.0의 NEW API와 호환됩니다. 수동 패치 없이 깨끗한 코드베이스를 유지할 수 있습니다.

**Q: 이전 Megatron-LM-250908 백엔드로 롤백할 수 있나요?**
A: 네, `configs/env.yaml`에서 `megatron_version: "Megatron-LM-250908"`로 변경하면 됩니다. 단, Muon optimizer는 사용할 수 없고 Adam으로 돌아갑니다.

---

## 백엔드 마이그레이션 내역

### 2025-11-25: Megatron-LM-251125 추가
- **변경사항**: Megatron-LM dev branch를 새 백엔드로 추가
- **이유**: Muon optimizer 공식 지원 (NEW API 호환)
- **영향**:
  - `backends/megatron/Megatron-LM-251125/` 추가 (dev branch)
  - `configs/env.yaml`: `megatron_version` 업데이트
  - Megatron-LM-250908의 수동 Muon 패치 제거 (원상복구)
- **장점**:
  - ✅ 수동 패치 불필요 (공식 통합됨)
  - ✅ emerging-optimizers 0.2.0과 호환
  - ✅ 코드베이스 깨끗함 (non-invasive)
  - ✅ 쉬운 롤백 (env.yaml만 변경)

---

**마지막 업데이트**: 2025-11-25
**작성자**: Claude Code
**버전**: Alpha v1.0
