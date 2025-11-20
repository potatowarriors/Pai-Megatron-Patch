# Alpha 프로젝트 실험 로그

실험 기록 및 결과 추적 문서

---

## 실험 템플릿

새 실험 추가 시 다음 템플릿을 사용하세요:

```markdown
## 실험 #X: [실험 이름]

**날짜**: YYYY-MM-DD
**연구자**: [이름]
**목표**: [실험 목표]

### 설정
- **모델**: [config file]
- **데이터**: [dataset]
- **하드웨어**: [GPU 정보]

### 변경사항
[이전 실험 대비 변경된 내용]

### 가설
[검증하려는 가설]

### 결과
- **Throughput**:
- **Loss**:
- **PPL**:
- **메모리 사용**:

### 관찰
[주요 발견사항]

### 결론 및 다음 단계
[결론 및 후속 실험 계획]
```

---

## 실험 #1: Baseline 24L 최초 학습

**날짜**: 2025-01-17
**연구자**: Alpha Team
**목표**: H100 8-GPU 환경에서 Alpha 모델 학습 가능성 검증

### 설정

**모델 설정**:
- Layers: 24
- Hidden: 2048
- Experts: 256
- TopK: 8
- Config: `configs/model/baseline_24L.yaml`

**학습 설정**:
- Learning Rate: 3.0e-4 → 3.0e-5
- Batch: MBS=2, GBS=256
- Seq Length: 4096
- Total Tokens: 10.66B
- Config: `configs/training/pretrain.yaml`

**인프라**:
- Hardware: 8× H100 (Single Node)
- Parallelism: TP=1, PP=1, EP=8
- Flash Attention: 3
- Config: `configs/training/h100x8.yaml`

**데이터**:
- Dataset: KORMo 1% subset
- Path: `/home/work/Datasets/KORMo_processed/mmap/qwen3_1pct/`
- Config: `configs/data/kormo_1pct.yaml`

### 변경사항

원본 Qwen3-Next-80B-A3B 대비:
1. **레이어**: 96 → 24 (75% 감소)
2. **Experts**: 512 → 256 (50% 감소)
3. **Attention Heads**: 16 → 32 (2배 증가)
4. **Head Dimension**: 256 → 64 (4배 감소)
5. **MoE FFN Hidden**: 512 → 768 (50% 증가)

### 가설

1. **H1**: 24 layers로 H100 8-GPU에서 OOM 없이 학습 가능
2. **H2**: Expert size 증가가 expert 수 감소를 보상
3. **H3**: More heads (32) + smaller head_dim (64)이 효율적
4. **H4**: 12.5% attention ratio가 충분한 표현력 제공

### 결과

*(학습 완료 후 업데이트)*

**Throughput**:
- Samples/sec:
- Tokens/sec:
- TFLOPs/GPU:

**Loss Curve**:
- Initial loss:
- Loss @ 100 iters:
- Loss @ 500 iters:
- Loss @ 1000 iters:

**Perplexity**:
- Train PPL:
- Valid PPL:

**메모리 사용**:
- Peak memory/GPU:
- Activation memory:
- Parameter memory:
- Optimizer memory:

**시간**:
- Time per iteration:
- Estimated time to completion:

### 관찰

*(학습 중/후 업데이트)*

**긍정적**:
1.

**문제점**:
1.

**예상 밖**:
1.

### 결론 및 다음 단계

*(학습 완료 후 업데이트)*

**결론**:
-

**다음 실험 계획**:
1.
2.
3.

**파일**:
- 실험 디렉토리: `experiments/20250117_baseline_24L/`
- 원본 스크립트: `run_original.sh`
- 설정 스냅샷: `config_snapshot.yaml`
- 실험 노트: `notes.md`

---

## 실험 #2: [다음 실험 제목]

**날짜**: YYYY-MM-DD
**연구자**: [이름]
**목표**: [목표]

*(실험 후 추가)*

---

## 실험 비교표

| 실험 ID | 날짜 | Layers | Experts | TopK | Throughput | Loss (final) | PPL | 메모리/GPU |
|---------|------|--------|---------|------|------------|--------------|-----|-----------|
| #1 baseline_24L | 2025-01-17 | 24 | 256 | 8 | - | - | - | - |
| #2 | - | - | - | - | - | - | - | - |

---

## Best Results Tracker

**Best Throughput**: [실험 ID] - [X tokens/sec]
**Best Loss**: [실험 ID] - [X.XX]
**Best PPL**: [실험 ID] - [X.XX]
**Most Memory Efficient**: [실험 ID] - [X GB/GPU]

---

## 실험 태그

실험에 다음 태그를 사용하여 분류:

- `#baseline`: 기준 실험
- `#architecture`: 아키텍처 변경
- `#hyperparameter`: 하이퍼파라미터 튜닝
- `#data`: 데이터 변경
- `#optimization`: 최적화 기법
- `#ablation`: Ablation study

---

## Ablation Studies

### Attention Ratio

| Ratio | Layers | Pattern | Loss | PPL | 비고 |
|-------|--------|---------|------|-----|------|
| 12.5% | 3/24 | M-M-M-* | - | - | Baseline |
| 25% | 6/24 | M-* | - | - | Planned |
| 6.25% | 1.5/24 | M-M-M-M-M-M-M-* | - | - | Planned |

### Expert Count vs Size

| Experts | FFN Hidden | TopK | Loss | PPL | 메모리 |
|---------|-----------|------|------|-----|--------|
| 256 | 768 | 8 | - | - | Baseline |
| 128 | 1024 | 6 | - | - | Planned |
| 512 | 512 | 10 | - | - | Original |

### Head Configuration

| Heads | Head Dim | KV Groups | Loss | PPL | KV Cache |
|-------|----------|-----------|------|-----|----------|
| 32 | 64 | 2 | - | - | Baseline |
| 16 | 128 | 2 | - | - | Original |
| 64 | 32 | 2 | - | - | Planned |

---

## 학습 곡선 분석

*(실험 완료 후 TensorBoard 스크린샷 및 분석 추가)*

### Loss Curves

- Training Loss
- Validation Loss
- Loss Spike 분석

### Throughput

- Tokens/sec over time
- GPU utilization
- Bottleneck 분석

### Memory

- Peak memory tracking
- Activation memory evolution
- OOM incidents

---

## 재현성 체크리스트

각 실험에 대해 다음을 확인:

- [ ] Config 파일 저장 (`config_snapshot.yaml`)
- [ ] 랜덤 시드 기록
- [ ] 환경 변수 기록 (`env.yaml`)
- [ ] 실행 명령 기록
- [ ] 체크포인트 저장
- [ ] TensorBoard 로그 보관
- [ ] 실험 노트 작성 (`notes.md`)

---

## 문제 해결 로그

### OOM (Out of Memory)

| 날짜 | 실험 | 원인 | 해결책 | 결과 |
|------|------|------|--------|------|
| - | - | - | - | - |

### 학습 불안정

| 날짜 | 실험 | 증상 | 원인 분석 | 해결책 |
|------|------|------|----------|--------|
| - | - | - | - | - |

### 성능 저하

| 날짜 | 실험 | 증상 | 병목 지점 | 개선 방법 |
|------|------|------|----------|----------|
| - | - | - | - | - |

---

## 참고 자료

- [ARCHITECTURE.md](ARCHITECTURE.md): 모델 아키텍처 상세
- [SETUP.md](SETUP.md): 환경 세팅 가이드
- `experiments/`: 개별 실험 디렉토리

---

**마지막 업데이트**: 2025-01-17
