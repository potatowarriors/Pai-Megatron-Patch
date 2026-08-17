# GDN Context Parallel 포팅 (upstream PR #2642) — 구현·검증 기록

**목적**: Long-context 학습(32K→128K)에는 CP가 사실상 필수(출력 logits `seq×163,968` +
활성값이 80GB를 초과). 48층 중 18층인 GatedDeltaNet(M)이 CP 미지원이 유일한 하드 블로커였다.
NVIDIA main에 머지된 GDN CP(PR #2642, 2026-04-13)를 Pai 커스텀 `GatedDeltaNetMixer`에
포팅했다 — **체크포인트 레이아웃/파라미터 이름 무변경** (stage2 ckpt 그대로 로드 가능).

**브랜치**: `feature/gdn-context-parallel` (main 9a89791에서 분기; 서브모듈 무수정)

## 설계 (a2a head-split)

GDN은 시퀀스 방향 recurrence라 시퀀스 분할이 불가능하다. 대신 레이어 입구에서
all-to-all로 "시퀀스 분할 → head 분할"로 전환해 각 CP rank가 **전체 시퀀스를
head의 1/cp에 대해 연속 스캔**하고, 출구에서 되돌린다 (Mamba2 `MambaContextParallel`과
동일 설계, upstream GDN CP와 동일 구현 패턴):

1. in_proj 출력 `[z|V|Q|K|b|a]`의 채널을 사전 permute → **단일 무섹션 a2a**가
   섹션별 a2a와 등가가 되게 함 (`build_head_perm_for_split_sections`)
2. a2a cp2hp + attention load-balancing 순서 복원 (2·cp 청크 재배열 undo)
3. conv/delta-rule 커널은 전체 시퀀스 × head/cp 로 무수정 실행
   (conv weight, A_log, dt_bias는 forward에서 rank별 슬라이스 — grad는 전체
   파라미터로 역전파되고 dp-cp all-reduce가 조각을 합산)
4. 출력 y를 load-balancing redo + a2a hp2cp로 원상복구 → out_proj

## 변경 파일

| 파일 | 내용 |
|---|---|
| `megatron_patch/model/qwen3_next/gdn_context_parallel.py` | **신규** — upstream `gated_delta_net/common.py`에서 CP 유틸 포팅 (a2a 래퍼, head perm, 파라미터 슬라이서). 하부 collective는 251125 서브모듈의 `mamba_context_parallel.py` 재사용 |
| `megatron_patch/model/qwen3_next/gated_deltanet.py` | forward에 CP 경로 연결(pre/post a2a), `A_log`/`dt_bias` CP 슬라이스 참조로 교체, conv bias None 가드, cp>ngroups(그룹 복제) assert |
| `tests/test_gdn_context_parallel.py` | **신규** — 아래 검증 스위트 |

## 검증 결과 (분석 노드, 2026-07-30)

단일 GPU + 2/4-rank NCCL (bf16, toy GDN: v-heads 8×16, k-heads 4×16, GQA 2:1 유지):

| 테스트 | 결과 |
|---|---|
| head perm / load-balancing roundtrip / 파라미터 슬라이싱 단위 테스트 | ✅ |
| **CP=1 회귀**: 수정본 vs 원본 구현 forward | ✅ **bit-identical** (`torch.equal`) — 진행 중인 학습에 무영향 |
| **CP=2 vs CP=1** forward (실 NCCL a2a) | ✅ **max diff 0.000e+00** (bit-identical) |
| CP=2 loss(rank 합산) vs 전체 시퀀스 loss | ✅ |
| CP=2 grad 6종(in_proj/out_proj/conv1d/A_log/dt_bias/norm) — CP all-reduce 후 vs CP=1 | ✅ |
| CP=4 (rank당 k-head 그룹 1개 경계) | ✅ CP=2와 동일 — forward bit-identical, loss/grad 전 rank 일치 |

forward가 bit-identical한 이유: per-position/per-head 연산이 완전 독립이라 a2a 분해가
수치적으로 정확 (재배열만 있고 리덕션 순서 변화 없음).

재실행:
```bash
# 단위 + CP=1 bitwise 회귀
GDN_ORIG_FILE=<원본 gated_deltanet.py> python tests/test_gdn_context_parallel.py
# CP=2 / CP=4 등가성
torchrun --nproc_per_node=2 tests/test_gdn_context_parallel.py
torchrun --nproc_per_node=4 tests/test_gdn_context_parallel.py
```

## 제약 (구현이 강제/전제하는 것)

- **cp_size ≤ num_k_heads/tp = 16** (alpha): 그룹 복제 경로(cp>ngroups) 미지원, `__init__` assert.
  실무는 노드 내 CP ≤ 8.
- `seq_length % (2·cp) == 0` (Megatron 공통 CP 제약).
- sequence packing(THD) + CP는 여전히 차단 (`helper.py`) — upstream THD CP 경로는 미포팅.
- `CUDA_DEVICE_MAX_CONNECTIONS=1` 필수 (CP>1에서 Megatron assert). **throughput lever 1
  (conn=8, +2.7%)은 CP>1과 양립 불가 — CP 도입 시 되돌릴 것.**

## H100 클러스터 검증 러너북 (노드 빌 때)

### 사전 준비

```bash
# (분석 노드에서) 브랜치를 origin에 push — 클러스터 checkout이 받을 수 있게
git push origin feature/gdn-context-parallel

# (클러스터에서) 학습 브랜치에 머지하거나 이 브랜치를 직접 checkout
git fetch origin && git checkout feature/gdn-context-parallel
```

주의사항 (전 항목 공통):
- `CUDA_DEVICE_MAX_CONNECTIONS=1` 이어야 함 (train.sh 기본값; **conn=8 throughput
  override를 쓰고 있었다면 해제** — CP>1에서 Megatron이 assert).
- data preset이 `mock`이면 train.sh가 wandb를 자동 차단 → 라이브 대시보드 오염 없음.
- `--save`는 train.sh가 타임스탬프 run 디렉토리로 자동 유도 — 라이브 체크포인트
  디렉토리를 절대 직접 지정하지 말 것.
- profile preset은 dist_muon + qk-clip:true + prod recompute(selective layernorm+moe)
  내장, GBS 96/MBS 3 → CP∈{1,2,4,8} 전부에서 GBS 나눗셈 성립 (dp=8/cp).

### 1. 포팅 유닛테스트 (H100/sm90 + NCCL 실환경 재확인, ~5분)

```bash
cd <repo-root>
export PYTHONPATH=$PWD:$PWD/backends/megatron/Megatron-LM-251125
python tests/test_gdn_context_parallel.py                      # 단위 + (옵션) bitwise 회귀
torchrun --nproc_per_node=2 tests/test_gdn_context_parallel.py
torchrun --nproc_per_node=4 tests/test_gdn_context_parallel.py
```
판정: `ALL ... TESTS PASSED`, forward max diff 0.0.
**주의**: toy 모델이 k-head 4개라 `--nproc_per_node=8`은 cp>ngroups assert로 의도적
실패 — 8-rank(CP=8)는 아래 full-stack 스모크(실모델 k-head 16)가 커버.

### 2. Full-stack CP=1 vs CP=2 A/B (Muon + QK-Clip + EP8×CP 결합, 항목 2·3·4 통합)

```bash
cd examples/alpha
# 기준선 CP=1 (8 GPU, EP8)
bash train.sh analysis_24L profile mock \
  --expert-model-parallel-size 8 --train-iters 16
# CP=2 (dp 4), CP=4 (dp 2), CP=8 (dp 1) — 각각 실행
bash train.sh analysis_24L profile mock \
  --expert-model-parallel-size 8 --context-parallel-size 2 --train-iters 16
bash train.sh analysis_24L profile mock \
  --expert-model-parallel-size 8 --context-parallel-size 4 --train-iters 16
bash train.sh analysis_24L profile mock \
  --expert-model-parallel-size 8 --context-parallel-size 8 --train-iters 16
```
판정:
- parallel_state init 성공 (EP8이 expert-DP로 CP 흡수 — 항목 4)
- iter-1 lm_loss: CP=1 대비 |Δ| ≲ 1e-2 (GDN은 bit-identical이지만 attention ring의
  리덕션 순서 차이로 bf16 수준 오차는 정상), iter 16까지 궤적 근접 + NaN 0
- Muon step 크래시 없음, grad-norm 로그 CP=1과 근사 (항목 2)
- `max_attention_logit` 로깅 정상 (TE가 max_logit로 flash를 끄고 fused+CP 경로 선택;
  cuDNN 9.24 LD_PRELOAD는 train.sh 자동) (항목 3)

QK-Clip **발동 경로**까지 강제 확인 (임계 1로 낮춰 clip 실행 유도):
```bash
bash train.sh analysis_24L profile mock \
  --expert-model-parallel-size 8 --context-parallel-size 2 \
  --qk-clip-threshold 1 --train-iters 5
```
판정: clip 실행 로그 + exit 0 (수치는 무의미, 크래시 여부만).

### 3. stage2 체크포인트 CP>1 로드 스모크 (항목 5)

```bash
# 학습 브랜치 머지 후, 라이브와 같은 model preset으로. 로드는 read-only.
bash train.sh baseline_48L <stage2_training_preset> <stage2_data_preset> \
  --load <stage2 ckpt 경로> --no-load-optim --no-load-rng \
  --context-parallel-size 2 --exit-interval 3
```
판정: torch_dist 로드 성공(CP는 파라미터 비샤딩 — 복제 로드), 첫 lm_loss가 라이브
stage2 수준(발산/스파이크 없음), 3 iter 후 정상 종료.

### 4. LC 메모리 프로파일 (항목 6) — 32K/128K × CP{4,8}

```bash
# 32K @ CP4 (권장 기준선)
bash train.sh baseline_48L profile mock \
  --expert-model-parallel-size 8 --context-parallel-size 4 \
  --seq-length 32768 --micro-batch-size 1 --global-batch-size 8 \
  --no-create-attention-mask-in-dataloader --train-iters 5
# 32K @ CP8, 128K @ CP8 (seq만 교체: 131072)
```
- 판정: iter-1 후 Megatron 메모리 리포트의 `max allocated` < ~76GB(마진 포함) — 수치를
  이 문서 §VRAM 표에 실측으로 기입.
- 128K가 selective recompute로 초과하면 full recompute로 재시도:
  `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`
  (profile preset의 `recompute-modules`와 충돌 assert가 나면 전용 preset yaml을 만들어
  recompute 키를 교체할 것).
- `--no-create-attention-mask-in-dataloader`는 LC에서 필수 (32K에서 [1,1,s,s] bool
  마스크 = 샘플당 1GiB; TE causal은 어차피 이 마스크를 무시).

### 요약 판정표

| # | 항목 | 커버하는 러너북 단계 | 핵심 판정 |
|---|---|---|---|
| 1 | GDN CP 수치 정합 (H100) | §1 | forward max diff 0.0 |
| 2 | Muon(dist_muon)+CP | §2 | loss/grad-norm 궤적 CP=1 근접, step 크래시 없음 |
| 3 | QK-Clip+CP | §2 | max_logit 로깅 + threshold=1 강제 발동 무크래시 |
| 4 | EP8×CP 프로세스 그룹 | §2 | CP{2,4,8} init 성공 |
| 5 | stage2 ckpt CP>1 로드 | §3 | torch_dist 복제 로드 + loss 연속 |
| 6 | 32K/128K 메모리 | §4 | max-alloc < 76GB, 실측치 기록 |

## Out of scope (후속 별도 안건)

- **THD/packed + CP** (문서경계 GDN state 리셋 + attention 문서 격리): upstream
  `_build_thd_cp_a2a_perm` 포팅 + `helper.py` 차단 해제 + QK-Clip(thd에서 max_logit
  미지원)과의 충돌 해소가 한 묶음. LC 레시피에서 문서 격리 채택 여부 결정 후 진행.
- inference/decode 경로 CP (학습 전용; `mamba_mixer` decode는 CP=1 assert 유지).
