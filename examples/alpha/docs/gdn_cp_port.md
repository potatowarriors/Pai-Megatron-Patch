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

## H100 클러스터 후속 검증 체크리스트 (노드 빌 때)

| # | 항목 | 명령 (예) | 판정 기준 |
|---|---|---|---|
| 1 | analysis_24L twin CP=2 스모크 | `bash train.sh analysis_24L profile mock --context-parallel-size 2 --train-iters 10` | exit 0, NaN 없음, iter-1 lm_loss가 CP=1 실행과 bf16 오차 내 일치 |
| 2 | Muon(dist_muon)+CP | 위와 동일 (profile preset이 Muon 사용 시 그대로) | grad-norm/loss 궤적 CP=1과 근사 일치, optimizer step 크래시 없음 |
| 3 | QK-Clip+CP | `--qk-clip` 활성 상태에서 #1 반복 (cuDNN 9.24 LD_PRELOAD 유지) | max_attention_logit 로깅 정상, fused-attn 엔진 선택 성공(TE가 max_logit로 flash를 끄고 fused+CP 경로 사용) |
| 4 | 프로세스 그룹 결합 | 8GPU: TP1·PP1·EP8·CP{2,4,8} | parallel_state init 성공 (expert-DP로 CP 흡수 확인) |
| 5 | stage2 ckpt CP>1 로드 | `--load <stage2 ckpt> --context-parallel-size 4` 스모크 | torch_dist 로드 성공(CP는 파라미터 비샤딩·복제), resume 후 loss 연속 |
| 6 | 32K 메모리 프로파일 | prod 48L, seq 32768, MBS 1, CP{4,8} | max-alloc < 80GB, 여유분 기록 → 128K 계획 수립 |

## Out of scope (후속 별도 안건)

- **THD/packed + CP** (문서경계 GDN state 리셋 + attention 문서 격리): upstream
  `_build_thd_cp_a2a_perm` 포팅 + `helper.py` 차단 해제 + QK-Clip(thd에서 max_logit
  미지원)과의 충돌 해소가 한 묶음. LC 레시피에서 문서 격리 채택 여부 결정 후 진행.
- inference/decode 경로 CP (학습 전용; `mamba_mixer` decode는 CP=1 assert 유지).
