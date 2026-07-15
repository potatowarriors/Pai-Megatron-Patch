# 2-Node 학습 프로젝트 종합 보고 (2026-07-12 ~ 07-15)

alpha_v2 stage2를 위한 H100 2노드(Backend.AI) 환경 구축부터, IB 부재 확인,
DiLoCo 도입·검증, full-state 복구, 토폴로지 전환까지 — 사흘간의 전 과정 기록.
상세 실험 수치는 [`diloco_pilot.md`](diloco_pilot.md), 운영 규칙은
[`../CLAUDE.md`](../CLAUDE.md) § Multi-Node Training 참조.

---

## 1. 한 페이지 요약

**출발점**: 2노드 세션 확보. 그러나 **InfiniBand 부재** (HCA 8개 전부 admin-Disabled,
노드 간 실효 1.07 GB/s TCP) → sync-DP 스케일링이 성립하지 않음 (GBS 3072 + bf16
grad-reduce로도 1.15×; GBS 256이면 1노드보다 느림).

**해법**: **DiLoCo** (H스텝 로컬 학습 + outer Nesterov 평균, Muon inner는 MuLoCo가
검증) — 노드 간 통신을 스텝 경로에서 제거.

**도달점**: 채택에 필요한 5개 블록 전부 실측 검증 완료.

| 블록 | 핵심 수치 |
|---|---|
| 통신 | 노출 오버헤드 **+0.35%**/step (v2: τ-오버랩 + dense dedup) |
| 품질 (from-scratch) | 동일 iter **−0.44 loss** (2× 데이터); sync-DP 대비 **1.6×** |
| 품질 (stage2 레짐) | 1노드와 **동률** (무해), step +1%로 2× 토큰 |
| 데이터 | 서로소 샤딩 (`DILOCO_DATA_SHARD=1`) — 중복 0 |
| 복구·탄력성 | full-state resume **bit-일치** 검증 + 2노드→1노드 인계 자동화 |

**부산물**: 이 클러스터의 환경 지뢰 5건 발견·수정 (cuDNN/TE, PIP_CONSTRAINT,
패키지 드리프트, NCCL resume OOM, 저장 크래시) — DiLoCo 무관하게 stage2 운영에 필수.

---

## 2. 타임라인

| 날짜 | 진행 |
|---|---|
| 07-12 | 노드 간 대역폭 실측 (TCP 9.1 Gbit/s, NCCL busbw 2.0 GB/s); IB Disabled 확인; sync-DP 예측 교정 (기존 문서의 "bf16 → 1.9×" 반증: overlap 창은 마지막 microbatch뿐) |
| 07-13 | 2노드 환경 셋업 (멀티노드 스크립트: PIP_CONSTRAINT·TE 서브모듈·canonical pin); cuDNN 9.24 픽스; sync-DP 실측 (fp32 0.89× / bf16 1.15× / GBS6144 1.45×); DiLoCo 구현·스모크 |
| 07-14 | 파일럿 A/B 500 iter PASSED (밤샘); v2 (오버랩+dedup); H=5 교차 발견; MuLoCo 리포 분석; full-state resume 요건 확정·구현 착수 |
| 07-15 | resume 환경 버그 격리·수정 (NCCL ch16); 3중 bit-일치 검증; stage2 레짐 A/B (동률); 서로소 샤딩; unshard 인계; 문서화 |

---

## 3. 실측 데이터 총괄

### 3.1 인터커넥트 (study/netbench)
- TCP: 단일/8-스트림 모두 **~9.1 Gbit/s** (물리 10GbE급 VXLAN 오버레이, MTU 1450)
- NCCL 16-GPU AllReduce: busbw **2.0 GB/s**, algbw 1.07 GB/s (1GB당 ~1.0초)
- IB: ConnectX-7 × 4/노드가 컨테이너에 보이지만 전부 `phys_state = Disabled` → 사용 불가 확정

### 3.2 sync-DP (기각된 경로)
| 구성 | step | vs 1노드 |
|---|---|---|
| 1노드 GBS 1536 (기준) | 61.1 s | 1.00× |
| 2노드 GBS 3072 fp32 | ~140 s | **0.89× (더 느림)** |
| 2노드 GBS 3072 bf16-reduce | ~108 s | 1.15× |
| 2노드 GBS 6144 bf16 | ~171 s | 1.45× |
- 원인: grad all-reduce는 마지막 microbatch에서만 시작 가능(no_sync 게이트) →
  노출 통신이 step당 상수 (fp32 78s / bf16 46s). conn=32·NCCL 소켓 튜닝 무효.

### 3.3 DiLoCo 실험 시리즈
| 실험 | 결과 |
|---|---|
| 파일럿 (H=30, τ=0, 500 iter, 실데이터, 역사적 stage1과 A/B) | iter≤60 bit-비교 일치(내부 대조군) → −0.44 loss @ 동일 iter; sync 16/16 θ 일치; 67.6s/step |
| v2 (τ-지연 적용 + dense dedup) | wire 완전 은닉, 노출 snap+apply ~6s/sync → **+0.35%**; wire 볼륨 53→32GB |
| H=5/τ=1 (500 iter) | **교차**: @120 −1.45 우세 → @500 +0.18 열세. H는 outer lr/μ와 결합 — H 변경 시 재튜닝 필수. H=30 유지 결정 |
| stage2 레짐 (ckpt 71526, LR 2.5e-5, GBS 3072, 200 iter) | 1노드와 동률 (101–200 평균 +0.002); step 133.0 vs 131.6s |

### 3.4 복구·탄력성
- **full-state resume**: 독립 3회(1노드 blocking / 1노드 / 2노드)의 재개 첫 loss
  **bit-identical (11.85106)** — 모델+optim+스케줄러+RNG+데이터 위치 완전 복원 증명
- **outer state**: rank당 fp32 ~27GB, 저장 ~140s (`<save>/diloco_outer/`)
- **2노드→1노드 인계** (`DILOCO_UNSHARD_RESUME=1`): 세 카운터 ×world 보정 —
  재개 첫 LR이 전역 위치 산식과 유효숫자 6자리 일치 (2.621919e-6)

---

## 4. 발견·수정한 환경 버그 (Known Issues에 상세 기록)

| # | 버그 | 수정 |
|---|---|---|
| 1 | NGC 25.03 cuDNN 9.8에 TE 2.9 QK-Clip(max_logit) 엔진 없음 → 첫 step 크래시 | pip cuDNN 9.24 (`--no-deps`) + train.sh LD_PRELOAD |
| 2 | 이미지 전역 PIP_CONSTRAINT가 명시 핀과 충돌 (importlib-metadata/packaging/TE) | 멀티노드 셋업 스크립트: 선별적 `env -u PIP_CONSTRAINT` |
| 3 | TE 서브모듈 checkout 순서 + mamba/fla 미고정 드리프트 (triton 3.7 유입 위험) | checkout 후 submodule init; canonical pin (mamba 2.2.6.post3 git, fla 0.4.1) |
| 4 | **optimizer-state resume이 NCCL comm-init OOM으로 크래시** (DiLoCo 무관, 순수 Megatron 재현) | `NCCL_MAX_NCHANNELS=16` train.sh 기본값 (성능 무손실) |
| 5 | 저장 시점 pending sync 워커 → 저장 NCCL gather 크래시 | save 훅의 자동 드레인 + τ-apply의 영구 scratch 버퍼 |

운영 관찰: NFS 공유 스토리지의 시간대별 변동 (동일 워크로드 야간 55s ↔ 주간 78s/step;
wire 40~408s) — 2달 런 일정 산정에 반영할 것.

---

## 5. 산출물 맵

**코드** (examples/alpha/): `diloco_patch.py` (코어: sync/오버랩/dedup/샤딩/resume/인계),
`pretrain_alpha_diloco.py` (엔트리), `launch_diloco.sh` (2노드 런처),
`study/netbench/` (측정 도구), `configs/training/stage2_ab.yaml` ·
`arxive/stage1_optsave.yaml` (프리셋)

**환경**: `../../setup_pai_megatron_env_multinode.sh` (repo 부모; 원본은 무수정 보존)

**문서**: `../CLAUDE.md` §Multi-Node + Known Issues 4건 / `diloco_pilot.md` (실험 상세)
/ `gradient_reduce.md` (이론+정정) / 본 보고서

**커밋** (experiment/fp8-compute, 13개): `81138f0` env+벤치 → `9a49a79` 파일럿 →
`582b8b5` v2 → `b4a1da8` resume 픽스 → `dfc38c4` stage2 A/B → `d69b271` 샤딩 →
`eba66e9`·`387bb5e` docs → `a827f39` unshard → `aacf715` docs

---

## 6. 의사결정 상태 & 열린 항목

**즉시 실행 가능**: stage2를 2노드 DiLoCo로 —
```bash
DILOCO_DATA_SHARD=1 DILOCO_H=30 DILOCO_TAU=2 \
  bash launch_diloco.sh stage2_prod baseline_48L <preset> stage2_v5_blend_packed
```
안전장치: 매 sync θ-checksum, full-state 체크포인트(+outer state), 언제든 1노드 인계.

**본 런 전 확정 필요**:
1. **예산 의미**: 0.7T가 전역이면 노드당 `train-samples`를 ÷2 (인계 산수도 이 관례에 정합)
2. (선택) 유효배치 2×에 맞춘 **LR √2 상향** 짧은 arm — 2× 토큰의 환전율을 높이는 유일한 남은 레버

**연구 백로그**: H 스윕(30 vs 100), router 밀동기화(19MB를 H=1로), H-warmup 스케줄,
outer delta 저비트 압축(MuLoCo 2-bit+EF)
