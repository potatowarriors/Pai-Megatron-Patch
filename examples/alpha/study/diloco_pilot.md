# DiLoCo 2-Node Pilot — 결과 기록 (2026-07-14)

IB 없는 2노드(H100×8 ×2, 노드 간 ~1 GB/s TCP)에서 DiLoCo(arXiv:2311.08105)로
sync-DP의 통신 병목을 우회하는 파일럿. 구현: `../diloco_patch.py` +
`../pretrain_alpha_diloco.py` (train.sh `PRETRAIN_SCRIPT` 오버라이드).

## 설계

- 노드별 **독립 단일노드 Megatron 인스턴스** (DP=8/EP=8, grad reduce는 NVLink 전용)
- H=30 inner step마다 outer sync: 로컬 rank r ↔ 상대 노드 rank r **페어별 Gloo 그룹**
  (EP 샤드 레이아웃이 rank별로 다르므로)으로 pseudo-gradient(θ_global − θ_local)를
  bf16 wire로 평균 → **outer Nesterov** (lr 0.7, μ 0.9) → 모델 반영 후
  `optimizer.reload_model_params()` (Muon fp32 master 갱신 — 생략하면 sync가 덮여 사라짐)
- outer state(θ_global + momentum, fp32)는 CPU 상주 (rank당 ~27 GB)
- 시작 시 node0 가중치 broadcast(bit-identical 출발) + 매 sync 후 양 노드 checksum 검증
- Muon inner 호환성 근거: MuLoCo (arXiv:2505.23725)

## A/B: 역사적 stage1 런(20260512, 단일노드 sync-DP) vs DiLoCo 500 iter

동일 config(`stage1.yaml` + `stage1_v5_blend`, GBS 1536/노드), node0 seed 1234
(역사적 런과 동일 데이터 순서), node1 seed 4321(데이터 셔플만 분리).

| iter | A(역사적) | B(DiLoCo) | B−A |
|---|---|---|---|
| 10 | 12.0740 | 12.0741 | +0.0000 |
| 60 | 11.7626 | 11.7624 | −0.0002 |
| 120 | 9.6499 | 9.1287 | **−0.52** |
| 300 | 5.8592 | 5.4515 | −0.41 |
| 500 | 4.5409 | 4.1009 | **−0.44** |

- iter ≤ 60: 곡선 **사실상 동일**(±0.0002) — node0가 역사적 런의 정확한 재현임을
  증명하는 내부 대조군 (LR이 작아 2× 데이터 효과가 노이즈 아래).
- iter 120+: 2× 데이터 효과 발현, **동일 iteration 기준 −0.44 loss 우위 지속**.
- 16회 outer sync 전부 replica checksum 일치, NaN 0, 500 iter 체크포인트 저장 성공.

## 효율 판정 (실측)

| 지표 | 값 |
|---|---|
| A 평균 step | 61.1 s |
| B 평균 step (sync 분할상환 포함) | 67.6 s (+10.6%) |
| B가 loss 4.1009 도달 | 500 iter = **9.39 h** |
| A가 같은 loss 도달 | 611 iter = **10.37 h** → B가 **~10% 빠름** |
| (비교) sync-DP 2노드 bf16, 같은 1.536M samples | 500 iter × 108 s = 15.0 h → B가 **1.6× 빠름** |

**결론**: 이 네트워크에서 2노드를 쓸 거라면 DiLoCo가 sync-DP(bf16-reduce)보다
엄격히 우월. 단일노드 대비 wall-clock 이득(~10%)은 warmup 초기 구간이라 최소치 —
대배치 이득은 LR 정점 이후 커지며, LR √k 재조정 시 추가 확보 여지.

**토큰-정합 관점의 정직한 주석**: B@500(4.10) > A@1000(3.47) — 소비 토큰당으로는
뒤짐. 이는 DiLoCo 고유 결함이 아니라 LR 스케줄을 1536 기준으로 둔 채 유효 배치를
3072로 키운 표준 대배치 비용. 채택 시 LR/스케줄을 유효 배치 기준으로 재조정할 것.

## v2: overlapped sync + dense dedup (2026-07-14, MuLoCo repo 참고)

facebookresearch/MuLoCo의 torchft(`fragment_sync_delay`) 방식을 이식한 개선판.
mock 스모크(H=6, τ=2, 15 iter)로 검증:

- **overlapped sync (`DILOCO_TAU`>0)**: 스냅샷만 메인 스레드(D2H ~3초), allreduce+
  outer 계산은 워커 스레드에서 compute와 병행, τ스텝 뒤 **additive correction**으로
  적용(지연 구간의 로컬 진행 보존). wire 40초가 완전히 은닉되어 적용 스텝의 step
  time이 일반 스텝과 동일(57.5초). **노출 비용 sync당 ~6.4초(snap+apply) →
  H=30 기준 +0.35%** (v1 blocking은 +10.6%였음).
  τ>0에서는 replica가 τ-구간 로컬 진행만큼 합법적으로 다르므로 divergence 가드는
  θ_global(결정적 CPU fp32) 기준으로 검사. τ=0이면 v1과 동일한 동기 방식(bit-일치).
- **dense dedup**: Megatron 마커 `param.allreduce==False`로 expert 판별. dense
  279개 텐서(1.53B)는 노드 내 DP-동일하므로 local_rank0 페어만 wire 전송, 나머지
  rank에는 기본 그룹 NCCL broadcast(NVLink)로 배포. wire 볼륨 53→32GB.
- 주의: v2의 τ>0 변형은 mechanics 검증만 완료(품질은 Streaming DiLoCo 문헌 근거).
  장기 채택 전 τ∈{0,2,5} 짧은 loss A/B 권장.

## H=5/τ=1 500-iter A/B (2026-07-14) — 교차(crossover) 발견

동일 프로토콜로 H=5(τ=1, overlapped) 실행. 3-way 비교 (같은 iteration 기준):

| iter | A(1노드) | B(H=30,τ=0) | C(H=5,τ=1) | C−B |
|---|---|---|---|---|
| 120 | 9.650 | 9.129 | 7.682 | **−1.45** |
| 300 | 5.859 | 5.452 | 5.240 | −0.21 |
| 400 | 5.100 | 4.646 | 4.692 | +0.05 |
| 500 | 4.541 | 4.101 | 4.286 | **+0.18** |

- **교차 패턴**: H=5가 초반(~350 iter)에 크게 앞서다 이후 H=30에 역전당함 — "실효
  outer step이 큰" 구성의 전형적 시그니처 (빠른 초기 하강, 정련 구간 손해).
- 원인 해석: outer lr 0.7/μ 0.9는 H 수십~수백용 문헌값. H=5에서는 momentum 누적
  (정상상태 이득 1/(1−μ)=10×)이 더 상관된 스텝들에 걸려 실효 구동력이 커짐.
  τ=1 staleness도 창(5스텝)의 20%로 상대적으로 큼.
- **결론**: H는 "sync-DP 근접 다이얼"이 아니라 outer optimizer의 실효 세기와 결합된
  변수. H를 바꾸면 outer lr/μ 재튜닝 필요. **채택 후보는 H=30 유지** (+τ=2~3).
- 처리량 관찰: 주간 NFS 혼잡 시 wire 68~89초 → τ=1 창(60초) 초과로 sync당 ~10초
  노출 + **데이터 로더 자체가 55→78초로 열화** (전날 야간 런도 동일 패턴 후 회복;
  DiLoCo 무관, 공유 스토리지 시간대 변동 — 2달 런 계획에 반영할 것).
- 저장 크래시 재발(500 iter 완주 후 exit-save에서): τ-apply가 apply당 파라미터별
  GPU 임시버퍼를 할당(~100회 누적)해 allocator 단편화 → **영구 per-dtype scratch
  버퍼로 apply의 GPU 할당 제거** (수정 완료). 학습 데이터는 무손실.

## Full-state resume 검증 (2026-07-15) — 2노드→1노드 인계 시나리오 PASSED

요건: **optimizer state 포함 저장 → 정확한 resume** (재워밍업 폴백 배제; 2노드→1노드
자원 축소 대비). `configs/training/stage1_optsave.yaml` (stage1 − no-save-optim).

- **환경 버그 발견 (DiLoCo 무관 — 순수 Megatron도 재현)**: optimizer state를 실은
  resume이 첫 collective에서 `Failed to CUDA calloc async N bytes` (NCCL WARN 'out of
  memory'). 원인: NCCL 2.25 기본 64채널의 comm당 GPU 버퍼가 크고, resume은 로드된
  optim state + 비동기 in-flight 할당 위에서 지연 초기화 comm들이 일제히 버퍼를 요구
  → OOM. fresh는 optim state가 첫 step 이후 생기므로 회피. `CUDA_LAUNCH_BLOCKING=1`
  통과(= 레이스/순서 문제)와 NCCL_DEBUG의 'out of memory'가 결정 근거.
- **수정: `NCCL_MAX_NCHANNELS=16`** (train.sh 기본값으로 반영) — comm 버퍼 4× 절감,
  step time 무손실(60.7s vs 61.1s). + diloco_patch의 resume 워밍업(그룹/coalesced).
- **드레인 규칙**: 저장 시점에 pending sync 워커가 살아 있으면 저장의 NCCL gather가
  재현성 있게 크래시 → save 훅이 pending을 join+apply 후 저장 (체크포인트가 일관된
  outer 상태를 담는 부수 효과).
- **outer state 영속화**: 매 저장마다 rank별 `<save>/diloco_outer/iter_N/`에
  θ_global+momentum(fp32, ~27GB/rank, 저장 ~140s). resume 시 자동 로드 + 페어 checksum
  검증, 초기 broadcast 생략. 1노드 인계 시에는 자동 무시됨.
- **판정 수치**: 독립적인 세 resume(1노드 blocking / 1노드 ch16 / 2노드 DiLoCo)의
  **iter-13 loss가 bit-identical (11.85106)** — 모델+optim+스케줄러+RNG+데이터 위치가
  완전 복원됨을 한 숫자로 증명. 저장 시점 궤적과의 연속성 ✓, 스파이크 없음.

## Stage2 레짐 A/B (2026-07-15) — 성숙 모델·저LR에서 무해성 확인

계획된 stage2 = stage1 레시피 + stage2 packed blend + GBS 3072, stage1 최종 ckpt
(71,526)에서 finetune, LR은 stage1 종료값 2.5e-5 연속 (`stage2_ab.yaml`). 200 iter,
X=1노드 vs Y=DiLoCo(H=30, τ=2, node1 seed 분리).

| iter | X(1노드) | Y(DiLoCo) | Y−X |
|---|---|---|---|
| 1–30 | 1.979→1.955 | 동일(±0.0000) | 첫 sync(32) 전 내부 대조군 |
| 100 | 1.8251 | 1.8351 | +0.010 |
| 200 | 1.7901 | 1.7860 | **−0.004** |

- **동률(101–200 평균 +0.002)**: 이 저LR 구간 200 iter에서는 2× 데이터 이득이
  가시화되지 않음(기대와 일치 — from-scratch의 −0.44는 고LR 급하강 구간의 효과).
  초반 +0.01 교란 후 회복→역전: averaging 무해.
- **비용**: step 133.0s vs 131.6s (+1%) — 같은 벽시계로 2× 토큰 처리.
- stage2 조건 특이사항: packed blend 스트리밍이 무거워 wire 273~408s로 증가
  (τ=2 창 266s 초과분 ~4.4s/step 노출). H=30 주기(4000s) 대비 여전히 10%.
- 해석: 중반 레짐에서 DiLoCo는 "무해 + 공짜 2× 데이터". 이득의 가시화는 장기
  누적 또는 유효배치 2×에 맞춘 LR 상향(√2)에서 기대 — 채택 판단은 이 프레임으로.

## 미해결/후속

- **H 스윕** (30 vs 100): sync 오버헤드 3.1%→0.9%; MoE expert drift 상한 확인
- **Router 밀동기화 arm**: router+expert_bias(19 MB)만 매 스텝 sync → H를 크게 밀 수
  있는지 (연구 아이디어, 2026-07-14 논의)
- outer sync 최적화: dense 파라미터 중복 동기화 제거(볼륨 −37%), 2–4-bit 압축(MuLoCo)
- 알려진 이슈(수정됨): outer step에서 GPU 위 dtype 변환 임시버퍼가 allocator 단편화를
  일으켜 torch_dist 저장의 NCCL gather를 깨뜨림 → D2H 후 CPU 변환 + `empty_cache()`로
  해결. 저장 문제 재발 시 `DILOCO_SKIP_SAVE=1` 사용.

## 실행

```bash
# 런처: /home/work/.claude/jobs/13634ad5/tmp/launch_diloco.sh (세션 임시) — 내용:
# 노드별 env: DILOCO_RANK={0,1} DILOCO_WORLD=2 DILOCO_MASTER=main1 DILOCO_PORT_BASE=<p>
#   DILOCO_H=30 DILOCO_OUTER_LR=0.7 DILOCO_OUTER_MOMENTUM=0.9
#   PRETRAIN_SCRIPT=$ALPHA/pretrain_alpha_diloco.py
#   NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
# node1에는 --seed <다른값> (데이터 셔플 분리; 가중치는 broadcast로 통일됨)
DILOCO_RANK=0 ... bash train.sh baseline_48L stage1 stage1_v5_blend   # main1
DILOCO_RANK=1 ... bash train.sh baseline_48L stage1 stage1_v5_blend --seed 4321  # sub1
```

## 데이터 소비 모드 (2026-07-15 추가)

| 모드 | 방식 | 용도 |
|---|---|---|
| 기본 (seed 분리) | 노드별 다른 셔플 seed — node0가 1노드 기준선과 bit-비교 가능 | **A/B 실험 전용** (풀 예산 시 ~17% 중복) |
| **`DILOCO_DATA_SHARD=1`** | 동일 seed의 단일 전역 셔플 순서를 node r이 `world*i+r` 위치로 분할 — sync-DP의 배치 분할과 동일 의미론, **중복 0/누락 0** | **프로덕션 필수** |

샤드 모드는 언더라잉 데이터셋을 world× 크기로 빌드해 노드당 --train-samples 예산을
정확히 커버하고(스모크 실측: 219.7M→109.87M = 예산 일치), 동일-seed를 페어 검증으로
강제하며(다른 seed면 즉시 assert), iter 1부터 노드별 loss가 갈리는 것으로 분할 동작을
확인했다. 트릴리언 스케일에서 seed-분리의 17% 중복 = 수십 GPU-일 낭비이므로 최종
학습은 반드시 샤드 모드로.

## 토폴로지 전환: 2노드 샤드 → 1노드 인계 (2026-07-15, DILOCO_UNSHARD_RESUME ✅)

샤드 체크포인트의 카운터는 **로컬**(N×GBS)이지만 페어는 전역 순서에서 world×를 소비했다.
naive resume 시: 래퍼 없으면 직전 절반 중복 / 래퍼 유지하면 홀수 샤드 영구 누락.
`DILOCO_UNSHARD_RESUME=1`(순수 1노드 런에 적용)이 load_checkpoint를 감싸 세 결합
카운터를 ×world 보정: ① consumed(데이터 전역 위치) ② scheduler num_steps(LR 위치)
③ iteration+fp-ops(예산·종료). 검증(2노드 12 iter → 1노드 재개 6 iter): consumed
18432→36864, **재개 첫 LR 2.621919e-6 = 2e-4×38400/2929152 정확 일치**(스케줄러 보정의
산술 증명; 미보정이면 1.36e-6), loss 연속·스파이크 없음·정상 종료. WSD stable 구간
전환이 가장 안전(LR 상수라 스케줄 위상 오차에 둔감). 역방향(1→2)은 ÷world + 샤드 on.
