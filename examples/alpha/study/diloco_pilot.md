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
