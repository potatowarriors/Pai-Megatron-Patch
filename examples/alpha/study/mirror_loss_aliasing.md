# DiLoCo 노드 간 거울상 loss 시소 — 원인 규명과 수정 (2026-08-17)

2-노드 DiLoCo stage2 학습에서 P2b 스위치(iter 18,000) 이후 두 노드의 lm loss와
seq bal loss가 **거울상(반상관)으로 진동**하는 현상의 전면 재조사 기록.
08-10의 1차 보고(STAGE2_CURRICULUM_LOG.md §2.1)와 독립적으로 재검증했고,
진동의 방아쇠(가중치 합 잔차)를 새로 규명했으며, 수정(블록-순환 샤딩)을 구현했다.

- **재현 스크립트**: [`mirror_loss_repro.py`](mirror_loss_repro.py) — tensorboard
  원본 + Megatron 컴파일된 C++ 블렌딩 코드에서 아래 수치 전부를 end-to-end 재현.
- **차트 포함 보고서**: https://claude.ai/code/artifact/6a25a435-e148-4781-af5c-6ccb2d56d8e6
- **수정 구현**: `diloco_patch.py::_install_data_shard` `DILOCO_SHARD_BLOCK` (아래 §5)

## 1. 현상 정량화 (iter 20,001–24,026, P2b, 08-11 재시작 이후)

| 항목 | lm loss | seq bal loss |
|---|---|---|
| 노드 간 상관 (밴드패스, lag 0) | **−0.95** | **−0.996** |
| 반대칭/공통 분산비 | 26–38× | 최대 516× |
| 진폭 (노드당, smoothed) | ±0.032 | ±0.009 |
| 지배 주기 | ~336 iter | ~330 iter |

- 요동은 사실상 순수 제로섬: 페어 평균은 평탄, gap만 진동.
- grad-norm은 반상관 없음(−0.18~+0.46) — 옵티마이저 동역학이 아니라 측정 데이터의 문제.
- 로깅되는 loss는 **노드-로컬**: 각 노드가 자기 샤드 배치로 계산, 교차 노드 리덕션 없음
  (`megatron_patch/template/helper.py` → intra-node DP group all-reduce만).

## 2. 메커니즘

1. **BlendedDataset 소스 수열은 결정론적**이다 (greedy max-error, `helpers.cpp:77`).
   셔플은 각 컴포넌트 내부에만 있다. `_install_data_shard` docstring의
   "shared global **shuffled** order"는 오기였다 — 최상위 수열은 셔플되지 않는다.
2. `DILOCO_DATA_SHARD=1` 레거시 매핑은 이 수열을 **짝/홀(stride 2)**로 가른다.
   분할은 제로섬이므로 구성 요동은 정의상 두 노드에서 반부호다.
3. 가중치는 소수 6자리 유리수다. **합이 정확히 1이면**(P2) 블렌드 패턴이 주기
   10⁶(짝수) 샘플로 정확히 반복 → 패리티 고정 → **상수 구성 오프셋**
   (node0 기준 cc_actual −3.3%p, math +2.6%p → lm loss 오프셋 −0.038; 08-03 보고의
   +0.037과 일치) + 고속 저진폭 잔물결(주기 10⁶/6144 ≈ 163 iter; P2의 약한 반상관
   −0.4~−0.7의 정체).
4. **합이 1±10⁻⁶이면**(P2b +1e-6, P3 −1e-6 — `compute_stage2_switch_blends.py`의
   `%.6f` 반올림 잔차) `normalize()`가 전 가중치를 상대 10⁻⁶ 밀어낸다. 블렌드 패턴이
   짝/홀 격자 위를 세차운동하고, 1포지션 밀릴 때마다 패리티가 반전된다.
   **주기 = 2/|Σw−1| 전역 샘플 = 325.5 iter @ GBS 3072×2노드.** 실측 336.
   구성 스윙은 iter당 최대 ±245샘플(qa_pairs, 8%p).

## 3. 증거

- **시뮬레이션 재현**: 실제 `helpers.build_blending_indices` + P2b 가중치 + consumed
  카운터로 iter별 노드 간 구성 델타 26개를 계산. 이것만으로 실측 lm gap을
  OLS **R²=0.834 (raw) / 0.981 (smoothed)**, seq bal gap을 **R²=0.927** 설명.
- **정렬 검증**: 윈도우를 ±1 iter 시프트하면 R² 0.83→0.40 붕괴 — 매핑
  (iter t ↔ 전역 위치 [2(c−GBS), 2c))이 샘플 단위로 정확.
- **반사실 인과 확정** (cc_code 가중치만 조작):

  | 잔차 | 시소 주기 |
  |---|---|
  | +1e-6 (원본) | 313 iter |
  | −1e-6 | 313 iter |
  | +3e-6 | **108 iter (= 1/3)** |
  | 0 (합=1) | 저속 시소 소멸, P2형 오프셋+잔물결 복귀 |

- **out-of-sample**: 08/07 구간(iter 18,001–20,489, 독립 로그) R²=0.842.
- **페어 합산 무결성**: 매 iter 6144샘플 합산 구성은 목표 블렌드 대비 최대
  1.4샘플 편차 — 클러스터가 학습하는 데이터는 의도 그대로.
- **배제**: LR/GBS 스케줄(양 노드 bit-identical), outer Nesterov(양 노드 동일 θ̄ +
  체크섬), expert_bias(08-11 fix 후 pair-sync 검증, 시소는 지속), seed(pair assert),
  H/τ(주기 무관), 로깅 아티팩트(lag-0 반부호).

## 4. 유해성 판정

- **측정 관점 무해**: 페어 합산 구성 정확, validation 양 노드 일치. 개별 노드
  곡선이 자기 샤드 구성을 비추는 것. 모니터링은 **페어 평균** 기준이 옳다.
- **완전 무해는 아님**: 같은 부호의 구성 편향이 반주기 ~160 iter(=5×H) 지속되는데
  inner Muon 모멘텀·QK-clip 통계 등 **노드-로컬 상태는 outer 평균에서 합쳐지지
  않는다**. 전례: bias-sync fix 이전 이 압력이 expert_bias 0.118 발산 →
  08-11 프로덕션 중단. P3는 진폭이 더 크다(아래 §6).

## 5. 수정 — `DILOCO_SHARD_BLOCK` 블록-순환 샤딩 (구현 완료, 활성화는 다음 재시작)

- 매핑: node r, 로컬 i → 전역 `world·B·(i÷B) + r·B + i mod B`, **B = GBS = 3072**.
  각 노드의 매 iter 배치가 블렌드 수열의 연속 3072샘플이 되고, greedy 수열의
  유계 오차 덕에 구성이 정확해진다.
- **P3 인덱스 실측** (3,000 iter 시뮬레이션):

  | 매핑 | 노드별 구성 오차 max | 노드 간 격차 max |
  |---|---|---|
  | 짝/홀 (현행) | 137샘플 | 274샘플 |
  | 블록-순환 B=3072 | **1.3샘플** | **2샘플** |

- **부작용 없음**: 변경은 dataloader 워커의 인덱스 산술뿐(샘플당 183ns vs 84ns,
  step 예산 ~60s). GPU 경로·통신·메모리·sync 케이던스·체크포인트 무변경.
  페어 합산 스트림은 iter 단위로 기존과 동일 — 병합 모델이 보는 데이터 불변.
- **전환 무결성**: consumed % B == 0(GBS 3072 일정 구간의 모든 iteration 경계)에서
  전환하면 중복 0·누락 0 (시뮬레이션 set-equality 검증 + `_setup` assert).
- **가드**: pair 간 DILOCO_SHARD_BLOCK 일치 assert(ENVV 화이트리스트로 양 노드 전달 —
  bias-sync 때의 비대칭 env 함정 방지), consumed 정렬 assert.
- **활성화 절차** (다음 재시작 시):
  ```bash
  DILOCO_CKPT_DIR=$PWD/outputs/diloco_stage2 \
  DILOCO_DATA_SHARD=1 DILOCO_SHARD_BLOCK=3072 DILOCO_H=30 DILOCO_TAU=1 \
    bash launch_diloco.sh stage2 baseline_48L stage2 stage2_v5_blend_packed_p3
  ```
  전환 직후 노드별 wandb 곡선에서 시소가 사라지고 두 곡선이 페어 평균으로
  수렴한다 — 정상이며 수정이 동작한다는 신호다.
- 가중치 생성기의 합=1 강제는 보조 수단일 뿐이다(시소를 P2형 상수 오프셋으로
  되돌림). 참고: sync-DP의 실제 노드 분할도 rank별 연속 mbs 블록(B=24 상당)이지
  stride-2가 아니다 — 짝/홀은 sync-DP 의미론 모사에도 실패한 선택이었다.

## 6. P3 예보 (레거시 매핑 유지 시)

잔차 −1e-6 → 같은 주기(~325 iter), 진폭 증가: math 최대 ±272샘플(8.9%p),
qa_pairs ±271, korean_web ±201(가중치 6%인데 ±6.5%p — 한 노드에서 일시적으로
0%까지 하락 가능). 이 예보 곡선(보고서 아티팩트 Chart D)을 라이브 wandb gap과
겹쳐보면 본 규명을 실시간 재검증할 수 있다.

## 7. 모니터링 기준 (레거시 매핑으로 도는 동안)

- 경보 대상: **페어 평균**의 이상, gap 진폭의 추세적 확대(기준선 ±0.032).
- 비경보: 개별 노드 곡선의 주기 ~325 iter 시소(구성 aliasing의 결정론적 산물).
