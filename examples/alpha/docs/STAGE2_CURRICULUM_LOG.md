# Stage 2 Mid-run Curriculum & DiLoCo Bias-Sync Log (2026-07-29 ~ 08-17)

stage2 프로덕션 런(DiLoCo 2노드, 총 0.6T 예산) 중반에 수행한 **blend 커리큘럼 도입**과,
그 과정에서 발견·수정한 **DiLoCo × MoE expert-bias 발산 결함**의 단일 진입점 기록이다.
결정의 근거가 된 Nemotron 3 계보 분석과 각 검증의 실측치를 함께 남긴다.

관련: [`../CLAUDE.md`](../CLAUDE.md) § Known Issues("DiLoCo expert-bias 발산"),
`../configs/training/stage2.yaml` 헤더(커리큘럼 표), `../study/diloco_pilot.md`(DiLoCo 기반 검증),
`../tools/compute_stage2_switch_blends.py`(blend 생성 도구).

## 타임라인

| 일자 | 사건 |
|---|---|
| 07-29 | Nemotron 3 계보 분석 → "무가중치 blend = 실험적 최악(BASE-ND)" 진단, 2-스위치 커리큘럼 설계 |
| 07-29~30 | Nemotron-Pretraining-Specialized v1/v1.1/v1.2 토크나이즈+팩 (실측 **323.8B**, 15 subset) |
| 08-01 | **P2 스위치** @ ckpt 14,000 (specialized 15%, math 20%, code 13%) |
| 08-03 | 노드 간 loss 오프셋(+0.037) 규명 — 짝/홀 샤딩 × blend 인덱스 aliasing (정적 형태) |
| 08-05~06 | Nemotron-CC-Code-v1 토크나이즈+팩 (실측 **412.0B** — raw code 0% 결손 해소) |
| 08-07 | resume 정밀 검증(사용자 제안) → **P2b 스위치** @ ckpt 18,000 (cc_code 8%) |
| 08-10 | 거울상 loss 시소 규명 (aliasing 동적 형태, R²=0.972) / P2b 후 loss V자 진단 |
| 08-11 | **expert-bias 발산 정밀 분석 → 프로덕션 중단 → bias-sync 구현 → 20,000 재시작** |
| 08-17 | **P3 스위치** @ ckpt 24,000 / 거울상 시소 전면 재조사(사용자 지시, §2.5) — 세차운동 메커니즘 규명·반사실 인과 확정 → **블록-순환 샤딩(`DILOCO_SHARD_BLOCK`) 구현** (활성화는 다음 재시작) |
| 08-17 | **블록-순환 활성화 재시작** @ ckpt 24,000 — B=3072는 기동 assert가 거부(GBS 램프가 남긴 consumed ≡ 2,880 mod 3072 오프셋 → 이 런에 3072 정렬점이 존재하지 않음). 실제 p3 인덱스로 후보 실측: parity 286.3 / **B=192: 35.8** / B=3072: 6.6 samples(iter당 노드 간 구성 격차). **B=192 채택** — gcd(consumed, 3072)라 이 런의 모든 미래 체크포인트에서 영구 정렬(post-ramp 증분이 192의 배수), 코드 무변경. **이후 모든 resume 명령에 `DILOCO_SHARD_BLOCK=192` 필수** (stage2.yaml 헤더에 명문화 — 누락 시 parity로 무음 회귀 → 중복/누락) |

## 1. Blend 커리큘럼

근거: Nemotron 3 Ultra(arXiv 2606.15007) 및 계보(Nemotron-H 2504.03624, Nano 2 2508.14444,
Nemotron-CC 2412.02595, Feng et al. 2412.15285). 핵심 원칙 — 자연(크기비례) 가중은 최악의
전략, 후반 phase일수록 math·SFT-style을 진하게, code는 14~20% 고정, blend 스위치는 LR
decay와 정렬. <1 epoch 예산에서는 upweight의 반복 비용이 0.

| 구간 | preset | 핵심 구성 |
|---|---|---|
| 0 – 14,000 | `stage2_v5_blend_packed` | 자연비율 (CC-HQ 59.6 / code 26.8 / math 12.2) |
| 14,000 – 18,000 | `_p2` | specialized **15**, math 20, code 13, CC 45.5 |
| 18,000 – 24,000 | `_p2b` | + cc_code **8** (재원: cc_act −4, cc_qa −2, math −1, code −1) |
| 24,000 – 26,832 (decay) | `_p3` | specialized **35**, cc_code 10, math 18 (P3 재원 배분은 24,000 전 재확인) |

- 데이터 자산: specialized(stem-sft 계열) 323.8B `stage2_packed/specialized/<ver>/<subset>/`,
  cc_code(실 CC 코드, Ultra의 nemotron-cc-code 카테고리) 412.0B `stage2_packed/cc_code/`.
  Nemotron-Pretraining-Code-v1/v3은 메타데이터-only(원문 없음)라 사용 불가 판정.
- 생성 도구: `tools/compute_stage2_switch_blends.py` (실측 bin 크기 비례 분배, 미완성 트리 거부).
- 스위치 절차: 체크포인트 경계에서 정지 → data preset만 교체 → full-state resume (검증 5종:
  카운터 연속·LR 연속·loss 계단 크기·seq_aux 평탄·blend 스모크).

## 2. 주요 발견 (시간순)

### 2.1 짝/홀 샤딩 × blend 인덱스 aliasing (08-03 / 08-10)

`DILOCO_DATA_SHARD=1`의 `world*i+r` 분할이 BlendedDataset의 결정론적 배치 순서와 간섭 —
두 노드가 받는 데이터 구성이 체계적으로 다르다.

- **정적 형태(P2)**: node1이 cc_actual +3.3%p / math −2.6%p → 상수 loss 오프셋 +0.037.
- **동적 형태(P2b)**: 평균 오프셋 ~0으로 재배열되며 창 단위 **거울상 시소**로 표출 —
  30-iter 창 delta 상관 **−0.958**, gap을 짝/홀 구성차 4변수 회귀로 **R²=0.972** 설명.
- 판정: 측정 잣대의 문제(클러스터 합산 분포는 정확, outer 평균이 흡수). **모니터링은 두 노드
  평균으로**; 격차 "확대"만 경보. 근본 해소는 해시/블록-순환 샤딩(미래 개선, 런 중 금지).
- **08-17 후속(§2.5)**: 이 판정은 재조사에서 대체로 유지됐으나 불완전했다 — 진동 주기의
  원인(가중치 합 잔차의 세차운동), seq bal loss의 동일 패턴, 노드-로컬 optimizer 상태로의
  2차 경로가 빠져 있었고 재현 산출물이 없었다. "런 중 금지"는 정정됨: 블록-순환 전환은
  consumed % B == 0 경계에서 정확(중복 0/누락 0)하다.

### 2.2 Resume 정밀 검증 + 실행 간 비결정성 (08-07)

ckpt 18,000에서 연속 궤적 vs resume×2 3중 비교 (P2b 스위치 전 사용자 제안으로 수행):

- **iter 18,001(첫 재개 스텝)은 세 궤적·양 노드 완전 일치** — 가중치·데이터 순서·RNG·LR
  클록·Muon fp32 master·DiLoCo outer state 복원이 전부 정확함을 증명.
- 이후 스텝은 실행마다 ~3×10⁻⁵ 산포 — **resume끼리도 다름** → 체크포인트 결함이 아니라
  **연산의 실행 간 비결정성**. 파일럿(bit-identical) 이후 유입된 것으로,
  `CUDA_DEVICE_MAX_CONNECTIONS=1→32` 채택이 1순위 용의 (atomic 리덕션 순서). 학습 무해.
- 교훈: 과거 "bit-identical resume" 기록은 resume끼리의 비교였음 — 연속-vs-resume 비교는
  이번이 최초. blend 스위치 재기동마다 이 프로토콜(기준 5 iter 확보→재현 대조) 재사용 가능.

### 2.3 P2b 전환 후 loss V자 (08-10)

클러스터 평균 1.388 → 1.379(~700 iter) → 1.392(~900 iter) → 평탄. 전 보조지표 평탄
(grad norm 0.20-0.24, seq_aux 0.612-0.618, max attn logit 19≪clip 100, pseudo-grad rms 4.13e-4).

- 가설 기각 2건(체크포인트 실측): MoE 재배치 에피소드 없음(bias 이동량 P2 대비 1.05×),
  노드 간 bias 격차 미성장. → 해석: **혼합 성분의 비대칭 학습 속도** (신규 8%의 급락 포화 후
  기존 92%의 간섭 상승이 노출, 평형에서 정지). 병리 아님.
- valid @20,000 = 1.4296(양 노드 0.0002 이내 일치) — 22,000과의 고정 척도 비교 기준점.

### 2.4 ★ DiLoCo × MoE expert-bias 발산 (08-11, 프로덕션 개입)

**결함**: diloco_patch의 wire 집합이 `named_parameters()`여서 aux-loss-free `expert_bias`
(buffer)가 outer 동기화에서 제외 — 각 노드의 라우팅 선택 기준이 독립 진화. 레퍼런스
(Megatron 표준)는 카운트를 전 DP rank에 걸쳐 **합산**해 전역 단일 bias를 유지하므로,
이는 구현 부산물이지 설계가 아니었다.

**정량** (체크포인트 6개 10k~20k 분석, 24 layer × 192 = 4,608 expert):

| 시점(레짐) | mean\|d\| | max\|d\| | >0.05 |
|---|---|---|---|
| 10k~14k (자연) | 0.005~0.007 | 0.029 | 0 |
| 18k (P2 말, 상수 비대칭) | 0.0207 | 0.080 | 169 |
| 20k (P2b) | 0.0186 (광범위 완화) | **0.118** (꼬리 성장) | 72 |

- 격차는 현 blend의 샤드 비대칭을 따라감. **지속 코어 24 expert**(layer 2에 8, layer 20에
  10; diff 자기상관 +0.93). 갱신의 99.3%는 공동 이동(발산/공통 0.06).
- 영향: 훈련 loss 피해 <0.005(검출한계) — 그러나 ① blend 전환마다 꼬리 재성장,
  ② **최종 배포 모델의 `e_score_correction_bias`가 단일 샤드 균형으로 오염**(실害),
  ③ P3(최대 전환)에서 악화 전망. → 사용자 결정: 즉시 중단·수정.

**수정** (diloco_patch.py, `DILOCO_BIAS_SYNC=1` 기본):

1. **카운트 합산 동기화**: 매 step `tokens_per_expert`를 DiLoCo pair 간 **SUM** 후 Megatron
   표준 갱신 → 양 노드가 결합 배치(GBS 6,144) 통계로 동일 갱신 → bias 영구 bit-identical.
   bias 벡터의 평균/전송 없음(레퍼런스 sync-DP 의미론 확장). 18KB/step, Gloo ~ms.
2. **전용 pair group** (포트 +100): τ-오버랩 wire 스레드와 group 공유 시 collective 순서가
   노드 간 어긋나 오염되므로 분리 필수.
3. **checksum 감시**: 매 outer sync 로그에 `bias in sync: True/False`.
4. **fp32 master 보존 가드**: fresh 경로에서 params pair-identical이면 broadcast/
   `reload_model_params()` 생략 (reload는 fp32 master를 bf16 재유도로 훼손).

**재시작 의미론** (사용자 지시): node0의 ckpt 20,000을 양 노드가 로드(node1 원본은
`node1/iter_0020000_pre_biasfix` 백업), **outer 상태 신규 초기화**(θ:=params, momentum 0 —
학습 내용 손실 없음, 수 sync 내 재가열), blend/LR/H/τ 무변경. 폐기: 20,000~20,490.

**과정에서 잡은 구현 버그 2건** (스모크 게이트가 검출):

- **패키지 속성 그림자 → monkey-patch 무음 스킵**: `megatron.core.distributed`의
  `__init__`이 `finalize_model_grads` 이름을 서브모듈→함수로 덮어써서 `import ... as _F`가
  함수를 잡음. `hasattr()` 가드가 이를 조용히 스킵 → "설치 로그는 찍히는데 실제 경로는
  미패치". 수정: `importlib.import_module()`로 진짜 모듈 획득 + 가드 제거(실패 시 크래시).
  *교훈: monkey-patch에서 대상 부재는 예외지 정상 분기가 아니다.*
- **비대칭 디버그 collective**: 진단 계측을 env로 게이트했는데 launch_diloco의 ENVV
  화이트리스트가 node1에 전달 안 함 → 한쪽만 추가 collective 호출 → pair 통신 오염.
  수정: `EXTRA_ENV` 경유 전달. *교훈: pair collective는 반드시 양 노드 대칭 호출.*

**검증**: 스모크 5차(2노드 mock, H=3/τ=1) — 디버그 3중 대조(pre-bias/summed-counts/
post-update) 21/21 일치, `bias in sync: True`. 재기동 — 양 노드 20,000 로드, fresh outer,
fp32 가드 발동, 초기 bias checksum 동일(16105.270248413), 첫 iter 클러스터 평균 1.3925
(정지 직전 1.391과 연속).

### 2.5 ★ 거울상 시소 전면 재조사 → 세차운동 규명 → 블록-순환 샤딩 (08-17)

사용자 지시("이전 보고를 믿을 수 없다, 전부 재검토하라")로 §2.1 동적 형태를 독립
재조사했다. **정본 기록: [`../study/mirror_loss_aliasing.md`](../study/mirror_loss_aliasing.md)**
(재현: `../study/mirror_loss_repro.py`, 차트 보고서 아티팩트 링크 포함). 요지:

- **§2.1의 결론 방향은 확인**됐으나(구성 aliasing이 원인), 근거 산출물이 없었고 핵심
  질문 — 왜 P2는 정적이고 P2b부터 진동하나 — 이 미해결이었다.
- **신규 규명**: 6자리 가중치 합의 반올림 잔차가 방아쇠. P2 합=1.000000(정적),
  P2b 1.000001 / P3 0.999999. `normalize()`가 잔차를 나누며 블렌드 패턴이 짝/홀 격자
  위를 세차운동 → 패리티 주기 반전. **주기 = 2/|Σw−1| 샘플 = 325.5 iter** (실측 336).
  반사실로 인과 확정: 잔차 3e-6 → 주기 108 iter(1/3).
- **정량**: 반상관 −0.95(lm)/−0.996(seq bal), 제로섬(분산비 26~516×), 구성 스윙 ±8%p.
  실측 gap을 구성 델타 26변수 OLS로 R²=0.834(raw)/0.981(smoothed) 설명, ±1 iter
  시프트 시 0.40 붕괴, out-of-sample(08/07 로그) R²=0.842. 페어 합산은 매 iter 정확
  블렌드(≤1.4샘플) — §2.1의 "클러스터 합산 정확" 판정을 실측으로 확정.
- **유해성 재판정**: 측정 무해 + 2차 경로 유의 — 같은 부호 편향이 ~160 iter(5×H)
  지속되며 inner Muon 모멘텀 등 노드-로컬 상태는 outer 평균에 안 합쳐짐. §2.4의
  bias 발산이 이 압력의 실증 사례.
- **수정 구현**: `DILOCO_SHARD_BLOCK=3072`(=GBS) 블록-순환 매핑. P3 인덱스 실측 —
  노드별 구성 오차 137→**1.3**샘플, 노드 간 격차 274→**2**샘플. 전환은
  consumed % 3072 == 0 경계에서 중복 0/누락 0 (set-equality 검증 + 기동 assert +
  pair env 일치 assert). 부작용 없음(인덱스 산술만, 페어 합산 스트림 불변).
  단위 테스트 9종: `tests/test_diloco_shard_view.py`. **활성화는 다음 재시작에서**
  `DILOCO_SHARD_BLOCK=3072` — 라이브 P3 런은 레거시 매핑으로 계속 (예보: math ±272샘플,
  주기 ~325 iter — 실측 대조로 본 규명을 재검증 가능).

## 3. 현재 상태와 남은 작업

- 학습: iter 20,000+에서 p2b로 진행 중 (bias-sync 활성). **첫 outer sync(inner 31) 검증
  통과 — `bias in sync: True`, pseudo-grad rms 4.134e-4(정상 대역)**. 추가 확인 대기:
  22,000 체크포인트에서 노드 간 bias 격차 = 0 실측 (체크포인트 수준 재확인).
- 부수 실측 (재시작 후 로그 대조): node0은 iter 20,001이 폐기분과 완전 일치(자기 상태),
  node1은 −2.3e-5 차이(상태 교체의 지문) — bias 전면 교체의 즉각 loss 영향이 10⁻⁵ 오더임을
  직접 측정. "피해 <0.005" 상한의 재확인이며, 수정 근거는 누적·전파 경로(배포 오염)였음과 정합.
- P3 스위치(ckpt 24,000): 감시 재장착 필요. 사용자 결정 2건 대기 — P3 재원 배분(현안:
  cc_qa −5%p 최대 몫), decay 창 연장 여부(기각됨 — 유지).
- 문서 부채: CLAUDE.md "conn 변경 bit-identical" 기록 정정(§2.2), diloco_pilot.md에
  resume 검증 3중 비교 결과 추가.
- 커밋 완료 (08-17, branch `experiment/fp8-compute`):
  `0b3b308` feat(alpha) — blend preset 3종(`_p2/_p2b/_p3`) +
  `compute_stage2_switch_blends.py` + `run_cc_code_v5.sh`;
  `edd3a1d` fix(alpha) — `diloco_patch.py`(bias sync + `DILOCO_SHARD_BLOCK`) +
  `launch_diloco.sh` + stage2.yaml 헤더 + CLAUDE.md(alpha/root) +
  `study/mirror_loss_aliasing.md` + `study/mirror_loss_repro.py` +
  `tests/test_diloco_shard_view.py`. 본 문서 포함 `examples/alpha/docs/` 전체는
  사용자 결정으로 트래킹 전환·커밋됨(9ce9999; WANDB.md의 API 키는 redact —
  단 같은 키가 `scripts/setup_wandb.sh` 경유로 기존 히스토리에 존재, 외부 공개 전 rotate 필요).
- 다음 재시작 시: `DILOCO_SHARD_BLOCK=3072` 활성화 (§2.5; 명령은
  `study/mirror_loss_aliasing.md` §5). 전환 직후 노드별 곡선의 시소 소멸이 정상 신호.
