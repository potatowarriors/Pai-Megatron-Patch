# Post-LC SFT·RL(MOPD) 데이터셋 정리 (2026-08-01)

alpha 훈련 순서는 **LC-phase → SFT → RL(MOPD)**이며, RL 단계는 Nemotron 3 Ultra의
post-training 파이프라인(SFT → Student-RLVR → 전문 교사 RL → **MOPD** 증류)을
재현한다. 이 문서는 그 단계에 투입할 보유 데이터 자산의 전수 정리 + Ultra 레시피
매핑 + alpha 규모 적용 설계. LC 단계 데이터는 [`LC_DATASETS.md`](LC_DATASETS.md).

**보유고** (2026-08-01, `/home/work/Datasets/LL_datasets/posttraining/`):
SFT 25종 988G · RL 26종 62G(블렌드 3종 포함) — Nemotron-Post-Training-v3 컬렉션
49종 전체 + 구세대 6종. 전 항목 다운로드 검증 완료(50/50). 추가로 LongBlocks
193,894행(교사응답 3열 포함 — SFT/증류 소재, `LC_DATASETS.md` §5.1).

## 1. 재현 대상: Nemotron 3 Ultra post-training 파이프라인

NeMo RL 공식 레시피(github.com/NVIDIA-NeMo/RL, `ultra-v3` 브랜치 가이드) 기준:

```
SFT (Megatron-Bridge)                        ← SFT-* 데이터셋
  ↓
Student RLVR (GRPO, 검증가능 보상)
  Phase1: rlvr1.jsonl @ctx 49,152, ~128 steps, GBS 8192(512 prompts×16 gen)
  Phase2: rlvr2.jsonl @ctx 65,536, ~50 steps
  ↓
전문 교사 RL (4종 병렬; general 교사 = Student RLVR 자신)
  IFBench   교사: ifbench.jsonl   @49k,  lr 2.5e-6, GBS 2048
  RLHF      교사: rlhf.jsonl(GenRM) @49k, lr 2.5e-6
  Reasoning 교사: reasoning.jsonl @65k,  lr 3e-6, ≤10 epochs
  SWE       교사: swe.jsonl       @192k, lr 3e-6, GBS 512, ≤4 epochs
  ↓
MOPD (multi-teacher on-policy distillation) @ctx 192k
  mopd.jsonl — agent별로 교사 슬롯 라우팅, 학생이 per-agent 교사 분포를 매칭
```

MOPD 교사 라우팅(레시피 명시): general(=Student RLVR, 미지정 agent 폴백) /
RLHF(genrm_*) / IFBench(IF·abstention 계열) / Reasoning(math_with_judge,
equivalence_llm_judge, mcqa, ns_tools, code_gen) / SWE(SWE 계열 전부).

## 2. SFT 데이터셋 인벤토리

`used_in` 필드(행 단위 샘플 300×4파일)로 확인한 **데이터셋↔모델 매핑**이 조직 원리다.
alpha(15B-A3B)는 모델 급으로는 Nano(30B-A3B)에 가깝지만, 데이터 품질은 최신 버전이
개선판이므로 **ultra_v3 계열(최신) 우선 + 규모만 alpha에 맞게 서브샘플**을 권고.

### 2.1 ultra_v3 SFT 세트 (1순위)

| 데이터셋 | 크기 | 길이 p50/p90 (tok) | fit@32k/64k | 비고 |
|---|---|---|---|---|
| SFT-Instruction-Following-Chat-v3 | 19G | 5.6k/15.1k (chat) | 98.8%/100% | chat + IF 2파일 |
| SFT-Math-v4 | 19G | 9.9k/46.6k | 80.9%/97.4% | reasoning 트레이스 김 |
| SFT-Science-v2 | 50G | 0.7~4.1k/2.4~11.5k | ~100%/100% | rqa/so/syn_mcq 3파일 |
| SFT-Multilingual-v2 | 12G | 9.3~10.5k/20~22k | ~99.5%/100% | **ko/ja/hi/pt × code/math/stem** — hi만 alpha 미지원 |
| SFT-ARC-AGI-v1 | 18G | — | — | inductive reasoning |
| SFT-CUDA-v1 | 80M | — | — | 소형 특화 |
| SFT-Safety-v2 | 552M | — | — | super와 공유 |

### 2.2 버전 최신이지만 used_in 미표기 (1순위에 준함)

| 데이터셋 | 크기 | 길이 p50/p90 | fit@32k/64k | 비고 |
|---|---|---|---|---|
| SFT-OpenCode-v1 | 31G | 8.5~12.1k/19~25k | ~97%/100% | 코드 생성 |
| SFT-SWE-v3 | 11G | — | — | SWE-v2(super)의 후속 |
| Math-Proofs-v2 | 17G | **41.1k/95.6k** | 28.2%/**78.6%** | max 274k — 최장문 SFT |
| SFT-Competitive-Programming-v2 | 91G | 14~26k/62~76k | ~59-69%/85-91% | super 표기이나 CP 최신판 |

### 2.3 super_v3 / nano_v3 세트 (보조 — 구버전이거나 소형모델용)

super: Agentic-v2(6.9G)·IF-Chat-v2(15G)·Multilingual-v1(89G)·SWE-v2(17G)·Safety-v1·
SpecializedDomains-Finance-v1(30G, super 10회 표기).
nano: Agentic-v1·IF-Chat-v1·Math-Proofs-v1·**Math-v2(192G)**·SWE-v1(11G)·Science-v1.
→ 기본적으로 상위 버전이 있으면 제외. 예외 검토: Finance-v1(대체재 없음),
Math-v2(nano급 실증 — Math-v4와 중복도 확인 후 택1).

### 2.4 길이 실측 요약 (파일당 3k행 샘플, chars/4 근사)

전체 샘플 기준 **fit@32k 88.2% / fit@64k 96.8%** → **SFT max-seq 64k 확정**.
>64k 잔여 3.2%(주로 Math-Proofs-v2·CP-v2)는 증명/트레이스 중간 절단이 치명적이므로
드롭보다 **128k 소량 버킷**(LC 이후 가능) 권고. 캐비앳: 파일 앞 3k행 샘플이라 정렬
편향 가능성 있음 — 블렌드 확정 시 전수 재측정.

### 2.5 train_turns 실측 — "마지막 턴만 학습"은 chat split 한정 (2026-08-23)

`metadata.train_turns`(메시지 인덱스 기준 bool 리스트)는 **턴별 loss 마스크의
단일 진실 원천**이다. Chat-v3 두 split 전수 스캔(887,411행) 결과, 통념과 달리
두 split 의 규약이 다르다:

| split | 행수 | last-turn-only | **multi-True** |
|---|---|---|---|
| chat (`chat.with_prompts.jsonl`) | 637,663 | **100%** | 0 |
| IF (`instruction_following.jsonl`) | 249,748 | 39.1% (97,760) | **60.9% (151,988)** |

multi-True 예: 5메시지 대화의 `[F,F,T,F,T]` — **중간 assistant 턴도 학습 대상**
(IF 는 턴마다 제약 준수를 학습시키는 설계). 함정: 파일 **앞 30k행은 두 split 모두
100% last-only** 라 샘플 검사로는 오판한다 — 반드시 전수로 확인할 것.

**하류 영향**:
- `build_alpha_sft_idxmap.py` 는 이미 안전 — 리스트를 턴별로 일반 처리하고
  (`train_mask` 조립, ~L295), 리스트 부재 시 전 assistant 턴 True 폴백.
  multi-True 는 정상 마스킹된다. 코드 수정 불요.
- **(2026-08-24 갱신)** 마스킹은 안전했지만 **reasoning 소실은 별개 문제로 실재**:
  multi-True 의 중간 학습 턴은 템플릿이 history think 를 제거한 채 loss 를 받아
  IF 전수 기준 reasoning 26.7% chars 소실 + 빈 `<think></think>` 를 정답으로
  학습(no-think 오신호). → NVIDIA식 턴별 fan-out 채택:
  `build_alpha_sft_idxmap.py --fanout-train-turns` 로 IF 재변환
  (`chat_v3_if_fanout`, 변환기 docstring 의도적 차이 #2·유닛 6종).
  chat split 은 전수 last-only 라 대상 아님. tool 루프 셋은 기존 렌더가
  think 를 보존하므로 불요.
- **loss mask 를 다루는 모든 신규 작업**(디버깅·통계·합성 데이터 생산)은
  train_turns 를 "마지막만 True" 로 가정하지 말고 리스트 그대로 소비할 것.
- 학습 토큰 수 산정: last-only 가정 시 IF 의 학습 토큰이 과소평가된다.
- 합성 데이터 생산 규약 (ko_chat 트랙이 이 실측으로 모드 분기):
  chat 계열 = 마지막 턴 재생성(학습 턴이 네이티브), IF 계열 = 전량 번역
  (중간 학습 턴의 제약 준수 보존), 네이티브 생성 = 전 assistant True.

재현: `python3 - <<'E'` 로 각 jsonl 을 순회하며
`tt=r["metadata"]["train_turns"]; sum(tt)==1 and tt[-1]` 집계 (전수 ~2분).

## 3. RL 자산

### 3.1 훈련 블렌드 3종 (즉시 실행 가능한 레시피 — NeMo Gym 소비 포맷)

행 = 프롬프트 + `agent_ref`(환경/보상) + 검증 메타. 행 수 실측:

| 블렌드 | 파일별 행 수 |
|---|---|
| **Ultra** (재현 대상) | rlvr1 98,424 · rlvr2 99,116 · ifbench 34,649 · rlhf 6,500 · reasoning 5,236 · swe 7,816 · **mopd 85,980** |
| Super (참고) | rlvr1 138,712 · rlvr2 156,278 · rlvr3 107,037 · rlhf 25,171 · swe1 50,661 · swe2 1,444 |
| Nano (참고) | train 93,244 (11 agent 그룹 단일 블렌드) |

구성 상세(agent×dataset×source별 카운트)는
`posttraining/RL/nemotron_blend_recipe.json` (이 문서와 같이 생성).
**주의**: math 일부 행은 DAPO/Skywork 라이선스로 질문·정답이 마스킹 —
각 블렌드 동봉 `fill_placeholders.py`로 복원 필요(원본 HF 데이터셋 자동 다운로드).

### 3.2 RL 환경 데이터셋 26종 분류

| 분류 | 데이터셋 |
|---|---|
| IF 계열 (8) | RL-Instruction-Following-{Structured-Outputs-v2, Citation-Formatting, Free-Form-Formatting, Calendar-v2, MultiTurnChat, Adversarial} + RL-Identity-Following + RL-InverseIFEval |
| Agentic (4) | RL-Agentic-{Function-Calling-Pivot, Conversational-Tool-Use-Pivot, SWE-Pivot(4.8G), Indirect-Prompt-Injection} |
| Reasoning (5) | RL-Math-v2 · RL-Science-v1 · RL-ARC-AGI-v1 · RL-ReasoningGym-v1 · RLHF-GenRM-v1(5.1G) |
| Safety/기타 (3) | RL-Safety-v1 · RL-QA-Abstention-v1 · RL-litmus-bench-v0.1(평가·모니터링용) |
| 벤치 유래 (4) | RL-SysBench · RL-CFBench · RL-Multichallenge · RL-Multichallenge 계열 |
| 블렌드 (3) | §3.1 |

실행 스택: NeMo RL + NeMo Gym (둘 다 Apache 2.0 공개; 블렌드가 이 스택의 입력 포맷).

## 4. alpha 적용 설계

1. **컨텍스트 정합이 좋다**: Ultra의 RLVR ctx 49k→65k는 우리 SFT max 64k·LC 32k~64k
   계획과 자연스럽게 맞는다. 충돌 지점은 **SWE 교사·MOPD의 192k** — alpha LC 상한이
   128k이므로 **128k로 캡**(SWE rollout 축소) 또는 SWE 슬롯 축소가 필요.
2. **교사 패널 현실화** (Ultra는 550B 학생 + 전문 교사들; alpha는 15B-A3B):
   - general 교사 = alpha Student-RLVR 자신 (레시피 그대로, 추가 자원 불요)
   - 전문 교사 = alpha 체크포인트에서 각각 소규모 RL (교사 RL은 GBS 2048·수백 step
     규모라 우리 클러스터로 가능; 교사 수를 2~3종으로 축소 검토: Reasoning/IF 우선)
   - 외부 교사 보강: LongBlocks의 응답 3열(Qwen3-Next-80B 등)은 **오프라인 증류**
     소재로 즉시 사용 가능 — on-policy 전에 워밍업으로 유용
3. **한국어 SFT**: Multilingual-v2 ko 81,646행(2.8G) + ja/pt 동급. LC 한국어 갭과
   별개로 SFT 단계 한국어는 이것으로 상당 부분 커버.
4. **LC 능력 유지 게이트**: SFT 블렌드에 장문 샘플(LongBlocks doc-QA+응답, fit@64k
   84.6%) 수 % 포함 + 각 단계 통과 시 RULER@32k/64k(가능하면 128k) 회귀 측정.
   RL은 rollout 비용상 long-context 환경 입력 ≤32k (Nemotron Nano 관행).
5. **chat template**: SFT 데이터는 messages 포맷(system/user/assistant, tool 필드
   포함) — alpha tokenizer_v5의 chat template 정의·검증이 SFT 착수 전 선행 과제.

## 5. 데이터 준비 체크리스트 (SFT 착수 전)

1. `fill_placeholders.py` 실행 → Ultra 블렌드 math 마스킹 복원 (DAPO/Skywork 다운로드 수반)
2. ultra_v3 세트 + §2.2 4종의 **전수 길이 재측정**(tokenizer_v5) → 64k 버킷 구성 확정
3. Multilingual-v2에서 hi 제외 (alpha 미지원 언어 — `LC_DATASETS.md`의 20+2 언어 기준)
4. ~~chat template 정의~~ **완료 (2026-08-04)**: `tokenizer_v5/chat_template.jinja` =
   Nemotron 3 Ultra 템플릿 기반, 2026-08-24 DSV4 tool-시나리오 분기 추가로 바이트 동일 아님(4사 비교 후 채택 — Kimi-K3/Qwen3.5/
   GLM-5.2와 think·tool 규약 수렴 확인). `tokenizer_config.json`에 등록,
   `tools/verify_chat_template.py` 24개 테스트 통과. 변환기 구현 시 **필수 규약 2건**:
   ① 멀티턴 loss mask는 **assistant 스팬 스캔 방식**(prefix-diff는 히스토리 think 제거
   때문에 불성립 — 테스트가 실증), ② content 세그먼트는 `split_special_tokens=True`로
   인코딩(사용자 텍스트 내 `<|im_end|>` 등 injection 차단, Kimi-K3 규약).
   messages → idxmap 변환기(스팬 마스킹 적용)는 별도 구현 필요.
5. SFT 블렌드 비율 설계 (Ultra의 도메인 구성 참조: chat/IF·math·science·code·SWE·
   multilingual·safety) + LongBlocks-SFT 소량 편입
6. MOPD 재현 범위 결정: 교사 슬롯 수(2~3 vs 5), 192k→128k 캡, NeMo RL/Gym 스택
   포팅 vs 자체 구현(verl/ChatLearn 백엔드 검토)

## 6. 미해결/후속

- SFT-OpenCode·SWE-v3·Math-Proofs-v2·CP-v1의 used_in 부재 — 전 파일 스캔으로 확정 필요
- Ultra SFT 자체의 블렌드 비율은 미공개(레시피는 "SFT 체크포인트에서 시작"만 명시) —
  Megatron-Bridge SFT 레시피 공개 여부 추적
- litmus-bench 활용법(모니터링 셋) 조사
- 교사 rollout 서빙: sglang alpha 어댑터(`examples/alpha/sglang/`)의 batch 성능 실측
