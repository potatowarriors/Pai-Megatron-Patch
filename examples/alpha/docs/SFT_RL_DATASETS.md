# Post-LC SFT·RL(MOPD) 데이터셋 정리 (2026-08-01)

alpha 훈련 순서는 **LC-phase → SFT → RL(MOPD)**이며, RL 단계는 Nemotron 3 Ultra의
post-training 파이프라인(SFT → Student-RLVR → 전문 교사 RL → **MOPD** 증류)을
재현한다. 이 문서는 그 단계에 투입할 보유 데이터 자산의 전수 정리 + Ultra 레시피
매핑 + alpha 규모 적용 설계. LC 단계 데이터는 [`LC_DATASETS.md`](LC_DATASETS.md).

**보유고** (2026-08-01, `/home/work/Datasets/LL_datasets/posttraining/`):
SFT 25종 988G · RL 26종 62G(블렌드 3종 포함) — Nemotron-Post-Training-v3 컬렉션
49종 전체 + 구세대 6종. 전 항목 다운로드 검증 완료(50/50 — **2026-09-04 정정**: 존재·파싱만 본 검증이라
Agentic-v2 `tool_calling.jsonl` 절단(1.2% 만 수신)을 놓쳤다. 크기 검증은 HF API 의 LFS 크기와 대조해야 한다, §6). 추가로 LongBlocks
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
**Agentic-v2 — 미편입(2026-08-24) → phase-2 편입으로 번복(2026-09-04 사용자 결정)**: 미편입 근거였던
`used_in=super_v3` 는 **생성 세대 표시이지 사용 이력이 아니다**. Ultra 기술보고서 16쪽 "Search Capabilities" 는
Super 의 Wikidata 4~8홉 검색 트라젝토리(= 이 셋의 `search` split 5,968행, Tavily·MiniMax 2.1 교사)를 **retain**
했다고 명시한다. `tool_calling`·`interactive_agent` 는 used_in 없음(OpenCode·SWE-v3 와 같은 지위). 세 split 전부
phase-2 편입 — 설계·epoch 은 `SFT_PHASE2_PLAN.md` §11. 로컬 `tool_calling.jsonl` 은 절단본(8,444/707,052행,
0.44GB/14.94GB)이었음을 2026-09-04 발견, HF 재다운로드로 정정(크기·행수 일치 확인, 절단본은 `.truncated_*` 보존).
당시 판단의 기록은 `INTERLEAVED_THINKING.md` §5.

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

### 2.6 Reasoning effort / budget — IF split 의 정체와 RL 블렌드 마커 (2026-08-25)

결론: Ultra 는 effort 를 데이터 필드가 아니라 **렌더 시점 템플릿 kwarg** 로 심는다.
alpha 는 NVIDIA 레시피를 재현한다 (사용자 결정 2026-08-25).

| 사실 | 근거 |
|---|---|
| Ultra 3 모드: off / full / **medium effort** (`chat_template_kwargs={"medium_effort": True}` → 마지막 user 턴 끝 `{reasoning effort: efficient}`) + budget control (추론 시 `max_tokens` 절단 후 `</think>` 강제) | Ultra 기술보고서 §3.1.1 "Efficiency and Control", 모델 카드 |
| SFT 초기화 = ① GPT-OSS-120B medium-effort 샘플(math/STEM/IF) ② 기존 trace 를 무작위 예산으로 절단한 샘플(응답 동일, `</think>` loss 마스킹) → RLVR 에서 최적화 | 같은 절 |
| 공개 SFT 행에 흔적 없음: `chat_template_kwargs` 전수 null (IF-Chat-v3 chat 20k). **`instruction_following.jsonl` 자체가 GPT-OSS-120B medium-effort 데이터** (README:50, `metadata.model` 100%) — 마커는 학습기가 렌더할 때 붙는다 | 컬렉션 README·메타 실측 |
| 길이로는 구분 불가: IF think p50 6.9k chars > chat(GLM-5) 4.6k — medium 은 GPT-OSS high 대비 상대값 | 3k행 샘플 실측 |
| 절단-예산 컴포넌트는 미공개 (파생 데이터) | — |
| **RL 블렌드에는 이미 있다**: rlvr1 3,429/98,424 (3.5%) · rlvr2 3,429/99,116 (3.5%) · mopd 3,429/85,980 (4.0%); agent code_gen 1,618 / math_with_judge 995 / mcqa 816. 마스킹 math 행은 `_hf_question_placeholder.trail` 에 실려 `fill_placeholders.py` 복원 후 user 프롬프트 끝에 붙음. `alpha_blends/rlvr1_alpha.jsonl` 도 3,429 그대로 승계 | 전수 grep (마커 문자열 1종) |
| Super 보고서: RL 프롬프트 2%→1% 를 effort 모드로, 보상 = 정답 + 생성 토큰 수 조정 | Super §3.2.1 |

**alpha 적용 (NVIDIA 레시피 재현)**

| 단계 | 조치 | 구현 |
|---|---|---|
| SFT ① | IF split 을 `--medium-effort` 로 재변환 → `chat_v3_if_fanout_me` (fan-out 결합: 서브샘플마다 학습 턴 직전 user 에 마커 = 추론 토큰열) | `build_alpha_sft_idxmap.py` docstring §Effort/Budget, `convert_sft_64k.sh` |
| SFT ② | 절단-예산 파생 셋 `budget_trunc_v1_{if,math}`: 학습 턴 reasoning 을 B=int(L·U(0.1,0.9)) 토큰으로 절단(응답 불변), 잘린 자리 `</think>` 비학습, (seed,uuid,턴) 결정적. 블렌드 1% 슬롯 제안 (Super low-effort 2% 참조; 나머지는 ①이 담당) | 〃 `--truncate-reasoning-budget --row-stride` |
| RL | 마커 프롬프트 유지. 길이 보상은 NeMo-RL 내장 `env.nemo_gym.effort_levels` (`low_string="{reasoning effort: efficient}"`, `low_weight>0`, `low_ub`, `low_penalty`; `lr=min(1, w·(1−len/ub))`, `reward += reward·max(lr,0) + penalty·min(lr,0)`) — GRPO yaml 에 설정 필수 | `NeMo-RL/nemo_rl/experience/rollouts.py:330` |
| 검증 | `verify_chat_template.py` 34 (medium_effort 4종) · `test_alpha_sft_idxmap.py` 34 (§6 effort/budget 9종) · 스모크 변환 IF 3k/2k 행 `verify_sft_bins` PASS, 절단률 실측 50.8% (=U 평균) | tests |

본 변환·블렌드 반영 완료 (2026-08-25, sub1에서 ko_chat vLLM 과 병행 ~3분):

| 셋 | rows→samples | real | trainable | 절단률 | 게이트 |
|---|---|---|---|---|---|
| `chat_v3_if_fanout_me` | 249,748→529,472 | 1.0439B (+8.0 tok/샘플 = 마커) | 686.05M (**구본과 동일** — 마커 비학습 실증) | — | PASS |
| `budget_trunc_v1_if` (stride 4) | 62,437→123,291 (trunc_none 8,987 드롭) | 194.75M | 115.56M | 49.7% | PASS |
| `budget_trunc_v1_math` (stride 20) | 27,272→27,192 | 175.05M | 166.55M | 50.0% (37,108턴 전부 `</think>` 마스킹) | PASS |

블렌드(`sft_40b_blend.yaml`): 설계단위 chat 21 → 20 + budget_trunc 1, 카테고리 합 0.230180 불변,
내부 real-token 재비례 — if_me 0.069680 / chat 0.149539 (ep 2.81→2.67), budget 0.005772/0.005189 (각 1.19ep).
타 엔트리 불변, 23경로 전부 .idx 확인.

사전 점검 재확인 (2026-08-28, SFT 개시 전):
- KoChat-v1 합류(08-26) 후에도 effort 엔트리 보존 — 블렌드 26경로 합 0.999998, budget 가중치 불변(1.19ep),
  chat 계열은 KoChat 합류로 2.67→2.28ep 재비례 (가중치 정본은 yaml 헤더 08-26 주석).
- KoChat IF 트랜치(t2 포함)도 규칙 8대로 `--medium-effort --fanout-train-turns` 렌더 실증
  (`kochat_if_fanout_me{,_t2}` stats: medium_effort=True). kochat_chat/b 는 마커 없음 — 원본이
  GPT-OSS medium-effort 생성분이 아니므로 정상.
- 128k 버킷(`sft_128k_blend.yaml`)에 effort 셋 없음은 **의도** — Ultra effort 도메인(math/STEM/IF)은
  전부 64k 수용, 128k 는 long-tail 6종 전용. `sft_smoke_64k.yaml` 이 구 `chat_v3_if` 를 가리키는 것도
  무해 (preset 스모크 전용 단일셋, 학습 블렌드 아님).

### 2.7 128k 혼합 블렌드 실측 검증 (2026-09-01, phase-1 iter 1,045 시점)

본 런 `data_path` = `sft_128k_mixed_blend.yaml` **26/26 일치**. 예산 2,448 iter × GBS 160 = 391,680 샘플 = 51.34B bin-tok.
epoch = w × 51.34B / real_tokens(`data.stats.json`). 설계값(SWE 1-pass·chat 공통·budget 1%·대형 코퍼스 sub-1ep) 그대로 실현.

| 멤버 | w | ep | trainable/real | 비고 |
|---|---|---|---|---|
| cp_v2 | 0.2367 | 0.45 | 97% | |
| swe_v3_keepthink | 0.1873 | **1.00** | 29% | 앵커 |
| science_v2 | 0.1159 | 0.40 | 96% | |
| math_v4 | 0.1117 | 0.86 | 97% | |
| chat_v3_chat | 0.0830 | 1.89 | 59% | 복원율 88.2% 정본 — 잔여 11.8% 회수 불가 확정 (KNOWN_ISSUES 09-01 ③) |
| opencode_v1 | 0.0427 | 0.31 | 17% | tool 결과 Python repr · reasoning 0% (〃 ①) |
| chat_v3_if_fanout_me | 0.0385 | 1.89 | 66% | fan-out + effort |
| arc_agi_v1_keepthink | 0.0340 | 0.20 | 59% | |
| kochat_chat_t2 / kochat_if_fanout_me_t2 / kochat_b_fanout_t2 | 0.0247 / 0.0238 / 0.0012 | 1.89 | 37 / 70 / 62% | chat 공통 epoch |
| math_proofs_v2 | 0.0216 | 0.28 | 95% | |
| ml_* 9종 | 합 0.0575 | 1.00 | 86~99% | |
| safety_v2 | 0.0085 | 4.31 | 81% | E_max 4~5 상단 |
| budget_trunc_v1_if / _math | 0.0045 / 0.0042 | 1.19 | 59 / 95% | |
| identity_v1 | 0.0043 | 15.4(×12 파일) = **원본 ≈180회** | 61% | 〃 ② |
| cuda_v1 | 0.0001 | 0.31 | 57% | |

드롭은 전부 설계된 사유: too_long(>128k: proofs 6,563·arc 1,555·swe 1,193), injection(chat 97·science 68),
trunc_none(budget_if 8,987), null_content(chat 75,287). 게이트 `verify_sft_bins` PASS·fill 98.7~100%(커밋 4e12f2d).
loss: train 1.051 → 0.716(iter 1,045), valid 0.657 → 0.617 → 0.599(300/600/900) 단조 감소.

에이전틱 셋 형식 대조(표본 300행/파일): swe_v3 tool content **str 100%**·reasoning 66%(9,287/14,117) · arc_agi str 100%·
reasoning 65% · **opencode list 100%·reasoning 0%**. 결함 3건의 서사는 `KNOWN_ISSUES.md`(2026-09-01), 수정 계획은
`SFT_PHASE2_PLAN.md`.

### 2.8 미사용 SFT 셋 실측 인벤토리 (2026-09-04)

보유 29 디렉터리 중 phase-1 블렌드 원천은 13개. 나머지 16개 중 Identity-v1·KoChat-v1 은 v2 로 대체된 셋이고,
13개는 **phase-1 어디에도 없는 독립 데이터**다(§2.3 의 "상위 버전이 있으면 제외"는 세대 우선 규칙이었고 포함 관계가
아니다 — 첫 user 프롬프트 해시 겹침: Safety v1→v2 96.4%, Chat v2→v3 5.1%, Agentic v1→v2 0%, Science v1→v2 0%).
토큰은 1,500~2,000행 표본을 실제 변환기(`--measure-only`)로 렌더해 행수로 외삽한 값.

| 셋 | 추정 real 토큰 | trainable | 교사 | phase-2 처리 (§`SFT_PHASE2_PLAN.md` §11) |
|---|---|---|---|---|
| Math-v2 (nano) | 70.9B | 99% | gpt-oss-120b | 제외 — math_v4 가 0.86ep 로 미소진 |
| Competitive-Programming-v1 | 54.8B | 97~100% | DeepSeek-R1-0528 | 제외 — cp_v2 가 0.45ep 로 미소진 |
| Math-v3 | 51.4B | 99% | DeepSeek-V3.2 | 제외 — 〃 |
| Multilingual-v1 (de/es/fr/it/ja/zh, 전부 `ALPHA_LANGS`) | 24.5B | 97~99% | Qwen 번역 백본 | 0.05ep |
| Math-Proofs-v1 lean (`messages=="[]"` 행 ≈33% bad_row) | ≈11B 유효 | 81% | gpt-oss-120b | 0.05ep (RL `math_formal_lean` 환경 대비) |
| Finance-v1 | 9.4B | **4.9%** | GPT-OSS-120B | 0.1ep |
| Agentic-v2 (search 0.14B · interactive_agent 1.56B · tool_calling 3.92B) | 5.6B | 18~36% | MiniMax 2.1 / DeepSeek-V3.2 | search 2.0 · ia 0.25 · tc 0.10 |
| SWE-v2 (openhands 2.3B reasoning 0% · agentless 1.5B) | 3.8B | 25~47% | Qwen3-Coder-480B | 0.3ep (swe_v3 리플레이 대체) |
| Chat-v2 (reasoning_on 2.1B · reasoning_off 1.3B = no-think) | 3.4B | 76~87% | Qwen3-235B, Kimi-K2-Thinking | 0.3ep (chat_v3 리플레이 대체) |
| SWE-v1 r2e_gym (reasoning 0%) | 2.4B | 27% | Qwen3-Coder-480B | 0.3ep |
| Chat-v1 (nano) | 1.6B | 89% | GPT-OSS-120B | 제외 |
| Agentic-v1 (tool_calling 변환기 크래시 — bool 필드) | ≈1.4B | 9% | Qwen3-235B | 제외 |
| Science-v1 | 0.6B | 86~88% | GPT-OSS-120B | 1.0ep (science_v2 리플레이 대체) |
| Safety-v1 | 0.03B | 82% | — | 제외 — v2 와 프롬프트 96% 중복 |

합계 ≈240B = phase-1 예산(51.34B)의 4.7배. 편입 여부가 아니라 epoch·iters 가 결정 변수다.

### 2.9 터미널 에이전트(Terminal-Bench 형) 데이터 실측 (2026-09-07)

질문: phase-1·phase-2 에 Terminal-Bench 에 맞는 터미널 에이전트 데이터가 있는가. 답: **phase-1 에 극소량, phase-2 신규분에는 없음.**

| 원천 | 하니스 (SWE-v3 전수 237,970행 분류) | 행 비중 | 블렌드 토큰 비중(추정) |
|---|---|---|---|
| SWE-v3 | **Terminus 형**: system "solving command-line tasks in a Linux environment", 응답 JSON `{"analysis","plan","commands":[{keystrokes,duration}],"task_complete"}`(= **Terminus-2 스키마**), 터미널 출력은 user 턴 "New Terminal Output:" 주입, reasoning 37%, 턴 중앙값 67, **과제는 전부 GitHub 이슈**(일반 터미널 과제 0) | 1.5% (≈3,540행) | ≈0.28% |
| SWE-v3 | bash 전용 셸 에이전트(mini-SWE-agent, tool `bash` 하나) | 9.5% | ≈1.2% |
| SWE-v3 | SWE-agent(bash+editor+submit) 20.0% · OpenHands 36.0% · opencode 6.8% · Codex CLI 1.2% · 기타 SWE 에이전트 프롬프트 25% | 89% | ≈16.7% |
| OpenCode-v1 | `bash_only_tool` + `bash_only_tool_skills` 193k행(42%) — 도구는 bash 뿐이나 과제는 코딩 조수 질문(표본 19% 는 순수 개념 질문) | 42% | ≈1.8% |
| phase-2 신규 | swe_v2_openhands·swe_v1_r2e = 100% OpenHands(execute_bash+str_replace_editor), agentless 도구 없음, Agentic-v2 = API 함수 호출·웹 검색 | — | Terminus 형 **0** |

Ultra 는 Terminus-2(Harbor)로 **≈370K 대화**의 전용 terminal-use 셋을 만들어 SFT 했고(기술보고서 "Terminal-Use Capabilities": 시드 OpenCodeReasoning·
OpenMathReasoning·SWE-bench·SWE-Fixer·SWE-rebench·SWE-smith, DeepSeek-V3.2 에이전트, 일반 터미널 과제 포함) 미공개다. 공개 컬렉션에 남은 것이
SWE-v3 의 3.5k행이다.

**평가 형식 불일치**: 현행 평가는 terminal-bench 0.2.18 + core 0.1.1(80 tasks) + Terminus **v1**(4필드 JSON `state_analysis/explanation/commands/
is_task_complete`)인데 학습 데이터는 Terminus-2 스키마다. 스키마는 하니스 에이전트가 정하므로 "v1 하니스에 v2 스키마 적용"은 성립하지 않는다 →
**결정(사용자, 2026-09-07): Terminal-Bench 2.x + Harbor + Terminus-2 로 평가 경로 전환**(학습 형식·Ultra TB 2.0/2.1 비교 조건 일치, 기존 유효
수치 없음). 벤치 세션이 진행(`SFT_BENCHMARKS.md`).

보강 선택지: ① SWE-v3 Terminus 3.5k행을 별도 멤버로 추출해 2~3ep 상향(실토큰 ≈0.14B) ② **Terminus-2 형 터미널 트라젝토리 합성(Ultra 방식 축소
재현) — 2026-09-07 사용자 결정으로 타당성 검토 착수**(별도 세션, H100×1노드 + GLM 5.3 open weight 교사; 상태는 `STATUS.md`) ③ RL 에서 Gym
`terminus_judge`·`terminal_multi_harness_*` 환경으로 보강(Ultra 의 terminal 교사도 RL).

**Ultra 의 terminal-use 데이터 제작 레시피 (기술보고서 §Terminal-Use Capabilities·§Software Issue Resolution·§Terminal-use Teacher·부록 A.2; 데이터 자체는 미공개)**

| 단계 | Ultra | alpha 재현 시 대응물 |
|---|---|---|
| 시드 | OpenCodeReasoning · OpenMathReasoning · SWE-bench · SWE-Fixer-Train-110K · SWE-rebench · SWE-smith (전부 공개) | 동일 공개 셋 + 보유 Nemotron 셋 |
| 과제 조립 | (a) Cascade 수학·코딩 SFT 데이터를 터미널 환경용으로 재포맷 (b) DeepSeek-V3.2 로 "기존 벤치에 없는 터미널 시나리오" 합성 | (a) 보유 math/code 셋 재포맷 (b) 교사 LLM 로 시나리오 합성 |
| 트라젝토리 수집 | **Harbor 프레임워크의 Terminus-2 에이전트** 안에서 DeepSeek-V3.2 가 행동 주체, 실제 터미널 환경과 다중 에피소드 상호작용 | Harbor(오픈소스) + Terminus-2 + gpu06 DinD 컨테이너(도커 필요, `EVAL_DOCKER_NODE.md`) + 교사 = GLM 5.3(검토) |
| 형식 | Terminus-2 JSON `{"analysis","plan","commands":[{keystrokes,duration}],"task_complete"}`, 터미널 출력은 user 턴 "New Terminal Output:" | 동일 (SWE-v3 Terminus 행과 바이트 수준 정합, 렌더 검사 규칙 9) |
| 규모·구성 | ≈370K 다중턴 대화, reasoning·non-reasoning 혼합 | 초기 목표는 타당성 검토 후 결정 (참고: SWE-v3 Terminus 3.5k) |
| 하니스 다양화 | 모든 과제 분포를 Stirrup·OpenHands·OpenCode·Terminus·Droid·내부 중 **≥2 하니스**로 학습 (부록 A.2) | Terminus-2 + OpenHands/opencode 병행 |
| 품질 필터 (SWE 절에 기술) | 제출 무결성 · 금지 git 명령(push/pull/fetch/clone/cherry-pick/reflog/fsck/remote/ls-remote) · 편집-테스트 무한 반복 · 탐색만 하고 편집 없음 · 도구 호출 불량률 · 디버그 잔재(print/pdb/breakpoint) · 편집 후 테스트 미실행 | 동일 7신호 휴리스틱 + 과제 성공 판정(테스트 또는 judge) |
| RL 교사 | 최대 1시간 타임아웃 과제의 전문가 트라젝토리 → PivotRL(arXiv 2603.21383) 반복 개선(포화 시 re-profiling) → MOPD 로 증류 | Gym `terminus_judge`·`terminal_multi_harness_*` + GRPO |
| 평가 | Terminal-Bench 2.1 (Harbor) | TB-2 + Terminus-2 (`SFT_BENCHMARKS.md`) |

미공개: 과제 합성 프롬프트, 필터 임계값, 에피소드 분할 규칙, 터미널 RL 보상 정의, 교사 크기·PivotRL 하이퍼파라미터.

**정정·추가 (2026-09-10)**: 위 "≈370K 미공개" 는 틀렸다 — NVIDIA 가 `nvidia/Nemotron-Terminal-Corpus`(366,154행, cc-by-4.0, arXiv 2602.21193)를 Post-Training-v3
컬렉션 밖에 단독 공개했고(형제 `Nemotron-Terminal-Synthetic-Tasks`, RL `Nemotron-RL-Agentic-Terminal-Pivot-v1`, `Nemotron-Cascade-2-SFT-Data` Terminal Agent 스플릿),
이 조사가 놓쳤다. 사용자 결정으로 자체 합성은 폐기하고 공개 코퍼스를 채택했다. 정본·변환·게이트·교훈은 `sdg/terminal/README.md`, 사고 서사는 `KNOWN_ISSUES.md` 2026-09-10.

### 2.10 SFT 데이터 일관성 검토 (2026-09-09) — 정본은 `KNOWN_ISSUES.md` 2026-09-09

51 멤버의 원본 구조(400행/멤버)와 packed bins 토큰열(40문서/멤버)을 스캔했다. 결함 5건(SWE-v3 도구 선언 부재·opencode 스키마 렌더 파손·
Chat-v2 on 히스토리 think 학습·identity fan-out 누락·agentless user 프롬프트 특수토큰)과 도구 영역 무사고 타깃 비중(p1 ≈35% → p2 43.6% →
p3 22.9%)은 그 항목에, 교정 멤버 4종은 `convert_sft_128k_terminal_fix.sh`. 스캔 스크립트·산출물: `/home/work/vidsearch/tools/sft_consistency/`.

### 2.11 SFT 멤버 개념 분류 — general / agentic 두 SFT base 를 위한 버킷 (2026-09-22, 사용자 결정)

**목적.** SFT 뒤 MOPD 는 Ultra 와 같이 **두 SFT base** 에서 출발한다 — general(STEM·Chat·IF teacher 의 base, reasoning·chat 비중 ↑)
과 agentic(agent·coding·reasoning 비중 ↑). 현 57 멤버(`SFT_FINAL_PLAN.md` §1.1)를 두 블렌드에 배분하기 위한 **개념 분류**이며
디렉토리·파일은 옮기지 않는다. 최종 블렌드 비율은 이 절이 아니라 각 블렌드 계획 문서에서 정한다.

**분류 축 = 상호작용 형태.** 도메인(code/math/…)이 아니라 형태로 자른다 — reasoning 은 단일 턴 사고, agent 는 다중 턴 도구 루프라
RL 단계(PivotRL·MOPD)의 rollout·verifier 가 형태로 갈린다. **code 는 독립 버킷이 아니다**: 경쟁 프로그래밍(1턴 사고)은 reasoning,
저장소·터미널 조작(도구 루프)은 agent. Ultra Figure 10 도 Coding Teacher 는 STEM 계보, SWE Teacher 는 agentic 계보에 둔다.
도메인은 태그로 남겨 두 블렌드에서 code 비중을 따로 조절한다.
근거 측정(2026-09-21): 원천 파일 57 종 × 300 행 — assistant 턴 수·think 비율·도구 선언/호출.

| 버킷 | 기준 | 멤버 수 | bin-tok(현 런) | 학습 tok | general SFT | agentic SFT |
|---|---|---|---|---|---|---|
| **reasoning** | 1턴·긴 사고·도구 없음·검증 가능 정답 | 27 | 28.37B (47.3%) | 27.24B (56.6%) | ↑ | 유지 |
| **chat** | 대화·지시 준수·형식 준수, 사고 짧음 | 9 | 16.89B (28.1%) | 12.21B (25.4%) | ↑ | ↓ |
| **agent** | 다중 턴 도구 루프 또는 도구 호출 판단 | 18 | 6.20B (10.3%) | 2.14B (4.4%) | ↓ | ↑↑ |
| ~~shared~~ → **reasoning** | `science_v2` — 09-22 정정(§2.12): reasoning 으로 확정 | 1 | 6.54B (10.9%) | 6.30B (13.1%) | 포함 | reasoning 몫 |
| **etc/long-context** | `finance_v1` | 1 | 1.73B (2.9%) | 0.08B (0.2%) | LC 유지용 | LC 유지용 |
| **etc/identity** | `identity_v2_fanout` | 1 | 0.30B (0.5%) | 0.16B (0.3%) | 소량 | 소량 |

**reasoning (27)**

| 도메인 태그 | 멤버 | bin-tok | 형태 근거 |
|---|---|---|---|
| code | `cp_v2` | 13.07B | 1턴, think 100%, 도구 0 |
| code·다국어 | `ml_ultra-v3_code_{ja,ko,pt}` `ml_super-v3_code_{de,es,fr,it,ja,zh}` | 1.88B | 1턴 번역본 |
| math | `math_v4` `math_proofs_v2` `math_proofs_v1_lean` `budget_trunc_v1_math` | 10.22B | 1턴, think 71~100% |
| math·다국어 | `ml_ultra-v3_math_{ja,ko,pt}` `ml_super-v3_math_{de,es,fr,it,ja,zh}` | 2.01B | 1턴 번역본 |
| science | `science_v1` | 0.83B | 1턴 MCQ/RQA |
| stem·다국어 | `ml_ultra-v3_stem_{ja,ko,pt}` | 0.37B | 1턴 번역본 |

**chat (9)**

| 하위 | 멤버 | bin-tok |
|---|---|---|
| 일반 대화 | `chat_v3_chat` `chat_v2_on_scrub` `chat_v2_off_scrub` `kochat_v3_think` `kochat_v3_nothink` | 11.22B |
| 지시 준수 | `chat_v3_if_fanout_me` `ifchat_v1_chat_if`(capability_target chat 80%·IF 20%) `budget_trunc_v1_if` | 5.46B |
| 구조화 출력 | `ifchat_v1_structured` | 0.16B |

**agent (18)**

| 하위 | 멤버 | bin-tok | 형태 근거 |
|---|---|---|---|
| code(저장소·터미널 조작) | `swe_v3_tools_keepthink` `opencode_tools` `cuda_v1` `ntc_v1_swe` `ntc_v1_code` | 3.64B | `cuda_v1` 은 이름과 달리 OpenCode 에이전트 궤적(bash·glob·read·write, assistant p50 14턴·max 108) |
| terminal | `ntc_v1_math` `ntc_v1_syn_easy` `ntc_v1_syn_medium` `ntc_v1_syn_mixed` | 0.62B | Terminus-2 다중 턴 |
| search | `agentic_v2_search` `search_ko_v1` `research_ko_v1` | 0.25B | p50 4~12턴 검색 루프 |
| tool-call | `agentic_v2_tc` `agentic_v2_ia` `kotool_v1_x2` `when2call_v1_x2` | 1.24B | API 호출·대화형 에이전트·미호출 판단 |
| usability·safety | `usab_v1` `safety_v2` | 0.45B | 사용자 결정 09-22. Ultra Figure 10 도 Usability·Agentic Safety Teacher 를 agentic 계보에 둔다 |

**shared — `science_v2`** (rqa/so/syn_mcq 3 split). 내용은 과학 QA 이지만 행의 80~96% 가 `evaluate_code`(선택지 채점) 도구를
1회 호출하는 `<tool_call>` → tool 응답 → 최종 답 사이클이다. reasoning 에는 내용을, agent 에는 도구 문법과 tool 응답 뒤 재개를
가르치므로 양쪽에 넣는다(사용자 결정 09-22, agent 규모 보강 겸). 단 agent 계열에 합치면 학습 tok 8.44B 중 75% 가 이 셋이라
**agentic 블렌드의 실제 도구 루프 비중은 science_v2 를 제외하고 따로 관리**한다. 재료는 §1.1 미사용 에이전틱 22.5B.

**etc/long-context — `finance_v1`** (SEC 10-K/10-Q 템플릿 SDG, 326,698행). alpha tokenizer 1,500행 표본(2026-09-22):

| | mean | p10 | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| 전체 | 27,736 | 17,258 | 30,396 | 30,642 | 30,995 | 31,530 |
| user(문서+질문) | 27,507 | 17,129 | 30,213 | 30,317 | 30,391 | 30,651 |
| assistant | 229 | 96 | 195 | 400 | 720 | 1,254 |

≤8K 1.9% · ≤16K 9.3% · ≤32K 100%. 길이가 **30K 상한에 몰린 분포**(생성 시 문서를 ~30K 에서 절단)라 128K 창의 1/4 만
커버하고, 답이 ~200 tok 이라 train% 4.8. long-context 유지용 소량 외의 용도(reasoning 뼈대)로는 쓰지 않는다(사용자 결정 09-22).
128K 급 연습이 필요하면 문서 다중 결합 재구성이 선행돼야 한다.

**다국어 21 종**은 별도 버킷이 아니라 reasoning 안의 태그다 — 전부 1턴 사고형 번역본이라 형태 기준에 맞는다(ko 우선).

### 2.12 두 SFT base 블렌드 목표 — general = 현 런 ckpt, agentic = 추가 SFT (2026-09-22, 사용자 결정)

**정정 (09-22).** §2.11 의 `science_v2` 는 shared 가 아니라 **reasoning** 으로 확정. agent 버킷은 도구 루프만이다.

**전제 두 가지.**
- "agent 약 6B" 는 현 런의 **소비량**(0.15~0.37 에폭)이지 보유량이 아니다. 버킷별 보유 풀(1 에폭):

| 버킷 | bin-tok | 학습 tok | train% |
|---|---|---|---|
| agent/code (SWE·opencode·cuda·NTC swe/code) | 18.52B | 4.82B | 26% |
| agent/terminal | 4.18B | 2.51B | 60% |
| agent/tool-call | 5.51B | 1.82B | 33% |
| agent/search | 0.34B | 0.05B | 15% |
| agent/usab+safety | 0.14B | 0.12B | 86% |
| **agent 합** | **28.69B** | **9.32B** | 32% |
| reasoning (science_v2 포함) | 90.25B | 85.84B | 95% |
| chat | 9.76B | 7.07B | 72% |
| etc (finance + identity) | 9.76B | 0.46B | 5% |

- **블렌드 목표는 gradient(학습 tok) 기준으로 정하고 bin 비중으로 역산한다.** train% 가 버킷마다 5~95% 라 같은 bin 비중이라도
  학습 신호는 최대 20배 다르다. 예: bin A40/R30/C15/E5 는 gradient 로 agent 25 / reasoning 54 / chat 21 — "agentic" 이 아니다.
  학습 tok 은 변환기가 chat template 로 렌더한 열 위에서 센 값이라 tool 시나리오의 이전 턴 think 보존(DSV4 7(a), NTC 는
  `--keep-history-think` 강제)이 반영돼 있다.

**general SFT base = 현 런(`sft_128k_final`) 최종 ckpt 그대로.** 현 런은 bin 기준 reasoning 58 / chat 28 / agent 10 / etc 3,
gradient 기준 reasoning 70 / chat 25 / agent 4 / etc 1. STEM teacher 의 출발점(Ultra STEM teacher SFT 는 reasoning 92.5%)으로
reasoning 우위가 맞고, agent 4% 는 도구 문법 노출로 MOPD warmup 을 대신한다. 제안됐던 R40/C40/A+E20 재실행은 기각 —
chat 풀 9.76B 라 60B 에서 2.5 에폭, 현 런 1.7 에폭과 누적 4 에폭 초과(문체 암기 위험), reasoning 은 오히려 감소.
**조건**: 최종 ckpt(iter 2,862) T1·T3·에이전틱 게이트 통과. chat·IF 가 iter600/1800 대비 퇴행이면 5~10B 낮은 LR chat/IF anneal 만 얹는다.

**agentic SFT base = general ckpt 에서 추가 SFT, 40~60B.** 반복은 어느 예산에서도 없다(60B·A58 에서 agent 1.21 에폭).
예산은 시간이 정한다: 60B = 단일 노드 ≈10.4일(314 s/iter) / DiLoCo 2노드 ≈5.4일, 40B = ≈7일 / ≈3.6일.
**사용자 결정 09-22: 1차 예산 60B.** 이어서 하는 학습이라 짧은 warmup 뒤 **constant LR** 만 유지하므로(cosine 아님), 60B 로 계획하고
40B 에서 중단해도 스케줄 왜곡이 없다. 중단 시점 ckpt 를 그대로 base 로 쓰되, 짧은 decay 를 얹을지는 그때 게이트로 판단.

| 버킷 | bin 비중 | 60B 학습 tok | gradient 비중 | 60B 에폭 |
|---|---|---|---|---|
| agent | **58%** | 11.3B | **37%** | 1.21 |
| reasoning (science_v2 포함, 어려운 서브셋 우선) | 22% | 12.5B | 41% | 0.15 |
| chat (망각 방지 replay) | 15% | 6.5B | 21% | 0.92 |
| etc (identity 0.5 + finance ≤1.5 권고; 5% 유지 시 finance 는 그 안에서) | 5% | 0.14B | 0.5% | — |

agent 58% 내부 배분(풀 비율 code 65 / tool-call 19 / terminal 15 / search 1.2 를 목표 벤치 SWE-bench·TB-2·τ³ 에 맞게 조정):

| 하위 | 비중 | 60B 에폭 | 비고 |
|---|---|---|---|
| code | 45% | 0.85 | SWE·opencode·cuda·NTC swe/code |
| terminal | 25% | 2.1 | |
| tool-call | 25% | 1.6 | agentic_v2 tc/ia·kotool·when2call |
| search | 5% | 5.1 | 풀 0.34B. **사용자 결정 09-22: 5%·5 에폭 그대로 간다**(SDG 추가 생성 없음) |

소형 셋(search·kotool·when2call·usab·identity)의 4~8 에폭은 생성기 `fixed` 멤버로 고정한다. 생성기는 §1.1 의 bin-tok 기준 정정을
먼저 반영한다(`SFT_FINAL_PLAN.md` §1.1). 재개 중 가중치 변경 금지(`KNOWN_ISSUES` 09-01 ④)이므로 새 블렌드는 새 런이다.

**산출물 (2026-09-22, 커밋).**

| 파일 | 내용 |
|---|---|
| `toolkits/sft_data_preprocessing/gen_sft_blend_v2.py` | 생성기 v2 — bin-tok 기준(samples = ep × bins × 0.99), 2단 목표(`--targets`, `--sub-targets`), fixed 멤버. 구 생성기의 fill 편차 정정 |
| `toolkits/sft_data_preprocessing/sft_128k_agentic_spec.tsv` | 57 멤버 bucket/sub/base_ep/fixed. cuda 4·usab 4·safety 3·kotool 2(≡4)·when2call 3(≡6)·structured 8·identity 18 고정 |
| `examples/alpha/configs/data/sft_128k_agentic_blend.yaml` | 생성 결과. bin agent 58 / reasoning 22 / chat 15 / etc 5 → **gradient 39.5 / 39.5 / 20.1 / 0.9**, 학습 tok 32.08B. 검증: `audit_blend_epochs.py` 와 57 멤버 ep 최대 차 0.001 |
| `examples/alpha/configs/training/sft_128k_agentic.yaml` | `sft_128k_final.yaml` 전체 복제 + diff 6키: warmup 50 iters, **constant LR**, load = general 최종 ckpt, finetune + no-load-optim. **⚠ lr 1.0e-5 는 자리표시자** — 사용자 결정 필요 |

하위별 gradient 비중(agent 58% 안): code 12.6 / terminal 16.0 / tool-call 8.9 / search 0.9 / 고정(usab·safety·cuda) 1.4 — sub 목표는 bin 기준이라
train% 낮은 SWE·opencode(17~27%)가 gradient 에서는 terminal 보다 작다. 이 배분을 gradient 로 다시 맞출지는 열린 결정.

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
| Agentic (4) | RL-Agentic-{Function-Calling-Pivot, Conversational-Tool-Use-Pivot, SWE-Pivot(4.8G), Indirect-Prompt-Injection}. **주의(2026-09-22)**: `SWE-Pivot-v1` 은 이름과 달리 단일 스텝 피벗이 아니라 E2E SWE-RL 과제셋(6,436 이슈)이다. 진짜 피벗 행은 `Terminal-Pivot-v1`(31k, 미프로파일)·Conversational-Tool-Use(통과율 동봉). 실사·PivotRL 적용 설계는 `study/pivotrl_study.md` §4 |
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
7. ~~effort/budget 재변환 (§2.6)~~ **완료 (2026-08-25 변환·블렌드 반영, 08-28 사전 점검 재확인)**

## 6. 미해결/후속

- SFT-OpenCode·SWE-v3·Math-Proofs-v2·CP-v1의 used_in 부재 — 전 파일 스캔으로 확정 필요.
  **`used_in` 은 생성 세대 표시다**(Agentic-v2 search 가 `super_v3` 이면서 Ultra 에 retain 됨, §2.3) — 편입 판단은
  태그가 아니라 Ultra 기술보고서의 데이터 절과 대조한다.
- 다운로드 50건의 **크기 검증**(HF API `/tree` 의 LFS size 대 로컬 size) 미실시 — Agentic-v2 tool_calling 절단 사고(2026-09-04).
- Ultra SFT 자체의 블렌드 비율은 미공개(레시피는 "SFT 체크포인트에서 시작"만 명시) —
  Megatron-Bridge SFT 레시피 공개 여부 추적
- litmus-bench 활용법(모니터링 셋) 조사
- 교사 rollout 서빙: sglang alpha 어댑터(`examples/alpha/sglang/`)의 batch 성능 실측
- effort/budget 가정 2건 (§2.6): 절단 예산 분포 U(0.1,0.9)·블렌드 1% 는 Ultra 미공개라 가정;
  NeMo-RL `effort_levels` 계수(`low_weight`/`low_ub`/`low_penalty`)도 미공개 — RL 착수 시
  medium-effort 응답 길이 실측으로 정한다
