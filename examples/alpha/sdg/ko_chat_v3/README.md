# ko_chat v3 — 한국어 chat SFT 재합성 (2026-09-09~)

ko_chat v1/v2 폐기 후 재합성 트랙. 폐기 사유·서사: `docs/KNOWN_ISSUES.md` 2026-09-09
("ko_chat v1/v2 폐기 — 비-reasoning 교사의 가짜 reasoning"). 핵심 교훈: **reasoning 데이터의 교사는
반드시 reasoning 모델**. v1/v2 는 gemma-4-31B(비-reasoning)로 사고를 지시문으로 지어내 얕은 요약형
reasoning(중앙값 204자)을 학습시켰다.

## 설계 = NVIDIA Nemotron Chat-v3 레시피 + 이번 사고 교정

NVIDIA Chat-v3 카드 실측(HF):
- 프롬프트 = **실사용자 원문** (lmarena·lmsys-chat-1m·WildChat-1M). 합성 프롬프트 아님.
- 응답 = reasoning 모델(GLM-5)의 **네이티브 사고 흔적**을 `reasoning_content` 그대로 수록.
- 멀티턴 = 교사가 사용자 역할 시뮬레이션으로 확장.
- 선별 = 여러 응답 샘플링 → 쌍대 비교 보상모델(GenRM)로 최선 선택. 탈락 응답은 히스토리로 재사용.
- chat 은 마지막 턴만 학습, reasoning on/off 두 판.

| 축 | v1/v2 (폐기) | v3 (이 트랙) |
|---|---|---|
| 교사 | gemma-4-31B (비-reasoning) | **GLM-5.3-Flash** (reasoning, `--reasoning-parser glm45` 로 네이티브 사고 분리) |
| reasoning | 지시문으로 지어낸 요약 204자 | 교사 네이티브 사고 흔적 그대로 |
| 프롬프트 | 영어 lmsys/WildChat **번역** + 샘플러 | **실사용자 한국어 원문**(lmsys/WildChat Korean) + 영어의 **현지화 재작성** + 한국 맥락 샘플러 |
| 선별 | 단일 샘플 + gemma 셀프 심판(관대) | best-of-N + **gemini-3.7-flash 쌍대 심판** |
| 정체성 | 시스템 프롬프트 미주입 → 교사 자기귀속 유입 | identity 카드 시스템 주입 (alpha-banana / CJ) |
| 누출 필터 | gemma·gemini 두 단어 | **벤더 전체**(Google·Gemini·Gemma·OpenAI·Anthropic·Zhipu·GLM·Qwen…) |
| 도구 | 한국어 도구 행 0 (도구=영어 모드 상관) | **도구 비상관 슬라이스**: 선언 있으나 미호출 + 한국어 도구 호출 (When2Call 2:1) |
| reasoning 언어 | 강제 한국어(가짜) | **파일럿 A/B 로 결정**(네이티브 vs "한국어로 사고"), 한자 혼입 필터 |

## 범위 — NVIDIA Chat-v2 와 Chat-v3 를 "한국어로 각각 복제"하는 게 아니다

NVIDIA 공개 셋 두 개는 태스크 구성이 다르다:
- **Chat-v2**: 6개 교사(Kimi-K2-Thinking·GLM-4.6·Qwen3-235B-Thinking·GPT-OSS-120B 등)의 합성 대화. `reasoning_on`(2.1B)
  + `reasoning_off`(1.3B, no-think) **두 스플릿**.
- **Chat-v3**: 실사용자 프롬프트(lmarena/lmsys/wildchat) → GLM-5 응답 + GLM-5 사용자 시뮬레이션 + GenRM best-of-N.
  IF 스플릿은 GPT-OSS-120B + tulu 페르소나. 마지막 턴만 학습.

ko_chat v3 는 **한 개의 한국어 셋**으로 두 셋의 **태스크 표면 합집합**(오픈 chat + IF + reasoning-on + reasoning-off/no-think
+ 멀티턴)을 덮되, **더 나은 v3 레시피**(실프롬프트 + reasoning 교사 + GenRM)로 만든다. Chat-v2 의 가치는 이렇게 흡수한다:
no-think 스플릿 = 30% reasoning 제거판, 다교사 다양성 = 아래 "교사·교차검증". 즉 "v2 복제 + v3 복제"가 아니라
"v3 레시피로 v2+v3 표면을 한국어로 커버"다.

## 교사·교차검증 (2026-09-09 결정 대기)

- **생성**: GLM-5.3-Flash (main1·sub1 양 노드). DSV4-flash 가중치는 로컬에 없다(models 에 GLM-5.3-Flash·Qwen3-ASR 뿐).
  2번째 생성 교사 옵션은 STATUS/보고 참조 — 확정 전 기본은 GLM 단일 생성.
- **교차검증(심판)**: **gemini-3.7-flash**(리포 판정 정본, `.env` 키). GLM 이 생성하고 gemini 가 쌍대 비교·사실성/정체성/누출을
  검증하는 **생성-검증 2모델 분리**가 기본. 2번째 생성 교사가 확정되면 상호 교차검증(A 생성→B 심판, B 생성→A 심판)으로 확장.

## 관련 — 누락 데이터셋 (터미널 선례)

Ultra 에 있었으나 미공개라 우리가 없는 SFT 도메인은 터미널 외에도 있다(office work·usability·agentic safety·대규모
agentic search). 파악·우선순위는 `docs/SFT_RL_DATASETS.md` §2.8·§6 및 STATUS. ko_chat 이 회귀 교정 최우선이고, 그 뒤
가용 노드로 순차 합성.

## 파이프라인 (스테이지)

1. **시드 추출** `extract_ko_seeds.py` — lmsys/WildChat 스트리밍 language=Korean 필터 → `out/seeds_ko_*.jsonl`.
   실사용자 원문·멀티턴 보존. (GPU 불필요, CPU 백그라운드)
2. **교사 서빙** `serve/serve_glm53_main1.sh` — main1 8×H100 EP+DP8, cu129 venv, 포트 8000, glm45 사고 분리.
3. **생성** (작성 예정 `generate_v3.py`): 시드별 (a) 시스템에 identity 주입 (b) best-of-N 샘플 (c) gemini 쌍대 심판
   최선 선택 (d) 멀티턴은 교사 사용자 시뮬레이션 확장, 탈락 응답 히스토리 재사용 (e) 30% no-think 판.
4. **현지화 재작성** — 영어 원문 프롬프트를 "한국 사용자가 이렇게 물었을" 형태로 인물·지역·서비스·화폐 치환(번역 아님).
5. **도구 비상관 슬라이스** — 무관 도구 스펙 부착 후 직접 답(미호출) + Agentic-v2 한국어화(호출). 비율 2:1.
6. **품질 게이트** — 한글비율(LLM 재판정)·코드펜스·특수토큰 4중 가드·접두/후미 중복·**벤더 전체 누출 필터**·
   render_check. 산출은 `verify_sft_bins.py` 전 PASS.
7. **변환** — `build_alpha_sft_idxmap.py` (스키마 = Chat-v3 동일).

## 착수 전 게이트 (P1 파일럿, ≈300 시드)

- **처리량 실측**: GLM-5.3-Flash on main1 EP+DP8 의 chat 생성 tok/s (터미널 트랙 실측은 에이전틱 rollout 기준
  단일 스트림 64.5 / 동시 32 집계 1,251 tok/s — chat 워크로드는 재측정 필요).
- **네이티브 사고 검증**: 교사가 실제로 긴 사고를 내는가 (v1/v2 실패의 핵심 — 반드시 통과해야 함).
- **reasoning 언어 A/B**: 네이티브 vs "한국어로 사고" 품질(gemini 심판) → 정책 결정. 한자 혼입 필터 확인.
- **누출 0**: 벤더 자기귀속 0건. **정체성**: alpha-banana/CJ 로 자기소개.

## 노드

- main1 8×H100: GLM-5.3 교사 서빙 (이 트랙 착수와 함께 가동).
- sub1 8×H100: phase-2 iter602 T1·T3 벤치 종료 후 2번째 교사 인스턴스로 합류(대규모 생성 단계).

## 교사 서빙 실측 (2026-09-10)

| 교사 | 노드·구성 | 스크립트 | 처리량 (bench_throughput.py, 256 req/level) |
|---|---|---|---|
| GLM-5.3-Flash | main1, EP+DP8, cu129 venv, seqs 96 | `serve/serve_glm53_main1.sh` :8000 | 동시 64: 1,701 · 128: **4,002 tok/s** (seqs 96 포화) |
| Qwen3.8-Flash-Next-FP8 | sub1, **TP8+EP** (TP8 단독은 전문가 gate/up 80 이 8 로 안 나뉘어 실패), glm_serve_venv(cu130, `Qwen4ExpForConditionalGeneration` 등록) + jit595, vLLM 레시피 정합(seqs 256·prefix caching·moe triton·qwen3_xml) | `serve/serve_qwen_sub1.sh` :8300 | 동시 64: 4,074 · 128: 5,959 · **256: 9,178 tok/s** (오류 0, p95 24 s) |

- 두 교사 모두 한국어 프롬프트에서 **영어(혼합)로 사고, 한자 0** → reasoning 언어 규칙(영어·영한 허용, 중국어 금지) 통과. 답변은 한국어.
- MTP 투기 디코딩은 레시피가 H100 에서 8~36% 느리다고 명시 → 미사용. DSV4-Flash-0731 은 Blackwell 전용이라 H100 불가(채택 안 함).
- 스모크 함정: identity 미주입 시 Qwen 은 "저는 Qwen" 이라 답함 → 생성 시 identity 시스템 주입 필수. 긴 사고가 답변을 굶기는 행(사고 13k자·답변 0) → max_tokens 12,288 + 빈 답변 리젝.

## 본 생성기 `generate_v3.py` (2026-09-10 밤샘 런 r1)

시드 → 마지막 assistant 턴을 교사 라운드로빈(GLM/Qwen)으로 N=2 생성 → **상대 교사가 쌍대 심판** → 게이트 통과 최선 1개 채택.
게이트(샘플 단위): 빈 답변·미완결·빈 사고 / special-token 리터럴 / **사고·답변 한자 비율(>2%·>1%) 리젝** / 벤더 **자기귀속** 문맥 리젝
(블랭킷 벤더명 아님) / 사고 퇴행(4-gram 5회 반복 등). P1 파일럿(GLM 48시드): 네이티브 사고 중앙값 1,444자(v1/v2 204자), 누출 3건 중
2건은 오탐(클로드 마켈렐레·OpenAI CLIP)이라 자기귀속 스코프로 좁힘.

- **r1 런**: `out/v3_r1_lmsys.jsonl`(채택 행) · `out/v3_r1_lmsys.rejects.jsonl`(양 샘플 리젝 행) · 로그 `out/gen_v3_r1.log`.
  시드 = lmsys-chat-1m Korean 2,561행(`out/seeds_ko_pilot.jsonl`). WildChat-1M-Full 은 **gated 접근권 없음**(CLAUDE.md 의 "승인 완료" 와
  달리 HF 토큰이 거부됨) → 사용자 계정에서 접근 재요청 필요.
- **재개**: 같은 명령 재실행(`conv_id` 로 건너뜀): `python3 generate_v3.py --seeds out/seeds_ko_pilot.jsonl --out out/v3_r1_lmsys.jsonl --workers 96`
- 심판 위치 편향(A 9 : B 4, 스모크) 미보정 — 다음 라운드에 순서 무작위화 예정. 다음 시드 확장: 현지화 재작성(chat_v3 영어 원문) + 트랙 B 샘플러.

### r1 결과 (2026-09-10 새벽 완주, 4,573 s, 교사 합산 ≈1,360 tok/s; 검수 `inspect_run.py` → `out/R1_INSPECT.md`)

| 지표 | 값 |
|---|---|
| 채택 행 / 양샘플 리젝 행 | **2,327 / 234 → 채택률 90.9%** (시드 2,561) |
| 채택 행 교사 | GLM 1,190 · Qwen 1,137 |
| 사고 중앙값(p25/p75) | GLM **1,666자**(934/3,001) · Qwen **983자**(499/2,490) — v1/v2 204자 |
| 사고 한글비 / 한자비 | GLM 0.11 / **0.000** · Qwen 0.20 / **0.000** (중국어 혼입 0 = 언어 규칙 통과) |
| 답변 한글비 · 중앙값 | GLM 0.84 · 624자, Qwen 0.81 · 980자 |
| 채택 행 자기귀속 재검 | **0 건** |
| 멀티턴(user≥2) 행 | 1,040 (44.7%) — 히스토리 보존, 마지막 턴만 학습 |
| 심판 | A 834 · B 857 · 판정 없음 636(그중 2샘플 행 322 = **심판 실패 12%**: 심판 사고가 4,096 토큰 소진 → 빈 답변) |
| 샘플 리젝 사유 | reasoning_degenerate 170 · content_chinese 42 · empty_reasoning 33 · vendor_self_attribution 25 · empty_content 17 · finish_length 16 · reasoning_chinese 5 · gen_error 5 |

**r1 교훈 → r2 반영(코드 반영 완료, r1 은 커밋 a0a54a4 판으로 실행됨)**: ① 심판은 thinking 을 끄고(`enable_thinking: false`)
예산 1,024 로 → 빈 판정 해소, A/B 제시 순서 무작위화(원 순서로 복원) ② `reasoning_degenerate` 170 건 중 다수가 오탐 —
Qwen 의 "We need … Need respond." 단문 사고·GLM 목록형 사고가 "8문장·고유비 40%" 규칙에 걸림(4-gram 최다 반복 1회) →
12문장·30% 로 엄격화, 4-gram≥5 반복 규칙은 유지 ③ 리젝 파일에 사고·답변 전문 보존(게이트 재보정용).
④ Qwen 은 사고가 길어 답변이 비는 행(empty_content·finish_length)이 GLM 보다 많음 — max_tokens 12,288 유지, 필요 시 16k.

**P2 스크럽 결과(`scrub_vendor_self_attribution.py`, Chat-v2)**: reasoning_on 929,237행 중 1,823(0.20%) · reasoning_off
1,068,273행 중 829(0.08%) 드롭 → `*.p2scrub.jsonl`. 예시에 "i was drunk and google" 류 오탐이 섞임(`I was … Google` 느슨 매칭)
— 손실은 무시할 수준이지만 최종 B 블렌드 전 정규식을 조인다(자기귀속 동사·"AI/model/assistant" 명사 근접 요구).

## P0 파일럿 (2026-09-10, 사용자 승인 후 실행) — `reports/P0_PILOT_REPORT.md`

**시드 계획 확정(사용자)**: 1차 트랜치 10만 행 = S1 현지화 40k · **S2 기사 기반 40k** · S3 실사용자 한국어 3k · S4 한국 맥락 샘플러 17k.
lmsys 유래 행 포함(사내 연구 전용), 기사 원문 포함(모드 A) 허용. 프롬프트는 **thinking 켠 채** 교사가 쓰고 content 만 사용(no-think 불필요).

- 기사 코퍼스: NIKL 신문 말뭉치 2025 — 974,034 기사(2024년), 본문 중앙값 916자, 9주제·20매체. `news_prompts.py`: 주제·월 층화 →
  모드 A(문서 제공: 요약·수치 표·근거 QA·재서술·영문 요약…) / B(영감형: 제도·비교·절차·의견·작문, 2024 사건 특정성 금지 게이트).
  파일럿 198기사 → 396 프롬프트 리젝 0, 페르소나·과업 자연스러움 육안 양호.
- S1 현지화 `build_seeds_localized.py`: Chat-v3 복원 영어 프롬프트 637k(단일턴 30%, 출처 lmsys 31%/WildChat 27%/lmarena 42%,
  `metadata.seed_dataset` 로 행 단위 태그) → 한국 사용자 요청으로 재작성. GLM thinking-on 100개: 76 통과(영어 유지 요청·base64 등 정당 리젝).
- **생성 결과(교사 GLM, 철저 사고 지시, 심판 Qwen no-think → 이후 상대 교사로 교체)**:

| 시드 | 채택 | 사고 tok 중앙값 (p25/p75) | 답변 tok | 사고 한글비 | 한자 | 답변 한글비 |
|---|---|---|---|---|---|---|
| S2-A 문서 제공 | 185 | 3,036 (1,370/5,079) | 507 | 0.28 | 0 | 0.89 |
| S2-B 영감형 | 188 | 3,370 (2,376/4,940) | 786 | 0.33 | 0 | 0.98 |
| S1 현지화 | 67 | 2,354 (1,113/4,021) | 548 | 0.18 | 0 | 0.82 |
| (대조) NVIDIA Chat-v3 | — | 1,256 (970/1,606) | 878 | — | — | — |

- **교사 비교(같은 시드, DSV4-Flash-0731)**: 사고 중앙값 A 505 / B 782 / 현지화 701 토큰, **사고 한글비 0.80~0.93(한국어로 사고)**, 한자 0,
  채택 100%, GLM 저효율 심판 실패 0. GLM 은 NVIDIA 의 2.3~2.7×(영어 사고), DSV4 는 0.4~0.6×(한국어 사고) — 둘을 섞으면 분포가 NVIDIA 근처.
- 게이트 재보정: 퇴행 규칙 4-gram≥5 단독 → (4-gram≥5 ∧ 고유 문장 비율<0.7) ∨ (12문장 ∧ <0.3) ∨ 같은 줄 3연속. 구제 `rescue_rejects.py` 로
  기사 +48·현지화 +8 회수(채택률 94%/88%). 남은 리젝은 사고가 예산(12,288)을 소진해 답변이 빈 행 → 기본 max_tokens 16,384.
- **심판 = 상대 교사(저효율)** 로 확정(사용자 2026-09-10): GLM `reasoning_effort: low`(4,096) / DSV4 `thinking: false`(2,048, 실측 R=0) —
  세 번째 모델(Qwen) 제거. GLM 템플릿에는 `enable_thinking` 이 없어(`clear_thinking`·`reasoning_effort` 만) 사고를 끌 수 없다.
- **DSV4-Flash-0731 서빙(sub1)**: `serve/serve_dsv4_sub1.sh` — TP8+EP, KV fp8, FP4 indexer off. 체크포인트의 전문가는 **MXFP4**(FP8 라벨과 달리)
  → Triton mxfp4 는 SiLU 미지원으로 실패, **Marlin** 백엔드로 기동 성공(KV 903k 토큰). identity 미주입 시 "저는 DeepSeek" → 주입 필수.
  처리량 실측 ≈1,000 tok/s(동시 48) — 저부하 측정. P1 벤치는 아래 P1 절(seqs 256, 7,849 tok/s@256).

## P1 본 트랜치 (2026-09-10 착수, 사용자 승인 "착수 진행해") — 시드 10만 행 → P2 생성

**교사 배분 1:1·GLM 철저 사고 지시 그대로·DSV4 처리량 최대화(사용자 확정)**. 시드 4종은 교사별 파일로 나눠 두 서버를 동시에 채운다.

| 트랙 | 스크립트 | GLM 몫 | DSV4 몫 | 서로소 보장 |
|---|---|---|---|---|
| S1 현지화 40k | `build_seeds_localized.py` (`LOCALIZER=glm\|dsv4`, thinking-on content-only) | 20k | 20k `--offset 20000` | 후보 순서가 결정적 → 오프셋 |
| S2 기사 40k | `news_prompts.py --articles 10000 --per-article 2` | seed 11 | seed 12 | 974k 중 10k 표본 2개(겹침 ≈1%, 작가가 다름) |
| S3 실사용자 | `select_real_seeds.py` (lmsys 2,561 + WildChat 2,833 → ≥25자·한글비·한자·정체성·**히스토리 벤더 자기귀속 111행 제외**) | 2,348 (목표 3k 미달, 다른 트랙이 흡수) | | |
| S4 한국 맥락 17k | `context_prompts.py` (10 도메인×58 상황×15 페르소나×10 과업×5 어투 격자, 실존 인물·특정 사건 금지, `SPECIFIC` 게이트) | 8.5k seed 6 | 8.5k seed 5 | 격자 난수 시드 |

- S4 스모크(DSV4 40건): 38 통과(리젝 2 = 특정 시점 언급), 중앙값 338자, 숫자·조건 제약이 붙은 실제 요청 형태.
- **병합 `merge_seeds.py`**: conv_id sha1 홀짝으로 A/B 배정(결정적) → 시드가 자라도 배정 불변. A = GLM 생성·DSV4 심판, B = DSV4 생성·GLM 심판.
  프롬프트 작가와 생성 교사가 독립이라 4 조합이 모두 등장.
- **DSV4 처리량**: seqs 128 기준선 conc 64/128/256 = 3,895/4,954/4,816 tok/s → **seqs 256 재기동: conc 128/256 = 4,206/7,849 tok/s**(+58%).
  P0 의 ≈1,000 tok/s 는 동시 48 의 저부하 측정이었다. GLM(DP8, 랭크당 96) 은 시드 3종 255 동시에서 4,113 tok/s·KV 90% = 포화 →
  P2 는 시드 완료 후 순차 투입(겹치면 선점만 는다).
- 시드 생성 속도(GLM, 초반 실측): 기사 1.42건/s · 현지화 0.82건/s(채택 77%, 리젝은 low_hangul·codefence) · 맥락 —. GLM 몫 48.5k ≈ 6h 전망.
- **GLM 병목 분석(09-10 07:30)**: 시드당 소요 GLM ≈2.5 s vs DSV4 ≈0.94 s(2.6×) = 서버 처리량 1.8×(328 GB FP8·288 전문가 top-8 vs 167 GB MXFP4 전문가·256 top-6)
  × 샘플당 토큰 1.7×(GLM 사고 중앙값 6,020 tok vs DSV4 3,477). 심판(GLM low effort)은 실측 프롬프트 ≈1.2k·완성 6~400 tok 으로 싸다. 사용자 결정: 교사 1:1 유지, **서빙 재설정만 적용**.
- **GLM 재서빙(09-10 08:20, 런북 `out/p1/glm_reserve.sh`)**: `--kv-cache-dtype fp8` 은 **실패** — GLM-5.3-Flash 는 희소 MLA(FlashInfer MLA sparse SM90 백엔드)라 이 vLLM 빌드가 KV 를 uint8 로 넘겨
  `MLA kv_data_type torch.uint8 is not supported` (DSV4 는 자체 어텐션 경로라 fp8 KV 가능). 후퇴 = KV auto + **prefix caching**(같은 프롬프트의 2번째 샘플 프리필 절약). 후보: `fp8_ds_mla` 형식·FlashMLA sparse 백엔드(유휴 시 재시도).
  재기동 후 벤치(짧은 프롬프트): 128 동시 3,706 · **256 동시 7,391 tok/s** — GLM 도 256 동시에서 DSV4 급. 실부하(긴 사고·기사 프롬프트·심판)가 3.7~4.5k 에 머무는 건 KV 선점·프리필 몫.
- **생성 타임아웃**: GLM 부하 시 스트림당 10~15 tok/s → 6k+ 토큰 사고가 900 s 를 넘겨 P2-A 초반 56 샘플 타임아웃(양샘플 리젝 66행, GPU 작업 폐기) → `GEN_TIMEOUT`(기본 3,600 s). 리젝 행은 재개 시 자동 재시도.
- P2 실행: `GEN_TEACHERS=glm53-flash JUDGE=dsv4-flash python3 generate_v3.py --seeds out/p1/seeds_p1_A.jsonl --out out/p1/gen_A.jsonl --workers 256`
  / `GEN_TEACHERS=dsv4-flash JUDGE=glm53-flash … seeds_p1_B → gen_B --workers 224`(DSV4 KV 903k 토큰 한도 고려). 런처 `run_p2.sh A|B [workers]`(구제·재병합 후 기동, conv_id 재개).
- **P3 내보내기 `export_sft.py`**: Chat-v3 스키마(`metadata.train_turns` 마지막만 True — 없으면 변환기가 전 턴 학습, 실사용자 멀티턴의 원 모델 히스토리를 배우게 됨),
  no-think 30% 파생(해시 결정적, 기본 서로소 70/30; `--nothink-mode duplicate` 가능), 최종 게이트(special·EOD 리터럴·한자·자기귀속) 재검, 출처·라이선스 태그.
- **P4 경로 사전 검증(파일럿 384행, 2026-09-10)**: 128k 레시피(`--seq-length 131072 --pad-doc-multiple 16`) 변환 → thinking 282행 10 bins(실 1.20M·학습 1.14M 토큰, 행당 ≈4.2k)
  · no-think 102행 1 bin(행당 ≈830) → `verify_sft_bins` **PASS 11/11** → `render_check` clean(thinking `<think>…`, no-think `<think></think>`). 본 변환은 `convert_sft_128k_mixed_p2b.sh` 의 `run` 규약으로.

## 상태

진행 상태는 `docs/STATUS.md`. 이 README 는 설계 정본.
