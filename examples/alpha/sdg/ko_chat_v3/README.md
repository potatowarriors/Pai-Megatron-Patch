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

## 상태

진행 상태는 `docs/STATUS.md`. 이 README 는 설계 정본.
