# SFT 최종 블렌드 — 데이터 구성과 비율 (`configs/data/sft_128k_final_blend.yaml`, 2026-09-13)

본 런 `outputs/alpha_baseline_48L_sft_128k_final_resume_20260914_013856`(A iter 150 승계) 가 학습 중인 블렌드의 정본 설명. 설계 근거·결정 이력은 `docs/SFT_FINAL_PLAN.md`, 생성기는 `toolkits/sft_data_preprocessing/gen_sft_128k_final_blend.py` + `sft_128k_final_spec.tsv`, bins 트리는 `/home/work/Datasets/LL_preprocessed/v5/sft_packed_128k_final_pad16` (57 멤버, 128k packed, pad 16).

## 1. 총량

| 항목 | 값 |
|---|---|
| 예산 | **60.02B bin-token** = 2,862 iters × GBS 160 × 131,072 = 457,920 samples (Nemotron 3 Ultra 1단계 SFT ≈60B 미러) |
| 학습(loss) 토큰 | ≈48.1B (80% — system·user·tool 결과·히스토리 사고는 비학습 스팬) |
| 한국어 비중 | ≈1.70B (2.8%: kochat_v3·kotool·search_ko·research_ko·ml ko 3종·usab 20%·identity 절반) |
| no-think 비중 | ≈5.62B (9.4%: chat_v2_off·kochat_v3_nothink·usab 30%·when2call·opencode·chat_if 의 무사고 64%) |
| alpha 자체 합성 | ≈2.12B (3.5%: ko_chat v3·U·T·D·identity·budget_trunc) — 나머지는 NVIDIA 공개 셋 |
| 비율 원칙 | Ultra 공개 블렌드(`recipes/ultra3/stage1_sft/config/data_prep/data_blend_raw.json`, 가중치 합 105.8)의 **카테고리 비중을 토큰 기준으로 적용**. 장문맥 1.89% 제외(LC-A/B 기학습), identity 0.5% 추가, 나머지 99.5% 재정규화. 집계는 bin 1표라 토큰 비중 = gradient 비중 |
| 카테고리 안 배분 | 스펙의 base_ep 상대 비율을 유지한 채 공통 배율 k 로 목표 토큰을 채움. 작은 셋(구조화 8ep·usab 4ep·cuda 4ep)은 에폭 고정 |
| 분할 | train 99 / valid 1 (valid 4 iters × 100 iters 마다), `mid-level-dataset-surplus 0.05` |

## 2. 카테고리 비중

| 카테고리 | Ultra % | 적용 % | 토큰(B) | 멤버 수 | 학습 토큰(B) |
|---|---|---|---|---|---|
| chat / IF / 구조화 | 27.03 | 27.41 | 16.45 | 9 | 12.00 |
| 코드 | 27.13 | 27.51 | 16.51 | 4 | 13.55 |
| 수학 | 15.88 | 16.10 | 9.67 | 3 | 9.17 |
| 과학 | 12.1 | 12.27 | 7.37 | 2 | 7.04 |
| 다국어 | 6.99 | 7.09 | 4.26 | 21 | 4.10 |
| 에이전틱(터미널·도구·검색) | 3.78 | 3.83 | 2.30 | 13 | 0.91 |
| 금융 | 2.84 | 2.88 | 1.73 | 1 | 0.08 |
| 저노력 reasoning | 1.89 | 1.92 | 1.15 | 2 | 0.88 |
| safety | 0.47 | 0.48 | 0.29 | 1 | 0.23 |
| identity(alpha) | — | 0.50 | 0.30 | 1 | 0.16 |
| **합계** | 100 (장문맥 제외 98.1) | 100 | 60.02 | 57 | 48.13 |

## 3. 멤버별 구성 (소비 토큰 내림차순)

epoch = 소비 토큰 ÷ 셋 실토큰. `_x2` 멤버는 행을 2벌 넣은 것이라 고유 데이터 기준 epoch 은 표의 2배. 학습% = 셋 안에서 loss 가 걸리는 토큰 비율.

| # | 멤버 | 카테고리 | 원천 | 언어 | 사고 | epoch | 소비(B) | 비중 % | 실토큰(B) | 학습% | bins |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `cp_v2` | code | nvidia/Nemotron-SFT-Competitive-Programming-v2 (cpp·python, 사고형) | EN | think | 0.49 | 13.065 | 21.77 | 26.751 | 97 | 204,147 |
| 2 | `science_v2` | science | nvidia/Nemotron-SFT-Science-v2 (rqa·so·syn_mcq) | EN | think | 0.44 | 6.537 | 10.89 | 14.766 | 96 | 112,818 |
| 3 | `math_v4` | math | nvidia/Nemotron-SFT-Math-v4 | EN | think | 0.97 | 6.456 | 10.76 | 6.631 | 97 | 50,627 |
| 4 | `chat_v2_on_scrub` | chat | nvidia/Nemotron-SFT-Instruction-Following-Chat-v2 reasoning_on, 자기귀속 1,823행 스크럽 + 암묵 fan-out | EN | think | 1.70 | 4.697 | 7.83 | 2.770 | 76 | 21,200 |
| 5 | `chat_v3_chat` | chat | nvidia/Nemotron-SFT-Instruction-Following-Chat-v3 chat (WildChat·lmsys 프롬프트 복원판 88.2%) | EN | think | 1.70 | 3.816 | 6.36 | 2.250 | 59 | 17,199 |
| 6 | `ifchat_v1_chat_if` | chat | nvidia/Nemotron-Instruction-Following-Chat-v1 chat_if (Ultra 최상위 가중치 셋, 암묵 fan-out) | EN | mixed(36% think) | 1.70 | 3.096 | 5.16 | 1.826 | 83 | 13,966 |
| 7 | `chat_v2_off_scrub` | chat | 〃 reasoning_off, 829행 스크럽 | EN | nothink | 1.70 | 2.325 | 3.87 | 1.371 | 77 | 10,520 |
| 8 | `swe_v3_tools_keepthink` | code | nvidia/Nemotron-SFT-SWE-v3 (도구 선언 사이드카 주입, 히스토리 사고 보존) | EN | think(66%) | 0.22 | 2.193 | 3.65 | 10.102 | 27 | 77,416 |
| 9 | `math_proofs_v2` | math | nvidia/Nemotron-Math-Proofs-v2 (자연어 증명, 최장문) | EN | think | 0.51 | 2.014 | 3.36 | 3.953 | 95 | 30,538 |
| 10 | `chat_v3_if_fanout_me` | chat | 〃 instruction_following, train_turns fan-out + `{reasoning effort: efficient}` 마커 | EN | think | 1.70 | 1.770 | 2.95 | 1.044 | 66 | 7,995 |
| 11 | `finance_v1` | finance | nvidia/Nemotron-SpecializedDomains-Finance-v1 (GenSelect 금융 QA, 학습 스팬 4.8%) | EN | think | 0.18 | 1.729 | 2.88 | 9.464 | 5 | 74,289 |
| 12 | `math_proofs_v1_lean` | math | nvidia/Nemotron-Math-Proofs-v1 lean subset | EN | think | 0.11 | 1.196 | 1.99 | 10.565 | 81 | 80,682 |
| 13 | `opencode_tools` | code | nvidia/Nemotron-SFT-OpenCode-v1 (MCP 스키마 정규화, agentic programming) | EN | nothink | 0.17 | 1.167 | 1.94 | 6.939 | 17 | 53,638 |
| 14 | `science_v1` | science | nvidia/Nemotron-Science-v1 | EN | think | 1.11 | 0.828 | 1.38 | 0.748 | 89 | 5,724 |
| 15 | `ml_ultra-v3_math_ja_translated_final` | multilingual | nvidia/Nemotron-SFT-Multilingual-v2 math ja (ultra-v3 분할) | JA | think | 1.09 | 0.612 | 1.02 | 0.559 | 99 | 4,268 |
| 16 | `budget_trunc_v1_if` | low_effort | alpha 파생 — Chat-v3 IF 학습 턴 reasoning 을 예산 B=int(L·U(0.1,0.9)) 로 절단 (저노력 reasoning 대응) | EN | think(절단) | 3.06 | 0.596 | 0.99 | 0.195 | 59 | 1,493 |
| 17 | `agentic_v2_ia` | agentic | 〃 interactive_agent | EN | think | 0.37 | 0.581 | 0.97 | 1.563 | 26 | 12,004 |
| 18 | `agentic_v2_tc` | agentic | nvidia/Nemotron-SFT-Agentic-v2 tool_calling (707k행 정본) | EN | think | 0.15 | 0.579 | 0.97 | 3.895 | 36 | 29,762 |
| 19 | `budget_trunc_v1_math` | low_effort | alpha 파생 — Math-v4 stride 20 절단 | EN | think(절단) | 3.06 | 0.554 | 0.92 | 0.181 | 95 | 1,384 |
| 20 | `ml_ultra-v3_code_pt_translated_final` | multilingual | nvidia/Nemotron-SFT-Multilingual-v2 code pt (ultra-v3 분할) | PT | think | 1.09 | 0.480 | 0.80 | 0.438 | 96 | 3,347 |
| 21 | `ml_ultra-v3_code_ko_translated_final` | multilingual | nvidia/Nemotron-SFT-Multilingual-v2 code ko (ultra-v3 분할) | KO | think | 1.09 | 0.463 | 0.77 | 0.423 | 96 | 3,233 |
| 22 | `ml_ultra-v3_code_ja_translated_final` | multilingual | nvidia/Nemotron-SFT-Multilingual-v2 code ja (ultra-v3 분할) | JA | think | 1.09 | 0.463 | 0.77 | 0.423 | 93 | 3,230 |
| 23 | `ml_ultra-v3_math_ko_translated_final` | multilingual | nvidia/Nemotron-SFT-Multilingual-v2 math ko (ultra-v3 분할) | KO | think | 1.09 | 0.456 | 0.76 | 0.417 | 99 | 3,181 |
| 24 | `ml_ultra-v3_math_pt_translated_final` | multilingual | nvidia/Nemotron-SFT-Multilingual-v2 math pt (ultra-v3 분할) | PT | think | 1.09 | 0.381 | 0.63 | 0.348 | 96 | 2,655 |
| 25 | `kochat_v3_think` | chat | alpha ko_chat v3 — GLM-5.3-Flash·DSV4-Flash 교차 심판, Chat-v3 레시피, 한국어 (sdg/ko_chat_v3) | KO | think | 1.70 | 0.379 | 0.63 | 0.223 | 94 | 1,709 |
| 26 | `ml_ultra-v3_stem_pt_translated_postedit_final` | multilingual | nvidia/Nemotron-SFT-Multilingual-v2 stem pt postedit | PT | think | 1.09 | 0.307 | 0.51 | 0.280 | 91 | 2,144 |
| 27 | `ntc_v1_math` | agentic | nvidia/Nemotron-Terminal-Corpus math (전량, 파싱실패 턴 splice) | EN | think(99.8%) | 0.15 | 0.300 | 0.50 | 2.020 | 67 | 15,431 |
| 28 | `identity_v2_fanout` | identity | alpha-SFT-Identity-v2 (카드 v1.2 제작자, train_x12, fan-out) | KO/EN | think | 18.14 | 0.300 | 0.50 | 0.017 | 53 | 133 |
| 29 | `safety_v2` | safety | nvidia/Nemotron-SFT-Safety-v2 | EN | think | 2.81 | 0.286 | 0.48 | 0.102 | 81 | 784 |
| 30 | `ntc_v1_syn_medium` | agentic | nvidia/Nemotron-Terminal-Corpus syn_medium (전량, 파싱실패 턴 splice) | EN | think(99.8%) | 0.15 | 0.230 | 0.38 | 1.543 | 54 | 11,835 |
| 31 | `ifchat_v1_structured` | chat | 〃 structured_outputs (XML/JSON 스키마 준수) | EN | think | 8.00 | 0.164 | 0.27 | 0.020 | 55 | 157 |
| 32 | `usab_v1` | chat | alpha 트랙 U — 구조화 출력·사용성 15 제약군, 프로그램 검증기 (EN 80/KO 20, think 70/nothink 30) | EN80/KO20 | mixed | 4.00 | 0.160 | 0.27 | 0.040 | 89 | 307 |
| 33 | `agentic_v2_search` | agentic | 〃 search (Wikidata 다중 홉 + web-search, held-out 300 분리) | EN | think | 1.06 | 0.144 | 0.24 | 0.135 | 18 | 1,037 |
| 34 | `ntc_v1_swe` | agentic | nvidia/Nemotron-Terminal-Corpus swe (전량, 파싱실패 턴 splice) | EN | think(99.8%) | 0.15 | 0.123 | 0.21 | 0.827 | 62 | 6,332 |
| 35 | `ml_super-v3_math_it_translated_final` | multilingual | nvidia/Nemotron-Multilingual-v1 math it (super-v3 분할) | IT | think | 0.04 | 0.111 | 0.19 | 2.538 | 99 | 19,381 |
| 36 | `ml_super-v3_math_de_translated_final` | multilingual | nvidia/Nemotron-Multilingual-v1 math de (super-v3 분할) | DE | think | 0.04 | 0.109 | 0.18 | 2.484 | 99 | 18,966 |
| 37 | `ml_super-v3_math_fr_translated_final` | multilingual | nvidia/Nemotron-Multilingual-v1 math fr (super-v3 분할) | FR | think | 0.04 | 0.097 | 0.16 | 2.215 | 99 | 16,912 |
| 38 | `cuda_v1` | code | nvidia/Nemotron-SFT-CUDA-v1 | EN | think | 4.00 | 0.089 | 0.15 | 0.022 | 56 | 170 |
| 39 | `ml_super-v3_code_zh_translated_final` | multilingual | nvidia/Nemotron-Multilingual-v1 code zh (super-v3 분할) | ZH | think | 0.04 | 0.088 | 0.15 | 2.002 | 97 | 15,287 |
| 40 | `ml_super-v3_math_es_translated_final` | multilingual | nvidia/Nemotron-Multilingual-v1 math es (super-v3 분할) | ES | think | 0.04 | 0.086 | 0.14 | 1.961 | 99 | 14,974 |
| 41 | `ml_super-v3_code_it_translated_final` | multilingual | nvidia/Nemotron-Multilingual-v1 code it (super-v3 분할) | IT | think | 0.04 | 0.082 | 0.14 | 1.879 | 97 | 14,348 |
| 42 | `ml_super-v3_math_ja_translated_final` | multilingual | nvidia/Nemotron-Multilingual-v1 math ja (super-v3 분할) | JA | think | 0.04 | 0.082 | 0.14 | 1.875 | 99 | 14,317 |
| 43 | `ntc_v1_syn_easy` | agentic | nvidia/Nemotron-Terminal-Corpus syn_easy (전량, 파싱실패 턴 splice) | EN | think(99.8%) | 0.15 | 0.080 | 0.13 | 0.538 | 55 | 4,116 |
| 44 | `ml_super-v3_code_fr_translated_final` | multilingual | nvidia/Nemotron-Multilingual-v1 code fr (super-v3 분할) | FR | think | 0.04 | 0.079 | 0.13 | 1.801 | 97 | 13,749 |
| 45 | `ml_super-v3_code_de_translated_final` | multilingual | nvidia/Nemotron-Multilingual-v1 code de (super-v3 분할) | DE | think | 0.04 | 0.077 | 0.13 | 1.762 | 97 | 13,453 |
| 46 | `ml_super-v3_code_es_translated_final` | multilingual | nvidia/Nemotron-Multilingual-v1 code es (super-v3 분할) | ES | think | 0.04 | 0.076 | 0.13 | 1.736 | 97 | 13,254 |
| 47 | `ml_super-v3_math_zh_translated_final` | multilingual | nvidia/Nemotron-Multilingual-v1 math zh (super-v3 분할) | ZH | think | 0.04 | 0.074 | 0.12 | 1.685 | 99 | 12,867 |
| 48 | `search_ko_v1` | agentic | alpha 트랙 D1 — Wikidata 한국어 연쇄 다중 홉 검색 (로컬 BM25 kowiki+NIKL, Tavily 형) | KO | think | 0.53 | 0.073 | 0.12 | 0.137 | 15 | 1,049 |
| 49 | `ml_super-v3_code_ja_translated_final` | multilingual | nvidia/Nemotron-Multilingual-v1 code ja (super-v3 분할) | JA | think | 0.04 | 0.072 | 0.12 | 1.648 | 97 | 12,584 |
| 50 | `ntc_v1_code` | agentic | nvidia/Nemotron-Terminal-Corpus code (전량, 파싱실패 턴 splice) | EN | think(99.8%) | 0.15 | 0.072 | 0.12 | 0.485 | 68 | 3,704 |
| 51 | `when2call_v1_x2` | agentic | nvidia/When2Call train_sft — 전부 미호출(되묻기·불가), BFCL→JSON Schema, 행 2벌 | EN | nothink | 3.19 | 0.054 | 0.09 | 0.017 | 5 | 131 |
| 52 | `kochat_v3_nothink` | chat | 〃 no-think 파생 30% | KO | nothink | 1.70 | 0.047 | 0.08 | 0.028 | 81 | 214 |
| 53 | `ml_ultra-v3_stem_ko_translated_postedit_final` | multilingual | nvidia/Nemotron-SFT-Multilingual-v2 stem ko postedit | KO | think | 1.09 | 0.044 | 0.07 | 0.040 | 87 | 309 |
| 54 | `research_ko_v1` | agentic | alpha 트랙 D2 — 한국어 딥 리서치 보고서 (검색 ≥5, 인용 심판 ≥0.8) | KO | think | 0.53 | 0.033 | 0.05 | 0.062 | 14 | 521 |
| 55 | `kotool_v1_x2` | agentic | alpha 트랙 T — 한국어 도구 호출/미호출 1:2 (Agentic-v2 스키마), 행 2벌 | KO | think | 1.06 | 0.022 | 0.04 | 0.021 | 39 | 161 |
| 56 | `ml_ultra-v3_stem_ja_translated_postedit_final` | multilingual | nvidia/Nemotron-SFT-Multilingual-v2 stem ja postedit | JA | think | 1.09 | 0.017 | 0.03 | 0.015 | 86 | 118 |
| 57 | `ntc_v1_syn_mixed` | agentic | nvidia/Nemotron-Terminal-Corpus syn_mixed (전량, 파싱실패 턴 splice) | EN | think(99.8%) | 0.15 | 0.010 | 0.02 | 0.068 | 53 | 523 |

## 4. 제외한 것과 이유

| 제외 | 이유 |
|---|---|
| kochat v1/v2 (trackA/B, gemma-4-31B 교사) | 비-reasoning 교사의 가짜 reasoning — 2026-09-09 폐기, ko_chat v3 로 대체 |
| ARC-AGI-v1 | Ultra 블렌드에 대응 범주 없음 |
| SWE-v1 r2e · SWE-v2 openhands/agentless · swe_v3_keepthink(선언 없는 tool_call) · swe_v3_terminus_keephist | SWE-v3 도구 선언 주입판(`swe_v3_tools_keepthink`) 하나로 통일 |
| opencode_v1 · opencode_fixed | MCP 스키마 정규화판 `opencode_tools` 로 대체 |
| chat_v2_on · chat_v2_on_fanout · chat_v2_off | 자기귀속("trained by Google") 스크럽판으로 대체 |
| identity_v1 · identity_v2(비 fan-out) | 카드 v1.2 fan-out 판으로 대체 |
| terminal_terminus2_synth (자체 합성 터미널) | 공개 Nemotron-Terminal-Corpus 채택으로 폐기(2026-09-10) |
| Competitive-Programming-v1 · Math-v2 · Chat-v1 · Agentic-v1 · Safety-v1 · Multilingual-v2 hi | 상위 판이 미소진이거나 중복·미지원 언어 |
| 장문맥(Ultra 1.89%) | LC-A(32K)·LC-B(128K) 단계에서 이미 학습 |

## 5. 생성·검증 절차

1. 멤버 bins: `build_alpha_sft_idxmap.py --seq-length 131072 --pad-doc-multiple 16` (train_turns 마지막-턴 규약, IF 는 `--fanout-train-turns --medium-effort`, Chat-v2 on·IF-Chat-v1 chat_if 는 `--fanout-implicit-turns`, SWE-v3 는 `--tools-sidecar`). 신규 5종은 `convert_sft_128k_final_new.sh`, bins<100 해소는 `convert_sft_128k_final_small.sh`.
2. 게이트: `verify_sft_bins.py` PASS(EOD 오염 0·%16 정렬·리터럴 special 0), `render_check.py` 봉투 흔적 0·`<tools>` 선언 결함 0.
3. 블렌드: `gen_sft_128k_final_blend.py --tree … --spec sft_128k_final_spec.tsv --budget-tokens 60e9` → 가중치 = 소비 토큰 비율, iters = ceil(예산 / (GBS×seq)).
4. bins<100 멤버 금지(valid 0-doc 무한대기): usab think+nothink 병합, kotool·when2call 행 2벌.
