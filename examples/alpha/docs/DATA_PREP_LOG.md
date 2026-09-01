# LC·SFT 데이터 준비 작업 기록 (2026-07-31 ~ 2026-08-10)

alpha의 **LC-phase → SFT → RL(MOPD)** 훈련 계획을 위한 데이터 준비 작업의 종합 기록.
상세 분석은 [`LC_DATASETS.md`](LC_DATASETS.md)(LC 풀·전략)와
[`SFT_RL_DATASETS.md`](SFT_RL_DATASETS.md)(post-LC 자산·파이프라인)에 있고,
이 문서는 **무엇을 했고, 지금 어디에 무엇이 있으며, 다음이 무엇인지**의 단일 진입점.

## 0. 현재 상태 스냅샷

| 자산 | 위치 (`/home/work/Datasets/`) | 규모 | 상태 |
|---|---|---|---|
| LC 원천 3종 (PG19/EDGAR/peS2o v3) | `LL_datasets/longcontext/en/` | 159G raw | ✅ 분석·변환·**pre-tokenize 완료** (08-06) |
| **LongBlocks LC doc-QA (완성본)** | `…/LongBlocks/_jsonl/` | 45파일 / 191,758샘플 / 30G | ✅ **pre-tokenize까지 완료** (08-06) |
| **LC pre-tokenized 4종** | `LL_preprocessed/v5/cpt_lc/` + **`cpt_lc_packed_32k_pad16/`** | unpacked 24.21B tok (272G) · **32k pad16 packed 15.56B/477,456 bins** · **≥64k 보존 8.65B** | ✅ **pad16 재패킹 완료 (08-21, 러너북 작업 1)** — %16 정렬 검증 전 종 0 miss. 구 `cpt_lc_packed_32k/`(pad 없음, THD+CP 비호환)는 블렌드 yaml 전환 후 삭제 가능 |
| **LC KO 2종 (자체 제작, syn_data)** | `LL_preprocessed/v5/cpt_lc_packed_32k_pad16/{ko_news,ko_grounded}/` | **ko_news 423.8M/13,141 bins (fill 98.4%, 잘림 0)** · **ko_grounded 44.3M/1,515 bins** | ✅ pad16 형식으로 인계 완료 (08-21) — NIKL 이벤트-스레드 팩 + 그라운디드 합성. 출처·게이트는 각 디렉토리 README, 128k KO 팩은 아래 행 |
| **LC 128k 팩 (LC-B용, EN 4종 + KO 2종)** | `LL_preprocessed/v5/cpt_lc_packed_128k_pad16/` | **EN 90,112 bins / real 8.65B** (longblocks 3.75B·pg19 2.40B·edgar 2.49B·pes2o 6.8M; fill 66~89% 구조적) + KO 128k 2종(ko_news 423.7M + ko_grounded 44.2M, syn_data 08-23 인계) | ✅ **완료 (08-23, main1)** — ge64k 보존분을 러너북 §4 절차로 패킹, post-verify·round-trip·전수 scan_internal_eod 통과(longblocks 2/40,876만 오염 — snap 가드 수리 범위). filler는 32k 팩 재사용(재패킹 불필요 — `lc_b_128k_blend.yaml` 헤더 근거). 블렌드 `lc_b_128k_blend.yaml` + preset `lc_b.yaml` 커밋됨 |
| **LC-A filler (P3 미러, 일반 11종 + specialized 15종)** | `LL_preprocessed/v5/lc_filler_packed_32k_pad16/` (+unpacked `lc_filler/` 보존) | **real 365.01B** (일반 42.90B + specialized 322.10B; stem_sft 81.8B·rqa 137.2B) | ✅ **완료 (08-23)** — %16 표본검사 전 종 bad=0, 블렌드 `lc_filler_32k_pad16.yaml` 26멤버 완비. 집계는 `LC_DATASETS.md` §6.5 filler 표, 실행 기록 `LC_FILLER_HANDOFF.md` §6 |
| Nemotron post-training v3 (49종) | `LL_datasets/posttraining/{SFT,RL}` | SFT 988G + RL 62G | ✅ 다운로드·검증 완료 |
| 블렌드 레시피 분석 | `posttraining/RL/nemotron_blend_recipe.json` | — | ✅ |
| chat template | `examples/alpha/tokenizer_v5/chat_template.jinja` | — | ✅ 등록·24테스트 통과 |
| **정체성 SFT (자체 제작)** | `LL_datasets/posttraining/SFT/alpha-SFT-Identity-v1/` | train 7,315 + eval 400 | ✅ 완료 (08-10, card v1.1) |
| **정체성 RL (자체 제작)** | `LL_datasets/posttraining/RL/alpha-RL-Identity-Following-v1/` | 16,510 프롬프트 | ✅ 완료 (08-10) |
| stage2 v5 unpacked 10종 | ~~`LL_preprocessed/v5/stage2/`~~ | — | ❌ **소실 확인 (08-21)** — `specialized/`만 잔존, 나머지는 4k-packed(`stage2_packed/`)뿐이라 32k 재패킹 불가. filler 확보 옵션은 [`LC_REPACK_RUNBOOK.md`](LC_REPACK_RUNBOOK.md) §3 |
| **stage2/P3 raw 소스** | `LL_datasets/pretraining/{stage2,stage3}/` | code 1.9T + math 718G + CC-Math 244G + **Specialized v1/v1.1/v1.2** (+cc_code 1.1T) | ✅ **잔존 확인 (08-21)** — 소실은 unpacked 계층뿐, raw는 CC-HQ만 삭제(MinIO 복원 ~3h). 주의: `v5/stage2/specialized/`는 빈 스캐폴딩(.bin 0개). LC filler는 **P3 블렌드(specialized 35%가 최대) 미러**를 서브셋 재토크나이즈로 재현 → 러너북 §3 옵션 B |
| stage1 v5 unpacked 3종 (dclm/korean_web/fineweb2hq) | `LL_preprocessed/v5/stage1/` | ~466B tok | ✅ 보존 확인 (08-21) — LC filler 대안 소스 |

실행 중인 백그라운드 작업: 없음 (2026-08-04 기준).

## 1. 타임라인

**08-25 — reasoning effort/budget: Ultra 레시피 재현 (최신)**
- 발견: Ultra 의 medium-effort 마커 `{reasoning effort: efficient}` 는 데이터 필드가 아니라
  **렌더 kwarg** — 공개 SFT 행엔 흔적이 없고, IF-Chat-v3 `instruction_following.jsonl`
  자체가 GPT-OSS-120B medium-effort 생성분. RL 블렌드(rlvr1/2·mopd)엔 이미 3.5% 프롬프트에
  붙어 있어(`alpha_blends` 승계) SFT 초기화 없이는 RL 이 처음 보는 토큰을 만난다.
- 결정(사용자): NVIDIA 레시피 재현. 변환기 `--medium-effort`(IF 재변환 `chat_v3_if_fanout_me`)
  + `--truncate-reasoning-budget --row-stride`(파생 셋 `budget_trunc_v1_{if,math}`, 잘린 자리
  `</think>` 비학습). 테스트 34/34·34/34. **본 변환 완료(같은 날, sub1·ko_chat vLLM 병행 ~3분)**:
  3종 게이트 PASS, if_me trainable 구본 동일(마커 비학습 실증), 절단률 49.7/50.0%. 블렌드 반영
  chat 21→20+budget 1 (ep 2.67/1.19). RL 측은 NeMo-RL 내장 `effort_levels` 로 길이 보상.
  [`SFT_RL_DATASETS.md` §2.6]

**08-21 — THD+CP 재패킹 요건 확정**
- LC는 CP≥2 학습이고 dense 마스크 격리는 32k에서 불가 → **THD/cu_seqlens 문서 격리**
  채택 (`LC_ENTRY_GATE.md` §1.5). 데이터 요건: 모든 packed 세그먼트 길이 %16
  (`bestfit_pack --pad-doc-multiple 16`, main 70220d7).
- **기존 `cpt_lc_packed_32k`는 재패킹 대상** — 별도 세션 실행용 러너북 작성:
  [`LC_REPACK_RUNBOOK.md`](LC_REPACK_RUNBOOK.md) (사전조건→명령→%16 정렬 검증→후처리).
- **발견: stage2 v5 unpacked 10종 소실** — 큐 B의 "stage2 블렌드 32k 재패킹" 전제
  붕괴. filler 확보 옵션 3안(stage1 unpacked 사용 / 재토크나이즈 / packed 복원)은
  러너북 §3, 결정은 LC-A 블렌드 yaml 설계와 함께.
- `run_cpt_lc_v5.sh`에 env 노브 추가: `PACKED_DIR`/`PAD_DOC_MULTIPLE`/`SEQ`
  (기본값 = 종전 동작).
- **✅ 재패킹 실행 완료 (08-21, syn_data 세션)**: EN 4종 + KO 2종 →
  `cpt_lc_packed_32k_pad16/`. pytest 19/19 · 내장 post-verify 통과 · %16 표본검사
  전 종 misaligned=0 · real 토큰 구본 동일(EN 15.56B). per-doc pad 오버헤드
  0.023~0.104%. 실측 표는 `LC_DATASETS.md` §6.5 추기 참조.

**07-31 — LC 분석·전략**
- LC 원천 3종 전수 스캔(문자) + tokenizer_v5 실측 환산 → `LC_DATASETS.md` 작성.
  핵심 수치: ≥32k 풀 9.0B / ≥64k 4.8B / ≥128k 1.74B(85%가 PG19).
- Nemotron 3 Ultra LC-Phase 레시피 확보(33B tokens, 46/54 블렌드, 92%@1M+8%@4K,
  RULER류 배제) + 가족 공통 NoPE·직행 CPT 근거 → §5.1로 문서화.
- 자원 제약(2×8×H100, IB 없음, CP≤8) 반영 전략: **G0 게이트 → LC-A(32k 단일 점프,
  12~16B) → LC-B(64~128k, 3~5B)**. 128k는 다문서 합성 경로로 조건부 GO.
- LongBlocks(utter-project) 원격 분석: 193,894행, doc mean 38.1k tok,
  fit@32k 56.5%, 55.6%가 Institutional-Books(문서 null).

**08-01 — 대량 확보·정정**
- 다운로드 큐 50건(LongBlocks + v3 컬렉션 49종) **1.3h 전체 성공**. 기존 보유
  Math-v2 결손(143→192G) 자동 보완.
- **언어 필터 정정**: "alpha=en+ko" 가정이 오류 — tokenized fineweb2hq bin에서
  1,500문서 디코드·언어판별로 **20개 언어 실재** 확인 → 필터를 en+ko+code+20개
  언어(`ALPHA_LANGS`)로 확정, 변환 재실행(31,909 → 83,942행).
- Institutional-Books gate 이슈 해결(계정 혼동이 원인; fine-grained 토큰
  `canReadGatedRepos` 확인 절차 기록) → 946GB 스트리밍 재구성 착수.
- **길이 배분 정책** 수립: ≥64k 문서는 32k 스테이지에서 제외, 128k용 보존
  (32k 슬롯은 <64k 풀 ~11B로 자체 충당). `split_by_doclen.py` 작성·검증.
- Ultra/Super/Nano RL 블렌드 구성 전수 집계 → `nemotron_blend_recipe.json`.
- SFT 길이 실측(파일당 3k행): fit@32k 88.2% / fit@64k 96.8% → **SFT max 64k**.
- Multilingual-v2에서 **한국어 SFT 81,646행**(ko×code/math/stem) 발견.

**08-06 — LC pre-tokenization 전체 완료**
- `run_cpt_lc_v5.sh`(신규 오케스트레이터) + EDGAR/peS2o 컨버터 2종으로 4개 데이터셋
  전 체인(convert→tokenize→split→pack 32k) 처리. **unpacked 24.21B / 32k packed
  15.56B (fill ~99.5%) / ≥64k 보존 8.65B** — 상세 `LC_DATASETS.md` §6.5.
- 트러블슈팅: ① pg19 OOM(문서 개수 기준 배치 → 데이터셋별 BATCH_SIZE 64~1000),
  ② pes2o fill 76.8%(≥16k 필터가 filler 문서까지 제거 → 4k~14k 3% 해시 샘플링
  티어 추가, 98.78% 회복), ③ 모니터 오탐(`tail -f` 기본 -n 10 → `-n 0` 필수).
- **≥64k 보존 풀 8.65B 실측** — 추정(4.84B) 대비 1.8배 (Institutional ge 3.75B 기여).

**08-04 — 재구성 완료·SFT 준비**
- Institutional 재구성 **완료**: ~7h, 107,816행/73,373권, 실패 0 →
  LongBlocks `_jsonl` 최종 191,758샘플/30G.
- `used_in` 전수 검증: **v3 컬렉션은 세대 리셋·자기완결** — Chat-v2 200만 행 중
  ultra 태그 0건, 구세대(Post-Training-Dataset-v1/v2)는 불필요 판정.
- GenRM 분석: 생성형 심판(RLVR로 훈련) — RM 훈련 데이터 299.5k행, decoder 무수정.
- **chat template 결정·구현**: Kimi-K3/Nemotron/Qwen3.5/GLM-5.2 원문 비교(4사 수렴
  확인) → Nemotron 사본 채택 + Kimi injection 방어 → `tokenizer_config.json` 등록,
  `verify_chat_template.py` 24/24 통과.

**08-07~10 — 정체성(identity) 데이터셋 자체 제작**
- NeMo DataDesigner 로 alpha 자기인식 SFT 데이터 생성. 교사 `google/gemma-4-12B-it`(vLLM).
  15,000행 **2h34m · 47,620 요청 · 실패 0** → 필터 후 train 7,315 / eval 400.
- **핵심 설계**: 사실은 `identity_card.yaml`(단일 진실 원천)에서 결정론적으로 주입하고
  교사 LLM 은 문체만 담당. 규칙 게이트가 사실 이탈·교사모델 누출을 차단.
- **오귀속 거부가 최대 비중(25.5%)** — 사전학습 코퍼스가 "나는 ChatGPT다"로 사전확률을
  밀어놓기 때문. `Nemotron-RL-Identity-Following-v1`(CC-BY-4.0) 프롬프트 뱅크를 시드로
  재사용(최종의 23%).
- **RL 은 SFT 로 대체 불가**: Nemotron SFT 25종 전수조사 결과 정체성 Q&A **0건**,
  NVIDIA 는 RL 로만 처리(Super 블렌드 1.4~2.9%, **Ultra 는 0%**). 550B 는 사전학습 표상을
  RL 로 *끌어내면* 되지만, alpha 는 `alpha-banana` 가 코퍼스에 없어 **SFT 주입이 선행**해야 한다.
  → RL 용으로 뱅크의 `principle` 만 교체한 16,510행도 함께 제작(SFT 소비분 1,756건 제외).
- **card v1.1 정정**(08-10): 개발자 1인 → 2인 팀. `creator_individual` 슬라이스 461행만
  재생성(12분)해 535행으로 교체 — 전량 재생성 불필요.
- 상세: [`sdg/identity/README.md`](../sdg/identity/README.md),
  `syn_data/docs/identity-dataset.md`(실전 교훈·함정)

## 2. 신규 도구 (전부 `toolkits/pretrain_data_preprocessing/`, 검증 스크립트만 alpha/tools)

| 도구 | 역할 | 검증 |
|---|---|---|
| `convert_longblocks_to_jsonl.py` | LongBlocks doc+Q+A → jsonl (`--languages alpha` 필터, 교사응답 제외) | 실행 완료, 카운트 전수 대조 |
| `reconstruct_longblocks_ib.py` | Institutional 원문 스트리밍 조인(946GB 미러링 없이) + OCR 우세본 선택 | 107,816/107,816행 복원 |
| `split_by_doclen.py` | unpacked를 길이 기준 lt/ge 분리(기본 T=64k) — 배분 정책 구현 | 유닛+실데이터 dry-run |
| `examples/alpha/tools/verify_chat_template.py` | 템플릿 검증 24 tests + 전처리 규약 실증 | 24/24 |
| `examples/alpha/sdg/identity/` (6종) | 정체성 SDG 파이프라인: 시드 빌더·DataDesigner DAG·SFT 변환·슬라이스 교체·RL 변환 | preview 64/64 규칙통과, chat template 렌더 검증 |

## 3. 핵심 결정 기록 (상세 근거는 각 문서)

1. **32k GO / 128k는 다문서 합성 경로** — 네이티브 ≥128k 1.74B로는 부족하나
   Nemotron 레시피(128K~1M을 합성 다문서로 채움)로 조립 가능. [`LC_DATASETS.md` §5]
2. **길이 배분**: <64k만 32k 스테이지에, ≥64k(4.84B)는 128k 보존 — 비용 없는 보험.
   [§5.1 배분 정책]
3. **언어 필터 = alpha 22개 언어** — 근거는 추정이 아니라 tokenized bin 역디코드
   실측. [`convert_longblocks_to_jsonl.py` docstring]
4. **Institutional 라이선스 주의** — Early-Access 약관(비상업·재배포 금지):
   상업 배포 국면에서 이 파트 포함 여부 재검토. [§5.1]
5. **SFT max-seq 64k** (+ >64k 3.2%는 128k 소량 버킷) — 실측 fit@64k 96.8%.
   [`SFT_RL_DATASETS.md` §2.4]
6. **v3 컬렉션 자기완결** — 세대 간 additive 아님, `used_in` 태그가 멤버십 정답.
   구세대 다운로드 불필요. [§2]
7. **chat template = Nemotron 3 사본 + Kimi 방어** — 4사 수렴(빈 think=no-think,
   히스토리 think 제거 만장일치, tool XML 포맷 Nemotron≡Qwen). 구현 규약 2건:
   멀티턴 loss mask는 **스팬 스캔**(prefix-diff 불성립 실증됨), content 인코딩은
   `split_special_tokens=True`. [`SFT_RL_DATASETS.md` §5-4]

8. **정체성은 SFT 주입 → RL 정련 순서** — 없는 표상은 강화할 수 없다. NVIDIA(550B)는
   RL 만으로 충분했으나 alpha(15B-A2B)는 SFT 가 선행해야 한다. [`syn_data/docs/identity-dataset.md` §2]
9. **정체성 데이터 SFT 블렌드 비중 0.3~1.0% 상한** — 과다 주입 시 무엇을 물어도
   자기소개하는 과적합. NVIDIA 도 21,660행만 사용. identity 단독 반복 에폭 금지.
   [보강 2026-09-01] 비중 상한은 **반복 횟수와 함께** 본다 — 0.43% 비중이 1.2M tok 셋에서는 원본 ≈180회였다
   (`KNOWN_ISSUES.md` 2026-09-01 ②). 카드 1.2 부터 "만든 사람" 질문은 개인(이동호) 선행·조직 후행 (사용자 결정,
   `SFT_PHASE2_PLAN.md` §3).

10. **LC-A filler = P3 블렌드 미러** (사용자 확정 2026-08-21) — replay의 목적은
    파괴적 망각 방지이므로 모델이 마지막으로 본 분포(P3: specialized 35 / CC-HQ 20 /
    math 18 / cc_code 10 / code 8 / korean 6 / fw2hq 3)를 **조성 변경 없이** 재현.
    가중치는 `stage2_v5_blend_packed_p3.yaml` 재사용, 경로만 pad16 트리로 치환.
    Nemotron 3 Ultra의 "LC 54% = 직전 phase 블렌드 재사용"과 동형.
    실행 절차: [`LC_REPACK_RUNBOOK.md`](LC_REPACK_RUNBOOK.md) §3.

11. **effort/budget 는 NVIDIA 레시피 그대로** (사용자 확정 2026-08-25) — RL 블렌드에 마커가
    이미 3.5% 섞여 있어 "무시"는 선택지가 아니다(제거 or 초기화). IF split 은 `--medium-effort`
    렌더, 절단-예산 파생 셋 1% 슬롯, RL 은 `effort_levels` 길이 보상. 예산 분포 U(0.1,0.9)와
    1% 는 미공개라 가정 — RL 착수 시 실측으로 조정. [`SFT_RL_DATASETS.md` §2.6]

## 4. 남은 작업 큐 (우선순위 순)

**A. ~~LongBlocks 토크나이즈 파이프라인~~ 완료 (08-06)** — `run_cpt_lc_v5.sh`,
   결과는 `LC_DATASETS.md` §6.5. 트러블슈팅 2건 기록: pg19 OOM(배치=문서 개수 기준
   → 데이터셋별 BATCH_SIZE), pes2o fill 77%(길이 필터가 filler 제거 → filler 티어).

**B. LC-A 잔여** — ~~자연 장문 변환·토크나이즈~~ 완료(08-06, A와 통합 처리).
   (08-22 갱신, 실행 절차는 [`LC_REPACK_RUNBOOK.md`](LC_REPACK_RUNBOOK.md) — 상태 블록 참조):
   ① ~~LC 4종 32k 재패킹~~ **완료 (08-22)** — `cpt_lc_packed_32k_pad16/` 산출,
      ko_news·edgar는 THD+CP 풀스택 검증에 실사용되어 %16 정렬 실전 확인
   ② ~~filler 생산~~ **완료 (08-23)** — P3 미러(결정 #10), 일반 11종 42.90B +
      specialized **15/15** 322.10B = **365.01B real** (`LC_DATASETS.md` §6.5 filler 표,
      실행 기록 `LC_FILLER_HANDOFF.md` §6). yaml 26멤버 전 경로 완비, %16 표본검사
      전 종 bad=0 + 32멤버 deep preflight (`scripts/lc_a_preflight.py`).
   ③ ~~LC-A training preset~~ **완료 → LC-A 본 런 개시 (08-23)** —
      run `outputs/alpha_baseline_48L_lc_a_20260822_191424`, P3(iter 26832)에서
      CP4·32K·THD·GBS384·LR 7.5e-6 constant, ~4분/iter(~3일). moe recompute 해제는
      본 런 iter 2 OOM으로 원복(9b1e7b1) — resume 시 재평가.

**C. G0 시스템 게이트** (H100 유휴 시)
   zero-shot RULER/NLL(8k~64k) — CPT 예산·직행 가능성 캘리브레이션.
   ~~GDN CP 클러스터 러너북(CP=2/4/8×EP=8, 32k/64k/128k 메모리·스루풋)~~
   완료 (08-22) — LC 게이트 판정 1~6 + THD+CP 스티치까지 (`LC_ENTRY_GATE.md`).

**D. 128k 준비** (LC-A 진행과 병행) — **자원 제약 해소로 데이터가 유일 병목 (08-22)**:
   Muon chunked offload로 128K@CP8 GO 전환(max-alloc 54.9~58.8GB,
   `MUON_OFFLOAD_BACKPORT.md` S5). 남은 것 = 다문서 합성 파이프라인(그룹핑 키:
   EDGAR cik·peS2o 주제·repo 단위, **내부 EOD 금지 불변량**), 한국어 장문 소싱,
   128k 재패킹(러너북 §4).

**E. SFT 준비** (LC 학습 중 병행 가능)
   messages→idxmap 변환기(스팬 마스킹, injection 방어 적용 — 기존
   `build_idxmap_sft_dataset.py` 개조). `fill_placeholders.py` 복원 실행.
   ultra_v3 세트 전수 길이 재측정 → 64k 버킷·블렌드 비율 설계.

**F. MOPD/RL 준비** (SFT 이후)
   NeMo RL/Gym 스택 포팅 vs verl/ChatLearn 검토. 교사 패널 축소안(2~3종) 확정.
   GenRM 심판 확보 방식 결정(자체 훈련 299.5k행 보유 vs 외부 vs 후순위).
