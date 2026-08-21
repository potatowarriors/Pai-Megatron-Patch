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
| **LC pre-tokenized 4종** | `LL_preprocessed/v5/cpt_lc{,_packed_32k}/` | unpacked 24.21B tok (272G) · **32k packed 15.56B/477k bins** (59G) · **≥64k 보존 8.65B** | ⚠️ unpacked/≥64k는 유효하나 **32k packed는 THD+CP 비호환 — `--pad-doc-multiple 16` 재패킹 필요** ([`LC_REPACK_RUNBOOK.md`](LC_REPACK_RUNBOOK.md), 08-21) |
| Nemotron post-training v3 (49종) | `LL_datasets/posttraining/{SFT,RL}` | SFT 988G + RL 62G | ✅ 다운로드·검증 완료 |
| 블렌드 레시피 분석 | `posttraining/RL/nemotron_blend_recipe.json` | — | ✅ |
| chat template | `examples/alpha/tokenizer_v5/chat_template.jinja` | — | ✅ 등록·24테스트 통과 |
| **정체성 SFT (자체 제작)** | `LL_datasets/posttraining/SFT/alpha-SFT-Identity-v1/` | train 7,315 + eval 400 | ✅ 완료 (08-10, card v1.1) |
| **정체성 RL (자체 제작)** | `LL_datasets/posttraining/RL/alpha-RL-Identity-Following-v1/` | 16,510 프롬프트 | ✅ 완료 (08-10) |
| stage2 v5 unpacked 10종 | ~~`LL_preprocessed/v5/stage2/`~~ | — | ❌ **소실 확인 (08-21)** — `specialized/`만 잔존, 나머지는 4k-packed(`stage2_packed/`)뿐이라 32k 재패킹 불가. filler 확보 옵션은 [`LC_REPACK_RUNBOOK.md`](LC_REPACK_RUNBOOK.md) §3 |
| stage1 v5 unpacked 3종 (dclm/korean_web/fineweb2hq) | `LL_preprocessed/v5/stage1/` | ~466B tok | ✅ 보존 확인 (08-21) — LC filler 대안 소스 |

실행 중인 백그라운드 작업: 없음 (2026-08-04 기준).

## 1. 타임라인

**08-21 — THD+CP 재패킹 요건 확정 (최신)**
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

## 4. 남은 작업 큐 (우선순위 순)

**A. ~~LongBlocks 토크나이즈 파이프라인~~ 완료 (08-06)** — `run_cpt_lc_v5.sh`,
   결과는 `LC_DATASETS.md` §6.5. 트러블슈팅 2건 기록: pg19 OOM(배치=문서 개수 기준
   → 데이터셋별 BATCH_SIZE), pes2o fill 77%(길이 필터가 filler 제거 → filler 티어).

**B. LC-A 잔여** — ~~자연 장문 변환·토크나이즈~~ 완료(08-06, A와 통합 처리).
   남은 것 (08-21 전면 갱신, 실행 절차는 [`LC_REPACK_RUNBOOK.md`](LC_REPACK_RUNBOOK.md)):
   ① **LC 4종 32k 재패킹** (`--pad-doc-multiple 16`, THD+CP 요건 — 확정, 즉시 실행 가능)
   ② **filler 소스 결정** — ~~stage2 v5 재패킹~~ 불가(unpacked 소실). 옵션 3안 중
      블렌드 yaml 설계와 함께 결정 (korean_web+fineweb2hq는 어느 안에서도 쓰이므로 선행 가능)
   ③ **LC-A 블렌드 yaml 작성** (pad16 경로 + THD 플래그 셋은 `LC_ENTRY_GATE.md` §1.5 기준).

**C. G0 시스템 게이트** (H100 유휴 시)
   zero-shot RULER/NLL(8k~64k) — CPT 예산·직행 가능성 캘리브레이션.
   GDN CP 클러스터 러너북(CP=2/4/8×EP=8, 32k/64k/128k 메모리·스루풋).

**D. 128k 준비** (LC-A 진행과 병행)
   다문서 합성 파이프라인(그룹핑 키: EDGAR cik·peS2o 주제·repo 단위, **내부 EOD
   금지 불변량**). 한국어 장문 소싱. 128k 재패킹.

**E. SFT 준비** (LC 학습 중 병행 가능)
   messages→idxmap 변환기(스팬 마스킹, injection 방어 적용 — 기존
   `build_idxmap_sft_dataset.py` 개조). `fill_placeholders.py` 복원 실행.
   ultra_v3 세트 전수 길이 재측정 → 64k 버킷·블렌드 비율 설계.

**F. MOPD/RL 준비** (SFT 이후)
   NeMo RL/Gym 스택 포팅 vs verl/ChatLearn 검토. 교사 패널 축소안(2~3종) 확정.
   GenRM 심판 확보 방식 결정(자체 훈련 299.5k행 보유 vs 외부 vs 후순위).
