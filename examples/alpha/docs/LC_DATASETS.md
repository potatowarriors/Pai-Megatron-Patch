# LC(long-context) 확보 데이터셋 전수 분석 (2026-07-31)

`/home/work/Datasets/LL_datasets/longcontext/en/` 아래 확보된 LC 후보 3종(PG19,
EDGAR-corpus, peS2o v3)의 **전수 스캔** 결과. 문서 수·문자 수는 raw 전체를 실측했고,
토큰 수는 tokenizer_v5로 데이터셋별 chars/token 비율을 실측해 환산한 추정치(±2~3%).
32k/128k context-extension 스테이지의 데이터 계획 판단 근거로 쓰는 문서.

## TL;DR

- 3종 합계 **~83B tokens**이지만, LC 학습에 실질 기여하는 "긴 문서" 풀은 소수:
  **≥32k 9.0B / ≥64k 4.8B / ≥128k 1.74B tokens**.
- **32k 스테이지: 현 풀로 가능**. long-fraction 10B 수준 예산이면 ≥32k 풀 1 epoch
  근처로 충당, CPT 관례상 long 데이터 2~4 epoch 반복도 허용되므로 여유 있음.
- **128k 스테이지: 네이티브 풀만으로는 부족 + 편중** — ≥128k 1.74B 중 85%가 PG19(1919년
  이전 영문 고서). 단, Nemotron 3 레시피가 보여준 **다문서 합성**(중단신 문서 조립 +
  cross-document 과제)으로 조립 경로가 열려 있어, 소싱이 아니라 합성 파이프라인이
  선행 과제 (§5.1).
- **도메인 갭이 최대 리스크**: 책·SEC 공시·논문뿐 — **코드/웹/한국어 long 문서 전무**.
  base 모델 분포(dclm + korean_web + code)와 괴리.
- peS2o는 LC 드라이버가 아님: s2orc 본문도 mean 5.8k tok(4~16k 대역에 토큰의 77%).
  16k~32k 대역 꼬리(3.0B)와 32k 패킹 filler로서 가치. s2ag(초록 8.8B tok, max 2k)는 LC 무관.
- 전처리는 아직 미착수 (`LL_preprocessed/v5/cpt_lc/` 빈 디렉토리).

## 1. 인벤토리

| 데이터셋 | 경로 (`…/longcontext/en/`) | 포맷 | 디스크 | 내용 | 라이선스 |
|---|---|---|---|---|---|
| PG19 | `pg19/data/*.parquet` (25) | HF parquet | 6.7G | Project Gutenberg 1919년 이전 출간 서적 (DeepMind pg19의 parquet 미러) | Apache 2.0 (원본 기준) |
| EDGAR-corpus | `edgar-corpus/<1993..2020>/{train,validate,test}.jsonl` (84) | jsonl (섹션 분리) | 39G | 미국 SEC 10-K 연차보고서, 연도별 | Apache 2.0 (HF `eloukas/edgar-corpus`) |
| peS2o v3 | `peS2o/data/v3/*.zst` (train 136 + valid 60) | zstd jsonl | 113G | 학술 논문 (S2ORC 파생). 레코드 `version: v3-fos-license` | ODC-BY |

주의: peS2o 로컬 README는 V2까지만 기술("V2 (Latest)", 42B whitespace tokens)하지만
**실제 데이터는 v3**다. 아래 수치는 README가 아니라 v3 실측이다.

## 2. 측정 방법

1. **문자 수 전수 스캔** (샘플링 아님):
   - PG19: parquet `text` 컬럼 utf8_length (pyarrow).
   - EDGAR: 라인별 `section_*` 필드 문자 수 합 + 섹션 사이 `\n\n` 2자 가산
     (사용 시점의 섹션 결합 방식을 그대로 반영).
   - peS2o: zstd 스트리밍 해제 후 라인별 `len(text)`, `source`(s2orc/s2ag)별 분리 집계.
     8-proc 전수 스캔으로 196샤드 ~3.5분.
2. **토큰 환산**: tokenizer_v5(`examples/alpha/tokenizer_v5/tokenizer.json`)로
   데이터셋별 무작위 24~40문서(문서당 최대 200k chars)를 실제 인코딩해 비율 산출:

   | | pg19 | edgar | peS2o s2orc | peS2o s2ag |
   |---|---|---|---|---|
   | chars/token | 4.156 | 5.090 | 4.844 | 5.098 |

   문서별 토큰 길이 = 문자 길이 / 비율. 분포·버킷은 이 추정 토큰 길이 기준.

## 3. 데이터셋별 상세

### 3.1 PG19 — 유일한 실질 ≥128k 소스

train 28,602권 / **2.75B tokens** (별도 validation 50권, test 100권).

| mean | p50 | p90 | p99 | max |
|---|---|---|---|---|
| 96.1k | 74.4k | 198k | 433k | 6.72M |

토큰 질량 분포 (문서 길이 버킷별, 토큰 가중):

| <4k | 4k–16k | 16k–32k | 32k–64k | 64k–128k | ≥128k |
|---|---|---|---|---|---|
| 0.0% | 1.1% | 3.2% | 10.3% | 32.3% | **53.1%** |

- 토큰의 95.7%가 ≥32k 문서, 53.1%(1.46B)가 ≥128k 문서에 있음. **128k 학습의 사실상
  유일한 네이티브 소스** (6,555권).
- 리스크: 1919년 이전 영문 문학 — 어휘·문체가 낡고 도메인이 단일함. 128k 스테이지를
  이것만으로 채우면 스타일 편향 우려.

### 3.2 EDGAR-corpus — ≥32k 최대 공급원

전 연도·전 스플릿 합계 220,375건 / **7.89B tokens**. (train/validate/test를 모두
합산 — CPT 용도로는 전부 사용 가능하나 스플릿 존재를 기록해 둠.)

| mean | p50 | p90 | p99 | max |
|---|---|---|---|---|
| 35.8k | 30.8k | 70.9k | 122.5k | 841k |

| <4k | 4k–16k | 16k–32k | 32k–64k | 64k–128k | ≥128k |
|---|---|---|---|---|---|
| 0.4% | 5.2% | 18.1% | **44.8%** | 28.0% | 3.5% |

- ≥32k 토큰 6.02B로 **32k 스테이지의 최대 공급원**. 반면 ≥128k는 0.28B에 불과.
- 연도별 추이: 문서 수는 1996년 이후 연 7~10k로 안정, **평균 길이는 1993년 20.8k →
  2020년 55.1k로 단조 증가** (공시 규제 강화로 문서가 길어짐). 최신 연도일수록 LC 가치↑.
- 사용 전 변환 필요: 레코드가 `section_1`…`section_15` 필드로 쪼개져 있어 **섹션 결합
  컨버터**(jsonl → `text` 단일 필드)가 필요하다. 섹션 순서는 필드 순서 그대로.
- 리스크: 10-K는 연도 간·기업 간 보일러플레이트가 많음(같은 기업이 매년 유사 문서를
  재제출). LC 능력 학습에는 오히려 장거리 참조 구조(재무제표 ↔ 주석)가 유용하다는
  견해도 있으나, 반복 학습 시 암기 경향을 모니터링할 것.

### 3.3 peS2o v3 — LC 드라이버가 아니라 중장문 품질 코퍼스

train: s2orc(본문) 11.0M건 / **63.5B tokens**, s2ag(제목+초록) 33.9M건 / **8.85B
tokens**. valid: 345k건 / 0.53B. 샤드는 소스 순 정렬(초반 s2ag → 후반 s2orc).

s2orc (본문 전체):

| mean | p50 | p90 | p99 | max |
|---|---|---|---|---|
| 5.8k | 5.0k | 10.1k | 17.8k | 193k |

| <4k | 4k–16k | 16k–32k | 32k–64k | 64k–128k | ≥128k |
|---|---|---|---|---|---|
| 17.5% | **77.2%** | 4.7% | 0.6% | 0.0% | 0.0% |

- ≥32k는 0.36B, ≥64k는 48건(0.004B), ≥128k는 **1건**. 논문 본문은 LC 소재로는 짧다
  (자연적 분포 + 업스트림 필터 가능성). **32k/128k 능력을 가르치는 데이터가 아님.**
- 대신 4~16k 대역 49B tokens는 (a) LC 스테이지 블렌드의 품질 유지용 중장문 filler,
  (b) 16k~32k 대역 3.0B는 32k 패킹의 보조 소재로 유효.
- s2ag는 max 2k의 초록 코퍼스 — LC 문맥에서는 무관. stage2류 품질 블렌드 소재로나 고려.

## 4. 통합 native-long 풀 (train 기준)

문서 길이 임계값 이상 문서가 담고 있는 토큰 질량:

| 임계 | 문서 수 | tokens | pg19 | edgar | peS2o s2orc |
|---|---|---|---|---|---|
| ≥16k | 345,572 | **13.54B** | 2.72B | 7.45B | 3.37B |
| ≥32k | 134,258 | **9.01B** | 2.63B | 6.02B | 0.36B |
| ≥64k | 44,144 | **4.84B** | 2.35B | 2.49B | 0.004B |
| ≥128k | 8,169 | **1.74B** | 1.46B | 0.28B | 0 |

## 5. 스테이지 계획 함의

**전제**: alpha는 `--reset-attention-mask`(+ BFP 패킹)로 학습하므로 **유효 문맥 학습
길이는 bin 크기가 아니라 문서 길이**다. 짧은 문서로 128k bin을 채워도 long-range
신호는 생기지 않는다. 패킹은 효율 장치일 뿐 context 확장 장치가 아님.

- **32k 스테이지 — GO.** 예시 예산 20B tokens·long-fraction 50%(≈ProLong류 레시피)
  기준 long 수요 10B ↔ 보유 ≥32k 9.0B + 16k–32k 대역 4.5B. 반복 없이도 근접, long
  데이터 2~4 epoch 반복 관례까지 감안하면 여유. 블렌드의 나머지 절반은 stage2 단문
  믹스를 32k로 재패킹해 채우면 됨.
- **128k 스테이지 — 네이티브 문서만으로는 NO-GO, 다문서 합성 경로로 조건부 GO.**
  네이티브 ≥128k 1.74B(8.2k건, PG19 85%)로는 수 epoch을 돌려도 절대량·다양성 모두
  부족하고 64k–128k 대역(3.1B)을 포함해도 얇음 — 은 여전히 사실. 단 Nemotron 3
  레시피(§5.1)가 보여주듯 128k+ 대역은 어차피 어느 코퍼스에도 자연 문서가 희소해서
  **합성 다문서 조립**으로 채우는 것이 업계 관행이며, 이 경로에서는 우리의 4k~32k
  중단신 풀(peS2o s2orc 63.5B, EDGAR 기업·연도 시리즈, stage2 웹 코퍼스)이 전부
  건축 자재가 된다. 병목이 "소싱"에서 "합성 파이프라인 구축"으로 바뀜.
- **도메인 갭 해소가 소싱 1순위**: 현재 풀은 영문 산문(고서·공시·논문)뿐.
  - 코드: repo 단위 concat(이미 확보한 Nemotron-Pretraining-Code 계열을 repo-level로
    재구성하면 신규 다운로드 없이 가능성 있음 — 파일 단위로 저장돼 있는지 확인 필요).
  - 웹: Nemotron-CC / dclm의 long-tail(≥32k) 추출.
  - 한국어: long 문서 소스 자체가 전무 — 국내 서적/법령·판례/장문 커뮤니티 스레드 등
    별도 확보 필요. base가 한국어 비중이 큰 만큼 LC에서 한국어가 빠지면 한국어 장문
    능력이 확장되지 않음.
  - synthetic long(다문서 조립)은 §5.1의 레시피로 격상 — 보강이 아니라 128k의 주력.
- **128k 학습 시스템 전제**: seq 32k 초과는 CP 필요(GDN CP 포팅:
  [`gdn_cp_port.md`](gdn_cp_port.md) — 유닛 검증 완료, H100 클러스터 검증 대기).
  attention RoPE는 θ=10M(envelope 256k) + partial rotary 0.25 + GDN implicit position
  조합이라 theta 재조정 없는 단일 점프 CPT가 1차 실험 후보 (§5.1의 NoPE 근거 참조).

### 5.1 레퍼런스 레시피: Nemotron 3 Ultra LC-Phase (2026-07-31 추가)

테크리포트(arXiv 2606.15007) + 공개 학습 레시피(GitHub)에서 확인된 사실:

| 항목 | 값 |
|---|---|
| 시점/규모 | pretraining(20T = 15T+5T) 종료 직후 CPT, **33B tokens**(전체의 ~0.17%) |
| 블렌드 | **~46% LC 데이터 + ~54% Phase 2 데이터**(`data_blend_raw_phase2.json` 재사용) |
| LC 데이터 구성 | ① 장문 문서 + doc-QA 결합 샘플 ② **합성 다문서 SFT-style 샘플(128K~1M): multi-document reasoning / retrieval / synthesis** |
| 길이 스케줄 | iteration의 **92%는 1M, 8%는 4K**(math/code SFT류만 — short 벤치 유지). iteration 내 길이 혼합 없음, iter당 토큰 상수(25,165,824) |
| 제외 | **RULER/needle-style 합성 데이터 불사용** |
| LR / 병렬화 | constant 2.5e-6; CP=32/TP=8/EP=128/PP=2 (GB200) |
| 결과 (RULER) | 64k 95.3 / 128k 92.5 / 256k 86.2 / 512k 84.5 / 1M 76.8 |

같은 가족 근거(arXiv 2512.20856 §2.5): Nemotron 3는 attention에 RoPE가 없는(NoPE)
hybrid — "Mamba가 implicit position 제공"이라 **8k→512k 직행, staged ladder 불필요**를
실증했고, dense hybrid 대비 MoE hybrid가 길이 외삽에서 graceful degradation을 보임
(RULER@1M 54.19 vs 23.43). Alpha는 attention 6/24 믹서 + partial rotary 0.25 +
θ=10M으로 근사-NoPE 조건이라 같은 성질을 기대할 근거가 있다(실측 검증 전제).

**Alpha 이식안 (블렌드 구체안):**

| 구성 | 32k 스테이지 | 128k 스테이지 |
|---|---|---|
| ~54% 일반 | stage2 v5 블렌드 32k 재패킹 | 동일, 128k 재패킹 |
| ~46% LC | 자연 장문 **<64k만** (EDGAR <64k ~5.4B + peS2o 16–32k ~3.0B + PG19 <64k ~0.4B) + LongBlocks doc-QA | **보존한 ≥64k 네이티브 4.84B**(PG19 2.35B + EDGAR 2.49B + fineweb2hq 꼬리 0.12B) + LongBlocks + **자체 다문서 합성(64k~128k)** ← 주력 |
| ~8% short replay | stage3 specialized math/code SFT류. 1차: 같은 32k bin에 혼합(attention은 4k 학습과 등가, GDN 상태 프리픽스만 상이) → short 벤치 열화 시 4k 전용 블록 핑퐁(full-state resume + blend switch 인프라로 코드 변경 0) | 동일 |
| 제외 | needle/RULER류 안 만듦 | 동일 |

**길이 기반 배분 정책 (2026-08-01 추가)**: ≥64k 문서는 유일하게 >32k 의존성을
가르칠 수 있는 희소 자원(전 풀 4.84B)이므로 **32k 스테이지 블렌드에서 제외하고
128k 스테이지용으로 신선하게 보존**한다. 32k LC 슬롯(~10B)은 <64k 풀(~11B)로 자체
충당됨을 실측 확인. 낭비 우려에 대한 답: 패킹은 unpacked 원본의 "뷰"라서 32k 조각화가
데이터를 파괴하지 않지만(KEEP_UNPACKED), ≥64k 문서의 고유 가치(>32k 의존성)는 128k
스테이지에서만 실현되므로 노출 epoch을 아껴 두는 쪽이 비용 없는 보험이다.
구현: 패킹 전 `split_by_doclen.py --threshold 65536`으로 unpacked를 lt/ge 분리 →
lt만 `bestfit_pack --seq-length 32768`. (fineweb2hq조차 ≥64k 1,137 docs / 0.119B
보유 — 전 소스에 일괄 적용.)

**LongBlocks** (HF `utter-project/LongBlocks`, CC BY-SA 4.0, UTTER 프로젝트): 193,894행
doc + 합성 Q + grounded 정답(+교사응답 3열 — CPT에선 제외, 후속 SFT/distill용).
실측(11% 샘플, tokenizer_v5): 문서 mean 38.1k tok, doc+QA **fit@32k 56.5% /
fit@128k 98.3%**, 비null 문서 총량 ~3.3B tok. 주의: **55.6%(107,817행)는
Institutional-Books 소스로 document=null** — barcode id 매칭으로 재구성
(`reconstruct_longblocks_ib.py`, **2026-08-04 완료**: 946GB 스트리밍 조인 ~7h,
107,816행/73,373권 복원, 누락 0). 최종 `_jsonl/` = **45파일 / 191,758 샘플 / 30G**
(변환분 83,942 + 재구성분 107,816). 한국어 142행(0.07%)뿐이라 한국어 LC는 별도 과제. **라이선스 주의**: Institutional 재구성분은 원 코퍼스
`institutional/institutional-books-1.0`의 Early-Access 약관(**비상업·재배포 금지·
출처표기**)이 적용되어 다른 소스와 성격이 다름 — 상업 배포 국면에서는 이 파트의
포함 여부를 재검토할 것.

**합성 다문서 샘플의 EOD 불변량 (중요)**: 구성 문서들 사이에 EOD(id 0)를 넣지 말 것.
`--reset-attention-mask`가 EOD 경계로 attention을 격리하므로, 내부 EOD가 있으면
"문서 간 추론"을 가르치려는 샘플이 attention 수준에서 조각난다. 샘플 전체(문서들 +
과제 + 답)를 하나의 논리적 문서로 만들어 **끝에만 EOD** — jsonl 생성 단계의 스펙으로
명시할 것(문서 구분은 제목/헤더 등 텍스트 구분자로).
**⚠️ 위반 실사고 (2026-08-23)**: 32k LongBlocks 팩에서 ~0.05% bins에 내부 EOD가
발견되어 LC-A 학습이 iter 170에서 정지했다(원문 속 리터럴 `<|endoftext|>`를
토크나이저가 id 0으로 매칭). 텍스트 스펙만으로는 부족하다 — 생성 파이프라인에
① 원문 special-token 리터럴 이스케이프, ② packed 산출 후
`toolkits/pretrain_data_preprocessing/scan_internal_eod.py` 게이트를 필수로 넣을 것.
전말·런타임 2차 방어(snap_cu_seqlens_to_grid)는 alpha CLAUDE.md Known Issues 참조.
관련 문서 그룹핑이 품질의 절반:
EDGAR `cik`(같은 기업 연도 시리즈), peS2o field-of-study, repo 단위 코드가 그대로
그룹핑 키가 된다. 무작위 묶음은 retrieval만, 연결된 묶음은 aggregation/synthesis까지
가르친다.

## 6. 전처리 파이프라인 연계 (착수 시 체크리스트)

1. **EDGAR 섹션 결합 컨버터** 신규 필요: `section_*` → `\n\n` join → `{"text": ...}`.
   peS2o는 zst 해제 + `text` 추출만으로 기존 `preprocess_stage2_v5.sh` 경로에 태울 수
   있음. PG19는 parquet → jsonl(`text`) 변환(스키마상 `convert_parquet_to_jsonl.py`
   재사용 가능, 컬럼명만 확인).
2. **문서 단위 unpacked mmap을 canonical로 보존** — `run_stage3_v5.sh` 기본값
   (`KEEP_UNPACKED=0`)은 pack 후 unpacked를 삭제한다. LC는 seq-length별 재패킹이
   전제이므로 **`KEEP_UNPACKED=1` 필수**. packed 산출물은 L에 비가역(패딩 EOD와 문서
   EOD 구분 불가, 장문 조각 분산)이라 packed에서 재패킹 불가.
3. **패킹은 스테이지 seq-length별로**: `bestfit_pack.py --seq-length 32768`(→ 이후
   131072) — 학습 `--seq-length`와 정확히 일치해야 함. 4096-packed 데이터를 32k
   학습에 넣으면 문서가 이미 4k에서 잘려 있어 LC 신호가 없다.
   **패킹 전 길이 분할**: `split_by_doclen.py --threshold 65536`으로 unpacked를
   lt/ge로 나눠 32k 스테이지는 lt만 패킹, ge는 128k 스테이지 보존 (§5.1 배분 정책).
4. 산출 경로 제안: `LL_preprocessed/v5/cpt_lc/<dataset>/`(unpacked) +
   `v5/cpt_lc_packed_32k/<dataset>/`. (현 `v5/cpt_lc/`는 빈 디렉토리.)
5. 토크나이즈 후 `.idx`의 `sequence_lengths`로 본 문서의 추정치(±2~3%)를 정확값으로
   대체할 것 (`count_tokens.py` 참조).
6. **다문서 합성 샘플 생성기** (128k 준비물, §5.1): 관련 문서 그룹핑(EDGAR `cik`,
   peS2o 주제, repo 단위) → cross-document 과제 합성 → **내부 EOD 없이** 단일 논리
   문서로 emit(끝에만 EOD — §5.1 불변량). LongBlocks 사용 시 doc+Q+A 연결 컨버터와
   Institutional-Books 재구성기(barcode id 매칭)도 이 단계 산출물.

## 6.5 전처리 실측 결과 (2026-08-06 — §6 체크리스트 1~3 완료)

> **⚠️ 2026-08-21 추기**: 아래 32k packed 산출물은 **THD+CP 문서 격리와 비호환** —
> LC는 CP≥2 학습이라 packed 세그먼트 길이 %16(`--pad-doc-multiple 16`) 요건이
> 추가됐다 (`LC_ENTRY_GATE.md` §1.5). unpacked/≥64k 보존분은 그대로 유효하며,
> **재패킹 절차는 [`LC_REPACK_RUNBOOK.md`](LC_REPACK_RUNBOOK.md)** (별도 세션 실행용).
> 같은 문서 §3에 filler(§5.1의 "stage2 v5 재패킹") 전제 붕괴 — stage2 unpacked
> 소실 — 와 대안 옵션도 기록되어 있다.

> **✅ 재패킹 완료 (2026-08-21, 러너북 작업 1 + KO 2종)** — 산출:
> `LL_preprocessed/v5/cpt_lc_packed_32k_pad16/<ds>/`. pytest 19/19, 도구 내장
> post-verify 전 종 통과, 러너북 §2.3 %16 EOD-run 경계 표본검사(종당 500 bins)
> **전 종 misaligned=0**. real 토큰은 구본과 동일 (EN 4종 합 15.56B). **KO 2종이
> 같은 트리에 추가됨** (`ko_news` 423.8M — NIKL 이벤트-스레드 팩, bin=팩 1개·잘림 0;
> `ko_grounded` 44.3M — NIKL 그라운디드 합성; 출처·게이트는 각 디렉토리 README).
>
> | ds | bins | fill | per-doc pad(%16) |
> |---|---|---|---|
> | longblocks | 130,461 | 99.70% | +0.027% |
> | pg19 | 12,172 | 99.69% | +0.024% |
> | edgar | 168,789 | 99.96% | +0.026% |
> | pes2o | 166,034 | 98.72% | +0.053% |
> | **ko_news** | 13,141 | 98.42% | +0.023% |
> | **ko_grounded** | 1,515 | 89.32% | +0.104% |
>
> 구 `cpt_lc_packed_32k/`는 LC-A 블렌드 yaml이 pad16 경로로 확정된 뒤 삭제 가능.

> **✅ filler 재생산 완료 (2026-08-23, 러너북 §3 — P3 미러, 결정 #10)** — 산출:
> `LL_preprocessed/v5/lc_filler_packed_32k_pad16/` (일반 11종 + specialized 15종).
> 블렌드 `configs/data/lc_filler_32k_pad16.yaml` 26멤버 전 경로 완비, %16 EOD-run
> 경계 표본검사 전 종 bad=0 + 32멤버 deep preflight 통과
> (`scripts/lc_a_preflight.py --deep specialized`). 실행 기록: `LC_FILLER_HANDOFF.md` §6.
>
> | 그룹 | real tokens | 주요 구성 |
> |---|---|---|
> | 일반 filler 11종 | 42.90B | korean_web 16.96 · fw2hq 5.72 · math 4.53 · code 5종 7.04 · cc_code 4.94 · cchq 2종 3.71 |
> | specialized v1 (6종) | 272.56B | rqa 137.20 · stem_sft 81.81 · math_textbooks 25.73 · infinibyte_reasoning 19.36 · wiki_rewrite 7.25 · scientific_coding 1.21 |
> | specialized v1.1 (5종) | 9.17B | code_concepts 7.22 · multiple_choice 1.56 · 외 3종 0.40 |
> | specialized v1.2 (4종) | 40.37B | fact_seeking 32.79 · multiple_choice 6.89 · generative 0.67 · moral_scenarios 0.015 |
> | **합계** | **365.01B** | specialized 15종 322.10B — 계획치(~322B) 부합 |
>
> - fact_seeking fill 88.43%는 pad16 구조 비용(문서 5.74억 개 × 평균 ~57 real tok →
>   문서당 ~7.5tok 패딩). **가중/집계는 반드시 real tokens 기준** — 나머지 종은 fill 97~99.7%.
> - 운영 노트: 초단문 셋은 bestfit_pack emit이 NFS 대역이 아니라 **per-doc Python
>   천장(~35k docs/s ≈ 9MB/s)** 에 걸림 (fact_seeking emit 단독 6.5h). 128k filler
>   재패킹(러너북 작업 3) 시간 예산은 문서 길이 분포부터 확인할 것.
> - unpacked 전량 보존(`KEEP_UNPACKED=1`) — 128k 재패킹 재료.

`run_cpt_lc_v5.sh`로 4종 전체 처리 완료 (convert → tokenize → split(T=64k) →
bestfit_pack 32k). 산출: `LL_preprocessed/v5/cpt_lc{,_packed_32k}/<ds>/`.

| 데이터셋 | unpacked | lt<64k (32k 학습) | **ge≥64k (128k 보존)** | 32k fill |
|---|---|---|---|---|
| longblocks | 191,758 docs / 8.02B | 4.26B | **3.75B** | 99.73% |
| pg19 (train) | 28,602 / 2.80B | 0.40B | **2.40B** | 99.70% |
| edgar (전 스플릿) | 218,768 / 8.02B | 5.53B | **2.49B** | 99.99% |
| pes2o (s2orc ≥16k + filler 3%) | 378,867 / 5.38B | 5.37B | 0.01B | 98.78% |
| **합계** | **24.21B** | **15.56B** (477,275 bins) | **8.65B** | ~99.5% |

- **≥64k 보존 풀 실측 8.65B** — §4의 자연문서 추정(4.84B)의 1.8배. LongBlocks
  Institutional 재구성분(ge 3.75B)이 더해진 결과로, 128k 스테이지의 네이티브
  재료가 크게 보강됨 (다문서 합성 의존도 하락).
- 토큰 환산 추정 대비 실측 오차: pg19 +1.9%, edgar +1.6% (§2 방법론 검증).
- pes2o는 ≥16k 주 필터만으로는 fill 76.8%(bin당 1문서)에 그쳐 **filler 티어**
  (4k~14k tok 문서 3% 결정적 해시 샘플링)를 추가해 98.78%로 회복 — 상류 길이
  필터가 하류 bin-packing 효율을 결정하는 상호작용 사례.
- 운영 노트: fast_tokenize의 배치는 문서 "개수" 기준 — 책 단위 데이터셋에서
  기본 5000은 OOM (pg19 실측). `run_cpt_lc_v5.sh`가 데이터셋별 BATCH_SIZE 지정.

## 7. 재현

일회성 스크립트(scratchpad)로 수행: ① 3종 전수 문자 스캔(8-proc, peS2o 196샤드
~3.5분 + EDGAR 84파일 ~40초) → 문서별 char 길이 npz, ② tokenizer_v5 표본 인코딩으로
chars/token 산출, ③ 토큰 환산 percentile·버킷·임계값 집계. 본 문서의 표가 산출물
전체이며, 원 npz가 필요하면 동일 방법으로 재스캔하면 된다(입력 데이터는 불변).
