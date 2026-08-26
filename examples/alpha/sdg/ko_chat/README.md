# ko_chat — SFT 한국어 chat 데이터 합성 (2026-08-23~)

LC phase 동안 유휴인 **sub1(H100×8)** 로 SFT용 한국어 chat 데이터를 합성한다.
교사 모델 **gemma-4-31B-it** (Apache 2.0 — 증류 제약 없음), 서빙은 vLLM TP2×DP4.
출력 스키마는 `Nemotron-SFT-Instruction-Following-Chat-v3` 와 동일 →
`toolkits/sft_data_preprocessing/build_alpha_sft_idxmap.py` 를 그대로 탄다.

## 두 트랙

| 트랙 | 방법 | 원천 | 이유 |
|---|---|---|---|
| **A** `translate_regen.py` | 컨텍스트 번역 + **마지막 assistant 턴 재생성** (chat split) / **전체 번역** (IF split) | chat_v3 (적격 775k 대화) | 실사용자 발화 분포. 학습 턴이 번역투가 되지 않게 재생성 |
| **B** `ko_chat_sdg.py` | DataDesigner 네이티브 생성 (축: task×domain×persona×어투×턴수) | 시드 없음 (샘플러) | 한국 생활 맥락(전세·청약·연말정산·민원)은 번역이 못 만든다 |

### 트랙 A 모드 결정 (실측 근거)

`chat.with_prompts.jsonl` 637,663행 = **100% last-turn-only** 학습 →
컨텍스트(번역) + 마지막 턴(재생성). `instruction_following.jsonl` 249,748행 중
**61% multi-True** + IF 제약(특정 단어 시작·쉼표 금지)은 재생성이 위반 가능 →
IF 는 전량 번역(제약 참조 영어 단어는 원문 유지 지시).

## 실행

```bash
K=examples/alpha/sdg/ko_chat        # 모든 경로는 절대경로 사용 권장 (ssh 실행)
VENV=/home/work/vidsearch/repos/project_s/syn_data/.venv/bin/python  # DD 0.6.1

# 0) 서빙 (sub1) — 기동 3~4분
nohup bash $K/serve/serve_gemma31b.sh > $K/serve/logs/serve_<ts>.log 2>&1 &

# 1) 스루풋 확인 (선택)
python3 $K/bench_throughput.py --concurrency 64,128,192

# 2) 트랙 A: 시드 추출 → 실행 (재개는 같은 --out 재실행, uuid 멱등)
python3 $K/extract_sources.py --num-if 10000 --num-chat 20000 \
    --out $K/seeds_r1.jsonl --exclude $K/seeds_pilot.jsonl
python3 $K/translate_regen.py --seeds $K/seeds_r1.jsonl --out $K/out/r1_a --workers 128

# 3) 트랙 B: 시드 → (프리뷰로 프롬프트 반복) → 생성 → export
$VENV $K/prepare_ko_seed.py --num-records 20000 --out $K/ko_seed.parquet
$VENV $K/ko_chat_sdg.py --vllm-endpoint http://127.0.0.1:8000/v1 \
    --model gemma-4-31b --preview 8
$VENV $K/ko_chat_sdg.py --vllm-endpoint http://127.0.0.1:8000/v1 \
    --model gemma-4-31b --num-records 20000 --dataset-name ko_chat_b_r1 --no-tui
python3 $K/export_ko_chat.py --dataset "$K/artifacts/ko_chat_b_r1/**/*.parquet" \
    --out $K/out/trackB_r1.jsonl
```

## 실측 (2026-08-23 파일럿)

- 서빙: TP2×DP4, 동시성 ~96+에서 **~3,400 output tok/s ≈ 0.3B tok/일**.
- 트랙 A: 파일럿 16건 → ok 14 / reject 2 (codefence 게이트, 내용은 정상 — 튜닝 여지).
  동시성 16에서 0.145 rec/s → 128 워커 외삽 **일 3.5만~4.5만 레코드**.
- 트랙 B: 프리뷰 8/8 규칙 통과, 심판 4점 만점 다수 (31b 셀프심판은 관대 —
  export 게이트는 약하게, 품질은 규칙 게이트+중복제거가 주력. ko_grounded 전례).

## 품질 게이트

- 인라인(트랙 A): 한글비율 ≥0.40(코드 제외), 코드펜스 개수 보존, 번역 붕괴(<25%),
  교사모델 누출(gemma/gemini), 메타응답("번역할 내용을 주세요") 감지·재시도.
- **오탐 복구 경로 (2026-08-23, 사용자 지적)**: 한글비율 미달은 하드 리젝이 아니라
  **LLM 재판정 1회** — 영문 작성·번역·코드·약어 답변 등 요청상 정당하면 통과
  (`qc_note: low_hangul_llm_ok`). r1 실측: 리젝의 70%(479건)가 이 사유였고 표본
  8/8이 정당 판정 = 대부분 오탐이었음. 코드펜스 불일치도 해당 턴 temp0 재번역
  1회 후 재판정. r1 rejects(생성물 보존됨)는 이관 시 동일 로직으로 salvage.
- 트랙 B: `ko_check` 규칙(한글비율·누출·상투 서두) + `judge` 4점수 + export 중복제거
  (접두 **및 후미** 40자 버킷 — identity 실측: 후미 수렴이 더 심각).
- 다운스트림: `build_alpha_sft_idxmap.py` 가 injection 드롭·스팬 마스킹 재검증.
- **special-token 리터럴 방어 (LC-A iter170 정지 사고 반영, 2026-08-23)**: 원문에
  `<|...|>`·`<tool_call>`·`<think>` 류 리터럴이 남으면 토크나이저가 실토큰으로
  매칭될 수 있다(LC-A는 문서 내부 EOD → THD+CP 가드 정지). 4중 가드:
  extract(시드 드롭) → 트랙 A qc(리젝) → 트랙 B `ko_check`(리젝) →
  export(소급 게이트, 가드 이전 런 커버). 실측: seeds_r1 3/30,000(전부 타 모델
  토큰 형태), r1_a 부분 산출 0/4,018. **r1_a results.jsonl 은 가드 이전 시작이라
  이관 시 export 게이트 통과 필수.**

## reasoning 규약 (2026-08-24 결함 수정 — 사용자 결정: 폐기 후 재생성)

chat_v3 는 **전 행이 `reasoning_content` 보유**(chat·IF 전수 실측)이고 변환기가
이를 렌더하므로, 한국어 산출물만 think 를 비우면 "한국어→사고 생략" 언어-모드
상관을 학습시킨다. 수정:
- regen(chat): `<사고>…</사고>` 마커로 reasoning+답변 동시 생성 → 파싱 부착
  (2회 파싱 실패 시 리젝 `reasoning_parse_fail`).
- full_translate(IF): **학습 턴만** 소스 reasoning 을 번역 부착 (비학습 턴 think
  는 템플릿이 history 에서 제거하므로 생략 — 비용 절약).
- 트랙 B: `AssistantTurn.reasoning` 구조화 필드 (identity 전례).
- 신규 행은 `ko_synthesis.think: true`. 구판 think-less 행: regen분 폐기
  (`results_nothink_discarded.jsonl`) 후 재생성, IF분은
  `supplement_if_reasoning.py` 로 소급 번역(r1 은 redo 체인이 자동 수행,
  **r2_a 는 파일이 조용해진 뒤** — 완주 후 packaging 때 실행. 동시 append 와
  finalize 교체가 경합하므로 quiescent 필수).

## 두 번째 모델: OxAlpha (OpenRouter `stealth/ox-alpha`, 무료, 2026-08-25)

사용자 제안으로 도입. GLM 계열로 추정되는 stealth 모델 — 1M ctx, 입출력 $0,
reasoning 모델. 키는 `syn_data/.env` 의 `OPENROUTER`(chain_common.sh 가 source).

**⚠️ 계정당 1,000 요청/일 상한** (`free-models-per-day-stealth`, 리셋 UTC 00:00 =
KST 09:00). 도입 당일 A 재생성 40% 투입 20분 만에 소진되어 발견 — `:free` 접미
모델의 문서 한도(20/분·50/일)와 별개 버킷이고, 그 위의 상한을 "66건 연속 통과"로는
볼 수 없었다. **대규모 생성·전수 심판 백엔드로는 불가.**

**확정 용도 (사용자 결정 2026-08-25): 심판 캘리브레이션 전용.** 하루 ~900건을
트랙 B(Gemma 생성+Gemma 심판) 표본의 **재심판**에 써서 Gemma 셀프심판의 관대함을
정량화하고 export 임계를 보정한다 — 3,600건 판정이 B 100k+ 전체의 필터 품질을
좌우하므로 레버리지가 가장 크다. reasoning effort **high**(사용자 지시; 요청 상한
체제에선 요청당 가치 극대화가 맞다), max_tokens 12288(숨은 사고가 예산을 소비).
- `calibrate_judge.py` — 같은 4축 루브릭, 결과 `out/judge_calib/<date>.jsonl`,
  `--report` 로 축별 평균·차이·임계(<4) 통과율 비교.
- `calib_daily.sh` — 매일 UTC 00:05 자동 실행 (sub1 nohup 루프).
- 생성 경로(`translate_regen.py --openrouter-frac`, `ko_chat_sdg.py --gen-backend
  openrouter`, `chain_b_r2.sh` `B_OR_SLICE`)는 구현·검증됐으나 **기본 off** —
  상한이 풀리거나 유료 전환 시 스위치만 켜면 된다. 레코드별 `ko_synthesis.regen_model`
  과 최상위 `model` 에 교사 기록.
- 일일 상한 429 는 재시도 무의미 → `DailyCapExceeded` 서킷브레이커(1시간 비활성,
  포괄 except 보다 먼저 잡아야 함 — 삼켜서 5회 재시도한 버그 실측).

**캘리브레이션 1일차 결과 (2026-08-26, B r1 887행, OxAlpha effort high)**:

| 축 | Gemma 심판 | OxAlpha | OxAlpha <4 |
|---|---|---|---|
| korean_naturalness | 5.00 | 4.69 | 0% |
| helpfulness | 4.98 | 4.38 | 6% |
| factuality | 4.99 | **4.49** | **10%** |
| coherence | 4.99 | 4.95 | 1% |

- Gemma 셀프심판은 사실상 무정보(전부 5). factuality 2점 사례는 **전부 지어낸
  상호·지점명·가격·시설 정보**(맛집·한정식 코스 가격 등). 취약 도메인: 금융·재테크
  3.86, 부동산·주거 3.89, 여행 4.1.
- **프록시 가설 기각**: 같은 엄격 루브릭을 Gemma 에 줘도(`--backend local`) factuality
  4.84 — OxAlpha 가 <4 로 잡은 것의 재현율 0.23(helpfulness 0.42), r≈0.4~0.5.
  Gemma 심판은 어떤 프롬프트로도 전수 필터가 못 된다.
- **결론 = 원천 차단**: B r2(100k) 생성 프롬프트에 `--strict-facts` 규칙(확실치 않은
  상호·가격·수치·조항 금지, 확인 방법 안내) 부착. 효과는 다음날 캘리브레이션(r2 행)의
  factuality<4 비율로 측정 — 10% 대비 감소 폭이 판정. r1(20k, 구 프롬프트)은 이관 시
  OxAlpha 판정분(<4) 직접 드롭 + 나머지는 규칙 게이트만.

**실측 특성·대응** (전부 translate_regen.py 에 구현):
| 특성 | 대응 |
|---|---|
| `response_format` 스키마 **미강제** (실전 6/6 파싱 실패) | 바깥 `{…}` 추출 + `strict=False`(문자열 내 리터럴 개행) + 2차 구분자 salvage(이스케이프 안 된 따옴표) |
| 시스템 지시보다 마지막 턴 지시를 따름 | OR 전용으로 마지막 user 턴 끝에 형식 지시 부착(산출물 미저장) |
| 숨은 사고가 **영어** | 쓰지 않음 — 한국어 사고는 JSON 필드로 생성. `reasoning.effort=low` 로 지연 절감 |
| 사고에 "JSON 형식을 지켜" 누출 | 프롬프트 금지 + `REASONING_FORMAT_LEAK_RE` 가드(재시도) |
| 429 (동시 48 에서 10%) | 지수 백오프+지터 5회 → 실패 시 **Gemma 자동 폴백**(`FALLBACK` 로그). stealth 모델은 예고 없이 사라질 수 있어 무인 런이 멈추지 않게 |
| 일일 상한 1,000 요청(`free-models-per-day-stealth`) | `:free` 문서 한도와 별개 버킷. 재시도 무의미 → 즉시 감지·1시간 비활성. 용도를 캘리브레이션(≤900/일)으로 한정 |

파일럿(16시드): 재생성 8/8 OxAlpha 처리, 파싱 실패 0, 누출 0, 폴백 0.
**주의**: 무료 stealth 모델은 프롬프트가 제공자 학습에 쓰일 수 있음 — 보내는 것은
공개 데이터(Nemotron CC-BY-4.0·WildChat) 파생물과 우리 합성 출력뿐.

## 함정 (재발 방지)

1. **빈 system content 를 LLM에 넣으면 메타응답**("번역할 지시문을 주세요")이
   그대로 저장된다 — 빈/공백 턴은 무호출 통과 (파일럿 실증).
2. DD 0.6.1(syn_data venv): `RunConfig(display_tui=…)` 없음 → `progress_bar`;
   `ResumeMode` 는 `data_designer.interface` 에서 import; LOCAL_CALLABLE 검증
   함수는 **DataFrame 을 반환**해야 한다 (list 반환 시 to_dict 크래시).
3. identity README 의 `uv run` 은 이 호스트에 uv 없음 — syn_data venv 직접 사용.
4. ssh 원격 nohup 은 러너/단일 명령만 (`A && nohup X &` 체인 금지 — 루트 CLAUDE.md).
5. WildChat 원천엔 러시아어·일본어 대화 상당수 — 번역 프롬프트는 다국어→한국어.

## 산출물 위치

- 트랙 A: `out/<run>/results.jsonl` (+ `rejects.jsonl` — 생성물 포함, 진단용)
- 트랙 B: `artifacts/<dataset>/**.parquet` → `export_ko_chat.py` → jsonl
- 최종 이관(예정): `/home/work/Datasets/LL_datasets/posttraining/SFT/alpha-SFT-KoChat-v1/`
  → 64k 변환(`build_alpha_sft_idxmap.py`; IF 번역본은 `--fanout-train-turns --medium-effort` — 원본이 GPT-OSS medium-effort 생성분, `docs/SFT_RL_DATASETS.md` §2.6) → 블렌드 yaml (chat 21% 내 비중은 미정)
