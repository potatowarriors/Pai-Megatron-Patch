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
  → 64k 변환(`build_alpha_sft_idxmap.py`) → 블렌드 yaml (chat 21% 내 비중은 미정)
