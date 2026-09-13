# 트랙 U — 구조화 출력·사용성 SFT 셋 (usab_v1)

**목적**: Ultra 미공개 도메인 "usability" 와 구조화 출력(JSON 스키마·표·YAML·"JSON만") 결함(벤치 실측 `SFT_BENCHMARKS.md` "extra text after the JSON") 교정.
**언어**: 영어 80 / 한국어 20 (사용자 결정 2026-09-12 — 영어 공개 셋에 없는 능력을 만드는 트랙이므로 영어 중심). 한국어 IF 셋은 v1/v2 폐기 후 부재.
**규모**: 36k 잡 → ≈30k 행(리젝 15%), thinking/no-think 70/30. 24시간 예산 계획의 일부(`docs/STATUS.md`).

## 레시피
- 시드: EN = Chat-v3 복원 영어 프롬프트(`chat.with_prompts.jsonl`, 단일턴·영문·40~1,500자·정체성 제외), KO = ko-chat v3 P1 시드(`seeds_p1_A/B`, 기사 첨부형 20%).
  되묻기 템플릿은 짧은 시드(EN 15~90자 / KO 8~40자) 풀 별도.
- 제약 템플릿 15군(`u_templates.py`, 가중치): json_schema 6 · json_keys 2 · csv 1.5 · md_table 1.5 · yaml 1 · regex 2 · toolcall_json 1 · length 3 · format 3 · tone 1.5 · language 1.5 · keywords 1 · clarify 2 · refuse 1.5 · followup 2.5.
  각 템플릿 = EN/KO 지시문 + **프로그램 검증기**(JSON 파서·jsonschema·CSV/표 파서·YAML·정규식·단어/글자/문장/문단 수·헤딩/굵게/코드블록·언어 비율·존댓말/반말 어미·되묻기(질문형·짧음)·거절(한계 인정 어구)·후속턴 변환(길이 반감·표·번역·글머리표·문장 단순화)).
- 교사: GLM-5.3-Flash(**v2 철저 지시문**, 예산 8k) + DSV4-Flash 1:1 라운드로빈, n=2 → 검증기 통과 샘플만 후보 → 상대 교사 심판(언어 중립 프롬프트, 순서 무작위) → `generate_v3.gate`(중국어·자기귀속·special·퇴행).
  후속턴(followup)은 1턴 답을 먼저 생성해 히스토리로 두고 변환 지시를 2턴으로 붙여 **마지막 턴만 학습**.
- 시스템 프롬프트: EN 은 영어 identity + 영어 v2 지시, KO 는 `generate_v3.SYS_GEN`(학습 행에는 미포함).

## P0 (100 잡, 2026-09-12) — `out/p0/U_P0_REPORT.md`
채택 85 · 양샘플 리젝 15 · 심판 A/B 36:35 · completion tok p50 1,244(ko-chat 의 1/3). 리젝 사유: 빈 답변 14(8k 초과)·검증기 13·GLM 중국어 사고 8·퇴행 3.
15군 전부 채택 발생, 검증기가 형식 위반(답 대신 질문 미출력·표 누락·단어수 초과)을 정확히 리젝. 사용자 승인 후 본생성 36k 착수(`out/p1/u_main.jsonl`).

## 실행
```
GEN_THOROUGH=2 GEN_TEACHERS=glm53-flash,dsv4-flash JUDGE=other python3 u_generate.py --n 36000 --out out/p1/u_main.jsonl --workers 352 --seed 2
```
출력 행: Chat-v3 스키마(messages, 마지막 assistant 에 reasoning_content) + `lang`, `u_meta{family,...}`, `ko_synthesis{validator: pass, ...}`. 내보내기·변환은 ko_chat_v3 의 `export_sft.py`(train_turns 마지막만, no-think 30%) → 128k bins `usab_v1_think/nothink`.

## 완료 (2026-09-12 14:45 KST)
본생성 36k 잡 → 채택 30,018(84%; EN 24,134 / KO 5,884), 15군 전부(json_schema 6,383 … clarify 779), 교사 DSV4 15,664 / GLM 14,354, 판정 A/B 12,966:12,813, 자기귀속 0.
발견: 1,065행(3.5%)이 러시아어 등 비영어 요청(Chat-v3 영어 풀의 키릴 프롬프트가 라틴 비율 필터를 통과) → 사용자 결정으로 제외.
**bins** `sft_packed_128k_terminal_pad16/usab_v1_think` 20,272행 → 273 bins(실 35.6M·학습 32.5M tok) · `usab_v1_nothink` 8,539행(29.6%) → 35 bins(4.4M·3.1M). `verify_sft_bins` PASS, `render_check` clean.
교훈: 영어 시드 필터는 라틴 비율이 아니라 비라틴 문자 비율(키릴·아랍·데바나가리 등)로 걸러야 한다(`u_generate.load_en` 개선 과제).
