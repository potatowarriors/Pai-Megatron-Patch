---
paths:
  - "toolkits/sft_data_preprocessing/**"
  - "examples/alpha/sdg/**"
  - "examples/alpha/configs/data/sft_*.yaml"
---

# SFT 데이터 작업 규칙 (Alpha)

- **새 SFT 셋을 추가하기 전에 `examples/alpha/docs/INTERLEAVED_THINKING.md` §"새 SFT 셋 추가 시 규칙 9건"을 읽는다.**
  think-히스토리 규약(DSV4식 tool-시나리오 분기, IF fan-out, keepthink)은 거기가 정본이다.
- 변환기는 `toolkits/sft_data_preprocessing/build_alpha_sft_idxmap.py`. 마스킹 규약의 정본은
  `examples/alpha/tools/verify_chat_template.py`(34 tests) — 둘이 어긋나면 verify 쪽이 맞다.
- 산출물은 학습 투입 전 **`verify_sft_bins.py` 게이트 전 PASS** 필수 (EOD 오염 0, %16 정렬, 리터럴 special-token 0).
- 원문의 리터럴 `<|endoftext|>`는 4중 가드(extract 드롭 → 인라인 리젝 → export 게이트 → 학습 전 스캔)로 차단한다.
  LC-A iter170 사고의 원인이었다 (`docs/KNOWN_ISSUES.md` 2026-08-23).
- SFT×THD×CP 학습에서 `--eval-iters 0` 금지 (valid split이 있으면 sampler assert). bins<100인 셋은 valid 0-doc 무한대기 —
  입력을 늘려 회피.
- 한국어 합성 트랙(`examples/alpha/sdg/ko_chat/`)의 함정 5건은 그 README에 있다.
- 블렌드 예산·epoch 원칙: SWE 1-pass 앵커, chat E_max 4~5. 근거는 `SFT_RL_DATASETS.md`.
- 새 셋은 `render_check.py` 로 `<tool_response>` 와 **`<tools>` 선언부**를 본다. tool_call 이 있는데 tools 가 없으면 사이드카로
  선언을 복원(`--tools-sidecar`), 도구 스키마가 MCP 형이면 `normalize_tool_schema` 가 정규화한다. `train_turns` 없는 멀티 user
  reasoning 셋은 `--fanout-implicit-turns`. 근거·실측은 `docs/KNOWN_ISSUES.md` 2026-09-09.
- effort 마커 `{reasoning effort: efficient}` 는 데이터에 없고 렌더 플래그(`--medium-effort`)로 붙는다.
  IF split 은 항상 이 플래그로 변환. 근거·RL `effort_levels` 요건은 `SFT_RL_DATASETS.md` §2.6.
- **새 도메인 데이터를 합성하기 전에 공개 릴리스를 조직 단위로 훑는다** — 컬렉션(`Nemotron-Post-Training-*`) 안만 보지 말고 `huggingface.co/nvidia` datasets 탭 전체와
  최근 arXiv 의 데이터 절까지. 2026-09-10 `Nemotron-Terminal-Corpus`(366k, 컬렉션 밖)를 놓치고 3일 합성한 사고 (`docs/KNOWN_ISSUES.md` 2026-09-10).
