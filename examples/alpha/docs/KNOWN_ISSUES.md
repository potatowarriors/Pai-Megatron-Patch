# Alpha Known Issues & Fixes — 전체 기록

`examples/alpha/CLAUDE.md`에서 2026-08-25 이관한 사고·수정 기록 전문 (최신순).
CLAUDE.md의 "함정 표"는 이 문서의 한 줄 요약이며, 새 사고는 **여기에 서사를 쓰고 CLAUDE.md 표에는 한 줄만** 추가한다.
날짜는 절대 표기. 두 스테이지 이상 지난 항목은 스테이지 경계에서 `archive/`로 이동.

## 진행 감시 스크립트가 로컬만 봐서 정상 실행을 "중단"으로 읽었다 (2026-09-01 ✅)

iter900 스위트 상태를 `progress.sh` 로 보니 **프로세스 (없음) / 백엔드 0/8 / 프록시 000** 이었다.
동시에 main1 GPU 8장은 88~90% 로 돌고 있었다. "죽었는데 GPU 는 왜 바쁜가" 로 20분을 썼다.

실제로는 **전부 정상**이었다. 원인은 셋이 겹쳤다.

| 관측 | 진짜 원인 |
|---|---|
| 프로세스 없음 | 스위트는 **sub1** 에서 돈다. `progress.sh` 는 `pgrep` 을 로컬(main1)에서 했다 |
| 백엔드 0/8 | fleet 도 sub1 의 localhost:8000~8007. main1 에서 curl 하면 당연히 000 |
| GPU 88~90% | fleet 이 아니라 **SFT 학습**(`pretrain_alpha.py`, 4일째)이다 |

로그(`/home/work/vidsearch/tools/bench_logs/*.log`)는 NFS 공유라 main1 에서 보인다. **로그는 보이는데
프로세스는 안 보이는 조합**이 "죽은 실행" 의 외형을 정확히 흉내냈다.

교훈은 `progress.sh` 주석에 이미 적혀 있던 것과 같다 — *진행은 산출물로 센다*. 그런데 정작 그 스크립트의
프로세스·fleet 절만 로컬 가정으로 남아 있었다. 자기 진단 코드에는 자기 규칙을 적용하지 않은 셈이다.

수정: `BENCH_HOST`(기본 `sub1`) 를 두고 프로세스·fleet 조회를 그 호스트에서 한다. 로컬이면 ssh 를 건너뛴다.
겸사겸사 pgrep 패턴에 빠져 있던 **`run_suite.sh`**(오케스트레이터)와 `lm_eval` 을 넣었다 — 이게 없어서
스위트 본체가 목록에 뜨지 않았다.

판별법: GPU 사용률로 fleet 생존을 판단하지 말 것. 같은 노드에서 학습이 돌면 구분되지 않는다.
`bash eval_sft/progress.sh` 의 `백엔드 8/8 프록시 200` 만 신뢰한다.

## SFT 블렌드 실측 검증에서 나온 데이터 결함 3건 (2026-09-01 🔶 phase-2에서 수정)

phase-1 본 런(`alpha_baseline_48L_sft_128k_full_20260828_081911`, iter 1,045/2,448 시점)의 블렌드를
"본 런 인자 = yaml → 멤버별 epoch → 렌더 플래그 → 게이트 → loss" 순으로 검증하다 찾았다. 비율·epoch·플래그·
게이트는 전부 설계대로였다(수치는 `SFT_RL_DATASETS.md` §2.7). 결함 셋은 **"렌더된 토큰열이 배포와 같은가"를 셋
단위로 눈으로 본 적이 없다**는 한 뿌리에서 나왔다. 진행 중 런은 건드리지 않고 phase-2(`SFT_PHASE2_PLAN.md`)에서 고친다.

### ① opencode_v1 의 tool 결과가 Python repr 로 렌더된다

- **증상**: `Nemotron-SFT-OpenCode-v1` 6 서브셋 전부, tool 메시지 `content` 가 문자열이 아닌 **list**
  (`[{'type':'tool-result','toolCallId':…,'toolName':…,'output':{'type':'text','value':'…'}}]`; 표본 12,359건 100%,
  12,358건 1-item·1건 3-item, `output.value` 전부 str). 템플릿 `{{ message.content }}` 가 list 를 그대로 str() 하므로
  `<tool_response>` 안에 Python repr 이 들어가고 값 안의 줄바꿈은 리터럴 `\n` 두 글자가 된다 — 파일 목록·diff·코드가
  한 줄로 뭉개진다.
- **대조**: swe_v3(앵커 18.7%)·arc_agi 는 tool content 100% str 평문. 배포 하네스(mini-swe-agent·terminus)도 평문.
- **영향**: tool_response 는 비학습 스팬이라 틀린 정답은 아니지만, 블렌드 4.27%(0.31ep, 문맥 7.15B tok)가 배포에서
  절대 안 나오는 봉투 형식으로 "tool 출력 읽기"를 가르친다. 덧붙여 이 셋은 assistant 16,270턴 표본 중 reasoning **0** —
  전 스텝 `<think></think>` no-think 에이전틱인데, 에이전틱 벤치 러너(`run_swe.sh`/`run_terminal.sh`)는 thinking ON.
- **왜 못 잡았나**: `verify_sft_bins` 는 EOD 오염·%16·리터럴 special-token 만 본다. 변환기 `normalize_row` 는 tool
  content 타입을 검사하지 않는다(str 가정).
- **대응(phase-2)**: `normalize_row` 에서 list → `"\n".join(item.output.value)` 평문화(미지 형식은 `bad_row` 드롭, 조용한
  str() 금지) + 유닛 4종 + 렌더 육안 1건. 규칙 9 신설(`INTERLEAVED_THINKING.md` §7): 새 셋은 tool_response 렌더 1건을 본다.

### ② identity_v1 실효 반복 ≈180회

- **사실**: 원본 7,315행 ≈ 1.18M tok. bins≥100 확보용 ×12 복제 파일(87,780행·14.2M tok·114 bins) 사용. 가중치 0.4271%
  → 소비 219M tok = ×12 파일 15.4ep = **원본 기준 ≈180회**. per-seq 평균(짧은 샘플 = 1표)이라 토큰당 가중도 크다.
- **규칙 대조**: `DATA_PREP_LOG.md` 결정 #9 "비중 0.3~1.0% 상한"(0.43% ✔)·"identity 단독 반복 에폭 금지"(혼합 ✔)는
  문자 그대로 충족. 그러나 비중 상한을 1.2M tok 셋에 적용하면 180회가 따라오고, 결정 #9 의 동기("무엇을 물어도
  자기소개하는 과적합")가 바로 이것이다. 반복 횟수는 어디에도 계산돼 있지 않았다.
- **징후 기록**: 없음(TRACKING·chat README 검색 0건). 채팅 fleet 프로브(정체성 무관 20문항 자기소개 혼입률)는 09-01 시점
  fleet 다운으로 **미실행** — phase-2 게이트 G-P6 에 편입.
- **집계 확인(09-01)**: 본 런 `calculate_per_token_loss=False` → `schedules.py:249-255` 가 마이크로배치(bin)마다 자기 num_seqs 로 나눔 = **bin 1표**.
  샘플 1표로 가정하면 identity 표 점유 15.6%(×36)로 보이지만 착시. gradient 점유는 토큰 비중과 같은 0.43%.
- **대응(phase-2, 사용자 요구 반영)**: 사용자는 "누가 만들었어" → 한 문장으로 조직·팀 소속을 앞세워 **"이동호" 명시**
  ("저를 만든 사람은 CJ주식회사 AI/DT추진실 영상콘텐츠담당 이동호입니다.", 회피 금지)를 요구. 현 카드 v1.1 은 모호 질문에 조직만
  답한다 → 카드 v1.2 + creator 슬라이스 재생성 후 identity_v2 를 0.6% 로 연속학습해 덮어쓴다. `SFT_PHASE2_PLAN.md` §3.

### 심각성 판정 (2026-09-01, 재실행 여부 결정용)

| | ① | ② | ③ |
|---|---|---|---|
| 규모 | 블렌드 4.3%; 에이전틱 tool-루프 **문맥** 소비 중 ≈20%(repr 1.84B vs 평문 swe 6.84B+arc 0.72B) | gradient 0.43%(bin 1표) | chat 행 11.8% = 전체 토큰 0.6% |
| loss 오염 | 없음(비학습 스팬) | 없음 | 없음 |
| 배포 발동 | **없음** — repr 봉투는 배포에서 안 나옴 | 조건부(정체성 무관 질문에 누출 시) | 없음(다양성) |
| 판정 | **중간** | **낮음~중간(측정 전)** | **낮음** — 미복원 = 공개판 제외 행(toxic 제외판 추정) |

→ 재실행을 강제하는 결함 없음. 연속형(500~600 iters) 채택(사용자 2026-09-01). 상세 `SFT_PHASE2_PLAN.md` §8.

### ③ chat_v3_chat 프롬프트 복원 잔여 11.8%

- **사실**: `chat.with_prompts.jsonl` 637,663행 중 첫 user 가 null 인 75,287행(11.8%)이 `null_content` 드롭, 562,376행 사용.
  출처별 미복원: WildChat-1M 58,977/221,621(26.6%) · lmsys-chat-1m 16,310/134,161(12.2%). lmarena·HelpSteer2 출처는 0.
- **원인(가설)**: 복원은 `seed_prompt_sha256` 정확 일치. 공개 `allenai/WildChat-1M` 은 toxic 제외판(전량은 gated
  `WildChat-1M-Full`), lmsys 도 일부 리댁션 → 해시 미매칭. 로컬 `prepare_chat_prompts_partial.py` 는 미매칭을 null 로 남기고
  진행하는 변형이다.
- **영향**: epoch 는 남은 행 기준이라 블렌드 비율 왜곡 없음. 드롭이 두 출처에 몰려 chat 다양성이 줄었고 문서 기록이 없었다.
- **대응**: `prepare_chat_prompts_full.py`(= partial 변형, `WILDCHAT_DATASET` 만 `allenai/WildChat-1M-Full`) 를 `.env` 의 HF_TOKEN 으로
  `chat.with_prompts.jsonl` 위에 재실행(2026-09-01). **결과: 회수 불가로 종결 — 88.2% 정본.**
  ① lmsys: null 16,310행 = 5,110 해시가 공개 lmsys-chat-1m 에 없음(리댁션/리비전) — 회수 불가 확정.
  ② WildChat: null 58,977행(14,286 해시) → `allenai/WildChat-1M-Full` 은 수동 승인 gated, `.env` 토큰 계정(potatowarriors)은
     **미승인**(파일 요청 403 "not in the authorized list"; `dataset_info`·파일 목록은 gated 라도 공개라 접근되는 것처럼 보였음).
     승인을 받으면 `RESTORE_FAMILIES=wildchat python3 prepare_chat_prompts_full.py …` 로 재시도(별칭 매핑·null 행 한정 패치 포함) →
     회수분 `chat.restored_only.jsonl` → `convert_sft_128k_mixed_p2.sh` 가 자동 변환. iter 1200 교체에는 미포함.
  ③ 부수 사고: Full 변형 1차 실행에서 `WILDCHAT_DATASET` 상수만 바꿔 행 메타 `seed_dataset`(공개판 이름)과 매핑이 어긋나 WildChat 이
     처리 대상에서 빠짐(회수 0) → 별칭 매핑으로 수정.

## 판정기·프롬프트가 침묵을 점수로 위장한 사고 2건 (2026-08-31 ✅)

RULER 저점(35/10/5/0)과 SimpleQA 저점(1.2%)을 조사하다 찾았다. **둘 다 "작은 토큰 예산을
받은 추론 모델은 침묵한다"는 같은 뿌리**다 — 우리 모델이 128토큰 RULER 에서 겪은 것과 같다.

### ① RULER — `gen_prefix` 가 완결 메시지로 전송돼 생성이 0토큰

- **증상**: single_1 35% / single_2 10% / multikey 5% / multivalue 0%.
  **64K 점수가 128K 보다 낮은 역전** — 진짜 롱컨텍스트 열화면 반대여야 한다.
- **원인**: lm_eval 의 `gen_prefix` 가 chat 경로에서
  `{"role": "assistant", "content": "The special magic number ... is"}` 라는 **완결된
  assistant 메시지**로 나간다. 완결된 턴이므로 모델은 이어쓰지 않고, 서버는 prefix 를
  그대로 돌려준다. vLLM 에서 이어쓰려면 `continue_final_message` 가 필요한데 lm_eval 은
  보내지 않는다.
- **증거**: 160 샘플 중 **131건(82%)** 의 `resps` 가 `gen_prefix` 와 글자 그대로 동일.
  35/10/5/0 은 측정이 아니라 나머지 18% 의 잡음이었다 (구간당 n=20).
- **수정 (하니스 v3.0)**: `gen_prefix` 제거, T1 방식대로 프롬프트에 답 형식을 지시
  ("Answer with only the special magic number(s), separated by commas. Do not explain.").
  `no_answer` 게이트 추가 — 응답이 비었거나 프롬프트 꼬리를 반향하면 무효.
- **LC-B 는 무관**: LC-B 자체 NIAH(4k~131k 200/200, 256K 95%)는 별개 하니스 측정이고
  이 버그의 영향을 받지 않는다. 이 수치로 LC-B 실패를 주장할 수 없다.

### ② SimpleQA — judge 가 8토큰을 받고 침묵, 그 침묵이 NOT_ATTEMPTED 로 흡수

- **증상**: accuracy 1.2%, **not_attempted 761/1000 (76%)**.
- **원인**: `run_simpleqa.py` 가 `judge_batch(..., max_tokens=8)` 을 넘겼다.
  judge(`gemini-3.7-flash`)도 **추론 모델**이라 8토큰을 내부 사고에 다 쓰고 텍스트를
  내놓지 않는다. 실측: **8토큰 → `''`, 64토큰 → `'A'`**.
  그리고 `cls()` 가 빈 응답을 조용히 `NOT_ATTEMPTED` 로 매핑했다.
- **재채점 (132건 표본, judge 정상화 후)**: 판정 실패 0건.
  기존 NOT_ATTEMPTED 80건 → INCORRECT 64 / NOT_ATTEMPTED 16 / **CORRECT 0**.
  → **accuracy 1.2% 는 유효**(정답 누락 없음). `attempted_rate` 23.9%→~85%,
  `correct_given_attempted` 5.0%→~1.4% 로 왜곡돼 있었다.
- **모델의 실제 행동은 회피가 아니라 환각**이다. 예: "1971년까지 승무원으로 일한
  아이슬란드 前 총리" 에 실존하지 않는 "Einar Kárason (1910–1971), 1967~71 총리" 를
  상세히 지어낸다.
- **수정**: judge `max_tokens` 8→256. 빈 응답을 `JUDGE_FAIL` 로 분리해 등급으로 위장하지
  않고, 그 비율을 `no_answer` 로 노출해 임계 초과 시 집계기가 자동 무효 처리.
  `gemini_judge.MIN_OUTPUT_TOKENS=64` 하한을 강제 — 호출부가 작은 값을 넘겨도 막는다.

- **교훈**: **침묵을 등급으로 매핑하지 말 것.** 측정 실패와 "모델이 못했다"는 다른 사건이다.
  기본값(`.get(..., "NOT_ATTEMPTED")`, `empty=0`)이 실패를 정상으로 흡수하면 게이트가
  통과시킨다 — 실제로 `summarize.py` 가 두 벤치 모두 "유효"로 찍었다.

## Terminal-Bench 0/80 — terminus 는 잘린 응답에서 태스크를 버린다 (2026-08-31 ✅)

- **증상**: iter300 Terminal-Bench 80/80 미해결. 실패 원인 `unknown_agent_error` 44건(55%),
  `agent_timeout` 14건, `unset` 20건.
- **원인**: `max_tokens=16384` 로 돌렸는데 추론 모델의 한 턴이 그것을 넘었다. terminus 는
  `finish_reason == "length"` 를 받으면 `OutputLengthExceededError` 를 던지고 **태스크를
  즉시 중단한다 — 재시도가 없다** (`terminal_bench/llms/lite_llm.py:175`).
  mini-swe-agent 가 `RepeatedFormatError` 로 몇 번 더 시도하는 것과 다르다.
  실제 로그: `Error running agent for task openssl-selfsigned-cert: Model openai/alpha hit
  max_tokens limit.` 44건의 출력 토큰 중앙값이 **0** 이다 — 첫 턴에서 죽었다는 뜻.
- **수정**: `max_tokens` 16384 → **32768** (`TERM_MAX_TOKENS` 로 조절). 컨테이너
  `alpha_model_registry.json` 의 `max_output_tokens` 도 함께 올렸다 — litellm 이 이 값으로
  판정한다.
- **동반 결함**: 결과 파서가 `n_tasks`/`total` 을 찾았는데 terminal-bench 0.2.x 스키마는
  최상위에 `n_resolved`/`n_unresolved`/`accuracy` 를 둔다. `total=0` 이 되어 정상 결과도
  무효로 찍혔다. → `n_resolved + n_unresolved` 로 계산하고, `failure_mode` 분포를 결과 JSON 에
  함께 남긴다(0점이 모델 실패인지 하니스 실패인지 가르는 신호).
- **교훈**: 에이전틱 하니스마다 **잘린 응답의 처리 방식이 다르다.** SWE(mini-swe-agent)는
  재시도하고 Terminal(terminus)은 버린다. 같은 `max_tokens` 로 둘을 돌리면 한쪽만 죽는다.

## 에이전틱(SWE·Terminal) 0점의 진짜 원인 3중 — 모델이 아니었다 (2026-08-30 ✅)

2026-08-30 SWE 0/20 · Terminal 0/10 은 모델 실패가 아니라 **설정 결함 3건**의 합이었다.
게이트 A1~A3 를 통과시킨 뒤에도 두 겹이 더 남아 있었다.

- **① litellm 이 미등록 모델의 비용 계산에서 죽는다.** `RuntimeError: Error calculating
  cost for model openai/alpha: This model isn't mapped yet.` 3/3 인스턴스가 **3초 만에**
  실패. 로컬 모델은 `LITELLM_MODEL_REGISTRY_PATH` 로 등록하는 것이 정식 경로
  (mini-swe-agent 로컬 모델 가이드). `MSWEA_COST_TRACKING=ignore_errors` 도 함께.
  → `/opt/{swebench,terminalbench}/alpha_model_registry.json` 작성, 러너에서 export.

- **② tool-call 파서가 모델이 배운 형식과 달랐다 (핵심).** fleet 를 `--tool-call-parser
  hermes` 로 띄웠는데 hermes 는 `<tool_call>{"name":…,"arguments":{…}}</tool_call>` 라는
  **JSON 본문**을 기대한다. 우리 챗 템플릿(Nemotron 3 Ultra + DSV4 분기)이 가르친 형식은
  XML 이다:

      <tool_call>
      <function=bash>
      <parameter=command>
      ls -la
      </parameter>
      </function>
      </tool_call>

  파서가 파싱에 실패해 `tool_calls: null` 이 되고 원문이 `content` 에 남는다. 에이전트는
  "No tool calls found in the response" 를 받고 `RepeatedFormatError` 로 종료한다.
  **모델은 처음부터 정확한 형식을 내고 있었다** — 실측으로 원문을 떠서 확인했다.
  → vLLM 파서 목록에서 `<function=`/`<parameter=` 를 다루는 것은 `qwen3_xml`(=
  `qwen3_engine_tool_parser`)과 `step3p5` 둘뿐. `qwen3_xml` 로 교체하니 즉시
  `finish_reason: tool_calls` + 인자 정상 파싱.
  → `serve_alpha.sh` 의 파서를 `TOOL_PARSER` 환경변수로 뺐다(기본 `qwen3_xml`).

- **③ 예측 파일 경로가 틀렸다.** mini-swe-agent 는 `preds.json`(단수, dict)을 쓰는데
  러너는 `preds.jsonl` 만 찾아 채점이 매번 `Invalid value: pass --gold or --predictions`
  로 실패했다. 예측이 있어도 점수가 0 으로 남는다.

- **부수**: 컨텍스트 40,960 은 에이전틱에 좁다(`ContextWindowExceededError` 발생).
  에이전틱 fleet 는 **106,496** 로 띄우고 레지스트리 `max_input_tokens` 도 98,304 로 맞춘다
  (litellm 은 레지스트리 값으로 컨텍스트 초과를 판정하므로 서빙 창과 함께 올려야 한다).

- **교훈**: 에이전틱 0점은 원인이 여러 겹이다. "게이트 통과 = 측정 가능"이 아니었다 —
  A1(`tool_choice=auto` 수용)은 파서가 **등록됐는지**만 보지 그 파서가 **맞는지**는 보지
  않는다. 모델이 실제로 내는 원문을 떠서 파서와 대조하는 것이 유일한 확인 방법이다.

## RULER 를 추론 켠 채 128토큰으로 돌렸다 (2026-08-30 ✅)

- **증상**: RULER-NIAH 65536 구간 6~25%. 같은 모델이 LC-B 자체 NIAH 하니스에서는 4k~131k
  **200/200**(100%)이었다. 두 수치가 모순된다.
- **원인**: 추론을 **켠 채** 출력 예산만 128 토큰으로 조였다. 추론 모델은 그 128 토큰을
  서두 분석에 전부 쓴다 — 실제 응답이
  `"The user is asking for the special magic number for straight-place mentioned in the
  provided text.\n\n1. **Analyze the Re"` 에서 잘렸다. needle 을 쓸 기회가 없었다.
- **프론티어는 RULER 에서 추론을 끈다**: Nemotron Nano 9B v2 카드 —
  *"except RULER, which is evaluated in **Reasoning-Off** mode"*. Nemotron 3 Ultra 는
  RULER 를 instruct 가 아닌 base 스위트로 분리하고 `temp 0.00001 / top_p 0.99` 를 쓴다.
  Qwen3-235B 는 thinking budget 을 8,192 로 제한한다("to avoid overly verbose reasoning").
- **실측 대조** (iter300, 동일 needle 프롬프트): thinking ON = `finish=length`, 512토큰
  소진, needle 실패 / thinking OFF = **`finish=stop`, 21토큰, needle 성공**.
- **수정**: `eval_sft/tasks/ruler_niah_*_aa.yaml` 4종 + `ruler_utils.py`.
  요청에 `chat_template_kwargs: {enable_thinking: false}` — alpha 챗 템플릿이 그때
  `<|im_start|>assistant\n<think></think>` 로 사고를 미리 닫아 렌더한다. 상세는
  `docs/SFT_BENCHMARKS.md` §3.9.
- **동반 결함**: 구 `common_utils.process_results` 가 센티넬 dict 를 하드코딩된
  `DEFAULT_SEQ_LENGTHS = [4096]` 로 만들어, 샘플이 0개인 4096 구간에 `-1.0` 이 결과에
  남았다. 새 `ruler_utils.SEQ_LENGTHS` 는 yaml 3곳과 일치를 강제한다(어긋나면 예외).
  **모듈 전역에 실행 중 값을 쌓는 방식은 쓸 수 없다** — lm_eval 의 `!function` 로더가
  모듈을 파일 경로로 따로 로드해 인스턴스가 갈린다(이 사실도 이번에 실측으로 확인).
- **교훈**: 벤치 설정을 모델 종류에 맞추지 않으면 모델이 아니라 설정을 측정한다. 두 하니스가
  같은 모델에 모순된 값을 주면 **모델을 의심하기 전에 설정을 대조**한다.

## T1 벤치 태스크가 base 모델용이었다 — 추출 실패·avg@k 무효화 (2026-08-30 ✅)

체크포인트 종료 토큰 사고(아래 항목)를 고치는 과정에서 발견한 **별개 결함 3건**. 셋 다
lm_eval 내장 태스크를 채팅·추론 모델에 그대로 쓴 데서 나왔다.

- **① GPQA `strict-match` 는 원리적으로 0점이 나온다.** 내장 정규식이
  `(?<=The answer is )(.*)` 인데 채팅 모델은 그 문구로 답하지 않는다. 실측 0.0(iter300)·
  0.0101(baseline). `flexible-extract` 23.2% 도 4지선다 무작위(25%) 수준이라 사실상 미측정.
- **② MMLU-Pro 가 5-shot 이었다.** 내장 `_default_template_yaml` 은 `num_fewshot: 5` +
  단일 패턴 `answer is \(?([ABCDEFGHIJ])\)?`. 프론티어는 0-shot CoT + 답 형식 지시가
  표준이다(OpenAI simple-evals: few-shot 은 base 모델 유물). 흥미롭게도 그 파일에는
  프론티어 형식 정규식이 **주석 처리된 채** 남아 있다.
- **③ `avg@16` 이 실제로는 avg@1 이었다.** `repeats: k` 는 동일 Instance 를 k번 복제해
  `resps` 에 k개를 쌓는다(`evaluator.py`: `cloned_reqs.extend([req] * req.repeats)`).
  그런데 `filter_list` 를 지정하지 않으면 lm_eval 이 기본 `take_first` 를 꽂아
  **1개만 남긴다**(`api/task.py:771` 의 TODO 주석이 이 동작을 인정한다). 구
  `aime25_avg16`·`hmmt_feb_2025_avg16` 은 16배 연산을 쓰고 첫 샘플만 채점했다.
  덤으로 요청 바디의 `seed` 가 1234 로 고정돼 있어, 필터를 고쳐도 k개가 동일 표본이 될
  뻔했다 — 태스크 yaml 에 `seed: null` 을 넣어 해소.

- **수정 (2026-08-30)**: `eval_sft/tasks/*_aa.yaml` 5종 + `aa_utils.py` 로 재작성.
  0-shot, AA 표준 프롬프트(보기 개수 가변 대응), 8단 폴백 추출(마지막 매치),
  `take_first_k` 로 avg@k 실제 작동, `seed: null`. 상세는
  `docs/SFT_BENCHMARKS.md` §3.6.
- **부수 발견**: 추출은 반드시 `</think>` **이후 구간**에서 해야 한다. 사고 구간에는 보기
  나열·중간 후보·자기부정이 가득해 전문에 정규식을 걸면 사고 중의 잘못된 후보를 집는다.
  `aa_utils.split_think` 가 처리하고, 사고를 닫았는지를 `think_closed` 지표로 함께 보고한다.
- **교훈**: 하니스의 기본 태스크는 그 하니스가 만들어진 시점의 모델을 가정한다. 채팅·추론
  모델을 base 모델용 태스크로 재면 "모델이 약하다"로 읽히는 측정 실패가 나온다. 점수 옆에
  `no_answer` 를 항상 함께 낸다(NeMo-Skills 규약) — 그 열이 있었다면 34% 를 보고 즉시 멈췄다.

## SFT 체크포인트에 종료 토큰이 없어 벤치 전량 무효 (2026-08-30 ✅ 원인규명 / 재측정 대기)

- **증상**: iter300 SFT ckpt와 LC-B iter320 베이스라인의 T1~T4 벤치 전 항목이 비정상.
  MMLU-Pro 추출 실패 34%/42%, GPQA 54%/61%, AIME25·HMMT 0/30, SimpleQA 1000건 중 805건
  `not_attempted`, LogicKor 1.32/10, SWE 0/20, Terminal 0/10. "모델이 약하다"로 읽히지만
  **측정 자체가 성립하지 않았다**.
- **원인 (변환 산출물 결함 2건 동시 작용)**:
  1. Megatron→HF 변환기가 **`generation_config.json`을 만들지 않는다.** `config.json`의
     `eos_token_id`는 사전학습 관례인 `0`(`<|endoftext|>`)에 머무는데, SFT 챗 템플릿이
     턴을 끝내는 토큰은 `<|im_end|>`(id **3**) — eos 집합에 없다. 서버가 턴 종료를
     인식하지 못해 **max_tokens까지 계속 생성**한다.
  2. `</think>`(id 15)가 `special=True`라 vLLM 기본 `skip_special_tokens=True`가
     출력에서 **삭제**한다. 하니스는 thinking 종료를 영원히 관측할 수 없다.
- **증거 (전수 집계)**: `mmlu_pro_engineering` 969건 중 `</think>` 포함 **0건**
  (추출 성공한 230건조차 0). `aime25` 30건 중 `boxed` 포함 **0건**, 응답 길이 중앙값
  75,814자. 추출 실패 응답 중앙값 70,170자 vs 성공 16,351자 — 답을 쓰기 전에 잘렸다.
- **수정본 프로브 결과 (결정적)**: `eos=[3,0]` + think/tool 태그 `special=False`로
  고친 임시 ckpt를 `--reasoning-parser deepseek_r1`로 서빙:
  - 쉬운 질문: `finish=stop`, tokens=161, content=`"17 multiplied by 23 is 391."` → **설정 결함은 해소**
  - 어려운 질문(AIME): `finish=length`, tokens=16384, reasoning 48,226자, **content 0자**
  → 즉 **iter300 모델은 어려운 문제에서 `</think>`를 닫지 못한다.** 설정 결함과
  별개인 **모델 미성숙**(300/2448 = 12% 학습)이며, 설정을 고쳐도 그 자체로는 점수가 나오지 않는다.
- **별건 결함 — RULER 출력 예산 128토큰**: `SFT_BENCHMARKS.md`가 정한 128토큰을 reasoning
  모델은 서두 설명에 전부 소진해 needle을 못 쓴다(65536 구간 6~25%). 같은 모델이 LC-B
  자체 NIAH 하니스에서는 4k~131k **200/200**이었다 — 모순의 원인은 모델이 아니라 태스크 설정.
  덤으로 샘플 0개인 4096 구간에 센티넬 `-1.0`이 결과에 기록된다.
- **처리 (2026-08-30)**: 무효 수치·샘플 전량 삭제(1.4GB), `results/TRACKING.md` 백지화,
  wandb `alpha-post-eval` 로컬 run 2개 삭제, 미검증 미커밋 튜닝 되돌림, `tools/` 임시
  프로브·진단 스크립트 42종 제거.
- **재발 방지 (착수 전 필수)**:
  1. **변환 게이트**: HF 변환 산출물에 `generation_config.json`이 있고 `eos_token_id`가
     챗 템플릿의 턴 종료 토큰을 포함하는지 검사. 불일치 시 exit 1.
  2. **서빙 스모크 게이트**: 벤치 투입 전 1건 생성으로 `finish_reason=stop`,
     `</think>` 관측, `content` 비어있지 않음을 확인. 통과 전 수치 기록 금지.
  3. **관측 가능성**: `</think>`·`<tool_call>` 등 하니스가 파싱해야 하는 태그는
     `special=False`여야 출력에 살아남는다.
  4. **길이 정합**: `--max-model-len`은 `max_gen_toks` + 프롬프트를 담아야 한다
     (32768 모델길이에 32768 생성예산은 성립 불가).
  5. **RULER 예산**: reasoning 모델에 128토큰 예산은 무효. 태스크 재설계 필요.
- **작업 방식 교훈**: 원인 확정 전에 설정을 여러 개 동시에 바꿔(길이·온도·파서·모델
  디렉토리) 무엇이 무엇을 고쳤는지 분리 불가능해졌다. 또 드라이버를 포그라운드로 띄워
  세션이 끊길 때마다 검증이 소실됐다 — 장시간 프로브·러너는 `setsid` + NFS 로그로 분리한다.

## 합성 원문의 리터럴 EOD → THD+CP 문서 분열 크래시 (2026-08-23 ✅)

- **증상**: LC-A 본 학습 iter 170에서 THD+CP 가드가
  `All per-sequence lengths in cu_seqlens must be divisible by 2*cp_size` ValueError로
  정지. 세그먼트 목록에 %16 비정렬 값들이 섞임(예: 608토큰 문서가 [318, 35, 255]로
  분열 — 부분합이 원래 문서 길이).
- **원인 사슬**: 합성 데이터 **원문에 리터럴 `<|endoftext|>` 문자열**이 남아 있으면
  HF tokenizers가 added special token을 본문에서도 매칭해 **id 0이 문서 중간에** 박힘
  → `--reset-position-ids`가 EOD마다 position을 리셋하므로 문서가 격자(%16) 비정렬
  위치에서 분열 → THD+CP a2a의 %2cp 요건 위반 → 가드 정지(설계대로 crash > silent).
  pad16 재패킹의 per-doc 검증은 문서 단위라 내부 EOD를 못 보고, %16 표본검사(50
  bins/set)는 ~0.05% 희소 오염을 확률적으로 놓침.
- **오염 실측** (2000-bin 표본, 2026-08-23): longblocks 1/2008 · code_review 1/2058 ·
  rewriting 1/2064. specialized 신규 15종 포함 나머지 29멤버 청정. longblocks 건은
  "내부 EOD 금지 불변량"(`docs/LC_DATASETS.md` §5.1) 위반 사례.
- **수리 (런타임, 2차 방어)**: `megatron_patch/data/utils.py::snap_cu_seqlens_to_grid`
  — pad16 데이터의 **진짜 문서 경계는 전부 %16 위치**(bestfit 적재 구조 보장)이므로
  격자 밖 경계 = 가짜 경계로 판정·제거해 분열된 문서를 복원. 내부 EOD 토큰은
  `--eod-mask-loss`가 이미 loss에서 제외하므로 의미론 무해. **CP>1 경로 전용**(SFT류
  임의-경계 packing에 적용하면 진짜 경계를 파괴함 — helper의 게이트 유지 필수).
  회귀 테스트: `tests/test_gdn_varlen_thd.py::test_snap_cu_seqlens_*` (실사고 패턴).
- **예방 (데이터 측, 1차 방어)**: 새 packed 산출물은 학습 투입 전
  `toolkits/pretrain_data_preprocessing/scan_internal_eod.py`로 스캔(blend yaml 단위
  가능, 오염 시 exit 1). 근본 예방은 토크나이즈 단계에서 원문 내 special-token
  리터럴을 이스케이프/제거하는 것 — 128k 합성 파이프라인(러너북 §4)에 필수 반영.
- **부수 함정**: 감시 스크립트의 `pgrep -f pretrain_alpha`가 **자기 자신의 명령줄을
  매칭**해 사망 감지가 무음 실패 — 감시 패턴은 `[p]retrain_alpha.py`처럼 자기제외
  형태로 쓸 것.

## THD+CP에서 MoE 크래시로 위장한 rope 잠복버그 (2026-08-22 ✅)

- **증상**: THD+CP≥2 첫 스텝에서 MoE dispatcher가 `Split sizes doesn't match total
  dim 0 size`로 크래시. CP=1 THD와 dense CP는 정상이라 스티치 배선을 의심하기 쉬움.
- **원인 사슬**: `mamba_model.forward`가 gpt_model과 달리 rope에 packed_seq_params를
  안 넘김 → `RotaryEmbedding.forward`가 dense CP 관례로 테이블을 rank별 zigzag
  **사전 슬라이스**(로컬 길이) → THD rope 함수는 풀 테이블을 받아 자체 CP 슬라이싱
  하므로 fused rope 커널이 테이블 밖을 읽음 → q/k 일부 NaN → 3레이어 뒤 MoE
  라우터에서 **CUDA topk가 NaN에 중복 인덱스를 반환** → routing_map 행합 < topk →
  split 불일치. 증상 지점과 원인 지점이 레이어 3개 + 모듈 2개 떨어져 있었음.
- **수정**: mamba_model에 upstream gpt_model 미러(packed_seq_params 전달, 서브모듈
  직접 수정 — 루트 CLAUDE.md 비-upstream #5) + alpha 래퍼 전달 + helper max_seqlen
  int화. 부수 발견 2건(core utils NameError·None 가드)도 동시 수정.
- **교훈**: ① NaN이 낀 CUDA topk는 중복 인덱스라는 미정의 동작을 낳는다 — MoE
  라우팅 크래시를 보면 hidden NaN부터 의심. ② mock 데이터는 THD+CP 검증에 쓸 수
  없다 — 무작위 토큰에 EOD(id 0)가 섞여 %16 미정렬 세그먼트가 생기고
  `resolve_cu_seqlens` 가드가 거부한다(정상 동작). 실데이터나 단일 세그먼트로 검증.
- **규명 전 과정**: [`docs/gdn_cp_port.md`](docs/gdn_cp_port.md) 분석노트 3.

## DiLoCo: 짝/홀 샤딩 × blend 인덱스 aliasing → 거울상 loss 시소 (2026-08-17 ✅)

- **증상**: P2b 스위치(iter 18k) 이후 두 노드의 lm/seq-bal loss가 거울상 진동
  (상관 −0.95/−0.996, 주기 ~336 iter, 노드당 ±0.032). P2에서는 상수 오프셋(+0.038)이었음.
- **원인**: 로깅 loss는 노드-로컬(자기 샤드)인데, `DILOCO_DATA_SHARD=1`의 짝/홀
  분할이 BlendedDataset의 **셔플 없는 결정론적** 소스 수열과 alias. 6자리 가중치
  합의 잔차(P2b +1e-6 / P3 −1e-6)를 `normalize()`가 나누며 전 가중치를 밀어
  블렌드 패턴이 짝/홀 격자 위를 세차운동 → 패리티 주기 반전 = 시소, 주기
  2/|Σw−1| 샘플 = 325.5 iter @ GBS 3072×2 (반사실로 인과 확정: 잔차 3e-6 → 주기 1/3).
  구성 델타 26변수만으로 실측 gap R²=0.83(raw)/0.98(smoothed) 설명, ±1 iter
  시프트 시 붕괴. 페어 합산 구성은 매 iter 정확(≤1.4샘플) — 학습 무결성은 유지,
  단 같은 부호 편향이 ~160 iter(5×H) 지속되어 노드-로컬 optimizer 상태에 적분됨
  (expert-bias 발산의 압력이었던 그 비대칭).
- **수정**: `DILOCO_SHARD_BLOCK=3072`(=GBS) 블록-순환 매핑 (`diloco_patch.py`).
  노드별 배치가 블렌드 수열의 연속 3072샘플이 되어 구성 오차 ≤2샘플, 오프셋·시소
  동시 소멸. 전환은 consumed % 3072 == 0에서만(assert), pair 간 env 일치 assert.
  부작용 없음(인덱스 산술만, 페어 합산 스트림 불변). 다음 재시작에서 활성화.
- **전체 기록**: [`study/mirror_loss_aliasing.md`](study/mirror_loss_aliasing.md)
  (재현: `study/mirror_loss_repro.py`, 검증: `tests/test_diloco_shard_view.py`).

## DiLoCo: MoE expert-bias가 outer 동기화에서 제외 → 노드별 발산 (2026-08-11 ✅)

- **증상**: aux-loss-free `expert_bias`(buffer)는 wire 집합(`named_parameters()`)에 미포함 —
  각 노드가 자기 샤드 통계로 독립 갱신. 짝/홀 샤딩의 구성 비대칭이 지속 압력이 되어
  체크포인트 분석(10k~20k)에서 지속 코어 24 expert(layer 2/20 집중), 최대 격차 0.118
  (선택 척도의 ~12%)까지 성장. 훈련 loss 피해는 검출한계(<0.005) 이하였으나 **최종 배포
  모델의 `e_score_correction_bias`가 단일 샤드 균형으로 오염**되는 실해 + P3에서 악화 전망.
- **수정** (`diloco_patch.py`, `DILOCO_BIAS_SYNC=1` 기본): 매 step `tokens_per_expert`를
  **전용 Gloo pair group**(포트+100 — τ-오버랩 wire 스레드와 group 공유 금지)으로 pair
  **SUM** → 양 노드가 결합 배치 통계로 동일 갱신 → bias 영구 bit-identical (레퍼런스
  sync-DP 의미론; bias 벡터의 평균/전송 없음). 매 outer sync에 `bias in sync` checksum 로그.
  부수: fresh 경로에 params-identical 시 broadcast/`reload_model_params()` 생략 가드
  (fp32 master 보존).
- **적용**: iter 20,000에서 node0 체크포인트를 양 노드 채택(+outer 신규 초기화) 후 재시작.
- **함정 2건**: ① `import megatron.core.distributed.finalize_model_grads as _F`는 패키지
  `__init__`의 속성 재바인딩 탓에 **함수**를 반환 — `hasattr()` 가드가 패치를 무음 스킵했다.
  `importlib.import_module()`로 진짜 모듈을 잡을 것 (monkey-patch에 soft-fail 가드 금지).
  ② pair collective는 양 노드 **대칭 호출** 필수 — env 게이트 계측은 `EXTRA_ENV`로 전달
  (launch_diloco의 ENVV는 화이트리스트라 임의 env를 node1에 안 넘긴다).
- **전체 기록**: [`docs/STAGE2_CURRICULUM_LOG.md`](docs/STAGE2_CURRICULUM_LOG.md) §2.4
  (blend 커리큘럼·샤드 aliasing·resume 정밀 검증 포함 단일 진입점).

## NGC 25.03에서 QK-Clip fused-attn 크래시 — cuDNN 9.8에 max_logit 엔진 없음 (2026-07-13 ✅)

- **증상**: 학습 첫 step에서 `cuDNN Error: No valid engine configs for Matmul_MUL_GEN_INDEX_..._Matmul_`.
  기본 attention은 통과하는데 **`return_max_logit=True`(QK-Clip 경로)만 실패**.
- **원인**: TE 2.9의 max-logit fused-attn 그래프 엔진이 cuDNN 9.11+에만 존재. NGC 25.03은
  9.8. TE의 backend 선택기는 이 케이스에 cudnn 버전 게이트가 없어(utils.py — thd/fp8만
  거름) 폴백 대신 런타임 크래시.
- **수정**: `pip install --no-deps nvidia-cudnn-cu12==9.24.0.43` (**--no-deps 필수** —
  의존성으로 딸려오는 cublas 12.9가 환경을 깨뜨림) + train.sh가 pip cuDNN 발견 시 전체
  서브라이브러리를 **LD_PRELOAD** (LD_LIBRARY_PATH만으로는 TE RUNPATH 탓에 9.24/9.8이
  섞여 `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`). 멀티노드 셋업 스크립트가 자동 설치.

## Optimizer-state resume이 첫 collective에서 크래시 — NCCL comm-init OOM (2026-07-15 ✅)

- **증상**: `--load`로 optimizer state까지 실은 resume이 `Failed to CUDA calloc async N bytes`
  (N은 4~608B로 미미)로 사망. fresh 학습은 정상. **DiLoCo 무관 — 순수 Megatron도 재현.**
- **원인**: NCCL 2.25 기본 64채널의 comm당 GPU 버퍼가 크고, Megatron은 comm이 많다.
  fresh는 optim state가 첫 step 이후 생성되어 comm 초기화가 저메모리 구간에서 일어나지만,
  resume은 로드된 state + 비동기 in-flight 할당 위에서 지연 초기화 comm들이 일제히 버퍼를
  요구 → NCCL 'out of memory'. 판별 근거: `CUDA_LAUNCH_BLOCKING=1`이면 통과 + NCCL_DEBUG.
- **수정**: **`NCCL_MAX_NCHANNELS=16`** (train.sh 기본값) — comm 버퍼 4× 절감, step time
  무손실(60.7s vs 61.1s). 검증: 3개 독립 resume의 iter-13 loss **bit-identical**.

## DiLoCo: 저장 시점에 pending sync 살아있으면 저장 크래시 (2026-07-14 ✅)

- **증상**: `--exit-interval`(또는 save-interval)이 H의 배수일 때 마지막 sync가 시작만 된 채
  (τ>0, 미적용) torch_dist 저장의 NCCL gather가 `unhandled cuda error`로 사망. 100% 재현.
- **수정**: diloco_patch의 save 훅이 pending sync를 **join+apply 후 저장**(드레인). 부수
  효과로 체크포인트가 일관된 outer 상태를 담음. 같은 시기 수정: τ-apply의 파라미터별 GPU
  임시버퍼(→allocator 단편화)를 per-dtype 영구 scratch 버퍼로 대체.

## 2노드 환경 셋업: NGC 25.03의 PIP_CONSTRAINT·TE 서브모듈 순서 (2026-07-13 ✅)

- 기존 `setup_pai_megatron_env.sh`가 이 이미지에서 3중 드리프트로 연쇄 실패: ① TE upstream
  main의 서브모듈 구성 변경(checkout *후* `git submodule update` 필요), ② 이미지 전역
  `PIP_CONSTRAINT=/etc/pip/constraint.txt`가 명시 핀과 충돌(importlib-metadata/packaging/
  **transformer-engine 자체**), ③ mamba-ssm/fla 미고정 설치가 현 PyPI 최신(triton 3.7
  강제)을 끌어옴. → **`setup_pai_megatron_env_multinode.sh`** (repo 부모 디렉토리) 사용:
  선별적 `env -u PIP_CONSTRAINT`, canonical pin(mamba v2.2.6.post3 git 빌드, fla==0.4.1),
  `NVTE_CUDA_ARCHS=90`(빌드 30분+→3분), TE wheel을 workspace에 보존(타 노드 재빌드 생략),
  cuDNN 9.24 설치 포함. 원본 스크립트는 참조용 무수정 보존.

## A100 단일-GPU 벤치 환경: modelopt 몽키패치·typing_extensions·--multi_gpu 3중 이슈 (2026-07-22 ✅)

- A100(Ampere) 박스에서 evaluate.sh 경로만 밟는 3중 이슈 — 셋 다 H100 멀티노드에서는
  미노출. ① NGC 번들 `nvidia-modelopt`가 import 시 transformers `from_pretrained`를
  구버전 시그니처로 몽키패치 → 핀 4.57.0.dev0과 충돌(TypeError). 유입은 Megatron
  `checkpointing.py`의 guarded import → **modelopt 제거**(alpha 미사용; `sudo -E env -u
  PIP_CONSTRAINT /usr/local/bin/pip uninstall --break-system-packages ...` — 일반
  `sudo pip`은 Debian pip+PEP 668에 막힘). ② lm-eval 0.4.12가 `typing_extensions>=4.13`
  (`TypedDict extra_items`) 요구 → 업그레이드. ③ `run_benchmarks.sh`의
  `accelerate launch --multi_gpu` 하드코딩이 NUM_GPUS=1에서 즉사하는데 **exit 0 무음
  실패** → NUM_GPUS>1 조건부로 패치. 성공 판정은 `eval_results/results_*.json` 존재로.
- 환경 셋업은 `setup_pai_megatron_env_A100_v2.sh`(repo 부모, NVTE_CUDA_ARCHS=80 —
  workspace 루트의 TE wheel은 **sm_90 전용이라 A100 재사용 금지**, 아치별 캐시
  `te_wheels_sm80/` 사용). DiLoCo per-node ckpt 벤치는 unshard 보정 불필요(순수 Megatron
  로드가 outer state 무시). 상세·stage2 벤치 기록(node0 iter7120·iter10000):
  [`docs/A100_SINGLE_GPU_EVAL.md`](docs/A100_SINGLE_GPU_EVAL.md)
- (07-27) ①②는 A100_v2 스크립트 **Step 13.5로 내장** — 수동 조치 불필요. 세션
  재생성(재부팅) 시 전체 복원 runbook: repo 부모의 `RESTORE_AFTER_REBOOT.md`
  (환경 = 스크립트 1회 + TE wheel 캐시, Gemma 서빙 = `reboot_restore/restore_gemma_serving.sh`).

## MG↔HF weight 검증이 `expert_bias` 1개에서 지속 실패 — fp32 router bias 다운캐스트 (2026-06-15 ✅)

- **증상**: `evaluate.sh` Stage 2 (`validate_mg_hf_full.py`)가 `✗ WEIGHT MISMATCH DETECTED`로 exit 1. 요약은 **`14180/14181 matched` — 정확히 1개 비교만 실패**. coverage 갭(unchecked 24 / phantom 48)은 전부 filtered=0이라 실패 원인이 아니고, 실패한 1개는 layer 0의 `router.expert_bias ↔ gate.e_score_correction_bias`. iter가 진행될수록 **악화**(bias가 단조 누적).
- **근본 원인 (컨버터 아님, 검증 로드의 dtype 평탄화)**:
  - Megatron은 `router.expert_bias`를 **의도적으로 fp32로 유지**(`core/.../moe/router.py::_maintain_float32_expert_bias`, "to avoid routing errors when updating the expert_bias"). 컨버터는 저장 dtype을 **MG 소스 텐서에서 물려받으므로**(`m2h_synchronizer.py`의 `_local_params`가 소스 보관) → **이미 fp32로 정확히 저장**됨. 나머지 가중치는 bf16. (safetensors `stored dtype` 직접 확인: bias=`float32`, weights=`bfloat16`.)
  - 버그는 검증기의 HF **로드**에 있었음: `load_hf_model()`이 `from_pretrained(torch_dtype=torch.bfloat16)`로 **모든 텐서를 로드 시점에 bf16으로 평탄화** → 디스크의 fp32 bias가 bf16으로 다운캐스트.
  - layer 0 bias 크기 ~4.5는 bf16 binade [4,8)에 속해 ulp/2 = **0.0156 > 0.01**(고정 절대 임계). → MG(fp32) vs HF(로드시 bf16)에서 max_diff ≈ 0.0156으로 1개 실패. 다른 23개 레이어는 bias ≤ 1.82(binade [1,2), 오차 0.0039)라 통과. 디스크 artifact엔 없는 **유령 오차**.
- **왜 expert_bias만**: bf16 ulp/2가 0.01을 넘으려면 값이 ≥ 4.0이어야 하는데, fp32-on-MG이면서 그만큼 큰 텐서는 aux-loss-free `expert_bias`(누적되어 layer 0에서 ~4.5)뿐. gate.weight 등은 bf16-vs-bf16 정확 복사라 max_diff 0.
- **수정 방향 (충실 변환 + 엄격 검증; 오차 완화 거부)**: bias를 **end-to-end fp32**로 유지해서 fp32-vs-fp32 정확 비교가 되게 함. DSV3 공식 HF와 동일한 형태.
  - `hf_model/modeling_alpha.py`: `e_score_correction_bias`를 `register_buffer` → **fp32 `nn.Parameter(requires_grad=False)`**, `AlphaPreTrainedModel`에 **`_keep_in_fp32_modules_strict = ["e_score_correction_bias"]`** 추가.
  - `validate_mg_hf_full.py`: 허용오차를 **원래의 엄격한 기준 그대로** 유지(`max_diff < threshold and cos_sim > 0.999`). (변경 없음 — 완화 안 함.)
  - 컨버터: **변경 없음** (이미 fp32 저장). `gate.weight`도 bf16 그대로 — MG가 그것만 fp32로 유지하므로 expert_bias만 fp32가 정확한 "MG 동일".
- **놓치기 쉬운 함정 2개**:
  1. **`_keep_in_fp32_modules`(strict 아님)는 fp16에서만 발동** — bf16 로드는 안 지킴. bf16까지 커버하려면 **`_keep_in_fp32_modules_strict`** 필요 (transformers 4.57 `modeling_utils` 주석/분기 확인).
  2. **이 플래그는 `named_parameters()`만 보호, 버퍼는 미보호** (`_load_state_dict_into_meta_model`이 params만 순회). 그래서 buffer→`nn.Parameter` 전환이 필수. toy `PreTrainedModel`로 직접 검증: 동일 이름이라도 **Parameter는 fp32 유지 / Buffer는 bf16 다운캐스트**.
- **회귀 가드** (`tests/test_alpha_pipeline_config.py`, +3 → 12개):
  - `test_modeling_alpha_router_bias_is_fp32_parameter` — Parameter 형태 + strict 플래그 등록 확인.
  - `test_keep_in_fp32_modules_strict_protects_param_not_buffer` — transformers의 param-vs-buffer 동작을 toy 모델로 잠금(버전업으로 깨지면 멀티시간 eval 전에 차단).
  - `test_compare_tensors_is_strict` — fp32-vs-fp32 정확 통과 / +0.5 drift 실패 / ~4.5 fp32의 bf16 다운캐스트는 **엄격 기준에서 실패**(= bias를 fp32로 두는 이유).
- **적용**: `bash evaluate.sh <run> --gpus 4` 재실행 시 재변환이 새 `modeling_alpha.py`를 HF 디렉토리로 복사(`run_convert.sh:190 cp .../hf_model/*.py`)하고 Stage 2가 14181/14181 통과. **기존 HF 디렉토리는 bias가 이미 fp32 저장**이므로 재변환 없이 `cp examples/alpha/hf_model/*.py <hf_dir>/` 후 `validate.sh`만 재실행해도 통과(로드되는 modeling 클래스만 갱신).
- **부수 효과 (긍정)**: 추론(`forward_sanity.py`·lm-eval·HF serving)도 이제 bias를 fp32로 로드 → 학습-시점 라우팅과 더 충실하게 일치(selection은 원래 fp32 계산이라 bias만 bf16이면 borderline expert가 미세하게 흔들렸음).

## HF `AlphaRMSNorm`이 zero-centered(1p) → 모든 벤치마크 random (silent, 2026-05-26 ✅)

- **증상**: v2 체크포인트를 `evaluate.sh`로 끝까지 돌리면 **모든 게이트(weight 검증·config 대조·tokenizer)는 통과**하는데 Stage 3 벤치마크만 random (ARC-easy 정확히 25%).
- **근본 원인**: `hf_model/modeling_alpha.py::AlphaRMSNorm`가 Qwen3-Next에서 물려받은 **zero-centered `x_norm * (1 + γ)`** 를 적용. 그러나 Alpha v2는 Megatron을 **`apply-layernorm-1p` OFF(표준 `x_norm * γ`, γ≈1)** 로 학습 (checkpoint `common.pt`: `apply_layernorm_1p=False`, 저장 γ mean≈0.69~1.5). → 모든 norm(input/post/q/k/final, 24레이어)이 **~1.7~2.5× 과증폭** → 잔차 스트림 누적 왜곡 → near-uniform logit (perplexity≈vocab). NaN 아님(scale 오류). 같은 파일 `AlphaRMSNormGated`는 이미 표준(`*γ`)이라 GDN norm은 정상이었음 → 두 norm 클래스가 불일치했던 것.
- **수정**: `AlphaRMSNorm`를 표준으로 — forward `output * self.weight.float()` (1+ 제거), init `torch.ones`. `AlphaRMSNormGated`와 일관.
- **실측 검증** (iter_0010000, 단일 모델 로드 monkeypatch): `(1+γ)` ppl=295,440 / greedy `'…andNV and) From Form'` → `γ` ppl=8.84 / greedy `'The capital of France is Paris.'`. ARC-easy 0-shot(100): **25% → acc 0.73 / acc_norm 0.76**.
- **왜 weight 검증이 못 잡았나 (교훈)**: `validate_mg_hf_full.py`는 **weight tensor만** 비교하고 forward를 안 한다. converter가 γ를 그대로 복사 → 검증은 MG γ==HF weight 통과. 차이는 forward의 `1+` 에서만 발생 → 사각지대. 게다가 attention 비교는 converter의 reshape를 복제(`Reference: m2h_synchronizer.py`)해서 해석 오류를 양쪽이 공유.
- **회귀 가드**: ① `examples/alpha/forward_sanity.py` — 변환 HF 모델 perplexity 게이트(임계 100; random≈vocab). ② `evaluate.sh` **Stage 2.5**로 편입(weight 검증 후·벤치마크 전). ③ `tests/test_alpha_pipeline_config.py::test_modeling_alpha_rmsnorm_is_standard_not_1p`. **기존 변환 산출물은 재변환 불필요** — weight는 정상이므로 `hf_model/modeling_alpha.py`만 HF 디렉토리에 재복사하면 됨.
- **부수**: `toolkits/distributed_checkpoints_convertor/impl/alpha/m2h_synchronizer.py:246` bias 경로 `linear_qkv`→`linear_qgkv` 오타 정리(dormant: `attention_bias=False`). bias 활성화 시 q bias에 weight-path의 gate-interleave transpose 필요(주석 추가).

## Stage 1 재개 (10k) + 처리량 최적화 — `stage1_resume.yaml` (2026-05-26)

컴퓨팅 세션 장애로 Stage 1 run(`outputs/alpha_baseline_48L_stage1_20260512_170157`)이 중단.
실제로는 iter ~16k까지 갔으나 디스크엔 **iter 10000 체크포인트만** 존재(원인: `save-interval: 10000`).
iter 10k에서 재개하는 신규 프리셋 `configs/training/stage1_resume.yaml` 추가
(`stage1.yaml`은 from-scratch 레시피로 보존). **실제 적용된 변경은 4개**(아래 표).
처리량 최적화로 시도했던 `moe-shared-expert-overlap`은 throughput 회귀를 일으켜 되돌렸고,
`micro-batch-size 3→6`·`recompute += core_attn`은 계획만 하고 커밋되지 않았다(아래 "⚠️ 처리량 회귀" 참조).

```bash
bash train.sh baseline_48L stage1_resume stage1_v5_blend
```

| 변경 | 값 | 이유 |
|---|---|---|
| `load` + `no-load-optim: true` | 10k ckpt | 재개. ckpt에 옵티마이저 상태 없음(no-save-optim)이라 no-load-optim 필수 |
| **`finetune` 미설정** | — | consumed_train_samples(15.36M) 보존 → **데이터 위치** 연속. (Stage *전환*에만 finetune) |
| **LR 스케줄 재구성** | warmup 200it / decay 94.5M samples | **no-save-optim ckpt엔 스케줄러 상태가 없어 재개 시 scheduler num_steps가 0으로 리셋**(consumed_samples로 재시드하는 코드 없음 — `checkpointing.py:848,1708`). stage1.yaml 스케줄 그대로 쓰면 풀 1907it warmup + cooldown 미발동. 그래서 *남은 구간*(61,526it=94.5M samples) 기준으로 짧은 re-warmup(200it)+WSD cooldown(6,358it) 재정의 — stage2_2.yaml 패턴과 동일 |
| **LR 상향** 2e-4 → **2.5e-4** (min-lr 2e-5 → 2.5e-5) | peak 2.5e-4 / min 2.5e-5 | **√k 배치 스케일링 ratio로 선정.** 참조점 GBS=256 → lr=1e-4 (Stage 2-2 레시피). 현재 GBS=1536 → 배율 **k = 1536/256 = 6** → 제곱근 스케일 lr = **√6 × 1e-4 ≈ 2.449e-4**, 이를 깔끔히 **2.5e-4**로 반올림(+2%). 이전 2e-4는 이 ratio를 과소 적용한 값. min-lr도 동일 배율로 올려 **WSD 10:1 감쇠 비율** 보존. **k는 *글로벌* 배치 비율**(GBS 불변=1536)이지 micro-batch-size(6)와 무관 — 숫자 우연 일치 주의. 원본 run 대비 +25% 점프는 200it re-warmup이 0→2.5e-4 램프로 흡수 |
| `save-interval`·`eval-interval` 10000→5000 | — | 이번 손실의 직접 원인. weights-only 저장이라 빈번 저장 부담 적음 |

## ⚠️ 처리량 회귀 (2026-05-26) — `moe-shared-expert-overlap`이 범인

stage1_resume이 throughput ~50으로 stage1 대비 급락. 원인은 시도했던 `moe-shared-expert-overlap: true`
한 줄이었음(되돌림). 이 플래그는 shared expert를 **별도 CUDA 스트림**에 올려
(`shared_experts.py:120,160,275`) routed-expert 디스패치 A2A와 겹치려 하지만, `train.sh:71`이
`CUDA_DEVICE_MAX_CONNECTIONS=1`을 하드코딩 → **단일 하드웨어 큐**에서 두 스트림이 **직렬화**됨:
겹침 이득은 0인데 cross-stream 이벤트 배리어 비용만 **24개 MoE 레이어 × 매 스텝** 누적.
(YAML에 적혀 있던 "CUDA_DEVICE_MAX_CONNECTIONS=1 유지 → low risk"는 정반대였다.)
- **`=1`이 강제되는 건 TP>1 또는 CP>1일 때뿐**(`arguments.py:1005,1029`). alpha는 TP=1/CP=1이라
  `=1`은 표준 레시피에서 복사된 관습이지 정렬상 필요조건이 아님. 이 플래그를 *실제로* 쓰려면
  `CUDA_DEVICE_MAX_CONNECTIONS`를 8~32로 올리고 mock tokens/sec A/B로 순이득을 확인해야 함.
- **`micro-batch-size 3→6` / `recompute += core_attn`은 끝내 커밋되지 않았다.** stage1_resume의
  실제 값은 여전히 MBS=3, `recompute-modules: "layernorm moe"`(= stage1.yaml과 동일). 둘이 함께
  계획됐으나(core_attn 재계산으로 MBS↑의 메모리 재원 확보) MBS가 3에 머물러 둘 다 무효. 향후
  실험으로 보류 — MBS=6 적용 시 OOM 점검 필요(OOM이면 4; num_microbatches=192/MBS).

**비채택(단일 노드 EP=8 + dist_muon 제약)**: `tp-comm-overlap`/`overlap-p2p-comm`(TP=1·PP=1 무효),
`use-distributed-optimizer`/`overlap-param-gather`(Muon 비호환), DeepEP(단일 노드 alltoall이 ~7% 빠름, 기측정),
`overlap-moe-expert-parallel-comm`·`moe-shared-expert-overlap`(둘 다 `CUDA_DEVICE_MAX_CONNECTIONS>1` 필요 — **다중 노드 전환 시 재검토**).
**검증**: ① mock 30iter 스모크(throughput이 stage1 수준으로 복귀 확인) → ② 실데이터 짧게(iter 10000 시작·consumed_samples 연속·loss 연속 확인).

## v2 평가 파이프라인 통합 + v1→v2 검증 (2026-05-26 ✅)

iter_0010000(첫 v2 체크포인트) 평가를 위해 수동 3단계(MG→HF 변환 → `validate.sh` → `run_benchmarks.sh`)를 `evaluate.sh` 하나로 통합. **근본 원인**: 같은 모델 config가 3곳(학습 YAML / 변환 `baseline_48L.sh` / `validate.sh` 하드코딩)에 중복되어 drift. 해결: **모든 변환/검증 args를 체크포인트 `common.pt`(ground truth)에서 유도**(`tools/alpha_config.py::load_config_from_checkpoint` + `emit-megatron-flags`), 병렬화(EP)만 런타임 GPU 수에서 유도. 전체 감사는 [`docs/V2_PIPELINE_VERIFICATION.md`](docs/V2_PIPELINE_VERIFICATION.md).

**파이프라인이 잡아낸 3개의 실제 버그** ("crash > silent corruption" — 올바른 플래그를 켜자마자 갭에서 멈춤):

1. **Config drift (stale 변환 경로)**: `scripts/alpha/configs/baseline_48L.sh`가 완전 v1(128 experts, head 32, kv 128, vocab 151936, pattern 49자, softmax)이고 `validate.sh`는 nested-YAML 파싱(flat v2에선 빈 값)이었음. 또 변환기가 **mamba 차원을 아예 안 넘겨** code default 64 ≠ 학습 128. → checkpoint 유도로 일괄 해결. **184 vs 192 / aux-free vs seq_aux_loss drift**도 여기 포함(아래 별도 항목).

2. **HF MoE 라우팅 부정합 (silent benchmark 오염)**: `hf_model/modeling_alpha.py::AlphaSparseMoeBlock`이 **plain softmax + 전역 top-k + no bias**로 라우팅 — 학습은 DSV3(sigmoid + 8×4 group-limited + aux-loss-free `expert_bias` + routed_scaling 2.5). 변환이 처음으로 MoE 단계까지 도달하자 `gate.e_score_correction_bias` 부재로 크래시 → 표면화. **수정**: `DeepseekV3TopkRouter`와 동일하게 재작성 + `gate.e_score_correction_bias` persistent buffer; `configuration_alpha.py`에 `scoring_func/n_group/topk_group/routed_scaling_factor`; `generate_hf_config`가 해당 키 emit; `verify_pipeline.py` Stage 1.5가 대조. (없었으면 모든 벤치마크 수치가 wrong-routing으로 무효)

3. **검증 coverage 갭**: 변환은 14133/14133 weight 일치로 성공했으나 `validate.sh`가 72 MG weight 미비교로 exit 1. `validate_mg_hf_full.py`(변환기와 독립 매핑)에 `router.expert_bias↔gate.e_score_correction_bias`, `shared_experts.gate_weight↔shared_expert_gate.weight` 비교 추가 + transient `router.local_tokens_per_expert` 제외 → exit 0.

**회귀 가드**: `tests/test_alpha_pipeline_config.py`(9개; config↔checkpoint 일치, HF config v2 필드, emit 누락, DSV3 라우팅 동작, stale 경로 grep). `configuration_alpha.py` 및 `test_alpha_tokenizer_eod.py`의 stale `num_experts=184`도 192로 정정(테스트 자체가 drift 피해자였음).

## EOS designation 통합: chat-end → pre-training EOD 분리 (2026-05-12 preflight ✅)
- **문제**: alpha v5 tokenizer가 처음에 `eos_token = <|im_end|>` (id 3)으로 설정되어 있었음. 이는 **chat-turn-end marker를 pre-training EOD로도 겸용**하는 것 — frontier convention (Qwen3 / Llama 3 / DSV3가 모두 두 의미를 분리)과 어긋남.
- **수정 (3개 파일 모두)**: `tokenizer_v5/{tokenizer_config.json, special_tokens_map.json, training_config.yaml}` 모두 `eos_token = <|endoftext|>` (id 0)으로 통일.
- **의미 분리**: pre-training은 `<|endoftext|>` (id 0)로 doc boundary, SFT 단계의 chat template은 `<|im_end|>` (id 3)을 turn boundary로. 미래 chat tuned model 출시 시 `generation_config.json`에 `eos_token_id = [3, 0]` override만 추가하면 됨 — tokenizer 파일은 안 건드림 (Qwen3 패턴과 동일).
- **`_AlphaTokenizer.eod` 자동 갱신**: 코드 변경 없음 — property가 이미 `tokenizer.eos_token_id`에 위임 (`megatron_patch/tokenizer/__init__.py:372`). config 한 줄 바꾸자 downstream 모두 자동으로 id 0 반환.
- **놓치기 쉬운 함정**: `tokenizer_config.json`만 바꾸면 HF AutoTokenizer는 OK (그게 우선 source). 하지만 `special_tokens_map.json`을 직접 읽는 도구 (vLLM, SGLang 일부 chat util)는 stale 상태 → silent breakage 가능. **세 파일 동기화 필수**.
- **회귀 테스트**: `tests/test_alpha_tokenizer_eod.py`의 `test_tokenizer_config_eos_is_endoftext` + `test_special_tokens_map_eos_is_endoftext` 가 향후 drift 차단.

## 데이터 EOD remap: id 3 → id 0 (2026-05-12 preflight ✅)
- **상황**: Stage 1 pre-tokenized `.bin` 파일들 (DCLM 443B + Korean Web 17B + FineWeb2-HQ 5.7B) 이 위 designation 변경 *전*의 tokenizer로 토큰화되어 모든 doc 끝에 `<|im_end|>` (id 3)를 갖고 있었음.
- **검증으로 발견된 단서**: id 3이 mid-document에 0 occurrences / 100% doc-end에만 존재 → **doc separator로만 사용된 게 empirically 확인됨**. 따라서 안전한 byte-level substitution 가능.
- **도구**: `toolkits/pretrain_data_preprocessing/remap_eod.py` — `IndexedDatasetBuilder` + numpy memmap으로 `.idx`의 `sequence_pointers + sequence_lengths`로부터 모든 doc-end 4 byte 위치를 계산 → in-place int32 substitution (3 → 0). `.idx` 변경 없음, 토큰 수 보존, fully reversible.
- **사용**:
  ```bash
  python toolkits/pretrain_data_preprocessing/remap_eod.py \
    --prefix /path/to/data_text_document \
    --old-eod 3 --new-eod 0 [--dry-run]
  ```
- **실측 wall time** (NFS-backed `.bin`):
  - FineWeb2-HQ 22 GB / 6.1M docs: **2.6 min**
  - Korean Web 64 GB / 15.7M docs: **10.8 min**
  - **DCLM 1.78 TB / 312M docs: 2h 55m** (NFS read-modify-write overhead dominates)
- **검증 protocol (자동 내장)**: pre-verify 200k samples 모두 `--old-eod` 보유 확인 → patch → post-verify 200k samples 모두 `--new-eod` 보유 확인 + 처음 100 / 마지막 100 docs boundary check.

## alpha_config.py Qwen3 default token IDs (silent bug, 2026-05-12 preflight ✅)
- **문제**: `examples/alpha/tools/alpha_config.py:48-49`의 `DEFAULT_BOS_TOKEN_ID = 151643`, `DEFAULT_EOS_TOKEN_ID = 151645`가 **Qwen3 vocab의 ID**. 이 파일은 `toolkits/distributed_checkpoints_convertor/scripts/alpha/run_*.sh`가 MG→HF 변환 시 `config.json` 생성 (`alpha_config.py generate-hf-config`)에 사용.
- **잠재 영향**: 변환된 HF model의 `config.json`이 `eos_token_id = 151645` 로 박힘 → 이는 alpha v5 vocab에서 *전혀 다른 BBPE 서브워드*. SGLang/vLLM serving 시 잘못된 stop token → 무한 generation 또는 엉뚱한 위치에서 멈춤. **학습은 영향 없지만 inference deployment 시점에 silent breakage**.
- **수정**: `DEFAULT_BOS_TOKEN_ID = None`, `DEFAULT_EOS_TOKEN_ID = 0`, `TokenConfig.pad_token_id = 1` default 추가. 즉 alpha v5 실제 IDs 반영.
- **회귀 테스트**: `test_alpha_config_token_defaults_are_alpha_v5`.

## configuration_alpha.py stale defaults (silent bug, 2026-05-12 preflight 2nd-pass ✅)
- **상황**: 1차 preflight (2026-05-12)에서 `examples/alpha/tools/alpha_config.py`의 stale Qwen3 token IDs를 잡은 후, F_decisions.md Item 12에 `examples/alpha/hf_model/configuration_alpha.py`의 stale defaults는 "Documented-cleanup (deferred — affects only no-kwargs instantiation)"로 라벨링하고 미수정. 2차 검증 (multi-month run 직전 final pass) 시 3개 parallel Explore agent 중 audit agent가 동일 패턴을 재발견 → 사용자가 promote-to-fix 결정.
- **놓친 이유**: 1차 검증의 audit-grep이 `151643/151645/im_end` 같은 토큰 IDs에 집중. `configuration_alpha.py`는 토큰 IDs가 아닌 *모델 구조 defaults* (vocab_size, intermediate_size, num_experts, ...)를 갖고 있어서 그 grep에서 빠짐. 또한 `alpha_config.py` (tools/, MG→HF converter용)와 `configuration_alpha.py` (hf_model/, HF AutoConfig용) 두 파일 이름이 비슷해서 1차는 전자만 수정.
- **수정**: 7개 stale defaults 모두 `baseline_48L.yaml` 현재 값으로 갱신. `__init__` 시그니처 + docstring 동기화.

| Param | 옛 default | 새 default | 출처 |
|---|---|---|---|
| `vocab_size` | 151936 | **163968** | `baseline_48L.yaml::padded-vocab-size` |
| `intermediate_size` | 5632 | **8192** | `baseline_48L.yaml::ffn-hidden-size` |
| `max_position_embeddings` | 32768 | **262144** | `baseline_48L.yaml::max-position-embeddings` |
| `rope_theta` | 10000.0 | **10000000.0** | frontier 10M (alpha RoPE) |
| `num_experts_per_tok` | 10 | **8** | `baseline_48L.yaml::moe-router-topk` |
| `num_experts` | 512 | **192** | `baseline_48L.yaml::num-experts` (2026-05-26: 184→192 정정; 위 §"v2 평가 파이프라인" 참조) |
| `router_aux_loss_coef` | 0.001 | **1.0e-4** | `baseline_48L.yaml::moe-aux-loss-coeff` (DSV3) |

- **영향 (왜 학습 안전, 배포 위험)**: Stage 1 학습은 Megatron-native config + YAML로 굴러가서 AlphaConfig() 자체를 호출 안 함 → 학습 자체엔 무관. **MG→HF 변환 후** HF/SGLang/vLLM이 `AlphaConfig.from_pretrained` 시 config.json에 없는 키 (예: 옛 checkpoint json) 가 있으면 stale default로 fall back → embedding-table mismatch / wrong topk shape / 잘못된 RoPE 주기 같은 silent corruption.
- **2차 검증 의의**: 1차에서 "이건 deferred해도 안전" 판단이 *Stage 1 학습 자체*에 한정해 맞았지만, "deployment 시점 silent footgun"이라는 별도 risk surface를 closing.
- **회귀 테스트**: `tests/test_alpha_tokenizer_eod.py::test_configuration_alpha_defaults_match_baseline_48L` — 7개 default를 각각 assert (총 test count 9 → 10).

## Document boundary handling 활성화 (2026-05-12 preflight ✅)
- **변경**: `stage1.yaml`에 `reset-position-ids: true`, `reset-attention-mask: true`, `eod-mask-loss: true` 추가.
- **이유**: yanring/Megatron-MoE-ModelZoo Qwen3-Next-80B-A3B 레퍼런스 recipe와 정렬. 매 packed sample 안에서 EOD (id 0) 위치마다 position vector reset + cross-doc attention 차단 + EOD 토큰을 loss에서 제외.
- **필수 조건**: 데이터에 EOD가 stream 토큰으로 존재해야 함. Megatron의 `gpt_dataset.py:683` `eod_index = position_ids[data == eod_token]`이 `.bin` 안 id 0을 스캔해서 reset 위치 결정. `.idx::document_indices`는 *sample packing* 단계에서만 쓰이고 runtime reset에는 미사용. 따라서 위 "데이터 EOD remap"이 필수 선행 조건.
- **Differential 검증** (Phase C-loader, `tests/preflight_stage1/C_loader_audit.md`):
  - ON: cross-doc attn 차단 100%, max position_id 평균 ~2000, loss_mask coverage ~99.9%
  - OFF (control): 차단 0%, max position_id 항상 4095, coverage 100%
  - 모든 source에서 expected delta 관찰 → 머신 정상 작동 입증.

## pretrain_auxfree.yaml → stage1.yaml 마이그레이션 (2026-05-12 ✅)
- **변경**: Stage 1 training preset이 `pretrain_auxfree.yaml`에서 `stage1.yaml`로 이동. 새 파일은 더 보수적인 hyperparam (LR 4e-4 → 2e-4, GBS 2688 → 1536, save-interval 25000 → 10000, eval cadence 강화) + 위 3개 reset flags.
- **`pretrain_auxfree.yaml`**: deprecation header 추가, 삭제는 안 함 (in-flight 스크립트 호환성 + git history 가시성).
- **사용**: `bash train.sh baseline_48L stage1 stage1_v5_blend`.

## apply-layernorm-1p 제거 (Qwen3.5 정렬, 2026-05-20 ✅)
- **변경**: `baseline_48L.yaml`에서 `apply-layernorm-1p: true` 제거 → 표준 RMSNorm (γ=1 init).
- **이유**: Qwen3.5 official `config.json`에는 zero-centered γ flag 없음. 표준 RMSNorm 채택이 baseline 정렬과 일치.
- **QK-Clip 호환성**: `gated_attention.py:325-329`의 `_clip_layernorm_gamma()`가 `if config.layernorm_zero_centered_gamma`로 분기되어, 1p가 꺼지면 자동으로 표준 `w * scale` 분기로 fall-through. **별도 코드 수정 불필요**.
- **smoke 검증**: 1p ON vs OFF에서 iter 1 forward 동등 (loss 11.99280 일치), iter 2부터 backward dynamics 분기 시작.

## WD policy 통일: `apply_wd_to_qk_layernorm` (Qwen3-Next NVIDIA 레시피, 2026-05-20 ✅)
- **변경**: `pretrain_auxfree.yaml`, `stage2_3.yaml`의 `apply_wd_to_all_layernorm` → `apply_wd_to_qk_layernorm`. (`stage2_2.yaml`은 이미 그러함.)
- **이유**: yanring/Megatron-MoE-ModelZoo `Qwen3-Next-80B-A3B.yaml`이 `--no-weight-decay-cond-type: qwen3_next` 명시 ("Qwen3-Next applies weight decay to qk layernorm as a special case"). 이는 `apply_wd_to_qk_layernorm`과 동의어. 즉 **QK norm γ에만 WD, 다른 layernorm γ는 WD 제외**.
- **이전 `apply_wd_to_all_layernorm` 도입 이력**: Stage 2-3에서 LN γ 폭발 fix 시도였으나, Qwen3 family와 정렬을 위해 QK-only로 회귀.

## Tokenizer migration to alpha v5 (in-repo, 2026-05-20 ✅)
- **변경**: 기존 `examples/alpha/tokenizer/` (Qwen 호환 BBPE, 7 files, vocab 151,936) → 신규 `examples/alpha/tokenizer_v5/` (alpha 전용 BBPE, 5 files, vocab 163,860; padded 163,968).
- **자동 갱신된 참조** (총 9곳): `configs/model/baseline_48L.yaml`, `configs/model/smoke.yaml`, `tools/alpha_config.py` (default), 7개 `toolkits/pretrain_data_preprocessing/preprocess_*.sh`, `toolkits/data_extraction/extract_training_samples.py`.
- **데이터 호환성**: 새 vocab 163,968은 옛 .bin/.idx (Qwen3 tokenizer로 토큰화)와 mismatch → 모든 학습 데이터 재토큰화 필요.
- **Verification**: smoke test에서 in-repo path와 beta path가 byte-perfect 동일 (iter 1 lm_loss 12.07105 일치) → 5 files만으로 HF AutoTokenizer 동작 충분 확인.

## Smoke / mock 자동 wandb 비활성화 (2026-05-20 ✅)
- **변경**: `train.sh`에 `SMOKE_RUN` 자동 감지 (preset 이름 중 `smoke` 또는 data preset이 `mock`이면 true). True 시 `WANDB_MODE=disabled` export + dummy `--wandb-exp-name smoke_<TS>` emit (Megatron `--wandb-project` argparse validation 통과용).
- **이유**: smoke test가 wandb project를 오염시키지 않도록. 기존엔 `mock` data 사용해도 wandb upload 일어남.
- **Banner**: `wandb: DISABLED (smoke preset detected)` / `online (project: alpha-pretraining)` / `off (no WANDB_API_KEY)` 중 하나로 시작 시 즉시 확인 가능.

## Muon Nesterov 버그 (Stage 2-2에서 발견, 자동 수정됨)
- **증상**: YAML에서 `muon_use_nesterov: true` 설정했으나 실제로는 Nesterov가 비활성화
- **원인**: `--muon-use-nesterov`는 argparse `store_true`(default=False). 구식 셸이 `true`일 때 플래그를 전달하지 않아 항상 False. `false`일 때 전달하는 `--muon-no-use-nesterov`도 Megatron에 미정의
- **영향**: Stage 1~2-1 전체에서 일반 heavy ball momentum으로 학습 (Nesterov 미적용)
- **현재 상태**: ✅ 새 train.sh의 `yaml_to_flags`가 store_true semantics를 정확히 재현 (`muon-use-nesterov: true` → `--muon-use-nesterov` emit / false → omit). 같은 부류의 버그는 새 launcher에서는 구조적으로 발생 불가능

## QK LayerNorm Gamma 폭발 (Stage 1에서 발견, Stage 2에서 수정)
- **증상**: 마지막 attention layer(Layer 23)의 `q_norm`/`k_norm` gamma가 11.9~12.9로 폭발 (정상: ~1.97)
- **원인**: QK LayerNorm gamma는 1D param → weight decay 미적용 + QK-Clip이 gradient 신호 차단 → gamma 성장 무제한
- **수정**: `--no-weight-decay-cond-type apply_wd_to_qk_layernorm` (NVIDIA GatedDeltaNet 공식 레시피)
- **설정 위치**: `configs/training/stage2.yaml` → `training.no_weight_decay_cond_type`
- **버그 수정**: `megatron_patch/training.py`에서 `no_weight_decay_cond`를 `setup_model_and_optimizer()`에 전달하지 않던 버그 수정 (upstream Megatron과 동기화)
- **Confluence**: [QK LayerNorm Weight Decay 적용 (Stage 2 버그 수정)](https://alphabanana.atlassian.net/wiki/spaces/AB/pages/10944513)

## QK-Clip crash on hybrid model (Stage 2에서 발견)
- **증상**: `--qk-clip` 사용 시 `AttributeError: 'MambaLayer' object has no attribute 'self_attention'`
- **원인**: Upstream `clip_qk()` (Megatron-LM)이 모든 decoder layer에 `self_attention`이 있다고 가정 → MambaLayer에서 크래시
- **수정**: `pretrain_alpha.py`에서 `clip_qk`을 monkey-patch하여 `hasattr(layer, 'self_attention')` 가드 추가
- **위치**: `examples/alpha/pretrain_alpha.py` (line ~105-136)

## QK-Clip 로깅이 안 되던 문제 (해결 완료 ✅)
- **증상**: `--qk-clip` 설정해도 max attention logit이 로그에 안 나옴
- **원인 분석**:
  - `pretrain_alpha.py`는 `from megatron.training import pretrain` — **upstream** `pretrain()` 사용 (megatron_patch/training.py 미사용)
  - Upstream `train_step()`에 이미 `clip_qk()` 호출이 있어 **QK-Clip 자체는 동작 중**이었음
  - 문제는 upstream `training_log()`가 `--log-max-attention-logit` 플래그 없으면 TensorBoard/WandB에 기록하지 않고, 콘솔에는 아예 출력하지 않음
- **수정**: `train_stage2.sh`의 QK-Clip 인자에 `--log-max-attention-logit` 추가
- **검증**: WandB에서 `max_attention_logit` ≈ 100 (threshold) 근처로 안정 동작 확인
- **참고**: `megatron_patch/training.py`에도 `clip_qk()` 호출 + 로깅을 포팅함 (다른 모델이 patched `pretrain()` 사용 시 필요)
- **Confluence**: [QK-Clip 완전 활성화 (Stage 2)](https://alphabanana.atlassian.net/wiki/spaces/AB/pages/12845058)

## QK-Clip LayerNorm Gamma 스케일링 (GQA+QK-Norm 고유 수정, 구현 완료 ✅)
- **증상**: QK-Clip 적용 후에도 max attention logit이 threshold 근처로 내려가지 않음
- **원인**: QK-Norm(RMSNorm)이 W_q/W_k 스케일링을 상쇄하여 QK-Clip이 사실상 무력화
  - MuonCLIP 논문의 `W_qr`은 MLA query rotary projection이지 LayerNorm gamma가 아님
  - 우리 GQA+QK-Norm 아키텍처에서는 RMSNorm이 projection 스케일링을 정규화하므로, gamma도 함께 스케일링해야 함
  - 이것은 논문에 없는, GQA+QK-Norm 아키텍처 고유의 수정
- **수정**: `megatron_patch/model/qwen3_next/gated_attention.py`에 `_clip_layernorm_gamma()` 메서드 추가
  - `clip_qk()` 내에서 Q/K projection 스케일링 후 `q_layernorm`/`k_layernorm`의 gamma도 스케일링
  - `layernorm_zero_centered_gamma` (1p layernorm) 처리: `(1+w)*scale - 1`
  - 공유 layernorm이므로 `min(eta)` (worst-case head) 사용

