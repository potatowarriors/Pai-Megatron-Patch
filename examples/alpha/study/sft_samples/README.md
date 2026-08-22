# SFT 데이터셋 샘플 (직접 뜯어보기용, 2026-08-20)

`/home/work/Datasets/LL_datasets/posttraining/SFT/`에서 대표 행을 추출한 것.
각 JSON의 `_provenance` 필드에 원본 파일 경로와 행 번호가 있다.
`*__rendered.txt` 2개는 같은 이름의 JSON을 alpha v5 chat template로 실제 렌더링한 결과다.

## 파일 목록

| 파일 | 원본 | 내용 |
|---|---|---|
| `agentic_v2__tool_calling__row1.json` | SFT-Agentic-v2 `tool_calling.jsonl:1` | 15턴 고객상담 에이전트. tool_calls 4건 + tool 응답. policy system prompt |
| `agentic_v2__tool_calling__rendered.txt` | 위 행의 템플릿 렌더 | 4,287 토큰. XML tool call·think truncation 확인용 |
| `agentic_v2__interactive_agent__row14.json` | SFT-Agentic-v2 `interactive_agent.jsonl:14` | 인터랙티브 에이전트 5턴 |
| `if_chat_v3__chat__row35.json` | SFT-IF-Chat-v3 `chat.jsonl:35` | **system/첫 user가 null** (마스킹, 아래 §발견 2) + `train_turns` |
| `if_chat_v3__instruction_following__row5.json` | SFT-IF-Chat-v3 `instruction_following.jsonl:5` | 제약 지시 이행 + `train_turns` |
| `math_v4__train__row4.json` | SFT-Math-v4 `train.jsonl:4` | 단일턴 수학. reasoning 3.0k chars |
| `multilingual_v2__math_ko__row3.json` | SFT-Multilingual-v2 `..math_ko..jsonl:3` | 한국어 수학 |
| `multilingual_v2__math_ko__rendered.txt` | 위 행의 템플릿 렌더 | 1,192 토큰 |

## 스키마 ↔ chat template 매핑

원본 데이터는 렌더된 텍스트가 아니라 **구조화된 messages 포맷**으로 저장돼 있고,
필드명이 alpha 템플릿(= Nemotron 3 Ultra)의 입력 변수와 1:1로 맞는다.

| 데이터 필드 | 템플릿에서의 소비처 |
|---|---|
| `messages[].role/content` | 턴 렌더링 (`<\|im_start\|>role ... <\|im_end\|>`) |
| `messages[].reasoning_content` | assistant의 `<think>...</think>` 블록 (content와 분리 저장) |
| `messages[].tool_calls[].function.{name,arguments}` | `<tool_call><function=...><parameter=...>` XML |
| `messages[].role == "tool"` + `tool_call_id` | user 턴 속 `<tool_response>` 병합 |
| 행 최상위 `tools` | system 턴의 `<tools>` 함수 선언 블록 |
| 행 최상위 `chat_template_kwargs` | 렌더 옵션 (예: `{"thinking": true}`) |

## 발견 4건 (messages→idxmap 변환기 구현 시 필수 반영)

1. **`tool_calls[].function.arguments`는 JSON *문자열*이다** (OpenAI wire format).
   템플릿은 dict를 기대하므로(`|items` 순회) 그대로 넣으면 `TypeError`로 렌더 실패 —
   재현됨. 변환기는 렌더 전 `json.loads()` 전처리가 필수.
2. **`chat.jsonl`은 전수(확인 표본 2000/2000행) seed prompt 마스킹.** LMSYS/WildChat
   라이선스 탓에 system·첫 user가 `null`. 데이터셋 동봉
   `prepare_chat_prompts.py`가 HF 원본에서 SHA-256 매칭으로 복원한다 — SFT 착수 전
   실행 필수 (RL 블렌드 `fill_placeholders.py`와 같은 패턴,
   `docs/SFT_RL_DATASETS.md` §5.1).
3. **`metadata.train_turns` = 턴별 loss mask 지시.** IF-Chat-v3 계열에 존재
   (예: `[false,false,false,false,true]` — 마지막 assistant 턴만 학습).
   agentic/math/multilingual에는 없음 → 전 assistant 턴 학습으로 해석.
   변환기의 assistant-스팬 마스킹(`tools/verify_chat_template.py` 규약)과 AND로
   결합해야 한다.
4. **`chat_template_kwargs` 키가 `thinking`인데 템플릿 변수는 `enable_thinking`.**
   SFT 렌더(add_generation_prompt=False)에는 영향 없음 — 이 변수는 generation
   prompt에만 관여. 행의 `thinking: true`는 "이 행에 reasoning trace가 있다"는
   신호로 읽으면 된다.

## 렌더 결과에서 확인되는 템플릿 동작

`agentic_v2__tool_calling__rendered.txt`에서 직접 볼 수 있는 것:

- 과거 assistant 턴의 think는 `<think></think>`로 비워짐 (`truncate_history_thinking`
  기본 True). 마지막 user 턴 이후의 assistant think만 원문 보존.
- tool call은 XML 규약: `<tool_call><function=get_artwork_details><parameter=artwork_id>...`.
- tool 응답은 assistant도 tool role도 아닌 **user 턴 속 `<tool_response>`**로 들어감.
- 열린 설계 질문: 멀티턴 행에서 history think가 렌더 시 제거되므로, 과거 턴의
  reasoning_content를 학습 대상으로 쓰려면 턴별로 별도 샘플로 펼쳐야 한다.
  truncation 유지(추론 분포와 일치) vs 턴별 전개(reasoning 데이터 활용) 중 블렌드
  설계 시 결정 필요.

## 재현

추출 스크립트는 세션 스크래치패드에서 실행했다. 동일 로직 요지:
첫 3000행 스캔 → 크기 제한(가독성) + 조건(tool role 존재 / reasoning 존재 /
멀티턴)으로 첫 매칭 행 선택 → `_provenance` 붙여 저장 → 렌더는
`AutoTokenizer.from_pretrained(examples/alpha/tokenizer_v5).apply_chat_template`
(tool arguments만 `json.loads` 전처리).
