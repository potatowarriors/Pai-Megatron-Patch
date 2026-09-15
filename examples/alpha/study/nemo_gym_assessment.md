# NeMo-Gym 분석 — alpha_v2 post-train(SFT 단계) 활용 검토

_작성 2026-09-15. 대상 사본: `project_s/NeMo-RL/3rdparty/Gym-workspace/Gym`(커밋 bb9b26d, 2026-08-05, 버전 0.5.0-dev). 업스트림 최신 태그 v0.6.0._

## 0. 결론

NeMo-Gym 은 "환경 = 데이터셋 + 에이전트 하니스 + 검증기 + 상태"를 FastAPI 서버 3종(Agent · Model · Resources)으로
표준화한 **평가·롤아웃 수집 프레임워크**다. 학습기가 아니다. 학습은 NeMo-RL(GRPO·온폴리시 증류)이 Gym 을 호출해서 한다.

우리가 SFT 벤치·서빙에서 겪은 문제 4종과의 대응은 다음과 같다.

| 문제 | Gym 이 직접 해결하는가 | 근거 |
|---|---|---|
| 추론 파서 (`</think>` 소실, tool 파서 유무에 따른 content 오염) | **예.** Model Server 가 vLLM 의 `reasoning_content` 를 받아 `<think>…</think>` 로 감싸 Responses 궤적에 `reasoning` 항목으로 보존한다. 파서 미설정은 서버 기동 시 assert 로 거부 | §1.4 |
| 이전 턴 추론 보존 (restore/strip, `tau_proxy`) | **예.** 다음 호출 때 `<think>` 를 다시 `reasoning_content` 필드로 분리해 vLLM 에 보낸다. vLLM 0.25.1 의 `reasoning` 키와 구형 `reasoning_content` 를 모두 읽고 모두 쓴다(우리 하니스가 `reasoning_content` 만 읽어 조용히 strip 되던 원인 제거). 도구가 선언된 흐름은 alpha 템플릿의 `truncate_history_thinking` 분기로 자동 restore, TB-2 처럼 도구 미선언 하니스는 서버 설정 `chat_template_kwargs.truncate_history_thinking: false` 한 줄 — Harbor 브리지의 학습용 권장 설정이 정확히 이것이다. `tau_proxy` 두 대(:8110·:8111)가 없어진다 | §1.4, §3.2 |
| 하니스 (SWE · Terminal-Bench · τ² /τ³ · 도구 호출) | **대부분 내장.** mini-swe-agent 2 · OpenHands · Harbor+Terminus-2 · τ²/τ³(banking_knowledge) · 단일/다단계 도구 호출 검증기. 우리 자체 러너(`run_swe.sh`·`run_tau.sh`·`tau_proxy.py`)를 대체할 수 있다 | §3.3 |
| 평가 docker 환경 관리 | **부분.** 샌드박스 API(Docker/Apptainer/OpenSandbox/…)가 컨테이너 생성·exec·정리를 표준화하지만 이미지 빌드·풀·디스크·주소풀은 여전히 우리 몫 | §3.4 |

SFT 단계에서 바로 쓸 수 있는 용도는 세 가지다. 우선순위 순.

1. **에이전틱 평가 티어 이관** — fleet + 역터널 + `tau_proxy` + 자체 러너 조합을 `gym eval run` 하나로. 이전 턴 추론 복원이 프레임워크 기본 동작이 되어 09-14 유형 사고(파서 누락)가 구조적으로 막힌다.
2. **학습 데이터 게이트의 정량화** — 유령 호출 프로브(도구 25종 주입 33문항)·BFCL 양방향을 Gym 환경(`single_step_tool_use_with_argument_comparison`, `xlam_fc`)으로 옮기면 `--num-repeats` 로 산포까지 잰다.
3. **롤아웃 → SFT 데이터** — SDG 트랙(kotool·search_ko·usab)의 교사 궤적 생성을 Gym 환경 + 검증기로 돌리면 보상이 붙은 Responses 형식 궤적이 나오고, 같은 환경이 RL 단계 GRPO 환경이 된다(재작업 0).

도입 비용은 작지 않다. Python 3.13.14 + uv 전용 venv(Pai 스택 3.12 와 분리), 현재 main1 에는 uv 도 docker 도 없고 `NeMo-RL/.venv` 는 재부팅으로 끊긴 심볼릭 링크다(§5).

## 1. NeMo-Gym 이란

### 1.1 구성

| 개념 | Gym 구현 | 우리 대응물 |
|---|---|---|
| Dataset | JSONL, 행마다 `responses_create_params`(OpenAI Responses API 입력) + 검증 메타 + `agent_ref` | `eval_sft/*.jsonl`, SWE/TB 태스크 디렉토리 |
| Agent Harness | `responses_api_agents/<name>/app.py` — `run()` 이 시드 → 모델 호출 루프 → 검증 | `run_swe.sh`(mini-swe-agent), `run_terminal.sh`(Harbor), `run_tau.sh`, `search_agent_eval.py` |
| Model Server | `responses_api_models/vllm_model` 등 — Chat Completions ↔ Responses 변환 + 추론 왕복 + 토큰 ID | `serve_alpha.sh` fleet + `tau_proxy.py` |
| Resources Server (검증기+상태) | `resources_servers/<name>/app.py` — `verify()`, 도구 엔드포인트, 세션 상태 | 벤치별 채점 스크립트 |

세 서버는 aiohttp 로 통신하고(재시도 3회·커넥션 풀), 세션 쿠키로 롤아웃별 상태를 격리한다. Gym 자체는 **CPU 전용**이다.
GPU 는 우리가 띄운 vLLM 이 쓴다.

### 1.2 규모·버전

| 항목 | 벤더 사본 (08-05) | 업스트림 v0.6.0 |
|---|---|---|
| resources_servers | 103 | 더 많음 |
| benchmarks | 89 (aime·gpqa·hle·ifeval·livecodebench·mmlu_pro·ruler·mrcr·simpleqa·tau2·swe 계열·terminal…) | + Terminal-Bench 2.1(OpenCode sandboxed), RULER v2, AgentIF, LongMemEval… |
| agents | 34 (simple, hermes, claude_code, codex, opencode, harbor(Terminus-2), mini_swe_agent_2, anyswe, anyterminal, tau2…) | 47 (+ `terminus_2_agent`, `terminus_2_sandboxed_agent`, `simple_agent_with_compaction`, `cline_agent`…) |
| 샌드박스 프로바이더 | OpenSandbox · Apptainer · Docker · Daytona · ECS Fargate · Enroot · OpenShell | + E2B |
| 진단 | `gym eval profile`, `gym eval reverify`, 모델 호출 캡처(`ng_trajectory`), BLADE 분석 스킬 | + `gym eval health-check`(턴 누락·토큰 불일치·폭주 생성 자동 검출), `failure_reason` |

v0.6.0 변경 중 우리에게 중요한 것: `agent_ref` 만으로 라우팅하던 데이터셋은 deprecated(`task_source` 필요, `gym dataset collate` 재실행) —
`SFT_RL_DATASETS.md` §3.1 의 RL 블렌드가 이 형식이다. `openai==2.44.0` 강제 핀.

### 1.3 데이터 행 형식

```json
{"responses_create_params": {"input": [{"role": "user", "content": "..."}], "tools": [...]},
 "expected_answer": "...", "agent_ref": {"name": "mcqa_simple_agent"}}
```

롤아웃 출력 행 = 입력 행 + `response`(Responses `output` 항목 열: `reasoning` · `message` · `function_call` · `function_call_output`)
+ `reward` + 검증기 필드 + `mask_sample`(타임아웃 등 신뢰 불가 표시) + `_ng_task_index`/`_ng_rollout_index`.
학습용(`return_token_id_information: true`)이면 항목마다 `prompt_token_ids`·`generation_token_ids`·logprob 이 붙는다.

Responses 형식을 택한 이유는 멀티턴·도구 호출·추론을 직렬화 규약 없이 표현하기 위해서다(`fern/…/responses-api-evolution.mdx`).
Chat Completions 로의 역변환기(`nemo_gym/responses_converter.py`)가 있어 우리 SFT 변환기 입력(messages 리스트)으로 되돌릴 수 있다.

### 1.4 핵심 메커니즘 — 추론의 왕복 (우리 문제 1·2 의 답)

`responses_api_models/vllm_model/app.py` 기준.

**응답 방향** (vLLM → Gym): `uses_reasoning_parser: true` 이면 choice 의 `reasoning_content` **또는** 신형 `reasoning` 키를 꺼내
`<think>…</think>` 로 감싸 content 앞에 붙인다(app.py:674-687). 09-14 잔여 위험으로 기록된 키 분열(vLLM 0.25.1 은 `reasoning`, harbor `lite_llm.py:428`·`check_gates.py`·LibreChat 은 `reasoning_content`)이 모델 서버 한 곳에서 흡수된다. 변환기가 이를 `reasoning` 출력 항목으로 분리한다
(responses_converter.py:407-420). `uses_reasoning_parser: false` 인데 vLLM 이 reasoning 키를 주면 **assert 로 중단**한다 —
09-14 처럼 "파서가 켜졌는지 모르는 채 돌아가는" 상태가 성립하지 않는다.

**요청 방향** (Gym → vLLM): 이력의 `reasoning` 항목은 `<think>` 로 재결합돼 assistant content 에 들어가고(responses_converter.py:330-345),
vLLM 으로 보내기 직전 다시 추출해 `reasoning_content` + `reasoning` 필드로 옮긴다(app.py:442-475). 즉 **이전 턴 추론을 필드로 보존한 채**
vLLM 채팅 템플릿에 넘긴다. 이후는 템플릿의 몫이다.

alpha `tokenizer_v5/chat_template.jinja` 는 이미 이 필드를 소비한다.

| 템플릿 줄 | 동작 |
|---|---|
| 19-28 | `tools` 선언 또는 `tool`/`tool_calls` 턴이 있으면 `tool_scenario`; `truncate_history_thinking` 기본값 = `not tool_scenario` |
| 114-115 | `message.reasoning_content` 가 있으면 `<think>\n…</think>` + content 로 재구성 |
| 128-140 | `truncate_history_thinking` 이고 마지막 user 이전 턴이면 `</think>` 뒤만 남기고 `<think></think>` 접두 |

따라서 도구가 선언된 흐름(SWE·τ²·도구 호출 프로브)에서는 Gym 모델 서버 + alpha 템플릿만으로 restore 가 성립한다. **TB-2 는 예외**다: terminus-2 는 도구를 선언하지 않고 터미널 출력을 user 턴으로 보내므로 템플릿이 무도구로 분류해 절단한다(`SFT_BENCHMARKS.md` §3.14). 우리는 이를 `tau_proxy` + `interleaved_thinking=true` + `truncate_history_thinking=false` 세 조건으로 풀었고, 셋 중 하나라도 빠지면 **오류 없이 strip 으로 퇴화**했다. Gym 에서는 같은 세 조건이 프록시 대신 설정으로 표현된다 — `harbor_agent/README.md` "NeMo RL Training" 절의 권장 설정이 `chat_template_kwargs: {enable_thinking: true, truncate_history_thinking: false}` + `harbor_agent_kwargs.interleaved_thinking: true` 다. 판정 방법은 동일하다(요청 원문의 이력 assistant 턴에 `reasoning_content` 존재 + `prompt_tokens` 증가). `tau_proxy.py` 의 역할이 없어진다.

부수 효과: 지금은 T1/T3 fleet 가 파서 없이(채점기가 content 의 `</think>` 를 읽음), 에이전틱 fleet 가 파서 켜고 떠서 티어마다 fleet 설정이 결합돼 있다(`SFT_BENCHMARKS.md` §2.5). Gym 은 추론을 항상 별도 항목으로 분리해 검증기에 깨끗한 content 를 주므로 **전 티어 한 가지 fleet 설정(파서 ON)** 이 가능하다.
per-row 오버라이드도 있다: 행의 `metadata.chat_template_kwargs`(JSON 문자열)가 전역 `chat_template_kwargs` 와 병합된다(app.py:404-415) —
RULER 등 무추론 평가는 행별 `{"enable_thinking": false}` 로 끝난다(08-30 함정 표의 해법과 동일).

관련 설정 5개 (`VLLMModelConfig`, app.py:147-199):

| 키 | 기본 | 의미 |
|---|---|---|
| `uses_reasoning_parser` | 필수 | vLLM `--reasoning-parser` 사용 여부. 우리 fleet = `nemotron_v3` → `true` |
| `uses_interleaved_reasoning` | true | 이전 턴 추론을 `reasoning_content` 로 되돌려 보낼지. false 면 content 에서 `<think>` 만 제거(= 강제 strip) |
| `preserve_reasoning_in_assistant_content` | false | true 면 이력의 `<think>` 태그를 content 에 그대로 둔다(직접-vLLM 계약 모델용) |
| `sequential_reasoning_allowed` | true | 추론 뒤 다시 추론 생성 허용 |
| `chat_template_kwargs` / `extra_body` | null | vLLM 로 전달. `truncate_history_thinking`·`enable_thinking` 여기 |

**미종결 `</think>` 처리**: 변환기 정규식은 열림·닫힘 쌍만 잡는다(`THINK_TAG_PATTERN`). 미종결 사고는 content 에 남아 그대로 이력에
들어간다 — 09-14 관찰한 TB-2 restore 루프와 같은 위험이 simple_agent 경로에는 그대로 있다. Harbor 브리지(`harbor_agent/custom_agents/llms/nemo_gym_llm.py:154-185`)는
열림 태그만 있는 경우를 명시적으로 분기해 `reasoning_content=None` 으로 둔다. 업스트림 v0.6 `simple_agent_with_compaction` 은
"오래된 reasoning 블록 생략" 정책을 제공한다.

**도구 호출 파싱**은 Gym 이 하지 않는다. vLLM `--tool-call-parser` 가 낸 `tool_calls` 를 그대로 `function_call` 항목으로 옮긴다.
우리 XML `<function=…>` 형식은 지금처럼 `qwen3_xml` 로 vLLM 에서 파싱한다(08-30 함정 표 A4 게이트와 동일 전제).
`/v1/completions` 경로(`use_completions_api: true`)를 쓰면 파서가 아예 안 돌아 호출자가 직접 파싱해야 한다.

## 2. 사용법

### 2.1 설치

```bash
git clone https://github.com/NVIDIA-NeMo/Gym.git && cd Gym      # 또는 벤더 사본
uv venv --python 3.13.14 && source .venv/bin/activate && uv sync  # Python ≥3.13.14 필수, GPU 불필요
```

`env.yaml`(리포 루트, gitignored) 에 정책 모델 엔드포인트를 둔다.

```yaml
policy_base_url: http://<sub1>:8000/v1     # serve_alpha.sh fleet
policy_api_key: dummy
policy_model_name: alpha                    # --served-model-name
```

리소스 서버마다 `gym env start` 가 전용 venv 를 만든다(첫 기동 느림, `skip_venv_if_present` 로 재사용). NeMo-RL 쪽에
`examples/nemo_gym/prefetch_venvs.py` 가 있다(`configs/alpha/README.md` 미결 항목).

### 2.2 서버 기동 · 롤아웃 · 산출물

```bash
# 1) 서버 3종 기동 (외부 vLLM 사용)
gym env start --benchmark tau2 --model-type vllm_model \
  ++policy_model.responses_api_models.vllm_model.uses_reasoning_parser=true

# 2) 롤아웃 수집 (다른 셸)
gym eval run --no-serve --agent tau2 --input benchmarks/tau2/data/airline.jsonl \
  --output results/tau2_airline.jsonl --num-repeats 4 --concurrency 64 --resume

# 3) 부가
gym eval profile   --inputs results/tau2_airline_materialized_inputs.jsonl --rollouts results/tau2_airline.jsonl  # 태스크별 pass rate·분산
gym eval reverify  --config <resources-only.yaml> --inputs … --rollouts … --output …  # 추론 재실행 없이 채점만 재계산
gym eval aggregate --input-glob 'results/shard*.jsonl' --output results/all.jsonl     # 샤드 병합
gym env validate / gym env status / gym list benchmarks|agents
```

산출물: `<out>.jsonl`(롤아웃) · `<out>_materialized_inputs.jsonl`(펼친 입력, resume·reverify 키) · `<out>_aggregate_metrics.json`
(`mean/reward`, `pass@1[avg-of-k]`, 태스크별 group_level_metrics) · `<out>_failures.jsonl`. `--resume` 는 `(task_index, rollout_index)` 로 완료분을 건너뛴다.
wandb 업로드는 `+wandb_project=… +wandb_name=…`.

설정은 Hydra 병합이다. `--config` 여러 개 + `++경로=값` 오버라이드. `gym env resolve` 로 최종 병합본을 덤프한다.

### 2.3 환경 만들기 (우리 프로브·SDG 검증기용)

`resources_servers/<name>/app.py` 에 `SimpleResourcesServer` 를 상속해 `verify()`(필수)와 도구 엔드포인트(선택)를 구현하고,
`configs/<name>.yaml` 로 agent(대개 `simple_agent`) 와 묶는다. 튜토리얼 4단계(single-step → multi-step → stateful → multi-reward)가
`fern/…/environment-tutorials/` 에 있다. 규칙: `/run` 은 async, 외부 프로세스는 `asyncio.Semaphore`, HTTP 는 반드시 aiohttp(httpx 금지).

## 3. 우리 문제와의 대응 (상세)

### 3.1 파싱

| 우리 사고 | 원인 | Gym 에서는 |
|---|---|---|
| 09-14 fleet `</think>` 소실 (SWE 0/148,836턴) | `TOOLS=1` 만 켜면 vLLM 0.25.1 파서 엔진이 THINK_END 를 소비, reasoning 파서 부재 | `uses_reasoning_parser` 를 서버 설정으로 선언. 파서 상태와 설정이 어긋나면 assert. 응답의 reasoning 은 항상 별도 항목 |
| 08-30 SWE/Terminal 0점 — hermes(JSON) vs XML 파서 불일치 | vLLM 파서 선택 | 동일하게 vLLM 파서 문제. Gym 디버깅 스킬(`.agents/skills/nemo-gym-debugging`)에 tool 스키마 정적 검사기·요청 경계 로그(`++global_aiohttp_client_request_debug=True`) 제공 |
| 09-09 OpenWebUI 도구 25종 주입 | UI 발 요청 변형 | Gym 은 행에 적힌 `tools` 만 보낸다. 프롬프트 증분은 `ng_model_call_capture` 로 요청 원문 확인 가능 |
| `reasoning` vs `reasoning_content` 키 분열 → 하니스가 조용히 strip | vLLM 0.25.1 신형 키 | 모델 서버가 양쪽을 읽고 양쪽을 쓴다(§1.4) |
| T1 채점기가 content 의 `</think>` 를 읽어 티어별 fleet 파서 설정이 갈림 | 채점기·파서 결합 | 추론이 항상 별도 항목이라 파서 ON 단일 fleet(§1.4) |
| TB-1 `response_format=json_schema` 로 추론 차단(15,819 에피소드 100% `{` 시작) | 하니스가 guided decoding 강제 | Gym 도 하니스가 `extra_body` 로 보내면 막지 못한다. 요청 원문 캡처로 검출은 즉시 |

### 3.2 이전 턴 보존

우리 규약(`INTERLEAVED_THINKING.md`, 09-14 사용자 규칙): 도구 사용 평가 = restore, 무도구 = strip. 구현은 `tau_proxy`(gpu06 컨테이너 안 SWE :8110·TB-2 :8111, 체인 해시 캐시로 원문 복원)다.
하니스별로 경로가 다르다: mini-swe-agent 는 `reasoning_content` 를 이력에 되돌려 보내므로 필드 인라인으로 restore 되고(miss 0/0), tau2-bench·harbor 는 필드를 버려 프록시 캐시가 필요했다.
Gym 에서는 두 경로 모두 모델 서버가 담당한다 — 필드가 오면 그대로 통과, `<think>` 가 content 에 오면 필드로 분리(§1.4). TB-2 는 `truncate_history_thinking: false` 명시가 필요하다(§1.4).
검증 방법: Gym `vllm_model` 로 τ² 1문항을 돌리고 `ng_model_call_capture` 의 요청 원문에서 2턴째 assistant 메시지에 `reasoning_content` 가 있는지,
무도구 행에서는 템플릿이 `<think></think>` 로 절단했는지(vLLM `--enable-log-requests` 프롬프트)를 본다 — 09-14 스모크(restore·strip 양쪽 PASS, d8d40bf)와 같은 판정.

남는 것: 미종결 `</think>` 루프(§1.4). `steps_reasoning_only` 지표는 우리 러너 것이라 Gym 으로 옮기면 `compute_metrics()` 훅으로 재구현해야 한다.

### 3.3 하니스

| 벤치 | 우리 (2026-09-14) | Gym 벤더 사본 | 비고 |
|---|---|---|---|
| SWE-bench Verified | mini-swe-agent + vLLM + `tau_proxy` + 인스턴스 이미지 수동 관리 | `mini_swe_agent_2`(v2.1.0, 샌드박스 API) · `swe_agents`(OpenHands) · `anyswe_agent`(Hermes/Claude Code/OpenCode…) | 이미지는 `NeMo-RL/examples/nemo_gym/download_swe_images.py` 가 `.sif` 로 내려받음 |
| Terminal-Bench 2 | Harbor + Terminus-2 + `tau_proxy` | `harbor_agent`(Terminus2NemoGym + Singularity env, `prepare_terminal_bench_2_1.py`) · `anyterminal_agent`(Apptainer) | 학습용 설정은 `interleaved_thinking: true` + `truncate_history_thinking: false`. 관측성 매트릭스에서 harbor_agent 는 전 항목 X. v0.6 에 `terminus_2_agent`·OpenCode sandboxed TB-2.1 추가 |
| τ² / τ³ | tau2-bench v1.0.1 직접 실행(`run_tau.sh`), 시뮬레이터 gemma-4-12B, telecom 은 시뮬레이터가 `tools` 를 못 받아 제외 | `benchmarks/tau2`(airline·retail·telecom) + τ³ `banking_knowledge`(bm25_grep · terminal_use · alltools) | 시뮬레이터가 GPT-5.4-mini 로 고정된 config → `vllm_model`(tool-choice 켠 로컬 서버)로 교체. telecom 제약은 동일. tau2-bench 는 NVIDIA 핀 포크. litellm 모델 등록 `.pth` 훅 같은 우회는 불필요(Gym 은 litellm 을 안 씀) |
| 검색 에이전트 | `search_agent_eval.py` — 파서를 일부러 우회(템플릿 자체 렌더 + `/v1/completions` + XML 자체 파싱)해 모델만 잰다 | `tavily_search`·`google_search`·`hotpotqa_qa`·`browsecomp_advanced_harness`. 같은 우회 경로가 `use_completions_api: true` + `render_chat_template: true` 로 있음(도구 파싱은 호출자 몫) | 로컬 BM25 는 우리 리소스 서버로 감싸야 함 |
| 도구 호출 (유령 호출·When2Call·BFCL) | 프로브 스크립트 33문항 | `single_step_tool_use_with_argument_comparison`(기대 행동이 "메시지"면 도구 호출 시 음수 보상) · `xlam_fc` · `toolsandbox` · `workplace_assistant`(27 도구 다단계) | When2Call 전용 서버는 없음 |
| IF / 지식 / 수학 / 코드 | lm-eval `_aa` 태스크 | `ifeval`·`ifbench`·`gpqa`·`mmlu_pro`·`hle`·`aime`·`math-500`·`livecodebench`·`ruler`·`mrcr`·`simpleqa` | T1/T3 이관은 별도 결정. Nemotron 3 Ultra 공개 스위트가 `benchmarks/nemotron_3_ultra/` 에 그대로 있음(no_external = gpqa·livecodebench·spider2·ruler 256k) |

### 3.4 평가 docker 환경

Gym 샌드박스 API(`nemo_gym.sandbox`)는 프로바이더 중립이다. Docker 프로바이더는 컨테이너 1개를 keep-alive 로 띄우고 `docker exec/cp/rm` 을 CLI 로 호출하며,
`nemo-gym.sandbox=1` 라벨로 누수 정리(`docker rm -f $(docker ps -aq --filter label=nemo-gym.sandbox=1)`), `ttl_s` 자기 소멸, CPU/메모리 상한, `network: none`·`pids_limit` 등 격리 노브를 준다.

| 우리 사고 | Gym 이 덮는가 |
|---|---|
| 09-08 DinD 주소풀 고갈(Harbor compose 네트워크) | 아니오. Harbor 경로는 Harbor 가 컨테이너를 만든다. Gym 자체 프로바이더는 compose 를 안 써 이 문제가 없음 |
| 09-08 이미지 레이어 미공유 940MB/트라이얼 | 아니오. 이미지 빌드는 우리 몫 |
| 09-07 에이전트 산출물 누적 | 부분. `results/runs/…`·`harbor_jobs_dir` 로 위치가 고정돼 회전 스크립트 대상이 명확해짐 |
| 09-14 vLLM 캐시 ENOSPC · 절대경로 캐시 | 아니오. vLLM 은 우리가 띄움 |
| SWE 인스턴스 이미지 정리 정책 | 부분. `.sif` 일괄 다운로드 스크립트·컨테이너 `--rm` 규약 |

HPC 성향이 강하다: Harbor 브리지는 Singularity 우선, `anyterminal_agent` 는 Apptainer 전용, NeMo-RL SWE 이미지는 `.sif`. Docker 로 가려면 `mini_swe_agent_2` + Docker 프로바이더가 검증된 경로다.

배치: Backend.AI 세션(main1·sub1)은 docker 를 못 돌린다(`cap_sys_admin` 부재). 따라서 SWE·TB-2 의 Gym Agent/Resources 서버는 지금 하니스와 같이 gpu06 `alpha-eval` DinD 컨테이너 안에서 돌고, Model Server 는 역터널(컨테이너 :8199 → sub1 `lb_proxy` :8100)을 가리킨다. Gym 이 CPU 전용이라 이 배치에 GPU 가 더 필요하지 않다. dockerd 수동 기동·`default-address-pools` 재적용·산출물 회전(`docker_gc.sh`) 은 그대로 남는다.

## 4. SFT 단계 활용 방안

### A. 에이전틱 평가 티어 이관 (권고 1순위)

현재 체인: `serve_alpha.sh`(TOOLS=1, nemotron_v3, DP8) → 역터널 → `tau_proxy` ×2 → `run_swe.sh`/`run_terminal.sh`/`run_tau.sh` → 자체 채점.
이관 후: 같은 fleet → Gym `vllm_model`(`uses_reasoning_parser: true`) → `gym eval run --benchmark {tau2, swebench, terminal}`.

얻는 것: 파서 계약 강제, 추론 왕복 기본 동작, `--resume`·`--num-repeats`·`profile`·`reverify`, 롤아웃 원문 캡처, 벤치 간 동일 산출 형식.
비용: Python 3.13 venv, 벤치별 데이터 prepare, τ³ 시뮬레이터 교체, 채점 등가성 확인(같은 ckpt 로 구 하니스 vs Gym 1회 대조 필수 — 이관 전 수치와 계열이 끊긴다는 점은 09-14 파서 변경 때와 같은 열린 결정).

### B. 학습 데이터 게이트 정량화

| 게이트 | 지금 | Gym |
|---|---|---|
| 유령 호출 (도구 25종 주입 33문항, 임계 ≤1/33) | 프로브 스크립트, 1회 생성 | 33행 × 기대행동 `message` 데이터셋 + `single_step_tool_use_with_argument_comparison` → `--num-repeats 8` 로 유령률 평균·분산, `profile` 로 문항별 |
| BFCL AST 양방향 | 별도 | `xlam_fc`(호출 인자 비교) |
| 검색 format/live | `search_agent_eval.py` | 우리 BM25 리소스 서버(도구 `search` + verify) 1개 작성 |

같은 데이터셋을 300 iters 마다 ckpt 에 돌리면 `sft_final_eval_watch.sh` 의 프로브가 표준 산출물(집계 JSON + wandb)로 바뀐다.

### C. 롤아웃 → SFT 데이터 (SDG 트랙)

`fern/…/offline-training-w-rollouts.mdx` 는 **experimental** 표시다. 파이프라인 자체는 단순하다: 교사(`inference_provider` 또는 `vllm_model` 로 GLM/DSV4) → `simple_agent` + 우리 리소스 서버(도구·검증) → `reward` 필터 → `responses_converter` 로 messages 로 되돌려 `build_alpha_sft_idxmap.py` 입력으로.
적용 후보: `kotool_v1`(도구 호출 — 검증기가 인자 비교), `search_ko_v1`/`research_ko_v1`(BM25 도구 + 정답 일치 D1·심판 D2 를 `verify()` 로), `usab`(심판 = `equivalence_llm_judge` 패턴).
장점: 채택/리젝 판정이 재현 가능(`reverify`), 궤적이 Responses 표준 형식이라 RL 단계에서 같은 행을 프롬프트 뱅크로 재사용.
주의: 우리 SFT 변환기의 규약(fan-out, `train_turns`, effort 마커, `--tools-sidecar`)은 Gym 과 무관하므로 변환 뒤 `verify_sft_bins`·`render_check` 는 그대로 필요하다.

### D. RL 단계 연속성

NeMo-RL 은 Gym 을 Ray 액터로 띄워 GRPO·온폴리시 증류 롤아웃을 받는다(`NeMo-RL/docs/design-docs/nemo-gym-integration.md`). Ultra RL 블렌드(`rlvr1/2_alpha.jsonl`)는 이미 Gym 행 형식이다.
SFT 단계에서 만든 리소스 서버(B·C)는 RL 환경으로 그대로 쓰인다. 단 v0.6 `task_source` 라우팅 변경(§1.2)을 벤더 사본 갱신 시 반영해야 한다.

## 5. 도입 제약·리스크

| 항목 | 실측 (2026-09-15, main1) | 대응 |
|---|---|---|
| Python | Gym·NeMo-RL 모두 `>=3.13.14`. 시스템은 3.12. `NeMo-RL/.venv` → `/opt/uv-cache/…cp3.13.14…` 심볼릭 링크가 **끊김**(재부팅으로 `/opt/uv-cache` 45G 소멸, `RESTORE_AFTER_REBOOT.md`) | uv 가 Python 을 내려받는다. Gym 전용 venv 를 NFS 에 두고 `UV_CACHE_DIR` 은 `/tmp`(HOME 49GB 회피) |
| uv | main1 PATH 에 없음 | `setup_nemo_rl_env.sh` 가 0.11.28 핀 설치. Gym 단독이면 최신 uv 무방 |
| 컨테이너 런타임 | main1·sub1 은 docker 불가(Backend.AI). 에이전틱은 gpu06 `alpha-eval` DinD 컨테이너(2-hop ssh, `EVAL_DOCKER_NODE.md`) | Docker 프로바이더는 로컬 데몬 전용 → SWE/TB-2 용 Gym 서버는 그 컨테이너 안에 Python 3.13 venv 로. 무도구 벤치·프로브·SDG 는 main1/sub1 CPU 에서 |
| GPU 가용 | main1 GPU 장애(09-15 03:47, `nvidia-smi` 실패) · sub1 은 iter2448 스위트 점유 | 설치·CPU 단위 테스트(`pytest tests/unit_tests`)는 지금 가능, 모델 스모크는 fleet 여유 시 |
| 벤더 사본 나이 | 08-05 (0.5.0-dev). 업스트림 v0.6.0 은 TB-2.1·Terminus-2 에이전트·health-check·`task_source` | SFT 평가용은 **별도 클론(v0.6.0)** 권장. 벤더 사본 갱신은 NeMo-RL uv.lock(uv 0.11.28)과 결합돼 있어 RL 재개 시점에 |
| vLLM 의존 | `nemo-gym[vllm]` 은 flashinfer 0.6.12 핀(`NEMO_RL_SETUP.md` #10) | 외부 vLLM(`vllm_model`)만 쓰면 Gym 에 vLLM 을 설치하지 않는다. alpha 플러그인은 지금처럼 `alpha_serve_venv` 쪽 |
| 하니스 관측성 | `harbor_agent`·`swe_agents`·`mini_swe_agent`(v1) 는 궤적 캡처 전 항목 X. `mini_swe_agent_2`·`tau2`·`simple_agent` 는 C1/C2/C4 V | TB-2 를 Gym 으로 옮겨도 턴별 증거는 Harbor 원본(`harbor_jobs_dir`)에서. 우리 프록시 지표(`steps_reasoning_only`·`think_unclosed_stop`·`mixed_content_and_tools`·`miss_rate` 무효화 규칙)는 `compute_metrics()`/캡처 후처리로 재구현 |
| 비용 구조 | SWE 500 문항 ≈ 59,500 LLM 호출(T1 전체의 3.1×), 부분 표본은 무효 규약 | Gym 이 줄여주지 않는다. `--concurrency`·`--resume`·샤드 `aggregate` 로 운용만 편해짐 |
| 미종결 think | simple_agent 경로는 미처리(§1.4) | 열린 결정(`STATUS.md` TB-2 복원 제외 플래그)과 동일 판단 필요 |
| 데이터 형식 이중화 | Responses(Gym) vs messages(우리 SFT 변환기) | 변환기 1회 작성(`chat_completions_messages_to_responses_items` 역방향 존재) |

## 6. 권고 실행 순서

1. **CPU 설치 스모크 (0.5일, GPU 불필요)**: v0.6.0 별도 클론, `uv venv --python 3.13.14`, `pytest tests/unit_tests -x`, `gym env start --resources-server example_multi_step --model-type openai_model`(교사 API 또는 `inference_provider`).
2. **추론 왕복 게이트 (fleet 여유 시 1시간)**: sub1 fleet(TOOLS=1, nemotron_v3) 에 `vllm_model` 연결, τ² airline 5문항 × repeats 2. 판정 = 캡처 요청 원문의 2턴째 `reasoning_content` 존재(restore) + 무도구 행 프롬프트의 `<think></think>` 절단(strip). 09-14 스모크와 동일 기준.
3. **유령 호출 프로브 이관 (1일)**: 33문항 데이터셋 + `single_step_tool_use_with_argument_comparison`, 현 ckpt 로 구 프로브와 수치 대조.
4. **에이전틱 티어 등가성 (2~3일)**: 같은 ckpt 로 SWE 50문항·TB-2 부분집합을 구 하니스 vs Gym 대조 → 열린 결정으로 상정(계열 단절).
5. SDG 트랙 C 는 위 1~3 이 통과한 뒤.

## 7. 참조

- 벤더 사본 `NeMo-RL/3rdparty/Gym-workspace/Gym/`: `CLAUDE.md`(구조·규칙), `README.md`(환경 표), `fern/versions/latest/pages/`(문서 원문: `model-server/vllm.mdx`, `reference/cli-commands.mdx`, `infrastructure/sandbox/docker.mdx`, `training-tutorials/offline-training-w-rollouts.mdx`, `reference/trajectory-capabilities.mdx`)
- 코드: `responses_api_models/vllm_model/app.py`(추론 왕복), `nemo_gym/responses_converter.py`(형식 변환), `responses_api_agents/harbor_agent/`(TB-2), `responses_api_agents/mini_swe_agent_2/`, `benchmarks/tau2/`, `benchmarks/nemotron_3_ultra/`
- 우리 쪽: `tokenizer_v5/chat_template.jinja` 19-28·114-140, `eval_sft/serve_alpha.sh`, `docs/SFT_BENCHMARKS.md` §3.10·§3.13·§3.14, `docs/INTERLEAVED_THINKING.md`, `docs/EVAL_DOCKER_NODE.md`, `docs/SFT_RL_DATASETS.md` §3, `project_s/NEMO_RL_SETUP.md`, `NeMo-RL/examples/configs/alpha/README.md`
- 업스트림: https://github.com/NVIDIA-NeMo/Gym (v0.6.0), 문서 https://docs.nvidia.com/nemo/gym/main/about/

## 8. 결정·진행 (2026-09-15)

사용자 결정: 채택. 버전은 v0.6.0 하나로 고정, 우리 환경은 Gym 포크 없이 `examples/alpha/gym/` 에, 범위는 §"적극의 범위" 표대로
(에이전틱 평가 이관 · 데이터 게이트 리소스 서버 · SDG 는 다음 트랜치부터 · T1/T2/T3 와 SFT 변환기는 유지).
§6 1단계 완료 — 설치·단위 테스트·sub1 CPU 롤아웃 PASS, main1 은 GPU 장애 중 Ray 즉사(노드 제약으로 기록). 이후 상태는 `STATUS.md` 와
[`../gym/README.md`](../gym/README.md) 에만 쓴다.
