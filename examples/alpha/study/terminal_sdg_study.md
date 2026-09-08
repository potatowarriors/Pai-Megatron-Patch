# 터미널 에이전트 증강 데이터 — 만드는 흐름·도구·검증 스터디 (2026-09-08)

트랙 정본은 [`../sdg/terminal/README.md`](../sdg/terminal/README.md)(결정·게이트·수치)이고, 이 문서는 **공부용 설명**이다.
질문 다섯 개에 순서대로 답한다: ① 어떤 흐름으로 만드는가 ② 어떤 도구를 쓰는가 ③ 도구의 어떤 기능을 쓰는가
④ 어떤 데이터가 나오는가 ⑤ 믿을 수 있는가, 어떻게 검사하는가.

---

## 1. 흐름 — "과제를 만들고, 교사가 터미널에서 풀게 하고, 채점을 통과한 대화만 남긴다"

우리가 만드는 것은 **터미널 에이전트의 행동 데이터**다. 텍스트 생성 한 번으로 끝나는 chat 합성과 달리, 에이전트 데이터는
모델이 실제 환경(리눅스 컨테이너)과 여러 번 주고받은 **상호작용 기록**이어야 한다. 그래서 흐름은 다음 세 축으로 갈라진다.

```
[A] 과제 공급                      [B] 트라젝토리 수집                        [C] 선별·변환
 시드(공개 데이터)                   Harbor 가 과제마다 컨테이너를 띄우고        보상=1 · 완료 핸드셰이크 ·
  ├ OpenCodeReasoning 재포맷  ─┐     Terminus-2 에이전트(교사 GLM-5.3-Flash)가    JSON 유효 · 오염/특수토큰 검사
  └ 교사 시나리오 합성         ├─►   "JSON 명령 → 터미널 출력" 을 반복,     ─►   → SWE-v3 Terminus 행 jsonl
 과제 검증(oracle=1, nop=0) ─┘      끝나면 tests/test.sh 로 채점                → build_alpha_sft_idxmap --keep-history-think
                                                                               → verify_sft_bins · render_check → 블렌드
 (main1 CPU + 교사 LLM)             (gpu06 Docker ↔ sub1 vLLM, 역터널)          (main1 CPU)
```

각 단계가 하는 일:

| 단계 | 무엇을 | 왜 |
|---|---|---|
| A1 시드 | 공개 코딩 문제(OpenCodeReasoning)와 카테고리 카드(로그 분석·데이터 정리·빌드 디버그·git·셸 스크립트…) | Ultra 보고서의 시드 구성을 따른다. 저작권·라이선스가 명확한 것만 |
| A2 과제 조립 | Harbor 과제 패키지(`task.toml`·`instruction.md`·`Dockerfile`·`tests/test.sh`·`solution/solve.sh`) 생성 | 에이전트는 instruction 만 보고, 채점은 tests 가, 유효성 검증은 solution 이 담당 |
| A3 과제 검증 | 정답 절차(oracle)를 돌리면 1점, 아무것도 안 하면(nop) 0점인 과제만 채택 | "풀 수 있고, 안 풀면 안 되는" 과제만 남긴다 |
| B 수집 | Harbor `run -a terminus-2 -m openai/glm53-flash`: 교사가 매 스텝 JSON 으로 명령을 내고 터미널 출력을 받는다 | 실제 환경의 출력이 대화에 들어와야 "행동 데이터"가 된다 |
| C1 선별 | 채점 통과·완료 재확인 턴 존재·모든 턴 JSON 파싱·벤치 canary 없음·특수토큰 없음 | 실패·미완·형식 불량 트라젝토리는 학습 신호가 아니라 잡음 |
| C2 변환 | Harbor 의 전체 메시지(`all_messages`) → `{system, user, assistant(+reasoning_content), user, …}` 행 | 기존 SWE-v3 Terminus 행과 같은 스키마여야 한 블렌드에 섞인다 |
| C3 빌드·검사 | 토큰화·패킹(128k), 특수토큰·정렬 게이트, 렌더 육안 검사 | 학습기가 실제로 보는 토큰열을 확인한다 |

왜 이 순서인가: Nemotron-3 Ultra 기술보고서(§Terminal-Use Capabilities)가 정확히 이 구조다 — "시드 → Harbor + Terminus-2 안에서
교사(DeepSeek-V3.2)가 행동 주체 → 필터". 우리는 교사를 GLM-5.3-Flash 로, 규모를 1노드 이틀로 바꿨다 (근거 `docs/SFT_RL_DATASETS.md` §2.9).

---

## 2. 도구 — 무엇을 쓰고, 무엇을 안 쓰는가

| 도구 | 역할 | 어디서 도는가 |
|---|---|---|
| **vLLM** (nightly 0.28.1rc1 cu130, TP8) | 교사 GLM-5.3-Flash 서빙 (OpenAI 호환 API) | sub1 8×H100 |
| **Harbor** 0.22 (terminal-bench 2.x 프레임워크) | 과제 패키지 실행·에이전트 오케스트레이션·채점·트라젝토리 기록 | gpu06 `alpha-eval` 컨테이너 |
| **Terminus-2** (Harbor 내장 에이전트) | 모델 무관 터미널 에이전트: 프롬프트 템플릿, JSON 파싱, tmux 에 키 입력, 출력 수집 | Harbor 안 |
| **Docker-in-Docker** | 과제마다 격리 컨테이너(공유 베이스 이미지 + 과제 파일) | gpu06 |
| **litellm** | Harbor → vLLM 클라이언트 (필드 매핑 `reasoning`→`reasoning_content`) | Harbor 안 |
| **우리 스크립트** (`sdg/terminal/`) | 과제 생성·검증·수리, 수집 배치, 변환, 요약 | main1 |
| **변환기·게이트** (`toolkits/sft_data_preprocessing/`) | `build_alpha_sft_idxmap.py`, `verify_sft_bins.py`, `render_check.py` | main1 |

**NeMo Data Designer 는 이 트랙에서 쓰지 않는다.** 한국어 chat 합성(`sdg/ko_chat/`)에서는 썼다. 차이는 데이터의 성격이다.

| | Data Designer | 이 트랙 |
|---|---|---|
| 모델 | 컬럼 DAG: 샘플러(주제·페르소나) → LLM 컬럼(질문) → LLM 컬럼(답) → 판정 컬럼 | 환경 루프: 명령 → 실행 → 출력 → 다음 명령 |
| 한 행의 생성 | LLM 호출 몇 번, 외부 상태 없음 | 컨테이너 안에서 수십 스텝, 실제 파일·프로세스 상태 |
| 정답 판정 | LLM 심판(judge) 또는 규칙 | 테스트 스크립트 실행(reward 0/1) — 결정적 |
| 적합한 데이터 | 대화·지시 따르기·번역 | 에이전트 트라젝토리 |

에이전트 데이터는 "LLM 컬럼"으로 표현되지 않는다. 그래서 오케스트레이터는 Harbor 가 맡고, Data Designer 가 하던 "축 샘플링"
역할은 우리 생성기의 카테고리 카드 × 소주제 × 난이도 × 트위스트 무작위 조합이 대신한다.

---

## 3. 도구의 어떤 기능을 쓰는가

### 3.1 vLLM (교사 서빙)

| 기능 | 어떻게 쓰는가 | 왜 |
|---|---|---|
| OpenAI 호환 `/v1/chat/completions` | Harbor·생성기 모두 이 API 로 호출 | 하니스가 모델을 몰라도 된다 |
| reasoning 파서 `--reasoning-parser glm45` | 모델의 `<think>…</think>` 를 잘라 응답의 `reasoning` 필드로 분리 | 트라젝토리에 reasoning 을 따로 저장하려면 분리가 필요 |
| tool 파서 `--tool-call-parser glm47` + auto tool choice | Terminus-2 는 텍스트 JSON 이라 안 쓰지만, litellm 이 `tool_choice=auto` 를 보내도 400 이 나지 않게 | 게이트 A1/A4 상당 |
| `reasoning_effort` (low/high/max) | 수집은 high 70%/low 30%, 과제 명세 생성은 low | 스텝당 reasoning 길이 = 트라이얼 지연 |
| `response_format={"type":"json_object"}` | 시나리오 명세 생성에서 JSON 강제 | high 에서 reasoning 이 12k 토큰을 먹어 본문이 잘리던 문제 해결 |
| 접두 캐시(prefix caching) | 같은 트라젝토리의 다음 스텝은 앞 문맥을 재사용 | 스텝당 프리필 4.6초 → 1초 (TP8 채택 근거) |
| `--max-model-len 131072`, 청크 4096, 시퀀스 256 | 학습 상한 128k 와 정합, 프로파일링 OOM 회피 | |
| `/metrics` | 실행/대기 요청, KV 사용률, 접두 적중률, 선점 수 | 병목이 어디인지 숫자로 본다 |

### 3.2 Harbor + Terminus-2 (과제 실행·수집)

| 기능 | 어떻게 쓰는가 | 왜 |
|---|---|---|
| 과제 패키지 형식 | `task.toml`(시간 상한·자원) · `instruction.md` · `environment/Dockerfile` · `tests/test.sh`(→`/logs/verifier/reward.txt`) · `solution/solve.sh` | TB-2 과제와 같은 계약이라 평가·수집이 같은 코드로 돈다 |
| `-p <dir>` 로컬 데이터셋 | 우리가 만든 과제 디렉토리 묶음을 그대로 실행 | |
| 에이전트 `oracle` / `nop` | 과제 검증: 정답 절차 실행 / 아무것도 안 함 | 과제 유효성 판정 |
| 에이전트 `terminus-2` + `-m openai/glm53-flash` | 수집: 교사가 행동 주체 | Ultra 와 같은 하니스 |
| `--ak parser_name=json` | 학습 데이터 스키마와 같은 JSON 형식 | xml 형식은 다른 템플릿 |
| `--ak store_all_messages=true` | 모델이 실제로 주고받은 메시지 전부를 `result.json` 에 보존 | 변환기의 정본 입력 |
| `--ak interleaved_thinking=true` | 앞 턴 reasoning 을 다음 요청에 되돌려 보냄 | "학습 조건부 = 배포 조건부" (보존 렌더 결정) |
| `--ak max_turns=30`, 과제 `agent.timeout_sec` | 스텝·시간 상한 | 128k 안에 들어오고 요약 압축 발동을 막는다 |
| `-n 64~96`, `-k 1` | 동시 트라이얼, 과제당 1회 | 처리량 · 재시도는 실패 과제만 나중에 |
| 완료 핸드셰이크 | `task_complete:true` 뒤 "정말 완료?" 재확인 턴 | 학습 행의 마지막 두 턴을 이룬다 |
| 출력 절단(10KB/스텝), 요약 압축 | 긴 출력은 잘리고, 문맥 초과 시 요약 | 요약된 트라젝토리는 선형 이력이 아니라 **제외** |

### 3.3 우리 스크립트 (`examples/alpha/sdg/terminal/`)

| 파일 | 기능 |
|---|---|
| `tasks/common.py` | 과제 디렉토리 작성기(공유 베이스 Dockerfile), 교사 호출(재시도), JSON/코드 블록 추출 |
| `tasks/gen_opencode_tasks.py` | 문제+정답 → 교사가 입력 생성기(gen.py) 작성 → 정답 코드 실행으로 기대 출력 → 숨은 테스트 ≥6 + 공개 예제 2 |
| `tasks/gen_scenario_tasks.py` | 카테고리 카드 → 교사가 과제 명세 JSON(지시문·자료·Dockerfile 추가·test.sh·solve.sh) → 구문·금지어 검사 |
| `tasks/validate_tasks.sh` | Harbor oracle/nop 실행 → `VALID.txt`/`INVALID.txt`, 채택본 복사 |
| `tasks/repair_scenario_tasks.py` | 떨어진 과제의 실패 증거(테스트 출력·빌드 로그)를 교사에게 되먹여 명세 수정 → 재검증 |
| `collect/run_collect.sh` | 과제 묶음 전송 → Harbor 수집 → 회수 → 요약 → 변환 |
| `convert/traj_to_terminus.py` | `all_messages` → SWE-v3 Terminus 행 + MANIFEST(`conversion_note`) |
| `pilot/summarize_pilot.py` | 성공률·형식 유효율·턴·토큰·reasoning 분포 |
| `gates/gate_t0.py` | 교사 서버 준비 게이트 7항목 |

---

## 4. 어떤 데이터가 나오는가

### 4.1 형태

한 행 = 한 트라젝토리 = **Terminus-2 프로토콜의 멀티턴 대화**. 역할은 `system, user, assistant, user, assistant, …, assistant`.

```
system   : Terminus-2 지시문 2,832자 — "명령을 JSON {analysis, plan, commands[{keystrokes,duration}], task_complete} 로 내라"
user     : "Task Description:\n<과제>\n\nCurrent terminal state:\nroot@…:/app# "
assistant: reasoning_content = "We need act commands, inspect problem…"
           content = {"analysis": "…", "plan": "…", "commands": [{"keystrokes": "cat problem.md\n", "duration": 0.1}, …], "task_complete": false}
user     : "New Terminal Output:\nroot@…:/app# cat problem.md\n<실제 출력>"
…        (반복)
assistant: {"…", "commands": [], "task_complete": true}
user     : "Current terminal state: … Are you sure you want to mark the task as complete? …"
assistant: reasoning_content = <정답성 재검토>, content = {"…", "commands": [], "task_complete": true}
```

학습 렌더(`--keep-history-think`)에서는 assistant 턴마다 `<think>reasoning</think>{JSON}` 이 정답이 되고, 앞 턴의 think 가 문맥에 남는다.
이것이 SWE-v3 의 기존 Terminus 행(기본 렌더는 마지막 턴만 think)과 다른 점이며 사용자 결정이다(README §1.1).

### 4.2 과제의 두 원천

| 원천 | 과제 예 | 트라젝토리 성격 |
|---|---|---|
| OpenCodeReasoning 재포맷 (`oc-*`) | "/app/problem.md 를 읽고 /app/solution.py 를 작성, samples/ 로 확인" | 4~8턴, 파일 읽기 → heredoc 으로 코드 작성 → 예제 실행·diff → 제출 |
| 교사 시나리오 (`sc-*`) | "telemetry.bin 을 512 KiB 로 split, SHA256SUMS 작성, 재조립해 바이트 동일 확인" · "dotenv 우선순위 버그 수정" · "깨진 Makefile 고쳐 바이너리 생성" | 8~25턴, 탐색·진단·수정·검증 |

### 4.3 지금까지의 수치 (2026-09-08)

| 항목 | 값 |
|---|---|
| 파일럿(TB-2 10과제, 측정 전용) | 형식 유효율 99.1~100%, 성공 55~70%, 트라이얼당 15~17턴·24~31k 출력 토큰 |
| 재포맷 과제 생성 | 1,500문제 → 1,240과제(83%), 과제당 교사 토큰 1.8k |
| 시나리오 과제 생성 | 48건 → 명세 38 → Harbor 채택 11 (29%), 수리 루프 적용 중 |
| 수집 배치 1 (effort high, 동시 64) | 첫 179건: 성공 144(80%), 시간 초과 35, 시간당 ≈300건 |
| 행 길이 | 파일럿 최장 35.9k 토큰, 평균 ≈22k (128k 상한 안) |

저장 위치: `tools/glm53/tasks/<배치>/`(과제), `tools/glm53/collect/<태그>/rows/*.jsonl`(행). 수집이 끝나면
`/home/work/Datasets/LL_datasets/posttraining/SFT/alpha-SFT-Terminal-v1/` 로 모아 bins(`terminal_terminus2_synth`)를 만든다.

---

## 5. 믿을 수 있는가 — 검증은 층마다 있다

"검증 없이 진행하지 않는다"(루트 CLAUDE.md 검증 규칙)는 이 트랙에서 네 층으로 구현돼 있다.

### 5.1 과제 층 — 과제 자체가 올바른가

| 검사 | 방법 | 걸러낸 것 |
|---|---|---|
| oracle | Harbor `-a oracle`: `solution/solve.sh` 실행 후 `tests/test.sh` → 반드시 1 | 정답 절차와 테스트가 어긋난 시나리오 11/38 |
| nop | Harbor `-a nop`: 아무것도 안 한 컨테이너에서 test.sh → 반드시 0 | "안 풀어도 통과"·뒤집힌 테스트 3/38 |
| 빌드 | Dockerfile 빌드 실패 = 즉시 탈락 | setup.sh 가 없는 파일 참조 7/38 |
| 재포맷 과제의 로컬 재현 | 숨은 테스트의 기대 출력을 **정답 코드를 실제로 실행**해 만든다. 정답이 실패하는 입력은 버린다 | 생성기 오류가 과제를 오염시키지 않음 (스모크 7/7 oracle 1·nop 0) |
| 구문·금지어 | `bash -n`, `git push/pull/fetch/clone`·네트워크 명령·canary 문자열 금지 | |

### 5.2 트라젝토리 층 — 이 대화를 학습해도 되는가

| 검사 | 기준 |
|---|---|
| 채점 | `verifier/reward.txt == 1` (테스트 통과) — 교사 심판이 아니라 **실행 결과** |
| 완료 핸드셰이크 | 마지막 assistant 턴에 `task_complete:true` (시간 초과 뒤 우연히 채점만 통과한 것은 제외) |
| 형식 | 모든 assistant 턴의 JSON 이 `analysis/plan/commands` 를 갖고 파싱됨 |
| 오염 | TB-2 canary GUID 검출 시 드롭 (벤치 과제 트라젝토리는 학습 금지) |
| 특수토큰 | 본문에 `<|im_start|>`·`<|im_end|>`·`<|endoftext|>` 리터럴이 있으면 드롭 |
| 요약 압축 | Terminus-2 가 문맥을 요약한 트라젝토리(선형 이력 아님)는 제외 |
| 길이 | 렌더 후 128k 초과 행 드롭 |

### 5.3 데이터 층 — 학습기가 보는 토큰열이 맞는가

| 검사 | 방법 |
|---|---|
| 변환 통계 | `build_alpha_sft_idxmap.py` 의 drops(`too_long`, `injection`, `render_error` …) 가 MANIFEST 에 남는다. 파일럿: 드롭 0 |
| `verify_sft_bins.py` | EOD 오염 0, %16 정렬, 리터럴 special-token 0 |
| `render_check.py` (규칙 9) | 실제 렌더 문자열을 눈으로 확인: `<think>` 가 모든 assistant 턴에 있는지, 봉투 흔적 없음 |
| 변환기 유닛 테스트 | `tests/test_alpha_sft_idxmap.py` 42/42 — `--keep-history-think` 가 Terminus 행에서 think 를 보존하고 도구 시나리오는 불변 |

### 5.4 평가 층 — 학습 뒤 능력이 올라갔는가

Terminal-Bench 2.0(89과제 × 8회, Harbor + Terminus-2) 을 학습 전후로 비교하고 LogicKor·IFEval 무회귀를 본다 (`docs/SFT_BENCHMARKS.md` §3.11).
이 데이터로 학습한 모델이 TB-2 에서 오르지 않으면 데이터가 아니라 우리의 가정을 의심한다.

### 5.5 아직 없는 것 (정직한 목록)

- Ultra 의 7신호 휴리스틱(금지 git 명령·편집-테스트 무한반복·탐색만 하고 편집 없음·도구 호출 불량률·디버그 잔재·편집 후 미테스트·제출 무결성)은 **미구현**. P3 변환 단계에서 추가한다.
- 과제 중복·유사도 제거(같은 문제 변형)와 난이도 분포 관리가 없다.
- 시나리오 과제의 "테스트가 지시문을 정확히 반영하는가"는 oracle/nop 으로만 본다. 지시문이 모호해도 정답 절차가 통과하면 채택된다.

---

## 6. 용어

| 용어 | 뜻 |
|---|---|
| 트라젝토리 | 과제 하나를 푸는 동안의 전체 상호작용 기록 = 학습 행 하나 |
| 트라이얼 | Harbor 가 과제 하나를 한 번 실행한 것 (`-k` 로 여러 번 가능) |
| oracle / nop | Harbor 에이전트: 정답 절차 실행 / 아무것도 안 함 |
| 핸드셰이크 | `task_complete:true` 뒤 "정말 완료?" 재확인과 최종 응답 |
| 보존 렌더 | 모든 assistant 턴의 reasoning 을 학습 토큰에 남기는 렌더 (`--keep-history-think`) |
| canary | 벤치 데이터에 박힌 "학습 금지" 표식 문자열 |

더 읽을 것: `sdg/terminal/README.md`(정본), `docs/SFT_RL_DATASETS.md` §2.9(Ultra 레시피 표), `docs/INTERLEAVED_THINKING.md`(think 규약),
`docs/KNOWN_ISSUES.md` 2026-09-07/08(compat JIT 불일치·Docker 주소 풀 사고).
