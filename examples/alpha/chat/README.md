# alpha 채팅 서빙 (vLLM + LibreChat)

SFT 체크포인트와 **사람이 직접 대화**하기 위한 최소 구성. 벤치 스위트(`../eval_sft/`)와는
목적이 달라 설정이 다르다 — §3 이 그 차이의 근거다.

구성: 브라우저 → LibreChat(:8080, Node 20 + MongoDB 8.0) → vLLM OpenAI 호환 API(:8001) → alpha (main1 GPU 3, 40GB A100 슬라이스).
창 128K, KV 캐시 1,311,257 토큰, 가중치 29.9GiB.

**2026-09-09 OpenWebUI → LibreChat 교체.** 이유 둘: ① Open WebUI License 는 사용자 50명 초과 시 브랜딩 변경을
금지한다, LibreChat 은 MIT. ② OpenWebUI 0.11 이 UI 발 요청마다 내장 도구 25종을 주입해 모델이 에이전트 모드로
흘렀다 (`../docs/KNOWN_ISSUES.md` 2026-09-09). 구 런처(`run_openwebui.sh`, `init_openwebui_db.py`)는 커밋 `ca8ada7` 이력에 있다.

## 1. 기동

```bash
cd examples/alpha
bash chat/serve_chat.sh                       # vLLM  (기본: phase-2 iter500, 128K, :8001, GPU3)
bash chat/run_librechat.sh                    # UI    (기본: :8080, MongoDB 자동 기동·재사용)
```

데몬화는 호출 측에서: `nohup bash chat/run_librechat.sh > /home/work/vidsearch/tools/chat_logs/librechat_8080.log 2>&1 &`.
체크포인트를 바꾸려면 `serve_chat.sh` 첫 인자로 준다. vLLM 주소를 바꾸려면 `run_librechat.sh` 둘째 인자.

| 경로 | 내용 |
|---|---|
| `/home/work/vidsearch/tools/librechat/` | 설치본 (소스 @`8da4ae7` v0.8.8-rc2 + 빌드 산출물). `.env` 는 런처가 매번 다시 쓴다 |
| `/home/work/vidsearch/tools/librechat_data/` | MongoDB dbpath · `logs/` · `secrets`(JWT/CREDS) · `smoke_credentials`. NFS 라 컨테이너 재생성에도 남는다 |
| `/home/work/vidsearch/tools/mongodb-8.0.16/` | MongoDB 공식 tarball 바이너리 (ubuntu2404). docker 없음 |
| `chat/librechat.yaml` | 설정 정본 (리포에서 버전 관리). 왜 각 항목이 있는지는 파일 상단 주석 |

## 2. 접속·계정

포트 8080 은 이 컨테이너의 `BACKENDAI_SERVICE_PORTS` 중 `nniboard` preopen 슬롯이다. Backend.AI 앱 프록시로
열리지 않으면 SSH 터널을 쓴다:

```bash
ssh -N -L 8080:localhost:8080 main1     # ~/.ssh/config 의 Host main* (포트 2200)
```

LibreChat 은 **로그인 필수**다 (auth-off 옵션 없음). 데모용 관리자 계정은 만들어져 있다:

| 계정 | 역할 | 비밀번호 | 용도 |
|---|---|---|---|
| `admin@alpha.local` | ADMIN | `librechat_data/admin_credentials` (0600) | 사람이 쓰는 계정. 첫 로그인 후 UI 에서 비밀번호 변경 권장 |
| `smoke@alpha.local` | USER | `librechat_data/smoke_credentials` | `smoke_ui_gate.py` 전용. 게이트가 자동 생성 |

계정 추가는 `cd /home/work/vidsearch/tools/librechat && npm run create-user -- <email> <name> <username> <password> --email-verified=true`.
UI 회원가입(`ALLOW_REGISTRATION=true`)도 열려 있지만 **첫 가입자가 자동으로 ADMIN** 이 되는 규칙이 있다 (2026-09-09 실측: 스모크 계정이
ADMIN 이 돼 역할을 DB 에서 교정했다). 브라우저 회원가입이 무응답이면 요청이 서버에 닿지 않은 것이다 — Backend.AI 앱 프록시 경로에서
`POST /api/…` 가 유실될 수 있으므로 SSH 터널(`http://localhost:8080`)로 접속한다. 포트에 닿는 누구나 가입할 수 있다는 점은
OpenWebUI `WEBUI_AUTH=false` 와 같은 노출 수준이다.

## 3. 벤치 fleet 과 무엇이 다른가

| 항목 | 벤치 (`eval_sft/serve_alpha.sh`) | 채팅 (`chat/serve_chat.sh`) | 이유 |
|---|---|---|---|
| reasoning 파서 | off | **`nemotron_v3`** | UI 가 사고 과정을 접어서 보여주려면 별도 필드로 분리돼야 한다 |
| tool 파서 | `qwen3_xml` (TOOLS=1 일 때만) | `qwen3_xml` 상시 | 벤치와 같은 플래그. LibreChat 은 도구를 보내지 않으므로 발동하지 않는다 |
| 레플리카 | DP 8 (H100 fleet) | 단일 GPU | 1인 사용 |
| GPU | sub1 H100 | main1 GPU3 (40GB A100 슬라이스) | 벤치와 자원 분리 |
| `--max-num-seqs` | 기본 | 8 | 1인 사용. KV 여유는 충분하다(창의 10배) |
| 모델명 | `alpha` | `alpha-v2-sft` + 별칭 `alpha` | UI 표시는 구체적으로, 게이트 호환은 별칭으로 |

**게이트 G2 는 채팅 fleet 에서 의도적으로 FAIL 한다.** ① reasoning 파서가 `</think>` 를 본문에서 떼어간다,
② `check_gates.py` 는 `reasoning_content` 를 보는데 vLLM 0.25.1 이 내보내는 이름은 **`reasoning`** 이다.
채팅 fleet 의 검증은 G1 + G3 + `smoke_chat.sh` 다 (§6).

## 4. chat template

`hfmodel_*/tokenizer_config.json` 의 `chat_template` 필드에 내장되어 있다 — vLLM 이 자동으로 집어가므로
`--chat-template` 플래그가 필요 없고, 학습·평가·서빙이 같은 렌더러를 쓴다. 템플릿 규약(think 히스토리, tool 분기)은
`../docs/INTERLEAVED_THINKING.md`, 검증은 `../tools/verify_chat_template.py`.

## 5. LibreChat 설정이 지키는 것

UI 가 vLLM 에 보내는 본문은 `model` · `stream` · `messages` 뿐이다 (2026-09-09 PoC 요청 로그). 그래서 "안녕?" 의
프롬프트가 학습 렌더와 같은 17 토큰이다. 이를 지키는 항목:

| `librechat.yaml` | 역할 |
|---|---|
| `customParams.reasoningKey: reasoning` | vLLM 0.25.1 의 사고 과정 필드명. 기본값 `reasoning_content` 면 사고 과정이 본문에 섞인다 |
| `customParams.includeReasoningHistory: true` | tool 호출 턴에만 reasoning 을 히스토리에 복원 — 템플릿의 DSV4 분기와 동일 규칙. 일반 대화의 think 는 버린다 |
| `dropParams` | `stop`·`user`·penalty 를 보내지 않는다. temperature/top_p 는 사용자가 안 건드리면 아예 안 보내 vLLM `generation_config`(1.0/0.95) 적용 |
| `titleConvo: false` | 제목 생성(영어 메타 프롬프트)을 같은 모델에 보내지 않는다 |
| `interface.*: false` | 코드실행·파일검색·에이전트 UI 를 감춘다. 도구가 하나라도 붙으면 템플릿이 tool 시나리오로 분기한다 |
| `interface.webSearch: true` + `webSearch:` | 웹검색 도구만 **대화별 opt-in** 으로 연다 (§5.1). 토글이 꺼진 대화는 위 규칙 그대로 도구 없이 렌더된다 |

알아둘 것: 미등록 모델명의 컨텍스트 창은 LibreChat 기본 **32,000 토큰**이다 (vLLM 의 `max_model_len` 을 읽지 않는다).
더 긴 대화가 필요하면 파라미터 패널의 Max Context Tokens 로 올린다. 웹검색은 도구 결과가 크므로(호출당 1~3만 자) 검색을
많이 쓰는 대화는 이 값을 올리는 편이 안전하다.

### 5.1 도구 (웹검색, 2026-09-09)

alpha 의 도구 규약은 템플릿이 소유한다 — 도구 선언은 시스템 프롬프트의 `<tools>` XML, 호출은 `<tool_call><function=…>`,
결과는 `<tool_response>` (`../docs/INTERLEAVED_THINKING.md` §4). LibreChat 은 OpenAI `tools` 필드로 선언하고 `role=tool` 로
결과를 돌려주며, vLLM 의 `qwen3_xml` 파서가 모델의 XML 을 `tool_calls` 로 구조화한다. 그래서 LibreChat 쪽에서는 **도구를 붙이기만**
하면 된다. 켠 것은 내장 웹검색 하나다:

| 항목 | 값 | 이유 |
|---|---|---|
| 검색·본문추출 | Tavily (`searchProvider`·`scraperProvider`), 리랭커 없음 | 학습 데이터의 검색 도구 `web-search(query)` 와 벤치 하니스(`eval_sft/search_agent_eval.py`)가 Tavily 라 결과 형식이 가장 가깝다. 스크레이퍼는 LibreChat 필수 카테고리 |
| 키 | `examples/alpha/.env` 의 `TAVILY_API_KEY` | 런처가 그 줄만 뽑아 `.env` 로 옮긴다 (`source` 하지 않는다 — 파일의 다른 줄 형식 오류를 피함) |
| 사용법 | 채팅 입력창의 **Web Search** 토글을 켠다 | 대화별 opt-in. 끄면 도구 없는 렌더 |
| 모델이 보는 것 | 도구 `web_search(query, date, country, images, videos, news)` + LibreChat 의 인용 형식 지시문(영어) | 이름이 학습의 `web-search` 와 한 글자 다르고 결과가 가공 텍스트라 분포 차이는 있다. 실측으로는 호출·답변 정상 |

실측 (`smoke_ui_gate.py` §7, 2026-09-09): "web_search 도구로 오늘 코스피 지수를 검색해서 알려줘" → `web_search` 2회 호출
(`{"query":"KOSPI index today","date":"h"}`), 결과 28,500·18,068 자, 본문 467 자에 지수 7,075.74 와 출처. 프롬프트 증분 32,055 토큰
(선언 + 결과 재투입 누적). 답변이 영어로 나온 것은 도구 지시문이 영어인 영향으로 보이며 모델 쪽 관찰 사항이다.

## 6. 검증

```bash
python3 tools/emit_generation_config.py <CKPT> --check                  # G1
bash chat/smoke_chat.sh                                                 # vLLM 5항목 + UI 게이트 (LC_URL 기본 :8080)
python3 eval_sft/check_gates.py --base-url http://localhost:8001/v1     # G3 (G2 는 §3 사유로 FAIL 정상)
```

§6 UI 게이트(`smoke_ui_gate.py`)는 vLLM `/metrics` 의 `vllm:prompt_tokens_total` 을 요청 전후로 읽어 **UI 경유 프롬프트
크기**를 잰다 (≤64 토큰, 학습 렌더 17). UI 가 무엇을 보내는지 UI 를 믿지 않고 서버 카운터로 확인하는 장치다.
§7 은 같은 스크립트가 웹검색 토글을 켠 대화로 도구 경로 전체(선언 → XML 호출 구조화 → 실행 → 본문)를 확인한다.
1인 사용 중에 돌린다 — 동시에 다른 대화가 있으면 증분이 오염된다. UI 없는 fleet 은 `LC_URL=none` 으로 명시 스킵.

2026-09-09 실측 (8080 이관 직후): smoke vLLM **9/9 PASS** · UI 게이트 §6 **7/7** · §7 **8/8 PASS**, 일반 대화 프롬프트 증분 **17 토큰**.

## 7. 설치·재빌드 (컨테이너 재생성 등으로 설치본이 없을 때)

```bash
T=/home/work/vidsearch/tools
node --version                                              # v20 이상
git clone https://github.com/danny-avila/LibreChat.git $T/librechat && cd $T/librechat && git checkout 8da4ae7
npm ci --no-audit --no-fund                                 # ≈5분, node_modules 2.3 GB
npm install --no-save unrun                                 # tsdown 이 요구하는데 npm ci 가 안 깔아 준다
npm run frontend                                            # 패키지 + 클라이언트 빌드 ≈30초
curl -sL https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2404-8.0.16.tgz | tar xz -C $T && mv $T/mongodb-linux-x86_64-ubuntu2404-8.0.16 $T/mongodb-8.0.16
```

데이터(`librechat_data/`)는 그대로 남으므로 재설치 후 대화·계정이 유지된다. 버전을 올릴 때는 `librechat.yaml` 스키마
검증이 엄격하다는 점(§9)을 감안해 3080 등 다른 포트로 먼저 띄우고 §6 을 통과시킨다.

## 8. 종료

```bash
kill -TERM $(pgrep -f "node api/server/index.js")                                     # UI
kill -TERM $(pgrep -f "mongod --dbpath /home/work/vidsearch/tools/librechat_data")    # DB (보통 유지)
pkill -TERM -f "alpha_serve_venv/bin/vllm"      # GPU 메모리 회수 확인은 eval_sft/stop_fleet.sh 3,7 절 참조
```

## 9. 구축하며 밟은 함정 (2026-09-09)

| 증상 | 원인 | 대응 |
|---|---|---|
| `npm run frontend` 가 `Failed to import module "unrun"` | `tsdown` 의 선택 의존성을 `npm ci` 가 설치하지 않음 | `npm install --no-save unrun` |
| 기동 거부 `ZodError … interface.mcpServers Expected object` | 설정 스키마가 엄격. 키 하나만 틀려도 종료 | 해당 키 삭제. 새 키는 `librechat.example.yaml` 로 형식 확인 |
| 스크립트 요청이 `Illegal request` | `uaParser` 미들웨어가 브라우저 User-Agent 만 통과 | 게이트가 Chrome UA 를 흉내 낸다 |
| `npm run create-user` 가 사용자를 만들지 않음 | 미확인 | UI 회원가입 또는 `POST /api/auth/register` |
| `/api/models` 에 openAI·google 등 미설정 제공자가 섞임 | 정적 기본 목록 | 설정된 엔드포인트는 `/api/endpoints` 로 고른다 |
| 채팅 POST 응답에 본문이 없음 | 재개 가능 스트림: POST 는 `streamId` 만 반환 | 이벤트는 `GET /api/agents/chat/stream/{streamId}` (SSE, `final` 후 닫힘) |
| 긴 스크립트가 도중 `401` | 액세스 토큰 15분 만료 | 재로그인 |
| 브라우저 회원가입 무응답 (로그에 register 요청 자체가 없음) | 요청이 서버에 미도달 — 앱 프록시 경로 의심 | SSH 터널로 접속. 계정은 `npm run create-user` 로 생성 |
| `npm run reset-password` 가 멈춤 | readline 대화형 스크립트 — 인자를 줘도 확인 입력을 기다린다 | 비대화 재설정은 `bcryptjs` 해시(salt 10)를 `users.password` 에 직접 기록 (2026-09-09 실측, 로그인 확인) |
| 스모크 계정이 ADMIN | LibreChat 은 첫 가입자를 ADMIN 으로 승격 | `users` 컬렉션 role 을 직접 교정 (admin@alpha.local=ADMIN, smoke=USER) |
| `/api/config` 에 `webSearch` 가 없음 | 비인증 응답은 도구 구성을 숨긴다 | 게이트는 로그인 토큰으로 다시 읽는다 |
| 첫 턴 "안녕?" 에 영어 답변 | 프레임워크 아님 — tools 없는 vLLM 직접 호출도 4샘플 중 1건 영어 | 모델(SFT 진행 중) 문제로 기록 |
