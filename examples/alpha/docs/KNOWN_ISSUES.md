# Alpha Known Issues & Fixes — 전체 기록

`examples/alpha/CLAUDE.md`에서 2026-08-25 이관한 사고·수정 기록 전문 (최신순).
CLAUDE.md의 "함정 표"는 이 문서의 한 줄 요약이며, 새 사고는 **여기에 서사를 쓰고 CLAUDE.md 표에는 한 줄만** 추가한다.
날짜는 절대 표기. 두 스테이지 이상 지난 항목은 스테이지 경계에서 `archive/`로 이동.

## ko_chat v1/v2 폐기 — 비-reasoning 교사(gemma-4-31B)의 가짜 reasoning (2026-09-09 ✅ 폐기 결정, GLM-5.3 재합성)

**증상**: 도구가 선언되지 않은 일반 한국어 대화에서 모델 reasoning 이 100자 안팎(중앙값 ~110자)으로 짧고 얕다. 같은
조건 영어 셋 reasoning 은 2,700~7,000자 — 10~30배 차이. iter1200·1500·1800·phase-2 iter500 이 전부 동일해
phase-2 회귀가 아니라 **데이터 태생 문제**로 확인(재생 실험 `eval_sft/results/reasoning_probe/`).

**원인**: ko_chat v1/v2 의 교사가 **gemma-4-31B-it — 비-reasoning 모델**이었다. 네이티브 사고 흔적이 없으니
`translate_regen.py`·`ko_chat_sdg.py` 가 guided JSON 으로 "요청 핵심·접근·주의점을 간결하게" 쓰라고 **지시문으로
사고를 지어내게** 했다(원본 reasoning 중앙값 204자). 학생은 그 요약문을 reasoning 으로 학습했다. reasoning 모델로부터
사고를 증류하려던 목적과 정반대로, 껍데기 사고를 주입한 것이다.

**왜 폐기인가**: 짧은 요약형 사고를 학습한 모델은 한국어 질문에서 사고를 전개하지 않는다. backfill·부분 수정으로는
사고 깊이를 만들 수 없다(원천에 없음). 데이터 태생 결함이라 전량 폐기 후 재합성이 유일한 교정이다.

**대응**: ko_chat v1/v2(`alpha-SFT-KoChat-v1/v2`, `sft_packed_*/kochat_*` bins, `sdg/ko_chat/out/`)를 **REJECTED**
표기·파이프라인 제외. 원본은 기록용 보존(하드 삭제는 사용자 지시 시). NVIDIA Chat-v3 레시피로 재합성 — 교사
**GLM-5.3-Flash**(네이티브 사고를 `--reasoning-parser glm45` 로 분리 수록), 실사용자 프롬프트(lmsys/WildChat 한국어
+ 현지화 재작성), best-of-N + gemini 심판, identity 주입, 도구 비상관 슬라이스. 설계·진행은 `sdg/ko_chat_v3/README.md`
와 STATUS.md.

**재발 방지 규칙 (불변)**: reasoning 데이터의 교사는 **반드시 reasoning 모델**이다. 비-reasoning 모델에 사고를 지시문으로
생성시키지 않는다. 신규 합성 트랙은 착수 전 교사가 네이티브 사고를 내는지 확인하고, 산출물 reasoning 의 길이·언어 분포를
동일 도메인 영어 셋과 대조하는 게이트를 둔다.

## phase-2 회귀 2건 — 도구 과잉 호출·교사 정체성 오염 → iter2448 에서 교정 재실행(B안) (2026-09-09)

**배경**: 사용자가 LibreChat 으로 phase-2 iter500 을 테스트하다 정체성·품질 저하를 보고. 재생 실험(09-07 대화 재현 +
도구 25종 주입 + 프로브, `eval_sft/results/reasoning_probe/`·`results/identity_probe/`)으로 세 갈래로 분리.

**P1 도구 과잉 호출 (phase-2 신규 회귀)**: 도구가 하나라도 선언되면 무관한 질문에도 강제로 도구를 호출하고 답변이 빈다.
09-07 대화 11턴 재생(도구 25종 주입) 유령 호출률 — iter1500 **0/33**, iter1800 1/33, p2 iter500 **8/33**. 원인은
phase-2 신규 Agentic-v2(범용 API): tool_calling 첫 assistant 턴 호출률 96%·미호출 행 3%, interactive_agent 미호출 8%.
"도구가 보이면 부른다"를 학습했다. phase-1 의 SWE 도구(코드 편집)는 채팅 도구와 부류가 달라 이 규칙이 전이되지 않았다
(iter1500 이 범용 도구 25종 앞에서 0/33). 교정: 함수 호출 서브셋에 미호출 예시 2:1(When2Call `arXiv 2504.18851` —
SFT 부정 예시만 넣으면 과보수화, 비율·게이트 필수) + 한국어 도구 행. 게이트: 도구 25종 주입 재생 유령률 ≤1/33 **과**
BFCL AST(써야 할 때 쓰는가) 양방향.

**P2 교사 정체성 오염 (phase-2 신규 회귀)**: 영어 "Who are you?" 24샘플 CJ 언급이 iter1800 22 → p2 iter500 12,
"developed by Google" 2건. 원인: phase-2 신규 Chat-v2 reasoning_on assistant 턴 205개가 교사 자기귀속
("trained by Google"), 그중 87행이 정체성 질문 답. ko_chat 원본에도 5턴. 한국어는 128샘플 0건(드묾). 교정: 누출
필터를 gemma·gemini → 벤더 전체(Google·Gemini·Gemma·OpenAI·Anthropic·Zhipu·GLM·Qwen…)로 확장, 해당 행 드롭 후 재변환.

**P3 제작자 프로브 미달 (phase-1 부터, 회귀 아님)**: 제작자 프로브 30문항 iter1800 11/30(37%)·p2 iter500 10~11/30,
기준 ≥95%. iter900 이후 미측정(게이트 미배선)이라 500 iters 동안 아무도 못 봤다. 교정: identity creator 슬라이스 상향
+ `identity_probe.py` 를 `eval_new_ckpt.sh` 체인에 배선(100 iters 마다).

**판단(B안, 사용자)**: SFT 처음부터 재시작은 기각(phase-1 궤적 상승, 결함은 phase-2 신규 셋·정체성 정책에 국한).
**phase-1 최종 iter2448(`…swap_20260901_101523/checkpoints/iter_0002448`)에서 교정 phase-2 데이터 + 터미널 셋 흡수
블렌드로 한 번에 재실행**. phase-3(터미널) 런은 09-09 종료 — 오염 Chat-v2 를 리플레이하고 "항상 호출"형 터미널
데이터를 더 얹어 회귀를 심화시키므로. 프론티어 대조(웹 확인): DSV4 는 도메인별 전문가 SFT→GRPO 후 on-policy
distillation, Nemotron 3 Super 는 단일 혼합 2단계 SFT(2단계가 1단계 블렌드 85% 리플레이)→RLVR→MOPD. 도구 절제·정체성
최종 마감은 SFT 가 아니라 RL/선호(RPO)의 몫 — B 로 얻는 것은 "RL 입력용 깨끗한 SFT ckpt"이지 최종 품질이 아니다.

## SFT 데이터 일관성 검토 — chat template 기준 결함 5건·도구 영역 무사고 비중 (2026-09-09 🔶 phase-3 데이터 교정, 롤백 없음)

**계기**: 사용자 요청으로 phase-2(진행 중)·phase-3(대기) 블렌드 51 멤버의 도구호출·추론 데이터 일관성을 검토했다. 방법은 둘:
원본 400행/멤버(랜덤 시크 표본, SWE-v3 는 parquet 3파일 450행)의 구조 스캔 + packed bins 40문서/멤버의 실제 토큰열 스캔. 비율·게이트·마스킹은
전부 설계대로였고(user/system 스팬 학습 토큰 0, 비표준 role 0, reasoning 은 전 셋 `reasoning_content` 필드라 `<think>\n…</think>` 렌더 균일),
결함은 모두 **"렌더된 토큰열이 배포와 같은가"** 축에서 나왔다 — 09-01 opencode repr 결함과 같은 뿌리다.

| # | 셋 | 실측 | 비중 p1 / p2 / p3 | 배포와의 차이 | 판정 |
|---|---|---|---|---|---|
| ① | swe_v3_keepthink | `tools` 컬럼 자체가 없음(행 키 = messages·uuid·license). 전수 237,970행 중 232,057행(97.5%)이 구조화 tool_call 을 쓰고 도구명 64종. system·첫 user 어디에도 정의 없음 | 18.7 / 2.8 / 14.2 % | 배포(mini-swe-agent·OpenHands·TB-2)는 API 로 tools 를 선언 → `# Tools` 블록이 항상 있음. 학습은 "선언 없이 호출" | **높음** |
| ② | opencode_fixed | 도구 스키마 키가 `id`/`inputSchema.jsonSchema`(MCP 형) → 템플릿이 `<name></name>`·`<parameters>\n</parameters>` 빈값으로 렌더, 스키마는 `<inputSchema>` JSON 덤프. `render_check` 는 `<tool_response>` 만 봐서 미검출 | 4.3 / 0.6 / 2.5 % | 배포 tools 블록은 name·parameters 가 채워짐 | **높음** |
| ③ | chat_v2_on | metadata 에 train_turns 없음 → 전 턴 학습. 멀티 user 행 25.5%, 이전 assistant 턴이 `<think></think>` 로 렌더된 채 **학습 타깃**(학습 턴의 21.7%, reasoning 22.7% 소실) = 08-24 IF 결함 재발. `--fanout-train-turns` 는 train_turns 리스트가 있을 때만 전개하므로 플래그로도 안 잡힘 | — / 5.0 / 1.0 % | 무사고 오신호 | 중간 (phase-2 88% 소비 시점 발견) |
| ④ | identity_v2 | multi-True train_turns 23.8% 인데 `--fanout-train-turns` 미적용 (`INTERLEAVED_THINKING.md` §7 규칙 1 위반) | 0.4 % (누적 19.8 ep) | 〃 | 낮음~중간 |
| ⑤ | swe_v2_agentless | user 프롬프트 43% 행에 리터럴 `<think>`/`</think>`("reasoning 을 <think> 블록에") → user 스팬 안에 특수토큰 14/15 (bins 697 표본 중 297건). injection 가드는 im_start/im_end/eod 만 | — / 2.9 / 1.3 % | 비학습 스팬이라 loss 오염 없음. 서빙(vLLM)도 같은 토큰화라 분포는 일치 — 구조 토큰 누출로 기록만 | 낮음 |
| ⑥ | ml_ultra-v3_code_ja | Terminus **v1** 스키마 행 2.5%(`state_analysis`…, system 이 user 턴에, reasoning 0) | 0.02 % | phase-3 Terminus-2 와 상이 | 무시 |

도구 영역 추론 일관성(설계상 수용됐으나 규모가 문서에 없던 것):

| 항목 | 실측 |
|---|---|
| 도구 영역 학습 토큰 중 `<think></think>` 타깃 비중 | p1 ≈35% → **p2 43.6%** → p3 22.9% (평가 하니스는 전부 thinking ON) |
| 100% 무사고 셋 | swe_v1_r2e·swe_v2_openhands·opencode (p2 도구 영역 학습 토큰의 38.4%); 부분 무사고 swe_v3 38.5%·terminus_keephist 50.2%·cuda 64.4% |
| agentic_v2_ia 사용자 경계 reasoning 보존 학습 행 | 75%. TB-2·search 하니스는 재전달하나 채팅 서빙(`--reasoning-parser`)은 분리 후 미재전송 — DSV4 결정의 알려진 한계 |
| tool_call 인자 non-string 값 | agentic_v2_tc 28.5% 호출(bool 81·int 162·list 61) — 템플릿이 bool 을 Python `True/False` 로 렌더(upstream 동일). 서빙 파서 복원 미검증 |
| `</think>` 뒤 공백 | swe_v3 22% 만 `</think>\n\n답`, 나머지 `</think>답`; reasoning 끝 개행은 agentless 100%·swe_v3 72%·arc 50% vs 나머지 0 |
| phase-3 Terminus 이중 렌더 | 같은 3.5k 대화가 swe_v3_keepthink(제거 렌더 3.4M tok)·terminus_keephist(보존 122M tok)에 공존 — 무시 가능. phase-1 은 122M 전량을 제거 렌더로 1ep 학습 |

**판정 — 롤백 없음, phase-3 데이터 교정(사용자 결정 2026-09-09)**: ①②는 비학습 스팬(context)의 결함이라 학습된 트라젝토리 자체는 옳고,
빠진 것은 "선언된 tools 블록을 보고 호출하는" 조건부뿐이다 → 교정 데이터로 이어서 학습하면 붙는다. 롤백하려면 SWE-v3 가 iter 0 부터 든
phase-1 시작점(LC-B)까지 12일을 버려야 한다. ③은 phase-2 gradient 의 0.7%(5.0% × 14.4%)가 무사고 오신호였고 LR 은 1e-5→1.5e-6 구간이라
정상 데이터 리플레이로 되돌릴 수 있는 규모. phase-3 iteration 은 늘리지 않고(설계 비율 유지, 90 iters) TB-2·SWE-bench before/after 로 회귀가
보이면 phase-3b(swe_v3 교정본 ≈0.3ep ≈140 iters ≈12h)를 붙인다.

**교정(커밋 5cd4392, `convert_sft_128k_terminal_fix.sh` → P3 트리 실제 디렉터리 4종, 구 멤버 symlink 보존)**

| 결함 | 조치 | 교정 멤버 |
|---|---|---|
| ① | `build_swe_v3_tools_sidecar.py`: 전수 스캔(19 패밀리 = system 앞 200자 md5). 실제 하니스 6 패밀리(OpenHands 85.7k·SWE-agent 47.6k·mini-swe-agent 20.1k·opencode 16.1k·Codex 2.9k)는 도구 집합이 고정 → **합집합 선언**, 합성 11 패밀리(≈50k)는 같은 프롬프트 아래 행마다 별칭(bash_exec/run_bash/shell…)이 바뀜 → **호출된 별칭만 선언**. 스키마는 인자 서명으로 클래스 판정(18종)하고 설명문은 SWE-v2(OpenHands 4종)·OpenCode-v1(opencode 10종) 실제 스키마에서. 64 도구명 전부 판정, 표본 2,000행 미선언 0. 변환기 `--tools-sidecar` 가 주입(호출 없는 Terminus 행 불변) | swe_v3_tools_keepthink |
| ② | `normalize_tool_schema`: MCP 형·Anthropic 형을 name/parameters 로(정상 형태는 객체 동일 → 기존 셋 렌더 불변). `render_check` 에 `<tools>` 검사(선언 없는 tool_call·빈 name·빈 parameters) | opencode_tools |
| ③ | `--fanout-implicit-turns`: train_turns 없는 **일반** 시나리오 ∧ user ≥2 ∧ 마지막 user 이전 assistant 에 reasoning → 전 턴 True 로 전개(tool 시나리오는 템플릿이 보존하므로 제외, no-think 셋은 loss 등가라 제외) | chat_v2_on_fanout |
| ④ | `--fanout-train-turns` 적용 | identity_v2_fanout |
| ⑤ | `count_special_literals` 로 stats 집계(`special_literals`), 드롭은 `--drop-special-literals` 옵션 — 드롭하면 agentless 43% 를 잃고 서빙 토큰화와도 어긋나므로 기본은 기록만 | (재변환 없음) |

**산출물·게이트 (2026-09-09 14:52~15:14 KST sub1 160 core, `sft_packed_128k_terminal_pad16/`)**

| 멤버 | rows → samples | bins | real / trainable (M) | 구본 대비 | 게이트 |
|---|---|---|---|---|---|
| swe_v3_tools_keepthink | 237,970 → 236,600 (too_long 1,369, 구본 1,193 — tools 블록 ≈2.5k tok 만큼 길어짐) | 77,416 | 10,102 / 2,761 | real +5.1%, 주입 232,025행(합집합 172,355·호출별 59,670), 합성 폴백 0 | verify PASS · render 클린 · bins 60문서 tool_call 샘플 173 중 미선언 0 · 학습 토큰 think 66.2% |
| opencode_tools | 460,254 → 460,254 | 53,638 | 6,939 / 1,206 | 스키마 정규화 2.3M건, `<tools>` 빈 name/parameters 0 | verify PASS · render 클린 |
| chat_v2_on_fanout | 929,237 → 1,199,847 (implicit fan-out 행 ≈25%) | 21,252 | 2,776 / 2,115 | real +32%, 학습 토큰 think 비율 85.6% → **100%** | verify PASS · render 클린 |
| identity_v2_fanout | 86,640 → 107,076 | 133 | 16.5 / 8.7 | fan-out 행 23% | verify PASS · render 클린 |

블렌드 `sft_128k_terminal_blend_p3.yaml` 재생성(가중치 동일, 경로 4개만 교체, 90 iters 불변) → 런처 sanity 51 경로 OK → sub1 jit595 2-iter 스모크
**PASS**(15:15~15:30 KST: loss 0.828→0.832, 321 s/iter·270 TFLOP/s, max-alloc 55.5 GB, 오류 0, 데이터 캐시 667 파일 선빌드 — 직전 스모크 0.832→0.831 과 동급).
sub1 벤치 스위트(iter500 T1 48%)는 이 작업을 위해 중단됐다(사용자 결정, 벤치 세션이 TRACKING.md 에 기록).

**왜 못 잡았나**: `verify_sft_bins` 는 EOD·%16·리터럴 special-token 만, `render_check` 는 `<tool_response>` 블록만 봤다. "tools 선언이 있는가·
name/parameters 가 채워졌는가"는 어느 게이트도 묻지 않았고, fan-out 규칙은 `train_turns` 리스트의 존재를 전제했다.
**규칙(`INTERLEAVED_THINKING.md` §7 규칙 1·9·10 갱신)**: 새 셋은 `<tools>` 블록까지 눈으로 보고, tool_call 이 있는데 tools 가 없으면 사이드카로
선언을 복원하며, train_turns 가 없어도 멀티 user + reasoning 이면 `--fanout-implicit-turns` 다.

## OpenWebUI 0.11 이 UI 발 요청마다 내장 도구 25종을 주입 — 채팅 응답 이상 (2026-09-09 ✅, LibreChat 으로 교체)

- **증상**: 한국어 질문에 영어 답변("I'm your AI assistant. I can … create notes, set up automations"), 빈 답변, 존재하지 않는 도구 호출
  (`create_note` → `Tool not found`), reasoning 에 "I should use the search_knowledge_files tool" · "Looking at the system prompt".
  08-31 의 `"auto" tool choice requires --enable-auto-tool-choice…` 도 같은 원인 — OpenWebUI 는 `tool_choice` 가 아니라 **`tools` 배열**을 보냈고
  vLLM 이 `tool_choice=auto` 를 기본 적용했다 (README 의 옛 서술은 부정확).
- **원인**: `utils/middleware.py` `use_builtin_tools` = session_id 존재 ∧ `function_calling != legacy` ∧ 모델 capability `builtin_tools`(기본 True) →
  `get_builtin_tools` 25종(time 2 · ask_user · knowledge 7 · chats 2 · notes 4 · automations 5 · calendar 4, 저장된 채팅은 tasks 2 추가) →
  alpha 템플릿이 `tools | length > 0` 로 tool 시나리오 분기 → "안녕?" 프롬프트 **17 → 5,440 토큰**, 시스템 프롬프트에 영어 도구 명세.
  일반 chat·IF·identity·ko_chat 학습 데이터는 tools 0% 이므로 모델은 매 대화를 에이전트 세션으로 인식했다.
  템플릿·토크나이저·렌더러(transformers 4.57 vs 5.16)·샘플링(1.0/0.95)·eos 는 학습과 일치함을 실측으로 확인 — 문제는 이 주입 하나였다.
- **실측** (같은 6질문 × 4샘플, 서버 기본 샘플링): tools 없음 → 도구 호출 0/24 · 빈 답변 0/24 · 한글 정상. 도구 25종 주입 → 도구 호출 **9/24** ·
  빈 답변 **9/24** · "너는 누구야?" **4/4 영어**("I'm your AI assistant … create notes, set up automations" — 정체성 상실) · "세탁기 추천해줘" 3/4 `ask_user` 호출.
- **대응**: 라이선스 문제(Open WebUI License 브랜딩 조항)와 겹쳐 **LibreChat(MIT) 으로 교체**. vLLM 에 도착하는 본문이 model/stream/messages 뿐임을
  요청 로그로 확인. `customParams.reasoningKey: reasoning`(vLLM 0.25.1 필드) · `includeReasoningHistory`(tool 턴만 reasoning 복원 = DSV4 분기)로 정합.
  `chat/smoke_chat.sh` §6 UI 게이트 — vLLM `/metrics` `prompt_tokens_total` 증분 ≤64 (실측 17). OpenWebUI 로 되돌린다면 최소 조치는 모델 capability
  "Builtin Tools" 해제와 제목·태그·후속질문 생성 off. 상세: `chat/README.md`.
- **부수 관찰**: OpenWebUI 제목·후속질문 생성이 같은 모델에 영어 메타 프롬프트 + `max_tokens 1000` 을 보내 영어 제목·assistant 말투 후속질문이 생겼다.
  도구 호출 없는 빈 답변 3건(reasoning 안에 완성 답)은 tools 없는 조건 24샘플에서 재현 0 — 원인 미확정(사용자 중단 가능성).
- **교훈**: UI 가 vLLM 에 무엇을 보내는지는 UI 를 믿지 말고 **서버 카운터로 잰다**. 파서 게이트(A1)는 도구를 받아들이는지만 보지, 도구가 주입되는지는
  `prompt_tokens` 로만 보인다. 템플릿의 시나리오 분기는 도구 "선언"만으로 발동하므로 클라이언트가 몰래 붙이는 도구 하나가 학습 분포 전체를 바꾼다.

## phase-3 프리셋·데이터 결함 2건 — 차이 키만 담은 평면 프리셋 · 51 멤버 valid 블렌드 surplus 미배관 (2026-09-09 ✅, sub1 사전 스모크가 검출)

- **증상 ①**: `sft_128k_terminal_p3.yaml` 1판이 20초 만에 `validate_args: assert args.micro_batch_size is not None`. 프리셋을 "phase-2 와의 차이 6키"만으로
  썼는데 training preset 은 평면 YAML(`yaml_to_flags`)이라 상속이 없다.
- **증상 ②**: 전체 복제 후 valid 블렌드에서 `IndexError: The valid blend oversamples … requests 436 samples from GPTDataset number 2 in excess of its size 434`.
- **원인 ②**: `BlendedMegatronDatasetBuilder` 는 top-level 크기를 멤버별 `ceil(w×N)` 의 **합**으로 잡는다(51 멤버 valid 3,200 → 3,241, +1.28%). 멤버 버퍼는
  `ceil(w×N×(1+surplus))`, 기본 surplus 0.005 → 큰 멤버가 2~3 샘플 모자람. train 은 같은 부풀림이 14,400 에 분산(+0.26%)돼 통과, phase-2(49 멤버,
  valid 4,480)는 운 좋게 안 걸림. 게다가 alpha 데이터 제공자(`megatron_patch/data/__init__.py`)가 `args.mid_level_dataset_surplus` 를 `GPTDatasetConfig` 로
  넘기지 않아 YAML 값이 조용히 무시됐다.
- **대응**: ① phase-2 프리셋 64키 전체 복제(값 6개 + `no-load-rng` 만 상이, 스크립트 대조) ② 제공자 두 경로에 surplus 배관 + 프리셋
  `mid-level-dataset-surplus: 0.05`(`helpers.build_blending_indices` 재현: 0.005 부족 · 0.02 여유 0 · 0.05 통과) + `tests/test_dataset_config_surplus.py` 3건.
  3차 스모크 PASS(loss 0.832→0.831, 326 s/iter). 기록 `outputs/smoke_failed_p3_{preset,valid_surplus}_*/`, `outputs/smoke_pass_p3_sub1_jit595_20260909_122350/`.
- **교훈**: 새 프리셋은 전체 복제 후 diff 로 의도한 키만 다른지 확인. 멤버가 많고 valid 가 작은 블렌드(짧은 런)는 surplus 를 올린다 — 인자는 로그 args
  덤프로 적용 여부를 확인한다(존재≠적용). 자동 런처 전 2-iter 스모크는 생략 불가 — 둘 다 본 런 기동 직후 터졌을 결함이다.

## 교사가 만든 setup.sh 가 빌드 중 2.36 TB 파일을 써서 gpu06 디스크 고갈 (2026-09-08 ✅)

- **증상**: 시나리오 배치 sc_b1 검증(oracle·nop) 중 gpu06 `/var/lib/docker` 여유 2.1 TB → 519 MB (20분). `/tmp/containerd-mount…/app/vault.bin` 2,358,518,644,736 바이트.
- **원인**: 과제 `sc-packaging-arch-chunk-split-me…`(큰 파일 분할·병합 주제, "성능 제약" 트위스트)의 setup.sh 가
  `fallocate -l $(df 기반 여유-900MiB) /app/vault.bin` — 컨테이너의 df 가 호스트 전체를 보므로 호스트 디스크를 채우도록 설계됨.
  이미지 빌드(`RUN setup.sh`)에는 task.toml 의 `storage` 상한이 적용되지 않는다. oracle·nop 두 빌드가 동시에 실행돼 2배.
- **대응**: 빌드 kill + `docker builder prune` 로 2.4 TB 회수. 생성기에 금지 패턴(`DISK_DANGER`: df 기반 크기, 대용량 fallocate/dd/truncate)과
  프롬프트 규칙(setup.sh ≤ 50 MB·60초) 추가. **`tasks/precheck_setup.sh`**: setup.sh 를 tmpfs 512 MB·fsize 200 MB·120초 샌드박스
  컨테이너에서 먼저 실행해 실패·300 MB 초과 과제를 격리 — validate 전 필수 단계.
- **교훈**: 교사가 쓴 스크립트는 빌드 단계에서도 신뢰하지 않는다. 자원 상한이 없는 곳(이미지 빌드)에 교사 코드를 넣기 전에 상한이 있는 곳에서 먼저 돌린다.

## 합성 과제 이미지가 과제마다 940 MB — 베이스 레이어 미공유 (2026-09-08 ✅)

- **증상**: 터미널 SDG 수집 배치 1(동시 64) 3시간 동안 gpu06 `/var/lib/docker` 여유 291 → 124 GB. 사용자가 긴급히 다른 파일을 정리해 1.8 TB 확보.
- **원인**: 과제 Dockerfile 이 `python:3.12-slim` 위에 apt·pip 설치를 과제마다 반복했고, 이 호스트의 Docker(containerd 스냅샷 저장소)는
  그 레이어를 과제 간에 공유하지 않았다 → 트라이얼마다 940 MB 이미지 + 빌드 중간 레이어. 파일럿(10과제)에서는 안 보였다.
  SWE-bench 평가 이미지 500개의 실제 점유는 ≈430 GB 다(`docker images` Size 합산 2.4 TB 는 공유 레이어 중복 집계). 이 트랙의 항구 점유는 트라이얼당 ≈1 MB.
- **대응**: 베이스 이미지 `alpha-terminal-base:1` 을 한 번 빌드(`sdg/terminal/tasks/base/Dockerfile`)하고 모든 과제 Dockerfile 을
  `FROM alpha-terminal-base:1` + COPY 로 교체(원격 1,493 · NFS 3,550). 과제 이미지 = 16 KB 레이어, 빌드 1초. `common.py` 기본값 변경.
- **교훈**: 과제 수천 개를 돌리는 하니스에서는 이미지 레이어 공유를 **가정하지 말고 측정**한다 (`docker system df`, 여유 공간 추이).

## 에이전틱 노드 Docker 주소 풀 고갈 — 동시 64 수집에서 트라이얼 즉사 (2026-09-08 ✅)

- **증상**: 터미널 SDG 수집 배치 1(과제 1,240, Harbor 동시 64)에서 완료 275건 중 263건이 1분 안에 `RuntimeError`.
  job.log: `failed to create network …: all predefined address pools have been fully subnetted`. 성공 12건은 전부 보상 1 — 파이프라인은 정상.
- **원인**: Harbor 는 트라이얼마다 docker compose 프로젝트(전용 브리지 네트워크)를 만든다. DinD dockerd 의 기본 주소 풀은
  172.17~31/16 + 192.168/20 로 약 30개뿐이라 동시 64 에서 고갈됐고, 죽은 트라이얼의 네트워크(22개)가 남아 더 좁아졌다.
  파일럿(동시 5~10)에서는 드러나지 않았다.
- **대응**: `/etc/docker/daemon.json` 에 `default-address-pools` 10.100~10.102.0.0/16 (size 24, 768 네트워크) 설정 후 dockerd 재기동,
  `docker network prune -f`. `EVAL_DOCKER_NODE.md` §4 에 기록 — 컨테이너 재구축 시 daemon.json 도 복원해야 한다.
- **교훈**: 동시성을 올릴 때는 GPU·KV 뿐 아니라 컨테이너 호스트의 네트워크·포트·파일 디스크립터 한도도 같이 확인한다.

## sub1 NCCL 초기화 `munmap_chunk(): invalid pointer` — compat 스왑이 절반만 적용돼 있었다 (2026-09-07 ✅ 우회, 🔶 영구 수정은 root)

- **증상**: 터미널 SDG 트랙의 GLM-5.3-Flash 서빙(vLLM nightly 0.28.1rc1, torch 2.13 cu130, NCCL 2.29.7)이 EP+DP8 로 뜨다
  워커 8개 전부 `munmap_chunk(): invalid pointer` SIGABRT. torchrun 2-GPU `all_reduce` 하나로 재현. NCCL cu13 2.27~2.31 전 버전,
  cu129 변형 venv(nvidia-nccl-cu12 2.29.7)도 동일. `NCCL_CUMEM_ENABLE=0`·`NCCL_NVLS_ENABLE=0` 무효. 시스템 torch(2.7 cu12.8, NCCL 2.25.1)는 정상.
  `eval_sft/serve_fleet.sh` 주석의 "vLLM DP munmap 크래시 → DP1 ×N" 도 같은 증상이었다.
- **원인 (gdb 백트레이스, 단일 프로세스 `ncclCommInitAll` 재현기 `tools/glm53/nccl_initall.py`)**: NCCL 정적 cudart → `cuLibraryLoadData`
  → `libcudahook` → `libcuda.so.595.91.07` → **`libnvidia-ptxjitcompiler.so.1` 의 `__cuda_CallJitEntryPoint` 에서 free() 오류**.
  `/usr/local/cuda/compat/lib.real` 을 보면 `libcuda.so.1 → 595.91.07` 이지만 `libnvidia-ptxjitcompiler.so.1 → 570.124.06`,
  `libnvidia-nvvm.so.4 → 570.124.06` 이다. 08-29 `restore_bench_env.sh` 가 595 파일은 복사했지만 심볼릭 링크는 libcuda 만 바꿨다.
  PTX JIT 가 일어나는 로드(NCCL 2.29 커널, 그리고 아마 TE cuDNN norm)마다 570 JIT 가 595 드라이버 힙을 깨뜨린다.
- **우회 (sudo 불필요, 프로세스 범위)**: 595 JIT·NVVM 을 가리키는 심볼릭 링크만 담은 사용자 디렉토리를 `LD_LIBRARY_PATH` 앞에 둔다.
  libcuda 가 soname 으로 dlopen 하므로 이것만으로 595 가 잡힌다 (`libcudahook` 은 libcuda 경로만 강제).
  ```bash
  J=/home/work/vidsearch/tools/cuda_compat13/jit595; R=/usr/local/cuda/compat/lib.real; mkdir -p $J
  ln -sfn $R/libnvidia-ptxjitcompiler.so.595.91.07 $J/libnvidia-ptxjitcompiler.so.1
  ln -sfn $R/libnvidia-nvvm.so.595.91.07 $J/libnvidia-nvvm.so.4
  LD_LIBRARY_PATH=$J:$LD_LIBRARY_PATH …   # → ncclCommInitAll rc 0, torchrun 8-GPU all_reduce PASS (NCCL 2.29.7)
  ```
  `sdg/terminal/serve/serve_glm53.sh` 가 이를 내장. `restore_bench_env.sh` 에 jit595 생성 + (root 로 돌릴 때) 링크 정정을 추가.
- **영구 수정 (root)**: `ln -sf libnvidia-ptxjitcompiler.so.595.91.07 $R/libnvidia-ptxjitcompiler.so.1; ln -sf libnvidia-nvvm.so.595.91.07 $R/libnvidia-nvvm.so.4`.
  자동 모드에서는 sudo 가 차단돼 사용자 실행 필요.
- **함의 (미검증)**: 아래 09-04 "sub1 학습 불가"(TE cuDNN norm 에서 같은 munmap_chunk)도 같은 원인일 가능성이 크다. 링크 정정 후
  `scripts/sub1_compat_smoke.sh` 방식으로 재검증하면 sub1 이 595 compat 인 채로 학습 가능해질 수 있다.

## sub1 은 Megatron 학습을 못 돌린다 — CUDA compat 595 스왑 + Backend.AI libcudahook (2026-09-04 → 09-09 ✅ jit595 우회, 🔶 영구 수정은 root)

- **증상**: SFT phase-2 스모크(sub1, `sft_128k_full_p2` CP8+offload, 2026-09-04)가 첫 스텝의 TE `apply_normalization`
  (cuDNN norm, train.sh 의 `NVTE_NORM_FWD_USE_CUDNN=1`)에서 **전 rank `munmap_chunk(): invalid pointer` SIGABRT**.
  `PYTHONFAULTHANDLER=1` 스택: `layernorm_linear.py:208 apply_normalization` ← 첫 GDN 층 in_proj. phase-1 데이터
  (`sft_128k_mixed_blend_swap`)로도 동일 → 데이터·프리셋 무관.
- **원인**: 2026-08-29 `eval_sft/restore_bench_env.sh` 가 vLLM 0.25.1(CUDA 13 빌드)용으로 sub1 의
  `/usr/local/cuda/compat/lib.real/libcuda.so.1` 을 **570.124.06 → 595.91.07** 로 교체했다(`nvidia-smi` CUDA Version 13.2).
  학습 스택(torch 2.7 cu12.8 · TE 2.9.0 · cuDNN 9.24 LD_PRELOAD)은 595 compat 위에서 힙을 깨뜨린다. main1 은 570 그대로라 정상.
  라이브러리 버전은 두 노드가 전부 동일함을 대조로 확인(torch/TE/triton/fla/mamba/cuDNN/NCCL/causal-conv1d).
- **우회 불가**: Backend.AI 의 `/opt/kernel/libcudahook.ubuntu18.04.x86_64.so` 가 libcuda 로드를 가로채 compat 경로를 강제한다.
  `LD_LIBRARY_PATH` 앞세우기는 무시되고(595 그대로), `LD_PRELOAD=libcuda.so.570` 은 535/570/595 **3중 매핑**만 만든다.
- **대응**: G-P5 스모크는 main1 phase-2 개시 시점의 **첫 iteration 게이트**로 대체(`scripts/launch_p2_after_phase1.sh`: loss 유한·
  Traceback 없음, 실패 시 `outputs/P2_CHAIN_ALERT.txt`). 데이터 인덱스 캐시(`configs/data/.cache/sft_128k_mixed_blend_p2`, 398 파일)는
  abort 전에 완성돼 본 런이 재사용한다. sub1 에서 학습이 필요하면 symlink 를 570 으로 되돌려야 하고(root, CUDA 13 vLLM fleet 중단 —
  `restore_bench_env.sh` 로 재적용 가능). RL 단계는 **main1 학습 / sub1 롤아웃 분리 토폴로지**(사용자 확정 2026-09-06,
  NeMo-RL `generation.colocated.enabled: false`)라 충돌 없음 — 단 Ray placement 에서 policy(mcore) 워커가 sub1 에 배치되지 않도록 고정할 것.
- **원인 확정(2026-09-05)**: fleet 종료 후 `scripts/sub1_compat_smoke.sh` 로 symlink 를 570 으로 되돌리자 같은 프리셋·블렌드가
  2 iter 정상 완주(loss 0.711→0.733, 55.9GB, traceback 0) — 595 compat 단독 원인. 스크립트는 종료 시 595 를 자동 복원하므로
  sub1 에서 학습이 필요할 때마다 이 스크립트 방식(임시 570 → 복원)으로 쓴다. RL 은 노드 분리로 해당 없음(위 참조).
- **교훈**: 노드 시스템 라이브러리 변경은 STATUS·KNOWN_ISSUES 에 기록한다 — 08-29 스왑은 `chat/README.md` 함정 표에만 있었다.
  실패 스모크 로그·런 디렉터리: `outputs/smoke_failed_sub1_compat_20260904/`.
- **우회 확정(2026-09-09)**: 09-07 진단(libcuda 595 + JIT 570 혼합)이 학습 스택에도 원인이었다. `scripts/sub1_jit595_smoke.sh`(jit595 링크 디렉터리를
  `LD_LIBRARY_PATH` 앞에, sudo 불필요)로 phase-3 프리셋·블렌드 2 iter 완주(loss 0.832→0.831, 326 s/iter, 55.5 GB, traceback 0). `libcudahook` 이 강제하는 건
  libcuda 뿐이고 JIT 라이브러리는 LD_LIBRARY_PATH 를 따른다. sub1 학습이 필요하면 이 방식(symlink 변경 없음)을 쓴다 — 570 되돌리기(`sub1_compat_smoke.sh`, root)는 불필요.

## SFT 데이터 인벤토리 사고 2건 — 절단 다운로드 미검출 · used_in 오독 (2026-09-04 ✅)

### ① Agentic-v2 `tool_calling.jsonl` 절단본이 "다운로드 검증 50/50"을 통과했다

- **증상**: 로컬 파일 0.44GB·8,444행, HF 정본 14.94GB·707,052행(1.2% 만 수신). 2026-08-01 검증은 존재·JSON 파싱만 봤다.
- **대응**: HF 재다운로드 → 크기(14,941,561,688 B)·행수 일치 확인 후 교체, 절단본은 `tool_calling.jsonl.truncated_8444rows_20260904` 보존.
- **교훈**: 다운로드 검증은 HF API `/tree` 의 LFS size 와 로컬 size 대조로 한다. 나머지 49건 크기 대조는 미실시(`SFT_RL_DATASETS.md` §6).

### ② `used_in=super_v3` 를 사용 이력으로 읽어 Ultra 가 retain 한 검색 셋을 11일간 미편입

- **증상**: 2026-08-24 "Agentic-v2 는 super 셋" 판단으로 미편입. Ultra 기술보고서 16쪽 "Search Capabilities" 는 Super 의 Wikidata
  4~8홉 검색 트라젝토리(= `search` split)를 **retain** 했다고 명시한다. `used_in` 은 **생성 세대** 표시였다.
- **대응**: 2026-09-04 번복, phase-2 편입(`SFT_PHASE2_PLAN.md` §11). 교훈은 `SFT_RL_DATASETS.md` §2.3·§6.

## SWE-bench 인스턴스 이미지 500개를 무기한 쌓고 있었다 (2026-09-08 ✅)

사용자 지적: "500개 이미지를 다 따로따로 보관하는 건 docker 철학과 안 맞다".

사실 확인 — **인스턴스별 이미지 자체는 SWE-bench 공식 설계가 맞다.** 다만 3층 구조다:
base(1) → environment(~60, `pip install` 비용이 여기) → instance(500, 커밋별 체크아웃).
실측(django 인스턴스 3개): 이미지당 레이어 10개 중 **9개가 공유**, 고유는 1개.
"500개 = 500벌 사본" 이 아니다.

**문제는 보존 정책을 선택하지 않고 빠뜨린 것이다.** 구 하니스에는
`--cache_level {none,base,env,instance}`(기본 `env`)가 있었지만 설치본은 **swebench 5.0.2**
로, 그 옵션이 없다 — Docker Hub 에서 인스턴스 이미지를 받아 쓰고 **정리하지 않는 것이 기본**
동작이다. 리포트의 `unremoved_images: 500` 이 하니스가 스스로 남긴 증거였다.
`swebench eval` CLI 도 `run_evaluation` 모듈도 보존 플래그를 노출하지 않는다 → 우리가 지워야 한다.

곁들여 나온 버그: `docker_gc.sh --images` 는 `grep "^sweb.eval"` 로 걸렀는데 실제 저장소명은
`swebench/sweb.eval.x86_64.…` 다. **한 번도 매치된 적이 없어** 이미지가 지워진 적이 없다
(실측: 구 패턴 0개 / 올바른 패턴 500개).

내 판단 오류도 함께 기록한다. `docker_gc.sh` 주석에 "지우면 500 × 4.77GB 를 다시 받아야 하니
디스크보다 비싸다" 고 써 두었는데, **공유 레이어를 감안하지 않은 계산**이었다. 그 근거로 500개를
쌓아 두었고 A3 임계(300GB)를 반복해 밑돌았다. 명목 크기 합산으로 비용을 추정하면 안 된다 —
docker 는 레이어를 공유한다.

수정: `docker_gc.sh` 기본값을 **정리**로 바꾸고(`--keep-images` 로 끈다) grep 패턴을 고쳤다.
`--dry-run`·`--rotate-only` 는 이미지 정리 전에 종료하므로 실행 중에도 안전하다.

## 컨테이너 호스트 디스크 누수 — docker 가 못 보는 곳에 쌓였다 (2026-09-07 ✅)

에이전틱 노드 여유가 **592GB(08-31) → 290GB(09-07)** 로 7일간 −302GB. A3 게이트 임계(300GB)를
밑돌아 다음 실행이 막힐 상태였다.

각 실행의 A3 게이트가 남긴 시계열이 결정적이었다 — 단발 `df` 가 아니라 **같은 지점에서 잰 값**이라
추세가 보인다.

| 08-31 | 09-01 | 09-02 | 09-05 | 09-06 | 09-07 |
|---|---|---|---|---|---|
| 592GB | 396 | 395 | 361 | 304 | 290 |

**원인: `docker_gc.sh` 는 docker 객체만 본다.** 실제로 쌓이는 것은 컨테이너 안 `/opt` 의 에이전트
산출물이다 — SWE 예측·궤적(`preds_*`), Terminal 세션 기록(`runs/tb*`). docker 명령에 보이지 않아
`docker system df` 로는 잡히지 않는다.

| 구성 | 크기(09-07) | 성격 |
|---|---:|---|
| sweb.eval 이미지 500개 | 509GB | 일회성 · 의도적 보존 |
| build cache | 100GB | 누적 (prune -f 는 dangling 만) |
| **/opt/swebench** | **69GB** | **누적 · gc 대상 아님** |
| **/opt/terminalbench** | **39GB** | **누적 · gc 대상 아님** |

`/opt` 는 09-02 42GB → 09-07 108GB (5일 +66GB). gc 가 매 실행 "30GB 회수" 를 찍는 동안 같은 실행이
그보다 많이 쌓았다 — **회수 로그만 보면 관리되는 것처럼 보이는 것**이 닷새간 안 보인 이유다.

2026-09-02 에 "톱니 진동이고 누수 없음" 으로 결론냈던 것도 이 때문이다. 4분 관측 창에서 컨테이너
생성·삭제의 ±31GB 진동은 실재했지만, 그 위에 얹힌 추세는 그 창으로 볼 수 없었다. **주기 신호를
저빈도로 표본하면 추세와 진동을 가를 수 없다** — 판단하려면 A3 기록처럼 같은 조건에서 잰 장기
계열이 필요하다.

수정: `docker_gc.sh` 에 산출물 회전 추가. 실행별 디렉토리(`preds_*_iter<숫자>`, `runs/tb<해시>`)만
mtime 순 최근 N개(기본 2) 유지, 최근 120분 수정분은 진행 중으로 보고 보존. `oracle_healthcheck`
같은 기준선은 패턴에서 제외된다. 점수·리포트는 `results/<tag>/` 에 복사돼 있어 궤적을 지워도 수치는
남는다. `--dry-run` 으로 목록 확인, `--rotate-only` 로 에이전틱 실행 중에도 안전하게 회전만 수행.

실측(2026-09-07): 47GB 회수 (290 → 336GB), iter1800 실행 중 무영향.

곁다리 함정: 원격 정리 스크립트를 `ssh "..."` 안에 인라인으로 넣었더니 따옴표가 3중 중첩돼 `$d` 가
로컬에서 먼저 치환됐다(`[dry] 0GB $d`). 원격 스크립트는 **heredoc 으로 stdin 에 넘긴다**.

## `pgrep -f` 로 종료를 기다린 체인이 24시간 헛돌았다 (2026-09-05 ✅)

iter1200 스위트가 끝나면 iter1500 평가를 잇도록 체인을 걸었다. 스위트는 **09-04 08:29 에
정상 종료(rc=0)** 했는데 iter1500 은 **09-05 08:52 까지 시작되지 않았다**. 체인 프로세스는
살아 있었다 — 종료 대기 루프를 24시간 돌고 있었다.

조건이 이랬다.

```bash
while ssh sub1 "pgrep -f 'run_suite.sh' >/dev/null"; do sleep 300; done
```

`pgrep -f` 는 명령줄 **문자열**을 본다 — 자기 자신 포함. ssh 가 원격에 띄운 셸이 이렇게 뜬다.

```
3733803 bash -c pgrep -af 'run_suite.sh'      ← 패턴이 자기 명령줄에 있다
```

아무것도 안 돌아도 조건은 항상 참이다. 이 저장소 CLAUDE.md 에 `pkill -f` 가 자기 ssh 명령줄을
잡아 세션을 죽인 사고(브래킷 `[p]attern` 을 쓰라는 지침)가 이미 적혀 있었는데, 형태만 바꿔
같은 함정을 다시 밟았다.

**브래킷만으로는 부족했다.** `[r]un_suite.sh` 로 자기 매치는 막아도, 스크립트 이름을 **언급만 한**
다른 셸(로그 grep, 다른 세션의 편집 명령)이 그대로 걸린다 — main1 에서 실측하니 Claude 세션 셸
2개가 잡혔다.

`progress.sh` 가 여태 멀쩡했던 것은 설계가 아니라 우연이다. 거기엔 `| grep -v pgrep` 이 붙어
있었고 자기 매치 줄에 "pgrep" 이 들어 있어 스스로 걸러졌다. `>/dev/null` 만 붙이면 바로 깨진다.

수정: 판정을 `eval_sft/suite_running.sh` 한 곳에 모으고, 문자열이 아니라 **argv 구조**로 본다 —
argv0 이 셸이고 argv1 이 우리 스크립트 파일일 때만 실행 중이다. `bash -c "... run_suite.sh ..."`
는 argv1 이 `-c` 라 걸리지 않는다. `progress.sh` 도 이 스크립트를 호출한다.

대조 검증 (2026-09-05): 진짜 `bash …/run_suite.sh` 실행 중 → 잡음(rc=0). 이름 셋을 명령줄에
나열만 한 셸 → 무시(rc=1). sub1·main1 양쪽에서 오탐 0.

새 감시 스크립트는 직접 `pgrep` 하지 말고 이 스크립트를 부른다:
`while bash eval_sft/suite_running.sh; do sleep 300; done`

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

### ⑥ 교체 재개 첫 기동에서 LR 이 warmup 부터 재시작 (no-load-optim 상속)

- **증상**: iter 900 교체 런(`…swap_20260901_095004`) 첫 반복 901 의 LR 이 **2.17e-7**(= 2.5e-5 × 160/18,400, warmup 1스텝) — cosine 연속값 ≈1.9e-5 여야 했다.
  iteration(901)·consumed(144,160)·가중치는 승계됐는데 LR 만 0에서 시작.
- **원인**: swap preset 을 phase-1 preset 에서 파생할 때 `finetune: true` 는 지웠지만 같은 스테이지-전환용 키 **`no-load-optim: true`** 가 남았다.
  Megatron 은 이 플래그면 optimizer state 와 함께 저장된 opt_param_scheduler 상태를 로드하지 않는다 → Muon 모멘텀·스케줄러 num_steps 리셋.
  (`finetune` 은 iteration·consumed 까지 리셋, `no-load-optim` 은 optimizer·스케줄러만 — 둘 다 재개에서는 있으면 안 된다.)
- **대응**: 키 제거(9dace52) → 1 iter 만 진행한 런 중단 → `…swap_20260901_100244` 로 재기동(`no_load_optim=None`). 게이트: 첫 반복 LR ≈1.9e-5·loss ±0.05.
- **규칙**: 재개 preset 은 `load` 외에 `finetune`/`no-load-optim`/`no-load-rng` 가 **모두 없어야** 한다. 파생 시 `grep -n "^no-load\|^finetune"` 로 확인.

### ⑤ identity reasoning 에 교사 스캐폴딩 용어 "identity-facts" 누출

- **증상**: iter 900 프로브(thinking on)에서 모델 사고가 "identity-facts에 따라 CJ주식회사 … 개인 개발자 이름은 언급하지 않아야 함" 형태로
  교사 프롬프트의 사실 블록 태그명(`<identity-facts>`)을 그대로 말한다. 데이터 실측: v1 train 7,315행 중 reasoning 728턴(10%), v2 7,220행 중
  710턴(신규 creator 1,081행 중 191) 에 `identity-facts` 포함, content 는 0.
- **원인**: `identity_sdg.py` 가 교사에게 `<identity-facts>` 블록을 주고 reasoning 도 생성시키는데, 하드 게이트 `TEACHER_LEAK` 는 gemma/gemini 만 본다.
- **영향**: 답(content)은 깨끗. 사고 흔적에 배포에서 의미 없는 용어가 남는 품질 결함. phase-1 900 iters 로 이미 습관화됨.
- **대응(완료 2026-09-01, 사용자 지시)**: `sdg/identity/scrub_reasoning_scaffold.py` 로 v2 reasoning 세척(언어별 자연어 치환: identity-facts→"제 정체성 정보"/"my identity
  information"/ja·zh 동형, 섹션명 SCALE·ARCHITECTURE·ORIGIN·DEVELOPED BY·NOT DISCLOSED→규모·아키텍처·출처·개발 주체·비공개, enum lead_only/all_members→
  자연어). train 906턴·eval 59턴 치환, 잔여 0(CJK 인접 토큰까지 경계 없는 패턴), content 불변, 원본 `data_backup_scrub_prev/`. bins 재생성(111, verify PASS,
  전 bin 스캔 스캐폴딩 0·신형식 3,456) → 미세척본으로 기동된 런 중단 후 재기동. 생성기 게이트 `scaffold_leak` 신설(향후 재생성 시 원천 차단). v1 은 phase-1 이력이라 그대로.

### ④ (미수) 재개 블렌드의 가중치 재정규화 = 셔플 순열 재생성

- **증상(실행 전 발견)**: iter 1200 교체용 첫 swap yaml 이 identity 제외·opencode 부스트로 가중치를 재정규화했다. Megatron 재개는
  `consumed_samples` 인덱스만 이어 가고 블렌드 인덱스·셋별 셔플은 (가중치, 총량, seed)로 재생성되므로, 가중치가 다르면 새 순열이
  되어 앞 1200 iters 와 무관한 샘플이 뽑힌다 — SWE 1.00ep 앵커가 기대상 ≈25% 중복·≈25% 누락. 사용자 질문("순서가 보존되나")으로 발견.
- **대응**: swap yaml = phase-1 yaml 에 경로 2개만 치환(가중치 문자열 동일 검증). 교체 멤버는 전임자 가중치 승계. `SFT_PHASE2_PLAN.md` §10.4.
- **규칙**: **재개(resume)에서 데이터를 바꿀 때는 가중치를 건드리지 않는다.** 비율 변경은 스테이지 경계(finetune: true, 카운터 리셋)에서만.

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

