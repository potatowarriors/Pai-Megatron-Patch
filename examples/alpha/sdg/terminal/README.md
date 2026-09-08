# 터미널 에이전트 SFT 데이터 합성 (Terminus-2 형, 교사 GLM-5.3-Flash)

이 트랙의 정본. 상태 한 줄은 [`docs/STATUS.md`](../../docs/STATUS.md), 데이터 실측·Ultra 레시피는
[`docs/SFT_RL_DATASETS.md`](../../docs/SFT_RL_DATASETS.md) §2.9, 평가 경로는 [`docs/SFT_BENCHMARKS.md`](../../docs/SFT_BENCHMARKS.md) §3.11.

## 0. 문제와 목표

Terminal-Bench 형(Terminus-2 스키마) 학습 데이터가 phase-1 블렌드의 ≈0.3%(SWE-v3 안 ≈3.5k행, 전부 GitHub 이슈 과제)뿐이고
phase-2 신규 23셋에는 0이다. Ultra 는 Harbor + Terminus-2 안에서 DeepSeek-V3.2 를 행동 주체로 ≈370K 대화를 합성해 SFT 했다(미공개).
같은 방식을 H100×1노드(sub1) + GLM-5.3-Flash open weight 교사로 축소 재현한다.

## 1. 사용자 결정 (2026-09-07)

| 결정 | 내용 |
|---|---|
| 교사 | **GLM-5.3-Flash** (321B 총량 / 18B 활성, FP8 305.8 GiB, MIT, TB 2.1 공개 84.3). 사용자 제공 vLLM 레시피의 플래그 승계 |
| 규모·투입 방식 | P1 파일럿의 **처리량 실측 후** 결정 |
| think 렌더 | **보존 렌더** — Terminus 행의 모든 assistant 턴에 reasoning 을 남긴다 (DeepSeek V4·GLM-5.3 방식). 합성 데이터와 SWE-v3 Terminus 3.5k행을 같은 규약으로 변환. 템플릿은 불변, 변환기 명시 kwarg(`--keep-history-think`)로 렌더하고 평가 하니스는 reasoning 재전달(`interleaved_thinking=true` + `chat_template_kwargs`)로 학습 조건과 맞춘다. 근거는 §1.1 |

### 1.1 think 렌더 결정의 근거

`docs/INTERLEAVED_THINKING.md` 의 판정식은 tool 시나리오(tools 선언 ∨ tool_calls ∨ role=tool)에서만 히스토리 think 를 보존한다.
Terminus 는 터미널 출력을 role=user 평문으로 주입하고 명령을 content 안의 JSON 으로 쓰므로 구조 신호가 없어 일반 chat 렌더를
받는다. 실측: reasoning 보유 SWE-v3 Terminus 행(57 assistant 턴) 기본 렌더 = 빈 think 56 / 보존 1, `truncate_history_thinking=False`
= 보존 57. 기본 렌더로 학습하면 중간 턴의 정답이 빈 think 가 되어 "중간 스텝에서는 생각하지 않는다"를 배운다. fan-out 은 정확하지만
턴 수 제곱 비용(중앙값 67턴). 보존 렌더는 하니스가 reasoning 을 되돌려 보내면(Terminus-2 `interleaved_thinking`) "학습 조건부 = 배포
조건부" 원칙을 만족한다. 비용은 문맥 길이 — 128k 초과 행은 드롭되므로 교사 `reasoning_effort` 와 스텝 상한으로 제어한다.

## 2. 파이프라인

```
시드 (OpenCodeReasoning 재포맷 · OpenMathReasoning 재포맷(정답 파일형, 교사 호출 0) · 교사 시나리오 합성)
 → 과제 조립: Harbor task dir (task.toml / instruction.md / environment/Dockerfile / tests/test.sh / solution/solve.sh)   [main1 CPU]
 → 과제 검증: oracle(solve.sh) reward 1 AND 빈 환경 reward 0 인 과제만 채택                                          [gpu06 Harbor]
 → 트라젝토리 수집: harbor run -a terminus-2, 교사 = GLM-5.3-Flash(sub1 vLLM :8300, 역터널 8299), 과제당 k회          [gpu06 ↔ sub1]
 → 성공 판정(reward=1) + 7신호 필터 + TB-2 오염 제거
 → SWE-v3 Terminus 행 스키마 jsonl + MANIFEST(conversion_note) → build_alpha_sft_idxmap --keep-history-think → verify_sft_bins + render_check
 → 블렌드 멤버 편입 → SFT 보정 스테이지 → TB-2(89과제×8) before/after
```

## 3. 단계와 게이트

| 단계 | 할 일 | 게이트 |
|---|---|---|
| **P0 서빙 환경** | venv `tools/glm_serve_venv`, 가중치 `models/GLM-5.3-Flash`, `serve/serve_glm53.sh`, 역터널 `serve/glm_tunnel.sh`, gpu06 `docker_gc.sh` | G-T0 `gates/gate_t0.py`: models·reasoning 분리·tools 수용·Terminus JSON 유효율·처리량·GPU 메모리 |
| **P1 파일럿** (TB-2 10과제, 측정 전용, 학습 미투입) | `harbor run -d terminal-bench@2.0 -l 10 -a terminus-2 -m openai/glm53-flash --ak parser_name=json --ak 'trajectory_config={"raw_content":true}' --ak store_all_messages=true --ak interleaved_thinking=true -k 2`; 변환기 `traj_to_terminus.py` v0 | G-T1: 형식 유효율 ≥95%, 성공률, 과제당 토큰·벽시계·스텝, 스텝당 reasoning 토큰 분포, 시스템 프롬프트 SWE-v3 대비 diff 0, 변환 드롭 0, render_check 봉투 0 |
| **P2 과제 합성** | (a) OpenCode/OpenMath 재포맷 (b) 교사 시나리오 합성(카테고리 카드). 베이스 이미지 5~10종 공유. canary GUID + TB-2 instruction n-gram 중복 제거 | G-T2: 200과제 표본 유효 비율(oracle PASS ∧ noop FAIL), 카테고리 분포, 교사 성공률 20~80% |
| **P3 수집·필터·변환** | k=2~4 시도, reward=1 채택(실패는 RL용 보관). 필터: 금지 git 명령·편집-테스트 무한반복·탐색만·JSON 파싱 실패율·디버그 잔재·편집 후 테스트 미실행·제출 무결성 + 스텝 상한·128k | G-T3: 필터별 탈락, 변환 드롭률, 실토큰·평균 턴, reasoning 보유율 |
| **P4 투입·평가** | 블렌드 멤버 `terminal_terminus2_synth`, phase-2 완주 ckpt 위 보정 스테이지 | TB-2 전량 before/after, LogicKor·IFEval 무회귀 |

## 4. P0 런북 (sub1)

```bash
# 1) 설치 기록 (2026-09-07): python3 -m venv tools/glm_serve_venv && pip install -U pip uv
#    uv pip install -U vllm --torch-backend=cu130 --extra-index-url https://wheels.vllm.ai/nightly/cu130 --prerelease=allow
#    uv pip install "flashinfer-python>=0.6.17"
#    → vllm 0.28.1rc1.dev485+g58ad1f3b8, torch 2.13.0+cu130 (로그 tools/glm53/install.log)
#    가중치: hf download zai-org/GLM-5.3-Flash --local-dir models/GLM-5.3-Flash --max-workers 16 (62 파일, 308 GB)
# 2) 서빙
nohup bash sdg/terminal/serve/serve_glm53.sh > /home/work/vidsearch/tools/glm53/serve_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# 3) 게이트 (준비 대기 포함)
python3 sdg/terminal/gates/gate_t0.py --wait 3600 --out /home/work/vidsearch/tools/glm53/gate_t0.json
# 4) 역터널 (컨테이너:8299 → sub1:8300)
bash sdg/terminal/serve/glm_tunnel.sh start && bash sdg/terminal/serve/glm_tunnel.sh status
```

## 5. 결과 기록

### P0 — G-T0 PASS (2026-09-07 22:10 KST, sub1, 3차 기동)

| 항목 | 값 |
|---|---|
| 서빙 | vLLM 0.28.1rc1.dev485+g58ad1f3b8 (nightly cu130), torch 2.13.0+cu130, NCCL 2.29.7, flashinfer 0.6.18 |
| 구성 | EP + DP8, max-model-len 131072, 프리필 청크 4096, 랭크당 시퀀스 64, KV BF16, 텍스트 전용 |
| 가중치 | 랭크당 50.9 GiB, 로드 261초(콜드) / 32초(웜) |
| 컴파일·캡처 | 그래프 캡처 247초 + 14초 |
| KV 캐시 | 랭크당 11.08 GiB, GPU 메모리 8장 전부 71.7 GB |
| T0-2 reasoning | PASS — 필드명은 `reasoning`(nightly), `reasoning_content` 아님. litellm 1.100.0 은 `reasoning_content` 로 매핑 |
| T0-3 tools | PASS — glm47 파서가 `bash` tool_call 생성 |
| T0-4 Terminus JSON | 8/8 유효, 완성 180~479 토큰 |
| T0-5 처리량 | 단일 스트림 64.5 tok/s, 동시 32 요청 집계 1,251 tok/s (동시성 상향 시 재측정) |
| T0-7 effort | low: reasoning 0자·441토큰, max: reasoning 3,067자·1,620토큰 — `reasoning_effort` 로 길이 제어 가능 |

**기동 사고 2건** (상세 `docs/KNOWN_ISSUES.md` 2026-09-07):
1. NCCL 초기화 `munmap_chunk` — compat 디렉토리의 PTX JIT 컴파일러·NVVM 이 570 인 채로 libcuda 만 595 (08-29 스왑 반쪽 적용).
   595 링크 디렉토리 `tools/cuda_compat13/jit595` 를 `LD_LIBRARY_PATH` 앞에 두는 것으로 우회(`serve_glm53.sh` 내장). 영구 수정(root 링크 정정)은 사용자 예정.
2. 프로파일링 OOM — 기본 청크 8192 에서 EP+DP8 은 8×8192 토큰이 한 랭크 전문가로 모여 FlashInfer CUTLASS FP8 MoE 작업공간이 잔여 28 GiB 초과.
   청크 4096·시퀀스 64 로 해결.

**nightly 함정**: 응답 메시지의 reasoning 필드가 `reasoning` 이다 (`reasoning_content` 는 입력 메시지에서만 하위호환 rename).
게이트·변환기는 두 이름을 모두 읽는다.

### P1 — 파일럿 1 (TB-2 10과제 × 2회, effort 기본=max, 동시 5) — G-T1 형식 PASS, 비용·지연 실측 (2026-09-07 22:13 ~ 00:06 KST)

측정 전용(TB-2 = 벤치, canary GUID) — 트라젝토리는 학습에 넣지 않는다. 수치 정본 `tools/glm53/pilot/pilot1/summary.json`.

| 항목 | 값 | 판정 |
|---|---|---|
| 형식 유효율 (assistant JSON) | **334 / 337 = 99.1%** | ≥95% PASS |
| reasoning 보유 턴 | 337 / 337 = 100% | litellm `reasoning_content` 매핑 확인 |
| 과제 성공률 | 11 / 20 = 55% (시간 초과 12건 중 4건은 초과 후에도 채점 통과) | 참고: 공개 84.3 은 시간 제한 없는 조건 |
| 트라이얼당 턴 / 출력 토큰 / 누적 입력 토큰 | 평균 16.9 (중앙값 14) / 30.6k / 500k | reasoning 재전달로 문맥이 매 스텝 누적 |
| 벽시계 | 평균 22.9분 (성공만 22.1분, 범위 1.7~63.5분) | 동시 5 → 시간당 ≈ 11 트라이얼 |
| 스텝당 reasoning | 중앙값 2,577자 · p90 14,352자 · 최대 59,468자 | effort max 가 스텝당 3~4분을 먹는 원인 |
| 문맥 요약 발동 | 6 / 20 트라이얼 (Terminus-2 summarization, 131k 상한) | 요약된 트라젝토리는 선형 이력이 아니다 |

**변환·빌드·렌더 (규칙 9)**: `convert/traj_to_terminus.py` → reward 1 이고 마지막 턴 `task_complete` 인 7행 (드롭: reward 0 = 9, 시간 초과 후
채점 통과라 완료 핸드셰이크 없음 = 4) → `build_alpha_sft_idxmap.py --keep-history-think`: 드롭 0, 실토큰 158,287 · 학습 93,042, 최장 행 35,856 토큰 →
`render_check` 봉투 0, `<think>` 14/14·7/7 (모든 턴 보존). 학습 스팬은 `<think>reasoning</think>{JSON}` 형태.

**SWE-v3 와의 형식 차이 (배포 하니스 = Harbor 를 따른다)**: Harbor Terminus-2 템플릿 머리 2,832자 vs SWE-v3 system 2,836자 — 33행 한 문장만 다름
(`- You must end every command with a newline (\n) or it will not execute.` vs `- Most bash commands should end with a newline (\n) to cause them to execute`).
첫 user 턴 라벨 `Task Description:` / `Current terminal state:` vs SWE-v3 `Task:` / `Current Terminal Screen:`. 이후 프로토콜(`New Terminal Output:`,
완료 재확인 턴, JSON 스키마)은 동일. 정본 프롬프트는 `convert/swe_v3_terminus_system_prompt.txt`.

**함의·다음 레버**: 실패는 처리량이 아니라 **스텝당 지연**(단일 스트림 64 tok/s × reasoning 10~15k 토큰)이다. P3 설계에 반영할 것:
① `reasoning_effort=high` 로 스텝당 reasoning 축소(파일럿 2 에서 실측) ② 동시 트라이얼 5 → 10~30 ③ 스텝 상한·128k 예산으로 요약 발동 회피
(요약된 트라젝토리는 제외 또는 `linear_history`) ④ 시간 초과 후 채점 통과 4건은 완료 핸드셰이크가 없어 드롭 — 수집 시 시간 상한을 과제 난이도에 맞춘다.

### P1b — 파일럿 2 (같은 10과제 × 1회, `reasoning_effort=high`, 동시 10) — effort·동시성 레버 실측 (2026-09-08 00:09 ~ 01:12 KST)

| 항목 | 파일럿 1 (max, 동시 5) | 파일럿 2 (high, 동시 10) |
|---|---|---|
| 형식 유효율 | 334/337 = 99.1% | **151/151 = 100%** |
| 성공률 | 11/20 = 55% | **7/10 = 70%** (llm-inference·write-compressor 가 성공으로 전환, pytorch-model-cli 는 실패로) |
| 트라이얼당 턴 / 출력 토큰 / 누적 입력 | 16.9 / 30.6k / 500k | 15.1 / 23.7k / 364k |
| 벽시계 평균 | 22.9분 | 19.3분 |
| 스텝당 reasoning 중앙값 / p90 | 2,577 / 14,352자 | 2,000 / 11,977자 |
| 문맥 요약 발동 | 6/20 | 2/10 (둘 다 60분 상한 과제) |
| 서버 | 안정, 동시 10 에서 여유 | 〃 |

수치 정본 `tools/glm53/pilot/pilot2_high/summary.json`. n 이 작아 성공률 차이는 참고치지만, effort high 는 reasoning 을 ≈20% 줄이면서 성공률을
떨어뜨리지 않았다. 시간 초과는 두 파일럿 모두 같은 과제(gpt2-codegolf·largest-eigenval·reshard-c4-data·winning-avg-corewars)에서 났다 —
과제 난이도와 시간 상한의 문제이지 effort 의 문제가 아니다.

**수집 설정 확정 제안 (P3)**: `reasoning_effort=high` · Harbor 동시 24 · Terminus-2 `max_turns` 40 · 합성 과제 시간 상한 30분 · 요약 발동
트라젝토리 제외(`linear_history` 또는 드롭) · 128k 초과 행 드롭. 처리량 추정: 트라이얼당 ≈19분 × 동시 24 → 시간당 ≈75 트라이얼, 성공 55~70%
와 핸드셰이크·요약 필터 후 **하루 ≈900~1,000 트라젝토리**. 첫 합성 배치에서 실측 후 규모 확정.

### P2·P3 진행 기록 (2026-09-08)

**과제 원천 3종과 수율**

| 원천 | 생성기 | 교사 비용/과제 | 유효 과제 수율 | 비고 |
|---|---|---|---|---|
| OpenCodeReasoning 재포맷 (`oc-*`) | `tasks/gen_opencode_tasks.py` | ≈1.8k 토큰 (입력 생성기) | 83~86% | 숨은 테스트 = 정답 코드 실행. HARD/VERY_HARD 제외(배치 1 실측 성공 32%) |
| OpenMathReasoning 재포맷 (`om-*`) | `tasks/gen_openmath_tasks.py` | 0 | 100% (스모크 20/20 oracle 1·nop 0) | 정수·소수·분수 정답만(56%), 채점기 동치 검사 9/9. `/app/answer.txt` 한 줄 |
| 교사 시나리오 (`sc-*`) | `tasks/gen_scenario_tasks.py` (effort low + JSON 강제) → `precheck_setup.sh` → `validate_tasks.sh` → `repair_scenario_tasks.py` | ≈5k + 수리 5k | 명세 80% × 검증 30~45% (+수리 회수 30%) | 다양성 담당, 비중 ≈5% |

**수집 배치 (Harbor terminus-2, GLM-5.3-Flash TP8)**

| 배치 | 과제 | 설정 | 결과 |
|---|---|---|---|
| oc_b1_high | 1,240 | 동시 64 · 상한 15분 · effort high | 성공 ≈70% (easy 97%, medium 76%, hard 32%), 시간당 ≈217 트라이얼, 성공 소요 중앙값 3.0분·p75 6.3분 |
| oc_b2 (low 40% / high 60%) | ≈2,300 (hard 제외) | 동시 96 · 상한 10분 | 자동 체인 |
| sc_m1 (sc-b0 17 + sc-b1 52+수리) | ≈80~100 | 동시 48 · high | 자동 체인 |
| oc_b3 (low/high) | ≈5,000 (중복 제거) | 동시 96 · 상한 10분 | 자동 체인 |
| om_b1 (low/high 50:50) | 2,000 | 동시 96 · 상한 10분 · max_turns 20 | 자동 체인 |

**사고 3건 (전부 `docs/KNOWN_ISSUES.md` 2026-09-08)**: ① Docker 주소 풀 고갈(동시 64 즉사) → daemon.json 풀 확장 ② 과제 이미지 940 MB 미공유 → 베이스 이미지
`alpha-terminal-base:1` ③ 교사 setup.sh 가 2.36 TB 파일 생성 → `precheck_setup.sh` 샌드박스. 운영 교훈: **실행 중인 스크립트를 편집하지 않는다**
(bash 가 파일을 점진적으로 읽어 두 번 검증이 깨졌다) · `pkill -f` 패턴은 자기 명령줄과 매치되지 않게 `[x]` 브래킷과 별도 호출로.
