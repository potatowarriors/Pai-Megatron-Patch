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
| **공개 코퍼스 채택 (2026-09-10)** | NVIDIA 가 `nvidia/Nemotron-Terminal-Corpus`(366,154행, 8.2 GB, cc-by-4.0, TB 2.0 대상, arXiv 2602.21193)를 Post-Training-v3 컬렉션 밖에 별도 공개 — Ultra 의 ~370K 와 규모 일치. **직접 합성을 중단하고 공개 코퍼스를 B안 블렌드의 터미널 멤버로 채택**. 다운로드·렌더 스키마 검증(§2.9 규칙 9)은 phase-2 세션 담당. 이미 만든 `alpha-SFT-Terminal-v1`·`swe_v3_terminus_keephist` 는 폐기하지 않고 대조·보강용으로 보존 — 공개 코퍼스가 Terminus-2 스키마인지 검증 후 채택 또는 부족분 보강으로 확정. §5 마지막 절 |
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

### P3 완료 — 최종 조립 (2026-09-09 11:55 KST) 및 P4 보정 스테이지 확정

**수집 총괄 (2026-09-08 12:49 ~ 09-09 11:54 KST, 23시간)**

| 배치 | 과제 | 설정 | 트라이얼 | 채택 행 |
|---|---|---|---|---|
| oc_b1 high | 1,240 | 동시 64·15분 | 1,240 | 747 |
| oc_b2 low/high | 883 / 1,325 | 동시 96·10분·hard 제외 | 2,208 | 415 / 639 |
| oc_b3 low/high | 528 / 794 | 〃 (중복 제거) | 1,322 | 309 / 388 |
| oc_b4 low/high | 498 / 747 | 〃 | 1,245 | 333 / 385 |
| oc_b5 low/high | 580 / 871 | 〃 | 1,451 | 353 / 519 |
| oc_b6 low/high | 1,102 / 1,653 | 〃 (전체 풀 잔여) | 2,755 | 634 / 920 |
| om_b1~b3 low/high | 1,000 × 6 | 동시 96·10분·max_turns 20 | 4,000 | 563+652 / 348+330 / 327+336 |
| sc_m1~m3 high | 111 / 222 / 165 | 동시 48·20분 | 498 | 87 / 175 / 136 |
| **합계** | | | **14,719** | **8,596** (필터 24 드롭 후) |

처리량은 동시 96 에서 시간당 680~750 트라이얼로 일정했다. 코드 성공률 58~73% (low < high), 수학 60~80%, 시나리오 79~84%.

**정본 `alpha-SFT-Terminal-v1`** (`/home/work/Datasets/LL_datasets/posttraining/SFT/alpha-SFT-Terminal-v1/{train.jsonl, MANIFEST.json, FILTER_STATS.json}`)

| 항목 | 값 |
|---|---|
| 행 | 8,596 = 코드 5,642 + 시나리오 398 + 수학 2,556 (코드·시나리오 : 수학 = 70.3 : 29.7, 요청 7:3) |
| assistant 턴 | 43,855 (전부 reasoning 보유) |
| 품질 필터 | 8,620 → 8,596 (반복 루프 10·편집 없음 13·파싱 오류율 1·금지 git 1) |
| bins `sft_packed_128k_terminal_pad16/terminal_terminus2_synth` | 485 bins, 실토큰 63,381,097, 학습 42,265,199, 드롭 0 |
| 게이트 | verify_sft_bins PASS · render_check 클린(`<think>` 전 턴) |
| 동반 멤버 `swe_v3_terminus_keephist` | SWE-v3 Terminus 3,534행 보존 렌더, 936 bins, 실토큰 122.2M (128k 초과 8행 드롭) |

**P4 보정 스테이지 (사용자 결정 2026-09-09)**: 프리셋 `configs/training/sft_128k_terminal_p3.yaml`, 블렌드 `configs/data/sft_128k_terminal_blend_p3.yaml`.
- 리플레이 80% = phase-1+2 **누적 소비 분포** 기준 49종 (`toolkits/sft_data_preprocessing/terminal_p3_replay_shares.tsv`): cp_v2 14.7 · swe_v3_keepthink 14.2 · math_v4 8.0 ·
  science_v2 7.0 · chat_v3_chat 5.0 · IF 3.9 · 한국어 4.5 · SWE 이웃(r2e·openhands·agentless) 5.3 (×1.7) · 에이전틱 3.0 (×1.8) · finance 1.2 (따라잡기 7.5% → 누적 수준) · 정체성 0.5 · 안전 0.3 · ml 2.6.
- 신규 20% = terminal_terminus2_synth **4 epoch**(0.254B) + swe_v3_terminus_keephist **1 epoch**(0.122B).
- 예산 `--solve-iters`: **90 iters × GBS 160 × 128k = 1.89B**, ≈8시간(320 s/iter). LR 1e-5 → 1.5e-6 cosine, warmup 4 iters (constant 1.5e-6 안은 누적 갱신량이 phase-2 의 2% 라 기각).
- 게이트: 2-iter 스모크(G-P5 상당) → 첫 iteration 게이트 → TB-2 before/after(Terminus-2 `interleaved_thinking=true`, 89과제×8) · LogicKor·IFEval·정체성 프로브 무회귀.
  형식 준수율 개선 +15pp 미만이면 LR 재검토.
- 실행: `cd examples/alpha && bash train.sh baseline_48L sft_128k_terminal_p3 sft_128k_terminal_blend_p3` (phase-2 완주 09-09 ≈21:00 KST 이후, load = phase-2 ckpt).
- **2026-09-09 데이터 교정(`docs/KNOWN_ISSUES.md` 09-09)**: 리플레이 멤버 4종을 교정본으로 교체 — swe_v3_keepthink→swe_v3_tools_keepthink(도구 선언 주입),
  opencode_fixed→opencode_tools(스키마 정규화), chat_v2_on→chat_v2_on_fanout, identity_v2→identity_v2_fanout. 비중·iters 불변(90), 블렌드 재생성.

### P4 사전 게이트 — sub1 2-iter 스모크 PASS, 자동 런처 대기 (2026-09-09 12:00 ~ 12:38 KST)

- **자동 런처** `scripts/launch_p3_after_phase2.sh` 를 main1 에 12:00 KST 가동: phase-2 latest==602 → GPU 유휴 → 51 멤버 sanity → 본 런 →
  첫 iteration 게이트(loss 유한·Traceback 0). 실패는 `outputs/P3_CHAIN_ALERT.txt` 에 남기고 재시도 없음. 해제 `pkill -f "[l]aunch_p3_after_phase2.sh"`.
- **사전 스모크(G-P5)**: sub1 GPU 가 비어 `scripts/sub1_jit595_smoke.sh` 로 3회. sudo 없이 09-07 의 jit595 링크 디렉터리를 `LD_LIBRARY_PATH` 앞에 두는 방식.

| 회차 | 결과 | 원인 | 조치 |
|---|---|---|---|
| 1차 12:13 | 20초 만에 `micro_batch_size is None` | 프리셋이 phase-2 와의 차이 키만 담음 — 평면 YAML 은 상속이 없다 | phase-2 64키 전체 복제, 값 6개 + `no-load-rng` 만 상이(스크립트 대조) — 2ea365a |
| 2차 12:15 | valid 블렌드 `IndexError`(436 요청 > 434 보유) | 51 멤버 top-level 크기 Σceil(w×3200)=3,241 이 멤버 버퍼 여유 0.5% 초과. `--mid-level-dataset-surplus` 가 데이터 제공자에서 미배관 | 배관 + 프리셋 0.05 + 테스트 3건 — 63f79d2 |
| 3차 12:23 | **PASS** | | loss 0.832 → 0.831 · iter 476 s → 326 s(266 TFLOP/s/GPU) · 55.5 GB · 오류 0 · 데이터 캐시 629 파일 선빌드. 기록 `outputs/smoke_pass_p3_sub1_jit595_20260909_122350/` |

- 두 결함 모두 sub1 고유가 아니라 본 런 기동 직후 그대로 터졌을 것 — **자동 런처 전 2-iter 스모크는 생략 불가**. 실패 기록 `outputs/smoke_failed_p3_{preset,valid_surplus}_*/`.
- 부수 확인: 09-04 "sub1 학습 불가"(TE cuDNN norm `munmap`)는 09-07 진단대로 libcuda 595 + JIT 570 혼합이 원인. jit595 우회로 학습 스택도 정상 →
  **sub1 에서 root 없이 학습 가능** (`docs/KNOWN_ISSUES.md` 09-04 갱신). 영구 수정(symlink)은 사용자 항목.
- loss 0.83 은 phase-2 iter 500 의 0.65 보다 높다. 신규 도메인 20% + 1 배치 표본이라 절대값은 기록만(G-P5 기준: 유한·traceback 0), 판단은 본 런 곡선으로.
- 데이터 캐시 `configs/data/.cache/sft_128k_terminal_blend_p3`(NFS)는 본 런이 재사용. 128k 패킹 bins 는 51 멤버도 1분 내 빌드(phase-2 "26종 25분"은 64k 기준).
- phase-2 완주 예상 ≈ 21:00 KST (ckpt 100개당 8h41m 일정; main1 시계는 UTC, sub1 은 KST).

### P4 본 런 — 자동 개시 → iter 30 에서 종료 (2026-09-09 21:07 ~ 23:59 KST, 사용자 결정)

- phase-2 602 완주 직후 런처가 본 런을 자동 개시했다 (`outputs/alpha_baseline_48L_sft_128k_terminal_p3_20260909_120750`, 21:07 KST).
  첫 iteration 게이트 통과(loss 0.828, 469 s), 이후 316 s/iter. 블렌드는 09-09 15:30 교정본(리플레이 멤버 4종 교체, 위 STATUS 기록).
- 진행: **iter 30/90**. loss 0.828 → 0.792(iter 30). iter 20 valid loss 0.672 / PPL 1.96. ckpt `iter_0000020` 저장(옵티마이저 포함, ≈90 GB — 분석용으로 보존).
- **종료 (23:59 KST, SIGTERM — 사용자 결정, 다른 세션 통보)**: phase-2 에서 회귀 2건이 발견돼 교정이 먼저다.
  ① 도구가 선언되면 무관한 질문에도 강제 호출하는 과잉 호출(phase-2 신규 Agentic-v2 의 호출률 96%, When2Call 계열)
  ② Chat-v2 의 교사 정체성 오염("trained by Google" 205턴). phase-3 블렌드는 오염된 Chat-v2 를 리플레이하고 "항상 호출" 형 터미널
  데이터를 얹어 회귀를 심화시키므로 교정 전 진행은 손실. 상세는 phase-2 트랙(STATUS)·`KNOWN_ISSUES`.
- **후속(B안)**: phase-1 최종(iter 2448)에서 교정 블렌드로 phase-2 를 재실행하고 그 블렌드에 터미널 셋을 흡수한다. 터미널 데이터
  (`alpha-SFT-Terminal-v1`, bins `terminal_terminus2_synth`·`swe_v3_terminus_keephist`)는 유효하므로 폐기하지 않는다. 흡수 시 참고:
  둘 다 `--keep-history-think` 렌더(§1.1)이고 Terminus 행은 tool-시나리오가 아니라 `tools` 선언 없이 렌더된다 — 과잉 호출 회귀와는 경로가
  다르다(도구 선언 없음). 4 epoch 근거는 §P4 표. 멤버 50+ 블렌드는 `mid-level-dataset-surplus: 0.05` 가 필요하다(위 사전 게이트 2차).
- main1 GPU 는 회수돼 한국어 chat 재합성에 쓰인다. TB-2 before/after 는 B안 phase-2 완주 후로 이월. `scripts/launch_p3_after_phase2.sh` 는
  첫 iteration 게이트 후 exit 0 으로 끝나 재발동하지 않는다.

### 트랙 전환 — 공개 코퍼스 채택, 합성 중단 (2026-09-10 00:30 KST, 사용자 결정)

- NVIDIA 공개 릴리스 `nvidia/Nemotron-Terminal-Corpus`(366,154행, 8.2 GB, cc-by-4.0) 및 형제 셋 `Nemotron-Terminal-Synthetic-Tasks`, RL 셋
  `Nemotron-RL-Agentic-Terminal-Pivot-v1`; `Nemotron-Cascade-2-SFT-Data` 의 Terminal Agent 스플릿(~324k)에도 포함. §2.9 조사 당시 Post-Training-v3
  컬렉션 밖이라 놓쳤다. 규모가 Ultra 기술보고서의 ~370K 와 일치한다.
- **결정**: 직접 합성 중단(진행 중 작업 없음 — GLM 교사 서버·수집은 09-09 12:00 이전에 이미 종료). 공개 코퍼스를 B안 블렌드의 터미널 멤버로 채택.
  다운로드(`/home/work/Datasets/LL_datasets/posttraining/SFT/Nemotron-Terminal-Corpus`)와 tokenizer_v5 렌더 스키마 검증은 phase-2 세션이 맡는다.
- **보존**: `alpha-SFT-Terminal-v1`(8,596행)·bins `terminal_terminus2_synth`·`swe_v3_terminus_keephist` 는 대조·보강용. 검증 결과에 따라
  (a) 공개 코퍼스 단독 채택 또는 (b) 부족분(예: 한국어·수학 재포맷·특정 카테고리)만 우리 합성으로 보강.
- 검증 시 대조 기준(이 트랙의 정본 스키마): Terminus-2 JSON 응답 `{analysis, plan, commands[{keystrokes,duration}], task_complete}`, 사용자 턴
  `New Terminal Output:` 접두, 완료 재확인 핸드셰이크, system prompt(SWE-v3 Terminus 계열 md5 fa616539·2,836자), assistant 턴마다 `reasoning_content`,
  tool-시나리오 아님(`tools` 없음) → 보존 렌더는 `--keep-history-think`. 품질 신호 7종(`convert/filter_rows.py`)·canary 제외·JSON 이스케이프 수리는 그대로 적용 가능.
- 이 트랙의 파이프라인·도구·실측(P0~P4)은 공개 코퍼스가 부족할 때의 보강 경로로 유지한다. 스터디 문서 `study/terminal_sdg_study.md` 는 공부용으로 유효.

### 공개 코퍼스 검증·변환 — Nemotron-Terminal-Corpus 366k 행 전량 변환, 채택 판정 자료 (2026-09-10 01:00 ~ 03:30 KST)

phase-2 세션의 1차 검증(133k 표본: 계열 md5 fa616539 100%, 인라인 `<think>` 99.8%, 교사 DeepSeek-V3.2, agent terminus-2)을 이어받아
변환기·필터·bins 경로를 끝까지 확인하고 전량을 변환했다. 산출 `/home/work/Datasets/LL_datasets/posttraining/SFT/alpha-SFT-Terminal-NTC-v1/`
(`ntc_v1.jsonl` 20.6 GB, `MANIFEST.json`, `filtered/`). 변환기 `convert/nemotron_terminal_to_rows.py`(c3bdeae), 필터 보정 1037783.

**원본의 실체 (표본 20k 행, adapters)**
- system 역할이 없고 첫 user 턴 = SWE-v3 Terminus 프롬프트 2,836자(`swe_v3_terminus_system_prompt.txt` 와 바이트 일치) + `\n\nTask Description:\n…`. → system 으로 분리.
- assistant content = `<think>…</think>` + Terminus-2 JSON. → `reasoning_content` 로 분리(변환기 special-token 가드 정합).
- **행의 64% 에 파싱 실패 턴**이 있다: think 만 있고 content 가 빈 턴(표본 22,265턴, 전부 중간 턴) 뒤에 harness 의
  `Previous response had parsing errors: ERROR: No valid JSON found in response` 가 따라온다. 학습 목표로 두면 "사고만 하고 응답을 비우는" 행동을
  가르치므로 **(무효 assistant, 파싱오류 user) 쌍을 잘라낸다(splice)** — 파싱 오류에서는 명령이 실행되지 않아 터미널 상태가 그대로라 대화가 일관된다.
- 행의 8.6% 는 마지막 assistant 가 JSON `null`(에피소드 끊김) → 그 턴과 직전 user 턴 제거(`null_tail`).
- `Previous response had warnings:`(유효 JSON + 개행 누락 경고)는 프로토콜의 일부라 보존.

**전량 변환 결과 (`--keep-not-completed`, 18분, 12 워커)**

| 분할 | 입력 | 출력 | 완료(task_complete) | 미완료(플래그) | 미완료% | 턴/행 | splice/행 | null_tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adapters:code | 31,960 | 31,927 | 20,152 | 11,775 | 36.9% | 6.8 | 1.72 | 4,742 |
| adapters:math | 162,692 | 162,662 | 147,533 | 15,129 | 9.3% | 6.5 | 1.17 | 4,300 |
| adapters:swe | 31,661 | 31,421 | 28,885 | 2,536 | 8.1% | 13.6 | 0.59 | 103 |
| synthetic:easy | 44,809 | 44,798 | 35,871 | 8,927 | 19.9% | 7.5 | 0.64 | 1,430 |
| synthetic:medium | 89,343 | 89,216 | 22,541 | 66,675 | 74.7% | 7.6 | 0.37 | 6,462 |
| synthetic:mixed | 5,689 | 5,681 | 739 | 4,942 | 87.0% | 5.8 | 0.26 | 518 |
| **합계** | **366,154** | **365,705** | **255,721** | **109,984** | 30.1% | 7.5 | 0.90 | 17,555 |

드롭 449 (json_invalid 314 · think_residual 126 · injection 6 · no_assistant 3). 사고 보유 턴 99.8%.
미완료 행은 테스트는 통과했지만 완료 선언 전에 에피소드가 끝난 궤적(마지막 턴이 명령 2~6개, 턴 수 분포는 완료 행과 같음 → 상한 절단이 아니라
시간 제한 추정). synthetic medium/mixed 는 대부분이 미완료라, 완료 행만 쓰면 NVIDIA 가 가장 큰 향상을 보고한 범주(data querying·debugging 등)를 잃는다.

**품질 필터 보정** (`filter_rows.py`, 1037783): 그대로 적용하면 표본 통과율 70% — `repeat_loop` 21% 는 DeepSeek 에이전트가 턴마다 앞세우는
`cd /app` 과 `df -h`·`cat patch`·`sleep 2` 같은 비연속 반복 점검이었고(연속 ≥4 는 4행), `no_edit` 5% 는 의존성·질의형 synthetic 과제(파일 쓰기 없음).
→ repeat_loop 는 **연속** 횟수(cd/ls/pwd/clear 제외), no_edit 는 코드·SWE 과제만, dup_task 는 분할별 키. 통과율 98.8%(8,602→8,503). 자체 합성 행은
8,596→8,611 로 5행 차이(기조립 셋 불변).

**렌더·bins 검증 (시험 7,732행)**: `--keep-history-think` 로 859 bins, 실토큰 112.3M / 학습 61.6M, 드롭 0, `verify_sft_bins` PASS,
`render_check` 3문서 `<think>` 전 턴(20/20·6/6·5/5) 봉투 흔적 없음.

**3자 대조 (변환 표본 8,602행, 필터 전; `REFERENCE_STATS.md` 는 필터 후 20k 표본으로 갱신)**

| 지표 | 합성 v1 | SWE-v3 Terminus | 공개 코퍼스(변환) |
|---|---|---|---|
| assistant 턴/행 median (p90 / max) | 5 (7 / 20) | 34 (66 / 232) | 7 (11 / 35) |
| reasoning 보유 턴 | 99.99% | 39.2% | 99.7% |
| reasoning 토큰/턴 median (p90) | 286 (1,839) | 44 (283) | 236 (1,088) |
| 응답 토큰/턴 median · commands/턴 | 161 · 1.1 | 169 · 1.2 | 364 · 3.4 |
| 행 토큰 median (p90 / max) · 128k 초과 | 5.1k (15k / 89k) · 0 | 30k (60k / 204k) · 7 | 13.8k (23k / 67k) · 0 |
| system md5 | 7665e733 (Harbor) | fa616539 | fa616539 |

**판정 자료 요약 (결정은 사용자)**
- 규모: 완료 행만 255.7k ≈ 3.8B 토큰, 전량 365.7k ≈ 5.5B 토큰(행당 ≈15k 기준). 어느 쪽이든 블렌드에서는 부분 표본·ep≤1.
- 원천 비율: math 162.7k(44%) vs code+swe+synthetic 203k. 사용자 결정 코드:수학 7:3 을 유지하려면 math 를 ≈87k 로 서브샘플.
- **완료 행만 vs 전량**: 전량 권고 — 미완료 행도 매 턴이 유효한 행동(테스트 통과)이고 medium/mixed 범주가 여기 있다. 완료 핸드셰이크 예시는
  완료 행 255.7k 로 충분. 미완료는 `metadata.completed=false`·`quality_flags=["not_completed"]` 로 표시돼 있어 서브셋은 필터 한 줄로 만든다.
- **자체 합성 셋의 보강 역할**: system md5 가 다르다 — 공개 코퍼스는 fa616539, **TB-2 평가 harness(Harbor terminus-2)는 7665e733** 이고 우리
  합성 8,596행이 그 프롬프트 계열의 유일한 데이터다. 평가 프롬프트 일치를 위해 합성 v1 을 소량 멤버로 병행하는 것이 안전하다(수학 om 2,556행 포함).
- 블렌드 투입 시: `--keep-history-think`, 멤버 50+ 면 `mid-level-dataset-surplus 0.05`, 130k 행 이상이면 128k bins ≈ 40k 개 → 빌드 ≈1시간(16 워커 추정).
