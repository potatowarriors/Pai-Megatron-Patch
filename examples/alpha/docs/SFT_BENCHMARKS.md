# SFT 벤치마크 스위트 — 설계·판정·인프라 (2026-08-29)

SFT 본 런(`sft_128k_full`, 2,448 iters, save 300)의 체크포인트별 평가 체계.
기존 base-스타일 표준 11종([EVALUATION.md](EVALUATION.md))과 별개로, **chat/thinking 모드**의
instruct 능력을 측정한다. 참조 좌표는 DSV4 post-training 표 + Nemotron 3 Ultra Table 10
(SFT 데이터가 Ultra 레시피이므로 후자가 1차 준거). 진행 상태는 [STATUS.md](STATUS.md).

> **2026-08-30 — T1 을 프론티어 규약으로 재작성했다.** 이전 수치는 전량 무효 판정·삭제
> (원인·증거: [KNOWN_ISSUES.md](KNOWN_ISSUES.md) 2026-08-30). 새 T1 은 §3.6, 게이트는 §7.
> 게이트 G1~G3 를 통과하기 전에는 어떤 수치도 `results/TRACKING.md` 에 기록하지 않는다.

## 0. 결정 로그 (사용자)

| 결정 | 내용 |
|---|---|
| SWE-bench·Terminal-Bench | **필수 요구**. Backend.AI 노드는 docker 불가(§4) → **외부 docker 호스트 gpu06 DinD 컨테이너로 해결·검증 완료**(2026-08-29, [EVAL_DOCKER_NODE.md](EVAL_DOCKER_NODE.md)). 하니스 구축 대기 |
| judge | **gemini-3.7-flash** 확정(사용자, 2026-08-29). 키 검증 완료(`examples/alpha/.env`). 러너는 provider-agnostic(Gemini/OpenAI 호환) |
| 실행 노드 | sub1 (유휴 8×H100). main1은 SFT 학습 전용 |
| Terminal-Bench 버전 (09-07) | TB-1 → **TB 2.0 + Harbor + Terminus-2** 가 정본, TB-1 은 참고치. 학습 데이터가 Terminus-2 스키마다 (§3.11) |
| SWE 인스턴스 이미지 (09-08) | **표준 정책** — 인스턴스별 이미지 500개를 상시 보존하지 않고 실행 후 정리(`docker_gc.sh` 기본). 레이어는 공유된다 (§7 디스크, `KNOWN_ISSUES` 09-08) |
| 평가 대상 (09-13) | phase-1·2·3 계보 폐기 → 최종 단일 SFT(`SFT_FINAL_PLAN.md`). phase-1 최종 iter2448 은 **기준선**으로 잰다 |
| 에이전틱 추론 조건 (09-14) | **도구를 쓰는 평가 = restore**(이전 턴 think 보존), **도구 없는 평가 = strip**(DSV4). SFT 데이터가 이 기준으로 만들어졌다. 에이전틱 fleet 는 reasoning 파서 `nemotron_v3` + 게이트 A5, SWE·TB-2 는 `tau_proxy` 로 복원 (§3.14) |

## 1. DSV4 테이블 26종 판정

채택 8 / 대체·보강 5 / 차단·불가 13.

| 벤치 | 판정 | 근거 |
|---|---|---|
| MMLU-Pro | ✅ T1 | lm_eval 0.4.12 내장 (CoT EM) |
| GPQA Diamond | ✅ T1 | gated 승인·캐시 완료 |
| LiveCodeBench | ✅ T1 | 공식 하니스 로컬 subprocess 실행 — docker 불필요 |
| HMMT Feb / (AIME) | ✅ T1 | MathArena 공개분(2025 Feb) + lm_eval aime25 |
| SimpleQA-Verified | ✅ T3 | 1,000문항, mini judge 채점 |
| MRCR | ✅ T2 (≤256K) | openai/mrcr. 1M은 모델 창(262K, 실사용 ~256K) 초과 |
| SWE Verified | ⚠️ 프록시→풀 | 풀 에이전틱은 docker 경로 확정 후. 그 전엔 oracle-patch + sb-cli 채점 |
| Terminal Bench 2.0 | **전환 결정 (2026-09-07 사용자)** — TB 2.x + Harbor + **Terminus-2** 를 Terminal 정본 게이트로, TB-1(core 0.1.1 + terminus v1)은 참고치 | 근거: 학습 데이터의 Terminus 행이 Terminus-2 스키마(`analysis/plan/commands{keystrokes,duration}/task_complete`)이고 Ultra 수치가 TB 2.0/2.1 기준. 유효 TB-1 수치가 아직 없어 연속성 손실 없음. 실행 규약(반복 8·temp 1.0/top_p 0.95·max_tokens 65,536·A1~A4 게이트)은 승계. docker 경로는 gpu06 DinD 로 확정됨. 데이터 실측은 `SFT_RL_DATASETS.md` §2.9 |
| **τ³-bench** (tau2-bench v1.0.1) | ✅ 에이전틱 — **편입 2026-09-14 (사용자)** | tool+agent+user 3자 상호작용. docker 불요 → sub1 직접 + `tau_proxy`(think 분리·복원). 상대역 `gemma-4-12B-it`(외부 vLLM, 비용 0, 사용자 지정) → 리더보드(gpt-5.2 user-sim) 비교 불가, ckpt 추이용. retail 114 + airline 50 base split × 4 trials; telecom 은 상대역이 tools 를 거절해 조건부. §3.13 |
| IMOAnswerBench / HLE | 후순위 | 15B-A1.8B에 변별력 낮음. HLE는 gated(HF 토큰 필요) |
| Chinese-SimpleQA | 🔁 대체 | 중국어는 학습 언어 아님 → KMMLU 유지 + KoChat 판정(T3) |
| Codeforces / Apex / CorpusQA-1M / BrowseComp / HLE-tools / MCPAtlas / Toolathlon / GDPval-AA / SWE Pro·Multilingual | ❌ | 실시간 저지·비공개 scaffolding·1M 창·웹서치 스택·MCP 팜·유료 서비스·docker |

**보강 (SFT 데이터 정합)**: IFEval(IF-Chat fan-out), BFCL v4(tool-call 템플릿 분기, 로컬 AST 채점),
RULER 64/128K(STATUS의 "RULER 풀 하니스 미구축" 해소), 표준 11종(망각 게이트, T4).

## 2. 티어 구성·주기

ckpt 주기 = 300 iters ≈ 27h. 스위트는 그 안에 완주해야 추이가 그려진다.
아래는 **2026-08-30 iter300 실측** 기준 (7~8 레플리카, sub1 H100).

| 티어 | 태스크 | 문항 × 반복 = 생성 | 실측 소요 | 주기 |
|---|---|---:|---:|---|
| **T1** 코어 | `mmlu_pro_aa` | 12,032 × 1 | | 매 ckpt |
| | `gpqa_diamond_aa` | 198 × 8 | | |
| | `ifeval_aa` | 541 × 8 | | |
| | `aime25_aa` · `hmmt_feb_2025_aa` | 30 × 16 (각) | | |
| | **T1 합계** | **18,904** | **3h10m** (7 레플리카) | |
| **T2** 롱 | `ruler_niah_{single_1,single_2,multikey_1,multivalue}_aa` | 4 × 2구간 × 20 = 160 | 미측정 | 격 ckpt |
| **T3** 판정 | SimpleQA-Verified · LogicKor | 1,000 + 42×2턴 | **16분** | 격 ckpt |
| **에이전틱** | SWE-bench Verified | 500 × 1 (**~59,500 LLM 호출**) | **9.4h** (W=6) | 격 ckpt |
| | Terminal-Bench (core 0.1.1) | 80 × 1 | 미측정 | |
| | **τ³-bench** retail 114 + airline 50 | 164 × **4** = 656 sims (agent ≈1만 · 상대역 ≈8천 호출) | 추정 W=8 6~8h (스모크 후 실측) | 격 ckpt |

### 에이전틱은 T1 보다 3배 무겁다 — 소요 추정 시 주의

SWE-bench 인스턴스 하나가 **평균 119.4 턴**(중앙 113, 최대 250)을 돈다. 턴 하나가 LLM
호출 하나이므로 500 인스턴스 = **약 59,500 호출** — T1 전체 스위트(18,904)의 **3.1배**다.

2026-08-30 iter300 실측:

| 항목 | 값 |
|---|---|
| 경과 (W=6, 8 레플리카) | **9.4시간** (0.883건/분) |
| 호출당 지연 | 3.4초 (정상 — 느린 게 아니라 호출 수가 많다) |
| 종료 상태 | `Submitted` 50% · **`ContextWindowExceededError` 31%** · `RepeatedFormatError` 16% · `LimitsExceeded` 3% |

**추정 시 흔한 실수**: 소규모 프로브의 "N건 / T분"은 이미 병렬도가 반영된 **처리량**이다.
거기에 워커 수로 또 나누면 안 된다 (2026-08-31: 4건/W=4/10분 → "2.5분/건" 으로 읽고
÷6 하여 3.5h 로 추정했으나 실제 9.4h. 2.7배 오차).

**동시성이 병목이었다.** iter300 실행 중 GPU 사용률은 절반 이하, vLLM 대기열은 0,
컨테이너 호스트는 64 CPU 에 load 1.2 였다. 8 레플리카에 동시 요청이 6개뿐이었기 때문이다.
→ 기본 워커를 **SWE 12 / Terminal 8** 로 올렸다 (사용자 승인 2026-08-31). Nemotron 3
Ultra 도 SWE `max_concurrent 10` 을 쓴다.

반복 횟수는 Nemotron 3 Ultra `num_repeats` 준거 (§3.4). 문항 수가 적을수록 k 를 키운다 —
GPQA 198 문항에 1회 측정은 분산이 크다.

**T4(표준 11종 회귀)는 이 스위트 범위에서 제외** (사용자 결정 2026-08-30). LiveCodeBench·
MRCR 도 미착수. 착수 시 §6 작업 큐에 올린다.

생성 규약은 §3.4 정본. **서빙·평가는 tokenizer_v5 apply_chat_template 만 사용**
(INTERLEAVED_THINKING.md §7-4).

## 2.5 운영 절차 — 한 체크포인트 전체 돌리기

**한 줄로 끝내려면**: `bash eval_sft/run_suite.sh <HF_CKPT> <RUN_TAG> [t1,t3,agentic,t2]`
— 티어별 fleet 교체·게이트·판정·집계까지 자동. 아래는 그 안에서 무슨 일이 일어나는지다.

### fleet 구성이 티어마다 다르다 (가장 흔한 사고 원인)

| 단계 | `--max-model-len` | TOOLS | 파서 | 게이트 |
|---|---:|---|---|---|
| T1 코어 · T3 판정 | 40,960 | off | — | G1·G2·G3 |
| **에이전틱** (SWE · TB-2) | **262,144** | **on** | **`qwen3_xml` + reasoning 파서 `nemotron_v3`** · 컨테이너 내 `tau_proxy`(SWE :8110 · TB-2 :8111) | A1·A2·A3·A4·**A5** |
| **τ³-bench** (에이전틱 fleet 그대로, 재기동 없음 — 3799c62) | 262,144 | **on** | 같은 fleet + sub1 `tau_proxy` :8110 | + **T1·T1b·T2** |
| T2 롱컨텍스트 (RULER) | 262,144 | off | — | G1·G2·G3 |

길이는 `run_suite.sh` 가 정본이다(`AGENTIC_MAX_LEN`·`T2_MAX_LEN` 로 덮어쓴다). 에이전틱 fleet 에 reasoning 파서가 빠지면
**도구 선언 여부와 무관하게 모든 응답의 `</think>` 가 사라진다** — 2026-09-14 이전 SWE·TB 수치가 그 조건이었다(§3.14).

잘못된 fleet 로 돌리면 **전량 0점**이 나오고, 그 0점은 모델 실패와 구분되지 않는다.
2026-08-30 SWE 0/20 · Terminal 0/10 이 그 상태였다.

### 수동 절차 (단계별로 확인하며 돌릴 때)

```bash
cd examples/alpha
CK=outputs/<sft_run>/hfmodel_<iter>

# 0) 변환 시 G1 은 자동 (run_convert.sh 내장). 사후 확인:
python3 tools/emit_generation_config.py $CK --check

# 1) T1·T3 — 표준 fleet
TOOLS=0 GPUS=0,1,2,3,4,5,6,7 bash eval_sft/serve_fleet.sh $CK 40960 8 8100
python3 eval_sft/check_gates.py --hf-dir $CK --base-url http://localhost:8100/v1   # G1~G3
bash eval_sft/run_tier1.sh http://localhost:8100/v1 <RUN_TAG>
python3 eval_sft/runners/run_simpleqa.py --base-url http://localhost:8100/v1 --run-name <RUN_TAG>
python3 eval_sft/runners/run_logickor.py --base-url http://localhost:8100/v1 --run-name <RUN_TAG>

# 2) 에이전틱 — TOOLS + reasoning 파서 fleet + 역터널
bash eval_sft/stop_fleet.sh 0,1,2,3,4,5,6,7
TOOLS=1 TOOL_PARSER=qwen3_xml REASONING_PARSER=nemotron_v3 GPUS=0,1,2,3,4,5,6,7 bash eval_sft/serve_fleet.sh $CK 262144 8 8100
bash /home/work/vidsearch/tools/start_swe_tunnel.sh
python3 eval_sft/check_agentic_gates.py --base-url http://localhost:8100/v1        # A1~A5
bash eval_sft/run_swe.sh <RUN_TAG> 0 12          # 0 = 전량 500. SWE_THINK=restore(기본)|strip
bash eval_sft/run_terminal_tb2.sh <RUN_TAG> 0 8  # 0 = 전량 89 × 8회. TB2_THINK=restore(기본)|strip
bash eval_sft/run_tau.sh <RUN_TAG> 0 4 8         # τ³ retail+airline 전량 × 4 trials (docker·역터널 불요, 프록시 자동 기동)

# 3) T2 — 롱 fleet
bash eval_sft/stop_fleet.sh 0,1,2,3,4,5,6,7
TOOLS=0 GPUS=0,1,2,3,4,5,6,7 bash eval_sft/serve_fleet.sh $CK 139264 8 8100
bash eval_sft/run_tier2.sh http://localhost:8100/v1 <RUN_TAG>

# 4) 판정 → 집계
python3 eval_sft/summarize.py eval_sft/results/<RUN_TAG>            # 유효/무효 판정
python3 eval_sft/aggregate_results.py --results-dir eval_sft/results --out eval_sft/results/TRACKING.md
```

### 진행 확인 — 로그를 믿지 말 것

`bash eval_sft/progress.sh <RUN_TAG>`

에이전틱 러너는 ssh 출력을 `tail -N` 으로 파이프한다. `tail` 은 스트림이 끝나야 출력하므로
**실행 중에는 로그에 아무것도 안 나온다**. 로그 크기가 멈춰 있다고 정체로 판단하면 오판이다
(2026-08-30: 97바이트에 멈춘 로그를 보고 완료로 착각했으나 실제로는 282/500 진행 중이었다).

진행은 **산출물 개수**로 센다 — 컨테이너의 `preds_<TAG>/<instance>/` 디렉토리 수,
`runs/<rid>/` 하위 태스크 수, `docker ps` 개수. `progress.sh` 가 이를 한 화면에 모은다.

### wandb — 키 규약 `<task>/<metric>`

`python3 eval_sft/log_eval_wandb.py --results-dir eval_sft/results --run-tag <TAG>`
(`run_suite.sh` 가 집계 후 자동 호출. `WANDB=0` 이면 건너뜀. `--all-tags` 백필,
`--dry-run` 미리보기.)

**키는 `<task>/<metric>`.** 기존 `alpha-evals` 프로젝트와 같은 규약이다. wandb 는 첫 `/`
앞을 패널 섹션으로 잡으므로 이렇게 해야 **벤치마크별로 묶인다**. `bench/` 같은 공통
접두사를 붙이면 전부 한 섹션에 뭉쳐 읽을 수 없다.

    mmlu_pro_aa/exact_match,avg
    ifeval_aa/prompt_level_strict_acc,avg
    ruler_niah_single_1_aa/258048,none
    gsm8k/exact_match,flexible-extract     ← 필터 접미사는 원본 그대로

**평가 결과만 올린다.** 진단 지표(`no_answer` · `think_closed` · `judge_fail` · `empty` ·
`gen_chars` · `samples_k`)는 제외한다 — 측정 성립 여부는 `summarize.py` 와 결과 JSON 에서
본다. wandb 는 점수를 보는 곳이다.

**규약이 바뀌면 run id 를 올린다** (`eval-v3-*`). wandb 는 기록된 키를 지울 수 없어서,
같은 run 에 계속 쓰면 구 키가 누적된다 — 2026-09-01 정리 전 한 run 에 237개 키
(`bench/<task>`, `bench/<task>/<metric>`, `diag/`, `latest/` 4세대)가 섞여 있었다.
정리 후 **36개 / 13섹션**.

**계보가 이어지는 런은 한 곡선으로 합친다.** `log_eval_wandb.py` 의 `RUN_ALIASES` 가 swap 런
(`…_full_swap_20260901_101523`)을 `…_full_20260828_081911` 곡선에 잇는다 — iter900 에서 블렌드 경로 2개만 바꿔 재개한
같은 모델이다(LR·loss 연속 확인). 교체 지점은 run config `blend_swap_at_iter=900` 으로 남긴다. phase-1 최종 iter2448 도
이 곡선의 마지막 점으로 기록된다.

`summary` 에 `latest/` 사본을 만들지 않는다. wandb 가 `run.log` 의 마지막 값을 자동으로
summary 에 넣으므로, 사본을 만들면 패널 목록이 두 배가 된다.

**불변식 넷**:
1. 게이트를 통과하기 전에는 어떤 수치도 `TRACKING.md` 에 넣지 않는다.
2. 생성 파라미터는 **태스크 yaml / `gen_common.py` 가 정본**. 러너가 CLI 로 덮어쓰지 않는다.
3. 부분 표본은 결과 JSON 에 `subsampled=true` 가 박혀 자동으로 `무효` 가 된다.
4. fleet 교체는 반드시 `stop_fleet.sh`(SIGTERM → 회수 확인). hard-kill 반복이 GPU 좀비를 만든다.

## 2.6 구성 요소 인벤토리

| 파일 | 역할 |
|---|---|
| **서빙** | |
| `serve_alpha.sh` | 단일 vLLM 서버. `TOOLS=1` 이면 `--enable-auto-tool-choice --tool-call-parser ${TOOL_PARSER:-qwen3_xml}` |
| `serve_fleet.sh` / `lb_proxy.py` / `stop_fleet.sh` | N개 단일서버 + 라운드로빈 프록시(:8100) / 정상 종료·GPU 회수 확인 |
| **게이트** | |
| `../tools/emit_generation_config.py` | **G1** — `generation_config.json` 생성·eos 정합 검사. `run_convert.sh` 에 내장 |
| `check_gates.py` | **G1·G2·G3** — eos 정합 / `</think>` 관측 / 서빙 스모크 |
| `check_agentic_gates.py` | **A1~A5** — `tool_choice` 수용 / 역터널 / 디스크 / **파서가 실제로 파싱하는지** / **think + 도구호출 턴에서 추론이 분리되는지**(`--tool-path required\|report`) |
| **태스크** | |
| `tasks/*_aa.yaml` + `tasks/aa_utils.py` | T1 5종 (AA 규약: 0-shot·8단 폴백·avg@k·사고 분리) |
| `tasks/ruler_niah_*_aa.yaml` + `tasks/ruler_utils.py` | T2 4종 (Reasoning-Off) |
| `tasks/math_utils.py` | 수학 정답 동등성(`is_equiv`) — lm_eval 원본 재사용 |
| **러너** | |
| `run_tier1.sh` / `run_tier2.sh` | T1 / T2. 설정은 태스크 yaml 정본, CLI 덮어쓰기 없음 |
| `runners/gen_common.py` | T3 공용 생성 헬퍼 — T1 과 같은 파라미터 정본 |
| `runners/run_simpleqa.py` / `run_logickor.py` / `gemini_judge.py` | T3 판정 + 심판 |
| `run_swe.sh` / `run_terminal_tb2.sh` | 에이전틱 정본. 전량이 기본, 부분 표본은 무효 표시. 컨테이너 안에 `tau_proxy` 를 띄워 추론을 복원(SWE :8110 · TB-2 :8111, `SWE_THINK`·`TB2_THINK`) — 프록시 통계로 무효 판정 (§3.14) |
| `run_terminal.sh` | TB-1 (참고치, `TERMINAL_HARNESS=tb1`) |
| `run_tau.sh` / `tau_proxy.py` / `tau_combine.py` | **τ³-bench** 러너(sub1 직접) / think 분리·히스토리 복원 프록시(:8110) / pass^k 합산·무효 규칙. 설치 `install_tau2.sh`, 검증 `tau_smoke.sh`·`tau_render_check.py`·`tau_tasks_count.py`, litellm 레지스트리 `configs/alpha_model_registry.json` (§3.13) |
| `run_suite.sh` | 전 티어 오케스트레이터 (fleet 교체·게이트·판정·집계) |
| `eval_new_ckpt.sh <RUN_DIR> <ITER> [stages]` | **체크포인트 하나의 정본 진입점** — MG→HF 변환(있으면 `tools/verify_hf_export.py` 로 검증 후 재사용) → `run_suite.sh` |
| `suite_running.sh` | 스위트 실행 여부 판정 — argv 구조로 본다. 감시·체인 스크립트는 `pgrep -f` 대신 이것을 쓴다(`KNOWN_ISSUES` 09-05) |
| `docker_gc.sh` | 에이전틱 후 컨테이너 호스트 회수 — build cache·SWE 인스턴스 이미지(기본 정리, `--keep-images`)·산출물 회전(`GC_KEEP_RUNS`) |
| **집계** | |
| `bench_registry.py` | **태스크↔지표 매핑 정본**. 이름을 바꾸면 여기만 고친다 |
| `summarize.py` | 유효/무효 판정 (`no_answer>10%` 또는 `think_closed<50%` → 무효) |
| `aggregate_results.py` / `log_eval_wandb.py` | `TRACKING.md` 추이표 / wandb (무효 셀은 업로드 제외) |
| **환경** | |
| `restore_bench_env.sh` | 세션 재생성 후 휘발 항목 복원 (CUDA13 compat, ifeval 의존성) |

## 3. 인프라 (sub1)

```
main1 SFT save → sub1 감시 → ① run_convert.sh(EP=8, ~30–40분)
  → ② vLLM 서빙: alpha 1-GPU 레플리카 DP4~6 (15B bf16 30GB, attn 6층이라 128K KV ~1.6GB)
  → ③ lm_eval(local-chat-completions) + 커스텀 러너(LCB·MRCR·SimpleQA·HMMT·SWE프록시)
  → ④ wandb alpha-evals 업로드 → 서버 종료, 다음 ckpt 대기
```

- 서빙 venv: sub1 `/tmp/alpha-eval-venv` — **vllm==0.25.1 핀** (08-24 서빙 패리티 게이트가
  이 버전 기준: argmax·top5 일치, KL ≤ 0.0013). 플러그인은
  `project_s/NeMo-RL/examples/configs/alpha/vllm_alpha_plugin/` editable 설치.
  /tmp은 재부팅 시 소멸 → requirements 스냅샷을 `vidsearch/tools/alpha_eval/`에 유지.
- 하니스: lm_eval 0.4.12 (`vidsearch/tools/lmeval0412` 격리 설치와 동일 버전).
- 러너·오케스트레이터 코드: `examples/alpha/eval_sft/` (이 리포).
- 베이스라인: 전 스위트를 **LC-B iter320 hfmodel**로 먼저 완주(하니스 검증 겸 기준점).

## 3.3 대표 지표 — 프론티어 보고 관행 대조 (감사 2026-08-31)

벤치마다 여러 지표가 나온다. **어느 것을 대표로 쓰느냐가 점수를 10~15pp 움직인다.**
정본은 `eval_sft/bench_registry.py` 의 `HEADLINE`.

| 벤치 | 대표 지표 | 프론티어 관행 | 판정 |
|---|---|---|---|
| MMLU-Pro | `exact_match` | accuracy / EM | ✅ 일치 |
| GPQA-Diamond | `exact_match` (avg@8) | pass@1 평균 | ✅ 일치 |
| AIME25 · HMMT | `exact_match` (avg@16) | pass@1[avg-of-k] | ✅ 일치 |
| **IFEval** | **`prompt_level_strict_acc`** | prompt-level strict | ✅ **2026-08-31 교정** |
| **RULER** | **구간 평균(64K·128K)** | 구간별 + Avg (Qwen3 카드) | ✅ **2026-08-31 교정** |
| SimpleQA | `accuracy` (= correct/n) | % correct | ✅ 일치 (F-score 는 JSON 에 동봉) |
| LogicKor | `overall/10` → 100분율 | 원래 /10 표기 | ⚠️ 표 통일 위한 환산 |
| SWE-bench Verified | `% resolved` | % resolved | ✅ 일치 |
| Terminal-Bench | `% resolved` | accuracy | ✅ 일치 |

**교정 2건의 근거**:

- **IFEval**: 4지표 중 `inst_level_loose_acc` 를 쓰고 있었다 — 가장 관대한 값이다.
  iter600 기준 `inst_level_loose` 70.3 vs `prompt_level_strict` **55.6**, 차이 14.7pp.
  프론티어 카드는 prompt-level strict 를 쓴다. 4지표 모두 결과 JSON 에 남으므로 정보
  손실은 없다.
- **RULER**: 131072 한 구간만 대표로 잡아 65536 신호를 버리고 있었다. Qwen3 카드처럼
  구간 평균을 쓴다.

### 반복 횟수 — 프론티어 관행 조사·확정 (2026-08-31)

**원칙 (DeepSeek-R1 논문 §평가 설정)**: *"we use a sampling temperature of 0.6 and a
top-p value of 0.95 to generate k responses (**typically between 4 and 64, depending on
the test set size**)"* — **k 는 데이터셋 크기에 반비례한다.** 작은 셋일수록 크게.

**1차 출처**

| 출처 | 확인한 값 |
|---|---|
| DeepSeek-R1 (arXiv 2501.12948) | k = 4~64, 셋 크기에 따라. AIME 는 추가로 `cons@64` |
| Nemotron 3 Ultra (공식 eval yaml) | GPQA 8 · SciCode 8 · IFBench 8 · Multi-Challenge 8 · AA-LCR 16 · Omniscience 10 · CritPt 5 · IMO-AnswerBench 5 · MMLU-Pro 1 · MMLU-ProX 1 · HLE 1 |
| Nemotron 3 Ultra (에이전틱) | SWE-bench 3 · Terminal-bench 8 |
| Qwen3 | RULER 길이당 20 샘플 (13 서브태스크 × 20 = 260) |
| Artificial Analysis | GPQA 5 · Terminal-Bench 3 · SciCode 3 · AA-LCR 3 · τ³ 5 · CritPt 5 · HLE 1 |
| **SWE-bench 리더보드 규약** | **pass@1 = 인스턴스당 1 예측.** best@k 는 SWE-bench 테스트를 쓰지 않는 **독립 선택기**가 있어야 하고 별도 표기 |

**확정 (우리 스위트)**

| 벤치 | 문항 | k | 근거 |
|---|---:|---:|---|
| MMLU-Pro | 12,032 | 1 | Nemotron 1. 표본이 이미 크다 |
| SimpleQA-Verified | 1,000 | 1 | 표본이 크다. Nemotron Omniscience 10 은 더 작은 셋 |
| IFEval | 541 | 8 | Nemotron IFBench 8 |
| **SWE-bench Verified** | 500 | **1** | **리더보드 규약(pass@1 단일 시도)**. Nemotron 의 3 은 런 평균이지 제출 형식이 아니다 |
| GPQA-Diamond | 198 | 8 | Nemotron 8 (AA 는 5) |
| **Terminal-Bench** | 80 | **8** | Nemotron 8 (AA 3). `tb --n-attempts` |
| **τ³-bench** | 164 (retail 114 + airline 50) | **4** | 리더보드 "≥4 선호"(pass^1 주지표, pass^2~4 병기), AA 5. 사용자 확정 2026-09-14 |
| **LogicKor** | **42** | **8** | 우리 스위트에서 **가장 작다**. Nemotron Multi-Challenge 8 |
| **AIME25 · HMMT** | 30 | **32** | R1 은 최대 64. 30문항이라 분산이 가장 크다. 32 는 비용 절충 |
| RULER (구간당) | 20 | 1 | Qwen3 가 길이당 20 샘플을 쓰는 것이 곧 반복이다 |

**변경 이력**: AIME·HMMT 16→32, LogicKor 1→8, Terminal-Bench 1→8 (2026-08-31).
SWE-bench 는 1 을 **유지**한다 — 리더보드 규약이 그렇다.

## 3.35 추론 길이 — 구조적 상한과 실측 창 (2026-09-01)

세 축을 구분한다. 흔히 혼동된다.

| 축 | 값 | 성격 |
|---|---|---|
| 학습 시퀀스 길이 | 128K | LC-B CPT · SFT 가 지불한 비용 |
| `max_position_embeddings` | **262,144** | 구조적 상한 (RoPE θ=10M, partial 0.25) |
| **실측 유효 창** | **~256K (단순 검색 기준)** | `study/lc_b_final_eval.md` §2 |

**128K 로 학습해 256K 를 쓸 수 있다 — 단순 검색(single needle) 기준이다.** LC-B iter320 실측: 196,608 에서 100%,
**262,144 에서 95%**, 393,216 에서 0%. 어려운 검색은 128K 를 넘으면 떨어진다(T2 RULER single_2 90 → 20~50,
multikey 50 → 20~35 @252K, 아래 "128K 학습·262K 서빙 점검"). 같은 RoPE 설정의 LC-A(32K 학습)는 262,144 에서
42.5% 로 붕괴했다 — **학습 길이를 늘린 것이 외삽 여유를 만들었다.**

### 128K 학습 · 262K 서빙 점검 (2026-09-15, iter2448 SWE 중간 표본)

**RoPE 스케일링은 걸려 있지 않다.** HF `config.json` 이 `rope_scaling: null`, `max_position_embeddings: 262144` 이고
`serve_alpha.sh` 는 `--hf-overrides`·`--rope-scaling` 을 주지 않는다(vLLM 로그 `max_seq_len=262144` 만). 따라서 131K~262K 는
보간 없는 RoPE 외삽이고, **131K 이하 위치는 학습과 똑같이 계산된다.**

"SWE 실패가 학습 길이 초과 탓인가" 를 턴별 `prompt_tokens` 로 검증했다 — **주원인이 아니다.**

| 같은 인스턴스 174개 | iter1800 (구 조건) | iter2448 (현행) |
|---|---:|---:|
| 제출 / 스텝 한도 초과 | 137 / 23 | 72 / **101** |
| 131K 를 넘은 궤적 | **10** | **11** |
| 턴 오류율 0~32K · 32~64K · 64~96K · 96~128K | 24 · 28 · 33 · 35% | 30 · 39 · 43 · 48% |
| 턴 오류율 128K 초과 | 24% (294턴) | 60% (247턴, 선택 편향) |

- 131K 초과 궤적 수는 같은데 결과가 갈렸고, 오류율 차이는 **가장 짧은 구간부터** 있다.
- 한도 초과 궤적의 지속 루프 37건은 **전부 131K 전에 시작**(시작 시 프롬프트 중앙 75K, 사분위 47~91K).
- 두 실행 모두 오류·재실행율이 짧은 구간부터 길이에 따라 꾸준히 오른다 — 128K 절벽이 아니라 **긴 에이전트 이력을
  다루는 능력의 문제**다. 128K 를 넘는 구간의 약화(위 RULER)는 따로 실재하지만 SWE 궤적의 약 6% 만 그 구간에 간다.

### 외부 사례 — Qwen3.8-Flash-Next (2026-08 공개, 256K 학습 · 1M 확장)

같은 계열 구조라 서빙 관행을 직접 대조할 수 있다(RoPE θ=10M · partial 0.25 · head_dim 256 · KV 2 · 4층마다 attention 1 ·
`max_position_embeddings` 262,144). 출처: HF 모델 카드·`config.json`·`chat_template.jinja`, vLLM 레시피, 기술 보고서(2026-09-15 확인).

| 항목 | Qwen3.8-Flash-Next | alpha (현행 평가) |
|---|---|---|
| 롱컨텍스트 학습 | QSA 도입 CPT **256K 시퀀스 × 약 200B 토큰** (+ 인덱서 2B) | 128K (LC-B 4.03B 토큰 + SFT) |
| 네이티브 서빙 창 | 262,144 = **학습 길이** (`rope_type: default`) | 262,144 = **학습 길이의 2배, 스케일링 없음** |
| 학습 길이 초과 | **정적 YaRN 옵트인** — `factor 4.0, original_max_position_embeddings 262144`, `--max-model-len 1000000` | 무보정 외삽 |
| 초과 시 경고 | "정적 YaRN 은 **짧은 텍스트 성능을 해칠 수 있다** — 필요할 때만, factor 는 실제 길이에 맞춰(524K 면 2.0)". 레시피도 "단문 품질 평가 후 기본값으로" | — |
| SWE 계열 평가 창 | DeepSWE 1.1 · SWE-bench Pro · Multilingual 모두 **256K** (temp 1.0 · top_p 0.95) | 262K (입력 상한 229,376) |
| 학습 길이 초과 품질 | MRCR 8-needle 128K 96.0 · 256K 93.0 → **512K 40.5 · 1M 26.4** (RULER 는 93 이상 유지) | RULER single_2·multikey 128K 초과 하락 |
| 이전 턴 think | 템플릿 기본 **전 시나리오 보존**(`preserve_thinking` 미지정=true, false 면 마지막 질의 이후만). "에이전트 결정 일관성·KV 캐시 활용" | 도구 사용만 restore (DSV4 규칙) |
| 서빙 인자 | `--reasoning-parser qwen3` · `--tool-call-parser qwen3_coder` · prefix caching | `nemotron_v3` · `qwen3_xml` |
| 샘플링 | thinking temp 1.0 · top_p 0.95 · **top_k 20**, 끝없는 반복엔 `presence_penalty` 0~2 | top_k 없음 (AA/Nemotron 규약) |
| 에이전틱 출력 예산 | 추론 262,144 · 최종 131,072 (1M 창) | SWE 32,768 · TB-2 65,536 |

시사점:
- **Qwen 은 "네이티브 창 = 학습 길이" 를 지킨다.** 그 관행으로는 alpha 의 네이티브는 131K 이고 262K 는 YaRN factor 2 옵트인
  구간이다. 다만 YaRN 은 교환이다 — Qwen 도 단문 손해를 경고하고, 우리 512K 그리드에서 yarn4 가 131K multikey 를 60 → 35 로
  깎았다(`study/lc_512k_eval.md` §5).
- **학습 길이를 넘으면 256K 학습 모델도 어려운 검색이 무너진다**(MRCR 93 → 41). "지원 길이" 는 학습 길이로 보는 게 안전하다.
- **롱컨텍스트 학습량 차이(약 50배)** 가 위치 인코딩보다 위 SWE 길이 곡선(짧은 구간부터 오류 증가)과 더 잘 맞는다.
- **restore 는 Qwen 기본값과 같은 방향**이다(에이전트 이력 보존). 반복 루프 대응으로 Qwen 은 top_k 20·presence_penalty 를 권한다 —
  AA/Nemotron 규약 비교성이 바뀌므로 규약 변경이 아니라 진단 실험으로만 본다.
- 열린 선택: SWE 평가 창을 학습 길이(131K)로 맞출지(Qwen 방식, 131K 초과 약 6% 궤적이 창 초과로 바뀜) vs 262K 유지(계열 연속성).

### KV 캐시가 싸다 — 하이브리드 구조의 직접 이득

`full_attention_interval: 4` → 24층 중 **6층만 full attention**. 나머지 18층은
GatedDeltaNet 으로 KV 를 쓰지 않는다(고정 크기 state).

토큰당 KV = 6층 × 2(K,V) × 2 kv-head × 256 dim × 2B = **12 KB**
(같은 크기 순수 attention 모델이면 48 KB — 4배)

| model_len | KV/시퀀스 | fleet(8 레플리카) 동시 시퀀스 |
|---:|---:|---:|
| 131,072 | 1.50 GB | 232 |
| **262,144** | **3.00 GB** | **112** |

**즉 256K 서빙은 메모리 문제가 아니다.** 에이전틱의 병목은 창이 아니라 **턴당 출력 예산**이었다.

**262,144 를 초과할 수 없다.** vLLM 은 `max_model_len > max_position_embeddings` 를
거부한다 — 2026-09-01 실측 오류: *"User-specified max_model_len (270336) is greater than
the derived max_model_len (max_position_embeddings=262144)"*. 따라서 프롬프트 + 생성이
262,144 안에 들어가야 하고, RULER 최장 구간은 **258,048** (= 262,144 − 4,096)으로 잡는다.

실측 KV 용량 (262,144 서빙, H100 80GB 1장): **GPU KV cache 3,200,318 토큰**,
262,144 토큰 요청 기준 **동시 12.21 시퀀스** — 계산치(14)와 일치한다.

### 서빙 설정 (2026-09-01 갱신)

| 단계 | `--max-model-len` | 턴당 `max_tokens` |
|---|---:|---:|
| T1 · T3 | 40,960 | 32,768 |
| **에이전틱** | **262,144** | SWE 32,768 · **Terminal 65,536** |
| **T2 RULER** | **262,144** | 512 (Reasoning-Off) |

Terminal 이 65,536 인 이유: terminus 는 응답을 4필드 JSON
(`CommandBatchResponse`: state_analysis / explanation / commands / is_task_complete)으로
요구하고, **잘린 응답에서 재시도 없이 태스크를 버린다**. 장문 사고 뒤에 그 JSON 을 내야
하는데 32,768 로도 부족했다 — 2026-09-01 실측: `unknown_agent_error` 291/640(45%),
그 중 181건의 출력 토큰 중앙값이 **0**(첫 턴 사망). iter300 도 동일(180회).

## 3.4 생성 세팅 — AA / Nemotron 3 Ultra 규약 (2026-08-30 재작성)

조사 근거는 프론티어 1차 자료: DeepSeek-R1·Qwen3·Nemotron 모델 카드, NVIDIA
[Nemotron 3 Ultra 재현 설정](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/examples/nemotron/nemotron-3-ultra),
OpenAI simple-evals, Artificial Analysis 방법론.

**온도 1.0 확정 (사용자, 2026-08-30)** — Nemotron 3 Ultra 와 동일. 우리 SFT 데이터가
Ultra 레시피이므로 그 쪽을 준거로 삼는다. (R1·Qwen3 계열은 0.6 을 권고하며, AA 규약은
"모델 랩 권고를 따른다"이다 — 즉 우리가 우리 기준을 정하는 셈.)

| 항목 | 값 | 근거 |
|---|---|---|
| temperature / top_p | **1.0 / 0.95** | Nemotron 3 Ultra eval yaml `params` |
| max_tokens | **32768** | R1 카드. Ultra 는 상한 자체를 제거(`params_to_remove`)하나 우리는 창 한계로 고정 |
| few-shot | **0-shot CoT** | simple-evals: few-shot 은 base 모델 유물 |
| 시스템 프롬프트 | **없음** | R1 카드 / Ultra `use_system_prompt: false` |
| `skip_special_tokens` | **false** | Ultra `params_to_add` — `</think>` 가 출력에 살아남아야 한다 |
| `seed` | **null** | 시드를 고정하면 반복 k개가 동일 표본이 된다 |
| 반복 (avg@k) | MMLU-Pro 1 · GPQA 8 · IFEval 8 · AIME/HMMT 16 | Ultra `num_repeats`; 수학은 R1 64→실무 16 |

**RULER 는 예외** — 추론을 끄고 `temp 0.00001 / top_p 0.99 / max_gen 512` 로 돈다 (§3.9).

`pass@1` 은 greedy 1회가 아니라 **k회 평균**이다 (R1 카드: "generate 64 responses per
query to estimate pass@1"). 문항 수가 적을수록 k 를 키운다 — GPQA 198 문항에 1회 측정은
분산이 크다.

## 3.5 sub1 서빙 환경 — vllm 0.25.1 + CUDA 13 compat (2026-08-29)

서빙 venv: `/home/work/vidsearch/tools/alpha_serve_venv` (NFS, 재부팅 유지). vllm==0.25.1
(패리티 검증 버전) + `vllm_alpha_plugin` editable. 설치 시 함정 3건:

1. **PIP_CONSTRAINT 해제 필수.** NGC 이미지 전역 `/etc/pip/constraint.txt`가 torch를
   시스템 NGC 버전으로 고정해 vllm 의존성 해결이 깨진다. 모든 pip·serve에 `PIP_CONSTRAINT=`.
2. **transformers 5.16.1로 딸려 옴** → `regex`·`safetensors`·`typing_extensions` 핀 연쇄
   충돌. `pip install -U "regex>=2025.10.22" "safetensors>=0.8.0" "typing_extensions>=4.15"`로 해소.
   (HF remote-code modeling_alpha.py의 5.x 비호환은 **플러그인 경로와 무관** — 서빙은 플러그인 사용.)
3. **vllm 0.25.1 = CUDA 13 빌드** (`_C.so`가 libcudart.so.13). sub1 드라이버 535(CUDA 12.8)로는
   기동 불가(`driver too old 12080`). **해법: CUDA 13 forward-compat** — `setup_nemo_rl_env.sh`와
   동일하게 `cuda-compat-13-2_595.91.07` 설치, sub1 `/usr/local/cuda/compat/lib.real`의
   libcuda 570→595 교체(사용자 승인 2026-08-29). 595는 CUDA 12·13 모두 지원해 기존 Pai
   시스템 torch(cu12.8)도 회귀 없음(검증됨). **백업**: `/home/work/vidsearch/tools/cuda_compat13/backup_570_*`.
   **재부팅 시 컨테이너 초기화되면 이 교체를 재수행해야 함** (deb·백업은 NFS에 보존:
   `/home/work/vidsearch/tools/cuda_compat13/`).

서빙 기동: `bash eval_sft/serve_alpha.sh <hfmodel> [max_len] [DP] [port]` (§3).

## 3.6 T1 커스텀 태스크 (`eval_sft/tasks/*_aa.yaml`)

lm_eval 내장 태스크를 쓰지 않는다. 내장은 base 모델용이라 채팅·추론 모델에서 추출이
대량 실패한다 — GPQA `strict-match` 정규식이 `(?<=The answer is )` 라 채팅 모델은 원리적으로
맞출 수 없고(실측 0.0), MMLU-Pro 는 5-shot 에 `answer is (A)` 단일 패턴이다.

| 태스크 | 문항 | 반복 | 내장 대비 바뀐 점 |
|---|---:|---:|---|
| `mmlu_pro_aa` | 12,032 | 1 | 0-shot, AA 프롬프트(보기 개수 가변 — 3~10), 8단 폴백 추출 |
| `gpqa_diamond_aa` | 198 | 8 | 0-shot, AA 프롬프트, **보기 순열 결정론적**(내장은 시드 없는 shuffle → 재현 불가) |
| `ifeval_aa` | 541 | 8 | 사고 구간 제거 후 채점, 예산 1280→32768. 지시 검사기는 내장 그대로 |
| `aime25_aa` | 30 | 16 | boxed 프롬프트, **avg@16 실제 작동** |
| `hmmt_feb_2025_aa` | 30 | 16 | 동일 |

**공통 규약 세 가지**:

1. **생성 파라미터는 태스크 yaml 이 단일 정본.** 러너가 `--gen_kwargs` 로 덮어쓰지 않는다.
   덮어쓰면 사후에 어느 설정이 적용됐는지 알 수 없다.
2. **추출은 사고 이후 구간에서.** `</think>` 뒤를 잘라 쓴다. 사고 구간에는 보기 나열·중간
   후보·자기부정이 가득해 전문에 정규식을 걸면 잘못된 후보를 집는다.
3. **`no_answer`·`think_closed` 를 함께 보고.** 점수만 보면 측정 실패가 오답으로 위장된다
   (NeMo-Skills 규약). 판정은 `eval_sft/summarize.py` — `no_answer>10%` 또는
   `think_closed<50%` 면 **무효**로 판정하고 기록을 막는다.

**구 avg@16 버그 (2026-08-30 발견·수정)**: `repeats: k` 는 동일 Instance 를 k번 복제해
`resps` 에 k개를 쌓는데, `filter_list` 를 지정하지 않으면 lm_eval 이 기본 `take_first` 를
꽂아 1개만 남긴다. 구 `aime25_avg16`·`hmmt_feb_2025_avg16` 은 16배 연산을 쓰고 avg@1 을
계산하고 있었다. 새 태스크는 `take_first_k` 로 k개를 모두 넘긴다.

## 3.7 세션 재생성 후 환경 복원 (sub1)

컨테이너 세션이 초기화되면 **NFS는 살아남고 시스템·user-site는 소멸**한다.

| 살아남음 (NFS `vidsearch/tools/`) | 소멸 → 재적용 필요 |
|---|---|
| serve venv(9.5G)·`lm_eval0412`·compat deb+백업·참조로짓 / HF캐시(`Datasets/benchmarks`) | ① CUDA13 compat 시스템 스왑(`/usr/local/cuda/compat`) ② ifeval leaf 의존성(nltk≥3.9.1·langdetect·immutabledict, `~/.local`) |

**복원 = 명령 하나** (멱등):
```bash
cd examples/alpha && bash eval_sft/restore_bench_env.sh
```
이 스크립트가 ①② 재적용 + 검증(serve venv vllm/플러그인, lm_eval, 참조로짓, HF캐시)까지 한다.
서빙 전 필수. **먼저 Pai 환경 복원**(`RESTORE_AFTER_REBOOT.md`)이 되어 있어야 시스템 python·GPU 접근이 선다.

주의: serve venv의 torch는 cu13이라 **compat 스왑 없이는 `torch.cuda.is_available()=False`** → 복원 스크립트가
compat부터 처리한다. gpu06 docker 노드는 별도([EVAL_DOCKER_NODE.md]) — 그쪽은 컨테이너 `--restart`로 살아남고
dockerd만 수동 기동.

## 3.8 첫 유효 측정 — iter300 (2026-08-30)

재작성한 T1(§3.6)·게이트(§7)로 얻은 **첫 유효 수치**. 이전 08-29 베이스라인 표는 전량
무효 판정·삭제됐다(경위: [KNOWN_ISSUES.md](KNOWN_ISSUES.md) 2026-08-30 두 항목).

체크포인트 `alpha_baseline_48L_sft_128k_full_20260828_081911` iter **300/2448 (12%)**.
게이트 G1·G2·G3 통과 후 측정. 총 18,904 생성, 7 레플리카, 3시간 10분.

| 벤치 | 점수 | n | k | 추출실패 | 사고마감 | 판정 |
|---|---:|---:|---:|---:|---:|---|
| MMLU-Pro | **47.0** | 12,032 | 1 | 1.7% | 89.6% | 유효 |
| GPQA-Diamond | **32.0** | 198 | 8 | 1.4% | 56.7% | 유효 |
| IFEval (inst loose) | **65.6** | 541 | 8 | — | 91.3% | 유효 |
| AIME 2025 | 0.6 | 30 | 16 | 1.7% | **16.2%** | **무효** |
| HMMT Feb 2025 | 0.2 | 30 | 16 | 4.2% | **10.8%** | **무효** |

**하니스 재작성의 효과** — 같은 체크포인트, 같은 모델:

| 벤치 | 구 하니스(무효) | 새 하니스(유효) | 차이 |
|---|---:|---:|---|
| MMLU-Pro | 36.9 (추출실패 34%) | **47.0** (1.7%) | +10.1 |
| GPQA-D | 23.2 (flexible, 무작위 수준) | **32.0** | +8.8 |

구 수치는 모델이 아니라 하니스 결함을 측정하고 있었다. 추출 실패율이 34% → 1.7% 로
떨어진 것이 그 증거다.

**수학 2종이 무효인 이유는 설정이 아니라 모델이다.** 사고 마감률 16%/11% — 32,768 토큰
예산 안에서 `</think>` 에 도달하지 못한다. 추출 실패율은 정상(1.7%/4.2%)이므로 하니스는
작동하고 있다. iter300 은 전체 SFT 의 12% 지점이다. 예산을 Qwen3 권고치(81,920)로 늘리는
것은 별개 선택지이나, 근본 원인은 학습량이다.

## 3.9 T2 RULER — Reasoning-Off 재설계 (2026-08-30)

**RULER 는 needle 추출 과제이지 추론 과제가 아니다.** 프론티어 셋이 방법은 달라도 같은
판단을 한다:

| 출처 | RULER 처리 |
|---|---|
| Nemotron Nano 9B v2 | *"except RULER, which is evaluated in **Reasoning-Off** mode"* |
| Nemotron 3 Ultra | instruct 가 아닌 **base 스위트**로 분리, `temp 0.00001 / top_p 0.99` |
| Qwen3-235B | **thinking budget 8,192** — *"To avoid overly verbose reasoning"* |

구 설정은 추론을 켠 채 출력 예산만 128 토큰으로 조였다. 모델이 서두 분석에 예산을 소진해
needle 에 도달하지 못했고 65536 구간 6~25% 가 나왔다 — 같은 모델이 LC-B 자체 NIAH 에서는
4k~131k **200/200** 이었다. 모순의 원인은 모델이 아니라 태스크 설정이었다.

**실측 대조** (iter300, 동일 needle 프롬프트, 2026-08-30):

| 모드 | finish | tokens | needle 추출 |
|---|---|---:|---|
| thinking ON | length | 512 소진 | 실패 |
| thinking OFF | **stop** | **21** | **성공** |

alpha 챗 템플릿은 `enable_thinking=false` 일 때 `<|im_start|>assistant\n<think></think>` 로
사고를 미리 닫아 렌더한다. 요청의 `chat_template_kwargs` 한 줄로 Reasoning-Off 가 된다.

| 항목 | 값 |
|---|---|
| 태스크 | `ruler_niah_{single_1,single_2,multikey_1,multivalue}_aa` |
| 구간 / 표본 | 65536 · 131072, 구간당 20 (Qwen3 도 길이당 20) |
| temp / top_p | 0.00001 / 0.99 (Nemotron base 스위트) |
| max_gen_toks | 512 (실측 21토큰이면 충분) |
| 서빙 | **롱 fleet `--max-model-len ≥ 139264`** — 표준 fleet(40960)로 돌리면 전량 실패 |
| 러너 | `eval_sft/run_tier2.sh` |

**같이 고친 것**: 구 `common_utils.process_results` 는 센티넬 dict 를 하드코딩된
`DEFAULT_SEQ_LENGTHS = [4096]` 로 만들어, 샘플이 0개인 4096 구간에 `-1.0` 이 결과에 남았다.
새 `ruler_utils.SEQ_LENGTHS` 는 yaml 의 `metric_list`·`metadata.max_seq_lengths` 와 일치를
**강제**한다(어긋나면 `_build` 가 예외). 모듈 전역에 실행 중 값을 쌓는 방식은 lm_eval 이
모듈을 경로별로 따로 로드해 인스턴스가 갈리므로 쓸 수 없다.

## 3.11 Terminal-Bench 2.0 전환 — Harbor + Terminus-2 (2026-09-07 구축)

사용자 결정(2026-09-07)에 따라 Terminal 정본 게이트를 TB-1 에서 **TB 2.0** 으로 옮겼다.
근거는 `SFT_RL_DATASETS.md` §2.9 — 학습 데이터의 Terminus 행이 Terminus-2 스키마인데
구 하니스는 terminus **v1**(4필드)이었다. 스키마는 하니스 에이전트가 정하므로 v1 하니스에
v2 형식을 끼워 넣을 수 없다. Ultra 공개 수치도 TB 2.0/2.1 기준이다. TB-1 은 참고치.

### 설치 (컨테이너 `alpha-eval`, 2026-09-07 실측)

`terminal-bench` 패키지는 PyPI 에 **0.2.18 까지만** 있다 — TB 2.x 는 `harbor` 로 배포 경로가
바뀌었다. 기존 `/opt/terminalbench/venv`(TB-1)는 **건드리지 않고** 별도 venv 를 쓴다.

```bash
mkdir -p /opt/harbor && cd /opt/harbor
python3 -m venv venv && ./venv/bin/pip install harbor        # 0.22.0, Python>=3.12 (컨테이너 3.12.3)
HOME=/opt/harbor ./venv/bin/harbor download terminal-bench@2.0   # 89 tasks (TB-1 은 80)
```

### 플래그 대응 — `harbor run --help` 실측

| TB-1 (`tb run`) | TB-2 (`harbor run`) |
|---|---|
| `--agent terminus` | `-a terminus-2` |
| `-k api_base=…` | `--ak api_base=…` (값은 JSON/파이썬 리터럴로 파싱) |
| `--dataset terminal-bench-core==0.1.1` | `-d terminal-bench@2.0` |
| `--n-attempts K` | `-k K` ⚠️ TB-1 의 `-k`(agent kwarg)와 의미가 다르다 |
| `--n-concurrent W` | `-n W` |
| `--n-tasks N` | `-l N` |
| `--run-id` / `--output-path` | `--job-name` / `-o` |

`Terminus2.__init__` 가 노출하는 인자: `api_base` · `model_name` · **`parser_name`('json'\|'xml')** ·
`temperature` · `max_turns` · `llm_call_kwargs`(dict) · `max_thinking_tokens`.
**`parser_name=json`** 을 쓴다 — 학습 데이터가 Terminus-2 JSON 스키마다. (json/xml 은 파서뿐
아니라 **프롬프트 템플릿도 다르다**.) top_p·max_tokens 는 `llm_call_kwargs` 로 넘긴다.

실행 규약은 TB-1 에서 승계: 반복 8 · temp 1.0 / top_p 0.95 · max_tokens 65,536 · A1~A4 게이트 ·
전량 실행. 러너는 `eval_sft/run_terminal_tb2.sh`.

### 검증 (2026-09-07)

| 단계 | 결과 |
|---|---|
| oracle 3태스크 (하니스 자체) | **3/3, mean reward 1.000, 예외 0**, 49초 |
| 실모델 1태스크 스모크 (iter1800 fleet) | 예외 0, 9스텝 완주, prompt 29,851 / completion 5,986 토큰 |

oracle 을 먼저 돌리는 이유: 0점이 나왔을 때 하니스 결함인지 모델 실패인지 갈라야 한다.
2026-08-30 에 파서 오설정으로 SWE 0/20 · Terminal 0/10 을 "모델 실패" 로 읽었던 선례가 있다.

**추출 명령 형태가 학습 데이터와 일치**한다:
```json
{"function_name": "bash_command", "arguments": {"keystrokes": "ls -la\n", "duration": 0.5}}
```

다만 그 1태스크에서 **에이전트 응답 8개 중 4개만 명령이 추출**됐다(50%). 파싱 실패한 응답은
모델이 스스로 형식을 고치려는 내용이었다("I need to fix the command to properly output JSON",
"there's extra text after the JSON"). 하니스를 맞춰도 형식 준수는 완전하지 않다 — 학습 블렌드에서
Terminus 행이 토큰 기준 ≈0.3% 뿐이기 때문으로 보인다(§2.9). n=1 이므로 관측이지 측정은 아니다.

## 3.12 phase-3 데이터 교정 before/after 게이트 (2026-09-09)

`KNOWN_ISSUES.md` 2026-09-09 판정문이 **"phase-3 iteration 은 늘리지 않고 TB-2·SWE-bench
before/after 로 검증"** 이라고 정했다. before = **phase-2 최종(602)**, after = phase-3 완주.

교정된 결함 ①②(swe_v3 `tools` 컬럼 부재, opencode 스키마가 MCP 형이라 템플릿이 빈값 렌더)는
**비학습 스팬(context)** 의 결함이다. 학습된 트라젝토리는 멀쩡하고, 빠진 것은 *"선언된 tools
블록을 보고 호출하는"* 조건부뿐이다. 그래서 **resolved 만 보면 교정 효과가 안 보인다** —
조건부가 붙었는지는 형식·추출 지표에 먼저 나타난다.

| 게이트 | 출처 | before(602) 기대 | 교정이 들었다면 |
|---|---|---|---|
| SWE **빈 패치율** | `results_swe.json` › `swe_detail.empty_patch_rate` | ≈40% (i1500 40.4 / i1800 40.6) | 하락 |
| SWE **채점기준 적중** | `swe_detail.resolved_given_completed` | i1500 13.5 / i1800 10.7% | 상승 |
| TB-2 **명령 추출률** | `results_terminal.json` › `terminal_detail.extraction_rate` | 미측정 (1태스크 스모크 50%) | 상승 |
| TB-2 예외 분포 | `terminal_detail.exception_stats` | — | parse 계열 감소 |

**wandb 에는 올리지 않는다.** 이 값들은 평가 결과가 아니라 진단이고, wandb 는 평가 결과만
올린다(사용자 결정 2026-09-01). 결과 JSON 과 이 표에서 본다.

왜 이 셋인가 — 관측된 형식 손실이 한 자리에 모인다:

| 실측 | 값 |
|---|---|
| SWE `RepeatedFormatError`(빈 패치 176건 중, iter900) | 51건 = 전체의 10.2% |
| TB-1 `parse_error` (iter1500, 640 trial) | 64건 = 10.0% |
| TB-2 명령 추출 (1태스크 스모크, 2026-09-07) | **4/8 = 50%** |
| 도구 영역 학습 토큰 중 `<think></think>` 타깃 (KNOWN_ISSUES 09-09) | p1 ≈35% → **p2 43.6%** → p3 22.9% |

평가 하니스는 전부 thinking ON 이다. 즉 형식 손실이 "모델이 형식을 모른다" 가 아니라
**학습 모드와 평가 모드의 불일치**일 수 있다. phase-3 는 이 비중을 22.9% 로 낮추므로,
before/after 로 그 가설이 갈린다.

구현: `run_swe.sh` 파서가 `empty_patch/errors/completed` 를 `swe_detail` 에 넣고,
`run_terminal_tb2.sh` 가 실행 후 `trajectory.json` 들을 훑어 `extraction_rate` 를 넣는다
(검증 2026-09-09: 스모크 잡 재집계 8스텝 중 4 = 50%, 수기 계수와 일치).

## 3.13 τ³-bench 온보딩 — tau2-bench v1.0.1 + think 분리·복원 프록시 (2026-09-14)

**왜**: 도구 사용 능력을 tool + agent + **user** 3자 상호작용에서 재는 표준 벤치. 기술보고서·리더보드가 τ 계열을 쓴다.
사용자 결정(09-14): τ³ v1.0.1 `base` split(τ² 대비 과제 수정 75건+ → τ² 보고서 수치와 직접 비교 불가, ckpt 간 추이가 목적),
4 trials, 상대역 = `gemma-4-12B-it` @ `https://gemma4.withai.cj.net:10206/v1`(외부 vLLM, 비용 0 — 매 ckpt API 비용을 피한다).

**구성**
```
tau2 run (sub1, tools/tau2-bench/.venv)
  ├─ agent : litellm openai/alpha ─▶ tau_proxy :8110 ─▶ lb_proxy :8100 ─▶ 에이전틱 fleet (TOOLS=1, reasoning 파서 없음, 262144)
  └─ user  : litellm openai/gemma-4-12B-it ─▶ 외부 엔드포인트 (tools 없음)
```
- docker 불요(도구 = JSON DB 위의 순수 파이썬) → gpu06 컨테이너·역터널 없이 sub1 직접. 설치 `install_tau2.sh`(NFS, 멱등).
- 과제 수(v1.0.1 실측, `tau_tasks_count.py`): airline 50 · retail 114 · telecom 114 (`base`), mock 10.
- **telecom 은 조건부**: dual-control 이라 상대역이 도구를 써야 하는데 기본 상대역 엔드포인트는 `tools` 요청을 400 으로 거절
  (`--enable-auto-tool-choice` 미기동, 09-14 실측). preflight T2 가 판정해 건너뛰고 `tau_detail.skipped_domains` 에 기록.
- 상대역 12B → 리더보드(gpt-5.2)·보고서 수치와 **비교 불가**. 상대역이 고정이라 ckpt 추이는 유효. 외부 비교치는
  `TAU_USER_LLM/TAU_USER_ARGS` 로 별도 런(비용 발생, 승인 후).

**왜 프록시인가 (규칙 5 vs 하니스)**: tau2 는 응답의 `content`·`tool_calls` 만 저장하고 `reasoning_content` 를 읽지도
재전송하지도 않는다. 에이전틱 fleet 는 G2 때문에 reasoning 파서 없이 떠서 content 가 `{think}</think>{answer}` 다.
그대로 두면 상대역과 COMMUNICATE 채점기가 think 를 발화로 읽고, 파서를 켜면 히스토리에서 think 가 사라진다
(`INTERLEAVED_THINKING.md` §7 규칙 5). `tau_proxy.py` 가 둘 다 해결한다:
(1) 응답에서 think 를 떼어 `content` 는 발화만(도구 턴은 `null`), `reasoning_content` 에 think;
(2) 다음 요청의 히스토리 assistant 턴에 원문(`<think>\n…</think>…`)을 바이트 동일 복원 — 템플릿 119행이 인라인 think 를
그대로 렌더한다. 키 = 체인 해시 `h_i = sha256(h_{i-1} ‖ canon(m_i))`, canon 은 think 유무·tool-call id·인자 직렬화에
불변이고 자기 턴까지 포함해 동시 trial 이 충돌하지 않는다. 첫 assistant(tau2 합성 인사 "Hi! How can I help you today?")는
생성된 적이 없어 `miss_first_assistant` 로 따로 센다. seed 제거(k 반복 붕괴 방지)·`skip_special_tokens=false` 강제.
한계: 스트리밍 미처리(tau2 는 안 씀), `choices[0]` 만, 문장+도구호출 혼합 턴은 프로토콜 미강제라 오류는 아니지만
그 문장은 사용자에게 전달되지 않는다(`mixed_content_and_tools` 로 정량화 — 높으면 학습 포맷 신호).

**litellm 함정**: `LITELLM_MODEL_REGISTRY_PATH` 는 mini-swe-agent 의 기능이지 litellm 의 것이 아니다. tau2 는 비용 계산
실패를 잡아 0 으로 두지만 호출마다 ERROR 로그 → venv 의 `.pth` import 훅(`_tau2_litellm_registry`)이
`configs/alpha_model_registry.json` 을 `litellm.register_model()`. (`sitecustomize.py` 는 데비안 시스템 파이썬 것이 먼저 잡혀
무효 — 실측.) 키는 요청 모델명(`openai/alpha`)과 **응답 모델명**(`alpha`, `google/gemma-4-12B-it` — vLLM 이 정식 이름을
돌려준다) 둘 다 필요.

**결과 규약** (`tau_combine.py`): `results_tau.json` → `tau_retail/airline/telecom` 과 도메인 평균 `tau_bench`, 각
`pass1..pass4,none`(과제별 comb(c,k)/comb(n,k) 평균, INFRASTRUCTURE_ERROR 제외 = `tau2.metrics` 와 동일, venv 에서 tau2 자체
계산과 대조) + 진단 `no_answer`(하니스 실패율: infra/unexpected/user_error/timeout/agent_error) · `think_closed`(프록시).
무효 → `no_answer=1.0`: 부분 표본 · 결과 부재 · sims < tasks×trials · 복원 켠 채 miss_rate>5% · think 미관측 · pass^1 불일치.

**검증 기록**
| 항목 | 결과 |
|---|---|
| 유닛 테스트 | `tests/test_tau_proxy.py` 14 + `tests/test_tau_combine.py` 5 = **19 passed** (분리/복원 왕복·trial 격리·혼합 턴·unclosed·동시성 16스레드·무효 규칙) |
| 상대역 preflight (실측 09-14) | 등록 4키 · chat OK · 비용 0.0 · **tools 거절**(USER_TOOLS=0) · `extra_body`/`top_p` 수용 |
| 라이브 fleet 왕복 (도구 없음, 3턴) | think 분리 3/3, 히스토리 복원 reinlined 2 / miss 0. 렌더(`tau_render_check.py`): 도구 시나리오 히스토리 2/2 preserved, 복원 ON 464 vs OFF 323 토큰(차 141 = think). 비도구 시나리오는 템플릿이 자르므로 복원 무효 — 규칙대로 |
| 템플릿 동일성 | tokenizer_v5 = p2 iter602 hfmodel = phase-1 2448 hfmodel (sha1 동일) → 로컬 렌더 = vLLM 렌더 |
| **도구 경로 스모크** (`tau_smoke.sh`, airline 2×1, 09-14 16:21) | **10/10 PASS** — 도구 호출 32턴 파싱, think 분리 34/34(reasoning 필드), 히스토리 복원 reinlined 273 / miss 0, `<think>` 누출 0, reward 채점 2/2, 하니스 실패 0. 렌더: 히스토리 12턴 중 11 preserved(첫 턴 = 합성 인사), 복원 ON 8,295 vs OFF 6,396 토큰 |
| **복원 ON/OFF differential** (airline task 0~4 × 1 trial, 09-14) | 아래 표. **n=5 관측치 — 방향이 예상과 반대**, 첫 본 측정은 양쪽 모두 돌려 n=164 로 확정할 것 |

**vLLM 0.25.1 parser engine 함정 (첫 스모크에서 발견, `KNOWN_ISSUES` 09-14 "에이전틱 fleet 가 도구 호출 턴마다 `</think>` 를 삼켰다")**: TOOLS=1 만 켠 fleet 는 **모든 응답**(요청의 tools 선언·도구 호출 여부와 무관 — 서버 플래그로 켜짐, 동료 세션 실측 6/6)의
content 가 `{think}{답변}` 으로 **`</think>` 마커 없이 붙어** 나온다(qwen3_xml = Qwen3ParserToolAdapter 가 THINK_END 토큰을
터미널로 소비, reasoning 파서가 없으면 분리하지 않고 마커만 제거). 프록시 텍스트 분리 불가 → 복원 miss 57%, 상대역이
think 를 읽음. **τ fleet 는 `REASONING_PARSER=nemotron_v3` 필수**(`serve_alpha.sh` env, `run_suite.sh` τ 단계가 재기동, 게이트
T1b 가 검사). think 는 `reasoning` 필드로 오고 프록시가 그것을 캐시·복원한다. 같은 TOOLS=1 fleet 를 쓰는 **SWE·TB-2 하니스도
같은 content 를 본다** — 히스토리에 마커 없는 think 텍스트가 답변 자리에 들어간다(템플릿은 `<think></think>` 를 앞에 붙임).
G2 게이트 전제(`</think>` 관측)는 tool 파서 없는 fleet 에서만 성립한다. 조치는 사용자 결정 대기(측정 조건 변경).

**differential — 복원 ON vs OFF (airline task 0~4, 1 trial, temp 1.0, 같은 fleet·상대역)**

| | sims | reward | 평균 agent 턴 | 종료 사유 | agent prompt 토큰 | 문장+도구호출 혼합 턴 | 프록시 |
|---|---:|---|---:|---|---:|---:|---|
| **ON** (규칙 5, 기본) | 5 | **0/5** | 30.4 | too_many_errors 3 · max_steps 1 · user_stop 1 | 2,170,586 | 37/41 | reinlined 5,216 / miss 0 / from_field 147 |
| **OFF** (`TAU_REATTACH=0`) | 5 | **3/5** | 10.4 | too_many_errors 2 · user_stop 3 | 290,071 | 5/29 | reinlined 284 / miss 0 / from_field 58 |

관찰: 복원 ON 에서는 히스토리가 정확히 `<think>…</think>답변` 으로 재현됨을 확인했는데도(중복·결함 없음, 렌더 preserved),
모델이 "조회하겠다"고 말만 하고 도구를 부르지 않는 턴을 반복해 100턴(max_steps) 을 채운 sim 이 있다. OFF 는 빈
`<think></think>` 히스토리에서 오히려 도구를 부르고 3/5 를 푼다. 학습 분포에 충실한 쪽(ON)이 점수가 낮다 → **기본값
결정은 사용자 몫**(n=5). 도구 오류 유형(스모크 2 sims): 스키마 설명의 **예시값을 실제 ID 로 사용**(`ZFA04Y` 5회,
`sara_doe_496`), 없는 인자명(`search`·`dest`·`code`), 없는 도구(`list_all_flights`) — 도구 선언 렌더는 중첩 `$defs`
까지 완전하므로 모델 행동이다.

**실행**: `bash eval_sft/run_tau.sh <RUN_TAG> [N=0] [TRIALS=4] [W=8]` (스위트는 `run_suite.sh` 에이전틱 단계, `TAU_N/TAU_TRIALS/TAU_W`).
스모크: `bash eval_sft/tau_smoke.sh <HF_CKPT>` (sub1, fleet 자동 기동·종료).

## 3.14 에이전틱 fleet 의 추론 분리 — reasoning 파서 · 게이트 A5 · SWE 추론 복원 (2026-09-14)

사고 서사: `KNOWN_ISSUES.md` 2026-09-14. 이 절은 **평가 조건과 운영 규칙**이다.

### 핵심 사실 — TOOLS=1 fleet 는 reasoning 파서가 없으면 **모든 응답**의 `</think>` 를 잃는다

vLLM 0.25.1 의 파서 엔진(qwen3_xml)은 서버 플래그 `--enable-auto-tool-choice --tool-call-parser qwen3_xml`
로 켜지고, reasoning 파서가 없으면 THINK_END 를 소비하고 마커만 떨어뜨린다. **요청의 `tools` 필드와 무관하다.**

| 실측 (sub1, hfmodel_0002448, 2026-09-14) | 파서 없음 (:8000) | nemotron_v3 (:8001) |
|---|---|---|
| A5 (도구 선언 + thinking ON) | **FAIL** (결론 2건 fail 2) | **PASS** (reasoning 필드 319자) |
| 도구 미선언 + thinking ON × 6 | **6/6 추론문이 마커 없이 JSON 앞에 붙음** | content 가 `{` 로 바로 시작, reasoning 필드 4,476자 |

"도구를 안 보내면 도구 파서 경로를 안 타니 안전하다" 는 가정은 **틀렸다** — T1 fleet(TOOLS=0)에서 G2 가
늘 `</think>` 를 본 것과 차이는 TOOLS 플래그뿐이다.

### 세 하니스는 서로 다른 추론 조건에서 측정됐다 (iter300~1800)

| 하니스 | 요청 tools | response_format | 실제 추론 조건 | 근거 |
|---|---|---|---|---|
| SWE (mini-swe-agent) | 보냄 | 없음 | thinking ON, `</think>` 소실 → 이력이 `<think></think>`+추론문으로 재렌더 | 보존 궤적 0/148,836턴 |
| **TB-1** (terminus v1) | 안 보냄 | **json_schema** | **추론이 문법으로 차단** — 한 번도 생성되지 않음 | iter1500 에피소드 15,819건 100% `{` 로 시작 |
| TB-2 (terminus-2) | 안 보냄 | 없음 | thinking ON, `</think>` 소실 → 추론문이 JSON 앞에 붙음 | 위 6/6 · 09-07 스모크 명령 추출 4/8 |

TB-1 은 KNOWN_ISSUES 09-09 의 "평가 하니스는 전부 thinking ON" 전제가 성립하지 않는다. 계열 내 추이는
같은 조건끼리라 유효하나 세 하니스를 서로, 또 외부 수치와 비교할 수 없다.

### fleet 플래그 규칙 — 티어마다 갈린다

| fleet | TOOLS | reasoning 파서 | 이유 |
|---|---|---|---|
| T1 · T3 | 0 | **없음** | T1 채점(`split_think`)이 content 의 `</think>` 로 사고 마감률을 잰다. 파서가 켜지면 전부 0 |
| 에이전틱 (SWE · TB-2) · τ³ | 1 | **nemotron_v3** | 없으면 위 결함. `run_suite.sh` 에이전틱 블록 `fleet_up … 1 nemotron_v3`(30df05b) |

τ³ 단계는 에이전틱 fleet 를 그대로 쓴다(재기동 제거 3799c62).

### 게이트 A5 — think + 도구호출 동시 경로

`check_agentic_gates.py`. thinking ON + tools 선언으로 실제 호출을 시키고, 그 턴에서 추론이 **reasoning 필드로
분리**되거나 **content 의 `</think>` 로 보존**되면 PASS. 결론 난 관측 2개가 일치하면 멈추고 갈리면 3개째 다수결,
최대 8회. 경로를 한 번도 못 보면 FAIL(미관측을 통과로 쓰지 않는다).

| 러너 | `--tool-path` | 이유 |
|---|---|---|
| `run_swe.sh` · `run_tau.sh` · `tau_smoke.sh` · 스위트 에이전틱 블록 | required(기본) | 결함이 측정을 오염 |
| `run_terminal_tb2.sh` | **required** | 도구 미선언이어도 TOOLS=1 fleet 에서 결함 발생(위 6/6) |
| `run_terminal.sh` (TB-1) | report | json_schema 라 추론 자체가 없어 결함 무관 |

단위 테스트 `tests/test_check_agentic_gates.py` 18건 — 결함 표본은 iter1800 SWE 궤적의 vLLM 원응답 3건.

### SWE 추론 복원 — 컨테이너 내 tau_proxy

mini-swe-agent → **컨테이너 안 tau_proxy :8110** → :8199 역터널 → sub1 lb_proxy. 터널 불변. TB-2 도 자기 프록시
(:8111)를 둔다 — 아래 "TB-2 추론 복원".

| `SWE_THINK` | 동작 |
|---|---|
| `restore`(기본) | 이력에 추론 인라인 — 학습 형식(도구 시나리오 interleaved) |
| `strip` | `--no-reattach` — 이력의 추론 필드를 떼어 버린다(`reasoning_field_dropped`) |

**mini-swe-agent 는 이력에 `reasoning_content` 를 다시 실어 보낸다**(tau2 와 다름). 그래서 SWE 에서 복원은 캐시
(`reinlined`)가 아니라 **필드 인라인**(`reasoning_field_inlined`)으로 일어나고 miss 가 원천적으로 0/0 이다.

| 스모크 (1인스턴스 · step 12) | 필드 인라인 | 필드 버림 | upstream 이력의 `<think>` |
|---|---:|---:|---:|
| restore (수정 전) | 66 (=1+…+11) | — | 11/11 |
| strip (수정 전) | **66** | — | **11/11** ← strip 이 안 됨 |
| **restore (77040d3 후)** | **66** | 0 | **9/9** |
| **strip (77040d3 후)** | 0 | **66** | **0/10** |

**tau_proxy 버그와 수정**: `on_request()` 의 필드 인라인 블록이 `self.reattach` 를 확인하지 않았다(캐시 경로만 확인).
tau2 는 필드를 안 보내 드러나지 않았다. integrate-tau3-bench-alpha 세션이 77040d3 에서 고쳤다 — reattach OFF 면
필드를 떼기만 하고 `reasoning_field_dropped` 로 센다. 수정 후 SWE 경로 실측에서 두 모드가 갈렸다(위 표).
참고: strip 에서도 `reinlined`(캐시 적중)는 증가한다. 실제 삽입은 `restored`(a834e48)가 센다 — strip 은 0.

**무효 규칙** (`run_swe.sh` 파서, `swe_detail.invalid`). 복원 판정은 tau_proxy 의 **`restored`**
(= 이력 content 에 실제로 넣은 총수, 캐시 복원 + 필드 인라인, a834e48) 하나로 한다:
- 프록시 통계 없음
- 복원 켠 채 `miss_rate > 0.05` (τ³ 와 동일 — SWE 는 필드 경로라 miss 가 0/0 이어서 사실상 발동 안 함)
- 추론 미관측 (`think_stripped + think_from_field == 0`)
- **restore 인데 다회차 요청에서 `restored == 0`**
- **strip 인데 `restored > 0`** — 77040d3 이전 버그의 재발 방지

`restored` 가 없는 과거 통계는 모드별로 대체한다. restore 는 `reinlined + reasoning_field_inlined`, strip 은
`reasoning_field_inlined` 만 — **strip 의 `reinlined` 는 캐시 적중일 뿐 적용되지 않으므로 합산하면 정상 strip 을
누수로 오판한다**(수정 후 strip 실측: 적중 66 · 이력 0/10). 스모크 통계 4건 재판정으로 확인했다.

복원이 낫다고 가정하지 않는다: τ airline 5과제 ON 0/5 vs OFF 3/5(§3.13). 복원은 매 턴 프롬프트를 추론만큼 키워
긴 궤적에서 262144 창 초과를 늘릴 수 있다. 77040d3 로 ON/OFF 비교가 성립하므로 정식 측정에서 함께 잰다.

### TB-2 추론 복원 — 컨테이너 내 tau_proxy :8111 (2026-09-14)

**설계 규칙 (사용자 2026-09-14)**: 도구를 쓰는 에이전트 궤적은 **restore**(이전 턴 추론 유지), 도구 없는 일반 대화는
**strip**(DSV4 방식). 학습 데이터도 이 기준으로 만들었다. 터미널 학습셋 `nvidia/Nemotron-Terminal-Corpus`
(`ntc_v1_*`)는 `--keep-history-think` 로 구웠다(bins `data.stats.json` `keep_history_think: True`, 디코드 실측
마지막 user 이전 assistant 턴 56/56 think 보존). 폐기된 alpha-SFT-Terminal 은 해당 없음.

**평가는 strip 으로 돌고 있었다.** terminus-2 는 도구를 선언하지 않고 터미널 출력을 **user 메시지**로 보낸다
(`lite_llm.py` 299행). 템플릿(28행)은 이를 비도구 시나리오로 보고 마지막 user 이전의 think 를 자른다.

restore 에는 세 가지가 **모두** 필요하다. 하나라도 빠지면 오류 없이 strip 으로 돈다.

| 구성 | 없으면 | 이유 |
|---|---|---|
| tau_proxy (:8111) | 하니스가 추론을 못 받는다 | vLLM 0.25.1 은 추론을 **`reasoning`** 키로 준다. harbor(`lite_llm.py` 428행)는 **`reasoning_content`** 만 읽는다. 프록시가 응답에 `reasoning_content` 를 써 넣고, 다음 요청에서 이력 content 에 `<think>\n…</think>` 로 인라인한다 |
| `--ak interleaved_thinking=true` | 하니스가 추론을 이력에 안 싣는다 | terminus-2 `chat.py` 114행. 이 인자는 그 한 줄에만 쓰인다 |
| `truncate_history_thinking=false` | 템플릿이 자른다 | `llm_call_kwargs.extra_body.chat_template_kwargs` 로 전달 |

**첫 시도는 조용히 실패했다**: 프록시 없이 아래 두 개만 켰다. upstream 기록 요청에서 kwargs 는 도착했지만
이력 assistant 2턴 중 `reasoning_content` 0, 재전송 prompt_tokens 차이 0. 인자가 받아들여졌다고 동작한 것은
아니다 — **upstream 요청 본문과 prompt_tokens 로 확인한다.** (`/tokenize` 는 `reasoning_content` 를 버려 렌더
검증에 쓸 수 없다.)

**검증 (sub1 phase-1 swap hfmodel_0002448, `fix-git` 1과제 × 모드별 1회, 하니스 → 프록시 → 역터널 → 기록기 → lb_proxy)**

| 항목 | restore | strip |
|---|---|---|
| upstream 이력 assistant 의 `<think>` | **매 요청 전부** (425턴 요청까지 0+1+…) | **0** / 263 |
| upstream `chat_template_kwargs` | `truncate_history_thinking: false` | 없음 |
| 이력의 `reasoning_content` 필드 잔존 | 0 (프록시가 인라인으로 정규화) | 0 (하니스가 안 실음) |
| 프록시 `restored` | **90,525** = 0+1+…+425 | **0** (캐시 적중 34,716 은 미적용) |
| miss · upstream 오류 | 0 · 0 | 0 · 0 |
| 무효 판정 | 없음 | 없음 |

| 렌더 differential (restore 19턴 요청, prompt_tokens) | 값 |
|---|---:|
| 그대로 (인라인 think + truncate=false) | **10,993** |
| `chat_template_kwargs` 제거 (템플릿 기본) | 7,914 (−3,079) |
| 인라인 think 제거 | 7,914 (−3,079) |

복원 형식은 `<think>\n{추론}</think>{JSON 답변}` — 템플릿 114행이 학습 행 `reasoning_content` 를 렌더하는 공식과 같다.

kwargs 제거와 think 제거가 **같은 토큰 수**다 — 템플릿 기본값은 이력 추론을 전부 자른다는 뜻이다.

**모델 행동 관찰 — restore 가 `</think>` 미종결 턴을 되먹여 루프를 굳혔다 (n=1, 판정 아님)**

같은 스모크에서 두 모드 모두 시간 초과(900초)로 0점이었지만 궤적이 달랐다.

| 같은 과제·체크포인트 | restore | strip |
|---|---:|---:|
| 에이전트 스텝 | 425 | 262 |
| 명령 추출 | 20 (4.7%) | 80 (30.5%) |
| **추론만 있고 답변 빈 스텝** (`steps_reasoning_only`) | **385 (91%)** | 38 (15%) |
| 첫 미종결 턴 → 이후 | 턴 38 → **끝까지 연속** | 스텝 1 → 산발, 매번 회복 |

미종결 턴은 모델이 JSON 을 다 쓰고 `</think>` 를 닫지 않은 턴이다. reasoning 파서가 출력 전체를 추론으로 분류해 답변이
비고, 하니스는 "No valid JSON found" 로 되받는다. **두 모드 모두에서 나왔다** — restore 가 만든 실패가 아니다.
restore 에서는 그 턴이 이력에 `<think>{JSON}</think>` + 빈 답변으로 되돌아가고, 모델이 그 모양을 그대로 베꼈다
(마지막 5턴 think 길이 841자 동일). strip 에서는 같은 턴이 빈 답변으로만 보여 고립됐다.

해석의 한계:
- 표본이 과제 1개 × 1회다.
- 스모크 체크포인트(phase-1 swap, 09-01 시작)는 **NTC 를 학습하지 않았다**(설정 스냅샷에 `ntc_v1` 없음).
- NTC 학습 턴은 깨끗하다: 6개 멤버 표본 7,683턴 전부 `</think>` 종결 + JSON 답변, 미종결·think 안 JSON 0.

NTC 를 학습한 체크포인트에서 `steps_reasoning_only` 가 0 에 가까우면 이 경로는 드러나지 않는다. 러너가 매 실행마다
이 수치를 기록한다.

| `TB2_THINK` | 동작 |
|---|---|
| `restore`(기본) | 위 세 구성 모두 — NTC 학습 조건 |
| `strip` | 프록시 `--no-reattach`, 나머지 둘 끔 — 비교용 |

프록시 포트를 SWE(:8110)와 나눈 이유: 같은 컨테이너에서 겹쳐 돌면 기동 시 `pkill` 이 상대 프록시를 죽인다.
업스트림 대기 상한은 `--timeout 7200` — 65K 토큰 생성이 기본 1800초를 넘으면 프록시가 502 로 끊는데, 프록시가
없던 시절에는 없던 실패다. 스모크용 과제 필터 `TB2_INCLUDE=<glob>`(harbor `-i`, 결과는 부분 표본으로 표시).

**무효 규칙** (`terminal_detail.invalid`, `run_swe.sh` 와 동일 + 하니스 측 확인): 프록시 통계 없음 · 추론 미관측 ·
restore 인데 다회차에서 `restored == 0` · restore 인데 `miss_rate > 0.05` · strip 인데 `restored > 0` ·
restore 인데 궤적 `reasoning_content` 가 있는 스텝 0.

과거 TB-2 수치(09-07 스모크 등)는 strip 이다. 복원은 매 턴 프롬프트를 추론만큼 키운다(위 19턴에서 +39%).

## 3.10 반복 실행 워크플로 (학습 중 체크포인트마다)

학습이 진행되며 체크포인트(300 iters마다)가 나오면 반복 평가한다. 스크립트는 모두
`examples/alpha/eval_sft/`. 멱등적(이미 한 건 skip)이고 GPU 정리를 내장한다.

| 스크립트 | 역할 |
|---|---|
| **`eval_new_ckpt.sh <RUN_DIR> <ITER> [stages]`** | **전 티어 정본 진입점** — 변환 → `run_suite.sh`(티어별 fleet 교체·게이트 G1~G3·A1~A5·`summarize.py` 판정·집계·wandb). 에이전틱을 포함하면 반드시 이 경로 |
| `eval_ckpt.sh <RUN_DIR> [ITER\|latest] [tiers]` | 08월 말 경량 래퍼: 변환→fleet(maxlen 49,152)→티어 실행→종료→집계. `summarize.py` 판정 출력·에이전틱 규약(§3.14) 없음 — **T1·T3 전용**(최종 SFT 계획의 300 iter T1 이 이것을 쓴다) |
| `eval_watch.sh <RUN_DIR> [tiers] [poll_s]` | 새 체크포인트 감시→자동 평가 (무인 반복). 중단: `touch <RUN_DIR>/.eval_watch_stop` |
| `serve_fleet.sh` / `lb_proxy.py` | 단일GPU 서버 N개 + 라운드로빈 프록시 (vLLM DP munmap 우회) |
| `stop_fleet.sh [GPUS]` | 프로세스그룹 SIGTERM→GPU 회수 검증→필요시 SIGKILL (누수 방지) |
| `run_tier1.sh` | T1 lm_eval chat-completions (mmlu_pro·gpqa·ifeval·aime25·hmmt) |
| `aggregate_results.py` | 전 체크포인트 결과→`results/TRACKING.md` 추이표 |

**wandb**: 결과는 프로젝트 **`alpha-post-eval`**에 학습 run 별로 로깅(iter=step, resume 누적).
post-train(`alpha-evals`)과 별도. `eval_ckpt.sh`가 집계 후 자동 업로드(WANDB=0 이면 skip).
수동: `python3 eval_sft/log_eval_wandb.py --results-dir eval_sft/results --run-tag <run>_iter<N>`.

**전형적 사용** (SFT 학습 중, sub1에서):
```bash
cd examples/alpha
# 무인 반복: 새 ckpt 나올 때마다 자동 평가
GPUS=0,1,2,3,4,5,6,7 bash eval_sft/eval_watch.sh outputs/<sft_run> t1 600
# 또는 특정 ckpt 1회:
GPUS=0,1,2,3,4,5,6,7 bash eval_sft/eval_ckpt.sh outputs/<sft_run> 300 t1
```
결과: `eval_sft/results/<run>_iter<N>/` (lm_eval 원자료) + `results/TRACKING.md`(iter별 추이).

### 측정 이력 — 체크포인트별 범위와 중단 기록 (2026-09-14 기준)

수치는 `results/TRACKING.md`(자동 생성)에만 둔다. 여기에는 **무엇을 쟀고 무엇이 왜 빠졌는지**를 적는다 — 집계기가
TRACKING.md 를 매번 새로 쓰므로 그 파일에 손으로 쓴 기록은 사라진다(09-09 중단 기록이 그렇게 지워졌다).

| 체크포인트 | T1 | T3 | T2 | SWE | Terminal | 비고 |
|---|---|---|---|---|---|---|
| full_0828 iter300·600·900 | ✅ | ✅ | ✅ | ✅ | ✅ TB-1 | AIME·HMMT 무효(사고 마감률) |
| swap iter1200·1500 | ✅ | ✅ | ✅ | ✅ | ✅ TB-1 | 1500 은 첫 변환 실패(09-05) 후 재실행 |
| swap iter1800 | ✅ | ✅ | ✗ | ✅ | ✗ | 사용자 결정(09-07): SWE 까지만 재고 중단 |
| p2 iter500 | ⛔ 48% | LogicKor 만 | ✗ | ✗ | ✗ | 09-09 T1 중단 — 데이터 결함 재변환이 sub1 GPU 필요(사용자). 결과 없음 |
| p2 iter602 | ⛔ 43% | ✗ | ✗ | ✗ | ✗ | 09-09 fleet 연결 끊김(`Connection closed`, 다른 세션 실행). 결과 없음 |
| **swap iter2448 (phase-1 최종)** | ⛔ **94.3%** | ✗ | ✗ | ✗ | ✗ | 09-14 11:07 KST `eval_ckpt.sh … t1,t3` 기동, ~16:05 KST sub1 이 τ³ 스모크·프로브로 재배정되며 결과 없이 중단. 프로브만 남음(identity FAIL · 유령 호출 1/33 PASS). **전 티어 재측정**(사용자): 21:22 KST 1차는 sub1 HOME 볼륨 ENOSPC 로 fleet 기동 실패(`KNOWN_ISSUES` 09-14) → 캐시 이전 후 21:51 KST 재기동. T1·T3 완료(09-15 03:39), 에이전틱 fleet 가 옮긴 컴파일 캐시의 절대경로로 사망 → 캐시 삭제 후 에이전틱·T2 만 07:36 KST 재기동 |

| **final_resume_0915 iter600 (SFT 최종 런)** | ✅ | ✅ | ✅ | ⏳ | ⏳ | **main1 GPU 0~6**(GPU 7 결함 제외, 변환 EP=6·fleet 7대) 09-16 23:18 → 09-17 06:00. AIME·HMMT 무효(사고마감 16%·12%, 전 ckpt 동일). 에이전틱은 **A4 1회 표본 오판**(모델이 도구 대신 되물음 → 파서 실패로 오독, A5 는 2/2 PASS)으로 건너뜀 → A4 를 A5 처럼 다중 표본·판정 분리(`judge_a4`)로 고친 뒤 09-17 09:55 `run_suite.sh … agentic` 재기동. 관찰: RULER single_2 131K **0.25**(phase-1 iter600 0.90) — 실패 43건 중 37건이 `0, 1`·`0.5, 1.0, …` 퇴행 출력(Reasoning-Off 조건, single_1 은 100%). LogicKor 31.2(phase-1 iter600 40.2). MMLU-Pro 49.5·GPQA 35.4·IFEval 61.1 은 phase-1 iter600(48.6/32.2/55.6) 우위 |

- **p2 계열은 재측정하지 않는다** — phase-2·3 계보가 09-13 폐기됐다.
- **09-14 이전 에이전틱 값(SWE·TB-1)은 reasoning 파서 없는 fleet 조건**이다 — 이력 `</think>` 가 전 턴에서 사라진 채
  측정됐다. 계열 내 추이는 같은 조건끼리라 유효하나 외부 수치·현행 규약 수치와 비교할 수 없다(§3.14). 표기 방식은
  사용자 결정 대기.
- lm_eval 은 끝에서만 결과를 쓴다 — T1 중단은 진행률과 무관하게 **결과 0**이다. 다른 세션이 sub1 을 쓰기 전에
  `suite_running.sh` 로 실행 여부를 확인한다.

**반복성 불변식**:
- lm_eval 0.4.12 고정, 태스크·few-shot·seed·gen 파라미터 러너에 하드코딩(`run_tier1.sh`).
- 서버 `max-model-len 49152`·`max_gen_toks 24576`(prompt 여유 확보, 400 방지). thinking 기본.
- **GPU 정리 규율**: 교체 시 `stop_fleet.sh`로 SIGTERM→회수확인. hard-kill 반복이 GPU 좀비
  누수를 만든다(2026-08-29 GPU0 사고). **GPU0 은 2026-08-30 회수 확인 후 복귀** — 실측
  78.6GiB 여유, bf16 matmul 222 TFLOP/s 정상. fleet 기본 GPU = **0~7 전부**.
  다만 GPU0 만 `nvidia-smi` 의 `utilization.gpu` 가 `[Not Found]` 로 나온다(컨테이너
  패스스루 계측 한계, 연산은 정상). GPU0 사용률로 fleet 건강을 판단하지 말 것.
- 변환은 `evaluate.sh`(forward_sanity ppl 게이트 포함) 재사용 — 잘못 변환된 ckpt는 게이트가 막음.

## 4. docker 부재 — 실측과 경로

main1·sub1 공통 실측(2026-08-29): sudo는 passwordless로 존재하나 CapBnd에
`cap_sys_admin` 없음 + Seccomp filter 활성(`unshare` EPERM) + `/dev/fuse` 없음.
→ 컨테이너 안에서는 root여도 dockerd/rootless podman/apptainer 전부 기동 불가.
Backend.AI 세션 생성 시점 설정이라 내부에서 해결 불가.

경로 3택 (SWE·T-Bench 공통, 미정):

| 경로 | 비용 | 비고 |
|---|---|---|
| 관리자 privileged 세션(또는 docker socket 마운트) | 0 | 정석. 유휴 노드 1대면 충분 |
| Modal/Daytona 클라우드 샌드박스 | SWE 회당 ~$50–150, TB 회당 ~$10–40 | 하니스 공식 지원. 추론은 sub1 vLLM이라 무과금 |
| 별도 docker 호스트 (**CPU 전용으로 충분**) | 호스트 확보 | 16–32코어/64GB/300GB↑ 권장. sub1→호스트 ssh 역터널로 엔드포인트 연결 |

## 5. judge — gemini-3.7-flash (확정 2026-08-29)

키: `examples/alpha/.env` GEMINI_API_KEY (gitignore됨). generateContent 라이브 호출 검증 완료.
Google Generative Language API v1beta 엔드포인트
(`models/gemini-3.7-flash:generateContent`). 러너는 provider-agnostic으로 작성해
나중에 OpenAI-호환 judge로도 교체 가능. 비용은 flash급이라 SFT 런 전체 수 달러 내외.
공식 관례 참고: SimpleQA-Verified=Gemini 2.5 Pro, Arena-Hard-v2=프런티어 judge.

## 6. 작업 큐

**완료 (2026-08-29~30)**

- [x] sub1 유휴·환경 실사, docker 가능성 판정 → gpu06 DinD 컨테이너 경로 확정
- [x] vLLM 0.25.1 + 플러그인 서빙, fleet(8 레플리카 + 라운드로빈 프록시)
- [x] **프론티어 벤치 규약 조사** — R1·Qwen3·Nemotron 카드, NVIDIA 재현 설정,
      simple-evals, Artificial Analysis 방법론 (§3.4)
- [x] **T1 5종 재작성** — 내장 lm_eval 태스크는 base 모델용이라 폐기 (§3.6)
- [x] **T2 RULER 재설계** — Reasoning-Off (§3.9)
- [x] **T3 러너 규약 통일** — `gen_common.py` 정본
- [x] **게이트 G1~G3 · A1~A4** — 투입 전 자동 검사 (§7)
- [x] **집계 정본화** — `bench_registry.py`, 무효 셀 자동 차단
- [x] **에이전틱 온보딩** — SWE-bench Verified 500 / Terminal-Bench core 0.1.1
- [x] **iter300 첫 유효 측정** (§3.8)

**남은 것**

- [ ] T2 RULER 본 측정 (롱 fleet 139,264 필요)
- [ ] 에이전틱 컨텍스트 초과 대응 — 106,496 로도 일부 초과. 추론 히스토리 누적이 원인
- [ ] iter600 이후 재측정으로 추이 확보 (수학 2종이 유효로 전환되는 지점 확인)
- [ ] 오케스트레이터 상시화 — ckpt 감시 → 변환 → `run_suite.sh` → wandb
- [x] τ³-bench 도구 경로 스모크 10/10 · differential (09-14, §3.13)
- [ ] **τ³ 첫 본 측정** (retail 114 + airline 50 × 4, ON/OFF 양쪽) → 복원 기본값 확정(사용자) → `run_suite.sh` 에이전틱 단계로 정례화
- [x] SWE·TB-2 fleet 의 `</think>` 소실 조치 — nemotron_v3 fleet · A5 · SWE/TB-2 추론 복원 (09-14, §3.14)
- [ ] τ³ telecom — 상대역 엔드포인트가 tools 를 받으면(`--enable-auto-tool-choice`) preflight T2 가 자동 포함
- [ ] **phase-1 최종 iter2448 기준선** — 09-14 T1 이 94.3% 에서 결과 없이 중단(§3.10 측정 이력). 전 티어 재측정 실행 중(09-14 12:51 UTC~, 사용자 결정: 현행 규약 첫 기준선)
- [ ] 09-14 이전 에이전틱 수치(iter300~1800 SWE·TB-1) 표기 — 구 조건 계열로 둘지 무효로 내릴지 (사용자 결정 대기)
- [ ] NTC 학습 체크포인트에서 TB-2 `steps_reasoning_only`·`think_unclosed_stop` 확인 → 미종결 턴 복원 제외 옵션 필요 여부 판단 (§3.14 관찰)
- [ ] 미착수 벤치: LiveCodeBench, MRCR. (T4 표준 11종은 범위 제외 — 사용자 결정 2026-08-30)

## 7. 투입 전 게이트 (2026-08-30 신설, 필수)

2026-08-30 사고의 재발 방지. **셋 다 통과해야 수치를 `results/TRACKING.md` 에 기입한다.**
실행: `python3 eval_sft/check_gates.py --hf-dir <HF_CKPT> --base-url <URL>`

| # | 게이트 | 판정 기준 | 자동화 |
|---|---|---|---|
| G1 | **변환 산출물 eos 정합** | `generation_config.json` 존재 + `eos_token_id` 가 챗 종료 토큰(`<\|im_end\|>`)을 포함 | `tools/emit_generation_config.py` — `run_convert.sh` 가 MG→HF 마다 자동 실행, 실패 시 exit 1 |
| G2 | **태그 관측 가능성** | 서빙 응답에 `</think>` 가 살아있음 | 요청에 `skip_special_tokens: false` (NVIDIA Ultra 설정과 동일). **체크포인트 tokenizer 를 고치지 않는다** |
| G3 | **서빙 스모크** | 쉬운 질문 1건이 `finish_reason=stop` + `content` 비어있지 않음 | `check_gates.py` |

G3 의 어려운 질문은 **진단**이지 게이트가 아니다. `finish=length` + `content` 없음은
설정 결함이 아니라 모델 미성숙 신호다 — 그 구분이 2026-08-30 사고의 핵심이었다.

### 에이전틱 게이트 A1~A3 (SWE·Terminal)

T1 의 G1~G3 에 더해 세 전제가 더 있다. 하나라도 깨지면 에이전트가 매 스텝 실패하고
0점이 나오는데, **그 0점은 "모델이 못 푼다"와 구분되지 않는다** — 2026-08-30 SWE 0/20 ·
Terminal 0/10 이 그 상태였다. 실행: `python3 eval_sft/check_agentic_gates.py --base-url <URL>`

| # | 게이트 | 판정 | 실패 시 |
|---|---|---|---|
| A1 | `tool_choice=auto` 수용 | HTTP 200 | fleet 를 **TOOLS=1** 로 재기동. T1 용 fleet 는 이 플래그 없이 뜬다 |
| A4 | 파서가 모델 형식을 **실제로 파싱** | `tool_calls` 가 채워짐 | **`TOOL_PARSER=qwen3_xml`**. alpha 는 XML `<function=…><parameter=…>` 형식을 배웠다 — hermes(JSON 본문)는 A1 을 통과하고 여기서 걸린다 |
| A2 | 컨테이너 역터널 | 컨테이너 `:8199` → 200 | `ssh sub1 'bash /home/work/vidsearch/tools/start_swe_tunnel.sh'` |
| A3 | 컨테이너 디스크 | SWE 300GB / Terminal 150GB 여유 | `docker image prune` |
| A5 | think + 도구호출 턴에서 추론 분리 | thinking ON + tools 로 호출을 시켜 추론이 reasoning 필드 또는 content 의 `</think>` 로 남음. 결론 2건 일치 시 종료, 최대 8회, 미관측 = FAIL | fleet 를 `REASONING_PARSER=nemotron_v3` 로 재기동. TB-2 도 required(도구 미선언이어도 결함 발생), TB-1 만 report (§3.14) |

에이전틱 fleet 는 **`--max-model-len 106496`** 로 띄운다(T1 의 40960 은 좁아 `ContextWindowExceededError` 가 난다). litellm 은 `alpha_model_registry.json` 의 `max_input_tokens` 로 초과를 판정하므로 서빙 창과 함께 올려야 한다. 러너는 `LITELLM_MODEL_REGISTRY_PATH` 를 export 한다 — 미등록 모델은 비용 계산에서 죽는다.

### τ³-bench 게이트 T1·T2 (`run_tau.sh` 내장, 2026-09-14)

| # | 게이트 | 판정 | 실패 시 |
|---|---|---|---|
| T1 | `tau_proxy` 기동·fleet 통과 + 복원 hit-rate | `/stats` 200, `/v1/models` 200; 런 후 `miss_rate ≤ 5%`, `think_stripped > 0` | miss 는 하니스가 히스토리를 바꿔 보낸 것, think 0 은 경로 우회/파서 ON 의심 → 셀 무효(`no_answer=1.0`) |
| T1b | fleet 가 think 를 `reasoning` 필드로 분리하는가 | 응답에 reasoning 필드, content 에 `</think>` 없음 | `REASONING_PARSER=nemotron_v3 TOOLS=1` 로 fleet 재기동 (tool 파서만 켜면 `</think>` 소실, §3.13) |
| T2 | 상대역 LLM 응답 + tools 수용 여부 | chat 비어있지 않음(필수) · tools 200 이면 `USER_TOOLS=1` | chat 실패 → 중단. tools 거절 → telecom 자동 스킵·사유 기록 |
| 렌더 | 복원된 요청이 템플릿에서 `<think>\n…</think>` 로 남는가 | `tau_render_check.py last_request.json` 보존 ≥1, 'other' 0 | 프록시 복원 또는 템플릿 경로 의심 |

### 컨테이너 디스크 — 무엇이 쌓이고 무엇을 지우나 (2026-08-31 실측)

에이전틱을 돌리면 컨테이너 호스트 디스크가 준다. **누수는 아니다** — 실측에서 정지
컨테이너 0개였고 mini-swe-agent 가 매 인스턴스 후 정리한다.

| 항목 | 크기 | 처리 |
|---|---:|---|
| `sweb.eval` 태스크 이미지 498개 | 479.7 GB (명목 합산) | **실행 후 정리한다 (09-08 표준 정책).** 인스턴스 이미지는 base → environment → instance 3층 중 맨 위 얇은 층이라 레이어 대부분이 공유된다. 08-31 에는 "남긴다" 로 적었으나 공유 레이어를 빼고 계산한 과장이었다(`KNOWN_ISSUES` 09-08) |
| build cache | 48 GB (회수가능 31.8) | **매 실행 후 회수** |
| 미사용 볼륨 · 정지 컨테이너 · dangling 이미지 | 소량 | 회수 |
| Terminal-Bench 태스크 이미지 | 0개 | 누적하지 않는다 — 태스크마다 빌드하고 정리한다 |

**`bash eval_sft/docker_gc.sh`** 가 회수를 담당하고 `run_suite.sh` 의 에이전틱 단계가
끝날 때 자동 호출된다. 기본값이 `sweb.eval` 이미지 정리이고, 재취득 시간을 아껴야 할 때만
`--keep-images` 로 남긴다. 컨테이너 `/opt` 의 에이전트 산출물(궤적·세션 기록)은 docker 명령에
안 보이는 누적원이라 실행별 디렉토리를 최근 `GC_KEEP_RUNS`(기본 2)개만 남긴다(`KNOWN_ISSUES` 09-07).

게이트 A3 은 `df` 여유에 **build cache 회수 가능량을 더해** 판정한다 — 여유만 보면
실제보다 적게 보인다(실측: 여유 589GB + 회수가능 31.8GB).

**부분 표본은 무효로 기록한다.** `run_swe.sh`/`run_terminal.sh` 에 N 을 주면 결과 JSON 에
`subsampled=true` 와 `no_answer=1.0` 이 박혀 집계기가 `무효` 로 표시한다 — NVIDIA 재현
문서의 *"Never report sub-sampled / limited runs"* 를 코드로 강제한 것이다. 기본은 전량
(SWE Verified 500 / terminal-bench@2.0 89 × 8회).

부수 불변량:
- `--max-model-len` ≥ `max_gen_toks` + 프롬프트 최대치. 32768 모델길이에 32768 생성예산은 성립 불가.
- 장시간 러너·프로브는 `setsid` + NFS 로그로 분리 실행 (세션 종료에 죽지 않도록).
- 한 번에 한 변수만 바꾼다. 길이·온도·파서·모델 디렉토리를 동시에 바꾸면 원인 분리가 불가능하다.
- 부분 표본 결과는 기록하지 않는다 (NVIDIA 재현 문서: "Never report sub-sampled / limited runs").

### 검색 에이전트 게이트 (phase-2)

phase-2 에 편입한 web-search 도구 데이터(Nemotron-SFT-Agentic-v2 search split)의 능력 게이트.
홀드아웃 300문항(`.../Nemotron-SFT-Agentic-v2/splits/search_heldout300.jsonl`, **학습 미투입**)에
두 게이트를 건다. 하네스는 `eval_sft/search_agent_eval.py` 하나이고 vLLM 서빙만 있으면 된다.

```bash
python3 eval_sft/search_agent_eval.py --selftest                    # 서버 불요 (24 검사)
python3 eval_sft/search_agent_eval.py --base-url http://HOST:PORT/v1 --model alpha \
    --gate format --n 300 --seed 0 --tag <ckpt>                     # 게이트 1
python3 eval_sft/search_agent_eval.py --base-url http://HOST:PORT/v1 --model alpha \
    --gate live --backend replay --n 5 --tag smoke                  # 루프 스모크 (API 키 불요)
TAVILY_API_KEY=... python3 eval_sft/search_agent_eval.py --base-url ... \
    --gate live --backend tavily --n 300 --seed 0 --tag <ckpt>      # 게이트 2
```

| 게이트 | 재는 것 | 검색 백엔드 | 통과 기준 |
|---|---|---|---|
| **format** | teacher-forced 프리픽스에서 **다음 한 턴의 형식** | 불요 (오프라인) | `tool_call_parse_rate` ≥ **0.99** |
| **live** | system+user 만 주고 실제 검색 루프를 돌린 **최종 답 정확도** | tavily / replay | phase-2 `accuracy` > **phase-1 베이스라인** (같은 300문항·같은 시드) |

- **format 지표**: `tool_call_parse_rate`(생성이 유효한 `web-search` 호출 또는 `</think>` 이후
  최종 답 중 하나로 파싱되는 비율), `malformed_rate` + 사유 분포(`think_unclosed` /
  `unclosed_tool_call` / `unknown_tool` / `empty_query`), `think_closed_rate`,
  `next_action_agreement`(행동 종류가 레퍼런스 다음 턴과 일치), 평균·최대 생성 토큰,
  `truncated_rate`. 행마다 프리픽스 3건(system+user 1 + tool 턴 직후 무작위 2, `--seed` 고정)
  → 300행 = 900건. 결과는 `cut=start` / `cut=mid` 로 쪼개 출력한다.
- **live 지표**: `accuracy`(정규화 exact **또는** ground_truth 포함), `exact_match`,
  `no_final_answer_rate`, `mean_calls`, `call_cap_rate`(`--max-calls` 기본 20 도달),
  `malformed_rate`, `ctx_exhausted_rate`. 출력은
  `eval_sft/results/search_agent/<tag>/{format,live}_gate.{json,jsonl}`.
- **live 판정은 상대값이다.** 레퍼런스 궤적 자체가 `ground_truth` 와 어긋나는 행이 있어
  (row 0: 레퍼런스 "Sulfide minerals" vs GT "molybdenite mineral group") 절대 정확도의 상한이
  1 이 아니다. **phase-1 ckpt 를 같은 300문항·같은 시드로 먼저 재고 그 값을 베이스라인으로 박는다** —
  베이스라인 없이 나온 phase-2 단독 수치는 판정에 쓰지 않는다.
- **Tavily 쿼터.** 레퍼런스 궤적의 도구 호출은 평균 **10.6회/문항**(2~20, 300문항 합 3,171)이고
  모델이 더 부를 수 있으므로 **문항당 ~13회, 전량 1회 ≈ 4,000 콜**로 예산을 잡는다
  (`--max-calls 20` 상한이면 최악 6,000). 쿼터가 빠듯해도 `--n` 을 줄이지 않는다(부분 표본 무효
  원칙) — `--backend replay`(행이 실제로 받았던 결과를 토큰 Jaccard 로 되돌리는 오프라인 목업)로
  루프를 먼저 스모크하고 tavily 는 전량 1회만 돈다. 키는 환경변수 `TAVILY_API_KEY` 에서만 읽는다.
- **하네스는 vLLM 툴 파서를 쓰지 않는다.** tokenizer_v5 `apply_chat_template` 로 렌더한 원문
  프롬프트를 `/v1/completions` 로 보내고 XML `<function=…>` 을 직접 판다 — 게이트가 모델이 아니라
  파서를 재던 2026-08-30 사고(A4)의 재발 방지이고, INTERLEAVED_THINKING §7 규칙 4·5(프롬프트
  조립 단일화, 히스토리 `reasoning_content` 재전송)를 코드로 강제한 것이다. `skip_special_tokens:
  false` 필수(G2) — `</think>`·`<tool_call>` 이 단일 special id 라 기본 디코드에서 사라진다.
- tool 결과는 학습 데이터와 같은 JSON 모양(`query`/`follow_up_questions`/`answer`/`images`/
  `results[url,title,content,score,raw_content]`/`response_time`/`request_id`, `ensure_ascii=False`)
  으로 되돌린다. 모양이 어긋나면 `<tool_response>` 분포가 학습과 달라진다 (INTERLEAVED_THINKING §7 규칙 9 — 렌더 1건 육안 확인).
