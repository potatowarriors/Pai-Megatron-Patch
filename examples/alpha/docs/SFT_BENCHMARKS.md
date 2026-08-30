# SFT 벤치마크 스위트 — 설계·판정·인프라 (2026-08-29)

SFT 본 런(`sft_128k_full`, 2,448 iters, save 300)의 체크포인트별 평가 체계.
기존 base-스타일 표준 11종([EVALUATION.md](EVALUATION.md))과 별개로, **chat/thinking 모드**의
instruct 능력을 측정한다. 참조 좌표는 DSV4 post-training 표 + Nemotron 3 Ultra Table 10
(SFT 데이터가 Ultra 레시피이므로 후자가 1차 준거). 진행 상태는 [STATUS.md](STATUS.md).

> **2026-08-30 — T1 을 프론티어 규약으로 재작성했다.** 이전 수치는 전량 무효 판정·삭제
> (원인·증거: [KNOWN_ISSUES.md](KNOWN_ISSUES.md) 2026-08-30). 새 T1 은 §3.6, 게이트는 §7.
> 게이트 G1~G3 를 통과하기 전에는 어떤 수치도 `results/TRACKING.md` 에 기록하지 않는다.

## 0. 결정 로그 (사용자, 2026-08-29)

| 결정 | 내용 |
|---|---|
| SWE-bench·Terminal-Bench | **필수 요구**. Backend.AI 노드는 docker 불가(§4) → **외부 docker 호스트 gpu06 DinD 컨테이너로 해결·검증 완료**(2026-08-29, [EVAL_DOCKER_NODE.md](EVAL_DOCKER_NODE.md)). 하니스 구축 대기 |
| judge | **gemini-3.7-flash** 확정(사용자, 2026-08-29). 키 검증 완료(`examples/alpha/.env`). 러너는 provider-agnostic(Gemini/OpenAI 호환) |
| 실행 노드 | sub1 (유휴 8×H100). main1은 SFT 학습 전용 |

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
| Terminal Bench 2.0 | ⚠️ 대기 | docker 경로 확정 후 (프록시 없음) |
| IMOAnswerBench / HLE | 후순위 | 15B-A1.8B에 변별력 낮음. HLE는 gated(HF 토큰 필요) |
| Chinese-SimpleQA | 🔁 대체 | 중국어는 학습 언어 아님 → KMMLU 유지 + KoChat 판정(T3) |
| Codeforces / Apex / CorpusQA-1M / BrowseComp / HLE-tools / MCPAtlas / Toolathlon / GDPval-AA / SWE Pro·Multilingual | ❌ | 실시간 저지·비공개 scaffolding·1M 창·웹서치 스택·MCP 팜·유료 서비스·docker |

**보강 (SFT 데이터 정합)**: IFEval(IF-Chat fan-out), BFCL v4(tool-call 템플릿 분기, 로컬 AST 채점),
RULER 64/128K(STATUS의 "RULER 풀 하니스 미구축" 해소), 표준 11종(망각 게이트, T4).

## 2. 티어 구성·주기

ckpt 주기 = 300 iters ≈ 27h. 스위트는 그 안에 완주해야 추이가 그려진다.

| 티어 | 벤치 | 주기 | 예상 소요(sub1) |
|---|---|---|---|
| T1 코어 | MMLU-Pro, GPQA-D, AIME25+HMMT25Feb, LiveCodeBench, IFEval | 매 ckpt | ~4–6h |
| T2 롱 | RULER 64/128K(태스크당 100샘플), MRCR 128/256K, NIAH 연속성 | 격 ckpt | ~2h |
| T3 판정 | SimpleQA-Verified, KoChat 품질, (최종만) Arena-Hard류 | 격 ckpt | ~2h |
| T4 회귀 | 기존 표준 11종 (LC-B 63.72 대비 망각 감시) | ckpt 2–3개마다 | ~3h |
| SWE 프록시 | oracle-patch 생성 → sb-cli 클라우드 채점 | 격 ckpt | 생성 ~1h |

생성 규약: thinking 기본(reasoning parser로 `<think>` 분리), 최종 평가에서
medium-effort 렌더(`{reasoning effort: efficient}`) 2차 패스로 토큰 효율 곡선 측정.
**서빙·평가는 tokenizer_v5 apply_chat_template만 사용** (INTERLEAVED_THINKING.md §7-4).

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

## 3.6 반복 실행 워크플로 (학습 중 체크포인트마다)

학습이 진행되며 체크포인트(300 iters마다)가 나오면 반복 평가한다. 스크립트는 모두
`examples/alpha/eval_sft/`. 멱등적(이미 한 건 skip)이고 GPU 정리를 내장한다.

| 스크립트 | 역할 |
|---|---|
| `eval_ckpt.sh <RUN_DIR> [ITER\|latest] [tiers]` | 한 체크포인트 전체: 변환(MG→HF)→fleet 기동→티어 실행→**깨끗한 종료**→집계 |
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

**반복성 불변식**:
- lm_eval 0.4.12 고정, 태스크·few-shot·seed·gen 파라미터 러너에 하드코딩(`run_tier1.sh`).
- 서버 `max-model-len 49152`·`max_gen_toks 24576`(prompt 여유 확보, 400 방지). thinking 기본.
- **GPU 정리 규율**: 교체 시 `stop_fleet.sh`로 SIGTERM→회수확인. hard-kill 반복이 GPU 좀비
  누수를 만든다(2026-08-29 GPU0 사고). fleet 기본 GPU = 1~7 (GPU0 회수 전까지 제외).
- 변환은 `evaluate.sh`(forward_sanity ppl 게이트 포함) 재사용 — 잘못 변환된 ckpt는 게이트가 막음.

## 3.8 베이스라인 — 무효, 재측정 대기 (2026-08-30)

**2026-08-29 에 기록한 LC-B iter320 베이스라인 표는 전량 무효 판정·삭제됐다.** 체크포인트
종료 토큰 결함 + base 모델용 태스크로 측정한 값이었다(경위: [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
2026-08-30 두 항목). 새 T1(§3.6)·T2(§3.9)로 다시 재야 한다.

살아남은 유일한 롱컨텍스트 수치는 LC-B 자체 NIAH 하니스 결과다 — 이번 스위트와 무관한
별도 측정이라 영향이 없다: 4k~131k **200/200**, 256K **95%**, 384K 0%
(정본 `study/lc_b_final_eval.md`).

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

- [x] sub1 유휴·환경 실사, docker 가능성 판정 (2026-08-29)
- [ ] Phase 0: vLLM 0.25.1+플러그인 설치 → LC-B iter320 서빙 스모크 + 패리티 + 템플릿/파서 검증
- [ ] Phase 1: lm_eval API 하니스 + T1 태스크 구성 (HMMT 커스텀 yaml 포함) → LC-B 베이스라인
- [ ] Phase 2: LCB·MRCR·RULER 러너 + T2 베이스라인
- [ ] Phase 3: judge 러너(T3, 키 수령 후) + SWE 프록시 레인
- [ ] Phase 4: 오케스트레이터(ckpt 감시→변환→스위트→wandb) 상시화
- [ ] docker 경로 확정 시: SWE-bench Verified·Terminal-Bench 2.0 온보딩

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

부수 불변량:
- `--max-model-len` ≥ `max_gen_toks` + 프롬프트 최대치. 32768 모델길이에 32768 생성예산은 성립 불가.
- 장시간 러너·프로브는 `setsid` + NFS 로그로 분리 실행 (세션 종료에 죽지 않도록).
- 한 번에 한 변수만 바꾼다. 길이·온도·파서·모델 디렉토리를 동시에 바꾸면 원인 분리가 불가능하다.
- 부분 표본 결과는 기록하지 않는다 (NVIDIA 재현 문서: "Never report sub-sampled / limited runs").
