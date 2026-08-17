# A100 단일-GPU 평가 환경 구축 + DiLoCo stage2 첫 벤치마크 (2026-07-22 ~ 07-23)

Gemma4 추론 서빙이 돌고 있는 A100(Backend.AI) 박스에 alpha 평가 환경을 추가 구축하고,
**diloco_stage2 node0 iter_0007120**의 MG→HF 변환 → 검증 → standard suite 벤치마크를
완주한 기록. H100 멀티노드 외의 환경(Ampere, 단일 GPU, 서빙 공존)에서 평가 파이프라인을
재현할 때의 canonical 레퍼런스.

## 1. 환경

| 항목 | 값 |
|---|---|
| 플랫폼 | Backend.AI (GPU 이름 마스킹, fractional GPU) |
| 컨테이너 | NGC pytorch:25.03-py3 (torch 2.7.0a0+nv25.03, Python 3.12, cuDNN 9.8) |
| GPU | 5개 전부 compute capability **8.0 (A100/Ampere)** — 16/16/8/40/80GB 슬라이스 |
| 동거 워크로드 | Gemma4-12B vLLM 서빙 — `/tmp/gemma-venv`(system-site-packages=false, 완전 격리), GPU 4 점유 |
| CPU/RAM | 8 cores / 1.5TB |

서빙 공존 판정: 서빙 venv가 완전 격리라 시스템 Python 설치와 패키지 충돌 없음.
접점은 ① GPU(→ `CUDA_VISIBLE_DEVICES`로 분리), ② 셸 rc 누수(→ 아래 v2 스크립트가
기본 차단), ③ 빌드 CPU 경합(→ `MAX_JOBS` 제한)뿐.

## 2. 환경 설치 — `setup_pai_megatron_env_A100_v2.sh`

repo 부모 디렉토리의 `setup_pai_megatron_env_A100.sh`(2026-04-07)는 07-13 의존성
픽스(CLAUDE.md Known Issues "2노드 환경 셋업" 참조) 이전 버전이라 NGC 25.03에서 연쇄
실패한다. **`setup_pai_megatron_env_A100_v2.sh`**가 multinode 스크립트의 픽스를
A100에 이식한 버전(원본은 참조용 무수정 보존):

- TE 서브모듈 순서(checkout 후 init), 선별적 `env -u PIP_CONSTRAINT`,
  mamba v2.2.6.post3 git 빌드 + fla==0.4.1 핀, cuDNN 9.24(`--no-deps`) — multinode와 동일
- **`NVTE_CUDA_ARCHS=80`** (multinode의 90을 복사하면 안 됨)
- **TE wheel 캐시가 아치별 분리**: `$WORKSPACE/te_wheels_sm80/`.
  ⚠ workspace 루트의 `transformer_engine-2.9.0+70f53666-*.whl`은 **sm_90 전용**
  (cuobjdump로 52개 ELF 전부 sm_90 확인) — A100에서 재사용하면 "no kernel image" 런타임 크래시.
- 서빙 공존 하드닝: `.bashrc`/`.zshrc` 자동 수정 기본 OFF(`--add-bashrc`로 활성화,
  `CUDA_DEVICE_MAX_CONNECTIONS=1`·PYTHONPATH가 서빙 재시작 셸에 누수되는 것 방지),
  `MAX_JOBS` 기본 6, `TORCH_CUDA_ARCH_LIST=8.0`
- FA2: NGC 번들(25.03: 2.7.3)이 import되면 그대로 사용(무핀 pip 재설치는 최신 2.8.x
  소스 리빌드 유발 — NGC ABI 보호 규칙과 동일 논리)
- 클론 URL: alibaba → potatowarriors fork

```bash
cd <repo 부모 디렉토리>
./setup_pai_megatron_env_A100_v2.sh --workspace <repo 부모 디렉토리>
# 셸마다: source ~/.pai_megatron_env
```

실측: 전체 약 1h40m (TE sm80 빌드 ~25분, 8코어 MAX_JOBS=6). wheel 캐시 히트 시 대폭 단축.

## 3. 벤치 경로에서 발견한 3중 이슈 (전부 해결 ✅)

설치 검증은 전부 통과했지만 evaluate.sh 실행에서 순차적으로 3개가 터졌다. 셋 다
H100 멀티노드에서는 안 밟았던 경로다.

> **2026-07-27 업데이트**: 3-1·3-2는 `setup_pai_megatron_env_A100_v2.sh` **Step 13.5**로
> 내장되어 더 이상 수동 조치가 필요 없다(아래는 원인 기록). 3-3은 repo 파일 패치로 유지.
> 재부팅/세션 재생성 시 복원 절차는 repo 부모의 `RESTORE_AFTER_REBOOT.md` 참조.

### 3-1. nvidia-modelopt의 transformers 몽키패치 충돌

- **증상**: Stage 2 `validate_mg_hf_full.py`의 `from_pretrained`에서
  `TypeError: _new__load_pretrained_model() missing 1 required positional argument`.
  변환(Stage 1)은 통과 — HF 로드를 안 하기 때문.
- **원인**: NGC 번들 `nvidia-modelopt` 0.25.0은 import되는 순간 transformers 내부
  (`_load_pretrained_model`)를 구버전 시그니처 기준으로 몽키패치. 프로젝트 핀
  transformers 4.57.0.dev0과 불일치. 유입 경로는 Megatron `training/checkpointing.py:59`의
  guarded `from modelopt.torch.opt.plugins import ...` — Megatron을 import하는 모든
  프로세스에서 패치가 발동된다.
- **수정**: modelopt 제거 (alpha 스택 미사용 — 양자화/TRT 전용, Megatron 쪽은 guard가 폴백):
  ```bash
  sudo -E env -u PIP_CONSTRAINT /usr/local/bin/pip uninstall -y --break-system-packages \
      nvidia-modelopt nvidia-modelopt-core
  ```
  함정: 일반 `sudo pip`은 Debian pip(`/usr/bin/pip`)으로 해석되고 PEP 668
  (externally-managed-environment)에 막힘 — **NGC pip 경로와 `--break-system-packages` 명시 필수**.

### 3-2. lm-eval 0.4.12 ↔ 구버전 typing_extensions

- **증상**: lm-eval 시작 즉시 `TypeError: _TypedDictMeta.__new__() got an unexpected
  keyword argument 'extra_items'` (`lm_eval/result_schema.py`).
- **원인**: lm-eval 0.4.12가 `TypedDict(..., extra_items=)`(PEP 728, typing_extensions
  ≥4.13) 사용. NGC 이미지 번들이 구버전.
- **수정**: `pip install -U "typing_extensions>=4.13"` 후 `import lm_eval.evaluator`로 확인.

### 3-3. run_benchmarks.sh의 `--multi_gpu` 하드코딩 (무음 실패)

- **증상**: `NUM_GPUS=1`에서 4개 태스크 그룹 전부
  `ValueError: You need to use at least 2 processes to use --multi_gpu`로 즉사하는데
  스크립트는 **exit 0으로 "완료"** — eval_results/ 산출물이 없어야만 알 수 있다.
- **수정**: `scripts/run_benchmarks.sh`의 run_eval이 `NUM_GPUS>1`일 때만 `--multi_gpu`를
  붙이도록 패치(반영 완료). **주의**: run_eval 실패는 여전히 전체 exit code에 안 잡히므로
  성공 판정은 반드시 `eval_results/results_*.json` 존재로 할 것.

## 4. DiLoCo per-node 체크포인트 평가 규칙

`DILOCO_UNSHARD_RESUME`(×world 카운터 보정)은 **학습 resume 전용** — 벤치마크에는 불필요.
근거: ① 보정 대상(consumed/LR 스케줄러/iteration)은 변환·평가가 읽지 않음, ② 저장 시
pending sync는 자동 드레인되어 weights가 일관 상태(diloco_pilot.md 규칙 3), ③ outer
state(`diloco_outer/`)는 순수 Megatron 로드에서 자동 무시. 즉 per-node 체크포인트는
그냥 `evaluate.sh <node_dir> --gpus 1 --benchmark`로 평가하면 된다.

## 5. 실행 방법 (이 박스 기준)

```bash
source ~/.pai_megatron_env
cd examples/alpha
# GPU 4 = Gemma 서빙 — 제외. GPU 3(40GB)에서 EP=1 변환 + 벤치.
CUDA_VISIBLE_DEVICES=3 bash evaluate.sh \
    outputs/diloco_stage2/node0 --gpus 1 --benchmark
# 변환 재사용 시: --skip-convert
# 벤치만 다시: CUDA_VISIBLE_DEVICES=3 NUM_GPUS=1 bash scripts/run_benchmarks.sh \
#     outputs/diloco_stage2/node0/hfmodel_0007120 --tasks standard
```

15.08B 모델이 bf16 ~30GB로 40GB 슬라이스에 적재됨(EP=8→EP=1 torch_dist 리샤딩,
153GB ckpt 중 model 파트만 읽음). batch auto가 메모리를 추가로 크게 잡으므로
(실측 ~78GB — Backend.AI 슬라이스가 표기보다 탄력적) 여유 없는 슬라이스에서는
`--batch-size`를 명시할 것.

## 6. 결과 — node0 iter_0007120 (stage2 DiLoCo, 첫 벤치)

게이트: weight 검증 **14,181/14,181** 일치 · forward sanity(ppl gate) **PASS**.

| 태스크 | shot | 점수 |
|---|---|---|
| MMLU | 5 | 55.87 |
| KMMLU | 5 | 34.27 |
| HellaSwag | 5 | 58.93 / **78.35** (acc/acc_norm) |
| ARC-easy | 5 | 83.04 / 84.26 |
| ARC-challenge | 5 | 55.12 / 58.53 |
| Winogrande | 5 | 71.19 |
| BoolQ | 5 | 81.16 |
| PIQA | 5 | 80.58 / 82.37 |
| GSM8K | 4 | 31.31 (strict) / 31.46 (flexible) |
| HumanEval | 0 | 33.54 (pass@1) |
| MBPP | 3 | 34.20 (pass@1) |

- 산출물: `outputs/diloco_stage2/node0/hfmodel_0007120/eval_results/` (results JSON 4개)
- WandB: `alpha-evals/node0_iter0007120`
- 소요(단일 A100 40GB): 5-shot 그룹 ~6h(26.3만 요청, 7→23 it/s로 워밍업) +
  GSM8K/HumanEval/MBPP 합 ~7.5h ≈ **총 13.5h**. 변환+검증은 ~30분.

## 6b. 추적 — node0 stage2 진행 추이 (iter 7120 → 10000 → 12000)

동일 파이프라인 반복 실행(매회 게이트 통과: 14,181/14,181 · forward sanity PASS).
평가일: 7120=07-22, 10000=07-26, 12000=07-29.

| 태스크 (metric) | 7120 | 10000 | 12000 | 누적 Δ |
|---|---|---|---|---|
| MMLU (acc) | 55.87 | 56.77 | 57.06 | +1.19 |
| KMMLU (acc) | 34.27 | 36.61 | 36.04 | +1.77 |
| HellaSwag (acc_norm) | 78.35 | 78.50 | 78.91 | +0.56 |
| ARC-easy (acc_norm) | 84.26 | 84.64 | 84.93 | +0.67 |
| ARC-challenge (acc_norm) | 58.53 | 58.45 | 58.87 | +0.34 |
| Winogrande (acc) | 71.19 | 71.51 | 70.96 | −0.23 |
| BoolQ (acc) | 81.16 | 80.49 | 81.38 | +0.22 |
| PIQA (acc_norm) | 82.37 | 82.48 | 81.94 | −0.43 |
| GSM8K (strict) | 31.31 | 34.95 | 36.47 | **+5.16** |
| HumanEval (pass@1) | 33.54 | 35.98 | 39.63 | **+6.09** |
| MBPP (pass@1) | 34.20 | 37.40 | 40.20 | **+6.00** |

패턴 (3점 기준):
- **코드가 최대 상승 축** — HumanEval/MBPP 누적 +6, 구간별로도 가속(12000 구간 +3.7/+2.8).
- **수학 3점 연속 상승** — GSM8K 누적 +5.2.
- 지식(MMLU) 완만한 연속 상승(+1.2). **KMMLU는 10000에서 점프 후 ~36에서 보합**.
- 상식 계열(HellaSwag/ARC/WG/BoolQ/PIQA)은 ±0.7 노이즈 밴드 내 — stage2 믹스의
  목표(수학·코드·한국어)와 일치하는 선택적 개선. 10000의 BoolQ −0.67은 12000에서
  회복(81.38) — 노이즈였음이 확인됨.

산출물: `node0/hfmodel_{0007120,0010000,0012000}/eval_results/`,
wandb `alpha-evals/node0_iter{0007120,0010000,0012000}`.

## 7. 후속 후보

- **node1 대칭 평가** — DiLoCo 페어의 노드 간 편차 측정 (명령 동일, node1로 교체)
- stage2 진행에 따라 새 iter 재평가 (환경·HF 캐시 준비 완료 — 순수 벤치 시간만 소요)
