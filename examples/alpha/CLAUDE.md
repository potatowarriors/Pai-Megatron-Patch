# Alpha Model — Claude Code Guide

GatedDeltaNet + Attention + MoE 하이브리드 ~15.08B 모델. Qwen3.5 dimensioning + DeepSeek-V3 MoE routing,
표준 RMSNorm, alpha 전용 v5 tokenizer, Muon(`dist_muon`) optimizer.

이 파일은 **지침만** 담는다. 현재 상태는 [`docs/STATUS.md`](docs/STATUS.md), 문서 색인은 [`docs/README.md`](docs/README.md),
사고 서사는 [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md), 학습 이력은 [`docs/TRAINING_HISTORY.md`](docs/TRAINING_HISTORY.md).
새 사실은 그 문서들에 쓰고 여기에는 한 줄만 (루트 CLAUDE.md "메모리·문서 규칙").

## Baseline 수치 (baseline_48L, 2026-05-20 마이그레이션 이후)

| 항목 | 값 |
|---|---|
| Total / Active / Per-rank(EP=8) params | **15.08B** / 1.79B / 3.26B |
| Layers (MG / HF) | 48 / 24 (2:1) |
| Hidden / Dense FFN | 2048 / 8192 |
| Q heads × head_dim / KV groups | 16 × 256 / 2 |
| GDN num_v_heads × head_v_dim | 32 × 128 |
| MoE routed × FFN, shared, topk | **192 × 512**, 1 × 512, 8 |
| DSV3 routing | sigmoid, 8 groups × top-4, scaling 2.5, `seq_aux_loss` + expert bias (coeff 1e-4) |
| Vocab (effective / padded) | 163,860 / **163,968** — `tokenizer_v5/` |
| EOD / chat-end | `<\|endoftext\|>` id 0 (pre-training) / `<\|im_end\|>` id 3 (SFT 전용) |
| Document boundary | stage1/2 4K dense: `--reset-attention-mask --eod-mask-loss`. **LC/THD(2026-08-22~)**: `--reset-position-ids --no-create-attention-mask-in-dataloader` + MBS 1 |
| Norm / QK-LayerNorm | 표준 RMSNorm(1p 제거) / 활성 + WD `apply_wd_to_qk_layernorm` |
| Optimizer | Muon `dist_muon` (+QK-Clip은 4K 단계만; LC preset에서 제거) |
| RoPE / context | θ=10M, partial 0.25 / 4096 학습, max-position 262K |

## Architecture

```
M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-   (48 layers)
```
`M` GatedDeltaNet(Megatron `MambaLayer` 슬롯) 18 · `*` Full attention(GatedSoftmax, QK-norm) 6 · `-` MoE MLP 24 · `D` Dense 0.

| 제약 | 값 | 이유 |
|---|---|---|
| TP | **1** | GatedDeltaNet은 TP>1 미지원 |
| EP | 8 (학습) | 192/8 = 24 experts/GPU. 변환·검증은 EP=#GPU (torch_dist resharding) |
| CP | LC 단계 4~8 | GDN CP 포팅 완료 (`docs/gdn_cp_port.md`). THD 필수 (dense mask는 O(seq²)) |
| Backend | Megatron-LM-251125 | Muon + DSV3 routing flags native |

## Quick Commands

```bash
cd examples/alpha
bash train.sh <model> <training> <data> [extra-megatron-args...]   # preset = configs/<group>/<name>.yaml
bash train.sh smoke smoke mock                                     # 2-iter smoke (wandb 자동 차단)
bash train.sh baseline_48L lc_a lc_a_32k_blend                     # LC-A 32K@CP4 THD
bash evaluate.sh outputs/<run> --gpus 4 [--benchmark --tasks standard] [--iter N] [--skip-convert]
python calculate_parameters.py --config configs/model/baseline_48L.yaml
python tools/alpha_config.py emit-megatron-flags --from-checkpoint <ckpt>   # GPU 불필요
```

- YAML 키 = Megatron CLI 플래그명. `yaml_to_flags`가 `true`→`--flag`, `false`→생략, 리스트→`a,b`로 emit. 뒤에 붙인 인자는 그대로 전달(YAML 덮어씀).
- preset 이름에 `smoke`이 있거나 data가 `mock`이면 `WANDB_MODE=disabled` 자동. 배너에 `wandb: DISABLED/online/off` 표시.
- `evaluate.sh`는 preflight → convert → verify(config.json↔common.pt) → weight diff → **forward_sanity(ppl 게이트)** → 벤치. 모든 변환·검증 args는 ckpt `common.pt`에서 유도.
  `--skip-convert`/`--skip-validate`는 **이미 통과한 산출물 재사용 전용** — 검증 실패를 우회하는 용도로 쓰지 않는다 (루트 CLAUDE.md "검증 규칙").
- 추론 서빙: SGLang 스택은 2026-08-18 제거. 벤치는 HF 기반 `scripts/run_benchmarks.sh`. **RL(post-training)은 이 리포 범위 밖** — alpha vLLM 플러그인·Megatron-Bridge·NeMo-RL 환경은 `project_s/NeMo-RL/`(`NEMO_RL_SETUP.md`)에 있다.
- 프로파일: `NSYS=1 bash train.sh analysis_24L profile mock --profile-ranks 0 --profile-step-start 6 --profile-step-end 7 --train-iters 9` → `tools/analyze_nsys_trace.py`. 레버·판정은 `docs/throughput_optimization.md` (per-step 레버(optimizer)는 prod로 이전 안 됨, per-token 레버만).

## Config Structure

Flat YAML. 현행 preset (★ = 현재 파이프라인; `arxive/`는 레거시·호환 X):

```
configs/model/     baseline_48L ★  analysis_24L(프로파일용 반깊이)  smoke
configs/training/  stage1, stage1_resume, stage2 ★(P2/P2b/P3 커리큘럼 헤더 참조), stage2_ab,
                   lc_a ★, lc_a_resume ★, lc_b ★, sft_64k ★, sft_128k ★, profile*, smoke
configs/data/      stage1_v5_blend, stage2_v5_blend_packed{,_p2,_p2b,_p3} ★,
                   lc_a_32k_blend ★, lc_filler_32k_pad16 ★, lc_b_128k_blend ★, lc_thd_check_32k, lc_a_smoke_blend,
                   sft_40b_blend ★, sft_128k_blend ★, sft_smoke_64k, mock
```

**Tokenizer** `tokenizer_v5/`: HF `PreTrainedTokenizerFast`, 5 파일. EOS/EOD id 0, PAD id 1, BOS 없음. chat template = Nemotron 3 Ultra 기반 + DSV4식 tool-시나리오 분기(2026-08-24). `tokenizer_config.json`·`special_tokens_map.json`·`training_config.yaml` **3파일 동기화 필수**.

**Resume·전환 규칙** (training preset YAML 안에 평면적으로 — 셸이 끼워넣지 않음):

| 상황 | 키 | consumed_samples |
|---|---|---|
| dataset 변경 (stage 전환) | `finetune: true` | 0 리셋 |
| 같은 dataset 연장 | `load: <path>` (+ weights-only ckpt면 `no-load-optim: true`) | 이어서 |

- **optimizer state는 저장한다** (`no-save-optim` 미설정, 2026-08-23 사용자 확정). 세이브당 ~90GB. weights-only ckpt 재개 시에만 `no-load-optim`.
- 스케줄러 상태 없는 ckpt를 재개하면 num_steps가 0으로 리셋 — 남은 구간 기준으로 warmup/decay 재정의 필요.
- DiLoCo 샤드 ckpt → 1노드: `DILOCO_UNSHARD_RESUME=1 DILOCO_WORLD=2` (카운터 ×world 보정).

## Critical Files

| File | Purpose |
|---|---|
| `train.sh` | 단일 런처: yaml_to_flags, 멀티노드 감지, env(cuDNN 9.24 LD_PRELOAD, `NCCL_MAX_NCHANNELS=16`), run 경로 유도, smoke/mock wandb 차단, `NSYS=1` |
| `pretrain_alpha.py` | 엔트리 (upstream `pretrain()` + alpha monkey-patch: `clip_qk` MambaLayer 가드 등) |
| `evaluate.sh` / `validate.sh` / `validate_mg_hf_full.py` | 통합 평가 · MG↔HF weight 검증 (args는 ckpt에서 유도; `router.expert_bias↔gate.e_score_correction_bias` 포함) |
| `forward_sanity.py` | 변환 HF 모델 perplexity 게이트 (evaluate.sh Stage 2.5) |
| `tools/alpha_config.py` | `load_config_from_checkpoint`, `emit-megatron-flags`, `generate-hf-config` |
| `tools/verify_pipeline.py` | `preflight` / `compare-config` / `tokenizer-roundtrip` |
| `tools/verify_chat_template.py` | chat template 34 tests — **마스킹 규약·injection 방어의 정본** |
| `tools/compute_blend_weights.py` | blend 가중치·epoch 표 |
| `hf_model/` | HF 구현. `AlphaSparseMoeBlock`은 DSV3 라우팅 미러. **`e_score_correction_bias`는 fp32 `nn.Parameter` + `_keep_in_fp32_modules_strict` — 되돌리지 말 것**. `AlphaRMSNorm`은 표준(`*γ`) |
| `../../toolkits/distributed_checkpoints_convertor/scripts/alpha/run_convert.sh` | GPU-agnostic 변환기 (EP=#GPU) |
| `scripts/setup_wandb.sh` | `WANDB_API_KEY` 해석 (env → `$WANDB_KEY_FILE` → `scripts/.wandb_key` → `~/.wandb_key`). **키 하드코딩 금지** |
| `diloco_patch.py` / `pretrain_alpha_diloco.py` / `launch_diloco.sh` | DiLoCo 2노드 코어·엔트리·런처 (env knob은 런처 헤더) |
| `scripts/lc_a_preflight.py`, `scripts/launch_lc_a_when_ready.sh` | LC 데이터 deep preflight, 자동 런처 |
| `sdg/ko_chat/` | 한국어 chat 합성 트랙 (README에 함정 5건) |

## Environment

```bash
bash scripts/validate_environment.sh
```
- 핀: **triton 3.3.0 / mamba-ssm 2.2.6.post3(git 빌드) / fla 0.4.1 / TE 2.9.0**. 최신판은 첫 step `Unsupported function referenced: get_int_dtype`.
- 2노드 H100(Backend.AI NGC 25.03)은 repo 부모의 **`setup_pai_megatron_env_multinode.sh`**. A100은 `_A100_v2.sh`(sm_80 TE wheel 별도).
- Claude Code 설정은 `CLAUDE_CONFIG_DIR`로 NFS에 영속 (`/home/work/vidsearch/setup-claude.sh`, 위 셋업 스크립트가 자동 source).

## Multi-Node — DiLoCo (2노드, IB 없음)

클러스터에 InfiniBand가 **없다**(영구). 노드 간 ~1GB/s. sync-DP는 bf16-reduce+GBS 3072에서도 1.15×. 해법은 DiLoCo
(outer Nesterov lr 0.7/μ 0.9, H=30). 전체 실측: `study/diloco_pilot.md`.

```bash
# 프로덕션 (완전 서로소 샤딩 필수)
DILOCO_DATA_SHARD=1 DILOCO_SHARD_BLOCK=<GBS> DILOCO_H=30 DILOCO_TAU=2 bash launch_diloco.sh <tag> baseline_48L <training> <data>
# resume: 각 노드가 자기 ckpt
NODE0_ARGS="--load <n0>" NODE1_ARGS="--load <n1>" DILOCO_DATA_SHARD=1 ... bash launch_diloco.sh ...
```

1. `DILOCO_DATA_SHARD=1` + `DILOCO_SHARD_BLOCK`(=GBS; 램프업 이력이 있는 런은 `gcd(consumed, GBS)`) 필수. seed-split은 A/B 전용. 전환은 `consumed % B == 0` 경계에서만(assert).
2. H를 바꾸면 outer lr/μ 재튜닝.
3. outer state(θ+momentum ~27GB/rank)는 `<save>/diloco_outer/`에 동봉, 1노드 인계 시 자동 무시. 저장 전 pending sync 자동 드레인.
4. τ는 wire p99 기준 (H=30이면 2~3). `NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 GLOO_SOCKET_IFNAME=eth0`는 런처가 설정.
5. `DILOCO_BIAS_SYNC=1`(기본) — expert_bias는 pair SUM으로 동기. pair collective는 양 노드 대칭 호출, env는 `EXTRA_ENV`로.
6. 2노드→1노드는 반드시 `DILOCO_UNSHARD_RESUME=1`.

## Muon Quick Reference

```yaml
optimizer: dist_muon
muon_momentum: 0.95
muon_num_ns_steps: 5
muon_scale_mode: spectral
```
TP=1 전용. **chunked optimizer-state offload 지원**: `--chunked-optimizer-state-offload --optimizer-state-offload-chunk-size-mb 256`
(128K@CP8 −21GB; 32K@CP4에는 불필요). QGKV 4-way split은 자동 — 시작 로그 `Muon QKV matcher: 3-way=0, 4-way=N` 확인.
기존 ckpt에 적용하면 optimizer 역학이 바뀌므로 stage 경계에서.

## 비-upstream 학습 기능 (Megatron-LM-251125)

전문·CLI·구현 위치·테스트 명령: [`../../docs/CUSTOM_TRAINING_FEATURES.md`](../../docs/CUSTOM_TRAINING_FEATURES.md).

| # | 기능 | CLI / 트리거 | 위치 |
|---|---|---|---|
| 1 | Step-wise GBS 스케줄 (토큰 임계마다 GBS 계단) | `--step-batch-size-schedule "0:768 250B:1536 …"` (`--rampup-batch-size`와 배타, `--train-samples` 전용) | submodule `num_microbatches_calculator.py`, `arguments.py`, `training.py` |
| 2 | Progressive auxiliary blend (aux 데이터셋 선형 램프) | `--progressive-blend-config blend.yaml` (`--data-path`류와 배타) | `megatron_patch/data/progressive_mix_dataset.py` |
| 3 | Muon QGKV 4-way split (Gated Attention `linear_qgkv`) | 자동 (`dist_muon`, `--muon-no-split-qkv` 아님). 시작 로그 `Muon QKV matcher` 확인 | submodule `optimizer/muon.py` |
| 4 | Muon chunked optimizer-state CPU offload (PR #6244 백포트) | `--chunked-optimizer-state-offload --optimizer-state-offload-chunk-size-mb 256` (torch_dist·동기 save 필수) | submodule `optimizer/cpu_offloading/`, `layer_wise_optimizer.py` |
| 5 | THD+CP 잠복버그 3건 (utils NameError·None 가드, mamba_model THD rope 미배선) | 자동 | submodule `core/utils.py`, `models/mamba/mamba_model.py` |
| + | DiLoCo 2노드 (IB 없는 클러스터) | `examples/alpha/launch_diloco.sh` | `examples/alpha/diloco_patch.py` (submodule 아님) |

테스트 전체 목록은 위 문서 § Tests. 대표: `tests/test_progressive_mix_dataset.py`, `tests/test_diloco_shard_view.py`,
submodule `tests/unit_tests/test_step_batch_size_schedule.py`, `test_muon_optimizer.py`, `test_chunked_offload_s*.py`, `tests/test_gdn_*.py`.

## 함정 표 — Known Issues 한 줄 요약

서사·재현·회귀 테스트는 [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) (같은 제목). 새 사고는 거기에 쓰고 여기엔 한 줄.

| 날짜 | 증상 | 원인 → 대응 |
|---|---|---|
| 09-01 | opencode_v1 tool 결과가 `<tool_response>` 안에 Python repr(`[{'type':'tool-result',…}]`, 줄바꿈 리터럴 `\n`)로 렌더 — 블렌드 4.3%, reasoning 0% | tool content 가 list 인데 템플릿이 str() → phase-2 에서 `normalize_row` 평문화 + 규칙 9(렌더 육안). `KNOWN_ISSUES` 09-01 ① |
| 09-01 | identity_v1 원본 기준 ≈180회 반복 (0.43% 비중 × 1.2M tok 셋) | 결정 #9 는 비중 상한만 규정. 집계는 bin 1표(`calculate_per_token_loss=False`)라 gradient 0.43% — 증폭 없음. phase-2: 카드 v1.2(제작자 개인 우선) 슬라이스 재생성 + identity_v2 0.6% 연속학습 + 프로브 2종. 〃 ② |
| 09-01 | chat_v3_chat 75,287행(11.8%) null_content 드롭 — WildChat 26.6%·lmsys 12.2% 미복원 | 공개판 해시 미매칭(toxic 제외판 추정) → gated 원천 접근 시도, 불가 시 88.2% 정본 기록. 〃 ③ |
| 08-30 | 에이전틱 SWE/Terminal 0점 — 모델 아님 | ① litellm 미등록 모델 비용계산 RuntimeError ② **tool-call 파서 불일치**(hermes=JSON vs 우리 모델=XML `<function=…>`) ③ preds.json↔.jsonl 경로. → `TOOL_PARSER=qwen3_xml`, 레지스트리 등록, 게이트 A4(파서가 실제로 파싱하는지) |
| 08-30 | 벤치 태스크가 base 모델용 (GPQA strict 원리적 0점, MMLU-Pro 5-shot, avg@16이 실질 avg@1) | 내장 lm_eval 태스크를 채팅·추론 모델에 그대로 사용. → `tasks/*_aa.yaml` 재작성(0-shot·8단 폴백·take_first_k·사고 분리) |
| 08-30 | RULER 6~25% (자체 NIAH는 200/200) | 추론 켠 채 128토큰 예산 → 서두에서 소진. → Reasoning-Off(`enable_thinking:false`), 실측 512토큰 소진→21토큰 stop |
| 08-30 | SFT ckpt 벤치 전 항목 무효 (MMLU 추출실패 34%, AIME 0/30, SWE 0/20) | HF 변환기가 `generation_config.json` 미생성 → eos가 `<\|endoftext\|>`(0)뿐이라 `<\|im_end\|>`(3)에서 안 멈춤 + `</think>` special=True라 출력에서 삭제. **변환 후 eos 정합·서빙 1건 스모크(finish=stop) 게이트 통과 전 수치 기록 금지** |
| 08-23 | THD+CP `cu_seqlens must be divisible by 2*cp_size` 정지 (LC-A iter170) | 합성 원문의 리터럴 `<\|endoftext\|>`가 문서 중간 id 0 → 문서 분열. 런타임 `snap_cu_seqlens_to_grid`(CP>1 전용) + 투입 전 `scan_internal_eod.py` |
| 08-22 | THD+CP≥2 첫 스텝 MoE `Split sizes doesn't match` | mamba_model rope에 packed_seq_params 미전달 → q/k NaN → CUDA topk 중복 인덱스. gpt_model 미러. **MoE 라우팅 크래시는 hidden NaN부터**; mock 데이터로 THD+CP 검증 불가 |
| 08-17 | DiLoCo 두 노드 loss 거울상 시소 (주기 ~336 iter) | 짝/홀 샤딩 × blend 가중치 합 ±1e-6 잔차 세차운동 → `DILOCO_SHARD_BLOCK` 블록-순환 |
| 08-11 | DiLoCo expert_bias 노드 간 발산 (0.118) | buffer라 outer sync 제외 → `DILOCO_BIAS_SYNC`. 함정: `import a.b as _F`가 함수 반환 → `importlib.import_module` |
| 07-22 | A100 evaluate.sh 3중 이슈 | modelopt 몽키패치 · typing_extensions · `--multi_gpu` **무음 exit 0** → A100_v2 Step 13.5 내장. sm_90 TE wheel 재사용 금지 |
| 07-15 | optimizer-state resume `Failed to CUDA calloc async` | NCCL 64채널 comm 버퍼 OOM → `NCCL_MAX_NCHANNELS=16`. DiLoCo 무관 |
| 07-14 | DiLoCo 저장 시 `unhandled cuda error` | pending sync 중 save → 드레인 후 저장 (수정됨) |
| 07-13 | 첫 step `cuDNN Error: No valid engine configs` (QK-Clip만) | NGC 25.03 cuDNN 9.8에 max_logit 엔진 없음 → `nvidia-cudnn-cu12==9.24 --no-deps` + LD_PRELOAD |
| 07-13 | 2노드 셋업 연쇄 실패 | 이미지 전역 `PIP_CONSTRAINT` · TE 서브모듈 순서 · 미고정 핀 → `_multinode.sh` |
| 06-15 | MG↔HF 검증 `14180/14181`, expert_bias 1개 실패 | 검증 로드가 fp32 bias를 bf16 평탄화 → HF bias = fp32 Parameter + `_keep_in_fp32_modules_strict` (buffer는 미보호) |
| 05-26 | 벤치 전부 random (ARC-easy 25%), 가중치 검증은 통과 | HF `AlphaRMSNorm`이 1p(`1+γ`), 학습은 표준 → 수정 + forward_sanity ppl 게이트 |
| 05-26 | stage1 재개 후 throughput 급락 | `moe-shared-expert-overlap`이 `CUDA_DEVICE_MAX_CONNECTIONS=1`에서 직렬화 → 되돌림 |
| 05-26 | 평가 파이프라인 config drift 3건 (184/192, HF 라우팅 softmax, coverage 갭) | 모델 config 3곳 중복 → 모든 args를 ckpt `common.pt`에서 유도 |
| 05-20 | 1p 제거 / WD `apply_wd_to_qk_layernorm` / tokenizer v5 in-repo / smoke wandb 차단 | Qwen3.5·Qwen3-Next 정렬. v5 vocab은 옛 .bin과 불일치 → 전량 재토크나이즈 |
| 05-12 | EOS가 `<\|im_end\|>`(3) | pre-training EOD `<\|endoftext\|>`(0)로 분리. tokenizer 3파일 동기화 |
| 05-12 | 기토크나이즈 데이터 EOD=id 3 | `remap_eod.py --old-eod 3 --new-eod 0` (DCLM 1.78TB 2h55m) |
| 05-12 | HF config 생성기·`configuration_alpha.py`에 Qwen3 ID·stale default | 배포 시 silent corruption → v5 값 + 회귀 테스트 |
| 05-12 | document boundary flags 활성화 | 데이터에 EOD id 0 스트림이 전제. ON/OFF differential(Phase C-loader)로 검증 |
| Stage 2-2 | Muon Nesterov 미적용 | 구 셸이 store_true 미전달 → `yaml_to_flags`가 구조적 해결 |
| Stage 1→2 | QK LayerNorm γ 폭발 (12.9) | 1D param WD 미적용 → `apply_wd_to_qk_layernorm` + megatron_patch 전달 버그 수정 |
| Stage 2 | `MambaLayer has no attribute self_attention` | upstream `clip_qk` 가정 → `pretrain_alpha.py` hasattr 가드 |
| Stage 2 | QK-Clip 로그 없음 / logit 안 내려감 | `--log-max-attention-logit` / QK-Norm이 스케일 상쇄 → `_clip_layernorm_gamma`로 γ도 스케일 |

## 환경 불변량·운영 함정

- **wandb 409 "filestream at capacity" 1건**으로 run이 죽은 것처럼 보여도 학습은 무영향. 판별은 노드 로그의 iteration 타임스탬프 + `[diloco]` sync 라인. `wandb sync <run_dir>`로 백필.
- **sub1 시계가 main1보다 ~5.4분 늦고 TZ 라벨도 다름** — 두 노드 로그 시각 직접 비교 금지. Megatron iteration 로그는 **마지막 rank**(2노드면 sub1)에 찍힘.
- **alpha 풀모델은 실행 간 비결정** (TE fused attention bwd, iter 3부터 상대 ~2.7e-3). `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` 무효. A/B는 같은 구성 재실행의 산포 포락선으로 판정, 비트 검증은 결정론 유닛 골든 (`study/nondeterminism_probe.md`).
- **CP>1의 PyTorch UserWarning `c10d::allreduce_: an autograd kernel was not registered`는 무해** (첫 backward 랭크당 1회). helper.py loss CP 집계가 fallback identity-backward에 의존 — 수학적으로 정확(∂Σ/∂local=1), upstream pretrain_gpt 동일 패턴. 증거: CP{1,2,4} grad 등가 1.2e-4 + LC-A 완주. CP1에선 안 나옴.
- 감시 스크립트의 `pgrep -f`는 자기 명령줄과 자기매치 → `[p]attern`. ssh 원격 `A && nohup X &`는 체인 전체가 백그라운드로 가 nohup 미도달.
- 연속 torchrun은 잔류 프로세스로 EADDRINUSE — 런 사이 GPU idle 대기. 생산 병행 중 `.bin`만 있고 `.idx` 없는 반완성 파일 경합 — 존재 확인이 아니라 로드 검증.
- NFS mtime이 UTC보다 9h 늦게 찍힘 (정체 오판 주의). 콜드 캐시 블렌드 첫 로드 80분+ > NCCL 기본 60분 → 타임아웃 180분.
- `IndexedDataset.get()`은 memmap 뷰를 반환 — 보관하려면 복사. sub1의 NGC `modelopt`는 PYTHONPATH shim으로 무력화(영구 제거는 sudo).
- `/home/work/Datasets/LL_preprocessed/mmap/`(8.4T)는 다른 사용자 소유 — **삭제 금지**.
