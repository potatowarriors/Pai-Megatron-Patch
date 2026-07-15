# Alpha Model - Claude Code Guide

Alpha는 GatedDeltaNet + Attention + MoE 하이브리드 아키텍처 기반의 실험적 LLM 학습 프로젝트입니다. Baseline은 Qwen3.5-35B-A3B의 dimensioning recipe와 DeepSeek-V3-style MoE routing을 차용했고, **표준 RMSNorm + alpha 전용 v5 tokenizer + Muon optimizer + QK-Clip**을 결합한 ~15.08B 모델입니다.

## Migration Summary (Qwen3.5 dimensioning + DSV3 MoE)

이 표는 현재 baseline의 핵심 numbers를 한눈에 보여줍니다. 자세한 history와 의사결정 근거는 아래 "Training Plan" 및 "Known Issues & Fixes" 섹션 참조.

| 항목 | 값 |
|------|---|
| Total params (실측) | **15.08B** |
| Active params (per token) | 1.79B |
| Per-rank params (EP=8) | 3.26B |
| HF layers / Megatron layers | 24 / 48 |
| Hidden / Dense FFN | 2048 / 8192 |
| Q heads × head_dim | 16 × 256 (Q upcast → 4096) |
| KV groups × head_dim | 2 × 256 |
| GatedDeltaNet num_v_heads × head_v_dim | 32 × 128 |
| MoE routed experts × FFN | **192 × 512** (184→192: 8×24, divisibility) |
| MoE shared expert × FFN | 1 × 512 |
| MoE topk | 8 |
| MoE group routing | 8 groups × top-4 (4×24 = 96 candidates) |
| MoE topk scaling factor | 2.5 |
| MoE score function | sigmoid (모든 stage 통일) |
| MoE balancing | **Stage 1+: `seq_aux_loss` + expert bias (1e-4 coeff)** — DSV3-strict (aux-loss-free bias + complementary seq_aux safety net). *(live run = seq_aux_loss; ground truth는 checkpoint `common.pt`)* |
| Vocab (effective / padded) | 163,860 / **163,968** |
| Tokenizer | `examples/alpha/tokenizer_v5/` (alpha 전용 BBPE) |
| EOS / pre-training EOD | `<\|endoftext\|>` (id 0) — `<\|im_end\|>` (id 3)는 SFT 단계 chat turn 전용 |
| Document boundary handling | `--reset-attention-mask` + `--eod-mask-loss` ON; `--reset-position-ids` **OFF** (alpha hybrid: Mamba가 packed_seq_params 미지원, attention은 RoPE 상대성으로 무영향 — stage1.yaml 주석 참조) |
| RMSNorm | 표준 (γ=1 init, **1p 제거**) |
| QK-LayerNorm | 활성 + WD `apply_wd_to_qk_layernorm` |
| Optimizer | Muon (`dist_muon`) + QK-Clip (γ scaling 포함) |
| RoPE | θ=10M, partial 0.25 |
| Training context | 4096 (max-position 262K) |

## Architecture Overview

### Hybrid Pattern (`M-M-M-*-` × 6 = 48 Megatron layers)
```
M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-M-M-M-*-
```
구성: 18 GatedDeltaNet (M) + 6 Full Attention (*) + 24 MoE MLP (-) + 0 Dense MLP (D, 현재 미사용).

| Symbol | Type | Description |
|--------|------|-------------|
| `M` | GatedDeltaNet | Linear attention with gated delta rule (Megatron `MambaLayer` 슬롯에 호스팅; O(n) 복잡도) |
| `*` | Multi-Head Attention | Full attention layers (12.5% of total, GatedSoftmaxAttention with QK-norm + QK-Clip) |
| `-` | MoE MLP | Mixture-of-Experts FFN (**192 routed experts + 1 shared, top-8, FFN 512**) |
| `D` | Dense MLP | Standard SwiGLU FFN (현재 0 layer; 향후 DSV3-style 도입 시 재활성화 가능) |

### Layer Mapping (2:1)
- **Megatron**: 48 layers (each = 1 pattern token)
- **HuggingFace**: 24 layers (MG layer i → HF layer i/2)

## Key Constraints

| Constraint | Value | Reason |
|------------|-------|--------|
| TP (Tensor Parallel) | **1** | GatedDeltaNet (Megatron `MambaLayer` 슬롯) don't support TP > 1 |
| EP (Expert Parallel) | 8 (학습) | **192 experts / 8 GPUs = 24 experts per GPU**. 변환/검증 시 EP는 GPU 수에서 유도(EP=#GPU); torch_dist가 학습 EP=8 → 임의 EP로 resharding |
| Backend | Megatron-LM-251125 | Muon optimizer + DSV3 MoE routing flags (group-limited, topk-scaling, seq_aux_loss) native 지원 |

## Quick Commands

### Training (3-axis preset selection)
```bash
cd examples/alpha
bash train.sh <model> <training> <data> [extra-megatron-args...]

# Examples (post-migration baseline):
bash train.sh baseline_48L stage1 stage1_v5_blend                  # Stage 1 from-scratch (post-2026-05-12 preflight; DCLM+Korean+FW2HQ blend)
bash train.sh baseline_48L stage2_3 stage2_2                       # Stage 2-3 (legacy data, vocab mismatch — 사용 시 주의)
bash train.sh smoke smoke mock                                     # 2-iter smoke test (auto wandb disable)
bash train.sh baseline_48L stage1 mock --train-iters 100           # Mock data 검증 (auto wandb disable)
bash train.sh baseline_48L stage2_3 stage2_2 --lr 5e-4             # CLI override
```

Each preset name resolves to `configs/<group>/<preset>.yaml`. YAML keys are Megatron CLI flag names directly — `train.sh` expands them via `yaml_to_flags`. Anything after the three preset names is forwarded verbatim.

**Smoke / mock auto-detect**: preset 이름 중 어느 하나가 `smoke`이거나 data preset이 `mock`이면 `train.sh`가 자동으로 `WANDB_MODE=disabled`를 export하고 dummy `--wandb-exp-name`을 emit해서 wandb 로깅을 차단합니다. Banner에 `wandb: DISABLED (smoke preset detected)` / `online` / `off (no API key)` 중 하나가 표시됩니다.

The launcher derives run-identity flags: `--save`, `--tensorboard-dir`, `--data-cache-path`, and (when not in smoke mode + `WANDB_API_KEY` 있음) `--wandb-exp-name`.

### Evaluation Pipeline (unified — MG→HF 변환 → 검증 → 벤치마크)

기존의 수동 3단계(변환 → `validate.sh` → `run_benchmarks.sh`)를 한 launcher로 통합.
**모든 변환/검증 args는 체크포인트 `common.pt`(ground truth)에서 유도** → config drift(184/192,
mamba-dim 등) 구조적 차단. EP는 런타임 GPU 수에서 유도(EP=#GPU; torch_dist resharding).

```bash
cd examples/alpha
# Stage 0 preflight → 1 convert → 1.5 verify(config.json↔common.pt + tokenizer) → 2 weight diff
bash evaluate.sh outputs/alpha_baseline_48L_stage1_<TS> --gpus 4
# 벤치마크까지: --benchmark --tasks standard
# 변환 건너뛰고 재검증만: --skip-convert   /  검증 생략: --skip-validate
# 특정 iter: --iter 10000  (기본 latest)
```

개별 단계도 그대로 사용 가능:
```bash
# 변환만 (GPU 수 매개변수화; run_8xH20.sh/run_4xGPU.sh는 GPUS=8/4 shim)
GPUS=4 bash ../../toolkits/distributed_checkpoints_convertor/scripts/alpha/run_convert.sh \
  baseline_48L <ckpt_or_run_dir> auto true true bf16
# 검증만 (모델 args는 checkpoint에서 자동 유도)
bash validate.sh <mg_ckpt_iter_dir> <hfmodel_dir>
# config 점검 (GPU 불필요)
python tools/alpha_config.py emit-megatron-flags --from-checkpoint <ckpt>
python tools/verify_pipeline.py preflight --from-checkpoint <ckpt> --gpus 4
```

전체 v1→v2 검증 감사 + 발견·수정 내역: [`docs/V2_PIPELINE_VERIFICATION.md`](docs/V2_PIPELINE_VERIFICATION.md).

### SGLang Deployment (Inference)
```bash
# One-time setup (patches SGLang submodule + installs model adapter)
bash backends/sglang/setup.sh

# Option A: HF Fallback (quick, no hybrid optimizations)
bash examples/alpha/sglang/deploy.sh /path/to/alpha-hf --mode fallback

# Option B: Native Qwen3-Next (MambaRadixCache + dual memory pool)
bash examples/alpha/sglang/deploy.sh /path/to/alpha-hf --mode native --ep 8
```
SGLang backend is managed as a git submodule at `backends/sglang/sglang-v0.5.2/`.

### Parameter Analysis
```bash
python calculate_parameters.py --config configs/model/baseline_48L.yaml
```

## Config Structure

Flat YAMLs whose top-level keys are Megatron CLI flag names directly. No nested `model.moe.X` schema — `yaml_to_flags` (in `train.sh`) reads each key as `--<key>`. Boolean true emits `--flag` (store_true), false omits it. Lists become `--flag a,b`. Strings with spaces (e.g. `data-path: "0.3 /foo 0.7 /bar"`) survive as multiple argv tokens.

```
configs/
├── model/baseline_48L.yaml          # Architecture + tokenizer_v5 + DSV3 MoE routing
├── model/smoke.yaml                 # 2-layer toy model for smoke tests (uses tokenizer_v5)
├── training/pretrain_auxfree.yaml   # Stage 1 from-scratch, aux-loss-free routing (DSV3-aligned)
├── training/stage2_2.yaml           # Stage 2-2 cosine (DSV3 routing: seq_aux_loss + sigmoid + bias) — 실패한 v1 레시피, 참조용
├── training/stage2_ab.yaml          # ★ v2 stage2 A/B: stage1 레시피 + GBS 3072 + LR 2.5e-5 연속 + ckpt 71526 finetune
├── training/arxive/stage1_optsave.yaml  # stage1 − no-save-optim (optimizer state 포함 저장; resume 테스트용 — arxive로 이동됨)
├── training/stage2_3.yaml           # Stage 2-3 (4× LR, DSV3 routing)
├── training/smoke.yaml              # 2-iter, no-Muon smoke
├── data/stage1_v5_blend.yaml        # ★ v5 Stage 1 blend (DCLM+Korean+FW2HQ, ~466B)
├── data/stage2_v5_blend.yaml        # ★ v5 Stage 2 blend (10 datasets, 1.686T, weight-less → CC-HQ 60%)
├── data/stage2_v5_blend_cc40.yaml   # ★ same 9, CC-HQ capped 40% (compute_blend_weights.py; for small budgets)
├── data/kormo_1pct.yaml             # legacy: pre-tokenized with Qwen3 tokenizer (vocab 151,936)
├── data/kormo_50pct.yaml            # legacy: pre-tokenized with Qwen3 tokenizer (vocab 151,936)
├── data/kormo_code_balanced.yaml    # legacy
├── data/arxive/stage2.yaml          # legacy: Stage 2 9-dataset blend (Qwen3 tokenizer; v5 source-of-truth for stage2_v5_blend)
├── data/arxive/stage2_2.yaml        # legacy: same blend (Qwen3 tokenizer)
└── data/mock.yaml                   # --mock-data for smoke tests
```

**Tokenizer**: `examples/alpha/tokenizer_v5/` — alpha 전용 BBPE (HuggingFace `PreTrainedTokenizerFast`). 5개 파일 (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `tokenizer_metadata.json`, `training_config.yaml`), effective vocab 163,860, padded to 163,968. **EOS/EOD=`<|endoftext|>` (id 0)**, PAD=`<|pad|>` (id 1), BOS=None, +80 reserved special tokens. `<|im_end|>` (id 3)은 vocab에 존재하지만 SFT 단계의 chat-turn-end 용도로 reserve (Qwen3 / Llama 3 / DSV3가 pre-training EOD와 chat token을 분리하는 frontier convention과 align). 2026-05 pre-flight verification 중 `<|im_end|>`을 EOS로 designate 하던 이전 설정이 발견되어 변경.

**Legacy data presets** (`kormo_*`, `stage2*`)는 옛 Qwen3 tokenizer로 토큰화된 .bin/.idx를 가리키므로 새 baseline (vocab 163,968)과 **호환되지 않습니다**. Stage 2 학습 재개 시 v5 tokenizer로 재토큰화해야 함 (`toolkits/pretrain_data_preprocessing/preprocess_*.sh` — 모두 `tokenizer_v5/` 경로로 갱신됨).

Env vars (CUDA, NCCL, TE) are exported by `train.sh` directly; there is no `env.yaml`. Multi-node distributed args (`WORLD_SIZE`, `RANK`, `KUBERNETES_CONTAINER_RESOURCE_GPU`) are detected automatically.

### Key Config Fields (baseline_48L.yaml — flat, post-migration)
```yaml
# Architecture (Qwen3.5 dimensioning)
num-layers: 48                                              # MG layers (= pattern length)
hybrid-override-pattern: "M-M-M-*-M-M-M-*-..."              # 18 M + 6 * + 24 -, no D
is-hybrid-model: true
hidden-size: 2048
ffn-hidden-size: 8192
num-attention-heads: 16                                     # Q heads (was 32)
kv-channels: 256                                            # head_dim (was 128, Q upcast 16×256=4096)
num-query-groups: 2                                         # GQA

# MoE (Qwen3.5 expert dim, alpha-tuned count for ~15B)
num-experts: 192                                            # 8-multiple, 24/GPU at EP=8 (128→184→192)
moe-router-topk: 8
moe-ffn-hidden-size: 512                                    # was 768
moe-shared-expert-intermediate-size: 512                    # was 768

# DSV3 MoE routing (group-limited + topk scaling)
moe-router-num-groups: 8
moe-router-group-topk: 4                                    # 4×24=96 candidate experts
moe-router-topk-scaling-factor: 2.5
moe-aux-loss-coeff: 1.0e-4                                  # DSV3 (was 1e-3)

# Normalization (Qwen3.5 standard, 1p removed)
normalization: RMSNorm
qk-layernorm: true
# apply-layernorm-1p: removed — _clip_layernorm_gamma() auto-routes via if/else

# Tokenizer (alpha v5)
padded-vocab-size: 163968                                   # was 151936
tokenizer-model: /…/Pai-Megatron-Patch/examples/alpha/tokenizer_v5

# Sequence
seq-length: 4096
max-position-embeddings: 262144
```

### Adding a new training preset
1. Drop a flat YAML at `configs/training/<name>.yaml` (look at `stage2_3.yaml` as a template).
2. `bash train.sh baseline_48L <name> <data>` — that's it. No shell edits.
3. To override a flag for one run, append it: `bash train.sh ... --lr 5e-4`. Shell-CLI overrides YAML.

## Critical Files

| File | Purpose |
|------|---------|
| `train.sh` | Single launcher (~210 lines): yaml_to_flags + multi-node detect + env exports + derived run paths + **smoke/mock auto-detect → wandb auto-disable** |
| `pretrain_alpha.py` | Training entry point (Megatron pretrain() + alpha-specific monkey-patches) |
| `evaluate.sh` | **통합 평가 오케스트레이터** (preflight → convert → verify → validate → opt-in benchmark). `--gpus N`/`--iter`/`--benchmark`/`--skip-*` |
| `validate.sh` | MG↔HF weight validation wrapper — **모델 args를 checkpoint `common.pt`에서 유도** (nested-YAML 파싱·하드코딩 v1 제거) |
| `validate_mg_hf_full.py` | Comprehensive weight validation. v2 MoE 대응: `router.expert_bias↔gate.e_score_correction_bias`, `shared_experts.gate_weight↔shared_expert_gate.weight` 비교, transient `local_tokens_per_expert` 제외 |
| `tools/alpha_config.py` | Config inspection + **`load_config_from_checkpoint()`** + **`emit-megatron-flags`**(convert/validate 공용 단일 매핑) + `generate-hf-config`(DSV3 routing 키 포함). `--from-checkpoint` 지원 |
| `tools/verify_pipeline.py` | 검증 게이트: `preflight` / `compare-config`(config.json↔common.pt) / `tokenizer-roundtrip` |
| `calculate_parameters.py` | Parameter count tool — accepts flat YAML; reports 15.08B for current baseline |
| `tools/compute_blend_weights.py` | Stage-2 blend weight 계산 — CC-HQ cap + 자연비율 재분배, epoch 표(over-epoch 경고), data-path/YAML emit. `--cap-cchq`/`--budget-t`/`--write` (§"Stage 2 v5 Re-tokenization") |
| `tokenizer_v5/` | **Alpha 전용 v5 tokenizer** (5 files, 12.6MB; HF `PreTrainedTokenizerFast`, vocab 163,860) |
| `hf_model/` | HuggingFace model implementation. `AlphaSparseMoeBlock`는 **DSV3 라우팅**(sigmoid + group-limited 8×4 + `e_score_correction_bias` + routed_scaling 2.5; `DeepseekV3TopkRouter` mirror) — 학습과 일치해야 벤치마크 유효. `configuration_alpha.py`에 `scoring_func/n_group/topk_group/routed_scaling_factor`, num_experts default=192. **`e_score_correction_bias`는 fp32 `nn.Parameter` + `_keep_in_fp32_modules_strict`** — buffer/bf16으로 되돌리지 말 것(MG fp32 정렬·라우팅 충실; Known Issues "fp32 router bias 다운캐스트" 참조) |
| `../../toolkits/distributed_checkpoints_convertor/scripts/alpha/run_convert.sh` | **GPU-agnostic 변환기** (EP=#GPU, `num_experts%GPU` 검증, 모델 flags는 `emit-megatron-flags`에서). `run_8xH20.sh`/`run_4xGPU.sh`는 `GPUS=8`/`4` shim |
| `sglang/deploy.sh` | SGLang deployment script (Option A/B, uses local backend) |
| `sglang/convert_config_for_sglang.py` | Alpha→Qwen3-Next config converter (head_dim 256 / vocab 163,968 호환성 검증 필요) |
| `sglang/sglang_alpha_model.py` | SGLang model adapter (mlp_only_layers support) |
| `../../backends/sglang/setup.sh` | SGLang backend setup (patch + adapter install) |
| `scripts/setup_wandb.sh` | Sourced by train.sh to set `WANDB_API_KEY` (smoke/mock 시 auto-override됨) |
| `diloco_patch.py` | **DiLoCo 2노드 코어**: train_step 래핑, 페어별 Gloo sync, outer Nesterov, τ-오버랩, dense dedup, outer-state 저장/복원, 데이터 샤딩. 검증 기록은 `study/diloco_pilot.md` |
| `pretrain_alpha_diloco.py` | DiLoCo 엔트리 (diloco_patch 설치 후 pretrain_alpha 실행; train.sh `PRETRAIN_SCRIPT` env로 주입) |
| `launch_diloco.sh` | 2노드 런처 (env knob·데이터 모드·resume 규칙은 파일 헤더 주석) |

## Troubleshooting

### Common Errors

| Error | Solution |
|-------|----------|
| `assert self.tp_size == 1` | Set `--tensor-model-parallel-size 1` |
| `Pattern length mismatch` | Pattern must have exactly `num_layers` characters |
| `Invalid characters in pattern` | Only use M, *, -, D |
| `Attention ratio mismatch` | Count of `*` should match `hybrid_attention_ratio × num_layers` |
| `MambaLayer has no attribute self_attention` | QK-Clip bug with hybrid models — fixed in `pretrain_alpha.py` via monkey-patch |
| `Unsupported function referenced: get_int_dtype` (Triton JIT, 1st step) | Wrong package versions — pin **triton 3.3.0 / mamba-ssm 2.2.6.post3 / fla 0.4.1**; see `### Environment Issues` |

### Environment Issues
```bash
bash scripts/validate_environment.sh  # Check CUDA, Flash Attn 3, TE version
```
**Pinned package stack (required to run alpha forward/backward).** The GatedDeltaNet
and TE-MoE-permute Triton kernels need **triton 3.3.0 / mamba-ssm 2.2.6.post3 /
fla 0.4.1** (TE stays 2.9.0). Newer versions (triton 3.7, mamba-ssm 2.3.2, fla
0.5.0) crash the 1st step with `Unsupported function referenced: get_int_dtype`.
Canonical pins + rationale: `../../../setup_pai_megatron_env_with_deepep.sh` (repo
parent dir) Steps 8/9/11/14b — mamba-ssm must be **built from git** (PyPI sdist
lacks `csrc/`); sudo is passwordless on the analysis node.

**2노드 H100 클러스터(Backend.AI NGC 25.03)는 `setup_pai_megatron_env_multinode.sh`
사용** (repo 부모 디렉토리; PIP_CONSTRAINT 우회 + canonical pin + cuDNN 9.24 +
TE wheel 노드 간 재사용 — Known Issues "2노드 환경 셋업" 참조). 추가 런타임 요건
둘은 train.sh가 자동 처리: cuDNN 9.24 LD_PRELOAD(QK-Clip fused-attn),
`NCCL_MAX_NCHANNELS=16`(optimizer-state resume).

## Profiling & Throughput Optimization

Full guide: [`docs/throughput_optimization.md`](docs/throughput_optimization.md).
Profile on an idle H100×N node (not the live training run):

```bash
# Capture an nsys trace of a half-depth twin (analysis_24L) at EP=N.
NSYS=1 bash train.sh analysis_24L profile mock \
    --profile-ranks 0 --profile-step-start 6 --profile-step-end 7 --train-iters 9
#   -> outputs/<run>/logs/nsys_<run>.nsys-rep   (open in Nsight Systems >= 2025.3)

# Quantify what's optimizable (sweepline: idle / exposed-comm / launch-bound tail).
python3 tools/analyze_nsys_trace.py outputs/<run>/logs/nsys_<run>.nsys-rep
```

Artifacts: model preset `configs/model/analysis_24L.yaml`, training preset
`configs/training/profile.yaml`, the `NSYS=1` wrapper in `train.sh`, and
`tools/analyze_nsys_trace.py`.

**Headline findings (A/B validated 2026-06-09, analysis_24L EP=4):**
- **Lever 1 (free ~3%):** `CUDA_DEVICE_MAX_CONNECTIONS=1→8` gives **+2.7%**
  throughput (bit-identical; safe because alpha is TP=1/CP=1). `train.sh` now
  honors the env override (default still 1).
- **Lever 2 (A/B VALIDATED 2026-06-10, +15.2%):** `--recompute-modules layernorm`
  (drop `moe`) = 133.9→154.3 TFLOP/s at +9.4 GB max-alloc on analysis_24L. Prod 48L
  ≈ +19 GB/rank — verify live-run memory headroom, then adopt at checkpoint resume.
- **Tested & rejected (2026-06-10): layer-scope CUDA graphs** (GB200 DSV3 recipe,
  `--cuda-graph-impl transformer_engine`) — moe scopes conflict with moe recompute
  (hard error) and OOM without it; attn scope crashes alpha QK-Clip (host-side
  max-logit stash never runs under graph replay); mamba scope captures fine but
  nets **−4.0%**. See `docs/throughput_optimization.md` § CUDA-graph A/B.
- **CRITICAL methodology note:** the analysis model's GBS (96) vs prod (1536)
  inflates **per-step** costs ~8×. The **Muon optimizer looked like the top lever
  (~12%/44%-idle here) but is GBS-invariant → ~1.6% on prod** (NVIDIA: opt = 1–3%
  of a real step) — **do NOT prioritize it**. Per-token levers (comm, GEMM/fp8,
  recompute) transfer by fraction; per-step levers (optimizer) do not.
- **Tested & rejected:** `moe-shared-expert-overlap` even at conn=8 nets ~0 — it
  *does* hide the A2A (SendRecv 76%→58% exposed) but per-layer cross-stream
  barriers eat the gain. The ~20% "exposed" EP all-to-all is mostly **structural**
  (critical-path dispatch→GEMM→combine), not a cheap overlap miss. (Same root cause
  makes DeepEP a net loss on single-node NVLink — it's a multi-node tool.)

See the guide for the full A/B table and the **mid-training application protocol**
(scheduling-only changes apply at checkpoint resume; dynamics-changing ones wait
for a stage boundary).

## Multi-Node Training — DiLoCo (2-node, IB 없음) (2026-07-13~15 검증 ✅)

이 클러스터(Backend.AI, H100×8 ×2노드)는 **InfiniBand가 없다** (HCA는 보이지만 전부
admin-Disabled; 노드 간 실효 ~1 GB/s TCP/VXLAN, `study/netbench/`로 측정). sync-DP는
bf16 grad-reduce + GBS 3072에서도 **1.15×**에 그친다(실측; GBS 256이면 1노드보다 느림).
해법은 **DiLoCo** (arXiv:2311.08105; Muon inner 호환은 MuLoCo arXiv:2505.23725):
노드별 독립 단일노드 Megatron 인스턴스가 H스텝 로컬 학습 후 pseudo-gradient를 평균
(outer Nesterov lr 0.7/μ 0.9) — 노드 간 통신이 스텝 경로에서 사라진다.
**전체 실측·검증 기록: [`study/diloco_pilot.md`](study/diloco_pilot.md)** (필독).

### Quick Start

```bash
# 프로덕션 (완전 서로소 데이터 샤딩 — 필수):
DILOCO_DATA_SHARD=1 DILOCO_H=30 DILOCO_TAU=2 \
  bash launch_diloco.sh <tag> baseline_48L <training_preset> <data_preset> [extra...]

# A/B 실험 (seed 분리 — node0가 1노드 기준선과 bit-비교 가능; 풀 예산 시 ~17% 중복):
DILOCO_H=30 bash launch_diloco.sh <tag> baseline_48L stage1 stage1_v5_blend --exit-interval 500

# resume: 각 노드가 자기 체크포인트를 로드해야 함
NODE0_ARGS="--load <node0 ckpt>" NODE1_ARGS="--load <node1 ckpt>" DILOCO_DATA_SHARD=1 ... bash launch_diloco.sh ...
```

구현: `diloco_patch.py`(코어) + `pretrain_alpha_diloco.py`(엔트리, train.sh
`PRETRAIN_SCRIPT` env로 주입) + `launch_diloco.sh`(2노드 런처, env knob 문서는 파일 헤더).

### 검증 요약 (상세는 study/diloco_pilot.md)

| 항목 | 결과 |
|---|---|
| from-scratch 500-iter A/B vs 역사적 stage1 | 동일 iter −0.44 loss (2× 데이터), sync 16/16 일치 |
| v2 overlap (τ>0) + dense dedup | 노출 오버헤드 +10.6% → **+0.35%** (H=30) |
| H=5 vs H=30 | **교차 발견**: H↓≠sync-DP 근접 — H와 outer lr/μ는 결합, H 변경 시 재튜닝 필요 |
| **full-state resume** (optim state 포함) | 2노드 ckpt→1노드 인계 포함 PASSED, 3중 bit-일치 검증 |
| stage2 레짐 (성숙 모델, LR 2.5e-5, GBS 3072) | 1노드와 동률(무해), step +1%로 2× 토큰 |
| 데이터 샤딩 (`DILOCO_DATA_SHARD=1`) | 단일 전역 순서의 `world*i+r` 분할, 중복 0 (스모크 검증) |

### 규칙/함정

1. **프로덕션은 `DILOCO_DATA_SHARD=1` 필수** — 동일 seed 강제(페어 assert). seed-split은 A/B 전용.
2. **H를 바꾸면 outer lr/μ 재튜닝** — H=5에서 문헌 기본값(0.7/0.9)은 초반 우세 후 역전됨.
3. 체크포인트 저장 시 pending sync는 자동 드레인됨; outer state(θ+momentum, fp32
   ~27GB/rank)는 `<save>/diloco_outer/iter_N/`에 동봉되어 resume 시 자동 복원.
   1노드 인계 시에는 자동 무시(순수 Megatron 로드).
4. wire 시간은 NFS/시간대에 따라 40~130초(주간 최대 408초@GBS3072 관측) 요동 —
   τ는 wire p99 기준으로 (H=30이면 τ=2~3 권장).
5. 2노드 실행엔 반드시 `NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 GLOO_SOCKET_IFNAME=eth0`
   (런처가 자동 설정).
6. **2노드→1노드 축소**: 샤드 체크포인트의 카운터는 로컬(N×GBS)이므로 naive resume은
   직전 절반 중복 또는 홀수 샤드 영구 누락. 반드시 unshard 모드 사용:
   ```bash
   PRETRAIN_SCRIPT=$ALPHA/pretrain_alpha_diloco.py DILOCO_UNSHARD_RESUME=1 DILOCO_WORLD=2 \
     bash train.sh baseline_48L <preset> <data> --load <node0_ckpt>
   ```
   consumed/스케줄러/iteration 세 카운터를 ×world 보정 (검증: 재개 첫 LR이
   전역 위치 산식과 유효숫자 6자리 일치). WSD stable 구간에서 전환하는 것이 가장 안전.
   역방향(1→2노드 확장)은 같은 규칙의 역(÷world + 샤드 래퍼 on).

## Training Plan (Long-term)

### Stage 0 — Qwen3.5 dimensioning + DSV3 MoE migration (이번 세션 완료 ✅)

**목표**: Qwen3-Next reference에서 Qwen3.5 dimensioning + DeepSeek-V3 MoE routing으로 baseline 정렬, alpha 전용 v5 tokenizer 채택, RMSNorm 표준화.

**5번의 smoke test** (각 단계, 2-iter, 8 H100, mock data, exit 0 / NaN 0 검증):

| # | 변경 | 검증 결과 |
|---|------|-----------|
| 1 | Qwen3.5 dim (head_dim 256, 184 experts × 512) + DSV3 routing | per-rank 3.21B, total 15.03B, loss 11.99→11.44 |
| 2 | `apply-layernorm-1p` 제거 (Qwen3.5 정렬) | iter 1 forward 동등 (loss 11.99280 동일), iter 2 분기 |
| 3 | WD policy `apply_wd_to_qk_layernorm` (Stage 1+2 통일, Qwen3-Next NVIDIA 레시피) | grad norm 무영향, loss curve 안정 |
| 4 | tokenizer v5 (beta path 참조) | per-rank 3.26B, total 15.08B, loss 12.07→11.51 |
| 5 | tokenizer v5 (in-repo `tokenizer_v5/`) | iter 1 lm_loss byte-perfect 일치 → migration 완전 동등 입증 |

**옛 checkpoint 호환성**: Stage 2-3까지의 모든 checkpoint(`outputs/alpha_baseline_48L_*`)는 (1) head_dim, (2) expert 수, (3) MoE FFN dim, (4) vocab size가 모두 변경되어 새 baseline과 **structurally incompatible**합니다. 새 stage 1을 from-scratch로 시작해야 함.

**Stage 1 (post-migration) 시작 명령**:
```bash
bash train.sh baseline_48L pretrain_auxfree stage1_v5_korean_web
# (또는 다른 v5-tokenized data preset)
```

---

### Pre-migration history (legacy, 호환 X)

> 아래 Stage 1/2-1/2-2/2-3은 Qwen3-Next-호환 baseline (128 experts × 768 FFN, vocab 151,936, 1p RMSNorm) 시절의 학습 기록입니다. 새 baseline과 구조적으로 호환되지 않으므로 **재현 또는 continual learning 불가**. Historical reference (학습 곡선, 발견된 bug fix 등)로만 보존.

### Stage 1: Initial Pre-training (완료)
- **Dataset**: kormo_50pct (~1.13T tokens)
- **Iterations**: 400k + 40k cooldown = 440k
- **Run**: `bash train.sh baseline_48L pretrain_auxfree kormo_50pct` (current preset, with auxfree routing)
- **Checkpoint**: `outputs/alpha_baseline_48L_cooldown_20260209_200711/checkpoints`

### Stage 2-1: Continual Pre-training (200k/400k에서 중단)
- **Dataset**: stage2 blend (~3.1T tokens) — Korean Web + Math + Nemotron CC-HQ + Nemotron Code v2
- **전략**: WSD continual learning (12k warmup + 348k stable + 40k decay)
- **All-to-All dispatcher** — DeepEP 대비 ~7% 빠름 (단일 노드 벤치마크)
- **QK LayerNorm WD**: `no-weight-decay-cond-type: apply_wd_to_qk_layernorm` — Stage 1에서 발견된 gamma 폭발 방지
- **Checkpoint**: `outputs/alpha_baseline_48L_stage2_20260301_015403/checkpoints` (200k iter)
- **중단 사유**: swap memory 포화로 throughput 불안정

### Stage 2-2: Continual Pre-training (200k→400k, cosine) (진행 예정)
- **Dataset**: stage2 blend 이어서 (consumed_samples 유지)
- **전략**: cosine decay (500 warmup + 199.5k cosine decay), WSD에서 변경
- **Config**: `configs/training/stage2_2.yaml` + `configs/data/stage2_2.yaml`
- **변경사항 (vs Stage 2-1)**:
  - LR scheduler: WSD → cosine warmup+decay
  - num-workers: 32 → 8 (swap memory 문제 해결)
  - Nesterov 버그 자동 수정 (yaml_to_flags가 `muon-use-nesterov: true` → `--muon-use-nesterov` 정확히 emit)
  - `no-load-optim: true`로 optimizer/scheduler 리셋, data position 유지
- **실행**: `bash train.sh baseline_48L stage2_2 stage2_2`

### Stage 2-3: Continual Pre-training (375k→800k, 4× LR boost)
- **Dataset**: stage2 blend (consumed_samples 유지, `data/stage2_2.yaml` 그대로)
- **전략**: cosine decay (2k warmup + 423k cosine), 4× LR boost
- **Config**: `configs/training/stage2_3.yaml`
- **변경사항 (vs Stage 2-2)**:
  - LR: 1e-4 → 4e-4 (4× boost), min-lr: 1e-5 → 4e-5
  - Warmup: 500 → 2000 (LR jump 안정화)
  - LayerNorm WD: `apply_wd_to_qk_layernorm` → `apply_wd_to_all_layernorm`
- **실행**: `bash train.sh baseline_48L stage2_3 stage2_2`

#### Stage 전환 규칙 (training preset YAML에 직접 명시)
| 전환 | Dataset 변경? | YAML 키 | consumed_samples |
|------|:---:|---|---|
| Stage 1 → 2-1 | **Yes** | `finetune: true` | 0 리셋 |
| Stage 2-1 → 2-2 | No | `load: <path>` + `no-load-optim: true` | 이어서 |
| Stage 2-N → 2-N+1 | No | `load: <path>` + `no-load-optim: true` | 이어서 |

#### 핵심 설계 원칙
1. **`no-save-optim: true`**: Muon optimizer는 warmup으로 충분히 복구 (Stage 1 cooldown에서 검증)
2. **`finetune: true`는 dataset 변경 시에만**: iteration/consumed_samples 리셋 필요할 때
3. **`no-load-optim: true`는 같은 dataset 연장 시**: 데이터 위치는 유지, scheduler만 리셋
4. resume 관련 키는 모두 training preset YAML 안에 평면적으로 (`load:`, `finetune:`, `no-load-optim:`) — 셸이 조건부로 끼워넣지 않음

## Known Issues & Fixes

### NGC 25.03에서 QK-Clip fused-attn 크래시 — cuDNN 9.8에 max_logit 엔진 없음 (2026-07-13 ✅)

- **증상**: 학습 첫 step에서 `cuDNN Error: No valid engine configs for Matmul_MUL_GEN_INDEX_..._Matmul_`.
  기본 attention은 통과하는데 **`return_max_logit=True`(QK-Clip 경로)만 실패**.
- **원인**: TE 2.9의 max-logit fused-attn 그래프 엔진이 cuDNN 9.11+에만 존재. NGC 25.03은
  9.8. TE의 backend 선택기는 이 케이스에 cudnn 버전 게이트가 없어(utils.py — thd/fp8만
  거름) 폴백 대신 런타임 크래시.
- **수정**: `pip install --no-deps nvidia-cudnn-cu12==9.24.0.43` (**--no-deps 필수** —
  의존성으로 딸려오는 cublas 12.9가 환경을 깨뜨림) + train.sh가 pip cuDNN 발견 시 전체
  서브라이브러리를 **LD_PRELOAD** (LD_LIBRARY_PATH만으로는 TE RUNPATH 탓에 9.24/9.8이
  섞여 `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`). 멀티노드 셋업 스크립트가 자동 설치.

### Optimizer-state resume이 첫 collective에서 크래시 — NCCL comm-init OOM (2026-07-15 ✅)

- **증상**: `--load`로 optimizer state까지 실은 resume이 `Failed to CUDA calloc async N bytes`
  (N은 4~608B로 미미)로 사망. fresh 학습은 정상. **DiLoCo 무관 — 순수 Megatron도 재현.**
- **원인**: NCCL 2.25 기본 64채널의 comm당 GPU 버퍼가 크고, Megatron은 comm이 많다.
  fresh는 optim state가 첫 step 이후 생성되어 comm 초기화가 저메모리 구간에서 일어나지만,
  resume은 로드된 state + 비동기 in-flight 할당 위에서 지연 초기화 comm들이 일제히 버퍼를
  요구 → NCCL 'out of memory'. 판별 근거: `CUDA_LAUNCH_BLOCKING=1`이면 통과 + NCCL_DEBUG.
- **수정**: **`NCCL_MAX_NCHANNELS=16`** (train.sh 기본값) — comm 버퍼 4× 절감, step time
  무손실(60.7s vs 61.1s). 검증: 3개 독립 resume의 iter-13 loss **bit-identical**.

### DiLoCo: 저장 시점에 pending sync 살아있으면 저장 크래시 (2026-07-14 ✅)

- **증상**: `--exit-interval`(또는 save-interval)이 H의 배수일 때 마지막 sync가 시작만 된 채
  (τ>0, 미적용) torch_dist 저장의 NCCL gather가 `unhandled cuda error`로 사망. 100% 재현.
- **수정**: diloco_patch의 save 훅이 pending sync를 **join+apply 후 저장**(드레인). 부수
  효과로 체크포인트가 일관된 outer 상태를 담음. 같은 시기 수정: τ-apply의 파라미터별 GPU
  임시버퍼(→allocator 단편화)를 per-dtype 영구 scratch 버퍼로 대체.

### 2노드 환경 셋업: NGC 25.03의 PIP_CONSTRAINT·TE 서브모듈 순서 (2026-07-13 ✅)

- 기존 `setup_pai_megatron_env.sh`가 이 이미지에서 3중 드리프트로 연쇄 실패: ① TE upstream
  main의 서브모듈 구성 변경(checkout *후* `git submodule update` 필요), ② 이미지 전역
  `PIP_CONSTRAINT=/etc/pip/constraint.txt`가 명시 핀과 충돌(importlib-metadata/packaging/
  **transformer-engine 자체**), ③ mamba-ssm/fla 미고정 설치가 현 PyPI 최신(triton 3.7
  강제)을 끌어옴. → **`setup_pai_megatron_env_multinode.sh`** (repo 부모 디렉토리) 사용:
  선별적 `env -u PIP_CONSTRAINT`, canonical pin(mamba v2.2.6.post3 git 빌드, fla==0.4.1),
  `NVTE_CUDA_ARCHS=90`(빌드 30분+→3분), TE wheel을 workspace에 보존(타 노드 재빌드 생략),
  cuDNN 9.24 설치 포함. 원본 스크립트는 참조용 무수정 보존.

### MG↔HF weight 검증이 `expert_bias` 1개에서 지속 실패 — fp32 router bias 다운캐스트 (2026-06-15 ✅)

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

### HF `AlphaRMSNorm`이 zero-centered(1p) → 모든 벤치마크 random (silent, 2026-05-26 ✅)

- **증상**: v2 체크포인트를 `evaluate.sh`로 끝까지 돌리면 **모든 게이트(weight 검증·config 대조·tokenizer)는 통과**하는데 Stage 3 벤치마크만 random (ARC-easy 정확히 25%).
- **근본 원인**: `hf_model/modeling_alpha.py::AlphaRMSNorm`가 Qwen3-Next에서 물려받은 **zero-centered `x_norm * (1 + γ)`** 를 적용. 그러나 Alpha v2는 Megatron을 **`apply-layernorm-1p` OFF(표준 `x_norm * γ`, γ≈1)** 로 학습 (checkpoint `common.pt`: `apply_layernorm_1p=False`, 저장 γ mean≈0.69~1.5). → 모든 norm(input/post/q/k/final, 24레이어)이 **~1.7~2.5× 과증폭** → 잔차 스트림 누적 왜곡 → near-uniform logit (perplexity≈vocab). NaN 아님(scale 오류). 같은 파일 `AlphaRMSNormGated`는 이미 표준(`*γ`)이라 GDN norm은 정상이었음 → 두 norm 클래스가 불일치했던 것.
- **수정**: `AlphaRMSNorm`를 표준으로 — forward `output * self.weight.float()` (1+ 제거), init `torch.ones`. `AlphaRMSNormGated`와 일관.
- **실측 검증** (iter_0010000, 단일 모델 로드 monkeypatch): `(1+γ)` ppl=295,440 / greedy `'…andNV and) From Form'` → `γ` ppl=8.84 / greedy `'The capital of France is Paris.'`. ARC-easy 0-shot(100): **25% → acc 0.73 / acc_norm 0.76**.
- **왜 weight 검증이 못 잡았나 (교훈)**: `validate_mg_hf_full.py`는 **weight tensor만** 비교하고 forward를 안 한다. converter가 γ를 그대로 복사 → 검증은 MG γ==HF weight 통과. 차이는 forward의 `1+` 에서만 발생 → 사각지대. 게다가 attention 비교는 converter의 reshape를 복제(`Reference: m2h_synchronizer.py`)해서 해석 오류를 양쪽이 공유.
- **회귀 가드**: ① `examples/alpha/forward_sanity.py` — 변환 HF 모델 perplexity 게이트(임계 100; random≈vocab). ② `evaluate.sh` **Stage 2.5**로 편입(weight 검증 후·벤치마크 전). ③ `tests/test_alpha_pipeline_config.py::test_modeling_alpha_rmsnorm_is_standard_not_1p`. **기존 변환 산출물은 재변환 불필요** — weight는 정상이므로 `hf_model/modeling_alpha.py`만 HF 디렉토리에 재복사하면 됨.
- **부수**: `toolkits/distributed_checkpoints_convertor/impl/alpha/m2h_synchronizer.py:246` bias 경로 `linear_qkv`→`linear_qgkv` 오타 정리(dormant: `attention_bias=False`). bias 활성화 시 q bias에 weight-path의 gate-interleave transpose 필요(주석 추가).

### Stage 1 재개 (10k) + 처리량 최적화 — `stage1_resume.yaml` (2026-05-26)

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

#### ⚠️ 처리량 회귀 (2026-05-26) — `moe-shared-expert-overlap`이 범인

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

### v2 평가 파이프라인 통합 + v1→v2 검증 (2026-05-26 ✅)

iter_0010000(첫 v2 체크포인트) 평가를 위해 수동 3단계(MG→HF 변환 → `validate.sh` → `run_benchmarks.sh`)를 `evaluate.sh` 하나로 통합. **근본 원인**: 같은 모델 config가 3곳(학습 YAML / 변환 `baseline_48L.sh` / `validate.sh` 하드코딩)에 중복되어 drift. 해결: **모든 변환/검증 args를 체크포인트 `common.pt`(ground truth)에서 유도**(`tools/alpha_config.py::load_config_from_checkpoint` + `emit-megatron-flags`), 병렬화(EP)만 런타임 GPU 수에서 유도. 전체 감사는 [`docs/V2_PIPELINE_VERIFICATION.md`](docs/V2_PIPELINE_VERIFICATION.md).

**파이프라인이 잡아낸 3개의 실제 버그** ("crash > silent corruption" — 올바른 플래그를 켜자마자 갭에서 멈춤):

1. **Config drift (stale 변환 경로)**: `scripts/alpha/configs/baseline_48L.sh`가 완전 v1(128 experts, head 32, kv 128, vocab 151936, pattern 49자, softmax)이고 `validate.sh`는 nested-YAML 파싱(flat v2에선 빈 값)이었음. 또 변환기가 **mamba 차원을 아예 안 넘겨** code default 64 ≠ 학습 128. → checkpoint 유도로 일괄 해결. **184 vs 192 / aux-free vs seq_aux_loss drift**도 여기 포함(아래 별도 항목).

2. **HF MoE 라우팅 부정합 (silent benchmark 오염)**: `hf_model/modeling_alpha.py::AlphaSparseMoeBlock`이 **plain softmax + 전역 top-k + no bias**로 라우팅 — 학습은 DSV3(sigmoid + 8×4 group-limited + aux-loss-free `expert_bias` + routed_scaling 2.5). 변환이 처음으로 MoE 단계까지 도달하자 `gate.e_score_correction_bias` 부재로 크래시 → 표면화. **수정**: `DeepseekV3TopkRouter`와 동일하게 재작성 + `gate.e_score_correction_bias` persistent buffer; `configuration_alpha.py`에 `scoring_func/n_group/topk_group/routed_scaling_factor`; `generate_hf_config`가 해당 키 emit; `verify_pipeline.py` Stage 1.5가 대조. (없었으면 모든 벤치마크 수치가 wrong-routing으로 무효)

3. **검증 coverage 갭**: 변환은 14133/14133 weight 일치로 성공했으나 `validate.sh`가 72 MG weight 미비교로 exit 1. `validate_mg_hf_full.py`(변환기와 독립 매핑)에 `router.expert_bias↔gate.e_score_correction_bias`, `shared_experts.gate_weight↔shared_expert_gate.weight` 비교 추가 + transient `router.local_tokens_per_expert` 제외 → exit 0.

**회귀 가드**: `tests/test_alpha_pipeline_config.py`(9개; config↔checkpoint 일치, HF config v2 필드, emit 누락, DSV3 라우팅 동작, stale 경로 grep). `configuration_alpha.py` 및 `test_alpha_tokenizer_eod.py`의 stale `num_experts=184`도 192로 정정(테스트 자체가 drift 피해자였음).

### EOS designation 통합: chat-end → pre-training EOD 분리 (2026-05-12 preflight ✅)
- **문제**: alpha v5 tokenizer가 처음에 `eos_token = <|im_end|>` (id 3)으로 설정되어 있었음. 이는 **chat-turn-end marker를 pre-training EOD로도 겸용**하는 것 — frontier convention (Qwen3 / Llama 3 / DSV3가 모두 두 의미를 분리)과 어긋남.
- **수정 (3개 파일 모두)**: `tokenizer_v5/{tokenizer_config.json, special_tokens_map.json, training_config.yaml}` 모두 `eos_token = <|endoftext|>` (id 0)으로 통일.
- **의미 분리**: pre-training은 `<|endoftext|>` (id 0)로 doc boundary, SFT 단계의 chat template은 `<|im_end|>` (id 3)을 turn boundary로. 미래 chat tuned model 출시 시 `generation_config.json`에 `eos_token_id = [3, 0]` override만 추가하면 됨 — tokenizer 파일은 안 건드림 (Qwen3 패턴과 동일).
- **`_AlphaTokenizer.eod` 자동 갱신**: 코드 변경 없음 — property가 이미 `tokenizer.eos_token_id`에 위임 (`megatron_patch/tokenizer/__init__.py:372`). config 한 줄 바꾸자 downstream 모두 자동으로 id 0 반환.
- **놓치기 쉬운 함정**: `tokenizer_config.json`만 바꾸면 HF AutoTokenizer는 OK (그게 우선 source). 하지만 `special_tokens_map.json`을 직접 읽는 도구 (vLLM, SGLang 일부 chat util)는 stale 상태 → silent breakage 가능. **세 파일 동기화 필수**.
- **회귀 테스트**: `tests/test_alpha_tokenizer_eod.py`의 `test_tokenizer_config_eos_is_endoftext` + `test_special_tokens_map_eos_is_endoftext` 가 향후 drift 차단.

### 데이터 EOD remap: id 3 → id 0 (2026-05-12 preflight ✅)
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

### alpha_config.py Qwen3 default token IDs (silent bug, 2026-05-12 preflight ✅)
- **문제**: `examples/alpha/tools/alpha_config.py:48-49`의 `DEFAULT_BOS_TOKEN_ID = 151643`, `DEFAULT_EOS_TOKEN_ID = 151645`가 **Qwen3 vocab의 ID**. 이 파일은 `toolkits/distributed_checkpoints_convertor/scripts/alpha/run_*.sh`가 MG→HF 변환 시 `config.json` 생성 (`alpha_config.py generate-hf-config`)에 사용.
- **잠재 영향**: 변환된 HF model의 `config.json`이 `eos_token_id = 151645` 로 박힘 → 이는 alpha v5 vocab에서 *전혀 다른 BBPE 서브워드*. SGLang/vLLM serving 시 잘못된 stop token → 무한 generation 또는 엉뚱한 위치에서 멈춤. **학습은 영향 없지만 inference deployment 시점에 silent breakage**.
- **수정**: `DEFAULT_BOS_TOKEN_ID = None`, `DEFAULT_EOS_TOKEN_ID = 0`, `TokenConfig.pad_token_id = 1` default 추가. 즉 alpha v5 실제 IDs 반영.
- **회귀 테스트**: `test_alpha_config_token_defaults_are_alpha_v5`.

### configuration_alpha.py stale defaults (silent bug, 2026-05-12 preflight 2nd-pass ✅)
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

### Document boundary handling 활성화 (2026-05-12 preflight ✅)
- **변경**: `stage1.yaml`에 `reset-position-ids: true`, `reset-attention-mask: true`, `eod-mask-loss: true` 추가.
- **이유**: yanring/Megatron-MoE-ModelZoo Qwen3-Next-80B-A3B 레퍼런스 recipe와 정렬. 매 packed sample 안에서 EOD (id 0) 위치마다 position vector reset + cross-doc attention 차단 + EOD 토큰을 loss에서 제외.
- **필수 조건**: 데이터에 EOD가 stream 토큰으로 존재해야 함. Megatron의 `gpt_dataset.py:683` `eod_index = position_ids[data == eod_token]`이 `.bin` 안 id 0을 스캔해서 reset 위치 결정. `.idx::document_indices`는 *sample packing* 단계에서만 쓰이고 runtime reset에는 미사용. 따라서 위 "데이터 EOD remap"이 필수 선행 조건.
- **Differential 검증** (Phase C-loader, `tests/preflight_stage1/C_loader_audit.md`):
  - ON: cross-doc attn 차단 100%, max position_id 평균 ~2000, loss_mask coverage ~99.9%
  - OFF (control): 차단 0%, max position_id 항상 4095, coverage 100%
  - 모든 source에서 expected delta 관찰 → 머신 정상 작동 입증.

### pretrain_auxfree.yaml → stage1.yaml 마이그레이션 (2026-05-12 ✅)
- **변경**: Stage 1 training preset이 `pretrain_auxfree.yaml`에서 `stage1.yaml`로 이동. 새 파일은 더 보수적인 hyperparam (LR 4e-4 → 2e-4, GBS 2688 → 1536, save-interval 25000 → 10000, eval cadence 강화) + 위 3개 reset flags.
- **`pretrain_auxfree.yaml`**: deprecation header 추가, 삭제는 안 함 (in-flight 스크립트 호환성 + git history 가시성).
- **사용**: `bash train.sh baseline_48L stage1 stage1_v5_blend`.

### apply-layernorm-1p 제거 (Qwen3.5 정렬, 이번 세션 ✅)
- **변경**: `baseline_48L.yaml`에서 `apply-layernorm-1p: true` 제거 → 표준 RMSNorm (γ=1 init).
- **이유**: Qwen3.5 official `config.json`에는 zero-centered γ flag 없음. 표준 RMSNorm 채택이 baseline 정렬과 일치.
- **QK-Clip 호환성**: `gated_attention.py:325-329`의 `_clip_layernorm_gamma()`가 `if config.layernorm_zero_centered_gamma`로 분기되어, 1p가 꺼지면 자동으로 표준 `w * scale` 분기로 fall-through. **별도 코드 수정 불필요**.
- **smoke 검증**: 1p ON vs OFF에서 iter 1 forward 동등 (loss 11.99280 일치), iter 2부터 backward dynamics 분기 시작.

### WD policy 통일: `apply_wd_to_qk_layernorm` (Qwen3-Next NVIDIA 레시피, 이번 세션 ✅)
- **변경**: `pretrain_auxfree.yaml`, `stage2_3.yaml`의 `apply_wd_to_all_layernorm` → `apply_wd_to_qk_layernorm`. (`stage2_2.yaml`은 이미 그러함.)
- **이유**: yanring/Megatron-MoE-ModelZoo `Qwen3-Next-80B-A3B.yaml`이 `--no-weight-decay-cond-type: qwen3_next` 명시 ("Qwen3-Next applies weight decay to qk layernorm as a special case"). 이는 `apply_wd_to_qk_layernorm`과 동의어. 즉 **QK norm γ에만 WD, 다른 layernorm γ는 WD 제외**.
- **이전 `apply_wd_to_all_layernorm` 도입 이력**: Stage 2-3에서 LN γ 폭발 fix 시도였으나, Qwen3 family와 정렬을 위해 QK-only로 회귀.

### Tokenizer migration to alpha v5 (in-repo, 이번 세션 ✅)
- **변경**: 기존 `examples/alpha/tokenizer/` (Qwen 호환 BBPE, 7 files, vocab 151,936) → 신규 `examples/alpha/tokenizer_v5/` (alpha 전용 BBPE, 5 files, vocab 163,860; padded 163,968).
- **자동 갱신된 참조** (총 9곳): `configs/model/baseline_48L.yaml`, `configs/model/smoke.yaml`, `tools/alpha_config.py` (default), 7개 `toolkits/pretrain_data_preprocessing/preprocess_*.sh`, `toolkits/data_extraction/extract_training_samples.py`.
- **데이터 호환성**: 새 vocab 163,968은 옛 .bin/.idx (Qwen3 tokenizer로 토큰화)와 mismatch → 모든 학습 데이터 재토큰화 필요.
- **Verification**: smoke test에서 in-repo path와 beta path가 byte-perfect 동일 (iter 1 lm_loss 12.07105 일치) → 5 files만으로 HF AutoTokenizer 동작 충분 확인.

### Smoke / mock 자동 wandb 비활성화 (이번 세션 ✅)
- **변경**: `train.sh`에 `SMOKE_RUN` 자동 감지 (preset 이름 중 `smoke` 또는 data preset이 `mock`이면 true). True 시 `WANDB_MODE=disabled` export + dummy `--wandb-exp-name smoke_<TS>` emit (Megatron `--wandb-project` argparse validation 통과용).
- **이유**: smoke test가 wandb project를 오염시키지 않도록. 기존엔 `mock` data 사용해도 wandb upload 일어남.
- **Banner**: `wandb: DISABLED (smoke preset detected)` / `online (project: alpha-pretraining)` / `off (no WANDB_API_KEY)` 중 하나로 시작 시 즉시 확인 가능.

### Muon Nesterov 버그 (Stage 2-2에서 발견, 자동 수정됨)
- **증상**: YAML에서 `muon_use_nesterov: true` 설정했으나 실제로는 Nesterov가 비활성화
- **원인**: `--muon-use-nesterov`는 argparse `store_true`(default=False). 구식 셸이 `true`일 때 플래그를 전달하지 않아 항상 False. `false`일 때 전달하는 `--muon-no-use-nesterov`도 Megatron에 미정의
- **영향**: Stage 1~2-1 전체에서 일반 heavy ball momentum으로 학습 (Nesterov 미적용)
- **현재 상태**: ✅ 새 train.sh의 `yaml_to_flags`가 store_true semantics를 정확히 재현 (`muon-use-nesterov: true` → `--muon-use-nesterov` emit / false → omit). 같은 부류의 버그는 새 launcher에서는 구조적으로 발생 불가능

### QK LayerNorm Gamma 폭발 (Stage 1에서 발견, Stage 2에서 수정)
- **증상**: 마지막 attention layer(Layer 23)의 `q_norm`/`k_norm` gamma가 11.9~12.9로 폭발 (정상: ~1.97)
- **원인**: QK LayerNorm gamma는 1D param → weight decay 미적용 + QK-Clip이 gradient 신호 차단 → gamma 성장 무제한
- **수정**: `--no-weight-decay-cond-type apply_wd_to_qk_layernorm` (NVIDIA GatedDeltaNet 공식 레시피)
- **설정 위치**: `configs/training/stage2.yaml` → `training.no_weight_decay_cond_type`
- **버그 수정**: `megatron_patch/training.py`에서 `no_weight_decay_cond`를 `setup_model_and_optimizer()`에 전달하지 않던 버그 수정 (upstream Megatron과 동기화)
- **Confluence**: [QK LayerNorm Weight Decay 적용 (Stage 2 버그 수정)](https://alphabanana.atlassian.net/wiki/spaces/AB/pages/10944513)

### QK-Clip crash on hybrid model (Stage 2에서 발견)
- **증상**: `--qk-clip` 사용 시 `AttributeError: 'MambaLayer' object has no attribute 'self_attention'`
- **원인**: Upstream `clip_qk()` (Megatron-LM)이 모든 decoder layer에 `self_attention`이 있다고 가정 → MambaLayer에서 크래시
- **수정**: `pretrain_alpha.py`에서 `clip_qk`을 monkey-patch하여 `hasattr(layer, 'self_attention')` 가드 추가
- **위치**: `examples/alpha/pretrain_alpha.py` (line ~105-136)

### QK-Clip 로깅이 안 되던 문제 (해결 완료 ✅)
- **증상**: `--qk-clip` 설정해도 max attention logit이 로그에 안 나옴
- **원인 분석**:
  - `pretrain_alpha.py`는 `from megatron.training import pretrain` — **upstream** `pretrain()` 사용 (megatron_patch/training.py 미사용)
  - Upstream `train_step()`에 이미 `clip_qk()` 호출이 있어 **QK-Clip 자체는 동작 중**이었음
  - 문제는 upstream `training_log()`가 `--log-max-attention-logit` 플래그 없으면 TensorBoard/WandB에 기록하지 않고, 콘솔에는 아예 출력하지 않음
- **수정**: `train_stage2.sh`의 QK-Clip 인자에 `--log-max-attention-logit` 추가
- **검증**: WandB에서 `max_attention_logit` ≈ 100 (threshold) 근처로 안정 동작 확인
- **참고**: `megatron_patch/training.py`에도 `clip_qk()` 호출 + 로깅을 포팅함 (다른 모델이 patched `pretrain()` 사용 시 필요)
- **Confluence**: [QK-Clip 완전 활성화 (Stage 2)](https://alphabanana.atlassian.net/wiki/spaces/AB/pages/12845058)

### QK-Clip LayerNorm Gamma 스케일링 (GQA+QK-Norm 고유 수정, 구현 완료 ✅)
- **증상**: QK-Clip 적용 후에도 max attention logit이 threshold 근처로 내려가지 않음
- **원인**: QK-Norm(RMSNorm)이 W_q/W_k 스케일링을 상쇄하여 QK-Clip이 사실상 무력화
  - MuonCLIP 논문의 `W_qr`은 MLA query rotary projection이지 LayerNorm gamma가 아님
  - 우리 GQA+QK-Norm 아키텍처에서는 RMSNorm이 projection 스케일링을 정규화하므로, gamma도 함께 스케일링해야 함
  - 이것은 논문에 없는, GQA+QK-Norm 아키텍처 고유의 수정
- **수정**: `megatron_patch/model/qwen3_next/gated_attention.py`에 `_clip_layernorm_gamma()` 메서드 추가
  - `clip_qk()` 내에서 Q/K projection 스케일링 후 `q_layernorm`/`k_layernorm`의 gamma도 스케일링
  - `layernorm_zero_centered_gamma` (1p layernorm) 처리: `(1+w)*scale - 1`
  - 공유 layernorm이므로 `min(eta)` (worst-case head) 사용

## Pre-tokenization Performance (v5 tokenizer) — Critical Lessons (2026-05)

**75시간 DCLM 토큰화가 4시간으로 끝났어야 했음.** 이 섹션은 v5 (HF `PreTrainedTokenizerFast`) 토크나이저로 대규모 데이터 토큰화 시 같은 실수 반복 방지용. **옛 Qwen tokenizer (slow path, `.encoder` dict 있음) 사용 시 해당 없음** — 그 때는 `preprocess_data_megatron.py` 그대로 OK.

### What went wrong
`toolkits/pretrain_data_preprocessing/preprocess_data_megatron.py` 를 default `--workers 64 --partitions 8` (`preprocess_koreanweb_half.sh` 템플릿 default — 74 GB 작업용으로 튜닝됨)로 1.9 TB DCLM에 적용:
- 75시간 wall clock
- 8개 partition 중 data_2가 라인 분포 편차로 5-7h 단독 tail
- Per-worker effective throughput **~13K tok/s** (Rust BPE 잠재력 500K tok/s의 ~3%)

### Root cause
1. **`pool.imap(encoder.encode, fin, chunksize=32)` per-doc IPC overhead** — 각 doc마다 pickle/unpickle, Pool master 단일 스레드 dispatch. 워커가 99% CPU여도 실제 토큰화는 그 시간의 일부.
2. **HF 75× speedup은 fast vs slow 비교** — 우리는 이미 fast 사용 중. 진짜 lever는 **`encode_batch([texts])` batched encoding**인데 코드에서 안 씀.
3. **너무 적은 partition 수 (8)** — 큰 데이터셋에서 한 partition slow가 전체 tail-effect 만듦.
4. **Default RAYON_NUM_THREADS** (= num_cpus = 224) — thread contention으로 single-process throughput 50% 감소.

### Architectural rules (>100 GB 입력 토큰화 시 필수 적용)
1. **`preprocess_data_megatron.py` 사용하지 말 것** for v5 tokenizer + 큰 입력. 직접 `tokenizers.Tokenizer.from_file(tokenizer.json)` + `encode_batch` 사용. (둘 다 byte-perfect 동일 토큰 ID 검증됨 — English/Korean/Mixed/Code 4종 샘플)
2. **Optimal config**: 16-28 Python processes × **8 Rayon threads** each, batch_size **5000 docs**.
3. **`RAYON_NUM_THREADS=8` 명시** before `from tokenizers import Tokenizer`. Default는 thread contention 유발.
4. **Process count >> thread count** for aggregate throughput. 64 cores 사용 시 8p×8t = 18M tok/s, 4p×16t = 10.8M tok/s, 2p×32t = 5.5M tok/s — 같은 cores지만 3× 차이.
5. **Output**: `IndexedDatasetBuilder.add_document(arr, [len(arr)]) + finalize(idx_path)` (in `backends/megatron/Megatron-LM-251125/megatron/core/datasets/indexed_dataset.py`). Per-process parts merge via `IndexedDatasetBuilder.add_index(part_prefix)` — deterministic 순서.
6. **Append EOD (im_end, id=3)** at doc end if `len(ids) > 0` — matches `preprocess_data_megatron.py` semantics for `--append-eod`.

### Verification protocol (mandatory before any multi-hour run)
새 토큰화 스크립트 작성 시 100-doc 샘플로 byte-perfect compare to `preprocess_data_megatron.py`:

```bash
head -n 100 <input>.jsonl > /tmp/sample.jsonl

# Legacy (correctness baseline)
python preprocess_data_megatron.py --input /tmp/sample.jsonl \
  --output-prefix /tmp/legacy --patch-tokenizer-type AlphaTokenizer \
  --load examples/alpha/tokenizer_v5 --workers 1 --partitions 1 --append-eod

# New (under test)
python <new_script.py> --input /tmp/sample.jsonl --output-prefix /tmp/new ...

cmp /tmp/legacy_text_document.bin /tmp/new_text_document.bin && echo BIN_OK
cmp /tmp/legacy_text_document.idx /tmp/new_text_document.idx && echo IDX_OK
```

BIN_OK + IDX_OK 둘 다 통과해야 production run. 영어/한국어/Arabic/CJK 다양한 스크립트 cover.

### Throughput rule of thumb (Intel Xeon 8480+, 224 logical cores, 2 TB RAM, v5 tokenizer)

| Config | Cores busy | Aggregate throughput |
|---|---|---|
| `preprocess_data_megatron.py` (per-doc imap) 128w × 8p | 128 | **1.7M tok/s** ← 옛 방식 |
| 1 proc × 1 Rayon thread | 1 | 0.94M tok/s |
| 1 proc × 8 Rayon threads | 8 | **4.7M tok/s** ← single-proc sweet spot |
| 1 proc × 192 Rayon threads | 192 | 2.2M tok/s ← contention |
| 4 procs × 8 threads | 32 | 10.6M tok/s |
| 8 procs × 8 threads | 64 | 18.0M tok/s |
| 16 procs × 8 threads | 128 | 23.8M tok/s |
| **28 procs × 8 threads** | 224 | **32.4M tok/s** ← best measured |
| 4 procs × 56 threads | 224 | 8.3M tok/s ← thread-heavy 나쁜 예 |

**예산 (보수적)**: **25M tok/s aggregate**. 계획 단계에서 `T` total tokens 예상 wallclock = `T / 25M / 3600` hours.

**STOP rule**: 예산 대비 실측이 >2× 느리면 architecture 잘못된 것. 1+ hour 작업 commit 전에 1 GB sample throughput 실측 후 재검토.

### Why this matters specifically for alpha v5
- v5 tokenizer는 `tokenizer.json` 만 ship — `vocab.json/merges.txt` 없음. HF가 `use_fast=False` 무시하고 **항상 `PreTrainedTokenizerFast` 반환**.
- 옛 `_Qwen3Tokenizer` wrapper는 `.encoder` dict 접근 가정 (slow path) → fast tokenizer로는 `AttributeError`. 이번 세션에서 `_AlphaTokenizer` wrapper 신규 추가한 이유.
- 즉 v5 → fast tokenizer 강제 → batched API (`encode_batch`) 가 진짜 lever. Per-doc API는 fast tokenizer 잠재력의 ~3%만 활용.

### Reference: optimized pipeline location
`toolkits/pretrain_data_preprocessing/fast_tokenize_v5.py` (created in this session for korean_web + FineWeb workloads — production-ready Rust encode_batch pipeline with multi-process scaling).

## Stage 1 Pre-flight Verification — Methodology (2026-05-12)

이 섹션은 multi-month run 전 데이터 + 토크나이저 + config 정합성을 끝까지 점검하는 **재사용 가능한 protocol**입니다. Stage 2/3에서 데이터 패치하거나 새 baseline 만들 때 같은 phase 구조를 그대로 활용 가능. 실제 실행 산출물은 `tests/preflight_stage1/`에 보존되어 있음.

### 왜 이런 protocol이 필요한가

Multi-month 학습은 silent failure가 가장 위험. 학습 중반에 발견된 토크나이저/데이터 bug는 전체 run 폐기로 이어짐. 79시간 DCLM tokenization이 끝나고도 **EOD가 stream에 없는 것을 미리 잡지 못했다면 학습 시작 후 cross-doc 잡음을 학습하다 한참 후 깨달았을 것**. 자동화된 verification net을 한 번 짜두면 같은 사고 재발 시 즉시 차단.

### 6-Phase 구조

각 phase가 standalone runnable script + markdown artifact + (선택) JSON report 페어로 구성. 모두 `tests/preflight_stage1/` 아래.

| Phase | 목적 | 산출물 |
|---|---|---|
| **0** | Bug fix + config 정합 (tokenizer 파일 3종 + alpha_config.py + 데이터 remap) | `00_eod_bug_diagnosis.md`, `01_eod_repro.md`, `02_audit_grep.md`, `03_eod_regression_tests.md` |
| **A** | Tokenizer round-trip + frontier deviation matrix | `A_tokenizer.md`, `A_roundtrip_report.json`, `run_phase_a.py` |
| **B** | `.idx`/`.bin` 구조적 audit (header, dtype, size consistency, token ID range, doc-boundary EOD, empty docs) | `B_dataset_integrity.md`, `B_<src>_report.json`, `run_phase_b.py` |
| **C** | Decoded sample 사람-눈 sanity check | `C_decoded_samples.md`, `C_decoded_samples.txt`, `run_phase_c.py` |
| **C-loader** | 실제 `GPTDataset.__getitem__` 흐름을 ON/OFF differential로 검증 (reset flags가 정말로 작동하는지) | `C_loader_audit.md`, `C_loader_report.json`, `run_phase_c_loader.py` |
| **D** | Training-time data flow (단일 packed sample의 tokens/position_ids/loss_mask/attention_mask 4중 snapshot) | `D_dataflow.md`, `D_sample_snapshot.txt`, `run_phase_d.py` |
| **E** | 100-iter smoke (model + optimizer + multi-GPU 통합 동작 확인) | `run_phase_e_smoke.sh` |
| **F** | Decisions log — 모든 deviation을 intentional/will-fix/accepted로 분류 | `F_decisions.md` |

### Phase C-loader가 가장 가치 있는 도구 — Differential ON/OFF 검증

이 한 가지 패턴이 verification net의 핵심 invention:

```python
# 같은 데이터를 두 번 로드하되 reset flags만 토글
ds_on  = GPTDataset(..., config=GPTDatasetConfig(reset_position_ids=True, ...))
ds_off = GPTDataset(..., config=GPTDatasetConfig(reset_position_ids=False, ...))
# 100 samples × 4 metrics × 2 configs 비교
```

만약 flag가 silent하게 무시되고 있다면 (예: 코드 회귀, 데이터 EOD 누락 등) ON/OFF가 같은 결과를 냄. 만약 진짜 작동한다면 다음 4가지 invariant이 ON/OFF 간에 명확히 다름:

| Metric | ON 기대값 | OFF 기대값 | 의미 |
|---|---|---|---|
| Cross-doc attention block rate | 1.0 | 0.0 | `reset-attention-mask` 작동 |
| Max position_id 평균 | < seq_len (~doc 길이) | seq_len - 1 (4095) | `reset-position-ids` 작동 |
| Loss mask coverage | < 1.0 (1 - eod_density) | 1.0 | `eod-mask-loss` 작동 |
| EOD count per sample | 둘 다 동일 | (data identity 확인) |

수치적 일치까지 검증 가능: loss mask coverage drop = EOD count / seq_len이 수학적으로 일치해야 함.

### Verification가 잡은 silent failure 종류 (이번 세션 실측)

- **Type 1 — Designation drift**: `tokenizer_config.json:eos_token`이 chat-only 토큰으로 잘못 지정. HF는 lucky하게 우선 처리하지만 `special_tokens_map.json`이 stale → 일부 도구만 silent하게 wrong 동작.
- **Type 2 — Stale hardcoded defaults**: HF config 생성기에 다른 모델 family의 default token IDs 박혀있음 (alpha_config.py Qwen3 IDs). 학습 영향 0이지만 inference deployment 시점에 silent breakage.
- **Type 3 — Empirical exploration mistake**: 사람 (또는 LLM agent) 의 sampling-based empirical check이 wrong 결론에 도달 ("0/100 docs end in EOD" 라고 잘못 보고). Differential / 다중 seed 재검증이 정정.
- **Type 4 — Data ↔ runtime mismatch**: 데이터엔 EOD 있고 runtime flag도 ON인데 `tokenizer.eod`가 다른 ID를 반환 → flag는 silent하게 no-op. Phase C-loader의 ON/OFF differential이 즉시 잡음.

### 새 데이터셋에 protocol 적용하는 법 (Stage 2/3 사용 예상)

```bash
# 1. 데이터 패치 (필요 시)
python toolkits/pretrain_data_preprocessing/remap_eod.py \
  --prefix /path/to/stage2_dataset/data_text_document --dry-run    # 먼저 dry
python toolkits/pretrain_data_preprocessing/remap_eod.py \
  --prefix /path/to/stage2_dataset/data_text_document              # 실제 적용

# 2. tests/preflight_stage1/run_phase_b.py 의 DATA dict 갱신 + 실행
# 3. tests/preflight_stage1/run_phase_c_loader.py 갱신 + 실행
# 4. 새 stage용 training preset yaml 생성 (stage1.yaml 기반)
# 5. Phase E smoke launch
```

회귀 테스트 `tests/test_alpha_tokenizer_eod.py` (9개) 는 CI에 통합하면 EOD designation drift / Qwen3 default 회귀 / preprocess 패스 회귀를 영구적으로 차단.

## Stage 2 v5 Re-tokenization (2026-06-26, 완료 ✅)

레거시 Stage-2 blend(`configs/data/arxive/stage2.yaml`의 9-dataset, 옛 Qwen3 vocab
151,936)를 alpha v5 tokenizer(163,968)로 **전부 재토크나이징**. Stage 1과 동일한
fast-path 파이프라인을 재사용했고, 결과는 **1.686T tokens / 10 datasets**
(레거시 9개 + FineWeb2-HQ 2nd half — stage1/stage2 complementary-halves 설계상 추가).

### 결과 & blend config

산출물: `/home/work/Datasets/LL_preprocessed/v5/stage2/<name>/data_text_document.{bin,idx}`.
Blend: `configs/data/stage2_v5_blend.yaml` (weight-less → Megatron이 `.idx` 크기 비례
auto-mix; 실측 토큰 비율은 YAML 주석에 기록).

| 데이터셋 | tokens | 비중 | 출처 |
|---|---|---|---|
| nemotron_cc_hq/actual | 530.2B | 31.6% | MinIO 복원 |
| nemotron_cc_hq/qa_pairs | 475.7B | 28.3% | MinIO 복원 → decompress |
| code/question_answering | 241.1B | 14.4% | 로컬 jsonl |
| math | 205.0B | 12.2% | 로컬 (Nemotron-CC-Math-v1) |
| code/code_review | 77.1B | 4.6% | 로컬 |
| code/rewriting | 77.2B | 4.6% | 로컬 |
| code/transpilation | 29.3B | 1.7% | 로컬 |
| code/student_teacher | 25.8B | 1.5% | 로컬 |
| korean_web | 19.0B | 1.1% | 재사용(remap 3→0) |
| fineweb2hq | 5.7B | 0.3% | 재사용 2nd half(remap 3→0) |
| **합계** | **1.686T** | CC-HQ 59.7% / Code 26.7% / Math 12.2% / Korean 1.1% / FineWeb 0.3% | |

학습 시작 (**채택된 기본 방식** — 자연 blend + budget 제한):
```bash
bash train.sh baseline_48L <stage2_training_preset> stage2_v5_blend --train-samples 170898438
```
(training preset은 `stage2_2`/`stage2_3` 등 별도 선택 — 이 데이터 작업 범위 밖.)

**학습량(token budget) 제어**: 코퍼스 1.680T 전부 돌릴 필요 없이 `--train-samples`로
원하는 토큰만 학습. seq_length 4096 → `train_samples = tokens / 4096` (0.5T→122.07M,
0.6T→146.48M, **0.7T→170,898,438**). **iter 직접 지정보다 `--train-samples` 권장** —
step-wise GBS 스케줄에서 iter당 작업량이 달라져 "iter=토큰"이 깨지므로(GBS-invariant).
weight-less 자연 blend @ 0.7T는 **모든 소스 0.42 epoch**(반복 0, 코퍼스가 budget의
2.4배) → CC-HQ 60%/Code 27%/Math 12%/Korean 1% 비율 그대로 (CC-HQ 419B / Code 188B
/ Math 85B / Korean 7.9B). 이 비율이 적절하면 reweighting 불필요.

### (옵션) Blend 비율 조정 — CC-HQ down-weight

작은 budget에서 curated(math/code/korean) 비중을 자연비율 이상으로 키우고 싶을 때만.
data-path 각 경로 앞에 명시적 weight를 붙이면 크기와 무관하게 샘플링 비율 지정 가능
(Megatron normalize; **런타임 검증으로 CC-HQ 40% 반영 확인됨** — weights honored).

**도구**: `tools/compute_blend_weights.py` — CC-HQ를 목표치로 cap하고 남는 몫을
math/code/korean의 **자연 상대비율대로 재분배**(세 카테고리 내부 믹스 유지 + 동일
epoch). 토큰 수는 `bin_bytes/4`로 즉시 산출(GPU 불필요).
```bash
python tools/compute_blend_weights.py --cap-cchq 40 --budget-t 0.6 \
  --write configs/data/stage2_v5_blend_cc40.yaml
```
산출물 예 `configs/data/stage2_v5_blend_cc40.yaml` (옵션, CC-HQ 40% cap, 0.6T 기준):

| 카테고리 | share | tokens@0.6T | epoch |
|---|---|---|---|
| CC-HQ | 40.0% | 240.0B | 0.24 |
| Code | 40.1% | 240.4B | 0.53 |
| Math | 18.2% | 109.4B | 0.53 |
| Korean | 1.7% | 10.2B | 0.53 |

`★ 핵심 트레이드오프 — over-epoch`: 명시적 weight는 크기와 sampling을 분리하므로,
**작은 데이터셋을 자연 share 이상으로 키우면 epoch>1(반복)** 됨. "CC-HQ cap +
비례 재분배" 방식은 어느 카테고리도 자연 share를 *초과*하지 않아 0.6T에서 셋 다
0.53 epoch(<1)로 안전. 반대로 flat 비율(예: Korean 5% 고정)은 Korean(19B뿐)을
1.5~2.2 epoch로 반복시킴 — `compute_blend_weights.py`의 epoch 표로 항상 확인할 것.
학습: `bash train.sh baseline_48L <preset> stage2_v5_blend_cc40`.

### Tooling (전부 `toolkits/pretrain_data_preprocessing/`)

레거시 `preprocess_kormo_*.sh`/`preprocess_nemotron_*.sh`는 **재사용 불가**(stale:
`--patch-tokenizer-type Qwen2Tokenizer`가 fast-only v5에서 크래시, 느린 per-doc 경로,
옛 250624 backend). 대신 다음을 신규 작성/확장:

| 파일 | 역할 |
|---|---|
| `run_stage2_v5.sh` | 정규 레시피. sub-targets: `restore` / `local` / `cchq` / `all` / `<dataset>` |
| `preprocess_stage2_v5.sh` | 범용 multi-shard FAST 드라이버. 파일 round-robin → P개 `fast_tokenize_v5.py` 병렬 → `merge_indices.py --dtype int32`. 기본 `PROCS=12` ×8 rayon ≈ 110-core cgroup의 96. `AUTO_CLEAN_PARTS=1`로 merge 검증 후 parts 자동 삭제 |
| `fast_tokenize_v5.py` | `--input-list-file` 추가(멀티 jsonl 파일 → 1 part). parity-preserving |
| `minio_restore.py` | boto3 다운로더(`preprocess_data_megatron.py`엔 없던 restore). `/home/work/Datasets/LL_datasets/minio_backup.py`를 import해 creds + `_patch_botocore_time` 재사용. **MinIO 서버 clock ~9h skew → time-patch 필수**(없으면 SignatureDoesNotMatch). resumable(size-match skip + `.part` atomic rename) |
| `tests/preflight_stage2/run_phase_b.py` | Stage-1 Phase B 적응판. **B4가 ~100% doc-end=id 0을 assert**(Stage 1은 pre-injection이라 0을 기대 — 정반대). per-dataset try/except로 non-halting |

### 핵심 발견 (재현 주의)

1. **재사용 v5 데이터는 EOD가 id 3.** `korean_web`·`fineweb2hq`는 2026-05-12 designation
   수정 *전*(eos=`<|im_end|>` id 3)에 토크나이즈돼 doc 끝이 id 3. Stage 1 데이터는
   remap됐지만 이 둘은 안 됨. Phase B가 즉시 검출 → **둘 다 `remap_eod.py --old-eod 3
   --new-eod 0`로 수정 완료**(dry-run이 200k doc-end 전부 id 3 보유를 pre-verify; post-verify
   200k 전부 id 0 + first/last 100 확인). 신규 토크나이즈 데이터(`--append-eod`)는 id 0
   직접 부여 → remap 불필요(Stage 1과 달라진 점). **교훈**: 다른 stage/session에서
   토크나이즈된 v5 `.bin`을 blend에 넣기 전 반드시 EOD(id 0 vs 3)를 Phase B로 확인할 것.

2. **smoke test로는 이 blend를 검증할 수 없음.** `configs/{model,training}/smoke.yaml`은
   toy preset이고 251125 backend와 **기존 config drift**가 있음: ① GBS=1이 multi-GPU에서
   `÷ data_parallel_size` 안 나눠짐, ② `moe-router-topk:1`이 `--moe-router-pre-softmax`
   요구. 둘 다 **dataset 로드 *전* model-build에서 크래시**라 blend와 무관. 데이터 정합성은
   `BlendedMegatronDataset`를 직접 빌드해 검증함(CPU-only):
   - `torch.distributed.init_process_group("gloo", rank=0, world_size=1)` 필요
   - tokenizer stub에 `.eod`(=0) + `.unique_identifiers` property 필요(후자는 cache-key
     JSON 직렬화용)
   - 검출 항목: blend auto-weight(= 실측 비율 일치), 샘플 token range <163968,
     EOD 기반 `reset-position-ids`/`eod-mask-loss` 작동(maxpos<4095, loss_mask cov<1.0)
   - 결과: 9-source 빌드 + 샘플 조립 OK. 실제 학습은 `baseline_48L`이라 smoke drift 무영향.

### 디스크

검증 완료 후 회수: CC-HQ raw `.jsonl`(actual 2.5T + qa_pairs 2.3T), 모든 `_parts`(~2.4T),
qa_pairs `.jsonl.zstd`(814G) — **~8TB**. CC-HQ는 MinIO에서 `run_stage2_v5.sh restore`로
재복원 가능(~3h). **옛 Qwen3 `/home/work/Datasets/LL_preprocessed/mmap/`(8.4T)는
다른 사용자 소유 — 삭제 금지.** 로컬 raw math/code는 MinIO 백업이 없어 보존.

## Best-fit Packing — 문서 truncation 최소화 (2026-06-30, 구현 ✅)

논문 **"Fewer Truncations Improve Language Modeling"** (arXiv 2404.10830, ICML 2024)의
**Best-fit Packing (BFP)** 를 offline 전처리로 구현. Megatron-LM은 이 기능을 제공하지 않음
(기본 `GPTDataset`은 concat-and-chunk로 문서를 seq_length 경계에서 무차별 절단).

### 문제 & 해법
- **문제**: 학습 시 `GPTDataset`이 모든 문서를 이어붙여 4096씩 자름 → 경계에 걸친 문서가
  매 sample마다 잘림. (alpha의 `--reset-attention-mask`는 *한 sequence 안*의 cross-doc
  attention만 막을 뿐 절단 자체는 못 막음 — BFP와 **직교/상보**.)
- **해법**: 문서 길이 배열만 읽어 **Best-Fit-Decreasing**(segment tree, O(N log L))로 문서를
  4096 bin에 통째로 packing → 각 bin을 EOD(id 0)로 4096에 padding → **bin 1개 = .idx 문서
  1개**로 재출력. 학습의 concat-and-chunk가 정확히 bin 경계에서 잘려 문서가 안 잘림
  (4096 초과 문서만 불가피하게 분할 — 논문도 동일).

### 사용
```bash
# stats만 (truncation 감소/fill ratio 확인, 쓰기 없음)
bash toolkits/pretrain_data_preprocessing/run_stage2_v5.sh pack-dry
# 전체 10개 데이터셋 packing → /home/work/Datasets/LL_preprocessed/v5/stage2_packed/
bash toolkits/pretrain_data_preprocessing/run_stage2_v5.sh pack
# 1개만: bash ... pack code/transpilation   (또는 직접 bestfit_pack.py --input … --output … --dry-run)
# 학습: data preset을 packed blend으로
bash train.sh baseline_48L <preset> stage2_v5_blend_packed --train-samples <N>
```

### 실측 결과 — 전체 10개 완료 (2026-07-02 ✅, `stage2_packed/`)
| 데이터셋 | fill | 감소 | 크기 |
|---|---|---|---|
| fineweb2hq | 99.95% | −72.8% | 22 G |
| korean_web | 99.99% | −78.5% | 72 G |
| code/transpilation | 99.01% | −100% | 111 G |
| code/student_teacher | 99.05% | −100% | 98 G |
| code/code_review | 99.95% | −99.5% | 289 G |
| code/rewriting | 99.78% | −100% | 290 G |
| math | 99.69% | −82.5% | 770 G |
| code/question_answering | 99.08% | −100% | 911 G |
| nemotron_cc_hq/actual | 100.00% | −81.0% | 2.0 T |
| nemotron_cc_hq/qa_pairs | 99.81% | −100% | 1.8 T |
| **합계** | **전부 ≥99%** | code −100% / web·math·CC-HQ-actual −73~82% (긴 문서 잔존) | **6.2 TiB** |

각 셋 자체 post-verify(전부 4096 / 토큰 보존 / round-trip 20/20) 통과 + 두 CC-HQ는 독립
재로드로 bins·토큰 재확인(actual 129,442,937 / qa 116,351,363 bins). 학습: `train.sh
baseline_48L <preset> stage2_v5_blend_packed --train-samples N`.

**Loader differential** (실제 `GPTDataset` 경로, fineweb2hq 3000 samples,
`tests/preflight_stage2/run_pack_loader_check.py`): **bad-truncation
(whole docs + 잘린 small doc) 82.0% → 0.0%**. ends-on-doc-boundary 0.03% → 72.6%
(나머지 27.4%는 4096 초과 문서의 *순수 단일-문서 chunk* — fragmentation 아님).

### emit 성능 — threaded prefetch (`--emit-threads`, 기본 48)
emit 병목은 소스 `.bin`에서 bin 멤버 문서를 **무작위로 읽는 NFS latency**(대역폭 아님).
`os.pread`(syscall이 GIL 해제)로 read 48개 동시 발행해 latency를 숨기고, 쓰기는 메인
스레드에서 순서대로 → **출력 byte-identical**(회귀 테스트로 잠금). 실측: cchq_actual
**8,183 bins/s = serial(~1,000) 대비 ~8.2×**(단독). 병렬 2개 동시엔 합계 쓰기 ~132 MB/s로
수렴(그 시점 병목 = NFS write 대역폭). **대형 셋 필수** — 없으면 CC-HQ가 각 ~30h+, 있으면
두 개 병렬로 ~반나절.

### 핵심 설계 (정합성, helpers.cpp/gpt_dataset.py 소스 대조 검증)
- **bin capacity = seq_length = 4096 (L+1 아님).** Megatron이 sample당 4097 토큰을 읽어
  "+1"을 스스로 공급(공유 경계). 4097로 packing하면 정렬이 깨짐.
- **`add_document(bin_arr, [L])` — 길이 리스트는 단일 원소 `[L]`.** GPTDataset은 *sequence*
  단위로 자르고 `document_indices`를 무시 → 문서별 길이를 넣으면 packing이 조용히 무효화.
  (tool에 hard assert.)
- **pad = EOD(id 0).** `--eod-mask-loss`가 pad 구간 loss를 masking, `--reset-attention-mask`가
  pad를 격리. 별도 pad id는 masking 안 됨(모델이 pad 예측 학습) → EOD-pad가 유일 정답.
- **데이터셋별로** 실행. Megatron의 `BlendedDataset`은 한 constituent에서 *통째* sequence를
  샘플 → per-dataset packing이 blend로 보존. (cross-dataset packing은 blend 비율 오염.)
- **놓치기 쉬운 점**: 4096 초과 문서의 full-L head chunk는 content로 끝나 1토큰 eod-unmasked
  leak(fineweb2hq ~0.0066% positions, 논문 수용 범위). `--strict-eod`로 제거 가능(대신 mid-doc
  false EOD 도입 — 기본은 accept+report). 또 weight-less blend는 dataset별 padding(<1%) 차이로
  실토큰 비율이 sub-% drift — 정확히 맞추려면 `compute_blend_weights.py`로 명시 weight.

### 도구
| 파일 | 역할 |
|---|---|
| `toolkits/pretrain_data_preprocessing/bestfit_pack.py` | BFP packer (segment-tree BFD + pad-to-L emit + pre/post-verify + round-trip + `--dry-run`/`--strict-eod`). BFD ~0.25M docs/s(pure Python); 대형 셋은 emit가 I/O bound |
| `run_stage2_v5.sh` `pack`/`pack-dry` sub-target | 10개 blend member 일괄 packing (env `SEQLEN`/`EOD`/`OUT_PACKED`) |
| `examples/alpha/configs/data/stage2_v5_blend_packed.yaml` | packed blend (stage2_v5_blend의 packed 트리 미러) |
| `tests/test_bestfit_pack.py` | 16 unit tests (segtree↔brute, BFD↔naive parity, piece coverage, baseline 추정, 실 IndexedDataset round-trip) |
| `tests/preflight_stage2/run_pack_loader_check.py` | packed vs unpacked GPTDataset loader differential |

## Related Documentation

- **Architecture**: `docs/ARCHITECTURE.md`
- **Conversion**: `docs/CONVERSION.md`
- **Muon Optimizer**: `docs/MUON.md`
- **Setup**: `docs/SETUP.md`
- **Evaluation**: `docs/EVALUATION.md`

## Muon Optimizer Quick Reference

Alpha uses `dist_muon` (LayerWise distributed Muon) for faster convergence:

```yaml
optimizer: dist_muon
muon_momentum: 0.95
muon_num_ns_steps: 5
muon_scale_mode: spectral
```

**Compatibility**: TP=1 only, NOT compatible with CPU optimizer offloading.

### QGKV Split (auto-enabled, fixed 2026-05)

Alpha's Gated Attention uses `linear_qgkv.weight` (4-way fused: Q, **Gate**, K, V).
Newton-Schulz now runs on each sub-projection independently; previously it
silently ran on the whole fused matrix. **No config change needed** — the fix
activates automatically for any run with `optimizer: dist_muon` (and without
`--muon-no-split-qkv`).

- Look for `Muon QKV matcher: 3-way=0, 4-way=N, attention-like 2D weights total=N`
  in startup logs to confirm the 4-way path is active. `4-way == 0` while
  `total > 0` means a config issue — check the WARNING log.
- **Existing checkpoints**: trained under the old (whole-matrix) NS regime.
  Resuming with this fix changes optimizer dynamics — K/V updates will be
  larger relative to Q/Gate. Prefer to adopt at a stage boundary.
- See root `CLAUDE.md` § "Custom Training Features (Alpha Stage1)" #3 and
  `megatron_patch/CLAUDE.md` § "QKV / QGKV Split..." for mechanism details.
