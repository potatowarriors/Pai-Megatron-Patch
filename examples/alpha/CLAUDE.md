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
| MoE routed experts × FFN | **184 × 512** |
| MoE shared expert × FFN | 1 × 512 |
| MoE topk | 8 |
| MoE group routing | 8 groups × top-4 (4×23 = 92 candidates) |
| MoE topk scaling factor | 2.5 |
| MoE score function | sigmoid (모든 stage 통일) |
| MoE balancing | Stage 1: `none` + expert bias / Stage 2+: `seq_aux_loss` + expert bias (1e-4 coeff) |
| Vocab (effective / padded) | 163,860 / **163,968** |
| Tokenizer | `examples/alpha/tokenizer_v5/` (alpha 전용 BBPE) |
| EOS / pre-training EOD | `<\|endoftext\|>` (id 0) — `<\|im_end\|>` (id 3)는 SFT 단계 chat turn 전용 |
| Document boundary handling | `--reset-position-ids` + `--reset-attention-mask` + `--eod-mask-loss` 전부 ON |
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
| `-` | MoE MLP | Mixture-of-Experts FFN (**184 routed experts + 1 shared, top-8, FFN 512**) |
| `D` | Dense MLP | Standard SwiGLU FFN (현재 0 layer; 향후 DSV3-style 도입 시 재활성화 가능) |

### Layer Mapping (2:1)
- **Megatron**: 48 layers (each = 1 pattern token)
- **HuggingFace**: 24 layers (MG layer i → HF layer i/2)

## Key Constraints

| Constraint | Value | Reason |
|------------|-------|--------|
| TP (Tensor Parallel) | **1** | GatedDeltaNet (Megatron `MambaLayer` 슬롯) don't support TP > 1 |
| EP (Expert Parallel) | 8 | **184 experts / 8 GPUs = 23 experts per GPU** |
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

### Checkpoint Conversion (MG → HF)
```bash
cd toolkits/distributed_checkpoints_convertor
bash scripts/alpha/run_8xH20.sh baseline_48L /path/to/mcore /path/to/hf true true bf16
# Auto mode: use 'auto' as checkpoint path to convert latest
```

### Validation
```bash
bash validate.sh /path/to/mg /path/to/hf baseline_48L
```

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
├── training/stage2_2.yaml           # Stage 2-2 cosine (DSV3 routing: seq_aux_loss + sigmoid + bias)
├── training/stage2_3.yaml           # Stage 2-3 (4× LR, DSV3 routing)
├── training/smoke.yaml              # 2-iter, no-Muon smoke
├── data/stage1_v5_korean_web.yaml   # ★ v5-tokenized Korean web (144GB, current Stage 1)
├── data/kormo_1pct.yaml             # legacy: pre-tokenized with Qwen3 tokenizer (vocab 151,936)
├── data/kormo_50pct.yaml            # legacy: pre-tokenized with Qwen3 tokenizer (vocab 151,936)
├── data/kormo_code_balanced.yaml    # legacy
├── data/stage2.yaml                 # legacy: Stage 2 9-dataset blend (Qwen3 tokenizer)
├── data/stage2_2.yaml               # legacy: same blend (Qwen3 tokenizer)
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
num-experts: 184                                            # 8-multiple, 23/GPU at EP=8 (was 128)
moe-router-topk: 8
moe-ffn-hidden-size: 512                                    # was 768
moe-shared-expert-intermediate-size: 512                    # was 768

# DSV3 MoE routing (group-limited + topk scaling)
moe-router-num-groups: 8
moe-router-group-topk: 4                                    # 4×23=92 candidate experts
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
| `validate.sh` | MG↔HF weight validation wrapper |
| `validate_mg_hf_full.py` | Comprehensive weight validation |
| `tools/alpha_config.py` | Config inspection (validate, generate HF config) — defaults updated to vocab 163,968 |
| `calculate_parameters.py` | Parameter count tool — accepts flat YAML; reports 15.08B for current baseline |
| `tokenizer_v5/` | **Alpha 전용 v5 tokenizer** (5 files, 12.6MB; HF `PreTrainedTokenizerFast`, vocab 163,860) |
| `hf_model/` | HuggingFace model implementation |
| `sglang/deploy.sh` | SGLang deployment script (Option A/B, uses local backend) |
| `sglang/convert_config_for_sglang.py` | Alpha→Qwen3-Next config converter (head_dim 256 / vocab 163,968 호환성 검증 필요) |
| `sglang/sglang_alpha_model.py` | SGLang model adapter (mlp_only_layers support) |
| `../../backends/sglang/setup.sh` | SGLang backend setup (patch + adapter install) |
| `scripts/setup_wandb.sh` | Sourced by train.sh to set `WANDB_API_KEY` (smoke/mock 시 auto-override됨) |

## Troubleshooting

### Common Errors

| Error | Solution |
|-------|----------|
| `assert self.tp_size == 1` | Set `--tensor-model-parallel-size 1` |
| `Pattern length mismatch` | Pattern must have exactly `num_layers` characters |
| `Invalid characters in pattern` | Only use M, *, -, D |
| `Attention ratio mismatch` | Count of `*` should match `hybrid_attention_ratio × num_layers` |
| `MambaLayer has no attribute self_attention` | QK-Clip bug with hybrid models — fixed in `pretrain_alpha.py` via monkey-patch |

### Environment Issues
```bash
bash scripts/validate_environment.sh  # Check CUDA, Flash Attn 3, TE version
```

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
| `num_experts` | 512 | **184** | `baseline_48L.yaml::num-experts` |
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
