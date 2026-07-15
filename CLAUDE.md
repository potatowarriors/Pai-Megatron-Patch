# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Pai-Megatron-Patch is a production-grade deep learning training toolkit for Large Language Models (LLMs) and Vision Language Models (VLMs) using NVIDIA's Megatron framework. It bridges high-level model definitions (HuggingFace) with high-performance distributed training (Megatron-LM/Megatron-Core).

**Core Design Philosophy**: Non-invasive patching. Functions are provided as patches rather than modifying Megatron-LM source code, allowing users to stay current with Megatron-LM updates.

## Module-Specific Guides

| Module | CLAUDE.md Location | Description |
|--------|-------------------|-------------|
| **Alpha Model** | [`examples/alpha/CLAUDE.md`](examples/alpha/CLAUDE.md) | GatedDeltaNet hybrid architecture, training, validation |
| **Checkpoint Converter** | [`toolkits/distributed_checkpoints_convertor/CLAUDE.md`](toolkits/distributed_checkpoints_convertor/CLAUDE.md) | HF↔Megatron conversion |
| **Megatron Patch** | [`megatron_patch/CLAUDE.md`](megatron_patch/CLAUDE.md) | Core library, Muon optimizer |

## Architecture

### Directory Structure

- **`megatron_patch/`**: Core library (model/, data/, ssm/, training.py, arguments.py)
- **`examples/`**: Model-specific training scripts (run_mcore_*.sh)
- **`toolkits/`**: Utilities (checkpoint converters, data preprocessing)
- **`backends/`**: Git submodules (Megatron-LM versions, ChatLearn, verl)

### Training Pipeline Flow

```
examples/{model}/run_mcore_*.sh
  ↓
megatron_patch/arguments.py → initialize.py → training.py::pretrain()
  ↓
backends/megatron/Megatron-LM-*/megatron/core/ (Distributed execution)
```

## Common Commands

### Environment Setup
```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
export PYTHONPATH=/path/to/Pai-Megatron-Patch:/path/to/backends/megatron/Megatron-LM-250908:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
```

### Data Preprocessing
```bash
# Pre-training
cd toolkits/pretrain_data_preprocessing/
bash run_make_pretraining_dataset.sh <vocab_file> <input_jsonl> <output_prefix> <workers>

# SFT
cd toolkits/sft_data_preprocessing/
bash convert_sft_dataset.sh <input_jsonl> <output_dir> <tokenizer_path>
```

### Checkpoint Conversion
```bash
# HuggingFace → Megatron
cd toolkits/model_checkpoints_convertor/{model}/
bash hf2mcore_*.sh <model_size> <hf_dir> <output_dir> <tp> <pp>

# Megatron → HuggingFace
bash mcore2hf_*.sh <model_size> <megatron_dir> <hf_output_dir> <tp> <pp>
```

### Training
```bash
cd examples/{model}/
bash run_mcore_{model}.sh <ENV> <MODEL_SIZE> <BATCH_SIZE> <GLOBAL_BATCH_SIZE> <LR> ...
```

## Key Technical Details

### Parallelism Strategy
| Type | Description | When to Use |
|------|-------------|-------------|
| TP (Tensor) | Splits weights across GPUs | Large models (TP=4 or 8) |
| PP (Pipeline) | Splits layers across GPUs | Memory constraints |
| EP (Expert) | Splits MoE experts | MoE models |
| CP (Context) | Splits sequences | Ultra-long contexts (>32K) |

**Rule of thumb**: 8×GPU with 70B model → TP=8, PP=1

### Checkpoint Formats
- **Legacy**: `model_optim_rng.pt` files
- **torch_dist**: Distributed sharded (recommended for 100B+)
- **HuggingFace**: `.safetensors` or `.bin`

### Optimizers
- **Adam/AdamW**: Standard, supports all parallelism
- **Muon** (`dist_muon`): ~2x faster convergence, requires TP support, NO CPU offloading
  - See [`megatron_patch/CLAUDE.md`](megatron_patch/CLAUDE.md) for details

## Model-Specific Notes

### Qwen Models
- Use `NullTokenizer`, GQA support, RoPe theta: 1000000

### MoE Models (DeepSeek-V3, Mixtral)
- Require `--moe-grouped-gemm`, set EP and ETP

### Hybrid SSM (Qwen3-Next, Alpha)
- **TP=1 required** (Mamba layers don't support TP > 1)
- See [`examples/alpha/CLAUDE.md`](examples/alpha/CLAUDE.md) for Alpha-specific guide

### LLaMA Models
- Use `SentencePieceTokenizer`, RoPe theta: 500000

## Custom Training Features (Alpha Stage1)

Three non-upstream features added on Megatron-LM-251125 for Alpha Stage1 pre-training.
(4번째 비-upstream 기능인 **DiLoCo 2노드 학습** — IB 없는 클러스터용 저통신 분산 — 은
`megatron_patch`/submodule이 아니라 `examples/alpha/diloco_patch.py`에 살며,
[`examples/alpha/CLAUDE.md`](examples/alpha/CLAUDE.md) § Multi-Node Training 참조.)

### 1. Step-wise Global Batch Size Schedule

Backport of upstream main's `--step-batch-size-schedule`. Lets GBS step up at token thresholds during training (more flexible than the linear `--rampup-batch-size`).

**CLI**:
```bash
--step-batch-size-schedule "0:768 250B:1536 500B:3072 750B:6144"
```

- Format: `T0:BS0 T1:BS1 ...`. Suffixes K/M/B/T (1e3/1e6/1e9/1e12) supported.
- Thresholds are **tokens** (converted to samples via `--seq-length`). First threshold must be `0`.
- `--global-batch-size` is auto-derived as the max BS in the schedule. Setting it explicitly to anything other than that max is a hard error.
- **Mutually exclusive** with `--rampup-batch-size`.
- Sample-based training only (`--train-samples`, not `--train-iters`).

**Implementation**:
- `backends/megatron/Megatron-LM-251125/megatron/core/num_microbatches_calculator.py` — `parse_step_batch_size_schedule()`, `StepBatchSizeScheduleCalculator`
- `backends/megatron/Megatron-LM-251125/megatron/training/arguments.py` — argparse + validation
- `backends/megatron/Megatron-LM-251125/megatron/training/training.py::update_train_iters` — segment-wise iteration count
- Note: this is one of the rare cases where `backends/megatron/Megatron-LM-*/` is patched directly (the num-microbatches calculator is core infrastructure not exposed for patching from `megatron_patch/`).

### 2. Progressive Auxiliary Dataset Blending

Smoothly ramp up an arbitrary number of auxiliary datasets (e.g. code, math, multi-lingual) on top of a static base blend. Each aux has its own start/end thresholds and start/final ratios. Lives entirely in `megatron_patch/data/` — Megatron core is untouched for the dataset side.

**CLI**:
```bash
--progressive-blend-config /path/to/blend.yaml
# Optional: tag every sample with its source dataset for diagnostic logging
--progressive-blend-emit-source
```

- **Mutually exclusive** with `--data-path`, `--data-args-path`, `--train-data-path`, etc.
- The base blend is used for validation/test splits. Progressive ramping applies to train only.

**YAML config format**:
```yaml
base:
  data_path: ["0.7", "/path/dclm", "0.3", "/path/korean_web"]

# Default unit for all aux schedules (overridable per-aux). One of: tokens|samples|iterations.
schedule_unit: tokens   # default if omitted

auxiliary:
  - name: code
    data_path: ["1.0", "/path/code_corpus"]
    schedule:
      start_at: "100B"        # 100B tokens
      reach_full_at: "500B"
      start_ratio: 0.0
      final_ratio: 0.10

  - name: math
    data_path: ["1.0", "/path/math_corpus"]
    schedule:
      unit: iterations        # override default for this aux
      start_at: 50000
      reach_full_at: 150000
      start_ratio: 0.01
      final_ratio: 0.08

  - name: multilingual
    data_path: ["0.5", "/path/lang_a", "0.5", "/path/lang_b"]
    schedule:
      unit: samples
      start_at: 0
      reach_full_at: "100M"
      start_ratio: 0.02
      final_ratio: 0.20
```

**Schedule unit semantics**:
| `unit` | Meaning | Robust against |
|---|---|---|
| `tokens` (recommended) | `sample_idx * seq_length` | seq_length and GBS changes |
| `samples` | global sample index | GBS changes (not seq_length) |
| `iterations` | training iteration | (precomputed from `--step-batch-size-schedule` or constant GBS) |

`iterations` is intuitive (matches wandb x-axis) but each iter represents different work amounts under step-wise GBS.

**Per-aux schedule semantics** (linear ramp):
- `sample_idx <= start_at` → `start_ratio`
- `start_at < sample_idx < reach_full_at` → linear interpolation
- `sample_idx >= reach_full_at` → `final_ratio` (constant thereafter)
- Constraint: `Σ aux_ratios <= 1.0` at every idx (validated at build time on a 200-point grid).

**Reproducibility**:
- Each `__getitem__(idx)` uses `np.random.default_rng(seed=[args.seed, idx])` for source selection — bit-reproducible across runs.
- Checkpoint resume: Megatron's `MegatronPretrainingSampler` advances from `consumed_train_samples`, so the same `idx` is replayed and the schedule continues seamlessly.

**Implementation**:
- `megatron_patch/data/progressive_mix_dataset.py` — `LinearRampSchedule`, `ProgressiveMixDataset`, `parse_progressive_blend_config`, `build_iter_to_samples_table`
- `megatron_patch/data/__init__.py::_build_progressive_blend_dataset` — builds base + N aux `BlendedMegatronDataset` and wraps them
- `backends/megatron/Megatron-LM-251125/megatron/training/arguments.py` — argparse + mutual-exclusion checks

### Combining Both Features

Typical Alpha Stage1 recipe:
```bash
--seq-length 4096 \
--train-samples <total> \
--step-batch-size-schedule "0:768 250B:1536 500B:3072 750B:6144" \
--progressive-blend-config configs/stage1_blend.yaml
```
GBS scales with budget; one or more auxiliary datasets ramp in at chosen thresholds independently.

### 3. Muon QGKV Split for Gated Attention

Fix to a silent bug in the upstream Muon optimizer's QKV-split path. The upstream
detector matched only `linear_qkv.weight` (standard 3-way fused projection),
which **silently failed** for Alpha's Gated Attention parameter `linear_qgkv.weight`
(4-way fused: Q, **Gate**, K, V). As a result every Alpha checkpoint trained
before this fix ran Newton-Schulz on the entire `[Q|Gate|K|V]` block as one
matrix instead of orthogonalizing each sub-projection independently.

**No CLI changes required** — the fix is automatic when `optimizer: dist_muon`
is selected and `--muon-no-split-qkv` is not passed (default).

**What the fix does**:
- Recognizes `linear_qgkv.weight` and assigns 4-way split shapes
  `[q_per_group, q_per_group, kv_channels, kv_channels]`.
- Per-parameter `qkv_split_shapes` attribute lets one optimizer instance
  serve both standard 3-way and gated 4-way attention in the same model.
- Adds an INFO log of matched/unmatched counts at startup, plus a WARNING when
  `split_qkv=True` is on but no fused QKV name pattern matches (catches future
  attention layouts like MLA before they go unnoticed).

**Implementation**:
- `backends/megatron/Megatron-LM-251125/megatron/core/optimizer/muon.py`
  — `get_megatron_muon_optimizer` (param marking + warning), `TensorParallelMuon.orthogonalize`
  (per-param shapes via `getattr`).
- Same precedent as features #1 and #2: optimizer is core infra not exposed for
  patching from `megatron_patch/`, so the submodule is edited directly.

**Behavioral note** for in-flight Alpha runs: applying this fix changes optimizer
dynamics mid-training (K/V updates were previously underweighted relative to
Q/Gate). Prefer a stage boundary for adoption.

**Out of scope**: MLA (DeepSeek-V3-style latent attention) is still untouched —
the WARNING log is what alerts you if a future MLA model is trained with Muon.

### Tests

```bash
# Step-wise GBS calculator (8 tests)
cd backends/megatron/Megatron-LM-251125 && \
  WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500 \
  python -m pytest tests/unit_tests/test_step_batch_size_schedule.py -v --noconftest

# Progressive mix dataset (13 tests)
cd <repo-root> && python -m pytest tests/test_progressive_mix_dataset.py -v

# Muon QGKV split — full suite incl. oracle per-block independence (20 tests)
cd backends/megatron/Megatron-LM-251125 && \
  NVIDIA_PYTORCH_VERSION=25.06 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
  MASTER_ADDR=localhost MASTER_PORT=29500 \
  python -m pytest tests/unit_tests/test_muon_optimizer.py -v --noconftest
```

## Debugging Tips

### Common Issues
| Issue | Solution |
|-------|----------|
| OOM | Reduce batch size, enable AC=1 or AC=2, increase PP |
| Slow training | Enable Flash Attention (FL=true), check CUDA_DEVICE_MAX_CONNECTIONS=1 |
| Checkpoint load error | Verify TP/PP match, check Megatron version compatibility |
| Data loading error | Ensure .bin/.idx exist together, check tokenizer vocab |

### Logging
- TensorBoard: `OUTPUT_BASEPATH/tensorboard/`
- Checkpoints: `OUTPUT_BASEPATH/checkpoints/`

## Version Compatibility

### Megatron-LM Versions
| Version | Status | Use Case |
|---------|--------|----------|
| **251125** | Dev | Alpha, Muon optimizer |
| **250908** | Stable | Qwen3, DeepSeek-V3 |
| **250624** | Stable | Qwen3, Moonlight |

**Version Selection**: Set PYTHONPATH in training scripts (line 6 of `run_mcore_*.sh`)

### Framework Requirements
- PyTorch: ≥2.0 (2.3+ recommended)
- Transformer Engine: ≥2.9.0 (for Muon QK-Clip)
- CUDA: 11.8+ (12.1+ for FA3)

## Development Workflow

### Adding a New Model
1. Create `megatron_patch/model/{model_name}/`
2. Add tokenizer in `megatron_patch/tokenizer/`
3. Create conversion scripts in `toolkits/model_checkpoints_convertor/{model_name}/`
4. Add training script in `examples/{model_name}/`

### Modifying Training Logic
- **DON'T** modify `backends/megatron/Megatron-LM-*/` (submodule) by default.
- **DO** modify `megatron_patch/training.py` or add patches.
- **Exception**: core infrastructure not exposed for patching (e.g. the num-microbatches calculator) may be patched directly — see [Custom Training Features](#custom-training-features-alpha-stage1) for the precedent.

## References

- Main README: [README.md](README.md)
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- Model guides: `examples/{model}/README.md`
