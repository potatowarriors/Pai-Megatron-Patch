# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Pai-Megatron-Patch is a production-grade deep learning training toolkit for Large Language Models (LLMs) and Vision Language Models (VLMs) using NVIDIA's Megatron framework. It bridges high-level model definitions (HuggingFace) with high-performance distributed training (Megatron-LM/Megatron-Core).

**Core Design Philosophy**: Non-invasive patching. Functions are provided as patches rather than modifying Megatron-LM source code, allowing users to stay current with Megatron-LM updates.

## Architecture

### Directory Structure

- **`megatron_patch/`** (225 files): Core library
  - `model/`: 32+ model architectures (Qwen, LLaMA, DeepSeek, etc.)
  - `data/`: Data loading and preprocessing
  - `generation/`: Inference and text generation
  - `training.py` (820 lines): Main training orchestrator
  - `arguments.py` (573 lines): Argument parsing and patching
  - `initialize.py`: Megatron initialization wrapper

- **`examples/`** (104 files): Model-specific training scripts
  - Each model has: `run_mcore_*.sh`, README.md, conversion scripts
  - Naming: `run_mcore_{model}.sh` for training, `run_mcore_{model}_chatlearn.sh` for RL

- **`toolkits/`**: Essential utilities
  - `model_checkpoints_convertor/`: HuggingFace ↔ Megatron conversion
  - `distributed_checkpoints_convertor/`: Sharded checkpoint conversion
  - `pretrain_data_preprocessing/`: Binary mmap format data prep
  - `sft_data_preprocessing/`: SFT data prep (JSONL → mmap)
  - `auto_configurator/`: Auto training config suggestions

- **`backends/`**: Git submodules
  - `megatron/Megatron-LM-*`: 9 versions (Megatron-LM-251125 is latest dev, Megatron-LM-250908 is latest stable)
  - `rl/ChatLearn/`: RL training framework (GRPO, GSPO, PPO)
  - `rl/verl/`: Alternative RL framework
  - `LM-Evaluation-Harness-240310/`: Benchmark evaluation

- **`verl_patch/`**: VERL integration modules

### Training Pipeline Flow

```
examples/{model}/run_mcore_*.sh
  ↓ (Sets PYTHONPATH to Megatron-LM version + megatron_patch)
megatron_patch/arguments.py (Parse & patch arguments)
  ↓
megatron_patch/initialize.py (Initialize Megatron distributed env)
  ↓
megatron_patch/training.py::pretrain() (Main orchestrator)
  ├─→ Model Provider (megatron_patch/model/{model}/model.py)
  ├─→ Data Provider (megatron_patch/data/)
  ├─→ Tokenizer (megatron_patch/tokenizer/)
  └─→ Optimizer (Megatron core)
  ↓
backends/megatron/Megatron-LM-*/megatron/core/ (Distributed execution)
  ├─→ Tensor Parallelism (TP)
  ├─→ Pipeline Parallelism (PP)
  ├─→ Data Parallelism (DP)
  ├─→ Sequence Parallelism (SP)
  ├─→ Context Parallelism (CP)
  └─→ Expert Parallelism (EP) for MoE
```

### Model Architecture Pattern

Models use Megatron-Core's `ModuleSpec` composition pattern:

```
megatron_patch/model/{model_name}/
  ├── model.py                  # GPTModel class (main entry)
  ├── transformer_config.py     # ModelConfig dataclass
  ├── layer_specs.py            # ModuleSpec definitions
  ├── transformer_layer.py      # Layer implementation
  └── transformer/              # Attention, MLP modules
```

Key: Models are composed via specs, not inheritance chains.

## Common Commands

### Environment Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git

# Update submodules
git submodule update --init --recursive

# Set PYTHONPATH (adjust Megatron version as needed)
export PYTHONPATH=/path/to/Pai-Megatron-Patch:/path/to/Pai-Megatron-Patch/backends/megatron/Megatron-LM-250624:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

### Data Preprocessing

**Pre-training (Basic):**
```bash
cd toolkits/pretrain_data_preprocessing/
bash run_make_pretraining_dataset.sh <vocab_file> <input_jsonl> <output_prefix> <workers>
# Output: .bin/.idx binary mmap files
```

**Pre-training (Large-scale Multi-TB Data):**

For processing multi-TB datasets (e.g., DCLM, FineWeb, SlimPajama), use the optimized workflow:

```bash
cd toolkits/pretrain_data_preprocessing/

# Step 1: Convert Arrow format to JSONL (if needed)
# Uses file-level parallelism for maximum throughput (40x faster than naive approach)
python convert_arrow_to_jsonl_v2.py \
  --input-dir /data/arrow_dataset/ \
  --output-file /data/dataset.jsonl \
  --text-column text \
  --workers 224  # Use all CPU cores

# Step 2 (Optional): Create subset for testing
# Reservoir sampling for memory-efficient random sampling
python streaming_random_sample.py \
  --input /data/dataset.jsonl \
  --output /data/dataset_1pct.jsonl \
  --sample-rate 0.01 \
  --seed 42 \
  --method reservoir

# Step 3a: Quick subset preprocessing (for testing)
bash preprocess_kormo_subset.sh 1  # Process 1% subset

# Step 3b: Full dataset preprocessing
python preprocess_data.py \
  --input /data/dataset.jsonl \
  --output-prefix /data/mmap/dataset \
  --dataset-impl mmap \
  --patch-tokenizer-type Qwen2Tokenizer \
  --load /path/to/tokenizer \
  --workers 64 \
  --append-eod \
  --extra-vocab-size 0
```

**Performance Notes:**
- **Arrow → JSONL**: v2 uses file-level parallelism (IPC-free), ~40x faster than sequential
- **Sampling**: Reservoir method is memory-efficient for any file size (constant memory usage)
- **Binary conversion**: Set `--workers` to your CPU core count for optimal speed
- **Full pipeline**: For 4.5TB dataset on 224-core machine: ~8min Arrow→JSONL + ~4hrs preprocessing

**Fine-tuning (SFT):**
```bash
cd toolkits/sft_data_preprocessing/
bash convert_sft_dataset.sh <input_jsonl> <output_dir> <tokenizer_path>
```

### Model Checkpoint Conversion

**HuggingFace → Megatron:**
```bash
cd toolkits/model_checkpoints_convertor/{model_name}/
bash hf2mcore_*.sh <model_size> <hf_dir> <output_dir> <tp> <pp>
```

**Megatron → HuggingFace:**
```bash
bash mcore2hf_*.sh <model_size> <megatron_dir> <hf_output_dir> <tp> <pp>
```

### Training

**Standard training script structure:**
```bash
cd examples/{model_name}/
bash run_mcore_{model}.sh \
  <ENV>                      # dsw (single-node) or dlc (multi-node)
  <MODEL_SIZE>               # e.g., 7B, 14B, 72B
  <BATCH_SIZE>               # per-GPU micro batch size
  <GLOBAL_BATCH_SIZE>        # total batch size across all GPUs
  <LR>                       # learning rate
  <MIN_LR>                   # minimum learning rate
  <SEQ_LEN>                  # sequence length
  <PAD_LEN>                  # padding length
  <PR>                       # precision (bf16/fp16)
  <TP>                       # tensor parallel size
  <PP>                       # pipeline parallel size
  <CP>                       # context parallel size
  <ETP>                      # expert tensor parallel (MoE)
  <EP>                       # expert parallel (MoE)
  <SP>                       # sequence parallel (true/false)
  <DO>                       # distributed optimizer (true/false)
  <FL>                       # flash attention (true/false)
  <SFT>                      # supervised fine-tuning mode (true/false)
  <AC>                       # activation checkpointing level
  <OPTIMIZER_OFFLOAD>        # CPU optimizer offloading
  <SAVE_INTERVAL>            # checkpoint save interval
  <DATASET_PATH>             # training data path
  <VALID_DATASET_PATH>       # validation data path
  <PRETRAIN_CHECKPOINT_PATH> # initial checkpoint
  <TRAIN_TOKENS>             # total training tokens
  <WARMUP_TOKENS>            # warmup tokens
  <OUTPUT_BASEPATH>          # output directory
```

**Example (Qwen3 8B on 8 GPUs):**
```bash
bash run_mcore_qwen3.sh dsw 8B 1 128 1e-5 1e-6 2048 2048 bf16 4 1 1 1 1 true true false false 1 false 2000 /data/train.bin /data/valid.bin "" 100000000 1000000 /outputs
```

### Evaluation

**LM-Evaluation-Harness integration:**
```bash
cd examples/{model_name}/
python evaluate_megatron_{model}.py \
  --model-path /path/to/checkpoint \
  --tasks mmlu,hellaswag,arc_challenge \
  --tp <tensor_parallel> \
  --pp <pipeline_parallel>
```

### Testing

**Test checkpoint conversion:**
```bash
cd toolkits/model_checkpoints_convertor/{model}/
# Run HF→Megatron→HF roundtrip and compare weights
```

**Test data loading:**
```bash
cd toolkits/pretrain_data_preprocessing/
python test_dataset.py --data-path /path/to/dataset
```

## Key Technical Details

### Parallelism Strategy Selection

- **TP (Tensor Parallel)**: Splits model weights across GPUs. Use for large models (TP=4 or 8).
- **PP (Pipeline Parallel)**: Splits layers across GPUs. Use for memory constraints (PP=2, 4, 8).
- **DP (Data Parallel)**: Automatically determined as `total_gpus / (TP * PP * CP * EP)`.
- **SP (Sequence Parallel)**: Splits sequence dimension. Enable with `SP=true` when using TP.
- **CP (Context Parallel)**: For ultra-long contexts (>32K tokens). Use CP=2 or 4.
- **EP (Expert Parallel)**: For MoE models. Splits experts across GPUs.

**Rule of thumb**: For 8xGPU node with 70B model: TP=8, PP=1. For 32xGPU with 70B: TP=4, PP=2, DP=4.

### Attention Backends

- **Flash Attention** (`FL=true`): Fastest for most cases. Set `NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0`.
- **Fused Attention** (`FL=false`): More memory efficient. Set `NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1`.

### Activation Checkpointing

- `AC=0`: No checkpointing (highest memory)
- `AC=1`: Checkpoint full layers (balanced)
- `AC=2`: Checkpoint per-layer components (lowest memory)

### Optimizers

**Adam/AdamW (Default):**
- Standard optimizer for most training
- Supports distributed optimizer with ZeRO-1
- Compatible with all parallelism strategies

**Muon Optimizer (Advanced):**
- **Location**: `megatron/core/optimizer/muon.py` (Megatron-LM-251125+)
- **Description**: Hybrid Muon+Adam optimizer using Newton-Schulz orthogonalization
- **Benefits**: ~2x faster convergence in tokens, better loss trajectories
- **Use case**: Alpha model and other experimental training
- **Dependency**: `emerging-optimizers==0.2.0`

**Configuration:**
```yaml
# In training config (YAML format)
optimizer: "dist_muon"  # LayerWise distributed mode (memory-efficient)
# or
optimizer: "muon"       # Standard mode (higher memory)

# Muon hyperparameters
muon_momentum: 0.95              # Momentum (like Adam beta1)
muon_use_nesterov: true          # Nesterov momentum
muon_num_ns_steps: 5             # Newton-Schulz iterations (3-7 recommended)
muon_scale_mode: "spectral"      # Scaling mode
muon_fp32_matmul_prec: "medium"  # Matrix multiply precision
muon_tp_mode: "blockwise"        # Tensor parallel mode
muon_split_qkv: true             # Split QKV for GQA models
```

**Compatibility:**
- ✅ TP (any size, including TP=1 for Mamba)
- ✅ EP (Expert Parallel) - tested with EP=8, 256 experts
- ✅ MoE models
- ✅ BF16 precision
- ❌ Distributed optimizer (`use_distributed_optimizer: false` required)
- ❌ FP16 precision (not supported)
- ❌ CPU optimizer offloading (not compatible with Muon)

**Muon Parameter Selection:**
- **2D parameters** (Linear/Conv weights): Use Muon with Newton-Schulz
- **1D parameters** (LayerNorm, biases): Fall back to Adam
- **Expert parameters**: Automatically handled via `expert_tp` process group

**Optional: QK-Clip Stabilization**
```yaml
qk_clip: true
qk_clip_alpha: 0.5
qk_clip_threshold: 100
```

**References:**
- Implementation: [backends/megatron/Megatron-LM-251125/megatron/core/optimizer/muon.py](backends/megatron/Megatron-LM-251125/megatron/core/optimizer/muon.py)
- Tests: `tests/unit_tests/test_muon_optimizer.py`
- Example: [examples/alpha/](examples/alpha/) (uses dist_muon)

### Optimizer Offloading

Enable CPU optimizer offloading for large models with limited GPU memory:
```bash
OPTIMIZER_OFFLOAD=true  # Automatic offloading based on memory
```

**Note**: CPU optimizer offloading is **NOT compatible** with Muon optimizer. Use LayerWise distributed Muon (`dist_muon`) for memory efficiency instead.

### Checkpoint Formats

- **Legacy format**: Single directory with `model_optim_rng.pt` files
- **torch_dist format**: Distributed sharded checkpoints (recommended for 100B+ models)
- **HuggingFace format**: `.safetensors` or `.bin` files

Convert between formats using `toolkits/distributed_checkpoints_convertor/`.

### SFT vs Pre-training Differences

**Pre-training:**
- Uses binary mmap datasets (`.bin/.idx`)
- Data: `megatron_patch/data/pretrain_dataset.py`
- Loss: Standard next-token prediction

**Supervised Fine-tuning (SFT):**
- Uses JSONL datasets (conversations)
- Data: `megatron_patch/data/finetune_dataset.py`
- Loss: Per-sequence SFT loss (only compute loss on responses)
- Supports sequence packing for efficiency

### Reinforcement Learning Workflows

**ChatLearn (Alibaba):**
```bash
cd examples/{model}/
bash run_mcore_{model}_chatlearn.sh
# Supports: GRPO, GSPO, PPO
```

**Verl:**
```bash
cd examples/{model}/
bash run_mcore_{model}_verl.sh
# Supports: GRPO, PPO
```

Both require: SFT model checkpoint + reward model checkpoint + RL data.

## Model-Specific Notes

### Qwen Models

- Use `NullTokenizer` in Megatron (actual tokenizer loaded internally)
- Support GQA (Group Query Attention)
- ROPe theta: 1000000 for Qwen3
- 8B+ models use `--untie-embeddings-and-output-weights`

### MoE Models (Qwen MoE, DeepSeek-V3, Mixtral)

- Require `--moe-grouped-gemm` flag
- Set `ETP` (expert tensor parallel) and `EP` (expert parallel)
- Use `--moe-token-dispatcher-type alltoall` or `allgather`
- DeepSeek-V3 uses MLA (Multi-Latent Attention) via Transformer Engine

### Vision-Language Models (Qwen-VL, LLaVA)

- Image data: WebDataset format (`.tar` archives)
- Preprocessing: `toolkits/multimodal_data_preprocessing/`
- Training: Uses `megatron_patch/data/multimodal_dataset.py`
- Support spatial merging and temporal patch sizes

### LLaMA Models

- Use `SentencePieceTokenizer` in Megatron
- LLaMA 3.1: context length 128K with CP (Context Parallel)
- ROPe theta: 500000 (LLaMA3.1)

### Alpha Model (Experimental)

Alpha is a Qwen3-Next based Mamba Hybrid architecture for efficient LLM training experiments.

**Architecture:**
- **Base**: Qwen3-Next Mamba Hybrid (**GatedDeltaNet** + Multi-Head Attention)
  - **GatedDeltaNet**: ICLR 2025 linear attention architecture (O(n) complexity)
  - Combines gating mechanism with delta rule updates for improved in-context retrieval
  - Superior to Mamba2 in long-context understanding and common-sense reasoning
- **Implementation**: Custom Pai-Megatron GatedDeltaNet with Context Parallel support
  - File: `megatron_patch/model/qwen3_next/gated_deltanet.py` (445 lines)
  - Extends `MambaMixer` for optimal hybrid architecture integration
  - Context Parallel enabled (not available in official Megatron GDN yet)
- **Layer Mapping**: 2:1 ratio (24 Megatron layers → 12 HuggingFace layers)
- **Hybrid Pattern**: `M-M-M-*-` (3 GDN + 1 Full Attention, repeating)
  - M = GatedDeltaNet layer (Linear Attention with Gated Delta Rule)
  - `*` = Full Attention layer (Multi-Head Attention)
  - `-` = MLP layer (Feed-Forward Network)
- **Typical Config (baseline_24L)**:
  - 24 MG layers → 12 HF layers
  - 256 experts, Top-8 routing
  - 2048 hidden size, 32 attention heads
  - Attention ratio: 0.125 (12.5%), MLP ratio: 0.5 (50%)

**Training:**
- Location: [examples/alpha/](examples/alpha/)
- Training script: `bash train.sh <config_name>`
- Config files: `examples/alpha/configs/` (YAML-based, modular design)
- Model provider: `examples/alpha/pretrain_alpha.py`
- **Optimizer**: Uses Muon optimizer (`dist_muon`) for faster convergence
- **Backend**: Megatron-LM-251125 (dev branch with Muon support)

**Checkpoint Conversion (MG ↔ HF):**
- Converter: [toolkits/distributed_checkpoints_convertor/scripts/alpha/](toolkits/distributed_checkpoints_convertor/scripts/alpha/)
- **Alpha Config Tool**: [examples/alpha/tools/alpha_config.py](examples/alpha/tools/alpha_config.py) - 통합 설정에서 자동 생성
- Quick start:
  ```bash
  cd toolkits/distributed_checkpoints_convertor

  # Megatron → HuggingFace (with HF config auto-generation)
  bash scripts/alpha/run_8xH20.sh \
    baseline_24L \
    /path/to/mcore-checkpoint \
    /path/to/hf-output \
    true true bf16
  # HF_DIR 생략 시 config.json이 통합 config에서 자동 생성됨

  # 또는 기존 HF 모델 참조 사용
  bash scripts/alpha/run_8xH20.sh \
    baseline_24L \
    /path/to/mcore-checkpoint \
    /path/to/hf-output \
    true true bf16 \
    /path/to/hf-reference
  ```
- Validation: Automatic validation checks pattern length, TP constraint, attention ratio
- Documentation: [examples/alpha/tools/README.md](examples/alpha/tools/README.md)

**Unified Config Tool (NEW):**
- **Single Source of Truth**: `examples/alpha/configs/model/baseline_24L.yaml`에서 모든 설정 관리
- **자동 생성**: 변환 스크립트, HF config.json을 통합 config에서 자동 생성
- **검증**: 패턴 길이, attention ratio, MLP ratio 자동 검증
  ```bash
  cd examples/alpha
  python tools/alpha_config.py validate baseline_24L  # 설정 검증
  python tools/alpha_config.py sync baseline_24L      # 변환 스크립트 동기화
  python tools/alpha_config.py generate-hf-config baseline_24L  # HF config 생성
  ```
- **효과**: 새 모델 추가 시 수정 파일 5-7개 → 1개로 감소

**Key Constraints:**
- **TP=1 required**: Mamba layers do not support Tensor Parallelism > 1
- **EP=8 recommended**: 256 experts / 8 GPUs = 32 experts per GPU
- **Pattern validation**: Must match `--num-layers` (each layer = 1 token)

**Adding New Model Sizes (Simplified):**
1. Create YAML config: `examples/alpha/configs/model/<model_size>.yaml`
2. Validate and sync: `python tools/alpha_config.py sync <model_size>`
3. Run conversion: `bash run_8xH20.sh <model_size> ...` (HF config 자동 생성)

> **상세 가이드**: [examples/alpha/tools/README.md](examples/alpha/tools/README.md)에 통합 Config 도구 사용법 제공

**Common Issues:**
- `assert self.tp_size == 1`: Set `--tensor-model-parallel-size 1`
- `Pattern length mismatch`: Pattern must have exactly num_layers tokens
- `Invalid characters in pattern`: Only use M, *, - characters
- `Attention ratio mismatch`: Count of `*` should match `hybrid_attention_ratio × num_layers`

**Environment Variables:**
- `ALPHA_TOKENIZER_PATH`: 토크나이저 경로 (default: `${MEGATRON_PATCH_PATH}/models/Qwen3-Next-tokenizer`)
- `ALPHA_DATA_PATH`: 데이터셋 경로 (default: `/home/work/Datasets/KORMo_processed/mmap/qwen3_1pct`)
- `MEGATRON_PATCH_PATH`: Pai-Megatron-Patch 루트 (필수)

**Related Files:**
- **GatedDeltaNet**: [megatron_patch/model/qwen3_next/gated_deltanet.py](megatron_patch/model/qwen3_next/gated_deltanet.py)
- **Layer Specs**: [megatron_patch/model/qwen3_next/layer_specs.py](megatron_patch/model/qwen3_next/layer_specs.py)
- **Model Provider**: [toolkits/.../impl/alpha/model_provider.py](toolkits/distributed_checkpoints_convertor/impl/alpha/model_provider.py)
- **Synchronizer (MG→HF)**: [toolkits/.../impl/alpha/m2h_synchronizer.py](toolkits/distributed_checkpoints_convertor/impl/alpha/m2h_synchronizer.py)
- **Synchronizer (HF→MG)**: [toolkits/.../impl/alpha/h2m_synchronizer.py](toolkits/distributed_checkpoints_convertor/impl/alpha/h2m_synchronizer.py)
- **Common Utils**: [toolkits/.../impl/alpha/common.py](toolkits/distributed_checkpoints_convertor/impl/alpha/common.py) - 패턴 검증, 로깅 공통 모듈
- **Training Config**: [examples/alpha/configs/model/baseline_24L.yaml](examples/alpha/configs/model/baseline_24L.yaml)
- **Refactoring Guide**: [examples/alpha/docs/REFACTORING.md](examples/alpha/docs/REFACTORING.md)

**GatedDeltaNet vs Official Megatron:**
- Pai-Megatron implementation is optimized for hybrid architectures
- Context Parallel support (official version: TODO)
- Simpler codebase (445 vs 669 lines)
- Production-ready and tested with Muon optimizer
- See [MIGRATION_251125.md](examples/alpha/docs/MIGRATION_251125.md#gateddeltanet-implementation-analysis) for detailed comparison

## Debugging Tips

### Common Issues

**OOM (Out of Memory):**
- Reduce `BATCH_SIZE` or increase `GLOBAL_BATCH_SIZE` (gradient accumulation)
- Enable activation checkpointing (`AC=1` or `AC=2`)
- Enable optimizer offloading (`OPTIMIZER_OFFLOAD=true`)
- Increase PP (pipeline parallelism)

**Slow training:**
- Enable Flash Attention (`FL=true`)
- Enable Sequence Parallel with TP (`SP=true`)
- Check CUDA_DEVICE_MAX_CONNECTIONS=1 is set
- Enable communication-computation overlap (`--overlap-grad-reduce`)

**Checkpoint loading errors:**
- Verify TP/PP values match checkpoint's parallelism settings
- Check Megatron version compatibility (use same version as training)
- For torch_dist checkpoints, use `--use-dist-ckpt` flag

**Data loading errors:**
- Ensure `.bin` and `.idx` files exist together
- Check dataset path doesn't include `.bin` extension (provide prefix only)
- Verify dataset was created with same tokenizer vocab

### Logging and Monitoring

- TensorBoard logs: `OUTPUT_BASEPATH/tensorboard/`
- Checkpoints: `OUTPUT_BASEPATH/checkpoints/`
- Logs: `OUTPUT_BASEPATH/logs/`
- Use `print_rank_0()` for rank-0 only logging in code

### Validation

**Quick validation:**
```bash
# Generate text with trained model
cd examples/{model}/
python generate_text.py --checkpoint /path/to/checkpoint --prompt "Hello"
```

**Benchmark evaluation:**
```bash
python evaluate_megatron_{model}.py --checkpoint /path/to/checkpoint --tasks mmlu
```

## Development Workflow

### Adding a New Model

1. Create directory: `megatron_patch/model/{model_name}/`
2. Implement: `model.py`, `transformer_config.py`, `layer_specs.py`
3. Add tokenizer: `megatron_patch/tokenizer/`
4. Create conversion scripts: `toolkits/model_checkpoints_convertor/{model_name}/`
5. Add training script: `examples/{model_name}/run_mcore_{model}.sh`
6. Document in: `examples/{model_name}/README.md`

### Modifying Training Logic

- **DON'T** modify `backends/megatron/Megatron-LM-*/` (submodule)
- **DO** modify `megatron_patch/training.py` or add patches in `megatron_patch/fixes/`
- Override Megatron functions via `megatron_patch/initialize.py`

### Code Organization Principles

- Keep model implementations in `megatron_patch/model/`
- Keep training orchestration in `megatron_patch/training.py`
- Keep Megatron patches in `megatron_patch/fixes/`
- Keep examples self-contained in `examples/{model}/`
- Keep utilities in `toolkits/`

## Version Compatibility

### Megatron-LM Versions

- **Megatron-LM-251125** (latest dev, **recommended for Alpha**):
  - **Branch**: `dev` (experimental features)
  - **Commit**: `31f5049e8f8bc5a5550e74948223525b30bdc8f0` (2025-11-21)
  - **Key Features**:
    - ✅ Muon optimizer with LayerWise distribution
    - ✅ Enhanced Mamba Mixer (+37% code improvement)
    - ✅ Gated Delta Net (new SSM building block)
    - ✅ Fine-grained activation offloading for MoE
    - ✅ MoE router fusion and A2A overlapping
    - ✅ Improved TEGroupedMLP with bias support
    - ✅ HybridEP support (DeepSeek's expert parallelism)
  - **Use case**: Alpha model, experimental training with Muon
  - **Stability**: Dev branch (6-month feature lifecycle)
  - **Migration guide**: [examples/alpha/docs/MIGRATION_251125.md](examples/alpha/docs/MIGRATION_251125.md)

- **Megatron-LM-250908** (latest stable): Use for Qwen3, DeepSeek-V3
- **Megatron-LM-250624**: Use for Qwen3, Moonlight
- **Megatron-LM-241113**: General purpose
- **PAI-Megatron-LM-240718**: Legacy PAI version

Select version by setting PYTHONPATH in training scripts (line 6 of `run_mcore_*.sh`).

**Version Selection Guide:**
- **Stable production**: Use Megatron-LM-250908
- **Alpha/Experimental with Muon**: Use Megatron-LM-251125
- **Legacy models**: Use version-specific backends

### Framework Requirements

- PyTorch: ≥2.0 (2.3+ recommended)
- Transformer Engine: ≥2.9.0 (required for QK-Clip support in Muon optimizer)
- Flash Attention: 2.x or 3.x
- CUDA: 11.8+ (12.1+ recommended for FA3)

**Note**: Transformer Engine 2.9+ is required for QK-Clip stabilization feature used with Muon optimizer. Use `release_v2.9` branch for maximum compatibility with Megatron-LM-251125.

## References

- Main README: [README.md](README.md)
- Megatron-LM docs: https://github.com/NVIDIA/Megatron-LM
- Technical reports: Listed in README.md (Chinese articles on WeChat)
- Model-specific guides: `examples/{model}/README.md`
