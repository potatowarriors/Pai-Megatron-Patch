# Megatron Patch Library - Claude Code Guide

Megatron-LM을 위한 비침습적(non-invasive) 패치 라이브러리. Megatron-LM 소스 수정 없이 기능을 확장합니다.

## Core Design Philosophy

**Non-invasive Patching**: Megatron-LM 코드를 직접 수정하지 않고, 런타임에 함수/클래스를 패치하여 기능 추가. 이를 통해 Megatron-LM 업데이트 시 충돌을 최소화합니다.

## Directory Structure

```
megatron_patch/
├── training.py          # Main training orchestrator (820 lines)
├── arguments.py         # Argument parsing & patching (573 lines)
├── initialize.py        # Megatron initialization wrapper
├── model/               # 32+ model architectures
│   ├── qwen3/
│   ├── qwen3_next/      # Mamba hybrid models (Alpha base)
│   ├── llama/
│   └── ...
├── data/                # Data loading & preprocessing
├── generation/          # Inference & text generation
├── ssm/                 # Mamba blocks, hybrid layer allocation
└── tokenizer/           # Tokenizer implementations
```

## Key Entry Points

| File | Purpose | When Modified |
|------|---------|---------------|
| `training.py` | `pretrain()` function - main loop | Training logic changes |
| `arguments.py` | Argument parsing & validation | New training args |
| `initialize.py` | Megatron initialization | Backend changes |

## Training Pipeline Flow

```
examples/{model}/run_mcore_*.sh
  ↓
arguments.py (Parse & patch args)
  ↓
initialize.py (Init Megatron distributed)
  ↓
training.py::pretrain()
  ├─→ Model Provider (model/{model}/model.py)
  ├─→ Data Provider (data/)
  └─→ Optimizer
```

## Muon Optimizer Integration

Alpha 모델은 `dist_muon` (LayerWise distributed Muon) 옵티마이저 사용:

```yaml
# In training config
optimizer: dist_muon
muon_momentum: 0.95
muon_num_ns_steps: 5
muon_scale_mode: spectral
```

**Location**: `backends/megatron/Megatron-LM-251125/megatron/core/optimizer/muon.py`

**Compatibility Matrix**:
| Feature | Supported |
|---------|-----------|
| TP (any size) | ✅ |
| EP (Expert Parallel) | ✅ |
| BF16 | ✅ |
| Distributed Optimizer | ❌ (must be false) |
| FP16 | ❌ |
| CPU Offloading | ❌ |

### QKV / QGKV Split for Per-Projection Newton-Schulz

Muon orthogonalizes 2D weight matrices via Newton-Schulz iteration. For fused
attention projections, applying NS to the *whole* fused matrix mixes the spectra
of semantically distinct sub-projections (Q, K, V — and Gate for gated attention).
The optimizer therefore splits these matrices per-projection, runs NS on each
sub-block independently, and concatenates back.

**Detection (`get_megatron_muon_optimizer`)**:
| Pattern in `name` | Layout | Split shapes per query-group |
|---|---|---|
| `linear_qkv.weight` | `[Q \| K \| V]` (standard) | `[q_per_group, kv_channels, kv_channels]` |
| `linear_qgkv.weight` | `[Q \| Gate \| K \| V]` (gated, e.g. Qwen3-Next/Alpha) | `[q_per_group, q_per_group, kv_channels, kv_channels]` |

`q_per_group = (num_attention_heads // num_query_groups) * kv_channels`.
Shapes are stored as `param.qkv_split_shapes` so one optimizer instance can serve
both layouts in the same model.

**Silent-failure guard**: at setup the optimizer logs an INFO line with
3-way / 4-way / total-attention-2D-weight counts. If `split_qkv=True` but no
fused QKV pattern matched yet attention-like 2D weights exist, it logs a
WARNING — catches future attention layouts (e.g. MLA, renamed projections)
before they silently regress to whole-matrix NS.

**Historical note (2026-05)**: Before this fix, Alpha's Gated Attention used
`linear_qgkv.weight` which the upstream substring matcher (`'linear_qkv.weight' in name`)
did **not** match. All Alpha checkpoints trained under that bug ran NS on the
fused `[Q|Gate|K|V]` matrix as a single block. K/V updates were
underweighted relative to Q/Gate because the larger Q/Gate spectra dominated
the unified spectral normalization.

**Verification**: `tests/unit_tests/test_muon_optimizer.py` includes
`test_muon_optimizer_qgkv_per_block_independence` — an oracle test that
asserts `optimizer.orthogonalize(...)` output equals manually-split-per-block
scaled NS at `atol=1e-5`, proving each sub-projection truly gets its own NS
iteration with no cross-block contamination.

## Adding New Models

1. Create `megatron_patch/model/{model_name}/`
2. Implement required files:
   - `model.py` - GPTModel class
   - `transformer_config.py` - ModelConfig dataclass
   - `layer_specs.py` - ModuleSpec definitions
3. Add tokenizer support in `tokenizer/`
4. Create training script in `examples/{model_name}/`

## SSM/Hybrid Models

For Mamba-based hybrid models (Qwen3-Next, Alpha):

```
megatron_patch/ssm/
├── mamba_layer.py           # Mamba layer implementation
├── mamba_hybrid_layer_allocation.py  # Layer pattern parsing
└── ...

megatron_patch/model/qwen3_next/
├── gated_deltanet.py        # GatedDeltaNet implementation (445 lines)
├── layer_specs.py           # Hybrid layer specifications
└── ...
```

**Key Constraint**: Mamba layers require `TP=1` (Tensor Parallelism = 1)

**QK-Clip (MuonCLIP Algorithm 1)**:
- `gated_attention.py`의 `clip_qk()`에서 Q/K projection weight + LayerNorm gamma 스케일링
- `_clip_layernorm_gamma()`: zero-centered gamma (1p layernorm) 대응 — `(1+w)*scale - 1`
- 공유 layernorm이므로 `min(eta)` 사용 (worst-case head 기준)

## Upstream Sync Notes

`megatron_patch/training.py`의 `pretrain()` 함수는 upstream Megatron의 `pretrain()`을 대체합니다. upstream에 새로운 기능이 추가되면 수동으로 포팅해야 합니다.

**주의**: `pretrain_alpha.py`는 `from megatron.training import pretrain` (upstream)을 직접 사용합니다. 따라서 Alpha 모델은 이 파일의 `pretrain()`을 거치지 않습니다. 아래 포팅은 `megatron_patch/training.py`의 `pretrain()`을 사용하는 다른 모델을 위한 것입니다.

### 포팅 완료된 기능
| Feature | Upstream Location | Patch Location | Date |
|---------|------------------|----------------|------|
| `no_weight_decay_cond` | `training.py:767-776` | `training.py:114-120` | 2026-02-19 |
| `clip_qk()` 호출 | `training.py:1419-1423` | `training.py:train_step()` | 2026-02-28 |
| `log_max_attention_logit` 반환 | `training.py:1482-1484` | `training.py:train_step()` | 2026-02-28 |
| `max_attention_logit` 로깅 | `training.py:1663-1666` | `training.py:training_log()` | 2026-02-28 |

### 주의: `setup_model_and_optimizer()` 호출
upstream은 `no_weight_decay_cond`, `scale_lr_cond`, `checkpointing_context` 등을 전달합니다. 새 인자가 추가되면 patch에도 반영 필요:
```python
# megatron_patch/training.py — upstream과 동기화 필요한 호출
from megatron.training.training import get_no_weight_decay_cond

no_weight_decay_cond = get_no_weight_decay_cond(
    args.no_weight_decay_cond_type,
    default_skip_embedding_weight_decay=args.embedding_init_method_std is not None,
)
model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
    model_provider, model_type,
    no_weight_decay_cond=no_weight_decay_cond)
```

## Development Guidelines

### DO
- Add patches in `megatron_patch/` or `megatron_patch/fixes/`
- Override Megatron functions via runtime patching
- Keep model implementations in `megatron_patch/model/`

### DON'T
- Modify `backends/megatron/Megatron-LM-*/` (submodule)
- Add hardcoded paths
- Break compatibility with upstream Megatron

## Debugging

```bash
# Quick validation
cd examples/{model}/
python generate_text.py --checkpoint /path --prompt "Hello"

# Full benchmark
python evaluate_megatron_{model}.py --checkpoint /path --tasks mmlu
```

## Related Files

- **Training Script**: `examples/alpha/train.sh`
- **Alpha Model Provider**: `examples/alpha/pretrain_alpha.py`
- **GatedDeltaNet**: `model/qwen3_next/gated_deltanet.py`
