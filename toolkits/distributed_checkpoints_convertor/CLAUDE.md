# Distributed Checkpoint Converter - Claude Code Guide

HuggingFace ↔ Megatron 체크포인트 변환 도구. 대규모 모델(100B+)의 분산 체크포인트를 효율적으로 변환합니다.

## Architecture

```
distributed_checkpoints_convertor/
├── impl/
│   ├── convert.py              # Main entry point (monkey-patches Megatron)
│   ├── general/                # Base synchronizer classes
│   └── alpha/                  # Alpha-specific implementation
│       ├── model_provider.py   # Model architecture definition
│       ├── m2h_synchronizer.py # Megatron → HF weight sync
│       ├── h2m_synchronizer.py # HF → Megatron weight sync
│       └── common.py           # Pattern validation, logging
│
└── scripts/
    └── alpha/
        ├── run_8xH20.sh        # Main conversion script
        └── configs/            # Model configs (baseline_48L.sh)
```

## Quick Commands

### Alpha: Megatron → HuggingFace
```bash
cd toolkits/distributed_checkpoints_convertor

# Standard conversion
bash scripts/alpha/run_8xH20.sh baseline_48L /mcore/path /hf/output true true bf16

# Auto mode (detect latest checkpoint)
bash scripts/alpha/run_8xH20.sh baseline_48L /training/outputs auto true true bf16

# Auto mode (specific iteration)
bash scripts/alpha/run_8xH20.sh baseline_48L /training/outputs auto:50000 true true bf16
```

### Alpha: HuggingFace → Megatron
```bash
bash scripts/alpha/run_8xH20.sh baseline_48L /hf/path /mcore/output false true bf16
```

## Script Arguments

```bash
bash scripts/alpha/run_8xH20.sh \
  <MODEL_SIZE>      # Config name (e.g., baseline_48L)
  <LOAD_DIR>        # Input checkpoint path (or training output dir for auto)
  <SAVE_DIR>        # Output path ('auto' generates timestamped dir)
  <MG2HF>           # true = MG→HF, false = HF→MG
  <USE_GPU>         # true = GPU conversion (fast), false = CPU only
  <PRECISION>       # bf16, fp16, or fp32
  [HF_DIR]          # (Optional) HF reference for config.json
```

## Alpha Weight Mapping

### 2:1 Layer Mapping
```
MG Layer:  0  1  2  3  4  5  ...  47
HF Layer:  0  0  1  1  2  2  ...  23

hf_layer_id = mg_layer_id // 2
```

### Layer Types
| MG Pattern | HF Component | Sync Method |
|------------|--------------|-------------|
| `M` (Mamba) | `linear_attn` | `set_mamba_layer_state()` |
| `*` (Attention) | `self_attn` | `set_gated_selfattn_state()` |
| `-` (MoE) | `mlp` (sparse) | `set_moe_layer_state()` |
| `D` (Dense) | `mlp` (dense) | `set_mlp_state()` |

### Mamba Weight Reshaping
```
MG in_proj.weight → split into [z, v, q, k, b, a]
                  → recombine as HF: [q, k, v, z] + [b, a]
```

## Key Implementation Files

| File | Lines | Responsibility |
|------|-------|----------------|
| `impl/convert.py` | ~100 | Entry point, Megatron patching |
| `impl/alpha/m2h_synchronizer.py` | ~256 | MG→HF weight mapping |
| `impl/alpha/h2m_synchronizer.py` | ~235 | HF→MG weight mapping |
| `impl/alpha/model_provider.py` | ~183 | Model architecture validation |
| `impl/alpha/common.py` | ~183 | Pattern validation, logging |

## Adding New Models

1. Create `impl/{model}/` directory
2. Implement `model_provider.py` (model architecture)
3. Implement `m2h_synchronizer.py` (MG → HF)
4. Implement `h2m_synchronizer.py` (HF → MG)
5. Create `scripts/{model}/` with conversion script
6. Add config in `scripts/{model}/configs/`

See: `scripts/alpha/docs/ADDING_NEW_MODELS.md`

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `Pattern length mismatch` | Pattern != num_layers | Match pattern length to `--num-layers` |
| `TP must be 1` | Alpha requires TP=1 | Set `--tensor-model-parallel-size 1` |
| `HF config not found` | Missing reference | Provide HF_DIR or let script auto-generate |
| `CUDA OOM` | Large model | Use `USE_GPU=false` for CPU conversion |

## Performance Reference

| Model | Direction | Time |
|-------|-----------|------|
| DeepSeek-V3-671B | HF→MG (GPU) | ~5min 22s |
| DeepSeek-V3-671B | MG→HF (GPU) | ~4min 43s |
| Alpha baseline_48L | Either | ~30s |

## Related Documentation

- **Alpha Conversion Guide**: `examples/alpha/docs/CONVERSION.md`
- **Alpha Architecture**: `examples/alpha/docs/ARCHITECTURE.md`
- **Adding New Models**: `scripts/alpha/docs/ADDING_NEW_MODELS.md`
