# Alpha v2 — Evaluation Pipeline & v1→v2 Verification

This document records the unification of the alpha evaluation pipeline (MG→HF →
validate → benchmark) and the verification performed for the **first alpha_v2
checkpoint**:

```
examples/alpha/outputs/alpha_baseline_48L_stage1_20260512_170157/checkpoints/iter_0010000
```

## 1. Ground truth (from `iter_0010000/common.pt`)

`common.pt` is the immutable record of what the run was actually launched with.
The pipeline derives all conversion/validation flags from it — never from a
hand-maintained `.sh`/hardcoded value.

| Field | Value | | Field | Value |
|---|---|---|---|---|
| num_layers | 48 | | num_experts | **192** |
| hidden_size | 2048 | | moe_ffn_hidden_size | 512 |
| ffn_hidden_size | 8192 | | moe_shared_expert_intermediate_size | 512 |
| num_attention_heads | 16 | | moe_router_topk | 8 |
| kv_channels (head_dim) | 256 | | moe_router_score_function | **sigmoid** |
| num_query_groups | 2 | | moe_router_load_balancing_type | **seq_aux_loss** |
| hybrid_override_pattern | `M-M-M-*-…` (48, no D) | | moe_aux_loss_coeff | 1e-4 |
| mamba_state_dim / head_dim | 128 / **128** | | moe_router_num_groups × group_topk | 8 × 4 |
| mamba_num_heads / groups | 32 / 16 | | moe_router_topk_scaling_factor | 2.5 |
| padded_vocab_size | **163968** | | moe_router_enable_expert_bias | True |
| tokenizer_model | `examples/alpha/tokenizer_v5` | | moe_shared_expert_gate | True |
| seq_length / max_position | 4096 / 262144 | | rotary_base / percent | 10M / 0.25 |
| optimizer | dist_muon | | EP (training) | 8 |

## 2. v1→v2 drift that the old eval path missed

The same model config lived in three places that silently diverged after the
v2 migration. The eval path read the **stale** ones:

| Where | v1 (stale) | v2 (truth) |
|---|---|---|
| `scripts/alpha/configs/baseline_48L.sh` (sourced by converter) | 128 experts, head 32, kv 128, moe-ffn 768, vocab 151936, pattern `MDM-M-*-` (49) | 192 / 16 / 256 / 512 / 163968 / `M-M-M-*-` (48) |
| `validate.sh` | nested-YAML parse (→ empty on flat v2) + vocab 151936 + Qwen3Tokenizer + nonexistent `tokenizer/` + seq 2048 | derived from common.pt |
| converter `GPT_MODEL_ARGS` | `--kv-channels 128`, `--max-position-embeddings 4096`, **no `--mamba-*`** (default head_dim 64 ≠ 128) | derived from common.pt |

Net effect if run as-was: the converter would build a **128-expert / head-128 /
mamba-head-64** skeleton and fail (or silently corrupt) loading a 192-expert /
head-256 / mamba-head-128 checkpoint; `validate.sh` would pass empty args.

### 2b. HF MoE routing was not faithful to training (caught at first real run)

Once the skeleton was correct, the first conversion reached the MoE copy and
crashed on `gate.e_score_correction_bias` missing — surfacing a deeper gap: the
HF `AlphaSparseMoeBlock` (the lm-eval path via `trust_remote_code`) routed with
**plain `F.softmax` + global top-k + no expert bias**, while the checkpoint was
trained with DeepSeek-V3 routing (**sigmoid + 8×4 group-limited + aux-loss-free
`expert_bias` + `routed_scaling_factor` 2.5**). A softmax/no-group/no-bias router
selects *different experts* → every benchmark number would be silently wrong.

Fix (mirrors `transformers` `DeepseekV3TopkRouter`):
- `hf_model/modeling_alpha.py::AlphaSparseMoeBlock` — sigmoid scores, group-limited
  + biased top-k *selection*, gather original scores, norm, `× routed_scaling`;
  `gate.e_score_correction_bias` registered as a persistent buffer (converter
  copy target).
- `hf_model/configuration_alpha.py` — `scoring_func`, `n_group`, `topk_group`,
  `routed_scaling_factor` (v2 defaults sigmoid / 8 / 4 / 2.5).
- `tools/alpha_config.py::generate_hf_config` emits those keys; `verify_pipeline.py`
  Stage 1.5 asserts them against the checkpoint.

## 3. Unified pipeline

Single entrypoint, GPU-count agnostic (EP = #GPU; torch_dist reshards EP=8→N):

```bash
bash examples/alpha/evaluate.sh <RUN_DIR|CKPT_DIR> [--iter N|latest] [--gpus N] \
     [--out DIR] [--benchmark] [--tasks standard] [--skip-convert] [--skip-validate]
```

Stages: **0** preflight gate → **1** convert (`run_convert.sh`) → **1.5** verify
(`config.json ↔ common.pt` + tokenizer round-trip) → **2** weight validation
(`validate.sh`) → **3** benchmark (opt-in).

Single source of truth: `tools/alpha_config.py`
- `load_config_from_checkpoint(common.pt)` → `ModelConfig`
- `emit-megatron-flags --from-checkpoint <ckpt>` → complete skeleton flags
  (now including the previously-missing `--mamba-*`, `--max-position-embeddings`,
  group routing, expert bias). One token per line → no `*`-glob hazard.

Converter consolidation: `run_8xH20.sh` / `run_4xGPU.sh` are now thin shims over
`run_convert.sh` (`GPUS=8` / `GPUS=4`).

## 4. Verification status (this checkpoint)

| Check | Status | Evidence |
|---|---|---|
| `common.pt` parses; ground truth extracted | ✅ | §1 |
| YAML `baseline_48L.yaml` matches checkpoint structure | ✅ | `tests/test_alpha_pipeline_config.py::test_yaml_config_matches_checkpoint_structure` |
| Generated `config.json` carries v2 fields (192 / 163968 / 256 / 24 / 512 / eos 0 / no BOS) | ✅ | `…::test_generate_hf_config_v2_fields` |
| `emit-megatron-flags` includes mamba dims + experts + routing | ✅ | `…::test_emit_flags_*` |
| HF MoE block routes DSV3 (sigmoid+group+bias), bias buffer persistent | ✅ | `…::test_modeling_alpha_moe_is_dsv3_routing` |
| No stale v1 references in convert/validate/evaluate | ✅ | `…::test_no_stale_v1_references_in_scripts` |
| Stage 0 preflight on iter_0010000 (`--gpus 4`) | ✅ | `verify_pipeline.py preflight` → "preflight passed" |
| Stage 1 convert (MG→HF, EP=4 resharded from EP=8) | ✅ | **14133/14133 weights matched**; final_norm & lm_head cos_sim=1.0 |
| Stage 2 MG↔HF weight diff (threshold 0.01) | ✅ | exit 0 — "ALL WEIGHTS MATCHED - CONVERSION VERIFIED" after coverage fix (below) |
| Stage 3 benchmark (lm-eval) | ▶ running | `bash evaluate.sh <run> --gpus 4 --benchmark` |

**Stage 2 coverage fix**: the first run matched every compared weight but `validate.sh`
exited 1 because 72 MG tensors were "not compared" — the validator's (independent)
weight map missed three v2-MoE tensors. Fixed in `validate_mg_hf_full.py`:
`router.expert_bias ↔ gate.e_score_correction_bias` and
`shared_experts.gate_weight ↔ shared_expert_gate.weight` are now compared; the
transient `router.local_tokens_per_expert` buffer (no HF counterpart) is excluded
from coverage. **Confirmed: `validate.sh` now exits 0** ("ALL WEIGHTS MATCHED -
CONVERSION VERIFIED"); the pipeline proceeded to Stage 3 (lm-eval benchmark).

To complete the end-to-end run on the current H100×4 box:

```bash
bash examples/alpha/evaluate.sh \
  examples/alpha/outputs/alpha_baseline_48L_stage1_20260512_170157 --gpus 4
# add --benchmark --tasks standard to also run lm-eval
```

## 5. Note on the CLAUDE.md config-drift memory

The alpha `CLAUDE.md` migration table still narrates "184 experts / aux-loss-free
(none)". The trained checkpoint is **192 experts / seq_aux_loss** (see §1). The
pipeline trusts `common.pt`, so this doc/narrative lag no longer affects
conversion or validation. Update the CLAUDE.md table when convenient.
