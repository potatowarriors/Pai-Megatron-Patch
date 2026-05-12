# Phase 0.2 — Audit-grep for related silent failures

## Goal

The EOD silent bug surviving 79 h of preprocessing means our **verification net has structural gaps**. This phase asks: what *else* could be silently wrong because no one re-validates baseline assumptions before a multi-month run?

## Audit surfaces

1. `_AlphaTokenizer` / `tokenizer.eod` callsites (data loaders, generation, masking)
2. Token-ID hard-codes anywhere in alpha tooling
3. Vocab-size and special-token defaults in HF model code + conversion tools
4. Smoke / regression test coverage (or absence thereof)

## Findings

### ✅ Safe — current AlphaTokenizer code path is correct

- `_AlphaTokenizer.eod` (property at `megatron_patch/tokenizer/__init__.py:372-373`) delegates to `tokenizer.eos_token_id`. Post-Phase-0.0, this returns 0 (`<|endoftext|>`). Verified end-to-end in Phase 0.1.
- `preprocess_data_megatron.py:107-108` correctly appends `Encoder.tokenizer.eod` to `doc_ids` when `--append-eod` is on. Verified empirically.
- `preprocess_data.py:94-100` (separate, older script) has defensive cascade: tries `eos_token_id`, then `eod_id`, then `eod`. For `_AlphaTokenizer` it lands on line 96 (`eos_token_id` → 0). Safe.
- `fast_tokenize_v5.py:55-62` reads `eos_token` *directly from `tokenizer_config.json`* and calls `tok.token_to_id(eos_str)`. Bypasses `_AlphaTokenizer` entirely. Post-0.0 this returns 0. Safe.
- `megatron_patch/data/utils.py:315` (loss-mask construction) uses `tokenizer.eod` — same delegation chain as above. Will use id 0 post-0.0.
- `megatron_patch/generation/tokenization.py:161-162` has `if not hasattr(tokenizer, 'eod')` fallback — `_AlphaTokenizer` has `.eod` as a `@property`, so `hasattr` returns True and the fallback never fires. Safe.

### 🚨 Silent bug #2 — `examples/alpha/tools/alpha_config.py` Qwen3 default token IDs (FIXED)

**File**: `examples/alpha/tools/alpha_config.py:47-49` (now lines 47-55).

**Before**:
```python
DEFAULT_BOS_TOKEN_ID = 151643  # Qwen3's <|endoftext|>
DEFAULT_EOS_TOKEN_ID = 151645  # Qwen3's <|im_end|>
```

**Impact** (had it shipped):
- `alpha_config.py generate-hf-config` is the canonical tool used by `toolkits/distributed_checkpoints_convertor/scripts/alpha/run_{4xGPU,8xH20}.sh:147-150` to emit `config.json` for HF model conversion.
- A Megatron-trained alpha checkpoint converted via this path would receive **Qwen3 IDs (151643, 151645) in its HF `config.json`** — pointing at *some other token* in alpha's 163,968-vocab space (just whichever subword happens to live at those IDs).
- Downstream consequences: `transformers.generate()` would stop at the wrong token; SGLang / vLLM serving would inherit the wrong stop-token; chat templates that consult `eos_token_id` would mis-identify the model's EOS.
- Training is unaffected — token IDs are tokenizer-level, not used by Megatron's training loop directly. So the bug would only surface at first HF deployment.

**Fix** (committed in this phase):
```python
DEFAULT_BOS_TOKEN_ID = None  # alpha has no BOS
DEFAULT_EOS_TOKEN_ID = 0     # <|endoftext|>
```

Also updated `TokenConfig.bos_token_id` type to `Optional[int]` and `pad_token_id` default to `1` (matching alpha's `<|pad|>`).

**Verification**:
```text
TokenConfig defaults: bos_token_id=None, eos_token_id=0, pad_token_id=1  ✓
```

### ⚠️ Cleanup #3 — `examples/alpha/hf_model/configuration_alpha.py` stale defaults

**File**: `examples/alpha/hf_model/configuration_alpha.py:152-187` (`AlphaConfig.__init__` kwargs).

Stale defaults (Qwen3-era values):
- `vocab_size=151936` → alpha v5 is 163,968.
- `intermediate_size=5632` → alpha is 8192.
- `num_hidden_layers=48` → alpha HF is 24 (Megatron 48 → HF 24).
- `max_position_embeddings=32768` → alpha is 262,144.
- `num_experts=512` → alpha is 184.
- `num_experts_per_tok=10` → alpha is 8.
- `router_aux_loss_coef=0.001` → alpha is 1e-4 (DSV3 reference).

**Impact**: minimal. These defaults only matter for code paths that instantiate `AlphaConfig()` *without* kwargs (e.g. unit-test fixtures, ad-hoc Python sessions). Production conversion via `alpha_config.py generate-hf-config` writes the *actual* values into `config.json`, and `AutoModel.from_pretrained(path)` reads from that file — so end-user inference is correct.

**Disposition**: documented for cleanup; **not fixed in this preflight cycle** because (a) training is unaffected, (b) the fix touches the HF model class and risks subtle interactions with HF auto-detection, (c) the defaults serve only as misleading documentation, and (d) this preflight's primary risk is Stage 1 training quality, not HF UX. File a follow-up before the first checkpoint conversion.

### ✅ Safe — Qwen3 ID hard-codes outside alpha are intentional

`grep '151645\|151643'` hits in:
- `toolkits/model_checkpoints_convertor/qwen/*.py` — Qwen 1.5 conversion tools, correctly hardcoded for that model family. Not used by alpha.
- `backends/megatron/*/examples/post_training/modelopt/finetune.py` — NVIDIA's reference Qwen examples. Read-only submodule.
- `examples/alpha/tokenizer_v5/tokenizer.json:152731-152733` — token IDs `151643`, `151645` inside the alpha v5 vocab correspond to ordinary BBPE subwords (Korean/Greek text), unrelated to Qwen's special tokens. Pure numeric coincidence.

### Test coverage gap

`tests/` directory contains only `test_progressive_mix_dataset.py`. **No tests for**:
- `_AlphaTokenizer` encode/decode round-trip
- `_AlphaTokenizer.eod` returning the right ID under given `tokenizer_config.json`
- `preprocess_data_megatron.py --append-eod` end-to-end EOD insertion
- `IndexedDataset` integrity checks (header/dtype/sizes)
- `alpha_config.generate_hf_config` token ID emission

This is the **root cause of the verification net's structural gaps** — there's no regression test guarding against "EOS designation drifts" or "default token IDs go stale." Phase 0.3 will close the highest-priority of these gaps with `tests/test_alpha_tokenizer_eod.py`.

## Summary

| Bug | Severity | Status |
|---|---|---|
| Historical `.bin` has no EOD (Phase 0.1) | 🔴 critical (must patch data) | confirmed historical-only; Phase 0.4 will fix |
| `alpha_config.py` Qwen3 defaults | 🔴 critical (silent HF UX breakage) | **fixed in this phase** |
| `configuration_alpha.py` stale defaults | 🟡 documentation only | follow-up filed |
| Test coverage gap | 🟡 systemic | Phase 0.3 partial mitigation |

**Status**: Phase 0.2 complete. Two silent bugs identified; the actively-impactful one is fixed.
