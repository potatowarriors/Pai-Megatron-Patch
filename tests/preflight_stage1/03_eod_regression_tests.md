# Phase 0.3 — EOD regression tests

## Goal

Phase 0.1 found that the current code path is correct — no patch needed. Phase 0.2 found one production-impactful silent bug (`alpha_config.py` Qwen3 defaults) and fixed it. This phase closes the **verification gap** with regression tests that fail loudly if any of the corrected invariants regress.

## Tests added

File: `tests/test_alpha_tokenizer_eod.py` (8 tests, ~21 s runtime, no GPU required).

| # | Test | Guards against |
|---|---|---|
| A1 | `test_tokenizer_config_eos_is_endoftext` | tokenizer_config.json:eos_token drifting back to `<\|im_end\|>` or any other chat-only token |
| A2 | `test_hf_tokenizer_resolves_eos_to_id_0` | HF AutoTokenizer resolving EOS to wrong ID (e.g. if the JSON config is internally inconsistent with tokenizer.json) |
| B1 | `test_alpha_tokenizer_eod_returns_zero` | `_AlphaTokenizer.eod` property regressing (shadowing, attribute assignment elsewhere) |
| B2 | `test_alpha_tokenizer_roundtrip` | tokenize/detokenize losing bytes for English/Korean/code/Arabic |
| C1 | `test_preprocess_encoder_appends_eod` | preprocess_data_megatron.py `Encoder.encode` silently dropping EOD when `--append-eod` is on |
| C2 | `test_preprocess_encoder_no_eod_when_flag_off` | confirming the flag is *actually* gating the append (not always-on / always-off) |
| D | `test_alpha_config_token_defaults_are_alpha_v5` | `alpha_config.TokenConfig` defaults drifting back to Qwen3 IDs |
| E | `test_preprocess_end_to_end_writes_eod_in_bin` | the exact regression that produced Stage 1's EOD-less .bin files — subprocess run of preprocess_data_megatron.py + IndexedDataset read-back |

## Test E is the most important

It runs `preprocess_data_megatron.py` end-to-end via subprocess (matching real CLI invocation), then reads the produced `.bin` back via `IndexedDataset` and asserts every doc's last token is `0`. This is the **inverse of the historical 0/300 finding** — it would have caught the original bug immediately.

## Result

```text
tests/test_alpha_tokenizer_eod.py::test_tokenizer_config_eos_is_endoftext PASSED
tests/test_alpha_tokenizer_eod.py::test_hf_tokenizer_resolves_eos_to_id_0 PASSED
tests/test_alpha_tokenizer_eod.py::test_alpha_tokenizer_eod_returns_zero PASSED
tests/test_alpha_tokenizer_eod.py::test_alpha_tokenizer_roundtrip PASSED
tests/test_alpha_tokenizer_eod.py::test_preprocess_encoder_appends_eod PASSED
tests/test_alpha_tokenizer_eod.py::test_preprocess_encoder_no_eod_when_flag_off PASSED
tests/test_alpha_tokenizer_eod.py::test_alpha_config_token_defaults_are_alpha_v5 PASSED
tests/test_alpha_tokenizer_eod.py::test_preprocess_end_to_end_writes_eod_in_bin PASSED
======================== 8 passed, 4 warnings in 21.50s
```

## Code patches in this phase

| File | Change |
|---|---|
| `examples/alpha/tools/alpha_config.py:47-69` | (from Phase 0.2) Default `bos_token_id=None, eos_token_id=0, pad_token_id=1`. |
| `tests/test_alpha_tokenizer_eod.py` (new) | 8 regression tests above. |

No changes required to `preprocess_data_megatron.py` or `megatron_patch/tokenizer/__init__.py` — current code is correct (verified end-to-end in Phase 0.1).

## Status

Phase 0.3 **complete**. Regression net in place — same class of silent failure (EOD drift, chat-token conflation, Qwen3-default leakage) will fail loudly in CI now, not silently after 79 h of preprocessing.
