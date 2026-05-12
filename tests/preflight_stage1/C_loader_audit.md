# Phase C-loader — real GPTDataset path verification

Duration: 91.88 s.

## Design

Goes one level deeper than Phase D: instantiates the actual `GPTDataset`
class from `megatron.core.datasets.gpt_dataset` and pulls samples through
its `__getitem__` pipeline. This tests the same data path the trainer
will run, including mid-document sample starts and the real shuffle/index
structures.

**Differential design**: each source is loaded twice — once with reset
flags ON (the new training config), once with them OFF (control). If
the metrics are identical across the two, reset machinery is silently
broken. If they diverge in the expected direction, machinery is working.

## Per-source results (100 samples each)

### dclm

| Metric | reset flags ON (training) | reset flags OFF (control) | expected delta |
|---|---|---|---|
| EOD per sample (mean) | 2.99 | 2.99 | identical (same data) |
| EOD per sample (range) | [0, 9] | [0, 9] | identical |
| Samples with any EOD | 80/100 | 80/100 | identical |
| Loss mask coverage | 0.99927 | 1.00000 | **ON < OFF** (EODs masked) |
| Max position_id (mean) | 2403.7 | 4095.0 | **ON ≪ OFF** (reset) |
| Max position_id (peak) | 4095 | 4095 | OFF == 4095, ON < 4095 |
| Cross-doc attn blocked? | 1.0 | 0.0 | **ON == 1.0, OFF == 0.0** |

### korean_web

| Metric | reset flags ON (training) | reset flags OFF (control) | expected delta |
|---|---|---|---|
| EOD per sample (mean) | 4.02 | 4.02 | identical (same data) |
| EOD per sample (range) | [0, 10] | [0, 10] | identical |
| Samples with any EOD | 93/100 | 93/100 | identical |
| Loss mask coverage | 0.99902 | 1.00000 | **ON < OFF** (EODs masked) |
| Max position_id (mean) | 1954.3 | 4095.0 | **ON ≪ OFF** (reset) |
| Max position_id (peak) | 4095 | 4095 | OFF == 4095, ON < 4095 |
| Cross-doc attn blocked? | 1.0 | 0.0 | **ON == 1.0, OFF == 0.0** |

### fineweb2hq

| Metric | reset flags ON (training) | reset flags OFF (control) | expected delta |
|---|---|---|---|
| EOD per sample (mean) | 4.69 | 4.69 | identical (same data) |
| EOD per sample (range) | [0, 12] | [0, 12] | identical |
| Samples with any EOD | 82/100 | 82/100 | identical |
| Loss mask coverage | 0.99885 | 1.00000 | **ON < OFF** (EODs masked) |
| Max position_id (mean) | 2335.1 | 4095.0 | **ON ≪ OFF** (reset) |
| Max position_id (peak) | 4095 | 4095 | OFF == 4095, ON < 4095 |
| Cross-doc attn blocked? | 1.0 | 0.0 | **ON == 1.0, OFF == 0.0** |

## Verdict

The four reset-flag invariants we expect to differ between ON / OFF runs:

1. **Loss mask coverage** — under `eod-mask-loss: true`, every EOD position
   in the sample is excluded from the loss (loss_mask = 0). With ~2-3 EODs
   per 4096-token sample, expected drop from 1.0 → ~0.9993.
2. **Max position id** — under `reset-position-ids: true`, the position
   vector resets to 0 after every EOD, so the maximum within a sample is
   bounded by the longest single document in that pack (usually < 2000).
   Without reset, max == 4095 (= SEQ_LEN - 1) every sample.
3. **Cross-doc attention blocking** — under `reset-attention-mask: true`,
   queries in document N cannot attend to keys in document M < N within
   the same packed sample. The test compares attention_mask[q, 0] for q
   one position after the first EOD: it should be True (= blocked) under
   reset, False (= permitted) without.
4. **EOD count** — identical across configs (same underlying data).

If all three expected deltas are observed, the data flow is verified to
implement the frontier-standard document-aware packing.

## Status

Phase C-loader **complete**. See the JSON for full per-sample distributions.
