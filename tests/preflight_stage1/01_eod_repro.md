# Phase 0.1 — Reproducing the historical EOD silent failure

## Goal

Confirm whether the current AlphaTokenizer encode path silently drops `--append-eod` (as observed in the historical pre-tokenized `.bin` files — 0/300 docs ending in any EOD ID).

## Empirical setup

Three small JSONL docs (English / Korean / Python) → run `preprocess_data_megatron.py` end-to-end with `--patch-tokenizer-type AlphaTokenizer --append-eod` → inspect resulting `.bin` last-tokens via `IndexedDataset.get()`.

Also: isolated harness that builds `Encoder` directly and calls `Encoder.encode(json_line)` per doc.

## Result

**Both isolated harness and end-to-end preprocessing append EOD correctly under the current code.**

Post-0.0 (`eos_token` = `<|endoftext|>`, id 0):

```text
doc 0 ("Hello world."):           last 5 = [28625, 1552, 109, 0]              last == eod(0) ? True
doc 1 ("한국어 문서 샘플입니다."):   last 5 = [17032, 44353, 1751, 109, 0]      last == eod(0) ? True
doc 2 (Python snippet):           last 5 = [163844, 890, 316, 4800, 0]        last == eod(0) ? True
```

Pre-0.0 (`eos_token` = `<|im_end|>`, id 3) — temporary revert for verification:

```text
PRE-0.0 state: eos_token='<|im_end|>', eod=3
  last 5 of doc = [28625, 1552, 109, 3]              last == eod(3) ? True
  last 5 of doc = [100291, 6862, 3]                   last == eod(3) ? True
```

Both configurations produce documents with the configured EOD ID appended at the end. **The current code path is correct under both eos_token settings.**

## Why the historical data lacks EOD — unresolved

Git archaeology:

- DCLM `.bin` was completed at **2026-05-11 22:27:16** (after 79 h of preprocessing).
- 79 h back ≈ **2026-05-08 ~15:27** preprocessing start.
- `preprocess_data_megatron.py`: filesystem mtime **2026-05-08 14:42:26**.
- `megatron_patch/tokenizer/__init__.py`: filesystem mtime **2026-05-08 14:34:51**.
- `preprocess_dclm_v5.sh`: filesystem mtime **2026-05-08 14:40:58**.
- Commit history: `906f054 feat(alpha): Migrate to Qwen3.5 + DSV3 MoE + alpha v5 tokenizer` (committed 2026-05-07) did **not** include the `AlphaTokenizer` branches in either file. Those additions exist only as uncommitted local modifications (visible in `git diff HEAD`).

So the AlphaTokenizer code path *was* in the local working copy when preprocessing started, but never committed. **The transient state of the working copy between May 7 (commit 906f054) and May 8 14:42 (last preprocess_data_megatron.py edit) is unrecoverable** — we cannot prove what exactly ran during the 79 h preprocessing.

Plausible hypotheses (none provable):

1. An interim version of the AlphaTokenizer wrapper had `.eod` returning `None` (e.g. if `eos_token` was missing from tokenizer_config at that point) → `doc_ids.append(None)` would error or be silently skipped depending on Python version.
2. The line-89 special-handling branch did not include "AlphaTokenizer" at the time, so the code fell through to the line-94 `else` branch using `Encoder.tokenizer(sentence)`. The `_AlphaTokenizer.__call__` returns the inner HF tokenizer's BatchEncoding — both paths produce the same token list, so this alone doesn't drop EOD.
3. `preprocess_dclm_v5.sh` was edited later — perhaps `--append-eod` was added *after* the May 8 preprocessing kick-off and we mis-recall when.

All hypotheses are consistent with: **the current code is correct; the historical data is permanently EOD-less**.

## Conclusion

- ✅ No code patch required in Phase 0.3 — the encode path already works.
- ✅ Add regression tests to prevent re-introduction.
- 🔁 Phase 0.4 (inject_eod.py + 32-min `.bin` patch on all 3 datasets) is still required: historical data is the only EOD-less artifact.
- 🧩 Add Phase 0.2 verification (audit-grep) to confirm no other silent failures elsewhere in the AlphaTokenizer code path.

## Status

Phase 0.1: **complete** — silent bug found to be historical-only; current code verified correct via two independent reproductions (isolated harness + end-to-end run).
