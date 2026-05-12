# Phase 0 — Tokenizer config + EOD bug diagnosis

Pre-flight verification log for Stage 1 launch (see `.claude/plans/synthetic-wibbling-minsky.md`).

## 0.0 Tokenizer config: designate `<|endoftext|>` as EOS/EOD

### Changes

The alpha v5 tokenizer ships **three** files that can carry an `eos_token`
designation, and they must agree:

| File | Read by | Pre-flight state | Action |
|---|---|---|---|
| `tokenizer_config.json` | HF AutoTokenizer (primary) | `<\|im_end\|>` | → `<\|endoftext\|>` (initial Phase 0.0) |
| `special_tokens_map.json` | vLLM, SGLang, some chat-template utils | `<\|im_end\|>` (stale) | → `<\|endoftext\|>` (catch-up after user-spotted inconsistency) |
| `training_config.yaml` | Historical record of v5 training | `<\|im_end\|>` (stale) | → `<\|endoftext\|>` + explanatory comment (catch-up) |

```diff
-  "eos_token": "<|im_end|>",       # tokenizer_config.json:1032
+  "eos_token": "<|endoftext|>",
-  "eos_token": "<|im_end|>",       # special_tokens_map.json:100
+  "eos_token": "<|endoftext|>",
-  eos_token: <|im_end|>            # training_config.yaml:62
+  eos_token: <|endoftext|>         (with explanatory comment)
```

**Lesson learned**: Phase 0.0 initially missed the second and third files
because the runtime verification (`AutoTokenizer.from_pretrained(...).eos_token_id`)
returned the correct `0` — HF's loader prioritizes `tokenizer_config.json`.
The inconsistency was a latent bug for non-HF tools that read
`special_tokens_map.json` directly. User caught this via manual file
inspection, which surfaced what the automated verification missed.

### Rationale

Frontier-universal pattern (Qwen3, Llama 3, DeepSeek-V3, Mistral) separates pre-training EOD from chat-turn-end markers. `<|im_end|>` is conventionally the chat-turn-end token in ChatML; using it as pre-training EOD overloads its semantics. By designating `<|endoftext|>` (id 0) as EOS/EOD, we (a) align with the frontier convention and (b) reserve `<|im_end|>` (id 3) for future SFT chat templates.

### Verification

```text
$ python -c "from transformers import AutoTokenizer; \
              t = AutoTokenizer.from_pretrained('examples/alpha/tokenizer_v5'); \
              print(t.eos_token, t.eos_token_id, t.pad_token_id, len(t))"
<|endoftext|> 0 1 163860
```

All four invariants hold:
- `eos_token` resolves to `<|endoftext|>`
- `eos_token_id == 0`
- `pad_token_id == 1` (unchanged)
- `len(tokenizer) == 163860` (unchanged — `<|im_end|>` is still in vocab at id 3, just not designated EOS)

### Downstream propagation (no code edit needed)

`megatron_patch/tokenizer/__init__.py:372-373`:
```python
@property
def eod(self):
    return self.tokenizer.eos_token_id   # automatically 0 now
```

The `_AlphaTokenizer.eod` property delegates to `tokenizer.eos_token_id`, so the tokenizer-config change automatically propagates everywhere `.eod` is referenced. No Python edits required.

### Stale documentation updated

- `examples/alpha/configs/model/baseline_48L.yaml:93` — comment updated.
- `examples/alpha/CLAUDE.md` — Tokenizer section updated with rationale + 2026-05 pre-flight note.

### Cross-repo audit for hard-coded references

```bash
grep -rn 'im_end\|151645\|161645' megatron_patch/ examples/alpha/ \
  --include='*.py' --include='*.yaml' --include='*.sh' | grep -v 'tokenizer_v5/'
```

Findings (none requiring change):
- `megatron_patch/tokenizer/__init__.py:195` — chat template uses `<|im_end|>` as ChatML turn end. **Correct**: chat template should use chat tokens.
- `megatron_patch/tokenizer/tokenization_qwen_vl.py:33,168` — Qwen VL specific, separate tokenizer wrapper from `_AlphaTokenizer`. Not used by alpha.
- `megatron_patch/model/llava/language_model.py:511` — LLaVA-specific comment about `<im_end>` (without pipes), referring to a multimodal image-end token. Unrelated.
- `examples/alpha/scripts/wandb/run-*/files/config.yaml` — historical wandb run records. Read-only artifacts, no effect on new training.

```bash
grep -rn '\.eod\b' megatron_patch/ examples/alpha/ --include='*.py'
```

Findings (all auto-correct via property delegation):
- `megatron_patch/tokenizer/__init__.py:110` — for `_Qwen2Tokenizer` (different model family), not relevant.
- `megatron_patch/generation/{tokenization,generation}.py:145,162,165,189` — base-model generation termination. Now stops at `<|endoftext|>` (id 0), which is correct: pre-training generation should stop at document end.
- `megatron_patch/data/utils.py:315` — pre-training data utilities. Now uses id 0.

Status: ✅ Phase 0.0 complete. No additional code changes required for downstream.
