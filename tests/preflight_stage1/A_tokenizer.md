# Phase A — Tokenizer round-trip & frontier comparison

Duration: 21.03 s. Status: **PASS**

## A1 — Special-token contract

```json
{
  "len_tokenizer": 163860,
  "eos_token": "<|endoftext|>",
  "eos_token_id": 0,
  "bos_token_id": null,
  "pad_token": "<|pad|>",
  "pad_token_id": 1,
  "decode_0": "<|endoftext|>",
  "decode_1": "<|pad|>",
  "decode_2": "<|im_start|>",
  "decode_3": "<|im_end|>"
}
```

All assertions hold: `len=163860`, `eos_token_id=0`, `pad_token_id=1`, `bos=None`,
decode(0)='<|endoftext|>', decode(1)='<|pad|>', decode(2)='<|im_start|>', decode(3)='<|im_end|>'.

## A2 — Encode/decode determinism (closed loop via existing .bin)

Sample 1000 docs from each Stage 1 `.bin`, decode → re-encode → compare token IDs.

| Source | Sampled | Mismatches | Total chars decoded |
|---|---:|---:|---:|
| dclm | 1000 | 0 | 6,326,542 |
| korean_web | 1000 | 0 | 2,581,931 |
| fineweb2hq | 1000 | 0 | 2,983,680 |

Mismatches = 0 across all sources → tokenizer encode/decode is bit-reproducible on real corpus tokens.

## A3 — _AlphaTokenizer ↔ raw rust tokenizer parity

200 short sample texts: HF (add_special_tokens=False) vs rust tokenizer.encode → `0 mismatches / 200`.
Confirms the Megatron wrapper introduces no silent re-encoding versus a fresh rust load.

## A4 — Frontier deviation matrix

| Property | Alpha v5 | Qwen3 | Llama 3 | DeepSeek-V3 | Intentional? |
|---|---|---|---|---|---|
| EOS / pre-train EOD | <|endoftext|> (id 0) | <|endoftext|> (151643) | <|end_of_text|> (128001) | <|end_of_sentence|> | ✅ aligns with all three |
| BOS | None | None (per Qwen3) | <|begin_of_text|> (128000) | <|begin_of_sentence|> | ✅ matches Qwen3 / DSV3-ish |
| PAD | <|pad|> (id 1) | <|endoftext|> (re-used) | <|finetune_right_pad_id|> (128004) | <|end_of_sentence|> | ✅ dedicated pad token |
| Chat turn end | <|im_end|> (id 3, reserved for SFT) | <|im_end|> (151645) | <|eot_id|> (128009) | <|Assistant|>/<|User|> wrap | ✅ matches Qwen3 |
| Effective vocab | 163,860 | 151,936 | 128,256 | ~100K | 📝 alpha-specific (multi-lingual + reserved) |
| Padded vocab | 163,968 (mult of 128) | 151,936 | 128,256 | varies | ✅ standard padding |
| Reserved special slots | 80 | <5 | <10 | <10 | 📝 alpha-specific (futures: more tools/modalities) |
| Tokenizer algo | BBPE (Rust) | BBPE | BBPE | BBPE-ish | ✅ universal |
| use_fast | True (only fast path) | True | True | True | ✅ standard |
| bos_token in tokenizer | None | None | Present | Present | 📝 alpha + Qwen3 share this choice |

All deviations from frontier defaults are either (a) ✅ aligned with at least one major frontier model,
or (b) 📝 alpha-specific intentional choices (multi-lingual vocab size, reserved-token headroom).
No 🚨 row.

## Status

Phase A **complete**. All four checks pass.
