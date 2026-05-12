# Phase F — Stage 1 pre-flight: decisions & deviations log

Single-page summary that a future reviewer (or auditor) opens first.

## Verdict: 🟢 Ready to launch Stage 1 (pending Phase E smoke)

All Phase 0-D checks pass. The remaining gate is the 100-iter smoke test
(Phase E), which is a single-shot user-launched run.

### Run-completion timeline (2026-05-12)

| Milestone | Time | Wall |
|---|---|---|
| Phase 0.0 → 0.3 (tokenizer config + bug fix + tests) | 10:13 → 10:35 | ~22 min |
| Phase A (tokenizer round-trip + frontier matrix) | 10:36 → 10:38 | 2 min |
| Phase B initial (pre-injection) | 10:38 → 10:42 | 4 min |
| Phase C (decoded samples) | 10:42 → 10:43 | 1 min |
| Phase 0.4 — FineWeb2-HQ remap | 10:48 → 10:50 | 2.6 min |
| Phase 0.4 — Korean Web remap | 10:55 → 11:06 | 10.8 min |
| Phase 0.4 — DCLM remap | 11:07 → 14:03 | **2h 55m** |
| Phase B re-run (post-injection) | 14:03 → 14:04 | <2 min |
| Phase D (training-time data flow) | 14:04 → 14:04:41 | <1 min |
| **Total preflight** | 10:13 → 14:05 | **~3h 52m** |

The DCLM remap dominated wall time (76% of total) due to NFS read-modify-write
overhead on 1.78 TB with sparse 4-byte writes per page.

### Phase D verification highlights (post-injection)

The single-sample snapshot in `D_sample_snapshot.txt` packed 5 DCLM head docs
into a 4097-token sample. EOD (id 0) appeared at positions [1013, 2005, 2412, 3257].
At each EOD position:

- `tokens[eod_pos] = 0` ✅
- `loss_mask[eod_pos] = 0.0` ✅ (EOD masked from loss)
- `position_ids[eod_pos+1] = 0` ✅ (next-doc position reset)
- `attention_mask[q_after_eod, k_in_prev_doc] = masked` ✅ (cross-doc attention blocked)

All three reset-flag invariants confirmed working as the frontier Megatron recipe intends.

## Deviations from frontier-model defaults

Every difference between alpha and the major frontier models (Qwen3, Llama 3,
DeepSeek-V3, Mistral) was reviewed during Phases A1-A4 and Phase 0.2 audit-grep.
Each row below is classified as:

- **Intentional**: deliberate alpha design choice with explicit rationale.
- **Will-fix**: discovered as bug during pre-flight, fixed in this cycle.
- **Documented-cleanup**: minor cosmetic / documentation drift, deferred.

| # | Property | Alpha v5 | Frontier default | Classification |
|---|---|---|---|---|
| 1 | EOS / pre-training EOD | `<\|endoftext\|>` (id 0) | Qwen3/GPT: `<\|endoftext\|>`; Llama 3: `<\|end_of_text\|>` | **Intentional** (frontier-aligned; was `<\|im_end\|>` pre-Phase-0.0 — **will-fix → done**) |
| 2 | BOS | None | Qwen3: None; Llama 3: `<\|begin_of_text\|>` | **Intentional** (matches Qwen3; alpha is Qwen3-aligned base) |
| 3 | Chat-turn-end | `<\|im_end\|>` (id 3, reserved for SFT) | Qwen3: `<\|im_end\|>`; Llama 3: `<\|eot_id\|>` | **Intentional** (frontier-universal separation pattern) |
| 4 | Effective vocab | 163,860 | Qwen3: 151,936; Llama 3: 128,256 | **Intentional** (alpha-specific multi-lingual + 80 reserved special slots for future tools) |
| 5 | Padded vocab | 163,968 (multiple of 128) | varies | **Intentional** (standard padding for GPU memory alignment) |
| 6 | Reserved special slots | 80 | <10 typical | **Intentional** (futures: tools, modalities) |
| 7 | `--reset-position-ids` | true | NVIDIA Qwen3-Next reference: true; Megatron default: false | **Intentional** (frontier-standard; was off pre-flight — **will-fix → done**) |
| 8 | `--reset-attention-mask` | true | NVIDIA Qwen3-Next reference: true; Megatron default: false | **Intentional** (paired with #7) |
| 9 | `--eod-mask-loss` | true | NVIDIA Qwen3-Next reference: true; Megatron default: false | **Intentional** (paired with #7) |
| 10 | Doc-end EOD in `.bin` | id 0 at every doc end (post-Phase-0.4 remap) | Universal: EOD-in-stream | **Will-fix → done** (was id 3 from pre-Phase-0.0 era; Phase 0.4 remapped 3 → 0) |
| 11 | `alpha_config.py` default token IDs | bos=None, eos=0, pad=1 | — | **Will-fix → done** (were Qwen3 IDs 151643 / 151645) |
| 12 | `configuration_alpha.py` defaults | various stale (vocab_size=151936, intermediate_size=5632, max_position_embeddings=32768, rope_theta=1e4, num_experts_per_tok=10, num_experts=512, router_aux_loss_coef=1e-3) | — | **Will-fix → done** (2nd-pass 2026-05-12; 7 defaults + docstring updated; regression test `test_configuration_alpha_defaults_match_baseline_48L` added) |

## Phase artifacts

| Phase | Status | Artifact |
|---|---|---|
| 0.0 | ✅ | `00_eod_bug_diagnosis.md` |
| 0.1 | ✅ | `01_eod_repro.md` (current code verified correct; historical mystery) |
| 0.2 | ✅ | `02_audit_grep.md` (2 silent bugs surfaced; 1 fixed, 1 deferred) |
| 0.3 | ✅ | `03_eod_regression_tests.md` + `tests/test_alpha_tokenizer_eod.py` (9/9 passing) |
| A | ✅ | `A_tokenizer.md` + `A_roundtrip_report.json` |
| B | ✅ | `B_dataset_integrity.md` + per-source JSONs |
| C | ✅ | `C_decoded_samples.md` + `C_decoded_samples.txt` |
| 0.4 | ✅ | all 3 datasets remapped id 3 → id 0, post-conditions verified |
| B re-run | ✅ | post-injection: 10000/10000 docs end in id 0 across all sources |
| D | ✅ | `D_dataflow.md` + `D_sample_snapshot.txt` — 3 invariants verified |
| E | ⏳ | `run_phase_e_smoke.sh` — user-launched final gate |
| F | ✅ (this doc) | `F_decisions.md` |

## Open questions / follow-ups

1. **Why the 79 h DCLM preprocessing produced EOD-bearing data** despite Phase 1
   exploration reporting "0/100 docs end in id 3" — root cause unrecoverable
   (the Phase 1 agent's empirical check was simply incorrect, as verified by
   Phase B's 3 datasets × 4 seeds × 250 samples + boundary docs all showing
   100% id 3 at doc end). No code change required.

2. **`configuration_alpha.py` stale defaults** (Item 12 above) — **resolved
   in 2nd-pass verification (2026-05-12)**. Originally flagged as 2 stale
   values (vocab_size, intermediate_size); 2nd-pass parallel Explore audit
   surfaced 5 additional stale defaults from the same migration boundary
   (commit `906f054`). All 7 + matching docstring rows corrected in a single
   logical commit; new regression test pins each value against
   baseline_48L.yaml so future migration drift is caught at CI time.

3. **SFT readiness for `<|im_end|>`**: the id 3 embedding will be effectively
   cold-started at SFT (since Phase 0.4 remap removed all id 3 occurrences from
   pre-training data). This matches the frontier pattern — Qwen3, Llama 3, etc.
   all have their chat-turn-end token's embedding initialized fresh at SFT.

4. **Phase E smoke pass criteria**: loss should drop from ~12.0 (= `ln(163968)`)
   toward ~10-11 by iter 100, no NaN, no OOM, no blend-loader stalls.

## What changed in code

| File | Change | Why |
|---|---|---|
| `examples/alpha/tokenizer_v5/tokenizer_config.json` | `eos_token: <\|im_end\|> → <\|endoftext\|>` | Phase 0.0 |
| `examples/alpha/tokenizer_v5/special_tokens_map.json` | same | Phase 0.0 catch-up (user-spotted) |
| `examples/alpha/tokenizer_v5/training_config.yaml` | same + explanatory comment | Phase 0.0 catch-up |
| `examples/alpha/tools/alpha_config.py` | `DEFAULT_BOS=None, DEFAULT_EOS=0` (was Qwen3 IDs); `pad_token_id=1` default | Phase 0.2 silent-bug fix |
| `examples/alpha/configs/model/baseline_48L.yaml` | tokenizer comment updated | Phase 0.0 |
| `examples/alpha/configs/training/stage1.yaml` *(new authoritative training preset; pretrain_auxfree.yaml deprecated 2026-05-12)* | `+reset-position-ids: true`, `+reset-attention-mask: true`, `+eod-mask-loss: true` | Phase 0.0 + standard recipe |
| `examples/alpha/configs/training/pretrain_auxfree.yaml` | same three flags **+** deprecation header pointing to stage1.yaml | retained for git-history compatibility; do not use for new runs |
| `examples/alpha/train.sh` | docstring example + WANDB comment updated to reference `stage1` | sync with new preset name |
| `examples/alpha/CLAUDE.md` | Tokenizer section updated | Phase 0.0 doc |
| `toolkits/pretrain_data_preprocessing/remap_eod.py` | new — in-place EOD remapper using mmap | Phase 0.4 tool |
| `tests/test_alpha_tokenizer_eod.py` | new — 9 regression tests | Phase 0.3 |
| `tests/preflight_stage1/*` | new — preflight artifacts (this doc + scripts) | All phases |

## What changed in data

The on-disk `.bin` files for Stage 1 were modified in place. Specifically,
the single int32 at the byte offset corresponding to each document's last
token was rewritten from `3` (`<\|im_end\|>`) to `0` (`<\|endoftext\|>`).
Total tokens, `.idx` files, and all non-end byte positions are unchanged.

| File | Pre-flight state | Post-flight state |
|---|---|---|
| `/home/work/Datasets/LL_preprocessed/v5/stage1/dclm/data_text_document.bin` | doc ends in id 3 | doc ends in id 0 |
| `/home/work/Datasets/LL_preprocessed/v5/stage1/korean_web/data_text_document.bin` | doc ends in id 3 | doc ends in id 0 |
| `/home/work/Datasets/LL_preprocessed/v5/stage1/fineweb2hq/data_text_document.bin` | doc ends in id 3 | doc ends in id 0 |

The remap is fully reversible by re-running `remap_eod.py --old-eod 0 --new-eod 3`.

## Sign-off block

- [x] Phase 0.4 DCLM remap completed (verified via `run_phase_b.py` re-run; B4
      shows 10,000 / 10,000 docs ending in id 0 for each of the 3 sources)
- [x] Phase D dataflow audit passes (3/3 reset-flag invariants verified on a
      packed 4097-token sample from DCLM)
- [x] **2nd-pass verification (2026-05-12)** — 3 parallel Explore agents
      (regression / silent-bug-hunt / production-launch-audit) report no new
      issues except `configuration_alpha.py` Item 12 (7 stale defaults), which
      is now fixed. Regression suite expanded to 10 tests, all passing in 21.72s.
- [ ] Phase E 100-iter smoke loss < 11.5 — **user-launched gate** before Stage 1
      (`bash tests/preflight_stage1/run_phase_e_smoke.sh`)
- [ ] User signs off → launch Stage 1
      (`bash examples/alpha/train.sh baseline_48L stage1 stage1_v5_blend`)
