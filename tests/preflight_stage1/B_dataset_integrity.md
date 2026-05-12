# Phase B — Pre-tokenized dataset structural audit

Total runtime per source recorded in JSON. All Stage 1 sources tested:
DCLM, Korean Web, FineWeb2-HQ. Each runs five checks (B1-B5).

## dclm

- **Header**: magic `4d4d49444944580000` (match=True), version 1, dtype `int32` (code 4)
- **Docs**: 312,031,637 (idx) / sequences: 312,031,636
- **Tokens (total)**: 443,787,161,344
- **.bin size**: 1,775,148,645,376 bytes = sum_lens × 4? True
- **B3 (range)**: sampled 419,430,400 tokens; range [0, 163859]; max < 163,860? True
- **B3 top-10 decoded**: `' the'` (15,811,592), `','` (15,502,157), `'.'` (14,900,169), `' to'` (8,659,650), `' of'` (8,376,790)
- **B4 (EOD presence)**: **10000 / 10000** docs end in id 0 ✅ post-Phase-0.4 verified
- **B4 top last-tokens**: `'<|endoftext|>'` (10000)
- **B5 (lengths)**: empty=0, <16 toks=1, mean=1422, median=769, p95=4530, max=474,743
- Wall: 34.26 s

## korean_web

- **Header**: magic `4d4d49444944580000` (match=True), version 1, dtype `int32` (code 4)
- **Docs**: 15,738,377 (idx) / sequences: 15,738,376
- **Tokens (total)**: 16,964,085,144
- **.bin size**: 67,856,340,576 bytes = sum_lens × 4? True
- **B3 (range)**: sampled 419,430,400 tokens; range [0, 163859]; max < 163,860? True
- **B3 top-10 decoded**: `'.'` (11,998,788), `' '` (8,338,104), `','` (7,975,410), `'.\n'` (3,109,219), `'\n'` (2,275,401)
- **B4 (EOD presence)**: **10000 / 10000** docs end in id 0 ✅ post-Phase-0.4 verified
- **B4 top last-tokens**: `'<|endoftext|>'` (10000)
- **B5 (lengths)**: empty=0, <16 toks=683, mean=1078, median=729, p95=3138, max=70,111
- Wall: 25.7 s

## fineweb2hq

- **Header**: magic `4d4d49444944580000` (match=True), version 1, dtype `int32` (code 4)
- **Docs**: 6,137,776 (idx) / sequences: 6,137,775
- **Tokens (total)**: 5,719,562,440
- **.bin size**: 22,878,249,760 bytes = sum_lens × 4? True
- **B3 (range)**: sampled 419,430,400 tokens; range [0, 163859]; max < 163,860? True
- **B3 top-10 decoded**: `','` (12,220,623), `'.'` (7,535,130), `' a'` (3,583,455), `'.\n'` (3,529,383), `' '` (3,445,424)
- **B4 (EOD presence)**: **10000 / 10000** docs end in id 0 ✅ post-Phase-0.4 verified
- **B4 top last-tokens**: `'<|endoftext|>'` (10000)
- **B5 (lengths)**: empty=0, <16 toks=0, mean=932, median=395, p95=3033, max=263,528
- Wall: 22.95 s

## Status (post-Phase-0.4 re-run, 2026-05-12 14:04:41)

✅ **All five checks pass on all three sources.** B4 was re-run after Phase 0.4
completed the in-place `id 3 → id 0` remap on all `.bin` files:

- DCLM: 10000/10000 docs end in id 0 (`<|endoftext|>`)
- Korean Web: 10000/10000 docs end in id 0
- FineWeb2-HQ: 10000/10000 docs end in id 0

B3 (token-range) also picked up the change: previous range was [3, 163859];
post-injection range is now [0, 163859] — the new minimum of 0 is the EOD
markers we just injected, and the maximum is still well under the effective
vocab boundary of 163,860.

B1/B2 (header + size consistency) and B5 (length statistics) are unchanged
by the remap, as expected (no `.idx` modifications, no token-count changes).
