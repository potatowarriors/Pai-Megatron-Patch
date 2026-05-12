"""Phase C — decoded-sample sanity check.

C1: decode 5 documents from each source (indices 0, 1, 100, 1000, 100000) via
    _AlphaTokenizer.detokenize, write to a human-readable file for visual inspection.
C2: FineWeb2-HQ language distribution — sample 200 docs, run a lightweight
    script-based language classifier (without external langdetect dependency).
"""
import json
import os
import re
import sys
import unicodedata

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "backends", "megatron", "Megatron-LM-251125"))
sys.path.insert(0, os.path.join(REPO, "examples", "alpha", "tools"))

DATA = {
    "dclm":       "/home/work/Datasets/LL_preprocessed/v5/stage1/dclm/data_text_document",
    "korean_web": "/home/work/Datasets/LL_preprocessed/v5/stage1/korean_web/data_text_document",
    "fineweb2hq": "/home/work/Datasets/LL_preprocessed/v5/stage1/fineweb2hq/data_text_document",
}
OUT_MD = os.path.join(REPO, "tests", "preflight_stage1", "C_decoded_samples.md")
OUT_DUMP = os.path.join(REPO, "tests", "preflight_stage1", "C_decoded_samples.txt")

import argparse
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron_patch.tokenizer import build_tokenizer

TOK = os.path.join(REPO, "examples", "alpha", "tokenizer_v5")
args = argparse.Namespace(
    patch_tokenizer_type="AlphaTokenizer", load=TOK, extra_vocab_size=0,
    padded_vocab_size=163968, rank=0,
    make_vocab_size_divisible_by=128, tensor_model_parallel_size=1)
alpha = build_tokenizer(args)


# ---------------------------------------------------------------------------
# C1 — decoded samples
# ---------------------------------------------------------------------------
SAMPLE_IDS = [0, 1, 100, 1000, 100_000]
dump_lines = ["# Phase C — decoded samples", ""]
md = ["# Phase C — decoded sample sanity check", ""]

for src, prefix in DATA.items():
    ds = IndexedDataset(prefix)
    n_docs = int(ds.sequence_lengths.shape[0])
    md.append(f"## {src}  ({n_docs:,} docs)\n")
    dump_lines.append(f"\n{'='*80}\n## SOURCE: {src}  ({n_docs:,} docs)\n{'='*80}\n")
    for sid in SAMPLE_IDS:
        if sid >= n_docs:
            continue
        tokens = ds.get(sid)
        # Trim very long docs for the dump
        head = tokens[:400].tolist()
        tail = tokens[-50:].tolist()
        try:
            text_head = alpha.detokenize(head)
            text_tail = alpha.detokenize(tail)
        except Exception as e:
            text_head = f"<decode-error: {e}>"
            text_tail = ""
        dump_lines += [
            f"\n--- doc {sid} (len={len(tokens)} tokens, last 5 ids = {tokens[-5:].tolist()}) ---",
            "HEAD (first ≤400 tokens decoded):",
            text_head,
            "...",
            "TAIL (last ≤50 tokens decoded, including EOD):",
            text_tail,
            "",
        ]
        md.append(f"- **doc {sid}**: len={len(tokens)} tokens, ends `{tokens[-5:].tolist()}` "
                  f"= `{alpha.detokenize(tokens[-5:].tolist())!r}` (decoded tail).")
    md.append("")

with open(OUT_DUMP, "w", encoding="utf-8") as f:
    f.write("\n".join(dump_lines))

# ---------------------------------------------------------------------------
# C2 — script-based language distribution for fineweb2hq
# ---------------------------------------------------------------------------
def script_of_char(ch):
    """Return a coarse script bucket for a character."""
    if ch.isspace() or ch in '.,!?;:()-"\'`/*+=<>[]{}|\\':
        return "punct"
    cp = ord(ch)
    if cp < 0x80:
        return "Latin-ASCII"
    if 0x0400 <= cp <= 0x04FF or 0x0500 <= cp <= 0x052F:
        return "Cyrillic"
    if 0x0590 <= cp <= 0x05FF:
        return "Hebrew"
    if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
        return "Arabic"
    if 0x0900 <= cp <= 0x097F:
        return "Devanagari"
    if 0x0E00 <= cp <= 0x0E7F:
        return "Thai"
    if 0x1100 <= cp <= 0x11FF or 0xAC00 <= cp <= 0xD7AF:
        return "Hangul"
    if 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
        return "Japanese-kana"
    if 0x4E00 <= cp <= 0x9FFF:
        return "CJK-Han"
    if 0x0370 <= cp <= 0x03FF:
        return "Greek"
    if 0x00C0 <= cp <= 0x024F:
        return "Latin-extended"
    return f"Other-U+{cp:04X}"


def dominant_script(text, sample_chars=2000):
    counts = {}
    for ch in text[:sample_chars]:
        s = script_of_char(ch)
        if s in ("punct",):
            continue
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return "empty"
    return max(counts.items(), key=lambda kv: kv[1])[0]


import random
random.seed(42)
ds = IndexedDataset(DATA["fineweb2hq"])
n_docs = int(ds.sequence_lengths.shape[0])
sample = random.sample(range(n_docs), 200)
script_counts = {}
for sid in sample:
    toks = ds.get(sid).tolist()[:300]
    try:
        text = alpha.detokenize(toks)
    except Exception:
        continue
    script = dominant_script(text)
    script_counts[script] = script_counts.get(script, 0) + 1

script_dist = sorted(script_counts.items(), key=lambda kv: -kv[1])
md += [
    "## C2 — FineWeb2-HQ language script distribution (200-doc sample)",
    "",
    "Dominant Unicode script per doc (coarse classifier — no external langdetect dep).",
    "",
    "| Script | Count | % |",
    "|---|---:|---:|",
]
total = sum(script_counts.values())
for s, c in script_dist:
    md.append(f"| {s} | {c} | {100*c/total:.1f} |")

# Sanity check: no single script should dominate > 30% if 20 languages are balanced
top_pct = 100 * script_dist[0][1] / total if script_dist else 0.0
md += [
    "",
    f"Dominant script: **{script_dist[0][0]}** at {top_pct:.1f}% — "
    f"{'✅ within balanced expectation (<30%)' if top_pct < 30 else '⚠️ one script dominates'}",
    "",
    "Note: 'Latin-ASCII' will be inflated by tokens shared across many European languages",
    "(spa, fra, deu, ita, etc.) — combined Latin script share should still be the majority",
    "of FineWeb2-HQ since 16 of its 20 languages use Latin script.",
]

# Status
md += ["", "## Status", "", "Phase C complete. Decoded samples written to "
       "`C_decoded_samples.txt`; language distribution table above."]

with open(OUT_MD, "w") as f:
    f.write("\n".join(md) + "\n")

print(f"Phase C written: {OUT_MD}\n                 {OUT_DUMP}")
print("\nLanguage script distribution (FineWeb2-HQ 200-doc sample):")
for s, c in script_dist[:10]:
    print(f"  {s:<20} {c:>4} ({100*c/total:.1f}%)")
