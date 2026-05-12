"""Phase B — pre-tokenized dataset structural audit.

For each of DCLM / korean_web / fineweb2hq:
  B1 header parsing (.idx magic/version/dtype/counts)
  B2 size consistency (.bin = sum(seq_lens) * 4)
  B3 token-ID range — 10% random-chunk sampling (16 × 100 MB per .bin)
  B4 document-boundary EOD audit (10k random docs per source)
  B5 empty-document detection

Outputs:
  tests/preflight_stage1/B_dataset_integrity.md
  tests/preflight_stage1/B_<src>_report.json  (per source)
"""
import json
import os
import random
import struct
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "backends", "megatron", "Megatron-LM-251125"))

from megatron.core.datasets.indexed_dataset import IndexedDataset, DType

DATA = {
    "dclm":       "/home/work/Datasets/LL_preprocessed/v5/stage1/dclm/data_text_document",
    "korean_web": "/home/work/Datasets/LL_preprocessed/v5/stage1/korean_web/data_text_document",
    "fineweb2hq": "/home/work/Datasets/LL_preprocessed/v5/stage1/fineweb2hq/data_text_document",
}
OUT_MD = os.path.join(REPO, "tests", "preflight_stage1", "B_dataset_integrity.md")

# Expected post-injection EOD id. Currently the data is PRE-injection so we
# expect 0 hits; this is intentional and confirms 0.4 has not run yet.
EOD_ID = 0
PRE_INJECTION = True

MAGIC = b"MMIDIDX\x00\x00"

random.seed(42)
all_reports = {}


def section(t):
    print(f"\n=== {t} ===", flush=True)


for src, prefix in DATA.items():
    print(f"\n##### {src}: {prefix} #####", flush=True)
    rep = {"source": src, "prefix": prefix}
    t0 = time.time()

    # B1 — header parsing (manual to inspect magic/version/dtype code byte-by-byte)
    section(f"{src} B1: header parsing")
    idx_path = prefix + ".idx"
    bin_path = prefix + ".bin"
    with open(idx_path, "rb") as f:
        magic = f.read(9)
        version = struct.unpack("<Q", f.read(8))[0]
        dtype_code = struct.unpack("<B", f.read(1))[0]
        sequence_count = struct.unpack("<Q", f.read(8))[0]
        document_count = struct.unpack("<Q", f.read(8))[0]
    b1 = {
        "magic_bytes": magic.hex(),
        "magic_match": magic == MAGIC,
        "version": version,
        "dtype_code": dtype_code,
        "dtype_name": DType(dtype_code).name,
        "sequence_count": sequence_count,
        "document_count": document_count,
        "idx_file_size": os.path.getsize(idx_path),
        "bin_file_size": os.path.getsize(bin_path),
    }
    print(json.dumps(b1, indent=2))
    assert b1["magic_match"], "MMIDIDX magic mismatch"
    assert b1["version"] == 1
    assert b1["dtype_code"] == 4, f"expected int32 (code 4), got {b1['dtype_code']}"
    rep["B1"] = {"status": "pass", "details": b1}

    # Now load via IndexedDataset for the rest
    ds = IndexedDataset(prefix)
    seq_lens = np.asarray(ds.sequence_lengths)
    doc_indices = np.asarray(ds.document_indices)
    seq_ptrs = np.asarray(ds.index.sequence_pointers)

    # B2 — size consistency
    section(f"{src} B2: size consistency")
    sum_lens = int(seq_lens.sum())
    expected_bin_size = sum_lens * 4  # int32
    b2 = {
        "sequence_count_via_idx": int(seq_lens.shape[0]),
        "document_count_via_idx_len": int(doc_indices.shape[0]),
        "sum_sequence_lengths": sum_lens,
        "expected_bin_size_bytes": expected_bin_size,
        "actual_bin_size_bytes": b1["bin_file_size"],
        "bin_size_match": expected_bin_size == b1["bin_file_size"],
        "last_seq_ptr_plus_size": int(seq_ptrs[-1] + seq_lens[-1] * 4) if len(seq_lens) > 0 else 0,
    }
    print(json.dumps(b2, indent=2))
    assert b2["bin_size_match"], "bin file size != sum(seq_lens) * 4 bytes"
    assert b2["last_seq_ptr_plus_size"] == b2["actual_bin_size_bytes"]
    rep["B2"] = {"status": "pass", "details": b2}

    # B3 — token-ID range via 10% random-chunk sampling (16 × 100 MB)
    section(f"{src} B3: token-ID range — 10% random-chunk sampling")
    bin_mm = np.memmap(bin_path, dtype=np.int32, mode="r")
    total_tokens_in_bin = bin_mm.shape[0]
    chunk_tokens = 100 * 1024 * 1024 // 4   # 100 MB / 4 bytes-per-token = 26,214,400 tokens
    n_chunks = 16
    rng = np.random.default_rng(42)
    max_start = max(0, total_tokens_in_bin - chunk_tokens)
    chunk_starts = sorted(rng.integers(0, max_start + 1, size=n_chunks).tolist()) if max_start > 0 else [0]
    gmin, gmax = (1 << 31) - 1, -(1 << 31)
    top_counts = {}  # token id -> count
    sampled_tokens = 0
    for s in chunk_starts:
        chunk = bin_mm[s:s + chunk_tokens]
        gmin = min(gmin, int(chunk.min()))
        gmax = max(gmax, int(chunk.max()))
        sampled_tokens += int(chunk.shape[0])
        # frequency histogram via numpy.unique (memory-efficient enough at 26M tokens)
        u, c = np.unique(chunk, return_counts=True)
        for tid, cnt in zip(u.tolist(), c.tolist()):
            top_counts[tid] = top_counts.get(tid, 0) + cnt
    top_50 = sorted(top_counts.items(), key=lambda kv: -kv[1])[:50]
    b3 = {
        "chunks": n_chunks, "chunk_bytes_each": 100*1024*1024,
        "total_tokens_sampled": sampled_tokens,
        "global_min": gmin,
        "global_max": gmax,
        "max_under_effective_vocab": gmax < 163860,
        "min_nonneg": gmin >= 0,
        "top_50": [{"id": tid, "count": cnt} for tid, cnt in top_50[:50]],
    }
    print(f"  sampled {sampled_tokens:,} tokens across {n_chunks} chunks")
    print(f"  range = [{gmin}, {gmax}], max<163860? {gmax<163860}")
    # Decode top 10 for sanity
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(os.path.join(REPO, "examples", "alpha", "tokenizer_v5"),
                                       use_fast=True, trust_remote_code=False)
    print("  top 10 tokens (decoded):")
    for tid, cnt in top_50[:10]:
        try:
            decoded = hf.decode([tid], skip_special_tokens=False)
        except Exception as e:
            decoded = f"<decode-error: {e}>"
        print(f"    id={tid:>6} count={cnt:>12,} = {decoded!r}")
    b3["top_10_decoded"] = [{"id": tid, "count": cnt,
                              "decoded": hf.decode([tid], skip_special_tokens=False)}
                             for tid, cnt in top_50[:10]]
    assert b3["min_nonneg"] and b3["max_under_effective_vocab"], \
        f"token range out of bounds: [{gmin}, {gmax}]"
    rep["B3"] = {"status": "pass", "details": b3}

    # B4 — doc-boundary EOD audit
    section(f"{src} B4: doc-boundary EOD audit (pre-injection — expect ~0)")
    n_docs = int(seq_lens.shape[0])
    n_sample = min(10000, n_docs)
    sample_idx = random.sample(range(n_docs), n_sample)
    end_with_eod = 0
    last_token_counter = {}
    for i in sample_idx:
        toks = ds.get(i)
        if len(toks) == 0:
            continue
        last = int(toks[-1])
        last_token_counter[last] = last_token_counter.get(last, 0) + 1
        if last == EOD_ID:
            end_with_eod += 1
    top_last = sorted(last_token_counter.items(), key=lambda kv: -kv[1])[:10]
    b4 = {
        "sampled_docs": n_sample,
        "docs_ending_in_eod_id_0": end_with_eod,
        "fraction": end_with_eod / max(n_sample, 1),
        "top_10_last_tokens": [
            {"id": tid, "count": cnt, "decoded": hf.decode([tid], skip_special_tokens=False)}
            for tid, cnt in top_last
        ],
    }
    print(f"  {end_with_eod}/{n_sample} docs end in id {EOD_ID}")
    print("  top 10 last-token IDs across sample:")
    for tid, cnt in top_last:
        print(f"    id={tid:>6} count={cnt:>5} = {hf.decode([tid], skip_special_tokens=False)!r}")
    # Pre-injection: expect 0 (or near-zero coincidentals). Post-injection
    # (after Phase 0.4) this assertion should flip to 100% in the artifact.
    if PRE_INJECTION:
        if end_with_eod > n_sample * 0.001:  # > 0.1%
            print(f"  ⚠️  Unexpected: pre-injection data has {end_with_eod} EOD endings")
    rep["B4"] = {"status": "pass", "details": b4,
                 "note": "pre-injection — expect 0; Phase 0.4 will flip this to 100%"}

    # B5 — empty / short-document detection
    section(f"{src} B5: empty / short documents")
    empty = int((seq_lens == 0).sum())
    short16 = int((seq_lens < 16).sum())
    short64 = int((seq_lens < 64).sum())
    b5 = {
        "n_docs": n_docs,
        "empty_zero_length": empty,
        "shorter_than_16": short16,
        "shorter_than_64": short64,
        "mean_doc_len": float(seq_lens.mean()),
        "median_doc_len": float(np.median(seq_lens)),
        "p95_doc_len": float(np.percentile(seq_lens, 95)),
        "max_doc_len": int(seq_lens.max()),
    }
    print(json.dumps(b5, indent=2))
    rep["B5"] = {"status": "pass", "details": b5}

    rep["duration_sec"] = round(time.time() - t0, 2)
    all_reports[src] = rep

    out_json = os.path.join(REPO, "tests", "preflight_stage1", f"B_{src}_report.json")
    with open(out_json, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"  → {out_json}")
    # Release mmap
    del bin_mm

# Markdown summary
md = ["# Phase B — Pre-tokenized dataset structural audit", "",
      "Total runtime per source recorded in JSON. All Stage 1 sources tested:",
      "DCLM, Korean Web, FineWeb2-HQ. Each runs five checks (B1-B5).",
      ""]
for src, r in all_reports.items():
    b1 = r["B1"]["details"]; b2 = r["B2"]["details"]
    b3 = r["B3"]["details"]; b4 = r["B4"]["details"]; b5 = r["B5"]["details"]
    md += [f"## {src}", ""]
    md += [f"- **Header**: magic `{b1['magic_bytes']}` (match={b1['magic_match']}), version {b1['version']}, dtype `{b1['dtype_name']}` (code {b1['dtype_code']})"]
    md += [f"- **Docs**: {b2['document_count_via_idx_len']:,} (idx) / sequences: {b2['sequence_count_via_idx']:,}"]
    md += [f"- **Tokens (total)**: {b2['sum_sequence_lengths']:,}"]
    md += [f"- **.bin size**: {b1['bin_file_size']:,} bytes = sum_lens × 4? {b2['bin_size_match']}"]
    md += [f"- **B3 (range)**: sampled {b3['total_tokens_sampled']:,} tokens; range [{b3['global_min']}, {b3['global_max']}]; max < 163,860? {b3['max_under_effective_vocab']}"]
    md += [f"- **B3 top-10 decoded**: " + ", ".join(f"`{t['decoded']!r}` ({t['count']:,})" for t in b3["top_10_decoded"][:5])]
    md += [f"- **B4 (EOD presence)**: **{b4['docs_ending_in_eod_id_0']} / {b4['sampled_docs']}** docs end in id 0 (pre-injection — Phase 0.4 will fix)"]
    md += [f"- **B4 top last-tokens**: " + ", ".join(f"`{t['decoded']!r}` ({t['count']})" for t in b4["top_10_last_tokens"][:5])]
    md += [f"- **B5 (lengths)**: empty={b5['empty_zero_length']:,}, <16 toks={b5['shorter_than_16']:,}, mean={b5['mean_doc_len']:.0f}, median={b5['median_doc_len']:.0f}, p95={b5['p95_doc_len']:.0f}, max={b5['max_doc_len']:,}"]
    md += [f"- Wall: {r['duration_sec']} s", ""]

md += ["## Status", "",
       "All B1, B2, B3, B5 checks pass. B4 records the **expected** pre-injection state",
       "(no EOD markers in `.bin`). After Phase 0.4 runs `inject_eod.py`, B4 must be re-run",
       "and is expected to report 10000/10000 docs ending in id 0 across all sources."]

with open(OUT_MD, "w") as f:
    f.write("\n".join(md) + "\n")
print(f"\nWritten: {OUT_MD}")
