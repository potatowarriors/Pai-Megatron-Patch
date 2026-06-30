"""Phase B (Stage 2 v5) — pre-tokenized dataset structural audit.

Adapted from tests/preflight_stage1/run_phase_b.py for the Stage-2 v5 blend
(math + 5 Nemotron code subsets + Nemotron CC-HQ actual/qa_pairs, plus the
reused korean_web). Same B1-B5 checks, with ONE key difference:

  Stage-2 data tokenized by run_stage2_v5.sh uses --append-eod, so EOS id 0 is
  appended at every doc end DIRECTLY (no remap step). B4 therefore expects ~100%
  of sampled docs to end in id 0 (asserts fraction > 0.99).

  NOTE: reused datasets (korean_web, fineweb2hq) were tokenized in a PRIOR
  session under the old eos designation (id 3 = <|im_end|>) and must be remapped
  3→0 with remap_eod.py before they pass B4 — see the Stage-1 precedent.

Robust audit: each dataset is isolated in try/except, so one failure records a
FAIL and the audit continues to the rest. Exits non-zero if any dataset fails or
is missing-but-expected. Datasets not yet tokenized are SKIPPED (not failed), so
this can run incrementally as run_stage2_v5.sh finishes each dataset.

Outputs:
  tests/preflight_stage2/B_dataset_integrity.md
  tests/preflight_stage2/B_<src>_report.json  (per source)
"""
import json
import os
import random
import struct
import sys
import time
import traceback

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "backends", "megatron", "Megatron-LM-251125"))

from megatron.core.datasets.indexed_dataset import IndexedDataset, DType

V5 = "/home/work/Datasets/LL_preprocessed/v5/stage2"
DATA = {
    "korean_web":            f"{V5}/korean_web/data_text_document",          # reused (remap 3→0 first)
    "fineweb2hq":            f"{V5}/fineweb2hq/data_text_document",          # reused 2nd-half (remap 3→0 first)
    "math":                  f"{V5}/math/data_text_document",
    "code_review":           f"{V5}/code/code_review/data_text_document",
    "question_answering":    f"{V5}/code/question_answering/data_text_document",
    "rewriting":             f"{V5}/code/rewriting/data_text_document",
    "student_teacher":       f"{V5}/code/student_teacher/data_text_document",
    "transpilation":         f"{V5}/code/transpilation/data_text_document",
    "cchq_actual":           f"{V5}/nemotron_cc_hq/actual/data_text_document",
    "cchq_qa_pairs":         f"{V5}/nemotron_cc_hq/qa_pairs/data_text_document",
}
OUT_DIR = os.path.join(REPO, "tests", "preflight_stage2")
EOD_ID = 0
EFFECTIVE_VOCAB = 163860
MAGIC = b"MMIDIDX\x00\x00"

random.seed(42)

from transformers import AutoTokenizer
hf = AutoTokenizer.from_pretrained(os.path.join(REPO, "examples", "alpha", "tokenizer_v5"),
                                   use_fast=True, trust_remote_code=False)


def audit_one(src, prefix):
    """Run B1-B5 on one dataset; return rep dict. Raises AssertionError on failure."""
    rep = {"source": src, "prefix": prefix}
    t0 = time.time()

    # B1 — header parsing
    idx_path, bin_path = prefix + ".idx", prefix + ".bin"
    with open(idx_path, "rb") as f:
        magic = f.read(9)
        version = struct.unpack("<Q", f.read(8))[0]
        dtype_code = struct.unpack("<B", f.read(1))[0]
        sequence_count = struct.unpack("<Q", f.read(8))[0]
        document_count = struct.unpack("<Q", f.read(8))[0]
    b1 = {"magic_match": magic == MAGIC, "version": version, "dtype_code": dtype_code,
          "dtype_name": DType(dtype_code).name, "sequence_count": sequence_count,
          "document_count": document_count, "idx_file_size": os.path.getsize(idx_path),
          "bin_file_size": os.path.getsize(bin_path)}
    assert b1["magic_match"], "MMIDIDX magic mismatch"
    assert b1["version"] == 1
    assert b1["dtype_code"] == 4, f"expected int32 (code 4), got {b1['dtype_code']}"
    rep["B1"] = b1

    ds = IndexedDataset(prefix)
    seq_lens = np.asarray(ds.sequence_lengths)

    # B2 — size consistency
    sum_lens = int(seq_lens.sum())
    expected = sum_lens * 4
    b2 = {"sequence_count_via_idx": int(seq_lens.shape[0]), "sum_sequence_lengths": sum_lens,
          "expected_bin_size_bytes": expected, "actual_bin_size_bytes": b1["bin_file_size"],
          "bin_size_match": expected == b1["bin_file_size"]}
    assert b2["bin_size_match"], "bin size != sum(seq_lens) * 4"
    rep["B2"] = b2

    # B3 — token-ID range (random 100MB chunks)
    bin_mm = np.memmap(bin_path, dtype=np.int32, mode="r")
    total = bin_mm.shape[0]
    chunk = 100 * 1024 * 1024 // 4
    rng = np.random.default_rng(42)
    max_start = max(0, total - chunk)
    starts = sorted(rng.integers(0, max_start + 1, size=16).tolist()) if max_start > 0 else [0]
    gmin, gmax, sampled, top = (1 << 31) - 1, -(1 << 31), 0, {}
    for s in starts:
        c = bin_mm[s:s + chunk]
        gmin, gmax = min(gmin, int(c.min())), max(gmax, int(c.max()))
        sampled += int(c.shape[0])
        u, cnt = np.unique(c, return_counts=True)
        for t, n in zip(u.tolist(), cnt.tolist()):
            top[t] = top.get(t, 0) + n
    top50 = sorted(top.items(), key=lambda kv: -kv[1])[:50]
    b3 = {"total_tokens_sampled": sampled, "global_min": gmin, "global_max": gmax,
          "max_under_effective_vocab": gmax < EFFECTIVE_VOCAB, "min_nonneg": gmin >= 0,
          "top_10_decoded": [{"id": t, "count": n, "decoded": hf.decode([t], skip_special_tokens=False)}
                             for t, n in top50[:10]]}
    del bin_mm
    assert b3["min_nonneg"] and b3["max_under_effective_vocab"], f"range out of bounds [{gmin},{gmax}]"
    rep["B3"] = b3

    # B4 — doc-boundary EOD audit (expect ~100% end in id 0)
    n_docs = int(seq_lens.shape[0])
    n_sample = min(10000, n_docs)
    end_eod, last_counter = 0, {}
    for i in random.sample(range(n_docs), n_sample):
        toks = ds.get(i)
        if len(toks) == 0:
            continue
        last = int(toks[-1])
        last_counter[last] = last_counter.get(last, 0) + 1
        if last == EOD_ID:
            end_eod += 1
    frac = end_eod / max(n_sample, 1)
    b4 = {"sampled_docs": n_sample, "docs_ending_in_eod_id_0": end_eod, "fraction": frac,
          "top_10_last_tokens": [{"id": t, "count": n, "decoded": hf.decode([t], skip_special_tokens=False)}
                                 for t, n in sorted(last_counter.items(), key=lambda kv: -kv[1])[:10]]}
    rep["B4"] = b4
    assert frac > 0.99, (f"expected ~100% docs ending in EOD id 0 (got {frac:.4f}). "
                         f"If reused/legacy data, run remap_eod.py --old-eod 3 --new-eod 0.")

    # B5 — empty / short docs
    b5 = {"n_docs": n_docs, "empty_zero_length": int((seq_lens == 0).sum()),
          "shorter_than_16": int((seq_lens < 16).sum()), "mean_doc_len": float(seq_lens.mean()),
          "median_doc_len": float(np.median(seq_lens)), "p95_doc_len": float(np.percentile(seq_lens, 95)),
          "max_doc_len": int(seq_lens.max())}
    assert b5["empty_zero_length"] == 0, "unexpected zero-length docs"
    rep["B5"] = b5

    rep["duration_sec"] = round(time.time() - t0, 2)
    return rep


all_reports, failures, skipped = {}, {}, []
for src, prefix in DATA.items():
    if not (os.path.exists(prefix + ".bin") and os.path.exists(prefix + ".idx")):
        print(f"\n##### {src}: SKIP (not tokenized yet) #####", flush=True)
        skipped.append(src)
        continue
    print(f"\n##### {src}: {prefix} #####", flush=True)
    try:
        rep = audit_one(src, prefix)
        all_reports[src] = rep
        b2, b4 = rep["B2"], rep["B4"]
        print(f"  ✓ PASS  tokens={b2['sum_sequence_lengths']/1e9:.2f}B  "
              f"EOD={b4['fraction']*100:.2f}%  ({rep['duration_sec']}s)", flush=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, f"B_{src}_report.json"), "w") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
    except Exception as e:
        failures[src] = str(e)
        print(f"  ✗ FAIL  {e}", flush=True)
        traceback.print_exc()

# Markdown summary
md = ["# Phase B (Stage 2 v5) — Pre-tokenized dataset structural audit", "",
      "EOD present in-stream (--append-eod, id 0); B4 asserts ~100% doc-end coverage.",
      f"PASS={len(all_reports)}  FAIL={len(failures)}  SKIP={len(skipped)}", ""]
for src, r in all_reports.items():
    b1, b2, b3, b4, b5 = r["B1"], r["B2"], r["B3"], r["B4"], r["B5"]
    md += [f"## {src}  ✓",
           f"- dtype `{b1['dtype_name']}` (code {b1['dtype_code']}), magic ok={b1['magic_match']}",
           f"- docs {b2['sequence_count_via_idx']:,} | tokens {b2['sum_sequence_lengths']:,} | bin=sum×4? {b2['bin_size_match']}",
           f"- B3 range [{b3['global_min']},{b3['global_max']}], max<{EFFECTIVE_VOCAB}? {b3['max_under_effective_vocab']}",
           f"- B4 EOD: {b4['docs_ending_in_eod_id_0']}/{b4['sampled_docs']} end in id 0 (frac {b4['fraction']:.4f})",
           f"- B5: empty={b5['empty_zero_length']}, <16={b5['shorter_than_16']:,}, mean={b5['mean_doc_len']:.0f}, max={b5['max_doc_len']:,}",
           f"- wall {r['duration_sec']}s", ""]
if failures:
    md += ["## FAILURES", ""] + [f"- **{s}**: {e}" for s, e in failures.items()] + [""]
if skipped:
    md += ["## SKIPPED (not tokenized yet)", "", ", ".join(skipped), ""]
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "B_dataset_integrity.md"), "w") as f:
    f.write("\n".join(md) + "\n")
print(f"\n{'='*60}\nPASS={len(all_reports)} FAIL={len(failures)} SKIP={len(skipped)}")
if failures:
    print("FAILED:", ", ".join(failures))
    sys.exit(1)
