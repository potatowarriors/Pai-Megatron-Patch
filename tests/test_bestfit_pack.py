"""Unit tests for toolkits/pretrain_data_preprocessing/bestfit_pack.py.

Covers the pure packing logic (no megatron needed) plus one end-to-end
round-trip on a real tiny IndexedDataset (skipped if torch/megatron absent).

Run:
    python -m pytest tests/test_bestfit_pack.py -v
"""
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRE = os.path.join(REPO, "toolkits", "pretrain_data_preprocessing")
MEGATRON = os.path.join(REPO, "backends", "megatron", "Megatron-LM-251125")
if PRE not in sys.path:
    sys.path.insert(0, PRE)

import bestfit_pack as bfp  # noqa: E402


# --------------------------------------------------------------------------
# Segment tree successor query
# --------------------------------------------------------------------------
def _brute_find_ge(counts, s):
    for v in range(s, len(counts)):
        if counts[v] > 0:
            return v
    return -1


def test_segtree_find_ge_matches_bruteforce():
    cap = 37
    rng = np.random.default_rng(0)
    seg = bfp._SegTreeSuccessor(cap)
    counts = [0] * (cap + 1)
    for _ in range(3000):
        v = int(rng.integers(0, cap + 1))
        # randomly add or (if present) remove a bin at remaining v
        if counts[v] > 0 and rng.random() < 0.4:
            counts[v] -= 1
            seg.add(v, -1)
        else:
            counts[v] += 1
            seg.add(v, 1)
        s = int(rng.integers(0, cap + 2))
        assert seg.find_ge(s) == _brute_find_ge(counts, s)


def test_segtree_find_ge_edges():
    seg = bfp._SegTreeSuccessor(10)
    assert seg.find_ge(0) == -1          # empty
    seg.add(5, 1)
    assert seg.find_ge(5) == 5
    assert seg.find_ge(6) == -1          # nothing >= 6
    assert seg.find_ge(0) == 5
    assert seg.find_ge(11) == -1         # beyond cap


# --------------------------------------------------------------------------
# Best-Fit-Decreasing: invariants + parity with a naive reference
# --------------------------------------------------------------------------
def _naive_bfd(item_lens, cap):
    """Same algorithm as bestfit_decreasing but with a linear successor scan.
    Identical bookkeeping/tie-break (LIFO within a remaining bucket) so the
    bin assignment must match the segment-tree version exactly."""
    n = item_lens.shape[0]
    item_bin = np.empty(n, dtype=np.int64)
    order = np.argsort(item_lens, kind="stable")[::-1]
    bins_at = [[] for _ in range(cap + 1)]
    bin_remaining = []
    num_bins = 0
    for it in order:
        s = int(item_lens[it])
        r = next((rr for rr in range(s, cap + 1) if bins_at[rr]), -1)
        if r == -1:
            b = num_bins; num_bins += 1
            bin_remaining.append(cap - s)
            item_bin[it] = b
            nr = cap - s
            if nr > 0:
                bins_at[nr].append(b)
        else:
            b = bins_at[r].pop()
            item_bin[it] = b
            nr = r - s
            bin_remaining[b] = nr
            if nr > 0:
                bins_at[nr].append(b)
    return item_bin, num_bins, bin_remaining


@pytest.mark.parametrize("seed", range(8))
def test_bfd_invariants_and_parity(seed):
    cap = 64
    rng = np.random.default_rng(seed)
    n = int(rng.integers(50, 400))
    item_lens = rng.integers(1, cap, size=n).astype(np.int64)  # all in [1, cap-1]

    item_bin, num_bins, bin_remaining = bfp.bestfit_decreasing(item_lens, cap, log_every=0)
    nb, nn, nr = _naive_bfd(item_lens, cap)

    # parity with naive reference (segment tree must give identical packing)
    assert num_bins == nn
    assert np.array_equal(item_bin, nb)
    assert bin_remaining == nr

    # every item assigned to a valid bin
    assert item_bin.min() >= 0 and item_bin.max() < num_bins
    # every bin's contents fit, and bin_remaining is exact
    fill = np.zeros(num_bins, dtype=np.int64)
    np.add.at(fill, item_bin, item_lens)
    assert (fill <= cap).all()
    assert np.array_equal(fill, cap - np.asarray(bin_remaining))
    # no empty bins
    assert (fill > 0).all()


def test_bfd_packs_better_than_one_per_bin():
    # 4 items of size 16 into cap 64 should pack into 1 bin, not 4.
    item_lens = np.array([16, 16, 16, 16], dtype=np.int64)
    item_bin, num_bins, _ = bfp.bestfit_decreasing(item_lens, 64, log_every=0)
    assert num_bins == 1
    assert (item_bin == 0).all()


# --------------------------------------------------------------------------
# build_pieces: coverage, long-doc splitting, full bins, leak count
# --------------------------------------------------------------------------
def _coverage_ok(pool, full, lengths):
    """Pieces must tile every doc exactly: total piece tokens == sum lengths,
    and per doc the offsets+lengths reconstruct [0, L)."""
    assert int(pool["len"].sum() + full["len"].sum()) == int(lengths.sum())
    spans = {}
    for arrs in (pool, full):
        for d, o, l in zip(arrs["doc"], arrs["off"], arrs["len"]):
            spans.setdefault(int(d), []).append((int(o), int(l)))
    for d, sp in spans.items():
        sp.sort()
        cur = 0
        for o, l in sp:
            assert o == cur, (d, sp)
            cur += l
        assert cur == int(lengths[d])


def test_build_pieces_default():
    cap = 100
    # small(50), exactly-cap(100), long(250 -> 2 full heads + tail 50),
    # long exact multiple(200 -> 2 full, last is doc-end), tiny(1)
    lengths = np.array([50, 100, 250, 200, 1], dtype=np.int64)
    pool, full, leak_chunks, n_empty = bfp.build_pieces(lengths, cap, strict_eod=False)
    assert n_empty == 0
    _coverage_ok(pool, full, lengths)

    # pool: doc0(50), doc4(1), tail of doc2(50)  -> 3 pieces all < cap
    assert pool["len"].shape[0] == 3
    assert (pool["len"] < cap).all()

    # full bins: doc1(1 piece, ends eod), doc2 heads(2, content), doc3 heads(2, last ends eod)
    assert full["len"].shape[0] == 5
    assert (full["len"] == cap).all()
    # leak = full chunks NOT ending in eod = doc2's 2 heads + doc3's first head = 3
    assert leak_chunks == 3
    # ends_eod true count = doc1 + doc3-last-head = 2
    assert int(full["ends_eod"].sum()) == 2


def test_build_pieces_strict_eod_has_no_full_bins():
    cap = 100
    lengths = np.array([50, 100, 250], dtype=np.int64)
    pool, full, leak_chunks, _ = bfp.build_pieces(lengths, cap, strict_eod=True)
    _coverage_ok(pool, full, lengths)
    # exactly-cap doc still becomes a full bin (ends in real EOD, leak-free);
    # the long doc is chunked into <= cap-1 pool pieces instead of full heads.
    assert (pool["len"] <= cap - 1).all()
    assert leak_chunks == 0          # nothing ends in content
    # long doc 250 -> ceil over (cap-1=99): 99+99+52 = 3 pool pieces
    long_pieces = [(d, l) for d, l in zip(pool["doc"], pool["len"]) if d == 2]
    assert sorted(l for _, l in long_pieces) == [52, 99, 99]


def test_build_pieces_skips_empty():
    cap = 50
    lengths = np.array([0, 10, 0, 20], dtype=np.int64)
    pool, full, leak_chunks, n_empty = bfp.build_pieces(lengths, cap, strict_eod=False)
    assert n_empty == 2
    assert int(pool["len"].sum()) == 30


# --------------------------------------------------------------------------
# baseline truncation estimate
# --------------------------------------------------------------------------
def test_estimate_baseline_truncations():
    cap = 10
    # cumulative boundaries: 6, 14, 20. cuts at 10, 20.
    #   cut@10 inside doc1 (6..14) -> truncation
    #   cut@20 == boundary 20    -> aligned, not a truncation
    lengths = np.array([6, 8, 6], dtype=np.int64)
    assert bfp.estimate_baseline_truncations(lengths, cap) == 1

    # perfectly aligned docs -> zero truncations
    assert bfp.estimate_baseline_truncations(np.array([10, 10, 10]), 10) == 0


# --------------------------------------------------------------------------
# End-to-end on a real tiny IndexedDataset (needs torch + megatron)
# --------------------------------------------------------------------------
@pytest.fixture
def tiny_dataset(tmp_path):
    pytest.importorskip("torch")
    if MEGATRON not in sys.path:
        sys.path.insert(0, MEGATRON)
    idx = pytest.importorskip("megatron.core.datasets.indexed_dataset",
                              reason="megatron.core not importable")
    from megatron.core.datasets import indexed_dataset

    cap, eod = 16, 0
    # docs each ending in EOD(0); mix of short, exactly-cap, and one long(>cap)
    rng = np.random.default_rng(3)
    docs = []
    for L in [5, 8, 16, 3, 7, 40, 2, 11, 9, 4]:
        body = rng.integers(1, 100, size=L - 1).astype(np.int32)  # non-eod content
        docs.append(np.concatenate([body, np.array([eod], dtype=np.int32)]))
    prefix = str(tmp_path / "in")
    builder = indexed_dataset.IndexedDatasetBuilder(prefix + "_text_document.bin",
                                                    dtype=np.int32)
    for d in docs:
        builder.add_document(d, [len(d)])
    builder.finalize(prefix + "_text_document.idx")
    return prefix, docs, cap, eod, indexed_dataset


def test_end_to_end_pack_roundtrip(tiny_dataset, tmp_path):
    in_prefix, docs, cap, eod, indexed_dataset = tiny_dataset
    out_prefix = str(tmp_path / "out")

    r = subprocess.run(
        [sys.executable, os.path.join(PRE, "bestfit_pack.py"),
         "--input", in_prefix, "--output", out_prefix,
         "--seq-length", str(cap), "--eod", str(eod), "--megatron-path", MEGATRON],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr

    out = indexed_dataset.IndexedDataset(out_prefix + "_text_document")
    lens = np.asarray(out.sequence_lengths)
    # every emitted bin is exactly L
    assert (lens == cap).all()
    # total real tokens preserved (emitted minus pad-eods... check via content)
    real = sum(len(d) for d in docs)
    assert int(lens.sum()) == len(out) * cap >= real

    # Reconstruct the multiset of documents from the packed bins and confirm
    # every original (short/exact) document appears intact, never split.
    # We detect doc boundaries by EOD and compare against originals.
    short_docs = [tuple(d.tolist()) for d in docs if len(d) <= cap]
    found = []
    for b in range(len(out)):
        seq = np.asarray(out[b])
        # split on EOD, keeping the EOD as terminator of each doc
        cur = []
        for tok in seq.tolist():
            cur.append(tok)
            if tok == eod:
                found.append(tuple(cur))
                cur = []
    # each short/exact doc must be present as an intact contiguous run
    found_multiset = found
    for sd in short_docs:
        assert sd in found_multiset, f"document was split or lost: {sd}"
