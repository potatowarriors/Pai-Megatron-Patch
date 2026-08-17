"""Unit tests for DiLoCo data-shard mappings (examples/alpha/diloco_patch.py).

Covers the legacy sample-parity mapping (DILOCO_SHARD_BLOCK=0), the block-cyclic
mapping (DILOCO_SHARD_BLOCK=B, the 2026-08-17 blend-aliasing fix — see
examples/alpha/study/mirror_loss_aliasing.md), the disjoint-cover guarantee of
each, and the exactness of a parity->block switch at a block boundary.

Run from the repo root:
    python -m pytest tests/test_diloco_shard_view.py -v
No GPU / no torch.distributed needed — only the index arithmetic is exercised.
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backends/megatron/Megatron-LM-251125"))
sys.path.insert(0, str(REPO / "examples/alpha"))

WORLD = 2


def _install(rank, block, n_underlying_half):
    """Install the shard provider against a stub dataset; return the shard view."""
    os.environ["DILOCO_RANK"] = str(rank)
    os.environ["DILOCO_WORLD"] = str(WORLD)
    os.environ["DILOCO_SHARD_BLOCK"] = str(block)
    import megatron_patch.data as mpd
    import diloco_patch
    importlib.reload(diloco_patch)  # re-read env at module scope safely

    saved = mpd.train_valid_test_datasets_provider

    def stub_provider(sizes, *a, **k):
        # honours the world-multiplied train size the wrapper passes down
        return list(range(sizes[0])), None, None

    mpd.train_valid_test_datasets_provider = stub_provider
    try:
        diloco_patch._install_data_shard()
        provider = mpd.train_valid_test_datasets_provider
        train, valid, test = provider([n_underlying_half, 0, 0])
    finally:
        mpd.train_valid_test_datasets_provider = saved
    return train


def test_parity_mapping():
    n = 1000
    for rank in range(WORLD):
        view = _install(rank, 0, n)
        assert len(view) == n
        for j in (0, 1, 2, 499, 999):
            assert view[j] == WORLD * j + rank


def test_block_mapping():
    n, B = 1024, 4
    for rank in range(WORLD):
        view = _install(rank, B, n)
        L = WORLD * n
        assert len(view) == (L // (WORLD * B)) * B == n
        for j in (0, 1, 3, 4, 5, 8, 511, 1023):
            assert view[j] == WORLD * B * (j // B) + rank * B + j % B
        # first iteration-like block: rank r owns the r-th contiguous B-slice
        assert [view[j] for j in range(B)] == list(range(rank * B, (rank + 1) * B))


def test_block_len_floors_to_whole_blocks():
    # underlying length 2*n with n NOT a multiple of B: trailing partial block
    # must be dropped, otherwise rank 1 would index past the end
    n, B = 1030, 64
    views = [_install(r, B, n) for r in range(WORLD)]
    L = WORLD * n
    expect_len = (L // (WORLD * B)) * B
    for view in views:
        assert len(view) == expect_len
        assert view[len(view) - 1] < L  # in range


@pytest.mark.parametrize("block", [0, 4, 64])
def test_disjoint_exact_cover(block):
    n = 1024
    views = [_install(r, block, n) for r in range(WORLD)]
    covered = []
    for view in views:
        covered.extend(view[j] for j in range(len(view)))
    assert len(covered) == len(set(covered)), "duplication across ranks"
    span = len(covered)
    assert set(covered) == set(range(span)), "omission inside the covered prefix"


def test_parity_to_block_switch_is_exact_at_block_boundary():
    """Consuming [0, c0) under parity then [c0, end) under block-cyclic must
    tile the global range exactly once per pair — the mid-run switch rule."""
    n, B = 4096, 64
    c0 = 16 * B  # any multiple of B (an iteration boundary at constant GBS=B)
    L = WORLD * n
    pre = {WORLD * j + r for r in range(WORLD) for j in range(c0)}
    post = {WORLD * B * (j // B) + r * B + j % B
            for r in range(WORLD) for j in range(c0, n)}
    assert not (pre & post), "duplication across the switch"
    assert pre | post == set(range(L)), "omission across the switch"


def test_switch_off_boundary_would_break():
    """Documents WHY the consumed % B == 0 assert exists: off-boundary switches
    duplicate some samples and drop others."""
    n, B = 4096, 64
    c0 = 16 * B + 7  # NOT block-aligned
    pre = {WORLD * j + r for r in range(WORLD) for j in range(c0)}
    post = {WORLD * B * (j // B) + r * B + j % B
            for r in range(WORLD) for j in range(c0, n)}
    assert pre & post, "expected duplication off-boundary"
    assert pre | post != set(range(WORLD * n)), "expected omission off-boundary"


def test_getattr_passthrough():
    class DS(list):
        magic_attr = "megatron-needs-me"

    os.environ["DILOCO_RANK"] = "0"
    os.environ["DILOCO_WORLD"] = str(WORLD)
    os.environ["DILOCO_SHARD_BLOCK"] = "8"
    import megatron_patch.data as mpd
    import diloco_patch
    importlib.reload(diloco_patch)
    saved = mpd.train_valid_test_datasets_provider
    mpd.train_valid_test_datasets_provider = \
        lambda sizes, *a, **k: (DS(range(sizes[0])), None, None)
    try:
        diloco_patch._install_data_shard()
        train, _, _ = mpd.train_valid_test_datasets_provider([64, 0, 0])
    finally:
        mpd.train_valid_test_datasets_provider = saved
    assert train.magic_attr == "megatron-needs-me"
