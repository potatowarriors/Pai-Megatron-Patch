"""THD (packed-sequence) CP permutation — pure-index CPU validation.

Validates `build_thd_cp_a2a_perm` against an independent, readable reference
of the per-rank THD layout (each CP rank owns the load-balanced chunk pair
(r, 2*cp-1-r) of EVERY packed segment — the layout TE's
`thd_get_partitioned_indices` produces and `_all_to_all_cp2hp` concatenates
rank-by-rank). No GPU, no collectives: the permutation is index arithmetic.

    cd <worktree-root> && python -m pytest tests/test_gdn_thd_cp_perm.py -v
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(REPO_ROOT, "backends", "megatron", "Megatron-LM-251125")
for p in (REPO_ROOT, BACKEND):
    if p not in sys.path:
        sys.path.insert(0, p)

from megatron_patch.model.qwen3_next.gdn_context_parallel import (  # noqa: E402
    build_thd_cp_a2a_perm,
    resolve_cu_seqlens,
)


def reference_gathered_order(cu_seqlens, cp_size):
    """Global token ids in gathered (post-a2a) order, built the slow obvious way.

    Rank r's local slice holds, for each segment in order, that segment's
    load-balanced chunks 2r then 2r+1 — i.e. natural chunks r and 2*cp-1-r.
    The gathered tensor is rank 0's slice, then rank 1's, etc.
    """
    cu = cu_seqlens.tolist()
    order = []
    for r in range(cp_size):
        for s in range(len(cu) - 1):
            start, length = cu[s], cu[s + 1] - cu[s]
            half = length // (2 * cp_size)
            for natural in (r, 2 * cp_size - 1 - r):
                chunk_start = start + natural * half
                order.extend(range(chunk_start, chunk_start + half))
    return torch.tensor(order, dtype=torch.long)


CASES = [
    ("uniform", torch.tensor([0, 16, 32, 48]), 4),
    ("non_uniform", torch.tensor([0, 24, 32, 64, 96]), 4),
    ("single_segment", torch.tensor([0, 64]), 4),
    ("cp2", torch.tensor([0, 12, 20, 32]), 2),
    ("cp8_min_segments", torch.tensor([0, 16, 48]), 8),
]


@pytest.mark.parametrize("name,cu,cp", CASES, ids=[c[0] for c in CASES])
def test_perm_restores_natural_order(name, cu, cp):
    t = int(cu[-1])
    idx, inv = build_thd_cp_a2a_perm(cu, cp, t)
    gathered = reference_gathered_order(cu, cp)
    # index_select(0, idx) on the gathered tensor must yield natural order.
    assert torch.equal(gathered[idx], torch.arange(t))
    # inv maps natural order back to the gathered layout.
    assert torch.equal(torch.arange(t)[inv], gathered)


@pytest.mark.parametrize("name,cu,cp", CASES, ids=[c[0] for c in CASES])
def test_perm_inverse_round_trip(name, cu, cp):
    t = int(cu[-1])
    idx, inv = build_thd_cp_a2a_perm(cu, cp, t)
    assert torch.equal(idx[inv], torch.arange(t))
    assert torch.equal(inv[idx], torch.arange(t))


def test_single_segment_equals_global_zigzag_undo():
    """With one segment spanning the window, per-segment THD chunking equals the
    plain global zigzag, so idx must reproduce `_undo_attention_load_balancing`."""
    from megatron.core.ssm.mamba_context_parallel import _undo_attention_load_balancing

    cu, cp = torch.tensor([0, 64]), 4
    t = 64
    idx, _ = build_thd_cp_a2a_perm(cu, cp, t)
    gathered = reference_gathered_order(cu, cp).unsqueeze(-1).unsqueeze(-1).float()
    via_perm = gathered[idx]
    via_zigzag = _undo_attention_load_balancing(gathered, cp)
    assert torch.equal(via_perm, via_zigzag)


def test_perm_on_tensor_payload():
    """End-to-end on a random [T, b, c] payload: scatter per reference layout,
    concat, permute — must equal the original tensor."""
    torch.manual_seed(3)
    cu, cp = torch.tensor([0, 24, 32, 64, 96]), 4
    t = int(cu[-1])
    x = torch.randn(t, 2, 5)
    gathered = x[reference_gathered_order(cu, cp)]
    idx, inv = build_thd_cp_a2a_perm(cu, cp, t)
    assert torch.equal(gathered.index_select(0, idx), x)
    assert torch.equal(x.index_select(0, inv), gathered)


def test_resolve_cu_seqlens_prefers_padded_and_validates():
    padded = torch.tensor([0, 16, 32])
    actual = torch.tensor([0, 13, 30])
    out = resolve_cu_seqlens(padded, actual, 32, "cu_seqlens_q", cp_size=4)
    assert torch.equal(out, padded)

    with pytest.raises(ValueError, match="does not match total_sequence_length"):
        resolve_cu_seqlens(padded, actual, 40, "cu_seqlens_q", cp_size=4)


def test_resolve_cu_seqlens_rejects_non_divisible():
    # 2*cp_size = 8; a 12-length segment must be rejected (upstream's weaker
    # % cp_size check would wrongly pass it, then halves = 12 // 8 truncates).
    cu = torch.tensor([0, 12, 32])
    with pytest.raises(ValueError, match="divisible by 2\\*cp_size"):
        resolve_cu_seqlens(cu, None, 32, "cu_seqlens_q", cp_size=4)
    # The same cu_seqlens is fine at cp_size=2 (2*cp=4 divides 12 and 20).
    out = resolve_cu_seqlens(cu, None, 32, "cu_seqlens_q", cp_size=2)
    assert torch.equal(out, cu)
