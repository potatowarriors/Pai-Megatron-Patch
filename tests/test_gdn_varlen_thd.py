"""GDN varlen/THD packing — boundary arithmetic (CPU) + kernel equivalence (GPU).

CPU tests always run and gate the feature branch pre-merge.
GPU tests are the post-P3 validation payload (run on any idle GPU node):

    cd <repo-root> && python -m pytest tests/test_gdn_varlen_thd.py -v

Context: feature/gdn-varlen-thd — THD document isolation for LC 32k+ and SFT
sequence packing. Dense reset-attention-mask matrices scale O(seq^2) (1 GiB at
32k) and are replaced by cu_seqlens metadata consumed by FlashAttention (thd),
the fla gated-delta-rule kernel, and causal_conv1d's seq_idx.
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

CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# CPU: cu_seqlens_from_position_ids (megatron_patch/data/utils.py)
# ---------------------------------------------------------------------------

def _cu_seqlens_from_position_ids():
    # Import lazily: megatron_patch.data.utils pulls megatron.* which needs the
    # backend on sys.path (done above) but no GPU.
    from megatron_patch.data.utils import cu_seqlens_from_position_ids

    return cu_seqlens_from_position_ids


def test_cu_seqlens_multi_segment():
    fn = _cu_seqlens_from_position_ids()
    # segments of lengths 3, 2, 4
    pos = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3])
    cu, max_seqlen = fn(pos)
    assert cu.dtype == torch.int
    assert cu.tolist() == [0, 3, 5, 9]
    assert int(max_seqlen) == 4


def test_cu_seqlens_single_segment_no_interior_reset():
    # One document filling the whole window (the common LC bin case).
    # The pre-refactor inline code crashed here: seqlens.max() on empty tensor.
    fn = _cu_seqlens_from_position_ids()
    pos = torch.arange(16)
    cu, max_seqlen = fn(pos)
    assert cu.tolist() == [0, 16]
    assert int(max_seqlen) == 16


def test_cu_seqlens_trailing_pad_runs():
    # BFP pads bins with EOD; each pad token resets to a length-1 segment.
    fn = _cu_seqlens_from_position_ids()
    pos = torch.tensor([0, 1, 2, 0, 0, 0])
    cu, max_seqlen = fn(pos)
    assert cu.tolist() == [0, 3, 4, 5, 6]
    assert int(max_seqlen) == 3


# ---------------------------------------------------------------------------
# CPU: seq_idx_from_cu_seqlens (megatron_patch/model/qwen3_next/gated_deltanet.py)
# ---------------------------------------------------------------------------

def _seq_idx_from_cu_seqlens():
    from megatron_patch.model.qwen3_next.gated_deltanet import seq_idx_from_cu_seqlens

    return seq_idx_from_cu_seqlens


def test_seq_idx_expansion():
    fn = _seq_idx_from_cu_seqlens()
    cu = torch.tensor([0, 3, 5, 9])
    seq_idx = fn(cu, 9)
    assert seq_idx.dtype == torch.int32
    assert seq_idx.shape == (1, 9)
    assert seq_idx[0].tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]


def test_seq_idx_single_segment():
    fn = _seq_idx_from_cu_seqlens()
    seq_idx = fn(torch.tensor([0, 7]), 7)
    assert seq_idx[0].tolist() == [0] * 7


def test_seq_idx_rejects_mismatched_span():
    fn = _seq_idx_from_cu_seqlens()
    with pytest.raises(AssertionError):
        fn(torch.tensor([0, 3, 8]), 9)


def test_round_trip_position_ids_to_seq_idx():
    # position_ids -> cu_seqlens -> seq_idx must reproduce segment structure.
    cu_fn = _cu_seqlens_from_position_ids()
    idx_fn = _seq_idx_from_cu_seqlens()
    pos = torch.tensor([0, 1, 2, 3, 0, 1, 0, 1, 2])
    cu, _ = cu_fn(pos)
    seq_idx = idx_fn(cu, pos.shape[0])
    assert seq_idx[0].tolist() == [0, 0, 0, 0, 1, 1, 2, 2, 2]


# ---------------------------------------------------------------------------
# GPU: kernel-level varlen equivalence (post-P3 validation payload)
# ---------------------------------------------------------------------------

SEGS = [96, 160, 64]  # three packed sequences; chunk kernels want /64-friendly sizes
T = sum(SEGS)


def _cu(device):
    return torch.tensor(
        [0] + list(torch.tensor(SEGS).cumsum(0).tolist()), device=device, dtype=torch.long
    )


@pytest.mark.skipif(not CUDA, reason="needs GPU (fla triton kernel)")
def test_fla_varlen_matches_per_segment():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(7)
    H, DK, DV = 4, 64, 64
    dev = "cuda"
    q = torch.randn(1, T, H, DK, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, DK, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, DV, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(1, T, H, device=dev, dtype=torch.float32)
    beta = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)

    out_varlen, _ = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True, cu_seqlens=_cu(dev),
    )

    outs = []
    ofs = 0
    for L in SEGS:
        sl = slice(ofs, ofs + L)
        o, _ = chunk_gated_delta_rule(
            q[:, sl], k[:, sl], v[:, sl], g=g[:, sl], beta=beta[:, sl],
            initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
        )
        outs.append(o)
        ofs += L
    out_ref = torch.cat(outs, dim=1)

    torch.testing.assert_close(
        out_varlen.float(), out_ref.float(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.skipif(not CUDA, reason="needs GPU (causal_conv1d kernel)")
def test_causal_conv1d_seq_idx_matches_per_segment():
    from causal_conv1d import causal_conv1d_fn

    from megatron_patch.model.qwen3_next.gated_deltanet import seq_idx_from_cu_seqlens

    torch.manual_seed(11)
    D, W = 32, 4
    dev = "cuda"
    # seq_idx requires channel-last layout: (1, D, T) view over (1, T, D) storage
    # (matches the mixer's varlen branch).
    x = torch.randn(1, T, D, device=dev, dtype=torch.bfloat16).transpose(1, 2)
    w = torch.randn(D, W, device=dev, dtype=torch.bfloat16)
    b = torch.randn(D, device=dev, dtype=torch.bfloat16)
    seq_idx = seq_idx_from_cu_seqlens(_cu(dev), T).to(dev)

    out_varlen = causal_conv1d_fn(x=x, weight=w, bias=b, seq_idx=seq_idx, activation="silu")

    outs = []
    ofs = 0
    for L in SEGS:
        outs.append(
            causal_conv1d_fn(
                x=x[:, :, ofs:ofs + L].contiguous(), weight=w, bias=b, activation="silu"
            )
        )
        ofs += L
    out_ref = torch.cat(outs, dim=2)

    torch.testing.assert_close(out_varlen.float(), out_ref.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not CUDA, reason="needs GPU (fla triton kernel)")
def test_fla_varlen_actually_isolates_segments():
    """Negative control: without cu_seqlens, segment 2's output must differ,
    because recurrent state carries over from segment 1. Guards against a fla
    version silently ignoring the cu_seqlens kwarg."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(13)
    H, DK, DV = 2, 64, 64
    dev = "cuda"
    q = torch.randn(1, T, H, DK, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, DK, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, DV, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(1, T, H, device=dev, dtype=torch.float32) * 0.1  # slow decay → visible leak
    beta = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)

    kwargs = dict(g=g, beta=beta, initial_state=None, output_final_state=False,
                  use_qk_l2norm_in_kernel=True)
    out_packed, _ = chunk_gated_delta_rule(q, k, v, cu_seqlens=_cu(dev), **kwargs)
    out_leaky, _ = chunk_gated_delta_rule(q, k, v, cu_seqlens=None, **kwargs)

    seg2 = slice(SEGS[0], SEGS[0] + SEGS[1])
    diff = (out_packed[:, seg2].float() - out_leaky[:, seg2].float()).abs().max()
    assert diff > 1e-3, (
        "cu_seqlens had no effect on a downstream segment — varlen isolation "
        "is not active (fla version ignoring the kwarg?)"
    )


# ---------------------------------------------------------------------------
# CPU: merge_eod_pad_segments (megatron_patch/data/utils.py) — THD+CP prep
# ---------------------------------------------------------------------------

def _merge_eod_pad_segments():
    from megatron_patch.data.utils import merge_eod_pad_segments

    return merge_eod_pad_segments


def test_merge_absorbs_pad_run_after_doc():
    """Doc(len 5 incl. its EOD) + 3 pad EODs: position-id resets make the pads
    length-1 segments; merging restores one 8-aligned segment."""
    fn = _merge_eod_pad_segments()
    #        |-- doc: c c c c E --|  E  E  E  |-- doc2: c c c E --|
    tokens = torch.tensor([7, 8, 9, 5, 0, 0, 0, 0, 4, 6, 2, 0])
    cu = torch.tensor([0, 5, 6, 7, 8, 12], dtype=torch.int)
    out = fn(cu, tokens, eod_id=0)
    assert out.tolist() == [0, 8, 12]
    assert out.dtype == torch.int


def test_merge_keeps_content_segments():
    fn = _merge_eod_pad_segments()
    tokens = torch.tensor([7, 8, 0, 4, 6, 0])
    cu = torch.tensor([0, 3, 6])
    out = fn(cu, tokens, eod_id=0)
    assert out.tolist() == [0, 3, 6]  # nothing to merge


def test_merge_bin_tail_pad_into_last_doc():
    fn = _merge_eod_pad_segments()
    tokens = torch.tensor([7, 0, 3, 0, 0, 0])  # doc, doc, then 2 tail pads
    cu = torch.tensor([0, 2, 4, 5, 6])
    out = fn(cu, tokens, eod_id=0)
    assert out.tolist() == [0, 2, 6]


def test_merge_leading_all_eod_segment_kept():
    fn = _merge_eod_pad_segments()
    tokens = torch.tensor([0, 7, 8, 0])
    cu = torch.tensor([0, 1, 4])
    out = fn(cu, tokens, eod_id=0)
    assert out.tolist() == [0, 1, 4]  # nothing before it to merge into


def test_merge_round_trip_with_position_id_derivation():
    """End-to-end over the real derivation chain: padded-doc token stream ->
    GPTDataset-style position ids -> cu_seqlens -> merge -> M-aligned segments."""
    from megatron_patch.data.utils import cu_seqlens_from_position_ids

    fn = _merge_eod_pad_segments()
    M = 8
    # two docs padded to %8 (5+3 pads, 7+1 pad), then 8 tail pads to fill 24+8=32? keep 24
    tokens = torch.tensor(
        [1, 2, 3, 4, 0, 0, 0, 0,          # doc1 (5 incl EOD) + 3 pads -> 8
         5, 6, 7, 8, 9, 10, 0, 0]         # doc2 (7 incl EOD) + 1 pad  -> 8
    )
    # replicate GPTDataset reset: position restarts after every EOD
    pos = torch.zeros_like(tokens)
    p = 0
    for i, t in enumerate(tokens):
        pos[i] = p
        p = 0 if t == 0 else p + 1
    cu, _ = cu_seqlens_from_position_ids(pos)
    merged = fn(cu, tokens, eod_id=0)
    assert merged.tolist() == [0, 8, 16]
    seg = torch.diff(merged.to(torch.long))
    assert (seg % M == 0).all()


# ---------------------------------------------------------------------------
# CPU: snap_cu_seqlens_to_grid — 문서 내부 잡탕 EOD의 가짜 경계 제거 (THD+CP)
# LC-A iter 170 실사고 재현: 608토큰 문서가 내부 EOD 2개로 [318, 35, 255] 분열.
# ---------------------------------------------------------------------------


def test_snap_cu_seqlens_restores_split_document():
    import torch
    from megatron_patch.data.utils import snap_cu_seqlens_to_grid

    # 실사고 세그먼트 구성: 608×52, 318, 35, 255, 544 (총 32768)
    lens = [608] * 52 + [318, 35, 255, 544]
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0)), dtype=torch.int32)
    assert int(cu[-1]) == 32768
    snapped = snap_cu_seqlens_to_grid(cu, 16)
    seg = snapped[1:] - snapped[:-1]
    # 가짜 경계 2개(318·353 지점)만 제거되어 608 문서가 복원되고 전 세그먼트 %16
    assert (seg % 16 == 0).all(), seg.tolist()
    assert seg.tolist() == [608] * 53 + [544]
    assert int(snapped[-1]) == 32768


def test_snap_cu_seqlens_is_noop_on_clean_pad16():
    import torch
    from megatron_patch.data.utils import snap_cu_seqlens_to_grid

    lens = [48, 80, 16, 112, 32512]  # 전부 %16 (합 32768)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0)), dtype=torch.int32)
    snapped = snap_cu_seqlens_to_grid(cu, 16)
    assert torch.equal(snapped, cu)


def test_snap_cu_seqlens_rejects_non_pad16_packing():
    """pad16 아닌 패킹(예: SFT 임의 경계)을 CP>1 경로에 넣는 실수는 하드 에러."""
    import pytest
    import torch
    from megatron_patch.data.utils import snap_cu_seqlens_to_grid

    torch.manual_seed(0)
    lens = torch.randint(50, 500, (100,))          # 임의 길이 100개 대화
    lens[-1] += 32768 - int(lens.sum()) % 32768    # 총합만 32768로
    cu = torch.tensor([0] + list(torch.cumsum(lens, 0)), dtype=torch.int32)
    with pytest.raises(ValueError, match="does not look --pad-doc-multiple"):
        snap_cu_seqlens_to_grid(cu, 16)
