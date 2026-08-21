# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Alibaba PAI / Alpha Project Team.
#
# Context-parallel utilities for GatedDeltaNet, ported from upstream
# Megatron-LM main (NVIDIA/Megatron-LM PR #2642, merged 2026-04-13):
#   megatron/core/ssm/gated_delta_net/common.py
#
# Design ("all-to-all head-split" CP, same as Mamba2's MambaContextParallel):
# GDN is a sequential recurrence over the sequence, so instead of splitting the
# sequence, each CP rank converts its sequence shard into a head shard via
# all-to-all, processes the FULL sequence for its subset of heads, and converts
# back afterwards. The low-level collectives and the attention-load-balancing
# reorder are reused from megatron.core.ssm.mamba_context_parallel (present in
# Megatron-LM-251125); this module adds the layout-generic wrappers that the
# 251125 snapshot lacks.
#
# Scope note (updated 2026-08-21): the upstream THD (packed-sequence) CP path
# IS ported below (build_thd_cp_a2a_perm / a2a_cp_to_hp / a2a_hp_to_cp) — it is
# required for LC document isolation at CP>1 (dense reset-attention-mask is
# O(seq^2) and unusable at 32K; see docs/LC_ENTRY_GATE.md §1.5). The mixer
# stitch that feeds packed_seq_params into these wrappers lands when this
# branch merges with feature/gdn-varlen-thd; until then they are inert.

from functools import lru_cache
from typing import List, Optional

import torch

from megatron.core.ssm.mamba_context_parallel import (
    _all_to_all_cp2hp,
    _all_to_all_hp2cp,
    _redo_attention_load_balancing,
    _undo_attention_load_balancing,
)


@lru_cache(maxsize=8)
def build_head_perm_for_split_sections(
    split_sections: tuple, cp_size: int, device: torch.device
) -> torch.Tensor:
    """Build a channel permutation so ONE unsectioned all-to-all is equivalent
    to a per-section all-to-all.

    _all_to_all_cp2hp splits the channel (last) dim into cp_size equal chunks
    and sends chunk r to CP rank r. For a multi-section projection output
    (e.g. GDN's [z, V, Q, K, b, a]) the correct distribution is "each rank gets
    the r-th slice of EVERY section". Pre-permuting channels so that chunk r of
    the permuted tensor is [z_r | V_r | Q_r | K_r | b_r | a_r] achieves that
    with a single collective instead of len(split_sections) collectives.
    """
    assert all(
        s % cp_size == 0 for s in split_sections
    ), f"split_sections {split_sections} must be divisible by cp_size {cp_size} for GDN"
    offset = 0
    parts = []
    for s in split_sections:
        parts.append(
            torch.arange(offset, offset + s, device=device, dtype=torch.long).view(cp_size, -1)
        )
        offset += s

    return torch.cat(parts, dim=-1).view(-1)


def get_parameter_local_cp(
    param: torch.Tensor,
    dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
) -> torch.Tensor:
    """Get the local slice of a (replicated) parameter for the current CP rank.

    Slicing is done in the forward path so that gradients backpropagate into
    the full cp_size=1 parameter; the unused slices receive zero gradient on
    this rank and are filled in by the grad all-reduce over the dp-cp group.

    Args:
        param: The full parameter tensor.
        dim: The dimension to split along (usually the head dimension).
        cp_group: The context parallel process group.
        split_sections: If not None, first split `param` along `dim` into these
            sections, slice each section for this rank, then re-concatenate
            (used for the conv1d weight/bias covering [V, Q, K] channels).
    """
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()

    if cp_size == 1:
        return param

    if split_sections is not None:
        inputs = torch.split(param, split_sections, dim=dim)
        outputs = []
        for p in inputs:
            p = get_parameter_local_cp(p, dim, cp_group)
            outputs.append(p)
        return torch.cat(outputs, dim=dim)

    slices = [slice(None)] * param.dim()
    dim_size = param.size(dim=dim)
    slices[dim] = slice(cp_rank * dim_size // cp_size, (cp_rank + 1) * dim_size // cp_size)
    return param[slices]


def tensor_a2a_cp2hp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
    undo_attention_load_balancing: bool = True,
):
    """All-to-all: context-parallel (sequence-split) to hidden-parallel (head-split).

    [seq/cp, batch, channels] -> [seq, batch, channels/cp], and (by default)
    reorder the gathered sequence from the attention load-balanced chunk order
    back to natural order for sequential processing by conv/SSM.
    """
    cp_size = cp_group.size()

    if cp_size == 1:
        return tensor

    assert seq_dim == 0, f"tensor_a2a_cp2hp only supports seq_dim == 0 for now, but got {seq_dim=}"
    assert (
        head_dim == -1 or head_dim == 2
    ), f"tensor_a2a_cp2hp only supports head_dim == -1 or 2 for now, but got {head_dim=}"
    assert (
        tensor.dim() == 3
    ), f"tensor_a2a_cp2hp only supports 3-d input tensor for now, but got {tensor.dim()=}"

    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_cp2hp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                undo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_cp2hp(tensor, cp_group)

    if undo_attention_load_balancing:
        tensor = _undo_attention_load_balancing(tensor, cp_size)
    return tensor


def tensor_a2a_hp2cp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group: torch.distributed.ProcessGroup,
    split_sections: Optional[List[int]] = None,
    redo_attention_load_balancing: bool = True,
):
    """All-to-all: hidden-parallel (head-split) back to context-parallel (sequence-split).

    [seq, batch, channels/cp] -> [seq/cp, batch, channels], re-applying the
    attention load-balanced chunk order first so the output lines up with the
    layout the surrounding (attention) layers expect.
    """
    cp_size = cp_group.size()

    if cp_size == 1:
        return tensor

    assert seq_dim == 0, f"tensor_a2a_hp2cp only supports seq_dim == 0 for now, but got {seq_dim=}"
    assert (
        head_dim == -1 or head_dim == 2
    ), f"tensor_a2a_hp2cp only supports head_dim == -1 or 2 for now, but got {head_dim=}"
    assert (
        tensor.dim() == 3
    ), f"tensor_a2a_hp2cp only supports 3-d input tensor for now, but got {tensor.dim()=}"

    if redo_attention_load_balancing:
        tensor = _redo_attention_load_balancing(tensor, cp_size)

    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_hp2cp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                redo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_hp2cp(tensor, cp_group)

    return tensor


# ---------------------------------------------------------------------------
# THD (packed-sequence) CP path — port of the same upstream file's
# `_build_thd_cp_a2a_perm` / `a2a_cp_to_hp` / `a2a_hp_to_cp` (+ the
# cu_seqlens validation from `_GDNBase._resolve_cu_seqlens`).
#
# With sequence packing, the batch is sharded per *segment*: TE's
# `thd_get_partitioned_indices` gives each CP rank the load-balanced chunk
# pair (r, 2*cp-1-r) of EVERY packed segment, so the plain global
# `_undo_attention_load_balancing` (which assumes whole-sequence chunking)
# produces the wrong order after the cp2hp all-to-all. The permutation built
# here restores natural order per segment and folds the load-balancing undo
# into one index_select; its inverse is applied before the hp2cp all-to-all.
# ---------------------------------------------------------------------------


def resolve_cu_seqlens(cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name, cp_size=1):
    """Pick the padded cu_seqlens when present and validate it for the CP a2a.

    Deviation from upstream `_resolve_cu_seqlens`: upstream only validates
    `% cp_size == 0`, but `build_thd_cp_a2a_perm` computes per-segment
    half-chunks as `len // (2 * cp_size)` (and TE's
    `thd_get_partitioned_indices` makes the same assumption), so lengths not
    divisible by 2*cp_size would silently truncate. We enforce the real
    requirement.
    """
    cu_seqlens = cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens_actual

    total_cu = cu_seqlens[-1].cpu().item()
    if total_cu != total_seq_len:
        raise ValueError(
            f"GDN: {name}[-1]={total_cu} does not match total_sequence_length={total_seq_len}. "
            f"({cu_seqlens_padded=}, {cu_seqlens_actual=})."
        )

    if cp_size > 1:
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if (seq_lengths % (2 * cp_size) != 0).any():
            raise ValueError(
                f"All per-sequence lengths in cu_seqlens must be divisible by "
                f"2*cp_size={2 * cp_size} for the THD CP a2a, but got lengths: "
                f"{seq_lengths.tolist()}. Pack LC/SFT bins with per-document padding "
                f"(bestfit_pack.py --pad-doc-multiple) to satisfy this."
            )

    return cu_seqlens


def build_thd_cp_a2a_perm(cu_seqlens, cp_size, t_global):
    """Sequence-dim permutation (and inverse) for the THD CP all-to-all.

    After `_all_to_all_cp2hp` (without load-balancing undo), the gathered
    sequence dim is [rank0's local slice | rank1's | ...] where rank r's slice
    holds, for every packed segment, that segment's load-balanced chunk pair.
    `index_select(0, idx)` restores natural (original) token order;
    `index_select(0, inv)` maps back. Pure index arithmetic, upstream-verbatim.
    """
    cu = cu_seqlens.to(dtype=torch.long)
    t_local = t_global // cp_size

    positions = torch.arange(t_global, device=cu.device)
    seq_idx = torch.bucketize(positions, cu[1:], right=True)
    seq_lens = torch.diff(cu)
    halves = seq_lens // (2 * cp_size)  # per-sequence half-chunk size
    local_starts = cu[:-1] // cp_size
    global_starts = cu[:-1]

    half_i = halves[seq_idx]
    pos_in_seq = positions - global_starts[seq_idx]

    natural_chunk = pos_in_seq // half_i  # in [0, 2*cp)
    offset = pos_in_seq - natural_chunk * half_i

    # Invert the ordering produced by `_undo_attention_load_balancing`:
    #   natural_chunk < cp:   load_balanced = 2 * natural_chunk
    #   natural_chunk >= cp:  load_balanced = 4*cp - 2*natural_chunk - 1
    lb_chunk = torch.where(
        natural_chunk < cp_size, 2 * natural_chunk, 4 * cp_size - 2 * natural_chunk - 1
    )

    # In the per-sequence load-balanced layout each rank owns load-balanced
    # chunks (2r) and (2r+1), in that order, of every sequence.
    rank = lb_chunk // 2
    half_within_rank = lb_chunk - 2 * rank
    k = half_within_rank * half_i + offset

    idx = rank * t_local + local_starts[seq_idx] + k

    inv = torch.empty_like(idx)
    inv[idx] = positions

    return idx, inv


def a2a_cp_to_hp(
    qkvzba: torch.Tensor,
    in_proj_split_sections,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    cu_seqlens_q: Optional[torch.Tensor],
    seq_len: int,
    packed_seq_params,
):
    """GDN cp->hp all-to-all returning the inverse context for :func:`a2a_hp_to_cp`.

    Returns (hidden-parallel tensor, thd_cp_a2a_inv). The inverse permutation is
    None outside the thd + cp_size>1 case.
    """
    if cp_size > 1:
        # Pre-permute head dim so a single unsectioned a2a is equivalent to per-section a2a.
        head_perm = build_head_perm_for_split_sections(
            tuple(in_proj_split_sections), cp_size, qkvzba.device
        )
        qkvzba = qkvzba.index_select(-1, head_perm)

    thd_cp_a2a_inv = None
    if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
        qkvzba = tensor_a2a_cp2hp(
            qkvzba, seq_dim=0, head_dim=-1, cp_group=cp_group, undo_attention_load_balancing=False
        )
        if cp_size > 1:
            # Permute at the seq dim so that a single unsectioned a2a is
            # equivalent to per-sequence a2a. This also folds the
            # `_undo_attention_load_balancing` step.
            thd_cp_a2a_idx, thd_cp_a2a_inv = build_thd_cp_a2a_perm(cu_seqlens_q, cp_size, seq_len)
            qkvzba = qkvzba.index_select(0, thd_cp_a2a_idx)
    else:
        qkvzba = tensor_a2a_cp2hp(qkvzba, seq_dim=0, head_dim=-1, cp_group=cp_group)

    return qkvzba, thd_cp_a2a_inv


def a2a_hp_to_cp(
    norm_out: torch.Tensor,
    cp_size: int,
    cp_group: torch.distributed.ProcessGroup,
    packed_seq_params,
    thd_cp_a2a_inv: Optional[torch.Tensor],
) -> torch.Tensor:
    """GDN hp->cp all-to-all using the inverse context from :func:`a2a_cp_to_hp`."""
    if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
        if cp_size > 1:
            assert thd_cp_a2a_inv is not None
            norm_out = norm_out.index_select(0, thd_cp_a2a_inv)
        norm_out = tensor_a2a_hp2cp(
            norm_out, seq_dim=0, head_dim=-1, cp_group=cp_group, redo_attention_load_balancing=False
        )
    else:
        norm_out = tensor_a2a_hp2cp(norm_out, seq_dim=0, head_dim=-1, cp_group=cp_group)

    return norm_out
