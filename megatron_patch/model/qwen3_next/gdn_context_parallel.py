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
# Scope note: the upstream THD (packed-sequence) CP path is intentionally NOT
# ported yet — sequence packing with CP>1 is blocked at the data layer
# (megatron_patch/template/helper.py) and alpha pre-training does not pack.

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
