# Copyright (c) 2026, Alibaba PAI / Alpha Project Team.
#
# Equivalence tests for GatedDeltaNetMixer context parallelism
# (a2a head-split, ported from NVIDIA/Megatron-LM PR #2642).
#
# Usage:
#   # Single process: helper unit tests (+ optional bitwise CP=1 regression)
#   python tests/test_gdn_context_parallel.py
#
#   # 2-rank CP=2 vs CP=1 numerical equivalence (needs 2 GPUs)
#   torchrun --nproc_per_node=2 tests/test_gdn_context_parallel.py
#
# Env:
#   GDN_ORIG_FILE=<path to pre-CP gated_deltanet.py>
#       adds a bitwise CP=1 forward regression against the original
#       implementation (single-process mode only).
#   GDN_TEST_DTYPE=bf16|fp32   (default bf16, matching production)

import os
import sys

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Model dimensions for the toy GDN (small, but preserves alpha's structure:
# num_v_heads = 2 * num_k_heads so the GQA repeat_interleave path is active).
# ---------------------------------------------------------------------------
HIDDEN = 64
NUM_V_HEADS = 8
HEAD_V_DIM = 16
NUM_K_HEADS = 4
HEAD_K_DIM = 16
SEQ_LEN = 64  # must be divisible by 2 * cp_size
BATCH = 2
SEED = 1234

FWD_TOL = dict(rtol=5e-3, atol=5e-3)
GRAD_TOL = dict(rtol=2e-2, atol=2e-2)


def _dtype():
    return {"bf16": torch.bfloat16, "fp32": torch.float32}[
        os.environ.get("GDN_TEST_DTYPE", "bf16")
    ]


# ---------------------------------------------------------------------------
# Pure-function unit tests (no GPU / no distributed required)
# ---------------------------------------------------------------------------
def test_head_perm_chunk_layout():
    """Chunk r of the permuted channels must be [sec0_r | sec1_r | ... ]."""
    from megatron_patch.model.qwen3_next.gdn_context_parallel import (
        build_head_perm_for_split_sections,
    )

    for cp in (2, 4):
        sections = (16, 16, 8, 8, 4, 4)  # scaled-down [z, V, Q, K, b, a]
        perm = build_head_perm_for_split_sections(sections, cp, torch.device("cpu"))
        total = sum(sections)
        assert sorted(perm.tolist()) == list(range(total)), "perm must be a permutation"

        x = torch.arange(total)
        permuted = x[perm]
        chunks = permuted.chunk(cp)
        offsets = [0]
        for s in sections:
            offsets.append(offsets[-1] + s)
        for r in range(cp):
            expected = torch.cat(
                [
                    torch.arange(
                        offsets[i] + r * (sections[i] // cp),
                        offsets[i] + (r + 1) * (sections[i] // cp),
                    )
                    for i in range(len(sections))
                ]
            )
            assert torch.equal(chunks[r], expected), (cp, r)
    print("[ok] head_perm chunk layout")


def test_load_balancing_roundtrip():
    from megatron.core.ssm.mamba_context_parallel import (
        _redo_attention_load_balancing,
        _undo_attention_load_balancing,
    )

    for cp in (2, 4):
        x = torch.randn(8 * cp, 3, 5)
        y = _redo_attention_load_balancing(_undo_attention_load_balancing(x, cp), cp)
        assert torch.equal(x, y), f"load-balancing roundtrip failed for cp={cp}"
    print("[ok] load-balancing undo/redo roundtrip")


def test_cp1_helpers_are_identity():
    """With cp_size==1 every helper must return its input unchanged (same object),
    which is what guarantees the CP=1 training path is bit-identical to the
    pre-CP implementation."""
    from megatron_patch.model.qwen3_next.gdn_context_parallel import (
        get_parameter_local_cp,
        tensor_a2a_cp2hp,
        tensor_a2a_hp2cp,
    )

    class _FakeGroup:
        def size(self):
            return 1

        def rank(self):
            return 0

    g = _FakeGroup()
    x = torch.randn(4, 2, 8)
    assert tensor_a2a_cp2hp(x, 0, -1, g) is x
    assert tensor_a2a_hp2cp(x, 0, -1, g) is x
    p = torch.randn(8)
    assert get_parameter_local_cp(p, 0, g) is p
    print("[ok] cp_size==1 helpers are identity")


def test_get_parameter_local_cp_sections():
    from megatron_patch.model.qwen3_next.gdn_context_parallel import (
        get_parameter_local_cp,
    )

    class _FakeGroup:
        def __init__(self, size, rank):
            self._s, self._r = size, rank

        def size(self):
            return self._s

        def rank(self):
            return self._r

    # conv layout [V, Q, K] with V=8, Q=4, K=4; cp=2
    p = torch.arange(16.0)
    r0 = get_parameter_local_cp(p, 0, _FakeGroup(2, 0), split_sections=[8, 4, 4])
    r1 = get_parameter_local_cp(p, 0, _FakeGroup(2, 1), split_sections=[8, 4, 4])
    assert r0.tolist() == [0, 1, 2, 3, 8, 9, 12, 13], r0.tolist()
    assert r1.tolist() == [4, 5, 6, 7, 10, 11, 14, 15], r1.tolist()
    print("[ok] get_parameter_local_cp sectioned slicing")


# ---------------------------------------------------------------------------
# Mixer construction
# ---------------------------------------------------------------------------
def _build_mixer(pg_collection, dtype, module=None):
    from megatron.core.ssm.mamba_mixer import MambaMixerSubmodules
    from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

    from megatron_patch.model.qwen3_next.transformer_config import (
        Qwen3NextTransformerConfig,
    )

    if module is None:
        from megatron_patch.model.qwen3_next.gated_deltanet import GatedDeltaNetMixer
    else:
        GatedDeltaNetMixer = module.GatedDeltaNetMixer

    config = Qwen3NextTransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        head_k_dim=HEAD_K_DIM,
        head_v_dim=HEAD_V_DIM,
        num_k_heads=NUM_K_HEADS,
        num_v_heads=NUM_V_HEADS,
        params_dtype=dtype,
        bf16=(dtype == torch.bfloat16),
    )
    submodules = MambaMixerSubmodules(
        in_proj=ColumnParallelLinear, out_proj=RowParallelLinear
    )
    return GatedDeltaNetMixer(
        config,
        submodules,
        d_model=HIDDEN,
        pg_collection=pg_collection,
    )


def _make_pg_collection(tp_group, cp_group):
    from megatron.core.process_groups_config import ProcessGroupCollection

    try:
        return ProcessGroupCollection(tp=tp_group, cp=cp_group)
    except TypeError:
        pgc = ProcessGroupCollection()
        pgc.tp = tp_group
        pgc.cp = cp_group
        return pgc


# ---------------------------------------------------------------------------
# Single-process: optional bitwise CP=1 regression against the original file
# ---------------------------------------------------------------------------
def run_cp1_regression(orig_file):
    import importlib.util

    from megatron.core import parallel_state, tensor_parallel

    dtype = _dtype()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, context_parallel_size=1
    )
    tensor_parallel.model_parallel_cuda_manual_seed(SEED)

    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_group = parallel_state.get_context_parallel_group()
    pgc = _make_pg_collection(tp_group, cp_group)

    torch.manual_seed(SEED)
    new_mixer = _build_mixer(pgc, dtype)

    spec = importlib.util.spec_from_file_location("gdn_orig_module", orig_file)
    orig_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig_module)
    torch.manual_seed(SEED)
    orig_mixer = _build_mixer(pgc, dtype, module=orig_module)
    orig_mixer.load_state_dict(new_mixer.state_dict())

    x = torch.randn(
        SEQ_LEN, BATCH, HIDDEN, device="cuda", dtype=dtype, generator=None
    )
    with torch.no_grad():
        out_new, _ = new_mixer(x)
        out_orig, _ = orig_mixer(x)
    assert torch.equal(out_new, out_orig), (
        "CP=1 forward is NOT bit-identical to the original implementation! "
        f"max diff = {(out_new - out_orig).abs().max().item()}"
    )
    print("[ok] CP=1 forward bit-identical to original implementation")


# ---------------------------------------------------------------------------
# Distributed: CP=2 vs CP=1 forward/backward equivalence
# ---------------------------------------------------------------------------
def run_distributed_equivalence():
    from megatron.core import parallel_state, tensor_parallel

    dtype = _dtype()
    rank = dist.get_rank()
    world = dist.get_world_size()
    cp_size = world

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, context_parallel_size=cp_size
    )
    tensor_parallel.model_parallel_cuda_manual_seed(SEED)

    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_group = parallel_state.get_context_parallel_group()

    # Singleton groups for the CP=1 reference mixer (one per rank; new_group
    # must be called symmetrically on all ranks).
    self_groups = [dist.new_group([r]) for r in range(world)]
    self_group = self_groups[rank]

    pgc_cp = _make_pg_collection(tp_group, cp_group)
    pgc_ref = _make_pg_collection(self_group, self_group)

    torch.manual_seed(SEED)
    mixer_cp = _build_mixer(pgc_cp, dtype)
    # Force identical weights on all ranks.
    with torch.no_grad():
        for p in mixer_cp.parameters():
            dist.broadcast(p, src=0)
    mixer_ref = _build_mixer(pgc_ref, dtype)
    mixer_ref.load_state_dict(mixer_cp.state_dict())

    # Identical full-sequence input on all ranks.
    g = torch.Generator(device="cpu").manual_seed(SEED + 1)
    x_full = torch.randn(SEQ_LEN, BATCH, HIDDEN, generator=g, dtype=torch.float32)
    x_full = x_full.to(device="cuda", dtype=dtype)

    # Reference: full-sequence forward/backward at CP=1.
    x_ref = x_full.clone().requires_grad_(False)
    out_ref, _ = mixer_ref(x_ref)
    loss_ref = out_ref.float().sum()
    loss_ref.backward()

    # CP: each rank gets its load-balanced shard: chunks [r, 2*cp-1-r].
    chunks = x_full.chunk(2 * cp_size, dim=0)
    x_shard = torch.cat([chunks[rank], chunks[2 * cp_size - 1 - rank]], dim=0)
    out_cp, _ = mixer_cp(x_shard)
    loss_cp = out_cp.float().sum()
    loss_cp.backward()

    # ---- forward equivalence ----
    ref_chunks = out_ref.detach().chunk(2 * cp_size, dim=0)
    expected_shard = torch.cat(
        [ref_chunks[rank], ref_chunks[2 * cp_size - 1 - rank]], dim=0
    )
    torch.testing.assert_close(
        out_cp.detach().float(), expected_shard.float(), **FWD_TOL
    )
    max_diff = (out_cp.detach().float() - expected_shard.float()).abs().max().item()
    print(f"[rank {rank}] [ok] CP={cp_size} forward matches CP=1 (max diff {max_diff:.3e})")

    # ---- loss equivalence (sum over ranks == full-sequence loss) ----
    loss_sum = loss_cp.detach().clone()
    dist.all_reduce(loss_sum, group=cp_group)
    torch.testing.assert_close(loss_sum, loss_ref.detach(), rtol=1e-3, atol=1e-2)
    print(f"[rank {rank}] [ok] CP-summed loss matches full-sequence loss")

    # ---- gradient equivalence (grads all-reduced over the CP group, as the
    #      dp-cp gradient reduction would do in real training) ----
    named_cp = dict(mixer_cp.named_parameters())
    named_ref = dict(mixer_ref.named_parameters())
    checked = 0
    for name, p_cp in named_cp.items():
        if p_cp.grad is None:
            assert named_ref[name].grad is None, f"{name}: grad presence mismatch"
            continue
        g_cp = p_cp.grad.detach().float().clone()
        dist.all_reduce(g_cp, group=cp_group)
        g_ref = named_ref[name].grad.detach().float()
        torch.testing.assert_close(
            g_cp, g_ref, **GRAD_TOL, msg=lambda m, _n=name: f"grad mismatch for {_n}: {m}"
        )
        checked += 1
    assert checked >= 5, f"too few grads checked ({checked}) — wiring problem?"
    print(f"[rank {rank}] [ok] {checked} parameter grads match after CP all-reduce")


# ---------------------------------------------------------------------------
# Distributed: THD (packed) CP vs CP=1 forward/backward equivalence — the
# THD+CP stitch (a2a_cp_to_hp / a2a_hp_to_cp with per-segment permutation).
# ---------------------------------------------------------------------------
THD_SEQ_LEN = 256
# Uneven segment lengths, each % 16 == 0 -> divisible by 2*cp for cp in {2,4,8}.
THD_SEGMENTS = [48, 80, 16, 112]


def _thd_partition_index(cu, cp_size, cp_rank, device):
    """Reference per-segment partitioning: for EVERY segment this rank takes
    load-balanced natural chunk pair (r, 2cp-1-r), chunk size len/(2cp).
    Mirrors TE's thd_get_partitioned_indices (cross-checked below)."""
    idx = []
    bounds = cu.tolist()
    for s, e in zip(bounds[:-1], bounds[1:]):
        h = (e - s) // (2 * cp_size)
        a = s + cp_rank * h
        b = s + (2 * cp_size - 1 - cp_rank) * h
        idx.append(torch.arange(a, a + h, device=device))
        idx.append(torch.arange(b, b + h, device=device))
    return torch.cat(idx)


def run_distributed_thd_equivalence():
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.packed_seq_params import PackedSeqParams

    dtype = _dtype()
    rank = dist.get_rank()
    world = dist.get_world_size()
    cp_size = world

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, context_parallel_size=cp_size
        )
        tensor_parallel.model_parallel_cuda_manual_seed(SEED)

    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_group = parallel_state.get_context_parallel_group()
    self_groups = [dist.new_group([r]) for r in range(world)]
    self_group = self_groups[rank]

    pgc_cp = _make_pg_collection(tp_group, cp_group)
    pgc_ref = _make_pg_collection(self_group, self_group)

    torch.manual_seed(SEED + 7)
    mixer_cp = _build_mixer(pgc_cp, dtype)
    with torch.no_grad():
        for p in mixer_cp.parameters():
            dist.broadcast(p, src=0)
    mixer_ref = _build_mixer(pgc_ref, dtype)
    mixer_ref.load_state_dict(mixer_cp.state_dict())

    assert sum(THD_SEGMENTS) == THD_SEQ_LEN
    cu = torch.tensor(
        [0] + torch.cumsum(torch.tensor(THD_SEGMENTS), 0).tolist(),
        device="cuda",
        dtype=torch.int32,
    )
    max_seqlen = max(THD_SEGMENTS)

    # Cross-check our partition layout against TE's actual kernel helper — the
    # helper.py data path uses TE, the mixer's a2a perm assumes this layout.
    idx = _thd_partition_index(cu, cp_size, rank, device="cuda")
    try:
        import transformer_engine_torch as tex

        idx_te = tex.thd_get_partitioned_indices(cu, THD_SEQ_LEN, cp_size, rank).to(idx.device)
        assert torch.equal(idx_te.long(), idx.long()), (
            "TE thd_get_partitioned_indices layout differs from the layout the "
            "GDN THD a2a permutation assumes!"
        )
        if rank == 0:
            print("[ok] partition layout cross-checked against TE thd_get_partitioned_indices")
    except ImportError:
        pass

    g = torch.Generator(device="cpu").manual_seed(SEED + 11)
    x_full = torch.randn(THD_SEQ_LEN, 1, HIDDEN, generator=g, dtype=torch.float32)
    x_full = x_full.to(device="cuda", dtype=dtype)

    psp_ref = PackedSeqParams(
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        qkv_format="thd",
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )
    out_ref, _ = mixer_ref(x_full.clone(), packed_seq_params=psp_ref)
    loss_ref = out_ref.float().sum()
    loss_ref.backward()

    psp_cp = PackedSeqParams(
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        qkv_format="thd",
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )
    x_shard = x_full.index_select(0, idx)
    out_cp, _ = mixer_cp(x_shard, packed_seq_params=psp_cp)
    loss_cp = out_cp.float().sum()
    loss_cp.backward()

    # ---- forward equivalence (per-segment sharded slice of the reference) ----
    expected_shard = out_ref.detach().index_select(0, idx)
    torch.testing.assert_close(out_cp.detach().float(), expected_shard.float(), **FWD_TOL)
    max_diff = (out_cp.detach().float() - expected_shard.float()).abs().max().item()
    print(f"[rank {rank}] [ok] THD CP={cp_size} forward matches CP=1 (max diff {max_diff:.3e})")

    # ---- loss equivalence ----
    loss_sum = loss_cp.detach().clone()
    dist.all_reduce(loss_sum, group=cp_group)
    torch.testing.assert_close(loss_sum, loss_ref.detach(), rtol=1e-3, atol=1e-2)
    print(f"[rank {rank}] [ok] THD CP-summed loss matches full-sequence loss")

    # ---- gradient equivalence ----
    named_cp = dict(mixer_cp.named_parameters())
    named_ref = dict(mixer_ref.named_parameters())
    checked = 0
    for name, p_cp in named_cp.items():
        if p_cp.grad is None:
            assert named_ref[name].grad is None, f"{name}: grad presence mismatch"
            continue
        g_cp = p_cp.grad.detach().float().clone()
        dist.all_reduce(g_cp, group=cp_group)
        g_ref = named_ref[name].grad.detach().float()
        torch.testing.assert_close(
            g_cp, g_ref, **GRAD_TOL, msg=lambda m, _n=name: f"THD grad mismatch for {_n}: {m}"
        )
        checked += 1
    assert checked >= 5, f"too few grads checked ({checked}) — wiring problem?"
    print(f"[rank {rank}] [ok] THD: {checked} parameter grads match after CP all-reduce")


def main():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        try:
            run_distributed_equivalence()
            run_distributed_thd_equivalence()
            if dist.get_rank() == 0:
                print("ALL DISTRIBUTED TESTS PASSED")
        finally:
            dist.destroy_process_group()
        return

    # Single-process mode.
    test_head_perm_chunk_layout()
    test_load_balancing_roundtrip()
    test_cp1_helpers_are_identity()
    test_get_parameter_local_cp_sections()

    orig_file = os.environ.get("GDN_ORIG_FILE")
    if orig_file:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29511")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        try:
            torch.cuda.set_device(0)
            run_cp1_regression(orig_file)
        finally:
            dist.destroy_process_group()
    else:
        print("[skip] CP=1 bitwise regression (set GDN_ORIG_FILE to enable)")

    print("ALL SINGLE-PROCESS TESTS PASSED")


if __name__ == "__main__":
    main()
