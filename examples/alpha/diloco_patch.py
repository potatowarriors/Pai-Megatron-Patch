"""DiLoCo-lite outer loop for alpha 2-node training over a slow interconnect.

Each node runs an INDEPENDENT single-node Megatron instance (DP=8/EP=8, all
gradient traffic on NVLink). Every H inner steps, paired local ranks
(node0.local_rank_r <-> node1.local_rank_r) average their parameter deltas over
a private per-pair Gloo group and apply a Nesterov outer update.

v2 (after the validated 500-iter pilot, see study/diloco_pilot.md):
- Dense dedup: dense (non-expert) params are wire-synced by the local_rank-0
  pair ONLY (they are DP-identical within a node); results reach ranks 1-7 via
  an intra-node NCCL broadcast on the default group. Expert params (marked by
  Megatron with param.allreduce == False) are synced by every pair — each rank
  owns a distinct EP shard. Wire volume: 53 GB -> ~32 GB per sync.
- Overlapped sync (DILOCO_TAU > 0, Streaming-DiLoCo-style delayed apply,
  cf. facebookresearch/MuLoCo torchft fragment_sync_delay): the CPU-side
  snapshot is handed to a worker thread that allreduces and computes the outer
  update while training continues; the update is applied additively tau steps
  later (local progress made during the delay is kept). With tau=0 the original
  blocking semantics (bit-identical replicas after sync) are preserved.
  With tau>0 replicas legitimately differ by their tau-window local progress,
  so the divergence guard checks theta_global (deterministic CPU fp32) instead.

References: DiLoCo (arXiv:2311.08105), Streaming DiLoCo (arXiv:2501.18512),
MuLoCo — Muon validated as the inner optimizer (arXiv:2505.23725).

Env contract (set by the launcher):
  DILOCO_RANK            0|1  node index (0 hosts the TCPStores)
  DILOCO_WORLD           2
  DILOCO_MASTER          hostname of node 0
  DILOCO_PORT_BASE       base port; local_rank r uses base+r
  DILOCO_H               inner steps per outer sync (default 30)
  DILOCO_TAU             apply delay in inner steps, 0 = blocking (default 0)
  DILOCO_SHARD_BLOCK     shard mapping for DILOCO_DATA_SHARD=1: 0 = legacy
                         sample-parity world*i+r; B>0 = block-cyclic with
                         block B (use B = GBS, 3072 — kills the blend-index
                         parity aliasing, see study/mirror_loss_aliasing.md)
  DILOCO_OUTER_LR        default 0.7   (DiLoCo paper default)
  DILOCO_OUTER_MOMENTUM  default 0.6   (MuLoCo best for Muon inner; was 0.9 DiLoCo default)
  DILOCO_SKIP_SAVE       1 = disable checkpoint saving (pilot escape hatch)

The outer state (theta_global + momentum, fp32) lives on CPU.
"""
import datetime
import os
import threading
import time

import torch
import torch.distributed as dist

_state = None


class _State:
    def __init__(self):
        self.node_rank = int(os.environ["DILOCO_RANK"])
        self.world = int(os.environ.get("DILOCO_WORLD", "2"))
        self.master = os.environ.get("DILOCO_MASTER", "main1")
        self.port_base = int(os.environ.get("DILOCO_PORT_BASE", "31000"))
        self.H = int(os.environ.get("DILOCO_H", "30"))
        self.tau = int(os.environ.get("DILOCO_TAU", "0"))
        self.outer_lr = float(os.environ.get("DILOCO_OUTER_LR", "0.7"))
        self.outer_momentum = float(os.environ.get("DILOCO_OUTER_MOMENTUM", "0.6"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        assert 0 <= self.tau < self.H, "DILOCO_TAU must be in [0, H)"
        self.pg = None
        self.bias_pg = None     # dedicated pair group for per-step MoE bias-count sync
                                # (separate from self.pg: the tau-overlap wire thread uses
                                # self.pg concurrently, and Gloo matches collectives by
                                # call order per group — sharing would interleave & corrupt)
        self.bias_bufs = None   # router expert_bias buffers (24 x [192], fp32, CUDA)
        self.wire_dtype = torch.bfloat16
        self.params = None      # all named params (initial broadcast)
        self.owned = None       # params THIS rank wire-syncs (experts [+ dense on lr0])
        self.dense = None       # dense params (intra-node broadcast targets)
        self.theta = None       # cpu fp32 globals, aligned with owned
        self.momentum = None
        self.inner_step = 0
        self.outer_step_count = 0
        self.pending = None     # in-flight sync: dict(snap, thread, stats, apply_at)
        self.scratch = {}       # per-dtype persistent GPU staging buffers (apply path)


def _log(st, msg):
    if st.local_rank == 0:
        print(f"[diloco][node{st.node_rank}] {msg}", flush=True)


def _bcast_pair(pg, t, root):
    opts = dist.BroadcastOptions()
    opts.rootRank = root
    opts.rootTensor = 0
    pg.broadcast([t], opts).wait()


def _allreduce_sum(pg, t):
    opts = dist.AllreduceOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    pg.allreduce([t], opts).wait()


def _pair_minmax_equal(pg, value):
    lo = torch.tensor([value], dtype=torch.float64)
    hi = lo.clone()
    opts = dist.AllreduceOptions(); opts.reduceOp = dist.ReduceOp.MIN
    pg.allreduce([lo], opts).wait()
    opts = dist.AllreduceOptions(); opts.reduceOp = dist.ReduceOp.MAX
    pg.allreduce([hi], opts).wait()
    return bool(torch.equal(lo, hi)), lo.item(), hi.item()


def _make_pair_group(st):
    port = st.port_base + st.local_rank
    store = dist.TCPStore(
        st.master, port, st.world, st.node_rank == 0,
        timeout=datetime.timedelta(minutes=30),
    )
    try:
        pg = dist.ProcessGroupGloo(store, st.node_rank, st.world,
                                   datetime.timedelta(minutes=60))
    except TypeError:
        pg = dist.ProcessGroupGloo(store, st.node_rank, st.world)
    try:
        probe = torch.zeros(4, dtype=torch.bfloat16)
        _allreduce_sum(pg, probe)
        st.wire_dtype = torch.bfloat16
    except Exception:
        st.wire_dtype = torch.float32
    st.pg = pg
    _log(st, f"pair group up (port {port}, wire dtype {st.wire_dtype})")


def _make_bias_group(st):
    """Second Gloo pair group dedicated to the per-step expert-bias count sync.

    Must NOT share st.pg: the tau-overlap worker thread issues long sequences of
    collectives on st.pg while training steps (and their bias syncs) proceed on
    the main thread — Gloo matches ops per group by call order, so two threads
    on one group would interleave differently on the two nodes and corrupt both.
    Port offset +100 keeps clear of the main pair ports (base+local_rank, 0..7).
    """
    port = st.port_base + 100 + st.local_rank
    store = dist.TCPStore(
        st.master, port, st.world, st.node_rank == 0,
        timeout=datetime.timedelta(minutes=30),
    )
    try:
        pg = dist.ProcessGroupGloo(store, st.node_rank, st.world,
                                   datetime.timedelta(minutes=60))
    except TypeError:
        pg = dist.ProcessGroupGloo(store, st.node_rank, st.world)
    st.bias_pg = pg
    _log(st, f"bias-sync pair group up (port {port})")


def _named_params(model_chunks):
    out = []
    for chunk in model_chunks:
        for name, p in chunk.named_parameters():
            if p.requires_grad:
                out.append((name, p))
    return out


def _outer_state_path(save_dir, iteration, node_rank, local_rank):
    return os.path.join(save_dir, "diloco_outer", f"iter_{iteration:07d}",
                        f"node{node_rank}_rank{local_rank}.pt")


def _save_outer_state(st, iteration):
    """Persist theta_global + outer momentum next to the Megatron checkpoint."""
    from megatron.training import get_args
    args = get_args()
    if not getattr(args, "save", None) or st is None or st.theta is None:
        return
    t0 = time.time()
    path = _outer_state_path(args.save, iteration, st.node_rank, st.local_rank)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"names": [n for n, _ in st.owned], "theta": st.theta,
                "momentum": st.momentum, "H": st.H, "tau": st.tau}, path)
    _log(st, f"outer state saved for iter {iteration} ({time.time() - t0:.1f}s)")


def _try_load_outer_state(st):
    """On resume (--load), restore theta/momentum from the latest saved state."""
    from megatron.training import get_args
    args = get_args()
    load_dir = getattr(args, "load", None)
    if not load_dir:
        return False
    base = os.path.join(load_dir, "diloco_outer")
    if not os.path.isdir(base):
        _log(st, "resume WITHOUT diloco outer state (none found) — "
                 "theta:=params, outer momentum resets (re-warms in a few syncs)")
        return False
    latest = sorted(os.listdir(base))[-1]
    path = os.path.join(base, latest, f"node{st.node_rank}_rank{st.local_rank}.pt")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert blob["names"] == [n for n, _ in st.owned], \
        "diloco outer state param layout mismatch — wrong checkpoint or config"
    st.theta = blob["theta"]
    st.momentum = blob["momentum"]
    _log(st, f"outer state restored from {latest} (H was {blob['H']}, tau {blob['tau']})")
    return True


def _setup(st, model, optimizer):
    """First-call init: pair group, cross-node weight broadcast, snapshots."""
    t0 = time.time()
    _make_pair_group(st)
    _make_bias_group(st)
    st.params = _named_params(model)

    # Collect the MoE routers' aux-loss-free expert_bias buffers. They are NOT
    # parameters (register_buffer), so the outer average never touches them; the
    # per-step count sync (see _install_bias_count_sync) keeps them bit-identical
    # across nodes, and _apply_outer verifies that with a pair checksum.
    st.bias_bufs = []
    for chunk in model:
        for m in chunk.modules():
            b = getattr(m, "expert_bias", None)
            if b is not None:
                st.bias_bufs.append(b)
    _log(st, f"expert_bias buffers tracked: {len(st.bias_bufs)}")
    if st.bias_bufs:
        bsum = sum(float(b.detach().sum(dtype=torch.float64)) for b in st.bias_bufs)
        ok, lo, hi = _pair_minmax_equal(st.bias_pg, bsum)
        _log(st, f"initial expert_bias checksum {bsum:.9f} "
                 f"(pair equal: {ok}, lo={lo:.9f} hi={hi:.9f})")

    # Megatron marks EP-sharded expert params with allreduce=False; dense params
    # (attention, GDN, embeddings, router, shared experts) are DP-replicated
    # within a node, so one wire pair suffices for them.
    expert = [(n, p) for n, p in st.params if not getattr(p, "allreduce", True)]
    st.dense = [(n, p) for n, p in st.params if getattr(p, "allreduce", True)]
    st.owned = expert + (st.dense if st.local_rank == 0 else [])

    meta = torch.tensor([len(st.owned), sum(p.numel() for _, p in st.owned)],
                        dtype=torch.float64)
    ok, lo, hi = _pair_minmax_equal(st.pg, float(meta.sum()))
    assert ok, f"param layout mismatch across nodes: {lo} != {hi}"

    if os.environ.get("DILOCO_DATA_SHARD", "0") == "1":
        # disjoint sharding requires ONE shared global order -> identical seeds
        from megatron.training import get_args
        args = get_args()
        ok, lo, hi = _pair_minmax_equal(st.pg, float(args.seed))
        assert ok, (f"DILOCO_DATA_SHARD=1 requires the SAME --seed on both nodes "
                    f"(shared global order), got {lo} vs {hi}")
        # both nodes must use the SAME shard mapping — a mixed pair would
        # duplicate part of the corpus and permanently drop the rest
        block = int(os.environ.get("DILOCO_SHARD_BLOCK", "0"))
        ok, lo, hi = _pair_minmax_equal(st.pg, float(block))
        assert ok, (f"DILOCO_SHARD_BLOCK differs across nodes: {lo} vs {hi} — "
                    f"launch_diloco.sh must forward it via ENVV to both nodes")
        if block > 0:
            # a parity->block switch is exact (0 duplication / 0 omission) ONLY
            # when resuming at a block boundary: consumed % B == 0, i.e. any
            # iteration boundary while GBS == B (verified in
            # study/mirror_loss_repro.py; a fresh start trivially satisfies 0 % B == 0)
            consumed = int(getattr(args, "consumed_train_samples", 0) or 0)
            assert consumed % block == 0, (
                f"DILOCO_SHARD_BLOCK={block} needs consumed_train_samples % "
                f"block == 0, got {consumed} — resume at an iteration boundary "
                f"with constant GBS == block, or unset DILOCO_SHARD_BLOCK")
            _log(st, f"data shard mapping: block-cyclic B={block} "
                     f"(consumed {consumed} block-aligned)")

    if _try_load_outer_state(st):
        # resumed: each node loaded its own Megatron checkpoint; do NOT broadcast
        # params (would erase node1's resumed replica / tau-window state). Just
        # verify both nodes restored the same theta_global.
        s = sum(float(t.sum(dtype=torch.float64)) for t in st.theta)
        ok, lo, hi = _pair_minmax_equal(st.pg, s)
        assert ok, f"resumed theta_global differs across nodes: {lo} != {hi}"
        # Checkpoint load (incl. full optimizer state) fills the GPU before the
        # LAZILY-initialized NCCL groups grab their cudaMallocAsync pool pages —
        # the first MoE all_reduce then dies with "Failed to CUDA calloc async
        # N bytes" (N tiny; the pool cannot get pages, and it is separate from
        # PyTorch's allocator so empty_cache alone does not help). Warm every
        # parallel group NOW: activations don't exist yet, so this is the
        # lowest-memory point after load. Fresh runs never hit this because
        # optimizer state materializes only after the first forward.
        torch.cuda.empty_cache()
        from megatron.core import mpu
        warm = torch.ones(1, device="cuda")
        for getter in ("get_data_parallel_group", "get_tensor_model_parallel_group",
                       "get_pipeline_model_parallel_group", "get_model_parallel_group",
                       "get_expert_model_parallel_group", "get_expert_tensor_parallel_group",
                       "get_expert_data_parallel_group", "get_context_parallel_group"):
            try:
                g = getattr(mpu, getter)()
                dist.all_reduce(warm.clone(), group=g)
            except Exception:
                pass
        # also warm the coalesced-allreduce path (the bucketed grad sync uses
        # torch.distributed._coalescing_manager, which performs its own lazy
        # NCCL allocations — rt_b3 crashed exactly in its __exit__). Megatron's
        # start_grad_sync() asserts outside fwd-bwd, so mimic the path directly
        # with tiny tensors on the same groups.
        from torch.distributed import _coalescing_manager
        for getter in ("get_data_parallel_group", "get_expert_data_parallel_group"):
            try:
                g = getattr(mpu, getter)()
                with _coalescing_manager(group=g, async_ops=False):
                    for _ in range(2):
                        dist.all_reduce(warm.clone(), group=g)
            except Exception as e:
                _log(st, f"coalesced warmup ({getter}) skipped: {type(e).__name__}")
        torch.cuda.synchronize()
        _log(st, "resume: NCCL parallel groups + coalesced-allreduce path warmed")
    else:
        # fresh start. If both nodes already hold bit-identical params (e.g. the
        # 2026-08-11 bias-fix restart: node1 resumes from a copy of node0's
        # checkpoint), skip the broadcast AND the reload_model_params() — the
        # reload would re-derive the Muon fp32 masters from bf16 params and
        # silently discard the precise masters loaded from the checkpoint.
        s = sum(float(p.data.detach().sum(dtype=torch.float64)) for _, p in st.params)
        identical, lo, hi = _pair_minmax_equal(st.pg, s)
        if identical:
            _log(st, "fresh start: params already pair-identical — "
                     "broadcast/reload skipped (fp32 masters preserved)")
        else:
            # broadcast node0 weights so both replicas start bit-identical
            # (D2H in the param's own dtype, cast on CPU — GPU-side dtype temps
            # fragment the allocator and break torch_dist save)
            with torch.no_grad():
                for _, p in st.params:
                    buf = p.data.detach().cpu().to(torch.float32)
                    _bcast_pair(st.pg, buf, root=0)
                    p.data.copy_(buf.to(p.dtype))
            torch.cuda.empty_cache()
            optimizer.reload_model_params()
        st.theta = [p.data.detach().cpu().to(torch.float32) for _, p in st.owned]
        st.momentum = [torch.zeros_like(t) for t in st.theta]
    _log(st, f"setup done in {time.time() - t0:.1f}s: owned {len(st.owned)} params "
             f"({sum(p.numel() for _, p in st.owned) / 1e9:.2f}B; dense dedup: "
             f"{len(st.dense)} dense on lr0 pair only), H={st.H}, tau={st.tau}, "
             f"outer lr={st.outer_lr} mu={st.outer_momentum}")


def _worker(st, snap, stats):
    """Off-thread: allreduce pseudo-gradients, update theta/momentum (CPU fp32)."""
    t0 = time.time()
    mu = st.outer_momentum
    delta_sq, numel = 0.0, 0
    for (name, p), th, m, sn in zip(st.owned, st.theta, st.momentum, snap):
        delta = th - sn                          # pseudo-gradient
        wire = delta.to(st.wire_dtype)
        _allreduce_sum(st.pg, wire)
        delta_avg = wire.to(torch.float32).div_(st.world)
        m.mul_(mu).add_(delta_avg)
        th.sub_(delta_avg.add_(m, alpha=mu), alpha=st.outer_lr)   # Nesterov
        delta_sq += float(delta_avg.pow(2).sum())
        numel += delta_avg.numel()
    s = sum(float(t.sum(dtype=torch.float64)) for t in st.theta)
    stats["in_sync"], stats["lo"], stats["hi"] = _pair_minmax_equal(st.pg, s)
    stats["rms"] = (delta_sq / max(numel, 1)) ** 0.5
    stats["wire_s"] = time.time() - t0


def _start_outer(st):
    t0 = time.time()
    with torch.no_grad():
        snap = [p.data.detach().cpu().to(torch.float32) for _, p in st.owned]
    stats = {"snap_s": time.time() - t0}
    th = threading.Thread(target=_worker, args=(st, snap, stats), daemon=True)
    th.start()
    st.pending = {"snap": snap, "thread": th, "stats": stats,
                  "apply_at": st.inner_step + st.tau}


def _apply_outer(st, optimizer):
    pend = st.pending
    st.pending = None
    pend["thread"].join()
    stats = pend["stats"]
    t0 = time.time()
    with torch.no_grad():
        for (name, p), th, sn in zip(st.owned, st.theta, pend["snap"]):
            if st.tau == 0:
                p.data.copy_(th.to(p.dtype))     # exact, replicas bit-identical
            else:
                # additive correction keeps the tau-window local progress.
                # Stage through a persistent per-dtype GPU buffer: per-apply GPU
                # temporaries fragmented the allocator over ~100 applies and
                # broke the torch_dist save's NCCL gather (2026-07-14).
                corr = (th - sn).to(p.dtype).view(-1)          # CPU, param dtype
                buf = st.scratch.get(p.dtype)
                if buf is None or buf.numel() < corr.numel():
                    buf = torch.empty(corr.numel(), dtype=p.dtype, device=p.device)
                    st.scratch[p.dtype] = buf
                buf[: corr.numel()].copy_(corr)                # H2D, no alloc
                p.data.view(-1).add_(buf[: corr.numel()])
        # distribute lr0's dense result to local ranks 1-7 (NVLink, default group)
        for _, p in st.dense:
            dist.broadcast(p.data, src=0)
    torch.cuda.empty_cache()
    optimizer.reload_model_params()

    st.outer_step_count += 1

    # verify the per-step count sync is keeping expert_bias bit-identical
    # across nodes (bias_pg is idle here — the wire thread has joined)
    bias_sync = "n/a"
    if st.bias_bufs:
        bsum = sum(float(b.detach().sum(dtype=torch.float64)) for b in st.bias_bufs)
        ok, blo, bhi = _pair_minmax_equal(st.bias_pg, bsum)
        bias_sync = ok
        if not ok:
            _log(st, f"WARNING: expert_bias diverged across nodes: {blo} != {bhi}")

    _log(st, f"outer step {st.outer_step_count} @ inner {st.inner_step}: "
             f"snap {stats['snap_s']:.1f}s, wire {stats['wire_s']:.1f}s "
             f"(overlapped tau={st.tau}), apply {time.time() - t0:.1f}s, "
             f"|pseudo-grad| rms {stats['rms']:.3e}, theta in sync: {stats['in_sync']}, "
             f"bias in sync: {bias_sync}")
    if not stats["in_sync"]:
        raise RuntimeError(
            f"[diloco] theta divergence detected: {stats['lo']} != {stats['hi']}")


def _install_data_shard():
    """DILOCO_DATA_SHARD=1: exact disjoint data sharding across nodes.

    Both nodes MUST use the same seed (guarded in _setup): they then share one
    global sample order and split it disjointly — zero duplication, zero
    omission (vs the seed-split mode, which duplicates ~17% of the corpus over
    a full 0.7T-per-node budget). Two mappings (DILOCO_SHARD_BLOCK):

    - B=0, legacy sample-parity: node r takes positions world*i + r.
      CAUTION under explicit blend weights: the top-level BlendedDataset
      source sequence is deterministic (NOT shuffled — only each component's
      internal order is), and the stride-2 split aliases with it. Measured on
      stage2: constant per-node composition offsets (P2, e.g. cc_actual
      ±3.3%p -> lm-loss offset 0.038), and when the 6-decimal weight sum is
      1 ± 1e-6 a slow MIRROR SEESAW between the nodes with period
      2/|sum(w)-1| samples (P2b/P3: 325 iters at GBS 3072, composition swing
      up to ±8%p, per-node lm loss ±0.032). Root cause + reproduction:
      study/mirror_loss_aliasing.md, study/mirror_loss_repro.py.

    - B>0, block-cyclic (use B = GBS): node r takes position
      world*B*(i//B) + r*B + i%B. Each node's iteration batch is then one
      CONTIGUOUS B-slice of the blend sequence, and the greedy blender's
      bounded discrepancy makes every batch match the exact blend within ~2
      samples (P3 measured: per-node max deviation 1.3 samples, cross-node
      max 2 — vs 137/274 under parity). Offset and seesaw both vanish.
      Switching parity->block mid-run is exact iff consumed % B == 0 (any
      iteration boundary at constant GBS == B; asserted in _setup): the two
      mappings tile the same remaining global range, so the pair-union
      stream per iteration is IDENTICAL — only the node assignment changes.

    The underlying train dataset is built with world x the requested samples so
    each node's shard still covers its full --train-samples budget.
    """
    import megatron_patch.data as _MPD
    world = int(os.environ.get("DILOCO_WORLD", "2"))
    rank = int(os.environ["DILOCO_RANK"])
    block = int(os.environ.get("DILOCO_SHARD_BLOCK", "0"))
    orig_provider = _MPD.train_valid_test_datasets_provider

    class _ShardView(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            if block > 0:
                # whole blocks only — a trailing partial block would map node1
                # past the end of the underlying dataset
                return (len(self.ds) // (world * block)) * block
            return len(self.ds) // world
        def __getitem__(self, idx):
            i = int(idx)
            if block > 0:
                return self.ds[world * block * (i // block) + rank * block + i % block]
            return self.ds[world * i + rank]
        def __getattr__(self, name):          # passthrough for Megatron attrs
            return getattr(self.ds, name)

    def provider(train_val_test_num_samples, *args, **kwargs):
        sizes = list(train_val_test_num_samples)
        sizes[0] = sizes[0] * world
        train, valid, test = orig_provider(sizes, *args, **kwargs)
        view = _ShardView(train)
        mapping = (f"{world}*{block}*(i//{block}) + {rank}*{block} + i%{block}"
                   if block > 0 else f"{world}*i+{rank}")
        print(f"[diloco] data shard: node {rank}/{world} takes positions "
              f"{mapping} of a shared global order "
              f"(underlying len {len(train)} -> shard len {len(view)})",
              flush=True)
        return view, valid, test

    for attr in ("is_distributed",):
        if hasattr(orig_provider, attr):
            setattr(provider, attr, getattr(orig_provider, attr))
    _MPD.train_valid_test_datasets_provider = provider


def install_unshard_resume():
    """DILOCO_UNSHARD_RESUME=1: continue a DILOCO_DATA_SHARD=1 pair run on ONE node.

    A sharded 2-node run's checkpoint stores LOCAL counters (consumed samples,
    iteration, scheduler position = N x GBS), but the pair jointly consumed
    world x that from the shared global order. On a plain single-node
    continuation all THREE coupled counters must be multiplied by world:
      (a) consumed_train_samples  -> data resumes at the true global position
          (no duplication of the last half, no omission of the odd shard)
      (b) opt_param_scheduler.num_steps -> LR reflects globally consumed tokens
      (c) iteration (+ fp-op counter) -> the train-samples budget terminates
          correctly instead of overshooting by the 2-node phase's volume
    This mode installs NO DiLoCo machinery — the continuation is plain Megatron.
    """
    world = int(os.environ.get("DILOCO_WORLD", "2"))
    import megatron.training.checkpointing as _C
    import megatron.training.training as _T
    orig_load = _C.load_checkpoint

    def load_unshard(model, optimizer, opt_param_scheduler, *a, **k):
        iteration, num_fp_ops = orig_load(model, optimizer, opt_param_scheduler, *a, **k)
        if iteration == 0:
            return iteration, num_fp_ops       # fresh start — nothing to unshard
        from megatron.training import get_args
        from megatron.core.num_microbatches_calculator import update_num_microbatches
        args = get_args()
        old_consumed = args.consumed_train_samples
        args.consumed_train_samples *= world
        if getattr(args, "skipped_train_samples", 0):
            args.skipped_train_samples *= world
        update_num_microbatches(consumed_samples=args.consumed_train_samples,
                                verbose=False)
        if opt_param_scheduler is not None:
            opt_param_scheduler.num_steps *= world
        new_iter = iteration * world
        args.iteration = new_iter
        load_dir = getattr(args, "load", "") or ""
        if not os.path.isdir(os.path.join(load_dir, "diloco_outer")):
            print("[diloco] unshard WARNING: checkpoint has no diloco_outer/ — "
                  "is this really a DILOCO_DATA_SHARD checkpoint?", flush=True)
        print(f"[diloco] unshard resume (world={world}): iteration {iteration}->"
              f"{new_iter}, consumed {old_consumed}->{args.consumed_train_samples}, "
              f"scheduler num_steps x{world}", flush=True)
        return new_iter, num_fp_ops * world

    _C.load_checkpoint = load_unshard
    _T.load_checkpoint = load_unshard
    print(f"[diloco] unshard-resume mode installed (world={world}) — "
          f"plain single-node continuation of a sharded pair run", flush=True)


def _install_bias_count_sync():
    """Cross-node sync of the aux-loss-free router bias, at its native semantics.

    Megatron's get_updated_expert_bias() SUMS local_tokens_per_expert across the
    instance's TPxCPxDP ranks, then nudges each expert's bias by
    sign(mean_load - load) * rate. We extend the SUM across the DiLoCo pair
    BEFORE the intra-instance reduce, so every rank of both nodes computes the
    identical update from the combined 2-node global batch (reference sync-DP
    semantics; arXiv 2408.15664). Both nodes' biases stay bit-identical given
    identical starting biases. Wire cost: 24x192 fp32 = 18KB per step on the
    dedicated bias pair group (Gloo/TCP, ~ms).
    """
    import importlib
    # NB: `import megatron.core.distributed.finalize_model_grads as _F` yields the
    # FUNCTION, not the module — the package __init__ rebinds the attribute
    # (`from .finalize_model_grads import finalize_model_grads`). importlib
    # returns the real module from sys.modules, whose global namespace is what
    # the call site actually reads. (The original hasattr() guard silently
    # skipped the critical patch because a function has no such attribute.)
    _MU = importlib.import_module("megatron.core.transformer.moe.moe_utils")
    _F = importlib.import_module("megatron.core.distributed.finalize_model_grads")
    orig_fn = _MU.get_updated_expert_bias

    call_stat = {"n": 0}

    def get_updated_expert_bias_pair_synced(tokens_per_expert, expert_bias,
                                            expert_bias_update_rate):
        st = _state
        call_stat["n"] += 1
        if call_stat["n"] == 1:
            print(f"[diloco][bias-sync] FIRST CALL rank={os.environ.get('LOCAL_RANK')}"
                  f" state={'ok' if st is not None else 'None'}"
                  f" bias_pg={'ok' if (st and st.bias_pg is not None) else 'None'}",
                  flush=True)
        if st is not None and st.bias_pg is not None:
            buf = tokens_per_expert.detach().to("cpu", torch.float32)
            _allreduce_sum(st.bias_pg, buf)
            # NOTE: debug collectives must run SYMMETRICALLY on both nodes —
            # pass the flag via EXTRA_ENV so node1 gets it too (launch_diloco's
            # ENVV whitelist does not forward arbitrary DILOCO_* vars).
            dbg = os.environ.get("DILOCO_BIAS_DEBUG", "0") == "1" and st.local_rank == 0
            if dbg:
                pre = float(expert_bias.detach().sum(dtype=torch.float64))
                cnt = float(buf.sum(dtype=torch.float64))
                for tag, val in (("pre-bias", pre), ("summed-counts", cnt)):
                    eq, lo, hi = _pair_minmax_equal(st.bias_pg, val)
                    print(f"[diloco][bias-debug][node{st.node_rank}] {tag} "
                          f"local={val:.9f} eq={eq} lo={lo:.9f} hi={hi:.9f}",
                          flush=True)
            tokens_per_expert = buf.to(tokens_per_expert.device,
                                       tokens_per_expert.dtype)
            out = orig_fn(tokens_per_expert, expert_bias, expert_bias_update_rate)
            if dbg:
                post = float(out.detach().sum(dtype=torch.float64))
                eq, lo, hi = _pair_minmax_equal(st.bias_pg, post)
                print(f"[diloco][bias-debug][node{st.node_rank}] post-update "
                      f"local={post:.9f} eq={eq} lo={lo:.9f} hi={hi:.9f}",
                      flush=True)
            return out
        return orig_fn(tokens_per_expert, expert_bias, expert_bias_update_rate)

    _MU.get_updated_expert_bias = get_updated_expert_bias_pair_synced
    _F.get_updated_expert_bias = get_updated_expert_bias_pair_synced
    print(f"[diloco] expert-bias count sync installed "
          f"(per-step pair SUM of tokens_per_expert; "
          f"_F now={_F.get_updated_expert_bias.__name__}, "
          f"_MU now={_MU.get_updated_expert_bias.__name__})", flush=True)


def install():
    """Wrap megatron.training.training.train_step with the DiLoCo outer loop."""
    global _state
    import megatron.training.training as _T
    orig = _T.train_step

    if os.environ.get("DILOCO_DATA_SHARD", "0") == "1":
        _install_data_shard()

    if os.environ.get("DILOCO_BIAS_SYNC", "1") == "1":
        _install_bias_count_sync()

    def train_step_diloco(*args, **kwargs):
        global _state
        if _state is None:
            _state = _State()
        st = _state
        model = kwargs.get("model", args[2] if len(args) > 2 else None)
        optimizer = kwargs.get("optimizer", args[3] if len(args) > 3 else None)
        if st.owned is None:
            _setup(st, model, optimizer)
        if st.pending is not None and st.inner_step >= st.pending["apply_at"]:
            _apply_outer(st, optimizer)
        out = orig(*args, **kwargs)
        st.inner_step += 1
        if st.inner_step % st.H == 0:
            _start_outer(st)
            if st.tau == 0:
                _apply_outer(st, optimizer)
        return out

    _T.train_step = train_step_diloco
    print(f"[diloco] train_step wrapped (H={os.environ.get('DILOCO_H', '30')}, "
          f"tau={os.environ.get('DILOCO_TAU', '0')})", flush=True)

    if os.environ.get("DILOCO_SKIP_SAVE", "0") == "1":
        def _skip_save(*args, **kwargs):
            print("[diloco] DILOCO_SKIP_SAVE=1 — checkpoint save skipped", flush=True)
        _T.save_checkpoint_and_time = _skip_save
        print("[diloco] checkpoint saving DISABLED (DILOCO_SKIP_SAVE=1)", flush=True)
    else:
        # piggyback the DiLoCo outer state (theta_global + outer momentum) onto
        # every Megatron checkpoint save, enabling exact resume of the outer loop
        orig_save = _T.save_checkpoint_and_time
        def _save_with_outer_state(iteration, *args, **kwargs):
            st = _state
            if st is not None and st.pending is not None:
                # Drain the in-flight sync BEFORE saving: (a) the checkpoint then
                # captures a consistent post-apply outer state, and (b) every save
                # crash observed so far ("NCCL Error 1" in the save's gather)
                # occurred exactly when a pending worker was alive during save.
                _log(st, "draining in-flight outer sync before checkpoint save")
                _apply_outer(st, args[1])   # args: (model, optimizer, ...)
            out = orig_save(iteration, *args, **kwargs)
            _save_outer_state(_state, iteration)
            return out
        _T.save_checkpoint_and_time = _save_with_outer_state
