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
  DILOCO_OUTER_LR        default 0.7   (DiLoCo paper defaults)
  DILOCO_OUTER_MOMENTUM  default 0.9
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
        self.outer_momentum = float(os.environ.get("DILOCO_OUTER_MOMENTUM", "0.9"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        assert 0 <= self.tau < self.H, "DILOCO_TAU must be in [0, H)"
        self.pg = None
        self.wire_dtype = torch.bfloat16
        self.params = None      # all named params (initial broadcast)
        self.owned = None       # params THIS rank wire-syncs (experts [+ dense on lr0])
        self.dense = None       # dense params (intra-node broadcast targets)
        self.theta = None       # cpu fp32 globals, aligned with owned
        self.momentum = None
        self.inner_step = 0
        self.outer_step_count = 0
        self.pending = None     # in-flight sync: dict(snap, thread, stats, apply_at)


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


def _named_params(model_chunks):
    out = []
    for chunk in model_chunks:
        for name, p in chunk.named_parameters():
            if p.requires_grad:
                out.append((name, p))
    return out


def _setup(st, model, optimizer):
    """First-call init: pair group, cross-node weight broadcast, snapshots."""
    t0 = time.time()
    _make_pair_group(st)
    st.params = _named_params(model)

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

    # broadcast node0 weights so both replicas start bit-identical
    # (D2H in the param's own dtype, cast on CPU — GPU-side dtype temps
    #  fragment the allocator and break torch_dist checkpoint save)
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
                # additive correction keeps the tau-window local progress
                p.data.add_((th - sn).to(p.device, p.dtype))
        # distribute lr0's dense result to local ranks 1-7 (NVLink, default group)
        for _, p in st.dense:
            dist.broadcast(p.data, src=0)
    torch.cuda.empty_cache()
    optimizer.reload_model_params()

    st.outer_step_count += 1
    _log(st, f"outer step {st.outer_step_count} @ inner {st.inner_step}: "
             f"snap {stats['snap_s']:.1f}s, wire {stats['wire_s']:.1f}s "
             f"(overlapped tau={st.tau}), apply {time.time() - t0:.1f}s, "
             f"|pseudo-grad| rms {stats['rms']:.3e}, theta in sync: {stats['in_sync']}")
    if not stats["in_sync"]:
        raise RuntimeError(
            f"[diloco] theta divergence detected: {stats['lo']} != {stats['hi']}")


def install():
    """Wrap megatron.training.training.train_step with the DiLoCo outer loop."""
    global _state
    import megatron.training.training as _T
    orig = _T.train_step

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
