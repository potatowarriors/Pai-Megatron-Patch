"""DiLoCo-lite outer loop for alpha 2-node training over a slow interconnect.

Each node runs an INDEPENDENT single-node Megatron instance (DP=8/EP=8, all
gradient traffic on NVLink). Every H inner steps, paired local ranks
(node0.local_rank_r <-> node1.local_rank_r) average their parameter deltas over
a private per-pair Gloo group and apply a Nesterov outer update.

References: DiLoCo (arXiv:2311.08105), Streaming DiLoCo (arXiv:2501.18512),
MuLoCo — Muon validated as the inner optimizer (arXiv:2505.23725).

Why pairwise per local rank: with EP=8 each local rank owns a distinct expert
shard, so rank r's parameter set only matches rank r on the other node. Dense
params are synced redundantly by all 8 pairs (identical values; harmless).

Env contract (set by the launcher):
  DILOCO_RANK            0|1  node index (0 hosts the TCPStores)
  DILOCO_WORLD           2
  DILOCO_MASTER          hostname of node 0
  DILOCO_PORT_BASE       base port; local_rank r uses base+r
  DILOCO_H               inner steps per outer sync (default 30)
  DILOCO_OUTER_LR        default 0.7   (DiLoCo paper defaults)
  DILOCO_OUTER_MOMENTUM  default 0.9

The outer state (theta_global + momentum, fp32) lives on CPU: ~27 GB per rank,
~216 GB per node — fine on these 2 TB hosts.
"""
import datetime
import os
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
        self.outer_lr = float(os.environ.get("DILOCO_OUTER_LR", "0.7"))
        self.outer_momentum = float(os.environ.get("DILOCO_OUTER_MOMENTUM", "0.9"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.pg = None
        self.wire_dtype = torch.bfloat16
        self.params = None      # list[(name, param)]
        self.theta = None       # list[cpu fp32 tensors] — global params
        self.momentum = None    # list[cpu fp32 tensors] — outer Nesterov buffer
        self.inner_step = 0
        self.outer_step_count = 0


def _log(st, msg):
    if st.local_rank == 0:
        print(f"[diloco][node{st.node_rank}] {msg}", flush=True)


def _bcast(pg, t, root):
    opts = dist.BroadcastOptions()
    opts.rootRank = root
    opts.rootTensor = 0
    pg.broadcast([t], opts).wait()


def _allreduce_sum(pg, t):
    opts = dist.AllreduceOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    pg.allreduce([t], opts).wait()


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
    # probe bf16 support for the wire format; fall back to fp32
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

    # sanity: identical layout on both sides
    meta = torch.tensor([len(st.params), sum(p.numel() for _, p in st.params)],
                        dtype=torch.float64)
    lo, hi = meta.clone(), meta.clone()
    opts = dist.AllreduceOptions(); opts.reduceOp = dist.ReduceOp.MIN
    st.pg.allreduce([lo], opts).wait()
    opts = dist.AllreduceOptions(); opts.reduceOp = dist.ReduceOp.MAX
    st.pg.allreduce([hi], opts).wait()
    assert torch.equal(lo, hi), (
        f"param layout mismatch across nodes: min={lo.tolist()} max={hi.tolist()}")

    # broadcast node0 weights so both replicas start bit-identical.
    # D2H in the param's own dtype, cast on CPU — avoids per-param GPU fp32
    # temporaries (allocator fragmentation broke a later checkpoint save).
    with torch.no_grad():
        for _, p in st.params:
            buf = p.data.detach().cpu().to(torch.float32)
            _bcast(st.pg, buf, root=0)
            p.data.copy_(buf.to(p.dtype))
    torch.cuda.empty_cache()
    optimizer.reload_model_params()

    st.theta = [p.data.detach().to("cpu", torch.float32) for _, p in st.params]
    st.momentum = [torch.zeros_like(t) for t in st.theta]
    _log(st, f"setup done in {time.time() - t0:.1f}s: {len(st.params)} params, "
             f"{sum(p.numel() for _, p in st.params) / 1e9:.2f}B/rank, H={st.H}, "
             f"outer lr={st.outer_lr} mu={st.outer_momentum}")


def _outer_step(st, optimizer):
    t0 = time.time()
    mu, lr = st.outer_momentum, st.outer_lr
    delta_sq, numel = 0.0, 0
    with torch.no_grad():
        for (name, p), th, m in zip(st.params, st.theta, st.momentum):
            local = p.data.detach().cpu().to(torch.float32)   # D2H in own dtype, cast on CPU
            delta = th - local                      # pseudo-gradient
            wire = delta.to(st.wire_dtype)
            _allreduce_sum(st.pg, wire)
            delta_avg = wire.to(torch.float32).div_(st.world)
            m.mul_(mu).add_(delta_avg)
            th.sub_(delta_avg.add_(m, alpha=mu), alpha=lr)   # Nesterov
            p.data.copy_(th.to(p.dtype))
            delta_sq += float(delta_avg.pow(2).sum())
            numel += delta_avg.numel()
    torch.cuda.empty_cache()
    optimizer.reload_model_params()

    # divergence guard: both sides must hold bit-identical theta
    s = sum(float(t.sum(dtype=torch.float64)) for t in st.theta)
    chk = torch.tensor([s], dtype=torch.float64)
    lo, hi = chk.clone(), chk.clone()
    opts = dist.AllreduceOptions(); opts.reduceOp = dist.ReduceOp.MIN
    st.pg.allreduce([lo], opts).wait()
    opts = dist.AllreduceOptions(); opts.reduceOp = dist.ReduceOp.MAX
    st.pg.allreduce([hi], opts).wait()
    in_sync = bool(torch.equal(lo, hi))

    st.outer_step_count += 1
    _log(st, f"outer step {st.outer_step_count} @ inner {st.inner_step}: "
             f"{time.time() - t0:.1f}s, |pseudo-grad| rms {(delta_sq / max(numel, 1)) ** 0.5:.3e}, "
             f"replicas in sync: {in_sync}")
    if not in_sync:
        raise RuntimeError(f"[diloco] replica divergence detected: {lo.item()} != {hi.item()}")


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
        if st.params is None:
            _setup(st, model, optimizer)
        out = orig(*args, **kwargs)
        st.inner_step += 1
        if st.inner_step % st.H == 0:
            _outer_step(st, optimizer)
        return out

    _T.train_step = train_step_diloco
    print(f"[diloco] train_step wrapped (H={os.environ.get('DILOCO_H', '30')})", flush=True)

    # optional escape hatch: skip checkpoint saving entirely (loss-curve pilots
    # don't need checkpoints; torch_dist save is the one path still under debug)
    if os.environ.get("DILOCO_SKIP_SAVE", "0") == "1":
        def _skip_save(*args, **kwargs):
            print("[diloco] DILOCO_SKIP_SAVE=1 — checkpoint save skipped", flush=True)
        _T.save_checkpoint_and_time = _skip_save
        print("[diloco] checkpoint saving DISABLED (DILOCO_SKIP_SAVE=1)", flush=True)
