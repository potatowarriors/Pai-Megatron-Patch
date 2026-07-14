#!/usr/bin/env python3
"""NCCL all-reduce benchmark via torch.distributed (torchrun launcher).

Mirrors the transport path Megatron training actually uses.
Reports algbw and busbw (nccl-tests convention: busbw = algbw * 2(n-1)/n).
"""
import os
import time

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"world_size={world}  nccl={torch.cuda.nccl.version()}", flush=True)
        print(f"{'size':>12}  {'time/iter':>10}  {'algbw':>10}  {'busbw':>10}", flush=True)

    sizes = [32 * 1024, 1 << 20, 8 << 20, 64 << 20, 256 << 20, 1 << 30]
    for size_bytes in sizes:
        n = size_bytes // 2
        x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
        for _ in range(3):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()

        iters = 10 if size_bytes <= (64 << 20) else 4
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters

        algbw = size_bytes / dt / 1e9
        busbw = algbw * 2 * (world - 1) / world
        if rank == 0:
            print(f"{size_bytes / 1e6:9.2f} MB  {dt * 1e3:8.2f} ms  "
                  f"{algbw:7.2f} GB/s  {busbw:7.2f} GB/s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
