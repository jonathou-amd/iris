#!/usr/bin/env python3
"""Capture driver: independent GEMM + all-gather via iris.concurrent.gemm.all_gather (default tritonblas GEMM)."""

import argparse
import os

import torch
import torch.distributed as dist

import iris
import iris.concurrent.gemm as cg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fused", "concurrent", "gemm_only"], required=True)
    p.add_argument("--gemm_m", type=int, default=512)
    p.add_argument("--gemm_n", type=int, default=256)
    p.add_argument("--gemm_k", type=int, default=512)
    p.add_argument("--comm_m", type=int, default=2048)
    p.add_argument("--comm_n", type=int, default=2048)
    p.add_argument("--gemm_wgs", type=int, default=128)
    p.add_argument("--comm_wgs", type=int, default=128)
    p.add_argument("--num_warps", type=int, default=8)
    p.add_argument("--backend", choices=["triton", "gluon"], default="triton")
    p.add_argument("--comm_block_m", type=int, default=None)
    p.add_argument("--comm_block_n", type=int, default=None)
    p.add_argument("--gemm_block", type=int, nargs=3, default=None)
    p.add_argument("--check", action="store_true", help="compare C and this rank's comm_dst slice against CPU refs")
    p.add_argument("--num_stages", type=int, default=2)
    p.add_argument("--heap_size", type=int, default=1 << 30)
    a = p.parse_args()

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="gloo")
    shmem = iris.iris(a.heap_size)
    dt = torch.float16
    # FFM's device RNG returns constants (all ones), which makes a GEMM check blind to indexing bugs, so fill
    # from a seeded host RNG instead (same values on every rank for A/B; per-rank for the all-gather source).
    g = torch.Generator().manual_seed(1234)
    A = shmem.zeros(a.gemm_m, a.gemm_k, device="cuda", dtype=dt)
    A.copy_(torch.randn(a.gemm_m, a.gemm_k, generator=g).to(dt))
    Bt = shmem.zeros(a.gemm_n, a.gemm_k, device="cuda", dtype=dt)
    Bt.copy_(torch.randn(a.gemm_n, a.gemm_k, generator=g).to(dt))
    B = Bt.T
    src = shmem.zeros(a.comm_m, a.comm_n, device="cuda", dtype=dt)
    src.copy_(torch.randn(a.comm_m, a.comm_n, generator=torch.Generator().manual_seed(77 + shmem.get_rank())).to(dt))
    shmem.barrier()
    kw = {}
    if a.backend == "gluon":
        kw["backend"] = "gluon"
        kw["comm_block"] = (a.comm_block_m or 8, a.comm_block_n or 256)
    elif a.comm_block_m:
        kw["comm_block"] = (a.comm_block_m, a.comm_block_n)
    if a.gemm_block:
        kw["gemm_block"] = tuple(a.gemm_block)
    num_wgs = a.gemm_wgs + a.comm_wgs if a.mode == "fused" else None
    C, dst = cg.all_gather(
        shmem,
        A,
        B,
        src,
        mode=a.mode,
        num_wgs=num_wgs,
        gemm_wgs=a.gemm_wgs,
        comm_wgs=a.comm_wgs,
        num_warps=a.num_warps,
        num_stages=a.num_stages,
        **kw,
    )
    torch.cuda.synchronize()
    shmem.barrier()
    if a.check:
        ref = A.cpu().float() @ B.cpu().float()
        err = (C.cpu().float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
        r = shmem.get_rank()
        own = True if a.mode == "gemm_only" else torch.equal(dst[r * a.comm_m : (r + 1) * a.comm_m].cpu(), src.cpu())
        print(
            f"CHECK rank{r} backend={a.backend} gemm_rel_err={err:.3e} own_slice_ok={own} A_absmax={A.abs().max().item():.3f} B_absmax={B.abs().max().item():.3f} ref_absmax={ref.abs().max().item():.3f} src_absmax={src.abs().max().item():.3f}",
            flush=True,
        )
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
