# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon backend for :mod:`iris.concurrent` (GEMM + all-gather, fused and two-kernel).

Why this exists: the Triton GEMM tile (tritonblas ``GemmContext`` -> ``tl.dot``) lowers to
``v_wmma_f32_16x16x32_f16`` on gfx1260, which AM's SH decoder rejects (SH_INST_ILLEGAL_ENC_OR_OP).
The GEMM here uses the shape AM can decode -- WMMA version 4 at ``[32, 64, 64]`` with A, B and the
accumulator in laneshared, reached through ``wmma_laneshared`` -- ported from
``iris/ops/gluon/all_gather_matmul.py`` (muhaawad/gemm-ag, commit 3cd4dde).

A ``@triton.jit`` kernel cannot call Gluon code, so the whole persistent kernel is Gluon, including
the all-gather tile (flat-2D tiling, as in ``iris/ccl/gluon/all_gather.py``). The kernel names are
the same as the Triton ones (``ws_gemm``, ``ws_all_gather``, ``fused_ws_gemm_all_gather``) so
``roccap -k`` filters and dispatch names do not change.

Constraints: fp16/bf16; GEMM tile multiples of (32, 64, 64); comm tile elements a multiple of
``32 * num_warps``.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

from iris.mem.gluon.context import Context as IrisDeviceCtx

_TPW = 32  # gfx1260 is wave32 (host side)
THREADS_PER_WARP = ttgl.constexpr(_TPW)  # device side: jit code may only read constexpr globals
# Epilogue: accumulator -> blocked layout -> 16 B stores. (With two inlined copies of the GEMM tile in one kernel this
# hung the emulator; the fused kernel now inlines each tile function once, see fused_ws_gemm_all_gather.)
BLOCKED_C = ttgl.constexpr(True)


@gluon.constexpr_function
def _wmma_layouts(INSTR_SHAPE_K: ttgl.constexpr, WARPS: ttgl.constexpr):
    # All warps along M: the instruction is 32x64 and the tiles used here are 64 wide in N, so splitting warps
    # over N (the reference's [[0,1],[1,0]]) makes pairs of warps compute the same output.
    bases = []
    w = 1
    while w < WARPS:
        bases.append([w, 0])
        w *= 2
    WMMA = ttgl.amd.AMDWMMALayout(4, False, warp_bases=bases, instr_shape=[32, 64, INSTR_SHAPE_K])
    DOT_A = ttgl.DotOperandLayout(0, WMMA, INSTR_SHAPE_K)
    DOT_B = ttgl.DotOperandLayout(1, WMMA, INSTR_SHAPE_K)
    return WMMA, DOT_A, DOT_B


@gluon.constexpr_function
def _c_blocked(BLOCK_N: ttgl.constexpr, WARPS: ttgl.constexpr):
    """Blocked layout for the C store: 8 contiguous elements per thread along N (16 B for fp16/bf16)."""
    tn = min(BLOCK_N // 8, 32)
    return ttgl.BlockedLayout([1, 8], [32 // tn, tn], [WARPS, 1], [1, 0])


@gluon.jit
def _grab(counter, WARPS_PER_CTA: ttgl.constexpr):
    """Atomically take the next work index; the result is uniform across the workgroup.

    A scalar ``atomic_add`` in Gluon is per-thread (every lane would take its own index), so issue the
    add from lane 0 only through a tensor pointer and broadcast the old value with a sum reduction.
    """
    L: ttgl.constexpr = ttgl.BlockedLayout([1], [THREADS_PER_WARP], [WARPS_PER_CTA], [0])
    offs = ttgl.arange(0, THREADS_PER_WARP * WARPS_PER_CTA, layout=L)
    ptrs = counter + offs * 0
    ones = ttgl.full([THREADS_PER_WARP * WARPS_PER_CTA], 1, ttgl.int32, layout=L)
    old = ttgl.atomic_add(ptrs, ones, mask=offs == 0, sem="relaxed", scope="gpu")
    # Masked-off lanes return an undefined old value: keep only lane 0's before reducing.
    return ttgl.sum(ttgl.where(offs == 0, old, 0), axis=0)


@gluon.jit
def _gemm_tile(
    tile_id,
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    num_pid_n,
    lsmem_a,
    lsmem_b,
    lsmem_acc,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_K: ttgl.constexpr,
    GROUP_M: ttgl.constexpr,
    INSTR_SHAPE_K: ttgl.constexpr,
    EVEN: ttgl.constexpr,
    WARPS_PER_CTA: ttgl.constexpr,
    BLOCKED_C: ttgl.constexpr,
):
    """One output tile of ``C = A @ B`` (local), accumulated in laneshared.

    ``EVEN`` (M, N, K all divide the tile) drops every bounds mask: with masks each load becomes a guarded
    branch and the C store scalarises.
    """
    layouts: ttgl.constexpr = _wmma_layouts(INSTR_SHAPE_K, WARPS_PER_CTA)
    WMMA_LAYOUT: ttgl.constexpr = layouts[0]
    DOT_A_LAYOUT: ttgl.constexpr = layouts[1]
    DOT_B_LAYOUT: ttgl.constexpr = layouts[2]

    num_pid_m = ttgl.cdiv(M, BLOCK_M)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, DOT_A_LAYOUT))
    offs_ak = ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(0, DOT_A_LAYOUT))
    offs_bn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, DOT_B_LAYOUT))
    offs_bk = ttgl.arange(0, BLOCK_K, layout=ttgl.SliceLayout(1, DOT_B_LAYOUT))

    lsmem_acc.store(ttgl.zeros((BLOCK_M, BLOCK_N), dtype=ttgl.float32, layout=WMMA_LAYOUT))

    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    nk = ttgl.cdiv(K, BLOCK_K)

    # Software pipeline: the loads for K step k+1 are issued before the WMMA of step k, so their latency
    # overlaps the matrix op instead of being waited on in line (the first tile is loaded before the loop).
    if EVEN:
        a = ttgl.load(a_ptr + offs_a)
        b = ttgl.load(b_ptr + offs_b)
    else:
        mask_a = (offs_ak[None, :] < K) & (offs_am[:, None] < M)
        mask_b = (offs_bk[:, None] < K) & (offs_bn[None, :] < N)
        a = ttgl.load(a_ptr + offs_a, mask=mask_a, other=0.0)
        b = ttgl.load(b_ptr + offs_b, mask=mask_b, other=0.0)
    for k in range(0, nk):
        lsmem_a.store(a)
        lsmem_b.store(b)
        offs_a += BLOCK_K * stride_ak
        offs_b += BLOCK_K * stride_bk
        if k + 1 < nk:
            if EVEN:
                a = ttgl.load(a_ptr + offs_a)
                b = ttgl.load(b_ptr + offs_b)
            else:
                mask_a = (offs_ak[None, :] < K - (k + 1) * BLOCK_K) & (offs_am[:, None] < M)
                mask_b = (offs_bk[:, None] < K - (k + 1) * BLOCK_K) & (offs_bn[None, :] < N)
                a = ttgl.load(a_ptr + offs_a, mask=mask_a, other=0.0)
                b = ttgl.load(b_ptr + offs_b, mask=mask_b, other=0.0)
        ttgl.amd.gfx1260.wmma_laneshared(lsmem_acc, lsmem_a, lsmem_b, lsmem_acc, WMMA_LAYOUT)

    acc = lsmem_acc.load(WMMA_LAYOUT)
    if BLOCKED_C:
        # The accumulator layout leaves each lane's elements non-contiguous (scalar 2-byte stores); convert to a
        # blocked layout with 8 contiguous elements per thread so the store is 16 B wide.
        C_LAYOUT: ttgl.constexpr = _c_blocked(BLOCK_N, WARPS_PER_CTA)
        out = ttgl.convert_layout(acc.to(c_ptr.type.element_ty), C_LAYOUT)
        offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, C_LAYOUT))
        offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, C_LAYOUT))
    else:
        out = acc.to(c_ptr.type.element_ty)
        offs_cm = pid_m * BLOCK_M + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, WMMA_LAYOUT))
        offs_cn = pid_n * BLOCK_N + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, WMMA_LAYOUT))
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    # No cache_modifier: ".wt" lowers to a system-scope non-temporal store with a full wait after each one.
    if EVEN:
        ttgl.store(c_ptr + offs_c, out)
    else:
        mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        ttgl.store(c_ptr + offs_c, out, mask=mask_c)


@gluon.jit
def _all_gather_tile(
    tile_id,
    heap_bases,
    local_base,
    comm_src,
    comm_dst,
    Mc,
    Nc,
    stride_sm,
    stride_sn,
    stride_dm,
    stride_dn,
    num_pid_n,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    GROUP_M: ttgl.constexpr,
    cur_rank: ttgl.constexpr,
    world_size: ttgl.constexpr,
    WARPS_PER_CTA: ttgl.constexpr,
    EVEN: ttgl.constexpr,
):
    """All-gather one local (Mc, Nc) block into rows [cur_rank*Mc : ...] of every rank's comm_dst."""
    TOTAL_ELEMS: ttgl.constexpr = BLOCK_M * BLOCK_N
    ELEMS_PER_THREAD: ttgl.constexpr = TOTAL_ELEMS // (THREADS_PER_WARP * WARPS_PER_CTA)
    flat_layout: ttgl.constexpr = ttgl.BlockedLayout([ELEMS_PER_THREAD], [THREADS_PER_WARP], [WARPS_PER_CTA], [0])

    num_pid_m = ttgl.cdiv(Mc, BLOCK_M)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m

    flat_idx = ttgl.arange(0, TOTAL_ELEMS, layout=flat_layout)
    row = pid_m * BLOCK_M + flat_idx // BLOCK_N
    col = pid_n * BLOCK_N + flat_idx % BLOCK_N
    if EVEN:
        data = ttgl.load(comm_src + row * stride_sm + col * stride_sn)
    else:
        mask = (row < Mc) & (col < Nc)
        data = ttgl.load(comm_src + row * stride_sm + col * stride_sn, mask=mask, other=0.0)
    out_ptrs = comm_dst + (cur_rank * Mc + row) * stride_dm + col * stride_dn

    # Stagger the write order per rank so each rank hits a different target at any given moment.
    for i in range(world_size):
        dest = (cur_rank + i) % world_size
        if dest == cur_rank:
            if EVEN:
                ttgl.store(out_ptrs, data)
            else:
                ttgl.store(out_ptrs, data, mask=mask)
        else:
            ptr_delta = ttgl.load(heap_bases + dest) - local_base
            remote = tl.cast(tl.cast(out_ptrs, ttgl.uint64) + ptr_delta, out_ptrs.dtype)
            if EVEN:
                ttgl.store(remote, data)
            else:
                ttgl.store(remote, data, mask=mask)


@gluon.jit
def fused_ws_gemm_all_gather(
    IrisDeviceCtx: ttgl.constexpr,
    context_tensor,
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    comm_src,
    comm_dst,
    Mc,
    Nc,
    stride_sm,
    stride_sn,
    stride_dm,
    stride_dn,
    gemm_counter,
    comm_counter,
    GEMM_TOTAL_TILES,
    COMM_TOTAL_TILES,
    GEMM_WGS,
    GEMM_BLOCK_M: ttgl.constexpr,
    GEMM_BLOCK_N: ttgl.constexpr,
    GEMM_BLOCK_K: ttgl.constexpr,
    GEMM_GROUP_M: ttgl.constexpr,
    COMM_BLOCK_M: ttgl.constexpr,
    COMM_BLOCK_N: ttgl.constexpr,
    COMM_GROUP_M: ttgl.constexpr,
    INSTR_SHAPE_K: ttgl.constexpr,
    cur_rank: ttgl.constexpr,
    world_size: ttgl.constexpr,
    WARPS_PER_CTA: ttgl.constexpr,
    GEMM_EVEN: ttgl.constexpr,
    COMM_EVEN: ttgl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    heap_bases = ctx.heap_bases
    local_base = ttgl.load(heap_bases + cur_rank)

    layouts: ttgl.constexpr = _wmma_layouts(INSTR_SHAPE_K, WARPS_PER_CTA)
    lsmem_a = ttgl.amd.gfx1260.laneshared.allocate_laneshared(
        element_ty=A.type.element_ty, shape=[GEMM_BLOCK_M, GEMM_BLOCK_K], layout=layouts[1]
    )
    lsmem_b = ttgl.amd.gfx1260.laneshared.allocate_laneshared(
        element_ty=B.type.element_ty, shape=[GEMM_BLOCK_K, GEMM_BLOCK_N], layout=layouts[2]
    )
    lsmem_acc = ttgl.amd.gfx1260.laneshared.allocate_laneshared(
        element_ty=ttgl.float32, shape=[GEMM_BLOCK_M, GEMM_BLOCK_N], layout=layouts[0]
    )

    pid = ttgl.program_id(0)
    gemm_num_pid_n = ttgl.cdiv(N, GEMM_BLOCK_N)
    comm_num_pid_n = ttgl.cdiv(Nc, COMM_BLOCK_N)

    # Two stages: drain the home queue, then steal from the other one. Written as ONE loop with a runtime
    # queue choice so each tile function is inlined exactly once -- two inlined copies of the GEMM tile hang
    # the emulator with the all-M warp layout.
    gemm_home = pid < GEMM_WGS
    comm_counter = gemm_counter + 1  # counters[0] = gemm queue, counters[1] = comm queue (same int32 tensor)
    for stage in range(2):
        is_gemm = gemm_home == (stage == 0)
        counter = gemm_counter if is_gemm else comm_counter
        total = GEMM_TOTAL_TILES if is_gemm else COMM_TOTAL_TILES
        idx = _grab(counter, WARPS_PER_CTA)
        while idx < total:
            if is_gemm:
                _gemm_tile(idx, A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                           gemm_num_pid_n, lsmem_a, lsmem_b, lsmem_acc,
                           GEMM_BLOCK_M, GEMM_BLOCK_N, GEMM_BLOCK_K, GEMM_GROUP_M, INSTR_SHAPE_K, GEMM_EVEN,
                           WARPS_PER_CTA, BLOCKED_C)  # fmt: skip
            else:
                _all_gather_tile(idx, heap_bases, local_base, comm_src, comm_dst, Mc, Nc, stride_sm, stride_sn,
                                 stride_dm, stride_dn, comm_num_pid_n, COMM_BLOCK_M, COMM_BLOCK_N, COMM_GROUP_M,
                                 cur_rank, world_size, WARPS_PER_CTA, COMM_EVEN)  # fmt: skip
            idx = _grab(counter, WARPS_PER_CTA)


@gluon.jit
def ws_gemm(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    gemm_counter,
    GEMM_TOTAL_TILES,
    GEMM_BLOCK_M: ttgl.constexpr,
    GEMM_BLOCK_N: ttgl.constexpr,
    GEMM_BLOCK_K: ttgl.constexpr,
    GEMM_GROUP_M: ttgl.constexpr,
    INSTR_SHAPE_K: ttgl.constexpr,
    WARPS_PER_CTA: ttgl.constexpr,
    GEMM_EVEN: ttgl.constexpr,
):
    layouts: ttgl.constexpr = _wmma_layouts(INSTR_SHAPE_K, WARPS_PER_CTA)
    lsmem_a = ttgl.amd.gfx1260.laneshared.allocate_laneshared(
        element_ty=A.type.element_ty, shape=[GEMM_BLOCK_M, GEMM_BLOCK_K], layout=layouts[1]
    )
    lsmem_b = ttgl.amd.gfx1260.laneshared.allocate_laneshared(
        element_ty=B.type.element_ty, shape=[GEMM_BLOCK_K, GEMM_BLOCK_N], layout=layouts[2]
    )
    lsmem_acc = ttgl.amd.gfx1260.laneshared.allocate_laneshared(
        element_ty=ttgl.float32, shape=[GEMM_BLOCK_M, GEMM_BLOCK_N], layout=layouts[0]
    )
    gemm_num_pid_n = ttgl.cdiv(N, GEMM_BLOCK_N)
    idx = _grab(gemm_counter, WARPS_PER_CTA)
    while idx < GEMM_TOTAL_TILES:
        _gemm_tile(idx, A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   gemm_num_pid_n, lsmem_a, lsmem_b, lsmem_acc,
                   GEMM_BLOCK_M, GEMM_BLOCK_N, GEMM_BLOCK_K, GEMM_GROUP_M, INSTR_SHAPE_K, GEMM_EVEN, WARPS_PER_CTA, BLOCKED_C)  # fmt: skip
        idx = _grab(gemm_counter, WARPS_PER_CTA)


@gluon.jit
def ws_all_gather(
    IrisDeviceCtx: ttgl.constexpr,
    context_tensor,
    comm_src,
    comm_dst,
    Mc,
    Nc,
    stride_sm,
    stride_sn,
    stride_dm,
    stride_dn,
    comm_counter,
    COMM_TOTAL_TILES,
    COMM_BLOCK_M: ttgl.constexpr,
    COMM_BLOCK_N: ttgl.constexpr,
    COMM_GROUP_M: ttgl.constexpr,
    cur_rank: ttgl.constexpr,
    world_size: ttgl.constexpr,
    WARPS_PER_CTA: ttgl.constexpr,
    COMM_EVEN: ttgl.constexpr,
):
    ctx = IrisDeviceCtx.initialize(context_tensor)
    heap_bases = ctx.heap_bases
    local_base = ttgl.load(heap_bases + cur_rank)
    comm_num_pid_n = ttgl.cdiv(Nc, COMM_BLOCK_N)
    idx = _grab(comm_counter, WARPS_PER_CTA)
    while idx < COMM_TOTAL_TILES:
        _all_gather_tile(idx, heap_bases, local_base, comm_src, comm_dst, Mc, Nc, stride_sm, stride_sn,
                         stride_dm, stride_dn, comm_num_pid_n, COMM_BLOCK_M, COMM_BLOCK_N, COMM_GROUP_M,
                         cur_rank, world_size, WARPS_PER_CTA, COMM_EVEN)  # fmt: skip
        idx = _grab(comm_counter, WARPS_PER_CTA)


# ---------------------------------------------------------------------------
# Host launch
# ---------------------------------------------------------------------------
def check_gluon_args(A, gemm_block, comm_block, num_warps):
    if A.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Gluon backend supports fp16/bf16, got {A.dtype}")
    gm, gn, gk = gemm_block
    if gm % 32 or gn % 64 or gk % 64:
        raise ValueError(f"gemm_block {gemm_block} must be multiples of (32, 64, 64) for the [32, 64, 64] WMMA")
    cm, cn = comm_block
    if num_warps != 4:
        # LaneSharedLayout asserts warps == NUM_WAVE_GROUPS (4); 8 warps fails at compile time.
        raise ValueError(f"Gluon backend needs num_warps=4 (laneshared WMMA requires 4 wave groups), got {num_warps}")
    if gm % (32 * num_warps):
        raise ValueError(f"gemm_block M={gm} must be a multiple of 32 * num_warps = {32 * num_warps}: all warps tile M")
    if (cm * cn) % (_TPW * num_warps):
        raise ValueError(f"comm_block elems {cm * cn} must be a multiple of {_TPW * num_warps}")


def launch(
    mode, shmem, A, B, C, comm_src, comm_dst, *, gemm_wgs, comm_wgs, num_wgs, gemm_block, comm_block,
    gemm_group_m, comm_group_m, num_warps, counters, streams,
):  # fmt: skip
    """Launch the Gluon kernels. ``counters``: int32[2+] zeroed here; ``streams``: (gemm_stream, comm_stream)."""
    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    context_tensor = shmem.get_device_context()
    M, K = A.shape
    _, N = B.shape
    Mc, Nc = comm_src.shape
    gm, gn, gk = gemm_block
    cm, cn = comm_block
    gemm_total = triton.cdiv(M, gm) * triton.cdiv(N, gn)
    comm_total = triton.cdiv(Mc, cm) * triton.cdiv(Nc, cn)
    gemm_counter, comm_counter = counters[0:1], counters[1:2]
    gemm_even = M % gm == 0 and N % gn == 0 and K % gk == 0
    comm_even = Mc % cm == 0 and Nc % cn == 0
    counters.zero_()

    gemm_args = (A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1))
    comm_args = (comm_src, comm_dst, Mc, Nc, comm_src.stride(0), comm_src.stride(1), comm_dst.stride(0),
                 comm_dst.stride(1))  # fmt: skip

    if mode == "gemm_only":
        # Baseline: the same ws_gemm kernel/grid/tile/buffers, with no all-gather launched (comm_dst stays zero).
        ws_gemm[(gemm_wgs,)](*gemm_args, gemm_counter, gemm_total, gm, gn, gk, gemm_group_m, 64, num_warps,
                             gemm_even, num_warps=num_warps)  # fmt: skip
        return C, comm_dst
    if mode == "fused":
        fused_ws_gemm_all_gather[(num_wgs,)](
            IrisDeviceCtx, context_tensor, *gemm_args, *comm_args, gemm_counter, comm_counter,
            gemm_total, comm_total, gemm_wgs, gm, gn, gk, gemm_group_m, cm, cn, comm_group_m, 64,
            rank, world_size, num_warps, gemm_even, comm_even, num_warps=num_warps,
        )  # fmt: skip
    else:
        gemm_stream, comm_stream = streams
        current = torch.cuda.current_stream()
        gemm_stream.wait_stream(current)
        comm_stream.wait_stream(current)
        with torch.cuda.stream(gemm_stream):
            ws_gemm[(gemm_wgs,)](*gemm_args, gemm_counter, gemm_total, gm, gn, gk, gemm_group_m, 64,
                                 num_warps, gemm_even, num_warps=num_warps)  # fmt: skip
        with torch.cuda.stream(comm_stream):
            ws_all_gather[(comm_wgs,)](
                IrisDeviceCtx, context_tensor, *comm_args, comm_counter, comm_total, cm, cn, comm_group_m,
                rank, world_size, num_warps, comm_even, num_warps=num_warps,
            )  # fmt: skip
        current.wait_stream(gemm_stream)
        current.wait_stream(comm_stream)
    return C, comm_dst
