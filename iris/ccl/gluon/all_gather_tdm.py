# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon TDM all-gather for gfx1250/gfx1260.

Two variants (select via Config.all_gather_tdm_variant):

hoisted — persistent_all_gather_tdm_gfx1250
    Each tile loop iteration: 1 HBM->LDS load + world_size stores (descriptors
    hoisted at kernel entry; world_size <= 8).

stepwise — persistent_all_gather_tdm_gfx1250_stepwise
    Same tile-parallel CTA assignment as hoisted/Triton: each CTA loads a tile
    once then stores to all ranks with dynamically built output descriptors
    (arbitrary world_size).
"""

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.amd.gfx1250 import tdm as gfx1250_tdm

    GFX1250_TDM_AVAILABLE = True
except ImportError as e:
    raise ValueError("Gluon TDM is not available. Install Triton with Gluon TDM support or set use_tdm=False.") from e

import torch

from iris.ccl.gluon.all_to_all_tdm import _max_lds_bytes, _validate_tdm_tile
from iris.host.tracing.kernel_artifacts import iris_launch


@gluon.jit
def persistent_all_gather_tdm_gfx1250(
    input_ptr,
    output_ptr,
    elem_deltas,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    group_rank: gl.constexpr,
    world_size: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    COMM_SMS: gl.constexpr,
):
    """
    All-gather via TDM: 1 HBM->LDS load + world_size HBM/XGMI stores per tile.

    Rank g writes its input tiles to output[g*M : (g+1)*M, :] on every rank.
    """
    pid = gl.program_id(0)

    dtype: gl.constexpr = input_ptr.dtype.element_ty
    smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[block_n, 8]], [block_m, block_n], [1, 0])
    smem = gl.allocate_shared_memory(dtype, [block_m, block_n], layout=smem_layout)

    out_m = M * world_size
    num_tiles_m = gl.cdiv(M, block_m)
    num_tiles_n = gl.cdiv(N, block_n)
    total_tiles = num_tiles_m * num_tiles_n

    input_desc = gfx1250_tdm.make_tensor_descriptor(
        base=input_ptr,
        shape=[M, N],
        strides=[stride_in_m, stride_in_n],
        block_shape=[block_m, block_n],
        layout=smem_layout,
    )

    # Hoist per-destination output descriptors (traffic-shaped store order).
    if world_size > 0:
        d0 = gl.load(elem_deltas + ((group_rank + 0) % world_size))
        out_desc_0 = gfx1250_tdm.make_tensor_descriptor(
            base=output_ptr + d0,
            shape=[out_m, N],
            strides=[stride_out_m, stride_out_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 1:
        d1 = gl.load(elem_deltas + ((group_rank + 1) % world_size))
        out_desc_1 = gfx1250_tdm.make_tensor_descriptor(
            base=output_ptr + d1,
            shape=[out_m, N],
            strides=[stride_out_m, stride_out_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 2:
        d2 = gl.load(elem_deltas + ((group_rank + 2) % world_size))
        out_desc_2 = gfx1250_tdm.make_tensor_descriptor(
            base=output_ptr + d2,
            shape=[out_m, N],
            strides=[stride_out_m, stride_out_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 3:
        d3 = gl.load(elem_deltas + ((group_rank + 3) % world_size))
        out_desc_3 = gfx1250_tdm.make_tensor_descriptor(
            base=output_ptr + d3,
            shape=[out_m, N],
            strides=[stride_out_m, stride_out_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 4:
        d4 = gl.load(elem_deltas + ((group_rank + 4) % world_size))
        out_desc_4 = gfx1250_tdm.make_tensor_descriptor(
            base=output_ptr + d4,
            shape=[out_m, N],
            strides=[stride_out_m, stride_out_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 5:
        d5 = gl.load(elem_deltas + ((group_rank + 5) % world_size))
        out_desc_5 = gfx1250_tdm.make_tensor_descriptor(
            base=output_ptr + d5,
            shape=[out_m, N],
            strides=[stride_out_m, stride_out_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 6:
        d6 = gl.load(elem_deltas + ((group_rank + 6) % world_size))
        out_desc_6 = gfx1250_tdm.make_tensor_descriptor(
            base=output_ptr + d6,
            shape=[out_m, N],
            strides=[stride_out_m, stride_out_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 7:
        d7 = gl.load(elem_deltas + ((group_rank + 7) % world_size))
        out_desc_7 = gfx1250_tdm.make_tensor_descriptor(
            base=output_ptr + d7,
            shape=[out_m, N],
            strides=[stride_out_m, stride_out_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )

    for tile_id in range(pid, total_tiles, COMM_SMS):
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n
        row_off = tile_m * block_m
        col_off = tile_n * block_n
        out_row_off = group_rank * M + row_off

        gfx1250_tdm.async_load(input_desc, [row_off, col_off], smem)
        gfx1250_tdm.async_wait(0)

        if world_size > 0:
            gfx1250_tdm.async_store(out_desc_0, [out_row_off, col_off], smem)
        if world_size > 1:
            gfx1250_tdm.async_store(out_desc_1, [out_row_off, col_off], smem)
        if world_size > 2:
            gfx1250_tdm.async_store(out_desc_2, [out_row_off, col_off], smem)
        if world_size > 3:
            gfx1250_tdm.async_store(out_desc_3, [out_row_off, col_off], smem)
        if world_size > 4:
            gfx1250_tdm.async_store(out_desc_4, [out_row_off, col_off], smem)
        if world_size > 5:
            gfx1250_tdm.async_store(out_desc_5, [out_row_off, col_off], smem)
        if world_size > 6:
            gfx1250_tdm.async_store(out_desc_6, [out_row_off, col_off], smem)
        if world_size > 7:
            gfx1250_tdm.async_store(out_desc_7, [out_row_off, col_off], smem)

        gfx1250_tdm.async_wait(0)


@gluon.jit
def persistent_all_gather_tdm_gfx1250_stepwise(
    input_ptr,
    output_ptr,
    elem_deltas,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    group_rank: gl.constexpr,
    world_size: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    COMM_SMS: gl.constexpr,
):
    """
    All-gather via TDM: same structure as hoisted/Triton all-gather.

    Outer loop: tile_id = pid, pid + COMM_SMS, ... (one load per tile per CTA).
    Inner loop: traffic-shaped stores to all ranks with a dynamically built
    output descriptor per destination (no world_size unroll cap).
    """
    pid = gl.program_id(0)

    dtype: gl.constexpr = input_ptr.dtype.element_ty
    smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[block_n, 8]], [block_m, block_n], [1, 0])
    smem = gl.allocate_shared_memory(dtype, [block_m, block_n], layout=smem_layout)

    out_m = M * world_size
    num_tiles_m = gl.cdiv(M, block_m)
    num_tiles_n = gl.cdiv(N, block_n)
    total_tiles = num_tiles_m * num_tiles_n

    input_desc = gfx1250_tdm.make_tensor_descriptor(
        base=input_ptr,
        shape=[M, N],
        strides=[stride_in_m, stride_in_n],
        block_shape=[block_m, block_n],
        layout=smem_layout,
    )

    for tile_id in range(pid, total_tiles, COMM_SMS):
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n
        row_off = tile_m * block_m
        col_off = tile_n * block_n
        out_row_off = group_rank * M + row_off

        gfx1250_tdm.async_load(input_desc, [row_off, col_off], smem)
        #gfx1250_tdm.async_wait(0)

        for dest_idx in gl.static_range(world_size):
            dest_group_rank = (group_rank + dest_idx) % world_size
            delta = gl.load(elem_deltas + dest_group_rank)
            out_desc = gfx1250_tdm.make_tensor_descriptor(
                base=output_ptr + delta,
                shape=[out_m, N],
                strides=[stride_out_m, stride_out_n],
                block_shape=[block_m, block_n],
                layout=smem_layout,
            )
            gfx1250_tdm.async_store(out_desc, [out_row_off, col_off], smem)

        #gfx1250_tdm.async_wait(0)


def _build_elem_deltas(input_tensor, ctx, rank_global, world_size, rank_start, rank_stride):
    heap_bases = ctx.get_heap_bases()
    local_base = heap_bases[rank_global]
    elem_size = input_tensor.element_size()
    elem_deltas = torch.empty(world_size, dtype=torch.int64, device=input_tensor.device)
    for i in range(world_size):
        target_iris_rank = rank_start + i * rank_stride
        elem_deltas[i] = (heap_bases[target_iris_rank] - local_base) // elem_size
    return elem_deltas


def launch(
    input_tensor,
    output_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    config,
):
    """Launch the Gluon TDM all-gather kernel."""
    if not GFX1250_TDM_AVAILABLE:
        raise ValueError("TDM all-gather requires GFX1250 TDM support (gfx1250/gfx1260 + Gluon TDM)")

    if config.all_gather_variant != "persistent":
        raise ValueError(
            f"TDM all_gather only supports all_gather_variant='persistent', got '{config.all_gather_variant}'."
        )

    tdm_variant = config.all_gather_tdm_variant
    if tdm_variant == "hoisted":
        kernel = persistent_all_gather_tdm_gfx1250
        algorithm = "all_gather_tdm"
    elif tdm_variant == "stepwise":
        kernel = persistent_all_gather_tdm_gfx1250_stepwise
        algorithm = "all_gather_tdm_stepwise"
    else:
        raise ValueError(f"Unknown all_gather_tdm_variant: {tdm_variant}")

    M, N = input_tensor.shape[:2]
    expected_output_shape = (world_size * M, N)
    if output_tensor.shape[:2] != expected_output_shape:
        raise ValueError(f"Output shape {output_tensor.shape[:2]} does not match expected {expected_output_shape}")

    if tdm_variant == "hoisted" and world_size > 8:
        raise ValueError(f"TDM all-gather (hoisted) supports world_size <= 8, got {world_size}")

    elem_size = input_tensor.element_size()
    device_index = input_tensor.device.index
    if device_index is None:
        device_index = 0
    max_lds = _max_lds_bytes(device_index)
    _validate_tdm_tile(config, elem_size, max_lds)

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    elem_deltas = _build_elem_deltas(input_tensor, ctx, rank_global, world_size, rank_start, rank_stride)

    iris_launch(
        kernel,
        (config.comm_sms,),
        input_tensor,
        output_tensor,
        elem_deltas,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        rank_in_group,
        world_size,
        config.block_size_m,
        config.block_size_n,
        config.comm_sms,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm=algorithm,
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
