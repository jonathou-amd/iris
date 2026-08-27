# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon TDM all-to-all for gfx1250/gfx1260.

Each persistent step performs one destination, one spatial tile:
    load  local input[:, d*N:(d+1)*N]
    store remote rank d output[:, g*N:(g+1)*N]

Single-buffered LDS: async_wait after load (RAW) and after store (WAR).
"""

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.amd.gfx1250 import tdm as gfx1250_tdm

    GFX1250_TDM_AVAILABLE = True
except ImportError as e:
    raise ValueError("Gluon TDM is not available. Install Triton with Gluon TDM support or set use_tdm=False.") from e

import torch

from iris.host.tracing.kernel_artifacts import iris_launch

TDM_ROW_BYTES = 256
TDM_MAX_DIM = 65535


@gluon.jit
def persistent_all_to_all_tdm_gfx1250(
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
    Persistent all-to-all via TDM: 1 load + 1 store per (destination, tile) step.

    Input/output are (M, N * world_size). Rank g sends input[:, d*N:(d+1)*N] to
    rank d's output[:, g*N:(g+1)*N].
    """
    pid = gl.program_id(0)

    dtype: gl.constexpr = input_ptr.dtype.element_ty
    smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[block_n, 8]], [block_m, block_n], [1, 0])
    smem = gl.allocate_shared_memory(dtype, [block_m, block_n], layout=smem_layout)

    n_total = N * world_size
    num_tiles_m = gl.cdiv(M, block_m)
    num_tiles_n = gl.cdiv(N, block_n)
    tiles_per_dest = num_tiles_m * num_tiles_n
    total_steps = world_size * tiles_per_dest

    input_desc = gfx1250_tdm.make_tensor_descriptor(
        base=input_ptr,
        shape=[M, n_total],
        strides=[stride_in_m, stride_in_n],
        block_shape=[block_m, block_n],
        layout=smem_layout,
    )

    for step in range(pid, total_steps, COMM_SMS):
        dest = step // tiles_per_dest
        tile_local = step % tiles_per_dest
        tile_m = tile_local // num_tiles_n
        tile_n = tile_local % num_tiles_n

        row_off = tile_m * block_m
        col_off = dest * N + tile_n * block_n
        out_col_off = group_rank * N + tile_n * block_n

        gfx1250_tdm.async_load(input_desc, [row_off, col_off], smem)
        gfx1250_tdm.async_wait(0)

        delta = gl.load(elem_deltas + dest)
        out_desc = gfx1250_tdm.make_tensor_descriptor(
            base=output_ptr + delta,
            shape=[M, n_total],
            strides=[stride_out_m, stride_out_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
        gfx1250_tdm.async_store(out_desc, [row_off, out_col_off], smem)
        gfx1250_tdm.async_wait(0)


def _max_lds_bytes(device_index: int = 0) -> int:
    """Return per-block LDS cap for the active Triton target (reflects LLVM backend)."""
    try:
        import triton.runtime.driver as triton_driver

        props = triton_driver.active.utils.get_device_properties(device_index)
        max_lds = int(props["max_shared_mem"])
        if max_lds > 0:
            return max_lds
    except Exception as exc:
        triton_err = exc
    else:
        triton_err = None

    try:
        import torch

        if torch.cuda.is_available():
            max_lds = int(torch.cuda.get_device_properties(device_index).shared_memory_per_block)
            if max_lds > 0:
                return max_lds
    except Exception:
        pass

    hint = (
        "Could not query max shared memory. "
        'Try: python3 -c "import triton.runtime.driver as d; '
        "print(d.active.utils.get_device_properties(0)['max_shared_mem'])\""
    )
    if triton_err is not None:
        raise RuntimeError(f"{hint} (Triton query failed: {triton_err})") from triton_err
    raise RuntimeError(hint)


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _validate_tdm_tile(config, elem_size: int, max_lds: int) -> None:
    block_m = config.block_size_m
    block_n = config.block_size_n

    if not _is_power_of_2(block_m) or not _is_power_of_2(block_n):
        raise ValueError(
            f"TDM block_size_m and block_size_n must each be a power of 2 "
            f"(PaddedSharedLayout requirement), got block_size_m={block_m}, block_size_n={block_n}."
        )

    if block_m > TDM_MAX_DIM or block_n > TDM_MAX_DIM:
        raise ValueError(
            f"TDM block_size_m and block_size_n must be <= {TDM_MAX_DIM}, "
            f"got block_size_m={block_m}, block_size_n={block_n}."
        )

    row_bytes = block_n * elem_size
    if row_bytes % TDM_ROW_BYTES != 0:
        raise ValueError(
            f"TDM inner tile width must be a multiple of {TDM_ROW_BYTES} bytes "
            f"(block_size_n={block_n}, elem_size={elem_size} -> {row_bytes} bytes/row)."
        )

    smem_bytes = block_m * (block_n + 8) * elem_size
    if smem_bytes > max_lds:
        raise ValueError(
            f"TDM tile LDS {smem_bytes} bytes exceeds device max {max_lds} bytes "
            f"(block_m={block_m}, block_n={block_n}, elem_size={elem_size}). "
            f"PaddedSharedLayout adds 8 elements to the inner dimension."
        )


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
    """Launch the Gluon TDM all-to-all kernel."""
    if not GFX1250_TDM_AVAILABLE:
        raise ValueError("TDM all-to-all requires GFX1250 TDM support (gfx1250/gfx1260 + Gluon TDM)")

    M, total_n = input_tensor.shape[:2]
    if total_n % world_size != 0:
        raise ValueError(f"Input width {total_n} must be divisible by world_size {world_size}")
    N = total_n // world_size

    if output_tensor.shape[:2] != (M, total_n):
        raise ValueError(f"Output shape {output_tensor.shape[:2]} does not match input shape {(M, total_n)}")

    elem_size = input_tensor.element_size()
    device_index = input_tensor.device.index
    if device_index is None:
        device_index = 0
    max_lds = _max_lds_bytes(device_index)
    _validate_tdm_tile(config, elem_size, max_lds)

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    heap_bases = ctx.get_heap_bases()
    local_base = heap_bases[rank_global]
    elem_deltas = torch.empty(world_size, dtype=torch.int64, device=input_tensor.device)
    for i in range(world_size):
        target_iris_rank = rank_start + i * rank_stride
        elem_deltas[i] = (heap_bases[target_iris_rank] - local_base) // elem_size

    iris_launch(
        persistent_all_to_all_tdm_gfx1250,
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
        algorithm="all_to_all_tdm",
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
