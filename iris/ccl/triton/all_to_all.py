# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Triton kernel for all-to-all collective communication.
"""

import triton
import triton.language as tl
import iris
from iris.host.tracing.kernel_artifacts import iris_launch
from ..utils import chiplet_transform_chunked


@triton.jit()
def persistent_all_to_all(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Persistent all-to-all kernel.

    Each rank sends input data to all ranks and receives data from all ranks.
    Similar to all-scatter but bidirectional.

    Args:
        input_ptr: Pointer to input tensor (local rank's data to send)
        output_ptr: Pointer to output tensor (will receive from all ranks)
        M: Number of rows
        N: Number of columns per rank (output will be N * world_size)
        stride_in_m, stride_in_n: Strides for input tensor
        stride_out_m, stride_out_n: Strides for output tensor
        heap_bases: Heap base pointers for all ranks
        group_rank: Rank within the ProcessGroup (0 to group_size-1), used for tile assignment and comparisons
        iris_rank: Rank in the iris context, used for iris RMA operations (heap_bases indexing)
        world_size: Total number of ranks in the group
        BLOCK_SIZE_M, BLOCK_SIZE_N: Block sizes for tiling
        GROUP_SIZE_M: Group size for M dimension tiling
        COMM_SMS: Number of SMs for communication
        NUM_XCDS: Number of XCDs
        CHUNK_SIZE: Chunk size for chiplet transform
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # Compute base indices for this tile
        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N

        # Check if this tile is fully within bounds (no edge cases)
        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        # Build indices (used by both paths)
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Pre-compute base offsets for better memory access patterns and vectorization
        input_base_m = rm[:, None] * stride_in_m
        output_base_m = rm[:, None] * stride_out_m
        input_base_n = rn[None, :] * stride_in_n
        output_base_n = rn[None, :] * stride_out_n

        # Fast path: NO MASKS (full tiles)
        # The masking is problem size dependent, and the compiler does not recognize it can have two paths
        # (one with masks and one without). Separate unmasked paths allow the compiler to generate
        # more efficient vectorized instructions.
        if is_full:
            # Process local rank first for better cache locality
            input_offset_local = input_base_m + (input_base_n + group_rank * N * stride_in_n)
            output_offset_local = output_base_m + (output_base_n + group_rank * N * stride_out_n)
            input_ptr_local = input_ptr + input_offset_local
            output_ptr_local = output_ptr + output_offset_local
            input_ptr_local = tl.multiple_of(input_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))
            output_ptr_local = tl.multiple_of(output_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            data = tl.load(input_ptr_local)
            tl.store(output_ptr_local, data, cache_modifier=".cs")

            # Process all remote ranks
            for i in range(world_size):
                target_rank = rank_start + i * rank_stride
                if i != group_rank:
                    # Calculate which chunk of input to read based on rank_in_group
                    rank_in_group_target = i
                    input_offset_remote = input_base_m + (input_base_n + rank_in_group_target * N * stride_in_n)
                    output_offset_remote = output_base_m + (output_base_n + group_rank * N * stride_out_n)
                    input_ptr_remote = input_ptr + input_offset_remote
                    output_ptr_remote = output_ptr + output_offset_remote
                    input_ptr_remote = tl.multiple_of(input_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))
                    output_ptr_remote = tl.multiple_of(output_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))

                    remote_data = tl.load(input_ptr_remote)
                    iris.store(
                        output_ptr_remote,
                        remote_data,
                        iris_rank,
                        target_rank,
                        heap_bases,
                        hint=(1, BLOCK_SIZE_N),
                        cache_modifier=".cs",
                    )

        # Slow path: MASKED (only boundary tiles land here)
        # This path handles tiles at tensor boundaries where not all elements are valid.
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)

            # Process local rank first for better cache locality
            input_offset_local = input_base_m + (input_base_n + group_rank * N * stride_in_n)
            output_offset_local = output_base_m + (output_base_n + group_rank * N * stride_out_n)
            input_ptr_local = input_ptr + input_offset_local
            output_ptr_local = output_ptr + output_offset_local
            input_ptr_local = tl.multiple_of(input_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))
            output_ptr_local = tl.multiple_of(output_ptr_local, (BLOCK_SIZE_M, BLOCK_SIZE_N))

            data = tl.load(input_ptr_local, mask=mask)
            tl.store(output_ptr_local, data, mask=mask, cache_modifier=".cs")

            # Process all remote ranks
            for i in range(world_size):
                target_rank = rank_start + i * rank_stride
                if i != group_rank:
                    # Calculate which chunk of input to read based on rank_in_group
                    rank_in_group_target = i
                    input_offset_remote = input_base_m + (input_base_n + rank_in_group_target * N * stride_in_n)
                    output_offset_remote = output_base_m + (output_base_n + group_rank * N * stride_out_n)
                    input_ptr_remote = input_ptr + input_offset_remote
                    output_ptr_remote = output_ptr + output_offset_remote
                    input_ptr_remote = tl.multiple_of(input_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))
                    output_ptr_remote = tl.multiple_of(output_ptr_remote, (BLOCK_SIZE_M, BLOCK_SIZE_N))

                    remote_data = tl.load(input_ptr_remote, mask=mask)
                    iris.store(
                        output_ptr_remote,
                        remote_data,
                        iris_rank,
                        target_rank,
                        heap_bases,
                        mask=mask,
                        hint=(1, BLOCK_SIZE_N),
                        cache_modifier=".cs",
                    )


@triton.jit()
def persistent_all_to_all_partitioned(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Persistent all-to-all kernel with rank-partitioned work distribution.

    Each PID is assigned to send tiles to a specific destination rank, and
    multiple PIDs partition the tiles for that destination. This avoids the
    inner loop over world_size.

    Work distribution (COMM_SMS must be divisible by world_size):
    - pids_per_rank = COMM_SMS // world_size
    - dest_rank_idx = pid // pids_per_rank
    - pid_in_rank_group = pid % pids_per_rank
    - Tiles for dest d: tile_id in range(pid_in_rank_group, total_tiles, pids_per_rank)

    For fixed dest d, each tile loads input[:, d*N:(d+1)*N] and stores to
    rank d's output[:, group_rank*N:(group_rank+1)*N].
    """
    pid = tl.program_id(0)

    pids_per_rank = COMM_SMS // world_size
    dest_rank_idx = pid // pids_per_rank
    pid_in_rank_group = pid % pids_per_rank
    target_rank = rank_start + dest_rank_idx * rank_stride

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    tl.assume(total_tiles > 0)

    for tile_id in range(pid_in_rank_group, total_tiles, pids_per_rank):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        is_full = (rm_base + BLOCK_SIZE_M <= M) & (rn_base + BLOCK_SIZE_N <= N)

        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        input_base_m = rm[:, None] * stride_in_m
        output_base_m = rm[:, None] * stride_out_m
        input_base_n = rn[None, :] * stride_in_n
        output_base_n = rn[None, :] * stride_out_n

        input_offset = input_base_m + (input_base_n + dest_rank_idx * N * stride_in_n)
        output_offset = output_base_m + (output_base_n + group_rank * N * stride_out_n)
        input_ptr_tile = input_ptr + input_offset
        output_ptr_tile = output_ptr + output_offset
        input_ptr_tile = tl.multiple_of(input_ptr_tile, (BLOCK_SIZE_M, BLOCK_SIZE_N))
        output_ptr_tile = tl.multiple_of(output_ptr_tile, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        if is_full:
            data = tl.load(input_ptr_tile)
            if dest_rank_idx == group_rank:
                tl.store(output_ptr_tile, data, cache_modifier=".cs")
            else:
                iris.store(
                    output_ptr_tile,
                    data,
                    iris_rank,
                    target_rank,
                    heap_bases,
                    hint=(1, BLOCK_SIZE_N),
                    cache_modifier=".cs",
                )
        else:
            mask = (rm[:, None] < M) & (rn[None, :] < N)
            data = tl.load(input_ptr_tile, mask=mask)
            if dest_rank_idx == group_rank:
                tl.store(output_ptr_tile, data, mask=mask, cache_modifier=".cs")
            else:
                iris.store(
                    output_ptr_tile,
                    data,
                    iris_rank,
                    target_rank,
                    heap_bases,
                    mask=mask,
                    hint=(1, BLOCK_SIZE_N),
                    cache_modifier=".cs",
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
    """Launch the Triton all-to-all kernel."""
    M, total_N = input_tensor.shape[:2]
    N = total_N // world_size

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    if config.all_to_all_variant == "partitioned" and config.comm_sms % world_size != 0:
        raise ValueError(
            f"For all_to_all_variant='partitioned', COMM_SMS ({config.comm_sms}) must be divisible by "
            f"world_size ({world_size}). Please adjust config.comm_sms to be a multiple of {world_size}."
        )

    if config.all_to_all_variant == "persistent":
        kernel_fn = persistent_all_to_all
        algorithm = "all_to_all"
    elif config.all_to_all_variant == "partitioned":
        kernel_fn = persistent_all_to_all_partitioned
        algorithm = "all_to_all_partitioned"
    else:
        raise ValueError(f"Unknown all_to_all_variant: {config.all_to_all_variant}")

    iris_launch(
        kernel_fn,
        (config.comm_sms,),
        input_tensor,
        output_tensor,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        ctx.get_heap_bases(),
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config.block_size_m,
        config.block_size_n,
        config.swizzle_size,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm=algorithm,
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
