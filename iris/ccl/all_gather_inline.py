# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-gather collective communication primitive for Iris.
Gathers tensors from all ranks and concatenates them along the last dimension.
"""

import triton
import triton.language as tl
from .config import Config

# Number of ranks we hoist heap bases for (fixed for SGPR-friendly codegen)
_NUM_RANKS_HOISTED: int = 8


@triton.jit
def _translate_with_bases(ptr, from_base, to_base):
    """Translate pointer from one heap base to another (same math as iris.__translate)."""
    ptr_int = tl.cast(ptr, tl.uint64)
    offset = ptr_int - from_base
    to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))
    translated_ptr_byte = to_base_byte + offset
    translated_ptr = tl.cast(translated_ptr_byte, ptr.dtype)
    translated_ptr = tl.multiple_of(translated_ptr, (32, 32))
    translated_ptr = tl.max_contiguous(translated_ptr, (1, 32))
    return translated_ptr


@triton.jit()
def persistent_all_gather_inline(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    CACHE_MODIFIER: tl.constexpr,
):
    """
    Persistent all-gather kernel.
    Each rank sends its input tensor to all ranks, and all ranks receive
    and concatenate all input tensors along dimension 0 (rows), matching
    torch.distributed.all_gather_into_tensor behavior.
    Args:
        input_ptr: Pointer to input tensor (local rank's data to send) of shape (M, N)
        output_ptr: Pointer to output tensor (will receive from all ranks) of shape (world_size * M, N)
        M: Number of rows per rank (output will be world_size * M rows)
        N: Number of columns
        stride_in_m, stride_in_n: Strides for input tensor
        stride_out_m, stride_out_n: Strides for output tensor
        heap_bases: Heap base pointers for all ranks
        cur_rank: Current rank
        world_size: Total number of ranks
        BLOCK_SIZE_M, BLOCK_SIZE_N: Block sizes for tiling
        GROUP_SIZE_M: Group size for M dimension tiling
        COMM_SMS: Number of SMs for communication
        NUM_XCDS: Number of XCDs
        CHUNK_SIZE: Chunk size for chiplet transform
    """
    pid = tl.program_id(0)

    # Hoist heap bases for all ranks into SGPRs (one s_load_dwordx16-style load)
    heap_base_r0 = tl.load(heap_bases + 0)
    heap_base_r1 = tl.load(heap_bases + 1)
    heap_base_r2 = tl.load(heap_bases + 2)
    heap_base_r3 = tl.load(heap_bases + 3)
    heap_base_r4 = tl.load(heap_bases + 4)
    heap_base_r5 = tl.load(heap_bases + 5)
    heap_base_r6 = tl.load(heap_bases + 6)
    heap_base_r7 = tl.load(heap_bases + 7)
    my_base = heap_base_r0
    if cur_rank == 1:
        my_base = heap_base_r1
    if cur_rank == 2:
        my_base = heap_base_r2
    if cur_rank == 3:
        my_base = heap_base_r3
    if cur_rank == 4:
        my_base = heap_base_r4
    if cur_rank == 5:
        my_base = heap_base_r5
    if cur_rank == 6:
        my_base = heap_base_r6
    if cur_rank == 7:
        my_base = heap_base_r7

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n
    tl.assume(total_tiles > 0)
    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        tl.assume(tile_id >= 0)
        tl.assume(stride_in_m >= 0)
        tl.assume(stride_in_n >= 0)
        tl.assume(stride_out_m >= 0)
        tl.assume(stride_out_n >= 0)

        # Compute local row and column indices for input tensor
        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm_input = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm_input = tl.max_contiguous(tl.multiple_of(rm_input, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Mask for local input bounds
        input_mask = (rm_input[:, None] < M) & (rn[None, :] < N)

        # Compute input offset and load local shard data once
        input_base_m = rm_input[:, None] * stride_in_m
        input_base_n = rn[None, :] * stride_in_n
        input_offset = input_base_m + input_base_n
        input_ptr_source = input_ptr + input_offset
        input_ptr_source = tl.multiple_of(input_ptr_source, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        # Load local input data once for this tile
        data = tl.load(input_ptr_source, mask=input_mask, other=0.0)

        # Send local shard data to all destination ranks via translated pointers (no rank branch)
        # Each rank's input goes to output[cur_rank * M : (cur_rank + 1) * M, :] on all ranks
        rm_output = rm_input + cur_rank * M
        output_mask = (rm_output[:, None] < (cur_rank + 1) * M) & (rn[None, :] < N)
        combined_mask = input_mask & output_mask
        output_base_m = rm_output[:, None] * stride_out_m
        output_base_n = rn[None, :] * stride_out_n
        output_offset = output_base_m + output_base_n
        output_ptr_target = output_ptr + output_offset
        output_ptr_target = tl.multiple_of(output_ptr_target, (BLOCK_SIZE_M, BLOCK_SIZE_N))

        dest_0 = _translate_with_bases(output_ptr_target, my_base, heap_base_r0)
        dest_1 = _translate_with_bases(output_ptr_target, my_base, heap_base_r1)
        dest_2 = _translate_with_bases(output_ptr_target, my_base, heap_base_r2)
        dest_3 = _translate_with_bases(output_ptr_target, my_base, heap_base_r3)
        dest_4 = _translate_with_bases(output_ptr_target, my_base, heap_base_r4)
        dest_5 = _translate_with_bases(output_ptr_target, my_base, heap_base_r5)
        dest_6 = _translate_with_bases(output_ptr_target, my_base, heap_base_r6)
        dest_7 = _translate_with_bases(output_ptr_target, my_base, heap_base_r7)

        tl.store(dest_0, data, mask=combined_mask, cache_modifier=CACHE_MODIFIER)
        tl.store(dest_1, data, mask=combined_mask, cache_modifier=CACHE_MODIFIER)
        tl.store(dest_2, data, mask=combined_mask, cache_modifier=CACHE_MODIFIER)
        tl.store(dest_3, data, mask=combined_mask, cache_modifier=CACHE_MODIFIER)
        tl.store(dest_4, data, mask=combined_mask, cache_modifier=CACHE_MODIFIER)
        tl.store(dest_5, data, mask=combined_mask, cache_modifier=CACHE_MODIFIER)
        tl.store(dest_6, data, mask=combined_mask, cache_modifier=CACHE_MODIFIER)
        tl.store(dest_7, data, mask=combined_mask, cache_modifier=CACHE_MODIFIER)


def all_gather_inline(output_tensor, input_tensor, shmem, config=None, async_op=False):
    """
    Internal all-gather collective operation implementation.
    This function is called internally by shmem.ccl.all_gather().
    Users should use the Iris instance method instead:
        >>> shmem.ccl.all_gather(output_tensor, input_tensor)
    Each rank sends its input tensor to all ranks, and all ranks receive
    and concatenate all input tensors along dimension 0 (rows), matching
    torch.distributed.all_gather_into_tensor behavior.
    Args:
        output_tensor: Output tensor of shape (world_size * M, N) - will contain concatenated inputs
        input_tensor: Input tensor of shape (M, N) - local rank's data to send
        shmem: Iris shmem context
        config: Config instance with kernel parameters (default: None).
                If None, uses default Config values.
        async_op: If False, performs a barrier at the end. If True, returns immediately.
                  Default: False.
    """
    # Use provided config or create default one
    if config is None:
        config = Config(block_size_m=32, block_size_n=64)

    # Check for unsupported options
    if config.use_gluon:
        raise ValueError(
            "all_gather does not support use_gluon=True. "
            "Gluon implementation is not available for all_gather. "
            "Use default config (use_gluon=False)."
        )

    rank = shmem.get_rank()
    world_size = shmem.get_num_ranks()
    if world_size != _NUM_RANKS_HOISTED:
        raise ValueError(
            f"all_gather currently requires world_size == {_NUM_RANKS_HOISTED} (got {world_size}). "
            "Use a build with the dynamic rank loop for other world sizes."
        )

    M, N = input_tensor.shape[:2]
    expected_output_shape = (world_size * M, N)

    if output_tensor.shape[:2] != expected_output_shape:
        raise ValueError(
            f"Output tensor shape {output_tensor.shape[:2]} does not match expected shape {expected_output_shape}. "
            f"Expected (world_size * M, N) = ({world_size * M}, {N})"
        )

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    heap_bases = shmem.get_heap_bases()

    persistent_all_gather_inline[(config.comm_sms,)](
        input_tensor,
        output_tensor,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        heap_bases,
        rank,
        world_size,
        config.block_size_m,
        config.block_size_n,
        config.swizzle_size,
        config.comm_sms,
        config.num_xcds,
        config.chunk_size,
        config.cache_modifier,
    )

    if not async_op:
        shmem.barrier()