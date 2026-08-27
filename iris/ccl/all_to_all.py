# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
All-to-all collective operation — public API.

Routes to triton/ or gluon/ based on config.use_gluon.
"""

from iris.ccl.utils import extract_group_info


def all_to_all(output_tensor, input_tensor, ctx, group=None, async_op=False, config=None):
    """
    All-to-all: each rank sends a chunk to every other rank.

    Input/output shape: (M, N * world_size).

    Args:
        output_tensor: Shape (M, N * world_size)
        input_tensor: Shape (M, N * world_size)
        ctx: Iris instance
        group: ProcessGroup or None
        async_op: If True, skip trailing barrier
        config: Config with kernel parameters
    """
    from iris.ccl.config import Config

    if config is None:
        config = Config(block_size_m=32, block_size_n=128)

    rank_in_group, rank_global, world_size, rank_start, rank_stride = extract_group_info(group, ctx)

    M, total_n = input_tensor.shape[:2]
    if total_n % world_size != 0:
        raise ValueError(f"Input width {total_n} must be divisible by world_size {world_size}")
    if output_tensor.shape[:2] != (M, total_n):
        raise ValueError(f"Output tensor shape {output_tensor.shape[:2]} does not match input shape {(M, total_n)}")

    if config.use_gluon and config.use_tdm:
        from iris.ccl.gluon.all_to_all_tdm import launch
    elif config.use_gluon:
        from iris.ccl.gluon.all_to_all import launch
    else:
        if config.use_tdm:
            raise ValueError("use_tdm=True requires use_gluon=True")
        from iris.ccl.triton.all_to_all import launch

    launch(
        input_tensor,
        output_tensor,
        ctx,
        rank_in_group,
        rank_global,
        world_size,
        rank_start,
        rank_stride,
        config,
    )

    if not async_op:
        ctx.barrier()
