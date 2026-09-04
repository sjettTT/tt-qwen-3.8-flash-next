# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""DRAM-sharded decode matmul configuration shared by the device modules.

Every converted decode ``ttnn.linear`` in this model multiplies one padded
tile row of activations (M == one tile) by a resident weight.  The default
interleaved program serializes one small in1 DRAM read per K block regardless
of N or grid, so kernel time scales with K blocks alone; the DRAM-sharded
program config (the tt_transformers decode pattern) instead streams each
weight shard from its own DRAM bank into worker cores next to that bank.  The
operands, dtypes, and per-element sequential K accumulation under the
caller's compute kernel config are unchanged.

Usage: upload the weight with :func:`dram_sharded_weight_memory_config`, move
the activation to the activation config returned by
:func:`dram_sharded_matmul_configs`, and call ``ttnn.linear`` with the
returned program config and ``memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG``.
"""

from __future__ import annotations

import math

import ttnn


def _dram_bank_count(mesh_device) -> int:
    grid = mesh_device.dram_grid_size()
    if grid.y != 1:
        raise ValueError(f"DRAM weight sharding expects an Nx1 bank grid, got {grid.x}x{grid.y}")
    return grid.x


def dram_sharded_weight_memory_config(mesh_device, k: int, n: int):
    """WIDTH_SHARDED DRAM layout for one local ``[1, 1, k, n]`` linear weight."""

    banks = _dram_bank_count(mesh_device)
    padded_n = math.ceil(n / (ttnn.TILE_SIZE * banks)) * (ttnn.TILE_SIZE * banks)
    bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    shard_spec = ttnn.ShardSpec(bank_grid, (k, padded_n // banks), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def dram_sharded_matmul_configs(mesh_device, k: int, n: int, *, num_cores: int):
    """Activation memory config plus program config for one decode linear.

    ``num_cores`` is the L1 storage grid holding the width-sharded activation
    and output; the kernel pins its compute workers to the DRAM banks holding
    the weight, so the grid only needs to split ``k`` into whole tiles per
    core (which also fixes the in0 block width to the largest tile divisor).
    """

    _dram_bank_count(mesh_device)
    if num_cores <= 8:
        storage_grid = ttnn.CoreGrid(x=num_cores, y=1)
    elif num_cores % 8 == 0 and num_cores <= 64:
        storage_grid = ttnn.CoreGrid(x=8, y=num_cores // 8)
    else:
        raise ValueError(f"decode matmul storage grid must be rectangular over 8 columns, got {num_cores} cores")
    if k % (ttnn.TILE_SIZE * num_cores):
        raise ValueError(f"K={k} does not split into whole tiles over {num_cores} storage cores")
    k_tiles_per_core = k // (ttnn.TILE_SIZE * num_cores)
    in0_block_w = next(width for width in range(8, 0, -1) if k_tiles_per_core % width == 0)
    activation_memory_config = ttnn.create_sharded_memory_config(
        (ttnn.TILE_SIZE, k // num_cores),
        storage_grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=1,
        per_core_N=math.ceil(n / (ttnn.TILE_SIZE * num_cores)),
        fused_activation=None,
    )
    return activation_memory_config, program_config
