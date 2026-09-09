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


def _largest_divisor(value: int, cap: int) -> int:
    return next(divisor for divisor in range(min(value, cap), 0, -1) if value % divisor == 0)


def prefill_matmul_program_config(mesh_device, rows: int, k: int, n: int):
    """The 2D-multicast program config of a ``[rows, k] x [k, n]`` prefill-slab linear on interleaved operands.

    The grid width is the one that keeps the widest output subblock (the runtime's automatic config is 2-2.5x slower
    on the K = 2560 linears), ``in0_block_w`` the largest divisor of the K tiles up to 8, the output subblock one tile
    high and up to four wide (fp32 accumulation halves the destination registers), ``per_core_M`` the row tiles over
    the grid's rows.  Measured 2026-09-09 on 4x p150: 10-12x the per-tile DRAM-sharded form at 2048 rows, within one
    bf16 ULP of it.
    """

    grid = mesh_device.compute_with_storage_grid_size()
    m_tiles, k_tiles, n_tiles = rows // ttnn.TILE_SIZE, k // ttnn.TILE_SIZE, math.ceil(n / ttnn.TILE_SIZE)
    if rows % ttnn.TILE_SIZE or k % ttnn.TILE_SIZE:
        raise ValueError(f"prefill linear needs whole row and K tiles, got rows={rows} k={k}")
    best_cols, best_key = 1, None
    for cols in range(1, min(int(grid.x), n_tiles) + 1):
        key = (_largest_divisor(math.ceil(n_tiles / cols), 4), cols)
        if best_key is None or key > best_key:
            best_key, best_cols = key, cols
    grid_rows = min(int(grid.y), m_tiles)
    per_core_n = math.ceil(n_tiles / best_cols)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(best_cols, grid_rows),
        in0_block_w=_largest_divisor(k_tiles, 8),
        out_subblock_h=1,
        out_subblock_w=_largest_divisor(per_core_n, 4),
        per_core_M=math.ceil(m_tiles / grid_rows),
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def prefill_linear(activation, weight, program_config, *, compute_kernel_config, dtype=None):
    """One prefill-slab linear: the resident DRAM-width-sharded ``weight`` copied interleaved (the 2D-multicast
    program admits a sharded in1 but reads it wrong: 96 % of the outputs, measured 2026-09-09), the matmul on the
    interleaved activation into an interleaved DRAM output, the copy released.  The caller's compute config and
    output dtype are the decode linear's."""

    weight_interleaved = ttnn.to_memory_config(weight, ttnn.DRAM_MEMORY_CONFIG)
    try:
        return ttnn.linear(
            activation,
            weight_interleaved,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            dtype=dtype,
        )
    finally:
        ttnn.deallocate(weight_interleaved)


def dram_sharded_row_tiles(rows, activation_memory_config) -> list:
    """The four 32-row tiles of an interleaved ``[1,1,128,W]`` activation (the long prefill chunk), each moved into
    a decode linear's activation shard (``None``: left interleaved, for the per-tile routed expert stream): the
    DRAM-sharded program admits one row tile per call (``per_core_M == 1``), so the 128-row activation runs four
    calls of the same program on the same rows.  The slice bounds are literal (the captured bodies carry no
    host-int-dependent shape op).  The caller deallocates the tiles."""

    width = rows.shape[3]
    if tuple(int(value) for value in rows.shape) != (1, 1, 4 * ttnn.TILE_SIZE, width):
        raise ValueError(f"row tiles need a [1,1,128,W] activation, got {list(rows.shape)}")
    tiles = []
    for tile in (
        ttnn.slice(rows, (0, 0, 0, 0), (1, 1, ttnn.TILE_SIZE, width), memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ttnn.slice(
            rows, (0, 0, ttnn.TILE_SIZE, 0), (1, 1, 2 * ttnn.TILE_SIZE, width), memory_config=ttnn.DRAM_MEMORY_CONFIG
        ),
        ttnn.slice(
            rows,
            (0, 0, 2 * ttnn.TILE_SIZE, 0),
            (1, 1, 3 * ttnn.TILE_SIZE, width),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        ttnn.slice(
            rows,
            (0, 0, 3 * ttnn.TILE_SIZE, 0),
            (1, 1, 4 * ttnn.TILE_SIZE, width),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    ):
        if activation_memory_config is None:
            tiles.append(tile)
            continue
        sharded = ttnn.to_memory_config(tile, activation_memory_config)
        ttnn.deallocate(tile)
        if tuple(int(value) for value in sharded.shape) != (1, 1, ttnn.TILE_SIZE, width):
            raise RuntimeError(f"row tile in the activation shard is {list(sharded.shape)}, expected [1,1,32,{width}]")
        tiles.append(sharded)
    return tiles
