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
The readers per DRAM bank
(``num_workers_per_dram_bank``: the builder's ``decode_dram_workers_per_bank``,
``default_decode_dram_workers``) go to both calls of one linear.
"""

from __future__ import annotations

import math
import os
from typing import Mapping

import ttnn


def _dram_bank_count(mesh_device) -> int:
    grid = mesh_device.dram_grid_size()
    if grid.y != 1:
        raise ValueError(f"DRAM weight sharding expects an Nx1 bank grid, got {grid.x}x{grid.y}")
    return grid.x


# The DRAM-sharded program reads each weight bank with one worker core; on Blackhole it also admits two readers per
# bank (``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig.num_workers_per_dram_bank``, in1 on NOC_0).  Two
# readers serve the projections in ``TWO_WORKER_PROJECTIONS`` by default (2026-09-16, 4x p150: bitwise at rows 1..32
# and at the model level; kernel us per call GDN in-proj 85 -> 67, GDN/QSA out 34 -> 28, QSA query-gate 59 -> 43,
# LM-head chunks 163/185 -> 93/107; 40.27 -> 38.67 ms per decode step); ``QWEN38_DRAM_WORKERS=1`` keeps one reader
# per bank everywhere (the one-reader caches keep their identity).  A bank's shard holds a whole number of tiles per
# reader, which widens only the GDN input projection (4160 columns: 17 -> 18 tiles per bank, 544 -> 576 columns).
WORKERS_ENV = "QWEN38_DRAM_WORKERS"
DEFAULT_WORKERS_PER_DRAM_BANK = 2
WORKERS_PER_DRAM_BANK = (1, 2)
# (K, N) -> activation storage cores of the decode linears that run two readers per bank: the GDN input (2560 x 4160)
# and output (1536 x 2560) projections, the QSA query-gate (2560 x 3072) and output (1536 x 2560) projections, the
# two LM-head chunk widths (2560 x 8192 and 7040).  The K/V/index projections, the router, the gated-residual linears
# and the shared expert keep one reader.
TWO_WORKER_PROJECTIONS: dict[tuple[int, int], int] = {
    (2560, 4160): 8,
    (1536, 2560): 16,
    (2560, 3072): 8,
    (2560, 8192): 40,
    (2560, 7040): 40,
}


def validate_decode_dram_workers(value) -> int:
    if type(value) is not int or value not in WORKERS_PER_DRAM_BANK:
        raise ValueError(f"decode DRAM workers per bank must be one of {WORKERS_PER_DRAM_BANK}, got {value!r}")
    return value


def default_decode_dram_workers(environ: Mapping[str, str] | None = None) -> int:
    """``QWEN38_DRAM_WORKERS`` (1 or 2) when set, else the serving default of two readers per bank."""

    raw = (os.environ if environ is None else environ).get(WORKERS_ENV, "").strip()
    if not raw:
        return DEFAULT_WORKERS_PER_DRAM_BANK
    if raw not in {str(value) for value in WORKERS_PER_DRAM_BANK}:
        raise ValueError(f"{WORKERS_ENV} must be one of {WORKERS_PER_DRAM_BANK}, got {raw!r}")
    return int(raw)


def bank_tiles(mesh_device, k: int, n: int, num_workers_per_dram_bank: int = 1) -> int:
    """Weight tiles per DRAM bank: ``n`` over the banks, padded to whole tiles per reader."""

    validate_decode_dram_workers(num_workers_per_dram_bank)
    banks = _dram_bank_count(mesh_device)
    if num_workers_per_dram_bank != 1 and (banks != 8 or (k, n) not in TWO_WORKER_PROJECTIONS):
        raise ValueError(f"two readers per DRAM bank are not qualified for banks={banks}, K={k}, N={n}")
    return math.ceil(n / (ttnn.TILE_SIZE * banks * num_workers_per_dram_bank)) * num_workers_per_dram_bank


def dram_sharded_weight_memory_config(mesh_device, k: int, n: int, *, num_workers_per_dram_bank: int = 1):
    """WIDTH_SHARDED DRAM layout for one local ``[1, 1, k, n]`` linear weight."""

    banks = _dram_bank_count(mesh_device)
    bank_width = bank_tiles(mesh_device, k, n, num_workers_per_dram_bank) * ttnn.TILE_SIZE
    bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    shard_spec = ttnn.ShardSpec(bank_grid, (k, bank_width), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def weight_layout_tag(mesh_device, k: int, n: int, *, num_workers_per_dram_bank: int) -> str:
    """``""`` when the bank shard is the one-reader layout, else ``_bank<columns>``: the cache-file suffix of a weight
    whose padding changed, so a tensorbin of the narrower layout is never loaded into the wider allocation
    (``ttnn.as_tensor`` loads a cached tensorbin in the layout it was written in)."""

    tiles = bank_tiles(mesh_device, k, n, num_workers_per_dram_bank)
    return "" if tiles == bank_tiles(mesh_device, k, n) else f"_bank{tiles * ttnn.TILE_SIZE}"


def validate_dram_sharded_weight(tensor, mesh_device, k: int, n: int, *, num_workers_per_dram_bank: int) -> None:
    """Every device member of ``tensor`` holds the bank layout its decode linear runs on (a stale tensorbin fails)."""

    expected = dram_sharded_weight_memory_config(mesh_device, k, n, num_workers_per_dram_bank=num_workers_per_dram_bank)
    members = ttnn.get_device_tensors(tensor)
    if not members:
        raise ValueError(f"decode weight K={k}, N={n} has no device members")
    for member in members:
        if member.memory_config() != expected:
            raise ValueError(
                f"decode weight K={k}, N={n} is not in the {num_workers_per_dram_bank}-reader DRAM bank layout "
                f"(a stale tensorbin?): {member.memory_config()} != {expected}"
            )


def dram_sharded_matmul_configs(mesh_device, k: int, n: int, *, num_cores: int, num_workers_per_dram_bank: int = 1):
    """Activation memory config plus program config for one decode linear.

    ``num_cores`` is the L1 storage grid holding the width-sharded activation;
    the kernel pins its compute workers to the DRAM banks holding the weight
    (``num_workers_per_dram_bank`` of them per bank), so the grid only needs to
    split ``k`` into whole tiles per core (which also fixes the in0 block width
    to the largest tile divisor).  ``per_core_N`` is the output storage width:
    the activation grid's share of ``n`` with one reader, the bank's tiles with
    two (the configuration the two-reader form was qualified with).
    """

    tiles = bank_tiles(mesh_device, k, n, num_workers_per_dram_bank)
    if num_workers_per_dram_bank != 1 and num_cores != TWO_WORKER_PROJECTIONS[(k, n)]:
        raise ValueError(
            f"two readers per bank for K={k}, N={n} were qualified with {TWO_WORKER_PROJECTIONS[(k, n)]} activation "
            f"storage cores, got {num_cores}"
        )
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
        per_core_N=math.ceil(n / (ttnn.TILE_SIZE * num_cores)) if num_workers_per_dram_bank == 1 else tiles,
        fused_activation=None,
        # the one-reader constructor call stays the call of 2026-09-04 (the field's own default is 1)
        **({} if num_workers_per_dram_bank == 1 else {"num_workers_per_dram_bank": num_workers_per_dram_bank}),
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
