# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel Qwen3.8-Flash-Next Gated DeltaNet.

This is the device implementation of the pinned ``Qwen4Exp`` linear-attention
block, not the CPU oracle.  It deliberately has a narrow boundary:

* input and output are hidden-width shards on the columns of an exact 1x4 mesh;
* Q/K heads (16) and value heads (48) are split 4/12 per device;
* convolution and recurrent state are genuinely head sharded, never represented
  by four replicated tensors that happen to contain different values;
* the recurrent state remains FP32 and is updated in preallocated buffers;
* the Qwen3.8 output gate is ``sigmoid(z)`` (Qwen3.6 uses ``silu(z)``);
* the row-parallel output is reduced and scattered back to H/4.

The ordinary-decode path is the primary bring-up path.  ``forward_prefill`` is
an exact device-only serial baseline over that same transition.  A future
chunk-prefill optimization may replace it only after matching this state and
output contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import INDEX_SHA256, Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.tt.gdn import Qwen38GDNWeights
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    MESH_SHAPE,
    Qwen38MeshContract,
    TensorPlacement,
    replicate_tensor_2d_mesh_mapper,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.decode_matmul import (
    dram_sharded_matmul_configs,
    dram_sharded_weight_memory_config,
)

TP_SIZE = 4
TP_AXIS = 1
HIDDEN_SIZE = 2560
HIDDEN_SIZE_PER_DEVICE = HIDDEN_SIZE // TP_SIZE
QK_HEADS = 16
QK_HEADS_PER_DEVICE = QK_HEADS // TP_SIZE
VALUE_HEADS = 48
VALUE_HEADS_PER_DEVICE = VALUE_HEADS // TP_SIZE
HEAD_DIM = 128
QK_WIDTH = QK_HEADS * HEAD_DIM
QK_WIDTH_PER_DEVICE = QK_HEADS_PER_DEVICE * HEAD_DIM
VALUE_WIDTH = VALUE_HEADS * HEAD_DIM
VALUE_WIDTH_PER_DEVICE = VALUE_HEADS_PER_DEVICE * HEAD_DIM
QKV_WIDTH = 2 * QK_WIDTH + VALUE_WIDTH
QKV_WIDTH_PER_DEVICE = 2 * QK_WIDTH_PER_DEVICE + VALUE_WIDTH_PER_DEVICE
CONV_KERNEL_SIZE = 4
QK_REPEAT_FACTOR = VALUE_HEADS // QK_HEADS
# Checkpoint order [q|k|v|z|a|b] packs to 4120 columns.  On device the packed
# weight keeps that order but starts ``a`` and ``b`` on tile boundaries, with
# zero weight columns between, so the decode split is four tile-aligned slices.
QKVZAB_WIDTH_PER_DEVICE = QKV_WIDTH_PER_DEVICE + VALUE_WIDTH_PER_DEVICE + 2 * VALUE_HEADS_PER_DEVICE
A_COLUMN = QKV_WIDTH_PER_DEVICE + VALUE_WIDTH_PER_DEVICE
B_COLUMN = A_COLUMN + ttnn.TILE_SIZE
PROJECTION_WIDTH_PER_DEVICE = B_COLUMN + ttnn.TILE_SIZE
RMS_NORM_EPS = 1.0e-6
# FLA l2 normalization of q/k: rms_norm with eps / head_dim, then head_dim ** -0.5.
QK_L2_NORM_EPS = 1.0e-6
# Multi-row (MTP verify) path: the chunk kernel is handed exactly one full chunk of
# CHUNK_SIZE rows; the rows past the real ones are zeroed on device.  The FIR
# history is the CONV_KERNEL_SIZE - 1 rows before the first new row.
CHUNK_SIZE = ttnn.TILE_SIZE
CONV_HISTORY_ROWS = CONV_KERNEL_SIZE - 1
CONV_WINDOW_ROWS = CONV_HISTORY_ROWS + CHUNK_SIZE
# The rows path keeps the FIR history in a full 32-row tile (rows 0..2 hold the history, rows 3..31
# are never read) so ``[history | qkv]`` is a tile-aligned concat of two tiles; every row read from
# that 64-row window (the FIR taps, the next history) is an exact 0/1 selection matmul against a
# constant, never a row-unaligned slice.  Logical window row m (m = 0..34: the three history rows,
# then the CHUNK_SIZE new rows) sits at buffer row ``_window_buffer_row(m)``.
CONV_WINDOW_TILE_ROWS = 2 * CHUNK_SIZE
CHUNK_QUADRANT_MASK_WIDTH = 3 * ttnn.TILE_SIZE


def _window_buffer_row(logical_row: int) -> int:
    if logical_row < CONV_HISTORY_ROWS:
        return logical_row
    return CHUNK_SIZE + logical_row - CONV_HISTORY_ROWS


def rows_window_select_tiles(rows: int) -> dict[str, torch.Tensor]:
    """Host images of the rows path's exact 0/1 selection matrices (one nonzero per selected element).

    ``conv_taps[t]`` ``[32, 64]``: row j selects logical window row t + j (FIR tap t, t = 0..2; tap 3 is
    the new rows themselves).  ``history_select_stack`` ``[32, 2048]``: row a is the flattened ``[32, 64]``
    select whose rows 0..2 pick logical window rows a + 1 .. a + 3 (the next history after committing
    a + 1 rows; rows 3..31 are zero).  ``qk_expand`` ``[512, 1536]``: the GQA expansion, key head h to
    value heads 3h .. 3h + 2 (one 1.0 per output column), so the chunk kernel sees H = HV = 12 heads and
    skips its own repeat_interleave.
    """

    taps = torch.zeros(CONV_HISTORY_ROWS, CHUNK_SIZE, CONV_WINDOW_TILE_ROWS)
    for tap in range(CONV_HISTORY_ROWS):
        for row in range(CHUNK_SIZE):
            taps[tap, row, _window_buffer_row(tap + row)] = 1.0
    stack = torch.zeros(CHUNK_SIZE, CHUNK_SIZE * CONV_WINDOW_TILE_ROWS)
    for accepted in range(rows):
        select = torch.zeros(CHUNK_SIZE, CONV_WINDOW_TILE_ROWS)
        for index in range(CONV_HISTORY_ROWS):
            select[index, _window_buffer_row(accepted + 1 + index)] = 1.0
        stack[accepted] = select.reshape(-1)
    expand = torch.zeros(QK_WIDTH_PER_DEVICE, VALUE_WIDTH_PER_DEVICE)
    for head in range(QK_HEADS_PER_DEVICE):
        for repeat in range(QK_REPEAT_FACTOR):
            value_head = head * QK_REPEAT_FACTOR + repeat
            expand[head * HEAD_DIM : (head + 1) * HEAD_DIM, value_head * HEAD_DIM : (value_head + 1) * HEAD_DIM] = (
                torch.eye(HEAD_DIM)
            )
    return {"conv_taps": taps, "history_select_stack": stack, "qk_expand": expand}


def pack_projection_columns(shards) -> torch.Tensor:
    """Pack the four device-local ``[q|k|v|z|a|b]`` streams into one ``[1,1,K,4*N]`` linear weight.

    Each device block is ``[HIDDEN_SIZE, PROJECTION_WIDTH_PER_DEVICE]``: columns up to
    ``A_COLUMN + 12`` are the checkpoint order, ``b`` moves to ``B_COLUMN`` and the
    two 20-column gaps are zero.  Zero weight columns produce zero outputs that
    the decode split never reads.
    """

    a_end = A_COLUMN + VALUE_HEADS_PER_DEVICE
    blocks = []
    for shard in shards:
        fused = shard.fused_qkvzab.transpose(0, 1)
        expected = (HIDDEN_SIZE, QKVZAB_WIDTH_PER_DEVICE)
        if tuple(fused.shape) != expected:
            raise RuntimeError(f"fused qkvzab shard shape {tuple(fused.shape)} != {expected}")
        block = torch.zeros(HIDDEN_SIZE, PROJECTION_WIDTH_PER_DEVICE, dtype=fused.dtype)
        block[:, :a_end] = fused[:, :a_end]
        block[:, B_COLUMN : B_COLUMN + VALUE_HEADS_PER_DEVICE] = fused[:, a_end:QKVZAB_WIDTH_PER_DEVICE]
        blocks.append(block)
    return torch.cat(blocks, dim=1).reshape(1, 1, HIDDEN_SIZE, len(blocks) * PROJECTION_WIDTH_PER_DEVICE)


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def _tensor_key(tensor) -> tuple[str, int]:
    tensor_id = getattr(tensor, "tensor_id", None)
    if callable(tensor_id):
        tensor_id = tensor_id()
    return ("ttnn", int(tensor_id)) if tensor_id is not None else ("python", id(tensor))


def _deallocate(*tensors) -> None:
    """Deallocate each distinct backing tensor at most once."""

    seen: set[tuple[str, int]] = set()
    for tensor in tensors:
        if tensor is None:
            continue
        key = _tensor_key(tensor)
        if key in seen:
            continue
        seen.add(key)
        ttnn.deallocate(tensor)


def _copy_inplace(source, target, *, label: str) -> None:
    """Copy into a persistent target and reject an address-changing result."""

    target_id = _tensor_key(target)
    copied = ttnn.copy(source, target)
    if _tensor_key(target) != target_id:
        raise RuntimeError(f"{label} target address changed during ttnn.copy")
    if copied is not None and _tensor_key(copied) != target_id:
        raise RuntimeError(f"{label} ttnn.copy returned a different tensor")


def _require_shape(tensor, expected: tuple[int, ...], *, label: str) -> None:
    actual = _shape(tensor)
    if actual != expected:
        raise RuntimeError(f"{label} local shape must be {expected}, got {actual}")


def _retag_head_shard_after_reshape(tensor, *, reference, shard_dim: int) -> None:
    """Record the new logical head axis after a local-only reshape.

    TTNN reshape/repeat currently preserve the input ``PlacementShard`` axis.
    GDN splits the fused last-axis shard into ``[..., heads, head_dim]`` without
    moving bytes between devices, so the same physical shard becomes the head
    axis in the reshaped tensor.
    """

    topology = reference.tensor_topology()
    tensor.update_tensor_topology(
        ttnn.TensorTopology(
            topology.distribution_shape(),
            [ttnn.PlacementReplicate(), ttnn.PlacementShard(shard_dim)],
            topology.mesh_coords(),
        )
    )


def _allocate_head_sharded_zero(
    mesh_device,
    mesh_contract: Qwen38MeshContract,
    *,
    local_shape: tuple[int, ...],
    dtype,
    shard_dim: int,
    label: str,
):
    """Allocate one zero-filled local tensor per device and record its TP shard.

    ``moreh_full`` avoids the host-to-device zero upload while preserving the
    exact local storage that the former ``ShardTensor2dMesh`` path produced.
    The fill initially reports replicated placement metadata; changing that
    metadata to the known head shard is local-only and moves no bytes.
    """

    tensor = ttnn.moreh_full(
        list(local_shape),
        0.0,
        mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    try:
        topology = tensor.tensor_topology()
        distribution_shape = tuple(int(value) for value in topology.distribution_shape())
        mesh_coords = tuple(tuple(int(value) for value in coord) for coord in topology.mesh_coords())
        placement_names = tuple(type(value).__name__ for value in topology.placements())
        expected_coords = tuple((0, column) for column in range(TP_SIZE))
        if distribution_shape != MESH_SHAPE or mesh_coords != expected_coords:
            raise RuntimeError(
                f"{label} device allocation has unexpected topology: "
                f"distribution={distribution_shape} coordinates={mesh_coords}"
            )
        if placement_names != ("PlacementReplicate", "PlacementReplicate"):
            raise RuntimeError(f"{label} device allocation is not initially replicated: {placement_names}")

        tensor.update_tensor_topology(
            ttnn.TensorTopology(
                topology.distribution_shape(),
                [ttnn.PlacementReplicate(), ttnn.PlacementShard(shard_dim)],
                topology.mesh_coords(),
            )
        )
        mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=shard_dim)
        local_tensors = tuple(ttnn.get_device_tensors(tensor))
        if len(local_tensors) != TP_SIZE:
            raise RuntimeError(f"{label} device allocation has {len(local_tensors)} locals, expected {TP_SIZE}")
        for index, local in enumerate(local_tensors):
            if _shape(local) != local_shape:
                raise RuntimeError(f"{label} local {index} shape must be {local_shape}, got {_shape(local)}")
            if local.dtype != dtype or local.layout != ttnn.TILE_LAYOUT:
                raise RuntimeError(f"{label} local {index} has unexpected dtype/layout")
            if local.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                raise RuntimeError(f"{label} local {index} must be DRAM resident")
        return tensor
    except BaseException:
        _deallocate(tensor)
        raise


def _require_pinned_config(checkpoint: Qwen38Checkpoint, layer_index: int) -> None:
    config = checkpoint.config
    exact = {
        "hidden_size": HIDDEN_SIZE,
        "gdn_qk_heads": QK_HEADS,
        "gdn_value_heads": VALUE_HEADS,
        "gdn_key_head_dim": HEAD_DIM,
        "gdn_value_head_dim": HEAD_DIM,
        "gdn_conv_kernel": CONV_KERNEL_SIZE,
        "gdn_output_gate": "sigmoid",
        "rms_norm_eps": RMS_NORM_EPS,
    }
    for name, expected in exact.items():
        actual = getattr(config, name)
        if actual != expected:
            raise ValueError(f"pinned GDN field {name} must be {expected!r}, got {actual!r}")
    if not 0 <= layer_index < config.num_hidden_layers:
        raise ValueError(f"GDN layer index is outside [0,{config.num_hidden_layers}): {layer_index}")
    if config.layer_types[layer_index] != "linear_attention":
        raise ValueError(f"layer {layer_index} is {config.layer_types[layer_index]!r}, not linear_attention")


def _cache_directory(
    cache_root: str | Path,
    checkpoint: Qwen38Checkpoint,
    mesh_contract: Qwen38MeshContract,
    layer_index: int,
    tt_metal_sha: str,
) -> Path:
    if len(tt_metal_sha) != 40 or any(character not in "0123456789abcdef" for character in tt_metal_sha):
        raise ValueError(f"tt_metal_sha must be a lowercase 40-hex commit, got {tt_metal_sha!r}")
    physical = "-".join(str(device_id) for device_id in mesh_contract.physical_ids)
    path = (
        Path(cache_root).resolve()
        / "gdn"
        / f"index-{INDEX_SHA256}"
        / checkpoint.config.config_sha256
        / f"tt-metal-{tt_metal_sha}"
        / f"mesh-1x4-physical-{physical}"
        / f"layer-{layer_index:02d}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class Qwen38TTNNGDNWeights:
    """Resident TP4 weights for one Qwen3.8 GDN layer."""

    layer_index: int
    qkvzab: Any
    out: Any
    conv_taps: tuple[Any, Any, Any, Any]
    dt_bias: Any
    neg_exp_A: Any
    norm: Any
    projection_dtype: Any

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Qwen38Checkpoint,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        cache_root: str | Path,
        *,
        layer_index: int,
        tt_metal_sha: str,
        projection_dtype=None,
    ) -> "Qwen38TTNNGDNWeights":
        """Pack and upload one exact checkpoint layer.

        The host packing order is ``[device0 local streams | ... | device3]``.
        Sharding that reordered dimension gives each coordinate exactly its four
        Q/K heads and twelve value heads.  No projection tensor is replicated.
        Cache paths bind the checkpoint config, exact tt-metal commit, physical
        mesh order, layer, and dtype so a stale cross-runtime tensorbin cannot
        be selected silently.
        """

        mesh_contract.validate_mesh(mesh_device)
        _require_pinned_config(checkpoint, layer_index)
        if projection_dtype is None:
            # The continuation's correctness path keeps non-expert projection
            # weights in BF16.  BF8_B is an explicit later experiment, never
            # an implicit substitute during ordinary-decode bring-up.
            projection_dtype = ttnn.bfloat16
        if projection_dtype not in (ttnn.bfloat8_b, ttnn.bfloat16):
            raise ValueError("GDN projection weights must use BFLOAT8_B or BFLOAT16")

        source = Qwen38GDNWeights.from_checkpoint(checkpoint, layer_index)
        shards = tuple(source.device_shard(device_index) for device_index in range(TP_SIZE))
        cache_dir = _cache_directory(cache_root, checkpoint, mesh_contract, layer_index, tt_metal_sha)
        dtype_tag = "bf8b" if projection_dtype == ttnn.bfloat8_b else "bf16"

        # Each device-local TTNN linear weight is [K,N].  Concatenating the
        # transposed local blocks along N makes Shard(dim=3) select one whole
        # [q|k|v|z|a|b] block, rather than slicing through the stream boundary.
        qkvzab_host = pack_projection_columns(shards)
        expected_qkvzab = (1, 1, HIDDEN_SIZE, TP_SIZE * PROJECTION_WIDTH_PER_DEVICE)
        if tuple(qkvzab_host.shape) != expected_qkvzab:
            raise RuntimeError(f"packed qkvzab shape {tuple(qkvzab_host.shape)} != {expected_qkvzab}")
        # Projection weights are DRAM width-sharded for the decode matmul
        # program; the renamed tensorbins deliberately orphan interleaved
        # caches and the unpadded 4120-column packing.
        qkvzab = ttnn.as_tensor(
            qkvzab_host,
            dtype=projection_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=dram_sharded_weight_memory_config(mesh_device, HIDDEN_SIZE, PROJECTION_WIDTH_PER_DEVICE),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
            cache_file_name=cache_dir / f"qkvzab_tile_aligned_ab_dram_sharded.{dtype_tag}",
        )

        # Row-parallel output projection: each coordinate owns the checkpoint
        # columns for its twelve value heads, transposed to local [1536,2560].
        out_host = torch.cat([shard.out.transpose(0, 1).contiguous() for shard in shards], dim=0).reshape(
            1, 1, VALUE_WIDTH, HIDDEN_SIZE
        )
        out = ttnn.as_tensor(
            out_host,
            dtype=projection_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=dram_sharded_weight_memory_config(mesh_device, VALUE_WIDTH_PER_DEVICE, HIDDEN_SIZE),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 2)),
            cache_file_name=cache_dir / f"out_dram_sharded.{dtype_tag}",
        )

        def upload_head_vector(value: torch.Tensor, name: str, *, dtype):
            value = value.reshape(1, 1, 1, -1).contiguous()
            return ttnn.as_tensor(
                value,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
                cache_file_name=cache_dir / name,
            )

        # Reorder every convolution tap with the same per-device stream packing
        # as qkvzab.  The activation remains BF16; parameters driving the decay
        # are uploaded as FP32 to match the pinned SSM arithmetic.
        conv_taps = tuple(
            upload_head_vector(
                torch.cat([shard.conv[:, 0, tap] for shard in shards], dim=0).to(torch.bfloat16),
                f"conv-tap-{tap}.bf16",
                dtype=ttnn.bfloat16,
            )
            for tap in range(CONV_KERNEL_SIZE)
        )
        dt_bias = upload_head_vector(
            torch.cat([shard.dt_bias for shard in shards], dim=0).float(),
            "dt-bias.fp32",
            dtype=ttnn.float32,
        )
        neg_exp_A = upload_head_vector(
            -torch.exp(torch.cat([shard.A_log for shard in shards], dim=0).float()),
            "neg-exp-A.fp32",
            dtype=ttnn.float32,
        )
        norm = ttnn.as_tensor(
            source.norm.reshape(1, 1, 1, HEAD_DIM).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
            cache_file_name=cache_dir / "norm.bf16",
        )

        result = cls(layer_index, qkvzab, out, conv_taps, dt_bias, neg_exp_A, norm, projection_dtype)
        result.validate(mesh_contract)
        return result

    def validate(self, mesh_contract: Qwen38MeshContract) -> None:
        mesh_contract.validate_tensor(self.qkvzab, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        mesh_contract.validate_tensor(self.out, placement=TensorPlacement.HEAD_SHARDED, shard_dim=2)
        _require_shape(self.qkvzab, (1, 1, HIDDEN_SIZE, PROJECTION_WIDTH_PER_DEVICE), label="GDN qkvzab")
        _require_shape(self.out, (1, 1, VALUE_WIDTH_PER_DEVICE, HIDDEN_SIZE), label="GDN out")
        for tap_index, tap in enumerate(self.conv_taps):
            mesh_contract.validate_tensor(tap, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
            _require_shape(tap, (1, 1, 1, QKV_WIDTH_PER_DEVICE), label=f"GDN conv tap {tap_index}")
            if tap.dtype != ttnn.bfloat16:
                raise RuntimeError(f"GDN conv tap {tap_index} must be BF16, got {tap.dtype}")
        for name, tensor in (("dt_bias", self.dt_bias), ("neg_exp_A", self.neg_exp_A)):
            mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
            _require_shape(tensor, (1, 1, 1, VALUE_HEADS_PER_DEVICE), label=f"GDN {name}")
            if tensor.dtype != ttnn.float32:
                raise RuntimeError(f"GDN {name} must be FP32, got {tensor.dtype}")
        mesh_contract.validate_tensor(self.norm, placement=TensorPlacement.REPLICATED)
        _require_shape(self.norm, (1, 1, 1, HEAD_DIM), label="GDN gated RMSNorm weight")
        if self.norm.dtype != ttnn.bfloat16:
            raise RuntimeError(f"GDN gated RMSNorm weight must be BF16, got {self.norm.dtype}")

    def deallocate(self) -> None:
        _deallocate(self.qkvzab, self.out, *self.conv_taps, self.dt_bias, self.neg_exp_A, self.norm)


@dataclass
class Qwen38TTNNGDNSnapshot:
    """Preallocated rollback image for one layer's mutable state."""

    layer_index: int
    recurrent: Any
    conv: tuple[Any, Any, Any, Any]
    captured: bool = False
    conv_phase: int = 0

    def validate(self, mesh_contract: Qwen38MeshContract) -> None:
        mesh_contract.validate_tensor(self.recurrent, placement=TensorPlacement.HEAD_SHARDED, shard_dim=1)
        _require_shape(
            self.recurrent,
            (1, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM),
            label="GDN snapshot recurrent state",
        )
        if self.recurrent.dtype != ttnn.float32:
            raise RuntimeError(f"GDN snapshot recurrent state must be FP32, got {self.recurrent.dtype}")
        for index, tensor in enumerate(self.conv):
            mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
            _require_shape(tensor, (1, 1, 1, QKV_WIDTH_PER_DEVICE), label=f"GDN snapshot conv[{index}]")
        if not 0 <= self.conv_phase < CONV_KERNEL_SIZE:
            raise RuntimeError(f"GDN snapshot conv phase must be in [0,{CONV_KERNEL_SIZE}), got {self.conv_phase}")

    def deallocate(self) -> None:
        _deallocate(self.recurrent, *self.conv)


@dataclass
class Qwen38TTNNGDNState:
    """Fixed-address mutable state owned by exactly one GDN layer.

    The four convolution buffers form a ring: the token at step ``n`` is
    written straight into slot ``n % 4`` and the FIR pairs tap ``i`` with slot
    ``(n + 1 + i) % 4``, so no shift copies run.  ``conv_phase`` is the slot
    the next token lands in; snapshots carry it.  A captured graph binds one
    phase, so a trace is valid at positions congruent mod 4 (the decoder's
    graphs are already per position mod 32 through the QSA layers).
    """

    layer_index: int
    recurrent: Any
    conv: tuple[Any, Any, Any, Any]
    zero_recurrent: Any
    zero_conv: Any
    mesh_contract: Qwen38MeshContract
    conv_phase: int = 0

    @classmethod
    def allocate(
        cls,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        *,
        layer_index: int,
        batch_size: int = 1,
    ) -> "Qwen38TTNNGDNState":
        mesh_contract.validate_mesh(mesh_device)
        if batch_size != 1:
            raise ValueError(f"Qwen3.8 interactive GDN admits true global batch one, got {batch_size}")
        if layer_index < 0:
            raise ValueError("layer_index must be nonnegative")

        allocated: list[Any] = []

        def allocate_recurrent(label: str):
            tensor = _allocate_head_sharded_zero(
                mesh_device,
                mesh_contract,
                local_shape=(batch_size, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM),
                dtype=ttnn.float32,
                shard_dim=1,
                label=label,
            )
            allocated.append(tensor)
            return tensor

        def allocate_conv(label: str):
            tensor = _allocate_head_sharded_zero(
                mesh_device,
                mesh_contract,
                local_shape=(1, 1, batch_size, QKV_WIDTH_PER_DEVICE),
                dtype=ttnn.bfloat16,
                shard_dim=3,
                label=label,
            )
            allocated.append(tensor)
            return tensor

        try:
            result = cls(
                layer_index=layer_index,
                recurrent=allocate_recurrent("GDN recurrent state"),
                conv=tuple(allocate_conv(f"GDN conv[{index}]") for index in range(CONV_KERNEL_SIZE)),
                zero_recurrent=allocate_recurrent("GDN zero recurrent source"),
                zero_conv=allocate_conv("GDN zero conv source"),
                mesh_contract=mesh_contract,
            )
            result.validate()
            return result
        except BaseException:
            _deallocate(*allocated)
            raise

    def validate(self) -> None:
        if len(self.conv) != CONV_KERNEL_SIZE:
            raise RuntimeError(f"GDN state requires {CONV_KERNEL_SIZE} conv buffers, got {len(self.conv)}")
        owned = (self.recurrent, *self.conv, self.zero_recurrent, self.zero_conv)
        if len({_tensor_key(tensor) for tensor in owned}) != CONV_KERNEL_SIZE + 3:
            raise RuntimeError("GDN state requires seven distinct backing tensors")
        self.mesh_contract.validate_tensor(self.recurrent, placement=TensorPlacement.HEAD_SHARDED, shard_dim=1)
        self.mesh_contract.validate_tensor(self.zero_recurrent, placement=TensorPlacement.HEAD_SHARDED, shard_dim=1)
        recurrent_shape = (1, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM)
        _require_shape(self.recurrent, recurrent_shape, label="GDN recurrent state")
        _require_shape(self.zero_recurrent, recurrent_shape, label="GDN zero recurrent source")
        if self.recurrent.dtype != ttnn.float32 or self.zero_recurrent.dtype != ttnn.float32:
            raise RuntimeError("GDN recurrent state and zero source must both be FP32")

        for name, tensor in (("zero_conv", self.zero_conv), *[(f"conv[{i}]", t) for i, t in enumerate(self.conv)]):
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
            _require_shape(tensor, (1, 1, 1, QKV_WIDTH_PER_DEVICE), label=f"GDN {name}")
            if tensor.dtype != ttnn.bfloat16:
                raise RuntimeError(f"GDN {name} must be BF16, got {tensor.dtype}")
        if not 0 <= self.conv_phase < CONV_KERNEL_SIZE:
            raise RuntimeError(f"GDN conv phase must be in [0,{CONV_KERNEL_SIZE}), got {self.conv_phase}")

    def conv_window(self) -> tuple[Any, Any, Any, Any]:
        """FIR operands for the next token, oldest first; the last slot receives that token."""

        return tuple(self.conv[(self.conv_phase + 1 + index) % CONV_KERNEL_SIZE] for index in range(CONV_KERNEL_SIZE))

    def advance_conv_window(self) -> None:
        self.conv_phase = (self.conv_phase + 1) % CONV_KERNEL_SIZE

    def reset_inplace(self) -> None:
        """Zero all state without changing any address (trace safe)."""

        self.validate()
        _copy_inplace(self.zero_recurrent, self.recurrent, label="GDN recurrent reset")
        for index, tensor in enumerate(self.conv):
            _copy_inplace(self.zero_conv, tensor, label=f"GDN conv[{index}] reset")
        self.conv_phase = 0

    def allocate_snapshot(self) -> Qwen38TTNNGDNSnapshot:
        """Allocate one persistent capture target for commit/rollback."""

        recurrent = ttnn.empty_like(
            self.recurrent,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        conv = tuple(
            ttnn.empty_like(
                tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for tensor in self.conv
        )
        snapshot = Qwen38TTNNGDNSnapshot(self.layer_index, recurrent, conv)
        snapshot.validate(self.mesh_contract)
        return snapshot

    def capture_into(self, snapshot: Qwen38TTNNGDNSnapshot) -> None:
        if snapshot.layer_index != self.layer_index:
            raise ValueError(
                f"cannot capture layer {self.layer_index} state into layer {snapshot.layer_index} snapshot"
            )
        snapshot.validate(self.mesh_contract)
        _copy_inplace(self.recurrent, snapshot.recurrent, label="GDN recurrent snapshot capture")
        for index, (source, target) in enumerate(zip(self.conv, snapshot.conv)):
            _copy_inplace(source, target, label=f"GDN conv[{index}] snapshot capture")
        snapshot.conv_phase = self.conv_phase
        snapshot.captured = True

    def restore_from(self, snapshot: Qwen38TTNNGDNSnapshot) -> None:
        if snapshot.layer_index != self.layer_index:
            raise ValueError(
                f"cannot restore layer {snapshot.layer_index} snapshot into layer {self.layer_index} state"
            )
        if not snapshot.captured:
            raise RuntimeError("cannot restore an uncaptured GDN snapshot")
        snapshot.validate(self.mesh_contract)
        _copy_inplace(snapshot.recurrent, self.recurrent, label="GDN recurrent snapshot restore")
        for index, (source, target) in enumerate(zip(snapshot.conv, self.conv)):
            _copy_inplace(source, target, label=f"GDN conv[{index}] snapshot restore")
        self.conv_phase = snapshot.conv_phase

    def deallocate(self) -> None:
        _deallocate(self.recurrent, *self.conv, self.zero_recurrent, self.zero_conv)


@dataclass(frozen=True)
class Qwen38TTNNGDNResult:
    hidden_sharded: Any
    state: Qwen38TTNNGDNState


def _upload_replicated_constant(
    mesh_device, mesh_contract: Qwen38MeshContract, host: torch.Tensor, dtype, *, label: str
):
    tensor = ttnn.from_torch(
        host.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
    )
    mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)
    _require_shape(tensor, tuple(host.shape), label=label)
    return tensor


def _require_landed(result, target, *, label: str) -> None:
    """An ``output_tensor=`` op must return the persistent target itself."""

    if result is not None and _tensor_key(result) != _tensor_key(target):
        raise RuntimeError(f"{label} did not land in its persistent buffer")


def chunk_constant_tiles() -> dict[str, torch.Tensor]:
    """Host images of the chunk kernel's constant tiles: eye / tril / ones ``[1,1,32,32]`` and the three
    32x32 quadrant masks packed into ``[1,1,32,96]`` (top-left, bottom-right, bottom-left)."""

    rows = torch.arange(CHUNK_SIZE).unsqueeze(1) < CHUNK_SIZE // 2
    columns = torch.arange(CHUNK_SIZE).unsqueeze(0) < CHUNK_SIZE // 2
    masks = torch.cat([(rows & columns).float(), (~rows & ~columns).float(), (~rows & columns).float()], dim=1)
    return {
        "eye": torch.eye(CHUNK_SIZE).reshape(1, 1, CHUNK_SIZE, CHUNK_SIZE),
        "tril": torch.tril(torch.ones(CHUNK_SIZE, CHUNK_SIZE)).reshape(1, 1, CHUNK_SIZE, CHUNK_SIZE),
        "ones": torch.ones(1, 1, CHUNK_SIZE, CHUNK_SIZE),
        "masks": masks.reshape(1, 1, CHUNK_SIZE, CHUNK_QUADRANT_MASK_WIDTH),
    }


@dataclass(frozen=True)
class Qwen38TTNNGDNRowsConstants:
    """Read-only device tensors shared by every GDN layer's ``rows``-row path (one set per row count).

    ``row_mask_*`` is 1.0 for rows below ``rows`` and 0.0 for the padding rows up to the chunk; ``arange``
    (a column) and ``arange_row`` (a row) hold the row index (fp32, exact); ``one`` is the fp32 scalar 1.0;
    ``conv_taps``, ``history_select_stack`` and ``qk_expand`` are the 0/1 selection matrices of
    :func:`rows_window_select_tiles` (bf16: exactly representable); ``eye``/``tril``/``ones``/``masks`` are
    the chunk kernel's constant tiles.  Everything is uploaded here so nothing is uploaded inside a trace.
    ``select_compute_config`` (HiFi4, fp32 accumulation, no approximations) is the matmul configuration
    under which a 0/1 selection is exact: every output element is one bf16 value times 1.0 plus exact zeros.
    """

    rows: int
    row_mask_bf16: Any  # [1, CHUNK_SIZE, 1, 1] BF16, multiplies the [1, T, heads, 128] q/k
    row_mask_bf16_col: Any  # [1, 1, CHUNK_SIZE, 1] BF16, multiplies the token-major flat [1, 1, T, 1536] v
    row_mask_fp32: Any  # [1, 1, CHUNK_SIZE, 1] FP32, multiplies the [1, 1, T, heads] beta / log decay
    arange: Any  # [1, 1, CHUNK_SIZE, 1] FP32 row index
    arange_row: Any  # [1, 1, 1, CHUNK_SIZE] FP32 row index
    one: Any  # [1, 1, 1, 1] FP32
    conv_taps: tuple[Any, Any, Any]  # 3 x [1, 1, CHUNK_SIZE, CONV_WINDOW_TILE_ROWS] BF16
    history_select_stack: Any  # [1, 1, CHUNK_SIZE, CHUNK_SIZE * CONV_WINDOW_TILE_ROWS] BF16
    qk_expand: Any  # [1, 1, QK_WIDTH_PER_DEVICE, VALUE_WIDTH_PER_DEVICE] BF16
    eye: Any
    tril: Any
    ones: Any
    masks: Any
    select_compute_config: Any
    mesh_contract: Qwen38MeshContract

    @classmethod
    def allocate(cls, mesh_device, mesh_contract: Qwen38MeshContract, *, rows: int) -> "Qwen38TTNNGDNRowsConstants":
        mesh_contract.validate_mesh(mesh_device)
        if not 1 <= rows <= CHUNK_SIZE:
            raise ValueError(f"GDN rows path admits 1..{CHUNK_SIZE} rows, got {rows}")
        keep = (torch.arange(CHUNK_SIZE) < rows).float()
        tiles = chunk_constant_tiles()
        selects = rows_window_select_tiles(rows)
        uploaded: list[Any] = []

        def upload(host: torch.Tensor, dtype, label: str):
            tensor = _upload_replicated_constant(mesh_device, mesh_contract, host, dtype, label=label)
            uploaded.append(tensor)
            return tensor

        try:
            return cls(
                rows=rows,
                row_mask_bf16=upload(keep.reshape(1, CHUNK_SIZE, 1, 1).to(torch.bfloat16), ttnn.bfloat16, "row mask"),
                row_mask_bf16_col=upload(
                    keep.reshape(1, 1, CHUNK_SIZE, 1).to(torch.bfloat16), ttnn.bfloat16, "row mask column"
                ),
                row_mask_fp32=upload(keep.reshape(1, 1, CHUNK_SIZE, 1), ttnn.float32, "row mask fp32"),
                arange=upload(torch.arange(CHUNK_SIZE).float().reshape(1, 1, CHUNK_SIZE, 1), ttnn.float32, "arange"),
                arange_row=upload(
                    torch.arange(CHUNK_SIZE).float().reshape(1, 1, 1, CHUNK_SIZE), ttnn.float32, "arange row"
                ),
                one=upload(torch.ones(1, 1, 1, 1), ttnn.float32, "one"),
                conv_taps=tuple(
                    upload(
                        selects["conv_taps"][tap].reshape(1, 1, CHUNK_SIZE, CONV_WINDOW_TILE_ROWS),
                        ttnn.bfloat16,
                        f"conv tap {tap} select",
                    )
                    for tap in range(CONV_HISTORY_ROWS)
                ),
                history_select_stack=upload(
                    selects["history_select_stack"].reshape(1, 1, CHUNK_SIZE, CHUNK_SIZE * CONV_WINDOW_TILE_ROWS),
                    ttnn.bfloat16,
                    "history select stack",
                ),
                qk_expand=upload(
                    selects["qk_expand"].reshape(1, 1, QK_WIDTH_PER_DEVICE, VALUE_WIDTH_PER_DEVICE),
                    ttnn.bfloat16,
                    "q/k GQA expand",
                ),
                eye=upload(tiles["eye"], ttnn.float32, "chunk eye"),
                tril=upload(tiles["tril"], ttnn.float32, "chunk tril"),
                ones=upload(tiles["ones"], ttnn.float32, "chunk ones"),
                masks=upload(tiles["masks"], ttnn.float32, "chunk quadrant masks"),
                select_compute_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                ),
                mesh_contract=mesh_contract,
            )
        except BaseException:
            _deallocate(*uploaded)
            raise

    def deallocate(self) -> None:
        _deallocate(
            self.row_mask_bf16,
            self.row_mask_bf16_col,
            self.row_mask_fp32,
            self.arange,
            self.arange_row,
            self.one,
            *self.conv_taps,
            self.history_select_stack,
            self.qk_expand,
            self.eye,
            self.tril,
            self.ones,
            self.masks,
        )


@dataclass(frozen=True)
class Qwen38TTNNRowsSelectors:
    """Per-pass device selects derived once from the accept count and shared by every layer's commit.

    ``accepted`` is the number of accepted drafts (fp32 scalar on device, 0 <= accepted < rows); row 0 is
    the base token and is always committed, so ``committed_mask`` is ``arange <= accepted`` (accepted + 1
    rows).  ``history_select`` is the ``[32, 64]`` 0/1 matrix that reads the next GDN FIR history (window
    rows accepted + 1 .. accepted + 3) out of the ``[history | qkv]`` window with one exact matmul per
    layer; ``onehot_bf16[j]`` (``accepted == j``) drives the PLE commit's exact multiply/add select.
    """

    rows: int
    committed_mask: Any  # [1, 1, CHUNK_SIZE, 1] FP32
    onehot_bf16: tuple[Any, ...]  # rows x [1, 1, 1, 1] BF16
    history_select: Any  # [1, 1, CHUNK_SIZE, CONV_WINDOW_TILE_ROWS] BF16

    def validate(self, rows: int) -> None:
        if self.rows != rows or len(self.onehot_bf16) != rows:
            raise ValueError(f"rows selectors were built for {self.rows} rows, the path runs {rows}")
        _require_shape(self.committed_mask, (1, 1, CHUNK_SIZE, 1), label="committed rows mask")
        _require_shape(self.history_select, (1, 1, CHUNK_SIZE, CONV_WINDOW_TILE_ROWS), label="rows history select")
        if self.committed_mask.dtype != ttnn.float32 or self.history_select.dtype != ttnn.bfloat16:
            raise RuntimeError(
                f"rows selector dtypes are {self.committed_mask.dtype} (mask) / {self.history_select.dtype} (select)"
            )
        for index, bf16 in enumerate(self.onehot_bf16):
            _require_shape(bf16, (1, 1, 1, 1), label=f"rows selector onehot_bf16[{index}]")
            if bf16.dtype != ttnn.bfloat16:
                raise RuntimeError(f"rows selector onehot_bf16[{index}] dtype is {bf16.dtype}")

    def deallocate(self) -> None:
        _deallocate(self.committed_mask, *self.onehot_bf16, self.history_select)


def build_rows_selectors(accepted, constants: Qwen38TTNNGDNRowsConstants) -> Qwen38TTNNRowsSelectors:
    """Derive the commit selects from the device accept count (fp32 ``[1,1,1,1]``, replicated).

    rows + 4 tiny ops per pass, not per layer: the mask, the one-hot row (``arange_row == accepted``,
    written as bf16 by the comparison), the history select as ``onehot_row @ history_select_stack``
    (exact: one row of the stack) reshaped to ``[32, 64]``, and the rows one-hot scalars for the PLE.
    Comparisons on fp32 integers are exact.
    """

    _require_shape(accepted, (1, 1, 1, 1), label="accept count")
    if accepted.dtype != ttnn.float32:
        raise RuntimeError(f"accept count must be FP32, got {accepted.dtype}")
    dram = ttnn.DRAM_MEMORY_CONFIG
    committed_mask = ttnn.le(constants.arange, accepted, memory_config=dram)
    onehot_row = ttnn.eq(constants.arange_row, accepted, dtype=ttnn.bfloat16, memory_config=dram)
    _require_shape(onehot_row, (1, 1, 1, CHUNK_SIZE), label="rows one-hot row")
    select_flat = ttnn.matmul(
        onehot_row,
        constants.history_select_stack,
        memory_config=dram,
        compute_kernel_config=constants.select_compute_config,
    )
    _deallocate(onehot_row)
    _require_shape(select_flat, (1, 1, 1, CHUNK_SIZE * CONV_WINDOW_TILE_ROWS), label="rows history select row")
    # The last dim changes, so this is a real relayout into a new buffer, not a view of select_flat.
    history_select = ttnn.reshape(select_flat, (1, 1, CHUNK_SIZE, CONV_WINDOW_TILE_ROWS))
    _deallocate(select_flat)
    onehot_bf16 = tuple(
        ttnn.eq(accepted, float(index), dtype=ttnn.bfloat16, memory_config=dram) for index in range(constants.rows)
    )
    selectors = Qwen38TTNNRowsSelectors(constants.rows, committed_mask, onehot_bf16, history_select)
    selectors.validate(constants.rows)
    return selectors


@dataclass
class Qwen38TTNNGDNRowsState:
    """Per-layer persistent buffers of the ``rows``-row path (fixed addresses across passes).

    ``history`` is the FIR history (rows 0..2: the CONV_HISTORY_ROWS q/k/v rows before the first new
    row, in a full 32-row tile whose rows 3..31 are never read; it replaces the phase-indexed ring of
    the 1-row path, which a variable-advance trace cannot bind).  ``qkv`` holds this pass's projected
    q|k|v rows (CHUNK_SIZE rows, rows past ``rows`` zero) and ``q``/``k``/``v``/``beta``/``g`` the chunk
    kernel's inputs (q/k already expanded to the 12 value heads; v token-major flat, the composite's
    ``flat v`` form that the prep reader addresses by tile without a head-split relayout), kept so the
    commit can rerun the kernel over the committed prefix from the committed recurrent state.
    """

    layer_index: int
    constants: Qwen38TTNNGDNRowsConstants
    history: Any  # [1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE] BF16, rows 0..CONV_HISTORY_ROWS-1 valid
    qkv: Any  # [1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE] BF16
    q: Any  # [1, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE, HEAD_DIM] BF16, l2-normalized, GQA-expanded
    k: Any
    v: Any  # [1, 1, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE] BF16, token-major flat (head h at columns 128h..)
    beta: Any  # [1, 1, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE] FP32
    g: Any  # same, the log decay
    output: Any  # [1, 1, rows, HIDDEN_SIZE_PER_DEVICE] BF16: the pass's output rows (valid until the next pass)
    mesh_contract: Qwen38MeshContract

    @classmethod
    def allocate(
        cls, mesh_device, mesh_contract: Qwen38MeshContract, constants: Qwen38TTNNGDNRowsConstants, *, layer_index: int
    ) -> "Qwen38TTNNGDNRowsState":
        mesh_contract.validate_mesh(mesh_device)
        if constants.mesh_contract != mesh_contract:
            raise ValueError("GDN rows constants belong to a different physical mesh contract")
        allocated: list[Any] = []

        def zero(local_shape: tuple[int, ...], dtype, shard_dim: int, label: str):
            tensor = _allocate_head_sharded_zero(
                mesh_device, mesh_contract, local_shape=local_shape, dtype=dtype, shard_dim=shard_dim, label=label
            )
            allocated.append(tensor)
            return tensor

        qk_shape = (1, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE, HEAD_DIM)
        try:
            result = cls(
                layer_index=layer_index,
                constants=constants,
                history=zero((1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), ttnn.bfloat16, 3, "GDN rows history"),
                qkv=zero((1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), ttnn.bfloat16, 3, "GDN rows qkv"),
                q=zero(qk_shape, ttnn.bfloat16, 2, "GDN rows q"),
                k=zero(qk_shape, ttnn.bfloat16, 2, "GDN rows k"),
                v=zero((1, 1, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE), ttnn.bfloat16, 3, "GDN rows v"),
                beta=zero((1, 1, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE), ttnn.float32, 3, "GDN rows beta"),
                g=zero((1, 1, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE), ttnn.float32, 3, "GDN rows log decay"),
                output=zero((1, 1, constants.rows, HIDDEN_SIZE_PER_DEVICE), ttnn.bfloat16, 3, "GDN rows output"),
                mesh_contract=mesh_contract,
            )
            result.validate()
            return result
        except BaseException:
            _deallocate(*allocated)
            raise

    def validate(self) -> None:
        owned = (self.history, self.qkv, self.q, self.k, self.v, self.beta, self.g, self.output)
        if len({_tensor_key(tensor) for tensor in owned}) != len(owned):
            raise RuntimeError("GDN rows state requires eight distinct backing tensors")
        expected = {
            "history": ((1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), ttnn.bfloat16, 3),
            "qkv": ((1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), ttnn.bfloat16, 3),
            "q": ((1, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE, HEAD_DIM), ttnn.bfloat16, 2),
            "k": ((1, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE, HEAD_DIM), ttnn.bfloat16, 2),
            "v": ((1, 1, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE), ttnn.bfloat16, 3),
            "beta": ((1, 1, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE), ttnn.float32, 3),
            "g": ((1, 1, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE), ttnn.float32, 3),
            "output": ((1, 1, self.constants.rows, HIDDEN_SIZE_PER_DEVICE), ttnn.bfloat16, 3),
        }
        for name, (shape, dtype, shard_dim) in expected.items():
            tensor = getattr(self, name)
            _require_shape(tensor, shape, label=f"GDN rows {name}")
            if tensor.dtype != dtype:
                raise RuntimeError(f"GDN rows {name} must be {dtype}, got {tensor.dtype}")
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=shard_dim)

    def deallocate(self) -> None:
        _deallocate(self.history, self.qkv, self.q, self.k, self.v, self.beta, self.g, self.output)


@dataclass(frozen=True)
class Qwen38TTNNGDNRowsResult:
    """``hidden_rows`` is ``rows_state.output`` (``[1,1,rows,640]``, persistent: read it before the next pass,
    never deallocate it); ``final_state`` is the chunk kernel's state after all ``rows`` rows in a new FP32
    buffer (the committed state is untouched until ``commit_rows``); the caller owns and deallocates it."""

    hidden_rows: Any
    final_state: Any
    state: Qwen38TTNNGDNState
    rows_state: Qwen38TTNNGDNRowsState


class Qwen38TTNNGDN:
    """One exact TP4 Qwen3.8 Gated DeltaNet layer."""

    def __init__(
        self,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        weights: Qwen38TTNNGDNWeights,
        *,
        collective_topology=None,
    ) -> None:
        mesh_contract.validate_mesh(mesh_device)
        weights.validate(mesh_contract)
        self.mesh_device = mesh_device
        self.mesh_contract = mesh_contract
        self.weights = weights
        self.collective_topology = collective_topology or ttnn.Topology.Linear
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.in_proj_act_memory_config, self.in_proj_program_config = dram_sharded_matmul_configs(
            mesh_device, HIDDEN_SIZE, PROJECTION_WIDTH_PER_DEVICE, num_cores=8
        )
        self.out_proj_act_memory_config, self.out_proj_program_config = dram_sharded_matmul_configs(
            mesh_device, VALUE_WIDTH_PER_DEVICE, HIDDEN_SIZE, num_cores=16
        )
        # Recurrent step kernels: the qualified configuration of the shared FLA
        # decode step for K = V = 128 (HiFi2, FP32 accumulation).  Changing any
        # field moves a rounding point.
        head_tiles = HEAD_DIM // ttnn.TILE_SIZE
        self.recurrent_matmul_program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            in0_block_w=head_tiles,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=head_tiles,
        )
        self.recurrent_read_compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.recurrent_write_compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def allocate_state(self) -> Qwen38TTNNGDNState:
        return Qwen38TTNNGDNState.allocate(
            self.mesh_device,
            self.mesh_contract,
            layer_index=self.weights.layer_index,
            batch_size=1,
        )

    def _validate_state(self, state: Qwen38TTNNGDNState) -> None:
        if state.layer_index != self.weights.layer_index:
            raise ValueError(f"GDN layer {self.weights.layer_index} received state owned by layer {state.layer_index}")
        if state.mesh_contract != self.mesh_contract:
            raise ValueError("GDN state belongs to a different physical mesh contract")
        state.validate()

    def _all_gather_hidden(self, hidden_sharded):
        _require_shape(hidden_sharded, (1, 1, 1, HIDDEN_SIZE_PER_DEVICE), label="GDN decode input")
        self.mesh_contract.validate_tensor(hidden_sharded, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        # The gather writes the in-projection's eight-core activation layout;
        # the tensor stays allocated through the layer (20 KB per core).
        full_hidden = ttnn.all_gather(
            hidden_sharded,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=self.in_proj_act_memory_config,
        )
        self.mesh_contract.validate_tensor(full_hidden, placement=TensorPlacement.REPLICATED)
        _require_shape(full_hidden, (1, 1, 1, HIDDEN_SIZE), label="GDN gathered hidden")
        return full_hidden

    def _project(self, full_hidden, newest):
        """Project the gathered hidden row; the q/k/v columns land in ``newest``, the ring slot for this token."""

        projected_ws = ttnn.linear(
            full_hidden,
            self.weights.qkvzab,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.in_proj_program_config,
            compute_kernel_config=self.compute_config,
        )
        projected = ttnn.to_memory_config(projected_ws, ttnn.L1_MEMORY_CONFIG)
        _deallocate(projected_ws)
        self.mesh_contract.validate_tensor(projected, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(projected, (1, 1, 1, PROJECTION_WIDTH_PER_DEVICE), label="GDN fused projection")

        # Every slice starts on a tile boundary, so each is one device op; the
        # newest-token slice writes the persistent slot (no staging copy).
        ttnn.slice(projected, (0, 0, 0, 0), (1, 1, 1, QKV_WIDTH_PER_DEVICE), output_tensor=newest)
        z = ttnn.slice(
            projected,
            (0, 0, 0, QKV_WIDTH_PER_DEVICE),
            (1, 1, 1, A_COLUMN),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        a = ttnn.slice(
            projected,
            (0, 0, 0, A_COLUMN),
            (1, 1, 1, A_COLUMN + VALUE_HEADS_PER_DEVICE),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        b = ttnn.slice(
            projected,
            (0, 0, 0, B_COLUMN),
            (1, 1, 1, B_COLUMN + VALUE_HEADS_PER_DEVICE),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _deallocate(projected)
        for name, tensor, expected in (
            ("z", z, (1, 1, 1, VALUE_WIDTH_PER_DEVICE)),
            ("a", a, (1, 1, 1, VALUE_HEADS_PER_DEVICE)),
            ("b", b, (1, 1, 1, VALUE_HEADS_PER_DEVICE)),
        ):
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
            _require_shape(tensor, expected, label=f"GDN projected {name}")
        return z, a, b

    def _causal_conv_decode(self, window):
        # ``window`` is oldest -> newest; tap i pairs with window[i].
        conv = ttnn.multiply(window[0], self.weights.conv_taps[0], memory_config=ttnn.L1_MEMORY_CONFIG)
        for index in range(1, CONV_KERNEL_SIZE):
            previous = conv
            conv = ttnn.mac(window[index], self.weights.conv_taps[index], previous)
            if _tensor_key(conv) != _tensor_key(previous):
                _deallocate(previous)
        conv = ttnn.silu(conv, memory_config=ttnn.L1_MEMORY_CONFIG)
        self.mesh_contract.validate_tensor(conv, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(conv, (1, 1, 1, QKV_WIDTH_PER_DEVICE), label="GDN causal convolution")
        return conv

    def _make_recurrent_inputs(self, conv, a, b):
        q_slice = ttnn.slice(
            conv,
            (0, 0, 0, 0),
            (1, 1, 1, QK_WIDTH_PER_DEVICE),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        k_slice = ttnn.slice(
            conv,
            (0, 0, 0, QK_WIDTH_PER_DEVICE),
            (1, 1, 1, 2 * QK_WIDTH_PER_DEVICE),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        v_slice = ttnn.slice(
            conv,
            (0, 0, 0, 2 * QK_WIDTH_PER_DEVICE),
            (1, 1, 1, QKV_WIDTH_PER_DEVICE),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _deallocate(conv)

        # Heads go on dim 1 so every recurrent operand is already in matmul
        # form [1,H,1,128]: the head split is one repack per operand and the
        # GQA repeat over dim 1 is one page-copy op.  k's tile padding is
        # zeroed because k^T's padding lanes enter the outer-product
        # contraction (0 * finite = 0); q's and v's padding never does.
        q_heads = ttnn.reshape(q_slice, (1, QK_HEADS_PER_DEVICE, 1, HEAD_DIM))
        k_heads = ttnn.reshape(k_slice, (1, QK_HEADS_PER_DEVICE, 1, HEAD_DIM), pad_value=0.0)
        q = ttnn.repeat_interleave(q_heads, QK_REPEAT_FACTOR, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.repeat_interleave(k_heads, QK_REPEAT_FACTOR, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        v = ttnn.reshape(v_slice, (1, VALUE_HEADS_PER_DEVICE, 1, HEAD_DIM))
        for tensor, reference in ((q, q_slice), (k, k_slice), (v, v_slice)):
            _retag_head_shard_after_reshape(tensor, reference=reference, shard_dim=1)
        _deallocate(q_slice, k_slice, v_slice, q_heads, k_heads)
        for name, tensor in (("query", q), ("key", k), ("value", v)):
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=1)
            _require_shape(
                tensor,
                (1, VALUE_HEADS_PER_DEVICE, 1, HEAD_DIM),
                label=f"GDN recurrent {name}",
            )

        # The authenticated token17 layer-0 boundary requires the explicit
        # FP32 sigmoid path; beta stays FP32 straight into the FP32 recurrent
        # step. Retain every producer until the recurrent consumer is
        # enqueued: reshape/topology metadata does not transfer the underlying
        # buffer lifetime.  [1,H,1,1] is the broadcast form the step consumes.
        b_fp32 = ttnn.typecast(b, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        beta_fp32 = ttnn.sigmoid(b_fp32, memory_config=ttnn.L1_MEMORY_CONFIG)
        beta = ttnn.reshape(beta_fp32, (1, VALUE_HEADS_PER_DEVICE, 1, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        _retag_head_shard_after_reshape(beta, reference=beta_fp32, shard_dim=1)
        beta_producers = (b, b_fp32, beta_fp32)

        a_fp32 = ttnn.typecast(a, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        _deallocate(a)
        softplus = ttnn.add(
            a_fp32,
            self.weights.dt_bias,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _deallocate(a_fp32)
        log_decay_raw = ttnn.multiply(
            self.weights.neg_exp_A,
            softplus,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _deallocate(softplus)
        log_decay = ttnn.reshape(log_decay_raw, (1, VALUE_HEADS_PER_DEVICE, 1, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        _retag_head_shard_after_reshape(log_decay, reference=log_decay_raw, shard_dim=1)
        beta_producers = (*beta_producers, log_decay_raw)
        for name, tensor in (("beta", beta), ("log_decay", log_decay)):
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HEAD_SHARDED, shard_dim=1)
            _require_shape(tensor, (1, VALUE_HEADS_PER_DEVICE, 1, 1), label=f"GDN recurrent {name}")
        if log_decay.dtype != ttnn.float32:
            raise RuntimeError(f"GDN log decay must be FP32, got {log_decay.dtype}")
        return q, k, v, beta, log_decay, beta_producers

    def _recurrent_decode(self, q, k, v, beta, log_decay, beta_producers, state: Qwen38TTNNGDNState):
        """One FP32 gated delta-rule step on [1,H,1,128] operands, written into ``state.recurrent``.

        q and k are l2-normalized in BF16 before the FP32 promotion (the pinned
        Qwen4Exp order); beta and the log decay arrive FP32 and are used as is.
        v is promoted by the subtract: BF16 -> FP32 widening is exact wherever
        it happens.
        """

        l1 = ttnn.L1_MEMORY_CONFIG
        q_normed = ttnn.rms_norm(q, epsilon=QK_L2_NORM_EPS / HEAD_DIM)
        q_unit = ttnn.multiply(q_normed, HEAD_DIM**-0.5, memory_config=l1)
        k_normed = ttnn.rms_norm(k, epsilon=QK_L2_NORM_EPS / HEAD_DIM)
        k_unit = ttnn.multiply(k_normed, HEAD_DIM**-0.5, memory_config=l1)
        _deallocate(q, k, q_normed, k_normed)
        q_fp32 = ttnn.typecast(q_unit, ttnn.float32, memory_config=l1)
        k_row = ttnn.typecast(k_unit, ttnn.float32, memory_config=l1)
        _deallocate(q_unit, k_unit)
        q_row = ttnn.multiply(q_fp32, HEAD_DIM**-0.5, memory_config=l1)
        _deallocate(q_fp32)

        # Decay the state straight from its DRAM home; exp(log_decay) is fused
        # into the multiply.
        decayed = ttnn.multiply(
            state.recurrent,
            log_decay,
            input_tensor_b_activations=[ttnn.UnaryOpType.EXP],
            memory_config=l1,
        )
        v_read = ttnn.matmul(
            k_row,
            decayed,
            memory_config=l1,
            program_config=self.recurrent_matmul_program_config,
            compute_kernel_config=self.recurrent_read_compute_config,
        )
        delta = ttnn.subtract(v, v_read, dtype=ttnn.float32, memory_config=l1)
        _deallocate(v, v_read)
        k_col = ttnn.transpose(k_row, 2, 3, memory_config=l1)
        _deallocate(k_row)
        outer = ttnn.matmul(
            k_col,
            delta,
            memory_config=l1,
            compute_kernel_config=self.recurrent_write_compute_config,
        )
        _deallocate(k_col, delta)
        update = ttnn.multiply(outer, beta, memory_config=l1)
        _deallocate(outer)
        # The state add lands in the persistent buffer: fixed address, no copy.
        new_recurrent = ttnn.add(decayed, update, output_tensor=state.recurrent)
        _deallocate(decayed, update, beta, log_decay, *beta_producers)
        if new_recurrent is not None and _tensor_key(new_recurrent) != _tensor_key(state.recurrent):
            raise RuntimeError("GDN recurrent update did not land in the persistent state")
        _require_shape(
            state.recurrent,
            (1, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM),
            label="GDN next recurrent state",
        )
        if state.recurrent.dtype != ttnn.float32:
            raise RuntimeError(f"GDN next recurrent state must be FP32, got {state.recurrent.dtype}")

        output = ttnn.matmul(
            q_row,
            state.recurrent,
            memory_config=l1,
            program_config=self.recurrent_matmul_program_config,
            compute_kernel_config=self.recurrent_read_compute_config,
        )
        _deallocate(q_row)
        _require_shape(output, (1, VALUE_HEADS_PER_DEVICE, 1, HEAD_DIM), label="GDN recurrent output")
        return output

    def _gate_and_project(self, recurrent_output, z, full_hidden):
        # The serial reference returns a BF16 per-token recurrent output while
        # retaining the carried state in FP32.  Match that boundary explicitly.
        # The per-head RMSNorm runs on the [1,H,1,128] step output directly.
        output_bf16 = (
            recurrent_output
            if recurrent_output.dtype == ttnn.bfloat16
            else ttnn.typecast(recurrent_output, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        )
        if output_bf16 is not recurrent_output:
            _deallocate(recurrent_output)
        normalized_heads = ttnn.rms_norm(
            output_bf16,
            weight=self.weights.norm,
            epsilon=RMS_NORM_EPS,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _deallocate(output_bf16)
        normalized = ttnn.reshape(normalized_heads, (1, 1, 1, VALUE_WIDTH_PER_DEVICE))
        _retag_head_shard_after_reshape(normalized, reference=normalized_heads, shard_dim=3)
        _deallocate(normalized_heads)
        self.mesh_contract.validate_tensor(normalized, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(normalized, (1, 1, 1, VALUE_WIDTH_PER_DEVICE), label="GDN normalized output")

        # Exact Qwen4Exp difference from the Qwen3.6 jumping-off point.
        z_fp32 = ttnn.typecast(z, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        _deallocate(z)
        sigmoid_fp32 = ttnn.sigmoid(z_fp32, memory_config=ttnn.L1_MEMORY_CONFIG)
        _deallocate(z_fp32)
        sigmoid_bf16 = ttnn.typecast(sigmoid_fp32, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        _deallocate(sigmoid_fp32)
        # The gate product is written straight into the out-projection's
        # sixteen-core activation layout.
        gated = ttnn.multiply(normalized, sigmoid_bf16, memory_config=self.out_proj_act_memory_config)
        _deallocate(normalized, sigmoid_bf16)
        self.mesh_contract.validate_tensor(gated, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(gated, (1, 1, 1, VALUE_WIDTH_PER_DEVICE), label="GDN sigmoid-gated output")

        partial_ws = ttnn.linear(
            gated,
            self.weights.out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.out_proj_program_config,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(gated)
        # The line reduce-scatter reads the sixteen-core partial in place.
        self.mesh_contract.mark_local_partial(
            partial_ws,
            replicated_reference=full_hidden,
            expected_shape=(1, 1, 1, HIDDEN_SIZE),
        )
        output = ttnn.reduce_scatter(
            partial_ws,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.collective_topology,
        )
        _deallocate(partial_ws)
        self.mesh_contract.mark_collective_shard(
            output,
            replicated_reference=full_hidden,
            shard_dim=3,
            expected_local_shape=(1, 1, 1, HIDDEN_SIZE_PER_DEVICE),
        )
        _deallocate(full_hidden)
        return output

    def forward_decode(self, hidden_sharded, state: Qwen38TTNNGDNState) -> Qwen38TTNNGDNResult:
        """Advance one true-global-B1 token and mutate ``state`` in place."""

        self._validate_state(state)
        full_hidden = self._all_gather_hidden(hidden_sharded)
        window = state.conv_window()
        z, a, b = self._project(full_hidden, window[-1])
        state.advance_conv_window()
        conv = self._causal_conv_decode(window)
        q, k, v, beta, log_decay, beta_producers = self._make_recurrent_inputs(conv, a, b)
        recurrent_output = self._recurrent_decode(q, k, v, beta, log_decay, beta_producers, state)
        output = self._gate_and_project(recurrent_output, z, full_hidden)
        return Qwen38TTNNGDNResult(output, state)

    def forward_prefill(self, hidden_sharded, state: Qwen38TTNNGDNState) -> Qwen38TTNNGDNResult:
        """Exact device-only serial prefill over the ordinary decode transition.

        ``hidden_sharded`` has local shape ``[1,1,T,640]`` and must carry the
        same hidden-sharded mesh topology as decode.  This path intentionally
        establishes the state-transition golden before the chunk kernel is
        admitted; it performs no host inference and does not replace GDN with a
        dense or CPU fallback.
        """

        self._validate_state(state)
        shape = _shape(hidden_sharded)
        if len(shape) != 4 or shape[:2] != (1, 1) or shape[3] != HIDDEN_SIZE_PER_DEVICE or shape[2] <= 0:
            raise ValueError(f"GDN prefill input must have local shape [1,1,T,640], got {shape}")
        self.mesh_contract.validate_tensor(hidden_sharded, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

        outputs = []
        for position in range(shape[2]):
            token = ttnn.slice(
                hidden_sharded,
                (0, 0, position, 0),
                (1, 1, position + 1, HIDDEN_SIZE_PER_DEVICE),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            result = self.forward_decode(token, state)
            _deallocate(token)
            outputs.append(result.hidden_sharded)
        if len(outputs) == 1:
            output = outputs[0]
        else:
            output = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _deallocate(*outputs)
        self.mesh_contract.validate_tensor(output, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        _require_shape(output, (1, 1, shape[2], HIDDEN_SIZE_PER_DEVICE), label="GDN prefill output")
        return Qwen38TTNNGDNResult(output, state)

    # ------------------------------------------------------------------ rows path (MTP v2 verify)
    #
    # ``forward_rows`` runs R = rows consecutive positions in one pass: the same projection, FIR,
    # normalization, gate and output ops as ``forward_decode`` on CHUNK_SIZE-row tiles (the input
    # is zero padded to one tile, so every matmul keeps per_core_M = 1), and the recurrence through
    # ``ttnn.transformer.chunk_gated_delta_rule`` over exactly one full chunk with the rows past R
    # zeroed on device (k = v = 0, beta = 0, g = 0: an identity update).  It reads the committed
    # recurrent state and never writes it; ``commit_rows`` reruns the kernel over the committed
    # prefix (beta and g masked by the per-pass selectors) and lands the state in place, then moves
    # the FIR history forward by accepted + 1 rows with an exact 0/1 selection matmul.  Row-count and row
    # index never appear as per-pass host arguments: one program set per R, fixed buffers.

    def allocate_rows_constants(self, rows: int) -> Qwen38TTNNGDNRowsConstants:
        return Qwen38TTNNGDNRowsConstants.allocate(self.mesh_device, self.mesh_contract, rows=rows)

    def allocate_rows_state(self, constants: Qwen38TTNNGDNRowsConstants) -> Qwen38TTNNGDNRowsState:
        return Qwen38TTNNGDNRowsState.allocate(
            self.mesh_device, self.mesh_contract, constants, layer_index=self.weights.layer_index
        )

    def _validate_rows_state(self, rows_state: Qwen38TTNNGDNRowsState) -> int:
        if rows_state.layer_index != self.weights.layer_index:
            raise ValueError(
                f"GDN layer {self.weights.layer_index} received rows state owned by layer {rows_state.layer_index}"
            )
        if rows_state.mesh_contract != self.mesh_contract:
            raise ValueError("GDN rows state belongs to a different physical mesh contract")
        rows_state.validate()
        return rows_state.constants.rows

    def sync_rows_history_from_state(self, state: Qwen38TTNNGDNState, rows_state: Qwen38TTNNGDNRowsState) -> None:
        """Eager mode switch (1-row -> rows): copy the ring's three history slots into ``history``."""

        self._validate_state(state)
        self._validate_rows_state(rows_state)
        window = state.conv_window()
        combined = ttnn.concat(list(window[:CONV_HISTORY_ROWS]), dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Zero rows 3..31 of the tile: ``ttnn.pad`` inside the input's own tile returns a view of
        # ``combined`` on this runtime, so only ``combined`` is deallocated.
        padded = ttnn.pad(
            combined,
            [(0, 0), (0, 0), (0, CHUNK_SIZE - CONV_HISTORY_ROWS), (0, 0)],
            0.0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _require_shape(padded, (1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), label="GDN rows history tile")
        _copy_inplace(padded, rows_state.history, label="GDN rows history load")
        _deallocate(combined)

    def sync_state_from_rows_history(self, rows_state: Qwen38TTNNGDNRowsState, state: Qwen38TTNNGDNState) -> None:
        """Eager mode switch (rows -> 1-row): the ring's three history slots take ``history``; the phase is kept."""

        self._validate_state(state)
        self._validate_rows_state(rows_state)
        window = state.conv_window()
        for index in range(CONV_HISTORY_ROWS):
            row = ttnn.slice(
                rows_state.history,
                (0, 0, index, 0),
                (1, 1, index + 1, QKV_WIDTH_PER_DEVICE),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            _copy_inplace(row, window[index], label=f"GDN rows history store {index}")
            _deallocate(row)

    def _all_gather_rows(self, hidden_rows, rows: int):
        """The caller hands either the ``rows`` rows ``[1,1,rows,640]`` or a full 32-row tile ``[1,1,32,640]``
        whose rows past ``rows`` are already zero (a persistent input tile written in place): the second form
        skips the pad."""

        if _shape(hidden_rows) != (1, 1, CHUNK_SIZE, HIDDEN_SIZE_PER_DEVICE):
            _require_shape(hidden_rows, (1, 1, rows, HIDDEN_SIZE_PER_DEVICE), label="GDN rows input")
        self.mesh_contract.validate_tensor(hidden_rows, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        # Zero rows up to one tile on device: the projection of a zero row is exactly zero.  The padding
        # stays inside the input's own tile, so ``ttnn.pad`` returns a view of the input on this runtime
        # (measured in the lab: deallocating it freed the caller's rows); it is never deallocated here.
        if _shape(hidden_rows)[2] < CHUNK_SIZE:
            padded = ttnn.pad(
                hidden_rows,
                [(0, 0), (0, 0), (0, CHUNK_SIZE - rows), (0, 0)],
                0.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            _retag_head_shard_after_reshape(padded, reference=hidden_rows, shard_dim=3)
        else:
            padded = hidden_rows
        _require_shape(padded, (1, 1, CHUNK_SIZE, HIDDEN_SIZE_PER_DEVICE), label="GDN rows padded input")
        full_hidden = ttnn.all_gather(
            padded,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=self.in_proj_act_memory_config,
        )
        self.mesh_contract.validate_tensor(full_hidden, placement=TensorPlacement.REPLICATED)
        _require_shape(full_hidden, (1, 1, CHUNK_SIZE, HIDDEN_SIZE), label="GDN rows gathered hidden")
        return full_hidden

    def _project_rows(self, full_hidden, rows_state: Qwen38TTNNGDNRowsState):
        """The 1-row projection on CHUNK_SIZE rows; the q|k|v columns land in the persistent ``qkv``."""

        projected_ws = ttnn.linear(
            full_hidden,
            self.weights.qkvzab,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.in_proj_program_config,
            compute_kernel_config=self.compute_config,
        )
        projected = ttnn.to_memory_config(projected_ws, ttnn.L1_MEMORY_CONFIG)
        _deallocate(projected_ws)
        self.mesh_contract.validate_tensor(projected, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(projected, (1, 1, CHUNK_SIZE, PROJECTION_WIDTH_PER_DEVICE), label="GDN rows projection")
        landed = ttnn.slice(
            projected, (0, 0, 0, 0), (1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), output_tensor=rows_state.qkv
        )
        _require_landed(landed, rows_state.qkv, label="GDN rows qkv slice")
        columns = {
            "z": (QKV_WIDTH_PER_DEVICE, A_COLUMN),
            "a": (A_COLUMN, A_COLUMN + VALUE_HEADS_PER_DEVICE),
            "b": (B_COLUMN, B_COLUMN + VALUE_HEADS_PER_DEVICE),
        }
        pieces = {}
        for name, (start, end) in columns.items():
            piece = ttnn.slice(
                projected, (0, 0, 0, start), (1, 1, CHUNK_SIZE, end), memory_config=ttnn.L1_MEMORY_CONFIG
            )
            self.mesh_contract.validate_tensor(piece, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
            _require_shape(piece, (1, 1, CHUNK_SIZE, end - start), label=f"GDN rows projected {name}")
            pieces[name] = piece
        _deallocate(projected)
        return pieces["z"], pieces["a"], pieces["b"]

    def _conv_window_rows(self, rows_state: Qwen38TTNNGDNRowsState):
        """``[history tile | qkv tile]``: two whole tiles on dim 2 (no row padding, so a plain tile concat).

        Logical window row m (history rows 0..2, then the new rows) is buffer row ``_window_buffer_row(m)``;
        the FIR taps and the next history are read out of it with the constant 0/1 selection matmuls.
        """

        window = ttnn.concat([rows_state.history, rows_state.qkv], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _retag_head_shard_after_reshape(window, reference=rows_state.qkv, shard_dim=3)
        _require_shape(window, (1, 1, CONV_WINDOW_TILE_ROWS, QKV_WIDTH_PER_DEVICE), label="GDN rows conv window")
        return window

    def _select_rows(self, select, window, *, memory_config, label: str, output_tensor=None):
        """``select @ window``: an exact 0/1 row selection (HiFi4, fp32 accumulation, one nonzero term per
        output element: x * 1.0 plus exact zeros is x, and the bf16 result is the selected bf16 row).  With
        ``output_tensor`` the matmul writes the persistent buffer itself."""

        selected = ttnn.matmul(
            select,
            window,
            memory_config=memory_config,
            compute_kernel_config=self.compute_config,
            optional_output_tensor=output_tensor,
        )
        if output_tensor is not None:
            _require_landed(selected, output_tensor, label=label)
            selected = output_tensor
        _retag_head_shard_after_reshape(selected, reference=window, shard_dim=3)
        _require_shape(selected, (1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), label=label)
        return selected

    def _causal_conv_rows(self, rows_state: Qwen38TTNNGDNRowsState):
        window = self._conv_window_rows(rows_state)
        # Tap t reads logical window rows t .. t + 31; tap 3 is the new rows themselves (the persistent qkv).
        pieces = [
            self._select_rows(select, window, memory_config=ttnn.L1_MEMORY_CONFIG, label=f"GDN rows FIR tap {tap}")
            for tap, select in enumerate(rows_state.constants.conv_taps)
        ]
        _deallocate(window)
        pieces.append(rows_state.qkv)
        # Same tap arithmetic as _causal_conv_decode; the [1,1,1,2560] taps broadcast over the rows.
        conv = ttnn.multiply(pieces[0], self.weights.conv_taps[0], memory_config=ttnn.L1_MEMORY_CONFIG)
        for index in range(1, CONV_KERNEL_SIZE):
            previous = conv
            conv = ttnn.mac(pieces[index], self.weights.conv_taps[index], previous)
            if _tensor_key(conv) != _tensor_key(previous):
                _deallocate(previous)
        _deallocate(*pieces[:CONV_HISTORY_ROWS])
        conv = ttnn.silu(conv, memory_config=ttnn.L1_MEMORY_CONFIG)
        _retag_head_shard_after_reshape(conv, reference=rows_state.qkv, shard_dim=3)
        _require_shape(conv, (1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), label="GDN rows causal convolution")
        return conv

    def _make_chunk_inputs(self, conv, a, b, rows_state: Qwen38TTNNGDNRowsState) -> None:
        """Write the chunk kernel's inputs into the persistent rows buffers.

        q and k are expanded to the 12 value heads first (``qk_expand``: an exact 0/1 matmul, so the chunk
        kernel sees H = HV and skips its GQA repeat), then l2-normalized per row in BF16 exactly as
        ``_recurrent_decode`` does it (rms_norm with eps / head_dim, then head_dim ** -0.5): a copied head
        normalizes to the same bits, so the kernel reads what the 1-row path's repeat_interleave would
        have given it.  beta and the log decay use the 1-row arithmetic.  The row masks zero rows >= R
        (x * 1.0 and x * 0.0 are exact) and are the ops that land in the persistent buffers.
        """

        l1 = ttnn.L1_MEMORY_CONFIG
        constants = rows_state.constants
        q_slice = ttnn.slice(conv, (0, 0, 0, 0), (1, 1, CHUNK_SIZE, QK_WIDTH_PER_DEVICE), memory_config=l1)
        k_slice = ttnn.slice(
            conv, (0, 0, 0, QK_WIDTH_PER_DEVICE), (1, 1, CHUNK_SIZE, 2 * QK_WIDTH_PER_DEVICE), memory_config=l1
        )
        v_slice = ttnn.slice(
            conv, (0, 0, 0, 2 * QK_WIDTH_PER_DEVICE), (1, 1, CHUNK_SIZE, QKV_WIDTH_PER_DEVICE), memory_config=l1
        )
        _deallocate(conv)
        # Heads on dim 2 ([B, T, H, K], the kernel's token-major form); the tile padding of the head
        # axis is zeroed so no padding value reaches the kernel's head split.
        for name, source, target in (("query", q_slice, rows_state.q), ("key", k_slice, rows_state.k)):
            expanded = ttnn.matmul(
                source, constants.qk_expand, memory_config=l1, compute_kernel_config=self.compute_config
            )
            _retag_head_shard_after_reshape(expanded, reference=source, shard_dim=3)
            _require_shape(expanded, (1, 1, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE), label=f"GDN rows expanded {name}")
            _deallocate(source)
            heads_tensor = ttnn.reshape(expanded, (1, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE, HEAD_DIM), pad_value=0.0)
            _retag_head_shard_after_reshape(heads_tensor, reference=expanded, shard_dim=2)
            _deallocate(expanded)
            normed = ttnn.rms_norm(heads_tensor, epsilon=QK_L2_NORM_EPS / HEAD_DIM)
            _deallocate(heads_tensor)
            unit = ttnn.multiply(normed, HEAD_DIM**-0.5, memory_config=l1)
            _deallocate(normed)
            landed = ttnn.multiply(unit, constants.row_mask_bf16, output_tensor=target)
            _require_landed(landed, target, label=f"GDN rows {name}")
            _deallocate(unit)
        # v stays token-major flat [1, 1, T, 1536]: the composite's flat-v form (no head split, no fill, no
        # transpose; the prep reader addresses head h's tiles at columns 128h.. of the same tile row).
        landed = ttnn.multiply(v_slice, constants.row_mask_bf16_col, output_tensor=rows_state.v)
        _require_landed(landed, rows_state.v, label="GDN rows value")
        _deallocate(v_slice)

        b_fp32 = ttnn.typecast(b, ttnn.float32, memory_config=l1)
        _deallocate(b)
        beta_fp32 = ttnn.sigmoid(b_fp32, memory_config=l1)
        _deallocate(b_fp32)
        landed = ttnn.multiply(beta_fp32, constants.row_mask_fp32, output_tensor=rows_state.beta)
        _require_landed(landed, rows_state.beta, label="GDN rows beta")
        _deallocate(beta_fp32)

        a_fp32 = ttnn.typecast(a, ttnn.float32, memory_config=l1)
        _deallocate(a)
        softplus = ttnn.add(
            a_fp32,
            self.weights.dt_bias,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
            memory_config=l1,
        )
        _deallocate(a_fp32)
        log_decay = ttnn.multiply(self.weights.neg_exp_A, softplus, memory_config=l1)
        _deallocate(softplus)
        landed = ttnn.multiply(log_decay, constants.row_mask_fp32, output_tensor=rows_state.g)
        _require_landed(landed, rows_state.g, label="GDN rows log decay")
        _deallocate(log_decay)
        rows_state.validate()

    def _chunk_rows(self, rows_state: Qwen38TTNNGDNRowsState, initial_state, committed_mask=None):
        """One full-chunk run of the kernel from ``initial_state`` (read only).

        Returns the head-major output ``[VALUE_HEADS_PER_DEVICE, CHUNK_SIZE, HEAD_DIM]`` (TILE, the kernel's
        own layout and output dtype: FP32 on the pinned runtime, measured in the lab; ``output_head_major``
        skips the composite's untilize / row-major permute) and the FP32 final state
        ``[1, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM]`` in a new buffer.  With ``committed_mask`` beta
        and g of the rows past the committed prefix are zeroed first (the catch-up), which makes those
        rows an identity update.
        """

        constants = rows_state.constants
        if committed_mask is None:
            beta, g, masked = rows_state.beta, rows_state.g, ()
        else:
            beta = ttnn.multiply(rows_state.beta, committed_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            g = ttnn.multiply(rows_state.g, committed_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            masked = (beta, g)
        # [1, 1, T, HV] -> [1, T, HV] and [1, 1, T, HV * V] -> [1, T, HV * V]: the tile grid is unchanged, so
        # these are views of the buffers.  The rank-3 v is the composite's token-major flat form.
        beta_rows = ttnn.reshape(beta, (1, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE))
        g_rows = ttnn.reshape(g, (1, CHUNK_SIZE, VALUE_HEADS_PER_DEVICE))
        v_rows = ttnn.reshape(rows_state.v, (1, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE))
        output, final_state = ttnn.transformer.chunk_gated_delta_rule(
            rows_state.q,
            rows_state.k,
            v_rows,
            g_rows,
            beta_rows,
            scale=HEAD_DIM**-0.5,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=CHUNK_SIZE,
            output_head_major=True,
            eye=constants.eye,
            tril=constants.tril,
            ones=constants.ones,
            masks=constants.masks,
        )
        _deallocate(*masked)
        if final_state is None:
            raise RuntimeError("chunk_gated_delta_rule returned no final state")
        _retag_head_shard_after_reshape(final_state, reference=initial_state, shard_dim=1)
        _require_shape(final_state, (1, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM), label="GDN rows final state")
        if final_state.dtype != ttnn.float32:
            raise RuntimeError(f"GDN rows final state must be FP32, got {final_state.dtype}")
        _retag_head_shard_after_reshape(output, reference=rows_state.v, shard_dim=0)
        _require_shape(output, (VALUE_HEADS_PER_DEVICE, CHUNK_SIZE, HEAD_DIM), label="GDN rows recurrent output")
        if output.dtype not in (ttnn.bfloat16, ttnn.float32) or output.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError(
                f"GDN rows recurrent output must be BF16 or FP32 TILE, got {output.dtype} {output.layout}"
            )
        return output, final_state

    def _gate_and_project_rows(self, recurrent_output, z, full_hidden, rows_state: Qwen38TTNNGDNRowsState):
        """``_gate_and_project`` on CHUNK_SIZE rows; the first ``rows`` rows of the reduce-scatter land in ``rows_state.output``."""

        rows = rows_state.constants.rows

        l1 = ttnn.L1_MEMORY_CONFIG
        # Head-major [HV, T, V] TILE -> [1, HV, T, V] is a view (last two dims unchanged): one (head, token)
        # row per tile row, so the per-head RMSNorm is the same [.., 128] row op as the 1-row path's, on the
        # same values.  The view shares the kernel output's buffer; only the original is deallocated.
        head_rows = ttnn.reshape(recurrent_output, (1, VALUE_HEADS_PER_DEVICE, CHUNK_SIZE, HEAD_DIM))
        _retag_head_shard_after_reshape(head_rows, reference=z, shard_dim=1)
        # The 1-row path's boundary: the FP32 recurrent output becomes BF16 before the per-head RMSNorm.
        head_rows_bf16 = ttnn.typecast(head_rows, ttnn.bfloat16, memory_config=l1)
        _deallocate(recurrent_output)
        _retag_head_shard_after_reshape(head_rows_bf16, reference=z, shard_dim=1)
        normalized_heads = ttnn.rms_norm(
            head_rows_bf16, weight=self.weights.norm, epsilon=RMS_NORM_EPS, memory_config=l1
        )
        _deallocate(head_rows_bf16)
        _require_shape(
            normalized_heads, (1, VALUE_HEADS_PER_DEVICE, CHUNK_SIZE, HEAD_DIM), label="GDN rows normalized heads"
        )
        # Head-major [1, HV, 32, 128] and token-major [1, 1, 32, HV * 128] TILE tensors store their tiles in the
        # same order (tile (h, c) at 4h + c), so the fold to the gate's row form is a metadata view of the
        # same buffer: no permute, no relayout.  The view owns the buffer from here on.
        normalized = ttnn.experimental.view(normalized_heads, (1, 1, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE))
        _retag_head_shard_after_reshape(normalized, reference=z, shard_dim=3)
        self.mesh_contract.validate_tensor(normalized, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(normalized, (1, 1, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE), label="GDN rows normalized output")

        z_fp32 = ttnn.typecast(z, ttnn.float32, memory_config=l1)
        _deallocate(z)
        sigmoid_fp32 = ttnn.sigmoid(z_fp32, memory_config=l1)
        _deallocate(z_fp32)
        sigmoid_bf16 = ttnn.typecast(sigmoid_fp32, ttnn.bfloat16, memory_config=l1)
        _deallocate(sigmoid_fp32)
        gated = ttnn.multiply(normalized, sigmoid_bf16, memory_config=self.out_proj_act_memory_config)
        _deallocate(normalized, sigmoid_bf16)
        self.mesh_contract.validate_tensor(gated, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(gated, (1, 1, CHUNK_SIZE, VALUE_WIDTH_PER_DEVICE), label="GDN rows sigmoid-gated output")

        partial_ws = ttnn.linear(
            gated,
            self.weights.out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.out_proj_program_config,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(gated)
        self.mesh_contract.mark_local_partial(
            partial_ws,
            replicated_reference=full_hidden,
            expected_shape=(1, 1, CHUNK_SIZE, HIDDEN_SIZE),
        )
        output = ttnn.reduce_scatter(
            partial_ws,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.collective_topology,
        )
        _deallocate(partial_ws)
        self.mesh_contract.mark_collective_shard(
            output,
            replicated_reference=full_hidden,
            shard_dim=3,
            expected_local_shape=(1, 1, CHUNK_SIZE, HIDDEN_SIZE_PER_DEVICE),
        )
        _deallocate(full_hidden)
        # The row slice of the first tile may alias its input on this runtime; writing it into the persistent
        # output buffer (the 1-row path's slice form) keeps a fixed address and lets the 32-row tensor go.
        landed = ttnn.slice(output, (0, 0, 0, 0), (1, 1, rows, HIDDEN_SIZE_PER_DEVICE), output_tensor=rows_state.output)
        _require_landed(landed, rows_state.output, label="GDN rows output slice")
        _deallocate(output)
        self.mesh_contract.validate_tensor(rows_state.output, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        _require_shape(rows_state.output, (1, 1, rows, HIDDEN_SIZE_PER_DEVICE), label="GDN rows output")
        return rows_state.output

    def forward_rows(
        self, hidden_rows, state: Qwen38TTNNGDNState, rows_state: Qwen38TTNNGDNRowsState
    ) -> Qwen38TTNNGDNRowsResult:
        """Run ``rows`` consecutive positions from the committed state without committing anything.

        ``hidden_rows`` has local shape ``[1,1,rows,640]`` (hidden sharded).  ``state.recurrent`` is read
        only; the ring and its phase are not touched (the rows path keeps its own ``history``).  The
        result's ``final_state`` is the state after all rows in a new buffer.
        """

        self._validate_state(state)
        rows = self._validate_rows_state(rows_state)
        full_hidden = self._all_gather_rows(hidden_rows, rows)
        z, a, b = self._project_rows(full_hidden, rows_state)
        conv = self._causal_conv_rows(rows_state)
        self._make_chunk_inputs(conv, a, b, rows_state)
        recurrent_output, final_state = self._chunk_rows(rows_state, initial_state=state.recurrent)
        output = self._gate_and_project_rows(recurrent_output, z, full_hidden, rows_state)
        return Qwen38TTNNGDNRowsResult(output, final_state, state, rows_state)

    def _advance_history_rows(self, rows_state: Qwen38TTNNGDNRowsState, selectors: Qwen38TTNNRowsSelectors) -> None:
        """``history <- window[c : c + 3]`` with ``c = accepted + 1``: one exact 0/1 selection matmul against
        the per-pass ``history_select`` (rows 0..2 of the result are the three rows, rows 3..31 exact zeros),
        written by the matmul into the persistent history tile (the window is a separate buffer)."""

        window = self._conv_window_rows(rows_state)
        self._select_rows(
            selectors.history_select,
            window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            label="GDN rows history select",
            output_tensor=rows_state.history,
        )
        _deallocate(window)

    def _step_row_state(self, rows_state: Qwen38TTNNGDNRowsState, initial_state, row: int):
        """Row ``row`` of the pass through the 1-row FP32 step arithmetic of ``_recurrent_decode``, into a new buffer.

        The rows buffers already hold the l2-normalized, GQA-expanded BF16 q/k (the 12 head rows the 1-row
        path's repeat_interleave produces), so this starts at the step's FP32 promotion: the same ops,
        program and compute configs from there on (bitwise the 1-row path's state update).
        """

        l1 = ttnn.L1_MEMORY_CONFIG
        # Only the state update is needed (no q read): k, v, beta and the log decay of the row.
        k_row_tile = ttnn.slice(
            rows_state.k, (0, row, 0, 0), (1, row + 1, VALUE_HEADS_PER_DEVICE, HEAD_DIM), memory_config=l1
        )
        k_unit = ttnn.reshape(k_row_tile, (1, VALUE_HEADS_PER_DEVICE, 1, HEAD_DIM))
        _deallocate(k_row_tile)
        v_row = ttnn.slice(rows_state.v, (0, 0, row, 0), (1, 1, row + 1, VALUE_WIDTH_PER_DEVICE), memory_config=l1)
        v = ttnn.reshape(v_row, (1, VALUE_HEADS_PER_DEVICE, 1, HEAD_DIM))
        _deallocate(v_row)
        scalars = []
        for source in (rows_state.beta, rows_state.g):
            scalar_row = ttnn.slice(source, (0, 0, row, 0), (1, 1, row + 1, VALUE_HEADS_PER_DEVICE), memory_config=l1)
            scalars.append(ttnn.reshape(scalar_row, (1, VALUE_HEADS_PER_DEVICE, 1, 1), memory_config=l1))
            _deallocate(scalar_row)
        beta, log_decay = scalars
        for name, tensor, expected in (
            ("key", k_unit, (1, VALUE_HEADS_PER_DEVICE, 1, HEAD_DIM)),
            ("value", v, (1, VALUE_HEADS_PER_DEVICE, 1, HEAD_DIM)),
            ("beta", beta, (1, VALUE_HEADS_PER_DEVICE, 1, 1)),
            ("log_decay", log_decay, (1, VALUE_HEADS_PER_DEVICE, 1, 1)),
        ):
            _require_shape(tensor, expected, label=f"GDN rows row-{row} step {name}")

        k_row = ttnn.typecast(k_unit, ttnn.float32, memory_config=l1)
        _deallocate(k_unit)
        decayed = ttnn.multiply(
            initial_state,
            log_decay,
            input_tensor_b_activations=[ttnn.UnaryOpType.EXP],
            memory_config=l1,
        )
        v_read = ttnn.matmul(
            k_row,
            decayed,
            memory_config=l1,
            program_config=self.recurrent_matmul_program_config,
            compute_kernel_config=self.recurrent_read_compute_config,
        )
        delta = ttnn.subtract(v, v_read, dtype=ttnn.float32, memory_config=l1)
        _deallocate(v, v_read)
        k_col = ttnn.transpose(k_row, 2, 3, memory_config=l1)
        _deallocate(k_row)
        outer = ttnn.matmul(
            k_col,
            delta,
            memory_config=l1,
            compute_kernel_config=self.recurrent_write_compute_config,
        )
        _deallocate(k_col, delta)
        update = ttnn.multiply(outer, beta, memory_config=l1)
        _deallocate(outer)
        stepped = ttnn.add(decayed, update, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(decayed, update, beta, log_decay)
        _retag_head_shard_after_reshape(stepped, reference=initial_state, shard_dim=1)
        _require_shape(stepped, (1, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM), label=f"GDN rows row-{row} step state")
        return stepped

    def _step_committed_rows_state(
        self, rows_state: Qwen38TTNNGDNRowsState, initial_state, selectors: Qwen38TTNNRowsSelectors
    ):
        """The state after the committed rows through the 1-row FP32 step arithmetic, into a new buffer.

        All ``rows`` rows are stepped in sequence from ``initial_state`` (a trace cannot bind the accept count)
        and the state after row ``accepted`` is selected with the exact one-hot multiply/add (``x * 1.0``,
        ``x * 0.0`` and ``+ 0.0`` are exact in FP32).  ``rows`` x (the step's ops) + 3 ``rows`` select ops.
        """

        dram = ttnn.DRAM_MEMORY_CONFIG
        state = initial_state
        selected = None
        for row in range(rows_state.constants.rows):
            stepped = self._step_row_state(rows_state, state, row)
            if state is not initial_state:
                _deallocate(state)
            state = stepped
            weight = ttnn.typecast(selectors.onehot_bf16[row], ttnn.float32, memory_config=dram)
            weighted = ttnn.multiply(stepped, weight, memory_config=dram)
            _deallocate(weight)
            if selected is None:
                selected = weighted
            else:
                summed = ttnn.add(selected, weighted, memory_config=dram)
                _deallocate(selected, weighted)
                selected = summed
        _deallocate(state)
        _retag_head_shard_after_reshape(selected, reference=initial_state, shard_dim=1)
        _require_shape(
            selected, (1, VALUE_HEADS_PER_DEVICE, HEAD_DIM, HEAD_DIM), label="GDN rows committed rows step state"
        )
        if selected.dtype != ttnn.float32:
            raise RuntimeError(f"GDN rows committed rows step state must be FP32, got {selected.dtype}")
        return selected

    def commit_rows(
        self,
        state: Qwen38TTNNGDNState,
        rows_state: Qwen38TTNNGDNRowsState,
        selectors: Qwen38TTNNRowsSelectors,
        *,
        step_on_full_rejection: bool = False,
        step_committed_rows: bool = False,
    ) -> None:
        """Commit ``accepted + 1`` rows of the last ``forward_rows`` pass into ``state`` in place.

        The catch-up reruns the chunk kernel from the committed state with beta and g of the rows past the
        committed prefix zeroed (``selectors.committed_mask``); the resulting state lands in
        ``state.recurrent`` and the FIR history moves forward by ``accepted + 1`` rows.

        Row 0 (the base token) is always committed, so there is no identity pass: the design's earlier
        "restore the old state when nothing was accepted" switch (written for the ``arange < a`` mask) has no
        case left.  The default therefore accepts the kernel's TF32-class state rounding on every commit (the
        design's decision: 4.9e-4 on a passthrough, below the kernel's own 2.1e-3 state error).  The switch
        kept in its place, ``step_on_full_rejection``, commits a full rejection (one row) through the 1-row
        FP32 step arithmetic instead, selected exactly against the kernel's result
        (``step * [a == 0] + chunk * [a != 0]``); it is a diagnostic for a state divergence that tracks full
        rejections and costs about 20 more ops per layer.  ``step_committed_rows`` (the state re-anchor)
        replaces the kernel's committed state altogether: every committed row goes through the 1-row FP32 step
        arithmetic (``_step_committed_rows_state``), so the committed state is the 1-row path's; the chunk
        kernel is not rerun.  About ``rows`` x 16 + 3 ``rows`` ops per layer.
        """

        self._validate_state(state)
        rows = self._validate_rows_state(rows_state)
        selectors.validate(rows)
        if step_committed_rows:
            stepped = self._step_committed_rows_state(rows_state, state.recurrent, selectors)
            _copy_inplace(stepped, state.recurrent, label="GDN rows committed rows step state")
            _deallocate(stepped)
            self._advance_history_rows(rows_state, selectors)
            state.validate()
            return
        output, final_state = self._chunk_rows(
            rows_state, initial_state=state.recurrent, committed_mask=selectors.committed_mask
        )
        _deallocate(output)
        if step_on_full_rejection:
            dram = ttnn.DRAM_MEMORY_CONFIG
            stepped = self._step_row_state(rows_state, state.recurrent, 0)
            # ``accepted == 0`` and its complement as fp32 scalars (exact 0/1 selects of the state source).
            full_rejection = ttnn.typecast(selectors.onehot_bf16[0], ttnn.float32, memory_config=dram)
            partial_acceptance = ttnn.subtract(rows_state.constants.one, full_rejection, memory_config=dram)
            from_step = ttnn.multiply(stepped, full_rejection, memory_config=dram)
            from_chunk = ttnn.multiply(final_state, partial_acceptance, memory_config=dram)
            landed = ttnn.add(from_step, from_chunk, output_tensor=state.recurrent)
            _require_landed(landed, state.recurrent, label="GDN rows committed state select")
            _deallocate(stepped, from_step, from_chunk, final_state, full_rejection, partial_acceptance)
        else:
            _copy_inplace(final_state, state.recurrent, label="GDN rows committed state")
            _deallocate(final_state)
        self._advance_history_rows(rows_state, selectors)
        state.validate()


def validate_gdn_static_contract() -> None:
    """No-device invariant gate used by static bring-up tests."""

    exact = {
        "TP_SIZE": TP_SIZE,
        "HIDDEN_SIZE": HIDDEN_SIZE,
        "HIDDEN_SIZE_PER_DEVICE": HIDDEN_SIZE_PER_DEVICE,
        "QK_HEADS": QK_HEADS,
        "QK_HEADS_PER_DEVICE": QK_HEADS_PER_DEVICE,
        "VALUE_HEADS": VALUE_HEADS,
        "VALUE_HEADS_PER_DEVICE": VALUE_HEADS_PER_DEVICE,
        "HEAD_DIM": HEAD_DIM,
        "QKV_WIDTH": QKV_WIDTH,
        "QKV_WIDTH_PER_DEVICE": QKV_WIDTH_PER_DEVICE,
        "VALUE_WIDTH_PER_DEVICE": VALUE_WIDTH_PER_DEVICE,
        "QKVZAB_WIDTH_PER_DEVICE": QKVZAB_WIDTH_PER_DEVICE,
        "A_COLUMN": A_COLUMN,
        "B_COLUMN": B_COLUMN,
        "PROJECTION_WIDTH_PER_DEVICE": PROJECTION_WIDTH_PER_DEVICE,
        "CONV_KERNEL_SIZE": CONV_KERNEL_SIZE,
        "QK_REPEAT_FACTOR": QK_REPEAT_FACTOR,
    }
    expected = {
        "TP_SIZE": 4,
        "HIDDEN_SIZE": 2560,
        "HIDDEN_SIZE_PER_DEVICE": 640,
        "QK_HEADS": 16,
        "QK_HEADS_PER_DEVICE": 4,
        "VALUE_HEADS": 48,
        "VALUE_HEADS_PER_DEVICE": 12,
        "HEAD_DIM": 128,
        "QKV_WIDTH": 10240,
        "QKV_WIDTH_PER_DEVICE": 2560,
        "VALUE_WIDTH_PER_DEVICE": 1536,
        "QKVZAB_WIDTH_PER_DEVICE": 4120,
        "A_COLUMN": 4096,
        "B_COLUMN": 4128,
        "PROJECTION_WIDTH_PER_DEVICE": 4160,
        "CONV_KERNEL_SIZE": 4,
        "QK_REPEAT_FACTOR": 3,
    }
    if exact != expected:
        raise RuntimeError(f"Qwen3.8 GDN static contract drifted: {exact} != {expected}")


__all__ = [
    "Qwen38TTNNGDN",
    "Qwen38TTNNGDNResult",
    "Qwen38TTNNGDNRowsConstants",
    "Qwen38TTNNGDNRowsResult",
    "Qwen38TTNNGDNRowsState",
    "Qwen38TTNNGDNWeights",
    "Qwen38TTNNGDNState",
    "Qwen38TTNNGDNSnapshot",
    "Qwen38TTNNRowsSelectors",
    "build_rows_selectors",
    "chunk_constant_tiles",
    "rows_window_select_tiles",
    "validate_gdn_static_contract",
]
