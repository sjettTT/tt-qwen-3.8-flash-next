# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed TTNN mesh and tensor-placement contracts.

Numerically identical tensors with different mesh topology are not
interchangeable in this model.  Every device module calls these guards after
uploads, cache loads, layout changes, and collectives.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import torch

import ttnn

TP_SIZE = 4
MESH_SHAPE = (1, TP_SIZE)
POSITION_SCALAR_SHAPE = (1, 1, 1, 1)
POSITION_INDEX_ROW_SHAPE = (1, 1, 1, ttnn.TILE_SIZE)
# P & ~3 lane mask.  Kept in a tensor: values >= 2**31 must not take the scalar
# operand path of the binary ops.
BLOCK_START_LANE_MASK = 0xFFFFFFFC
UINT32_LIMIT = 2**32
# Rows of one prefill chunk: one 32-row tile, so every decode matmul keeps its program.
CHUNK_ROWS = ttnn.TILE_SIZE


class TensorPlacement(str, Enum):
    HIDDEN_SHARDED = "hidden_sharded"
    EXPERT_SHARDED = "expert_sharded"
    ROUTER_SHARDED = "router_sharded"
    INTERMEDIATE_SHARDED = "intermediate_sharded"
    VOCAB_SHARDED = "vocab_sharded"
    HEAD_SHARDED = "head_sharded"
    KV_PAIR_GROUPED = "kv_pair_grouped"
    REPLICATED = "replicated"
    LOCAL_PARTIAL = "local_partial"


def _exact_integer(value, *, label: str, error_type: type[Exception] = RuntimeError) -> int:
    """Return an integer-protocol value without accepting scalar aliases.

    Nanobind integer results and other legitimate bound integer types implement
    ``__index__``.  Decimal strings and floating-point values do not.  ``bool``
    also implements ``__index__``, so reject it explicitly before conversion.
    """

    if isinstance(value, bool):
        raise error_type(f"{label} must be an exact integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise error_type(f"{label} must be an exact integer, got {value!r}") from error
    if isinstance(result, bool) or not isinstance(result, int):
        raise error_type(f"{label} must be an exact integer, got {value!r}")
    return result


def _shape_tuple(
    value,
    *,
    label: str = "shape",
    error_type: type[Exception] = RuntimeError,
) -> tuple[int, ...]:
    try:
        items = tuple(value)
    except TypeError as error:
        raise error_type(f"{label} must be an iterable of exact integers") from error
    return tuple(
        _exact_integer(item, label=f"{label}[{index}]", error_type=error_type) for index, item in enumerate(items)
    )


def _placement_name(value) -> str:
    return type(value).__name__


def tensor_metadata(tensor) -> str:
    """The four fields every metadata contract checks, for actual-vs-expected error text."""

    return (
        f"shape={list(_shape_tuple(tensor.shape))} padded_shape={list(_shape_tuple(tensor.padded_shape))} "
        f"dtype={tensor.dtype} layout={tensor.layout}"
    )


@dataclass(frozen=True)
class Qwen38TensorBackingIdentity:
    """Exact ownership identity for one distributed device allocation.

    ``tensor_id`` identifies a TTNN graph object and Python object identity
    identifies only one wrapper.  Neither is an allocation identity.  Device
    lifetime ledgers instead bind every local backing buffer to its exact mesh
    coordinate and physical device.  This permits two wrappers over the same
    four buffers to share one release while rejecting ambiguous topology.
    """

    distribution_shape: tuple[int, int]
    members: tuple[tuple[tuple[int, int], int, int], ...]


def qwen38_tensor_backing_identity(tensor: Any) -> Qwen38TensorBackingIdentity:
    """Return the exact ordered 1x4 physical backing identity of ``tensor``.

    The function is deliberately fail closed.  A device object without the
    full tensor-topology and per-device buffer-address APIs cannot be safely
    deduplicated for deallocation and must remain owned until mesh teardown.
    """

    try:
        device = tensor.device()
        topology = tensor.tensor_topology()
        distribution_shape = _shape_tuple(
            topology.distribution_shape(),
            label="backing tensor distribution shape",
        )
        coordinates = tuple(
            _shape_tuple(coordinate, label=f"backing tensor mesh coordinate[{index}]")
            for index, coordinate in enumerate(tuple(topology.mesh_coords()))
        )
        mesh_shape = _shape_tuple(device.shape, label="backing tensor mesh-device shape")
        physical_ids = _shape_tuple(
            device.get_device_ids(),
            label="backing tensor mesh-device physical IDs",
        )
        local_tensors = tuple(ttnn.get_device_tensors(tensor))
    except BaseException as error:
        raise RuntimeError("cannot resolve exact distributed tensor backing identity") from error

    expected_coordinates = tuple((0, column) for column in range(TP_SIZE))
    if (
        distribution_shape != MESH_SHAPE
        or mesh_shape != MESH_SHAPE
        or coordinates != expected_coordinates
        or len(physical_ids) != TP_SIZE
        or len(set(physical_ids)) != TP_SIZE
        or any(physical_id < 0 for physical_id in physical_ids)
        or len(local_tensors) != TP_SIZE
    ):
        raise RuntimeError(
            "distributed tensor backing topology differs from exact 1x4: "
            f"distribution={distribution_shape} device={mesh_shape} coordinates={coordinates} "
            f"physical_ids={physical_ids} local_count={len(local_tensors)}"
        )

    members: list[tuple[tuple[int, int], int, int]] = []
    for index, (coordinate, expected_physical_id, local_tensor) in enumerate(
        zip(coordinates, physical_ids, local_tensors, strict=True)
    ):
        try:
            local_device = local_tensor.device()
            local_coordinates = tuple(
                _shape_tuple(item, label=f"backing tensor local device coordinate[{index}]")
                for item in tuple(local_tensor.device_coords())
            )
            physical_id = _exact_integer(
                device.get_device_id(ttnn.MeshCoordinate(*coordinate)),
                label=f"backing tensor parent-mesh physical ID[{index}]",
            )
            buffer_address = _exact_integer(
                local_tensor.buffer_address(),
                label=f"backing tensor local buffer address[{index}]",
            )
        except BaseException as error:
            raise RuntimeError(f"cannot resolve backing tensor member at mesh coordinate {coordinate}") from error
        if (
            local_device is not device
            or local_coordinates != (coordinate,)
            or physical_id != expected_physical_id
            or buffer_address < 0
        ):
            raise RuntimeError(
                f"backing tensor member {coordinate} resolved parent/coords/physical/address "
                f"{local_device is device}/{local_coordinates}/{physical_id}/{buffer_address}, "
                f"expected parent/coords/physical True/{(coordinate,)}/{expected_physical_id}"
            )
        members.append((coordinate, physical_id, buffer_address))
    return Qwen38TensorBackingIdentity(MESH_SHAPE, tuple(members))


def replicate_tensor_2d_mesh_mapper(mesh_device):
    """Replicate over both axes of the model's explicit ``(1, 4)`` topology."""

    return ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()],
            ttnn.MeshShape(*MESH_SHAPE),
        ),
    )


@dataclass(frozen=True)
class Qwen38MeshContract:
    """The physical and logical mesh admitted by the four-P150 model.

    ``physical_ids`` are the TTNN IDs returned by ``MeshDevice.get_device_ids``
    in mesh-coordinate order.  The caller must derive and record their node,
    BDF, board-id, and fabric mapping before opening the mesh.
    """

    physical_ids: tuple[int, int, int, int]
    mesh_shape: tuple[int, int] = MESH_SHAPE

    def __post_init__(self) -> None:
        canonical_mesh_shape = _shape_tuple(self.mesh_shape, label="mesh_shape", error_type=ValueError)
        if canonical_mesh_shape != MESH_SHAPE:
            raise ValueError(f"Qwen3.8-Flash-Next requires mesh {MESH_SHAPE}, got {self.mesh_shape}")
        if not isinstance(self.physical_ids, tuple):
            raise ValueError(f"expected four distinct physical IDs, got {self.physical_ids}")
        try:
            canonical_physical_ids = _shape_tuple(
                self.physical_ids,
                label="physical_ids",
                error_type=ValueError,
            )
        except ValueError as error:
            raise ValueError(f"expected four distinct physical IDs, got {self.physical_ids}") from error
        if (
            len(canonical_physical_ids) != TP_SIZE
            or len(set(canonical_physical_ids)) != TP_SIZE
            or any(device_id < 0 for device_id in canonical_physical_ids)
        ):
            raise ValueError(f"expected four distinct physical IDs, got {self.physical_ids}")
        object.__setattr__(self, "mesh_shape", canonical_mesh_shape)
        object.__setattr__(self, "physical_ids", canonical_physical_ids)

    def validate_mesh(self, mesh_device) -> None:
        actual_shape = _shape_tuple(mesh_device.shape, label="opened mesh shape")
        if actual_shape != MESH_SHAPE:
            raise RuntimeError(f"opened mesh shape {actual_shape} does not match leased {self.mesh_shape}")
        actual_count = _exact_integer(mesh_device.get_num_devices(), label="opened mesh device count")
        if actual_count != TP_SIZE:
            raise RuntimeError(f"opened mesh has {actual_count} devices, expected {TP_SIZE}")
        actual_ids = _shape_tuple(mesh_device.get_device_ids(), label="opened mesh physical IDs")
        if actual_ids != self.physical_ids:
            raise RuntimeError(
                f"opened mesh physical order {actual_ids} does not match leased order {self.physical_ids}"
            )

    def validate_tensor(
        self,
        tensor,
        *,
        placement: TensorPlacement,
        shard_dim: int | None = None,
        require_device: bool = True,
    ) -> None:
        """Validate the actual ``tensor_topology()``, not only its values.

        All model tensors use a 2-D distribution shape ``(1, 4)``.  A sharded
        tensor must replicate over the size-one row and shard over the four-way
        column; an intentionally replicated tensor must replicate on both axes.
        Pair-grouped KV is represented by an expanded two-head tensor sharded on
        the head dimension, not by a replicated topology.
        """

        device = tensor.device()
        if require_device:
            if device is None:
                raise RuntimeError(f"{placement.value} tensor is not device resident")
            self.validate_mesh(device)
        topology = tensor.tensor_topology()
        distribution_shape = _shape_tuple(
            topology.distribution_shape(),
            label=f"{placement.value} tensor distribution shape",
        )
        if distribution_shape != self.mesh_shape:
            raise RuntimeError(f"{placement.value} tensor distribution {distribution_shape} must be {self.mesh_shape}")
        raw_coords = tuple(topology.mesh_coords())
        coords = tuple(
            _shape_tuple(coord, label=f"{placement.value} tensor mesh coordinate[{index}]")
            for index, coord in enumerate(raw_coords)
        )
        expected_coords = tuple((0, column) for column in range(TP_SIZE))
        if coords != expected_coords:
            raise RuntimeError(f"{placement.value} tensor mesh-coordinate order {coords} must be {expected_coords}")

        placements = tuple(topology.placements())
        names = tuple(_placement_name(item) for item in placements)
        if len(placements) != 2:
            raise RuntimeError(f"{placement.value} tensor must have two placement axes, got {names}")
        if placement in (TensorPlacement.REPLICATED, TensorPlacement.LOCAL_PARTIAL):
            if names != ("PlacementReplicate", "PlacementReplicate"):
                raise RuntimeError(f"{placement.value} tensor has unexpected placements {names}")
            return

        if shard_dim is None:
            raise ValueError(f"shard_dim is required for {placement.value}")
        expected_dim = _exact_integer(shard_dim, label=f"{placement.value} shard_dim", error_type=ValueError)
        if names != ("PlacementReplicate", "PlacementShard"):
            raise RuntimeError(f"{placement.value} tensor has unexpected placements {names}")
        actual_dim = _exact_integer(
            getattr(placements[1], "dim"),
            label=f"{placement.value} tensor PlacementShard.dim",
        )
        if actual_dim != expected_dim:
            raise RuntimeError(f"{placement.value} tensor shards dim {actual_dim}, expected {expected_dim}")

    @staticmethod
    def require_equal_local_shapes(tensors: Iterable, *, label: str) -> tuple[int, ...]:
        shapes = tuple(_shape_tuple(tensor.shape, label=f"{label} tensor shape") for tensor in tensors)
        if not shapes:
            raise ValueError(f"{label} has no tensors")
        if len(set(shapes)) != 1:
            raise RuntimeError(f"{label} local shapes differ across the mesh: {shapes}")
        return shapes[0]

    def mark_local_partial(self, tensor, *, replicated_reference, expected_shape: tuple[int, ...]) -> None:
        """Label a full-shape per-device additive partial without copying values.

        TTNN has no placement kind for values which have the complete local shape
        but differ by device pending an all-reduce/reduce-scatter.  Such tensors
        must carry Replicate *shape* metadata so the following collective sees a
        full logical operand; ``LOCAL_PARTIAL`` records the stronger semantic
        invariant in this model.  This method is only valid after a row-parallel
        operation whose local output width is complete.
        """

        self.validate_tensor(replicated_reference, placement=TensorPlacement.REPLICATED)
        if _shape_tuple(tensor.shape) != _shape_tuple(expected_shape):
            raise RuntimeError(
                f"local partial shape {_shape_tuple(tensor.shape)} does not match expected {expected_shape}"
            )
        reference_topology = replicated_reference.tensor_topology()
        topology = ttnn.TensorTopology(
            reference_topology.distribution_shape(),
            [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()],
            reference_topology.mesh_coords(),
        )
        tensor.update_tensor_topology(topology)
        self.validate_tensor(tensor, placement=TensorPlacement.LOCAL_PARTIAL)

    def mark_collective_shard(
        self,
        tensor,
        *,
        replicated_reference,
        shard_dim: int,
        expected_local_shape: tuple[int, ...],
    ) -> None:
        """Repair the known legacy reduce-scatter topology omission.

        Standard ``ttnn.reduce_scatter`` changes the physical/local tensor shape
        but currently has no ``compute_output_topologies`` hook.  Call this only
        immediately after that collective; it records the actual shard placement
        without moving or replicating data.
        """

        canonical_shard_dim = _exact_integer(shard_dim, label="collective shard_dim", error_type=ValueError)
        self.validate_tensor(replicated_reference, placement=TensorPlacement.REPLICATED)
        if _shape_tuple(tensor.shape) != _shape_tuple(expected_local_shape):
            raise RuntimeError(
                f"collective shard shape {_shape_tuple(tensor.shape)} does not match expected "
                f"{expected_local_shape}"
            )
        reference_topology = replicated_reference.tensor_topology()
        topology = ttnn.TensorTopology(
            reference_topology.distribution_shape(),
            [ttnn.PlacementReplicate(), ttnn.PlacementShard(canonical_shard_dim)],
            reference_topology.mesh_coords(),
        )
        tensor.update_tensor_topology(topology)
        self.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=canonical_shard_dim)


def _tensor_key(tensor) -> tuple[str, int]:
    tensor_id = getattr(tensor, "tensor_id", None)
    if callable(tensor_id):
        tensor_id = tensor_id()
    return ("ttnn", int(tensor_id)) if tensor_id is not None else ("python", id(tensor))


def _host_uint32(shape: tuple[int, ...], value, *, label: str) -> torch.Tensor:
    value = _exact_integer(value, label=label, error_type=ValueError)
    if not 0 <= value < UINT32_LIMIT:
        raise ValueError(f"{label} must be in [0,{UINT32_LIMIT}), got {value}")
    return torch.full(shape, value, dtype=torch.int64).to(torch.uint32)


def _require_uint32_row(tensor, shape: tuple[int, ...], mesh_contract: Qwen38MeshContract, *, label: str) -> None:
    if _shape_tuple(tensor.shape) != shape or tensor.dtype != ttnn.uint32 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise RuntimeError(f"{label} must be UINT32 ROW_MAJOR {list(shape)}, got {tensor_metadata(tensor)}")
    mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)


@dataclass(frozen=True)
class Qwen38TTNNDevicePosition:
    """Device-resident absolute position ``P`` of the token a generic decode body decodes.

    ``scalar`` is the only source of the position inside the captured body: it
    is read at the start, every derived quantity comes from exact UINT32 device
    ops, and :meth:`advance` increments it in place as the body's last op so a
    replay alone moves to the next position.  The host writes it only through
    :meth:`reset`.  All three tensors keep their addresses for the life of the
    object.
    """

    scalar: Any
    ones_row: Any
    block_start_mask_row: Any
    mesh_device: Any = field(repr=False, compare=False)
    mesh_contract: Qwen38MeshContract = field(repr=False, compare=False)

    @classmethod
    def allocate(cls, mesh_device, mesh_contract: Qwen38MeshContract, *, position: int = 0) -> Qwen38TTNNDevicePosition:
        mesh_contract.validate_mesh(mesh_device)
        uploaded: list[Any] = []
        try:
            for shape, value, label in (
                (POSITION_SCALAR_SHAPE, position, "device position"),
                (POSITION_INDEX_ROW_SHAPE, 1, "device position ones row"),
                (POSITION_INDEX_ROW_SHAPE, BLOCK_START_LANE_MASK, "device position block-start mask row"),
            ):
                tensor = ttnn.from_torch(
                    _host_uint32(shape, value, label=label),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
                )
                uploaded.append(tensor)
                _require_uint32_row(tensor, shape, mesh_contract, label=label)
        except BaseException:
            for tensor in uploaded:
                ttnn.deallocate(tensor)
            raise
        return cls(uploaded[0], uploaded[1], uploaded[2], mesh_device, mesh_contract)

    def reset(self, position: int) -> None:
        """Host write of ``P`` (outside any trace); the only host path into ``scalar``."""

        host = ttnn.from_torch(
            _host_uint32(POSITION_SCALAR_SHAPE, position, label="device position"),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, self.scalar)

    def advance(self) -> None:
        """In-trace ``P += 1``; must be the last op of the model body."""

        key = _tensor_key(self.scalar)
        advanced = ttnn.add(self.scalar, 1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        copied = ttnn.copy(advanced, self.scalar)
        if _tensor_key(self.scalar) != key or (copied is not None and _tensor_key(copied) != key):
            raise RuntimeError("device position advance did not write the resident scalar in place")
        ttnn.deallocate(advanced)

    def advance_by(self, count: int) -> None:
        """In-trace ``P += count`` (the prefill chunk's ``CHUNK_ROWS``); must be the last op of the chunk body."""

        if isinstance(count, bool) or type(count) is not int or count <= 0:
            raise ValueError(f"device position advance needs a positive int, got {count!r}")
        key = _tensor_key(self.scalar)
        advanced = ttnn.add(self.scalar, count, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        copied = ttnn.copy(advanced, self.scalar)
        if _tensor_key(self.scalar) != key or (copied is not None and _tensor_key(copied) != key):
            raise RuntimeError("device position advance did not write the resident scalar in place")
        ttnn.deallocate(advanced)

    def index_row(self):
        """``[1,1,1,32]`` UINT32 row with every lane ``= P`` (an embedding index row after reshape)."""

        row = ttnn.multiply(self.ones_row, self.scalar, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _require_uint32_row(row, POSITION_INDEX_ROW_SHAPE, self.mesh_contract, label="device position index row")
        return row

    def block_start_index_row(self, index_row):
        """``[1,1,1,32]`` UINT32 row with every lane ``= P & ~3`` (the current index block's first position)."""

        row = ttnn.bitwise_and(index_row, self.block_start_mask_row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _require_uint32_row(row, POSITION_INDEX_ROW_SHAPE, self.mesh_contract, label="device position block-start row")
        return row

    def read(self) -> int:
        """Diagnostic readback of ``P`` from coordinate 0 (outside any trace)."""

        return int(ttnn.to_torch(ttnn.get_device_tensors(self.scalar)[0]).reshape(-1)[0].item())

    def deallocate(self) -> None:
        for tensor in (self.scalar, self.ones_row, self.block_start_mask_row):
            ttnn.deallocate(tensor)
