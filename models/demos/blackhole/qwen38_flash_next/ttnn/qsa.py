# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Correctness-first TP4 Qwen Sparse Attention decode.

This module is deliberately decode-only.  It implements the exact text-model
QSA geometry from the pinned ``Qwen4Exp`` checkpoint while keeping every
placement-changing operation explicit:

* 24 query heads are sharded six per device;
* the two K/V heads are expanded as ``[kv0, kv0, kv1, kv1]`` and then sharded,
  which is pair-grouping rather than four-way replication;
* the four 128-wide index queries are sharded one per device, scored with
  :func:`ttnn.experimental.indexer_score_dsa`, and genuinely summed over TP;
* four-token compressed blocks are expanded back to token indices and the
  incomplete causal tail is appended without being scored;
* the sparse value kernel consumes ``q=[zeros(256)|Q(256)]`` and
  ``kv=[V(256)|K(256)]`` with ``K_DIM=512`` and ``v_dim=256``.

``sparse_sdpa`` requires a row-major cache, whereas the generic per-token cache
writer is tile-only.  Eager bring-up therefore owns a 32-token row-major staging
tile and rewrites that one tile with the existing row-major-capable
``update_padded_kv_cache`` operation on mesh axis 0 (whose extent is exactly
one).  This is intentionally a correctness path, not the final traced decode
path.  It never converts or copies the full cache.

State objects are functional metadata views over shared append-only caches.
Old views remain usable for speculative rollback: bytes past ``next_position``
and ``compressed_blocks`` are ignored and deterministically overwritten on
replay.  Callers which retain an input state must pass ``retain_input_state``;
ordinary decode lets this module release replaced staging tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_prefill
from models.demos.blackhole.qwen38_flash_next.checkpoint import INDEX_SHA256, Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    CHUNK_ROWS,
    MESH_SHAPE,
    Qwen38MeshContract,
    TensorPlacement,
    replicate_tensor_2d_mesh_mapper,
    tensor_metadata,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.decode_matmul import (
    dram_sharded_matmul_configs,
    dram_sharded_weight_memory_config,
)

TP_SIZE = 4
TP_AXIS = 1
STAGING_AXIS = 0  # the admitted mesh is 1x4, so this axis has extent one

HIDDEN_SIZE = 2560
QUERY_HEADS = 24
QUERY_HEADS_PER_DEVICE = QUERY_HEADS // TP_SIZE
KV_HEADS = 2
EXPANDED_KV_HEADS = TP_SIZE
HEAD_DIM = 256
ROPE_DIM = 64
QUERY_WIDTH = QUERY_HEADS * HEAD_DIM
LOCAL_QUERY_WIDTH = QUERY_WIDTH // TP_SIZE

INDEX_QUERY_HEADS = 4
INDEX_QUERY_HEADS_PER_DEVICE = 1
INDEX_KV_HEADS = 1
INDEX_HEAD_DIM = 128
COMPRESS_RATIO = 4
TOKEN_BUDGET = 2048
BLOCK_TOPK = TOKEN_BUDGET // COMPRESS_RATIO

MAX_CONTEXT = 262144
MAX_COMPRESSED_BLOCKS = MAX_CONTEXT // COMPRESS_RATIO
CACHE_WRITE_ROWS = 32
QSA_CACHE_CAPACITY_ALIGNMENT = COMPRESS_RATIO * CACHE_WRITE_ROWS
MIN_QSA_CACHE_CAPACITY = max(QSA_CACHE_CAPACITY_ALIGNMENT, BLOCK_TOPK * COMPRESS_RATIO)
MAX_SPECULATIVE_STEPS = 4
MASKED_INDEX = 0xFFFFFFFF

# Regular QSA needs TOKEN_BUDGET plus at most COMPRESS_RATIO-1 tail tokens.
# Reusing a frozen MTP selection can extend that tail by four draft positions.
MAX_REUSE_TAIL = COMPRESS_RATIO - 1 + MAX_SPECULATIVE_STEPS
MAX_SELECTED_TOKENS = TOKEN_BUDGET + MAX_REUSE_TAIL
SPARSE_INDEX_CAPACITY = math.ceil(MAX_SELECTED_TOKENS / ttnn.TILE_SIZE) * ttnn.TILE_SIZE

# Position-generic decode: every position-dependent quantity is derived on
# device from a UINT32 position scalar, so one captured graph serves every
# position.  The query is row 0 of its own tile (a buffer view) and the fixed
# indexer window is the tile just past the last resident block, so every
# resident column is unmasked for that row; blocks at or past the
# complete-block count are hidden by adding the most negative finite BF16
# value to their scores.
INDEXER_MASK_VALUE = -3.3895313892515355e38
KV_ROW_MASK = CACHE_WRITE_ROWS - 1
KV_BLOCK_START_MASK = 0xFFFFFFFF ^ KV_ROW_MASK
ALL_ONES_U32 = 0xFFFFFFFF


def validate_qsa_cache_capacity(allocated_context: int) -> int:
    """Validate physical cache capacity without changing model semantics."""

    if isinstance(allocated_context, bool) or not isinstance(allocated_context, int):
        raise TypeError(f"QSA allocated context must be an integer, got {allocated_context!r}")
    if not MIN_QSA_CACHE_CAPACITY <= allocated_context <= MAX_CONTEXT:
        raise ValueError(
            f"QSA allocated context must be in [{MIN_QSA_CACHE_CAPACITY}, {MAX_CONTEXT}], " f"got {allocated_context}"
        )
    if allocated_context % QSA_CACHE_CAPACITY_ALIGNMENT:
        raise ValueError(
            f"QSA allocated context must be aligned to {QSA_CACHE_CAPACITY_ALIGNMENT} tokens, "
            f"got {allocated_context}"
        )
    return allocated_context


# Each expansion gather reads half the budget: the pinned runtime's ROW_MAJOR
# gather is exact only through its 1920-element single-core factory (the
# multi-core one past that width misplaces every slice not 64-byte aligned),
# and 4 KiB halves keep the joining concat on its aligned single-launch path.
EXPANSION_GATHER_WIDTH = TOKEN_BUDGET // 2


def qsa_row_constants() -> dict[str, torch.Tensor]:
    """Host values of the module's replicated UINT32 ROW_MAJOR rows, ``[1, 1, 1, width]`` int64.

    ``rep_index_lo`` and ``rep_index_hi`` gather each of the ``BLOCK_TOPK``
    block starts four times (``[0, 0, 0, 0, 1, 1, 1, 1, ...]`` in two halves),
    the two-gather form of the four-way last-dim ``repeat_interleave``;
    ``block_offsets`` then adds the token offset inside each block.
    ``sentinel_pad`` completes the 2048 expanded slots to the sparse row
    width.  The two step rows are twice that width: a window of
    ``SPARSE_INDEX_CAPACITY`` slots starting ``count`` slots before the
    middle holds ``count`` leading entries of the first half and then the
    second half, so one slice yields the mask row for any count.
    ``slot_zero`` is the metadata slot the generic body hands
    ``update_padded_kv_cache``.
    """

    ones = torch.full((SPARSE_INDEX_CAPACITY,), MASKED_INDEX)
    zeros = torch.zeros(SPARSE_INDEX_CAPACITY, dtype=torch.int64)
    rep_index = torch.arange(BLOCK_TOPK).repeat_interleave(COMPRESS_RATIO)
    rows = {
        "slot_zero": torch.zeros(1, dtype=torch.int64),
        "block_offsets": torch.arange(COMPRESS_RATIO).repeat(BLOCK_TOPK),
        "rep_index_lo": rep_index[:EXPANSION_GATHER_WIDTH],
        "rep_index_hi": rep_index[EXPANSION_GATHER_WIDTH:],
        "sentinel_pad": torch.full((SPARSE_INDEX_CAPACITY - TOKEN_BUDGET,), MASKED_INDEX),
        "arange_row": torch.arange(SPARSE_INDEX_CAPACITY),
        "keep_step": torch.cat([ones, zeros]),
        "sentinel_step": torch.cat([zeros, ones]),
    }
    return {name: values.to(torch.int64).reshape(1, 1, 1, -1) for name, values in rows.items()}


QSA_ROW_WIDTHS = {name: int(values.shape[-1]) for name, values in qsa_row_constants().items()}
# Eager per-token rows kept per mesh before the oldest is released.
SHARED_ROW_CACHE = 8
# Regime split: while the budget covers every complete block, scoring only
# orders the row, so the natural-order row could replace it.  Off: the order
# changes the sparse_sdpa accumulation and the bf16 output moves well past
# one ulp on this runtime (see the indexer-diet static test).  The path stays
# for an A/B under a later runtime; the constructor's regime_split overrides
# the default per layer.
QSA_INDEXER_REGIME_SPLIT = False


def qsa_natural_row_regime(complete_blocks: int, *, regime_split: bool = QSA_INDEXER_REGIME_SPLIT) -> bool:
    """True when the row is every causal token in natural order and no block is scored.

    Without a complete block that is the only row.  With the split on, it also
    covers every position whose complete blocks all fit the budget
    (``complete_blocks <= BLOCK_TOPK``, contexts up to 2050 tokens): the CPU
    oracle selects ``topk(min(BLOCK_TOPK, complete_blocks))`` of them, i.e.
    all of them, so the selected set is unchanged and only the order inside
    the row differs.
    """

    if isinstance(complete_blocks, bool) or not isinstance(complete_blocks, int) or complete_blocks < 0:
        raise ValueError(f"complete block count must be a non-negative integer, got {complete_blocks!r}")
    return complete_blocks == 0 or (regime_split and complete_blocks <= BLOCK_TOPK)


def emulate_block_expansion(starts: torch.Tensor) -> torch.Tensor:
    """Host form of the device expansion of ``BLOCK_TOPK`` block starts, ``[TOKEN_BUDGET]`` int64.

    Two gathers of the start row with the half ``rep_index`` rows, their
    concat, and the UINT32 add of ``block_offsets``.
    """

    rows = {name: values.reshape(-1) for name, values in qsa_row_constants().items()}
    starts = starts.reshape(-1).to(torch.int64)
    if starts.numel() != BLOCK_TOPK:
        raise ValueError(f"block expansion takes {BLOCK_TOPK} starts, got {starts.numel()}")
    repeated = torch.cat([starts[rows["rep_index_lo"]], starts[rows["rep_index_hi"]]])
    return (repeated + rows["block_offsets"]) & MASKED_INDEX


def emulate_step_window(step: torch.Tensor, count: int) -> torch.Tensor:
    """Host form of the one-slice mask row: ``count`` leading entries of the step's first half."""

    if not 0 <= count <= SPARSE_INDEX_CAPACITY:
        raise ValueError(f"step window count must be in [0, {SPARSE_INDEX_CAPACITY}], got {count}")
    return step[..., SPARSE_INDEX_CAPACITY - count : 2 * SPARSE_INDEX_CAPACITY - count]


def emulate_sparse_row(
    expanded: torch.Tensor | None, *, complete_token_count: int, tail_start: int, context_length: int
) -> torch.Tensor:
    """Host form of the materialized sparse row, ``[SPARSE_INDEX_CAPACITY]`` int64.

    ``expanded`` is the ``TOKEN_BUDGET``-wide expansion of the top-k slots
    (``None`` selects every causal token in natural order).  The device
    builds the same bytes from the template ``[expanded | sentinel_pad]``,
    the keep window and the tail-and-sentinel row.
    """

    rows = {name: values.reshape(-1) for name, values in qsa_row_constants().items()}
    if expanded is None:
        if tail_start or complete_token_count:
            raise ValueError("a natural-order row has no scored blocks")
        return rows["arange_row"] | emulate_step_window(rows["sentinel_step"], context_length)
    valid_count = complete_token_count + context_length - tail_start
    template = torch.cat([expanded.reshape(-1).to(torch.int64), rows["sentinel_pad"]])
    keep = emulate_step_window(rows["keep_step"], complete_token_count)
    tail_ids = rows["arange_row"] + (tail_start - complete_token_count)
    tail = tail_ids & emulate_step_window(rows["sentinel_step"], complete_token_count)
    return (template & keep) | tail | emulate_step_window(rows["sentinel_step"], valid_count)


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(item) for item in tensor.shape)


def _tensor_key(tensor) -> tuple[str, int]:
    tensor_id = getattr(tensor, "tensor_id", None)
    if callable(tensor_id):
        tensor_id = tensor_id()
    return ("ttnn", int(tensor_id)) if tensor_id is not None else ("python", id(tensor))


def _deallocate(*tensors) -> None:
    seen: set[tuple[str, int]] = set()
    for tensor in tensors:
        if tensor is None:
            continue
        key = _tensor_key(tensor)
        if key in seen:
            continue
        seen.add(key)
        ttnn.deallocate(tensor)


def _cache_dir(
    root: str | Path,
    checkpoint: Qwen38Checkpoint,
    mesh_contract: Qwen38MeshContract,
    tt_metal_sha: str,
    source_kind: str,
    layer_index: int,
) -> Path:
    if layer_index < 0:
        raise ValueError("QSA layer index must be nonnegative")
    if source_kind not in ("backbone", "mtp"):
        raise ValueError(f"unknown QSA checkpoint source {source_kind!r}")
    if len(tt_metal_sha) != 40 or any(character not in "0123456789abcdef" for character in tt_metal_sha):
        raise ValueError(f"tt_metal_sha must be a lowercase 40-hex revision, got {tt_metal_sha!r}")
    physical = "-".join(str(value) for value in mesh_contract.physical_ids)
    path = (
        Path(root).resolve()
        / "qsa"
        / f"index-{INDEX_SHA256}"
        / f"config-{checkpoint.config.config_sha256}"
        / f"tt-metal-{tt_metal_sha}"
        / f"mesh-1x4-physical-{physical}"
        / source_kind
        / f"layer-{layer_index:02d}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _require_shape(tensor, expected: tuple[int, ...], label: str) -> None:
    if _shape(tensor) != expected:
        raise RuntimeError(f"{label} shape must be {list(expected)}, got {tensor_metadata(tensor)}")


def _upload_uint32(mesh_device, mesh_contract: Qwen38MeshContract, values: torch.Tensor, *, layout) -> Any:
    tensor = ttnn.from_torch(
        values.to(torch.int64).to(torch.uint32),
        dtype=ttnn.uint32,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
    )
    mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)
    return tensor


def _retag_tensor(tensor, *, reference, shard_dim: int | None) -> None:
    """Update mesh metadata after a local-only reshape/transpose/typecast.

    No bytes move here.  Every call follows an operation whose per-coordinate
    values and local shape were checked by the caller.  ``shard_dim=None`` means
    full-shape per-device values (replicated shape metadata); otherwise the
    second axis of the exact 1x4 mesh shards ``shard_dim``.
    """

    topology = reference.tensor_topology()
    placements = [ttnn.PlacementReplicate()]
    placements.append(ttnn.PlacementReplicate() if shard_dim is None else ttnn.PlacementShard(shard_dim))
    tensor.update_tensor_topology(
        ttnn.TensorTopology(topology.distribution_shape(), placements, topology.mesh_coords())
    )


def _canonicalize_flat_replicated_topology(tensor, mesh_contract: Qwen38MeshContract) -> None:
    """Retag the legacy flat mesh returned by device-side zero allocation.

    ``ttnn.zeros`` creates the requested full local value on every worker but
    currently reports the four workers as one distribution axis while retaining
    their two-dimensional mesh coordinates.  The already opened model mesh is
    explicitly ``(1, 4)``.  Accept only that exact native representation, then
    change metadata to the canonical two-axis topology without moving bytes.
    """

    topology = tensor.tensor_topology()
    raw_coordinates = tuple(topology.mesh_coords())
    distribution = tuple(int(value) for value in topology.distribution_shape())
    coordinates = tuple(tuple(int(value) for value in coordinate) for coordinate in raw_coordinates)
    placements = tuple(type(value).__name__ for value in topology.placements())
    expected_coordinates = tuple((0, column) for column in range(TP_SIZE))
    if distribution != (TP_SIZE,) or coordinates != expected_coordinates or placements != ("PlacementReplicate",):
        raise RuntimeError(
            "device zero allocation topology is not the exact flat replicated TP4 form: "
            f"distribution={distribution} coordinates={coordinates} placements={placements}"
        )
    tensor.update_tensor_topology(
        ttnn.TensorTopology(
            ttnn.MeshShape(*mesh_contract.mesh_shape),
            [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()],
            raw_coordinates,
        )
    )
    mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)


def _expanded_pair_kv(weight: torch.Tensor) -> torch.Tensor:
    """Return checkpoint rows as ``[kv0, kv0, kv1, kv1]``.

    The resulting four-head tensor can be sharded over TP.  Devices 0/1 then
    own independent copies of KV head 0 while devices 2/3 own head 1.  This is
    intentional pair grouping and is represented as a sharded topology.
    """

    if tuple(weight.shape) != (KV_HEADS * HEAD_DIM, HIDDEN_SIZE):
        raise ValueError(f"QSA K/V checkpoint weight has unexpected shape {tuple(weight.shape)}")
    heads = weight.reshape(KV_HEADS, HEAD_DIM, HIDDEN_SIZE)
    return torch.stack((heads[0], heads[0], heads[1], heads[1]), dim=0).reshape(
        EXPANDED_KV_HEADS * HEAD_DIM, HIDDEN_SIZE
    )


@dataclass(frozen=True)
class Qwen38QSASelectionGeometry:
    context_length: int
    complete_blocks: int
    selected_blocks: int
    complete_token_count: int
    tail_start: int
    tail_count: int


def qsa_selection_geometry(context_length: int) -> Qwen38QSASelectionGeometry:
    """Pure geometry used by both runtime guards and no-device tests."""

    if not 1 <= context_length <= MAX_CONTEXT:
        raise ValueError(f"QSA context length must be in [1, {MAX_CONTEXT}], got {context_length}")
    complete_blocks, tail_count = divmod(context_length, COMPRESS_RATIO)
    selected_blocks = min(BLOCK_TOPK, complete_blocks)
    return Qwen38QSASelectionGeometry(
        context_length=context_length,
        complete_blocks=complete_blocks,
        selected_blocks=selected_blocks,
        complete_token_count=selected_blocks * COMPRESS_RATIO,
        tail_start=complete_blocks * COMPRESS_RATIO,
        tail_count=tail_count,
    )


@dataclass(frozen=True)
class Qwen38TTNNQSAWeights:
    """Static BF16 QSA tensors with exact TP4 placement."""

    source_kind: str
    layer_index: int
    qg: Any
    k_pair_grouped: Any
    v_pair_grouped: Any
    out: Any
    q_norm: Any
    k_norm: Any
    index_q: Any
    index_k: Any
    index_q_norm: Any
    index_k_norm: Any

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Qwen38Checkpoint,
        placement: Qwen38Placement,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        cache_root: str | Path,
        *,
        layer_index: int,
        tt_metal_sha: str,
        _mtp: bool = False,
    ) -> "Qwen38TTNNQSAWeights":
        mesh_contract.validate_mesh(mesh_device)
        config = checkpoint.config
        if placement.config.config_sha256 != config.config_sha256:
            raise ValueError("placement and checkpoint configurations differ")
        if tuple(placement.mesh_shape) != MESH_SHAPE:
            raise ValueError(f"placement mesh {placement.mesh_shape} is not the pinned {MESH_SHAPE}")
        if tuple(placement.physical_ids) != mesh_contract.physical_ids:
            raise ValueError(
                f"placement physical order {placement.physical_ids} differs from admitted "
                f"{mesh_contract.physical_ids}"
            )
        if _mtp:
            if not 0 <= layer_index < config.mtp_layers:
                raise ValueError(f"MTP QSA layer index is outside [0, {config.mtp_layers}): {layer_index}")
            source_kind = "mtp"
            prefix = f"mtp.layers.{layer_index}.self_attn."
        else:
            if not 0 <= layer_index < config.num_hidden_layers:
                raise ValueError(f"QSA layer index is outside [0, {config.num_hidden_layers}): {layer_index}")
            if config.layer_types[layer_index] != "full_attention":
                raise ValueError(f"layer {layer_index} is {config.layer_types[layer_index]!r}, not full_attention")
            source_kind = "backbone"
            prefix = f"model.language_model.layers.{layer_index}.self_attn."
        pinned = (
            config.hidden_size,
            config.qsa_query_heads,
            config.qsa_kv_heads,
            config.qsa_head_dim,
            config.qsa_rope_dim,
            config.index_query_heads,
            config.index_kv_heads,
            config.index_head_dim,
            config.index_budget,
            config.index_compress_ratio,
            config.max_position_embeddings,
        )
        expected = (
            HIDDEN_SIZE,
            QUERY_HEADS,
            KV_HEADS,
            HEAD_DIM,
            ROPE_DIM,
            INDEX_QUERY_HEADS,
            INDEX_KV_HEADS,
            INDEX_HEAD_DIM,
            TOKEN_BUDGET,
            COMPRESS_RATIO,
            MAX_CONTEXT,
        )
        if pinned != expected:
            raise ValueError(f"checkpoint QSA geometry {pinned} does not match pinned target {expected}")
        if config.rms_norm_eps != 1e-6:
            raise ValueError(f"pinned QSA RMS epsilon must be 1e-6, got {config.rms_norm_eps}")

        def load(name: str, expected_shape: tuple[int, ...]) -> torch.Tensor:
            value = checkpoint.tensor(prefix + name)
            if tuple(value.shape) != expected_shape or value.dtype != torch.bfloat16:
                raise ValueError(
                    f"{prefix}{name} must be BF16 {expected_shape}, got {value.dtype} {tuple(value.shape)}"
                )
            return value

        qg = load("q_proj.weight", (2 * QUERY_WIDTH, HIDDEN_SIZE))
        k = load("k_proj.weight", (KV_HEADS * HEAD_DIM, HIDDEN_SIZE))
        v = load("v_proj.weight", (KV_HEADS * HEAD_DIM, HIDDEN_SIZE))
        out = load("o_proj.weight", (HIDDEN_SIZE, QUERY_WIDTH))
        q_norm = load("q_norm.weight", (HEAD_DIM,))
        k_norm = load("k_norm.weight", (HEAD_DIM,))
        index_qk = load(
            "indexer.index_qk_proj.weight",
            ((INDEX_QUERY_HEADS + INDEX_KV_HEADS) * INDEX_HEAD_DIM, HIDDEN_SIZE),
        )
        index_q_norm = load("indexer.q_layernorm.weight", (INDEX_HEAD_DIM,))
        index_k_norm = load("indexer.k_layernorm.weight", (INDEX_HEAD_DIM,))

        cache = _cache_dir(
            cache_root,
            checkpoint,
            mesh_contract,
            tt_metal_sha,
            source_kind,
            layer_index,
        )
        output_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3))
        input_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 2))
        replicate_mapper = replicate_tensor_2d_mesh_mapper(mesh_device)

        def upload(value: torch.Tensor, name: str, mapper, memory_config):
            return ttnn.as_tensor(
                value.to(torch.bfloat16).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=memory_config,
                mesh_mapper=mapper,
                cache_file_name=cache / name,
            )

        # Projection weights are DRAM width-sharded for the decode matmul
        # program; the renamed tensorbins deliberately orphan interleaved
        # caches.
        qg_tt = upload(
            qg.transpose(0, 1).reshape(1, 1, HIDDEN_SIZE, 2 * QUERY_WIDTH),
            "qg_dram_sharded",
            output_mapper,
            dram_sharded_weight_memory_config(mesh_device, HIDDEN_SIZE, 2 * LOCAL_QUERY_WIDTH),
        )
        k_tt = upload(
            _expanded_pair_kv(k).transpose(0, 1).reshape(1, 1, HIDDEN_SIZE, TP_SIZE * HEAD_DIM),
            "k_pair_grouped_dram_sharded",
            output_mapper,
            dram_sharded_weight_memory_config(mesh_device, HIDDEN_SIZE, HEAD_DIM),
        )
        v_tt = upload(
            _expanded_pair_kv(v).transpose(0, 1).reshape(1, 1, HIDDEN_SIZE, TP_SIZE * HEAD_DIM),
            "v_pair_grouped_dram_sharded",
            output_mapper,
            dram_sharded_weight_memory_config(mesh_device, HIDDEN_SIZE, HEAD_DIM),
        )
        out_tt = upload(
            out.transpose(0, 1).reshape(1, 1, QUERY_WIDTH, HIDDEN_SIZE),
            "out_dram_sharded",
            input_mapper,
            dram_sharded_weight_memory_config(mesh_device, LOCAL_QUERY_WIDTH, HIDDEN_SIZE),
        )

        index_q = index_qk[: INDEX_QUERY_HEADS * INDEX_HEAD_DIM]
        index_k = index_qk[INDEX_QUERY_HEADS * INDEX_HEAD_DIM :]
        index_q_tt = upload(
            index_q.transpose(0, 1).reshape(1, 1, HIDDEN_SIZE, INDEX_QUERY_HEADS * INDEX_HEAD_DIM),
            "index_q_dram_sharded",
            output_mapper,
            dram_sharded_weight_memory_config(mesh_device, HIDDEN_SIZE, INDEX_QUERY_HEADS_PER_DEVICE * INDEX_HEAD_DIM),
        )
        index_k_tt = upload(
            index_k.transpose(0, 1).reshape(1, 1, HIDDEN_SIZE, INDEX_HEAD_DIM),
            "index_k_dram_sharded",
            replicate_mapper,
            dram_sharded_weight_memory_config(mesh_device, HIDDEN_SIZE, INDEX_HEAD_DIM),
        )

        # Qwen4Exp norms are zero-centred: the checkpoint stores delta-gamma.
        # Norm vectors are elementwise operands, not matmul weights: interleaved DRAM.
        def norm(value: torch.Tensor, name: str):
            return upload((value.float() + 1.0).reshape(1, 1, 1, -1), name, replicate_mapper, ttnn.DRAM_MEMORY_CONFIG)

        q_norm_tt = norm(q_norm, "q_norm_offset")
        k_norm_tt = norm(k_norm, "k_norm_offset")
        index_q_norm_tt = norm(index_q_norm, "index_q_norm_offset")
        index_k_norm_tt = norm(index_k_norm, "index_k_norm_offset")

        mesh_contract.validate_tensor(qg_tt, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        mesh_contract.validate_tensor(k_tt, placement=TensorPlacement.KV_PAIR_GROUPED, shard_dim=3)
        mesh_contract.validate_tensor(v_tt, placement=TensorPlacement.KV_PAIR_GROUPED, shard_dim=3)
        mesh_contract.validate_tensor(out_tt, placement=TensorPlacement.HEAD_SHARDED, shard_dim=2)
        mesh_contract.validate_tensor(index_q_tt, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        for value in (index_k_tt, q_norm_tt, k_norm_tt, index_q_norm_tt, index_k_norm_tt):
            mesh_contract.validate_tensor(value, placement=TensorPlacement.REPLICATED)
        result = cls(
            source_kind=source_kind,
            layer_index=layer_index,
            qg=qg_tt,
            k_pair_grouped=k_tt,
            v_pair_grouped=v_tt,
            out=out_tt,
            q_norm=q_norm_tt,
            k_norm=k_norm_tt,
            index_q=index_q_tt,
            index_k=index_k_tt,
            index_q_norm=index_q_norm_tt,
            index_k_norm=index_k_norm_tt,
        )
        result.validate(mesh_contract)
        return result

    @classmethod
    def from_mtp_checkpoint(
        cls,
        checkpoint: Qwen38Checkpoint,
        placement: Qwen38Placement,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        cache_root: str | Path,
        *,
        mtp_layer_index: int = 0,
        tt_metal_sha: str,
    ) -> "Qwen38TTNNQSAWeights":
        """Load the checkpoint's real MTP QSA tensors with the same TP contract."""

        return cls.from_checkpoint(
            checkpoint,
            placement,
            mesh_device,
            mesh_contract,
            cache_root,
            layer_index=mtp_layer_index,
            tt_metal_sha=tt_metal_sha,
            _mtp=True,
        )

    def validate(self, mesh_contract: Qwen38MeshContract) -> None:
        if self.source_kind not in ("backbone", "mtp"):
            raise RuntimeError(f"unknown QSA weight source {self.source_kind!r}")
        expected = (
            ("qg", self.qg, TensorPlacement.HEAD_SHARDED, 3, (1, 1, HIDDEN_SIZE, 2 * LOCAL_QUERY_WIDTH)),
            ("k", self.k_pair_grouped, TensorPlacement.KV_PAIR_GROUPED, 3, (1, 1, HIDDEN_SIZE, HEAD_DIM)),
            ("v", self.v_pair_grouped, TensorPlacement.KV_PAIR_GROUPED, 3, (1, 1, HIDDEN_SIZE, HEAD_DIM)),
            ("out", self.out, TensorPlacement.HEAD_SHARDED, 2, (1, 1, LOCAL_QUERY_WIDTH, HIDDEN_SIZE)),
            ("index_q", self.index_q, TensorPlacement.HEAD_SHARDED, 3, (1, 1, HIDDEN_SIZE, INDEX_HEAD_DIM)),
        )
        for name, tensor, placement, shard_dim, shape in expected:
            mesh_contract.validate_tensor(tensor, placement=placement, shard_dim=shard_dim)
            _require_shape(tensor, shape, f"QSA {name} weight")
            if tensor.dtype != ttnn.bfloat16:
                raise RuntimeError(f"QSA {name} weight must be BF16, got {tensor.dtype}")
        for name, tensor, width in (
            ("q_norm", self.q_norm, HEAD_DIM),
            ("k_norm", self.k_norm, HEAD_DIM),
            ("index_k", self.index_k, INDEX_HEAD_DIM),
            ("index_q_norm", self.index_q_norm, INDEX_HEAD_DIM),
            ("index_k_norm", self.index_k_norm, INDEX_HEAD_DIM),
        ):
            mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)
            expected_shape = (1, 1, HIDDEN_SIZE, width) if name == "index_k" else (1, 1, 1, width)
            _require_shape(tensor, expected_shape, f"QSA {name} weight")
            if tensor.dtype != ttnn.bfloat16:
                raise RuntimeError(f"QSA {name} weight must be BF16, got {tensor.dtype}")

    def deallocate(self) -> None:
        """Release all resident weights owned by this container."""

        _deallocate(
            self.qg,
            self.k_pair_grouped,
            self.v_pair_grouped,
            self.out,
            self.q_norm,
            self.k_norm,
            self.index_q,
            self.index_k,
            self.index_q_norm,
            self.index_k_norm,
        )


@dataclass(frozen=True)
class Qwen38TTNNQSASelection:
    """Selected complete blocks plus one materialized sparse-kernel row.

    ``complete_indices`` is the ``TOKEN_BUDGET``-wide expansion of the top-k
    block slots in score order; the first ``complete_token_count`` entries are
    the selected complete-block tokens and the rest are masked out of every
    row built from it.  It deliberately excludes the incomplete tail so an MTP
    caller can reuse the frozen block choice and append all in-flight positions
    from ``tail_start``.  ``None`` means no block was scored: the row holds
    every causal token in natural order and ``tail_start`` is 0.

    ``sparse_indices`` is released with the view that produced it unless
    ``owns_sparse_indices`` is false, in which case it is a per-token row the
    module shares between its layers.
    """

    layer_index: int
    epoch: int
    source_view_id: int
    source_position: int
    tail_start: int
    complete_indices: Any | None
    complete_token_count: int
    owns_complete_indices: bool
    sparse_indices: Any
    valid_token_count: int
    owns_sparse_indices: bool = True


@dataclass
class _SharedRow:
    """A per-token UINT32 row built once per mesh and read by every QSA layer at that position."""

    mesh_device: Any
    tensor: Any
    retained: bool  # owned by a module's trace-retained inputs; released with them, never evicted


@dataclass(frozen=True)
class Qwen38TTNNQSAState:
    """Append-only QSA decode state with rollback-safe length metadata."""

    layer_index: int
    epoch: int
    view_id: int
    next_position: int
    compressed_blocks: int
    raw_tail_count: int
    raw_index_tail: Any | None
    packed_kv_cache: Any
    compressed_index_cache: Any
    kv_staging: Any
    kv_staging_owned: bool
    last_selection: Qwen38TTNNQSASelection | None = None


@dataclass(frozen=True)
class Qwen38TTNNQSAResult:
    hidden_sharded: Any
    state: Qwen38TTNNQSAState
    selection: Qwen38TTNNQSASelection


@dataclass(frozen=True)
class Qwen38TTNNQSAPositionConstants:
    """Replicated UINT32 constants shared by every QSA layer of one model.

    ``arange32_col`` is TILE so the row one-hots compare against a TILE view
    of the position and stay TILE for the in-place staging/ring selects; the
    row constants are ROW_MAJOR like the UINT32 index rows they combine with.
    """

    allocated_compressed_blocks: int
    arange32_col: Any
    arange_blocks: Any
    arange_row: Any
    all_ones: Any
    high27_mask: Any

    @classmethod
    def build(
        cls,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        allocated_compressed_blocks: int,
    ) -> "Qwen38TTNNQSAPositionConstants":
        mesh_contract.validate_mesh(mesh_device)
        blocks = validate_qsa_cache_capacity(allocated_compressed_blocks * COMPRESS_RATIO) // COMPRESS_RATIO

        def upload(values: torch.Tensor, layout=ttnn.ROW_MAJOR_LAYOUT):
            return _upload_uint32(mesh_device, mesh_contract, values, layout=layout)

        return cls(
            allocated_compressed_blocks=blocks,
            arange32_col=upload(torch.arange(ttnn.TILE_SIZE).reshape(1, 1, ttnn.TILE_SIZE, 1), ttnn.TILE_LAYOUT),
            arange_blocks=upload(torch.arange(blocks).reshape(1, 1, 1, blocks)),
            arange_row=upload(torch.arange(SPARSE_INDEX_CAPACITY).reshape(1, 1, 1, SPARSE_INDEX_CAPACITY)),
            all_ones=upload(torch.full((1, 1, 1, 1), ALL_ONES_U32)),
            high27_mask=upload(torch.full((1, 1, 1, 1), KV_BLOCK_START_MASK)),
        )

    def deallocate(self) -> None:
        _deallocate(self.arange32_col, self.arange_blocks, self.arange_row, self.all_ones, self.high27_mask)


@dataclass(frozen=True)
class Qwen38TTNNQSAPositionInputs:
    """Per-token device tensors derived from the position; shared by all QSA layers.

    Integer rows are UINT32 ROW_MAJOR, the row one-hots BF16 TILE ``[1,1,32,1]``
    (exact 0.0/1.0 multiply masks), ``block_index_i32`` INT32 ``[1]`` for
    ``paged_update_cache``.
    """

    kv_block_start: Any
    kv_row_hit: Any
    kv_row_keep: Any
    ring_hit: Any
    ring_keep: Any
    block_index_i32: Any
    indexer_neg_mask: Any
    row_keep_bits: Any
    row_fill: Any

    def deallocate(self) -> None:
        _deallocate(
            self.kv_block_start,
            self.kv_row_hit,
            self.kv_row_keep,
            self.ring_hit,
            self.ring_keep,
            self.block_index_i32,
            self.indexer_neg_mask,
            self.row_keep_bits,
            self.row_fill,
        )


def derive_qsa_position_inputs(
    position_scalar, constants: Qwen38TTNNQSAPositionConstants
) -> Qwen38TTNNQSAPositionInputs:
    """Derive table rows 4-17 of the position spec with exact UINT32 device ops.

    ``position_scalar`` is the replicated UINT32 ROW_MAJOR ``[1,1,1,1]`` counter
    ``P``.  No value is read back to the host and no integer passes through a
    float stage: the only casts are the 0/1 masks to BF16 and the block index
    to INT32.
    """

    _require_shape(position_scalar, (1, 1, 1, 1), "QSA position scalar")
    if position_scalar.dtype != ttnn.uint32 or position_scalar.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise RuntimeError(f"QSA position scalar must be UINT32 ROW_MAJOR, got {tensor_metadata(position_scalar)}")
    dram = ttnn.DRAM_MEMORY_CONFIG
    u32 = ttnn.uint32

    def one_hot(position_tiled, modulus_mask: int):
        # eq() writes 0/1 UINT32; the typecast is exact and rsub(x, 1.0) is
        # exact on {0.0, 1.0}.  Padded tile columns are never read: every
        # consumer broadcasts column 0.
        remainder = ttnn.bitwise_and(position_tiled, modulus_mask, memory_config=dram)
        hit_bits = ttnn.eq(constants.arange32_col, remainder, dtype=u32, memory_config=dram)
        hit = ttnn.typecast(hit_bits, ttnn.bfloat16, memory_config=dram)
        keep = ttnn.rsub(hit, 1.0, memory_config=dram)
        _deallocate(remainder, hit_bits)
        return hit, keep

    position_tiled = ttnn.to_layout(position_scalar, ttnn.TILE_LAYOUT, memory_config=dram)
    kv_row_hit, kv_row_keep = one_hot(position_tiled, KV_ROW_MASK)
    ring_hit, ring_keep = one_hot(position_tiled, COMPRESS_RATIO - 1)
    _deallocate(position_tiled)

    kv_block_start = ttnn.bitwise_and(position_scalar, constants.high27_mask, memory_config=dram)
    block_index = ttnn.bitwise_right_shift(position_scalar, 2, memory_config=dram)
    block_index_i32 = ttnn.reshape(ttnn.typecast(block_index, ttnn.int32, memory_config=dram), (1,))
    _deallocate(block_index)

    # context_length = P + 1; complete_blocks = context_length // 4 (qsa_selection_geometry).
    context_length = ttnn.add(position_scalar, 1, memory_config=dram)
    complete_blocks = ttnn.bitwise_right_shift(context_length, 2, memory_config=dram)
    valid_bits = ttnn.lt(constants.arange_blocks, complete_blocks, dtype=u32, memory_config=dram)
    valid = ttnn.typecast(valid_bits, ttnn.bfloat16, memory_config=dram)
    invalid = ttnn.rsub(valid, 1.0, memory_config=dram)
    indexer_neg_mask = ttnn.multiply(invalid, INDEXER_MASK_VALUE, memory_config=dram)
    _deallocate(valid_bits, valid, invalid)

    # lo = 4 * min(BLOCK_TOPK, complete_blocks) expanded tokens, then the
    # causal tail [lo, hi) of absolute ids, then sentinels; the tail ids and
    # the sentinels occupy disjoint slots, so one OR'd row carries both.
    selected_blocks = ttnn.minimum(complete_blocks, BLOCK_TOPK, memory_config=dram)
    lo = ttnn.bitwise_left_shift(selected_blocks, 2, memory_config=dram)
    tail_count = ttnn.bitwise_and(context_length, COMPRESS_RATIO - 1, memory_config=dram)
    hi = ttnn.add(lo, tail_count, memory_config=dram)
    before_lo = ttnn.lt(constants.arange_row, lo, dtype=u32, memory_config=dram)
    row_keep_bits = ttnn.multiply(before_lo, constants.all_ones, memory_config=dram)
    skipped_blocks = ttnn.subtract(complete_blocks, selected_blocks, memory_config=dram)
    tail_shift = ttnn.bitwise_left_shift(skipped_blocks, 2, memory_config=dram)
    tail_ids = ttnn.add(constants.arange_row, tail_shift, memory_config=dram)
    from_lo = ttnn.ge(constants.arange_row, lo, dtype=u32, memory_config=dram)
    before_hi = ttnn.lt(constants.arange_row, hi, dtype=u32, memory_config=dram)
    tail_bits = ttnn.multiply(from_lo, before_hi, memory_config=dram)
    row_tail_fill = ttnn.multiply(tail_ids, tail_bits, memory_config=dram)
    from_hi = ttnn.ge(constants.arange_row, hi, dtype=u32, memory_config=dram)
    row_sentinel_bits = ttnn.multiply(from_hi, constants.all_ones, memory_config=dram)
    row_fill = ttnn.bitwise_or(row_tail_fill, row_sentinel_bits, memory_config=dram)
    _deallocate(
        context_length,
        complete_blocks,
        selected_blocks,
        lo,
        tail_count,
        hi,
        before_lo,
        skipped_blocks,
        tail_shift,
        tail_ids,
        from_lo,
        before_hi,
        tail_bits,
        row_tail_fill,
        from_hi,
        row_sentinel_bits,
    )
    return Qwen38TTNNQSAPositionInputs(
        kv_block_start=kv_block_start,
        kv_row_hit=kv_row_hit,
        kv_row_keep=kv_row_keep,
        ring_hit=ring_hit,
        ring_keep=ring_keep,
        block_index_i32=block_index_i32,
        indexer_neg_mask=indexer_neg_mask,
        row_keep_bits=row_keep_bits,
        row_fill=row_fill,
    )


def emulate_qsa_position_inputs(position: int, *, allocated_compressed_blocks: int) -> dict[str, torch.Tensor]:
    """Torch reference of :func:`derive_qsa_position_inputs` (same names, shapes, dtypes).

    UINT32 values are ``torch.int64`` in ``[0, 2**32)``; the BF16 one-hot
    masks use the device's ``1.0 - hit`` arithmetic and match bit for bit.
    ``indexer_neg_mask`` is a select, not the device's ``(1.0 - valid) *
    INDEXER_MASK_VALUE``: on this runtime a BF16 multiply of ``0.0`` by a
    negative scalar returns ``+0.0`` (binary_ng, calculate_sfpu_binary_mul in
    ckernel_sfpu_binary.h forces ``result = 0.0f`` whenever an input is zero,
    matching the FPU), where torch would give ``-0.0``; observed bitwise on
    silicon (the lab host, 2026-09-02). Visible blocks are therefore ``+0.0``.
    """

    if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position < MAX_CONTEXT:
        raise ValueError(f"QSA position must be an integer in [0, {MAX_CONTEXT}), got {position!r}")
    blocks = validate_qsa_cache_capacity(allocated_compressed_blocks * COMPRESS_RATIO) // COMPRESS_RATIO
    geometry = qsa_selection_geometry(position + 1)
    lo = geometry.complete_token_count
    hi = lo + geometry.tail_count
    one = torch.tensor(1.0, dtype=torch.bfloat16)

    def one_hot(row: int) -> tuple[torch.Tensor, torch.Tensor]:
        hit = (torch.arange(CACHE_WRITE_ROWS) == row).to(torch.bfloat16).reshape(1, 1, CACHE_WRITE_ROWS, 1)
        return hit, one - hit

    kv_row_hit, kv_row_keep = one_hot(position % CACHE_WRITE_ROWS)
    ring_hit, ring_keep = one_hot(position % COMPRESS_RATIO)
    valid = (torch.arange(blocks) < geometry.complete_blocks).to(torch.bfloat16).reshape(1, 1, 1, blocks)
    slots = torch.arange(SPARSE_INDEX_CAPACITY, dtype=torch.int64).reshape(1, 1, 1, SPARSE_INDEX_CAPACITY)
    return {
        "kv_block_start": torch.full((1, 1, 1, 1), position & KV_BLOCK_START_MASK, dtype=torch.int64),
        "kv_row_hit": kv_row_hit,
        "kv_row_keep": kv_row_keep,
        "ring_hit": ring_hit,
        "ring_keep": ring_keep,
        "block_index_i32": torch.tensor([position // COMPRESS_RATIO], dtype=torch.int32),
        "indexer_neg_mask": torch.where(
            valid.bool(), torch.zeros_like(valid), torch.full_like(valid, INDEXER_MASK_VALUE)
        ),
        "row_keep_bits": (slots < lo).to(torch.int64) * ALL_ONES_U32,
        "row_fill": (slots + (geometry.tail_start - lo)) * ((slots >= lo) & (slots < hi)).to(torch.int64)
        | (slots >= hi).to(torch.int64) * ALL_ONES_U32,
    }


# --------------------------------------------------------------------------- prefill chunk (32 rows)
# One chunk holds CHUNK_ROWS consecutive positions P .. P + 31 with P % 32 == 0: eight complete
# compressed blocks, one 32-row KV slab, and per row the selection geometry of position P + j.  Every
# per-row quantity below is the 1-row derivation on a same-shape template whose row j carries j, so
# the only broadcast is the scalar P (the class the decode already uses).

CHUNK_BLOCKS = CHUNK_ROWS // COMPRESS_RATIO


def qsa_chunk_constant_rows(allocated_compressed_blocks: int) -> dict[str, torch.Tensor]:
    """Host images of the chunk constants (UINT32 rows as int64; the two bf16 select tiles as float).

    ``pool_select`` row i (i < 8) holds 0.25 at columns 4i .. 4i+3: ``pool_select @ raw_keys`` is the block
    mean of the chunk's raw index keys (four exact power-of-two products summed in fp32, one bf16 rounding:
    the same value as the decode ring's quarter-scaled sum).  ``row_select[i]`` is the 0/1 tile whose row 0
    picks row i (the exact selection matmul of the GDN rows path), so block i's compressed row becomes the
    one-row input of ``paged_update_cache``.
    """

    blocks = validate_qsa_cache_capacity(allocated_compressed_blocks * COMPRESS_RATIO) // COMPRESS_RATIO
    rows = torch.arange(CHUNK_ROWS, dtype=torch.int64).reshape(1, 1, CHUNK_ROWS, 1)
    lanes = torch.arange(CHUNK_ROWS, dtype=torch.int64)
    block_start_lanes = torch.where(lanes < CHUNK_BLOCKS, lanes * COMPRESS_RATIO, torch.zeros_like(lanes))
    pool_select = torch.zeros(CHUNK_ROWS, CHUNK_ROWS)
    for block in range(CHUNK_BLOCKS):
        pool_select[block, block * COMPRESS_RATIO : (block + 1) * COMPRESS_RATIO] = 1.0 / COMPRESS_RATIO
    row_selects = torch.zeros(CHUNK_BLOCKS, CHUNK_ROWS, CHUNK_ROWS)
    for block in range(CHUNK_BLOCKS):
        row_selects[block, 0, block] = 1.0
    return {
        "arange32_lanes": lanes.reshape(1, 1, 1, CHUNK_ROWS),
        "block_start_lanes": block_start_lanes.reshape(1, 1, 1, CHUNK_ROWS),
        "row_index_blocks": rows.expand(1, 1, CHUNK_ROWS, blocks).contiguous(),
        "arange_blocks_rows": torch.arange(blocks, dtype=torch.int64)
        .reshape(1, 1, 1, blocks)
        .expand(1, 1, CHUNK_ROWS, blocks)
        .contiguous(),
        "row_index_slots": rows.expand(1, 1, CHUNK_ROWS, SPARSE_INDEX_CAPACITY).contiguous(),
        "arange_slots_rows": torch.arange(SPARSE_INDEX_CAPACITY, dtype=torch.int64)
        .reshape(1, 1, 1, SPARSE_INDEX_CAPACITY)
        .expand(1, 1, CHUNK_ROWS, SPARSE_INDEX_CAPACITY)
        .contiguous(),
        "all_ones_rows": torch.full((1, 1, CHUNK_ROWS, SPARSE_INDEX_CAPACITY), ALL_ONES_U32, dtype=torch.int64),
        "block_offsets_rows": qsa_row_constants()["block_offsets"].expand(1, 1, CHUNK_ROWS, TOKEN_BUDGET).contiguous(),
        "sentinel_pad_rows": qsa_row_constants()["sentinel_pad"]
        .expand(1, 1, CHUNK_ROWS, SPARSE_INDEX_CAPACITY - TOKEN_BUDGET)
        .contiguous(),
        "pool_select": pool_select.reshape(1, 1, CHUNK_ROWS, CHUNK_ROWS),
        "row_selects": row_selects.reshape(CHUNK_BLOCKS, 1, 1, CHUNK_ROWS, CHUNK_ROWS),
    }


@dataclass(frozen=True)
class Qwen38TTNNQSAChunkConstants:
    """Replicated constants of the chunk path, one set per model (about 2.3 MB at 8192 resident blocks).

    UINT32 ROW_MAJOR: ``arange32_lanes`` / ``block_start_lanes`` ``[1,1,1,32]`` (lane j = j; lane i = 4i for
    the eight block starts) added to the position's index row for the RoPE lookups; ``row_index_blocks`` /
    ``arange_blocks_rows`` ``[1,1,32,blocks]`` and ``row_index_slots`` / ``arange_slots_rows`` /
    ``all_ones_rows`` ``[1,1,32,2080]``, the same-shape templates of the per-row geometry;
    ``block_offsets_rows`` ``[1,1,32,2048]`` and ``sentinel_pad_rows`` ``[1,1,32,32]``, the 1-row index rows
    repeated per row.  BF16 TILE: ``pool_select`` ``[1,1,32,32]``, the eight ``row_selects`` (see
    :func:`qsa_chunk_constant_rows`) and ``zero_value_half_rows`` ``[1,6,32,256]``, the zero V half of the
    sparse_sdpa query rows.
    """

    allocated_compressed_blocks: int
    arange32_lanes: Any
    block_start_lanes: Any
    row_index_blocks: Any
    arange_blocks_rows: Any
    row_index_slots: Any
    arange_slots_rows: Any
    all_ones_rows: Any
    block_offsets_rows: Any
    sentinel_pad_rows: Any
    pool_select: Any
    row_selects: tuple[Any, ...]
    zero_value_half_rows: Any

    @classmethod
    def build(
        cls, mesh_device, mesh_contract: Qwen38MeshContract, allocated_compressed_blocks: int
    ) -> "Qwen38TTNNQSAChunkConstants":
        mesh_contract.validate_mesh(mesh_device)
        host = qsa_chunk_constant_rows(allocated_compressed_blocks)
        uploaded: list[Any] = []

        def upload_uint32(name: str):
            tensor = _upload_uint32(mesh_device, mesh_contract, host[name], layout=ttnn.ROW_MAJOR_LAYOUT)
            uploaded.append(tensor)
            return tensor

        def upload_bf16(values: torch.Tensor, label: str):
            tensor = ttnn.from_torch(
                values.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
            )
            uploaded.append(tensor)
            _require_shape(tensor, tuple(values.shape), label)
            mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)
            return tensor

        try:
            return cls(
                allocated_compressed_blocks=int(host["row_index_blocks"].shape[-1]),
                arange32_lanes=upload_uint32("arange32_lanes"),
                block_start_lanes=upload_uint32("block_start_lanes"),
                row_index_blocks=upload_uint32("row_index_blocks"),
                arange_blocks_rows=upload_uint32("arange_blocks_rows"),
                row_index_slots=upload_uint32("row_index_slots"),
                arange_slots_rows=upload_uint32("arange_slots_rows"),
                all_ones_rows=upload_uint32("all_ones_rows"),
                block_offsets_rows=upload_uint32("block_offsets_rows"),
                sentinel_pad_rows=upload_uint32("sentinel_pad_rows"),
                pool_select=upload_bf16(host["pool_select"], "QSA chunk pool select"),
                row_selects=tuple(
                    upload_bf16(host["row_selects"][block], f"QSA chunk row select {block}")
                    for block in range(CHUNK_BLOCKS)
                ),
                zero_value_half_rows=upload_bf16(
                    torch.zeros(1, QUERY_HEADS_PER_DEVICE, CHUNK_ROWS, HEAD_DIM), "QSA chunk zero value half rows"
                ),
            )
        except BaseException:
            _deallocate(*uploaded)
            raise

    def deallocate(self) -> None:
        _deallocate(
            self.arange32_lanes,
            self.block_start_lanes,
            self.row_index_blocks,
            self.arange_blocks_rows,
            self.row_index_slots,
            self.arange_slots_rows,
            self.all_ones_rows,
            self.block_offsets_rows,
            self.sentinel_pad_rows,
            self.pool_select,
            *self.row_selects,
            self.zero_value_half_rows,
        )


@dataclass(frozen=True)
class Qwen38TTNNQSAChunkInputs:
    """Per-chunk device tensors derived from the position; shared by all QSA layers.

    ``kv_block_start`` UINT32 ``[1,1,1,1]`` (= P), ``block_index_i32`` eight INT32 ``[1]`` (P/4 + i),
    ``indexer_neg_mask`` BF16 ROW_MAJOR ``[1,1,32,blocks]`` and the UINT32 ROW_MAJOR ``[1,1,32,2080]``
    ``row_keep_bits`` / ``row_fill``: row j is the 1-row input at position P + j.
    """

    kv_block_start: Any
    block_index_i32: tuple[Any, ...]
    indexer_neg_mask: Any
    row_keep_bits: Any
    row_fill: Any

    def deallocate(self) -> None:
        _deallocate(
            self.kv_block_start, *self.block_index_i32, self.indexer_neg_mask, self.row_keep_bits, self.row_fill
        )


def derive_qsa_chunk_inputs(
    position_scalar, constants: Qwen38TTNNQSAPositionConstants, chunk: Qwen38TTNNQSAChunkConstants
) -> Qwen38TTNNQSAChunkInputs:
    """:func:`derive_qsa_position_inputs` for the 32 rows of a chunk: the same exact UINT32 ops on the
    same-shape templates, with ``P`` the only broadcast operand (``pos = row_index + P``).  The staging
    and ring one-hots have no chunk form: the slab and the compressed blocks are written whole."""

    _require_shape(position_scalar, (1, 1, 1, 1), "QSA position scalar")
    if position_scalar.dtype != ttnn.uint32 or position_scalar.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise RuntimeError(f"QSA position scalar must be UINT32 ROW_MAJOR, got {tensor_metadata(position_scalar)}")
    dram = ttnn.DRAM_MEMORY_CONFIG
    u32 = ttnn.uint32
    blocks = chunk.allocated_compressed_blocks
    if blocks != constants.allocated_compressed_blocks:
        raise ValueError(
            f"QSA chunk constants were built for {blocks} blocks, "
            f"the position constants for {constants.allocated_compressed_blocks}"
        )

    kv_block_start = ttnn.bitwise_and(position_scalar, constants.high27_mask, memory_config=dram)
    block_index = ttnn.bitwise_right_shift(position_scalar, 2, memory_config=dram)
    block_indices_i32 = []
    for block in range(CHUNK_BLOCKS):
        shifted = block_index if block == 0 else ttnn.add(block_index, block, memory_config=dram)
        block_indices_i32.append(ttnn.reshape(ttnn.typecast(shifted, ttnn.int32, memory_config=dram), (1,)))
        if block:
            _deallocate(shifted)
    _deallocate(block_index)

    # Per row j: context = P + j + 1, complete = context // 4; blocks at or past complete are masked.
    context_blocks = ttnn.add(chunk.row_index_blocks, position_scalar, memory_config=dram)
    context_blocks_plus = ttnn.add(context_blocks, 1, memory_config=dram)
    complete_blocks_rows = ttnn.bitwise_right_shift(context_blocks_plus, 2, memory_config=dram)
    valid_bits = ttnn.lt(chunk.arange_blocks_rows, complete_blocks_rows, dtype=u32, memory_config=dram)
    valid = ttnn.typecast(valid_bits, ttnn.bfloat16, memory_config=dram)
    invalid = ttnn.rsub(valid, 1.0, memory_config=dram)
    indexer_neg_mask = ttnn.multiply(invalid, INDEXER_MASK_VALUE, memory_config=dram)
    _deallocate(context_blocks, context_blocks_plus, complete_blocks_rows, valid_bits, valid, invalid)

    # The sparse row per row j, as the 1-row chain on [1,1,32,2080] operands.
    positions = ttnn.add(chunk.row_index_slots, position_scalar, memory_config=dram)
    context_length = ttnn.add(positions, 1, memory_config=dram)
    complete_blocks = ttnn.bitwise_right_shift(context_length, 2, memory_config=dram)
    selected_blocks = ttnn.minimum(complete_blocks, BLOCK_TOPK, memory_config=dram)
    lo = ttnn.bitwise_left_shift(selected_blocks, 2, memory_config=dram)
    tail_count = ttnn.bitwise_and(context_length, COMPRESS_RATIO - 1, memory_config=dram)
    hi = ttnn.add(lo, tail_count, memory_config=dram)
    before_lo = ttnn.lt(chunk.arange_slots_rows, lo, dtype=u32, memory_config=dram)
    row_keep_bits = ttnn.multiply(before_lo, chunk.all_ones_rows, memory_config=dram)
    skipped_blocks = ttnn.subtract(complete_blocks, selected_blocks, memory_config=dram)
    tail_shift = ttnn.bitwise_left_shift(skipped_blocks, 2, memory_config=dram)
    tail_ids = ttnn.add(chunk.arange_slots_rows, tail_shift, memory_config=dram)
    from_lo = ttnn.ge(chunk.arange_slots_rows, lo, dtype=u32, memory_config=dram)
    before_hi = ttnn.lt(chunk.arange_slots_rows, hi, dtype=u32, memory_config=dram)
    tail_bits = ttnn.multiply(from_lo, before_hi, memory_config=dram)
    row_tail_fill = ttnn.multiply(tail_ids, tail_bits, memory_config=dram)
    from_hi = ttnn.ge(chunk.arange_slots_rows, hi, dtype=u32, memory_config=dram)
    row_sentinel_bits = ttnn.multiply(from_hi, chunk.all_ones_rows, memory_config=dram)
    row_fill = ttnn.bitwise_or(row_tail_fill, row_sentinel_bits, memory_config=dram)
    _deallocate(
        positions,
        context_length,
        complete_blocks,
        selected_blocks,
        lo,
        tail_count,
        hi,
        before_lo,
        skipped_blocks,
        tail_shift,
        tail_ids,
        from_lo,
        before_hi,
        tail_bits,
        row_tail_fill,
        from_hi,
        row_sentinel_bits,
    )
    inputs = Qwen38TTNNQSAChunkInputs(
        kv_block_start=kv_block_start,
        block_index_i32=tuple(block_indices_i32),
        indexer_neg_mask=indexer_neg_mask,
        row_keep_bits=row_keep_bits,
        row_fill=row_fill,
    )
    for name, tensor, shape, dtype, layout in (
        ("kv_block_start", kv_block_start, (1, 1, 1, 1), u32, ttnn.ROW_MAJOR_LAYOUT),
        ("indexer_neg_mask", indexer_neg_mask, (1, 1, CHUNK_ROWS, blocks), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        ("row_keep_bits", row_keep_bits, (1, 1, CHUNK_ROWS, SPARSE_INDEX_CAPACITY), u32, ttnn.ROW_MAJOR_LAYOUT),
        ("row_fill", row_fill, (1, 1, CHUNK_ROWS, SPARSE_INDEX_CAPACITY), u32, ttnn.ROW_MAJOR_LAYOUT),
        *(
            (f"block_index_i32[{i}]", t, (1,), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            for i, t in enumerate(block_indices_i32)
        ),
    ):
        _require_shape(tensor, shape, f"QSA chunk input {name}")
        if tensor.dtype != dtype or tensor.layout != layout:
            raise RuntimeError(
                f"QSA chunk input {name} must be {dtype} {layout} {list(shape)}, got {tensor_metadata(tensor)}"
            )
    return inputs


def emulate_qsa_chunk_inputs(position: int, *, allocated_compressed_blocks: int) -> dict[str, torch.Tensor]:
    """Torch reference of :func:`derive_qsa_chunk_inputs`: row j is :func:`emulate_qsa_position_inputs` at P + j."""

    if isinstance(position, bool) or not isinstance(position, int) or position % CHUNK_ROWS:
        raise ValueError(f"QSA chunk position must be an integer multiple of {CHUNK_ROWS}, got {position!r}")
    rows = [
        emulate_qsa_position_inputs(position + row, allocated_compressed_blocks=allocated_compressed_blocks)
        for row in range(CHUNK_ROWS)
    ]
    return {
        "kv_block_start": rows[0]["kv_block_start"],
        "block_index_i32": tuple(
            torch.tensor([position // COMPRESS_RATIO + block], dtype=torch.int32) for block in range(CHUNK_BLOCKS)
        ),
        "indexer_neg_mask": torch.cat([row["indexer_neg_mask"] for row in rows], dim=2),
        "row_keep_bits": torch.cat([row["row_keep_bits"] for row in rows], dim=2),
        "row_fill": torch.cat([row["row_fill"] for row in rows], dim=2),
    }


def chunk_handoff_ring_select_rows(prefilled: int) -> torch.Tensor:
    """Host image of the BF16 ``[1,1,32,32]`` 0/1 select whose matmul with the last chunk's kept raw keys is the
    raw-key ring at ``P = prefilled``: row j < prefilled % 4 picks kept row (prefilled % 32) - (prefilled % 4) + j,
    the raw key of position (prefilled & ~3) + j; every other row is zero (the ring's rows past the open block)."""

    if isinstance(prefilled, bool) or type(prefilled) is not int or prefilled < 0:
        raise ValueError(f"prefilled position count must be a non-negative int, got {prefilled!r}")
    select = torch.zeros(1, 1, CHUNK_ROWS, CHUNK_ROWS, dtype=torch.bfloat16)
    first = prefilled % CHUNK_ROWS - prefilled % COMPRESS_RATIO
    for row in range(prefilled % COMPRESS_RATIO):
        select[0, 0, row, first + row] = 1.0
    return select


@dataclass(frozen=True)
class Qwen38TTNNQSAChunkState:
    """Per-layer fixed-address sources of the prefill hand-off: the chunk's own ``[v|k]`` slab ``[1,1,32,512]``
    (the open block's staging when the prompt ends inside it) and its raw index keys ``[1,1,32,128]`` (the
    ring slots of an incomplete last block).  Written whole by every chunk, read by the eager fix-up."""

    layer_index: int
    epoch: int
    kept_kv: Any
    kept_raw: Any


@dataclass(frozen=True)
class Qwen38TTNNQSAGenericState:
    """Fixed-address QSA state for the position-generic body.

    ``kv_staging`` is BF16 TILE (the pinned runtime rejects a preallocated
    output for ROW_MAJOR binary operands) and is untilized once per token for
    the ROW_MAJOR cache write.  ``raw_key_ring`` rows 4-31 are never written
    and stay exactly zero, so a scaled sum over all 32 rows is the block mean.
    ``compressed_index_cache`` has one tile more than the resident blocks: the
    indexer's fixed query window, never written.
    """

    layer_index: int
    epoch: int
    packed_kv_cache: Any
    compressed_index_cache: Any
    kv_staging: Any
    raw_key_ring: Any


class Qwen38TTNNQSA:
    """One exact global-B1 QSA decode layer on an admitted 1x4 mesh."""

    max_context = MAX_CONTEXT
    allocated_context = MAX_CONTEXT
    allocated_compressed_blocks = MAX_COMPRESSED_BLOCKS

    def __init__(
        self,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        weights: Qwen38TTNNQSAWeights,
        *,
        layer_index: int,
        rms_norm_eps: float = 1e-6,
        max_context: int = MAX_CONTEXT,
        allocated_context: int = MAX_CONTEXT,
        collective_topology=None,
        regime_split: bool | None = None,
    ) -> None:
        mesh_contract.validate_mesh(mesh_device)
        weights.validate(mesh_contract)
        if regime_split is not None and not isinstance(regime_split, bool):
            raise TypeError(f"regime_split must be a bool or None, got {regime_split!r}")
        self.regime_split = QSA_INDEXER_REGIME_SPLIT if regime_split is None else regime_split
        if max_context != MAX_CONTEXT:
            raise ValueError(f"exact checkpoint context is {MAX_CONTEXT}; refusing max_context={max_context}")
        allocated_context = validate_qsa_cache_capacity(allocated_context)
        if layer_index < 0:
            raise ValueError("QSA layer index must be nonnegative")
        if weights.layer_index != layer_index:
            raise ValueError(f"QSA weights belong to layer {weights.layer_index}, got module layer {layer_index}")
        if rms_norm_eps != 1e-6:
            raise ValueError(f"pinned QSA RMS epsilon is 1e-6, got {rms_norm_eps}")
        self.mesh_device = mesh_device
        self.mesh_contract = mesh_contract
        self.weights = weights
        self.layer_index = layer_index
        self.rms_norm_eps = rms_norm_eps
        self.max_context = MAX_CONTEXT
        self.allocated_context = allocated_context
        self.allocated_compressed_blocks = allocated_context // COMPRESS_RATIO
        self.collective_topology = collective_topology or ttnn.Topology.Linear
        self._next_epoch = 1
        self._live_epochs: set[int] = set()
        self._next_view_id = 1
        self._live_views: dict[int, int] = {}
        self._protected_views: set[int] = set()
        # A decode-trace chain captures one exact-position graph per token.
        # Every graph consumes the preceding QSA metadata view, so its staging,
        # raw-tail, and selection buffers must keep their capture-time addresses
        # until all dependent traces are released.  Production eager decode
        # leaves this disabled and retains its existing immediate reclamation.
        self._trace_input_retention_active = False
        self._trace_retained_inputs: dict[tuple[str, int], Any] = {}
        self._trace_rollover_staging: list[Any] = []
        self._live_generic_epochs: set[int] = set()
        # Position-generic indexer window: the fixed 32-row query chunk is the
        # tile past the last resident block (the compressed cache of the generic
        # state carries that extra zero tile), so row 0 sees every resident column.
        self.indexer_chunk_start = self.allocated_compressed_blocks
        # paged_update_cache reads one height-sharded user per core.
        self.compressed_row_memory_config = ttnn.create_sharded_memory_config(
            (ttnn.TILE_SIZE, INDEX_HEAD_DIM),
            ttnn.CoreGrid(y=1, x=1),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # The custom indexer LLK explicitly rejects FP32 DEST/full-sync.
        self.indexer_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        # DRAM-sharded decode matmul configs; every K=2560 projection shares
        # one eight-core activation shard of the gathered hidden state.
        self.hidden_act_memory_config, self.qg_program_config = dram_sharded_matmul_configs(
            mesh_device, HIDDEN_SIZE, 2 * LOCAL_QUERY_WIDTH, num_cores=8
        )
        _, self.kv_program_config = dram_sharded_matmul_configs(mesh_device, HIDDEN_SIZE, HEAD_DIM, num_cores=8)
        _, self.index_program_config = dram_sharded_matmul_configs(
            mesh_device, HIDDEN_SIZE, INDEX_HEAD_DIM, num_cores=8
        )
        self.out_act_memory_config, self.out_program_config = dram_sharded_matmul_configs(
            mesh_device, LOCAL_QUERY_WIDTH, HIDDEN_SIZE, num_cores=16
        )

        self.index_gate = ttnn.from_torch(
            torch.full((1, 1, ttnn.TILE_SIZE, 1), INDEX_HEAD_DIM**-0.5, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
        )
        mesh_contract.validate_tensor(self.index_gate, placement=TensorPlacement.REPLICATED)
        _require_shape(self.index_gate, (1, 1, ttnn.TILE_SIZE, 1), "QSA index score gate")
        if self.index_gate.dtype != ttnn.bfloat16 or self.index_gate.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError("QSA index score gate must be BF16 TILE")
        # The zero V half of the sparse_sdpa query row, one per local head.
        self.zero_value_half = self._allocate_replicated_tile_zeros((1, QUERY_HEADS_PER_DEVICE, 1, HEAD_DIM))
        self.uint32_rows = {
            name: _upload_uint32(mesh_device, mesh_contract, values, layout=ttnn.ROW_MAJOR_LAYOUT)
            for name, values in qsa_row_constants().items()
        }
        for name, value in self.uint32_rows.items():
            _require_shape(value, (1, 1, 1, QSA_ROW_WIDTHS[name]), f"QSA {name} row")
            if value.dtype != ttnn.uint32 or value.layout != ttnn.ROW_MAJOR_LAYOUT:
                raise RuntimeError(f"QSA {name} row must be UINT32 ROW_MAJOR")
        self.block_offsets = self.uint32_rows["block_offsets"]
        self.rep_index_lo = self.uint32_rows["rep_index_lo"]
        self.rep_index_hi = self.uint32_rows["rep_index_hi"]
        self.sentinel_pad = self.uint32_rows["sentinel_pad"]
        self.arange_row = self.uint32_rows["arange_row"]
        self.keep_step = self.uint32_rows["keep_step"]
        self.sentinel_step = self.uint32_rows["sentinel_step"]
        self.slot_zero = self.uint32_rows["slot_zero"]

    def deallocate(self) -> None:
        """Release module-owned constants after every state epoch is gone.

        Static checkpoint weights have independent ownership and are released
        with :meth:`Qwen38TTNNQSAWeights.deallocate`.
        """

        if (
            self._live_epochs
            or self._live_views
            or self._trace_retained_inputs
            or self._trace_rollover_staging
            or self._live_generic_epochs
        ):
            raise RuntimeError(
                "cannot deallocate QSA module constants while state is live: "
                f"epochs={sorted(self._live_epochs)} views={sorted(self._live_views)} "
                f"trace_inputs={len(self._trace_retained_inputs)} "
                f"trace_rollover_staging={len(self._trace_rollover_staging)} "
                f"generic_epochs={sorted(self._live_generic_epochs)}"
            )
        _deallocate(self.index_gate, self.zero_value_half, *self.uint32_rows.values())
        self._release_shared_rows(keep=0)

    # Per-token rows shared by every layer of one mesh: the natural-order
    # sparse row and the template masks depend only on host ints, so the first
    # layer at a position builds them and the other eleven reuse the tensor.
    _shared_rows: dict[tuple, _SharedRow] = {}

    def _shared_row(self, key: tuple, build):
        """Return the row for ``key`` (built once); a trace that reads it retains it with this module."""

        full_key = (id(self.mesh_device), *key)
        entry = self._shared_rows.get(full_key)
        if entry is None:
            entry = self._shared_rows[full_key] = _SharedRow(self.mesh_device, build(), retained=False)
        if self._trace_input_retention_active and not entry.retained:
            entry.retained = True
            self._trace_retained_inputs.setdefault(_tensor_key(entry.tensor), entry.tensor)
        if not entry.retained:
            self._release_shared_rows(keep=SHARED_ROW_CACHE)
        return entry.tensor

    def _release_shared_rows(self, *, keep: int) -> None:
        """Release this mesh's eager rows beyond the ``keep`` most recent; retained rows stay."""

        eager = [
            key for key, entry in self._shared_rows.items() if key[0] == id(self.mesh_device) and not entry.retained
        ]
        for key in eager[: max(0, len(eager) - keep)]:
            ttnn.deallocate(self._shared_rows.pop(key).tensor)

    def _step_window(self, step, count: int):
        # count leading entries of the step's first half, then its second half.
        # The window is half the row, so slice.cpp never hands back the module
        # row itself (its full-width no-op) and every window can be released.
        if not 0 <= count <= SPARSE_INDEX_CAPACITY:
            raise ValueError(f"step window count must be in [0, {SPARSE_INDEX_CAPACITY}], got {count}")
        return ttnn.slice(
            step,
            (0, 0, 0, SPARSE_INDEX_CAPACITY - count),
            (1, 1, 1, 2 * SPARSE_INDEX_CAPACITY - count),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _natural_row(self, context_length: int):
        """``[0 .. P | sentinels]``: every causal token in natural order (two device ops per token)."""

        def build():
            sentinel_bits = self._step_window(self.sentinel_step, context_length)
            row = ttnn.bitwise_or(self.arange_row, sentinel_bits, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _deallocate(sentinel_bits)
            return row

        return self._shared_row(("natural", context_length), build)

    def _template_masks(self, complete_token_count: int, tail_start: int, valid_count: int):
        """Keep window over the expanded slots and the tail-ids-then-sentinels row (five or six ops per token)."""

        keep = self._shared_row(
            ("keep", complete_token_count), lambda: self._step_window(self.keep_step, complete_token_count)
        )

        def build_tail():
            # Slots [lo, valid) hold tail_start + (slot - lo); slots >= valid hold the sentinel.
            past_complete = self._step_window(self.sentinel_step, complete_token_count)
            sentinel_bits = self._step_window(self.sentinel_step, valid_count)
            shift = tail_start - complete_token_count
            ids = (
                self.arange_row
                if shift == 0
                else ttnn.add(self.arange_row, shift, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            )
            tail = ttnn.bitwise_and(ids, past_complete, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            row = ttnn.bitwise_or(tail, sentinel_bits, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _deallocate(past_complete, sentinel_bits, tail, None if ids is self.arange_row else ids)
            return row

        tail_sentinel = self._shared_row(("tail", complete_token_count, tail_start, valid_count), build_tail)
        return keep, tail_sentinel

    @property
    def trace_retained_input_count(self) -> int:
        return len(self._trace_retained_inputs)

    @property
    def trace_rollover_staging_count(self) -> int:
        return len(self._trace_rollover_staging)

    def begin_trace_input_retention(self) -> None:
        """Keep consumed view inputs alive while an exact-position trace chain is captured."""

        if self._trace_input_retention_active:
            raise RuntimeError("QSA trace-input retention is already active")
        if self._trace_retained_inputs:
            raise RuntimeError("QSA retained trace inputs were not released")
        if self._trace_rollover_staging:
            raise RuntimeError("QSA trace rollover staging was not released")
        self._trace_input_retention_active = True

    def preallocate_trace_rollover_staging(self, count: int) -> None:
        """Prepare exact-address QSA rollover buffers before trace capture starts."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("QSA trace rollover staging count must be a positive integer")
        if not self._trace_input_retention_active:
            raise RuntimeError("QSA trace-input retention must be active before rollover staging is prepared")
        if self._trace_rollover_staging:
            raise RuntimeError("QSA trace rollover staging is already prepared")
        for _ in range(count):
            self._trace_rollover_staging.append(
                self._allocate_pair_grouped(
                    (1, 1, CACHE_WRITE_ROWS, 2 * HEAD_DIM),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            )

    def end_trace_input_retention(self) -> int:
        """Stop retaining new inputs and return the number owned by this module."""

        if not self._trace_input_retention_active:
            raise RuntimeError("QSA trace-input retention is not active")
        self._trace_input_retention_active = False
        return len(self._trace_retained_inputs)

    def release_trace_retained_inputs(self) -> None:
        """Release trace-chain inputs after every trace using them has been released."""

        if self._trace_input_retention_active:
            raise RuntimeError("cannot release QSA trace inputs while retention is active")
        released = set(self._trace_retained_inputs)
        for key, tensor in tuple(self._trace_retained_inputs.items()):
            ttnn.deallocate(tensor)
            del self._trace_retained_inputs[key]
        for key in [key for key, entry in self._shared_rows.items() if _tensor_key(entry.tensor) in released]:
            del self._shared_rows[key]
        while self._trace_rollover_staging:
            ttnn.deallocate(self._trace_rollover_staging.pop())

    def _allocate_pair_grouped(self, shape: tuple[int, ...], *, layout) -> Any:
        value = ttnn.zeros(
            shape,
            dtype=ttnn.bfloat16,
            layout=layout,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _canonicalize_flat_replicated_topology(value, self.mesh_contract)
        _retag_tensor(value, reference=value, shard_dim=1)
        self.mesh_contract.validate_tensor(value, placement=TensorPlacement.KV_PAIR_GROUPED, shard_dim=1)
        return value

    def _take_next_kv_staging(self) -> Any:
        if not self._trace_input_retention_active:
            return self._allocate_pair_grouped(
                (1, 1, CACHE_WRITE_ROWS, 2 * HEAD_DIM),
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        if not self._trace_rollover_staging:
            raise RuntimeError("QSA trace capture reached an unprepared KV-staging rollover")
        return self._trace_rollover_staging.pop(0)

    def _allocate_replicated_tile_zeros(self, shape: tuple[int, ...]) -> Any:
        value = ttnn.zeros(
            shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _canonicalize_flat_replicated_topology(value, self.mesh_contract)
        return value

    def allocate_state(self) -> Qwen38TTNNQSAState:
        """Allocate this layer's persistent caches and empty decode staging."""

        packed_kv = self._allocate_pair_grouped(
            (1, 1, self.allocated_context, 2 * HEAD_DIM),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        compressed = self._allocate_replicated_tile_zeros((1, 1, self.allocated_compressed_blocks, INDEX_HEAD_DIM))
        staging = self._allocate_pair_grouped((1, 1, CACHE_WRITE_ROWS, 2 * HEAD_DIM), layout=ttnn.ROW_MAJOR_LAYOUT)
        epoch = self._next_epoch
        self._next_epoch += 1
        self._live_epochs.add(epoch)
        view_id = self._allocate_view_id(epoch)
        return Qwen38TTNNQSAState(
            layer_index=self.layer_index,
            epoch=epoch,
            view_id=view_id,
            next_position=0,
            compressed_blocks=0,
            raw_tail_count=0,
            raw_index_tail=None,
            packed_kv_cache=packed_kv,
            compressed_index_cache=compressed,
            kv_staging=staging,
            kv_staging_owned=True,
        )

    def release_state(self, state: Qwen38TTNNQSAState) -> None:
        """Release one cache epoch after every snapshot/view is dead."""

        self._validate_state(state)
        sibling_views = {
            view_id for view_id, epoch in self._live_views.items() if epoch == state.epoch and view_id != state.view_id
        }
        if sibling_views:
            raise RuntimeError(
                f"cannot release QSA cache epoch {state.epoch}; live sibling views remain {sorted(sibling_views)}"
            )
        self._release_view_transients(state)
        _deallocate(state.packed_kv_cache, state.compressed_index_cache)
        self._live_views.pop(state.view_id)
        self._protected_views.discard(state.view_id)
        self._live_epochs.remove(state.epoch)

    def reset_state(self, state: Qwen38TTNNQSAState) -> Qwen38TTNNQSAState:
        """Release an unbranched epoch and allocate a fresh empty state."""

        self.release_state(state)
        return self.allocate_state()

    def allocate_generic_state(self) -> Qwen38TTNNQSAGenericState:
        """Allocate the fixed-address caches, staging tile and raw-key ring of the generic body."""

        epoch = self._next_epoch
        self._next_epoch += 1
        self._live_generic_epochs.add(epoch)
        return Qwen38TTNNQSAGenericState(
            layer_index=self.layer_index,
            epoch=epoch,
            packed_kv_cache=self._allocate_pair_grouped(
                (1, 1, self.allocated_context, 2 * HEAD_DIM),
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            compressed_index_cache=self._allocate_replicated_tile_zeros(
                (1, 1, self.allocated_compressed_blocks + ttnn.TILE_SIZE, INDEX_HEAD_DIM)
            ),
            kv_staging=self._allocate_pair_grouped((1, 1, CACHE_WRITE_ROWS, 2 * HEAD_DIM), layout=ttnn.TILE_LAYOUT),
            raw_key_ring=self._allocate_replicated_tile_zeros((1, 1, CACHE_WRITE_ROWS, INDEX_HEAD_DIM)),
        )

    def release_generic_state(self, state: Qwen38TTNNQSAGenericState) -> None:
        self._validate_generic_state(state)
        _deallocate(state.packed_kv_cache, state.compressed_index_cache, state.kv_staging, state.raw_key_ring)
        self._live_generic_epochs.remove(state.epoch)

    def reset_generic_state_inplace(self, state: Qwen38TTNNQSAGenericState) -> None:
        """Zero the staging tile and raw-key ring without changing any address (trace safe).

        The two caches need no reset: sparse_sdpa gathers only rows <= P and
        the indexer masks blocks >= complete_blocks, so stale finite values are
        never observable.
        """

        self._validate_generic_state(state)
        for label, tensor in (("QSA KV staging", state.kv_staging), ("QSA raw-key ring", state.raw_key_ring)):
            zeroed = ttnn.fill(tensor, 0.0, output_tensor=tensor)
            if _tensor_key(zeroed) != _tensor_key(tensor):
                raise RuntimeError(f"{label} reset was not in place")

    def _validate_generic_state(self, state: Qwen38TTNNQSAGenericState) -> None:
        if state.layer_index != self.layer_index:
            raise ValueError(f"QSA generic state belongs to layer {state.layer_index}, expected {self.layer_index}")
        if state.epoch not in self._live_generic_epochs:
            raise ValueError(f"QSA generic state epoch {state.epoch} was not allocated by this module")
        expected = (
            ("packed QSA cache", state.packed_kv_cache, (1, 1, self.allocated_context, 2 * HEAD_DIM)),
            (
                "compressed index cache",
                state.compressed_index_cache,
                (1, 1, self.allocated_compressed_blocks + ttnn.TILE_SIZE, INDEX_HEAD_DIM),
            ),
            ("QSA KV staging", state.kv_staging, (1, 1, CACHE_WRITE_ROWS, 2 * HEAD_DIM)),
            ("QSA raw-key ring", state.raw_key_ring, (1, 1, CACHE_WRITE_ROWS, INDEX_HEAD_DIM)),
        )
        for label, tensor, shape in expected:
            _require_shape(tensor, shape, label)
            if tensor.dtype != ttnn.bfloat16:
                raise RuntimeError(f"{label} must be BF16, got {tensor_metadata(tensor)}")
        if state.packed_kv_cache.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError(f"packed QSA cache must be ROW_MAJOR, got {tensor_metadata(state.packed_kv_cache)}")
        for label, tensor, _ in expected[1:]:
            if tensor.layout != ttnn.TILE_LAYOUT:
                raise RuntimeError(f"{label} must be TILE, got {tensor_metadata(tensor)}")
        for tensor in (state.packed_kv_cache, state.kv_staging):
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.KV_PAIR_GROUPED, shard_dim=1)
        for tensor in (state.compressed_index_cache, state.raw_key_ring):
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)

    def _validate_position_inputs(self, position: Qwen38TTNNQSAPositionInputs) -> None:
        expected = (
            ("kv_block_start", position.kv_block_start, (1, 1, 1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            ("kv_row_hit", position.kv_row_hit, (1, 1, CACHE_WRITE_ROWS, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            ("kv_row_keep", position.kv_row_keep, (1, 1, CACHE_WRITE_ROWS, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            ("ring_hit", position.ring_hit, (1, 1, CACHE_WRITE_ROWS, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            ("ring_keep", position.ring_keep, (1, 1, CACHE_WRITE_ROWS, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            ("block_index_i32", position.block_index_i32, (1,), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            (
                "indexer_neg_mask",
                position.indexer_neg_mask,
                (1, 1, 1, self.allocated_compressed_blocks),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            ),
            (
                "row_keep_bits",
                position.row_keep_bits,
                (1, 1, 1, SPARSE_INDEX_CAPACITY),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
            ),
            (
                "row_fill",
                position.row_fill,
                (1, 1, 1, SPARSE_INDEX_CAPACITY),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
            ),
        )
        for name, tensor, shape, dtype, layout in expected:
            _require_shape(tensor, shape, f"QSA position input {name}")
            if tensor.dtype != dtype or tensor.layout != layout:
                raise RuntimeError(
                    f"QSA position input {name} must be {dtype} {layout} {list(shape)}, got {tensor_metadata(tensor)}"
                )

    def _allocate_view_id(self, epoch: int) -> int:
        view_id = self._next_view_id
        self._next_view_id += 1
        self._live_views[view_id] = epoch
        return view_id

    def checkpoint_state(self, state: Qwen38TTNNQSAState) -> Qwen38TTNNQSAState:
        """Protect and return an immutable rollback view.

        The first branch step must pass ``retain_input_state=True``.  The guard
        in :meth:`forward_decode` rejects accidental consumption of this view.
        """

        self._validate_state(state)
        epoch_views = {view_id for view_id, epoch in self._live_views.items() if epoch == state.epoch}
        if epoch_views != {state.view_id}:
            raise RuntimeError(f"QSA checkpoint requires one unbranched live view, got {sorted(epoch_views)}")
        if self._protected_views:
            raise RuntimeError(f"another QSA checkpoint is already protected: {sorted(self._protected_views)}")
        self._protected_views.add(state.view_id)
        return state

    def _validate_transaction_pair(
        self,
        checkpoint: Qwen38TTNNQSAState,
        current: Qwen38TTNNQSAState,
    ) -> None:
        if checkpoint.view_id not in self._protected_views:
            raise RuntimeError(f"QSA checkpoint view {checkpoint.view_id} is not protected")
        expected = {checkpoint.view_id, current.view_id}
        actual = {view_id for view_id, epoch in self._live_views.items() if epoch == checkpoint.epoch}
        if actual != expected:
            raise RuntimeError(
                "QSA transaction must have exactly its checkpoint and current branch live; "
                f"expected {sorted(expected)}, got {sorted(actual)}"
            )

    @staticmethod
    def _same_tensor(left, right) -> bool:
        return left is not None and right is not None and _tensor_key(left) == _tensor_key(right)

    def _release_view_transients(self, state: Qwen38TTNNQSAState, *, preserve_complete=None) -> None:
        owned = [state.raw_index_tail, state.kv_staging if state.kv_staging_owned else None]
        selection = state.last_selection
        if selection is not None:
            if (
                selection.owns_complete_indices
                and selection.complete_indices is not None
                and not self._same_tensor(selection.complete_indices, preserve_complete)
            ):
                owned.append(selection.complete_indices)
            if selection.owns_sparse_indices:
                owned.append(selection.sparse_indices)
        if self._trace_input_retention_active:
            for tensor in owned:
                if tensor is None:
                    continue
                self._trace_retained_inputs.setdefault(_tensor_key(tensor), tensor)
        else:
            _deallocate(*owned)

    def _consume_view(self, state: Qwen38TTNNQSAState, *, preserve_complete=None) -> None:
        self._release_view_transients(state, preserve_complete=preserve_complete)
        self._live_views.pop(state.view_id)
        self._protected_views.discard(state.view_id)

    def _transfer_complete_selection(
        self,
        source: Qwen38TTNNQSAState,
        target: Qwen38TTNNQSAState,
    ) -> tuple[Qwen38TTNNQSAState, Any | None]:
        """Transfer ownership when ``target`` borrows ``source`` block IDs."""

        old = source.last_selection
        new = target.last_selection
        same_complete = (
            old is not None
            and new is not None
            and (
                (old.complete_indices is None and new.complete_indices is None)
                or self._same_tensor(old.complete_indices, new.complete_indices)
            )
        )
        if old is None or new is None or new.source_view_id != source.view_id or not same_complete:
            return target, None
        transferred = replace(
            new,
            source_view_id=target.view_id,
            owns_complete_indices=old.owns_complete_indices,
        )
        preserve = old.complete_indices if old.owns_complete_indices else None
        return replace(target, last_selection=transferred), preserve

    def restore_state(
        self,
        current: Qwen38TTNNQSAState,
        checkpoint: Qwen38TTNNQSAState,
    ) -> Qwen38TTNNQSAState:
        """Discard one speculative branch and return its live checkpoint."""

        self._validate_state(current)
        self._validate_state(checkpoint)
        if current.epoch != checkpoint.epoch:
            raise ValueError("cannot restore QSA state across cache epochs")
        self._validate_transaction_pair(checkpoint, current)
        if current.view_id == checkpoint.view_id:
            self._protected_views.discard(checkpoint.view_id)
            return checkpoint
        self._consume_view(current)
        self._protected_views.discard(checkpoint.view_id)
        return checkpoint

    def commit_state(
        self,
        checkpoint: Qwen38TTNNQSAState,
        current: Qwen38TTNNQSAState,
    ) -> Qwen38TTNNQSAState:
        """Commit a branch and release only superseded checkpoint transients."""

        self._validate_state(checkpoint)
        self._validate_state(current)
        if current.epoch != checkpoint.epoch:
            raise ValueError("cannot commit QSA state across cache epochs")
        self._validate_transaction_pair(checkpoint, current)
        if current.view_id == checkpoint.view_id:
            self._protected_views.discard(checkpoint.view_id)
            return current
        current, preserve_complete = self._transfer_complete_selection(checkpoint, current)
        self._consume_view(checkpoint, preserve_complete=preserve_complete)
        self._protected_views.discard(current.view_id)
        return current

    def _validate_state(self, state: Qwen38TTNNQSAState) -> None:
        if state.layer_index != self.layer_index:
            raise ValueError(f"QSA state belongs to layer {state.layer_index}, expected {self.layer_index}")
        if state.epoch not in self._live_epochs:
            raise ValueError(f"QSA state cache epoch {state.epoch} was not allocated by this module")
        if self._live_views.get(state.view_id) != state.epoch:
            raise ValueError(f"QSA state view {state.view_id} is no longer live in epoch {state.epoch}")
        if not 0 <= state.next_position <= self.allocated_context:
            raise ValueError(
                f"QSA next position is outside allocated cache [0, {self.allocated_context}]: "
                f"{state.next_position}; semantic model limit remains {MAX_CONTEXT}"
            )
        expected_blocks, expected_tail = divmod(state.next_position, COMPRESS_RATIO)
        if state.compressed_blocks != expected_blocks or state.raw_tail_count != expected_tail:
            raise RuntimeError(
                "QSA compression metadata is inconsistent with position: "
                f"position={state.next_position} blocks={state.compressed_blocks} tail={state.raw_tail_count}"
            )
        if (state.raw_index_tail is None) != (state.raw_tail_count == 0):
            raise RuntimeError("QSA raw-tail tensor/count disagree")
        if state.raw_index_tail is not None:
            _require_shape(state.raw_index_tail, (1, 1, state.raw_tail_count, INDEX_HEAD_DIM), "raw index tail")
            if state.raw_index_tail.dtype != ttnn.bfloat16 or state.raw_index_tail.layout != ttnn.TILE_LAYOUT:
                raise RuntimeError("raw QSA index tail must be BF16 TILE")
            self.mesh_contract.validate_tensor(state.raw_index_tail, placement=TensorPlacement.REPLICATED)
        _require_shape(
            state.packed_kv_cache,
            (1, 1, self.allocated_context, 2 * HEAD_DIM),
            "packed QSA cache",
        )
        _require_shape(
            state.compressed_index_cache,
            (1, 1, self.allocated_compressed_blocks, INDEX_HEAD_DIM),
            "compressed index cache",
        )
        _require_shape(state.kv_staging, (1, 1, CACHE_WRITE_ROWS, 2 * HEAD_DIM), "QSA KV staging")
        self.mesh_contract.validate_tensor(
            state.packed_kv_cache, placement=TensorPlacement.KV_PAIR_GROUPED, shard_dim=1
        )
        self.mesh_contract.validate_tensor(state.compressed_index_cache, placement=TensorPlacement.REPLICATED)
        self.mesh_contract.validate_tensor(state.kv_staging, placement=TensorPlacement.KV_PAIR_GROUPED, shard_dim=1)
        if state.packed_kv_cache.dtype != ttnn.bfloat16 or state.packed_kv_cache.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError("packed QSA cache must be BF16 ROW_MAJOR")
        if (
            state.compressed_index_cache.dtype != ttnn.bfloat16
            or state.compressed_index_cache.layout != ttnn.TILE_LAYOUT
        ):
            raise RuntimeError("compressed QSA index cache must be BF16 TILE")
        if state.kv_staging.dtype != ttnn.bfloat16 or state.kv_staging.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError("QSA KV staging must be BF16 ROW_MAJOR")
        if state.last_selection is not None:
            selection = state.last_selection
            if selection.layer_index != state.layer_index or selection.epoch != state.epoch:
                raise RuntimeError("QSA state selection belongs to another layer/cache epoch")
            if self._live_views.get(selection.source_view_id) != state.epoch:
                raise RuntimeError("QSA state selection source view is no longer live")

    def _validate_rope(self, cos, sin, label: str) -> None:
        for value in (cos, sin):
            _require_shape(value, (1, 1, 1, ROPE_DIM), label)
            if value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT:
                raise RuntimeError(f"{label} must be BF16 TILE [1,1,1,{ROPE_DIM}], got {tensor_metadata(value)}")
            self.mesh_contract.validate_tensor(value, placement=TensorPlacement.REPLICATED)

    def _validate_reuse_selection(
        self,
        selection: Qwen38TTNNQSASelection,
        state: Qwen38TTNNQSAState,
    ) -> None:
        if selection.layer_index != self.layer_index or selection.epoch != state.epoch:
            raise ValueError("reused QSA selection belongs to another layer/cache epoch")
        if self._live_views.get(selection.source_view_id) != state.epoch:
            raise ValueError(f"reused QSA selection source view {selection.source_view_id} is no longer live")
        draft_distance = state.next_position - selection.source_position
        if not 0 <= draft_distance <= MAX_SPECULATIVE_STEPS:
            raise ValueError(
                f"QSA selection reuse distance must be in [0,{MAX_SPECULATIVE_STEPS}], got {draft_distance}"
            )
        if selection.complete_token_count % COMPRESS_RATIO or not (0 <= selection.complete_token_count <= TOKEN_BUDGET):
            raise ValueError("reused QSA selection has an invalid complete-token count")
        if (selection.complete_indices is None) != (selection.complete_token_count == 0):
            raise ValueError("reused QSA complete index tensor/count disagree")
        if selection.complete_indices is not None:
            _require_shape(
                selection.complete_indices,
                (1, 1, 1, TOKEN_BUDGET),
                "reused complete QSA indices",
            )
            if (
                selection.complete_indices.dtype != ttnn.uint32
                or selection.complete_indices.layout != ttnn.ROW_MAJOR_LAYOUT
            ):
                raise RuntimeError("reused complete QSA indices must be UINT32 ROW_MAJOR")
            self.mesh_contract.validate_tensor(
                selection.complete_indices,
                placement=TensorPlacement.REPLICATED,
            )
        elif selection.tail_start:
            raise ValueError("a reused QSA selection without scored blocks must start its tail at token 0")
        _require_shape(
            selection.sparse_indices,
            (1, 1, 1, SPARSE_INDEX_CAPACITY),
            "source sparse QSA indices",
        )
        if selection.sparse_indices.dtype != ttnn.uint32 or selection.sparse_indices.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError("source sparse QSA indices must be UINT32 ROW_MAJOR")
        self.mesh_contract.validate_tensor(selection.sparse_indices, placement=TensorPlacement.REPLICATED)

    def _all_gather_hidden(self, hidden_sharded):
        _require_shape(hidden_sharded, (1, 1, 1, HIDDEN_SIZE // TP_SIZE), "QSA hidden input")
        self.mesh_contract.validate_tensor(hidden_sharded, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        # The gather writes the eight-core activation shard the index and main
        # projections read; the tensor stays allocated through the layer (20 KB
        # per core), as the GDN and MoE gathers do.
        full_hidden = ttnn.all_gather(
            hidden_sharded,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=self.hidden_act_memory_config,
        )
        _require_shape(full_hidden, (1, 1, 1, HIDDEN_SIZE), "QSA hidden all-gather")
        self.mesh_contract.validate_tensor(full_hidden, placement=TensorPlacement.REPLICATED)
        return full_hidden

    def _index_projection(self, full_hidden, cos, sin):
        index_q_ws = ttnn.linear(
            full_hidden,
            self.weights.index_q,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.index_program_config,
            compute_kernel_config=self.compute_config,
        )
        raw_key_ws = ttnn.linear(
            full_hidden,
            self.weights.index_k,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.index_program_config,
            compute_kernel_config=self.compute_config,
        )
        index_q = ttnn.to_memory_config(index_q_ws, ttnn.DRAM_MEMORY_CONFIG)
        raw_key = ttnn.to_memory_config(raw_key_ws, ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(index_q_ws, raw_key_ws)
        self.mesh_contract.validate_tensor(index_q, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        self.mesh_contract.validate_tensor(raw_key, placement=TensorPlacement.REPLICATED)
        _require_shape(index_q, (1, 1, 1, INDEX_HEAD_DIM), "local index query")
        _require_shape(raw_key, (1, 1, 1, INDEX_HEAD_DIM), "raw index key")
        _retag_tensor(index_q, reference=full_hidden, shard_dim=1)

        normalized = ttnn.rms_norm(
            index_q,
            epsilon=self.rms_norm_eps,
            weight=self.weights.index_q_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(index_q)
        rotated = apply_partial_rope_prefill(normalized, cos, sin, INDEX_QUERY_HEADS_PER_DEVICE, ROPE_DIM)
        _deallocate(normalized)
        _retag_tensor(rotated, reference=full_hidden, shard_dim=1)
        self.mesh_contract.validate_tensor(rotated, placement=TensorPlacement.HEAD_SHARDED, shard_dim=1)
        return rotated, raw_key

    def _append_raw_index_key(
        self,
        state: Qwen38TTNNQSAState,
        raw_key,
        *,
        block_start_cos,
        block_start_sin,
    ) -> tuple[Any | None, int, int]:
        count = state.raw_tail_count + 1
        if count == COMPRESS_RATIO:
            if block_start_cos is None or block_start_sin is None:
                raise ValueError("closing a four-token QSA index block requires RoPE cos/sin for its first token")
            self._validate_rope(block_start_cos, block_start_sin, "QSA block-start RoPE")
        if state.raw_index_tail is None:
            combined = raw_key
        else:
            combined = ttnn.concat([state.raw_index_tail, raw_key], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _deallocate(raw_key)
        _require_shape(combined, (1, 1, count, INDEX_HEAD_DIM), "appended raw index tail")
        self.mesh_contract.validate_tensor(combined, placement=TensorPlacement.REPLICATED)
        if count < COMPRESS_RATIO:
            return combined, count, state.compressed_blocks

        pooled = ttnn.mean(
            combined,
            dim=2,
            keepdim=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        normalized = ttnn.rms_norm(
            pooled,
            epsilon=self.rms_norm_eps,
            weight=self.weights.index_k_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(pooled)
        rotated = apply_partial_rope_prefill(normalized, block_start_cos, block_start_sin, 1, ROPE_DIM)
        _deallocate(normalized)
        self.mesh_contract.validate_tensor(rotated, placement=TensorPlacement.REPLICATED)
        result = ttnn.update_cache(state.compressed_index_cache, rotated, state.compressed_blocks)
        if _tensor_key(result) != _tensor_key(state.compressed_index_cache):
            raise RuntimeError("compressed QSA index update was not in place")
        _deallocate(rotated, combined)
        return None, 0, state.compressed_blocks + 1

    def _main_projection(self, full_hidden, cos, sin):
        qg_ws = ttnn.linear(
            full_hidden,
            self.weights.qg,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.qg_program_config,
            compute_kernel_config=self.compute_config,
        )
        k_ws = ttnn.linear(
            full_hidden,
            self.weights.k_pair_grouped,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.kv_program_config,
            compute_kernel_config=self.compute_config,
        )
        v_ws = ttnn.linear(
            full_hidden,
            self.weights.v_pair_grouped,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.kv_program_config,
            compute_kernel_config=self.compute_config,
        )
        qg = ttnn.to_memory_config(qg_ws, ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.to_memory_config(k_ws, ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.to_memory_config(v_ws, ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(qg_ws, k_ws, v_ws)
        self.mesh_contract.validate_tensor(qg, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        self.mesh_contract.validate_tensor(k, placement=TensorPlacement.KV_PAIR_GROUPED, shard_dim=3)
        self.mesh_contract.validate_tensor(v, placement=TensorPlacement.KV_PAIR_GROUPED, shard_dim=3)
        _require_shape(qg, (1, 1, 1, 2 * LOCAL_QUERY_WIDTH), "local QSA query/gate")
        _require_shape(k, (1, 1, 1, HEAD_DIM), "local pair-grouped K")
        _require_shape(v, (1, 1, 1, HEAD_DIM), "local pair-grouped V")

        # One head per tile row: the [q_h | gate_h] pairs land as [1, heads, 1,
        # 512] (one tiled-reshape program) and the two tile-aligned last-dim
        # slices are q and gate in the [1, heads, 1, 256] shape the norm, RoPE
        # and gate multiply take; no transposes.
        qg_heads = ttnn.reshape(qg, (1, QUERY_HEADS_PER_DEVICE, 1, 2 * HEAD_DIM))
        _deallocate(qg)
        _retag_tensor(qg_heads, reference=full_hidden, shard_dim=1)
        q = ttnn.slice(
            qg_heads,
            (0, 0, 0, 0),
            (1, QUERY_HEADS_PER_DEVICE, 1, HEAD_DIM),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.slice(
            qg_heads,
            (0, 0, 0, HEAD_DIM),
            (1, QUERY_HEADS_PER_DEVICE, 1, 2 * HEAD_DIM),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if _tensor_key(qg_heads) in (_tensor_key(q), _tensor_key(gate)):
            raise RuntimeError("QSA query/gate half slice returned its input; expected two copies")
        _deallocate(qg_heads)
        _retag_tensor(q, reference=full_hidden, shard_dim=1)
        _retag_tensor(gate, reference=full_hidden, shard_dim=1)

        _retag_tensor(k, reference=full_hidden, shard_dim=1)
        _retag_tensor(v, reference=full_hidden, shard_dim=1)

        q_norm = ttnn.rms_norm(
            q,
            epsilon=self.rms_norm_eps,
            weight=self.weights.q_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        k_norm = ttnn.rms_norm(
            k,
            epsilon=self.rms_norm_eps,
            weight=self.weights.k_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(q, k)
        q_rotated = apply_partial_rope_prefill(q_norm, cos, sin, QUERY_HEADS_PER_DEVICE, ROPE_DIM)
        k_rotated = apply_partial_rope_prefill(k_norm, cos, sin, 1, ROPE_DIM)
        _deallocate(q_norm, k_norm)
        _retag_tensor(q_rotated, reference=full_hidden, shard_dim=1)
        _retag_tensor(k_rotated, reference=full_hidden, shard_dim=1)
        for value in (k_rotated, v):
            self.mesh_contract.validate_tensor(value, placement=TensorPlacement.KV_PAIR_GROUPED, shard_dim=1)
        return q_rotated, gate, k_rotated, v

    def _append_packed_kv(
        self,
        state: Qwen38TTNNQSAState,
        key,
        value,
    ) -> tuple[Any, bool]:
        packed_tiled = ttnn.concat([value, key], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(value, key)
        _retag_tensor(packed_tiled, reference=state.packed_kv_cache, shard_dim=1)
        packed = ttnn.to_layout(packed_tiled, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(packed_tiled)
        _retag_tensor(packed, reference=state.packed_kv_cache, shard_dim=1)
        _require_shape(packed, (1, 1, 1, 2 * HEAD_DIM), "current packed QSA KV")

        row = state.next_position % CACHE_WRITE_ROWS
        pieces = []
        temporaries = []
        if row:
            prefix = ttnn.slice(
                state.kv_staging,
                (0, 0, 0, 0),
                (1, 1, row, 2 * HEAD_DIM),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            pieces.append(prefix)
            temporaries.append(prefix)
        pieces.append(packed)
        if row + 1 < CACHE_WRITE_ROWS:
            suffix = ttnn.slice(
                state.kv_staging,
                (0, 0, row + 1, 0),
                (1, 1, CACHE_WRITE_ROWS, 2 * HEAD_DIM),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            pieces.append(suffix)
            temporaries.append(suffix)
        stage = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(*temporaries)
        if _tensor_key(stage) != _tensor_key(packed):
            _deallocate(packed)
        _retag_tensor(stage, reference=state.packed_kv_cache, shard_dim=1)
        _require_shape(stage, (1, 1, CACHE_WRITE_ROWS, 2 * HEAD_DIM), "filled QSA KV staging")
        self.mesh_contract.validate_tensor(stage, placement=TensorPlacement.KV_PAIR_GROUPED, shard_dim=1)

        block_start = state.next_position - row
        result = ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            state.packed_kv_cache,
            stage,
            0,  # slot_idx
            0,  # layer_idx inside this one-layer cache
            1,  # num_layers
            block_start,
            STAGING_AXIS,
        )
        if _tensor_key(result) != _tensor_key(state.packed_kv_cache):
            raise RuntimeError("row-major QSA KV update was not in place")

        if row + 1 == CACHE_WRITE_ROWS:
            _deallocate(stage)
            next_stage = self._take_next_kv_staging()
            return next_stage, True
        return stage, True

    def _front_padded_query(self, index_query, query_row: int):
        # Device TILE padding only permits trailing padding.  Stage through
        # ROW_MAJOR, whose device pad path supports the exact front placement,
        # then restore TILE for indexer_score_dsa.  One pad program per row.
        query_row_major = ttnn.to_layout(
            index_query,
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _retag_tensor(query_row_major, reference=index_query, shard_dim=1)
        query_padded_row_major = ttnn.pad(
            query_row_major,
            [(0, 0), (0, 0), (query_row, ttnn.TILE_SIZE - query_row - 1), (0, 0)],
            0.0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _deallocate(query_row_major)
        query_padded = ttnn.to_layout(
            query_padded_row_major,
            ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _deallocate(query_padded_row_major)
        _retag_tensor(query_padded, reference=index_query, shard_dim=1)
        return query_padded

    def _score_complete_blocks(self, index_query, state: Qwen38TTNNQSAState, complete_blocks: int):
        if not 1 <= complete_blocks <= self.allocated_compressed_blocks:
            raise ValueError(f"complete QSA block count is invalid: {complete_blocks}")
        valid_padded = math.ceil(complete_blocks / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        query_view = valid_padded + ttnn.TILE_SIZE <= self.allocated_compressed_blocks
        if query_view:
            # The query's own tile, presented as the 32-row logical shape the
            # kernel validates (a buffer view; no kernel).  With the query in
            # row 0 its causal diagonal tile is the one just past every
            # complete block, so the window ends one tile later.  Rows 1-31
            # hold whatever the tile carried: every kernel stage is per
            # element or per row, so row 0's scores are the bytes the padded
            # placement produced.
            query_row = 0
            chunk_start = valid_padded
            kv_len = valid_padded + ttnn.TILE_SIZE
            tile_shape = ttnn.Shape((1, INDEX_QUERY_HEADS_PER_DEVICE, ttnn.TILE_SIZE, INDEX_HEAD_DIM))
            query_tile = ttnn.reshape(index_query, tile_shape, tile_shape)
            _retag_tensor(query_tile, reference=index_query, shard_dim=1)
        else:
            # No tile past the last cache row: front-pad the query into its
            # causal row of the final chunk instead.
            chunk_start = valid_padded - ttnn.TILE_SIZE
            kv_len = valid_padded
            query_row = complete_blocks - 1 - chunk_start
            query_tile = self._front_padded_query(index_query, query_row)
        _require_shape(
            query_tile,
            (1, INDEX_QUERY_HEADS_PER_DEVICE, ttnn.TILE_SIZE, INDEX_HEAD_DIM),
            "tile-padded index query",
        )
        local_scores = ttnn.experimental.indexer_score_dsa(
            query_tile,
            state.compressed_index_cache,
            self.index_gate,
            chunk_start_idx=chunk_start,
            compute_kernel_config=self.indexer_compute_config,
            kv_len=kv_len,
            # Q is sharded in heads, not sequence.  Naming the size-one mesh
            # axis gives every TP coordinate the same causal query position;
            # [] would incorrectly add the flat TP rank to that position.
            seq_shard_axes=[STAGING_AXIS],
        )
        if not query_view:
            _deallocate(query_tile)
        score_row = ttnn.slice(
            local_scores,
            (0, 0, query_row, 0),
            (1, 1, query_row + 1, self.allocated_compressed_blocks),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _deallocate(local_scores)
        self.mesh_contract.mark_local_partial(
            score_row,
            replicated_reference=state.compressed_index_cache,
            expected_shape=(1, 1, 1, self.allocated_compressed_blocks),
        )
        scores = ttnn.all_reduce(
            score_row,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.collective_topology,
        )
        _deallocate(score_row)
        self.mesh_contract.validate_tensor(scores, placement=TensorPlacement.REPLICATED)
        # All BLOCK_TOPK slots are expanded at fixed width; slots past the
        # selected count (top-k sentinels below the valid length) are masked
        # out of every row built from the result, so no per-position slice.
        block_ids_rm = ttnn.experimental.topk_large_indices(
            scores,
            k=BLOCK_TOPK,
            valid_length=complete_blocks,
        )
        _deallocate(scores)
        if block_ids_rm.dtype != ttnn.uint32:
            raise RuntimeError("topk_large_indices must return UINT32 block IDs")
        if block_ids_rm.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError("topk_large_indices must return ROW_MAJOR block IDs")
        _require_shape(block_ids_rm, (1, 1, 1, BLOCK_TOPK), "top-k QSA block IDs")

        starts = ttnn.bitwise_left_shift(block_ids_rm, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(block_ids_rm)
        # Four copies of every start: two half-budget ROW_MAJOR gathers (see
        # EXPANSION_GATHER_WIDTH) and one aligned concat; a UINT32 last-dim
        # repeat_interleave is not codegen-eligible and lowers to eight ops.
        halves = [
            ttnn.gather(starts, 3, rep_index, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for rep_index in (self.rep_index_lo, self.rep_index_hi)
        ]
        _deallocate(starts)
        repeated = ttnn.concat(halves, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(*halves)
        expanded = ttnn.add(repeated, self.block_offsets, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(repeated)
        _require_shape(expanded, (1, 1, 1, TOKEN_BUDGET), "expanded QSA complete-block indices")
        if expanded.dtype != ttnn.uint32 or expanded.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError("expanded QSA indices must remain UINT32 ROW_MAJOR")
        self.mesh_contract.validate_tensor(expanded, placement=TensorPlacement.REPLICATED)
        return expanded, min(BLOCK_TOPK, complete_blocks) * COMPRESS_RATIO

    def _materialize_selection(
        self,
        *,
        state: Qwen38TTNNQSAState,
        source_view_id: int,
        source_position: int,
        tail_start: int,
        complete_indices,
        complete_token_count: int,
        owns_complete_indices: bool,
    ) -> Qwen38TTNNQSASelection:
        context_length = state.next_position + 1
        if not 0 <= tail_start <= context_length:
            raise ValueError(f"QSA tail start {tail_start} is outside context length {context_length}")
        if complete_token_count % COMPRESS_RATIO or not 0 <= complete_token_count <= TOKEN_BUDGET:
            raise ValueError(f"invalid complete-token selection count {complete_token_count}")
        if (complete_indices is None) != (complete_token_count == 0):
            raise ValueError("complete QSA index tensor/count disagree")
        if owns_complete_indices and complete_indices is None:
            raise ValueError("an empty QSA complete-block selection cannot own an index tensor")
        if complete_indices is None:
            if tail_start:
                raise ValueError("a QSA selection without scored blocks must start its tail at token 0")
            valid_count = context_length
        else:
            _require_shape(complete_indices, (1, 1, 1, TOKEN_BUDGET), "reused complete QSA indices")
            if complete_indices.dtype != ttnn.uint32 or complete_indices.layout != ttnn.ROW_MAJOR_LAYOUT:
                raise RuntimeError("reused complete QSA indices must be UINT32 ROW_MAJOR")
            self.mesh_contract.validate_tensor(complete_indices, placement=TensorPlacement.REPLICATED)
            tail_count = context_length - tail_start
            if tail_count > MAX_REUSE_TAIL:
                raise ValueError(f"reused QSA tail has {tail_count} tokens; at most {MAX_REUSE_TAIL} are admitted")
            valid_count = complete_token_count + tail_count
        if valid_count <= 0 or valid_count > MAX_SELECTED_TOKENS:
            raise RuntimeError(f"materialized QSA selection has invalid length {valid_count}")

        if complete_indices is None:
            # Every causal token, natural order: the per-token shared row.
            sparse_indices = self._natural_row(context_length)
        else:
            # [expanded | sentinel_pad] masked to the selected slots, then the
            # tail ids and sentinels OR'd in: three ops per layer, the two
            # mask rows built once per token.  Same bytes as the old
            # [complete | tail | sentinels] concat, without its unaligned
            # pieces (the transpose fallback of concat.cpp).
            keep, tail_sentinel = self._template_masks(complete_token_count, tail_start, valid_count)
            template = ttnn.concat([complete_indices, self.sentinel_pad], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            kept = ttnn.bitwise_and(template, keep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _deallocate(template)
            sparse_indices = ttnn.bitwise_or(kept, tail_sentinel, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _deallocate(kept)
        _require_shape(
            sparse_indices,
            (1, 1, 1, SPARSE_INDEX_CAPACITY),
            "padded QSA sparse indices",
        )
        if sparse_indices.dtype != ttnn.uint32 or sparse_indices.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError("sparse QSA indices must be UINT32 ROW_MAJOR")
        self.mesh_contract.validate_tensor(sparse_indices, placement=TensorPlacement.REPLICATED)
        return Qwen38TTNNQSASelection(
            layer_index=self.layer_index,
            epoch=state.epoch,
            source_view_id=source_view_id,
            source_position=source_position,
            tail_start=tail_start,
            complete_indices=complete_indices,
            complete_token_count=complete_token_count,
            owns_complete_indices=owns_complete_indices,
            sparse_indices=sparse_indices,
            valid_token_count=valid_count,
            owns_sparse_indices=complete_indices is not None,
        )

    def _select(
        self,
        index_query,
        state: Qwen38TTNNQSAState,
        complete_blocks: int,
        reuse_selection: Qwen38TTNNQSASelection | None,
    ) -> Qwen38TTNNQSASelection:
        if reuse_selection is not None:
            self._validate_reuse_selection(reuse_selection, state)
            return self._materialize_selection(
                state=state,
                source_view_id=reuse_selection.source_view_id,
                source_position=reuse_selection.source_position,
                tail_start=reuse_selection.tail_start,
                complete_indices=reuse_selection.complete_indices,
                complete_token_count=reuse_selection.complete_token_count,
                owns_complete_indices=False,
            )

        if qsa_natural_row_regime(complete_blocks, regime_split=self.regime_split):
            complete_indices, complete_token_count, tail_start = None, 0, 0
        else:
            complete_indices, complete_token_count = self._score_complete_blocks(index_query, state, complete_blocks)
            tail_start = complete_blocks * COMPRESS_RATIO
        return self._materialize_selection(
            state=state,
            source_view_id=state.view_id,
            source_position=state.next_position,
            tail_start=tail_start,
            complete_indices=complete_indices,
            complete_token_count=complete_token_count,
            owns_complete_indices=complete_indices is not None,
        )

    def _sparse_value_attention(self, query, gate, sparse_indices, state):
        # q = [zeros(256) | Q(256)] per local head (tile-aligned concat), untilized,
        # then the 26 zero heads appended in ROW_MAJOR, whose device pad is one
        # program on any dim (a TILE pad on the head dim is FillPad + Pad).  Same
        # bytes as the old pad / zeros_like / concat / untilize chain: +0.0 bf16
        # everywhere Q is not.
        sparse_query_tiled = ttnn.concat([self.zero_value_half, query], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(query)
        _retag_tensor(sparse_query_tiled, reference=state.packed_kv_cache, shard_dim=1)
        _require_shape(sparse_query_tiled, (1, QUERY_HEADS_PER_DEVICE, 1, 2 * HEAD_DIM), "local sparse QSA query")
        sparse_query_row_major = ttnn.to_layout(
            sparse_query_tiled, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        _deallocate(sparse_query_tiled)
        sparse_query = ttnn.pad(
            sparse_query_row_major,
            [(0, 0), (0, 32 - QUERY_HEADS_PER_DEVICE), (0, 0), (0, 0)],
            0.0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _deallocate(sparse_query_row_major)
        _retag_tensor(sparse_query, reference=state.packed_kv_cache, shard_dim=1)
        _require_shape(sparse_query, (1, 32, 1, 2 * HEAD_DIM), "padded sparse QSA query")
        if sparse_query.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError(f"padded sparse QSA query must be ROW_MAJOR, got {tensor_metadata(sparse_query)}")

        sparse_output = ttnn.transformer.sparse_sdpa(
            sparse_query,
            state.packed_kv_cache,
            sparse_indices,
            HEAD_DIM,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=HEAD_DIM**-0.5,
            k_chunk_size=ttnn.TILE_SIZE,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(sparse_query)
        local = ttnn.slice(
            sparse_output,
            (0, 0, 0, 0),
            (1, QUERY_HEADS_PER_DEVICE, 1, HEAD_DIM),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _deallocate(sparse_output)
        local_tiled = ttnn.to_layout(local, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(local)
        _retag_tensor(local_tiled, reference=gate, shard_dim=1)
        activated_gate = ttnn.sigmoid(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gated = ttnn.mul(local_tiled, activated_gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(local_tiled, activated_gate, gate)
        # Head-major flatten: the [1, heads, 1, 256] row-major order is the
        # [1, 1, 1, 1536] row, so one tiled-reshape program replaces the
        # transpose(1, 2) + reshape pair (same bytes).
        local_flat = ttnn.reshape(gated, (1, 1, 1, LOCAL_QUERY_WIDTH))
        if _tensor_key(local_flat) != _tensor_key(gated):
            _deallocate(gated)
        _retag_tensor(local_flat, reference=state.packed_kv_cache, shard_dim=3)
        self.mesh_contract.validate_tensor(local_flat, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        return local_flat

    def forward_decode(
        self,
        hidden_sharded,
        state: Qwen38TTNNQSAState,
        *,
        cos,
        sin,
        block_start_cos=None,
        block_start_sin=None,
        position: int | None = None,
        reuse_selection: Qwen38TTNNQSASelection | None = None,
        retain_input_state: bool = False,
    ) -> Qwen38TTNNQSAResult:
        """Decode one global token and return a rollback-capable next state.

        ``reuse_selection`` freezes the selected complete blocks.  The module
        appends every causal token from that selection's ``tail_start`` through
        this call's position, which is the required four-step MTP reuse rule.
        """

        self._validate_state(state)
        if state.next_position == self.allocated_context:
            raise ValueError(
                f"QSA cache is full at allocated context {self.allocated_context}; "
                f"semantic model limit remains {MAX_CONTEXT}"
            )
        if state.view_id in self._protected_views and not retain_input_state:
            raise RuntimeError(
                "protected QSA checkpoint cannot be consumed; pass retain_input_state=True for the first branch step"
            )
        if retain_input_state:
            if state.view_id not in self._protected_views:
                raise RuntimeError("retaining QSA input state requires checkpoint_state(state) first")
            epoch_views = {view_id for view_id, epoch in self._live_views.items() if epoch == state.epoch}
            if epoch_views != {state.view_id}:
                raise RuntimeError(
                    "a QSA checkpoint can start only one live branch; "
                    f"epoch {state.epoch} already has views {sorted(epoch_views)}"
                )
        if reuse_selection is not None:
            # Validate provenance/topology before mutating either append-only
            # cache, so a malformed external selection fails without changing
            # the recoverable state.
            self._validate_reuse_selection(reuse_selection, state)
        actual_position = state.next_position if position is None else position
        if actual_position != state.next_position:
            raise ValueError(
                f"QSA decode position {actual_position} does not match state position {state.next_position}"
            )
        self._validate_rope(cos, sin, "QSA current RoPE")

        full_hidden = self._all_gather_hidden(hidden_sharded)
        index_query, raw_key = self._index_projection(full_hidden, cos, sin)
        next_raw_tail, next_raw_count, next_compressed_blocks = self._append_raw_index_key(
            state,
            raw_key,
            block_start_cos=block_start_cos,
            block_start_sin=block_start_sin,
        )

        # Selection observes the just-appended key.  A temporary metadata view
        # exposes that new logical length while retaining the same caches/stage.
        selection_state = Qwen38TTNNQSAState(
            layer_index=state.layer_index,
            epoch=state.epoch,
            view_id=state.view_id,
            next_position=state.next_position,
            compressed_blocks=next_compressed_blocks,
            raw_tail_count=next_raw_count,
            raw_index_tail=next_raw_tail,
            packed_kv_cache=state.packed_kv_cache,
            compressed_index_cache=state.compressed_index_cache,
            kv_staging=state.kv_staging,
            kv_staging_owned=state.kv_staging_owned,
        )
        selection = self._select(index_query, selection_state, next_compressed_blocks, reuse_selection)
        _deallocate(index_query)

        query, gate, key, value = self._main_projection(full_hidden, cos, sin)
        next_staging, next_staging_owned = self._append_packed_kv(state, key, value)
        local_attention = self._sparse_value_attention(query, gate, selection.sparse_indices, state)
        attention_ws = ttnn.to_memory_config(local_attention, self.out_act_memory_config)
        _deallocate(local_attention)
        local_partial_ws = ttnn.linear(
            attention_ws,
            self.weights.out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.out_program_config,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(attention_ws)
        local_partial = ttnn.to_memory_config(local_partial_ws, ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(local_partial_ws)
        self.mesh_contract.mark_local_partial(
            local_partial,
            replicated_reference=full_hidden,
            expected_shape=(1, 1, 1, HIDDEN_SIZE),
        )
        # ttnn.reduce_scatter accumulates in its operand dtype at every hop of
        # the four-coordinate line, so BF16 partials are rounded to BF16 three
        # times before a shard is written.  The position-one device-sum check
        # measured that hop-wise rounding at 1.5x the certified single-rounding
        # bound.  Widen the four BF16 partials to FP32 (exact), let the
        # collective sum them in FP32 (the runtime selects HiFi4 with FP32 DEST
        # accumulation for FLOAT32 operands), and round the H/4 shard to BF16
        # exactly once.  The result matches an FP32 host sum of the same BF16
        # partials up to that one final rounding.
        local_partial_fp32 = ttnn.typecast(local_partial, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(local_partial)
        self.mesh_contract.mark_local_partial(
            local_partial_fp32,
            replicated_reference=full_hidden,
            expected_shape=(1, 1, 1, HIDDEN_SIZE),
        )
        output_fp32 = ttnn.reduce_scatter(
            local_partial_fp32,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.collective_topology,
        )
        _deallocate(local_partial_fp32)
        self.mesh_contract.mark_collective_shard(
            output_fp32,
            replicated_reference=full_hidden,
            shard_dim=3,
            expected_local_shape=(1, 1, 1, HIDDEN_SIZE // TP_SIZE),
        )
        if output_fp32.dtype != ttnn.float32:
            raise RuntimeError(f"QSA output reduce_scatter must sum FP32 partials, got {tensor_metadata(output_fp32)}")
        output = ttnn.typecast(output_fp32, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _retag_tensor(output, reference=output_fp32, shard_dim=3)
        _deallocate(output_fp32)
        _require_shape(output, (1, 1, 1, HIDDEN_SIZE // TP_SIZE), "QSA output projection")
        if output.dtype != ttnn.bfloat16:
            raise RuntimeError(f"QSA output projection must return BF16, got {tensor_metadata(output)}")
        self.mesh_contract.validate_tensor(output, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        _deallocate(full_hidden)

        next_view_id = self._allocate_view_id(state.epoch)
        if reuse_selection is None:
            selection = replace(selection, source_view_id=next_view_id)
        next_state = Qwen38TTNNQSAState(
            layer_index=self.layer_index,
            epoch=state.epoch,
            view_id=next_view_id,
            next_position=state.next_position + 1,
            compressed_blocks=next_compressed_blocks,
            raw_tail_count=next_raw_count,
            raw_index_tail=next_raw_tail,
            packed_kv_cache=state.packed_kv_cache,
            compressed_index_cache=state.compressed_index_cache,
            kv_staging=next_staging,
            kv_staging_owned=next_staging_owned,
            last_selection=selection,
        )
        if not retain_input_state:
            next_state, preserve_complete = self._transfer_complete_selection(state, next_state)
            self._consume_view(state, preserve_complete=preserve_complete)
            selection = next_state.last_selection
        return Qwen38TTNNQSAResult(output, next_state, selection)

    def _write_compressed_index_generic(
        self,
        state: Qwen38TTNNQSAGenericState,
        raw_key,
        position: Qwen38TTNNQSAPositionInputs,
        block_start_cos,
        block_start_sin,
    ) -> None:
        # Ring slot P % 4 <- raw key (exact one-hot select, in place).
        kept = ttnn.multiply(state.raw_key_ring, position.ring_keep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        placed = ttnn.multiply(raw_key, position.ring_hit, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(raw_key)
        _require_shape(placed, (1, 1, CACHE_WRITE_ROWS, INDEX_HEAD_DIM), "placed raw index key")
        ring = ttnn.add(kept, placed, output_tensor=state.raw_key_ring, fast_and_approximate_mode=False)
        if _tensor_key(ring) != _tensor_key(state.raw_key_ring):
            raise RuntimeError("raw QSA index key ring update was not in place")
        _deallocate(kept, placed)

        # Rows 4-31 are exactly zero, so the 1/4-scaled sum over 32 rows is the
        # same reduce as today's mean over the four-row concat.  The entry is
        # only complete (and unmasked) at P % 4 == 3; earlier writes of the
        # same block are finite garbage hidden by indexer_neg_mask.
        pooled = ttnn.sum(
            state.raw_key_ring,
            dim=2,
            keepdim=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
            scalar=1.0 / COMPRESS_RATIO,
        )
        _require_shape(pooled, (1, 1, 1, INDEX_HEAD_DIM), "pooled raw index key")
        normalized = ttnn.rms_norm(
            pooled,
            epsilon=self.rms_norm_eps,
            weight=self.weights.index_k_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(pooled)
        rotated = apply_partial_rope_prefill(normalized, block_start_cos, block_start_sin, 1, ROPE_DIM)
        _deallocate(normalized)
        self.mesh_contract.validate_tensor(rotated, placement=TensorPlacement.REPLICATED)
        rotated_sharded = ttnn.to_memory_config(rotated, self.compressed_row_memory_config)
        _deallocate(rotated)
        result = ttnn.experimental.paged_update_cache(
            state.compressed_index_cache,
            rotated_sharded,
            update_idxs_tensor=position.block_index_i32,
        )
        if _tensor_key(result) != _tensor_key(state.compressed_index_cache):
            raise RuntimeError("compressed QSA index update was not in place")
        _deallocate(rotated_sharded)

    def _score_blocks_generic(
        self, index_query, state: Qwen38TTNNQSAGenericState, position: Qwen38TTNNQSAPositionInputs
    ):
        # The query's own tile as the 32-row shape the kernel validates (a
        # buffer view, no kernel; the same view _score_complete_blocks takes).
        # Row 0's causal diagonal is the fixed window past the last resident
        # block, so every resident column is visible to it; rows 1-31 carry the
        # tile's padding and are sliced away (every kernel stage is per element
        # or per row).  The additive mask hides incomplete blocks.
        tile_shape = ttnn.Shape((1, INDEX_QUERY_HEADS_PER_DEVICE, ttnn.TILE_SIZE, INDEX_HEAD_DIM))
        query_tile = ttnn.reshape(index_query, tile_shape, tile_shape)
        _retag_tensor(query_tile, reference=index_query, shard_dim=1)
        _require_shape(
            query_tile,
            (1, INDEX_QUERY_HEADS_PER_DEVICE, ttnn.TILE_SIZE, INDEX_HEAD_DIM),
            "tile view of the index query",
        )
        # The window [chunk_start, chunk_start + 32) ends at the cache's last
        # row, so the kernel's deduced kv_len (T) is the window's end.
        local_scores = ttnn.experimental.indexer_score_dsa(
            query_tile,
            state.compressed_index_cache,
            self.index_gate,
            chunk_start_idx=self.indexer_chunk_start,
            compute_kernel_config=self.indexer_compute_config,
            seq_shard_axes=[STAGING_AXIS],
        )
        # query_tile shares index_query's buffer; the caller releases index_query.
        score_row = ttnn.slice(
            local_scores,
            (0, 0, 0, 0),
            (1, 1, 1, self.allocated_compressed_blocks),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _deallocate(local_scores)
        self.mesh_contract.mark_local_partial(
            score_row,
            replicated_reference=state.compressed_index_cache,
            expected_shape=(1, 1, 1, self.allocated_compressed_blocks),
        )
        scores = ttnn.all_reduce(
            score_row,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.collective_topology,
        )
        _deallocate(score_row)
        self.mesh_contract.validate_tensor(scores, placement=TensorPlacement.REPLICATED)
        masked = ttnn.add(
            scores,
            position.indexer_neg_mask,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            fast_and_approximate_mode=False,
        )
        _deallocate(scores)
        _require_shape(masked, (1, 1, 1, self.allocated_compressed_blocks), "masked QSA block scores")
        if masked.dtype != ttnn.bfloat16 or masked.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError(f"masked QSA block scores must be BF16 ROW_MAJOR, got {tensor_metadata(masked)}")
        return masked

    def _materialize_row_generic(self, masked_scores, position: Qwen38TTNNQSAPositionInputs):
        # Fixed k; the valid blocks form the sorted prefix because every masked
        # score is below any ReLU score.  Slots >= lo are then overwritten by
        # the position-derived row of tail ids and sentinels.
        block_ids = ttnn.experimental.topk_large_indices(masked_scores, k=BLOCK_TOPK)
        _deallocate(masked_scores)
        _require_shape(block_ids, (1, 1, 1, BLOCK_TOPK), "top-k QSA block IDs")
        if block_ids.dtype != ttnn.uint32 or block_ids.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError(
                f"topk_large_indices must return UINT32 ROW_MAJOR block IDs, got {tensor_metadata(block_ids)}"
            )
        starts = ttnn.bitwise_left_shift(block_ids, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(block_ids)
        repeated = ttnn.repeat_interleave(starts, repeats=COMPRESS_RATIO, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(starts)
        expanded = ttnn.add(repeated, self.block_offsets, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(repeated)
        _require_shape(expanded, (1, 1, 1, TOKEN_BUDGET), "expanded QSA block indices")
        template = ttnn.concat([expanded, self.sentinel_pad], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(expanded)
        kept = ttnn.bitwise_and(template, position.row_keep_bits, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(template)
        sparse_indices = ttnn.bitwise_or(kept, position.row_fill, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(kept)
        _require_shape(sparse_indices, (1, 1, 1, SPARSE_INDEX_CAPACITY), "generic QSA sparse indices")
        if sparse_indices.dtype != ttnn.uint32 or sparse_indices.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError(f"sparse QSA indices must be UINT32 ROW_MAJOR, got {tensor_metadata(sparse_indices)}")
        self.mesh_contract.validate_tensor(sparse_indices, placement=TensorPlacement.REPLICATED)
        return sparse_indices

    def _write_packed_kv_generic(
        self,
        state: Qwen38TTNNQSAGenericState,
        key,
        value,
        position: Qwen38TTNNQSAPositionInputs,
    ) -> None:
        packed_tiled = ttnn.concat([value, key], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(value, key)
        _require_shape(packed_tiled, (1, 1, 1, 2 * HEAD_DIM), "current packed QSA KV")
        # Staging row P % 32 <- packed row (exact one-hot select, in place on
        # the TILE staging), then one untilize feeds the ROW_MAJOR cache write
        # at row P & ~31, read on device from the metadata tensors.
        kept = ttnn.multiply(state.kv_staging, position.kv_row_keep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        placed = ttnn.multiply(packed_tiled, position.kv_row_hit, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(packed_tiled)
        _require_shape(placed, (1, 1, CACHE_WRITE_ROWS, 2 * HEAD_DIM), "placed packed QSA KV")
        staged = ttnn.add(kept, placed, output_tensor=state.kv_staging, fast_and_approximate_mode=False)
        if _tensor_key(staged) != _tensor_key(state.kv_staging):
            raise RuntimeError("QSA KV staging update was not in place")
        _deallocate(kept, placed)
        stage_row_major = ttnn.to_layout(state.kv_staging, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _retag_tensor(stage_row_major, reference=state.packed_kv_cache, shard_dim=1)
        _require_shape(stage_row_major, (1, 1, CACHE_WRITE_ROWS, 2 * HEAD_DIM), "filled QSA KV staging")
        self.mesh_contract.validate_tensor(stage_row_major, placement=TensorPlacement.KV_PAIR_GROUPED, shard_dim=1)
        result = ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            state.packed_kv_cache,
            stage_row_major,
            self.slot_zero,
            position.kv_block_start,
            0,  # layer_idx inside this one-layer cache
            1,  # num_layers
            STAGING_AXIS,
        )
        if _tensor_key(result) != _tensor_key(state.packed_kv_cache):
            raise RuntimeError("row-major QSA KV update was not in place")
        _deallocate(stage_row_major)

    def _project_output(self, local_attention, full_hidden):
        # Same o_proj boundary as forward_decode, whose text is pinned verbatim
        # by test_ttnn_components_static (FP32 reduce_scatter, one BF16 rounding).
        attention_ws = ttnn.to_memory_config(local_attention, self.out_act_memory_config)
        _deallocate(local_attention)
        local_partial_ws = ttnn.linear(
            attention_ws,
            self.weights.out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.out_program_config,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(attention_ws)
        local_partial = ttnn.to_memory_config(local_partial_ws, ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(local_partial_ws)
        self.mesh_contract.mark_local_partial(
            local_partial,
            replicated_reference=full_hidden,
            expected_shape=(1, 1, 1, HIDDEN_SIZE),
        )
        local_partial_fp32 = ttnn.typecast(local_partial, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(local_partial)
        self.mesh_contract.mark_local_partial(
            local_partial_fp32,
            replicated_reference=full_hidden,
            expected_shape=(1, 1, 1, HIDDEN_SIZE),
        )
        output_fp32 = ttnn.reduce_scatter(
            local_partial_fp32,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.collective_topology,
        )
        _deallocate(local_partial_fp32)
        self.mesh_contract.mark_collective_shard(
            output_fp32,
            replicated_reference=full_hidden,
            shard_dim=3,
            expected_local_shape=(1, 1, 1, HIDDEN_SIZE // TP_SIZE),
        )
        if output_fp32.dtype != ttnn.float32:
            raise RuntimeError(f"QSA output reduce_scatter must sum FP32 partials, got {tensor_metadata(output_fp32)}")
        output = ttnn.typecast(output_fp32, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _retag_tensor(output, reference=output_fp32, shard_dim=3)
        _deallocate(output_fp32)
        _require_shape(output, (1, 1, 1, HIDDEN_SIZE // TP_SIZE), "QSA output projection")
        if output.dtype != ttnn.bfloat16:
            raise RuntimeError(f"QSA output projection must return BF16, got {tensor_metadata(output)}")
        self.mesh_contract.validate_tensor(output, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        return output

    def forward_decode_generic(
        self,
        hidden_sharded,
        state: Qwen38TTNNQSAGenericState,
        *,
        cos,
        sin,
        block_start_cos,
        block_start_sin,
        position: Qwen38TTNNQSAPositionInputs,
    ):
        """Decode one token at the device-resident position; same op sequence for every position.

        ``state`` is mutated in place and keeps its addresses; ``position`` holds
        the per-token tensors of :func:`derive_qsa_position_inputs`; the RoPE
        rows are the table lookups at ``P`` and ``P & ~3``.  Returns the
        ``[1,1,1,640]`` BF16 TILE hidden-sharded output.  Like
        :meth:`forward_decode`, the input is left for the caller to release.
        """

        self._validate_generic_state(state)
        self._validate_rope(cos, sin, "QSA current RoPE")
        self._validate_rope(block_start_cos, block_start_sin, "QSA block-start RoPE")
        self._validate_position_inputs(position)

        full_hidden = self._all_gather_hidden(hidden_sharded)
        index_query, raw_key = self._index_projection(full_hidden, cos, sin)
        self._write_compressed_index_generic(state, raw_key, position, block_start_cos, block_start_sin)
        masked_scores = self._score_blocks_generic(index_query, state, position)
        _deallocate(index_query)
        sparse_indices = self._materialize_row_generic(masked_scores, position)

        query, gate, key, value = self._main_projection(full_hidden, cos, sin)
        self._write_packed_kv_generic(state, key, value, position)
        local_attention = self._sparse_value_attention(query, gate, sparse_indices, state)
        _deallocate(sparse_indices)
        output = self._project_output(local_attention, full_hidden)
        _deallocate(full_hidden)
        return output

    # ------------------------------------------------------------------ prefill chunk (32 rows)
    # forward_chunk_generic is forward_decode_generic over the 32 rows P .. P + 31 of one chunk (P % 32 == 0):
    # the same ops in the same order on 32-row operands, with the per-token staging/ring one-hots replaced
    # by whole-slab writes (one KV slab, eight complete compressed blocks) and the per-row selection built
    # from the chunk inputs.  The 1-row generic body above is untouched.

    def allocate_chunk_state(self) -> Qwen38TTNNQSAChunkState:
        epoch = self._next_epoch
        self._next_epoch += 1
        self._live_generic_epochs.add(epoch)
        return Qwen38TTNNQSAChunkState(
            layer_index=self.layer_index,
            epoch=epoch,
            kept_kv=self._allocate_pair_grouped((1, 1, CHUNK_ROWS, 2 * HEAD_DIM), layout=ttnn.TILE_LAYOUT),
            kept_raw=self._allocate_replicated_tile_zeros((1, 1, CHUNK_ROWS, INDEX_HEAD_DIM)),
        )

    def release_chunk_state(self, state: Qwen38TTNNQSAChunkState) -> None:
        self._validate_chunk_state(state)
        _deallocate(state.kept_kv, state.kept_raw)
        self._live_generic_epochs.remove(state.epoch)

    def _validate_chunk_state(self, state: Qwen38TTNNQSAChunkState) -> None:
        if state.layer_index != self.layer_index:
            raise ValueError(f"QSA chunk state belongs to layer {state.layer_index}, expected {self.layer_index}")
        if state.epoch not in self._live_generic_epochs:
            raise ValueError(f"QSA chunk state epoch {state.epoch} was not allocated by this module")
        _require_shape(state.kept_kv, (1, 1, CHUNK_ROWS, 2 * HEAD_DIM), "QSA chunk kept KV slab")
        _require_shape(state.kept_raw, (1, 1, CHUNK_ROWS, INDEX_HEAD_DIM), "QSA chunk kept raw keys")
        for label, tensor in (("QSA chunk kept KV slab", state.kept_kv), ("QSA chunk kept raw keys", state.kept_raw)):
            if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
                raise RuntimeError(f"{label} must be BF16 TILE, got {tensor_metadata(tensor)}")
        self.mesh_contract.validate_tensor(state.kept_kv, placement=TensorPlacement.KV_PAIR_GROUPED, shard_dim=1)
        self.mesh_contract.validate_tensor(state.kept_raw, placement=TensorPlacement.REPLICATED)

    def _validate_chunk_inputs(self, chunk: Qwen38TTNNQSAChunkInputs) -> None:
        blocks = self.allocated_compressed_blocks
        expected = (
            ("kv_block_start", chunk.kv_block_start, (1, 1, 1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            (
                "indexer_neg_mask",
                chunk.indexer_neg_mask,
                (1, 1, CHUNK_ROWS, blocks),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            ),
            (
                "row_keep_bits",
                chunk.row_keep_bits,
                (1, 1, CHUNK_ROWS, SPARSE_INDEX_CAPACITY),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
            ),
            ("row_fill", chunk.row_fill, (1, 1, CHUNK_ROWS, SPARSE_INDEX_CAPACITY), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            *(
                (f"block_index_i32[{i}]", tensor, (1,), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                for i, tensor in enumerate(chunk.block_index_i32)
            ),
        )
        if len(chunk.block_index_i32) != CHUNK_BLOCKS:
            raise ValueError(
                f"QSA chunk inputs carry {len(chunk.block_index_i32)} block indices, expected {CHUNK_BLOCKS}"
            )
        for name, tensor, shape, dtype, layout in expected:
            _require_shape(tensor, shape, f"QSA chunk input {name}")
            if tensor.dtype != dtype or tensor.layout != layout:
                raise RuntimeError(
                    f"QSA chunk input {name} must be {dtype} {layout} {list(shape)}, got {tensor_metadata(tensor)}"
                )

    def _validate_rope_rows(self, cos, sin, label: str) -> None:
        for name, tensor in (("cos", cos), ("sin", sin)):
            _require_shape(tensor, (1, 1, CHUNK_ROWS, ROPE_DIM), f"{label} {name}")
            if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
                raise RuntimeError(f"{label} {name} must be BF16 TILE, got {tensor_metadata(tensor)}")
            self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)

    def _all_gather_hidden_rows(self, hidden_rows):
        _require_shape(hidden_rows, (1, 1, CHUNK_ROWS, HIDDEN_SIZE // TP_SIZE), "QSA hidden rows")
        self.mesh_contract.validate_tensor(hidden_rows, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        full_hidden = ttnn.all_gather(
            hidden_rows,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=self.hidden_act_memory_config,
        )
        _require_shape(full_hidden, (1, 1, CHUNK_ROWS, HIDDEN_SIZE), "QSA hidden rows all-gather")
        self.mesh_contract.validate_tensor(full_hidden, placement=TensorPlacement.REPLICATED)
        return full_hidden

    def _index_projection_rows(self, full_hidden, cos, sin):
        index_q_ws = ttnn.linear(
            full_hidden,
            self.weights.index_q,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.index_program_config,
            compute_kernel_config=self.compute_config,
        )
        raw_key_ws = ttnn.linear(
            full_hidden,
            self.weights.index_k,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.index_program_config,
            compute_kernel_config=self.compute_config,
        )
        index_q = ttnn.to_memory_config(index_q_ws, ttnn.DRAM_MEMORY_CONFIG)
        raw_key = ttnn.to_memory_config(raw_key_ws, ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(index_q_ws, raw_key_ws)
        self.mesh_contract.validate_tensor(index_q, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        self.mesh_contract.validate_tensor(raw_key, placement=TensorPlacement.REPLICATED)
        _require_shape(index_q, (1, 1, CHUNK_ROWS, INDEX_HEAD_DIM), "local index query rows")
        _require_shape(raw_key, (1, 1, CHUNK_ROWS, INDEX_HEAD_DIM), "raw index key rows")
        _retag_tensor(index_q, reference=full_hidden, shard_dim=1)
        normalized = ttnn.rms_norm(
            index_q,
            epsilon=self.rms_norm_eps,
            weight=self.weights.index_q_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(index_q)
        rotated = apply_partial_rope_prefill(normalized, cos, sin, INDEX_QUERY_HEADS_PER_DEVICE, ROPE_DIM)
        _deallocate(normalized)
        _retag_tensor(rotated, reference=full_hidden, shard_dim=1)
        self.mesh_contract.validate_tensor(rotated, placement=TensorPlacement.HEAD_SHARDED, shard_dim=1)
        _require_shape(rotated, (1, 1, CHUNK_ROWS, INDEX_HEAD_DIM), "rotated index query rows")
        return rotated, raw_key

    def _write_compressed_index_chunk(
        self,
        state: Qwen38TTNNQSAGenericState,
        chunk_state: Qwen38TTNNQSAChunkState,
        raw_key,
        block_start_cos,
        block_start_sin,
        chunk: Qwen38TTNNQSAChunkInputs,
        constants: Qwen38TTNNQSAChunkConstants,
    ) -> None:
        kept = ttnn.copy(raw_key, chunk_state.kept_raw)
        if kept is not None and _tensor_key(kept) != _tensor_key(chunk_state.kept_raw):
            raise RuntimeError("QSA chunk raw keys were not kept in place")
        # Block means as one exact selection matmul (rows 0..7; rows 8..31 exact zeros), then the decode's
        # norm and block-start RoPE on the whole tile; the eight rows are written one per paged_update_cache
        # call, each picked into row 0 of its own tile by a 0/1 select and viewed as the one-row input.
        pooled = ttnn.matmul(
            constants.pool_select,
            raw_key,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(raw_key)
        _retag_tensor(pooled, reference=state.compressed_index_cache, shard_dim=None)
        _require_shape(pooled, (1, 1, CHUNK_ROWS, INDEX_HEAD_DIM), "pooled chunk index keys")
        normalized = ttnn.rms_norm(
            pooled,
            epsilon=self.rms_norm_eps,
            weight=self.weights.index_k_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(pooled)
        rotated = apply_partial_rope_prefill(normalized, block_start_cos, block_start_sin, 1, ROPE_DIM)
        _deallocate(normalized)
        _retag_tensor(rotated, reference=state.compressed_index_cache, shard_dim=None)
        self.mesh_contract.validate_tensor(rotated, placement=TensorPlacement.REPLICATED)
        _require_shape(rotated, (1, 1, CHUNK_ROWS, INDEX_HEAD_DIM), "rotated chunk index keys")
        one_row = ttnn.Shape((1, 1, 1, INDEX_HEAD_DIM))
        tile = ttnn.Shape((1, 1, CHUNK_ROWS, INDEX_HEAD_DIM))
        for block in range(CHUNK_BLOCKS):
            picked = ttnn.matmul(
                constants.row_selects[block],
                rotated,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_config,
            )
            _retag_tensor(picked, reference=rotated, shard_dim=None)
            _require_shape(picked, (1, 1, CHUNK_ROWS, INDEX_HEAD_DIM), f"picked compressed row {block}")
            row = ttnn.reshape(picked, one_row, tile)  # a view of picked's tile: row 0 logical
            _retag_tensor(row, reference=rotated, shard_dim=None)
            row_sharded = ttnn.to_memory_config(row, self.compressed_row_memory_config)
            _deallocate(picked)
            result = ttnn.experimental.paged_update_cache(
                state.compressed_index_cache,
                row_sharded,
                update_idxs_tensor=chunk.block_index_i32[block],
            )
            if _tensor_key(result) != _tensor_key(state.compressed_index_cache):
                raise RuntimeError("compressed QSA chunk index update was not in place")
            _deallocate(row_sharded)
        _deallocate(rotated)

    def _score_blocks_chunk(self, index_query, state: Qwen38TTNNQSAGenericState, chunk: Qwen38TTNNQSAChunkInputs):
        # The 32 query rows are the tile itself; the window past the last resident block ends at the
        # cache's last row, so every resident column is visible to every row and the per-row additive
        # mask hides the blocks at or past that row's complete-block count.
        local_scores = ttnn.experimental.indexer_score_dsa(
            index_query,
            state.compressed_index_cache,
            self.index_gate,
            chunk_start_idx=self.indexer_chunk_start,
            compute_kernel_config=self.indexer_compute_config,
            seq_shard_axes=[STAGING_AXIS],
        )
        score_rows = ttnn.slice(
            local_scores,
            (0, 0, 0, 0),
            (1, 1, CHUNK_ROWS, self.allocated_compressed_blocks),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _deallocate(local_scores)
        self.mesh_contract.mark_local_partial(
            score_rows,
            replicated_reference=state.compressed_index_cache,
            expected_shape=(1, 1, CHUNK_ROWS, self.allocated_compressed_blocks),
        )
        scores = ttnn.all_reduce(
            score_rows,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.collective_topology,
        )
        _deallocate(score_rows)
        self.mesh_contract.validate_tensor(scores, placement=TensorPlacement.REPLICATED)
        masked = ttnn.add(
            scores,
            chunk.indexer_neg_mask,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            fast_and_approximate_mode=False,
        )
        _deallocate(scores)
        _require_shape(masked, (1, 1, CHUNK_ROWS, self.allocated_compressed_blocks), "masked QSA chunk block scores")
        if masked.dtype != ttnn.bfloat16 or masked.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError(f"masked QSA chunk block scores must be BF16 ROW_MAJOR, got {tensor_metadata(masked)}")
        return masked

    def _materialize_rows_chunk(
        self, masked_scores, chunk: Qwen38TTNNQSAChunkInputs, constants: Qwen38TTNNQSAChunkConstants
    ):
        block_ids = ttnn.experimental.topk_large_indices(masked_scores, k=BLOCK_TOPK)
        _deallocate(masked_scores)
        _require_shape(block_ids, (1, 1, CHUNK_ROWS, BLOCK_TOPK), "top-k QSA chunk block IDs")
        if block_ids.dtype != ttnn.uint32 or block_ids.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError(
                f"topk_large_indices must return UINT32 ROW_MAJOR block IDs, got {tensor_metadata(block_ids)}"
            )
        starts = ttnn.bitwise_left_shift(block_ids, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(block_ids)
        repeated = ttnn.repeat_interleave(starts, repeats=COMPRESS_RATIO, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(starts)
        expanded = ttnn.add(repeated, constants.block_offsets_rows, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(repeated)
        _require_shape(expanded, (1, 1, CHUNK_ROWS, TOKEN_BUDGET), "expanded QSA chunk block indices")
        template = ttnn.concat([expanded, constants.sentinel_pad_rows], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(expanded)
        kept = ttnn.bitwise_and(template, chunk.row_keep_bits, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(template)
        sparse_indices = ttnn.bitwise_or(kept, chunk.row_fill, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(kept)
        _require_shape(sparse_indices, (1, 1, CHUNK_ROWS, SPARSE_INDEX_CAPACITY), "QSA chunk sparse indices")
        if sparse_indices.dtype != ttnn.uint32 or sparse_indices.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError(f"sparse QSA indices must be UINT32 ROW_MAJOR, got {tensor_metadata(sparse_indices)}")
        self.mesh_contract.validate_tensor(sparse_indices, placement=TensorPlacement.REPLICATED)
        return sparse_indices

    def _main_projection_rows(self, full_hidden, cos, sin):
        qg_ws = ttnn.linear(
            full_hidden,
            self.weights.qg,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.qg_program_config,
            compute_kernel_config=self.compute_config,
        )
        k_ws = ttnn.linear(
            full_hidden,
            self.weights.k_pair_grouped,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.kv_program_config,
            compute_kernel_config=self.compute_config,
        )
        v_ws = ttnn.linear(
            full_hidden,
            self.weights.v_pair_grouped,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.kv_program_config,
            compute_kernel_config=self.compute_config,
        )
        qg = ttnn.to_memory_config(qg_ws, ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.to_memory_config(k_ws, ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.to_memory_config(v_ws, ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(qg_ws, k_ws, v_ws)
        self.mesh_contract.validate_tensor(qg, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        self.mesh_contract.validate_tensor(k, placement=TensorPlacement.KV_PAIR_GROUPED, shard_dim=3)
        self.mesh_contract.validate_tensor(v, placement=TensorPlacement.KV_PAIR_GROUPED, shard_dim=3)
        _require_shape(qg, (1, 1, CHUNK_ROWS, 2 * LOCAL_QUERY_WIDTH), "local QSA query/gate rows")
        _require_shape(k, (1, 1, CHUNK_ROWS, HEAD_DIM), "local pair-grouped K rows")
        _require_shape(v, (1, 1, CHUNK_ROWS, HEAD_DIM), "local pair-grouped V rows")
        # Head h's [q_h | gate_h] columns are two tile-aligned slices of the row set; the heads stack on
        # dim 1 (whole 32-row tile blocks), so no padded intermediate reaches a reshape or permute.
        q_heads, gate_heads = [], []
        for head in range(QUERY_HEADS_PER_DEVICE):
            start = head * 2 * HEAD_DIM
            q_heads.append(
                ttnn.slice(
                    qg, (0, 0, 0, start), (1, 1, CHUNK_ROWS, start + HEAD_DIM), memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            )
            gate_heads.append(
                ttnn.slice(
                    qg,
                    (0, 0, 0, start + HEAD_DIM),
                    (1, 1, CHUNK_ROWS, start + 2 * HEAD_DIM),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
        _deallocate(qg)
        q = ttnn.concat(q_heads, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate = ttnn.concat(gate_heads, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(*q_heads, *gate_heads)
        _retag_tensor(q, reference=full_hidden, shard_dim=1)
        _retag_tensor(gate, reference=full_hidden, shard_dim=1)
        _require_shape(q, (1, QUERY_HEADS_PER_DEVICE, CHUNK_ROWS, HEAD_DIM), "local QSA query head rows")
        _require_shape(gate, (1, QUERY_HEADS_PER_DEVICE, CHUNK_ROWS, HEAD_DIM), "local QSA gate head rows")
        _retag_tensor(k, reference=full_hidden, shard_dim=1)
        _retag_tensor(v, reference=full_hidden, shard_dim=1)

        q_norm = ttnn.rms_norm(
            q,
            epsilon=self.rms_norm_eps,
            weight=self.weights.q_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        k_norm = ttnn.rms_norm(
            k,
            epsilon=self.rms_norm_eps,
            weight=self.weights.k_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(q, k)
        q_rotated = apply_partial_rope_prefill(q_norm, cos, sin, QUERY_HEADS_PER_DEVICE, ROPE_DIM)
        k_rotated = apply_partial_rope_prefill(k_norm, cos, sin, 1, ROPE_DIM)
        _deallocate(q_norm, k_norm)
        _retag_tensor(q_rotated, reference=full_hidden, shard_dim=1)
        _retag_tensor(k_rotated, reference=full_hidden, shard_dim=1)
        for value in (k_rotated, v):
            self.mesh_contract.validate_tensor(value, placement=TensorPlacement.KV_PAIR_GROUPED, shard_dim=1)
        _require_shape(q_rotated, (1, QUERY_HEADS_PER_DEVICE, CHUNK_ROWS, HEAD_DIM), "rotated QSA query head rows")
        _require_shape(k_rotated, (1, 1, CHUNK_ROWS, HEAD_DIM), "rotated pair-grouped K rows")
        return q_rotated, gate, k_rotated, v

    def _write_packed_kv_chunk(
        self,
        state: Qwen38TTNNQSAGenericState,
        chunk_state: Qwen38TTNNQSAChunkState,
        key,
        value,
        chunk: Qwen38TTNNQSAChunkInputs,
    ) -> None:
        packed_tiled = ttnn.concat([value, key], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(value, key)
        _require_shape(packed_tiled, (1, 1, CHUNK_ROWS, 2 * HEAD_DIM), "packed QSA chunk KV slab")
        kept = ttnn.copy(packed_tiled, chunk_state.kept_kv)
        if kept is not None and _tensor_key(kept) != _tensor_key(chunk_state.kept_kv):
            raise RuntimeError("QSA chunk KV slab was not kept in place")
        # The whole 32-row slab, untilized once, lands at row P (the decode's staging write with every row real).
        slab_row_major = ttnn.to_layout(packed_tiled, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(packed_tiled)
        _retag_tensor(slab_row_major, reference=state.packed_kv_cache, shard_dim=1)
        _require_shape(slab_row_major, (1, 1, CHUNK_ROWS, 2 * HEAD_DIM), "row-major QSA chunk KV slab")
        self.mesh_contract.validate_tensor(slab_row_major, placement=TensorPlacement.KV_PAIR_GROUPED, shard_dim=1)
        result = ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            state.packed_kv_cache,
            slab_row_major,
            self.slot_zero,
            chunk.kv_block_start,
            0,  # layer_idx inside this one-layer cache
            1,  # num_layers
            STAGING_AXIS,
        )
        if _tensor_key(result) != _tensor_key(state.packed_kv_cache):
            raise RuntimeError("row-major QSA chunk KV update was not in place")
        _deallocate(slab_row_major)

    def _sparse_value_attention_rows(self, query, gate, sparse_indices, state, constants: Qwen38TTNNQSAChunkConstants):
        # q = [zeros(256) | Q(256)] per local head over the 32 rows, untilized, 26 zero heads appended.
        sparse_query_tiled = ttnn.concat(
            [constants.zero_value_half_rows, query], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        _deallocate(query)
        _retag_tensor(sparse_query_tiled, reference=state.packed_kv_cache, shard_dim=1)
        _require_shape(
            sparse_query_tiled, (1, QUERY_HEADS_PER_DEVICE, CHUNK_ROWS, 2 * HEAD_DIM), "local sparse QSA query rows"
        )
        sparse_query_row_major = ttnn.to_layout(
            sparse_query_tiled, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        _deallocate(sparse_query_tiled)
        sparse_query = ttnn.pad(
            sparse_query_row_major,
            [(0, 0), (0, 32 - QUERY_HEADS_PER_DEVICE), (0, 0), (0, 0)],
            0.0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _deallocate(sparse_query_row_major)
        _retag_tensor(sparse_query, reference=state.packed_kv_cache, shard_dim=1)
        _require_shape(sparse_query, (1, 32, CHUNK_ROWS, 2 * HEAD_DIM), "padded sparse QSA query rows")
        if sparse_query.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError(f"padded sparse QSA query rows must be ROW_MAJOR, got {tensor_metadata(sparse_query)}")
        sparse_output = ttnn.transformer.sparse_sdpa(
            sparse_query,
            state.packed_kv_cache,
            sparse_indices,
            HEAD_DIM,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=HEAD_DIM**-0.5,
            k_chunk_size=ttnn.TILE_SIZE,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(sparse_query)
        local = ttnn.slice(
            sparse_output,
            (0, 0, 0, 0),
            (1, QUERY_HEADS_PER_DEVICE, CHUNK_ROWS, HEAD_DIM),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _deallocate(sparse_output)
        local_tiled = ttnn.to_layout(local, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(local)
        _retag_tensor(local_tiled, reference=gate, shard_dim=1)
        activated_gate = ttnn.sigmoid(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gated = ttnn.mul(local_tiled, activated_gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(local_tiled, activated_gate, gate)
        _require_shape(gated, (1, QUERY_HEADS_PER_DEVICE, CHUNK_ROWS, HEAD_DIM), "gated QSA attention head rows")
        # Head-major -> one [1,1,32,1536] row set: six whole-tile head slices concatenated on the last dim
        # (the 1-row path's flatten is a view only at S = 1).
        heads = [
            ttnn.slice(
                gated, (0, head, 0, 0), (1, head + 1, CHUNK_ROWS, HEAD_DIM), memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for head in range(QUERY_HEADS_PER_DEVICE)
        ]
        _deallocate(gated)
        local_flat = ttnn.concat(heads, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(*heads)
        _retag_tensor(local_flat, reference=state.packed_kv_cache, shard_dim=3)
        self.mesh_contract.validate_tensor(local_flat, placement=TensorPlacement.HEAD_SHARDED, shard_dim=3)
        _require_shape(local_flat, (1, 1, CHUNK_ROWS, LOCAL_QUERY_WIDTH), "flat QSA attention rows")
        return local_flat

    def _project_output_rows(self, local_attention, full_hidden):
        attention_ws = ttnn.to_memory_config(local_attention, self.out_act_memory_config)
        _deallocate(local_attention)
        local_partial_ws = ttnn.linear(
            attention_ws,
            self.weights.out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.out_program_config,
            compute_kernel_config=self.compute_config,
        )
        _deallocate(attention_ws)
        local_partial = ttnn.to_memory_config(local_partial_ws, ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(local_partial_ws)
        self.mesh_contract.mark_local_partial(
            local_partial, replicated_reference=full_hidden, expected_shape=(1, 1, CHUNK_ROWS, HIDDEN_SIZE)
        )
        local_partial_fp32 = ttnn.typecast(local_partial, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate(local_partial)
        self.mesh_contract.mark_local_partial(
            local_partial_fp32, replicated_reference=full_hidden, expected_shape=(1, 1, CHUNK_ROWS, HIDDEN_SIZE)
        )
        output_fp32 = ttnn.reduce_scatter(
            local_partial_fp32,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.collective_topology,
        )
        _deallocate(local_partial_fp32)
        self.mesh_contract.mark_collective_shard(
            output_fp32,
            replicated_reference=full_hidden,
            shard_dim=3,
            expected_local_shape=(1, 1, CHUNK_ROWS, HIDDEN_SIZE // TP_SIZE),
        )
        if output_fp32.dtype != ttnn.float32:
            raise RuntimeError(f"QSA output reduce_scatter must sum FP32 partials, got {tensor_metadata(output_fp32)}")
        output = ttnn.typecast(output_fp32, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _retag_tensor(output, reference=output_fp32, shard_dim=3)
        _deallocate(output_fp32)
        _require_shape(output, (1, 1, CHUNK_ROWS, HIDDEN_SIZE // TP_SIZE), "QSA output projection rows")
        if output.dtype != ttnn.bfloat16:
            raise RuntimeError(f"QSA output projection rows must be BF16, got {tensor_metadata(output)}")
        self.mesh_contract.validate_tensor(output, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        return output

    def forward_chunk_generic(
        self,
        hidden_rows,
        state: Qwen38TTNNQSAGenericState,
        chunk_state: Qwen38TTNNQSAChunkState,
        *,
        cos,
        sin,
        block_start_cos,
        block_start_sin,
        chunk: Qwen38TTNNQSAChunkInputs,
        constants: Qwen38TTNNQSAChunkConstants,
    ):
        """Prefill the 32 positions P .. P + 31 of one chunk (P % 32 == 0); same op sequence for every chunk.

        ``state`` is mutated in place (the KV slab at rows P .. P + 31, the eight compressed blocks
        P/4 .. P/4 + 7; the staging tile and the raw-key ring are left to the eager hand-off, which reads
        ``chunk_state``).  The RoPE rows are the table lookups at P + j and P + 4i.  Returns the
        ``[1,1,32,640]`` BF16 TILE hidden-sharded output rows; the input is left for the caller to release.
        """

        self._validate_generic_state(state)
        self._validate_chunk_state(chunk_state)
        self._validate_rope_rows(cos, sin, "QSA chunk RoPE")
        self._validate_rope_rows(block_start_cos, block_start_sin, "QSA chunk block-start RoPE")
        self._validate_chunk_inputs(chunk)
        if constants.allocated_compressed_blocks != self.allocated_compressed_blocks:
            raise ValueError(
                f"QSA chunk constants were built for {constants.allocated_compressed_blocks} blocks, "
                f"the layer has {self.allocated_compressed_blocks}"
            )

        full_hidden = self._all_gather_hidden_rows(hidden_rows)
        index_query, raw_key = self._index_projection_rows(full_hidden, cos, sin)
        self._write_compressed_index_chunk(
            state, chunk_state, raw_key, block_start_cos, block_start_sin, chunk, constants
        )
        masked_scores = self._score_blocks_chunk(index_query, state, chunk)
        _deallocate(index_query)
        sparse_indices = self._materialize_rows_chunk(masked_scores, chunk, constants)

        query, gate, key, value = self._main_projection_rows(full_hidden, cos, sin)
        self._write_packed_kv_chunk(state, chunk_state, key, value, chunk)
        local_attention = self._sparse_value_attention_rows(query, gate, sparse_indices, state, constants)
        _deallocate(sparse_indices)
        output = self._project_output_rows(local_attention, full_hidden)
        _deallocate(full_hidden)
        return output

    def handoff_chunk_state(
        self,
        state: Qwen38TTNNQSAGenericState,
        chunk_state: Qwen38TTNNQSAChunkState,
        *,
        ring_select,
        open_block: bool,
    ) -> None:
        """Eager hand-off after the last chunk of a prefill: the generic body's per-token buffers from the kept rows.

        ``kv_staging`` takes the kept ``[v|k]`` slab when the prompt ends inside the last chunk's block
        (``open_block``: rows past the prompt hold the padded rows' values, which the next replay rewrites or
        never gathers), zero when the block is complete.  ``raw_key_ring`` is ``ring_select @ kept_raw``
        (:func:`chunk_handoff_ring_select_rows`: the open compressed block's raw keys in rows 0 .. P % 4 - 1, every
        other row exact zero).  The two caches need nothing: rows past the prompt are never gathered and the open
        block's compressed row is rewritten from the ring before it is scored.
        """

        self._validate_generic_state(state)
        self._validate_chunk_state(chunk_state)
        _require_shape(ring_select, (1, 1, CHUNK_ROWS, CHUNK_ROWS), "QSA hand-off ring select")
        if ring_select.dtype != ttnn.bfloat16 or ring_select.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError(f"QSA hand-off ring select must be BF16 TILE, got {tensor_metadata(ring_select)}")
        if open_block:
            staged = ttnn.copy(chunk_state.kept_kv, state.kv_staging)
        else:
            staged = ttnn.fill(state.kv_staging, 0.0, output_tensor=state.kv_staging)
        if staged is not None and _tensor_key(staged) != _tensor_key(state.kv_staging):
            raise RuntimeError("QSA KV staging hand-off was not in place")
        # One exact 0/1 selection (HiFi4, fp32 accumulation: one bf16 term times 1.0 plus exact zeros per element)
        # written into the ring itself; the zero rows of the select are the ring's zero rows.
        ring = ttnn.matmul(
            ring_select,
            chunk_state.kept_raw,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
            optional_output_tensor=state.raw_key_ring,
        )
        if ring is not None and _tensor_key(ring) != _tensor_key(state.raw_key_ring):
            raise RuntimeError("QSA raw-key ring hand-off was not in place")
