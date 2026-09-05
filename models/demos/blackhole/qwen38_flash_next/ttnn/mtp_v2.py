# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MTP v2: the traced (k + 1)-row verify pass of the speculative decode (design sections 2.3, 2.4, 2.6, 2.7).

One pass at the device position ``P`` runs the R = k + 1 rows ``[t_P, d_1 .. d_k]`` through all 48 layers on
32-row tiles (rows R .. 31 are zero embedding rows; every op is per row or masked), resolves the R greedy ids
on device, accepts the longest matching draft prefix (``a`` in 0 .. k, exact integer arithmetic), feeds the R
target roots and the R target predictions ``[argmax_0 .. argmax_k]`` (row a's is ``t'``) through the MTP alignment
layer, and ends with ``P <- P + a + 1``.  The host reads one FP32 row per pass: ``[a, t', d_1', argmax_0 .. argmax_31]``.

State after a partial accept, without snapshots:

* GDN recurrent state and FIR history, PLE conv history, QSA raw-key history: committed at the START of the
  next pass from the device accept count (``catch_up=True``): every layer first replays its committed prefix
  (``commit_rows`` / ``commit_verify`` with the pass's ``build_rows_selectors``) and then runs the new rows.
  The first pass after an eager mode switch runs with ``catch_up=False`` (its histories come from the eager
  seed), so the two bodies are captured as two traces.
* QSA KV rows and compressed blocks: written by absolute position every pass and never rolled back; rows and
  blocks past the accepted prefix are finite garbage that the next pass rewrites before any sparse row or
  indexer mask can expose them (``ttnn/qsa.py``, the verify section).
* The MTP alignment layer's caches follow the same rule at positions P .. P + k; the next draft trace starts
  from row ``a`` of its residuals (``alignment.residual``).

The draft body (design section 2.5, :func:`forward_draft`) runs the k - 1 MTP rows at MTP positions
P .. P + k - 2 with a device token chain (d_1 out of the verify readback row, later ids the previous row's
argmax) and assembles the next verify pass's token row and draft lanes on device, so the host segment of a pass
is one 128-byte readback of the assembled ids and the PLE n-gram lookup + upload for the R rows
(:func:`write_verify_ple_rows`).  :func:`forward_commit` is the split form of the catch-up (all commits as one
body, replayed while the host does the lookup); :class:`Qwen38TTNNMTPChain` is the pass loop over the traces.

The 1-row production paths are untouched: everything here is a new entry point over the layers' rows paths.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    CHUNK_ROWS,
    MESH_SHAPE,
    Qwen38MeshContract,
    TensorPlacement,
    replicate_tensor_2d_mesh_mapper,
    tensor_metadata,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import TOKEN_ROW_SHAPE, ZERO_EMBEDDING_TOKEN
from models.demos.blackhole.qwen38_flash_next.ttnn.final_mixer import Qwen38TTNNFinalMixer
from models.demos.blackhole.qwen38_flash_next.ttnn.gdn import Qwen38TTNNGDN, Qwen38TTNNGDNRowsState
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import (
    BACKBONE_LAYERS,
    LOCAL_HIDDEN_SIZE,
    PLE_CHECKPOINT_LAYER,
    RESIDUAL_BRANCHES,
    Qwen38TTNNDecoderLayer,
    Qwen38TTNNDecoderLayerGenericState,
    Qwen38TTNNLayerNamespace,
    Qwen38TTNNLayerType,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.model import Qwen38TTNNTextModel, Qwen38TTNNTextModelGenericState
from models.demos.blackhole.qwen38_flash_next.ttnn.moe import SUPPORTED_ROWS, Qwen38TTNNMoE
from models.demos.blackhole.qwen38_flash_next.ttnn.mtp import Qwen38TTNNMTPInput
from models.demos.blackhole.qwen38_flash_next.ttnn.ple import (
    Qwen38TTNNPLE,
    Qwen38TTNNPLERowsPreparedInput,
    Qwen38TTNNPLERowsState,
)

SUPPORTED_DRAFTS = (3, 4, 5)
DEFAULT_DRAFTS = 4
RESIDUAL_ROWS_SHAPE = (1, RESIDUAL_BRANCHES, CHUNK_ROWS, LOCAL_HIDDEN_SIZE)
BLOCK_ROWS_SHAPE = (1, 1, CHUNK_ROWS, LOCAL_HIDDEN_SIZE)
MTP_RESIDUAL_SHAPE = (1, RESIDUAL_BRANCHES, 1, LOCAL_HIDDEN_SIZE)
# The verify readback row: accept count, next token, first draft of the next pass, then the 32 verify argmaxes.
READBACK_FIXED_LANES = ("accepted", "next_token", "first_draft")
READBACK_WIDTH = len(READBACK_FIXED_LANES) + CHUNK_ROWS
# The pass row (the one host readback per pass, landed by the draft body): the verify readback row, then the next
# pass's verify tokens ``[t', d_1 .. d_k, ZERO_EMBEDDING_TOKEN ...]`` on CHUNK_ROWS lanes.
PASS_ROW_WIDTH = READBACK_WIDTH + CHUNK_ROWS


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(item) for item in tensor.shape)


def _tensor_key(tensor) -> tuple[str, int]:
    return (str(tensor.device()), int(tensor.tensor_id))


def _deallocate(*tensors) -> None:
    seen: set[tuple[str, int]] = set()
    for tensor in tensors:
        if tensor is None:
            continue
        key = _tensor_key(tensor)
        if key not in seen:
            seen.add(key)
            ttnn.deallocate(tensor)


def _run_cleanup(label: str, actions: list[tuple[str, Callable[[], Any]]], *, primary: BaseException | None = None):
    errors: list[BaseException] = []
    for name, action in actions:
        try:
            action()
        except BaseException as error:  # noqa: BLE001 - every cleanup step must run
            errors.append(RuntimeError(f"{label}: {name} cleanup failed: {error}"))
    if errors:
        raise RuntimeError(f"{label}: {len(errors)} cleanup step(s) failed: {errors}") from (primary or errors[0])


def _upload_replicated(
    mesh_device, mesh_contract: Qwen38MeshContract, host: torch.Tensor, dtype, layout, *, label: str
):
    tensor = ttnn.from_torch(
        host,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh_device),
    )
    if _shape(tensor) != tuple(host.shape) or tensor.dtype != dtype or tensor.layout != layout:
        raise RuntimeError(f"{label} must be {dtype} {layout} {list(host.shape)}, got {tensor_metadata(tensor)}")
    mesh_contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)
    return tensor


def _allocate_hidden_sharded_zeros(
    mesh_device, mesh_contract: Qwen38MeshContract, local_shape: tuple[int, ...], *, label: str
):
    """A zero BF16 TILE buffer hidden-sharded on dim 3 (global width TP x local)."""

    global_shape = (*local_shape[:-1], local_shape[-1] * MESH_SHAPE[1])
    tensor = ttnn.from_torch(
        torch.zeros(global_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
    )
    if _shape(tensor) != tuple(local_shape):
        raise RuntimeError(f"{label} must have local shape {list(local_shape)}, got {tensor_metadata(tensor)}")
    mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
    return tensor


def _pad_rows(tensor, rows: int, *, label: str):
    """Rows ``rows .. 31`` of the tile zeroed in place; the result is a view of ``tensor`` (never deallocate it)."""

    shape = _shape(tensor)
    if shape[2] != rows:
        raise RuntimeError(f"{label} has {shape[2]} rows, expected {rows} before the pad")
    padded = ttnn.pad(
        tensor, [(0, 0), (0, 0), (0, CHUNK_ROWS - rows), (0, 0)], 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    padded.update_tensor_topology(tensor.tensor_topology())
    if _shape(padded) != (*shape[:2], CHUNK_ROWS, shape[3]):
        raise RuntimeError(f"{label} pad produced {_shape(padded)}, expected {(*shape[:2], CHUNK_ROWS, shape[3])}")
    return padded


# --------------------------------------------------------------------------- device accept (design 2.6)


@dataclass(frozen=True)
class Qwen38TTNNAcceptConstants:
    """Replicated constants of the accept logic for one draft count ``k``.

    ``prefix_upper`` FP32 TILE ``[1,1,32,32]`` (``U[i, j] = 1`` for ``i <= j``) turns the match flags into
    running counts with one matmul (0/1 terms, sums <= 32: exact through the FPU); ``arange_plus_one`` FP32
    TILE ``[1,1,1,32]`` (lane j = j + 1) marks the lanes whose every earlier flag was 1; ``arange_col`` FP32
    TILE ``[1,1,32,1]`` builds the one-hot row select of the alignment residual; ``sentinel_tail`` FP32
    ROW_MAJOR ``[1,1,1,31-k]`` and ``sentinel_lane`` ``[1,1,1,1]`` hold ``ZERO_EMBEDDING_TOKEN``.
    """

    drafts: int
    prefix_upper: Any
    arange_plus_one: Any
    arange_col: Any
    sentinel_tail: Any
    sentinel_lane: Any
    compute_config: Any

    @classmethod
    def build(cls, mesh_device, mesh_contract: Qwen38MeshContract, *, drafts: int) -> "Qwen38TTNNAcceptConstants":
        if drafts not in SUPPORTED_DRAFTS:
            raise ValueError(f"MTP v2 verify admits k in {SUPPORTED_DRAFTS}, got {drafts!r}")
        mesh_contract.validate_mesh(mesh_device)
        lane = torch.arange(CHUNK_ROWS, dtype=torch.float32)
        uploaded: list[Any] = []

        def upload(host: torch.Tensor, layout, label: str):
            tensor = _upload_replicated(mesh_device, mesh_contract, host, ttnn.float32, layout, label=label)
            uploaded.append(tensor)
            return tensor

        try:
            return cls(
                drafts=drafts,
                prefix_upper=upload(
                    (lane.reshape(CHUNK_ROWS, 1) <= lane.reshape(1, CHUNK_ROWS))
                    .float()
                    .reshape(1, 1, CHUNK_ROWS, CHUNK_ROWS),
                    ttnn.TILE_LAYOUT,
                    "accept prefix matrix",
                ),
                arange_plus_one=upload(
                    (lane + 1.0).reshape(1, 1, 1, CHUNK_ROWS), ttnn.TILE_LAYOUT, "accept lane counts"
                ),
                arange_col=upload(lane.reshape(1, 1, CHUNK_ROWS, 1), ttnn.TILE_LAYOUT, "accept row index column"),
                sentinel_tail=upload(
                    torch.full((1, 1, 1, CHUNK_ROWS - 1 - drafts), float(ZERO_EMBEDDING_TOKEN)),
                    ttnn.ROW_MAJOR_LAYOUT,
                    "alignment sentinel tail",
                ),
                sentinel_lane=upload(
                    torch.full((1, 1, 1, 1), float(ZERO_EMBEDDING_TOKEN)), ttnn.ROW_MAJOR_LAYOUT, "sentinel lane"
                ),
                compute_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                ),
            )
        except BaseException:
            _deallocate(*uploaded)
            raise

    def deallocate(self) -> None:
        _deallocate(self.prefix_upper, self.arange_plus_one, self.arange_col, self.sentinel_tail, self.sentinel_lane)


@dataclass(frozen=True)
class Qwen38TTNNAcceptResult:
    """``accepted_tile`` FP32 TILE ``[1,1,1,1]`` (the GDN/PLE selectors' input), ``accepted_lane`` the same value
    ROW_MAJOR, ``accepted_index`` UINT32 ROW_MAJOR (the gather index and the position increment), ``next_token``
    FP32 ROW_MAJOR ``[1,1,1,1]`` = verify argmax ``a``."""

    accepted_tile: Any
    accepted_lane: Any
    accepted_index: Any
    next_token: Any

    def deallocate(self) -> None:
        _deallocate(self.accepted_tile, self.accepted_lane, self.accepted_index, self.next_token)


def accept_rows(argmax_lanes, draft_lanes, constants: Qwen38TTNNAcceptConstants) -> Qwen38TTNNAcceptResult:
    """The accept count and the next token from the verify argmaxes and the uploaded drafts, exact integers.

    ``argmax_lanes`` (lane j = argmax of verify row j) and ``draft_lanes`` (lane j = d_{j+1} for j < k, else
    ``ZERO_EMBEDDING_TOKEN``) are FP32 ROW_MAJOR ``[1,1,1,32]``.  ``eq`` gives the k match flags (lanes >= k
    are 0), the prefix matmul their running counts, a second ``eq`` the lanes whose prefix is all ones, and the
    sum of those lanes is ``a``.  The next token is a 32-bit ``ttnn.gather`` copy of lane ``a``: no id ever
    enters an FPU or reduce stage (TF32 would round ids >= 2048).
    """

    for name, lanes in (("verify argmax lanes", argmax_lanes), ("draft lanes", draft_lanes)):
        if _shape(lanes) != TOKEN_ROW_SHAPE or lanes.dtype != ttnn.float32 or lanes.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError(f"{name} must be FP32 ROW_MAJOR {TOKEN_ROW_SHAPE}, got {tensor_metadata(lanes)}")
    dram = ttnn.DRAM_MEMORY_CONFIG
    flags = ttnn.eq(argmax_lanes, draft_lanes, dtype=ttnn.float32, memory_config=dram)
    flags_tile = ttnn.to_layout(flags, ttnn.TILE_LAYOUT, memory_config=dram)
    _deallocate(flags)
    running = ttnn.matmul(
        flags_tile, constants.prefix_upper, memory_config=dram, compute_kernel_config=constants.compute_config
    )
    _deallocate(flags_tile)
    prefix = ttnn.eq(running, constants.arange_plus_one, dtype=ttnn.float32, memory_config=dram)
    _deallocate(running)
    accepted_tile = ttnn.sum(prefix, dim=3, keepdim=True, memory_config=dram)
    _deallocate(prefix)
    if _shape(accepted_tile) != (1, 1, 1, 1) or accepted_tile.dtype != ttnn.float32:
        raise RuntimeError(f"accept count must be FP32 [1,1,1,1], got {tensor_metadata(accepted_tile)}")
    accepted_lane = ttnn.to_layout(accepted_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    accepted_index = ttnn.typecast(accepted_lane, ttnn.uint32, memory_config=dram)
    next_token = ttnn.gather(argmax_lanes, 3, accepted_index, memory_config=dram)
    if (
        _shape(next_token) != (1, 1, 1, 1)
        or next_token.dtype != ttnn.float32
        or next_token.layout != ttnn.ROW_MAJOR_LAYOUT
    ):
        raise RuntimeError(f"next-token select must be FP32 ROW_MAJOR [1,1,1,1], got {tensor_metadata(next_token)}")
    return Qwen38TTNNAcceptResult(accepted_tile, accepted_lane, accepted_index, next_token)


def select_residual_row(residual_rows, accepted_tile, constants: Qwen38TTNNAcceptConstants, *, output) -> None:
    """``output <- residual_rows[:, :, a]`` for the branch-major ``[1,4,32,640]`` BF16 rows: a one-hot row multiply
    and a sum over the rows.  Exact: every output element is one bf16 value plus exact zeros, whatever the
    reduce reads its sources as (the integer-id rule does not apply to bf16 activations)."""

    if _shape(residual_rows) != RESIDUAL_ROWS_SHAPE or residual_rows.dtype != ttnn.bfloat16:
        raise RuntimeError(f"residual rows must be BF16 {RESIDUAL_ROWS_SHAPE}, got {tensor_metadata(residual_rows)}")
    dram = ttnn.DRAM_MEMORY_CONFIG
    onehot = ttnn.eq(constants.arange_col, accepted_tile, dtype=ttnn.bfloat16, memory_config=dram)
    picked = ttnn.multiply(residual_rows, onehot, memory_config=dram)
    _deallocate(onehot)
    row = ttnn.sum(picked, dim=2, keepdim=True, memory_config=dram)
    _deallocate(picked)
    if _shape(row) != MTP_RESIDUAL_SHAPE:
        raise RuntimeError(f"selected residual row has shape {_shape(row)}, expected {MTP_RESIDUAL_SHAPE}")
    landed = ttnn.copy(row, output)
    if landed is not None and _tensor_key(landed) != _tensor_key(output):
        raise RuntimeError("selected residual row did not land in its persistent buffer")
    _deallocate(row)


# --------------------------------------------------------------------------- verify state


@dataclass(frozen=True)
class Qwen38TTNNVerifyLayerState:
    """One layer's verify buffers: the GDN rows state or the QSA verify state, the PLE rows state (layer 1), and
    the MoE instance at the verify state's MoE row count (``moe_input`` is its persistent input slice; None when
    the instance takes the whole 32-row tile).  ``gdn_step_anchor`` (GDN layers only): the layer's commits run
    the committed rows through the 1-row FP32 step recurrence instead of the chunk kernel (the state re-anchor,
    ``Qwen38TTNNGDN.commit_rows(step_committed_rows=True)``)."""

    namespace: Qwen38TTNNLayerNamespace
    layer_index: int
    rows: int
    attention: Any
    ple: Qwen38TTNNPLERowsState | None
    moe: Qwen38TTNNMoE
    moe_input: Any | None
    gdn_step_anchor: bool = False


@dataclass(frozen=True)
class Qwen38TTNNVerifyAlignment:
    """The MTP alignment rows' components and state: the MTP decoder layer with its own generic and verify
    state, the input and final mixers, and ``residual`` ``[1,4,1,640]`` = row ``a`` of the alignment residuals
    (the next draft trace's start)."""

    layer: Qwen38TTNNDecoderLayer
    input_mixer: Qwen38TTNNMTPInput
    final_mixer: Qwen38TTNNFinalMixer
    generic_state: Qwen38TTNNDecoderLayerGenericState
    verify_state: Qwen38TTNNVerifyLayerState
    residual: Any


@dataclass(frozen=True)
class Qwen38TTNNVerifyState:
    """Fixed-address buffers of the verify body, allocated beside the generic state before any capture.

    Host-written per pass (outside the traces): ``token_row`` FP32 TILE ``[1,1,1,32]`` (lane j = token j of
    ``[t_P, d_1 .. d_k]``, then ``ZERO_EMBEDDING_TOKEN``), ``draft_lanes`` FP32 ROW_MAJOR ``[1,1,1,32]`` (lane
    j = d_{j+1} for j < k, then ``ZERO_EMBEDDING_TOKEN``) and ``ple_rows`` (the R n-gram rows).  Device-written:
    ``accepted`` FP32 TILE ``[1,1,1,1]`` (the pass's accept count, read by the next pass's commits).
    ``moe_rows`` is every layer's verify MoE row count (``moe_rows_for(rows)`` unless the allocation was given
    an explicit override); ``gdn_step_anchor_layers`` the GDN layers whose commits run the state re-anchor.
    """

    drafts: int
    rows: int
    rows_constants: gdn_module.Qwen38TTNNGDNRowsConstants
    qsa_chunk_constants: qsa_module.Qwen38TTNNQSAChunkConstants
    qsa_verify_constants: qsa_module.Qwen38TTNNQSAVerifyConstants
    accept_constants: Qwen38TTNNAcceptConstants
    layers: tuple[Qwen38TTNNVerifyLayerState, ...]
    token_row: Any
    draft_lanes: Any
    ple_rows: Qwen38TTNNPLERowsPreparedInput
    accepted: Any
    alignment: Qwen38TTNNVerifyAlignment | None
    moe_rows: int
    gdn_step_anchor_layers: frozenset[int]
    _owner: object = field(repr=False, compare=False)


@dataclass
class Qwen38TTNNVerifyOutput:
    """The one readback row of a pass: FP32 ROW_MAJOR ``[1,1,1,READBACK_WIDTH]`` at a trace-stable address."""

    readback: Any
    active: bool = True

    def release_tensors(self) -> None:
        if not self.active:
            raise RuntimeError("verify output tensors were already released")
        _deallocate(self.readback)
        self.active = False


@dataclass(frozen=True)
class Qwen38TTNNVerifyReadback:
    accepted: int
    next_token: int
    first_draft: int | None
    argmaxes: tuple[int, ...]


def moe_rows_for(rows: int) -> int:
    """The smallest admitted MoE row count that holds ``rows`` (5 for k = 3 and 4; 32 for k = 5 until rows 6 is
    admitted in ``moe.SUPPORTED_ROWS``); a verify pass is one 32-row tile, so the 128-row prefill form is not a
    candidate."""

    admitted = [count for count in SUPPORTED_ROWS if rows <= count <= CHUNK_ROWS]
    if not admitted:
        raise ValueError(f"no admitted MoE row count holds {rows} rows (SUPPORTED_ROWS = {SUPPORTED_ROWS})")
    return min(admitted)


def resolve_moe_rows(rows: int, moe_rows: int | None) -> int:
    """The verify MoE row count: ``moe_rows_for(rows)``, or an explicit override in ``rows .. 32`` (a runner-level
    argument, e.g. 6 rows for k = 5 instead of the admitted 32; ``SUPPORTED_ROWS`` and the proof flags are not
    consulted for an override, so its instance is constructed with its own admitted set)."""

    if moe_rows is None:
        return moe_rows_for(rows)
    if isinstance(moe_rows, bool) or type(moe_rows) is not int or not rows <= moe_rows <= CHUNK_ROWS:
        raise ValueError(f"MoE rows override must be an int in [{rows}, {CHUNK_ROWS}], got {moe_rows!r}")
    return moe_rows


def _allocate_layer_verify_state(
    layer: Qwen38TTNNDecoderLayer,
    rows: int,
    rows_constants: gdn_module.Qwen38TTNNGDNRowsConstants,
    *,
    moe_rows: int,
    gdn_step_anchor: bool = False,
) -> Qwen38TTNNVerifyLayerState:
    if gdn_step_anchor and not isinstance(layer.attention, Qwen38TTNNGDN):
        raise ValueError(f"layer {layer.layer_index} is not a GDN layer; the state re-anchor has no commit there")
    attention = (
        layer.attention.allocate_rows_state(rows_constants)
        if isinstance(layer.attention, Qwen38TTNNGDN)
        else layer.attention.allocate_verify_state()
    )
    ple = None
    moe = None
    moe_input = None
    try:
        ple = None if layer.ple is None else layer.ple.allocate_rows_state(rows)
        moe = Qwen38TTNNMoE(
            layer.mlp.mesh_device,
            layer.mlp.mesh_contract,
            layer.mlp.weights,
            tt_ccl=layer.mlp.tt_ccl,
            collective_topology=layer.mlp.collective_topology,
            rows=moe_rows,
            synchronization_policy=layer.mlp.synchronization_policy,
            admitted_rows=SUPPORTED_ROWS if moe_rows in SUPPORTED_ROWS else (moe_rows,),
        )
        if moe_rows != CHUNK_ROWS:
            moe_input = _allocate_hidden_sharded_zeros(
                layer.mlp.mesh_device,
                layer.mlp.mesh_contract,
                (1, 1, moe_rows, LOCAL_HIDDEN_SIZE),
                label=f"layer {layer.layer_index} verify MoE input",
            )
        return Qwen38TTNNVerifyLayerState(
            layer.namespace, layer.layer_index, rows, attention, ple, moe, moe_input, gdn_step_anchor
        )
    except BaseException as error:
        actions: list[tuple[str, Callable[[], Any]]] = []
        if moe_input is not None:
            actions.append(("verify MoE input", lambda: _deallocate(moe_input)))
        if moe is not None:
            actions.append(("verify MoE buffers", moe.release_owned_buffers))
        if ple is not None:
            actions.append(("PLE rows state", ple.deallocate))
        actions.append(("verify attention state", lambda: _release_layer_attention(layer, attention)))
        _run_cleanup("decoder-layer verify state allocation", actions, primary=error)
        raise


def _release_layer_attention(layer: Qwen38TTNNDecoderLayer, attention) -> None:
    if isinstance(layer.attention, Qwen38TTNNGDN):
        attention.deallocate()
    else:
        layer.attention.release_verify_state(attention)


def _release_layer_verify_state(layer: Qwen38TTNNDecoderLayer, state: Qwen38TTNNVerifyLayerState) -> None:
    actions: list[tuple[str, Callable[[], Any]]] = [("verify MoE buffers", state.moe.release_owned_buffers)]
    if state.moe_input is not None:
        actions.append(("verify MoE input", lambda: _deallocate(state.moe_input)))
    if state.ple is not None:
        actions.append(("PLE rows state", state.ple.deallocate))
    actions.append(("verify attention state", lambda: _release_layer_attention(layer, state.attention)))
    _run_cleanup("decoder-layer verify state", actions)


def _validate_layer_verify_state(
    layer: Qwen38TTNNDecoderLayer,
    state: Qwen38TTNNVerifyLayerState,
    rows: int,
    moe_rows: int,
    *,
    gdn_step_anchor: bool = False,
) -> None:
    if state.namespace is not layer.namespace or state.layer_index != layer.layer_index or state.rows != rows:
        raise ValueError(
            f"verify layer state identity {(state.namespace, state.layer_index, state.rows)} does not match "
            f"{(layer.namespace, layer.layer_index, rows)}"
        )
    if isinstance(layer.attention, Qwen38TTNNGDN) != isinstance(state.attention, Qwen38TTNNGDNRowsState):
        raise TypeError(f"layer {layer.layer_index} verify attention state is {type(state.attention).__name__}")
    if (layer.ple is None) != (state.ple is None):
        raise ValueError("PLE rows state must be present exactly on the PLE layer")
    if state.moe.weights is not layer.mlp.weights or state.moe.rows != moe_rows:
        raise ValueError(
            f"verify MoE must be a rows-{moe_rows} instance over this layer's weights, got rows-{state.moe.rows}"
        )
    if (state.moe_input is None) != (state.moe.rows == CHUNK_ROWS):
        raise ValueError("verify MoE input slice must be present exactly when the MoE takes fewer than 32 rows")
    if state.gdn_step_anchor is not gdn_step_anchor:
        raise ValueError(
            f"layer {layer.layer_index} verify state gdn_step_anchor is {state.gdn_step_anchor}, expected {gdn_step_anchor}"
        )


def _validate_verify_state(model: Qwen38TTNNTextModel, verify: Qwen38TTNNVerifyState) -> None:
    if not isinstance(verify, Qwen38TTNNVerifyState) or verify._owner is not model._state_owner:
        raise ValueError("verify state was not allocated by this model owner")
    if verify.drafts not in SUPPORTED_DRAFTS or verify.rows != verify.drafts + 1:
        raise ValueError(f"verify state has k={verify.drafts} rows={verify.rows}")
    if len(verify.layers) != BACKBONE_LAYERS or verify.rows_constants.rows != verify.rows:
        raise ValueError(f"verify state must hold {BACKBONE_LAYERS} layer states over the {verify.rows}-row constants")
    if verify.qsa_verify_constants.rows != verify.rows or verify.accept_constants.drafts != verify.drafts:
        raise ValueError("verify constants were built for another row count")
    if verify.moe_rows != resolve_moe_rows(verify.rows, verify.moe_rows):
        raise ValueError(f"verify MoE row count {verify.moe_rows} is not admitted for {verify.rows} rows")
    gdn_layers = {layer.layer_index for layer in model.layers if isinstance(layer.attention, Qwen38TTNNGDN)}
    if not verify.gdn_step_anchor_layers <= gdn_layers:
        raise ValueError(
            f"GDN step anchor layers {sorted(verify.gdn_step_anchor_layers - gdn_layers)} are not GDN layers"
        )
    for layer, layer_state in zip(model.layers, verify.layers):
        _validate_layer_verify_state(
            layer,
            layer_state,
            verify.rows,
            verify.moe_rows,
            gdn_step_anchor=layer.layer_index in verify.gdn_step_anchor_layers,
        )
    model.model_io.embedding.validate_token_row(verify.token_row, label="verify token row")
    if _shape(verify.draft_lanes) != TOKEN_ROW_SHAPE or verify.draft_lanes.dtype != ttnn.float32:
        raise RuntimeError(
            f"verify draft lanes must be FP32 {TOKEN_ROW_SHAPE}, got {tensor_metadata(verify.draft_lanes)}"
        )
    if _shape(verify.accepted) != (1, 1, 1, 1) or verify.accepted.dtype != ttnn.float32:
        raise RuntimeError(f"verify accept scalar must be FP32 [1,1,1,1], got {tensor_metadata(verify.accepted)}")
    if not verify.ple_rows.active:
        raise RuntimeError("verify PLE rows were released")
    if verify.alignment is not None:
        _validate_layer_verify_state(
            verify.alignment.layer, verify.alignment.verify_state, verify.rows, verify.moe_rows
        )
        if _shape(verify.alignment.residual) != MTP_RESIDUAL_SHAPE:
            raise RuntimeError(
                f"alignment residual must be {MTP_RESIDUAL_SHAPE}, got {tensor_metadata(verify.alignment.residual)}"
            )


def _validate_mtp_components(model: Qwen38TTNNTextModel, mtp_components) -> tuple[Any, Any, Any]:
    layer, input_mixer, final_mixer = (
        mtp_components.decoder_layer,
        mtp_components.input_mixer,
        mtp_components.final_mixer,
    )
    if not isinstance(layer, Qwen38TTNNDecoderLayer) or layer.namespace is not Qwen38TTNNLayerNamespace.MTP:
        raise TypeError("MTP components must carry the MTP-namespace decoder layer")
    if layer.layer_type is not Qwen38TTNNLayerType.QSA:
        raise TypeError("the MTP decoder layer must be the QSA layer")
    if not isinstance(input_mixer, Qwen38TTNNMTPInput) or not isinstance(final_mixer, Qwen38TTNNFinalMixer):
        raise TypeError("MTP components must carry the exact input and final mixers")
    if final_mixer.weights.namespace != Qwen38TTNNLayerNamespace.MTP.value:
        raise ValueError("the MTP alignment needs the MTP-namespace final mixer")
    for name, component in (("MTP layer", layer), ("MTP input mixer", input_mixer), ("MTP final mixer", final_mixer)):
        if component.mesh_contract != model.mesh_contract:
            raise ValueError(f"{name} belongs to a different physical mesh contract")
    if layer.attention.allocated_context != model.allocated_context:
        raise ValueError("the MTP QSA layer must share the target's cache capacity")
    return layer, input_mixer, final_mixer


def allocate_verify_state(
    model: Qwen38TTNNTextModel,
    state: Qwen38TTNNTextModelGenericState,
    *,
    drafts: int = DEFAULT_DRAFTS,
    mtp_components=None,
    moe_rows: int | None = None,
    gdn_step_anchor_layers: Sequence[int] = (),
) -> Qwen38TTNNVerifyState:
    """Allocate the verify constants and every layer's verify buffers beside ``state`` (before any capture).

    ``mtp_components`` (the builder's ``decoder_layer`` / ``input_mixer`` / ``final_mixer``) enables the
    alignment rows; without it the pass reports ``first_draft = None`` and skips the MTP layer.  ``moe_rows``
    overrides every layer's verify MoE row count (:func:`resolve_moe_rows`); ``gdn_step_anchor_layers`` names
    the GDN layers whose commits run the committed rows through the 1-row FP32 step recurrence (the state
    re-anchor) instead of the chunk kernel.
    """

    if drafts not in SUPPORTED_DRAFTS:
        raise ValueError(f"MTP v2 verify admits k in {SUPPORTED_DRAFTS}, got {drafts!r}")
    if not isinstance(state, Qwen38TTNNTextModelGenericState) or state._owner is not model._state_owner:
        raise ValueError("generic text-model state was not allocated by this model owner")
    if model.rope_table is None or model.qsa_position_constants is None:
        raise RuntimeError("generic constants are missing; allocate the generic state through this owner")
    rows = drafts + 1
    moe_rows = resolve_moe_rows(rows, moe_rows)
    anchor_layers = frozenset(int(index) for index in gdn_step_anchor_layers)
    gdn_layers = {layer.layer_index for layer in model.layers if isinstance(layer.attention, Qwen38TTNNGDN)}
    if any(isinstance(index, bool) or type(index) is not int for index in gdn_step_anchor_layers) or not (
        anchor_layers <= gdn_layers
    ):
        raise ValueError(f"GDN step anchor layers must be GDN layer indices, got {list(gdn_step_anchor_layers)!r}")
    mtp = None if mtp_components is None else _validate_mtp_components(model, mtp_components)
    gdn = next(layer.attention for layer in model.layers if layer.layer_type is Qwen38TTNNLayerType.GDN)
    qsa = next(layer.attention for layer in model.layers if layer.layer_type is Qwen38TTNNLayerType.QSA)
    mesh_device, mesh_contract = model.mesh_device, model.mesh_contract
    actions: list[tuple[str, Callable[[], Any]]] = []
    try:
        rows_constants = gdn.allocate_rows_constants(rows)
        actions.append(("GDN rows constants", rows_constants.deallocate))
        chunk_constants = qsa_module.Qwen38TTNNQSAChunkConstants.build(
            mesh_device, mesh_contract, qsa.allocated_compressed_blocks
        )
        actions.append(("QSA chunk constants", chunk_constants.deallocate))
        verify_constants = qsa_module.Qwen38TTNNQSAVerifyConstants.build(
            mesh_device, mesh_contract, chunk_constants, rows=rows
        )
        actions.append(("QSA verify constants", verify_constants.deallocate))
        accept_constants = Qwen38TTNNAcceptConstants.build(mesh_device, mesh_contract, drafts=drafts)
        actions.append(("accept constants", accept_constants.deallocate))
        layers: list[Qwen38TTNNVerifyLayerState] = []
        for layer in model.layers:
            layer_state = _allocate_layer_verify_state(
                layer,
                rows,
                rows_constants,
                moe_rows=moe_rows,
                gdn_step_anchor=layer.layer_index in anchor_layers,
            )
            layers.append(layer_state)
            actions.append(
                (
                    f"layer {layer.layer_index} verify state",
                    lambda layer=layer, layer_state=layer_state: _release_layer_verify_state(layer, layer_state),
                )
            )
        token_row = model.model_io.embedding.upload_token_row(0)
        actions.append(("verify token row", lambda: _deallocate(token_row)))
        draft_lanes = _upload_replicated(
            mesh_device,
            mesh_contract,
            torch.full(TOKEN_ROW_SHAPE, float(ZERO_EMBEDDING_TOKEN)),
            ttnn.float32,
            ttnn.ROW_MAJOR_LAYOUT,
            label="verify draft lanes",
        )
        actions.append(("verify draft lanes", lambda: _deallocate(draft_lanes)))
        accepted = _upload_replicated(
            mesh_device,
            mesh_contract,
            torch.full((1, 1, 1, 1), float(rows - 1)),
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            label="verify accept scalar",
        )
        actions.append(("verify accept scalar", lambda: _deallocate(accepted)))
        ple_layer, ple_state = model.layers[PLE_CHECKPOINT_LAYER], layers[PLE_CHECKPOINT_LAYER].ple
        if ple_layer.ple is None or ple_state is None:
            raise RuntimeError("checkpoint layer 1 PLE owner/rows state is unavailable")
        ple_rows = ple_layer.ple.prepare_rows_input([0] * rows, ple_state)
        actions.append(("verify PLE rows", ple_rows.release))
        alignment = None
        if mtp is not None:
            mtp_layer, input_mixer, final_mixer = mtp
            generic_state = mtp_layer.allocate_generic_state()
            actions.append(("MTP layer generic state", lambda: mtp_layer.release_generic_state(generic_state)))
            mtp_verify = _allocate_layer_verify_state(mtp_layer, rows, rows_constants, moe_rows=moe_rows)
            actions.append(("MTP layer verify state", lambda: _release_layer_verify_state(mtp_layer, mtp_verify)))
            residual = _allocate_hidden_sharded_zeros(
                mesh_device, mesh_contract, MTP_RESIDUAL_SHAPE, label="MTP alignment residual"
            )
            actions.append(("MTP alignment residual", lambda: _deallocate(residual)))
            alignment = Qwen38TTNNVerifyAlignment(
                mtp_layer, input_mixer, final_mixer, generic_state, mtp_verify, residual
            )
        verify = Qwen38TTNNVerifyState(
            drafts=drafts,
            rows=rows,
            rows_constants=rows_constants,
            qsa_chunk_constants=chunk_constants,
            qsa_verify_constants=verify_constants,
            accept_constants=accept_constants,
            layers=tuple(layers),
            token_row=token_row,
            draft_lanes=draft_lanes,
            ple_rows=ple_rows,
            accepted=accepted,
            alignment=alignment,
            moe_rows=moe_rows,
            gdn_step_anchor_layers=anchor_layers,
            _owner=model._state_owner,
        )
        _validate_verify_state(model, verify)
        return verify
    except BaseException as error:
        _run_cleanup("verify state allocation", list(reversed(actions)), primary=error)
        raise


def release_verify_state(model: Qwen38TTNNTextModel, verify: Qwen38TTNNVerifyState) -> None:
    _validate_verify_state(model, verify)
    actions: list[tuple[str, Callable[[], Any]]] = []
    if verify.alignment is not None:
        alignment = verify.alignment
        actions.append(("MTP alignment residual", lambda: _deallocate(alignment.residual)))
        actions.append(
            ("MTP layer verify state", lambda: _release_layer_verify_state(alignment.layer, alignment.verify_state))
        )
        actions.append(
            ("MTP layer generic state", lambda: alignment.layer.release_generic_state(alignment.generic_state))
        )
    actions.append(("verify PLE rows", verify.ple_rows.release))
    actions.append(("verify accept scalar", lambda: _deallocate(verify.accepted)))
    actions.append(("verify draft lanes", lambda: _deallocate(verify.draft_lanes)))
    actions.append(("verify token row", lambda: _deallocate(verify.token_row)))
    for layer, layer_state in reversed(tuple(zip(model.layers, verify.layers))):
        actions.append(
            (
                f"layer {layer.layer_index} verify state",
                lambda layer=layer, layer_state=layer_state: _release_layer_verify_state(layer, layer_state),
            )
        )
    actions.append(("accept constants", verify.accept_constants.deallocate))
    actions.append(("QSA verify constants", verify.qsa_verify_constants.deallocate))
    actions.append(("QSA chunk constants", verify.qsa_chunk_constants.deallocate))
    actions.append(("GDN rows constants", verify.rows_constants.deallocate))
    _run_cleanup("verify state", actions)


def seed_verify_state_inplace(
    model: Qwen38TTNNTextModel, state: Qwen38TTNNTextModelGenericState, verify: Qwen38TTNNVerifyState, *, position: int
) -> None:
    """Eager mode switch (1-row generic decode at host-known ``P`` -> verify): the GDN FIR histories from the
    rings, the PLE history and n-gram context from the nine slots, the QSA raw histories from the raw-key rings
    (the alignment layer's from its own generic state, which the caller advanced to ``P`` as well).  The first
    pass afterwards runs with ``catch_up=False``; ``accepted`` is set to ``rows - 1`` for the record."""

    _validate_verify_state(model, verify)
    if isinstance(position, bool) or type(position) is not int or position < 0:
        raise ValueError(f"verify seed needs a non-negative int position, got {position!r}")
    if state.position.read() != position:
        raise RuntimeError(f"device position is {state.position.read()}, the caller says {position}")
    for layer, layer_state, layer_verify in zip(model.layers, state.layers, verify.layers):
        _seed_layer(layer, layer_state, layer_verify, position=position)
    if verify.alignment is not None:
        _seed_layer(
            verify.alignment.layer, verify.alignment.generic_state, verify.alignment.verify_state, position=position
        )
    write_verify_accepted(model, verify, verify.rows - 1)


def _seed_layer(layer, generic_state, layer_verify: Qwen38TTNNVerifyLayerState, *, position: int) -> None:
    if isinstance(layer.attention, Qwen38TTNNGDN):
        layer.attention.sync_rows_history_from_state(generic_state.attention, layer_verify.attention)
    else:
        layer.attention.sync_verify_raw_history_from_ring(
            generic_state.attention, layer_verify.attention, position=position
        )
    if layer_verify.ple is not None:
        layer_verify.ple.load_from_state(generic_state.ple)


def write_verify_accepted(model: Qwen38TTNNTextModel, verify: Qwen38TTNNVerifyState, accepted: int) -> None:
    """Host write of the accept scalar (outside any trace)."""

    if isinstance(accepted, bool) or type(accepted) is not int or not 0 <= accepted < verify.rows:
        raise ValueError(f"accept count must be an int in [0,{verify.rows}), got {accepted!r}")
    host = ttnn.from_torch(
        torch.full((1, 1, 1, 1), float(accepted)),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate_tensor_2d_mesh_mapper(model.mesh_device),
    )
    ttnn.copy_host_to_device_tensor(host, verify.accepted)


def write_verify_inputs(
    model: Qwen38TTNNTextModel, verify: Qwen38TTNNVerifyState, tokens: Sequence[int]
) -> tuple[tuple[int, int] | None, ...]:
    """Host writes of one pass's inputs (outside any trace): the R tokens ``[t_P, d_1 .. d_k]`` into the token row,
    the k drafts into the draft lanes, and their n-gram rows (looked up from the PLE rows state's committed
    context) into the persistent PLE rows.  Returns the R + 1 contexts (``contexts[c]`` after committing c rows)."""

    _validate_verify_state(model, verify)
    tokens = [int(token) for token in tokens]
    if len(tokens) != verify.rows:
        raise ValueError(f"a k={verify.drafts} verify pass takes {verify.rows} tokens, got {len(tokens)}")
    ple = model.layers[PLE_CHECKPOINT_LAYER].ple
    ple_state = verify.layers[PLE_CHECKPOINT_LAYER].ple
    if ple is None or ple_state is None:
        raise RuntimeError("checkpoint layer 1 PLE owner/rows state is unavailable")
    replicate = replicate_tensor_2d_mesh_mapper(model.mesh_device)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            model.model_io.embedding.host_verify_token_rows(tokens),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate,
        ),
        verify.token_row,
    )
    drafts = torch.full(TOKEN_ROW_SHAPE, float(ZERO_EMBEDDING_TOKEN))
    drafts[..., : verify.drafts] = torch.tensor(tokens[1:], dtype=torch.float32)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(drafts, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=replicate),
        verify.draft_lanes,
    )
    return write_verify_ple_rows(model, verify, tokens)


def write_verify_ple_rows(
    model: Qwen38TTNNTextModel, verify: Qwen38TTNNVerifyState, tokens: Sequence[int]
) -> tuple[tuple[int, int] | None, ...]:
    """The per-pass host segment of a traced chain: only the PLE n-gram rows of the R tokens (the token row and
    the draft lanes were assembled on device by the draft body).  Looked up sequentially from the PLE rows state's
    committed context; returns the R + 1 contexts."""

    _validate_verify_state(model, verify)
    tokens = [int(token) for token in tokens]
    if len(tokens) != verify.rows:
        raise ValueError(f"a k={verify.drafts} verify pass takes {verify.rows} tokens, got {len(tokens)}")
    ple = model.layers[PLE_CHECKPOINT_LAYER].ple
    ple_state = verify.layers[PLE_CHECKPOINT_LAYER].ple
    if ple is None or ple_state is None:
        raise RuntimeError("checkpoint layer 1 PLE owner/rows state is unavailable")
    rows, contexts = ple.host_rows(tokens, ple_state.token_context)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            rows.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(model.mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
        ),
        verify.ple_rows.embedding_rows,
    )
    verify.ple_rows.tokens = tuple(tokens)
    verify.ple_rows.contexts = contexts
    return contexts


def commit_verify_host(verify: Qwen38TTNNVerifyState, accepted: int) -> None:
    """Host bookkeeping after the readback: the PLE rows state's n-gram context moves to ``contexts[a + 1]``."""

    ple_state = verify.layers[PLE_CHECKPOINT_LAYER].ple
    if ple_state is None:
        raise RuntimeError("checkpoint layer 1 PLE rows state is unavailable")
    Qwen38TTNNPLE.commit_rows_host(ple_state, verify.ple_rows, accepted)


def _verify_readback(values, *, rows: int) -> Qwen38TTNNVerifyReadback:
    if values.numel() != READBACK_WIDTH:
        raise RuntimeError(f"verify readback has {values.numel()} lanes, expected {READBACK_WIDTH}")
    first_draft = int(values[2].item())
    return Qwen38TTNNVerifyReadback(
        accepted=int(values[0].item()),
        next_token=int(values[1].item()),
        first_draft=None if first_draft == ZERO_EMBEDDING_TOKEN else first_draft,
        argmaxes=tuple(int(value) for value in values[len(READBACK_FIXED_LANES) : len(READBACK_FIXED_LANES) + rows]),
    )


def read_verify_output(output: Qwen38TTNNVerifyOutput, *, rows: int) -> Qwen38TTNNVerifyReadback:
    """Host readback (outside any trace) of the verify row from coordinate 0 (the eager warm; a traced pass reads
    :func:`read_pass_row`)."""

    if not output.active:
        raise RuntimeError("verify output tensors were released")
    return _verify_readback(ttnn.to_torch(ttnn.get_device_tensors(output.readback)[0]).reshape(-1), rows=rows)


# --------------------------------------------------------------------------- the body


def _forward_layer_verify(
    layer: Qwen38TTNNDecoderLayer,
    residual,
    generic_state: Qwen38TTNNDecoderLayerGenericState,
    layer_verify: Qwen38TTNNVerifyLayerState,
    *,
    prepared_ple_rows: Qwen38TTNNPLERowsPreparedInput | None,
    rope_rows,
    qsa_verify: qsa_module.Qwen38TTNNQSAVerifyInputs,
    qsa_chunk_constants: qsa_module.Qwen38TTNNQSAChunkConstants,
    selectors: gdn_module.Qwen38TTNNRowsSelectors | None,
):
    """One layer of the verify pass on the 32-row tile; the input rows are consumed.

    Commit first, then the new rows: with ``selectors`` (``catch_up=True``) the PLE, the GDN and the QSA raw
    history commit the previous pass's accepted prefix out of their own persistent buffers before those buffers
    are overwritten.  PLE runs on the R real rows (a copy slice at row 0), the MoE on its admitted row count
    (a slice landing in the persistent ``moe_input``); both come back to the 32-row tile through an in-place
    pad whose result is a view (never deallocated; the owner is).  The GR reads walk branch-major <-> flat rows
    through ``ttnn.experimental.view`` (``flat_views``: the same tile pages, no permute or relayout kernels).
    """

    rows = layer_verify.rows
    if _shape(residual) != RESIDUAL_ROWS_SHAPE or residual.dtype != ttnn.bfloat16:
        raise RuntimeError(f"verify residual rows must be BF16 {RESIDUAL_ROWS_SHAPE}, got {tensor_metadata(residual)}")
    dram = ttnn.DRAM_MEMORY_CONFIG
    if layer.ple is not None:
        if prepared_ple_rows is None:
            raise ValueError("the PLE layer's verify pass needs its prepared persistent PLE rows")
        if selectors is not None:
            layer.ple.commit_rows(layer_verify.ple, selectors)
        generic_state.ple.token_context = None
        head = ttnn.slice(residual, (0, 0, 0, 0), (1, RESIDUAL_BRANCHES, rows, LOCAL_HIDDEN_SIZE), memory_config=dram)
        head.update_tensor_topology(residual.tensor_topology())
        injected = layer.ple.inject_rows(head, prepared_ple_rows, layer_verify.ple)  # consumes head
        _deallocate(residual)
        residual = _pad_rows(injected, rows, label="PLE-injected verify rows")
        residual_owner = injected
    else:
        if prepared_ple_rows is not None:
            raise ValueError("prepared PLE rows were supplied outside checkpoint layer 1")
        residual_owner = residual

    attention_input, attention_gr_state = layer.attention_gr.read_rows(residual, flat_views=True)
    if isinstance(layer.attention, Qwen38TTNNGDN):
        if selectors is not None:
            layer.attention.commit_rows(
                generic_state.attention,
                layer_verify.attention,
                selectors,
                step_committed_rows=layer_verify.gdn_step_anchor,
            )
        result = layer.attention.forward_rows(
            attention_input, generic_state.attention, layer_verify.attention, full_tile=True
        )
        _deallocate(result.final_state)
        attention_hidden = result.hidden_rows  # the 32-row output tile, rows past R exact zeros
        attention_owner = attention_hidden
    else:
        if selectors is not None:
            layer.attention.commit_verify(layer_verify.attention, selectors)
        attention_hidden = layer.attention.forward_verify_generic(
            attention_input,
            generic_state.attention,
            layer_verify.attention,
            cos=rope_rows.cos,
            sin=rope_rows.sin,
            block_start_cos=rope_rows.block_start_cos,
            block_start_sin=rope_rows.block_start_sin,
            verify=qsa_verify,
            constants=qsa_chunk_constants,
        )
        attention_owner = attention_hidden
    _deallocate(attention_input)
    if _shape(attention_hidden) != BLOCK_ROWS_SHAPE:
        raise RuntimeError(f"verify attention output has shape {_shape(attention_hidden)}, expected {BLOCK_ROWS_SHAPE}")

    residual = layer.attention_gr.write_rows(attention_hidden, attention_gr_state)
    # residual_owner owns the buffer the GR state's residual reads (the PLE layer's pad result is a view of it).
    _deallocate(attention_owner, residual_owner, attention_gr_state.injection)

    mlp_input, mlp_gr_state = layer.mlp_gr.read_rows(residual, flat_views=True)
    if layer_verify.moe_input is None:
        moe_input = mlp_input
    else:
        moe_rows = layer_verify.moe.rows
        landed = ttnn.slice(
            mlp_input, (0, 0, 0, 0), (1, 1, moe_rows, LOCAL_HIDDEN_SIZE), output_tensor=layer_verify.moe_input
        )
        if landed is not None and _tensor_key(landed) != _tensor_key(layer_verify.moe_input):
            raise RuntimeError("verify MoE input slice did not land in its persistent buffer")
        moe_input = layer_verify.moe_input
        moe_input.update_tensor_topology(mlp_input.tensor_topology())
    with layer.expert_streamer.layer(layer.layer_index, namespace=layer.namespace.value) as packed_experts:
        if not isinstance(packed_experts, tuple) or len(packed_experts) != 2:
            raise RuntimeError("BF4 streamer must yield exactly (packed_w0_w1, packed_w2)")
        mlp_result = layer_verify.moe.forward(moe_input, packed_experts[0], packed_experts[1])
    _deallocate(mlp_input)
    moe_hidden = mlp_result.hidden_sharded
    if layer_verify.moe_input is not None:
        moe_hidden = _pad_rows(moe_hidden, layer_verify.moe.rows, label="verify MoE output rows")
    if _shape(moe_hidden) != BLOCK_ROWS_SHAPE:
        raise RuntimeError(f"verify MoE output has shape {_shape(moe_hidden)}, expected {BLOCK_ROWS_SHAPE}")
    residual = layer.mlp_gr.write_rows(moe_hidden, mlp_gr_state)
    _deallocate(mlp_result.hidden_sharded, mlp_gr_state.residual, mlp_gr_state.injection)
    if _shape(residual) != RESIDUAL_ROWS_SHAPE:
        raise RuntimeError(f"verify layer output has shape {_shape(residual)}, expected {RESIDUAL_ROWS_SHAPE}")
    return residual


def _embed_rows(model: Qwen38TTNNTextModel, token_row):
    hidden = model.model_io.embedding.embed_device_token_rows(token_row)
    if _shape(hidden) != BLOCK_ROWS_SHAPE:
        raise RuntimeError(f"verify embedding rows have shape {_shape(hidden)}, expected {BLOCK_ROWS_SHAPE}")
    residual = ttnn.repeat_interleave(hidden, repeats=RESIDUAL_BRANCHES, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    _deallocate(hidden)
    if _shape(residual) != RESIDUAL_ROWS_SHAPE:
        raise RuntimeError(f"verify residual rows have shape {_shape(residual)}, expected {RESIDUAL_ROWS_SHAPE}")
    model.mesh_contract.validate_tensor(residual, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
    return residual


def _resolve_rows(model: Qwen38TTNNTextModel, hidden_rows, *, rows: int, sentinel_tail):
    """Final-mixer output rows -> the ``rows`` real rows -> LM head -> the FP32 ROW_MAJOR ``[1,1,1,32]`` row of
    per-row global greedy ids (lanes past ``rows`` hold ``ZERO_EMBEDDING_TOKEN`` from ``sentinel_tail``).

    The LM head, the local candidates (the argmax over the vocabulary, the head's cost, scales with the rows) and
    the resolve run on the real rows alone: the 32-row forms' first rows, bitwise.  The row slice at row 0 is a
    copy on this runtime (the PLE layer's form); the padding rows never reach the head.
    """

    lm_head = model.model_io.lm_head
    dram = ttnn.DRAM_MEMORY_CONFIG
    if rows == CHUNK_ROWS:
        head_rows = hidden_rows
    else:
        head_rows = ttnn.slice(hidden_rows, (0, 0, 0, 0), (1, 1, rows, LOCAL_HIDDEN_SIZE), memory_config=dram)
        head_rows.update_tensor_topology(hidden_rows.tensor_topology())
    logits = lm_head(head_rows)
    candidates = lm_head.greedy_candidates(logits, values_by_gather=True)
    _deallocate(logits.tensor)
    lanes = lm_head.resolve_greedy_rows_on_device(candidates)
    _deallocate(candidates.local_indices, candidates.local_values)
    if rows != CHUNK_ROWS:
        _deallocate(head_rows)
        padded = ttnn.concat([lanes, sentinel_tail], dim=3, memory_config=dram)
        _deallocate(lanes)
        lanes = padded
    if _shape(lanes) != TOKEN_ROW_SHAPE or lanes.dtype != ttnn.float32 or lanes.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise RuntimeError(
            f"resolved verify lanes must be FP32 ROW_MAJOR {TOKEN_ROW_SHAPE}, got {tensor_metadata(lanes)}"
        )
    return lanes


def _forward_alignment(
    model: Qwen38TTNNTextModel,
    verify: Qwen38TTNNVerifyState,
    roots,
    accept: Qwen38TTNNAcceptResult,
    argmax_lanes,
    *,
    rope_rows,
    qsa_verify,
    selectors,
):
    """The R alignment rows at P .. P + k: row j takes the target root at P + j and the target's own prediction
    there, ``argmax_j`` (the token at P + j + 1: the accepted draft ``d_{j+1}`` for j < a, ``t'`` for j = a), through
    the MTP layer; returns ``d_1'`` (lane ``a`` of its argmaxes) and lands row ``a`` of its residuals in
    ``alignment.residual``.

    The tokens must be the argmaxes, not the shifted verify tokens ``[d_1 .. d_k, t']``: that form gives row a the
    rejected draft ``d_{a+1}`` whenever a < k, so ``d_1'`` (and every draft grown from it) predicts the successor of
    a token the pass never committed.  Rows past a are speculative and rewritten by the next pass; rows past R stay
    zero-embedding rows.
    """

    alignment = verify.alignment
    constants = verify.accept_constants
    dram = ttnn.DRAM_MEMORY_CONFIG
    predicted = ttnn.slice(argmax_lanes, (0, 0, 0, 0), (1, 1, 1, verify.rows), memory_config=dram)
    shifted = ttnn.concat([predicted, constants.sentinel_tail], dim=3, memory_config=dram)
    _deallocate(predicted)
    if _shape(shifted) != TOKEN_ROW_SHAPE:
        raise RuntimeError(f"shifted alignment tokens have shape {_shape(shifted)}, expected {TOKEN_ROW_SHAPE}")
    shifted_tile = ttnn.to_layout(shifted, ttnn.TILE_LAYOUT, memory_config=dram)
    _deallocate(shifted)
    embedding_rows = model.model_io.embedding.embed_device_token_rows(shifted_tile)
    _deallocate(shifted_tile)
    mixed = alignment.input_mixer.rows(embedding_rows, roots)
    _deallocate(embedding_rows)
    residual = _forward_layer_verify(
        alignment.layer,
        mixed,
        alignment.generic_state,
        alignment.verify_state,
        prepared_ple_rows=None,
        rope_rows=rope_rows,
        qsa_verify=qsa_verify,
        qsa_chunk_constants=verify.qsa_chunk_constants,
        selectors=selectors,
    )
    hidden = alignment.final_mixer.rows(residual, flat_views=True)
    lanes = _resolve_rows(model, hidden, rows=verify.rows, sentinel_tail=constants.sentinel_tail)
    _deallocate(hidden)
    first_draft = ttnn.gather(lanes, 3, accept.accepted_index, memory_config=dram)
    _deallocate(lanes)
    select_residual_row(residual, accept.accepted_tile, constants, output=alignment.residual)
    _deallocate(residual)
    return first_draft


def forward_verify(
    model: Qwen38TTNNTextModel,
    verify: Qwen38TTNNVerifyState,
    state: Qwen38TTNNTextModelGenericState,
    *,
    catch_up: bool,
    observer: Callable[[str], Any] | None = None,
) -> Qwen38TTNNVerifyOutput:
    """One verify pass at the device position: fixed op sequence, fixed shapes, no host ints, no host I/O.

    ``catch_up=True`` commits the previous pass's accepted prefix first (from ``verify.accepted``); the first
    pass after :func:`seed_verify_state_inplace` runs with ``catch_up=False``.  ``P <- P + a + 1`` is the last
    op.  Any failure poisons the model owner: the in-place state cannot be rolled back.  ``observer`` (eager
    diagnostics only, never inside a capture) is called with a stage label after the prologue, after every layer,
    after the head, the accept, the alignment and the position update.
    """

    _validate_verify_state(model, verify)
    if not isinstance(catch_up, bool):
        raise TypeError(f"catch_up must be a bool, got {catch_up!r}")
    if model.poisoned:
        raise RuntimeError("the text model is poisoned")
    dram = ttnn.DRAM_MEMORY_CONFIG
    chunk_constants = verify.qsa_chunk_constants
    processed_layers = 0

    def stage(name: str) -> None:
        if observer is not None:
            observer(f"verify:{name}")

    try:
        selectors = gdn_module.build_rows_selectors(verify.accepted, verify.rows_constants) if catch_up else None
        index_row = state.position.index_row()
        index_rows = ttnn.add(index_row, chunk_constants.arange32_lanes, memory_config=dram)
        block_start_row = state.position.block_start_index_row(index_row)
        block_start_rows = ttnn.add(block_start_row, chunk_constants.block_start_lanes, memory_config=dram)
        rope = model.rope_table.rows_chunk(index_rows, block_start_rows)
        _deallocate(index_row, index_rows, block_start_row, block_start_rows)
        qsa_verify = qsa_module.derive_qsa_verify_inputs(
            state.position.scalar, model.qsa_position_constants, verify.qsa_verify_constants
        )
        residual = _embed_rows(model, verify.token_row)
        stage("prologue")
        for layer_index in range(BACKBONE_LAYERS):
            residual = _forward_layer_verify(
                model.layers[layer_index],
                residual,
                state.layers[layer_index],
                verify.layers[layer_index],
                prepared_ple_rows=verify.ple_rows if layer_index == PLE_CHECKPOINT_LAYER else None,
                rope_rows=rope,
                qsa_verify=qsa_verify,
                qsa_chunk_constants=chunk_constants,
                selectors=selectors,
            )
            processed_layers += 1
            stage(f"layer-{layer_index}")
        hidden = model.final_mixer.rows(residual, flat_views=True)
        argmax_lanes = _resolve_rows(
            model, hidden, rows=verify.rows, sentinel_tail=verify.accept_constants.sentinel_tail
        )
        _deallocate(hidden)
        stage("head")
        accept = accept_rows(argmax_lanes, verify.draft_lanes, verify.accept_constants)
        stage("accept")
        if verify.alignment is None:
            first_draft = verify.accept_constants.sentinel_lane
        else:
            first_draft = _forward_alignment(
                model,
                verify,
                residual,
                accept,
                argmax_lanes,
                rope_rows=rope,
                qsa_verify=qsa_verify,
                selectors=selectors,
            )
        stage("alignment")
        _deallocate(residual)
        readback = ttnn.concat(
            [accept.accepted_lane, accept.next_token, first_draft, argmax_lanes], dim=3, memory_config=dram
        )
        if _shape(readback) != (1, 1, 1, READBACK_WIDTH) or readback.dtype != ttnn.float32:
            raise RuntimeError(
                f"verify readback must be FP32 [1,1,1,{READBACK_WIDTH}], got {tensor_metadata(readback)}"
            )
        if verify.alignment is not None:
            _deallocate(first_draft)
        _deallocate(argmax_lanes)
        landed = ttnn.copy(accept.accepted_tile, verify.accepted)
        if landed is not None and _tensor_key(landed) != _tensor_key(verify.accepted):
            raise RuntimeError("verify accept count did not land in its persistent buffer")
        # P <- P + a + 1: exact UINT32 adds into the resident scalar, the body's last op.
        key = _tensor_key(state.position.scalar)
        advanced = ttnn.add(state.position.scalar, accept.accepted_index, memory_config=dram)
        advanced_next = ttnn.add(advanced, 1, memory_config=dram)
        copied = ttnn.copy(advanced_next, state.position.scalar)
        if _tensor_key(state.position.scalar) != key or (copied is not None and _tensor_key(copied) != key):
            raise RuntimeError("verify position advance did not write the resident scalar in place")
        _deallocate(advanced, advanced_next)
        accept.deallocate()
        qsa_verify.deallocate()
        rope.deallocate()
        if selectors is not None:
            selectors.deallocate()
        stage("epilogue")
        return Qwen38TTNNVerifyOutput(readback)
    except BaseException as error:
        model._mark_poisoned("forward_verify", processed_layers, error)


def capture_verify(
    model: Qwen38TTNNTextModel,
    verify: Qwen38TTNNVerifyState,
    state: Qwen38TTNNTextModelGenericState,
    *,
    catch_up: bool,
    guard: Callable[[str], AbstractContextManager[Any]],
    cq_id: int = 0,
) -> tuple[int, Qwen38TTNNVerifyOutput]:
    """Capture one verify body (``catch_up`` False for the first pass after a seed, True for every later pass);
    returns the trace id and the output whose readback address every replay rewrites.

    Same discipline as the model's captures: ``ttnn.corruptible_allocation_scope`` and the caller's no-host-I/O
    ``guard``.  Capture records without executing, so the device state is unchanged.
    """

    with ttnn.corruptible_allocation_scope(model.mesh_device):
        trace_id = ttnn.begin_trace_capture(model.mesh_device, cq_id=cq_id)
        with guard(f"verify capture catch_up={catch_up}"):
            output = forward_verify(model, verify, state, catch_up=catch_up)
        ttnn.end_trace_capture(model.mesh_device, trace_id, cq_id=cq_id)
    return trace_id, output


# --------------------------------------------------------------------------- the commit body (split form)


def forward_commit(
    model: Qwen38TTNNTextModel,
    verify: Qwen38TTNNVerifyState,
    state: Qwen38TTNNTextModelGenericState,
    *,
    observer: Callable[[str], Any] | None = None,
):
    """The commits of one pass as their own body: every layer's PLE / GDN / QSA commit of the previous pass's
    accepted prefix (``verify.accepted``), then the MTP alignment layer's raw-history commit.

    ``forward_verify(catch_up=True)`` runs these at the start of each layer; this body runs them all first so a
    pass can be ``draft -> commit -> forward_verify(catch_up=False)``: the draft body derives its history from the
    still-uncommitted alignment window, and the commit replay overlaps the host's PLE lookup of the draft ids.
    No host tensor, no position change.  ``observer`` (eager diagnostics only) is called after every layer's commit.
    """

    _validate_verify_state(model, verify)
    if model.poisoned:
        raise RuntimeError("the text model is poisoned")
    processed_layers = 0

    def stage(name: str) -> None:
        if observer is not None:
            observer(f"commit:{name}")

    try:
        selectors = gdn_module.build_rows_selectors(verify.accepted, verify.rows_constants)
        stage("selectors")
        for layer, generic_state, layer_verify in zip(model.layers, state.layers, verify.layers):
            if layer.ple is not None:
                layer.ple.commit_rows(layer_verify.ple, selectors)
                generic_state.ple.token_context = None
            if isinstance(layer.attention, Qwen38TTNNGDN):
                layer.attention.commit_rows(
                    generic_state.attention,
                    layer_verify.attention,
                    selectors,
                    step_committed_rows=layer_verify.gdn_step_anchor,
                )
            else:
                layer.attention.commit_verify(layer_verify.attention, selectors)
            processed_layers += 1
            stage(f"layer-{layer.layer_index}")
        if verify.alignment is not None:
            verify.alignment.layer.attention.commit_verify(verify.alignment.verify_state.attention, selectors)
        stage("alignment")
        selectors.deallocate()
    except BaseException as error:
        model._mark_poisoned("forward_commit", processed_layers, error)


def capture_commit(
    model: Qwen38TTNNTextModel,
    verify: Qwen38TTNNVerifyState,
    state: Qwen38TTNNTextModelGenericState,
    *,
    guard: Callable[[str], AbstractContextManager[Any]],
    cq_id: int = 0,
) -> int:
    with ttnn.corruptible_allocation_scope(model.mesh_device):
        trace_id = ttnn.begin_trace_capture(model.mesh_device, cq_id=cq_id)
        with guard("commit capture"):
            forward_commit(model, verify, state)
        ttnn.end_trace_capture(model.mesh_device, trace_id, cq_id=cq_id)
    return trace_id


# --------------------------------------------------------------------------- the draft body (design 2.5)


@dataclass(frozen=True)
class Qwen38TTNNDraftState:
    """Fixed-address buffers of the draft body: the MTP layer at one real row on the verify path's 32-row operands.

    ``qsa_state`` is the MTP layer's own draft raw history / raw rows (the alignment state keeps its window for
    the pass's real commit); ``qsa_constants`` the rows = 1 verify constants; ``layer_state`` the rows = 1 verify
    layer state over the MTP layer (the layer's own rows-1 MoE behind a persistent ``[1,1,1,640]`` input slice);
    ``advance_selectors`` the constant a = 0 selectors (one committed row: the previous draft row joins the
    history); ``sentinel_rest`` FP32 ROW_MAJOR ``[1,1,1,31]`` of ``ZERO_EMBEDDING_TOKEN``; ``pass_row`` FP32
    ROW_MAJOR ``[1,1,1,PASS_ROW_WIDTH]`` = the verify readback row this draft followed, then the next verify pass's
    ``[t', d_1 .. d_k, -1 ...]`` (device-written: the pass's one host readback).
    """

    drafts: int
    qsa_state: Any
    qsa_constants: qsa_module.Qwen38TTNNQSAVerifyConstants
    layer_state: Qwen38TTNNVerifyLayerState
    zero_accept: Any
    advance_selectors: gdn_module.Qwen38TTNNRowsSelectors
    sentinel_rest: Any
    pass_row: Any
    _owner: object = field(repr=False, compare=False)


def _validate_draft_state(model: Qwen38TTNNTextModel, verify: Qwen38TTNNVerifyState, draft: Qwen38TTNNDraftState):
    _validate_verify_state(model, verify)
    if verify.alignment is None:
        raise ValueError("the draft body needs the verify state's MTP alignment components")
    if not isinstance(draft, Qwen38TTNNDraftState) or draft._owner is not model._state_owner:
        raise ValueError("draft state was not allocated by this model owner")
    if draft.drafts != verify.drafts or draft.qsa_constants.rows != 1 or draft.advance_selectors.rows != verify.rows:
        raise ValueError(f"draft state (k={draft.drafts}) does not match the verify state (k={verify.drafts})")
    if draft.qsa_constants.chunk is not verify.qsa_chunk_constants:
        raise ValueError("draft QSA constants must sit beside the verify state's chunk constants")
    _validate_layer_verify_state(verify.alignment.layer, draft.layer_state, 1, 1)
    if draft.layer_state.attention is not draft.qsa_state or draft.layer_state.moe is not verify.alignment.layer.mlp:
        raise ValueError("draft layer state must run the draft QSA state through the MTP layer's own rows-1 MoE")
    for name, tensor, shape in (
        ("draft sentinel rest", draft.sentinel_rest, (1, 1, 1, CHUNK_ROWS - 1)),
        ("draft pass row", draft.pass_row, (1, 1, 1, PASS_ROW_WIDTH)),
    ):
        if _shape(tensor) != shape or tensor.dtype != ttnn.float32 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError(f"{name} must be FP32 ROW_MAJOR {shape}, got {tensor_metadata(tensor)}")
    if _shape(draft.zero_accept) != (1, 1, 1, 1) or draft.zero_accept.dtype != ttnn.float32:
        raise RuntimeError(f"draft zero accept must be FP32 [1,1,1,1], got {tensor_metadata(draft.zero_accept)}")


def allocate_draft_state(model: Qwen38TTNNTextModel, verify: Qwen38TTNNVerifyState) -> Qwen38TTNNDraftState:
    """Allocate the draft body's buffers beside ``verify`` (before any capture); needs the alignment components."""

    _validate_verify_state(model, verify)
    if verify.alignment is None:
        raise ValueError("the draft body needs the verify state's MTP alignment components")
    layer = verify.alignment.layer
    mesh_device, mesh_contract = model.mesh_device, model.mesh_contract
    actions: list[tuple[str, Callable[[], Any]]] = []
    try:
        qsa_state = layer.attention.allocate_verify_state()
        actions.append(("draft QSA state", lambda: layer.attention.release_verify_state(qsa_state)))
        qsa_constants = qsa_module.Qwen38TTNNQSAVerifyConstants.build(
            mesh_device, mesh_contract, verify.qsa_chunk_constants, rows=1
        )
        actions.append(("draft QSA constants", qsa_constants.deallocate))
        moe_input = _allocate_hidden_sharded_zeros(
            mesh_device, mesh_contract, (1, 1, 1, LOCAL_HIDDEN_SIZE), label="draft MoE input"
        )
        actions.append(("draft MoE input", lambda: _deallocate(moe_input)))
        layer_state = Qwen38TTNNVerifyLayerState(
            layer.namespace, layer.layer_index, 1, qsa_state, None, layer.mlp, moe_input
        )
        zero_accept = _upload_replicated(
            mesh_device,
            mesh_contract,
            torch.zeros((1, 1, 1, 1)),
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            label="draft zero accept",
        )
        actions.append(("draft zero accept", lambda: _deallocate(zero_accept)))
        advance_selectors = gdn_module.build_rows_selectors(zero_accept, verify.rows_constants)
        actions.append(("draft advance selectors", advance_selectors.deallocate))
        sentinel_rest = _upload_replicated(
            mesh_device,
            mesh_contract,
            torch.full((1, 1, 1, CHUNK_ROWS - 1), float(ZERO_EMBEDDING_TOKEN)),
            ttnn.float32,
            ttnn.ROW_MAJOR_LAYOUT,
            label="draft sentinel rest",
        )
        actions.append(("draft sentinel rest", lambda: _deallocate(sentinel_rest)))
        pass_row = _upload_replicated(
            mesh_device,
            mesh_contract,
            torch.full((1, 1, 1, PASS_ROW_WIDTH), float(ZERO_EMBEDDING_TOKEN)),
            ttnn.float32,
            ttnn.ROW_MAJOR_LAYOUT,
            label="draft pass row",
        )
        actions.append(("draft pass row", lambda: _deallocate(pass_row)))
        draft = Qwen38TTNNDraftState(
            drafts=verify.drafts,
            qsa_state=qsa_state,
            qsa_constants=qsa_constants,
            layer_state=layer_state,
            zero_accept=zero_accept,
            advance_selectors=advance_selectors,
            sentinel_rest=sentinel_rest,
            pass_row=pass_row,
            _owner=model._state_owner,
        )
        _validate_draft_state(model, verify, draft)
        return draft
    except BaseException as error:
        _run_cleanup("draft state allocation", list(reversed(actions)), primary=error)
        raise


def release_draft_state(model: Qwen38TTNNTextModel, verify: Qwen38TTNNVerifyState, draft: Qwen38TTNNDraftState):
    """Release the draft buffers; the MTP layer's own MoE is not owned here and stays."""

    _validate_draft_state(model, verify, draft)
    layer = verify.alignment.layer
    _run_cleanup(
        "draft state",
        [
            ("draft pass row", lambda: _deallocate(draft.pass_row)),
            ("draft sentinel rest", lambda: _deallocate(draft.sentinel_rest)),
            ("draft advance selectors", draft.advance_selectors.deallocate),
            ("draft zero accept", lambda: _deallocate(draft.zero_accept)),
            ("draft MoE input", lambda: _deallocate(draft.layer_state.moe_input)),
            ("draft QSA constants", draft.qsa_constants.deallocate),
            ("draft QSA state", lambda: layer.attention.release_verify_state(draft.qsa_state)),
        ],
    )


def _token_row_from_lane(model: Qwen38TTNNTextModel, lane, sentinel_rest):
    """The FP32 TILE ``[1,1,1,32]`` token row of one device id: lane 0 = the id, lanes 1..31 the zero-embedding
    sentinel (a concat and a relayout: the id is copied, never computed)."""

    dram = ttnn.DRAM_MEMORY_CONFIG
    lanes = ttnn.concat([lane, sentinel_rest], dim=3, memory_config=dram)
    token_row = ttnn.to_layout(lanes, ttnn.TILE_LAYOUT, memory_config=dram)
    _deallocate(lanes)
    model.model_io.embedding.validate_token_row(token_row, label="draft token row")
    return token_row


def _lane(tensor, index: int):
    """A 32-bit copy of lane ``index`` of an FP32 ROW_MAJOR ``[1,1,1,W]`` row as ``[1,1,1,1]``."""

    lane = ttnn.slice(tensor, (0, 0, 0, index), (1, 1, 1, index + 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if _shape(lane) != (1, 1, 1, 1) or lane.dtype != ttnn.float32 or lane.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise RuntimeError(f"lane {index} slice must be FP32 ROW_MAJOR [1,1,1,1], got {tensor_metadata(lane)}")
    return lane


def _land(source, target, *, label: str) -> None:
    landed = ttnn.copy(source, target)
    if landed is not None and _tensor_key(landed) != _tensor_key(target):
        raise RuntimeError(f"{label} did not land in its persistent buffer")


def forward_draft_history(model: Qwen38TTNNTextModel, verify: Qwen38TTNNVerifyState, draft: Qwen38TTNNDraftState):
    """The draft body's first op: the MTP layer's draft raw history <- the alignment window selected with the
    previous pass's accept count (the alignment state is read only).  As its own trace (``derive_history=False``
    on the draft body) it ends the draft's only read of a buffer the commit body rewrites, so a commit on a second
    command queue can start right behind it while the draft rows run (:class:`Qwen38TTNNCommitQueue`)."""

    _validate_draft_state(model, verify, draft)
    if model.poisoned:
        raise RuntimeError("the text model is poisoned")
    alignment = verify.alignment
    try:
        selectors = gdn_module.build_rows_selectors(verify.accepted, verify.rows_constants)
        alignment.layer.attention.commit_verify(alignment.verify_state.attention, selectors, target=draft.qsa_state)
        selectors.deallocate()
    except BaseException as error:
        model._mark_poisoned("forward_draft_history", 0, error)


def capture_draft_history(
    model: Qwen38TTNNTextModel,
    verify: Qwen38TTNNVerifyState,
    draft: Qwen38TTNNDraftState,
    *,
    guard: Callable[[str], AbstractContextManager[Any]],
    cq_id: int = 0,
) -> int:
    with ttnn.corruptible_allocation_scope(model.mesh_device):
        trace_id = ttnn.begin_trace_capture(model.mesh_device, cq_id=cq_id)
        with guard(f"draft history capture k={verify.drafts}"):
            forward_draft_history(model, verify, draft)
        ttnn.end_trace_capture(model.mesh_device, trace_id, cq_id=cq_id)
    return trace_id


def forward_draft(
    model: Qwen38TTNNTextModel,
    verify: Qwen38TTNNVerifyState,
    draft: Qwen38TTNNDraftState,
    state: Qwen38TTNNTextModelGenericState,
    verify_output: Qwen38TTNNVerifyOutput,
    *,
    derive_history: bool = True,
    observer: Callable[[str], Any] | None = None,
) -> None:
    """The k - 1 draft rows of one pass at the device position ``P`` (after the previous verify's ``P <- P + a + 1``).

    Row i (i = 1 .. k - 1) embeds ``d_i`` from the device token chain (``d_1`` = lane 2 of the verify readback
    row, later ids the previous row's row-0 argmax), mixes it with the previous MTP residual (``alignment.residual``
    for row 1), runs the MTP QSA layer at MTP position ``P + i - 1`` on the verify path with one real row (KV row and
    compressed blocks written by absolute position, raw history in the draft QSA state, advanced by one committed
    row between rows), the rows-1 MoE, the final mixer, the LM head and the on-device resolve, and emits ``d_{i+1}``.
    Before the rows the draft raw history is derived from the alignment window with the previous pass's accept
    count (the alignment state is read only).  Afterwards the next verify pass's inputs are assembled on device:
    ``verify.token_row`` = ``[t', d_1 .. d_k, -1 ...]`` and ``verify.draft_lanes`` = ``[d_1 .. d_k, -1 ...]``, and
    the pass row ``draft.pass_row`` = the verify readback row followed by the token lanes (the host's one readback
    per pass: the accept and the ids to look up).  No host tensor, no position change, no id through an FPU or
    reduce stage.  ``derive_history=False`` leaves the history derivation to :func:`forward_draft_history` (its own
    trace, run first).  ``observer`` (eager diagnostics only) is called after the history derivation, after every
    row's position inputs, embedding + mixer, MTP layer and head, and after the assembly.
    """

    def stage(name: str) -> None:
        if observer is not None:
            observer(f"draft:{name}")

    _validate_draft_state(model, verify, draft)
    if not isinstance(derive_history, bool):
        raise TypeError(f"derive_history must be a bool, got {derive_history!r}")
    if not isinstance(verify_output, Qwen38TTNNVerifyOutput) or not verify_output.active:
        raise ValueError("the draft body needs the live verify output whose readback row it reads")
    if _shape(verify_output.readback) != (1, 1, 1, READBACK_WIDTH) or verify_output.readback.dtype != ttnn.float32:
        raise RuntimeError(
            f"verify readback must be FP32 [1,1,1,{READBACK_WIDTH}], got {tensor_metadata(verify_output.readback)}"
        )
    if model.poisoned:
        raise RuntimeError("the text model is poisoned")
    alignment = verify.alignment
    qsa = alignment.layer.attention
    lm_head = model.model_io.lm_head
    constants = verify.accept_constants
    chunk_constants = verify.qsa_chunk_constants
    dram = ttnn.DRAM_MEMORY_CONFIG
    processed_rows = 0
    try:
        if derive_history:
            forward_draft_history(model, verify, draft)
        next_token = _lane(verify_output.readback, READBACK_FIXED_LANES.index("next_token"))
        drafts = [_lane(verify_output.readback, READBACK_FIXED_LANES.index("first_draft"))]
        # One real row per draft step: the mixers, the LM head and the resolve run their 1-row forms (row 0 of the
        # rows forms bitwise), the MTP layer its 32-row verify form on the row padded in place; only the layer's
        # input tile and its output are 32 rows.
        residual_row = alignment.residual  # the persistent row: never deallocated here
        stage("history")
        for row in range(verify.drafts - 1):
            if row:
                qsa.commit_verify(draft.qsa_state, draft.advance_selectors)  # the previous draft row joins the history
            position = ttnn.add(state.position.scalar, row, memory_config=dram)
            index_row = ttnn.multiply(state.position.ones_row, position, memory_config=dram)
            block_start_row = state.position.block_start_index_row(index_row)
            index_rows = ttnn.add(index_row, chunk_constants.arange32_lanes, memory_config=dram)
            block_start_rows = ttnn.add(block_start_row, chunk_constants.block_start_lanes, memory_config=dram)
            rope = model.rope_table.rows_chunk(index_rows, block_start_rows)
            _deallocate(index_row, block_start_row, index_rows, block_start_rows)
            qsa_inputs = qsa_module.derive_qsa_verify_inputs(
                position, model.qsa_position_constants, draft.qsa_constants, single_row=True
            )
            stage(f"row-{row}:inputs")
            token_row = _token_row_from_lane(model, drafts[-1], draft.sentinel_rest)
            embedding = model.model_io.embedding.embed_device_token(token_row)
            _deallocate(token_row)
            mixed_row = alignment.input_mixer(embedding, residual_row)
            _deallocate(embedding)
            if residual_row is not alignment.residual:
                _deallocate(residual_row)
            # The layer consumes its input tile: the pad is a view of mixed_row, so the layer's release of the view
            # is the one release of that buffer (mixed_row itself is never deallocated).
            mixed = _pad_rows(mixed_row, 1, label="MTP draft mixed row")
            stage(f"row-{row}:embed-mixer")
            residual = _forward_layer_verify(
                alignment.layer,
                mixed,
                alignment.generic_state,
                draft.layer_state,
                prepared_ple_rows=None,
                rope_rows=rope,
                qsa_verify=qsa_inputs,
                qsa_chunk_constants=chunk_constants,
                selectors=None,
            )
            stage(f"row-{row}:layer")
            residual_row = ttnn.slice(
                residual, (0, 0, 0, 0), (1, RESIDUAL_BRANCHES, 1, LOCAL_HIDDEN_SIZE), memory_config=dram
            )  # row 0: a copy on this runtime (the PLE layer's slice form)
            residual_row.update_tensor_topology(residual.tensor_topology())
            _deallocate(residual)
            if _shape(residual_row) != MTP_RESIDUAL_SHAPE:
                raise RuntimeError(
                    f"draft residual row has shape {_shape(residual_row)}, expected {MTP_RESIDUAL_SHAPE}"
                )
            hidden = alignment.final_mixer(residual_row)
            logits = lm_head(hidden)
            candidates = lm_head.greedy_candidates(logits, values_by_gather=True)
            _deallocate(logits.tensor, hidden)
            resolved = lm_head.resolve_greedy_on_device(candidates)  # FP32 TILE token row, column 0 = the id
            _deallocate(candidates.local_indices, candidates.local_values)
            resolved_lanes = ttnn.to_layout(resolved, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
            _deallocate(resolved)
            drafts.append(_lane(resolved_lanes, 0))
            _deallocate(resolved_lanes, position)
            qsa_inputs.deallocate()
            rope.deallocate()
            processed_rows += 1
            stage(f"row-{row}:head")
        if residual_row is not alignment.residual:
            _deallocate(residual_row)
        token_lanes = ttnn.concat([next_token, *drafts, constants.sentinel_tail], dim=3, memory_config=dram)
        draft_lanes = ttnn.concat(
            [*drafts, constants.sentinel_tail, constants.sentinel_lane], dim=3, memory_config=dram
        )
        pass_row = ttnn.concat([verify_output.readback, token_lanes], dim=3, memory_config=dram)
        for name, lanes, shape in (
            ("assembled verify token lanes", token_lanes, TOKEN_ROW_SHAPE),
            ("assembled draft lanes", draft_lanes, TOKEN_ROW_SHAPE),
            ("assembled pass row", pass_row, (1, 1, 1, PASS_ROW_WIDTH)),
        ):
            if _shape(lanes) != shape or lanes.dtype != ttnn.float32 or lanes.layout != ttnn.ROW_MAJOR_LAYOUT:
                raise RuntimeError(f"{name} must be FP32 ROW_MAJOR {shape}, got {tensor_metadata(lanes)}")
        _land(pass_row, draft.pass_row, label="assembled pass row")
        token_tile = ttnn.to_layout(token_lanes, ttnn.TILE_LAYOUT, memory_config=dram)
        model.model_io.embedding.validate_token_row(token_tile, label="assembled verify token row")
        _land(token_tile, verify.token_row, label="assembled verify token row")
        _land(draft_lanes, verify.draft_lanes, label="assembled draft lanes")
        _deallocate(token_lanes, token_tile, draft_lanes, pass_row, next_token, *drafts)
        stage("assemble")
    except BaseException as error:
        model._mark_poisoned("forward_draft", processed_rows, error)


def capture_draft(
    model: Qwen38TTNNTextModel,
    verify: Qwen38TTNNVerifyState,
    draft: Qwen38TTNNDraftState,
    state: Qwen38TTNNTextModelGenericState,
    verify_output: Qwen38TTNNVerifyOutput,
    *,
    guard: Callable[[str], AbstractContextManager[Any]],
    cq_id: int = 0,
    derive_history: bool = True,
) -> int:
    """Capture the draft body (after the verify body whose readback address it reads); returns the trace id."""

    with ttnn.corruptible_allocation_scope(model.mesh_device):
        trace_id = ttnn.begin_trace_capture(model.mesh_device, cq_id=cq_id)
        with guard(f"draft capture k={verify.drafts}"):
            forward_draft(model, verify, draft, state, verify_output, derive_history=derive_history)
        ttnn.end_trace_capture(model.mesh_device, trace_id, cq_id=cq_id)
    return trace_id


def read_pass_row(
    verify: Qwen38TTNNVerifyState, draft: Qwen38TTNNDraftState
) -> tuple[Qwen38TTNNVerifyReadback, tuple[int, ...]]:
    """Host readback (outside any trace, after a verify and the draft that followed it) of the pass row from
    coordinate 0: the verify's ``[a, t', d_1', argmaxes]`` and the assembled next verify tokens ``[t', d_1 .. d_k]``
    (whose first two lanes must be the verify's ``t'`` and ``d_1'``: the device token chain starts there)."""

    values = ttnn.to_torch(ttnn.get_device_tensors(draft.pass_row)[0]).reshape(-1)
    if values.numel() != PASS_ROW_WIDTH:
        raise RuntimeError(f"pass row readback has {values.numel()} lanes, expected {PASS_ROW_WIDTH}")
    readback = _verify_readback(values[:READBACK_WIDTH], rows=verify.rows)
    lanes = values[READBACK_WIDTH:]
    tokens = tuple(int(value) for value in lanes[: verify.rows])
    if any(token < 0 for token in tokens) or any(int(value) != ZERO_EMBEDDING_TOKEN for value in lanes[verify.rows :]):
        raise RuntimeError(f"pass row token lanes hold {lanes.tolist()}, expected {verify.rows} ids then the sentinel")
    if tokens[0] != readback.next_token or tokens[1] != readback.first_draft:
        expected = (readback.next_token, readback.first_draft)
        raise RuntimeError(f"pass row token lanes start {tokens[:2]}, its verify row says {expected}")
    return readback, tokens


def verify_pass_fits(position: int, allocated_context: int) -> bool:
    """The verify path writes the KV block at ``P & ~31`` and the next one; both must exist."""

    return (position & ~(CHUNK_ROWS - 1)) + 2 * CHUNK_ROWS <= allocated_context


# --------------------------------------------------------------------------- the pass loop


@dataclass(frozen=True)
class Qwen38TTNNMTPTraces:
    """The captured traces of one chain.  ``verify_catch_up`` (``forward_verify(catch_up=True)``) or ``commit``
    (``forward_commit``) is set, never both: the single-trace form runs ``draft -> verify_catch_up``, the split form
    ``draft -> commit -> verify_first`` so the commit replay overlaps the host's PLE lookup.  ``draft_history`` is
    the draft body's history derivation captured on its own (``forward_draft_history``; the draft trace then ran
    ``derive_history=False``): the two-command-queue form's fence point."""

    verify_first: int
    draft: int
    verify_catch_up: int | None = None
    commit: int | None = None
    draft_history: int | None = None

    def __post_init__(self) -> None:
        if (self.verify_catch_up is None) == (self.commit is None):
            raise ValueError("exactly one of verify_catch_up / commit must be captured")


class Qwen38TTNNCommitQueue:
    """The commit trace on its own command queue, overlapping the draft rows on the main queue.

    Two fences: the commit must not start before the draft's history derivation has read the alignment window the
    commit rewrites (``history_derived`` records that point on the main queue after the draft-history trace;
    ``enqueue_commit`` makes the commit queue wait for it, replays the commit there non-blocking and records the
    committed event), and the next verify must not start before the commit is done (``wait_committed`` makes the
    main queue wait for that event).  After the derivation the draft rows and the commit touch disjoint buffers
    (the draft: its own QSA / layer state, the MTP layer's caches, the verify inputs and the pass row; the commit:
    the backbone layers' rows histories and generic states, the alignment window).  What this does not settle is
    the traces' transient buffers, allocated at capture time from one allocator: two traces replaying at once
    share those addresses unless the captures kept them apart, so the form is a measurement until the runtime
    keeps them apart (the exactness gates of a run are the detector).
    """

    def __init__(self, mesh_device, *, cq_id: int = 1, main_cq_id: int = 0) -> None:
        if not (isinstance(cq_id, int) and isinstance(main_cq_id, int)) or cq_id == main_cq_id:
            raise ValueError(f"the commit queue needs two distinct command queue ids, got {cq_id} and {main_cq_id}")
        self.mesh_device, self.cq_id, self.main_cq_id = mesh_device, cq_id, main_cq_id
        self._history_event = None
        self._committed_event = None

    def history_derived(self) -> None:
        self._history_event = ttnn.record_event(self.mesh_device, cq_id=self.main_cq_id)

    def enqueue_commit(self, trace_id: int) -> None:
        if self._history_event is None:
            raise RuntimeError("the draft history must be derived (and its event recorded) before the commit")
        ttnn.wait_for_event(self.cq_id, self._history_event)
        ttnn._ttnn_execute_trace(self.mesh_device, trace_id, cq_id=self.cq_id, blocking=False)
        self._committed_event = ttnn.record_event(self.mesh_device, cq_id=self.cq_id)
        self._history_event = None

    def replay_commit(self, trace_id: int) -> None:
        """The blocking form (the segment-measuring probe): the commit on its queue, the host waiting."""

        ttnn._ttnn_execute_trace(self.mesh_device, trace_id, cq_id=self.cq_id, blocking=True)
        self._history_event = None

    def wait_committed(self) -> None:
        if self._committed_event is not None:
            ttnn.wait_for_event(self.main_cq_id, self._committed_event)
            self._committed_event = None


@dataclass(frozen=True)
class Qwen38TTNNMTPPassRecord:
    """One pass: its host-mirrored start position, the R verify tokens, the accept count, the a + 1 committed ids
    (cut at the first EOS), the 32 argmaxes, and the wall of every segment in ns (``commit_enqueue`` or
    ``commit_replay``, ``ple_rows`` = lookup + upload, ``verify_enqueue`` / ``draft_enqueue`` or ``verify_replay`` /
    ``draft_replay``, ``readback``; the bootstrap pass has ``host_inputs`` instead of the first two)."""

    index: int
    position: int
    tokens: tuple[int, ...]
    accepted: int
    committed: tuple[int, ...]
    argmaxes: tuple[int, ...]
    next_token: int
    first_draft: int
    finished: bool
    segments_ns: dict[str, int]


class Qwen38TTNNMTPChain:
    """The pass loop over the captured traces: ``replay(trace_id)`` is the caller's raw blocking replay (the timing
    runner's ``ttnn._ttnn_execute_trace`` with ``verify_before_replay`` called once outside the window),
    ``enqueue(trace_id)`` its non-blocking form.

    Host segment per pass, the pipelined form (``enqueue`` given): (1) the commit trace (split form) is enqueued
    non-blocking, so the device commits the previous pass while the host does the lookup; (2) the pass's R tokens
    (read with the previous pass) are looked up in the PLE n-gram table from the committed context and the
    ``[1,1,R,640]`` rows uploaded (queued behind the commit on the same command queue); (3) the verify body and,
    right behind it, the draft body of the NEXT pass are enqueued non-blocking; (4) one blocking 268-byte readback
    of the pass row the draft landed: the verify's ``[a, t', d_1', argmaxes]`` and the next pass's tokens
    ``[t', d_1 .. d_k]`` (checked to start at ``t'``, ``d_1'``); the PLE context commits, ``argmax_0 .. argmax_a``
    are emitted.  Without ``enqueue`` every trace is replayed blocking through ``replay`` (the segment-measuring
    form), the readback last.  The draft that runs before the host reads a pass writes only the draft state and the
    MTP layer's cache rows past the committed position (rewritten before any read); a pass loop that stops after the
    pass leaves it unused.  The token row and the draft lanes never cross the host.  ``observer(segment)`` is called
    before every segment (the runner's stall watchdog heartbeat).

    With ``commit_queue`` (:class:`Qwen38TTNNCommitQueue`; needs ``traces.commit`` and ``traces.draft_history``)
    the commit runs on its own command queue behind the draft-history fence and the verify waits for it
    (``commit_wait``); the blocking form replays it there.
    """

    def __init__(
        self,
        model: Qwen38TTNNTextModel,
        verify: Qwen38TTNNVerifyState,
        draft: Qwen38TTNNDraftState,
        traces: Qwen38TTNNMTPTraces,
        verify_output: Qwen38TTNNVerifyOutput,
        *,
        replay: Callable[[int], Any],
        position: int,
        eos_token_ids: Sequence[int] = (),
        clock_ns: Callable[[], int] = time.monotonic_ns,
        enqueue: Callable[[int], Any] | None = None,
        observer: Callable[[str], Any] | None = None,
        commit_queue: Qwen38TTNNCommitQueue | None = None,
    ) -> None:
        _validate_draft_state(model, verify, draft)
        if not isinstance(traces, Qwen38TTNNMTPTraces) or not callable(replay):
            raise TypeError("the chain needs the captured traces and a replay callable")
        if isinstance(position, bool) or type(position) is not int or position < 0:
            raise ValueError(f"the chain needs a non-negative int start position, got {position!r}")
        if (enqueue is not None and not callable(enqueue)) or (observer is not None and not callable(observer)):
            raise TypeError("enqueue and observer must be callables or None")
        if commit_queue is not None and (
            not isinstance(commit_queue, Qwen38TTNNCommitQueue) or traces.commit is None or traces.draft_history is None
        ):
            raise ValueError("a commit queue needs the split form with the draft history captured on its own")
        self.model, self.verify, self.draft, self.traces = model, verify, draft, traces
        self.verify_output, self.replay, self.clock_ns = verify_output, replay, clock_ns
        self.enqueue, self.observer, self.commit_queue = enqueue, observer, commit_queue
        self.position = position
        self.eos_token_ids = frozenset(int(token) for token in eos_token_ids)
        self.records: list[Qwen38TTNNMTPPassRecord] = []
        self.next_token: int | None = None
        self.first_draft: int | None = None
        self.next_tokens: tuple[int, ...] | None = None  # the next pass's verify tokens, assembled by the draft
        self.finished = False

    def _require_room(self) -> None:
        if self.finished:
            raise RuntimeError("the chain is finished (EOS)")
        if not verify_pass_fits(self.position, self.model.allocated_context):
            raise RuntimeError(
                f"position {self.position} has no room for a verify pass in {self.model.allocated_context} rows"
            )

    def _timed(self, segments: dict[str, int], name: str, action: Callable[[], Any]):
        if self.observer is not None:
            self.observer(name)
        started = self.clock_ns()
        result = action()
        segments[name] = self.clock_ns() - started
        return result

    def _finish_pass(self, tokens: Sequence[int], segments: dict[str, int]) -> Qwen38TTNNMTPPassRecord:
        verify_trace = (
            self.traces.verify_catch_up if self.records and self.traces.commit is None else self.traces.verify_first
        )
        launch, form = (self.replay, "replay") if self.enqueue is None else (self.enqueue, "enqueue")
        self._timed(segments, f"verify_{form}", lambda: launch(verify_trace))
        if self.traces.draft_history is not None:
            self._timed(segments, f"draft_history_{form}", lambda: launch(self.traces.draft_history))
            if self.commit_queue is not None:
                self.commit_queue.history_derived()  # the commit of this pass may start here, on its own queue
        self._timed(segments, f"draft_{form}", lambda: launch(self.traces.draft))
        readback, self.next_tokens = self._timed(segments, "readback", lambda: read_pass_row(self.verify, self.draft))
        if readback.first_draft is None:
            raise RuntimeError("the verify readback carries no first draft; the chain needs the alignment rows")
        commit_verify_host(self.verify, readback.accepted)
        committed = list(readback.argmaxes[: readback.accepted + 1])
        first_eos = next((i for i, token in enumerate(committed) if token in self.eos_token_ids), None)
        if first_eos is not None:
            committed = committed[: first_eos + 1]
            self.finished = True
        if committed[-1] != readback.next_token and first_eos is None:
            raise RuntimeError(f"verify readback next token {readback.next_token} is not argmax {readback.accepted}")
        record = Qwen38TTNNMTPPassRecord(
            index=len(self.records),
            position=self.position,
            tokens=tuple(tokens),
            accepted=readback.accepted,
            committed=tuple(committed),
            argmaxes=readback.argmaxes,
            next_token=readback.next_token,
            first_draft=readback.first_draft,
            finished=self.finished,
            segments_ns=segments,
        )
        self.records.append(record)
        self.position += readback.accepted + 1
        self.next_token, self.first_draft = readback.next_token, readback.first_draft
        return record

    def bootstrap(self, tokens: Sequence[int]) -> Qwen38TTNNMTPPassRecord:
        """The first traced pass after :func:`seed_verify_state_inplace`: host-written ``[t_P, d_1 .. d_k]`` (the
        caller's drafts, any exact ids), ``forward_verify(catch_up=False)``, then the draft of the next pass."""

        if self.records:
            raise RuntimeError("the chain was already bootstrapped")
        self._require_room()
        segments: dict[str, int] = {}
        self._timed(segments, "host_inputs", lambda: write_verify_inputs(self.model, self.verify, tokens))
        return self._finish_pass(tokens, segments)

    def step(self) -> Qwen38TTNNMTPPassRecord:
        """One traced pass on the tokens the previous pass read: [commit], PLE rows, verify, draft, readback."""

        if not self.records:
            raise RuntimeError("bootstrap the chain first")
        self._require_room()
        segments: dict[str, int] = {}
        tokens = self.next_tokens
        if self.traces.commit is not None and self.commit_queue is not None and self.enqueue is not None:
            self._timed(segments, "commit_enqueue", lambda: self.commit_queue.enqueue_commit(self.traces.commit))
        elif self.traces.commit is not None and self.commit_queue is not None:
            self._timed(segments, "commit_replay", lambda: self.commit_queue.replay_commit(self.traces.commit))
        elif self.traces.commit is not None and self.enqueue is not None:
            self._timed(segments, "commit_enqueue", lambda: self.enqueue(self.traces.commit))
        elif self.traces.commit is not None:
            self._timed(segments, "commit_replay", lambda: self.replay(self.traces.commit))
        self._timed(segments, "ple_rows", lambda: write_verify_ple_rows(self.model, self.verify, tokens))
        if self.commit_queue is not None:
            self._timed(segments, "commit_wait", self.commit_queue.wait_committed)  # the verify's fence
        return self._finish_pass(tokens, segments)

    def run(self, max_new_tokens: int, *, bootstrap_drafts: Sequence[int]) -> list[int]:
        """Bootstrap, then pass after pass until ``max_new_tokens`` ids are emitted, EOS, or the cache is full."""

        if isinstance(max_new_tokens, bool) or type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be a positive int, got {max_new_tokens!r}")
        emitted: list[int] = []
        record = self.bootstrap(bootstrap_drafts)
        emitted.extend(record.committed)
        while len(emitted) < max_new_tokens and not self.finished:
            if not verify_pass_fits(self.position, self.model.allocated_context):
                break
            emitted.extend(self.step().committed)
        return emitted[:max_new_tokens]


# --------------------------------------------------------------------------- the served chain (tools/qwen38_chat_session.py)
# A chat server that drafts with MTP runs the 1-row decode traces for prefill and the verify / draft / commit traces
# for generation.  The MTP layer must have consumed every position the target consumed before a pass can align
# and draft at P: the 1-row TAIL carries the MTP layer's row at P (the target's layer-47 residual at P and the token
# at P + 1: the next prompt token when the host knows it, else the step's own resolved argmax; the chain timing
# tool's prefill-step body split into the TAIL epilogue), and the prefill chunk carries the MTP layer's 32 rows
# (tokens P + 1 .. P + 32, host-written).  Two eager switches join the modes: enter_verify_mode seeds the verify
# buffers from the 1-row state at P; leave_verify_mode commits the last pass's rows and rebuilds the 1-row buffers
# from the verify buffers at the committed P.  A sampled step and the consumed EOS of a turn feed the MTP layer
# the step's argmax instead of the token the host later fed the target (the draft quality at those positions is
# approximate; the committed stream is the target's, exact).


@dataclass(frozen=True)
class Qwen38TTNNMTPStepInputs:
    """The MTP layer's host-written operands of one 1-row decode step (fixed addresses): ``host_lane`` FP32
    ROW_MAJOR ``[1,1,1,1]`` holds the token at P + 1 when the host knows it, ``select_index`` UINT32 ``[1,1,1,1]``
    picks it (0) or the step's resolved argmax (1) with a 32-bit gather (no id enters an FPU or reduce stage);
    ``sentinel_tail`` / ``sentinel_rest`` fill the 32-lane rows with ``ZERO_EMBEDDING_TOKEN``."""

    host_lane: Any
    select_index: Any
    sentinel_tail: Any
    sentinel_rest: Any
    mesh_device: Any = field(repr=False, compare=False)

    @classmethod
    def allocate(cls, mesh_device, mesh_contract: Qwen38MeshContract) -> "Qwen38TTNNMTPStepInputs":
        uploaded: list[Any] = []

        def upload(host: torch.Tensor, dtype, label: str):
            tensor = _upload_replicated(mesh_device, mesh_contract, host, dtype, ttnn.ROW_MAJOR_LAYOUT, label=label)
            uploaded.append(tensor)
            return tensor

        lanes = TOKEN_ROW_SHAPE[3]
        try:
            return cls(
                host_lane=upload(torch.zeros((1, 1, 1, 1)), ttnn.float32, "MTP step host lane"),
                select_index=upload(torch.ones((1, 1, 1, 1), dtype=torch.int32), ttnn.uint32, "MTP step select index"),
                sentinel_tail=upload(
                    torch.full((1, 1, 1, lanes - 2), float(ZERO_EMBEDDING_TOKEN)),
                    ttnn.float32,
                    "MTP step sentinel tail",
                ),
                sentinel_rest=upload(
                    torch.full((1, 1, 1, lanes - 1), float(ZERO_EMBEDDING_TOKEN)),
                    ttnn.float32,
                    "MTP step sentinel rest",
                ),
                mesh_device=mesh_device,
            )
        except BaseException:
            _deallocate(*uploaded)
            raise

    def write_next_token(self, next_token: int | None) -> None:
        """Host writes before a step's TAIL: the token at P + 1 (a prompt token) or ``None`` for the step's argmax."""

        if next_token is not None and (isinstance(next_token, bool) or type(next_token) is not int or next_token < 0):
            raise ValueError(f"MTP step next token must be a non-negative int or None, got {next_token!r}")
        replicate = replicate_tensor_2d_mesh_mapper(self.mesh_device)
        for target, host in (
            (self.host_lane, torch.full((1, 1, 1, 1), 0.0 if next_token is None else float(next_token))),
            (self.select_index, torch.full((1, 1, 1, 1), 1 if next_token is None else 0, dtype=torch.int32)),
        ):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(host, dtype=target.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=replicate), target
            )

    def deallocate(self) -> None:
        _deallocate(self.host_lane, self.select_index, self.sentinel_tail, self.sentinel_rest)


def forward_mtp_step_row(
    model: Qwen38TTNNTextModel,
    alignment: Qwen38TTNNVerifyAlignment,
    inputs: Qwen38TTNNMTPStepInputs,
    residual,
    resolved_row,
    *,
    rope,
    qsa_position,
) -> None:
    """The MTP layer's row at the position of one 1-row decode step (inside its TAIL, after the greedy resolve).

    ``residual`` is the target's layer-47 residual at P (the alignment root), ``resolved_row`` the step's FP32 TILE
    token row (column 0 = the argmax), ``rope`` / ``qsa_position`` the step's own position inputs; the MTP layer's
    generic state advances in place.  Nothing here is consumed: the caller releases the retained step tensors.
    """

    dram = ttnn.DRAM_MEMORY_CONFIG
    model.model_io.embedding.validate_token_row(resolved_row, label="MTP step resolved token row")
    resolved_lanes = ttnn.to_layout(resolved_row, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
    resolved_lane = _lane(resolved_lanes, 0)
    _deallocate(resolved_lanes)
    select_row = ttnn.concat([inputs.host_lane, resolved_lane, inputs.sentinel_tail], dim=3, memory_config=dram)
    _deallocate(resolved_lane)
    if _shape(select_row) != TOKEN_ROW_SHAPE or select_row.dtype != ttnn.float32:
        raise RuntimeError(f"MTP step select row must be FP32 {TOKEN_ROW_SHAPE}, got {tensor_metadata(select_row)}")
    selected = ttnn.gather(select_row, 3, inputs.select_index, memory_config=dram)
    _deallocate(select_row)
    token_row = _token_row_from_lane(model, selected, inputs.sentinel_rest)
    _deallocate(selected)
    embedding = model.model_io.embedding.embed_device_token(token_row)
    _deallocate(token_row)
    mixed = alignment.input_mixer(embedding, residual)
    _deallocate(embedding)
    out = alignment.layer.forward_decode_generic(
        mixed, alignment.generic_state, prepared_ple=None, rope=rope, qsa_position=qsa_position
    )
    _deallocate(out)


@dataclass(frozen=True)
class Qwen38TTNNMTPChunkExtension:
    """The MTP layer's 32 rows of a prefill chunk (the model's ``mtp`` keyword): its own chunk state over the
    alignment layer and ``token_row`` FP32 TILE ``[1,1,1,32]`` (lane j = the token at P + j + 1, host-written per
    chunk); the rows take the chunk's layer-47 residual rows as roots.  ``reset_chunk`` / ``finish_chunk`` are the per-layer
    chunk seed and hand-off of the MTP layer's generic state."""

    alignment: Qwen38TTNNVerifyAlignment
    layer_chunk_state: Any
    token_row: Any

    @classmethod
    def allocate(
        cls, model: Qwen38TTNNTextModel, verify: Qwen38TTNNVerifyState, chunk_state
    ) -> "Qwen38TTNNMTPChunkExtension":
        _validate_verify_state(model, verify)
        if verify.alignment is None:
            raise ValueError("the MTP chunk extension needs the verify state's MTP alignment components")
        layer_chunk_state = verify.alignment.layer.allocate_chunk_state(chunk_state.rows_constants)
        try:
            token_row = model.model_io.embedding.upload_token_row(0)
        except BaseException:
            verify.alignment.layer.release_chunk_state(layer_chunk_state)
            raise
        return cls(verify.alignment, layer_chunk_state, token_row)

    def release(self) -> None:
        _run_cleanup(
            "MTP chunk extension",
            [
                ("MTP chunk token row", lambda: _deallocate(self.token_row)),
                ("MTP layer chunk state", lambda: self.alignment.layer.release_chunk_state(self.layer_chunk_state)),
            ],
        )

    def reset_chunk(self) -> None:
        self.alignment.layer.reset_chunk_state_inplace(self.layer_chunk_state, self.alignment.generic_state)

    def write_tokens(self, model: Qwen38TTNNTextModel, token_ids: Sequence[int]) -> None:
        """Host write of the chunk's 32 MTP tokens (the tokens at P + 1 .. P + 32) into the token row."""

        token_ids = [int(token) for token in token_ids]
        if len(token_ids) != CHUNK_ROWS:
            raise ValueError(f"an MTP chunk takes {CHUNK_ROWS} tokens, got {len(token_ids)}")
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                model.model_io.embedding.host_token_rows(token_ids),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=replicate_tensor_2d_mesh_mapper(model.mesh_device),
            ),
            self.token_row,
        )

    def forward_chunk_rows(
        self, model: Qwen38TTNNTextModel, residual_rows, *, rope_rows, qsa_chunk, qsa_chunk_constants, selectors
    ):
        """The MTP layer's rows at the chunk's positions: the token rows' embedding mixed with the chunk's layer-47
        residual rows (not consumed), through the layer's chunk body on the MTP generic and chunk states."""

        if _shape(residual_rows) != RESIDUAL_ROWS_SHAPE:
            raise RuntimeError(f"MTP chunk roots must be {RESIDUAL_ROWS_SHAPE}, got {tensor_metadata(residual_rows)}")
        embedding_rows = model.model_io.embedding.embed_device_token_rows(self.token_row)
        if _shape(embedding_rows) != BLOCK_ROWS_SHAPE:
            raise RuntimeError(f"MTP chunk embedding rows must be {BLOCK_ROWS_SHAPE}, got {_shape(embedding_rows)}")
        mixed = self.alignment.input_mixer.rows(embedding_rows, residual_rows)
        _deallocate(embedding_rows)
        out = self.alignment.layer.forward_chunk_generic(
            mixed,
            self.alignment.generic_state,
            self.layer_chunk_state,
            prepared_ple_rows=None,
            rope_rows=rope_rows,
            qsa_chunk=qsa_chunk,
            qsa_chunk_constants=qsa_chunk_constants,
            selectors=selectors,
        )
        _deallocate(out)

    def finish_chunk(self, model: Qwen38TTNNTextModel, *, prefilled: int) -> None:
        """The MTP layer's hand-off after the last chunk (the model's ``finish_prefill`` form for one layer)."""

        ring_select = ttnn.from_torch(
            qsa_module.chunk_handoff_ring_select_rows(prefilled),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=model.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(model.mesh_device),
        )
        try:
            self.alignment.layer.finish_chunk_state_inplace(
                self.layer_chunk_state, self.alignment.generic_state, prefilled=prefilled, qsa_ring_select=ring_select
            )
        finally:
            _deallocate(ring_select)


def enter_verify_mode(
    model: Qwen38TTNNTextModel,
    state: Qwen38TTNNTextModelGenericState,
    verify: Qwen38TTNNVerifyState,
    *,
    position: int,
    ple_context: tuple[int, int] | None,
) -> None:
    """The served chain's eager switch 1-row -> verify at host-known ``P`` (after traced replays): the GDN ring
    phase and the PLE n-gram context onto the host bookkeeping the seed reads, then :func:`seed_verify_state_inplace`.
    The next pass runs with ``catch_up=False``."""

    _validate_verify_state(model, verify)
    for layer, layer_state in zip(model.layers, state.layers):
        if isinstance(layer.attention, Qwen38TTNNGDN):
            layer_state.attention.conv_phase = position % gdn_module.CONV_KERNEL_SIZE
    ple_state = state.layers[PLE_CHECKPOINT_LAYER].ple
    ple_state.token_context = None if ple_context is None else torch.tensor([list(ple_context)], dtype=torch.long)
    seed_verify_state_inplace(model, state, verify, position=position)
    expected = None if ple_context is None else tuple(int(value) for value in ple_context)
    actual = verify.layers[PLE_CHECKPOINT_LAYER].ple.token_context
    if actual != expected:
        raise RuntimeError(f"verify PLE context after the seed is {actual}, expected {expected}")


def leave_verify_mode(
    model: Qwen38TTNNTextModel,
    state: Qwen38TTNNTextModelGenericState,
    verify: Qwen38TTNNVerifyState,
    *,
    position: int,
    committed_rows: int,
    commit: Callable[[], Any],
) -> tuple[int, int] | None:
    """The served chain's eager switch verify -> 1-row after the last pass: commit ``committed_rows`` of that pass's
    rows (``commit`` runs the commit body after the host wrote the accept scalar; fewer rows than the pass accepted
    roll the state back to them), ``P <- position`` (the host's count of committed positions), then every layer's
    1-row buffers from the verify buffers: the GDN ring slots (phase P mod 4) from the rows history, the PLE slots
    from the rows history with the n-gram context of the committed rows, the QSA staging tile and raw-key ring
    from the cache block and the raw history (the backbone's QSA layers and the MTP alignment layer alike).
    Returns the n-gram context of the committed stream; synchronizes the device."""

    _validate_verify_state(model, verify)
    if isinstance(committed_rows, bool) or type(committed_rows) is not int or not 1 <= committed_rows <= verify.rows:
        raise ValueError(f"committed rows must be an int in [1,{verify.rows}], got {committed_rows!r}")
    if isinstance(position, bool) or type(position) is not int or position < committed_rows:
        raise ValueError(f"position must be an int >= {committed_rows}, got {position!r}")
    contexts = verify.ple_rows.contexts
    if len(contexts) != verify.rows + 1:
        raise RuntimeError(f"verify PLE rows carry {len(contexts)} contexts, expected {verify.rows + 1}")
    write_verify_accepted(model, verify, committed_rows - 1)
    commit()
    state.position.reset(position)
    ple_state = verify.layers[PLE_CHECKPOINT_LAYER].ple
    ple_state.token_context = contexts[committed_rows]
    for layer, layer_state, layer_verify in zip(model.layers, state.layers, verify.layers):
        _store_layer(layer, layer_state, layer_verify, position=position)
    if verify.alignment is not None:
        alignment = verify.alignment
        _store_layer(alignment.layer, alignment.generic_state, alignment.verify_state, position=position)
    ttnn.synchronize_device(model.mesh_device)
    actual = state.position.read()
    if actual != position:
        raise RuntimeError(f"device position after the verify hand-off is {actual}, expected {position}")
    return ple_state.token_context


def _store_layer(layer, generic_state, layer_verify: Qwen38TTNNVerifyLayerState, *, position: int) -> None:
    if isinstance(layer.attention, Qwen38TTNNGDN):
        # The next token lands in slot P % 4; the history rows are the three before it, oldest first.
        generic_state.attention.conv_phase = position % gdn_module.CONV_KERNEL_SIZE
        layer.attention.sync_state_from_rows_history(layer_verify.attention, generic_state.attention)
    else:
        layer.attention.handoff_verify_state(generic_state.attention, layer_verify.attention, position=position)
    if layer_verify.ple is not None:
        layer_verify.ple.store_to_state(generic_state.ple)
        generic_state.ple.token_context = None  # the generic body's caller owns the n-gram context


__all__ = [
    "accept_rows",
    "allocate_draft_state",
    "allocate_verify_state",
    "capture_commit",
    "capture_draft",
    "capture_draft_history",
    "capture_verify",
    "commit_verify_host",
    "DEFAULT_DRAFTS",
    "enter_verify_mode",
    "forward_commit",
    "forward_draft",
    "forward_draft_history",
    "forward_mtp_step_row",
    "forward_verify",
    "leave_verify_mode",
    "moe_rows_for",
    "Qwen38TTNNAcceptConstants",
    "Qwen38TTNNAcceptResult",
    "Qwen38TTNNCommitQueue",
    "Qwen38TTNNDraftState",
    "Qwen38TTNNMTPChain",
    "Qwen38TTNNMTPChunkExtension",
    "Qwen38TTNNMTPPassRecord",
    "Qwen38TTNNMTPStepInputs",
    "Qwen38TTNNMTPTraces",
    "Qwen38TTNNVerifyAlignment",
    "Qwen38TTNNVerifyLayerState",
    "Qwen38TTNNVerifyOutput",
    "Qwen38TTNNVerifyReadback",
    "Qwen38TTNNVerifyState",
    "PASS_ROW_WIDTH",
    "read_pass_row",
    "read_verify_output",
    "READBACK_WIDTH",
    "release_draft_state",
    "release_verify_state",
    "resolve_moe_rows",
    "seed_verify_state_inplace",
    "select_residual_row",
    "SUPPORTED_DRAFTS",
    "verify_pass_fits",
    "write_verify_accepted",
    "write_verify_inputs",
    "write_verify_ple_rows",
]
