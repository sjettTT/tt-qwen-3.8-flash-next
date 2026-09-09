# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The prefill slab (``--prefill-slab ROWS``) without a device: the row-count contract, the QSA slab constants and
chunk-input emulation, the driver's plan (slabs, then 128-row chunks, then 32-row chunks and the tail), the 2D-multicast
matmul config, and the source pins that keep the slab's forms apart from the 32/128-row bodies (whose own pins hold).
"""

from __future__ import annotations

import inspect
import itertools
import math
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_prefill_driver as driver_module
from models.demos.blackhole.qwen38_flash_next.ttnn import contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as decode_matmul_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gr as gr_module
from models.demos.blackhole.qwen38_flash_next.ttnn import moe as moe_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    CHUNK_ROW_COUNTS,
    CHUNK_ROWS,
    DEFAULT_SLAB_ROWS,
    LONG_CHUNK_ROWS,
    MAX_SLAB_ROWS,
    MIN_SLAB_ROWS,
    is_slab_rows,
)

SLAB = DEFAULT_SLAB_ROWS
BLOCKS = 8192  # the 32k context's compressed blocks


# --------------------------------------------------------------------------- contracts


def test_slab_row_contract() -> None:
    assert (MIN_SLAB_ROWS, MAX_SLAB_ROWS, DEFAULT_SLAB_ROWS) == (256, 4096, 2048)
    assert CHUNK_ROW_COUNTS == (32, 128)  # the chunk forms are untouched
    for rows in (256, 512, 1024, 2048, 4096):
        assert is_slab_rows(rows) and contracts_module.chunk_row_tiles(rows) == rows // 32
    for rows in (32, 128, 0, 31, 64, 96, 192, 2000, 8192, True, 2048.0, "2048"):
        assert not is_slab_rows(rows)
    for rows in (0, 31, 64, 96, 192, 8192):
        with pytest.raises(ValueError):  # allow-pytest.raises: pure contract test
            contracts_module.chunk_row_tiles(rows)
    assert gdn_module.rows_tile_count(SLAB) == SLAB and gdn_module.rows_tile_count(128) == 128
    assert gr_module.residual_rows_shape(SLAB) == (1, 4, SLAB, 640)
    assert moe_module.routed_tokens_per_call_for(SLAB) == 128 == moe_module.routed_tokens_per_call_for(128)
    assert moe_module.routed_tokens_per_call_for(32) == 32 and moe_module.routed_tokens_per_call_for(5) == 5
    contract = moe_module.Qwen38TTNNMoERowContract(SLAB, moe_module.SUPPORTED_ROWS + (SLAB,))
    assert contract.row_tiles == SLAB // 32 and contract.local_combine == (10, SLAB, 2560)
    assert moe_module.SUPPORTED_ROWS == (1, 5, 32, 128)  # the slab is admitted per instance, not globally
    with pytest.raises(ValueError):  # allow-pytest.raises: pure contract test
        moe_module.Qwen38TTNNMoERowContract(SLAB)


# --------------------------------------------------------------------------- the 2D-multicast matmul config


def test_prefill_matmul_config_covers_the_rows_and_columns_with_whole_subblocks() -> None:
    captured = {}

    class _Config:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    grid = SimpleNamespace(x=11, y=10)
    mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: grid)
    original = decode_matmul_module.ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
    decode_matmul_module.ttnn.MatmulMultiCoreReuseMultiCastProgramConfig = _Config
    try:
        for k, n in (
            (2560, 4160),
            (1536, 2560),
            (2560, 384),
            (384, 2560),
            (2560, 512),
            (2560, 160),
            (160, 2560),
            (2560, 1),
        ):
            captured.clear()
            decode_matmul_module.prefill_matmul_program_config(mesh, SLAB, k, n)
            cols, rows = captured["compute_with_storage_grid_size"]
            n_tiles, k_tiles = math.ceil(n / 32), k // 32
            assert 1 <= cols <= min(11, n_tiles) and rows == 10
            assert captured["per_core_M"] * rows >= SLAB // 32 and captured["per_core_N"] * cols >= n_tiles
            assert captured["out_subblock_h"] == 1 and captured["per_core_N"] % captured["out_subblock_w"] == 0
            assert 1 <= captured["out_subblock_w"] <= 4 and k_tiles % captured["in0_block_w"] == 0
            assert captured["in0_block_w"] <= 8 and captured["transpose_mcast"] is False
        with pytest.raises(ValueError):  # allow-pytest.raises: pure contract test
            decode_matmul_module.prefill_matmul_program_config(mesh, 2000, 2560, 512)
    finally:
        decode_matmul_module.ttnn.MatmulMultiCoreReuseMultiCastProgramConfig = original


# --------------------------------------------------------------------------- QSA slab constants and inputs


def test_qsa_slab_constants_pool_every_block_and_carry_no_row_templates() -> None:
    host = qsa_module.qsa_chunk_constant_rows(BLOCKS, SLAB)
    blocks = SLAB // 4
    tiles = -(-blocks // 32)
    assert "row_index_blocks" not in host and "arange_blocks_rows" not in host
    assert tuple(host["row_selects"].shape) == (0, 1, 1, 32, 32)
    assert tuple(host["block_start_lanes"].shape) == (1, 1, tiles, 32)
    assert host["block_start_lanes"].reshape(-1)[:blocks].tolist() == [4 * i for i in range(blocks)]
    pool = host["pool_select"]
    assert tuple(pool.shape) == (1, 1, tiles * 32, SLAB)
    expected = torch.zeros(tiles * 32, SLAB)
    for block in range(blocks):
        expected[block, 4 * block : 4 * block + 4] = 0.25
    assert torch.equal(pool[0, 0], expected)
    assert host["page_offsets"].reshape(-1).tolist() == list(range(tiles))
    assert tuple(host["arange_blocks_row"].shape) == (1, 1, 1, BLOCKS)
    assert host["row_index_col"].reshape(-1).tolist() == list(range(SLAB))
    assert tuple(host["row_keep_bits" if "row_keep_bits" in host else "row_index_slots"].shape) == (1, 1, SLAB, 2080)
    # The chunk forms are as before.
    short = qsa_module.qsa_chunk_constant_rows(BLOCKS, CHUNK_ROWS)
    assert tuple(short["row_index_blocks"].shape) == (1, 1, 32, BLOCKS) and len(short["row_selects"]) == 8
    assert tuple(short["block_start_lanes"].shape) == (1, 1, 1, 32)


@pytest.mark.parametrize("position", [0, 2048, 30720])
def test_qsa_slab_emulation_is_the_one_row_rule_per_row(position: int) -> None:
    inputs = qsa_module.emulate_qsa_chunk_inputs(position, allocated_compressed_blocks=BLOCKS, rows=SLAB)
    assert inputs["indexer_neg_mask"] is None and inputs["block_index_i32"] == ()
    tiles = SLAB // 128
    assert inputs["compressed_tile_i32"].tolist() == [[position // 128 + t for t in range(tiles)]]
    complete = inputs["complete_blocks_col"].reshape(-1)
    for row in (0, 1, 3, 4, 127, SLAB - 1):
        one = qsa_module.emulate_qsa_position_inputs(position + row, allocated_compressed_blocks=BLOCKS)
        visible = int((one["indexer_neg_mask"].reshape(-1) == 0).sum())
        assert int(complete[row]) == (position + row + 1) // 4 == visible
        assert torch.equal(inputs["row_keep_bits"][0, 0, row], one["row_keep_bits"].reshape(-1))
        assert torch.equal(inputs["row_fill"][0, 0, row], one["row_fill"].reshape(-1))


# --------------------------------------------------------------------------- the driver's plan


class _FakeModel:
    allocated_context = 70000

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def reset_chunk_state_inplace(self, state, chunk_state) -> None:
        self.calls.append(("reset", chunk_state.rows))

    def write_chunk_accepted(self, chunk_state, accepted: int) -> None:
        self.calls.append(("accepted", accepted))

    def prepare_chunk_inputs(self, chunk_state, token_ids, *, ple_context):
        tokens = list(token_ids)
        assert len(tokens) == chunk_state.rows
        self.calls.append(("write", chunk_state.rows, tuple(tokens)))
        return SimpleNamespace(rows=chunk_state.rows, contexts=tuple([ple_context] * (len(tokens) + 1)))

    def upload_chunk_inputs(self, chunk_state, prepared) -> None:
        assert prepared.rows == chunk_state.rows
        self.calls.append(("upload", chunk_state.rows))

    def finish_prefill(self, state, chunk_state, prefilled: int) -> None:
        self.calls.append(("finish", prefilled))

    def forward_prefill_chunk_generic(self, chunk_state, state, *, gdn_step_anchor: bool = False, mtp=None) -> None:
        self.calls.append(("eager", chunk_state.rows, gdn_step_anchor, mtp))


@pytest.fixture
def driver(monkeypatch):
    fake = SimpleNamespace(
        synchronize_device=lambda mesh: None,
        record_event=lambda mesh, cq_id: object(),
        event_synchronize=lambda event: None,
        _ttnn_execute_trace=lambda mesh, trace_id, cq_id, blocking: None,
    )
    monkeypatch.setattr(driver_module, "ttnn", fake)
    return fake


@pytest.mark.parametrize("start", [0, 5, 32])
@pytest.mark.parametrize("count", [100, 2048, 2048 + 128 + 40, 2 * 2048 + 3 * 128 + 32 + 7, 5000])
def test_driver_runs_slabs_then_long_then_short_chunks(driver, start: int, count: int) -> None:
    model = _FakeModel()
    states = {rows: SimpleNamespace(rows=rows) for rows in (32, 128, SLAB)}
    tokens = list(range(1000, 1000 + count))
    prefill = driver_module.Qwen38ChunkPrefill(
        model,
        object(),
        object(),
        states[32],
        None,
        forced_step=lambda token, context: context,
        long_chunk_state=states[128],
        slab_state=states[SLAB],
    )
    result = prefill.run(tokens, start_position=start, ple_context=None)
    aligned = driver_module.alignment_steps(start, count)
    remaining = count - aligned
    slabs = remaining // SLAB
    long = (remaining - slabs * SLAB) // 128
    short = remaining - slabs * SLAB - long * 128
    writes = [call for call in model.calls if call[0] == "write"]
    assert [rows for _, rows, _ in writes] == [SLAB] * slabs + [128] * long + [32] * math.ceil(short / 32)
    written = [token for _, _, chunk in writes for token in chunk]
    assert written[:remaining] == tokens[aligned:]
    assert result.position == start + count
    assert result.timing.slabs == slabs and result.timing.slab_rows == SLAB
    uploads = [call for call in model.calls if call[0] == "upload"]
    assert [rows for _, rows in uploads] == [rows for _, rows, _ in writes]
    timing = result.timing
    assert len(timing.slab_prepare_ms) == len(timing.slab_upload_ms) == len(timing.slab_wait_ms) == slabs
    assert all(ms >= 0.0 for ms in timing.slab_prepare_ms + timing.slab_upload_ms + timing.slab_wait_ms)
    assert result.timing.long_chunks == long and result.timing.chunks == math.ceil(short / 32)
    eager = [call for call in model.calls if call[0] == "eager"]
    assert [rows for _, rows, _, _ in eager] == [SLAB] * slabs + [128] * long + [32] * math.ceil(short / 32)
    assert all(not anchor and mtp is None for _, _, anchor, mtp in eager)
    resets = [call for call in model.calls if call[0] == "reset"]
    if remaining:
        assert resets[0] == ("reset", 32) and (slabs == 0 or ("reset", SLAB) in resets)
        assert model.calls[-1] == ("finish", start + count)


def test_driver_prepares_the_next_slab_while_the_current_one_replays(monkeypatch) -> None:
    """The slab cadence: with slab k's replay queued the host waits for slab k - 1's event and records slab k's, so
    slab k + 1's inputs are prepared and their copies queued while slab k runs; the first slab waits for nothing;
    the chunks after the slabs keep the every-``event_interval`` sync; the hand-off synchronizes everything."""

    log: list[tuple] = []
    events = itertools.count()

    def record_event(mesh, cq_id):
        event = next(events)
        log.append(("record", event))
        return event

    monkeypatch.setattr(
        driver_module,
        "ttnn",
        SimpleNamespace(
            synchronize_device=lambda mesh: log.append(("synchronize",)),
            record_event=record_event,
            event_synchronize=lambda event: log.append(("wait", event)),
            _ttnn_execute_trace=lambda mesh, trace_id, cq_id, blocking: log.append(("replay", trace_id, blocking)),
        ),
    )
    model = _FakeModel()
    model.calls = log
    states = {rows: SimpleNamespace(rows=rows) for rows in (32, 128, SLAB)}
    traces = {"slab": 3, "long": 2, "short": 1}
    count = 3 * SLAB + 5 * 128 + 32 + 8
    tokens = list(range(1000, 1000 + count))
    prefill = driver_module.Qwen38ChunkPrefill(
        model,
        object(),
        object(),
        states[32],
        traces["short"],
        forced_step=lambda token, context: context,
        verify_allocations=False,
        long_chunk_state=states[128],
        long_chunk_trace_id=traces["long"],
        slab_state=states[SLAB],
        slab_trace_id=traces["slab"],
    )
    result = prefill.run(tokens, start_position=0, ple_context=None)

    plan = [("slab", SLAB)] * 3 + [("long", 128)] * 5 + [("short", 32)] * 2
    expected: list[tuple] = [("synchronize",), ("reset", 32), ("reset", 128), ("reset", SLAB)]
    offset = 0
    pending = None
    event = 0
    for index, (kind, rows) in enumerate(plan):
        chunk = tokens[offset : offset + rows]
        offset += rows
        if len(chunk) < rows:
            expected.append(("accepted", len(chunk) - 1))
            chunk = chunk + [driver_module.CHUNK_PAD_TOKEN_ID] * (rows - len(chunk))
        expected += [("write", rows, tuple(chunk)), ("upload", rows), ("replay", traces[kind], False)]
        if kind == "slab":
            if pending is not None:
                expected.append(("wait", pending))
            expected.append(("record", event))
            pending, event = event, event + 1
        elif (index + 1) % driver_module.CHUNK_EVENT_INTERVAL == 0:
            expected += [("record", event), ("wait", event)]
            event += 1
    expected += [("synchronize",), ("finish", count), ("synchronize",)]
    assert log == expected
    assert result.position == count and result.timing.slabs == 3 and len(result.timing.slab_wait_ms) == 3
    # The blocking (timed) form keeps every replay blocking and records no event.
    log.clear()
    timed = prefill.run(tokens[: 2 * SLAB], start_position=0, ple_context=None, time_each_chunk=True)
    assert [entry for entry in log if entry[0] in ("replay", "record", "wait")] == [("replay", 3, True)] * 2
    assert len(timed.timing.slab_replay_ms) == 2 and timed.timing.slab_wait_ms == ()


def test_driver_rejects_a_slab_without_the_long_chunks_or_with_mtp(driver) -> None:
    states = {rows: SimpleNamespace(rows=rows) for rows in (32, 128, SLAB)}
    with pytest.raises(ValueError):  # allow-pytest.raises: pure contract test
        driver_module.Qwen38ChunkPrefill(
            _FakeModel(), object(), object(), states[32], None, forced_step=lambda t, c: c, slab_state=states[SLAB]
        )
    with pytest.raises(ValueError):  # allow-pytest.raises: pure contract test
        driver_module.Qwen38ChunkPrefill(
            _FakeModel(),
            object(),
            object(),
            states[32],
            None,
            forced_step=lambda t, c: c,
            long_chunk_state=states[128],
            slab_state=SimpleNamespace(rows=200),
        )
    with pytest.raises(ValueError):  # allow-pytest.raises: pure contract test
        driver_module.Qwen38ChunkPrefill(
            _FakeModel(),
            object(),
            object(),
            states[32],
            None,
            forced_step=lambda t, c: c,
            long_chunk_state=states[128],
            slab_state=states[SLAB],
            mtp=object(),
        )
    assert driver_module.slab_count(5000, slab_rows=SLAB) == 2 and driver_module.slab_count(5000, slab_rows=None) == 0


# --------------------------------------------------------------------------- source pins


def test_slab_source_pins() -> None:
    linear = inspect.getsource(decode_matmul_module.prefill_linear)
    # The 2D-multicast program misreads a DRAM-width-sharded in1: the helper always copies the weight interleaved
    # first and never hands the resident weight to the matmul.
    assert linear.index("ttnn.to_memory_config(weight, ttnn.DRAM_MEMORY_CONFIG)") < linear.index("ttnn.linear(")
    assert "weight_interleaved," in linear and "memory_config=ttnn.DRAM_MEMORY_CONFIG" in linear
    for module in (gdn_module, gr_module, moe_module, qsa_module):
        source = inspect.getsource(module)
        assert "prefill_linear(" in source and "prefill_matmul_program_config(" in source
        # Every slab linear goes through the helper: no direct ttnn.linear on a slab branch.
        assert "is_slab_rows" in source
    gdn = inspect.getsource(gdn_module.Qwen38TTNNGDN._causal_conv_rows)
    assert "if is_slab_rows(rows_state.constants.tile_rows):" in gdn and "_shifted_rows_slab" in gdn
    shifted = inspect.getsource(gdn_module.Qwen38TTNNGDN._shifted_rows_slab)
    assert "ttnn.ROW_MAJOR_LAYOUT" in shifted and "ttnn.concat([kept, new], dim=2" in shifted
    fold = inspect.getsource(gdn_module.Qwen38TTNNGDN._gate_and_project_rows)
    assert "elif is_slab_rows(tile_rows):" in fold and fold.index("if tile_rows == CHUNK_SIZE:") < fold.index(
        "elif is_slab_rows(tile_rows):"
    )
    moe_route = inspect.getsource(moe_module.Qwen38TTNNMoE._route)
    assert "self._slab_router_logits(full_hidden) if self.slab else logits_tiles[0]" in moe_route
    router = inspect.getsource(moe_module.Qwen38TTNNMoE._slab_router_logits)
    assert "prefill_linear(" in router and "ttnn.typecast(router_bf16, ttnn.float32" in router
    blocks = inspect.getsource(moe_module.Qwen38TTNNMoE._routed_partial_blocks)
    assert (
        "self.block_instance._routed_partial(rows, Qwen38TTNNRouting(scores, indices), packed_w0_w1, packed_w2)"
        in blocks
    )
    slab_select = inspect.getsource(qsa_module.Qwen38TTNNQSA._sparse_indices_slab)
    assert "ttnn.all_reduce(" in slab_select and "ttnn.ge(constants.arange_blocks_row, complete_col" in slab_select
    assert "ttnn.experimental.topk_large_indices(masked, k=BLOCK_TOPK)" in slab_select
    assert qsa_module.SLAB_SCORE_BLOCK_ROWS == 512
    forward = inspect.getsource(qsa_module.Qwen38TTNNQSA.forward_chunk_generic)
    assert (
        "if is_slab_rows(rows):" in forward
        and "self._sparse_indices_slab(index_query, state, chunk, constants)" in forward
    )
    # The 32-row and 128-row bodies keep their forms (their own pins hold): the per-tile loops are still there.
    assert "dram_sharded_row_tiles(full_hidden, self.in_proj_act_memory_config)" in inspect.getsource(gdn_module)
    assert "def _routed_partial_tiles" in inspect.getsource(moe_module)
