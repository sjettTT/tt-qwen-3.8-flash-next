# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The chunked prefill driver's sequence without a device: alignment steps, the seed, full chunks, the padded tail,
the event cadence, the hand-off and the n-gram context threading, recorded through fakes of the model and ttnn."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_prefill_driver as driver_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_prefill_driver import (
    CHUNK_PAD_TOKEN_ID,
    Qwen38ChunkPrefill,
    alignment_steps,
    chunk_accepts,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROWS

ALLOCATED_CONTEXT = 500  # not a tile multiple, so the padded tail's end can overrun it in the budget test
TRACE_ID = 77


def _next_context(context, token: int) -> tuple[int, int]:
    """The n-gram context after one token: ``(c1, token)`` (the resident lookup's rule)."""

    previous = 2 if context is None else context[1]
    return (previous, token)


class _FakeModel:
    """Records the driver's calls; ``prepare_chunk_inputs`` returns the 33 contexts the PLE lookup would."""

    allocated_context = ALLOCATED_CONTEXT

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def reset_chunk_state_inplace(self, state, chunk_state) -> None:
        self.calls.append(("reset_chunk_state_inplace", state, chunk_state))

    def write_chunk_accepted(self, chunk_state, accepted: int) -> None:
        self.calls.append(("write_chunk_accepted", accepted))

    def prepare_chunk_inputs(self, chunk_state, token_ids, *, ple_context):
        tokens = list(token_ids)
        assert len(tokens) == CHUNK_ROWS
        contexts = [ple_context]
        for token in tokens:
            contexts.append(_next_context(contexts[-1], token))
        self.calls.append(("prepare_chunk_inputs", tuple(tokens), ple_context))
        return SimpleNamespace(tokens=tuple(tokens), contexts=tuple(contexts))

    def upload_chunk_inputs(self, chunk_state, prepared) -> None:
        self.calls.append(("upload_chunk_inputs", prepared.tokens))

    def finish_prefill(self, state, chunk_state, prefilled: int) -> None:
        self.calls.append(("finish_prefill", prefilled))

    def forward_prefill_chunk_generic(self, chunk_state, state, *, gdn_step_anchor: bool = False, mtp=None) -> None:
        self.calls.append(("eager_chunk", gdn_step_anchor))


@pytest.fixture
def harness(monkeypatch):
    model = _FakeModel()
    log = model.calls

    def execute(mesh, trace_id, *, cq_id, blocking):
        assert mesh == "mesh" and trace_id == TRACE_ID and cq_id == 0
        log.append(("replay", blocking))

    fake_ttnn = SimpleNamespace(
        _ttnn_execute_trace=execute,
        record_event=lambda mesh, cq_id: log.append(("record_event",)) or "event",
        event_synchronize=lambda event: log.append(("event_synchronize", event)),
        synchronize_device=lambda mesh: log.append(("synchronize_device",)),
    )
    monkeypatch.setattr(driver_module, "ttnn", fake_ttnn)

    class FakeTracker:
        def __init__(self, mesh) -> None:
            assert mesh == "mesh"

        def verify_before_replay(self, trace_id) -> None:
            log.append(("verify_before_replay", trace_id))

    monkeypatch.setattr(driver_module, "UnsafeAllocationTracker", FakeTracker)

    def forced_step(token: int, context):
        log.append(("forced_step", token, context))
        return _next_context(context, token)

    prefill = Qwen38ChunkPrefill(model, "mesh", "state", "chunk_state", TRACE_ID, forced_step=forced_step)
    return SimpleNamespace(model=model, log=log, prefill=prefill)


def _expected_context(tokens, context):
    for token in tokens:
        context = _next_context(context, token)
    return context


@pytest.mark.parametrize("start", (0, 5, 32, 60, 95))
@pytest.mark.parametrize("count", (0, 1, 3, 31, 32, 33, 64, 96, 100, 137, 160))
def test_run_sequences_alignment_chunks_tail_and_handoff(harness, start: int, count: int) -> None:
    tokens = [1000 + index for index in range(count)]
    aligned = alignment_steps(start, count)
    assert aligned == min(count, (32 - start % 32) % 32) and (aligned == count or (start + aligned) % 32 == 0)
    remaining = tokens[aligned:]
    accepts = chunk_accepts(len(remaining))
    assert accepts == [31] * (len(remaining) // 32) + ([len(remaining) % 32 - 1] if len(remaining) % 32 else [])

    result = harness.prefill.run(tokens, start_position=start, ple_context=None)

    assert result.position == start + count
    assert result.ple_context == _expected_context(tokens, None)
    timing = result.timing
    assert (timing.alignment_steps, timing.chunks, timing.tail_rows) == (aligned, len(accepts), len(remaining) % 32)
    assert timing.chunk_replay_ms == () and timing.chunk_host_ms == () and timing.wall_ms >= 0.0
    assert timing.slab_prepare_ms == () and timing.slab_upload_ms == () and timing.slab_wait_ms == ()  # no slabs

    log = harness.log
    forced = [entry for entry in log if entry[0] == "forced_step"]
    assert [entry[1] for entry in forced] == tokens[:aligned]
    assert forced == [
        ("forced_step", token, _expected_context(tokens[:i], None)) for i, token in enumerate(tokens[:aligned])
    ]
    if not accepts:
        assert log[len(forced) :] == [] and timing.verify_ms is None and timing.handoff_ms == 0.0
        return

    expected = [
        ("synchronize_device",),
        ("reset_chunk_state_inplace", "state", "chunk_state"),
        ("verify_before_replay", TRACE_ID),
    ]
    context = _expected_context(tokens[:aligned], None)
    for index, accepted in enumerate(accepts):
        rows = remaining[32 * index : 32 * (index + 1)]
        real = len(rows)
        if accepted != 31:
            expected.append(("write_chunk_accepted", accepted))
            rows = rows + [CHUNK_PAD_TOKEN_ID] * (32 - real)
        expected += [("prepare_chunk_inputs", tuple(rows), context), ("upload_chunk_inputs", tuple(rows))]
        context = _expected_context(rows[:real], context)  # the pad rows never enter the committed context
        expected.append(("replay", False))
        if (index + 1) % 4 == 0:
            expected += [("record_event",), ("event_synchronize", "event")]
    expected += [("synchronize_device",), ("finish_prefill", start + count), ("synchronize_device",)]
    assert log[len(forced) :] == expected
    assert context == result.ple_context
    assert timing.verify_ms is not None and timing.handoff_ms >= 0.0
    assert timing.traced is True and timing.gdn_step_anchor is False
    # The accept scalar is written only before the padded tail; finish_prefill restores the full-chunk value.
    assert [entry for entry in log if entry[0] == "write_chunk_accepted"] == (
        [("write_chunk_accepted", len(remaining) % 32 - 1)] if len(remaining) % 32 else []
    )


@pytest.mark.parametrize("anchor", (False, True))
def test_eager_chunks_pass_the_gdn_step_anchor_per_chunk_and_skip_the_trace_verification(harness, anchor: bool) -> None:
    """Without a captured trace the driver runs the chunk body itself: one ``forward_prefill_chunk_generic`` per chunk
    with the re-anchor flag, no ``verify_before_replay``, the blocking form synchronizing after every chunk."""

    model = harness.model
    forced_step = harness.prefill.forced_step
    prefill = Qwen38ChunkPrefill(
        model, "mesh", "state", "chunk_state", None, forced_step=forced_step, gdn_step_anchor=anchor
    )
    result = prefill.run(list(range(1, 101)), start_position=32, ple_context=None)
    assert result.position == 132 and result.timing.chunks == 4 and result.timing.tail_rows == 4
    assert result.timing.traced is False and result.timing.gdn_step_anchor is anchor and result.timing.verify_ms is None
    chunks = [entry for entry in harness.log if entry[0] in ("eager_chunk", "replay", "verify_before_replay")]
    assert chunks == [("eager_chunk", anchor)] * 4
    synchronizes = [entry for entry in harness.log if entry[0] == "synchronize_device"]
    assert len(synchronizes) == 3  # the seed and the two around the hand-off; the non-blocking chunks add none
    # The blocking (timed) form synchronizes after every eager chunk so the wall is the chunk's.
    harness.log.clear()
    timed = prefill.run(list(range(64)), start_position=0, ple_context=None, time_each_chunk=True)
    assert len(timed.timing.chunk_replay_ms) == 2 and timed.timing.traced is False
    assert [entry[0] for entry in harness.log if entry[0] in ("eager_chunk", "synchronize_device")] == [
        "synchronize_device",
        "eager_chunk",
        "synchronize_device",
        "eager_chunk",
        "synchronize_device",
        "synchronize_device",
        "synchronize_device",
    ]
    # The traced form carries the flag its capture had (the chain passes its own); a replay never takes it per chunk.
    harness.log.clear()
    traced = Qwen38ChunkPrefill(
        model, "mesh", "state", "chunk_state", TRACE_ID, forced_step=forced_step, gdn_step_anchor=True
    )
    result = traced.run(list(range(32)), start_position=0, ple_context=None)
    assert result.timing.traced is True and result.timing.gdn_step_anchor is True
    assert [entry for entry in harness.log if entry[0] in ("eager_chunk", "replay")] == [("replay", False)]
    for bad in ("1", 1, None):
        with pytest.raises(ValueError):  # allow-pytest.raises: pure contract test
            Qwen38ChunkPrefill(model, "mesh", "s", "c", TRACE_ID, forced_step=forced_step, gdn_step_anchor=bad)
    with pytest.raises(ValueError):  # allow-pytest.raises: pure contract test
        Qwen38ChunkPrefill(model, "mesh", "s", "c", "77", forced_step=forced_step)


def test_timed_replays_block_and_skip_the_events(harness) -> None:
    result = harness.prefill.run(list(range(1, 161)), start_position=0, ple_context=None, time_each_chunk=True)
    assert result.timing.chunks == 5 and len(result.timing.chunk_replay_ms) == 5
    assert len(result.timing.chunk_host_ms) == 5 and all(ms >= 0.0 for ms in result.timing.chunk_host_ms)
    assert all(ms >= 0.0 for ms in result.timing.chunk_replay_ms)
    replays = [entry for entry in harness.log if entry[0] == "replay"]
    assert replays == [("replay", True)] * 5
    assert not any(entry[0] in ("record_event", "event_synchronize") for entry in harness.log)


def test_verify_can_be_skipped_and_the_budget_is_checked(harness) -> None:
    harness.prefill.verify_allocations = False
    result = harness.prefill.run(list(range(40)), start_position=0, ple_context=(3, 4))
    assert result.timing.verify_ms is None
    assert not any(entry[0] == "verify_before_replay" for entry in harness.log)
    assert result.ple_context == _expected_context(list(range(40)), (3, 4))
    with pytest.raises(ValueError):  # allow-pytest.raises: the prompt does not fit the allocated context
        harness.prefill.run(list(range(ALLOCATED_CONTEXT + 1)), start_position=0, ple_context=None)
    with pytest.raises(ValueError):  # allow-pytest.raises: the padded tail chunk would end past the allocated context
        harness.prefill.run(list(range(ALLOCATED_CONTEXT - 3)), start_position=0, ple_context=None)
    with pytest.raises(ValueError):  # allow-pytest.raises: pure contract test
        harness.prefill.run([1], start_position=-1, ple_context=None)
    with pytest.raises(ValueError):  # allow-pytest.raises: pure contract test
        Qwen38ChunkPrefill(harness.model, "mesh", "s", "c", TRACE_ID, forced_step=lambda t, c: c, event_interval=0)


def test_driver_source_pins() -> None:
    import inspect

    source = inspect.getsource(driver_module)
    assert source.count("ttnn._ttnn_execute_trace(") == 1 and "ttnn.execute_trace(" not in source
    assert source.count("UnsafeAllocationTracker(self.mesh).verify_before_replay(trace_id)") == 1
    run = inspect.getsource(Qwen38ChunkPrefill.run)
    order = (
        "self.forced_step(token, ple_context)",
        "ttnn.synchronize_device(self.mesh)",
        "self.model.reset_chunk_state_inplace(self.state, self.chunk_state)",
        "verify_before_replay",
        "self.model.write_chunk_accepted(self.chunk_state, accepted)",
        "prepared = self.model.prepare_chunk_inputs(chunk_state, rows, ple_context=ple_context)",
        "ple_context = prepared.contexts[real_rows]",
        "self.model.upload_chunk_inputs(chunk_state, prepared)",
        "self._run_chunk(blocking=True, kind=kind)",
        "self._run_chunk(blocking=False, kind=kind)",
        "ttnn.event_synchronize(ttnn.record_event(self.mesh, cq_id=0))",
        "self.model.finish_prefill(self.state, self.chunk_state, position)",
    )
    positions = [run.index(fragment) for fragment in order]
    assert positions == sorted(positions)
    # The host half (the lookup and the packing) is timed apart from the copies' enqueue and the slab's event wait.
    assert run.index("slab_prepare_ms.append((upload_started_ns - host_started_ns)") < run.index(
        "slab_upload_ms.append((replay_started_ns - upload_started_ns)"
    )
    assert "slab_wait_ms.append((time.perf_counter_ns() - wait_started_ns)" in run
    # A slab waits for the previous slab's event once its own replay is queued (the host runs one slab ahead).
    assert run.index("self._run_chunk(blocking=False, kind=kind)") < run.index("ttnn.event_synchronize(slab_event)")
    assert run.index("ttnn.event_synchronize(slab_event)") < run.index(
        "slab_event = ttnn.record_event(self.mesh, cq_id=0)"
    )
    assert run.count("ttnn.record_event(") == 2 and run.count("ttnn.event_synchronize(") == 2
    chunk = inspect.getsource(Qwen38ChunkPrefill._run_chunk)
    assert "ttnn._ttnn_execute_trace(self.mesh, trace_id, cq_id=0, blocking=blocking)" in chunk
    assert (
        "self.model.forward_prefill_chunk_generic(\n"
        "            chunk_state, self.state, gdn_step_anchor=self.gdn_step_anchor and short, mtp=self.mtp if short else None\n"
        "        )"
    ) in chunk
    assert 'short = kind == "short"' in chunk
    assert CHUNK_PAD_TOKEN_ID == 0 and driver_module.CHUNK_EVENT_INTERVAL == 4
