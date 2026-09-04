# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Chunked prefill of one prompt on the generic chain: alignment steps, traced 32-row chunks, the padded tail, the hand-off.

The chain owner (the chat session, a runner) hands over the model, its generic and chunk states, the captured chunk
trace and ``forced_step``, which teacher-forces one token through the decode traces at the current position and
returns the next n-gram context.  :meth:`Qwen38ChunkPrefill.run` then prefills ``token_ids`` from ``start_position``:
``(32 - P % 32) % 32`` forced steps so the first chunk starts at ``P % 32 == 0``; the chunk carry seeded from the
decode buffers (``reset_chunk_state_inplace``); ``N // 32`` full chunks (accept scalar 31); if ``N % 32 = r > 0`` one
padded chunk whose rows ``r .. 31`` are :data:`CHUNK_PAD_TOKEN_ID` with accept scalar ``r - 1``; ``finish_prefill``.
The host work of chunk i + 1 (the 32 n-gram lookups from the running context, the token-row and PLE-row writes) is
queued behind chunk i's replay and an event every ``event_interval`` chunks bounds the run-ahead; the eager seed and
hand-off run only after a device synchronize (their transients must not land in a running trace's addresses).

Timing (the campaign rule): ``verify_before_replay`` once, outside the window; the raw ``ttnn._ttnn_execute_trace``
per chunk is timed only with ``time_each_chunk`` (blocking replays, no host overlap); the end-to-end wall (forced
steps, host writes, replays, hand-off) is reported separately.

``gdn_step_anchor`` (default off) is the GDN state re-anchor of every chunk (``forward_prefill_chunk_generic``'s
keyword: the committed GDN state through the 1-row FP32 step arithmetic instead of the chunk kernel's).  With a
captured trace the driver replays what the capture baked in, so the chain owner passes the flag its capture carried;
without one (``chunk_trace_id=None``, the eager form of the micro-tests) the driver runs the chunk body itself and
passes the flag per chunk.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROWS
from ttnn.unsafe_allocation_tracker import UnsafeAllocationTracker

# Rows past the prompt in the padded tail chunk: any in-vocabulary id; their state is never read after the hand-off.
CHUNK_PAD_TOKEN_ID = 0
CHUNK_EVENT_INTERVAL = 4


def alignment_steps(start_position: int, count: int) -> int:
    """Tokens teacher-forced through the decode traces before the first chunk: up to the next multiple of 32."""

    return min(count, (CHUNK_ROWS - start_position % CHUNK_ROWS) % CHUNK_ROWS)


def chunk_accepts(count: int) -> list[int]:
    """The accept scalar of every chunk of a ``count``-token chunked prefill: 31 per full chunk, r - 1 for the tail."""

    full, tail = divmod(count, CHUNK_ROWS)
    return [CHUNK_ROWS - 1] * full + ([tail - 1] if tail else [])


@dataclass(frozen=True)
class Qwen38PrefillTiming:
    alignment_steps: int
    chunks: int
    tail_rows: int
    chunk_replay_ms: tuple[float, ...]  # raw blocking replays, one per chunk; empty unless time_each_chunk
    chunk_host_ms: tuple[float, ...]  # the host writes ahead of each chunk (lookups, rows); empty unless timed
    verify_ms: float | None
    handoff_ms: float
    wall_ms: float
    traced: bool = True  # chunks replayed the captured trace (False: the eager chunk body ran per chunk)
    gdn_step_anchor: bool = False  # the GDN state re-anchor the chunks ran with


@dataclass(frozen=True)
class Qwen38PrefillResult:
    position: int  # P after the prefill: the count of positions consumed
    ple_context: tuple[int, int] | None  # the n-gram context of the committed stream
    timing: Qwen38PrefillTiming


class Qwen38ChunkPrefill:
    """One prompt's chunked prefill on a chain whose chunk trace is captured (see the module docstring)."""

    def __init__(
        self,
        model: Any,
        mesh: Any,
        state: Any,
        chunk_state: Any,
        chunk_trace_id: Any | None,
        *,
        forced_step: Callable[[int, tuple[int, int] | None], tuple[int, int] | None],
        pad_token_id: int = CHUNK_PAD_TOKEN_ID,
        event_interval: int = CHUNK_EVENT_INTERVAL,
        verify_allocations: bool = True,
        gdn_step_anchor: bool = False,
    ) -> None:
        if isinstance(event_interval, bool) or type(event_interval) is not int or event_interval <= 0:
            raise ValueError(f"event interval must be a positive int, got {event_interval!r}")
        if isinstance(chunk_trace_id, (bool, str, float)):  # the runtime's trace handle (MeshTraceId) or None
            raise ValueError(f"chunk trace id must be a trace handle or None (the eager body), got {chunk_trace_id!r}")
        if type(gdn_step_anchor) is not bool:
            raise ValueError(f"gdn_step_anchor must be a bool, got {gdn_step_anchor!r}")
        self.model = model
        self.mesh = mesh
        self.state = state
        self.chunk_state = chunk_state
        self.chunk_trace_id = chunk_trace_id
        self.forced_step = forced_step
        self.pad_token_id = pad_token_id
        self.event_interval = event_interval
        self.verify_allocations = verify_allocations
        self.gdn_step_anchor = gdn_step_anchor

    def _run_chunk(self, *, blocking: bool) -> None:
        """One chunk at the device position: the captured trace's replay, or the eager chunk body with the
        re-anchor flag passed per chunk (a blocking eager chunk synchronizes so its wall is the chunk's)."""

        if self.chunk_trace_id is not None:
            ttnn._ttnn_execute_trace(self.mesh, self.chunk_trace_id, cq_id=0, blocking=blocking)
            return
        self.model.forward_prefill_chunk_generic(self.chunk_state, self.state, gdn_step_anchor=self.gdn_step_anchor)
        if blocking:
            ttnn.synchronize_device(self.mesh)

    def run(
        self,
        token_ids: Sequence[int],
        *,
        start_position: int,
        ple_context: tuple[int, int] | None,
        time_each_chunk: bool = False,
    ) -> Qwen38PrefillResult:
        """Prefill ``token_ids`` at positions ``start_position ..``; the caller's first decode replay consumes the
        token after them.  Returns the position after the prefill and the committed stream's n-gram context."""

        tokens = [int(token) for token in token_ids]
        if isinstance(start_position, bool) or type(start_position) is not int or start_position < 0:
            raise ValueError(f"start position must be a non-negative int, got {start_position!r}")
        if start_position + len(tokens) > self.model.allocated_context:
            raise ValueError(
                f"prefill of {len(tokens)} tokens from position {start_position} exceeds the allocated context "
                f"{self.model.allocated_context}"
            )
        started_ns = time.perf_counter_ns()
        aligned = alignment_steps(start_position, len(tokens))
        for token in tokens[:aligned]:
            ple_context = self.forced_step(token, ple_context)
        position = start_position + aligned
        remaining = tokens[aligned:]
        accepts = chunk_accepts(len(remaining))
        verify_ms = None
        handoff_ms = 0.0
        replay_ms: list[float] = []
        host_ms: list[float] = []
        if accepts:
            if position % CHUNK_ROWS:
                raise AssertionError(f"chunks must start at P % {CHUNK_ROWS} == 0, got P = {position}")
            if position + CHUNK_ROWS * len(accepts) > self.model.allocated_context:
                raise ValueError(
                    f"the padded tail chunk ends at {position + CHUNK_ROWS * len(accepts)}, past the allocated context "
                    f"{self.model.allocated_context}"
                )
            ttnn.synchronize_device(self.mesh)
            self.model.reset_chunk_state_inplace(self.state, self.chunk_state)
            if self.verify_allocations and self.chunk_trace_id is not None:
                verify_started_ns = time.perf_counter_ns()
                UnsafeAllocationTracker(self.mesh).verify_before_replay(self.chunk_trace_id)
                verify_ms = (time.perf_counter_ns() - verify_started_ns) / 1_000_000
            for index, accepted in enumerate(accepts):
                rows = remaining[CHUNK_ROWS * index : CHUNK_ROWS * (index + 1)]
                real_rows = len(rows)
                host_started_ns = time.perf_counter_ns()
                if accepted != CHUNK_ROWS - 1:
                    self.model.write_chunk_accepted(self.chunk_state, accepted)
                    rows = rows + [self.pad_token_id] * (CHUNK_ROWS - real_rows)
                contexts = self.model.write_chunk_inputs(self.chunk_state, rows, ple_context=ple_context)
                ple_context = contexts[real_rows]
                if time_each_chunk:
                    replay_started_ns = time.perf_counter_ns()
                    host_ms.append((replay_started_ns - host_started_ns) / 1_000_000)
                    self._run_chunk(blocking=True)
                    replay_ms.append((time.perf_counter_ns() - replay_started_ns) / 1_000_000)
                else:
                    self._run_chunk(blocking=False)
                    if (index + 1) % self.event_interval == 0:
                        ttnn.event_synchronize(ttnn.record_event(self.mesh, cq_id=0))
            position += len(remaining)
            handoff_started_ns = time.perf_counter_ns()
            ttnn.synchronize_device(self.mesh)
            self.model.finish_prefill(self.state, self.chunk_state, position)
            ttnn.synchronize_device(self.mesh)
            handoff_ms = (time.perf_counter_ns() - handoff_started_ns) / 1_000_000
        timing = Qwen38PrefillTiming(
            alignment_steps=aligned,
            chunks=len(accepts),
            tail_rows=len(remaining) % CHUNK_ROWS,
            chunk_replay_ms=tuple(replay_ms),
            chunk_host_ms=tuple(host_ms),
            verify_ms=verify_ms,
            handoff_ms=handoff_ms,
            wall_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
            traced=self.chunk_trace_id is not None,
            gdn_step_anchor=self.gdn_step_anchor,
        )
        return Qwen38PrefillResult(position, ple_context, timing)


__all__ = [
    "CHUNK_EVENT_INTERVAL",
    "CHUNK_PAD_TOKEN_ID",
    "Qwen38ChunkPrefill",
    "Qwen38PrefillResult",
    "Qwen38PrefillTiming",
    "alignment_steps",
    "chunk_accepts",
]
