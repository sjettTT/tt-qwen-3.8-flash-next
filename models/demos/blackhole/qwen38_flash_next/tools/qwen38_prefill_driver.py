# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Chunked prefill of one prompt on the generic chain: alignment steps, traced chunks (128 rows, then 32), the padded
tail, the hand-off.

The chain owner (the chat session, a runner) hands over the model, its generic and chunk states, the captured chunk
traces and ``forced_step``, which teacher-forces one token through the decode traces at the current position and
returns the next n-gram context.  :meth:`Qwen38ChunkPrefill.run` then prefills ``token_ids`` from ``start_position``:
``(32 - P % 32) % 32`` forced steps so the first chunk starts at ``P % 32 == 0``; the chunk carry seeded from the
decode buffers (``reset_chunk_state_inplace``, the 32-row state first: the 128-row state shares its histories);
``N // 128`` long chunks when the chain captured the 128-row trace; then ``M // 32`` full 32-row chunks (accept scalar
31) over the remainder ``M``; if ``M % 32 = r > 0`` one padded chunk whose rows ``r .. 31`` are
:data:`CHUNK_PAD_TOKEN_ID` with accept scalar ``r - 1``; ``finish_prefill`` from the 32-row state.  The host work of
chunk i + 1 (the n-gram lookups from the running context, the token-row and PLE-row writes) is queued behind chunk
i's replay and an event every ``event_interval`` chunks bounds the run-ahead; the eager seed and hand-off run only
after a device synchronize (their transients must not land in a running trace's addresses).

Timing (the rule every measurement here follows): ``verify_before_replay`` once per trace, outside the window; the raw
``ttnn._ttnn_execute_trace`` per chunk is timed only with ``time_each_chunk`` (blocking replays, no host overlap); the
end-to-end wall (forced steps, host writes, replays, hand-off) is reported separately.

``gdn_step_anchor`` (default off) is the GDN state re-anchor of every 32-row chunk (``forward_prefill_chunk_generic``'s
keyword: the committed GDN state through the 1-row FP32 step arithmetic instead of the chunk kernel's).  With a
captured trace the driver replays what the capture baked in, so the chain owner passes the flag its capture carried;
without one (``chunk_trace_id=None``, the eager form of the micro-tests) the driver runs the chunk body itself and
passes the flag per chunk.  The 128-row form has no anchor: a chain with the anchor on runs 32-row chunks only.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROWS, LONG_CHUNK_ROWS
from ttnn.unsafe_allocation_tracker import UnsafeAllocationTracker

# Rows past the prompt in the padded tail chunk: any in-vocabulary id; their state is never read after the hand-off.
CHUNK_PAD_TOKEN_ID = 0
CHUNK_EVENT_INTERVAL = 4


def alignment_steps(start_position: int, count: int) -> int:
    """Tokens teacher-forced through the decode traces before the first chunk: up to the next multiple of 32."""

    return min(count, (CHUNK_ROWS - start_position % CHUNK_ROWS) % CHUNK_ROWS)


def long_chunk_count(count: int, *, long_chunks: bool) -> int:
    """The 128-row chunks of a ``count``-token chunked prefill: ``count // 128`` when the chain has the long trace."""

    return count // LONG_CHUNK_ROWS if long_chunks else 0


def chunk_accepts(count: int) -> list[int]:
    """The accept scalar of every 32-row chunk of a ``count``-token chunked prefill: 31 per full chunk, r - 1 for the tail."""

    full, tail = divmod(count, CHUNK_ROWS)
    return [CHUNK_ROWS - 1] * full + ([tail - 1] if tail else [])


@dataclass(frozen=True)
class Qwen38PrefillTiming:
    alignment_steps: int
    chunks: int  # the 32-row chunks (full and the padded tail)
    tail_rows: int
    chunk_replay_ms: tuple[float, ...]  # raw blocking replays, one per 32-row chunk; empty unless time_each_chunk
    chunk_host_ms: tuple[float, ...]  # the host writes ahead of each 32-row chunk (lookups, rows); empty unless timed
    verify_ms: float | None
    handoff_ms: float
    wall_ms: float
    traced: bool = True  # chunks replayed the captured traces (False: the eager chunk body ran per chunk)
    gdn_step_anchor: bool = False  # the GDN state re-anchor the 32-row chunks ran with
    long_chunks: int = 0  # the 128-row chunks ahead of the 32-row ones
    long_chunk_replay_ms: tuple[float, ...] = ()  # raw blocking replays, one per 128-row chunk; empty unless timed
    long_chunk_host_ms: tuple[float, ...] = ()


@dataclass(frozen=True)
class Qwen38PrefillResult:
    position: int  # P after the prefill: the count of positions consumed
    ple_context: tuple[int, int] | None  # the n-gram context of the committed stream
    timing: Qwen38PrefillTiming
    stopped: str | None = None  # the should_stop reason that ended the prefill after the chunks already replayed


class Qwen38ChunkPrefill:
    """One prompt's chunked prefill on a chain whose chunk traces are captured (see the module docstring)."""

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
        long_chunk_state: Any | None = None,
        long_chunk_trace_id: Any | None = None,
        mtp: Any = None,
    ) -> None:
        if isinstance(event_interval, bool) or type(event_interval) is not int or event_interval <= 0:
            raise ValueError(f"event interval must be a positive int, got {event_interval!r}")
        for name, trace_id in (("chunk", chunk_trace_id), ("long chunk", long_chunk_trace_id)):
            if isinstance(trace_id, (bool, str, float)):  # the runtime's trace handle (MeshTraceId) or None
                raise ValueError(f"{name} trace id must be a trace handle or None (the eager body), got {trace_id!r}")
        if type(gdn_step_anchor) is not bool:
            raise ValueError(f"gdn_step_anchor must be a bool, got {gdn_step_anchor!r}")
        if long_chunk_state is not None and gdn_step_anchor:
            raise ValueError("the GDN step anchor is a 32-row chunk option: no long chunks with the anchor on")
        if (long_chunk_trace_id is not None) and (long_chunk_state is None or chunk_trace_id is None):
            raise ValueError("a long chunk trace needs the long chunk state and the 32-row chunk trace")
        if long_chunk_state is not None and mtp is not None:
            raise ValueError("the MTP chunk extension is a 32-row chunk option: no long chunks with MTP drafting")
        self.model = model
        self.mesh = mesh
        self.state = state
        self.chunk_state = chunk_state
        self.chunk_trace_id = chunk_trace_id
        self.long_chunk_state = long_chunk_state
        self.long_chunk_trace_id = long_chunk_trace_id
        self.forced_step = forced_step
        self.pad_token_id = pad_token_id
        self.event_interval = event_interval
        self.verify_allocations = verify_allocations
        self.gdn_step_anchor = gdn_step_anchor
        # The MTP-drafting chain's chunk extension (mtp_v2.Qwen38TTNNMTPChunkExtension): the chunk trace was captured
        # with it, so every chunk also takes its 32 MTP tokens (the tokens one position ahead) and the hand-off
        # includes the MTP layer.
        self.mtp = mtp

    def _run_chunk(self, *, blocking: bool, long: bool = False) -> None:
        """One chunk at the device position: the captured trace's replay, or the eager chunk body with the
        re-anchor flag and the MTP extension passed per chunk (a blocking eager chunk synchronizes so its wall is
        the chunk's)."""

        chunk_state = self.long_chunk_state if long else self.chunk_state
        trace_id = self.long_chunk_trace_id if long else self.chunk_trace_id
        if trace_id is not None:
            ttnn._ttnn_execute_trace(self.mesh, trace_id, cq_id=0, blocking=blocking)
            return
        self.model.forward_prefill_chunk_generic(
            chunk_state, self.state, gdn_step_anchor=self.gdn_step_anchor and not long, mtp=None if long else self.mtp
        )
        if blocking:
            ttnn.synchronize_device(self.mesh)

    def run(
        self,
        token_ids: Sequence[int],
        *,
        start_position: int,
        ple_context: tuple[int, int] | None,
        time_each_chunk: bool = False,
        following_token: int | None = None,
        should_stop: Callable[[], str | None] | None = None,
    ) -> Qwen38PrefillResult:
        """Prefill ``token_ids`` at positions ``start_position ..``; the caller's first decode replay consumes the
        token after them (``following_token``, the MTP token of the last prefilled position when the chain drafts).
        Returns the position after the prefill and the committed stream's n-gram context.  ``should_stop`` is
        polled at every event sync: a reason ends the prefill after the chunks already replayed (the hand-off runs
        at that position; ``stopped`` carries the reason, ``position`` what was consumed)."""

        tokens = [int(token) for token in token_ids]
        if isinstance(start_position, bool) or type(start_position) is not int or start_position < 0:
            raise ValueError(f"start position must be a non-negative int, got {start_position!r}")
        if self.mtp is not None and following_token is None:
            raise ValueError("an MTP-drafting chain's chunked prefill needs the token following the prefilled ones")
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
        long_chunks = long_chunk_count(len(remaining), long_chunks=self.long_chunk_state is not None)
        accepts = chunk_accepts(len(remaining) - long_chunks * LONG_CHUNK_ROWS)
        # The chunk sequence: (chunk state, trace, rows, accept scalar or None) with the long chunks first.
        plan: list[tuple[Any, bool, list[int], int | None]] = [
            (self.long_chunk_state, True, remaining[LONG_CHUNK_ROWS * index : LONG_CHUNK_ROWS * (index + 1)], None)
            for index in range(long_chunks)
        ]
        offset = long_chunks * LONG_CHUNK_ROWS
        plan += [
            (
                self.chunk_state,
                False,
                remaining[offset + CHUNK_ROWS * index : offset + CHUNK_ROWS * (index + 1)],
                accepted,
            )
            for index, accepted in enumerate(accepts)
        ]
        verify_ms = None
        handoff_ms = 0.0
        replay_ms: list[float] = []
        host_ms: list[float] = []
        long_replay_ms: list[float] = []
        long_host_ms: list[float] = []
        stopped = None
        consumed = len(remaining)
        chunks = len(accepts)
        long_done = long_chunks
        if plan:
            if position % CHUNK_ROWS:
                raise AssertionError(f"chunks must start at P % {CHUNK_ROWS} == 0, got P = {position}")
            if position + long_chunks * LONG_CHUNK_ROWS + CHUNK_ROWS * len(accepts) > self.model.allocated_context:
                raise ValueError(
                    f"the padded tail chunk ends at {position + long_chunks * LONG_CHUNK_ROWS + CHUNK_ROWS * len(accepts)}, "
                    f"past the allocated context {self.model.allocated_context}"
                )
            ttnn.synchronize_device(self.mesh)
            self.model.reset_chunk_state_inplace(self.state, self.chunk_state)
            if long_chunks:
                self.model.reset_chunk_state_inplace(self.state, self.long_chunk_state)
            if self.mtp is not None:
                self.mtp.reset_chunk()
            if self.verify_allocations and self.chunk_trace_id is not None:
                verify_started_ns = time.perf_counter_ns()
                for trace_id in (self.chunk_trace_id,) + ((self.long_chunk_trace_id,) if long_chunks else ()):
                    UnsafeAllocationTracker(self.mesh).verify_before_replay(trace_id)
                verify_ms = (time.perf_counter_ns() - verify_started_ns) / 1_000_000
            # The MTP layer's tokens sit one position ahead: the chunk's rows shifted by one, then the following token.
            following = remaining[1:] + [following_token if following_token is not None else self.pad_token_id]
            row_offset = 0
            for index, (chunk_state, long, rows, accepted) in enumerate(plan):
                real_rows = len(rows)
                start = row_offset
                row_offset += real_rows
                host_started_ns = time.perf_counter_ns()
                if accepted is not None and accepted != CHUNK_ROWS - 1:
                    self.model.write_chunk_accepted(self.chunk_state, accepted)
                    rows = rows + [self.pad_token_id] * (CHUNK_ROWS - real_rows)
                contexts = self.model.write_chunk_inputs(chunk_state, rows, ple_context=ple_context)
                ple_context = contexts[real_rows]
                if self.mtp is not None:
                    ahead = following[start : start + CHUNK_ROWS]
                    self.mtp.write_tokens(self.model, ahead + [self.pad_token_id] * (CHUNK_ROWS - len(ahead)))
                if time_each_chunk:
                    replay_started_ns = time.perf_counter_ns()
                    (long_host_ms if long else host_ms).append((replay_started_ns - host_started_ns) / 1_000_000)
                    self._run_chunk(blocking=True, long=long)
                    (long_replay_ms if long else replay_ms).append(
                        (time.perf_counter_ns() - replay_started_ns) / 1_000_000
                    )
                else:
                    self._run_chunk(blocking=False, long=long)
                    if (index + 1) % self.event_interval == 0:
                        ttnn.event_synchronize(ttnn.record_event(self.mesh, cq_id=0))
                        stopped = None if should_stop is None else should_stop()
                        if stopped is not None:
                            done = plan[: index + 1]
                            consumed = sum(len(done_rows) for _, _, done_rows, _ in done)
                            chunks = sum(1 for _, done_long, _, _ in done if not done_long)
                            long_done = sum(1 for _, done_long, _, _ in done if done_long)
                            break
            position += consumed
            handoff_started_ns = time.perf_counter_ns()
            ttnn.synchronize_device(self.mesh)
            self.model.finish_prefill(self.state, self.chunk_state, position)
            if self.mtp is not None:
                self.mtp.finish_chunk(self.model, prefilled=position)
            ttnn.synchronize_device(self.mesh)
            handoff_ms = (time.perf_counter_ns() - handoff_started_ns) / 1_000_000
        timing = Qwen38PrefillTiming(
            alignment_steps=aligned,
            chunks=chunks,
            tail_rows=consumed % CHUNK_ROWS,
            chunk_replay_ms=tuple(replay_ms),
            chunk_host_ms=tuple(host_ms),
            verify_ms=verify_ms,
            handoff_ms=handoff_ms,
            wall_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
            traced=self.chunk_trace_id is not None,
            gdn_step_anchor=self.gdn_step_anchor,
            long_chunks=long_done,
            long_chunk_replay_ms=tuple(long_replay_ms),
            long_chunk_host_ms=tuple(long_host_ms),
        )
        return Qwen38PrefillResult(position, ple_context, timing, stopped)


__all__ = [
    "CHUNK_EVENT_INTERVAL",
    "CHUNK_PAD_TOKEN_ID",
    "Qwen38ChunkPrefill",
    "Qwen38PrefillResult",
    "Qwen38PrefillTiming",
    "alignment_steps",
    "chunk_accepts",
    "long_chunk_count",
]
