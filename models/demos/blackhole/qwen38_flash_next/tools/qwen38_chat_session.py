# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Chat session over the single-trace HEAD/TAIL chain of the B=1 timing runner.

The served decode path is the runner's chain unchanged: 4 HEAD + 4 TAIL traces
of the position-generic body, replayed as ``HEAD[t mod 4]``, host PLE row
refresh, ``TAIL[t mod 4]``.  Prefill has two modes.  ``teacher_forced`` writes
each prompt token into the persistent token row before its HEAD, so the
device's own greedy resolve (the last op of every TAIL) is overwritten before
the embedding reads the row.  ``chunked`` (the default when the chain captured
the chunk trace) runs all prompt tokens but the last through the 32-row chunk
trace with ``Qwen38ChunkPrefill``: teacher-forced steps up to the next multiple
of 32, the chunk carry seeded from the decode buffers, full chunks, a padded
tail chunk, the eager hand-off (``finish_prefill``), then the last prompt token
is teacher-forced as the first decode replay.  Suffixes shorter than
``CHUNK_PREFILL_MIN_ROWS`` after the alignment steps are teacher-forced.
Generation is the runner's evented loop (non-blocking row read, event, HEAD,
event wait, PLE refresh, TAIL).  Multi-turn requests reuse the device state when
the new prompt extends the committed input sequence (the suffix is prefilled
from the committed position); otherwise the generic state is reset in place.

Sampling is an optional tail of the same chain (``tools/qwen38_sampling_step.py``):
TAIL's epilogue also writes a candidate row, and a request with ``temperature > 0``
runs the sampled loop (read the row after TAIL, sample on the host, write the token
into the row before HEAD).  Greedy requests take the loop above untouched.

``Qwen38ChatSession`` speaks to the device only through a chain object with the
per-step primitives; ``Qwen38TracedChain`` is the hardware one, the no-device
test drives the session with a scripted chain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
from ttnn.unsafe_allocation_tracker import UnsafeAllocationTracker

import ttnn
from models.demos.blackhole.qwen38_flash_next.chat import (
    EOS_TOKEN_IDS,
    IM_START_ID,
    REASONING_EFFORTS,
    TOKENIZER_SIZE,
    VOCAB_SIZE,
    Qwen38ChatFormatError,
)
from models.demos.blackhole.qwen38_flash_next.tools import hardware_profiles, physical_route
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_protocol as protocol
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling_step as sampling_step
from models.demos.blackhole.qwen38_flash_next.tools import resident_decode
from models.demos.blackhole.qwen38_flash_next.tools.evidence_records import Marker
from models.demos.blackhole.qwen38_flash_next.tools.hardware_profiles import ResidentHardwareProfile
from models.demos.blackhole.qwen38_flash_next.tools.live_decode_diagnostic import (
    Qwen38LiveDecodeConstruction,
    construct_live_decode_diagnostic,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_prefill_driver import (
    Qwen38ChunkPrefill,
    Qwen38PrefillResult,
    alignment_steps,
)
from models.demos.blackhole.qwen38_flash_next.tools.run_full_cpu_oracle import OFFICIAL_CHAT_SYSTEM_PROMPT
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import (
    RESIDENT_CONTEXT_HEADROOM,
    RESIDENT_MAX_QSA_CACHE_CAPACITY,
    RESIDENT_QSA_CACHE_CAPACITIES,
    Qwen38ResidentContext,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROWS, MESH_SHAPE, TP_SIZE, Qwen38MeshContract
from models.demos.blackhole.qwen38_flash_next.ttnn.model import GENERIC_HEAD_LAYERS, Qwen38TTNNGenericTraceKey

# The device position P must stay below allocated_context (RoPE table lookup, KV write); the consumed EOS step and a
# little slack are the headroom.  CONTEXT_LIMIT is the default (32k) build's; a session derives its own from the
# chain's allocated context.
CONTEXT_HEADROOM = RESIDENT_CONTEXT_HEADROOM
CONTEXT_LIMIT = Qwen38ResidentContext().context_limit
PREFILL_EVENT_INTERVAL = 16
# A request's completion budget is bounded by the remaining context, context_limit - prompt tokens, and defaults to
# it (require_budget).  MAX_TOKENS_BOUND only types an explicit max_tokens before the prompt is known: the largest
# admitted context.
MAX_TOKENS_BOUND = max(RESIDENT_QSA_CACHE_CAPACITIES)
PREFILL_MODES = ("chunked", "teacher_forced")
DEFAULT_PREFILL_MODE = "chunked"
# Rows left for the chunk trace after the alignment steps below which the chunk path costs more than teacher forcing
# them at about 50 ms per token: the serving hand-off measured 240-340 ms (4x p150b, 2026-09-03) on top of the seed and
# the padded chunk replay, so the break-even is about 10-12 rows.
CHUNK_PREFILL_MIN_ROWS = 16
# The chunk warm pass embeds one token per vocabulary owner in every lane group (the decode warm pass's ids).
WARM_CHUNK_TOKEN_IDS = tuple(
    resident_decode.SEQUENTIAL_TRACE_WARM_EMBEDDING_TOKEN_IDS[index % TP_SIZE] for index in range(CHUNK_ROWS)
)
# The CPU acceptance study's rendering (mtp_acceptance_cpu_v2 at a97cb9e6b0): the
# 12 prompt records render identically only with these.
SYSTEM_PROMPT = OFFICIAL_CHAT_SYSTEM_PROMPT
ENABLE_THINKING = False
PRESERVE_THINKING = True
REASONING_EFFORT = "low"
RESIDUE_CLASSES = resident_decode.SINGLE_TRACE_RESIDUE_CLASS_TRACES
SEED_TOKEN_ID = IM_START_ID
THINK_END_ID = protocol.THINK_END_ID


class Qwen38ChatRequestError(ValueError):
    """A request the session refuses (HTTP 400): bad messages, context length, bad token budget."""


class Qwen38ChatChainError(RuntimeError):
    """The device chain disagrees with its host mirror; the model owner cannot be trusted afterwards."""


@dataclass(frozen=True)
class Qwen38ChatCompletion:
    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int
    prefix_reused: int
    reset: bool
    prefill_tokens: int
    prefill_seconds: float
    ttft_seconds: float
    decode_seconds: float
    tokens_per_second: float | None
    position: int
    # The prefill path this request took ("chunked" or "teacher_forced") and its shape: teacher-forced steps
    # (alignment steps plus the last prompt token, or every token), chunk replays, real rows of the padded tail,
    # the eager hand-off time and the wall per prompt token.
    prefill_mode: str = "teacher_forced"
    prefill_forced_tokens: int = 0
    prefill_chunks: int = 0
    prefill_tail_rows: int = 0
    prefill_handoff_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "prefix_reused": self.prefix_reused,
            "reset": self.reset,
            "prefill_tokens": self.prefill_tokens,
            "prefill_seconds": round(self.prefill_seconds, 4),
            "prefill_mode": self.prefill_mode,
            "prefill_forced_tokens": self.prefill_forced_tokens,
            "prefill_chunks": self.prefill_chunks,
            "prefill_tail_rows": self.prefill_tail_rows,
            "prefill_handoff_ms": round(self.prefill_handoff_ms, 3),
            "prefill_ms_per_prompt_token": (
                None if not self.prefill_tokens else round(1000.0 * self.prefill_seconds / self.prefill_tokens, 3)
            ),
            "ttft_seconds": round(self.ttft_seconds, 4),
            "decode_seconds": round(self.decode_seconds, 4),
            "tokens_per_second": None if self.tokens_per_second is None else round(self.tokens_per_second, 3),
            "position": self.position,
        }


class Qwen38ChatSession:
    """One conversation owner: the committed input sequence, the PLE n-gram context, the traced chain.

    ``template`` is the pinned ``Qwen38OfficialChatTemplate`` (``render`` and
    ``tokenizer.decode`` are all it uses).  ``chain`` offers the per-step
    primitives of ``Qwen38TracedChain``.
    """

    def __init__(
        self,
        chain: Any,
        template: Any,
        *,
        context_limit: int | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        prefill_mode: str = DEFAULT_PREFILL_MODE,
    ) -> None:
        # The chain's allocated context (a scripted test chain without one is the default build's); the limit
        # defaults to that context less the headroom.
        allocated_context = getattr(chain, "allocated_context", RESIDENT_MAX_QSA_CACHE_CAPACITY)
        if context_limit is None:
            context_limit = Qwen38ResidentContext(allocated_context).context_limit
        if type(context_limit) is not int or not 1 < context_limit <= allocated_context:
            raise ValueError(f"context limit must be in (1, {allocated_context}], got {context_limit!r}")
        self.allocated_context = allocated_context
        if prefill_mode not in PREFILL_MODES:
            raise ValueError(f"prefill mode must be one of {PREFILL_MODES}, got {prefill_mode!r}")
        self.chain = chain
        self.template = template
        self.context_limit = context_limit
        self.clock_ns = clock_ns
        # A chain without the chunk trace (teacher-forced open, the scripted test chain) serves teacher forcing.
        self.chunk_trace_available = getattr(chain, "chunk_trace_id", None) is not None
        self.prefill_mode = prefill_mode if self.chunk_trace_available else "teacher_forced"
        self.sampling = getattr(chain, "sampling", None)  # the candidate-row extension, or None: greedy only
        self.committed: list[int] = []
        self.ple_context: tuple[int, int] | None = None
        self.last_finish: str | None = None
        self.row_unconsumed = False  # the next token sits in the row (after length, disconnected, a hook stop)
        self.requests_served = 0
        self.last_tokens_per_second: float | None = None
        self.poisoned = False

    # -- request rendering -------------------------------------------------------------------

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        enable_thinking: bool = ENABLE_THINKING,
        reasoning_effort: str = REASONING_EFFORT,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> list[int]:
        """Prompt token ids of ``messages`` (and the client's ``tools``) under the acceptance study's flags; the
        study's system prompt when none.  The protocol module renders: bitwise the reference template."""

        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
            raise Qwen38ChatRequestError("messages must be a nonempty list")
        if not all(isinstance(message, Mapping) for message in messages):
            raise Qwen38ChatRequestError("every message must be an object")
        if messages[0].get("role") != "system":
            messages = ({"role": "system", "content": SYSTEM_PROMPT}, *messages)
        if reasoning_effort not in REASONING_EFFORTS:
            raise Qwen38ChatRequestError(
                f"reasoning_effort must be one of {REASONING_EFFORTS}, got {reasoning_effort!r}"
            )
        try:
            return protocol.render_prompt(
                self.template.tokenizer,
                list(messages),
                list(tools),
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
            )
        except Qwen38ChatFormatError as error:
            raise Qwen38ChatRequestError(str(error)) from error

    def require_budget(self, prompt_tokens: int, max_tokens: int | None) -> int:
        """The request's completion budget, refused before touching the device when the prompt and the budget do
        not fit the context limit (which already holds the consumed EOS step).  ``None`` (the client sent no
        max_tokens) is the remaining context, ``context_limit - prompt_tokens``."""

        remaining = self.context_limit - prompt_tokens
        if max_tokens is None:
            if remaining < 1:
                raise Qwen38ChatRequestError(
                    f"context_length_exceeded: prompt {prompt_tokens} tokens leave {remaining} of the limit "
                    f"{self.context_limit} for the completion; at least 1 is needed"
                )
            return remaining
        if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_TOKENS_BOUND:
            raise Qwen38ChatRequestError(
                f"max_tokens must be an integer in [1, {MAX_TOKENS_BOUND}], got {max_tokens!r}"
            )
        if max_tokens > remaining:
            raise Qwen38ChatRequestError(
                f"context_length_exceeded: prompt {prompt_tokens} + max_tokens {max_tokens} = "
                f"{prompt_tokens + max_tokens} exceeds the limit {self.context_limit}; "
                f"at most {max(remaining, 0)} tokens remain"
            )
        return max_tokens

    # -- the steps ---------------------------------------------------------------------------

    def _forced_step(self, token_id: int) -> None:
        """A' write x_t into the token row (behind TAIL(t-1)), C HEAD(t), F PLE row for x_t, G TAIL(t)."""

        residue = len(self.committed) % RESIDUE_CLASSES
        self.chain.write_token_row(token_id)
        self.chain.execute_head(residue)
        self.ple_context = self.chain.refresh_ple_row(token_id, self.ple_context)
        self.chain.execute_tail(residue)
        self.committed.append(token_id)

    def _prefill(self, token_ids: Sequence[int], should_stop: Callable[[], str | None] | None) -> str | None:
        """Forced steps; at every event sync the hook may end the request (the row then holds an unconsumed token)."""

        for index, token_id in enumerate(token_ids, start=1):
            self._forced_step(token_id)
            if index % PREFILL_EVENT_INTERVAL == 0:
                # Bounds the host run-ahead and surfaces a device error at a known token.
                self.chain.event_synchronize(self.chain.record_event())
                reason = None if should_stop is None else should_stop()
                if reason is not None:
                    return reason
        return None

    def resolve_prefill_mode(self, prefill_mode: str | None) -> str:
        """The session's mode, or a request's override; ``chunked`` needs the chain's chunk trace."""

        if prefill_mode is None:
            return self.prefill_mode
        if prefill_mode not in PREFILL_MODES:
            raise Qwen38ChatRequestError(f"prefill_mode must be one of {PREFILL_MODES}, got {prefill_mode!r}")
        if prefill_mode == "chunked" and not self.chunk_trace_available:
            raise Qwen38ChatRequestError("prefill_mode chunked: this server did not capture the chunk trace")
        return prefill_mode

    def chunk_prefill_rows(self, count: int) -> int:
        """Rows the chunk trace would run for ``count`` tokens from the committed position (after the alignment steps)."""

        return count - alignment_steps(len(self.committed), count) if count > 0 else 0

    def _prefill_chunked(self, token_ids: Sequence[int]) -> Qwen38PrefillResult:
        """The chunk driver over ``token_ids`` from the committed position; its alignment steps are this session's
        forced steps (with the prefill event cadence).  Runs outside the loop guard: the seed and the hand-off
        synchronize and allocate.  The committed sequence and the n-gram context follow the device."""

        forced = 0

        def forced_step(token_id: int, context: tuple[int, int] | None) -> tuple[int, int] | None:
            nonlocal forced
            if context != self.ple_context:
                raise Qwen38ChatChainError(f"chunk prefill context {context} vs the session's {self.ple_context}")
            self._forced_step(token_id)
            forced += 1
            if forced % PREFILL_EVENT_INTERVAL == 0:
                self.chain.event_synchronize(self.chain.record_event())
            return self.ple_context

        result = self.chain.chunk_prefill(
            token_ids, start_position=len(self.committed), ple_context=self.ple_context, forced_step=forced_step
        )
        if result.timing.alignment_steps != forced:
            raise Qwen38ChatChainError(f"chunk prefill forced {forced} alignment steps, reported {result.timing}")
        self.committed.extend(token_ids[forced:])
        self.ple_context = result.ple_context
        if result.position != len(self.committed):
            raise Qwen38ChatChainError(
                f"chunk prefill ended at position {result.position} vs committed input sequence length "
                f"{len(self.committed)}"
            )
        return result

    def _generate(
        self,
        max_new_tokens: int,
        stop_ids: Sequence[int],
        think_budget: int | None,
        should_stop: Callable[[], str | None] | None,
    ) -> Iterator[tuple[int | None, str | None]]:
        """The runner's evented A-G loop; yields (x_t, finish) with finish set on the last item (x_t None: a hook stop).

        EOS is consumed (its HEAD is already queued when the host learns it), so
        every layer and the counter end at the same position and the next turn
        continues from ``<|im_end|>``.  On ``max_tokens`` the last token is read
        by the blocking read and not consumed.  A token at or above the tokenizer
        size is an LM-head padding row: the step is finished and the request
        ends with ``error``.  Before a step the hook may end the request with its
        own reason (the row keeps the unconsumed token), and a thinking budget
        forces ``</think>`` as a teacher-forced step once that many reasoning
        tokens were produced without the model closing the block.
        """

        produced = 0
        reasoning_tokens = 0
        thinking_open = think_budget is not None
        while True:
            reason = None if should_stop is None else should_stop()
            if reason is not None:
                yield None, reason
                return
            if thinking_open and reasoning_tokens >= think_budget:
                self._forced_step(THINK_END_ID)
                thinking_open = False
                produced += 1
                yield THINK_END_ID, "length" if produced == max_new_tokens else None
                if produced == max_new_tokens:
                    return
                continue
            if produced + 1 == max_new_tokens:
                token_id = self.chain.read_token_row()
                self._require_vocabulary(token_id)
                yield token_id, "length"
                return
            residue = len(self.committed) % RESIDUE_CLASSES
            pending = self.chain.read_token_row_nonblocking()
            event = self.chain.record_event()
            self.chain.execute_head(residue)
            self.chain.event_synchronize(event)
            token_id = self.chain.pending_value(pending)
            self._require_vocabulary(token_id)
            produced += 1
            if thinking_open:
                thinking_open = token_id != THINK_END_ID
                reasoning_tokens += 1
            self.ple_context = self.chain.refresh_ple_row(token_id, self.ple_context)
            self.chain.execute_tail(residue)
            self.committed.append(token_id)
            if token_id >= TOKENIZER_SIZE or token_id in stop_ids:
                self.chain.read_token_row()  # completes TAIL(t); x_{t+1} is discarded
                yield token_id, "error" if token_id >= TOKENIZER_SIZE else "stop"
                return
            yield token_id, None

    @staticmethod
    def _require_vocabulary(token_id: int) -> None:
        if not 0 <= token_id < VOCAB_SIZE:
            raise Qwen38ChatChainError(f"token row holds {token_id}, outside the vocabulary [0, {VOCAB_SIZE})")

    # -- one request -------------------------------------------------------------------------

    def reset(self) -> None:
        """Position 0 and position-zero state at the captured addresses; the committed sequence is dropped."""

        self.chain.reset_and_seed(SEED_TOKEN_ID)
        self.committed = []
        self.ple_context = None
        self.last_finish = None
        self.row_unconsumed = False

    def complete(
        self,
        token_ids: Sequence[int],
        max_tokens: int | None,
        *,
        stop_ids: Sequence[int] = EOS_TOKEN_IDS,
        on_token: Callable[[int], None] | None = None,
        prefill_mode: str | None = None,
        think_budget: int | None = None,
        should_stop: Callable[[], str | None] | None = None,
        sampling: sampling_step.Qwen38SamplingRequest | None = None,
    ) -> Qwen38ChatCompletion:
        """Prefill what the device does not already hold, then generate up to ``max_tokens`` tokens (the remaining
        context when ``None``; ``require_budget``).

        Any failure inside the device section leaves the chain's queue and the
        model owner in an unknown state: the session is poisoned and must not
        serve again.  ``on_token`` raising ``OSError`` (the client went away) is
        not such a failure: it is called between steps, so generation stops with
        ``disconnected`` and, as after ``length``, the next token stays in the row.
        ``prefill_mode`` overrides the session's mode for this request.
        ``should_stop`` is polled between steps (and at the teacher-forced
        prefill's event syncs; the chunk driver runs to its hand-off first) and
        ends the request with the reason it returns, the row unconsumed;
        ``think_budget`` forces ``</think>`` after that many reasoning tokens
        (the caller passes it only when the prompt left the think block open).
        ``sampling`` (a request with ``temperature > 0``) runs the sampled loop over
        the chain's candidate row instead of the greedy loop; ``None`` is greedy.
        """

        token_ids = list(token_ids)
        if not token_ids or any(type(value) is not int or not 0 <= value < VOCAB_SIZE for value in token_ids):
            raise Qwen38ChatRequestError("prompt token ids must be a nonempty list of vocabulary ids")
        max_tokens = self.require_budget(len(token_ids), max_tokens)
        mode = self.resolve_prefill_mode(prefill_mode)
        if sampling is not None and self.sampling is None:
            raise Qwen38ChatRequestError("sampling is unavailable: this chain captured no candidate row (greedy only)")
        if self.poisoned:
            raise Qwen38ChatChainError("session is poisoned by an earlier device failure")
        started_ns = self.clock_ns()
        common = 0
        while common < len(self.committed) and common < len(token_ids) and self.committed[common] == token_ids[common]:
            common += 1
        # The device continues an exact repeat only while the unconsumed next token
        # is still in the row; a partial match cannot be rewound.
        extends = common == len(self.committed) and (common < len(token_ids) or self.row_unconsumed)
        try:
            if not extends:
                self.reset()
                common = 0
            suffix = token_ids[common:]
            chunked: Qwen38PrefillResult | None = None
            # All but the last prompt token through the chunk trace when enough rows remain after the alignment
            # steps; the last one is the first decode replay and is always teacher-forced inside the guard.
            if mode == "chunked" and self.chunk_prefill_rows(len(suffix) - 1) >= CHUNK_PREFILL_MIN_ROWS:
                chunked = self._prefill_chunked(suffix[:-1])
                suffix = suffix[-1:]
            with self.chain.loop_guard():
                generated: list[int] = []
                finish = self._prefill(suffix, should_stop)
                hook_stopped = finish is not None
                prefill_done_ns = self.clock_ns()
                first_ns = last_ns = prefill_done_ns
                if finish is None:
                    finish = "length"
                    steps = (
                        self._generate(max_tokens, stop_ids, think_budget, should_stop)
                        if sampling is None
                        else sampling_step.generate_sampled(
                            self,
                            sampling,
                            max_tokens,
                            stop_ids=stop_ids,
                            tokenizer_size=TOKENIZER_SIZE,
                            think_budget=think_budget,
                            should_stop=should_stop,
                            forced_step=self._forced_step,
                            clock_ns=self.clock_ns,
                        )
                    )
                    for token_id, finish_reason in steps:
                        if token_id is None:
                            finish, hook_stopped = finish_reason, True
                            break
                        last_ns = self.clock_ns()
                        if not generated:
                            first_ns = last_ns
                        generated.append(token_id)
                        if finish_reason is not None:
                            finish = finish_reason
                        if on_token is not None and finish_reason != "error" and token_id not in stop_ids:
                            try:
                                on_token(token_id)
                            except OSError:
                                if finish_reason is None:
                                    finish = "disconnected"
                                break
            self.last_finish = finish
            self.row_unconsumed = hook_stopped or finish in ("length", "disconnected")
            position = self.chain.position()
            if position != len(self.committed):
                raise Qwen38ChatChainError(
                    f"device position {position} vs committed input sequence length {len(self.committed)}"
                )
        except BaseException:
            self.poisoned = True
            raise
        decode_seconds = (last_ns - first_ns) / 1e9
        tokens_per_second = (
            (len(generated) - 1) / decode_seconds if len(generated) >= 2 and decode_seconds > 0 else None
        )
        self.requests_served += 1
        self.last_tokens_per_second = tokens_per_second
        prefill_tokens = len(token_ids) - common
        return Qwen38ChatCompletion(
            token_ids=generated,
            finish_reason=finish,
            prompt_tokens=len(token_ids),
            prefix_reused=common,
            reset=not extends,
            prefill_tokens=prefill_tokens,
            prefill_seconds=(prefill_done_ns - started_ns) / 1e9,
            ttft_seconds=(first_ns - started_ns) / 1e9,
            decode_seconds=decode_seconds,
            tokens_per_second=tokens_per_second,
            position=position,
            prefill_mode="teacher_forced" if chunked is None else "chunked",
            prefill_forced_tokens=prefill_tokens if chunked is None else chunked.timing.alignment_steps + 1,
            prefill_chunks=0 if chunked is None else chunked.timing.chunks,
            prefill_tail_rows=0 if chunked is None else chunked.timing.tail_rows,
            prefill_handoff_ms=0.0 if chunked is None else chunked.timing.handoff_ms,
        )


class Qwen38TextStream:
    """Incremental detokenizer: decode the unemitted tail, hold it while it ends in U+FFFD (a split UTF-8 sequence)."""

    def __init__(self, decode: Callable[[list[int]], str]) -> None:
        self.decode = decode
        self.pending: list[int] = []

    def push(self, token_id: int) -> str:
        self.pending.append(token_id)
        text = self.decode(self.pending)
        if text.endswith("�"):
            return ""
        self.pending = []
        return text


def template_decoder(template: Any) -> Callable[[list[int]], str]:
    return lambda ids: template.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


# -- hardware ----------------------------------------------------------------------------------


def open_partition_b_mesh(marker: Marker, hardware_profile: ResidentHardwareProfile) -> tuple[Any, dict]:
    """The runner's mesh open for one lane: the profile's route derivation, FABRIC_1D, one 1D four-device mesh as
    the logical 1x4.

    Same calls and checks as ``run()`` of the timing runner on an eight-chip
    host (its partition-B values are the profile's defaults, hence the
    name; a partition-A profile brings its own nodes, route and locks; both
    are a 4x1 line reshaped to 1x4, ``derive_canonical_line_route``).  A ring
    profile (the QuietBox) opens the 1x4 its mesh graph descriptor reports in
    ``derive_ring_walk_route`` order.  Every check before the fabric enable
    raises with the fabric untouched; a failure after it disables the fabric
    before re-raising, so the caller owns the fabric only once this returns.
    """

    import yaml

    lane = f"{hardware_profile.host} partition-{hardware_profile.partition.upper()}"
    discovered = (int(ttnn.GetNumAvailableDevices()), int(ttnn.get_num_pcie_devices()), int(ttnn.get_num_devices()))
    if discovered != (4, 4, 4):
        raise Qwen38ChatChainError(f"{lane} visibility actual={discovered} expected=(4, 4, 4)")
    descriptor = ttnn._ttnn.multi_device.SystemMeshDescriptor()
    physical_shape = tuple(int(value) for value in descriptor.local_shape())
    if physical_shape != hardware_profile.system_mesh_local_shape or not bool(descriptor.all_local()):
        raise Qwen38ChatChainError(
            f"{lane} physical mesh is not one local {hardware_profile.system_mesh_local_shape} line: "
            f"actual={physical_shape}, all_local={bool(descriptor.all_local())}"
        )
    descriptor_path = Path(ttnn.cluster.serialize_cluster_descriptor()).resolve(strict=True)
    document = yaml.safe_load(descriptor_path.read_bytes())
    chips_with_mmio = physical_route.parse_chips_with_mmio(
        document, expected_device_nodes=set(hardware_profile.device_nodes)
    )
    derive_route = {
        "line": physical_route.derive_canonical_line_route,
        "ring": physical_route.derive_ring_walk_route,
    }[hardware_profile.ethernet_graph]
    route = derive_route(document, chips_with_mmio)
    route_nodes = physical_route.route_device_nodes(route, chips_with_mmio)
    if route != hardware_profile.route or route_nodes != hardware_profile.route_nodes:
        raise Qwen38ChatChainError(
            f"{lane} route actual=(logical={route}, nodes={route_nodes}) "
            f"expected=(logical={hardware_profile.route}, nodes={hardware_profile.route_nodes})"
        )
    lock_proof = hardware_profiles.verify_inherited_locks(hardware_profile)
    marker("before-fabric-enable")
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = None
    try:
        marker("before-mesh-open")
        mesh = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*physical_shape),
            physical_device_ids=list(route),
            l1_small_size=24576,
            trace_region_size=0,
        )
        if tuple(int(value) for value in mesh.shape) != MESH_SHAPE:
            mesh.reshape(ttnn.MeshShape(*MESH_SHAPE))
        Qwen38MeshContract(route).validate_mesh(mesh)
        live_mapping = hardware_profiles.live_mapping(mesh, chips_with_mmio, hardware_profile)
        mesh.enable_program_cache()
    except BaseException:
        if mesh is not None:  # a failed identity check after the open: release the cards before the fabric
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        raise
    return mesh, {
        "partition": hardware_profile.partition,
        "device_nodes": list(hardware_profile.device_nodes),
        "physical_open_shape": list(physical_shape),
        "logical_mesh_shape": list(MESH_SHAPE),
        "canonical_logical_route": list(route),
        "canonical_device_node_route": list(route_nodes),
        "route_derivation": derive_route.__name__,
        "cluster_descriptor": str(descriptor_path),
        "fabric_config": "FABRIC_1D",
        "collective_topology": "Linear",
        "live_mapping": live_mapping,
        "preopen_runner_lock_proof": lock_proof,
    }


@dataclass
class Qwen38TracedChain:
    """The runner's single-trace chain kept open for serving: 8 decode traces, the persistent token row, the PLE
    row, (``chunked_prefill``) the chunk state with its one chunk trace captured after the decode traces, and
    (``sampling``) the candidate-row epilogue with its readback row."""

    construction: Qwen38LiveDecodeConstruction
    built_target: Any
    state: Any
    prepared: Any
    token_row_io: Any
    ple_row_mapper: Any
    head_trace_ids: list[int]
    tail_trace_ids: list[int]
    captures: list[Any]
    trace_candidates: list[Any]
    trace_token_rows: list[Any]
    capture_ms: float
    program_cache_entries: int
    open_seconds: float
    misses_forbidden: bool = True
    chunk_state: Any = None
    chunk_trace_id: int | None = None
    chunk_capture_ms: float = 0.0
    # The GDN state re-anchor the chunk trace was captured with (every chunk replay commits through it).
    chunk_gdn_step_anchor: bool = False
    # The allocation tracker verified every trace once after the captures; per-prefill re-verification (140-290 ms
    # in the full-model gate) is a diagnostic switch.
    verify_each_prefill: bool = False
    sampling: sampling_step.Qwen38SamplingChainExtension | None = None
    closed: bool = field(default=False, repr=False)

    @property
    def mesh(self) -> Any:
        return self.construction.builder.mesh_device

    @property
    def allocated_context(self) -> int:
        """The build's allocated context: the KV caches, RoPE tables and position constants are sized by it."""

        return self.built_target.model.allocated_context

    # -- the per-step primitives the session calls (the runner's loop, one call each) --------

    def write_token_row(self, token_id: int) -> None:
        ttnn.copy_host_to_device_tensor(resident_decode.device_token_row(self.mesh, token_id), self.token_row_io)

    def execute_head(self, residue: int) -> None:
        ttnn._ttnn_execute_trace(self.mesh, self.head_trace_ids[residue], cq_id=0, blocking=False)

    def execute_tail(self, residue: int) -> None:
        ttnn._ttnn_execute_trace(self.mesh, self.tail_trace_ids[residue], cq_id=0, blocking=False)

    def refresh_ple_row(self, token_id: int, context: tuple[int, int] | None) -> tuple[int, int]:
        """Scalar n-gram hash of (x_t, context), 16 PLE rows by pread, 5,120 B ROW_MAJOR write behind HEAD(t)."""

        ple_owner = self.built_target.model.layers[1].ple
        payload, next_context = ple_owner.resident_lookup.lookup_token(token_id, context)
        host_row, _row = resident_decode.ple_host_row(self.ple_row_mapper, payload)
        ttnn.copy_host_to_device_tensor(host_row, self.prepared.ple.embedding_sharded)
        return next_context

    def read_token_row_nonblocking(self) -> Any:
        return ttnn.from_device(ttnn.get_device_tensors(self.token_row_io)[0], blocking=False)

    def record_event(self) -> Any:
        return ttnn.record_event(self.mesh, cq_id=0)

    @staticmethod
    def event_synchronize(event: Any) -> None:
        ttnn.event_synchronize(event)

    @staticmethod
    def pending_value(pending: Any) -> int:
        return resident_decode.token_row_host_value(ttnn.to_torch(pending))

    def read_token_row(self) -> int:
        return resident_decode.token_row_value(self.token_row_io)

    def position(self) -> int:
        return self.state.position.read()

    def loop_guard(self):
        """The runner's replay-loop guard: only the loop's own host I/O stays callable, a synchronize fails."""

        return resident_decode.forbid_trace_body_host_io_and_sync(
            phase="chat request device loop", allowed=resident_decode.REPLAY_LOOP_HOST_IO_CALLS
        )

    def reset_and_seed(self, token_id: int) -> None:
        """In-place state reset to position 0 and the seed token in the persistent row (the runner's prologue)."""

        self.built_target.model.reset_generic_state_inplace(self.state)
        self.write_token_row(token_id)
        ttnn.synchronize_device(self.mesh)
        position = self.state.position.read()
        if position != 0:
            raise Qwen38ChatChainError(f"reset left the device position at {position}, expected 0")
        resident_decode.require_token_row_holds(
            self.token_row_io, resident_decode.host_token_row(token_id), label="reset seed token row"
        )

    def chunk_prefill(
        self,
        token_ids: Sequence[int],
        *,
        start_position: int,
        ple_context: tuple[int, int] | None,
        forced_step: Callable[[int, tuple[int, int] | None], tuple[int, int] | None],
    ) -> Qwen38PrefillResult:
        """The chunk driver on this chain (the full-model gate's prefill path): alignment steps through
        ``forced_step``, the seed, chunk replays, the padded tail, the hand-off.  Not under the loop guard."""

        if self.chunk_trace_id is None:
            raise Qwen38ChatChainError("the chain was opened without the chunk trace")
        return Qwen38ChunkPrefill(
            self.built_target.model,
            self.mesh,
            self.state,
            self.chunk_state,
            self.chunk_trace_id,
            forced_step=forced_step,
            verify_allocations=self.verify_each_prefill,
            gdn_step_anchor=self.chunk_gdn_step_anchor,
        ).run(token_ids, start_position=start_position, ple_context=ple_context)

    # -- open / close ------------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        construction: Qwen38LiveDecodeConstruction,
        *,
        marker: Marker,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        chunked_prefill: bool = True,
        sampling: bool = False,
        warm_hook: Callable[[Qwen38TracedChain], None] | None = None,
        chunk_gdn_step_anchor: bool = False,
    ) -> Qwen38TracedChain:
        """Target build, generic state (+ chunk state), warm pass (+ one eager chunk and both hand-off forms), miss
        guard, 8 decode captures (+ the chunk capture): the runner's chain prologue and the full-model gate's order.

        The chunk state is allocated before any capture (the decode traces bake every address in) and the chunk
        trace is captured after the decode traces on the same generic state, with ``chunk_gdn_step_anchor`` (the
        GDN state re-anchor) baked into the warm chunk and the capture.  ``sampling`` (off by default: the
        greedy loop keeps its measured period) adds the candidate-row epilogue to every TAIL (its constants are
        allocated before the warm pass, its programs compile in the warm pass, its row is checked there against
        torch.topk of the eager full gather).  ``warm_hook`` runs after the warm pass, misses still allowed, on
        the open chain: a caller with an eager path of its own (the long-context chain's hidden windows) compiles
        its programs there.
        """

        if type(chunk_gdn_step_anchor) is not bool:
            raise ValueError(f"chunk_gdn_step_anchor must be a bool, got {chunk_gdn_step_anchor!r}")

        started_ns = clock_ns()
        runtime_surface = resident_decode.b5b_runtime_surface()
        if runtime_surface["nonblocking_read"] != "ttnn.from_device(local, blocking=False)":
            raise Qwen38ChatChainError(
                f"the binary offers {runtime_surface['nonblocking_read']!r}, expected ttnn.from_device(blocking=False)"
            )
        if not ttnn.TRACE_ALLOC_TRACKING:
            raise Qwen38ChatChainError("TT_METAL_TRACE_ALLOC_TRACKING=1 is required for the trace chain")
        builder = construction.builder
        mesh = builder.mesh_device
        set_misses_allowed = getattr(mesh, "set_program_cache_misses_allowed", None)
        if not callable(set_misses_allowed):
            raise Qwen38ChatChainError("MeshDevice.set_program_cache_misses_allowed is required")
        if RESIDUE_CLASSES != gdn_module.CONV_KERNEL_SIZE:
            raise Qwen38ChatChainError(
                f"residue classes {RESIDUE_CLASSES} vs GDN conv ring length {gdn_module.CONV_KERNEL_SIZE}"
            )

        def synchronize() -> None:
            ttnn.synchronize_device(mesh)

        marker("before-chat-target-build")
        built_target = builder.build_target()
        marker("after-chat-target-build")
        synchronize()
        resident_owner = builder.expert_streamer
        if len(built_target.components.layers) != 48 or built_target.components.expert_streamer is not resident_owner:
            raise Qwen38ChatChainError(f"target graph has {len(built_target.components.layers)} layers, expected 48")
        if resident_owner.cache_load_success_count != resident_decode.EXPECTED_CACHE_LOADS:
            raise Qwen38ChatChainError(
                f"resident expert pairs loaded {resident_owner.cache_load_success_count}, "
                f"expected {resident_decode.EXPECTED_CACHE_LOADS}"
            )
        model = built_target.model
        lm_head = model.model_io.lm_head
        state = model.allocate_generic_state()
        chunk_state = None
        if chunked_prefill:
            model.reset_generic_state_inplace(state)
            chunk_state = model.allocate_chunk_state(state)
            model.reset_chunk_state_inplace(state, chunk_state)
        token_row_io = model.model_io.embedding.upload_token_row(SEED_TOKEN_ID)
        prepared = model.prepare_generic_decode_inputs(SEED_TOKEN_ID, state, device_token=token_row_io)
        ple_row_mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=MESH_SHAPE, dims=(None, 3))
        synchronize()
        chain = cls(
            construction=construction,
            built_target=built_target,
            state=state,
            prepared=prepared,
            token_row_io=token_row_io,
            ple_row_mapper=ple_row_mapper,
            head_trace_ids=[],
            tail_trace_ids=[],
            captures=[],
            trace_candidates=[],
            trace_token_rows=[],
            capture_ms=0.0,
            program_cache_entries=0,
            open_seconds=0.0,
            misses_forbidden=False,
            chunk_state=chunk_state,
            sampling=sampling_step.Qwen38SamplingChainExtension(lm_head, mesh) if sampling else None,
            chunk_gdn_step_anchor=chunk_gdn_step_anchor,
        )

        # Warm pass, eager, misses allowed: every kernel of the generic body
        # compiles here.  Kept gates (actual vs expected): position counter,
        # resident PLE lookup vs the host oracle, device greedy resolve vs CPU,
        # the candidate row vs torch.topk of the full gather.
        marker("before-chat-warm-pass")
        ple_context: tuple[int, int] | None = None
        for position in range(resident_decode.SINGLE_TRACE_WARM_POSITIONS):
            warm_token_id = resident_decode.SEQUENTIAL_TRACE_WARM_EMBEDDING_TOKEN_IDS[position % TP_SIZE]
            chain.write_token_row(warm_token_id)
            synchronize()
            actual = state.position.read()
            if actual != position:
                raise Qwen38ChatChainError(f"warm position counter {actual} vs expected {position}")
            oracle_embedding, oracle_context = model.layers[1].ple.host_embedding.lookup(
                torch.tensor([[warm_token_id]], dtype=torch.long),
                None if ple_context is None else torch.tensor([ple_context], dtype=torch.long),
            )
            payload, ple_context = model.layers[1].ple.resident_lookup.lookup_token(warm_token_id, ple_context)
            host_row, row = resident_decode.ple_host_row(ple_row_mapper, payload)
            ttnn.copy_host_to_device_tensor(host_row, prepared.ple.embedding_sharded)
            if (
                not torch.equal(
                    row.reshape(-1).view(torch.int16), oracle_embedding.reshape(-1).contiguous().view(torch.int16)
                )
                or list(ple_context) != oracle_context.reshape(-1).tolist()
            ):
                raise Qwen38ChatChainError(f"warm position {position} resident PLE lookup differs from the host oracle")
            output = model.forward_decode_generic(prepared, state)
            if output.logits is None:
                raise Qwen38ChatChainError(f"warm position {position} returned no logits")
            candidates = lm_head.greedy_candidates(output.logits)
            resolved_row = lm_head.resolve_greedy_on_device(candidates)
            synchronize()
            actual = state.position.read()
            if actual != position + 1:
                raise Qwen38ChatChainError(f"warm position counter after step {actual} vs expected {position + 1}")
            resolved = resident_decode.require_token_row_resolves(
                resolved_row, candidates, lm_head, label=f"warm position {position} device greedy resolve"
            )
            ttnn.copy(resolved_row, token_row_io)  # the body's last op: warm its program
            synchronize()
            resident_decode.require_token_row_holds(
                token_row_io, resident_decode.host_token_row(resolved), label=f"warm position {position} row copy"
            )
            if chain.sampling is not None:
                chain.sampling.warm(output.logits, label=f"warm position {position} candidate row")
            ttnn.deallocate(candidates.local_indices)
            ttnn.deallocate(candidates.local_values)
            ttnn.deallocate(resolved_row)
            output.release_tensors()
        synchronize()
        marker("after-chat-warm-pass")
        if warm_hook is not None:
            marker("before-chat-warm-hook")
            warm_hook(chain)
            synchronize()
            marker("after-chat-warm-hook")

        if chunked_prefill:
            # One eager chunk from the reset state (the chunk body's programs compile here) and both hand-off
            # forms (a closed block fills the staging tile; an open block copies it and selects a non-empty ring).
            marker("before-chat-chunk-warm-pass")
            model.reset_generic_state_inplace(state)
            model.reset_chunk_state_inplace(state, chunk_state)
            model.write_chunk_inputs(chunk_state, list(WARM_CHUNK_TOKEN_IDS), ple_context=None)
            synchronize()
            model.forward_prefill_chunk_generic(chunk_state, state, gdn_step_anchor=chunk_gdn_step_anchor)
            synchronize()
            actual = state.position.read()
            if actual != CHUNK_ROWS:
                raise Qwen38ChatChainError(f"warm chunk position counter {actual} vs expected {CHUNK_ROWS}")
            model.finish_prefill(state, chunk_state, CHUNK_ROWS)
            model.finish_prefill(state, chunk_state, CHUNK_ROWS - 1)
            synchronize()
            actual = state.position.read()
            if actual != CHUNK_ROWS - 1:
                raise Qwen38ChatChainError(f"warm hand-off position counter {actual} vs expected {CHUNK_ROWS - 1}")
            marker("after-chat-chunk-warm-pass")

        # Pre-capture reset (the in-place reset programs compile here), the
        # allocation tracker's acknowledgements of every host- or trace-written
        # buffer, then the miss guard closes.
        chain.reset_and_seed(SEED_TOKEN_ID)
        for tensor in (prepared.ple.embedding_sharded, token_row_io, state.position.scalar):
            ttnn.mark_corruptible(tensor)
        if chunked_prefill:
            model.reset_chunk_state_inplace(state, chunk_state)
            for tensor in (chunk_state.token_row, chunk_state.ple_rows.embedding_rows, chunk_state.accepted):
                ttnn.mark_corruptible(tensor)
        if chain.sampling is not None:
            chain.sampling.mark_corruptible()
        chain.program_cache_entries = resident_decode.program_cache_count(mesh)
        if chain.program_cache_entries <= 0:
            raise Qwen38ChatChainError("program cache is empty after the warm pass")
        set_misses_allowed(False)
        chain.misses_forbidden = True

        def conv_phases() -> dict[int, int]:
            return {
                layer.layer_index: layer_state.attention.conv_phase
                for layer, layer_state in zip(model.layers, state.layers, strict=True)
                if isinstance(layer.attention, gdn_module.Qwen38TTNNGDN)
            }

        def require_ring_phases(label: str, residue: int) -> None:
            """HEAD_r advances the HEAD layers' ring phase, TAIL_r the rest: a gate around each capture."""

            marker(f"chat-capture-{label}-residue-{residue}")
            next_phase = (residue + 1) % RESIDUE_CLASSES
            head_phase = residue if label == "before-head-capture" else next_phase
            tail_phase = next_phase if label == "after-tail-capture" else residue
            actual = conv_phases()
            expected = {index: head_phase if index < GENERIC_HEAD_LAYERS else tail_phase for index in actual}
            if actual != expected:
                raise Qwen38ChatChainError(
                    f"GDN ring phases {label} residue {residue}: actual {actual} vs expected {expected}"
                )

        def capture_epilogue(trace_output: Any) -> tuple[Any, Any]:
            """TAIL's last ops: local candidates, device greedy resolve, copy into the persistent token row; a
            sampling chain's epilogue runs the same three and then the candidate row."""

            if chain.sampling is not None:
                return chain.sampling.capture_epilogue(trace_output, token_row_io)
            if trace_output.logits is None:
                raise Qwen38ChatChainError("TAIL capture returned no logits")
            candidates = lm_head.greedy_candidates(trace_output.logits)
            trace_token_row = lm_head.resolve_greedy_on_device(candidates)
            ttnn.copy(trace_token_row, token_row_io)
            return candidates, trace_token_row

        marker("before-chat-captures")
        capture_ns = 0
        for phase in range(RESIDUE_CLASSES):
            capture = model.capture_decode_generic(
                prepared,
                state,
                residue=phase,
                split=True,
                guard=lambda label: resident_decode.forbid_trace_body_host_io_and_sync(phase=f"chat {label}"),
                epilogue=capture_epilogue,
                phase_observer=lambda label, residue=phase: require_ring_phases(label, residue),
                regime=resident_decode.SINGLE_TRACE_INDEXER_REGIME,
                cq_id=0,
                clock_ns=clock_ns,
            )
            if (
                capture.parts != resident_decode.SINGLE_TRACE_TRACE_PARTS
                or capture.head is None
                or not capture.head.active
            ):
                raise Qwen38ChatChainError(
                    f"capture residue {phase} parts {capture.parts} did not retain a HEAD handoff"
                )
            if capture.guard_attempts:
                raise Qwen38ChatChainError(
                    f"capture residue {phase} attempted host I/O in-body: {capture.guard_attempts}"
                )
            chain.captures.append(capture)
            chain.head_trace_ids.append(
                capture.trace_ids[Qwen38TTNNGenericTraceKey("head", phase, resident_decode.SINGLE_TRACE_INDEXER_REGIME)]
            )
            chain.tail_trace_ids.append(
                capture.trace_ids[Qwen38TTNNGenericTraceKey("tail", phase, resident_decode.SINGLE_TRACE_INDEXER_REGIME)]
            )
            capture_ns += sum(capture.capture_ns.values())
            candidates, trace_token_row = capture.epilogue
            chain.trace_candidates.append(candidates)
            chain.trace_token_rows.append(trace_token_row)
            ttnn.mark_corruptible(candidates.local_indices)
            ttnn.mark_corruptible(candidates.local_values)
            if chain.sampling is not None:
                chain.sampling.mark_trace_rows_corruptible()
            synchronize()
        marker("after-chat-captures")
        chain.capture_ms = capture_ns / 1e6
        if chunked_prefill:
            # The chunk trace after the 8 decode traces, on the same generic state; capture records without
            # executing, so the ring phases and the position stay at 0.
            marker("before-chat-chunk-capture")
            chunk_capture_started_ns = clock_ns()
            chain.chunk_trace_id = model.capture_prefill_chunk(
                chunk_state,
                state,
                guard=lambda label: resident_decode.forbid_trace_body_host_io_and_sync(phase=f"chat {label}"),
                cq_id=0,
                gdn_step_anchor=chunk_gdn_step_anchor,
            )
            chain.chunk_capture_ms = (clock_ns() - chunk_capture_started_ns) / 1e6
            synchronize()
            marker("after-chat-chunk-capture")
        phases, position = conv_phases(), state.position.read()
        if set(phases.values()) != {0} or position != 0:
            raise Qwen38ChatChainError(f"after the captures ring phases {phases} and position {position}, expected 0")
        after_capture = resident_decode.program_cache_count(mesh)
        if after_capture != chain.program_cache_entries:
            raise Qwen38ChatChainError(
                f"program cache {after_capture} entries after capture vs {chain.program_cache_entries} before"
            )
        for trace_id in chain.trace_ids():
            UnsafeAllocationTracker(mesh).verify_before_replay(trace_id)
        chain.reset_and_seed(SEED_TOKEN_ID)
        chain.open_seconds = (clock_ns() - started_ns) / 1e9
        return chain

    def trace_ids(self) -> list[int]:
        """Every captured trace: 4 HEAD, 4 TAIL, then the chunk trace when captured."""

        return (
            self.head_trace_ids + self.tail_trace_ids + ([] if self.chunk_trace_id is None else [self.chunk_trace_id])
        )

    def close(self) -> None:
        """The runner's release order; skipped by the caller when a request failed mid-loop."""

        if self.closed:
            return
        mesh = self.mesh
        ttnn.synchronize_device(mesh)
        if self.misses_forbidden:
            mesh.set_program_cache_misses_allowed(True)
            self.misses_forbidden = False
        for trace_id in self.trace_ids():
            ttnn.release_trace(mesh, trace_id)
        self.head_trace_ids.clear()
        self.tail_trace_ids.clear()
        self.chunk_trace_id = None
        for candidates in self.trace_candidates:
            ttnn.deallocate(candidates.local_indices)
            ttnn.deallocate(candidates.local_values)
        self.trace_candidates.clear()
        for trace_token_row in self.trace_token_rows:
            ttnn.deallocate(trace_token_row)
        self.trace_token_rows.clear()
        if self.sampling is not None:
            self.sampling.release()
        for capture in self.captures:
            if capture.active:
                capture.release_tensors()
        self.captures.clear()
        if self.prepared.active:
            self.prepared.release()
        ttnn.deallocate(self.token_row_io)
        if self.chunk_state is not None:
            self.built_target.model.release_chunk_state(self.chunk_state)
            self.chunk_state = None
        self.built_target.model.release_generic_state(self.state)
        ttnn.synchronize_device(mesh)
        self.built_target.components.close_resident_experts()
        ttnn.synchronize_device(mesh)
        self.closed = True


def construct_chain(
    prepared: Any, mesh: Any, *, marker: Marker, chunked_prefill: bool = True, sampling: bool = False
) -> Qwen38TracedChain:
    """Live construction on the open mesh, then the chain prologue."""

    construction = construct_live_decode_diagnostic(
        prepared, mesh_device=mesh, collective_topology=ttnn.Topology.Linear, marker=marker
    )
    return Qwen38TracedChain.open(construction, marker=marker, chunked_prefill=chunked_prefill, sampling=sampling)
