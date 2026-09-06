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

The prompt-end snapshot (``REPLY-TAIL-SPLICE-DESIGN-20260906.md``): before the last prompt token's forced step
the chain copies the generic state's recurrent buffers (GDN recurrent states and ring slots, PLE slots, QSA staging
and raw-key rings; the caches are positional) into a resident snapshot and the session records the ids consumed so
far.  A later request whose render extends those ids but not the committed ones (a thinking conversation: the
template renders the past turn's think block as ``<think>\n\n</think>``, one token off the device's ``<think>\n``)
restores the snapshot and prefills only the rendered tail instead of resetting.

Sampling is an optional tail of the same chain (``tools/qwen38_sampling_step.py``):
TAIL's epilogue also writes a candidate row, and a request with ``temperature > 0``
runs the sampled loop (read the row after TAIL, sample on the host, write the token
into the row before HEAD).  Greedy requests take the loop above untouched.

MTP drafting is opt-in (``mtp=K``, K in 3 or 4; ``ttnn/mtp_v2.py``): the chain also
holds the verify / draft / commit traces and, in every TAIL and in the chunk body, the
MTP layer's rows, so the MTP layer follows the target through prefill.  A greedy
request on such a chain runs its prefill as above, reads the first token, switches
the device state into verify mode (``mtp_enter``) and generates with the pass loop:
draft replay, readback, PLE rows, commit, verify replay, readback, ``a + 1`` tokens
per pass streamed as they commit.  At the end of the request (or before a forced
token) the last pass's rows are committed as far as the request consumed them and
the 1-row buffers are rebuilt (``mtp_leave``), so 1-row, sampled and MTP requests
alternate on one server.  Sampled requests and ``prefill_mode`` ``teacher_forced``
requests take the loops above.

``Qwen38ChatSession`` speaks to the device only through a chain object with the
per-step primitives; ``Qwen38TracedChain`` is the hardware one, the no-device
test drives the session with a scripted chain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
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
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn.bf4 import (
    BF4_TILE_BYTES,
    BLACKHOLE_RING_SIZES,
    packed_bf4_bytes_per_device,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import (
    RESIDENT_CONTEXT_HEADROOM,
    RESIDENT_MAX_QSA_CACHE_CAPACITY,
    RESIDENT_QSA_CACHE_CAPACITIES,
    Qwen38ResidentContext,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    CHUNK_ROWS,
    LONG_CHUNK_ROWS,
    MESH_SHAPE,
    TP_SIZE,
    Qwen38MeshContract,
)
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
WARM_LONG_CHUNK_TOKEN_IDS = tuple(
    resident_decode.SEQUENTIAL_TRACE_WARM_EMBEDDING_TOKEN_IDS[index % TP_SIZE] for index in range(LONG_CHUNK_ROWS)
)
# The CPU acceptance study's template flags (mtp_acceptance_cpu_v2 at a97cb9e6b0): the 12 prompt records render
# identically only with these.  The records carry their own system turn inside their prompt ids; the session adds no
# system prompt of its own.
ENABLE_THINKING = False
PRESERVE_THINKING = True
REASONING_EFFORT = "low"
RESIDUE_CLASSES = resident_decode.SINGLE_TRACE_RESIDUE_CLASS_TRACES
SEED_TOKEN_ID = IM_START_ID
THINK_END_ID = protocol.THINK_END_ID
# MTP drafting: the draft counts a server may be opened with (the chain timing tool's measured arms), the GDN state
# re-anchor settings (off = every GDN layer commits through the chunk kernel; layer0 = layer 0's commits run the
# 1-row fp32 step recurrence over the committed rows), the bootstrap pass's placeholder drafts, the resident expert
# pairs with the MTP layer's.
MTP_DRAFTS = (3, 4)
MTP_GDN_ANCHORS = ("off", "layer0")
MTP_GDN_ANCHOR_LAYERS = {"off": (), "layer0": (0,)}
MTP_BOOTSTRAP_DRAFT_TOKEN = 0
MTP_EXPECTED_CACHE_LOADS = resident_decode.EXPECTED_CACHE_LOADS + 1
# The MTP chain's DRAM per bank beyond the 49th BF4 pair and the MTP layer's QSA state: the layer's non-expert weights,
# the verify / draft states, the step inputs, the chunk extension and the three traces (8.2 MB of states and traces
# measured at 32,768; the open checks the measured growth against this bound).  The free bytes per bank a resident build
# leaves after its captures without MTP (measured 2026-09-04, head 4149197252, 8 banks of 4,272,341,376 bytes; free and
# largest contiguous) are the admission table --mtp is refused against.  A resident BF4 payload is interleaved over
# the banks one 576-byte tile page at a time, and the resident loader needs 128 MB contiguous per bank.
MTP_CHAIN_BYTES_PER_BANK_UPPER_BOUND = 24 << 20
RESIDENT_DRAM_BANKS = 8
RESIDENT_MIN_CONTIGUOUS_BYTES_PER_BANK = 128 << 20
RESIDENT_FREE_BYTES_PER_BANK_AFTER_CAPTURES = {
    32768: (478_238_784, 477_584_640),
    65536: (423_336_000, 422_681_856),
    131072: (313_694_272, 313_040_128),
    262144: (94_378_048, 93_723_904),
}


def mtp_capacity_admission(allocated_context: int, *, ring_size: int = max(BLACKHOLE_RING_SIZES)) -> dict[str, Any]:
    """Whether the MTP chain fits beside the resident build at ``allocated_context``: the 49th BF4 pair (its two
    payloads interleaved over the banks, needing their contiguous room), the MTP layer's QSA state at that context and
    :data:`MTP_CHAIN_BYTES_PER_BANK_UPPER_BOUND`, against the free bytes per bank the build leaves after its captures
    (:data:`RESIDENT_FREE_BYTES_PER_BANK_AFTER_CAPTURES`).  Every byte count is in the record; ``fits`` decides."""

    free, largest = RESIDENT_FREE_BYTES_PER_BANK_AFTER_CAPTURES[
        Qwen38ResidentContext(allocated_context).allocated_context
    ]
    w01_bytes, w2_bytes = packed_bf4_bytes_per_device(ring_size=ring_size)
    w01_per_bank = -(-(w01_bytes // BF4_TILE_BYTES) // RESIDENT_DRAM_BANKS) * BF4_TILE_BYTES
    w2_per_bank = -(-(w2_bytes // BF4_TILE_BYTES) // RESIDENT_DRAM_BANKS) * BF4_TILE_BYTES
    qsa_state = -(-Qwen38ResidentContext(allocated_context).qsa_generic_state_bytes // RESIDENT_DRAM_BANKS)
    required = w01_per_bank + w2_per_bank + qsa_state + MTP_CHAIN_BYTES_PER_BANK_UPPER_BOUND
    contiguous = max(w01_per_bank, w2_per_bank, RESIDENT_MIN_CONTIGUOUS_BYTES_PER_BANK)
    return {
        "allocated_context": allocated_context,
        "ring_size": ring_size,
        "num_banks": RESIDENT_DRAM_BANKS,
        "free_bytes_per_bank_after_captures": free,
        "largest_contiguous_bytes_free_per_bank_after_captures": largest,
        "resident_pair_bytes_per_bank": w01_per_bank + w2_per_bank,
        "mtp_qsa_state_bytes_per_bank": qsa_state,
        "mtp_chain_bytes_per_bank_upper_bound": MTP_CHAIN_BYTES_PER_BANK_UPPER_BOUND,
        "required_free_bytes_per_bank": required,
        "required_largest_contiguous_bytes_per_bank": contiguous,
        "headroom_bytes_per_bank": free - required,
        "fits": free >= required and largest >= contiguous,
    }


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
    # MTP drafting when the request generated through the pass loop: {k, passes, accepted_drafts, tokens_per_pass,
    # anchor}; None for the 1-row and sampled loops.
    mtp: dict[str, Any] | None = None
    # The prompt-end snapshot was restored: prefix_reused is the snapshot's length, the rest of the prompt the tail.
    restored: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "prefix_reused": self.prefix_reused,
            "reset": self.reset,
            "prefix_restored": self.restored,
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
            "mtp": self.mtp,
        }


@dataclass(frozen=True)
class Qwen38PromptSnapshot:
    """The host record of the chain's prompt-end snapshot: the ids consumed when it was taken (the prompt without
    its last token) and the n-gram context after them."""

    ids: tuple[int, ...]
    ple_context: tuple[int, int] | None


@dataclass
class Qwen38MTPPassRun:
    """The pass loop's last pass until :meth:`Qwen38ChatSession._mtp_settle` commits it: the position it started at,
    the rows it fed (``[t_P, d_1 .. d_a]``), the tokens it emitted (``[d_1 .. d_a, t']``) and how many of them the
    request took."""

    position: int
    rows: list[int]
    emitted: list[int]
    yielded: int
    host_committed: bool = False  # the rows are in the session's committed list (the pass was consumed whole)


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
        self.mtp = getattr(chain, "mtp", None)  # the MTP drafting extension (verify / draft / commit traces), or None
        # A chain without the snapshot primitives (an older scripted chain) resets where a restore would apply.
        self.snapshot_available = callable(getattr(chain, "capture_prompt_snapshot", None))
        self.snapshot: Qwen38PromptSnapshot | None = None
        self.committed: list[int] = []
        self.ple_context: tuple[int, int] | None = None
        self.last_finish: str | None = None
        self.row_unconsumed = False  # the next token sits in the row (after length, disconnected, a hook stop)
        self.row_token: int | None = None  # a length finish's last token: read, delivered to the client, not committed
        self.requests_served = 0
        self.last_tokens_per_second: float | None = None
        self.poisoned = False
        self._mtp_run: Qwen38MTPPassRun | None = None  # the pass loop's last pass until complete() settles it

    # -- request rendering -------------------------------------------------------------------

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        enable_thinking: bool = ENABLE_THINKING,
        reasoning_effort: str = REASONING_EFFORT,
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> list[int]:
        """Prompt token ids of exactly ``messages`` (and the client's ``tools``) under the acceptance study's flags;
        no system turn is added when the client sends none.  The protocol module renders: bitwise the reference
        template."""

        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or not messages:
            raise Qwen38ChatRequestError("messages must be a nonempty list")
        if not all(isinstance(message, Mapping) for message in messages):
            raise Qwen38ChatRequestError("every message must be an object")
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

    def _forced_step(self, token_id: int, next_token_id: int | None = None) -> None:
        """A' write x_t into the token row (behind TAIL(t-1)), C HEAD(t), F PLE row for x_t, G TAIL(t).

        On an MTP chain the TAIL also runs the MTP layer's row at t, which consumes the token at t + 1:
        ``next_token_id`` when the host knows it (the next prompt token), else the step's own resolved argmax.
        """

        residue = len(self.committed) % RESIDUE_CLASSES
        self.chain.write_token_row(token_id)
        if self.mtp is not None:
            self.chain.write_mtp_next_token(next_token_id)
        self.chain.execute_head(residue)
        self.ple_context = self.chain.refresh_ple_row(token_id, self.ple_context)
        self.chain.execute_tail(residue)
        self.committed.append(token_id)

    def _prefill(
        self,
        token_ids: Sequence[int],
        should_stop: Callable[[], str | None] | None,
        before_last: Callable[[], None] | None = None,
    ) -> str | None:
        """Forced steps; at every event sync the hook may end the request (the row then holds an unconsumed token).
        ``before_last`` runs before the last token's step (the prompt-end snapshot)."""

        for index, token_id in enumerate(token_ids, start=1):
            if index == len(token_ids) and before_last is not None:
                before_last()
            self._forced_step(token_id, token_ids[index] if index < len(token_ids) else None)
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

    def _prefill_chunked(
        self, token_ids: Sequence[int], following_token: int, should_stop: Callable[[], str | None] | None
    ) -> Qwen38PrefillResult:
        """The chunk driver over ``token_ids`` from the committed position; its alignment steps are this session's
        forced steps (with the prefill event cadence).  Runs outside the loop guard: the seed and the hand-off
        synchronize and allocate.  The committed sequence and the n-gram context follow the device: a stop at one
        of the driver's event syncs (``result.stopped``) leaves the prompt committed up to the hand-off.
        ``following_token`` (the last prompt token, teacher-forced afterwards) is the MTP layer's token at the last
        prefilled position."""

        forced = 0
        start = len(self.committed)
        ahead = [*token_ids[1:], following_token]

        def forced_step(token_id: int, context: tuple[int, int] | None) -> tuple[int, int] | None:
            nonlocal forced
            if context != self.ple_context:
                raise Qwen38ChatChainError(f"chunk prefill context {context} vs the session's {self.ple_context}")
            self._forced_step(token_id, ahead[forced])
            forced += 1
            if forced % PREFILL_EVENT_INTERVAL == 0:
                self.chain.event_synchronize(self.chain.record_event())
            return self.ple_context

        result = self.chain.chunk_prefill(
            token_ids,
            start_position=start,
            ple_context=self.ple_context,
            forced_step=forced_step,
            following_token=following_token,
            should_stop=should_stop,
        )
        if result.timing.alignment_steps != forced:
            raise Qwen38ChatChainError(f"chunk prefill forced {forced} alignment steps, reported {result.timing}")
        self.committed.extend(token_ids[forced : result.position - start])
        self.ple_context = result.ple_context
        if result.position != len(self.committed) or (
            result.stopped is None and result.position != start + len(token_ids)
        ):
            raise Qwen38ChatChainError(
                f"chunk prefill ended at position {result.position} (stopped {result.stopped!r}) vs committed input "
                f"sequence length {len(self.committed)} of {start + len(token_ids)}"
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
                self.row_token = token_id
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

    # -- the MTP pass loop -------------------------------------------------------------------

    def _generate_mtp(
        self,
        max_new_tokens: int,
        stop_ids: Sequence[int],
        think_budget: int | None,
        should_stop: Callable[[], str | None] | None,
    ) -> Iterator[tuple[int | None, str | None]]:
        """The pass loop; yields (x_t, finish) like :meth:`_generate`, ``a + 1`` tokens per pass.

        Entry: the last prompt token's TAIL is enqueued (or the row holds an unconsumed token); the blocking
        read gives the model's next token, which the first pass consumes as its row 0.  Each pass feeds the rows
        ``[t_P, d_1 .. d_k]`` and commits ``a + 1`` of them on the device; the host learns ``[d_1 .. d_a, t']``,
        streams them and keeps the last one (``t'``, not yet fed) as the next pass's row 0.  The pass's device
        commit is deferred to the next pass (or to :meth:`_mtp_settle`, which commits as many rows as the request
        consumed and rebuilds the 1-row buffers).  A forced ``</think>`` needs the 1-row chain: the pass loop
        settles, forces the token, reads the model's next token and re-enters.  When the cache has no room for a
        pass the 1-row loop finishes the request.
        """

        pending = self.chain.read_token_row()  # completes the last TAIL: the model's next token, not consumed
        self._require_vocabulary(pending)
        pending_yielded = False  # a pass's t' was streamed with its pass; a token read from the row was not
        produced = 0
        reasoning_tokens = 0
        thinking_open = think_budget is not None
        entered = False

        def force_think_end(settle_finish: str) -> None:
            # The 1-row loop's forced </think>: the pass loop settles first ("budget": the tokens streamed so far
            # consumed; "length": a pending token the client saw is fed, one it never saw is dropped for the forced
            # one, as _generate drops the row's token).
            nonlocal entered
            if entered:
                self._mtp_settle(settle_finish)
                entered = False
            if settle_finish == "length" and pending_yielded:
                self._forced_step(pending)
            self._forced_step(THINK_END_ID)

        def finish_on_one_row_loop():
            # The 1-row loop takes the rest of the request; the pass loop settles and leaves the pending token
            # where that loop expects it: in the row when it was not streamed yet, consumed by a forced step when it was.
            nonlocal entered
            if entered:
                self._mtp_settle("length")
                entered = False
            if pending_yielded:
                self._forced_step(pending)
            else:
                self.chain.write_token_row(pending)
            with self.chain.loop_guard():
                yield from self._generate(
                    max_new_tokens - produced,
                    stop_ids,
                    None if not thinking_open else max(think_budget - reasoning_tokens, 0),
                    should_stop,
                )

        while True:
            reason = None if should_stop is None else should_stop()
            if reason is not None:
                yield None, reason
                return
            if thinking_open and reasoning_tokens >= think_budget:
                force_think_end("length")
                thinking_open = False
                produced += 1
                yield THINK_END_ID, "length" if produced == max_new_tokens else None
                if produced == max_new_tokens:
                    return
                pending = self.chain.read_token_row()
                self._require_vocabulary(pending)
                pending_yielded = False
                continue
            position = len(self.committed)
            if not pending_yielded:
                if produced + 1 == max_new_tokens:
                    self.row_token = pending
                    yield pending, "length"
                    return
                if pending >= TOKENIZER_SIZE or pending in stop_ids or not self.chain.mtp_pass_fits(position):
                    yield from finish_on_one_row_loop()
                    return
                produced += 1
                if thinking_open:
                    thinking_open = pending != THINK_END_ID
                    reasoning_tokens += 1
                yield pending, None
                pending_yielded = True
                reason = None if should_stop is None else should_stop()
                if reason is not None:
                    # The hit token is consumed as the 1-row loop consumes it: the previous pass settles whole, the
                    # token is fed by a forced step (the row then holds the model's next token).
                    if entered:
                        self._mtp_settle("length")
                        entered = False
                    self._forced_step(pending)
                    yield None, reason
                    return
            elif not self.chain.mtp_pass_fits(position):
                yield from finish_on_one_row_loop()
                return
            if entered:
                with self.chain.loop_guard():
                    record = self.chain.mtp_step()
            else:
                record = self.chain.mtp_enter(pending, self.ple_context)  # the eager seed, then the bootstrap pass
                entered = True
            accepted = record.accepted
            emitted = [int(token) for token in record.argmaxes[: accepted + 1]]
            for token_id in emitted:
                self._require_vocabulary(token_id)
            run = Qwen38MTPPassRun(position, [pending, *emitted[:accepted]], emitted, 0)
            self._mtp_run = run
            budget_hit = False
            for index, token_id in enumerate(emitted):
                run.yielded = index + 1
                produced += 1
                if thinking_open:
                    thinking_open = token_id != THINK_END_ID
                    reasoning_tokens += 1
                if token_id >= TOKENIZER_SIZE or token_id in stop_ids:
                    yield token_id, "error" if token_id >= TOKENIZER_SIZE else "stop"
                    return
                if produced == max_new_tokens:
                    self.row_token = token_id  # _mtp_settle leaves it in the row, as _generate does
                    yield token_id, "length"
                    return
                yield token_id, None
                reason = None if should_stop is None else should_stop()
                if reason is not None:
                    yield None, reason
                    return
                if thinking_open and reasoning_tokens >= think_budget:
                    budget_hit = True  # the 1-row loop forces </think> here: the rest of the pass is rolled back
                    break
            if budget_hit:
                force_think_end("budget")
                thinking_open = False
                produced += 1
                yield THINK_END_ID, "length" if produced == max_new_tokens else None
                if produced == max_new_tokens:
                    return
                pending = self.chain.read_token_row()
                self._require_vocabulary(pending)
                pending_yielded = False
                continue
            # The whole pass was consumed: its rows are the device's; its device commit is the next pass's first
            # job or the settle's, so the run stays until then.
            self.committed.extend(run.rows)
            run.host_committed = True
            pending = emitted[accepted]
            pending_yielded = True

    def _mtp_settle(self, finish: str) -> None:
        """Leave the pass loop after its last pass (``self._mtp_run``): commit the rows the request consumed and
        rebuild the 1-row buffers, then leave the row as :meth:`_generate` would.

        Of the pass's ``[t_P, d_1 .. d_a]`` rows and ``[d_1 .. d_a, t']`` emitted tokens, the request took the
        first ``yielded`` tokens; the last one is consumed (fed to the device) unless the request ended on
        ``length``, where the 1-row loop leaves the last token in the row unconsumed.  A consumed ``t'`` (not a row
        of the pass) is fed by a forced step after the hand-off.  The rows past the consumed prefix (drafts the
        device already ran) are rolled back by committing fewer rows.
        """

        run = self._mtp_run
        if run is None:
            return
        accepted = len(run.rows) - 1
        consume_last = finish != "length"
        consumed = min(run.yielded - 1 + consume_last, accepted)
        rows = 1 + consumed
        force_last = consume_last and run.yielded - 1 + consume_last > accepted
        if run.host_committed and rows != len(run.rows):
            raise Qwen38ChatChainError(f"a consumed pass settles whole: {rows} of {len(run.rows)} rows")
        self.ple_context = self.chain.mtp_leave(position=run.position + rows, committed_rows=rows)
        if not run.host_committed:
            self.committed.extend(run.rows[:rows])
        self._mtp_run = None
        if force_last:
            self._forced_step(run.emitted[accepted])
        else:
            self.chain.write_token_row(run.emitted[rows - 1])  # the model's next token, unconsumed, as after _generate

    # -- one request -------------------------------------------------------------------------

    def reset(self) -> None:
        """Position 0 and position-zero state at the captured addresses; the committed sequence is dropped."""

        self.chain.reset_and_seed(SEED_TOKEN_ID)
        self.committed = []
        self.ple_context = None
        self.last_finish = None
        self.row_unconsumed = False
        self.row_token = None

    def reusable_prefix(self, token_ids: Sequence[int]) -> tuple[int, str]:
        """How the device meets ``token_ids``: ``(n, "extends")`` when they extend the committed ``n`` ids (or repeat
        them exactly while the unconsumed next token is still in the row; a partial match cannot be rewound),
        ``(n, "snapshot")`` when they extend the ``n`` ids of the prompt-end snapshot instead, else ``(0, "reset")``."""

        common = 0
        while common < len(self.committed) and common < len(token_ids) and self.committed[common] == token_ids[common]:
            common += 1
        if common == len(self.committed) and (common < len(token_ids) or self.row_unconsumed):
            return common, "extends"
        snapshot = self.snapshot
        if (
            snapshot is not None
            and len(token_ids) > len(snapshot.ids)
            and tuple(token_ids[: len(snapshot.ids)]) == snapshot.ids
        ):
            return len(snapshot.ids), "snapshot"
        return 0, "reset"

    def _restore_prompt_snapshot(self) -> None:
        """The chain's snapshot back on device (position and phases included); the host follows its record."""

        self.chain.restore_prompt_snapshot()
        self.committed = list(self.snapshot.ids)
        self.ple_context = self.snapshot.ple_context
        self.last_finish = None
        self.row_unconsumed = False

    def _capture_prompt_snapshot(self) -> None:
        """The device state after the committed ids (every prompt token but the last), copied on the chain; the
        record makes a later render that extends these ids a restore instead of a reset."""

        if not self.snapshot_available or not self.committed:
            return
        self.chain.capture_prompt_snapshot(len(self.committed))
        self.snapshot = Qwen38PromptSnapshot(tuple(self.committed), self.ple_context)

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
        speculative: bool = True,
    ) -> Qwen38ChatCompletion:
        """Prefill what the device does not already hold, then generate up to ``max_tokens`` tokens (the remaining
        context when ``None``; ``require_budget``).

        Any failure inside the device section leaves the chain's queue and the
        model owner in an unknown state: the session is poisoned and must not
        serve again.  ``on_token`` raising ``OSError`` (the client went away) is
        not such a failure: it is called between steps, so generation stops with
        ``disconnected`` and, as after ``length``, the next token stays in the row.
        ``prefill_mode`` overrides the session's mode for this request.
        ``should_stop`` is polled between steps and at the prefill's event syncs
        (the teacher-forced cadence, the chunk driver's per-chunk events) and
        ends the request with the reason it returns, the row unconsumed (a stop
        inside the chunk driver hands off at the chunks replayed so far; the row
        then holds no prediction and an exact repeat of the committed prefix
        resets);
        ``think_budget`` forces ``</think>`` after that many reasoning tokens
        (the caller passes it only when the prompt left the think block open).
        ``sampling`` (a request with ``temperature > 0``) runs the sampled loop over
        the chain's candidate row instead of the greedy loop; ``None`` is greedy.
        On an MTP chain a greedy chunked-mode request generates through the pass
        loop unless ``speculative`` is False (the 1-row loop, for the hand-off gate).
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
        # MTP drafting serves the greedy requests of the chunked mode; sampled and teacher-forced requests take the
        # 1-row loops (speculative=False is the diagnostic form: a greedy request on the 1-row loop of an MTP chain).
        drafting = self.mtp is not None and speculative and sampling is None and mode == "chunked"
        started_ns = self.clock_ns()
        common, reuse = self.reusable_prefix(token_ids)
        try:
            self.row_token = None
            if reuse == "snapshot":
                self._restore_prompt_snapshot()
            elif reuse == "reset":
                self.reset()
            # The snapshot stays valid until a prefill reaching its last token replaces it (its buffers change on no
            # other path), so a reset or a stopped prefill leaves the earlier one restorable.
            suffix = token_ids[common:]
            chunked: Qwen38PrefillResult | None = None
            # All but the last prompt token through the chunk trace when enough rows remain after the alignment
            # steps; the last one is the first decode replay and is always teacher-forced inside the guard.  The
            # prompt-end snapshot is taken before that last token: after the hand-off, or inside the forced prefill.
            before_last: Callable[[], None] | None = self._capture_prompt_snapshot
            if mode == "chunked" and self.chunk_prefill_rows(len(suffix) - 1) >= CHUNK_PREFILL_MIN_ROWS:
                chunked = self._prefill_chunked(suffix[:-1], suffix[-1], should_stop)
                suffix = suffix[-1:]
                before_last = None
                if chunked.stopped is None:
                    self._capture_prompt_snapshot()
            # A stop inside the chunk driver ended the request at its hand-off: nothing more runs on the device.
            chunk_stopped = chunked is not None and chunked.stopped is not None
            generated: list[int] = []
            finish: str | None = chunked.stopped if chunk_stopped else None
            hook_stopped = chunk_stopped
            first_ns = last_ns = started_ns
            mtp_before = None if not drafting else (self.mtp.passes, self.mtp.accepted_drafts)

            def consume(steps) -> None:
                nonlocal finish, hook_stopped, first_ns, last_ns
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

            if chunk_stopped:
                prefill_done_ns = self.clock_ns()
                first_ns = last_ns = prefill_done_ns
            elif drafting:
                # The pass loop takes the guard per pass (its mode switches synchronize) and settles outside it.
                with self.chain.loop_guard():
                    finish = self._prefill(suffix, should_stop, before_last)
                hook_stopped = finish is not None
                prefill_done_ns = self.clock_ns()
                first_ns = last_ns = prefill_done_ns
                if not hook_stopped:
                    finish = "length"
                    consume(self._generate_mtp(max_tokens, stop_ids, think_budget, should_stop))
                    self._mtp_settle(finish)
            else:
                with self.chain.loop_guard():
                    finish = self._prefill(suffix, should_stop, before_last)
                    hook_stopped = finish is not None
                    prefill_done_ns = self.clock_ns()
                    first_ns = last_ns = prefill_done_ns
                    if not hook_stopped:
                        finish = "length"
                        consume(
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
            self.last_finish = finish
            self.row_unconsumed = not chunk_stopped and (hook_stopped or finish in ("length", "disconnected"))
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
            reset=reuse == "reset",
            restored=reuse == "snapshot",
            prefill_tokens=prefill_tokens,
            prefill_seconds=(prefill_done_ns - started_ns) / 1e9,
            ttft_seconds=(first_ns - started_ns) / 1e9,
            decode_seconds=decode_seconds,
            tokens_per_second=tokens_per_second,
            position=position,
            prefill_mode="teacher_forced" if chunked is None else "chunked",
            prefill_forced_tokens=(
                prefill_tokens if chunked is None else chunked.timing.alignment_steps + (0 if chunk_stopped else 1)
            ),
            prefill_chunks=0 if chunked is None else chunked.timing.chunks,
            prefill_tail_rows=0 if chunked is None else chunked.timing.tail_rows,
            prefill_handoff_ms=0.0 if chunked is None else chunked.timing.handoff_ms,
            mtp=(
                None
                if not drafting
                else self.mtp.summary(
                    passes=self.mtp.passes - mtp_before[0], accepted_drafts=self.mtp.accepted_drafts - mtp_before[1]
                )
            ),
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


def resolve_route(hardware_profile: ResidentHardwareProfile) -> tuple[ResidentHardwareProfile, dict]:
    """The profile with its route: derived from the cluster descriptor (no device is opened) and adopted when the
    profile leaves it ``None`` (the LoudBox: recorded, not pinned), checked against the pinned one otherwise."""

    import yaml

    lane = f"{hardware_profile.host} partition-{hardware_profile.partition.upper()}"
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
    if hardware_profile.route is None:
        hardware_profile = replace(hardware_profile, route=route, route_nodes=route_nodes)
    elif route != hardware_profile.route or route_nodes != hardware_profile.route_nodes:
        raise Qwen38ChatChainError(
            f"{lane} route actual=(logical={route}, nodes={route_nodes}) "
            f"expected=(logical={hardware_profile.route}, nodes={hardware_profile.route_nodes})"
        )
    return hardware_profile, {
        "cluster_descriptor": str(descriptor_path),
        "chips_with_mmio": chips_with_mmio,
        "route_derivation": derive_route.__name__,
    }


def open_partition_b_mesh(marker: Marker, hardware_profile: ResidentHardwareProfile) -> tuple[Any, dict]:
    """The runner's mesh open for one lane: the profile's route derivation, FABRIC_1D, one 1D four-device mesh as
    the logical 1x4.

    Same calls and checks as ``run()`` of the timing runner on a larger
    host (its partition-B values are the profile's defaults, hence the
    name; a partition-A profile brings its own nodes, route and locks; both
    are a 4x1 line reshaped to 1x4, ``derive_canonical_line_route``).  A ring
    profile (the QuietBox) opens the 1x4 its mesh graph descriptor reports in
    ``derive_ring_walk_route`` order.  Every check before the fabric enable
    raises with the fabric untouched; a failure after it disables the fabric
    before re-raising, so the caller owns the fabric only once this returns.
    """

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
    hardware_profile, derivation = resolve_route(hardware_profile)
    chips_with_mmio = derivation["chips_with_mmio"]
    route, route_nodes = hardware_profile.route, hardware_profile.route_nodes
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
        "route_derivation": derivation["route_derivation"],
        "cluster_descriptor": derivation["cluster_descriptor"],
        "fabric_config": "FABRIC_1D",
        "collective_topology": "Linear",
        "live_mapping": live_mapping,
        "preopen_runner_lock_proof": lock_proof,
    }


@dataclass
class Qwen38ChainMTP:
    """What an MTP-drafting chain adds (``mtp_v2``): the MTP components, the verify / draft states and their three
    traces, the TAIL rows' step inputs, the chunk extension, the live pass loop and the cumulative counters.

    ``step_written`` tracks the TAIL contract: every TAIL reads the step inputs written for its step (a forced step
    writes the next prompt token; a device-token step selects the resolved argmax).
    """

    drafts: int
    anchor: str
    components: Any
    verify: mtp_v2.Qwen38TTNNVerifyState
    draft: mtp_v2.Qwen38TTNNDraftState
    step_inputs: mtp_v2.Qwen38TTNNMTPStepInputs
    chunk_extension: mtp_v2.Qwen38TTNNMTPChunkExtension | None
    traces: mtp_v2.Qwen38TTNNMTPTraces | None = None
    verify_output: mtp_v2.Qwen38TTNNVerifyOutput | None = None
    chain: mtp_v2.Qwen38TTNNMTPChain | None = None
    step_written: bool = False
    passes: int = 0
    accepted_drafts: int = 0
    capture_ms: dict[str, float] = field(default_factory=dict)
    trace_dram_bytes_per_bank: dict[str, int] = field(default_factory=dict)
    admission: dict[str, Any] = field(default_factory=dict)  # mtp_capacity_admission at the build's context
    dram_bytes_per_bank: dict[str, int] = field(default_factory=dict)  # measured growth: components, states, traces

    @property
    def alignment(self) -> mtp_v2.Qwen38TTNNVerifyAlignment:
        return self.verify.alignment

    def captured_trace_ids(self) -> list[int]:
        if self.traces is None:
            return []
        return [self.traces.verify_first, self.traces.commit, self.traces.draft]

    def record(self, pass_record: mtp_v2.Qwen38TTNNMTPPassRecord) -> mtp_v2.Qwen38TTNNMTPPassRecord:
        self.passes += 1
        self.accepted_drafts += pass_record.accepted
        return pass_record

    def summary(self, *, passes: int | None = None, accepted_drafts: int | None = None) -> dict[str, Any]:
        """The ``qwen38.mtp`` object: k, anchor, passes, accepted drafts and tokens per pass (``a + 1`` per pass),
        cumulative (health) or over the counts a request added."""

        passes = self.passes if passes is None else passes
        accepted_drafts = self.accepted_drafts if accepted_drafts is None else accepted_drafts
        return {
            "k": self.drafts,
            "anchor": self.anchor,
            "passes": passes,
            "accepted_drafts": accepted_drafts,
            "tokens_per_pass": None if not passes else round((passes + accepted_drafts) / passes, 4),
        }


@dataclass
class Qwen38TracedChain:
    """The runner's single-trace chain kept open for serving: 8 decode traces, the persistent token row, the PLE
    row, (``chunked_prefill``) the chunk state with its one chunk trace captured after the decode traces,
    (``sampling``) the candidate-row epilogue with its readback row, and (``mtp``) the MTP drafting extension."""

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
    # (``long_chunks``) the 128-row chunk state beside the 32-row one and its trace, captured after the chunk trace;
    # the driver runs the 128-row chunks first, then the 32-row chunks and the padded tail.
    long_chunk_state: Any = None
    long_chunk_trace_id: int | None = None
    long_chunk_capture_ms: float = 0.0
    # The GDN state re-anchor the chunk trace was captured with (every chunk replay commits through it).
    chunk_gdn_step_anchor: bool = False
    # The allocation tracker verified every trace once after the captures; per-prefill re-verification (140-290 ms
    # in the full-model gate) is a diagnostic switch.
    verify_each_prefill: bool = False
    sampling: sampling_step.Qwen38SamplingChainExtension | None = None
    mtp: Qwen38ChainMTP | None = None
    # The prompt-end snapshot buffers (model.allocate_generic_snapshot over the generic state and the MTP alignment
    # layer's), allocated before the warm pass; the session captures and restores through the two methods below.
    snapshot: Any = None
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

    def write_mtp_next_token(self, next_token_id: int | None) -> None:
        """MTP chains: the TAIL's MTP row consumes the token at P + 1: this one, or None for the step's own argmax."""

        self.mtp.step_inputs.write_next_token(next_token_id)
        self.mtp.step_written = True

    def execute_head(self, residue: int) -> None:
        if self.mtp is not None and not self.mtp.step_written:
            self.write_mtp_next_token(None)  # a device-token step: the MTP row takes the resolved argmax
        ttnn._ttnn_execute_trace(self.mesh, self.head_trace_ids[residue], cq_id=0, blocking=False)

    def execute_tail(self, residue: int) -> None:
        ttnn._ttnn_execute_trace(self.mesh, self.tail_trace_ids[residue], cq_id=0, blocking=False)
        if self.mtp is not None:
            self.mtp.step_written = False

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
        if self.mtp is not None:
            self.mtp.alignment.layer.reset_generic_state_inplace(self.mtp.alignment.generic_state)
            self.mtp.step_written = False
        self.write_token_row(token_id)
        ttnn.synchronize_device(self.mesh)
        position = self.state.position.read()
        if position != 0:
            raise Qwen38ChatChainError(f"reset left the device position at {position}, expected 0")
        resident_decode.require_token_row_holds(
            self.token_row_io, resident_decode.host_token_row(token_id), label="reset seed token row"
        )

    def capture_prompt_snapshot(self, position: int) -> None:
        """The generic state's recurrent buffers into the resident snapshot: device copies queued behind the TAIL
        whose state they record (nothing allocated, no synchronize: callable under the loop guard)."""

        self.built_target.model.capture_generic_snapshot(self.state, self.snapshot, position=position)

    def restore_prompt_snapshot(self) -> None:
        """The snapshot back into the generic state (its position and GDN phases included), then the device
        position read back against it."""

        self.built_target.model.restore_generic_snapshot(self.snapshot, self.state)
        if self.mtp is not None:
            self.mtp.step_written = False
        ttnn.synchronize_device(self.mesh)
        position = self.state.position.read()
        if position != self.snapshot.position:
            raise Qwen38ChatChainError(
                f"restore left the device position at {position}, expected {self.snapshot.position}"
            )

    def chunk_prefill(
        self,
        token_ids: Sequence[int],
        *,
        start_position: int,
        ple_context: tuple[int, int] | None,
        forced_step: Callable[[int, tuple[int, int] | None], tuple[int, int] | None],
        following_token: int | None = None,
        should_stop: Callable[[], str | None] | None = None,
    ) -> Qwen38PrefillResult:
        """The chunk driver on this chain (the full-model gate's prefill path): alignment steps through
        ``forced_step``, the seed, chunk replays (``should_stop`` polled at their event syncs), the padded tail,
        the hand-off.  Not under the loop guard.  ``following_token`` is the MTP layer's token at the last
        prefilled position (MTP chains)."""

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
            long_chunk_state=self.long_chunk_state,
            long_chunk_trace_id=self.long_chunk_trace_id,
            mtp=None if self.mtp is None else self.mtp.chunk_extension,
        ).run(
            token_ids,
            start_position=start_position,
            ple_context=ple_context,
            following_token=following_token,
            should_stop=should_stop,
        )

    # -- the MTP pass loop primitives (mtp chains) ---------------------------------------------

    def mtp_pass_fits(self, position: int) -> bool:
        return mtp_v2.verify_pass_fits(position, self.built_target.model.allocated_context)

    def _replay(self, trace_id: int) -> None:
        ttnn._ttnn_execute_trace(self.mesh, trace_id, cq_id=0, blocking=True)

    def _enqueue(self, trace_id: int) -> None:
        ttnn._ttnn_execute_trace(self.mesh, trace_id, cq_id=0, blocking=False)

    def mtp_enter(self, first_token: int, ple_context: tuple[int, int] | None) -> mtp_v2.Qwen38TTNNMTPPassRecord:
        """Eager switch into verify mode at the device position (the host's committed count), then the bootstrap
        pass whose row 0 is ``first_token`` (placeholder drafts).  Returns the pass record."""

        mtp = self.mtp
        model = self.built_target.model
        if mtp.chain is not None:
            raise Qwen38ChatChainError("the MTP pass loop is already active")
        position = self.state.position.read()
        mtp_v2.enter_verify_mode(model, self.state, mtp.verify, position=position, ple_context=ple_context)
        mtp.chain = mtp_v2.Qwen38TTNNMTPChain(
            model,
            mtp.verify,
            mtp.draft,
            mtp.traces,
            mtp.verify_output,
            replay=self._replay,
            position=position,
            enqueue=self._enqueue,
        )
        return mtp.record(mtp.chain.bootstrap([first_token] + [MTP_BOOTSTRAP_DRAFT_TOKEN] * mtp.drafts))

    def mtp_step(self) -> mtp_v2.Qwen38TTNNMTPPassRecord:
        if self.mtp.chain is None:
            raise Qwen38ChatChainError("the MTP pass loop is not active")
        return self.mtp.record(self.mtp.chain.step())

    def mtp_leave(self, *, position: int, committed_rows: int) -> tuple[int, int] | None:
        """Commit ``committed_rows`` of the last pass (its commit trace) and rebuild the 1-row buffers at ``position``."""

        mtp = self.mtp
        if mtp.chain is None:
            raise Qwen38ChatChainError("the MTP pass loop is not active")
        mtp.chain = None
        mtp.step_written = False
        return mtp_v2.leave_verify_mode(
            self.built_target.model,
            self.state,
            mtp.verify,
            position=position,
            committed_rows=committed_rows,
            commit=lambda: self._replay(mtp.traces.commit),
        )

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
        long_chunks: bool = False,
        mtp: int | None = None,
        mtp_gdn_anchor: str = "off",
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
        its programs there.  ``mtp`` (off by default) builds the MTP components and allocates the verify / draft
        states, the TAIL step inputs and the chunk extension before the warm pass, adds the MTP layer's row to
        every TAIL and its rows to the chunk body, warms the pass loop and both mode switches at every position
        residue, and captures the verify, commit and draft traces after the chunk trace.
        """

        if type(chunk_gdn_step_anchor) is not bool:
            raise ValueError(f"chunk_gdn_step_anchor must be a bool, got {chunk_gdn_step_anchor!r}")
        if type(long_chunks) is not bool:
            raise ValueError(f"long_chunks must be a bool, got {long_chunks!r}")
        if long_chunks and (not chunked_prefill or chunk_gdn_step_anchor):
            raise ValueError("long chunks need the chunked prefill and run without the GDN step anchor")
        if mtp is not None and mtp not in MTP_DRAFTS:
            raise ValueError(f"mtp drafts must be one of {MTP_DRAFTS} or None, got {mtp!r}")
        if mtp_gdn_anchor not in MTP_GDN_ANCHORS:
            raise ValueError(f"mtp_gdn_anchor must be one of {MTP_GDN_ANCHORS}, got {mtp_gdn_anchor!r}")
        if long_chunks and mtp is not None:
            raise ValueError(
                "long chunks and MTP drafting are alternatives: the MTP chunk extension is a 32-row chunk option"
            )
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

        def dram_allocated_per_bank() -> int:
            return int(ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank)

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
        mtp_components = None
        mtp_dram_bytes_per_bank: dict[str, int] = {}
        if mtp is not None:
            # The admission's table row for this context (the server refused an unfit --mtp before the mesh opened);
            # the measured growth of the MTP build, its states and its traces is checked against its estimate.
            mtp_admission = mtp_capacity_admission(model.allocated_context, ring_size=builder.identity.ring_size)
            if not mtp_admission["fits"]:
                raise Qwen38ChatChainError(
                    f"MTP drafting does not fit at allocated context {model.allocated_context}: {mtp_admission}"
                )
            marker("before-chat-mtp-build")
            allocated_before_mtp_build = dram_allocated_per_bank()
            mtp_components = builder.build_mtp_components()
            synchronize()
            mtp_dram_bytes_per_bank["components"] = dram_allocated_per_bank() - allocated_before_mtp_build
            if resident_owner.cache_load_success_count != MTP_EXPECTED_CACHE_LOADS:
                raise Qwen38ChatChainError(
                    f"resident expert pairs loaded with the MTP layer {resident_owner.cache_load_success_count}, "
                    f"expected {MTP_EXPECTED_CACHE_LOADS}"
                )
            marker("after-chat-mtp-build")
        state = model.allocate_generic_state()
        chunk_state = None
        long_chunk_state = None
        if chunked_prefill:
            model.reset_generic_state_inplace(state)
            chunk_state = model.allocate_chunk_state(state)
            model.reset_chunk_state_inplace(state, chunk_state)
            if long_chunks:
                long_chunk_state = model.allocate_chunk_state(state, rows=LONG_CHUNK_ROWS, base=chunk_state)
                model.reset_chunk_state_inplace(state, long_chunk_state)
        # The MTP states sit beside the generic and chunk states, before any capture: every trace bakes their
        # addresses in (the verify / draft states, the TAIL step inputs, the chunk extension).
        chain_mtp = None
        if mtp is not None:
            allocated_before_mtp_states = dram_allocated_per_bank()
            verify = mtp_v2.allocate_verify_state(
                model,
                state,
                drafts=mtp,
                mtp_components=mtp_components,
                gdn_step_anchor_layers=MTP_GDN_ANCHOR_LAYERS[mtp_gdn_anchor],
            )
            chain_mtp = Qwen38ChainMTP(
                drafts=mtp,
                anchor=mtp_gdn_anchor,
                components=mtp_components,
                verify=verify,
                draft=mtp_v2.allocate_draft_state(model, verify),
                step_inputs=mtp_v2.Qwen38TTNNMTPStepInputs.allocate(mesh, model.mesh_contract),
                chunk_extension=(
                    None
                    if chunk_state is None
                    else mtp_v2.Qwen38TTNNMTPChunkExtension.allocate(model, verify, chunk_state)
                ),
                admission=mtp_admission,
                dram_bytes_per_bank=mtp_dram_bytes_per_bank,
            )
            synchronize()
            mtp_dram_bytes_per_bank["states"] = dram_allocated_per_bank() - allocated_before_mtp_states
        # The prompt-end snapshot buffers beside the states, before any capture (the tracker's post-capture check
        # then sees no later allocation); with MTP the alignment layer's generic state is a 49th layer of it.
        snapshot = model.allocate_generic_snapshot(
            state,
            extra_layers=() if chain_mtp is None else ((chain_mtp.alignment.layer, chain_mtp.alignment.generic_state),),
        )
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
            mtp=chain_mtp,
            snapshot=snapshot,
        )

        def warm_step(position: int, warm_token_id: int, ple_context, *, mtp_next: int | None):
            """One eager 1-row step: PLE row (checked against the host oracle), the body, the epilogue's ops."""

            chain.write_token_row(warm_token_id)
            if chain_mtp is not None:
                chain.write_mtp_next_token(mtp_next)
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
            if chain_mtp is None:
                output = model.forward_decode_generic(prepared, state)
            else:
                head = model.forward_decode_generic_head(prepared, state)
                output = model.forward_decode_generic_tail(head, prepared, state, retain_mtp_inputs=True)
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
            if chain_mtp is not None:
                mtp_v2.forward_mtp_step_row(
                    model,
                    chain_mtp.alignment,
                    chain_mtp.step_inputs,
                    output.residual,
                    resolved_row,
                    rope=output.rope,
                    qsa_position=output.qsa_position,
                )
                chain_mtp.step_written = False
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
            return resolved, ple_context

        # Warm pass, eager, misses allowed: every kernel of the generic body
        # compiles here.  Kept gates (actual vs expected): position counter,
        # resident PLE lookup vs the host oracle, device greedy resolve vs CPU,
        # the candidate row vs torch.topk of the full gather.
        marker("before-chat-warm-pass")
        ple_context: tuple[int, int] | None = None
        warm_tokens = [resident_decode.SEQUENTIAL_TRACE_WARM_EMBEDDING_TOKEN_IDS[p % TP_SIZE] for p in range(TP_SIZE)]
        for position in range(resident_decode.SINGLE_TRACE_WARM_POSITIONS):
            # The MTP row takes the next warm token as a prompt token would, the argmax at the last position.
            mtp_next = warm_tokens[position + 1] if position + 1 < len(warm_tokens) else None
            resolved, ple_context = warm_step(position, warm_tokens[position % TP_SIZE], ple_context, mtp_next=mtp_next)
        synchronize()
        marker("after-chat-warm-pass")
        if warm_hook is not None:
            marker("before-chat-warm-hook")
            warm_hook(chain)
            synchronize()
            marker("after-chat-warm-hook")

        if chain_mtp is not None:
            # The pass loop's programs and both eager mode switches at every position residue: the seed's ring
            # slot slices are position-dependent programs and the server switches at any P.  Each round: the
            # switch in, one verify pass (its MoE rows / alignment / accept programs), the draft rows, the switch
            # out through the commit (the request's settle form), then 1-row steps to the next residue.
            marker("before-chat-mtp-warm-pass")
            warm_fed = resident_decode.SINGLE_TRACE_WARM_POSITIONS  # positions the warm pass consumed so far
            for residue in range(RESIDUE_CLASSES):
                position = state.position.read()
                if position != warm_fed or position % RESIDUE_CLASSES != residue:
                    raise Qwen38ChatChainError(f"MTP warm round {residue} at position {position}, fed {warm_fed}")
                mtp_v2.enter_verify_mode(model, state, chain_mtp.verify, position=position, ple_context=ple_context)
                mtp_v2.write_verify_inputs(model, chain_mtp.verify, [resolved] + [MTP_BOOTSTRAP_DRAFT_TOKEN] * mtp)
                output = mtp_v2.forward_verify(model, chain_mtp.verify, state, catch_up=False)
                mtp_v2.forward_draft(model, chain_mtp.verify, chain_mtp.draft, state, output)
                synchronize()
                # The pass row: the verify's accept row and the draft's token chain (checked to start at its t', d_1').
                readback, _ = mtp_v2.read_pass_row(chain_mtp.verify, chain_mtp.draft)
                mtp_v2.commit_verify_host(chain_mtp.verify, readback.accepted)
                output.release_tensors()
                if state.position.read() != position + readback.accepted + 1:
                    raise Qwen38ChatChainError(
                        f"MTP warm verify pass left P = {state.position.read()}, expected "
                        f"{position + readback.accepted + 1}"
                    )
                # The settle form: every committed row this round, the next token unconsumed in the row.
                committed_rows = readback.accepted + 1
                ple_context = mtp_v2.leave_verify_mode(
                    model,
                    state,
                    chain_mtp.verify,
                    position=position + committed_rows,
                    committed_rows=committed_rows,
                    commit=lambda: mtp_v2.forward_commit(model, chain_mtp.verify, state),
                )
                warm_fed += committed_rows
                resolved = readback.argmaxes[readback.accepted]
                # 1-row steps to the next residue class (the seed's programs at every P mod 4).
                while residue + 1 < RESIDUE_CLASSES and warm_fed % RESIDUE_CLASSES != residue + 1:
                    resolved, ple_context = warm_step(warm_fed, resolved, ple_context, mtp_next=None)
                    warm_fed += 1
            synchronize()
            marker("after-chat-mtp-warm-pass")

        if long_chunks:
            # One eager 128-row chunk from the reset state: its programs compile here, before the miss guard.
            marker("before-chat-long-chunk-warm-pass")
            model.reset_generic_state_inplace(state)
            model.reset_chunk_state_inplace(state, chunk_state)
            model.reset_chunk_state_inplace(state, long_chunk_state)
            model.write_chunk_inputs(long_chunk_state, list(WARM_LONG_CHUNK_TOKEN_IDS), ple_context=None)
            synchronize()
            model.forward_prefill_chunk_generic(long_chunk_state, state)
            synchronize()
            actual = state.position.read()
            if actual != LONG_CHUNK_ROWS:
                raise Qwen38ChatChainError(f"warm long chunk position counter {actual} vs expected {LONG_CHUNK_ROWS}")
            marker("after-chat-long-chunk-warm-pass")
        if chunked_prefill:
            # One eager chunk from the reset state (the chunk body's programs compile here) and both hand-off
            # forms (a closed block fills the staging tile; an open block copies it and selects a non-empty ring).
            marker("before-chat-chunk-warm-pass")
            chunk_extension = None if chain_mtp is None else chain_mtp.chunk_extension
            model.reset_generic_state_inplace(state)
            model.reset_chunk_state_inplace(state, chunk_state)
            model.write_chunk_inputs(chunk_state, list(WARM_CHUNK_TOKEN_IDS), ple_context=None)
            if chunk_extension is not None:
                chain_mtp.alignment.layer.reset_generic_state_inplace(chain_mtp.alignment.generic_state)
                chunk_extension.reset_chunk()
                chunk_extension.write_tokens(model, [*WARM_CHUNK_TOKEN_IDS[1:], WARM_CHUNK_TOKEN_IDS[0]])
            synchronize()
            model.forward_prefill_chunk_generic(
                chunk_state, state, gdn_step_anchor=chunk_gdn_step_anchor, mtp=chunk_extension
            )
            synchronize()
            actual = state.position.read()
            if actual != CHUNK_ROWS:
                raise Qwen38ChatChainError(f"warm chunk position counter {actual} vs expected {CHUNK_ROWS}")
            model.finish_prefill(state, chunk_state, CHUNK_ROWS)
            model.finish_prefill(state, chunk_state, CHUNK_ROWS - 1)
            if chunk_extension is not None:
                chunk_extension.finish_chunk(model, prefilled=CHUNK_ROWS)
                chunk_extension.finish_chunk(model, prefilled=CHUNK_ROWS - 1)
            synchronize()
            actual = state.position.read()
            if actual != CHUNK_ROWS - 1:
                raise Qwen38ChatChainError(f"warm hand-off position counter {actual} vs expected {CHUNK_ROWS - 1}")
            marker("after-chat-chunk-warm-pass")

        # The snapshot round trip (its copy programs compile here; the raw-key ring copy has no other warm form):
        # capture at the warm position, read every recurrent buffer, reset the state, restore, read again: the
        # restored buffers must be the captured ones bitwise and the position the captured one.
        marker("before-chat-snapshot-warm-pass")

        def snapshot_rows() -> dict[str, list[torch.Tensor]]:
            return {
                label: [ttnn.to_torch(local) for local in ttnn.get_device_tensors(source)]
                for label, source, _ in snapshot.pairs
            }

        snapshot_position = state.position.read()
        chain.capture_prompt_snapshot(snapshot_position)
        synchronize()
        expected_rows = snapshot_rows()
        model.reset_generic_state_inplace(state)
        synchronize()
        chain.restore_prompt_snapshot()
        for label, expected_locals in snapshot_rows().items():
            for device, (expected, actual) in enumerate(zip(expected_rows[label], expected_locals, strict=True)):
                bits = torch.int16 if expected.dtype == torch.bfloat16 else torch.int32
                if not torch.equal(expected.view(bits), actual.view(bits)):
                    raise Qwen38ChatChainError(
                        f"snapshot round trip: {label} on device {device} differs after the restore at position "
                        f"{snapshot_position}: max abs {float((actual.float() - expected.float()).abs().max())}"
                    )
        marker("after-chat-snapshot-warm-pass")

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
        if chain_mtp is not None:
            verify = chain_mtp.verify
            for tensor in (
                chain_mtp.step_inputs.host_lane,
                chain_mtp.step_inputs.select_index,
                verify.token_row,
                verify.draft_lanes,
                verify.ple_rows.embedding_rows,
                verify.accepted,
                chain_mtp.draft.pass_row,
            ):
                ttnn.mark_corruptible(tensor)
            if chain_mtp.chunk_extension is not None:
                chain_mtp.chunk_extension.reset_chunk()
                ttnn.mark_corruptible(chain_mtp.chunk_extension.token_row)
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
            sampling chain's epilogue runs the same three and then the candidate row.  An MTP chain's epilogue
            runs the MTP layer's row (the retained residual, RoPE rows and position inputs, the resolved argmax
            as the fallback token) between the resolve and the row copy, then releases what the TAIL retained."""

            if chain_mtp is not None:
                if trace_output.logits is None or trace_output.residual is None:
                    raise Qwen38ChatChainError("TAIL capture returned no logits or no retained MTP inputs")
                candidates = lm_head.greedy_candidates(trace_output.logits)
                trace_token_row = lm_head.resolve_greedy_on_device(candidates)
                mtp_v2.forward_mtp_step_row(
                    model,
                    chain_mtp.alignment,
                    chain_mtp.step_inputs,
                    trace_output.residual,
                    trace_token_row,
                    rope=trace_output.rope,
                    qsa_position=trace_output.qsa_position,
                )
                ttnn.deallocate(trace_output.residual)
                trace_output.rope.deallocate()
                trace_output.qsa_position.deallocate()
                trace_output.residual = trace_output.rope = trace_output.qsa_position = None
                ttnn.copy(trace_token_row, token_row_io)
                if chain.sampling is not None:
                    chain.sampling.trace_rows.append(
                        lm_head.sampling_candidates(trace_output.logits, chain.sampling.constants)
                    )
                    chain.sampling.trace_logits.append(trace_output.logits)
                return candidates, trace_token_row
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
        dram_before_captures = dram_allocated_per_bank()
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
                retain_mtp_inputs=chain_mtp is not None,
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
        dram_after_decode = dram_allocated_per_bank()
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
                mtp=None if chain_mtp is None else chain_mtp.chunk_extension,
                gdn_step_anchor=chunk_gdn_step_anchor,
            )
            chain.chunk_capture_ms = (clock_ns() - chunk_capture_started_ns) / 1e6
            synchronize()
            marker("after-chat-chunk-capture")
        dram_after_chunk = dram_allocated_per_bank()
        if long_chunks:
            marker("before-chat-long-chunk-capture")
            long_chunk_capture_started_ns = clock_ns()
            chain.long_chunk_state = long_chunk_state
            chain.long_chunk_trace_id = model.capture_prefill_chunk(
                long_chunk_state,
                state,
                guard=lambda label: resident_decode.forbid_trace_body_host_io_and_sync(phase=f"chat long {label}"),
                cq_id=0,
            )
            chain.long_chunk_capture_ms = (clock_ns() - long_chunk_capture_started_ns) / 1e6
            synchronize()
            marker("after-chat-long-chunk-capture")
        if chain_mtp is not None:
            # The verify (first pass), commit and draft traces after the chunk trace; the draft body reads the
            # verify output's readback address, so the verify capture comes first.  Capture records without
            # executing: the device state is unchanged.
            marker("before-chat-mtp-captures")

            def guard(label: str):
                return resident_decode.forbid_trace_body_host_io_and_sync(phase=f"chat {label}")

            capture_started_ns = clock_ns()
            verify_first, verify_output = mtp_v2.capture_verify(
                model, chain_mtp.verify, state, catch_up=False, guard=guard, cq_id=0
            )
            chain_mtp.capture_ms["verify_first"] = (clock_ns() - capture_started_ns) / 1e6
            capture_started_ns = clock_ns()
            commit = mtp_v2.capture_commit(model, chain_mtp.verify, state, guard=guard, cq_id=0)
            chain_mtp.capture_ms["commit"] = (clock_ns() - capture_started_ns) / 1e6
            capture_started_ns = clock_ns()
            draft = mtp_v2.capture_draft(
                model, chain_mtp.verify, chain_mtp.draft, state, verify_output, guard=guard, cq_id=0
            )
            chain_mtp.capture_ms["draft"] = (clock_ns() - capture_started_ns) / 1e6
            chain_mtp.traces = mtp_v2.Qwen38TTNNMTPTraces(verify_first=verify_first, draft=draft, commit=commit)
            chain_mtp.verify_output = verify_output
            ttnn.mark_corruptible(verify_output.readback)
            synchronize()
            marker("after-chat-mtp-captures")
            chain_mtp.trace_dram_bytes_per_bank = {
                "decode_traces": dram_after_decode - dram_before_captures,
                "chunk_trace": dram_after_chunk - dram_after_decode,
                "mtp_traces": dram_allocated_per_bank() - dram_after_chunk,
            }
            chain_mtp.dram_bytes_per_bank["traces"] = chain_mtp.trace_dram_bytes_per_bank["mtp_traces"]
            mtp_growth = sum(chain_mtp.dram_bytes_per_bank.values())
            if mtp_growth > chain_mtp.admission["required_free_bytes_per_bank"]:
                raise Qwen38ChatChainError(
                    f"MTP DRAM growth {mtp_growth} bytes per bank {chain_mtp.dram_bytes_per_bank} exceeds the admission's "
                    f"estimate {chain_mtp.admission['required_free_bytes_per_bank']} at allocated context "
                    f"{model.allocated_context}"
                )
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
        """Every captured trace: 4 HEAD, 4 TAIL, then the chunk trace, the long chunk trace and the MTP traces
        when captured."""

        return (
            self.head_trace_ids
            + self.tail_trace_ids
            + [trace_id for trace_id in (self.chunk_trace_id, self.long_chunk_trace_id) if trace_id is not None]
            + ([] if self.mtp is None else self.mtp.captured_trace_ids())
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
        self.long_chunk_trace_id = None
        if self.mtp is not None:
            self.mtp.traces = None
            if self.mtp.verify_output is not None:
                self.mtp.verify_output.release_tensors()
                self.mtp.verify_output = None
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
        if self.long_chunk_state is not None:  # before the 32-row state whose histories it shares
            self.built_target.model.release_chunk_state(self.long_chunk_state)
            self.long_chunk_state = None
        if self.mtp is not None:
            # The MTP states before the chunk and generic states they sit beside.
            model = self.built_target.model
            if self.mtp.chunk_extension is not None:
                self.mtp.chunk_extension.release()
            self.mtp.step_inputs.deallocate()
            mtp_v2.release_draft_state(model, self.mtp.verify, self.mtp.draft)
            mtp_v2.release_verify_state(model, self.mtp.verify)
        if self.chunk_state is not None:
            self.built_target.model.release_chunk_state(self.chunk_state)
            self.chunk_state = None
        if self.snapshot is not None:
            self.built_target.model.release_generic_snapshot(self.snapshot)
            self.snapshot = None
        self.built_target.model.release_generic_state(self.state)
        ttnn.synchronize_device(mesh)
        self.built_target.components.close_resident_experts()
        ttnn.synchronize_device(mesh)
        self.closed = True


def construct_chain(
    prepared: Any,
    mesh: Any,
    *,
    marker: Marker,
    chunked_prefill: bool = True,
    sampling: bool = False,
    bf4_stage_limit: int | None = None,
    long_chunks: bool = False,
    mtp: int | None = None,
    mtp_gdn_anchor: str = "off",
) -> Qwen38TracedChain:
    """Live construction on the open mesh (missing BF4 layers converted first), then the chain prologue."""

    construction = construct_live_decode_diagnostic(
        prepared,
        mesh_device=mesh,
        collective_topology=ttnn.Topology.Linear,
        marker=marker,
        bf4_stage_limit=bf4_stage_limit,
    )
    return Qwen38TracedChain.open(
        construction,
        marker=marker,
        chunked_prefill=chunked_prefill,
        sampling=sampling,
        long_chunks=long_chunks,
        mtp=mtp,
        mtp_gdn_anchor=mtp_gdn_anchor,
    )
