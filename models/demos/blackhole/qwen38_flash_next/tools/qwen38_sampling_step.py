# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sampled decode steps for the 1-row chain: the request mapping, the chain additions, the step order, the discriminator.

The greedy loop of ``tools/qwen38_chat_session.py`` (``_generate``) stays as it is: ``temperature 0``
never reads the candidate row and is bitwise today's chain.  A sampled step is a forced step whose
token the host chooses after reading the candidate row that TAIL(t-1) wrote (the ordering F2 of
``REFRESH-B5B-TRACE-SPLIT-SPEC-20260902.md`` section 4.4 with the forced-step write in the middle)::

    D   row = sampling.read_candidate_row()                    # blocking: completes TAIL(t-1); the device idles
        x_t = sample(row)
    A'  chain.write_token_row(x_t)                             # overwrites the device's greedy id
    C   chain.execute_head(residue)
    F   ple_context = chain.refresh_ple_row(x_t, ple_context)  # hidden under HEAD(t) as today
    G   chain.execute_tail(residue)

The read is blocking on purpose: nothing could overlap it (HEAD(t) needs x_t), and the blocking read is
the cheaper wake-up (61-70 us vs the evented 71-79 us in the epilogue micro-test).  The price is the
lost HEAD overlap of the token read (the exposed segment the discriminator measures: read wake-up, parse
and sample, token write, HEAD launch); the PLE refresh still hides under HEAD.  Finish semantics are the greedy loop's: EOS is consumed, ``max_tokens``
leaves the last token unconsumed (written into the token row, so the row holds the token the client saw),
a token at or above the tokenizer size ends with ``error``, the ``should_stop`` hook ends the request
between steps, and a thinking budget forces ``</think>`` through the session's forced step.  A pure
continuation of a sampled request (nothing to prefill, the row unconsumed) samples that position again.

What the chain gains (``Qwen38SamplingChainExtension``): the epilogue constants and readback row, the
epilogue that runs the greedy resolve unchanged and then ``Qwen38TTNNLMHead.sampling_candidates``, the
warm-pass check of the row against the eager full gather, and two primitives (the blocking row read, the
fallback's eager full-vocabulary gather of the residue's trace logits).

Request fields (``parameters_from_request``; ``extra_body`` keys are merged by the server first):

    field                                    absent                                       notes
    temperature                              card profile keyed on enable_thinking        0 = the greedy loop, bitwise
                                             (thinking 1.0/0.95/20/0, instruct 0.7/0.8/20/1.5)
    greedy                                   false                                        true = the greedy loop (temperature must be absent or 0)
    top_p, top_k, min_p                      1.0 / 20 / 0 (profile values when             top_k > 32 refused (400);
                                             temperature is absent too)                   top_k 0 = full-vocabulary fallback
    presence_penalty, frequency_penalty      0 / 0 (instruct profile: presence 1.5)        OpenAI additive semantics over the
                                                                                          request's output (not the prompt)
    repetition_penalty                       1.0                                          transformers rule over prompt + output
    seed                                     the server's draw, echoed in the response    one torch.Generator per request
    logprobs, top_logprobs                   false / 0                                    log-softmax over the read candidates (above
                                                                                          the full value by -log of their mass);
                                                                                          sampled requests only
    n                                        1                                            n != 1 refused (one traced chain)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from statistics import median
from typing import Any

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_protocol import THINK_END_ID
from models.demos.blackhole.qwen38_flash_next.tools.resident_decode import (
    SINGLE_TRACE_RESIDUE_CLASS_TRACES as RESIDUE_CLASSES,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    Qwen38ShardedLogits,
    Qwen38TTNNLMHead,
    Qwen38TTNNSamplingCandidateConstants,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import (
    CANDIDATE_TOP_K_LIMIT,
    MAX_SEED,
    MAX_TOP_LOGPROBS,
    Qwen38CandidateFallback,
    Qwen38CandidateRow,
    Qwen38CandidateSample,
    Qwen38SamplingParameters,
    sample_candidates,
    sample_full_vocabulary,
)

SAMPLING_REQUEST_FIELDS = (
    "temperature",
    "greedy",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "seed",
    "logprobs",
    "top_logprobs",
    "n",
)
_PROFILE_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "min_p",
    "frequency_penalty",
    "repetition_penalty",
)
# The chain discriminator: the seeded card-profile stream, the row check over the four residue classes, the period bar.
DISCRIMINATOR_TOKENS = 128
DISCRIMINATOR_SEED = 20260904
DISCRIMINATOR_ROW_TOKENS = 33
DISCRIMINATOR_PERIOD_TARGET_MS = 50.4


class Qwen38SamplingRequestError(ValueError):
    """A sampling field the server refuses with 400; the message starts with the field's name."""


# -- the request mapping -------------------------------------------------------------------------------------


def parameters_from_request(
    document: Mapping[str, Any], *, enable_thinking: bool, seed: int
) -> Qwen38SamplingParameters | None:
    """The request's sampling fields as one validated policy (the table in the module docstring); ``None`` = greedy.

    ``temperature 0`` and ``greedy: true`` take the greedy loop (``None``); ``greedy`` with a
    positive temperature is a contradiction and refused.
    """

    def number(name: str, default: Real, *, integer: bool = False) -> Real:
        value = document.get(name)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, Real) or (integer and not isinstance(value, int)):
            kind = "an integer" if integer else "a number"
            raise Qwen38SamplingRequestError(f"{name} must be {kind}, got {value!r}")
        return value

    if number("n", 1, integer=True) != 1:
        raise Qwen38SamplingRequestError("n must be 1: the server runs one traced chain")
    greedy = document.get("greedy", False)
    if type(greedy) is not bool:
        raise Qwen38SamplingRequestError(f"greedy must be a boolean, got {greedy!r}")
    temperature = number("temperature", None)
    if temperature is not None and temperature < 0:
        raise Qwen38SamplingRequestError(f"temperature must be nonnegative, got {temperature}")
    if greedy and temperature:
        raise Qwen38SamplingRequestError(f"greedy is true but temperature is {temperature}: greedy needs 0 or none")
    if greedy or (temperature is not None and temperature == 0):
        return None
    request_seed = number("seed", seed, integer=True)
    if not 0 <= request_seed <= MAX_SEED:
        raise Qwen38SamplingRequestError(f"seed must be in [0,{MAX_SEED}], got {request_seed}")
    profile = (
        Qwen38SamplingParameters.official_thinking(seed=request_seed)
        if enable_thinking
        else Qwen38SamplingParameters.official_non_thinking(seed=request_seed)
    )
    if temperature is None:
        if all(document.get(name) is None for name in _PROFILE_FIELDS):
            return profile
        defaults = {name: getattr(profile, name) for name in _PROFILE_FIELDS}
    else:
        defaults = {
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": 20,
            "presence_penalty": 0.0,
            "min_p": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
        }
    fields = {name: number(name, default, integer=name == "top_k") for name, default in defaults.items()}
    if fields["top_k"] > CANDIDATE_TOP_K_LIMIT:
        raise Qwen38SamplingRequestError(f"top_k must be at most {CANDIDATE_TOP_K_LIMIT}, got {fields['top_k']}")
    try:
        return Qwen38SamplingParameters(seed=request_seed, **fields)
    except (TypeError, ValueError) as error:
        raise Qwen38SamplingRequestError(str(error)) from error


def logprobs_from_request(document: Mapping[str, Any]) -> tuple[bool, int]:
    """``(logprobs, top_logprobs)``: OpenAI's boolean and the 0..20 alternatives count."""

    logprobs = document.get("logprobs", False)
    if type(logprobs) is not bool:
        raise Qwen38SamplingRequestError("logprobs must be a boolean")
    top = document.get("top_logprobs", 0)
    if top is None:
        top = 0
    if isinstance(top, bool) or type(top) is not int or not 0 <= top <= MAX_TOP_LOGPROBS:
        raise Qwen38SamplingRequestError(f"top_logprobs must be an integer in [0,{MAX_TOP_LOGPROBS}], got {top!r}")
    if top and not logprobs:
        raise Qwen38SamplingRequestError("top_logprobs requires logprobs: true")
    return logprobs, top


def logprobs_content_item(
    sample: Qwen38CandidateSample | None, token_id: int, decode: Callable[[int], str]
) -> dict[str, Any]:
    """One ``choices[].logprobs.content[]`` item (OpenAI shape); a forced token (no sample) carries ``logprob`` null."""

    def item(token: int, logprob: float | None) -> dict[str, Any]:
        text = decode(token)
        return {"token": text, "logprob": logprob, "bytes": list(text.encode("utf-8"))}

    if sample is None:
        return {**item(token_id, None), "top_logprobs": []}
    document = item(sample.token_id, sample.logprob)
    document["top_logprobs"] = [item(token, logprob) for token, logprob in sample.top_logprobs]
    return document


# -- the chain additions -------------------------------------------------------------------------------------


class Qwen38SamplingChainExtension:
    """What a sampling chain adds to the traced chain: the epilogue, its warm-pass check and three per-step primitives."""

    def __init__(self, lm_head: Qwen38TTNNLMHead, mesh: Any) -> None:
        self.lm_head = lm_head
        self.mesh = mesh
        self.constants = Qwen38TTNNSamplingCandidateConstants.build(mesh, lm_head.mesh_contract)
        self.trace_logits: list[Qwen38ShardedLogits] = []  # TAIL residue r's logits (trace-stable), for the fallback
        self.trace_rows: list[Any] = []

    def capture_epilogue(self, trace_output: Any, token_row_io: Any) -> tuple[Any, Any]:
        """TAIL's last ops: the greedy path exactly as today, then the candidate row into the readback buffer."""

        if trace_output.logits is None:
            raise RuntimeError("TAIL capture returned no logits")
        candidates = self.lm_head.greedy_candidates(trace_output.logits)
        trace_token_row = self.lm_head.resolve_greedy_on_device(candidates)
        ttnn.copy(trace_token_row, token_row_io)
        self.trace_rows.append(self.lm_head.sampling_candidates(trace_output.logits, self.constants))
        self.trace_logits.append(trace_output.logits)
        return candidates, trace_token_row

    def mark_corruptible(self) -> None:
        """Before the miss guard closes: the readback row is rewritten by every replay, like the token row."""

        ttnn.mark_corruptible(self.constants.readback_row)

    def mark_trace_rows_corruptible(self) -> None:
        """After a capture: its row is rewritten by every replay, like the greedy candidates."""

        ttnn.mark_corruptible(self.trace_rows[-1])

    def warm(self, logits: Qwen38ShardedLogits, *, label: str) -> dict[str, Any]:
        """Eager epilogue and fallback gather (their programs compile here); the row must equal torch.topk of the gather."""

        row = self.lm_head.sampling_candidates(logits, self.constants)
        full = self._gather(logits)
        actual = self.read_candidate_row()
        ttnn.deallocate(row)
        agreement = actual.agreement(Qwen38CandidateRow.emulate(full.to(torch.bfloat16)))
        if not all(agreement["values_bitwise"]) or not all(agreement["ids_equal_up_to_boundary_ties"]):
            raise RuntimeError(
                f"{label}: candidate row {actual.values.tolist()} {actual.ids.tolist()} vs torch.topk of the full "
                f"gather beyond boundary ties: {agreement}"
            )
        return agreement

    def release(self) -> None:
        for tensor in (*self.trace_rows, self.constants.readback_row, self.constants.shard_vocab_start):
            ttnn.deallocate(tensor)
        self.trace_rows.clear()
        self.trace_logits.clear()

    def read_candidate_row(self) -> Qwen38CandidateRow:
        """Blocking: completes the queued TAIL and parses its row (device 0's replica; the four agree)."""

        return Qwen38CandidateRow.from_host_row(ttnn.to_torch(ttnn.get_device_tensors(self.constants.readback_row)[0]))

    def read_full_logits(self, residue: int) -> torch.Tensor:
        """The fallback's eager full-vocabulary gather of TAIL(residue)'s logits, about 2 MB, as fp32 ``[VOCAB_SIZE]``."""

        return self._gather(self.trace_logits[residue])

    def _gather(self, logits: Qwen38ShardedLogits) -> torch.Tensor:
        gathered = self.lm_head.gather_full_logits(logits)
        try:
            host = ttnn.to_torch(ttnn.get_device_tensors(gathered)[0])
        finally:
            ttnn.deallocate(gathered)
        return host.reshape(-1).to(torch.float32)


# -- the sampled loop -----------------------------------------------------------------------------------------


@dataclass
class Qwen38SamplingStepClocks:
    """Per-token instants of the sampled loop (``clock_ns``) and its counters, for the period arm of the discriminator."""

    row_available_ns: list[int] = field(default_factory=list)  # the blocking row read returned
    token_available_ns: list[int] = field(default_factory=list)  # the host chose x_t
    head_enqueued_ns: list[int] = field(default_factory=list)
    tail_enqueued_ns: list[int] = field(default_factory=list)
    fallbacks: int = 0  # steps sampled over the full vocabulary (top_k 0, a boosting penalty, the guard)
    candidate_misses: int = 0  # sampled tokens outside the read candidate row (only a fallback can produce one)

    def periods_ms(self) -> list[float]:
        instants = self.token_available_ns
        return [(later - earlier) / 1e6 for earlier, later in zip(instants[:-1], instants[1:])]

    def summary(self) -> dict[str, Any]:
        periods = self.periods_ms()
        # The device waits for all of this: parse + sample (row -> token), token-row write + HEAD launch
        # (token -> head).  The read wake-up before ``row_available`` is the rest of the period delta.
        sample = [(token - row) / 1e6 for row, token in zip(self.row_available_ns, self.token_available_ns)]
        exposed = [(head - token) / 1e6 for token, head in zip(self.token_available_ns, self.head_enqueued_ns)]
        return {
            "tokens": len(self.token_available_ns),
            "period_median_ms": median(periods) if periods else None,
            "period_min_ms": min(periods) if periods else None,
            "sample_median_ms": median(sample) if sample else None,
            "host_segment_median_ms": median(exposed) if exposed else None,
            "fallbacks": self.fallbacks,
            "candidate_misses": self.candidate_misses,
        }


@dataclass
class Qwen38SamplingRequest:
    """One sampled request: its policy, the generator its seed starts, the alternatives it wants, what it produced.

    ``samples`` is index-aligned with the completion's token ids; a token the thinking
    budget forced has no sample (``None``).
    """

    parameters: Qwen38SamplingParameters
    top_logprobs: int = 0
    generator: torch.Generator = field(init=False, repr=False)
    samples: list[Qwen38CandidateSample | None] = field(default_factory=list)
    clocks: Qwen38SamplingStepClocks = field(default_factory=Qwen38SamplingStepClocks)

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, Qwen38SamplingParameters) or self.parameters.temperature == 0:
            raise ValueError("a sampling request needs Qwen38SamplingParameters with temperature > 0")
        if isinstance(self.top_logprobs, bool) or type(self.top_logprobs) is not int:
            raise TypeError(f"top_logprobs must be an integer, got {self.top_logprobs!r}")
        self.generator = torch.Generator(device="cpu").manual_seed(self.parameters.seed)

    def as_dict(self) -> dict[str, Any]:
        """The response's ``qwen38.sampling`` object: the policy, its seed and the loop's counters."""

        return {
            **parameters_as_dict(self.parameters),
            "fallbacks": self.clocks.fallbacks,
            "candidate_misses": self.clocks.candidate_misses,
        }


def parameters_as_dict(p: Qwen38SamplingParameters) -> dict[str, Any]:
    return {
        "profile": p.profile.value,
        "temperature": p.temperature,
        "top_p": p.top_p,
        "top_k": p.top_k,
        "min_p": p.min_p,
        "presence_penalty": p.presence_penalty,
        "frequency_penalty": p.frequency_penalty,
        "repetition_penalty": p.repetition_penalty,
        "seed": p.seed,
    }


def choose_token(
    session: Any, row: Qwen38CandidateRow, request: Qwen38SamplingRequest, *, tail_residue: int, prompt_tokens: int
) -> Qwen38CandidateSample:
    """The candidate sampler, or the full-vocabulary sampler over TAIL(tail_residue)'s logits when the row cannot prove
    exactness; ``session.committed`` is the penalties' history, its first ``prompt_tokens`` the request's prompt."""

    try:
        sample = sample_candidates(
            row,
            request.parameters,
            token_history=session.committed,
            prompt_tokens=prompt_tokens,
            generator=request.generator,
            top_logprobs=request.top_logprobs,
        )
    except Qwen38CandidateFallback:
        request.clocks.fallbacks += 1
        full = session.sampling.read_full_logits(tail_residue)
        sample = sample_full_vocabulary(
            full,
            request.parameters,
            token_history=session.committed,
            prompt_tokens=prompt_tokens,
            generator=request.generator,
            top_logprobs=request.top_logprobs,
        )
    if sample.token_id not in row.ids.reshape(-1).tolist():
        request.clocks.candidate_misses += 1
    return sample


def generate_sampled(
    session: Any,
    request: Qwen38SamplingRequest,
    max_new_tokens: int,
    *,
    stop_ids: Sequence[int],
    tokenizer_size: int,
    think_budget: int | None = None,
    should_stop: Callable[[], str | None] | None = None,
    forced_step: Callable[[int], None] | None = None,
    clock_ns: Callable[[], int] | None = None,
) -> Iterator[tuple[int | None, str | None]]:
    """The sampled A-G loop over ``session.chain`` and ``session.sampling``; yields ``(x_t, finish)`` like ``_generate``.

    ``session`` carries ``chain`` (the traced chain's primitives), ``sampling`` (the extension),
    ``committed`` (the host list of every input token: the prompt when the loop starts, then this
    request's output; the penalties' history, presence and frequency over the output only) and
    ``ple_context``.  Every sampled token appends its sample to ``request.samples`` before it is
    yielded.  A hook stop yields ``(None, reason)``; the thinking budget forces ``</think>`` through
    ``forced_step`` (the session's) and appends ``None`` for it.
    """

    if not isinstance(request, Qwen38SamplingRequest):
        raise TypeError("generate_sampled needs a Qwen38SamplingRequest")
    if think_budget is not None and forced_step is None:
        raise TypeError("a thinking budget needs the session's forced step")
    chain, clocks = session.chain, request.clocks
    now = clock_ns or (lambda: 0)
    prompt_tokens = len(session.committed)
    produced = 0
    reasoning_tokens = 0
    thinking_open = think_budget is not None
    while True:
        reason = None if should_stop is None else should_stop()
        if reason is not None:
            yield None, reason
            return
        if thinking_open and reasoning_tokens >= think_budget:
            forced_step(THINK_END_ID)
            request.samples.append(None)
            thinking_open = False
            produced += 1
            yield THINK_END_ID, "length" if produced == max_new_tokens else None
            if produced == max_new_tokens:
                return
            continue
        tail_residue = (len(session.committed) - 1) % RESIDUE_CLASSES  # TAIL(t-1) ran one committed token ago
        if produced + 1 == max_new_tokens:
            row = session.sampling.read_candidate_row()  # completes TAIL(t-1)
            sample = choose_token(session, row, request, tail_residue=tail_residue, prompt_tokens=prompt_tokens)
            request.samples.append(sample)
            chain.write_token_row(sample.token_id)  # the row holds the token the client saw, as after the greedy loop
            yield sample.token_id, "length"
            return
        residue = len(session.committed) % RESIDUE_CLASSES
        row = session.sampling.read_candidate_row()  # completes TAIL(t-1)
        clocks.row_available_ns.append(now())
        sample = choose_token(session, row, request, tail_residue=tail_residue, prompt_tokens=prompt_tokens)
        token_id = sample.token_id
        clocks.token_available_ns.append(now())
        chain.write_token_row(token_id)
        chain.execute_head(residue)
        clocks.head_enqueued_ns.append(now())
        produced += 1
        if thinking_open:
            thinking_open = token_id != THINK_END_ID
            reasoning_tokens += 1
        session.ple_context = chain.refresh_ple_row(token_id, session.ple_context)
        chain.execute_tail(residue)
        clocks.tail_enqueued_ns.append(now())
        session.committed.append(token_id)
        request.samples.append(sample)
        if token_id >= tokenizer_size or token_id in stop_ids:
            chain.read_token_row()  # completes TAIL(t); x_{t+1} is discarded
            yield token_id, "error" if token_id >= tokenizer_size else "stop"
            return
        yield token_id, None


# -- the discriminator over a live session ------------------------------------------------------------------


def compare_candidate_rows_with_full_gathers(
    session: Any, forced_step: Callable[[int], None], tokens: Sequence[int]
) -> list[dict[str, Any]]:
    """Arm a: force ``tokens`` one by one; after each TAIL compare its row with the eager full gather (torch.topk)."""

    records = []
    for token in tokens:
        residue = len(session.committed) % RESIDUE_CLASSES
        forced_step(token)
        row = session.sampling.read_candidate_row()  # blocking: completes this TAIL
        full = session.sampling.read_full_logits(residue)
        greedy_argmax = int(torch.argmax(full))
        row_greedy = int(row.ids.reshape(-1)[int(torch.argmax(row.values.reshape(-1)))])
        records.append(
            {
                "input_token": token,
                "position": len(session.committed),
                "residue": residue,
                "row_vs_torch": row.agreement(Qwen38CandidateRow.emulate(full.to(torch.bfloat16))),
                "greedy_argmax": greedy_argmax,
                "row_greedy": row_greedy,
                "greedy_agrees": row_greedy == greedy_argmax,
            }
        )
    return records


def run_discriminator(
    session: Any,
    prompt_ids: Sequence[int],
    *,
    tokens: int = DISCRIMINATOR_TOKENS,
    seed: int = DISCRIMINATOR_SEED,
    row_tokens: int = DISCRIMINATOR_ROW_TOKENS,
    period_target_ms: float = DISCRIMINATOR_PERIOD_TARGET_MS,
) -> dict[str, Any]:
    """The chain discriminator on one prompt: rows vs truth (a), the greedy period (d), two seeded card-profile streams
    (c: identical), the sampled period against the target (d) and every sampled token inside its candidate row (e).

    Arm b (``temperature 0`` = the CPU greedy record, 96/96) is the server's acceptance replay, run before this.
    """

    prompt_ids = list(prompt_ids)
    session.reset()
    rows = compare_candidate_rows_with_full_gathers(session, session._forced_step, prompt_ids[:row_tokens])
    rows_pass = all(
        all(record["row_vs_torch"]["values_bitwise"])
        and all(record["row_vs_torch"]["ids_equal_up_to_boundary_ties"])
        and record["greedy_agrees"]
        for record in rows
    )
    greedy = session.complete(prompt_ids, tokens, stop_ids=())  # extends the forced prefix
    greedy_period_ms = None if greedy.tokens_per_second is None else 1e3 / greedy.tokens_per_second
    streams, clocks, completions = [], [], []
    for _ in range(2):
        request = Qwen38SamplingRequest(Qwen38SamplingParameters.official_thinking(seed=seed))
        completion = session.complete(prompt_ids, tokens, stop_ids=(), sampling=request)
        streams.append(completion.token_ids)
        clocks.append(request.clocks.summary())
        completions.append(
            {
                "tokens_per_second": completion.tokens_per_second,
                "period_ms": None if completion.tokens_per_second is None else 1e3 / completion.tokens_per_second,
                "finish_reason": completion.finish_reason,
                "samples": len(request.samples),
                "sampled_logprob_median": median(sample.logprob for sample in request.samples if sample is not None),
            }
        )
    sampled_periods = [summary["period_median_ms"] for summary in clocks]
    period_pass = all(period is not None and period <= period_target_ms for period in sampled_periods)
    exposed = []
    for summary in clocks:
        sample, host = summary["sample_median_ms"], summary["host_segment_median_ms"]
        exposed.append(None if sample is None or host is None else sample + host)
    result = {
        "prompt_tokens": len(prompt_ids),
        "tokens": tokens,
        "seed": seed,
        "profile": Qwen38SamplingParameters.official_thinking(seed=seed).profile.value,
        "rows": rows,
        "rows_pass": rows_pass,
        "greedy": {
            "token_ids": greedy.token_ids,
            "tokens_per_second": greedy.tokens_per_second,
            "period_ms": greedy_period_ms,
            "finish_reason": greedy.finish_reason,
        },
        "sampled": {"streams": streams, "clocks": clocks, "completions": completions},
        "streams_identical": streams[0] == streams[1] and len(streams[0]) == tokens,
        "sampled_differs_from_greedy": streams[0] != greedy.token_ids,
        "sampled_period_median_ms": sampled_periods,
        "period_target_ms": period_target_ms,
        "period_pass": period_pass,
        "period_delta_vs_greedy_ms": [
            None if period is None or greedy_period_ms is None else period - greedy_period_ms
            for period in sampled_periods
        ],
        # period delta = read wake-up (this residual) + parse and sample + token write and HEAD launch
        "exposed_host_median_ms": exposed,
        "read_wakeup_residual_ms": [
            None if period is None or greedy_period_ms is None or host is None else period - greedy_period_ms - host
            for period, host in zip(sampled_periods, exposed)
        ],
        "fallbacks": [summary["fallbacks"] for summary in clocks],
        "candidate_misses": [summary["candidate_misses"] for summary in clocks],
        "tokens_inside_candidates": all(summary["candidate_misses"] == 0 for summary in clocks),
    }
    result["pass"] = bool(
        rows_pass and result["streams_identical"] and result["tokens_inside_candidates"] and period_pass
    )
    return result


__all__ = [
    "DISCRIMINATOR_PERIOD_TARGET_MS",
    "DISCRIMINATOR_ROW_TOKENS",
    "DISCRIMINATOR_SEED",
    "DISCRIMINATOR_TOKENS",
    "RESIDUE_CLASSES",
    "SAMPLING_REQUEST_FIELDS",
    "Qwen38SamplingChainExtension",
    "Qwen38SamplingRequest",
    "Qwen38SamplingRequestError",
    "Qwen38SamplingStepClocks",
    "choose_token",
    "compare_candidate_rows_with_full_gathers",
    "generate_sampled",
    "logprobs_content_item",
    "logprobs_from_request",
    "parameters_as_dict",
    "parameters_from_request",
    "run_discriminator",
]
