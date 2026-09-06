# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The sampled step order, its finish semantics and the request mapping, on a fake chain (no device)."""

from __future__ import annotations

import inspect

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_protocol import THINK_END_ID
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import VOCAB_SIZE
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import (
    Qwen38CandidateRow,
    Qwen38SamplingParameters,
    Qwen38SamplingProfile,
    sample_full_vocabulary,
)
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling_step as step

EOS = (248_046, 248_044)
TOKENIZER_SIZE = 248_077


def _logits_for(history: list[int], *, eos_at: int | None = None) -> torch.Tensor:
    generator = torch.Generator().manual_seed(len(history) * 7919 + (history[-1] if history else 0))
    row = torch.randn(VOCAB_SIZE, generator=generator) * 2.0
    peaks = torch.randperm(VOCAB_SIZE, generator=generator)[:8]
    row[peaks] = 12.0 + torch.rand(8, generator=generator) * 6.0
    if eos_at is not None and len(history) >= eos_at:
        row[EOS[0]] = 60.0
    return row.to(torch.bfloat16)


class FakeChain:
    """The traced chain's primitives; TAIL(t) 'computes' the row of the logits after the tokens written so far."""

    def __init__(self, prefix: list[int], *, eos_at: int | None = None) -> None:
        self.inputs = list(prefix)  # every token HEAD consumed
        self.eos_at = eos_at
        self.log: list[str] = []
        self.tails = 0
        self.row: int | None = None

    def logits(self) -> torch.Tensor:
        return _logits_for(self.inputs, eos_at=self.eos_at)

    def write_token_row(self, token_id: int) -> None:
        self.log.append(f"write:{token_id}")
        self.row = token_id

    def execute_head(self, residue: int) -> None:
        self.log.append(f"head:{residue}")
        self.inputs.append(self.row)

    def execute_tail(self, residue: int) -> None:
        self.log.append(f"tail:{residue}")
        self.tails += 1
        self.row = int(torch.argmax(self.logits().to(torch.float32)))

    def refresh_ple_row(self, token_id: int, context):
        self.log.append(f"ple:{token_id}")
        return (token_id, 0 if context is None else context[0])

    def record_event(self):
        self.log.append("event")
        return object()

    def event_synchronize(self, event) -> None:
        self.log.append("sync")

    def read_token_row(self) -> int:
        self.log.append("read_token_row")
        return self.row


class FakeSampling:
    def __init__(self, chain: FakeChain) -> None:
        self.chain = chain
        self.full_reads = 0

    def read_candidate_row(self) -> Qwen38CandidateRow:
        self.chain.log.append("read_row")
        return Qwen38CandidateRow.emulate(self.chain.logits())

    def read_full_logits(self, residue: int) -> torch.Tensor:
        self.full_reads += 1
        self.chain.log.append(f"full_gather:{residue}")
        return self.chain.logits().to(torch.float32)


class FakeSession:
    def __init__(self, prefix: list[int], *, eos_at: int | None = None) -> None:
        self.chain = FakeChain(prefix, eos_at=eos_at)
        self.sampling = FakeSampling(self.chain)
        self.committed = list(prefix)
        self.ple_context = None

    def _forced_step(self, token_id: int) -> None:
        residue = len(self.committed) % step.RESIDUE_CLASSES
        self.chain.write_token_row(token_id)
        self.chain.execute_head(residue)
        self.ple_context = self.chain.refresh_ple_row(token_id, self.ple_context)
        self.chain.execute_tail(residue)
        self.committed.append(token_id)


THINKING = Qwen38SamplingParameters.official_thinking(seed=77)


def _request(parameters=THINKING, **fields) -> step.Qwen38SamplingRequest:
    return step.Qwen38SamplingRequest(parameters, **fields)


def _run(session: FakeSession, request: step.Qwen38SamplingRequest, max_new_tokens: int, **extra):
    extra.setdefault("forced_step", session._forced_step)
    return list(
        step.generate_sampled(session, request, max_new_tokens, stop_ids=EOS, tokenizer_size=TOKENIZER_SIZE, **extra)
    )


def test_step_order_is_blocking_read_sample_write_head_ple_tail() -> None:
    session, request = FakeSession([1, 2, 3]), _request()
    items = _run(session, request, 4)
    assert [finish for _, finish in items] == [None, None, None, "length"]
    first = session.chain.log[:5]
    token = items[0][0]
    assert first == ["read_row", f"write:{token}", "head:3", f"ple:{token}", "tail:3"]
    assert "event" not in session.chain.log and "sync" not in session.chain.log
    # The residue follows the committed length (3 prefix tokens, then 4, 5).
    assert [entry for entry in session.chain.log if entry.startswith("head:")] == ["head:3", "head:0", "head:1"]
    # max_tokens: the last token comes from a blocking row read, is not consumed, and is written into the row.
    last = items[-1][0]
    assert session.chain.log[-2:] == ["read_row", f"write:{last}"] and session.chain.tails == 3
    assert session.chain.row == last
    assert session.committed == [1, 2, 3] + [token for token, _ in items[:3]]
    assert session.ple_context is not None and session.ple_context[0] == items[2][0]
    assert [sample.token_id for sample in request.samples] == [token for token, _ in items]


def test_eos_is_consumed_then_tail_is_completed_by_a_blocking_read() -> None:
    session = FakeSession([5, 6], eos_at=4)
    items = _run(session, _request(), 32)
    tokens = [token for token, _ in items]
    assert tokens[-1] == EOS[0] and items[-1][1] == "stop" and len(tokens) == 3
    assert session.chain.log[-3:] == [f"ple:{EOS[0]}", "tail:0", "read_token_row"]
    assert session.committed[-1] == EOS[0]  # consumed, like the greedy loop


def test_same_seed_same_stream_and_the_sample_carries_logprobs() -> None:
    left = _run(FakeSession([9]), _request(top_logprobs=3), 24)
    right_request = _request(top_logprobs=3)
    right = _run(FakeSession([9]), right_request, 24)
    assert [t for t, _ in left] == [t for t, _ in right]
    assert _run(FakeSession([9]), _request(Qwen38SamplingParameters.official_thinking(seed=78)), 24) != left
    sample = right_request.samples[0]
    assert sample.logprob <= 0 and len(sample.top_logprobs) == 3
    assert sample.top_logprobs[0][1] >= sample.top_logprobs[-1][1]
    assert right_request.clocks.fallbacks == 0 and right_request.clocks.candidate_misses == 0


def test_top_k_zero_takes_the_full_vocabulary_fallback_every_step() -> None:
    session = FakeSession([4])
    parameters = Qwen38SamplingParameters(temperature=0.8, top_p=0.9, top_k=0, presence_penalty=0.0, seed=3)
    request = _request(parameters)
    items = _run(session, request, 6, clock_ns=lambda: 0)
    assert session.sampling.full_reads == 6 and request.clocks.fallbacks == 6
    # The fallback's residue is TAIL(t-1)'s: one committed token before the step's residue.
    gathers = [entry for entry in session.chain.log if entry.startswith("full_gather:")]
    assert gathers[:4] == ["full_gather:0", "full_gather:1", "full_gather:2", "full_gather:3"]
    # And it is the reference sampler on the same logits and generator state.
    expected = sample_full_vocabulary(
        _logits_for([4]).to(torch.float32), parameters, token_history=[4], generator=torch.Generator().manual_seed(3)
    )
    assert items[0][0] == expected.token_id


def test_presence_penalty_counts_the_requests_output_not_the_prompt() -> None:
    # top_k 1 makes the sampled token the penalized argmax.  The fake row depends on the prompt's length and last
    # token only, so the prompt can hold the top peak of its own row.
    last = 3
    top = int(torch.argmax(_logits_for([0, last]).to(torch.float32)))
    prefix = [top, last]
    parameters = Qwen38SamplingParameters(temperature=1.0, top_p=1.0, top_k=1, presence_penalty=2.0, seed=1)
    session, request = FakeSession(prefix), _request(parameters)
    tokens = [token for token, _ in _run(session, request, 3)]
    # The prompt's `top` is not penalized: the first token is the row's argmax.  Under the whole-history rule it
    # would have lost 2.0 and the argmax would have moved.
    assert tokens[0] == top
    assert (
        sample_full_vocabulary(_logits_for(prefix).to(torch.float32), parameters, token_history=prefix).token_id != top
    )
    # Every step is the reference over the committed history with the prompt exempt (the last token is the
    # unconsumed length read, chosen the same way).
    generator = torch.Generator().manual_seed(1)
    for index, token in enumerate(tokens):
        history = prefix + tokens[:index]
        expected = sample_full_vocabulary(
            _logits_for(history).to(torch.float32),
            parameters,
            token_history=history,
            prompt_tokens=len(prefix),
            generator=generator,
        )
        assert token == expected.token_id
    assert session.committed == prefix + tokens[:-1]


def test_should_stop_ends_the_loop_between_steps_without_a_sample() -> None:
    session, request = FakeSession([1]), _request()
    polls = []
    items = _run(session, request, 8, should_stop=lambda: (polls.append(1), "halt" if len(polls) == 3 else None)[1])
    assert items[-1] == (None, "halt") and len(items) == 3 and len(request.samples) == 2
    assert session.chain.log[-1].startswith("tail:") and len(session.committed) == 3


def test_think_budget_forces_think_end_through_the_forced_step() -> None:
    session, request = FakeSession([1]), _request()
    items = _run(session, request, 6, think_budget=2)
    tokens = [token for token, _ in items]
    assert tokens[2] == THINK_END_ID and request.samples[2] is None and len(request.samples) == 6
    forced = session.chain.log.index(f"write:{THINK_END_ID}")
    assert session.chain.log[forced - 1].startswith("tail:") and session.chain.log[forced + 1] == "head:3"
    assert session.committed[3] == THINK_END_ID
    with pytest.raises(TypeError, match="forced step"):
        list(step.generate_sampled(FakeSession([1]), _request(), 2, stop_ids=EOS, tokenizer_size=1, think_budget=1))


def test_temperature_zero_is_refused_by_the_request() -> None:
    with pytest.raises(ValueError, match="temperature > 0"):
        step.Qwen38SamplingRequest(Qwen38SamplingParameters.greedy())
    with pytest.raises(TypeError, match="Qwen38SamplingRequest"):
        list(step.generate_sampled(FakeSession([1]), THINKING, 4, stop_ids=EOS, tokenizer_size=TOKENIZER_SIZE))


def test_clocks_summary_reports_period_and_host_segment() -> None:
    clock = iter(range(0, 10_000_000, 100_000))
    request = _request()
    _run(FakeSession([2]), request, 5, clock_ns=lambda: next(clock))
    summary = request.clocks.summary()
    assert summary["tokens"] == 4 and summary["fallbacks"] == 0 and summary["candidate_misses"] == 0
    assert summary["period_median_ms"] == pytest.approx(0.4) and summary["host_segment_median_ms"] == pytest.approx(0.1)
    assert summary["sample_median_ms"] == pytest.approx(0.1)
    assert set(request.as_dict()) == {
        "profile",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "seed",
        "fallbacks",
        "candidate_misses",
        "logprobs_normalizer",
    }
    assert request.as_dict()["logprobs_normalizer"] == "candidate_row"  # the reported logprobs are row-relative


# --- the request mapping ----------------------------------------------------------------------------------------


def test_no_sampling_field_is_greedy_and_absent_temperature_maps_to_the_card_profile_keyed_on_thinking() -> None:
    # Decision B (2026-09-06): the launchers serve with --sampling, so a request naming no sampling field must stay
    # the bitwise greedy loop; a sampling field without a temperature (seed alone included) takes the card profile.
    assert step.parameters_from_request({"messages": []}, enable_thinking=True, seed=11) is None
    assert step.parameters_from_request({}, enable_thinking=False, seed=12) is None
    thinking = step.parameters_from_request({"seed": 11}, enable_thinking=True, seed=99)
    assert thinking.profile is Qwen38SamplingProfile.THINKING and thinking.seed == 11
    assert (thinking.temperature, thinking.top_p, thinking.top_k) == (1.0, 0.95, 20)
    instruct = step.parameters_from_request({"seed": 12}, enable_thinking=False, seed=99)
    assert instruct.profile is Qwen38SamplingProfile.NON_THINKING and instruct.presence_penalty == 1.5
    # A partial request keeps the profile's other values but is a custom policy.
    partial = step.parameters_from_request({"top_p": 0.5}, enable_thinking=True, seed=1)
    assert partial.profile is Qwen38SamplingProfile.CUSTOM and (partial.temperature, partial.top_p, partial.top_k) == (
        1.0,
        0.5,
        20,
    )


def test_present_temperature_uses_openai_defaults_with_the_card_top_k() -> None:
    parameters = step.parameters_from_request(
        {"temperature": 0.6, "seed": 5, "repetition_penalty": 1.1}, enable_thinking=True, seed=1
    )
    assert (parameters.temperature, parameters.top_p, parameters.top_k, parameters.presence_penalty) == (
        0.6,
        1.0,
        20,
        0.0,
    )
    assert parameters.repetition_penalty == 1.1 and parameters.seed == 5


def test_temperature_zero_and_greedy_take_the_greedy_loop() -> None:
    assert step.parameters_from_request({"temperature": 0}, enable_thinking=False, seed=1) is None
    assert step.parameters_from_request({"temperature": 0.0, "top_k": 5}, enable_thinking=True, seed=1) is None
    assert step.parameters_from_request({"greedy": True}, enable_thinking=True, seed=1) is None
    assert step.parameters_from_request({"greedy": True, "temperature": 0}, enable_thinking=True, seed=1) is None
    assert step.parameters_from_request({"greedy": False}, enable_thinking=True, seed=1) is None  # asks for nothing
    assert step.parameters_from_request({"greedy": False, "top_p": 0.5}, enable_thinking=True, seed=1) is not None


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"top_k": 33}, "top_k must be at most 32"),
        ({"temperature": "hot"}, "temperature must be a number"),
        ({"top_k": 2.5}, "top_k must be an integer"),
        ({"n": 2}, "n must be 1"),
        ({"seed": -1}, "seed must be in"),
        ({"temperature": -0.1}, "temperature must be nonnegative"),
        ({"top_p": 0.0}, "top_p must be in"),
        ({"presence_penalty": 3}, "presence_penalty must be in"),
        ({"min_p": 2}, "min_p must be in"),
        ({"greedy": "yes"}, "greedy must be a boolean"),
        ({"greedy": True, "temperature": 0.7}, "greedy is true but temperature is 0.7"),
    ],
)
def test_bad_fields_are_request_errors_naming_the_field_first(document, message) -> None:
    with pytest.raises(step.Qwen38SamplingRequestError, match=message) as info:
        step.parameters_from_request(document, enable_thinking=True, seed=0)
    assert str(info.value).split(" ", 1)[0] in document


def test_logprobs_fields_and_the_content_item_shape() -> None:
    assert step.logprobs_from_request({}) == (False, 0)
    assert step.logprobs_from_request({"logprobs": True, "top_logprobs": 5}) == (True, 5)
    for document in ({"logprobs": "yes"}, {"logprobs": True, "top_logprobs": 21}, {"top_logprobs": 2}):
        with pytest.raises(step.Qwen38SamplingRequestError):
            step.logprobs_from_request(document)
    sample = sample_full_vocabulary(_logits_for([1]).to(torch.float32), THINKING, top_logprobs=2)
    item = step.logprobs_content_item(sample, sample.token_id, lambda token: f"<{token}>")
    assert item["token"] == f"<{sample.token_id}>" and item["logprob"] == sample.logprob
    assert item["bytes"] == list(f"<{sample.token_id}>".encode()) and len(item["top_logprobs"]) == 2
    forced = step.logprobs_content_item(None, THINK_END_ID, lambda token: "</think>")
    assert forced == {"token": "</think>", "logprob": None, "bytes": list(b"</think>"), "top_logprobs": []}
    assert set(step.SAMPLING_REQUEST_FIELDS) >= {"temperature", "greedy", "top_p", "top_k", "seed", "logprobs", "n"}


# --- the chain extension's shape ------------------------------------------------------------------------------------


def test_chain_extension_runs_the_greedy_epilogue_unchanged_before_the_row() -> None:
    body = inspect.getsource(step.Qwen38SamplingChainExtension.capture_epilogue)
    greedy = (
        "greedy_candidates(trace_output.logits)",
        "resolve_greedy_on_device(candidates)",
        "ttnn.copy(trace_token_row, token_row_io)",
    )
    positions = [body.index(fragment) for fragment in greedy]
    assert positions == sorted(positions) and body.index("sampling_candidates(") > positions[-1]
    assert step.RESIDUE_CLASSES == 4
    primitives = ("read_candidate_row", "read_full_logits", "warm", "release", "mark_corruptible")
    assert all(hasattr(step.Qwen38SamplingChainExtension, name) for name in primitives)
    assert not hasattr(step.Qwen38SamplingChainExtension, "read_candidate_row_nonblocking")
    assert "ttnn.to_torch(ttnn.get_device_tensors(self.constants.readback_row)[0])" in inspect.getsource(
        step.Qwen38SamplingChainExtension.read_candidate_row
    )
    assert "ttnn.deallocate(gathered)" in inspect.getsource(step.Qwen38SamplingChainExtension._gather)
    warm = inspect.getsource(step.Qwen38SamplingChainExtension.warm)
    assert "sampling_candidates(logits, self.constants)" in warm and "self._gather(logits)" in warm
    assert "ids_equal_up_to_boundary_ties" in warm and "raise RuntimeError" in warm
