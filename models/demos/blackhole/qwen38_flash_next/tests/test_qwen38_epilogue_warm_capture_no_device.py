"""The TAIL epilogue's warm pass and its trace capture ask for the same programs (no device).

A program cache miss inside a trace capture is fatal, and a fused program's cache key is its compile-time form (its
kernels' compile-time arguments such as greedy_tail's ``copy_into`` and each io tensor's accessor placement), not the
buffer addresses of a call.  So the epilogue's "program key" here is the form of each call: which resolve form the
greedy row takes and which tensors, by role, the sampler reads and writes.  The session's warm pass and its capture
must agree for every fused epilogue the registry can serve: the greedy tail (pinned in the session's source) and the
device sampler (driven here with fakes through the extension's own ``warm`` and ``capture_epilogue``).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling_step as step

TOKEN_ROW = "fp32 TILE DRAM [1,1,1,32]"


class FakeTensor:
    def __init__(self, role: str, placement: str = TOKEN_ROW):
        self.role = role
        self.placement = placement


class FakeLMHead:
    """Records the resolve form; a fresh row comes back from every resolve, the ``into`` row receives its copy."""

    def __init__(self):
        self.resolves: list[str] = []

    def greedy_candidates(self, logits):
        return SimpleNamespace(local_values=FakeTensor("greedy values"), local_indices=FakeTensor("greedy indices"))

    def resolve_greedy_on_device(self, candidates, *, into=None):
        self.resolves.append("into=None" if into is None else f"into={into.role}")
        return FakeTensor("resolved token row")

    def sampling_candidates(self, logits, constants):
        return FakeTensor("candidate row", "fp32 ROW_MAJOR DRAM [1,1,1,256]")


class FakeSamplerConstants:
    def __init__(self):
        self.policies: list = []

    def write_policy(self, policy):
        self.policies.append(policy)

    def write_uniform(self, uniform):
        pass


class FakeTTNN:
    def __init__(self):
        self.copies: list[tuple[str, str]] = []

    def deallocate(self, tensor):
        pass

    def copy(self, source, target):
        self.copies.append((source.placement, target.placement))


class FakeCandidateRow:
    values = ids = None

    def agreement(self, other):
        return {"values_bitwise": [True], "ids_equal_up_to_boundary_ties": [True]}

    def to_host_row(self):
        return None


def extension_with_fakes(monkeypatch):
    """The extension without a device: the sampler set, every device read replaced, the sample call recorded."""

    ext = object.__new__(step.Qwen38SamplingChainExtension)
    ext.lm_head = FakeLMHead()
    ext.constants = SimpleNamespace(readback_row=FakeTensor("readback row"))
    ext.sampler = FakeSamplerConstants()
    ext.presence_on_device = False
    ext.trace_rows, ext.trace_logits = [], []
    keys: list[tuple[str, str, str]] = []

    def sample(row, greedy_row, constants):
        keys.append((row.role, greedy_row.role, ext.lm_head.resolves[-1]))
        return FakeTensor("token row")

    ext.sample = sample
    monkeypatch.setattr(step, "ttnn", FakeTTNN())
    monkeypatch.setattr(ext, "_gather", lambda logits: SimpleNamespace(to=lambda dtype: None))
    monkeypatch.setattr(ext, "read_candidate_row", lambda: FakeCandidateRow())
    monkeypatch.setattr(ext, "_token_of", lambda row: 7)
    monkeypatch.setattr(step, "candidate_row_lanes", lambda host_row: (None, None))
    monkeypatch.setattr(step, "device_sampler_reference", lambda *args, **kwargs: SimpleNamespace(token_id=7))
    monkeypatch.setattr(step.Qwen38CandidateRow, "emulate", staticmethod(lambda full: None))
    return ext, keys


def test_device_sampler_warm_and_capture_ask_for_the_same_programs(monkeypatch):
    ext, keys = extension_with_fakes(monkeypatch)
    token_row_io = FakeTensor("resident token row")
    ext.warm(logits=None, token_row_io=token_row_io, label="warm")
    warm_keys, warm_copies = set(keys), set(step.ttnn.copies)
    keys.clear()
    step.ttnn.copies.clear()
    trace_output = SimpleNamespace(logits=FakeTensor("logits"))
    candidates, trace_token_row = ext.capture_epilogue(trace_output, token_row_io)
    assert set(keys) == warm_keys == {("candidate row", "resolved token row", "into=resident token row")}
    assert ext.lm_head.resolves == ["into=resident token row"] * 3  # the warm's two passes, then the capture
    assert set(step.ttnn.copies) == warm_copies == {(TOKEN_ROW, TOKEN_ROW)}  # the token row into the resident row
    assert trace_token_row.role == "token row" and ext.trace_rows[-1].role == "candidate row"


def test_greedy_tail_warm_and_capture_share_the_resolve_form():
    """The greedy epilogue: the warm step and both captures resolve into the resident row (the MTP body keeps its
    own into=None form in the warm and in its capture)."""

    source = inspect.getsource(session)
    assert "into=None if chain_mtp is not None else token_row_io" in source  # the warm step
    assert "trace_token_row = lm_head.resolve_greedy_on_device(candidates, into=token_row_io)" in source  # the capture
    assert "trace_token_row = lm_head.resolve_greedy_on_device(candidates)" in source  # the MTP capture, into=None
    epilogue = inspect.getsource(
        step.Qwen38SamplingChainExtension._epilogue
    )  # one function, eager in the warm and captured
    assert epilogue.count("resolve_greedy_on_device(candidates, into=token_row_io)") == 1
    assert "resolve_greedy_on_device(candidates)" not in epilogue


def test_the_warm_pass_hands_the_extension_the_resident_row():
    """The session passes the resident token row (holding the warm step's resolved token) to ``warm``, which runs the
    capture's epilogue on it."""

    source = inspect.getsource(session)
    assert "chain.sampling.warm(output.logits, token_row_io, label=" in source
