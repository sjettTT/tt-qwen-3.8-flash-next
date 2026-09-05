# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible chat server over the single-trace decode chain (one device, requests queued in arrival order).

``POST /v1/chat/completions`` (streaming SSE or one JSON document; tools,
``reasoning_content``, stop strings, a per-request thinking budget; with
``--sampling`` also sampling with the OpenAI fields plus ``top_k``, ``min_p``,
``repetition_penalty`` and ``greedy``, ``seed`` echoed, ``logprobs`` from the
candidate row), ``GET /v1/models``, ``GET /health``.  On a sampling server a
request without ``temperature`` takes the model card's profile for its thinking
mode; ``temperature 0`` or ``greedy`` is the bitwise greedy loop.  Without
``--sampling`` (the default, ``--no-sampling`` the explicit form) TAIL captures
no candidate row, the loop is the greedy one at its measured period and explicit
sampling fields are refused.  Runs under ``tools/run_qwen38_chat_server.sh``
(or a development launcher): the server admits the runtime it
imports (the ttnn built from this checkout, ``runtime_admission``), prepares the
model inputs on the CPU, opens the mesh, converts the routed experts on the
first start, replays the CPU acceptance records after the captures
(``--sampling-discriminator`` then runs the sampling chain arms and stops),
writes READY, serves until SIGTERM, then releases the chain and the mesh in the
timing runner's order.  ``--host`` is loopback unless the profile serves the
LAN (the QuietBox, the LoudBox) or ``--allow-lan`` is given.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import http.server
import itertools
import json
import os
import secrets
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import ttnn
from models.demos.blackhole.qwen38_flash_next.chat import (
    EOS_TOKEN_IDS,
    PINNED_TOKENIZER_ARTIFACTS,
    VOCAB_SIZE,
    Qwen38OfficialChatTemplate,
)
from models.demos.blackhole.qwen38_flash_next.tools import hardware_profiles
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_protocol as protocol
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling_step as sampling_step
from models.demos.blackhole.qwen38_flash_next.tools import resident_decode, runtime_admission
from models.demos.blackhole.qwen38_flash_next.tools.evidence_records import (
    append_marker,
    append_phase_record,
    utc_now,
    write_result,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_protocol import Qwen38ChatRequestRejected
from models.demos.blackhole.qwen38_flash_next.tools.live_decode_diagnostic import (
    construct_live_decode_diagnostic,
    missing_bf4_layers,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import (
    DEFAULT_PREFILL_MODE,
    MAX_TOKENS_BOUND,
    MTP_DRAFTS,
    MTP_GDN_ANCHORS,
    PREFILL_MODES,
    Qwen38ChatChainError,
    Qwen38ChatRequestError,
    Qwen38ChatSession,
    construct_chain,
    mtp_capacity_admission,
    open_partition_b_mesh,
    resolve_route,
    template_decoder,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import (
    RESIDENT_MAX_QSA_CACHE_CAPACITY,
    RESIDENT_QSA_CACHE_CAPACITIES,
    Qwen38ResidentContext,
)

MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
ACCEPTANCE_CONTINUATION = 96
ACCEPTANCE_GATE_PROMPT = "json"
ACCEPTANCE_HANDOFF_SPLIT = 40  # MTP chains: the gate record replayed as 40 tokens in one mode then 56 in the other
DISCRIMINATOR_PROMPT = "story"
MAX_REQUEST_BYTES = 16 << 20
# Decision 6 (2026-09-03): thinking ON for Hermes, medium when the client sends no effort.
ENABLE_THINKING_DEFAULT = True
REASONING_EFFORT_DEFAULT = "medium"
# A request without max_tokens gets the remaining context (context limit less the prompt) and an explicit one is
# bounded by it: the session's require_budget, once the prompt is rendered.  /health and /v1/models say so.
MAX_TOKENS_RULE = {"default_max_tokens": "remaining context", "max_tokens_limit": "remaining context"}
QUEUE_LIMIT = 4
RETRY_AFTER_SECONDS = 5
HEARTBEAT_SECONDS = 30.0
# The template flags a client may send under chat_template_kwargs (the vLLM/SGLang convention the model card uses).
TEMPLATE_KWARGS = ("enable_thinking", "reasoning_effort", "preserve_thinking")
# Every field parse_chat_request reads; any other top-level key is logged as ignored.
KNOWN_REQUEST_FIELDS = frozenset(
    (
        "model",
        "messages",
        "tools",
        "tool_choice",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "response_format",
        "parallel_tool_calls",
        "logit_bias",
        "user",
        "chat_template_kwargs",
        "enable_thinking",
        "reasoning_effort",
        "thinking_budget",
        "ignore_eos",
        "prefill_mode",
    )
    + sampling_step.SAMPLING_REQUEST_FIELDS
)
# A request's seed when it sends none: 63 random bits, echoed in the response.
SEED_BITS = 63
PROFILER_VARIABLES = ("TT_METAL_DEVICE_PROFILER", "TT_METAL_PROFILER_CPP_POST_PROCESS", "TTNN_OP_PROFILER")


class Qwen38ChatServerStop(RuntimeError):
    """SIGTERM or SIGINT: the launcher's timeout or the user; a clean shutdown."""


def _handle_stop_signal(signum: int, _frame: Any) -> None:
    raise Qwen38ChatServerStop(f"chat server received signal {signum}")


def _log(event: str, **fields: Any) -> None:
    print(json.dumps({"utc": utc_now(), "event": event, **fields}, sort_keys=True), flush=True)


# -- requests and responses ----------------------------------------------------------------------


def parse_chat_request(document: Any, *, seed: int | None = None, sampling_available: bool = True) -> dict[str, Any]:
    """The fields the server honours, validated with actual-vs-expected messages (one completion; the sampling
    fields per ``qwen38_sampling_step``, ``extra_body`` merged under the top level, JSON null an absent field,
    ``chat_template_kwargs`` the thinking flags' other spelling).  ``seed`` is the server's draw for a request that
    sends none.  A greedy-only server (``sampling_available`` false) refuses explicit sampling fields and runs a
    request without them greedily.  A field the server cannot honour (structured output, logit_bias, serial tool
    calls) is refused, never dropped.  Raises ``ValueError`` subclasses carrying ``param`` and ``code``."""

    if not isinstance(document, Mapping):
        raise Qwen38ChatRequestRejected(f"request body must be a JSON object, got {type(document).__name__}")
    extra = document.get("extra_body")
    if extra is not None:
        if not isinstance(extra, Mapping):
            raise Qwen38ChatRequestRejected(
                f"extra_body must be an object, got {type(extra).__name__}", param="extra_body"
            )
        document = {**extra, **{key: value for key, value in document.items() if key != "extra_body"}}
    document = {key: value for key, value in document.items() if value is not None}
    template_kwargs = document.get("chat_template_kwargs", {})
    if not isinstance(template_kwargs, Mapping):
        raise Qwen38ChatRequestRejected(
            f"chat_template_kwargs must be an object, got {type(template_kwargs).__name__}",
            param="chat_template_kwargs",
        )
    for key, value in template_kwargs.items():
        if key not in TEMPLATE_KWARGS:
            raise Qwen38ChatRequestRejected(
                f"chat_template_kwargs.{key} is not a template flag of this server (the flags are {TEMPLATE_KWARGS})",
                param=f"chat_template_kwargs.{key}",
            )
        if key == "preserve_thinking" and value is not True:
            raise Qwen38ChatRequestRejected(
                f"chat_template_kwargs.preserve_thinking must be true (the server keeps the reasoning), got {value!r}",
                param="chat_template_kwargs.preserve_thinking",
            )
        if key in document and document[key] != value:
            raise Qwen38ChatRequestRejected(
                f"chat_template_kwargs.{key} {value!r} contradicts {key} {document[key]!r}",
                param=f"chat_template_kwargs.{key}",
            )
    document = {**{key: template_kwargs[key] for key in TEMPLATE_KWARGS[:2] if key in template_kwargs}, **document}
    messages = protocol.normalize_messages(document.get("messages"))
    tool_choice = document.get("tool_choice", "auto")
    if tool_choice not in protocol.TOOL_CHOICES:
        raise Qwen38ChatRequestRejected(
            f"tool_choice must be one of {protocol.TOOL_CHOICES} (greedy server), got {tool_choice!r}",
            param="tool_choice",
        )
    tools = protocol.validate_tools(document.get("tools")) if tool_choice == "auto" else []
    stream = document.get("stream", False)
    if type(stream) is not bool:
        raise Qwen38ChatRequestRejected(f"stream must be a boolean, got {stream!r}", param="stream")
    stream_options = document.get("stream_options", {})
    if not isinstance(stream_options, Mapping) or set(stream_options) - {"include_usage"}:
        raise Qwen38ChatRequestRejected(
            f"stream_options may only hold include_usage (usage rides on the final chunk), got {stream_options!r}",
            param="stream_options",
        )
    response_format = document.get("response_format", {"type": "text"})
    kind = response_format.get("type") if isinstance(response_format, Mapping) else response_format
    if kind != "text":
        raise Qwen38ChatRequestRejected(
            f"response_format.type must be 'text' (this server has no constrained decoding), got {kind!r}",
            param="response_format",
        )
    if document.get("logit_bias", {}) != {}:
        raise Qwen38ChatRequestRejected(
            f"logit_bias is not applied by this server: drop it, got {document.get('logit_bias')!r}", param="logit_bias"
        )
    if document.get("parallel_tool_calls", True) is not True:
        raise Qwen38ChatRequestRejected(
            f"parallel_tool_calls must be true (every call the model emits is returned), "
            f"got {document.get('parallel_tool_calls')!r}",
            param="parallel_tool_calls",
        )
    if not isinstance(document.get("user", ""), str):
        raise Qwen38ChatRequestRejected(f"user must be a string, got {document.get('user')!r}", param="user")
    if document.get("n", 1) != 1:
        raise Qwen38ChatRequestRejected(f"n must be 1 (one greedy completion), got {document.get('n')!r}", param="n")
    # None (neither name sent) stays None: the session resolves the remaining context once the prompt is known.
    max_tokens = document.get("max_tokens", document.get("max_completion_tokens"))
    if max_tokens is not None and (type(max_tokens) is not int or not 1 <= max_tokens <= MAX_TOKENS_BOUND):
        raise Qwen38ChatRequestRejected(
            f"max_tokens must be an integer in [1, {MAX_TOKENS_BOUND}] when given, got {max_tokens!r}",
            param="max_tokens",
        )
    enable_thinking = document.get("enable_thinking", ENABLE_THINKING_DEFAULT)
    if type(enable_thinking) is not bool:
        raise Qwen38ChatRequestRejected(
            f"enable_thinking must be a boolean, got {enable_thinking!r}", param="enable_thinking"
        )
    reasoning_effort = protocol.effort_level(document.get("reasoning_effort", REASONING_EFFORT_DEFAULT))
    requested_budget = document.get("thinking_budget")
    if requested_budget is not None and (type(requested_budget) is not int or requested_budget < 0):
        raise Qwen38ChatRequestRejected(
            f"thinking_budget must be a non-negative integer, got {requested_budget!r}", param="thinking_budget"
        )
    ignore_eos = document.get("ignore_eos", False)
    if type(ignore_eos) is not bool:
        raise Qwen38ChatRequestRejected(f"ignore_eos must be a boolean, got {ignore_eos!r}", param="ignore_eos")
    prefill_mode = document.get("prefill_mode")
    if prefill_mode is not None and prefill_mode not in PREFILL_MODES:
        raise Qwen38ChatRequestRejected(
            f"prefill_mode must be one of {PREFILL_MODES} when given, got {prefill_mode!r}", param="prefill_mode"
        )
    try:
        sampling = sampling_step.parameters_from_request(
            document, enable_thinking=enable_thinking, seed=secrets.randbits(SEED_BITS) if seed is None else seed
        )
        logprobs, top_logprobs = sampling_step.logprobs_from_request(document)
    except sampling_step.Qwen38SamplingRequestError as error:
        raise Qwen38ChatRequestRejected(str(error), param=str(error).split(" ", 1)[0]) from error
    if logprobs and sampling is None:
        raise Qwen38ChatRequestRejected(
            "logprobs need a sampled request (temperature > 0): the greedy loop reads no candidate row",
            param="logprobs",
        )
    if sampling is not None and not sampling_available:
        explicit = [
            name
            for name in sampling_step.SAMPLING_REQUEST_FIELDS
            if name not in ("n", "greedy") and document.get(name) is not None
        ]
        if explicit:
            raise Qwen38ChatRequestRejected(
                f"sampling is unavailable on this server (greedy only): drop {explicit} or send temperature 0",
                param=explicit[0],
            )
        sampling = None
    return {
        "messages": messages,
        "tools": tools,
        "stream": stream,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
        "reasoning_effort": reasoning_effort,
        "thinking_budget": requested_budget,
        "stop": protocol.validate_stop(document.get("stop")),
        "ignore_eos": ignore_eos,
        "prefill_mode": prefill_mode,
        "sampling": sampling,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "ignored": sorted(set(document) - KNOWN_REQUEST_FIELDS),
    }


def _usage(prompt_tokens: int, completion_tokens: int, queue_wait: float) -> dict[str, Any]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "queue_wait_seconds": round(queue_wait, 4),
    }


def _extension(
    completion: Any,
    assembler: protocol.Qwen38ReplyAssembler,
    *,
    queue_wait: float,
    spliced: bool,
    think_budget: int | None,
    sampling: sampling_step.Qwen38SamplingRequest | None,
) -> dict[str, Any]:
    """The ``qwen38`` object of a response (and the ledger's per-request fields)."""

    return {
        **completion.as_dict(),
        "finish": completion.finish_reason,
        "queue_wait_seconds": round(queue_wait, 4),
        "prompt_spliced": spliced,
        "sampling": None if sampling is None else sampling.as_dict(),
        "seed": None if sampling is None else sampling.parameters.seed,
        "reasoning_tokens": assembler.reasoning_tokens,
        "thinking_forced": protocol.THINK_END_ID in completion.token_ids
        and think_budget is not None
        and assembler.reasoning_tokens >= think_budget,
        "stop_string_hit": assembler.stop_hit,
        "tool_calls": len(assembler.calls),
        "tool_parse_errors": assembler.parse_errors,
        "truncated_tool_call": assembler.truncated_tool_call,
    }


def _completion_id() -> str:
    return f"chatcmpl-{secrets.token_hex(8)}"


def _vmrss_kib() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


# OpenAI finish_reason from the session's reason and the assembled reply.
def _finish_reason(session_finish: str, assembler: protocol.Qwen38ReplyAssembler) -> str:
    if assembler.stop_hit:
        return "stop"
    if assembler.calls:
        return "tool_calls"
    return {"deadline": "length", "disconnected": "stop"}.get(session_finish, session_finish)


class Qwen38ServerBusy(RuntimeError):
    """The bounded request queue is full: HTTP 503 with Retry-After."""


class Qwen38ChatHTTPServer(http.server.ThreadingHTTPServer):
    """Threaded accept, parse and validate; one device at a time through a bounded FIFO turnstile.

    A request that fails validation is answered at once even while the device
    is busy; a valid one waits its turn behind at most ``queue_limit`` others
    (``queue_wait_seconds`` is reported) or is refused with 503 above the bound.
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        session: Qwen38ChatSession,
        *,
        ledger: Path,
        queue_limit: int = QUEUE_LIMIT,
        request_deadline_seconds: float | None = None,
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
        system_fingerprint: str | None = None,
        dram_after_captures: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(address, Qwen38ChatHandler)
        self.session = session
        self.ledger = ledger
        self.queue_limit = queue_limit
        self.request_deadline_seconds = request_deadline_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.system_fingerprint = system_fingerprint  # source head + runtime .so: what a seed reproduces against
        # The mesh allocator read after the traces were captured (hardware_profiles.symmetric_mesh_dram_memory: one observation
        # that applies to every card): free_bytes_per_bank is the build's headroom, reported as read, never updated.
        self.dram_after_captures = dram_after_captures
        self.created = int(time.time())  # /v1/models created: when this server came up
        self.turnstile = threading.Condition()
        self.waiting: collections.deque[int] = collections.deque()
        self.tickets = itertools.count()
        self.busy = False
        self.served: protocol.Qwen38ServedTurn | None = None
        self.last_prefill_ms_per_token: float | None = None
        self.fatal: BaseException | None = None

    @property
    def queue_depth(self) -> int:
        return len(self.waiting)

    def acquire_device(self) -> float:
        """Wait for the device in arrival order; returns the seconds waited."""

        with self.turnstile:
            if len(self.waiting) >= self.queue_limit:
                raise Qwen38ServerBusy(f"{len(self.waiting)} requests are queued, the limit is {self.queue_limit}")
            ticket = next(self.tickets)
            self.waiting.append(ticket)
            started = time.perf_counter()
            self.turnstile.wait_for(lambda: not self.busy and self.waiting[0] == ticket)
            self.waiting.popleft()
            self.busy = True
            return time.perf_counter() - started

    def release_device(self) -> None:
        with self.turnstile:
            self.busy = False
            self.turnstile.notify_all()

    def service_actions(self) -> None:
        # Called once per serve_forever iteration: a device failure inside a
        # request ends the process (a poisoned model owner cannot serve).
        if self.fatal is not None:
            raise self.fatal


class Qwen38ChatHandler(http.server.BaseHTTPRequestHandler):
    server_version = "qwen38-chat/2"
    protocol_version = "HTTP/1.0"
    server: Qwen38ChatHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        _log("http", client=self.address_string(), line=format % args)

    def _send_json(self, status: int, document: Mapping[str, Any], **headers: str) -> None:
        body = json.dumps(document).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(
        self, status: int, message: str, kind: str, *, code: str | None = None, param: str | None = None, **headers: str
    ) -> None:
        self._send_json(
            status, {"error": {"message": message, "type": kind, "param": param, "code": code or kind}}, **headers
        )

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        session = self.server.session
        if path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_ID,
                            "object": "model",
                            "created": self.server.created,
                            "owned_by": "tenstorrent",
                            "context_length": session.context_limit,
                            "max_model_len": session.context_limit,
                            "limits": {
                                **MAX_TOKENS_RULE,
                                "thinking_token_caps": protocol.THINKING_TOKEN_CAPS,
                                "answer_reserve_tokens": protocol.ANSWER_RESERVE_TOKENS,
                            },
                        }
                    ],
                },
            )
        elif path == "/health":
            self._send_json(
                200,
                {
                    "status": "ready",
                    "model": MODEL_ID,
                    "busy": self.server.busy,
                    "queue_depth": self.server.queue_depth,
                    "committed_tokens": len(session.committed),
                    "requests_served": session.requests_served,
                    "last_tokens_per_second": session.last_tokens_per_second,
                    "last_prefill_ms_per_token": self.server.last_prefill_ms_per_token,
                    "context_limit": session.context_limit,
                    "prefill_mode": session.prefill_mode,
                    "chunk_trace_available": session.chunk_trace_available,
                    "limits": {
                        "context_length": session.context_limit,
                        **MAX_TOKENS_RULE,
                        "max_tokens_bound": MAX_TOKENS_BOUND,
                        "stop_strings": protocol.MAX_STOP_STRINGS,
                        "queue_depth": self.server.queue_limit,
                        "request_deadline_seconds": self.server.request_deadline_seconds,
                        "request_bytes": MAX_REQUEST_BYTES,
                    },
                    "defaults": {
                        "enable_thinking": ENABLE_THINKING_DEFAULT,
                        "reasoning_effort": REASONING_EFFORT_DEFAULT,
                        "thinking_token_caps": protocol.THINKING_TOKEN_CAPS,
                        "answer_reserve_tokens": protocol.ANSWER_RESERVE_TOKENS,
                    },
                    "supports": ["tools", "streaming", "reasoning_content", "stop", "thinking_budget", "ignore_eos"]
                    + (["sampling", "seed", "logprobs"] if session.sampling is not None else []),
                    "sampling": "greedy" if session.sampling is None else "candidate_row_host_sampler",
                    # logprobs are relative to the read candidates (above the vocabulary's by -log of the row's mass).
                    "logprobs_normalizer": None if session.sampling is None else "candidate_row",
                    "mtp": None if session.mtp is None else session.mtp.summary(),
                    "sampling_defaults": {
                        "thinking": sampling_step.parameters_as_dict(
                            sampling_step.Qwen38SamplingParameters.official_thinking(seed=0)
                        )
                        | {"seed": "random"},
                        "non_thinking": sampling_step.parameters_as_dict(
                            sampling_step.Qwen38SamplingParameters.official_non_thinking(seed=0)
                        )
                        | {"seed": "random"},
                        "top_k_limit": sampling_step.CANDIDATE_TOP_K_LIMIT,
                        "top_logprobs_limit": sampling_step.MAX_TOP_LOGPROBS,
                    },
                    "system_fingerprint": self.server.system_fingerprint,
                    "host_vmrss_kib": _vmrss_kib(),
                    "program_cache_entries": getattr(session.chain, "program_cache_entries", None),
                    "dram_after_captures": self.server.dram_after_captures,
                },
            )
        else:
            self._send_error_json(404, f"no such path: {path}", "not_found")

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path != "/v1/chat/completions":
            self._send_error_json(404, f"no such path: {path}", "not_found")
            return
        length = self.headers.get("Content-Length")
        if length is None or not length.isdigit() or int(length) > MAX_REQUEST_BYTES:
            self._send_error_json(
                400,
                f"Content-Length must be a number up to {MAX_REQUEST_BYTES}, got {length!r}",
                "invalid_request_error",
            )
            return
        session = self.server.session
        flags: dict[str, Any] = {}
        try:
            request = parse_chat_request(
                json.loads(self.rfile.read(int(length)).decode("utf-8")),
                sampling_available=session.sampling is not None,
            )
            flags = {"enable_thinking": request["enable_thinking"], "reasoning_effort": request["reasoning_effort"]}
            # The reference render: validates the whole request and bounds the budget before any queueing.
            rendered_ids = session.render(request["messages"], tools=request["tools"], **flags)
            session.require_budget(len(rendered_ids), request["max_tokens"])
        except (ValueError, UnicodeDecodeError) as error:
            code = getattr(error, "code", None)
            if code is None:
                code = "context_length_exceeded" if str(error).startswith("context_length_exceeded") else "bad_request"
            self._send_error_json(
                400, str(error), "invalid_request_error", code=code, param=getattr(error, "param", None)
            )
            return
        except Exception as error:  # noqa: BLE001  a validation defect: answered, not a dropped connection
            _log(
                "request_validation_failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()
            )
            self._send_error_json(500, f"{type(error).__name__}: {error}", "server_error")
            return
        if request["ignored"]:
            _log("ignored_request_fields", fields=request["ignored"])
        received_utc = utc_now()
        try:
            queue_wait = self.server.acquire_device()
        except Qwen38ServerBusy as error:
            self._send_error_json(503, str(error), "server_busy", Retry_After=str(RETRY_AFTER_SECONDS))
            return
        request_id = _completion_id()
        assembler = protocol.Qwen38ReplyAssembler(
            template_decoder(session.template),
            thinking_open=request["enable_thinking"],
            stop_strings=request["stop"],
            tools=request["tools"],
        )
        try:
            prompt_ids = protocol.splice_prompt(
                session.template.tokenizer, self.server.served, request["messages"], request["tools"], **flags
            )
            spliced = prompt_ids is not None
            # The budget is known only now: the served prompt's remaining context when the client sent no
            # max_tokens; the thinking budget is carved out of it.
            if spliced:
                try:
                    max_tokens = session.require_budget(len(prompt_ids), request["max_tokens"])
                except Qwen38ChatRequestError as error:
                    # The spliced prompt carries the served turns' reasoning, which the client's history does not;
                    # when it no longer fits, the reference render (within the budget above) resets the device.
                    _log(
                        "splice_over_budget",
                        request_id=request_id,
                        spliced_tokens=len(prompt_ids),
                        rendered_tokens=len(rendered_ids),
                        error=str(error),
                    )
                    spliced = False
            if not spliced:
                prompt_ids = rendered_ids
                max_tokens = session.require_budget(len(prompt_ids), request["max_tokens"])
        except ValueError as error:
            self.server.release_device()
            self._send_error_json(400, str(error), "invalid_request_error", code="context_length_exceeded")
            return
        request = {
            **request,
            "max_tokens_requested": request["max_tokens"],
            "max_tokens": max_tokens,
            "think_budget": (
                protocol.thinking_budget(request["reasoning_effort"], max_tokens, request["thinking_budget"])
                if request["enable_thinking"]
                else None
            ),
        }
        streaming_started = False
        heartbeat_stop = threading.Event()
        heartbeats = [0]
        started = time.perf_counter()
        deadline = self.server.request_deadline_seconds

        def heartbeat() -> None:
            while not heartbeat_stop.wait(self.server.heartbeat_seconds):
                heartbeats[0] += 1
                _log(
                    "heartbeat",
                    request_id=request_id,
                    elapsed_seconds=round(time.perf_counter() - started, 3),
                    committed_tokens=len(session.committed),
                    emitted_tokens=assembler.tokens,
                    reasoning_tokens=assembler.reasoning_tokens,
                    tool_calls=len(assembler.calls),
                    host_vmrss_kib=_vmrss_kib(),
                )

        def should_stop() -> str | None:
            if assembler.stop_hit:
                return "stop"
            if deadline is not None and time.perf_counter() - started > deadline:
                return "deadline"
            return None

        threading.Thread(target=heartbeat, name=f"heartbeat-{request_id}", daemon=True).start()
        sampling = (
            None
            if request["sampling"] is None
            else sampling_step.Qwen38SamplingRequest(request["sampling"], top_logprobs=request["top_logprobs"])
        )
        extension_of = lambda completion: _extension(  # noqa: E731
            completion,
            assembler,
            queue_wait=queue_wait,
            spliced=spliced,
            think_budget=request["think_budget"],
            sampling=sampling,
        )
        # One logprobs item per token the client sees (the sampled loop appends its sample before yielding).
        decode_one = lambda token_id: template_decoder(session.template)([token_id])  # noqa: E731
        logprobs_of = lambda token_id: sampling_step.logprobs_content_item(  # noqa: E731
            sampling.samples[-1], token_id, decode_one
        )
        logprob_items: list[dict[str, Any]] = []

        def on_token(token_id: int) -> None:
            if request["logprobs"]:
                logprob_items.append(logprobs_of(token_id))
            assembler.push(token_id)

        try:
            run = {
                "stop_ids": () if request["ignore_eos"] else EOS_TOKEN_IDS,
                "think_budget": request["think_budget"],
                "should_stop": should_stop,
                "prefill_mode": request["prefill_mode"],
                "sampling": sampling,
            }
            if request["stream"]:
                self._send_stream_head()
                streaming_started = True
                completion = self._stream(
                    session,
                    prompt_ids,
                    request,
                    assembler,
                    request_id,
                    run,
                    queue_wait,
                    extension_of,
                    logprobs_of if request["logprobs"] else None,
                )
            else:
                completion = session.complete(prompt_ids, request["max_tokens"], on_token=on_token, **run)
                assembler.finish()
        except Exception as error:  # noqa: BLE001  the device loop failed: report, then end the server
            _log(
                "request_failed",
                request_id=request_id,
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
            self.server.served = None
            if session.poisoned:
                self.server.fatal = error
            try:
                if streaming_started:
                    self._write_event(
                        {"error": {"message": f"{type(error).__name__}: {error}", "type": "server_error"}}
                    )
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    self._send_error_json(500, f"{type(error).__name__}: {error}", "server_error")
            except OSError:
                pass
            return
        finally:
            heartbeat_stop.set()
            self.server.release_device()
        finish_reason = _finish_reason(completion.finish_reason, assembler)
        # The next request continues this turn from the committed ids only when they are the reply the client will
        # echo: the prompt fully prefilled, no stop-string tail (the match's tokens are committed), the think block
        # closed, no tool block cut open.  A length finish leaves the reply's last token unconsumed in the row.
        continuable = (
            completion.finish_reason != "error"
            and len(session.committed) >= len(prompt_ids)
            and not assembler.stop_hit
            and assembler.phase == "content"
            and not assembler.truncated_tool_call
        )
        self.server.served = (
            protocol.Qwen38ServedTurn(
                messages=request["messages"],
                tools=request["tools"],
                enable_thinking=request["enable_thinking"],
                reasoning_effort=request["reasoning_effort"],
                reply=assembler.echo_reply(),
                committed=list(session.committed) + ([] if session.row_token is None else [session.row_token]),
            )
            if continuable
            else None
        )
        if completion.prefill_tokens:
            self.server.last_prefill_ms_per_token = round(
                1e3 * completion.prefill_seconds / completion.prefill_tokens, 3
            )
        extension = extension_of(completion)
        append_phase_record(
            self.server.ledger,
            {
                "phase": "chat-request",
                "request_id": request_id,
                "received_utc": received_utc,
                "stream": request["stream"],
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": len(completion.token_ids),
                "max_tokens": request["max_tokens"],
                "max_tokens_requested": request["max_tokens_requested"],
                "finish_reason": finish_reason,
                "text_characters": sum(len(piece) for piece in assembler.content),
                "reasoning_characters": sum(len(piece) for piece in assembler.reasoning),
                "enable_thinking": request["enable_thinking"],
                "reasoning_effort": request["reasoning_effort"],
                "think_budget": request["think_budget"],
                "tools_offered": len(request["tools"]),
                "ignore_eos": request["ignore_eos"],
                "logprobs": request["logprobs"],
                "deadline_seconds": deadline,
                "heartbeats": heartbeats[0],
                "host_vmrss_kib": _vmrss_kib(),
                "program_cache_entries": getattr(session.chain, "program_cache_entries", None),
                **extension,
                "position_after": completion.position,
            },
        )
        _log(
            "request",
            request_id=request_id,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=len(completion.token_ids),
            finish_reason=finish_reason,
            **extension,
        )
        if request["stream"]:
            return
        choice: dict[str, Any] = {"index": 0, "message": assembler.message(), "finish_reason": finish_reason}
        if request["logprobs"]:
            choice["logprobs"] = {"content": logprob_items}
        try:
            self._send_json(
                200,
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": MODEL_ID,
                    "system_fingerprint": self.server.system_fingerprint,
                    "choices": [choice],
                    "usage": _usage(completion.prompt_tokens, len(completion.token_ids), queue_wait),
                    "qwen38": extension,
                },
            )
        except OSError as error:
            _log("client_disconnected", request_id=request_id, error=f"{type(error).__name__}: {error}")

    def _send_stream_head(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _write_event(self, document: Mapping[str, Any]) -> None:
        self.wfile.write(b"data: " + json.dumps(document).encode("utf-8") + b"\n\n")
        self.wfile.flush()

    def _stream(
        self,
        session: Qwen38ChatSession,
        prompt_ids: list[int],
        request: Mapping[str, Any],
        assembler: protocol.Qwen38ReplyAssembler,
        completion_id: str,
        run: Mapping[str, Any],
        queue_wait: float,
        extension_of: Callable[[Any], dict[str, Any]],
        logprobs_of: Callable[[int], dict[str, Any]] | None,
    ) -> Any:
        """SSE in the chat.completion.chunk delta format: one chunk per assembled piece (reasoning_content, content,
        or a completed tool call), flushed per token; with ``logprobs_of`` every token's item rides on its first
        chunk (an empty delta when the assembler held the text back); the final chunk carries finish_reason, usage
        and qwen38."""

        created = int(time.time())

        def emit(choice: dict[str, Any], **extra: Any) -> None:
            self._write_event(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "system_fingerprint": self.server.system_fingerprint,
                    "choices": [{"index": 0, **choice}],
                    **extra,
                }
            )

        emit({"delta": {"role": "assistant", "content": ""}, "finish_reason": None})

        def on_token(token_id: int) -> None:
            deltas = assembler.push(token_id)
            if logprobs_of is not None:
                deltas = deltas or [{}]
                emit({"delta": deltas[0], "logprobs": {"content": [logprobs_of(token_id)]}, "finish_reason": None})
                deltas = deltas[1:]
            for delta in deltas:
                emit({"delta": delta, "finish_reason": None})

        # A client that went away raises OSError inside on_token; the session ends
        # the request as "disconnected" and the final chunk is not deliverable.
        completion = session.complete(prompt_ids, request["max_tokens"], on_token=on_token, **run)
        try:
            for delta in assembler.finish():
                emit({"delta": delta, "finish_reason": None})
            emit(
                {"delta": {}, "finish_reason": _finish_reason(completion.finish_reason, assembler)},
                usage=_usage(completion.prompt_tokens, len(completion.token_ids), queue_wait),
                qwen38=extension_of(completion),
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError as error:
            _log("client_disconnected", request_id=completion_id, error=f"{type(error).__name__}: {error}")
        return completion


# -- acceptance replay ---------------------------------------------------------------------------


def load_acceptance_records(directory: Path) -> list[dict[str, Any]]:
    """The CPU study's ``prompt-*-greedy.json`` records, digest-checked against the directory's SHA256SUMS."""

    directory = directory.resolve(strict=True)
    sums = {}
    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        sums[name.strip()] = digest
    records = []
    for path in sorted(directory.glob("prompt-*-greedy.json")):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if sums.get(path.name) != actual:
            raise ValueError(f"{path.name}: sha256 {actual} vs SHA256SUMS {sums.get(path.name)}")
        document = json.loads(path.read_text(encoding="utf-8"))
        prompt_ids = document.get("prompt_token_ids")
        generated = document.get("generated_token_ids")
        if (
            document.get("mode") != "greedy"
            or not isinstance(document.get("prompt"), str)
            or not isinstance(prompt_ids, list)
            or not isinstance(generated, list)
            or not prompt_ids
            or not generated
            or any(type(value) is not int or not 0 <= value < VOCAB_SIZE for value in prompt_ids + generated)
            or document.get("prompt_length") != len(prompt_ids)
            or document.get("stop_reason") not in ("eos", "max_continuation")
        ):
            raise ValueError(f"{path.name} is not a greedy prompt record with vocabulary-ranged token lists")
        records.append(
            {
                "prompt": document["prompt"],
                "prompt_token_ids": prompt_ids,
                "generated_token_ids": generated,
                "stop_reason": document["stop_reason"],
                "sha256": actual,
            }
        )
    if not records:
        raise ValueError(f"no prompt-*-greedy.json records in {directory}")
    # The gate prompt first: its 96/96 result is the go/no-go for serving.
    records.sort(key=lambda record: (record["prompt"] != ACCEPTANCE_GATE_PROMPT, record["prompt"]))
    return records


def _divergence(actual: Sequence[int], expected: Sequence[int]) -> int | None:
    divergence = next((index for index, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
    if divergence is None and len(actual) != len(expected):
        divergence = min(len(actual), len(expected))
    return divergence


def replay_acceptance(
    session: Qwen38ChatSession,
    records: list[dict[str, Any]],
    *,
    continuation: int = ACCEPTANCE_CONTINUATION,
    require_gate: bool,
    handoff_gate: bool = False,
    handoff_split: int = ACCEPTANCE_HANDOFF_SPLIT,
) -> dict[str, Any]:
    """Prefill each record's prompt ids, generate greedily, report the first index where the device differs.

    Stops are disabled so an early EOS record is compared over its whole CPU
    stream; the gate is the ``json`` record's full ``continuation`` match.
    ``handoff_gate`` (MTP chains) replays the gate record twice more, split at
    ``handoff_split`` tokens: the pass loop then the 1-row loop, and the 1-row
    loop then the pass loop, each half continuing the other's device state, so
    both mode switches must reproduce the CPU stream to count as exact.
    """

    results = []
    for record in records:
        expected = record["generated_token_ids"][:continuation]
        session.reset()
        completion = session.complete(record["prompt_token_ids"], len(expected), stop_ids=())
        actual = completion.token_ids
        divergence = _divergence(actual, expected)
        result = {
            "prompt": record["prompt"],
            "prompt_tokens": len(record["prompt_token_ids"]),
            "compared_tokens": len(expected),
            "divergence_index": divergence,
            "matched_tokens": len(expected) if divergence is None else divergence,
            "first_token_device": actual[0] if actual else None,
            "first_token_cpu": expected[0],
            "device_token_ids": actual,
            "cpu_token_ids": expected,
            "prefill_seconds": completion.prefill_seconds,
            "prefill_mode": completion.prefill_mode,
            "prefill_chunks": completion.prefill_chunks,
            "prefill_forced_tokens": completion.prefill_forced_tokens,
            "prefill_handoff_ms": completion.prefill_handoff_ms,
            "tokens_per_second": completion.tokens_per_second,
            "mtp": completion.mtp,
        }
        results.append(result)
        _log("acceptance_prompt", **{key: value for key, value in result.items() if not key.endswith("_ids")})
    gate = next((result for result in results if result["prompt"] == ACCEPTANCE_GATE_PROMPT), None)
    gate_pass = gate is not None and gate["divergence_index"] is None and gate["compared_tokens"] == continuation
    if require_gate and not gate_pass:
        raise Qwen38ChatChainError(
            f"acceptance gate: {ACCEPTANCE_GATE_PROMPT} matched "
            f"{None if gate is None else gate['matched_tokens']} of {continuation} CPU greedy tokens"
        )
    handoff = None
    if handoff_gate and gate is not None:
        record = next(record for record in records if record["prompt"] == ACCEPTANCE_GATE_PROMPT)
        expected = record["generated_token_ids"][:continuation]
        handoff = {"split": handoff_split, "orders": {}}
        for name, first_speculative in (("mtp_then_one_row", True), ("one_row_then_mtp", False)):
            session.reset()
            prompt = list(record["prompt_token_ids"])
            first = session.complete(prompt, handoff_split, stop_ids=(), speculative=first_speculative)
            # The second half extends the exact repeat: the first half's last token sits unconsumed in the row.
            second = session.complete(
                prompt + first.token_ids[:-1],
                len(expected) - handoff_split + 1,
                stop_ids=(),
                speculative=not first_speculative,
            )
            actual = first.token_ids[:-1] + second.token_ids
            handoff["orders"][name] = {
                "divergence_index": _divergence(actual, expected),
                "compared_tokens": len(expected),
                "second_half_prefix_reused": second.prefix_reused,
                "second_half_reset": second.reset,
                "first_half_mtp": first.mtp,
                "second_half_mtp": second.mtp,
                "device_token_ids": actual,
            }
            _log(
                "acceptance_handoff",
                order=name,
                **{k: v for k, v in handoff["orders"][name].items() if k != "device_token_ids"},
            )
        handoff["pass"] = all(order["divergence_index"] is None for order in handoff["orders"].values())
        if require_gate and not handoff["pass"]:
            raise Qwen38ChatChainError(
                f"acceptance hand-off gate: {ACCEPTANCE_GATE_PROMPT} split at {handoff_split} diverged: "
                f"{ {name: order['divergence_index'] for name, order in handoff['orders'].items()} }"
            )
    # The replays are not client requests: /health and result.json count clients from zero.
    session.requests_served = 0
    session.last_tokens_per_second = None
    return {
        "schema": "qwen38-chat-server-acceptance/v1",
        "continuation": continuation,
        "gate_prompt": ACCEPTANCE_GATE_PROMPT,
        "gate_pass": gate_pass,
        "prompts": results,
        "handoff": handoff,
    }


# -- main --------------------------------------------------------------------------------------

# The launcher's ``common`` block: the model inputs and caches; the runtime identity's arguments are ``runtime_admission``'s.
PATH_ARGUMENTS = (
    "checkpoint",
    "component-cache-root",
    "routed-bf4-scratch-root",
    "model-io-cache-root",
    "phase-log",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PATH_ARGUMENTS:
        parser.add_argument(f"--{name}", type=Path, required=True)
    runtime_admission.add_arguments(parser)
    parser.add_argument(
        "--bf4-corpus",
        type=Path,
        default=None,
        help="a CPU-staged BF4 expert corpus root (tools/stage_full_bf4_cpu.py); without it the production cache "
        "under --routed-bf4-scratch-root is used and the missing layers are converted on the first start",
    )
    parser.add_argument("--bf4-corpus-verification", type=Path, default=None, help="the corpus's verification.json")
    parser.add_argument(
        "--bf4-producer-identity",
        type=Path,
        default=None,
        help="JSON with the corpus's expected source_head, runtime and checkpoint (pins what the record claims)",
    )
    parser.add_argument(
        "--bf4-stage-limit",
        type=int,
        default=None,
        help="convert at most N missing BF4 layers this run (a host that bounds a job's wall time; resumable)",
    )
    parser.add_argument(
        "--cpu-oracle", type=Path, default=None, help="the two-step CPU oracle (default: the shipped one)"
    )
    parser.add_argument(
        "--device-nodes",
        default=None,
        help="four KMD device nodes for a profile on other chips of an eight-chip host, e.g. 4,5,6,7",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="open the mesh, convert the missing BF4 layers into the cache, close: no traces, no serving",
    )
    parser.add_argument("--evidence", type=Path, default=None, help="run directory (READY, STOPPED, ledgers)")
    parser.add_argument("--host", default="127.0.0.1", help="loopback unless the profile serves the LAN or --allow-lan")
    parser.add_argument(
        "--allow-lan", action="store_true", help="accept a non-loopback --host on a profile that does not serve the LAN"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--acceptance-prompts", type=Path, default=None, help="the CPU study's prompt-*-greedy.json")
    parser.add_argument("--require-json-96", action="store_true", help="refuse to serve unless json matches 96/96")
    parser.add_argument(
        "--prefill-mode",
        choices=PREFILL_MODES,
        default=DEFAULT_PREFILL_MODE,
        help="chunked: capture the chunk trace and prefill through it; teacher_forced: the decode traces only",
    )
    parser.add_argument(
        "--allocated-context",
        type=int,
        choices=RESIDENT_QSA_CACHE_CAPACITIES,
        default=RESIDENT_MAX_QSA_CACHE_CAPACITY,
        help="the resident build's allocated context; the KV caches, RoPE tables, the reserve and the context limit "
        "follow it (each non-default context builds its own component-cache identity; 262144 is single-user)",
    )
    parser.add_argument("--queue-limit", type=int, default=QUEUE_LIMIT, help="queued requests before 503")
    parser.add_argument(
        "--request-deadline-seconds", type=float, default=None, help="end a request with finish 'deadline' after this"
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=HEARTBEAT_SECONDS, help="progress log interval")
    parser.add_argument(
        "--hardware-profile",
        choices=tuple(hardware_profiles.hardware_profile_table()),
        default=None,
        help="tt-quietbox | bh-loudbox | tt-quietbox-2[-instance-1] (a private table adds development hosts)",
    )
    parser.add_argument("--validate-only", action="store_true", help="provenance and CPU preparation, no mesh")
    parser.add_argument(
        "--sampling",
        dest="sampling",
        action="store_true",
        default=False,
        help="capture TAIL with the candidate-row epilogue: sampled requests served (+0.3 ms per greedy token)",
    )
    parser.add_argument(
        "--no-sampling",
        dest="sampling",
        action="store_false",
        default=False,
        help="the default, explicit: capture TAIL without the candidate row, greedy requests only",
    )
    parser.add_argument(
        "--sampling-discriminator",
        action="store_true",
        help="after the acceptance replay run the sampling chain arms on the gate prompt, write "
        "sampling-discriminator.json and stop without serving",
    )
    parser.add_argument(
        "--mtp",
        type=int,
        choices=MTP_DRAFTS,
        default=None,
        help="draft K tokens per pass with the MTP layer (verify / draft / commit traces; greedy chunked-mode "
        "requests generate through the pass loop); default off",
    )
    parser.add_argument(
        "--mtp-gdn-anchor",
        choices=MTP_GDN_ANCHORS,
        default="off",
        help="MTP: the GDN state re-anchor (layer0 = layer 0's commits run the 1-row fp32 step recurrence)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        hardware_profile = hardware_profiles.resolve_hardware_profile(args.hardware_profile)
    except hardware_profiles.HardwareProfileError as error:
        raise SystemExit(str(error)) from error
    if args.host not in ("127.0.0.1", "::1", "localhost") and not (args.allow_lan or hardware_profile.lan_serving):
        raise SystemExit(f"--host must be loopback on {hardware_profile.host} without --allow-lan, got {args.host!r}")
    if not 1024 <= args.port <= 65535:
        raise SystemExit(f"--port must be in [1024, 65535], got {args.port}")
    if args.evidence is None and not args.validate_only:
        raise SystemExit("--evidence is required to serve")
    if not 1 <= args.queue_limit <= 64 or args.heartbeat_seconds <= 0:
        raise SystemExit(
            f"--queue-limit must be in [1, 64] and --heartbeat-seconds positive, got {args.queue_limit} {args.heartbeat_seconds}"
        )
    if args.request_deadline_seconds is not None and args.request_deadline_seconds <= 0:
        raise SystemExit(f"--request-deadline-seconds must be positive, got {args.request_deadline_seconds}")
    if args.sampling_discriminator and (not args.sampling or args.acceptance_prompts is None):
        raise SystemExit("--sampling-discriminator needs --sampling and the acceptance prompt records")
    if args.bf4_stage_limit is not None and args.bf4_stage_limit <= 0:
        raise SystemExit(f"--bf4-stage-limit must be positive, got {args.bf4_stage_limit}")
    if args.device_nodes is not None:
        try:
            hardware_profile = hardware_profile.with_device_nodes(
                tuple(int(node) for node in args.device_nodes.split(","))
            )
        except (ValueError, hardware_profiles.HardwareProfileError) as error:
            raise SystemExit(f"--device-nodes {args.device_nodes!r}: {error}") from error

    def marker(phase: str) -> None:
        append_marker(args.phase_log, phase)
        _log("phase", phase=phase)

    for name, expected in (
        ("TT_VISIBLE_DEVICES", hardware_profile.visible_devices),
        ("QWEN38_HARDWARE_MODE", "diagnostic_non_promoting"),
        ("TT_METAL_TRACE_ALLOC_TRACKING", "1"),
    ):
        if os.environ.get(name) != expected:
            raise SystemExit(f"{name} is {os.environ.get(name)!r}, expected {expected!r}")
    profiler = tuple(name for name in PROFILER_VARIABLES if os.environ.get(name) is not None)
    if profiler:
        raise SystemExit(f"profiler instrumentation is set: {profiler}")
    runtime = runtime_admission.admit_runtime(args)
    _log("runtime", **{key: value for key, value in runtime.items() if key != "bundle"})
    lock_proof = hardware_profiles.verify_inherited_locks(hardware_profile)
    # The route: pinned by the profile, or (the LoudBox) derived from the cluster descriptor now and recorded.
    hardware_profile, route_derivation = resolve_route(hardware_profile)
    _log(
        "route",
        lane=hardware_profile.lane,
        route=list(hardware_profile.route),
        nodes=list(hardware_profile.route_nodes),
    )
    prepared, _oracle = runtime_admission.prepare_cpu(
        args, marker=marker, hardware_profile=hardware_profile, identity=runtime
    )
    template = Qwen38OfficialChatTemplate(prepared.checkpoint.root)
    records = load_acceptance_records(args.acceptance_prompts) if args.acceptance_prompts is not None else []
    resident_context = Qwen38ResidentContext(args.allocated_context)
    # MTP is refused where the resident build's admission table says its pair, state and chain do not fit.
    mtp_admission = None if args.mtp is None else mtp_capacity_admission(resident_context.allocated_context)
    if mtp_admission is not None and not mtp_admission["fits"]:
        raise SystemExit(
            f"--mtp {args.mtp} does not fit at --allocated-context {resident_context.allocated_context}: the resident "
            f"build leaves {mtp_admission['free_bytes_per_bank_after_captures']} free bytes per DRAM bank "
            f"({mtp_admission['largest_contiguous_bytes_free_per_bank_after_captures']} contiguous) after its captures, "
            f"the MTP chain needs {mtp_admission['required_free_bytes_per_bank']} "
            f"({mtp_admission['required_largest_contiguous_bytes_per_bank']} contiguous): {mtp_admission}"
        )
    summary = {
        "mode": "chat_server_single_trace_chain",
        "model": MODEL_ID,
        "hardware_profile": hardware_profile.host,
        "hardware_partition": hardware_profile.partition,
        "device_nodes": list(hardware_profile.device_nodes),
        "route": list(hardware_profile.route),
        "route_nodes": list(hardware_profile.route_nodes),
        "route_derivation": route_derivation["route_derivation"],
        "allocated_context": resident_context.allocated_context,
        "context_limit": resident_context.context_limit,
        "source": {
            "worktree": runtime["repo"],
            "head": runtime["head"],
            "tree": runtime["tree"],
            "clean": not runtime["dirty"],
        },
        "runtime": runtime,
        "prepared": prepared.summary(),
        "template_sha256": PINNED_TOKENIZER_ARTIFACTS["chat_template.jinja"],
        "acceptance_records": [record["prompt"] for record in records],
        "require_json_96": bool(args.require_json_96),
        "prefill_mode": args.prefill_mode,
        "host": args.host,
        "lan_serving": bool(args.allow_lan or hardware_profile.lan_serving),
        "port": args.port,
        "queue_limit": args.queue_limit,
        "request_deadline_seconds": args.request_deadline_seconds,
        "defaults": {"enable_thinking": ENABLE_THINKING_DEFAULT, "reasoning_effort": REASONING_EFFORT_DEFAULT},
        "sampling": "candidate_row_host_sampler" if args.sampling else "greedy",
        "sampling_discriminator": bool(args.sampling_discriminator),
        "mtp": {
            "k": args.mtp,
            "anchor": args.mtp_gdn_anchor if args.mtp is not None else None,
            "admission": mtp_admission,
        },
        "system_fingerprint": f"{runtime['head'][:12]}-{runtime['extension_sha256'][:12]}",
    }
    if args.validate_only:
        print(json.dumps({"status": "pass", "mesh_open_requested": False, **summary}, sort_keys=True))
        return 0

    evidence = args.evidence.resolve(strict=True)
    ready_marker, stopped_marker = evidence / "READY", evidence / "STOPPED"
    for path in (ready_marker, stopped_marker):
        if path.exists():
            raise SystemExit(f"{path} exists: this evidence directory was already used")
    report: dict[str, Any] = {
        "schema": "qwen38-chat-server-run/v1",
        "status": "fail",
        "start_utc": utc_now(),
        "pid": os.getpid(),
        "lock_proof": lock_proof,
        "mesh_closed": False,
        "fabric_disabled": False,
        "cleanup_errors": [],
        **summary,
    }
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)
    started_ns = time.perf_counter_ns()
    mesh = chain = server = None
    fabric_enabled = False
    uncertain = False
    cleanup_errors: list[str] = []
    try:
        mesh, report["topology"] = open_partition_b_mesh(marker, hardware_profile)
        fabric_enabled = True
        if args.prepare_only:
            # The caches only: the builder converts the missing BF4 layers (bounded by --bf4-stage-limit) and the
            # run stops; a later start finds them.  The component and model-I/O caches fill at the first target build.
            construction = construct_live_decode_diagnostic(
                prepared,
                mesh_device=mesh,
                collective_topology=ttnn.Topology.Linear,
                marker=marker,
                bf4_stage_limit=args.bf4_stage_limit,
                require_complete=False,
            )
            missing = missing_bf4_layers(construction.builder)
            report["prepare"] = {
                "bf4_cache_root": str(construction.production_cache.root),
                "staged_bf4_layers": [list(slot) for slot in construction.staged_bf4_layers],
                "missing_bf4_layers": [list(slot) for slot in missing],
            }
            _log("prepared", staged=len(construction.staged_bf4_layers), missing=len(missing))
            ttnn.synchronize_device(mesh)
            report["status"] = "stopped"
            raise Qwen38ChatServerStop("prepare-only run complete")
        chain = construct_chain(
            prepared,
            mesh,
            marker=marker,
            chunked_prefill=args.prefill_mode == "chunked",
            sampling=bool(args.sampling),
            bf4_stage_limit=args.bf4_stage_limit,
            mtp=args.mtp,
            mtp_gdn_anchor=args.mtp_gdn_anchor,
        )
        if chain.allocated_context != resident_context.allocated_context:
            raise Qwen38ChatChainError(
                f"chain allocated context {chain.allocated_context} vs requested {resident_context.allocated_context}"
            )
        session = Qwen38ChatSession(chain, template, prefill_mode=args.prefill_mode)
        if session.prefill_mode != args.prefill_mode:
            raise Qwen38ChatChainError(f"session prefill mode {session.prefill_mode} vs requested {args.prefill_mode}")
        if (session.sampling is not None) != bool(args.sampling):
            raise Qwen38ChatChainError(f"session sampling {session.sampling is not None} vs requested {args.sampling}")
        if (session.mtp is not None) != (args.mtp is not None):
            raise Qwen38ChatChainError(f"session mtp {session.mtp is not None} vs requested {args.mtp}")
        if session.context_limit != resident_context.context_limit:
            raise Qwen38ChatChainError(
                f"session context limit {session.context_limit} vs the build's {resident_context.context_limit}"
            )
        report["chain"] = {
            "allocated_context": chain.allocated_context,
            "context_limit": session.context_limit,
            "open_seconds": chain.open_seconds,
            "capture_ms": chain.capture_ms,
            "program_cache_entries": chain.program_cache_entries,
            "head_traces": len(chain.head_trace_ids),
            "tail_traces": len(chain.tail_trace_ids),
            "chunk_traces": 0 if chain.chunk_trace_id is None else 1,
            "chunk_capture_ms": chain.chunk_capture_ms,
            "prefill_mode": session.prefill_mode,
            "sampling": chain.sampling is not None,
            "mtp": (
                None
                if chain.mtp is None
                else {
                    "k": chain.mtp.drafts,
                    "anchor": chain.mtp.anchor,
                    "traces": len(chain.mtp.captured_trace_ids()),
                    "capture_ms": chain.mtp.capture_ms,
                    "trace_dram_bytes_per_bank": chain.mtp.trace_dram_bytes_per_bank,
                    "dram_bytes_per_bank": chain.mtp.dram_bytes_per_bank,
                    "admission": chain.mtp.admission,
                }
            ),
        }
        # The allocator after every capture (the traces bake their addresses in; nothing is allocated after this
        # point on the serving path): the build's per-device headroom, in the report, READY and /health.
        ttnn.synchronize_device(mesh)
        report["chain"]["dram_after_captures"] = hardware_profiles.symmetric_mesh_dram_memory(
            mesh, hardware_profile.route
        )
        _log("dram_after_captures", allocated_context=chain.allocated_context, **report["chain"]["dram_after_captures"])
        if records:
            marker("before-chat-acceptance-replay")
            report["acceptance"] = replay_acceptance(
                session, records, require_gate=args.require_json_96, handoff_gate=session.mtp is not None
            )
            (evidence / "acceptance.json").write_text(
                json.dumps(report["acceptance"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            marker("after-chat-acceptance-replay")
        if args.sampling_discriminator:
            # The chain arms (arm b is the replay above) on the creative record, whose distribution is flat enough
            # for a sampled stream to leave the greedy one (the json gate prompt is near-deterministic), then a
            # clean stop: no serving.
            marker("before-sampling-discriminator")
            record = next(
                (record for record in records if record["prompt"] == DISCRIMINATOR_PROMPT),
                next(record for record in records if record["prompt"] == ACCEPTANCE_GATE_PROMPT),
            )
            result = sampling_step.run_discriminator(session, record["prompt_token_ids"])
            result["prompt"] = record["prompt"]
            result["program_cache_delta"] = resident_decode.program_cache_count(mesh) - chain.program_cache_entries
            result["acceptance_gate_pass"] = report["acceptance"]["gate_pass"]
            result["pass"] = bool(
                result["pass"] and result["program_cache_delta"] == 0 and result["acceptance_gate_pass"]
            )
            report["sampling_discriminator"] = result
            (evidence / "sampling-discriminator.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            marker("after-sampling-discriminator")
            _log(
                "sampling_discriminator",
                **{key: value for key, value in result.items() if key not in ("rows", "greedy", "sampled")},
            )
            report["status"] = "stopped"
        else:
            server = Qwen38ChatHTTPServer(
                (args.host, args.port),
                session,
                ledger=evidence / "requests.jsonl",
                queue_limit=args.queue_limit,
                request_deadline_seconds=args.request_deadline_seconds,
                heartbeat_seconds=args.heartbeat_seconds,
                system_fingerprint=summary["system_fingerprint"],
                dram_after_captures=report["chain"]["dram_after_captures"],
            )
            ready = {
                "pid": os.getpid(),
                "host": args.host,
                "port": args.port,
                "utc": utc_now(),
                "startup_seconds": (time.perf_counter_ns() - started_ns) / 1e9,
                "capture_ms": chain.capture_ms,
                "allocated_context": chain.allocated_context,
                "free_bytes_per_bank": report["chain"]["dram_after_captures"]["free_bytes_per_bank"],
                "acceptance_gate_pass": None if not records else report["acceptance"]["gate_pass"],
                "mtp": report["chain"]["mtp"],
            }
            ready_marker.write_text(json.dumps(ready, sort_keys=True) + "\n", encoding="utf-8")
            marker("chat-server-ready")
            _log("ready", **ready, base_url=f"http://{args.host}:{args.port}/v1", model=MODEL_ID)
            try:
                server.serve_forever(poll_interval=0.5)
            except Qwen38ChatServerStop:
                uncertain = server.busy or session.poisoned
                report["status"] = "stopped" if not uncertain else "stopped_mid_request"
    except Qwen38ChatServerStop as stop:
        if report["status"] != "stopped":
            report["error"] = f"{type(stop).__name__}: {stop}"
            _log("failed", error=report["error"])
    except BaseException as error:
        uncertain = uncertain or (server is not None and (server.busy or server.session.poisoned))
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        _log("failed", error=report["error"])
    finally:
        if server is not None:
            server.server_close()
        # The runner's cleanup: a failure mid-loop leaves queue and trace ownership
        # uncertain; then the mesh is closed without release work.
        if chain is not None and not uncertain:
            try:
                marker("before-chat-chain-close")
                chain.close()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"chain_close:{type(error).__name__}:{error}")
        if mesh is not None:
            if not uncertain:
                try:
                    ttnn.synchronize_device(mesh)
                    mesh.disable_and_clear_program_cache()
                except BaseException as error:  # noqa: BLE001
                    cleanup_errors.append(f"program_cache_disable:{type(error).__name__}:{error}")
            try:
                marker("before-mesh-close")
                ttnn.close_mesh_device(mesh)
                report["mesh_closed"] = True
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"mesh_close:{type(error).__name__}:{error}")
        if fabric_enabled:
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                report["fabric_disabled"] = True
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"fabric_disable:{type(error).__name__}:{error}")
        if cleanup_errors:
            report["status"] = "fail"
        report.update(
            end_utc=utc_now(),
            cleanup_errors=cleanup_errors,
            uncertain_boundary=uncertain,
            requests_served=None if server is None else server.session.requests_served,
        )
        stopped_marker.write_text(
            json.dumps({"pid": os.getpid(), "utc": utc_now(), "status": report["status"]}) + "\n",
            encoding="utf-8",
        )
        write_result(evidence / "result.json", report)
        _log("stopped", status=report["status"], cleanup_errors=cleanup_errors)
    return 0 if report["status"] == "stopped" else 1


if __name__ == "__main__":
    sys.exit(main())
