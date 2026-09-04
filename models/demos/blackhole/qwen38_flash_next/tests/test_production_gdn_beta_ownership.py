# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""No-device contract for the qualified production GDN beta lifetime."""

from __future__ import annotations

import ast
from pathlib import Path

GDN_SOURCE = Path(__file__).parents[1] / "ttnn" / "gdn.py"


def _method_source(name: str) -> str:
    source = GDN_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38TTNNGDN")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    lines = source.splitlines()
    return "\n".join(lines[method.lineno - 1 : method.end_lineno])


def test_beta_uses_qualified_fp32_sigmoid_and_retains_all_producers() -> None:
    source = _method_source("_make_recurrent_inputs")
    required_in_order = (
        "b_fp32 = ttnn.typecast(b, ttnn.float32",
        "beta_fp32 = ttnn.sigmoid(b_fp32",
        "beta = ttnn.reshape(beta_fp32",
        "_retag_head_shard_after_reshape(beta, reference=beta_fp32, shard_dim=1)",
        "beta_producers = (b, b_fp32, beta_fp32)",
        "log_decay_raw = ttnn.multiply(",
        "log_decay = ttnn.reshape(log_decay_raw",
        "_retag_head_shard_after_reshape(log_decay, reference=log_decay_raw, shard_dim=1)",
        "beta_producers = (*beta_producers, log_decay_raw)",
        "return q, k, v, beta, log_decay, beta_producers",
    )
    offsets = [source.index(fragment) for fragment in required_in_order]
    assert offsets == sorted(offsets)
    # The FP32 sigmoid result feeds the recurrent step directly: no bf16
    # round-trip may reappear between the sigmoid and the reshape.
    assert "ttnn.typecast(beta_fp32" not in source
    for producer in ("b", "b_fp32", "beta_fp32", "log_decay_raw"):
        assert f"_deallocate({producer})" not in source


def test_beta_producers_are_released_only_after_their_recurrent_consumers() -> None:
    source = _method_source("_recurrent_decode")
    decay = source.index("decayed = ttnn.multiply(\n            state.recurrent,\n            log_decay,")
    gate = source.index("update = ttnn.multiply(outer, beta, memory_config=l1)")
    release = source.index("_deallocate(decayed, update, beta, log_decay, *beta_producers)")
    assert decay < gate < release
    # log_decay and beta arrive FP32 and are consumed as is: no promotion, no
    # other release of any beta producer anywhere in the step.
    assert "ttnn.typecast(log_decay" not in source and "ttnn.typecast(beta" not in source
    releases = [line.strip() for line in source.splitlines() if "_deallocate(" in line]
    assert [line for line in releases if "beta" in line or "log_decay" in line] == [
        "_deallocate(decayed, update, beta, log_decay, *beta_producers)"
    ]


def test_forward_decode_transfers_beta_producer_ownership_explicitly() -> None:
    source = _method_source("forward_decode")
    make = source.index("q, k, v, beta, log_decay, beta_producers = self._make_recurrent_inputs")
    consume = source.index("self._recurrent_decode(q, k, v, beta, log_decay, beta_producers, state)")
    project = source.index("self._gate_and_project(recurrent_output, z, full_hidden)")
    assert make < consume < project


def test_beta_contract_keeps_head_sharded_topology_and_exact_shape() -> None:
    source = _method_source("_make_recurrent_inputs")
    # [1,H,1,1] is the broadcast form the FP32 step multiplies against directly.
    assert "(1, VALUE_HEADS_PER_DEVICE, 1, 1)" in source
    assert "placement=TensorPlacement.HEAD_SHARDED, shard_dim=1" in source
    assert "shard_dim=2" not in source
    assert "_retag_head_shard_after_reshape(beta, reference=beta_fp32, shard_dim=1)" in source
