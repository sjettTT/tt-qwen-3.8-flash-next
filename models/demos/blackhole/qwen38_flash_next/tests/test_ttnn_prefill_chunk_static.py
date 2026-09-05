# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Static pins of the prefill chunk body (layer and model): the decode's stage order on 32-row operands, no host
ints or host I/O inside the body, the chunk state beside the generic state, ``P += 32`` last, and the 1-row
position-generic bodies untouched."""

from __future__ import annotations

import ast
import inspect

from models.demos.blackhole.qwen38_flash_next.ttnn import layer as layer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROWS
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import Qwen38TTNNDecoderLayer, Qwen38TTNNDecoderLayerChunkState
from models.demos.blackhole.qwen38_flash_next.ttnn.model import Qwen38TTNNTextModel, Qwen38TTNNTextModelChunkState

HOST_IO = ("from_torch(", "to_torch(", "copy_host_to_device_tensor(", ".item()", "synchronize", ".shape[")


def _dedent(function) -> str:
    lines = inspect.getsource(function).splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join(line[indent:] for line in lines)


def _self_calls(function) -> list[str]:
    tree = ast.parse(_dedent(function))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name.startswith("self.") and "validate" not in name and "_require" not in name:
                calls.append((node.lineno, node.col_offset, name))
    return [name for _line, _col, name in sorted(calls)]


def test_layer_chunk_body_is_the_decode_order_on_rows_with_the_history_commits() -> None:
    body = inspect.getsource(Qwen38TTNNDecoderLayer.forward_chunk_generic)
    for forbidden in HOST_IO:
        assert forbidden not in body, forbidden
    calls = _self_calls(Qwen38TTNNDecoderLayer.forward_chunk_generic)
    assert calls == [
        "self.ple.inject_rows",
        "self.ple.commit_rows",
        "self.attention_gr.read_rows",
        "self.attention.forward_rows",
        "self.attention.commit_rows",
        "self.attention.forward_chunk_generic",
        "self.attention_gr.write_rows",
        "self.mlp_gr.read_rows",
        "self.expert_streamer.layer",
        "self.mlp_gr.write_rows",
    ]
    # The MoE is the chunk state's rows-32 instance, not the layer's 1-row instance.
    assert "chunk_state.moe.forward(mlp_input, packed_experts[0], packed_experts[1])" in body
    assert "self.mlp.forward(" not in body
    # The GDN output rows are the persistent rows-state buffer and are never deallocated by the layer.
    assert "persistent_hidden = True" in body and "None if persistent_hidden else attention_hidden" in body
    assert "_deallocate_unique(result.final_state)" in body
    decode = inspect.getsource(Qwen38TTNNDecoderLayer.forward_decode_generic)
    route = inspect.getsource(Qwen38TTNNDecoderLayer._route_through_gr_and_moe)
    for source in (decode, route):
        assert "chunk" not in source and "_rows" not in source.replace("_rows_", "")
    fields = tuple(Qwen38TTNNDecoderLayerChunkState.__dataclass_fields__)
    assert fields == ("namespace", "layer_index", "attention", "ple", "moe")
    allocate = inspect.getsource(Qwen38TTNNDecoderLayer.allocate_chunk_state)
    assert "rows=PREFILL_CHUNK_ROWS" in allocate and "self.mlp.weights" in allocate
    assert "self.attention.allocate_rows_state(constants)" in allocate
    assert "self.attention.allocate_chunk_state()" in allocate
    assert "self.ple.allocate_rows_state(CHUNK_ROWS)" in allocate
    reset = inspect.getsource(Qwen38TTNNDecoderLayer.reset_chunk_state_inplace)
    assert "sync_rows_history_from_state(generic_state.attention, state.attention)" in reset
    assert (
        "state.ple.load_from_state(generic_state.ple)" in reset
        and "ttnn.fill(tensor, 0.0, output_tensor=tensor)" in reset
    )
    # The 1-row generic reset is untouched.
    assert "chunk" not in inspect.getsource(Qwen38TTNNDecoderLayer.reset_generic_state_inplace)


def test_model_chunk_body_derives_everything_on_device_and_advances_by_32_last() -> None:
    body = inspect.getsource(Qwen38TTNNTextModel.forward_prefill_chunk_generic)
    for forbidden in HOST_IO:
        assert forbidden not in body, forbidden
    order = (
        "gdn_module.build_rows_selectors(chunk_state.accepted, chunk_state.rows_constants)",
        "state.position.index_row()",
        "chunk_state.qsa_chunk_constants.arange32_lanes",
        "chunk_state.qsa_chunk_constants.block_start_lanes",
        "self.rope_table.rows_chunk(index_rows, block_start_rows)",
        "qsa_module.derive_qsa_chunk_inputs(",
        "self._embed_residual_rows_from_device_token(chunk_state.token_row)",
        "for layer_index in range(BACKBONE_LAYERS):",
        "layer.forward_chunk_generic(",
        "prepared_ple_rows=chunk_state.ple_rows if layer_index == PLE_CHECKPOINT_LAYER else None",
        "selectors.deallocate()",
        "state.position.advance_by(CHUNK_ROWS)",
    )
    positions = [body.index(fragment) for fragment in order]
    assert positions == sorted(positions)
    # No final mixer, no LM head, no per-token position advance inside the chunk.
    for forbidden in ("self.final_mixer(", "lm_head(", "position.advance()", "resolve_greedy"):
        assert forbidden not in body, forbidden
    tree = ast.parse(_dedent(Qwen38TTNNTextModel.forward_prefill_chunk_generic))
    try_body = next(node for node in ast.walk(tree) if isinstance(node, ast.Try)).body
    assert ast.unparse(try_body[-1]) == "state.position.advance_by(CHUNK_ROWS)"
    assert CHUNK_ROWS == 32


def test_model_chunk_state_is_allocated_before_capture_with_host_written_inputs() -> None:
    fields = tuple(Qwen38TTNNTextModelChunkState.__dataclass_fields__)
    assert fields == ("rows_constants", "qsa_chunk_constants", "layers", "token_row", "ple_rows", "accepted", "_owner")
    allocate = inspect.getsource(Qwen38TTNNTextModel.allocate_chunk_state)
    assert "gdn.allocate_rows_constants(CHUNK_ROWS)" in allocate
    assert "qsa_module.Qwen38TTNNQSAChunkConstants.build(" in allocate
    assert "layer.allocate_chunk_state(rows_constants)" in allocate
    assert "self.model_io.embedding.upload_token_row(0)" in allocate
    assert "ple_layer.ple.prepare_rows_input([0] * CHUNK_ROWS, ple_rows_state)" in allocate
    assert "float(CHUNK_ROWS - 1)" in allocate  # a full chunk by default
    # The host writers are the only host paths into the chunk buffers, all outside the body.
    accepted = inspect.getsource(Qwen38TTNNTextModel.write_chunk_accepted)
    assert "ttnn.copy_host_to_device_tensor(host, chunk_state.accepted)" in accepted
    assert "0 <= accepted < CHUNK_ROWS" in accepted
    inputs = inspect.getsource(Qwen38TTNNTextModel.write_chunk_inputs)
    assert inputs.count("ttnn.copy_host_to_device_tensor(") == 2
    assert "self.model_io.embedding.host_token_rows(token_ids)" in inputs
    assert "ple.host_rows(token_ids, ple_context)" in inputs and "chunk_state.ple_rows.embedding_rows" in inputs
    reset = inspect.getsource(Qwen38TTNNTextModel.reset_chunk_state_inplace)
    assert "layer.reset_chunk_state_inplace(layer_chunk, layer_state)" in reset
    assert "self.write_chunk_accepted(chunk_state, CHUNK_ROWS - 1)" in reset
    assert "position.reset" not in reset  # the position is the decode's; the driver sets it
    capture = inspect.getsource(Qwen38TTNNTextModel.capture_prefill_chunk)
    assert "ttnn.corruptible_allocation_scope(self.mesh_device)" in capture
    assert "ttnn.begin_trace_capture(self.mesh_device, cq_id=cq_id)" in capture
    assert "self.forward_prefill_chunk_generic(chunk_state, state, gdn_step_anchor=gdn_step_anchor, mtp=mtp)" in capture
    assert "ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=cq_id)" in capture


def test_gdn_step_anchor_is_a_keyword_off_by_default_from_the_chain_to_the_gdn_commit() -> None:
    """The GDN state re-anchor (``commit_rows(step_committed_rows=True)``) reaches the chunk body as one keyword,
    default off at every level: the layer, the model body and its capture, the chain's open and the driver."""

    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module
    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_prefill_driver as driver_module
    from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module

    signatures = {
        gdn_module.Qwen38TTNNGDN.commit_rows: "step_committed_rows",
        Qwen38TTNNDecoderLayer.commit_gdn_rows: "step_committed_rows",
        Qwen38TTNNDecoderLayer.forward_chunk_generic: "gdn_step_anchor",
        Qwen38TTNNTextModel.forward_prefill_chunk_generic: "gdn_step_anchor",
        Qwen38TTNNTextModel.capture_prefill_chunk: "gdn_step_anchor",
        session_module.Qwen38TracedChain.open: "chunk_gdn_step_anchor",
        driver_module.Qwen38ChunkPrefill.__init__: "gdn_step_anchor",
    }
    for function, name in signatures.items():
        parameter = inspect.signature(function).parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default is False, function
    layer = inspect.getsource(Qwen38TTNNDecoderLayer.forward_chunk_generic)
    assert (
        "self.attention.commit_rows(\n"
        "                generic_state.attention, chunk_state.attention, selectors, step_committed_rows=gdn_step_anchor\n"
        "            )"
    ) in layer
    body = inspect.getsource(Qwen38TTNNTextModel.forward_prefill_chunk_generic)
    assert body.count("gdn_step_anchor") == 4  # the signature, the docstring, the call's keyword and its value
    assert body.count("gdn_step_anchor=gdn_step_anchor,") == 1
    opened = inspect.getsource(session_module.Qwen38TracedChain.open)
    assert "chunk_state, state, gdn_step_anchor=chunk_gdn_step_anchor, mtp=chunk_extension" in opened
    assert "gdn_step_anchor=chunk_gdn_step_anchor,\n            )" in opened  # the capture
    assert "chunk_gdn_step_anchor=chunk_gdn_step_anchor," in opened  # the chain records what its trace carries
    assert "gdn_step_anchor=self.chunk_gdn_step_anchor," in inspect.getsource(
        session_module.Qwen38TracedChain.chunk_prefill
    )
    assert session_module.Qwen38TracedChain.__dataclass_fields__["chunk_gdn_step_anchor"].default is False
    # The 1-row commit keeps the rows path's own switch beside it; the anchor form never reruns the chunk kernel.
    commit = inspect.getsource(gdn_module.Qwen38TTNNGDN.commit_rows)
    assert commit.index("if step_committed_rows:") < commit.index("output, final_state = self._chunk_rows(")
    assert "stepped = self._step_committed_rows_state(rows_state, state.recurrent, selectors)" in commit


def test_one_row_generic_bodies_and_their_reset_are_untouched() -> None:
    for function in (
        Qwen38TTNNTextModel.forward_decode_generic,
        Qwen38TTNNTextModel.forward_decode_generic_head,
        Qwen38TTNNTextModel.forward_decode_generic_tail,
        Qwen38TTNNTextModel.capture_decode_generic,
        Qwen38TTNNTextModel.reset_generic_state_inplace,
        Qwen38TTNNTextModel.allocate_generic_state,
        Qwen38TTNNTextModel.release_generic_state,
    ):
        source = inspect.getsource(function)
        assert "chunk" not in source.lower(), function.__name__
    assert "chunk" not in inspect.getsource(Qwen38TTNNDecoderLayer.allocate_generic_state)
    assert "chunk" not in inspect.getsource(Qwen38TTNNDecoderLayer.release_generic_state)
    assert model_module.GENERIC_HEAD_LAYERS >= 1 and layer_module.PLE_CHECKPOINT_LAYER == 1
