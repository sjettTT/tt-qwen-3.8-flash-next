# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The qsa_rows family's registry entry and its hooks in the QSA verify path (no device): program 1 (the score
merge over the rows) and program 2 (the main tail with the verify rows' KV stage)."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from pathlib import Path

from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block, qsa_rows

# the package rebinds its ``main_tail_rows`` attribute to the function; the module is reached through sys.modules
main_tail_rows_module = importlib.import_module("models.demos.blackhole.qwen38_flash_next.ttnn.fused.qsa_rows.main_tail_rows")

MODEL_DIR = Path(__file__).resolve().parents[1]


def test_qsa_rows_is_registered_bitwise_and_default_on() -> None:
    kernel = fused.kernel("qsa_rows")
    assert kernel.tolerance == fused.BITWISE
    assert kernel.default_on, "qsa_rows serves by default since its pass pair, A3 pins and D1 band read clean (2026-09-26)"
    assert kernel.fused is qsa_rows.score_blocks_rows
    assert kernel.composed is qsa_rows.score_blocks_rows_composed
    assert "qsa_rows" in fused.__all__ and fused.qsa_rows is qsa_rows


def test_score_blocks_rows_composes_the_proven_merge() -> None:
    """Program 1 is the decode chain's qsa_score_merge over the tile's rows: the same page form and the same merge."""

    assert qsa_rows.PAGE == qsa_block.SCORE_CHUNK == qsa_module.SCORE_GATHER_PAGE == 1024
    assert qsa_rows.DEVICES == qsa_block.DEVICES == qsa_module.TP_SIZE == 4
    source = inspect.getsource(qsa_rows.score_blocks_rows)
    assert "ttnn.all_gather(paged, dim=2, cluster_axis=cluster_axis" in source
    assert "qsa_block.score_merge(gathered, mask)" in source
    composed = inspect.getsource(qsa_rows.score_blocks_rows_composed)
    assert "ttnn.all_reduce(score_rows, cluster_axis=cluster_axis" in composed
    assert "fast_and_approximate_mode=False" in composed


def test_the_verify_path_takes_program_one_on_the_single_tile_only() -> None:
    """forward_verify_generic's score step runs the fused form on the 32-row tile with the chunk's row mask; the slab
    (rows > 32, block masks instead of one mask) and the unfused module keep the chain."""

    source = inspect.getsource(qsa_module.Qwen38TTNNQSA._score_blocks_chunk)
    hook = "if self._rows_fused is not None and rows == CHUNK_ROWS and chunk.indexer_neg_mask is not None:"
    assert hook in source
    assert "self._rows_fused.score_blocks_rows(score_rows, chunk.indexer_neg_mask, cluster_axis=TP_AXIS)" in source
    assert source.index(hook) < source.index("scores = ttnn.all_reduce("), "the chain stays as the fallback"
    init = inspect.getsource(qsa_module.Qwen38TTNNQSA.__init__)
    assert 'if fused_kernels.enabled("qsa_rows"):' in init and "self._rows_fused = fused_kernels.qsa_rows" in init
    assert qsa_module.Qwen38TTNNQSA._rows_fused is None


def test_main_tail_rows_is_the_decode_program_with_the_kv_stage() -> None:
    """Program 2 reuses qsa_block.main_tail (its norm / RoPE / head-split / query kernels on the 32-row tile) with the
    staging cores replaced by the verify rows' KV core; the decode form keeps its default (kv_stage None)."""

    assert qsa_rows.main_tail_rows is main_tail_rows_module.main_tail_rows
    assert qsa_rows.main_tail_rows_composed is main_tail_rows_module.main_tail_rows_composed
    assert inspect.signature(qsa_block.main_tail).parameters["kv_stage"].default is None
    source = inspect.getsource(qsa_rows.main_tail_rows)
    assert "kv_stage(position, rows, kv_cache, single_row=single_row" in source and "kv_stage=stage" in source
    for name in ("rows_kv_cbs.h", "rows_kv_reader.cpp", "rows_kv_writer.cpp"):
        assert (MODEL_DIR / "ttnn/fused/qsa_rows/kernels" / name).is_file(), name
    reader = (MODEL_DIR / "ttnn/fused/qsa_rows/kernels/rows_kv_reader.cpp").read_text()
    assert "rows_read < tile_rows::TILE_ROWS ? rows_read : tile_rows::TILE_ROWS" in reader, "R is read on the core"
    assert "slot = (P & 31u) + rows; slot < tile_rows::TILE_ROWS" in reader, "the rows past the pass are zeroed"


def test_the_verify_path_takes_program_two_when_the_pass_carries_its_scalars() -> None:
    """forward_verify_generic runs the fused main tail when the verify inputs carry P (and the constants R); the
    chain stays as the fallback; the two scalars default to None on the inputs and the constants."""

    source = inspect.getsource(qsa_module.Qwen38TTNNQSA.forward_verify_generic)
    hook = "if self._rows_fused is not None and verify.position is not None:"
    assert hook in source and "self._main_tail_rows_step(full_hidden, cos, sin, state, verify, constants)" in source
    assert source.index(hook) < source.index("self._main_projection_rows(full_hidden, None, cos, sin, constants)")
    step = inspect.getsource(qsa_module.Qwen38TTNNQSA._main_tail_rows_step)
    assert "self._rows_fused.main_tail_rows(" in step and "rows=verify.rows_u32" in step
    assert "_retag_tensor(sparse_query, reference=state.packed_kv_cache, shard_dim=1)" in step
    fields = {f.name: f for f in dataclasses.fields(qsa_module.Qwen38TTNNQSAVerifyInputs)}
    assert fields["position"].default is None and fields["rows_u32"].default is None
    constants = {f.name: f for f in dataclasses.fields(qsa_module.Qwen38TTNNQSAVerifyConstants)}
    assert constants["rows_u32"].default is None
    host = qsa_module.qsa_verify_constant_rows(5, 512)
    assert tuple(host["rows_u32"].shape) == (1, 1, 1, 1) and int(host["rows_u32"]) == 5


def test_the_manifest_lists_the_family() -> None:
    manifest = (MODEL_DIR / "tools/release/manifest.json").read_text()
    for path in (
        "ttnn/fused/qsa_rows/__init__.py",
        "ttnn/fused/qsa_rows/main_tail_rows.py",
        "ttnn/fused/qsa_rows/kernels/rows_kv_cbs.h",
        "ttnn/fused/qsa_rows/kernels/rows_kv_reader.cpp",
        "ttnn/fused/qsa_rows/kernels/rows_kv_writer.cpp",
        "tests/test_fused_qsa_rows_static.py",
    ):
        assert f'"{path}"' in manifest, path
