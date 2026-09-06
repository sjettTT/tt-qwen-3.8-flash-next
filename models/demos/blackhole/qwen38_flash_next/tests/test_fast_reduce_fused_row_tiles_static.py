# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Source pins for the fused weighted reduce (``deepseek_moe_fast_reduce_nc_fused``) past one 32-token tile.

The op's reader used to build one score tile per expert (token rows 0..31) and the compute kernel scaled every output
tile with it, so an input taller than one row tile (tokens > 32) weighted rows 32.. with the scores of rows 0..31 (the
128-row MoE chunk's rows past the first tile came back wrong; measured on two 4x p150 hosts, 2026-09-06).  The fix builds one score tile
per (row tile, expert) and picks the group by the output tile's row tile.  These pins keep that shape (no device, no
ttnn import).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
OP = REPO_ROOT / "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused"
KERNELS = OP / "device/kernels"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_reader_builds_one_score_tile_per_row_tile_and_expert() -> None:
    reader = _source(KERNELS / "deepseek_moe_fast_reduce_nc_fused_reader.cpp")

    required = (
        "constexpr uint32_t num_row_tiles = num_tokens_x32 / 32;",
        "cb_scores.reserve_back(num_row_tiles * reduction_dim_size);",
        "for (uint32_t r = 0; r < num_row_tiles; ++r) {",
        "scores_tile_u16 + (r * reduction_dim_size + k) * tile_u16_stride;",
        "const uint32_t t = r * 32 + j;",
        "const uint32_t col0_index = (j < 16) ? j * 16 : face2_offset + (j - 16) * 16;",
        "cb_scores.push_back(num_row_tiles * reduction_dim_size);",
    )
    for statement in required:
        assert statement in reader, statement
    # The one-tile form's face-2 write ran past the tile for tokens >= 32.
    assert "expert_tile[face2_offset + (t - 16) * 16]" not in reader
    assert "cb_scores.reserve_back(reduction_dim_size);" not in reader
    # Padding rows of the last row tile stay zero-scored inside the same fill.
    assert "if (t >= num_tokens) {" in reader


def test_compute_scales_each_output_tile_with_its_row_tile_scores() -> None:
    compute = _source(KERNELS / "deepseek_moe_fast_reduce_nc_fused_compute.cpp")

    required = (
        "constexpr uint32_t num_cores_to_be_used = get_compile_time_arg_val(6);",
        "constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(7);",
        "constexpr uint32_t num_row_tiles = get_compile_time_arg_val(8);",
        "const uint32_t start_tile = get_arg_val<uint32_t>(0);",
        "constexpr uint32_t num_score_tiles = num_row_tiles * reduction_dim_size;",
        "cb_in1.wait_front(num_score_tiles);",
        "const uint32_t row_tile = (tile_id / input_tensor_Wt) % num_row_tiles;",
        "const uint32_t score_tile_base = row_tile * reduction_dim_size;",
        "compute_input_cb_id_0, compute_input_cb_id_1, k, score_tile_base + expert_tile, dst0);",
        "tile_id += num_cores_to_be_used;",
        "cb_in1.pop_front(num_score_tiles);",
    )
    for statement in required:
        assert statement in compute, statement
    assert "cb_in1.wait_front(reduction_dim_size);" not in compute


def test_program_factory_sizes_the_score_cb_and_passes_the_tile_geometry() -> None:
    factory = _source(OP / "device/deepseek_moe_fast_reduce_nc_fused_program_factory.cpp")

    required = (
        "const uint32_t num_row_tiles = num_tokens_x32 / tt::constants::TILE_HEIGHT;",
        ".total_size = num_row_tiles * reduction_dim_size * scores_tile_size,",
        "compute_desc.emplace_runtime_args(core, {start_tiles_read});",
    )
    for statement in required:
        assert statement in factory, statement
    # Both compute core groups carry the same three geometry compile-time args after the six the kernel had.
    assert (
        factory.count("        cb_out_id,\n        num_cores,\n        input_tensor_Wt,\n        num_row_tiles,\n") == 1
    )
    assert (
        factory.count(
            "            cb_out_id,\n            num_cores,\n            input_tensor_Wt,\n            num_row_tiles,\n"
        )
        == 1
    )


def test_device_operation_admits_any_number_of_row_tiles() -> None:
    operation = _source(OP / "device/deepseek_moe_fast_reduce_nc_fused_device_operation.cpp")

    assert "(scores_shape[0] > num_tokens - tt::constants::TILE_HEIGHT) && (scores_shape[0] <= num_tokens)" in operation
    assert "Any number of row tiles is supported" in operation
