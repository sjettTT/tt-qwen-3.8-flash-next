// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "gtest/gtest.h"
#include "ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_program_factory.hpp"

namespace {

TEST(MoEComputeHostHelpers, Qwen38FusedSourceBufferFitsPhysicalL1Shard) {
    // Qwen3.8 Flash-Next on one Blackhole card: each physical output shard is
    // [2 buffers, 32 token rows, 2560 hidden] in BF16. Four combine columns
    // consume disjoint 640-element segments from that physical shard.
    constexpr uint32_t hidden_size = 2560;
    constexpr uint32_t data_parallel_cores = 4;
    constexpr uint32_t element_size_bytes = 2;
    constexpr uint32_t source_shard_height = 64;
    constexpr uint32_t source_shard_width = hidden_size;
    constexpr uint32_t num_buffers = 2;
    constexpr uint32_t token_segment_size_bytes = hidden_size / data_parallel_cores * element_size_bytes;
    constexpr uint32_t source_buffer_size_bytes = source_shard_height * source_shard_width * element_size_bytes;

    const auto layout = ttnn::experimental::prim::detail::compute_fused_source_buffer_layout(
        source_shard_height, source_buffer_size_bytes, token_segment_size_bytes, num_buffers);

    EXPECT_EQ(layout.rows_per_buffer, 32);
    EXPECT_EQ(layout.buffer_block_size_bytes, 40960);
    EXPECT_EQ(layout.circular_buffer_size_bytes, 81920);
    EXPECT_LE(layout.circular_buffer_size_bytes, source_buffer_size_bytes);

    // Producer and consumer both toggle between offsets 0 and 40960. The
    // final 1280-byte token segment of either 32-row block remains in bounds.
    EXPECT_EQ(layout.buffer_block_size_bytes, layout.rows_per_buffer * token_segment_size_bytes);
    EXPECT_EQ(layout.circular_buffer_size_bytes, num_buffers * layout.buffer_block_size_bytes);
    EXPECT_LE(
        layout.buffer_block_size_bytes + (layout.rows_per_buffer - 1) * token_segment_size_bytes +
            token_segment_size_bytes,
        layout.circular_buffer_size_bytes);
}

}  // namespace
