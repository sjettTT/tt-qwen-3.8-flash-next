// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One core: the per-core (value, id) pairs of scan.cpp -> per row the local maximum (bf16 TILE [1,1,rows,1]: lane
// (r, 0) of a zero tile), the local argmax (uint32 [1,1,rows]) and the packed fp32 row [value | float(id)] the resolve's
// all_gather takes.  Cores are visited in increasing order (increasing id ranges), so the first strict maximum is the
// lowest id.  ROWS = 1 (the decode step): the packed row is [1,1,1,2] (8 bytes at page 0).  ROWS > 1 (the lanes): the
// pairs rows are pages 0..rows-1, the packed rows [1,1,rows,16] (row r = page r: value, float(id), zeros; 64-byte
// pages at the DRAM grain); the lanes' single row (rows = 1, packed_lanes = 16) takes the same path.
// Named compile-time args: cb_stage, cores, rows, packed_lanes.  Compile-time args: TensorAccessorArgs(pairs), (zero tile),
// (values), (indices), (packed).  Runtime args: 0 pairs addr, 1 zero-tile addr, 2 values addr, 3 indices addr,
// 4 packed addr.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_STAGE = get_named_compile_time_arg_val("cb_stage");
constexpr uint32_t CORES = get_named_compile_time_arg_val("cores");
constexpr uint32_t ROWS = get_named_compile_time_arg_val("rows");
constexpr uint32_t PACKED_LANES = get_named_compile_time_arg_val("packed_lanes");  // 2: [value | id]; 16: the lanes' 64-byte rows
constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t PAIRS_BYTES = ((16 * CORES) + 63) & ~63u;
constexpr uint32_t PACKED_ROW_BYTES = PACKED_LANES * 4;
static_assert(PACKED_LANES == 2 || PACKED_LANES == 16, "packed rows are [value | id] or 64-byte lane rows");

FORCE_INLINE uint32_t key_of(uint32_t fp32_bits) {
    fp32_bits = (fp32_bits & 0x7FFFFFFFu) ? fp32_bits : 0u;  // scan.cpp already canonicalized -0.0; keep the rule here
    return (fp32_bits & 0x80000000u) ? ~fp32_bits : (fp32_bits | 0x80000000u);
}

union IdFp32 {
    float f;
    uint32_t u;
};

void kernel_main() {
    constexpr auto a_pairs = TensorAccessorArgs<0>();
    constexpr auto a_zero = TensorAccessorArgs<a_pairs.next_compile_time_args_offset()>();
    constexpr auto a_values = TensorAccessorArgs<a_zero.next_compile_time_args_offset()>();
    constexpr auto a_indices = TensorAccessorArgs<a_values.next_compile_time_args_offset()>();
    constexpr auto a_packed = TensorAccessorArgs<a_indices.next_compile_time_args_offset()>();
    const auto pairs = TensorAccessor(a_pairs, get_arg_val<uint32_t>(0));
    const auto zero = TensorAccessor(a_zero, get_arg_val<uint32_t>(1));
    const auto values = TensorAccessor(a_values, get_arg_val<uint32_t>(2));
    const auto indices = TensorAccessor(a_indices, get_arg_val<uint32_t>(3));
    const auto packed = TensorAccessor(a_packed, get_arg_val<uint32_t>(4));

    Noc noc;
    DataflowBuffer stage(CB_STAGE);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
    constexpr uint32_t STAGE_TILE = 0, STAGE_PAIRS = TILE_BYTES;
    noc.async_read(zero, stage, TILE_BYTES, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_TILE});
    if constexpr (ROWS == 1 && PACKED_LANES == 2) {
        constexpr uint32_t STAGE_OUT = TILE_BYTES + PAIRS_BYTES;
        noc.async_read(pairs, stage, PAIRS_BYTES, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = STAGE_PAIRS});
        noc.async_read_barrier();

        uint32_t best_key = 0, best_bits = 0, best_id = 0;
        for (uint32_t c = 0; c < CORES; ++c) {
            const uint32_t bits = words[STAGE_PAIRS / 4 + 4 * c];
            const uint32_t key = key_of(bits);
            if (c == 0 || key > best_key) {
                best_key = key;
                best_bits = bits;
                best_id = words[STAGE_PAIRS / 4 + 4 * c + 1];
            }
        }
        tile[0] = best_bits >> 16;  // lane (0, 0) of the value tile
        IdFp32 id_fp32;
        id_fp32.f = static_cast<float>(best_id);  // the chain's typecast: exact below 2^24 (soft-float int -> fp32)
        words[STAGE_OUT / 4] = best_bits;
        words[STAGE_OUT / 4 + 1] = id_fp32.u;
        words[STAGE_OUT / 4 + 2] = 0;
        words[STAGE_OUT / 4 + 3] = 0;
        words[STAGE_OUT / 4 + 4] = best_id;
        noc.async_write(stage, values, TILE_BYTES, {.offset_bytes = STAGE_TILE}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(stage, packed, 8, {.offset_bytes = STAGE_OUT}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(stage, indices, 4, {.offset_bytes = STAGE_OUT + 16}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write_barrier();
    } else {
        // pairs row r at STAGE_PAIRS + r * PAIRS_BYTES; the packed rows then the indices row after them
        constexpr uint32_t STAGE_PACKED = STAGE_PAIRS + ROWS * PAIRS_BYTES;
        constexpr uint32_t STAGE_IDX = STAGE_PACKED + ROWS * PACKED_ROW_BYTES;
        for (uint32_t r = 0; r < ROWS; ++r) {
            noc.async_read(pairs, stage, PAIRS_BYTES, {.page_id = r, .offset_bytes = 0}, {.offset_bytes = STAGE_PAIRS + r * PAIRS_BYTES});
        }
        noc.async_read_barrier();
        for (uint32_t r = 0; r < ROWS; ++r) {
            const uint32_t row_words = (STAGE_PAIRS + r * PAIRS_BYTES) / 4;
            uint32_t best_key = 0, best_bits = 0, best_id = 0;
            for (uint32_t c = 0; c < CORES; ++c) {
                const uint32_t bits = words[row_words + 4 * c];
                const uint32_t key = key_of(bits);
                if (c == 0 || key > best_key) {
                    best_key = key;
                    best_bits = bits;
                    best_id = words[row_words + 4 * c + 1];
                }
            }
            // lane (r, 0) of the value tile: face (r >> 4) * 2, row r & 15 (16-bit word (r >> 4) * 512 + (r & 15) * 16)
            tile[(r >> 4) * 512 + (r & 15) * 16] = best_bits >> 16;
            IdFp32 id_fp32;
            id_fp32.f = static_cast<float>(best_id);
            const uint32_t packed_words = (STAGE_PACKED + r * PACKED_ROW_BYTES) / 4;
            for (uint32_t k = 0; k < PACKED_ROW_BYTES / 4; ++k) {
                words[packed_words + k] = 0;
            }
            words[packed_words] = best_bits;
            words[packed_words + 1] = id_fp32.u;
            words[STAGE_IDX / 4 + r] = best_id;
            noc.async_write(stage, packed, PACKED_ROW_BYTES, {.offset_bytes = STAGE_PACKED + r * PACKED_ROW_BYTES}, {.page_id = r, .offset_bytes = 0});
        }
        noc.async_write(stage, values, TILE_BYTES, {.offset_bytes = STAGE_TILE}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write(stage, indices, ROWS * 4, {.offset_bytes = STAGE_IDX}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write_barrier();
    }
    stage.push_back(1);
}
