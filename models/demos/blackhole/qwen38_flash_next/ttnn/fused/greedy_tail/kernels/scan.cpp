// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Per core: the maximum of each valid row of a range of bf16 TILE pages of the local logits and the lowest id that
// holds it (ttnn.argmax's tie rule).  ROWS = 1 (the decode step): row 0 of a tile is the first row of faces 0 and 1 (32
// bytes each, at tile offsets 0 and 512); both are read with the 64-byte DRAM grain.  ROWS > 1 (the lanes, row r =
// lane r): rows 0..15 are the first ROWS x 32 bytes of faces 0 and 1, rows 16.. the same of faces 2 and 3; each tile
// is staged at its own 2 KB slot and every row is scanned in id order.  The pair (value as fp32 bits, id) of row r goes
// to lanes 4c..4c+1 of row r of the fp32 pairs rows (16 bytes per core per row).  bf16 values compare through a
// sign-magnitude key (zeros canonicalized to +0.0): the first strict maximum in id order is the lowest id.
// Named compile-time args: cb_stage, lanes_per_tile, rows.  Compile-time args: TensorAccessorArgs(logits), (pairs).
// Runtime args: 0 logits addr, 1 pairs addr, 2 first tile, 3 tile count, 4 core index.

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_STAGE = get_named_compile_time_arg_val("cb_stage");
constexpr uint32_t LANES = get_named_compile_time_arg_val("lanes_per_tile");
constexpr uint32_t ROWS = get_named_compile_time_arg_val("rows");
constexpr uint32_t GRAIN = 64;
constexpr uint32_t FACE1_OFFSET = 512;
constexpr uint32_t TILE_BYTES = 2048, FACE_BYTES = 512, ROW_BYTES = 32;

// -0.0 is canonicalized to +0.0 first: the chain compares as floats (the zeros tie, the first lane wins) and its max
// reduce returns +0.0 for a zero maximum.
FORCE_INLINE uint16_t canonical(uint16_t bits) { return (bits & 0x7FFFu) ? bits : uint16_t(0); }
FORCE_INLINE uint32_t key_of(uint16_t bits) { return (bits & 0x8000u) ? (~bits & 0xFFFFu) : (bits | 0x8000u); }

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t pairs_addr = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t core = get_arg_val<uint32_t>(4);
    constexpr auto a_logits = TensorAccessorArgs<0>();
    constexpr auto a_pairs = TensorAccessorArgs<a_logits.next_compile_time_args_offset()>();
    const auto logits = TensorAccessor(a_logits, logits_addr);
    const auto pairs = TensorAccessor(a_pairs, pairs_addr);

    Noc noc;
    DataflowBuffer stage(CB_STAGE);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    volatile tt_l1_ptr uint16_t* halves = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
    if constexpr (ROWS == 1) {
        // tile t of this core: its two 64-byte face-row reads land at 128 t (face 0) and 128 t + 64 (face 1)
        for (uint32_t t = 0; t < count; ++t) {
            noc.async_read(logits, stage, GRAIN, {.page_id = first + t, .offset_bytes = 0}, {.offset_bytes = 128 * t});
            noc.async_read(logits, stage, GRAIN, {.page_id = first + t, .offset_bytes = FACE1_OFFSET}, {.offset_bytes = 128 * t + 64});
        }
        noc.async_read_barrier();

        uint32_t best_key = 0, best_id = 0, best_bits = 0;
        bool any = false;
        for (uint32_t t = 0; t < count; ++t) {
            for (uint32_t lane = 0; lane < LANES; ++lane) {
                const uint32_t word = 64 * t + (lane < 16 ? lane : 32 + (lane - 16));  // face 0 row 0, then face 1 row 0
                const uint16_t bits = canonical(halves[word]);
                const uint32_t key = key_of(bits);
                if (!any || key > best_key) {
                    any = true;
                    best_key = key;
                    best_bits = bits;
                    best_id = (first + t) * LANES + lane;
                }
            }
        }
        const uint32_t out_offset = ((128 * count) + 15) & ~15u;
        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + out_offset);
        out[0] = best_bits << 16;  // the bf16 maximum widened to fp32 (exact)
        out[1] = best_id;
        out[2] = 0;
        out[3] = 0;
        noc.async_write(stage, pairs, 16, {.offset_bytes = out_offset}, {.page_id = 0, .offset_bytes = 16 * core});
        noc.async_write_barrier();
    } else {
        // tile t at slot 2048 t: rows 0..15 of faces 0 and 1 (their first LO_BYTES), rows 16.. of faces 2 and 3
        constexpr uint32_t LO_ROWS = ROWS < 16 ? ROWS : 16;
        constexpr uint32_t HI_ROWS = ROWS > 16 ? ROWS - 16 : 0;
        constexpr uint32_t LO_BYTES = (LO_ROWS * ROW_BYTES + GRAIN - 1) & ~(GRAIN - 1);
        constexpr uint32_t HI_BYTES = (HI_ROWS * ROW_BYTES + GRAIN - 1) & ~(GRAIN - 1);
        for (uint32_t t = 0; t < count; ++t) {
            const uint32_t slot = TILE_BYTES * t;
            noc.async_read(logits, stage, LO_BYTES, {.page_id = first + t, .offset_bytes = 0}, {.offset_bytes = slot});
            noc.async_read(logits, stage, LO_BYTES, {.page_id = first + t, .offset_bytes = FACE_BYTES}, {.offset_bytes = slot + FACE_BYTES});
            if constexpr (HI_ROWS > 0) {
                noc.async_read(logits, stage, HI_BYTES, {.page_id = first + t, .offset_bytes = 2 * FACE_BYTES}, {.offset_bytes = slot + 2 * FACE_BYTES});
                noc.async_read(logits, stage, HI_BYTES, {.page_id = first + t, .offset_bytes = 3 * FACE_BYTES}, {.offset_bytes = slot + 3 * FACE_BYTES});
            }
        }
        noc.async_read_barrier();

        const uint32_t out_offset = TILE_BYTES * count;  // 16-byte aligned
        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + out_offset);
        for (uint32_t r = 0; r < ROWS; ++r) {
            uint32_t best_key = 0, best_id = 0, best_bits = 0;
            bool any = false;
            const uint32_t row_half = (((r >> 4) << 1) * FACE_BYTES + (r & 15) * ROW_BYTES) / 2;  // face (r >> 4) * 2, row r & 15
            for (uint32_t t = 0; t < count; ++t) {
                const uint32_t slot_half = (TILE_BYTES * t) / 2;
                for (uint32_t lane = 0; lane < LANES; ++lane) {
                    // lanes 0..15 in the even face, 16..31 in the odd face (FACE_BYTES / 2 halves further on)
                    const uint32_t word = slot_half + row_half + (lane < 16 ? lane : (FACE_BYTES / 2) + (lane - 16));
                    const uint16_t bits = canonical(halves[word]);
                    const uint32_t key = key_of(bits);
                    if (!any || key > best_key) {
                        any = true;
                        best_key = key;
                        best_bits = bits;
                        best_id = (first + t) * LANES + lane;
                    }
                }
            }
            out[4 * r] = best_bits << 16;  // the bf16 maximum widened to fp32 (exact)
            out[4 * r + 1] = best_id;
            out[4 * r + 2] = 0;
            out[4 * r + 3] = 0;
            noc.async_write(stage, pairs, 16, {.offset_bytes = out_offset + 16 * r}, {.page_id = r, .offset_bytes = 16 * core});
        }
        noc.async_write_barrier();
    }
    stage.push_back(1);
}
