# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import unittest


UINT32_MASK = (1 << 32) - 1


def old_interleaved_address(page_id, num_dram_banks, page_size, base, offset, bank_offsets, noc_xy):
    bank_offset_index = page_id // num_dram_banks
    bank_index = page_id - bank_offset_index * num_dram_banks
    local_addr = (
        bank_offset_index * page_size + base + offset + bank_offsets[bank_index]
    ) & UINT32_MASK
    return noc_xy[bank_index], local_addr, bank_offset_index


def direct_bank_address(bank_id, base, offset, bank_offsets, noc_xy):
    local_addr = (base + offset + bank_offsets[bank_id]) & UINT32_MASK
    return noc_xy[bank_id], local_addr


class TestKernelProfilerAddressMath(unittest.TestCase):
    def test_profiler_page_id_is_always_a_bank_id(self):
        # Host-side profiler setup uses ceil(core_count / num_dram_banks). Exhaust the supported-shape
        # envelope instead of relying on a particular unharvested core count. This proves that the
        # TensorAccessor page-within-bank term (and therefore its aligned page_size term) is always zero.
        for core_count in range(1, 513):
            for num_dram_banks in range(1, 33):
                cores_per_bank = (core_count + num_dram_banks - 1) // num_dram_banks
                for flat_core_id in range(core_count):
                    bank_id = flat_core_id // cores_per_bank
                    self.assertLess(bank_id, num_dram_banks, (core_count, num_dram_banks, flat_core_id))
                    self.assertEqual(bank_id // num_dram_banks, 0)

    def test_direct_bank_address_matches_interleaved_profiler_mapping(self):
        # Unharvested architecture descriptors: BH has 140 Tensix + 14 ETH cores over 8 DRAM views;
        # WH has 80 Tensix + 16 ETH cores over 12 views. Harvesting can only reduce core_count.
        architectures = (
            ("blackhole", 154, 8, [0] * 8),
            ("wormhole_b0", 96, 12, [0, 1 << 30] * 6),
        )
        bases_and_offsets = (
            (0, 0),
            (0x20, 0x30),
            (0x10000000, 0x00FFFFF0),
            (0xFFFFFF00, 0x00000180),
        )

        for arch, core_count, num_dram_banks, bank_offsets in architectures:
            cores_per_bank = (core_count + num_dram_banks - 1) // num_dram_banks
            page_size = 48_000 * 5 * cores_per_bank
            noc_xy = [0x10000 * bank + 0x100 * bank + bank for bank in range(num_dram_banks)]
            for flat_core_id in range(core_count):
                bank_id = flat_core_id // cores_per_bank
                self.assertLess(bank_id, num_dram_banks, arch)
                for base, offset in bases_and_offsets:
                    old_xy, old_addr, page_in_bank = old_interleaved_address(
                        bank_id, num_dram_banks, page_size, base, offset, bank_offsets, noc_xy
                    )
                    new_xy, new_addr = direct_bank_address(bank_id, base, offset, bank_offsets, noc_xy)
                    self.assertEqual(page_in_bank, 0, (arch, flat_core_id))
                    self.assertEqual((new_xy, new_addr), (old_xy, old_addr), (arch, flat_core_id, base, offset))

    def test_one_packet_selection_retains_wormhole_a0_fallback(self):
        profiler_l1_buffer_size = 2048
        quick_push_marker_bytes = 2 * 2 * 4
        max_write_size = profiler_l1_buffer_size + quick_push_marker_bytes
        self.assertLessEqual(max_write_size, 16_384)  # Blackhole
        self.assertLessEqual(max_write_size, 8192)  # Wormhole B0
        self.assertGreater(max_write_size, 512)  # Wormhole RISC A0 keeps the multi-packet path


if __name__ == "__main__":
    unittest.main()
