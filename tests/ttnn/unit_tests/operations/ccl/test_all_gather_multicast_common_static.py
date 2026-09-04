# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for all-gather FabricWriter state initialization."""

from pathlib import Path


SCATTER_SET_STATE = "fabric_multicast_noc_scatter_write_set_state"
UNICAST_SET_STATE = "fabric_multicast_noc_unicast_write_set_state"


def _source() -> str:
    relative = Path("ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/multicast_common.hpp")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    raise AssertionError(f"cannot locate {relative}")


def _block(source: str, anchor: str, start: int = 0) -> tuple[str, int]:
    anchor_index = source.index(anchor, start)
    opening = source.index("{", anchor_index)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index], index + 1
    raise AssertionError(f"unterminated block after {anchor!r}")


def test_constructor_guards_both_scatter_initializers_and_preserves_unicast_state() -> None:
    source = _source()
    constructor = source[source.index("    FabricWriter(") : source.index("    ~FabricWriter()")]

    assert constructor.count(SCATTER_SET_STATE) == 2
    assert constructor.count(UNICAST_SET_STATE) == 2

    first_guard, after_first = _block(constructor, "if constexpr (use_scatter_write)")
    second_guard, _ = _block(constructor, "if constexpr (use_scatter_write)", after_first)
    assert first_guard.count(SCATTER_SET_STATE) == 1
    assert second_guard.count(SCATTER_SET_STATE) == 1
    assert UNICAST_SET_STATE not in first_guard
    assert UNICAST_SET_STATE not in second_guard

    assert "fabric_multicast_noc_scatter_write_with_state" in source
    assert "fabric_multicast_noc_unicast_write_with_state" in source
