# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of ``gdn_rows_scan`` (the verify rows' fold): the CB table shared by the kernels and the Python
side, the kernel argument layouts, the registry entry (opt-in, COMPONENT, never a default without a proof), the
shape-only admission, what ``attach`` gives a rows state and when, the layer's dispatch and commit wiring in
ttnn/gdn.py, the pick's accept-count read, and ``reference_rows`` as rows sequential ``gdn_step.reference_step`` calls."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_scan as module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_wrap as wrap
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

NAME = module.NAME
KERNELS = Path(module.__file__).parent / "kernels"
TILE, HEADS, HEAD_DIM = module.TILE, module.HEADS, module.HEAD_DIM
GDN_SOURCE = Path(gdn_module.__file__).read_text(encoding="utf-8")


def _constants(source: str) -> dict[str, int]:
    text = (KERNELS / source).read_text()
    return {name: int(value) for name, value in re.findall(r"\b(CB_[A-Z0-9]+)\s*=\s*(\d+)", text)}


# ------------------------------------------------------------------------------------------- the CB table


def test_cb_indices_agree_between_kernels_and_python():
    compute, reader, writer = _constants("compute.cpp"), _constants("reader.cpp"), _constants("writer.cpp")
    for name, index in reader.items():
        assert compute[name] == index, name
    for name, index in writer.items():
        assert compute[name] == index, name
    for rows, vbt in ((5, 4), (2, 1), (8, 4)):
        table = module.cb_table(rows, vbt)
        declared = {index for index, _, _ in table} | {module.CB_SNEWC, module.CB_OUTS, module.CB_DEBUG}
        assert len(declared) == len(table) + 3 and max(declared) <= 31
        used = set(compute.values())
        assert used <= declared, sorted(used - declared)
        # the depths that follow the item: the conv tiles, the state tiles, one a/b pair and one mask per row
        pages = {index: pages for index, _, pages in table}
        assert pages[0] == 8 + vbt and pages[1] == 3 * (8 + vbt) and pages[2] == 4 * (8 + vbt) and pages[11] == 8 + vbt
        assert pages[7] == pages[20] == pages[21] == pages[24] == 4 * vbt
        assert pages[4] == 2 * rows and pages[8] == rows and pages[31] == 2 * vbt
        assert pages[18] == rows and pages[19] == rows  # one beta / decay tile per row, computed before the recurrences
    assert (
        compute["CB_SNEWC"] == module.CB_SNEWC == 30
        and compute["CB_OUTS"] == module.CB_OUTS == 25
        and compute["CB_OROWS"] == module.CB_OROWS == 29
        and compute["CB_OBF"] == module.CB_OBF == 31
        and compute["CB_STATE"] == module.CB_STATE
    )
    # the o hand-off to the writer is its own buffer (gdn_step aliases it with the conv sum, which only the compute pops)
    assert compute["CB_OBF"] != compute["CB_CONVSUM"]
    # exact fp32 copies: the CBs the compute consumes with copy_tile; matmul / reduce / bcast operands stay Default
    assert set(module.FP32_COPY_CBS) == {
        compute[n] for n in ("CB_STATE", "CB_DTNA", "CB_BETA", "CB_DECAY", "CB_SDECC", "CB_VREAD", "CB_SNEWC")
    }
    assert not set(module.FP32_COPY_CBS) & {
        compute[n] for n in ("CB_QROW", "CB_KROW", "CB_KCOL", "CB_SDEC", "CB_DELTAB", "CB_SNEW", "CB_SCALER", "CB_RS")
    }
    # the state carry is read back by copy_tile (CB_SNEWC) and multiplied by matmul (CB_SNEW): two packs, two modes
    text = (KERNELS / "compute.cpp").read_text()
    assert "pack_tile(j, CB_SNEW);" in text and "pack_tile(j, CB_SNEWC);" in text
    assert "matmul_tiles(CB_QROW, CB_SNEW" in text and "copy_tile(src, t, 0)" in text


def test_per_core_l1_of_the_largest_form_fits():
    for rows, vbt in ((module.MAX_ROWS, 4), (5, 4), (5, 1)):
        table = module.cb_table(rows, vbt)
        total = (
            sum(pages * fp.TILE_BYTES[dtype] for _, dtype, pages in table) + 2 * 4 * vbt * fp.TILE_BYTES[ttnn.float32]
        )
        assert total <= 1_100_000, (rows, vbt, total)  # a Blackhole Tensix holds 1.5 MB of L1


def test_kernel_argument_layouts():
    reader = (KERNELS / "reader.cpp").read_text()
    assert reader.count("TensorAccessorArgs<") == 9  # projected, history, 4 taps, dtna, norm, state
    assert reader.count("get_arg_val<uint32_t>(arg++)") == 10 + 2  # 9 addresses, items, then (head, vb)
    assert "TensorAccessorArgs<2>" in reader and "get_compile_time_arg_val(0)" in reader
    writer = (KERNELS / "writer.cpp").read_text()
    assert writer.count("TensorAccessorArgs<") == 3 and "TensorAccessorArgs<2>" in writer  # prefix, out, debug
    assert writer.count("get_arg_val<uint32_t>(arg++)") == 3 + 1 + 2 + 2  # + the debug address, the peers' (x, y)
    assert "noc_semaphore_wait_min(sem, PEERS)" in writer and "get_semaphore(0)" in writer
    pick = (KERNELS / "pick.cpp").read_text()
    assert pick.count("TensorAccessorArgs<") == 3 and "TensorAccessorArgs<1>" in pick  # accepted, prefix, recurrent
    assert pick.count("get_arg_val<uint32_t>(arg++)") == 4 + 2
    compute = (KERNELS / "compute.cpp").read_text()
    assert "get_arg_val<uint32_t>(0)" in compute
    assert "get_compile_time_arg_val(0)" in compute and "get_compile_time_arg_val(1)" in compute
    # the pick converts the fp32 accept count to a slot index on the RISC and clamps it to the rows
    assert "reinterpret_cast<volatile tt_l1_ptr float*>" in pick and "accepted >= ROWS" in pick
    # the row helpers are the shared header's
    for source in ("reader.cpp", "writer.cpp"):
        assert '#include "../../kernels/row_mask.h"' in (KERNELS / source).read_text()
    header = (KERNELS.parent.parent / "kernels" / "row_mask.h").read_text()
    assert "one_hot_row_bf16" in header and "copy_row_bf16" in header and "face_element" in header


def test_the_builder_passes_the_rows_and_the_split_as_compile_time_args():
    source = inspect.getsource(module.run)
    assert "reader_cta = [rows, vbt]" in source and "writer_cta = [rows, vbt," in source and "[rows, vbt]," in source
    assert "gr_read.noc_map(mesh)" in source and "fp.semaphore_descriptor(0, cores)" in source
    assert "unpack_to_dest_fp32=FP32_COPY_CBS" in source and "fidelity=ttnn.MathFidelity.HiFi4" in source
    assert "fp32_dest=True" in source and 'fp.program_meta(\n        NAME,\n        "verify_rows",' in source
    assert module.SPLIT in (1, 4) and module.HT % module.SPLIT == 0
    assert module.MAX_ROWS == 8 and module.PROGRAMS_PER_LAYER == 7 and module.COMMIT_PROGRAMS_PER_LAYER == 3
    assert (
        module.PROGRAMS_PER_LAYER == wrap.PROGRAMS_PER_LAYER - 6 + 1 + 1
    )  # the wrap's six become one (+ the landing slice)
    pick = inspect.getsource(module.run_pick)
    assert 'fp.program_meta(\n        NAME,\n        "commit_pick",' in pick and "fp.reader_kernel(PICK" in pick


# --------------------------------------------------------------------------------------- the registry entry


def test_registry_entry_is_an_opt_in_component_kernel():
    entry = fused.kernel(NAME)
    assert entry.tolerance == fused.COMPONENT and entry.default_on is False and NAME not in fused.DEFAULT_ON
    assert entry.fused is module.rows_body_scan and entry.composed is module.rows_body_fallback
    assert entry.admits is module.admits and entry.gate is None and entry.component_proof is None
    assert "gdn_pre_rows" in entry.replaces and "prefix" in entry.replaces
    # off by default: the composed callable (today's stream) serves outright
    assert fused.resolve_admitted(NAME, {}) is module.rows_body_fallback
    assert fused.resolve(NAME, {}) is module.rows_body_fallback
    # opt-in: the admitted dispatcher
    on = fused.resolve_admitted(NAME, {fused.ENV: NAME})
    assert isinstance(on, fused.AdmittedStep) and on.fused is module.rows_body_scan
    assert on.composed is module.rows_body_fallback and on.admits is module.admits
    assert fused.resolve_admitted(NAME, {fused.ENV: NAME, fused.OFF_ENV: NAME}) is module.rows_body_fallback
    assert fused.enabled(NAME, {}) is False and fused.enabled(NAME, {fused.ENV: NAME}) is True
    # the wrap stays the default beside it
    assert fused.kernel(wrap.NAME).default_on is True


# ------------------------------------------------------------------------------------- admission and attach


class _Fake:
    def __init__(self, shape=(1,), dtype=None):
        self.shape, self.dtype = tuple(shape), dtype

    def is_allocated(self) -> bool:
        return True

    def buffer_address(self) -> int:
        return 0


class _HostFake(_Fake):
    buffer_address = None


class _Constants:
    def __init__(self, *, rows=5, tile_rows=TILE):
        self.rows, self.tile_rows = rows, tile_rows


class _RowsState:
    def __init__(self, *, rows=5, tile_rows=TILE, flat_qk=False, owns_body=True, host=False):
        self.constants = _Constants(rows=rows, tile_rows=tile_rows)
        self.flat_qk, self.owns_body = flat_qk, owns_body
        fake = _HostFake if host else _Fake
        self.v = fake((1, 1, tile_rows, module.VALUE_WIDTH), ttnn.bfloat16)
        self.history = fake((1, 1, TILE, module.QKV_WIDTH), ttnn.bfloat16)
        self.qkv = fake((1, 1, tile_rows, module.QKV_WIDTH), ttnn.bfloat16)


def _state(shape=(1, HEADS, HEAD_DIM, HEAD_DIM), dtype=ttnn.float32):
    return SimpleNamespace(recurrent=_Fake(shape, dtype))


def test_qualifies_is_the_verify_form_with_up_to_max_rows():
    assert (
        module.qualifies(_RowsState()) and module.qualifies(_RowsState(rows=1)) and module.qualifies(_RowsState(rows=8))
    )
    assert not module.qualifies(_RowsState(rows=9))  # the prefix states are rows x 786 KB per layer
    assert not module.qualifies(_RowsState(rows=32)) and not module.qualifies(_RowsState(rows=128, tile_rows=128))
    assert not module.qualifies(_RowsState(flat_qk=True)) and not module.qualifies(_RowsState(owns_body=False))
    assert not module.qualifies(_RowsState(host=True))  # a host fake keeps today's stream
    assert not module.qualifies(SimpleNamespace())


def test_attach_needs_the_switch_and_the_form(monkeypatch):
    allocated = []
    monkeypatch.setattr(
        fp, "allocate", lambda shape, dtype, layout, mesh, *a: allocated.append((shape, dtype)) or _Fake(shape, dtype)
    )
    monkeypatch.setattr(fp, "stamp_topology", lambda tensor, reference, shard_dim=None: tensor)
    monkeypatch.setattr(gdn_step, "_constants", lambda gdn: "dt-na-tiles")
    gdn = SimpleNamespace(mesh_device="mesh", out_proj_act_memory_config="out-proj-shard")
    assert module.attach(gdn, _RowsState(), {}) is None and allocated == []  # off: nothing allocated
    assert module.attach(gdn, _RowsState(rows=9), {fused.ENV: NAME}) is None and allocated == []
    state = _RowsState(rows=5)
    buffers = module.attach(gdn, state, {fused.ENV: NAME})
    assert buffers is module.buffers_of(state) is state.scan_buffers
    assert allocated == [((5, HEADS, HEAD_DIM, HEAD_DIM), ttnn.float32)]
    assert buffers.rows == 5 and buffers.constants == "dt-na-tiles" and buffers.gated_memory_config == "out-proj-shard"
    assert module.buffers_of(_RowsState()) is None


def test_admits_is_the_fused_bodys_input_contract():
    state = _RowsState()
    assert module.admits(None, None, state, _state()) is False  # no buffers attached
    state.scan_buffers = "buffers"
    assert module.admits(None, None, state, _state()) is True
    assert module.admits(None, None, state, _state((2, HEADS, HEAD_DIM, HEAD_DIM))) is False
    assert module.admits(None, None, state, _state(dtype=ttnn.bfloat16)) is False
    assert module.admits(None, None, state, SimpleNamespace()) is False
    other = _RowsState(rows=9)
    other.scan_buffers = "buffers"
    assert module.admits(None, None, other, _state()) is False


def test_commit_needs_the_selectors_accept_count():
    with pytest.raises(RuntimeError):
        module.commit(None, None, SimpleNamespace(prefix=None), _state(), SimpleNamespace(accepted=None))
    selectors = gdn_module.Qwen38TTNNRowsSelectors(5, None, (), None)
    assert selectors.accepted is None  # defaulted: older constructions and the wrap's selectors carry none


# ------------------------------------------------------------------------------------------ the gdn.py wiring


def _methods():
    tree = ast.parse(GDN_SOURCE)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen38TTNNGDN")
    return {n.name: ast.get_source_segment(GDN_SOURCE, n) for n in cls.body if isinstance(n, ast.FunctionDef)}


def test_the_layer_resolves_the_fold_once_and_dispatches_through_it():
    methods = _methods()
    assert f'self._rows_scan_call = fused.resolve_admitted("{NAME}")' in methods["__init__"]
    assert 'self.__dict__.get("_rows_scan_call")' in methods["_rows_body"]
    assert f'fused.resolve_admitted("{NAME}")' in methods["_rows_body"]
    assert 'self.__dict__.get("_rows_body_call")' in methods["_rows_body_wrap"]
    assert "body = self._rows_body()" in methods["forward_rows"]
    # the fallback is the layer's wrap-or-chain getter
    assert "gdn._rows_body_wrap()(gdn, full_hidden, rows_state, state, full_tile=full_tile)" in inspect.getsource(
        module.rows_body_fallback
    )
    # the fold is attached first; the wrap only where it did not attach
    allocate = methods["allocate_rows_state"]
    assert "if fused.gdn_rows_scan.attach(self, rows_state) is None:" in allocate
    assert "fused.gdn_rows_wrap.attach(self, rows_state)" in allocate
    assert allocate.index("gdn_rows_scan.attach") < allocate.index("gdn_rows_wrap.attach")


def test_the_commit_under_the_fold_is_the_pick_and_the_history_advance():
    methods = _methods()
    commit = methods["commit_rows"]
    fold = commit[commit.index("scan = fused.gdn_rows_scan.buffers_of(rows_state)") :]
    fold = fold[: fold.index("if step_committed_rows:")]
    assert "fused.gdn_rows_scan.commit(self, rows_state, scan, state, selectors)" in fold
    assert "self._advance_history_rows(rows_state, selectors)" in fold and "state.validate()" in fold
    assert "raise ValueError" in fold and "step_on_full_rejection" in fold  # the 1-row anchors are refused
    assert "_chunk_rows" not in fold  # no masked re-run
    full = methods["commit_rows_full"]
    assert "if final_state is None and scan is not None:" in full
    assert "fused.gdn_rows_scan.commit_all_rows(scan, state)" in full
    # the fused body returns no final state: the rows result says so
    assert ", None\n" in inspect.getsource(module.rows_body_scan)
    assert "under the verify-rows fold" in GDN_SOURCE
    # the selectors carry the accept count the pick reads
    assert "accepted=accepted" in inspect.getsource(gdn_module.build_rows_selectors)
    assert "accepted: Any = None" in GDN_SOURCE and "scan_buffers: Any = None" in GDN_SOURCE


def test_the_fused_body_is_the_wraps_frame_around_one_program():
    calls = [
        ast.unparse(node.func)
        for node in sorted(
            (n for n in ast.walk(ast.parse(inspect.getsource(module.rows_body_scan))) if isinstance(n, ast.Call)),
            key=lambda n: (n.lineno, n.col_offset),
        )
    ]
    assert [c for c in calls if c.startswith("gdn._")] == [
        "gdn._project_rows_linear",
        "gdn._land_rows_qkv",
        "gdn._out_proj_tile",
        "gdn._rows_output_tile",
    ]
    # the placements come from the launch itself (the program meta's outputs), not from a hand stamp after it
    assert calls.count("run") == 1 and "fp.stamp_topology" not in calls
    source = inspect.getsource(module)
    assert "outputs=((out, 3), (prefix, 1), *([(debug, None)] if debug is not None else []))," in source
    assert "outputs=((recurrent, 1),)," in source  # commit_pick: the state slot written in place


# ------------------------------------------------------------------------------------------ the reference


def test_reference_rows_is_rows_sequential_reference_steps():
    g = torch.Generator().manual_seed(9)
    rows = 3
    projected = (torch.randn(TILE, module.PROJECTION_WIDTH, generator=g) * 0.6).to(torch.bfloat16)
    history = (torch.randn(TILE, module.QKV_WIDTH, generator=g) * 0.6).to(torch.bfloat16)
    taps = [(torch.randn(module.QKV_WIDTH, generator=g) * 0.5).to(torch.bfloat16) for _ in range(4)]
    dt, na = torch.randn(HEADS, generator=g), -torch.exp(torch.rand(HEADS, generator=g) * 3)
    norm = (1 + torch.randn(HEAD_DIM, generator=g) * 0.1).to(torch.bfloat16)
    state = torch.randn(HEADS, HEAD_DIM, HEAD_DIM, generator=g) * 0.4
    prefix, gated = module.reference_rows(projected, history, taps, dt, na, norm, state, rows)
    assert prefix.shape == (rows, HEADS, HEAD_DIM, HEAD_DIM) and prefix.dtype == torch.float32
    assert gated.shape == (rows, module.VALUE_WIDTH) and gated.dtype == torch.bfloat16
    # by hand: the ring of row r is window rows r .. r + 2 of [history rows 0..2 | the rows' q|k|v]
    window = torch.cat([history[:3], projected[:, : module.QKV_WIDTH]], dim=0)
    current = state.unsqueeze(0)
    for row in range(rows):
        older = [window[row + i : row + i + 1] for i in range(3)]
        current, expected, *_ = gdn_step.reference_step(projected[row : row + 1], older, taps, dt, na, norm, current)
        assert torch.equal(prefix[row], current[0]) and torch.equal(gated[row], expected[0])
    # rows past the real ones do not enter: the same prefix from a tile whose tail differs
    other = projected.clone()
    other[rows:] = 1.0
    prefix2, gated2 = module.reference_rows(other, history, taps, dt, na, norm, state, rows)
    assert torch.equal(prefix, prefix2) and torch.equal(gated, gated2)


def test_the_served_admission_charges_the_prefix_states_when_the_fold_is_on():
    """The MTP DRAM admission (tools/qwen38_chat_session.py) carries the fold's persistent prefix states, derived from
    k + 1 and the GDN layer count, when QWEN38_FUSED names the kernel: the 4x p150 line measured 87,394,112 bytes per
    bank at k = 4 with the fold (states 31,093,824) where the estimate without the term was 77,095,515."""

    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session

    assert session.fold_prefix_states_bytes_per_bank(4) == 36 * -(-(5 * HEADS * (HEAD_DIM // TILE) ** 2) // 8) * 4096
    assert session.fold_prefix_states_bytes_per_bank(module.MAX_ROWS) == 0
    record = session.mtp_capacity_admission(32768, drafts=4, verify_forms=2, gdn_rows_scan=True)
    assert record["mtp_growth_estimate_bytes_per_bank"]["states"] >= 31_093_824
    assert record["required_free_bytes_per_bank"] >= 87_394_112 and record["fits"]
    assert session.mtp_capacity_admission(32768, drafts=4, verify_forms=2)["required_free_bytes_per_bank"] < 87_394_112
