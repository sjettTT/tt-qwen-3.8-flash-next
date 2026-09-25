# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
KERNELS = REPO_ROOT / "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels"


def _source(name: str) -> str:
    return (KERNELS / name).read_text(encoding="utf-8")


def test_total_chunks_has_one_consumer_and_one_pop() -> None:
    reader = _source("tilize_reader.cpp")
    compute = _source("tilize_compute.cpp")
    writer = _source("tilize_writer.cpp")

    assert "cb_total_chunks.push_back(one_page);" in reader
    assert compute.count("cb_total_chunks.wait_front(one_page);") == 1
    assert compute.count("cb_total_chunks.pop_front(one_page);") == 1
    assert "total_chunks" not in writer


def test_writer_derives_work_from_per_expert_counts_only() -> None:
    writer = _source("tilize_writer.cpp")

    required = (
        "cb_per_expert_total_tokens.wait_front(1);",
        "num_tokens_per_expert[e] = per_expert_counts[e];",
        "uint32_t num_expert_tokens = num_tokens_per_expert[e];",
        "uint32_t num_expert_chunks = (num_expert_tokens + tokens_per_chunk - 1) / tokens_per_chunk;",
        "cb_per_expert_total_tokens.pop_front(one_page);",
    )
    for statement in required:
        assert statement in writer


def test_empty_rank_executes_no_writer_chunk_wait() -> None:
    writer = _source("tilize_writer.cpp")
    chunk_loop = "for (uint32_t chunk = 0; chunk < num_expert_chunks; chunk++)"

    assert chunk_loop in writer
    # the wait on the tilized chunk sits inside the per-chunk loop (a rank with no chunks never waits); the pipelined
    # feed waits per staging slot (chunk_slot_pages), the whole buffer (shared_cb_num_pages) is the slots together
    assert "cb_tilize_output.wait_front(chunk_slot_pages);" in writer[writer.index(chunk_loop) :]
    assert "cb_tilize_output.wait_front(" not in writer[: writer.index(chunk_loop)]


def test_study_delay_points_are_declared_defaulted_and_used_once() -> None:
    """The feed-protocol study delays (MOE_STUDY_DELAY): every point the kernels use is listed in the header with a
    zero default (the served kernels compile them to nothing), the factory admits exactly that list, and each point
    sits at one handoff."""
    header = _source("moe_ring_common.h")
    kernels = "".join(
        _source(name) for name in ("tilize_writer.cpp", "tilize_reader.cpp", "dm0.cpp", "dm1.cpp", "compute.cpp")
    )
    factory = (KERNELS.parent / "moe_compute_program_factory.cpp").read_text(encoding="utf-8")

    used = (
        re.findall(r"MOE_STUDY_DELAY\((\w+)\);", kernels)
        + re.findall(r"MOE_STUDY_FAULT\((X_\w+)\)", kernels)
        + re.findall(r"MOE_STUDY_PARAM\((\w+)\)", kernels)
    )
    declared = re.findall(r"#ifndef MOE_STUDY_DELAY_(\w+)\n#define MOE_STUDY_DELAY_\1 0\n#endif", header)
    admitted = re.findall(r'"([A-Z]_[A-Z0-9_]+)"', factory[factory.index("study_points = {") :].split("};")[0])
    assert used and len(used) == len(set(used)), used
    assert sorted(used) == sorted(declared) == sorted(admitted)
    for point in used:
        assert f"//   {point}" in header, point  # each point is described
    assert "TTNN_MOE_COMPUTE_STUDY_DELAYS" in factory


def test_a2a_exchange_has_backpressure_and_never_writes_a_core_s_own_buffer() -> None:
    """The ring exchange's fix (2026-09-25): the partials travel once per chunk (the first W2 iteration only) and the
    last a2a step sends no data (it would write the successor's own partial back into its buffer 0, which only that
    core's compute writes); each core credits its predecessor once per owned chunk when its compute has read every
    a2a buffer, and the predecessor waits for the credits before its first write of its next owned chunk. The three
    ring kernels parse one runtime-argument layout (the predecessor's coordinates follow the ring index), and the
    factory creates the credit semaphore on the ring cores and passes it."""
    dm1 = _source("dm1.cpp")
    factory = (KERNELS.parent / "moe_compute_program_factory.cpp").read_text(encoding="utf-8")

    step_loop = "for (uint32_t step = 0; step < num_a2a_steps_per_iter; ++step)"
    a2a = dm1[dm1.index(step_loop) : dm1.index("++a2a_chunks_exchanged;")]
    assert "if (i == 0 && step != num_cores - 1) {" in a2a  # data on the first iteration only, never at the last step
    assert "LOCAL_BUFFER_OFFSET[step + 1]" in a2a and "? 0 : (step + 1)" not in a2a
    # every data write of a step sits between that guard and the step's semaphore increment (none before the guard,
    # none from the increment on), and every step ends in the posted-writes flush the credit rests on
    guard = a2a.index("if (i == 0 && step != num_cores - 1) {")
    increment = a2a.index("noc_semaphore_inc</*posted=*/true>(neighbor_semaphore_noc_addr", guard)
    assert "noc_async_write_one_packet_with_state" not in a2a[:guard]
    assert "noc_async_write_one_packet_with_state" in a2a[guard:increment]
    assert "noc_async_write_one_packet_with_state" not in a2a[increment:]
    assert "noc1_obj.async_writes_flushed<NocOptions::POSTED>();" in a2a[increment:]
    # the wait precedes the chunk's first write; the credit follows the W2 output wait (every buffer read)
    assert dm1.index("a2a_free_sem.wait_min(a2a_chunks_exchanged);") < dm1.index(step_loop)
    credit = re.compile(r"noc_semaphore_inc</\*posted=\*/false>\(\s*predecessor_a2a_free_noc_addr,")
    credits = [m.start() for m in credit.finditer(dm1)]
    assert len(credits) == 1
    output_wait = dm1.index("cb_c2s_out.wait_front(num_w0_w1_tiles_h);", dm1.index("++a2a_chunks_exchanged;"))
    assert dm1.index("++a2a_chunks_exchanged;") < output_wait < credits[0]
    # every exit path barriers the credits' responses: the local-output / compute-only branch through the full
    # barrier, the combine branch through an atomic barrier (each branch of the exit `if constexpr (!has_combine)`)
    exit_if = dm1.rindex("if constexpr (!has_combine) {")
    exit_else = dm1.index("} else {", exit_if)
    assert "noc1_obj.async_full_barrier();" in dm1[exit_if:exit_else]
    assert "noc1_obj.async_atomic_barrier();" in dm1[exit_else:]

    layout = re.compile(r"const auto (\w+) = get_arg_val<uint32_t>\(argidx\+\+\);")
    layouts = {name: layout.findall(_source(name))[:12] for name in ("dm0.cpp", "dm1.cpp", "compute.cpp")}
    expected = [
        "dram_bank_id",
        "vchannel",
        "w0_w1_addr",
        "w2_addr",
        "out_addr",
        "ring_semaphore_id",
        "ring_core_id",
        "ring_neighbor_physical_x",
        "ring_neighbor_physical_y",
        "ring_index",
        "ring_predecessor_physical_x",
        "ring_predecessor_physical_y",
    ]
    assert all(names == expected for names in layouts.values()), layouts
    pushes = factory[
        factory.index("std::vector<uint32_t> matmul_runtime_args;") : factory.index("// Append shard_to_bank")
    ]
    # 9 commented slots + the one push inside the tensor loop (3 tensors) = the 12 leading args the kernels parse
    assert pushes.count("matmul_runtime_args.push_back(") == 10
    assert "for (const auto& tensor : matmul_tensors)" in pushes
    comments = re.findall(r"matmul_runtime_args\.push_back\([^;]*\);\s*//\s*([^\n]*)", pushes)
    assert [c.split("(")[0].strip().lower() for c in comments] == [
        "dram bank id placeholder",
        "vchannel placeholder",
        "semaphore id",
        "ring core id placeholder",
        "neighbor physical x",
        "neighbor physical y",
        "ring index",
        "predecessor physical x",
        "predecessor physical y",
    ], comments
    assert "a2a_free_semaphore_id = tt::tt_metal::CreateSemaphore(program, ring_core_range_set, INVALID);" in factory
    assert '{"a2a_free_semaphore_id", a2a_free_semaphore_id}' in factory
    assert "matmul_runtime_args[10] = static_cast<uint32_t>(prev_physical.x);" in factory
    assert "matmul_runtime_args[11] = static_cast<uint32_t>(prev_physical.y);" in factory
    assert "ring_pos2bank_id[(ring_pos + matmul_num_cores - 1) % matmul_num_cores]" in factory
    # the override checks the exact argument count it was built with (a missing tail would otherwise be written over
    # the bank table)
    assert ".matmul_runtime_args_size = static_cast<uint32_t>(matmul_runtime_args.size())" in factory
    assert "matmul_runtime_args.size() == shared_variables.matmul_runtime_args_size" in factory
