# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Persistent-descriptor PLE row reads must return the host oracle's bytes.

Also the decode-step form of the lookup: the scalar-integer n-gram hash against
``reference.ngram_token_ids`` over the real id ranges, the row payload against
the host oracle, and the ROW_MAJOR persistent row with its one in-body tilize.
"""

import ast
import inspect
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint, Qwen38CheckpointRowReader
from models.demos.blackhole.qwen38_flash_next.reference import build_ngram_hash_spec, ngram_token_ids
from models.demos.blackhole.qwen38_flash_next.tt.ple import Qwen38HostPLEEmbedding
from models.demos.blackhole.qwen38_flash_next.ttnn.ple import (
    Qwen38ResidentPLELookup,
    Qwen38TTNNPLE,
    ngram_token_ids_decode_step,
)

CHECKPOINT = Path(os.environ.get("QWEN38_CHECKPOINT", "/nonexistent/Qwen3.8-Flash-Next"))
PARTS = 4
ROWS_PER_PART = 1024
HEAD_DIM = 8
EOS = 999
# The pinned model's hash geometry (config.py pins every field): 248,320 unigram
# ids, 16 heads over primes above 19,999,999, EOS 248044.
REAL_VOCAB_SIZE = 248_320
REAL_EOS = 248_044
REAL_SPEC = build_ngram_hash_spec(
    unigram_vocab_size=REAL_VOCAB_SIZE,
    ngram_size=3,
    heads_per_ngram=8,
    ngram_vocab_size_base=20_000_000,
    ple_layer_index=0,
    seed=1234,
    divisible_by=128,
)
# The 33-token chain the per-position sequential mode produced on 4x p150 at aef39cde.
CHAIN_33 = (
    17,
    15,
    16,
    22,
    95859,
    16,
    17,
    96212,
    3709,
    101857,
    97191,
    98478,
    95759,
    26076,
    98817,
    97035,
    123772,
    9616,
    17,
    15,
    16,
    22,
    12,
    17,
    15,
    18,
    20,
    95859,
    7313,
    24273,
    10992,
    271,
    107680,
)


class _SyntheticCheckpoint(Qwen38Checkpoint):
    """Real header/guard/pread code over a tiny corpus without the pinned index."""

    def __init__(self, root: Path, weight_map: dict[str, str], file_guard=None) -> None:
        self.root = Path(root).resolve()
        self._file_guard = file_guard
        self.weight_map = dict(weight_map)
        self.shards = tuple(sorted(set(weight_map.values())))


def _table_name(part: int) -> str:
    return f"table.shard_{part}.weight"


def _synthetic_corpus(root: Path) -> dict[str, str]:
    torch.manual_seed(1)
    weight_map = {}
    for part in range(PARTS):
        shard = f"part-{part}.safetensors"
        # The leading tensor gives the table a nonzero data offset inside the file.
        save_file(
            {
                f"leading_{part}.bias": torch.randn(37, dtype=torch.bfloat16),
                _table_name(part): torch.randn(ROWS_PER_PART, HEAD_DIM, dtype=torch.bfloat16),
            },
            str(root / shard),
        )
        weight_map[f"leading_{part}.bias"] = shard
        weight_map[_table_name(part)] = shard
    return weight_map


def _synthetic_host_embedding(checkpoint: Qwen38Checkpoint) -> SimpleNamespace:
    spec = build_ngram_hash_spec(
        unigram_vocab_size=1000,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=101,
        ple_layer_index=0,
        seed=1234,
        divisible_by=PARTS * ROWS_PER_PART // 2,
    )
    assert spec.padded_vocab_size == PARTS * ROWS_PER_PART
    return SimpleNamespace(
        checkpoint=checkpoint,
        table_names=tuple(_table_name(part) for part in range(PARTS)),
        rows_per_shard=ROWS_PER_PART,
        embedding_head_dim=HEAD_DIM,
        spec=spec,
        config=SimpleNamespace(eos_token_id=EOS, vocab_size=1000),
    )


def _assert_bitwise_equal(got: torch.Tensor, expected: torch.Tensor) -> None:
    assert got.dtype == expected.dtype == torch.bfloat16
    assert got.shape == expected.shape
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))


def test_row_reader_matches_tensor_rows_bitwise_after_one_guarded_open(tmp_path) -> None:
    weight_map = _synthetic_corpus(tmp_path)
    events = []
    checkpoint = _SyntheticCheckpoint(tmp_path, weight_map, file_guard=lambda path, phase: events.append(phase))
    name = _table_name(2)
    reader = Qwen38CheckpointRowReader(checkpoint, name)
    assert [phase for phase in events if "tensor-rows" in phase] == [
        f"before-tensor-rows-open:{name}",
        f"after-tensor-rows-open:{name}",
    ]
    assert (reader.rows, reader.row_shape, reader.row_bytes) == (ROWS_PER_PART, (HEAD_DIM,), 2 * HEAD_DIM)

    torch.manual_seed(2)
    indices = torch.randint(0, ROWS_PER_PART, (3, 5), dtype=torch.long)
    indices[0, 1] = indices[0, 0]
    indices[1, 2] = 0
    indices[2, 4] = ROWS_PER_PART - 1
    events.clear()
    got = reader.read(indices)
    assert not [phase for phase in events if "tensor-rows" in phase]
    _assert_bitwise_equal(got, checkpoint.tensor_rows(name, indices))
    _assert_bitwise_equal(
        reader.read(torch.tensor([7], dtype=torch.long)), checkpoint.tensor_rows(name, torch.tensor([7]))
    )
    assert reader.read(torch.zeros((0,), dtype=torch.long)).shape == (0, HEAD_DIM)
    assert reader.read_rows([5, 5, 9]) == [reader.read_rows([5])[0]] * 2 + reader.read_rows([9])

    with pytest.raises(IndexError, match="outside"):
        reader.read_rows([ROWS_PER_PART])
    with pytest.raises(ValueError, match="torch.long"):
        reader.read(torch.tensor([1], dtype=torch.int32))
    reader.close()
    reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        reader.read_rows([0])


def test_row_reader_rejects_non_bf16_and_scalar_tensors(tmp_path) -> None:
    weight_map = _synthetic_corpus(tmp_path)
    checkpoint = _SyntheticCheckpoint(tmp_path, weight_map)
    with pytest.raises(ValueError, match="rank>=2 BF16"):
        Qwen38CheckpointRowReader(checkpoint, "leading_0.bias")


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="procfs descriptor links are Linux only")
def test_row_reader_opens_the_guard_supplied_procfs_duplicate(tmp_path) -> None:
    weight_map = _synthetic_corpus(tmp_path)
    admitted = os.open(tmp_path / "part-0.safetensors", os.O_RDONLY | os.O_CLOEXEC)
    try:
        checkpoint = _SyntheticCheckpoint(
            tmp_path,
            weight_map,
            file_guard=lambda path, phase: Path(f"/proc/self/fd/{admitted}") if "before" in phase else None,
        )
        reader = Qwen38CheckpointRowReader(checkpoint, _table_name(0))
        indices = torch.tensor([[1, 2], [3, 1]])
        _assert_bitwise_equal(reader.read(indices), checkpoint.tensor_rows(_table_name(0), indices))
        reader.close()
    finally:
        os.close(admitted)


def test_row_reader_fails_closed_when_the_admitted_inode_changes(tmp_path) -> None:
    weight_map = _synthetic_corpus(tmp_path)
    checkpoint = _SyntheticCheckpoint(tmp_path, weight_map)
    replaced = Qwen38CheckpointRowReader(checkpoint, _table_name(1))
    rewritten = Qwen38CheckpointRowReader(checkpoint, _table_name(3))
    assert replaced.read_rows([3, 4]) == replaced.read_rows([3, 4])

    # Replacing the path unlinks the admitted inode (link count and ctime
    # change), the same drift the gate's revalidate_identities refuses.
    shard = tmp_path / "part-1.safetensors"
    replacement = tmp_path / "replacement.tmp"
    replacement.write_bytes(b"\0" * shard.stat().st_size)
    os.replace(replacement, shard)
    with pytest.raises(RuntimeError, match="identity drifted"):
        replaced.read_rows([3, 4])

    # Writing into the admitted inode changes its identity and is refused.
    time.sleep(0.05)
    with open(tmp_path / "part-3.safetensors", "r+b") as stream:
        stream.seek(8)
        stream.write(b"!")
    with pytest.raises(RuntimeError, match="identity drifted"):
        rewritten.read_rows([0])
    with pytest.raises(RuntimeError, match="identity drifted"):
        rewritten.advise([0])
    replaced.close()
    rewritten.close()


def test_resident_lookup_matches_the_host_oracle_bitwise(tmp_path) -> None:
    weight_map = _synthetic_corpus(tmp_path)
    events = []
    checkpoint = _SyntheticCheckpoint(tmp_path, weight_map, file_guard=lambda path, phase: events.append(phase))
    host = _synthetic_host_embedding(checkpoint)
    resident = Qwen38ResidentPLELookup(host)
    assert len(resident.readers) == PARTS
    assert sum("tensor-rows-open" in phase for phase in events) == 2 * PARTS

    tokens = torch.tensor([[17, 29, 31, EOS, 43, 47, 0, 998]])
    expected, expected_context = Qwen38HostPLEEmbedding.lookup(host, tokens)
    events.clear()
    got, got_context = resident.lookup(tokens)
    assert not events
    assert got.shape == (1, tokens.shape[1], 16 * HEAD_DIM)
    _assert_bitwise_equal(got, expected)
    assert torch.equal(got_context, expected_context)

    context = oracle_context = None
    for position in range(tokens.shape[1]):
        step = tokens[:, position : position + 1]
        got, context = resident.lookup(step, context)
        expected, oracle_context = Qwen38HostPLEEmbedding.lookup(host, step, oracle_context)
        _assert_bitwise_equal(got, expected)
        assert torch.equal(context, oracle_context)
    with pytest.raises(ValueError, match="host-resident"):
        resident.lookup(tokens.to("meta"))
    resident.close()


def test_prepared_input_keeps_the_oracle_default_and_offers_the_resident_lookup() -> None:
    prepare = inspect.getsource(Qwen38TTNNPLE.prepare_decode_input)
    assert "resident_lookup: bool = False" in prepare
    assert "lookup = self.resident_lookup.lookup if resident_lookup else self.host_embedding.lookup" in prepare
    assert "lookup" not in inspect.getsource(Qwen38TTNNPLE.forward_prepared)
    assert isinstance(Qwen38TTNNPLE.resident_lookup, property)
    resident = inspect.getsource(Qwen38ResidentPLELookup.lookup)
    for forbidden in ("os.open(", "torch.unique(", "torch.nonzero(", "index_copy_(", "tensor_rows("):
        assert forbidden not in resident
    reader = inspect.getsource(Qwen38CheckpointRowReader)
    assert reader.count("os.open(") == 1
    assert reader.count("_guard(") == 2


def _scalar_ids(token_id: int, context, *, eos: int, spec):
    return ngram_token_ids_decode_step(
        token_id,
        context,
        eos_token_id=eos,
        multipliers=tuple(spec.layer_multipliers.tolist()),
        head_vocab_sizes=tuple(spec.head_vocab_sizes.tolist()),
        head_offsets=tuple(spec.head_offsets.tolist()),
    )


def test_decode_step_hash_matches_the_reference_bitwise_over_the_real_id_ranges() -> None:
    # 8 EOS patterns over (c0, c1, x), 12,500 random triples each, plus every
    # boundary id in every slot; the reference is evaluated as one batch of
    # single-token steps with explicit contexts.
    generator = torch.Generator().manual_seed(20260902)
    boundary = torch.tensor([0, 1, REAL_EOS - 1, REAL_EOS, REAL_EOS + 1, REAL_VOCAB_SIZE - 2, REAL_VOCAB_SIZE - 1])
    triples = []
    for pattern in range(8):
        sample = torch.randint(0, REAL_VOCAB_SIZE, (12_500, 3), generator=generator)
        for slot in range(3):
            if pattern >> slot & 1:
                sample[:, slot] = REAL_EOS
        triples.append(sample)
    triples.append(torch.cartesian_prod(boundary, boundary, boundary))
    triples = torch.cat(triples)
    assert triples.shape[0] == 100_000 + 7**3
    contexts = triples[:, :2].contiguous()
    tokens = triples[:, 2:3].contiguous()
    expected_ids, expected_context = ngram_token_ids(tokens, contexts, REAL_EOS, REAL_SPEC)
    assert expected_ids.shape == (triples.shape[0], 1, 16)
    padded = REAL_SPEC.padded_vocab_size
    for index, (c0, c1, x) in enumerate(triples.tolist()):
        ids, context = _scalar_ids(x, (c0, c1), eos=REAL_EOS, spec=REAL_SPEC)
        assert ids == expected_ids[index, 0].tolist(), (c0, c1, x)
        assert context == tuple(expected_context[index].tolist()) == (c1, x)
        assert all(0 <= value < padded for value in ids)
    # A fresh history: None and the explicit (eos, eos) context are the same step.
    for x in (0, 17, REAL_EOS, REAL_VOCAB_SIZE - 1):
        fresh_ids, fresh_context = ngram_token_ids(torch.tensor([[x]]), None, REAL_EOS, REAL_SPEC)
        assert _scalar_ids(x, None, eos=REAL_EOS, spec=REAL_SPEC) == (fresh_ids[0, 0].tolist(), (REAL_EOS, x))
        assert _scalar_ids(x, (REAL_EOS, REAL_EOS), eos=REAL_EOS, spec=REAL_SPEC)[0] == fresh_ids[0, 0].tolist()
        assert fresh_context.tolist() == [[REAL_EOS, x]]
    # Head partition: bigram heads never see c0 (only c1 and x), trigram heads do.
    a, _ = _scalar_ids(17, (5, 9), eos=REAL_EOS, spec=REAL_SPEC)
    b, _ = _scalar_ids(17, (6, 9), eos=REAL_EOS, spec=REAL_SPEC)
    assert a[:8] == b[:8] and a[8:] != b[8:]
    # No product may wrap int64: the largest token times the largest multiplier stays below 2**63.
    assert (REAL_VOCAB_SIZE - 1) * max(REAL_SPEC.layer_multipliers.tolist()) < 1 << 63


def test_decode_step_hash_threads_the_context_through_the_33_token_chain() -> None:
    reference_context = None
    context = None
    for token in CHAIN_33:
        expected_ids, reference_context = ngram_token_ids(
            torch.tensor([[token]]), reference_context, REAL_EOS, REAL_SPEC
        )
        ids, context = _scalar_ids(token, context, eos=REAL_EOS, spec=REAL_SPEC)
        assert ids == expected_ids[0, 0].tolist()
        assert list(context) == reference_context[0].tolist()


def test_lookup_token_returns_the_oracle_rows_and_context(tmp_path) -> None:
    weight_map = _synthetic_corpus(tmp_path)
    checkpoint = _SyntheticCheckpoint(tmp_path, weight_map)
    host = _synthetic_host_embedding(checkpoint)
    resident = Qwen38ResidentPLELookup(host)
    assert resident.vocab_size == 1000 and resident.eos_token_id == EOS
    assert len(resident.multipliers) == 3 and len(resident.head_vocab_sizes) == len(resident.head_offsets) == 16
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randint(0, 1000, (2_000,), generator=generator)
    tokens[::37] = EOS
    tokens[1::53] = EOS
    context = oracle_context = None
    for token in tokens.tolist():
        payload, context = resident.lookup_token(token, context)
        expected, oracle_context = Qwen38HostPLEEmbedding.lookup(host, torch.tensor([[token]]), oracle_context)
        assert isinstance(payload, bytearray) and len(payload) == 16 * HEAD_DIM * 2
        assert torch.equal(
            torch.frombuffer(payload, dtype=torch.bfloat16).view(torch.int16),
            expected.reshape(-1).contiguous().view(torch.int16),
        )
        assert list(context) == oracle_context[0].tolist()
    for bad in (-1, 1000, 1.0, True, EOS + 1):
        with pytest.raises((ValueError, TypeError)):
            resident.lookup_token(bad, None)
    with pytest.raises(ValueError, match="exact integers"):
        resident.lookup_token(3, (1000, 2))
    # The stream form (the chunk drivers' host_rows): one batched read, the per-token payloads concatenated in order
    # and the contexts chained as the steps chain them, from a fresh history and from a mid-stream context.
    stream = tokens.tolist()
    for start_context in (None, (EOS, EOS), (5, EOS), (EOS, 9), (11, 12)):
        payloads = []
        contexts = [start_context]
        context = start_context
        for token in stream:
            payload, context = resident.lookup_token(token, context)
            payloads.append(bytes(payload))
            contexts.append(context)
        batched, batched_contexts = resident.lookup_tokens(stream, start_context)
        assert isinstance(batched, bytearray) and bytes(batched) == b"".join(payloads)
        assert batched_contexts == tuple(contexts) and len(batched_contexts) == len(stream) + 1
    with pytest.raises(ValueError, match="exact integers"):
        resident.lookup_tokens([3, 1000], None)
    with pytest.raises(ValueError, match="exact integers"):
        resident.lookup_tokens([3], (1000, 2))
    resident.close()


def test_prepared_row_is_row_major_and_the_body_tilizes_it_once() -> None:
    upload = inspect.getsource(Qwen38TTNNPLE._upload_embedding)
    assert "layout=ttnn.ROW_MAJOR_LAYOUT" in upload and "TILE_LAYOUT" not in upload
    assert 'self._validate_prepared_row(tensor, label="PLE upload")' in upload
    validate = inspect.getsource(Qwen38TTNNPLE._validate_prepared_row)
    assert "tensor.layout != ttnn.ROW_MAJOR_LAYOUT" in validate
    assert "placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3" in validate
    forward = inspect.getsource(Qwen38TTNNPLE.forward_prepared)
    calls = [
        node
        for node in ast.walk(ast.parse(inspect.cleandoc(forward)))
        if isinstance(node, ast.Call) and ast.unparse(node.func).startswith("ttnn.")
    ]
    layout_calls = [ast.unparse(call) for call in calls if ast.unparse(call.func) == "ttnn.to_layout"]
    assert layout_calls == [
        "ttnn.to_layout(prepared.embedding_sharded, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)"
    ]
    validate_row = forward.index('self._validate_prepared_row(prepared.embedding_sharded, label="prepared PLE row")')
    tilize = forward.index("embedding_tile = ttnn.to_layout(", validate_row)
    retag = forward.index("embedding_tile.update_tensor_topology(prepared.embedding_sharded.tensor_topology())", tilize)
    project = forward.index("key, value = self._project(embedding_tile)", retag)
    assert validate_row < tilize < retag < project
    assert "retain_input" not in forward
    project_source = inspect.getsource(Qwen38TTNNPLE._project)
    assert "def _project(self, embedding_tile):" in project_source
    assert "_deallocate(embedding_tile)" in project_source and "retain_input" not in project_source
    # The persistent row is never consumed by the body; only its tile is.
    assert "_deallocate(prepared.embedding_sharded)" not in forward
    # 4 x 1,280 B row-major shards on the wire instead of 4 x 40,960 B host tiles.
    assert 640 * 2 == 1_280 and 4 * 1_280 == 5_120


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="pinned checkpoint is not present on this host")
def test_resident_lookup_matches_the_real_checkpoint_oracle_over_a_decode_chain() -> None:
    embedding = Qwen38HostPLEEmbedding(Qwen38Checkpoint(CHECKPOINT))
    resident = Qwen38ResidentPLELookup(embedding)
    tokens = torch.tensor([[17, 15, 16, 29, 31, 248044, 43, 47, 101, 202, 303, 501, 502, 503, 1000, 247_999]])
    expected, expected_context = embedding.lookup(tokens)
    got, got_context = resident.lookup(tokens)
    _assert_bitwise_equal(got, expected)
    assert torch.equal(got_context, expected_context)
    context = oracle_context = None
    for position in range(tokens.shape[1]):
        step = tokens[:, position : position + 1]
        got, context = resident.lookup(step, context)
        expected, oracle_context = embedding.lookup(step, oracle_context)
        _assert_bitwise_equal(got, expected)
        assert torch.equal(context, oracle_context)
    resident.close()
