# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import models.demos.blackhole.qwen38_flash_next.ttnn.bf4 as bf4_module
import models.demos.blackhole.qwen38_flash_next.ttnn.contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.checkpoint import (
    CHECKPOINT_FILE_MANIFEST_SHA256,
    CHECKPOINT_TENSOR_MANIFEST_SHA256,
    PINNED_CHECKPOINT_REVISION,
)
from models.demos.blackhole.qwen38_flash_next.config import CONFIG_SHA256
from models.demos.blackhole.qwen38_flash_next.ttnn.bf4 import (
    BF4Artifact,
    BF4CacheIdentity,
    BF4LayerRecord,
    Qwen38BF4Cache,
    Qwen38BF4Streamer,
    packed_bf4_bytes_per_device,
    packed_bf4_model_bytes_per_device,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import BACKBONE_LAYERS, Qwen38TTNNBuilder
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    Qwen38MeshContract,
    TensorPlacement,
    replicate_tensor_2d_mesh_mapper,
)

RING7_WORKERS = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0))
EXPERT_RANGES = ((0, 128), (128, 256), (256, 384), (384, 512))
_TEST_TENSORBIN_HEADER_BYTES = 64


def _identity() -> BF4CacheIdentity:
    return BF4CacheIdentity(
        checkpoint_revision=PINNED_CHECKPOINT_REVISION,
        checkpoint_config_sha256=CONFIG_SHA256,
        checkpoint_file_manifest_sha256=CHECKPOINT_FILE_MANIFEST_SHA256,
        checkpoint_hash_manifest_sha256=CHECKPOINT_TENSOR_MANIFEST_SHA256,
        tt_metal_revision="181ac080751bccbaea2e5106fdf880f4b1ca04c4",
        mesh_shape=(1, 4),
        physical_ids=(0, 1, 2, 3),
        ring_size=7,
        dram_bank_worker_order=RING7_WORKERS,
    )


def _fast_sparse_digest_fd(descriptor: int) -> str:
    """Test-only digest that samples the materialized edges of sparse fixtures."""

    metadata = os.fstat(descriptor)
    window = min(4096, metadata.st_size)
    digest = hashlib.sha256(str(metadata.st_size).encode())
    digest.update(os.pread(descriptor, window, 0))
    if metadata.st_size > window:
        digest.update(os.pread(descriptor, window, metadata.st_size - window))
    return digest.hexdigest()


def _fast_sparse_digest(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        return _fast_sparse_digest_fd(descriptor)
    finally:
        os.close(descriptor)


def _write_sparse_tensorbin(path: Path, *, payload_bytes: int, marker: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_size = bf4_module.TENSORBIN_HEADER_PREFIX_BYTES + _TEST_TENSORBIN_HEADER_BYTES + payload_bytes
    with path.open("wb") as stream:
        stream.write(_TEST_TENSORBIN_HEADER_BYTES.to_bytes(8, byteorder="little", signed=False))
        stream.write(bytes(_TEST_TENSORBIN_HEADER_BYTES))
        stream.write(marker)
        stream.seek(file_size - 1)
        stream.write(b"\0")


def _copy_sparse_tensorbin(source: Path, destination: Path) -> None:
    size = source.stat().st_size
    window = min(4096, size)
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        prefix = input_stream.read(window)
        input_stream.seek(size - window)
        suffix = input_stream.read(window)
        output_stream.write(prefix)
        output_stream.seek(size - window)
        output_stream.write(suffix)
        output_stream.truncate(size)


def _write_layer(
    cache: Qwen38BF4Cache,
    layer_index: int,
    *,
    namespace: str = "backbone",
) -> tuple[BF4LayerRecord, dict[str, Path]]:
    paths = {name: bf4_module._tensorbin_path(cache._base(namespace, layer_index, name)) for name in ("w0_w1", "w2")}
    logical_shapes = bf4_module._canonical_packed_shapes(ring_size=cache.identity.ring_size)
    artifacts = {}
    for name, path in paths.items():
        payload = f"packed:{namespace}:{layer_index}:{name}".encode()
        _write_sparse_tensorbin(
            path,
            payload_bytes=bf4_module._packed_payload_bytes(logical_shapes[name]),
            marker=payload,
        )
        artifacts[name] = BF4Artifact(
            name=name,
            relative_path=str(path.relative_to(cache.root)),
            sha256=_fast_sparse_digest(path),
            bytes=path.stat().st_size,
            logical_shape=logical_shapes[name],
        )
    record = BF4LayerRecord(
        namespace=namespace,
        layer_index=layer_index,
        expert_ranges=EXPERT_RANGES,
        ring_size=cache.identity.ring_size,
        **artifacts,
    )
    cache._record(record)
    return record, paths


class _TrackingMeshContract:
    def __init__(self, physical_ids: tuple[int, int, int, int], mesh: object) -> None:
        self.physical_ids = physical_ids
        self.mesh = mesh
        self.mesh_validations = 0
        self.tensor_validations = 0

    def validate_mesh(self, mesh: object) -> None:
        if mesh is not self.mesh:
            raise RuntimeError("unexpected mesh object")
        self.mesh_validations += 1

    def validate_tensor(self, _tensor, *, placement, shard_dim) -> None:
        if placement is not TensorPlacement.EXPERT_SHARDED or shard_dim != 2:
            raise RuntimeError("unexpected BF4 tensor placement")
        self.tensor_validations += 1


class _FakeMeshDevice:
    shape = (1, 4)

    @staticmethod
    def get_device_ids():
        return (0, 1, 2, 3)

    @staticmethod
    def get_device_id(coordinate):
        row, column = tuple(coordinate)
        assert row == 0
        return column


class _FakeLocalTensor:
    def __init__(self, mesh, physical_id, address):
        self._device = mesh
        self._coordinate = (0, physical_id)
        self._address = address

    def device(self):
        return self._device

    def device_coords(self):
        return (self._coordinate,)

    def buffer_address(self):
        return self._address


class _FakeTopology:
    @staticmethod
    def distribution_shape():
        return (1, 4)

    @staticmethod
    def mesh_coords():
        return ((0, 0), (0, 1), (0, 2), (0, 3))


class _FakeBackingTensor:
    _next_backing = 1
    _mesh = _FakeMeshDevice()

    def __init__(self, *, backing=None):
        self.backing = type(self)._next_backing if backing is None else backing
        if backing is None:
            type(self)._next_backing += 1
        self.locals = tuple(
            _FakeLocalTensor(self._mesh, physical_id, self.backing * 0x10000 + physical_id * 0x1000)
            for physical_id in range(4)
        )

    def device(self):
        return self._mesh

    @staticmethod
    def tensor_topology():
        return _FakeTopology()


class _FakeBF4Tensor(_FakeBackingTensor):
    def __init__(self, memory_config, shape) -> None:
        super().__init__()
        self._memory_config = memory_config
        self.shape = shape
        self.dtype = bf4_module.ttnn.bfloat4_b

    def memory_config(self):
        return self._memory_config


class TTNNBF4StaticTest(unittest.TestCase):
    def setUp(self):
        # Sparse fixtures preserve the exact multi-GiB file-size/header contract
        # without reading holes.  Production continues to use full SHA-256.
        patcher = mock.patch.object(bf4_module, "_sha256_fd", side_effect=_fast_sparse_digest_fd)
        patcher.start()
        self.addCleanup(patcher.stop)
        backing_api = mock.patch.object(
            contracts_module.ttnn,
            "get_device_tensors",
            side_effect=lambda tensor: tensor.locals,
        )
        backing_api.start()
        self.addCleanup(backing_api.stop)

    def test_replication_mapper_is_explicit_two_axis_topology(self):
        mesh = object()
        mapper = object()
        with (
            mock.patch.object(
                contracts_module.ttnn,
                "PlacementReplicate",
                side_effect=["row-replicate", "tp-replicate"],
            ),
            mock.patch.object(contracts_module.ttnn, "MeshShape", return_value="mesh-1x4") as mesh_shape,
            mock.patch.object(contracts_module.ttnn, "MeshMapperConfig", return_value="explicit-config") as config,
            mock.patch.object(contracts_module.ttnn, "create_mesh_mapper", return_value=mapper) as create,
        ):
            self.assertIs(replicate_tensor_2d_mesh_mapper(mesh), mapper)
        mesh_shape.assert_called_once_with(1, 4)
        config.assert_called_once_with(["row-replicate", "tp-replicate"], "mesh-1x4")
        create.assert_called_once_with(mesh, "explicit-config")

    def test_qwen38_production_sources_do_not_use_legacy_replication_mapper(self):
        model_root = Path(__file__).resolve().parents[1]
        production_sources = (*(model_root / "ttnn").glob("*.py"), *(model_root / "tools").glob("*.py"))
        offenders = [path for path in production_sources if "ReplicateTensorToMesh" in path.read_text()]
        self.assertEqual(offenders, [])

    def test_exact_packer_storage_not_raw_bf4_storage(self):
        self.assertEqual(packed_bf4_bytes_per_device(ring_size=7), (346_816_512, 130_056_192))
        self.assertEqual(packed_bf4_bytes_per_device(ring_size=8), (396_361_728, 148_635_648))
        self.assertEqual(packed_bf4_model_bytes_per_device(ring_size=7), 23_366_762_496)
        self.assertEqual(packed_bf4_model_bytes_per_device(ring_size=8), 26_704_871_424)
        self.assertEqual(
            bf4_module._canonical_packed_shapes(ring_size=7),
            {
                "w0_w1": (7, 1, 512, 2, 2688, 128),
                "w2": (7, 1, 512, 3, 672, 128),
            },
        )
        for ring_size in (7, 8):
            shapes = bf4_module._canonical_packed_shapes(ring_size=ring_size)
            expected = packed_bf4_bytes_per_device(ring_size=ring_size)
            self.assertEqual(
                tuple(bf4_module._packed_payload_bytes(shapes[name]) // 4 for name in ("w0_w1", "w2")),
                expected,
            )

    def test_tensorbin_header_and_exact_payload_contract_rejects_size_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def validate(
                name: str,
                *,
                header_size: int,
                payload_bytes: int,
                expected_payload: int,
                prefix_bytes: int = 8,
                materialized_header_bytes: int | None = None,
            ):
                path = root / f"{name}.tensorbin"
                with path.open("wb") as stream:
                    stream.write(header_size.to_bytes(8, byteorder="little", signed=False)[:prefix_bytes])
                    remaining_header = (
                        min(header_size, 128) if materialized_header_bytes is None else materialized_header_bytes
                    )
                    stream.write(bytes(remaining_header))
                    stream.write(bytes(payload_bytes))
                descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
                try:
                    return bf4_module._validate_tensorbin_payload_fd(
                        descriptor,
                        signature=bf4_module._artifact_fd_signature(descriptor),
                        expected_payload_bytes=expected_payload,
                    )
                finally:
                    os.close(descriptor)

            self.assertEqual(validate("valid", header_size=16, payload_bytes=32, expected_payload=32), 16)
            with self.assertRaisesRegex(RuntimeError, "truncated before"):
                validate("tiny", header_size=0, payload_bytes=0, expected_payload=32, prefix_bytes=4)
            with self.assertRaisesRegex(RuntimeError, "header size/alignment"):
                validate(
                    "truncated",
                    header_size=64,
                    payload_bytes=0,
                    expected_payload=32,
                    materialized_header_bytes=32,
                )
            with self.assertRaisesRegex(RuntimeError, "header size/alignment"):
                validate("misaligned", header_size=10, payload_bytes=32, expected_payload=32)
            with self.assertRaisesRegex(RuntimeError, "header size/alignment"):
                validate(
                    "oversized-header",
                    header_size=bf4_module.MAX_TENSORBIN_HEADER_BYTES + 8,
                    payload_bytes=0,
                    expected_payload=32,
                )
            with self.assertRaisesRegex(RuntimeError, "payload byte count"):
                validate("short-payload", header_size=16, payload_bytes=31, expected_payload=32)
            with self.assertRaisesRegex(RuntimeError, "payload byte count"):
                validate("oversized-payload", header_size=16, payload_bytes=33, expected_payload=32)

    def test_self_hashed_tiny_tensorbin_cannot_claim_a_canonical_global_shape(self):
        identity = _identity()
        contract = Qwen38MeshContract(identity.physical_ids)
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            _, paths = _write_layer(cache, 0)
            tiny = paths["w0_w1"]
            with tiny.open("wb") as stream:
                stream.write((16).to_bytes(8, byteorder="little", signed=False))
                stream.write(bytes(16))
                stream.write(b"tiny-payload")
            document = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
            artifact = document["layers"]["backbone:0"]["w0_w1"]
            artifact["bytes"] = tiny.stat().st_size
            artifact["sha256"] = _fast_sparse_digest(tiny)
            bf4_module._atomic_json(cache.manifest_path, document)
            with self.assertRaisesRegex(RuntimeError, "payload byte count"):
                cache.verify_layer("backbone", 0)

    def test_cache_identity_and_shape_normalization_reject_bool_float_and_string_aliases(self):
        identity = _identity()
        mutations = (
            {"ring_size": 7.0},
            {"format_version": True},
            {"mesh_shape": ("1", 4)},
            {"dram_bank_worker_order": ((0.0, 0), *RING7_WORKERS[1:])},
        )
        for changes in mutations:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, "exact integer|exact integer tuples"):
                    replace(identity, **changes)

        canonical = bf4_module._canonical_packed_shapes(ring_size=7)["w0_w1"]
        aliases = (
            (canonical[0], True, *canonical[2:]),
            (*canonical[:2], 512.0, *canonical[3:]),
            (*canonical[:-1], "128"),
        )
        for shape in aliases:
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(RuntimeError, "shape contains"):
                    bf4_module._native_integer_shape(shape, label="BF4 conversion")

    def test_cache_identity_requires_complete_live_ring_signature(self):
        with self.assertRaisesRegex(ValueError, "one distinct logical worker"):
            replace(_identity(), dram_bank_worker_order=((0, 0),) * 7)

    def test_cache_identity_distinguishes_complete_file_and_tensor_manifests(self):
        identity = _identity()
        with self.assertRaisesRegex(ValueError, "checkpoint_file_manifest_sha256"):
            replace(identity, checkpoint_file_manifest_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "checkpoint_hash_manifest_sha256"):
            replace(identity, checkpoint_hash_manifest_sha256="0" * 64)

    def test_manifest_round_trip_preserves_tuple_contract_and_hashes(self):
        identity = _identity()
        contract = Qwen38MeshContract(identity.physical_ids)
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            record, paths = _write_layer(cache, 0)
            real_sha256_fd = bf4_module._sha256_fd
            with mock.patch.object(bf4_module, "_sha256_fd", wraps=real_sha256_fd) as sha256:
                self.assertEqual(cache.verify_layer("backbone", 0), record)
                self.assertEqual(sha256.call_count, 2)
                self.assertEqual(cache.verify_layer("backbone", 0), record)
                self.assertEqual(sha256.call_count, 2, "unchanged artifacts must not be re-hashed per decode")

                fresh_cache = Qwen38BF4Cache(directory, identity, contract)
                self.assertEqual(fresh_cache.verify_layer("backbone", 0), record)
                self.assertEqual(sha256.call_count, 4, "a fresh cache object must establish its own verification")

                original_stat = paths["w0_w1"].stat()
                payload_offset = bf4_module.TENSORBIN_HEADER_PREFIX_BYTES + _TEST_TENSORBIN_HEADER_BYTES
                with paths["w0_w1"].open("r+b") as stream:
                    stream.seek(payload_offset)
                    original_byte = stream.read(1)
                    stream.seek(payload_offset)
                    stream.write(bytes((original_byte[0] ^ 1,)))
                # Restoring mtime does not defeat the inode/ctime session guard.
                os.utime(paths["w0_w1"], ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
                with self.assertRaisesRegex(RuntimeError, "hash validation"):
                    cache.verify_layer("backbone", 0)
                self.assertEqual(sha256.call_count, 5)

                untrusted_fresh_cache = Qwen38BF4Cache(directory, identity, contract)
                with self.assertRaisesRegex(RuntimeError, "hash validation"):
                    untrusted_fresh_cache.verify_layer("backbone", 0)
                self.assertEqual(sha256.call_count, 6, "fresh cache state must not inherit another object's trust")

    def test_complete_backbone_hashes_once_then_streams_by_verified_identity(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            for layer_index in range(BACKBONE_LAYERS):
                _write_layer(cache, layer_index)

            builder = object.__new__(Qwen38TTNNBuilder)
            builder.bf4_cache = cache
            builder.placement = SimpleNamespace(expert_ranges=EXPERT_RANGES)
            builder.identity = SimpleNamespace(ring_size=identity.ring_size)
            streamer = Qwen38BF4Streamer(cache, mesh)
            builder.expert_streamer = streamer
            self.assertIs(builder.bf4_cache, builder.expert_streamer.cache)

            memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
            logical_shapes = bf4_module._canonical_packed_shapes(ring_size=identity.ring_size)
            retained_paths = []

            def load_tensor(path, *, device):
                self.assertIs(device, mesh)
                self.assertRegex(str(path), r"\A/proc/self/fd/[1-9][0-9]*\Z")
                retained_paths.append(Path(path))
                name = Path(os.readlink(path)).name
                artifact_name = "w0_w1" if name.startswith("w0_w1_") else "w2"
                memory_config = getattr(memory_configs, artifact_name)
                return _FakeBF4Tensor(memory_config, logical_shapes[artifact_name])

            real_sha256_fd = bf4_module._sha256_fd
            with (
                mock.patch.object(cache, "_read_manifest", wraps=cache._read_manifest) as read_manifest,
                mock.patch.object(bf4_module, "_sha256_fd", wraps=real_sha256_fd) as sha256,
                mock.patch.object(
                    bf4_module,
                    "qualify_live_bf4_ring",
                    return_value=RING7_WORKERS,
                ) as qualify_ring,
                mock.patch.object(bf4_module.ttnn, "load_tensor", side_effect=load_tensor) as load,
                mock.patch.object(
                    bf4_module.ttnn.experimental,
                    "get_weight_mem_configs",
                    return_value=memory_configs,
                ),
                mock.patch.object(bf4_module.ttnn, "deallocate") as deallocate,
            ):
                records = builder.require_complete_backbone_bf4()
                self.assertEqual(len(records), BACKBONE_LAYERS)
                self.assertEqual(tuple(record.layer_index for record in records), tuple(range(BACKBONE_LAYERS)))
                self.assertEqual(sha256.call_count, 2 * BACKBONE_LAYERS)

                for traversal in range(2):
                    for layer_index in range(BACKBONE_LAYERS):
                        with streamer.layer(layer_index) as tensors:
                            self.assertEqual(streamer._active, ("backbone", layer_index))
                            self.assertEqual(len(tensors), 2)
                            if traversal == layer_index == 0:
                                with self.assertRaisesRegex(RuntimeError, "already owns active layer"):
                                    with streamer.layer(1):
                                        pass
                    self.assertIsNone(streamer._active)

                self.assertEqual(
                    sha256.call_count,
                    2 * BACKBONE_LAYERS,
                    "decode traversal must bind fresh FDs to the session hash without rehashing the model",
                )
                self.assertEqual(read_manifest.call_count, 3 * BACKBONE_LAYERS)
                self.assertEqual(qualify_ring.call_count, 2 * BACKBONE_LAYERS)
                self.assertEqual(load.call_count, 4 * BACKBONE_LAYERS)
                self.assertEqual(deallocate.call_count, 4 * BACKBONE_LAYERS)
                self.assertEqual(contract.mesh_validations, 1 + 2 * BACKBONE_LAYERS)
                self.assertEqual(contract.tensor_validations, 4 * BACKBONE_LAYERS)
                self.assertEqual(len(retained_paths), 4 * BACKBONE_LAYERS)
                self.assertTrue(all(not path.exists() for path in retained_paths))

    def test_load_tensor_wrapper_owns_canonical_proc_fd_through_native_read(self):
        core_module = bf4_module.ttnn.operations.core
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "weight.tensorbin"
            artifact.write_bytes(b"valid-flatbuffer-placeholder")
            descriptor = os.open(artifact, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            caller_path = Path(f"/proc/self/fd/{descriptor}")
            native_paths = []

            def native_load(path, device):
                self.assertIsNone(device)
                self.assertRegex(path, r"\A/proc/self/fd/[1-9][0-9]*\Z")
                self.assertNotEqual(path, str(caller_path))
                self.assertEqual(os.readlink(path), str(artifact))
                os.fstat(int(Path(path).name))
                native_paths.append(Path(path))
                return sentinel

            try:
                with (
                    mock.patch.object(
                        core_module.ttnn._ttnn.tensor,
                        "load_tensor_flatbuffer",
                        side_effect=native_load,
                    ) as native,
                    mock.patch.object(core_module.ttnn, "deallocate") as deallocate,
                ):
                    self.assertIs(core_module.load_tensor(caller_path), sentinel)
                    self.assertTrue(caller_path.exists(), "the wrapper must not close its caller's descriptor")
                    self.assertEqual(len(native_paths), 1)
                    self.assertFalse(native_paths[0].exists(), "the wrapper-owned duplicate must close on success")

                    self.assertIs(core_module.load_tensor(artifact), sentinel)
                    self.assertRegex(native.call_args_list[-1].args[0], r"\A/proc/self/fd/[1-9][0-9]*\Z")
                    self.assertEqual(len(native_paths), 2)
                    self.assertFalse(native_paths[1].exists(), "the ordinary-path descriptor must close on success")
                    deallocate.assert_not_called()
            finally:
                os.close(descriptor)
            self.assertFalse(caller_path.exists())

    def test_load_tensor_wrapper_rejects_descriptor_aliases_and_invalid_targets(self):
        core_module = bf4_module.ttnn.operations.core
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "weight.tensorbin"
            artifact.write_bytes(b"payload")
            descriptor = os.open(artifact, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            proc_path = Path(f"/proc/self/fd/{descriptor}")
            alias = root / "alias.tensorbin"
            alias.symlink_to(proc_path)
            inner_alias = root / "inner.tensorbin"
            inner_alias.symlink_to(proc_path)
            outer_alias = root / "outer.tensorbin"
            outer_alias.symlink_to(inner_alias)
            pid_alias = root / "pid-alias.tensorbin"
            pid_alias.symlink_to(f"/proc/{os.getpid()}/fd/{descriptor}")
            plain = root / "weight.bin"
            plain.write_bytes(b"payload")
            plain_descriptor = os.open(plain, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            directory_descriptor = os.open(root, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            closed_descriptor = os.dup(descriptor)
            os.close(closed_descriptor)
            try:
                with mock.patch.object(core_module.ttnn._ttnn.tensor, "load_tensor_flatbuffer") as native:
                    invalid_spellings = (
                        f"/proc/self/fd/{descriptor}.tensorbin",
                        f"/proc/self/fd/0{descriptor}",
                        f"/proc/self/fd/{descriptor}/../{descriptor}",
                        f"/proc/self/fd/{descriptor}/",
                        f"/proc/self/fd/./{descriptor}",
                        f"/proc/self/fd//{descriptor}",
                        "/proc/self/fd/0",
                    )
                    for invalid in invalid_spellings:
                        with self.subTest(invalid=invalid):
                            with self.assertRaisesRegex(RuntimeError, "canonical /proc/self/fd"):
                                core_module.load_tensor(invalid)
                    with self.assertRaisesRegex(RuntimeError, "must not alias"):
                        core_module.load_tensor(alias)
                    with self.assertRaisesRegex(RuntimeError, "must not alias"):
                        core_module.load_tensor(outer_alias)
                    with self.assertRaisesRegex(RuntimeError, "must not alias"):
                        core_module.load_tensor(pid_alias)
                    with self.assertRaisesRegex(RuntimeError, r"absolute \.tensorbin"):
                        core_module.load_tensor(f"/proc/self/fd/{plain_descriptor}")
                    with self.assertRaisesRegex(RuntimeError, "regular file"):
                        core_module.load_tensor(f"/proc/self/fd/{directory_descriptor}")
                    with self.assertRaisesRegex(RuntimeError, "does not name an open descriptor"):
                        core_module.load_tensor(f"/proc/self/fd/{closed_descriptor}")
                    native.assert_not_called()
            finally:
                os.close(directory_descriptor)
                os.close(plain_descriptor)
                os.close(descriptor)

    def test_load_tensor_wrapper_retains_ordinary_symlink_compatibility_and_bounds_loops(self):
        core_module = bf4_module.ttnn.operations.core
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "weight.tensorbin"
            artifact.write_bytes(b"payload")
            ordinary_alias = root / "ordinary.tensorbin"
            ordinary_alias.symlink_to(artifact)
            loop_a = root / "loop-a.tensorbin"
            loop_b = root / "loop-b.tensorbin"
            loop_a.symlink_to(loop_b)
            loop_b.symlink_to(loop_a)

            def native_load(path, device):
                self.assertIsNone(device)
                self.assertRegex(path, r"\A/proc/self/fd/[1-9][0-9]*\Z")
                self.assertEqual(os.readlink(path), str(artifact))
                return sentinel

            with mock.patch.object(
                core_module.ttnn._ttnn.tensor,
                "load_tensor_flatbuffer",
                side_effect=native_load,
            ) as native:
                self.assertIs(core_module.load_tensor(ordinary_alias), sentinel)
                with self.assertRaisesRegex(RuntimeError, "hop bound"):
                    core_module.load_tensor(loop_a)
                native.assert_called_once()

    def test_load_tensor_wrapper_binds_ordinary_path_to_owned_fd_across_replacement(self):
        core_module = bf4_module.ttnn.operations.core
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "weight.tensorbin"
            artifact.write_bytes(b"original")
            replacement = root / "replacement.tensorbin"
            replacement.write_bytes(b"replacement")
            stable_paths = []

            def replace_during_native(path, device):
                self.assertIsNone(device)
                stable_paths.append(Path(path))
                self.assertEqual(os.readlink(path), str(artifact))
                os.replace(replacement, artifact)
                return sentinel

            with (
                mock.patch.object(
                    core_module.ttnn._ttnn.tensor,
                    "load_tensor_flatbuffer",
                    side_effect=replace_during_native,
                ),
                mock.patch.object(core_module.ttnn, "deallocate") as deallocate,
            ):
                with self.assertRaisesRegex(RuntimeError, "changed during tensor load"):
                    core_module.load_tensor(artifact)
                deallocate.assert_called_once_with(sentinel)
            self.assertEqual(len(stable_paths), 1)
            self.assertFalse(stable_paths[0].exists())

    def test_load_tensor_wrapper_detects_native_read_mutation_and_cleans_up(self):
        core_module = bf4_module.ttnn.operations.core
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "weight.tensorbin"
            artifact.write_bytes(b"payload")
            descriptor = os.open(artifact, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            proc_path = Path(f"/proc/self/fd/{descriptor}")
            native_paths = []

            def mutate_during_native(path, _device):
                native_paths.append(Path(path))
                artifact.write_bytes(b"payload-mutated")
                return sentinel

            try:
                with (
                    mock.patch.object(
                        core_module.ttnn._ttnn.tensor,
                        "load_tensor_flatbuffer",
                        side_effect=mutate_during_native,
                    ),
                    mock.patch.object(core_module.ttnn, "deallocate") as deallocate,
                ):
                    with self.assertRaisesRegex(RuntimeError, "changed during tensor load"):
                        core_module.load_tensor(proc_path)
                    deallocate.assert_called_once_with(sentinel)
                self.assertEqual(len(native_paths), 1)
                self.assertFalse(native_paths[0].exists(), "the duplicate must close on validation failure")
                self.assertTrue(proc_path.exists(), "the caller retains ownership after wrapper failure")
            finally:
                os.close(descriptor)

    def test_load_tensor_wrapper_reports_returned_tensor_cleanup_failure(self):
        core_module = bf4_module.ttnn.operations.core
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "weight.tensorbin"
            artifact.write_bytes(b"payload")
            descriptor = os.open(artifact, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            proc_path = Path(f"/proc/self/fd/{descriptor}")
            native_paths = []

            def mutate_during_native(path, _device):
                native_paths.append(Path(path))
                artifact.write_bytes(b"payload-mutated")
                return sentinel

            try:
                with (
                    mock.patch.object(
                        core_module.ttnn._ttnn.tensor,
                        "load_tensor_flatbuffer",
                        side_effect=mutate_during_native,
                    ),
                    mock.patch.object(
                        core_module.ttnn,
                        "deallocate",
                        side_effect=RuntimeError("synthetic core cleanup failure"),
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "returned-tensor cleanup also failed") as raised:
                        core_module.load_tensor(proc_path)
                self.assertRegex(str(raised.exception.primary_error), "changed during tensor load")
                self.assertRegex(str(raised.exception.cleanup_error), "synthetic core cleanup failure")
                self.assertFalse(native_paths[0].exists(), "the wrapper duplicate must close despite cleanup failure")
                self.assertTrue(proc_path.exists(), "the caller descriptor must remain caller-owned")
            finally:
                os.close(descriptor)

    def test_bf4_load_retains_verified_fds_through_topology_validation(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
        logical_shapes = bf4_module._canonical_packed_shapes(ring_size=identity.ring_size)
        retained_paths = []
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            _write_layer(cache, 0)

            def load_tensor(path, *, device):
                self.assertIs(device, mesh)
                self.assertRegex(str(path), r"\A/proc/self/fd/[1-9][0-9]*\Z")
                retained_paths.append(Path(path))
                target = Path(os.readlink(path)).name
                artifact_name = "w0_w1" if target.startswith("w0_w1_") else "w2"
                return _FakeBF4Tensor(getattr(memory_configs, artifact_name), logical_shapes[artifact_name])

            original_validate = contract.validate_tensor

            def validate_while_retained(tensor, *, placement, shard_dim):
                self.assertEqual(len(retained_paths), 2)
                self.assertTrue(all(path.exists() for path in retained_paths))
                return original_validate(tensor, placement=placement, shard_dim=shard_dim)

            with (
                mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_WORKERS),
                mock.patch.object(bf4_module.ttnn, "load_tensor", side_effect=load_tensor),
                mock.patch.object(
                    bf4_module.ttnn.experimental,
                    "get_weight_mem_configs",
                    return_value=memory_configs,
                ),
                mock.patch.object(contract, "validate_tensor", side_effect=validate_while_retained),
                mock.patch.object(bf4_module.ttnn, "deallocate") as deallocate,
            ):
                tensors = cache.load_layer(mesh, layer_index=0)
            self.assertEqual(len(tensors), 2)
            self.assertTrue(all(not path.exists() for path in retained_paths))
            deallocate.assert_not_called()

    def test_bf4_load_rejects_wrong_loaded_shape_and_releases_both_tensors(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
        logical_shapes = bf4_module._canonical_packed_shapes(ring_size=identity.ring_size)
        loaded = []
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            _write_layer(cache, 0)

            def load_tensor(path, *, device):
                self.assertIs(device, mesh)
                target = Path(os.readlink(path)).name
                artifact_name = "w0_w1" if target.startswith("w0_w1_") else "w2"
                shape = logical_shapes[artifact_name]
                if artifact_name == "w2":
                    shape = (*shape[:-1], shape[-1] + 32)
                tensor = _FakeBF4Tensor(getattr(memory_configs, artifact_name), shape)
                loaded.append(tensor)
                return tensor

            with (
                mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_WORKERS),
                mock.patch.object(bf4_module.ttnn, "load_tensor", side_effect=load_tensor),
                mock.patch.object(
                    bf4_module.ttnn.experimental,
                    "get_weight_mem_configs",
                    return_value=memory_configs,
                ),
                mock.patch.object(bf4_module.ttnn, "deallocate") as deallocate,
            ):
                with self.assertRaisesRegex(RuntimeError, "loaded shape"):
                    cache.load_layer(mesh, layer_index=0)
            self.assertEqual([call.args[0] for call in deallocate.call_args_list], loaded)

    def test_bf4_load_rejects_bool_float_and_string_shape_aliases_and_releases_both_tensors(self):
        identity = _identity()
        mesh = object()
        memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
        logical_shapes = bf4_module._canonical_packed_shapes(ring_size=identity.ring_size)
        aliases = (
            (7, True, 512, 2, 2688, 128),
            (7, 1, 512.0, 2, 2688, 128),
            (7, 1, 512, 2, 2688, "128"),
        )
        for invalid_shape in aliases:
            with self.subTest(invalid_shape=invalid_shape), tempfile.TemporaryDirectory() as directory:
                contract = _TrackingMeshContract(identity.physical_ids, mesh)
                cache = Qwen38BF4Cache(directory, identity, contract)
                _write_layer(cache, 0)
                loaded = []

                def load_tensor(path, *, device):
                    self.assertIs(device, mesh)
                    target = Path(os.readlink(path)).name
                    artifact_name = "w0_w1" if target.startswith("w0_w1_") else "w2"
                    shape = invalid_shape if artifact_name == "w0_w1" else logical_shapes[artifact_name]
                    tensor = _FakeBF4Tensor(getattr(memory_configs, artifact_name), shape)
                    loaded.append(tensor)
                    return tensor

                with (
                    mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_WORKERS),
                    mock.patch.object(bf4_module.ttnn, "load_tensor", side_effect=load_tensor),
                    mock.patch.object(
                        bf4_module.ttnn.experimental,
                        "get_weight_mem_configs",
                        return_value=memory_configs,
                    ),
                    mock.patch.object(bf4_module.ttnn, "deallocate") as deallocate,
                ):
                    with self.assertRaisesRegex(RuntimeError, "shape contains"):
                        cache.load_layer(mesh, layer_index=0)
                self.assertEqual([call.args[0] for call in deallocate.call_args_list], loaded)

    def test_bf4_load_rejects_path_replacement_after_native_reads_and_closes_fds(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
        logical_shapes = bf4_module._canonical_packed_shapes(ring_size=identity.ring_size)
        retained_paths = []
        loaded_tensors = []
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            _, paths = _write_layer(cache, 0)

            def replace_after_both_reads(path, *, device):
                self.assertIs(device, mesh)
                retained_paths.append(Path(path))
                target = os.readlink(path)
                artifact_name = "w0_w1" if Path(target).name.startswith("w0_w1_") else "w2"
                tensor = _FakeBF4Tensor(getattr(memory_configs, artifact_name), logical_shapes[artifact_name])
                loaded_tensors.append(tensor)
                if len(loaded_tensors) == 2:
                    replacement = paths["w0_w1"].with_name("replacement.tensorbin")
                    _copy_sparse_tensorbin(paths["w0_w1"], replacement)
                    os.replace(replacement, paths["w0_w1"])
                return tensor

            with (
                mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_WORKERS),
                mock.patch.object(bf4_module.ttnn, "load_tensor", side_effect=replace_after_both_reads),
                mock.patch.object(
                    bf4_module.ttnn.experimental,
                    "get_weight_mem_configs",
                    return_value=memory_configs,
                ),
                mock.patch.object(bf4_module.ttnn, "deallocate") as deallocate,
            ):
                with self.assertRaisesRegex(RuntimeError, "identity changed while its descriptor was retained"):
                    cache.load_layer(mesh, layer_index=0)
            self.assertEqual(len(loaded_tensors), 2)
            self.assertEqual([call.args[0] for call in deallocate.call_args_list], loaded_tensors)
            self.assertTrue(all(not path.exists() for path in retained_paths))

    def test_bf4_load_rechecks_in_place_mutation_after_topology_validation(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
        logical_shapes = bf4_module._canonical_packed_shapes(ring_size=identity.ring_size)
        retained_paths = []
        loaded_tensors = []
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            _, paths = _write_layer(cache, 0)

            def load_tensor(path, *, device):
                self.assertIs(device, mesh)
                retained_paths.append(Path(path))
                target = Path(os.readlink(path)).name
                artifact_name = "w0_w1" if target.startswith("w0_w1_") else "w2"
                tensor = _FakeBF4Tensor(getattr(memory_configs, artifact_name), logical_shapes[artifact_name])
                loaded_tensors.append(tensor)
                return tensor

            original_validate = contract.validate_tensor

            def mutate_during_topology(tensor, *, placement, shard_dim):
                self.assertTrue(all(path.exists() for path in retained_paths))
                if contract.tensor_validations == 0:
                    with paths["w2"].open("ab") as stream:
                        stream.write(b"mutation")
                return original_validate(tensor, placement=placement, shard_dim=shard_dim)

            with (
                mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_WORKERS),
                mock.patch.object(bf4_module.ttnn, "load_tensor", side_effect=load_tensor),
                mock.patch.object(
                    bf4_module.ttnn.experimental,
                    "get_weight_mem_configs",
                    return_value=memory_configs,
                ),
                mock.patch.object(contract, "validate_tensor", side_effect=mutate_during_topology),
                mock.patch.object(bf4_module.ttnn, "deallocate") as deallocate,
            ):
                with self.assertRaisesRegex(RuntimeError, "changed while its descriptor was retained"):
                    cache.load_layer(mesh, layer_index=0)
            self.assertEqual(contract.tensor_validations, 2)
            self.assertEqual([call.args[0] for call in deallocate.call_args_list], loaded_tensors)
            self.assertTrue(all(not path.exists() for path in retained_paths))

    def test_bf4_load_failure_closes_both_fds_and_releases_first_tensor(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
        logical_shapes = bf4_module._canonical_packed_shapes(ring_size=identity.ring_size)
        retained_paths = []
        first_tensor = _FakeBF4Tensor(memory_configs.w0_w1, logical_shapes["w0_w1"])
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            _write_layer(cache, 0)

            def fail_second_load(path, *, device):
                self.assertIs(device, mesh)
                retained_paths.append(Path(path))
                if len(retained_paths) == 1:
                    return first_tensor
                raise RuntimeError("synthetic second load failure")

            with (
                mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_WORKERS),
                mock.patch.object(bf4_module.ttnn, "load_tensor", side_effect=fail_second_load),
                mock.patch.object(
                    bf4_module.ttnn.experimental,
                    "get_weight_mem_configs",
                    return_value=memory_configs,
                ),
                mock.patch.object(bf4_module.ttnn, "deallocate") as deallocate,
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic second load failure"):
                    cache.load_layer(mesh, layer_index=0)
            deallocate.assert_called_once_with(first_tensor)
            self.assertEqual(len(retained_paths), 2)
            self.assertTrue(all(not path.exists() for path in retained_paths))

    def test_bf4_cache_load_aggregates_load_and_cleanup_failures(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
        logical_shapes = bf4_module._canonical_packed_shapes(ring_size=identity.ring_size)
        first_tensor = _FakeBF4Tensor(memory_configs.w0_w1, logical_shapes["w0_w1"])
        retained_paths = []
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            record, paths = _write_layer(cache, 0)
            self.assertEqual(cache.verify_layer("backbone", 0), record)

            def fail_second_load(path, *, device):
                self.assertIs(device, mesh)
                retained_paths.append(Path(path))
                if len(retained_paths) == 1:
                    return first_tensor
                raise RuntimeError("synthetic cache load failure")

            with (
                mock.patch.object(bf4_module.ttnn, "load_tensor", side_effect=fail_second_load),
                mock.patch.object(
                    bf4_module.ttnn,
                    "deallocate",
                    side_effect=RuntimeError("synthetic cache cleanup failure"),
                ) as deallocate,
            ):
                with self.assertRaisesRegex(bf4_module.BF4CleanupError, "cache load failed") as raised:
                    cache._load_verified_tensors(
                        mesh,
                        record=record,
                        w01_path=paths["w0_w1"],
                        w2_path=paths["w2"],
                        memory_configs=memory_configs,
                    )
            self.assertRegex(str(raised.exception.primary_error), "synthetic cache load failure")
            self.assertEqual(len(raised.exception.cleanup_errors), 1)
            self.assertEqual(raised.exception.unreleased_tensors, (first_tensor,))
            self.assertEqual(len(raised.exception.tensor_cleanup_outcomes), 2)
            self.assertTrue(raised.exception.tensor_cleanup_outcomes[0].release_attempted)
            self.assertFalse(raised.exception.tensor_cleanup_outcomes[0].released)
            self.assertIs(raised.exception.tensor_cleanup_outcomes[0].tensor, first_tensor)
            self.assertIsNone(raised.exception.tensor_cleanup_outcomes[1].tensor)
            deallocate.assert_called_once_with(first_tensor)
            self.assertTrue(all(not path.exists() for path in retained_paths))

    def test_bf4_verification_rejects_symlink_artifact_without_loading(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            _, paths = _write_layer(cache, 0)
            artifact = paths["w0_w1"]
            target = artifact.with_name("target.tensorbin")
            _copy_sparse_tensorbin(artifact, target)
            artifact.unlink()
            artifact.symlink_to(target)
            with (
                mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_WORKERS),
                mock.patch.object(bf4_module.ttnn, "load_tensor") as load,
            ):
                with self.assertRaisesRegex(RuntimeError, "unavailable or invalid"):
                    cache.load_layer(mesh, layer_index=0)
                load.assert_not_called()

    def test_manifest_mutation_cannot_redirect_verified_or_loaded_layer(self):
        identity = _identity()
        mesh = object()
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            _write_layer(cache, 0)
            cache.verify_layer("backbone", 0)
            baseline = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
            key = "backbone:0"

            mutations = (
                (
                    "record identity",
                    lambda document: document["layers"][key].__setitem__("layer_index", 1),
                ),
                (
                    "expert ownership",
                    lambda document: document["layers"][key].__setitem__(
                        "expert_ranges", [[128, 256], [0, 128], [256, 384], [384, 512]]
                    ),
                ),
                (
                    "requested slot",
                    lambda document: document["layers"][key]["w0_w1"].__setitem__(
                        "relative_path",
                        "backbone/layer-01/w0_w1_dtype_BFLOAT4_B_layout_TILE.tensorbin",
                    ),
                ),
                (
                    "canonical slot shape",
                    lambda document: document["layers"][key]["w0_w1"].__setitem__(
                        "logical_shape", [7, 1, 512, 2, 2688, 160]
                    ),
                ),
                (
                    "scalar schema",
                    lambda document: document["layers"][key]["w2"].__setitem__("bytes", True),
                ),
                (
                    "manifest schema",
                    lambda document: document["layers"][key]["w2"].__setitem__("unbound", "field"),
                ),
                (
                    "identity digest",
                    lambda document: document.__setitem__("identity_key", "0" * 64),
                ),
                (
                    "manifest identity",
                    lambda document: document["identity"].__setitem__("ring_size", 7.0),
                ),
                (
                    "manifest identity",
                    lambda document: document["identity"]["physical_ids"].__setitem__(0, False),
                ),
                (
                    "manifest format",
                    lambda document: document.__setitem__("format_version", True),
                ),
                (
                    "manifest format",
                    lambda document: document.__setitem__("format_version", 1.0),
                ),
            )
            with (
                mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_WORKERS),
                mock.patch.object(bf4_module.ttnn, "load_tensor") as load,
            ):
                for expected, mutate in mutations:
                    with self.subTest(expected=expected):
                        document = copy.deepcopy(baseline)
                        mutate(document)
                        bf4_module._atomic_json(cache.manifest_path, document)
                        with self.assertRaisesRegex(RuntimeError, expected):
                            cache.load_layer(mesh, layer_index=0)
                        load.assert_not_called()

    def test_cache_rejects_layer_aliases_and_non_target_slots_before_manifest_access(self):
        identity = _identity()
        contract = Qwen38MeshContract(identity.physical_ids)
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            with mock.patch.object(cache, "_read_manifest", side_effect=AssertionError("manifest was read")):
                for invalid in (True, 0.0, "0"):
                    with self.subTest(invalid=invalid):
                        with self.assertRaisesRegex(TypeError, "exact integer"):
                            cache.verify_layer("backbone", invalid)
                with self.assertRaisesRegex(ValueError, r"\[0,48\)"):
                    cache.verify_layer("backbone", BACKBONE_LAYERS)
                with self.assertRaisesRegex(ValueError, r"\[0,1\)"):
                    cache.verify_layer("mtp", 1)

    def test_layer_host_packing_fills_four_canonical_ranges_without_cat(self):
        shapes = bf4_module._canonical_packed_shapes(ring_size=7)
        allocations = []
        copies = []
        loaded = []

        class Prepared:
            def __init__(self, name, device_index, shape):
                self.name = name
                self.device_index = device_index
                self.shape = shape
                self.dtype = bf4_module.torch.bfloat16

        class DestinationSlice:
            def __init__(self, name, start, length):
                self.name = name
                self.start = start
                self.length = length

            def copy_(self, source):
                copies.append((self.name, self.start, self.length, source.name, source.device_index))
                return self

        class Destination:
            def __init__(self, shape, dtype):
                self.name = "w0_w1" if not allocations else "w2"
                self.shape = shape
                self.dtype = dtype
                allocations.append((self.name, shape, dtype))

            def narrow(self, dimension, start, length):
                if dimension != 2:
                    raise AssertionError(f"unexpected expert dimension {dimension}")
                return DestinationSlice(self.name, start, length)

        class Weights:
            expert_ranges = EXPERT_RANGES

            @staticmethod
            def routed_device_shard(device_index):
                loaded.append(device_index)
                return SimpleNamespace(
                    expert_range=EXPERT_RANGES[device_index],
                    gate_up=f"gate-up-{device_index}",
                    down=f"down-{device_index}",
                )

        def split(gate_up, split_size, *, dim):
            device_index = int(gate_up.rsplit("-", 1)[1])
            self.assertEqual((split_size, dim), (640, -1))
            return f"gate-{device_index}", f"up-{device_index}"

        def prepare_w01(gate, up, layers, experts, hidden, intermediate, _shard_map):
            device_index = int(gate.rsplit("-", 1)[1])
            self.assertEqual(up, f"up-{device_index}")
            self.assertEqual((layers, experts, hidden, intermediate), (1, 128, 2560, 640))
            local_shape = shapes["w0_w1"][:2] + (128,) + shapes["w0_w1"][3:]
            return Prepared("w0_w1", device_index, local_shape)

        def prepare_w2(down, layers, experts, intermediate, hidden, _w2_map, _w01_map):
            device_index = int(down.rsplit("-", 1)[1])
            self.assertEqual((layers, experts, intermediate, hidden), (1, 128, 640, 2560))
            local_shape = shapes["w2"][:2] + (128,) + shapes["w2"][3:]
            return Prepared("w2", device_index, local_shape)

        with (
            mock.patch.object(bf4_module.torch, "empty", side_effect=Destination) as empty,
            mock.patch.object(bf4_module.torch, "split", side_effect=split),
            mock.patch.object(bf4_module, "prepare_w0_w1_tensor_for_moe_compute", side_effect=prepare_w01),
            mock.patch.object(bf4_module, "prepare_w2_tensor_for_moe_compute", side_effect=prepare_w2),
            mock.patch.object(bf4_module.torch, "cat", side_effect=AssertionError("whole-layer cat is forbidden")),
        ):
            w01, w2 = bf4_module._prepare_routed_layer_host_tensors(Weights(), ring_size=7)

        self.assertEqual(w01.shape, shapes["w0_w1"])
        self.assertEqual(w2.shape, shapes["w2"])
        self.assertEqual(loaded, [0, 1, 2, 3])
        self.assertEqual(
            copies,
            [
                item
                for device_index, start in enumerate((0, 128, 256, 384))
                for item in (
                    ("w0_w1", start, 128, "w0_w1", device_index),
                    ("w2", start, 128, "w2", device_index),
                )
            ],
        )
        self.assertEqual(empty.call_count, 2)
        self.assertEqual([shape for _, shape, _ in allocations], [shapes["w0_w1"], shapes["w2"]])

    def test_fresh_conversion_aggregates_second_upload_and_first_tensor_cleanup_failures(self):
        identity = _identity()
        mesh = SimpleNamespace(shape=(1, 4))
        contract = _TrackingMeshContract(identity.physical_ids, mesh)
        placement = SimpleNamespace(expert_ranges=EXPERT_RANGES)
        memory_configs = SimpleNamespace(w0_w1=object(), w2=object())
        logical_shapes = bf4_module._canonical_packed_shapes(ring_size=identity.ring_size)
        first_tensor = _FakeBF4Tensor(memory_configs.w0_w1, logical_shapes["w0_w1"])

        class HostTensor:
            pass

        weights = SimpleNamespace(
            expert_ranges=EXPERT_RANGES,
        )
        with tempfile.TemporaryDirectory() as directory:
            cache = Qwen38BF4Cache(directory, identity, contract)
            with (
                mock.patch.object(cache, "verify_layer", return_value=None),
                mock.patch.object(bf4_module, "qualify_live_bf4_ring", return_value=RING7_WORKERS),
                mock.patch.object(bf4_module, "Qwen38MoEWeights", return_value=weights),
                mock.patch.object(bf4_module.ttnn, "ShardTensor2dMesh", return_value=object()),
                mock.patch.object(
                    bf4_module.ttnn.experimental,
                    "get_weight_mem_configs",
                    return_value=memory_configs,
                ),
                mock.patch.object(
                    bf4_module,
                    "_prepare_routed_layer_host_tensors",
                    return_value=(HostTensor(), HostTensor()),
                ) as prepare,
                mock.patch.object(
                    bf4_module.ttnn,
                    "as_tensor",
                    side_effect=[first_tensor, RuntimeError("synthetic second upload failure")],
                ),
                mock.patch.object(
                    bf4_module.ttnn,
                    "deallocate",
                    side_effect=RuntimeError("synthetic conversion cleanup failure"),
                ) as deallocate,
            ):
                with self.assertRaisesRegex(
                    bf4_module.BF4CleanupError, "owned tensor cleanup was incomplete"
                ) as raised:
                    cache._convert_and_upload_locked(
                        object(),
                        placement,
                        mesh,
                        layer_index=0,
                    )
            self.assertRegex(str(raised.exception.primary_error), "synthetic second upload failure")
            self.assertEqual(len(raised.exception.cleanup_errors), 1)
            self.assertRegex(str(raised.exception.cleanup_errors[0]), "synthetic conversion cleanup failure")
            deallocate.assert_called_once_with(first_tensor)
            prepare.assert_called_once_with(weights, ring_size=identity.ring_size)

    def test_streamer_attempts_both_releases_and_clears_active_slot(self):
        class Tensor(_FakeBackingTensor):
            def __init__(self, name, identity):
                super().__init__(backing=identity)
                self.name = name

            def __str__(self):
                return self.name

        w01, w2 = Tensor("w01", 1), Tensor("w2", 2)

        class Contract:
            @staticmethod
            def validate_mesh(_mesh):
                return None

        class Cache:
            mesh_contract = Contract()

            @staticmethod
            def load_layer(_mesh, *, layer_index, namespace):
                self.assertEqual((namespace, layer_index), ("backbone", 7))
                return (w01, w2)

        streamer = Qwen38BF4Streamer(Cache(), object())
        released = []

        def release(tensor):
            released.append(tensor)
            if tensor is w01:
                raise RuntimeError("first release failed")

        with mock.patch.object(bf4_module.ttnn, "deallocate", side_effect=release):
            with self.assertRaisesRegex(RuntimeError, "first release failed"):
                with streamer.layer(7):
                    pass
        self.assertEqual(released, [w01, w2])
        self.assertIsNone(streamer._active)

    def test_streamer_aggregates_body_and_all_cleanup_failures(self):
        class Tensor(_FakeBackingTensor):
            def __init__(self, name, identity):
                super().__init__(backing=identity)
                self.name = name

            def __str__(self):
                return self.name

        w01, w2 = Tensor("w01", 1), Tensor("w2", 2)

        class Contract:
            @staticmethod
            def validate_mesh(_mesh):
                return None

        class Cache:
            mesh_contract = Contract()

            @staticmethod
            def load_layer(_mesh, *, layer_index, namespace):
                self.assertEqual((namespace, layer_index), ("backbone", 7))
                return (w01, w2)

        streamer = Qwen38BF4Streamer(Cache(), object())
        released = []

        def release(tensor):
            released.append(tensor)
            raise RuntimeError(f"release failed for {tensor}")

        with mock.patch.object(bf4_module.ttnn, "deallocate", side_effect=release):
            with self.assertRaisesRegex(bf4_module.BF4CleanupError, "streamer body failed") as raised:
                with streamer.layer(7):
                    raise ValueError("synthetic body failure")
        self.assertRegex(str(raised.exception.primary_error), "synthetic body failure")
        self.assertEqual(len(raised.exception.cleanup_errors), 2)
        self.assertEqual(released, [w01, w2])
        self.assertIsNone(streamer._active)


if __name__ == "__main__":
    unittest.main()
