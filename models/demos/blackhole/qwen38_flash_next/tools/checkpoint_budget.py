# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed checkpoint accounting and four-P150 placement budget.

This module accounts for checkpoint payloads only. Runtime allocator, trace,
collective, and activation reservations belong in the bring-up admission
budget and must not be inferred from these values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

TILE_ELEMENTS = 32 * 32
BF4_TILE_BYTES = 576
BF8_TILE_BYTES = 1088
BF16_BYTES = 2

TARGET_TENSOR_COUNT = 1658
TARGET_CHECKPOINT_BYTES = 359_999_963_128
TARGET_BF16_ELEMENTS = 179_999_981_424
TARGET_I64_ELEMENTS = 35
TARGET_EXPERT_COUNT = 512
TARGET_QSA_KV_HEADS = 2
TARGET_QSA_HEAD_DIM = 256
TARGET_INDEX_QUERY_HEADS = 4
TARGET_INDEX_KV_HEADS = 1
TARGET_INDEX_HEAD_DIM = 128

# Exact output of moe_compute_utils' ring-aware packer for one device and one
# Qwen4Exp layer (128 resident experts).  Raw BF4 tile accounting is not a
# valid device admission bound for this kernel layout.
PACKED_ROUTED_BYTES_PER_LAYER = {
    7: {"routed_gate_up": 346_816_512, "routed_down": 130_056_192},
    8: {"routed_gate_up": 396_361_728, "routed_down": 148_635_648},
}


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    elements: int
    data_bytes: int


def read_tensor_manifest(path: Path) -> list[TensorRecord]:
    """Read the deterministic TSV emitted by ``safetensors_metadata.py``."""

    records: list[TensorRecord] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"{path}:{line_number}: expected five tab-separated fields")
            name, dtype, shape_text, elements_text, data_bytes_text = fields
            try:
                shape = tuple(int(value) for value in shape_text.split("x"))
                elements = int(elements_text)
                data_bytes = int(data_bytes_text)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid numeric field") from error
            if not name or not shape or any(dimension < 0 for dimension in shape):
                raise ValueError(f"{path}:{line_number}: invalid name or shape")
            if math.prod(shape) != elements:
                raise ValueError(f"{path}:{line_number}: shape does not match element count")
            dtype_bytes = {"BF16": 2, "I64": 8}.get(dtype)
            if dtype_bytes is None or elements * dtype_bytes != data_bytes:
                raise ValueError(f"{path}:{line_number}: unsupported or inconsistent dtype")
            records.append(TensorRecord(name, dtype, shape, elements, data_bytes))
    if len({record.name for record in records}) != len(records):
        raise ValueError(f"{path}: duplicate tensor name")
    return records


def checkpoint_category(name: str) -> str:
    """Classify every known checkpoint tensor into an explicit load domain."""

    if name.startswith("model.visual."):
        return "vision_omitted"
    if ".ple.ple_embedding.ngram_embedding.shard_" in name:
        return "ple_host_table"
    if ".ple.ple_embedding." in name:
        return "ple_host_metadata"
    if name.endswith(".mlp.experts.gate_up_proj"):
        return "routed_gate_up"
    if name.endswith(".mlp.experts.down_proj"):
        return "routed_down"
    if name == "lm_head.weight":
        return "lm_head"
    if name == "model.language_model.embed_tokens.weight":
        return "token_embedding"
    if name.startswith("mtp."):
        return "mtp_nonexpert"
    if ".ple." in name and name.startswith("model.language_model."):
        return "ple_device_non_table"
    if name.startswith("model.language_model."):
        return "backbone_nonexpert"
    raise ValueError(f"unclassified tensor {name!r}")


def aggregate_categories(records: list[TensorRecord]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"tensors": 0, "elements": 0, "checkpoint_bytes": 0})
    for record in records:
        category = checkpoint_category(record.name)
        totals[category]["tensors"] += 1
        totals[category]["elements"] += record.elements
        totals[category]["checkpoint_bytes"] += record.data_bytes
    return dict(sorted(totals.items()))


def bfp_tile_storage_bytes(record: TensorRecord, tile_bytes: int) -> int:
    """Return packed TT tile storage for a tile-aligned matrix or expert stack."""

    if len(record.shape) < 2 or any(dimension % 32 for dimension in record.shape[-2:]):
        raise ValueError(f"{record.name}: final matrix dimensions are not tile aligned: {record.shape}")
    if tile_bytes <= 0:
        raise ValueError("tile_bytes must be positive")
    if record.elements % TILE_ELEMENTS:
        raise ValueError(f"{record.name}: element count is not a whole number of tiles")
    return record.elements // TILE_ELEMENTS * tile_bytes


def _is_qsa_kv_projection(name: str) -> bool:
    return name.endswith(".self_attn.k_proj.weight") or name.endswith(".self_attn.v_proj.weight")


def _is_qsa_index_projection(name: str) -> bool:
    return name.endswith(".self_attn.indexer.index_qk_proj.weight")


def plan_a_per_device_weight_bytes(
    records: list[TensorRecord], *, mesh_size: int = 4, ring_size: int
) -> dict[str, int]:
    """Account for the conservative baseline weight placement on each device.

    Routed experts use BF4_B and shard the 512-expert axis. Other matrix
    payloads remain BF16 in this admission bound even when the eventual loader
    uses BF8_B. QSA's two KV heads use two head groups, with each head replicated
    on two devices. The one-head indexer key is replicated while its four query
    heads are sharded. One-dimensional parameters are conservatively replicated.
    """

    if mesh_size != 4:
        raise ValueError(f"the required placement is TP4, not mesh_size={mesh_size}")
    if ring_size not in PACKED_ROUTED_BYTES_PER_LAYER:
        raise ValueError(f"Blackhole ring_size must be 7 or 8, got {ring_size}")
    totals = {
        "routed_expert_bf4_ep": 0,
        "qsa_kv_grouped_bf16": 0,
        "qsa_indexer_split_bf16": 0,
        "replicated_bf16": 0,
        "tp4_bf16": 0,
        "host_only": 0,
        "vision_omitted": 0,
    }
    routed_layer_bytes: dict[str, int] = defaultdict(int)
    for record in records:
        category = checkpoint_category(record.name)
        if category in {"ple_host_table", "ple_host_metadata"}:
            totals["host_only"] += record.data_bytes
            continue
        if category == "vision_omitted":
            totals["vision_omitted"] += record.data_bytes
            continue
        if record.dtype != "BF16":
            raise ValueError(f"unexpected device-resident dtype for {record.name}: {record.dtype}")
        if category in {"routed_gate_up", "routed_down"}:
            if len(record.shape) != 3 or record.shape[0] != TARGET_EXPERT_COUNT:
                raise ValueError(f"{record.name}: expected expert axis {TARGET_EXPERT_COUNT}, got {record.shape}")
            if record.shape[0] % mesh_size:
                raise ValueError(f"{record.name}: expert axis is not divisible by TP4")
            packed_bytes = PACKED_ROUTED_BYTES_PER_LAYER[ring_size][category]
            totals["routed_expert_bf4_ep"] += packed_bytes
            layer_key = record.name.rsplit(".mlp.experts.", 1)[0]
            routed_layer_bytes[layer_key] += packed_bytes
            continue
        if _is_qsa_kv_projection(record.name):
            expected_rows = TARGET_QSA_KV_HEADS * TARGET_QSA_HEAD_DIM
            if record.shape != (expected_rows, 2560):
                raise ValueError(f"{record.name}: unexpected QSA KV projection shape {record.shape}")
            # One KV head per device, replicated within each two-device Q-head group.
            totals["qsa_kv_grouped_bf16"] += TARGET_QSA_HEAD_DIM * record.shape[1] * BF16_BYTES
            continue
        if _is_qsa_index_projection(record.name):
            query_rows = TARGET_INDEX_QUERY_HEADS * TARGET_INDEX_HEAD_DIM
            key_rows = TARGET_INDEX_KV_HEADS * TARGET_INDEX_HEAD_DIM
            if record.shape != (query_rows + key_rows, 2560):
                raise ValueError(f"{record.name}: unexpected QSA index projection shape {record.shape}")
            local_query_rows = query_rows // mesh_size
            # The single key head is present on every device.
            totals["qsa_indexer_split_bf16"] += (local_query_rows + key_rows) * record.shape[1] * BF16_BYTES
            continue
        if len(record.shape) == 1:
            totals["replicated_bf16"] += record.data_bytes
            continue
        if record.data_bytes % mesh_size:
            raise ValueError(f"{record.name}: BF16 payload is not evenly TP4-shardable")
        totals["tp4_bf16"] += record.data_bytes // mesh_size

    totals["device_weight_total"] = sum(
        totals[key]
        for key in (
            "routed_expert_bf4_ep",
            "qsa_kv_grouped_bf16",
            "qsa_indexer_split_bf16",
            "replicated_bf16",
            "tp4_bf16",
        )
    )
    totals["routed_expert_bf4_stream_slot"] = max(routed_layer_bytes.values(), default=0)
    totals["device_weight_total_streamed"] = (
        totals["device_weight_total"] - totals["routed_expert_bf4_ep"] + totals["routed_expert_bf4_stream_slot"]
    )
    return totals


def routed_precision_variants(records: list[TensorRecord], *, mesh_size: int = 4) -> dict[str, int]:
    if mesh_size != 4:
        raise ValueError(f"the required placement is TP4, not mesh_size={mesh_size}")
    gate_up = [record for record in records if checkpoint_category(record.name) == "routed_gate_up"]
    down = [record for record in records if checkpoint_category(record.name) == "routed_down"]
    gate_up_bf4 = sum(bfp_tile_storage_bytes(record, BF4_TILE_BYTES) for record in gate_up)
    down_bf4 = sum(bfp_tile_storage_bytes(record, BF4_TILE_BYTES) for record in down)
    gate_up_bf8 = sum(bfp_tile_storage_bytes(record, BF8_TILE_BYTES) for record in gate_up)
    down_bf8 = sum(bfp_tile_storage_bytes(record, BF8_TILE_BYTES) for record in down)
    variants = {
        "all_bf4_global": gate_up_bf4 + down_bf4,
        "all_bf4_per_device": (gate_up_bf4 + down_bf4) // mesh_size,
        "bf8_down_bf4_gate_up_global": gate_up_bf4 + down_bf8,
        "bf8_down_bf4_gate_up_per_device": (gate_up_bf4 + down_bf8) // mesh_size,
        "all_bf8_global": gate_up_bf8 + down_bf8,
        "all_bf8_per_device": (gate_up_bf8 + down_bf8) // mesh_size,
    }
    return variants


def validate_target_checkpoint(records: list[TensorRecord]) -> None:
    """Fail if the manifest is not the exact release checkpoint contract."""

    if len(records) != TARGET_TENSOR_COUNT:
        raise ValueError(f"expected {TARGET_TENSOR_COUNT} tensors, got {len(records)}")
    if sum(record.data_bytes for record in records) != TARGET_CHECKPOINT_BYTES:
        raise ValueError("checkpoint tensor byte total does not match the pinned release")
    elements_by_dtype: dict[str, int] = defaultdict(int)
    for record in records:
        elements_by_dtype[record.dtype] += record.elements
    expected_elements = {"BF16": TARGET_BF16_ELEMENTS, "I64": TARGET_I64_ELEMENTS}
    if dict(elements_by_dtype) != expected_elements:
        raise ValueError(f"checkpoint dtype totals differ: {dict(elements_by_dtype)}")

    categories = aggregate_categories(records)
    expected_category_counts = {
        "backbone_nonexpert": 1059,
        "lm_head": 1,
        "mtp_nonexpert": 29,
        "ple_device_non_table": 6,
        "ple_host_metadata": 3,
        "ple_host_table": 128,
        "routed_down": 49,
        "routed_gate_up": 49,
        "token_embedding": 1,
        "vision_omitted": 333,
    }
    actual_category_counts = {name: values["tensors"] for name, values in categories.items()}
    if actual_category_counts != expected_category_counts:
        raise ValueError(f"checkpoint category counts differ: {actual_category_counts}")

    for record in records:
        category = checkpoint_category(record.name)
        if category == "routed_down" and record.shape != (512, 2560, 640):
            raise ValueError(f"unexpected routed down shape: {record.name} {record.shape}")
        if category == "routed_gate_up" and record.shape != (512, 1280, 2560):
            raise ValueError(f"unexpected routed gate/up shape: {record.name} {record.shape}")
        if category == "ple_host_table" and record.shape != (2_500_012, 160):
            raise ValueError(f"unexpected PLE table shard shape: {record.name} {record.shape}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_budget_report(manifest: Path) -> dict[str, object]:
    records = read_tensor_manifest(manifest)
    validate_target_checkpoint(records)
    return {
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "target_contract_valid": True,
        "categories": aggregate_categories(records),
        "plan_a_per_device_weight_bytes_ring7": plan_a_per_device_weight_bytes(records, ring_size=7),
        "plan_a_per_device_weight_bytes_ring8": plan_a_per_device_weight_bytes(records, ring_size=8),
        "routed_precision_variants": routed_precision_variants(records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(build_budget_report(arguments.manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
