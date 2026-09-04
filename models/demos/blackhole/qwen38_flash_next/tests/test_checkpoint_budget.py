# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import unittest

from models.demos.blackhole.qwen38_flash_next.tools.checkpoint_budget import (
    BF4_TILE_BYTES,
    TensorRecord,
    bfp_tile_storage_bytes,
    checkpoint_category,
    plan_a_per_device_weight_bytes,
)


def _tensor(name: str, shape: tuple[int, ...], *, dtype: str = "BF16") -> TensorRecord:
    dtype_bytes = {"BF16": 2, "I64": 8}[dtype]
    elements = 1
    for dimension in shape:
        elements *= dimension
    return TensorRecord(name, dtype, shape, elements, elements * dtype_bytes)


class CheckpointBudgetTest(unittest.TestCase):
    def test_every_checkpoint_domain_is_explicit(self):
        cases = {
            "model.visual.blocks.0.norm1.weight": "vision_omitted",
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": "ple_host_table",
            "model.language_model.layers.1.ple.ple_embedding.layer_multipliers": "ple_host_metadata",
            "model.language_model.layers.1.ple.key_proj.weight": "ple_device_non_table",
            "model.language_model.layers.0.mlp.experts.down_proj": "routed_down",
            "mtp.layers.0.mlp.experts.gate_up_proj": "routed_gate_up",
            "model.language_model.embed_tokens.weight": "token_embedding",
            "lm_head.weight": "lm_head",
            "mtp.fc_hidden.weight": "mtp_nonexpert",
            "model.language_model.layers.0.linear_attn.A_log": "backbone_nonexpert",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(checkpoint_category(name), expected)

        with self.assertRaisesRegex(ValueError, "unclassified tensor"):
            checkpoint_category("unexpected.weight")

    def test_bfp_storage_uses_tenstorrent_tile_payload(self):
        routed_down = _tensor(
            "model.language_model.layers.0.mlp.experts.down_proj",
            (512, 2560, 640),
        )
        self.assertEqual(bfp_tile_storage_bytes(routed_down, BF4_TILE_BYTES), 471_859_200)

        misaligned = _tensor(
            "model.language_model.layers.0.mlp.experts.down_proj",
            (512, 2559, 640),
        )
        with self.assertRaisesRegex(ValueError, "tile aligned"):
            bfp_tile_storage_bytes(misaligned, BF4_TILE_BYTES)

    def test_plan_a_models_tp4_grouped_kv_and_replicated_vectors(self):
        records = [
            _tensor("model.language_model.layers.0.mlp.experts.down_proj", (512, 2560, 640)),
            _tensor("model.language_model.layers.3.self_attn.k_proj.weight", (512, 2560)),
            _tensor("model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight", (640, 2560)),
            _tensor("model.language_model.layers.3.self_attn.q_norm.weight", (256,)),
            _tensor("model.language_model.layers.0.linear_attn.out_proj.weight", (2560, 6144)),
            _tensor(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                (2_500_012, 160),
            ),
            _tensor("model.visual.pos_embed", (2304, 1152)),
        ]

        result = plan_a_per_device_weight_bytes(records, mesh_size=4, ring_size=8)

        self.assertEqual(result["routed_expert_bf4_ep"], 148_635_648)
        self.assertEqual(result["routed_expert_bf4_stream_slot"], 148_635_648)
        self.assertEqual(result["qsa_kv_grouped_bf16"], 1_310_720)
        # Four query index heads are sharded; the single index key head is replicated.
        self.assertEqual(result["qsa_indexer_split_bf16"], 1_310_720)
        self.assertEqual(result["replicated_bf16"], 512)
        self.assertEqual(result["tp4_bf16"], 7_864_320)
        self.assertEqual(result["host_only"], 800_003_840)
        self.assertEqual(result["vision_omitted"], 5_308_416)

    def test_expert_parallel_fails_if_expert_axis_is_not_divisible(self):
        record = _tensor("mtp.layers.0.mlp.experts.down_proj", (510, 2560, 640))
        with self.assertRaisesRegex(ValueError, "expert axis"):
            plan_a_per_device_weight_bytes([record], mesh_size=4, ring_size=8)

    def test_ring_aware_packed_expert_budget(self):
        records = [
            _tensor("model.language_model.layers.0.mlp.experts.gate_up_proj", (512, 1280, 2560)),
            _tensor("model.language_model.layers.0.mlp.experts.down_proj", (512, 2560, 640)),
        ]
        ring7 = plan_a_per_device_weight_bytes(records, ring_size=7)
        ring8 = plan_a_per_device_weight_bytes(records, ring_size=8)
        self.assertEqual(ring7["routed_expert_bf4_ep"], 476_872_704)
        self.assertEqual(ring8["routed_expert_bf4_ep"], 544_997_376)
        self.assertEqual(ring7["device_weight_total_streamed"], ring7["device_weight_total"])


if __name__ == "__main__":
    unittest.main()
