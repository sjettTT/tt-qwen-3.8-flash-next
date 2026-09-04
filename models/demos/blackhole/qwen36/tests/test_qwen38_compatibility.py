# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only compatibility checks for the Qwen3.8-27B checkpoint contract."""

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
_MODEL_PARAMS = _REPO_ROOT / "models" / "tt_transformers" / "model_params"


def _config(model_name):
    with open(_MODEL_PARAMS / model_name / "config.json") as f:
        return json.load(f)


def test_qwen38_architecture_matches_qwen36_runtime_contract():
    qwen36 = _config("Qwen3.6-27B")
    qwen38 = _config("Qwen3.8-27B")

    # Qwen3.8 was exported with a newer Transformers build. All runtime-bearing
    # config fields are otherwise identical to the already-supported Qwen3.6.
    qwen36.pop("transformers_version")
    qwen38.pop("transformers_version")
    assert qwen38 == qwen36


def test_qwen38_text_geometry_and_layer_pattern():
    config = _config("Qwen3.8-27B")
    text = config["text_config"]

    assert config["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert config["model_type"] == "qwen3_5"
    assert (
        text["num_hidden_layers"],
        text["hidden_size"],
        text["intermediate_size"],
        text["vocab_size"],
    ) == (64, 5120, 17408, 248320)
    assert (text["num_attention_heads"], text["num_key_value_heads"], text["head_dim"]) == (24, 4, 256)
    assert (
        text["linear_num_key_heads"],
        text["linear_key_head_dim"],
        text["linear_num_value_heads"],
        text["linear_value_head_dim"],
    ) == (16, 128, 48, 128)
    assert (
        text["layer_types"]
        == [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
        * 16
    )


def test_qwen38_gdn_conv_weight_remaps_to_27b_stream_widths():
    import torch

    from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict

    # Meta tensors validate names and shapes without allocating multi-gigabyte weights.
    raw = {
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": torch.empty((10240, 5120), device="meta"),
        "model.language_model.layers.0.linear_attn.conv1d.weight": torch.empty((10240, 1, 4), device="meta"),
        "model.language_model.embed_tokens.weight": torch.empty((248320, 5120), device="meta"),
        "lm_head.weight": torch.empty((248320, 5120), device="meta"),
    }

    remapped = remap_qwen36_state_dict(raw)
    assert remapped["layers.0.linear_attn.qkv_proj.weight"].shape == (10240, 5120)
    assert remapped["layers.0.linear_attn.q_conv.weight"].shape == (2048, 1, 4)
    assert remapped["layers.0.linear_attn.k_conv.weight"].shape == (2048, 1, 4)
    assert remapped["layers.0.linear_attn.v_conv.weight"].shape == (6144, 1, 4)
    assert remapped["tok_embeddings.weight"].shape == (248320, 5120)
    assert remapped["output.weight"].shape == (248320, 5120)
