# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Remap Hugging Face Qwen3.5/3.6/3.8 state dicts to the internal format.

Handles:
- Stripping 'model.language_model.' prefix
- Filtering out vision encoder and MTP weights
- Renaming combined in_proj_qkv → qkv_proj (DeltaNet layers; the op uses the fused weight)
- Splitting combined conv1d.weight into separate Q, K, V conv weights (DeltaNet layers)
- Renaming lm_head.weight → output.weight
- Renaming embed_tokens → tok_embeddings
"""
import json
from pathlib import Path
from typing import Dict

import torch

# DeltaNet QKV split dimensions (used to split the combined conv1d.weight into
# per-stream Q/K/V conv weights — the QKV projection itself stays combined).
# These key-stream dimensions are shared by the supported 9B and 27B configs;
# the value stream is the remainder (4096 for 9B, 6144 for 27B).
# Q: num_key_heads(16) × key_head_dim(128) = 2048
# K: num_key_heads(16) × key_head_dim(128) = 2048
# V is the remaining slice: 32×128 = 4096 for 9B, 48×128 = 6144 for 27B.
LINEAR_Q_DIM = 2048
LINEAR_K_DIM = 2048


MTP_REQUIRED_KEYS = {
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
}


def remap_qwen36_mtp_layer_state_dict(
    mtp_state_dict: Dict[str, torch.Tensor], layer_idx: int
) -> Dict[str, torch.Tensor]:
    """Present ``mtp.layers.0`` as a regular dense Qwen decoder layer.

    ``Qwen36DecoderLayer`` dispatches from a checkpoint layer index.  MTP has a
    dedicated dense layer outside ``model.layers``, so map it onto any known
    full-attention index while retaining the dedicated tensors.
    """
    prefix = "mtp.layers.0."
    remapped = {
        f"layers.{layer_idx}.{key[len(prefix):]}": tensor
        for key, tensor in mtp_state_dict.items()
        if key.startswith(prefix)
    }
    if not remapped:
        raise ValueError("MTP state dict does not contain mtp.layers.0 weights")
    return remapped


def load_qwen36_mtp_state_dict(model_path) -> Dict[str, torch.Tensor]:
    """Load only the dedicated MTP tensors from a local safetensors checkpoint.

    This avoids loading a second copy of the full 27B checkpoint.  Qwen3.8 stores
    all MTP tensors in one shard, but the implementation follows the index and
    therefore also supports checkpoints that split them over multiple shards.
    """
    from safetensors import safe_open

    from models.demos.blackhole.qwen36.tt.tp_common import dequant_fp8_block

    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        mtp_keys = {key for key in weight_map if key.startswith("mtp.")}
        file_to_keys: Dict[str, list[str]] = {}
        for key in mtp_keys:
            file_to_keys.setdefault(weight_map[key], []).append(key)
    else:
        single_path = model_path / "model.safetensors"
        if not single_path.is_file():
            raise FileNotFoundError(f"No safetensors checkpoint found under {model_path}")
        with safe_open(str(single_path), framework="pt") as sf:
            mtp_keys = {key for key in sf.keys() if key.startswith("mtp.")}
        file_to_keys = {single_path.name: sorted(mtp_keys)}

    missing = MTP_REQUIRED_KEYS - mtp_keys
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Checkpoint is missing required Qwen MTP weights: {missing_str}")

    raw: Dict[str, torch.Tensor] = {}
    for filename, keys in file_to_keys.items():
        with safe_open(str(model_path / filename), framework="pt") as sf:
            present = set(sf.keys())
            for key in keys:
                if key in present:
                    raw[key] = sf.get_tensor(key)

    state_dict: Dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        if key.endswith(".weight_scale_inv"):
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = key + "_scale_inv"
            state_dict[key] = (
                dequant_fp8_block(tensor, raw[scale_key]) if scale_key in raw else tensor.to(torch.bfloat16)
            )
        else:
            state_dict[key] = tensor
    return state_dict


def remap_qwen36_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap an HF Qwen3.5-family state dict to the internal format.

    Args:
        state_dict: Raw HuggingFace state dict loaded from safetensors.

    Returns:
        Remapped state dict with internal naming convention.
    """
    remapped = {}

    for key, tensor in state_dict.items():
        # Filter out vision encoder weights (check original key — no prefix stripping yet)
        if "visual" in key or key.startswith("model.visual"):
            continue
        # Filter out MTP (multi-token prediction) weights (original key)
        if key.startswith("mtp"):
            continue

        # Strip the language-model prefix. Two checkpoint sources produce different
        # prefixes for the same internal weights:
        #   - raw sharded safetensors:           model.language_model.X
        #   - AutoModelForCausalLM.from_pretrained (text-only Qwen3_5ForCausalLM): model.X
        # Strip whichever matches, longest first, so BOTH sources yield identical
        # internal keys. Top-level weights like lm_head.weight have no prefix and are
        # matched against the original `key` below.
        new_key = key
        for prefix in ("model.language_model.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break

        # Rename top-level weights
        if new_key == "embed_tokens.weight":
            remapped["tok_embeddings.weight"] = tensor
            continue
        if key == "lm_head.weight":
            remapped["output.weight"] = tensor
            continue
        # Final norm (model.language_model.norm.weight)
        if new_key == "norm.weight":
            remapped["norm.weight"] = tensor
            continue

        # Handle per-layer weights
        if new_key.startswith("layers."):
            parts = new_key.split(".")
            layer_idx = int(parts[1])
            layer_prefix = f"layers.{layer_idx}"
            sub_key = ".".join(parts[2:])

            # DeltaNet layers: keep ONLY the combined QKV weight. The split q/k/v_proj
            # were dead — the op runs the fused QKV projection from the combined weight
            # (it only read the splits in a fallback reached when qkv_proj_weight is None,
            # which never happens for the 9B).
            if sub_key == "linear_attn.in_proj_qkv.weight":
                remapped[f"{layer_prefix}.linear_attn.qkv_proj.weight"] = tensor  # [8192, 4096]
                continue

            if sub_key == "linear_attn.conv1d.weight":
                conv = tensor  # [8192, 1, 4]
                q_conv = conv[:LINEAR_Q_DIM, :, :]
                k_conv = conv[LINEAR_Q_DIM : LINEAR_Q_DIM + LINEAR_K_DIM, :, :]
                v_conv = conv[LINEAR_Q_DIM + LINEAR_K_DIM :, :, :]
                remapped[f"{layer_prefix}.linear_attn.q_conv.weight"] = q_conv
                remapped[f"{layer_prefix}.linear_attn.k_conv.weight"] = k_conv
                remapped[f"{layer_prefix}.linear_attn.v_conv.weight"] = v_conv
                continue

            # All other keys pass through unchanged
            remapped[new_key] = tensor
            continue

        # Any remaining keys pass through
        remapped[new_key] = tensor

    return remapped


def is_fp8_checkpoint(model_path) -> bool:
    """True when the checkpoint dir holds block-wise FP8 safetensors.

    Detected by a ``*.weight_scale_inv`` entry in the safetensors index (the
    per-block dequant scales that accompany float8_e4m3fn weights). Such a
    checkpoint cannot be loaded via AutoModelForCausalLM here; use
    ``load_qwen36_state_dict_fp8`` instead.
    """
    index_path = Path(model_path) / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    except (KeyError, ValueError, OSError):
        return False
    return any(k.endswith(".weight_scale_inv") for k in weight_map)


def load_qwen36_state_dict_fp8(model_path) -> Dict[str, torch.Tensor]:
    """Load Qwen3.5 FP8 weights: block-wise dequant + minimal key remap.

    Produces the SAME internal key scheme as ``remap_qwen36_state_dict`` for the
    shared/simple weights (``layers.N.mlp.*``, ``layers.N.self_attn.*``,
    ``input_layernorm`` / ``post_attention_layernorm``, ``tok_embeddings``,
    ``norm``, ``output``) so ``layer.py``'s substate extraction is unchanged.

    The one deliberate difference vs the single-device remap: GDN ``linear_attn.*``
    projections are kept RAW (fused ``in_proj_qkv``, fused ``conv1d``, plus
    ``in_proj_z`` / ``in_proj_a`` / ``in_proj_b`` / ``out_proj`` / ``A_log`` /
    ``dt_bias`` / ``norm.weight``) — NOT split or renamed — so the tensor-parallel
    GDN weight-prep helpers (prepare_gdn_qkv / prepare_conv_taps) can reorder and
    shard them per device. The TP module loaders branch on this raw layout.
    """
    from safetensors import safe_open

    from models.demos.blackhole.qwen36.tt.tp_common import dequant_fp8_block

    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    file_to_keys: Dict[str, list] = {}
    for key, filename in weight_map.items():
        file_to_keys.setdefault(filename, []).append(key)

    raw: Dict[str, torch.Tensor] = {}
    for filename, keys in file_to_keys.items():
        with safe_open(str(model_path / filename), framework="pt") as sf:
            present = set(sf.keys())
            for key in keys:
                if key in present:
                    raw[key] = sf.get_tensor(key)

    # Dequantize FP8 (skip the scale tensors themselves)
    dequantized: Dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        if key.endswith(".weight_scale_inv"):
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = key + "_scale_inv"
            dequantized[key] = (
                dequant_fp8_block(tensor, raw[scale_key]) if scale_key in raw else tensor.to(torch.bfloat16)
            )
        else:
            dequantized[key] = tensor

    state_dict: Dict[str, torch.Tensor] = {}
    for key, tensor in dequantized.items():
        if "visual" in key or key.startswith("mtp"):
            continue
        short = key
        for prefix in ("model.language_model.", "model."):
            if short.startswith(prefix):
                short = short[len(prefix) :]
                break
        if "embed_tokens" in short:
            state_dict["tok_embeddings.weight"] = tensor
        elif key == "lm_head.weight" or short == "lm_head.weight":
            state_dict["output.weight"] = tensor
        else:
            # Everything else (layers.N.mlp.*, self_attn.*, linear_attn.* RAW,
            # input_layernorm/post_attention_layernorm, norm) passes through.
            state_dict[short] = tensor

    return state_dict
