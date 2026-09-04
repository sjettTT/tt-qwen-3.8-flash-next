# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-device Qwen3.5/3.8 multi-token predictor prototype."""

import json
import math
import os
import time

import torch

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_mtp_layer_state_dict
from models.tt_transformers.tt.common import Mode


class Qwen36MTP:
    """The checkpoint's one-layer second-token predictor.

    The target embedding and LM head are shared.  The MTP-only parameters are
    two input RMSNorms, a 2H->H projection, one dense full-attention decoder
    layer, and a final RMSNorm.  This first prototype intentionally fails closed
    on tensor parallelism; it is used to establish correctness, acceptance, and
    proposer cost on one P150 before adding a packed verifier.
    """

    def __init__(self, target_model, state_dict, tensor_cache_path=None):
        self.target_model = target_model
        self.device = target_model.device
        self.args = target_model.args
        self.num_devices = target_model.num_devices
        if self.num_devices != 1:
            raise NotImplementedError("Qwen MTP prototype currently supports exactly one P150")

        mtp_cache = tensor_cache_path / "mtp" if tensor_cache_path else None
        norm_kwargs = dict(
            device=self.device,
            dim=self.args.dim,
            state_dict=state_dict,
            state_dict_prefix="mtp.",
            weight_cache_path=mtp_cache,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=self.args.norm_eps,
        )
        self.embedding_norm = RMSNorm(weight_key="pre_fc_norm_embedding", **norm_kwargs)
        self.hidden_norm = RMSNorm(weight_key="pre_fc_norm_hidden", **norm_kwargs)
        self.norm = RMSNorm(weight_key="norm", **norm_kwargs)

        fc_weight = state_dict["mtp.fc.weight"]
        expected_shape = (self.args.dim, 2 * self.args.dim)
        if tuple(fc_weight.shape) != expected_shape:
            raise ValueError(f"mtp.fc.weight has shape {tuple(fc_weight.shape)}, expected {expected_shape}")
        self.fc_weight = ttnn.as_tensor(
            fc_weight.T.contiguous(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(mtp_cache / "fc.weight") if mtp_cache else None,
        )
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        full_attention_idx = next(
            (i for i, layer_type in enumerate(self.args.attention_type_list) if layer_type == "full_attention"),
            None,
        )
        if full_attention_idx is None:
            raise ValueError("Qwen checkpoint has no full-attention layer type for the MTP decoder")
        layer_state = remap_qwen36_mtp_layer_state_dict(state_dict, full_attention_idx)
        self.layer = Qwen36DecoderLayer(
            self.device,
            self.args,
            layer_state,
            full_attention_idx,
            mtp_cache,
            tt_ccl=None,
        )
        # MTP residuals and attention outputs share a device with the wide
        # eager target, so keep them out of its L1 workspace.
        self.layer.attention.decode_memory_config = ttnn.DRAM_MEMORY_CONFIG
        self.layer.decode_residual_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Concat KV changes shape at every speculative depth and forces TTNN to
        # build a fresh host program for each unseen sequence length.  Keep the
        # MTP layer on the existing paged-attention path instead: all decode
        # steps then share one fixed cache shape and differ only by the runtime
        # position tensor.  B=1 uses an identity page table.
        self.kv_block_size = 32
        self.num_kv_blocks = math.ceil(self.args.max_seq_len / self.kv_block_size)
        kv_shape = (
            self.num_kv_blocks,
            self.args.n_kv_heads,
            self.kv_block_size,
            self.args.head_dim,
        )
        self.k_cache = ttnn.zeros(
            kv_shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.v_cache = ttnn.zeros(
            kv_shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.page_table = ttnn.from_torch(
            torch.arange(self.num_kv_blocks, dtype=torch.int32).reshape(1, -1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        self.layer.attention.set_paged_kv_cache(self.k_cache, self.v_cache)

    def reset_state(self):
        self.layer.attention.reset_cache()

    def checkpoint_state(self):
        """Return a lightweight checkpoint of the MTP cache.

        Paged decode writes by absolute position.  A rejected block is restored
        by replaying target tokens over the same positions, so the fixed cache
        does not need a tensor snapshot.
        """
        if self.layer.attention.use_paged_attention:
            return None
        return self.layer.attention.past_key, self.layer.attention.past_value

    def restore_state(self, checkpoint):
        """Restore concat state, or rewind paged state for target replay."""
        if self.layer.attention.use_paged_attention:
            if checkpoint is not None:
                raise ValueError("Paged MTP cache checkpoint must be None")
            return
        if not isinstance(checkpoint, tuple) or len(checkpoint) != 2:
            raise ValueError("MTP cache checkpoint must be a (past_key, past_value) tuple")
        saved_key, saved_value = checkpoint
        attention = self.layer.attention
        if attention.past_key is not None and attention.past_key is not saved_key:
            ttnn.deallocate(attention.past_key)
        if attention.past_value is not None and attention.past_value is not saved_value:
            ttnn.deallocate(attention.past_value)
        attention.past_key = saved_key
        attention.past_value = saved_value

    def _project(self, hidden_states, token_ids, mode):
        token_ids_tt = ttnn.from_torch(token_ids.to(torch.int32), dtype=ttnn.uint32, device=self.device)
        token_embedding = self.target_model.embd(token_ids_tt)
        ttnn.deallocate(token_ids_tt)

        norm_mode = Mode.PREFILL if mode == "prefill" else Mode.DECODE
        token_norm = self.embedding_norm(token_embedding, mode=norm_mode)
        hidden_norm = self.hidden_norm(hidden_states, mode=norm_mode)
        ttnn.deallocate(token_embedding)

        combined = ttnn.concat([token_norm, hidden_norm], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(token_norm)
        ttnn.deallocate(hidden_norm)
        projected = ttnn.linear(
            combined,
            self.fc_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(combined)
        return projected

    def prefill(self, hidden_states, shifted_token_ids):
        """Prime the MTP layer cache from target hidden[i] and prompt token[i+1]."""
        if hidden_states.shape[0] != shifted_token_ids.shape[0] or hidden_states.shape[1] != shifted_token_ids.shape[1]:
            raise ValueError(
                f"MTP prefill hidden/token mismatch: {tuple(hidden_states.shape)} vs {tuple(shifted_token_ids.shape)}"
            )
        self.reset_state()
        x = self._project(hidden_states, shifted_token_ids, mode="prefill")
        seq_len = int(shifted_token_ids.shape[1])
        cos, sin = self.target_model.rope.get_prefill_rot_mats(0, seq_len)
        output = self.layer.forward(x, cos=cos, sin=sin, mode="prefill")
        ttnn.deallocate(x)
        # Short prefill intentionally uses the concat attention path.  Copy its
        # K/V into the fixed paged cache once, then release the concat tensors.
        attention = self.layer.attention
        if attention.past_key is not None:
            ttnn.experimental.paged_fill_cache(self.k_cache, attention.past_key, self.page_table, batch_idx=0)
            ttnn.experimental.paged_fill_cache(self.v_cache, attention.past_value, self.page_table, batch_idx=0)
            ttnn.deallocate(attention.past_key)
            ttnn.deallocate(attention.past_value)
            attention.past_key = None
            attention.past_value = None
        # The output is intentionally discarded; only the attention cache is
        # needed to bootstrap autoregressive MTP decode.
        ttnn.deallocate(output)

    def decode(self, hidden_states, token_ids, position, return_hidden_states=False):
        """Predict the token after ``token_ids`` using the preceding hidden state.

        ``return_hidden_states`` exposes the final normalized MTP state.  That
        state is the recurrent input for the next speculative depth, matching
        the Qwen/vLLM MTP contract.  The default keeps the original logits-only
        API and releases the state after the shared LM head has consumed it.
        """
        if tuple(token_ids.shape) != (1, 1):
            raise ValueError(f"MTP decode expects token_ids [1, 1], got {tuple(token_ids.shape)}")
        profile = os.environ.get("QWEN_MTP_PROFILE") == "1"
        # The eager target and proposer share two command queues.  Explicit
        # boundaries keep temporary lifetimes ordered across those queues;
        # without them a long target stream can make this one-layer proposer
        # stall for hundreds of milliseconds.  Callers can override the list
        # for queue-level bringup experiments.
        sync_stages = set(
            filter(
                None,
                os.environ.get(
                    "QWEN_MTP_SYNC_STAGES",
                    "projection,rope_and_position,decoder_layer,final_norm,lm_head",
                ).split(","),
            )
        )
        profile_ms = {}

        def mark(stage, start):
            if not profile and stage not in sync_stages:
                return start
            ttnn.synchronize_device(self.device)
            now = time.perf_counter()
            if profile:
                profile_ms[stage] = (now - start) * 1000.0
            return now

        if profile:
            ttnn.synchronize_device(self.device)
        stage_start = time.perf_counter()
        x = self._project(hidden_states, token_ids, mode="decode")
        stage_start = mark("projection", stage_start)
        position_ids = torch.tensor([[int(position)]], dtype=torch.long)
        cos, sin = self.target_model.rope.get_rot_mats(position_ids)
        cur_pos_tensor = ttnn.from_torch(
            torch.full((1,), int(position), dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        stage_start = mark("rope_and_position", stage_start)
        output = self.layer.forward(
            x,
            cos=cos,
            sin=sin,
            mode="decode",
            position_tensor=cur_pos_tensor,
            page_table=self.page_table,
        )
        ttnn.deallocate(x)
        ttnn.deallocate(cur_pos_tensor)
        stage_start = mark("decoder_layer", stage_start)
        normed = self.norm(output, mode=Mode.DECODE)
        ttnn.deallocate(output)
        stage_start = mark("final_norm", stage_start)
        logits = self.target_model._lm_head(normed)
        mark("lm_head", stage_start)
        if profile:
            print(f"MTP_PROFILE_JSON={json.dumps(profile_ms, sort_keys=True)}", flush=True)
        if return_hidden_states:
            return logits, normed
        ttnn.deallocate(normed)
        return logits
