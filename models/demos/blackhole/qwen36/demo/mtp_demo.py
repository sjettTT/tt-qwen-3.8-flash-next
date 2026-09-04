# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness-first recurrent Qwen3.8 MTP acceptance benchmark.

Run on one P150::

    MESH_DEVICE=P150 pytest models/demos/blackhole/qwen36/demo/mtp_demo.py -v -s

This measures a real K-token recurrent MTP proposer and a target teacher-forcing
stream.  A rejected block rewinds the MTP cache logically and replays the target
tokens over the speculative positions, so every later block starts from the
committed trajectory.  The
``optimistic_same_cost_packed_tsu`` field remains a projection, not an
implemented speculative decoder: it assumes a future packed K-token verifier
costs the same as one measured B=1 target step.
"""

import gc
import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _get_prompt
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2}]


def _argmax_token(logits):
    token = int(torch.argmax(ttnn.to_torch(logits).reshape(-1).float()).item())
    ttnn.deallocate(logits)
    return token


def _timed(device, fn):
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    output = fn()
    ttnn.synchronize_device(device)
    return output, (time.perf_counter() - start) * 1000.0


def _median(values):
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _accepted_prefix_length(draft_tokens, target_tokens):
    if len(draft_tokens) != len(target_tokens):
        raise ValueError("draft and target blocks must have the same length")
    for depth, (draft_token, target_token) in enumerate(zip(draft_tokens, target_tokens)):
        if draft_token != target_token:
            return depth
    return len(draft_tokens)


def _to_dram_hidden(hidden_states):
    if hidden_states.memory_config() == ttnn.DRAM_MEMORY_CONFIG:
        return hidden_states
    hidden_dram = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(hidden_states)
    return hidden_dram


def _draft_block(model, root_hidden, current_token, position, draft_tokens):
    """Generate a recurrent MTP block and return tokens plus per-depth times."""
    token = current_token
    hidden = root_hidden
    owns_hidden = False
    tokens = []
    times_ms = []
    for depth in range(draft_tokens):
        (logits, next_hidden), step_ms = _timed(
            model.device,
            lambda hidden=hidden, token=token, depth=depth: model.mtp.decode(
                hidden,
                torch.tensor([[token]], dtype=torch.long),
                position + depth,
                return_hidden_states=True,
            ),
        )
        next_token = _argmax_token(logits)
        if owns_hidden:
            ttnn.deallocate(hidden)
        hidden = next_hidden
        owns_hidden = True
        token = next_token
        tokens.append(next_token)
        times_ms.append(step_ms)
    if owns_hidden:
        ttnn.deallocate(hidden)
    return tokens, times_ms


def _replay_target_block(model, root_hidden, current_token, position, target_tokens):
    """Advance the MTP cache over the committed target path after rejection."""
    token = current_token
    hidden = root_hidden
    owns_hidden = False
    times_ms = []
    for depth, next_token in enumerate(target_tokens):
        (logits, next_hidden), step_ms = _timed(
            model.device,
            lambda hidden=hidden, token=token, depth=depth: model.mtp.decode(
                hidden,
                torch.tensor([[token]], dtype=torch.long),
                position + depth,
                return_hidden_states=True,
            ),
        )
        ttnn.deallocate(logits)
        if owns_hidden:
            ttnn.deallocate(hidden)
        hidden = next_hidden
        owns_hidden = True
        token = next_token
        times_ms.append(step_ms)
    if owns_hidden:
        ttnn.deallocate(hidden)
    return times_ms


def _log_l1(device, label):
    """Emit allocator state for opt-in bringup diagnostics."""
    if os.environ.get("QWEN_MTP_DEBUG_MEMORY") != "1":
        return
    ttnn.synchronize_device(device)
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    print(
        "MTP_L1 "
        f"stage={label} banks={view.num_banks} "
        f"allocated_per_bank={view.total_bytes_allocated_per_bank} "
        f"free_per_bank={view.total_bytes_free_per_bank} "
        f"largest_free_per_bank={view.largest_contiguous_bytes_free_per_bank}",
        flush=True,
    )


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_qwen38_mtp_prototype(mesh_device):
    from transformers import AutoTokenizer

    device = mesh_device
    device.enable_program_cache()
    prompt_len = int(os.environ.get("QWEN_MTP_PROMPT_LEN", "128"))
    draft_tokens = int(os.environ.get("QWEN_MTP_DRAFT_TOKENS", "4"))
    verify_blocks = int(os.environ.get("QWEN_MTP_VERIFY_BLOCKS", "8"))
    timing_warmup_blocks = int(os.environ.get("QWEN_MTP_TIMING_WARMUP_BLOCKS", "2"))
    decode_steps = verify_blocks * draft_tokens
    target_layers_env = os.environ.get("QWEN_MTP_TARGET_LAYERS")
    target_layers = int(target_layers_env) if target_layers_env else None
    if prompt_len < 32 or prompt_len % 32:
        raise ValueError("QWEN_MTP_PROMPT_LEN must be a multiple of 32 and at least 32")
    if draft_tokens <= 0:
        raise ValueError("QWEN_MTP_DRAFT_TOKENS must be positive")
    if verify_blocks <= timing_warmup_blocks:
        raise ValueError("QWEN_MTP_VERIFY_BLOCKS must exceed QWEN_MTP_TIMING_WARMUP_BLOCKS")

    load_start = time.perf_counter()
    model = Qwen36Model.from_pretrained(
        device,
        max_batch_size=1,
        max_seq_len=prompt_len + decode_steps + 32,
        n_layers=target_layers,
        enable_mtp=True,
    )
    model_load_s = time.perf_counter() - load_start
    assert model.mtp is not None

    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompt_ids = _get_prompt(prompt_len, tokenizer)[:, :prompt_len]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = getattr(model.args.hf_config, "pad_token_id", None)
    if pad_id is None:
        pad_id = 0
    shifted_ids = torch.full_like(prompt_ids, int(pad_id))
    shifted_ids[:, :-1] = prompt_ids[:, 1:]

    # Compile both prefill graphs once.  MTP prefill consumes the raw target
    # hidden states and only keeps its dedicated attention cache.
    warm_logits, warm_hidden = model.prefill(prompt_ids, return_hidden_states=True)
    model.mtp.prefill(warm_hidden, shifted_ids)
    ttnn.synchronize_device(device)
    ttnn.deallocate(warm_logits)
    ttnn.deallocate(warm_hidden)
    gc.collect()
    _log_l1(device, "after_warmup_release")

    (prefill_logits, hidden_states), target_prefill_ms = _timed(
        device, lambda: model.prefill(prompt_ids, return_hidden_states=True)
    )
    _log_l1(device, "after_target_prefill")
    _, mtp_prefill_ms = _timed(device, lambda: model.mtp.prefill(hidden_states, shifted_ids))
    _log_l1(device, "after_mtp_prefill")

    current_token = _argmax_token(prefill_logits)
    position = prompt_len
    # A decode-shaped interleaved tensor is tile-padded in the sequence
    # dimension.  Keep the bootstrap row in DRAM so it cannot occupy the
    # target decoder's core-0 L1 workspace, then release both prefill tensors
    # as soon as the MTP has consumed them.
    bootstrap_row = hidden_states[:, -1:, :]
    root_hidden = ttnn.clone(bootstrap_row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(bootstrap_row)
    ttnn.deallocate(hidden_states)
    _log_l1(device, "after_bootstrap_release_before_gc")
    gc.collect()
    _log_l1(device, "before_target_decode")
    if os.environ.get("QWEN_MTP_DEBUG_MEMORY") == "1":
        attention = model.mtp.layer.attention
        key_cache = model.mtp.k_cache if attention.use_paged_attention else attention.past_key
        value_cache = model.mtp.v_cache if attention.use_paged_attention else attention.past_value
        print(
            "MTP_CACHE " f"key={key_cache.memory_config()} " f"value={value_cache.memory_config()}",
            flush=True,
        )

    target_step_times_ms = []
    target_block_times_ms = []
    mtp_step_times_ms = []
    mtp_block_times_ms = []
    mtp_depth_times_ms = [[] for _ in range(draft_tokens)]
    replay_times_ms = []
    prefix_lengths = []
    depth_equal = [0] * draft_tokens
    depth_conditional_accepts = [0] * draft_tokens
    depth_conditional_verifies = [0] * draft_tokens
    generated = []
    bootstrap_mtp_ms = None
    for block in range(verify_blocks):
        cache_checkpoint = model.mtp.checkpoint_state()
        draft_block, draft_times = _draft_block(model, root_hidden, current_token, position, draft_tokens)
        if bootstrap_mtp_ms is None:
            bootstrap_mtp_ms = sum(draft_times)

        target_block = []
        target_times = []
        target_input = current_token
        next_root_hidden = None
        for depth in range(draft_tokens):
            (target_logits, target_hidden), target_ms = _timed(
                device,
                lambda target_input=target_input, depth=depth: model.decode(
                    torch.tensor([[target_input]], dtype=torch.long),
                    position + depth,
                    return_hidden_states=True,
                ),
            )
            target_token = _argmax_token(target_logits)
            if next_root_hidden is not None:
                ttnn.deallocate(next_root_hidden)
            next_root_hidden = _to_dram_hidden(target_hidden)
            target_block.append(target_token)
            target_times.append(target_ms)
            target_input = target_token

        prefix_len = _accepted_prefix_length(draft_block, target_block)
        replay_ms = 0.0
        if prefix_len < draft_tokens:
            model.mtp.restore_state(cache_checkpoint)
            replay_ms = sum(
                _replay_target_block(
                    model,
                    root_hidden,
                    current_token,
                    position,
                    target_block,
                )
            )

        if block >= timing_warmup_blocks:
            prefix_lengths.append(prefix_len)
            target_step_times_ms.extend(target_times)
            target_block_times_ms.append(sum(target_times))
            mtp_step_times_ms.extend(draft_times)
            mtp_block_times_ms.append(sum(draft_times))
            replay_times_ms.append(replay_ms)
            eligible = True
            for depth, (draft_token, target_token) in enumerate(zip(draft_block, target_block)):
                is_equal = draft_token == target_token
                depth_equal[depth] += int(is_equal)
                if eligible:
                    depth_conditional_verifies[depth] += 1
                    depth_conditional_accepts[depth] += int(is_equal)
                eligible = eligible and is_equal
                mtp_depth_times_ms[depth].append(draft_times[depth])

        generated.extend(target_block)
        ttnn.deallocate(root_hidden)
        root_hidden = next_root_hidden
        current_token = target_block[-1]
        position += draft_tokens

    ttnn.deallocate(root_hidden)

    measured_blocks = len(prefix_lengths)
    target_step_ms = sum(target_step_times_ms) / len(target_step_times_ms)
    target_block_ms = sum(target_block_times_ms) / measured_blocks
    mtp_step_ms = sum(mtp_step_times_ms) / len(mtp_step_times_ms)
    mtp_block_ms = sum(mtp_block_times_ms) / measured_blocks
    replay_ms = sum(replay_times_ms) / measured_blocks
    average_accepted_prefix = sum(prefix_lengths) / measured_blocks
    expected_output_tokens = 1.0 + average_accepted_prefix
    full_block_accept_rate = sum(prefix == draft_tokens for prefix in prefix_lengths) / measured_blocks
    target_tsu = 1000.0 / target_step_ms
    proposer_tsu = 1000.0 / mtp_step_ms
    teacher_forced_tsu = 1000.0 * draft_tokens / (target_block_ms + mtp_block_ms + replay_ms)
    sequential_verifier_spec_tsu = 1000.0 * expected_output_tokens / (target_block_ms + mtp_block_ms)
    optimistic_packed_tsu = 1000.0 * expected_output_tokens / (target_step_ms + mtp_block_ms)
    break_even_verify_ms = expected_output_tokens * target_step_ms - mtp_block_ms
    verify_budget_20_tsu_ms = 1000.0 * expected_output_tokens / 20.0 - mtp_block_ms
    verify_budget_100_tsu_ms = 1000.0 * expected_output_tokens / 100.0 - mtp_block_ms

    depth_metrics = []
    for depth in range(draft_tokens):
        conditional_verifies = depth_conditional_verifies[depth]
        depth_metrics.append(
            {
                "depth": depth + 1,
                "unconditional_equal_rate": round(depth_equal[depth] / measured_blocks, 6),
                "conditional_accepts": depth_conditional_accepts[depth],
                "conditional_verifies": conditional_verifies,
                "conditional_accept_rate": (
                    round(depth_conditional_accepts[depth] / conditional_verifies, 6) if conditional_verifies else None
                ),
                "mtp_step_ms": round(sum(mtp_depth_times_ms[depth]) / measured_blocks, 4),
            }
        )

    result = {
        "prototype": f"qwen38_mtp_k{draft_tokens}_recurrent_teacher_forced",
        "device_count": 1,
        "prompt_tokens": prompt_len,
        "draft_tokens": draft_tokens,
        "verify_blocks": verify_blocks,
        "timing_warmup_blocks": timing_warmup_blocks,
        "timed_verify_blocks": measured_blocks,
        "decode_steps": decode_steps,
        "timed_decode_steps": len(target_step_times_ms),
        "model_load_s": round(model_load_s, 4),
        "target_prefill_ms": round(target_prefill_ms, 4),
        "target_prefill_tsu": round(prompt_len * 1000.0 / target_prefill_ms, 4),
        "mtp_cache_prime_ms": round(mtp_prefill_ms, 4),
        "mtp_cache_prime_tsu": round(prompt_len * 1000.0 / mtp_prefill_ms, 4),
        "mtp_bootstrap_ms_including_compile": round(bootstrap_mtp_ms, 4),
        "target_decode_ms": round(target_step_ms, 4),
        "target_decode_p50_ms": round(_median(target_step_times_ms), 4),
        "target_decode_min_ms": round(min(target_step_times_ms), 4),
        "target_decode_max_ms": round(max(target_step_times_ms), 4),
        "target_decode_tsu": round(target_tsu, 4),
        "target_sequential_verify_block_ms": round(target_block_ms, 4),
        "mtp_proposer_ms": round(mtp_step_ms, 4),
        "mtp_proposer_p50_ms": round(_median(mtp_step_times_ms), 4),
        "mtp_proposer_min_ms": round(min(mtp_step_times_ms), 4),
        "mtp_proposer_max_ms": round(max(mtp_step_times_ms), 4),
        "mtp_proposer_tsu": round(proposer_tsu, 4),
        "mtp_draft_block_ms": round(mtp_block_ms, 4),
        "mtp_repair_replay_ms_per_block": round(replay_ms, 4),
        "teacher_forced_target_plus_mtp_tsu": round(teacher_forced_tsu, 4),
        "sequential_verifier_spec_tsu": round(sequential_verifier_spec_tsu, 4),
        "accepted_prefix_lengths": prefix_lengths,
        "average_accepted_prefix": round(average_accepted_prefix, 6),
        "expected_output_tokens_per_cycle": round(expected_output_tokens, 6),
        "full_block_accept_rate": round(full_block_accept_rate, 6),
        "depth_metrics": depth_metrics,
        "optimistic_same_cost_packed_tsu": round(optimistic_packed_tsu, 4),
        "break_even_packed_verify_ms": round(break_even_verify_ms, 4),
        "packed_verify_budget_for_20_tsu_ms": round(verify_budget_20_tsu_ms, 4),
        "packed_verify_budget_for_100_tsu_ms": round(verify_budget_100_tsu_ms, 4),
        "generated_token_ids": generated,
    }
    result_json = json.dumps(result, sort_keys=True)
    logger.info(f"MTP_RESULT_JSON={result_json}")
    print(f"MTP_RESULT_JSON={result_json}", flush=True)

    output_path = os.environ.get("QWEN_MTP_RESULT_PATH")
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")

    assert all(0 <= prefix <= draft_tokens for prefix in prefix_lengths)
    assert len(generated) == decode_steps
    assert len(set(generated)) > 1, f"degenerate target generation: {generated}"
