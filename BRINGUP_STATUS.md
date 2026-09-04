# Qwen3.8-27B Blackhole bring-up status

Updated: 2026-08-14 UTC

## Current hypothesis

Qwen3.8-27B is checkpoint-compatible with the existing `models/demos/blackhole/qwen36` Qwen3.6-27B path. The live configs are identical except for producer `transformers_version` (`5.8.0.dev0` versus `4.57.1`). All 1,199 indexed tensors match in name, BF16 dtype, and shape; the 851 text tensors also match a Transformers 5.12.1 meta-model exactly. Qwen3.8 changes weight values, shard packing, seven reserved audio special tokens, and the chat template, but not the TT text architecture.

## Completed

- Confirmed branch `sjett/qwen38-27b-bringup-20260814` starts clean at `9a04829078`, matching `origin/main`.
- Searched for repository guidance; no `AGENTS.md` exists in the repository root or its immediate parent.
- Inventoried the shared Qwen3.5/Qwen3.6 implementation and tests under `models/demos/blackhole/qwen36`.
- Inspected `models/tt_transformers/model_params/Qwen3.6-27B/config.json`, `tt/model_config.py`, and `tt/weight_mapping.py`.
- Read the live Qwen3.8 config and safetensors index from Hugging Face. The checkpoint has 1,199 indexed tensors in 18 shards and reports 55,562,855,904 bytes of tensor data.
- Compared every Qwen3.8 safetensors header with live Qwen3.6: zero key/dtype/shape differences; Qwen3.6 uses 15 shards while Qwen3.8 uses 18.
- Downloaded revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` once to `/home/sjett/hf-cache/qwen38-27b` (52 GiB on disk).
- Loaded the local config and tokenizer with Transformers 5.12.1. A meta `Qwen3_5ForCausalLM` has 851 text tensors / 26,895,998,464 parameters, with zero key or shape differences from the real text checkpoint.
- Loaded and remapped real BF16 layer-0 GDN weights. The combined QKV is `(10240, 5120)` and conv streams split to Q `(2048,1,4)`, K `(2048,1,4)`, V `(6144,1,4)` as required.
- Exercised `Qwen36ModelArgs.load_state_dict()` on the complete real checkpoint: 851 HF text tensors became 947 internal tensors (48 fused conv weights each expand from one tensor to Q/K/V); layers 0, 3, and 63 have the expected real shapes.
- Generated an eager Transformers reference for `The capital of France is`: input IDs `[760,6511,314,9338,369]`, greedy IDs `[11751,13,198,760,6511,314,9564,369]`, continuation ` Paris.\nThe capital of Germany is` (load 0.33 s, generation 6.67 s).
- Added Qwen3.8 local model params, model/tokenizer recognition, target aliases, compatibility tests, and README coverage. CPU compatibility tests: 3 passed.
- Focused CPU tests now total 10 passed: 3 Qwen3.8 compatibility, 2 substate, and 5 centralized target-resolver tests. Qwen3.8 aliases resolve the existing 27B P150x4 target.
- Built the current checkout successfully in Release mode with `./build_metal.sh --release --enable-ccache --build-dir build_qwen38_release --install-prefix build_qwen38_install`; the matching extension is installed at `ttnn/ttnn/_ttnn.so`.
- Committed the static checkpoint-support milestone locally as `5f9934aef8` (`Add Qwen3.8-27B checkpoint support`); nothing was pushed.
- Ran the real-weight layer-0 decode MLP on all eight Blackhole devices with TP=8 and `FABRIC_1D`: PCC `0.9992268419104252` against the Torch SwiGLU oracle, 1 passed / 1 deselected in 29.99 s. The devices closed cleanly.
- Ran the real-weight layer-0 prefill MLP at sequence length 2048 on TP=8: PCC `0.9992292473075579` against Torch, 1 passed / 1 deselected in 23.05 s. A repeated preflight again found no handles, and teardown completed cleanly.
- Ran real-weight layer-3 full-attention prefill at sequence length 64 on TP=8: PCC `0.9997408725301266` against the full Torch causal-GQA/partial-RoPE/qk-norm/gate oracle, 1 passed / 6 deselected in 27.19 s. This exercises the 4-KV-head replication across eight devices.
- Ran real-weight layer-3 full-attention eager decode at batch 8 on TP=8. Position-0 per-user PCC ranged from `0.99993` to `0.99997` against Torch; the position-1 two-key cache step was finite and nonzero. 1 passed / 6 deselected in 28.11 s.
- Ran real-weight layer-0 Gated DeltaNet eager decode at batch 1 on TP=8. Position-0 PCC was `0.99990` against the hand-written Torch recurrent/gating/conv oracle; the second recurrent step was finite and nonzero. 1 passed / 12 deselected in 38.69 s.
- Ran real-weight layer-0 Gated DeltaNet prefill for 128 tokens on TP=8. The intended fused phased chunk kernel (`chunk_size=32`) matched 128 eager decode steps at PCC `0.9999650181225375`; 1 passed / 12 deselected in 34.76 s.
- Ran an integrated real-weight 8-layer TP=8 model contract (layers 0-7 include 6 GDN + 2 full-attention blocks, embedding, final norm, and vocabulary-sharded LM head). Paged-vs-bespoke logits PCC was `0.9998349047` at prefill and `0.9999122435`, `0.9999161966`, `0.9999264768` over three decode steps; masked-bucket prefill PCC was `0.9998309836`. 1 passed / 13 deselected in 115.36 s, no trace captured.
- Ran the complete real-weight 64-layer model through eager text prefill and eight greedy decode steps on all eight Blackhole cards. It produced `The capital of France is Paris.\nThe capital of Germany is`, exactly matching the Transformers reference text. 1 passed in 254.78 s including first-time cache creation; all devices closed cleanly.
- Strengthened `test_generate_tp_stateful` to assert the complete Qwen3.8 eight-token Transformers oracle while preserving the existing first-token contract for Qwen3.5/3.6. The cached full-model TP=8 rerun produced exact IDs `[11751,13,198,760,6511,314,9564,369]`, passed in 100.40 s with 873/873 JIT cache hits, and closed all devices cleanly.
- Final focused CPU/static rerun: 10 passed in 0.99 s; Black check passed for both touched test files, Qwen3.8 config JSON validation passed, and `git diff --check` passed.
- Hardware preflight at 15:16 UTC: 8 PCI functions, 8 character device nodes, and no `fuser` handles. Other users have old shells/wait loops but no device holder.

## Commands and results

```text
git status --short --branch
## sjett/qwen38-27b-bringup-20260814...origin/main

curl -fsSL https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json
# Qwen3_5ForConditionalGeneration / qwen3_5; text dimensions recorded above.

curl -fsSL https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/model.safetensors.index.json
# metadata.total_size=55562855904, weight_count=1199, shards=18.

python -m pytest --noconftest -q models/demos/blackhole/qwen36/tests/test_qwen38_compatibility.py
# 3 passed in 1.00s

python -m pytest --noconftest -q \
  models/demos/blackhole/qwen36/tests/test_qwen38_compatibility.py \
  models/demos/blackhole/qwen36/tests/unit/test_substate.py \
  tools/tests/github_scripts/test_model_targets_resolver.py
# 10 passed in 1.00s

./build_metal.sh --release --enable-ccache \
  --build-dir build_qwen38_release --install-prefix build_qwen38_install
# Completed all 1,404 build/install steps; exit 0.

TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MESH_DEVICE=P150x8 \
HF_MODEL=/home/sjett/hf-cache/qwen38-27b \
python -m pytest models/demos/blackhole/qwen36/tests/test_mlp_tp.py \
  -v -s -k 'test_mlp_tp and not prefill'
# 1 passed, 1 deselected in 29.99s; MLP TP PCC=0.9992268419104252.
# Full output: /home/sjett/qwen38-bringup-logs/mlp_tp8_decode.log

TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MESH_DEVICE=P150x8 \
HF_MODEL=/home/sjett/hf-cache/qwen38-27b \
python -m pytest models/demos/blackhole/qwen36/tests/test_mlp_tp.py \
  -v -s -k test_mlp_tp_prefill
# 1 passed, 1 deselected in 23.05s; prefill PCC=0.9992292473075579 at T=2048.
# Full output: /home/sjett/qwen38-bringup-logs/mlp_tp8_prefill.log

TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MESH_DEVICE=P150x8 \
HF_MODEL=/home/sjett/hf-cache/qwen38-27b \
python -m pytest models/demos/blackhole/qwen36/tests/test_attention_tp.py \
  -v -s -k test_attention_tp_prefill
# 1 passed, 6 deselected in 27.19s; full-attention prefill PCC=0.9997408725301266.
# Full output: /home/sjett/qwen38-bringup-logs/attention_tp8_prefill.log

TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MESH_DEVICE=P150x8 \
HF_MODEL=/home/sjett/hf-cache/qwen38-27b \
python -m pytest models/demos/blackhole/qwen36/tests/test_attention_tp.py \
  -v -s -k 'test_attention_tp and B8 and not paged and not prefill and not qknorm'
# 1 passed, 6 deselected in 28.11s; pos0 per-user PCC min=0.99993, max=0.99997.
# Full output: /home/sjett/qwen38-bringup-logs/attention_tp8_decode_b8.log

TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MESH_DEVICE=P150x8 \
HF_MODEL=/home/sjett/hf-cache/qwen38-27b \
python -m pytest models/demos/blackhole/qwen36/tests/test_gdn_tp.py \
  -v -s -k 'test_gdn_tp and B1 and not prefill and not recurrence and not peruser and not write_slot and not batched and not fused'
# 1 passed, 12 deselected in 38.69s; GDN position-0 PCC=0.99990.
# Full output: /home/sjett/qwen38-bringup-logs/gdn_tp8_decode_b1.log

TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MESH_DEVICE=P150x8 \
HF_MODEL=/home/sjett/hf-cache/qwen38-27b \
python -m pytest models/demos/blackhole/qwen36/tests/test_gdn_tp.py \
  -v -s --timeout=600 -k 'test_gdn_tp_prefill and not batched and not fused'
# 1 passed, 12 deselected in 34.76s; prefill-vs-decode PCC=0.9999650181225375.
# Full output: /home/sjett/qwen38-bringup-logs/gdn_tp8_prefill.log

TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MESH_DEVICE=P150x8 \
HF_MODEL=/home/sjett/hf-cache/qwen38-27b \
python -m pytest models/demos/blackhole/qwen36/tests/test_model_tp.py \
  -v -s --timeout=1500 -k test_model_tp_contract
# 1 passed, 13 deselected in 115.36s; per-step and masked-bucket PCCs recorded above.
# Full output: /home/sjett/qwen38-bringup-logs/model_tp8_contract_8layer.log

TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MESH_DEVICE=P150x8 \
HF_MODEL=/home/sjett/hf-cache/qwen38-27b \
python -m pytest models/demos/blackhole/qwen36/tests/test_generate_tp.py \
  -v -s --timeout=3300
# 1 passed in 254.78s; exact generated text matches the Transformers reference.
# Full output: /home/sjett/qwen38-bringup-logs/generate_tp8_full64.log

# Strengthened exact-token rerun using the same command and completed cache:
# 1 passed in 100.40s; IDs [11751,13,198,760,6511,314,9564,369].
# Full output: /home/sjett/qwen38-bringup-logs/generate_tp8_full64_exact_rerun.log

HF_MODEL=/home/sjett/hf-cache/qwen38-27b Qwen36ModelArgs(mesh_device=None, max_seq_len=256)
# dim=5120, hidden_dim=17408, layers=64 (48 GDN + 16 full), Q/KV heads=24/4,
# linear key/value dims=2048/6144, tokenizer prompt ids=[760,6511,314,9338,369].
```

## Failures / blockers

- The first current-source build configure failed because submodules were absent. `git submodule update --init --recursive` resolved it; the isolated Release rebuild then completed successfully.
- One ad-hoc full-loader assertion incorrectly expected the pre-remap layer-key count (848); observed 944 is correct because 48 conv tensors each split into three (+96). The corrected rerun passed.

## Next step

Commit the hardware-validation milestone locally. No unresolved text-only bring-up blocker remains; traced serving and vision are follow-on coverage rather than prerequisites for this milestone.
