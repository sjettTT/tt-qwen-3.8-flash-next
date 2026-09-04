# Qwen3.8-Flash-Next on Blackhole

A TTNN port of Qwen3.8-Flash-Next (48 hybrid layers: gated delta-net attention, sparse attention with an indexer,
a 256-expert MoE routed from a BF4 corpus; one multi-token-prediction layer) served from one 1x4 mesh of Blackhole
chips: an OpenAI-compatible chat server with chunked prefill, contexts to 262,144 tokens, greedy or sampled decode,
thinking and tool calls in the Qwen chat template.

| hardware | profile | status |
|---|---|---|
| QuietBox, 4x p150b (fw 19.4.1.0) | `tt-quietbox` | verified 2026-09-04: startup acceptance 96/96 tokens against the CPU, 19.6 tokens/s at 32k context |
| QuietBox 2, 2x p300c (4 dies) | `qb2` | **untested**: designed from the p300 ring topology, never run |
| 4x p300 host (8 dies) | `qb2 --instance 0\|1` | **untested**: two independent 1x4 instances |

Every number below was measured on 4x p150b; the lab's eight-chip hosts were used the same way, one 1x4 mesh at a time.

## Performance (4x p150, measured 2026-09-04)

| path | measured | notes |
|---|---|---|
| prompt prefill | 300 tok/s (and climbing) | 32-token chunk trace, 3.0-3.5 ms per prompt token, flat from 2k to 261k tokens; 128- and 256-token chunks are in progress |
| decode, one stream | 19.9 tok/s | position-generic traced decode, 50 ms per token, flat with depth |
| decode with MTP (`--mtp 4`) | 37 tok/s aggregate, 55 tok/s on structured output | speculative drafting with exact acceptance: the committed stream equals greedy decode |
| contexts | 32k, 64k, 128k, 256k | 256k is single-user; MTP fits at 32k, 64k and 128k |
| correctness | bitwise repeatable; 96/96 greedy token match against the CPU reference on the acceptance prompt | chunked prefill is tolerance-class against the CPU reference on all 48 layers |

## 1. What you need

- A Blackhole QuietBox with tt-kmd and the firmware bundle the chips shipped with (19.4.1.0 or later).
- A `tt-metal` checkout, built, with its `python_env`.  This model needs runtime fixes that were developed on
  tt-metal main `d04395ed86` (2026-08-29) and are not yet on main; the launcher lists the ones your checkout lacks:
  - `Fix empty-rank moe compute metadata ownership` (moe_compute tilize writer; routed experts at batch 1)
  - `skip idle-expert combine sync in moe_compute B=1` (-1.2 ms per token; optional)
  - `Add exact TP4 TTNN component path` (`moe_compute(..., local_combine=True)`, DRAM-bank-to-worker query)
  - `Fix fused MoE source buffer double counting`
  - all-gather and fabric guards: `Guard all-gather scatter state initialization`, `Preserve ring connections in
    all-gather endpoint guard`, `Fix all-gather endpoint no-target connection access`, `Clear fabric router packet
    tags on teardown`
  - the four `#23023` commits (`Bind BF4 cache loads to verified file descriptors`, `Reject lexical aliases for
    descriptor loads`, `Fail closed BF4 cache shape and cleanup`, `Bind BF4 tensorbin payload and exact types`):
    `ttnn.load_tensor` on `/proc/self/fd` paths, used by `ttnn/bf4.py`

  Until these are on tt-metal main a public PR of this directory must carry them (they live outside this directory,
  under `ttnn/` and `tt_metal/`); the tree here has no other runtime dependency: no archive, no digest, no seal.
- This repository checked out next to it (the model imports as `models.demos.blackhole.qwen38_flash_next`).
- The checkpoint: 360 GB of safetensors (131 shards) plus the tokenizer and chat template.
- Disk for the caches: about 23 GB for the 32k context and 10 GB for each other allocated context (converted
  non-expert weights, the model I/O cache), the BF4 expert scratch, and the JIT kernel cache (about 1.3 GB).
- Host memory: the CPU preparation reads the checkpoint once; 64 GB is comfortable.  The optional CPU reference
  (`tools/run_full_cpu_oracle.py`) needs 170-240 GB and is not a user step.

## 2. The checkpoint

Download `Qwen/Qwen3.8-Flash-Next` from ModelScope at the revision this port was verified against
(`checkpoint.PINNED_CHECKPOINT_REVISION`; the index, file-manifest and tensor-manifest digests are pinned next to it,
131 shards, 360 GB):

    pip install modelscope
    modelscope download --model Qwen/Qwen3.8-Flash-Next \
        --revision f5d08274bafd880402bd16f5e3e6c514136ec06c --local_dir /data/Qwen3.8-Flash-Next

Optionally hash the download and compare it with the repository's file listing (the JSON the ModelScope API returns
for the revision):

    python -m models.demos.blackhole.qwen38_flash_next.tools.verify_checkpoint_files \
        --checkpoint /data/Qwen3.8-Flash-Next --modelscope-tree <file listing JSON> --output verify.json

The server checks the checkpoint itself at startup (every shard's header, the pinned digests) and refuses a different
one.

The tokenizer, `chat_template.jinja` and `config.json` are pinned by digest in `chat.py` and `config.py`; a checkpoint
with different files is refused with the digests printed.  `tools/checkpoint_budget.py` prints the per-device
residency budget for a checkpoint; `tools/safetensors_metadata.py` lists tensors.

## 3. Start the server (QuietBox)

    export TT_METAL_HOME=/path/to/tt-metal
    tools/run_qwen38_chat_server.sh --profile tt-quietbox \
        --checkpoint /data/Qwen3.8-Flash-Next --cache-root /data/qwen38-cache

The first start builds the caches (the weights are converted from the checkpoint, a few minutes; the JIT kernel
cache fills during the warm pass, about two more minutes).  Warm starts reach `READY` in about five minutes: the
weights load, the decode traces and the prefill chunk trace are captured, the acceptance prompts (if given) replay
against the CPU records, then the server listens.  The run directory (`<cache-root>/runs/<stamp>/`) holds `READY`,
`phase-markers.jsonl`, `requests.jsonl`, `server.log` and, at shutdown, `result.json` and `STOPPED`.

Options: `--allocated-context 32768|65536|131072|262144` selects the resident build (KV caches, RoPE tables and the
context limit; the limit is the context minus 64 for the consumed EOS step), `--port`, `--host` (the QuietBox
profiles serve the LAN; a lab profile needs `--allow-lan`), `--acceptance-prompts DIR --require-json-96` to refuse
serving unless the startup replay matches the CPU records, `--serve-seconds N` to stop after N seconds,
`--validate-only` to run the checks and the CPU preparation without opening the mesh.

What the launcher does not do: no device locks, no runtime archives or digests, no evidence conventions.  It exports
the QuietBox mesh graph descriptor (`tools/qb_p150_x4_1x4_line_mesh_graph_descriptor.textproto`: the four chips'
ethernet ring opened as one 1x4 line), the device set, the cache and log roots, and passes the identity of the
`ttnn` it imports to the server, which records it.

### Runtime admission (pending)

The server's startup admission (`tools/runtime_admission.py`, `tools/live_decode_diagnostic.py`, `diagnostic_bf4.py`)
still checks the runtime against the identities of the campaign's pinned build and binds the BF4 expert corpus by
the paths it was produced under.  Until that admission is rewritten to accept a runtime built from a tt-metal
checkout and a corpus produced by `tools/stage_full_bf4_cpu.py` on the same machine, a checkout build is refused at
`prepare_live_decode_diagnostic` with the differing identity printed.  `tools/release/manifest.json` lists these three
files as pending; everything else in the public tree is free of the campaign's hosts, paths and seals.

## 4. Talk to it

OpenAI-compatible HTTP on the port you chose:

    curl -s http://<quietbox>:8000/health
    curl -s http://<quietbox>:8000/v1/models
    curl -s http://<quietbox>:8000/v1/chat/completions -H 'content-type: application/json' -d '{
      "model": "Qwen/Qwen3.8-Flash-Next",
      "messages": [{"role": "user", "content": "Why does ice float?"}],
      "max_tokens": 256, "stream": true}'

`POST /v1/chat/completions` (streaming or one document), `GET /v1/models`, `GET /health` (context limit, sampling
mode, free DRAM after the captures).  Requests: `messages`, `max_tokens` or `max_completion_tokens` (default and limit: the remaining context, the context limit less the prompt), `stream`,
`stop`, `tools` / `tool_choice` (OpenAI shape; `tool_calls` finish reason), `enable_thinking` (default true;
reasoning streams as `reasoning_content`), `reasoning_effort`, `thinking_budget`, `ignore_eos`, `seed`,
`temperature` / `top_p` / `top_k` / `min_p` / `logprobs` (a server started with `--sampling`; the default server is
greedy and accepts and ignores them).  One request decodes at a time; up to four wait in the queue
(`queue_wait_seconds` in `usage`), the fifth gets HTTP 503.  A prompt over the context limit gets HTTP 400
`context_length_exceeded`.

The command-line client:

    python -m models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_cli --url http://<quietbox>:8000/v1 --thinking --tools

## 5. Long context and MTP

- `--allocated-context 65536` serves 65,472 tokens; `131072` and `262144` the same minus 64.  Prompt prefill runs
  through the chunk trace at about 3.2-3.5 ms per token (a 40k prompt: 125 s to the first token; 200k: 671 s), decode
  stays at 17-19 tokens/s to 256k.  Each context has its own cache set under `--cache-root`; 256k leaves about
  750 MB per device free.
- MTP drafting (`--mtp 3|4`, 31-37 tokens/s on the lab mesh) lives on the `q38-serve-mtp` branch and is merged into
  this server separately; `--mtp` on a server without the path is refused by the launcher.

## 6. QuietBox 2 and p300 hosts (untested)

A p300 card is two Blackhole dies joined on the card; a QuietBox 2 (2x p300c) has four dies in one ring (the two
on-card links and the two Warp400 links), so it is one 1x4 instance:

    tools/run_qwen38_chat_server.sh --profile qb2 --checkpoint ... --cache-root ...

The profile exports `tools/qb2_p300_1x4_line_mesh_graph_descriptor.textproto` (a 1x4 LINE over three of the four ring
links, two channels per link as in tt-metal's `p300_x2` descriptor); tt-metal classifies a p300 cluster that is not
exactly two or four dies as CUSTOM and refuses to open without a descriptor, so the launcher always exports one.  The
route is unpinned: the first start derives the ring-walk order from the cluster descriptor, prints it and stops; pin
it in `tools/hardware_profiles.py` (`route`, `route_nodes`) and start again.  On a host with four p300 cards (eight
dies, the `p150_x8` [2,4] mesh) `--instance 0` uses nodes 0-3 and `--instance 1` nodes 4-7, each with its own caches
and run directory; the two servers need different ports.  Nothing here has run on p300 hardware.

## 7. Layout

    chat.py checkpoint.py config.py reference.py   the checkpoint, the chat template, the torch reference model
    tt/                                            torch reference components (the CPU oracle every test compares against)
    ttnn/                                          the device model: builder, layers, GDN, QSA, MoE/BF4, embedding, sampling, MTP
    tools/qwen38_chat_server.py                    the HTTP server; qwen38_chat_session.py the traced decode chain;
                                                   qwen38_chat_protocol.py the request/reply protocol; qwen38_sampling_step.py
    tools/hardware_profiles.py                     the profiles (tt-quietbox, tt-quietbox-2); resident_decode.py the chain's
                                                   fixed points; evidence_records.py the run records
    tools/run_qwen38_chat_server.sh                the launcher; qwen38_chat_cli.py the client
    tools/stage_full_bf4_cpu.py verify_full_bf4_cpu.py bind_full_bf4_corpus_cpu.py probe_full_bf4_binding_cpu.py
                                                   the BF4 expert corpus (produce, verify, bind, probe)
    tools/prewarm_ple_table.py verify_checkpoint_files.py checkpoint_budget.py safetensors_metadata.py
    tools/qb_mesh_smoke.py                         open the mesh and check the route without the model
    tools/release/                                 the public-tree manifest and exporter (section 8)
    tests/                                         no-device tests (set QWEN38_CHECKPOINT for the checkpoint-reading ones)
    tools/dev/ tests/dev/                          the lab tooling: launchers, micro-tests, discriminators, profilers, gates

Run the tests from the repository root with a python that imports `ttnn`:

    QWEN38_CHECKPOINT=/data/Qwen3.8-Flash-Next python -m pytest models/demos/blackhole/qwen38_flash_next/tests

## 8. The public tree

`tools/release/manifest.json` lists every public file, the three pending files and the forbidden token patterns
(lab host names, user paths, lab addresses, staging roots, device locks, runtime seals, board identities).
`tools/release/export_public_tree.py --check` scans the tree; `--out DIR` copies the public files; `--git-tree`
writes a git tree object of the repository with only the public files under this directory (no branch); `--with-dev`
adds `tools/dev` and `tests/dev`.  A forbidden token in a public file fails the export with every hit listed; nothing
is rewritten.  `tests/test_release_export_static.py` runs the scan.
