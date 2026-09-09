# Qwen3.8-Flash-Next on Blackhole

A TTNN port of Qwen3.8-Flash-Next (48 hybrid layers: gated delta-net attention, sparse attention with an indexer,
a 256-expert MoE routed from a BF4 corpus; one multi-token-prediction layer) served from one 1x4 mesh of Blackhole
chips: an OpenAI-compatible chat server with chunked prefill, contexts to 262,144 tokens, greedy or sampled decode,
thinking and tool calls in the Qwen chat template.

This repository is a tt-metal checkout that carries the runtime fixes the model needs (section 1).  You clone it,
build it the standard way, download the checkpoint and start the server; the first start converts the weights into
its caches.  Nothing else is required: no prebuilt archive, no pinned binary, no host-specific configuration.

| hardware | profile | status |
|---|---|---|
| QuietBox, 4x p150c (fw 19.4.1.0) | `tt-quietbox` | verified 2026-09-04: startup acceptance 96/96 tokens against the CPU, 19.6 tokens/s at 32k context; `docs/PROOFS.md`: the fresh-clone proof of 2026-09-06 |
| Blackhole LoudBox, 4x p150 (one line) | `bh-loudbox` | section 9: what was verified and where |
| QuietBox 2, 2x p300c (4 dies) | `qb2` | **untested**: designed from the p300 ring topology, should run fine |

Every number below was measured on 4x p150.

#### IMPORTANT NOTE: 
This was implemented with the intention of the n-gram model residing in system memory. I have not tested performance with disk. If you are not using a QB, please make sure you have room to fit the 51.2B parameters (they take up >100GB at BF16).

## Performance (4x p150, measured 2026-09-04)

| path | measured | notes |
|---|---|---|
| prompt prefill | 300 tok/s (and climbing); 650 tok/s with `--long-chunks`; 1,150 tok/s with `--prefill-slab 2048` | 32-token chunk trace, 3.0-3.5 ms per prompt token, flat from 2k to 261k tokens; 128-token chunks at 1.45-1.56 ms per prompt token (`--long-chunks`); 2,048-row slabs at 0.87-1.08 ms per prompt token (`--prefill-slab 2048`, 2026-09-09, tolerance class: see `docs/PREFILL.md`) |
| decode, one stream | 19.9 tok/s | position-generic traced decode, 50 ms per token, flat with depth |
| decode with MTP (`--mtp 4`) | 37 tok/s aggregate, 55 tok/s on structured output | speculative drafting with exact acceptance: the committed stream leaves the CPU reference at the same token as greedy decode on 8 of the 12 acceptance prompts and at a different token on the other 4 (section 6, `docs/NUMERICS.md`) |
| contexts | 32k, 64k, 128k, 256k | 256k is single-user; MTP fits at 32k, 64k and 128k |
| correctness | bitwise repeatable; 96/96 greedy token match against the CPU reference on the acceptance prompt | chunked prefill is tolerance-class against the CPU reference on all 48 layers |

## 1. What you need

- Four Blackhole chips in one host as one 1x4 mesh: a QuietBox (4x p150c in an ethernet ring), a Blackhole LoudBox
  (4x p150 in an ethernet line), or four chips of a larger host (`--devices`).  tt-kmd and the firmware bundle the
  chips shipped with (19.4.1.0 or later).
- This repository, built (section 2).  It is tt-metal main `28238f903b` (2026-09-06; the previous base `d04395ed86`
  of 2026-08-29 was merged forward, the device token streams are bitwise the same) plus the runtime fixes the model
  needs, which are not on main yet and have no upstream equivalent:
  - `Fix empty-rank moe compute metadata ownership` (moe_compute tilize writer; routed experts at batch 1)
  - `skip idle-expert combine sync in moe_compute B=1` (-1.2 ms per token)
  - `Add exact TP4 TTNN component path` (`moe_compute(..., local_combine=True)`, DRAM-bank-to-worker query)
  - `Fix fused MoE source buffer double counting`
  - all-gather and fabric guards: `Guard all-gather scatter state initialization`, `Preserve ring connections in
    all-gather endpoint guard`, `Fix all-gather endpoint no-target connection access`, `Clear fabric router packet
    tags on teardown`
  - the four `#23023` commits (`Bind BF4 cache loads to verified file descriptors`, `Reject lexical aliases for
    descriptor loads`, `Fail closed BF4 cache shape and cleanup`, `Bind BF4 tensorbin payload and exact types`):
    `ttnn.load_tensor` on `/proc/self/fd` paths, used by `ttnn/bf4.py`

  The server admits only a `ttnn` imported from this checkout's own build and records the checkout's commit, tree
  and extension digest with every run (`tools/runtime_admission.py`; `docs/SERVER.md`).
- The checkpoint: 360 GB (131 safetensors shards, the tokenizer, the chat template; section 3).
- Disk under `--cache-root`: 107 GB for the BF4 expert cache (built once, shared by every context, kept across
  runtime rebuilds), about 23 GB for the 32k context and 10 GB for each other allocated context (the converted
  non-expert weights and the model I/O cache), and the JIT kernel cache (about 1.3 GB).
- Host memory: the first start reads the checkpoint once and converts one MoE layer at a time (about 10 GB of host
  tensors in flight); 64 GB is comfortable.  The optional CPU reference (`tools/run_full_cpu_oracle.py`) needs
  170-240 GB and is not a user step.
- Build tooling: what tt-metal's `install_dependencies.sh` installs (clang-20 or gcc-12, cmake, ninja, python 3.10,
  `uv`).

## 2. Build

From a fresh clone, the standard tt-metal flow (about 10 minutes on 64 cores; on the QuietBox's 32 cores `build_metal.sh`
took 684 s and `create_venv.sh` 95 s, measured 2026-09-06):

    git clone https://github.com/sjettTT/tt-qwen-3.8-flash-next.git
    cd tt-qwen-3.8-flash-next
    git submodule update --init tt_metal/third_party/umd tt_metal/third_party/tracy tt_metal/third_party/tt-cluster-descriptors
    ./build_metal.sh
    ./create_venv.sh

`build_metal.sh` takes its defaults (Release, clang-20 with libstdc++; `--toolchain-path cmake/x86_64-linux-gcc-12-toolchain.cmake`
for gcc); `create_venv.sh` makes `python_env/` with torch and installs `ttnn` from this tree in editable mode (on a later
update it asks before reusing an existing `python_env/`; answer `y`), so
`python_env/bin/python -c 'import ttnn'` resolves inside the checkout.  The launcher below uses that interpreter and
sets `TT_METAL_HOME` to the checkout itself; nothing needs to be exported by hand.

## 3. The checkpoint

    python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.download_checkpoint --out /data/Qwen3.8-Flash-Next

fetches `Qwen/Qwen3.8-Flash-Next` from ModelScope at the release commit `2741eec155d03a8ce151b993ccce1a7b1e398d6b`
(145 files, 360,023,351,829 bytes) as parallel 128 MiB range requests (`--threads`, default 32), verifies every
file's SHA-256 against the ModelScope listing (saved as `<out>/.download/files.json`) and resumes after an
interruption.  The port pins an earlier ModelScope revision label (`checkpoint.PINNED_CHECKPOINT_REVISION`) whose
weights, config, tokenizer and chat template are byte-identical to the release commit; the server checks the files by
digest (every shard's header, the index, the file and tensor manifests, `config.json`, the tokenizer,
`chat_template.jinja`) and refuses a checkpoint that differs, with the digests printed.

A copy that is already on the host (another download, a relay from another machine) is checked the same way with
`--verify-only`: every file present is hashed against the listing, a marker is written per verified file and nothing is
fetched; a following run without `--verify-only` then fetches only the files that are absent or differ.
`tools/verify_checkpoint_files.py --checkpoint DIR --modelscope-tree <out>/.download/files.json --output report.json`
re-hashes a download later (standard library only, any `python3`); `tools/checkpoint_budget.py` prints the per-device
residency budget; `tools/safetensors_metadata.py` lists tensors.

## 4. Start the server

First the n-gram table.  Every decode token reads sixteen 320-byte rows of the PLE n-gram table, which stays in the
checkpoint (104 GB in 33 shards of layer 1) and is read by the host through the page cache, so on a host whose RAM
holds the table read it once before the start (13 s from a warm cache on the QuietBox; `docs/SERVER.md`):

    python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.prewarm_ple_table --checkpoint /data/Qwen3.8-Flash-Next

Then the server:

    models/demos/blackhole/qwen38_flash_next/tools/run_qwen38_chat_server.sh --profile tt-quietbox \
        --checkpoint /data/Qwen3.8-Flash-Next --cache-root /data/qwen38-cache --acceptance

    models/demos/blackhole/qwen38_flash_next/tools/run_qwen38_chat_server.sh --profile bh-loudbox \
        --checkpoint /data/Qwen3.8-Flash-Next --cache-root /data/qwen38-cache --acceptance

(`tools/` is `models/demos/blackhole/qwen38_flash_next/tools/`.)  The launcher prints the checkout it runs from (its
commit, whether the tree is modified), the interpreter and the `ttnn` extension, then the profile, the device set,
the context and the run directory, and starts the server.

**The first start** converts the routed experts of all 49 MoE layers into the BF4 cache
(`<cache-root>/caches/bf4-experts/`, 107 GB; a `bf4-stage-backbone-NN` phase per layer in the log), then builds the
component and model I/O caches of the chosen context (a few minutes), compiles the kernels (two to four minutes
cold) and captures the decode traces and the prefill chunk trace.  A layer takes about 33 s on a 4x p150 host (on the
QuietBox 2026-09-06: 49 layers in 1772 s), the 49 under half an hour.  A machine that bounds a job's wall time can
build the expert cache in pieces: `--prepare-only --bf4-stage-limit N` converts at most N missing layers and stops,
and the next run continues.  The cache is keyed by the checkpoint and the converter's sources, not by the tt-metal
revision: a rebuilt runtime keeps it, and every start checks one re-packed expert against it (`docs/SERVER.md`).

**Warm starts** reach `READY` in about five minutes: the weights load, the traces are captured, the acceptance
prompts (`--acceptance`: the twelve shipped CPU greedy records under `tools/acceptance/greedy-prompts/`) replay
against the CPU, then the server listens.  `--require-json-96` refuses to serve unless the `json` record matches
the CPU 96/96; the other records diverge from the CPU after 6-75 tokens, the known greedy-prompt pattern, and a start
whose replay leaves earlier than the pinned table is a regression.  The pinned tables, what they measure and how the
acceptance mechanism works are in `docs/NUMERICS.md`.

The run directory (`<cache-root>/runs/<stamp>/`) holds `READY`, `phase-markers.jsonl`, `requests.jsonl`,
`acceptance.json` and, at shutdown, `result.json` and `STOPPED`.

| launcher flag | meaning |
|---|---|
| `--profile tt-quietbox\|bh-loudbox\|qb2` | the hardware profile: the mesh graph descriptor, the device set, the route |
| `--allocated-context 32768\|65536\|131072\|262144` | the resident build (KV caches, RoPE tables and the context limit; the limit is the context minus 64 for the consumed EOS step); default 32768 |
| `--acceptance`, `--require-json-96` | replay the twelve CPU greedy records at start; refuse to serve unless `json` matches 96/96 |
| `--prepare-only --bf4-stage-limit N` | convert at most N missing expert layers into the BF4 cache and stop |
| `--long-chunks` | 128-row prefill chunks where the prompt allows (section 6); off by default, not combined with `--mtp` |
| `--prefill-slab 2048` | prefill slabs of 2048 rows ahead of the 128-row chunks: one matmul per dense linear, tolerance-class against the chunk bodies (`docs/PREFILL.md`); off by default, not combined with `--mtp` |
| `--mtp 3\|4` | speculative drafting on greedy requests (section 6); off by default |
| `--port`, `--host` | the listening port; `--host` default `0.0.0.0`: the QuietBox and LoudBox profiles serve the LAN |
| `--serve-seconds N` | stop after N seconds (a drain: the request in flight gets its reply) |
| `--sampling` / `--no-sampling` | the launcher passes `--sampling`: sampled requests are served, a request naming no sampling field is still the bitwise greedy stream; `--no-sampling` refuses sampling fields with HTTP 400 (+0.3 ms per token saved) |
| `--stall-seconds N` | the watchdog on zero device progress, 300 through the launcher; 0 disables it (`docs/SERVER.md`) |
| `--validate-only` | run the checks and the CPU preparation without opening the mesh (a profile whose route is derived at start, the LoudBox, still needs the chips present) |
| `--devices A,B,C,D` | run `bh-loudbox` on four other KMD device nodes (four chips of a larger host) |
| `--python` | another interpreter of this checkout |

What the launcher exports, how the runtime is admitted and what `result.json` and `/health` record: `docs/SERVER.md`.

## 5. Talk to it

OpenAI-compatible HTTP on the port you chose:

    curl -s http://<host>:8000/health
    curl -s http://<host>:8000/v1/models
    curl -s http://<host>:8000/v1/chat/completions -H 'content-type: application/json' -d '{
      "model": "Qwen/Qwen3.8-Flash-Next",
      "messages": [{"role": "user", "content": "Why does ice float?"}],
      "max_tokens": 256, "stream": true}'

The command-line client:

    python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_cli --url http://<host>:8000/v1 --thinking --tools

`POST /v1/chat/completions` (streaming or one document), `GET /v1/models`, `GET /health`.  The request rules in short
(`docs/SERVER.md` has them in full):

| request | rule |
|---|---|
| `messages` | the prompt, exactly: the server adds no system prompt, and the device prompt is the reference render (`tokenizer.apply_chat_template`), so `usage.prompt_tokens` is the count the client computes itself |
| `max_tokens` / `max_completion_tokens` | default and limit: the remaining context (the context limit less the prompt); a prompt over the limit gets HTTP 400 `context_length_exceeded` |
| `stream`, `stop`, `ignore_eos`, `seed`, `logprobs` | as in OpenAI; `logprobs` are relative to the read candidate row, not the vocabulary |
| `tools` / `tool_choice` | OpenAI shape, `tool_calls` finish reason; arguments are typed by the tool's parameter schema |
| `enable_thinking` (default true), `reasoning_effort`, `thinking_budget` | reasoning streams as `reasoning_content`; `chat_template_kwargs` means the same as the top-level fields |
| no sampling field | greedy: the argmax stream, bitwise equal to the greedy loop the acceptance replay measures (`temperature 0` and `greedy: true` are the same path) |
| `temperature > 0` | samples with it (`top_p` 1.0, `top_k` 20, no penalties unless given) |
| another sampling field alone (`top_p`, `top_k`, `min_p`, a penalty, `seed`) | samples with the model card's profile for the thinking mode (`/health.sampling_defaults`); `seed` alone is a reproducible sampled stream |
| `response_format` other than `text`, `logit_bias`, `parallel_tool_calls: false` | refused with HTTP 400 rather than dropped; unknown fields are logged |
| follow-up turns | the device keeps its committed prefix and prefills only the new turn; with thinking on it restores the prompt-end snapshot instead (`docs/SERVER.md`) |
| concurrency | one request decodes at a time; up to four wait in the queue (`queue_wait_seconds` in `usage`), the fifth gets HTTP 503 |

A client that hangs up is noticed at the next device step; a streaming request gets an SSE keepalive every 30 s
through the queue wait and the prefill; the stop signal (`--serve-seconds`, SIGTERM) drains.  The full serving
contract (hang-ups, stalled readers, deadlines, the stall watchdog, `/health` fields) is in `docs/SERVER.md`.

## 6. Long context and MTP

- `--allocated-context 65536` serves 65,472 tokens; `131072` and `262144` the same minus 64.  Prompt prefill runs
  through the chunk trace at about 3.2-3.5 ms per token (a 40k prompt: 125 s to the first token; 200k: 671 s), decode
  stays at 17-19 tokens/s to 256k.  Each context has its own component and model I/O caches under `--cache-root`
  (the BF4 expert cache is shared); 256k leaves about 750 MB per device free and is single-user.
- `--long-chunks` prefills in 128-row chunks where the prompt allows (the remainder in 32-row chunks): 1.55 ms per prompt
  token through the server (a 6942-token prompt in 10.8 s) against 3.3 with 32-row chunks alone, the same tokens (bitwise on
  all 48 layers); off by default and not combined with `--mtp`, whose chain prefills in 32-row chunks.
- MTP drafting (`--mtp 3|4`, 31-37 tokens/s on 4x p150) is off by default; greedy requests in the chunked prefill
  mode draft K tokens per pass with exact acceptance.  The MTP path is not bitwise with plain decode on 4 of the 12
  acceptance prompts (measured 2026-09-06): the committed stream leaves the CPU reference at a different token on
  `chat`, `list`, `math` and `summary`, at the plain-decode token on the other eight (`json` 96/96), and every gate
  passes; the three earlier tokens are near-ties within one bf16 step, not a defect (`docs/NUMERICS.md` has the
  indices and the pinned table).  `--mtp-gdn-anchor layer0` (server flag) re-anchors the layer-0 GDN state from the
  1-row recurrence.  MTP does not fit at 256k (94 MB free per bank against the 128 MiB contiguous it needs); 32k, 64k
  and 128k fit.

## 7. QuietBox 2 (untested)

A p300 card is two Blackhole dies joined on the card; a QuietBox 2 (2x p300c) has four dies in one ring (the two
on-card links and the two Warp400 links), so it is one 1x4 instance:

    models/demos/blackhole/qwen38_flash_next/tools/run_qwen38_chat_server.sh --profile qb2 --checkpoint ... --cache-root ...

The profile exports `tools/qb2_p300_1x4_line_mesh_graph_descriptor.textproto` (a 1x4 LINE over three of the four ring
links, two channels per link as in tt-metal's `p300_x2` descriptor); tt-metal classifies a p300 cluster that is not
exactly two or four dies as CUSTOM and refuses to open without a descriptor, so the launcher always exports one.  The
route is derived from the cluster descriptor at start and recorded.  Nothing here has run on p300 hardware.

## 8. Layout

    chat.py checkpoint.py config.py reference.py   the checkpoint, the chat template, the torch reference model
    diagnostic_bf4.py                              the binder of a CPU-staged BF4 expert corpus
    tt/                                            torch reference components (the CPU oracle every test compares against)
    ttnn/                                          the device model: builder, layers, GDN, QSA, MoE/BF4, embedding, sampling, MTP
    tools/run_qwen38_chat_server.sh                the launcher; qwen38_chat_cli.py the client
    tools/qwen38_chat_server.py                    the HTTP server; qwen38_chat_session.py the traced decode chain and the
                                                   mesh open; qwen38_chat_protocol.py the request/reply protocol;
                                                   qwen38_sampling_step.py
    tools/runtime_admission.py                     the runtime identity (this checkout's build); live_decode_diagnostic.py the
                                                   CPU preparation and the live construction (BF4 conversion on the first start)
    tools/hardware_profiles.py                     the profiles (tt-quietbox, bh-loudbox, tt-quietbox-2); physical_route.py the
                                                   route derivation; resident_decode.py the chain's fixed points;
                                                   evidence_records.py the run records
    tools/download_checkpoint.py                   the ModelScope download with SHA-256 verification
    tools/acceptance/                              the CPU greedy records the startup replay compares against, the two-step CPU oracle
    tools/qwen38_reference_corpus.py               the teacher-forced reference corpus Q38-REF-v1 (tools/reference/) and its
                                                   columns: HF on the CPU, the tt/ oracle, a served chain's agreement records; score
    tools/stage_full_bf4_cpu.py verify_full_bf4_cpu.py bind_full_bf4_corpus_cpu.py probe_full_bf4_binding_cpu.py
                                                   a BF4 expert corpus staged on the CPU (produce, verify, bind, probe)
    tools/prewarm_ple_table.py verify_checkpoint_files.py checkpoint_budget.py safetensors_metadata.py
    tools/qb_mesh_smoke.py                         open the mesh and check the route without the model
    tools/ci/q38_ci.py                             the regression harness: pins.json, baselines/, job runner, verdicts, seeding
    tests/                                         no-device tests (set QWEN38_CHECKPOINT for the checkpoint-reading ones)
    docs/                                          PROOFS, NUMERICS, SERVER, TESTING (the "More" section below)

Run the tests from the repository root (`docs/TESTING.md` has the regression harness and the reference corpus):

    QWEN38_CHECKPOINT=/data/Qwen3.8-Flash-Next python_env/bin/python -m pytest models/demos/blackhole/qwen38_flash_next/tests

## 9. What is verified, and the known limits

- `tt-quietbox`: a QuietBox (4x p150c, fw 19.4.1.0) served the model on 2026-09-04 from a pinned build: startup
  acceptance 96/96 against the CPU, 19.6 tokens/s at 32k.  The launcher and profile are the ones here; on 2026-09-06
  the same box served the model from a fresh clone of this repository (`docs/PROOFS.md`: build, checkpoint by digest,
  the first-start expert conversion, the acceptance replay, then `--mtp 4`, `--long-chunks` and the 64k context).
- The checkout build and `bh-loudbox`: `docs/PROOFS.md` records the fresh-clone proof (the build, the checkpoint by
  digest, the first-start expert conversion, the acceptance replay) and the hardware it ran on.
- Numerics, 2026-09-06 on 4x p150: `json` 96/96 against the CPU, the other eleven records leave the CPU greedy stream
  between token 6 and 75 (`docs/NUMERICS.md`); MTP is not bitwise with plain decode on 4 of 12 prompts (near-ties).
- `qb2`: designed, never run.
- 256k context is single-user; MTP is not available at 256k.
- One request decodes at a time (the traced chain is single-stream); the queue holds four more.
- The first start is long (the 107 GB expert conversion); a host that kills long jobs needs `--prepare-only
  --bf4-stage-limit N` runs first.
- Python 3.10 (`create_venv.sh` default); Linux x86_64.

## More

- `docs/PROOFS.md`: the fresh-clone release proof of 2026-09-06 on the QuietBox (build, checkpoint, first start,
  acceptance, `--mtp 4`, `--long-chunks`, 64k), with its run logs.
- `docs/NUMERICS.md`: the acceptance mechanism, the pinned divergence tables (plain, `--long-chunks`, MTP,
  teacher-forced), the GDN gate fix of 2026-09-06, what "bitwise" means here, the reference columns.
- `docs/SERVER.md`: the request rules in full, sampling, thinking, the prompt-end snapshot, the stall watchdog,
  `/health`, runtime admission, the BF4 cache identity, the n-gram table pre-warm, disk and memory.
- `docs/TESTING.md`: the no-device tests, the reference corpus Q38-REF-v1 and its scorer, the regression harness
  (`tools/ci/q38_ci.py`), how to run the acceptance gate.
- `docs/PREFILL.md`: the prefill chunk bodies and the opt-in 2048-row slab (`--prefill-slab`): what runs as one
  matmul, its numerics class, what it costs and saves.
