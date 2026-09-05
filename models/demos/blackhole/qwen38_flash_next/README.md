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
| QuietBox, 4x p150c (fw 19.4.1.0) | `tt-quietbox` | verified 2026-09-04: startup acceptance 96/96 tokens against the CPU, 19.6 tokens/s at 32k context |
| Blackhole LoudBox, 4x p150 (one line) | `bh-loudbox` | section 9: what was verified and where |
| QuietBox 2, 2x p300c (4 dies) | `qb2` | **untested**: designed from the p300 ring topology, should run fine |

Every number below was measured on 4x p150.

#### IMPORTANT NOTE: 
This was implemented with the intention of the n-gram model residing in system memory. I have not tested performance with disk. If you are not using a QB, please make sure you have room to fit the 51.2B parameters (they take up >100GB at BF16).

## Performance (4x p150, measured 2026-09-04)

| path | measured | notes |
|---|---|---|
| prompt prefill | 300 tok/s (and climbing) | 32-token chunk trace, 3.0-3.5 ms per prompt token, flat from 2k to 261k tokens; 128- and 256-token chunks are in progress |
| decode, one stream | 19.9 tok/s | position-generic traced decode, 50 ms per token, flat with depth |
| decode with MTP (`--mtp 4`) | 37 tok/s aggregate, 55 tok/s on structured output | speculative drafting with exact acceptance: the committed stream equals greedy decode |
| contexts | 32k, 64k, 128k, 256k | 256k is single-user; MTP fits at 32k, 64k and 128k |
| correctness | bitwise repeatable; 96/96 greedy token match against the CPU reference on the acceptance prompt | chunked prefill is tolerance-class against the CPU reference on all 48 layers |

## 1. What you need

- Four Blackhole chips in one host as one 1x4 mesh: a QuietBox (4x p150c in an ethernet ring), a Blackhole LoudBox
  (4x p150 in an ethernet line), or one four-chip half of an eight-chip host.  tt-kmd and the firmware bundle the
  chips shipped with (19.4.1.0 or later).
- This repository, built (section 2).  It is tt-metal main `d04395ed86` (2026-08-29) plus the runtime fixes the
  model needs, which are not on main yet:
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
  and extension digest with every run (`tools/runtime_admission.py`).
- The checkpoint: 360 GB (131 safetensors shards, the tokenizer, the chat template; section 3).
- Disk under `--cache-root`: 107 GB for the BF4 expert cache (built once, shared by every context), about 23 GB for
  the 32k context and 10 GB for each other allocated context (the converted non-expert weights and the model I/O
  cache), and the JIT kernel cache (about 1.3 GB).
- Host memory: the first start reads the checkpoint once and converts one MoE layer at a time (about 10 GB of host
  tensors in flight); 64 GB is comfortable.  The optional CPU reference (`tools/run_full_cpu_oracle.py`) needs
  170-240 GB and is not a user step.
- Build tooling: what tt-metal's `install_dependencies.sh` installs (clang-20 or gcc-12, cmake, ninja, python 3.10,
  `uv`).

## 2. Build

From a fresh clone, the standard tt-metal flow (about 10 minutes on 64 cores, longer on fewer):

    git clone --branch release/qwen38-preview git@github.com:sjettTT/qwen-3.8-flash-next.git
    cd qwen-3.8-flash-next
    git submodule update --init tt_metal/third_party/umd tt_metal/third_party/tracy tt_metal/third_party/tt-cluster-descriptors
    ./build_metal.sh
    ./create_venv.sh

`build_metal.sh` takes its defaults (Release, clang-20 with libstdc++; `--toolchain-path cmake/x86_64-linux-gcc-12-toolchain.cmake`
for gcc); `create_venv.sh` makes `python_env/` with torch and installs `ttnn` from this tree in editable mode, so
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

`tools/verify_checkpoint_files.py --checkpoint DIR --modelscope-tree <out>/.download/files.json` re-hashes a
download later; `tools/checkpoint_budget.py` prints the per-device residency budget; `tools/safetensors_metadata.py`
lists tensors.

## 4. Start the server

    tools/run_qwen38_chat_server.sh --profile tt-quietbox \
        --checkpoint /data/Qwen3.8-Flash-Next --cache-root /data/qwen38-cache --acceptance

    tools/run_qwen38_chat_server.sh --profile bh-loudbox \
        --checkpoint /data/Qwen3.8-Flash-Next --cache-root /data/qwen38-cache --acceptance

(`tools/` is `models/demos/blackhole/qwen38_flash_next/tools/`.)  The launcher prints the checkout it runs from (its
commit, whether the tree is modified), the interpreter and the `ttnn` extension, then the profile, the device set,
the context and the run directory, and starts the server.

**The first start** converts the routed experts of all 49 MoE layers into the BF4 cache
(`<cache-root>/caches/bf4-experts/`, 107 GB: each layer read from the checkpoint, packed on the host, uploaded to the
mesh and written back as one tensorbin per weight; a `bf4-stage-backbone-NN` phase per layer in the log), then builds
the component and model I/O caches of the chosen context (a few minutes), compiles the kernels (the JIT cache fills
during the warm pass, two to four minutes cold) and captures the decode traces and the prefill chunk trace.  A
machine that bounds a job's wall time can build the expert cache in pieces: `--prepare-only --bf4-stage-limit N`
converts at most N missing layers and stops; every layer is manifested as it completes, so the next run continues.
**Warm starts** reach `READY` in about five minutes: the weights load, the traces are captured, the acceptance
prompts (`--acceptance`: the twelve shipped CPU greedy records under `tools/acceptance/greedy-prompts/`) replay
against the CPU, then the server listens.  `--require-json-96` refuses to serve unless the `json` record matches
the CPU 96/96 (the other records diverge from the CPU after 6-75 tokens, the known greedy-prompt pattern; their first
divergence index is in `acceptance.json`).

The run directory (`<cache-root>/runs/<stamp>/`) holds `READY`, `phase-markers.jsonl`, `requests.jsonl`,
`acceptance.json` and, at shutdown, `result.json` and `STOPPED`.  `result.json` and `/health` carry the runtime
identity: the checkout's commit and tree, the extension's SHA-256, the route the mesh opened in.

Options: `--allocated-context 32768|65536|131072|262144` selects the resident build (KV caches, RoPE tables and the
context limit; the limit is the context minus 64 for the consumed EOS step), `--port`, `--host` (default
`0.0.0.0`: the QuietBox and LoudBox profiles serve the LAN), `--serve-seconds N` to stop after N seconds,
`--validate-only` to run the checks and the CPU preparation without opening the mesh (a profile whose route is
derived at start, the LoudBox, still needs the chips present), `--devices A,B,C,D` to run `bh-loudbox` on four other
KMD device nodes (one half of an eight-chip host), `--python` for another interpreter of this checkout.

What the launcher does not do: no device locks, no runtime archives or digests.  It exports the QuietBox mesh graph
descriptor for `tt-quietbox` (`tools/qb_p150_x4_1x4_line_mesh_graph_descriptor.textproto`: the four chips' ethernet
ring opened as one 1x4 line), the device set, the cache and log roots, and `TT_METAL_HOME` = this checkout.

### Runtime admission

`tools/runtime_admission.py` admits the runtime before the mesh opens: the interpreter's `ttnn` package and its
compiled extension must resolve under this repository (`ttnn` built from this checkout), the checkout's `git`
head and tree are read, the extension is hashed.  That identity (`{repo, head, tree, dirty, extension,
extension_sha256}`) becomes the builder provenance (`tt_metal_sha` = the head, `ttnn_runtime_sha256` = the digest),
the cache namespaces, the `system_fingerprint` of every response and the `runtime` block of `result.json`.  A
modified tree is admitted and recorded as `dirty`.  A `ttnn` from elsewhere is refused with both paths printed.  The
CPU preparation (`tools/live_decode_diagnostic.py`) derives the consumer identity from that runtime identity, the
pinned checkpoint digests and the mesh order; the BF4 experts come from the production cache described above, or,
with `--bf4-corpus DIR --bf4-corpus-verification FILE`, from a corpus staged on the CPU by
`tools/stage_full_bf4_cpu.py` and verified by `tools/verify_full_bf4_cpu.py` (`diagnostic_bf4.py` binds it; the
producer identity is what the verification record claims, `--bf4-producer-identity` pins it).

## 5. Talk to it

OpenAI-compatible HTTP on the port you chose:

    curl -s http://<host>:8000/health
    curl -s http://<host>:8000/v1/models
    curl -s http://<host>:8000/v1/chat/completions -H 'content-type: application/json' -d '{
      "model": "Qwen/Qwen3.8-Flash-Next",
      "messages": [{"role": "user", "content": "Why does ice float?"}],
      "max_tokens": 256, "stream": true}'

`POST /v1/chat/completions` (streaming or one document), `GET /v1/models`, `GET /health` (context limit, sampling
mode, free DRAM after the captures, the runtime identity).  Requests: `messages`, `max_tokens` or
`max_completion_tokens` (default and limit: the remaining context, the context limit less the prompt), `stream`,
`stop`, `tools` / `tool_choice` (OpenAI shape; `tool_calls` finish reason), `enable_thinking` (default true;
reasoning streams as `reasoning_content`), `reasoning_effort`, `thinking_budget`, `ignore_eos`, `seed`,
`temperature` / `top_p` / `top_k` / `min_p` / `logprobs` (a server started with `--sampling`; the default server is
greedy and refuses them with HTTP 400 unless `temperature` is 0).  `chat_template_kwargs` (`enable_thinking`,
`reasoning_effort`, the vLLM spelling) means the same as the top-level fields; JSON `null` is an absent field.  What
the server cannot honour is refused with HTTP 400 rather than dropped: `response_format` other than `text`,
`logit_bias`, `parallel_tool_calls: false`; unknown fields are logged.  Tool-call arguments are typed by the tool's
parameter schema (a `string` parameter is returned as text whatever it looks like).  `logprobs` are relative to the
read candidate row, not the vocabulary (`logprobs_normalizer` in `/health` and `qwen38.sampling`).  One request
decodes at a time; up to four wait in the queue (`queue_wait_seconds` in `usage`), the fifth gets HTTP 503.  A prompt
over the context limit gets HTTP 400 `context_length_exceeded`.

The command-line client:

    python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_cli --url http://<host>:8000/v1 --thinking --tools

## 6. Long context and MTP

- `--allocated-context 65536` serves 65,472 tokens; `131072` and `262144` the same minus 64.  Prompt prefill runs
  through the chunk trace at about 3.2-3.5 ms per token (a 40k prompt: 125 s to the first token; 200k: 671 s), decode
  stays at 17-19 tokens/s to 256k.  Each context has its own component and model I/O caches under `--cache-root`
  (the BF4 expert cache is shared); 256k leaves about 750 MB per device free and is single-user.
- MTP drafting (`--mtp 3|4`, 31-37 tokens/s on 4x p150) is off by default; greedy requests in the chunked prefill
  mode draft K tokens per pass and the committed stream equals greedy decode.  `--mtp-gdn-anchor layer0` (server
  flag) re-anchors the layer-0 GDN state from the 1-row recurrence.  MTP does not fit at 256k (94 MB free per bank
  against the 128 MiB contiguous it needs); 32k, 64k and 128k fit.

## 7. QuietBox 2 (untested)

A p300 card is two Blackhole dies joined on the card; a QuietBox 2 (2x p300c) has four dies in one ring (the two
on-card links and the two Warp400 links), so it is one 1x4 instance:

    tools/run_qwen38_chat_server.sh --profile qb2 --checkpoint ... --cache-root ...

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
    tools/stage_full_bf4_cpu.py verify_full_bf4_cpu.py bind_full_bf4_corpus_cpu.py probe_full_bf4_binding_cpu.py
                                                   a BF4 expert corpus staged on the CPU (produce, verify, bind, probe)
    tools/prewarm_ple_table.py verify_checkpoint_files.py checkpoint_budget.py safetensors_metadata.py
    tools/qb_mesh_smoke.py                         open the mesh and check the route without the model
    tests/                                         no-device tests (set QWEN38_CHECKPOINT for the checkpoint-reading ones)

Run the tests from the repository root:

    QWEN38_CHECKPOINT=/data/Qwen3.8-Flash-Next python_env/bin/python -m pytest models/demos/blackhole/qwen38_flash_next/tests

## 9. What is verified, and the known limits

- `tt-quietbox`: a QuietBox (4x p150c, fw 19.4.1.0) served the model on 2026-09-04 from a pinned build: startup
  acceptance 96/96 against the CPU, 19.6 tokens/s at 32k.  The launcher and profile are the ones here.
- The checkout build and `bh-loudbox`: section 9a records the fresh-clone proof of this release (built with
  `build_metal.sh` + `create_venv.sh`, the checkpoint by digest, the first-start expert conversion, the acceptance
  replay) and the hardware it ran on.
- `qb2`: designed, never run.
- 256k context is single-user; MTP is not available at 256k.
- One request decodes at a time (the traced chain is single-stream); the queue holds four more.
- The first start is long (the 107 GB expert conversion); a host that kills long jobs needs `--prepare-only
  --bf4-stage-limit N` runs first.
- Python 3.10 (`create_venv.sh` default); Linux x86_64.

### 9a. Release proof

Filled in by the proof run of the release branch.
