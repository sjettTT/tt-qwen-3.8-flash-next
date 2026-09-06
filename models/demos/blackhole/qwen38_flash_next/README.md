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
| QuietBox, 4x p150c (fw 19.4.1.0) | `tt-quietbox` | verified 2026-09-04: startup acceptance 96/96 tokens against the CPU, 19.6 tokens/s at 32k context; section 9a: the fresh-clone proof of 2026-09-06 |
| Blackhole LoudBox, 4x p150 (one line) | `bh-loudbox` | section 9: what was verified and where |
| QuietBox 2, 2x p300c (4 dies) | `qb2` | **untested**: designed from the p300 ring topology, should run fine |

Every number below was measured on 4x p150.

#### IMPORTANT NOTE: 
This was implemented with the intention of the n-gram model residing in system memory. I have not tested performance with disk. If you are not using a QB, please make sure you have room to fit the 51.2B parameters (they take up >100GB at BF16).

## Performance (4x p150, measured 2026-09-04)

| path | measured | notes |
|---|---|---|
| prompt prefill | 300 tok/s (and climbing); 500 tok/s with `--long-chunks` | 32-token chunk trace, 3.0-3.5 ms per prompt token, flat from 2k to 261k tokens; 128-token chunks at 1.9-2.0 ms per prompt token (`--long-chunks`, one routed-expert stream per 128 rows since 2026-09-06); 256-token chunks are in progress |
| decode, one stream | 19.9 tok/s | position-generic traced decode, 50 ms per token, flat with depth |
| decode with MTP (`--mtp 4`) | 37 tok/s aggregate, 55 tok/s on structured output | speculative drafting with exact acceptance: the committed stream leaves the CPU reference at the same token as greedy decode on 8 of the 12 acceptance prompts and at a different token on the other 4 (section 6) |
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
  and extension digest with every run (`tools/runtime_admission.py`).
- The checkpoint: 360 GB (131 safetensors shards, the tokenizer, the chat template; section 3).
- Disk under `--cache-root`: 107 GB for the BF4 expert cache (built once, shared by every context, kept across
  runtime rebuilds), about 23 GB for
  the 32k context and 10 GB for each other allocated context (the converted non-expert weights and the model I/O
  cache), and the JIT kernel cache (about 1.3 GB).
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

A copy that is already on the host (another download, a relay from another machine) is checked the same way with
`--verify-only`: every file present is hashed against the listing, a marker is written per verified file and nothing is
fetched; a following run without `--verify-only` then fetches only the files that are absent or differ.
`tools/verify_checkpoint_files.py --checkpoint DIR --modelscope-tree <out>/.download/files.json --output report.json`
re-hashes a download later (standard library only, any `python3`); `tools/checkpoint_budget.py` prints the per-device
residency budget; `tools/safetensors_metadata.py` lists tensors.

## 4. Start the server

First the n-gram table.  Every decode token reads sixteen 320-byte rows of the PLE n-gram table, which stays in the
checkpoint (104 GB in 33 shards of layer 1) and is read by the host through the page cache; on a cold cache each row
is an NVMe page-in, so on a host whose RAM holds the table (the QuietBox has 503 GB) read it once before the start:

    python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.prewarm_ple_table --checkpoint /data/Qwen3.8-Flash-Next

It prints the residency before and after (`fincore`, util-linux) and the read rate; nothing is written.  On the
QuietBox the pass over the 104 GB took 13 s from a warm cache (7.8 GB/s); a cold NVMe cache is 35-60 s at 2-3 GB/s.
`--report-only` only prints the residency.  Then the server:

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
during the warm pass, two to four minutes cold) and captures the decode traces and the prefill chunk trace.  A layer
takes about 33 s on a 4x p150 host (2.2 GB written; measured 2026-09-05: 25 layers in 812 s; on the QuietBox 2026-09-06:
49 layers in 1772 s, 35.9-36.9 s each, 100 GB), the 49 under half an hour;
the payload of every tensorbin is byte for byte the CPU-staged corpus's (`tools/stage_full_bf4_cpu.py`), the manifest records the slot's global shape
(512 experts) while the mesh tensor presents one device's 128.  A machine that bounds a job's wall time can build the
expert cache in pieces: `--prepare-only --bf4-stage-limit N` converts at most N missing layers and stops; every layer
is manifested as it completes (a refused layer publishes nothing), so the next run continues.
The expert cache is keyed by the checkpoint and by the converter's sources (`ttnn/bf4.py`, the `moe_compute` layout
packer, tt-metal's BFP4 packer), not by the tt-metal revision: a rebuilt runtime keeps the cache.  Every start re-packs
one routed expert of the first cached layer from the checkpoint and compares the bytes with the cache (about a second,
the `bf4-cache-admission` phase); a cache converted by different code is refused, the layer, expert and tensor named.
A cache built by an earlier runtime, keyed by its tt-metal revision, is adopted on the first start: its manifest is
rewritten and the slot renamed under the new key, nothing is reconverted.
**Warm starts** reach `READY` in about five minutes: the weights load, the traces are captured, the acceptance
prompts (`--acceptance`: the twelve shipped CPU greedy records under `tools/acceptance/greedy-prompts/`) replay
against the CPU, then the server listens.  `--require-json-96` refuses to serve unless the `json` record matches
the CPU 96/96 (the other records diverge from the CPU after 6-75 tokens, the known greedy-prompt pattern; their first
divergence index is in `acceptance.json`).  The records were rendered by the CPU study with the system prompt
`You are a helpful assistant.`; that system turn is inside their recorded prompt ids, which the replay feeds to the
device as they are.  The server itself adds no system prompt to a client's request (section 5).

The pinned table (`tools/ci/baselines/A3-chunked-32k-divergence_index.json`), measured 2026-09-06 on 4x p150 with
the chunked prefill, is the first index where each record leaves the CPU greedy stream; a start whose replay leaves
earlier is a regression:

| json | chat | code | fact | list | math | multilingual | prose | refactor | sky | story | summary |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none (96/96) | 43 | 32 | 15 | 56 | 61 | 9 | 13 | 24 | 19 | 6 | 75 |

Before 2026-09-06 the table read chat 8, code 24, list 46, refactor 22 (the other eight as above).  The change is the
GDN decay gate: the fused `add + softplus` activation the gate used returned exactly 0 wherever its input was below
-5.02 (and was 1.5e-3 off elsewhere); the gate now runs `ttnn.add` then `ttnn.softplus`, which follows the reference
everywhere (4e-5 absolute).  Against the CPU oracle the device's GDN state error on a decode step fell from 0.29 to
0.013; with the chunked prefill the device stays on the CPU stream longer on 4 of the 12 prompts and no prompt leaves
earlier (the teacher-forced table, `A3-forced-32k-divergence_index.json`, moved later on four prompts and earlier on
three); the cost is one more program per layer, about 0.4 ms per decoded token (50.0 -> 50.4 ms).  The proofs of
section 9a predate this fix where they quote the old indices.

**Reference corpus.**  `tools/reference/` freezes `Q38-REF-v1`: 36 teacher-forced items (the twelve acceptance
prompts as their records render them and as the server renders a request today, four requests with tools, the first
1024 tokens of two books, four evaluation items, two long prompts scored in 32-position windows), about 13k positions,
with sha256s.  `tools/qwen38_reference_corpus.py hf` runs the Transformers `qwen4_exp` model on the CPU over it (bf16
weights, fp32 LM head; a transformers checkout with the `qwen4_exp` model class, several hundred GB of RAM, hours) and
keeps per position the top-32 ids and log-probs and the teacher's log-prob; `oracle` does the same through `tt/` (bf16
or BF4-emulated experts); `device` converts a served chain's agreement records; `score` compares two columns (top-1 /
top-5 agreement, clear-margin top-1, truncated KL over the shared top-32 support, first divergence).  The HF column is
the acceptance reference the other columns are read against.

The run directory (`<cache-root>/runs/<stamp>/`) holds `READY`, `phase-markers.jsonl`, `requests.jsonl`,
`acceptance.json` and, at shutdown, `result.json` and `STOPPED`.  `result.json` and `/health` carry the runtime
identity: the checkout's commit and tree, the extension's SHA-256, the route the mesh opened in.

Options: `--allocated-context 32768|65536|131072|262144` selects the resident build (KV caches, RoPE tables and the
context limit; the limit is the context minus 64 for the consumed EOS step), `--port`, `--host` (default
`0.0.0.0`: the QuietBox and LoudBox profiles serve the LAN), `--serve-seconds N` to stop after N seconds,
`--validate-only` to run the checks and the CPU preparation without opening the mesh (a profile whose route is
derived at start, the LoudBox, still needs the chips present), `--devices A,B,C,D` to run `bh-loudbox` on four other
KMD device nodes (four chips of a larger host), `--python` for another interpreter of this checkout.  The launcher
starts the server with `--sampling` (sampled requests are served; a request naming no sampling field is still the
bitwise greedy stream, section 5) and `--stall-seconds 300` (the watchdog, section 5); `--no-sampling` serves greedy
requests only (+0.3 ms per token saved, sampling fields refused with HTTP 400) and `--stall-seconds 0` disables the
watchdog.

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
`temperature` / `top_p` / `top_k` / `min_p` / `presence_penalty` / `frequency_penalty` / `repetition_penalty` /
`logprobs`.  Sampling: a request that names none of the sampling fields is greedy, the argmax stream bitwise equal to
the greedy loop the acceptance replay and the evaluations measure, on the launcher's `--sampling` server too
(`qwen38.decode_loop` is `greedy` in the response and the ledger; `temperature 0` and `greedy: true` are the same
path); `temperature > 0` samples with it (`top_p` 1.0, `top_k` 20, no penalties unless given); another sampling
field alone (`top_p`, `top_k`, `min_p`, a penalty, `seed`) samples with the model card's profile for the thinking
mode (`/health.sampling_defaults`), so `seed` alone is a reproducible sampled stream.  A `--no-sampling` server
refuses every sampling field with HTTP 400 unless `temperature` is 0.  `chat_template_kwargs` (`enable_thinking`,
`reasoning_effort`, the vLLM spelling) means the same as the top-level fields; JSON `null` is an absent field.  What
the server cannot honour is refused with HTTP 400 rather than dropped: `response_format` other than `text`,
`logit_bias`, `parallel_tool_calls: false`; unknown fields are logged.  Tool-call arguments are typed by the tool's
parameter schema (a `string` parameter is returned as text whatever it looks like).  `logprobs` are relative to the
read candidate row, not the vocabulary (`logprobs_normalizer` in `/health` and `qwen38.sampling`).  One request
decodes at a time; up to four wait in the queue (`queue_wait_seconds` in `usage`), the fifth gets HTTP 503.  A prompt
over the context limit gets HTTP 400 `context_length_exceeded`.

The prompt is the client's messages, exactly: the server adds no system prompt when the request carries none
(`/health.defaults.system_prompt` is null), and the device prompt of every request is the reference render
(`tokenizer.apply_chat_template` on the request), so `usage.prompt_tokens` is the count the client computes itself.
A follow-up turn holds the served reply as the template re-renders it from the client's echo (its content and tool
calls, an empty think block); reasoning never re-enters the device context (`qwen38.served_reasoning_tokens` is
always 0).  The device keeps its committed prefix when the render extends it and prefills only the new turn
(`qwen38.reset` false, `prefix_reused` the reused count): with thinking off, the template renders the past reply as
the generation prompt plus its text, so a conversation continues at the cost of the new turn.  With thinking on, the
template renders the past turn's think block empty (`<think>\n\n</think>`), which the tokenizer merges differently
from the `<think>\n` the model generated after, so the render never extends the committed ids; the server then
restores the prompt-end snapshot instead (`qwen38.prefix_restored` true, `prefix_reused` = the prompt length less
one): before the last prompt token of every request the chain copies the recurrent part of the device state (GDN
states and ring slots, PLE slots, QSA staging and raw-key rings, ~53 MB per device; the KV and compressed caches are
positional and rewritten by the tail) into a resident snapshot, and a follow-up whose render extends those ids
copies it back and prefills only the rendered tail (the re-rendered reply and the new turn) from that position.  An
exact repeat of a prompt restores the same way.  A history that diverges earlier (an edited turn) still resets and
prefills the whole conversation (3.3 ms per token of history, `qwen38.reset` true).

A client that hangs up is noticed at the next device step (or prefill event) whether or not anything was being
streamed to it, and a queued request whose client left gives up its place: the device never runs a request for
nobody.  A streaming request gets its head and role chunk as soon as it is admitted and an SSE comment
(`: keepalive`) every `--heartbeat-seconds` (30 s) through the queue wait and the prefill, so a 60 s proxy or SDK
read timeout does not cut a long prompt.  A socket write blocked for `--socket-timeout-seconds` (60 s: a reader that
stopped reading) ends the request as `disconnected`.  `--request-deadline-seconds` (off by default) is honoured
inside the chunked prefill too.  `/health.current_request` shows the request holding the device with the seconds
since its last completed step.

The stall watchdog (`--stall-seconds`, 300 through the launchers) fires only on zero progress.  Its clock belongs to
the request holding the device: it starts when the request is admitted from the queue and restarts at every
completed device step: every decode step (50 ms), every teacher-forced prefill event (16 forced tokens, under a
second) and every chunk-prefill event sync (4 chunks, about 0.4 s; 1.3 s with `--long-chunks`), so a 200k-token
prefill restarts it several times a second and a long answer every token; it is not measured while no request holds
the device or while requests only wait in the queue.  A request whose device call has not returned for that long is
a wedge: the server logs `stalled`, ends with exit status 1 without releasing the chain, and a supervisor restarts
it (the launcher exits with the server's status).  The value must exceed `--socket-timeout-seconds` (a client write
blocked for that long is not a device step); `--stall-seconds 0` on the launcher disables the watchdog.  The stop
signal (`--serve-seconds`, SIGTERM) drains: the request in flight ends at its next step with `qwen38.finish`
`shutdown` and gets its reply, queued and new requests get 503, then the chain is released.  `HEAD` and `OPTIONS` are
served (no CORS headers); a body needs `Content-Length`.

The command-line client:

    python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_cli --url http://<host>:8000/v1 --thinking --tools

## 6. Long context and MTP

- `--allocated-context 65536` serves 65,472 tokens; `131072` and `262144` the same minus 64.  Prompt prefill runs
  through the chunk trace at about 3.2-3.5 ms per token (a 40k prompt: 125 s to the first token; 200k: 671 s), decode
  stays at 17-19 tokens/s to 256k.  Each context has its own component and model I/O caches under `--cache-root`
  (the BF4 expert cache is shared); 256k leaves about 750 MB per device free and is single-user.
- `--long-chunks` prefills in 128-row chunks where the prompt allows (the remainder in 32-row chunks): 2.0 ms per prompt
  token through the server (a 6942-token prompt in 13.6 s) against 3.3 with 32-row chunks alone, the same tokens (bitwise on
  all 48 layers); off by default and not combined with `--mtp`, whose chain prefills in 32-row chunks.
- MTP drafting (`--mtp 3|4`, 31-37 tokens/s on 4x p150) is off by default; greedy requests in the chunked prefill
  mode draft K tokens per pass with exact acceptance.  The MTP path is not bitwise with plain decode on 4 of the 12
  acceptance prompts (measured 2026-09-06 with the GDN gate fix of section 4; the verify rows and the 1-row loop
  round differently): the committed stream leaves the CPU reference on `chat` at token 56 where plain decode leaves
  at 43, on `list` at 46 against 56, on `math` at 56 against 61 and on `summary` at 1 against 75 (`In 1947,` becomes
  `Invented at Bell Labs in`); the other eight records leave the reference at the plain-decode token (`json` 96/96,
  `code` and `refactor` bitwise the plain streams), and every gate passes.  Before the gate fix the two paths differed
  on `code` (44 against 24) and `fact` (16 against 15) only.  `--mtp-gdn-anchor layer0` (server flag) re-anchors the
  layer-0 GDN state from the 1-row recurrence.  MTP does not fit at 256k (94 MB free per bank against the 128 MiB
  contiguous it needs); 32k, 64k and 128k fit.

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
    tools/qwen38_reference_corpus.py               the teacher-forced reference corpus Q38-REF-v1 (tools/reference/) and its
                                                   columns: HF on the CPU, the tt/ oracle, a served chain's agreement records; score
    tools/stage_full_bf4_cpu.py verify_full_bf4_cpu.py bind_full_bf4_corpus_cpu.py probe_full_bf4_binding_cpu.py
                                                   a BF4 expert corpus staged on the CPU (produce, verify, bind, probe)
    tools/prewarm_ple_table.py verify_checkpoint_files.py checkpoint_budget.py safetensors_metadata.py
    tools/qb_mesh_smoke.py                         open the mesh and check the route without the model
    tools/ci/q38_ci.py                             the regression harness: pins.json, baselines/, job runner, verdicts, seeding (8a)
    tests/                                         no-device tests (set QWEN38_CHECKPOINT for the checkpoint-reading ones)

Run the tests from the repository root:

    QWEN38_CHECKPOINT=/data/Qwen3.8-Flash-Next python_env/bin/python -m pytest models/demos/blackhole/qwen38_flash_next/tests

### 8a. The regression harness (development)

`tools/ci/q38_ci.py` (standard library only) turns the numbers the tools already produce into gated verdicts.
`tools/ci/pins.json` names every gated value as `<job>/<configuration>/<metric>` with a rule and a status:

- rules: `band` (a symmetric relative band, two-sided as in tt-metal's model targets: a result better than the band
  is a stale target), `floor` (target minus slack), `ceiling` (target times 1 + tolerance, with a warning level),
  `not_earlier` (divergence indices; null is the largest), `exact`, `at_most`, `flips` (per-item eval answers: items
  the baseline passes and the run fails, at most N per task);
- status `todo` warns only, `active` gates; a pin with `baseline: true` compares a per-key map (per prompt, item or
  task) against `tools/ci/baselines/<pin id with - for />.json`;
- jobs: `A1` corpus agreement against the HF reference (the scorer's `score.json`), `A2` the CPU oracle against HF,
  `A3` the startup acceptance replay plus the runner's probes (the verbatim echo of a sentence at temperature 0,
  N short completions that must all finish, one-token replies at several prompt lengths for TTFT), `D1` perf (the
  timing runner's period, the ledger's prefill and decode rates, TTFT, startup, captures, program cache, DRAM
  headroom), `C1`/`C2` lm-eval per-item answers and accuracies. Device-timed pins are `not_gated` when the 1-minute
  load average is above `idle_loadavg_1min`, or use their `loaded_tolerance`.

A job list (`qwen38-ci-jobs/v1`) names the commands to run on one lane: a `command` job runs to completion, a
`server` job starts a launcher, waits for its `READY` marker, probes the server over HTTP and sends the server pid
SIGTERM; each job then collects its artifacts (globs over the launchers' evidence directories, `$Q38_CI_RUN_DIR`
for an earlier job's files) and validates them. One result file per job (`qwen38-ci-result/v1`: host, lane, head,
runtime identity, load average, item ids, every observed value, one verdict per pin with observed and expected
side by side) lands under `<out>/<date>/<stamp>-<lane>-<head12>/<job>/`, with `summary.json`, `summary.txt` and
`progress.log` beside them.

    python -m models.demos.blackhole.qwen38_flash_next.tools.ci.q38_ci run --jobs JOBS.json --out RESULTS
    python -m models.demos.blackhole.qwen38_flash_next.tools.ci.q38_ci report --run RESULTS/<date>/<run>
    python -m models.demos.blackhole.qwen38_flash_next.tools.ci.q38_ci seed --runs RESULTS/<date>/<run>... --write

`seed` promotes `todo` pins whose last three idle runs agree within the pin's tolerance: the target becomes the
median (`band`, `ceiling`), the minimum or the preset proposal (`floor`) or the common value, the run ids are
recorded on the pin, and baseline-backed pins get their baseline file written. Until then the committed targets
are proposals from the runs named in each pin's `note` and the baselines' `source`; only the rules that already
gate today are `active` (the json record's 96/96 replay, the echo, request completion, a program-cache delta of 0).

## 9. What is verified, and the known limits

- `tt-quietbox`: a QuietBox (4x p150c, fw 19.4.1.0) served the model on 2026-09-04 from a pinned build: startup
  acceptance 96/96 against the CPU, 19.6 tokens/s at 32k.  The launcher and profile are the ones here; section 9a
  is the same box served from a fresh clone of this repository.
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

The QuietBox (`tt-quietbox`: 4x p150c, which `tt-smi` reports as p150b; firmware bundle 19.4.1.0, tt-kmd 2.6.0-rc1, 32 cores, 503 GB RAM, Ubuntu
22.04, clang-20, Python 3.10.19 through `uv`) served the model on 2026-09-06 from a fresh clone of the public repository
at `cadebdff7c1c`, following sections 2-5 as written (the deviations found on the way are folded into the text above):

- clone 53 s, the three submodules 15 s, `build_metal.sh` 684 s with its defaults, `create_venv.sh` 95 s; the
  runtime identity of every run: head `cadebdff7c1c`, tree `dd25966f522c`, clean, extension
  `f3d1fb4c3ab4...`.
- the checkpoint copy already on the host: `download_checkpoint.py --verify-only` verified 142 of the 145 listed files
  in 13 s (LICENSE differed, `.gitattributes` and `configuration.json` were absent), the plain run fetched those three in
  17 s (145/145); `verify_checkpoint_files.py`: 131/131 shards, 360,000,192,888 bytes, every SHA-256 equal to the
  ModelScope listing (176 s, 4 workers).
- `prewarm_ple_table.py`: the 104,298,732,704 B of the n-gram table in 13.4 s (already resident), 33/33 files resident.
- the first start (`--profile tt-quietbox --acceptance --require-json-96`, 32k): mesh open 9.6 s; the 49 BF4 layers
  1772.5 s (35.9-36.9 s each, 100 GB written); target build 41.4 s; warm pass with a cold JIT cache 150.2 s; captures
  5.1 s and the chunk capture 1.7 s; acceptance replay 59.2 s; `READY` 2096.6 s after the mesh open (launched
  17:19:30Z, `READY` 17:54:33Z); 474,261,568 bytes free per bank after the captures.
- acceptance: `json` 96/96 (the gate passed); the other eleven records leave the CPU stream at the same indices as the
  4x p150 hosts did (chat 8, code 24, fact 15, list 46, math 61, multilingual 9, prose 13, refactor 22, sky 19, story
  6, summary 75: the table before the GDN gate fix of 2026-09-06, section 4 has the current one); 19.4-19.6 tokens/s
  in the replays.
- requests over the LAN: a 36-token answer at 19.1 tokens/s (first token 0.30 s after a 33-token prompt), a 128-token
  generation at 19.6 tokens/s (first token 0.39 s, 47-token prompt in 2 chunks); the CLI's question answered.  SIGTERM
  stopped it cleanly (`result.json` status `stopped`, mesh closed, launcher exit 0).
- the same launcher line with `--mtp 4` (warm caches; the MTP kernels compiled on this start): `READY` 225 s after the
  launch (MTP warm pass 40 s, acceptance replay 48 s); `json` 96/96 through MTP at 55.0 tokens/s (4.8 tokens per
  pass), the split hand-off gate passed in both orders; `code` left the CPU stream at 44 and `fact` at 16, the other
  nine records at the plain-decode indices of that day (before the GDN gate fix of 2026-09-06; section 6 has the
  current MTP indices); 375,594,496 bytes free per bank (98.7 MB less than without MTP).
  Requests: the `json` prompt as a chat request reproduced the CPU record's 96 tokens at 55.2 tokens/s; a 6942-token
  prompt prefilled in 23.4 s (3.37 ms per prompt token, 217 chunks of 32 rows) then decoded at 31.3 tokens/s (3.0 per
  pass), its follow-up turn reused the 6980 committed tokens (first token 1.6 s, 43.1 tokens/s); a 128-token generation
  36.7 tokens/s (27.0 ms per token); a 220-token prose answer 31.0 tokens/s (2.6 per pass).
- the same line with `--long-chunks` (warm caches; the 128-row chunk kernels compiled on this start): `READY` 214 s
  after the launch; acceptance identical to the plain start (`json` 96/96, the same eleven divergence indices, 19.4-19.6
  tokens/s); 439,384,896 bytes free per bank (34.9 MB less than without).  The 6942-token prompt prefilled in 18.8 s =
  2.71 ms per prompt token (first token 18.9 s; 3.37 with 32-row chunks on the MTP start above), hand-off 805 ms, the
  same answer; decode 50.6 ms per token (19.6 tokens/s on a 128-token generation).  The `json` chat request again
  reproduced the CPU record's 96 tokens.
- `--allocated-context 65536` (the 64k component and model I/O caches built on this start, 9.8 GB): `READY` 220 s after
  the launch (target build 41 s, warm pass 13.5 s, acceptance replay 60 s); `json` 96/96, the same divergence indices;
  `/health` `context_limit` 65472; 419,440,704 bytes free per bank.  Left serving the LAN on port 8000
  (`--serve-seconds 86400`); a client on the LAN got its first token 0.30 s after a 33-token prompt.
- Every start above was stopped with SIGTERM between runs and closed its mesh (`result.json` status `stopped`,
  launcher exit 0); no board needed a reset.
