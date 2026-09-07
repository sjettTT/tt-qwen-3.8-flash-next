# The server in detail

What the README's sections 4 and 5 leave out: the launcher's environment, the runtime admission, the BF4 expert
cache, the request rules in full, the follow-up-turn mechanics, the serving contract under hang-ups and wedges, the
`/health` fields, and disk and memory.

## The n-gram table pre-warm

Every decode token reads sixteen 320-byte rows of the PLE n-gram table, which stays in the checkpoint (104 GB in 33
shards of layer 1) and is read by the host through the page cache; on a cold cache each row is an NVMe page-in, so on a
host whose RAM holds the table (the QuietBox has 503 GB) read it once before the start:

    python_env/bin/python -m models.demos.blackhole.qwen38_flash_next.tools.prewarm_ple_table --checkpoint /data/Qwen3.8-Flash-Next

It prints the residency before and after (`fincore`, util-linux) and the read rate; nothing is written.  On the
QuietBox the pass over the 104 GB took 13 s from a warm cache (7.8 GB/s); a cold NVMe cache is 35-60 s at 2-3 GB/s.
`--report-only` only prints the residency.

## What the launcher does

`tools/run_qwen38_chat_server.sh` prints the checkout it runs from (its commit, whether the tree is modified), the
interpreter and the `ttnn` extension, then the profile, the device set, the context and the run directory, and starts
the server with `--sampling` (sampled requests are served; a request naming no sampling field is still the bitwise
greedy stream) and `--stall-seconds 300` (the watchdog, below); `--no-sampling` serves greedy requests only (+0.3 ms
per token saved, sampling fields refused with HTTP 400) and `--stall-seconds 0` disables the watchdog.

What the launcher does not do: no device locks, no runtime archives or digests.  It exports the QuietBox mesh graph
descriptor for `tt-quietbox` (`tools/qb_p150_x4_1x4_line_mesh_graph_descriptor.textproto`: the four chips' ethernet
ring opened as one 1x4 line), the device set, the cache and log roots, and `TT_METAL_HOME` = this checkout.

## Runtime admission

`tools/runtime_admission.py` admits the runtime before the mesh opens: the interpreter's `ttnn` package and its
compiled extension must resolve under this repository (`ttnn` built from this checkout), the checkout's `git`
head and tree are read, the extension is hashed.  That identity (`{repo, head, tree, dirty, extension,
extension_sha256}`) becomes the builder provenance (`tt_metal_sha` = the head, `ttnn_runtime_sha256` = the digest),
the cache namespaces, the `system_fingerprint` of every response and the `runtime` block of `result.json`.  A
modified tree is admitted and recorded as `dirty`.  A `ttnn` from elsewhere is refused with both paths printed.  The
CPU preparation (`tools/live_decode_diagnostic.py`) derives the consumer identity from that runtime identity, the
pinned checkpoint digests and the mesh order; the BF4 experts come from the production cache described below, or,
with `--bf4-corpus DIR --bf4-corpus-verification FILE`, from a corpus staged on the CPU by
`tools/stage_full_bf4_cpu.py` and verified by `tools/verify_full_bf4_cpu.py` (`diagnostic_bf4.py` binds it; the
producer identity is what the verification record claims, `--bf4-producer-identity` pins it).

## The BF4 expert cache (the first start)

The first start converts the routed experts of all 49 MoE layers into the BF4 cache
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

Warm starts reach `READY` in about five minutes: the weights load, the traces are captured, the acceptance prompts
replay against the CPU (`NUMERICS.md`), then the server listens.

## The run directory and `/health`

The run directory (`<cache-root>/runs/<stamp>/`) holds `READY`, `phase-markers.jsonl`, `requests.jsonl`,
`acceptance.json` and, at shutdown, `result.json` and `STOPPED`.  `result.json` and `/health` carry the runtime
identity: the checkout's commit and tree, the extension's SHA-256, the route the mesh opened in.

`GET /health` reports the context limit (`context_limit`: the allocated context minus 64), the sampling mode and the
sampling profile defaults (`sampling_defaults`), the free DRAM after the captures, the runtime identity, the defaults
(`defaults.system_prompt` is null), the `logprobs_normalizer`, and `current_request`: the request holding the device
with the seconds since its last completed step.

## Requests

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

## The prompt, follow-up turns and the prompt-end snapshot

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

## Hang-ups, stalled readers, deadlines

A client that hangs up is noticed at the next device step (or prefill event) whether or not anything was being
streamed to it, and a queued request whose client left gives up its place: the device never runs a request for
nobody.  A streaming request gets its head and role chunk as soon as it is admitted and an SSE comment
(`: keepalive`) every `--heartbeat-seconds` (30 s) through the queue wait and the prefill, so a 60 s proxy or SDK
read timeout does not cut a long prompt.  A socket write blocked for `--socket-timeout-seconds` (60 s: a reader that
stopped reading) ends the request as `disconnected`.  `--request-deadline-seconds` (off by default) is honoured
inside the chunked prefill too.  `/health.current_request` shows the request holding the device with the seconds
since its last completed step.

## The stall watchdog and the stop signal

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

## Disk and memory

| what | size | notes |
|---|---|---|
| the checkpoint | 360 GB | 131 safetensors shards, the tokenizer, the chat template; 360,023,351,829 bytes in 145 files |
| the n-gram table inside it | 104 GB | 33 shards of layer 1, read through the page cache at decode; pre-warm it on a host whose RAM holds it (above) |
| BF4 expert cache (`<cache-root>/caches/bf4-experts/`) | 107 GB | built once on the first start (100 GB written on the QuietBox), shared by every context, kept across runtime rebuilds |
| 32k context caches | about 23 GB | the converted non-expert weights and the model I/O cache |
| each other allocated context | about 10 GB | the 64k caches measured 9.8 GB on the QuietBox |
| JIT kernel cache | about 1.3 GB | fills during the warm pass, two to four minutes cold |
| host memory, first start | about 10 GB in flight | one MoE layer at a time; 64 GB is comfortable |
| host memory, CPU reference (`tools/run_full_cpu_oracle.py`) | 170-240 GB | not a user step |
| device DRAM free per bank after the captures, 32k | 474,261,568 bytes | QuietBox 2026-09-06; 375,594,496 with `--mtp 4` (98.7 MB less), 439,384,896 with `--long-chunks` (34.9 MB less) |
| device DRAM free per bank, 64k | 419,440,704 bytes | QuietBox 2026-09-06 |
| device DRAM free, 256k | about 750 MB per device | single-user; MTP does not fit (94 MB free per bank against the 128 MiB contiguous it needs) |
| the prompt-end snapshot | ~53 MB per device | resident; the recurrent part of the device state (above) |
