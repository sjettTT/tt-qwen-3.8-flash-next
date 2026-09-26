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
(`<cache-root>/caches/bf4-experts/`, 69 GB in `moe_compute`'s compact expert layout: each layer read from the checkpoint, packed on the host, uploaded to the
mesh and written back as one tensorbin per weight; a `bf4-stage-backbone-NN` phase per layer in the log), then builds
the component and model I/O caches of the chosen context (a few minutes), compiles the kernels (the JIT cache fills
during the warm pass, two to four minutes cold) and captures the decode traces and the prefill chunk trace.  A layer
takes about 27 s on a 4x p150 host (1.4 GB written; measured on the QuietBox 2026-09-25: 26.3-27.4 s each, 49 layers in
1448 s including three layers slowed by a concurrent build, 69,363,701,982 bytes; the previous per-core stride layout
took 35.9-36.9 s and 2.2 GB per layer, 107 GB), the 49 under half an hour;
the payload of every tensorbin is byte for byte the CPU-staged corpus's (`tools/stage_full_bf4_cpu.py`), the manifest records the slot's global shape
(512 experts) while the mesh tensor presents one device's 128.  A machine that bounds a job's wall time can build the
expert cache in pieces: `--prepare-only --bf4-stage-limit N` converts at most N missing layers and stops; every layer
is manifested as it completes (a refused layer publishes nothing), so the next run continues.

The expert cache is keyed by the checkpoint and by the converter's sources (`ttnn/bf4.py`, the `moe_compute` layout
packer, tt-metal's BFP4 packer), not by the tt-metal revision: a rebuilt runtime keeps the cache.  Every start re-packs
one routed expert of the first cached layer from the checkpoint and compares the bytes with the cache (about a second,
the `bf4-cache-admission` phase); a cache converted by different code is refused, the layer, expert and tensor named.
The identity also carries the mesh's DRAM ring as the packing sees it: the bank count and the bank order (the banks
sorted by the worker core that serves them), not the worker coordinates.  `moe_compute` builds every die's program
from the first die's bank-to-worker assignment and the packed bytes never see worker coordinates, so dies harvested
differently share one cache and one conversion (a QuietBox 2 was observed with one die serving its banks from worker
column 5 and three from column 6; its ring workers on that die sit one column from their banks and read the same bank
ids in the same order); a mesh whose dies differ in bank count or bank order is refused before anything is converted.
A cache built by an earlier runtime (keyed by its tt-metal revision, or by the first die's worker coordinates) is
adopted on the first start: its manifest is rewritten and the slot renamed under the new key, nothing is reconverted.

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

The sampled draw runs on the device by default: for `temperature` up to 4, `top_k` 1..32, `top_p`, `min_p` and a
`presence_penalty` in [0, 2] the TAIL trace samples the token from the read candidate row (the request's emitted tokens
are the device's own history), the same law as the host sampler (`docs/NUMERICS.md`, the law gate).  A request with
`frequency_penalty`, `repetition_penalty`, a negative `presence_penalty`, `top_k` 0, a temperature above 4 or `logprobs`
samples on the host over the same candidate row (`qwen38.sampling.sampler` in the response names the path);
`--host-sampler` keeps every request on the host, and an `--mtp` server samples on the host by design (its tail resolves
the greedy token for the draft row and the pass loop's point-mass decision is the host's; `--device-sampler` with `--mtp`
is refused).  The composite device-sampler path (`--device-sampler` before the
one-program sampler) never captured inside the chat server before 2026-09-25: its TAIL capture resolved the greedy row
in a form the warm pass had not compiled; the same change fixes both paths.

Sampled requests and MTP drafting: on an `--mtp` `--sampling` server the pass loop drafts for sampled requests by
default (`QWEN38_MTP_SAMPLED` unset or `1`; `QWEN38_MTP_SAMPLED=0` in the server's environment restores the plain
sampled path, the fused verify alone with sampled requests on the 1-row sampled loop; the launcher passes the variable
through; a server without `--mtp` or without `--sampling` has no drafting for sampled requests and refuses an
explicit `1` at start), by exact speculative sampling: the device drafts by argmax as for a greedy request, and the
host accepts draft `d` at a verify row with probability `p(d)` under the request's own
sampling policy (its temperature, top-k, top-p, min-p and penalties applied to that row exactly as the 1-row loop
applies them), else it samples the row's distribution with `d` removed and renormalised; every emitted token therefore
follows the distribution plain sampling would draw from that row's logits, whatever the draft, and the drafts change
only how many tokens a pass emits.  The rows are the verify rows' logits: the rows path the greedy pass loop decodes
on, which differs from the 1-row row by near-ties (`docs/NUMERICS.md`).  `p` comes from a per-row top-32-per-shard
candidates readback, exact under the same guard as the 1-row row (a row it cannot bound is sampled over the full
vocabulary, counted in
`qwen38.sampling.mtp.fallbacks`).  Requests the rows cannot bound every step (`top_k` 0, a penalty that raises logits)
and `logprobs` requests stay on the 1-row loop; `qwen38.sampling.mtp_drafting` says which loop served the request and
why, `qwen38.sampling.mtp` carries its passes, accepted drafts, draws and fallbacks, and every field of `qwen38.mtp`
(passes, accepted drafts, tokens per pass; with the split verify captured also the sampled passes with their draws
and fallbacks, and `accept_checks`, the greedy passes decided on the host: 0 on the served path) counts that request
alone; `/health.mtp` carries the same
counters cumulative since the server started, `/health.mtp.sampled` the switch.  `/health.mtp.admission` is the DRAM
admission the server opened under: the free bytes per bank the resident build leaves after its captures, the estimate
of the MTP chain's growth per bank by part (`components`: the 49th BF4 pair and the layer's weights beyond it;
`states`: the layer's QSA state at the context and the states of `k`'s verify MoE form, 5 rows for k = 3 and 4, 32 rows
for k = 5; `traces`: one verify form's, plus a verify and a draft trace per further form, up to three; `verify_forms` counts
the forms the chain captures and `verify_forms_captured` names them, `["fused"]` or, with drafting for sampled
requests on, `["fused", "split"]`), every remainder
carrying a 10 % margin (4x p150
line, 2026-09-25: k = 4 with both verify forms 50,468,032 + 12,938,112 + 11,107,904, k = 5 with the split verify
50,468,032 + 20,920,192 + 6,390,144), and `fits`; `/health.mtp.dram_bytes_per_bank` is the growth the open measured,
refused above the estimate.
A `seed` reproduces a stream against the same `system_fingerprint`, which carries the switch and `k`: with drafting
on, the pass loop consumes the request's draws in the pass's order, so the seed reproduces the drafting stream, not
the 1-row loop's stream (the 1-row stream is the `QWEN38_MTP_SAMPLED=0` server's).  With drafting on the chain holds
both verify forms: a greedy request runs the fused verify, the pinned greedy stream (the traces the
`QWEN38_MTP_SAMPLED=0` server runs, bitwise), and a sampled request the split form, the host deciding between its head
and its tail.
`QWEN38_MTP_DRAFTS_PER_REQUEST=1` (a server switch, default off; the launcher passes it through) opens TWO drafting
chains: the `--mtp K` chain and the other member of the pair (4, 5), the k = 5 chain on the 6-row verify MoE form; a
request picks one with `extra_body.mtp_drafts` (`4` or `5`; anything else, or the field on a one-chain server, is HTTP
400 with the admitted list -- the field is refused, never dropped), and the response's `qwen38.mtp.k` says which ran.
The default chain and its stream are untouched (the `system_fingerprint` and the acceptance baselines are the default
chain's; `/health` lists every chain under `mtp_chains`).  Measured 2026-09-26 on the 4x p150 line (256-token requests):
k = 5 pays on structured output -- json +4.7..+6.8 % and code +6.6..+8.8 % tokens per second over k = 4 (5.2 vs 4.55
tokens per pass against a pass 3.1..3.7 ms longer) -- and loses on chat (-12 %) and prose (-25 %), whose k = 5 stream
accepts fewer tokens per pass; the two chains share the model's state, so a request may pick either at any point of a
conversation.  DRAM: the second chain adds its verify window, draft state and traces (about 27 MB per bank at 32k; the
MTP components, the alignment history, the decode step's MTP inputs and the prefill extensions are shared), admitted per
chain at open.  The switch does not combine with `QWEN38_MTP_DEVICE_ACCEPT=1` (the device acceptance's
constants are one k's; the open refuses the pair).
`--mtp-gdn-anchor layer0` (a server flag) re-anchors the layer-0 GDN state from the 1-row recurrence.
Since 2026-09-25 the `--mtp` admission reads the mesh allocator's free bytes per DRAM bank after the resident weights
are built (less the build's own remaining state and traces) and refuses only when the MTP pair, state and growth
estimate for k and the verify forms do not fit there; the 2026-09-04 free-after-captures table is the no-device fallback and is logged, not enforced,
before the mesh opens.
`QWEN38_MTP_DEVICE_ACCEPT=1` (default off; needs the split verify) captures a third form beside the two verify forms:
the head, the device's point-mass acceptance (`fused.mtp_accept`, the device sampler's arithmetic) and the tail in one
trace, for the sampled requests its admission takes (`qwen38.sampling.mtp_acceptance_arithmetic` names the law
realisation, `device-theta` or `host-fp32`); `/health.mtp.device_accept` is the switch, `device_accept_passes` and
`device_accept_guard_deviations` its counters, and `QWEN38_MTP_DEVICE_ACCEPT_DUMP=<dir>` (dev) writes one JSON per
request with every device-decided pass's rows, tokens, statistics and uniforms for the
development-side law gate that re-derives each decision on the host.
`QWEN38_MTP_MOE_ROWS=5|6|32` (diagnostic, default unset) forces the verify MoE row count the chain runs (`moe_rows_for(k + 1)` otherwise: 5 for k = 3 and 4, 6 for k = 5 since the 6-row form's silicon proof of 2026-09-26, its states term provisional until the first served 6-row open re-seeds it), keyed into the admission's states term and reported under `/health` `mtp.moe_rows`; the 6-row form is under proof for k = 5.
When `QWEN38_FUSED` names `gdn_rows_scan` (the verify-rows fold, opt-in) the `states` estimate also carries the fold's persistent prefix states, (k + 1) x 786,432 bytes per GDN layer per device spread over the banks (17,694,720 bytes per bank at k = 4), the figure the line measured the fold's growth against (docs/NUMERICS.md); with `QWEN38_MTP_DRAFTS_PER_REQUEST` every drafting chain allocates its own GDN rows states, so each chain's admission charges its own k + 1.

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
| BF4 expert cache (`<cache-root>/caches/bf4-experts/`) | 69 GB | built once on the first start (69,363,701,982 bytes written on the QuietBox 2026-09-25, the compact expert layout; 107 GB before it), shared by every context, kept across runtime rebuilds |
| 32k context caches | about 23 GB | the converted non-expert weights and the model I/O cache |
| each other allocated context | about 10 GB | the 64k caches measured 9.8 GB on the QuietBox |
| JIT kernel cache | about 1.3 GB | fills during the warm pass, two to four minutes cold |
| host memory, first start | about 10 GB in flight | one MoE layer at a time; 64 GB is comfortable |
| host memory, CPU reference (`tools/run_full_cpu_oracle.py`) | 170-240 GB | not a user step |
| device DRAM free per bank after the captures, 32k | 1,603,483,392 bytes | QuietBox 2026-09-25 with the compact expert layout (largest contiguous 1,602,833,984); 1,531,678,912 with `--mtp 4` (71.8 MB less), 1,569,890,816 with `--long-chunks` (33.6 MB less); with both, the MTP admission takes the 33.6 MB off the free bytes and adds the MTP layer's 128-row chunk extension (measured by the first `--mtp --long-chunks` open); 474,261,568 / 375,594,496 / 439,384,896 before it (2026-09-06) |
| device DRAM free per bank, 64k | 419,440,704 bytes | QuietBox 2026-09-06, before the compact expert layout (which frees a further 1,146,621,952 bytes per bank at 32k) |
| device DRAM free, 256k | about 750 MB per device | QuietBox 2026-09-06, before the compact expert layout; single-user; MTP did not fit then (94 MB free per bank against the 128 MiB contiguous it needs) |
| the prompt-end snapshot | ~53 MB per device | resident; the recurrent part of the device state (above) |

## The server behind uvicorn (the container form)

`tools/qwen38_asgi.py` is this server as an ASGI application, the form a container that starts its model with uvicorn
runs:

    python -m uvicorn --lifespan on models.demos.blackhole.qwen38_flash_next.tools.qwen38_asgi:app

The lifespan startup composes the server's arguments from the environment, starts `qwen38_chat_server` as a child
process (`python -m`, the composed arguments and environment: the server keeps its own main thread, signal handlers and
device teardown exactly as the shell launcher runs it, and the uvicorn process never touches a device) and returns only
once the server has written `READY` (the mesh open, the chain captured, the acceptance records replayed with
`--require-json-96` in the default flags), so uvicorn's `Application startup complete` line means what `READY` means; a
server that ends before the record fails the startup with its status.  The shutdown sends the server SIGTERM (its drain,
then the chain's release and the mesh close in the server's order) and waits for its exit, logged with its status; a
server that ends on its own (the stall watchdog's exit 1) ends the uvicorn process with the same status, so the container
exits as the server did.  The server is a child rather than a thread because tt-metal's teardown at process exit must run
in the process and thread that opened the devices: with the server in a thread of the uvicorn process the exit unlocked
the UMD chip lock from the wrong thread and the process died with status 139 after an otherwise clean close, and a
container that exits 139 is what `tt-model stop` answers with a board reset; as a child the container exits 0 after
SIGTERM (a gate of the package).  Requests are forwarded byte for byte over the loopback
(`QWEN38_INNER_PORT`, 18000): nothing of a request or a reply is parsed there, so the rules above apply unchanged (what
the server cannot honour it refuses with HTTP 400; nothing is dropped), a streamed reply reaches the client as the
server writes it, and a client's hang-up closes the server's connection too.  Before `READY` and after the stop every
request gets HTTP 503 in the server's error shape.  The environment: `QWEN38_HARDWARE_PROFILE` (`p150-line` default;
`tt-quietbox`, `tt-quietbox-2`), `QWEN38_ALLOCATED_CONTEXT` (32768 default, 65536, 131072, 262144),
`QWEN38_SERVER_ARGS` (further server flags, shell-split; default `--mtp 4 --sampling --stall-seconds 300
--require-json-96`; a flag the server does not know is refused by its parser), `QWEN38_CACHE_ROOT` (`/tensor-cache`
default; the launcher's layout under it), `QWEN38_CHECKPOINT` (unset: the pinned revision of `HF_MODEL` in the local
Hugging Face hub cache), `QWEN38_STOP_SECONDS` (300).  The device nodes are the four `/dev/tenstorrent` entries the
container was given (`--device-nodes` when they are not the profile's own); `TT_VISIBLE_DEVICES`, the mesh graph
descriptor, `QWEN38_HARDWARE_MODE` and `TT_METAL_TRACE_ALLOC_TRACKING` are set before the server starts, as the shell
launcher sets them.  A tree without git history (an image ships none) is admitted by `tools/runtime_admission.py` only
with `QWEN38_TT_METAL_SHA` naming the commit it was built from (`source` `declared` in the identity); a checkout with
history refuses a declared sha that differs from its head.

The tt-model container package of this tree (`sjettTT/qwen3.8-flash-next_p150x4`) serves this form since 2026-09-26
(kind `tt-dit-server`: uvicorn starts the application above; the profiles `c32k`, `c64k`, `c128k`, `c32k-quietbox`,
`c32k-quietbox2` set the context and the hardware profile, every one with `--mtp 4 --sampling`); the package's vLLM
form (below) is its previous revision.

## Serving under vLLM

`tools/qwen38_vllm.py` is the model class vllm-tt-plugin drives (`Qwen38ForCausalLM`): the same traced chain as this
server, opened with the sampling epilogue, one resident slot, every token's full-vocabulary logits read back for vLLM's
sampler.  The plugin owns the mesh, the scheduler, the tokenizer and the OpenAI API; the adapter owns the device:
`prefill_forward` resets the state and runs the chunk driver over the prompt, `decode_forward` teacher-forces the token
vLLM sampled, both return CPU fp32 `[1, 1, 248320]` logits with the 243 LM-head padding rows at `-inf` (under the
plugin's device-sampling contract, below, a sampled decode step returns the token instead).  The venv is the plugin's
`docs/install-vllm-tt.sh` over this checkout's `python_env` (vLLM 0.26.0 built with `VLLM_TARGET_DEVICE=empty`, the
plugin editable), then `transformers==5.16.1` (the `qwen4_exp` config class).  From a built checkout `$REPO`:

    export TT_METAL_HOME=$REPO PYTHONPATH=$REPO:$REPO/ttnn:$REPO/tools PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
    export TT_VISIBLE_DEVICES=0,1,2,3 MESH_DEVICE="(1, 4)" TT_METAL_TRACE_ALLOC_TRACKING=1
    export MODEL_WEIGHTS_DIR=$CKPT QWEN38_CACHE_ROOT=<cache-root> TT_METAL_CACHE=<cache-root>/jit-cache
    export EXTRA_MODELS_DIR=$REPO/models/demos/blackhole/qwen38_flash_next/tools/vllm_bundle
    python -m models.demos.blackhole.qwen38_flash_next.tools.prewarm_ple_table --checkpoint $CKPT
    python <plugin>/examples/server_example_tt.py --model $CKPT --served-model-name Qwen/Qwen3.8-Flash-Next \
        --max_num_seqs 1 --block_size 64 --max-model-len 32704 \
        --hf-overrides '{"architectures": ["TTQwen4ExpForConditionalGeneration"]}' \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        --additional-config '{"tt": {"fabric_config": "FABRIC_1D", "l1_small_size": 24576, "trace_region_size": 0,
                                     "sample_on_device_mode": "decode_only"}}'

`$REPO/tools` on `PYTHONPATH` is the `tracy` package the extension imports at start (`import ttnn` fails without it).
`--max-model-len` picks the smallest resident context whose limit (the capacity less 64) holds it, 8128 to 262080;
unset, vLLM resolves 262144 and the adapter refuses.  `--max_num_seqs` is the underscore spelling (`server_example_tt.py`
parses only that form).  The tt block reproduces this server's mesh parameters; `"sample_on_device_mode": "decode_only"`
is the plugin's device-sampling contract (the sampling row of the table below; without it vLLM's full-row sampler runs
on the host).  `/health` answers 3-5 minutes after the launch from warm caches.  The adapter's own settings come from
the environment, every one checked before the device is touched:

| variable | default | meaning |
|---|---|---|
| `MODEL_WEIGHTS_DIR` | unset | the checkpoint directory; unset, `--model` is taken as a directory, else as a repo id (default `Qwen/Qwen3.8-Flash-Next`, `QWEN38_HF_REPO` overrides it) whose pinned revision must already be in the Hugging Face hub cache under `HF_HOME`: no download happens at start |
| `QWEN38_CACHE_ROOT` | required | this server's cache layout: `caches/<label>/{components,model-io}` and `caches/bf4-experts` |
| `QWEN38_CACHE_LABEL` | `c<context>-vllm` | the component / model-io cache label, one per resident context |
| `QWEN38_BF4_CORPUS`, `QWEN38_BF4_CORPUS_VERIFICATION` | unset | a CPU-staged BF4 expert corpus in place of the cache under `caches/bf4-experts`, which must otherwise exist: its first-start conversion runs inside vLLM start-up only with `QWEN38_VLLM_ALLOW_BF4_CONVERSION=1` |
| `QWEN38_PREFILL_SLAB` | `2048` | the prefill slab's rows, a multiple of 128 in 256..4096: a prompt runs as `N // rows` slabs, then 128-row chunks, then 32-row chunks (`docs/PREFILL.md`); `0` or `off` serves without the slab; any other value is refused by name |
| `QWEN38_LONG_CHUNKS` | unset | with the slab off, `1` keeps the 128-row chunks ahead of the 32-row ones (a slab implies them) |
| `QWEN38_TT_METAL_SHA` | unset | the cache identity's tt-metal sha (40 lowercase hex); unset, the checkout head, else the runtime extension's digest prefix, so a checkout without `.git` or `git` serves |

The slab's resident dense weights are admitted against the free DRAM when the chain opens (`ttnn/prefill_dense.py`):
a refusal ends the start-up with the admission's numbers and the sentence `QWEN38_PREFILL_SLAB=0 serves this context
without the slab`, never a silent fallback (the serving profile decides per context).  The adapter logs the form it
opened (`prefill form: 2048-row slabs, then 128-row chunks, then 32-row chunks`), so a served number is attributable
to it.  Numerics: the slab body is tolerance-class against the 32-row chunk bodies (`docs/PREFILL.md`, "Numerics
class": each dense linear's output within one bf16 ULP of its scale, top-1 moving at a handful of positions on prompts
longer than a slab), the 128-row chunks give the 32-row chunks' tokens (`docs/NUMERICS.md`), and the `json` acceptance
record (85 prompt tokens, shorter than one 128-row chunk) stays bitwise in every form.  Prefill rates measured on the
standalone server (README, 2026-09-25): 3.0-3.5 ms per prompt token in 32-row chunks, 1.45-1.56 with the 128-row
chunks, 0.74-0.90 with 2,048-row slabs (a 31,716-token prompt in 23.6 s, 2,118 tokens in 1.91 s).

| capability | under vLLM | note |
|---|---|---|
| batch | 1 | one resident slot: `--max_num_seqs 1`, no data parallelism |
| contexts | 8k, 32k, 64k, 128k, 256k | by `--max-model-len`; 256k is single-user |
| sampling | vLLM's host sampler over the full row; or, with `"sample_on_device_mode": "decode_only"` in the `tt` block of `--additional-config`, the adapter's host sampler over the row's top 1024 candidates | temperature, top-p, top-k, min-p, penalties, `logit_bias`, logprobs, structured output; a greedy request reproduces this server's greedy stream (the `json` acceptance record matches the CPU reference 96/96; the other records diverge at the pinned A3 indices). vLLM's `Sampler` sorts the 248,320-wide row on every sampled (non-greedy) decode step, about 22 ms per token; under `decode_only` the plugin hands the request's temperature / top-k / top-p / seed / penalties to `decode_forward` and the adapter draws the token itself from the same row with vLLM's sampler semantics (same distribution, not the same token stream per seed), well under a millisecond; greedy stays the argmax. Requests with min-p, `logit_bias`, `bad_words`, `allowed_token_ids`, `min_tokens`, logprobs or structured output keep vLLM's full-row sampler (the plugin decides per step); prefill stays on it too (`"all"` is refused) |
| prefill | the slab and chunk traces over the whole prompt | `QWEN38_PREFILL_SLAB` above; vLLM's chunked prefill and prefix caching are declared unsupported (the plugin disables them) |
| decode | one traced step per token | `trace_mode` other than `all` is not honoured |
| follow-up turns | re-prefilled | the prompt-end snapshot and prefix reuse are not used |
| KV | the chain's resident caches | vLLM's block table and KV cache are accepted and ignored |
| MTP, the on-device sampler (`--device-sampler`), async decode | no | `--mtp` and the on-device sampler stay on this server; under vLLM the sampled token is drawn on the host (`decode_only`, above) |

The tt-model container package of this tree (`sjettTT/qwen3.8-flash-next_p150x4`) ran this path until 2026-09-26
(its revision `484576b04b11` on the Hub; the current package serves the chat server through uvicorn, above) with one profile per
box: `c32k`, `c64k`, `c128k` (no mesh graph descriptor: ttnn's auto-discovery, a 1x4 line on a 4x p150 host and, on a
QuietBox 2, the four dies in the fabric's order), `c32k-quietbox` (the 4x p150 TT-QuietBox: exports
`tools/qb_p150_x4_1x4_line_mesh_graph_descriptor.textproto`, 4 ethernet channels per link, so it refuses a QuietBox 2
at fabric init with `Expected 4 eth links`) and `c32k-quietbox2` (a TT-QuietBox 2, 2x p300c: `hardware p300x2`,
`MESH_DEVICE=P300x2`, exports `tools/qb2_p300_1x4_line_mesh_graph_descriptor.textproto`, 2 channels per link).  Under
the plugin the mesh is opened without a physical order, so the 1x4 order is the fabric's embedding (a QuietBox 2 on
2026-09-21: devices [1, 0, 3, 2], an ethernet path) and no route is derived.  Measured on a QuietBox 2 on 2026-09-21
through `tt-model serve` with `c32k-quietbox2` and with the default `c32k` (the same results): ready in about 100 s from
warm caches (241 s with cold component caches); sampled decode under the serving default (the checkpoint's
generation_config profile, temperature 1.0, top_k 20, top_p 0.95) 44 to 46 ms per token on streamed 128-, 200- and
256-token requests, about 22 tokens/s, TTFT 0.21 s on a 17-token prompt; one DRAM reader per bank on the mixed-harvest
dies.

Cost: this server's greedy loop is 50 ms per token; under vLLM every step adds the full-vocabulary gather, vLLM's
sampler and its step overhead: 56.7 ms per token greedy and 79 ms sampled, measured on 4x p150 (2026-09-09).  The
plugin's own host suite with the bundle set, in the serving venv (expected: 368 passed with transformers 5.16.1, and the
line `Registered TT model TTQwen4ExpForConditionalGeneration`):

    EXTRA_MODELS_DIR=$REPO/models/demos/blackhole/qwen38_flash_next/tools/vllm_bundle PYTHONPATH=<plugin>/ci/host-stubs \
        python -m pytest <plugin>/tests --ignore=<plugin>/tests/tt --log-cli-level=INFO
