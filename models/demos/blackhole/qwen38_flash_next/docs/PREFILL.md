# Prefill: the chunk bodies and the opt-in slab

The server prefills a prompt through traced chunk bodies on the same device state the decode traces use: 32-row
chunks (the default), 128-row chunks ahead of them with `--long-chunks`, and, with `--prefill-slab ROWS`, slabs of
ROWS rows (a multiple of 128 from 256 to 4096; 2048 is the measured form) ahead of the 128-row chunks.  Whatever the
chunk sizes, the last prompt token is teacher-forced through the decode traces, so the decode, the MTP hand-off and the
prompt-end snapshot see the same objects: the committed GDN recurrent state (fp32) and conv history, the QSA KV and
compressed caches, the staging tile and raw-key ring, the PLE history and the device position.

## The slab (`--prefill-slab 2048`)

Inside a slab every layer runs its ROWS rows in one pass:

- every dense linear (GDN in/out projections, the gated-residual down+inject and up, the MoE router and shared experts,
  the QSA index / query-gate / key / value / output projections) runs as one 2D-multicast matmul over the rows on an
  interleaved copy of its resident weight (the decode weights stay DRAM-width-sharded; the copy is made and released
  inside the slab body, 0.04-0.15 ms each);
- the GDN kernel (`chunk_gated_delta_rule`) takes the whole slab in one call (ROWS / 32 sub-chunks, the recurrent
  state carried inside the kernel in fp32) and commits its final state; the FIR taps are row shifts;
- the routed experts run in one `moe_compute` call over the whole slab on the op's local output path (each expert's
  tokens packed into 32-row chunks once per slab, its outputs written straight into the slab's `[10, ROWS, 2560]`
  page), and that page is reduced in 512-row blocks; `QWEN38_MOE_SLAB_ONE_CALL=0` restores the 16 calls of 128 tokens
  through the 128-row instance's own call, tilize and weighted reduce (*The routed experts in one call* below);
- the QSA layer writes one ROWS-row KV slab and ROWS / 128 compressed tiles through one page table, scores and selects
  its blocks in 512-row blocks (the indexer, the score all-reduce, a broadcast causal block mask, `topk_large_indices`),
  and runs `sparse_sdpa` over all ROWS query rows in one call.

The remainder of a prompt after the slabs runs through the 128-row chunks, then the 32-row chunks and the padded tail,
then the ordinary hand-off from the 32-row state.  `--prefill-slab` implies `--long-chunks`; like `--long-chunks` it is
not combined with `--mtp` (the MTP chain prefills in 32-row chunks).  The 32-row and 128-row bodies are unchanged.

## Numerics class

The slab is **tolerance-class** against the chunk bodies: a 2D-multicast matmul accumulates K in another block order
than the per-tile DRAM-sharded program, so each linear's output differs from the chunk form by up to one bf16 ULP of its
scale (58 % of the elements one step apart, measured on 4x p150 at 2048 rows); the GDN kernel over the slab is bitwise
the chained 128-row calls, and the slab's other ops (the KV writes, `sparse_sdpa`, the block selection, the head sum)
are bitwise per row.  The acceptance mechanism is the same as for the other prefill modes (`NUMERICS.md`): `json` must
reproduce the CPU record 96/96 and the eleven divergence indices are reported against the pinned table.  The twelve
acceptance prompts are shorter than one slab, so the slab body itself is exercised by prompts of 2048 tokens and more
(the agreement corpus's long items, a long system prompt).

## What it costs and what it saves

Per device, per 2048-row slab (measured on 4x p150, 2026-09-09): a dense linear at 2048 rows runs 10-12x faster than
its 64 per-tile calls (GDN in-proj 6.03 -> 0.49 ms + 0.15 ms for the weight copy; QSA query-gate 4.28 -> 0.48); the GDN
kernel 1.34 ms in one call against 2.47 ms chained; `sparse_sdpa` at S = 2048 10.99 ms against 13.31 for 16 x 128; the
KV slab write 0.056 ms against 0.206.  What did not change with the row count in these measurements is the MoE
expert stream (16 calls of 128 tokens per layer per slab, the most the kernel then admitted per call) and its per-call
combine tilize and weighted reduce, together about 0.4 ms per prompt token: the slab's floor at the time.  The
tables of this section and the next were measured in that form; *The routed experts in one call* below is the
default since 2026-09-25 and lowers the floor.

Measured end to end on 4x p150 (2026-09-09, 32k context; the 6942-token prompt is 3 slabs, 6 long chunks, 2 short
chunks and a 30-row tail, so its rate blends the slab's with the remainder's):

| | 32-row chunks | `--long-chunks` | `--prefill-slab 2048` |
|---|---|---|---|
| ms per prompt token (6942-token prompt) | 3.3 | 1.55 | 1.04 (TTFT 7.27 s) |
| json acceptance | 96/96 | 96/96 | 96/96 |
| divergence table vs the pinned table | pinned | identical | identical |
| decode after the prefill | 19.9 tok/s | 19.9 tok/s | 19.7-20.1 tok/s |
| HF agreement, all 8416 positions (top-1 / clear top-1 / KL) | 0.9518 / 0.9607 / 0.0543 | same | 0.9515 / 0.9605 / 0.0544 |
| HF agreement, the two long windows (160 positions) | 0.9938 / 0.9938 / 0.014 | same | 0.9812 / 0.9812 / 0.0225 |

The agreement column is the server's `--agreement-reference` run against the HF reference; every part except the
two long windows (an 8192- and a 32704-token prompt, the only items longer than a slab) is bitwise the chunked
column, and the long windows moved two of 160 top-1 positions (the tolerance class above).  Ready after 214 s on a
warm cache (the slab's warm pass through the 48 layers takes 31 s and its capture 5.5 s).

Long prompts (one chat completion each, the prompt a body of natural text, `max_tokens` 64; TTFT is the request's
time to its first token, the rate the server's prefill ms per prompt token):

| prompt tokens | `--long-chunks` TTFT / ms per token | `--prefill-slab 2048` TTFT / ms per token (32k context) | the same at a 64k context |
|---|---|---|---|
| 4,179 | 6.36 s / 1.51 | 4.17 s / 0.98 | 4.40 s / 1.04 |
| 8,176 | 12.3 s / 1.50 | 8.85 s / 1.08 | 9.22 s / 1.12 |
| 16,322 | 24.4 s / 1.49 | 15.7 s / 0.96 | 16.5 s / 1.01 |
| 31,988 | 47.4 s / 1.48 | 28.0 s / 0.87 | 29.8 s / 0.93 |
| 64,655 | - | - | 58.1 s / 0.90 (31 slabs) |
| 127,872 | - | - | 151 s / 1.18 at a 128k context (62 slabs; measured before the input pipeline below) |

Where the slab's time goes: the device runs one slab in 1.6-1.8 s (0.78-0.88 ms per token).  The traced body is
19,161 programs; under the device profiler on 32,000 tokens of natural text it is kernel-bound at 1.52-1.60 s per
slab (1.58 s of kernel time at P = 0, 1.51 s at P = 28672): the 768 `moe_compute` calls 0.25-0.31 ms per token (their
time follows the experts the tokens select: 0.20 on a slab of repeated tokens), the collectives 0.11, the element-wise
ops 0.06, the dense matmuls 0.045, `sparse_sdpa` 0.04-0.06, the combine tilize + weighted reduce 0.07; what the server
measures beyond the body (under 0.1 s per slab) is the input copies and the dispatch of the replay.  The host's work
per slab is the preparation of the next slab's inputs (the n-gram lookup of 2048 tokens, read from the table in one
batch, and the row packing: 0.1-0.25 s, against 0.36 s when the rows were read one token at a time) and the enqueue
of their copies, a few ms.  The driver overlaps that preparation with the running replay: the copies are queued behind the
replay on the same command queue (in order, so the device finishes reading the input buffers before they change), and
the driver waits for slab k - 1's event only once slab k's replay is queued, then prepares slab k + 1 while slab k runs.
Only the first slab's preparation is exposed, so a prompt of N slabs costs N device periods plus one preparation; the
served tokens are bitwise those of the un-overlapped driver (the same bytes reach the same buffers before the same
replay).

The slab takes 41 percent less time per prompt token than the 128-row chunks at 32k (0.87 against 1.48 ms: TTFT 28.0 s
against 47.4 s) and 35 percent less at 4k.  The slab body itself runs at about 0.86-0.88 ms per prompt token at 32k
(about 1,150 prompt tokens per second); the shorter prompts pay more for their remainder (the 128- and 32-row chunks
after the last slab), the first slab's preparation and the fixed hand-off.  A 64k context (`--allocated-context 65536`)
costs about 4 percent more per token (the QSA selection scores over twice the blocks) and 56 MB per DRAM bank more; a
128k context about 13 percent more per token (226 MB per bank left); the slab state and trace themselves cost 45 MB per
bank at any context.

Memory: the slab state adds about 200 MB per device at 32k (one set of GDN pass buffers shared by the 35 GDN layers,
the PLE rows, the QSA slab constants and kept rows, the slab trace); the transients inside a layer peak around
150 MB (about 350 MB at 256k, where the score all-reduce per 512-row block is the largest).

## The routed experts in one call (`QWEN38_MOE_SLAB_ONE_CALL`, `QWEN38_MOE_SLAB_RINGS`)

Since 2026-09-25 the slab routes all of its rows through one `moe_compute` call per layer (`QWEN38_MOE_SLAB_ONE_CALL`
unset or `1`) on the op's local output path: each expert's tokens are packed into 32-row chunks once per slab (223
chunks per layer per device on natural text against 745 in 128-row blocks), the op writes each token's expert output
straight into the slab's `[10, 2048, 2560]` page, and the weighted reduce runs over that page in 512-row blocks.  The
rows of the experts a device does not hold are not zero-filled: the page is zero at allocation and only ever holds
finite expert outputs, which the reduce multiplies by an exact 0 where unowned.  The one call is bitwise the 16 x
128-row blocks: on 4x p150 the twelve acceptance records, the 3232 agreement rows and the four long-prompt completions
are identical at three heads.  `QWEN38_MOE_SLAB_ONE_CALL=0` restores the blocks.

Per device, per layer, per 2048-row slab on one p150 with a captured natural-text routing, the expert stream takes
3.25 ms in one call against 13.45 ms in the 16 calls.  Under the device profiler on the 4-chip line (P = 0, a 2048-row
slab of a 32k record, one chip) the slab's kernel time falls from 1425.3 ms (`moe_compute` 510.1 ms over 768 calls,
19,161 programs) to 1044.9 ms with the one call (`moe_compute` 158.7 ms over 48 calls, 13,401 programs; -26.7 percent,
1,426 -> 1,945 prompt tokens per second of kernel time) and to 971.4 ms with two rings (`moe_compute` 98.3 ms; -31.8
percent, 2,092 tokens per second).  Beside `moe_compute` the one call saves the per-block combine tilize (-25.6 ms,
988 -> 412 calls) and the combine reduce-scatter (-21.9 ms; -35.5 with two rings) and adds 21.3 ms of `slice` (the
reduce's fewer, larger k-strided slices of the one-call page); the page costs 12.2 MB more DRAM per bank.  Served
(4x p150, 2026-09-25, 32k context, natural-text prompts, `max_tokens` 64, TTFT the request's time to its first token
and the rate the server's prefill ms per prompt token):

| prompt tokens | 16 x 128-row blocks | one call, one ring (`QWEN38_MOE_SLAB_RINGS=0`) | one call, two rings (the default) |
|---|---|---|---|
| 2,118 | | 1.59 s / 0.736 | 1.54 s / 0.712 |
| 2,764 | | 2.40 s / 0.855 | 2.31 s / 0.826 |
| 25,546 | | 14.75 s / 0.576 | 13.90 s / 0.543 |
| 31,716 | 24.73 s / 0.779 | 18.09 s / 0.569 | 17.06 s / 0.537 |

At 32k the one call on one ring takes 27 percent less time per prompt token than the blocks and the two rings 31
percent less (the blocks were measured at the longest prompt only in this pass).  The line gate of 2026-09-25 with the
ring exchange's backpressure credit in place (the served kernels of this landing) measured, per 2048-row slab under
the profiler, 1041.7 ms of kernel time on one ring (1,951 prompt tokens per second) and 971.6 ms on two rings
(2,090); served, 31,716 tokens reached their first token in 18.03 s (one ring) and 17.04 s (two rings), 2,118 tokens in
1.58 and 1.53 s; the acceptance records, the agreement rows and the completions were identical across the blocks, the
one ring and the two rings, and the decode after the prefill identical to the lineage's own run.

`QWEN38_MOE_SLAB_RINGS` selects how the one call streams the expert weights; only the one-call slab reads it, the
32- and 128-row chunks and decode never do.  Two rings are the default since 2026-09-25 (the ring exchange's
backpressure credit landed the same day); `QWEN38_MOE_SLAB_RINGS=0` restores one ring:

- `2` (unset): the chunks are split over two rings of cores, each reading the slices of the experts it owns.
  Bitwise on the 4-chip line at 32k in three runs (the acceptance records, agreement rows and completions identical
  to the one-ring form's); 73.5 ms less kernel time per slab under the profiler (971.4 against 1044.9 ms);
- `0`: the op's one-ring, three-slot weight stream (the form the one call was first gated with);
- `1`: one ring with each expert's weight slice read from DRAM once per slab (a replay ring); bitwise the one-ring
  stream on one die, not measured on the 4-chip line;
- `3`: refused by the switch.  The op implements three rings, but on the 4-chip line their output was
  nondeterministic (2026-09-25: two runs of the same prompts differed from each other and from the default in
  different places while every other form was identical); the cause is under investigation.

## Served rates

The README's prefill row, served through the chat server on 4x p150 at the 2026-09-25 head (the routed experts in one
call on two rings by default): the 32-row chunk trace at 2.4 ms per prompt token (413 / 418 prompt tokens per second in
two runs: about 410), flat from 2k to 261k tokens; `--long-chunks` at 1.27-1.33 ms (750-790); `--prefill-slab 2048` at
0.53 ms (1,870: TTFT 17.0 s for a 31,716-token prompt, 1.53 s for 2,118 tokens; the table above).  The release's
long-context figures of 2026-09-04, through the 32-row chunk trace alone at about 3.2-3.5 ms per prompt token: a 40k
prompt reached its first token in 125 s and a 200k prompt in 671 s, and decode stayed at 17-19 tokens/s to 256k (the
decode of that day; the README's decode rows are the current step).

## Glue forms (`QWEN38_PREFILL_GLUE`)

Read program by program, the slab body's non-MoE time is glue around a few kernels: the gated-residual read's
gather, the QSA block selection's masks and score sums, the GDN q/k preparation.  The glue forms are other
arrangements of the same ops for those terms, read only by the slab body (the decode path and the 32/128-row bodies
never see one).  Bitwise forms keep every output element's arithmetic and reduction order; tolerance forms move a
summation order or a rounding point and are judged like the slab itself (`NUMERICS.md`, the long windows against the
references).

The two bitwise forms run by default.  Measured on the 4-chip line (1x4 p150, a 32k context, `--prefill-slab 2048`,
2026-09-25) against the previous slab body in the same process form: the acceptance pins 12/12 identical; the device
column of the agreement corpus's two long windows identical at all 160 scored positions (KL 0.0, top-1 1.0 between
the two columns; both 0.9812 / 0.0225 top-1 / KL against the HF reference); the slab's kernel time 1584.6 -> 1524.3 ms
per 2048-row slab (-60.3 ms, -3.8 %: the gather 34.0 of it, the hoist 26.5, additive to 0.2 ms), 1283 -> 1334 prompt
tokens per second on the body, TTFT at 32k 27.57 -> 26.64 s (-3.4 %), 220 fewer programs per slab.  On the landed
base (BF8 dense weights, the compact expert layout, two DRAM readers per bank; 2026-09-25) the same pair measured in
the served path: TTFT 24.55 -> 23.64 s at 31,716 tokens (0.773 -> 0.744 ms per prompt token, -3.7 %) and 1.979 ->
1.910 s at 2,118 tokens, with the acceptance pins 12/12, all 3,232 scored positions of the agreement corpus and the
four probe completions identical between the two bodies.
`QWEN38_PREFILL_GLUE=today` restores the previous slab body; `QWEN38_PREFILL_GLUE=<name>[,...]` runs exactly the
named forms (a list that wants a default form names it); an unknown name, an exclusive pair or `today` beside a name
refuses to start.

- `gr_gather_generic` (bitwise, default) / `gr_gather_tuned` (bitwise, exclusive with it): the gated-residual read
  gathers its fp32 partials with the generic `all_gather` (0.23 ms per read at 42 GB/s of ingress), or with the async
  op at two workers per link and ten chunks per sync, instead of `today`'s async op at one worker and one chunk per
  sync (the slowest collective of the slab: 0.59 ms per read at 16 GB/s of ingress, 96 reads per slab).  The gathered
  bytes and their page order are the same; the reduce that follows reads the same tensor.
- `qsa_mask_hoist` (bitwise, default): the QSA block selection's causal block masks depend on the position and the row
  index only, so they are derived once per slab and read by all twelve QSA layers instead of derived in each (26.5 ms
  per slab of mask ops).  The masks hold 1 KiB per token of context at 2048 rows: 32 MiB at a 32k context, 64 MiB at
  64k, 128 MiB at 128k, 256 MiB at 256k.  The hoist is admitted once, when the slab's QSA chunk constants are built
  after the model (`hoist_masks` on them, with the numbers; one log line at the build names the decision and
  its numbers in the same words): the masks must be within the 64 MiB cap
  (`HOISTED_MASK_BYTES_MAX`) and the DRAM free per device at that point must hold them plus the slab's working set
  (the dense-linear admission's term: 988.5 MiB at 2048 rows).  So contexts to 64k hoist their masks when that DRAM
  is free; 128k and 256k contexts, or a process short of the headroom, derive the masks per layer as before (the
  bitwise slab body either way; only the mask programs' count differs).
- `qsa_scores_rs_ag` (tolerance, off): each 512-row score block's four device partials are masked on every device,
  summed by a `reduce_scatter` over the block's rows, ranked per device on its 128 rows and the block ids gathered, in
  place of broadcasting every device's scores to every device and summing locally.  A third of the score payload
  crosses the line; the four terms are summed in the collective's hop order, so a near-tie between two blocks can
  rank the other way and change that layer's attention set for that row.
- `gr_partial_rs_ag` (tolerance, off): the gated-residual partials are summed by `reduce_scatter` + `all_gather`
  instead of the gather + device-order reduce; the four fp32 terms are summed in hop order.  It supersedes the gather
  forms when both are named.
- `gdn_qk_flat` (tolerance, off): the GDN chunk kernel takes the raw convolution q/k rows in its flat form and maps
  value heads to key heads and l2-normalizes them itself, in fp32 with the scale folded in, in place of the model's
  0/1 expand, bf16 `rms_norm` and bf16 scale (about 66 ms per slab of the model's and the kernel adapter's glue).  The
  normalized values differ by bf16 rounding and the recurrent state carries the difference into the decode.

## Running it

```
models/demos/blackhole/qwen38_flash_next/tools/run_qwen38_chat_server.sh --profile <profile> --devices <nodes> \
    --checkpoint <DIR> --cache-root <DIR> --prefill-slab 2048 [--acceptance --require-json-96]
```

The first start compiles the slab body's programs in the warm pass (an eager slab from the reset state) and captures
its trace after the 128-row chunk trace; `/health` reports `prefill_slab_rows`.  `QWEN38_MOE_SLAB_ONE_CALL=0` in the
server's environment restores the 16 x 128-row expert calls, `QWEN38_MOE_SLAB_RINGS=0` restores one ring for the
one call; both are read when the slab's layer instances are built, so they take effect at the next start.

### The dense-linear switches

Four environment variables, read once when the model is built, move the slab's dense linears (the GDN in/out
projections, the gated-residual down+inject and up, the shared expert's gate/up/down/scalar, the QSA query-gate / K /
V / output projections) and nothing else.  The dtype, fidelity and accumulation defaults keep the slab's arithmetic
(the decode weights copied per slab in their format, `QWEN38_DENSE_WEIGHT_DTYPE`, with the module's compute config
for it; fp32 accumulation); the grid's default is `wide`, and
`QWEN38_PREFILL_DENSE_GRID=today` restores the earlier slab bitwise (the grids of
`decode_matmul.prefill_matmul_program_config`, no resident copy):

| variable | values | default | what it changes |
|---|---|---|---|
| `QWEN38_PREFILL_DENSE_DTYPE` | `bf16`, `bf8` | `bf16` | `bf8`: a resident DRAM-interleaved bfloat8_b copy of each weight, read in place by the slab (no per-slab copy); the decode weights stay bf16 |
| `QWEN38_PREFILL_DENSE_FIDELITY` | `hifi4`, `hifi2`, `lofi` | `hifi4` | `hifi4`: the module's own compute config (the fidelity of its weight format: HiFi4 for bf16, HiFi2 for the bf8 default, LoFi for bf4); `hifi2` / `lofi`: that fidelity for the slab's dense linears instead (`lofi` reads a 5-bit weight mantissa: a quality point to measure, not a free speed knob) |
| `QWEN38_PREFILL_DENSE_GRID` | `today`, `wide` | `wide` | `wide`: more columns for the narrow-N shapes (the query-gate 80 -> 110 cores, the gated-residual down+inject 30 -> 60), and the pair-grouped K/V and the shared expert's gate/up as one linear each on resident copies in the weights' own format (the same tiles the separate linears read: about 58 MB per device as bf8, 110 MB as bf16); `today`: the earlier grids and per-slab copies |
| `QWEN38_PREFILL_DENSE_FP32_ACC` | `1`, `0` | `1` | fp32 accumulation across the K blocks |

The wide grid, measured on the 4-chip line (1x4 p150, a 32k context, `--prefill-slab 2048`, 2026-09-25, with the
bf16 dense weights of record then) against `today` in the same process form: the device column of the agreement corpus's two long windows is identical at all
160 scored positions (KL 0.0, top-1 1.0 between the two columns; both 0.9812 / 0.0225 top-1 / KL against the HF
reference) and the 16-token completions of the long-prompt probe (2k to 32k tokens) are byte-identical; the dense
family (the Matmul class of the slab's device profile) 92.57 -> 75.84 ms per 2048-row slab (-18 %), the slab's
kernel time 1584.3 -> 1567.7 ms (-16.6 ms, -1.05 %: 1284 -> 1297 prompt tokens per second on the body, TTFT at
32k 27.56 -> 27.32 s); 60 fewer programs per slab (the fused siblings); +13.8 MB of DRAM per bank for the two
resident bf16 siblings and no other resident; the acceptance pins 12/12 identical.  The wide grid changes the
block order over the cores and fuses two pairs of sibling linears whose per-column arithmetic is the same (the
siblings are the separate linears' tiles in the same format), so its numerics class is `today`'s under any
`QWEN38_DENSE_WEIGHT_DTYPE`; the grid alone does not touch the fidelity or the accumulation.

The MoE router and the QSA index projections never read the switches: their outputs pick the routed experts and the
attended blocks, so they keep today's config, fidelity and per-slab weight copy under every setting.  The resident
copies (about 950 MB per device as bfloat8_b; the fused siblings alone about 58 MB as bf8) are allocated after the decode
weights and the resident experts and refused, with the numbers, when the DRAM free at that point would not hold
them plus what the slab process allocates after the build: the context-scaled state (417 MiB at 32k), the 2048-row
slab's working set and a 512 MiB margin.  The working set is a measured band (the 4-chip line, 32k, 2026-09-25):
the slab ran with 251.3 MB per bank free after the build and failed in its first layer with 141.0 (the bf8 copies
under the device profiler's default 64,000-program DRAM reservation), and the admission takes the band's upper end,
the only value known to fit, so at 32k it asks for 251.3 MB per bank after the weights.  The bf8 copies fit the
served process at 32k (443 MB per bank after them) and the profiler at a 32,000-program reservation (295 MB), not
the profiler's default (141 MB: refused with the numbers).  The switches act only on a build that runs a slab (the
chain tells the builder the `--prefill-slab` rows before the target is built; a slab above 2048 rows scales the
working-set term with its rows): a build without a slab, such as the single-user 256k server that prefills in
chunks, allocates no resident copy, runs no admission and is unchanged.  The acceptance gate's prompts are
shorter than a slab, so a non-default setting is judged on long prompts: the agreement corpus's long windows and the full-model
gate's 4k / 8k synthetic prompts with `--prefill-slab`.  The design, the arms and their expected gains: the prefill
dense design note (a development document, not shipped).
