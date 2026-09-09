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
- the routed experts run in `moe_compute` calls of 128 tokens (the largest call the kernel admits), one per 128-row
  block, through the 128-row instance's own call, tilize and weighted reduce;
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
KV slab write 0.056 ms against 0.206.  What does not change with the row count is the MoE expert stream (16 calls of
128 tokens per layer per slab: `moe_compute` refuses more tokens per call on this runtime) and its per-call combine
tilize and weighted reduce, together about 0.4 ms per prompt token: the slab's floor.

Measured end to end on 4x p150 (2026-09-09, 32k context; the 6942-token prompt is 3 slabs, 6 long chunks, 2 short
chunks and a 30-row tail, so its rate blends the slab's with the remainder's):

| | 32-row chunks | `--long-chunks` | `--prefill-slab 2048` |
|---|---|---|---|
| ms per prompt token (6942-token prompt) | 3.3 | 1.55 | 1.155 (TTFT 8.07 s) |
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
| 4,179 | 6.36 s / 1.51 | 4.57 s / 1.08 | 4.77 s / 1.13 |
| 8,176 | 12.3 s / 1.50 | 9.65 s / 1.17 | 10.1 s / 1.23 |
| 16,322 | 24.4 s / 1.49 | 18.0 s / 1.10 | 18.9 s / 1.16 |
| 31,988 | 47.4 s / 1.48 | 33.6 s / 1.05 | 35.0 s / 1.09 |
| 64,655 | - | - | 70.3 s / 1.09 (31 slabs) |
| 127,872 | - | - | 151 s / 1.18 at a 128k context (62 slabs) |

Where the slab's time goes (the device profiler over the traced body, tt-perf-report per replay): the traced slab
body itself runs 1.32-1.44 s per 2048 rows (0.64-0.70 ms per token, kernel-bound: 19,161 programs, of which the 768
`moe_compute` calls are 0.20 ms per token, the collectives 0.11, `sparse_sdpa` 0.06, the element-wise ops 0.06, the
dense matmuls 0.045); the rest of the measured 1.05 ms per token, about 0.35-0.40, is spent outside the trace between
slabs (the host's n-gram lookup and the 10 MB embedding-rows upload for the next slab, then the event sync), which the
current driver does not overlap with the replay.

The slab takes 28-29 percent less time per prompt token than the 128-row chunks at every length (1.05 against 1.48 ms
at 32k tokens: TTFT 33.6 s against 47.4 s).  The slab body itself runs at about 1.05 ms per prompt token (about 950
prompt tokens per second); the shorter prompts pay more for their remainder (the 128- and 32-row chunks after the last slab) and the fixed hand-off.  A 64k context
(`--allocated-context 65536`) costs about 4 percent more per token (the QSA selection scores over twice the blocks) and
56 MB per DRAM bank more; a 128k context about 13 percent more per token (1.18-1.34 ms; 226 MB per bank left); the slab
state and trace themselves cost 45 MB per bank at any context.

Memory: the slab state adds about 200 MB per device at 32k (one set of GDN pass buffers shared by the 35 GDN layers,
the PLE rows, the QSA slab constants and kept rows, the slab trace); the transients inside a layer peak around
150 MB (about 350 MB at 256k, where the score all-reduce per 512-row block is the largest).

## Running it

```
models/demos/blackhole/qwen38_flash_next/tools/run_qwen38_chat_server.sh --profile <profile> --devices <nodes> \
    --checkpoint <DIR> --cache-root <DIR> --prefill-slab 2048 [--acceptance --require-json-96]
```

The first start compiles the slab body's programs in the warm pass (an eager slab from the reset state) and captures
its trace after the 128-row chunk trace; `/health` reports `prefill_slab_rows`.
