# Numerics: the acceptance gate, the pinned tables, what "bitwise" means here

Every number here was measured on 4x p150 unless a date and host say otherwise.

## What "bitwise" means here

- **Bitwise repeatable**: the same request on the same build gives the same device token stream on every run.  The
  greedy argmax stream a request without sampling fields gets is bitwise equal to the greedy loop the acceptance replay
  and the evaluations measure, on the launcher's `--sampling` server too (`qwen38.decode_loop` is `greedy` in the
  response and the ledger; `temperature 0` and `greedy: true` are the same path).
- **Bitwise between two device paths**: `--long-chunks` (128-row prefill chunks) produces the same tokens as 32-row
  chunks alone, bitwise on all 48 layers.  The MTP path is not bitwise with plain decode on 4 of the 12 acceptance
  prompts (below).  The device token streams on tt-metal `28238f903b` (2026-09-06) are bitwise the same as on the
  previous base `d04395ed86` (2026-08-29).
- **Against the CPU reference**: the device is not bitwise with the CPU.  Chunked prefill is tolerance-class against the
  CPU reference on all 48 layers; the greedy token streams agree with the CPU records for a prefix and then leave them
  (the divergence indices below).  96/96 on the `json` record is the gate.
- **BF4 experts**: the routed experts run from a BF4 cache whose every tensorbin payload is byte for byte the CPU-staged
  corpus's (`tools/stage_full_bf4_cpu.py`); the CPU oracle can run with bf16 or BF4-emulated experts (`oracle` below).
  The cache is keyed by the converter's sources (`ttnn/bf4.py`, the `moe_compute` layout packer, tt-metal's BFP4
  packer), and every start re-packs one routed expert of the first cached layer from the checkpoint and compares the
  bytes with the cache; a cache converted by different code is refused (`SERVER.md`).

## Fused decode kernels and two-reader decode linears (2026-09-16)

Decode chains run as fused programs (`ttnn/fused/`, built on `ttnn.generic_op`) where a kernel is bitwise against the
chain it replaces on device, leaves every pinned table above unchanged and beats the previous step time in its own
timing slot.  On by default: `gr_read` with `gr_fold`, `gr_write`, `greedy_tail`, `moe_post`, `ple`, `position_derive`,
`qsa_block`, `router_tail`, `shared_expert`: the gated-residual read as two programs with its two all-gathers inside
them (the stats, their gather, normalize + down-project and the partial gather as one program whose transport cores
send the tiles over the 1D fabric line into the pages the stock collectives write, then low-rank + gate: 18 programs
per read as 2, the chain's LLK sequences call for call, its reduce scaler and spill/reload rounding included; a gather
is data movement, so the fold is bitwise, 2026-09-25); the gated-residual write as one program (SFPU multiply, FPU add, as
the chain); the tail's greedy epilogue (24 programs as 4 plus one gather); the MoE post program (fill, tilize, the
score-weighted reduce over the ten expert slots in slot order, the shared expert's x sigmoid and the partial add as one
program; its reader takes each routing tensor in one read from moe_compute's drain-core shard and seeds the score
tiles with the NoC's zeros, so the program is 7.5 -> 5.9 us at one row and flat in the row count, bitwise; at several
rows the routed dispatch untilizes the sharded hidden directly and takes the rows view of the result, 2026-09-25); the
prologue's position derivation (40 programs as 1); the sparse-attention block's decode glue as six
programs (index tail, main tail, post-attention, partial widen, selection row, score merge); the MoE router tail
(softmax, top-10, sum, div, casts and layouts: 12 programs per layer as one, its top-k on one core per eight-token
group: the LLK sort's four independent passes, so every token sees the chain's instructions, bitwise, 88 -> 52 us per
layer at one row, 2026-09-18); the shared expert as three programs (one
DRAM-sharded linear over the concatenated [gate | up | scalar] weight, one silu / product / sigmoid program, the down
linear); the layer-1 PLE (stats, group norm, gate, conv with the state shift and the layer's permute + add: 56 programs
as 9, the SFPU `mac_tile` of `ttnn.mac`, the accurate fp32 reduce of the gate's sum); and, since 2026-09-25, `gdn_step`,
the GDN decode step from the projection to the gated output as one program (the conv, the head split, the gates, the
l2 norms, the fp32 delta-rule update, the read-out, the gated RMSNorm and the sigmoid gate: 49 programs per layer as 1),
a COMPONENT-class kernel that serves by default on its component-gate proof against the CPU oracle (layer-0 probe, four
p150: state error 0.0048 and gated output 0.0098 for the fused step, 0.0081 and 0.0234 for the composed chain) and
runs where its input contract holds (the one-row step and, since 2026-09-25, the batched-decode lanes body on B rows,
one item per (lane, value head) and one state slot per lane: 40.9 -> 35.0 ms per step at 4 lanes and 50.4 -> 43.3 ms at
8 on the 200-replay lane sweep, 28.5 and 23.1 tok/s per user, 114 and 185 aggregate, the one-row step unchanged; the MTP
draft body and verify rows keep the chain).  The composite lane body, kept as the fallback when the fused GDN step is
not admitted, differs from the one-row fused body at one near-tie in the acceptance chain (219/225 at 4 and 8 lanes);
the fused lanes form matches 225/225.  Opt-in
through `QWEN38_FUSED=<name>`: `final_mixer`, `position_advance`.  The kernels cover rows 1..32 (decode, the MTP
verify rows); the 128-row prefill chunk and the slab keep their chains.  `QWEN38_FUSED_OFF=<name>[,...]` (or `all`) in
the server's environment falls back to the composed chains; an unknown name in either variable refuses to start;
`QWEN38_FUSED_OFF=gr_fold` runs the GR read's merged three-program form with the stock collectives (5 programs per read);
`QWEN38_FUSED_GR_READ_MERGED=0` runs the GR read's split form (7 programs per read); `QWEN38_ROUTER_TAIL_LANES=0` runs
the router tail's top-k on one core per tile (the same program, the LLK's four passes on that core).

The DRAM-sharded decode linears of the GDN input and output projections, the sparse attention's query-gate and output
projections and the LM-head chunks read each DRAM bank with two worker cores (`num_workers_per_dram_bank=2`; the K/V and
index projections, the router, the gated-residual linears and the shared expert keep one).  The arithmetic is the same
(the same K order into the same fp32 destination): bitwise at rows 1..32 and at the model level, every pinned table
above holds.  `QWEN38_DRAM_WORKERS=1` restores one reader per bank.  The GDN input weight's bank shard holds a whole
number of tiles per reader, 18 instead of 17 (576 instead of 544 columns: 1.3 MB per layer, 47 MB per card of the
4.6 GB the 32k build leaves free), under its own cache file (`qkvzab_tile_aligned_ab_dram_sharded_bank576`); a cached
tensorbin loads in the layout it was written in, so every loaded member is checked against the layout its linear runs
and a one-reader file is never accepted for the wider allocation.

Measured 2026-09-16 on 4x p150 (200 traced decode steps, host wall, one hold): 49.89 ms per token with the
chains and one reader per bank (`QWEN38_FUSED_OFF=all QWEN38_DRAM_WORKERS=1`), 38.67 ms with the defaults
(25.9 tokens/s; 38.26 min, 38.85 p90); 3,370 programs and 34.5 ms of kernel time per
step on chip 0 under the device profiler (38.9 ms span).  The 2026-09-15 numbers (13 kernels, one reader per
bank): 42.14 ms, 4,618 programs, 37.9 ms of kernel time.

## The acceptance mechanism

`--acceptance` replays the twelve shipped CPU greedy records under `tools/acceptance/greedy-prompts/` against the CPU
before the server listens.  `--require-json-96` refuses to serve unless the `json` record matches the CPU 96/96 (the
other records diverge from the CPU after 6-75 tokens, the known greedy-prompt pattern; their first divergence index is
in `acceptance.json` in the run directory).  The records were rendered by the CPU study with the system prompt
`You are a helpful assistant.`; that system turn is inside their recorded prompt ids, which the replay feeds to the
device as they are.  The server itself adds no system prompt to a client's request (`SERVER.md`).

## The pinned table: chunked prefill, 32k, 2026-09-25

The pinned table (`tools/ci/baselines/A3-chunked-32k-divergence_index.json`), measured 2026-09-25 on two 4x p150 hosts
with identical device tokens record for record, with the chunked prefill, is the first index where each record leaves
the CPU greedy stream; a start whose replay leaves earlier is a regression:

| json | chat | code | fact | list | math | multilingual | prose | refactor | sky | story | summary |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none (96/96) | 2 | 32 | 15 | 56 | 61 | 9 | 13 | 24 | 19 | 6 | 75 |

The 2026-09-25 change is the BF8 dense weights, the fused GDN step and the compact expert layout by default: `chat`
leaves at 2 where the 2026-09-06 table had 43 (at index 2 the device's own candidate row holds its pick, 449, one bf16
step above the CPU's 264; the BF8 weights and the fused step each move that margin one step, alone either keeps 264),
the other eleven indices are unchanged, and six streams differ from the previous table after their divergence
(`fact` at 60, `multilingual` at 32, `prose` at 76, `refactor` at 83, `story` at 16).  Against the HF reference over the
36-item corpus the device column stays at top-1 0.9502 (the 2026-09-06 record, weighted the same way, 0.9518) and
against the bf16 CPU oracle at top-1 0.9539, truncated KL 0.0511.  The 2026-09-06 table read chat 43 with the rest as
above.  Before 2026-09-06 the table read chat 8, code 24, list 46, refactor 22 (the other eight as above).  The change is the
GDN decay gate: the fused `add + softplus` activation the gate used returned exactly 0 wherever its input was below
-5.02 (and was 1.5e-3 off elsewhere); the gate now runs `ttnn.add` then `ttnn.softplus`, which follows the reference
everywhere (4e-5 absolute).  Against the CPU oracle the device's GDN state error on a decode step fell from 0.29 to
0.013; with the chunked prefill the device stays on the CPU stream longer on 4 of the 12 prompts and no prompt leaves
earlier (the teacher-forced table, `A3-forced-32k-divergence_index.json`, moved later on four prompts and earlier on
three); the cost is one more program per layer, about 0.4 ms per decoded token (50.0 -> 50.4 ms).  The proofs in
`PROOFS.md` predate this fix where they quote the old indices.

## `--long-chunks`

The 128-row chunk path gives the same tokens as 32-row chunks alone, bitwise on all 48 layers; on the QuietBox
(2026-09-06, before the gate fix) the acceptance replay with `--long-chunks` was identical to the plain start (`json`
96/96, the same eleven divergence indices, 19.4-19.6 tokens/s).

## MTP (`--mtp 4`), 32k, 2026-09-25

The MTP path is not bitwise with plain decode on 3 of the 12 acceptance prompts (measured 2026-09-25 with the fused GDN
step serving the one-row step while the draft body and the verify rows run the composed chain, and the verify rows and
the 1-row loop rounding differently): the committed stream leaves the CPU reference on `chat` at token 43 where plain
decode leaves at 2, on `math` at 63 against 61 and on `summary` at 1 against 75 (`In 1947,` becomes `Invented at Bell
Labs in`); the other nine records leave the reference at the plain-decode token (`json` 96/96), and every gate passes.
The 2026-09-06 table (below, the GDN gate fix of that day) had four such prompts: `chat` 56 against 43, `list` 46
against 56, `math` 56 against 61, `summary` 1 against 75; `list` now agrees at 56.  At
each of the three earlier tokens the device's own 1-row logits hold the CPU's token and the MTP token within one bf16
step (an exact tie on `summary` and `list`, which the 1-row loop breaks toward the lower id), the verify row lands one
step the other way, and the CPU oracle rates the two 0.4-1.0 logits apart: near-ties, not a defect (the rows-path gate
is the 1-row gate bitwise on the same row; the pinned MTP table is `tools/ci/baselines/A3-mtp4-32k-divergence_index.json`).
Before the gate fix the two paths differed on `code` (44 against 24) and `fact` (16 against 15) only.  Throughput
on the 2026-09-16 body (14 fused kernels, two DRAM readers per bank; quiet host, the acceptance replay of the 12
prompts, both tables as pinned): `--mtp 4` 39.0 tokens/s median over the prompts and 67.9 on `json` (4.80 tokens per
pass, 70.7 ms per pass); `--mtp 3` 38.0 median, 55.6 on `json`.

| | json | chat | code | fact | list | math | multilingual | prose | refactor | sky | story | summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain decode, 2026-09-25 | none (96/96) | 2 | 32 | 15 | 56 | 61 | 9 | 13 | 24 | 19 | 6 | 75 |
| `--mtp 4`, 2026-09-25 | none (96/96) | 43 | 32 | 15 | 56 | 63 | 9 | 13 | 24 | 19 | 6 | 1 |
| plain decode, 2026-09-06 | none (96/96) | 43 | 32 | 15 | 56 | 61 | 9 | 13 | 24 | 19 | 6 | 75 |
| `--mtp 4`, 2026-09-06 | none (96/96) | 56 | 32 | 15 | 46 | 56 | 9 | 13 | 24 | 19 | 6 | 1 |

## The device sampler's law (2026-09-25)

The on-device sampler (`sampler_tail`, one program on one core after the top-32 candidate row) is gated on its law, not
on tokens: at teacher-forced prefixes of the acceptance prompts (24 positions: chat, code, json, story at offsets 0, 16,
48, both model-card profiles) the tool draws 4096 device tokens per position with a fresh seeded stream and compares
their counts with the host sampler's law over the same candidate row (`sample_candidates` before its draw) by the
sampled-MTP lane's method: the G statistic with cells under five pooled and alpha 0.01 split over the positions
(Bonferroni, 0.000417 each), the total variation of the empirical law against the host law beside its null expectation,
the device sampler's own law (its draw intervals over the 2^24 uniform grid, exact) against the host law, and every
device token against the integer-exact host reference (`device_sampler_reference`, 0 mismatches required).  The presence
penalty runs on the device from the request's own history.  Verdict 2026-09-25: PASS: every position passes the G-test
(minimum p 0.0740), the largest empirical TV is 0.0079, the largest device-law TV is 0.0000022, mismatches 0.

Profile `thinking`:

| prompt | offset | kept lanes | G | df | p | TV empirical vs host | TV null expectation | TV device law vs host | draws | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| json | 0 | 2 | 0.04 | 1 | 0.8381 | 0.0008 | 0.0030 | 0.0000010 | 4096 | 0 |
| json | 16 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| json | 48 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| chat | 0 | 5 | 2.72 | 4 | 0.6060 | 0.0079 | 0.0095 | 0.0000022 | 4096 | 0 |
| chat | 16 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| chat | 48 | 2 | 3.19 | 1 | 0.0740 | 0.0046 | 0.0020 | 0.0000004 | 4096 | 0 |
| code | 0 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| code | 16 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| code | 48 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| story | 0 | 3 | 1.70 | 2 | 0.4273 | 0.0056 | 0.0057 | 0.0000012 | 4096 | 0 |
| story | 16 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| story | 48 | 3 | 1.82 | 2 | 0.4030 | 0.0049 | 0.0075 | 0.0000009 | 4096 | 0 |

Profile `instruct`:

| prompt | offset | kept lanes | G | df | p | TV empirical vs host | TV null expectation | TV device law vs host | draws | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| json | 0 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| json | 16 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| json | 48 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| chat | 0 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| chat | 16 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| chat | 48 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| code | 0 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| code | 16 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| code | 48 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| story | 0 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| story | 16 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| story | 48 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |

The instruct card where its policy keeps two or more lanes, so the draw is a real one: the tool walked each record's
offsets 1..95 reading the host law at every prefix and took the first 4 offsets per prompt with at least 2 kept lanes
(12 positions over chat, code, story), 4096 draws each, alpha 0.01 split over the 12 positions (0.000833 each).  Verdict
2026-09-25: PASS: G-test 12/12 (minimum p 0.1427), largest empirical TV 0.0119, largest device-law TV 0.0000011,
mismatches 0.

| prompt | offset | kept lanes | G | df | p | TV empirical vs host | TV null expectation | TV device law vs host | draws | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chat | 1 | 2 | 0.00 | 1 | 0.9695 | 0.0003 | 0.0060 | 0.0000001 | 4096 | 0 |
| chat | 2 | 2 | 2.15 | 1 | 0.1427 | 0.0110 | 0.0060 | 0.0000001 | 4096 | 0 |
| chat | 3 | 3 | 1.22 | 2 | 0.5423 | 0.0070 | 0.0079 | 0.0000011 | 4096 | 0 |
| chat | 8 | 2 | 0.39 | 1 | 0.5320 | 0.0049 | 0.0062 | 0.0000000 | 4096 | 0 |
| code | 24 | 2 | 1.38 | 1 | 0.2396 | 0.0076 | 0.0052 | 0.0000011 | 4096 | 0 |
| code | 26 | 2 | 0.13 | 1 | 0.7211 | 0.0024 | 0.0054 | 0.0000006 | 4096 | 0 |
| code | 32 | 2 | 0.13 | 1 | 0.7152 | 0.0028 | 0.0062 | 0.0000005 | 4096 | 0 |
| code | 40 | 2 | 0.02 | 1 | 0.8773 | 0.0012 | 0.0060 | 0.0000004 | 4096 | 0 |
| story | 1 | 3 | 2.32 | 2 | 0.3141 | 0.0119 | 0.0085 | 0.0000005 | 4096 | 0 |
| story | 2 | 2 | 1.96 | 1 | 0.1612 | 0.0105 | 0.0060 | 0.0000004 | 4096 | 0 |
| story | 5 | 2 | 0.21 | 1 | 0.6466 | 0.0030 | 0.0052 | 0.0000011 | 4096 | 0 |
| story | 6 | 2 | 1.49 | 1 | 0.2216 | 0.0092 | 0.0060 | 0.0000004 | 4096 | 0 |

## The teacher-forced table

`tools/ci/baselines/A3-forced-32k-divergence_index.json` pins the startup replay with the teacher-forced prefill
(`--prefill-mode teacher_forced`), measured 2026-09-06 on 4x p150 with the GDN gate fix:

| json | chat | code | fact | list | math | multilingual | prose | refactor | sky | story | summary |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none (96/96) | 25 | none (96/96) | 15 | 56 | 61 | 43 | 13 | 24 | 19 | 6 | 75 |

Against the 2026-09-04 table the fix moved chat 10 -> 25, code 44 -> 96/96, list 46 -> 56 and summary 61 -> 75 later,
math 63 -> 61, multilingual 45 -> 43 and story 19 -> 6 earlier; fact, prose, refactor and sky are unchanged.

## The reference columns

`tools/reference/` freezes the teacher-forced corpus `Q38-REF-v1` (36 items, about 13k positions, with sha256s;
`TESTING.md` lists what is in it and how to run the tool).  `tools/qwen38_reference_corpus.py hf` keeps per position
the top-32 ids and log-probs and the teacher's log-prob from the Transformers `qwen4_exp` model on the CPU (bf16
weights, fp32 LM head); `oracle` does the same through `tt/` (bf16 or BF4-emulated experts); `device` converts a served
chain's agreement records; `score` compares two columns (top-1 / top-5 agreement, clear-margin top-1, truncated KL over
the shared top-32 support, first divergence).  The HF column is the acceptance reference the other columns are read
against.
