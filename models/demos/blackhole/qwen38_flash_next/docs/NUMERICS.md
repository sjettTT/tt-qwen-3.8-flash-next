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
timing slot. On by default: `gr_read` with `gr_fold`, `gr_write`, `greedy_tail`, `moe_combine` (the prefill slab's
MoE combine as one program), `moe_post`, `ple`, `position_derive`,
`qsa_block`, `qsa_rows`, `router_tail`, `shared_expert`: the gated-residual read as two programs with its two all-gathers inside
them (the stats, their gather, normalize + down-project and the partial gather as one program whose transport cores send
the tiles over the 1D fabric line into the pages the stock collectives write, then low-rank + gate: 18 programs per read
as 2, the chain's LLK sequences call for call, its reduce scaler and spill/reload rounding included; a gather is data
movement, so the fold is bitwise, 2026-09-25); the gated-residual write as one program (SFPU multiply, FPU add, as the
chain); the tail's greedy epilogue (24 programs as 4 plus one gather); the MoE post program (fill, tilize, the
score-weighted reduce over the ten expert slots in slot order, the shared expert's x sigmoid and the partial add as one
program; its reader takes each routing tensor in one read from moe_compute's drain-core shard and seeds the score tiles
with the NoC's zeros, so the program is 7.5 -> 5.9 us at one row and flat in the row count, bitwise; at several rows the
routed dispatch untilizes the sharded hidden directly and takes the rows view of the result, 2026-09-25); the prologue's
position derivation (40 programs as 1); the sparse-attention block's decode glue as six programs (index tail, main tail,
post-attention, partial widen, selection row, score merge); the MoE router tail (softmax, top-10, sum, div, casts and
layouts: 12 programs per layer as one, its top-k on one core per eight-token group: the LLK sort's four independent
passes, so every token sees the chain's instructions, bitwise, 88 -> 52 us per layer at one row, 2026-09-18; its precise
exp runs over the vector pairs that hold the core's live rows only -- a dead row's exp is never read -- 52 -> 39 us
at one row and 53 -> 42 at the MTP verify's five, bitwise, 2026-09-26); the shared
expert as three programs (one DRAM-sharded linear over the concatenated [gate | up | scalar] weight, one silu / product
/ sigmoid program, the down linear); the layer-1 PLE (stats, group norm, gate, conv with the state shift and the layer's
permute + add: 56 programs as 9, the SFPU `mac_tile` of `ttnn.mac`, the accurate fp32 reduce of the gate's sum); and,
since 2026-09-25, `gdn_step`, the GDN decode step from the projection to the gated output as one program (the conv, the
head split, the gates, the l2 norms, the fp32 delta-rule update, the read-out, the gated RMSNorm and the sigmoid gate:
49 programs per layer as 1), a COMPONENT-class kernel that serves by default on its component-gate proof against the CPU
oracle (layer-0 probe, four p150: state error 0.0048 and gated output 0.0098 for the fused step, 0.0081 and 0.0234 for
the composed chain) and runs where its input contract holds (the one-row step and, since 2026-09-25, the batched-decode
lanes body on B rows, one item per (lane, value head) and one state slot per lane: 40.9 -> 35.0 ms per step at 4 lanes
and 50.4 -> 43.3 ms at 8 on the 200-replay lane sweep, 28.5 and 23.1 tok/s per user, 114 and 185 aggregate, the one-row
step unchanged; the MTP draft body and verify rows keep the chain). The composite lane body, kept as the fallback when
the fused GDN step is not admitted, differs from the one-row fused body at one near-tie in the acceptance chain (219/225
at 4 and 8 lanes); the fused lanes form matches 225/225.  The lanes' greedy tail scans (row, tile-group) items since
2026-09-25, one lane row per core on up to 128 cores (the scan program 164 -> 86 us at 4 lanes and 322 -> 173 at 8, the
step 34.7 -> 34.6 and 42.8 -> 42.7 ms at 4 and 8 lanes on the 200-replay lane sweep), bitwise the per-core all-rows scan
it replaces (`QWEN38_FUSED_GREEDY_TAIL_LANE_SPLIT=0`).  Since 2026-09-26 the MoE block's router top-k, the shared
expert's eltwise and down linear and the routed dispatch untilize run as ONE program (`moe_dense`: four kernel groups on
disjoint cores, the top-k's lane cores on a placement rectangle off the dense linears' storage cores, the eltwise on
those five storage cores multicasting its intermediate into 16 worker cores that run the down linear as a streaming
matmul with the DRAM-sharded matmul's spill and reload after every K tile, the untilize on eight cores), so the 13.5 us
of shared work per layer run under the 51 us top-k instead of after it: 144 programs per step fewer, the composite 51.2
us where the four programs took 64.9, the one-row step 26.50 -> 25.89 ms per token on the landed head beside the live-row exp (37.7 -> 38.6 tok/s, the greedy
request of the sampling server, 200 traced steps in one hold; alone on the previous head 27.24 -> 26.65), the MTP verify pass 59.3 -> 58.6 ms (json) and 57.9 -> 57.5 (prose)
with identical committed streams; bitwise the four programs on every row 1..32 at both down-weight formats and both
routing placements (the device test and the audit-capture microtest), the acceptance table unchanged (12/12);
`QWEN38_FUSED_OFF=moe_dense` restores the four programs.  And, since 2026-09-25, `gdn_rows_wrap`: the MTP verify
rows' GDN body between the projection's sharded-to-interleaved and the out-projection matmul as the prefill lane's
`gdn_pre_rows` + the two chunk prims `ttnn.prim.chunk_gdn_prep` / `chunk_gdn_scan` + `gdn_post_rows`, 6 programs per
GDN layer where the chain ran 53 (the layer's verify segment 11 against 58), bitwise on the die (the gated tile, o and
the final state against the composite's, unmasked and at every commit mask, eager and replayed: the rows micro-test's
wrap arm) and at the pass level (the served pass pair on the same tree: tokens per pass and the accept histograms
identical, the k = 4 pass 7.3-9.8 ms shorter; per GDN layer 0.4805 -> 0.3061 ms and 58 -> 12 device ops).  The
served pair on that tree (the acceptance lineage, so greedy ran the split verify form on both arms; fixed 256-token
answers through the like-for-like client; `QWEN38_FUSED=gdn_rows_wrap` against the switch unset): 560-token chat
55.40 vs 48.81 tok/s (+13.5 %), 177-token multi-turn chat 51.75 vs 45.81 (+13.0 %), json 82.58 vs 72.98 (+13.2 %),
code 82.79 vs 73.33 (+12.9 %), prose 40.99 vs 36.10 (+13.5 %), tokens per pass identical on every class (3.1205 /
2.8333 / 4.5536 / 4.5714 / 2.1983, the same pass counts); `A3-mtp4-32k` with the wrap on 12/12 with the sha, every
pinned divergence index identical (chat 43, code 32, list 56, math 63, ...), json 96/96 at 85.3 tok/s, the hand-offs
matched in both orders.  `QWEN38_FUSED_OFF=gdn_rows_wrap` restores the chain, and the prefill slab keeps its own form
(`gdn_prefill_rows`, on by default after its line gate).
Opt-in through `QWEN38_FUSED=<name>`: `final_mixer`, `position_advance`, `gdn_rows_prims_direct` (the MTP
verify rows' chunk recurrence as the composite's two phase prims called directly with the composite's own relayout ops
in its order, bitwise by construction and on the line 2026-09-25: the seam the wrap's programs were proven against, a
diagnostic beside the default), and `gdn_rows_scan` (2026-09-26, class COMPONENT against the served stream): the same
verify-rows body as ONE program per GDN layer that runs the fused `gdn_step`'s serial fp32 recurrence row after row
over the k + 1 rows and keeps every prefix state, so the commit is one pick of the prefix state after the accepted rows
instead of the masked re-run of the two chunk prims.  Measured on the 4x p150 line at k = 4 on real layers 1 and 33
(the rows micro-test's scan arm): every prefix state and every committed state bitwise the state after the same rows
of the served 1-row fused step, and the GDN layer's output rows bitwise plain decode's 1-row outputs (max abs 0.0),
where the wrap / chain form (the chunk / WY recurrence at TF32 operand precision) deviates from those outputs by 1.2e-3
on the state at a scale of 0.185 (layer 1; 9.2e-4 at 0.066 on layer 33) and up to 1.2e-2 on the bf16 output rows;
device operations per GDN layer 10 for the forward against the wrap's 12 (the chain's 58) and 3 for the commit against
5 (12); on one die the program runs in 113 us at k = 4 on 48 cores against 382 us for five sequential fused steps, the
pick in 11 us.  The fold's persistent DRAM: (k + 1) x 786,432 bytes per GDN layer per device (3.9 MB at k = 4;
141.6 MB over the 36 GDN layers); on the line at k = 4 with two verify forms the MTP chain's states term measured
31,093,824 bytes per bank against the admission's 13,785,664 estimated without the fold (+17,308,160, of which the
prefix states are 17,694,720 by construction and the rest sits inside the margin), its traces 5,832,256 against
12,218,695 estimated (one program per GDN layer where the wrap runs six), 87,394,112 per bank in all, so the admission's
estimate carries the prefix states -- derived from k + 1 and the GDN layer count -- whenever `QWEN38_FUSED` names the
kernel; the term is per drafting chain (a second chain of `QWEN38_MTP_DRAFTS_PER_REQUEST` allocates its own GDN
rows states and is charged its own k + 1).  The whole verify stream with the kernel on is measured in the MTP
section's opt-in row below (the served line, 2026-09-26); the QSA and MoE rows forms are unchanged by this kernel, and
the MTP table's pins move only at the kernel's default flip.  The kernels cover rows 1..32
(decode, the MTP verify rows); the 128-row prefill chunk and the slab keep their chains.
`qsa_rows` (2026-09-26, on by default; `QWEN38_FUSED_OFF=qsa_rows` restores the composed glue) runs the QSA verify
tile's glue as fused programs: program 1, the indexer scores' all-reduce composite and mask add as one all-gather plus the fused
score merge over the 32 rows (6 programs per layer as 2; bitwise on the four dies at three positions; the MTP lead's pass
pair on the line: -0.50 / -0.60 / -0.51 ms per pass on json / prose / the 560-token chat, the committed streams bitwise,
the A3-mtp4-32k pins 12/12, D1 25.904 against 25.889 ms); program 2, the main tail with the verify rows' KV stage -- the
decode `qsa_main_tail` program's norm, RoPE, head-split and query kernels on the tile's 32 rows with its staging cores
replaced by one KV core that reads the position P and the row count R from device scalars (the verify form's k + 1, the
commit form's accepted count: one program), reads the current cache block back, scatters rows j < R of the packed [v | k]
row into slot (P + j) & 31 of that block or of the next block's zero rows, zeroes the block's rows past the pass as the
chain's kept-0 rows are, and writes both blocks (the head split, two norms, two RoPEs, the concat, the block lookup, the
keep multiply, two one-hot placements, the add, two cache writes and the sparse-query concat / untilize / pad as one
program); bitwise the chain's on the four dies at rows 5 and 6 at the block's first row, inside, crossing the edge and at
its last row, in the single-row form and with the row count 0..5 read from the device, the untouched cache rows
byte-identical, 0.094 against the chain's 0.143 ms per traced call at five rows.  The family's gate on the line
(programs 1 + 2 against the composed glue, the `--mtp 4` server): the pass wall json 51.03 -> 49.65, prose 49.82 -> 48.56,
560-token chat 51.51 -> 50.25 ms (-1.38 / -1.26 / -1.26 ms per pass; the verify replay -1.0, the draft replay -0.33 --
the MTP layer's draft row runs the same main tail at one row), acceptance histograms and tokens per pass identical,
the A3-mtp4-32k pins 12/12 at the default and under `QWEN38_MTP_SAMPLED=0` (the json stream's sha equal to the composed
glue's), the plain-decode D1 band untouched (25.602 against 25.610 ms per step median over 200 steps).Since 2026-09-26 the slab has a fused form of its own, `gdn_prefill_rows` (on by default: bitwise the chain on the 4x p150
line in two independent runs -- pins 12/12, records 12/12, 3,232 agreement columns, 4/4 probes -- with the time to the first
token at 32k -13.1 %; a four-die device test guards the mesh-topology seam the one-die tests cannot see;
`QWEN38_FUSED_OFF=gdn_prefill_rows` restores the chain): the GDN body from the projection to the gated output as two rows-form programs around the unchanged chunk prims (called directly through
`ttnn.prim.chunk_gdn_prep` / `chunk_gdn_scan`), bitwise the chain's on one p150 die at 64 and 2048 rows, the recurrent
state and the FIR history included (PREFILL.md has the measured per-call figures).
`QWEN38_FUSED_OFF=<name>[,...]` (or `all`) in the server's environment falls back to the composed
chains; an unknown name in either variable refuses to start; `QWEN38_FUSED_OFF=gr_fold` runs the GR read's merged
three-program form with the stock collectives (5 programs per read); `QWEN38_FUSED_GR_READ_MERGED=0` runs the GR read's
split form (7 programs per read); `QWEN38_ROUTER_TAIL_LANES=0` runs the router tail's top-k on one core per tile (the
same program, the LLK's four passes on that core); `QWEN38_ROUTER_TAIL_EXP_LIVE=0` runs the router tail's exp over every
vector of every tile (the same bits, the full cost).

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

Measured 2026-09-25 on 4x p150 with the defaults of that day (BF8 dense weights, the fused GDN step, the compact expert
layout, the MoE post program's one-read routing, two readers per bank), the README's decode rows: one stream 27.2 ms per
token on the 200-step pin recipe (36.8 tokens/s greedy), flat with depth; the device sampler's step 27.4 ms (36.5
tokens/s sampled) on both model cards (thinking; instruct with its presence penalty); the batched-decode lane body at
4 / 8 lanes 34.6 / 42.7 ms per step, 28.9 / 23.4 tokens/s per user and 116 / 187 aggregate (the chat server serves one
stream; the lane sweep measures the lane body directly).

The slab's one-pass MoE combine (`moe_combine`, the default since 2026-09-26; `QWEN38_FUSED_OFF=moe_combine` restores the
512-row blocks) is bitwise class: it issues the fused reduce's own multiply-accumulate in its slot order on the
page's owned rows, and the line gate of 2026-09-26 (4x p150, 32k context, `--prefill-slab 2048`) read the twelve
acceptance records, the 3232 agreement columns and the four probes identical to the blocks', with the attention
identity through the stack unchanged; TTFT at 31,716 tokens 14.30 -> 12.38 s (`PREFILL.md`).  The one call's three
rings (`QWEN38_MOE_SLAB_RINGS` unset, the default since 2026-09-26) are bitwise the two rings on the same gate: records
12/12, columns 3232/3232, probes 4/4, in two runs (`PREFILL.md`).

`moe_compute`'s W2 ring exchange runs its handshake in the first a2a iteration only (2026-09-26, a runtime patch: the
partials travel once per chunk and the compute's later iterations read the resident buffers; the dropped iterations'
wait / increment pairs disappear on every core alike): bitwise on every form the op serves here (the decode rows, the
128-token chunk, the 2048-row slab: the 12-prompt pins and their token-stream digests, the `--mtp 4` pins and tokens
per pass, a 41-routing soak of the combine pages), 12.98 -> 11.6 us per distinct local expert per launch on the 1x4
line (-10 %), the B=1 step 27.14 -> 26.89 ms on the 200-step pin recipe, the sampled `--mtp 4` pass -0.84 ms per token
served, the slab's one call 2.09 -> 1.94 ms per layer (-7 %) on one die.

`moe_compute`'s a2a pipeline (2026-09-26, a runtime patch, the streaming decode ring only): compute runs W0/W1 of the
next owned chunk while dm1 exchanges the current chunk's partials (two parity slots of a2a buffers; one chunk's credit
may be outstanding), dm1 exchanges the next chunk before it writes this one's output rows, and the feed carries three
chunk slots (a fourth does not fit the slab-mode server's L1); the prefill ring forms keep the serial order. The same matmuls in the same order: the combine pages are
bitwise on the rows bench (11 cases, three builds); 11.42 -> 10.91 us per distinct local expert per launch on the 1x4
line (-4.5 %); the ring core is now paced by its DRAM weight stream (9.7-9.9 us per expert, about 285 GB/s of the
die's 512). The served numbers follow with the rebuild that carries it. The streaming decode ring's weight CB holds
four blocks instead of three (2026-09-26): 10.88 -> 10.68 us per distinct local expert on the same bench, bitwise;
five and six blocks and dual-NoC reads are no better -- the ring core's stream is paced by its one DRAM bank (about
36 GB/s of the channel's 64); the prefill ring forms keep three (their L1 has no room for a fourth).

## The slab's block-shared attention (`QWEN38_FUSED=sparse_sdpa_tiled`, 2026-09-25)

A tolerance-class fused kernel, opt-in: `sparse_sdpa_tiled` replaces the prefill slab's block-id expansion, zero V
half, head pad, `sparse_sdpa` and head slice with one program per QSA layer (`PREFILL.md`).  The attended set per query
and the bf16 scores are the chain's; the flash loop's order and chunk size differ, so the running max / sum / out round
elsewhere.  Judged as the slab itself is (the long windows against the references) and, per layer, against an fp32
oracle on the same bf16 inputs, where it must be no farther than the chain: on a QuietBox 2 die (2x p300c) with the
captured slab selections it reads PCC 0.99979 against the oracle (the chain 0.99978), row rel_err p50 0.021 (0.022),
PCC 0.99954 against the chain at P = 28672 and 0.99988 at P = 0; bit-exact run to run and across trace replays.  The
chunk forms and the decode path never see it.

The line gate (4x p150, 32k context, `--prefill-slab 2048`, 2026-09-26; the chain as the control; acceptance pins
12/12 and the acceptance columns identical on every arm; the 31,716-token long part identical through the first slab
and diverging from position 8128, where a query first attends across a slab boundary):

| arm | KL control -> arm, mean / max (long part) | top-1 vs the control | top-1 vs HF | TTFT 31,716 tokens | attention per 2048-row slab (device profiler, the prompt's first slab, 12 layers) |
|---|---|---|---|---|---|
| control (the chain) | - | - | 0.9875 | 15.92 s | 89.77 ms (`sparse_sdpa` 81.14 + expansion, pad, concat, slice 8.63); the slab's kernel time 899.5 ms |
| kernel, HiFi4 / fp32 DEST (`QWEN38_SPARSE_SDPA_TILED_FIDELITY=hifi4`) | 0.0059 / 0.55 | 159/160 | 0.9812 | 14.36 s | 18.45 ms; 818.6 ms |
| kernel, HiFi2 / bf16 DEST (the default) | 0.0028 / 0.084 | 160/160 | 0.9875 | 14.25 s | 16.28 ms; 816.6 ms (2,257 -> 2,484 tok/s over the slab) |

The HiFi2 form is the default: closer to the control and to the reference than the HiFi4 form (whose fp32
destination changes where the scores round) and 10 % faster to the first token at 32k; under the device profiler the
attention class of one slab is 5.5x shorter than the chain's (89.77 -> 16.28 ms) and the slab's kernel time 9 % shorter.

## The acceptance mechanism

`--acceptance` replays the twelve shipped CPU greedy records under `tools/acceptance/greedy-prompts/` against the CPU
before the server listens.  `--require-json-96` refuses to serve unless the `json` record matches the CPU 96/96 (the
other records diverge from the CPU after 2-75 tokens, the known greedy-prompt pattern; their first divergence index is
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
pass, 70.7 ms per pass); `--mtp 3` 38.0 median, 55.6 on `json`.  Sampled drafting (2026-09-25, `--mtp 4` on the
27.7 ms D1 step, the card profiles non-thinking / thinking): 40.8 / 40.7 tokens/s at 24.5 / 24.6 ms per token against
33.5 / 33.9 plain sampled, 2.64 / 2.68 tokens per pass, 0 fallbacks.

| | json | chat | code | fact | list | math | multilingual | prose | refactor | sky | story | summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain decode, 2026-09-25 | none (96/96) | 2 | 32 | 15 | 56 | 61 | 9 | 13 | 24 | 19 | 6 | 75 |
| `--mtp 4`, 2026-09-25 | none (96/96) | 43 | 32 | 15 | 56 | 63 | 9 | 13 | 24 | 19 | 6 | 1 |
| plain decode, 2026-09-06 | none (96/96) | 43 | 32 | 15 | 56 | 61 | 9 | 13 | 24 | 19 | 6 | 75 |
| `--mtp 4`, 2026-09-06 | none (96/96) | 56 | 32 | 15 | 46 | 56 | 9 | 13 | 24 | 19 | 6 | 1 |

Opt-in row, `QWEN38_FUSED=gdn_rows_scan` (2026-09-26, the served `--mtp 4` line; the pins above stay the wrap's until
the kernel's default flip, the user's decision).  Every number of this row was measured on the sealed fork runtime built at `e74d3b864ff`,
at head `93b9e857bd2` (the 2026-09-26 unified `5307c930416` plus the fold); the runtime kit of record is
now the fork runtime built at `5eac9c778edb` (the MoE a2a pipeline lives in the runtime C++), the fold-against-wrap comparison
stands as measured on the sealed kit; the kit of record's numbers (the fork runtime built at `5eac9c778edb`, `_ttnncpp.so`
`211530d77b2b`, the same line, 2026-09-26, the tree = unified `5eac9c778ed` plus the fold) are the numbers of record
and the sealed kit's the original measurement, both given below.  The committed stream leaves the CPU
reference at plain decode's token on 12/12 prompts (`chat` 2, `math` 61, `summary` 75 where the served row above reads
43 / 63 / 1), and its device token stream is bitwise plain decode's pinned stream on 8/12 (`json`, `chat`, `code`, `fact`,
`math`, `sky`, `story`, `summary`; the served row's pinned streams equal plain decode's on 5/12).  The greedy server
(`QWEN38_MTP_SAMPLED=0`) with the kernel on reads the same twelve rows.  On `list`, `multilingual`, `prose` and `refactor`
the stream leaves the reference at plain decode's token but its later tokens differ from both pinned streams (on
`refactor` the served row's stream equals plain decode's where the fold's does not): under the fold the GDN rows are
measured bitwise the 1-row fused step, and no other verify-rows form is measured against its 1-row counterpart row for
row -- the linears are the same tile programs at 1 and 32 rows, the residual read / write, RoPE and norms the same
per-row kernels, the QSA glue the decode's own kernels over the rows (bitwise the chain at rows 5 and 6), the MoE rows
form measured bitwise the 1-row path (rows 5, 2026-09-03/04; the down linear at rows 1 / 5 / 32, 2026-09-25), the
candidates' argmax exact -- so the forms bitwise by construction only, the attention over the R query rows, the PLE rows
body and the compressed-index block mean, are where a rounding difference can still sit: by elimination, not by
measurement; the near-tie check (the device's own 1-row logits at the first differing token) is the test that settles it.
Plain decode's own step is untouched: 25.612 against 25.619 ms per token over 200 traced steps with the kernel on / off.

| | json | chat | code | fact | list | math | multilingual | prose | refactor | sky | story | summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `--mtp 4` + `QWEN38_FUSED=gdn_rows_scan`, 2026-09-26 | none (96/96) | 2 | 32 | 15 | 56 | 61 | 9 | 13 | 24 | 19 | 6 | 75 |
| stream bitwise plain decode's pin | yes | yes | yes | yes | no | yes | no | no | no | yes | yes | yes |
| stream bitwise the `--mtp 4` pin | yes | no | yes | yes | no | no | no | no | no | yes | no | no |
| fold stream sha256 (first 12) | a5b4defa72fa | 34686e9146e7 | 0234032be259 | 9afd57cc0303 | ea554701b6f5 | 19f55ccb4bb2 | e83d74f6b8bd | 129b10287cbd | 8a812a8ea01b | de86bbc09c48 | d1897b72f57d | d9d024681580 |

The pass (measured by the MTP lead on a second 1x4 p150 line at the landing head, 120 + 20 passes per cell,
`--stop-at-end`): the fold removes 2.9-3.1 ms of program time from a k = 4 pass (the commit's masked re-run of the two
chunk prims becomes one pick, 0.91 against 3.13 ms; the verify's six GDN programs per layer become one, -0.55..-0.62;
the draft replay equal) and 0.8-1.3 ms of the pipelined wall on every prompt, since the commit replay already overlapped
the host's PLE lookup: fused verify, greedy, p50 json 47.84 against 49.17 ms, prose 47.23 against 48.24, 560-token chat
49.23 against 50.38 (the all-blocking probe 49.15 / 48.21 / 50.07 against 52.03 / 51.07 / 53.19) on the sealed kit, and
on the kit of record (runs re5-decomp-on / re5-decomp-off) json 47.42 against 48.63, prose 46.69 against 47.78, 560-token
chat 48.76 against 49.73 with the same tokens per pass; split verify, sampled,
json 51.61 against 51.81, prose 50.59 against 51.55, 560-token chat 52.71 against 53.88 (probe -3.01 / -2.91 / -2.89,
the sampled tail unchanged; json's stream identical at 4.750 tokens per pass, prose 2.033 against 2.000 and the chat
2.951 against 2.527 a different sampled continuation of one seeded request).  Served tokens/s move with the stream, not
with the pass: the fold's stream is plain decode's, so a request's first ~96-150 tokens are identical under the fold and
the wrap and then a different continuation with its own draftability follows; no systematic acceptance loss is
attributed (the fold's drafter is not less accurate on its own committed stream than the wrap's on its: `fact`, an
identical 67-token answer, 2.58 against 2.48 tokens per pass, depth-1 acceptance 0.808 against 0.741, one acceptance
apart; `summary`, 3.31 against 2.67 tokens per pass over its own 86- / 64-token answer).  Served rows, EOS honoured
(the lmx-like client, one warm-up and the median of 3, greedy, at most 256 tokens): the kit of record's columns are the
numbers of record (runs re5-rows-on / re5-rows-off, servers `20260926T162840Z-3851428` fold and `20260926T162159Z-3842121`
wrap; the wrap's rows are the README's), the sealed kit's the original measurement:

| served greedy, tokens/s over the answer to its end marker | fold, kit of record | wrap, kit of record | fold, sealed kit | wrap, sealed kit | tokens per pass, fold / wrap |
|---|---|---|---|---|---|
| 560-token chat (256 tokens, finish length: the answer exceeds 256) | 62.52 | 61.30 | 61.70 | 60.55 | 3.08 / 3.12 |
| 177-token multi-turn chat (256, length; all answer) | 57.60 | 57.62 | 56.89 | 57.00 | 2.80 / 2.83 |
| code (256, length; the answer is 283 tokens) | 95.51 | 91.93 | 94.16 | 90.99 | 4.65 / 4.57 |
| json (153 tokens, stop) | 97.27 | 95.25 | 95.74 | 94.02 | 4.75 / 4.75 |
| prose (stop; the fold's answer 203 tokens, the wrap's 196) | 43.42 | 44.73 | 42.87 | 44.28 | 2.09 / 2.17 |
| sampled, ms per token (non-thinking / thinking) | 19.67 / 20.93 | 19.75 / 19.90 | 19.91 / 20.56 | 19.90 / 20.20 | |

The wrap's kit-of-record rows sit 0.7-1.3 tokens/s above the sealed kit's (the MoE exchange pipelining), lengths and
tokens per pass identical; at 256 forced tokens the kit of record reads fold 62.54 / 76.87 / 57.62 / 37.13 / 95.49
against wrap 61.29 / 92.03 / 57.68 / 45.49 / 91.92 (560-token chat / json / chat / prose / code).

At 256 forced tokens (`ignore_eos`, the previous form) the same requests read mixed by stream -- 560-token chat +1.9 %,
code +3.8 %, json -16.7 %, prose -18.4 %; the 12 acceptance records net -2.1 % at 256 tokens and +1.2 % on the 96-token
replay; `summary` -47.5 % where the wrap's post-marker text repeats and drafts at depth-1 0.864 and the fold's does not
(0.370), which never enters a verdict.  The 43-prompt pair in that form is contaminated the same way (sum +3.2 %, paired
mean +1.94 +- 1.49 tokens/s, n = 43); the flip-decision sample is its EOS-honoured re-run: Over 43 distinct prompts -- the 12 acceptance records, the reference corpus's 16 served renders (12 with
thinking on and 4 requests with tools), its 4 evaluation items, the two books' first 256 / 512 / 1024 tokens as raw user
text, and the synthetic chat prompt at 128 / 256 / 560 / 1024 / 2048 rendered tokens -- one greedy request each with EOS
honoured (max 256; 21 of the 43 stop before 256), tokens per pass from the server's pass counters: the fold reads 3005.8
against the wrap's 2947.4 tokens per second summed (+2.0 %), 30 prompts up and 13 down; the paired mean delta is +1.36
+- 0.61 tokens per second (standard error, n 43) and the mean relative delta +2.44 % +- 1.09 %; no part of the set is
negative on average (acceptance records +2.6 %, served renders +1.1 %, evaluation items +1.1 %, synthetic chat +5.9 %,
book prefixes +3.9 %).  Where the two arms' answers are identical the fold's tokens per pass is equal or one acceptance
higher.  The
lmx-like client's `ignore_eos` form matched the localmaxxing reference's 256 output tokens; the reference record does
not say whether EOS was forced.

The sampled law gate under the kernel (the host-fp32 derivation; json / chat / code / story at offsets 0 / 16 / 48,
4,096 seeds, 32 loop requests of 10 tokens) passes on both cards, 12 positions each, loop mismatches / fallbacks /
guard deviations 0 / 0 / 0; the rows-against-plain total variation is 0.0 at the median on both cards, max 0.0253
(thinking) and 0.2227 (non-thinking) where the wrap-era rows of record read 0.0083 / 0.0438.  The non-thinking maximum
is one position, read from the gate's position dump: at `chat` offset 48 the fold's verify row puts the top logit one
bf16 step below the 1-row tail's (20.625 against 20.75, the runner-up equal), 0.029 of probability before truncation;
the non-thinking nucleus cut at 0.8 sits between the two (0.8066 against 0.7772), so the runner-up survives under the
fold and not under plain decode -- the 0.2227 total variation is that one token's mass; the argmax is unchanged, and
the wrap's row equals plain decode's at this position.  The layer that moves the logit is not attributed (the fold's
identity with the 1-row step was measured on GDN layers 01 and 33).

Per GDN layer on the 1x4 line (30 traced replays, quiet host): the verify forward 0.459 -> 0.271 ms and the commit
0.136 -> 0.051-0.058 against the chain (the wrap's 0.306 / 0.110), device operations 58 -> 10 and 12 -> 3.  The fold's
DRAM stays the clause's (k + 1) x 786,432 bytes per GDN layer per device, 141.6 MB per device at k = 4, per drafting chain.

## The MTP pass decomposition (2026-09-25)

Measured on the 4-chip p150 line (a shared host) at the landed head with the chain opened as the server opens it,
`--mtp 4`, the `json` and `prose` acceptance prompts, 120-280 pipelined passes per cell and 20-40 all-blocking probe passes
(every trace replayed blocking on its own: its raw device wall), the device profiler for the per-family split (its per-program
markers inflate a trace by 3-4 %; the raw replay walls are the timing numbers).  The pass wall does not depend on what is
accepted: with the host decision stubbed to accept-all / reject-all the json pass is 62.20 / 62.00 ms against 62.14 normal.

| term of one k = 4 pass (json greedy, ms) | fused verify (`QWEN38_MTP_SAMPLED=0`) | split verify (the default, greedy) | split verify, sampled |
|---|---|---|---|
| commit trace (the previous pass's a*+1 rows; hides the host's PLE lookup) | 4.07 raw, ~1.3 exposed | 4.06 | 4.06 |
| PLE n-gram rows of the k+1 tokens (host; the table is host memory) | 2.8, hidden | 2.7 | 2.7 |
| verify body (48 layers on the 5-row tile, LM head, rows argmax, accept, alignment) | 47.35 raw | head 45.25 + tail 2.79 | same |
| host between head and tail | - | read 0.67, decision 0.02, writes 0.86 | read 0.67, decision 0.6 + 0.8 per row evaluated (4.6 at a* = 4; 0.7 batched), writes 0.84 |
| draft trace (k - 1 rows: embed, mixer, MTP layer, rows-1 MoE, final mixer, LM head, the fused greedy tail) | 7.38 raw = 3 x 2.46 | 7.33 = 3 x 2.44 | 7.33 |
| pass-row readback (268 B, the one host read) | 0.72 | 0.54 | 0.52 |
| pipelined pass wall, p50 [p90] | **60.28** [61.54] | **62.14** [63.40] | **66.92** [68.07] |
| tokens per pass / tok/s over the passes | 4.657 / 76.9 | 4.657 / 74.6 | 4.677 / 70.0 |

Prose: fused 58.92, split greedy 60.86 (2.529 tokens per pass, 41.3 tok/s), split sampled 63.28 (2.583, 40.5).  The verify
body is flat in k inside the rows-5 MoE form (raw head 44.2 / 45.2 / 46.0 / 45.3 ms for 2 / 3 / 4 / 5 rows) and 1.6x the
27.7 ms one-row decode step even at two rows: the rows forms are the cost, not the row count.  Each draft row adds 2.4 ms of
device time (draft trace 0.14 / 2.51 / 4.89 / 7.33 ms for 0 / 1 / 2 / 3 rows); k = 1 / 2 / 3 / 4 give json 1.98 / 2.92 / 3.18 /
4.66 tokens per pass at 53.3 / 56.2 / 60.6 / 62.1 ms (37.0 / 51.5 / 52.4 / 74.6 tok/s).  k = 5 is refused by the MTP DRAM
admission on this head (the 32-row MoE verify form's states).

The verify head by family (device profiler, chip 0 kernel ms of 43.0 over 4,479 programs; 36 GDN layers at 833 us, 12 QSA
layers at 986 us, head epilogue 1.2):

| family | GDN layers | QSA layers | total |
|---|---|---|---|
| MoE compute (156 us per layer at 5 rows against ~65 at 1 row) + the combine collective (58 us) | 7.73 | 2.58 | 10.30 |
| fused programs (gated-residual read/write, router top-k, dispatch, MoE post) | 7.75 | 2.48 | 10.23 |
| dense matmuls (projections, shared expert, router) | 3.95 | 1.63 | 5.57 |
| Chunk-GDN prep + scan (the recurrence over the rows) | 1.87 | - | 1.87 |
| composed glue (binary / unary / reshape / transpose / slice / concat / norms / rope / tilize / KV update / indexer / SDPA) | 8.70 | 5.15 | 13.85 |

One draft row (254 programs, 2.30 ms kernel, 0.18 ms dispatch): LM head 0.54 (eight weight-bound chunk matmuls), the MTP QSA
layer ~0.7, the input mixer / norm chain ~0.6, the rows-1 MoE ~0.33, the final mixer ~0.12, the resolve 0.055 (the fused greedy
tail).  Ceiling at five tokens per pass = 5 / pass wall: fused greedy 82.9 (json) / 84.9 (prose) tok/s, split greedy 80.5 / 82.2,
split sampled 74.7 / 79.0; accept-all realizes 80.0 on json.

k = 5 (`--mtp 5`, admissible since the k-aware DRAM estimate) against k = 4 on the same runtime (2026-09-26, the 4-chip p150
line, 32k): the pass costs 65.4-67.4 ms in the split form (k = 4: 62.1-63.3) and about 68.5 ms served on the fused greedy
form (k = 4: 60.2, from tokens per pass and tok/s), +4 to +8 ms per pass for one more draft row (+2.4 ms of draft trace):
the 6-row verify tile and the 32-row MoE verify form pay the rest.  Served 256-token greedy rows, k = 5 against k = 4: json
75.6 against 75.6 tok/s (5.18 against 4.55 tokens per pass), code 77.1 against 75.7 (5.24 / 4.57), the 177-token multi-turn
chat 45.6 against 47.5 (3.10 / 2.83), the 560-token chat 42.1 against 50.6 (2.88 / 3.12), prose 26.8 against 37.3 (1.76 /
2.20).  k = 4 stays the default; k = 5 is opt-in.  The k = 5 stream is not bitwise with k = 4's (the pass partition moves the
commit chunks and the verify MoE runs the 32-row instance): `json` stays 96/96, `code` holds the reference to 96 (k = 4
leaves at 32), `chat` leaves at 39 (43), `list` 46 (56), `math` 61 (63), `multilingual` 9 (9, a different tail): the
divergence tokens are the reference's #2 or #3 at fp32-oracle margins of 0.09-0.36 logits (chat 39: 0.27, list 46: 0.09,
math 61: 0.36, multilingual 9: 0.11; the k = 4 positions: chat 43 0.04, list 56 0.42, math 63 0.04, code 32 0.42), the
near-tie class above (the "0.4-1.0 logits apart" wording of the 2026-09-25 table holds for list and code; chat and math sit
at 0.04).  Per-depth acceptance
rows measured with a fixed completion budget and no stop ids overstate short-answer classes: `json` reaches `<|im_end|>`
after about 151 tokens and then repeats its answer, so a 600-token json row runs about 75 % in that loop; read the passes
before the first end marker.

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

The candidate row itself (`candidate_row`, 2026-09-25) is folded into greedy_tail's scan and merge: each scan core keeps
its sorted top-32 behind the argmax's compare (a stable insertion, so ties keep the lowest ids and the list's first
entry is the argmax pair; the greedy outputs are unchanged), and the merge takes the shard's 32 from the cores' lists in
core order and writes the fp32 row [values | global ids] the row's all_gather takes: the row's 10 programs and 312 us
per step become 2 and 8 (the gather and the copy) while the scan and merge grow 66 -> 128 us (net -242 us on the tail),
the sampled step 27.57 -> 27.37 ms. The values are bitwise the chain's (ttnn.topk's) and the ids equal above the kth
value; the boundary group is the lowest ids among the ties, where ttnn.topk's pick is implementation-defined (the
candidate row's agreement rule tolerates boundary ties). `QWEN38_FUSED_OFF=candidate_row` restores the chain's row.

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

With `QWEN38_MTP_DEVICE_ACCEPT=1` an `--mtp --sampling` server decides its passes on the device under the same law
(`mtp_accept`, one program on one core between the verify head and its tail: row `j` accepts draft `d_{j+1}` iff
`fl32(u_j S_j) < w_j(d_{j+1})` over the table weights, a tie rejects, the first rejection and the bonus row draw with a
second uniform, no division); the response's `mtp_acceptance_arithmetic` names `device-theta` or `host-fp32` (the
host-decided pass, the host sampler's fp32 law) and the fingerprint carries `-device-accept`: one seed reproduces one
stream per arithmetic; the law gate runs per arithmetic (2026-09-26).

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
