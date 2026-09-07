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

## The acceptance mechanism

`--acceptance` replays the twelve shipped CPU greedy records under `tools/acceptance/greedy-prompts/` against the CPU
before the server listens.  `--require-json-96` refuses to serve unless the `json` record matches the CPU 96/96 (the
other records diverge from the CPU after 6-75 tokens, the known greedy-prompt pattern; their first divergence index is
in `acceptance.json` in the run directory).  The records were rendered by the CPU study with the system prompt
`You are a helpful assistant.`; that system turn is inside their recorded prompt ids, which the replay feeds to the
device as they are.  The server itself adds no system prompt to a client's request (`SERVER.md`).

## The pinned table: chunked prefill, 32k, 2026-09-06

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
three); the cost is one more program per layer, about 0.4 ms per decoded token (50.0 -> 50.4 ms).  The proofs in
`PROOFS.md` predate this fix where they quote the old indices.

## `--long-chunks`

The 128-row chunk path gives the same tokens as 32-row chunks alone, bitwise on all 48 layers; on the QuietBox
(2026-09-06, before the gate fix) the acceptance replay with `--long-chunks` was identical to the plain start (`json`
96/96, the same eleven divergence indices, 19.4-19.6 tokens/s).

## MTP (`--mtp 4`), 32k, 2026-09-06

The MTP path is not bitwise with plain decode on 4 of the 12 acceptance prompts (measured 2026-09-06 with the GDN gate
fix above; the verify rows and the 1-row loop round differently): the committed stream leaves the CPU reference on
`chat` at token 56 where plain decode leaves at 43, on `list` at 46 against 56, on `math` at 56 against 61 and on
`summary` at 1 against 75 (`In 1947,` becomes `Invented at Bell Labs in`); the other eight records leave the reference
at the plain-decode token (`json` 96/96, `code` and `refactor` bitwise the plain streams), and every gate passes.  At
each of the three earlier tokens the device's own 1-row logits hold the CPU's token and the MTP token within one bf16
step (an exact tie on `summary` and `list`, which the 1-row loop breaks toward the lower id), the verify row lands one
step the other way, and the CPU oracle rates the two 0.4-1.0 logits apart: near-ties, not a defect (the rows-path gate
is the 1-row gate bitwise on the same row; the pinned MTP table is `tools/ci/baselines/A3-mtp4-32k-divergence_index.json`).
Before the gate fix the two paths differed on `code` (44 against 24) and `fact` (16 against 15) only.

| | json | chat | code | fact | list | math | multilingual | prose | refactor | sky | story | summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain decode | none (96/96) | 43 | 32 | 15 | 56 | 61 | 9 | 13 | 24 | 19 | 6 | 75 |
| `--mtp 4` | none (96/96) | 56 | 32 | 15 | 46 | 56 | 9 | 13 | 24 | 19 | 6 | 1 |

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
