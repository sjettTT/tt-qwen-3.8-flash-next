# Qwen3.8-Flash-Next four-P150 memory and placement gate

Status: TTNN BF4_B streaming admission budget, pinned to checkpoint
`f5d08274bafd880402bd16f5e3e6c514136ec06c` and repository HEAD
`8b4e56ea84f1362a5f838f1baae77e0f939fc42c` plus the task-owned TTNN changes
recorded in `RUN_MANIFEST.md`.

This budget is for the required text backbone plus its real MTP layer on one
four-device mesh. The vision tower is intentionally not loaded. It is not a
claim that the runtime has allocated these tensors yet; actual TTNN topology,
allocator free space, and tensor storage must agree before the full load is
admitted.

## Exact source payload

The bounded safetensors-header manifest contains 1,658 tensors and passes the
release contract in `checkpoint_budget.py`:

| Domain | Tensors | Elements | Checkpoint bytes | Runtime disposition |
|---|---:|---:|---:|---|
| Routed gate/up, 48 target + 1 MTP layer | 49 | 82,208,358,400 | 164,416,716,800 | BF4_B, expert-parallel |
| Routed down, 48 target + 1 MTP layer | 49 | 41,104,179,200 | 82,208,358,400 | BF4_B, expert-parallel |
| Backbone non-expert | 1,059 | 3,643,460,480 | 7,286,920,960 | TP4/grouped/replicated as below |
| Token embedding | 1 | 635,699,200 | 1,271,398,400 | vocab-sharded TP4 |
| LM head | 1 | 635,699,200 | 1,271,398,400 | vocab-sharded TP4 |
| MTP non-expert | 29 | 90,568,448 | 181,136,896 | same TP4 rules as target |
| PLE device-side projection/conv | 6 | 32,839,680 | 65,679,360 | TP4 |
| PLE n-gram table | 128 | 51,200,245,760 | 102,400,491,520 | host-only BF16 mmap/lookup |
| PLE integer metadata | 3 | 35 | 280 | host-only, copied only when required |
| Vision tower | 333 | 448,931,056 | 897,862,112 | retained on disk, not loaded |
| **Exact checkpoint total** | **1,658** | **179,999,981,459** | **359,999,963,128** | |

The 35 non-BF16 elements are I64 PLE hashing metadata; the other
179,999,981,424 elements are BF16. The manifest SHA-256 is
`ebf4de2015233c20e21ff1e7e388e871208158ed350fea42964dbd52ca69359b`.

## Tensor placement contract

| Tensor/state family | Four-device placement | Per-device logical share | Required topology assertion |
|---|---|---:|---|
| Routed experts | shard expert axis | exactly 128 of 512 experts | shard, never replicated; physical expert ranges are disjoint |
| Router logits | shard 512 outputs, then global selection | 128 logits before global top-10 | global top-10 indices/scores equal CPU oracle |
| Hidden dimension | TP4 | 640 of 2,560 | distribution shape and shard axis match |
| Token embedding / LM head | shard vocabulary rows | 62,080 of 248,320 | four disjoint vocab ranges |
| GDN Q/K heads | head-sharded | 4 of 16 | four disjoint head ranges |
| GDN value heads/state | head-sharded | 12 of 48 | recurrent FP32 state follows the same head range |
| QSA query heads | head-sharded | 6 of 24 | four disjoint Q ranges |
| QSA KV heads | two explicit head groups | one of two KV heads, replicated on two devices | two distinct KV values globally; replication only inside the owning pair |
| QSA index query heads | head-sharded | 1 of 4 | four disjoint query-index heads |
| QSA index key head | explicit replication | one shared key head on every device | placement says replicate; values and metadata agree |
| GR branches / hidden | four branches retained, hidden-sharded | 4 x 640 | never collapse branches; low rank is 80 of 320 where sharded |
| PLE n-gram table | host-resident, 128 checkpoint shards | zero device table bytes | exact hash/EOS history and 16 head lookups; transfer 4 x 640 result shards |
| Shared expert and dense projections | tensor-parallel | one quarter payload | collectives reconstruct the reference result |
| MTP layer | same expert/QSA/GR placement as target | included in every byte total | distinct mutable cache with explicit commit/rollback |

The QSA KV grouping is deliberate. Treating two KV heads as a four-way head
shard is invalid; replicating both heads on all four devices is also rejected.
The concrete pairing will be bound to verified mesh coordinates, not numerical
logical IDs, after the physical fabric test.

## Static device payload

The `moe_compute` packer pads each local expert matrix to the live Blackhole
DRAM-bank ring, so raw element-count BF4 arithmetic is not an admission bound.
For `H=2560`, `I=640`, and 128 local experts, the exact packer output is:

| Live ring | W01 bytes/layer/device | W2 bytes/layer/device | Routed bytes for 49 layers/device | Result |
|---|---:|---:|---:|---|
| 7 banks | 346,816,512 | 130,056,192 | 23,366,762,496 | full residency rejected |
| 8 banks | 396,361,728 | 148,635,648 | 26,704,871,424 | full residency rejected |

The correctness path therefore streams one distributed packed expert layer at
a time into a fixed device slot. Conversion and cache identity bind the live
ring size and per-bank logical-worker ordering on all four physical devices;
mixed or differently ordered rings fail closed. The static payload becomes:

| Per-device item | 7-bank bytes | 8-bank bytes |
|---|---:|---:|
| One streamed packed BF4_B expert slot | 476,872,704 | 544,997,376 |
| Ordinary dense/matrix payload, conservative BF16 TP4 bound | 2,490,901,760 | 2,490,901,760 |
| QSA KV projection, grouped BF16 | 34,078,720 | 34,078,720 |
| Split QSA index query/key projection, BF16 | 17,039,360 | 17,039,360 |
| Conservatively replicated 1-D BF16 parameters | 2,171,136 | 2,171,136 |
| **Streamed static weights/device** | **3,021,063,680** | **3,089,188,352** |

All 49 packed host layers occupy about 106.82 GB aggregate for an 8-bank ring.
They are a local conversion cache, not simultaneously device resident. Initial
correctness execution uses one slot and one command queue; overlap or double
buffering is deferred until values, ordering, and ownership pass.

The Blackhole descriptor exposes eight DRAM views of 4,278,190,080 bytes each:
34,225,520,640 raw bytes/device (31.875 GiB). Runtime-reserved bases and trace
regions reduce that raw value. A bounded TTNN allocator probe must record the
actual allocatable number before loading weights.

## Batch-one 8K persistent state

The first full-model qualification target is 8,192 tokens, not the native
262,144-token maximum. Per-device baseline state is:

| State | Formula | Bytes/device |
|---|---|---:|
| QSA target + MTP KV | 13 layers x 8,192 x one grouped KV head x 256 x K/V x BF16 | 109,051,904 |
| QSA index raw keys | 13 x 8,192 x 128 x BF16, replicated | 27,262,976 |
| cached 3-axis positions | 3 x 8,192 x I64 | 196,608 |
| GDN recurrent state | 36 x 12 local V heads x 128 x 128 x FP32 | 28,311,552 |
| GDN causal-conv state | 36 x 2,560 local channels x 4 x BF16 | 737,280 |
| PLE dilated-conv state | 2,560 local channels x 9 x BF16 | 46,080 |
| PLE two-token history + one live four-branch residual | exact scalar/live tensors | 5,136 |
| **Persistent-state subtotal** | | **165,611,536** |

Speculative snapshots, rejection replay, selected-index buffers, causal-tail
scratch, and operator outputs are accounted in runtime reserves below. The MTP
layer is included as the thirteenth QSA cache layer.

## Admission reserve and precision decision

| Per-device admission item | 7-bank bytes | 8-bank bytes |
|---|---:|---:|
| Streamed static weights | 3,021,063,680 | 3,089,188,352 |
| Exact batch-one 8K persistent state | 165,611,536 | 165,611,536 |
| Dense layout/padding/metadata reserve | 1,073,741,824 | 1,073,741,824 |
| QSA selection/tail/verifier scratch reserve | 268,435,456 | 268,435,456 |
| MoE/collective/activation runtime workspace reserve | 4,294,967,296 | 4,294,967,296 |
| Trace region reserve | 1,073,741,824 | 1,073,741,824 |
| Allocator fragmentation/emergency reserve | 2,147,483,648 | 2,147,483,648 |
| **Admission total/device** | **12,045,045,264** | **12,113,169,936** |
| **Headroom versus raw descriptor capacity** | **22,180,475,376** | **22,112,350,704** |

This admits one streamed BF4_B routed layer with BF16 activations and a
conservative BF16 non-expert bound. It does not claim useful decode performance:
without overlap, the fallback transfers roughly 26.7 GB/card per generated
token for an 8-bank ring. That is acceptable only as the exact correctness path
while a resident or more granular schedule is measured and qualified.

For comparison, keeping all packed routed layers resident fails the conservative
admission gate:

| Live ring | Full packed static/device | Static plus reserves/device | Result |
|---|---:|---:|---|
| 7 banks | 25,910,953,472 | 34,934,935,056 | exceeds raw capacity by 709,414,416 bytes |
| 8 banks | 29,249,062,400 | 38,273,043,984 | exceeds raw capacity by 4,047,523,344 bytes |

BF8_B exploration is deferred by the current execution directive and is not an
alternative admission path. No FP8 checkpoint is treated as a numerically
interchangeable source for BF16-to-TT conversion.

## Executable gate

Run the deterministic accounting check with:

```bash
python3 -m models.demos.blackhole.qwen38_flash_next.tools.checkpoint_budget \
  /home/sjett/qwen38-flash-next-data/sources/huggingface/tensor-manifest-f5d08274.tsv
```

The tool rejects unknown tensor names, any mismatch from the exact release
manifest, non-512 expert axes, non-tile-aligned expert matrices, a mesh other
than four, and any payload that the declared placement cannot divide exactly.
Hardware loading adds a second gate: every created tensor must expose the
expected `distribution_shape()`, `placements()`, `mesh_coords()`, and
`tensor_topology()` metadata, and measured DRAM usage must stay within this
admission total.
