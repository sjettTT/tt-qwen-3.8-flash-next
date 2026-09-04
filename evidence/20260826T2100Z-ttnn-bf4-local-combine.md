# TTNN BF4_B / true-global-B1 local-combine gate

- UTC: `2026-08-26T21:00:13Z`
- Host: `f07cs02`
- Repository HEAD: `8b4e56ea84f1362a5f838f1baae77e0f939fc42c`
- tt-metal provenance revision: `181ac080751bccbaea2e5106fdf880f4b1ca04c4`
- Checkpoint revision: `f5d08274bafd880402bd16f5e3e6c514136ec06c`
- Checkpoint tensor manifest SHA-256:
  `ebf4de2015233c20e21ff1e7e388e871208158ed350fea42964dbd52ca69359b`
- Scope: TTNN-only continuation. Frozen CPU/MTP files were neither changed nor
  executed; `tools/run_full_cpu_mtp_oracle.py` was not run.

## Packed memory admission

The exact `moe_compute_utils` host packer, rather than raw BF4 tile payload,
defines the routed allocation:

| live BH ring | W0/W1 bytes/device/layer | W2 bytes/device/layer | 49 layers | streamed total static weights/device |
|---|---:|---:|---:|---:|
| 7 | 346,816,512 | 130,056,192 | 23,366,762,496 | 3,021,063,680 |
| 8 | 396,361,728 | 148,635,648 | 26,704,871,424 | 3,089,188,352 |

The full exact manifest report also accounts for 2,490,901,760 bytes/device of
TP4 BF16 matrices, 34,078,720 grouped-QSA-KV bytes, 17,039,360 split indexer
bytes, and 2,171,136 replicated BF16 bytes. All-resident routed experts fail the
existing runtime-reserve gate; one-layer serialized streaming passes it.

## Implemented source boundary

- Added `moe_compute(local_combine=True)`. It is legal only with
  `compute_only=False`, no CCL/mux/semaphore options, no fused shared experts,
  fully replicated logical-token input topology, and a selected mesh axis of
  size one.
- On `(1,4)`, axis 0 yields one true logical token at every coordinate and no
  fabric dispatch. FullLocal output topology is explicitly copied from the
  replicated token rather than inferred from expert-sharded weights.
- Added a per-mesh-coordinate DRAM-bank worker query so BF4 cache qualification
  checks all four physical cards instead of using the mesh's first-device
  compatibility query.
- Added Python TTNN contracts, BF4_B conversion/cache/streaming, and routed plus
  dynamic-shared MoE modules under
  `models/demos/blackhole/qwen38_flash_next/ttnn/`.
- The mapping is replicated RM UINT16 `[4,512]` with identical rows and exact
  contiguous 128-expert owners. Indices/scores are copied to the exact tilize
  drain-core L1 height shard. Local combine output is cleared every invocation
  before the writer fills only locally owned top-k slots.
- The DeepSeek reducer receives the real normalized scores; the separately
  sharded shared expert uses its own dynamic sigmoid scalar. The final additive
  partial is reduce-scattered over axis 1 and tagged as hidden-dimension sharded.
- Cache creation holds a bounded process lock, stages both tensorbins under a
  unique temporary directory, publishes with same-filesystem atomic renames,
  hashes the final artifacts, and rejects any unmanifested final tensorbin.

## Build and no-device results

Command:

```text
cmake --build /home/sjett/qwen38-flash-next-data/build/tt-metal-181ac08075-release --target ttnn/_ttnn.so -j 8
```

Result: pass, 4 incremental steps, 24.23 seconds.

- `_ttnn.so` SHA-256:
  `1911ac8b460f0d5d4b943319cf121c22ab5484cd96edfe3728c4dae94ff9cbc6`
- `_ttnncpp.so` SHA-256:
  `4eb634018f466d5329d03c5b15cd141616cf706e52dcf2d3ff7bbfceb32eb61c`
- Qualified no-device runtime:
  `/home/sjett/qwen38-flash-next-data/runtime-local-combine-ring-qualified-20260826`
- The no-device import resolved that runtime, found the `local_combine` kwarg,
  loaded `ttnn.operations.ccl.MoEActivationFunction.SILU`, and found the new
  per-coordinate DRAM query.
- `ldd` found no missing dependency on this host. `readelf` records absolute
  RUNPATH entries into this exact local build, so this bundle is not accepted
  for cross-host reuse without the prompt's full ABI/RPATH qualification.

Static command result:

```text
pytest -q test_ttnn_bf4_static.py test_checkpoint_budget.py
8 passed, 10 subtests passed
```

`py_compile` and `git diff --check` also passed for the TTNN/BF4/MoE boundary.

## Hardware state and remaining gate

No TT device was opened. `TT_VISIBLE_DEVICES=0,1,2,3` remains present, and the
continuation launcher log records PID 4106323 plus locks on FDs 200--203. The
command-executor namespace does not expose that PID and intentionally closes
the high descriptors in grandchildren, so visibility alone is not treated as
a job lease. Failed f07cs02 devices 0--3 were not retried or reset; devices 4--7
and candidate hosts were not touched.

Blackhole issue #50038 remains relevant: upstream tests skip/xfail some
matmul-output numerical checks on BH, including non-tile token counts. The new
H=2560, I=640, E-local=128, B1 FullLocal path therefore requires a hard device
numerical gate covering cache miss/hit, every owner distribution, a device with
zero selected experts, output topology, exact weighted reduction, and cleanup.
