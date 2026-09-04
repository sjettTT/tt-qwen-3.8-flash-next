# Qwen3.8-Flash-Next bring-up status

Last update: `2026-08-26T22:30:08Z`

## Current post-maintenance phase

Hardware work is stopped at a classified `runtime_unverified` blocker.  The
single post-reset hardware-opening attempt authorized by the user was consumed
at `2026-08-26T22:28:23Z`.  The qualified extension imported correctly from
`/tmp` and emitted its exact path, SHA-256
`1911ac8b460f0d5d4b943319cf121c22ab5484cd96edfe3728c4dae94ff9cbc6`,
build ID `91ee0b9ea782dd46bbf354e841fa337cece291fb`, and both required coordinate
bindings before discovery.  UMD then discovered IDs 0--3 and immediately
failed in `GetNumAvailableDevices()` because installed TTNN had selected its
unbundled package directory as the runtime root:

```text
RuntimeError: bad file: /home/sjett/qwen38-flash-next-data/runtime-local-combine-ring-qualified-20260826/site-packages/ttnn/tt_metal/soc_descriptors/blackhole_140_arch.yaml
```

Fabric was never configured.  The process destroyed its UMD cluster and left
no handle.  The post-run offline snapshot remained healthy on all four A
cards.  A no-device source diagnosis proved that
`TT_METAL_RUNTIME_ROOT=/home/sjett/tt-metal-qwen38-flash-next-agent-20260826`
selects the present, pinned SoC/core descriptors; the topology gate now
requires their exact paths and hashes before any future discovery call.  It was
not rerun.  See `BLOCKED.md` and
`/home/sjett/qwen38-flash-next-data/evidence/20260826T222657Z-strict-topology-gate-a`.

The frozen CPU oracle/MTP files remain untouched and were not run.  All
subagent worktrees and the main dirty/untracked state remain preserved.  No
partition-B device, reset, power cycle, KMD reload, service stop, process kill,
or relaxed fabric mode was used.

## Preserved implementation state before maintenance

The user-directed continuation freezes all CPU-oracle/MTP work and makes TTNN
the only critical path. Existing CPU goldens and their logs are preserved and
`tools/run_full_cpu_mtp_oracle.py` has not been run or changed. The TTNN path
now has fail-closed `(1,4)` tensor-topology contracts, exact ring-aware BF4_B
packing/cache identity, one-layer routed-expert streaming, and a true-global-B1
MoE design. A new `moe_compute(local_combine=True)` mode executes independent
128-expert partials on the four coordinates, with real normalized top-10 scores
applied by the DeepSeek reducer and the dynamic shared expert kept separate.
The C++ and Python runtime import gates pass; numerical silicon evidence is
still required. TP4 gated residual, GDN, vocabulary-sharded embedding/LM head,
QSA/KV, and the decoder-layer shell are integrated as commits `fedcac6c0e`,
`a92fcfc78e`, `c6d7a2b3ce`, `6b52ab874d`, and `66c36a9f50`. Exact PLE and the
terminal mixer are present in the task-owned diff. Cross-module review found
and fixed a provisional decoder-layer/PLE result-type mismatch before device
use; the layer now owns the canonical fixed-address PLE state and one-shot
snapshot. The ordinary 48-layer model owner and its provenance-bound builder
are integrated through commit `76ab274cca`: the exact no-device object graph is
48 ordered slots, 36 GDN, 12 QSA, PLE only at layer 1, one shared BF4 streamer,
vocabulary-row-sharded model I/O, and the backbone terminal mixer. Fourteen
TTNN static tests plus 10 parameterized checkpoint-domain subtests pass. BF4
artifact SHA-256 verification is memoized only after a complete hash; every
later load checks regular-file inode, size, mtime and ctime, and any mutation
forces a fresh full hash. A post-integration audit found transaction exception
safety gaps before MTP use; an isolated owner is implementing all-48 preflight
validation and fault-injection coverage now.

The original partition-A gate remains failed: the Blackhole fabric handshake
reproducibly stopped on device 2 Ethernet channel 4. It has not been retried or
reset. The continuation launcher records PID 4106323 and inherited node locks,
but the command-executor namespace does not expose that PID or FDs 200--203.
Accordingly, no local hardware action is admitted until a wrapper revalidates
the host-visible ancestor locks and obtains a fresh enforcing exact-device
lease. A read-only external audit found `f07cs04` occupied by recurring broker
jobs and only 74 GB free; no task job, worktree, build, or device action was
started there. It must be freshly requalified after that workload stops. At
`2026-08-26T21:30Z`, sandboxed and automatically approved external read-only
SSH checks both failed before connection because `f07cs04-ext` could not be
resolved; `f07cs05-ext` had already failed the same way. A second bounded,
automatically reviewed check at `2026-08-26T21:49Z` failed identically for both
hosts. No remote broker or device state was touched.

## Last passing gate

- Exact BF16 snapshot: 144 files, 131 weight shards, 1,658 tensors,
  360,000,192,888 shard-file bytes and 359,999,963,128 tensor-data bytes.
- All 131 weight files match the pinned ModelScope release SHA-256 manifest.
- Current-tree Release build: all 1,938 build/install targets completed.
- Dedicated runtime imports the current `_ttnn.so` and has a pinned freeze hash.
- Incremental TTNN build passed after adding the local-combine topology override
  and per-coordinate live DRAM-ring query. The current `_ttnn.so` SHA-256 is
  `1911ac8b460f0d5d4b943319cf121c22ab5484cd96edfe3728c4dae94ff9cbc6`;
  no-device import resolves the new runtime at
  `/home/sjett/qwen38-flash-next-data/runtime-local-combine-ring-qualified-20260826`.
- The exact checkpoint budget now uses `moe_compute`'s packed, ring-aware BF4_B
  layout. Keeping all 49 routed layers resident would consume
  23,366,762,496 bytes/device on ring-7 or 26,704,871,424 on ring-8 before
  non-expert weights. The admitted serialized stream holds one layer:
  3,021,063,680 total static bytes/device on ring-7 or 3,089,188,352 on ring-8.
- Fourteen TTNN component/BF4/builder/budget static tests and ten parameterized subtests
  pass. Cache
  publication is process-locked, stages tensorbins under unique names, rejects
  unmanifested artifacts, hashes every published file, and binds the exact
  physical-ID order plus identical per-card DRAM-bank worker order.
- TP4 gated residual, GDN, PLE, QSA/KV, terminal mixer,
  vocabulary-sharded embedding/LM head, and the decoder-layer shell pass
  `py_compile`, the pinned-extension no-device import/API gate, and their
  static shape/config invariants together.
  GDN keeps genuinely head-sharded FP32 recurrent state and preallocated
  snapshot/restore buffers; GR retains an explicit `[1,1,4,640]` branch-major
  state and marks row-parallel values as local partials before collectives.
- Pinned-Transformers CPU suite: 59 tests and 10 parameterized subtests passed
  with no skips. This includes exact-checkpoint GDN, QSA, PLE, gated-residual,
  MoE, and composed-layer comparisons, TP4/EP4 shard reconstruction, and
  prefill/tokenwise GDN/PLE state equality.
- A bounded full-48-layer CPU run completed two ordinary greedy decode steps
  through exact checkpoint weights: token `17 -> 15 -> 16`, finite
  248,320-way BF16 logits, positions `1 -> 2`, and 14.80 GiB peak RSS.
- The matching two-token prefill preserved the same greedy token, all 48 router
  top-1 choices, at least 9/10 selected experts per layer, exact PLE token
  history, final-logit PCC `0.9998929`, and logit p99 absolute error
  `0.065918` versus tokenwise decode. Full state minimum PCC was `0.9996639`.
- PLE lookup reads only selected rows from the 128 host-resident table shards;
  it does not materialize the 51.2B-parameter table. Each selected vector is
  split into four non-replicated 640-wide shards.
- GDN preserves FP32 recurrent state and enforces the checkpoint's
  `output_gate_type="sigmoid"`; the older Qwen3.6 TT path's hard-coded SiLU
  output gate is rejected as semantically incompatible.
- QSA placement is explicit: six query heads per device, each of the two KV
  heads replicated only inside its assigned two-device group, and no accidental
  four-way KV replication. Its short-context dense value path is an oracle only,
  not final sparse-QSA support.
- Routed MoE ownership is exactly 128 of 512 experts/device. Packed BF4_B costs
  476,872,704 bytes/device/layer on ring-7 or 544,997,376 on ring-8; it is
  streamed one layer at a time. Router output is 128-way sharded then gathered,
  and exact normalized top-10 scores are retained for weighted reduction.
- The all-to-all dispatch remains rejected for true global-B1. The new local
  path replicates the one logical activation only, computes locally owned
  experts without fabric dispatch, zero-initializes every unwritten expert
  slot, applies owner-masked real scores, adds the separately TP4-sharded and
  dynamically sigmoid-gated shared expert, then reduce-scatters over the EP
  axis. It never replicates the 512 expert weights.
- Every pre/post hardware audit passed ownership and cleanup checks. Partition
  A remains protected by the launcher's four inherited locks; B is untouched.

The four-card gate itself did **not** pass. Visibility and auto-discovery did:
exactly four devices were visible, UMD mapped PCIe IDs `[2,3,1,0]`, and both
logical and physical adjacency were a four-chip line with degree histogram
`{1:2, 2:2}`. Fabric initialization then timed out before the mesh opened, so
no collective or `tensor_topology()` result is accepted.

## Active processes and ownership

- Current launcher: `resume-ultra-fast-after-reset.sh` PID `65610`; Codex PID
  `65694`, exact `gpt-5.6-sol` / `ultra` / Fast invocation.
- Lease ancestor PID `65694` exposes FDs 200--203 for exactly
  `/run/lock/tt-device-node-{0,1,2,3}.lock`; the kernel flock owners are the
  launcher's retained FDs.  The one-shot job also held
  `/run/lock/qwen38-flash-next-partition-a-job.lock` for its full process-tree
  lifetime.
- Telemetry service PID `62612` is active/running and identified as the expected
  system observer.  The two preflight samples and post-run sample found no
  unexplained or leaked `/dev/tenstorrent/0..7` handle.
- Partition A mapping remains nodes `0,1,2,3` -> BDFs
  `61:00.0,41:00.0,01:00.0,21:00.0`, all NUMA 0.  Partition B remains
  unleased and unopened.
- No model, conversion, build, profiler, or hardware process is active.  The
  preserved agent worktrees are idle and unchanged from their maintenance
  snapshot.

The following bullets are the preserved pre-maintenance process record:

- This continuation is exact `gpt-5.6-sol`, `ultra`, Fast service. Its preserved
  launcher log records PID `4106323`; that process is outside the command
  executor's observable PID namespace.
- Visibility: `TT_VISIBLE_DEVICES=0,1,2,3`
- A nodes: `0,1,2,3`; the launcher records locks on FDs 200--203. The command
  executor does not inherit them, so future hardware wrappers must verify the
  host-visible ancestor and an enforcing broker lease before use.
- B nodes: `4,5,6,7`; unleased, idle, and unused
- One software-only builder agent is active in a disjoint worktree with
  separate build/cache/conversion/evidence paths. The builder is now integrated;
  isolated agents are hardening model transactions and preparing a fail-closed
  device-gate harness. The GR, GDN, embedding, QSA, layer, builder, and model
  worktrees remain separate. No hardware process is active.
- Last post-flight samples: `2026-08-26T19:05:58Z` and
  `2026-08-26T19:07:26Z`; no device-node handles remained after teardown

The command executor closes high-numbered descriptors in grandchildren. Each
hardware wrapper therefore verifies that a live ancestor still holds the exact
four launcher lock descriptions and takes the separate nonblocking job lock
`/run/lock/qwen38-flash-next-partition-a-job.lock`. It never reacquires the
launcher locks or treats visibility as ownership.

## Selected skills

- `test-driven-development`: exact CPU contracts and red/green tests precede
  each GR, PLE, GDN, QSA, MoE, MTP, and integration implementation.
- `bash-safety`: hardware/lifecycle scripts use strict mode, explicit paths,
  bounded timeouts, fail-closed ownership checks, and task-scoped cleanup.
- `perf-report`: reserved for kernel profiling after topology and correctness;
  failed fabric attempts produce no performance metrics.
- `openai-docs`: established and recorded the required `gpt-5.6-sol` / `max`
  Codex invocation.

No repository skill exists for model porting, TTNN operations, device ownership,
or documentation. Glean/MCP search is unavailable in the active tool catalog;
this is recorded once and repository primary source is used instead.

## Blocker

Current classification: `runtime_unverified`.

The authorized post-reset strict gate opened UMD discovery once, selected the
exact qualified `_ttnn.so`, and then failed before fabric initialization on a
missing installed-package runtime resource.  This is not evidence that the
historical Ethernet path still fails after reset.  The failure is fully
classified in `BLOCKED.md`; its transcript SHA-256 is
`901bd88acc9252db6c192ea3ef67688d808140eae83bdbaaff12c3eeafe08650`.
The user's one-opening allowance is consumed, so no further hardware command
is authorized in this continuation.

Historical pre-maintenance classification: `hardware_failed`.

Three bounded A-only jobs failed during `open_mesh_device` before any model or
collective workload. Strict initialization failed at `18:57Z`; a corrected
physical-`4x1` open followed by planned logical reshape failed identically at
`19:00Z`; documented `RELAXED_INIT` failed identically at `19:05Z`. In every
case device 2 channel 4 remained `STARTED`, channels 5--7 reached
`REMOTE_HANDSHAKE_COMPLETE`, and the runtime timed out waiting for
`LOCAL_HANDSHAKE_COMPLETE`. See `BLOCKED.md` and the immutable evidence record.

The single explicitly authorized reset between the historical and current
results completed successfully and must never be repeated.  No service change,
process kill, partition substitution, B-device access, or reset-like recovery
is authorized. A source-only model, single-device component run, or
host-bounced pseudo-collective is not the requested completion.

## Exact next command

No hardware command is currently authorized.  The smallest next command, only
after the user grants a new opening attempt, is the same direct strict topology
gate with the qualified site-packages first and this additional pre-import
binding:

```text
TT_METAL_RUNTIME_ROOT=/home/sjett/tt-metal-qwen38-flash-next-agent-20260826
```

Before that command, repeat both ownership samples and the exact path/hash
proof.  The strengthened gate now rejects a missing/wrong runtime root or
descriptor before discovery.  Do not use relaxed mode and do not perform any
recovery command.
