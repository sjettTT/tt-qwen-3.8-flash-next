# Qwen3.8-Flash-Next bring-up blockers

## Current post-maintenance blocker

Classification: `runtime_unverified`

UTC established: `2026-08-26T22:28:24Z`

The one post-reset hardware-opening gate authorized by the user has been
consumed.  It failed before fabric configuration because the qualified
installed TTNN package selected itself as the TT-Metal runtime root and did not
contain the required Blackhole SoC descriptor.  Per the explicit one-shot
constraint, there will be no retry, relaxed-mode attempt, reset, power cycle,
KMD reload, or component device test in this continuation.

### Exact failing command

The complete preflight and command record is
`/home/sjett/qwen38-flash-next-data/evidence/20260826T222657Z-strict-topology-gate-a/precheck.txt`.
The hardware-opening command was the following direct Python invocation under
the live ancestor leases and the scoped job lock
`/run/lock/qwen38-flash-next-partition-a-job.lock`:

```text
PYTHONPATH=/home/sjett/qwen38-flash-next-data/runtime-local-combine-ring-qualified-20260826/site-packages:/home/sjett/tt-metal-qwen38-flash-next-agent-20260826 \
TT_METAL_HOME=/home/sjett/tt-metal-qwen38-flash-next-agent-20260826 \
TT_VISIBLE_DEVICES=0,1,2,3 \
QWEN38_FABRIC_RELIABILITY=strict \
timeout --signal=TERM --kill-after=15s 300s \
  /home/sjett/qwen38-flash-next-data/runtime-venv/bin/python \
  /home/sjett/tt-metal-qwen38-flash-next-agent-20260826/models/demos/blackhole/qwen38_flash_next/tools/topology_gate.py \
  --checkpoint /home/sjett/qwen38-flash-next-data/checkpoints/Qwen3.8-Flash-Next-f5d08274 \
  --output /home/sjett/qwen38-flash-next-data/evidence/20260826T222657Z-strict-topology-gate-a/topology-gate.json
```

`TT_METAL_RUNTIME_ROOT` was absent.  That omission is the exact failure cause:
installed `ttnn/__init__.py` calls `SetRootDir()` on its package directory when
the variable is absent; `TT_METAL_HOME` does not override this path in the
current runtime.

### Failure transcript

```text
{"event":"runtime_import_proof",
 "ttnn_module":".../runtime-local-combine-ring-qualified-20260826/site-packages/ttnn/__init__.py",
 "ttnn_extension":".../runtime-local-combine-ring-qualified-20260826/site-packages/ttnn/_ttnn.so",
 "ttnn_extension_sha256":"1911ac8b460f0d5d4b943319cf121c22ab5484cd96edfe3728c4dae94ff9cbc6",
 "coordinate_binding_ttnn_device":true,
 "coordinate_binding_extension_device":true}
Opening user mode device driver
Creating TopologyDiscovery for architecture: blackhole
Completed topology discovery
Opening local chip ids/PCIe ids: {0, 1, 2, 3}/[2, 3, 1, 0]
Cluster constructor completed.
Cluster destructor completed.
RuntimeError: bad file: .../site-packages/ttnn/tt_metal/soc_descriptors/blackhole_140_arch.yaml
PYTHON_EXIT=1
POST_HANDLES=none
RESULT_EXISTS=no
```

Full transcript SHA-256:
`901bd88acc9252db6c192ea3ef67688d808140eae83bdbaaff12c3eeafe08650`.

### Source location

- Installed-package root selection:
  `ttnn/ttnn/__init__.py:633-643`
- Environment/runtime-root selection:
  `tt_metal/llrt/rtoptions.cpp:320-351`
- Blackhole descriptor lookup:
  `tt_metal/llrt/tt_cluster.cpp:58-66`
- One-shot gate:
  `models/demos/blackhole/qwen38_flash_next/tools/topology_gate.py`

### What passed

- Source HEAD and dirty/untracked state were recorded without changing or
  reformatting protected files.  The maintenance snapshot and reset-evidence
  `SHA256SUMS` digests match the user's pins.
- The import-only proof ran from `/tmp` with the qualified site-packages first.
  `ttnn.__file__`, `_ttnn.so`, SHA-256 `1911...cbc6`, build ID
  `91ee...1fb`, and the required coordinate binding all matched.
- Two host-visible ownership samples found lease ancestor PID `65694` with
  FDs 200--203 targeting exactly nodes 0--3, no unexplained device handle, an
  uncontended scoped job lock, and active telemetry PID `62612`.
- The gate's UMD discovery saw physical IDs 0--3 with PCIe IDs `[2,3,1,0]`,
  firmware `19.8.1`, then destroyed the cluster cleanly.
- The post-run audit found no leaked handle.  All four A cards remained P150b,
  DRAM healthy, PCIe Gen5 x16, zero GDDR uncorrectables, firmware `19.8.1.0`,
  and Ethernet firmware `1.10.1` with the unchanged node/BDF/board mapping.
- A subsequent **import-only** negative/positive proof established the missing
  pre-import contract: exporting
  `TT_METAL_RUNTIME_ROOT=/home/sjett/tt-metal-qwen38-flash-next-agent-20260826`
  resolves the pinned SoC and core descriptor files and passes the strengthened
  no-device guard.  It did not reopen hardware.

### What did not pass

- The post-reset process did not complete device visibility counts or create a
  `SystemMeshDescriptor` because `GetNumAvailableDevices()` failed on the
  missing descriptor path.
- Fabric was not configured or initialized.  Therefore the reset's effect on
  the historical Ethernet handshake failure remains unknown; it is not a new
  hardware failure result.
- No mesh open, physical fabric-node mapping, allocator measurement,
  all-gather, tensor-topology result, TTNN component test, ordinary decode,
  MTP verification, chat demo, or clean demo restart passed.

### Device health and ownership

The launcher still owns the four node locks through the live ancestor.  The
one-shot process exited, its UMD cluster destructor completed, and `fuser` and
`lsof` found no task-owned device-node handle afterward.  Root telemetry is the
only expected system observer.  Nodes 4--7 were not opened or leased.  No
reset-like or recovery command was issued in this continuation.

### Smallest concrete next action

The software guard has already been corrected to require, hash, and emit
`TT_METAL_RUNTIME_ROOT` plus the Blackhole SoC/core descriptors before any
discovery call.  Resuming hardware now requires new external authority because
the user authorized exactly one opening attempt.  With that authority, the
smallest action is a fresh two-sample ownership audit followed by one strict
gate whose process exports, **before importing TTNN**:

```text
TT_METAL_RUNTIME_ROOT=/home/sjett/tt-metal-qwen38-flash-next-agent-20260826
```

No rebuild is indicated: the qualified extension and its coordinate binding
passed, and the missing repository resource path is independently proven.

## Preserved pre-maintenance result

Classification: `hardware_failed`

UTC established: `2026-08-26T19:05:37Z`

## Exact failing command

```text
QWEN38_FABRIC_RELIABILITY=relaxed \
models/demos/blackhole/qwen38_flash_next/scripts/run_topology_gate.sh \
  /home/sjett/qwen38-flash-next-data/evidence/topology-gate-partition-a-relaxed.json
```

The wrapper verified `TT_VISIBLE_DEVICES=0,1,2,3`, a live ancestor holding all
four exact launcher lock descriptions, an uncontended task job lock, and no
pre-existing device-node handle. It bounded the child to 300 seconds and never
requested a reset.

## Failure transcript

```text
TOPOLOGY_JOB_START_UTC=2026-08-26T19:05:12Z PID=3978449 LEASE_PID=3875356 PARTITION=0,1,2,3
{"available_devices": 4, "devices": 4, "event": "visibility_verified", "pcie_devices": 4, "system_mesh": {"all_local": true, "local_shape": [4, 1], "shape": [4, 1]}, "utc": "2026-08-26T19:05:15Z"}
{"event": "fabric_configured", "fabric_config": "FABRIC_1D", "reliability": "RELAXED_INIT", "utc": "2026-08-26T19:05:15Z"}
Fabric initialized on 4 devices
Fabric Router Sync: Timeout after 10000 ms on Device 2. Expected status 0xa2b2c2d2 (LOCAL_HANDSHAKE_COMPLETE)
master chan=5 logical=e0-5 status=0xa1b1c1d1 (REMOTE_HANDSHAKE_COMPLETE)
sub chan=4 logical=e0-4 status=0xa0b0c0d0 (STARTED) <-- least progress
sub chan=6 logical=e0-6 status=0xa1b1c1d1 (REMOTE_HANDSHAKE_COMPLETE)
sub chan=7 logical=e0-7 status=0xa1b1c1d1 (REMOTE_HANDSHAKE_COMPLETE)
Hint: Ethernet handshake likely failed -- the link may not be healthy.
RuntimeError: TT_THROW @ tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:271
{"event": "fabric_disabled", "utc": "2026-08-26T19:05:36Z"}
Closing devices in cluster completed.
```

Job record:
`/home/sjett/qwen38-flash-next-data/jobs/topology-gate-partition-a-3978449.record`.
The two prior strict-mode transcripts are preserved at:

- `/home/sjett/qwen38-flash-next-data/logs/topology-gate-partition-a.log`
  (`c4d950a031bfbbba0d554e685f032d68b3af0b799f240ada69b92bee02245c85`)
- `/home/sjett/qwen38-flash-next-data/logs/topology-gate-partition-a-v2.log`
  (`d0974780e468dc39d3bf6a147722ef6c3cf997abb008f1276525bcff083eac7a`)

The second strict attempt opened the auto-discovered physical `4x1` line and
would reshape only after initialization, ruling out the first attempt's logical
orientation as the cause. `RELAXED_INIT`, which primary TT-Metal documentation
defines as tolerating down links by selecting fewer routing planes, still
selected the failing channel and produced the same handshake state.

## Source location

- Throw: `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:271`
- Handshake wait/report: the same file, `wait_for_fabric_router_sync`
- Reliability contract: `ttnn/cpp/ttnn-nanobind/fabric.cpp`
- Troubleshooting direction:
  `tech_reports/Programming_Multiple_Meshes/Programming_Multiple_Meshes.md`
- Gate: `models/demos/blackhole/qwen38_flash_next/tools/topology_gate.py`
- Ownership wrapper:
  `models/demos/blackhole/qwen38_flash_next/scripts/run_topology_gate.sh`

## What passed

- Exact four-device visibility and UMD discovery.
- Partition-A node/BDF/board-ID/NUMA mapping from independent sysfs and TT-SMI
  checks; logical IDs map to nodes `2,3,1,0`.
- Auto-discovered logical and physical four-chip line topology, degree
  histogram `{1:2, 2:2}`.
- Two clean ownership samples before each hardware phase and clean post-flight
  samples after teardown.
- DRAM trained, PCIe Gen5 x16, zero uncorrectable GDDR errors, firmware
  `19.8.1.0`, and normal temperatures after each attempt.
- Exact BF16 download/hash gate, current-tree Release build, dedicated runtime,
  static four-card memory admission, and all CPU oracle/loader tests.

## What did not pass

- Fabric router initialization on partition A in either strict or relaxed mode.
- Four-device mesh open, physical-ID-to-fabric-node capture, device allocator
  measurement, shard/replica `tensor_topology()` verification, or all-gather.
- Consequently no TT component, full 48-layer model, ordinary decode, MTP
  verification, demo, or clean demo restart can be claimed.

## Device health and ownership

The launcher retains exclusive advisory locks for `/dev/tenstorrent/0..3` on
FDs 200--203. No task-owned hardware process or device-node handle remained
after the failure. The only expected system process reported by TT-SMI is root
`tt_telemetry_server` PID `843856`; it has no `fuser`/`lsof` device-node handle.
Partition B nodes 4--7 were never leased or opened. No reset, service stop, or
process kill was performed.

## Smallest concrete next action

An authorized operator must repair/retrain partition A's device-2/channel-4
Ethernet path (or perform the site's approved diagnostic/reset procedure), then
the unattended run can repeat the existing two-sample ownership audit and
bounded strict topology gate. The alternative is explicit allocation of a
different four-P150 partition after the prompt's cross-process fabric/reset
isolation proof and four disjoint locks. Repeating the same initialization on
the current state is not a meaningful or safe next action.
