# Qwen3.8-Flash-Next bring-up decisions

Decisions are append-only by numbered entry. A superseding decision cites the
earlier entry rather than rewriting it.

## D001 — Partition A only

- UTC: `2026-08-26T17:32:01Z`
- Decision: Run all work on nodes 0--3 under the inherited lifetime locks.
  Partition B remains idle.
- Reason: PCIe/NUMA separation alone does not prove fabric, reset, host, thermal,
  or cross-process isolation. The prompt explicitly withholds A+B concurrency
  until a stronger gate passes.

## D002 — Exact semantic source hierarchy

- UTC: `2026-08-26T17:32:01Z`
- Decision: The pinned Qwen4Exp checkpoint and official reference code define
  GR, PLE, QSA, GDN, MTP, cache, tokenizer, and tool-call semantics. Existing
  Qwen3.6/Qwen3.8-27B code is reusable only after a compatibility matrix proves
  each behavior.
- Reason: Flash-Next is neither the dense 27B model nor a shape-only A3B port.

## D003 — Text-only scope and license boundary

- UTC: `2026-08-26T17:32:01Z`
- Decision: Implement the complete language backbone and checkpoint MTP head;
  omit vision execution while retaining its manifest and license provenance.
  The service remains internal and loopback-only.
- Reason: The required outcome is a text-only demo, and the Qwen Community
  License imposes additional conditions outside this internal use case.

## D004 — Precision admission order

- UTC: `2026-08-26T17:32:01Z`
- Decision: Use BF4_B for routed experts as the feasibility baseline. Consider
  BF8_B first for dense/control/router/shared/embedding/head tensors, then expert
  down-projections, then only measured sensitive expert layers. Never use full
  BF8_B for the required four-card model.
- Reason: TT tile encodings and the checkpoint parameter split make full BF8_B
  exceed four cards before runtime state. Every promotion must pass an exact
  manifest, padded-byte, operator-golden, and runtime-headroom check.

## D005 — Correctness before optimization

- UTC: `2026-08-26T17:32:01Z`
- Decision: Synchronous host PLE lookup, eager execution, ordinary decode, and
  serial-equivalent MTP verification are correctness gates before prefetch,
  tracing, or speed claims.
- Reason: Async overlap and tracing change lifetime/address constraints; they
  cannot be used to mask semantic or placement failures.

## D006 — No implicit placement changes

- UTC: `2026-08-26T17:32:01Z`
- Decision: Every sharded tensor has a fail-closed placement contract checked
  through actual TTNN topology metadata. The two QSA KV heads require an
  explicit grouped/replicated design with byte accounting; accidental mesh
  replication is a test failure.
- Reason: Equal tensor values do not prove sharding, and two KV heads do not
  divide over a four-device tensor-parallel axis.

## D007 — Explicit two-group QSA KV placement

- UTC: `2026-08-26T17:58:16Z`
- Decision: Shard 24 QSA query heads as six/device. Assign each of the two KV
  heads to one verified two-device mesh group, replicating that one head only
  within its group. Shard the four index-query heads one/device and explicitly
  replicate the single index-key head.
- Reason: This preserves two distinct KV heads and gives every local query head
  its owning KV value without silently replicating both KV heads mesh-wide.
  Concrete groups remain unbound until physical mesh/fabric coordinates pass.

## D008 — BF4 expert baseline admitted; BF8 promotions deferred

- UTC: `2026-08-26T17:58:16Z`
- Decision: Admit all 49 routed expert layers in BF4_B for initial full-model
  work. Reject all-routed BF8_B on four cards. Defer BF8_B down-projections or
  selected expert layers until the BF4_B golden passes and measured allocator
  usage demonstrates safe headroom.
- Reason: Exact TT tile accounting gives 19,885,016,576 static bytes/device for
  the BF4 baseline and 5,316,522,480 bytes/device of raw headroom after the full
  provisional admission reserve. All-routed BF8_B requires 35,299,083,776
  static bytes/device, exceeding the 34,225,520,640-byte raw capacity. BF8_B
  downs leave only 178,500,080 raw bytes under the same reserve, which is not a
  safe runtime margin.

## D009 — First full-context qualification is batch-one 8K

- UTC: `2026-08-26T17:58:16Z`
- Decision: Qualify batch-one 4K and then 8K state before extending toward the
  native 262,144-token context. The full target and real MTP layer remain
  loaded; this is a context admission sequence, not a reduced model.
- Reason: Eight-thousand-token cache/state accounting is exact and small relative to
  weights, while QSA long-context semantics and scratch behavior still require
  device measurement before reserving native-context storage.

## D010 — SGLang pull request pins missing MTP serving semantics

- UTC: `2026-08-26T18:31:00Z`
- Decision: Use official SGLang Qwen4Exp pull-request head
  `73a255206f916366c8d26d4022f82ddfb0ab558d` for the MTP execution contract
  absent from pinned Transformers: shared per-branch hidden projection,
  broadcast embedding projection, frozen selected QSA indices during drafting,
  and accepted-step state commit during verification.
- Reason: Transformers defines the target architecture but neither loads nor
  executes the checkpoint's `mtp.*` tensors. The SGLang implementation and its
  focused tests match the official four-step report and exact checkpoint names.

## D011 — Reuse TP4 machinery, replace Qwen3.6 semantics

- UTC: `2026-08-26T18:31:00Z`
- Decision: Reuse the in-tree Qwen3.6 TP4 lifecycle, CCL, sharded linear,
  paged-cache, GDN state, sampling, and trace scaffolding only behind the
  boundaries in `COMPATIBILITY_MATRIX.md`. Implement GR, PLE, QSA, EP512 MoE,
  and Qwen4Exp MTP as new semantic paths. The separately pinned A3B bundle is a
  single-card mechanism reference, not TP4 placement evidence.
- Reason: The reusable infrastructure is substantial, but ordinary residual
  adds, dense attention, all-expert sparse placement, and concat-based MTP are
  provably different from the exact target.

## D012 — Start payload and build as independent long poles

- UTC: `2026-08-26T18:31:00Z`
- Decision: Download the exact pinned 131-shard BF16 payload with four workers
  while building the current worktree in an isolated data-root build directory.
  Neither job opens hardware. Device work remains gated on a successful build,
  fresh ownership audit, and A-only topology test.
- Reason: Static memory admission has passed, disk space is sufficient, and the
  two jobs do not share TT device or fabric resources. Serializing them would
  delay the first full ordinary-decode path without improving correctness.

## D013 — Preserve the ancestor lease across the command-executor FD boundary

- UTC: `2026-08-26T18:53:00Z`
- Decision: Hardware wrappers must verify that a live ancestor still holds the
  exact four launcher lock descriptions on FDs 200--203, check visibility and
  node handles, and take a separate nonblocking task job lock. They must not
  reacquire the launcher lock paths.
- Reason: The command executor closes high-numbered descriptors in
  grandchildren. Opening the same paths again creates unrelated open-file
  descriptions and can either deadlock or falsely imply inheritance. The
  ancestor verification preserves the lifetime lease's observable exclusivity
  without weakening the independent per-job guard.

## D014 — Partition A fabric is failed; no unattended reset or substitution

- UTC: `2026-08-26T19:05:37Z`
- Decision: Classify the current partition-A gate as `hardware_failed` after
  two strict attempts and one documented relaxed-initialization attempt failed
  on device 2 Ethernet channel 4 with the same handshake state. Do not repeat
  the same initialization, reset cards, stop services, kill processes, or use
  partition B. Continue safe no-device implementation while recording the
  smallest operator action in `BLOCKED.md`.
- Reason: The second strict attempt used the auto-discovered physical 4x1 line,
  ruling out logical mesh orientation. `RELAXED_INIT` is the primary-source
  fallback for down links and still failed. Every teardown and delayed audit
  was clean, so further identical retries have no new falsifiable variable and
  would violate the unattended hardware-safety boundary.

## D015 — Preserve true global-B=1 at the MoE boundary

- UTC: `2026-08-26T19:40:59Z`
- Decision: Admit the current `all_to_all_dispatch_metadata` -> `moe_compute`
  -> `deepseek_moe_fast_reduce_nc_fused` path only when global batch divides
  over the four dispatch devices. Reject it for the required true global-B=1
  decode workload. Do not pad B=1 to B=4 or report one row/device as global
  B=1. Keep 128 routed experts/device and preserve exact top-10 scores through
  weighted reduction. Treat the indexed sparse alternative as unavailable
  until an indexed, compact-output contract exists in and is proven against
  this tree; the current `sparse_matmul` API accepts a sparsity tensor, not the
  Qwen3.6 bundle's expert-index argument.
- Reason: The current all-to-all source computes batch as local input rows
  multiplied by dispatch-device count, while the required interactive decode
  has one global token. Both padding and all-expert replication change the
  workload or violate four-card memory scaling. The exact CPU EP4 decomposition
  proves the intended ownership and arithmetic but is not device evidence.

## D016 — Gate BF16 prefill/decode on distributions and greedy identity

- UTC: `2026-08-26T19:55:24Z`
- Decision: For the full CPU BF16 oracle, require final hidden PCC at least
  `0.9998` and p99 absolute error at most `0.15`, final-logit PCC at least
  `0.9998` and p99 absolute error at most `0.1`, identical greedy token, at
  least 8/10 routed-expert overlap on every layer, full-state minimum PCC at
  least `0.999`, and exact PLE token history. Record exact top-k ordering as a
  diagnostic rather than require it to be bit-identical across a two-row
  prefill GEMM and two one-row decode GEMMs.
- Reason: A deliberately stricter first run showed the expected BF16 batch-shape
  rounding boundary: final logits remained PCC `0.9998929`, their p99 error was
  `0.065918`, the greedy token and all 48 router top-1 choices agreed, and every
  layer retained at least 9/10 experts, while only 14 layers retained the exact
  ordering of all ten near-tied routes. Component gates still compare each
  equation directly to pinned Transformers. Requiring bitwise router rank
  identity would test CPU GEMM batching, not model semantics or token output.

## D017 — Supersede raw-BF4 admission with ring-packed expert streaming

- UTC: `2026-08-26T21:00:13Z`
- Decision: Supersede D008's all-resident routed-expert admission. Use the exact
  `moe_compute` packer layout and keep only one BF4_B routed layer resident at a
  time for the correctness bring-up. Bind every cache artifact to checkpoint
  hashes, tt-metal revision, exact `(1,4)` physical-ID order, and an identical
  live DRAM-bank worker ordering on all four Blackhole devices. Serialize cache
  publication under a bounded lock, stage under unique names, hash before
  manifesting, and reject orphan/unmanifested final tensorbins.
- Reason: Raw BF4 tile payload is not the kernel's allocation. The exact packed
  layout costs 476,872,704 bytes/device/layer on a seven-bank ring and
  544,997,376 on an eight-bank ring. Across the 48 backbone layers plus MTP this
  is 23,366,762,496 or 26,704,871,424 bytes/device. With non-expert weights and
  the existing 9,023,981,584-byte runtime reserve, both all-resident plans fail
  the conservative device gate. One streamed layer leaves total static weights
  at 3,021,063,680 or 3,089,188,352 bytes/device and preserves 128 experts/card.

## D018 — True-global-B1 uses local expert partials, not padded A2A

- UTC: `2026-08-26T21:00:13Z`
- Decision: Extend `moe_compute` with an explicit `local_combine=True` mode on a
  degenerate mesh axis. On the required `(1,4)` mesh, replicate one logical
  input token and routing metadata, execute the 128 locally owned experts on
  each coordinate, produce full-width additive partials, apply owner-masked
  normalized top-10 scores with `deepseek_moe_fast_reduce_nc_fused`, add the
  separately TP4-sharded dynamic shared-expert partial, and reduce-scatter over
  mesh axis 1. Require the local output topology to remain fully replicated in
  shape metadata while tracking its stronger local-partial semantic explicitly.
- Reason: The existing A2A path defines global tokens as local tokens times
  dispatch devices and therefore turns B1 into B4. Compute-only mode exposes a
  two-section streaming buffer and cannot recover all selected experts. The
  existing FullLocal selective writer already has the correct local slot-stack
  contract; selecting the size-one axis removes fabric dispatch without
  changing the workload. Unwritten slots are cleared on every invocation so a
  masked `0 * NaN` cannot contaminate the weighted sum. This route remains a
  source/build result until the Blackhole B1 numerical gate passes.

## D019 — Consume the post-reset one-shot as runtime-unverified

- UTC: `2026-08-26T22:30:08Z`
- Decision: Classify the single authorized post-reset opening as
  `runtime_unverified`, stop all further hardware work, and require a new user
  authorization before any next opening.  Preserve the historical
  `hardware_failed` evidence separately; do not infer that the reset either
  repaired or failed to repair the Ethernet channel because the current run
  never configured fabric.
- Reason: The exact qualified extension and coordinate binding passed before
  discovery, and UMD opened IDs 0--3, but the installed Python package selected
  its own incomplete directory as the TT-Metal runtime root.  The first device
  discovery call failed on its missing Blackhole SoC descriptor and closed the
  cluster cleanly.  Source inspection and an import-only experiment show that
  `TT_METAL_RUNTIME_ROOT`—not `TT_METAL_HOME`—must be exported before import.
  The gate now hashes that root's SoC/core descriptors before any future
  discovery, but the explicit one-shot instruction prohibits testing the fix
  on hardware in this continuation.
