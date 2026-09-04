# Phase-0 oracle, compatibility, and long-pole evidence

UTC cutoff: `2026-08-26T18:31:00Z`

## Red/green component oracle

The first collection run failed with two `ModuleNotFoundError` errors because
`models.demos.blackhole.qwen38_flash_next.reference` did not exist. The
implementation was then added in small semantic units. A later red test exposed
incorrect prime advancement for a nonzero PLE layer index; the hash-spec builder
was corrected before the full run.

Green command:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
QWEN38_TRANSFORMERS_SRC=/home/sjett/qwen38-flash-next-data/sources/transformers/transformers-dabae5fcb924a8eece0e727b627ca5f050b40d40/src \
/home/sjett/qwen38-flash-next-data/oracle-venv/bin/python -m pytest -q \
--confcutdir=models/demos/blackhole/qwen38_flash_next/tests \
models/demos/blackhole/qwen38_flash_next/tests/test_reference_semantics.py \
models/demos/blackhole/qwen38_flash_next/tests/test_transformers_oracle.py
```

Result:

```text
20 passed, 3 warnings in 3.25s
```

Direct pinned-Transformers comparisons cover GR read/write, n-gram hash and EOS
boundaries, FP32 GDN recurrence, cached causal convolution, and QSA block/tail
selection. Independent invariants cover MTP four-branch fusion, all five
verifier commit depths, incremental-state equivalence, router normalization,
and frozen-index draft-tail construction. Exact SplitMix64 multipliers are
pinned to `[23703573157769, 20109073645365, 8052911324071]`.

## MTP source resolution

Transformers at `dabae5fcb924a8eece0e727b627ca5f050b40d40` ignores the
checkpoint's `mtp.*` namespace. Official SGLang Qwen4Exp integration commit
`73a255206f916366c8d26d4022f82ddfb0ab558d` supplies the missing serving
contract. Relevant SHA-256 values:

```text
f406977eb2373937393241f453477867f7dc943bd4839216db8fe66fa9f921d8  qwen4_exp.py
b52c063123611d521ff8c8c111b9fd935c06bdef638e829735b99da12266c77d  qwen4_exp_mtp.py
6cb122eb32ad5c07d42dfc2bfb2ae43cb3111befa32ae95d03b09fc732bc2de6  nsa_backend.py
ca41859fb992add3d24175f5d08f1ef6fc90dc63dbfc5fcc7665985286463943  test_qsa_mtp_shared_indexer.py
3f1b99093a8ba48d3472fc9d0aa918c3a1551a64e5717d14bb76b72d61b8e0a8  test_verify_commit_triton.py
```

The checkpoint manifest independently confirms `mtp.fc_embedding.weight`
`[2560,2560]`, `mtp.fc_hidden.weight` `[2560,2560]`, and
`mtp.pre_fc_norm_hidden.weight` `[10240]`.

## Qwen3.6 reuse audit

The exact implementation boundary is recorded in `COMPATIBILITY_MATRIX.md`.
The external pinned A3B bundle's demos and runner all open `MeshShape(1, 1)`;
its expert weights are `[1,E,in,out]` and its MTP input is a concat projection.
The in-tree Qwen3.6 code contains the reusable TP4, CCL, paged-cache, GDN-state,
and lifecycle implementation. Ordinary residuals, dense attention, replicated
expert storage, and generic MTP are rejected for Qwen4Exp execution.

## Parallel long poles

- Weight download started `2026-08-26T18:28:28Z`, exact revision
  `f5d08274bafd880402bd16f5e3e6c514136ec06c`, four workers, dedicated data
  root. At the cutoff it had materialized 58 shards and 161 GiB.
- Isolated release build first configured at `2026-08-26T18:28:39Z` and failed
  before compilation with `Missing submodules`. It restarted at
  `2026-08-26T18:29:16Z` and is initializing this worktree's exact recorded
  gitlinks before reconfiguration.
- Neither job opened a Tenstorrent device. Partition B remains untouched.
