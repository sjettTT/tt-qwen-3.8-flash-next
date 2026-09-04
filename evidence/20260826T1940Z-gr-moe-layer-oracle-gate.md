# GR, MoE, and composed-layer oracle gate

UTC: `2026-08-26T19:40:59Z`

## Scope

- Exact BF16 checkpoint tensors from the pinned `f5d08274` snapshot.
- Both attention and MLP gated-residual modules, including explicit TP4
  decomposition of four residual branches and the rank-320 bottleneck.
- Complete target-shape MoE arithmetic at `E=512`, top-10, `H=2560`, `I=640`,
  including shared expert, scalar gate, real normalized routing scores, and
  exactly 128 routed experts/device.
- Exact decoder-layer order over a GDN layer and a PLE-to-QSA short alternating
  stack. This remains a CPU integration oracle, not a TT device result.

## Command and result

```text
QWEN38_TRANSFORMERS_SRC=/home/sjett/qwen38-flash-next-data/sources/transformers/transformers-dabae5fcb924a8eece0e727b627ca5f050b40d40/src \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/sjett/qwen38-flash-next-data/oracle-venv/bin/python -m pytest -q \
  --confcutdir=/home/sjett/tt-metal-qwen38-flash-next-agent-20260826/models/demos/blackhole/qwen38_flash_next/tests \
  /home/sjett/tt-metal-qwen38-flash-next-agent-20260826/models/demos/blackhole/qwen38_flash_next/tests

54 passed, 3 warnings, 10 subtests passed in 4.45s
```

No test was skipped. The warnings are the repository's missing optional pytest
timeout plugin plus SWIG deprecation warnings; no numerical gate was relaxed.

## Proven contracts

- GR CPU and simulated TP4 paths agree with pinned Transformers on real layer-0
  weights. TP4 stores norm `(4,640)`, down `(320,4,640)`, up `(4,640,320)`, and
  injection `(4,4,640)` per device and reconstructs the exact checkpoint.
- MoE CPU and simulated EP4 paths agree with pinned Transformers on exact
  router, ten selected experts, shared expert, and scalar gate. Routed BF4 tile
  payload is exactly `353,894,400` bytes/device/layer.
- Real normalized scores are required by the weighted combine; missing or dummy
  scores fail closed.
- Current source `ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/device/all_to_all_dispatch_program_factory.cpp`
  computes batch as `input_shape[0] * dispatch_devices`. The model contract
  rejects true global-B=1 on that TP4 dispatch route.
- Current `ttnn.sparse_matmul` source exposes a sparsity tensor, but no expert
  index/compact-gather argument. The separately pinned Qwen3.6 bundle API is not
  assumed to exist in this revision.
- Exact layer-0 GDN prefill agrees with two tokenwise recurrent transitions.
  Exact checkpoint layer 1 applies PLE before attention, and layer 3 exercises
  QSA in the short-context dense oracle boundary.

## Hardware boundary

No TT device was opened. Partition A remains blocked by the previously recorded
fabric handshake failure; B remains unleased and untouched. These results are
software-oracle and placement-contract evidence only.
