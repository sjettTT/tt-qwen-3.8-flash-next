# GDN, QSA, and PLE component gate

UTC cutoff: `2026-08-26T19:23:33Z`

## Scope

This is a no-device correctness gate over exact checkpoint tensors. It does not
claim a four-device run and does not weaken the partition-A hardware blocker.

- GDN: exact checkpoint load and TP4 Q/K/V packing; FP32 recurrent state;
  prefill/decode transition; comparison to pinned Transformers. The target
  config requires a sigmoid output gate, so the Qwen3.6 TT implementation's
  hard-coded SiLU output gate is not reused.
- QSA: checkpoint Q/g interleave, six query heads per device, explicit
  pair-local KV replicas, one index query head per device, explicit index-key
  replica, and a pinned-reference short-context oracle. The dense value core is
  deliberately restricted to oracle work and is not final sparse QSA.
- PLE: all 128 table-part shapes validated, exact n-gram hashing/EOS behavior,
  direct sparse row reads from safetensors with no full table materialization,
  four 640-wide result shards, and exact convolution/history state.

## Passing command

```text
QWEN38_TRANSFORMERS_SRC=/home/sjett/qwen38-flash-next-data/sources/transformers/transformers-dabae5fcb924a8eece0e727b627ca5f050b40d40/src \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/sjett/qwen38-flash-next-data/oracle-venv/bin/python -m pytest -q \
  --confcutdir=models/demos/blackhole/qwen38_flash_next/tests \
  models/demos/blackhole/qwen38_flash_next/tests
```

Result: `46 passed, 10 subtests passed, 0 skipped` in 4.22 seconds. The three
warnings are one pre-existing unknown pytest `timeout` option and two SWIG type
deprecations. Bash syntax checks, Python bytecode compilation, and
`git diff --check` also passed.

No device was opened for this gate. Partition B remained untouched.
