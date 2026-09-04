# Full-48-layer CPU ordinary-decode gate

UTC: `2026-08-26T19:55:24Z`

## Boundary

This is the exact BF16 checkpoint's lazy CPU integration oracle. It executes
all 48 language layers, the host-resident PLE lookup, all selected routed and
shared experts, final four-branch mixer, and untied LM head. It opened no TT
device and is not presented as the required four-P150 run.

## Passing command

```text
PYTHONPATH=/home/sjett/tt-metal-qwen38-flash-next-agent-20260826 \
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 \
/home/sjett/qwen38-flash-next-data/oracle-venv/bin/python -u \
  models/demos/blackhole/qwen38_flash_next/tools/run_full_cpu_oracle.py \
  --checkpoint /home/sjett/qwen38-flash-next-data/checkpoints/Qwen3.8-Flash-Next-f5d08274 \
  --token-id 17 --decode-steps 2 --verify-prefill-equivalence
```

The outer command used `timeout --signal=TERM --kill-after=30s 1800s` and
`pipefail`; it exited zero.

## Immutable runtime log

- Path:
  `/home/sjett/qwen38-flash-next-data/logs/qwen38_flash_next_cpu_oracle/20260826T1955Z-full48-prefill-equivalence.jsonl`
- SHA-256:
  `138f648bd52f226ac4fece52337be7917c228ef9735fd6cf47b0c9fd20239797`
- Size: 40,299 bytes; 149 JSONL records
- Peak RSS: 15,518,752 KiB (14.80 GiB)

## Results

- Step 0: input token 17, output token 15, state position 1, 48/48 layers.
- Step 1: input token 15, output token 16, state position 2, 48/48 layers.
- Both full-vocabulary logit tensors were finite and hashed in the log.
- Two-token prefill versus the second tokenwise position:
  - same greedy token;
  - final hidden PCC `0.9998914003`, mean/p99/max absolute error
    `0.033665 / 0.125 / 0.375`;
  - final logits PCC `0.9998928905`, mean/p99/max absolute error
    `0.020249 / 0.065918 / 0.140625`;
  - all 48 router top-1 choices match; mean overlap `9.7083/10`, minimum
    overlap `9/10`; exact ten-rank ordering on 14 layers;
  - full-state minimum PCC `0.9996638894`, maximum state p99 absolute error
    `0.0859375`, and exact PLE token history.

The passing distribution thresholds are recorded in `DECISIONS.md` D016. The
preceding strict run requiring exact ordering of all ten router choices is
retained, not hidden:

- Path:
  `/home/sjett/qwen38-flash-next-data/logs/qwen38_flash_next_cpu_oracle/20260826T1953Z-full48-prefill-equivalence-strict-failed.jsonl`
- SHA-256:
  `8d21c8d7db56e0aa69d83609c387a15a6fa3d4a3460df5c3766ad7f1580e226d`

## Test suite

The post-run pinned-Transformers/component suite passed `59` tests plus `10`
parameterized subtests, with no skips. QSA cache tests separately match pinned
Transformers tokenwise and prove immutable prefix rollback.

## Hardware state

Partition A was not opened. Its inherited lease remains held by the launcher;
partition B remains unleased and untouched. The previously classified fabric
failure is unchanged.
