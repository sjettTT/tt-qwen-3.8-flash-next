# Checkpoint, runtime, and partition-A fabric evidence

UTC cutoff: `2026-08-26T19:07:46Z`

## Completed long poles

- Exact BF16 snapshot completed at `2026-08-26T18:34:29Z` with status 0:
  144 files, 131 safetensors shards, 360,000,192,888 aggregate shard-file
  bytes. Download log SHA-256:
  `cfac2ecbd0bf4a598788b45af861800e1abcb073d6ef093a76a9aae9d97c7592`.
- Full local SHA-256 verification matched every weight shard to the exact
  ModelScope release manifest. Generated 144-file manifest:
  `/home/sjett/qwen38-flash-next-data/manifests/checkpoint-f5d08274-sha256.json`,
  SHA-256
  `13c88f393ffbe4f5e9733d8e48a59f21c77e69a5f02b76bb073835d9a1ca0ea9`.
- Current-tree Release build completed all 1,938 targets at
  `2026-08-26T18:42:33Z`. Build log SHA-256:
  `ad40097ebdfa3c96d3f3e603cda94ef85d93dcdd0790950eb510e1de9d7fde97`.
- Runtime `_ttnn.so` SHA-256:
  `fb00a910d149f40848d47ccdd4b4cadcf96e21246109568bb6b4a2fe4cd8951d`.
  `libtt_metal.so` SHA-256:
  `6f0f3b4146bd7d059b818382b2e7401a5565af7381a48626fc0768ca9463feae`.
- Dedicated runtime is Python 3.10.19; its sorted `uv pip freeze` SHA-256 is
  `df89ac0f229cffc902f0bfd0d45785c862d7cf156716f09a64446ccd6e18026b`.
- Exact config/checkpoint and prior semantic-oracle suite: 32 tests passed, 10
  parameterized subtests passed, one deliberate environment-dependent skip.

## Hardware attempts

All attempts used partition A only, exact visibility `0,1,2,3`, launcher lease
verification, an independent nonblocking per-job lock, bounded timeouts, and no
reset. Pre/post ownership samples found no unexplained handle.

1. `2026-08-26T18:57:12Z`: strict `FABRIC_1D`, initial logical 1x4 request;
   failed on device 2 channel 4 during router handshake.
2. `2026-08-26T19:00:39Z`: strict `FABRIC_1D`, auto-discovered physical 4x1
   open followed by planned 1x4 reshape; failed identically before reshape.
3. `2026-08-26T19:05:12Z`: documented `RELAXED_INIT`, physical 4x1 open;
   failed identically. Channels 5, 6, and 7 reached remote handshake while
   channel 4 remained started.

The visibility and auto-discovery prefix passed: four available/PCIe/total
devices, local shape 4x1, and matching logical/physical degree histogram
`{1:2, 2:2}`. The mesh never opened, so no CCL or topology result exists.

Partition B remained unleased and untouched. Post-flight telemetry showed
firmware and PCIe/DRAM health unchanged, and no device-node handle remained.
