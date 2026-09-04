# Qwen3.8-Flash-Next four-P150 run manifest

This manifest records the reproducibility boundary for the internal, text-only
bring-up of `Qwen/Qwen3.8-Flash-Next`. It is updated only when a new source,
runtime, firmware, or artifact pin is established. Timestamped raw evidence is
kept under `evidence/`.

## Codex invocation

- Launch UTC: `2026-08-26T17:19:54Z`
- Host: `f07cs02`
- Codex CLI: `codex-cli 0.147.0`
- Model: `gpt-5.6-sol`
- Reasoning effort: `max`
- Fast mode: disabled
- Approval mode: automatic approval review (`--approve-for-me`)
- Web search: enabled
- Workspace: `/home/sjett/tt-metal-qwen38-flash-next-agent-20260826`
- Data root: `/home/sjett/qwen38-flash-next-data`
- Goal thread: `01a03f15-e590-7e93-a8bc-9c5c10dcb973`
- Invocation:

```text
/home/sjett/.local/bin/codex \
  --model gpt-5.6-sol \
  --config 'model_reasoning_effort="max"' \
  --approve-for-me \
  --search \
  --disable fast_mode \
  --cd /home/sjett/tt-metal-qwen38-flash-next-agent-20260826 \
  --add-dir /home/sjett/qwen38-flash-next-data \
  exec \
  --output-last-message /home/sjett/qwen38-flash-next-data/codex-final-message.txt \
  - < AGENT_PROMPT.md
```

The launcher, Codex process, and code-mode host all inherited FDs 200--203 for
`/run/lock/tt-device-node-{0,1,2,3}.lock`. The same open-file descriptions are
retained for the lifetime of this run; they are not reacquired by descendants.

### Ultra/Fast continuation

- Resume UTC: `2026-08-26T20:23:11Z`
- Host: `f07cs02`
- Codex CLI: `codex-cli 0.147.0`
- Model: `gpt-5.6-sol`
- Reasoning effort: `ultra`
- Service tier: `fast`; Fast mode enabled
- Delegation: proactive ultra task delegation
- Approval mode: automatic approval review (`--approve-for-me`)
- Goal thread: `01a03f15-e590-7e93-a8bc-9c5c10dcb973`
- Recorded launcher PID: `4106323`
- Resume script SHA-256:
  `eb28976569334197855c193e0b41861602cd1fe764ad930d3d22080d1b25c583`
- Resume prompt SHA-256 recorded by the launcher:
  `3d4f570f16b134f1c334db1612d9c0fc1bbb065a7c30415f40dfe38f7209cab5`
- User TTNN-priority update SHA-256:
  `8ddf61e5aff1d283176ff9001f76912e245d92b173d94fbd38262ea83b08d8d8`
- Controlled-restart snapshot retained unchanged at
  `/home/sjett/qwen38-flash-next-data/logs/controlled-restart-20260826T202000Z`.

Invocation reconstructed from the preserved launcher and corroborated by
`ultra-fast-launcher.log`:

```text
/home/sjett/.local/bin/codex \
  --model gpt-5.6-sol \
  --config 'model_reasoning_effort="ultra"' \
  --config 'service_tier="fast"' \
  --enable fast_mode \
  --approve-for-me \
  --search \
  --cd /home/sjett/tt-metal-qwen38-flash-next-agent-20260826 \
  --add-dir /home/sjett/qwen38-flash-next-data \
  exec resume \
  --output-last-message /home/sjett/qwen38-flash-next-data/codex-ultra-fast-final-message.txt \
  01a03f15-e590-7e93-a8bc-9c5c10dcb973 \
  - < FOLLOWUP_BF8_PROMPT.md
```

The launcher records locks for nodes 0--3 on FDs 200--203 and exports exact
visibility. The command executor intentionally closes those high descriptors
in grandchildren and its process namespace does not expose PID 4106323, so an
individual command may not infer ownership from `/proc/self/fd` or `lslocks`.
Every future hardware wrapper must re-establish the host-visible launcher/lock
evidence plus the enforcing exact-device broker lease; no device was opened by
the continuation while that proof was unavailable to the command boundary.

## Repository start state

- HEAD: `d49ca492924e57af40db4f2f69dfef4ddda76fab`
- Branch: `sjett/qwen38-flash-next-agent-20260826`
- Origin fetch/push: `https://github.com/tenstorrent/tt-metal.git`
- HEAD subject: `Prototype recurrent K4 Qwen MTP drafting`
- Author/commit UTC: `2026-08-14T21:48:30Z`
- Tracked worktree diff at launch: empty
- Staged diff at launch: empty
- Empty-diff SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Initial untracked files: `AGENT_PROMPT.md`, `launch-agent.sh`
- User-owned files added while this run was active: `FOLLOWUP_BF8_PROMPT.md`,
  `watch-and-resume-bf8.sh`; these are preserved and excluded from task edits.
- This is an isolated worktree. Other worktrees listed in timestamped evidence
  are read-only and out of scope.

Submodule gitlinks were uninitialized at launch. The isolated build initialized
these exact revisions without changing the recorded gitlinks:

- llama reference: `29125b7ad8b5513eeaa4417ed92892bf39c8bd74`
- Tracy: `117100515bb21d9a6b3a8f0eee50ecd91f961408`
- cluster descriptors: `7b2176e2fe913089f8cd2be9dfb738ead6e7aa27`
- UMD: `1476f9298cd178b71636860f843f22f416031681`

## Host software start state

- OS kernel: Linux `6.8.0-110-generic`, x86-64, Ubuntu 22.04 userspace
- Python: `3.10.12`
- Git: `2.34.1`
- CMake: `4.3.2`
- Ninja: `1.10.1`
- GCC: `11.4.0`
- Clang: absent from the default `PATH`
- Tenstorrent KMD: `2.7.0`, module source version
  `6F5DF618C14D0C708A55788`
- TT-SMI utility selected for read-only telemetry: `6.2.0`
- TT-SMI UMD dependency: `0.9.8`
- TT-SMI pyluwen dependency: `0.8.5`
- Partition-A firmware bundle: `19.8.1.0` on all four cards
- Partition-A Ethernet firmware: `1.10.1` on all four cards
- Partition-A CM firmware: `0.30.1.0` on all four cards
- Partition-A DM application firmware: `0.24.1.0` on all four cards
- Partition-A GDDR firmware: `2.15` on all four cards
- Available memory at startup: about `707 GiB`
- Free workspace filesystem space at startup: about `1.6 TiB`
- Explicit huge pages: zero at startup; the host hugepage service is inactive
- Expected system handle: `tt-telemetry.service`, PID `843856`, active since
  `2026-08-21T15:22:42Z`

The current worktree has no build tree or dedicated Python environment at
startup. Runtime/build pins will be added before they are used.

## Physical partition pin

Only partition A is leased or eligible for jobs. Partition B remains idle.

| Node | BDF | NUMA | IOMMU | PCIe | Stable board ID | Status |
|---:|---|---:|---:|---|---|---|
| 0 | `0000:61:00.0` | 0 | 10 | Gen5 x16 | `360E5028754A871F` | leased A |
| 1 | `0000:41:00.0` | 0 | 21 | Gen5 x16 | `9D5052CD17DFF36F` | leased A |
| 2 | `0000:01:00.0` | 0 | 34 | Gen5 x16 | `AF4F213752F4D208` | leased A |
| 3 | `0000:21:00.0` | 0 | 49 | Gen5 x16 | `8FD48C3DF1CF1C85` | leased A |
| 4 | `0000:e1:00.0` | 1 | 60 | Gen5 x16 | `4E3FC21149A3FEF0` | unleased B, idle |
| 5 | `0000:c1:00.0` | 1 | 70 | Gen5 x16 | `B411631997534A40` | unleased B, idle |
| 6 | `0000:81:00.0` | 1 | 81 | Gen5 x16 | `BB82CB05ED639CCF` | unleased B, idle |
| 7 | `0000:a1:00.0` | 1 | 90 | Gen5 x16 | `670987CC04D2A001` | unleased B, idle |

`TT_VISIBLE_DEVICES=0,1,2,3` is necessary but is not treated as ownership or
topology proof. TTNN logical IDs, firmware, Ethernet/fabric adjacency, and
`tensor_topology()` placement remain gated until their bounded tests pass.

TT-SMI/UMD maps the visible logical IDs in BDF order, not `/dev` node order:
logical `0,1,2,3` map to nodes `2,3,1,0` respectively. The first snapshot found
DRAM training passed, zero uncorrectable GDDR errors, Gen5 x16 links, and ASIC
temperatures from 49.1 to 52.1 C on all four A cards. Raw `ETH_LIVE_STATUS` was
`0x0` on each card, so TT-SMI alone is not accepted as fabric-adjacency proof.

## Input prompt provenance

- `AGENT_PROMPT.md` SHA-256:
  `ba14d7f91e190a354abffa40122b57296392621ed5ff944ad749c65229b43c39`
- `launch-agent.sh` SHA-256:
  `f6a293d5c93a5d8a613e9c26457fdf01f30b01d822abb96c88afc596db465537`
- Precision continuation prompt SHA-256:
  `16a17f88a51cac547ab82b027ac31d85442499e345593f60bd256b258042c3ce`
- User-owned continuation watcher SHA-256:
  `fbe5968c7b99b2f550a8c92788c4c1cacd26915f0ada930729f68ddc95565731`
- User priority update SHA-256:
  `8ddf61e5aff1d283176ff9001f76912e245d92b173d94fbd38262ea83b08d8d8`

## External source and artifact pins

### Exact model checkpoint

- Hugging Face repository: `Qwen/Qwen3.8-Flash-Next`
- Hugging Face revision:
  `f5d08274bafd880402bd16f5e3e6c514136ec06c`
- Hugging Face revision metadata SHA-256:
  `4effe5bb8fc7f4ad7065da0e4b9fd1f1a0c9a3f7be09a26d058a498425d7c082`
- ModelScope release artifact revision:
  `2741eec155d03a8ce151b993ccce1a7b1e398d6b`
- ModelScope current master at validation time:
  `fa2551ab7dd1c12000ea602a390e6cd080e96a93`; its only later change is
  system metadata/README provenance, not model payload
- ModelScope exact release tree SHA-256:
  `161d8e96d0bec4be97b80bc7b69d13ce1f53d14fe2ff21f9343f0ef048554644`
- Safetensors index SHA-256:
  `99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de`
- Bounded 131-shard header manifest SHA-256:
  `c4b3bdf6e57d6473caf89083a520a67f2975d225c0dc1050e360eba84d0c7c33`
- Flat 1,658-tensor manifest SHA-256:
  `ebf4de2015233c20e21ff1e7e388e871208158ed350fea42964dbd52ca69359b`
- Exact tensor data: 359,999,963,128 bytes: 179,999,981,424 BF16
  elements and 35 I64 PLE-metadata elements
- Exact weight shard files including safetensors headers:
  360,000,192,888 bytes

All 131 computed file sizes match the exact SHA-256/size entries in the
ModelScope release manifest. Hugging Face and ModelScope semantic artifacts and
weights match by hash. Their `.gitattributes` differ, and the two retained
license files differ only by CRLF versus LF line endings.

Key small artifacts:

- `config.json`: `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b`
- `generation_config.json`: `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e`
- `chat_template.jinja`: `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`
- `tokenizer.json`: `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`
- `tokenizer_config.json`: `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27`
- Hugging Face license: `a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08`
- ModelScope LF license: `465dcafcac7d3542a0cfc150fc550e0b7228c22ee68c8f1f81db942e7c658cb4`

The exact snapshot is complete at
`/home/sjett/qwen38-flash-next-data/checkpoints/Qwen3.8-Flash-Next-f5d08274`.
All 144 local files are recorded in
`manifests/checkpoint-f5d08274-sha256.json` (SHA-256
`13c88f393ffbe4f5e9733d8e48a59f21c77e69a5f02b76bb073835d9a1ca0ea9`).
Every one of the 131 weight-shard hashes matches the exact ModelScope release
manifest; the aggregate shard-file size is 360,000,192,888 bytes.

### Official architecture sources

- Qwen report repository revision:
  `513aa6e18a335296fc13e538232a8735b230877d`
- `tech_report.pdf` SHA-256:
  `04f263446d74a35cb7cea368574e0c561f3b05c133be2c777ac884404063655d`
- Official repository README SHA-256:
  `d8263b56c71a467667183f7d6ce10addf59fba4da881341b16def76bd7c40f02`
- Official architecture-blog snapshot SHA-256:
  `752c3f68eb550a94d534ad4b0c04c5db3083b67a8cf0da9204e9c54d45d1486e`

### Reference implementations

- Transformers repository revision:
  `dabae5fcb924a8eece0e727b627ca5f050b40d40`
- Revision time: `2026-08-26T14:45:18Z`
- Full source archive SHA-256:
  `fa12f6f0a2dfc21ff3e9c3901c15e4ede139dcf5118f78af8df2a929eaffeca7`
- Imported pinned source version: `5.16.0.dev0`
- Official SGLang Qwen4Exp pull-request head:
  `73a255206f916366c8d26d4022f82ddfb0ab558d`
- Pull-request parent recorded by the upstream API:
  `861ecaef282431fc83f505d131a39f5c6b833459`
- SGLang `qwen4_exp.py` SHA-256:
  `f406977eb2373937393241f453477867f7dc943bd4839216db8fe66fa9f921d8`
- SGLang `qwen4_exp_mtp.py` SHA-256:
  `b52c063123611d521ff8c8c111b9fd935c06bdef638e829735b99da12266c77d`
- SGLang QSA backend SHA-256:
  `6cb122eb32ad5c07d42dfc2bfb2ae43cb3111befa32ae95d03b09fc732bc2de6`
- SGLang shared-index test SHA-256:
  `ca41859fb992add3d24175f5d08f1ef6fc90dc63dbfc5fcc7665985286463943`
- SGLang verifier-commit test SHA-256:
  `3f1b99093a8ba48d3472fc9d0aa918c3a1551a64e5717d14bb76b72d61b8e0a8`
- vLLM main audit pin: `1a085dadf254b7cb2b5f904747a5750256d413fa`;
  no Qwen4Exp implementation was present at this revision
- Qwen3.6 Blackhole bundle revision:
  `42a71032659a9d0e3b860b6104c59cfa6ca82149`
- Qwen3.6 revision metadata SHA-256:
  `5d607799cea61a4c78f8bb180378cdd1fb4d10d6d9a63ec29a2d43fe8c55d74c`
- Qwen3.6 source checkout: detached, clean, and fsck-clean under the dedicated
  data root; its last source commit is
  `a7c60b02592ba41779777301c02d65785b250f92`, followed only by the pinned
  README commit

The SGLang integration is required because the pinned Transformers model class
does not load or execute `mtp.*`. It establishes the exact four-branch
`fc_hidden` plus broadcast `fc_embedding` input mix, frozen QSA selection reuse
during draft steps, and accepted-step verifier state commit.

### CPU oracle runtime

The dedicated runtime is
`/home/sjett/qwen38-flash-next-data/oracle-venv`. Its imported versions are:

- Python `3.10.12`
- PyTorch `2.11.0+cpu`
- tokenizers `0.23.1`
- safetensors `0.8.0`
- pytest `9.0.3`
- huggingface_hub `1.16.1`

The exact tokenizers wheel SHA-256 is
`5075b405006415ea148a992d093699c66eb01952bf59f4d5727089a98bda45a4`.
Oracle commands set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, confine pytest discovery
to the new test directory, and put the pinned Transformers source first on
`PYTHONPATH`.

### Completed payload and runtime artifacts

- Exact BF16 snapshot destination:
  `/home/sjett/qwen38-flash-next-data/checkpoints/Qwen3.8-Flash-Next-f5d08274`
- Snapshot source/revision: `Qwen/Qwen3.8-Flash-Next` at
  `f5d08274bafd880402bd16f5e3e6c514136ec06c`
- Download concurrency: four workers; completed
  `2026-08-26T18:34:29Z` with status 0
- Download log:
  `/home/sjett/qwen38-flash-next-data/logs/weights-f5d08274-download.log`
  (SHA-256
  `cfac2ecbd0bf4a598788b45af861800e1abcb073d6ef093a76a9aae9d97c7592`)
- Isolated release build root:
  `/home/sjett/qwen38-flash-next-data/build/tt-metal-181ac08075-release`
- Dedicated CPM cache:
  `/home/sjett/qwen38-flash-next-data/build/cpm-cache`
- Build log:
  `/home/sjett/qwen38-flash-next-data/logs/tt-metal-181ac08075-release-build.log`
  (SHA-256
  `ad40097ebdfa3c96d3f3e603cda94ef85d93dcdd0790950eb510e1de9d7fde97`)

The first configure at `2026-08-26T18:28:39Z` failed closed because this
isolated worktree's recorded submodules were uninitialized. The restarted build
initialized only those exact gitlinks, then built and installed all 1,938
Release targets at `2026-08-26T18:42:33Z`. Neither long-pole job opened a
Tenstorrent device. Build options include `ENABLE_DISTRIBUTED=ON`,
`ENABLE_TRACY=ON`, and `BUILD_TESTING=ON`.

The dedicated execution environment is
`/home/sjett/qwen38-flash-next-data/runtime-venv`, Python 3.10.19, PyTorch
2.11.0+cpu, safetensors 0.8.0, pytest 9.0.3, with TTNN installed editable from
this exact worktree. Its sorted `uv pip freeze` SHA-256 is
`df89ac0f229cffc902f0bfd0d45785c862d7cf156716f09a64446ccd6e18026b`.
The active extension and runtime library hashes are:

- `_ttnn.so`:
  `fb00a910d149f40848d47ccdd4b4cadcf96e21246109568bb6b4a2fe4cd8951d`
- `libtt_metal.so`:
  `6f0f3b4146bd7d059b818382b2e7401a5565af7381a48626fc0768ca9463feae`

The TTNN-only continuation incrementally rebuilt the extension after adding
the true-B1 local-combine mode, explicit output-topology contract, and exact
per-mesh-coordinate DRAM-ring query. The qualified no-device runtime snapshot
is `/home/sjett/qwen38-flash-next-data/runtime-local-combine-ring-qualified-20260826`.
Its current hashes are:

- `_ttnn.so`:
  `1911ac8b460f0d5d4b943319cf121c22ab5484cd96edfe3728c4dae94ff9cbc6`
- `_ttnncpp.so`:
  `4eb634018f466d5329d03c5b15cd141616cf706e52dcf2d3ff7bbfceb32eb61c`

`ldd` resolves every dependency on `f07cs02`. `readelf` shows absolute RUNPATH
entries under the exact local build root, so this snapshot is not portable to
another host without the required ABI, dependency, RPATH, hash, and bounded
locked-smoke qualification.

## Hardware execution and fabric result

The command executor does not propagate FDs 200--203 to its grandchildren.
Hardware wrappers therefore fail closed unless a live ancestor still owns all
four exact launcher lock descriptions, then take the distinct nonblocking job
lock `/run/lock/qwen38-flash-next-partition-a-job.lock`. They do not reacquire
the launcher locks, which would create new open-file descriptions and would not
preserve the original lease semantics.

Three bounded topology jobs ran on A only. Visibility and auto-discovery found
exactly four local PCIe devices in a physical 4x1 line. Both strict attempts and
one documented `RELAXED_INIT` attempt failed before mesh open on the same
router: device 2, Ethernet channel 4 remained `STARTED` while channels 5--7
reached `REMOTE_HANDSHAKE_COMPLETE`. The runtime timed out waiting for
`LOCAL_HANDSHAKE_COMPLETE`, disabled fabric, and closed all devices. Strict
transcript hashes are recorded in `BLOCKED.md`; the relaxed job record is
`/home/sjett/qwen38-flash-next-data/jobs/topology-gate-partition-a-3978449.record`.

Post-flight ownership/health samples at `2026-08-26T19:05:58Z` and
`2026-08-26T19:07:26Z` found no leaked device handle and no unexpected owner.
No reset, service change, process kill, or B-partition access occurred. Because
the mesh never opened, allocator capacity, actual tensor topology, and CCL
values remain unverified at runtime.

## Static memory admission pin

The pinned Blackhole descriptor exposes eight 4,278,190,080-byte DRAM views,
or 34,225,520,640 raw bytes/device (31.875 GiB). The exact four-device baseline
places 128 routed experts/device in BF4_B and conservatively budgets other
device weights as BF16. It requires 19,885,016,576 static bytes/device and
28,908,998,160 bytes/device after batch-one 8K state plus layout, scratch,
workspace, trace, and fragmentation reserves. Raw remaining headroom is
5,316,522,480 bytes/device. See `MEMORY_BUDGET.md` and immutable evidence for
the formula and precision rejections.

This raw-BF4 admission is superseded by D017. `moe_compute`'s ring-aware packed
layout requires 23,366,762,496 routed bytes/device for all 49 layers on a
seven-bank Blackhole or 26,704,871,424 on an eight-bank Blackhole. Both violate
the conservative runtime-reserve gate when fully resident. The admitted
correctness path streams one routed layer and totals 3,021,063,680 or
3,089,188,352 static bytes/device respectively, without changing 128-expert
ownership or substituting replication.

## Post-maintenance one-shot gate — 2026-08-26T22:28Z

- Resumed source HEAD:
  `76ab274ccaca8edf9dac8c5ba3759e559cdb8ace`
- Maintenance snapshot `SHA256SUMS` SHA-256:
  `e4c3318df80de07023c5e8baff002e497def943e7a5845227d6e2b387484ade5`
- Authorized-reset evidence `SHA256SUMS` SHA-256:
  `001632b3cb22bb247d60e9be119010892e1d667a7f7cd38f22433beb09511dfa`
- Qualified TTNN Python package:
  `/home/sjett/qwen38-flash-next-data/runtime-local-combine-ring-qualified-20260826/site-packages/ttnn`
- Qualified `_ttnn.so` SHA-256 / ELF build ID:
  `1911ac8b460f0d5d4b943319cf121c22ab5484cd96edfe3728c4dae94ff9cbc6` /
  `91ee0b9ea782dd46bbf354e841fa337cece291fb`
- Import proof: run from `/tmp`; qualified site-packages preceded the repository;
  both mesh-coordinate bindings were present.
- Live launcher/Codex PIDs: `65610` / `65694`; exact node locks appeared on
  ancestor FDs 200--203.  Scoped job lock:
  `/run/lock/qwen38-flash-next-partition-a-job.lock`.
- Telemetry PID: `62612`, active/running.  No unexplained handle in either
  preflight sample or the post-run sample.
- Exact gate evidence:
  `/home/sjett/qwen38-flash-next-data/evidence/20260826T222657Z-strict-topology-gate-a`
- Gate transcript SHA-256:
  `901bd88acc9252db6c192ea3ef67688d808140eae83bdbaaff12c3eeafe08650`
- Result: `runtime_unverified`.  UMD discovered local IDs 0--3 with PCIe IDs
  `[2,3,1,0]`, then `GetNumAvailableDevices()` failed before fabric setup
  because installed TTNN selected an unbundled runtime root and could not read
  `tt_metal/soc_descriptors/blackhole_140_arch.yaml`.
- Teardown: cluster destructor completed; exit 1; no result JSON; no leaked
  handle.  Post-run cards remained P150b/DRAM healthy/Gen5x16/zero GDDR
  uncorrectables/FW 19.8.1.0/ETH FW 1.10.1.
- Import-only diagnosis: setting
  `TT_METAL_RUNTIME_ROOT=/home/sjett/tt-metal-qwen38-flash-next-agent-20260826`
  before import selects the pinned SoC descriptor SHA-256
  `2aa71c2d4321c186d2958ee05a82db4c4c405c4420ed31c1d592f490994942ef`
  and core descriptor SHA-256
  `a74bbec1be32402cf0033550085f3364d9b31be62d314a7bde525d0748400348`.
  This proof did not open hardware.
- The sole post-reset hardware-opening allowance is consumed.  No further
  hardware job, relaxed mode, reset-like action, or partition-B access is
  authorized in this continuation.

## Full CPU ordinary-decode oracle

The exact-checkpoint lazy CPU integration oracle ran all 48 language layers,
the final hyper-connection mixer, and the untied 248,320-row LM head for two
stateful greedy positions. It opened no Tenstorrent device. The passing log is:

- path:
  `/home/sjett/qwen38-flash-next-data/logs/qwen38_flash_next_cpu_oracle/20260826T1955Z-full48-prefill-equivalence.jsonl`
- SHA-256:
  `138f648bd52f226ac4fece52337be7917c228ef9735fd6cf47b0c9fd20239797`
- size: 40,299 bytes, 149 JSONL records
- peak RSS: 15,518,752 KiB
- greedy tokens: `17 -> 15 -> 16`

The log also compares the second tokenwise position against a two-token BF16
prefill. It records final hidden/logit distributions, all per-layer routing
overlaps, and recurrent/KV/PLE state distributions. A preceding stricter gate
that incorrectly required all ten router ranks to remain bit-identical across
different BF16 matmul batch shapes failed and is retained at
`20260826T1953Z-full48-prefill-equivalence-strict-failed.jsonl`, SHA-256
`8d21c8d7db56e0aa69d83609c387a15a6fa3d4a3460df5c3766ad7f1580e226d`.
