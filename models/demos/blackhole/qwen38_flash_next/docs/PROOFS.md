# Release proofs

The runs that verified this release end to end on hardware, from a fresh clone, following the README as written.
Where a proof quotes acceptance divergence indices measured before the GDN gate fix of 2026-09-06, the current
indices are in `NUMERICS.md`.

## The QuietBox, 2026-09-04 (pinned build)

A QuietBox (`tt-quietbox`: 4x p150c, fw 19.4.1.0) served the model on 2026-09-04 from a pinned build: startup
acceptance 96/96 against the CPU, 19.6 tokens/s at 32k.  The launcher and profile are the ones in this repository.

## 4x p150, 2026-09-05 (expert conversion rate)

On a 4x p150 host the first start's BF4 expert conversion took about 33 s per layer (2.2 GB written per layer;
measured 2026-09-05: 25 layers in 812 s).

## The QuietBox, 2026-09-06 (fresh clone of the public repository)

The QuietBox (`tt-quietbox`: 4x p150c, which `tt-smi` reports as p150b; firmware bundle 19.4.1.0, tt-kmd 2.6.0-rc1, 32 cores, 503 GB RAM, Ubuntu
22.04, clang-20, Python 3.10.19 through `uv`) served the model on 2026-09-06 from a fresh clone of the public repository
at `cadebdff7c1c`, following the README's sections 2-5 as written (the deviations found on the way are folded into the
README):

- clone 53 s, the three submodules 15 s, `build_metal.sh` 684 s with its defaults, `create_venv.sh` 95 s; the
  runtime identity of every run: head `cadebdff7c1c`, tree `dd25966f522c`, clean, extension
  `f3d1fb4c3ab4...`.
- the checkpoint copy already on the host: `download_checkpoint.py --verify-only` verified 142 of the 145 listed files
  in 13 s (LICENSE differed, `.gitattributes` and `configuration.json` were absent), the plain run fetched those three in
  17 s (145/145); `verify_checkpoint_files.py`: 131/131 shards, 360,000,192,888 bytes, every SHA-256 equal to the
  ModelScope listing (176 s, 4 workers).
- `prewarm_ple_table.py`: the 104,298,732,704 B of the n-gram table in 13.4 s (already resident), 33/33 files resident.
- the first start (`--profile tt-quietbox --acceptance --require-json-96`, 32k): mesh open 9.6 s; the 49 BF4 layers
  1772.5 s (35.9-36.9 s each, 100 GB written); target build 41.4 s; warm pass with a cold JIT cache 150.2 s; captures
  5.1 s and the chunk capture 1.7 s; acceptance replay 59.2 s; `READY` 2096.6 s after the mesh open (launched
  17:19:30Z, `READY` 17:54:33Z); 474,261,568 bytes free per bank after the captures.
- acceptance: `json` 96/96 (the gate passed); the other eleven records leave the CPU stream at the same indices as the
  4x p150 hosts did (chat 8, code 24, fact 15, list 46, math 61, multilingual 9, prose 13, refactor 22, sky 19, story
  6, summary 75: the table before the GDN gate fix of 2026-09-06, `NUMERICS.md` has the current one); 19.4-19.6
  tokens/s in the replays.
- requests over the LAN: a 36-token answer at 19.1 tokens/s (first token 0.30 s after a 33-token prompt), a 128-token
  generation at 19.6 tokens/s (first token 0.39 s, 47-token prompt in 2 chunks); the CLI's question answered.  SIGTERM
  stopped it cleanly (`result.json` status `stopped`, mesh closed, launcher exit 0).
- the same launcher line with `--mtp 4` (warm caches; the MTP kernels compiled on this start): `READY` 225 s after the
  launch (MTP warm pass 40 s, acceptance replay 48 s); `json` 96/96 through MTP at 55.0 tokens/s (4.8 tokens per
  pass), the split hand-off gate passed in both orders; `code` left the CPU stream at 44 and `fact` at 16, the other
  nine records at the plain-decode indices of that day (before the GDN gate fix of 2026-09-06; `NUMERICS.md` has the
  current MTP indices); 375,594,496 bytes free per bank (98.7 MB less than without MTP).
  Requests: the `json` prompt as a chat request reproduced the CPU record's 96 tokens at 55.2 tokens/s; a 6942-token
  prompt prefilled in 23.4 s (3.37 ms per prompt token, 217 chunks of 32 rows) then decoded at 31.3 tokens/s (3.0 per
  pass), its follow-up turn reused the 6980 committed tokens (first token 1.6 s, 43.1 tokens/s); a 128-token generation
  36.7 tokens/s (27.0 ms per token); a 220-token prose answer 31.0 tokens/s (2.6 per pass).
- the same line with `--long-chunks` (warm caches; the 128-row chunk kernels compiled on this start): `READY` 214 s
  after the launch; acceptance identical to the plain start (`json` 96/96, the same eleven divergence indices, 19.4-19.6
  tokens/s); 439,384,896 bytes free per bank (34.9 MB less than without).  The 6942-token prompt prefilled in 18.8 s =
  2.71 ms per prompt token (first token 18.9 s; 3.37 with 32-row chunks on the MTP start above), hand-off 805 ms, the
  same answer; decode 50.6 ms per token (19.6 tokens/s on a 128-token generation).  The `json` chat request again
  reproduced the CPU record's 96 tokens.
- `--allocated-context 65536` (the 64k component and model I/O caches built on this start, 9.8 GB): `READY` 220 s after
  the launch (target build 41 s, warm pass 13.5 s, acceptance replay 60 s); `json` 96/96, the same divergence indices;
  `/health` `context_limit` 65472; 419,440,704 bytes free per bank.  Left serving the LAN on port 8000
  (`--serve-seconds 86400`); a client on the LAN got its first token 0.30 s after a 33-token prompt.
- Every start above was stopped with SIGTERM between runs and closed its mesh (`result.json` status `stopped`,
  launcher exit 0); no board needed a reset.

## What the other profiles have

- `bh-loudbox` (Blackhole LoudBox, 4x p150 in one line): the README's numbers were measured on 4x p150 hosts (the
  performance table of 2026-09-04, the pinned divergence tables of 2026-09-06 in `NUMERICS.md`, the conversion rate of
  2026-09-05 above); a fresh-clone run of the form recorded above for the QuietBox is not recorded for a LoudBox.
- `qb2` (QuietBox 2, 2x p300c): designed from the p300 ring topology, never run on p300 hardware.
