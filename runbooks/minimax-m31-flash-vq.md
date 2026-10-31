# Recipe: MiniMax-M3.1-Flash VQ on the DS4 CUDA engine

MiniMax's 428B native multimodal MoE, quantized to about 2.0 bits per routed-expert
weight, served on one DGX Spark at 524,288 tokens on port 8011 — the author's own
configuration for a 128 GB host, and the shape his card protocol allocated. 61 GGUF
parts, 115,526,325,376 bytes, in a GVQ1 release directory read by the ds4-dfm-rs Rust
host. Text, images and video and the DSpark draft head come with it; `MINIMAX31_CTX=131072`
holds the K/V resident with no tier file, and `MINIMAX31_CTX=` serves the native
1,048,576, which his own measurement warns against on this box.

**The payload is on the reference Spark and verifies; nothing has been loaded, and
nothing here has been measured.** The engine is no longer the obstacle it was when this
file was written: PR #110, the `minimax_m3_vq` family, merged into main on
2026-10-25T02:33:10Z (head `4988ae7`), so a current `main` build carries it. What is
missing is the measurement — no `--check-config` quote, no decode rate, no time to first
token has been taken for this artifact anywhere but on its author's own box, and every
speed and quality figure below is his, labelled with the build it came from. The first
start's own output belongs in this file once it has been run.

- Contract: [`recipes/minimax-m31-flash-vq.yaml`](../recipes/minimax-m31-flash-vq.yaml)
- Installer: [`scripts/install-minimax-m31-flash-vq.sh`](../scripts/install-minimax-m31-flash-vq.sh)
- Artifact: [`Baekpica/MiniMax-M3.1-Flash-VQ-GGUF`](https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF) at `2789cbaf`, weights unchanged since `74e5a9fd`
- Engine: [`Baekpica/ds4-dfm-rs`](https://github.com/Baekpica/ds4-dfm-rs), family `minimax_m3_vq`, [PR #110](https://github.com/Baekpica/ds4-dfm-rs/pull/110) — merged into `main` 2026-10-25T02:33:10Z
- Source model: [`MiniMaxAI/MiniMax-M3.1-Flash`](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) at `54e4e69d`

## What this artifact is

Sixty layers, three dense and 57 MoE, hidden 6,144, 128 routed experts with top-4
routing plus a shared expert, expert FFN 3,072, GQA with 64 query and 4 KV heads of
128, per-head Gemma QK-norm, partial RoPE on 64 of 128 dimensions, MiniMax Sparse
Attention on every layer, vocabulary 200,064. 413.1B of the 428B parameters are in
the routed experts, and they are what the conversion attacked:

```
routed experts        13-bit VQ on all 57 layers, 8-bit residual stage on 20 of them
                      1.63 bits per weight on 37 layers, 2.63 on 20, 1.98 on average
attention q/k/v/o     Q4_K, per-column calibration importance
shared experts        Q4_K (shared with the dense FFN of layers 0-2 and the head)
MSA indexer           Q8_0 index_q and index_k
router, bias, norms   F32 and source precision, kept exactly
token table           BF16, its own part, read in rows
vision tower          BF16 at source precision, its own part, mapped on demand
DSpark drafter        Q8_0, its own part, 2.03 GiB resident
```

The residual stage (R13+8) is the interesting part of the recipe: the author
converted every MoE layer both ways, switched each layer alone to the residual stage
and re-ran the held-out set end to end, then took the twenty layers whose assistant-turn
KL dropped most — 16, 18-32, 35, 40, 43 and 53, in the middle of the stack. The
rule of thumb (protect the bottom layers) would have given KL 0.171 instead of 0.150
on the same byte budget.

## Artifacts

Sixty-one independent GGUF parts, not numbered splits of one file: each expert layer
is a complete GGUF for that layer, and the release is opened as a **directory**.

| Part | Bytes | Notes |
|---|---:|---|
| 37 R13 expert parts | 1,475,503,488 (layers 03-09) and 1,475,503,872 (the rest) | GVQ1 container 1, 385 tensors each |
| 20 R13+8 expert parts | 2,381,629,312 | container 2, 770 tensors each |
| `body-nonexpert.gguf` | 6,930,488,832 | attention, shared experts, dense FFN, head, indexer, router, norms |
| `embed-tokens-bf16.gguf` | 2,458,387,456 | the token table, read in rows while serving |
| `vision-bf16.gguf` | 1,734,197,248 | image and video tower, mapped on demand |
| `dspark-q8_0.gguf` | 2,177,025,024 | the draft head, resident |
| **all 61** | **115,526,325,376** | 107.59 GiB |

`SHA256SUMS` covers 224 files: the parts, the engine's own inputs (`release.json`,
`files.json`, `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja`), the calibration and quality evidence, the conversion tools
and a per-layer receipt for every expert part. Every part digest in `release.json`
matches its `SHA256SUMS` line.

The weights are under the **MiniMax Community License**, not MIT: non-commercial use
is free, commercial use requires displaying "Built with MiniMax M3.1-Flash" and
notifying MiniMax unless annual revenue is above 20 million USD, and the license's
prohibited uses apply. This is a deliberate difference from the other VQ artifacts in
this repository, whose weights are MIT.

## Where the pieces go, and why

The rule this repository follows is that bytes read once at load and then held
resident belong on the roomy volume, and bytes read while the server runs belong on
the fastest device. This artifact is the first one here where the engine's own layout
puts part of the second group inside the first group's directory:

| Where | What | Why |
|---|---|---|
| `/mnt/models/MiniMax-M3.1-Flash-VQ-GGUF` | the 57 expert parts, `body-nonexpert.gguf`, `dspark-q8_0.gguf` — 103.69 GiB | copied to the device once at open, never read again |
| `/mnt/models/MiniMax-M3.1-Flash-VQ-GGUF` | `embed-tokens-bf16.gguf` (2.29 GiB), `vision-bf16.gguf` (1.62 GiB) | **must stay in that directory**: the engine reads every part as a filename inside the release directory and refuses a directory holding a GGUF that `release.json` does not declare |
| NVMe, by `DS4_KV_OFFLOAD_DIR` | the K/V tier: an unlinked `O_DIRECT` file, 33.75 GiB at 1M | read and written per token and per prefill chunk once the context passes 131,072 tokens |

The third row is the one that matters for speed, and it is an engine option, so the
launcher sets it: the default is `~/.cache/ds4/kv-offload`, which on a Spark would put
36.2 GiB of per-token reads behind the home partition's own traffic. The first two
rows are the same directory because the engine gives no way to split it — the
DeepSeek recipes' separate `--engram-dir` has no counterpart here. What that costs is
small and bounded: the token table is read one 12,288-byte row per token (about
245 KB/s at 20 tok/s), and the vision tower is streamed whole, 1.7 GB, on the first
image after a cold start or an eviction, then dropped from the page cache. Moving
those two files to a faster device would need symlinks inside the release directory,
which this recipe does not do.

The tier's file is reserved in full with `fallocate` when the bank is created, before
the port binds, so a start at 1M needs 36.2 GB free on the tier's filesystem. It is an
unlinked file: nothing is left behind after a stop, and a full disk fails at open
rather than mid-forward.

## Why the engine requirement is a format floor

The parts declare their own source revision — `minimax_m3_vq.source_revision` must be
`54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e` — and a container version per part: 1 for
the R13 layers, 2 for the twenty with the residual stage. The reader checks the pair,
`release.json`'s format (`minimax-m3-vq8`), architecture (`minimax_m3_vq`), context
(1,048,576), the 57-layer list, and the twenty R13+8 layers, and it refuses the
directory rather than misreading it. An engine without this family does not fail with
a bad answer; it fails at identify. That is why the requirement is stated as a
**format floor**: any build carrying the `minimax_m3_vq` family will do, and no build
before it will pretend to.

The reference checkout here carries the work on `origin/feat/add-minimax-vq`
(`fd7192f4` when this recipe was written), and the author's measured builds were
`ff430197` (the gated DSpark decode) and `c90530f1` (the benchmark tree). Since the
pull request is a draft, the engine is rebuilt from that branch, not from `main`.

## Why the non-obvious flags are on the command line

```
DS4_KV_OFFLOAD_DIR="$HOME/models/minimax-m31-vq/kv-offload" \
  ./ds4-dfm-rs/ds4-server --cuda \
      -m /mnt/models/MiniMax-M3.1-Flash-VQ-GGUF \
      -c 524288 \
      --mtp-mode on --mtp-draft 3 --cont-width 0 \
      --host 0.0.0.0 --port 8011 --model-id minimax-m31-flash-vq
# with DS4_SERVER_CONTINUOUS=0 exported as well
```

- `--mtp-mode on --mtp-draft 3`: the DSpark part joins the trunk only when the draft
  count is above 1, and the family's `auto` resolves to off, so `on` is not optional
  if the drafter is wanted. Three proposals is the card's own sweep setting and the
  engine's useful range on this hardware: a four-row verify costs about 2.4 decodes,
  so drafting only pays while a trial keeps more than about 2.6 tokens. The engine
  measures that itself and decodes plainly while drafting runs slower — the rate gate
  is the difference between 18.65 and 19.56 tok/s in the author's sweep, and greedy
  output is identical with or without it.
- `--cont-width 0` with `DS4_SERVER_CONTINUOUS=0`: **this is the flag pair that decides
  whether the model serves at all.** MiniMax-M3.1 VQ is a persistent-bank family whose
  speculation runs on the serial lane (`serving.rs`, the `ServingCaps` for `MinimaxVq`:
  `banks: Persistent`, `spec_lane: Serial`). The engine refuses the opposite pairing, and
  the refusal is a boot failure rather than a warning:

```
error: minimax_m3_vq MTP runs only on the serial lane (mtp_lane)
ds4-server-rs: serving plan rejected unsupported options
```

  The rule behind it is `resolve_mtp`: for a serial-lane family it turns MTP off with
  that error whenever `bank_driver()` reports the bank driver present, and `bank_driver()`
  is present for any `--max-seqs` width on a persistent-bank family unless the serial
  route was selected. So asking for a bank width and the drafter together cannot work,
  which is exactly what this launcher did before the first Spark run taught it otherwise.
  The engine's own way to select the route is in the author's serving contract
  (`docs/serving-contract.md`, "Serving lanes"): `--cont-width 0` and
  `DS4_SERVER_COALESCE_MAX=0` keep the legacy serial meaning, no bank lane, while
  `DS4_SERVER_CONTINUOUS=0` is narrower — it forces the static/serial route so no request
  enters the bank driver. The resolved plan emits `DS4_SERVER_CONTINUOUS=0` itself for a
  serial-MTP family (`ResolvedPlan::env_overrides`), so the launcher passes both. Note
  that `--max-seqs` accepts only `auto` or 1..64: `0` is the legacy alias's value and is
  passed as `--cont-width 0`, never as a width.
- The bank width is therefore `--max-seqs 1` only when the drafter is off, where one bank
  is what the memory plan funds (113.95 GiB of the 117.2 GiB the engine sees on GB10).
  Written down rather than left to the automatic width, so the file, not the fit,
  decides.
- `-c 524288`: the served context, and the one flag that changes what a decode step reads.
  It is the author's own configuration for a 128 GB host (see the section above), and his
  card protocol's allocation. `-c 131072` holds the K/V resident and creates no tier file;
  the family would otherwise follow its native 1,048,576, whose tier is 33.75 GiB.
- `--model-id`: the served name. The family's syntax comes from the loaded model, not
  from this string.
- `--mem-floor-gb` is left at the engine's default of 4 GiB; the launcher's own gate
  and watchdog are described below.

Two lines of the plan the engine prints before it loads are worth reading every time. The
`effective:` line is what will actually serve — `mtp` there is the drafter's real state,
and `mtp=off` with a requested `mtp=on` is the signature of the rejection above. The
`qualified:` line carries the engine's own honesty about this family: `reuse=unverified`,
`disk=none`, `mtp=none`, which is why the launcher's flags are the only thing standing
between a working serve and a silently plainer one.

## The author's serving shape, which is the default here

His guide (`docs/minimax-vq.md` in the engine checkout, "DSpark drafting" and the prefill
and K/V sections) and the card's sweep describe one configuration, and it is what this
launcher runs without being asked:

| Decision | Value | Where it comes from |
|---|---|---|
| Context | 524,288 | the author's configuration for a 128 GB host: his guide's pool note names the 512K DSpark and 1M configurations, and his table calls the 1M-DSpark peak "not a safe configuration on a 128 GB host" |
| Drafter | `--mtp-mode on --mtp-draft 3` | his sweep setting; "two or three proposals are the useful range on GB10" |
| Rate gate | on | the engine's default, and his measured best: it moved the 2K-64K mean decode from 18.65 to 19.56 tok/s |
| Prefill chunks | the engine's 4,096 rows | his default *is* the maximum; `DS4_MINIMAX_PREFILL_CHUNK` selects up to it |
| Lane | serial | where DSpark runs; selected by `--cont-width 0` and `DS4_SERVER_CONTINUOUS=0` |
| K/V | index keys resident, main K/V in the SSD tier | his recipe; this launcher overrides only the directory, to put the tier on the NVMe |
| Pool | the engine's own sizing | deliberately small for a 1M plan; his measurement at 256K: 14.7 -> 16.7 tok/s from 9,660 to 65,000 slots |

Nothing in the launcher pins an engine tuning variable. Those defaults are already his
measured ones, and a silent pin would make a future engine's better default unreachable;
each can be set in the environment exactly as he documents it
(`DS4_MINIMAX_DSPARK_GATE`, `DS4_MINIMAX_PREFILL_CHUNK`, `DS4_MINIMAX_KV_POOL_SLOTS`,
`DS4_MINIMAX_KV_RESIDENT_CTX`). `plan` prints that fact with every run.

One number to hold on to when judging speed: a four-row verify costs about 2.4 decodes,
so the drafter pays only while a trial keeps more than roughly 2.6 tokens. On raw prose it
often does not, and his ungated sweep decoded 10.7-26.5 tok/s per frontier against about
19 plain — which is precisely why the gate exists and why it is on.

## The K/V tiers, and what the indexer buys

MSA keeps every position on every layer: 72 bytes of index key plus 576 bytes of K/V
per token and layer, 38,880 bytes per token, 37.97 GiB at 1M and 18.99 GiB at 512K.
Selection saves reads, not storage. The indexer needs only the index keys to decide what attention reads, so:

- **index keys stay resident**, in NVFP4, in every tier: 4.22 GiB at 1M, 2.11 GiB at the
  524,288 tokens this launcher serves by default;
- **K/V pages stay resident up to 131,072 tokens** — 4.5 GB per bank — and
  `DS4_MINIMAX_KV_RESIDENT_CTX` moves that threshold (0 sends everything to the SSD);
- **past it they live in the SSD tier**, and the pool of pinned slots the GPU reads in
  place holds what a forward touches: the selected blocks per layer, the blocks a
  prefill chunk's rows write, and the previous step's blocks.

The measured cost of the tier on this hardware, from the author's notes: a decode step
fetches 11-13 MB at 8K and 29 MB at 32K with only the previous step's blocks pooled
(66 MB worst case), and an LRU of 64 blocks per KV head and layer brings 32K to 6 MB.
At 256K the default pool left 300 misses and 10.9 ms of reads per token (14.7 tok/s);
65,000 slots (4.46 GiB) took that to 68 misses, 3.3 ms and 16.7 tok/s. That is why
`DS4_MINIMAX_KV_POOL_SLOTS` is worth reading about before tuning a long-context
deployment: it trades memory the 1M plan does not have for misses.

## Running, and what a start looks like

**No start has been observed on the reference Spark.** What follows is written from
the engine's source, and the first real start should replace it.

1. The host hashes the release before it loads anything. Every part is read with
   `O_DIRECT` in up to 16 threads, its Q4_K and Q8_0 scales are checked as the bytes
   stream past, and the digest is compared against `release.json`; each part's page
   cache is then dropped, because streaming 110 GB through the page cache would
   fragment the free memory the device's 2 MiB pages come from. This is a full read of
   115.53 GB, so the first minutes of a start are disk-bound, and it is why the
   installer's `check` action is not instant either: `--check-config` runs the same
   open.
2. The trunk is copied to the device, the embedding part is left mapped, the vision
   part is mapped; the bank is created, which is where the tier file is fallocated.
3. The port binds, and `/v1/models` answers. Boot prewarms the first-use module loads
   (1.3 s once, on the author's box).
4. A first request pays the drafter's warm-up: a 64-token prefill was 0.36 s on the
   bench path, and a ~60-token chat prompt took 0.59-0.61 s to its first token through
   the server.

The launcher's gates, all of them settable from the environment:

| Gate | Default | Meaning |
|---|---:|---|
| `MINIMAX31_START_FLOOR_MB` | 105000 | refuse to start unless `MemAvailable` is at least this; the card's 1M plan is 113.95 GiB |
| `MINIMAX31_RUN_FLOOR_MB` | 4000 | reported after loading, for the record |
| `MINIMAX31_KILL_MB` | 2048 | two consecutive 5 s samples below this stop the server. 2 GiB is the author's own guard's trip figure (`tools/host_memory_guard.py --max-gib 117 --high-gib 116 --reserve-gib 2 --trip-gib 2`), and his hardware doc's rule for this box is to "run model loads under the memory guard, one resident model at a time" - the launcher's single-owner probe is that second rule |
| `MINIMAX31_WATCHDOG` | 1 | set 0 to run without the watchdog |
| `MINIMAX31_ALLOW_SHARED` | 0 | set 1 to overrule the single-owner probe |

Two gates rather than one because they answer different questions: the start floor
catches a box that cannot hold the plan, and the watchdog catches the case the plan
did not predict — at 1M with the drafter the author measured 114.75 GiB peak, which
they describe as letting systemd-oomd kill other user scopes on a 128 GB host.

## What verify checks, and the one manifest entry that is stale upstream

`verify` hashes the release's own `SHA256SUMS`, but in two tiers, because the manifest
the author published is not maintained for every file it lists.

Strict is everything the engine opens — the 61 parts and the six files it reads beside
them (`release.json`, `files.json`, `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `chat_template.jinja`). A mismatch there fails the run: a part
that does not match is not this release's.

Advisory is the documentation the release ships: the model card, `RECIPE.md`,
`FORMAT.md`, `SOURCE.md`, the licences and the conversion metadata. The engine never
reads them, and the author is free to edit them after generating the manifest, so a
mismatch is reported with both digests and does not fail the run.

That distinction has one live case, and the first Spark run hit it: **`README.md` is
stale in the release's `SHA256SUMS`.** The manifest lists 21,014 bytes / `e2f68289…`
while the file at the pinned revision is 24,071 bytes / `2493b23c…`, because the
author updated his model card twice on 2026-10-24, after writing the manifest. Nothing
is corrupt and re-downloading changes nothing: `hf download` compares a file against
the revision it is fetching, which is what you already have. The payload — every part
and every file the engine opens — matches. Before this was understood, the failure
read like a broken fetch:

```
minimax31: hashing 107.64 GiB; this reads every byte
  README.md: FAILED
minimax31: error: 1 file(s) do not match SHA256SUMS; a mismatching part is not this release's
```

## Quick start

```
# On the Spark, with the engine checked out and built:
cd ~/ds4-dfm-rs && git fetch origin && git checkout origin/feat/add-minimax-vq && make cuda-spark

cd ~/dgx-recipe
bash scripts/install-minimax-m31-flash-vq.sh preflight
bash scripts/install-minimax-m31-flash-vq.sh download     # 115.53 GB, resumable
bash scripts/install-minimax-m31-flash-vq.sh verify       # SHA256SUMS over all 115.53 GB
bash scripts/install-minimax-m31-flash-vq.sh check        # the engine's own open, and its plan quote
bash scripts/install-minimax-m31-flash-vq.sh start --daemon
bash scripts/install-minimax-m31-flash-vq.sh test
```

The launcher defaults: release in `/mnt/models/MiniMax-M3.1-Flash-VQ-GGUF`, the K/V
tier in `$HOME/models/minimax-m31-vq/kv-offload`, logs and pid files in
`$HOME/models/minimax-m31-vq`, port 8011, the model's native 1,048,576-token context,
the serial lane, the drafter with three proposals. Each is an environment variable with the `MINIMAX31_`
prefix and a `--` flag of the same name.

Two more things his hardware doc says about this exact box, both already enforced here: the
guard's trip figure is the launcher's `MINIMAX31_KILL_MB`, and "one resident model at a
time" is the single-owner probe `start` runs. Three contexts worth knowing, because the
tier's size follows the one you pick:

| Served context | K/V tier file | Resident index keys | How |
|---|---:|---:|---|
| **524,288 (default)** | **16.88 GiB** | **2.11 GiB** | `bash scripts/install-minimax-m31-flash-vq.sh start` - the author's configuration for a 128 GB host, and his card protocol's own allocation |
| 131,072 | none | 0.53 GiB | `MINIMAX31_CTX=131072 ...` - at or below the engine's resident threshold, so no tier file exists and no decode step reads K/V from disk. A served context he does not prescribe: what his guide gives here is the threshold itself, "K/V pages stay resident up to 131,072 tokens", not advice to set it |
| 1,048,576 (the model's own) | 33.75 GiB | 4.22 GiB | `MINIMAX31_CTX= ...` - his own table calls the DSpark run at this context "not a safe configuration on a 128 GB host" |

**Why 524,288 is the default.** His guide names exactly two configurations that fit a
128 GB host, in the sentence about the K/V pool: "The default stays small because the 512K
DSpark and the 1M configurations leave a 128 GB host only a few GiB." His performance
table then rules the second out where it matters: the 1M-DSpark run peaked at 114.75 GiB
and "let systemd-oomd kill other user scopes: not a safe configuration on a 128 GB host".
Linux sees 121.6 GiB of that box's 128 GB (`docs/gb10-hardware.md`), which is the box this
recipe runs on. So the first of the two is the default, and it is also the shape every
measurement of his we quote was taken in: the card protocol is
`ds4-bench --ctx-alloc 524288 --mtp-draft 3`.

In his numbers decode is flat from the short frontiers through 64K and falls past 128K of
live context. From the 2026-10-25 optimization round (build `39517dc2`, one GB10, 2,048-row segments, DSpark draft 3 with the gate, a
524,288-token allocation):

| Frontier | Decode tok/s | Prefill tok/s |
|---:|---:|---:|
| 2K / 4K / 6K | 24.07 / 24.87 / 27.80 | 820 / 1,036 / 1,032 |
| 8K / 10K / 12K | 19.55 / 30.05 / 19.03 | 1,024 / 1,024 / 1,010 |
| 16K / 24K / 32K | 21.75 / 20.09 / 20.00 | 996 / 1,000 / 976 |
| mean 2K-32K | **21.83** | ~1,000 |
| 64K | 19.46 | 941 |

His stated targets for that round were at least 1,000 prefill and 22 decode at every
frontier, decode first. Two things follow for a deployment here. The per-frontier spread
(19-30 tok/s) is the drafter's acceptance, not the context: a trial costs about 109 ms
against a 49.5 ms gated plain step at that build, so drafting pays only above about 2.2
kept tokens, and the gate does the switching. And he rejected the one change that would
have sped up the tier — an SSD pool sized for every 64K block, 9,660 to 32,700 slots and
+1.6 GiB, took 64K plain decode from 18.6 to 20.2-20.4 but got his 512K run killed by
systemd-oomd, so it was reverted. Do not raise `DS4_MINIMAX_KV_POOL_SLOTS` on a box that
is also running other things.

Above 128K the tier takes over and the flat band ends: the guide's plain figures are 20.2
tok/s at 8K and 19.0 at 124K against 14.7 at 256K, and the card's exact 1M frontier was
5.5 plain and 10.6 with the gated drafter.

The tier is `(context / 128) * 60 layers * 73,728 bytes`, reserved in full with
`fallocate` before the port binds, so it is the number to check against the free space
on whatever `DS4_KV_OFFLOAD_DIR` names. `selftest` checks that arithmetic against the
two published figures.

`preflight` and `plan` need no artifact and no GPU; `selftest` needs neither. A quick
loop that exercises the script without the Spark:

```
bash scripts/install-minimax-m31-flash-vq.sh selftest
MINIMAX31_RELEASE_DIR=/some/sparse/fixture bash scripts/install-minimax-m31-flash-vq.sh plan
```

## The open-grok client entry

The model is wired into open-grok as `minimax-m31-flash-vq`, pointing at
`http://192.168.1.91:8011/v1` with `api_key = "dummy"` (the ds4 host takes no auth),
`context_window = 524288` (which must match what the server serves: changing
`MINIMAX31_CTX` on the Spark means editing this entry too), `supports_images`, `supports_tool_result_images` and
`supports_reasoning_effort` true, and effort `high`. The entry lives in
`~/.opengrok/config.toml` and appears in the model list after an open-grok restart.

`max_completion_tokens` is deliberately **unset** in that entry, which is unusual for
the DS4 entries here. The host reports the field per launch in `/v1/models` as
`min(its own server default, the context)` (`append_model_json_values`,
`ds4_server.c:21561`), so it is a property of how the server was started, not of the
artifact: a number copied from the GLM entry would either cap long answers or be
rejected by a server started with a smaller default. Read it from the running server
and set it:

```
curl -fsS http://127.0.0.1:8011/v1/models | python3 -m json.tool | grep -i max_completion
```

Nothing answers on that port yet: the artifact is not on the Spark.

## Numbers that belong to another box

Everything in this section was measured by the artifact's author on one DGX Spark
(GB10, `sm_121a`, CUDA 13.4, driver 615.71.10) with `ds4-dfm-rs ff430197`, model
`74e5a9fd`, on 2026-10-24, three fresh `ds4-bench` processes into a 524,288-token
allocation, so the K/V pages sat in the SSD tier with the index keys resident, and
4,096-row prefill chunks.

| Configuration | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| DSpark draft 3, 2K-64K, gated | 905.1 | 19.56 |
| DSpark draft 3, 2K-32K, gated | 924.9 | 19.51 |
| the same sweep, ungated (dashed line) | 902.7 | 18.65 |

Per frontier, decode spans 16.4-22.8 tok/s as the drafter's proposals are accepted or
rejected; prefill is flat at 866-969 tok/s because each frontier is one 2,048-row
forward on a cached prefix. Cold single prompts prefill at 1,126 tok/s (8K), 999
(124K) and 884 (256K), and an exact 1,048,319-token prompt prefilled at 580 tok/s in
1,807 s and then decoded at 5.5 tok/s plain or 10.6 tok/s with the gated drafter — at
115.1 GiB peak, which the card calls unsafe.

Fidelity, teacher-forced against the original checkpoint on held-out chats (27 text
cases and 4 image/video cases, 60,431 scored positions, H100, BF16 attention):

| Routed experts | KL, assistant turns | Same top-1 |
|---|---:|---:|
| 13-bit on all 57 layers | 0.223 | — |
| R13+8 on the bottom 20 MoE layers | 0.171 | 87.2% |
| **R13+8 on the measured 20 layers (this release)** | **0.150** | **88.2%** |
| the original against itself, batched differently | 0.003 | — |

The NVFP4 K/V contract costs a further KL 0.0205 on assistant turns when measured on
the original weights, and the resident index keys alone cost 0.0002. None of that is
a served measurement: the runtime's own logits were not scored here.

## Limits

- **Lossy by design** at about 2.0 bits per routed-expert weight. On assistant turns
  the release agrees with the original's next token 88.2% of the time; long free
  generation, tool use and 1M-token retrieval were not measured by the author, and
  the card warns about reasoning loops and broken characters.
- **The 1M + drafter shape is the author's own warning**, not a round number: 114.75
  GiB peak, systemd-oomd killing other user scopes. The no-drafter 1M run peaked at
  110.7 GiB. Run this artifact alone on the box.
- **Long prompts pay for the tier.** A 2,048-row prefill chunk selects most of the
  visible blocks, so the prefix's K/V is streamed back per chunk; that is why the 1M
  prefill took 30 minutes.
- **The engine support is settled, the measurements are not.** `minimax_m3_vq` has been
  on `main` since PR #110 merged on 2026-10-25, but nothing here has been reproduced on
  the reference Spark: no engine quote, no decode rate, no time to first token.
- **One ds4 model at a time**, whatever the port says.
- **The vision and embedding parts sit on the roomy volume** because the engine reads
  all 61 parts from one directory; that is a layout consequence, not a measurement of
  how much it costs.

## Checks

What the installer can prove without a Spark:

| Check | Result |
|---|---|
| `bash -n scripts/install-minimax-m31-flash-vq.sh` | syntax clean |
| `selftest` | the script's own contract: 61 part names whose expected sizes sum to 115,526,325,376, the 7/30/20 R13 split, the resident and read-while-serving sums, and the drafter's proposal range. Falsifiable: `MINIMAX31_DRAFT=1 ... selftest` exits 1 naming the accepted range. |
| the expected sizes against `release.json` | 61 parts, 0 mismatches, and the twenty R13+8 layers identical to the release's own list |
| the fetch list against the revision's listing | all 20 named files exist; all 61 GGUFs are at the repository root, so `--include "*.gguf"` takes exactly them (81 of 227 files) |
| `preflight` on a non-GB10 machine | reports the arch, GPU and driver mismatches and exits non-zero |
| `preflight` with the release partly fetched | names how many of the 61 parts are on disk, a stray `.gguf`, a part at the wrong size, and how much is still to fetch |
| `verify` against a fixture whose `SHA256SUMS` matches its contents | 67 strict entries match, the size/mtime record is written, and `status` then reports it as matching |
| `verify` with a stale documentation digest | warns, names the file and both digests, and exits 0 with the record written - the payload is what decides |
| `verify` with a part corrupted at the same size | names the part and exits 1 without writing a record |
| `start` with a stale or absent record | refuses, naming the command that fixes it |
| `plan` | prints the exact launch line, the two directories and the disk need, without touching the network |

What only the Spark can prove, in the order that settles it: `check` (the engine's own
open of all 61 parts and its memory quote), `start` (the tier file's fallocation, the
port, the resident figure against the card's 113.95 GiB), `test` (the smoke request's
time to first token), and a decode rate against the 19.56 tok/s above.

## Troubleshooting

- `error: memory quote cannot host the mix at one bank (quote_overflow)`, then
  `ds4-server-rs: opened serial plan rejected`: the engine's own memory quote refused the
  shape. Note where it appears in the log — after the model has opened and the device cache
  has been prepared (the reference Spark shows `CUDA startup model cache prepared 103.68 GiB`
  about 110 s in), because the quote needs the host facts that the open produces. A rejected
  start therefore costs the 115 GB hash plus the load, and the flags are what change.

```
family: minimax_m3_vq
requested: reuse=auto max_seqs=off mtp=on ctx=1048576 floor=4G disk=false
effective: reuse=exact max_seqs=1 mtp=on weights=true ctx=1048576 floor=4G disk=false
error: memory quote cannot host the mix at one bank (quote_overflow)
ds4-server-rs: opened serial plan rejected
```

  This is the card's own warning made mechanical: at 1M with the drafter the run peaked at
  114.75 GiB and let systemd-oomd kill other user scopes, against 110.7 GiB without the
  drafter. The launcher's `start` prints this list when it sees that error line. The first
  two are settings the artifact and the engine support rather than served contexts he
  prescribes — he prescribes none — and the last two are his own words and switches:

  1. `MINIMAX31_CTX=131072` — larger than nothing, and no tier file at all: at or below the
     engine's own resident threshold (`DS4_MINIMAX_KV_RESIDENT_CTX`, default 131072) every
     K/V page stays in device memory and the index keys fall from 4.22 GiB to 0.53.
  2. `MINIMAX31_CTX=524288` — the default, and his card protocol's own allocation.
  3. `MINIMAX31_MTP_MODE=off` — the drafter's weights and rings are the difference between
     the two peaks above.
  4. `DS4_MINIMAX_KV_RESIDENT_CTX=0` — keeps K/V out of device memory entirely (his switch;
     the SSD reads are what you pay instead).

  The quote is computed from the host's own facts at open, so a box with other residents
  refuses shapes a quiet box accepts: check `free -g` and stop other models before blaming
  the flags. `--check-config` resolves the plan without opening the device, which is why a
  passing `check` does not promise that `start` fits.

- `README.md: FAILED` (or another documentation file) out of `verify`: the release's
  manifest is stale for that file, not your fetch. The full report names both digests
  and continues; if your build of this script predates the two-tier check, the same
  facts can be seen by hashing the part list alone:
  `cd <release dir> && grep -E ' -(0[3-9]|[1-5][0-9])-experts\.gguf$' SHA256SUMS | sha256sum -c -`.
  A part reported as FAILED is a different matter: re-run `download`, which re-fetches
  by file.
- `unexpected GGUF part` or a file-name complaint at open: the release directory holds
  a `.gguf` that `release.json` does not declare. The engine refuses the whole
  directory rather than ignore the stray file; delete it or re-fetch.
- `release format/recipe/source revision` at open: the directory is a different
  artifact, or an incomplete fetch whose `release.json` is not this release's. Check
  `sha256sum release.json` against `2110c1e5cba8d59987e370d50dab6dde5d5f7fc82009cdc27d8131452536f4b0`.
- `identify failed` / `unsupported tensor type`: the engine has no `minimax_m3_vq`
  family. Rebuild from `origin/feat/add-minimax-vq`.
- The port never answers, memory is low: the plan is 113.95 GiB; another ds4 model or
  another large process is resident. `status` prints the available memory and any
  other owner.
- The K/V file fails to be created: the tier's filesystem does not support
  `O_TMPFILE`, or it is full. The size reserved at 1M is 36.2 GB; point
  `DS4_KV_OFFLOAD_DIR` at a local filesystem with room, or lower the context to
  131,072 so no tier file is created at all.
- Decode is slower than the card's 19.56 tok/s at a long context: read the tier's own
  counters with `DS4_MINIMAX_KV_TRACE=1`, then raise `DS4_MINIMAX_KV_POOL_SLOTS` if
  memory allows — the author measured 14.7 -> 16.7 tok/s at 256K that way.
- Image or video input is refused: the release's vision part is mapped at open, so a
  refusal means the part is missing from the directory or the request exceeded four
  media.
- An answer that loops or breaks mid-character is the quantization's known behaviour,
  not a serving fault. Post the prompt, the sampling settings and the engine commit in
  the artifact's Community tab, which is what its card asks for.

## Attribution

The conversion, the GVQ1 container, the GB10 kernels and the `minimax_m3_vq` family are
Baekpica's; the 8-dim shared-codebook VQ method, the F16 row gains and the bit planes
are YoungAi's (github.com/fodelf/YoungAi); MiniMax-M3.1-Flash, its DSpark draft head
and its vision tower are MiniMax's, under the MiniMax Community License. This repository
contributes the operational recipe: where the bytes go, what the launch shape needs, and
which numbers were measured by whom.
