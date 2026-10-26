# Recipe: MiniMax-M3.1-Flash VQ on the DS4 CUDA engine

MiniMax's 428B native multimodal MoE, quantized to about 2.0 bits per routed-expert
weight, served on one DGX Spark at the model's own 1,048,576-token context on port
8011. 61 GGUF parts, 115,526,325,376 bytes, in a GVQ1 release directory read by the
ds4-dfm-rs Rust host. Text, images and video, the DSpark draft head and the native
context all come with it.

**Nothing in this recipe has been loaded on the reference Spark.** The engine
support is a draft pull request that is not merged, so the artifact is not on the
box, no `--check-config` quote has been taken, and every speed and quality figure
below was measured by the artifact's author on their own Spark and is labelled with
the build it came from. What the first start prints belongs in this file once it has
been run.

- Contract: [`recipes/minimax-m31-flash-vq.yaml`](../recipes/minimax-m31-flash-vq.yaml)
- Installer: [`scripts/install-minimax-m31-flash-vq.sh`](../scripts/install-minimax-m31-flash-vq.sh)
- Artifact: [`Baekpica/MiniMax-M3.1-Flash-VQ-GGUF`](https://huggingface.co/Baekpica/MiniMax-M3.1-Flash-VQ-GGUF) at `2789cbaf`, weights unchanged since `74e5a9fd`
- Engine: [`Baekpica/ds4-dfm-rs`](https://github.com/Baekpica/ds4-dfm-rs), family `minimax_m3_vq`, [draft PR #110](https://github.com/Baekpica/ds4-dfm-rs/pull/110)
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
      --mtp-mode on --mtp-draft 3 --max-seqs 1 \
      --host 0.0.0.0 --port 8011 --model-id minimax-m31-flash-vq
```

- `--mtp-mode on --mtp-draft 3`: the DSpark part joins the trunk only when the draft
  count is above 1, and the family's `auto` resolves to off, so `on` is not optional
  if the drafter is wanted. Three proposals is the card's own sweep setting and the
  engine's useful range on this hardware: a four-row verify costs about 2.4 decodes,
  so drafting only pays while a trial keeps more than about 2.6 tokens. The engine
  measures that itself and decodes plainly while drafting runs slower — the rate gate
  is the difference between 18.65 and 19.56 tok/s in the author's sweep, and greedy
  output is identical with or without it.
- `--max-seqs 1`, explicitly: the card's plan for the native context is 113.95 GiB of
  the 117.2 GiB the engine sees on GB10, which is one bank. The number is written down
  rather than left to the automatic width so the file, not the fit, decides.
- No `-c`: this family, like the GLM VQ family, follows the model's native maximum
  unless `--ctx` is given. Passing `-c 131072` is the interesting alternative: it keeps
  every K/V page resident (4.5 GB per bank) and needs no tier file at all.
- `--model-id`: the served name. The family's syntax comes from the loaded model, not
  from this string.
- `--mem-floor-gb` is left at the engine's default of 4 GiB; the launcher's own gate
  and watchdog are described below.

## The K/V tiers, and what the indexer buys

MSA keeps every position on every layer: 72 bytes of index key plus 576 bytes of K/V
per token and layer, 38,880 bytes per token, 37.97 GiB at 1M. Selection saves reads,
not storage. The indexer needs only the index keys to decide what attention reads, so:

- **index keys stay resident**, in NVFP4, 4.22 GiB at 1M, in every tier;
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
| `MINIMAX31_KILL_MB` | 2500 | two consecutive 5 s samples below this stop the server |
| `MINIMAX31_WATCHDOG` | 1 | set 0 to run without the watchdog |
| `MINIMAX31_ALLOW_SHARED` | 0 | set 1 to overrule the single-owner probe |

Two gates rather than one because they answer different questions: the start floor
catches a box that cannot hold the plan, and the watchdog catches the case the plan
did not predict — at 1M with the drafter the author measured 114.75 GiB peak, which
they describe as letting systemd-oomd kill other user scopes on a 128 GB host.

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
`$HOME/models/minimax-m31-vq`, port 8011, the native 1M context, one bank, the drafter
with three proposals. Each is an environment variable with the `MINIMAX31_` prefix and
a `--` flag of the same name.

`preflight` and `plan` need no artifact and no GPU; `selftest` needs neither. A quick
loop that exercises the script without the Spark:

```
bash scripts/install-minimax-m31-flash-vq.sh selftest
MINIMAX31_RELEASE_DIR=/some/sparse/fixture bash scripts/install-minimax-m31-flash-vq.sh plan
```

## The open-grok client entry

The model is wired into open-grok as `minimax-m31-flash-vq`, pointing at
`http://192.168.1.91:8011/v1` with `api_key = "dummy"` (the ds4 host takes no auth),
`context_window = 1048576`, `supports_images`, `supports_tool_result_images` and
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
- **The engine is a draft PR.** Its family, its flags and its numbers can still move,
  and nothing here has been reproduced on the reference Spark.
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
| `verify` against a fixture whose `SHA256SUMS` matches its contents | 67 digest lines match, the size/mtime record is written, and `status` then reports it as matching |
| `verify` with one digest corrupted | names the file and exits 1 without writing a record |
| `start` with a stale or absent record | refuses, naming the command that fixes it |
| `plan` | prints the exact launch line, the two directories and the disk need, without touching the network |

What only the Spark can prove, in the order that settles it: `check` (the engine's own
open of all 61 parts and its memory quote), `start` (the tier file's fallocation, the
port, the resident figure against the card's 113.95 GiB), `test` (the smoke request's
time to first token), and a decode rate against the 19.56 tok/s above.

## Troubleshooting

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
