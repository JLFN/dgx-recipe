# dgx-recipe

Custom inference recipes for the NVIDIA DGX Spark (GB10, SM121, aarch64, 121.6 GiB
unified memory). One recipe per directory entry, each one carrying the exact
artifact list, the launcher, and the measured evidence behind its settings.

Seven artifacts are covered, each with a contract, a runbook and a launcher:

| Artifact | Size | Context | Port | State |
|---|---|---:|---:|---|
| Qwen3.8-Flash-Next Uncensored Q5 SSD-PLE + FP8 sidecar | 134.48 GB | 196,608 (two banks) | 8003 | Verified here 2026-09-26 |
| Prism Ternary Bonsai 2 27B (`qwen35`) | 6.71 GiB | 262,144 (one serial lane) | 8005 | Verified here 2026-10-01 |
| YoungAi DeepSeek-V4.1-Flash | 113.6 GB + 203 GB tables | 1,048,576 | 8006 | Third-party artifact; committed here with its numbers unmeasured, and its recipe predates the 2026-10-17 volume split |
| GLM-5.3-Flash VQ, Q4_K nonexpert v3 | 71.96 GiB | 1,048,576 (two banks) | 8007 | Verified here 2026-10-17 |
| YoungAi DeepSeek-V4.1-Flash-1017 (VQ, five sidecars) | 317.8 GB | 1,048,576 | 8008 | Fetched and assembled on the box 2026-10-18; never loaded, nothing measured |
| GLM-5.3-Flash Uncensored VQ, R13+8 (Q4_K nonexpert v3) | 94.49 GB | 1,048,576 (one bank) | 8009 | The all-R13 revision was loaded and served here on 2026-10-17; the 19-layer revision needs a 17.2 GB refetch over it and nothing about it has been loaded |
| DeepSeek-V4.1-Flash-1017 UNCENSORED VQ | 317.6 GB | 1,048,576 | 8010 | Fetched and assembled on the Spark 2026-10-18; never loaded, nothing measured |

The path each launcher uses follows one rule, measured rather than assumed: the
base GGUF is read once at load and then stays resident, so it can sit on the roomy
953 GB volume, while everything read while the server runs — n-gram tables and PLE
sidecars on every token that needs them, the vision tower on the first image after
a cold start or an eviction — stays on the fastest device. On the reference Spark
that is `/mnt/models` for the base and NVMe for the rest, which is why
`ds41_install.sh` carries `--gguf-dir` and the Qwen launcher carries
`PLE_FP8_DIR`.

The GLM-5.3 VQ releases are the edge case worth naming, because they look like
they contradict it: decode touches all 72 GiB of weights on every token, and they
still belong on the roomy volume. The load stages the whole payload into the device
cache once — 71.96 GiB of tensor spans in 9.862 s on the box — and nothing reads
the file again, so the volume's speed stops mattering after the first minute. The
same reasoning does not extend to the n-gram tables and PLE sidecars above, which
are read in rows for the life of the server. Nothing in this repository uses the
engine's `--ssd-streaming` expert cache, and the GLM-5.3 VQ family cannot: that
path requires a single-shard, non-VQ artifact.

## What is in this repository

- `recipes/` - one contract per artifact: the artifact list with sizes and digests, the launch shape, the engine's memory quote, the client wiring, and an explicit list of what is and is not measured.
- `runbooks/` - the operator's account: why the non-obvious flags are on the command line, what a start actually looks like in the log, the limits, and troubleshooting.
- `scripts/` - one launcher or installer per artifact, plus `bank-budget.sh` for sizing banks and context from the engine's own quote and `selfcheck.sh` for the launch-path checks.
- `docs/` - the long-form usage guide for the largest recipe.
- `qa-evidence/` - the QA report for the unit that introduced the budget tool.

Ports are assigned one per artifact so no two launchers argue: 8000 the V4.1 port
deployment, 8003 Qwen3.8, 8005 Bonsai, 8006 YoungAi V4.1, 8007 GLM-5.3 VQ, 8008
YoungAi DeepSeek V4.1 Flash 1017, 8009 GLM-5.3 Uncensored VQ and 8010 DeepSeek
V4.1-Flash-1017 Uncensored VQ. Only one ds4 model can be resident at a time
whatever the port says, because the engine takes a single global lock.

The first two, verified end to end on a single Spark. Starting with the first,
verified on 2026-09-26:

**Qwen3.8-Flash-Next Uncensored (Q5 SSD-PLE) with the FP8 E4M3FN PLE sidecar**,
served by the `ds4-server` Rust host from the `Baekpica/ds4-dfm-rs` fork, at
196,608 tokens of context in the model card's two-bank shape, with a one-bank
262,144 mode, on port 8003.

- Recipe contract: [`recipes/qwen-38-uncensored-fp8-ple-ds4.yaml`](recipes/qwen-38-uncensored-fp8-ple-ds4.yaml)
- Full usage guide: [`docs/USAGE.md`](docs/USAGE.md) - install, shapes, log levels, the HTTP API, vision, benchmarking, memory, troubleshooting, and an honest list of what is verified and what is not
- Runbook: [`runbooks/qwen-38-uncensored-fp8-ple-ds4.md`](runbooks/qwen-38-uncensored-fp8-ple-ds4.md)
- Launcher: [`scripts/start-qwen38-uncensored-fp8.sh`](scripts/start-qwen38-uncensored-fp8.sh)

The second is **Prism Ternary Bonsai 2 27B (family `qwen35`)**, on the same
engine, native (no Docker), at the artifact's declared 262,144 context in the one
lane this family serves, on port 8005. It is the reason this repo now carries a
memory-budget tool: the model is 6.71 GiB against the Spark's 121.6 GiB, so the
question worth answering is what that memory actually buys.

- Recipe contract: [`recipes/prism-bonsai-2-27b-qwen35.yaml`](recipes/prism-bonsai-2-27b-qwen35.yaml)
- Runbook: [`runbooks/prism-bonsai-2-27b-qwen35.md`](runbooks/prism-bonsai-2-27b-qwen35.md)
- Launcher: [`scripts/start-bonsai-spark.sh`](scripts/start-bonsai-spark.sh)
- Budget report: [`scripts/bank-budget.sh`](scripts/bank-budget.sh) - how much room there is for banks, and how much context each bank can hold, read from the engine's own plan quote rather than estimated

## Verified configuration

| Item | Value |
|---|---|
| Model | `Baekpica/Qwen3.8-Flash-Next-Uncensored-Mixed-Quant-SSD-PLE-GGUF` |
| Main weights | `MQ-Q5-SSD-PLE-BF16/` three shards, 32.07 + 31.81 + 19.40 GB |
| PLE sidecar | `PLE-FP8/` four `ple-fp8-*.bin` of 12.80 GB, `ple-manifest.json`, 2-byte weight scale |
| Download total | 134,475,461,654 bytes (134.48 GB / 125.24 GiB) |
| Engine | [`Baekpica/ds4-dfm-rs`](https://github.com/Baekpica/ds4-dfm-rs) `main` @ `7a78fcd`, FP8 PLE support from PR #22 |
| Build | `make cuda-spark` (SM121; also links `ds4_weight_server`) |
| Runtime split | `ds4_weight_server` weight owner + `ds4-server` Rust worker (IPC weight manifest) |
| Context | 196,608 with two banks (default, the card's production shape); 262,144 with one bank |
| Banks | 2 by default; 1 in the 262k and 512k shapes |
| Port | 8003 |
| Sidecar logging | `dtype=FP8_E4M3FN` in the worker log |
| Disk KV | separate directory per sidecar format, 32,768 MB cap |

Measured on the Spark: `/v1/models` reports `context_length` 262144 with a
32,768 completion cap; a 47-token prompt returned its first token in 234.3 ms at
222.4 tok/s prefill.

## Shapes

The launcher ships three named shapes, chosen as the first argument to `start`,
so a change of context and bank count is one word:

| Shape | Context | Banks | MTP | Notes |
|---|---:|---:|---|---|
| `196k` | 196,608 | 2 | on | Default. The model card's canonical command and the shape behind every throughput figure it publishes. Two sequences can be in flight. |
| `262k` | 262,144 | 1 | on | The artifact's declared `qwen4exp.context_length`, verified working on this setup. Deeper prompts, but concurrent requests serialize. |
| `512k` | 524,288 | 1 | off | The card's reduced YaRN shape. Output capped at 256 tokens and reuse disabled. The card claims execution, not long-context quality. |

Individual knobs win over the shape, so `MAXTOK=8192 bash scripts/start-qwen38-uncensored-fp8.sh start 196k`
keeps two banks at 196,608 with an 8192-token cap. `bash scripts/start-qwen38-uncensored-fp8.sh shapes`
prints this table resolved from the launcher's own profile code.

## Why the default is two banks at 196,608

The card gives one canonical serving command and calls it the production server
shape: two banks, 196,608 context, `--mtp-draft 2`, 8,192-token prefill chunks and
a 512 MiB PLE cache with 16 workers. Everything it measures uses that shape, so it
is the default here.

Depth beyond it is possible but has to trade concurrency, and the ceiling the
artifact declares is not the ceiling this box will serve with two banks. The
engine repository's FP8 document reports that both FP8 Q5 servers passed short
HTTP checks at 262,144 with two banks, but on the reference Spark that same
two-bank request was refused with `banks_not_quoted`, with the weight owner
holding about 81 GiB resident and each bank's graph plan measuring 23.35 GiB at
262,144. Whether that is the commit, the owner reserve or the free-memory baseline
at the time is unresolved here, so this repo treats 262,144 as a one-bank shape
rather than claiming one bank is inherent to that context. The launcher retries
once at one bank if the quote refuses a plan, and says so when it does.

## Quick start

Run on the Spark, as the user that owns the model directory:

```bash
bash scripts/start-qwen38-uncensored-fp8.sh download   # 134.5 GB, resumable
bash scripts/start-qwen38-uncensored-fp8.sh verify     # published SHA256SUMS
bash scripts/start-qwen38-uncensored-fp8.sh build      # make cuda-spark
bash scripts/start-qwen38-uncensored-fp8.sh start      # two banks, 196,608, port 8003
```

Add the shape when you want something else: `start 262k` for 262,144 in one bank,
`start 512k` for the reduced 524,288 shape. `stop`, `status`, `logs`, `shapes`,
`test` and `plan` round it out, and every path, port, context and memory knob is an
environment override documented in the script header.

## What the memory buys, and what the engine refuses

`scripts/bank-budget.sh` answers the sizing question with the engine's own
numbers. It asks `ds4-server --check-config` for the resolved plan at several
contexts - a check that prints the plan as JSON and **opens no weights** - and
derives the per-bank line from two of those quotes, checking a third against it.
It reports the per-bank size, the plan total, the headroom against the live free
memory (or a ceiling you set), the largest context that fits, the largest number
of banks that fits, and which of those the family will actually admit:

```sh
SERVER=~/ds4-dfm-rs/ds4-server MODEL=~/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  CTX_CEILING=262144 bash scripts/bank-budget.sh
```

For Bonsai the answer has two halves. The line, measured on the reference box on
2026-10-01, is exact:

    per_bank(ctx) = 422,566,912 + 65,552 x ctx   bytes
    total(ctx)    = weights + banks x per_bank + floor

so the artifact's declared 262,144 ceiling costs a 16.397 GiB bank and a 24.109
GiB plan - about a fifth of the Spark. The other half is that the family serves
**one request at a time**: its caps declare `BankLane::Serial` with
`bank_support: Support::None`, and `--max-seqs 2` is refused with code
`banks_unsupported`. The Spark's memory is therefore not what limits this model,
and extra banks are not something the Spark can be talked into.

The tool is not Bonsai-specific: point `MODEL` at any artifact in the engine's
tree and it sizes that family instead. For the Qwen3.8 recipe above, whose caps
do admit banks, the same command reports how many fit at what context on this
box.

## YoungAi DeepSeek-V4.1-Flash, a third-party artifact

One 113.6 GB base assembled from 40 byte-range parts, a domain sidecar of about
40 MB, and the 203 GB of n-gram tables the engine reads off the device at run
time. Port 8006. Its launcher fetches the base parts, the sidecars and the
binaries the artifact ships, merges the base with each part checked as it lands,
and serves it under a memory watchdog.

- Recipe contract: [`recipes/youngai-deepseek-v4.1-flash-ds4.yaml`](recipes/youngai-deepseek-v4.1-flash-ds4.yaml)
- Runbook: [`runbooks/youngai-deepseek-v4.1-flash-ds4.md`](runbooks/youngai-deepseek-v4.1-flash-ds4.md)
- Launcher: [`scripts/start-youngai-v41-flash.sh`](scripts/start-youngai-v41-flash.sh)

It is the one entry here whose artifact and runtime come from a third party
rather than from this repository's own measurement work, and its recipe predates
the 2026-10-17 volume split that put the read-once payloads on the 953 GB volume
and everything read at run time on the fastest device. The numbers in its recipe
and runbook are the artifact's, not this repository's, and nothing in it has been
re-measured here.

## GLM-5.3-Flash VQ, measured here on 2026-10-17

A 72 GiB vector-quantized GLM-5.3-Flash: the routed experts of layers 3 to 44 are
8-dimensional vector quantization at roughly 1.6 bits per weight with a per-layer
E4M3 codebook, 316 nonexpert matrices are Q4_K, and the embedding, routers, norms
and the MTP layer keep source precision. Served at the model's native 1,048,576
context with two banks, on port 8007.

- Recipe contract: [`recipes/glm-53-flash-vq.yaml`](recipes/glm-53-flash-vq.yaml)
- Runbook: [`runbooks/glm-53-flash-vq.md`](runbooks/glm-53-flash-vq.md)
- Launcher: [`scripts/start-glm53-flash-vq.sh`](scripts/start-glm53-flash-vq.sh)

The release is a **directory**, not a file: 56 GGUF parts with the tables that
describe them beside them (`files.json`, `tensor-manifest.json`, `tokenizer.json`,
`tokenizer_config.json`, `chat_template.jinja`). `-m` takes the directory, and the
engine verifies every file, tensor and payload SHA256 during load, about 20
seconds for this artifact.

The engine's own quote at the default shape, read with `--check-config` on the
Spark on 2026-10-18 (engine `fix/glm53-vq-penalties`, `416d4f5a`): shared weights
77,272,028,416 B, per bank 13,787,683,840 B, mtp state 783,368,192 B, scratch
11,440,271,104 B, floor 4 GiB, two banks, total 121,366,002,688 B against
127,133,745,152 B available — 95 percent of what the engine sees, and
2,937,047,040 B more than the 2026-10-17 checkout quoted for the same shape.

Measured with 256-token greedy completions: 24.2 and 24.3 tok/s on one stream, and
12.1 / 12.0 tok/s each with two streams at once, the pair finishing in 22.67 s
against 22.65 s for the same two requests back to back. Two banks buy a second
client not waiting, not extra tokens per second. Decode is bytes per token against
the weight stream, and batching two rows does not halve those bytes.

Two findings worth carrying forward. The embedded predictor is armed either way —
the plan reports `mtp=on weights=true` and `DS4_CONT_MTP_MODE=2` /
`DS4_CONT_MTP_DEPTH=2` are confirmed in the running process's environment — but
what it is worth depends on the engine build. On the 2026-10-17 optimize4 checkout
every request reported `tok_per_step` 1.0; on the 2026-10-18 build under the
launcher, requests reported 2.46 and 2.61 at 38.4 tok/s on one stream and 18.6 /
20.4 with two, which is the artifact card's 34.6 to 40.6 tok/s band reached on the
served lane. And thinking can run away: on a bare prompt, `max` effort —
or an omitted field, which the template resolves to `max` — ran to the cap with no
answer at all, byte-identical at temperature 0 and 1.0, while `low` closed in 13
tokens. The model's author recommends `max`, and it behaves on real tasks; the
practical guard is a bounded completion and `/effort low` for trivial prompts.

## GLM-5.3-Flash Uncensored VQ, refetched to the 19-layer R13+8 revision

The abliterated GLM-5.3-Flash through the same conversion as the base VQ release, on
port 8009. The all-R13 revision of it was loaded and served here on 2026-10-17; the
artifact then moved twice on 2026-10-18, first adding an 8-bit residual stage to
routed layers 3-16 and then to layers 17, 22, 27, 41 and 44 as well. This repo
follows the newest of the three: 19 of the 42 routed layers carry the residual stage,
and the other 23 stay at R13. The payload went from 77,272,028,544 bytes (no stage)
to 89,960,211,968 (14 layers) to 94,491,712,768, and 19 of the 56 part files were
replaced in place upstream: layers 03-09 from 1,477,184,128 to 2,383,481,600 bytes,
and layers 10-16, 17, 22, 27, 41 and 44 from 1,477,184,896 to 2,383,485,056. The
other 37 parts — 23 R13 layers, 9 native, 4 MTP and the vision tower — are
byte-identical across all three revisions: 37 of the 56 digests match, checked
against each revision's own Hub listing.

The trap this repository already guarded against got a second half. The two release
markers (`chat_template.jinja` at 10,644 bytes and the MTP-02 part at 1,997,061,632)
are **identical across all three revisions**, so a half-refetched directory passes
them, and the engine — whose shape name is still the literal string
`GLM-5.3-Flash-VQ` — would serve stale layers without a word. The installer therefore
carries a revision fingerprint as well: it reads the 19 upgraded parts by size in
three groups (layers 03-09, layers 10-16, and the five the newest revision adds) and
reports the directory as `pinned`, `l3-16`, `r13`, `mixed`, `absent` or `unknown`,
naming every part that is stale or missing. Every action that touches weights refuses
anything but the revision that run asked for.

The residual stage is also a format change, not just a date: the 19 parts declare
GVQ1 container version 2 and carry `GVR8` residual tensors, and a build older than
PR #96 has no reader for them — its loader falls through to
`unsupported tensor type`. The floor is therefore `34c46d1`, and the check that
proves it is `--check-config`, which opens no weights.

Memory changed shape too. The card's quote for the 14-layer revision is 112.0 GiB
(weights 83.8, bank 12.8, scratch 10.7, MTP state 0.7, floor 4) of the 118.7 GiB the
engine sees on GB10, and this revision's five extra layers weigh 4.2 GiB more: about
116.2 GiB, and about 2.5 GiB of headroom, so **one** 1M bank fits where the all-R13
revision fitted two. The installer passes `--max-seqs 1` at the native context
explicitly, and its post-load and watchdog memory floors sit below that headroom
rather than at the 4 GB that suited the wider 14-layer margin.

The refetch is in place, by operator decision: one directory
(`/mnt/models/GLM-5.3-Flash-Uncensored-VQ-GGUF`) and one endpoint
(`glm-5.3-flash-uncensored-vq` on 8009), so an open-grok entry written for an earlier
revision keeps working. It costs 17.2 GB over the all-R13 revision and 4.5 GB over
the 14-layer one, rather than 94.5 GB, because the 37 identical parts are not
re-read. Both earlier revisions stay available upstream, at the tags `q4k-v3` and
`r13p8-l3-16`, and the installer can fetch either into a directory of its own. The
all-R13 installer could not tell any of the three apart, and it has been retired.

Nothing about this revision has been loaded or timed here. The operator's download
run on 2026-10-18 answered `fetching the whole release`, which is what the installer
prints when it finds none of the 19 upgraded parts in that directory: what the Spark
holds at that path right now is an open question, not an assumption. The speed and
quality figures in the contract are the artifact author's, measured on the **base**
release: 1,113.7 prefill / 31.81 decode tok/s at 14 layers against R13's
1,175.4 / 35.00, and — run on this release's own parts — PPL 3.780 / KL 0.282 at 19
layers against 3.927 / 0.331 at 14.

- Recipe contract: [`recipes/glm-53-flash-uncensored-vq.yaml`](recipes/glm-53-flash-uncensored-vq.yaml)
- Runbook: [`runbooks/glm-53-flash-uncensored-vq.md`](runbooks/glm-53-flash-uncensored-vq.md)
- Installer: [`scripts/install-glm53-uncensored-vq-r13p8.sh`](scripts/install-glm53-uncensored-vq-r13p8.sh)
- Earlier revisions: [`r13p8-l3-16`](https://huggingface.co/Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF/tree/r13p8-l3-16) and [`q4k-v3`](https://huggingface.co/Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF/tree/q4k-v3)

## YoungAi DeepSeek-V4.1-Flash-1017, the corrected packaging

DeepSeek's October 17 V4.1 Flash checkpoint through YoungAi's
`vq8sh14-q4k-mtpnative` recipe, repackaged with an explicitly dated base, a dated
vision sidecar and five 1017-specific domain sidecars. One 113.56 GB base GGUF
assembled from 40 parts, a 971 MB vision sidecar, five sidecars of about 42 MB
each, and the two official 1017 Engram tables (2 x 101.5 GB) fetched separately
from DeepSeek and read in rows from NVMe. Port 8008.

- Recipe contract: [`recipes/youngai-deepseek-v41-flash-1017.yaml`](recipes/youngai-deepseek-v41-flash-1017.yaml)
- Runbook: [`runbooks/youngai-deepseek-v41-flash-1017.md`](runbooks/youngai-deepseek-v41-flash-1017.md)
- Launcher: [`scripts/ds41flash_1017_install.sh`](scripts/ds41flash_1017_install.sh)

The layout follows the same rule as everything else here: the base is read once
at load and stays resident, so it goes on the roomy `/mnt/models`; the five
sidecars, the vision tower and the Engram tables are read while the server runs,
so they go under `/home/leandro/models` on NVMe. The payload is 317.8 GB, of
which 203.1 GB is the Engram tables.

This is a different base from the Baekpica packaging below, not a rename: both
are 113,556,639,424 bytes and their digests differ (`0da4ae2d...` against
`9f3afe6a...`). The header read from the first 16 MB of part01 with the engine's
own `ds41_inspect` reports 73 KV entries, 1000 tensors, the native 1,048,576
context, three MTP towers with targets 37, 38 and 39, and Engram layers 1 and 14.
Its internal `general.name` is still "DeepSeek V4.1 Flash"; the date is in the
repository and file names. The five sidecar directories were checked against a
SHA-256 aggregate computed from the repository's LFS metadata, and the finance
directory was downloaded in full here to verify the formula end to end. The
launcher has since fetched and assembled the artifact on the box itself
(2026-10-18), and those files match the published sizes over the sftp mount.
Nothing has been loaded or measured on the Spark.

## The superseded Baekpica 1017 packaging, fetched on the box, never loaded

DeepSeek's October 17 V4.1 Flash checkpoint through YoungAi's
`vq8sh14-q4k-mtpnative` recipe: a 113.56 GB base assembled from 40 byte-range
parts that carries the model's own three-tower DSpark drafter, a 971 MB vision
sidecar, and the 1017 Engram tables (2 x 101.5 GB) read in rows from disk. Port
8008.

Superseded on 2026-10-18 by the corrected packaging above. Its base has a
different digest at the same byte size, it ships no domain sidecars, and the five
sidecars in the new packaging were solved against the new base, so the two must
not be mixed. The section remains because those files were fetched and assembled
on the box on 2026-10-17 and the older recipe stays reproducible.

- Recipe contract: [`recipes/ds41-1017-vq.yaml`](recipes/ds41-1017-vq.yaml)
- Runbook: [`runbooks/ds41-1017-vq.md`](runbooks/ds41-1017-vq.md)
- Launcher: [`scripts/start-ds41-1017.sh`](scripts/start-ds41-1017.sh)

Three silent failures the launcher exists to prevent, all of them name-and-size
identical between V4.1 and 1017: the Engram tables, the vision sidecar
(532,623,100 of its 971,066,016 bytes differ), and YoungAI's V4.1 domain sidecars,
which do not fit these weights at all. `verify` hashes both Engram files and the
sidecar against the digests the card publishes, rather than trusting names.

Nothing here has been measured, and now that the artifact is on the box that
distinction is worth stating precisely: the fetch and the assembly have run on the
reference Spark on 2026-10-17, and they produced exactly the published files -
base GGUF 113,556,639,424 bytes from the 40 parts, vision sidecar 971,066,016, and
the two Engram shards at 101,535,150,936 and 101,537,926,640 bytes. No load, no
request, no speed and no quality figure exists: the numbers in its recipe and
runbook are V4.1 Flash's, measured with the same recipe, and the card says plainly
that the artifact is lossy at roughly 1.6 bits per routed-expert weight and that no
1017 speed or fidelity figure exists anywhere yet. The base is read once at load
and then resident, so it goes on the roomy volume; the tables are read in rows, so
they belong on the fastest device, and on the box they are in
`~/models/ds41-1017-engram` while the base sits under `/mnt/models`.

## DeepSeek-V4.1-Flash-1017 UNCENSORED VQ, published 2026-10-18, measured nowhere

dealignai's uncensored DeepSeek-V4.1-Flash-1017 through YoungAi's published
`vq8sh14-q4k-mtpnative` recipe — the same conversion that built the Baekpica 1017
packaging above, with the input swapped for the uncensored FP8 checkpoint. Port
8010. One 113.56 GB base assembled from 40 byte-range parts that carries the
model's three DSpark draft towers, a 971 MB vision sidecar, and the two 1017 Engram
tables, which this repository ships inside the artifact rather than fetching from
DeepSeek. No domain sidecar ships at all.

It is four files that answer to names shared with other packagings, and every one
of them is a silent mismatch if it is the wrong one: the Engram tables (identical
names and sizes to the older V4.1 tables, different bytes), the vision sidecar
(same size, 532,623,100 bytes different), and the base (same 113,556,639,424-byte
size as the other two 1017 packagings, digests `9f3afe6a…` and `0da4ae2d…` against
this release's `2680747e…`). The installer checks each of the four, names the wrong
digest when it sees one instead of reporting "unknown", and verifies each part
against the release's `SHA256SUMS` as it is appended to the base.

Two design points are worth naming, because both were found while building it.
`sha256sum -c --ignore-missing` exits 0 on a directory where none of the listed
files exist, so `verify` checks explicitly and counts instead — a vacuous pass is
not possible. And the digests are read once, by `verify`, which writes a
verification record keyed on the size and mtime of the base, the sidecar and both
tables; `start` refuses to serve unless that record still matches what is on disk,
so the expensive read happens once per fetch instead of on every `status`.

The vision sidecar is reused the same way when this release's copy already sits
under another state directory — it is name-and-size identical to the older V4.1
file, so candidates are hashed and only a `3a8b194a…` copy is adopted. The base is
assembled on the roomy volume by the card's own loop — append a part,
delete it as it lands — so the peak extra space is one part rather than a second
copy of the base. The Engram tables are reused from whatever 1017 copy is already
on the box when one of the published sizes is found, which is the artifact author's
own advice; otherwise they cost a 203 GB fetch on the fast volume. Both the search
and the fetch account for the release's one subdirectory payload: the repository
keeps the two tables in `deepseek-engram/`, and the Hugging Face CLI preserves that
path under `--local-dir`, so a fetch writes `<dir>/deepseek-engram/<file>` while the
parts and the sidecar (at the repository root) land where the script looks for
them. `download` and `verify` move such a fetch up with a rename rather than
re-fetching 203 GB, and the reuse search checks each candidate one level deeper
too. The first Spark run fetched the tables that way and then refused them on size;
the runbook records the measurement and the repair.

The artifact was fetched, assembled and verified on the Spark on 2026-10-18: all 40
parts passed their per-part `SHA256SUMS` digests as they were appended, the assembled
base measured 113,556,639,424 bytes on `/mnt/models/ds41-1017-unc`, and a copy of
each table at the published sizes sits in `~/models/ds41-1017/deepseek-engram`
together with the vision sidecar, which the search proved by digest before adopting
it. `verify` printed all four identity proofs — base `2680747e…`, vision
`3a8b194a…`, tables `76a05106…` and `9dc51433…` — and then failed to write its
record, because the state directory did not exist and only `download` created it
(fixed in `e37c3a4`). Nothing has been loaded, so the artifact is still measured
nowhere — the card says serving on DGX Spark has not been measured. Its reference
numbers are V4.1 Flash's with the same recipe (1,039.65 prefill / 38.17 decode
tok/s over 2K-64K with embedded MTP) and dealignai's FP8 checkpoint's (MMLU 83.77
against 87.63 for the model it edits, HarmBench-320 at 320 of 320). The card's own
warning matters more than those: quantizing an FP8 bundle keeps the quantization
error on top of the uncensoring, and compliance can regress at low bit widths.

- Recipe contract: [`recipes/ds41-1017-uncensored-vq.yaml`](recipes/ds41-1017-uncensored-vq.yaml)
- Runbook: [`runbooks/ds41-1017-uncensored-vq.md`](runbooks/ds41-1017-uncensored-vq.md)
- Installer: [`scripts/install-ds41-1017-uncensored-vq.sh`](scripts/install-ds41-1017-uncensored-vq.sh)

## Attribution

Model artifacts and the FP8 PLE runtime work are by
[Baekpica](https://huggingface.co/Baekpica); the engine is a fork of
[`antirez/ds4`](https://github.com/antirez/ds4). This repo contributes the
operational recipe: the artifact set, the launch shape that fits, and the
measurements behind it. Recipes here are provided as-is, with no warranty.
