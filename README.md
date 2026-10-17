# dgx-recipe

Custom inference recipes for the NVIDIA DGX Spark (GB10, SM121, aarch64, 121.6 GiB
unified memory). One recipe per directory entry, each one carrying the exact
artifact list, the launcher, and the measured evidence behind its settings.

Six artifacts are covered, each with a contract, a runbook and a launcher:

| Artifact | Size | Context | Port | State |
|---|---|---:|---:|---|
| Qwen3.8-Flash-Next Uncensored Q5 SSD-PLE + FP8 sidecar | 134.48 GB | 196,608 (two banks) | 8003 | Verified here 2026-09-26 |
| Prism Ternary Bonsai 2 27B (`qwen35`) | 6.71 GiB | 262,144 (one serial lane) | 8005 | Verified here 2026-10-01 |
| YoungAi DeepSeek-V4.1-Flash | 113.6 GB + 203 GB tables | 1,048,576 | 8006 | Third-party artifact; committed here with its numbers unmeasured, and its recipe predates the 2026-10-17 volume split |
| GLM-5.3-Flash VQ, Q4_K nonexpert v3 | 71.96 GiB | 1,048,576 (two banks) | 8007 | Verified here 2026-10-17 |
| DeepSeek-V4.1-Flash-1017 VQ | 318 GB | 1,048,576 | 8008 | Fetched and assembled on the box 2026-10-17; never loaded, nothing measured |
| GLM-5.3-Flash Uncensored VQ, Q4_K nonexpert v3 | 71.97 GiB | 1,048,576 (two banks) | 8009 | Downloaded, loaded and serving on the box 2026-10-17; no speed or quality figure yet |

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
DeepSeek V4.1 Flash 1017 and 8009 GLM-5.3 Uncensored VQ. Only one ds4 model can be
resident at a time whatever the port says, because the engine takes a single global
lock.

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
Spark: shared weights 77,272,028,416 B, per bank 12,710,844,416 B, scratch
11,440,271,104 B, floor 4 GiB, two banks, total 118,428,955,648 B against
127,386,820,608 B available — 93 percent of what the engine sees.

Measured with 256-token greedy completions: 24.2 and 24.3 tok/s on one stream, and
12.1 / 12.0 tok/s each with two streams at once, the pair finishing in 22.67 s
against 22.65 s for the same two requests back to back. Two banks buy a second
client not waiting, not extra tokens per second. Decode is bytes per token against
the weight stream, and batching two rows does not halve those bytes.

Two findings worth carrying forward. The embedded predictor loads and is offered
but is not committed on this engine build: the plan reports `mtp=on weights=true`,
`DS4_CONT_MTP_MODE=2` and `DS4_CONT_MTP_DEPTH=2` are confirmed in the running
process's environment, and every request still reports `tok_per_step` 1.0. The
artifact card's 34.6 to 40.6 tok/s belongs to the single-stream greedy bench path,
not to this HTTP lane. And thinking can run away: on a bare prompt, `max` effort —
or an omitted field, which the template resolves to `max` — ran to the cap with no
answer at all, byte-identical at temperature 0 and 1.0, while `low` closed in 13
tokens. The model's author recommends `max`, and it behaves on real tasks; the
practical guard is a bounded completion and `/effort low` for trivial prompts.

## GLM-5.3-Flash Uncensored VQ, loaded on the box, nothing timed

The abliterated GLM-5.3-Flash through the same two conversion recipes as the base
VQ release above, on port 8009. Its card says the artifact repeats that release's
structure at `4da527cd`, and that is literal: all 56 GGUF part names are the same,
tensor order and types are the same, and one part differs in size —
`GLM-5.3-Flash-MTP-02-of-04.gguf`, 1,997,061,632 bytes against the base's
1,997,061,504.

Nothing downstream can tell the two apart, because the engine identifies the
weight format rather than the weights: its shape name is the literal string
`GLM-5.3-Flash-VQ` (`crates/ds4-core/src/shape.rs:1122`, served as the variant
name at `crates/ds4-core/src/serving.rs:1485`). So the installer keeps this
release in a directory of its own and fingerprints two markers whose digests each
release's `SHA256SUMS` publishes — `chat_template.jinja`, 10,644 bytes here
against the base's 10,950, and the MTP-02 part — refusing, by name, a directory
that holds the base release. It also pins the revision by default, where the base
release's installer tracks main.

- Recipe contract: [`recipes/glm-53-flash-uncensored-vq.yaml`](recipes/glm-53-flash-uncensored-vq.yaml)
- Runbook: [`runbooks/glm-53-flash-uncensored-vq.md`](runbooks/glm-53-flash-uncensored-vq.md)
- Installer: [`scripts/install-glm53-uncensored-vq.sh`](scripts/install-glm53-uncensored-vq.sh)

The guard was not needed on the first run, because only one of the two releases is
on the box: the artifact went to `/mnt/models/GLM-5.3-Flash-Uncensored-VQ-GGUF`,
56 parts and 73 GB, and both markers matched on the real bytes (the template hashes
to `34d5ee66b12f…fada3891` and MTP-02 is 1,997,061,632 bytes, against the base
release's `0c4099f3382d…d9e6309c5` and 1,997,061,504). It loads and serves: the
engine's own log reports the catalog at 39,862 tensors and 37,152 routed-expert,
prepares 71.96 GiB of tensor spans into the device cache in 9.862 s, and comes up
on port 8009 as `glm-5.3-flash-uncensored-vq` at the native 1,048,576 context with
the continuous lane at `width=2 seq_cap=1048576`.

That log also makes the collision concrete rather than theoretical: it names the
family `GLM-5.3-Flash-VQ`, which is the shape's name and not the weights', and only
`--model-id` makes the endpoint answer as the uncensored model. Nothing else on the
box would have noticed if the other release's files had been sitting there.

What is still missing is what the artifact actually costs to use: no request has
been timed, so no decode rate and no time to first token exist for it, and no
quality evaluation has been done. The numbers in its recipe are the base release's
and are labelled as such, and the card's own figures are weight-reconstruction
error rather than model quality. It also needs its open-grok entry added before it
can be selected from a client.

One thing about it is worth stating plainly rather than implying: the source model
is `orcarouter/GLM-5.3-Flash-Uncensored-FP8`, the abliterated build in the
official block-FP8 layout, so the alignment that the original carries was largely
removed. The card says it can comply with requests the original refuses and that it
is meant for refusal-mechanism research, red-teaming and robustness evaluation,
with safety and moderation layers added before any deployment. This repository
covers the operational half only: the artifact list, the launch shape, and the
guard between it and its sibling.

## DeepSeek-V4.1-Flash-1017 VQ, fetched on the box, never loaded

DeepSeek's October 17 V4.1 Flash checkpoint through YoungAi's
`vq8sh14-q4k-mtpnative` recipe: a 113.56 GB base assembled from 40 byte-range
parts that carries the model's own three-tower DSpark drafter, a 971 MB vision
sidecar, and the 1017 Engram tables (2 x 101.5 GB) read in rows from disk. Port
8008.

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

## Attribution

Model artifacts and the FP8 PLE runtime work are by
[Baekpica](https://huggingface.co/Baekpica); the engine is a fork of
[`antirez/ds4`](https://github.com/antirez/ds4). This repo contributes the
operational recipe: the artifact set, the launch shape that fits, and the
measurements behind it. Recipes here are provided as-is, with no warranty.
