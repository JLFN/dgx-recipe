# Recipe: GLM-5.3-Flash Uncensored VQ (R13+8) on the DS4 CUDA engine

The abliterated GLM-5.3-Flash with an 8-bit residual stage added to 14 of its 42
routed-expert layers, served on the DGX Spark at the artifact's native
1,048,576-token context on port 8009. 56 GGUF parts, 89,960,211,968 bytes, in a
GVQ1 release directory read by the ds4-dfm-rs Rust host. The 316 nonexpert Q4_K
matrices, the native-precision tensors and the embedded MTP predictor are the same
bytes the previous revision shipped.

This is the second revision of this artifact and it is not a small edit: 14 of the
56 parts were rewritten in place upstream, the payload grew by 12.7 GB, the engine
requirement moved from PR #86 to PR #96, and the release that used to fit two 1M
banks now fits one.

- Contract: [`recipes/glm-53-flash-uncensored-vq.yaml`](../recipes/glm-53-flash-uncensored-vq.yaml)
- Installer: [`scripts/install-glm53-uncensored-vq-r13p8.sh`](../scripts/install-glm53-uncensored-vq-r13p8.sh)
- Artifact: [`Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF`](https://huggingface.co/Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF) at `cc70aef5`
- Previous revision: the same repository at `1764dd87`, kept upstream at the tag [`q4k-v3`](https://huggingface.co/Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF/tree/q4k-v3); this file described it before 2026-10-18 and that text is in this repository's git history
- Sibling: [`Baekpica/GLM-5.3-Flash-VQ-GGUF`](https://huggingface.co/Baekpica/GLM-5.3-Flash-VQ-GGUF), which moved to its own R13+8 revision `35d3833d` in the same minute

## What this revision changes

Fourteen routed layers — the first fourteen, layers 3 to 16 — keep their 13-bit
stage and gain an 8-bit residual stage on top of it:

```
stage 1   w[r, 8j:8j+8]  ~=  g_r . C1[i1]      C1: 8192 x 8 E4M3, one per layer
stage 2   + C2[i2]                             C2:  256 x 8 E4M3, one per layer
bits per weight: 13/8 + 8/8 = 2.625
```

Stage 2 is a `GVR8` tensor (type `0x47565238`, header `GVR1`) holding one index
byte per 8-weight vector, and `glm_vq.residuals` links each GVQ1 matrix to its
residual. Only these 14 parts declare `glm_vq.format_version` 2, and a layer is
all-or-none. The card's own comparison, on the base model's weights: layer 23 goes
from 0.1438 relative MSE at 1.625 bits to 0.0471 at 2.625 bits with a plain
nearest-codeword stage 2, and 0.0412 with the joint beam assignment this release
uses.

The 14 layers were not chosen by intuition. Upgrading one layer at a time (all
others R13) and measuring held-out KL against the original ranks the bottom of the
stack far ahead — layer 5 by 0.066, 8 by 0.064, 6 by 0.064, 14 by 0.056, 3 by
0.045, and layers 3-16 take the first fourteen places out of 42. A cheaper local
screen that normalized each layer's routed-FFN output error by that layer's FFN
output energy chose layers 4, 5, 9 and 25-40 instead, and lost end to end. The
card's held-out comparison, 5,435 tokens with original-precision nonexperts:

| Routed layers upgraded | PPL | KL vs original |
|---|---:|---:|
| none (previous revision) | 4.858 | 0.607 |
| layers 3-16 | 3.987 | 0.354 |
| layers 3-16, joint stages (this release) | 3.969 | 0.342 |
| all 42 (does not fit the budget) | 3.595 | 0.151 |
| original model | 3.538 | 0 |

## Artifacts

| Item | Value |
|---|---|
| Repository | `Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF`, revision `cc70aef5a5ad370fbd59a69dc0a39522a1ed5c08` (committed 2026-10-18T11:23:06Z, now the head of main) |
| Payload | 89,960,211,968 bytes (83.78 GiB) as 56 GGUF parts; 90,023,456,383 bytes across all 76 files of the revision |
| Parts | 42 VQ layers: 7 at 2,383,481,600 and 7 at 2,383,485,056 (layers 03-16, the R13+8 ones), 28 at 1,477,184,896 (layers 17-44); 9 native parts (6,609,275,264 total); 4 embedded-MTP parts (7,493,712,896); 1 vision tower (1,127,280,128) |
| Tables | `files.json`, `tensor-manifest.json`, `native-tensors.json`, `q4k-selection.json`, `recipe.json`, `source-files.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `SHA256SUMS`, `FORMAT.md`, `RECIPE.md`, `config.json`, `generation_config.json`, `processor_config.json`, `LICENSE`, `LICENSE-YoungAi`, and the throughput graph |
| Runtime | `ds4-dfm-rs` Rust host, `ds4-server`; **PR #96 (`34c46d1`, "Add the GLM VQ R13+8 residual stage") or later** |
| Context | 1,048,576 native, **one bank** |
| Port | 8009 |

The repository is public and not gated, so no token is required; a token only
raises the CDN rate limit, and the installer reads `HF_TOKEN` from the
environment when it is set. The fetch is resumable at the file level.

## Trap 1: two releases, one set of part names

Unchanged from the previous revision. The Uncensored release repeats the base
release's 56 part names, tensor order and payload sizes, and the engine identifies
both as the same artifact, because its shape name is the literal string
`GLM-5.3-Flash-VQ` (`crates/ds4-core/src/shape.rs:1122`, served as the variant name
at `crates/ds4-core/src/serving.rs:1485`). Only two things differ:

| File | Uncensored | Base |
|---|---|---|
| `chat_template.jinja` | 10,644 B, `34d5ee66b12f…fada3891` | 10,950 B, `0c4099f3382d…d9e6309c5` |
| `GLM-5.3-Flash-MTP-02-of-04.gguf` | 1,997,061,632 B, `68a41631cabc…f1c28c26` | 1,997,061,504 B, `194058b2ae15…f45ced51af` |

Both pairs are the same at R13+8 as they were at R13, verified against the base
release's own `SHA256SUMS` at its new revision `35d3833d`. The installer checks
both on every action that touches weights and refuses, by name, a directory that
holds the base release.

## Trap 2, new here: two revisions, the same 56 names, 14 parts rewritten

R13+8 replaced 14 parts in place. Comparing the two Hub listings file by file:
**42 of the 56 part digests are identical**, and the 14 that moved are exactly
layers 03-16:

| Layer parts | R13 (`1764dd87`) | R13+8 (`cc70aef5`) |
|---|---:|---:|
| `GLM-5.3-Flash-VQ-layer-03…09.gguf` | 1,477,184,128 B | 2,383,481,600 B |
| `GLM-5.3-Flash-VQ-layer-10…16.gguf` | 1,477,184,896 B | 2,383,485,056 B |
| `GLM-5.3-Flash-VQ-layer-17…44.gguf` | 1,477,184,896 B | unchanged |
| native, MTP and vision parts | — | unchanged |

The two release markers above cannot see any of this: they are identical between
the revisions. A directory holding half of one revision and half of the other
passes the release check, loads, and serves 14 stale layers without a word. Three
things guard it instead: the revision is pinned and fetched by the new installer;
a revision fingerprint counts both halves of the 14 parts (7 + 7) and reports the
directory as `r13p8`, `r13`, `mixed` or `unknown`, naming the stale parts; and
`verify` re-runs the release's own `SHA256SUMS` over all 90 GB.

The fingerprint was exercised on a fixture while this recipe was written: a
directory holding R13 sizes is refused by `start` and `check` with the names of
the parts and the command that fixes them, and a half-refetched directory is
reported as `mixed` with exactly the seven parts that are still stale.

**The previous installer must not be pointed at an R13+8 directory.** Its part
count is 56 and both of its markers match, so `install-glm53-uncensored-vq.sh`
would report the newer revision as its own, and its engine floor (PR #86) is not
this release's floor. The two scripts deliberately share the release directory;
use the one named for the revision you are serving.

## The refetch is in place, and that is a decision

The operator chose one directory and one endpoint: this revision replaces the R13
bytes in `/mnt/models/GLM-5.3-Flash-Uncensored-VQ-GGUF`, and the endpoint stays
`glm-5.3-flash-uncensored-vq` on port 8009, so an open-grok entry written for the
previous revision keeps working. Because 42 parts are byte-identical, the refetch
adds 12,688,183,424 bytes rather than 89.96 GB, and the installer's disk check asks
for the smaller number when it finds R13 on disk.

What is lost is the R13 bytes on the box. They stay available upstream at the tag
`q4k-v3`, and going back means a directory of their own plus
`GLM53R13P8_REVISION=1764dd87b8ee92dd288d7df15bf62d120ae010d8`.

## The engine floor is a format floor, not a date

Container 1 holds R13 layers; container 2 adds the GVR8 residual stage
(`crates/ds4-core/src/glm_vq.rs:21`, `FORMAT_VERSIONS 1..=2` on the PR #96 build).
A build older than PR #96 has no GVR8 reader at all: its `glm_vq.rs` match falls
through to `unsupported tensor type` (`c942523:crates/ds4-core/src/glm_vq.rs:370`),
so it rejects the 14 upgraded parts. `--mtp-mode`, which the previous installer
uses as its floor test, is advertised by both builds, so the decisive check here is
`check`: `--check-config` opens no weights and reads the container versions. The
engine's Rust validation is also the only reader that checks the residual stage's
headers, offsets, payload checksums, padding, book links, parent shapes and
versions (`residual_stage_is_validated`).

## Why the two non-obvious flags are still on the command line

`--mtp-mode on` is required for the predictor, and its absence is silent. The
family reports `mtp_support` unqualified, so an automatic mode resolves to `Off`
(`crates/ds4-core/src/serving.rs:2410`), and `open_draft_tokens` returns `None`
whenever the mode is off (`serving.rs:2517`). Only an explicit mode allocates the
predictor state.

`--mtp-draft 2` is the draft count the base release's card measured, and it is what
the R13+8 sweeps above were run with.

`--max-seqs 1` is new here and is deliberate. The card's own `--check-config` quote
for this revision is 112.0 GiB (weights 83.8, bank 12.8, scratch 10.7, MTP state
0.7, floor 4) against the 118.7 GiB the engine sees on GB10: **one** 1M bank fits,
where R13 fitted two, because the residual stage adds 0.844 GiB per upgraded layer.
Two banks would be 124.8 GiB. The installer therefore passes `--max-seqs 1` at the
native context rather than relying on the plan's automatic width; a shallower
context makes a second bank affordable again, and `GLM53R13P8_MAX_SEQS=2` overrules
the default if you have measured that it fits on your box.

The previous revision's measured quote on the reference Spark, for contrast — read
with `ds4-server --check-config -m <release dir>` on 2026-10-17 at the same context,
for two banks:

    shared weights   77,272,028,416   72.0 GiB
    per bank         12,710,844,416   11.8 GiB
    scratch          11,440,271,104   10.7 GiB
    floor             4,294,967,296    4.0 GiB
    banks                             2
    total           118,428,955,648  110.3 GiB
    available       127,386,820,608  118.6 GiB

## The speculator, and the honest state of it

The base release's measurement on the 2026-10-17 build: the plan reports `mtp=on
weights=true`, the log warns "MTP execution is present but not qualified
(mtp_unverified)", `DS4_CONT_MTP_MODE=2` and `DS4_CONT_MTP_DEPTH=2` are confirmed
inside the running process's environment (the continuous decode loop reads those two
variables rather than the plan: mode 0 is off, 1 a non-accepting probe, 2
verify+accept, `ds4.c:62014-62031`; the C oracle server arms mode 2 by itself when
an MTP source is loaded, `ds4_server.c:23970`, and the Rust host does not), and every
request still reported `tok_per_step` 1.0. The drafts were loaded and offered and
nothing was committed.

At R13+8 the upstream measurement is different in kind: on the base release the two
draft cycle reports 2.226 tokens per verify and a 70.0 ms cycle against R13's 2.321
and 66.3 ms, so the predictor is committing tokens on that build. Nothing about the
Uncensored weights has been re-run, and the flag that makes the predictor load at
all is unchanged. `GLM53R13P8_MTP_MODE=off` stops paying for predictor state if you
find this build does not use it.

## Running, and what a start looks like

The log below is the previous revision's first start on the reference Spark,
2026-10-17, kept because it is the record of the sequence (release check, marker
check, load, readiness, smoke request) and because its first line is the collision
this recipe guards against — the engine calls the artifact `GLM-5.3-Flash-VQ`:

    family: GLM-5.3-Flash-VQ
    requested: reuse=auto max_seqs=auto mtp=on ctx=1048576 floor=4G disk=false
    ds4: model catalog base: 39862 tensors, 37152 routed-expert, 0 artifact-replaced
    ds4: CUDA startup model cache prepared 71.96 GiB of tensor spans in 9.862s
    ds4: GLM-5.3 VQ source eager graph: ctx 1048576, prefill 4096, KV 11.69 GiB, state 148.82 MiB, workspace 10897.05 MiB
    ds4-server-rs: continuous lane ready (width=2 seq_cap=1048576)
    ds4-server-rs: listening on 0.0.0.0:8009 model_id=glm-5.3-flash-uncensored-vq engine=open

Two lines will read differently at R13+8, and both are worth watching on the first
start: the staged payload is 83.78 GiB rather than 71.96 GiB, and the lane will be
`width=1` with the installer's `--max-seqs 1`. Everything else about the sequence
should be the same, and nothing here has been run on the Spark for this revision.

`/v1/models` on the running server reported `context_length` 1048576 and
`max_completion_tokens` 393216 at R13, with `tools`, `tool_choice`, `max_tokens`,
`temperature`, `top_p`, `top_k`, `min_p`, `stop`, `seed`, `stream` and
`reasoning_effort` among the supported parameters. Those were not re-read at R13+8.

Two behaviours come from the template and will surprise you. Every reply opens a
thinking channel first, so a request with a small `max_tokens` returns
`finish_reason: length`, empty `content`, and the text in `reasoning_content`; give
it room or send `reasoning_effort: low`. On the base release an omitted effort field
resolved to `max`, which ran to the cap on a bare prompt. And the id this endpoint
serves is `glm-5.3-flash-uncensored-vq`, which the installer passes explicitly with
`--model-id` because the engine's own name for the weights is the sibling's.

## Quick start

Run on the Spark, as the user who owns the model directory:

```bash
bash scripts/install-glm53-uncensored-vq-r13p8.sh preflight   # GPU, CUDA 13, memory, disk, ports, both fingerprints, engine floor
bash scripts/install-glm53-uncensored-vq-r13p8.sh download    # 12.7 GB if R13 is on disk, 90 GB on an empty directory
bash scripts/install-glm53-uncensored-vq-r13p8.sh check       # the plan quote, opens no weights, proves the container-v2 reader
bash scripts/install-glm53-uncensored-vq-r13p8.sh start --daemon   # serve on 0.0.0.0:8009
```

`plan`, `verify`, `stop`, `status`, `logs`, `test` and `all` round it out. Every
path, port, context and memory knob is an environment override documented in the
script header, and the build step is deliberately not one of the actions: run
`cd ~/ds4-dfm-rs && make cuda-spark` first, and `preflight` will tell you whether
the checkout you have knows the commit this revision needs.

## Nothing here needs SSD streaming

`--ssd-streaming` is off, and it is not available to this family at all: the
checks require one full, single-shard `Glm53` CUDA artifact
(`crates/ds4-core/src/ssd_quote.rs:139`, `crates/ds4-core/src/ssd.rs:68`), and this
release is family `Glm53Vq` across 56 parts. The practical consequence is that
these releases want a volume that reads them once quickly and then never matters
again, which is why the payload lives on the roomy `/mnt/models`.

## Numbers that belong to another artifact

Every figure below was measured by the artifact author on one GB10, on the **base**
release at this revision (`35d3833d`), in the same interleaved runs and with the
same binary; the Uncensored release has the same part sizes and engine path but
different weights, and the card says its own DGX Spark measurements follow.

| Release | Prefill tok/s | Decode tok/s | Tokens per verify | Verify cycle |
|---|---:|---:|---:|---:|
| R13+8 (layers 3-16) | 1,113.7 | 31.81 | 2.226 | 70.0 ms |
| R13 | 1,175.4 | 35.00 | 2.321 | 66.3 ms |
| Change | −5.2% | −9.1% | −4.1% | +5.5% |

The mechanism, from the same record: decode reads 7.85 instead of 7.5 GB per token
because 14 of 42 routed layers gained a bit per weight; the R13+8 expert kernels
stream at the R13 kernels' rate (isolated 205-213 against 198-217 GB/s). Prefill
pays a second `m16n8k32.e4m3` MMA per k32 step in those layers — a 4K-token gate
launch takes 10.8 ms against R13's 8.0 ms with 64-token tiles — and a whole 8K
prompt prefills at 1,225.7 against 1,307.4 tok/s. MTP-off decode measured 24.40
against 25.14 tok/s.

Quality on the same box, `tests/glm_vq_quality`, 6 cases and 106 target tokens: mean
target NLL 0.313 → 0.285, and the Korean summary R13 garbles is answered. Greedy
`대한민국의 수도는…` loops under R13 and ends normally under R13+8, in both thinking
modes. On the Uncensored card the upgraded layers' weight error is 0.1438 → 0.0408.

## Limits

- No speed and no quality figure exists for this artifact at either revision. What
  was measured for the previous revision is the download, the marker check on the
  real bytes, the load and the served model card; nothing was timed and no request
  was evaluated.
- The card's own measurements are weight-reconstruction error, not model-quality
  scores. Lossy by construction at 2.625 bits per routed-expert weight on 14 layers
  and 1.625 on the other 28.
- Needs an engine at or after PR #96 (`34c46d1`). The previous revision's floor is
  not enough.
- One 1M bank. Two banks at the native context do not fit.
- The refetch is in place, so the R13 bytes are gone from the box once it
  completes.
- The vision tower ships in the release and this recipe does not load it: text only.
- One DS4 model at a time: the engine takes a single global lock, so this release
  and its base sibling cannot both be resident.
- Long-context behaviour at 1,048,576 has not been evaluated for this artifact.
- The two release markers cannot distinguish the two revisions, and the engine
  cannot distinguish either release from its sibling.

## Checks

- `test` sends one request and prints the server's own timing line and a slice of
  the answer.
- `check` prints the plan quote, which is where `mtp`, `banks`, `ctx` and the
  memory total are visible without a load — and where an engine without container
  v2 fails.
- `status` prints the marker line and the revision line: the part count, the
  template digest, the MTP-02 size, and the sizes of two of the 14 upgraded parts.
- `verify` re-runs the release's own `SHA256SUMS` and reads every byte.

## What is verified, and where

Verified on the reference Spark on 2026-10-17, for the **previous** revision: the
download into `/mnt/models/GLM-5.3-Flash-Uncensored-VQ-GGUF` (56 parts, 73 GB),
both markers on the real bytes, the start and its readiness, the engine's catalog
line, the device-cache staging of 71.96 GiB in 9.862 s, the plan's per-bank lines,
the continuous lane at `width=2 seq_cap=1048576`, and the served `/v1/models` card.

Verified while writing this recipe, from sources rather than from the box, for
**this** revision: the Hub listing at `cc70aef5` and every part size; the 42-of-56
digest comparison against `1764dd87`; both releases' `SHA256SUMS` at both revisions;
the base release's move to `35d3833d`; the container-v2 constants and validation in
the engine tree (`FORMAT_VERSIONS 1..=2`, `RESIDUAL 0x47565238`, GVR1, the
all-or-none rule, `residual_stage_is_validated`); the pre-PR96 `unsupported tensor
type` path; the PR #96 merge into main at `34c46d1`; and the card's residual-stage
recipe, layer selection and memory quote. The revision fingerprint, the `mixed`
report and the base-release refusal were exercised on a fixture built from the real
table files and sparse parts at the published sizes.

Not verified: throughput and time to first token, quality of any kind,
long-context behaviour, image handling, the predictor's benefit at this revision,
and anything at all about this revision on a Spark.

## Troubleshooting

- `check` fails with `unsupported tensor type`: the engine predates PR #96. Build
  from `34c46d1` or later.
- `start` or `check` refuses because the directory "still holds the R13 revision":
  run `download`, which refetches the 14 parts in place.
- `status` says `mixed`: a refetch was interrupted. It names the stale parts;
  re-run `download` and then `verify`.
- The installer refuses the directory and names the base release: point
  `GLM53R13P8_MODEL_DIR` at a directory of its own and fetch again.
- `mtp=off` in the plan: `--mtp-mode on` was not passed.
- The endpoint never appears and the log stops after the plan line: check
  `MemAvailable`. The plan is 112.0 GiB of a 118.7 GiB budget, so another resident
  owner will starve it.
- A refusal at start, or an error code about banks: another ds4 model owns the
  engine's global lock. Stop it first.
- `not listening after 900s`: the first load hashes 90 GB on the same disk the
  weights live on. Raise `GLM53R13P8_READY_TIMEOUT` if the volume is slow.
- A request returns empty `content`: the thinking channel consumed the token
  budget. Raise `max_tokens` or lower `reasoning_effort`.
- `the release is incomplete`: fewer than 56 parts, or a leftover `.incomplete`
  file. Re-run `download`; it resumes.

## Attribution

The artifact, the VQ format and the conversion tools are by
[Baekpica](https://huggingface.co/Baekpica); the source weights are
`orcarouter/GLM-5.3-Flash-Uncensored-FP8`, itself the abliterated Z.AI
GLM-5.3-Flash, and the algorithms carry YoungAi's attribution in
`LICENSE-YoungAi`. The engine is `ds4-dfm-rs`, the Rust-host continuation of
[`antirez/ds4`](https://github.com/antirez/ds4). This repository contributes the
operational recipe: the artifact list, the launch shape, the two guards (release
and revision), and the honest statement of what has not been measured. Provided
as-is, with no warranty.
