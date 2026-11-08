# Recipe: GLM-5.3-Flash Uncensored VQ (R13+8) on the DS4 CUDA engine

The abliterated GLM-5.3-Flash with an 8-bit residual stage added to 19 of its 42
routed-expert layers, served on the DGX Spark at the artifact's native
1,048,576-token context on port 8009. 56 GGUF parts, 94,491,712,768 bytes, in a
GVQ1 release directory read by the ds4-dfm-rs Rust host. The 316 nonexpert Q4_K
matrices, the native-precision tensors and the embedded MTP predictor are the same
bytes the previous revisions shipped.

This is the third revision of this artifact, published the same day as the second
one, and it is not a small edit either: five more parts were rewritten in place
upstream, the payload grew by another 4.5 GB, and the card says the same engine
floor covers it, because the residual stage is the same GVR1 container at the same
`format_version` 2 — it is simply applied to five more routed layers.

- Contract: [`recipes/glm-53-flash-uncensored-vq.yaml`](../recipes/glm-53-flash-uncensored-vq.yaml)
- Installer: [`scripts/install-glm53-uncensored-vq-r13p8.sh`](../scripts/install-glm53-uncensored-vq-r13p8.sh)
- Artifact: [`Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF`](https://huggingface.co/Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF) at `743a912d`
- Previous revisions: the same repository at `cc70aef5`, kept upstream at the tag [`r13p8-l3-16`](https://huggingface.co/Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF/tree/r13p8-l3-16) (R13+8 on layers 3-16), and at `1764dd87`, kept at the tag [`q4k-v3`](https://huggingface.co/Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF/tree/q4k-v3) (all routed layers R13); this file described the latter before 2026-10-18 and that text is in this repository's git history
- Sibling: [`Baekpica/GLM-5.3-Flash-VQ-GGUF`](https://huggingface.co/Baekpica/GLM-5.3-Flash-VQ-GGUF), which carries the same structure at its own revision `71c73e7e`, published 27 minutes later

## What this revision changes

Nineteen routed layers keep their 13-bit stage and gain an 8-bit residual stage on
top of it. The first fourteen are layers 3 to 16, the ones the previous revision
already upgraded; this revision adds layers 17, 22, 27, 41 and 44, which are
places 15 to 19 of the same end-to-end single-layer KL sweep:

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

The layers were not chosen by intuition. Upgrading one layer at a time (all others
R13) and measuring held-out KL against the original ranks the bottom of the stack
far ahead — layer 5 by 0.066, 8 by 0.064, 6 by 0.064, 14 by 0.056, 3 by 0.045, and
layers 3-16 take the first fourteen places out of 42; the next five are 17 (0.021),
41 (0.017), 44 (0.015), 27 (0.013) and 22 (0.011), and with them the 19 upgraded
layers hold 77 percent of the summed single-layer gains. A cheaper local screen that
normalized each layer's routed-FFN output error by that layer's FFN output energy
chose layers 4, 5, 9 and 25-40 instead, and lost end to end. The card's held-out
comparison, 5,435 tokens with original-precision nonexperts:

| Routed layers upgraded | PPL | KL vs original |
|---|---:|---:|
| none (the `q4k-v3` revision) | 4.858 | 0.607 |
| layers 3-16 | 3.987 | 0.354 |
| layers 3-16, joint stages (the `r13p8-l3-16` revision) | 3.969 | 0.342 |
| all 42 (does not fit the budget) | 3.595 | 0.151 |
| original model | 3.538 | 0 |

The card also runs that forward on this revision's own parts (5,195 held-out and
fixture tokens, KL against this model's original weights): layers 3-16 give PPL
3.927 / KL 0.331 / top-1 agreement 80.6 percent, and layers 3-16, 17, 22, 27, 41
and 44 give 3.780 / 0.282 / 82.0 percent. Those are the artifact author's numbers,
from the reference forward, not from this engine.

## Artifacts

| Item | Value |
|---|---|
| Repository | `Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF`, revision `743a912d805e563340fa9d107e636cc0584145fd` (committed 2026-10-18T16:21:21Z, the head of main) |
| Payload | 94,491,712,768 bytes (88.0 GiB) as 56 GGUF parts; 94,557,705,365 bytes across all 76 files of the revision |
| Parts | 42 VQ layers: 7 at 2,383,481,600 and 12 at 2,383,485,056 (layers 03-16 plus 17, 22, 27, 41, 44, the R13+8 ones), 23 at 1,477,184,896; 9 native parts (6,609,275,264 total); 4 embedded-MTP parts (7,493,712,896); 1 vision tower (1,127,280,128) |
| Tensors | 56,297, per the revision's own commit message |
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

Both pairs are byte-identical at all three revisions of this artifact, checked
against each revision's own `SHA256SUMS`, and identical at the sibling's revision
`71c73e7e` as well. The installer checks both on every action that touches weights
and refuses, by name, a directory that holds the base release.

## Trap 2, new here: three revisions, the same 56 names, 19 parts rewritten

The residual stage rewrites a layer's part in place upstream, and it has been
published three times in two days. Comparing the Hub listings revision by revision:
**37 of the 56 part digests are identical across all three**, and the 19 that moved
are exactly the layers the stage was added to:

| Layer parts | all-R13 (`1764dd87`) | 3-16 (`cc70aef5`) | the pin (`743a912d`) |
|---|---:|---:|---:|
| `GLM-5.3-Flash-VQ-layer-03…09.gguf` | 1,477,184,128 B | 2,383,481,600 B | 2,383,481,600 B |
| `GLM-5.3-Flash-VQ-layer-10…16.gguf` | 1,477,184,896 B | 2,383,485,056 B | 2,383,485,056 B |
| `GLM-5.3-Flash-VQ-layer-17, 22, 27, 41, 44` | 1,477,184,896 B | 1,477,184,896 B | 2,383,485,056 B |
| the other 23 routed layers, native, MTP and vision parts | — | unchanged | unchanged |

`recipe.json` at the pin is what names those 19 layers (`residual_stage.layers`
3-16, 17, 22, 27, 41 and 44); every size above was read from the Hub listing of the
revision it sits under.

The two release markers cannot see any of this: they are identical across the
revisions, and a directory holding part of one revision and part of another passes
the release check, loads, and serves stale layers without a word. Three things guard
it instead: the revision is pinned and fetched by this installer; a revision
fingerprint reads the 19 upgraded parts by size in three groups — layers 03-09,
layers 10-16, and the five the pin adds — and reports the directory as `pinned`,
`l3-16`, `r13`, `mixed`, `absent` or `unknown`, naming every part that is stale or
missing; and `verify` re-runs the release's own `SHA256SUMS`, which is the authority
on content.

The fingerprint and every guard were exercised on fixtures built from the Hub's own
listing while this revision was written: each state is refused or accepted by name,
`download` names the state it found before it fetches, and an upgrade run over each
older revision lands on `pinned`.

The fingerprint has a stated blind spot, not a hidden one: it compares sizes, so a
fourth revision that rewrote the same 19 layers to the same sizes would read as the
pin. `verify` settles that question, and it reads every byte.

**The all-R13 installer is gone.** `scripts/install-glm53-uncensored-vq.sh` was
retired and deleted on 2026-10-18: its part count and both of its markers match
every revision, and its engine floor (PR #86) is not this revision's floor, so it
would have reported whichever revision it found as its own. The retired file stays
in this repository's git history, and this installer covers all three states in
place.

## The refetch is in place, and that is a decision

The operator chose one directory and one endpoint: this revision replaces the
earlier bytes in `/mnt/models/GLM-5.3-Flash-Uncensored-VQ-GGUF`, and the endpoint
stays `glm-5.3-flash-uncensored-vq` on port 8009, so an open-grok entry written for
an earlier revision keeps working. Because 37 parts are byte-identical across all
three revisions, the refetch adds only the difference, and the installer's disk
check asks for the smaller number: 17,219,684,224 bytes (17.2 GB) over the all-R13
revision, 4,531,500,800 bytes (4.5 GB) over `r13p8-l3-16`, and the whole 94.49 GB
only on a directory that holds none of the three.

What is lost is the earlier bytes on the box. Both stay available upstream, at the
tags `q4k-v3` and `r13p8-l3-16`, and going back means a directory of their own plus
`GLM53R13P8_REVISION=1764dd87b8ee92dd288d7df15bf62d120ae010d8` (or
`GLM53R13P8_REVISION=r13p8-l3-16`).

## The engine floor is a format floor, not a date

Container 1 holds R13 layers; container 2 adds the GVR8 residual stage
(`crates/ds4-core/src/glm_vq.rs:21`, `FORMAT_VERSIONS 1..=2` on the PR #96 build).
A build older than PR #96 has no GVR8 reader at all: its `glm_vq.rs` match falls
through to `unsupported tensor type` (`c942523:crates/ds4-core/src/glm_vq.rs:370`),
so it rejects the upgraded parts. `--mtp-mode`, which the retired all-R13 installer
used as its floor test, is advertised by both builds, so the decisive check here is
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
the 14-layer sweeps above were run with.

`--max-seqs 1` is deliberate, and this revision makes the reason sharper. The
card's quote for the 14-layer revision is 112.0 GiB (weights 83.8, bank 12.8,
scratch 10.7, MTP state 0.7, floor 4) of the 118.7 GiB the engine sees on GB10;
this revision's five extra layers weigh 4.2 GiB more, so about 116.2 GiB by that
quote and about 2.5 GiB of headroom. **One** 1M bank fits, where the all-R13
revision fitted two; two banks would be 124.8 GiB. The installer therefore passes
`--max-seqs 1` at the native context rather than relying on the plan's automatic
width, and a shallower context makes a second bank affordable again
(`GLM53R13P8_MAX_SEQS=2` overrules the default if you have measured that it fits).

That thin margin is why this installer's post-load floor is 2,000 MB and its
watchdog floor 1,500 MB, where 4,000 MB was right for the 14-layer revision's wider
margin: a floor set above the headroom the card itself sizes the model to would
stop a server that had loaded exactly as designed.

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

At the 14-layer revision the upstream measurement is different in kind: on the base
release the two
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

Two lines will read differently at this revision, and both are worth watching on the
first start: the staged payload is 88.0 GiB rather than 71.96 GiB, and the lane will
be `width=1` with the installer's `--max-seqs 1`. Everything else about the sequence
should be the same, and nothing here has been run on the Spark for either R13+8
revision.

`/v1/models` on the running server reported `context_length` 1048576 and
`max_completion_tokens` 393216 at R13, with `tools`, `tool_choice`, `max_tokens`,
`temperature`, `top_p`, `top_k`, `min_p`, `stop`, `seed`, `stream` and
`reasoning_effort` among the supported parameters. Those were not re-read since.

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
bash scripts/install-glm53-uncensored-vq-r13p8.sh download    # 17.2 GB over R13, 4.5 GB over r13p8-l3-16, 94.5 GB on an empty directory
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
release at the 14-layer revision (`35d3833d`), in the same interleaved runs and with
the same binary; the Uncensored release has the same part sizes and engine path but
different weights, and the card says its own DGX Spark measurements follow.

| Release | Prefill tok/s | Decode tok/s | Tokens per verify | Verify cycle |
|---|---:|---:|---:|---:|
| R13+8, layers 3-16 | 1,113.7 | 31.81 | 2.226 | 70.0 ms |
| R13 | 1,175.4 | 35.00 | 2.321 | 66.3 ms |
| Change | −5.2% | −9.1% | −4.1% | +5.5% |

The mechanism, from the same record: decode reads 7.85 instead of 7.5 GB per token
because 14 of 42 routed layers gained a bit per weight; the R13+8 expert kernels
stream at the R13 kernels' rate (isolated 205-213 against 198-217 GB/s). Prefill
pays a second `m16n8k32.e4m3` MMA per k32 step in those layers — a 4K-token gate
launch takes 10.8 ms against R13's 8.0 ms with 64-token tiles — and a whole 8K
prompt prefills at 1,225.7 against 1,307.4 tok/s. MTP-off decode measured 24.40
against 25.14 tok/s.

This revision reads 7.98 GB of weights per token instead of 7.85, because five more
layers carry the residual stage, and the card's words for that are "expected to be
slightly slower than the previous revision". No number exists for it yet.

Quality on the same box, `tests/glm_vq_quality`, 6 cases and 106 target tokens: mean
target NLL 0.313 → 0.285, and the Korean summary R13 garbles is answered. Greedy
`대한민국의 수도는…` loops under R13 and ends normally under R13+8, in both thinking
modes. On the Uncensored card the upgraded layers' weight error is 0.1438 → 0.0408.

## Limits

- No speed and no quality figure exists for this artifact at any revision. What was
  measured for the all-R13 revision is the download, the marker check on the real
  bytes, the load and the served model card; nothing was timed and no request was
  evaluated.
- The card's own measurements are weight-reconstruction error and a reference-forward
  KL, not model-quality scores on this engine. Lossy by construction at 2.625 bits
  per routed-expert weight on 19 layers and 1.625 on the other 23.
- Needs an engine at or after PR #96 (`34c46d1`). The all-R13 revision's floor is
  not enough.
- One 1M bank, with about 2.5 GiB of headroom by the card's arithmetic. Two banks at
  the native context do not fit.
- The refetch is in place, so the earlier revisions' bytes are gone from the box
  once it completes.
- The vision tower ships in the release and this recipe does not load it: text only.
- One DS4 model at a time: the engine takes a single global lock, so this release
  and its base sibling cannot both be resident.
- Long-context behaviour at 1,048,576 has not been evaluated for this artifact.
- The two release markers cannot distinguish the three revisions, and the engine
  cannot distinguish either release from its sibling. The installer's fingerprint
  compares sizes, so a fourth revision that reused the same 19 sizes would pass it;
  `verify` is what settles content.

## Checks

- `test` sends one request and prints the server's own timing line and a slice of
  the answer.
- `check` prints the plan quote, which is where `mtp`, `banks`, `ctx` and the
  memory total are visible without a load — and where an engine without container
  v2 fails.
- `status` prints the revision line: the part count, the template digest, the
  MTP-02 size, the three group counts of the 19 upgraded parts, the state they add
  up to, and the state this run wants.
- `verify` re-runs the release's own `SHA256SUMS` and reads every byte.

## What is verified, and where

Verified on the reference Spark on 2026-10-17, for the **all-R13** revision: the
download into `/mnt/models/GLM-5.3-Flash-Uncensored-VQ-GGUF` (56 parts, 73 GB),
both markers on the real bytes, the start and its readiness, the engine's catalog
line, the device-cache staging of 71.96 GiB in 9.862 s, the plan's per-bank lines,
the continuous lane at `width=2 seq_cap=1048576`, and the served `/v1/models` card.

Verified while writing this recipe, from sources rather than from the box, for
**this** revision: each revision's Hub listing and every part size; the 37-of-56
digest comparison across all three; every revision's `SHA256SUMS`; the container-v2
constants and validation in the engine tree (`FORMAT_VERSIONS 1..=2`,
`RESIDUAL 0x47565238`, GVR1, the all-or-none rule, `residual_stage_is_validated`);
the pre-PR96 `unsupported tensor type` path; the PR #96 merge into main at
`34c46d1`; `recipe.json`'s layer list at the pin; and the card's memory quote,
speed reading and quality tables.

The installer's six states, its refusals by name, its in-place upgrade from each
older revision, the base-release refusal, the revision overrides and the
whole-release diagnosis were exercised on fixtures built from the Hub listing and
sparse parts at the published sizes.

On the box, 2026-10-18: the operator ran `download` against the directory the
all-R13 revision was recorded in, and it answered `fetching the whole release`,
which is what this installer prints when its fingerprint finds none of the 19
upgraded parts there. So what that directory holds now is an open question on the
Spark, recorded here rather than assumed.

Not verified: throughput and time to first token, quality of any kind,
long-context behaviour, image handling, the predictor's benefit at this revision,
and anything at all about this revision on a Spark.

## Troubleshooting

- `check` fails with `unsupported tensor type`: the engine predates PR #96. Build
  from `34c46d1` or later.
- `start` or `check` refuses the directory by name: it reads `r13` ("still holds the
  all-R13 revision"), `l3-16` ("still holds the r13p8-l3-16 revision") or `mixed`.
  Run `download` in each case: it refetches the parts that are not the pin's, in
  place, and the message names them.
- `download` says `fetching the whole release` when you expected a small refetch: it
  found none of the 19 upgraded parts in `$MODEL_DIR`, and it prints that directory's
  own counts, part sizes and markers before it fetches. Check the path
  (`GLM53R13P8_MODEL_DIR`) and the disk rows in `preflight` before letting 94 GB go.
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
- `not listening after 900s`: the first load hashes 94 GB on the same disk the
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
