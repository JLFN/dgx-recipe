# Recipe: GLM-5.3-Flash Uncensored VQ on the DS4 CUDA engine

The abliterated GLM-5.3-Flash through the same two recipes as the base VQ
release: 72 GiB of GVQ1 parts served on the DGX Spark at the artifact's native
1,048,576-token context with two banks, on port 8009. The routed experts are
vector-quantized with a fresh per-layer codebook, 316 nonexpert matrices are
Q4_K, everything else keeps source precision, and the embedded MTP predictor
ships inside the release.

- Contract: [`recipes/glm-53-flash-uncensored-vq.yaml`](../recipes/glm-53-flash-uncensored-vq.yaml)
- Installer: [`scripts/install-glm53-uncensored-vq.sh`](../scripts/install-glm53-uncensored-vq.sh)
- Artifact: [`Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF`](https://huggingface.co/Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF) at `1764dd87`
- Sibling: [`Baekpica/GLM-5.3-Flash-VQ-GGUF`](https://huggingface.co/Baekpica/GLM-5.3-Flash-VQ-GGUF) at `4da527cd`, whose recipe is [`glm-53-flash-vq.md`](glm-53-flash-vq.md)

## What this recipe is, and what it is not

It is the operational half of the artifact: the exact file set, the launch shape,
the flags that are not optional, and the guard that keeps this release's
directory from being confused with its sibling's.

It is not a measurement. No DGX Spark has opened this artifact, here or upstream:
the card records weight-reconstruction error and a host-side validation, and says
in its own words that it made no Spark run. Every speed and quality figure that
appears in this recipe belongs to another artifact and is labelled as such. Treat
the first start on your box as the measurement.

The provenance matters more than usual here. The source model is
`orcarouter/GLM-5.3-Flash-Uncensored-FP8`, the abliterated build of
Z.AI GLM-5.3-Flash: its safety alignment was substantially removed, and the card
says plainly that it can comply with requests the original refuses and is meant
for research such as refusal-mechanism study, red-teaming and robustness
evaluation. Anything deployed from it needs its own safety and moderation layers.

## Artifacts

| Item | Value |
|---|---|
| Repository | `Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF`, revision `1764dd87b8ee92dd288d7df15bf62d120ae010d8` |
| Published | 2026-10-17T12:11:32Z, last modified 2026-10-17T12:51:38Z |
| Parts | 56 GGUF files, 77,272,028,544 bytes (71.97 GiB): 42 VQ layers, 9 native parts, 4 embedded-MTP parts, 1 vision tower (not loaded here) |
| Tables | `files.json`, `tensor-manifest.json`, `native-tensors.json`, `q4k-selection.json`, `recipe.json`, `source-files.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `SHA256SUMS`, `FORMAT.md`, `RECIPE.md` |
| Runtime | `ds4-dfm-rs` Rust host, `ds4-server`; engine from `c5384b5e` (PR #86) onward |
| Context | 1,048,576 native, two banks |
| Port | 8009 |

The repository is public and not gated, so no token is required; a token only
raises the CDN rate limit, and the installer reads `HF_TOKEN` from the
environment when it is set. The fetch is resumable at the file level.

## The trap: two releases, one set of part names

The card says this artifact has the *structure* of the base VQ release at its
final revision. That is literal, and it is the reason this recipe carries a
fingerprint. Comparing the two Hub listings: all 56 GGUF part names are the same,
the tensor order is the same, and every part is the same size except one. The
engine does not help, because it identifies the weight format, not the weights:
the shape for both is named `GLM-5.3-Flash-VQ`
(`crates/ds4-core/src/shape.rs:1122`), served as the variant name at
`crates/ds4-core/src/serving.rs:1485`.

The two places the releases actually differ, and the digests each one's
`SHA256SUMS` publishes:

| File | Uncensored | Base |
|---|---|---|
| `chat_template.jinja` | 10,644 B, `34d5ee66b12f…fada3891` | 10,950 B, `0c4099f3382d…d9e6309c5` |
| `GLM-5.3-Flash-MTP-02-of-04.gguf` | 1,997,061,632 B, `68a41631cabc…f1c28c26` | 1,997,061,504 B, `194058b2ae15…f45ced51af` |

So the installer checks both markers on every action that touches weights and
refuses, by name, a directory that holds the base release. It also pins the
revision by default, unlike the base release's installer, because the revision is
the only other thing that says which bytes are on disk.

## The release is a directory, and that is not a detail

`-m` takes the release directory. The host resolves the parts from `files.json`
and the tensor layout from `tensor-manifest.json`, both of which must sit beside
the parts, together with the tokenizer and the chat template. A flat copy of the
56 parts into a shared models folder is not a working release, and pointing `-m`
at one of the parts is not either.

The engine verifies every file, tensor and payload SHA256 during load, about 20
seconds for this artifact. `verify` in the installer is a second, independent
opinion over `SHA256SUMS`, and it reads all 77 GB, so it is slow by nature.

## Why the two non-obvious flags are on the command line

`--mtp-mode on` is required for the predictor, and its absence is silent. The
family reports `mtp_support` unqualified, so an automatic mode resolves to `Off`
(`crates/ds4-core/src/serving.rs:2410`), and `open_draft_tokens` returns `None`
whenever the mode is off (`serving.rs:2517`). The plan line then reads `mtp=off`
and `--mtp-draft 2` is discarded without a word. Only an explicit mode allocates
the predictor state.

`--mtp-draft 2` is the draft count the base release's card measured. There is no
separate `--mtp` path, because this artifact's predictor is four parts of the same
release rather than a sidecar file. The engine tree's document for a *different*
uncensored artifact (the IQ2_XXS/IQ2_XS/Q2_K mixed one) ran three drafts, and its
accepted-prefix gates saw at most four accepted tokens; that is a different file
set, and its numbers do not transfer.

## The speculator, and the honest state of it

The base release's measurement on this engine build: the plan reports `mtp=on
weights=true`, the log warns "MTP execution is present but not qualified
(mtp_unverified)", `DS4_CONT_MTP_MODE=2` and `DS4_CONT_MTP_DEPTH=2` are confirmed
inside the running process's environment (the continuous decode loop reads those
two variables rather than the plan: mode 0 is off, 1 a non-accepting probe, 2
verify+accept, `ds4.c:62014-62031`; the C oracle server arms mode 2 by itself when
an MTP source is loaded, `ds4_server.c:23970`, and the Rust host does not), and
every request still reports `tok_per_step` 1.0. The drafts are loaded and offered
and nothing is committed.

Nothing about that has been re-run for this artifact. It is carried here because
the two releases share the MTP parts' names and structure, and because the flag
that makes the predictor load at all is the same flag. `GLM53UNC_MTP_MODE=off`
stops paying for predictor state this build does not use.

## The memory quote, from the engine

There is no quote for this artifact. The base release's quote on the reference
Spark, read with `ds4-server --check-config -m <release dir>` (which opens no
weights), 2026-10-17, at the native context with the plan's automatic width:

    shared weights   77,272,028,416   72.0 GiB
    per bank         12,710,844,416   11.8 GiB
    scratch          11,440,271,104   10.7 GiB
    floor             4,294,967,296    4.0 GiB
    banks                             2
    total           118,428,955,648  110.3 GiB
    available       127,386,820,608  118.6 GiB

This artifact's payload is 77,272,028,544 bytes, 128 bytes more than that
release's, so the expected quote is the same lines with the shared-weights and
total figures raised by 128 bytes — derived, not measured, and the reason the
installer's `check` action exists. 93.0 percent of what the engine sees is the
base release's figure; the margin is thin enough that the installer runs the
watchdog two consecutive samples of `MemAvailable` below 2500 MB stop the server,
which beats a machine that has started swapping and cannot be typed into.

## Quick start

Run on the Spark, as the user who owns the model directory:

```bash
bash scripts/install-glm53-uncensored-vq.sh preflight   # GPU, CUDA 13, memory, disk, ports, fingerprint, engine floor
bash scripts/install-glm53-uncensored-vq.sh download    # 77 GB, resumable
bash scripts/install-glm53-uncensored-vq.sh check       # the plan quote, opens no weights
bash scripts/install-glm53-uncensored-vq.sh start --daemon   # serve on 0.0.0.0:8009
```

`plan`, `verify`, `stop`, `status`, `logs`, `test` and `all` round it out. Every
path, port, context and memory knob is an environment override documented in the
script header, and the build step is deliberately not one of the actions: run
`cd ~/ds4-dfm-rs && make cuda-spark` first, and `preflight` will tell you whether
the binary you have is new enough.

## Running, and what a start looks like

A start verifies the release, maps the weights and binds the port, so the endpoint
appears about a minute after the command, not immediately. The installer waits for
`/v1/models` and then sends one chat request of its own. Expect the same line the
base release produces, with the uncensored model id:

    effective: reuse=exact max_seqs=2 mtp=on weights=true ctx=1048576 floor=4G disk=false
    warn: MTP execution is present but not qualified (mtp_unverified)

Two behaviours come from the template and will surprise you. Every reply opens a
thinking channel first, so a request with a small `max_tokens` returns
`finish_reason: length`, empty `content`, and the text in `reasoning_content`;
give it room or send `reasoning_effort: low`. On the base release an omitted
effort field resolved to `max`, which ran to the cap on a bare prompt. And the id
this endpoint serves is `glm-5.3-flash-uncensored-vq`, which the installer passes
explicitly with `--model-id` because the engine's own name for the weights is the
sibling's.

## Limits

- Nothing here is measured. No load, no request, no speed, no quality figure for
  this artifact. The card's own numbers are weight-reconstruction error: routed
  experts 0.143756-0.144532 relative MSE per layer against the official
  conversion's 0.143720-0.144518, and Q4_K 0.00516 on average over the 316
  matrices. Those are not model-quality scores.
- Lossy by construction at roughly 1.6 bits per routed-expert weight. The base
  release's quality notes apply in kind and were not re-measured: held-out prose
  perplexity 4.8555 to 8.0797, code-completion 10.5 to 18.4 percent, Korean
  meeting-summary degeneration under greedy decoding.
- One DS4 model at a time. The engine takes a single global lock, so this release
  and its base sibling cannot both be resident on a 121.6 GiB box; they are
  alternatives on two ports (8009 here, 8007 there).
- Two banks is the plan's automatic width at this context, not a qualified claim,
  and the base release's measurement there was that a second stream splits the
  first stream's rate rather than adding throughput.
- The vision tower ships in the release and this recipe does not load it: text
  only.
- Long-context behaviour at 1,048,576 has not been evaluated for this artifact.
- The predictor is loaded and offered and not committed on the base release's
  build.

## Using it from open-grok

The base release's entry is in `~/.opengrok/config.toml`; this one's is not yet
written, and open-grok needs a restart after it is added. The sibling entry is the
template, with the port and the id changed:

```toml
[model."glm-5.3-flash-uncensored-vq"]
model = "glm-5.3-flash-uncensored-vq"
name = "GLM 5.3 Flash Uncensored VQ (Spark, 1M, two banks)"
base_url = "http://192.168.1.91:8009/v1"
api_backend = "chat_completions"
api_key = "dummy"
context_window = 1048576
max_completion_tokens = 32768
temperature = 1.0
top_p = 0.95
supports_images = false
supports_reasoning_effort = true
```

Effort is client-global in the sibling entry because per-model
`reasoning_effort` is not a supported key there, and thinking cannot be disabled
on this model.

## Checks

- `bash scripts/install-glm53-uncensored-vq.sh test` sends one request and prints
  the server's own timing line and a slice of the answer.
- `check` prints the plan quote, which is where `mtp`, `banks`, `ctx` and the
  memory total are visible without a load.
- `status` prints the marker line: the part count, the chat template's digest and
  the MTP-02 part's size, which together say which release is on disk.
- `verify` re-runs the release's own `SHA256SUMS` and reads every byte.

## What is verified, and where

Verified while writing this recipe, from sources rather than from a Spark: the
repository listing and every part size (Hub API at revision `1764dd87`), the two
distinguishing digests and sizes (the release's own `SHA256SUMS`, compared against
the base release's), the engine's identity string for this weight format
(`shape.rs:1122`, `serving.rs:1485`), the native context
(`rope_orig_ctx: 1_048_576`), and the two flags and variables the C oracle arms
for itself (`ds4.c:62014-62031`, `ds4_server.c:23970`).

Not verified, and the reason this recipe reads the way it does: everything about
running it. No download, no load, no request, no timing, no quality evaluation,
and no `--check-config` quote for this artifact.

## Troubleshooting

- The installer refuses the directory and names the base release: the markers
  match `Baekpica/GLM-5.3-Flash-VQ-GGUF`. Point `GLM53UNC_MODEL_DIR` at a
  directory of its own and fetch again.
- `mtp=off` in the plan: `--mtp-mode on` was not passed. It is not implied by
  `--mtp-draft`.
- The endpoint never appears and the log stops after the plan line: check
  `MemAvailable`. The plan total is 110.3 GiB of a 118.6 GiB budget on the sibling
  release, so another resident owner will starve it.
- A refusal at start, or an error code about banks: another ds4 model owns the
  engine's global lock. Stop it first.
- `not listening after 900s`: the first load hashes 77 GB on the same disk the
  weights live on. Raise `GLM53UNC_READY_TIMEOUT` if the volume is slow.
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
operational recipe: the artifact list, the launch shape, the guard between the two
releases, and the honest statement of what has not been measured. Provided as-is,
with no warranty.
