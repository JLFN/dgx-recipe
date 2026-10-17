# Recipe: GLM-5.3-Flash VQ on the DS4 CUDA engine

A 72 GiB vector-quantized GLM-5.3-Flash, served on the DGX Spark at the
artifact's native 1,048,576-token context. The routed experts are quantized to
Q4_K and carry a codebook; everything else keeps the original precision. The
whole thing is a directory of 56 GGUF parts with their tables beside them, and
the embedded MTP predictor ships inside the same release.

- Contract: [`recipes/glm-53-flash-vq.yaml`](../recipes/glm-53-flash-vq.yaml)
- Launcher: [`scripts/start-glm53-flash-vq.sh`](../scripts/start-glm53-flash-vq.sh)
- Artifact: [`Baekpica/GLM-5.3-Flash-VQ-GGUF`](https://huggingface.co/Baekpica/GLM-5.3-Flash-VQ-GGUF) at `4da527cd`

## What this recipe buys you

One 72 GiB model resident on a 121.6 GiB box, at a context depth nothing else in
this repository reaches: 1,048,576 tokens, which is the model's own declared
maximum rather than a number this repo chose. Two conversations can be in flight
at that depth.

What it does not buy you is a trivially fast server. Measured here at
temperature 0: 24.2 to 24.3 tok/s on one stream, and 12.1 / 12.0 tok/s each when
two streams run at once. The pair finishes in the same wall time as running the
two requests one after the other (22.67 s against 22.65 s), so the second bank
buys a second client not waiting rather than a second stream's worth of tokens.
Decode is bytes per token against the weight stream, and batching two rows does
not halve the bytes read.

The artifact is deliberately lossy where it matters least and exact where it
matters most, so read the card's own quality numbers before using it for
anything that has to be right: held-out prose perplexity 4.8555 against 8.0797,
code-completion 10.5 against 18.4 percent, and Korean meeting-summary
degeneration under greedy decoding. Those are the accepted trade, not a serving
fault.

## Artifacts

| Item | Value |
|---|---|
| Repository | `Baekpica/GLM-5.3-Flash-VQ-GGUF`, revision `4da527cd0f9e83ccd9fb451a1feaf3ba861f7c77` |
| Storage | 92,601,773,476 bytes (86.2 GiB), of which 77,272,028,416 bytes (71.96 GiB) are GGUF |
| Parts | 56 GGUF files: 42 VQ layers (1.37 GiB each), 9 native parts (6.15 GiB), 4 embedded-MTP parts (6.97 GiB), 1 vision tower part (1.04 GiB) |
| Tables | `files.json`, `tensor-manifest.json`, `native-tensors.json`, `q4k-selection.json`, `recipe.json`, `source-files.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `SHA256SUMS`, `FORMAT.md`, `RECIPE.md` |
| Runtime | `ds4-dfm-rs` Rust host, `ds4-server`; engine from `c5384b5e` (PR #86) onward |
| Context | 1,048,576 native, two banks |
| Port | 8007 |

The repository is public and not gated, so no token is required; a token only
raises the CDN rate limit, and the launcher reads `HF_TOKEN` from the
environment when it is set. The fetch is resumable at the file level: re-run the
same command.

## The release is a directory, and that is not a detail

`-m` takes the release directory. The host resolves the parts from `files.json`
and the tensor layout from `tensor-manifest.json`, both of which must sit beside
the parts, together with the tokenizer and the chat template. A flat copy of the
56 parts into a shared models folder is not a working release, and pointing `-m`
at one of the parts is not either.

The engine verifies every file, tensor and payload SHA256 during load, about
20 seconds for this artifact. `verify` in the launcher is a second, independent
opinion over `SHA256SUMS`, and it reads all 72 GiB, so it is slow by nature.

## Why the two non-obvious flags are on the command line

`--mtp-mode on` is required for the predictor, and its absence is silent. The
family reports `mtp_support` unqualified, so an automatic mode resolves to `Off`
(`crates/ds4-core/src/serving.rs:2410`), and `open_draft_tokens` returns `None`
whenever the mode is off (`serving.rs:2517`). The plan line then reads
`mtp=off` and `--mtp-draft 2` is discarded without a word. Only an explicit mode
allocates the predictor state.

`--mtp-draft 2` is the card's measured draft count; there is no separate `--mtp`
path, because this artifact's predictor is four parts of the same release rather
than a sidecar file.

## The speculator: loaded, offered, and not committed here

Both halves of the machinery are in place on the reference Spark, and the
measurement says the drafts still are not used:

- The plan reports `mtp=on weights=true`, and the log warns
  "MTP execution is present but not qualified (mtp_unverified)".
- `DS4_CONT_MTP_MODE=2` and `DS4_CONT_MTP_DEPTH=2` are confirmed present inside
  the running process's own environment (`/proc/<pid>/environ`), which matters
  because the continuous decode loop reads those two variables rather than the
  plan: mode 0 is off, 1 is a non-accepting probe, 2 is verify+accept
  (`ds4.c:62014-62031`). The C oracle server arms mode 2 by itself whenever an MTP
  source is loaded (`ds4_server.c:23970`); the Rust host does not, so the launcher
  exports them.
- Every request, greedy and at the native context, still reports
  `tok_per_step` 1.0.

So the drafts are loaded and offered and nothing is committed. The artifact
card's 34.6 to 40.6 tok/s numbers come from the single-stream greedy bench path,
whose speculation gate lives in the CLI (`ds4-cli`: `use_glm_spec`), not from
this HTTP lane. Nothing here claims those numbers for the server. `MTP_MODE=off`
stops paying for predictor state that this build does not use.

## The memory quote, from the engine

`ds4-server --check-config -m <release dir>` prints the plan as JSON and opens
no weights. On the reference Spark, 2026-10-17:

    shared weights   77,272,028,416   72.0 GiB
    per bank         12,710,844,416   11.8 GiB
    scratch          11,440,271,104   10.7 GiB
    floor             4,294,967,296    4.0 GiB
    banks                             2
    total           118,428,955,648  110.3 GiB
    available       127,386,820,608  118.6 GiB

93.0 percent of what the engine sees as available. The weights are mmap-backed,
so residency follows demand rather than jumping to 72 GiB at once, but the
margin is thin enough that the launcher runs a watchdog: two consecutive samples
of `MemAvailable` below 2500 MB stop the server, which beats a machine that has
started swapping and cannot be typed into. The preflight floors are 100,000 MB
before loading and 10,000 MB after.

`per_bank` is the price of the second conversation, not of the model.

## Quick start

Run on the Spark, as the user who owns the model directory:

```bash
bash scripts/start-glm53-flash-vq.sh preflight   # GPU, CUDA 13, memory, disk, ports
bash scripts/start-glm53-flash-vq.sh download    # 72 GiB, resumable
bash scripts/start-glm53-flash-vq.sh check       # the plan quote, opens no weights
bash scripts/start-glm53-flash-vq.sh start       # serve on 0.0.0.0:8007
```

`plan`, `verify`, `stop`, `status`, `logs`, `test` and `all` round it out. Every
path, port, context and memory knob is an environment override documented in the
script header.

## Running, and what a start looks like

A start hashes the release, maps the weights and binds the port, so the endpoint
appears about a minute after the command, not immediately; the launcher waits for
`/v1/models` and then sends one chat request of its own. It also quotes the
engine's own plan and warning lines from the log, rather than paraphrasing them:

    effective: reuse=exact max_seqs=2 mtp=on weights=true ctx=1048576 floor=4G disk=false
    warn: MTP execution is present but not qualified (mtp_unverified)

Two behaviours surprise people, both from the template. Every reply opens a
thinking channel first, so a request with a small `max_tokens` returns
`finish_reason: length`, empty `content`, and the text in `reasoning_content`;
give it room or send `reasoning_effort: low`. And the served model id is
`glm-5.3-flash-vq`.

## Limits

- One DS4 model at a time: the engine takes a single global lock, and a second
  process refuses to start while this one runs.
- Two banks is the plan's automatic width at this context, not a qualified
  claim: the plan reports `banks=unverified`, and nothing here has pushed two
  long conversations at depth at once.
- Partial prefix reuse is implemented and unqualified (`partial_unqualified`),
  which the plan warns about on every start.
- The vision tower ships in the release and this launcher does not load it, so
  the endpoint serves text only.
- Long-context behaviour at 1,048,576 was not evaluated here. The artifact
  declares that depth; this repository has served short requests at it.

## Using it from open-grok

The entry is in `~/.opengrok/config.toml` and appears after an open-grok
restart:

```toml
[model."glm-5.3-flash-vq"]
model = "glm-5.3-flash-vq"
base_url = "http://192.168.1.91:8007/v1"
api_backend = "chat_completions"
context_window = 1048576
max_completion_tokens = 393216
supports_reasoning_effort = true
```

## Checks

- `bash scripts/start-glm53-flash-vq.sh test` sends one request and prints the
  server's own timing line and a slice of the answer.
- `check` prints the plan quote, which is where `mtp`, `banks`, `ctx` and the
  memory total are visible without a load.
- Two concurrent requests are the honest way to see the two-bank behaviour; the
  numbers in this runbook came from exactly that, twice, on 2026-10-17.

## What is verified, and where

Verified on the reference Spark on 2026-10-17: the download and its byte count,
the SHA256SUMS check, the plan quote, a start and its readiness, single-stream
decode at 24.2 and 24.3 tok/s, two concurrent streams at 12.1 and 12.0 tok/s
with the aggregate unchanged, time to first token with and without contention,
and the `tok_per_step` 1.0 observation with both MTP halves confirmed in the
running process.

Not verified: any shape other than the default native-context two-bank one; the
predictor's benefit (it is loaded, it is not committed); deep or long-context
behaviour; quality of any kind beyond the card's own published numbers.

## Troubleshooting

- `mtp=off` in the plan: `--mtp-mode on` was not passed. It is not implied by
  `--mtp-draft`.
- The endpoint never appears and the log stops after the plan line: check
  `MemAvailable`. The plan total is 110.3 GiB of a 118.6 GiB budget, so another
  resident owner will starve it.
- `error code banks_unsupported` or a refusal at start: another ds4 model owns
  the engine's global lock. Stop it first.
- A request returns empty `content`: the thinking channel consumed the token
  budget. Raise `max_tokens` or lower `reasoning_effort`.
- `the release is incomplete`: the directory has fewer than 56 parts or a
  leftover `.incomplete` file. Re-run `download`; it resumes.

## Attribution

The artifact and the VQ format are by [Baekpica](https://huggingface.co/Baekpica).
The engine is `ds4-dfm-rs`, the Rust-host continuation of
[`antirez/ds4`](https://github.com/antirez/ds4). This repository contributes the
operational recipe: the artifact list, the launch shape, and the measurements
behind it. Provided as-is, with no warranty.
