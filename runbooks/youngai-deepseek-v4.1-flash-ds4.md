# Recipe: YoungAi DeepSeek-V4.1-Flash (VQ-8 base + domain sidecar) on the DGX Spark

One 113.6 GB vector-quantized DeepSeek V4.1 Flash, one ~40 MB domain sidecar, and
203 GB of n-gram tables the engine reads off the SSD. The engine is the
`ds4-server` binary that ships inside the same Hugging Face repository, so there
is nothing to compile. Launcher:
[`scripts/start-youngai-v41-flash.sh`](../scripts/start-youngai-v41-flash.sh).
Contract: [`recipes/youngai-deepseek-v4.1-flash-ds4.yaml`](../recipes/youngai-deepseek-v4.1-flash-ds4.yaml).
Source: <https://huggingface.co/wenzhouwu/YoungAi-DeepSeek-V4.1-Flash>.

## Read this first: what is ours and what is somebody else's

This is the first recipe in this repository that is not built from source. The
other two compile `ds4-dfm-rs` on the Spark. Here the engine, the quantizer and
the solvers are the model author's, their source is not published, and **two
opaque aarch64 binaries run on your machine**. The card says so itself: "Source
code of the engine, the quantizer and the solvers is not public yet." The only
integrity check available is the repository's own `SHA256SUMS`, which is
self-referential: a repository that had been tampered with would carry matching
hashes. This recipe contributes the operational part, the artifact list, the
launch shape that fits the box, and the checks around it. It cannot vouch for
what the binaries do.

**No ds4 checkout on either machine can serve this file.** There are four: the
Spark's `~/ds4-dfm-rs`, and `/data/ds4-dfm-rs`, `/data/ds4` and `/data/ds4-work`
on the workstation. A recursive search of each for what this artifact needs
returns nothing at all — `zchain` (the sidecar flag) 0 files, `DQV3` (the base's
format magic) 0 files, `--engram-dir` 0 files, `youngai` and `posttrain` 0 files
(checked 2026-10-01, local copies on disk and the Spark copy over SFTP). The one
related concept that *does* exist in the lineage is Engram, and it is packaged
differently: in `/data/ds4` and `/data/ds4-work` the tables live inside the single
GGUF artifact (`ds4_engram.c`, a `dsv41_engram_kernel`, and the
`--ssd-streaming*` flag family; `/data/ds4/QA_BEFORE_RELEASES.md` records "188.8
GiB of Engram remains disk-only"). YoungAi externalizes them instead to two
official FP8 shards and reads them through `--engram-dir`. Same idea, different
packaging — and there is no `--zchain` anywhere, because a domain sidecar is not a
feature of that engine at all.

The card is explicit that this matters: the base carries a version-3 payload with
a new magic (`DQV3`) and the loader whitelists format versions, so "an engine that
does not know a format refuses the file at load", while one that predates the
whitelist "outputs all-zero experts from the first sentence". The author's
installer makes the same point operationally by checking `bin/ds4-server --help`
for `--engram-dir` before starting. The shipped `bin/ds4-server` is therefore the
only engine for this artifact, and this recipe installs into
`~/models/youngai`: weights in `~/models` next to the other models on that box,
engine checkouts in `~/ds4-dfm-rs`, as the rest of the box already does it.

## What this recipe buys you

- DeepSeek V4.1 Flash running on one Spark, from a 113.6 GB file instead of the
  510 GB official checkpoint, with no second machine and no Docker.
- 1M tokens of context, taken from the model's own metadata. There is no context
  flag to set: this engine rejects `--ctx`.
- Five selectable domains (finance, code, law, medicine, science). A sidecar is
  about 40 MB, and switching domains is one argument.
- The 203 GB n-gram memory stays on the SSD, untouched, about 6 KB read per
  decoded token, which is what makes the 113.6 GB base fit in 121.6 GiB of
  unified memory with room to run.

## Artifacts

| Part | What | Size |
|---|---|---|
| Base, `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf` | 40 parts, `part01-of-40` … `part40-of-40`; parts 1-24 are 2,838,915,986 bytes, parts 25-40 are 2,838,915,985 | 113,556,639,424 bytes (113.6 GB), sha256 `9469cfa9ff47c5b9f1e2bbf31b4226ef11b4de9ddb92a0c7641c4a9b623e4654` |
| N-gram ("engram") tables | `model-00047-of-00048.safetensors` and `model-00048-of-00048.safetensors` from the official checkpoint | 101,535,150,936 + 101,537,926,640 bytes (203 GB), read-only |
| Engine | `bin/ds4`, `bin/ds4-server`, aarch64, built by the author on a Spark | prebuilt; no source published |
| Sidecars | five directories, each `gr_L*.bin` (1,044,492 bytes/layer) + `rb_L*.bin` (1,544 bytes) + `manifest.txt` | 40.8-41.8 MB each, about 206 MB together |
| Post-training file | `posttrain-experimental-20260924/`: `gr_L39.bin` + `base.fnv`, fingerprinted to the finance sidecar | experimental, finance only |

What the 113.56 GB is made of, from the card: 104.90 GB of routed-expert VQ-8
indices (26 layers at 12 bit, the 14 shallowest at 13 bit), 0.30 GB of row gains,
4.96 GB of q4_K attention and output head, 2.56 GB of speculative draft towers,
and 0.85 GB of everything else. That is roughly 1.6 bits per weight across the
whole file.

## The one flag that is not optional

`--engram-dir`. The GGUF records the two n-gram shards by the absolute path they
had on the machine that built it, a path that exists nowhere else. Without the
override the engine warns at load and then fails the first request with an
engram-table error, after a two-minute load. The shards are read-only inputs; the
launcher never writes to them and accepts `ENGRAM_DIR=` pointing at a copy you
already have, which skips their 203 GB entirely.

## Quick start

On the Spark, as the user who owns the install directory:

```sh
cd ~/  # wherever the launcher is
bash start-youngai-v41-flash.sh preflight     # GPU, CUDA 13, memory, disk, ports, ~1 s
bash start-youngai-v41-flash.sh plan          # what would be fetched, with sizes; touches nothing
bash start-youngai-v41-flash.sh download      # 317 GB, resumable at three levels
bash start-youngai-v41-flash.sh verify        # parts, assembled base, engram sizes, sidecar manifest
bash start-youngai-v41-flash.sh build         # chmod +x, and check the flags the binary must know
bash start-youngai-v41-flash.sh start         # finance sidecar, port 8006, about two minutes to load
```

At any point: `status`, `logs`, `test`, `domains`, `stop`. `all` runs preflight,
download, verify, build, start in one go. `start code` (or `DOMAIN=code`) serves
with the coding sidecar instead.

Two things make the big download bearable: the repository pass excludes the 40
base parts, and the parts are then fetched one at a time, each one checked
against the published `SHA256SUMS` before it is appended to the assembling file
and deleted. Peak extra disk is one part, not 113 GB, and re-running the same
command resumes.

## The ports on this box

`8000` (llama.cpp DeepSeek-V4-Flash), `8003` (Qwen3.8-Flash-Next Uncensored),
`8004` (DeepSeek V4.1 Flash Q2) and `8005` (Prism Ternary Bonsai 2 27B) are
already held by the other launchers in this home directory, and `config.toml`
references 8000-8003. The model card's own installer uses 8000. This launcher
therefore defaults to **8006**, and `preflight` refuses to start on a port that
already answers `/v1/models`. Only one of these models fits in memory at a time:
stop the resident one first.

## Commands

| Command | What it does |
|---|---|
| `preflight` | uname/arch, GB10, CUDA 13 in the ldconfig cache, MemTotal, free disk against what is still to fetch, occupied ports |
| `plan` | paths, artifact list with sizes, what is still to fetch, free space, and the exact command `start` would run. Touches nothing |
| `download` | repository pass (sidecars, binaries, `SHA256SUMS`), the 40 parts one at a time with per-part sha256, then the two official shards |
| `merge` | assemble the base from the parts alone, with a `.merge-progress` counter and a truncate-back guard against a half-written append |
| `verify` | re-check every part present, the assembled base against the published hash, the engram sizes, and that the selected sidecar has its `manifest.txt` |
| `build` | `chmod +x` the shipped binaries and fail if the binary does not advertise `--zchain`, `--engram-dir` and `--mem-budget-mb` |
| `start [domain]` | start the server with the chosen sidecar, a memory watchdog, then wait for `/v1/models` and run one smoke request |
| `stop` | stop the watchdog first, then the server; TERM, then KILL after 20 s |
| `status` | pids, elapsed time and RSS, free memory, sidecar layer counts, engram state, endpoint answer, last log lines |
| `logs` | tails the server and watchdog logs |
| `test` | one chat request in the served domain's own language, `temperature: 0` |
| `domains` | the sidecars on disk, with their gains and router-bias layer counts |
| `all` | preflight, download, verify, build, start |

Environment overrides: `DIR`, `DOMAIN`, `PORT`, `HOST_ADDR`, `ENGRAM_DIR`,
`POSTTRAIN=1`, `BUDGET_MB`, `START_FLOOR_MB`, `RUN_FLOOR_MB`, `KILL_MB`,
`WATCHDOG=0`, `READY_TIMEOUT`, `HF_BIN`, `HF_MAX_WORKERS`, `ENDPOINT`, `NO_XET=1`.

## Running it, and what a start looks like

```
[youngai 12:00:01] starting ds4-server on 0.0.0.0:8006, domain finance (loading takes about two minutes)
[youngai 12:00:01]   log: /home/leandro/models/youngai/logs/server.log
[youngai 12:00:01]   watchdog pid 1234567, stops the server if MemAvailable stays below 2500 MB
[youngai 12:00:03] waiting for the endpoint (up to 900s; the port binds only after the weights are in)
[youngai 12:02:14] loaded; 6800 MB of memory left
[youngai 12:02:15]   -> A price-to-earnings ratio is a stock's price divided by its earnings per share.
[youngai 12:02:15] ready: http://0.0.0.0:8006/v1
```

The watchdog samples `MemAvailable` every five seconds and stops the server after
two consecutive samples below 2500 MB. That is not politeness: with 113.6 GB
resident on a 121.6 GiB box, a machine that starts swapping is a machine you
cannot type into.

Two more floors come from the card's installer and are kept here: 100,000 MB
available before loading, 10,000 MB after. `start` refuses rather than crossing
either.

## Client behaviour that surprises people

- A request that omits sampling fields gets the card's recipe: `temperature` 1.0,
  `top_p` 1.0, no `min_p`, speculative decoding on. Send `"temperature": 0`
  explicitly for byte-reproducible output.
- Requests default to **non-thinking** mode. Turn thinking on per request with
  `"reasoning_effort": "high"` (or `"max"`), `"thinking": {"type": "enabled"}`,
  `"think": true`, or `"model": "deepseek-reasoner"`.
- The card warns that the quantized weights under-weight the stop decisions
  (`</think>`, EOS), so a long thinking segment can fail to close, and that
  greedy can loop verbatim inside the thinking section on questions with very few
  valid answers. Non-thinking is the safe default.
- Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`
  (OpenAI style) and `/v1/messages` (Anthropic style).

## What a sidecar is, and why five of them are 200 MB and not 200 GB

A sidecar re-solves one multiplicative gain per output channel of every expert's
down projection, plus one router bias per expert, against text from one domain.
It rescales rows that already exist and nudges the router; it adds no layers and
the kernels are the same with or without it. The card measures +2.8 to +3.7
points of Same top-1 on each domain's own held-out text, and no regression on
general English. The bare base is a complete model: `none` serves it, and the
sidecar question is a question about a few tenths of a point, not about whether
the thing runs.

## What is verified, and what is not

Verified in this repository, from the published sources:

- Every size, hash and file name above, cross-checked against the repository's
  file listing (`huggingface.co/api/models/wenzhouwu/YoungAi-DeepSeek-V4.1-Flash`),
  its `SHA256SUMS`, and its `install.sh`. The part-size arithmetic
  (24 x 2,838,915,986 plus 16 x 2,838,915,985 = 113,556,639,424) closes on the
  published assembled size and on the published hash for the assembled file.
- The installer's own semantics: the `--exclude "*.part*-of-*"` repository pass,
  per-part sha256 before append, `.merge-progress` resume, `chmod +x` after
  download, `--engram-dir` as the mandatory override, the 110,000 MB budget, the
  100,000/10,000 MB floors, the 2500 MB watchdog, and the container/port defaults
  this launcher departs from.
- The launcher in this repository, exercised on a scratch tree with a stub `hf`
  and scaled-down part sizes: assembly reproduces the expected hash exactly, a
  corrupted part is rejected and deleted while the merge point is kept, resume
  picks up from the recorded part, `verify` passes on a complete tree and fails
  on a missing sidecar, `build` fails a binary that lacks the flags, and `start`
  cleans up its pidfile when the server dies during load.
- The state of the DGX itself, read over SFTP at 2026-10-01: Ubuntu 24.04.5,
  CUDA 13 at `/usr/local/cuda-13`, `libcudart.so.13` present in the ldconfig
  cache, an `hf` CLI at `~/.local/bin/hf`, and **no** DeepSeek V4.1 or engram
  artifacts on disk, so the full 317 GB download is ahead.

Not verified, and not claimed:

- Nothing about this model has been run on the Spark. No artifact here has been
  downloaded, no request has been served, and no figure in the recipe contract is
  ours. The throughput numbers are the author's, labelled as such.
- Free disk space on the Spark. The launcher's preflight measures it; this
  repository has not, because the SFTP mount does not expose `df`.
- Whether any other engine can open this base. The card states the format carries
  a version-3 payload with a new magic (`DQV3`) and that the loader whitelists
  format versions, and that the engine source is not public. The shipped binary
  is therefore the only engine; that is an inference from the card, not a test.

## Limits worth stating plainly

- **An opaque third-party binary.** See the top of this page. There is no
  build-from-source path, no signed release, and no independent hash to check
  against.
- **The card contradicts itself on sampling plus speculation.** Section 8 says
  "Speculative decoding is greedy-only. Sampling requests fall back to plain
  decode." Section 4.4 describes sampled requests drawing drafts from the draft
  towers' distribution and verifying them by rejection sampling, and section 5.2
  publishes sampled speculative figures (34-37 t/s). Which is true of the shipped
  binary is unresolved here; measure it if the difference matters to you.
- **Post-training is an experiment.** The card is explicit: it flips decisions on
  the requests it was solved on and does not transfer across days, and one
  flipped token changes one sentence, not a chain of reasoning. It is
  fingerprinted to the finance sidecar and refused on any other.
- **All five domains are judged teacher-forced on held-out text.** None has been
  evaluated end to end on agentic coding, legal or clinical work.
- **One machine type was tested by the author**: a DGX Spark. CUDA only; the
  FP8 codebook path needs sm_89 or newer.
- **Quality is not parity.** The card's own numbers put the quantized base at
  71.7-78.6% Same top-1 against the original, depending on the domain. It is a
  1.6-bits-per-weight model that runs on one box, not a lossless copy.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| First request fails with an engram-table error after a two-minute load | `--engram-dir` missing or pointing at a directory without both shards. Check with `verify` or `status` |
| `error while loading shared libraries: libcudart.so.13` | The CUDA 13 runtime is not visible. `preflight` checks it in the ldconfig cache |
| The server refuses to start, complaining it cannot fit the budget | Something else holds memory. `8000`, `8003`, `8004` and `8005` are other launchers on this box: stop one |
| The port never answers | The port binds only after the weights are loaded. `logs` shows progress; `start` reports and cleans up if the process dies |
| The assembled file is a few bytes short | A part was appended twice or an append was interrupted. Delete the `.gguf.merging` and `.merge-progress` files and run `download` again; every part is re-checked |
| Downloads keep dying with "peer closed connection" | `NO_XET=1` switches to the plain LFS channel |
| A `sha256` mismatch on one part | The launcher deletes that part and stops. Run `download` again; the rest stays |
| A domain switch seems to have no effect | `stop` first: `start` refuses to start a second server, and `status` prints the domain the running server was started with |
| Thinking output that never closes | The card's known issue with the quantized weights and stop tokens. Use non-thinking (the default), or cap `max_tokens` |

## Attribution

The model, the quantization format (VQ-8, the 12+1 bit-plane layout, FP8
codewords), the sidecar solver, the post-training method and the engine are
Wenzhou Wu's work, published at
<https://huggingface.co/wenzhouwu/YoungAi-DeepSeek-V4.1-Flash>. The engine is a
fork of [`antirez/ds4`](https://github.com/antirez/ds4) (DwarfStar), and inherits
GGUF and q4_K from llama.cpp and GGML. The model weights derive from
`deepseek-ai/DeepSeek-V4.1-Flash`; the n-gram tables this recipe streams are two
shards of that official checkpoint. This repository contributes the operating
recipe: the artifact inventory, the launch shape that fits a Spark alongside the
other models that live on it, the checks around the download, and this page's
honest separation of what has been tested from what has been published.
