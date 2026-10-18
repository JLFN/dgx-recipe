# Recipe: DeepSeek-V4.1-Flash-1017 UNCENSORED VQ on the DS4 CUDA engine

dealignai's uncensored DeepSeek-V4.1-Flash-1017 through YoungAi's published
`vq8sh14-q4k-mtpnative` recipe, packaged so one 128 GB DGX Spark can hold it: a
510 GB FP8 checkpoint becomes a 113.56 GB base GGUF that carries the model's own
three-tower DSpark drafter, plus a 971 MB vision sidecar and the 1017 Engram
tables read from disk. The context stays at the native 1,048,576.

- Contract: [`recipes/ds41-1017-uncensored-vq.yaml`](../recipes/ds41-1017-uncensored-vq.yaml)
- Installer: [`scripts/install-ds41-1017-uncensored-vq.sh`](../scripts/install-ds41-1017-uncensored-vq.sh)
- Artifact: [`Baekpica/DeepSeek-V4.1-Flash-1017-UNCENSORED-VQ-GGUF`](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-UNCENSORED-VQ-GGUF) at `610a2d71`
- Source checkpoint: `dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8` at `c19ca5ca`, itself `deepseek-ai/DeepSeek-V4.1-Flash-1017` at `1bae0e10` with dealignai's weight edit in layers 10-36

## What this is, and what it is not

It is the operational half of the artifact: the exact file set, the assembly, the
launch shape, the four digests that separate this release from the three other
1017 packagings that share its file names, and the guard that keeps a wrong file
out of the serving path.

It is not a measurement. Nothing in this recipe has been loaded or requested on
the Spark. The card says so itself: the conversion is verified, the weights pass
the engine's loader check, and serving on DGX Spark has not been measured. Every
speed and fidelity figure below belongs to V4.1 Flash with the same recipe, or to
dealignai's FP8 checkpoint, and each is labelled with the artifact it came from.

The provenance matters more than usual. The source checkpoint is DeepSeek's 1017
model with the refusal behaviour removed by a weight-level edit: 27 of the 48
shards differ, covering layers 10-36, and within those layers only the attention
output projection and the shared expert's down projection changed per layer, in
weights and block scales, with no fine-tune, no adapter, and no tokenizer or
template change. dealignai's own summary of the FP8 checkpoint is MMLU 83.77
against 87.63 for the model it edits, and 320 of 320 HarmBench prompts answered
at reasoning effort `max`. Those are its scores for the FP8 source. The card for
this GGUF warns in turn that quantizing the FP8 bundle carries the quantization
error on top of the uncensoring, and that compliance can regress at low bit
widths.

## Artifacts

| Item | Value |
|---|---|
| Repository | `Baekpica/DeepSeek-V4.1-Flash-1017-UNCENSORED-VQ-GGUF`, revision `610a2d71400313f63228589bf4cea4182ad360e9` (created 2026-10-18T14:08:51Z, weights 14:43:29Z, head of main) |
| Payload | 317,601,046,990 bytes (317.6 GB / 295.8 GiB) across 51 files; 40 of them are the base parts |
| Base | `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf`, 113,556,639,424 bytes from 40 parts (24 x 2,838,915,986 then 16 x 2,838,915,985), SHA-256 `2680747e…` |
| Vision | `DeepSeek-V4.1-Flash.vision.gguf`, 971,066,016 bytes, SHA-256 `3a8b194a…` |
| Engram | `deepseek-engram/model-00047/48-of-00048.safetensors`, 101,535,150,936 and 101,537,926,640 bytes, SHA-256 `76a05106…` and `9dc51433…` |
| Runtime | `ds4-dfm-rs` built with `make cuda-spark`; DeepSeek V4.1 support is that repo's PR #80, vision PR #83 |
| Context | 1,048,576 native, from the model metadata; `-c` is never passed |
| Port | 8010 |

The repository is public and not gated, so no token is required. Every fetch
resumes: re-run the same command.

## Where the pieces go, and why

`--gguf-dir` (default `/mnt/models/ds41-1017-unc`) holds the 40 parts and the base
they assemble into. The base is read exactly once at load and then stays resident
at about 106 GiB, so disk speed stops mattering after the first minutes and the
roomy volume is the right home for it.

`--dir` (default `$HOME/models/ds41-1017-unc`) holds the vision sidecar, the logs
and the pid files. The tower is read from disk when an image arrives, on the first
image after a cold start or an eviction. If this release's sidecar already sits
under another state directory, the installer finds it there and uses it rather
than fetching 1 GB again: it is name-and-size identical to the older V4.1 file, so
each candidate is hashed (971 MB, about a second) and only a copy whose digest is
`3a8b194a…` is adopted, while a size-correct file holding V4.1 bytes is named and
passed over. `--vision PATH` names a copy explicitly and is then taken as given,
which leaves the digest to `verify`.

`--engram-dir` holds the two n-gram tables, which are read in rows on the tokens
that hit an n-gram. By default the installer looks for a copy of the published
sizes under `$HOME/models` — the boxes here already carry these 1017 tables for
the other 1017 recipe — and reuses it, which saves a 203 GB fetch. Each candidate
is checked both as it stands and one level deeper, because a copy fetched by this
repository's own path can sit one directory in (see the next trap). The card gives
the same advice and the same `--exclude deepseek-engram/*` idiom.

Assembly happens on the roomy volume, one part at a time: each part is hashed
against the release's `SHA256SUMS`, appended, and then deleted, so the peak extra
space is one part (2.84 GB) rather than a second copy of the base. That is the
card's own loop, with the hash check added. It prints one line per part — part
number, bytes so far, seconds elapsed — and says before the final read that the
whole base is about to be hashed once. Hashing 2.84 GB and appending it takes a
few seconds, so an unadorned loop of forty of them looked like a hang; the first
run of this installer on the Spark was reported as exactly that on 2026-10-18.

An interrupted assembly is diagnosed rather than guessed at: the loop deletes
each part as it appends it, so a base file that is neither absent nor the
published size cannot be continued. `assemble` says so and gives the three
commands that recover it (delete the wreck, re-fetch the parts it consumed, run
`assemble` again).

## Quick start on the Spark

```bash
bash scripts/install-ds41-1017-uncensored-vq.sh preflight
bash scripts/install-ds41-1017-uncensored-vq.sh download    # 317.6 GB, or 114.5 GB with the tables reused
bash scripts/install-ds41-1017-uncensored-vq.sh assemble    # 40 parts -> the base, each hash-checked
bash scripts/install-ds41-1017-uncensored-vq.sh verify      # reads everything; writes the verification record
bash scripts/install-ds41-1017-uncensored-vq.sh build       # make cuda-spark
bash scripts/install-ds41-1017-uncensored-vq.sh start       # 0.0.0.0:8010, under a memory watchdog
```

`install` runs all of that in order and then starts. `plan`, `stop`, `status`,
`logs` and `test` round it out, and every path, port and memory knob is a flag or
an environment variable documented in the script header. `--no-vision` serves text
only; `--mtp-mode off|auto|on` controls the embedded drafter and defaults to the
card's `auto`.

## The trap: four files, two names, one size

Three different silent mismatches are possible here, and none of them is visible
by looking at the files:

| File | This release | What else answers to that name |
|---|---|---|
| `deepseek-engram/model-0004{7,8}-of-00048.safetensors` | 1017 tables: `76a05106…`, `9dc51433…` | the older V4.1 tables, same names, same sizes, `824db488…` and `976330f4…` |
| `DeepSeek-V4.1-Flash.vision.gguf` | `3a8b194a…`, the same bytes the 1017 VQ release ships | the older V4.1 sidecar, same size, 532,623,100 bytes different |
| `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf` | `2680747e…` | the other 1017 packagings' bases: `9f3afe6a…` and `0da4ae2d…`, same byte size |

The engine takes the Engram file names from the GGUF header and loads whatever
sits at those names, so a directory holding the V4.1 tables runs a mismatched
model without a word. The installer therefore names the wrong digests when it
sees them rather than reporting "unknown": `verify` says which set it found, and
the vision and engram digests are checked directly.

## The second trap: the Engram tables arrive one directory too deep

The two tables are the only payload of this release that the repository keeps in
a subdirectory, and the Hugging Face CLI preserves the repository path under
`--local-dir`. Fetching `--include "deepseek-engram/*"` into a directory therefore
writes `<dir>/deepseek-engram/model-0004{7,8}-of-00048.safetensors`, not
`<dir>/model-0004{7,8}-of-00048.safetensors`, while the parts, `SHA256SUMS` and the
vision sidecar (all at the repository root) land exactly where the script looks
for them.

Measured on 2026-10-18 against `huggingface_hub` 1.23.0 with this same repository:
a fetch of a 264 KB subdirectory file with `--include "docs/*" --local-dir X`
produced `X/docs/img/…`, and a top-level include produced `X/<file>`. The first
run of this installer on the Spark fetched all 203 GB of tables that way and then
refused them, because the size check reads `<dir>/model-0004{7,8}-of-00048.safetensors`:

```
ds41unc: the Engram tables are not the published sizes; re-run to resume
```

`download` and `verify` now adopt that layout instead of failing on it: if the
tables are complete one level deep and the top level is empty, the script moves
them up with `mv`. The two directories are on one filesystem, so this is a rename
— no byte is copied and nothing is re-fetched, which matters at 203 GB. If a mount
point does sit between them the move is refused and the script says so rather than
silently copying the tables. `status` names the nested layout too, and `start`
points at `verify` instead of calling the tables missing.

Two consequences worth knowing:

- A copy that a previous fetch left nested is still reusable as it stands, so the
  reuse search checks each candidate directory one level deeper as well and uses a
  match in place; it never rearranges a directory that belongs to another
  deployment.
- Only these two files are affected. Nothing else in this release lives in a
  subdirectory, and the repository that the other 1017 recipe fetches from
  (`deepseek-ai/DeepSeek-V4.1-Flash-1017`) keeps the same shards at its root, which
  is why that launcher never met this trap.

## Why start demands a verification record

Hashing the four identity files costs a full read of the payload, and the earlier
shape of this launcher paid it on every `status`, which made the script look hung.
The work is now done once, by `verify`, which writes `.verified` in the state
directory with the digests it proved plus the size and mtime of the base, the
sidecar and both tables. `start` refuses to serve unless that record still matches
the bytes on disk, so swapping a table set for one of the same size is caught by
the mtime, and the expensive read happens once per fetch instead of once per
start.

`verify` itself is deliberately explicit and counted: it checks the 40 parts (or
the assembled base) one by one against `SHA256SUMS`, then the vision digest, then
both table digests. It does not rely on `sha256sum -c --ignore-missing` alone,
because on a directory where none of the listed files exist that command exits 0
without verifying anything — measured on 2026-10-18 while building this recipe.

## What a start looks like

```
starting ds4-server on 0.0.0.0:8010 (mapping 113.6 GB; the card reports about two minutes)
  base:    /mnt/models/ds41-1017-unc/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf
  engram:  $HOME/models/ds41-1017/deepseek-engram
  vision:  $HOME/models/ds41-1017-unc/DeepSeek-V4.1-Flash.vision.gguf
  log:     $HOME/models/ds41-1017-unc/ds41unc.log
```

Then the readiness wait, a smoke request that reads the model id back from
`/v1/models` rather than assuming it, and a memory report. The watchdog stops the
server after two consecutive 5-second samples below 2500 MB of `MemAvailable`,
which beats a box that has started swapping and cannot be typed into.

Nothing in this sequence has been run on the reference Spark for this artifact.

## Numbers from other artifacts

| Figure | Value | Belongs to |
|---|---|---|
| 2K-64K sweep, embedded MTP | 1,039.65 prefill / 38.17 decode tok/s (median of three runs) | V4.1 Flash, same recipe, card link |
| Same sweep, later build `ae40aa1d` | 1,222.4 prefill tok/s at 16K; 54.10 decode for code, 65.85 for a table | V4.1 Flash |
| Same top-1 as the original | 78.61 code, 78.52 English, 69.08 medicine | V4.1 Flash, per YoungAi |
| MMLU | 83.77 against 87.63 for the base checkpoint | dealignai's FP8 checkpoint |
| HarmBench-320, effort `max` | 320 / 320 complied | dealignai's FP8 checkpoint |

Speed should land near the V4.1 Flash figures, because decode is bound by bytes
read per token and this release reads the same bytes; drafter acceptance depends
on the generated text and can differ; fidelity and compliance have to be measured
again on these weights. Treat the first start on your box as the measurement.

## Limits

- No measurement of any kind exists for this artifact: no load, no throughput, no
  time to first token, no fidelity, no compliance score.
- Abliterated by construction, and lossy on top of it: the routed experts are
  vector-quantized at roughly 1.6 bits per weight, and dealignai warns that
  quantizing its FP8 bundle keeps the quantization error and can regress
  compliance further. It is meant for research such as refusal-mechanism study,
  red-teaming and robustness evaluation, 18+, lawful use, with your own safety
  layers before any deployment.
- No domain sidecar. YoungAi's V4.1 sidecars were solved against a different base
  and do not fit these weights, so `--zchain` is never passed.
- Thinking is off by default for this weight recipe because it under-weights stop
  decisions; long thinking segments can fail to close.
- One ds4 model at a time: the engine takes a single global lock.
- The custom VQ format needs ds4-dfm-rs or YoungAi's engine. llama.cpp, Ollama and
  Transformers cannot read it.

## Troubleshooting

- `download` reports a missing `--include` option: the Hugging Face CLI is too old
  to route the parts, the sidecar and the tables to different volumes. Upgrade
  `huggingface_hub`.
- `verify` names the V4.1 tables: point `--engram-dir` at a fresh directory and let
  `download` fetch this release's copies, or find a 1017 directory and point
  `--engram-dir` there.
- `start` refuses with "no verification record matches the bytes on disk": run
  `verify`. That refusal is the guard against serving a table set that was swapped
  after verification.
- `assemble` stops with "does not match SHA256SUMS": a part is truncated or came
  from another 1017 packaging. Delete the parts and fetch again.
- The port never answers: check `MemAvailable` first — the base is about 106 GiB
  resident and another owner will starve it. Then check the log for the mapping
  progress; the card reports about two minutes for 113.6 GB.
- A refused start because another model is resident: the engine's global lock is
  doing its job. Stop the other launcher, or pass `--allow-shared` only when you
  know what is running.

## Attribution

The recipe and the format are YoungAi's (Wenzhou Wu, MIT): the quantizer and the
converter at `7fb26cc`, the vision sidecar converter at `e4ea078`, the DQVL v3 VQ
payload format, and the reference artifact
`wenzhouwu/YoungAi-DeepSeek-V4.1-Flash`. The uncensored checkpoint is dealignai's,
MIT. The weights are DeepSeek's, MIT. The engine is `ds4-dfm-rs`, the Rust-host
continuation of [`antirez/ds4`](https://github.com/antirez/ds4), with the V4.1 port
by [JLFN](https://github.com/JLFN) (PR #80, vision PR #83). The quantization route
and the packaging are Baekpica's. This repository contributes the operational
recipe: the artifact list, the assembly, the digest guards, and the honest
statement of what has not been measured. Provided as-is, with no warranty.
