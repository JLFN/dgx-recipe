# Gemma-4 26B A4B VQ — operator runbook (DGX Spark)

The contract is [recipes/gemma4-26b-a4b-vq.yaml](../recipes/gemma4-26b-a4b-vq.yaml);
this is the operator's account: why the non-obvious settings exist, what a start
looks like, and where the edges are.

## What the box serves

A text-only Gemma-4 26B A4B converted to the R13+8 dense-VQ format and served by
the `gemma4_vq` family of `ds4-server`. 30 layers, 25 sliding-window-1024 and 5
full-attention; 128 routed experts per layer in a 13-bit VQ stage plus an 8-bit
GVR8 residual plane; attention and dense gate/up in Q4_K, dense down in Q8_0, the
token table and the tied head in BF16. 25.2B parameters total, 3.8B active,
9.95 GiB on disk.

## The artifact is a directory, not a file

`release.json` names three GGUF parts and the engine reads each one by filename
from the directory it is handed. Point `-m` at the directory:

    /mnt/models/gemma-4-26b-a4b-vq-gguf/
      release.json, files.json, config.json, recipe.json, SHA256SUMS, chat_template.jinja
      gemma4-vq8-trunk.gguf          8.00 GiB   23,756 tensors, resident
      gemma4-vq8-embedding-BF16.gguf 1.38 GiB        1 tensor,  ssd-rows
      gemma4-vq8-mtp-Q4_K.gguf       0.58 GiB       48 tensors, resident when enabled

The embedding part is the 262,144 x 2,816 BF16 token table, and it is also the
output head: the gather and the logits read one table. It is marked `ssd-rows`,
so its rows come from the file while the server runs.

## Where the directory should live

This repository's rule is measured, not aesthetic: a part read once at load may
sit on the roomy volume, a part read while the server runs belongs on the fastest
device. Here the *whole* directory currently sits on `/mnt/models`, which is
`/dev/sda1` — a USB-attached "OSC PSSD" that reports `rotational=1` — while `/`
is the NVMe. The trunk is read once; the embedding is read one row per token for
the life of the server, so it is the part paying for that volume.

Nothing has measured the difference yet, so treat it as a lever rather than a
verdict:

    # 9.95 GiB to the NVMe; the parts must stay together in one directory
    cp -a /mnt/models/gemma-4-26b-a4b-vq-gguf $HOME/models/

Then point the launcher at it (`--dir $HOME/models/gemma-4-26b-a4b-vq-gguf`) and
compare decode at a fixed prompt. Until that pair of runs exists, the recipe
keeps the artifact where it is.

## Build

    cd ~/ds4-dfm-rs && make cuda-spark

One trap cost a build on this box: a non-interactive SSH shell has no Rust
toolchain on `PATH`, so `make` dies at the Rust link step with
`cargo: No such file or directory` (exit 127) after the whole C/CUDA half has
already built. `cargo` lives in `~/.cargo/bin` and is only on a login shell's
path:

    ssh 192.168.1.91 'bash -lc "cd ~/ds4-dfm-rs && make cuda-spark"'

The C and CUDA objects stay valid, so a retry after the failure only redoes the
Rust link.

## Actions

    bash scripts/start-gemma4-26b-a4b-vq.sh preflight   # host, gpu, port, memory, release
    bash scripts/start-gemma4-26b-a4b-vq.sh verify      # every published digest
    bash scripts/start-gemma4-26b-a4b-vq.sh check       # engine --check-config, no weights opened
    bash scripts/start-gemma4-26b-a4b-vq.sh start       # 0.0.0.0:8013, two banks
    bash scripts/start-gemma4-26b-a4b-vq.sh test        # one smoke request
    bash scripts/start-gemma4-26b-a4b-vq.sh stop | status | logs | bench | plan

`check` takes 15 s at one bank and about two minutes at two banks on this box: it
opens the release, validates every GVQ1 header and fits the plan. It writes
`check.json` beside the artifact, which is where the memory quote below comes
from.

## Two banks: parallel requests

The family declares the persistent bank lane with qualified bank support
(`crates/ds4-core/src/serving.rs:987-993`). Each bank owns its own KV caches of
both layouts over graph 0's scratch; every cache row is position-indexed, so a
bank's prefix is exact and partial reuse needs no checkpoint slots. Two banks
therefore means two conversations resident and being served at once, not two
models: the engine still holds one model lock.

Measured here on 2026-11-07 with `--max-seqs 2`, ctx 262144, BF16 full-layer KV:

| Term | Bytes | GiB |
|---|---:|---:|
| shared weights | 10,064,857,472 | 9.37 |
| per bank (KV + session state) | 5,997,854,720 | 5.59 |
| scratch | 879,866,368 | 0.82 |
| floor | 4,294,967,296 | 4.00 |
| **total, two banks** | **27,235,400,576** | **25.37** |
| available to the engine | 127,813,386,240 | 119.03 |
| headroom | 100,577,985,664 | 93.66 |

One bank quotes 21,237,545,856 B; the second bank adds its 5.59 GiB and nothing
else, because the weights are shared.

What banks do and do not buy:

- **They do** admit a second request instead of queueing it. Prefill of the
  second bank overlaps the first bank's decode.
- **They do not** raise aggregate decode. The GPU is one, so two streams split
  the decode rate (the GLM sibling measured 24.2 tok/s on one stream and 12.1 on
  each of two). Choose one bank when single-stream latency is what matters.
- **They forbid drafting.** The drafter runs on the serial lane only; the bank
  driver has no draft/verify round for this family
  (`crates/ds4-core/src/serving.rs:999-1004`). The launcher refuses
  `--mtp` with more than one bank rather than starting a server whose drafts
  never fire.

`--max-seqs 1` is the serial fallback and the lane the artifact card's numbers
describe.

## What a start looks like

    gemma4: starting ds4-server on 0.0.0.0:8013 (ctx 262144)
    gemma4:   lane:    GPU only, weights copied to the device
    gemma4:   release: /mnt/models/gemma-4-26b-a4b-vq-gguf (9.95 GiB; trunk 8.00 GiB copied to the device)
    gemma4:   log:     /mnt/models/gemma-4-26b-a4b-vq-gguf/gemma4.log
    gemma4: ready:     http://0.0.0.0:8013/v1/models after N s

The served id is the `--model-id` value, `gemma-4-26b-a4b-vq`. State files
(`gemma4.pid`, `gemma4.log`, `gemma4.lock`, the verify record, `check.json`) live
beside the artifact, so `--dir` moves them with it.

## Limits worth remembering

- **Qualified context is unset.** Servers boot at 262,144 and answered a
  248,182-token prompt on this box, but no gate covers past the 64K sweep.
- **Text only.** The checkpoint's vision tower is not in the release; audio is
  declared with zero tensors. Do not wire either.
- **The drafter is present, not qualified**: draft 6 lost to plain decode at a
  13K prompt, its verify blocks stay on the tile attention kernel, and its
  exact-path byte gate fails.
- **Lossy by construction**: 23.38% mean relL2 over the 60 routed expert stacks,
  23.3772-24.1997% per stack. Two reference probes (28 and 1,243 tokens) measure
  cosine 0.9906/0.9901; there is no perplexity or benchmark.
- **The Hub page lies about size.** Its GGUF panel parses the BF16 embedding part
  alone and reports 0.7B parameters; the text model is 25.2B.
- **One DS4 model per box.** The engine holds one global lock; ports 8000-8012
  belong to the other launchers in this home.
- **`ds4-server` keeps a 384 MiB free-memory floor** and refuses to listen,
  naming the deficit, when the serial session cannot hold it.

## Troubleshooting

- `cargo: No such file or directory` at the link step — non-login shell; use
  `bash -lc`.
- `no verification record matches the bytes on disk` — the artifact moved or was
  refetched; run `verify`.
- `another ds4 model is running` — one model lock; `stop` the other, or
  `ALLOW_SHARED=1` if that is genuinely what you want.
- `only N MB available, need 40000` — the start floor; free memory before
  starting, or lower `START_FLOOR_MB` deliberately.
- Port 8013 already bound — `status` and `logs` name the owner.
