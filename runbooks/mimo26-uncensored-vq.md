# Recipe: MiMo-V2.6-Flash-RL UNCENSORED VQ on the DS4 CUDA engine

dealignai's refusal-removed 309B omnimodal MoE through Baekpica's `mimo2-vq8`
conversion, packaged so one 128 GB DGX Spark can hold it at the model's own
1,048,576-token context: 98.5 GB of GGUF parts, a Q8_0 vision tower, a Q8_0 audio
tokenizer, a DFlash drafter, three embedded MTP blocks, and a token embedding
read a row at a time from the fast disk.

- Contract: [`recipes/mimo26-uncensored-vq.yaml`](../recipes/mimo26-uncensored-vq.yaml)
- Launcher: [`scripts/install-mimo26-uncensored-vq.sh`](../scripts/install-mimo26-uncensored-vq.sh)
- Artifact: [`Baekpica/MiMo-V2.6-Flash-RL-UNCENSORED-VQ-GGUF`](https://huggingface.co/Baekpica/MiMo-V2.6-Flash-RL-UNCENSORED-VQ-GGUF) at `2b65148d`
- Source checkpoint: `dealignai/MiMo-V2.6-Flash-RL-UNCENSORED` at `0bdb11d7`, which is `XiaomiMiMo/MiMo-V2.6-Flash-RL` at `f657ba2f` with the refusal behaviour removed at the weight level

## What this is, and what it is not

It is the operational half of the artifact: the exact file set, the volume each
part belongs on, the digests, the launch shapes and the guards that keep a wrong
or incomplete file out of the serving path.

It is not a measurement. Nothing in this recipe has been loaded or timed on the
Spark. The numbers it prints are the artifact author's, measured at engine
`2fa8c08e` on a GB10 with driver 615.71.10 and CUDA 13.4.92; the box here records
driver 580.178.04 and CUDA 13.0, so they are a reference point, not a promise.

## Artifacts

| Item | Value |
|---|---|
| Repository | `Baekpica/MiMo-V2.6-Flash-RL-UNCENSORED-VQ-GGUF`, revision `2b65148d9a95a7768ef2e553823c39149ee18c6b` (main head, 2026-10-31T03:16:14Z) |
| Parts | 53 independent GGUF parts in one directory, 98,520,657,024 bytes |
| Layers | 47 `layer-NN-experts.gguf`: 15 at R13 (1.31 GB) and 32 at R13+8 (2.12 GB) |
| Native parts | `body-nonexpert.gguf` 5.74 GB, `embed-tokens-bf16.gguf` 1.25 GB, `vision-q8_0.gguf` 0.78 GB, `audio-q8_0.gguf` 0.65 GB, `dflash-q8_0.gguf` 1.56 GB, `mtp-q8_0.gguf` 1.05 GB |
| Engine | `ds4-dfm-rs`, family `mimo2_vq`: PR #127 at `446951cc`, PR #128 at `35f5b462`; built at `2fa8c08e`, the commit the author's numbers were taken on |
| Context | 1,048,576 native (no `-c` is passed) |
| Port | 8012 |

The repository is public and not gated. Every fetch resumes.

## The fastest setup, and why it is the default

```bash
./ds4-server --cuda -m /mnt/models/mimo26-unc \
    --mtp-mode on --mtp-draft 2 \
    --host 0.0.0.0 --port 8012 \
    --model-id mimo-v2.6-flash-uncensored-vq --no-update-check
```

One serial sequence, native 1M context, the three embedded MTP blocks drafting
two tokens per verify, exact prefix reuse, no banks, no disk K/V.

The reason is that decode is one trunk pass per verify, so the only knob that
moves it is drafting:

| Configuration | Decode tok/s | Where it was measured |
|---|---:|---|
| No drafting, 2K-64K | 23.96 | author's sweep, 3 runs per variant |
| **Two MTP drafts, 2K-64K** | **26.35** | the default here |
| DFlash, 7 drafts, 32K raw prose | 17.4 | author's comparison, lost to plain decode (24.2) |

On chat prompts, where drafts are accepted more often, the default measured
35.96 tok/s writing a Rust LRU cache (105-token prompt), 36.29 rewriting a C file
(2,365-token prompt) and 25.40 on Korean prose (152-token prompt), against
24.9-25.1 without drafting.

Two honest edges:

- At the 2K frontier drafting lost to plain decode, 23.18 against 24.86, because
  one verify cycle costs more than a draft saves at short context. Everything
  from 8K up favoured drafting.
- Prefill does not care: 1,292.2 against 1,294.9 tok/s.

What is deliberately not in the fast default:

- `--mtp-draft 3`: three blocks exist, so three drafts are expressible, but every
  published number is at two and nobody measured three.
- The DFlash drafter: slower on the same hardware.
- Banks or a shorter context: they are the shapes for two concurrent streams with
  MTP off, and their gate measured reuse correctness, not throughput.
- Any disk choice at run time: after the load the trunk is resident, so NVMe
  against `/mnt` changes cold-start time, not tokens per second.

## Where the pieces go, and why

**Release directory** (`--gguf-dir`, default `/mnt/models/mimo26-unc`): all 53
parts and the tables. The engine hashes every part with `O_DIRECT` at every open
(`crates/ds4-core/src/mimo2_vq/release.rs:675` `verify_dir` -> `:557`
`part_digest`) and then reads the bytes again to copy them to the device
(`docs/mimo2-vq.md`: "Engine open copies 86.82 GiB (trunk) to the device"), so a
cold start streams the trunk twice: about 2 x 93 GB.

**Fast root** (`--dir`, default `$HOME/models/mimo26-unc`): the token embedding,
the disk K/V store when banks are used, the logs, the pid files and the verify
record. The embedding's release entry is residency `ssd-rows`
(`release.rs:83`), and `FORMAT.md` says the host reads one 8,192-byte row per
token, through the page cache while serving. It is 1.25 GB and it belongs where a
cold miss is cheap.

The split is a symlink, not a copy: `<release>/embed-tokens-bf16.gguf` points at
`<fast>/embed/embed-tokens-bf16.gguf`. The loader opens the path it read from
`release.json` and stats it, so a symlink reads exactly like a file. The launcher
hashes through the link at verify time, so the split is proven, not assumed. When
`--dir` and `--gguf-dir` are the same filesystem the split is skipped.

**Why this is measured and not assumed**, on 2026-10-31:

| Probe | NVMe root | `/mnt/models` (USB 10 Gbps) |
|---|---:|---:|
| Sequential, `dd iflag=direct bs=1M` | 2.8 GB/s | 63.7 MB/s |
| Random 8 KiB row, 300 reads, `O_DIRECT` | 98.6 us | 13,883.6 us |

Both `/mnt` figures were taken while `hf download` was writing to the same disk,
so they are a floor. The link is 10 Gbps, so the link is not the ceiling. At
63.7 MB/s the 2 x 93 GB of one cold start is about 48 minutes; at 2.8 GB/s it is
about a minute. `preflight` re-measures both roots at rest, prints both numbers,
and says which layout they support. Moving the bulk is a flag, not a code change.

## Quick start on the Spark

```bash
bash scripts/install-mimo26-uncensored-vq.sh preflight
bash scripts/install-mimo26-uncensored-vq.sh layout      # adopt a loose fetch, split the embedding
bash scripts/install-mimo26-uncensored-vq.sh download    # 98.5 GB, resumable
bash scripts/install-mimo26-uncensored-vq.sh verify      # sizes, digests, split, record
bash scripts/install-mimo26-uncensored-vq.sh build       # detached worktree at 2fa8c08e + make cuda-spark
bash scripts/install-mimo26-uncensored-vq.sh check       # the engine's own --check-config plan
bash scripts/install-mimo26-uncensored-vq.sh start       # 0.0.0.0:8012
```

`install` runs all of that in order and then starts. `plan`, `stop`, `status`,
`logs`, `test` and `bench` round it out.

`layout` exists because a fetch made with `hf download --local-dir` lands loose.
On 2026-10-31 the operator's fetch went into the shared `/mnt/models` root beside
three other artifacts; `layout` moves exactly the repository's own top-level
entries into the release directory, never anything else, and it refuses to move
anything while a Hugging Face download is still writing.

## The shapes

| Shape | Command | Plan (author's quote) | Lane |
|---|---|---:|---|
| **1M, MTP 2 (default)** | `--mtp-mode on --mtp-draft 2` | 109.79 GiB | serial |
| 1M, DFlash 7 | `--mtp <dir>/dflash-q8_0.gguf --mtp-mode on --mtp-draft 7` | 110.27 GiB | serial |
| 512K, one text bank | `-c 524288 --mtp-mode off` | 111.77 GiB | one bank + serial media |
| 256K, two text banks | `-c 262144 --max-seqs 2 --prefix-reuse partial --kv-disk-dir DIR` | 109.89 GiB | two banks + serial media |

A text bank and the serial media session each hold their own full-layer K/V rows,
so a 1M bank beside serial media is 123.2 GiB and is refused. At 1M there is no
bank shape at all, which is why the fast default is a single sequence.

MTP and banks do not combine: speculation runs on the serial lane
(`spec_lane: Serial`, `serving.rs:823-841`), and every published bank shape
carries `--mtp-mode off`. The engine sets `DS4_SERVER_CONTINUOUS=0` itself from
the resolved plan (`serving.rs:1721`), so do not set it by hand.

## The traps

**An undeclared `.gguf` beside the parts.** The loader walks the release
directory and fails on any `.gguf` that `release.json` does not declare
(`release.rs:392`, "unexpected GGUF part"). A stray layer part from the RL
release or a half-finished file is a load failure, not a warning. The launcher
counts them before every start and `verify` refuses on one.

**The directory, not a file.** `-m` takes the release directory. The tables
(`release.json`, `files.json`, the tokenizer files, the chat template) must sit
beside the parts; `release.json` is the entry point and the authority for all 53
names, sizes and digests.

**The two sibling releases.** `Baekpica/MiMo-V2.6-Flash-RL-VQ-GGUF` carries the
same recipe on the aligned original, and 38 of this release's 53 parts are its
bytes, marked `shared_with_rl`. The parts are interchangeable; the release
directory is not, because `release.json` pins this edition's source revision
(`0bdb11d7`) and the body part carries its own `o_proj`.

**MTP mode.** `--mtp-mode on` loads the embedded blocks; `auto` also loads them,
because this family's MTP support is Qualified (`serving.rs:823-841`, weights rule
at `:2560-2575`); `off` leaves the 1.05 GB part on disk. The launcher pins `on`
with two drafts so the plan is the same every time.

**Building over the operator's checkout.** The build is a detached git worktree
at `2fa8c08e` (`$HOME/ds4-dfm-rs-mimo`), because `$HOME/ds4-dfm-rs` is mid-unit on
another branch with local modifications. `build` fetches and switches the
worktree; it never touches the source tree.

## Verification

`verify` does five things, in this order:

1. Asserts every part's presence at the size `release.json` declares, so a
   truncated download is named before 90 GB of hashing starts.
2. Refuses any undeclared `.gguf` in the release directory.
3. Resolves the embedding symlink and proves the target is the published
   1,249,903,744-byte file.
4. Hashes all 53 parts in parallel against `release.json`, and every table
   against `files.json` (falling back to `SHA256SUMS`), printing one line per
   part.
5. Writes `mimo26-verify.record` under the fast root: digest, size and mtime for
   every part and table, with the embedding recorded at its real path.

`start` requires that record to still match the bytes on disk (size and mtime),
so it never re-hashes 98 GB on a normal start. The engine then re-hashes
everything itself at open, so the record is a fast pre-check, not the only check.

`sha256sum -c --ignore-missing` is deliberately not used on its own: on a
directory where none of the listed files exist it exits 0 without verifying
anything (measured in this project on 2026-10-18).

## What has not been measured

Everything about speed and quality on this box. The author's figures are in the
contract under `benchmark`, each labelled with the commit, the driver and the
clock range. When the box is free, `bench` prints the reproduction: the same
32-frontier sweep with `ds4-bench` at `2fa8c08e`, three fresh processes per
variant. Whatever it reads belongs in the contract only beside the command, the
clock range and the date.

## Ports and the single lock

8000 V4.1, 8003 Qwen3.8, 8005 Bonsai, 8006 YoungAi V4.1, 8007 GLM-5.3 VQ, 8008
YoungAi 1017, 8009 GLM-5.3 Uncensored VQ, 8010 DeepSeek 1017 Uncensored VQ, 8011
MiniMax-M3.1-Flash VQ, 8012 this artifact, 8090 MiniMax-H3 video.

The engine takes a single global lock, so only one ds4 model is resident at a
time, whatever the ports say. The H3 video host is a separate process tree on the
same GPU, and it competes for the same unified memory.

## Provenance

Xiaomi's MiMo-V2.6-Flash-RL is a 309B-parameter omnimodal MoE with 15B active: 48
layers, 9 global-attention and 39 sliding-window-128, 256 routed experts top-8
with an expert FFN of 2,048, a 152,576-token vocabulary and a 1M context. dealignai
removed its refusal behaviour at the weight level, changing only the attention
output projection of layers 34-47; experts, router, embeddings, head, vision,
audio, DFlash and MTP are byte-identical to the original. Baekpica converted it
with YoungAi's 8-dimensional shared-codebook VQ method, 13 bits on all 47 MoE
layers plus an 8-bit residual stage on 32 of them, fitted with an importance
matrix from 4.13M calibration tokens.

Fidelity on the engine, teacher-forced against the edition's own BF16 hidden
states over 24,785 assistant-turn targets: KL 0.0884, the same next token 91.77%
of the time. That is the artifact's own measurement, not this box's.

Licence: MIT for the weights, with YoungAi's notice kept in `LICENSE-YoungAi`.
The upstream model's use restrictions apply to derivatives.
