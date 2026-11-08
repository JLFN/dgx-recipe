# Recipe: MiniMax-H3 stage-1 clip from the command line

This is the operator's account of the first clip rendered on the Spark by the
engine's own CLI, `h3-rs`, with no HTTP server and no ComfyUI in the path. The
contract is [`recipes/minimax-h3-cli.yaml`](../recipes/minimax-h3-cli.yaml).
The server path is a separate recipe: [`minimax-h3`](minimax-h3.md).

The clip is stage 1 only: the 50-block DiT denoises the packed sequence, the H3
video VAE decodes it, the audio VAE decodes the soundtrack and ffmpeg muxes the
MP4. The latent path (upscaler, adapter, LTX refiner, LTX VAE) exists in the same
tree but is not composed into this run.

## Which tree, and which binary

Three checkouts matter on the reference box, and they are not interchangeable:

| | |
|---|---|
| `/home/leandro/ds4-dfm-rs` | `main` at `35f5b462`. The mainstream tree. Its `h3-rs` binary was removed by a forced rebuild, so nothing there produced this clip. |
| `/home/leandro/ds4-sol-h3` | `feat/sol-h3` at `19937e37`, with uncommitted working-tree changes (`Makefile`, `crates/ds4-cli/Cargo.toml`, `crates/ds4-cli/src/bin/h3-rs.rs`, `crates/ds4-core/src/minimax_h3.rs`, plus an untracked `crates/ds4-cli/src/bin/h3-sol-rs.rs`). This is the tree that holds the recent work and the binary that ran. |
| `/data/ds4-dfm-rs` (this host) | `feat/h3-apiplant`, branched from `feat/sol-h3` at `cec392fb`, with the `--save-latent` and `--save-audio` flags type-checked but not yet built on the Spark. |

The render used the pre-rebuild `/home/leandro/ds4-sol-h3/h3-rs` (5,946,992 B,
built 2026-10-29 17:15). The file on disk was replaced at 10:31:04, while the
render was still running, by a later rebuild (6,900,248 B); the running process
kept the inode it started with. Record the binary identity when it matters,
because the name alone does not identify these builds.

## The command

```sh
cd $HOME/ds4-sol-h3

python3 tools/host_memory_guard.py \
  --max-gib 120 --high-gib 120 \
  --reserve-gib 2 --trip-gib 2 \
  --psi-full-limit 0 --timeout 1800 \
  --log scratch/fox.memory.jsonl -- \
  $HOME/ds4-sol-h3/h3-rs \
  -d /mnt/h3-artifacts/MiniMax-H3 \
  -p "A red fox trots through fresh snow at dawn, crunching footsteps." \
  --width 512 --height 512 --frames 56 --steps 20 --seed 42 \
  -o /mnt/h3/fox.mp4
```

Flags that decide the cost: `--width/--height` the canvas, `--frames` aligned up
to 5+17n, `--steps` the denoise evaluations, `--reuse` the DiT core-refresh
window, `--attention tensor` (default) or `scalar` (byte-exact, much slower),
`--vae-attention` and `--vae-gemm` the decoder's own switches.

The same binary also writes the handoff files the latent path consumes:

```sh
  --save-latent PATH   raw BF16 [channel][time][height][width], 24 channels
  --save-audio PATH    raw little-endian F32 PCM, interleaved
```

They were added in this branch and were not used for the clip below.

## What it produced, measured

| phase | seconds |
|---|---:|
| text encode | 28.983 |
| DiT load (50 blocks) | 38.835 |
| denoise, 20 of 20 steps | 56.672 |
| audio decode | 1.552 |
| frame decode (H3 video VAE, 56 frames) | 39.747 |
| mux | 0.962 |
| **sum** | **166.751** |

Wall clock from process start 10:27:23 CET to MP4 write 10:30:11 CET is
168.7 s, so the phases account for essentially all of it. Per step the denoise
is 2.834 s, below the 6.97 s implied by the recorded 139.418 s run at the same
shape; one run on a warm host cannot explain that difference, and no A/B was
run, so it is recorded as unexplained rather than as a gain.

Output, verified on the box:

```text
/mnt/h3/fox.mp4
278,232 bytes
sha256 2bace19f542b2b044b63cf32f04869484b8cbf46d5352b0273ba7a6962a1f35f
h264 512x512 24/1, aac 32000 Hz 2 channels, duration 2.333333 s
ffmpeg -v error -i /mnt/h3/fox.mp4 -f null -   -> exit 0, no output
```

From this host the file is readable through the SFTP mount at
`/run/user/1000/gvfs/sftp:host=192.168.1.91,user=leandro/mnt/h3/fox.mp4`.

Memory: the guard admitted the run with 118.96 GiB available, the minimum
observed during load was 80.47 GiB, PSI full avg10 peaked at 3.39 %, and the
2 GiB trip was never reached.

## What is not measured

- The refined clip. The upscaler, adapter, refiner and LTX VAE are ported and
  gated, but this CLI does not call them.
- The release shape: 672x384x124 drafted, doubled to 1344x768, cropped to 16
  latent frames.
- The stage-2 prompt cache. The build on the box exited 1 with
  `missing model.layers.0.self_attn.q_proj.comfy_quant`: it was pointed at the
  BF16 Gemma4 release, while the loader expects the INT8 convrot checkpoint.
  The refined path is blocked on that artifact or on a BF16 read arm.

## Reproducing

1. Use the sol-h3 tree, not the mainstream one, and record the binary's size and
   mtime before the run.
2. Run under the guard with the flags above.
3. Verify with `ffprobe` and a full `ffmpeg -f null -` decode; the phase lines in
   stdout are the timing record.
4. Keep the log next to the clip; the guard log carries the memory samples.
