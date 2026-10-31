# Recipe: MiniMax-H3 rendered directly at 1344x768

A full-resolution render with no draft and no upscale: the DiT runs at the
output canvas itself, the H3 video VAE decodes it, and the H3 soundtrack is
muxed as AAC. The contract is
[`recipes/minimax-h3-direct-1344.yaml`](../recipes/minimax-h3-direct-1344.yaml).

This route exists because the two-stage route cannot give "higher resolution
and the same content": the LTX refiner starts from sigma 0.909375, which keeps
9.1 percent of the draft and is free to redraw the rest, and the measured
result is a different animal (sharpness 5.54 at SSIM 0.777 against the same
latent decoded without it). The direct canvas has no such trade: what the
model draws is what you get, and there is no second stage to change it.

## Measured on the reference box

gx10-e1d2, NVIDIA GB10, aarch64, 121.6 GiB unified memory, one render at a
time under `tools/host_memory_guard.py`. Base MiniMax-H3 at 20 steps, the
NVFP4-AWQ Qwen3-VL 32B prompt encoder, VSA 0.9, tensor attention and bf16
GEMM in the decoder.

| phase | seconds |
|---|---|
| prompt encode (NVFP4, cold) | 15.610 |
| DiT load (66.3 GB on disk) | 41.068 |
| denoise, 20 steps at 1344x768 | 502.929 |
| audio decode | 3.145 |
| video VAE decode, 124 frames | 66.630 |
| mux | 0.958 |
| sum | 630.340 |
| wall | 631.5 |

Per step: 25.15 s. Memory: 72.79 GiB available at the lowest sample, PSI full
avg10 peaked at 3.19 percent, the guard never tripped.

Output:

- `/home/leandro/ds4-sol-h3/outputs/direct-1344x768.mp4`
- from this host:
  `/run/user/1000/gvfs/sftp:host=192.168.1.91,user=leandro/home/leandro/ds4-sol-h3/outputs/direct-1344x768.mp4`
- 4,684,968 bytes, sha256 `6cf5a33dfc87ffa0f09485a96f61778d907213d3f6b528e7065eb060e7819844`
- H.264 1344x768 at 24 fps, 5.175 s, AAC 32 kHz stereo, full decode clean
- sharpness at 2.0 s: 17.54 Laplacian variance. For scale: the upscaled 384p
  draft decoded in H3 space 14.11; the LTX refiner 5.54; the transfer-only
  decode through the adapter and LTX VAE 3.82; a bicubic 2x of the draft 3.34.

## Files this route needs from Hugging Face

Two repositories, nothing else. Sizes are the repository's own.

| repo | revision | path | bytes |
|---|---|---|---|
| MiniMaxAI/MiniMax-H3 | `42ed227ee7df40d41602854ae760620d6eb651fe` | `FL2VA/transformer/*` (13 shards + index + config) | 66,280,524,863 |
| MiniMaxAI/MiniMax-H3 | same | `FL2VA/video_vae/source/*` | 10,415,549,484 |
| MiniMaxAI/MiniMax-H3 | same | `FL2VA/audio_vae/*` | 605,431,281 |
| MiniMaxAI/MiniMax-H3 | same | `FL2VA/tokenizer/*` | 11,492,078 |
| MiniMaxAI/MiniMax-H3 | same | `FL2VA/processor/*` | 11,498,352 |
| MiniMaxAI/MiniMax-H3 | same | `FL2VA/model_index.json` | 719 |
| Comfy-Org/MiniMax-H3 | `0543966fbdce5ba05709a8f2031c94bdba629b4a` | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15,687,142,551 |

Total 93,011,639,328 bytes. The NVFP4 Qwen file's sha256 is
`35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6`.

Download with the Hugging Face CLI:

```
huggingface-cli download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include "FL2VA/transformer/*" "FL2VA/video_vae/source/*" \
            "FL2VA/audio_vae/*" "FL2VA/tokenizer/*" \
            "FL2VA/processor/*" "FL2VA/model_index.json" \
  --local-dir /mnt/h3-artifacts/MiniMax-H3

huggingface-cli download Comfy-Org/MiniMax-H3 \
  --revision 0543966fbdce5ba05709a8f2031c94bdba629b4a \
  --include "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
  --local-dir /mnt/h3-artifacts/Comfy-MiniMax-H3
```

The engine reads `<model_dir>/FL2VA/…`, which is the repository's own layout:
`transformer/`, `video_vae/source/`, `audio_vae/`, `tokenizer/`, `processor/`,
`model_index.json` directly under `FL2VA/`. No renaming is required.

### What this route does not download, and why

- `FL2VA/text_encoder/*.safetensors`, 14 shards, about 65.9 GB: the NVFP4
  encoder replaces it, and `--text-nvfp4` never opens that directory
  (`crates/ds4-core/src/minimax_h3/text_forward.rs:270-282`).
- `FL2VA/Ref2VA/*` and `transformer_ref/*`: only a request carrying ordered
  references reads that partition.
- `FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA`: the four-step
  distillation that the draft route uses; a 20-step render does not need it.
- Everything from `Lightricks/LTX-2.5`: the dev transformer, the distilled
  LoRA, the video VAE, the audio VAE, the INT8 Gemma tower and the INT8
  connector. This route has no LTX stage and no Gemma.
- `LBH-123-AI/Minimax_h3_latent_Upscaler` and
  `Efficient-Large-Model/H3-to-LTX-Latent-Adapter`: nothing is upscaled or
  transferred.

## The command

```
cd $HOME/ds4-sol-h3
H3_DIT_SOL=0.9 H3_VAE_ATTENTION=tensor H3_VAE_GEMM=bf16 \
python3 tools/host_memory_guard.py --max-gib 120 --high-gib 120 \
    --reserve-gib 2 --trip-gib 2 --psi-full-limit 0 --timeout 3600 \
    --log scratch/direct1344.memory.jsonl -- \
    ./h3-rs \
    -d /mnt/h3-artifacts/MiniMax-H3 \
    --text-nvfp4 /mnt/h3-artifacts/Comfy-MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
    -p "A red fox trots through fresh snow at dawn, crunching footsteps." \
    --width 1344 --height 768 --frames 124 --steps 20 --seed 42 \
    -o outputs/direct-1344x768.mp4
```

`H3_DIT_SOL=0.9` is VSA: 90 percent of the attention blocks are skipped over
64-token tiles, with the text and audio prefix exempt. `H3_VAE_ATTENTION=tensor`
and `H3_VAE_GEMM=bf16` are the decoder's fast arm; both are the settings the
reference uses, and both change rounding, so a comparison run must keep them
fixed.

## Verify

```
ffprobe -v error -show_entries format=duration,size \
  -show_entries stream=codec_name,width,height,sample_rate,channels \
  -of default=noprint_wrappers=1 outputs/direct-1344x768.mp4
ffmpeg -v error -i outputs/direct-1344x768.mp4 -f null - && echo DECODE_OK
sha256sum outputs/direct-1344x768.mp4
```

## Cost, and when to prefer the draft route

502.9 s of denoise buys detail the 384p draft does not contain. The same prompt
by the draft route costs 21.4 s of denoise, 2.8 s of upscaler and 60-67 s of H3
decode, and reaches 14.11 sharpness instead of 17.54. Both routes preserve
content; neither uses the LTX refiner, which does not.
