# dgx-recipe

Custom inference recipes for the NVIDIA DGX Spark (GB10, SM121, aarch64, 121.6 GiB
unified memory). One recipe per directory entry, each one carrying the exact
artifact list, the launcher, and the measured evidence behind its settings.

This repo starts with one recipe, verified end to end on a single Spark on
2026-09-26:

**Qwen3.8-Flash-Next Uncensored (Q5 SSD-PLE) with the FP8 E4M3FN PLE sidecar**,
served by the `ds4-server` Rust host from the `Baekpica/ds4-dfm-rs` fork, at
262,144 tokens of context on port 8003.

- Recipe contract: [`recipes/qwen-38-uncensored-fp8-ple-ds4.yaml`](recipes/qwen-38-uncensored-fp8-ple-ds4.yaml)
- Runbook: [`runbooks/qwen-38-uncensored-fp8-ple-ds4.md`](runbooks/qwen-38-uncensored-fp8-ple-ds4.md)
- Launcher: [`scripts/start-qwen38-uncensored-fp8.sh`](scripts/start-qwen38-uncensored-fp8.sh)

## Verified configuration

| Item | Value |
|---|---|
| Model | `Baekpica/Qwen3.8-Flash-Next-Uncensored-Mixed-Quant-SSD-PLE-GGUF` |
| Main weights | `MQ-Q5-SSD-PLE-BF16/` three shards, 32.07 + 31.81 + 19.40 GB |
| PLE sidecar | `PLE-FP8/` four `ple-fp8-*.bin` of 12.80 GB, `ple-manifest.json`, 2-byte weight scale |
| Download total | 134,475,461,654 bytes (134.48 GB / 125.24 GiB) |
| Engine | [`Baekpica/ds4-dfm-rs`](https://github.com/Baekpica/ds4-dfm-rs) `main` @ `7a78fcd`, FP8 PLE support from PR #22 |
| Build | `make cuda-spark` (SM121; also links `ds4_weight_server`) |
| Runtime split | `ds4_weight_server` weight owner + `ds4-server` Rust worker (IPC weight manifest) |
| Context | 262,144 (the artifact's own declared `qwen4exp.context_length`) |
| Banks | 1 |
| Port | 8003 |
| Sidecar logging | `dtype=FP8_E4M3FN` in the worker log |
| Disk KV | separate directory per sidecar format, 32,768 MB cap |

Measured on the Spark: `/v1/models` reports `context_length` 262144 with a
32,768 completion cap; a 47-token prompt returned its first token in 234.3 ms at
222.4 tok/s prefill.

## Why one bank at 262K

The artifact declares 262,144 as its ceiling and the FP8 short HTTP checks in the
model card passed at that context, so 262,144 is the right depth. It fits only one
bank on this hardware, and that is arithmetic rather than tuning:

- the weight owner maps all three shards into one 77.56 GiB logical model, uploads
  77.55 GiB across 93 ranges and builds a 3.62 GiB q8 aligned repack (278 tensors):
  about 81 GiB resident;
- the worker's Qwen graph plan is 23.35 GiB per bank at 262,144;
- two banks would need roughly 128 GiB against 121.6 GiB of unified memory, so the
  engine refused with `requested 2 banks but the memory quote fits 1 (banks_not_quoted)`.

One bank at 262,144 lands near 104 GiB and starts cleanly. Two banks is reachable at
196,608, which is the shape the model card's throughput numbers were measured with.

## Quick start

Run on the Spark, as the user that owns the model directory:

```bash
bash scripts/start-qwen38-uncensored-fp8.sh download   # 134.5 GB, resumable
bash scripts/start-qwen38-uncensored-fp8.sh verify     # published SHA256SUMS
bash scripts/start-qwen38-uncensored-fp8.sh build      # make cuda-spark
bash scripts/start-qwen38-uncensored-fp8.sh start      # owner + worker, port 8003
```

`stop`, `status`, `logs`, `test` and `plan` round it out, and every path, port,
context and memory knob is an environment override documented in the script header.

## Attribution

Model artifacts and the FP8 PLE runtime work are by
[Baekpica](https://huggingface.co/Baekpica); the engine is a fork of
[`antirez/ds4`](https://github.com/antirez/ds4). This repo contributes the
operational recipe: the artifact set, the launch shape that fits, and the
measurements behind it. Recipes here are provided as-is, with no warranty.
