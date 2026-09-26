# dgx-recipe

Custom inference recipes for the NVIDIA DGX Spark (GB10, SM121, aarch64, 121.6 GiB
unified memory). One recipe per directory entry, each one carrying the exact
artifact list, the launcher, and the measured evidence behind its settings.

This repo starts with one recipe, verified end to end on a single Spark on
2026-09-26:

**Qwen3.8-Flash-Next Uncensored (Q5 SSD-PLE) with the FP8 E4M3FN PLE sidecar**,
served by the `ds4-server` Rust host from the `Baekpica/ds4-dfm-rs` fork, at
196,608 tokens of context in the model card's two-bank shape, with a one-bank
262,144 mode, on port 8003.

- Recipe contract: [`recipes/qwen-38-uncensored-fp8-ple-ds4.yaml`](recipes/qwen-38-uncensored-fp8-ple-ds4.yaml)
- Full usage guide: [`docs/USAGE.md`](docs/USAGE.md) - install, shapes, log levels, the HTTP API, vision, benchmarking, memory, troubleshooting, and an honest list of what is verified and what is not
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
| Context | 196,608 with two banks (default, the card's production shape); 262,144 with one bank |
| Banks | 2 by default; 1 in the 262k and 512k shapes |
| Port | 8003 |
| Sidecar logging | `dtype=FP8_E4M3FN` in the worker log |
| Disk KV | separate directory per sidecar format, 32,768 MB cap |

Measured on the Spark: `/v1/models` reports `context_length` 262144 with a
32,768 completion cap; a 47-token prompt returned its first token in 234.3 ms at
222.4 tok/s prefill.

## Shapes

The launcher ships three named shapes, chosen as the first argument to `start`,
so a change of context and bank count is one word:

| Shape | Context | Banks | MTP | Notes |
|---|---:|---:|---|---|
| `196k` | 196,608 | 2 | on | Default. The model card's canonical command and the shape behind every throughput figure it publishes. Two sequences can be in flight. |
| `262k` | 262,144 | 1 | on | The artifact's declared `qwen4exp.context_length`, verified working on this setup. Deeper prompts, but concurrent requests serialize. |
| `512k` | 524,288 | 1 | off | The card's reduced YaRN shape. Output capped at 256 tokens and reuse disabled. The card claims execution, not long-context quality. |

Individual knobs win over the shape, so `MAXTOK=8192 bash scripts/start-qwen38-uncensored-fp8.sh start 196k`
keeps two banks at 196,608 with an 8192-token cap. `bash scripts/start-qwen38-uncensored-fp8.sh shapes`
prints this table resolved from the launcher's own profile code.

## Why the default is two banks at 196,608 and not three banks at 262,144

The card gives one canonical serving command and calls it the production server
shape: two banks, 196,608 context, `--mtp-draft 2`, 8,192-token prefill chunks and
a 512 MiB PLE cache with 16 workers. Everything it measures uses that shape, so it
is the default here.

Depth beyond it is possible but has to trade concurrency, and the ceiling the
artifact declares is not the ceiling this box will serve with two banks. The
engine repository's FP8 document reports that both FP8 Q5 servers passed short
HTTP checks at 262,144 with two banks, but on the reference Spark that same
two-bank request was refused with `banks_not_quoted`, with the weight owner
holding about 81 GiB resident and each bank's graph plan measuring 23.35 GiB at
262,144. Whether that is the commit, the owner reserve or the free-memory baseline
at the time is unresolved here, so this repo treats 262,144 as a one-bank shape
rather than claiming one bank is inherent to that context. The launcher retries
once at one bank if the quote refuses a plan, and says so when it does.

## Quick start

Run on the Spark, as the user that owns the model directory:

```bash
bash scripts/start-qwen38-uncensored-fp8.sh download   # 134.5 GB, resumable
bash scripts/start-qwen38-uncensored-fp8.sh verify     # published SHA256SUMS
bash scripts/start-qwen38-uncensored-fp8.sh build      # make cuda-spark
bash scripts/start-qwen38-uncensored-fp8.sh start      # two banks, 196,608, port 8003
```

Add the shape when you want something else: `start 262k` for 262,144 in one bank,
`start 512k` for the reduced 524,288 shape. `stop`, `status`, `logs`, `shapes`,
`test` and `plan` round it out, and every path, port, context and memory knob is an
environment override documented in the script header.

## Attribution

Model artifacts and the FP8 PLE runtime work are by
[Baekpica](https://huggingface.co/Baekpica); the engine is a fork of
[`antirez/ds4`](https://github.com/antirez/ds4). This repo contributes the
operational recipe: the artifact set, the launch shape that fits, and the
measurements behind it. Recipes here are provided as-is, with no warranty.
