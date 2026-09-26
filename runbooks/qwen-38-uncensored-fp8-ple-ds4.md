# Recipe: Qwen3.8-Flash-Next Uncensored Q5 + FP8 PLE on the DS4 CUDA engine

**Status:** Working — verified end to end on a single DGX Spark, 2026-09-26
**Served name:** `Qwen3.8-Flash-Next-Uncensored-Mixed-Quant`
**Engine:** [Baekpica/ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs) `main` @ `7a78fcd` (fork of `antirez/ds4`), native C/CUDA plus a Rust host, no Docker
**Context / banks / port:** 262,144 / 1 / 8003
**Sidecar:** FP8 E4M3FN PLE (`DS4_QWEN_PLE_DIR`), confirmed by `dtype=FP8_E4M3FN` in the worker log

> **Recipe contract:** [`recipes/qwen-38-uncensored-fp8-ple-ds4.yaml`](../recipes/qwen-38-uncensored-fp8-ple-ds4.yaml)
> **Launcher:** [`scripts/start-qwen38-uncensored-fp8.sh`](../scripts/start-qwen38-uncensored-fp8.sh)
> **Artifacts:** [Baekpica/Qwen3.8-Flash-Next-Uncensored-Mixed-Quant-SSD-PLE-GGUF](https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Uncensored-Mixed-Quant-SSD-PLE-GGUF) (public, no token needed)

---

## What this recipe buys you

The model card for this artifact describes the runtime path but not a single
working launch shape, and the FP8 sidecar is selected by an environment variable
whose consequences are easy to get wrong. This recipe pins the whole thing: the
exact artifact set, the checksums, the two-process launch, the context that fits,
and the log lines that prove the FP8 sidecar loaded.

## Artifacts

Both halves come from the same Hugging Face repo, `Baekpica/Qwen3.8-Flash-Next-Uncensored-Mixed-Quant-SSD-PLE-GGUF`.

Main weights, `MQ-Q5-SSD-PLE-BF16/`:

| File | Bytes | sha256 |
|---|---|---|
| `...-00001-of-00003.gguf` | 32,066,883,584 | `8897179bba036cc74471711b392f8a45bea5f4e85eeb7b465609d12531b8a1fb` |
| `...-00002-of-00003.gguf` | 31,811,595,296 | `d3c658c4748ee92e44df14997e1f70860e8405fd36624a804ed3fda7172a979c` |
| `...-00003-of-00003.gguf` | 19,396,505,568 | `96c9ed051ab18c96aa07055192a5bdd1e3571f39586eaddf270a0713d671a52a` |
| `SHA256SUMS.main` | 405 | — |

PLE sidecar, `PLE-FP8/`:

| File | Bytes | sha256 |
|---|---|---|
| `ple-fp8-00001-of-00004.bin` | 12,800,098,304 | `3cffb8fcc5f070d34bff3eddc67a721b26f284b0ec71f5da5aecda8b468543a7` |
| `ple-fp8-00002-of-00004.bin` | 12,800,098,304 | `f2f63094e120bc207d6ad72117343b911a62b24ad659e99467eddf5f6ba23a36` |
| `ple-fp8-00003-of-00004.bin` | 12,800,098,304 | `a4396a17923598c1905820102e0a996f7b8f9032b5edbb30d734bcdef932ce0d` |
| `ple-fp8-00004-of-00004.bin` | 12,800,098,304 | `deeaaf2efd5f1983cfd2306a6da5135f4c2731a73d977e09de270ea097a65c16` |
| `ple-fp8-weight-scale.bf16.bin` | 2 | `c7c58bd6007672362da2106fdbfaf9f50629e4bdf8598169c598027394ef9791` |
| `ple-manifest.json` | 83,031 | `507885ac42e4a5f631a12cf95d12755980a4c61f3fc09b17f665be2a4af9312e` |

Download total: **134,475,461,654 bytes** (134.48 GB / 125.24 GiB). The repo also
publishes a BF16 `ple/` directory of four 25.6 GB files; it is **not** needed. The
loader, with `DS4_QWEN_PLE_DIR` set, reads `<override>/ple-manifest.json` and never
looks at the in-tree `ple/` path (in this checkout that branch sits at `ds4.c`
lines 35657 to 35664, with the dtype string printed at line 35687).

Model metadata read from the downloaded GGUF: `qwen4exp.context_length` 262144,
48 blocks, embedding 2560, vocab 248320, 24 attention heads, 2 KV heads, RoPE
base 1e7, one MTP block, and `qwen4exp.vision` metadata for 27 vision blocks —
but no projector is published in this repo, so image input is off.

## Install and run

On the Spark, as the user that owns the model directory:

```bash
git clone https://github.com/Baekpica/ds4-dfm-rs ~/ds4-dfm-rs
bash scripts/start-qwen38-uncensored-fp8.sh download   # 134.5 GB, resumable
bash scripts/start-qwen38-uncensored-fp8.sh verify     # published SHA256SUMS
bash scripts/start-qwen38-uncensored-fp8.sh build      # make cuda-spark
bash scripts/start-qwen38-uncensored-fp8.sh start      # owner, then worker on 8003
```

`stop`, `status`, `logs`, `test` and `plan` complete the set; `help` prints every
override. The launcher defaults to `$HOME`-relative paths and needs no edits.

Expect these lines, in this order:

1. `ds4_weight_server: ready manifest=/tmp/.../qwen38-uncensored-q5.weights.manifest ranges=371`
2. `ds4: Qwen graph allocated: ctx=262144 prefill=8192 plan=23.35 GiB`
3. `ds4-server-rs: listening on 0.0.0.0:8003 model_id=Qwen3.8-Flash-Next-Uncensored-Mixed-Quant`
4. `dtype=FP8_E4M3FN`

## The context and bank question

The artifact declares 262,144, and that is what this recipe serves. It fits
**one** bank on a 121.6 GiB Spark:

| Component | Resident |
|---|---|
| Weight owner: 3 shards mapped as one logical model | 77.56 GiB |
| Weight owner: uploaded base ranges | 77.55 GiB across 93 ranges |
| Weight owner: q8 aligned repack (278 tensors) | 3.62 GiB |
| Worker graph plan per bank at ctx=262144 | 23.35 GiB |

One bank lands near 104 GiB and starts cleanly. Asking for two produced the
engine's own refusal, quoted verbatim from the log:

```
requested: reuse=partial max_seqs=2 mtp=auto ctx=262144 floor=2G disk=true
effective: reuse=partial max_seqs=1 mtp=auto weights=true ctx=262144 floor=2G disk=true
error: requested 2 banks but the memory quote fits 1 (banks_not_quoted)
ds4-server-rs: fitted serving plan rejected
```

The launcher reacts to that rejection by retrying once with `--max-seqs 1` and
saying so. Ladder, cheapest change first:

| Want | Setting |
|---|---|
| 262,144 with one bank (this recipe) | `MAX_SEQS=1` |
| Two banks, card's measured serving shape | `CTX=196608` |
| 524,288 (reduced shape) | `MAX_SEQS=1 SERVER_FORK=0 SERVER_FORK_PARTIAL=0 COALESCE_MAX=1 PREFILL_CHUNK=1024 MAXTOK=256 CTX=524288` |

The 524,288 rung follows the model card's one-bank YaRN run (verified there at
524,240 prompt tokens plus one generated token, 215.4 tok/s prefill), which the
card describes as verifying near-full 512K execution and not long-context
quality, multi-bank serving, MTP decode or sustained throughput. It is not
attempted by this recipe.

A bank is a serving slot, not a speed knob: one bank holds one in-flight
sequence, so concurrent requests serialize, while two banks could overlap. The
throughput figures on the model card were measured on the two-bank 196,608 shape,
so do not read them as this one-bank configuration's numbers.

## Client configuration (open-grok)

```toml
[model.qwen38-uncensored-fp8]
model = "Qwen3.8-Flash-Next-Uncensored-Mixed-Quant"
name = "Qwen 3.8 Flash Next Uncensored Q5 + FP8 PLE (262K)"
base_url = "http://<spark-address>:8003/v1"
api_backend = "chat_completions"
api_key = "dummy"
context_window = 262144
max_completion_tokens = 32768
supports_images = false
supports_tool_result_images = false
```

The `model` string must match what the server advertises; confirm with
`curl -s http://127.0.0.1:8003/v1/models` on the Spark. That endpoint reports
`context_length` 262144, a 32,768 completion cap, and support for `tools`,
`tool_choice` and `reasoning_effort`. The model emits reasoning first: with a
one-token budget the answer arrives as `reasoning_content` and `content` stays
empty, which is expected rather than a fault.

## Measured

First request against a freshly started one-bank server: 47 prompt tokens, first
token at 234.3 ms, prefill 222.4 tok/s. That is a single short request used as a
smoke test, not a throughput claim.

## Operational notes

- A refused plan used to leave the weight owner holding ~81 GiB. The launcher now
  stops both processes when startup fails, and removes a stale owner IPC socket
  before rebinding.
- Use a separate KV disk directory per main model and sidecar format: BF16 and FP8
  snapshots describe different weights and a cross-format restore is rejected.
- `DS4_SESSION_GRAPH_FIT=0` is a fit-check override, not a guarantee of fit.
- The BF16 to FP8 choice is one variable: unset `DS4_QWEN_PLE_DIR` and the runtime
  loads the in-tree BF16 `ple/` instead, which needs ~95 GiB more disk.

## Attribution

Model artifacts, the FP8 PLE runtime support (PR #22) and the published
measurements are by [Baekpica](https://huggingface.co/Baekpica); the engine is a
fork of [`antirez/ds4`](https://github.com/antirez/ds4). This recipe adds the
operational shape and the local verification recorded above.
