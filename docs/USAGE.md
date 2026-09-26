# Using this recipe

Everything needed to download, build, serve, monitor and troubleshoot the
Qwen3.8-Flash-Next Uncensored Q5 SSD-PLE model with its FP8 E4M3FN PLE sidecar,
on a single NVIDIA DGX Spark.

- [1. What you get](#1-what-you-get)
- [2. Requirements](#2-requirements)
- [3. Install](#3-install)
- [4. Starting a server](#4-starting-a-server)
- [5. Shapes: context and banks](#5-shapes-context-and-banks)
- [6. Log levels](#6-log-levels)
- [7. The HTTP API](#7-the-http-api)
- [8. Vision](#8-vision)
- [9. Benchmarking](#9-benchmarking)
- [10. Day-to-day operations](#10-day-to-day-operations)
- [11. Memory, and why the bank count is what it is](#11-memory-and-why-the-bank-count-is-what-it-is)
- [12. The PLE page cache](#12-the-ple-page-cache)
- [13. Clients: open-grok and anything OpenAI-compatible](#13-clients-open-grok-and-anything-openai-compatible)
- [14. Environment reference](#14-environment-reference)
- [15. Troubleshooting](#15-troubleshooting)
- [16. Measured on the reference box](#16-measured-on-the-reference-box)
- [17. What is verified and what is not](#17-what-is-verified-and-what-is-not)

---

## 1. What you get

One script, `scripts/start-qwen38-uncensored-fp8.sh`, that does five things:

| Command | What it does |
|---|---|
| `download` | Fetches the three main GGUF shards (83.28 GB) and the FP8 PLE sidecar (51.20 GB) from Hugging Face |
| `verify` | Checks both sets against the published `SHA256SUMS`, plus the presence of the six files the loader requires |
| `build` | Runs `make cuda-spark` in the engine checkout (SM121), which also links `ds4_weight_server` |
| `start` | Starts the weight owner and then the Rust worker host, on one port, in a chosen shape and log level |
| `stop` / `status` / `logs` / `shapes` / `bench` / `plan` / `test` | Operate and inspect what is running |

No containers, no Python environment beyond the `hf` CLI used for downloading,
no web UI. The server is a plain HTTP API on your LAN.

## 2. Requirements

- Hardware: one DGX Spark or equivalent GB10 / SM121 machine with 121.6 GiB of
  unified memory. The recipe has only ever been run on that class of box.
- Disk: 134.48 GB for the artifacts, plus a few GB for the build tree. The
  launcher refuses to download below 145 GB free.
- The engine checkout: [`Baekpica/ds4-dfm-rs`](https://github.com/Baekpica/ds4-dfm-rs),
  `main` at `7a78fcd` or later. Earlier revisions have no FP8 PLE support; the
  launcher checks `ds4.c` for `DS4_QWEN_PLE_DIR` and refuses to run without it.
- Build toolchain: `cargo` on `PATH`, CUDA toolkit able to target `sm_121`.
- The `hf` CLI, either on `PATH` or at `~/hfenv/bin/hf`. `HF_BIN=<path>` overrides.
- A public Hugging Face repo: no token is required, and none is stored anywhere
  by this recipe. Exported `HF_TOKEN` in your own shell is passed through.

## 3. Install

```bash
# 1. the engine
git clone https://github.com/Baekpica/ds4-dfm-rs ~/ds4-dfm-rs

# 2. the launcher from this repo
cp scripts/start-qwen38-uncensored-fp8.sh ~/

# 3. the artifacts (134.5 GB, resumable: re-run after an interruption)
bash ~/start-qwen38-uncensored-fp8.sh download

# 4. check the published checksums before trusting the files
bash ~/start-qwen38-uncensored-fp8.sh verify

# 5. build (SM121)
bash ~/start-qwen38-uncensored-fp8.sh build
```

`download` places everything under
`~/models/Qwen3.8-Flash-Next-Uncensored-Mixed-Quant-SSD-PLE-GGUF/`:

```
MQ-Q5-SSD-PLE-BF16/     three main shards + SHA256SUMS.main   (83.28 GB)
PLE-FP8/                four ple-fp8-*.bin, ple-manifest.json,
                        ple-fp8-weight-scale.bf16.bin, SHA256SUMS (51.20 GB)
```

The BF16 sidecar directory that the same repo publishes is deliberately **not**
downloaded: it is 95 GiB more, and with `DS4_QWEN_PLE_DIR` set the loader reads
the override directory's own manifest and never looks at the in-tree `ple/`
path. Set `SIDECAR=bf16` if you want it anyway.

## 4. Starting a server

The simplest form asks two questions and starts:

```bash
bash ~/start-qwen38-uncensored-fp8.sh start
```

```
Which shape?
  196k  two banks at 196,608   the card's measured shape, two callers at once
  262k  one bank at 262,144    deepest verified context, callers queue
  512k  one bank at 524,288    reduced YaRN shape, output capped at 256

shape [196k]:

Which log level?
  normal  the standard lines only
  debug   admissions, prefill and decode profiles, MTP accept trace
  trace   debug plus per-step timing, top logits and a token trace

log level [normal]:
```

Skip either question by passing them:

```bash
bash ~/start-qwen38-uncensored-fp8.sh start 262k            # shape only
bash ~/start-qwen38-uncensored-fp8.sh start 262k debug      # shape and log level
PROFILE=262k LOGLEVEL=debug bash ~/start-qwen38-uncensored-fp8.sh start
```

With no terminal attached, or with `ASK_SHAPE=0` / `ASK_LOGLEVEL=0`, the defaults
in `DEFAULT_PROFILE` and `DEFAULT_LOGLEVEL` are used silently, which is what a
cron job or systemd unit should rely on.

What start does, in order:

1. Resolves the shape and log level, then prints both along with the model,
   sidecar, KV directory and endpoint.
2. Refuses to continue if the port already has a listener.
3. Starts `ds4_weight_server` and waits for its
   `ready manifest=... ranges=371` line. This process maps the shards, uploads
   the weights and stays alive; it is the 80+ GiB process in `nvidia-smi`.
4. Starts `ds4-server` with `DS4_QWEN_PLE_DIR` pointing at the sidecar, waits for
   the endpoint to answer `/v1/models`, then reports the `dtype=` line, which
   must read `FP8_E4M3FN`.
5. If the memory quote refuses the plan, it says so, retries once with one bank,
   and tells you it did (`AUTO_RETRY=0` disables the retry).

Example of a healthy start:

```
== starting weight owner
  owner pid 1380284, log /tmp/qwen38-unc-fp8/owner.log
  owner ready

== starting ds4-server
  server pid 1380399, log /tmp/qwen38-unc-fp8/server.log
  serving on port 8003
  runtime reports: dtype=FP8_E4M3FN
  effective: reuse=partial max_seqs=1 mtp=auto weights=true ctx=262144 floor=2G disk=true
```

## 5. Shapes: context and banks

| Shape | Context | Banks | MTP | What it is for |
|---|---:|---:|---|---|
| `196k` | 196,608 | 2 | on | The model card's canonical command and the shape behind every throughput figure it publishes. Two requests can be in flight at once. |
| `262k` | 262,144 | 1 | on | The artifact's declared `qwen4exp.context_length` ceiling. Deepest verified context. Concurrent requests queue behind each other. |
| `512k` | 524,288 | 1 | off | The card's reduced YaRN shape. Also caps output at 256 tokens and disables warm and fork reuse. The card claims execution, not long-context quality. |

A **bank** is one request slot: one sequence's KV cache, recurrent state, PLE
gather state and sampling rows. One bank serves one request at a time; a second
request queues and runs as soon as the row frees. Two banks let two requests
overlap. Banks are a concurrency purchase, not a per-request speed purchase.

A profile only sets what you have not set yourself, so any single knob wins:

```bash
MAXTOK=8192 bash ~/start-qwen38-uncensored-fp8.sh start 196k   # two banks, smaller output cap
CTX=131072 MAX_SEQS=2 bash ~/start-qwen38-uncensored-fp8.sh start
```

`bash ~/start-qwen38-uncensored-fp8.sh shapes` prints the three resolved from the
launcher's own code, so the table cannot drift.

## 6. Log levels

The server has no verbosity setting. This build exposes individual trace switches
in `ds4.c`, and the launcher bundles the useful ones:

| Level | Switches | What you get |
|---|---|---|
| `normal` | none | The standard engine lines, artifact-selection notices, the host's one line per rolling call |
| `debug` | `DS4_ADMIT_DEBUG`, `DS4_PREFILL_PROFILE_DETAIL`, `DS4_DECODE_PROFILE_DETAIL`, `DS4_MTP_ACCEPT_TRACE`, `DS4_CONT_PROFILE` | Admission and plan decisions (the reasoning behind refusals like `banks_not_quoted`), prefill and decode phase breakdowns, drafted versus accepted speculation, lane profile |
| `trace` | `debug` plus `DS4_TOKEN_TIMING`, `DS4_TOKEN_TRACE`, `DS4_TRACE_TOP` | Per-decode-step milliseconds, top-ten logits every step, full token trace. Loud, and the log grows fast |

Start prints the switches it resolved, so the log can be grepped for them. Watch
it with `bash ~/start-qwen38-uncensored-fp8.sh logs` or `tail -f /tmp/qwen38-unc-fp8/server.log`.

Do not benchmark with `debug` or `trace` on: measure first, then diagnose.

## 7. The HTTP API

The server speaks OpenAI Chat Completions, OpenAI Responses and Anthropic
Messages. All of these were exercised against the reference box:

```bash
BASE=http://127.0.0.1:8003
MODEL=Qwen3.8-Flash-Next-Uncensored-Mixed-Quant

# model list: id, context_length, completion cap, supported parameters
curl -sS $BASE/v1/models

# chat completion
curl -sS -X POST $BASE/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Say OK."}],"max_tokens":16}'

# streaming (Server-Sent Events)
curl -N -X POST $BASE/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Count to three."}],"max_tokens":32,"stream":true}'

# Anthropic Messages shape
curl -sS -X POST $BASE/v1/messages -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","max_tokens":16,"messages":[{"role":"user","content":"Say OK."}]}'

# OpenAI Responses shape
curl -sS -X POST $BASE/v1/responses -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","input":"Say OK.","max_tokens":16}'

# lane state for the last request
curl -sS $BASE/v1/stats
```

There is no authentication, so any non-empty API key string works with clients.

### The response carries the performance numbers

This is the only place they exist. The log never records them.

```json
"usage":   {"prompt_tokens": 43, "completion_tokens": 15, "total_tokens": 58,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 43}},
"timings": {"ttft_ms": 249.7, "prefill_tokens": 43, "prefill_cached_tokens": 0,
            "prefill_tok_s": 172.8, "decode_tok_s": 29.1, "tok_per_step": 1.78}
```

| Field | Meaning |
|---|---|
| `ttft_ms` | Time to first token: prefill plus any PLE page reads. The field that swings from 0.2 s to a minute depending on cache warmth |
| `prefill_tok_s` | Prompt tokens per second during prefill. Meaningless on very short prompts, where fixed costs dominate |
| `prefill_cached_tokens` | Prompt tokens served from the prefix cache |
| `decode_tok_s` | Generation speed after the first token. Roughly constant for a given shape and model |
| `tok_per_step` | Tokens committed per decode step. Above 1.0 means MTP speculative decoding is landing extra tokens, about 1.78 here |

This model thinks before it answers. With a small `max_tokens` the whole budget
can go to `reasoning_content` and `content` comes back empty, which is expected,
not a fault. Give it room: 200 tokens rather than 16.

## 8. Vision

Image input works with no flag, no projector file and no configuration. The
artifact carries its vision tower in the GGUF (`qwen4exp.vision.present = 1`,
27 blocks, embedding 1152, output 2560, patch 16, merge 2), and the qwen4exp
graph binds those tensors from the model's own map; the server logs
`Qwen vision ready` when it prepares them. The `--vision <mmproj>` flag exists
for other model families and is not used here.

To check an endpoint for yourself, the recipe ships a probe:

```bash
bash ~/vision-test.sh                          # uses ~/vision-test.png
bash ~/vision-test.sh ~/photo.jpg
VISION_MAXTOK=256 bash ~/vision-test.sh ~/photo.jpg
```

It posts the image inline as a base64 data URI and prints the answer, the
reasoning and the timings. Use at least 200 output tokens, for the reason in the
previous section.

Two cold/warm numbers from the reference box, same image, cache already warm:
a first-ever look at an image cost 58.5 s to first token, while the same image
sent again cost 0.57 s, three times in a row.

## 9. Benchmarking

```bash
bash ~/start-qwen38-uncensored-fp8.sh bench                    # 3 requests, 32 output tokens
BENCH_N=5 BENCH_MAXTOK=256 bash ~/start-qwen38-uncensored-fp8.sh bench
BENCH_PROMPT="your own prompt" bash ~/start-qwen38-uncensored-fp8.sh bench
BENCH_BASE=http://192.168.1.91:8003 bash ~/start-qwen38-uncensored-fp8.sh bench   # bench a remote box
```

It reads the `timings` fields out of each response and prints a row per request
plus means. Judge the second and later rows: the first request against new text
or a new image is disk-bound on PLE pages, and that is a property of the model,
not of your settings.

For model-level sweeps, the engine repository ships `ds4-bench` (context
frontiers with CSV output) and `ds4-perf` (workload evidence runs, the harness
that reads `DS4_PLE_LATENCY_STATS`). Those are the tools behind the numbers on
the model card; this recipe's `bench` is the quick server-side check.

## 10. Day-to-day operations

```bash
bash ~/start-qwen38-uncensored-fp8.sh status   # pids, endpoint, dtype, effective shape
bash ~/start-qwen38-uncensored-fp8.sh logs     # tail both logs
bash ~/start-qwen38-uncensored-fp8.sh stop     # worker first, then owner
```

`stop` kills the worker before the owner on purpose: the owner holds the weight
ranges the worker is using. Stopping only the owner leaves the worker pointing at
memory that is gone.

Files it writes:

```
/tmp/qwen38-unc-fp8/owner.log              weight owner log
/tmp/qwen38-unc-fp8/server.log             worker log, where trace output lands
/tmp/qwen38-unc-fp8/owner.pid, server.pid  pids, used by stop and status
/tmp/qwen38-unc-fp8/qwen38-uncensored-q5.weights.manifest   IPC weight manifest
~/ds4-dfm-rs/qwen38-uncensored-q5-kv-<sidecar>/            disk KV cache, capped at 32768 MB
```

Set `RUNTIME=<dir>` to move the first group somewhere else.

## 11. Memory, and why the bank count is what it is

The engine's own accounting, from `crates/ds4-core/src/serving.rs`:

```
total = weights + banks x per_bank + mtp_state + scratch + checkpoint_pool
        + ple + media_reserve + floor        <=  MemAvailable
```

Measured on the reference box:

| Component | Size |
|---|---|
| Owner: three shards mapped as one logical model | 77.56 GiB |
| Owner: uploaded base ranges | 77.55 GiB across 93 ranges |
| Owner: q8 aligned repack, 278 tensors | 3.62 GiB |
| Worker: graph plan per bank at ctx 262,144 | 23.35 GiB |

At 262,144 with one bank the total lands near 104 GiB and starts. Asking for two
banks at that context produced:

```
requested: reuse=partial max_seqs=2 mtp=auto ctx=262144 floor=2G disk=true
effective: reuse=partial max_seqs=1 mtp=auto weights=true ctx=262144 floor=2G disk=true
error: requested 2 banks but the memory quote fits 1 (banks_not_quoted)
ds4-server-rs: fitted serving plan rejected
```

Why lowering the context can help: the per-bank plan is `W + C x ctx`, where
almost every term is scaled by the prefill capacity `p` (8,192) and therefore
context-independent, and exactly one term scales with context
(`ds4.c`, `qwen4exp_graph_bytes_estimate`: the full-attention KV plus indexer
keys). Dropping from 262,144 to 196,608 cuts that one part to three quarters and
leaves the rest alone.

## 12. The PLE page cache

The 51.2 GB of per-layer embeddings live on disk and are read in 4 KiB pages as
prefill needs them. `PLE_CACHE_MB` is the RAM budget for that page cache.

- The engine accepts exactly `512`, `1024` or `2048` MiB and silently replaces
  anything else with 2048 (`ds4.c`, `qwen4exp_ple_cache_mb_valid`). The launcher
  validates the value before use, so an unsupported number is reported rather
  than quietly ignored.
- This recipe defaults to `2048`, which is both the engine's own default and the
  value every measurement in the card's FP8 document was taken with.
- It shortens prefill when pages are warm. It cannot help the first read of new
  text, which is disk-bound either way.
- Cost: 1.5 GiB more RAM than `512`. Irrelevant at one bank, possibly relevant
  when trying to fit a second bank, where `1024` or `512` may be needed.

## 13. Clients: open-grok and anything OpenAI-compatible

open-grok, both shapes pre-configured:

```toml
[model.qwen38-uncensored-fp8]
model = "Qwen3.8-Flash-Next-Uncensored-Mixed-Quant"
name = "Qwen 3.8 Flash Next Uncensored Q5 (Spark, 196K, two banks)"
base_url = "http://<spark-address>:8003/v1"
api_backend = "chat_completions"
api_key = "dummy"
context_window = 196608
max_completion_tokens = 32768
supports_images = true
supports_tool_result_images = false

[model.qwen38-uncensored-fp8-262k]
model = "Qwen3.8-Flash-Next-Uncensored-Mixed-Quant"
name = "Qwen 3.8 Flash Next Uncensored Q5 (Spark, 262K, one bank)"
base_url = "http://<spark-address>:8003/v1"
api_backend = "chat_completions"
api_key = "dummy"
context_window = 262144
max_completion_tokens = 32768
supports_images = true
supports_tool_result_images = false
```

`context_window` must describe the shape the server is actually running, since
the two entries share one port and one model id. `supports_images` is true
because image input works, as described above; `supports_tool_result_images` is
false conservatively, which only means an image produced by a tool is moved into
a following user message.

Anything else that speaks OpenAI Chat Completions, OpenAI Responses or Anthropic
Messages works too: point it at `http://<spark-address>:8003/v1` with any
non-empty key.

Security note: the endpoint is unauthenticated and `start` binds `0.0.0.0`, so
anything on your network can use the model. `HOST_ADDR=127.0.0.1` restricts it to
the box.

## 14. Environment reference

Every value the launcher reads. Command-line arguments win over the environment,
and an explicit knob wins over the shape profile.

| Variable | Default | Purpose |
|---|---|---|
| `REPO_DIR` | `$HOME/ds4-dfm-rs` | Engine checkout |
| `MODEL_ROOT` | `$HOME/models/Qwen3.8-Flash-Next-Uncensored-Mixed-Quant-SSD-PLE-GGUF` | Where the artifacts live |
| `VARIANT` | `MQ-Q5-SSD-PLE-BF16` | Published packaging directory |
| `SIDECAR` | `fp8` | `fp8` or `bf16` |
| `HF_BIN` | autodetected | `hf` CLI path |
| `HF_MAX_WORKERS` | `8` | Download concurrency |
| `NEED_GB` | `145` | Free space the download insists on |
| `PROFILE` | empty | `196k`, `262k`, `512k` |
| `DEFAULT_PROFILE` | `196k` | What Enter means at the shape prompt |
| `ASK_SHAPE` | `1` | `0` never asks |
| `LOGLEVEL` | `normal` | `normal`, `debug`, `trace` |
| `DEFAULT_LOGLEVEL` | `normal` | What Enter means at the log prompt |
| `ASK_LOGLEVEL` | `1` | `0` never asks |
| `HOST_ADDR` | `0.0.0.0` | Bind address |
| `PORT` | `8003` | Port |
| `CTX` | shape | Context length |
| `MAXTOK` | shape | `-n`, completion cap |
| `MAX_SEQS` | shape | Bank count; empty means the runtime default |
| `MTP_DRAFT` | shape | `0` disables MTP entirely |
| `MEM_FLOOR_GB` | `2` | `--mem-floor-gb` |
| `PREFILL_CHUNK` | shape | `DS4_QWEN_PREFILL_CHUNK` |
| `COALESCE_MAX`, `COALESCE_MAX_TOKENS`, `COALESCE_WAIT_MS` | shape | Request coalescing |
| `SERVER_WARM`, `SERVER_FORK`, `SERVER_FORK_PARTIAL` | shape | Prefix reuse machinery |
| `PLE_CACHE_MB` | `2048` | PLE page cache; 512, 1024 or 2048 |
| `PLE_WORKERS` | `16` | PLE page workers |
| `KV_DISK_MB` | `32768` | Disk KV budget |
| `KV_DIR` | repo, per sidecar | Disk KV directory |
| `USE_OWNER` | `1` | `0` runs the worker without the weight owner |
| `RESERVE_GB` | `32` | Owner VMM reserve |
| `AUTO_RETRY` | `1` | Retry once at one bank when the quote refuses |
| `RUNTIME` | `/tmp/qwen38-unc-fp8` | Logs, pidfiles, weight manifest |
| `VISION` | `0` | Path to an mmproj, for families that need one; not needed here |
| `BENCH_BASE`, `BENCH_N`, `BENCH_MAXTOK`, `BENCH_PROMPT`, `BENCH_TIMEOUT` | | See section 9 |
| `VISION_MAXTOK`, `VISION_QUESTION` | | See section 8 |

## 15. Troubleshooting

**The start refuses the shape.** Look for `banks_not_quoted` or
`memory quote cannot host the mix at one bank` in the log. Either accept the
launcher's automatic retry at one bank, or set `CTX=196608`, or lower
`PLE_CACHE_MB` to `1024`/`512` to free 0.5 to 1.5 GiB.

**Port already in use.** `status` shows both pids; `stop` clears them. If a
process was started outside the launcher, no pidfile exists, so `stop` reports
"not running" and you will need to kill it by pid. Confirm nothing else on the
box uses 8003: the recipe's earlier launchers use 8000, 8002 and 8003.

**The owner is running but no worker ever appears.** The start died between the
two. `stop` then `start` again. A stale owner IPC socket is removed by the
launcher before each start, so this is not usually the cause.

**`dtype=BF16` in the log when you wanted FP8.** `DS4_QWEN_PLE_DIR` did not
reach the worker, or the override directory is incomplete: it must hold
`ple-manifest.json`, the four `ple-fp8-*.bin` files and
`ple-fp8-weight-scale.bf16.bin`, and `verify` checks exactly that.

**First token takes a minute.** Expected on a first read of new text or a new
image: the PLE pages come off the SSD. Repeated prompts drop to a fraction of a
second. Raise `PLE_CACHE_MB` to 2048, keep a stable system prompt, and reuse
conversations. `debug` level's `DS4_PREFILL_PROFILE_DETAIL` shows where the time
goes.

**The answer is empty and `finish_reason` is `length`.** The model spent the
whole budget on `reasoning_content`. Raise `max_tokens`.

**Speed looks much worse than the model card.** The card's numbers are for
2,000 to 65,000 token prompts on the two-bank 196,608 shape. Short prompts and
one-bank shapes are not comparable; check `tok_per_step` is around 1.78, which
tells you MTP is working.

**An unsupported `PLE_CACHE_MB` seemed to be ignored.** It was: the engine
replaces anything outside 512/1024/2048 with 2048. The launcher now reports it.

**A power user knob does nothing.** Many `DS4_*` switches in `ds4.c` belong to
the benchmark harness rather than the server. The launcher only sets ones that
were read in the serving path.

## 16. Measured on the reference box

DGX Spark, GB10, 121.6 GiB, 262,144 context, one bank, FP8 sidecar, `PLE_CACHE_MB=2048`.

| Measurement | Value |
|---|---|
| Warm decode, text prompt, 128 tokens | 29.9 tok/s, 4.49 s wall, three runs within 10 ms |
| Warm time to first token | 228 ms |
| Tokens per decode step | 1.78 |
| Cold first request, unseen text prompt | 1.97 s ttft |
| Cold first request, unseen image | 58.5 s ttft |
| Warm repeated image | 0.57 s ttft, 0.586/0.573/0.572 s over three runs |
| First image, decode | 19.7 tok/s, 1.85 tok/step |
| Model list reports | `context_length` 262144, completion cap 32768 |

For comparison, the model card's published mean decode for this FP8 artifact is
28.96 tok/s, which the 29.9 above matches.

## 17. What is verified and what is not

Verified on the reference box:

- Download, checksums and the six-file sidecar precondition.
- `make cuda-spark` produces `ds4`, `ds4-server`, `ds4-bench`, `ds4-agent`,
  `ds4-eval` and `ds4_weight_server`.
- One bank at 262,144 starts at `dtype=FP8_E4M3FN`; two banks at that context are
  refused.
- The HTTP surface: chat completions, streaming, Anthropic Messages, OpenAI
  Responses, model list, stats.
- Image input working with no projector and no flag, on a fresh image.
- The timings fields in the response, and the warm/cold behaviour above.
- The launcher's own behaviour: shape and log-level resolution, the interactive
  prompts on a terminal, override precedence, cache validation, and every path
  in the script that does not require a loaded model.

Not verified, and stated as such:

- Two banks at 196,608 has never been run on this box, only documented by the
  authors. It may fit; the arithmetic in section 11 is why it is not certain.
- The 512k shape has not been attempted here.
- 2048 against 512 MiB of PLE cache has not been compared head to head. The
  default follows the engine and the card's sweeps, not a measurement here.
- `debug` and `trace` were verified as wired switches, not run against a loaded
  model.
- `USE_OWNER=0` has not been tried.
- Image input inside a tool-result message has not been tested.
