# Recipe: Prism Ternary Bonsai 2 27B (qwen35) on the DS4 CUDA engine

**Status:** Working on the reference RTX 4070 SUPER box (start, request, stop verified 2026-10-01). Not yet started on the Spark: this session has no SSH access to it, so every Spark number below is a projection from the engine's own measured line, labelled as such.
**Served name:** `Ternary-Bonsai-2-27B-PQ2_0` (aliases `prism-bonsai-2-27b*`)
**Engine:** [Baekpica/ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs) `main`, native C/CUDA plus a Rust host, no Docker. Bonsai support arrived with "Add Prism Bonsai inference support (#69)".
**Default context / banks / port:** 262,144 / 1 / 8005
**Family:** `qwen35`, serial lane

> **Recipe contract:** [`recipes/prism-bonsai-2-27b-qwen35.yaml`](../recipes/prism-bonsai-2-27b-qwen35.yaml)
> **Launcher:** [`scripts/start-bonsai-spark.sh`](../scripts/start-bonsai-spark.sh)
> **Budget report:** [`scripts/bank-budget.sh`](../scripts/bank-budget.sh)
> **Artifact:** [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (public, no token needed)

---

## What this recipe buys you

The artifact is 6.71 GiB and the box it is meant for has 121.6 GiB of unified
memory, so the interesting question is not whether it fits - it is what that
memory actually buys. The answer here is depth, not concurrency: the whole
context ceiling of the artifact costs about a quarter of the Spark's memory and
leaves three quarters untouched, while the number of requests that can be in
flight at once is fixed at one by the engine, not by the memory.

The recipe pins the artifacts and checksums, the one environment variable that
decides whether a CUDA run lives or dies, the four context shapes, and the
engine's own memory arithmetic in a form you can re-run on any box.

## Artifacts

One file from `prism-ml/Ternary-Bonsai-2-27B-gguf`:

| File | Bytes | sha256 |
|---|---|---|
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7,206,168,928 | `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` |

`bash scripts/start-bonsai-spark.sh download` fetches it; `verify` checks the
size and the checksum. Both were checked against the copy on the reference box
on 2026-10-01 (the size matched to the byte and the SHA-256 matched the
published value), so the checksum in the recipe is the published one, not one
recorded from a download of unknown provenance.

Model metadata, from the engine's own tree rather than from prose:

| Item | Value |
|---|---|
| Architecture | `qwen35` |
| Trunk | dense 26.90 B, 64 blocks: 48 gated delta-net layers, 16 gated-attention layers, dense SwiGLU FFN, no MoE, no MTP block |
| Quant | PQ2_0 ternary, 2.125 bits per weight, stored Hadamard-folded (the engine rotates the activation, not the weight) |
| Tensors | 851 (402 pq2_0, 353 f32, 96 bf16) |
| Declared context | 262,144 (`crates/ds4-core/src/qwen35.rs`, `CTX_MAX`) |
| KV geometry | 16 attention layers x 4 kv heads x 256 head dim, fp16 k/v |

## Shapes

| Shape | Context | Max tokens | Note |
|---|---:|---:|---|
| `45k` | 45,056 | 8,192 | The shape the RTX 4070 SUPER runs; the only shape with measured serving numbers behind it. |
| `64k` | 65,536 | 16,384 | Two and a half times the 4070 shape. |
| `131k` | 131,072 | 32,768 | Half the declared ceiling. |
| `262k` | 262,144 | 32,768 | The declared ceiling; the Spark shape. Default. |

An explicit `CTX`, `MAXTOK` or `PREFILL_CHUNK` in the environment wins over the
shape, and the launcher says so when it happens.

## The serial lane, stated plainly

The Spark has room for a great many banks *of memory*. It cannot serve a great
many requests, and the two facts are unrelated:

- The family's serving caps declare `banks: BankLane::Serial` with
  `bank_support: Support::None` (`crates/ds4-core/src/serving.rs`,
  `ModelFamily::Qwen35`).
- Asking the engine for two banks at 32,768 context is refused, in the engine's
  own words: `error: qwen35 live serving is serial; --max-seqs 2 is not
  available`, error code `banks_unsupported`, exit 2. Reproduced on the
  reference box on 2026-10-01.
- `--kv-disk-dir` is refused the same way: `qwen35 session snapshots are
  unsupported`, code `disk_unsupported`.
- The engine also takes a single global lock, `/tmp/ds4.lock`: one ds4 process
  at a time on the box, whichever model it serves.

So the recipe does not pretend to bank: `BANKS` above 1 is refused by the
launcher before a start, with the engine's own reason quoted. What the memory
does buy is context, and the report below says exactly how much.

## The memory budget, from the engine's own quote

`bash scripts/start-bonsai-spark.sh budget` (which drives
`scripts/bank-budget.sh`) asks the engine for its plan with `--check-config`,
which prints the resolved plan as JSON **without opening the weights**, and
reads the numbers out of that quote:

    quote.shared_weights   the model mapping the plan charges
    quote.per_bank         what one bank costs at this context
    quote.floor            the memory floor held back
    quote.available        the free device memory the quote read, live
    quote.banks            how many banks the engine resolved
    quote.total            shared_weights + banks x per_bank + floor

Two quotes at different contexts give the per-bank line; a third checks it.
Measured on the reference box (RTX 4070 SUPER, 2026-10-01):

| ctx | per_bank (bytes) | per_bank | total |
|---:|---:|---:|---:|
| 32,768 | 2,570,354,688 | 2.394 GiB | 10.105 GiB |
| 40,960 | 3,107,356,672 | 2.894 GiB | 10.606 GiB |
| 49,152 | 3,644,358,656 | 3.394 GiB | 11.105 GiB |
| 65,536 | 4,718,362,624 | 4.394 GiB | 12.106 GiB |
| 131,072 | 9,014,378,496 | 8.395 GiB | 16.107 GiB |
| 262,144 | 17,606,410,240 | 16.397 GiB | 24.109 GiB |

The line is exact, not approximate:

    per_bank(ctx) = 422,566,912 + 65,552 x ctx   bytes
    total(ctx)    = 7,206,168,928 + per_bank(ctx) + 1,073,741,824

65,552 bytes per token is the family's own number, not a rounded 64 KiB: the 16
attention layers at 4 kv heads x 256 dimensions x 2 (k and v) x 2 bytes account
for 65,536 of it, and the engine charges 16 bytes per token on top. The
deviation between the line and a third quote was 0 bytes at 49,152 context.

The `floor` term is the memory reserve the engine keeps for everything that is
not the model, `MEM_FLOOR_GB`, defaulting to 1 GiB here because that is this
family's own default (`run-bonsai.sh` line 97, `DS4_BONSAI_MEM_FLOOR:-1`) - the
generic 4 GiB floor refused every usable context on the 12 GiB reference card
(`docs/BONSAI.md`). It is a reserve, not a cap: it is charged inside the plan
total, so a larger value makes the engine demand more free memory before it
agrees to listen, and it never makes the model smaller. Measured at ctx 131,072:
a 1 GiB floor gives a total of 17,294,289,248 bytes, an 8 GiB floor gives
24,810,482,016, the difference being exactly the 7 GiB of extra reserve.

Projected on the Spark (its 121.6 GiB against the same line; a projection from
the measured line, not a measurement, because no start has been run there):

| ctx | per_bank | total | headroom |
|---:|---:|---:|---:|
| 32,768 | 2.394 GiB | 10.105 GiB | about 110 GiB |
| 65,536 | 4.394 GiB | 12.106 GiB | about 108 GiB |
| 131,072 | 8.395 GiB | 16.107 GiB | about 104 GiB |
| 262,144 | 16.397 GiB | 24.109 GiB | about 96 GiB |

That last row is the whole story of this recipe: the artifact's deepest context
uses about a fifth of the Spark, and the remaining four fifths cannot be spent
on more requests, because the family serves one.

## Quick start

On the Spark, as the user that owns `~/models`:

```sh
bash scripts/start-bonsai-spark.sh download    # 6.71 GiB, resumable
bash scripts/start-bonsai-spark.sh verify      # size and published SHA-256
bash scripts/start-bonsai-spark.sh build       # make cuda-spark (sm_121a)
bash scripts/start-bonsai-spark.sh budget      # what fits on this box, from the engine
bash scripts/start-bonsai-spark.sh start 262k  # the declared ceiling
```

`stop`, `status`, `logs`, `plan`, `test` and `all` round it out, and every path,
port, context and memory knob is an environment override documented in the
script header and in `--help`.

A deliberate consequence worth knowing: the model is fine on a 12 GiB card at
the `45k` shape and below, so the same launcher describes both boxes. On the
reference RTX 4070 SUPER only 32,768 and 40,960 opened a plan; 45,056 opened in
one run and was refused in another, minutes apart, because the quote reads live
free device memory.

## The environment requirement that decides the run

Every CUDA run of this family needs `DS4_CUDA_COPY_MODEL=1`. The 6.71 GiB map
cannot be pinned on the reference box (`RLIMIT_MEMLOCK` is 8192 KiB and that is
also the hard limit), so without the variable the backend falls back to lazy
per-range materialisation and dies part-way through the trunk with
`Bonsai matmul failed for blk.<n>.<tensor>`. With it, the log reads
`CUDA copying 6.71 GiB model to device memory` and the copy costs about 0.6 s.
The launcher sets it for CUDA starts; `COPY_MODEL=0` turns it off deliberately.

Whether the Spark's memlock limit differs is unverified here; setting the
variable is harmless either way and costs the copy.

## Running, and what a start looks like

```sh
bash scripts/start-bonsai-spark.sh start 262k      # asks nothing when a shape is given
bash scripts/start-bonsai-spark.sh status
curl -s http://127.0.0.1:8005/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Ternary-Bonsai-2-27B-PQ2_0","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":64,"temperature":0}'
bash scripts/start-bonsai-spark.sh stop
```

The reference run (vizzio, ctx 32,768, one request at temperature 0):

```
finish: length
answer:
reasoning: The user is asking a simple factual question: "The capital of France is".
           I just need to complete the sentence. The capital of France is Paris.
usage:  {'prompt_tokens': 45, 'completion_tokens': 32, 'total_tokens': 77, ...}
wall:   1.075s
```

An empty `answer` with a full `reasoning` block is this family behaving as
documented: the reasoning block alone runs about 33 tokens, so a 32-token cap
stops before any content appears. Ask for 64 or more to see content.

## Limits

- **Serial.** Banks, MTP/DSpark drafting, SSD/disk KV, snapshots and media input
  are refused by name for this family. Do not read the caps as a qualified
  service: this family carries no qualified context, bank or prompt limits.
- **One ds4 model at a time.** The global `/tmp/ds4.lock` is held while a ds4
  process runs, including the Spark's other ds4 servers.
- **A fitting plan is an upper bound.** `--check-config` reported 45,056 opening
  and being refused on the same box within minutes; the real start is the only
  verdict that counts.
- **No long-context evaluation.** The declared ceiling is 262,144, which is what
  the artifact says it can address. Nothing has been measured here about quality
  at that depth.
- **The CPU reference is the slow path.** `TEST_PARITY=1 bash start-bonsai-spark.sh test`
  compares the CUDA graph against the CPU reference token for token; on the
  reference box the CPU side is about 3 s per forward, so that gate takes
  minutes.

## The model path trap on the Spark

The engine's own defaults point at the reference box's model directory:

    Makefile:1649      DS4_BONSAI_MODEL ?= /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf
    run-bonsai.sh:101  MODEL="${DS4_BONSAI_MODEL:-/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf}"

On the Spark that file does not exist - the model lives in `~/models` - so the
engine's bare commands fail there with:

    ds4: cannot open model '/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf': No such file or directory

which is also what `make bonsai-cuda-check`, `make bonsai-cuda-parity`, the
`test-qwen35-*` targets and `./run-bonsai.sh` produce on that box, since they all
inherit that default. The launcher here does not: its own default is
`$HOME/models` and it passes that path into every make invocation it runs
(`DS4_BONSAI_MODEL` exported, which wins over the Makefile's `?=`).

To use the engine's own commands on the Spark, name the path once:

```sh
DS4_BONSAI_MODEL=~/models/Ternary-Bonsai-2-27B-PQ2_0.gguf make bonsai-cuda-check
DS4_BONSAI_MODEL=~/models/Ternary-Bonsai-2-27B-PQ2_0.gguf ./run-bonsai.sh ids
```

or make it the default for every shell on that box, with no engine file edited:

```sh
echo 'export DS4_BONSAI_MODEL="$HOME/models/Ternary-Bonsai-2-27B-PQ2_0.gguf"' >> ~/.bashrc
```

## Picking the shape from the memory that is free (fits and auto)

A start should not have to guess which shape the box can hold, so the launcher
has a step that measures instead. `fits` reads the operating system's own view
(MemTotal and MemAvailable from /proc/meminfo), then asks the engine for its
verdict on each shape in turn, deepest first, with `--check-config` - which opens
no weights and starts nothing - and prints the plan, the headroom and the verdict
for each. On the 12 GiB reference box:

```
$ bash start-bonsai-spark.sh fits
memory on this box right now
  MemTotal              31.2 GiB   (the operating system)
  MemAvailable          19.9 GiB   (what the OS says is free; on a
                                  unified-memory box this is the same pool the model
                                  is served from, not a separate card's memory)

  shape   ctx       plan         headroom      verdict
  262k    262144    24.109 GiB   -             refused (quote_overflow)
  131k    131072    16.107 GiB   -             refused (quote_overflow)
  64k     65536     12.106 GiB   -             refused (quote_overflow)
  45k     45056     10.855 GiB   0.5 GiB       opens

  the engine reads 11.3 GiB free when it checks a plan, and the
  verdicts above are what it decided against that figure

the right step here is 45k: the deepest shape the engine accepts with the
memory free right now.
```

`start auto` uses the same check to choose for you: it takes the deepest shape
the engine accepts at that moment, says which it chose, and starts that. `plan
auto` and `budget auto` accept it too, so the same word works everywhere a shape
does.

Two things this makes visible, and they matter on a box that also serves other
models. The first is that the engine's own figure (`quote.available`, the free
memory its plan check reads) is the one the verdicts are computed against, and it
is printed next to the OS's; on a discrete-GPU box the two differ because they
are different pools (11.3 GiB of card memory against 19.9 GiB of system memory
above), while on the Spark's unified memory they should be close. The second is
that the answer moves with whatever else is running: a shape that opens now can
be refused a minute later, which is why the launcher runs the check again as part
of every start rather than trusting a shape recorded earlier.

## Sharing the box with another ds4 model (the Qwen3.8 case)

This Spark also serves Qwen3.8 on the same engine, so two independent gates
decide whether Bonsai can start. Only the first of them is about memory.

1. **Memory.** `fits` measures it, per shape, against the engine's own live
   reading of what is free.
2. **The engine's lock file.** `ds4_engine_open` takes a lock on `/tmp/ds4.lock`
   (`LOCK_EX | LOCK_NB`, ds4.c lines 44100 to 44126) and refuses a second
   process on that file:

       ds4: another ds4 process is already running (pid N); refusing to start

   The path is the engine's own environment variable, `DS4_LOCK_FILE`
   (ds4.c:44099), and the lock is exclusive per file rather than per machine. Two
   ds4 models therefore run side by side as soon as each holds its own lock file.
   Measured on the reference box with a ds4 server up as the holder:

       CONTROL, same lock file:  ds4: another ds4 process is already running (pid 117121); refusing to start
       TEST, own lock file:      CUDA backend initialized on NVIDIA GeForce RTX 4070 SUPER (sm_89)
                                 model catalog base: 851 tensors, ...   (started; own lock pid 117145)

The launcher passes its lock explicitly (`DS4_LOCK_FILE=$LOCK_FILE`), so sharing
the box is one variable:

```sh
LOCK_FILE=/tmp/ds4-bonsai.lock bash start-bonsai-spark.sh start auto
```

Only Bonsai needs it; the Qwen3.8 recipe can keep the engine's default lock,
because exclusivity is per file. With the default lock and the neighbour holding
it, `start` refuses and names this way out, and `fits` prints both gates: the
memory verdicts, computed against what is free with the neighbour already
running, and the lock state.

Two things to keep in mind when sharing:

- That lock is a guard against an accidental second run that would map tens of
  GiB, in the engine author's own comment. Overriding it deliberately puts the
  memory decision on the operator, and `fits` is the measurement that decision
  needs: run it with the neighbour up and it names the largest shape that fits.
- The Qwen3.8 uncensored recipe is a large tenant. Its own recorded figures are
  about 81 GiB resident for weights and repack, plus roughly 17.5 GiB per bank at
  196,608 context (scaled from the 23.35 GiB measured at 262,144), so at its
  default two-bank shape the box is effectively full and Bonsai has no room.
  Coexistence is realistic with that side on a smaller shape, or with Bonsai on
  one of the smaller profiles: 45k is a 10.9 GiB plan, 64k 12.1 GiB, 131k
  16.1 GiB, 262k 24.1 GiB. `fits`, with the neighbour running, is the answer that
  counts rather than this arithmetic.

## Using it from open-grok

## Using it from open-grok

Register the endpoint once in `~/.opengrok/config.toml` on the workstation:

```toml
[model.bonsai-spark]
model = "prism-bonsai-2-27b"
name = "Prism Bonsai 2 27B (Spark, 262K, one serial lane)"
base_url = "http://192.168.1.91:8005/v1"
api_backend = "chat_completions"
api_key = "dummy"
context_window = 262144
max_completion_tokens = 32768
supports_images = false
```

`context_window` has to match the shape you start: 262144 for the default `262k`,
45056 for `45k`, 65536 for `64k`, 131072 for `131k`. The id the server advertises
is `Ternary-Bonsai-2-27B-PQ2_0`, and `prism-bonsai-2-27b` is accepted as an alias
(the server does not validate the model field of a chat request).

Verify the registration before blaming the server:

```sh
open-grok models | grep bonsai
```

Then select it in a session with `/model bonsai-spark`. An entry whose server is
stopped is harmless and normal: it becomes usable the moment
`bash scripts/start-bonsai-spark.sh start 262k` runs on the Spark. Note that the
engine serves one ds4 model at a time, so starting Bonsai means stopping whichever
other ds4 server holds `/tmp/ds4.lock`.

## Checks

`bash scripts/selfcheck.sh` runs this unit's regression checks: syntax; that the
help prints nothing it executes (a backticked word in the help is a command
substitution here, which has bitten this file twice); the dispatch exit codes for
an unknown command and a bare shape; and, against the engine, that `plan` reaches
a verdict rather than reporting a failed run, that the quote satisfies
`total = weights + per_bank + floor`, that the slope the budget report prints
equals the slope two direct `--check-config` quotes imply, and that `fits` reaches
a recommendation. It takes the engine and model paths from the environment like
the launcher does, and skips that group with a stated reason when they are
missing, so the same command runs on the Spark, on a workstation, or in CI.

## What is verified, and where

Verified on the reference box (vizzio, RTX 4070 SUPER 12 GiB, engine checkout
`/data/ds4-dfm-rs`), 2026-10-01:

- the artifact's size and SHA-256 against the published values;
- the plan quote at six contexts, the exact linear per-bank line, and the
  `total = weights + banks x per_bank + floor` identity;
- the family's refusals: `--max-seqs 2` gives `banks_unsupported`,
  `--kv-disk-dir` gives `disk_unsupported`;
- a full lifecycle on the launcher: `start` at ctx 32,768 (plan accepted, pid
  recorded, listener up), a real chat request answered, `status`, `stop`, and no
  leftover process or listener.

Not verified, and named so nobody assumes it:

- any start on the Spark itself: the checkout there, the `make cuda-spark` build
  and the `262k` shape are all untested here, because this session cannot reach
  the Spark over SSH (public keys are not installed for it);
- the Spark's usably free memory, which the engine's own quote will report the
  first time `budget` runs there;
- `DS4_QWEN35_PREFILL_CHUNK=1024`, the family's documented maximum, which the
  launcher exposes but defaults below (512, the documented default).

## Troubleshooting

- `error: qwen35 live serving is serial; --max-seqs 2 is not available` - the
  launcher refused `BANKS>1` before starting. Run `budget` to see what the
  memory would allow and what the family will actually serve.
- `Bonsai matmul failed for blk.<n>.<tensor>` - `DS4_CUDA_COPY_MODEL` was not
  set for a CUDA run.
- `another ds4 process holds the single engine slot` - the global lock is held;
  stop the other ds4 server first (the launcher names it).
- `the engine rejected this plan before any weights were opened` - the shape is
  too deep for the free memory at that moment. Lower the shape or re-run; the
  quote moves with the box.

## Attribution

The model and its PQ2_0 packaging are by [Prism](https://huggingface.co/prism-ml);
the engine is a fork of [`antirez/ds4`](https://github.com/antirez/ds4). This
repo contributes the operational recipe: the artifact and checksum, the launch
shape that fits, the budget report, and the measurements behind both. Recipes
here are provided as-is, with no warranty.
