# Recipe: MiniMax-H3 video on the DS4 H3 host

MiniMax-H3 is MiniMax's diffusion transformer that generates video and its
soundtrack together, and `h3-server-rs` from `ds4-dfm-rs` serves it behind the
local v1 video contract: create, poll, download. This is the operator's account
of getting it onto the Spark, of what the host does and does not do at startup,
and of the numbers that belong to somebody else's box.

The contract is [`recipes/minimax-h3.yaml`](../recipes/minimax-h3.yaml) and the
launcher is
[`scripts/start-minimax-h3.sh`](../scripts/start-minimax-h3.sh). The client
side is the open-grok `h3-video` skill, which is plain shell plus curl.

## What this artifact is

Two partitions of one checkpoint, and the difference between them is what a
request may carry:

- **FL2VA** answers a prompt alone or a prompt with first/last-frame anchors.
- **Ref2VA** answers ordered image, video and audio references. At least one
  image or video is required, audio alone conditions nothing, a video
  reference needs at least two seconds of soundtrack, and all soundtracks
  share one fifteen-second budget. Keyframes and references never appear in
  the same request.

The engine opens the directory in metadata only: it requires
`FL2VA/transformer/config.json` and `FL2VA/tokenizer/tokenizer.json` as regular
files, then inventories `text_encoder`, `transformer`, `video_vae/source` and
`audio_vae`. Ref2VA is optional and its entire presence test is the regular
file `Ref2VA/transformer/model.safetensors.index.json`; when it is there, the
Ref2VA transformer is inventoried too, so a broken optional component fails at
open instead of at somebody's first reference request
(`crates/ds4-core/src/minimax_h3/partition.rs:41-46`, `:146-155`).

There is no autoregressive machinery anywhere in this path: no context, no KV
cache, no token stream, no MTP, no GGUF and no VQ loader. Nothing from the LLM
launchers in this repository applies, and this runbook shares none of their
numbers.

## Artifacts

| | |
|---|---|
| Checkpoint | `MiniMaxAI/MiniMax-H3` at `42ed227ee7df40d41602854ae760620d6eb651fe` |
| Visibility | public, not gated (Hub API, 2026-10-27) |
| On the box | `/mnt/h3-artifacts/MiniMax-H3` |
| Engine | `Baekpica/ds4-dfm-rs`, `main`; the reference box reads `be372776` |
| Build | `make CUDA_ARCH=sm_121 h3-server-rs` |
| Port | 8090 (the skill's default; 8000-8011 belong to the LLM recipes) |
| Licence | MiniMax-H3 Community License, not MIT |

The engine's required set is 102 unique files and 210,331,771,721 bytes: all
of FL2VA (81 files), the Ref2VA transformer and its own model index (16), and
five root metadata files. The five remaining Ref2VA components are
byte-identical to their FL2VA counterparts at this revision, which is why the
release's own instructions link them:

```sh
hf download MiniMaxAI/MiniMax-H3 --revision 42ed227e... --local-dir "$MODEL_DIR" \
  --include 'FL2VA/*' --include 'Ref2VA/transformer/*' \
  --include 'Ref2VA/model_index.json' \
  --include '.gitattributes' --include 'LICENSE' --include 'README.md' \
  --include 'model_index.json' --include 'modular_model_index.json'
for component in audio_vae processor text_encoder tokenizer video_vae; do
  ln -s "../FL2VA/$component" "$MODEL_DIR/Ref2VA/$component"
done
```

The checkpoint that is on the reference box was fetched on 2026-10-24 **without**
that last step: the five components are real copies, each with its own record
in the downloader's cache. That costs 134 GiB of duplicate payload and serves
exactly the same bytes, so nothing about it is broken, and the launcher reports
it rather than complaining:

```
h3: shared:     Ref2VA/audio_vae is a real copy (577.41 MiB); the release links it to
h3:             ../FL2VA/audio_vae instead, which saves that much disk and serves the same bytes
```

Measured on that box on 2026-10-27 with `find -printf`, so every figure is the
bytes on disk rather than a rounded `du`:

| Path | Bytes | Files |
|---|---:|---:|
| `FL2VA` | 144,051,182,625 | 81 |
| `FL2VA/transformer` | 66,280,524,863 | 15 |
| `FL2VA/text_encoder` | 66,726,510,529 | 23 |
| `FL2VA/video_vae` | 10,415,694,629 | 18 |
| `FL2VA/audio_vae` | 605,461,455 | 13 |
| `FL2VA/tokenizer` | 11,492,078 | 4 |
| `FL2VA/processor` | 11,498,352 | 7 |
| `Ref2VA` | 144,051,182,613 | 81 |
| `Ref2VA/transformer` | 66,280,524,863 | 15 |

Both partitions carry their own `model_index.json` (719 bytes in FL2VA, 707 in
Ref2VA), which is the twelve-byte difference between the two totals. A fetch
that follows the release's instructions produces 66,280,525,570 bytes of
Ref2VA instead, and the launcher accepts both layouts.

## Why there is no daemon flag, and what the launcher does instead

The question this unit started from was whether the H3 server has a daemon
option. It does not, and neither does the LLM engine's own binary: the whole
flag surface of `h3-server-rs` is `--bind`, `--out-dir`, `--model-dir`,
`--steps`, `--reuse`, `--renderer`, `--renderer-arg`, `--max-jobs`,
`--timeout-secs` and `--help`, and `grep -n daemon` over both server sources on
the box returns nothing.

Daemonising is therefore the launcher's job, exactly as it is for every other
artifact in this repository — `start --daemon` runs the server under
`setsid nohup`, writes a pid file, and waits until the port answers before it
returns. Nothing else is installed and nothing else is needed: the daemon is
the launcher, and the open-grok `h3-video` skill points at it.

## Beware: `make cuda-spark` does not build the H3 binaries

This is worth its own heading because it cost a question. The engine's
`cuda-spark` target builds

```
$(MAKE) -B ds4-eval ds4 ds4-server ds4-bench ds4-agent $(CUDA_EXTRA_BINS) CUDA_ARCH=sm_121
```

which is `ds4`, `ds4-agent`, `ds4-bench`, `ds4-eval`, `ds4-server` and
`ds4_weight_server` — not `h3-server-rs` and not the `h3-rs` CLI
(`Makefile:206-207`). The fingerprint of a `cuda-spark` build is exactly what
the box showed before this unit: six binaries stamped at 08:15 and no
`h3-server-rs` anywhere.

The H3 targets are named explicitly, as the engine's own documentation says:

```bash
cd ~/ds4-dfm-rs && make CUDA_ARCH=sm_121 h3-server-rs    # 5,084,200 bytes on this box
```

`make` also keeps the arch in `.ds4-cuda-config.mk`, which the Makefile includes
(`-include .ds4-cuda-config.mk`, line 50), so after a `cuda-spark` run a bare
`make h3-server-rs` inherits `sm_121` too — but the launcher passes the arch
explicitly, which also works on a fresh clone where that file does not exist
yet.

## Where the bytes go, and why the usual rule does not transfer

Every other recipe in this repository splits its payload by lifetime: the bytes
read once at load and then held resident go on the roomy volume, the bytes read
while the server runs go on the fastest device. **H3 has no such split.**
Nothing is copied to the device at open, and the components are loaded per
render, so all 288 GB of the directory is a read-while-serving payload and the
disk it sits on decides what a first render costs. On the box it lives on a 938
GB SATA SSD volume (`/mnt/h3-artifacts`, 269 GB used, 660 GB free) while the
home partition is the NVMe; that is the wrong way round for this artifact by the
letter of the rule, and it is what exists.

What the launcher does keep on the fast device is nothing at all: the host has
no cache directory, no offload tier and no scratch of its own. The only thing it
writes is the MP4, one file per job, never pruned, so the launcher puts that
under `$HOME/h3-videos` rather than next to the checkpoint.

## What a start actually looks like

The host opens metadata only. That means the port answers within seconds of a
start, and the tokenizer, the text encoder, the DiT and both VAEs load inside
the first render. From the acceptance run on the box on 2026-10-27:

```
h3: starting detached (setsid, nohup): 0.0.0.0:8090, 20 steps, out /home/leandro/h3-videos
h3: pid 150357, log /home/leandro/h3-server/h3-server.log
h3: ready in 2s: the host answers the contract

h3: ready: MiniMax-H3 video API on 0.0.0.0:8090
h3:   local   http://127.0.0.1:8090/v1/videos
h3:   lan     http://192.168.1.91:8090/v1/videos
h3:   steps   20 (a 20-step 768P render is hours; the API's smallest request is 1:1 at 768x768, 107 frames)
h3:   out     /home/leandro/h3-videos (never pruned: one MP4 per job until you remove it)
h3:   stop    bash start-minimax-h3.sh stop
```

and the host's own log is three lines:

```
h3-server-rs: local v1 video API on http://0.0.0.0:8090/v1/videos
  output /home/leandro/h3-videos
  engine /mnt/h3-artifacts/MiniMax-H3 (steps 20, reuse 1)
```

The resident set at idle was 213,072 KiB, and the box had 122,147 MB available
of 124,544 MB with the host running. That is the whole cost of "started".

Two consequences follow, and both are easy to get wrong:

1. A port that answers is a wire that is up, not a host that renders quickly.
   The readiness probe is deliberately a refusal — `GET /v1/videos/<unknown>`
   answers 404 `video_not_found` without touching the GPU — because the
   contract has no `/health` and no `/v1/models`.
2. The first render of a session pays the component loads. In the maintainer's
   API gate those were 37.774 s of text encoding and 48.538 s of DiT load
   inside the first request, on top of the denoise.

## Quick start

On the box that has the weights (run over its own ssh session; there is no ssh
route from this workstation to the Spark):

```bash
# the launcher, fetched from this repository (or copied by hand)
bash start-minimax-h3.sh preflight         # checkpoint, engine, box: no writes
bash start-minimax-h3.sh verify --quick    # presence and partition byte totals
bash start-minimax-h3.sh build             # make CUDA_ARCH=sm_121 h3-server-rs
bash start-minimax-h3.sh start --daemon    # 0.0.0.0:8090, log and pid under $HOME/h3-server
bash start-minimax-h3.sh test              # five refusals over the wire, no render
bash start-minimax-h3.sh status
bash start-minimax-h3.sh stop
```

`verify` (without `--quick`) is the expensive one: it reads every byte of both
partitions and compares each file against the digest the Hugging Face
downloader recorded for it when it fetched it. On this box that is about 288 GB
and it is the only check that catches a flipped bit — the engine does not hash
the shards at open, so a bad byte otherwise surfaces hours into a render. The
launcher's `start` reports whether a record matches the bytes on disk and does
not refuse without one, because a first start should not require a ten-minute
hash pass; run it once per fetch.

Then a clip, from anywhere on the network:

```bash
# the open-grok client, on the machine that runs open-grok
skills/h3-video/scripts/h3-video.sh --url http://192.168.1.91:8090 \
  --seconds 4 --ratio 1:1 "a red kite over a grey sea at dusk"
# or through the launcher, on the box
bash start-minimax-h3.sh render "a red kite over a grey sea at dusk" 4 1:1 1
```

## Watching a render: what a client sees, and what it cannot

A client that has the id can follow a job two ways, and the difference matters
when a clip takes hours.

**Over HTTP, by id.** `GET /v1/videos/{id}` answers the stored record, and the
only field the store derives at read time is `queue_position` — 1 for the job
the engine takes next, null while a job holds the single render slot and for a
job that has already run (`video_api.rs:223-238`). Two jobs posted back to back
on the stand-in host read like this, one second apart:

```json
{"id":"video_..._0_2","status":"in_progress","queue_position":null,
 "size":"768x768","seconds":15,"frames":362,"seed":1,"created_at":1793088177,
 "completed_at":null,"error":null}
{"id":"video_..._0_3","status":"queued","queue_position":1,
 "size":"768x768","seconds":15,"frames":362,"seed":2,"created_at":1793088177,
 "completed_at":null,"error":null}
```

and after the first finished, the second reported `in_progress` and then
`completed` with `completed_at` set. There is no percentage and no
steps-done counter in the record: it carries the state, the place in line, the
resolved target (`size`, `seconds`, `frames`), the seed, both timestamps and
any error. `created_at` and `completed_at` are epoch seconds, so a poller can
compute wall time itself.

**In the host's log, per denoise step.** The in-process engine writes one line
per evaluated denoise step to stderr (`crates/ds4-core/src/minimax_h3/engine.rs:311-322`):

```
STEP 3/20 seconds=42.180 total=118.442
```

`seconds` is the step just finished and `total` is the denoise phase so far, so
`bash start-minimax-h3.sh logs` (or
`tail -f $HOME/h3-server/h3-server.log`) is where real progress is visible,
step by step, while the HTTP record stays at `in_progress`. This is a property
of the in-process engine only; an external renderer writes whatever it writes
and the host does not relay it.

**When it is done.** `GET /v1/videos/{id}/content` returns the MP4 with a
`Content-Length`; `HEAD` returns the same headers with no body; before
completion it is 404 `video_not_ready`. The measured download of the second
queued job above was HTTP 200 and 629,407 bytes. The file is also on the box at
`<out-dir>/<id>.mp4`.

**What a client cannot do.** There is no list route, so an id that was never
seen cannot be discovered — the skill prints it as `id=...` when it creates a
job, and that string is the handle. And the store is in memory only: a host
restart forgets every record while preserving the MP4s. The practical
consequence is a good one — a render keeps running inside the host even if the
client that started it goes away, so a later session with the id can poll it
and download it — but only while that same host process is alive.

**Exercising all of this at no GPU cost.** The engine ships
`tools/minimax_h3/standin-renderer.sh`, a renderer that writes a real MP4 with
ffmpeg and never touches the model, and the launcher can drive it:

```bash
bash start-minimax-h3.sh --renderer $HOME/ds4-dfm-rs/tools/minimax_h3/standin-renderer.sh \
  --bind 0.0.0.0:8091 --state-dir $HOME/h3-server-standin start --daemon
```

`--renderer` replaces `--model-dir` exactly as it does in the binary, so no
checkpoint is opened and the memory floor does not apply. The open-grok client
then ran the full flow against it from another host:

```
$ skills/h3-video/scripts/h3-video.sh --url http://192.168.1.91:8091 \
    --ratio 1:1 --seconds 4 --seed 7 --poll 1 --out /tmp/h3-standin.mp4 "a red kite..."
id=video_18e25428a5472606_24cb5_0_1 task=t2va ratio=1:1 seconds=4 seed=7
status=in_progress size=768x768 frames=107 elapsed=0s
status=completed size=768x768 frames=107 elapsed=1s
wrote /tmp/h3-standin.mp4 (177349 bytes)
```

Two caveats when reading that transcript. The stand-in renderer's own frames
follow its own request, so an ffprobe of its file shows 96 frames for 4 seconds
where the record's `frames` says 107 — the record reports what the engine's grid
would plan, and a real render agrees with it. And none of this exercises the
model: it proves the wire, the queue, the state machine and the download, not
generation.

## The open-grok client entry

There is nothing to configure: the h3-video skill is plain shell and curl, so no
`config.toml` entry, no MCP server and no restart. The base URL is
`http://192.168.1.91:8090` and the only flag that matters is `--url`.

The client was exercised against this daemon from another host on 2026-10-27,
and what came back is the server's own refusal, which is the point:

```
$ skills/h3-video/scripts/h3-video.sh --dry-run --url http://192.168.1.91:8090 \
    --ratio 1:1 --seconds 4 "a red kite over a grey sea at dusk"
{"task":"t2va","prompt":"a red kite over a grey sea at dusk","conditions":[],
 "target":{"short_edge":768,"aspect_ratio":"1:1","duration_seconds":4},"seed":1}

$ skills/h3-video/scripts/h3-video.sh --url http://192.168.1.91:8090 --ratio 21:9 "a red kite"
{"error":{"message":"21:9 needs 1792x768 at this tier, which exceeds the 1344x768 released
 area; the widest canvas is 1344x768, a 1.333x stretch. Use 16:9",
 "type":"invalid_request_error","param":"target.aspect_ratio","code":"invalid_aspect_ratio"}}
h3-video: create refused (invalid_aspect_ratio)
```

No clip is queued by a refusal, so this is a safe way to prove the whole path
from the client to the GPU box without spending an hour of GPU.

## Numbers that belong to another box

Nothing has been rendered through this launcher. Every figure below was measured
by the engine's maintainer on a GB10 with the model at the pinned revision, and
recorded in `docs/minimax-h3/validation-2026-10-26.md`. They are what a first
render here should be compared against.

| Workload | Result |
|---|---|
| API gate: 768x768, 107 frames, 2 steps, seed 42 | 1,340.264 s client-observed, including 10 s polling |
| its phases | text 37.774 s, DiT load 48.538 s, denoise 975.987 s, audio decode 3.026 s, frame decode 263.217 s, mux 0.731 s |
| its memory | minimum MemAvailable 67.066 GiB, maximum PSI full avg10 6.29 percent, no guard trip |
| its output | 1,565,090 bytes, sha256 `252e88ceb94085af5cc6d8d072da6bacf43109157d6f7dc0d829dd31de427f56` |
| CLI: 512x512, 56 frames, 20 steps | 762.36 s in Rust against 764.62 s in the C reference, byte-identical MP4 |
| CLI: first+last frame, 2 steps | 113.97 s, byte-identical to the reference |
| CLI: image + video with soundtrack + audio, 2 steps | 184.92 s, byte-identical to the reference |
| per-step cost at small shapes | 40.2 s at 4,536 tokens, 112.9 s at 9,072 tokens |

The arithmetic that matters for planning: the API's smallest expressible request
is 1:1 at 768x768, 107 frames, which measured 22m20s at **two** steps. Steps are
fixed at startup, so a server started at the reference's own 20 will spend hours
on that same clip. Keep `--steps 2` on a shared server until the shape you
actually want has been timed, and remember that two-step output is an execution
check, not a quality setting.

A cheaper execution check exists below the wire: the CLI (`h3-rs`, a separate
make target) at 512x512 with a few frames and `--steps 2`. The wire refuses
every size but 768.

## What verify checks, and why it is offline

There is no `SHA256SUMS` in this release, so the launcher uses a stronger and
more local source: the Hugging Face downloader records one
`.cache/huggingface/download/<path>.metadata` per file it fetches, three lines
of commit, object id and fetch time. For an LFS payload the object id is the
sha256 of the bytes; for the small text files it is the git blob id, so the
launcher hashes those the way git does a blob. Both were written when the file
was fetched, which is what makes the check meaningful: nothing on the box can
make a corrupted byte agree with them.

Files with no fetch record — a component that was linked or copied instead of
downloaded — are compared byte-for-byte against their FL2VA twin, and a file
with neither a record nor a twin is reported as unchecked rather than counted
as verified. On this box all 162 partition files have records (81 per
partition), so the fallback is not exercised there.

`verify --quick` skips the hashes and compares presence and the partition byte
totals in the table above. That catches truncation, a half-finished fetch and a
different revision — not a flipped bit. The acceptance run recorded:

```
h3: quick verify: presence and byte totals
h3:   ok    FL2VA 134.16 GiB [144051182625]
h3:   ok    Ref2VA 134.16 GiB [144051182613]
```

## Limits

- **No authentication.** Any host that can reach the listener queues renders and
  spends the GPU for as long as it likes. `0.0.0.0:8090` is the default because
  a client on another machine is the point; `H3_BIND=127.0.0.1:8090` is the
  private option.
- **One render at a time**, in arrival order. A restart forgets every job record
  and leaves the MP4s exactly where they were; the queue is in memory only.
- **No small canvas, no short clip.** `short_edge` must be 768 and duration is
  4 through 15 seconds, so the cheapest request is 1:1 at 768x768 with 107
  frames — about 22 minutes at two steps and hours at twenty.
- **No `/health`, no `/v1/models`, no job list, no delete.** `--max-jobs`
  (default 8) caps retained jobs, and at capacity every create is 429
  `queue_full` until the host restarts.
- **Body caps**: 64 MiB body, 64 KiB headers, `Content-Length` required (chunked
  refused), at most 64 concurrent connections, one request per connection.
- **A condition URI must be a base64 data URI** (PNG/JPEG/WebP, at most 16 MiB
  decoded). Paths and http URLs are refused: the host never reads a caller-named
  file and never fetches a URL.
- **CUDA only, BF16 only.** There is no H3 GGUF/VQ/FP8 path and no Metal
  implementation; Darwin compilation is not qualified.
- **The licence is not MIT.** MiniMax-H3 Community License, carried in the
  release.
- **The duplicate Ref2VA components cost 134 GiB** on the reference box, and the
  launcher deliberately does not "fix" that by deleting or relinking anything:
  the layout is the operator's call, and both layouts are accepted.

## Checks

What was verified on 2026-10-27, in this unit, on the reference box
(`gx10-e1d2`, 192.168.1.91, GB10, driver 580.178.04, CUDA 13.0.88, ffmpeg
6.1.1, 121 GB unified memory):

- `preflight` passed: both required files present, all four FL2VA components
  inventoried, the Ref2VA index present, byte totals equal to the measured
  baselines, ffmpeg and ffprobe present, nvcc present, port 8090 free, 122,147 MB
  available.
- `build` produced `~/ds4-dfm-rs/h3-server-rs` (5,084,200 bytes) with
  `make CUDA_ARCH=sm_121 h3-server-rs`.
- `start --daemon` bound 0.0.0.0:8090, answered the contract in 2 s, and the
  listener was on 0.0.0.0:8090 with the process at 213,072 KiB resident.
- `test` asserted five refusals: 404 `video_not_found` for an unknown id, 400
  `invalid_task` for an empty body, 400 `unknown_field` for a stray field, 400
  `unsupported_short_edge` for a 512 tier, and 404 `unknown_url` for the LLM
  route `/v1/models`.
- `verify --quick` matched both partition byte totals; `verify` (the full digest
  pass) was NOT run.
- The open-grok client reached the daemon from this workstation and printed the
  server's `invalid_aspect_ratio` refusal; the launcher's `stop` then ended it.
- The whole flow was then exercised at no GPU cost against a stand-in renderer on
  0.0.0.0:8091: the client created a job, saw `in_progress` and then `completed`,
  downloaded 177,349 bytes and ffprobed it; two jobs posted back to back showed
  `queued` with `queue_position` 1 behind an `in_progress` job, and the second
  job's content downloaded as 629,407 bytes. The host and the stand-in were then
  stopped.
- `bash -n` and the launcher's `selftest` pass, and a sparse fixture exercises
  the three verify outcomes (digest match, digest mismatch, twin mismatch).

What was NOT done: no render of any kind, so no latency, memory or quality
figure from this box exists yet; and no full digest verification of the 288 GB.

## Troubleshooting

**`make cuda-spark` succeeded but there is no `h3-server-rs`.** Correct — that
target does not build the H3 binaries. Run
`make CUDA_ARCH=sm_121 h3-server-rs`.

**`start` complains the checkpoint is incomplete.** Run `preflight`: it names
which required file or component is missing and what each partition weighs. A
Ref2VA partition of 66,280,525,570 bytes is a linked layout and is fine; a
smaller one is a partial fetch or a wrong revision.

**The port answers but a create is slow to appear.** Expected: the first render
pays the component loads (about 38 s of text encoding and 49 s of DiT load in
the maintainer's gate) before denoise starts.

**Every create is 429 `queue_full`.** The host retains `--max-jobs` jobs and
there is no delete route. Restart it: records are in memory, the MP4s are not.

**A create is refused with `ordered references require the Ref2VA checkpoint`.**
`Ref2VA/transformer/model.safetensors.index.json` is missing, which is the
engine's whole presence test for the optional partition.

**503 `renderer_disabled`.** The host was started with neither `--model-dir` nor
`--renderer`, so it serves the wire and refuses every create. Start it through
the launcher, which always passes `--model-dir`.

**A render seems to hang.** It is probably just long. Poll the record: a job
behind another render reads `queued` with its place in line, and a running one
reads `in_progress`. There is no per-job timeout on the engine path; the CLI is
the cheap way to time a shape.

**`status` says the verification record is stale.** The checkpoint changed after
the last `verify`. Run `verify --quick` for the byte totals, or `verify` for the
digests.

## Attribution

The artifact is MiniMax's (`MiniMaxAI/MiniMax-H3`), the engine is
Baekpica's `ds4-dfm-rs` H3 port, which follows `h3.c` at `eae5d54` (MIT notice
preserved in the engine tree at `native/h3/LICENSE-h3.c`). The measured numbers
in the recipe and in this runbook are the maintainer's GB10 validation
(`docs/minimax-h3/validation-2026-10-26.md`), quoted with the workload they were
measured on. Nothing in this repository claims an H3 number of its own yet.
