# Recipe: MiniMax-H3's Sol-Attn acceleration line on the DS4 H3 host

MiniMax-H3's serving path is dense: the DiT's attention is `h3d_sdpa`, and on
the campaign's yardstick job the video VAE decoder dominates the wall. This
runbook is the operator's account of a *second* line, the acceleration line,
which ports NVIDIA's Sol-H3-Spark sparse attention into the same engine and
measures it — and of the honest status of each piece of it. It is documentation
for work that is ported and priced but not wired: nothing here changes what
`h3-server-rs` serves, and no clip has been produced through it.

The contract is [`recipes/minimax-h3-sol.yaml`](../recipes/minimax-h3-sol.yaml).
The launcher is the shared
[`scripts/start-minimax-h3.sh`](../scripts/start-minimax-h3.sh), for a reason
this runbook gives its own heading. The serving recipe this line accelerates is
[`recipes/minimax-h3.yaml`](../recipes/minimax-h3.yaml) and its runbook is
[`runbooks/minimax-h3.md`](minimax-h3.md); where this file would repeat them it
points there instead.

## What this path is

The reference is NVIDIA's Sol-H3-Spark pipeline (`NVlabs/Sana`, branch
`sol-engine` @ `6e4a440`), a two-stage video-and-audio pipeline for one Spark:
an H3 draft, a latent transfer, an LTX-2.5 refiner, and a decode
(`docs/minimax-h3/sol-h3-spark-recon-2026-10-28.md:13-31`). Two things in it
are portable to this engine without new weights, and they are the ported pieces:

1. **Sol-Attn**, the refiner's sparse attention: a query-block threshold over
   light QK scores, exact blocks walked against real K and V, skipped blocks
   corrected from pooled K/V summaries in the *same* online softmax, one pass,
   training-free.
2. **VSA**, the reference H3 stage's own sparsity policy: video sparsity 0.9 in
   64-token tiles with the text/audio prefix exempt.

The port is in the engine's own CUDA. `h3d_sol_search` builds the routing mask
and `h3d_sol_forward` consumes it; both are declared in
`native/h3/h3_dit_gpu.h` (`:474-478` and `:507-511`), with the routing plan
struct at `:65-74` and the fixed shapes `BLOCK` 64, `HEAD_DIM` 128, `GROUP` 32
at `:45`. The engine work is split across three dated documents in the engine
tree, `docs/minimax-h3/sol-search-step-2026-10-28.md`,
`docs/minimax-h3/sol-forward-2026-10-28.md` and
`docs/minimax-h3/vsa-policy-2026-10-28.md`, cited throughout this runbook.

**This line does not serve anything.** At the branch tip the DiT still calls
`h3d_sdpa` (`crates/ds4-core/src/minimax_h3/dit.rs:2424`), and the only callers
of `h3d_sol_search` / `h3d_sol_forward` are the engine's own bench and test
sources (`tests/h3/sol_route_bm.cu`, `tests/h3/sol_forward_bm.cu`,
`tests/h3/test_native.cu`). There is no flag, no environment variable and no
API field that selects them, and this recipe invents none.

## Artifacts

| | |
|---|---|
| Checkpoint | `MiniMaxAI/MiniMax-H3` at `42ed227ee7df40d41602854ae760620d6eb651fe` (unchanged) |
| On the box | `/mnt/h3-artifacts/MiniMax-H3` |
| Engine | `JLFN/ds4-dfm-rs`, branch `feat/sol-h3` @ `72a47033` |
| Build | `make CUDA_ARCH=sm_121 h3-server-rs`; the sparse arms as `make tests/h3/sol_forward_bm`, `tests/h3/sol_route_bm`, `tests/h3/native_test` |
| Reference | `NVlabs/Sana` `sol-engine` @ `6e4a440`, read-only, never built |
| Licence | MiniMax-H3 Community License (checkpoint); the reference's own weights are gated or licence-unclear |

The engine's required set is unchanged from the serving recipe: 102 unique
files, 210,331,771,721 bytes
(`recipes/minimax-h3.yaml`, `crates/ds4-core/src/minimax_h3/partition.rs`). The
reference's own pipeline weights — a 42,018,190,584-byte LTX transformer and an
8,899,889,568-byte refiner LoRA (both gated), a 5,339,117,712-byte FastH3 VSA
LoRA, a 15,687,142,551-byte NVFP4-AWQ Qwen, and an LBH upscaler that declares
no explicit license at the pinned revision — are a different artifact set and
none of it is fetched.
(`docs/minimax-h3/sol-port-plan-2026-10-28.md:176-181`,
`docs/minimax-h3/sol-h3-spark-recon-2026-10-28.md:92-97`.)

## Why there is no switch, and what the shared launcher does instead

This is the question the line provokes, so it gets the plain answer: **there is
no switch to turn the sparse path on.** A search of the engine for one comes up
empty — no `H3_SOL`, no `H3_SPARSE`, no `SOL_ATTN`, no `--attention sparse`,
no top-k or prefix-exempt selector anywhere in `native/h3` or
`crates/ds4-core/src/minimax_h3`. The only attention control that exists is
`--attention tensor|scalar` (CLI) and the create body's optional `attention`
field, `tensor` or `scalar` (`crates/ds4-cli/src/bin/h3-rs.rs:76-78`,
`:836-841`; the wire field and its refusal are in
`docs/minimax-h3/local-v1-contract.md:73, :154`), and neither value reaches a
sparse kernel.

So `scripts/start-minimax-h3-sol.sh` does not exist, on purpose: with nothing
to change, a second launcher would be a rename. The recipe sets
`launcher: scripts/start-minimax-h3.sh`, and the shared launcher starts the same
host over the same checkpoint and the same three-route wire. The acceleration
arms are entered one layer down, at the engine's own test and bench targets:

```sh
cd $HOME/ds4-dfm-rs            # on feat/sol-h3 @ 72a47033
rm -f native/h3/*.o
make tests/h3/sol_forward_bm   # dense vs sparse forward, with control arms
make tests/h3/sol_route_bm     # the search step, priced per kernel
make tests/h3/native_test      # the correctness gate against the CPU oracle
```

Daemonising, the no-`--daemon` finding, and `setsid nohup` are all exactly as
`runbooks/minimax-h3.md` records them: none of that is touched by this line.

## Beware: `make cuda-spark` does not build the H3 binaries

Unchanged from the serving runbook, and it matters here too because the bench
targets are named the same way. `make cuda-spark` builds `ds4`, `ds4-agent`,
`ds4-bench`, `ds4-eval`, `ds4-server` and `ds4_weight_server`
(`Makefile:206-207`) — not `h3-server-rs`, and not the `tests/h3/sol_*_bm`
targets. Name the target:

```bash
cd ~/ds4-dfm-rs && make CUDA_ARCH=sm_121 h3-server-rs
make tests/h3/sol_forward_bm tests/h3/sol_route_bm tests/h3/native_test
```

## What the ported pieces are, and what each one gates

The reference rule, read first-hand and cited file by file
(`docs/minimax-h3/sol-forward-2026-10-28.md:12-30`):

- Blocks are 64 tokens, query and key alike; the forward walks key blocks in
  groups of 32 (`triton_ref/fwd.py:20-21`).
- `kc` is the block **mean** of K and `vc` the block **sum** of V, both stored
  BF16 (`preprocess.py:45`, `:82`, `:287-291`).
- Per group, the pooled scores `q @ kc^T * scale*log2(e)` decide the split
  (`fwd.py:88-108`).
- The approximated blocks are **folded into the same online softmax**, never
  skipped: `o += bf16(p) @ vc` and `row_sum += p * block_length` — which is
  why `vc` is a sum (`fwd.py:109-131`).
- The exact blocks are walked in increasing block order, one 64-key tile at a
  time, in the same running-max form (`fwd.py:132-170`).
- The output is `o / row_sum`, BF16 (`fwd.py:171-173`); a group with no
  approximated block leaves the running maximum untouched (`fwd.py:256-262`).

The port's two entry points and their gates:

| Piece | Entry point | Gate |
|---|---|---|
| Search (route) | `h3d_sol_search` | `tests/h3/native_test`, 19 new check names, mask against a transcription of the reference's own rule |
| Sparse forward | `h3d_sol_forward` | `tests/h3/native_test: 102 PASS / 0 FAIL` |
| Both, priced | `sol_route_bm`, `sol_forward_bm` | the tables below |

One deviation is named rather than hidden: the reference computes routing
*inside* the attention kernel and never materialises a mask; this port writes
one (`plan->mask`, `plan->count`). That costs the measured search step and buys
a consumer that does not recompute the predicate
(`docs/minimax-h3/sol-port-plan-2026-10-28.md:130-138`).

## Quick start

Nothing to start, because nothing is wired. What can be run today, on the box,
against the branch:

```bash
# 1. On the engine tree, at feat/sol-h3:
cd ~/ds4-dfm-rs
git rev-parse HEAD                      # expect 72a47033...
rm -f native/h3/*.o
make tests/h3/sol_forward_bm tests/h3/sol_route_bm tests/h3/native_test

# 2. The correctness gate (no checkpoint needed; the oracle is in the test file):
timeout 120s tests/h3/native_test        # expect 102 PASS / 0 FAIL

# 3. The pricing benches at the two denoise shapes:
tests/h3/sol_route_bm                     # the search step, per kernel
tests/h3/sol_forward_bm                   # dense vs sparse, with controls

# 4. The serving host is unchanged and needs no sparse flag:
bash scripts/start-minimax-h3.sh start --daemon
```

## The measurements, and exactly what they are

Everything here is **kernel-level**. Every figure is one attention call (or one
search step) timed back-to-back on the Spark, against synthetic fixtures, by
the engine's own bench binaries — not an end-to-end render, and not a denoise
pass. The two shapes are `18432 x 56 x 128` and `4352 x 56 x 128`. The first
is the one the campaign's yardstick job A (768x768, 107 frames) actually runs,
and the shape this engine's own bench calls "the denoise shape"; the second is
the 512x512 shape, which the wire refuses and only the CLI serves
(`docs/minimax-h3/speed-2026-10-27.md:44-49, :208-213`;
`docs/minimax-h3/sol-forward-2026-10-28.md:82-84`).

The search step, event-timed over back-to-back launches
(`docs/minimax-h3/sol-search-step-2026-10-28.md:136-149`):

| Kernel | 18,432 x 56 x 128 | 4,352 x 56 x 128 |
|---|---:|---:|
| pool (`kc`, `vc`) | 2.493 ms | 0.592 ms |
| `kc` statistics | 0.008 ms | 0.004 ms |
| threshold | 1.323 ms | 0.323 ms |
| route | 0.852 ms | 0.126 ms |
| search total | 4.924 ms | 1.158 ms |
| dense attention, same shape | 173.163 ms | 9.935 ms |
| search share | 2.8% | 11.7% |

The sparse forward, against the dense `h3d_sdpa` at the same shape
(`docs/minimax-h3/sol-forward-2026-10-28.md:84-100`):

| Arm | Exact | Forward | + search | vs dense |
|---|---:|---:|---:|---:|
| dense `h3d_sdpa` | — | 173.0-174.3 ms | — | 1.00x |
| sparse, tau 0.5 | 49.1% | 76.0 ms | 80.9 ms | 2.14x |
| sparse, tau 1.25 | 47.2% | 73.4 ms | 78.3 ms | 2.22x |
| sparse, tau 3.0 | 42.6% | 67.6 ms | 72.5 ms | 2.40x |
| control: all blocks exact | 100% | 150.3-152.1 ms | — | 0.87x |
| control: same count, contiguous window | 47.2% | 72.1 ms | — | — |

and at the 512 shape (`:96-100`):

| Arm | Exact | Forward | + search | vs dense |
|---|---:|---:|---:|---:|
| dense `h3d_sdpa` | — | 9.85-10.00 ms | — | 1.00x |
| sparse, tau 1.25 | 48.5% | 4.46 ms | 5.57 ms | 1.77x |
| control: all blocks exact | 100% | 8.82 ms | — | 0.88x |

Three things must be read into that table before any of it is called a win.

First, the **all-exact control is 0.87x dense**. That is the honest no-gain
lower bound of the same walk: with every block routed exact, the sparse forward
does the dense kernel's own work and comes out 13% *slower*. The whole gain
comes from the folded blocks, so a mask that routes almost everything exact
pays a penalty.

Second, the **realized exact share is a property of the fixture, not of the
model**. The bench fills K, V and Q with a per-block signal, and the tau
routing marks 42-49% of block pairs exact on that data. Only the model's own
logits on real latents decide the real share, and that has not been measured
(`docs/minimax-h3/sol-forward-2026-10-28.md:154-163`). The reference's own
headline sparsity (90%) is its VSA policy, which is a different rule.

Third, the timing sits at the defect the first measurement exposed: at 176
registers the kernel ran two CTAs per SM and measured 145.0 ms at 47.2% exact;
`__launch_bounds__(128, 3)` brought it to 165 registers with zero spills and
3 CTAs, same binary and same host, and the same arm measured 73.1-73.4 ms — a 2x
`docs/minimax-h3/sol-forward-2026-10-28.md:109-138`).

Correctness against the CPU oracle, which transcribes the reference's own rule
and consumes the mask as an input (`Kernel-level, over all live elements`):

| case | folded share | worst element | relative RMS |
|---|---:|---:|---:|
| router mask, tau 0.5 | 22.9% | 7.812e-03 | 5.3e-05 |
| router mask, tau 1.5 | 22.9% | 7.812e-03 | 5.3e-05 |
| all blocks exact | 0% | 7.812e-03 | 4.7e-05 |
| all blocks exact, against `h3d_sdpa` | 0% | 7.812e-03 | 5.5e-05 |
| 33 blocks over two words | 51.1% | 7.812e-03 | 1.6e-04 |
| hand-built, two far blocks, no band | 75.0% | 7.812e-03 | 1.2e-04 |
| hand-built, nothing exact | 100% | 3.906e-03 | 1.5e-05 |
| hand-built, one word exact, one folded | 3.0% | 7.812e-03 | 3.2e-04 |

The worst deltas are one BF16 ULP on values of order 1 (BF16's step near 1.0 is
7.8125e-03), and the gate is relative RMS below 1e-3
(`docs/minimax-h3/sol-forward-2026-10-28.md:49-67`).

The absolute figures carry a caveat of their own: the host has no
`nvidia-clocks.service`, so the campaign's 300-2200 MHz clock policy is not
applied and these runs sit above the campaign's 2184-2190 MHz reference range
(`docs/minimax-h3/sol-search-step-2026-10-28.md:256-280`). They are
internally consistent but not directly comparable to `speed-2026-10-27.md`'s.

## VSA: the reference's own sparsity rule, and the selector this port has not written

The sparse forward consumes a mask; it does not decide one
(`docs/minimax-h3/vsa-policy-2026-10-28.md:3-4`). For the H3 stage this engine
actually serves, the reference's decision rule is **VSA**, not Sol-Attn's tau
threshold: stage 1 uses VSA at 90% sparsity, and Sol-Attn is stage 2, on the
LTX refiner (`configs/default.json:12`; `sol-port-plan-2026-10-28.md:25-30`).

The VSA rule, read from the FastVideo module the reference hashes and refuses to
run without (`video_sparse_attn_h3.py`, sha256
`b7c6d4e4...dc0fab9`, matching the reference's pinned `VSA_SOURCE_SHA`):

- tile summary is the tile **mean** of Q and of K, in F32 (`:380-383`);
- scores are `(q_pooled @ k_pooled^T) / sqrt(head_dim)` (`:629`);
- keep count is `k_vid = max(1, min(ceil((1 - sparsity) * num_video_tiles),
  num_video_tiles))` — at 0.9 and 288 video tiles, 29 tiles per query row
  (`video_sparse_attn.py:161-163`);
- in the reference's `exempt` mode, every prefix column is selected and every
  prefix query row is dense (`:387-410`).

**The port's selector for this rule is not written.** `h3d_sol_search` routes
against the diag threshold only; no top-k, no prefix-exempt selector exists
anywhere in the tree. The design the VSA document proposes — a per-row top-k in
the same shared-staged form as the route kernel, ~10-20 us per call against the
measured 4.9 ms search — is an estimate to be measured, not a capability
(`docs/minimax-h3/vsa-policy-2026-10-28.md:62-67, :91-105`). Three deviations
would have to be named when it is written, and are named in advance: the
reference's tiles are 3D `(4,4,4)` tokens reordered into tile order, this port's
mask is over linear 64-token blocks of its own sequence, so whether the two
groupings coincide is a property of the engine's layout and must be checked
before the masks are called equivalent; the prefix is this engine's own text and
audio span; and the reference *drops* unselected tiles where this port's forward
*folds* them, which is a strictly better approximation of the dropped mass at
the cost of the nine folded passes.

## The video VAE decoder's lever, measured on another branch

The largest lever on the yardstick job is not attention at all: the F32 video
VAE decoder is 197.945 s of the 277.019 s job-A wall, 71.5%
(`docs/minimax-h3/vae-options-2026-10-28.md:1-9`). That work gave the decoder an
option surface and measured the arms, and the result is real: FRAMES goes from
**196.873 s to 34.494 s** with the tensor-core attention plus the reduced GEMM
mode, on two renders of one binary
(`docs/minimax-h3/vae-options-2026-10-28.md:175-181`).

It is **not part of this branch's artifact.** That work is on
`feat/h3-vae-options` (off `509ac111`), a different and currently paused branch,
and it is recorded here as *measured elsewhere and pending merge*. On this
branch there is no such option and no such reading. The option names
(`H3_VAE_ATTENTION`, `H3_VAE_SDPA_ORDER`, `H3_VAE_GEMM`), their defaults, and
the quality metrics behind them live in that document; nothing from it is wired
on `feat/sol-h3`.

## Numbers that belong to another box

Two end-to-end numbers are worth having in view, and neither may be turned into
a ratio against the other.

**The port's own yardstick, job A** (768x768, 107 frames, 2 evaluations, seed
42, tensor DiT attention — the engine's default), from its own paired run
(`docs/minimax-h3/speed-2026-10-27.md:520-537`):

| Phase | Seconds |
|---|---:|
| TEXT | 27.533 |
| LOAD | 7.160 |
| DENOISE (2 evals) | 40.294 |
| AUDIO | 2.577 |
| FRAMES (video VAE decode) | 197.945 |
| VIDEO (mux) | 0.623 |
| Total wall | 277.019 |

**The reference's own hot end to end**: 56.17 s for one Spark, at 1344x768, 121
frames, 5 s at 24 FPS (`docs/minimax-h3/sol-h3-spark-recon-2026-10-28.md:33-37`).
It is a *completely different job* — a different canvas, frame count, step
count, model set (an H3 draft plus an LTX refiner), decoder and weight formats —
and the campaign says plainly that no quotient of the two walls means anything
and computes none (`docs/minimax-h3/sol-port-plan-2026-10-28.md:42-48`,
`:50-55`). The reference's README attributes the figure to its project page and
adds "This is not a throughput or cold-start claim", and it cannot be
re-derived from the local tree. It is quoted here as a landmark, not a target.

What *is* comparable is each side's split of its own wall. On job A the term
that decides the wall is the port's own decoder (FRAMES, 71.5%), and neither
Sol-Attn nor a LoRA touches it. On a 20-step job the balance moves the other
way, which is why the denoise levers matter more there
(`docs/minimax-h3/sol-port-plan-2026-10-28.md:50-70`).

## Limits

- **Not wired.** No production caller, no flag, no environment variable, no API
  field. `dit.rs:2424` runs `h3d_sdpa`; the sparse pair is reached only from the
  engine's bench and test sources. Nothing here changes a served render.
- **Kernel-level numbers are not a render speedup.** The 2.14x-2.40x and 1.77x
  figures are one call against a dense call of the same shape; no denoise has
  been run through this path and the realized exact share is a fixture property.
- **The all-exact control is slower than dense (0.87x).** The gain is entirely
  the folded blocks; a near-dense mask loses.
- **The VSA selector does not exist yet.** Only the tau-threshold router is
  implemented; the reference's 90% top-k policy is documented but unported.
- **The VAE lever is measured elsewhere**, on the unmerged `feat/h3-vae-options`
  branch, and is pending merge.
- **No weight-level work.** No distilled LoRA, no FP8, no NVFP4 text encoder,
  and the reference pins a different H3 revision for those.
- **Clock caveat.** The host has no clock policy applied, so the absolute kernel
  figures are above the campaign's reference range.

## Checks

What was verified for this documentation unit, and what was not:

- The engine branch was checked rather than assumed. `git ls-remote fork
  refs/heads/feat/sol-h3` returns `72a470339e8d5622e2a520806d28553aa6f50c79`;
  `git ls-remote origin refs/heads/feat/sol-h3` returns nothing, and
  `git merge-base --is-ancestor feat/sol-h3 origin/main` fails. So the branch is
  published on the operator's fork `JLFN/ds4-dfm-rs`, **absent from upstream
  `Baekpica/ds4-dfm-rs`**, and **not an ancestor of upstream `main`**: it is not
  a release line, and no released H3 binary carries these entry points. It is
  not strictly local-only — it is on the fork.
- The no-switch claim was checked, not asserted: a grep for `H3_SOL`,
  `H3_SPARSE`, `SOL_ATTN`, and for `topk`/`VSA`/`prefix_mode` over `native/h3`
  and `crates/ds4-core/src/minimax_h3` returns nothing about a serving option,
  and the only callers of the two entry points are the three test/bench sources.
  The DiT's attention call is `crates/ds4-core/src/minimax_h3/dit.rs:2424`.
- The source hashes recorded in the recipe were read from the clean checkout
  at `72a47033` (`git status --porcelain` empty).
- The recipe YAML parses: `python3 -c 'import yaml,sys;
  yaml.safe_load(open("recipes/minimax-h3-sol.yaml"))'` exits 0, and its
  top-level keys are exactly `recipes/minimax-h3.yaml`'s.
- **The rule 19 QA gate has not run for this unit.** There is no independent
  reviewer for this change set and no QA report; this repository's rule 19 gate
  has not been exercised here.
- **Not done here:** no build, no bench run, no render, no GPU job, no Spark
  session. Every measured figure in this runbook and in the recipe is quoted
  from the engine's dated documents, not taken here.

## Troubleshooting

**There is no `--sol` / `--sparse` option and no `start-minimax-h3-sol.sh`.**
Correct, and the point of the "Why there is no switch" section: the sparse path
has no serving surface. Use the shared launcher for serving and the bench
targets for the sparse arms.

**`make cuda-spark` finished but there is no `sol_forward_bm`.** Correct: that
target builds the `ds4` binaries, not the H3 benches. Name
`make tests/h3/sol_forward_bm` (and `tests/h3/native_test`) explicitly.

**A sparse arm measures slower than dense.** Check the exact share first: the
all-exact arm is 0.87x dense by design, so a mask that routes almost everything
exact is expected to lose. The gain is the folded blocks.

**The sparse bench and gate sources are not on `main`.** `tests/h3/native_test`
carries the correctness gate; `sol_route_bm` and `sol_forward_bm` carry the
pricing. All three live on `feat/sol-h3`, not on `main`.

## Attribution

The algorithm is NVIDIA's (`NVlabs/Sana`, `sol-engine` @ `6e4a440`); the port
and its measurements are the `ds4-dfm-rs` engine's, quoted from
`docs/minimax-h3/sol-search-step-2026-10-28.md`,
`sol-forward-2026-10-28.md`, `vsa-policy-2026-10-28.md` and
`vae-options-2026-10-28.md`. The checkpoint, the engine host and the wire are
Baekpica's `ds4-dfm-rs` H3 port, served exactly as
`runbooks/minimax-h3.md` records. This repository claims no Sol-Attn number of
its own: every figure in
[`recipes/minimax-h3-sol.yaml`](../recipes/minimax-h3-sol.yaml) is a citation,
and the rule 19 gate has not run for it.
