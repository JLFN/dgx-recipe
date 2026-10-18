---
project: dgx-recipe
plan_start_commit: 1614223979e6d40cafa8101a0859388662cd1df9
last_updated_commit: e4f4b5413baa7fc9f827cbf8390684bb75268af8
branch: dev
remote: origin (https://github.com/JLFN/dgx-recipe.git)
handoff_written_at_context_usage: 23
handoff_written_at: 2026-10-18T19:35:00Z
session_id: 01a14fe2-e87b-7a21-8195-250d5b93fc40
---
--- handoff probe (auto) ---
=== handoff probe (dgx-recipe) ===
--- units completed since plan start (1614223979e6d40cafa8101a0859388662cd1df9):
  (none)
--- progress diff stat:
README.md                                    |  177 +++--
 dgx-recipe-handoff.md                        |  113 +++
 graphify-out/GRAPH_REPORT.md                 |   45 ++
 graphify-out/changeindex.json                |    2 +-
 qa-evidence/qa-waiver-2026-10-18.md          |   82 ++
 recipes/ds41-1017-uncensored-vq.yaml         |  216 ++++++
 recipes/glm-53-flash-uncensored-vq.yaml      |  252 ++++---
 runbooks/ds41-1017-uncensored-vq.md          |  296 ++++++++
 runbooks/glm-53-flash-uncensored-vq.md       |  533 ++++++-------
 scripts/install-ds41-1017-uncensored-vq.sh   | 1035 ++++++++++++++++++++++++++
 scripts/install-glm53-uncensored-vq-r13p8.sh |  817 ++++++++++++++++++++
 11 files changed, 3141 insertions(+), 427 deletions(-)
--- staleness (last_updated_commit 3edb380c8a4be348255f7fe78312383e2f0677a2 .. HEAD):
df4911d Merge branch 'fix/ds41-unc-drop-zchain-mention' into dev
3378364 docs(recipe): drop the --zchain mention from the DeepSeek installer
4609014 Merge branch 'fix/ds41-unc-norehash-vision' into dev
dbab6b2 fix(recipe): stop the DeepSeek installer hashing the vision sidecar on every run
6fe53a5 docs(handoff): record the DeepSeek verify-record fix and the Spark's proven digests
9280de4 Merge branch 'fix/ds41-unc-verify-record' into dev
4ba135b docs(recipe): record what the first fetch, assembly and verify printed on the Spark
e37c3a4 fix(recipe): let the DeepSeek installer's verify create its state directory
--- end handoff probe ---


Handoff + executable plan — dgx-recipe (2026-10-18)

0. Rules for writing THIS document (read before editing, not just before reading)

- Only write or replace this file at a clean, verified, committed boundary. Never checkpoint mid-work.
- Start drafting once context usage crosses ~25%; finalize and commit before the auto-compact threshold (this machine: [session] auto_compact_threshold_percent = 60). The writing is a summarization task and degrades like any other task under context pressure.
- Every claim in "Current state" must be re-verified by running the relevant command in this session, not copied or paraphrased from the previous handoff. If you cannot re-run it, mark it unverified.
- Hard caps: "Current state" max 15 lines. "Guardrails and open items" max 15 lines. "Next unit plan" has no cap — it is the actual work spec.
- Do NOT: replay prior units' logs (git log is the record), re-explain global rules, or record acceptance evidence you did not personally observe this session.
- This project had no handoff before 2026-10-18. This file is the first, written at a unit close, and it lives on dev per the rule 23 handoff exception.

1. Session protocol

This project is a sequence of units; each unit ends with a verified commit set and, at that boundary, the handoff and the memory entry are refreshed.

1. Read context usage from the session's signals.json (sessions/<percent-encoded-cwd>/<session-id>/signals.json, newest session dir). NOTE for this workspace: no session directory carries a signals.json (checked 2026-10-18, twice); fall back to chat_history.jsonl bytes / 4 against summary.json's context window, and say so when you report it. The 23 in this frontmatter comes from that fallback (956,649 bytes / 4 ≈ 239k of a 1,048,576 window).
2. If usage is below the threshold, continue immediately: write the next unit's plan into section 3, execute it, verify, commit, re-check, repeat. Do not stop at a unit boundary while under the threshold.
3. At the threshold the harness compacts this session in place; no new session is opened. This handoff must be written and committed BEFORE the threshold.
4. Preferred cadence: after every handoff, call compact_session as the last action of that turn.
5. Push is an OFFER, never an action (rule 23). The unit commits stay local until the operator answers, and the offer carries the branch, the range, the remote visibility just read, the gate results and the rollback.

2. Current state (2026-10-18, evening — ONE UNIT IN FLIGHT, verified this session)

- `dev` is local-only, 8 commits ahead of `origin/dev` (`3edb380`): the DeepSeek verify/fetch fixes (`e37c3a4`, `4ba135b`, `9280de4`, `dbab6b2`, `3378364`, their merges) plus the previous handoff. That push is still an OPEN offer (section 5). The DeepSeek unit's Spark state, the proven digests and the fixed installer's sha256 are in section 4 and `runbooks/ds41-1017-uncensored-vq.md`.
- The GLM-5.3 Uncensored VQ artifact was published a THIRD time, `743a912d` at 2026-10-18T16:21:21Z: R13+8 on 19 routed layers (3-16 plus 17, 22, 27, 41, 44), 94,491,712,768 bytes of payload. The installer pinned the 11:23 revision (`cc70aef5`, now the tag `r13p8-l3-16`), which is why the operator's `download` on the Spark named the wrong revision.
- IN FLIGHT, local only, not merged and not pushed: branch `fix/glm53-uncensored-vq-l3-19` at `e4f4b54`, three commits off `df4911d` — `08d15e9` (repin to `743a912d`; the fingerprint reads all three revisions in three groups and reports pinned / l3-16 / r13 / mixed / absent / unknown; download states why it fetches; run and kill floors become 2000 and 1500 MB), `5541168` (retire and delete the all-R13 installer, the operator's decision), `e4f4b54` (contract, runbook and README to the new revision).
- The installer file at `e4f4b54`: 54,455 bytes, mode 755, sha256 `6e0fb403893a67a6d6e7107b10b773d728283840a88487e178b41a7f425a96ca`.
- VERIFIED this session from the Hub's own API: each revision's listing and blob ids (37 of the 56 part digests identical across all three; the 19 that moved are exactly the upgraded layers), the refs (`r13p8-l3-16` → `f7919163`, `q4k-v3` → `5a24b762`), the pin's `recipe.json` layer list (`residual_stage.layers` 3-16, 17, 22, 27, 41, 44, container 2), and both marker constants at the pin (template 10,644 B sha256 `34d5ee66…`, MTP-02 sha256 `68a41631…`). VERIFIED locally: `bash -n` and 49 fixture assertions over seven sparse directories built from that listing, 0 failures.
- Rule 19's QA gate is NEITHER passed NOR waived on paper for `e4f4b54`: the operator asked to be handed the one file instead of a QA run, and the run was cancelled. The unit is therefore not push-ready, and the four pre-push checks have not been run against it.
- The Spark is untouched and unmeasured for all of this. The operator's 18:39 run was stopped (Ctrl-Z) right after it began a whole-release fetch, so what `/mnt/models/GLM-5.3-Flash-Uncensored-VQ-GGUF` holds now is UNVERIFIED — reading that directory with the installer's own diagnosis is the first step of the next unit.
- No `~/.opengrok` config file changed, so rule 7's backup was not triggered.

3. Next unit plan — first Spark deployment and first measurement of both artifacts (2026-10-18)

Status: not started. The installer for the new revision is committed on `fix/glm53-uncensored-vq-l3-19` and is NOT yet on the Spark; handing it over and reading the box come first.

Ground truth to verify first, in this order (do not skip to the run):
- Get `e4f4b54:scripts/install-glm53-uncensored-vq-r13p8.sh` onto the Spark (54,455 bytes, sha256 `6e0fb403893a67a6d6e7107b10b773d728283840a88487e178b41a7f425a96ca`) by the operator's own route, or push the branch and check it out there. There is NO ssh route from this workstation to the Spark and no mount is present — rule 24 applies: ask first, then use the operator's tmux pane or their ssh route.
- Read what the directory actually holds BEFORE any fetch: `preflight`, and if it cannot classify the upgraded parts, the diagnosis block it prints. Expect `r13` (17.2 GB still to fetch), `l3-16` (4.5 GB), `pinned` (nothing), `mixed`, `unknown`, or `absent` (the whole 94.5 GB).
- Confirm the engine on the Spark: `cd ~/ds4-dfm-rs && make cuda-spark` and that the checkout knows `34c46d1` (PR #96). No R13+8 revision can be read by an older build.

Phase A — GLM-5.3 Uncensored VQ at the 19-layer revision (the refetch, then a first quote).
- `preflight` must name the state it finds (`revision: q4k-v3 (all-R13) on disk`, `revision: r13p8-l3-16 on disk`, or `revision ok: the pin … is on disk`) and the disk need that follows from it, plus 8 GB of working room.
- `download` refetches only the parts that are not the pin's; expect `revision ok: the directory now reads 'pinned' - the pin, all 19 upgraded parts at R13+8 sizes`, 56 parts, 94,491,712,768 bytes of payload.
- `verify` re-runs the release's own SHA256SUMS over all 94 GB: it is the authority on content, since the fingerprint only compares sizes.
- `check` is the engine-floor test (container v2). Capture its whole output; the interesting lines are `weights`, `bank`, `scratch`, `MTP`, `floor`, `banks` and `total`. Acceptance: it must succeed with one bank; replace `budget.pinned_revision_quote` (card arithmetic, about 116.2 GiB) with the MEASURED quote and flip `budget.measured` to true.
- `start --daemon` then `test`: capture the start's own lines (the staged payload size, the graph plan, `width=1`, the listening line), the post-load `MemAvailable` against the 2000 MB run floor, and one timing line. Acceptance: served on 8009 as `glm-5.3-flash-uncensored-vq`; record `tok_per_step` and the decode rate.
- The all-R13 installer is retired and deleted, so the live trap-check that used to live here is gone.

Phase B — DeepSeek-V4.1-Flash-1017 UNCENSORED VQ (fetch, assemble, verify, serve).
- `preflight` on the Spark: expect the engine to advertise `--engram-dir`, `--vision-gguf`, `--mem-budget-mb`, `--mtp-mode`; check which Engram directory it selects for reuse and that its sizes match.
- `download` (114.5 GB with the tables reused, 317.6 GB without), then `assemble`, then `verify`. Acceptance: `assemble` reports `digest 2680747e…`; `verify` reports `parts ok: 40/40`, `vision identity ok`, `engram identity ok: both 1017 tables`, and writes `.verified`.
- `start`: expect about two minutes to map 113.6 GB, `/v1/models` answering on 8010, and the smoke request reading the model id back. Record the resident figure against the card's ~106 GiB.
- Decision to record either way: whether the reused Engram directory was the 1017 tables (expected) or needed a fresh fetch.

Phase C — fold the measurements back into the artifacts.
- Update `recipes/glm-53-flash-uncensored-vq.yaml`: `budget.measured: true` with the exact byte quote; `benchmark.measured_here: true` with the first decode rate and TTFT if taken; correct the `max_tokens` client value from the served `/v1/models` (it is still the R13 reading).
- Update `recipes/ds41-1017-uncensored-vq.yaml`: `budget` with the real quote, `benchmark` with the first numbers, and the loader line from the start log.
- Both runbooks get a "what the first start actually printed" section replacing the borrowed numbers where they now exist.
- Acceptance: every number in both recipes is either measured on the box or labelled with the artifact it came from.

Phase D — the follow-ups.
- Decide whether the Spark keeps a copy of either older revision in a directory of its own: the installer fetches `r13p8-l3-16` or `q4k-v3` by tag through `GLM53R13P8_REVISION` plus `GLM53R13P8_MODEL_DIR`.
- Rule 19: run the QA gate for `e4f4b54`, or record a waiver, before any push.

Verify, package, handoff: `bash -n` both scripts; run the fixtures again after any script change (`/tmp/glmfix` and `/tmp/dsfix` patterns are described in the QA record); commit with a conventional message whose body carries the measured numbers; refresh this handoff and the memory entry; offer the push (rule 23) with the four pre-push checks.

Notes/decisions: the Spark work cannot be done from this workstation without a route, so Phase A/B require the operator in the loop. The measurement unit is the point of the branch: both artifacts are currently operationally complete and evidentially empty.

4. Project facts a fresh session cannot re-derive

- What exists: `recipes/` one contract per artifact (7); `runbooks/` the operator's account per artifact (8); `scripts/` one launcher/installer per artifact plus `bank-budget.sh` and `selfcheck.sh`; `qa-evidence/` the unit's QA record; `docs/USAGE.md` the long-form guide for the Qwen recipe; `graphify-out/` the (empty by construction) AST graph, tracked.
- Ports: 8000 V4.1 port deployment, 8003 Qwen3.8, 8005 Bonsai, 8006 YoungAi V4.1, 8007 GLM-5.3 VQ, 8008 YoungAi DeepSeek 1017, 8009 GLM-5.3 Uncensored VQ (R13 and R13+8 share it), 8010 DeepSeek 1017 Uncensored VQ. One ds4 model at a time regardless: the engine takes a single global lock.
- Repositories and branches: this repo's integration branch is `dev`, production is `main`; the in-flight work branch is `feature/artifact-intake-1018`. `dev` carries this handoff commit; the branch carries the unit commits.
- Companion checkouts on this workstation: `/data/ds4-dfm-rs` is a read-only reference for the engine (its own session may own it — do not fetch or modify it from here); `/data/ds4` is the C engine. The Spark carries its own checkouts: `~/dgx-recipe` (this repository, the copy the operator ran the installer from) and `~/ds4-dfm-rs` (the engine it builds with `make cuda-spark`).
- Volumes on the Spark: `/mnt/models` is the roomy 953 GB volume for read-once payloads; `$HOME/models` and NVMe hold everything read while the server runs (Engram tables, PLE sidecars, vision sidecars).
- Environment facts: no ssh route from this workstation to the Spark (checked: no `~/.ssh/config` host entry, no `known_hosts` match, no sshfs/sftp mount). The Spark's artifacts are reached only through the operator's tmux/ssh session.
- Applicable skills: rust-build (the engine builds with `make cuda-spark`, not the canonical builder), osv-mcp is not applicable here (no lockfile in this repository — checked: no Cargo.lock, package-lock.json or requirements.txt).

5. Guardrails and open items

- Global guardrails in effect: the always-on core is ~/.opengrok/AGENTS.md (rules 0, 0.1, 1, 2, 4, 8, 15, 16, 17, 18, 20, 21, 22). The project-workflow rules 3, 5, 6, 7, 9, 10, 12, 13, 14, 19 and 23 live in ~/.opengrok/project-work-rules.md. The previous unit's rule 19 waiver (`qa-evidence/qa-waiver-2026-10-18.md`) does not carry forward.
- Project-specific guardrails: this repository is public (JLFN/dgx-recipe, checked 2026-10-18: visibility PUBLIC); rule 6's visibility check applies before any push. Nothing here streams from disk: the GLM VQ family cannot use `--ssd-streaming` at all.
- OPEN: hand the operator `e4f4b54:scripts/install-glm53-uncensored-vq-r13p8.sh` (sha256 `6e0fb403…`), and offer the push of `fix/glm53-uncensored-vq-l3-19` (3 commits, based on the local `dev` tip).
- OPEN: offer the push of `dev`'s 8 unpushed commits, which are the DeepSeek fixes (`e37c3a4`, `4ba135b`, `9280de4`, `dbab6b2`, `3378364` and their merges).
- OPEN: rule 19 for `e4f4b54` — run the QA gate or record a waiver; required before any push.
- OPEN: section 3's Phase A/B/C/D, starting with reading what the Spark's model directory actually holds.

6. Progress and staleness check (run this first, every session)

Progress meter:
    cd /data/dgx-recipe
    git log --oneline 1614223979e6d40cafa8101a0859388662cd1df9..origin/dev
    git log --oneline origin/dev..feature/artifact-intake-1018
    git diff --stat 1614223979e6d40cafa8101a0859388662cd1df9..feature/artifact-intake-1018
Trust git over this plan: continue with the first phase above whose work is NOT in the diff. There are no "Unit: N complete" trailers in this repository's history and this handoff does not start that convention.

Staleness check:
    cd /data/dgx-recipe && git log --oneline 0eb6b76cf8c89538d5341ae053a339bab98e2cb9..HEAD
One commit is expected and is this handoff's own docs commit; read anything beyond it before trusting sections 2-5, then update last_updated_commit.

This document is COMPLETE and self-contained: a fresh session can continue the project from this file alone. History is deliberately absent — git log and the project memory entry are the historical record.
