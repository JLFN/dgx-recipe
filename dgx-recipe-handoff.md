---
project: dgx-recipe
plan_start_commit: 1614223979e6d40cafa8101a0859388662cd1df9
last_updated_commit: 643cd8131a8ce332b0202a899a7a8c569211f977
branch: dev
in_flight_branch: feature/minimax-m31-flash-vq @ 3ebdf7d (this unit), fix/glm53-uncensored-vq-l3-19 @ e4f4b54 (previous unit)
remote: origin (https://github.com/JLFN/dgx-recipe.git)
handoff_written_at_context_usage: 24        # fallback read: chat_history.jsonl 1,006,575 B / 4 of a 1,048,576 window, 2026-10-26
handoff_written_at: 2026-10-26T19:30:00Z
session_id: 01a14fe2-e87b-7a21-8195-250d5b93fc40
---
--- handoff probe (auto) ---
=== handoff probe (dgx-recipe) ===
--- units completed since plan start (1614223979e6d40cafa8101a0859388662cd1df9):
  (none)
--- progress diff stat:
README.md                                    |  177 +++--
 dgx-recipe-handoff.md                        |  141 ++++
 graphify-out/GRAPH_REPORT.md                 |   45 ++
 graphify-out/changeindex.json                |    2 +-
 qa-evidence/qa-waiver-2026-10-18.md          |   82 ++
 recipes/ds41-1017-uncensored-vq.yaml         |  216 ++++++
 recipes/glm-53-flash-uncensored-vq.yaml      |  252 ++++---
 runbooks/ds41-1017-uncensored-vq.md          |  296 ++++++++
 runbooks/glm-53-flash-uncensored-vq.md       |  533 ++++++-------
 scripts/install-ds41-1017-uncensored-vq.sh   | 1035 ++++++++++++++++++++++++++
 scripts/install-glm53-uncensored-vq-r13p8.sh |  817 ++++++++++++++++++++
 11 files changed, 3169 insertions(+), 427 deletions(-)
--- staleness (last_updated_commit e4f4b5413baa7fc9f827cbf8390684bb75268af8 .. HEAD):
643cd81 docs(handoff): record the 19-layer revision and the in-flight installer branch
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

2. Current state (2026-10-26, late — the MiniMax payload is on the Spark; one unit pushed, two branches and dev awaiting the operator)

- COMPLETED this session, local only, not merged and not pushed: branch `feature/minimax-m31-flash-vq` at `e977474`, four commits off `dev`'s tip `643cd81` — `83e76ce` (the installer), `3ebdf7d` (contract, runbook, README), `f2ce777` (verify's two tiers against a stale manifest) and `e977474` (the family's serial lane, and the served context back to the native 1,048,576 by operator decision). It adds the eighth artifact: `Baekpica/MiniMax-M3.1-Flash-VQ-GGUF` at `2789cbaf`, 61 GGUF parts in one release directory, 115,526,325,376 bytes, port 8011, read by the engine's `minimax_m3_vq` family, which exists only as draft PR #110 (not merged).
- Rule 19 for that unit is WAIVED by the operator, recorded in `qa-evidence/qa-waiver-2026-10-26.md` with the author's own checks and an explicit list of what stays unverified. `qa-evidence/qa-report.md` still holds the 2026-10-01 Bonsai report and was deliberately not overwritten, because no independent tester reviewed the unit.
- `~/.opengrok/config.toml` gained `[model."minimax-m31-flash-vq"]` → `http://192.168.1.91:8011/v1` (api_key `dummy`, `context_window` 524288 to match the served context, images and video, `max_completion_tokens` deliberately unset because the host reports it per launch in `/v1/models`). Rule 7's restic backup ran at both commit times; config.toml is gitignored, so none of it is committed. The prior copies are `~/.local/state/grok-config-backups/config.toml.20261026-191710.bak` and the later one from this session.
- PUSHED this session, on the operator's instruction: `fix/ds41-unc-kv-disk-store` at `50ef4cc` (the uncommitted `--kv-disk-dir` edit of 2026-10-21, completed: the option help now states what the V4.1 route's disk store does and that its support is present but not qualified, preflight checks the directory and its room, status reports it). Its push carried `dev`'s ten unpushed commits onto the remote as ancestors, so the ref `origin/fix/ds41-unc-kv-disk-store` now exists while `origin/dev` is still at `3edb380`. No PR has been opened for it.
- STILL IN FLIGHT, untouched by this session: `fix/glm53-uncensored-vq-l3-19` at `e4f4b54` (three commits; rule 19 neither passed nor waived on paper).
- THE SPARK HAS THE PAYLOAD AND THE FIXED SCRIPT: `~/dgx-recipe/scripts/install-minimax-m31-flash-vq.sh` was replaced twice through the operator's gvfs mount, latest sha256 `d2723f21fd49c9ed...` (`e977474`'s version, verified equal on both sides). The first `start` was REJECTED before loading — `minimax_m3_vq MTP runs only on the serial lane (mtp_lane)`, then `serving plan rejected unsupported options` — because the launcher asked for a bank width and the drafter at once; that is fixed and the fix is on the box. Their first `verify` had failed on `README.md`, a stale entry in the release's own manifest rather than a bad fetch (section 4). Nothing is loaded yet.
- `dev` is local-only, 10 commits ahead of `origin/dev` (`40ef644`). Its push, the MiniMax branch's push and the GLM branch's push are three separate OPEN offers; the DeepSeek fix's PR is a fourth.
- Nothing has been LOADED on the Spark for any of the three units: the MiniMax payload is there and unverified by the engine, and what `/mnt/models/GLM-5.3-Flash-Uncensored-VQ-GGUF` holds is still unread.

3. Next unit plan — the Spark: read the box, then deploy and measure the three artifacts (2026-10-26)

Status: not started. Nothing of any of the three units is on the Spark. There is no ssh route from this workstation (section 4), so every step below runs through the operator's own route, and rule 24 says to ask before using it.

Ground truth to verify first, in this order. Do not skip to a run.
- Get the code onto the Spark by the operator's own route, or by pushing and checking out there. Three units: `feature/minimax-m31-flash-vq` (`3ebdf7d`), `fix/glm53-uncensored-vq-l3-19` (`e4f4b54`; the installer's file sha256 is `6e0fb403893a67a6d6e7107b10b773d728283840a88487e178b41a7f425a96ca`), and the DeepSeek installer at `dev`'s tip. The Spark's copy of this repository is `~/dgx-recipe`.
- Read what each model directory actually holds BEFORE any fetch, with each installer's own `preflight`. Every fetch is resumable, so re-running the same action is the repair; the handler for the GLM directory is `preflight` + `download`, for DeepSeek `preflight` + `download` + `assemble` + `verify`, and for MiniMax `preflight` + `download` + `verify`.
- Confirm the engine, and note that three floors are in play on three branches: the GLM VQ R13+8 19-layer revision needs PR #96 (`34c46d1`) or later; the DeepSeek V4.1 route needs PR #80 with vision from PR #83; MiniMax needs the `minimax_m3_vq` family of draft PR #110, which is only on `origin/feat/add-minimax-vq` (read here at `fd7192f4`). The checkout on the Spark therefore has to be chosen per run, and `make cuda-spark` run after each checkout. Nothing can be served from a single checkout for all three.
- Confirm free space: 115.53 GB plus 10 GB of working room for the MiniMax release on `/mnt/models`, 36.2 GB for its K/V tier on the NVMe, and whatever the GLM and DeepSeek directories still need. The MiniMax launcher checks both volumes in `preflight` and checks the tier again in `start`, because the tier file is fallocated in full before the port binds.

Phase A — MiniMax-M3.1-Flash VQ: the first deployment of this artifact anywhere outside its author's box.
- `bash scripts/install-minimax-m31-flash-vq.sh selftest` (no artifact, no GPU needed), then `preflight`. Expect the 61/61 part state or the count still to fetch, the two directories on the volumes the recipe documents, and the tier's room.
- `download` (115.53 GB, 81 of the revision's 227 files), then `verify`. Acceptance: 67 or more matching digest lines, no FAILED line, and a written record; `status` then reports the record as matching the bytes.
- `check`. This is the engine's own open, and it is not cheap: it hashes all 115.53 GB with O_DIRECT reads, checks the Q4_K and Q8_0 scales as they stream past and compares each digest against `release.json`, then prints its plan. Capture the whole output. The lines that matter are the family, `effective: ... weights=`, `ctx=`, any per-bank, scratch or floor figure, and every issue line. Acceptance: exit 0, and the plan's weight figure agrees with the 111,333,740,672 bytes of resident parts. Fold the quote into `recipes/minimax-m31-flash-vq.yaml`: `budget.measured: true` with the exact bytes, replacing the card's design computation.
- `start --daemon`, then `test`. Capture the staged payload, the fallocated tier size, the listening line, the post-load `MemAvailable` against the 4,000 MB run floor, and the smoke request's own timings. Acceptance: served on 8011 as `minimax-m31-flash-vq`, one bank at the native 1,048,576 context, and a recorded decode rate and time to first token.
- Read the completion cap from the running server and set it in both places: `curl -fsS http://127.0.0.1:8011/v1/models` (the host reports `top_provider.max_completion_tokens` as min(its server default, the context)), then `client.max_completion_tokens` in the recipe and the same field in `~/.opengrok/config.toml`, where it is deliberately unset today.
- Decision to record either way: whether the 1M shape with the drafter fits with room to spare, or the box needs `MINIMAX31_MTP_MODE=off` or `MINIMAX31_CTX=524288`. The card measured 114.75 GiB peak on the former and calls it unsafe on a 128 GB host, and 110.7 GiB without the drafter.

Phase B — GLM-5.3 Uncensored VQ at the 19-layer revision: preflight the directory, fetch only the parts that are not the pin's, verify against the release's SHA256SUMS, `check` for the engine floor and the memory quote, then `start --daemon` and `test` on 8009. Acceptance: the installer's fingerprint reports `pinned`, `check` succeeds with one bank, its quote replaces `budget.pinned_revision_quote` in the recipe with `budget.measured: true`, and the served rate is recorded.

Phase C — DeepSeek-V4.1-Flash-1017 UNCENSORED VQ: preflight, download (114.5 GB when the Engram tables are reused, 317.6 GB when not), assemble, verify (`parts ok: 40/40`, both identity proofs, `.verified` written), then start on 8010 and record the resident figure against the card's ~106 GiB. Record the decision either way: whether the reused Engram directory was the 1017 tables.

Phase D — fold the measurements back and close the units.
- Update the three contracts with the measured quotes and the first decode rates, each number labelled with the artifact it came from, and add a "what the first start actually printed" section to each runbook.
- Rule 19 for `e4f4b54`: run the gate or record a waiver; required before that push.
- Decide the fate of the uncommitted `--kv-disk-dir` edit in `scripts/install-ds41-1017-uncensored-vq.sh`: commit it as its own unit on its own branch once its behaviour is exercised against an engine that carries the flag, or revert it.
- Verify, package, handoff: `bash -n` every script touched; re-run the fixture checks in `qa-evidence/qa-waiver-2026-10-26.md` for the MiniMax installer after any change to it; commit with conventional messages whose bodies carry the measured numbers; refresh this handoff and the memory entry; offer each push with the four pre-push checks.

Notes and decisions: the Spark work cannot be done from this workstation without a route, so every phase needs the operator in the loop. The measurement is the point of all three units: each artifact is operationally complete and evidentially empty.

4. Project facts a fresh session cannot re-derive

- What exists: `recipes/` one contract per artifact (8); `runbooks/` the operator's account per artifact (9); `scripts/` one launcher/installer per artifact plus `bank-budget.sh` and `selfcheck.sh`; `qa-evidence/` the unit's QA record; `docs/USAGE.md` the long-form guide for the Qwen recipe; `graphify-out/` the (empty by construction) AST graph, tracked.
- Ports: 8000 V4.1 port deployment, 8003 Qwen3.8, 8005 Bonsai, 8006 YoungAi V4.1, 8007 GLM-5.3 VQ, 8008 YoungAi DeepSeek 1017, 8009 GLM-5.3 Uncensored VQ (R13 and R13+8 share it), 8010 DeepSeek 1017 Uncensored VQ, 8011 MiniMax-M3.1-Flash VQ. One ds4 model at a time regardless: the engine takes a single global lock.
- Repositories and branches: this repo's integration branch is `dev`, production is `main`. Local-only branches awaiting the operator: `feature/minimax-m31-flash-vq` (`3ebdf7d`, this unit) and `fix/glm53-uncensored-vq-l3-19` (`e4f4b54`, the GLM 19-layer revision). `dev` carries this handoff commit; the branches carry their unit commits.
- Companion checkouts on this workstation: `/data/ds4-dfm-rs` is a read-only reference for the engine (its own session may own it — do not fetch or modify it from here); `/data/ds4` is the C engine. The Spark carries its own checkouts: `~/dgx-recipe` (this repository, the copy the operator ran the installer from) and `~/ds4-dfm-rs` (the engine it builds with `make cuda-spark`).
- Volumes on the Spark: `/mnt/models` is the roomy 953 GB volume for read-once payloads; `$HOME/models` and NVMe hold everything read while the server runs (Engram tables, PLE sidecars, vision sidecars, and MiniMax's K/V tier through `DS4_KV_OFFLOAD_DIR`, which its launcher sets to `$HOME/models/minimax-m31-vq/kv-offload` rather than the engine's own `~/.cache/ds4/kv-offload` under $HOME).
- The MiniMax release cannot be split across volumes: the engine opens all 61 parts from one directory, so its 3.90 GiB of read-while-serving parts (the token table and the vision tower) sit beside the 103.69 GiB that is copied to the device once. That is the one place in this repository where the read-once / read-at-run-time rule cannot be followed completely.
- Release manifests are not self-consistent, and verify must survive that: the MiniMax release's own `SHA256SUMS` lists its model card as 21,014 bytes / `e2f68289` while the file at the pinned revision is 24,071 bytes / `2493b23c`, because the author edited the card twice on 2026-10-24 after writing the manifest (checked 2026-10-26 against the Hub). Its installer therefore hashes the payload strictly and the documentation advisably; a stale documentation entry warns with both digests and never fails the run. The same check on any future artifact here is cheap: hash the file the Hub serves at the pin and compare it with the manifest's line.
- The MiniMax launcher serves 524288 tokens of the model's native 1048576, by operator decision (512K covers this box's work). The SSD K/V tier is `(ctx / 128) * 60 * 73728` bytes, reserved in full before the port binds: 18.13 GiB at 512K, 33.75 GiB at 1M, none at or below 131072 tokens. The client entry's `context_window` must match whatever is served.
- The MiniMax family's drafter needs the SERIAL lane, and the pairing is a boot failure rather than a warning: `resolve_mtp` refuses `--mtp-mode on` for a serial-lane family whenever `bank_driver()` reports the bank driver present, which it does for any `--max-seqs` width on a persistent-bank family unless the serial route is selected. The engine's own switches are `--cont-width 0` (alias of `DS4_SERVER_COALESCE_MAX=0`, legacy meaning: no bank lane) and `DS4_SERVER_CONTINUOUS=0` (static/serial route); the resolved plan emits the latter for this case. `--max-seqs` accepts only `auto` or 1..64, never 0. The launcher passes both switches with the drafter armed and the width only without it.
- The MiniMax engine support is on `main` since PR #110 merged on 2026-10-25T02:33:10Z (head `4988ae7`); an earlier claim in this repository that it existed only on an unmerged branch was wrong and is corrected.
- Environment facts: no ssh route from this workstation to the Spark (checked: no `~/.ssh/config` host entry, no `known_hosts` match, no sshfs/sftp mount). The Spark's artifacts are reached only through the operator's tmux/ssh session.
- Engine branches on the Spark: `origin/feat/add-minimax-vq` carries the `minimax_m3_vq` family (draft PR #110); the GLM VQ and DeepSeek V4.1 families live on `main` and their own PR branches. The checkout must be chosen per artifact and rebuilt with `make cuda-spark` after each switch.
- Applicable skills: rust-build (the engine builds with `make cuda-spark`, not the canonical builder), osv-mcp is not applicable here (no lockfile in this repository — checked: no Cargo.lock, package-lock.json or requirements.txt).

5. Guardrails and open items

- Global guardrails in effect: the always-on core is ~/.opengrok/AGENTS.md (rules 0, 0.1, 1, 2, 4, 8, 15, 16, 17, 18, 20, 21, 22). The project-workflow rules 3, 5, 6, 7, 9, 10, 12, 13, 14, 19 and 23 live in ~/.opengrok/project-work-rules.md. Rule 19 waivers are per unit: the 2026-10-18 one does not carry forward, and the 2026-10-26 one covers only the MiniMax unit.
- Project-specific guardrails: this repository is public (JLFN/dgx-recipe; the rule 6 visibility check applies before any push). No lockfile exists here, so rule 10's OSV scan does not apply (checked 2026-10-26: no Cargo.lock, package-lock.json or requirements.txt). Nothing here streams from disk: the GLM VQ family cannot use `--ssd-streaming` at all, and the MiniMax family keeps its K/V in an unlinked file that the engine manages itself.
- DONE: `fix/ds41-unc-kv-disk-store` pushed to `origin` on the operator's instruction (`50ef4cc`). Its PR is not opened: offer it.
- OPEN: offer the push of `feature/minimax-m31-flash-vq` (4 commits off `dev`'s tip; the branch is the source of the copy now running on the Spark).
- OPEN: offer the push of `dev`'s 10 unpushed commits (the DeepSeek fixes, the merges, the previous handoff and this one).
- OPEN: offer the push of `fix/glm53-uncensored-vq-l3-19` (3 commits), after rule 19 for `e4f4b54`.
- OPEN: `~/.opengrok/config.toml` carries the MiniMax entry with `max_completion_tokens` unset on purpose; set it from the running server's `/v1/models` (Phase A). The entry appears in the model list only after an open-grok restart.
- OPEN: section 3's Phases A-D, starting with reading what the Spark's model directories actually hold.

6. Progress and staleness check (run this first, every session)

Progress meter:
    cd /data/dgx-recipe
    git log --oneline 1614223979e6d40cafa8101a0859388662cd1df9..origin/dev
    git log --oneline origin/dev..feature/artifact-intake-1018
    git diff --stat 1614223979e6d40cafa8101a0859388662cd1df9..feature/artifact-intake-1018
Trust git over this plan: continue with the first phase above whose work is NOT in the diff. There are no "Unit: N complete" trailers in this repository's history and this handoff does not start that convention.

Staleness check:
    cd /data/dgx-recipe && git log --oneline 643cd8131a8ce332b0202a899a7a8c569211f977..HEAD
One commit is expected and is this handoff's own docs commit; read anything beyond it before trusting sections 2-5, then update last_updated_commit.

This document is COMPLETE and self-contained: a fresh session can continue the project from this file alone. History is deliberately absent — git log and the project memory entry are the historical record.
