---
project: dgx-recipe
plan_start_commit: 1614223979e6d40cafa8101a0859388662cd1df9
last_updated_commit: 0eb6b76cf8c89538d5341ae053a339bab98e2cb9
branch: dev
remote: origin (https://github.com/JLFN/dgx-recipe.git)
handoff_written_at_context_usage: 26
handoff_written_at: 2026-10-18T15:30:00Z
session_id: 01a14f75-3c17-78f0-b52b-37334bfafe9f
---

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

1. Read context usage from the session's signals.json (sessions/<percent-encoded-cwd>/<session-id>/signals.json, newest session dir). NOTE for this workspace: the current session directory (01a14f75-…) carries no signals.json (checked 2026-10-18); fall back to chat_history.jsonl bytes / 4 against summary.json's context window, and say so when you report it. The 26 in this frontmatter comes from that fallback (1,092,143 bytes / 4 ≈ 273k of a 1,048,576 window).
2. If usage is below the threshold, continue immediately: write the next unit's plan into section 3, execute it, verify, commit, re-check, repeat. Do not stop at a unit boundary while under the threshold.
3. At the threshold the harness compacts this session in place; no new session is opened. This handoff must be written and committed BEFORE the threshold.
4. Preferred cadence: after every handoff, call compact_session as the last action of that turn.
5. Push is an OFFER, never an action (rule 23). The unit commits stay local until the operator answers, and the offer carries the branch, the range, the remote visibility just read, the gate results and the rollback.

2. Current state (2026-10-18 — COMPLETE, verified)

- Newest unit outcome: two artifacts published 2026-10-18 were taken in. Three commits on `feature/artifact-intake-1018`, none pushed: `a0c0d93` (GLM-5.3 Uncensored VQ brought to its R13+8 revision: new installer plus recipe and runbook rewritten), `02447c3` (new installer, recipe and runbook for DeepSeek-V4.1-Flash-1017 UNCENSORED VQ), `0eb6b76` (README table and sections, QA record, graph index).
- The rule 19 gate was waived by the operator for this unit and recorded in `qa-evidence/qa-waiver-2026-10-18.md`, together with the author's own checks and an explicit list of what is unverified. No independent QA reviewed this unit; an adversarial review subagent was launched before the waiver and was still running when the unit closed.
- Three defects in the DeepSeek installer were found and fixed during the unit: an option parser that swallowed the action word (this is what the operator hit on the Spark), `printf %02d` treating 08/09 as invalid octal (part count read 38/40), and `sha256sum -c --ignore-missing` passing vacuously on an empty directory. The last one changed the design: `verify` now hashes once and writes a record that `start` requires to match the bytes on disk, so `status` no longer reads 203 GB.
- The operator's Spark copy of the DeepSeek installer is the broken version until this branch reaches it; nothing was downloaded there, so there is no partial state to clean up.
- Branch/commit/installed state: see the frontmatter, plus section 6 for the progress meter. Nothing is installed or registered on the Spark by this unit.
- No `~/.opengrok` config file changed in this unit (only the project memory entry), so rule 7's backup was not triggered.

3. Next unit plan — first Spark deployment and first measurement of both artifacts (2026-10-18)

Status: not started. The operator asked for the branch to be finished and offered for push; the Spark work below is the natural next unit and is what makes the unmeasured claims measured.

Ground truth to verify first, in this order (do not skip to the run):
- `git fetch origin && git checkout feature/artifact-intake-1018` on the Spark's own checkout at `~/dgx-recipe` (it exists; the operator ran the broken installer from `~/dgx-recipe/scripts`), or copy the two scripts over the operator's own route. There is NO ssh route from the workstation to the Spark (`~/.ssh/config` and `known_hosts` have no gx10 entry, no sshfs/sftp mount is present) — rule 24 applies: use the operator's tmux pane or their ssh route, and ask first.
- Confirm the engine on the Spark: `cd ~/ds4-dfm-rs && make cuda-spark` and that the checkout knows `34c46d1` (PR #96). The GLM R13+8 revision cannot be read by an older build.

Phase A — GLM-5.3 Uncensored VQ R13+8 (the refetch, then a first quote).
- `bash scripts/install-glm53-uncensored-vq-r13p8.sh preflight` must report `revision: R13 on disk` and a disk need of about 12.7 GB plus 8 GB of room.
- `download` refetches the 14 rewritten parts in place; expect `revision ok: all 14 upgraded parts are now at R13+8 sizes`, 56 parts, 89,960,211,968 bytes of payload.
- `check` is the engine-floor test (container v2). Capture its whole output; the interesting lines are `weights`, `bank`, `scratch`, `MTP`, `floor`, `banks` and `total`. Acceptance: it must succeed with one bank; record the exact quote into the recipe's `budget.reference_quote` as a MEASURED quote and flip `budget.measured` to true.
- `start --daemon` then `test`: capture the start's own lines (the staged payload size, the graph plan, `width=1`, the listening line) and one timing line. Acceptance: served on 8009 as `glm-5.3-flash-uncensored-vq`; record `tok_per_step` and the decode rate, since the R13+8 predictor behaviour is unknown for these weights.
- Guard to verify live, not by reading the script: after the refetch, `bash scripts/install-glm53-uncensored-vq.sh status` (the OLD installer) would report this directory as its own release; run it once to confirm the trap is real on the box, then do not start from it.

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

Phase D — the two follow-ups this unit knowingly left open.
- Add a revision guard to the OLD GLM installer (`scripts/install-glm53-uncensored-vq.sh`) so it refuses a directory whose 14 parts are R13+8-sized. It currently cannot tell, and it will happily serve the newer revision. The new installer already carries the reverse guard.
- Decide whether the Spark keeps an R13 copy at the `q4k-v3` tag in a directory of its own, or stays R13+8-only.

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

- Global guardrails in effect: the always-on core is ~/.opengrok/AGENTS.md (rules 0, 0.1, 1, 2, 4, 8, 15, 16, 17, 18, 20, 21, 22). The project-workflow rules 3, 5, 6, 7, 9, 10, 12, 13, 14, 19 and 23 live in ~/.opengrok/project-work-rules.md. Rule 19's QA gate is waived for the 2026-10-18 unit only, recorded in `qa-evidence/qa-waiver-2026-10-18.md`; the waiver does not carry forward to the next unit.
- Project-specific guardrails: this repository is public (JLFN/dgx-recipe); rule 6's visibility check applies before any push. Nothing here streams from disk: the GLM VQ family cannot use `--ssd-streaming` at all.
- OPEN: offer the push of `feature/artifact-intake-1018` (three commits) and of `dev`'s handoff commit, then get them onto the Spark.
- OPEN: Phase A/B/C/D above, in that order.
- OPEN: the independent review subagent launched before the waiver was still running at unit close; if it returns findings, triage them as a follow-up commit rather than assuming the unit was clean.
- OPEN: `scripts/install-glm53-uncensored-vq.sh` (the R13 installer) has no revision guard; see Phase D.

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
