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


Handoff + executable plan — dgx-recipe (2026-10-27)

0. Rules for writing THIS document (read before editing, not just before reading)

- Only write or replace this file at a clean, verified, committed boundary. Never checkpoint mid-work.
- Start drafting once context usage crosses ~25%; finalize and commit before the auto-compact threshold (this machine: [session] auto_compact_threshold_percent = 60).
- Every claim in "Current state" must be re-verified by running the relevant command in this session, not copied or paraphrased from the previous handoff. If you cannot re-run it, mark it unverified.
- Hard caps: "Current state" max 15 lines. "Guardrails and open items" max 15 lines. "Next unit plan" has no cap — it is the actual work spec.
- Do NOT: replay prior units' logs (git log is the record), re-explain global rules, or record acceptance evidence you did not personally observe this session.

1. Session protocol

1. Context usage is read from this session's signals.json (sessions/<percent-encoded-cwd>/<session-id>/signals.json, newest session dir). This workspace has one now (verified 2026-10-27: contextTokensUsed 342511 of 1000000, contextWindowUsage 34, compactionCount 0); the older note that no session directory carries one is obsolete. If it is ever missing, fall back to chat_history.jsonl bytes / 4 and say so when you report it.
2. If usage is below the threshold, continue immediately: write the next unit's plan into section 3, execute it, verify, commit, re-check, repeat. Do not stop at a unit boundary while under the threshold.
3. At the threshold the harness compacts this session in place; no new session is opened. This handoff must be written and committed BEFORE the threshold.
4. Preferred cadence: after every handoff, call compact_session as the last action of that turn.
5. Push is an OFFER, never an action (rule 23) — except that the operator may answer "push". This session's H3 unit was pushed on that instruction, with the remote visibility read first (gh repo view JLFN/dgx-recipe: PUBLIC).

2. Current state (2026-10-27, verified this session)

- The MiniMax-H3 unit is PUSHED: feature/minimax-h3 = 2d74687 = origin/feature/minimax-h3, three commits — e32a405 (launcher, recipe, runbook, README), 40296aa (deliverables cleaned of the stale systemd reference), 2d74687 (the QA waiver). The branch also carries the MiniMax-M3.1-Flash VQ unit (c1f7ec0 and nine commits under it) because the operator chose that base; origin/dev is an ancestor of the tip, so rule 23's strict check passes.
- dev is local at d46fcce, 12 commits ahead of origin/dev (3edb380) and unpushed. Neither the MiniMax unit (83e76ce) nor the H3 unit is in dev: both feature branches are still unmerged.
- The launcher is scripts/start-minimax-h3.sh (renamed from install-*, at the operator's instruction) and it is the daemon: start --daemon is setsid+nohup plus a pid file. The H3 binary has no daemon flag (whole surface: --bind, --out-dir, --model-dir, --steps, --reuse, --renderer, --renderer-arg, --max-jobs, --timeout-secs).
- On the box (gx10-e1d2, 192.168.1.91): engine main at be372776, ~/ds4-dfm-rs/h3-server-rs built 2026-10-27 (5,084,200 bytes), nothing running now (ports 8090 and 8091 free, no h3 process).
- Checkpoint /mnt/h3-artifacts/MiniMax-H3: complete at 42ed227e, FL2VA 144,051,182,625 B + Ref2VA 144,051,182,613 B = 288,102,365,238 B; the five shared Ref2VA components are real copies, so 134 GiB of the payload is duplicated. No render has ever run through this launcher.
- Acceptance evidence this session: preflight against the real checkpoint, the build, start --daemon ready in 2 s with 213,072 KiB resident and the listener on 0.0.0.0:8090, the five-refusal wire test, verify --quick matching both totals, stop leaving the port free, the open-grok client reaching the daemon from this workstation, and a stand-in-renderer host on 8091 driven through create, in_progress, queued+queue_position, completed and two downloads (177,349 and 629,407 bytes).
- Rule 19 is waived in writing for this unit at the operator's instruction (qa-evidence/qa-waiver-2026-10-27.md); no independent reviewer looked at it, and the gate has not run.
- The skills are installed and pushed in the config repo (main a9c6a9a): h3-video and h3-prompt-writing in ~/.opengrok/skills, and the Claude-compat copy in ~/.claude/skills/h3-video synced to match. Rule 7's backup ran twice (restic exit 0).
- Nothing about the MiniMax-M3.1-Flash VQ artifact has been loaded on the Spark: its record still reads fetched, digests good, never loaded.

3. Next unit plan

Phase A — the first real H3 render, and the numbers this recipe still lacks. Start the host with H3_STEPS=2 on 8090, run one clip at 1:1 768x768 4 s through the open-grok client, and capture: the STEP lines and their seconds from $HOME/h3-server/h3-server.log, the record's size/seconds/frames, the MP4's ffprobe and its sha256, the minimum MemAvailable during the render, and wall time from create to download. Acceptance: recipes/minimax-h3.yaml moves to budget.measured: true and benchmark.measured_here: true with those readings, and runbooks/minimax-h3.md gains a "what the first render actually printed" section. Nothing may be relabelled without a reading behind it. The release setting is 20 steps and takes hours: take the two-step reading first, then decide whether a twenty-step reading is worth the box time.
Phase B — the 134 GiB decision. Either link the five Ref2VA components to their FL2VA twins (per component, reversible) or keep the copies. The launcher already accepts both layouts (Ref2VA 144,051,182,613 for copies, 66,280,525,570 for links), so this is a disk decision, not a code change. Record the choice in the runbook either way.
Phase C — the MiniMax-M3.1-Flash VQ artifact is still unmeasured although its engine support is on main (PR #110). The open plan is unchanged from the 2026-10-26 memory entry: preflight, download/verify on the box, check, start, test, and the first decode rate; it needs the box free, since one model is resident at a time.
Phase D — integration. The two feature branches are not in dev. Merging feature/minimax-m31-flash-vq and feature/minimax-h3 into dev (then a dev->main PR) is the operator's call; the conflict-prone spots are the README artifact table and the ports paragraph, which both units edit.
Phase E — housekeeping in the config repo: the working tree carries unrelated in-flight changes (63 deleted paths, 62 modified, 161 untracked) and ~/.opengrok/AGENTS.md still tells a reader that the old H3 systemd unit runs the host as a user service. Neither was touched here; both deserve a decision.
Verify, package, handoff: bash -n and the launcher's selftest after any script change, the sparse fixture for the checkpoint logic, then commit with the measured numbers, refresh this handoff and the memory entry, and offer the push.

4. Project facts a fresh session cannot re-derive

- What exists: recipes/ one contract per artifact (nine); runbooks/ the operator's account (nine); scripts/ one launcher per artifact plus bank-budget.sh and selfcheck.sh; qa-evidence/ the QA records and waivers; docs/USAGE.md; graphify-out/ tracked.
- Ports: 8000 V4.1, 8003 Qwen3.8, 8005 Bonsai, 8006 YoungAi V4.1, 8007 GLM-5.3 VQ, 8008 YoungAi 1017, 8009 GLM-5.3 Uncensored VQ, 8010 DeepSeek 1017 Uncensored VQ, 8011 MiniMax-M3.1-Flash VQ, 8090 MiniMax-H3 video. One ds4 model at a time (the engine's global lock); H3 is a separate process tree on the same GPU, one render at a time by its own store.
- The H3 host is NOT the ds4 engine: a different binary, no context, no KV, no MTP, and no daemon flag. Its contract is three routes (create, poll, download) with no /health and no /v1/models; the cheap liveness probe is a 404 video_not_found.
- The box: gx10-e1d2 at 192.168.1.91, GB10, 121 GB unified, driver 580.178.04, CUDA 13.0, ffmpeg 6.1.1; /mnt/h3-artifacts on /dev/sda (938 GB, 269 GB used) and home on NVMe (916 GB, 287 GB used).
- Reaching the box from this workstation: there is no ssh key here (~/.ssh holds only known_hosts and ssh-add reports no identities), so the route is the operator's tmux session 6, which holds an open ssh as leandro. Drive it with tmux send-keys -t 6 ... Enter and read it with tmux capture-pane -p -t 6 -S -N. Keep -S small: the scrollback is long and easy to drown in.
- Getting a file to the box: run python3 -m http.server 8899 in this repository and curl -o from the box against http://192.168.1.128:8899/... then verify both sides with sha256sum. Used this session and stopped afterwards. The gvfs sftp mount under /run/user/1000/gvfs hangs for shell commands and must not be relied on.
- The skills: ~/.opengrok/skills/h3-video (SKILL.md, AGENTS.md, scripts/h3-video.sh the client, scripts/h3-serve.sh the foreground starter) and ~/.opengrok/skills/h3-prompt-writing; the Claude-compat copy at ~/.claude/skills/h3-video is a real file, not a link, so both must be kept in sync. open-grok reloads skills when files change on disk.
- Rule 5 exception: the engine builds with make CUDA_ARCH=sm_121 h3-server-rs (or make cuda-spark for the ds4 binaries) on the box, never with /data/build. cuda-spark does not build the H3 binaries.
- Rule 19: this project has no recorded QA model; the 2026-10-01 Bonsai report recorded that the operator named the session model then. Ask, never assume.

5. Guardrails and open items

- Global core is ~/.opengrok/AGENTS.md; rules 3, 5, 6, 7, 9, 10, 12, 14, 19 and 23 are in ~/.opengrok/project-work-rules.md. This repository is PUBLIC (gh repo view JLFN/dgx-recipe, this session); the config repo JLFN/open-grok-backup is PRIVATE.
- OPEN: the MiniMax-M3.1-Flash VQ unit (feature/minimax-m31-flash-vq, pushed) is unmeasured and unmerged, and its rule 19 gate has never run.
- OPEN: the H3 unit is pushed under a written waiver; no QA reviewer saw it.
- OPEN: H3 has never rendered; recipes/minimax-h3.yaml says measured_here: false and every timing in it belongs to the engine maintainer's GB10 validation.
- OPEN: whether to reclaim the 134 GiB of duplicated Ref2VA components.
- OPEN: the two feature branches are not in dev; no PR exists for either.
- OPEN: the config repo's unrelated in-flight changes and the stale H3 systemd sentence in ~/.opengrok/AGENTS.md.
- Project-specific guardrail: the H3 family has no --ssd-streaming path and the checkpoint is read in full while serving.

6. Progress and staleness check (run this first, every session)

Progress meter:
    cd /data/dgx-recipe
    git fetch origin --quiet
    git log --oneline origin/dev..feature/minimax-h3
    git log --oneline origin/dev..feature/minimax-m31-flash-vq
    git log --oneline origin/dev..dev
Trust git over this plan: continue with the first phase above whose work is NOT in the diff. There are no "Unit: N complete" trailers in this repository's history and this handoff does not start that convention.

Staleness check:
    cd /data/dgx-recipe && git log --oneline 2d74687656b4ef02cd440704ebc57da3053aa521..HEAD
This handoff's own commit is the expected first entry; read anything beyond it before trusting sections 2-5, then update last_updated_commit.

This document is COMPLETE and self-contained: a fresh session can continue the project from this file alone. History is deliberately absent — git log and the project memory entry are the historical record.
