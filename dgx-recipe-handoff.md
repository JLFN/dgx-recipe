---
project: dgx-recipe
plan_start_commit: 1614223979e6d40cafa8101a0859388662cd1df9
last_updated_commit: 09a9e46
branch: dev
remote: origin (https://github.com/JLFN/dgx-recipe.git)
handoff_written_at_context_usage: 24
handoff_written_at: 2026-10-31T09:34:19Z
session_id: 01a19126-3d8c-78a3-a6da-d6cdeea7d45c
---

Handoff + executable plan — dgx-recipe (2026-10-31)

0. Rules for writing THIS document (read before editing, not just before reading)

- Only write or replace this file at a clean, verified, committed boundary. Never checkpoint mid-work.
- Every claim in "Current state" must be re-verified by running the relevant command in this session, not copied or paraphrased from the previous handoff. If you cannot re-run it, mark it unverified.
- Hard caps: "Current state" max 15 lines. "Guardrails and open items" max 15 lines. "Next unit plan" has no cap — it is the actual work spec.
- Do NOT: replay prior units' logs (git log is the record), re-explain global rules, or record acceptance evidence you did not personally observe this session.
- Push is an OFFER, never an action (rule 23). This session pushed nothing.

1. Session protocol

1. Context usage is read from this session's signals.json (sessions/<percent-encoded-cwd>/<session-id>/signals.json, newest session dir). Verified this session: contextTokensUsed 242,662 of 1,000,000, contextWindowUsage 24, compactionCount 0.
2. If usage is below the threshold, continue immediately: write the next unit's plan into section 3, execute it, verify, commit, re-check, repeat. Do not stop at a unit boundary while under the threshold.
3. At the threshold the harness compacts this session in place; no new session is opened. This handoff must be written and committed BEFORE the threshold.
4. Preferred cadence: after every handoff, call compact_session as the last action of that turn.
5. Push is an OFFER, never an action (rule 23) — except that the operator may answer "push".

2. Current state (2026-10-31, verified this session)

- dev is at 09a9e46 on top of the handoff commit, 43 commits ahead of origin/dev (3edb380), unpushed; this repository is PUBLIC (checked in an earlier session, not re-read here).
- EVERY local branch is now merged into dev: feature/minimax-m31-flash-vq, feature/minimax-h3, feature/minimax-h3-sol, fix/glm53-uncensored-vq-l3-19, fix/ds41-unc-kv-disk-store, feature/ds41flash-1017-options (e95abc2, today's V4.1 1017 work) and feature/mimo26-uncensored-vq (9836c93). `git branch --no-merged dev` prints nothing (verified).
- Three integration checks pass on the merged tree: `bash -n` over every scripts/*.sh, `scripts/selfcheck.sh` (engine-dependent checks skipped, no engine here), and `scripts/selfcheck-ds41flash-1017.sh` (PASS). Every recipes/*.yaml parses and every metadata.runbook / metadata.launcher path it names exists.
- Two defects were found and fixed at integration rather than inherited: recipes/minimax-h3.yaml did not parse at all (an unquoted `: ` in metadata.upstream_model, fixed at e15167f), and the README still said nine artifacts (MiMo row, section, port 8012 and the count landed at 09a9e46).
- The operator's main checkout /data/dgx-recipe is STILL on feature/minimax-h3-sol with today's three DS4 files modified in the working tree. Their content is byte-identical to what is now in dev (sha256 compared this session), so they can be dropped with `git restore` once the operator is on dev; the tree also carries a modified dgx-recipe-handoff.md (a stale probe refresh).
- Two worktrees exist beside it: /data/dgx-recipe-int on dev (where this integration was done) and /data/dgx-recipe-mimo on feature/mimo26-uncensored-vq.
- On the box (gx10-e1d2, 192.168.1.91): the MiMo fetch is COMPLETE and loose in /mnt/models — 53 parts at the published sizes, 0 incomplete files, 98,520,657,024 bytes of parts plus the Bonsai file (verified through the sftp mount). It has not been laid out, verified, built, loaded or timed.
- The box's tmux pane (session 0) is the only shell route from this workstation (no ssh key, no ~/.ssh/config entry). During this session it was occupied by the operator's `make cuda-spark` in ~/ds4-dfm-rs and by htop, so the box-side MiMo steps did not run.
- Nothing in this repository has been loaded or timed on the box for any unit except the Qwen3.8, Bonsai, GLM R13 and H3-host cases recorded earlier.

3. Next unit plan

Phase A — MiMo-V2.6-Flash UNCENSORED VQ on the box, the unit this session prepared. With the pane free, copy scripts/install-mimo26-uncensored-vq.sh to ~/dgx-recipe/scripts/ and run, in order: `layout` (adopt the 53 loose parts into /mnt/models/mimo26-unc, move embed-tokens-bf16.gguf to $HOME/models/mimo26-unc/embed and symlink it back), `verify` (53 sizes, 53 digests, the tables, the split, the record), `build` (detached worktree at 2fa8c08e under ~/ds4-dfm-rs-mimo; the operator's checkout stays on pr-122), `check` (the engine's --check-config; expect family MiMo-V2.6-Flash-VQ and about 109.79 GiB of the box's quote), then `start` and `test`. Capture what start actually prints: the O_DIRECT hash pass and the device copy are 2 x 93 GB, so time the load and report MemAvailable after it. Acceptance for this phase: the record matches at start, /v1/models answers on 8012, one chat completion returns, and the runbook gains a "what the first start printed" section. Nothing is relabelled until a reading exists.
Phase B — the first decode rate and prefill rate on this box, with `bench` printing the reproduction of the author's 2K-64K sweep (32 frontiers, 2,048 incremental prompt tokens, 128 forced greedy tokens, three fresh processes). Record the command, the SM clock range and the date beside every number, then move benchmark.measured_here to true only for what was measured.
Phase C — the H3 first render, unchanged from the previous plan: H3_STEPS=2 on 8090, one clip at 1:1 768x768 4 s through the open-grok client, and capture the STEP lines, the record, the MP4's ffprobe and sha256, the minimum MemAvailable and the wall time. Then the 134 GiB duplicated-Ref2VA decision (link or keep), recorded in the runbook either way.
Phase D — the MiniMax-M3.1-Flash VQ artifact has never been loaded although its engine support is on main (PR #110); preflight, load, check, start, test, first decode rate. It needs the box free.
Phase E — the dev->main PR is the operator's call. All units are in dev now; the README table and the ports paragraph are the only multi-unit files, and they were resolved at integration.
Phase F — housekeeping in the config repo: its working tree carries unrelated in-flight changes, and ~/.opengrok/AGENTS.md still describes the retired H3 systemd unit. Neither was touched here.
Verify, package, handoff: `bash -n` plus both selfchecks after any script or recipe change, then commit with the evidence, refresh this handoff and offer the push.

4. Project facts a fresh session cannot re-derive

- What exists: recipes/ one contract per artifact (ten), runbooks/ the operator's account (ten), scripts/ one launcher per artifact plus bank-budget.sh, selfcheck.sh and selfcheck-ds41flash-1017.sh; qa-evidence/ the QA records and waivers; docs/USAGE.md; graphify-out/ tracked.
- Ports: 8000 V4.1, 8003 Qwen3.8, 8005 Bonsai, 8006 YoungAi V4.1, 8007 GLM-5.3 VQ, 8008 YoungAi 1017, 8009 GLM-5.3 Uncensored VQ, 8010 DeepSeek 1017 Uncensored VQ, 8011 MiniMax-M3.1-Flash VQ, 8012 MiMo-V2.6-Flash UNCENSORED VQ, 8090 MiniMax-H3 video. One ds4 model at a time (the engine's global lock); H3 is a separate process tree on the same GPU.
- The box: gx10-e1d2 at 192.168.1.91, GB10, 121 GB unified, 15 GB swap. Handoff-recorded driver 580.178.04 / CUDA 13.0; the MiMo card's numbers were taken on 615.71.10 / CUDA 13.4.92, so they do not transfer unmeasured.
- Disks measured this session with the fetch writing to /mnt/models: NVMe root 2.8 GB/s sequential and 98.6 us per random 8 KiB row; /mnt/models 63.7 MB/s and 13,883.6 us, on a 10 Gbps USB link (lsusb -t). /mnt/models is /dev/sdb1 (OSC PSSD, 951 GB, 551 GB free); the NVMe root has 565 GB free. Re-measure at rest before treating 63.7 MB/s as the drive.
- Reaching the box from this workstation: no ssh key and no ~/.ssh/config entry; the route is the operator's tmux session 0 (an open ssh as leandro). Drive it with `tmux send-keys -t 0` and read it with `tmux capture-pane -p -t 0 -S -N`, keeping -S small. Do not type into a pane running htop or a build. Files can be read and written through the gvfs mount at /run/user/1000/gvfs/sftp:host=192.168.1.91,user=leandro (verified this session for read and write).
- Rule 5 exception: the engine builds with `make cuda-spark` (or the H3 variant) on the box, never with /data/build.
- Rule 19: this project has no recorded QA model. Two units carry written waivers (2026-10-18, 2026-10-27); no independent reviewer has seen any unit merged this session.

5. Guardrails and open items

- Global core is ~/.opengrok/AGENTS.md; rules 3, 5, 6, 7, 9, 10, 12, 14, 19 and 23 are in ~/.opengrok/project-work-rules.md. This repository is PUBLIC; the config repo JLFN/open-grok-backup is PRIVATE.
- dev is 43 commits ahead of origin/dev: the push is an OFFER and has not been made.
- OPEN: the MiMo unit is committed but unverified on the box — layout, verify, build, check and start have not run (Phase A).
- OPEN: the MiMo launcher's file carries edits from two hands this session (a second writer removed the bank flags from its parser and added the one-sequence rationale). The merged content is coherent — bash -n, plan, status, preflight and layout were exercised — but a re-read before the box run is warranted.
- OPEN: H3 has never rendered; MiniMax-M3.1-Flash VQ has never been loaded; the GLM 19-layer revision has never been loaded.
- OPEN: the operator's main checkout is still on feature/minimax-h3-sol with today's DS4 edits uncommitted (now duplicated in dev) and a stale handoff probe refresh.
- OPEN: the config repo's unrelated in-flight changes and the stale H3 systemd sentence in ~/.opengrok/AGENTS.md.
- Project-specific guardrail: the MiMo release directory must hold only the 53 parts release.json declares; an undeclared .gguf is a load failure, not a warning.
- Project-specific guardrail: the H3 family has no --ssd-streaming path and reads its checkpoint in full while serving.

6. Progress and staleness check (run this first, every session)

`git -C /data/dgx-recipe log --oneline origin/dev..dev | wc -l` gives the unpushed count; `git branch --no-merged dev` must print nothing at a clean integration boundary, as it did at 09a9e46.
