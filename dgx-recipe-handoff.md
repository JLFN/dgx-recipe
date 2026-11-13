---
project: dgx-recipe
plan_start_commit: 014e441978d15df5e784a756b86e59e6773c7afc
last_updated_commit: 014e441978d15df5e784a756b86e59e6773c7afc
branch: feature/ds41-1017-hvq
remote: origin (https://github.com/JLFN/dgx-recipe.git)
handoff_written_at_context_usage: unknown
handoff_written_at: 2026-11-13T19:58:00Z
session_id: 01a1d630-9d6f-73c1-b6c2-c9e416a979ef
---

# HANDOFF dgx-recipe

Audience: the next agent. This file is executed, not read as prose.

## 1. READ ORDER

1. Run section 6. The branch in the frontmatter is unmerged; work there.
2. Read OPEN, then PLAN. Continue at the first step whose work is absent from the box.

## 2. STATE

Verified this session (2026-11-13).

- unit: DeepSeek-V4.1-Flash-1017 hVQ recipe intake
- outcome: recipes/ds41-1017-hvq.yaml, runbooks/ds41-1017-hvq.md, scripts/install-ds41-1017-hvq.sh (1,196 lines), README row/port/section; commit 014e441 on feature/ds41-1017-hvq above dev da209ac
- verified: bash -n clean; `install-ds41-1017-hvq.sh selftest` PASS against the live manifest (44 entries, 40 part names, 4 identity digests); `plan` renders; recipe yaml parses; part digests equal at pin 0d3cae07 and at 73c97d6 (weights commit)
- verified: ~/.opengrok/config.toml gained [model."deepseek-v4.1-flash-1017-hvq"] at 8014 and parses (21 model entries); prior copy at ~/.local/state/grok-config-backups/config.toml.20261113-194809.bak
- verified: box read through gvfs on 2026-11-13: /mnt/models 953 GB with 675 GB free, $HOME 916 GB with 677 GB free; ~/models/ds41-1017-unc/DeepSeek-V4.1-Flash.vision.gguf present, 971,066,016 B, digest 3a8b194a per the .verified beside it; no model-0004{7,8}-of-00048.safetensors under /mnt/models, $HOME, /data, /opt, /srv or /raid
- unverified: nothing fetched, assembled or loaded; engine floor not exercised against a real engine; rule 19 gate not run
- accepted risk: every speed and fidelity number in the contract is the release author's GB10 run, labelled with its engine commit

## 3. PLAN

STEP 1
  action: put this commit's launcher on the box without a push
  command: from /data/dgx-recipe, cp scripts/install-ds41-1017-hvq.sh "/run/user/1000/gvfs/sftp:host=192.168.1.91,user=leandro/home/leandro/dgx-recipe/scripts/"
  acceptance: sha256sum of the copy on the box equals the local file's
  on_fail: cp the file to ~/dgx-recipe/scripts/ from the operator's shell instead; the launcher is self-contained

STEP 2
  action: prove the box's constants against the release before spending 316 GB
  command: bash ~/dgx-recipe/scripts/install-ds41-1017-hvq.sh selftest
  acceptance: "selftest: PASS" with the 44-entry manifest line
  on_fail: stop; do not download with a launcher that cannot read the manifest

STEP 3
  action: preflight the box
  command: bash ~/dgx-recipe/scripts/install-ds41-1017-hvq.sh preflight
  acceptance: arch/gpu/cuda/memory lines ok, "base volume ok" and "fast volume ok", engine advertises --engram-dir --vision-gguf --mem-budget-mb --mtp-mode, engine floor reported (a warning older than 58738c04 is acceptable; note it)
  on_fail: if the engine is missing the four flags, `cd ~/ds4-dfm-rs && make cuda-spark`; if the engine is older than 58738c04, pull and rebuild to get the 1,662 tok/s prefill path

STEP 4
  action: fetch 316.6 GB (113.6 GB of parts plus 203 GB of tables; the sidecar is reused)
  command: nohup bash ~/dgx-recipe/scripts/install-ds41-1017-hvq.sh download > "$HOME/models/ds41-1017-download.log" 2>&1 &
  acceptance: tail of the log says "sizes ok" and "parts: 40/40 present"; the tables are deepseek-engram/* moved up with a rename
  on_fail: re-run the same command; it resumes. If a table fetch lands one level deep, download adopts it

STEP 5
  action: assemble the base, one hash-checked part at a time
  command: bash ~/dgx-recipe/scripts/install-ds41-1017-hvq.sh assemble
  acceptance: final line "assembled: 113.6 GB, digest d000e79199acd3e443fa6485e4927a0d5f207fa860d0b48f05bbc3dd514f52df"
  on_fail: an interrupted assembly cannot be continued (the parts it consumed are deleted); the script prints the recovery commands

STEP 6
  action: prove the four identity digests and write the record
  command: bash ~/dgx-recipe/scripts/install-ds41-1017-hvq.sh verify
  acceptance: base ok d000e791, vision identity ok 3a8b194a, engram identity ok 76a05106 and 9dc51433, "verification record written"
  on_fail: a MISMATCH names the wrong digest; replace that file, do not serve past it

STEP 7
  action: start and smoke-test
  command: bash ~/dgx-recipe/scripts/install-ds41-1017-hvq.sh start
  acceptance: /v1/models answers on 8014, the smoke request returns choices, "loaded; <n> MB of memory left" with n >= 6000
  on_fail: read the log; the card reports 153 s from launch to resident, 10.5 GiB left

STEP 8
  action: turn the author's numbers into this box's
  files: runbooks/ds41-1017-hvq.md, recipes/ds41-1017-hvq.yaml
  command: time the load, then a greedy 2K and a 16K request; record clock range, date, build
  acceptance: a "what the first start printed" section with MemAvailable, load seconds and two rates; benchmark.measured_here set true only for what was measured
  on_fail: leave measured_here false and say which figure is missing

## 4. FACTS

- paths: workspace = /data/dgx-recipe (this worktree, branch feature/ds41-1017-hvq); dev checked out at /data/dgx-recipe-int; other worktrees /data/dgx-recipe-kv and /data/dgx-recipe-mimo, both at da209ac
- paths: box = gx10-e1d2 192.168.1.91; clone ~/dgx-recipe (last updated 2026-11-07); engine ~/ds4-dfm-rs; HF CLI ~/hfenv/bin/hf (found by ensure_hf)
- paths: base /mnt/models/ds41-1017-hvq; run-time files $HOME/models/ds41-1017-hvq; logs ds41hvq.log and ds41hvq-watchdog.log
- installed: [model."deepseek-v4.1-flash-1017-hvq"] in ~/.opengrok/config.toml, base_url http://192.168.1.91:8014/v1
- measured: /mnt/models 953 GB total, 675 GB free; $HOME 916 GB total, 677 GB free (df through gvfs, 2026-11-13)
- constraint: no tmux binary on this workstation (command -v tmux fails), so the old tmux route is unavailable; the gvfs mount at /run/user/1000/gvfs/sftp:host=192.168.1.91,user=leandro reads the box's whole root and works
- constraint: five 1017 packagings share the base file name and the 113,556,639,424-byte size; only the assembled digest separates them
- constraint: one ds4 model at a time (engine global lock); H3 is a separate process tree
- constraint: rule 19 has no QA model for this project; waivers are in qa-evidence/

## 5. OPEN

- OPEN-1: push feature/ds41-1017-hvq (1 commit, 014e441) and offer the PR | owner: operator | blocked_by: the push offer, not yet answered
- OPEN-2: run PLAN steps 1-8 on the box | owner: agent | blocked_by: a command route (tmux absent; the operator's shell or a fresh ssh route is needed)
- OPEN-3: delete the stale dgx-recipe-handoff.md that dev still tracks from the H3 unit (rule 9 lifecycle: the PR merged) | owner: operator/agent | blocked_by: dev is checked out at /data/dgx-recipe-int
- OPEN-4: decide whether to keep 316.6 GB of Engram tables on the box again, or to accept re-fetching them per run | owner: operator | blocked_by: nothing
- OPEN-5: rule 19 gate has never run for any unit | owner: operator | blocked_by: no QA model recorded

## 6. STALENESS AND PROGRESS

    git log --oneline 014e441978d15df5e784a756b86e59e6773c7afc..HEAD
    git log --oneline origin/dev..HEAD
    git status --short
    git worktree list

- Non-empty first command: newer work exists; read it before trusting STATE.
- `git worktree list` must show dev at /data/dgx-recipe-int; never check out dev in this worktree.
- The box is authoritative for what it holds: read /mnt/models and $HOME/models through the gvfs mount before quoting any number from this file.
