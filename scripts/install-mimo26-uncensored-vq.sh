#!/usr/bin/env bash
# MiMo-V2.6-Flash-RL UNCENSORED VQ: fetch the release, put each part on the
# volume the engine actually reads it from, prove the bytes, build the engine,
# and serve it on the DGX Spark.
#
#   artifact   Baekpica/MiMo-V2.6-Flash-RL-UNCENSORED-VQ-GGUF
#              pinned at 2b65148d9a95a7768ef2e553823c39149ee18c6b (main head,
#              2026-10-31T03:16:14Z). Its 53 GGUF parts are byte-identical to
#              the release commit f1cd8523 and to 136fe729 (compared against the
#              Hub's tree API on 2026-10-31); only README.md and one PNG moved
#              between those commits.
#   layout     53 independent GGUF parts in ONE directory (a GVQ1 release
#              directory, never concatenated): 47 layer-NN-experts.gguf (15 at
#              1.31 GB R13, 32 at 2.12 GB R13+8), body-nonexpert.gguf (5.74 GB),
#              embed-tokens-bf16.gguf (1.25 GB), vision-q8_0.gguf (0.78 GB),
#              audio-q8_0.gguf (0.65 GB), dflash-q8_0.gguf (1.56 GB),
#              mtp-q8_0.gguf (1.05 GB), plus the tables (release.json,
#              files.json, config.json, tokenizer files, chat template).
#              98,520,657,024 bytes of parts.
#   engine     ds4-dfm-rs, family mimo2_vq: PR #127 merged at 446951cc, the
#              performance PR #128 merged at 35f5b462. Built here at 2fa8c08e,
#              the commit the artifact's own DGX Spark numbers were measured on
#              (1,294.9 prefill / 23.96 decode tok/s, 26.35 with two MTP drafts).
#   source     dealignai/MiMo-V2.6-Flash-RL-UNCENSORED at 0bdb11d7, itself
#              XiaomiMiMo/MiMo-V2.6-Flash-RL at f657ba2f with its refusal
#              behaviour removed at the weight level.
#
# THIS SCRIPT'S SHAPE, AND WHERE IT CAME FROM
#   scripts/install-ds41-1017-uncensored-vq.sh is the lineage: same actions, same
#   "read once at load goes on the roomy volume, read while serving stays on the
#   fast device" split, same watchdog, same explicit verification record. Three
#   things differ, and each is a property of this artifact: it is a directory of
#   53 parts rather than an assembled file, so the loader can be handed the
#   directory and every start re-reads all of it; the parts it must NOT see
#   (an undeclared .gguf beside them) are a hard load failure, so this script
#   counts them; and the engine build is pinned into a detached git worktree,
#   because the checkout on this box is mid-unit on another branch with local
#   modifications (feature work for the MiniMax-H3 host at pr-122).
#
# WHERE EACH PIECE GOES, AND WHY
#   --dir (default $HOME/models/mimo26-unc, the NVMe root):
#     embed-tokens-bf16.gguf, the only part the engine reads while serving. Its
#     release entry is residency "ssd-rows" (crates/ds4-core/src/mimo2_vq/
#     release.rs:83, Role::Embedding) and FORMAT.md says the host reads one
#     8,192-byte row per token; docs/mimo2-vq.md says the row comes from the page
#     cache. It is 1.25 GB, so it fits anywhere; it belongs where a cold miss is
#     cheap. Measured on this box 2026-10-31: 98.6 us per random 8 KiB row on
#     the NVMe root, 13,883.6 us on /mnt/models while the fetch was writing.
#     Also here: the disk-KV directory, the logs, the pid files, the verify
#     record.
#   --gguf-dir (default /mnt/models/mimo26-unc, the roomy volume):
#     everything else, because all of it is read exactly twice per cold start and
#     then stays resident. "Twice" is not an estimate: the loader hashes every
#     part with O_DIRECT reads at every open (release.rs:675 verify_dir ->
#     release.rs:557 part_digest) and the device copy then reads the bytes again
#     (docs/mimo2-vq.md: "Engine open copies 86.82 GiB (trunk) to the device"),
#     so one start streams about 2 x 93 GB.
#     THE MEASUREMENT THAT MAKES THIS A REAL CHOICE: on 2026-10-31, with the
#     fetch writing, /mnt/models read 63.7 MB/s sequentially against 2.8 GB/s on
#     the NVMe root, which would make that 2 x 93 GB about 48 minutes per cold
#     start rather than about a minute. `preflight` re-measures both roots at
#     rest and says plainly which layout the numbers support; `--gguf-dir` moves
#     the bulk without touching this code, and when both roots are the same
#     filesystem the split is skipped and nothing is symlinked.
#   The split is a SYMLINK, not a copy: <gguf-dir>/embed-tokens-bf16.gguf points
#   at <dir>/embed/embed-tokens-bf16.gguf. The loader opens the path it read from
#   release.json and stats it (release.rs:512 verify_part, :486 open_dir), so a
#   symlink is read exactly like a file; the name, not the inode, is what the
#   release declares. `layout` and `verify` both prove the target is the
#   published 1,249,903,744-byte file before the engine is pointed at it.
#
# THE MISTAKES THIS SCRIPT EXISTS TO PREVENT
#   1. Downloading into the shared /mnt/models root. 53 loose parts beside
#      ds41-1017-unc and the Bonsai and Qwen directories is how a 98.5 GB release
#      gets mixed with three others; and the release directory must hold only
#      declared parts. `layout` adopts a fetch that landed loose (that is what
#      the operator started on 2026-10-31) and moves the repository's own
#      top-level entries, never anything else.
#   2. A half-finished fetch. `download` refuses to start a second copy while an
#      incomplete file or a live `hf download` sits in the staging root, because
#      the second copy would need the whole 98.5 GB again.
#   3. An undeclared .gguf in the release directory. The loader walks the
#      directory and fails on any .gguf that release.json does not declare
#      (release.rs:392, "unexpected GGUF part"), so a stray layer part from
#      another release is a load failure, not a warning.
#   4. A truncated or foreign part. Every one of the 53 parts is checked against
#      release.json's own size and SHA-256 before the engine is started, and the
#      record written by verify must still match size and mtime at `start`.
#   5. Pointing the engine at the whole repository tree as if the sidecar files
#      were optional. vision-q8_0.gguf and audio-q8_0.gguf load with the model
#      (no --vision flag) and dflash/mtp are selected by flags; the script names
#      each one in `status` so a missing sidecar is visible before a request.
#   6. Building over the operator's checkout. The build is a detached worktree at
#      the pinned commit; the source tree is left exactly as found.
#
# WHAT THIS SCRIPT DOES NOT DO
#   It does not measure the model. The numbers in the recipe and runbook are the
#   artifact author's, measured at 2fa8c08e on this hardware model; `bench`
#   prints how to reproduce them with ds4-bench and where the author's receipts
#   live. Nothing here has been timed on this box.
#
# USAGE
#   bash install-mimo26-uncensored-vq.sh preflight
#   bash install-mimo26-uncensored-vq.sh layout      # adopt a loose fetch, split the embedding onto --dir
#   bash install-mimo26-uncensored-vq.sh download    # 98.5 GB, resumable
#   bash install-mimo26-uncensored-vq.sh verify      # sizes, digests, split, and the record
#   bash install-mimo26-uncensored-vq.sh build       # detached worktree + make cuda-spark
#   bash install-mimo26-uncensored-vq.sh check       # the engine's own --check-config plan
#   bash install-mimo26-uncensored-vq.sh start       # 0.0.0.0:8012
#   bash install-mimo26-uncensored-vq.sh install     # all of the above, then start
#   bash install-mimo26-uncensored-vq.sh stop | status | logs | test | plan | bench
#
# OPTIONS (each is also an environment variable of the same name)
#   --dir DIR          fast-device root: embedding, disk KV, logs, pid, records
#                      (default $HOME/models/mimo26-unc)
#   --gguf-dir DIR     the release directory (default /mnt/models/mimo26-unc)
#   --staging DIR      where a fetch made with --local-dir may sit loose
#                      (default /mnt/models; `layout` moves only this
#                      repository's own entries)
#   --port N           listen port (default 8012)
#   --host ADDR        listen address (default 0.0.0.0)
#   --mtp-mode MODE    off | auto | on, the three embedded MTP blocks
#                      (default on, two drafts; `auto` also loads them, because
#                      this family's MTP support is Qualified)
#   --mtp-draft N      drafts per verify (default 2, the family minimum)
#
# ONE SEQUENCE, NO BANKS. At the operator's instruction this artifact is served
# as a single serial sequence at its native 1,048,576 context with the two
# embedded MTP drafts, so the launcher passes no -c, no --max-seqs, no
# --prefix-reuse and no --kv-disk-dir: the engine's own default plan is exactly
# that shape (docs/mimo2-vq.md, "With no options the server plans the native
# context with two embedded MTP drafts on the serial lane (109.79 GiB), whose
# prefix reuse is exact"). The engine can do opt-in banks for this family, and
# the card's 512K and 256K shapes are those banks; they are not offered here
# because nothing on this box needs them. --mtp-mode off is still sold: it drops
# the MTP part for a text-only 1M plan (108.88 GiB).
#   --engine-src DIR   the ds4-dfm-rs checkout to build from (default
#                      $HOME/ds4-dfm-rs)
#   --engine-tree DIR  the detached worktree the build lands in (default
#                      $HOME/ds4-dfm-rs-mimo)
#   --engine-ref REF   the commit to build (default 2fa8c08e, the measured one)
#   --server-bin PATH  the binary to run (default <--engine-tree>/ds4-server)
#   --hf-bin PATH      the Hugging Face CLI to use
#   --revision SHA     pin another revision of the repository
#   --no-watchdog      do not start the memory watchdog
#   --no-build         `install`: skip build
#   --no-start         `install`: stop before start
#   --allow-shared     start even if another ds4 model process is running
#   --hash-jobs N      parallel sha256 workers during verify (default 8)
#
# THE OTHER PORT OWNERS, so no launcher argues with another: 8000 the V4.1 port
# deployment, 8003 Qwen3.8, 8005 Bonsai, 8006 YoungAi V4.1, 8007 GLM-5.3 VQ,
# 8008 YoungAi 1017, 8009 GLM-5.3 Uncensored VQ, 8010 DeepSeek 1017 Uncensored
# VQ, 8011 MiniMax-M3.1-Flash VQ, 8012 this artifact, 8090 MiniMax-H3 video.
#   - The engine takes a single global lock: only one ds4 model can be resident.

set -Eeuo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
REPO="${REPO:-Baekpica/MiMo-V2.6-Flash-RL-UNCENSORED-VQ-GGUF}"
PIN=2b65148d9a95a7768ef2e553823c39149ee18c6b
REVISION="${REVISION:-$PIN}"

MODEL_DIR="${MODEL_DIR:-$HOME/models/mimo26-unc}"     # fast device
GGUF_DIR="${GGUF_DIR:-/mnt/models/mimo26-unc}"        # roomy volume
STAGING="${STAGING:-/mnt/models}"                     # a loose --local-dir fetch may sit here
ENGINE_SRC="${ENGINE_SRC:-$HOME/ds4-dfm-rs}"
ENGINE_TREE="${ENGINE_TREE:-$HOME/ds4-dfm-rs-mimo}"
ENGINE_REF="${ENGINE_REF:-2fa8c08e0229d463340b0823f2b1d54cfad93087}"
SERVER_BIN="${SERVER_BIN:-}"
SERVER_BIN_EXPLICIT=0
[ -n "$SERVER_BIN" ] && SERVER_BIN_EXPLICIT=1
HOST_ADDR="${HOST_ADDR:-0.0.0.0}"
PORT="${PORT:-8012}"
MTP_MODE="${MTP_MODE:-on}"
MTP_DRAFT="${MTP_DRAFT:-2}"
MAX_SEQS="${MAX_SEQS:-1}"
PREFIX_REUSE="${PREFIX_REUSE:-exact}"
KV_DISK_DIR="${KV_DISK_DIR:-}"
WATCHDOG="${WATCHDOG:-1}"
NO_BUILD="${NO_BUILD:-0}"
NO_START="${NO_START:-0}"
ALLOW_SHARED="${ALLOW_SHARED:-0}"
MAX_WORKERS="${MAX_WORKERS:-8}"
HASH_JOBS="${HASH_JOBS:-8}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"
START_FLOOR_MB="${START_FLOOR_MB:-112000}"
RUN_FLOOR_MB="${RUN_FLOOR_MB:-6000}"
KILL_MB="${KILL_MB:-4096}"

EMBED="embed-tokens-bf16.gguf"
EMBED_BYTES=1249903744
EMBED_SHA=6b9ad974a178980947c3d3faa746cdfd225f2807cabac30ec1eb94e719f26ca3
VISION="vision-q8_0.gguf";       VISION_BYTES=776464384
AUDIO="audio-q8_0.gguf";         AUDIO_BYTES=653195008
DFLASH="dflash-q8_0.gguf";       DFLASH_BYTES=1559875456
MTP="mtp-q8_0.gguf";             MTP_BYTES=1049622912
BODY="body-nonexpert.gguf";      BODY_BYTES=5740191360

PARTS=53
PAYLOAD_BYTES=98520657024       # the 53 parts, from release.json
TRUNK_BYTES=93231595520         # payload minus embed, vision, audio, dflash, mtp
TABLES=(release.json files.json config.json generation_config.json
        tokenizer.json tokenizer_config.json chat_template.jinja
        preprocessor_config.json vocab.json merges.txt
        dflash/config.json audio_tokenizer/config.json)

MODEL_ID="${MODEL_ID:-mimo-v2.6-flash-uncensored-vq}"
PIDF="$MODEL_DIR/mimo26.pid"
SERVER_LOG="$MODEL_DIR/mimo26.log"
WD_PIDF="$MODEL_DIR/mimo26-watchdog.pid"
WD_LOG="$MODEL_DIR/mimo26-watchdog.log"
RECORD="$MODEL_DIR/mimo26-verify.record"
EMBED_FAST="$MODEL_DIR/embed/$EMBED"
TMP="${TMPDIR:-/tmp}/mimo26.$$"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
say()  { printf 'mimo26: %s\n' "$*"; }
warn() { printf 'mimo26: warning: %s\n' "$*" >&2; }
die()  { printf 'mimo26: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
total_mb() { awk '/MemTotal/{print int($2/1024)}' /proc/meminfo; }
fsize()    { stat -L -c %s "$1" 2>/dev/null || echo 0; }
fmtime()   { stat -L -c %Y "$1" 2>/dev/null || echo 0; }
df_avail() { df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' \n' || true; }
df_target(){ df --output=target "$1" 2>/dev/null | tail -1 || true; }
dev_of()   { stat -c %d "$1" 2>/dev/null || echo 0; }
human()    { awk -v b="$1" 'BEGIN{if (b>=1000000000) printf "%.1f GB", b/1000000000; else printf "%.0f MB", b/1000000}'; }
gib()      { awk -v b="${1:-0}" 'BEGIN{if (b ~ /^[0-9]+$/) printf "%.2f GiB", b/1073741824; else printf "%s", b}'; }
sha_file() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
alive()    { [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
cleanup()  { rm -rf "$TMP"; }
trap cleanup EXIT

ensure_dirs() { mkdir -p "$GGUF_DIR" "$MODEL_DIR"; }

# --------------------------------------------------------------------------
# hugging face CLI
# --------------------------------------------------------------------------
ensure_hf() {
    if [ -n "${HF_BIN:-}" ]; then HF="$HF_BIN"; return; fi
    if command -v hf >/dev/null 2>&1 && hf download --help >/dev/null 2>&1; then HF=hf; return; fi
    if command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli; return; fi
    if [ -x "$HOME/hfenv/bin/hf" ]; then HF="$HOME/hfenv/bin/hf"; return; fi
    die "no Hugging Face CLI found (looked for hf, huggingface-cli and ~/hfenv/bin/hf); install huggingface_hub"
}

hf_download() {   # $1 = target dir
    local target="$1"
    export HF_HUB_MAX_WORKERS="$MAX_WORKERS"
    export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
    [ -n "${ENDPOINT:-}" ] && export HF_ENDPOINT="$ENDPOINT"
    [ "${NO_XET:-0}" = 1 ] && export HF_HUB_DISABLE_XET=1
    mkdir -p "$target"
    "$HF" download "$REPO" --revision "$REVISION" --local-dir "$target" --max-workers "$MAX_WORKERS"
}

# --------------------------------------------------------------------------
# the release's own inventory: release.json is the entry point and the authority
# --------------------------------------------------------------------------
have_release_json() { [ -s "$GGUF_DIR/release.json" ]; }
need_release_json() { have_release_json || die "$GGUF_DIR/release.json is missing; run: bash $0 download"; }

# name<TAB>sha256<TAB>bytes, in release order (47 expert parts, body, mtp,
# then the embedding last - the order the loader itself builds the inventory in)
part_table() { jq -r '.parts[] | [.path,.sha256,(.bytes|tostring)] | @tsv' "$GGUF_DIR/release.json"; }
role_of()    { jq -r --arg p "$1" '.parts[] | select(.path==$p) | .role' "$GGUF_DIR/release.json"; }

parts_present() {
    local n=0 name want bytes
    while IFS=$'\t' read -r name want bytes; do
        if [ "$(fsize "$GGUF_DIR/$name")" = "$bytes" ]; then n=$((n + 1)); fi
    done < <(part_table)
    printf '%s' "$n"
}

# A .gguf the release does not declare is a load failure in the engine
# (release.rs:392). Count them instead of predicting them.
undeclared_gguf() {
    local name
    while IFS= read -r name; do
        jq -e --arg p "$name" '.parts[]|select(.path==$p)' "$GGUF_DIR/release.json" >/dev/null 2>&1 || printf '%s\n' "$name"
    done < <(find "$GGUF_DIR" -maxdepth 1 -name '*.gguf' -printf '%f\n' 2>/dev/null)
}

# The release directory is a directory of declared parts and tables, so a
# leftover of an interrupted fetch is visible as a .incomplete file.
staging_partials() { find "$STAGING" -maxdepth 4 -name '*.incomplete' 2>/dev/null | wc -l; }
hf_running() { pgrep -af "hf download|huggingface-cli download" | grep -v "$$" || true; }

# --------------------------------------------------------------------------
# preflight: the facts the rest of the script will act on
# --------------------------------------------------------------------------
seq_read_mbps() {   # $1 = a directory; reads up to 512 MiB from a file in it, O_DIRECT
    local dir="$1" f
    f=$(find "$dir" -maxdepth 2 -type f -size +1G 2>/dev/null | head -1)
    [ -n "$f" ] || { printf 'n/a'; return; }
    dd if="$f" of=/dev/null bs=1M count=512 iflag=direct 2>&1 | tail -1 | awk '
        { for (i = 1; i <= NF; i++) {
              if ($i == "s," && $(i - 1) ~ /^[0-9.]+$/) { secs = $(i - 1) }
              if ($i ~ /^(GB|MB|kB)\/s$/) { val = $(i - 1); unit = $i } } }
        unit == "GB/s" { printf "%.1f MB/s", val * 1000 }
        unit == "MB/s" { printf "%.1f MB/s", val }
        unit == "kB/s" { printf "%.1f MB/s", val / 1000 }
        END { if (unit == "") printf "n/a" }'
}

cmd_preflight() {
    need curl; need sha256sum; need jq
    say "host:        $(uname -m) $(uname -r), $(total_mb) MB total, $(avail_mb) MB available"
    command -v nvidia-smi >/dev/null 2>&1 \
        && say "gpu:         $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)" \
        || warn "nvidia-smi is not on PATH"
    ensure_dirs
    local fast_free roomy_free
    fast_free=$(df_avail "$MODEL_DIR"); fast_free=${fast_free:-0}
    roomy_free=$(df_avail "$GGUF_DIR"); roomy_free=${roomy_free:-0}
    say "fast root:   $MODEL_DIR  ($(df_target "$MODEL_DIR"), $(human "$fast_free") free)"
    say "roomy root:  $GGUF_DIR  ($(df_target "$GGUF_DIR"), $(human "$roomy_free") free)"
    say "payload:     $(human "$PAYLOAD_BYTES") of parts; the embedding alone is $(human "$EMBED_BYTES")"
    local same=0; [ "$(dev_of "$MODEL_DIR")" = "$(dev_of "$GGUF_DIR")" ] && same=1
    if [ "$same" = 1 ]; then
        say "split:       both roots are the same filesystem; no symlink will be made"
        [ "$roomy_free" -gt $((PAYLOAD_BYTES + 1000000000)) ] \
            || die "$(df_target "$GGUF_DIR") has $(human "$roomy_free") free; the release needs $(human "$PAYLOAD_BYTES")"
    else
        say "split:       $EMBED will live in $MODEL_DIR/embed and be symlinked into the release directory"
        [ "$roomy_free" -gt $((PAYLOAD_BYTES - EMBED_BYTES + 1000000000)) ] \
            || die "$(df_target "$GGUF_DIR") has $(human "$roomy_free") free; the parts need $(human $((PAYLOAD_BYTES - EMBED_BYTES)))"
        [ "$fast_free" -gt $((EMBED_BYTES * 2)) ] \
            || die "$(df_target "$MODEL_DIR") has $(human "$fast_free") free; the embedding needs $(human "$EMBED_BYTES")"
    fi
    # The measured reason the split exists. Label what was measured and when.
    local roomy_seq fast_seq
    roomy_seq=$(seq_read_mbps "$GGUF_DIR"); fast_seq=$(seq_read_mbps "$MODEL_DIR")
    say "read speed:  roomy $roomy_seq, fast $fast_seq  (O_DIRECT, up to 4 GiB, measured now; a concurrent fetch depresses both)"
    say "             the loader streams about $(human $((TRUNK_BYTES * 2))) per cold start: one hashing pass plus the device copy"
    say "start:       need $(human $((START_FLOOR_MB * 1000000))) available; have $(human $(( $(avail_mb) * 1000000 )))"
    local owners; owners=$(pgrep -af "ds4-server" | grep -v "$$" || true)
    if [ -n "$owners" ]; then
        warn "another ds4-server is running (the engine takes a single global lock):"
        printf '%s\n' "$owners" | sed 's/^/             /' >&2
    fi
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | awk -v p=":$PORT\$" '$4 ~ p {print}' | grep -q . \
            && warn "port $PORT is already listening" || say "port:        $PORT is free"
    fi
    [ -d "$ENGINE_SRC/.git" ] && say "engine src:  $ENGINE_SRC at $(git -C "$ENGINE_SRC" rev-parse --short HEAD 2>/dev/null)" \
        || warn "no git checkout at $ENGINE_SRC; build will need --engine-src"
    local tree_head="absent"
    [ -d "$ENGINE_TREE/.git" ] && tree_head=$(git -C "$ENGINE_TREE" rev-parse --short HEAD 2>/dev/null || echo unknown)
    say "engine tree: $ENGINE_TREE ($tree_head); pinned ref $(printf '%.8s' "$ENGINE_REF")"
    # What is already on the box.
    if have_release_json; then
        say "release:     $(parts_present)/$PARTS parts at the published sizes in $GGUF_DIR"
        local extra; extra=$(undeclared_gguf)
        [ -z "$extra" ] || warn "undeclared .gguf in the release directory (a load failure): $(printf '%s ' $extra)"
    else
        say "release:     no release.json in $GGUF_DIR yet"
    fi
    if [ -d "$STAGING" ]; then
        if [ -s "$STAGING/release.json" ]; then
            say "staging:     a fetch sits loose in $STAGING ($(find "$STAGING" -maxdepth 1 -name '*.gguf' | wc -l) *.gguf at the top level); run: layout"
        fi
        local live; live=$(hf_running)
        [ -z "$live" ] || warn "a Hugging Face download is running; layout and download will not touch the staging root until it exits"
        local inc; inc=$(staging_partials)
        [ "$inc" = 0 ] || say "staging:     $inc incomplete file(s) - the fetch is in flight or was interrupted; re-running download resumes it"
    fi
    if [ -f "$RECORD" ]; then
        if record_matches; then say "verified:    the record matches the bytes on disk"
        else say "verified:    a record exists but is stale against the bytes on disk; re-run verify"; fi
    else
        say "verified:    no record yet; run: verify"
    fi
}

# --------------------------------------------------------------------------
# layout: adopt a loose fetch, then put the embedding on the fast root
# --------------------------------------------------------------------------
# The repository's own top-level entries. Only these are moved out of a staging
# root, so nothing that belongs to another artifact is touched.
REPO_TOP=(.gitattributes FORMAT.md LICENSE-YoungAi README.md SHA256SUMS
          audio-q8_0.gguf body-nonexpert.gguf chat_template.jinja config.json
          dflash-q8_0.gguf embed-tokens-bf16.gguf files.json
          generation_config.json memory-budget.json merges.txt mtp-q8_0.gguf
          preprocessor_config.json recipe.json release.json
          tensor-manifest.json tokenizer.json tokenizer_config.json
          vision-q8_0.gguf vocab.json
          audio_tokenizer assets calibration conversion-receipts dflash
          quality tools .cache)

cmd_layout() {
    ensure_dirs
    local src="$STAGING"
    if [ "$(realpath -m "$src")" = "$(realpath -m "$GGUF_DIR")" ]; then
        say "the staging root is the release directory; nothing to adopt"
    else
        local live; live=$(hf_running)
        if [ -n "$live" ]; then
            say "a Hugging Face download is running; not moving files under it:"
            printf '%s\n' "$live" | sed 's/^/  /'
            say "wait for it to exit (or stop it), then run layout again"
            return 0
        fi
        if [ ! -s "$src/release.json" ]; then
            say "no loose fetch at $src; nothing to adopt"
        else
            local moved=0 name base
            for name in "${REPO_TOP[@]}"; do
                [ -e "$src/$name" ] || continue
                [ -e "$GGUF_DIR/$name" ] && continue
                if mv "$src/$name" "$GGUF_DIR/"; then moved=$((moved + 1)); fi
            done
            for name in "$src"/layer-??-experts.gguf; do
                [ -e "$name" ] || continue
                base=$(basename "$name")
                [ -e "$GGUF_DIR/$base" ] && continue
                if mv "$name" "$GGUF_DIR/"; then moved=$((moved + 1)); fi
            done
            say "adopted $moved entries from $src into $GGUF_DIR"
            local left; left=$(find "$src" -maxdepth 1 -name '*.gguf' 2>/dev/null | wc -l)
            [ "$left" = 0 ] || warn "$left *.gguf file(s) remain at the top of $src; they are not this repository's"
        fi
    fi
    if [ -e "$GGUF_DIR/$EMBED" ]; then
        split_embedding
        say "release:     $(parts_present)/$PARTS parts at the published sizes"
    else
        say "release:     no $EMBED in $GGUF_DIR yet; run download"
    fi
}

split_embedding() {
    local link="$GGUF_DIR/$EMBED"
    if [ "$(dev_of "$MODEL_DIR")" = "$(dev_of "$GGUF_DIR")" ]; then
        say "split:       same filesystem; the embedding stays in the release directory"
        return 0
    fi
    mkdir -p "$(dirname "$EMBED_FAST")"
    if [ -L "$link" ]; then
        local target; target=$(readlink -f "$link")
        if [ "$target" = "$(readlink -f "$EMBED_FAST")" ] && [ "$(fsize "$link")" = "$EMBED_BYTES" ]; then
            say "split:       $link -> $target ($(human "$(fsize "$link")")); already in place"
            return 0
        fi
        die "$link is a symlink to $target, not to $EMBED_FAST ($(human "$EMBED_BYTES")). Fix it by hand: this script will not delete a file it cannot account for."
    fi
    [ "$(fsize "$link")" = "$EMBED_BYTES" ] \
        || die "$GGUF_DIR/$EMBED is $(fsize "$link") bytes, not $EMBED_BYTES; run download"
    [ "$(fsize "$EMBED_FAST")" = "$EMBED_BYTES" ] \
        || [ ! -e "$EMBED_FAST" ] \
        || die "$EMBED_FAST exists at $(fsize "$EMBED_FAST") bytes, not $EMBED_BYTES; move it away by hand"
    [ "$(df_avail "$MODEL_DIR")" -gt $((EMBED_BYTES + 100000000)) ] \
        || die "$(df_target "$MODEL_DIR") has $(human "$(df_avail "$MODEL_DIR")") free; the copy needs $(human "$EMBED_BYTES") plus room"
    say "split:       moving $EMBED ($(human "$EMBED_BYTES")) to $EMBED_FAST and linking it back"
    cp -f "$link" "$EMBED_FAST.part" && mv -f "$EMBED_FAST.part" "$EMBED_FAST"
    rm -f "$link"
    ln -s "$EMBED_FAST" "$link"
    [ "$(fsize "$link")" = "$EMBED_BYTES" ] || die "the symlink does not resolve to the published size; check $EMBED_FAST"
    rm -f "$RECORD"      # the bytes moved, so any earlier record is stale
    say "split:       $(readlink "$link") ($(fsize "$link") bytes through the link)"
}

# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------
cmd_download() {
    ensure_hf
    ensure_dirs
    if [ -s "$STAGING/release.json" ] && [ "$(realpath -m "$STAGING")" != "$(realpath -m "$GGUF_DIR")" ]; then
        die "a fetch is already sitting loose in $STAGING; adopting it is free and fetching again is not.
       Run: bash $0 layout
       then: bash $0 download  (it resumes what layout moved)"
    fi
    local live; live=$(hf_running)
    [ -z "$live" ] || die "a Hugging Face download is already running:
$live
       stop it first (Ctrl-C in its pane) so two processes do not write the same files"
    if have_release_json && [ "$(parts_present)" = "$PARTS" ] && tables_ok; then
        say "already downloaded: all $PARTS parts at the published sizes and every table in $GGUF_DIR"
        return 0
    fi
    say "fetching $REPO at $(printf '%.8s' "$REVISION") into $GGUF_DIR ($(human "$PAYLOAD_BYTES"))"
    say "  resumable: re-run the same command; the CLI prints its own progress bars"
    hf_download "$GGUF_DIR" || die "the fetch failed; run the same command again to resume"
    local n; n=$(parts_present)
    [ "$n" = "$PARTS" ] || die "the fetch exited but only $n/$PARTS parts are at the published sizes; re-run to resume"
    tables_ok || die "the parts are complete but a table is missing ($(missing_tables | tr '\n' ' ')); re-run to resume"
    say "download ok: $PARTS parts, $(human "$PAYLOAD_BYTES")"
    say "next: bash $0 verify"
}

tables_ok() {
    local t
    for t in "${TABLES[@]}"; do [ -s "$GGUF_DIR/$t" ] || return 1; done
    return 0
}
missing_tables() { local t; for t in "${TABLES[@]}"; do [ -s "$GGUF_DIR/$t" ] || printf '%s\n' "$t"; done; }

# --------------------------------------------------------------------------
# verify: sizes, then digests, then the split, then a record that start checks
# --------------------------------------------------------------------------
cmd_verify() {
    need sha256sum; need jq
    need_release_json
    ensure_dirs
    tables_ok || die "missing table(s): $(missing_tables | tr '\n' ' '); run download"

    # 1. Presence at the published size first, so a truncated part is named
    #    before 90 GB of hashing starts.
    local name want got bytes i=0 bad=0
    local -a names=() shas=() sizes=()
    while IFS=$'\t' read -r name want bytes; do
        names+=("$name"); shas+=("$want"); sizes+=("$bytes")
    done < <(part_table)
    [ "${#names[@]}" = "$PARTS" ] || die "release.json declares ${#names[@]} parts, not $PARTS: $GGUF_DIR/release.json is not this release's"
    for name in "${names[@]}"; do
        i=$((i + 1))
        got=$(fsize "$GGUF_DIR/$name")
        [ "$got" = "${sizes[$((i - 1))]}" ] \
            || die "  [$i/$PARTS] $name is $got bytes, not ${sizes[$((i - 1))]}; run download"
    done
    say "sizes:       all $PARTS parts match release.json ($(human "$PAYLOAD_BYTES"))"

    # 2. No undeclared .gguf. The loader fails on one (release.rs:392).
    local extra; extra=$(undeclared_gguf)
    [ -z "$extra" ] || die "undeclared .gguf in $GGUF_DIR (the engine rejects the directory):
$(printf '  %s\n' $extra)"

    # 3. The split. The embedding is the one part read while serving.
    local link="$GGUF_DIR/$EMBED" target
    if [ -L "$link" ]; then
        target=$(readlink -f "$link")
        [ -n "$target" ] || die "$link is a dangling symlink; run layout"
        [ "$(fsize "$target")" = "$EMBED_BYTES" ] || die "$target is $(fsize "$target") bytes, not $EMBED_BYTES"
        case "$target" in
            "$(realpath -m "$MODEL_DIR")"/*) say "split:       $EMBED -> $target (fast root)" ;;
            *) warn "the embedding resolves to $target, outside $MODEL_DIR; the fast root is the point of the split" ;;
        esac
    else
        say "split:       $EMBED is a regular file in the release directory"
    fi

    # 4. Digests, in parallel, printing one line per part as it finishes.
    say "hashing:     $PARTS parts ($(human "$PAYLOAD_BYTES")) with $HASH_JOBS workers; one line per part"
    local out="$TMP/hash.out"; : >"$out"
    ( cd "$GGUF_DIR" && printf '%s\n' "${names[@]}" | xargs -r -P "$HASH_JOBS" -n 1 sha256sum ) >"$out" 2>&1 || true
    i=0
    for name in "${names[@]}"; do
        i=$((i + 1))
        want="${shas[$((i - 1))]}"
        got=$(awk -v p="$name" '$2==p{print $1}' "$out")
        if [ "$got" = "$want" ]; then
            [ $((i % 8)) -eq 0 ] && say "  [$i/$PARTS] ok"
        else
            warn "  [$i/$PARTS] $name MISMATCH: ${got:-no digest}, published $want"
            bad=$((bad + 1))
        fi
    done
    [ "$bad" = 0 ] || die "$bad part(s) do not match release.json; delete them and re-run download"

    # 5. The tables, and the embedding's own digest (hashed through the link, so
    #    the split is proven, not assumed).
    for name in "${TABLES[@]}"; do
        want=$(jq -r --arg p "$name" '.files[]? | select(.path==$p) | .sha256' "$GGUF_DIR/files.json" 2>/dev/null | head -1)
        [ -n "$want" ] || want=$(awk -v p="$name" '$2==p{print $1}' "$GGUF_DIR/SHA256SUMS" 2>/dev/null | head -1)
        [ -n "$want" ] || { warn "  $name is not listed in files.json or SHA256SUMS"; bad=$((bad + 1)); continue; }
        got=$(sha_file "$GGUF_DIR/$name")
        [ "$got" = "$want" ] || { warn "  $name MISMATCH: $got, published $want"; bad=$((bad + 1)); }
    done
    got=$(sha_file "$link")
    [ "$got" = "$EMBED_SHA" ] || { warn "  $EMBED MISMATCH: $got, published $EMBED_SHA"; bad=$((bad + 1)); }
    [ "$bad" = 0 ] || die "$bad digest(s) failed; the bytes on disk are not this release"

    # 6. The record. `start` re-checks sizes and mtimes, not 98 GB of hashing.
    {
        printf '# mimo26-uncensored-vq verify record\n'
        printf '# repo %s\n# revision %s\n# written %s\n' "$REPO" "$REVISION" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        for name in "${names[@]}"; do
            if [ "$name" = "$EMBED" ] && [ -L "$link" ]; then
                printf '%s  %s  %s  %s\n' "$EMBED_SHA" "$(fsize "$target")" "$(fmtime "$target")" "$target"
            else
                printf '%s  %s  %s  %s\n' "$(awk -v p="$name" '$2==p{print $1}' "$out")" "$(fsize "$GGUF_DIR/$name")" "$(fmtime "$GGUF_DIR/$name")" "$GGUF_DIR/$name"
            fi
        done
        for name in "${TABLES[@]}"; do
            printf '%s  %s  %s  %s\n' "$(sha_file "$GGUF_DIR/$name")" "$(fsize "$GGUF_DIR/$name")" "$(fmtime "$GGUF_DIR/$name")" "$GGUF_DIR/$name"
        done
    } >"$RECORD"
    say "verified:    $PARTS parts and ${#TABLES[@]} tables match the published digests"
    say "record:      $RECORD (start checks it against the bytes on disk)"
}

# The record must still describe what is on disk: same path, same size, same
# mtime. A rewritten part changes its mtime, so this cannot be fooled by a
# same-size substitution.
record_matches() {
    [ -s "$RECORD" ] || return 1
    local sha bytes mtime path n=0
    while read -r sha bytes mtime path; do
        case "${sha:-#}" in \#*) continue ;; esac
        [ -f "$path" ] || return 1
        [ "$(fsize "$path")" = "$bytes" ] || return 1
        [ "$(fmtime "$path")" = "$mtime" ] || return 1
        n=$((n + 1))
    done <"$RECORD"
    [ "$n" = $((PARTS + ${#TABLES[@]})) ]
}

# --------------------------------------------------------------------------
# build: a detached worktree at the measured commit, never the operator's tree
# --------------------------------------------------------------------------
cmd_build() {
    need git; need make
    [ -d "$ENGINE_SRC/.git" ] || die "$ENGINE_SRC is not a git checkout of ds4-dfm-rs"
    say "fetching the engine: git -C $ENGINE_SRC fetch origin"
    git -C "$ENGINE_SRC" fetch origin --quiet || die "the fetch failed; check the checkout's remote"
    git -C "$ENGINE_SRC" cat-file -e "$ENGINE_REF^{commit}" 2>/dev/null \
        || die "$ENGINE_REF is not in $ENGINE_SRC after the fetch"
    local short; short=$(git -C "$ENGINE_SRC" rev-parse --short "$ENGINE_REF")
    if [ -e "$ENGINE_TREE/.git" ]; then
        say "switching the build worktree to $short"
        git -C "$ENGINE_TREE" checkout --detach --quiet "$ENGINE_REF" \
            || die "could not switch $ENGINE_TREE to $ENGINE_REF; remove it and re-run build"
    else
        say "creating a detached worktree at $ENGINE_TREE ($short)"
        say "  the source tree stays on its branch with its local changes untouched"
        git -C "$ENGINE_SRC" worktree add --detach --quiet "$ENGINE_TREE" "$ENGINE_REF" \
            || die "could not create the worktree at $ENGINE_TREE"
    fi
    say "building with: make cuda-spark  (in $ENGINE_TREE)"
    ( cd "$ENGINE_TREE" && make cuda-spark ) || die "the engine build failed"
    [ -x "$SERVER_BIN" ] || die "the build finished but $SERVER_BIN is not executable"
    say "built:       $SERVER_BIN ($(fsize "$SERVER_BIN") bytes, $(git -C "$ENGINE_TREE" rev-parse --short HEAD))"
}

# --------------------------------------------------------------------------
# check: the engine's own plan, before anything is copied to the device
# --------------------------------------------------------------------------
check_args() {
    local -a a=(--cuda -m "$GGUF_DIR" --mtp-mode "$MTP_MODE" --mtp-draft "$MTP_DRAFT"
                --check-config)
    printf '%s\0' "${a[@]}"
}

cmd_check() {
    [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN; run: build"
    have_release_json || die "$GGUF_DIR/release.json is missing; run download"
    local -a args=()
    while IFS= read -r -d '' x; do args+=("$x"); done < <(check_args)
    mkdir -p "$MODEL_DIR"
    local json="$MODEL_DIR/check.json" report="$MODEL_DIR/check.report" rc=0
    say "running: $SERVER_BIN ${args[*]}"
    "$SERVER_BIN" "${args[@]}" >"$json" 2>"$report" || rc=$?
    sed 's/^/  /' "$report"
    if [ "$rc" != 0 ]; then
        say "the engine refused the plan (exit $rc); the JSON is in $json"
        return 1
    fi
    local family mode draft total avail issues
    family=$(jq -r '.family' "$json"); mode=$(jq -r '.effective.mtp_mode' "$json")
    draft=$(jq -r '.effective.mtp_draft' "$json")
    total=$(jq -r '.quote.total' "$json"); avail=$(jq -r '.quote.available' "$json")
    issues=$(jq -r '[.issues[]|select(.level=="error")]|length' "$json")
    say "family:      $family"
    say "effective:   mtp=$mode draft=$draft max_seqs=$(jq -r '.effective.max_seqs' "$json") ctx=$(jq -r '.effective.ctx' "$json")"
    say "plan:        $(gib "$total") of $(gib "$avail") available (the engine's own quote, bytes in $json)"
    say "             the card's published default plan is 109.79 GiB of 117.6 GiB on a GB10"
    [ "$family" = "MiMo-V2.6-Flash-VQ" ] || warn "the engine reports family '$family', not MiMo-V2.6-Flash-VQ"
    [ "$issues" = 0 ] || { jq -r '.issues[]|select(.level=="error")|"  error: \(.message) (\(.code))"' "$json"; return 1; }
    say "check ok:    the release opens and the plan fits"
}

# --------------------------------------------------------------------------
# daemon
# --------------------------------------------------------------------------
watchdog() {   # $1 pid, $2 floor MB, $3 log file
    local pid="$1" bad=0 a
    while kill -0 "$pid" 2>/dev/null; do
        a=$(avail_mb)
        if [ "$a" -lt "$2" ]; then bad=$((bad + 1)); else bad=0; fi
        if [ "$bad" -ge 2 ]; then
            printf '%s MemAvailable %s MB below %s MB twice; stopping the server\n' "$(date)" "$a" "$2" >>"$3"
            kill "$pid" 2>/dev/null || true
            return 0
        fi
        sleep 5
    done
}

cmd_start() {
    have_release_json || die "no release.json in $GGUF_DIR; run: layout && download"
    [ "$(parts_present)" = "$PARTS" ] || die "$(parts_present)/$PARTS parts are at the published sizes in $GGUF_DIR; run download"
    tables_ok || die "missing table(s): $(missing_tables | tr '\n' ' '); run download"
    local extra; extra=$(undeclared_gguf)
    [ -z "$extra" ] || die "undeclared .gguf in $GGUF_DIR, which the engine rejects: $(printf '%s ' $extra)"
    record_matches \
        || die "no verification record matches the bytes on disk ($RECORD).
       Run: bash $0 verify
       It hashes all $PARTS parts ($(human "$PAYLOAD_BYTES")) and the tables, and proves the embedding's split."
    [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN; run: build"
    need curl
    alive && die "already running (pid $(cat "$PIDF")); stop it first with: bash $0 stop"
    local owners; owners=$(pgrep -af "ds4-server" | grep -v "$$" || true)
    if [ -n "$owners" ] && [ "$ALLOW_SHARED" != 1 ]; then
        printf 'mimo26: another ds4 model is running:\n%s\n' "$owners" >&2
        die "stop it first (the engine takes a single global lock), or set ALLOW_SHARED=1 to overrule"
    fi
    local a; a=$(avail_mb)
    [ "$a" -ge "$START_FLOOR_MB" ] || die "only $a MB available, need $START_FLOOR_MB: stop another model first"
    mkdir -p "$MODEL_DIR"
    # One serial sequence at the artifact's own context: no -c, no --max-seqs,
    # no --prefix-reuse, no --kv-disk-dir. Those defaults are the plan the card
    # quotes (109.79 GiB) and the shape the operator asked for.
    local -a args=(--cuda -m "$GGUF_DIR" --mtp-mode "$MTP_MODE" --mtp-draft "$MTP_DRAFT"
                   --host "$HOST_ADDR" --port "$PORT"
                   --model-id "$MODEL_ID" --no-update-check)
    say "starting ds4-server on $HOST_ADDR:$PORT"
    say "  release: $GGUF_DIR ($PARTS parts, $(human "$PAYLOAD_BYTES"); the embedding is $([ -L "$GGUF_DIR/$EMBED" ] && readlink "$GGUF_DIR/$EMBED" || echo "in place"))"
    say "  log:     $SERVER_LOG"
    say "  a cold start hashes $(human "$TRUNK_BYTES") with O_DIRECT and then copies it to the device;"
    say "  the plan is the card's 109.79 GiB and the artifact author reports about two minutes on NVMe"
    setsid nohup "$SERVER_BIN" "${args[@]}" >"$SERVER_LOG" 2>&1 </dev/null &
    echo $! >"$PIDF"
    if [ "$WATCHDOG" = 1 ]; then
        export -f watchdog avail_mb
        nohup bash -c "watchdog $(cat "$PIDF") $KILL_MB '$WD_LOG'" >/dev/null 2>&1 </dev/null &
        echo $! >"$WD_PIDF"
        say "  watchdog pid $(cat "$WD_PIDF"), stops the server if MemAvailable stays below ${KILL_MB} MB"
    fi
    local waited=0
    say "waiting for the endpoint (up to ${READY_TIMEOUT}s)"
    while [ "$waited" -lt "$READY_TIMEOUT" ]; do
        if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then break; fi
        if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then
            tail -n 20 "$SERVER_LOG" 2>/dev/null | sed 's/^/  /' >&2
            die "the server exited while loading (full log: $SERVER_LOG)"
        fi
        sleep 5; waited=$((waited + 5))
        [ $((waited % 60)) -eq 0 ] && say "  still loading (${waited}s); tail: $(tail -n 1 "$SERVER_LOG" 2>/dev/null | cut -c1-100)"
    done
    [ "$waited" -lt "$READY_TIMEOUT" ] || { tail -n 20 "$SERVER_LOG" | sed 's/^/  /' >&2; die "no answer on port $PORT within ${READY_TIMEOUT}s"; }
    a=$(avail_mb)
    if [ "$a" -lt "$RUN_FLOOR_MB" ]; then
        cmd_stop >/dev/null 2>&1 || true
        die "only $a MB left after loading (need $RUN_FLOOR_MB); the machine would start swapping"
    fi
    say "loaded in ${waited}s; ${a} MB of memory left"
    cmd_test || warn "the smoke request failed; the server is up, so read $SERVER_LOG"
    local ip; ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    say "ready: http://127.0.0.1:$PORT/v1   (lan http://${ip:-<host>}:$PORT/v1, port $PORT)"
}

cmd_stop() {
    [ -s "$WD_PIDF" ] && kill "$(cat "$WD_PIDF")" 2>/dev/null || true
    if alive; then
        local pid; pid=$(cat "$PIDF")
        say "stopping ds4-server (pid $pid)"
        kill "$pid" 2>/dev/null || true
        local i=0
        while [ "$i" -lt 30 ] && kill -0 "$pid" 2>/dev/null; do sleep 1; i=$((i + 1)); done
        kill -0 "$pid" 2>/dev/null && { say "  still alive after 30 s, sending KILL"; kill -9 "$pid" 2>/dev/null || true; }
    else
        say "ds4-server: not running"
    fi
    rm -f "$PIDF" "$WD_PIDF"
}

cmd_status() {
    if alive; then
        say "ds4-server:  running (pid $(cat "$PIDF"))"
        ps -o pid=,etime=,rss= -p "$(cat "$PIDF")" 2>/dev/null | sed 's/^/  /'
    else
        say "ds4-server:  not running"
    fi
    [ -s "$WD_PIDF" ] && kill -0 "$(cat "$WD_PIDF")" 2>/dev/null \
        && say "watchdog:    running (pid $(cat "$WD_PIDF"))" || say "watchdog:    not running"
    say "memory:      $(avail_mb) MB available of $(total_mb) MB"
    if have_release_json; then
        say "release:     $(parts_present)/$PARTS parts at the published sizes in $GGUF_DIR"
        local t missing=()
        for t in "${TABLES[@]}"; do [ -s "$GGUF_DIR/$t" ] || missing+=("$t"); done
        [ "${#missing[@]}" = 0 ] && say "tables:      present" || say "tables:      missing ${missing[*]}"
        say "split:       $([ -L "$GGUF_DIR/$EMBED" ] && readlink "$GGUF_DIR/$EMBED" || echo "embedding in the release directory")"
        local extra; extra=$(undeclared_gguf)
        [ -z "$extra" ] && say "contents:    no undeclared .gguf" || say "contents:    UNDECLARED .gguf: $(printf '%s ' $extra)"
        say "sidecars:    vision $([ "$(fsize "$GGUF_DIR/$VISION")" = "$VISION_BYTES" ] && echo ok || echo missing), audio $([ "$(fsize "$GGUF_DIR/$AUDIO")" = "$AUDIO_BYTES" ] && echo ok || echo missing), dflash $([ "$(fsize "$GGUF_DIR/$DFLASH")" = "$DFLASH_BYTES" ] && echo ok || echo missing), mtp $([ "$(fsize "$GGUF_DIR/$MTP")" = "$MTP_BYTES" ] && echo ok || echo missing)"
    else
        say "release:     no release.json in $GGUF_DIR"
    fi
    say "verified:    $(if record_matches; then echo "yes - the record matches the bytes on disk"; elif [ -f "$RECORD" ]; then echo "no - the record is stale; re-run verify"; else echo "no record; run: verify"; fi)"
    say "engine:      $([ -x "$SERVER_BIN" ] && echo "$SERVER_BIN ($(git -C "$ENGINE_TREE" rev-parse --short HEAD 2>/dev/null || echo unknown))" || echo "not built")"
    say "shape:       max_seqs=$MAX_SEQS mtp=$MTP_MODE/$MTP_DRAFT reuse=$PREFIX_REUSE $([ "$MAX_SEQS" = 1 ] && echo "serial lane, no banks" || echo "bank lane, drafting off by design")"
    say "record:      $([ -f "$RECORD" ] && (record_matches && echo "fresh (size and mtime match)" || echo "stale") || echo "absent")"
    if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | head -c 300; then
        say ""; say "endpoint:    answering on port $PORT"
    else
        say "endpoint:    no answer on port $PORT"
    fi
    if [ -f "$SERVER_LOG" ]; then say ""; say "log tail:"; tail -n 5 "$SERVER_LOG" | sed 's/^/  /'; fi
}

cmd_logs() {
    say "server log:   $SERVER_LOG"
    [ -f "$SERVER_LOG" ] && tail -n 30 "$SERVER_LOG" | sed 's/^/  /'
    say ""
    say "watchdog log: $WD_LOG"
    [ -f "$WD_LOG" ] && tail -n 10 "$WD_LOG" | sed 's/^/  /'
    return 0
}

cmd_test() {
    need curl
    local mid out
    mid=$(curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
          | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
    mid=${mid:-$MODEL_ID}
    say "smoke request against model id '$mid' (read from /v1/models)"
    out=$(curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$mid\",\"temperature\":0,\"top_p\":0.95,\"max_tokens\":96,\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, what does a vector-quantized expert layer store?\"}]}")
    case "$out" in
        *'"choices"'*)
            printf '%s\n' "$out" | sed -n 's/.*"usage":{\(.*\)}.*/  usage: \1/p' | head -1
            printf '%s\n' "$out" | sed -n 's/.*"content":"\([^"]\{0,120\}\)".*/  answer: \1.../p' | head -1;;
        *) warn "the request failed: $(printf '%s' "$out" | head -c 400)"; return 1;;
    esac
}

cmd_plan() {
    ensure_dirs
    say "release dir:  $GGUF_DIR  ($(df_target "$GGUF_DIR"), $(human "$(df_avail "$GGUF_DIR")") free)"
    say "fast root:    $MODEL_DIR  ($(df_target "$MODEL_DIR"), $(human "$(df_avail "$MODEL_DIR")") free)"
    say "payload:      $(human "$PAYLOAD_BYTES") of parts; $(human "$TRUNK_BYTES") is hashed and copied at every start"
    say "layout:       $([ -L "$GGUF_DIR/$EMBED" ] && echo "$EMBED -> $(readlink "$GGUF_DIR/$EMBED")" || echo "$EMBED stays in the release directory")"
    say "engine:       $ENGINE_TREE at $(printf '%.8s' "$ENGINE_REF")"
    say "port:         $PORT, --mtp-mode $MTP_MODE --mtp-draft $MTP_DRAFT"
    say "effective:    $MAX_SEQS sequence, $PREFIX_REUSE prefix reuse, no -c (the engine's own default plan for this family)"
    say "lane:         $([ "$MAX_SEQS" = 1 ] && echo "one serial sequence (the fastest measured shape: 26.35 against 23.96 decode tok/s)" || echo "$MAX_SEQS banks; drafting runs on the serial lane only, so a bank shape is a concurrency trade")"
    say "banks:        $([ "$MAX_SEQS" = 1 ] && echo "none (a 1M bank beside serial media is 123.2 GiB and is refused)" || echo "disk K/V at ${KV_DISK_DIR:-<none>}; needed only for banks")"
    say "context:      the artifact's native 1,048,576 (no -c is passed)"
    say ""
    say "next steps:"
    local step=1
    [ -s "$STAGING/release.json" ] && { say "  $step. layout      # adopt the loose fetch at $STAGING"; step=$((step + 1)); }
    say "  $step. download    # $(human "$PAYLOAD_BYTES"), resumable"; step=$((step + 1))
    say "  $step. verify      # sizes, digests, the split, and the record"; step=$((step + 1))
    say "  $step. build       # detached worktree at $(printf '%.8s' "$ENGINE_REF") + make cuda-spark"; step=$((step + 1))
    say "  $step. check       # the engine's own plan"; step=$((step + 1))
    say "  $step. start       # $HOST_ADDR:$PORT"
}

cmd_bench() {
    say "The artifact author published the sweep this recipe quotes:"
    say "  https://github.com/Baekpica/ds4-dfm-rs/blob/d90f433546742ac367557450ba4f9d5d820385b4/docs/benchmarks/mimo2-vq-spark-64k-2026-10-31/README.md"
    say "It is 32 frontiers from 2K to 64K, 2,048 incremental prompt tokens and 128 forced"
    say "greedy tokens each, three fresh ds4-bench processes per variant, one binary at 2fa8c08e."
    say ""
    say "To reproduce on this box (the engine tree must be built at the pinned ref; the"
    say "server must be stopped, since one ds4 model is resident at a time):"
    say "  bash $0 stop"
    say "  $ENGINE_TREE/ds4-bench --cuda -m $GGUF_DIR --mtp-mode on --mtp-draft 2 --help"
    say "Nothing here has been timed yet. Whatever it prints belongs in the recipe only"
    say "with the command, the clock range and the date recorded beside it."
}

cmd_install() {
    cmd_preflight
    cmd_layout
    cmd_download
    cmd_verify
    [ "$NO_BUILD" = 1 ] || cmd_build
    cmd_check
    [ "$NO_START" = 1 ] || cmd_start
}

usage() { sed -n '2,/^set -Eeuo/{/^set -Eeuo/d;p}' "$0" | sed 's/^# \{0,1\}//'; }

# --------------------------------------------------------------------------
# options: consumed only while the first argument is an option. The action word
# (preflight, layout, ...) ends the loop.
# --------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --dir)             [ $# -ge 2 ] || die "--dir needs a directory"; MODEL_DIR="$2"; PIDF="$MODEL_DIR/mimo26.pid"; SERVER_LOG="$MODEL_DIR/mimo26.log"; WD_PIDF="$MODEL_DIR/mimo26-watchdog.pid"; WD_LOG="$MODEL_DIR/mimo26-watchdog.log"; RECORD="$MODEL_DIR/mimo26-verify.record"; EMBED_FAST="$MODEL_DIR/embed/$EMBED"; shift 2;;
        --gguf-dir)        [ $# -ge 2 ] || die "--gguf-dir needs a directory"; GGUF_DIR="$2"; shift 2;;
        --staging)         [ $# -ge 2 ] || die "--staging needs a directory"; STAGING="$2"; shift 2;;
        --port)            [ $# -ge 2 ] || die "--port needs a number"; PORT="$2"; shift 2;;
        --host)            [ $# -ge 2 ] || die "--host needs an address"; HOST_ADDR="$2"; shift 2;;
        --mtp-mode)        [ $# -ge 2 ] || die "--mtp-mode needs off | auto | on"; MTP_MODE="$2"; shift 2;;
        --mtp-draft)       [ $# -ge 2 ] || die "--mtp-draft needs a number"; MTP_DRAFT="$2"; shift 2;;
        --engine-src)      [ $# -ge 2 ] || die "--engine-src needs a directory"; ENGINE_SRC="$2"; shift 2;;
        --engine-tree)     [ $# -ge 2 ] || die "--engine-tree needs a directory"; ENGINE_TREE="$2"; shift 2;;
        --engine-ref)      [ $# -ge 2 ] || die "--engine-ref needs a commit"; ENGINE_REF="$2"; shift 2;;
        --server-bin)      [ $# -ge 2 ] || die "--server-bin needs a path"; SERVER_BIN="$2"; SERVER_BIN_EXPLICIT=1; shift 2;;
        --hf-bin)          [ $# -ge 2 ] || die "--hf-bin needs a path"; HF_BIN="$2"; shift 2;;
        --revision)        [ $# -ge 2 ] || die "--revision needs a sha"; REVISION="$2"; shift 2;;
        --hash-jobs)       [ $# -ge 2 ] || die "--hash-jobs needs a number"; HASH_JOBS="$2"; shift 2;;
        --no-watchdog)     WATCHDOG=0; shift;;
        --no-build)        NO_BUILD=1; shift;;
        --no-start)        NO_START=1; shift;;
        --allow-shared)    ALLOW_SHARED=1; shift;;
        -h|--help)         usage; exit 0;;
        -*)                die "unknown option: $1 (try --help)";;
        *)                 break;;
    esac
done

[ "$SERVER_BIN_EXPLICIT" = 1 ] || SERVER_BIN="$ENGINE_TREE/ds4-server"

action="${1:-}"
case "$action" in
    preflight) cmd_preflight;;
    layout)    cmd_layout;;
    download)  cmd_download;;
    verify)    cmd_verify;;
    build)     cmd_build;;
    check)     cmd_check;;
    start)     cmd_start;;
    stop)      cmd_stop;;
    status)    cmd_status;;
    logs)      cmd_logs;;
    test)      cmd_test;;
    plan)      cmd_plan;;
    bench)     cmd_bench;;
    install)   cmd_install;;
    ""|-h|--help) usage; exit 0;;
    *)         die "unknown action: $action (try --help)";;
esac
