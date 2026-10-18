#!/usr/bin/env bash
# GLM-5.3-Flash Uncensored VQ, R13+8 revision: fetch it into the release
# directory the R13 revision already occupies, prove which revision and which
# release the bytes belong to, and serve it on the DGX Spark.
#
#   artifact  Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF
#             pinned at cc70aef5a5ad370fbd59a69dc0a39522a1ed5c08 (main head,
#             2026-10-18T11:23:06Z), the R13+8 revision
#             56 GGUF parts, 89,960,211,968 bytes (83.78 GiB) of payload
#   engine    ds4-dfm-rs from 34c46d1 (PR #96, "Add the GLM VQ R13+8 residual
#             stage") on; that is the build that reads GVQ1 container v2.
#   parent    the base VQ release, Baekpica/GLM-5.3-Flash-VQ-GGUF, which moved
#             to its own R13+8 revision (35d3833d) the same day
#   previous  this artifact's own R13 revision (1764dd87), kept upstream at the
#             tag q4k-v3
#
# THIS SCRIPT'S SHAPE, AND WHERE IT CAME FROM
#   scripts/install-glm53-uncensored-vq.sh in this repository is the lineage:
#   same actions, same environment-override style, same "the release is a
#   DIRECTORY" rule, same watchdog and smoke test. Two things it could not carry
#   over are why this file exists: the R13+8 revision needs an engine floor that
#   the R13 script does not name, and a release directory that already holds R13
#   has to be refetched in place rather than treated as a complete release.
#
# TRAP 1: TWO RELEASES, ONE SET OF PART NAMES (unchanged from the R13 script)
#   The Uncensored release repeats the base release's 56 part names, tensor order
#   and payload sizes; only the source weights differ. The engine identifies both
#   as the same artifact, because its shape name is the literal string
#   "GLM-5.3-Flash-VQ" (crates/ds4-core/src/shape.rs:1122, served as the variant
#   name at crates/ds4-core/src/serving.rs:1485). Nothing in the runtime can tell
#   a directory of the wrong release from the right one, so two markers are read
#   from the release's own SHA256SUMS and checked on every action that touches
#   weights: chat_template.jinja (10,644 bytes here, 10,950 for the base) and the
#   MTP-02 part. A directory holding the base release is refused by name.
#
# TRAP 2, NEW HERE: TWO REVISIONS, THE SAME 56 NAMES, 14 PARTS REWRITTEN
#   R13+8 adds an 8-bit residual stage to routed layers 3-16, so 14 of the 56
#   parts were replaced in place upstream: layers 03-09 went from 1,477,184,128
#   to 2,383,481,600 bytes and layers 10-16 from 1,477,184,896 to 2,383,485,056.
#   The other 42 parts (28 R13 layers, 9 native, 4 MTP, vision) are byte-identical
#   across the two revisions - 42 of 56 digests match, verified against both Hub
#   listings. The two release markers above are IDENTICAL between the revisions,
#   so they cannot see this: a half-refetched directory passes them.
#   Three things guard it instead:
#     1. the revision is pinned by default and this script fetches that pin;
#     2. a revision fingerprint counts the 14 upgraded parts by size on disk and
#        refuses a directory that holds some of one revision and some of the
#        other (interrupted refetch), naming the parts that are stale;
#     3. verify re-runs the release's own SHA256SUMS, which is the authority.
#
# THE ENGINE FLOOR IS A FORMAT FLOOR, NOT A DATE
#   Container v1 holds R13 layers; v2 adds the GVR8 residual stage
#   (crates/ds4-core/src/glm_vq.rs:21, FORMAT_VERSIONS 1..=2 on the PR #96 build).
#   A build older than PR #96 has no GVR8 type and fails the 14 upgraded parts with
#   "unsupported tensor type" (that build's glm_vq.rs, c942523:370). --mtp-mode,
#   which the R13 script uses as its floor test, is advertised by both builds, so
#   the decisive check here is `check`: --check-config opens no weights and reads
#   the container versions.
#
# Usage (on the Spark, as the user who owns the model directory):
#   bash install-glm53-uncensored-vq-r13p8.sh preflight    GPU, CUDA 13, memory, disk, ports, both fingerprints, engine floor
#   bash install-glm53-uncensored-vq-r13p8.sh plan         what download and start would do
#   bash install-glm53-uncensored-vq-r13p8.sh download     the pinned revision, resumable file by file
#   bash install-glm53-uncensored-vq-r13p8.sh verify       sha256sum -c SHA256SUMS; reads all 90 GB
#   bash install-glm53-uncensored-vq-r13p8.sh check        the engine's plan quote, opens no weights
#   bash install-glm53-uncensored-vq-r13p8.sh start        serve in the foreground
#   bash install-glm53-uncensored-vq-r13p8.sh start --daemon   serve detached, wait for readiness
#   bash install-glm53-uncensored-vq-r13p8.sh status | stop | logs | test
#   bash install-glm53-uncensored-vq-r13p8.sh all          preflight, download, check, start
#
# Env overrides:
#   GLM53R13P8_REPO, GLM53R13P8_REVISION, GLM53R13P8_MODEL_ROOT,
#   GLM53R13P8_MODEL_DIR, GLM53R13P8_BIN, GLM53R13P8_ENGINE_SRC,
#   GLM53R13P8_HOST, GLM53R13P8_PORT, GLM53R13P8_MODEL_ID,
#   GLM53R13P8_MTP_MODE, GLM53R13P8_DRAFT, GLM53R13P8_CTX,
#   GLM53R13P8_MAX_SEQS, GLM53R13P8_STATE, GLM53R13P8_PID, GLM53R13P8_LOG,
#   GLM53R13P8_ALLOW_SHARED, GLM53R13P8_WATCHDOG, GLM53R13P8_MAX_WORKERS,
#   GLM53R13P8_FORCE (re-fetch every file even if the CLI would skip it),
#   HF_TOKEN, ENDPOINT, NO_XET.
#
# Notes
#   - The repository is public: no token is needed and none is ever written here.
#     HF_TOKEN only raises the CDN rate limit; the CLI reads it from the
#     environment on its own.
#   - This script and the R13 one default to the SAME release directory,
#      /mnt/models/GLM-5.3-Flash-Uncensored-VQ-GGUF, because the operator chose an
#     in-place refetch: one directory, one endpoint (port 8009, model id
#     glm-5.3-flash-uncensored-vq). The R13 installer cannot see the difference
#     (markers identical, part count identical) and must not be used on it.
#   - This release fits ONE 1M bank, not two: the artifact card's own quote puts
#     the plan at 112.0 GiB of the 118.7 GiB the engine sees on GB10. The default
#     here is therefore --max-seqs 1 at the native context.
#   - The engine verifies every file, tensor and payload SHA256 while it loads the
#     release, so verify is a second, independent opinion rather than a
#     precondition.
#   - The workflow this script does NOT run: building the host. Do that first with
#     cd ~/ds4-dfm-rs && make cuda-spark. The C oracle server arms continuous
#     speculation by itself when an MTP source is loaded (ds4_server.c:23970) and
#     the Rust host does not, so this script exports DS4_CONT_MTP_MODE=2 and
#     DS4_CONT_MTP_DEPTH alongside --mtp-mode on.
#   - One ds4 model at a time: the engine takes a single global lock. The port
#     probe and the owner probe below catch the cases the lock does not, such as
#     an owner that is still hashing its release.

set -Eeuo pipefail

# --------------------------------------------------------------------------
# Configuration (every value is overridable from the environment)
# --------------------------------------------------------------------------
REPO=${GLM53R13P8_REPO:-Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF}
PIN=cc70aef5a5ad370fbd59a69dc0a39522a1ed5c08
# Pinned by default. The revision is what says which bytes are on disk, and the
# release markers cannot: they are identical between R13 and R13+8. Set
# GLM53R13P8_REVISION= to track main, or to the R13 commit
# 1764dd87b8ee92dd288d7df15bf62d120ae010d8 to go back; the revision fingerprint
# then reports what it finds instead of refusing.
REVISION=${GLM53R13P8_REVISION:-$PIN}
R13_REVISION=1764dd87b8ee92dd288d7df15bf62d120ae010d8

RELEASE_NAME="GLM-5.3-Flash-Uncensored-VQ-GGUF"
MODEL_ROOT=${GLM53R13P8_MODEL_ROOT:-/mnt/models}
# The SAME directory the R13 revision occupies, deliberately: the refetch replaces
# the 14 upgraded parts in place. Point this elsewhere to keep both revisions.
MODEL_DIR=${GLM53R13P8_MODEL_DIR:-$MODEL_ROOT/$RELEASE_NAME}
SERVER_BIN=${GLM53R13P8_BIN:-$HOME/ds4-dfm-rs/ds4-server}
ENGINE_SRC=${GLM53R13P8_ENGINE_SRC:-$HOME/ds4-dfm-rs}
HOST=${GLM53R13P8_HOST:-0.0.0.0}
# 8000 is DeepSeek V4.1 Flash, 8003 Qwen3.8, 8005 Bonsai, 8006 YoungAi, 8007 the
# base GLM-5.3 VQ, 8008 the YoungAI 1017 launcher and 8010 the DeepSeek 1017
# Uncensored VQ, so this one keeps the port the R13 revision of this artifact
# already answers on.
PORT=${GLM53R13P8_PORT:-8009}
MODEL_ID=${GLM53R13P8_MODEL_ID:-glm-5.3-flash-uncensored-vq}
MTP_MODE=${GLM53R13P8_MTP_MODE:-on}     # auto resolves to off for this family: see the header
DRAFT=${GLM53R13P8_DRAFT:-2}            # the draft count the base release's card measured
CTX=${GLM53R13P8_CTX:-}                 # empty keeps the native 1,048,576
# One bank by default at the native context: the card's --check-config quote is
# 112.0 GiB (weights 83.8, bank 12.8, scratch 10.7, MTP 0.7, floor 4) against the
# 118.7 GiB the engine sees on GB10, so the automatic width is not relied on.
MAX_SEQS=${GLM53R13P8_MAX_SEQS:-1}
WATCHDOG=${GLM53R13P8_WATCHDOG:-1}      # 1 stops the server if MemAvailable stays low

STATE=${GLM53R13P8_STATE:-$HOME}
PID_FILE=${GLM53R13P8_PID:-$STATE/glm53unc.pid}
LOG_FILE=${GLM53R13P8_LOG:-$STATE/glm53unc.log}
WD_PID_FILE=${GLM53R13P8_WD_PID:-$STATE/glm53unc-watchdog.pid}
WD_LOG=${GLM53R13P8_WD_LOG:-$STATE/glm53unc-watchdog.log}

GGUF_PARTS=56                       # parts listed in files.json at this revision
PAYLOAD_BYTES=89960211968           # the 56 GGUF parts, 83.78 GiB
R13_PAYLOAD_BYTES=77272028544       # the same 56 names at the previous revision
DELTA_BYTES=$((PAYLOAD_BYTES - R13_PAYLOAD_BYTES))   # what an in-place refetch adds
REQUIRED_GIB=95                     # a fresh fetch: payload plus slack for partial files
WORK_ROOM_BYTES=8000000000          # 8 GB of headroom while a fetch is in flight
MAX_WORKERS=${GLM53R13P8_MAX_WORKERS:-8}
READY_TIMEOUT_S=${GLM53R13P8_READY_TIMEOUT:-900}
START_FLOOR_MB=${GLM53R13P8_START_FLOOR_MB:-100000}   # before loading
RUN_FLOOR_MB=${GLM53R13P8_RUN_FLOOR_MB:-4000}         # after loading; the card's plan is 112.0 of 118.7 GiB
KILL_MB=${GLM53R13P8_KILL_MB:-2500}                   # two consecutive 5 s samples stop the server
ENGINE_MIN_COMMIT=34c46d1           # PR #96: the GVQ1 container v2 reader
ENGINE_MIN_DESC="PR #96, the GLM VQ R13+8 residual stage"

# The two release markers, from the release's own SHA256SUMS. Each pair is
# (size, sha256) for the Uncensored release and for the base release, so the
# script can tell them apart and can name the wrong one. Both pairs are IDENTICAL
# between the R13 and R13+8 revisions.
TEMPLATE=chat_template.jinja
TEMPLATE_BYTES=10644
TEMPLATE_SHA=34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891
BASE_TEMPLATE_SHA=0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5
MARKER=GLM-5.3-Flash-MTP-02-of-04.gguf
MARKER_BYTES=1997061632
MARKER_SHA=68a41631cabc0b107e04ab2418e186b76e5debea60d95e1811953cf4f1c28c26
BASE_MARKER_BYTES=1997061504
BASE_MARKER_SHA=194058b2ae15048b010149539057cdedbaf2ef11da7f7e4adecf96f45ced51af

# The revision fingerprint: the 14 parts R13+8 rewrote, by size. Layers 03-09 and
# 10-16 differ from each other by their last 128 bytes, so both sizes are listed.
R13P8_PARTS=(GLM-5.3-Flash-VQ-layer-03.gguf GLM-5.3-Flash-VQ-layer-04.gguf
             GLM-5.3-Flash-VQ-layer-05.gguf GLM-5.3-Flash-VQ-layer-06.gguf
             GLM-5.3-Flash-VQ-layer-07.gguf GLM-5.3-Flash-VQ-layer-08.gguf
             GLM-5.3-Flash-VQ-layer-09.gguf GLM-5.3-Flash-VQ-layer-10.gguf
             GLM-5.3-Flash-VQ-layer-11.gguf GLM-5.3-Flash-VQ-layer-12.gguf
             GLM-5.3-Flash-VQ-layer-13.gguf GLM-5.3-Flash-VQ-layer-14.gguf
             GLM-5.3-Flash-VQ-layer-15.gguf GLM-5.3-Flash-VQ-layer-16.gguf)
R13P8_SIZE_SMALL=2383481600         # layers 03-09 at R13+8 (1,477,184,128 at R13)
R13P8_SIZE_LARGE=2383485056         # layers 10-16 at R13+8 (1,477,184,896 at R13)
R13_SIZE_SMALL=1477184128
R13_SIZE_LARGE=1477184896

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
say()  { printf 'glm53r13p8: %s\n' "$*"; }
warn() { printf 'glm53r13p8: warning: %s\n' "$*" >&2; }
die()  { printf 'glm53r13p8: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
total_mb() { awk '/MemTotal/{print int($2/1024)}' /proc/meminfo; }
fsize()    { stat -L -c %s "$1" 2>/dev/null || echo 0; }
df_avail() { df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' \n' || true; }
df_target(){ df --output=target "$1" 2>/dev/null | tail -1 || true; }
human()    { awk -v b="$1" 'BEGIN{printf "%.1f GB", b/1000000000}'; }
sha_file() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
alive()    { [ -s "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }

parts_present() { find "$MODEL_DIR" -maxdepth 1 -name '*.gguf' 2>/dev/null | wc -l; }
parts_partial() { find "$MODEL_DIR" -maxdepth 1 -name '*.incomplete' 2>/dev/null | wc -l; }

# A complete release has every part, its tables, and no partial file left behind.
# Sizes and hashes themselves are the loader's business: it verifies all of them.
release_ok() {
    [ -f "$MODEL_DIR/files.json" ] || return 1
    [ -f "$MODEL_DIR/tensor-manifest.json" ] || return 1
    [ -f "$MODEL_DIR/tokenizer_config.json" ] || return 1
    [ -f "$MODEL_DIR/$TEMPLATE" ] || return 1
    [ "$(parts_present)" -eq "$GGUF_PARTS" ] || return 1
    [ "$(parts_partial)" -eq 0 ]
}

# Which release is in MODEL_DIR: uncensored, base, or unknown. The template digest
# and the MTP-02 size are the two places the releases differ, and both are
# published in SHA256SUMS. This says nothing about which REVISION it is.
release_identity() {
    local t m
    [ -f "$MODEL_DIR/$TEMPLATE" ] || { printf 'absent'; return; }
    t=$(sha_file "$MODEL_DIR/$TEMPLATE")
    m=$(fsize "$MODEL_DIR/$MARKER")
    case "$t:$m" in
        "$TEMPLATE_SHA:$MARKER_BYTES") printf 'uncensored'; return;;
        "$BASE_TEMPLATE_SHA:$BASE_MARKER_BYTES") printf 'base'; return;;
        "$BASE_TEMPLATE_SHA:"*|*":$BASE_MARKER_BYTES") printf 'base'; return;;
    esac
    printf 'unknown'
}

# Which revision is on disk: r13p8, r13, mixed, or unknown. Only the 14 parts
# R13+8 rewrote can answer this; the release markers are identical between the
# revisions. Counted in two halves (03-09 and 10-16 have different sizes at each
# revision), so an interrupted refetch shows up as a mixed directory instead of
# being rounded to one side.
revision_identity() {
    [ -f "$MODEL_DIR/${R13P8_PARTS[0]}" ] || { printf 'absent'; return; }
    local small_new=0 small_old=0 large_new=0 large_old=0 p s
    for p in "${R13P8_PARTS[@]}"; do
        [ -f "$MODEL_DIR/$p" ] || continue
        s=$(fsize "$MODEL_DIR/$p")
        case "$p" in
            *layer-0[3-9].gguf)
                [ "$s" = "$R13P8_SIZE_SMALL" ] && small_new=$((small_new + 1))
                [ "$s" = "$R13_SIZE_SMALL" ]   && small_old=$((small_old + 1));;
            *layer-1[0-6].gguf)
                [ "$s" = "$R13P8_SIZE_LARGE" ] && large_new=$((large_new + 1))
                [ "$s" = "$R13_SIZE_LARGE" ]   && large_old=$((large_old + 1));;
        esac
    done
    if [ "$small_new" = 7 ] && [ "$large_new" = 7 ]; then printf 'r13p8'; return; fi
    if [ "$small_old" = 7 ] && [ "$large_old" = 7 ]; then printf 'r13'; return; fi
    if [ $((small_new + small_old + large_new + large_old)) -gt 0 ]; then printf 'mixed'; return; fi
    printf 'unknown'
}

# Names the parts that are not the pinned revision's, so a mixed directory gets an
# actionable message rather than a diagnosis.
stale_parts() {
    local p s out=()
    for p in "${R13P8_PARTS[@]}"; do
        [ -f "$MODEL_DIR/$p" ] || continue
        s=$(fsize "$MODEL_DIR/$p")
        case "$p" in
            *layer-0[3-9].gguf) [ "$s" = "$R13P8_SIZE_SMALL" ] || out+=("$p");;
            *layer-1[0-6].gguf) [ "$s" = "$R13P8_SIZE_LARGE" ] || out+=("$p");;
        esac
    done
    printf '%s' "${out[*]:-}"
}

assert_uncensored() {
    need sha256sum
    [ -f "$MODEL_DIR/$TEMPLATE" ] \
        || die "$MODEL_DIR/$TEMPLATE is missing: $MODEL_DIR does not look like a GVQ1 release"
    local id; id=$(release_identity)
    case "$id" in
        uncensored) return 0;;
        base) die "$MODEL_DIR holds the BASE release (Baekpica/GLM-5.3-Flash-VQ-GGUF): its chat
       template is $BASE_TEMPLATE_SHA and its MTP-02 part is $BASE_MARKER_BYTES bytes, against
       $TEMPLATE_SHA and $MARKER_BYTES for the Uncensored release. The engine cannot tell the
       two apart (its shape name is the literal string GLM-5.3-Flash-VQ), so nothing else
       would notice. Point GLM53R13P8_MODEL_DIR at a directory of its own.";;
    esac
    if [ "$REVISION" = "$PIN" ]; then
        die "$MODEL_DIR holds neither release: $TEMPLATE is $(sha_file "$MODEL_DIR/$TEMPLATE")
       and $MARKER is $(fsize "$MODEL_DIR/$MARKER") bytes, against $TEMPLATE_SHA and
       $MARKER_BYTES published for the Uncensored release at $PIN. Re-run download, or delete
       the directory and fetch it again."
    fi
    warn "$MODEL_DIR does not match the pinned revision's digests; the check is relaxed because
       GLM53R13P8_REVISION=$REVISION is not the pin $PIN (the base-release refusal still applies)"
}

assert_r13p8() {
    local rev; rev=$(revision_identity)
    case "$rev" in
        r13p8) return 0;;
        r13) die "$MODEL_DIR still holds the R13 revision ($R13_REVISION): its 14 upgraded parts
       are at R13 sizes. The two release markers cannot see this - they are identical between
       the revisions - so the engine would load 14 stale parts and serve them without a word.
       Run: bash $0 download  (it refetches the 14 parts in place)";;
        mixed) die "$MODEL_DIR holds a MIXED directory: some of the 14 upgraded parts are still at
       R13 sizes after a refetch that did not finish. Stale: $(stale_parts)
       Re-run: bash $0 download   (resumable; then bash $0 verify)";;
        absent) die "$MODEL_DIR has no upgraded layer part on disk; run: bash $0 download";;
        *) die "$MODEL_DIR holds layer parts whose sizes match neither revision. These 14 parts are
       2,383,481,600 (layers 03-09) and 2,383,485,056 (layers 10-16) at R13+8, and 1,477,184,128
       or 1,477,184,896 at R13. Run: bash $0 verify";;
    esac
}

# hf download appeared in huggingface_hub 0.34; older installs only ship the long
# form. Both accept --local-dir and --max-workers and resume by default.
ensure_hf() {
    if [ -n "${HF_BIN:-}" ]; then HF="$HF_BIN"; return; fi
    if command -v hf >/dev/null 2>&1 && hf download --help >/dev/null 2>&1; then HF=hf; return; fi
    if command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli; return; fi
    # The reference box keeps one at ~/hfenv, so use it before building another.
    if [ -x "$HOME/hfenv/bin/hf" ]; then HF="$HOME/hfenv/bin/hf"; return; fi
    HF="$MODEL_ROOT/.hfenv/bin/hf"
    [ -x "$HF" ] && return
    say "installing the Hugging Face CLI into $MODEL_ROOT/.hfenv (the system Python is left alone)"
    mkdir -p "$MODEL_ROOT"
    python3 -m venv "$MODEL_ROOT/.hfenv" || die "python3 venv failed; install it with: sudo apt install python3-venv"
    "$MODEL_ROOT/.hfenv/bin/pip" install -q -U "huggingface_hub[hf_xet]" \
        || die "pip install huggingface_hub failed (behind a firewall? try ENDPOINT= and NO_XET=1)"
    HF="$MODEL_ROOT/.hfenv/bin/hf"
}

# --force-download is not in every release of the CLI, so it is probed rather
# than assumed. It only matters when a fetch skips a file whose size changed,
# which is what the revision fingerprint after the fetch checks for anyway.
hf_supports_force() { "$HF" download --help 2>&1 | grep -q -- '--force-download'; }

hf_download() {
    export HF_HUB_MAX_WORKERS="$MAX_WORKERS"
    export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
    [ -n "${ENDPOINT:-}" ] && export HF_ENDPOINT="$ENDPOINT"
    [ "${NO_XET:-0}" = 1 ] && export HF_HUB_DISABLE_XET=1
    local -a rev=() extra=()
    [ -n "$REVISION" ] && rev=(--revision "$REVISION")
    if [ "${GLM53R13P8_FORCE:-0}" = 1 ]; then
        if hf_supports_force; then extra=(--force-download)
        else warn "GLM53R13P8_FORCE=1 but $HF does not advertise --force-download; fetching normally"; fi
    fi
    "$HF" download "$REPO" "${rev[@]}" "${extra[@]}" --local-dir "$MODEL_DIR" --max-workers "$MAX_WORKERS"
}

# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------
preflight() {
    local fatal=0

    if [ "$(uname -s)/$(uname -m)" = "Linux/aarch64" ]; then
        say "arch ok: Linux/aarch64"
    else
        warn "this is $(uname -s)/$(uname -m); the ds4 CUDA host and its kernels target the DGX Spark (sm_121)"; fatal=1
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu; gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        case "$gpu" in
            *GB10*) say "gpu ok: $gpu";;
            *) warn "GPU is '$gpu'; the CUDA kernels are built for GB10 (sm_121)"; fatal=1;;
        esac
    else
        warn "nvidia-smi not found: no NVIDIA driver on this machine"; fatal=1
    fi

    local ldc miss=() lib
    ldc=$(command -v ldconfig || echo /sbin/ldconfig)
    if [ -x "$ldc" ]; then
        for lib in libcudart.so.13 libcublas.so.13 libcublasLt.so.13; do
            "$ldc" -p 2>/dev/null | grep -q "$lib" || miss+=("$lib")
        done
        if [ "${#miss[@]}" -eq 0 ]; then
            say "cuda ok: CUDA 13 runtime registered with ldconfig"
        else
            warn "not in the ldconfig cache: ${miss[*]} (the CUDA 13 runtime is required; DGX OS ships it)"; fatal=1
        fi
    else
        warn "ldconfig not found; cannot check the CUDA 13 runtime"
    fi

    local t; t=$(total_mb)
    if [ "$t" -ge 115000 ]; then
        say "memory ok: ${t} MB total"
    else
        warn "only ${t} MB of memory; the card's plan for this revision quotes 112.0 GiB of the 118.7 GiB the engine sees on GB10"; fatal=1
    fi
    say "memory available now: $(avail_mb) MB (start refuses below ${START_FLOOR_MB} MB)"

    mkdir -p "$MODEL_ROOT" 2>/dev/null || { warn "cannot create $MODEL_ROOT"; fatal=1; }
    local free rev need=0
    rev=$(revision_identity)
    if ! release_ok; then
        # What this fetch still has to add: the whole payload on an empty or absent
        # directory, or only the difference on one that already holds R13.
        case "$rev" in
            r13|mixed) need="$DELTA_BYTES";;
            *)         need="$PAYLOAD_BYTES";;
        esac
    fi
    free=$(df_avail "$MODEL_ROOT")
    if [ -n "$free" ] && [ "$free" -gt $((need + WORK_ROOM_BYTES)) ]; then
        say "disk ok: $(human $need) still to fetch, $(human $free) free on $(df_target "$MODEL_ROOT")"
    else
        warn "not enough room on $(df_target "$MODEL_ROOT"): want $(human $need) plus 8 GB working room, have $(human ${free:-0})"
        warn "set GLM53R13P8_MODEL_ROOT= to a volume with room (a fresh fetch needs about ${REQUIRED_GIB} GB in total)"; fatal=1
    fi

    # Both fingerprints, when a release is already on disk.
    if [ -f "$MODEL_DIR/$TEMPLATE" ]; then
        local id; id=$(release_identity)
        case "$id" in
            uncensored) say "release ok: the Uncensored release is in $MODEL_DIR ($(parts_present)/$GGUF_PARTS parts)";;
            base) warn "$MODEL_DIR holds the BASE release, not this one; download would land beside it wrongly - point GLM53R13P8_MODEL_DIR elsewhere"; fatal=1;;
            *) warn "$MODEL_DIR holds a release whose markers match neither release; re-run download"; fatal=1;;
        esac
        case "$rev" in
            r13p8) say "revision ok: the 14 upgraded parts are at R13+8 sizes";;
            r13)   say "revision: R13 on disk ($(human $R13_PAYLOAD_BYTES) of payload); download replaces the 14 upgraded parts in place, +$(human $DELTA_BYTES)";;
            mixed) warn "revision: MIXED - stale parts: $(stale_parts)"; say "download refetches exactly those";;
            *)     warn "revision: cannot classify the 14 part sizes; run verify";;
        esac
    else
        say "no release yet in $MODEL_DIR ($(human $PAYLOAD_BYTES) to fetch)"
    fi

    # The engine's floor for reading this revision.
    if [ -x "$SERVER_BIN" ]; then
        say "runtime ok: $SERVER_BIN"
        local help; help=$("$SERVER_BIN" --help 2>&1 || true)
        case "$help" in
            *--mtp-mode*) ;;
            *) warn "$SERVER_BIN does not advertise --mtp-mode: it predates GLM-5.3 VQ support (PR #86, c5384b5e); rebuild with: cd $ENGINE_SRC && make cuda-spark"; fatal=1;;
        esac
        if [ -d "$ENGINE_SRC/.git" ]; then
            if git -C "$ENGINE_SRC" cat-file -e "${ENGINE_MIN_COMMIT}^{commit}" 2>/dev/null; then
                say "engine checkout ok: $ENGINE_SRC knows $ENGINE_MIN_COMMIT ($ENGINE_MIN_DESC)"
            else
                warn "$ENGINE_SRC does not contain $ENGINE_MIN_COMMIT ($ENGINE_MIN_DESC): a build older than it has no GVR8 type and rejects the 14 upgraded parts with 'unsupported tensor type'. Fetch the engine and rebuild."
            fi
        fi
        say "engine floor: container v2 is a FORMAT floor. --mtp-mode is advertised by both builds, so the decisive test is 'check' (--check-config opens no weights and reads the container versions)."
    else
        warn "no executable at $SERVER_BIN; build the host with: cd $ENGINE_SRC && make cuda-spark"; fatal=1
    fi

    # One resident model. The engine takes a single global lock by design, and this
    # box already answers on other ports; the probe catches the case the lock does
    # not, an owner that is still loading.
    if command -v curl >/dev/null 2>&1; then
        local busy=() p
        for p in 8000 8001 8002 8003 8004 8005 8006 8007 8008 8010; do
            curl -fsS --max-time 2 "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && busy+=("$p")
        done
        [ "${#busy[@]}" -gt 0 ] && warn "these ports already serve a model: ${busy[*]} - the engine's lock will refuse this start"
        if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            warn "port $PORT already answers /v1/models; pick another with GLM53R13P8_PORT="; fatal=1
        fi
    fi
    local owners
    owners=$(other_owners)
    [ -n "$owners" ] && warn "a ds4 owner is already running: $owners"

    [ "$fatal" = 0 ] && say "preflight: this machine can run the artifact" || return 1
}

other_owners() { pgrep -af 'ds4(-server|-bench)?($| )' | grep -v "$$" || true; }

# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
cmd_plan() {
    local rev; rev=$(revision_identity)
    say "install root:  $MODEL_ROOT"
    say "release dir:   $MODEL_DIR   (the R13 revision's directory: the refetch is in place)"
    say "runtime:       $SERVER_BIN"
    say "serve:         $HOST:$PORT, model id $MODEL_ID, watchdog below ${KILL_MB} MB"
    say ""
    say "artifacts"
    say "  revision     $REPO @ ${REVISION:-main}"
    say "  parts        $GGUF_PARTS GGUF files, $(human $PAYLOAD_BYTES)   $([ "$(parts_present)" -eq "$GGUF_PARTS" ] && echo present || echo "$(parts_present) on disk")"
    say "  14 of them   rewritten by R13+8: layers 03-16 carry an 8-bit residual stage (GVQ1 container v2)"
    say "  the other 42 unchanged from R13: 28 R13 layers, 9 native parts, 4 MTP parts, 1 vision tower"
    say "  engine floor $ENGINE_MIN_COMMIT ($ENGINE_MIN_DESC): container v2 is rejected by older builds"
    say "  tables       files.json, tensor-manifest.json, tokenizer*, chat_template.jinja, SHA256SUMS"
    say "  not loaded   GLM-5.3-Flash-vision-01-of-01.gguf ships in the release; this script does not pass --vision"
    say ""
    say "release identity (this release vs its base sibling; identical in both revisions)"
    say "  $TEMPLATE  $TEMPLATE_BYTES bytes, $TEMPLATE_SHA"
    say "  $MARKER  $MARKER_BYTES bytes, $MARKER_SHA"
    say "revision identity (R13 vs R13+8; the only on-disk difference)"
    say "  layers 03-09  $R13P8_SIZE_SMALL at R13+8, $R13_SIZE_SMALL at R13"
    say "  layers 10-16  $R13P8_SIZE_LARGE at R13+8, $R13_SIZE_LARGE at R13"
    say "  on disk now:  $rev"
    say ""
    say "the command start would run:"
    say "  $SERVER_BIN --cuda -m $MODEL_DIR \\"
    say "      --mtp-mode $MTP_MODE --mtp-draft $DRAFT --max-seqs $MAX_SEQS --host $HOST --port $PORT \\"
    say "      --model-id $MODEL_ID --no-update-check${CTX:+ -c $CTX}"
    [ "$MTP_MODE" != "off" ] && say "  with DS4_CONT_MTP_MODE=2 DS4_CONT_MTP_DEPTH=$DRAFT in the environment (see the header)"
    say "  (no -c unless GLM53R13P8_CTX is set: the family defaults to the native 1048576)"
    say "  (--max-seqs $MAX_SEQS is explicit: this revision fits one 1M bank, 112.0 GiB of 118.7 GiB on GB10)"
    say ""
    local need=0
    release_ok || case "$rev" in r13|mixed) need=$DELTA_BYTES;; *) need=$PAYLOAD_BYTES;; esac
    say "still to fetch: $(human $need); free on target $(df_target "$MODEL_ROOT"): $(human "$(df_avail "$MODEL_ROOT")")"
    say "ports 8000, 8003, 8004, 8005, 8006, 8007, 8008 and 8010 answer for the other launchers in this home"
}

# --------------------------------------------------------------------------
# download, verify, check
# --------------------------------------------------------------------------
cmd_download() {
    mkdir -p "$MODEL_DIR"
    local before; before=$(revision_identity)
    if release_ok && [ "$before" = "r13p8" ]; then
        assert_uncensored
        say "already downloaded: $(parts_present) parts in $MODEL_DIR, all 14 upgraded parts at R13+8 sizes"
        return 0
    fi
    ensure_hf
    local free; free=$(df_avail "$MODEL_ROOT")
    [ -z "$free" ] || [ "$free" -gt "$WORK_ROOM_BYTES" ] \
        || die "$MODEL_ROOT has $(human ${free:-0}) free, needs at least $(human $WORK_ROOM_BYTES) of working room"
    case "$before" in
        r13)   say "in-place refetch: this directory holds R13. The 14 upgraded parts are replaced and"
               say "  $(human $DELTA_BYTES) is added; the other 42 parts are byte-identical and are left alone.";;
        mixed) say "in-place refetch of a MIXED directory; stale parts: $(stale_parts)";;
        r13p8) say "the revision is already R13+8; the fetch only fills what is missing";;
        *)     say "fetching the whole release";;
    esac
    say "fetching $REPO${REVISION:+ @ $REVISION} into $MODEL_DIR"
    say "an interrupted run resumes from where it stopped; just run it again"
    hf_download || die "download failed; run the same command again to resume"

    local after; after=$(revision_identity)
    case "$after" in
        r13p8) say "revision ok: all 14 upgraded parts are now at R13+8 sizes";;
        r13)   warn "the fetch finished but the directory still holds R13 sizes."
               warn "The CLI can skip a file it believes it already has; re-run with GLM53R13P8_FORCE=1 to re-fetch every file."
               die "the release is still the R13 revision";;
        mixed) warn "the fetch left a mixed directory. Stale: $(stale_parts)"
               die "re-run the same command to resume; then bash $0 verify";;
        *)     die "the fetch finished but the 14 upgraded parts cannot be classified; run bash $0 verify";;
    esac
    release_ok || die "the fetch finished but the release is incomplete: $(parts_present)/$GGUF_PARTS parts, $(parts_partial) partial"
    assert_uncensored
    say "done: $GGUF_PARTS parts, $(human "$(du -sb "$MODEL_DIR" | cut -f1)") on disk, markers say $REPO's Uncensored release at R13+8"
    say "next: bash $0 verify (hashes all 90 GB), then bash $0 check, then bash $0 start"
}

cmd_verify() {
    release_ok || die "the release is incomplete in $MODEL_DIR; run download first"
    assert_uncensored
    assert_r13p8
    need sha256sum
    say "hashing $(human "$(du -sb "$MODEL_DIR" | cut -f1)") ; this reads every byte"
    ( cd "$MODEL_DIR" && sha256sum -c SHA256SUMS ) || die "a payload hash does not match SHA256SUMS"
    say "verify: every payload hash matches SHA256SUMS, so the 14 upgraded parts are the pinned revision's"
}

# --check-config prints the plan as JSON and opens no weights, so this is the
# cheapest way to see the family, the context and the memory quote before a load -
# and the only action that proves the engine reads container v2.
cmd_check() {
    release_ok || die "the release is incomplete in $MODEL_DIR; run download first"
    assert_uncensored
    assert_r13p8
    [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN (build it: cd $ENGINE_SRC && make cuda-spark)"
    say "release: $(( $(parts_present) ))/$GGUF_PARTS parts, revision $(revision_identity), markers say uncensored"
    local args=(--check-config -m "$MODEL_DIR")
    [ -n "$CTX" ] && args+=(-c "$CTX")
    [ -n "$MAX_SEQS" ] && args+=(--max-seqs "$MAX_SEQS")
    [ "$MTP_MODE" = "off" ] || args+=(--mtp-mode "$MTP_MODE" --mtp-draft "$DRAFT")
    if ! "$SERVER_BIN" "${args[@]}"; then
        die "the plan quote failed. A build older than $ENGINE_MIN_COMMIT ($ENGINE_MIN_DESC) has no
       GVR8 type and rejects the 14 upgraded parts as 'unsupported tensor type'; rebuild with:
       cd $ENGINE_SRC && make cuda-spark"
    fi
}

# --------------------------------------------------------------------------
# start, stop, status, logs, test
# --------------------------------------------------------------------------
# From scripts/start-glm53-flash-vq.sh: the plan is 112.0 GiB of the 118.7 GiB the
# engine sees on GB10, so the margin is thin enough to warrant this. Two
# consecutive 5 s samples of MemAvailable below KILL_MB stop the server, which
# beats a box that has started swapping and cannot be typed into.
watchdog() {   # $1 pid, $2 floor MB, $3 log file
    local pid="$1" bad=0 a
    while kill -0 "$pid" 2>/dev/null; do
        a=$(avail_mb)
        if [ "$a" -lt "$2" ]; then bad=$((bad + 1)); else bad=0; fi
        if [ "$bad" -ge 2 ]; then
            printf '%s MemAvailable %s MB below %s MB twice; stopping the server\n' \
                "$(date)" "$a" "$2" >>"$3"
            kill "$pid" 2>/dev/null || true
            return 0
        fi
        sleep 5
    done
}

cmd_start() {
    release_ok || die "the release is incomplete in $MODEL_DIR; run $0 download first"
    assert_uncensored
    assert_r13p8
    [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN (build it: cd $ENGINE_SRC && make cuda-spark)"
    need curl
    if alive; then
        die "already running (pid $(cat "$PID_FILE")); stop it first with $0 stop"
    fi

    local owners
    owners=$(other_owners)
    if [ -n "$owners" ] && [ "${GLM53R13P8_ALLOW_SHARED:-0}" != "1" ]; then
        printf 'glm53r13p8: another model owner is running:\n%s\n' "$owners" >&2
        die "stop it first, or set GLM53R13P8_ALLOW_SHARED=1 to overrule (one ds4 model fits on this box)"
    fi

    local a; a=$(avail_mb)
    [ "$a" -ge "$START_FLOOR_MB" ] \
        || die "only $a MB of memory available, need $START_FLOOR_MB: stop another model first"
    [ "$a" -ge $(( (PAYLOAD_BYTES / 1000000) )) ] \
        || warn "the plan for this revision is $(human $PAYLOAD_BYTES) of weights alone; $(human $((a * 1000000))) is available now"

    local -a args env_spec=()
    args=(--cuda -m "$MODEL_DIR" --host "$HOST" --port "$PORT"
          --model-id "$MODEL_ID" --no-update-check)
    if [ "$MTP_MODE" = "off" ]; then
        args+=(--mtp-mode off)
    else
        args+=(--mtp-mode "$MTP_MODE" --mtp-draft "$DRAFT")
        env_spec=(DS4_CONT_MTP_MODE=2 DS4_CONT_MTP_DEPTH="$DRAFT")
    fi
    [ -n "$CTX" ] && args+=(-c "$CTX")
    [ -n "$MAX_SEQS" ] && args+=(--max-seqs "$MAX_SEQS")

    mkdir -p "$(dirname "$LOG_FILE")"

    if [ "${1:-}" != "--daemon" ]; then
        say "serving $MODEL_ID in the foreground on $HOST:$PORT; the log is also written to $LOG_FILE"
        say "the host hashes 90 GB and maps the weights before it listens: about a minute"
        # Process substitution keeps $! as the server's own pid, which is what the
        # watchdog needs; a pipe would hand it tee's pid instead.
        env "${env_spec[@]}" "$SERVER_BIN" "${args[@]}" > >(tee -a "$LOG_FILE") 2>&1 </dev/null &
        local pid=$!
        printf '%s\n' "$pid" >"$PID_FILE"
        if [ "$WATCHDOG" = 1 ]; then
            export -f watchdog avail_mb
            nohup bash -c "watchdog $pid $KILL_MB '$WD_LOG'" >/dev/null 2>&1 </dev/null &
            printf '%s\n' "$!" >"$WD_PID_FILE"
            say "watchdog pid $! (stops the server if MemAvailable stays below ${KILL_MB} MB)"
        fi
        local rc=0
        wait "$pid" || rc=$?
        rm -f "$PID_FILE" "$WD_PID_FILE"
        return "$rc"
    fi

    setsid nohup env "${env_spec[@]}" "$SERVER_BIN" "${args[@]}" >>"$LOG_FILE" 2>&1 </dev/null &
    local pid=$!
    printf '%s\n' "$pid" >"$PID_FILE"
    say "pid $pid, log $LOG_FILE"
    say "loading 90 GB and verifying every hash; the port binds only after that"

    if [ "$WATCHDOG" = 1 ]; then
        export -f watchdog avail_mb
        nohup bash -c "watchdog $pid $KILL_MB '$WD_LOG'" >/dev/null 2>&1 </dev/null &
        printf '%s\n' "$!" >"$WD_PID_FILE"
        say "watchdog pid $! (stops the server if MemAvailable stays below ${KILL_MB} MB)"
    fi

    local waited=0
    while [ "$waited" -lt "$READY_TIMEOUT_S" ]; do
        if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then break; fi
        if ! kill -0 "$pid" 2>/dev/null; then
            tail -n 12 "$LOG_FILE" 2>/dev/null | sed 's/^/  /' >&2
            cmd_stop >/dev/null 2>&1 || true
            die "the server exited while loading (full log: $LOG_FILE)"
        fi
        sleep 5; waited=$((waited + 5))
        [ $((waited % 60)) -eq 0 ] && say "  still loading (${waited}s)"
    done
    if [ "$waited" -ge "$READY_TIMEOUT_S" ]; then
        tail -n 12 "$LOG_FILE" 2>/dev/null | sed 's/^/  /' >&2
        cmd_stop >/dev/null 2>&1 || true
        die "the server did not answer /v1/models within ${READY_TIMEOUT_S}s (log: $LOG_FILE)"
    fi

    a=$(avail_mb)
    if [ "$a" -lt "$RUN_FLOOR_MB" ]; then
        cmd_stop >/dev/null 2>&1 || true
        die "only $a MB left after loading (need $RUN_FLOOR_MB); the machine would start swapping"
    fi
    say "loaded; ${a} MB of memory left"
    grep -aE 'effective:|^warn:' "$LOG_FILE" 2>/dev/null | tail -3 | sed 's/^/  /' || true
    cmd_test || warn "the smoke request failed; the server is up, so read $LOG_FILE"
    report_endpoint
}

report_endpoint() {
    local ip; ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    say ""
    say "ready: $MODEL_ID on $HOST:$PORT/v1   (revision $(revision_identity))"
    say "  local   http://127.0.0.1:$PORT/v1"
    say "  lan     http://${ip:-<host>}:$PORT/v1"
    say "  context ${CTX:-native 1048576}   banks ${MAX_SEQS:-auto}   mtp $MTP_MODE/$DRAFT"
    say "  stop    bash $0 stop"
}

cmd_stop() {
    [ -s "$WD_PID_FILE" ] && kill "$(cat "$WD_PID_FILE")" 2>/dev/null || true
    if alive; then
        local pid; pid=$(cat "$PID_FILE")
        say "stopping ds4-server (pid $pid)"
        kill "$pid" 2>/dev/null || true
        local i=0
        while [ "$i" -lt 30 ] && kill -0 "$pid" 2>/dev/null; do sleep 1; i=$((i + 1)); done
        kill -0 "$pid" 2>/dev/null && { say "  still alive after 30 s, sending KILL"; kill -9 "$pid" 2>/dev/null || true; }
    else
        say "ds4-server: not running"
    fi
    rm -f "$PID_FILE" "$WD_PID_FILE"
}

cmd_status() {
    if alive; then
        say "ds4-server:  running (pid $(cat "$PID_FILE"))"
        ps -o pid=,etime=,rss= -p "$(cat "$PID_FILE")" 2>/dev/null | sed 's/^/  /'
    else
        say "ds4-server:  not running"
    fi
    if [ -s "$WD_PID_FILE" ] && kill -0 "$(cat "$WD_PID_FILE")" 2>/dev/null; then
        say "watchdog:    running (pid $(cat "$WD_PID_FILE"))"
    else
        say "watchdog:    not running"
    fi
    say "memory:      $(avail_mb) MB available of $(total_mb) MB"
    say "release:     $(( $(parts_present) ))/$GGUF_PARTS parts in $MODEL_DIR $([ "$(parts_partial)" = 0 ] && echo '(no partial files)' || echo "($(parts_partial) PARTIAL)")"
    say "identity:    $(release_identity) release, $(revision_identity) revision"
    if [ -f "$MODEL_DIR/$TEMPLATE" ]; then
        say "markers:     $TEMPLATE $(sha_file "$MODEL_DIR/$TEMPLATE" | cut -c1-12) ($(fsize "$MODEL_DIR/$TEMPLATE") B), $MARKER $(fsize "$MODEL_DIR/$MARKER") B"
    fi
    if [ "$(revision_identity)" = "r13p8" ] || [ "$(revision_identity)" = "r13" ]; then
        local p
        for p in "${R13P8_PARTS[@]:0:2}"; do
            say "             $p $(fsize "$MODEL_DIR/$p") B $([ "$(fsize "$MODEL_DIR/$p")" = "$R13P8_SIZE_SMALL" ] && echo '(R13+8)' || echo '(R13)')"
        done
    fi
    if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | head -c 300; then
        say ""
        say "endpoint:    answering on port $PORT"
    else
        say "endpoint:    no answer on port $PORT"
    fi
    if [ -f "$LOG_FILE" ]; then
        say ""
        say "log tail:"
        tail -n 5 "$LOG_FILE" | sed 's/^/  /'
    fi
}

cmd_logs() {
    say "server log:   $LOG_FILE"
    [ -f "$LOG_FILE" ] && tail -n 20 "$LOG_FILE" | sed 's/^/  /'
    say ""
    say "watchdog log: $WD_LOG"
    [ -f "$WD_LOG" ] && tail -n 10 "$WD_LOG" | sed 's/^/  /'
    return 0
}

cmd_test() {
    need curl
    local out
    # 64 tokens: the template opens a thinking channel first, so a shorter cap would
    # return an empty content field and look like a failure.
    out=$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL_ID\",\"temperature\":0,\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, what does a vector-quantized expert layer store?\"}]}")
    case "$out" in
        *'"choices"'*)
            printf '%s\n' "$out" | sed -n 's/.*"timings":{\(.*\)}.*/  timings: \1/p' | head -1
            printf '%s\n' "$out" | sed -n 's/.*"reasoning_content":"\([^"]\{0,90\}\)".*/  thinking: \1.../p' | head -1
            printf '%s\n' "$out" | sed -n 's/.*"content":"\([^"]\{0,90\}\)".*/  answer: \1.../p' | head -1;;
        *) warn "the request failed: $(printf '%s' "$out" | head -c 300)"; return 1;;
    esac
}

usage() { sed -n '2,103p' "$0" | sed 's/^# \{0,1\}//'; }

case "${1:-}" in
    preflight) shift; preflight "$@";;
    plan)      shift; cmd_plan "$@";;
    download)  shift; cmd_download "$@";;
    verify)    shift; cmd_verify "$@";;
    check)     shift; cmd_check "$@";;
    start)     shift; cmd_start "$@";;
    stop)      shift; cmd_stop "$@";;
    status)    shift; cmd_status "$@";;
    logs)      shift; cmd_logs "$@";;
    test)      shift; cmd_test "$@";;
    all)       shift; preflight; cmd_download; cmd_check; cmd_start "$@";;
    ""|-h|--help|help) usage;;
    *) printf 'unknown action: %s (see --help)\n' "$1" >&2; usage >&2; exit 2;;
esac
