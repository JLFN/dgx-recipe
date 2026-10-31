#!/usr/bin/env bash
# GLM-5.3-Flash Uncensored VQ, R13+8 revision: fetch it into the release
# directory an earlier revision already occupies, prove which revision and which
# release the bytes belong to, and serve it on the DGX Spark.
#
#   artifact  Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF
#             pinned at 743a912d805e563340fa9d107e636cc0584145fd (main head,
#             2026-10-18T16:21:21Z), R13+8 on 19 routed layers
#             56 GGUF parts, 94,491,712,768 bytes (88.0 GiB) of payload
#   engine    ds4-dfm-rs from 34c46d1 (PR #96, "Add the GLM VQ R13+8 residual
#             stage") on; that is the build that reads GVQ1 container v2. The
#             revision's own card agrees: container v2 has been on main since
#             PR #96 (merge 34c46d18), the runtime binds the residual stage on
#             any routed layer, and this revision needs no runtime change.
#   sibling   the base VQ release, Baekpica/GLM-5.3-Flash-VQ-GGUF, carrying the
#             same structure at its own concurrent revision (71c73e7e,
#             2026-10-18T16:48:01Z)
#   older     this artifact's own two previous revisions, both kept upstream:
#             cc70aef5 at the tag r13p8-l3-16 (R13+8 on layers 3-16) and
#             1764dd87 at the tag q4k-v3 (all routed layers R13)
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
# TRAP 2, NEW HERE: THREE REVISIONS, THE SAME 56 NAMES, 19 PARTS REWRITTEN
#   The R13+8 stage adds an 8-bit residual stage to a routed layer and replaces
#   that layer's part in place upstream. It has been published three times, all
#   of it on 2026-10-17/18:
#     1764dd87 (tag q4k-v3)       no upgraded layer            77,272,028,544
#     cc70aef5 (tag r13p8-l3-16)  layers 3-16                  89,960,211,968
#     743a912d (the pin)          layers 3-16, 17, 22, 27, 41, 44   94,491,712,768
#   Per part: layers 03-09 are 1,477,184,128 at R13 and 2,383,481,600 once
#   upgraded; every other upgraded layer is 1,477,184,896 at R13 and
#   2,383,485,056 once upgraded. The layer list is the artifact's own, not a
#   reading: recipe.json at the pin sets residual_stage.layers to 3-16, 17, 22,
#   27, 41 and 44. Byte-identical across all three revisions are the other 37
#   parts (23 R13 layers, 9 native, 4 MTP, vision) and - checked against each
#   revision's own listing - the two release markers above, so the markers
#   cannot see the revision and a half-refetched directory passes them.
#   Three things guard it instead:
#     1. the revision is pinned by default and this script fetches that pin;
#     2. a revision fingerprint reads the 19 upgraded parts by size on disk and
#        classifies the directory as the pin, as r13p8-l3-16, as all-R13, as
#        mixed or as absent, naming what is stale or missing; every action that
#        touches weights then refuses anything but the state it asked for;
#     3. verify re-runs the release's own SHA256SUMS, which is the authority.
#
# THE ENGINE FLOOR IS A FORMAT FLOOR, NOT A DATE
#   Container v1 holds R13 layers; v2 adds the GVR8 residual stage
#   (crates/ds4-core/src/glm_vq.rs:21, FORMAT_VERSIONS 1..=2 on the PR #96 build).
#   A build older than PR #96 has no GVR8 type and fails the upgraded parts with
#   "unsupported tensor type" (that build's glm_vq.rs, c942523:370). --mtp-mode,
#   which the R13 script uses as its floor test, is advertised by both builds, so
#   the decisive check here is `check`: --check-config opens no weights and reads
#   the container versions.
#
# Usage (on the Spark, as the user who owns the model directory):
#   bash install-glm53-uncensored-vq-r13p8.sh preflight    GPU, CUDA 13, memory, disk, ports, both fingerprints, engine floor
#   bash install-glm53-uncensored-vq-r13p8.sh plan         what download and start would do
#   bash install-glm53-uncensored-vq-r13p8.sh download     the pinned revision, resumable file by file
#   bash install-glm53-uncensored-vq-r13p8.sh verify       sha256sum -c SHA256SUMS; reads all 94 GB
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
#   GLM53R13P8_REVISION takes a commit or a tag: the pinned commit by default,
#   and cc70aef5 or r13p8-l3-16 and 1764dd87 or q4k-v3 are accepted and guarded
#   as those revisions; empty tracks main and relaxes the revision check.
#
# Notes
#   - The repository is public: no token is needed and none is ever written here.
#     HF_TOKEN only raises the CDN rate limit; the CLI reads it from the
#     environment on its own.
#   - This script and the R13 one default to the SAME release directory,
#      /mnt/models/GLM-5.3-Flash-Uncensored-VQ-GGUF, because the operator chose an
#     in-place refetch: one directory, one endpoint (port 8009, model id
#     glm-5.3-flash-uncensored-vq). So download upgrades whatever of the three
#     revisions it finds there. The R13 installer sees none of this - the markers
#     are identical between the releases and between the revisions, and the part
#     count is 56 either way - so it must not be used on this directory.
#   - This revision fits ONE 1M bank, not two, and the margin is thin: the card's
#     quote for the 14-layer revision is 112.0 GiB of the 118.7 GiB the engine
#     sees on GB10, and this revision adds the 4.2 GiB its five extra layers
#     weigh, so about 116.2 GiB by that quote. The card labels the quote as not
#     re-run at this revision, so treat it as arithmetic, not a measurement. The
#     defaults follow from it: --max-seqs 1, and a run floor (RUN_FLOOR_MB, 2000)
#     and watchdog floor (KILL_MB, 1500) below the ~2.5 GiB the quote leaves free
#     rather than the 4 GiB an unrelated margin would suggest.
#   - download never fetches on a guess: it names the state it found in the
#     directory first, and when it finds no upgraded part at all it prints that
#     directory's own counts, part sizes and markers before it says "the whole
#     release", so a full 88 GiB fetch is always explained rather than assumed.
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
PIN=743a912d805e563340fa9d107e636cc0584145fd
# Pinned by default: the revision of 2026-10-18T16:21:21Z, R13+8 on 19 routed
# layers. The revision is what says which bytes are on disk, and the release
# markers cannot: they are identical across the releases and the revisions.
# Set GLM53R13P8_REVISION to the tag or commit of either older revision to go
# back - each is then guarded as what it is, not merely accepted - or to empty
# to track main, which relaxes the revision check to a warning.
REVISION=${GLM53R13P8_REVISION:-$PIN}
# The two previous revisions, both kept upstream as tags.
L3_16_REVISION=cc70aef5a5ad370fbd59a69dc0a39522a1ed5c08
L3_16_TAG=r13p8-l3-16
R13_REVISION=1764dd87b8ee92dd288d7df15bf62d120ae010d8
R13_TAG=q4k-v3

RELEASE_NAME="GLM-5.3-Flash-Uncensored-VQ-GGUF"
MODEL_ROOT=${GLM53R13P8_MODEL_ROOT:-/mnt/models}
# The SAME directory the earlier revisions occupy, deliberately: the refetch
# replaces the upgraded parts in place. Point this elsewhere to keep a revision
# of its own; the guards then refuse a directory that holds a different one.
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
# One bank by default at the native context. The card's quote is for the 14-layer
# revision - 112.0 GiB (weights 83.8, bank 12.8, scratch 10.7, MTP 0.7, floor 4) of
# the 118.7 GiB the engine sees on GB10 - and this revision weights 4.2 GiB more,
# so about 116.2 GiB by that quote: the automatic width is not relied on.
MAX_SEQS=${GLM53R13P8_MAX_SEQS:-1}
WATCHDOG=${GLM53R13P8_WATCHDOG:-1}      # 1 stops the server if MemAvailable stays low

STATE=${GLM53R13P8_STATE:-$HOME}
PID_FILE=${GLM53R13P8_PID:-$STATE/glm53unc.pid}
LOG_FILE=${GLM53R13P8_LOG:-$STATE/glm53unc.log}
WD_PID_FILE=${GLM53R13P8_WD_PID:-$STATE/glm53unc-watchdog.pid}
WD_LOG=${GLM53R13P8_WD_LOG:-$STATE/glm53unc-watchdog.log}

GGUF_PARTS=56                       # parts listed in files.json at any revision
PAYLOAD_BYTES=94491712768           # the 56 GGUF parts at the pin, 88.0 GiB
L3_16_PAYLOAD_BYTES=89960211968     # the same 56 names at cc70aef5, r13p8-l3-16
R13_PAYLOAD_BYTES=77272028544       # the same 56 names at 1764dd87, q4k-v3
DELTA_BYTES=$((PAYLOAD_BYTES - R13_PAYLOAD_BYTES))          #  17,219,684,224 over R13
L3_16_DELTA_BYTES=$((PAYLOAD_BYTES - L3_16_PAYLOAD_BYTES))  #   4,531,500,800 over r13p8-l3-16
REQUIRED_GIB=99                     # a fresh fetch: payload plus slack for partial files
WORK_ROOM_BYTES=8000000000          # 8 GB of headroom while a fetch is in flight
MAX_WORKERS=${GLM53R13P8_MAX_WORKERS:-8}
READY_TIMEOUT_S=${GLM53R13P8_READY_TIMEOUT:-900}
START_FLOOR_MB=${GLM53R13P8_START_FLOOR_MB:-100000}   # before loading
# After loading, the card's own quote for this revision (116.2 GiB of the 118.7
# GiB the engine sees on GB10) leaves about 2.5 GiB, so these floors sit under
# that and not at the 4 GiB the 14-layer revision's wider margin would allow.
# Arithmetic from the card's quote, not a measurement: see the header notes.
RUN_FLOOR_MB=${GLM53R13P8_RUN_FLOOR_MB:-2000}         # after loading
KILL_MB=${GLM53R13P8_KILL_MB:-1500}                   # two consecutive 5 s samples stop the server
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

# The revision fingerprint: the parts whose payload size moves between the three
# revisions, in three groups. The names are literal because they are the
# artifact's own: files.json lists exactly these 56 names at every revision.
#   early7   layers 03-09, upgraded by both R13+8 revisions
#   middle7  layers 10-16, upgraded by both R13+8 revisions
#   late5    layers 17, 22, 27, 41, 44, the five the pin adds over r13p8-l3-16
EARLY7_PARTS=(GLM-5.3-Flash-VQ-layer-03.gguf GLM-5.3-Flash-VQ-layer-04.gguf
              GLM-5.3-Flash-VQ-layer-05.gguf GLM-5.3-Flash-VQ-layer-06.gguf
              GLM-5.3-Flash-VQ-layer-07.gguf GLM-5.3-Flash-VQ-layer-08.gguf
              GLM-5.3-Flash-VQ-layer-09.gguf)
MIDDLE7_PARTS=(GLM-5.3-Flash-VQ-layer-10.gguf GLM-5.3-Flash-VQ-layer-11.gguf
               GLM-5.3-Flash-VQ-layer-12.gguf GLM-5.3-Flash-VQ-layer-13.gguf
               GLM-5.3-Flash-VQ-layer-14.gguf GLM-5.3-Flash-VQ-layer-15.gguf
               GLM-5.3-Flash-VQ-layer-16.gguf)
LATE5_PARTS=(GLM-5.3-Flash-VQ-layer-17.gguf GLM-5.3-Flash-VQ-layer-22.gguf
             GLM-5.3-Flash-VQ-layer-27.gguf GLM-5.3-Flash-VQ-layer-41.gguf
             GLM-5.3-Flash-VQ-layer-44.gguf)
EARLY7_LABEL="layers 03-09 "
MIDDLE7_LABEL="layers 10-16 "
LATE5_LABEL="layers 17, 22, 27, 41, 44 "
UPGRADED_SIZE_EARLY=2383481600      # layers 03-09 once R13+8 (1,477,184,128 at R13)
UPGRADED_SIZE_OTHER=2383485056      # every other upgraded layer (1,477,184,896 at R13)
R13_SIZE_EARLY=1477184128
R13_SIZE_OTHER=1477184896

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

# The group counters: how many parts of each group are at the upgraded size, at
# the R13 size, missing, or at neither size. One counting pass, read by
# revision_state, stale_parts, diagnose_dir and status, so they can never
# disagree with each other.
count_groups() {
    COUNT_E_NEW=0; COUNT_E_OLD=0; COUNT_E_MISS=0; COUNT_E_ODD=0
    COUNT_M_NEW=0; COUNT_M_OLD=0; COUNT_M_MISS=0; COUNT_M_ODD=0
    COUNT_L_NEW=0; COUNT_L_OLD=0; COUNT_L_MISS=0; COUNT_L_ODD=0
    local p s
    for p in "${EARLY7_PARTS[@]}"; do
        s=$(fsize "$MODEL_DIR/$p")
        case "$s" in
            "$UPGRADED_SIZE_EARLY") COUNT_E_NEW=$((COUNT_E_NEW + 1));;
            "$R13_SIZE_EARLY")      COUNT_E_OLD=$((COUNT_E_OLD + 1));;
            0)                      COUNT_E_MISS=$((COUNT_E_MISS + 1));;
            *)                      COUNT_E_ODD=$((COUNT_E_ODD + 1));;
        esac
    done
    for p in "${MIDDLE7_PARTS[@]}"; do
        s=$(fsize "$MODEL_DIR/$p")
        case "$s" in
            "$UPGRADED_SIZE_OTHER") COUNT_M_NEW=$((COUNT_M_NEW + 1));;
            "$R13_SIZE_OTHER")      COUNT_M_OLD=$((COUNT_M_OLD + 1));;
            0)                      COUNT_M_MISS=$((COUNT_M_MISS + 1));;
            *)                      COUNT_M_ODD=$((COUNT_M_ODD + 1));;
        esac
    done
    for p in "${LATE5_PARTS[@]}"; do
        s=$(fsize "$MODEL_DIR/$p")
        case "$s" in
            "$UPGRADED_SIZE_OTHER") COUNT_L_NEW=$((COUNT_L_NEW + 1));;
            "$R13_SIZE_OTHER")      COUNT_L_OLD=$((COUNT_L_OLD + 1));;
            0)                      COUNT_L_MISS=$((COUNT_L_MISS + 1));;
            *)                      COUNT_L_ODD=$((COUNT_L_ODD + 1));;
        esac
    done
}

# Which revision the 19 upgraded parts say is on disk: pinned, l3-16, r13, mixed,
# absent, or unknown. Only these parts can answer it: TRAP 2's markers are
# identical across the revisions. Every group must agree, so an interrupted
# refetch reads as mixed rather than rounding to one side.
# The blind spot, stated rather than implied: a fourth upstream revision that
# rewrote the same 19 layers with the same sizes would read as the pin. Content
# is what settles that, and verify re-runs the release's own SHA256SUMS for it.
revision_state() {
    count_groups
    if [ $((COUNT_E_ODD + COUNT_M_ODD + COUNT_L_ODD)) -gt 0 ]; then printf 'unknown'; return; fi
    if [ "$COUNT_E_MISS" = 7 ] && [ "$COUNT_M_MISS" = 7 ] && [ "$COUNT_L_MISS" = 5 ]; then
        printf 'absent'; return
    fi
    if [ "$COUNT_E_NEW" = 7 ] && [ "$COUNT_M_NEW" = 7 ] && [ "$COUNT_L_NEW" = 5 ]; then
        printf 'pinned'; return
    fi
    if [ "$COUNT_E_NEW" = 7 ] && [ "$COUNT_M_NEW" = 7 ] && [ "$COUNT_L_OLD" = 5 ]; then
        printf 'l3-16'; return
    fi
    if [ "$COUNT_E_OLD" = 7 ] && [ "$COUNT_M_OLD" = 7 ] && [ "$COUNT_L_OLD" = 5 ]; then
        printf 'r13'; return
    fi
    printf 'mixed'
}

# Which revision this run asked for, so the guards know what to accept. An
# explicit older revision is guarded as itself; empty tracks main and any other
# value is a revision this script has no fingerprint for, so it warns instead.
expected_state() {
    case "$REVISION" in
        "$PIN")                          printf 'pinned';;
        "$L3_16_REVISION"|"$L3_16_TAG")  printf 'l3-16';;
        "$R13_REVISION"|"$R13_TAG")      printf 'r13';;
        "")                              printf 'pinned';;
        *)                               printf 'relaxed';;
    esac
}

# Names the parts that are not the pinned revision's, with the size found and the
# size wanted, so a mixed directory gets an actionable message rather than a
# diagnosis. A missing part is named with size 0.
stale_parts() {
    local p s want out=()
    for p in "${EARLY7_PARTS[@]}"; do
        s=$(fsize "$MODEL_DIR/$p"); want=$UPGRADED_SIZE_EARLY
        [ "$s" = "$want" ] || out+=("$p ($s, want $want)")
    done
    for p in "${MIDDLE7_PARTS[@]}" "${LATE5_PARTS[@]}"; do
        s=$(fsize "$MODEL_DIR/$p"); want=$UPGRADED_SIZE_OTHER
        [ "$s" = "$want" ] || out+=("$p ($s, want $want)")
    done
    printf '%s' "${out[*]:-}"
}

# Why the fingerprint says what it says: the counts per group, the files, the
# markers, and what each of the three revisions expects. Printed whenever
# download is about to fetch the whole release and whenever preflight cannot
# classify the directory, so a 88 GiB fetch is explained rather than assumed.
diagnose_dir() {
    count_groups
    say "directory:   $MODEL_DIR"
    if [ -d "$MODEL_DIR" ]; then
        say "  on disk:   $(parts_present) of $GGUF_PARTS parts, $(parts_partial) partial, files.json $(fsize "$MODEL_DIR/files.json") B, tensor-manifest.json $(fsize "$MODEL_DIR/tensor-manifest.json") B"
        say "  sizes:     $EARLY7_LABEL  $COUNT_E_NEW at $UPGRADED_SIZE_EARLY, $COUNT_E_OLD at $R13_SIZE_EARLY, $COUNT_E_MISS missing, $COUNT_E_ODD at neither"
        say "             $MIDDLE7_LABEL $COUNT_M_NEW at $UPGRADED_SIZE_OTHER, $COUNT_M_OLD at $R13_SIZE_OTHER, $COUNT_M_MISS missing, $COUNT_M_ODD at neither"
        say "             $LATE5_LABEL $COUNT_L_NEW at $UPGRADED_SIZE_OTHER, $COUNT_L_OLD at $R13_SIZE_OTHER, $COUNT_L_MISS missing, $COUNT_L_ODD at neither"
        say "  markers:   $(release_identity) release, $TEMPLATE $(fsize "$MODEL_DIR/$TEMPLATE") B, $MARKER $(fsize "$MODEL_DIR/$MARKER") B"
    else
        say "  on disk:   nothing; the directory does not exist"
    fi
    say "  pin        $PIN: $EARLY7_LABEL 7 at $UPGRADED_SIZE_EARLY and 12 at $UPGRADED_SIZE_OTHER ($MIDDLE7_LABEL + $LATE5_LABEL)"
    say "  $L3_16_TAG  $L3_16_REVISION: the same, except $LATE5_LABEL still at $R13_SIZE_OTHER"
    say "  $R13_TAG     $R13_REVISION: all 19 still at their R13 sizes"
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

# The one guard every action that touches weights calls: the directory must hold
# the revision this run asked for. Every other state is refused by name, with the
# single command that changes it. When REVISION is neither the pin nor a tagged
# older revision, the check relaxes to a warning, the same way the release-marker
# check relaxes in assert_uncensored.
assert_expected_revision() {
    local want rev
    want=$(expected_state)
    rev=$(revision_state)
    if [ "$want" = "relaxed" ]; then
        warn "GLM53R13P8_REVISION=$REVISION is not one of the three published revisions, so the
       revision fingerprint is a report and not a guard: it reads '$rev' in $MODEL_DIR. verify
       (sha256sum -c SHA256SUMS) is then the only thing that proves which bytes these are."
        return 0
    fi
    [ "$rev" = "$want" ] && return 0
    case "$rev" in
        pinned) die "$MODEL_DIR already holds the pinned revision ($PIN, $(human $PAYLOAD_BYTES) of
       payload) and this run asked for $REVISION. One directory cannot be two revisions, because
       the refetch is in place: point GLM53R13P8_MODEL_DIR at a directory of its own for the
       older revision, or drop GLM53R13P8_REVISION to use the pin.";;
        l3-16) die "$MODEL_DIR still holds the r13p8-l3-16 revision ($L3_16_REVISION): its upgraded
       parts stop at layer 16, while the pin also upgrades layers 17, 22, 27, 41 and 44, which is
       $(human $L3_16_DELTA_BYTES) more. The release markers cannot see the difference.
       Run: bash $0 download  (it refetches those five parts in place)";;
        r13) die "$MODEL_DIR still holds the all-R13 revision ($R13_REVISION, tag $R13_TAG): all 19
       upgraded parts are at R13 sizes. The release markers cannot see this - they are identical
       across the releases and the revisions - so the engine would load 19 stale parts and serve
       them without a word.
       Run: bash $0 download  (it refetches the 19 parts in place)";;
        mixed) die "$MODEL_DIR holds a MIXED directory: not all 19 upgraded parts are the pinned
       revision's, after a refetch that did not finish. Stale: $(stale_parts)
       Re-run: bash $0 download   (resumable; then bash $0 verify)";;
        absent) die "$MODEL_DIR holds no upgraded layer part at all; run: bash $0 download";;
        *) die "$MODEL_DIR holds layer parts whose sizes match none of the three revisions. An
       upgraded part is 2,383,481,600 (layers 03-09) or 2,383,485,056 (any other upgraded layer);
       at R13 it is 1,477,184,128 or 1,477,184,896. Run: bash $0 verify";;
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
    local free state need=0
    state=$(revision_state)
    if ! release_ok; then
        # What this fetch still has to add, from the state the fingerprint read:
        # nothing on a complete pinned directory, the five late parts over
        # r13p8-l3-16, all 19 over R13 (which is also the ceiling for a mixed
        # directory: no in-place refetch can add more than the 19 parts), and the
        # whole payload when the directory holds none of them.
        case "$state" in
            l3-16) need="$L3_16_DELTA_BYTES";;
            r13|mixed|unknown) need="$DELTA_BYTES";;
            *)     need="$PAYLOAD_BYTES";;
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
        case "$state" in
            pinned) say "revision ok: the pin ($PIN) is on disk; all 19 upgraded parts are at R13+8 sizes";;
            l3-16)  say "revision: $L3_16_TAG on disk; download replaces the five late parts in place, +$(human $L3_16_DELTA_BYTES)";;
            r13)    say "revision: $R13_TAG (all-R13) on disk, $(human $R13_PAYLOAD_BYTES) of payload; download replaces the 19 upgraded parts in place, +$(human $DELTA_BYTES)";;
            mixed)  warn "revision: MIXED - stale parts: $(stale_parts)"; say "download refetches exactly those";;
            *)      warn "revision: cannot classify the upgraded part sizes ($state)"; diagnose_dir;;
        esac
    else
        say "no release yet in $MODEL_DIR ($(human $PAYLOAD_BYTES) to fetch)"; diagnose_dir
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
                warn "$ENGINE_SRC does not contain $ENGINE_MIN_COMMIT ($ENGINE_MIN_DESC): a build older than it has no GVR8 type and rejects the upgraded parts with 'unsupported tensor type'. Fetch the engine and rebuild."
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
    local state; state=$(revision_state)
    say "install root:  $MODEL_ROOT"
    say "release dir:   $MODEL_DIR   (an earlier revision's directory: the refetch is in place)"
    say "runtime:       $SERVER_BIN"
    say "serve:         $HOST:$PORT, model id $MODEL_ID, watchdog below ${KILL_MB} MB"
    say ""
    say "artifacts"
    say "  revision     $REPO @ ${REVISION:-main}"
    say "  parts        $GGUF_PARTS GGUF files, $(human $PAYLOAD_BYTES)   $([ "$(parts_present)" -eq "$GGUF_PARTS" ] && echo present || echo "$(parts_present) on disk")"
    say "  19 of them   rewritten by R13+8, each with an 8-bit residual stage (GVQ1 container v2):"
    say "                 layers 03-16, and the five the pin adds over r13p8-l3-16: 17, 22, 27, 41, 44"
    say "  the other 37 unchanged from R13: 23 R13 layers, 9 native parts, 4 MTP parts, 1 vision tower"
    say "  engine floor $ENGINE_MIN_COMMIT ($ENGINE_MIN_DESC): container v2 is rejected by older builds"
    say "  tables       files.json, tensor-manifest.json, tokenizer*, chat_template.jinja, SHA256SUMS"
    say "  not loaded   GLM-5.3-Flash-vision-01-of-01.gguf ships in the release; this script does not pass --vision"
    say ""
    say "release identity (this release vs its base sibling; identical at every revision)"
    say "  $TEMPLATE  $TEMPLATE_BYTES bytes, $TEMPLATE_SHA"
    say "  $MARKER  $MARKER_BYTES bytes, $MARKER_SHA"
    say "revision identity (the only on-disk difference, by the sizes of the 19 upgraded parts)"
    say "  $EARLY7_LABEL       $UPGRADED_SIZE_EARLY at R13+8,        $R13_SIZE_EARLY at R13"
    say "  $MIDDLE7_LABEL      $UPGRADED_SIZE_OTHER at R13+8,        $R13_SIZE_OTHER at R13"
    say "  $LATE5_LABEL $UPGRADED_SIZE_OTHER at the pin,   $R13_SIZE_OTHER at r13p8-l3-16 and R13"
    say "  on disk now:  $state"
    say ""
    say "the command start would run:"
    say "  $SERVER_BIN --cuda -m $MODEL_DIR \\"
    say "      --mtp-mode $MTP_MODE --mtp-draft $DRAFT --max-seqs $MAX_SEQS --host $HOST --port $PORT \\"
    say "      --model-id $MODEL_ID --no-update-check${CTX:+ -c $CTX}"
    [ "$MTP_MODE" != "off" ] && say "  with DS4_CONT_MTP_MODE=2 DS4_CONT_MTP_DEPTH=$DRAFT in the environment (see the header)"
    say "  (no -c unless GLM53R13P8_CTX is set: the family defaults to the native 1048576)"
    say "  (--max-seqs $MAX_SEQS is explicit: the card's quote for this revision is ~116.2 GiB of the 118.7 GiB the engine sees on GB10, so two banks do not fit)"
    say ""
    local need=0
    if ! release_ok; then
        case "$state" in
            l3-16)             need=$L3_16_DELTA_BYTES;;
            r13|mixed|unknown) need=$DELTA_BYTES;;
            *)                 need=$PAYLOAD_BYTES;;
        esac
    fi
    say "still to fetch: $(human $need); free on target $(df_target "$MODEL_ROOT"): $(human "$(df_avail "$MODEL_ROOT")")"
    say "ports 8000, 8003, 8004, 8005, 8006, 8007, 8008 and 8010 answer for the other launchers in this home"
}

# --------------------------------------------------------------------------
# download, verify, check
# --------------------------------------------------------------------------
cmd_download() {
    mkdir -p "$MODEL_DIR"
    local before want; before=$(revision_state); want=$(expected_state)
    if release_ok && [ "$before" = "$want" ] && [ "$want" != "relaxed" ]; then
        assert_uncensored
        say "already downloaded: $(parts_present) parts in $MODEL_DIR, and the fingerprint reads '$before'"
        return 0
    fi
    ensure_hf
    local free; free=$(df_avail "$MODEL_ROOT")
    [ -z "$free" ] || [ "$free" -gt "$WORK_ROOM_BYTES" ] \
        || die "$MODEL_ROOT has $(human ${free:-0}) free, needs at least $(human $WORK_ROOM_BYTES) of working room"
    say "on disk:      $MODEL_DIR reads '$before' - the fingerprint names the revision by the sizes of the 19 upgraded parts"
    case "$before" in
        l3-16)  say "in-place upgrade from $L3_16_TAG: the five late parts (layers 17, 22, 27, 41, 44) are replaced,"
                say "  $(human $L3_16_DELTA_BYTES) is added, and the other 51 parts are byte-identical and are left alone.";;
        r13)    say "in-place refetch from $R13_TAG (all-R13): all 19 upgraded parts are replaced,"
                say "  $(human $DELTA_BYTES) is added, and the other 37 parts are byte-identical and are left alone.";;
        pinned) say "the pin is already on disk; the fetch only fills what is missing";;
        mixed)  say "in-place refetch of a MIXED directory; stale or missing parts: $(stale_parts)";;
        *)      say "the whole release is fetched: this directory holds none of the three revisions' 19 upgraded"
                say "  parts, so there is nothing to refetch in place - 56 parts, $(human $PAYLOAD_BYTES)"
                diagnose_dir;;
    esac
    say "fetching $REPO${REVISION:+ @ $REVISION} into $MODEL_DIR"
    say "an interrupted run resumes from where it stopped; just run it again"
    hf_download || die "download failed; run the same command again to resume"

    local after; after=$(revision_state)
    if [ "$want" = "relaxed" ]; then
        warn "the fetch finished and the fingerprint reads '$after'; this run asked for a revision it has
       no fingerprint for, so verify is what settles whether the bytes are right: bash $0 verify"
    elif [ "$after" = "$want" ]; then
        say "revision ok: the directory now reads '$after'$([ "$want" = "pinned" ] && printf ' - the pin, all 19 upgraded parts at R13+8 sizes')"
    else
        case "$after" in
            l3-16) warn "the fetch finished but the directory still holds $L3_16_TAG sizes: layers 17, 22, 27, 41 and 44 were not replaced.";;
            r13)   warn "the fetch finished but the directory still holds all-R13 sizes.";;
            *)     warn "the fetch left the directory reading '$after'.";;
        esac
        warn "The CLI can skip a file it believes it already has; GLM53R13P8_FORCE=1 re-fetches every file."
        die "the directory does not hold the revision this run asked for: run bash $0 verify, or GLM53R13P8_FORCE=1 bash $0 download"
    fi
    release_ok || die "the fetch finished but the release is incomplete: $(parts_present)/$GGUF_PARTS parts, $(parts_partial) partial"
    assert_uncensored
    say "done: $GGUF_PARTS parts, $(human "$(du -sb "$MODEL_DIR" | cut -f1)") on disk, markers say $REPO's Uncensored release, $(human $PAYLOAD_BYTES) of payload"
    say "next: bash $0 verify (hashes all 94 GB), then bash $0 check, then bash $0 start"
}

cmd_verify() {
    release_ok || die "the release is incomplete in $MODEL_DIR; run download first"
    assert_uncensored
    assert_expected_revision
    need sha256sum
    say "hashing $(human "$(du -sb "$MODEL_DIR" | cut -f1)") ; this reads every byte"
    ( cd "$MODEL_DIR" && sha256sum -c SHA256SUMS ) || die "a payload hash does not match SHA256SUMS"
    say "verify: every payload hash matches SHA256SUMS, so the parts on disk are the ones that revision published"
}

# --check-config prints the plan as JSON and opens no weights, so this is the
# cheapest way to see the family, the context and the memory quote before a load -
# and the only action that proves the engine reads container v2.
cmd_check() {
    release_ok || die "the release is incomplete in $MODEL_DIR; run download first"
    assert_uncensored
    assert_expected_revision
    [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN (build it: cd $ENGINE_SRC && make cuda-spark)"
    say "release: $(( $(parts_present) ))/$GGUF_PARTS parts, revision $(revision_state), markers say uncensored"
    local args=(--check-config -m "$MODEL_DIR")
    [ -n "$CTX" ] && args+=(-c "$CTX")
    [ -n "$MAX_SEQS" ] && args+=(--max-seqs "$MAX_SEQS")
    [ "$MTP_MODE" = "off" ] || args+=(--mtp-mode "$MTP_MODE" --mtp-draft "$DRAFT")
    if ! "$SERVER_BIN" "${args[@]}"; then
        die "the plan quote failed. A build older than $ENGINE_MIN_COMMIT ($ENGINE_MIN_DESC) has no
       GVR8 type and rejects the upgraded parts as 'unsupported tensor type'; rebuild with:
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
    assert_expected_revision
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
        say "the host hashes 94 GB and maps the weights before it listens: on the previous revision that took about a minute"
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
    say "loading 94 GB and verifying every hash; the port binds only after that"

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
    say "ready: $MODEL_ID on $HOST:$PORT/v1   (revision $(revision_state))"
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
    local state want; state=$(revision_state); want=$(expected_state)
    say "identity:    $(release_identity) release, revision '$state'"
    if [ -f "$MODEL_DIR/$TEMPLATE" ]; then
        say "markers:     $TEMPLATE $(sha_file "$MODEL_DIR/$TEMPLATE" | cut -c1-12) ($(fsize "$MODEL_DIR/$TEMPLATE") B), $MARKER $(fsize "$MODEL_DIR/$MARKER") B"
    fi
    # The group counts in the same words the guards use: this is the line that says
    # at a glance which of the three revisions the directory holds, and what this
    # run wants it to hold.
    count_groups
    say "fingerprint: $COUNT_E_NEW/7 + $COUNT_M_NEW/7 + $COUNT_L_NEW/5 upgraded parts in $EARLY7_LABEL $MIDDLE7_LABEL $LATE5_LABEL"
    say "             -> '$state'$([ "$state" = mixed ] && printf ', stale: %s' "$(stale_parts)")"
    say "wanted:      '$want' (GLM53R13P8_REVISION=${REVISION:-main})"
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

usage() { sed -n '2,130p' "$0" | sed 's/^# \{0,1\}//'; }

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
