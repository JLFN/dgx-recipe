#!/usr/bin/env bash
# GLM-5.3-Flash Uncensored VQ (Q4_K nonexpert v3): fetch the release, prove it is
# the uncensored one, and serve it on the DGX Spark.
#
#   artifact  Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF
#             pinned at 1764dd87b8ee92dd288d7df15bf62d120ae010d8
#             56 GGUF parts, 77,272,028,544 bytes (71.97 GiB) of payload
#   engine    ds4-dfm-rs from c5384b5e (PR #86) on; the artifact card's own
#             validation ran at 7ef77272 (PR #89). The reference box builds
#             optimize4.
#   parent    the base VQ release, Baekpica/GLM-5.3-Flash-VQ-GGUF at 4da527cd,
#             whose structure, part names and payload sizes this one repeats.
#
# THIS SCRIPT'S SHAPE, AND WHERE IT CAME FROM
#   The engine tree's own installer for the base release, misc/glm53flash_install.sh,
#   is the lineage: same actions (download, verify, check, start, stop, status,
#   logs, all), same environment-override style, same "the release is a DIRECTORY"
#   rule. Three things come from this repository's launcher for the base release,
#   scripts/start-glm53-flash-vq.sh, instead: the machine preflight, the memory
#   watchdog, and the smoke test with a 64-token cap. And one thing is new, because
#   this artifact is the first one here whose name collides with a sibling's.
#
# THE TRAP THIS SCRIPT EXISTS TO CLOSE: TWO RELEASES, ONE SET OF PART NAMES
#   The Uncensored release repeats the base release's 56 part names, tensor order
#   and payload sizes; only the source weights differ (its card says so, and the
#   Hub listings confirm it: the 56 GGUF names are identical, and the only part
#   whose size differs is GLM-5.3-Flash-MTP-02-of-04.gguf, 1,997,061,632 against
#   the base's 1,997,061,504). The engine identifies both as the same artifact:
#   its shape name is the literal string "GLM-5.3-Flash-VQ"
#   (crates/ds4-core/src/shape.rs:1122, served as the variant name at
#   crates/ds4-core/src/serving.rs:1485). So nothing downstream can tell a
#   directory of the wrong release from the right one.
#   Two guards therefore run on every action that touches weights:
#     1. the release lands in its own directory, default
#        /mnt/models/GLM-5.3-Flash-Uncensored-VQ-GGUF, never beside the base one;
#     2. two markers are checked against digests read from the release's own
#        SHA256SUMS: chat_template.jinja (this release 10,644 bytes, the base
#        10,950, because abliteration also changed the template's provenance) and
#        the MTP-02 part's size. A directory holding the base release is refused
#        by name.
#
# Usage (on the Spark, as the user who owns the model directory):
#   bash install-glm53-uncensored-vq.sh preflight   GPU, CUDA 13, memory, disk, ports, fingerprint
#   bash install-glm53-uncensored-vq.sh plan        what download and start would do
#   bash install-glm53-uncensored-vq.sh download    the release, resumable file by file
#   bash install-glm53-uncensored-vq.sh verify      sha256sum -c SHA256SUMS; reads all 77 GB
#   bash install-glm53-uncensored-vq.sh check       the engine's plan quote, opens no weights
#   bash install-glm53-uncensored-vq.sh start       serve in the foreground
#   bash install-glm53-uncensored-vq.sh start --daemon   serve detached, wait for readiness
#   bash install-glm53-uncensored-vq.sh status | stop | logs | test
#   bash install-glm53-uncensored-vq.sh all         preflight, download, check, start
#
# Env overrides:
#   GLM53UNC_REPO, GLM53UNC_REVISION, GLM53UNC_MODEL_DIR, GLM53UNC_BIN,
#   GLM53UNC_HOST, GLM53UNC_PORT, GLM53UNC_MODEL_ID, GLM53UNC_MTP_MODE,
#   GLM53UNC_DRAFT, GLM53UNC_CTX, GLM53UNC_MAX_SEQS, GLM53UNC_PID, GLM53UNC_LOG,
#   GLM53UNC_ALLOW_SHARED, GLM53UNC_WATCHDOG, GLM53UNC_MAX_WORKERS, HF_TOKEN.
#
# Notes
#   - The repository is public: no token is needed and none is ever written here.
#     HF_TOKEN only raises the CDN rate limit; the CLI reads it from the
#     environment on its own.
#   - The engine verifies every file, tensor and payload SHA256 while it loads the
#     release (about 20 s for this artifact), so verify is a second, independent
#     opinion rather than a precondition.
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
REPO=${GLM53UNC_REPO:-Baekpica/GLM-5.3-Flash-Uncensored-VQ-GGUF}
PIN=1764dd87b8ee92dd288d7df15bf62d120ae010d8
# Pinned by default, unlike the base installer which tracks main. The reason is
# the trap above: the two releases are indistinguishable by name, so the revision
# is the only thing that says which bytes are on disk. Set GLM53UNC_REVISION= to
# track main, or to another commit, and the digest guard relaxes to the
# base-release refusal only (it says so when it does).
REVISION=${GLM53UNC_REVISION:-$PIN}

RELEASE_NAME="GLM-5.3-Flash-Uncensored-VQ-GGUF"
MODEL_ROOT=${GLM53UNC_MODEL_ROOT:-/mnt/models}
MODEL_DIR=${GLM53UNC_MODEL_DIR:-$MODEL_ROOT/$RELEASE_NAME}
SERVER_BIN=${GLM53UNC_BIN:-$HOME/ds4-dfm-rs/ds4-server}
ENGINE_SRC=${GLM53UNC_ENGINE_SRC:-$HOME/ds4-dfm-rs}
HOST=${GLM53UNC_HOST:-0.0.0.0}
# 8000 is DeepSeek V4.1 Flash, 8003 Qwen3.8, 8005 Bonsai, 8006 YoungAi, 8007 the
# base GLM-5.3 VQ and 8008 the 1017 V4.1 launcher in this home, so the Uncensored
# GLM takes 8009 rather than answering on the base release's entry.
PORT=${GLM53UNC_PORT:-8009}
MODEL_ID=${GLM53UNC_MODEL_ID:-glm-5.3-flash-uncensored-vq}
MTP_MODE=${GLM53UNC_MTP_MODE:-on}       # auto resolves to off for this family: see the header
DRAFT=${GLM53UNC_DRAFT:-2}              # the draft count the base release's card measured
CTX=${GLM53UNC_CTX:-}                   # empty keeps the native 1,048,576
MAX_SEQS=${GLM53UNC_MAX_SEQS:-}         # empty leaves the plan's automatic width (two banks)
WATCHDOG=${GLM53UNC_WATCHDOG:-1}        # 1 stops the server if MemAvailable stays low

STATE=${GLM53UNC_STATE:-$HOME}
PID_FILE=${GLM53UNC_PID:-$STATE/glm53unc.pid}
LOG_FILE=${GLM53UNC_LOG:-$STATE/glm53unc.log}
WD_PID_FILE=${GLM53UNC_WD_PID:-$STATE/glm53unc-watchdog.pid}
WD_LOG=${GLM53UNC_WD_LOG:-$STATE/glm53unc-watchdog.log}

GGUF_PARTS=56                       # parts listed in files.json at this revision
PAYLOAD_BYTES=77272028544           # the 56 GGUF parts, 71.97 GiB
REQUIRED_GIB=85                     # payload plus slack for partial files
MAX_WORKERS=${GLM53UNC_MAX_WORKERS:-8}
READY_TIMEOUT_S=${GLM53UNC_READY_TIMEOUT:-900}
START_FLOOR_MB=${GLM53UNC_START_FLOOR_MB:-100000}   # before loading
RUN_FLOOR_MB=${GLM53UNC_RUN_FLOOR_MB:-10000}        # after loading, before the box swaps
KILL_MB=${GLM53UNC_KILL_MB:-2500}                   # two consecutive 5 s samples stop the server
ENGINE_MIN_COMMIT=c5384b5e          # PR #86; the artifact card's floor for reading this release

# The two markers, from the release's own SHA256SUMS at the pinned revision.
# Each pair is (size, sha256) for the Uncensored release and for the base release,
# so the script can tell them apart and can name the wrong one.
TEMPLATE=chat_template.jinja
TEMPLATE_BYTES=10644
TEMPLATE_SHA=34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891
BASE_TEMPLATE_SHA=0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5
MARKER=GLM-5.3-Flash-MTP-02-of-04.gguf
MARKER_BYTES=1997061632
MARKER_SHA=68a41631cabc0b107e04ab2418e186b76e5debea60d95e1811953cf4f1c28c26
BASE_MARKER_BYTES=1997061504
BASE_MARKER_SHA=194058b2ae15048b010149539057cdedbaf2ef11da7f7e4adecf96f45ced51af

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
say()  { printf 'glm53unc: %s\n' "$*"; }
warn() { printf 'glm53unc: warning: %s\n' "$*" >&2; }
die()  { printf 'glm53unc: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
total_mb() { awk '/MemTotal/{print int($2/1024)}' /proc/meminfo; }
fsize()    { stat -L -c %s "$1" 2>/dev/null || echo 0; }
df_avail() { df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' \n' || true; }
df_target(){ df --output=target "$1" 2>/dev/null | tail -1 || true; }
free_gib() { df -Pk "$1" | awk 'NR==2 { printf "%d", $4 / 1048576 }'; }
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
# and the MTP-02 size are the two places the releases actually differ, and both are
# published in SHA256SUMS.
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
       would notice. Point GLM53UNC_MODEL_DIR at a directory of its own.";;
    esac
    if [ "$REVISION" = "$PIN" ]; then
        die "$MODEL_DIR holds neither release: $TEMPLATE is $(sha_file "$MODEL_DIR/$TEMPLATE")
       and $MARKER is $(fsize "$MODEL_DIR/$MARKER") bytes, against $TEMPLATE_SHA and
       $MARKER_BYTES published for the Uncensored release at $PIN. Re-run download, or delete
       the directory and fetch it again."
    fi
    warn "$MODEL_DIR does not match the pinned revision's digests; the check is relaxed because
       GLM53UNC_REVISION=$REVISION is not the pin $PIN (the base-release refusal still applies)"
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

hf_download() {
    export HF_HUB_MAX_WORKERS="$MAX_WORKERS"
    export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
    [ -n "${ENDPOINT:-}" ] && export HF_ENDPOINT="$ENDPOINT"
    [ "${NO_XET:-0}" = 1 ] && export HF_HUB_DISABLE_XET=1
    local -a rev=()
    [ -n "$REVISION" ] && rev=(--revision "$REVISION")
    "$HF" download "$REPO" "${rev[@]}" --local-dir "$MODEL_DIR" --max-workers "$MAX_WORKERS"
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
        warn "only ${t} MB of memory; the base release's plan quotes 118.4 GB for two banks at the native context"; fatal=1
    fi
    say "memory available now: $(avail_mb) MB (start refuses below ${START_FLOOR_MB} MB)"

    mkdir -p "$MODEL_ROOT" 2>/dev/null || { warn "cannot create $MODEL_ROOT"; fatal=1; }
    local free need=0
    release_ok || need="$PAYLOAD_BYTES"
    free=$(df_avail "$MODEL_ROOT")
    if [ -n "$free" ] && [ "$free" -gt $((need + 5000000000)) ]; then
        say "disk ok: $(human $need) still to fetch, $(human $free) free on $(df_target "$MODEL_ROOT")"
    else
        warn "not enough room on $(df_target "$MODEL_ROOT"): want $(human $need) plus 5 GB working room, have $(human ${free:-0})"
        warn "set GLM53UNC_MODEL_ROOT= to a volume with room (the release needs about ${REQUIRED_GIB} GB in total)"; fatal=1
    fi

    # The fingerprint, when a release is already on disk.
    if [ -f "$MODEL_DIR/$TEMPLATE" ]; then
        local id; id=$(release_identity)
        case "$id" in
            uncensored) say "release ok: the Uncensored release is in $MODEL_DIR ($(parts_present)/$GGUF_PARTS parts)";;
            base) warn "$MODEL_DIR holds the BASE release, not this one; download would land beside it wrongly - point GLM53UNC_MODEL_DIR elsewhere"; fatal=1;;
            *) warn "$MODEL_DIR holds a release whose markers match neither release; re-run download"; fatal=1;;
        esac
    else
        say "no release yet in $MODEL_DIR ($(human $PAYLOAD_BYTES) to fetch)"
    fi

    # The engine's floor for reading this release.
    if [ -x "$SERVER_BIN" ]; then
        say "runtime ok: $SERVER_BIN"
        local help; help=$("$SERVER_BIN" --help 2>&1 || true)
        case "$help" in
            *--mtp-mode*) ;;
            *) warn "$SERVER_BIN does not advertise --mtp-mode: it predates GLM-5.3 VQ support (PR #86, $ENGINE_MIN_COMMIT); rebuild with: cd $ENGINE_SRC && make cuda-spark"; fatal=1;;
        esac
        if [ -d "$ENGINE_SRC/.git" ]; then
            if git -C "$ENGINE_SRC" cat-file -e "${ENGINE_MIN_COMMIT}^{commit}" 2>/dev/null; then
                say "engine checkout ok: $ENGINE_SRC knows $ENGINE_MIN_COMMIT"
            else
                warn "$ENGINE_SRC does not contain $ENGINE_MIN_COMMIT (PR #86): fetch the engine before trusting the binary"
            fi
        fi
    else
        warn "no executable at $SERVER_BIN; build the host with: cd $ENGINE_SRC && make cuda-spark"; fatal=1
    fi

    # One resident model. The engine takes a single global lock by design, and this
    # box already answers on other ports; the probe catches the case the lock does
    # not, an owner that is still loading.
    if command -v curl >/dev/null 2>&1; then
        local busy=() p
        for p in 8000 8001 8002 8003 8004 8005 8006 8007 8008; do
            curl -fsS --max-time 2 "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && busy+=("$p")
        done
        [ "${#busy[@]}" -gt 0 ] && warn "these ports already serve a model: ${busy[*]} - the engine's lock will refuse this start"
        if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            warn "port $PORT already answers /v1/models; pick another with GLM53UNC_PORT="; fatal=1
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
    say "install root:  $MODEL_ROOT"
    say "release dir:   $MODEL_DIR"
    say "runtime:       $SERVER_BIN"
    say "serve:         $HOST:$PORT, model id $MODEL_ID, watchdog below ${KILL_MB} MB"
    say ""
    say "artifacts"
    say "  revision     $REPO @ ${REVISION:-main}"
    say "  parts        $GGUF_PARTS GGUF files, $(human $PAYLOAD_BYTES)   $([ "$(parts_present)" -eq "$GGUF_PARTS" ] && echo present || echo "$(parts_present) on disk")"
    say "  tables       files.json, tensor-manifest.json, tokenizer*, $TEMPLATE"
    say "  predictor    the 4 embedded MTP parts (no --mtp path: they ship inside the release)"
    say "  not loaded   GLM-5.3-Flash-vision-01-of-01.gguf ships in the release; this script does not pass --vision"
    say ""
    say "the command start would run:"
    say "  $SERVER_BIN --cuda -m $MODEL_DIR \\"
    say "      --mtp-mode $MTP_MODE --mtp-draft $DRAFT --host $HOST --port $PORT \\"
    say "      --model-id $MODEL_ID --no-update-check${CTX:+ -c $CTX}${MAX_SEQS:+ --max-seqs $MAX_SEQS}"
    [ "$MTP_MODE" != "off" ] && say "  with DS4_CONT_MTP_MODE=2 DS4_CONT_MTP_DEPTH=$DRAFT in the environment (see the header)"
    say "  (no -c unless GLM53UNC_CTX is set: the family defaults to the native 1048576)"
    say ""
    say "still to fetch: $(human $([ "$(parts_present)" -eq "$GGUF_PARTS" ] && echo 0 || echo $PAYLOAD_BYTES)); free on target $(df_target "$MODEL_ROOT"): $(human "$(df_avail "$MODEL_ROOT")")"
    say "ports 8000, 8003, 8005, 8006, 8007 and 8008 answer for the other launchers in this home"
}

# --------------------------------------------------------------------------
# download, verify, check
# --------------------------------------------------------------------------
cmd_download() {
    mkdir -p "$MODEL_DIR"
    if release_ok; then
        assert_uncensored
        say "already downloaded: $(parts_present) parts in $MODEL_DIR"
        return 0
    fi
    ensure_hf
    local free; free=$(df_avail "$MODEL_ROOT")
    [ -z "$free" ] || [ "$free" -gt $((PAYLOAD_BYTES + 5000000000)) ] \
        || die "$MODEL_ROOT has $(human ${free:-0}) free, needs $(human $((PAYLOAD_BYTES + 5000000000)))"
    say "fetching $REPO${REVISION:+ @ $REVISION} into $MODEL_DIR"
    say "an interrupted run resumes from where it stopped; just run it again"
    hf_download || die "download failed; run the same command again to resume"
    release_ok || die "the fetch finished but the release is incomplete: $(parts_present)/$GGUF_PARTS parts, $(parts_partial) partial"
    assert_uncensored
    say "done: $GGUF_PARTS parts, $(human "$(du -sb "$MODEL_DIR" | cut -f1)") on disk, markers say $REPO's release"
    say "next: bash $0 check, then bash $0 start"
}

cmd_verify() {
    release_ok || die "the release is incomplete in $MODEL_DIR; run download first"
    assert_uncensored
    need sha256sum
    say "hashing $(human "$(du -sb "$MODEL_DIR" | cut -f1)"); this reads every byte"
    ( cd "$MODEL_DIR" && sha256sum -c SHA256SUMS ) || die "a payload hash does not match SHA256SUMS"
    say "verify: every payload hash matches SHA256SUMS"
}

# --check-config prints the plan as JSON and opens no weights, so this is the
# cheapest way to see the family, the context and the memory quote before a load.
cmd_check() {
    release_ok || die "the release is incomplete in $MODEL_DIR; run download first"
    assert_uncensored
    [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN (build it: cd $ENGINE_SRC && make cuda-spark)"
    say "release: $(( $(parts_present) ))/$GGUF_PARTS parts, $(human "$(du -sb "$MODEL_DIR" | cut -f1)") on disk, markers say uncensored"
    local args=(--check-config -m "$MODEL_DIR")
    [ -n "$CTX" ] && args+=(-c "$CTX")
    [ "$MTP_MODE" = "off" ] || args+=(--mtp-mode "$MTP_MODE" --mtp-draft "$DRAFT")
    "$SERVER_BIN" "${args[@]}"
}

# --------------------------------------------------------------------------
# start, stop, status, logs, test
# --------------------------------------------------------------------------
# From scripts/start-glm53-flash-vq.sh: the base release leaves 93 percent of what
# the engine sees available, so the margin is thin enough to warrant this. Two
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
    [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN (build it: cd $ENGINE_SRC && make cuda-spark)"
    need curl
    if alive; then
        die "already running (pid $(cat "$PID_FILE")); stop it first with $0 stop"
    fi

    local owners
    owners=$(other_owners)
    if [ -n "$owners" ] && [ "${GLM53UNC_ALLOW_SHARED:-0}" != "1" ]; then
        printf 'glm53unc: another model owner is running:\n%s\n' "$owners" >&2
        die "stop it first, or set GLM53UNC_ALLOW_SHARED=1 to overrule (one ds4 model fits on this box)"
    fi

    local a; a=$(avail_mb)
    [ "$a" -ge "$START_FLOOR_MB" ] \
        || die "only $a MB of memory available, need $START_FLOOR_MB: stop another model first"

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
        say "the host hashes 77 GB and maps the weights before it listens: about a minute"
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
    say "loading 77 GB and verifying every hash; the port binds only after that"

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
    say "ready: $MODEL_ID on $HOST:$PORT/v1"
    say "  local   http://127.0.0.1:$PORT/v1"
    say "  lan     http://${ip:-<host>}:$PORT/v1"
    say "  context ${CTX:-native 1048576}   banks ${MAX_SEQS:-auto (2)}   mtp $MTP_MODE/$DRAFT"
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
    say "markers:     $(release_identity) $([ -f "$MODEL_DIR/$TEMPLATE" ] && echo "($TEMPLATE $(sha_file "$MODEL_DIR/$TEMPLATE" | cut -c1-12), $MARKER $(fsize "$MODEL_DIR/$MARKER") bytes)")"
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
            printf '%s\n' "$out" | sed -n 's/.*"content":"\([^"]\{0,90\}\)".*/  answer: \1/p' | head -1;;
        *) warn "the request failed: $(printf '%s' "$out" | head -c 300)"; return 1;;
    esac
}

usage() { sed -n '2,74p' "$0" | sed 's/^# \{0,1\}//'; }

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
