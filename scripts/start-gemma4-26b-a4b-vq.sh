#!/usr/bin/env bash
# Gemma-4 26B A4B VQ: the OpenAI-compatible host on the DGX Spark, GPU only.
#
#   artifact   Jlfn/gemma-4-26b-a4b-vq-gguf, a GVQ1 release DIRECTORY of three
#              GGUF parts plus the tables: gemma4-vq8-trunk.gguf (8.59 GB,
#              R13+8 experts), gemma4-vq8-embedding-BF16.gguf (1.48 GB) and
#              gemma4-vq8-mtp-Q4_K.gguf (0.62 GB, only with --mtp). Head
#              5d123d82de071d040116c0950700a63afb10f07d, 17 files.
#   engine     ds4-dfm-rs, family gemma4_vq, merged upstream main (PR #141,
#              502019da) or later. Built with make cuda-spark (sm_121) on the GB10.
#   lane       GPU only by default: DS4_CUDA_COPY_MODEL=1 copies the trunk to
#              the device and every layer runs there. The CPU-MoE, host-slot
#              and SSD-stream switches exist for the 12 GiB card and are
#              selected explicitly (--cpu-moe, --moe-host, --ssd,
#              --resident 0); the launcher always writes every lane variable,
#              so an inherited environment cannot change the shape silently.
#
# usage:
#   bash start-gemma4-26b-a4b-vq.sh preflight     host, gpu, port, memory, release
#   bash start-gemma4-26b-a4b-vq.sh verify        sizes, digests, the record
#   bash start-gemma4-26b-a4b-vq.sh build         the engine, sm_121
#   bash start-gemma4-26b-a4b-vq.sh check         the engine's own --check-config plan
#   bash start-gemma4-26b-a4b-vq.sh choose        the shape it would pick, and the memory it sees
#   bash start-gemma4-26b-a4b-vq.sh start         0.0.0.0:8013
#   bash start-gemma4-26b-a4b-vq.sh stop | status | logs | test | plan | bench
#
# SHAPE IS CHOSEN, NOT ASSUMED. --kv, --max-seqs and --ctx default to auto:
# the launcher asks the engine what memory it can see (--check-config opens no
# weights), prices each candidate and takes the widest one that still leaves
# RESERVE_MB (default 1024, the last-gigabyte rule used on the 4070 SUPER)
# unallocated. Preference order inside one context is BF16 rows before E4M3 and
# two banks before one, because BF16 prefills faster and a second bank admits a
# second caller. Pass any of the three explicitly to pin it, and a pinned plan
# that would eat the reserve is refused rather than started.
#
# TWO BANKS, ONE MODEL. The family declares the persistent bank lane with
# qualified bank support (crates/ds4-core/src/serving.rs:987-990): each bank
# owns its own KV caches of both layouts over graph 0's scratch, and every
# cache row is position-indexed, so a bank's prefix is exact and needs no
# checkpoint slots. The script passes -c 262144 and --max-seqs 2, which keeps
# two conversations resident and serving at once. The engine still takes a
# single model lock: this is one model with two banks, not two models.
#
# BANKED DECODE IS SHARED DECODE. The GPU is one, so two concurrent streams
# split the decode rate instead of doubling it (the GLM sibling measured 24.2
# tok/s on one stream and 12.1 on each of two). Prefill of a second bank does
# overlap the first bank's decode.
#
# MTP IS SERIAL ONLY. The drafter executes on the serial lane; the bank driver
# has no draft/verify round for this family (serving.rs:999-1004). The script
# refuses --mtp with more than one bank instead of silently dropping drafts.
#
# THE MEASURED REFERENCE (GB10, ds4-bench, 13,447 tokens): 1,086.7 prefill and
# 33.96 gen tok/s at span 128, 35.72 gen tok/s at span 64; the 2K-64K sweep is
# speed-bench/gb10_spark_gemma4_*.csv in the engine repository.
#
# OPTIONS (each is also an environment variable of the same name)
#   --dir DIR           the release directory (default /mnt/models/gemma-4-26b-a4b-vq-gguf)
#   --port N            listen port (default 8013; the other owners are on
#                       8000-8012 per the dgx-recipe launchers)
#   --host ADDR         listen address (default 0.0.0.0)
#   --ctx N             context (default 262144; the 4070 lane uses 16384)
#   --engine-src DIR    the ds4-dfm-rs checkout (default $HOME/ds4-gemma4)
#   --engine-ref REF    commit to build (default: the checkout's HEAD)
#   --server-bin PATH   the binary to run (default <engine-src>/ds4-server)
#   --model-id ID       served model id (default gemma-4-26b-a4b-vq)
#   --kv MODE           bf16 (default) or fp8. E4M3 stores each full layer as
#                       one combined row per token (2,056 B instead of 4,096),
#                       the five full layers' cache halves, and the sliding
#                       rings stay bf16. The rows have run on the tensor-core
#                       kernels since 6dfe55cd, so the cost is modest: at
#                       32,768 tokens the card measures 746.6 against 804.6
#                       prefill tok/s and 19.27 against 18.71 decode. The 9.6x
#                       and 61x figures that circulated belong to the scalar
#                       fallback (DS4_GEMMA4_ATTN_MMA=0), not to this lane.
#                       Measured here at ctx 262144 with two banks: per bank
#                       5.59 -> 3.09 GiB, total 25.37 -> 20.38 GiB.
#   --mtp PATH          the MTP part path; without it the drafter stays off
#   --mtp-draft N       drafts per verify (default 2)
#   --cpu-moe [0|1]     host routed experts (default 0; the 12 GiB card's lane)
#   --cpu-moe-threads N the host expert pool (default 20)
#   --moe-host [0|1]    expert slots from host pages instead of the artifact
#   --moe-miss POLICY   evict (default) or cpu
#   --ssd [0|1]         stream the experts from the release (default 0)
#   --ssd-cache SIZE    expert cache for the stream (default 4GB)
#   --resident [0|1]    1 (default) copies the weights to the device, 0 maps them
#   --prefill N         prefill rows per forward (default 2048 on the Spark, 512 on the 4070)
#   --split-span N      split-K span (default: the binary's own, 64)
#   --max-seqs N        resident banks, i.e. concurrent conversations (default 2;
#                       the bank lane is qualified, 1 falls back to serial)
#   --kv-disk-dir DIR   persistent context checkpoints on disk (the disk-KV
#                       store; Gemma-4 is disk-qualified). The KV itself runs
#                       in device memory; there is no separate host-RAM tier.
#   --kv-disk-space S   disk budget, e.g. 32G (default 32G when a dir is set)
#   --no-watchdog       do not start the host-memory watchdog
#   --no-build          install: skip build
#   --no-start          install: stop before start
#   --allow-shared      start even if another ds4-server is running

set -Eeuo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
REPO="${REPO:-Jlfn/gemma-4-26b-a4b-vq-gguf}"
PIN=5d123d82de071d040116c0950700a63afb10f07d
REVISION="${REVISION:-$PIN}"

MODEL_DIR="${MODEL_DIR:-/mnt/models/gemma-4-26b-a4b-vq-gguf}"
ENGINE_SRC="${ENGINE_SRC:-$HOME/ds4-dfm-rs}"
ENGINE_REF="${ENGINE_REF:-}"
SERVER_BIN="${SERVER_BIN:-}"
SERVER_BIN_EXPLICIT=0
[ -n "$SERVER_BIN" ] && SERVER_BIN_EXPLICIT=1
HOST_ADDR="${HOST_ADDR:-0.0.0.0}"
PORT="${PORT:-8013}"
CTX="${CTX:-auto}"
MODEL_ID="${MODEL_ID:-gemma-4-26b-a4b-vq}"
# auto picks the rows and the width from the memory the engine can actually
# see; pin it with --kv bf16|fp8. See the shape section below.
KV="${KV:-auto}"
MTP="${MTP:-}"
MTP_DRAFT="${MTP_DRAFT:-2}"
# The lane, GPU only by default on the Spark. Every switch the launcher has is
# here so the same file can reproduce the 12 GiB card's shapes; the defaults
# are the Spark's.
CPU_MOE="${CPU_MOE:-0}"
CPU_MOE_THREADS="${CPU_MOE_THREADS:-20}"
MOE_HOST="${MOE_HOST:-0}"
MOE_MISS="${MOE_MISS:-evict}"
SSD="${SSD:-0}"
SSD_CACHE="${SSD_CACHE:-4GB}"
RESIDENT="${RESIDENT:-1}"
PREFILL="${PREFILL:-2048}"
SPLIT_SPAN="${SPLIT_SPAN:-}"
KV_DISK_DIR="${KV_DISK_DIR:-}"
KV_DISK_SPACE="${KV_DISK_SPACE:-32G}"
MAX_SEQS="${MAX_SEQS:-auto}"
# The last-gigabyte rule, as on the RTX 4070 SUPER: whatever shape is chosen,
# the plan must still leave this much unallocated. It is checked against the
# engine's own quote before a start, not against a round number of ours.
RESERVE_MB="${RESERVE_MB:-1024}"
WATCHDOG="${WATCHDOG:-1}"
NO_BUILD="${NO_BUILD:-0}"
NO_START="${NO_START:-0}"
ALLOW_SHARED="${ALLOW_SHARED:-0}"
HASH_JOBS="${HASH_JOBS:-8}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
START_FLOOR_MB="${START_FLOOR_MB:-40000}"
RUN_FLOOR_MB="${RUN_FLOOR_MB:-8000}"
KILL_MB="${KILL_MB:-6000}"

# Part names, sizes and digests come from the release directory's own
# release.json at run time: a locally assembled directory may name its parts
# differently from the published one (the Hub release adds the -BF16 and
# -Q4_K suffixes), and the loader reads whatever release.json declares.
EMBED=""
MTPPART=""
TRUNK=""
PARTS=0
PAYLOAD_BYTES=0
TRUNK_BYTES=0
load_release() {
    [ -s "$MODEL_DIR/release.json" ] || return 1
    PARTS=$(jq '.parts|length' "$MODEL_DIR/release.json")
    PAYLOAD_BYTES=$(jq '[.parts[].bytes]|add // 0' "$MODEL_DIR/release.json")
    TRUNK=$(jq -r '.parts[]|select(.path|test("trunk"))|.path' "$MODEL_DIR/release.json" | head -1)
    TRUNK_BYTES=$(jq -r --arg p "$TRUNK" '.parts[]|select(.path==$p)|.bytes' "$MODEL_DIR/release.json" | head -1)
    EMBED=$(jq -r '.parts[]|select(.path|test("embed"))|.path' "$MODEL_DIR/release.json" | head -1)
    MTPPART=$(jq -r '.parts[]|select(.path|test("mtp"))|.path' "$MODEL_DIR/release.json" | head -1)
    return 0
}
# The tables the loader reads. The published release carries more
# (tokenizer.json, the recipe and the license); a locally assembled directory
# may not, and the engine opens both.
TABLES=(release.json files.json config.json)

PIDF="$MODEL_DIR/gemma4.pid"
SERVER_LOG="$MODEL_DIR/gemma4.log"
WD_PIDF="$MODEL_DIR/gemma4-watchdog.pid"
WD_LOG="$MODEL_DIR/gemma4-watchdog.log"
RECORD="$MODEL_DIR/gemma4-verify.record"
TMP="${TMPDIR:-/tmp}/gemma4.$$"

# The launcher's fast set (run-gemma4.sh). The chunk is NOT here: lane_env
# sets DS4_GEMMA4_PREFILL_CHUNK from --prefill, and a second assignment after
# it would silently override the switch.
FAST_ENVS="DS4_GEMMA4_ATTN_FAST=1 DS4_GEMMA4_ATTN_PREFILL=1 \
DS4_GEMMA4_VQ_PREFILL=1 DS4_GEMMA4_Q8_BATCH=1 DS4_GEMMA4_FUSE_ROPE=1 \
DS4_GEMMA4_ROUTER_PAR=1 DS4_GEMMA4_HEAD_GEMV=1 DS4_GEMMA4_ROUTER_DOT=1 \
DS4_GEMMA4_ATTN_MMA=1 DS4_GEMMA4_ATTN_SPLIT=1 DS4_GEMMA4_ATTN_SPLIT_MMA=1 \
DS4_GEMMA4_MOE_PERSIST=1"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
say()  { printf 'gemma4: %s\n' "$*"; }
warn() { printf 'gemma4: warning: %s\n' "$*" >&2; }
die()  { printf 'gemma4: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
total_mb() { awk '/MemTotal/{print int($2/1024)}' /proc/meminfo; }
fsize()    { stat -L -c %s "$1" 2>/dev/null || echo 0; }
fmtime()   { stat -L -c %Y "$1" 2>/dev/null || echo 0; }
human()    { awk -v b="$1" 'BEGIN{if (b>=1000000000) printf "%.1f GB", b/1000000000; else printf "%.0f MB", b/1000000}'; }
gib()      { awk -v b="${1:-0}" 'BEGIN{printf "%.2f GiB", b/1073741824}'; }

# --------------------------------------------------------------------------
# Shape selection: how much memory is there, and what fits with the reserve left
# --------------------------------------------------------------------------
# The engine prices one bank's KV cache as
#   5 full layers * ctx * (4096 B BF16 | 2056 B E4M3)
# + 25 sliding    * (1024 window + max(chunk, 7) rows) * 8192 B
# and that reproduces its own quote to 0.04%: 5,997,854,720 B measured against
# 5,997,854,720 B computed at ctx 262144 / chunk 2048 / BF16, and 3,323,985,920
# against 3,322,787,840 B on E4M3 rows (docs/gemma4-vq.md:68-72).
kv_bank_bytes() {   # $1 ctx, $2 bf16|fp8, $3 chunk
    local row=4096 chunk=$3
    [ "$2" = "fp8" ] && row=2056
    [ "$chunk" -lt 7 ] && chunk=7
    awk -v c="$1" -v r="$row" -v ch="$chunk" 'BEGIN{printf "%.0f", 5*c*r + 25*(1024+ch)*8192}'
}

# One --check-config at the smallest shape (about 15 s, opens no weights) gives
# the artifact's fixed costs and the memory the engine can see right now, which
# is the honest answer to "how much is there": on a GB10 the pool is unified and
# nvidia-smi reports N/A, so the engine's own view is the only one that counts.
probe_facts() {
    [ -x "$SERVER_BIN" ] || return 1
    env DS4_CUDA_COPY_MODEL=1 DS4_GEMMA4_RESIDENT=1 DS4_GEMMA4_FULL_KV=bf16 \
        DS4_GEMMA4_PREFILL_CHUNK="$PREFILL" $FAST_ENVS \
        "$SERVER_BIN" -m "$MODEL_DIR" --cuda -c 8192 --max-seqs 1 \
        --host 127.0.0.1 --port "$PORT" --model-id "$MODEL_ID" --check-config \
        >"$TMP/probe.json" 2>"$TMP/probe.report" || return 1
    P_WEIGHTS=$(jq -r '.quote.shared_weights' "$TMP/probe.json" 2>/dev/null)
    P_SCRATCH=$(jq -r '.quote.scratch' "$TMP/probe.json" 2>/dev/null)
    P_FLOOR=$(jq -r '.quote.floor' "$TMP/probe.json" 2>/dev/null)
    P_AVAILABLE=$(jq -r '.quote.available' "$TMP/probe.json" 2>/dev/null)
    case "$P_WEIGHTS$P_SCRATCH$P_FLOOR$P_AVAILABLE" in
        *null*|"") return 1;;
    esac
    return 0
}

plan_bytes() {  # $1 ctx, $2 kv, $3 banks
    printf '%s' "$(( P_WEIGHTS + $3 * $(kv_bank_bytes "$1" "$2" "$PREFILL") + P_SCRATCH + P_FLOOR ))"
}

# Largest shape that still leaves RESERVE_MB unallocated. Within one context the
# order is BF16 before E4M3 (prefill is faster), and two banks before one.
select_shape() {
    probe_facts || return 1
    local ctx banks kv need spare
    for ctx in 262144 196608 131072 98304 65536 49152 32768 16384 8192; do
        for banks in 2 1; do
            for kv in bf16 fp8; do
                if [ "${EXPLICIT_CTX:-0}" = 1 ] && [ "$ctx" != "$CTX" ]; then { continue; }; fi
                if [ "${EXPLICIT_SEQS:-0}" = 1 ] && [ "$banks" != "$MAX_SEQS" ]; then { continue; }; fi
                if [ "${EXPLICIT_KV:-0}" = 1 ] && [ "$kv" != "$KV" ]; then { continue; }; fi
                need=$(plan_bytes "$ctx" "$kv" "$banks")
                spare=$(( P_AVAILABLE - need ))
                if [ "$spare" -ge $(( RESERVE_MB * 1024 * 1024 )) ]; then
                    CTX=$ctx; KV=$kv; MAX_SEQS=$banks
                    CHOSEN_NEED=$need; CHOSEN_SPARE=$spare
                    return 0
                fi
            done
        done
    done
    return 1
}

resolve_shape() {
    if [ "$CTX" != "auto" ] && [ "$KV" != "auto" ] && [ "$MAX_SEQS" != "auto" ]; then
        # fully pinned: still refuse a plan that eats the last gigabyte
        probe_facts || return 0
        local need spare
        need=$(plan_bytes "$CTX" "$KV" "$MAX_SEQS")
        spare=$(( P_AVAILABLE - need ))
        if [ "$spare" -lt $(( RESERVE_MB * 1024 * 1024 )) ]; then
            die "$KV rows, $MAX_SEQS bank(s), ctx $CTX need $(gib "$need") of $(gib "$P_AVAILABLE") and would leave $(gib "$spare"), under the ${RESERVE_MB} MiB reserve"
        fi
        say "shape:      $KV rows, $MAX_SEQS bank(s), ctx $CTX; $(gib "$need") planned, $(gib "$spare") left"
        return 0
    fi
    if select_shape; then
        say "shape:      $KV rows, $MAX_SEQS bank(s), ctx $CTX (keeps ${RESERVE_MB} MiB free)"
        say "memory:     $(gib "$CHOSEN_NEED") planned of $(gib "$P_AVAILABLE") available, $(gib "$CHOSEN_SPARE") left"
        return 0
    fi
    warn "could not price a shape (no engine binary or an unreadable release); falling back to bf16, 2 banks, ctx 262144"
    if [ "$CTX" = "auto" ]; then { CTX=262144; }; fi
    if [ "$KV" = "auto" ]; then { KV=bf16; }; fi
    if [ "$MAX_SEQS" = "auto" ]; then { MAX_SEQS=2; }; fi
    return 0
}

sha_file() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
alive()    { [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
cleanup()  { rm -rf "$TMP"; }
trap cleanup EXIT
mkdir -p "$MODEL_DIR"
TMP=$(mktemp -d)

# The lane as environment assignments, built from the switches above. Every
# value is written explicitly, so an inherited variable cannot change the lane
# behind the operator's back.
lane_env() {
    printf '%s\n' \
        "DS4_GEMMA4_CPU_MOE=$CPU_MOE" \
        "DS4_GEMMA4_CPU_MOE_THREADS=$CPU_MOE_THREADS" \
        "DS4_GEMMA4_MOE_HOST=$MOE_HOST" \
        "DS4_GEMMA4_MOE_MISS=$MOE_MISS" \
        "DS4_GEMMA4_RESIDENT=$RESIDENT" \
        "DS4_GEMMA4_PREFILL_CHUNK=$PREFILL" \
        "DS4_GEMMA4_FULL_KV=$KV"
    [ -n "$SPLIT_SPAN" ] && printf 'DS4_GEMMA4_ATTN_SPLIT_SPAN=%s\n' "$SPLIT_SPAN"
}

lane_name() {
    if [ "$SSD" = "1" ]; then printf 'SSD stream, %s expert cache' "$SSD_CACHE"
    elif [ "$CPU_MOE" = "1" ]; then printf 'host routed experts (CPU MoE, %s threads)' "$CPU_MOE_THREADS"
    elif [ "$RESIDENT" = "0" ]; then printf 'host-mapped weights'
    else printf 'GPU only, weights copied to the device'
    fi
}

# The GPU-only promise, checked instead of assumed: an inherited host-lane
# variable is overridden by lane_env, and the script says so.
refuse_host_lanes() {
    local v
    for v in DS4_GEMMA4_CPU_MOE DS4_GEMMA4_MOE_HOST DS4_GEMMA4_SSD DS4_GEMMA4_RESIDENT; do
        if [ -n "${!v:-}" ]; then
            warn "$v=${!v} is set in the environment; the launcher's own switches win (use --cpu-moe, --ssd, --resident)"
        fi
    done
}

# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------
cmd_preflight() {
    need curl; need sha256sum; need jq
    say "host:        $(uname -m) $(uname -r), $(total_mb) MB total, $(avail_mb) MB available"
    command -v nvidia-smi >/dev/null 2>&1 \
        && say "gpu:         $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)" \
        || warn "nvidia-smi is not on PATH"
    say "release:     $MODEL_DIR"
    if load_release; then
        say "parts:       $(parts_present)/$PARTS present at the published sizes ($(human "$PAYLOAD_BYTES") complete)"
        say "tables:      release.json present"
    else
        warn "no release.json in $MODEL_DIR (preflight only inspects)"
    fi
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | awk -v p=":$PORT\$" '$4 ~ p {print}' | grep -q . \
            && warn "port $PORT is already listening" || say "port:        $PORT is free"
    fi
    local owners; owners=$(pgrep -af "ds4-server" | grep -v "$$" || true)
    if [ -n "$owners" ]; then
        warn "another ds4-server is running (the engine takes a single global lock):"
        printf '%s\n' "$owners" | sed 's/^/             /' >&2
    fi
    [ -d "$ENGINE_SRC" ] && git -C "$ENGINE_SRC" rev-parse --show-toplevel >/dev/null 2>&1 \
        && say "engine src:  $ENGINE_SRC at $(git -C "$ENGINE_SRC" rev-parse --short HEAD 2>/dev/null)" \
        || warn "no git checkout at $ENGINE_SRC; build needs --engine-src"
    say "lane:        GPU only, ctx $CTX, model id $MODEL_ID, kv $KV"
}

# --------------------------------------------------------------------------
# verify: sizes, then digests, then a record start re-checks cheaply
# --------------------------------------------------------------------------
part_table() { jq -r '.parts[] | [.path,.sha256,(.bytes|tostring)] | @tsv' "$MODEL_DIR/release.json"; }

parts_present() {
    local n=0 name want bytes
    while IFS=$'\t' read -r name want bytes; do
        [ "$(fsize "$MODEL_DIR/$name")" = "$bytes" ] && n=$((n + 1))
    done < <(part_table)
    printf '%s' "$n"
}

undeclared_gguf() {
    local name
    while IFS= read -r name; do
        jq -e --arg p "$name" '.parts[]|select(.path==$p)' "$MODEL_DIR/release.json" >/dev/null 2>&1 \
            || printf '%s\n' "$name"
    done < <(find "$MODEL_DIR" -maxdepth 1 -name '*.gguf' -printf '%f\n' 2>/dev/null)
}

tables_ok() { local t; for t in "${TABLES[@]}"; do [ -s "$MODEL_DIR/$t" ] || return 1; done; }

cmd_verify() {
    need sha256sum; need jq
    [ -s "$MODEL_DIR/release.json" ] || die "$MODEL_DIR/release.json is missing; fetch the release first"
    load_release
    tables_ok || die "a table is missing from $MODEL_DIR (${TABLES[*]})"

    local name want bytes got i=0 bad=0
    local -a names=() shas=() sizes=()
    while IFS=$'\t' read -r name want bytes; do
        names+=("$name"); shas+=("$want"); sizes+=("$bytes")
    done < <(part_table)
    [ "${#names[@]}" = "$PARTS" ] || die "release.json declares ${#names[@]} parts, not $PARTS"

    for name in "${names[@]}"; do
        i=$((i + 1))
        got=$(fsize "$MODEL_DIR/$name")
        [ "$got" = "${sizes[$((i - 1))]}" ] \
            || die "  [$i/$PARTS] $name is $got bytes, not ${sizes[$((i - 1))]}"
    done
    say "sizes:       all $PARTS parts match release.json ($(human "$PAYLOAD_BYTES"))"

    local extra; extra=$(undeclared_gguf)
    [ -z "$extra" ] || die "undeclared .gguf in $MODEL_DIR (the loader rejects the directory):
$(printf '  %s\n' $extra)"

    say "hashing:     $PARTS parts ($(human "$PAYLOAD_BYTES")) with $HASH_JOBS workers"
    local out="$TMP/hash.out"; : >"$out"
    ( cd "$MODEL_DIR" && printf '%s\n' "${names[@]}" | xargs -r -P "$HASH_JOBS" -n 1 sha256sum ) >"$out" 2>&1 || true
    i=0
    for name in "${names[@]}"; do
        i=$((i + 1))
        want="${shas[$((i - 1))]}"
        got=$(awk -v p="$name" '$2==p{print $1}' "$out")
        [ "$got" = "$want" ] || { warn "  $name MISMATCH: ${got:-no digest}, published $want"; bad=$((bad + 1)); }
    done
    [ "$bad" = 0 ] || die "$bad part(s) do not match release.json"

    for name in "${TABLES[@]}"; do
        # files.json is the manifest the other digests are read from; it
        # cannot list its own digest, so the record keeps it and nothing
        # published is compared. Every other table must match.
        [ "$name" = "files.json" ] && continue
        want=$(jq -r --arg p "$name" '.files[]? | select(.path==$p) | .sha256' "$MODEL_DIR/files.json" 2>/dev/null | head -1)
        if [ -z "$want" ] && [ -f "$MODEL_DIR/SHA256SUMS" ]; then
            want=$(awk -v p="$name" '$2==p{print $1}' "$MODEL_DIR/SHA256SUMS" | head -1)
        fi
        [ -n "$want" ] || { warn "  $name is not listed in files.json or SHA256SUMS"; bad=$((bad + 1)); continue; }
        got=$(sha_file "$MODEL_DIR/$name")
        [ "$got" = "$want" ] || { warn "  $name MISMATCH: $got, published $want"; bad=$((bad + 1)); }
    done
    [ "$bad" = 0 ] || die "$bad digest(s) failed; the bytes on disk are not this release"

    {
        printf '# gemma4-26b-a4b-vq verify record\n'
        printf '# repo %s\n# revision %s\n# written %s\n' "$REPO" "$REVISION" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        for name in "${names[@]}" "${TABLES[@]}"; do
            printf '%s  %s  %s  %s\n' "$(sha_file "$MODEL_DIR/$name")" "$(fsize "$MODEL_DIR/$name")" \
                "$(fmtime "$MODEL_DIR/$name")" "$MODEL_DIR/$name"
        done
    } >"$RECORD"
    say "verified:    $PARTS parts and every published table digest match"
    say "record:      $RECORD"
}

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
# build / check
# --------------------------------------------------------------------------
cmd_build() {
    need git; need make
    [ -d "$ENGINE_SRC/.git" ] || die "$ENGINE_SRC is not a git checkout of ds4-dfm-rs"
    local ref="${ENGINE_REF:-HEAD}"
    say "engine:      $ENGINE_SRC at $ref"
    ( cd "$ENGINE_SRC" && git fetch origin --quiet || true )
    local short; short=$(git -C "$ENGINE_SRC" rev-parse --short "$ref")
    ( cd "$ENGINE_SRC" && PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH" \
        make -j"$(nproc)" ds4 ds4-server ds4-bench CUDA_ARCH=sm_121 ) || die "the build failed"
    [ -x "$SERVER_BIN" ] || die "the build finished but $SERVER_BIN is not executable"
    say "built:       $SERVER_BIN ($(fsize "$SERVER_BIN") bytes, $short)"
}

cmd_check() {
    [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN; run: build"
    refuse_host_lanes
    local json="$MODEL_DIR/check.json" report="$MODEL_DIR/check.report" rc=0
    local -a lane=()
    while IFS= read -r x; do [ -n "$x" ] && lane+=("$x"); done < <(lane_env)
    # shellcheck disable=SC2086
    env DS4_CUDA_COPY_MODEL=1 "${lane[@]}" $FAST_ENVS \
        "$SERVER_BIN" -m "$MODEL_DIR" --cuda -c "$CTX" --max-seqs "$MAX_SEQS" \
        --host 127.0.0.1 --port "$PORT" --model-id "$MODEL_ID" \
        ${KV_DISK_DIR:+--kv-disk-dir "$KV_DISK_DIR" --kv-disk-space "$KV_DISK_SPACE"} \
        $([ "$SSD" = "1" ] && printf -- '--ssd-streaming --ssd-streaming-cache-experts %s' "$SSD_CACHE") \
        --check-config >"$json" 2>"$report" || rc=$?
    sed 's/^/  /' "$report"
    [ "$rc" = 0 ] || { say "the engine refused the plan (exit $rc); JSON in $json"; return 1; }
    local family issues
    family=$(jq -r '.family' "$json"); issues=$(jq -r '[.issues[]|select(.level=="error")]|length' "$json")
    say "family:      $family"
    say "lane:        $(lane_name)"
    say "effective:   ctx=$(jq -r '.effective.ctx' "$json") max_seqs=$(jq -r '.effective.max_seqs' "$json") span=$(jq -r '.gemma4.attn_split_span // "?"' "$json") kv=$(jq -r '.gemma4.full_kv // "?"' "$json") cpu_moe=$(jq -r '.gemma4.cpu_moe // "?"' "$json")"
    [ "$family" = "gemma4_vq" ] || warn "the engine reports family '$family'"
    [ "$issues" = 0 ] || { jq -r '.issues[]|select(.level=="error")|"  error: \(.message) (\(.code))"' "$json"; return 1; }
    say "check ok:    the release opens and the plan fits"
}

# Print the shape the launcher would pick on this box right now, with the
# memory the engine can see and the reserve it keeps. Read-only: no weights.
cmd_choose() {
    if select_shape; then
        say "chosen:     $KV rows, $MAX_SEQS bank(s), ctx $CTX"
        say "memory:     $(gib "$CHOSEN_NEED") planned of $(gib "$P_AVAILABLE") available"
        say "reserve:    ${RESERVE_MB} MiB kept free ($(gib "$CHOSEN_SPARE") actually left)"
        return 0
    fi
    die "no shape fits with ${RESERVE_MB} MiB kept free"
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
            printf '%s MemAvailable %s MB below %s MB twice; stopping the server\n' \
                "$(date)" "$a" "$2" >>"$3"
            kill "$pid" 2>/dev/null || true
            return 0
        fi
        sleep 5
    done
}

cmd_start() {
    refuse_host_lanes
    [ -s "$MODEL_DIR/release.json" ] || die "no release.json in $MODEL_DIR"
    load_release
    [ "$(parts_present)" = "$PARTS" ] || die "$(parts_present)/$PARTS parts at the published sizes in $MODEL_DIR"
    tables_ok || die "a table is missing from $MODEL_DIR"
    record_matches || die "no verification record matches the bytes on disk ($RECORD); run: verify"
    [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN; run: build"
    need curl
    alive && die "already running (pid $(cat "$PIDF")); stop it first"
    local owners; owners=$(pgrep -af "ds4-server" | grep -v "$$" || true)
    if [ -n "$owners" ] && [ "$ALLOW_SHARED" != 1 ]; then
        printf 'gemma4: another ds4 model is running:\n%s\n' "$owners" >&2
        die "stop it first (one model lock), or set ALLOW_SHARED=1"
    fi
    local a; a=$(avail_mb)
    [ "$a" -ge "$START_FLOOR_MB" ] || die "only $a MB available, need $START_FLOOR_MB"
    local -a args=(-m "$MODEL_DIR" --cuda -c "$CTX" --max-seqs "$MAX_SEQS"
                   --host "$HOST_ADDR" --port "$PORT" --model-id "$MODEL_ID"
                   --mem-floor-gb 2 --prefix-reuse auto)
    [ -n "$KV_DISK_DIR" ] && args+=(--kv-disk-dir "$KV_DISK_DIR" --kv-disk-space "$KV_DISK_SPACE")
    [ "$SSD" = "1" ] && args+=(--ssd-streaming --ssd-streaming-cache-experts "$SSD_CACHE")
    [ -n "$MTP" ] && args+=(--mtp "$MTP" --mtp-draft "$MTP_DRAFT" --mtp-mode on)
    say "starting ds4-server on $HOST_ADDR:$PORT (ctx $CTX)"
    say "  lane:    $(lane_name)"
    say "  release: $MODEL_DIR ($(human "$PAYLOAD_BYTES"); trunk $(human "$TRUNK_BYTES") copied to the device)"
    say "  log:     $SERVER_LOG"
    local -a lane=()
    while IFS= read -r x; do [ -n "$x" ] && lane+=("$x"); done < <(lane_env)
    # shellcheck disable=SC2086
    env DS4_LOCK_FILE="$MODEL_DIR/gemma4.lock" DS4_CUDA_COPY_MODEL=1 \
        "${lane[@]}" $FAST_ENVS \
        setsid nohup "$SERVER_BIN" "${args[@]}" >"$SERVER_LOG" 2>&1 </dev/null &
    echo $! >"$PIDF"
    if [ "$WATCHDOG" = 1 ]; then
        export -f watchdog avail_mb
        nohup bash -c "watchdog $(cat "$PIDF") $KILL_MB '$WD_LOG'" >/dev/null 2>&1 </dev/null &
        echo $! >"$WD_PIDF"
        say "  watchdog pid $(cat "$WD_PIDF")"
    fi
    local waited=0
    say "waiting for the endpoint (up to ${READY_TIMEOUT}s)"
    while [ "$waited" -lt "$READY_TIMEOUT" ]; do
        curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
        if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then
            tail -n 20 "$SERVER_LOG" | sed 's/^/  /' >&2
            die "the server exited while loading (full log: $SERVER_LOG)"
        fi
        sleep 5; waited=$((waited + 5))
        [ $((waited % 60)) -eq 0 ] && say "  still loading (${waited}s)"
    done
    [ "$waited" -lt "$READY_TIMEOUT" ] || die "no answer on port $PORT within ${READY_TIMEOUT}s"
    a=$(avail_mb)
    [ "$a" -ge "$RUN_FLOOR_MB" ] || { cmd_stop >/dev/null 2>&1 || true; die "only $a MB left after loading"; }
    say "loaded in ${waited}s; ${a} MB of memory left"
    cmd_test || warn "the smoke request failed"
    local ip; ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    say "ready: http://127.0.0.1:$PORT/v1   (lan http://${ip:-<host>}:$PORT/v1, model id $MODEL_ID)"
}

cmd_stop() {
    [ -s "$WD_PIDF" ] && kill "$(cat "$WD_PIDF")" 2>/dev/null || true
    if alive; then
        local pid; pid=$(cat "$PIDF")
        say "stopping ds4-server (pid $pid)"
        kill "$pid" 2>/dev/null || true
        local i=0
        while [ "$i" -lt 30 ] && kill -0 "$pid" 2>/dev/null; do sleep 1; i=$((i + 1)); done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    else
        say "ds4-server: not running"
    fi
    rm -f "$PIDF" "$WD_PIDF"
}

cmd_status() {
    load_release || true      # PARTS and the record check read this
    if alive; then
        say "ds4-server:  running (pid $(cat "$PIDF"))"
        ps -o pid=,etime=,rss= -p "$(cat "$PIDF")" 2>/dev/null | sed 's/^/  /'
    else
        say "ds4-server:  not running"
    fi
    say "memory:      $(avail_mb) MB available of $(total_mb) MB"
    say "release:     $(parts_present)/$PARTS parts in $MODEL_DIR"
    say "verified:    $(record_matches && echo 'yes - record matches the bytes' || echo 'no - run verify')"
    if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | head -c 300; then
        say ""; say "endpoint:    answering on port $PORT"
    else
        say "endpoint:    no answer on port $PORT"
    fi
    [ -f "$SERVER_LOG" ] && { say ""; say "log tail:"; tail -n 5 "$SERVER_LOG" | sed 's/^/  /'; }
}

cmd_logs() {
    say "server log: $SERVER_LOG"
    [ -f "$SERVER_LOG" ] && tail -n 40 "$SERVER_LOG" | sed 's/^/  /'
    say ""
    say "watchdog:   $WD_LOG"
    [ -f "$WD_LOG" ] && tail -n 10 "$WD_LOG" | sed 's/^/  /'
    return 0
}

cmd_test() {
    need curl
    local mid out
    mid=$(curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
          | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
    mid=${mid:-$MODEL_ID}
    say "smoke request against model id '$mid'"
    out=$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$mid\",\"temperature\":0,\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, what does a vector-quantized expert layer store?\"}]}")
    case "$out" in
        *'"choices"'*)
            printf '%s\n' "$out" | sed -n 's/.*"usage":{\(.*\)}.*/  usage: \1/p' | head -1
            printf '%s\n' "$out" | sed -n 's/.*"content":"\([^"]\{0,120\}\)".*/  answer: \1.../p' | head -1;;
        *) warn "the request failed: $(printf '%s' "$out" | head -c 400)"; return 1;;
    esac
}

cmd_plan() {
    load_release || true
    say "release:     $MODEL_DIR ($(parts_present)/$PARTS parts, $(human "$PAYLOAD_BYTES") complete)"
    say "engine:      $ENGINE_SRC at $(git -C "$ENGINE_SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    say "lane:        $(lane_name), one serial sequence, ctx $CTX, kv $KV, port $PORT"
    say "prefill:     $PREFILL rows per forward; span ${SPLIT_SPAN:-the binary default (64)}"
    say "model id:    $MODEL_ID"
    say "reference:   1,086.7 prefill / 33.96 gen tok/s at 13,447; 35.72 at the shipped span 64"
    say "next:        verify, build, check, start"
}

cmd_bench() {
    say "The engine's own sweep is the reference:"
    say "  speed-bench/gemma4_spark_sweep.sh <server> <release-dir> [port]"
    say "  G4_SWEEP_PORT_ARGS='--prefix-reuse off' makes every frontier a full prefill."
    say "Measured at 13,447: sequence of 64 greedy tokens; plain, no drafter."
    say "Stop the server first: one ds4 model is resident at a time."
}

cmd_install() {
    cmd_preflight
    cmd_verify
    [ "$NO_BUILD" = 1 ] || cmd_build
    cmd_check
    [ "$NO_START" = 1 ] || cmd_start
}

usage() { sed -n '2,/^set -Eeuo/{/^set -Eeuo/d;p}' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --dir)          MODEL_DIR="$2"; shift 2;;
        --port)         PORT="$2"; shift 2;;
        --host)         HOST_ADDR="$2"; shift 2;;
        --ctx)          CTX="$2"; EXPLICIT_CTX=1; shift 2;;
        --engine-src)   ENGINE_SRC="$2"; shift 2;;
        --engine-ref)   ENGINE_REF="$2"; shift 2;;
        --server-bin)   SERVER_BIN="$2"; SERVER_BIN_EXPLICIT=1; shift 2;;
        --model-id)     MODEL_ID="$2"; shift 2;;
        --kv)           KV="$2"; EXPLICIT_KV=1; shift 2;;
        --mtp)          MTP="$2"; shift 2;;
        --mtp-draft)    MTP_DRAFT="$2"; shift 2;;
        --cpu-moe)      if [ "${2:-}" = "0" ] || [ "${2:-}" = "1" ]; then CPU_MOE="$2"; shift 2; else CPU_MOE=1; shift; fi;;
        --cpu-moe-threads) CPU_MOE_THREADS="$2"; shift 2;;
        --moe-host)     MOE_HOST="$2"; shift 2;;
        --moe-miss)     MOE_MISS="$2"; shift 2;;
        --ssd)          if [ "${2:-}" = "0" ] || [ "${2:-}" = "1" ]; then SSD="$2"; shift 2; else SSD=1; shift; fi;;
        --ssd-cache)    SSD_CACHE="$2"; shift 2;;
        --resident)     RESIDENT="$2"; shift 2;;
        --prefill)      PREFILL="$2"; shift 2;;
        --split-span)   SPLIT_SPAN="$2"; shift 2;;
        --max-seqs)     MAX_SEQS="$2"; EXPLICIT_SEQS=1; shift 2;;
        --kv-disk-dir)  KV_DISK_DIR="$2"; shift 2;;
        --kv-disk-space) KV_DISK_SPACE="$2"; shift 2;;
        --no-watchdog)  WATCHDOG=0; shift;;
        --no-build)     NO_BUILD=1; shift;;
        --no-start)     NO_START=1; shift;;
        --allow-shared) ALLOW_SHARED=1; shift;;
        -h|--help)      usage; exit 0;;
        -*)             die "unknown option: $1 (try --help)";;
        *)              break;;
    esac
done

[ "$SERVER_BIN_EXPLICIT" = 1 ] || SERVER_BIN="$ENGINE_SRC/ds4-server"

# The drafter runs on the serial lane only (serving.rs:999-1004). Refuse the
# combination instead of starting a banked server whose drafts never fire.
if [ -n "$MTP" ] && [ "$MAX_SEQS" -gt 1 ] 2>/dev/null; then
    die "--mtp needs the serial lane; use --max-seqs 1 for drafting, or drop --mtp"
fi

# --dir may move the state files with the release.
PIDF="$MODEL_DIR/gemma4.pid"; SERVER_LOG="$MODEL_DIR/gemma4.log"
WD_PIDF="$MODEL_DIR/gemma4-watchdog.pid"; WD_LOG="$MODEL_DIR/gemma4-watchdog.log"
RECORD="$MODEL_DIR/gemma4-verify.record"

# Every action that builds a plan resolves auto first: the shape depends on the
# memory the engine can see at this moment, not on a compile-time default.
case "${1:-}" in
    check|start|install|plan|bench|test) resolve_shape;;
esac

case "${1:-}" in
    preflight) cmd_preflight;;
    verify)    cmd_verify;;
    build)     cmd_build;;
    check)     cmd_check;;
    choose)    cmd_choose;;
    start)     cmd_start;;
    stop)      cmd_stop;;
    status)    cmd_status;;
    logs)      cmd_logs;;
    test)      cmd_test;;
    plan)      cmd_plan;;
    bench)     cmd_bench;;
    install)   cmd_install;;
    ""|-h|--help) usage; exit 0;;
    *)         die "unknown action: $1 (try --help)";;
esac
