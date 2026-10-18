#!/usr/bin/env bash
# DeepSeek-V4.1-Flash-1017 UNCENSORED VQ: fetch the release, prove which bytes
# are on the box, assemble the base GGUF, and serve it on the DGX Spark.
#
#   artifact  Baekpica/DeepSeek-V4.1-Flash-1017-UNCENSORED-VQ-GGUF
#             pinned at 610a2d71400313f63228589bf4cea4182ad360e9 (main head,
#             committed 2026-10-18T14:43:29Z)
#             the first revision of the repository, so the pin and main agree
#   layout    40 byte-range parts -> one 113,556,639,424-byte base GGUF that
#             carries the model's three DSpark draft towers; a 971,066,016-byte
#             vision sidecar; the two 1017 Engram tables (2 x ~101.5 GB) in
#             deepseek-engram/; SHA256SUMS; RECIPE.md and SOURCE.md
#             payload 317,601,046,990 bytes (317.6 GB / 295.8 GiB)
#   engine    ds4-dfm-rs with DeepSeek V4.1 support (PR #80) and vision (PR #83),
#             read here at 34c46d1; built with make cuda-spark
#   source    dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8 at
#             c19ca5ca4eaeb79d29fdf08aea8327da95b1116a, which is
#             deepseek-ai/DeepSeek-V4.1-Flash-1017 at 1bae0e10 with dealignai's
#             weight edit in layers 10-36
#
# THIS SCRIPT'S SHAPE, AND WHERE IT CAME FROM
#   scripts/ds41flash_1017_install.sh in this repository is the lineage: same
#   actions, same "the base is read once, everything read at run time stays on the
#   fast device" split, same watchdog, same append-and-delete assembly. Three
#   things differ, and each is a property of this artifact: it ships the Engram
#   tables inside the repository (the 1017 packaging fetched them from DeepSeek),
#   it ships no domain sidecar at all, and the parts must be assembled on the
#   roomy volume rather than the fast one, because the assembled base is 113.6 GB
#   and only one part's worth of extra room is ever needed.
#
# WHERE EACH PIECE GOES, AND WHY
#   --gguf-dir (default /mnt/models/ds41-1017-unc): the 40 parts and the base they
#   assemble into. The base is read exactly once at load and then stays resident
#   in unified memory, so disk speed stops mattering after the first minutes.
#   --dir (default $HOME/models/ds41-1017-unc): the vision sidecar, the logs and
#   the pid files. The vision tower is read from disk when an image arrives, on
#   the first image after a cold start or an eviction.
#   --engram-dir (default: the first existing verified copy, else
#   <--dir>/deepseek-engram): the two n-gram tables, read in rows on the tokens
#   that hit an n-gram, which is why they belong on the fastest device.
#
# THE MISTAKES THIS SCRIPT EXISTS TO PREVENT
#   1. Reusing the V4.1 Engram tables. This release's tables are DeepSeek's 1017
#      files, byte for byte; the older V4.1 tables have the same names and sizes
#      and different contents, so an engine pointed at them loads without a word
#      and runs a mismatched model. Their digests are known and are named.
#   2. Reusing the older V4.1 vision sidecar. 532,623,100 of its 971,066,016 bytes
#      differ under the same name. Only the published digest is accepted.
#   3. Reusing another 1017 packaging's base. Baekpica's DeepSeek-V4.1-Flash-1017-VQ
#      base and YoungAi's are both 113,556,639,424 bytes with different digests
#      (9f3afe6a... and 0da4ae2d... against 2680747e... here). The parts look the
#      same and are not; the assembled digest is checked, and every part is checked
#      against the release's own SHA256SUMS as it is appended.
#   4. Mounting a domain sidecar. This artifact ships none, and YoungAi's V4.1
#      sidecars were solved against a different base. --zchain is never passed.
#
# WHAT HAS NOT BEEN MEASURED
#   Nothing in this recipe has been loaded or requested on the reference Spark.
#   The card states that serving on DGX Spark has not been measured for this
#   artifact, and the speed and fidelity figures it prints are V4.1 Flash's, for
#   the same recipe. They are labelled as such in the runbook.
#
# USAGE
#   bash install-ds41-1017-uncensored-vq.sh preflight
#   bash install-ds41-1017-uncensored-vq.sh download    # 317.6 GB, resumable
#   bash install-ds41-1017-uncensored-vq.sh assemble    # 40 parts -> the base
#   bash install-ds41-1017-uncensored-vq.sh verify      # SHA256SUMS + the four identity digests
#   bash install-ds41-1017-uncensored-vq.sh build       # check (or build) the engine
#   bash install-ds41-1017-uncensored-vq.sh start       # 0.0.0.0:8010
#   bash install-ds41-1017-uncensored-vq.sh install     # all of the above, then start
#   bash install-ds41-1017-uncensored-vq.sh stop | status | logs | test | plan
#
# OPTIONS (each is also an environment variable of the same name)
#   --dir DIR          fast-device root: vision sidecar, logs, pid files
#                      (default $HOME/models/ds41-1017-unc)
#   --gguf-dir DIR     where the parts land and the base is assembled
#                      (default /mnt/models/ds41-1017-unc)
#   --engram-dir DIR   the two Engram tables (default: a verified existing copy
#                      under $HOME/models, else <--dir>/deepseek-engram)
#   --vision PATH      the vision sidecar to mount (default: this release's copy
#                      under --dir, else a digest-proven copy found under
#                      $HOME/models, else <--dir>/<the sidecar's own name>)
#   --port N           listen port (default 8010, this artifact's port)
#   --host ADDR        listen address (default 0.0.0.0)
#   --mtp-mode MODE    off | auto | on, the embedded DSpark drafter (default auto)
#   --mem-budget-mb N  the engine's load budget (default 110000)
#   --engine-dir DIR   the ds4-dfm-rs tree (default $HOME/ds4-dfm-rs)
#   --server-bin PATH  the binary to run (default <--engine-dir>/ds4-server)
#   --hf-bin PATH      the Hugging Face CLI to use
#   --revision SHA     pin another revision of the repository
#   --watchdog 0|1     memory watchdog (default 1)
#   --no-vision        serve text only; do not mount the vision tower
#   --no-build         never run make, even when the binary is missing
#   --no-start         everything except starting the server
#   --allow-shared     overrule the single-owner probe before starting
#   -h, --help         this text
#
# Notes
#   - The repository is public: no token is required and none is ever written
#     into this file. HF_TOKEN only raises the CDN rate limit.
#   - The context is not a flag on this family. The GGUF metadata carries
#     1,048,576 and the engine takes it from there; -c/--ctx is never passed.
#   - Thinking is off by default for this weight recipe (under-weighted stop
#     decisions), and the card recommends temperature 1.0 with top_p 1.0 sent
#     explicitly in each request. This launcher sets no temperature: the smoke
#     test sends 0.
#   - start checks sizes; verify checks digests. Run verify at least once after a
#     fetch, and always after assemble.
#   - The vision sidecar is name-and-size identical to the older V4.1 one, so the
#     search for an existing copy hashes it before adopting it (971 MB, about a
#     second) and passes over a file of the published size holding V4.1 bytes.
#   - The two Engram tables are the only payload of this release that lives in a
#     subdirectory (deepseek-engram/). The HF CLI keeps the repository path under
#     --local-dir, so they arrive at <engram-dir>/deepseek-engram/. download and
#     verify move them up into <engram-dir> with a rename: nothing is re-fetched.
#   - assemble and verify print one line per part and say how long each read is
#     expected to take; their long loops are never silent.
#   - The engine takes a single global lock: only one ds4 model can be resident.

set -Eeuo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
REPO="${REPO:-Baekpica/DeepSeek-V4.1-Flash-1017-UNCENSORED-VQ-GGUF}"
PIN=610a2d71400313f63228589bf4cea4182ad360e9
REVISION="${REVISION:-$PIN}"

MODEL_DIR="${MODEL_DIR:-$HOME/models/ds41-1017-unc}"      # fast device
GGUF_DIR="${GGUF_DIR:-/mnt/models/ds41-1017-unc}"         # roomy volume
ENGRAM_DIR="${ENGRAM_DIR:-}"                              # resolved below if empty
VISION_PATH="${VISION_PATH:-}"                            # resolved below if empty
ENGINE_DIR="${ENGINE_DIR:-$HOME/ds4-dfm-rs}"
SERVER_BIN="${SERVER_BIN:-$ENGINE_DIR/ds4-server}"
HOST_ADDR="${HOST_ADDR:-0.0.0.0}"
# 8000 is the V4.1 port deployment, 8003 Qwen3.8, 8005 Bonsai, 8006 YoungAi V4.1,
# 8007 GLM-5.3 VQ, 8008 YoungAi 1017 and 8009 GLM-5.3 Uncensored VQ, so this one
# takes 8010.
PORT="${PORT:-8010}"
MTP_MODE="${MTP_MODE:-auto}"
BUDGET_MB="${BUDGET_MB:-110000}"
WATCHDOG="${WATCHDOG:-1}"
NO_VISION="${NO_VISION:-0}"
NO_BUILD="${NO_BUILD:-0}"
NO_START="${NO_START:-0}"
ALLOW_SHARED="${ALLOW_SHARED:-0}"
REUSE_ENGRAM="${REUSE_ENGRAM:-1}"
MAX_WORKERS="${MAX_WORKERS:-8}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
START_FLOOR_MB="${START_FLOOR_MB:-100000}"
RUN_FLOOR_MB="${RUN_FLOOR_MB:-6000}"
KILL_MB="${KILL_MB:-2500}"

GGUF="DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf"
VISION="DeepSeek-V4.1-Flash.vision.gguf"
ENGRAM_47="model-00047-of-00048.safetensors"
ENGRAM_48="model-00048-of-00048.safetensors"

PARTS=40
PART_BYTES_BIG=2838915986      # parts 01-24
PART_BYTES_SMALL=2838915985    # parts 25-40
BASE_BYTES=113556639424
VISION_BYTES=971066016
ENGRAM_47_BYTES=101535150936
ENGRAM_48_BYTES=101537926640
PAYLOAD_BYTES=317601046990     # the Hub's usedStorage for this repository

# The four identity digests, from the release's own SHA256SUMS. These are what
# separate this release from the other 1017 packagings, whose files carry the same
# names and sizes.
BASE_SHA=2680747eaec26d62c3f40c1a8fce5b931d210a709f71868bde5d68766bccf0a1
VISION_SHA=3a8b194a011e630e8e49a51b5f4e9821e48a9e5674e678d52c0302d6b164698c
ENGRAM_47_SHA=76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5
ENGRAM_48_SHA=9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd
# The older V4.1 tables: same names, same sizes, different bytes. Recognised so the
# refusal can name what it found instead of saying "unknown".
V41_ENGRAM_47_SHA=824db4881320407ac340736d14dcee5ecd748c27d0f5836b8127ecc2e3781b0f
V41_ENGRAM_48_SHA=976330f4954338e1ad8b508c32aa912032c7ad908959fd53c8307650fe4520ed
# The other 1017 packagings' bases, for the same reason.
OTHER_BASE_SHAS="9f3afe6a (Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF) 0da4ae2d (wenzhouwu/YoungAi-DeepSeek-V4.1-Flash-1017)"

PIDF="$MODEL_DIR/ds41unc.pid"
SERVER_LOG="$MODEL_DIR/ds41unc.log"
WD_PIDF="$MODEL_DIR/ds41unc-watchdog.pid"
WD_LOG="$MODEL_DIR/ds41unc-watchdog.log"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
say()  { printf 'ds41unc: %s\n' "$*"; }
warn() { printf 'ds41unc: warning: %s\n' "$*" >&2; }
die()  { printf 'ds41unc: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
total_mb() { awk '/MemTotal/{print int($2/1024)}' /proc/meminfo; }
fsize()    { stat -L -c %s "$1" 2>/dev/null || echo 0; }
df_avail() { df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' \n' || true; }
df_target(){ df --output=target "$1" 2>/dev/null | tail -1 || true; }
human()    { awk -v b="$1" 'BEGIN{printf "%.1f GB", b/1000000000}'; }
sha_file() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
alive()    { [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

part_name()  { printf '%s.part%s-of-40' "$GGUF" "$1"; }
# Counted from the directory rather than from a generated name list: `printf %02d`
# treats 08 and 09 as invalid octal, which silently lost two parts when the list
# was generated (found on 2026-10-18).
parts_present() { find "$GGUF_DIR" -maxdepth 1 -name "$GGUF.part*-of-40" 2>/dev/null | wc -l; }
parts_partial() { find "$GGUF_DIR" -maxdepth 1 -name '*.part*.incomplete' 2>/dev/null | wc -l; }

# Which 1017 packaging's Engram tables are in ENGRAM_DIR: 1017, v41, or unknown.
# HASHES 203 GB, so it is called only by verify, where that read is the point.
# Every other caller uses engram_ok() (sizes) plus the verify marker.
engram_identity() {
    local a b
    a=$(sha_file "$ENGRAM_DIR/$ENGRAM_47"); b=$(sha_file "$ENGRAM_DIR/$ENGRAM_48")
    if [ "$a" = "$ENGRAM_47_SHA" ] && [ "$b" = "$ENGRAM_48_SHA" ]; then printf '1017'
    elif [ "$a" = "$V41_ENGRAM_47_SHA" ] && [ "$b" = "$V41_ENGRAM_48_SHA" ]; then printf 'v41'
    else printf 'unknown'; fi
}

engram_ok() {   # sizes only: cheap, and the only test that does not read 203 GB
    [ -n "$ENGRAM_DIR" ] || return 1
    [ "$(fsize "$ENGRAM_DIR/$ENGRAM_47")" = "$ENGRAM_47_BYTES" ] || return 1
    [ "$(fsize "$ENGRAM_DIR/$ENGRAM_48")" = "$ENGRAM_48_BYTES" ] || return 1
    return 0
}

# The two Engram tables are the only payload of this release that the repository
# keeps in a subdirectory (deepseek-engram/), and the HF CLI preserves the
# repository path under --local-dir: `hf download --include "deepseek-engram/*"
# --local-dir D` writes D/deepseek-engram/<file>, not D/<file>. Measured against
# huggingface_hub 1.23.0 on 2026-10-18 with this same repository and a 264 KB
# subdirectory file: `--include "docs/*" --local-dir X` produced X/docs/img/....,
# while a top-level include produced X/<file>. Everything else this script fetches
# (the 40 parts, SHA256SUMS, RECIPE.md, SOURCE.md, the vision sidecar) is at the
# repository root, which is why only these two arrive one level too deep.
engram_nested_dir() {   # prints the one-level-deep directory, when that is where the tables are
    local d
    [ -n "$ENGRAM_DIR" ] || return 1
    d="$ENGRAM_DIR/deepseek-engram"
    [ -d "$d" ] || return 1
    engram_ok && return 1
    [ "$(fsize "$d/$ENGRAM_47")" = "$ENGRAM_47_BYTES" ] || return 1
    [ "$(fsize "$d/$ENGRAM_48")" = "$ENGRAM_48_BYTES" ] || return 1
    printf '%s' "$d"
}

# True when the tables are on this disk in either place a fetch can put them, so
# the space arithmetic does not ask for 203 GB that are already here.
engram_here() { engram_ok || [ -n "$(engram_nested_dir || true)" ]; }

# Adopts a fetch that landed one level deep by moving the two tables up. That is a
# rename inside one filesystem, so no byte is copied; if the two directories are on
# different filesystems (a mount point in between) it refuses instead of silently
# copying 203 GB.
adopt_nested_engram() {
    local d f dev_nested dev_top
    d=$(engram_nested_dir) || return 1
    dev_nested=$(stat -L -c %d "$d" 2>/dev/null || echo -)
    dev_top=$(stat -L -c %d "$ENGRAM_DIR" 2>/dev/null || echo +)
    if [ "$dev_nested" != "$dev_top" ]; then
        warn "the Engram tables are in $d, on a different filesystem from $ENGRAM_DIR; moving them would copy 203 GB"
        warn "  move them yourself, or run everything with --engram-dir $d"
        return 1
    fi
    say "engram: the two tables are in $d, one level too deep (the HF CLI keeps the repository path under --local-dir)"
    say "        moving them up into $ENGRAM_DIR: a rename, nothing is copied or re-downloaded"
    for f in "$ENGRAM_47" "$ENGRAM_48"; do
        mv "$d/$f" "$ENGRAM_DIR/$f" || { warn "could not move $d/$f up into $ENGRAM_DIR"; return 1; }
    done
    rmdir "$d" 2>/dev/null || true
    say "        sizes now $ENGRAM_47=$(fsize "$ENGRAM_DIR/$ENGRAM_47") and $ENGRAM_48=$(fsize "$ENGRAM_DIR/$ENGRAM_48")"
    return 0
}

# A candidate for reuse, chosen by SIZE. The digests are not read here: hashing
# every candidate would read 203 GB per candidate, and the point of reusing is to
# avoid reading the tables at all. `verify` proves the digests once and records
# them; `start` refuses to serve until that record matches what is on disk, so a
# size-identical but different table set (the V4.1 tables) cannot slip through.
find_reusable_engram() {
    local c d
    for c in "$HOME/models/ds41flash-1017/deepseek-engram" \
             "$HOME/models/ds41-1017-engram" \
             "$HOME/models/ds41-1017/deepseek-engram" \
             "$HOME/models/ds41-1017-unc/deepseek-engram"; do
        # Each candidate is tried as it stands and one level deeper, because a copy
        # fetched by this repository's own path (see engram_nested_dir) can be
        # sitting under <candidate>/deepseek-engram. A copy found there is used in
        # place: it is not this script's directory to rearrange.
        for d in "$c" "$c/deepseek-engram"; do
            [ -d "$d" ] || continue
            ENGRAM_DIR="$d"
            engram_ok || continue
            printf '%s' "$d"; return 0
        done
    done
    return 1
}

resolve_engram_dir() {
    [ -n "$ENGRAM_DIR" ] && return 0
    if [ "$REUSE_ENGRAM" = 1 ]; then
        local found
        found=$(find_reusable_engram || true)
        if [ -n "$found" ]; then
            ENGRAM_DIR="$found"
            say "engram: found tables of the published sizes at $ENGRAM_DIR; verify proves their digests"
            return 0
        fi
    fi
    ENGRAM_DIR="$MODEL_DIR/deepseek-engram"
}

# The vision sidecar is 971 MB and name-and-size identical to the older V4.1 one,
# so unlike the tables it is cheap enough to hash: a candidate is adopted only when
# its digest is this release's 3a8b194a. A file of the published size with the V4.1
# bytes is named and passed over rather than mounted.
resolve_vision_path() {
    [ -n "$VISION_PATH" ] && return 0
    local f
    for f in "$MODEL_DIR/$VISION" \
             "$HOME/models/ds41-1017/$VISION" \
             "$HOME/models/ds41-1017-unc/$VISION" \
             "$HOME/models/ds41flash-1017/$VISION" \
             "$HOME/models/ds41-1017/model/$VISION"; do
        [ -f "$f" ] || continue
        [ "$(fsize "$f")" = "$VISION_BYTES" ] || continue
        if [ "$(sha_file "$f")" = "$VISION_SHA" ]; then
            VISION_PATH="$f"
            [ "$f" = "$MODEL_DIR/$VISION" ] \
                || say "vision: found this release's sidecar at $f (digest proven); using it instead of fetching 1 GB again"
            return 0
        fi
        warn "vision: $f has the published size but not this release's digest; it is the older V4.1 sidecar, and it is passed over"
    done
    # Last resort: the same names anywhere under $HOME/models, still digest-proven.
    while IFS= read -r f; do
        [ "$(fsize "$f")" = "$VISION_BYTES" ] || continue
        [ "$(sha_file "$f")" = "$VISION_SHA" ] || continue
        VISION_PATH="$f"
        say "vision: found this release's sidecar at $f (digest proven); using it instead of fetching 1 GB again"
        return 0
    done < <(find "$HOME/models" -maxdepth 3 -name "$VISION" -type f 2>/dev/null)
    VISION_PATH="$MODEL_DIR/$VISION"    # where download would put it
}

base_ok()  { [ "$(fsize "$GGUF_DIR/$GGUF")" = "$BASE_BYTES" ]; }
vision_ok() { [ "$(fsize "$VISION_PATH")" = "$VISION_BYTES" ]; }

# --------------------------------------------------------------------------
# The verification record. `verify` hashes the four identity files and writes what
# it proved; `start` refuses to serve unless the record still matches the bytes on
# disk (size and mtime for each file), so the expensive read happens once per
# fetch instead of once per start - and a table set that is swapped afterwards for
# one of the same size is caught by the mtime.
# --------------------------------------------------------------------------
verify_mark_path() { printf '%s/.verified' "$MODEL_DIR"; }

fingerprint() {
    local p
    for p in "$GGUF_DIR/$GGUF" "$VISION_PATH" \
             "${ENGRAM_DIR:+$ENGRAM_DIR/$ENGRAM_47}" "${ENGRAM_DIR:+$ENGRAM_DIR/$ENGRAM_48}"; do
        [ -n "$p" ] || continue
        if [ -f "$p" ]; then
            printf 'fp %s %s %s\n' "$p" "$(fsize "$p")" "$(stat -L -c %Y "$p" 2>/dev/null || echo -)"
        else
            printf 'fp %s absent -\n' "$p"
        fi
    done
}

write_verify_mark() {
    local m; m=$(verify_mark_path)
    { printf '# written by install-ds41-1017-uncensored-vq.sh verify at %s\n' "$(date -Is)"
      printf 'repo %s\n' "$REPO"
      printf 'revision %s\n' "$REVISION"
      printf 'base_sha %s\n' "$BASE_SHA"
      printf 'vision_sha %s\n' "$VISION_SHA"
      printf 'engram47_sha %s\n' "$ENGRAM_47_SHA"
      printf 'engram48_sha %s\n' "$ENGRAM_48_SHA"
      fingerprint
    } >"$m" 2>/dev/null || warn "could not write the verification record at $m"
}

# 0 when a verification record exists and still describes what is on disk.
verify_mark_ok() {
    local m; m=$(verify_mark_path)
    [ -f "$m" ] || return 1
    diff -q <(grep '^fp ' "$m") <(fingerprint) >/dev/null 2>&1
}

# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------
preflight() {
    local fatal=0
    resolve_engram_dir
    resolve_vision_path

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
        say "memory ok: ${t} MB total   (the base is about 106 GiB resident at run time)"
    else
        warn "only ${t} MB of memory; this base is about 106 GiB resident, and the card's serving line budgets 110000 MB"; fatal=1
    fi
    say "memory available now: $(avail_mb) MB (start refuses below ${START_FLOOR_MB} MB)"

    mkdir -p "$MODEL_DIR" "$GGUF_DIR" "$ENGRAM_DIR" 2>/dev/null \
        || { warn "cannot create one of $MODEL_DIR, $GGUF_DIR, $ENGRAM_DIR"; fatal=1; }

    local free need
    # The parts assemble in place, so the roomy volume needs the payload plus one
    # part of slack (the card's own loop deletes each part as it lands).
    need=0
    base_ok || need=$((BASE_BYTES + PART_BYTES_BIG))
    free=$(df_avail "$GGUF_DIR")
    if [ -n "$free" ] && [ "$free" -gt "$need" ]; then
        say "base volume ok: $(human $need) still to fetch, $(human $free) free on $(df_target "$GGUF_DIR")"
    else
        warn "$(df_target "$GGUF_DIR") has $(human ${free:-0}) free; the base alone needs $(human $need), assembly in place included"; fatal=1
    fi

    need=0
    vision_ok || need=$((need + VISION_BYTES))
    engram_here || need=$((need + ENGRAM_47_BYTES + ENGRAM_48_BYTES))
    free=$(df_avail "$MODEL_DIR")
    if [ -n "$free" ] && [ "$free" -gt "$need" ]; then
        say "fast volume ok: $(human $need) still to fetch, $(human $free) free on $(df_target "$MODEL_DIR")"
    else
        warn "$(df_target "$MODEL_DIR") has $(human ${free:-0}) free; the vision sidecar and the tables need $(human $need) there"
        warn "point --dir (or --engram-dir) at a volume with room, or set REUSE_ENGRAM=1 to pick up the 1017 tables already on the box"; fatal=1
    fi

    # What is already here, by identity rather than by name.
    if base_ok; then
        say "base ok: $GGUF_DIR/$GGUF is $(human "$BASE_BYTES") (assembled)"
    else
        local par=""
        [ "$(parts_partial)" != 0 ] && par=" ($(parts_partial) PARTIAL)"
        say "base: $(parts_present)/$PARTS parts present in $GGUF_DIR$par"
        say "the assembled base's digest is checked by assemble and verify, not here"
    fi
    vision_ok || warn "vision sidecar not present yet at $VISION_PATH"
    if engram_ok; then
        say "engram: both files are present at the published sizes in $ENGRAM_DIR"
        say "        the digests are what separates these 1017 tables from the older V4.1 ones, which carry the same names and sizes; verify proves them"
    else
        local nd; nd=$(engram_nested_dir || true)
        if [ -n "$nd" ]; then
            say "engram: the two tables are complete but one level too deep in $nd; download or verify moves them up with a rename (nothing is re-fetched)"
        else
            say "engram: will be fetched into $ENGRAM_DIR ($(human $((ENGRAM_47_BYTES + ENGRAM_48_BYTES))))"
        fi
    fi
    if verify_mark_ok; then
        say "verification record ok: $(verify_mark_path) matches the bytes on disk"
    else
        say "verification record: none, or stale ($(verify_mark_path)); run verify before start"
    fi

    if [ -x "$SERVER_BIN" ]; then
        say "runtime ok: $SERVER_BIN"
        local help; help=$("$SERVER_BIN" --help 2>&1 || true)
        local gap=()
        for f in --engram-dir --vision-gguf --mem-budget-mb --mtp-mode; do
            case "$help" in
                *"$f"*) ;;
                *) gap+=("$f");;
            esac
        done
        if [ "${#gap[@]}" -eq 0 ]; then
            say "runtime ok: advertises --engram-dir, --vision-gguf, --mem-budget-mb and --mtp-mode"
        else
            warn "$SERVER_BIN does not advertise ${gap[*]}: it predates the DeepSeek V4.1 route (PR #80) or its vision sidecar (PR #83); rebuild with: cd $ENGINE_DIR && make cuda-spark"; fatal=1
        fi
    else
        warn "no executable at $SERVER_BIN; build the host with: cd $ENGINE_DIR && make cuda-spark"; fatal=1
    fi

    if command -v curl >/dev/null 2>&1; then
        local busy=() p
        for p in 8000 8001 8002 8003 8004 8005 8006 8007 8008 8009; do
            curl -fsS --max-time 2 "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && busy+=("$p")
        done
        [ "${#busy[@]}" -gt 0 ] && warn "these ports already serve a model: ${busy[*]} - the engine's lock will refuse this start"
        if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            warn "port $PORT already answers /v1/models; pick another with --port"; fatal=1
        fi
    fi
    local owners; owners=$(other_owners)
    [ -n "$owners" ] && warn "a ds4 owner is already running: $owners"

    [ "$fatal" = 0 ] && say "preflight: this machine can run the artifact" || return 1
}

other_owners() { pgrep -af 'ds4(-server|-bench)?($| )' | grep -v "$$" || true; }

# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
cmd_plan() {
    resolve_engram_dir
    resolve_vision_path
    say "fast root:     $MODEL_DIR    (vision sidecar, logs, pid files)"
    say "gguf root:     $GGUF_DIR    (the 40 parts and the assembled base)"
    say "engram dir:    $ENGRAM_DIR"
    say "runtime:       $SERVER_BIN"
    say "serve:         $HOST_ADDR:$PORT, mtp $MTP_MODE, budget ${BUDGET_MB} MB, watchdog below ${KILL_MB} MB"
    say ""
    say "artifacts   $REPO @ ${REVISION:-main}"
    say "  base      $GGUF   $PARTS parts -> $(human $BASE_BYTES)   $(base_ok && echo assembled || echo "$(parts_present)/$PARTS parts")"
    say "            part sizes $PART_BYTES_BIG (01-24) and $PART_BYTES_SMALL (25-40)"
    say "            dissect $BASE_SHA"
    say "  vision    $VISION   $(human $VISION_BYTES)   $(vision_ok && echo present || echo absent)"
    say "            digest $VISION_SHA"
    say "  engram    $ENGRAM_47 + $ENGRAM_48   $(human $((ENGRAM_47_BYTES + ENGRAM_48_BYTES)))"
    say "            digests $ENGRAM_47_SHA and $ENGRAM_48_SHA   $(engram_ok && echo present || (engram_here && echo present-one-level-deep || echo absent)) (digests proven by verify)"
    say "  no domain sidecar ships with this artifact, so --zchain is never passed"
    say ""
    say "the command start would run:"
    say "  $SERVER_BIN --cuda -m $GGUF_DIR/$GGUF \\"
    say "      --engram-dir $ENGRAM_DIR \\"
    [ "$NO_VISION" = 1 ] || say "      --vision-gguf $VISION_PATH \\"
    say "      --mtp-mode $MTP_MODE --mem-budget-mb $BUDGET_MB --host $HOST_ADDR --port $PORT"
    say "  (no -c: this family takes its 1,048,576 context from the GGUF metadata)"
    say ""
    local need_gguf=0 need_fast=0
    base_ok || need_gguf=$((BASE_BYTES + PART_BYTES_BIG))
    vision_ok || need_fast=$((need_fast + VISION_BYTES))
    engram_here || need_fast=$((need_fast + ENGRAM_47_BYTES + ENGRAM_48_BYTES))
    say "still to fetch: $(human $need_gguf) on $(df_target "$GGUF_DIR") ($(human "$(df_avail "$GGUF_DIR")") free)"
    say "                $(human $need_fast) on $(df_target "$MODEL_DIR") ($(human "$(df_avail "$MODEL_DIR")") free)"
    say "total payload $(human $PAYLOAD_BYTES); ports 8000, 8003, 8005, 8006, 8007, 8008 and 8009 answer for the other launchers in this home"
}

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

# --include is what routes each half of the payload to its own volume. It is
# probed rather than assumed: without it the fetch cannot be split, and fetching
# 203 GB of tables onto the base volume would be the wrong layout.
hf_download() {   # $1 = target dir, rest = include patterns
    local target="$1"; shift
    "$HF" download --help 2>&1 | grep -q -- '--include' \
        || die "$HF does not support --include, which is how this script sends the parts, the vision sidecar and the tables to their own volumes. Upgrade huggingface_hub."
    export HF_HUB_MAX_WORKERS="$MAX_WORKERS"
    export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
    [ -n "${ENDPOINT:-}" ] && export HF_ENDPOINT="$ENDPOINT"
    [ "${NO_XET:-0}" = 1 ] && export HF_HUB_DISABLE_XET=1
    local -a rev=() inc=()
    [ -n "$REVISION" ] && rev=(--revision "$REVISION")
    local p; for p in "$@"; do inc+=(--include "$p"); done
    mkdir -p "$target"
    "$HF" download "$REPO" "${rev[@]}" "${inc[@]}" --local-dir "$target" --max-workers "$MAX_WORKERS"
}

# --------------------------------------------------------------------------
# download, assemble, verify
# --------------------------------------------------------------------------
cmd_download() {
    resolve_engram_dir
    resolve_vision_path
    adopt_nested_engram || true      # a fetch that landed one level deep is moved up, not re-fetched
    ensure_hf
    mkdir -p "$GGUF_DIR" "$MODEL_DIR" "$ENGRAM_DIR"

    if base_ok && vision_ok && engram_ok; then
        say "already downloaded: the base, the vision sidecar and both Engram tables are present at the published sizes"
        return 0
    fi

    local free
    if ! base_ok; then
        free=$(df_avail "$GGUF_DIR")
        [ -z "$free" ] || [ "$free" -gt $((BASE_BYTES + PART_BYTES_BIG)) ] \
            || die "$(df_target "$GGUF_DIR") has $(human ${free:-0}) free; the parts need $(human $((BASE_BYTES + PART_BYTES_BIG))) including assembly in place"
        say "fetching the 40 base parts into $GGUF_DIR ($(human $BASE_BYTES)), resumable"
        say "  the CLI prints its own progress bars; it is quiet for a while first while it resolves the file list"
        hf_download "$GGUF_DIR" "$GGUF.part*-of-40" SHA256SUMS RECIPE.md SOURCE.md \
            || die "the parts fetch failed; run the same command again to resume"
        say "parts: $(parts_present)/$PARTS present"
    else
        say "the base is already assembled at $GGUF_DIR/$GGUF"
    fi

    if ! vision_ok; then
        say "fetching the vision sidecar into $MODEL_DIR ($(human $VISION_BYTES))"
        hf_download "$MODEL_DIR" "$VISION" || die "the vision fetch failed; run the same command again to resume"
    else
        say "the vision sidecar is already present at $VISION_PATH"
    fi

    if engram_ok; then
        say "using the Engram tables already in $ENGRAM_DIR (sizes match; verify proves the digests)"
    else
        free=$(df_avail "$ENGRAM_DIR" 2>/dev/null || true)
        [ -z "$free" ] || [ "$free" -gt $((ENGRAM_47_BYTES + ENGRAM_48_BYTES)) ] \
            || die "$(df_target "$ENGRAM_DIR") has $(human ${free:-0}) free; the two tables need $(human $((ENGRAM_47_BYTES + ENGRAM_48_BYTES)))"
        say "fetching the two 1017 Engram tables into $ENGRAM_DIR ($(human $((ENGRAM_47_BYTES + ENGRAM_48_BYTES))))"
        hf_download "$ENGRAM_DIR" "deepseek-engram/*" || die "the Engram fetch failed; run the same command again to resume"
    fi

    # Sizes first, so a truncated file is named before any hashing starts.
    [ "$(parts_present)" = "$PARTS" ] || die "the fetch finished but only $(parts_present)/$PARTS parts are present"
    [ "$(parts_partial)" = 0 ] || die "the fetch left $(parts_partial) partial file(s); re-run to resume"
    vision_ok || die "the vision sidecar is not the published size: $(fsize "$VISION_PATH") against $VISION_BYTES bytes"
    if ! engram_ok; then
        local nd; nd=$(engram_nested_dir || true)
        [ -z "$nd" ] \
            || die "the Engram tables are complete but sit one level too deep in $nd and could not be moved up; run everything with --engram-dir $nd, or move the two files into $ENGRAM_DIR yourself"
        die "the Engram tables are not at the published sizes in $ENGRAM_DIR: $ENGRAM_47 is $(fsize "$ENGRAM_DIR/$ENGRAM_47") of $ENGRAM_47_BYTES bytes, $ENGRAM_48 is $(fsize "$ENGRAM_DIR/$ENGRAM_48") of $ENGRAM_48_BYTES. Re-run the same command to resume."
    fi
    say "sizes ok. Next: bash $0 assemble (verifies each part as it lands), then bash $0 verify"
}

cmd_assemble() {
    resolve_engram_dir
    [ -f "$GGUF_DIR/SHA256SUMS" ] || die "$GGUF_DIR/SHA256SUMS is missing; run download first"
    need sha256sum
    if base_ok; then
        say "the base is already assembled ($(human "$BASE_BYTES")); checking its digest"
        local d; d=$(sha_file "$GGUF_DIR/$GGUF")
        [ "$d" = "$BASE_SHA" ] || die "the assembled base hashes to $d, not $BASE_SHA. Delete it and re-run assemble."
        say "base digest ok: $BASE_SHA"
        return 0
    fi
    # A base file that is neither absent nor the published size is the wreck of an
    # interrupted assembly. It cannot be continued: the loop deletes each part as
    # it appends it, so the parts already inside it are gone from disk. Say that
    # plainly instead of letting the parts check below report a puzzle.
    local have; have=$(fsize "$GGUF_DIR/$GGUF")
    if [ "$have" != 0 ] && [ "$have" != "$BASE_BYTES" ]; then
        die "$GGUF_DIR/$GGUF exists but is $(human "$have") of $(human $BASE_BYTES): an interrupted assemble.
       The parts it already appended were deleted, so they must be fetched again:
         rm '$GGUF_DIR/$GGUF'
         bash $0 download
         bash $0 assemble"
    fi
    [ "$(parts_present)" = "$PARTS" ] || die "only $(parts_present)/$PARTS parts are present; run download first"

    # The card's own loop: append each part and delete it as it lands, so only one
    # part of extra room is ever needed. Each part is hashed against the release's
    # SHA256SUMS before it is appended, so a wrong or truncated part never enters
    # the base. The loop prints one line per part: hashing 2.8 GB and appending it
    # takes seconds, so silence here read as a hang (reported on the Spark,
    # 2026-10-18).
    say "assembling $PARTS parts into $GGUF_DIR/$GGUF; each part is verified against SHA256SUMS, appended, then removed"
    say "  one line per part; expect about $((PARTS * 5))s in total on a fast volume"
    local i p sz acc=0 t0=$SECONDS
    for i in $(seq -w 1 40); do
        p=$(part_name "$i")
        [ -f "$GGUF_DIR/$p" ] || die "missing $p"
        sz=$(fsize "$GGUF_DIR/$p")
        ( cd "$GGUF_DIR" && grep -qF "  $p" SHA256SUMS ) || die "$p is not in $GGUF_DIR/SHA256SUMS; this is not the pinned revision's part set"
        ( cd "$GGUF_DIR" && grep -F "  $p" SHA256SUMS | sha256sum -c --quiet - ) \
            || die "$p does not match SHA256SUMS; delete the parts and fetch them again"
        cat "$GGUF_DIR/$p" >> "$GGUF_DIR/$GGUF" || die "append of $p failed"
        rm -f "$GGUF_DIR/$p"
        acc=$((acc + sz))
        say "  [$i/$PARTS] $p verified and appended: $(human "$acc") of $(human "$BASE_BYTES")  ($((SECONDS - t0))s elapsed)"
    done
    [ "$(fsize "$GGUF_DIR/$GGUF")" = "$BASE_BYTES" ] \
        || die "the assembled base is $(fsize "$GGUF_DIR/$GGUF") bytes, not $BASE_BYTES; delete it and start over"
    say "the base is complete ($(human $BASE_BYTES)); hashing it once against the release's digest (this reads the whole file)"
    local d; d=$(sha_file "$GGUF_DIR/$GGUF")
    [ "$d" = "$BASE_SHA" ] || die "the assembled base hashes to $d, not the published $BASE_SHA. This is a different 1017 base ($OTHER_BASE_SHAS). Delete it and fetch this release's parts."
    say "assembled: $(human $BASE_BYTES), digest $BASE_SHA (the release's own)"
}

cmd_verify() {
    resolve_engram_dir
    resolve_vision_path
    adopt_nested_engram || true      # verify repairs the fetch's one-level-deep landing instead of failing on it
    need sha256sum
    local bad=0

    # Every check below is explicit and counted. `sha256sum -c --ignore-missing`
    # is deliberately NOT used on its own: on a directory where none of the listed
    # files exist it exits 0 without verifying anything (measured, 2026-10-18), so
    # a vacuous pass would be possible. Presence is asserted first instead.

    # 1. The parts, or the base they assemble into.
    if base_ok; then
        say "hashing the assembled base ($(human $BASE_BYTES)); this reads the whole file, so expect a minute or two"
        local d; d=$(sha_file "$GGUF_DIR/$GGUF")
        if [ "$d" = "$BASE_SHA" ]; then say "  base ok: $BASE_SHA"
        else warn "  base MISMATCH: $d, published $BASE_SHA (other 1017 bases: $OTHER_BASE_SHAS)"; bad=1; fi
    else
        [ "$(parts_present)" = "$PARTS" ] \
            || die "neither the assembled base nor all $PARTS parts are present in $GGUF_DIR (found $(parts_present)); run download"
        [ -f "$GGUF_DIR/SHA256SUMS" ] || die "$GGUF_DIR/SHA256SUMS is missing; run download first"
        say "hashing the $PARTS parts ($(human $BASE_BYTES)); one line per part"
        local i p want got ok=0
        for i in $(seq -w 1 40); do
            p=$(part_name "$i")
            want=$(awk -v n="$p" '$2 == n {print $1}' "$GGUF_DIR/SHA256SUMS")
            [ -n "$want" ] || { warn "  $p is not listed in SHA256SUMS; this is not the pinned revision's part set"; bad=1; continue; }
            got=$(sha_file "$GGUF_DIR/$p")
            if [ "$got" = "$want" ]; then ok=$((ok + 1)); say "  [$i/$PARTS] $p ok"
            else warn "  [$i/$PARTS] $p MISMATCH: $got, published $want"; bad=1; fi
        done
        [ "$ok" = "$PARTS" ] && say "  parts ok: $ok/$PARTS match SHA256SUMS"
    fi

    # 2. The vision sidecar. The older V4.1 sidecar is a different file under the
    #    same name and size.
    if vision_ok; then
        local dv; dv=$(sha_file "$VISION_PATH")
        if [ "$dv" = "$VISION_SHA" ]; then say "vision identity ok: $VISION_SHA"
        else warn "vision identity FAILED: $dv, published $VISION_SHA (the older V4.1 sidecar is a different file under the same name)"; bad=1; fi
    else
        warn "the vision sidecar is missing or the wrong size at $VISION_PATH ($(fsize "$VISION_PATH") against $VISION_BYTES)"; bad=1
    fi

    # 3. The two Engram tables, by digest. Same names and sizes as the V4.1 ones.
    if engram_ok; then
        say "hashing the two Engram tables ($(human $((ENGRAM_47_BYTES + ENGRAM_48_BYTES)))); this reads both files once, so expect several minutes"
        case "$(engram_identity)" in
            1017) say "  engram identity ok: both 1017 tables, $ENGRAM_47_SHA and $ENGRAM_48_SHA";;
            v41)  warn "  engram identity FAILED: these are the OLDER V4.1 tables ($V41_ENGRAM_47_SHA / $V41_ENGRAM_48_SHA)"; bad=1;;
            *)    warn "  engram identity FAILED: neither release's digests"; bad=1;;
        esac
    else
        warn "the Engram tables are missing or the wrong sizes in $ENGRAM_DIR"; bad=1
    fi

    [ "$bad" = 0 ] || die "verify found a mismatch; nothing here should be served until it is fixed. No verification record was written."
    say "verify: every digest matches the pinned revision"
    write_verify_mark
    say "verification record written: $(verify_mark_path); start will now accept this file set"
}

# --------------------------------------------------------------------------
# build, start, stop, status, logs, test
# --------------------------------------------------------------------------
cmd_build() {
    if [ -x "$SERVER_BIN" ] && [ "$NO_BUILD" = 1 ]; then
        say "runtime ok: $SERVER_BIN (--no-build: not checking the tree)"
        return 0
    fi
    [ -d "$ENGINE_DIR/.git" ] || die "$ENGINE_DIR is not a git checkout of ds4-dfm-rs"
    need make
    say "building the engine in $ENGINE_DIR with: make cuda-spark"
    ( cd "$ENGINE_DIR" && make cuda-spark ) || die "the engine build failed"
    [ -x "$SERVER_BIN" ] || die "the build finished but $SERVER_BIN is not executable"
    say "runtime ok: $SERVER_BIN"
}

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
    resolve_engram_dir
    resolve_vision_path
    base_ok || die "$GGUF_DIR/$GGUF is not assembled; run: bash $0 download && bash $0 assemble"
    if ! engram_ok; then
        local nd; nd=$(engram_nested_dir || true)
        [ -z "$nd" ] || die "the Engram tables are one level too deep in $nd; run: bash $0 verify (it moves them up and writes the verification record)"
    fi
    vision_ok || [ "$NO_VISION" = 1 ] \
        || die "the vision sidecar is missing or the wrong size at $VISION_PATH ($(fsize "$VISION_PATH") against $VISION_BYTES bytes); run download, pass --vision PATH, or pass --no-vision"
    engram_ok || die "the Engram tables are missing or the wrong size in $ENGRAM_DIR; run download"
    verify_mark_ok \
        || die "no verification record matches the bytes on disk ($(verify_mark_path)).
       Run: bash $0 verify
       It hashes the assembled base, the vision sidecar and the two tables (about 317 GB of reads) and
       proves that the tables are this release's 1017 files rather than the older V4.1 ones, which
       carry the same names and the same sizes."
    [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN; build it with: bash $0 build"
    need curl
    if alive; then die "already running (pid $(cat "$PIDF")); stop it first with: bash $0 stop"; fi

    local owners; owners=$(other_owners)
    if [ -n "$owners" ] && [ "$ALLOW_SHARED" != 1 ]; then
        printf 'ds41unc: another model owner is running:\n%s\n' "$owners" >&2
        die "stop it first (the engine takes a single global lock), or set ALLOW_SHARED=1 to overrule"
    fi

    local a; a=$(avail_mb)
    [ "$a" -ge "$START_FLOOR_MB" ] || die "only $a MB of memory available, need $START_FLOOR_MB: stop another model first"

    mkdir -p "$MODEL_DIR"
    local args=(--cuda -m "$GGUF_DIR/$GGUF" --engram-dir "$ENGRAM_DIR"
                --mtp-mode "$MTP_MODE" --mem-budget-mb "$BUDGET_MB"
                --host "$HOST_ADDR" --port "$PORT")
    [ "$NO_VISION" = 1 ] || args+=(--vision-gguf "$VISION_PATH")

    say "starting ds4-server on $HOST_ADDR:$PORT (mapping 113.6 GB; the card reports about two minutes)"
    say "  base:    $GGUF_DIR/$GGUF"
    say "  engram:  $ENGRAM_DIR"
    say "  vision:  $([ "$NO_VISION" = 1 ] && echo off || echo "$VISION_PATH")"
    say "  log:     $SERVER_LOG"
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
            tail -n 15 "$SERVER_LOG" 2>/dev/null | sed 's/^/  /' >&2
            die "the server exited while loading (full log: $SERVER_LOG)"
        fi
        sleep 5; waited=$((waited + 5))
        [ $((waited % 60)) -eq 0 ] && say "  still loading (${waited}s)"
    done
    [ "$waited" -lt "$READY_TIMEOUT" ] || { tail -n 15 "$SERVER_LOG" | sed 's/^/  /' >&2; die "no answer on port $PORT within ${READY_TIMEOUT}s"; }

    a=$(avail_mb)
    if [ "$a" -lt "$RUN_FLOOR_MB" ]; then
        cmd_stop >/dev/null 2>&1 || true
        die "only $a MB left after loading (need $RUN_FLOOR_MB); the machine would start swapping"
    fi
    say "loaded; ${a} MB of memory left"
    cmd_test || warn "the smoke request failed; the server is up, so read $SERVER_LOG"
    local ip; ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    say "ready: http://127.0.0.1:$PORT/v1   (lan http://${ip:-<host>}:$PORT/v1, port $PORT)"
    say "this family defaults to non-thinking mode; send temperature and top_p explicitly"
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
    resolve_engram_dir
    resolve_vision_path
    if alive; then
        say "ds4-server:  running (pid $(cat "$PIDF"))"
        ps -o pid=,etime=,rss= -p "$(cat "$PIDF")" 2>/dev/null | sed 's/^/  /'
    else
        say "ds4-server:  not running"
    fi
    [ -s "$WD_PIDF" ] && kill -0 "$(cat "$WD_PIDF")" 2>/dev/null \
        && say "watchdog:    running (pid $(cat "$WD_PIDF"))" || say "watchdog:    not running"
    say "memory:      $(avail_mb) MB available of $(total_mb) MB"
    say "base:        $([ "$(fsize "$GGUF_DIR/$GGUF")" = "$BASE_BYTES" ] && echo "assembled $(human "$BASE_BYTES")" || echo "$(parts_present)/$PARTS parts, not assembled") in $GGUF_DIR"
    say "vision:      $([ "$(fsize "$VISION_PATH")" = "$VISION_BYTES" ] && echo "present $(human "$VISION_BYTES")" || echo "absent or wrong size") at $VISION_PATH"
    say "engram:      $([ -z "$ENGRAM_DIR" ] && echo unresolved || (engram_ok && echo "present at the published sizes" || (engram_here && echo "present but one level too deep; run download or verify to move it up" || echo "absent or wrong size"))) in $ENGRAM_DIR"
    say "verified:    $(if verify_mark_ok; then echo "yes - $(verify_mark_path) matches the bytes on disk"; elif [ -f "$(verify_mark_path)" ]; then echo "no - the record is stale against the bytes on disk, re-run verify"; else echo "no record; run: bash $0 verify"; fi)"
    if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | head -c 300; then
        say ""; say "endpoint:    answering on port $PORT"
    else
        say "endpoint:    no answer on port $PORT"
    fi
    if [ -f "$SERVER_LOG" ]; then say ""; say "log tail:"; tail -n 5 "$SERVER_LOG" | sed 's/^/  /'; fi
}

cmd_logs() {
    say "server log:   $SERVER_LOG"
    [ -f "$SERVER_LOG" ] && tail -n 20 "$SERVER_LOG" | sed 's/^/  /'
    say ""
    say "watchdog log: $WD_LOG"
    [ -f "$WD_LOG" ] && tail -n 10 "$WD_LOG" | sed 's/^/  /'
    return 0
}

cmd_test() {
    need curl
    # The model id this endpoint advertises is read back rather than assumed.
    local mid out
    mid=$(curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
          | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
    mid=${mid:-deepseek-v4.1-flash-1017}
    say "smoke request against model id '$mid' (read from /v1/models)"
    out=$(curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$mid\",\"temperature\":0,\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, what does a vector-quantized expert layer store?\"}]}")
    case "$out" in
        *'"choices"'*)
            printf '%s\n' "$out" | sed -n 's/.*"timings":{\(.*\)}.*/  timings: \1/p' | head -1
            printf '%s\n' "$out" | sed -n 's/.*"content":"\([^"]\{0,90\}\)".*/  answer: \1.../p' | head -1;;
        *) warn "the request failed: $(printf '%s' "$out" | head -c 300)"; return 1;;
    esac
}

usage() { sed -n '2,118p' "$0" | sed 's/^# \{0,1\}//'; }

# --------------------------------------------------------------------------
# options: consumed only while the first argument is an option. The action word
# (preflight, download, ...) ends the loop, and the action case below reads it.
# --------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --dir)          [ $# -ge 2 ] || die "--dir needs a directory"; MODEL_DIR="$2"; PIDF="$MODEL_DIR/ds41unc.pid"; SERVER_LOG="$MODEL_DIR/ds41unc.log"; WD_PIDF="$MODEL_DIR/ds41unc-watchdog.pid"; WD_LOG="$MODEL_DIR/ds41unc-watchdog.log"; shift 2;;
        --gguf-dir)     [ $# -ge 2 ] || die "--gguf-dir needs a directory"; GGUF_DIR="$2"; shift 2;;
        --engram-dir)   [ $# -ge 2 ] || die "--engram-dir needs a directory"; ENGRAM_DIR="$2"; shift 2;;
        --vision)       [ $# -ge 2 ] || die "--vision needs a path"; VISION_PATH="$2"; shift 2;;
        --port)         [ $# -ge 2 ] || die "--port needs a number"; PORT="$2"; shift 2;;
        --host)         [ $# -ge 2 ] || die "--host needs an address"; HOST_ADDR="$2"; shift 2;;
        --mtp-mode)     [ $# -ge 2 ] || die "--mtp-mode needs off | auto | on"; MTP_MODE="$2"; shift 2;;
        --mem-budget-mb) [ $# -ge 2 ] || die "--mem-budget-mb needs a number"; BUDGET_MB="$2"; shift 2;;
        --engine-dir)   [ $# -ge 2 ] || die "--engine-dir needs a directory"; ENGINE_DIR="$2"; shift 2;;
        --server-bin)   [ $# -ge 2 ] || die "--server-bin needs a path"; SERVER_BIN="$2"; shift 2;;
        --hf-bin)       [ $# -ge 2 ] || die "--hf-bin needs a path"; HF_BIN="$2"; shift 2;;
        --revision)     [ $# -ge 2 ] || die "--revision needs a sha"; REVISION="$2"; shift 2;;
        --watchdog)     [ $# -ge 2 ] || die "--watchdog needs 0 or 1"; WATCHDOG="$2"; shift 2;;
        --no-vision)    NO_VISION=1; shift;;
        --no-build)     NO_BUILD=1; shift;;
        --no-start)     NO_START=1; shift;;
        --allow-shared) ALLOW_SHARED=1; shift;;
        -h|--help)      usage; exit 0;;
        -*)             printf 'unknown option: %s (see --help)\n' "$1" >&2; usage >&2; exit 2;;
        *)              break;;
    esac
done

case "${1:-}" in
    preflight) shift; preflight "$@";;
    plan)      shift; cmd_plan "$@";;
    download)  shift; cmd_download "$@";;
    assemble)  shift; cmd_assemble "$@";;
    verify)    shift; cmd_verify "$@";;
    build)     shift; cmd_build "$@";;
    start)     shift; cmd_start "$@";;
    stop)      shift; cmd_stop "$@";;
    status)    shift; cmd_status "$@";;
    logs)      shift; cmd_logs "$@";;
    test)      shift; cmd_test "$@";;
    install)
        shift
        preflight
        cmd_download
        cmd_assemble
        cmd_verify
        cmd_build
        [ "$NO_START" = 1 ] || cmd_start "$@"
        ;;
    "")        usage; exit 2;;
    *)         printf 'unknown action: %s (see --help)\n' "$1" >&2; usage >&2; exit 2;;
esac
