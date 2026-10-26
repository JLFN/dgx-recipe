#!/usr/bin/env bash
# MiniMax-M3.1-Flash VQ (428B MoE): fetch the release directory, prove which
# bytes are on the box, place the K/V tier on the fast device, and serve it on
# the DGX Spark.
#
#   artifact  Baekpica/MiniMax-M3.1-Flash-VQ-GGUF
#             pinned at 2789cbaf1b7dce505d55d6026a67c10e1815c78c (main head,
#             2026-10-24T13:00:24Z). The 61 parts are byte-identical at this
#             revision and at 74e5a9fd12fb57a5c1babb559536cf10733939ea, the
#             revision the engine's docs and the author's measurements name:
#             release.json and SHA256SUMS hash the same at both, and every part
#             digest in release.json matches SHA256SUMS in both (checked
#             2026-10-26 against the Hub's resolve API).
#   layout    61 independent GGUF parts (not numbered splits) in one directory:
#             37 R13 expert parts (7 x 1,475,503,488, 30 x 1,475,503,872),
#             20 R13+8 expert parts of 2,381,629,312, body-nonexpert.gguf
#             6,930,488,832, embed-tokens-bf16.gguf 2,458,387,456,
#             vision-bf16.gguf 1,734,197,248, dspark-q8_0.gguf 2,177,025,024.
#             115,526,325,376 bytes of payload; 227 files in the revision.
#   engine    ds4-dfm-rs with the minimax_m3_vq family (draft PR #110, not
#             merged); read here from origin/feat/add-minimax-vq fd7192f4, and
#             built with make cuda-spark
#   source    MiniMaxAI/MiniMax-M3.1-Flash at 54e4e69d, under the MiniMax
#             Community License (not MIT)
#
# WHERE EACH PIECE GOES, AND WHY
#   The rule this repository follows: bytes read once at load and then held
#   resident go on the roomy volume; bytes read while the server runs go on the
#   fastest device.
#   --release-dir (default /mnt/models/MiniMax-M3.1-Flash-VQ-GGUF): the whole
#   release. Its 103.69 GiB of trunk, body and drafter are copied to the device
#   once at open and never read again, so the volume's speed stops mattering.
#   --kv-dir (default $HOME/models/minimax-m31-vq/kv-offload), exported as
#   DS4_KV_OFFLOAD_DIR: the K/V tier, an unlinked O_DIRECT file of 33.75 GiB at
#   1M, read and written per token and per prefill chunk once the context passes
#   131,072 tokens. It is the run-time-read payload, so it belongs on NVMe; the
#   engine's own default is ~/.cache/ds4/kv-offload, which puts it behind the
#   home partition's traffic.
#   Inside the release directory, unavoidably: embed-tokens-bf16.gguf (2.29 GiB,
#   one 12,288-byte row read per token) and vision-bf16.gguf (1.62 GiB, streamed
#   whole per image, then dropped from the page cache). The engine reads every
#   part as a filename inside the release directory and refuses a directory
#   holding a GGUF release.json does not declare, so these two cannot be moved
#   the way the DeepSeek recipes move their Engram tables and PLE sidecars. They
#   are 3.90 GiB of the payload.
#
# THE MISTAKES THIS SCRIPT EXISTS TO PREVENT
#   1. Putting the K/V tier where the engine defaults. 36.2 GiB of per-token
#      reads on the home partition is the difference between 14.7 and 19 tok/s in
#      the long-context regime, and the tier is failocated in full before the
#      port binds, so a full disk fails at open.
#   2. Treating the parts as a split GGUF. They are independent GGUFs; -m takes
#      the directory, and a stray .gguf in it makes the open fail outright.
#   3. Starting at 1M with the drafter and expecting room for anything else. The
#      author measured 114.75 GiB peak on that shape and calls it unsafe on a
#      128 GB host; the no-drafter 1M run peaked at 110.7 GiB.
#   4. Reading "auto" as "the drafter is on". MTP mode auto resolves to off for
#      this family, exactly as it does for GLM VQ, so an automatic mode silently
#      serves without the draft head. This launcher passes an explicit mode.
#   5. Trusting a directory that merely looks complete. The engine hashes every
#      part at open against release.json, and this script's verify re-runs the
#      release's own SHA256SUMS independently; start refuses a directory whose
#      recorded sizes and mtimes do not match the bytes on disk.
#
# WHAT HAS NOT BEEN MEASURED
#   Nothing in this recipe has been loaded on the reference Spark, because the
#   engine support is a draft pull request. Every speed and fidelity figure in
#   the contract and the runbook belongs to the artifact's author's box; the
#   first start here should replace them.
#
# USAGE
#   bash install-minimax-m31-flash-vq.sh preflight
#   bash install-minimax-m31-flash-vq.sh selftest   # the script's own contract, no artifact needed
#   bash install-minimax-m31-flash-vq.sh download   # 115.53 GB, resumable
#   bash install-minimax-m31-flash-vq.sh verify     # SHA256SUMS over all 115.53 GB
#   bash install-minimax-m31-flash-vq.sh check      # the engine's own open + plan quote
#   bash install-minimax-m31-flash-vq.sh build      # make cuda-spark in the engine tree
#   bash install-minimax-m31-flash-vq.sh start      # 0.0.0.0:8011
#   bash install-minimax-m31-flash-vq.sh all        # preflight, download, check, start
#   bash install-minimax-m31-flash-vq.sh stop | status | logs | test | plan
#
# OPTIONS (each is also an environment variable of the same name)
#   --release-dir DIR   the release directory: all 61 parts, sidecars and
#                       release.json (default /mnt/models/MiniMax-M3.1-Flash-VQ-GGUF)
#   --kv-dir DIR        DS4_KV_OFFLOAD_DIR: the SSD K/V tier (default
#                       $HOME/models/minimax-m31-vq/kv-offload)
#   --state-dir DIR     logs, pid files and the verification record
#                       (default $HOME/models/minimax-m31-vq)
#   --port N            listen port (default 8011, this artifact's port)
#   --host ADDR         listen address (default 0.0.0.0)
#   --ctx N             serve a shorter context; empty (the default) keeps the
#                       native 1,048,576. -c 131072 keeps every K/V page
#                       resident and creates no tier file at all.
#   --max-seqs N        bank count (default 1: the 1M plan is 113.95 GiB)
#   --mtp-mode MODE     off | auto | on, the DSpark draft head (default on)
#   --mtp-draft N       proposals per step, 2..7 (default 3)
#   --revision SHA      pin another revision of the repository
#   --engine-dir DIR    the ds4-dfm-rs tree (default $HOME/ds4-dfm-rs)
#   --server-bin PATH   the binary to run (default <--engine-dir>/ds4-server)
#   --hf-bin PATH       the Hugging Face CLI to use
#   --start-floor-mb N  refuse to start below this MemAvailable (default 105000)
#   --kill-mb N         watchdog floor, two 5 s samples (default 2500)
#   --watchdog 0|1      memory watchdog (default 1)
#   --no-build          never run make, even when the binary is missing
#   --no-start          everything except starting the server
#   --allow-shared      overrule the single-owner probe before starting
#   -h, --help          this text
#
# Notes
#   - The repository is public: no token is required and none is ever written
#     into this file. HF_TOKEN only raises the CDN rate limit.
#   - start and check use the same flags, so what check approves is what start
#     runs. check opens and verifies all 61 parts (the engine's own hashing pass,
#     115.53 GB of reads) and prints the plan; it does not touch the device.
#   - The K/V tier is an unlinked file: nothing is left behind after a stop, but
#     its full size is fallocated when the bank is created. At 1M that is
#     36.2 GB on --kv-dir, and the check is made before the port binds.
#   - The launcher's post-load watchdog exists because this artifact's 1M margin
#     is thin: the card's plan is 113.95 GiB of the 117.2 GiB the engine sees.

set -Eeuo pipefail

# --------------------------------------------------------------------------
# Configuration (every value is overridable from the environment)
# --------------------------------------------------------------------------
REPO=${MINIMAX31_REPO:-Baekpica/MiniMax-M3.1-Flash-VQ-GGUF}
PIN=2789cbaf1b7dce505d55d6026a67c10e1815c78c
# The revision the engine's docs and the author's measurements name. Its weights
# are byte-identical to the pin (release.json and SHA256SUMS hash the same at
# both; checked 2026-10-26), so it is offered as an equal choice rather than an
# alternative artifact.
MEASURED_REVISION=74e5a9fd12fb57a5c1babb559536cf10733939ea
REVISION=${MINIMAX31_REVISION:-$PIN}

RELEASE_NAME="MiniMax-M3.1-Flash-VQ-GGUF"
MODEL_ROOT=${MINIMAX31_MODEL_ROOT:-/mnt/models}
RELEASE_DIR=${MINIMAX31_RELEASE_DIR:-$MODEL_ROOT/$RELEASE_NAME}
STATE_DIR=${MINIMAX31_STATE_DIR:-$HOME/models/minimax-m31-vq}
KV_DIR=${MINIMAX31_KV_DIR:-$STATE_DIR/kv-offload}
SERVER_BIN=${MINIMAX31_BIN:-$HOME/ds4-dfm-rs/ds4-server}
ENGINE_SRC=${MINIMAX31_ENGINE_SRC:-$HOME/ds4-dfm-rs}
ENGINE_BRANCH=${MINIMAX31_ENGINE_BRANCH:-feat/add-minimax-vq}
ENGINE_REF=${MINIMAX31_ENGINE_REF:-fd7192f4}    # the branch tip this recipe was read against
HOST=${MINIMAX31_HOST:-0.0.0.0}
# 8000 is DeepSeek V4.1 Flash, 8003 Qwen3.8, 8005 Bonsai, 8006 YoungAi V4.1,
# 8007 GLM-5.3 VQ, 8008 YoungAi 1017, 8009 GLM-5.3 Uncensored VQ and 8010 the
# DeepSeek 1017 Uncensored VQ, so this artifact takes the next free one.
PORT=${MINIMAX31_PORT:-8011}
MODEL_ID=${MINIMAX31_MODEL_ID:-minimax-m31-flash-vq}
CTX=${MINIMAX31_CTX:-}                   # empty keeps the native 1,048,576
MAX_SEQS=${MINIMAX31_MAX_SEQS:-1}        # one bank: the card's 1M plan is 113.95 GiB of 117.2
MTP_MODE=${MINIMAX31_MTP_MODE:-on}       # auto resolves to off for this family
DRAFT=${MINIMAX31_DRAFT:-3}              # the card's sweep setting; the engine's useful range is 2-3
WATCHDOG=${MINIMAX31_WATCHDOG:-1}
START_FLOOR_MB=${MINIMAX31_START_FLOOR_MB:-105000}   # before loading
RUN_FLOOR_MB=${MINIMAX31_RUN_FLOOR_MB:-4000}         # reported after loading
KILL_MB=${MINIMAX31_KILL_MB:-2500}                   # two consecutive 5 s samples stop the server
READY_TIMEOUT_S=${MINIMAX31_READY_TIMEOUT:-1800}
NO_BUILD=${MINIMAX31_NO_BUILD:-0}
MAX_WORKERS=${MINIMAX31_MAX_WORKERS:-8}

PID_FILE=${MINIMAX31_PID:-$STATE_DIR/minimax31.pid}
LOG_FILE=${MINIMAX31_LOG:-$STATE_DIR/minimax31.log}
WD_PID_FILE=${MINIMAX31_WD_PID:-$STATE_DIR/minimax31-watchdog.pid}
WD_LOG=${MINIMAX31_WD_LOG:-$STATE_DIR/minimax31-watchdog.log}
MARK_FILE=${MINIMAX31_MARK:-$STATE_DIR/verified.txt}

# The release contract, from the pinned revision's release.json / files.json.
RELEASE_JSON_SHA=2110c1e5cba8d59987e370d50dab6dde5d5f7fc82009cdc27d8131452536f4b0
RELEASE_JSON_BYTES=276
SHA256SUMS_SHA=23ed23bd95c164c6dbe67584b1a72f1cc1303da4ee7ea91884a7983382b86ccc
SHA256SUMS_BYTES=21747
PART_COUNT=61
PAYLOAD_BYTES=115526325376             # the 61 parts, 107.59 GiB
LISTING_BYTES=115610523745             # every file at the pinned revision, 227 files
WORK_ROOM_BYTES=10000000000            # 10 GB of headroom while a fetch is in flight
RESIDENT_BYTES=111333740672            # trunk + body + drafter, 103.69 GiB
SIDECAR_BYTES=4192584704               # token table + vision tower, 3.90 GiB
KV_TIER_BYTES_1M=36238786560           # the tier file, fallocated at bank creation
KV_TIER_BYTES_512K=18119393280

# The 57 expert parts: every MoE layer, 3 to 59.
LAYERS=()
for ((l = 3; l <= 59; l++)); do LAYERS+=("$(printf 'layer-%02d-experts.gguf' "$l")"); done
# The twenty layers the author measured into the R13+8 residual stage.
R13P8_LAYERS="16 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 35 40 43 53"
LAYER_SMALL_BYTES=1475503488           # R13, layers 03-09
LAYER_LARGE_BYTES=1475503872           # R13, the other 30
LAYER_RESIDUAL_BYTES=2381629312        # R13+8, 20 layers
BODY=body-nonexpert.gguf
BODY_BYTES=6930488832
EMBED=embed-tokens-bf16.gguf
EMBED_BYTES=2458387456
VISION=vision-bf16.gguf
VISION_BYTES=1734197248
DSPARK=dspark-q8_0.gguf
DSPARK_BYTES=2177025024
SIDECAR_FILES=(release.json files.json config.json tokenizer.json tokenizer_config.json chat_template.jinja SHA256SUMS)
# The sidecar files SHA256SUMS lists beside the parts. SHA256SUMS cannot list
# itself, so the strict minimum below counts these six, not the seven above.
SUMS_REQUIRED=(release.json files.json config.json tokenizer.json tokenizer_config.json chat_template.jinja)
EXTRA_FILES=(special_tokens_map.json added_tokens.json generation_config.json preprocessor_config.json memory-budget.json recipe.json tensor-manifest.json README.md RECIPE.md FORMAT.md SOURCE.md LICENSE LICENSE-YoungAi)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
say()  { printf 'minimax31: %s\n' "$*"; }
warn() { printf 'minimax31: warning: %s\n' "$*" >&2; }
die()  { printf 'minimax31: error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

human() {   # bytes -> human-readable
    awk -v b="${1:-0}" 'BEGIN {
        split("B KiB MiB GiB TiB", u, " ");
        i = 1;
        while (b >= 1024 && i < 5) { b /= 1024; i++ }
        printf (i == 1 ? "%.0f %s" : "%.2f %s"), b, u[i];
    }'
}

fsize() { if [ -f "$1" ]; then stat -c %s "$1" 2>/dev/null || echo 0; else echo 0; fi; }
fmtime() { if [ -f "$1" ]; then stat -c %Y "$1" 2>/dev/null || echo 0; else echo 0; fi; }
avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
total_mb() { awk '/MemTotal/{print int($2/1024)}' /proc/meminfo; }
df_avail() { df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' \n' || true; }
df_target() { df --output=target "$1" 2>/dev/null | tail -1 | tr -d '\n' || true; }
sha_file() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
alive() { [ -s "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }
other_owners() { pgrep -af 'ds4(-server|-bench)?($| )' 2>/dev/null | grep -v "$$" || true; }

# --------------------------------------------------------------------------
# The release contract: what should be on disk, and what state it is in
# --------------------------------------------------------------------------
layer_bytes() {   # $1 = layer number
    case " $R13P8_LAYERS " in
        *" $1 "*) echo "$LAYER_RESIDUAL_BYTES";;
        *) if [ "$1" -le 9 ]; then echo "$LAYER_SMALL_BYTES"; else echo "$LAYER_LARGE_BYTES"; fi;;
    esac
}

part_names() { printf '%s\n' "${LAYERS[@]}" "$BODY" "$EMBED" "$VISION" "$DSPARK"; }

part_expected() {   # $1 = part name -> expected bytes
    case "$1" in
        "$BODY")   echo "$BODY_BYTES";;
        "$EMBED")  echo "$EMBED_BYTES";;
        "$VISION") echo "$VISION_BYTES";;
        "$DSPARK") echo "$DSPARK_BYTES";;
        layer-*-experts.gguf)
            local n=${1#layer-}; n=${n%%-*}
            layer_bytes "$((10#$n))";;
        *) echo 0;;
    esac
}

parts_present() {   # parts on disk with the published size
    local p n=0
    while read -r p; do
        [ "$(fsize "$RELEASE_DIR/$p")" = "$(part_expected "$p")" ] && n=$((n + 1))
    done < <(part_names)
    printf '%s' "$n"
}

parts_wrong_size() {   # parts present but not at the published size
    local p out=() e a
    while read -r p; do
        a=$(fsize "$RELEASE_DIR/$p"); [ "$a" = 0 ] && continue
        e=$(part_expected "$p")
        [ "$a" = "$e" ] || out+=("$p ($a vs $e)")
    done < <(part_names)
    [ "${#out[@]}" -eq 0 ] || printf '%s\n' "${out[@]}"
}

parts_partial() { find "$RELEASE_DIR" -maxdepth 1 -name '*.incomplete' 2>/dev/null | wc -l; }

sidecars_present() {   # the files the engine reads besides the parts
    local f n=0
    for f in "${SIDECAR_FILES[@]}"; do [ -f "$RELEASE_DIR/$f" ] && n=$((n + 1)); done
    printf '%s' "$n"
}

release_ok() {   # every part at its published size, and every required sidecar
    [ "$(parts_present)" = "$PART_COUNT" ] && [ "$(sidecars_present)" = "${#SIDECAR_FILES[@]}" ]
}

release_identity() {   # what the directory's own release.json says
    local f="$RELEASE_DIR/release.json"
    [ -f "$f" ] || { echo "absent"; return; }
    local sha; sha=$(sha_file "$f")
    [ "$sha" = "$RELEASE_JSON_SHA" ] && echo "pinned" || echo "unknown:$sha"
}

stray_gguf() {   # a .gguf the release does not declare; the engine refuses the directory
    local declared=() f
    while read -r f; do declared+=("$f"); done < <(part_names)
    while read -r f; do
        [ -z "$f" ] && continue
        local base; base=$(basename "$f")
        local hit=0 d
        for d in "${declared[@]}"; do [ "$base" = "$d" ] && hit=1 && break; done
        [ "$hit" = 0 ] && printf '%s\n' "$base"
    done < <(find "$RELEASE_DIR" -maxdepth 1 -name '*.gguf' 2>/dev/null)
    return 0
}

write_mark() {   # size and mtime of everything start will read
    mkdir -p "$STATE_DIR"
    : >"$MARK_FILE"
    local p
    while read -r p; do printf '%s\t%s\t%s\n' "$p" "$(fsize "$RELEASE_DIR/$p")" "$(fmtime "$RELEASE_DIR/$p")" >>"$MARK_FILE"; done < <(part_names)
    local f
    for f in "${SIDECAR_FILES[@]}"; do printf '%s\t%s\t%s\n' "$f" "$(fsize "$RELEASE_DIR/$f")" "$(fmtime "$RELEASE_DIR/$f")" >>"$MARK_FILE"; done
}

mark_matches() {   # the recorded bytes are still the bytes on disk
    [ -s "$MARK_FILE" ] || return 1
    local tmp; tmp=$(mktemp); write_mark_to "$tmp"
    local rc=0
    cmp -s "$tmp" "$MARK_FILE" || rc=1
    rm -f "$tmp"
    return "$rc"
}

write_mark_to() {   # write_mark with a destination, for comparison
    local dest="$1" p
    : >"$dest"
    while read -r p; do printf '%s\t%s\t%s\n' "$p" "$(fsize "$RELEASE_DIR/$p")" "$(fmtime "$RELEASE_DIR/$p")" >>"$dest"; done < <(part_names)
    local f
    for f in "${SIDECAR_FILES[@]}"; do printf '%s\t%s\t%s\n' "$f" "$(fsize "$RELEASE_DIR/$f")" "$(fmtime "$RELEASE_DIR/$f")" >>"$dest"; done
}

# --------------------------------------------------------------------------
# Hugging Face CLI
# --------------------------------------------------------------------------
ensure_hf() {
    if [ -n "${HF_BIN:-}" ]; then HF="$HF_BIN"; return; fi
    if command -v hf >/dev/null 2>&1 && hf download --help >/dev/null 2>&1; then HF=hf; return; fi
    if command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli; return; fi
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
    local -a rev=() includes=(--include "*.gguf")
    [ -n "$REVISION" ] && rev=(--revision "$REVISION")
    local f
    for f in "${SIDECAR_FILES[@]}" "${EXTRA_FILES[@]}"; do [ "$f" = SHA256SUMS ] || includes+=(--include "$f"); done
    includes+=(--include SHA256SUMS)
    "$HF" download "$REPO" "${rev[@]}" "${includes[@]}" \
        --local-dir "$RELEASE_DIR" --max-workers "$MAX_WORKERS"
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
        warn "only ${t} MB of memory; this artifact's 1M plan is 113.95 GiB of the 117.2 GiB the engine sees on GB10"; fatal=1
    fi
    say "memory available now: $(avail_mb) MB (start refuses below ${START_FLOOR_MB} MB)"

    # The roomy volume holds the release; the fast device holds the tier.
    mkdir -p "$MODEL_ROOT" 2>/dev/null || { warn "cannot create $MODEL_ROOT"; fatal=1; }
    local need=0 free
    release_ok || need=$((PAYLOAD_BYTES + WORK_ROOM_BYTES))
    free=$(df_avail "$MODEL_ROOT")
    if [ -n "$free" ] && [ "$free" -gt "$need" ]; then
        say "release volume ok: $(human "$PAYLOAD_BYTES") payload, $(human $need) needed for a fresh fetch, $(human "$free") free on $(df_target "$MODEL_ROOT")"
    else
        warn "not enough room on $(df_target "$MODEL_ROOT"): want $(human $((PAYLOAD_BYTES + WORK_ROOM_BYTES))) for a fresh fetch, have $(human "${free:-0}")"
        warn "point MINIMAX31_RELEASE_DIR= or MINIMAX31_MODEL_ROOT= at a volume with room (the payload is 107.59 GiB)"; fatal=1
    fi

    mkdir -p "$KV_DIR" 2>/dev/null || { warn "cannot create $KV_DIR"; fatal=1; }
    local kvtier=$KV_TIER_BYTES_1M kvfree
    [ -n "$CTX" ] && [ "$CTX" -le 131072 ] && kvtier=0
    kvfree=$(df_avail "$KV_DIR")
    if [ "$kvtier" = 0 ]; then
        say "kv tier: not used at $( [ -n "$CTX" ] && echo "context $CTX" || echo "this context" ): every K/V page stays resident"
    elif [ -n "$kvfree" ] && [ "$kvfree" -gt "$kvtier" ]; then
        say "kv tier ok: $KV_DIR on $(df_target "$KV_DIR"), $(human "$kvfree") free, the tier reserves $(human "$kvtier") at $( [ -n "$CTX" ] && echo "$CTX" || echo "the native 1048576") tokens"
    else
        warn "the K/V tier needs $(human "$kvtier") free on $(df_target "$KV_DIR"), which has $(human "${kvfree:-0}")"
        warn "the tier is fallocated before the port binds, so this is a start failure rather than a slow path"
        warn "point MINIMAX31_KV_DIR= at the NVMe, or serve a context at or below 131072 so no tier file is created"; fatal=1
    fi

    if [ -f "$RELEASE_DIR/release.json" ]; then
        local id; id=$(release_identity)
        case "$id" in
            pinned) say "release ok: the pinned release.json is in $RELEASE_DIR ($(parts_present)/$PART_COUNT parts, $(sidecars_present)/${#SIDECAR_FILES[@]} sidecars)";;
            *)      warn "$RELEASE_DIR/release.json is not this artifact's ($id); download would land beside it wrongly - point MINIMAX31_RELEASE_DIR elsewhere"; fatal=1;;
        esac
        local stray; stray=$(stray_gguf | head -5)
        if [ -n "$stray" ]; then
            warn "these .gguf files are not declared by release.json, and the engine refuses a directory that holds one:"
            printf '%s\n' "$stray" | sed 's/^/  /' >&2
            fatal=1
        fi
        local wrong; wrong=$(parts_wrong_size | head -5)
        if [ -n "$wrong" ]; then
            warn "parts present at the wrong size (incomplete or a different revision):"
            printf '%s\n' "$wrong" | sed 's/^/  /' >&2
            say "download re-fetches exactly those"
        fi
        [ "$(parts_partial)" -gt 0 ] && say "$(parts_partial) partial file(s) in flight; a fetch resumes from them"
    else
        say "no release yet in $RELEASE_DIR ($(human "$PAYLOAD_BYTES") of payload to fetch, $(human "$LISTING_BYTES") of listing)"
    fi

    if [ -x "$SERVER_BIN" ]; then
        say "runtime ok: $SERVER_BIN"
        local help; help=$("$SERVER_BIN" --help 2>&1 || true)
        case "$help" in
            *--mtp-mode*) ;;
            *) warn "$SERVER_BIN does not advertise --mtp-mode; rebuild it: cd $ENGINE_SRC && make cuda-spark -- --branch $ENGINE_BRANCH"; fatal=1;;
        esac
        if [ -d "$ENGINE_SRC/.git" ]; then
            local tip; tip=$(git -C "$ENGINE_SRC" log -1 --format='%h %s' 2>/dev/null || echo unknown)
            say "engine checkout: $ENGINE_SRC at $tip"
            if grep -rqs "minimax_m3_vq" "$ENGINE_SRC/crates/ds4-core/src/" 2>/dev/null; then
                say "engine family ok: the checkout carries minimax_m3_vq (draft PR #110)"
            else
                warn "this checkout has no minimax_m3_vq family: the load fails at identify. Fetch and check out $ENGINE_BRANCH (read here at $ENGINE_REF) and rebuild."
                fatal=1
            fi
        fi
        say "engine floor: the family is a format floor, not a date. An older build rejects the release at identify rather than misreading it."
    else
        warn "no executable at $SERVER_BIN; build it with: cd $ENGINE_SRC && git checkout $ENGINE_BRANCH && make cuda-spark"; fatal=1
    fi

    if command -v curl >/dev/null 2>&1; then
        local busy=() p
        for p in 8000 8003 8005 8006 8007 8008 8009 8010; do
            curl -fsS --max-time 2 "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && busy+=("$p")
        done
        [ "${#busy[@]}" -gt 0 ] && warn "these ports already serve a model: ${busy[*]} - the engine holds one global lock"
        if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            warn "port $PORT already answers /v1/models; pick another with MINIMAX31_PORT="; fatal=1
        fi
    fi
    local owners; owners=$(other_owners)
    [ -n "$owners" ] && warn "a ds4 owner is already running: $owners"

    [ "$fatal" = 0 ] && say "preflight: this machine can run the artifact" || return 1
}

# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
cmd_plan() {
    say "release dir:   $RELEASE_DIR   ($(parts_present)/$PART_COUNT parts, $(sidecars_present)/${#SIDECAR_FILES[@]} sidecars, identity $(release_identity))"
    say "kv tier:       $KV_DIR   (DS4_KV_OFFLOAD_DIR; the engine's default would be ~/.cache/ds4/kv-offload)"
    say "state dir:     $STATE_DIR   (logs, pid files, the verification record)"
    say "runtime:       $SERVER_BIN"
    say "serve:         $HOST:$PORT, model id $MODEL_ID, watchdog below ${KILL_MB} MB"
    say ""
    say "artifacts"
    say "  revision     $REPO @ ${REVISION:-main}"
    say "  parts        $PART_COUNT GGUF files, $(human "$PAYLOAD_BYTES")   $([ "$(parts_present)" -eq "$PART_COUNT" ] && echo present || echo "$(parts_present) on disk")"
    say "               the engine needs them as one directory: they are independent GGUFs, not splits"
    say "  37 R13       $LAYER_SMALL_BYTES bytes for layers 03-09, $LAYER_LARGE_BYTES for the other 30"
    say "  20 R13+8     $LAYER_RESIDUAL_BYTES bytes: layers $R13P8_LAYERS"
    say "  body         $BODY $BODY_BYTES"
    say "  read at run  $EMBED $EMBED_BYTES (rows per token), $VISION $VISION_BYTES (per image or video)"
    say "  drafter      $DSPARK $DSPARK_BYTES"
    say "  sidecars     ${SIDECAR_FILES[*]}"
    say ""
    say "where the bytes live"
    say "  resident     $(human "$RESIDENT_BYTES") on $(df_target "$RELEASE_DIR") (read once at load, then on the device)"
    say "  run time     $(human "$SIDECAR_BYTES") beside it, in the same directory: the engine reads every part from there"
    say "  kv tier      $(human "$KV_TIER_BYTES_1M") at the native 1M on $(df_target "$KV_DIR"), unlinked and fallocated at bank creation"
    say "  memory       the card's 1M plan is 113.95 GiB of the 117.2 GiB the engine sees; the drafter adds 2.07 GiB of that"
    say ""
    say "the command start would run:"
    say "  DS4_KV_OFFLOAD_DIR=$KV_DIR $SERVER_BIN --cuda \\"
    say "      -m $RELEASE_DIR \\"
    say "      --mtp-mode $MTP_MODE --mtp-draft $DRAFT --max-seqs $MAX_SEQS \\"
    say "      --host $HOST --port $PORT --model-id $MODEL_ID${CTX:+ -c $CTX}"
    say "  (no -c unless MINIMAX31_CTX is set: the family follows the model's native 1048576)"
    say "  (--mtp-mode is explicit because auto resolves to off for this family)"
    say "  (--max-seqs $MAX_SEQS: one bank, because the 1M plan leaves 3.25 GiB of headroom)"
    say ""
    local need=0; release_ok || need=$((PAYLOAD_BYTES + WORK_ROOM_BYTES))
    say "still to fetch: $(human "$need"); free on target $(df_target "$MODEL_ROOT"): $(human "$(df_avail "$MODEL_ROOT")")"
    say "ports 8000, 8003, 8005, 8006, 8007, 8008, 8009 and 8010 answer for the other launchers in this home"
}

# --------------------------------------------------------------------------
# download, verify, check, build
# --------------------------------------------------------------------------
cmd_download() {
    mkdir -p "$RELEASE_DIR" "$STATE_DIR"
    if release_ok && mark_matches; then
        say "already downloaded and verified: $PART_COUNT parts in $RELEASE_DIR"
        return 0
    fi
    ensure_hf
    local free; free=$(df_avail "$MODEL_ROOT")
    [ -z "$free" ] || [ "$free" -gt "$WORK_ROOM_BYTES" ] \
        || die "$MODEL_ROOT has $(human "${free:-0}") free, needs at least $(human "$WORK_ROOM_BYTES") of working room"
    say "fetching $REPO${REVISION:+ @ $REVISION} into $RELEASE_DIR"
    say "the parts are independent GGUFs (each expert layer is its own file), so an interrupted run resumes file by file"
    hf_download || die "download failed; run the same command again to resume"

    local stray; stray=$(stray_gguf | head -5)
    [ -z "$stray" ] || die "the fetch left a .gguf that release.json does not declare; the engine refuses such a directory:
$(printf '%s\n' "$stray" | sed 's/^/  /')"
    if [ -f "$RELEASE_DIR/release.json" ]; then
        local id; id=$(release_identity)
        [ "$id" = "pinned" ] || die "release.json in $RELEASE_DIR is not the pinned artifact ($id). This is a different release; point MINIMAX31_RELEASE_DIR= elsewhere rather than mixing the two."
    fi
    release_ok || die "the fetch finished but the release is incomplete: $(parts_present)/$PART_COUNT parts, $(sidecars_present)/${#SIDECAR_FILES[@]} sidecars, $(parts_partial) partial"
    say "done: $PART_COUNT parts, $(human "$(du -sb "$RELEASE_DIR" | cut -f1)") on disk, release.json $(sha_file "$RELEASE_DIR/release.json" | cut -c1-12)"
    say "next: bash $0 verify (hashes all 115.53 GB), then bash $0 check, then bash $0 start"
}

cmd_verify() {
    release_ok || die "the release is incomplete in $RELEASE_DIR; run download first"
    need sha256sum
    local id; id=$(release_identity)
    [ "$id" = "pinned" ] || die "release.json is not the pinned artifact ($id); the SHA256SUMS below would prove the wrong release"
    say "hashing $(human "$(du -sb "$RELEASE_DIR" | cut -f1)"); this reads every byte"
    # SHA256SUMS lists 224 files and this fetch takes 68 of them, so the check is
    # --ignore-missing made strict: on a directory where none of the listed files
    # exist, sha256sum -c --ignore-missing exits 0 having verified nothing
    # (measured in this repository on 2026-10-18), so the OK count is what counts.
    local out ok failed
    out=$(cd "$RELEASE_DIR" && sha256sum -c --ignore-missing SHA256SUMS 2>&1) || true
    ok=$(printf '%s\n' "$out" | grep -c ': OK$' || true)
    failed=$(printf '%s\n' "$out" | grep -c ': FAILED$' || true)
    if [ "$failed" -gt 0 ]; then
        printf '%s\n' "$out" | grep ': FAILED$' | head -10 | sed 's/^/  /' >&2
        die "$failed file(s) do not match SHA256SUMS; a mismatching part is not this release's"
    fi
    local want=$((PART_COUNT + ${#SUMS_REQUIRED[@]}))
    [ "$ok" -ge "$want" ] || die "only $ok of the $want digest lines checked out; the directory is incomplete (SHA256SUMS carries 224 entries, this fetch takes $((PART_COUNT + ${#SUMS_REQUIRED[@]} + ${#EXTRA_FILES[@]})) of them, and --ignore-missing skips the rest without a word)"
    write_mark
    say "verify: $ok digest lines match SHA256SUMS - the 61 parts and the engine's own sidecars are the pinned revision's"
    say "record: $MARK_FILE ($(wc -l <"$MARK_FILE") entries of size and mtime); start refuses bytes that no longer match it"
}

cmd_check() {
    release_ok || die "the release is incomplete in $RELEASE_DIR; run download first"
    [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN (build it: cd $ENGINE_SRC && make cuda-spark)"
    say "release: $(parts_present)/$PART_COUNT parts, identity $(release_identity), $(human "$PAYLOAD_BYTES")"
    say "check opens and verifies all 61 parts (the engine's own O_DIRECT hashing pass, a full read of 115.53 GB)"
    local -a args=(--cuda --check-config -m "$RELEASE_DIR")
    [ -n "$CTX" ] && args+=(-c "$CTX")
    [ -n "$MAX_SEQS" ] && args+=(--max-seqs "$MAX_SEQS")
    [ "$MTP_MODE" = "off" ] || args+=(--mtp-mode "$MTP_MODE" --mtp-draft "$DRAFT")
    local rc=0
    DS4_KV_OFFLOAD_DIR="$KV_DIR" "$SERVER_BIN" "${args[@]}" || rc=$?
    if [ "$rc" != 0 ]; then
        die "the check failed (exit $rc). If it says the artifact cannot be identified or a tensor type is unsupported,
       this build has no minimax_m3_vq family: fetch $ENGINE_BRANCH (read here at $ENGINE_REF) and
       rebuild with: cd $ENGINE_SRC && make cuda-spark"
    fi
    say "check ok: the engine opened every part, verified its digest against release.json and printed its plan"
    say "record the plan's weights, per-bank, scratch and floor lines in the runbook before trusting them"
}

cmd_build() {
    [ -d "$ENGINE_SRC" ] || die "no engine tree at $ENGINE_SRC; clone Baekpica/ds4-dfm-rs and check out $ENGINE_BRANCH"
    say "building in $ENGINE_SRC with the project's own target (this engine builds with make cuda-spark, not the canonical Rust builder)"
    ( cd "$ENGINE_SRC" && make cuda-spark ) || die "make cuda-spark failed in $ENGINE_SRC"
    [ -x "$SERVER_BIN" ] || die "the build finished but there is no executable at $SERVER_BIN; check MINIMAX31_BIN="
    say "built: $SERVER_BIN"
}

# --------------------------------------------------------------------------
# start, stop, status, logs, test
# --------------------------------------------------------------------------
# The card's 1M plan is 113.95 GiB of the 117.2 GiB the engine sees on GB10, and
# the author measured 114.75 GiB peak on the same shape: two consecutive 5 s
# samples of MemAvailable below KILL_MB stop the server, which beats a box that
# has started swapping and cannot be typed into.
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
    release_ok || die "the release is incomplete in $RELEASE_DIR ($(parts_present)/$PART_COUNT parts); run download first"
    release_identity | grep -q '^pinned$' || die "release.json is not the pinned artifact; run download"
    mark_matches || die "no verification record matches the bytes on disk ($MARK_FILE).
       Run: bash $0 verify
       It hashes the 61 parts and the engine's sidecars (115.53 GB of reads) and records their
       sizes and mtimes; the engine hashes them again itself at open."
    [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN (build it: cd $ENGINE_SRC && make cuda-spark)"
    need curl
    alive && die "already running (pid $(cat "$PID_FILE")); stop it first with $0 stop"

    local owners
    owners=$(other_owners)
    if [ -n "$owners" ] && [ "${MINIMAX31_ALLOW_SHARED:-0}" != "1" ]; then
        printf 'minimax31: another model owner is running:\n%s\n' "$owners" >&2
        die "stop it first, or set MINIMAX31_ALLOW_SHARED=1 to overrule (the engine holds one global lock for the whole box)"
    fi

    local a; a=$(avail_mb)
    [ "$a" -ge "$START_FLOOR_MB" ] \
        || die "only $a MB of memory available, need $START_FLOOR_MB: stop another model first, or lower MINIMAX31_START_FLOOR_MB= knowingly"
    [ "$a" -ge $(( (RESIDENT_BYTES + SIDECAR_BYTES) / 1000000 )) ] \
        || warn "the resident parts alone are $(human $RESIDENT_BYTES) and $(human $SIDECAR_BYTES) more is read per token; $a MB is available now"

    # The tier is fallocated in full before the port binds, so check it here too.
    local kvtier=0
    { [ -z "$CTX" ] || [ "$CTX" -gt 131072 ]; } && kvtier=$KV_TIER_BYTES_1M
    if [ "$kvtier" != 0 ]; then
        mkdir -p "$KV_DIR" || die "cannot create the K/V tier directory $KV_DIR"
        local kvfree; kvfree=$(df_avail "$KV_DIR")
        [ -z "$kvfree" ] || [ "$kvfree" -gt "$kvtier" ] \
            || die "the K/V tier needs $(human "$kvtier") free on $KV_DIR, which has $(human "${kvfree:-0}").
       Point MINIMAX31_KV_DIR= at the fast device, or serve MINIMAX31_CTX=131072 so no tier file is created."
    fi

    local -a args=(--cuda -m "$RELEASE_DIR" --host "$HOST" --port "$PORT" --model-id "$MODEL_ID")
    if [ "$MTP_MODE" = "off" ]; then
        args+=(--mtp-mode off)
    else
        args+=(--mtp-mode "$MTP_MODE" --mtp-draft "$DRAFT")
    fi
    [ -n "$CTX" ] && args+=(-c "$CTX")
    [ -n "$MAX_SEQS" ] && args+=(--max-seqs "$MAX_SEQS")

    mkdir -p "$STATE_DIR" "$KV_DIR"
    export DS4_KV_OFFLOAD_DIR="$KV_DIR"

    if [ "${1:-}" != "--daemon" ]; then
        say "serving $MODEL_ID in the foreground on $HOST:$PORT; the log is also written to $LOG_FILE"
        say "the host hashes 115.53 GB and copies the trunk to the device before it listens: minutes, not seconds"
        env DS4_KV_OFFLOAD_DIR="$KV_DIR" "$SERVER_BIN" "${args[@]}" > >(tee -a "$LOG_FILE") 2>&1 </dev/null &
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

    setsid nohup env DS4_KV_OFFLOAD_DIR="$KV_DIR" "$SERVER_BIN" "${args[@]}" >>"$LOG_FILE" 2>&1 </dev/null &
    local pid=$!
    printf '%s\n' "$pid" >"$PID_FILE"
    say "pid $pid, log $LOG_FILE, tier $KV_DIR"
    say "loading: hashing 115.53 GB, copying $(human "$RESIDENT_BYTES") to the device, fallocating the tier"

    if [ "$WATCHDOG" = 1 ]; then
        export -f watchdog avail_mb
        nohup bash -c "watchdog $pid $KILL_MB '$WD_LOG'" >/dev/null 2>&1 </dev/null &
        printf '%s\n' "$!" >"$WD_PID_FILE"
        say "watchdog pid $! (stops the server if MemAvailable stays below ${KILL_MB} MB, two 5 s samples)"
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

    local post; post=$(avail_mb)
    say "loaded in ${waited}s; MemAvailable $(post) MB against the ${RUN_FLOOR_MB} MB run floor"
    [ "$post" -lt "$RUN_FLOOR_MB" ] && warn "under the run floor: with the drafter this shape measured 114.75 GiB peak on the author's box"
    report_endpoint
}

report_endpoint() {
    local ip; ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    say ""
    say "ready: $MODEL_ID on $HOST:$PORT/v1"
    say "  local   http://127.0.0.1:$PORT/v1"
    say "  lan     http://${ip:-<host>}:$PORT/v1"
    say "  context ${CTX:-native 1048576}   banks ${MAX_SEQS:-auto}   mtp $MTP_MODE/$DRAFT"
    say "  tier    $KV_DIR (DS4_KV_OFFLOAD_DIR)"
    say "  stop    bash $0 stop"
    say "next: bash $0 test, then record the decode rate in runbooks/minimax-m31-flash-vq.md"
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
    say "release:     $(parts_present)/$PART_COUNT parts, $(sidecars_present)/${#SIDECAR_FILES[@]} sidecars in $RELEASE_DIR $([ "$(parts_partial)" = 0 ] && echo '(no partial files)' || echo "($(parts_partial) PARTIAL)")"
    say "identity:    $(release_identity); verification record $([ -s "$MARK_FILE" ] && { mark_matches && echo 'matches the bytes on disk' || echo 'STALE - run verify'; } || echo 'absent - run verify')"
    say "kv tier:     $KV_DIR on $(df_target "$KV_DIR"), $(human "$(df_avail "$KV_DIR")") free, the tier reserves $(human "$KV_TIER_BYTES_1M") at the native context"
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
    # 256 tokens: this template opens a thinking span first, so a short cap returns
    # an empty content field and looks like a failure.
    out=$(curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL_ID\",\"temperature\":0,\"max_tokens\":256,\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, what does a vector-quantized expert layer store?\"}]}")
    case "$out" in
        *'"choices"'*)
            printf '%s\n' "$out" | sed -n 's/.*"timings":{\(.*\)}.*/  timings: \1/p' | head -1
            printf '%s\n' "$out" | sed -n 's/.*"usage":{\(.*\)}.*/  usage: \1/p' | head -1
            printf '%s\n' "$out" | sed -n 's/.*"reasoning_content":"\([^"]\{0,90\}\)".*/  thinking: \1.../p' | head -1
            printf '%s\n' "$out" | sed -n 's/.*"content":"\([^"]\{0,90\}\)".*/  answer: \1.../p' | head -1
            printf '%s\n' "$out" | sed -n 's/.*"finish_reason":"\([^"]*\)".*/  finish: \1/p' | head -1;;
        *) warn "the request failed: $(printf '%s' "$out" | head -c 300)"; return 1;;
    esac
}

# --------------------------------------------------------------------------
# selftest: the script's own contract, checkable without an artifact
# --------------------------------------------------------------------------
selftest() {
    local fail=0 total=0 n=0 p e small=0 large=0 residual=0 layers_seen=0

    while read -r p; do
        n=$((n + 1))
        e=$(part_expected "$p")
        [ "$e" -gt 0 ] || { warn "no expected size for $p"; fail=1; }
        total=$((total + e))
    done < <(part_names)
    [ "$n" = "$PART_COUNT" ] || { warn "part_names yields $n names, not $PART_COUNT"; fail=1; }
    [ "$total" = "$PAYLOAD_BYTES" ] || { warn "the 61 expected sizes sum to $total, not $PAYLOAD_BYTES"; fail=1; }
    say "parts:    $n names, sizes summing to $total bytes ($(human "$total")) $([ "$total" = "$PAYLOAD_BYTES" ] && echo '= PAYLOAD_BYTES' || echo 'MISMATCH')"

    local l; for l in "${LAYERS[@]}"; do
        case "$(part_expected "$l")" in
            "$LAYER_RESIDUAL_BYTES") residual=$((residual + 1));;
            "$LAYER_SMALL_BYTES")    small=$((small + 1));;
            "$LAYER_LARGE_BYTES")    large=$((large + 1));;
            *) warn "$l has no published size"; fail=1;;
        esac
        layers_seen=$((layers_seen + 1))
    done
    say "layers:   $layers_seen MoE layers: $small small R13, $large large R13, $residual R13+8"
    [ "$small" = 7 ] && [ "$large" = 30 ] && [ "$residual" = 20 ] || { warn "the R13/R13+8 split is not 7/30/20"; fail=1; }
    [ "$((small + large + residual))" = 57 ] || { warn "the MoE layer list is not 57 long"; fail=1; }

    local want_residual=0 r; for r in $R13P8_LAYERS; do want_residual=$((want_residual + 1)); done
    [ "$want_residual" = "$residual" ] || { warn "R13P8_LAYERS names $want_residual layers but $residual carry the residual size"; fail=1; }

    [ "$(( $(part_expected "$BODY") + $(part_expected "$DSPARK") ))" -gt 0 ] || { warn "the body or the drafter has no size"; fail=1; }
    local split=$((total - $(part_expected "$EMBED") - $(part_expected "$VISION")))
    [ "$split" = "$RESIDENT_BYTES" ] || { warn "the resident split computes to $split, not $RESIDENT_BYTES"; fail=1; }
    local side=$(( $(part_expected "$EMBED") + $(part_expected "$VISION") ))
    [ "$side" = "$SIDECAR_BYTES" ] || { warn "the read-while-serving split computes to $side, not $SIDECAR_BYTES"; fail=1; }
    say "splits:   resident $split $([ "$split" = "$RESIDENT_BYTES" ] && echo ok || echo MISMATCH), read while serving $side $([ "$side" = "$SIDECAR_BYTES" ] && echo ok || echo MISMATCH)"

    local f found=0
    for f in "${SUMS_REQUIRED[@]}"; do
        found=0
        local s; for s in "${SIDECAR_FILES[@]}"; do [ "$s" = "$f" ] && found=1; done
        [ "$found" = 1 ] || { warn "SUMS_REQUIRED names $f, which the fetch list does not"; fail=1; }
    done
    found=0
    for s in "${SIDECAR_FILES[@]}"; do [ "$s" = "SHA256SUMS" ] && found=1; done
    [ "$found" = 1 ] || { warn "the fetch list does not take SHA256SUMS, so verify would have nothing to check"; fail=1; }
    say "sidecars: ${#SIDECAR_FILES[@]} fetched, ${#SUMS_REQUIRED[@]} of them listed in SHA256SUMS (it cannot list itself)"

    case "$MTP_MODE" in
        off) say "mtp:      off";;
        on) if [ "$DRAFT" -ge 2 ] && [ "$DRAFT" -le 7 ]; then say "mtp:      on with $DRAFT proposals"; else warn "the family accepts 2 to 7 proposals, not $DRAFT"; fail=1; fi;;
        auto) warn "auto resolves to off for this family, so the drafter would never be used"; fail=1;;
        *) warn "unknown mtp mode: $MTP_MODE"; fail=1;;
    esac

    local sum=0
    for ((l = 3; l <= 59; l++)); do sum=$((sum + l)); done
    say "range:    MoE layers 3-59, $layers_seen of them, sum $sum (the layer numbers the release names)"
    [ "$layers_seen" = 57 ] || fail=1

    [ "$fail" = 0 ] && say "selftest: the script's own contract holds" || { warn "selftest: FAILED"; return 1; }
}

usage() { sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; }

# --------------------------------------------------------------------------
# Options (flags mirror the environment variables)
# --------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "${1:-}" in
        --release-dir) RELEASE_DIR="$2"; shift 2;;
        --kv-dir)      KV_DIR="$2"; shift 2;;
        --state-dir)   STATE_DIR="$2"; PID_FILE="$STATE_DIR/minimax31.pid"; LOG_FILE="$STATE_DIR/minimax31.log"
                       WD_PID_FILE="$STATE_DIR/minimax31-watchdog.pid"; WD_LOG="$STATE_DIR/minimax31-watchdog.log"
                       MARK_FILE="$STATE_DIR/verified.txt"; shift 2;;
        --port)        PORT="$2"; shift 2;;
        --host)        HOST="$2"; shift 2;;
        --ctx)         CTX="$2"; shift 2;;
        --max-seqs)    MAX_SEQS="$2"; shift 2;;
        --mtp-mode)    MTP_MODE="$2"; shift 2;;
        --mtp-draft)   DRAFT="$2"; shift 2;;
        --revision)    REVISION="$2"; shift 2;;
        --engine-dir)  ENGINE_SRC="$2"; shift 2;;
        --server-bin)  SERVER_BIN="$2"; shift 2;;
        --hf-bin)      HF_BIN="$2"; shift 2;;
        --start-floor-mb) START_FLOOR_MB="$2"; shift 2;;
        --kill-mb)     KILL_MB="$2"; shift 2;;
        --watchdog)    WATCHDOG="$2"; shift 2;;
        --no-build)    NO_BUILD=1; shift;;
        --no-start)    NO_START=1; shift;;
        --allow-shared) MINIMAX31_ALLOW_SHARED=1; shift;;
        -h|--help)     usage; exit 0;;
        -*)            printf 'unknown option: %s (see --help)\n' "$1" >&2; usage >&2; exit 2;;
        *)             break;;
    esac
done

case "${1:-}" in
    preflight) shift; preflight "$@";;
    selftest)  shift; selftest "$@";;
    plan)      shift; cmd_plan "$@";;
    download)  shift; cmd_download "$@";;
    verify)    shift; cmd_verify "$@";;
    check)     shift; cmd_check "$@";;
    build)     shift; cmd_build "$@";;
    start)     shift; cmd_start "$@";;
    stop)      shift; cmd_stop "$@";;
    status)    shift; cmd_status "$@";;
    logs)      shift; cmd_logs "$@";;
    test)      shift; cmd_test "$@";;
    all)
        shift
        preflight
        cmd_download
        cmd_verify
        if [ ! -x "$SERVER_BIN" ]; then
            [ "${NO_BUILD:-0}" = 1 ] && die "no executable at $SERVER_BIN and --no-build is set"
            cmd_build
        fi
        cmd_check
        [ "${NO_START:-0}" = 1 ] && { say "--no-start: stopping before the server"; exit 0; }
        cmd_start "$@";;
    ""|-h|--help|help) usage;;
    *) printf 'unknown action: %s (see --help)\n' "$1" >&2; usage >&2; exit 2;;
esac
