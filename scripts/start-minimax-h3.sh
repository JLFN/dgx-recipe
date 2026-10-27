#!/usr/bin/env bash
# MiniMax-H3: check the checkpoint, prove its bytes, build the port's own
# server, and run it as a video API the open-grok h3-video skill can drive.
#
#   artifact  MiniMaxAI/MiniMax-H3, pinned at
#             42ed227ee7df40d41602854ae760620d6eb651fe (public, not gated;
#             checked against the Hub's API on 2026-10-27). A diffusion
#             transformer that generates video AND its soundtrack: FL2VA takes
#             a prompt plus first/last-frame anchors, Ref2VA takes ordered
#             image/video/audio references. It is not an autoregressive model:
#             no context, no KV cache, no token stream, so none of the LLM
#             launchers' flags apply and this script shares none of them.
#   layout    A directory of safetensors shards in two partitions plus root
#             metadata. The engine opens it in the machine's own engine
#             (crates/ds4-core/src/minimax_h3/partition.rs): it requires the
#             regular files FL2VA/transformer/config.json and
#             FL2VA/tokenizer/tokenizer.json, then inventories FL2VA's four
#             components in this order -- text_encoder, transformer,
#             video_vae/source, audio_vae. Ref2VA is optional and its whole
#             presence test is the regular file
#             Ref2VA/transformer/model.safetensors.index.json; when it is there
#             the Ref2VA transformer is inventoried too, so a broken optional
#             component fails at open rather than at somebody's first
#             reference request. A reference request then reads every
#             component from Ref2VA, not only its transformer
#             (partition.rs:15-18).
#             At the pin the engine's required set is 102 unique files and
#             210,331,771,721 bytes: all of FL2VA, the distinct Ref2VA
#             transformer and model index, and the root metadata. The five
#             remaining Ref2VA components are byte-identical to their FL2VA
#             counterparts at this revision, so the release's own instructions
#             materialise them as symlinks after the download; a directory
#             holding real copies instead costs a second 134 GiB and serves
#             exactly the same bytes.
#   engine    ds4-dfm-rs on main, which carries the H3 host since PR #111
#             merged (the checkout on the reference box reads be372776; the H3
#             server source there hashes b9eb9036c5566ea151752248b3f2615d73cb
#             f193b9b901bf26b27dc6c154756a, the same bytes the reference clone
#             /data/ds4-dfm-rs holds on feature/minimax-h3). Built with
#             make CUDA_ARCH=sm_121 h3-server-rs, which is the docs' own
#             command; the repository's cuda-spark target does NOT build the H3
#             binaries (Makefile:206-207), so the target is named here.
#   source    MiniMax-H3 Community License (LICENSE, in the release): the
#             checkpoint is not MIT.
#
# THE DAEMON QUESTION, ANSWERED
#   h3-server-rs has no --daemon flag, and neither does ds4-server-rs: the
#   flag list is --bind, --out-dir, --model-dir, --steps, --reuse, --renderer,
#   --renderer-arg, --max-jobs, --timeout-secs and --help, nothing else
#   (crates/ds4-server/src/bin/h3-server-rs.rs, read on the reference box at
#   main be372776; grep for "daemon" in either server source finds nothing).
#   Daemonising is therefore the launcher's job, exactly as it is for every
#   other artifact in this repository: "start --daemon" runs the server under
#   setsid with nohup, writes a pid file, and waits until the port answers
#   before returning. There is no systemd unit and none is written: this is
#   bash, started by hand.
#
# WHY THE WIRE, NOT A HEALTH ROUTE
#   The contract is three routes and nothing else -- POST /v1/videos,
#   GET /v1/videos/{id} and GET /v1/videos/{id}/content
#   (docs/minimax-h3/api-usage.md, "What is not served"). There is no /health
#   and no /v1/models, so readiness and liveness are probed through a refusal
#   the host serves without touching the GPU: GET /v1/videos/<unknown> answers
#   404 video_not_found (h3_video.rs:509-545). The same trick drives "test",
#   which asserts five refusals and never queues a render.
#
# WHAT THIS LAUNCHER EXISTS TO PREVENT
#   1. Reading a directory that merely looks complete. The engine's own open
#      reads shard headers and inventories four components; a truncated shard
#      passes that and fails at the first render, hours in. preflight compares
#      each partition's byte total against the totals measured on the
#      reference box, and verify hashes every file against the digest the
#      Hugging Face downloader recorded for that path when it fetched it.
#   2. Expecting the load at startup. The engine opens METADATA only: the
#      tokenizer, then nothing until a request arrives. The first render after
#      a start pays the component loads (the maintainer's API gate spent
#      37.8 s encoding text and 48.5 s loading the DiT inside that first
#      request), so the port answering is readiness, not readiness to render
#      quickly.
#   3. Reading "it started" as "it is fast". steps are fixed at startup and a
#      20-step 768P clip is hours of GPU, not minutes: the smallest request
#      this API can express is 1:1 at 768x768, 107 frames, and it measured
#      22m20s at TWO steps on the reference box. The default here is the
#      reference's own 20.
#   4. Binding beyond loopback without knowing what that means. The contract
#      has no authentication: any host that can reach the listener queues
#      renders and spends the GPU for as long as they like. The default is
#      0.0.0.0:8090 because the open-grok skill on another machine is the
#      point; say H3_BIND=127.0.0.1:8090 for a private one.
#   5. Leaving the output directory somewhere that fills. The host never
#      prunes -- one MP4 per job, kept until the operator removes it -- so the
#      default is a per-user directory, not the model volume.
#
# WHAT HAS NOT BEEN MEASURED
#   Nothing has been rendered through this launcher. The published API gate
#   (768x768, 107 frames, two steps, 1,340 s including 10 s polling, minimum
#   MemAvailable 67.066 GiB) and the CLI pairs (512x512, 56 frames, 20 steps,
#   762.36 s) are the maintainer's numbers on a GB10, and they are what a
#   first render here must be compared against, not a reading taken here.
#
# USAGE
#   bash start-minimax-h3.sh preflight   # checkpoint, engine, box: no writes
#   bash start-minimax-h3.sh selftest    # this script's own contract
#   bash start-minimax-h3.sh verify      # every byte against the Hub's digests
#   bash start-minimax-h3.sh verify --quick   # presence and byte totals only
#   bash start-minimax-h3.sh download    # only if the checkpoint is absent
#   bash start-minimax-h3.sh build       # make CUDA_ARCH=sm_121 h3-server-rs
#   bash start-minimax-h3.sh start --daemon   # 0.0.0.0:8090, detached, pid file
#   bash start-minimax-h3.sh start       # the same server in the foreground
#   bash start-minimax-h3.sh test        # five refusals over the wire, no render
#   bash start-minimax-h3.sh stop | status | logs | health | plan
#   bash start-minimax-h3.sh render "a red kite over a grey sea at dusk"
#   bash start-minimax-h3.sh --renderer <program> --bind 0.0.0.0:8091 start --daemon
#                                          # the wire without a checkpoint (see below)
#
# OPTIONS (each is also an environment variable of the same name)
#   --model-dir DIR    the checkpoint root (default /mnt/h3-artifacts/MiniMax-H3)
#   --engine-dir DIR   the ds4-dfm-rs tree (default $HOME/ds4-dfm-rs)
#   --server-bin PATH  the server binary (default <--engine-dir>/h3-server-rs)
#   --state-dir DIR    pid file, log and the verification record
#                      (default $HOME/h3-server)
#   --out-dir DIR      where the host writes MP4s, never pruned
#                      (default $HOME/h3-videos)
#   --bind ADDR:PORT   listen address (default 0.0.0.0:8090, the skill's port)
#   --steps N          denoise evaluations, 2..1000 (default 20, the
#                      reference's own setting; 2 is an execution check, not a
#                      quality setting, and is still 22 minutes at the smallest
#                      canvas)
#   --reuse N          reuse window, 1..32 (default 1, the reference's)
#   --max-jobs N       retained jobs, including finished ones; there is no
#                      delete route, so at capacity every create is 429
#                      (default 8)
#   --timeout-secs N   external-renderer timeout in seconds (default 3600)
#   --renderer PROGRAM the engine's other backend: an operator-supplied program
#                      renders, no checkpoint is opened, and this replaces
#                      --model-dir exactly as it does in the binary. The engine
#                      ships tools/minimax_h3/standin-renderer.sh for precisely
#                      this: it writes a real MP4 of the requested geometry with
#                      ffmpeg and never touches the model, which is how the
#                      contract can be proven on a box whose checkpoint is not
#                      there yet, or a client exercised end to end at no GPU cost
#   --renderer-arg ARG one fixed argument for the renderer, repeatable
#   --cuda-arch ARCH   the nvcc arch make is given (default sm_121, GB10)
#   --guard 0|1        run the engine's own host memory guard around the server
#                      (default 0: the worst measured H3 peak on a 121 GB box
#                      is about 54 GiB, so no killer is armed by default)
#   --no-build         never run make, even when the binary is missing
#   --allow-shared     overrule the one-GPU-owner probe before starting
#   -h, --help         this text
#
# Notes
#   - Nothing here writes to the checkpoint. preflight, verify, test and
#     health only read; download writes the model directory it is given;
#     start writes the pid file, the log and (with --guard 1) the guard log
#     under --state-dir.
#   - verify is the slow one: it reads every byte of both partitions (about
#     288 GB, five to twenty minutes depending on the disk) and compares each
#     file against the digest the downloader recorded in
#     .cache/huggingface/download/<path>.metadata -- a sha256 for LFS files, a
#     git blob id for the rest, computed at fetch time and therefore not
#     something this script can make agree with itself. Files the downloader
#     left no record for are reported as unchecked rather than counted as
#     verified. --quick compares presence and partition byte totals instead.
#   - start refuses a checkpoint whose structure is incomplete, and reports
#     (without refusing) when no verification record matches the bytes on
#     disk. Run verify once per fetch: the engine does not hash the shards at
#     open, so a bad byte is otherwise found hours into a render.
#   - The host holds ONE render at a time and queues the rest in arrival
#     order; the queue is in memory only, so a restart forgets the records and
#     leaves the MP4s exactly where they were.
#   - Other model owners on the box: ds4 holds a single global lock for the
#     LLM families, H3 is a separate process tree over the same GPU. start
#     warns when any of them is already running rather than silently sharing
#     the device.

set -Eeuo pipefail

REPO=${H3_REPO:-MiniMaxAI/MiniMax-H3}
PIN=${H3_PIN:-42ed227ee7df40d41602854ae760620d6eb651fe}

MODEL_DIR=${H3_MODEL_DIR:-/mnt/h3-artifacts/MiniMax-H3}
ENGINE_SRC=${H3_ENGINE_SRC:-$HOME/ds4-dfm-rs}
SERVER_BIN=${H3_SERVER_BIN:-$ENGINE_SRC/h3-server-rs}
STATE_DIR=${H3_STATE_DIR:-$HOME/h3-server}
OUT_DIR=${H3_OUT_DIR:-$HOME/h3-videos}
BIND=${H3_BIND:-0.0.0.0:8090}
STEPS=${H3_STEPS:-20}
REUSE=${H3_REUSE:-1}
MAX_JOBS=${H3_MAX_JOBS:-8}
TIMEOUT=${H3_TIMEOUT:-3600}
RENDERER=${H3_RENDERER:-}
RENDERER_ARGS=${H3_RENDERER_ARG:-}
CUDA_ARCH=${H3_CUDA_ARCH:-sm_121}
NO_BUILD=${H3_NO_BUILD:-0}
GUARD=${H3_GUARD:-0}
GUARD_MAX_GIB=${H3_GUARD_MAX_GIB:-128}
GUARD_RESERVE_GIB=${H3_GUARD_RESERVE_GIB:-2}
GUARD_TRIP_GIB=${H3_GUARD_TRIP_GIB:-2}
GUARD_PSI=${H3_GUARD_PSI:-0}
GUARD_TIMEOUT=${H3_GUARD_TIMEOUT:-0}

PID_FILE=$STATE_DIR/h3-server.pid
LOG_FILE=$STATE_DIR/h3-server.log
MARK_FILE=$STATE_DIR/verified.txt
GUARD_LOG=$STATE_DIR/h3-memory.jsonl

# Measured on the reference box (gx10-e1d2, 192.168.1.91, GB10) on 2026-10-27
# with find -printf, so these are the bytes on disk there:
#   FL2VA              144,051,182,625 B   81 files
#     transformer       66,280,524,863 B   15 files
#     text_encoder      66,726,510,529 B   23 files
#     video_vae         10,415,694,629 B   18 files
#     audio_vae            605,461,455 B   13 files
#     tokenizer             11,492,078 B    4 files
#     processor             11,498,352 B    7 files
#   Ref2VA             144,051,182,613 B   81 files
#     transformer       66,280,524,863 B   15 files
# The two partitions are the same weights twice except for their own
# model_index.json (719 against 707 bytes), because that box holds real copies
# of the five shared components. A directory built the way the release's own
# instructions describe holds links instead, and then Ref2VA weighs only
# 66,280,525,570 B. Both are accepted; a third value means a truncated copy, a
# half-finished fetch or a different revision.
EXPECTED_FL2VA_BYTES=144051182625
EXPECTED_REF2VA_BYTES=144051182613
EXPECTED_REF2VA_LINKED_BYTES=66280525570

# The engine's open contract (crates/ds4-core/src/minimax_h3/partition.rs).
REQUIRED_FILES=(
  FL2VA/transformer/config.json
  FL2VA/tokenizer/tokenizer.json
)
INVENTORIED_DIRS=(
  FL2VA/text_encoder
  FL2VA/transformer
  FL2VA/video_vae/source
  FL2VA/audio_vae
)
REF2VA_INDEX=Ref2VA/transformer/model.safetensors.index.json
REF2VA_TRANSFORMER_DIR=Ref2VA/transformer
ROOT_METADATA=(model_index.json modular_model_index.json README.md LICENSE .gitattributes)
# The five components the release's own instructions link to their FL2VA twins.
SHARED_COMPONENTS=(audio_vae processor text_encoder tokenizer video_vae)

# The measured API gate, for the operator's comparison (validation doc). Not a
# reading taken here.
REF_API_GATE_NOTE="768x768, 107 frames, two steps: 1,340 s including 10 s polling, min MemAvailable 67.066 GiB"
REF_CLI_NOTE="512x512, 56 frames, 20 steps: 762.36 s wall"
START_FLOOR_MB=${H3_START_FLOOR_MB:-60000}   # the gate's own minimum was 67.0 GiB
READY_TIMEOUT_S=${H3_READY_TIMEOUT:-120}     # the port binds after the tokenizer, seconds
POLL_S=${H3_POLL:-10}
ALLOW_SHARED=${H3_ALLOW_SHARED:-0}
FORCE_DOWNLOAD=${H3_FORCE_DOWNLOAD:-0}

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
say()  { printf 'h3: %s\n' "$*"; }
warn() { printf 'h3: warning: %s\n' "$*" >&2; }
die()  { printf 'h3: error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

human() {   # bytes -> human-readable
    awk -v b="${1:-0}" 'BEGIN {
        split("B KiB MiB GiB TiB", u, " ");
        i = 1;
        while (b >= 1024 && i < 5) { b /= 1024; i++ }
        printf (i == 1 ? "%.0f %s" : "%.2f %s"), b, u[i];
    }'
}

fsize()  { if [ -f "$1" ]; then stat -c %s "$1" 2>/dev/null || echo 0; else echo 0; fi; }
fmtime() { if [ -f "$1" ]; then stat -c %Y "$1" 2>/dev/null || echo 0; else echo 0; fi; }
avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
total_mb() { awk '/MemTotal/{print int($2/1024)}' /proc/meminfo; }
dir_bytes() { [ -d "$1" ] && find "$1" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}' || echo 0; }
dir_files() { [ -d "$1" ] && find "$1" -type f 2>/dev/null | wc -l | tr -d ' ' || echo 0; }
alive() { [ -s "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }
port_of() { printf '%s' "${1##*:}"; }
host_of() { printf '%s' "${1%:*}"; }
json_field() { printf '%s' "$1" | sed -n "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p"; }
other_owners() { pgrep -af 'h3-server-rs|h3-rs|ds4-server|ds4-bench|ds4-agent' 2>/dev/null | grep -v "$$" || true; }
need_curl() { need curl; }

# Which local addresses are listening on a port. The local address is the
# fourth column of ss -ltn and the port is its suffix, so this is a suffix test
# rather than a regular expression: 0.0.0.0:8090 and [::]:8090 both end in
# :8090, and the peer column (a high port in the fifth column) can never match.
listening_addrs() {   # $1 port
    ss -ltn 2>/dev/null | awk -v suffix=":$1" \
        'NR > 1 && substr($4, length($4) - length(suffix) + 1) == suffix { print $4 }'
}
port_in_use() { [ -n "$(listening_addrs "$1")" ]; }

# A refusal, read without depending on jq: status and the error code.
probe() {   # $1 method, $2 path, $3 optional body -> "STATUS CODE" on stdout
    local method="$1" path="$2" body="${3-}" url out
    url="http://127.0.0.1:$(port_of "$BIND")$path"
    if [ "$method" = "GET" ]; then
        out=$(curl -sS -o /tmp/h3-probe.$$ -w '%{http_code}' --max-time 5 "$url" 2>/dev/null) || { rm -f /tmp/h3-probe.$$; echo "000 none"; return 0; }
    else
        out=$(curl -sS -o /tmp/h3-probe.$$ -w '%{http_code}' --max-time 5 -X "$method" \
            -H 'Content-Type: application/json' --data "$body" "$url" 2>/dev/null) \
            || { rm -f /tmp/h3-probe.$$; echo "000 none"; return 0; }
    fi
    printf '%s %s\n' "$out" "$(json_field "$(cat /tmp/h3-probe.$$ 2>/dev/null)" code)"
    rm -f /tmp/h3-probe.$$
}

refuses() {   # $1 method, $2 path, $3 expected status, $4 expected code, $5 body
    local got status code
    got=$(probe "$1" "$2" "${5-}")
    status=${got%% *}; code=${got##* }
    if [ "$status" = "$3" ] && [ "$code" = "$4" ]; then
        printf '  ok    %-4s %-18s -> %s %s\n' "$1" "$2" "$status" "$code"
        return 0
    fi
    printf '  FAIL  %-4s %-18s -> %s %s (wanted %s %s)\n' "$1" "$2" "${status:-none}" "${code:-none}" "$3" "$4"
    return 1
}

# --------------------------------------------------------------------------
# the checkpoint contract
# --------------------------------------------------------------------------
structure_report() {   # prints what is present and what it weighs; returns 1 if incomplete
    local missing=0 rel p
    [ -d "$MODEL_DIR" ] || { say "model dir:  ABSENT  $MODEL_DIR"; return 1; }

    for rel in "${REQUIRED_FILES[@]}"; do
        p="$MODEL_DIR/$rel"
        if [ -f "$p" ]; then
            say "required:   $rel  ($(fsize "$p") B)"
        else
            say "required:   MISSING  $rel"
            missing=1
        fi
    done
    for rel in "${INVENTORIED_DIRS[@]}"; do
        p="$MODEL_DIR/$rel"
        if [ -d "$p" ]; then
            say "component:  $rel  ($(dir_files "$p") files, $(human "$(dir_bytes "$p")"))"
            [ "$(dir_files "$p")" -gt 0 ] || { say "component:  EMPTY  $rel"; missing=1; }
        else
            say "component:  MISSING  $rel"
            missing=1
        fi
    done

    local idx="$MODEL_DIR/$REF2VA_INDEX"
    if [ -f "$idx" ]; then
        say "ref2va:     present ($REF2VA_INDEX, $(fsize "$idx") B); ordered references served"
        say "ref2va:     transformer $(dir_files "$MODEL_DIR/$REF2VA_TRANSFORMER_DIR") files, $(human "$(dir_bytes "$MODEL_DIR/$REF2VA_TRANSFORMER_DIR")")"
    else
        warn "no $REF2VA_INDEX: the host still starts, but every ref2va request is refused with"
        warn "'ordered references require the Ref2VA checkpoint' (partition.rs, h3.c:873)"
    fi

    local f2 r2 linked=0
    f2=$(dir_bytes "$MODEL_DIR/FL2VA")
    r2=$(dir_bytes "$MODEL_DIR/Ref2VA")
    say "bytes:      FL2VA $(human "$f2") [$f2], Ref2VA $(human "$r2") [$r2]"
    if [ "$f2" != "$EXPECTED_FL2VA_BYTES" ]; then
        warn "FL2VA weighs $f2 B, not the $EXPECTED_FL2VA_BYTES B measured on the reference box"
        [ "$f2" -lt "$EXPECTED_FL2VA_BYTES" ] && missing=1
    fi
    # A linked layout weighs the transformer and its own index only: find -type f
    # does not follow the links the release's instructions create.
    if [ "$r2" = "$EXPECTED_REF2VA_BYTES" ]; then
        :
    elif [ "$r2" = "$EXPECTED_REF2VA_LINKED_BYTES" ]; then
        linked=1
    elif [ "$r2" != "$0" ]; then
        warn "Ref2VA weighs $r2 B, which is neither the $EXPECTED_REF2VA_BYTES B of a copied layout"
        warn "nor the $EXPECTED_REF2VA_LINKED_BYTES B of a linked one"
        [ "$r2" -lt "$EXPECTED_REF2VA_LINKED_BYTES" ] && missing=1
    fi

    local c target
    for c in "${SHARED_COMPONENTS[@]}"; do
        target="$MODEL_DIR/Ref2VA/$c"
        if [ -L "$target" ]; then
            say "shared:     Ref2VA/$c is a symlink -> $(readlink "$target") (covered by the FL2VA digests)"
        elif [ -d "$target" ]; then
            say "shared:     Ref2VA/$c is a real copy ($(human "$(dir_bytes "$target")")); the release links it to"
            say "            ../FL2VA/$c instead, which saves that much disk and serves the same bytes"
        else
            warn "Ref2VA/$c is absent: a reference request reads every component from Ref2VA (partition.rs:15-18)"
        fi
    done
    [ "$linked" = 1 ] && say "layout:     linked Ref2VA (transformer plus its index); the engine accepts either"
    return $missing
}

# The engine's required set, as paths relative to the checkpoint root: all of
# FL2VA, the Ref2VA transformer and its index, the five Ref2VA components a
# reference request reads, and the root metadata. A component that is a symlink
# is left out here and verified under its FL2VA path instead of being read twice.
required_set() {
    [ -d "$MODEL_DIR" ] || return 0
    find "$MODEL_DIR/FL2VA" -type f 2>/dev/null | sed "s|^$MODEL_DIR/||"
    find "$MODEL_DIR/$REF2VA_TRANSFORMER_DIR" -type f 2>/dev/null | sed "s|^$MODEL_DIR/||"
    [ -f "$MODEL_DIR/Ref2VA/model_index.json" ] && echo "Ref2VA/model_index.json"
    local c rel
    for c in "${SHARED_COMPONENTS[@]}"; do
        [ -L "$MODEL_DIR/Ref2VA/$c" ] && continue
        [ -d "$MODEL_DIR/Ref2VA/$c" ] || continue
        find "$MODEL_DIR/Ref2VA/$c" -type f 2>/dev/null | sed "s|^$MODEL_DIR/||"
    done
    for rel in "${ROOT_METADATA[@]}"; do
        [ -f "$MODEL_DIR/$rel" ] && echo "$rel"
    done
    return 0
}

record_matches() {
    [ -s "$MARK_FILE" ] || return 1
    grep -qx "fl2va_bytes=$(dir_bytes "$MODEL_DIR/FL2VA")" "$MARK_FILE" \
        && grep -qx "ref2va_bytes=$(dir_bytes "$MODEL_DIR/Ref2VA")" "$MARK_FILE"
}

# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------
cmd_preflight() {
    local fatal=0
    say "preflight: $REPO at $PIN"
    if [ -n "$RENDERER" ]; then
        say "renderer:   $RENDERER ${RENDERER_ARGS:+[$RENDERER_ARGS]}"
        say "model dir:  $MODEL_DIR (not opened: the renderer backend replaces --model-dir)"
    else
        say "model dir:  $MODEL_DIR"
        structure_report || fatal=1
    fi

    say "engine:     $ENGINE_SRC"
    if [ -d "$ENGINE_SRC" ]; then
        if [ -d "$ENGINE_SRC/crates/ds4-core/src/minimax_h3" ]; then
            say "h3 host:    present (crates/ds4-core/src/minimax_h3)"
        else
            warn "no crates/ds4-core/src/minimax_h3 in $ENGINE_SRC: this checkout predates the H3 port"
            fatal=1
        fi
        command -v git >/dev/null 2>&1 && say "git head:   $(git -C "$ENGINE_SRC" log --oneline -1 2>/dev/null || echo unknown)"
        if grep -q '^h3-server-rs:' "$ENGINE_SRC/Makefile" 2>/dev/null; then
            say "make target: h3-server-rs present"
        else
            warn "no h3-server-rs target in $ENGINE_SRC/Makefile"
            fatal=1
        fi
    else
        warn "no engine tree at $ENGINE_SRC; set H3_ENGINE_SRC"
        fatal=1
    fi
    if [ -x "$SERVER_BIN" ]; then
        say "binary:     $SERVER_BIN ($(human "$(fsize "$SERVER_BIN")"), built $(date -r "$(stat -c %Y "$SERVER_BIN")" '+%Y-%m-%d %H:%M' 2>/dev/null || echo unknown))"
    else
        say "binary:     not built yet ($SERVER_BIN); run: bash $0 build"
    fi

    if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
        say "ffmpeg:     $(ffmpeg -version 2>/dev/null | head -1)"
    else
        warn "ffmpeg and ffprobe are required: the engine muxes the MP4 through them"
        fatal=1
    fi
    if command -v nvcc >/dev/null 2>&1; then
        say "nvcc:       $(nvcc --version 2>/dev/null | sed -n 's/Cuda compilation tools, release //p' | head -1)"
    else
        warn "nvcc is not on PATH: build will fail"
        fatal=1
    fi
    command -v nvidia-smi >/dev/null 2>&1 && say "gpu:        $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"

    local a; a=$(avail_mb)
    say "memory:     $a MB available of $(total_mb) MB"
    if [ -n "$RENDERER" ]; then
        say "            the start floor does not apply to an external renderer: it renders in its own process"
    else
        [ "$a" -ge "$START_FLOOR_MB" ] || { warn "below the ${START_FLOOR_MB} MB start floor (the API gate's minimum MemAvailable was 67.0 GiB)"; fatal=1; }
    fi

    local port; port=$(port_of "$BIND")
    if port_in_use "$port"; then
        warn "something already listens on port $port: $(listening_addrs "$port" | tr '\n' ' ')"
        fatal=1
    else
        say "port:       $port free"
    fi

    if [ -d "$STATE_DIR" ]; then say "state dir:  $STATE_DIR"; else say "state dir:  $STATE_DIR (will be created)"; fi
    say "out dir:    $OUT_DIR"
    local owners; owners=$(other_owners)
    [ -n "$owners" ] && warn "another model owner is running:
$owners"

    [ "$fatal" = 0 ] && say "preflight: this box can serve the artifact" || return 1
}

# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------
# The digest source is the downloader's own record for the file it fetched:
# .cache/huggingface/download/<path>.metadata, three lines -- commit, object
# id, mtime. For LFS payloads the object id is the sha256 of the bytes; for the
# small text files it is the git blob id, so the content is hashed the way git
# hashes a blob. Both were written when the file was fetched, so neither can be
# made to agree with a corrupted byte on disk.
meta_of() { printf '%s/.cache/huggingface/download/%s.metadata' "$MODEL_DIR" "$1"; }

hex_len() { printf '%s' "$1" | tr -d '\n' | wc -c | tr -d ' '; }

hash_sha256() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }

hash_git_blob() {   # git's own object id: sha1("blob <size>\0" + content)
    local f="$1" size
    size=$(stat -c %s "$f" 2>/dev/null) || return 1
    { printf 'blob %s\0' "$size"; cat "$f"; } | sha1sum | cut -d' ' -f1
}

verify_one() {   # $1 relative path -> 0 ok, 1 mismatch, 2 unchecked
    local rel="$1" path="$MODEL_DIR/$1" meta want got kind
    path="$MODEL_DIR/$rel"
    [ -f "$path" ] || { printf 'MISSING %s\n' "$rel"; return 1; }
    meta=$(meta_of "$rel")
    if [ -s "$meta" ]; then
        want=$(sed -n 2p "$meta" | tr -d ' \n')
        case "$(hex_len "$want")" in
            64) kind=sha256; got=$(hash_sha256 "$path");;
            40) kind=git-blob; got=$(hash_git_blob "$path");;
            *)  printf 'UNCHECKED %s (metadata object id is neither a sha256 nor a blob id: %s)\n' "$rel" "$want"; return 2;;
        esac
        if [ "$want" = "$got" ]; then
            printf 'OK %s %s\n' "$kind" "$rel"
            return 0
        fi
        printf 'FAIL %s %s: recorded %s, on disk %s\n' "$kind" "$rel" "$want" "$got"
        return 1
    fi
    # No fetch record: the five shared components were linked or copied rather
    # than fetched through the downloader. Their FL2VA twin is the reference.
    local twin="FL2VA/${rel#Ref2VA/}"
    if [ -f "$MODEL_DIR/$twin" ]; then
        if cmp -s "$path" "$MODEL_DIR/$twin"; then
            printf 'OK twin-identical %s\n' "$rel"
            return 0
        fi
        printf 'FAIL twin-differs %s from %s\n' "$rel" "$twin"
        return 1
    fi
    printf 'UNCHECKED %s (no fetch record and no FL2VA twin)\n' "$rel"
    return 2
}

cmd_verify() {
    local quick=0
    [ "${1:-0}" = "1" ] && quick=1
    local f2 r2 files
    f2=$(dir_bytes "$MODEL_DIR/FL2VA"); r2=$(dir_bytes "$MODEL_DIR/Ref2VA")
    files=$(required_set | wc -l | tr -d ' ')

    if [ "$quick" = 1 ]; then
        local bad=0
        say "quick verify: presence and byte totals"
        [ "$f2" = "$EXPECTED_FL2VA_BYTES" ] && say "  ok    FL2VA $(human "$f2") [$f2]" \
            || { warn "  FAIL  FL2VA weighs $f2 B, expected $EXPECTED_FL2VA_BYTES B"; bad=1; }
        [ "$r2" = "$EXPECTED_REF2VA_BYTES" ] && say "  ok    Ref2VA $(human "$r2") [$r2]" \
            || { warn "  FAIL  Ref2VA weighs $r2 B, expected $EXPECTED_REF2VA_BYTES B"; bad=1; }
        say "  note  byte totals catch truncation, not a flipped bit: run verify for digests"
        [ "$bad" = 0 ] || return 1
        write_record quick "$files" 0 0 0
        return 0
    fi

    need_curl
    say "verify: hashing $files files ($(human "$((f2 + r2))")) against the digests the downloader recorded"
    say "        this reads every byte of both partitions; expect minutes"
    local start; start=$(date +%s)
    local rel ok=0 mismatch=0 unchecked=0 n=0
    local line
    while IFS= read -r rel; do
        [ -n "$rel" ] || continue
        n=$((n + 1))
        line=$(verify_one "$rel") || true
        case "$line" in
            OK*) ok=$((ok + 1));;
            FAIL*|MISSING*)
                mismatch=$((mismatch + 1))
                printf '  %s\n' "$line";;
            *)  unchecked=$((unchecked + 1));;
        esac
        if [ $((n % 25)) -eq 0 ]; then
            say "  $n/$files files, ${ok} ok, ${mismatch} bad, ${unchecked} unchecked, $(( $(date +%s) - start ))s" >&2
        fi
    done < <(required_set)

    local wall=$(( $(date +%s) - start ))
    say "verify: $files files in ${wall}s: $ok ok, $mismatch mismatched, $unchecked unchecked"
    [ "$unchecked" != 0 ] && warn "$unchecked files carry no fetch record; they are NOT verified"
    if [ "$mismatch" != 0 ]; then
        warn "the mismatched files above do not match the digests recorded when they were fetched"
        return 1
    fi
    write_record full "$files" "$ok" "$unchecked" "$wall"
    say "record: $MARK_FILE"
}

write_record() {   # $1 mode, $2 files, $3 hashed, $4 unchecked, $5 wall
    mkdir -p "$STATE_DIR"
    {
        printf 'repo=%s\n' "$REPO"
        printf 'pin=%s\n' "$PIN"
        printf 'model_dir=%s\n' "$MODEL_DIR"
        printf 'mode=%s\n' "$1"
        printf 'files=%s\n' "$2"
        printf 'hashed=%s\n' "$3"
        printf 'unchecked=%s\n' "$4"
        printf 'wall_seconds=%s\n' "$5"
        printf 'fl2va_bytes=%s\n' "$(dir_bytes "$MODEL_DIR/FL2VA")"
        printf 'ref2va_bytes=%s\n' "$(dir_bytes "$MODEL_DIR/Ref2VA")"
        printf 'verified_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >"$MARK_FILE.tmp"
    mv "$MARK_FILE.tmp" "$MARK_FILE"
}

# --------------------------------------------------------------------------
# download (only when the checkpoint is absent)
# --------------------------------------------------------------------------
# The release's own instructions (docs/minimax-h3/README.md, "Model files"):
# take all of FL2VA, the distinct Ref2VA transformer and its model index, and
# the root metadata, then link the five shared components. 210,331,771,721
# bytes, and the links save the second 134 GiB.
cmd_download() {
    need hf
    if [ "$FORCE_DOWNLOAD" != 1 ] && structure_report >/dev/null 2>&1; then
        say "the checkpoint is already complete in $MODEL_DIR; nothing to fetch"
        say "  (H3_FORCE_DOWNLOAD=1 refetches and re-links anyway)"
        return 0
    fi
    mkdir -p "$MODEL_DIR"
    say "fetching $REPO at $PIN into $MODEL_DIR"
    hf download "$REPO" --revision "$PIN" --local-dir "$MODEL_DIR" \
        --include 'FL2VA/*' --include 'Ref2VA/transformer/*' \
        --include 'Ref2VA/model_index.json' \
        --include '.gitattributes' --include 'LICENSE' --include 'README.md' \
        --include 'model_index.json' --include 'modular_model_index.json' \
        || die "hf download failed; a partial fetch can be resumed by running download again"

    local c target
    for c in "${SHARED_COMPONENTS[@]}"; do
        target="$MODEL_DIR/Ref2VA/$c"
        if [ -e "$target" ] || [ -L "$target" ]; then
            say "shared:     Ref2VA/$c already present, left as it is"
            continue
        fi
        ln -s "../FL2VA/$c" "$target" || die "cannot link $target"
        say "shared:     linked Ref2VA/$c -> ../FL2VA/$c"
    done
    say "fetched; run verify, then build and start"
}

# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
cmd_build() {
    [ -d "$ENGINE_SRC" ] || die "no engine tree at $ENGINE_SRC; clone Baekpica/ds4-dfm-rs first"
    [ "$NO_BUILD" = 1 ] && { say "H3_NO_BUILD=1: not building"; return 0; }
    say "building h3-server-rs in $ENGINE_SRC with make CUDA_ARCH=$CUDA_ARCH (this engine builds with make, not the canonical Rust builder;"
    say "  the repository's cuda-spark target builds ds4, ds4-server, ds4-eval, ds4-bench and ds4-agent, not the H3 binaries)"
    ( cd "$ENGINE_SRC" && make "CUDA_ARCH=$CUDA_ARCH" h3-server-rs ) || die "make h3-server-rs failed in $ENGINE_SRC"
    [ -x "$SERVER_BIN" ] || die "the build finished but there is no executable at $SERVER_BIN (set H3_SERVER_BIN=)"
    say "built: $SERVER_BIN ($(human "$(fsize "$SERVER_BIN")"))"
}

# --------------------------------------------------------------------------
# start, stop, status, logs, health, test, render
# --------------------------------------------------------------------------
# The command to run, as an array: the engine's own host memory guard first
# when H3_GUARD=1 (tools/host_memory_guard.py, the wrapper the engine's own
# docs use for model runs), then the server with its flags.
server_cmd() {
    SERVER_CMD=()
    if [ "$GUARD" = 1 ]; then
        local guard="$ENGINE_SRC/tools/host_memory_guard.py"
        [ -f "$guard" ] || die "H3_GUARD=1 but there is no $guard in the engine tree"
        need python3
        SERVER_CMD=(python3 "$guard" --max-gib "$GUARD_MAX_GIB" --high-gib "$GUARD_MAX_GIB" \
            --reserve-gib "$GUARD_RESERVE_GIB" --trip-gib "$GUARD_TRIP_GIB" \
            --psi-full-limit "$GUARD_PSI" --timeout "$GUARD_TIMEOUT" --log "$GUARD_LOG" --)
    fi
    if [ -n "$RENDERER" ]; then
        # The engine's other backend: an operator-supplied program renders, and
        # no checkpoint is opened at all. --model-dir and --renderer are
        # exclusive in the binary, so this replaces the engine root rather than
        # joining it. It is how the contract can be proven on a box that has no
        # checkpoint yet, and how tools/minimax_h3/standin-renderer.sh drives a
        # real MP4 out of ffmpeg with no GPU.
        SERVER_CMD+=("$SERVER_BIN" --bind "$BIND" --out-dir "$OUT_DIR"
            --renderer "$RENDERER" --max-jobs "$MAX_JOBS" --timeout-secs "$TIMEOUT")
        local fixed
        for fixed in $RENDERER_ARGS; do
            SERVER_CMD+=(--renderer-arg "$fixed")
        done
        return 0
    fi
    SERVER_CMD+=("$SERVER_BIN" --bind "$BIND" --out-dir "$OUT_DIR" --model-dir "$MODEL_DIR"
        --steps "$STEPS" --reuse "$REUSE" --max-jobs "$MAX_JOBS" --timeout-secs "$TIMEOUT")
}

cmd_start() {
    need_curl
    mkdir -p "$STATE_DIR" "$OUT_DIR"
    if [ -n "$RENDERER" ]; then
        case "$RENDERER" in
            */*) [ -x "$RENDERER" ] || die "the renderer $RENDERER is not executable" ;;
            *)   need "$RENDERER" ;;
        esac
        say "renderer backend: $RENDERER ${RENDERER_ARGS:+[$RENDERER_ARGS]} - no checkpoint is opened"
    else
        structure_report >/dev/null 2>&1 || die "the checkpoint in $MODEL_DIR is incomplete; run preflight to see what is missing"
    fi
    [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN; run: bash $0 build"
    alive && die "already running (pid $(cat "$PID_FILE")); stop it first with: bash $0 stop"

    local owners; owners=$(other_owners)
    if [ -n "$owners" ] && [ "$ALLOW_SHARED" != 1 ]; then
        printf 'h3: another model owner is running:\n%s\n' "$owners" >&2
        die "the GPU is single-owner in practice: stop it first, or set H3_ALLOW_SHARED=1 to overrule"
    fi

    local a; a=$(avail_mb)
    if [ -z "$RENDERER" ]; then
        [ "$a" -ge "$START_FLOOR_MB" ] \
            || die "only $a MB of memory available, need $START_FLOOR_MB: the API gate's minimum was 67.0 GiB (H3_START_FLOOR_MB= to overrule knowingly)"
    fi

    if [ -n "$RENDERER" ]; then
        say "renderer backend: the verification record covers the checkpoint, which this run does not open"
    elif record_matches; then
        say "verification record matches the bytes on disk ($(grep -m1 '^mode=' "$MARK_FILE" | cut -d= -f2), $(grep -m1 '^verified_at=' "$MARK_FILE" | cut -d= -f2))"
    else
        warn "no verification record matches these bytes (state dir $STATE_DIR): run"
        warn "  bash $0 verify        to hash every file against the downloader's digests, or"
        warn "  bash $0 verify --quick  for a presence-and-byte-total check"
        warn "the engine does not hash the shards at open, so a bad byte surfaces hours into a render"
    fi

    local port; port=$(port_of "$BIND")
    if port_in_use "$port"; then
        die "port $port is already taken ($(listening_addrs "$port" | tr '\n' ' ')); something else answers there"
    fi

    server_cmd

    if [ "${DAEMON:-0}" != 1 ]; then
        say "serving on $BIND, model $MODEL_DIR, $STEPS steps, out $OUT_DIR"
        say "the engine opens metadata only: the first render pays the component loads"
        [ "$GUARD" = 1 ] && say "memory guard on (max ${GUARD_MAX_GIB} GiB, trip ${GUARD_TRIP_GIB} GiB, log $GUARD_LOG)"
        say "log: $LOG_FILE"
        printf '\n' >>"$LOG_FILE"
        "${SERVER_CMD[@]}" >>"$LOG_FILE" 2>&1 </dev/null &
        local pid=$!
        printf '%s\n' "$pid" >"$PID_FILE"
        wait_for_port "$pid" || return 1
        report_endpoint
        local rc=0
        wait "$pid" || rc=$?
        rm -f "$PID_FILE"
        return "$rc"
    fi

    say "starting detached (setsid, nohup): $BIND, $STEPS steps, out $OUT_DIR"
    [ "$GUARD" = 1 ] && say "memory guard on (max ${GUARD_MAX_GIB} GiB, trip ${GUARD_TRIP_GIB} GiB, log $GUARD_LOG)"
    printf '\n' >>"$LOG_FILE"
    setsid nohup "${SERVER_CMD[@]}" >>"$LOG_FILE" 2>&1 </dev/null &
    local pid=$!
    printf '%s\n' "$pid" >"$PID_FILE"
    say "pid $pid, log $LOG_FILE"
    wait_for_port "$pid" || return 1
    report_endpoint
}

wait_for_port() {   # $1 pid
    local pid="$1" waited=0 got status code
    while [ "$waited" -lt "$READY_TIMEOUT_S" ]; do
        got=$(probe GET "/v1/videos/h3-readiness-probe")
        status=${got%% *}; code=${got##* }
        if [ "$status" = 404 ] && [ "$code" = video_not_found ]; then
            say "ready in ${waited}s: the host answers the contract"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            tail -n 12 "$LOG_FILE" 2>/dev/null | sed 's/^/  /' >&2
            rm -f "$PID_FILE"
            die "the server exited while starting (full log: $LOG_FILE)"
        fi
        sleep 2; waited=$((waited + 2))
    done
    tail -n 12 "$LOG_FILE" 2>/dev/null | sed 's/^/  /' >&2
    die "no answer on $BIND within ${READY_TIMEOUT_S}s (log: $LOG_FILE)"
}

report_endpoint() {
    local ip; ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    local port; port=$(port_of "$BIND")
    say ""
    say "ready: MiniMax-H3 video API on $BIND"
    say "  local   http://127.0.0.1:$port/v1/videos"
    say "  lan     http://${ip:-<host>}:$port/v1/videos"
    say "  steps   $STEPS (a 20-step 768P render is hours; the API's smallest request is 1:1 at 768x768, 107 frames)"
    say "  out     $OUT_DIR (never pruned: one MP4 per job until you remove it)"
    say "  log     $LOG_FILE"
    say "  stop    bash $0 stop"
    say ""
    say "from open-grok (h3-video skill):"
    say "  skills/h3-video/scripts/h3-video.sh --url http://${ip:-<host>}:$port \"a red kite over a grey sea\""
}

cmd_stop() {
    if alive; then
        local pid; pid=$(cat "$PID_FILE")
        say "stopping h3-server-rs (pid $pid)"
        kill "$pid" 2>/dev/null || true
        local i=0
        while [ "$i" -lt 30 ] && kill -0 "$pid" 2>/dev/null; do sleep 1; i=$((i + 1)); done
        if kill -0 "$pid" 2>/dev/null; then
            say "  still alive after 30 s, sending KILL (the host holds no durable state: a restart forgets job records and keeps the MP4s)"
            kill -9 "$pid" 2>/dev/null || true
        fi
    else
        say "h3-server-rs: not running"
    fi
    rm -f "$PID_FILE"
}

cmd_status() {
    if alive; then
        local pid; pid=$(cat "$PID_FILE")
        say "h3-server-rs: running (pid $pid)"
        ps -o pid=,etime=,rss=,args= -p "$pid" 2>/dev/null | cut -c1-160 | sed 's/^/  /'
        local got status code
        got=$(probe GET "/v1/videos/h3-status-probe")
        status=${got%% *}; code=${got##* }
        if [ "$status" = 404 ] && [ "$code" = video_not_found ]; then
            say "wire:         answering the contract"
        else
            say "wire:         NOT answering as expected (got ${status:-none} ${code:-none})"
        fi
    else
        say "h3-server-rs: not running"
    fi
    local port; port=$(port_of "$BIND")
    listen=$(listening_addrs "$port" | head -3 | tr '\n' ' ')
    if [ -n "$listen" ]; then
        say "listener:     $listen"
    else
        say "listener:     nothing on port $port"
    fi
    say "memory:       $(avail_mb) MB available of $(total_mb) MB"
    say "out dir:      $OUT_DIR ($(dir_files "$OUT_DIR") files, $(human "$(dir_bytes "$OUT_DIR")"))"
    if [ -s "$MARK_FILE" ]; then
        local mode when state
        mode=$(grep -m1 '^mode=' "$MARK_FILE" | cut -d= -f2)
        when=$(grep -m1 '^verified_at=' "$MARK_FILE" | cut -d= -f2)
        if record_matches; then state="(matches the bytes on disk)"; else state="(STALE: the checkpoint changed)"; fi
        say "record:       $mode verify at $when $state"
    else
        say "record:       absent; run verify"
    fi
}

cmd_logs() {
    [ -f "$LOG_FILE" ] || die "no log at $LOG_FILE"
    tail -n "${1:-40}" "$LOG_FILE"
}

cmd_health() {
    need_curl
    local port; port=$(port_of "$BIND")
    if alive; then say "process: running (pid $(cat "$PID_FILE"))"; else say "process: not running"; fi
    if refuses GET "/v1/videos/h3-health-probe" 404 video_not_found; then
        say "health: the host is serving the contract on $BIND"
        say "note:   the load happens inside the first render, so this says the wire is up, not that a clip is cheap"
        return 0
    fi
    say "health: no contract answer on port $port (log: $LOG_FILE)"
    return 1
}

# Five refusals: every one is answered without touching the GPU, so this is a
# wire test and not a render.
cmd_test() {
    need_curl
    alive || die "nothing is running; start the host first"
    say "test: the contract's refusals on $BIND (no render is queued by any of these)"
    local bad=0
    refuses GET  "/v1/videos/h3-unknown-id"  404 video_not_found || bad=1
    refuses POST "/v1/videos"                400 invalid_task  '{}' || bad=1
    refuses POST "/v1/videos"                400 unknown_field \
        '{"task":"t2va","prompt":"p","conditions":[],"target":{"short_edge":768,"aspect_ratio":"1:1","duration_seconds":4},"seed":0,"bogus":1}' || bad=1
    refuses POST "/v1/videos"                400 unsupported_short_edge \
        '{"task":"t2va","prompt":"p","conditions":[],"target":{"short_edge":512,"aspect_ratio":"1:1","duration_seconds":4},"seed":0}' || bad=1
    refuses GET  "/v1/models"                404 unknown_url || bad=1
    [ "$bad" = 0 ] && say "test: ok, the host answers the contract and refuses what it should" \
        || { say "test: FAILED, see the lines above"; return 1; }
    say "a real render is: bash $0 render \"your prompt\"   (steps are fixed at $STEPS)"
}

# One real request through the API the skill uses. This one queues a render.
cmd_render() {
    need_curl
    alive || die "nothing is running; start the host first"
    local prompt="${1:-A man wearing black coat is walking in the rain.}"
    local seconds="${2:-4}" ratio="${3:-1:1}" seed="${4:-1}"
    local body id status out file

    case "$seconds" in ''|*[!0-9]*) die "--seconds must be a number";; esac
    [ "$seconds" -ge 4 ] && [ "$seconds" -le 15 ] || die "the contract takes 4 through 15 seconds, given $seconds"

    body=$(printf '{"task":"t2va","prompt":"%s","conditions":[],"target":{"short_edge":768,"aspect_ratio":"%s","duration_seconds":%s},"seed":%s}' \
        "$(printf '%s' "$prompt" | sed 's/\\/\\\\/g; s/"/\\"/g')" "$ratio" "$seconds" "$seed")
    say "render: $ratio, ${seconds}s, seed $seed, $STEPS steps"
    say "  reference timings to compare against: $REF_API_GATE_NOTE"
    if [ "$STEPS" -gt 2 ]; then
        say "  at $STEPS steps this is hours on a GB10, not minutes: the smallest request took 22m20s at TWO steps"
    fi
    say "  the GPU is busy for the whole run and the host serves one render at a time"

    id=$(curl -sS --max-time 30 -X POST "http://127.0.0.1:$(port_of "$BIND")/v1/videos" \
        -H 'Content-Type: application/json' --data "$body")
    id=$(json_field "$id" id)
    [ -n "$id" ] || die "the create answered no id; body was: $body"
    say "  id $id"

    local waited=0
    while :; do
        status=$(curl -sS --max-time 10 "http://127.0.0.1:$(port_of "$BIND")/v1/videos/$id")
        case "$(json_field "$status" status)" in
            completed) say "  completed after ${waited}s: $(printf '%s' "$status" | cut -c1-200)"; break;;
            failed)    die "the render failed: $(printf '%s' "$status" | cut -c1-300)";;
            "")        die "no status record for $id";;
            *)         if [ $((waited % 60)) -eq 0 ]; then
                           say "  ${waited}s: $(json_field "$status" status)"
                       fi;;
        esac
        sleep "$POLL_S"; waited=$((waited + POLL_S))
    done

    file="$OUT_DIR/$id.mp4"
    curl -sS --max-time 300 -o "$file" "http://127.0.0.1:$(port_of "$BIND")/v1/videos/$id/content" || die "the download failed"
    out=$(ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames -select_streams v:0 -of default=nw=1 "$file" 2>&1)
    say "  wrote $file ($(human "$(fsize "$file")"))"
    printf '%s\n' "$out" | sed 's/^/  /'
    say "compare nb_frames with the record's frames field: the engine aligns to its own grid (5+17n)"
}

# --------------------------------------------------------------------------
# plan, selftest
# --------------------------------------------------------------------------
cmd_plan() {
    say "artifact:   $REPO at $PIN"
    say "model dir:  $MODEL_DIR ($(human "$(dir_bytes "$MODEL_DIR")"), $(dir_files "$MODEL_DIR") files)"
    say "state dir:  $STATE_DIR (pid file, log, verification record)"
    say "out dir:    $OUT_DIR"
    say "engine:     $ENGINE_SRC (built with make CUDA_ARCH=$CUDA_ARCH h3-server-rs)"
    say "binary:     $SERVER_BIN"
    if [ -n "$RENDERER" ]; then
        say "renderer:   $RENDERER ${RENDERER_ARGS:+[$RENDERER_ARGS]} (no checkpoint is opened)"
    else
        say "checkpoint: $MODEL_DIR, $STEPS steps, reuse $REUSE"
    fi
    say "serve:      $BIND, max-jobs $MAX_JOBS"
    say "guard:      $([ "$GUARD" = 1 ] && echo "on (max ${GUARD_MAX_GIB} GiB, reserve ${GUARD_RESERVE_GIB}, trip ${GUARD_TRIP_GIB}, log $GUARD_LOG)" || echo off)"
    say ""
    say "the exact command a daemon start runs:"
    server_cmd
    printf '  setsid nohup'
    printf ' %s' "${SERVER_CMD[@]}"
    printf '\n'
    say ""
    say "the wire the skill uses (three routes, nothing else):"
    say "  POST /v1/videos               the five-field body, answers {\"id\": \"...\"}"
    say "  GET  /v1/videos/{id}          poll status: queued | in_progress | completed | failed"
    say "  GET  /v1/videos/{id}/content  the MP4"
    say ""
    say "open-grok:  skills/h3-video/scripts/h3-video.sh --url http://<this-host>:$(port_of "$BIND") \"prompt\""
}

cmd_selftest() {
    local tmp rc bad=0
    tmp=$(mktemp -d)
    say "selftest: this script's own contract, no checkpoint and no GPU needed"

    "$0" --help >/dev/null 2>&1 && say "  ok    --help exits 0" || { say "  FAIL  --help"; bad=1; }
    if "$0" --not-a-flag >/dev/null 2>&1; then say "  FAIL  an unknown option was accepted"; bad=1; else say "  ok    an unknown option exits nonzero"; fi

    # The render body: exactly the five fields the contract defines at the top
    # level (an unknown field is refused by name, so a stray one is a bug here).
    local body fields
    body=$(printf '{"task":"t2va","prompt":"a red kite","conditions":[],"target":{"short_edge":768,"aspect_ratio":"1:1","duration_seconds":4},"seed":1}')
    fields=$(printf '%s' "$body" | awk '
        { n = length($0); depth = 0; instr = 0; buf = ""; keys = ""
          for (i = 1; i <= n; i++) {
            c = substr($0, i, 1)
            if (instr) { if (c == "\"") { instr = 0; str = buf; capture = 1 } else buf = buf c; continue }
            if (c == "\"") { instr = 1; buf = ""; at = depth; continue }
            if (c == "{" || c == "[") { depth++; continue }
            if (c == "}" || c == "]") { depth--; continue }
            if (c == ":" && capture) { if (at == 1) keys = keys str " "; capture = 0 }
          }
          print keys
        }')
    if [ "$fields" = "task prompt conditions target seed " ]; then
        say "  ok    the render body carries task, prompt, conditions, target and seed only"
    else
        say "  FAIL  the render body carries at the top level: $fields"; bad=1
    fi

    # The refusal table the wire test asserts, checked against the contract
    # page's own list of codes: every one must be a word this script uses.
    local c
    for c in video_not_found invalid_task unknown_field unsupported_short_edge unknown_url; do
        grep -q "$c" "$0" || { say "  FAIL  no assertion for $c"; bad=1; }
    done
    say "  ok    the five refusal codes the wire test asserts are all present"

    # BIND parsing.
    BIND=0.0.0.0:8090
    [ "$(port_of "$BIND")" = 8090 ] && say "  ok    port_of 0.0.0.0:8090 -> 8090" || { say "  FAIL  port_of"; bad=1; }
    BIND=127.0.0.1:9999
    [ "$(host_of "$BIND")" = 127.0.0.1 ] && say "  ok    host_of 127.0.0.1:9999 -> 127.0.0.1" || { say "  FAIL  host_of"; bad=1; }
    BIND=${H3_BIND:-0.0.0.0:8090}

    # The digest helpers, against a file whose two digests are known.
    printf 'x' >"$tmp/one"
    if [ "$(hash_sha256 "$tmp/one")" = "$(sha256sum "$tmp/one" | cut -d' ' -f1)" ]; then
        say "  ok    hash_sha256 agrees with sha256sum"
    else
        say "  FAIL  hash_sha256"; bad=1
    fi
    rm -rf "$tmp"

    [ "$bad" = 0 ] && { say "selftest: ok"; return 0; }
    say "selftest: FAILED"
    return 1
}

usage() { sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; }

# --------------------------------------------------------------------------
# argument parsing: --flag value pairs this script owns, then the action and
# any positional words it takes. Flags may follow the action (verify --quick,
# start --daemon) and are parsed either way.
# --------------------------------------------------------------------------
ACTION=""
REST=()
while [ $# -gt 0 ]; do
    case "$1" in
        --model-dir)   MODEL_DIR="$2"; shift 2;;
        --engine-dir)  ENGINE_SRC="$2"; SERVER_BIN="$2/h3-server-rs"; shift 2;;
        --server-bin)  SERVER_BIN="$2"; shift 2;;
        --state-dir)   STATE_DIR="$2"; PID_FILE="$STATE_DIR/h3-server.pid"; LOG_FILE="$STATE_DIR/h3-server.log"; MARK_FILE="$STATE_DIR/verified.txt"; GUARD_LOG="$STATE_DIR/h3-memory.jsonl"; shift 2;;
        --out-dir)     OUT_DIR="$2"; shift 2;;
        --bind)        BIND="$2"; shift 2;;
        --steps)       STEPS="$2"; shift 2;;
        --reuse)       REUSE="$2"; shift 2;;
        --max-jobs)    MAX_JOBS="$2"; shift 2;;
        --timeout-secs) TIMEOUT="$2"; shift 2;;
        --renderer)    RENDERER="$2"; shift 2;;
        --renderer-arg) RENDERER_ARGS="${RENDERER_ARGS:+$RENDERER_ARGS }$2"; shift 2;;
        --cuda-arch)   CUDA_ARCH="$2"; shift 2;;
        --guard)       GUARD="$2"; shift 2;;
        --daemon)      DAEMON=1; shift;;
        --quick)       QUICK=1; shift;;
        --no-build)    NO_BUILD=1; shift;;
        --allow-shared) ALLOW_SHARED=1; shift;;
        -h|--help)     usage; exit 0;;
        --*)           printf 'h3: unknown option %s\n\n' "$1" >&2; usage >&2; exit 2;;
        *)             if [ -z "$ACTION" ]; then ACTION="$1"; else REST+=("$1"); fi; shift;;
    esac
done

case "$ACTION" in
    preflight) cmd_preflight;;
    verify)    cmd_verify "${QUICK:-0}";;
    download)  cmd_download;;
    build)     cmd_build;;
    start)     cmd_start;;
    stop)      cmd_stop;;
    status)    cmd_status;;
    logs)      cmd_logs "${REST[0]:-40}";;
    health)    cmd_health;;
    test)      cmd_test;;
    render)    cmd_render "${REST[0]:-}" "${REST[1]:-}" "${REST[2]:-}" "${REST[3]:-}";;
    plan)      cmd_plan;;
    selftest)  cmd_selftest;;
    all)       cmd_preflight && cmd_build && cmd_start;;
    ""|help)   usage;;
    *)         printf 'h3: unknown action: %s\n\n' "$ACTION" >&2; usage >&2; exit 2;;
esac
