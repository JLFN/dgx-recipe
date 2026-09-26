#!/usr/bin/env bash
#
# Qwen3.8-Flash-Next Uncensored (Q5 SSD-PLE) + FP8 PLE sidecar on the DGX Spark.
#
# WHERE THIS RUNS
#   On the Spark itself (GB10, aarch64), as the owning user. The model
#   is ~134 GB, so it is downloaded on the box that serves it, not here.
#   Everything it touches lives in $HOME on the Spark.
#
# WHAT IT DOES
#   download  fetch the Uncensored Q5 main GGUF shards and the FP8 PLE sidecar
#   verify    check the published SHA256SUMS for both artifact sets
#   build     make cuda-spark (sm_121), which also links ds4_weight_server
#   start     weight owner, then ds4-server with DS4_QWEN_PLE_DIR set
#   stop      terminate both processes
#   status    pids, port, and the dtype the runtime actually selected
#   logs      tail both logs
#   all       download, verify, build, start
#
# THE POINT OF THE SCRIPT (this is the part the docs leave implicit)
#   The FP8 PLE sidecar is selected only by environment. With DS4_QWEN_PLE_DIR
#   set, the loader in ds4.c opens <override>/ple-manifest.json instead of the
#   in-tree <model-dir>/ple/ple-manifest.json, so the 95 GiB BF16 sidecar never
#   has to be on disk. The selected directory must hold ple-manifest.json, the
#   four ple-fp8-*.bin files and ple-fp8-weight-scale.bf16.bin, or startup is a
#   hard error. The worker log then reports dtype=FP8_E4M3FN; that log line is
#   the pass/fail check the script waits for.
#
# ARTIFACTS (both from the same public repo, no token needed)
#   main     MQ-Q5-SSD-PLE-BF16/Qwen3.8-Flash-Next-Uncensored-MQ-Q5-SSD-PLE-BF16-0000{1,2,3}-of-00003.gguf
#            32.07 + 31.81 + 19.40 GB, verified against MQ-Q5-SSD-PLE-BF16/SHA256SUMS.main
#   sidecar  PLE-FP8/ple-fp8-0000{1..4}-of-00004.bin (4 x 12.80 GB), PLE-FP8/ple-manifest.json,
#            PLE-FP8/ple-fp8-weight-scale.bf16.bin, verified against PLE-FP8/SHA256SUMS
#
# SERVING SHAPE (defaults follow the model card's recommended configuration)
#   The card's canonical command is a two-bank, 196608-context server, and it
#   calls that "the production server shape" behind every throughput number it
#   publishes. That is the default here: CTX=196608 with no --max-seqs override,
#   so the runtime asks for two banks exactly as the card's command does.
#   The artifact itself declares qwen4exp.context_length = 262144 in its GGUF
#   metadata, and the engine repo's FP8 document reports two banks passing short
#   HTTP checks at 262144 (plain text, simultaneous requests, tool-call
#   continuation with KV reuse, image input, zero census or governor faults). On
#   this box the same two-bank request at 262144 was refused with
#   "banks_not_quoted", with the owner holding about 81 GiB resident and
#   23.35 GiB of graph plan per bank. Whether the difference is the engine
#   commit, the owner reserve or the memory baseline at the time is unresolved
#   here, so 262144 is documented as a one-bank shape below.
#   Ladder:
#     1. CTX=196608, no MAX_SEQS  two banks, the card's measured shape (DEFAULT)
#     2. CTX=262144 MAX_SEQS=1    deepest context this box has admitted so far
#     3. CTX=524288 MAX_SEQS=1 SERVER_FORK=0 SERVER_FORK_PARTIAL=0
#        COALESCE_MAX=1 PREFILL_CHUNK=1024 MAXTOK=256   reduced 512K shape
#   start still retries once at --max-seqs 1 when the memory quote refuses the
#   plan, and says so when it does (AUTO_RETRY=0 disables that safety net).
#   bash $0 plan prints the flag/plan shape without loading weights.
#
# USAGE
#   bash start-qwen38-uncensored-fp8.sh download
#   bash start-qwen38-uncensored-fp8.sh verify
#   bash start-qwen38-uncensored-fp8.sh build
#   bash start-qwen38-uncensored-fp8.sh start
#
#   SIDECAR=bf16 bash start-qwen38-uncensored-fp8.sh download   # BF16 instead (95 GiB more)
#   PORT=8005 USE_OWNER=0 bash start-qwen38-uncensored-fp8.sh start   # any free port
#
# Notes
#   - The download is resumable: re-run `download` after an interruption and
#     hf continues the incomplete files.
#   - The repos are public. If you want the CLI to authenticate anyway, export
#     HF_TOKEN in your own shell; the script itself stores no credential.
#   - Binding to 0.0.0.0 matches the other launchers in this home directory.
#     Use HOST_ADDR=127.0.0.1 for a loopback-only server.
#   - The default port is 8003, the same port the existing Qwen3.8-Flash-Next
#     Q4 launcher uses (start-qwen38.sh, engine in $HOME/ds4). Two
#     servers cannot share one port: stop that one first with
#     `bash $HOME/start-qwen38.sh stop`, or set PORT=<other>.
#   - Use a separate KV disk directory per main model and PLE format: BF16 and
#     FP8 snapshots describe different weights and a cross-format restore is
#     rejected. The default KV dir is therefore sidecar-specific.

set -Eeuo pipefail

# --------------------------------------------------------------------------
# Configuration (every value is overridable from the environment)
# --------------------------------------------------------------------------
# A shape profile (196k, 262k, 512k) supplies the context, bank count and the
# knobs that go with it. Anything you set explicitly in the environment wins
# over the profile, so profiles are a shorthand rather than a cage. That intent
# has to be captured before the defaults below blur the difference.
declare -A EXPLICIT_SHAPE=()
for _sv in CTX MAXTOK MAX_SEQS MTP_DRAFT PREFILL_CHUNK COALESCE_MAX \
           COALESCE_MAX_TOKENS SERVER_FORK SERVER_FORK_PARTIAL SERVER_WARM; do
  [ -n "${!_sv-}" ] && EXPLICIT_SHAPE[$_sv]="${!_sv}"
done
unset _sv

REPO_DIR="${REPO_DIR:-$HOME/ds4-dfm-rs}"                                  # checkout with the FP8 PLE support
MODEL_ROOT="${MODEL_ROOT:-$HOME/models/Qwen3.8-Flash-Next-Uncensored-Mixed-Quant-SSD-PLE-GGUF}"
HF_REPO="${HF_REPO:-Baekpica/Qwen3.8-Flash-Next-Uncensored-Mixed-Quant-SSD-PLE-GGUF}"
VARIANT="${VARIANT:-MQ-Q5-SSD-PLE-BF16}"                                          # published packaging directory
SIDECAR="${SIDECAR:-fp8}"                                                         # fp8 | bf16
HF_BIN="${HF_BIN:-}"                                                              # autodetected if empty
HF_MAX_WORKERS="${HF_MAX_WORKERS:-8}"

HOST_ADDR="${HOST_ADDR:-0.0.0.0}"
PORT="${PORT:-8003}"
CTX="${CTX:-196608}"
MAXTOK="${MAXTOK:-32768}"
MTP_DRAFT="${MTP_DRAFT:-2}"
MEM_FLOOR_GB="${MEM_FLOOR_GB:-2}"
MAX_SEQS="${MAX_SEQS:-}"        # empty = leave the bank count to the runtime (auto)
MODEL_ID="${MODEL_ID:-Qwen3.8-Flash-Next-Uncensored-Mixed-Quant}"
PLE_CACHE_MB="${PLE_CACHE_MB:-512}"
PLE_WORKERS="${PLE_WORKERS:-16}"
PREFILL_CHUNK="${PREFILL_CHUNK:-8192}"
KV_DISK_MB="${KV_DISK_MB:-32768}"
USE_OWNER="${USE_OWNER:-1}"                                                       # 1 = ds4_weight_server owner + worker
RESERVE_GB="${RESERVE_GB:-32}"                                                    # owner VMM reserve
SERVER_CONTINUOUS="${SERVER_CONTINUOUS:-1}"
COALESCE_MAX="${COALESCE_MAX:-2}"
COALESCE_MAX_TOKENS="${COALESCE_MAX_TOKENS:-16384}"
COALESCE_WAIT_MS="${COALESCE_WAIT_MS:-20}"
SERVER_WARM="${SERVER_WARM:-1}"
SERVER_FORK="${SERVER_FORK:-1}"
SERVER_FORK_PARTIAL="${SERVER_FORK_PARTIAL:-1}"
AUTO_RETRY="${AUTO_RETRY:-1}"                                                     # retry once at one bank if the quote rejects
PROFILE="${PROFILE:-}"                                                            # 196k | 262k | 512k (see SHAPES below)
VISION="${VISION:-0}"                                                             # 0 off; 1 = use VISION_MMPROJ; or a path
VISION_MMPROJ="${VISION_MMPROJ:-$HOME/ds4/gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf}"
RUNTIME="${RUNTIME:-/tmp/qwen38-unc-fp8}"                                         # logs, pidfiles, weight manifest
NEED_GB="${NEED_GB:-145}"                                                         # free space the download needs

case "$SIDECAR" in
  fp8)
    PLE_DIR="$MODEL_ROOT/PLE-FP8"
    PLE_FORMAT_LABEL="FP8_E4M3FN"
    PLE_REQUIRED=(ple-manifest.json ple-fp8-weight-scale.bf16.bin
                  ple-fp8-00001-of-00004.bin ple-fp8-00002-of-00004.bin
                  ple-fp8-00003-of-00004.bin ple-fp8-00004-of-00004.bin)
    PLE_INCLUDE="PLE-FP8/*"
    ;;
  bf16)
    PLE_DIR="$MODEL_ROOT/$VARIANT/ple"
    PLE_FORMAT_LABEL="BF16"
    PLE_REQUIRED=(ple-manifest.json ple-bf16-00001-of-00004.bin ple-bf16-00002-of-00004.bin
                  ple-bf16-00003-of-00004.bin ple-bf16-00004-of-00004.bin)
    PLE_INCLUDE="$VARIANT/ple/*"
    ;;
  *) echo "SIDECAR must be fp8 or bf16, got '$SIDECAR'" >&2; exit 2 ;;
esac

MODEL_GGUF="$MODEL_ROOT/$VARIANT/Qwen3.8-Flash-Next-Uncensored-$VARIANT-00001-of-00003.gguf"
MODEL_DIR="$MODEL_ROOT/$VARIANT"
KV_DIR="${KV_DIR:-$REPO_DIR/qwen38-uncensored-q5-kv-$SIDECAR}"
OWNER_BIN="$REPO_DIR/ds4_weight_server"
SERVER_BIN="$REPO_DIR/ds4-server"
WEIGHT_MANIFEST="$RUNTIME/qwen38-uncensored-q5.weights.manifest"
OWNER_LOG="$RUNTIME/owner.log"
SERVER_LOG="$RUNTIME/server.log"
OWNER_PIDFILE="$RUNTIME/owner.pid"
SERVER_PIDFILE="$RUNTIME/server.pid"

# --------------------------------------------------------------------------
# Shapes: the three serving profiles this recipe supports
# --------------------------------------------------------------------------
#   196k  two banks, 196,608 context. The model card's canonical command and the
#         shape behind every throughput figure it publishes. Two sequences can be
#         in flight at once. This is the default.
#   262k  one bank, 262,144 context, the artifact's declared GGUF ceiling. Deeper
#         prompts, but concurrent requests serialize. Verified on this setup.
#   512k  one bank, 524,288 context, the reduced YaRN shape the card verified for
#         near-full execution. Output is capped at 256 tokens, MTP is off and
#         warm/fork reuse is disabled. The card makes no long-context quality
#         claim for it, and it has not been run on this box.
set_shape() { # $1 = variable name, $2 = profile value; an explicit env value wins
  local name="$1" value="$2"
  [ -n "${EXPLICIT_SHAPE[$name]:-}" ] && return 0
  printf -v "$name" '%s' "$value"
}

apply_profile() {
  local p="$1" pc pm ps pd pch pco pf pp pw
  case "$p" in
    196k|196K|196)
      PROFILE_DESC="196k: two banks, the card's measured production shape"
      pc=196608; pm=32768; ps=""; pd=2; pch=8192; pco=2; pf=1; pp=1; pw=1 ;;
    262k|262K|262)
      PROFILE_DESC="262k: one bank, the artifact's declared ceiling"
      pc=262144; pm=32768; ps=1; pd=2; pch=8192; pco=2; pf=1; pp=1; pw=1 ;;
    512k|512K|512)
      PROFILE_DESC="512k: one bank, reduced shape (execution verified by the authors, quality not claimed)"
      pc=524288; pm=256; ps=1; pd=0; pch=1024; pco=1; pf=0; pp=0; pw=0 ;;
    *)
      printf 'unknown shape profile %s; use 196k, 262k or 512k\n' "$p" >&2
      exit 2 ;;
  esac
  set_shape CTX "$pc"
  set_shape MAXTOK "$pm"
  set_shape MAX_SEQS "$ps"
  set_shape MTP_DRAFT "$pd"
  set_shape PREFILL_CHUNK "$pch"
  set_shape COALESCE_MAX "$pco"
  set_shape SERVER_FORK "$pf"
  set_shape SERVER_FORK_PARTIAL "$pp"
  set_shape SERVER_WARM "$pw"
}

select_profile() { # $1 = profile from the command line, if any
  if [ "$#" -gt 0 ] && [ -n "${1:-}" ]; then
    apply_profile "$1"
  elif [ -n "$PROFILE" ]; then
    apply_profile "$PROFILE"
  else
    # No shape asked for: apply the default through the same code path, so the
    # label and the knobs cannot disagree, and an explicit variable still wins.
    apply_profile "196k"
    PROFILE_DESC="196k: two banks, the card's measured production shape (default)"
  fi
}

print_shape() {
  say "  shape:        $PROFILE_DESC"
  say "  context:      $CTX    max tokens: $MAXTOK    banks: ${MAX_SEQS:-2 (runtime default)}"
  say "  mtp-draft:    ${MTP_DRAFT:-runtime default}$( [ "${MTP_DRAFT:-1}" = "0" ] && printf ' (disabled)' )"
  say "  prefill:      chunk $PREFILL_CHUNK, coalesce max $COALESCE_MAX"
  say "  reuse:        warm=$SERVER_WARM fork=$SERVER_FORK fork-partial=$SERVER_FORK_PARTIAL"
  local vp
  if vp="$(vision_path)"; then say "  vision:       on, projector $vp"; else say "  vision:       off (set VISION=1 to enable)"; fi
}

# Vision is selected the same way the FP8 sidecar is: by environment alone.
# The artifact declares a vision tower in its own metadata
# (qwen4exp.vision.present = 1, 27 blocks, embedding 1152, output 2560, patch 16,
# merge 2), and the projector that supplies that tower is the base model's
# mmproj, whose clip.* dimensions match those values one for one.
vision_path() {
  case "${VISION:-0}" in
    0|""|off|no) return 1 ;;
    1|on|yes)    printf '%s' "$VISION_MMPROJ" ;;
    *)           printf '%s' "$VISION" ;;
  esac
}

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

say()  { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
step() { printf '\n== %s\n' "$*"; }

have() { command -v "$1" >/dev/null 2>&1; }

free_gb() {
  local path="$1"
  while [ ! -d "$path" ] && [ "$path" != "/" ]; do path="$(dirname "$path")"; done
  df -Pk "$path" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}'
}

port_busy() {
  if have ss; then ss -tln 2>/dev/null | grep -q ":$PORT "; else return 1; fi
}

running() { # $1 = pidfile
  [ -f "$1" ] && kill -0 "$(cat "$1" 2>/dev/null)" 2>/dev/null
}

find_hf() {
  if [ -n "$HF_BIN" ] && [ -x "$HF_BIN" ]; then printf '%s' "$HF_BIN"; return; fi
  for c in $HOME/hfenv/bin/hf "$HOME/.local/bin/hf" /usr/local/bin/hf; do
    [ -x "$c" ] && { printf '%s' "$c"; return; }
  done
  if have hf; then command -v hf; return; fi
  if have huggingface-cli; then command -v huggingface-cli; return; fi
  return 1
}

need_hf() {
  HF_BIN="$(find_hf)" || die "no hf CLI found; install huggingface_hub in $HOME/hfenv or set HF_BIN"
  say "hf CLI: $HF_BIN"
}

preflight_repo() {
  [ -d "$REPO_DIR" ] || die "repo not found: $REPO_DIR"
  [ -f "$REPO_DIR/Makefile" ] || die "not a ds4-dfm-rs checkout (no Makefile): $REPO_DIR"
}

preflight_sidecar_support() {
  # The override lives in ds4.c; a checkout without it cannot select the FP8 sidecar.
  grep -q "DS4_QWEN_PLE_DIR" "$REPO_DIR/ds4.c" 2>/dev/null \
    || die "$REPO_DIR/ds4.c has no DS4_QWEN_PLE_DIR support; this checkout predates the FP8 PLE change (git -C $REPO_DIR pull)"
}

preflight_binaries() {
  [ -x "$SERVER_BIN" ] || die "no $SERVER_BIN; run: bash $0 build"
  if [ "$USE_OWNER" = "1" ]; then
    [ -x "$OWNER_BIN" ] || die "no $OWNER_BIN; run: bash $0 build (cuda targets link it)"
  fi
}

preflight_vision() {
  local vp
  if vp="$(vision_path)"; then
    [ -f "$vp" ] || die "vision enabled but the projector is missing: $vp (set VISION_MMPROJ or VISION=<path>)"
  fi
}

preflight_model() {
  [ -f "$MODEL_GGUF" ] || die "missing main shard 00001: $MODEL_GGUF (run: bash $0 download)"
  local missing=0
  for i in 00001 00002 00003; do
    f="$MODEL_DIR/Qwen3.8-Flash-Next-Uncensored-$VARIANT-$i-of-00003.gguf"
    [ -f "$f" ] || { warn "missing shard: $f"; missing=1; }
  done
  [ "$missing" = 0 ] || die "the three main shards must all be present"
}

preflight_ple() {
  [ -d "$PLE_DIR" ] || die "missing sidecar directory: $PLE_DIR (run: bash $0 download)"
  local f
  for f in "${PLE_REQUIRED[@]}"; do
    [ -f "$PLE_DIR/$f" ] || die "missing sidecar file: $PLE_DIR/$f"
  done
}

# The server's flag surface is assembled from several modules and can change
# between revisions, so ask the binary instead of trusting this file.
SERVER_HELP=""
supports_flag() {
  # --help does not list every accepted flag (--no-update-check is accepted but
  # unlisted), so fall back to the CLI sources when the help text is silent.
  if [ -z "$SERVER_HELP" ]; then
    SERVER_HELP="$(cd "$REPO_DIR" && "$SERVER_BIN" --help 2>&1 || true)"
  fi
  grep -q -- "$1" <<<"$SERVER_HELP" && return 0
  local f
  for f in "$REPO_DIR/crates/ds4-server/src/bin/ds4-server-rs.rs" \
           "$REPO_DIR/crates/ds4-server/src/kv_cli.rs" \
           "$REPO_DIR/crates/ds4-server/src/dist_cli.rs"; do
    [ -f "$f" ] && grep -q -- "\"$1\"" "$f" && return 0
  done
  return 1
}

# Shared argument construction so that `plan` and `start` cannot drift apart.
SERVER_ENV=()
SERVER_ARGS=()
build_server_invocation() { # $1 = max_seqs override ("" = leave to the runtime)
  local max_seqs="$1"
  SERVER_ENV=( "CUDA_VISIBLE_DEVICES=0" "DS4_MEMGOV=observe"
    "DS4_SESSION_GRAPH_FIT=0" "DS4_SESSION_GRAPH_HEADROOM_MB=0"
    "DS4_QWEN_BATCH=1" "DS4_QWEN_PREFILL_CHUNK=$PREFILL_CHUNK"
    "DS4_QWEN_PLE_CACHE_MB=$PLE_CACHE_MB" "DS4_QWEN_PLE_WORKERS=$PLE_WORKERS"
    "DS4_QWEN_PLE_DIR=$PLE_DIR"
    "DS4_SERVER_CONTINUOUS=$SERVER_CONTINUOUS"
    "DS4_SERVER_COALESCE_MAX=$COALESCE_MAX"
    "DS4_SERVER_COALESCE_MAX_TOKENS=$COALESCE_MAX_TOKENS"
    "DS4_SERVER_COALESCE_WAIT_MS=$COALESCE_WAIT_MS"
    "DS4_SERVER_WARM=$SERVER_WARM" "DS4_SERVER_FORK=$SERVER_FORK"
    "DS4_SERVER_FORK_PARTIAL=$SERVER_FORK_PARTIAL" )
  if [ "$USE_OWNER" = "1" ]; then
    SERVER_ENV+=( "DS4_CUDA_WEIGHT_IPC_MANIFEST=$WEIGHT_MANIFEST" "DS4_CUDA_WEIGHT_IPC_SCOPE=base" )
  fi

  SERVER_ARGS=( --cuda -m "$MODEL_GGUF" -c "$CTX" -n "$MAXTOK" --host "$HOST_ADDR" --port "$PORT" )
  opt_pair() { if supports_flag "$1"; then SERVER_ARGS+=( "$1" "$2" ); else warn "this build has no $1; skipping"; fi; }
  opt_bare() { if supports_flag "$1"; then SERVER_ARGS+=( "$1" ); else warn "this build has no $1; skipping"; fi; }
  opt_pair --model-id "$MODEL_ID"
  opt_pair --mem-floor-gb "$MEM_FLOOR_GB"
  if [ "${MTP_DRAFT:-1}" = "0" ]; then
    say "  mtp: disabled for this shape; no --mtp-draft is passed"
  else
    opt_pair --mtp-draft "$MTP_DRAFT"
  fi
  local vp
  if vp="$(vision_path)"; then opt_pair --vision "$vp"; fi
  opt_pair --kv-disk-dir "$KV_DIR"
  opt_pair --kv-disk-space-mb "$KV_DISK_MB"
  if [ -n "$max_seqs" ]; then opt_pair --max-seqs "$max_seqs"; fi
  [ "$USE_OWNER" = "1" ] && opt_bare --no-update-check
  opt_bare --cors
}

wait_for_line() { # $1 = file, $2 = pattern, $3 = timeout seconds, $4 = what
  local file="$1" pattern="$2" timeout="$3" what="$4" waited=0
  while [ "$waited" -lt "$timeout" ]; do
    if [ -f "$file" ] && grep -q -- "$pattern" "$file" 2>/dev/null; then return 0; fi
    sleep 3; waited=$((waited + 3))
    if [ $((waited % 30)) -eq 0 ]; then say "  waiting for $what (${waited}s)"; fi
  done
  return 1
}

tail_log() { # $1 = file
  [ -f "$1" ] || { say "  (no log at $1)"; return; }
  say "  last lines of $1:"
  tail -n 15 "$1" | sed 's/^/    /'
}

# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------
cmd_download() {
  need_hf
  mkdir -p "$MODEL_ROOT" "$RUNTIME"
  local need free
  need="$NEED_GB"
  free="$(free_gb "$MODEL_ROOT")"
  say "model root:   $MODEL_ROOT"
  say "free space:   ${free} GB, need about ${need} GB"
  [ "${free:-0}" -ge "$need" ] || die "not enough free space at $MODEL_ROOT (${free} GB free, ${need} GB needed)"

  step "main GGUF shards (3 files, about 83.3 GB)"
  "$HF_BIN" download "$HF_REPO" \
    --include "$VARIANT/*.gguf" \
    --include "$VARIANT/SHA256SUMS.main" \
    --max-workers "$HF_MAX_WORKERS" \
    --local-dir "$MODEL_ROOT"

  step "PLE sidecar ($SIDECAR, include '$PLE_INCLUDE')"
  "$HF_BIN" download "$HF_REPO" \
    --include "$PLE_INCLUDE" \
    --max-workers "$HF_MAX_WORKERS" \
    --local-dir "$MODEL_ROOT"

  say ""
  say "download finished. Next: bash $0 verify"
}

# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------
cmd_verify() {
  step "main shards against $VARIANT/SHA256SUMS.main"
  [ -f "$MODEL_DIR/SHA256SUMS.main" ] || die "missing $MODEL_DIR/SHA256SUMS.main (re-run download)"
  ( cd "$MODEL_DIR" && sha256sum -c SHA256SUMS.main ) || die "main GGUF checksum mismatch"

  step "sidecar against $SIDECAR sums"
  preflight_ple
  if [ -f "$PLE_DIR/SHA256SUMS" ]; then
    ( cd "$PLE_DIR" && sha256sum -c SHA256SUMS ) || die "sidecar checksum mismatch"
  else
    warn "no SHA256SUMS in $PLE_DIR; only presence was checked"
  fi
  say ""
  say "all published checksums verified for $SIDECAR."
}

# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
cmd_build() {
  preflight_repo
  preflight_sidecar_support
  have cargo || die "cargo not on PATH (the Spark has it at ~/.cargo/bin)"
  mkdir -p "$RUNTIME"
  say "building in $REPO_DIR with make cuda-spark (CUDA_ARCH sm_121)"
  ( cd "$REPO_DIR" && make cuda-spark ) 2>&1 | tee "$RUNTIME/build.log"
  [ -x "$SERVER_BIN" ] || die "build finished but $SERVER_BIN is missing (see $RUNTIME/build.log)"
  if [ "$USE_OWNER" = "1" ]; then
    [ -x "$OWNER_BIN" ] || die "build finished but $OWNER_BIN is missing (see $RUNTIME/build.log)"
  fi
  say ""
  say "built: $SERVER_BIN"
  [ "$USE_OWNER" = "1" ] && say "built: $OWNER_BIN"
}

# --------------------------------------------------------------------------
# test (model-free PLE fixtures in this checkout)
# --------------------------------------------------------------------------
cmd_test() {
  preflight_repo
  mkdir -p "$RUNTIME"
  step "model-free PLE fixtures"
  ( cd "$REPO_DIR" && make test-qwen4exp-ple-compute test-qwen4exp-ple-forward ) 2>&1 | tee "$RUNTIME/test.log"
  if [ "$SIDECAR" = "fp8" ] && [ -d "$PLE_DIR" ]; then
    step "batch fixture against the selected sidecar"
    say "DS4_TEST_PLE_FP8=1 DS4_QWEN_PLE_DIR=$PLE_DIR"
    ( cd "$REPO_DIR" && DS4_TEST_PLE_FP8=1 DS4_QWEN_PLE_DIR="$PLE_DIR" make test-qwen4exp-batch ) 2>&1 | tee -a "$RUNTIME/test.log"
  else
    warn "skipping test-qwen4exp-batch: sidecar directory not present"
  fi
  say ""
  say "see $RUNTIME/test.log"
}

# --------------------------------------------------------------------------
# bench (the server logs one line per rolling call and nothing about timings;
# the timings travel in the HTTP response, which is what this reads)
# --------------------------------------------------------------------------
cmd_bench() {
  local shape="${1:-}"
  [ -n "$shape" ] && apply_profile "$shape"
  local base="${BENCH_BASE:-http://127.0.0.1:$PORT}"
  local n="${BENCH_N:-3}"
  local mt="${BENCH_MAXTOK:-32}"
  local prompt="${BENCH_PROMPT:-Explain in two sentences why the sky is blue.}"
  local i t0 t1 wall resp ttft pre dec tps comp cached ok=0

  say "benchmark: $base, $n sequential requests, max tokens $mt"
  curl -fsS --max-time 5 "$base/v1/models" >/dev/null 2>&1 \
    || die "no endpoint answering at $base (start the server first, or set BENCH_BASE)"
  printf '%s\n' "  request      wall_s    ttft_ms   prefill_tok_s   decode_tok_s   tok_per_step   cached_tok"
  for i in $(seq 1 "$n"); do
    t0="$(date +%s.%N)"
    resp="$(curl -sS --max-time "${BENCH_TIMEOUT:-600}" -X POST "$base/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$mt}")"
    t1="$(date +%s.%N)"
    wall="$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')"
    ttft="$(grep -o '"ttft_ms":[0-9.]*' <<<"$resp" | head -1 | cut -d: -f2)"
    pre="$(grep -o '"prefill_tok_s":[0-9.]*' <<<"$resp" | head -1 | cut -d: -f2)"
    dec="$(grep -o '"decode_tok_s":[0-9.]*' <<<"$resp" | head -1 | cut -d: -f2)"
    tps="$(grep -o '"tok_per_step":[0-9.]*' <<<"$resp" | head -1 | cut -d: -f2)"
    cached="$(grep -o '"cached_tokens":[0-9]*' <<<"$resp" | head -1 | cut -d: -f2)"
    comp="$(grep -o '"completion_tokens":[0-9]*' <<<"$resp" | head -1 | cut -d: -f2)"
    if [ -z "${dec:-}" ]; then
      warn "request $i returned no timings; first 200 bytes: $(printf '%s' "$resp" | head -c 200)"
      continue
    fi
    ok=$((ok + 1))
    printf '  %-12s %-8s %-9s %-15s %-14s %-14s %s\n' \
      "$i" "${wall:-?}" "${ttft:-?}" "${pre:-?}" "$dec" "${tps:-?}" "${cached:-?}"
    printf '%s\n' "$dec" >> "$RUNTIME/bench.decode"
    printf '%s\n' "$pre" >> "$RUNTIME/bench.prefill"
    printf '%s\n' "${ttft:-0}" >> "$RUNTIME/bench.ttft"
    say "               (completion tokens: ${comp:-?})"
  done
  if [ "$ok" -gt 0 ]; then
    say ""
    say "means over $ok requests:"
    awk '{s+=$1; n++} END {if (n) printf "  decode_tok_s   %.1f\n", s/n}' "$RUNTIME/bench.decode"
    awk '{s+=$1; n++} END {if (n) printf "  prefill_tok_s  %.1f\n", s/n}' "$RUNTIME/bench.prefill"
    awk '{s+=$1; n++} END {if (n) printf "  ttft_ms        %.1f\n", s/n}' "$RUNTIME/bench.ttft"
    rm -f "$RUNTIME/bench.decode" "$RUNTIME/bench.prefill" "$RUNTIME/bench.ttft"
  fi
  say ""
  say "note: the first request of a session usually shows the slowest ttft, because the"
  say "PLE sidecar pages for a new prompt are read from SSD; the card documents the same."
}

# --------------------------------------------------------------------------
# shapes (resolved from the same code that start uses, so it cannot drift)
# --------------------------------------------------------------------------
cmd_shapes() {
  local p
  say "Shapes available, resolved from this launcher's own profile code:"
  for p in 196k 262k 512k; do
    say ""
    ( apply_profile "$p"; print_shape )
  done
  say ""
  say "An explicitly set knob overrides its profile value; see 'bash $0 help'."
}

# --------------------------------------------------------------------------
# plan (dry run: validates flags and the model, no weights are loaded)
# --------------------------------------------------------------------------
cmd_plan() {
  select_profile "$@"
  preflight_repo
  preflight_binaries
  preflight_model
  preflight_ple
  preflight_vision
  mkdir -p "$RUNTIME" "$KV_DIR"
  print_shape
  say "this opens no weights and does not replace a real start; the live memory quote"
  say "is computed at startup, not here."
  build_server_invocation "$MAX_SEQS"
  local rc=0
  ( cd "$REPO_DIR" && env "${SERVER_ENV[@]}" ./ds4-server "${SERVER_ARGS[@]}" --print-plan --check-config ) || rc=$?
  say ""
  if [ "$rc" = 0 ]; then
    say "check-config exit 0: the plan may listen"
  else
    say "check-config exit $rc: the plan was rejected (see the lines above)"
  fi
}

# --------------------------------------------------------------------------
# start
# --------------------------------------------------------------------------
cmd_start() {
  select_profile "$@"
  preflight_repo
  preflight_binaries
  preflight_model
  preflight_ple
  preflight_vision
  mkdir -p "$RUNTIME" "$KV_DIR"

  if running "$SERVER_PIDFILE"; then
    die "server already running (pid $(cat "$SERVER_PIDFILE")); use: bash $0 stop"
  fi
  if port_busy; then
    warn "port $PORT already has a listener; stop it or set PORT=<other>"
    die "refusing to start on a busy port"
  fi

  say "model:        $MODEL_GGUF"
  say "sidecar:      $PLE_DIR ($SIDECAR, expecting dtype=$PLE_FORMAT_LABEL)"
  say "kv disk dir:  $KV_DIR"
  say "endpoint:     http://$HOST_ADDR:$PORT"
  print_shape

  if [ "$USE_OWNER" = "1" ]; then
    step "starting weight owner"
    rm -f "$OWNER_LOG"
    # A crashed owner leaves its IPC socket and manifest behind; both are
    # rebuilt on start, and a stale socket would block the rebind.
    rm -f "$WEIGHT_MANIFEST" "$WEIGHT_MANIFEST.sock" 2>/dev/null || true
    ( cd "$REPO_DIR" && nohup ./ds4_weight_server \
        --base "$MODEL_GGUF" \
        --manifest "$WEIGHT_MANIFEST" \
        --backend vmm \
        --scope base \
        --reserve-gb "$RESERVE_GB" \
        --no-repack-iq2-aligned \
        --no-repack-q2k-aligned \
        --repack-q8-aligned \
        </dev/null >"$OWNER_LOG" 2>&1 & echo $! >"$OWNER_PIDFILE" )
    say "  owner pid $(cat "$OWNER_PIDFILE"), log $OWNER_LOG"
    if ! wait_for_line "$OWNER_LOG" "ds4_weight_server: ready manifest=" 900 "the owner to report ready"; then
      tail_log "$OWNER_LOG"
      die "weight owner never reported ready; stopping"
    fi
    say "  owner ready"
  fi

  step "starting ds4-server"
  say "  banking: ${MAX_SEQS:-runtime default, which is two banks for the card shape}"

  launch_server() { # $1 = max_seqs ("" = runtime default)
    rm -f "$SERVER_LOG"
    build_server_invocation "$1"
    ( cd "$REPO_DIR" && nohup env "${SERVER_ENV[@]}" ./ds4-server "${SERVER_ARGS[@]}" \
        </dev/null >"$SERVER_LOG" 2>&1 & echo $! >"$SERVER_PIDFILE" )
    say "  server pid $(cat "$SERVER_PIDFILE"), log $SERVER_LOG"
  }

  await_endpoint() { # 0 up, 1 process exited, 2 timeout
    local waited=0
    while [ "$waited" -lt 900 ]; do
      if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then return 0; fi
      running "$SERVER_PIDFILE" || return 1
      sleep 3; waited=$((waited + 3))
      if [ $((waited % 30)) -eq 0 ]; then say "  waiting (${waited}s)"; fi
    done
    return 2
  }

  launch_server "$MAX_SEQS"
  step "waiting for the endpoint"
  local rc=0
  await_endpoint || rc=$?

  if [ "$rc" -ne 0 ] && grep -q "fitted serving plan rejected" "$SERVER_LOG" 2>/dev/null; then
    say "  the memory quote rejected this shape:"
    grep -E "requested:|effective:|error:|plan=" "$SERVER_LOG" | sed 's/^/    /'
    if [ "$AUTO_RETRY" = "1" ] && [ -z "$MAX_SEQS" ]; then
      say ""
      say "  retrying with --max-seqs 1 (one bank). One bank's graph plan at 262144"
      say "  measured 23.35 GiB, and the owner holds about 81 GiB resident."
      launch_server 1
      rc=0
      await_endpoint || rc=$?
      [ "$rc" = 0 ] && say "  one bank works at ctx=$CTX"
    fi
  fi

  if [ "$rc" -ne 0 ]; then
    # Leaving the owner behind after a refused plan would keep ~83 GB of mapped
    # weights alive, so clean both up before reporting the failure.
    tail_log "$SERVER_LOG"
    say ""
    say "  cleaning up: stopping the worker and the weight owner"
    stop_one "ds4-server" "$SERVER_PIDFILE" >/dev/null 2>&1 || true
    stop_one "ds4_weight_server" "$OWNER_PIDFILE" >/dev/null 2>&1 || true
    if [ "$rc" = 1 ]; then
      die "ds4-server exited while starting. Lower the shape with one of: MAX_SEQS=1 (one bank at $CTX), CTX=196608 (the card's shape, if $CTX was raised), or SERVER_FORK=0 SERVER_FORK_PARTIAL=0"
    fi
    die "server did not answer /v1/models within 900s"
  fi
  say "  serving on port $PORT"

  # The dtype line is the contract: it must name the sidecar we selected.
  local dtype
  dtype="$(grep -o 'dtype=[A-Za-z0-9_]*' "$SERVER_LOG" 2>/dev/null | head -1 || true)"
  if [ -n "$dtype" ]; then
    say "  runtime reports: $dtype"
    if [ "$dtype" != "dtype=$PLE_FORMAT_LABEL" ]; then
      warn "expected dtype=$PLE_FORMAT_LABEL for SIDECAR=$SIDECAR, got $dtype"
    fi
  else
    warn "no dtype= line in $SERVER_LOG; check that the sidecar actually loaded"
  fi
  local eff
  eff="$(grep -o 'effective:.*' "$SERVER_LOG" 2>/dev/null | head -1 || true)"
  [ -n "$eff" ] && say "  $eff"

  say ""
  say "stop with:  bash $0 stop"
  say "check with: bash $0 status"
}

# --------------------------------------------------------------------------
# stop / status / logs
# --------------------------------------------------------------------------
stop_one() { # $1 = name, $2 = pidfile
  local name="$1" pidfile="$2" pid waited=0
  if ! running "$pidfile"; then say "$name: not running"; return; fi
  pid="$(cat "$pidfile")"
  say "$name: sending TERM to $pid"
  kill "$pid" 2>/dev/null || true
  while [ "$waited" -lt 30 ] && kill -0 "$pid" 2>/dev/null; do sleep 1; waited=$((waited + 1)); done
  if kill -0 "$pid" 2>/dev/null; then
    say "$name: still alive after 30s, sending KILL"
    kill -9 "$pid" 2>/dev/null || true
    sleep 2
  fi
  rm -f "$pidfile"
  say "$name: stopped"
}

cmd_stop() {
  # The worker first: it is the client of the owner's weight ranges.
  stop_one "ds4-server" "$SERVER_PIDFILE"
  stop_one "ds4_weight_server" "$OWNER_PIDFILE"
}

cmd_status() {
  if running "$SERVER_PIDFILE"; then
    say "ds4-server:        running (pid $(cat "$SERVER_PIDFILE"))"
  else
    say "ds4-server:        not running"
  fi
  if running "$OWNER_PIDFILE"; then
    say "ds4_weight_server: running (pid $(cat "$OWNER_PIDFILE"))"
  else
    say "ds4_weight_server: not running"
  fi
  say "sidecar selection: $PLE_DIR ($SIDECAR)"
  say "kv disk dir:       $KV_DIR"
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | head -c 400; then
    say ""
    say "endpoint:          answering on port $PORT"
  else
    say "endpoint:          no answer on port $PORT"
  fi
  if [ -f "$SERVER_LOG" ]; then
    say "dtype line:        $(grep -o 'dtype=[A-Za-z0-9_]*' "$SERVER_LOG" 2>/dev/null | head -1 || say 'none')"
    say ""
    say "server log tail:"
    tail -n 5 "$SERVER_LOG" | sed 's/^/  /'
  fi
}

cmd_logs() {
  say "owner log:  $OWNER_LOG"
  [ -f "$OWNER_LOG" ] && tail -n 20 "$OWNER_LOG" | sed 's/^/  /'
  say ""
  say "server log: $SERVER_LOG"
  [ -f "$SERVER_LOG" ] && tail -n 20 "$SERVER_LOG" | sed 's/^/  /'
}

# --------------------------------------------------------------------------
usage() {
  cat <<EOF
Qwen3.8-Flash-Next Uncensored Q5 + $SIDECAR PLE on the DGX Spark

  bash $0 download   fetch main shards and the $SIDECAR sidecar
  bash $0 verify     check the published SHA256SUMS
  bash $0 build      make cuda-spark
  bash $0 test       model-free PLE fixtures
  bash $0 plan       dry-run the flag/plan shape (no weights)
  bash $0 shapes     list the 196k, 262k and 512k shapes this launcher can start
  bash $0 bench      send N requests and print ttft, prefill and decode per request
                     (the server log carries no timings; they travel in the response)
  bash $0 start      owner + ds4-server with DS4_QWEN_PLE_DIR
  bash $0 stop       stop both
  bash $0 status     pids, endpoint, selected dtype
  bash $0 logs       tail both logs
  bash $0 all        download, verify, build, start

Shapes (start and plan take a shape as their first argument)
  bash $0 start 196k      two banks at 196,608: the model card's measured
                          production shape, two sequences in flight. DEFAULT.
  bash $0 start 262k      one bank at 262,144: the artifact's declared GGUF
                          ceiling, deeper prompts, concurrent requests serialize.
  bash $0 start 512k      one bank at 524,288: the card's reduced YaRN shape.
                          Executes, but output is capped at 256 tokens, MTP is
                          off, reuse is off, and no quality claim is made.
  PROFILE=262k bash $0 start     same thing through the environment
  Any single knob still wins over the shape, for example MAXTOK=8192 with 196k.
  bash $0 shapes prints this table resolved from the code, not from prose.

Paths
  repo        $REPO_DIR
  model root  $MODEL_ROOT
  sidecar     $PLE_DIR
  runtime     $RUNTIME (logs, pidfiles)

Overrides (environment)
  PROFILE=196k|262k|512k   the shape, same as the first argument
  VISION=1                 enable image input with VISION_MMPROJ; VISION=<path>
                           sets the projector explicitly. Off by default.
  VISION_MMPROJ=<path>     projector GGUF; on this box the base model's
                           mmproj-Qwen3.8-Flash-Next-Q8_0.gguf matches the
                           artifact's declared vision tower dimensions exactly.
  SIDECAR=fp8|bf16   PORT, HOST_ADDR, CTX, MAXTOK, MTP_DRAFT, MEM_FLOOR_GB
  MAX_SEQS=1         one bank instead of the runtime default
  USE_OWNER=0        single process, no ds4_weight_server
  SERVER_FORK, SERVER_FORK_PARTIAL, SERVER_WARM, COALESCE_MAX,
  COALESCE_MAX_TOKENS, COALESCE_WAIT_MS, AUTO_RETRY=0
  KV_DISK_MB, PLE_CACHE_MB, PLE_WORKERS, PREFILL_CHUNK, NEED_GB, RESERVE_GB
  HF_BIN, HF_MAX_WORKERS, MODEL_ROOT, REPO_DIR, RUNTIME
EOF
}

case "${1:-}" in
  download) shift; cmd_download "$@" ;;
  verify)   shift; cmd_verify "$@" ;;
  build)    shift; cmd_build "$@" ;;
  test)     shift; cmd_test "$@" ;;
  plan)     shift; cmd_plan "$@" ;;
  bench)    shift; cmd_bench "$@" ;;
  shapes)   shift; cmd_shapes "$@" ;;
  start)    shift; cmd_start "$@" ;;
  stop)     shift; cmd_stop "$@" ;;
  status)   shift; cmd_status "$@" ;;
  logs)     shift; cmd_logs "$@" ;;
  all)      shift; cmd_download; cmd_verify; cmd_build; cmd_start ;;
  ""|-h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
