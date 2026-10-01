#!/usr/bin/env bash
#
# Prism Ternary Bonsai 2 27B (qwen35) on the DGX Spark / ASUS GX10 (GB10, sm_121).
#
# WHERE THIS RUNS
#   On the Spark itself, as the user that owns the model directory. The artifact
#   is fetched there and served there; nothing here runs over the network.
#   Default paths are the Spark's own: the model under ~/models, the engine
#   checkout at ~/ds4-dfm-rs, logs and pidfile under /tmp/bonsai-spark.
#
# WHAT IT DOES
#   download  fetch Ternary-Bonsai-2-27B-PQ2_0.gguf (7.21 GB) from the Hub
#   verify    check the size and the published SHA-256
#   build     make cuda-spark (sm_121a), which builds ds4-c, ds4-server and ds4
#   test      make bonsai-cuda-check (and bonsai-cuda-parity with TEST_PARITY=1)
#   shapes    the context shapes this launcher can start
#   plan      dry-run the engine's own plan at the chosen shape (no weights)
#   budget    how much room there is for banks, and how much context each bank
#             can hold: scripts/bank-budget.sh, answered by the engine's quote
#   start     ds4-server on the chosen shape, native (no Docker)
#   stop      terminate the server this launcher started
#   status    pid, port, the advertised model id and the engine's own plan
#   logs      tail the server log
#   all       download, verify, build, start
#
# THE POINT OF THE SCRIPT (what is easy to get wrong)
#   1. Every CUDA run of this family needs DS4_CUDA_COPY_MODEL=1. The 6.71 GiB
#      map cannot be pinned (RLIMIT_MEMLOCK is 8192 KiB and that is also the
#      hard limit on the reference box), so without it the backend falls back
#      to lazy per-range materialisation and dies part-way through the trunk
#      with "Bonsai matmul failed for blk.<n>.<tensor>". The copy costs about
#      0.7 s and the log then reads "CUDA copying 6.71 GiB model to device
#      memory". This launcher always sets it for a CUDA start (COPY_MODEL=0
#      turns it off deliberately).
#   2. This family is SERIAL. Its serving caps declare BankLane::Serial and
#      bank_support None (crates/ds4-core/src/serving.rs, ModelFamily::Qwen35),
#      so --max-seqs N>1 is refused by name with "qwen35 live serving is
#      serial; --max-seqs N is not available" (code banks_unsupported), and
#      --kv-disk-dir is refused with disk_unsupported. The Spark's memory is
#      not the constraint for this model: at the artifact's declared 262,144
#      ceiling one bank costs about 16.4 GiB and the whole plan about 24 GiB.
#      Use `budget` to see the numbers for this box rather than guessing.
#   3. One ds4 process at a time. The engine takes a single global lock
#      (/tmp/ds4.lock), so a second model process refuses to start by design.
#      This launcher names the process that holds the slot instead of failing
#      obscurely, and refuses to start while one is up.
#
# ARTIFACTS
#   model    prism-ml/Ternary-Bonsai-2-27B-gguf, file Ternary-Bonsai-2-27B-PQ2_0.gguf
#            7206168928 bytes, sha256 3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1
#   engine   the ds4-dfm-rs checkout in this tree's sibling project; `make
#            cuda-spark` is the sm_121 build (Makefile, the sm_121 gencode pair)
#
# USAGE
#   bash start-bonsai-spark.sh download
#   bash start-bonsai-spark.sh verify
#   bash start-bonsai-spark.sh build
#   bash start-bonsai-spark.sh budget
#   bash start-bonsai-spark.sh start              # asks for a shape when interactive
#   bash start-bonsai-spark.sh start 262k         # the declared ceiling
#   bash start-bonsai-spark.sh stop
#
#   MODEL=/path/to/other.gguf bash start-bonsai-spark.sh start 64k
#   PORT=8007 HOST_ADDR=127.0.0.1 bash start-bonsai-spark.sh start
#
# Notes
#   - The download is resumable; re-run `download` after an interruption and hf
#     continues the incomplete file.
#   - The repo is public and needs no token. Export HF_TOKEN yourself if you
#     want the CLI to authenticate anyway; nothing here stores a credential.
#   - Binding to 0.0.0.0 matches the other launchers on the Spark and is what
#     the workstation's open-grok config expects. HOST_ADDR=127.0.0.1 keeps it
#     loopback-only.

set -Eeuo pipefail

# A shape profile supplies the context, the token cap and the prefill chunk, but
# anything set explicitly in the environment wins over the profile. That intent
# has to be captured BEFORE the defaults below blur "unset" into "262144", so it
# is captured here, first.
declare -A SHAPE_EXPLICIT=()
for _sv in CTX MAXTOK PREFILL_CHUNK; do
  [ -n "${!_sv-}" ] && SHAPE_EXPLICIT[$_sv]="${!_sv}"
done
unset _sv

# --------------------------------------------------------------------------
# Configuration (every value is overridable from the environment)
# --------------------------------------------------------------------------
REPO_DIR="${REPO_DIR:-$HOME/ds4-dfm-rs}"                 # the engine checkout
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"                 # where the artifact lives
HF_REPO="${HF_REPO:-prism-ml/Ternary-Bonsai-2-27B-gguf}"
HF_FILE="${HF_FILE:-Ternary-Bonsai-2-27B-PQ2_0.gguf}"
MODEL="${MODEL:-$MODEL_ROOT/$HF_FILE}"
MODEL_BYTES="${MODEL_BYTES:-7206168928}"
MODEL_SHA256="${MODEL_SHA256:-3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1}"
HF_BIN="${HF_BIN:-}"                                     # autodetected if empty
HF_MAX_WORKERS="${HF_MAX_WORKERS:-8}"
NEED_GB="${NEED_GB:-10}"                                 # free space the download needs

HOST_ADDR="${HOST_ADDR:-0.0.0.0}"
PORT="${PORT:-8005}"                                     # 8000-8004 are used on the Spark
CTX="${CTX:-262144}"
MAXTOK="${MAXTOK:-32768}"
BACKEND="${BACKEND:-cuda}"
MEM_FLOOR_GB="${MEM_FLOOR_GB:-1}"
MODEL_ID="${MODEL_ID:-}"                                 # empty = the id the server derives
PREFILL_CHUNK="${PREFILL_CHUNK:-512}"                    # 512 default, 1024 is the family max
CTX_MAX_HINT="${CTX_MAX_HINT:-262144}"                   # the artifact's declared ceiling (qwen35.rs CTX_MAX)
COPY_MODEL="${COPY_MODEL:-1}"                            # DS4_CUDA_COPY_MODEL; see note 1 above
BANKS="${BANKS:-1}"                                      # >1 is refused by this family, by name
PREFLIGHT="${PREFLIGHT:-1}"                              # check the plan before loading weights
WAIT_LISTEN="${WAIT_LISTEN:-600}"                        # seconds to wait for "listening on"
TEST_PARITY="${TEST_PARITY:-0}"                          # 1 also runs the slow CPU parity gate
PROFILE="${PROFILE:-}"                                   # 45k | 64k | 131k | 262k
DEFAULT_PROFILE="${DEFAULT_PROFILE:-262k}"
ASK_SHAPE="${ASK_SHAPE:-1}"
RUNTIME="${RUNTIME:-/tmp/bonsai-spark}"
ENGINE_BIN="$REPO_DIR/ds4-server"
TEST_BIN="$REPO_DIR/ds4-c"
SERVER_LOG="$RUNTIME/server.log"
SERVER_PIDFILE="$RUNTIME/server.pid"
LOCK_FILE="${LOCK_FILE:-/tmp/ds4.lock}"

# --------------------------------------------------------------------------
# Shapes: the context profiles this recipe supports on the Spark
# --------------------------------------------------------------------------
#   45k   the shape the RTX 4070 SUPER runs today (BONSAI.md). Kept because it
#         is the only shape with measured serving numbers behind it.
#   64k   two and a half times that, still far inside the artifact's ceiling.
#   131k  half the declared ceiling.
#   262k  the declared ceiling (crates/ds4-core/src/qwen35.rs, CTX_MAX), and
#         what the Spark's memory buys: about 16.4 GiB of bank, 24 GiB of plan.
apply_profile() {
  local p="$1"
  case "$p" in
    45k|45K|45056)    PROFILE_DESC="45k: the RTX 4070 SUPER shape, the one with measured serving numbers"; CTX=45056;  MAXTOK=8192;  PREFILL_CHUNK=512 ;;
    64k|64K|65536)    PROFILE_DESC="64k: two and a half times the 4070 shape";                                CTX=65536;  MAXTOK=16384; PREFILL_CHUNK=512 ;;
    131k|131K|131072) PROFILE_DESC="131k: half the artifact's declared ceiling";                             CTX=131072; MAXTOK=32768; PREFILL_CHUNK=512 ;;
    262k|262K|262144) PROFILE_DESC="262k: the artifact's declared ceiling (CTX_MAX = 262144), the Spark shape"; CTX=262144; MAXTOK=32768; PREFILL_CHUNK=512 ;;
    *) printf 'unknown shape %s; use 45k, 64k, 131k or 262k\n' "$p" >&2; exit 2 ;;
  esac
  local name
  SHAPE_OVERRIDDEN=0
  for name in CTX MAXTOK PREFILL_CHUNK; do
    if [ -n "${SHAPE_EXPLICIT[$name]:-}" ]; then
      printf -v "$name" '%s' "${SHAPE_EXPLICIT[$name]}"
      SHAPE_OVERRIDDEN=1
    fi
  done
  # The loop above ends on a `&&` list that is false whenever no knob was set
  # explicitly, which would leave this function returning 1 and, under set -e,
  # end the script at the call site with no message.
  return 0
}

select_profile() { # $1 = shape from the command line, if any
  local requested="${1:-}"
  [ -z "$requested" ] && requested="${PROFILE:-}"

  # "auto" is resolved first and picked for the memory free at this moment, so a
  # start never guesses and never has to be told which shape the box can hold.
  if [ "$requested" = "auto" ]; then
    local chosen=""
    chosen="$(auto_shape)" || true
    if [ -z "$chosen" ]; then
      warn "auto: no shape was accepted with the memory free right now"
      die "free memory, or run: bash $0 fits   (it reports what would fit and why)"
    fi
    say "auto: chose $chosen, the deepest shape the engine accepted just now"
    apply_profile "$chosen"
    PROFILE_DESC="$PROFILE_DESC (auto)"
    return 0
  fi

  if [ -n "$requested" ]; then
    apply_profile "$requested"
    return 0
  fi

  # Nothing asked for: ask, when there is a human at the terminal; otherwise fall
  # back to DEFAULT_PROFILE so cron, systemd and pipelines still work.
  local pick="${DEFAULT_PROFILE:-262k}" note="(default)"
  if [ "${ASK_SHAPE:-1}" = "1" ] && [ -t 0 ] && [ -t 1 ]; then
    say ""
    say "Which shape?"
    say "  45k   ctx  45,056    the RTX 4070 SUPER shape, measured serving numbers"
    say "  64k   ctx  65,536    more depth, still far inside the ceiling"
    say "  131k  ctx 131,072    half the declared ceiling"
    say "  262k  ctx 262,144    the declared ceiling, about 24 GiB of plan on this box"
    say "  auto  the deepest of those that the memory free right now will serve"
    say ""
    local answer=""
    read -r -p "shape [$pick]: " answer || true
    if [ -n "${answer:-}" ]; then pick="$answer"; note="(selected)"; fi
    if [ "$pick" = "auto" ]; then
      local chosen=""
      chosen="$(auto_shape)" || true
      if [ -n "$chosen" ]; then
        pick="$chosen"; note="(auto)"
      else
        warn "no shape was accepted with the memory free right now; falling back to $DEFAULT_PROFILE"
        pick="$DEFAULT_PROFILE"; note="(default, auto found nothing)"
      fi
    fi
  fi
  apply_profile "$pick"
  PROFILE_DESC="$PROFILE_DESC $note"
}

print_shape() {
  say "  shape:        $PROFILE_DESC"
  if [ "${SHAPE_OVERRIDDEN:-0}" = "1" ]; then
    say "  note:         an explicit CTX, MAXTOK or PREFILL_CHUNK overrode the shape's own value"
  fi
  say "  context:      $CTX    max tokens: $MAXTOK"
  say "  prefill:      chunk $PREFILL_CHUNK (the family's ceiling is 1024)"
  say "  banks:        $BANKS (this family is serial; --max-seqs above 1 is refused by name)"
  say "  backend:      $BACKEND, copy model to device: $COPY_MODEL (DS4_CUDA_COPY_MODEL)"
  say "  memory floor: ${MEM_FLOOR_GB} GiB reserved; the engine counts it inside the plan total"
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

# True while any ds4 process, or the engine's global lock, holds the slot.
slot_holder() {
  local pid
  for name in ds4-server ds4-c ds4; do
    pid="$(pgrep -x "$name" 2>/dev/null | head -1 || true)"
    [ -n "$pid" ] && { printf '%s (pid %s)' "$name" "$pid"; return 0; }
  done
  if have fuser && fuser "$LOCK_FILE" >/dev/null 2>&1; then
    printf 'a process holding %s' "$LOCK_FILE"; return 0
  fi
  return 1
}

find_hf() {
  if [ -n "$HF_BIN" ] && [ -x "$HF_BIN" ]; then printf '%s' "$HF_BIN"; return; fi
  for c in "$HOME/hfenv/bin/hf" "$HOME/.local/bin/hf" /usr/local/bin/hf; do
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

# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
preflight_repo() {
  [ -d "$REPO_DIR" ] || die "engine checkout not found: $REPO_DIR (set REPO_DIR)"
  [ -f "$REPO_DIR/Makefile" ] || die "not a ds4-dfm-rs checkout (no Makefile): $REPO_DIR"
}

preflight_family() {
  # The family must be in this checkout, or the start is a long way of learning
  # that it is not.
  grep -q "qwen35" "$REPO_DIR/crates/ds4-core/src/shape.rs" 2>/dev/null \
    || die "$REPO_DIR has no qwen35 (Bonsai) family; this checkout predates it (git -C $REPO_DIR pull)"
}

preflight_binaries() {
  [ -x "$ENGINE_BIN" ] || die "no $ENGINE_BIN; run: bash $0 build"
}

preflight_model() {
  [ -f "$MODEL" ] || die "model not found: $MODEL (run: bash $0 download, or set MODEL)"
}

preflight_banks() {
  [ "$BANKS" = "1" ] && return 0
  warn "this family is serial: its caps declare BankLane::Serial with bank_support None"
  warn "(crates/ds4-core/src/serving.rs, ModelFamily::Qwen35), so the engine refuses"
  warn "--max-seqs $BANKS with code banks_unsupported."
  say ""
  say "The Spark's memory is not what limits this model. What the memory would allow"
  say "is reported by: bash $0 budget     (the engine's own per-bank quote and the"
  say "headroom, at any context, for this box)"
  die "refusing to start with BANKS=$BANKS for a serial family"
}

preflight_port_and_slot() {
  local holder
  if running "$SERVER_PIDFILE"; then
    die "server already running (pid $(cat "$SERVER_PIDFILE")); use: bash $0 stop"
  fi
  if holder="$(slot_holder)"; then
    warn "another ds4 process holds the single engine slot: $holder"
    die "stop it first; the engine serves one ds4 model at a time by design (single $LOCK_FILE lock)"
  fi
  if port_busy; then
    die "port $PORT already has a listener; stop it or set PORT=<other>"
  fi
}

# --------------------------------------------------------------------------
# download / verify / build / test
# --------------------------------------------------------------------------
cmd_download() {
  need_hf
  mkdir -p "$MODEL_ROOT" "$RUNTIME"
  local free
  free="$(free_gb "$MODEL_ROOT")"
  say "model root:   $MODEL_ROOT"
  say "free space:   ${free} GB, need about ${NEED_GB} GB"
  [ "${free:-0}" -ge "$NEED_GB" ] || die "not enough free space at $MODEL_ROOT (${free} GB free, ${NEED_GB} GB needed)"

  step "Ternary Bonsai 2 27B, PQ2_0 (one file, about 6.7 GiB)"
  say "repo:  $HF_REPO"
  say "file:  $HF_FILE"
  "$HF_BIN" download "$HF_REPO" \
    --include "$HF_FILE" \
    --max-workers "$HF_MAX_WORKERS" \
    --local-dir "$MODEL_ROOT"

  say ""
  say "download finished. Next: bash $0 verify"
}

cmd_verify() {
  preflight_model
  step "size"
  local bytes
  bytes="$(stat -c '%s' "$MODEL" 2>/dev/null || stat -f '%z' "$MODEL")"
  say "  $MODEL"
  say "  $bytes bytes (published: $MODEL_BYTES)"
  [ "$bytes" = "$MODEL_BYTES" ] || die "size mismatch; move the incomplete file aside and re-run download"

  step "sha256"
  say "  this reads 6.7 GiB; it takes a moment"
  local actual
  actual="$(sha256sum "$MODEL" | awk '{print $1}')"
  say "  $actual"
  [ "$actual" = "$MODEL_SHA256" ] || die "checksum mismatch against the published value"
  say ""
  say "verified: this is the published artifact."
}

cmd_build() {
  preflight_repo
  preflight_family
  have cargo || die "cargo not on PATH (the Spark has it at ~/.cargo/bin)"
  mkdir -p "$RUNTIME"
  say "building in $REPO_DIR with make cuda-spark (CUDA_ARCH sm_121a)"
  say "the target also builds ds4-c, ds4, ds4-bench and ds4-agent"
  ( cd "$REPO_DIR" && make cuda-spark ) 2>&1 | tee "$RUNTIME/build.log"
  [ -x "$ENGINE_BIN" ] || die "build finished but $ENGINE_BIN is missing (see $RUNTIME/build.log)"
  [ -x "$TEST_BIN" ] || die "build finished but $TEST_BIN is missing (see $RUNTIME/build.log)"
  say ""
  say "built: $ENGINE_BIN"
  say "built: $TEST_BIN"
}

cmd_test() {
  preflight_repo
  preflight_binaries
  [ -x "$TEST_BIN" ] || die "no $TEST_BIN; run: bash $0 build"
  mkdir -p "$RUNTIME"
  # The engine's own tree defaults to the reference box's path:
  # `DS4_BONSAI_MODEL ?= /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf`
  # (Makefile) and the same default in run-bonsai.sh. On a box where the model
  # lives elsewhere - the Spark keeps it in ~/models - those targets fail with
  # "cannot open model '/data/models/...': No such file or directory". An
  # exported value wins over `?=`, so every make invocation here is given this
  # launcher's own MODEL, and the failure cannot happen.
  step "make bonsai-cuda-check (the greedy stream on the CUDA graph)"
  ( cd "$REPO_DIR" && DS4_BONSAI_MODEL="$MODEL" make bonsai-cuda-check ) 2>&1 | tee "$RUNTIME/test.log"
  if [ "$TEST_PARITY" = "1" ]; then
    step "make bonsai-cuda-parity (CUDA against the CPU reference, token for token)"
    say "the CPU reference is about 3 s per forward, so this takes minutes"
    ( cd "$REPO_DIR" && DS4_BONSAI_MODEL="$MODEL" make bonsai-cuda-parity ) 2>&1 | tee -a "$RUNTIME/test.log"
  else
    say ""
    say "TEST_PARITY=1 bash $0 test   also runs the CPU-vs-CUDA parity gate (slow)"
  fi
  say ""
  say "see $RUNTIME/test.log"
}

# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------
cmd_shapes() {
  local p
  say "Shapes this launcher can start, resolved from its own profile code:"
  for p in 45k 64k 131k 262k; do
    say ""
    ( apply_profile "$p"; print_shape )
  done
  say ""
  say "An explicitly set CTX, MAXTOK or PREFILL_CHUNK wins over the shape's value."
  say "The declared ceiling is 262144 (crates/ds4-core/src/qwen35.rs, CTX_MAX)."
}

# --------------------------------------------------------------------------
# fits: what the memory free right now will actually serve
# --------------------------------------------------------------------------
# The engine's own plan check is the authority on whether a shape fits, and it
# reads the free memory itself at plan time. This command puts that verdict next
# to the operating system's own view and answers the question a start should
# never have to guess at: which shape is the right step here, and how much
# headroom it leaves.
SHAPE_ORDER=( 262k 131k 64k 45k )          # deepest first: the first that fits wins

mem_kib() { awk -v k="$1" '$1 == k":" { print $2; exit }' /proc/meminfo; }
kib_gib() { awk -v k="${1:-0}" 'BEGIN { printf "%.1f", k / 1048576 }'; }

# Prints the deepest shape the engine accepts with the memory free right now,
# and nothing at all when none of them is accepted.
auto_shape() {
  local p rc=0
  for p in "${SHAPE_ORDER[@]}"; do
    apply_profile "$p"
    server_args
    server_env
    rc=0
    ( cd "$REPO_DIR" && env "${SERVER_ENV[@]}" ./ds4-server "${SERVER_ARGS[@]}" \
        --check-config >/dev/null 2>&1 ) || rc=$?
    if [ "$rc" = 0 ]; then printf '%s' "$p"; return 0; fi
  done
  return 1
}

cmd_fits() {
  preflight_repo
  preflight_binaries
  preflight_model
  mkdir -p "$RUNTIME"

  local total avail engine_avail="" p rc out total_bytes avail_bytes head code
  total="$(mem_kib MemTotal)"
  avail="$(mem_kib MemAvailable)"

  say "memory on this box right now"
  say "  MemTotal              $(kib_gib "${total:-0}") GiB   (the operating system)"
  say "  MemAvailable          $(kib_gib "${avail:-0}") GiB   (what the OS says is free; on a"
  say "                                  unified-memory box this is the same pool the model"
  say "                                  is served from, not a separate card's memory)"

  printf '\n  %-7s %-9s %-12s %-13s %s\n' shape ctx plan headroom verdict
  local best=""
  for p in "${SHAPE_ORDER[@]}"; do
    apply_profile "$p"
    server_args
    server_env
    rc=0
    out="$( ( cd "$REPO_DIR" && env "${SERVER_ENV[@]}" ./ds4-server "${SERVER_ARGS[@]}" \
      --check-config 2>/dev/null ) )" || rc=$?
    total_bytes="$(printf '%s' "$out" | grep -o '"total":[0-9][0-9]*' | head -1 | cut -d: -f2 || true)"
    avail_bytes="$(printf '%s' "$out" | grep -o '"available":[0-9][0-9]*' | head -1 | cut -d: -f2 || true)"
    code="$(printf '%s' "$out" | grep -o '"level":"error","code":"[^"]*"' | head -1 | cut -d'"' -f8 || true)"
    [ -z "$engine_avail" ] && engine_avail="$avail_bytes"
    if [ "$rc" = 0 ]; then
      head="$(awk -v a="${avail_bytes:-0}" -v t="${total_bytes:-0}" 'BEGIN { printf "%.1f GiB", (a - t) / 1073741824 }')"
      printf '  %-7s %-9s %-12s %-13s %s\n' "$p" "$CTX" "$(awk -v b="${total_bytes:-0}" 'BEGIN { printf "%.3f GiB", b / 1073741824 }')" "$head" "opens"
      [ -z "$best" ] && best="$p"
    else
      printf '  %-7s %-9s %-12s %-13s %s\n' "$p" "$CTX" "$(awk -v b="${total_bytes:-0}" 'BEGIN { printf "%.3f GiB", b / 1073741824 }')" "-" "refused${code:+ ($code)}"
    fi
  done

  if [ -n "$engine_avail" ]; then
    say ""
    say "  the engine reads $(awk -v b="${engine_avail:-0}" 'BEGIN { printf "%.1f", b / 1073741824 }') GiB free when it checks a plan, and the"
    say "  verdicts above are what it decided against that figure"
  fi

  say ""
  if [ -n "$best" ]; then
    say "the right step here is $best: the deepest shape the engine accepts with the"
    say "memory free right now. Start it with:"
    say "  bash $0 start $best"
    say "or let the launcher choose, so the choice follows the memory:"
    say "  bash $0 start auto"
  else
    say "no shape fits the memory free right now, not even 45k. Free some memory (another"
    say "ds4 server, a foreign process) and re-run this, or run"
    say "  bash $0 budget"
    say "which reports the per-bank line and the largest context the memory would allow."
  fi
  say ""
  say "this opens no weights and starts nothing; it is the check the start itself runs"
  say "before it loads anything."
}

# --------------------------------------------------------------------------
# the engine invocation, shared by plan and start so they cannot drift
# --------------------------------------------------------------------------
server_args() {
  SERVER_ARGS=( -m "$MODEL" --backend "$BACKEND" -c "$CTX" )
  [ -n "$MAXTOK" ] && SERVER_ARGS+=( -n "$MAXTOK" )
  SERVER_ARGS+=( --mem-floor-gb "$MEM_FLOOR_GB" --host "$HOST_ADDR" --port "$PORT" )
  [ -n "$MODEL_ID" ] && SERVER_ARGS+=( --model-id "$MODEL_ID" )
  SERVER_ARGS+=( --cors )
}

# Written with if-blocks rather than `&&` lists: a trailing `&&` list that
# evaluates false leaves the function returning non-zero, and under `set -e`
# that ends the script with no message (verified: bash exits on a failing
# function call, but not on a failing `&&` list standing on its own).
server_env() {
  SERVER_ENV=()
  if [ "$COPY_MODEL" = "1" ]; then
    SERVER_ENV+=( "DS4_CUDA_COPY_MODEL=1" )
  fi
  if [ -n "$PREFILL_CHUNK" ]; then
    SERVER_ENV+=( "DS4_QWEN35_PREFILL_CHUNK=$PREFILL_CHUNK" )
  fi
}

# --------------------------------------------------------------------------
# plan: the engine's own check, no weights opened
# --------------------------------------------------------------------------
cmd_plan() {
  select_profile "${1:-}"
  preflight_repo
  preflight_binaries
  preflight_model
  mkdir -p "$RUNTIME"     # the plan JSON is written under it; without this the
                          # redirect fails and the verdict below would blame the
                          # shape for a missing directory
  server_args
  server_env
  say "model:        $MODEL"
  say "endpoint:     http://$HOST_ADDR:$PORT"
  print_shape
  say ""
  say "the engine's own plan (--check-config opens no weights):"
  say ""
  # The engine prints its human report on stderr and the machine-readable plan
  # on stdout. The report is what a reader wants here; the JSON is kept next to
  # the log for a script, and shown only when PLAN_JSON=1 asks for it.
  local rc=0
  ( cd "$REPO_DIR" && env "${SERVER_ENV[@]}" ./ds4-server "${SERVER_ARGS[@]}" \
      --check-config >"$RUNTIME/plan.json" ) || rc=$?
  say ""
  if [ "${PLAN_JSON:-0}" = "1" ]; then
    cat "$RUNTIME/plan.json"
    say ""
  else
    say "plan JSON:    $RUNTIME/plan.json"
  fi
  # Three outcomes, kept apart on purpose. The engine exits 2 when it refuses a
  # plan, and that is a verdict about the shape; any other non-zero exit means
  # the check itself did not run (a missing directory, a binary that will not
  # start), and calling that "rejected" would send the reader to shrink a shape
  # that was never the problem.
  case "$rc" in
    0)
      say "check-config exit 0: this plan may listen" ;;
    2)
      say "check-config exit 2: the engine refused this plan"
      say "lower the shape (45k, 64k, 131k) or set CTX to a value it accepts."
      say "bash $0 budget shows what fits on this box." ;;
    *)
      warn "the plan check did not reach a verdict (exit $rc)"
      say "that is not a statement about the shape: the check command itself did not"
      say "run to completion (the real error is in the lines above)." ;;
  esac
}

# --------------------------------------------------------------------------
# budget: how much room for banks, and how much context per bank
# --------------------------------------------------------------------------
cmd_budget() {
  # A shape argument centres the report on that context; without one the default
  # profile is taken silently. This never prompts, because a report is not a
  # start and asking a question mid-report would be noise.
  local requested="${1:-${DEFAULT_PROFILE:-262k}}"
  if [ "$requested" = "auto" ]; then
    local chosen=""
    chosen="$(auto_shape)" || true
    if [ -n "$chosen" ]; then
      say "auto: centring the report on $chosen, the deepest shape the engine accepted just now"
      apply_profile "$chosen"
    else
      warn "auto found no accepted shape; centring the report on ${DEFAULT_PROFILE:-262k}"
      apply_profile "${DEFAULT_PROFILE:-262k}"
    fi
  else
    apply_profile "$requested"
  fi
  preflight_repo
  preflight_binaries
  preflight_model
  local script
  script="$(cd "$(dirname "$0")" && pwd)/bank-budget.sh"
  [ -f "$script" ] || die "missing $script (it ships next to this launcher)"
  say "asking the engine for its own quote; no weights are opened"
  say "the grid walks to the artifact's declared ceiling $CTX_MAX_HINT, not to this shape"
  say ""
  # CTX_REF is the context the bank axis is probed at: the shape you asked for,
  # so the numbers answer the question you are actually about to start.
  SERVER="$ENGINE_BIN" MODEL="$MODEL" BACKEND="$BACKEND" \
    MEM_FLOOR_GB="$MEM_FLOOR_GB" CTX_CEILING="$CTX_MAX_HINT" \
    CTX_REF="$CTX" \
    bash "$script"
}

# --------------------------------------------------------------------------
# start / stop / status / logs
# --------------------------------------------------------------------------
cmd_start() {
  select_profile "${1:-}"
  preflight_repo
  preflight_family
  preflight_binaries
  preflight_model
  preflight_banks
  preflight_port_and_slot
  mkdir -p "$RUNTIME"
  server_args
  server_env

  say "model:        $MODEL"
  say "engine:       $ENGINE_BIN"
  say "endpoint:     http://$HOST_ADDR:$PORT"
  print_shape

  if [ "$PREFLIGHT" = "1" ]; then
    step "preflight: the engine's own plan, no weights opened"
    local prc=0
    ( cd "$REPO_DIR" && env "${SERVER_ENV[@]}" ./ds4-server "${SERVER_ARGS[@]}" \
        --check-config >"$RUNTIME/plan.json" ) || prc=$?
    if [ "$prc" != 0 ]; then
      say ""
      warn "the engine rejected this plan before any weights were opened"
      die "lower the shape (bash $0 start 131k) or run bash $0 budget to see what fits"
    fi
    say "  plan accepted"
  fi

  step "starting ds4-server"
  rm -f "$SERVER_LOG"
  # The pid written here has to be the server's own: `stop` kills what this
  # file names, and the wait loop below tests it. That is why the server is
  # started directly, with nohup and nothing else. Measured 2026-10-01 on
  # vizzio: through `setsid` the recorded pid was an intermediate fork that
  # exited at once (the real server was its child), so `stop` could never
  # have found it, and through a `( ... & )` subshell the pid belonged to a
  # shell rather than to the server. nohup execs the command without forking,
  # so `$!` and /proc/<pid>/cmdline agree.
  local saved_pwd="$PWD"
  cd "$REPO_DIR" || die "cannot enter $REPO_DIR"
  nohup env "${SERVER_ENV[@]}" ./ds4-server "${SERVER_ARGS[@]}" \
      </dev/null >"$SERVER_LOG" 2>&1 &
  local server_pid=$!
  cd "$saved_pwd" || true
  printf '%s\n' "$server_pid" >"$SERVER_PIDFILE"
  say "  pid $server_pid, log $SERVER_LOG"
  if [ "$COPY_MODEL" = "1" ]; then
    say "  DS4_CUDA_COPY_MODEL=1: the model is copied to the device first"
  fi

  step "waiting for the listener"
  local waited=0
  while [ "$waited" -lt "$WAIT_LISTEN" ]; do
    grep -q "listening on" "$SERVER_LOG" 2>/dev/null && break
    if ! running "$SERVER_PIDFILE"; then
      say ""
      say "server exited while starting; last lines:"
      tail -n 12 "$SERVER_LOG" | sed 's/^/  /'
      rm -f "$SERVER_PIDFILE"
      die "start failed (see $SERVER_LOG)"
    fi
    sleep 3; waited=$((waited + 3))
    if [ $((waited % 30)) -eq 0 ]; then say "  waiting (${waited}s)"; fi
  done
  if ! grep -q "listening on" "$SERVER_LOG" 2>/dev/null; then
    tail -n 12 "$SERVER_LOG" | sed 's/^/  /'
    die "server did not report a listener within ${WAIT_LISTEN}s (log: $SERVER_LOG)"
  fi
  grep -o 'listening on.*' "$SERVER_LOG" | tail -1 | sed 's/^/  /'

  # The advertised id is read back rather than assumed, so a renamed artifact
  # cannot leave this launcher claiming a name it is not serving.
  local id ctx_len
  id="$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"][0]; print(d.get("id",""))' 2>/dev/null || true)"
  ctx_len="$(curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"][0]; print(d.get("context_length",""))' 2>/dev/null || true)"
  if [ -n "$id" ]; then
    say "  id:           $id (aliases prism-bonsai-2-27b*)"
  fi
  if [ -n "$ctx_len" ]; then
    say "  context:      $ctx_len advertised"
  fi

  say ""
  say "stop with:  bash $0 stop"
  say "check with: bash $0 status"
}

cmd_stop() {
  if ! running "$SERVER_PIDFILE"; then
    say "ds4-server: not running (no live pid in $SERVER_PIDFILE)"
    rm -f "$SERVER_PIDFILE"
    return 0
  fi
  local pid waited=0
  pid="$(cat "$SERVER_PIDFILE")"
  # The recorded pid must still be a ds4-server, not a reused number.
  if ! tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "ds4-server"; then
    warn "pid $pid is not a ds4-server; leaving it alone and clearing the pidfile"
    rm -f "$SERVER_PIDFILE"
    return 0
  fi
  say "ds4-server: sending TERM to $pid"
  kill "$pid" 2>/dev/null || true
  while [ "$waited" -lt 30 ] && kill -0 "$pid" 2>/dev/null; do sleep 1; waited=$((waited + 1)); done
  if kill -0 "$pid" 2>/dev/null; then
    say "still alive after 30s, sending KILL"
    kill -9 "$pid" 2>/dev/null || true
    sleep 2
  fi
  rm -f "$SERVER_PIDFILE"
  say "ds4-server: stopped"
}

cmd_status() {
  if running "$SERVER_PIDFILE"; then
    say "ds4-server:        running (pid $(cat "$SERVER_PIDFILE"))"
  else
    say "ds4-server:        not running"
  fi
  say "engine:            $ENGINE_BIN"
  say "model:             $MODEL"
  say "port:              $PORT"
  if have nvidia-smi; then
    say "device memory:     $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  fi
  local holder
  if holder="$(slot_holder)"; then say "engine slot:       busy: $holder"; else say "engine slot:       free"; fi
  local served
  served="$(curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"][0]; print(d.get("id",""), d.get("context_length",""), d.get("top_provider",{}).get("max_completion_tokens",""))' 2>/dev/null || true)"
  if [ -n "$served" ]; then
    say "endpoint:          answering on port $PORT"
    say "serving:           $served   (id, context, completion cap)"
  else
    say "endpoint:          no answer on port $PORT"
  fi
  if [ -f "$SERVER_LOG" ]; then
    say ""
    say "server log tail:"
    tail -n 5 "$SERVER_LOG" | sed 's/^/  /'
  fi
}

cmd_logs() {
  say "server log: $SERVER_LOG"
  if [ -f "$SERVER_LOG" ]; then
    tail -n "${1:-20}" "$SERVER_LOG" | sed 's/^/  /'
  else
    say "  (no log yet)"
  fi
}

# --------------------------------------------------------------------------
# The heredoc below is deliberately unquoted so that $0 expands to the path this
# script was invoked by. The cost of that is real: a backtick or a $( ) in the
# text is executed by the shell rather than printed, which is how an earlier
# revision of this file ended up printing "budget: command not found" in the
# middle of its own help. Keep the help text free of both.
usage() {
  cat <<EOF
Prism Ternary Bonsai 2 27B (qwen35) on the DGX Spark

  bash $0 download   fetch $HF_FILE from $HF_REPO
  bash $0 verify     size and published SHA-256
  bash $0 build      make cuda-spark (sm_121a)
  bash $0 test       make bonsai-cuda-check (TEST_PARITY=1 adds the CPU parity gate)
  bash $0 shapes     the 45k, 64k, 131k and 262k shapes
  bash $0 plan       the engine's own plan, no weights opened
  bash $0 fits       what the memory free right now will serve, and the right shape
  bash $0 budget     room for banks, and context per bank, from the engine's quote
  bash $0 start      start the server (asks for a shape when interactive)
  bash $0 stop       stop the server this launcher started
  bash $0 status     pid, port, id, device memory, engine slot
  bash $0 logs [n]   tail the server log
  bash $0 all        download, verify, build, start

Shapes (start and plan take a shape as their first argument)
  bash $0 start 45k      ctx  45,056: the RTX 4070 SUPER shape, measured numbers
  bash $0 start 64k      ctx  65,536
  bash $0 start 131k     ctx 131,072
  bash $0 start 262k     ctx 262,144: the declared ceiling, the Spark shape. DEFAULT.
  bash $0 start auto     the deepest of those the memory free right now will serve,
                         chosen by the engine's own plan check (see `fits`)

Paths
  engine      $REPO_DIR
  model root  $MODEL_ROOT
  model       $MODEL
  runtime     $RUNTIME (log, pidfile)

Overrides (environment)
  MODEL, MODEL_ROOT, REPO_DIR, RUNTIME
  PORT (default 8005), HOST_ADDR (default 0.0.0.0), BACKEND, MODEL_ID
  PROFILE=45k|64k|131k|262k|auto  the shape, same as the first argument
  DEFAULT_PROFILE=262k           what Enter means at the prompt
  ASK_SHAPE=0                    never prompt; take DEFAULT_PROFILE
  CTX, MAXTOK, PREFILL_CHUNK     individual knobs; each wins over the shape
  MEM_FLOOR_GB=1                 memory reserved for everything that is not the
                                 model, counted inside the plan total, so a
                                 larger value makes the engine demand more free
                                 memory before it will listen (never smaller
                                 than the model needs). This family's own
                                 default is 1, because the generic 4 GiB floor
                                 refused every usable context on the 12 GiB
                                 reference card (docs/BONSAI.md).
  COPY_MODEL=0                   drop DS4_CUDA_COPY_MODEL (see the header note)
  BANKS=1                        must stay 1: this family is serial
  PREFLIGHT=0                    skip the engine's pre-start plan check
  PLAN_JSON=1                    in `plan`, also print the raw plan JSON, not
                                 only the path it was written to
  WAIT_LISTEN=600                seconds to wait for the listener
  TEST_PARITY=1                  run the slow CPU-vs-CUDA parity gate in test
  HF_BIN, HF_MAX_WORKERS, NEED_GB

This family is serial and that is a property of the engine, not of the memory:
the Spark's 121.6 GiB have room for many banks' worth of KV, but the serving
caps declare banks unsupported for qwen35 and the engine refuses --max-seqs N>1
with banks_unsupported. budget reports both halves of that sentence.
EOF
}

case "${1:-}" in
  download) shift; cmd_download "$@" ;;
  verify)   shift; cmd_verify "$@" ;;
  build)    shift; cmd_build "$@" ;;
  test)     shift; cmd_test "$@" ;;
  shapes)   shift; cmd_shapes "$@" ;;
  plan)     shift; cmd_plan "$@" ;;
  fits)     shift; cmd_fits "$@" ;;
  budget)   shift; cmd_budget "$@" ;;
  start)    shift; cmd_start "$@" ;;
  stop)     shift; cmd_stop "$@" ;;
  status)   shift; cmd_status "$@" ;;
  logs)     shift; cmd_logs "$@" ;;
  all)      shift; cmd_download; cmd_verify; cmd_build; cmd_start "$@" ;;
  ""|-h|--help|help) usage ;;
  *)
    # A shape on its own is the most likely near-miss (the shape is the first
    # argument of start and plan, not a command), so name the command it wants
    # instead of only printing the help.
    printf 'unknown command: %s\n' "$1" >&2
    case "$1" in
      45k|45K|45056|64k|64K|65536|131k|131K|131072|262k|262K|262144)
        printf 'did you mean:  bash %s start %s\n' "$0" "$1" >&2
        printf 'or to size it: bash %s budget %s\n' "$0" "$1" >&2
        ;;
    esac
    printf '\n' >&2
    usage >&2
    exit 2 ;;
esac
