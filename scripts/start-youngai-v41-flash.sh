#!/usr/bin/env bash
#
# YoungAi DeepSeek-V4.1-Flash on the DGX Spark: one 113.6 GB base, one ~40 MB
# domain sidecar, one 203 GB set of n-gram tables the engine reads off the SSD.
#
# WHERE THIS RUNS
#   On the Spark itself (GB10, sm_121, aarch64, 121.6 GiB unified memory), as the
#   user who owns the model directory. The download is 317 GB, so it happens on
#   the box that serves it. Everything it writes lives under DIR (default
#   $HOME/models/youngai, matching this box: weights in ~/models, engines in
#   ~/ds4-dfm-rs); nothing needs sudo.
#
# WHAT IT DOES
#   preflight  check the machine: GPU, CUDA 13 runtime, memory, disk, free ports
#   plan       what would be fetched and started, with sizes; touches nothing
#   download   sidecars + shipped binaries + the 40 base parts (resumable), then
#              the two official n-gram shards
#   merge      assemble the base GGUF from the 40 parts, each one sha256-checked
#   verify     re-check the parts, the assembled base and the n-gram shards
#   build      restore the executable bit and check the shipped binary's flags
#   start      serve with the chosen sidecar, under a memory watchdog
#   stop       stop the server and its watchdog
#   status     pids, port, free memory, sidecar and engram completeness
#   logs       tail both logs
#   test       one chat request against the running server
#   domains    the sidecars on disk, with their layer counts
#   all        preflight, download, verify, build, start
#
# WHY THE BINARIES COME FROM THE MODEL REPO AND ARE NOT BUILT HERE
#   bin/ds4 and bin/ds4-server ship inside wenzhouwu/YoungAi-DeepSeek-V4.1-Flash,
#   built on a Spark with make cuda-spark. The quantizer and the solvers are not
#   public, and the base carries a version-3 payload with a new magic (DQV3) that
#   the loader whitelists: an engine that does not know the format refuses the
#   file at load, and one that predates the whitelist emits all-zero experts from
#   the first sentence. So `build` here does NOT compile anything. It restores the
#   executable bit (downloads do not keep it) and checks that the shipped binary
#   advertises the three flags this script passes: --zchain, --engram-dir and
#   --mem-budget-mb.
#
# THE FLAG THAT IS NOT OPTIONAL, AND THE ONE THAT MUST NOT BE PASSED
#   --engram-dir is required. The GGUF records the two n-gram shards by the
#   absolute path they had on the machine that built it, which does not exist
#   anywhere else. Without the override the engine warns at load and then stops
#   at the first request with an "engram table cannot be opened" error. The shards
#   are only ever read, never written; ENGRAM_DIR may point at a copy you already
#   have.
#   There is no context knob. The bound comes from the model's own metadata (1M
#   tokens) and --ctx is rejected by this engine, so this script never passes it.
#   Anything of the sort belongs in an environment override, not on the command
#   line.
#
# SERVING SHAPE (this follows the model card, not a preference of ours)
#   --mem-budget-mb 110000, host and port as configured, one sidecar selected by
#   --zchain. The card's own installer starts the same command on 127.0.0.1:8000
#   with a memory watchdog; this launcher keeps the watchdog and the budget and
#   changes only the bind address and the port, because 8000, 8003, 8004 and 8005
#   are already held by the other launchers in this home directory. Default port
#   here is therefore 8006.
#
# SAMPLING AND THINKING (card behaviour that surprises people)
#   A request that omits sampling fields gets the model card's recipe: temperature
#   1.0, top_p 1.0, no min_p, speculative decoding on. Send "temperature": 0
#   explicitly for byte-reproducible output. Requests default to NON-thinking
#   mode; thinking is per-request (reasoning_effort "high" or "max",
#   thinking.enabled, think true, or model "deepseek-reasoner"). The card warns
#   that the quantized weights under-weight the stop decisions, so a long thinking
#   segment can fail to close, and that greedy can loop verbatim inside the
#   thinking section on questions with very few valid answers.
#
# THE MEMORY WATCHDOG, AND WHY IT EXISTS
#   The engine mmaps 113.6 GB of weights and the machine has 121.6 GiB, so the
#   margin is thin. A watchdog samples MemAvailable every axis seconds; two
#   consecutive samples below KILL_MB stop the server, which beats a machine that
#   has started swapping and cannot be typed into. The preflight floors are the
#   card installer's: START_FLOOR_MB before loading, RUN_FLOOR_MB after.
#
# USAGE
#   bash start-youngai-v41-flash.sh preflight
#   bash start-youngai-v41-flash.sh download
#   bash start-youngai-v41-flash.sh all              (preflight, download, verify, build, start)
#   bash start-youngai-v41-flash.sh start            (finance sidecar, the card default)
#   bash start-youngai-v41-flash.sh start code
#   bash start-youngai-v41-flash.sh stop | status | logs | test | domains
#
#   DOMAIN=law PORT=8010 bash start-youngai-v41-flash.sh start
#   ENGRAM_DIR=/mnt/backup/DeepSeek-V4.1-Flash bash start-youngai-v41-flash.sh download
#     use the two official shards where they already are; their 203 GB are skipped
#   POSTTRAIN=1 bash start-youngai-v41-flash.sh start finance
#     also load the experimental post-training file (finance only; see the card)
#   ENDPOINT=https://hf-mirror.com bash start-youngai-v41-flash.sh download
#   NO_XET=1 bash start-youngai-v41-flash.sh download   (if Xet transfers keep dying)
#
# Notes
#   - The download is resumable at three levels: the repository as a whole, each
#     of the 40 base parts, and each n-gram shard. Re-run the same command.
#   - Every base part is checked against the published SHA256SUMS before it is
#     appended, and a bad part is deleted rather than kept; the assembled file is
#     checked against the hash published for it.
#   - The watchdog and the server die together. stop removes both pidfiles.
#   - The card is a third-party publication: a single author, engine source not
#     public, and two opaque aarch64 binaries executed on your box. verify is the
#     only check available for the artifacts; there is no build-from-source path.

set -Eeuo pipefail

# --------------------------------------------------------------------------
# Configuration (every value is overridable from the environment)
# --------------------------------------------------------------------------
REPO_ID="${REPO_ID:-wenzhouwu/YoungAi-DeepSeek-V4.1-Flash}"
OFFICIAL_REPO="${OFFICIAL_REPO:-deepseek-ai/DeepSeek-V4.1-Flash}"

GGUF="DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf"
GGUF_BYTES=113556639424
GGUF_SHA256=9469cfa9ff47c5b9f1e2bbf31b4226ef11b4de9ddb92a0c7641c4a9b623e4654
NPART=40
PART_BASE=2838915985          # bytes in parts 25..40
PART_EXTRA=24                 # parts 1..24 carry one extra byte each

# The two n-gram ("engram") shards of the official checkpoint. They are not in
# this repository: the engine reads them, untouched, from wherever they are.
SHARDS=(model-00047-of-00048.safetensors model-00048-of-00048.safetensors)
SHARD_BYTES=(101535150936 101537926640)

POSTTRAIN_DIR="posttrain-experimental-20260924"
POSTTRAIN_DOMAIN="finance"

DIR="${DIR:-$HOME/models/youngai}"   # the box keeps weights in ~/models, engines in ~/ds4-dfm-rs
DOMAIN="${DOMAIN:-finance}"
HOST_ADDR="${HOST_ADDR:-0.0.0.0}"
PORT="${PORT:-8006}"
BUDGET_MB="${BUDGET_MB:-110000}"        # the card's engine budget
START_FLOOR_MB="${START_FLOOR_MB:-100000}"
RUN_FLOOR_MB="${RUN_FLOOR_MB:-10000}"
KILL_MB="${KILL_MB:-2500}"              # watchdog: two samples below this and the server stops
READY_TIMEOUT="${READY_TIMEOUT:-900}"   # loading takes about two minutes on the card's box
WATCHDOG="${WATCHDOG:-1}"
POSTTRAIN="${POSTTRAIN:-0}"
ENDPOINT="${ENDPOINT:-}"
NO_XET="${NO_XET:-0}"
HF_BIN="${HF_BIN:-}"                    # autodetected if empty
HF="${HF:-}"                            # resolved by ensure_hf
HF_MAX_WORKERS="${HF_MAX_WORKERS:-8}"
ENGRAM_DIR="${ENGRAM_DIR:-}"            # empty = DIR/engram
NEED_GB="${NEED_GB:-325}"               # the card asks for ~320 GB plus working room

MODEL_DIR="$DIR/model"
ENGRAM="${ENGRAM_DIR:-$DIR/engram}"
LOGS="$DIR/logs"
PIDF="$DIR/server.pid"
WDPIDF="$DIR/watchdog.pid"
INFOF="$DIR/server.info"
SERVER_LOG="$LOGS/server.log"
WATCHDOG_LOG="$LOGS/watchdog.log"

sidecar_of() {
  case "$1" in
    finance)  echo "DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-vqfin41_vqhalf_a_n8192-engine";;
    code)     echo "DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-code_fit_n15360-engine";;
    law)      echo "DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-law_fit_n15360-engine";;
    medicine) echo "DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-med_fit_n15360-engine";;
    science)  echo "DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-sci_fit_n15360-engine";;
    none)     echo "";;
    *)        return 1;;
  esac
}
DOMAINS="finance | code | law | medicine | science | none"

SIDECAR="$(sidecar_of "$DOMAIN")" || { echo "unknown DOMAIN '$DOMAIN' (choose $DOMAINS)" >&2; exit 2; }
MODEL_BIN="$MODEL_DIR/bin/ds4"
SERVER_BIN="$MODEL_DIR/bin/ds4-server"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
say()  { echo "[youngai $(date +%H:%M:%S)] $*"; }
warn() { echo "[youngai $(date +%H:%M:%S)] warning: $*" >&2; }
die()  { echo "[youngai] ERROR: $*" >&2; exit 1; }

avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
total_mb() { awk '/MemTotal/{print int($2/1024)}' /proc/meminfo; }
fsize()    { stat -L -c %s "$1" 2>/dev/null || echo 0; }
# df on a directory that does not exist yet must not fail the script under
# pipefail: it is a report, and the caller decides what a zero means.
df_avail()  { df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' \n' || true; }
df_target() { df --output=target "$1" 2>/dev/null | tail -1 || true; }
part_name() { printf '%s.part%02d-of-%d' "$GGUF" "$1" "$NPART"; }
# bytes held in parts 1..k
parts_len() { local k="$1"; echo $(( k * PART_BASE + (k < PART_EXTRA ? k : PART_EXTRA) )); }
alive()    { [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

human() { awk -v b="$1" 'BEGIN{printf "%.1f GB", b/1000000000}'; }

# The SHA256SUMS file is the source of truth for the parts and the assembled
# base; GGUF_SHA256 above is only a fallback for a missing manifest.
sum_of() {
  local f="$1"
  [ -s "$MODEL_DIR/SHA256SUMS" ] || return 1
  awk -v f="$f" '$2 == f {print $1; found=1} END{exit !found}' "$MODEL_DIR/SHA256SUMS"
}

sha_of() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }

# --------------------------------------------------------------------------
# preflight: the machine checks the card's installer makes, plus our own
# --------------------------------------------------------------------------
preflight() {
  local fatal=0

  if [ "$(uname -s)/$(uname -m)" = "Linux/aarch64" ]; then
    say "arch ok: Linux/aarch64"
  else
    warn "this is $(uname -s)/$(uname -m); the shipped binaries are aarch64 Linux for the DGX Spark"
    fatal=1
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu; gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    case "$gpu" in
      *GB10*) say "gpu ok: $gpu";;
      *) warn "GPU is '$gpu'; the binaries are compiled for GB10 (sm_121)"; fatal=1;;
    esac
  else
    warn "nvidia-smi not found: no NVIDIA driver on this machine"; fatal=1
  fi

  # The card needs the CUDA 13 runtime: libcudart.so.13, libcublas.so.13,
  # libcublasLt.so.13. Missing libraries show up at load as "error while loading
  # shared libraries: libcudart.so.13".
  local ldc miss=()
  ldc="$(command -v ldconfig || echo /sbin/ldconfig)"
  if [ -x "$ldc" ]; then
    local lib
    for lib in libcudart.so.13 libcublas.so.13 libcublasLt.so.13; do
      "$ldc" -p 2>/dev/null | grep -q "$lib" || miss+=("$lib")
    done
    if [ "${#miss[@]}" -eq 0 ]; then
      say "cuda ok: CUDA 13 runtime registered with ldconfig"
    else
      warn "not in the ldconfig cache: ${miss[*]} (the CUDA 13 runtime is required; DGX OS ships it)"
      fatal=1
    fi
  else
    warn "ldconfig not found; cannot check the CUDA 13 runtime"
  fi

  local t; t="$(total_mb)"
  if [ "$t" -ge 115000 ]; then
    say "memory ok: ${t} MB total"
  else
    warn "only ${t} MB of memory; the model needs a 128 GB DGX Spark"; fatal=1
  fi

  say "memory available now: $(avail_mb) MB (the server refuses to start below ${START_FLOOR_MB} MB)"

  # Disk. The card wants ~320 GB: 113.6 GB of base, 203 GB of n-gram shards, and
  # one part's worth of working room during assembly. Whatever is already on
  # disk is subtracted.
  mkdir -p "$DIR" 2>/dev/null || { warn "cannot create $DIR"; fatal=1; }
  local need=0 free
  [ "$(fsize "$MODEL_DIR/$GGUF")" = "$GGUF_BYTES" ] || need=$((need + GGUF_BYTES + PART_BASE + 1))
  if ! engram_ok; then
    local i
    for i in 0 1; do
      [ "$(fsize "$ENGRAM/${SHARDS[$i]}")" = "${SHARD_BYTES[$i]}" ] || need=$((need + SHARD_BYTES[i]))
    done
  fi
  free="$(df_avail "$DIR")"
  if [ -n "$free" ] && [ "$free" -gt $((need + 5000000000)) ]; then
    say "disk ok: $(human $need) still to fetch, $(human $free) free on $(df_target "$DIR")"
  else
    warn "not enough room on $(df_target "$DIR"): want $(human $need) plus 5 GB working room, have $(human ${free:-0})"
    warn "set DIR= to a volume with room (the card asks for $NEED_GB GB in total)"
    fatal=1
  fi

  # Ports: this box already serves other models. Starting a second 110 GB model
  # while one is resident is the failure this warning is about.
  local busy=() p
  if command -v curl >/dev/null 2>&1; then
    for p in 8000 8003 8004 8005 8006; do
      [ "$p" = "$PORT" ] && continue
      curl -fsS --max-time 2 "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && busy+=("$p")
    done
  fi
  if [ "$PORT" -le 8005 ] && curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    warn "port $PORT already answers /v1/models; pick another with PORT="
    fatal=1
  fi
  if [ "${#busy[@]}" -gt 0 ]; then
    warn "these ports already serve a model: ${busy[*]} - another resident model will not fit beside this one"
  fi

  [ "$fatal" = 0 ] && say "preflight: this machine can run the model" || return 1
}

# --------------------------------------------------------------------------
# Hugging Face CLI
# --------------------------------------------------------------------------
ensure_hf() {
  if [ -n "$HF_BIN" ]; then HF="$HF_BIN"; return; fi
  if command -v hf >/dev/null 2>&1; then HF=hf; return; fi
  if [ -x "$HOME/.local/bin/hf" ]; then HF="$HOME/.local/bin/hf"; return; fi
  HF="$DIR/.hfenv/bin/hf"
  [ -x "$HF" ] && return
  say "installing the Hugging Face CLI into $DIR/.hfenv (the system Python is left alone)"
  mkdir -p "$DIR"
  python3 -m venv "$DIR/.hfenv" || die "python3 venv failed; install it with: sudo apt install python3-venv"
  "$DIR/.hfenv/bin/pip" install -q -U "huggingface_hub[hf_xet]" \
    || die "pip install huggingface_hub failed (behind a firewall? try ENDPOINT= and NO_XET=1)"
}

hf_env() {
  [ -n "$ENDPOINT" ] && export HF_ENDPOINT="$ENDPOINT"
  [ "$NO_XET" = 1 ] && export HF_HUB_DISABLE_XET=1
  export HF_HUB_MAX_WORKERS="$HF_MAX_WORKERS"
  return 0
}

# --------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------
engram_ok() {   # both shards present, at the sizes the published installer states
  [ "$(fsize "$ENGRAM/${SHARDS[0]}")" = "${SHARD_BYTES[0]}" ] &&
  [ "$(fsize "$ENGRAM/${SHARDS[1]}")" = "${SHARD_BYTES[1]}" ]
}

download_repo() {
  ensure_hf; hf_env
  mkdir -p "$MODEL_DIR"
  say "fetching $REPO_ID except the base parts (sidecars, shipped binaries, SHA256SUMS, install.sh)"
  "$HF" download "$REPO_ID" --local-dir "$MODEL_DIR" --exclude "*.part*-of-*" \
    || die "download of $REPO_ID failed; run the same command again to resume"
  [ -s "$MODEL_DIR/SHA256SUMS" ] || die "SHA256SUMS missing from the download"
  chmod +x "$MODEL_DIR/bin/ds4" "$MODEL_DIR/bin/ds4-server" 2>/dev/null || true
  say "sidecars on disk: $(sidecar_count)"
}

# One part at a time: download, check its published sha256, append, delete. Peak
# extra disk is one part. .merge-progress records how many parts are already in
# the .merging file, and the file is cut back to exactly that length on restart,
# because an append interrupted halfway leaves junk at the end.
merge_parts() {
  local out="$MODEL_DIR/$GGUF.merging" prog="$MODEL_DIR/.merge-progress" k=0 i p want got
  if [ "$(fsize "$MODEL_DIR/$GGUF")" = "$GGUF_BYTES" ]; then say "base already assembled"; return; fi
  ensure_hf; hf_env
  [ -s "$prog" ] && k="$(cat "$prog")"
  case "$k" in ''|*[!0-9]*) die "$prog does not hold a part count";; esac
  if [ "$k" -gt 0 ] && [ "$(fsize "$out")" -lt "$(parts_len "$k")" ]; then
    die "$out is shorter than the $k parts recorded in $prog; delete both and run download again"
  fi
  truncate -s "$(parts_len "$k")" "$out"
  for ((i = k + 1; i <= NPART; i++)); do
    p="$(part_name "$i")"
    say "part $i/$NPART"
    "$HF" download "$REPO_ID" "$p" --local-dir "$MODEL_DIR" >/dev/null \
      || die "download of $p failed; run again to resume"
    want="$(sum_of "$p")" || die "$p is not listed in SHA256SUMS"
    got="$(sha_of "$MODEL_DIR/$p")"
    if [ "$got" != "$want" ]; then
      rm -f "$MODEL_DIR/$p"
      die "$p failed its sha256 check and was deleted; run download again"
    fi
    cat "$MODEL_DIR/$p" >>"$out" || die "cannot append $p (disk full?)"
    echo "$i" >"$prog"
    rm -f "$MODEL_DIR/$p"
  done
  [ "$(fsize "$out")" = "$GGUF_BYTES" ] || die "assembled file is $(fsize "$out") bytes, expected $GGUF_BYTES"
  mv "$out" "$MODEL_DIR/$GGUF" && rm -f "$prog"
  say "base assembled: $GGUF ($(human $GGUF_BYTES)), every part sha256-checked"
}

download_engram() {
  if engram_ok; then say "n-gram tables already in $ENGRAM, skipping their 203 GB"; return; fi
  ensure_hf; hf_env
  mkdir -p "$ENGRAM"
  say "n-gram tables: two shards of $OFFICIAL_REPO into $ENGRAM (203 GB)"
  say "  these are read-only inputs for the engine; they are never modified"
  "$HF" download "$OFFICIAL_REPO" "${SHARDS[@]}" --local-dir "$ENGRAM" \
    || die "download of the official shards failed; run again to resume"
  local i
  for i in 0 1; do
    [ "$(fsize "$ENGRAM/${SHARDS[$i]}")" = "${SHARD_BYTES[$i]}" ] \
      || die "${SHARDS[$i]} has the wrong size; run again to resume"
  done
}

cmd_download() {
  download_repo
  merge_parts
  download_engram
  say "download complete: base, sidecars and n-gram tables are in place"
  say "next: bash $0 build && bash $0 start $DOMAIN"
}

# --------------------------------------------------------------------------
# verify / build
# --------------------------------------------------------------------------
cmd_verify() {
  local bad=0 i p want got
  say "verifying against $MODEL_DIR/SHA256SUMS"
  [ -s "$MODEL_DIR/SHA256SUMS" ] || die "SHA256SUMS is missing; run download first"

  for ((i = 1; i <= NPART; i++)); do
    p="$(part_name "$i")"
    [ -f "$MODEL_DIR/$p" ] || continue
    want="$(sum_of "$p")" || { warn "$p not listed"; bad=1; continue; }
    got="$(sha_of "$MODEL_DIR/$p")"
    if [ "$got" = "$want" ]; then say "  part $i/$NPART ok"; else warn "  part $i/$NPART MISMATCH"; bad=1; fi
  done

  if [ -f "$MODEL_DIR/$GGUF" ]; then
    got="$(sha_of "$MODEL_DIR/$GGUF")"
    want="$(sum_of "$GGUF" || echo "$GGUF_SHA256")"
    if [ "$got" = "$want" ]; then
      say "  assembled base ok ($(human "$(fsize "$MODEL_DIR/$GGUF")"))"
    else
      warn "  assembled base MISMATCH"; bad=1
    fi
  else
    say "  assembled base not present yet (run merge)"
  fi

  for i in 0 1; do
    if [ "$(fsize "$ENGRAM/${SHARDS[$i]}")" = "${SHARD_BYTES[$i]}" ]; then
      say "  ${SHARDS[$i]} ok ($(human "${SHARD_BYTES[$i]}"), size check; no published hash exists for these)"
    else
      say "  ${SHARDS[$i]} missing or short ($(fsize "$ENGRAM/${SHARDS[$i]}") of ${SHARD_BYTES[$i]} bytes)"
      bad=1
    fi
  done

  if [ -n "$SIDECAR" ]; then
    if [ -s "$MODEL_DIR/$SIDECAR/manifest.txt" ]; then
      say "  sidecar $SIDECAR present ($(sidecar_layers "$SIDECAR"))"
    else
      warn "  sidecar $SIDECAR is missing its manifest.txt"; bad=1
    fi
  fi

  [ "$bad" = 0 ] && say "verify: everything checked out" || die "verify: something above is wrong"
}

cmd_build() {
  [ -f "$SERVER_BIN" ] || die "no $SERVER_BIN; run download first"
  if [ -f "$MODEL_BIN" ]; then
    chmod +x "$MODEL_BIN"
  else
    warn "bin/ds4 (the command-line client) is not present; the server does not need it"
  fi
  chmod +x "$SERVER_BIN"
  # The shipped binary must know the three flags this launcher passes. Captured
  # first, not piped into grep -q: grep quits at the match, the process can die of
  # SIGPIPE, and pipefail would then call a good binary old.
  local help; help="$("$SERVER_BIN" --help 2>&1 || true)"
  [ -n "$help" ] || die "$SERVER_BIN printed nothing for --help; is it the aarch64 build for this machine?"
  local f ok=1
  for f in --zchain --engram-dir --mem-budget-mb; do
    case "$help" in *"$f"*) ;; *) warn "$SERVER_BIN does not advertise $f"; ok=0;; esac
  done
  # The published installer fails hard here too, and for the same reason:
  # --engram-dir absent means the first request dies with an engram error after a
  # two-minute load, and --zchain absent means the sidecar is silently not used.
  [ "$ok" = 1 ] || die "$SERVER_BIN is not the binary this launcher expects; re-run: bash $0 download"
  say "shipped binary advertises --zchain, --engram-dir and --mem-budget-mb"
  say "nothing was compiled: this engine is not built from source (see the header)"
}

# --------------------------------------------------------------------------
# plan: what download and start would do, touching nothing
# --------------------------------------------------------------------------
cmd_plan() {
  local need=0 i free
  say "install dir:   $DIR"
  say "model dir:     $MODEL_DIR"
  say "engram dir:    $ENGRAM"
  if [ -n "$SIDECAR" ]; then
    say "domain:        $DOMAIN (sidecar $SIDECAR)"
  else
    say "domain:        none (bare base: a complete model, no sidecar)"
  fi
  say "serve:         $HOST_ADDR:$PORT, --mem-budget-mb $BUDGET_MB, watchdog below ${KILL_MB} MB"
  say ""
  say "artifacts"
  say "  base         $NPART parts, $(human $GGUF_BYTES) assembled   $([ "$(fsize "$MODEL_DIR/$GGUF")" = "$GGUF_BYTES" ] && echo "already assembled" || echo "not assembled")"
  say "  sidecars     5 domains, ~40 MB each (not per-domain downloads; all of them arrive together)"
  say "  binaries     bin/ds4, bin/ds4-server (aarch64, shipped; source is not public)"
  for i in 0 1; do
    say "  n-gram       ${SHARDS[$i]} $(human "${SHARD_BYTES[$i]}")   $([ "$(fsize "$ENGRAM/${SHARDS[$i]}")" = "${SHARD_BYTES[$i]}" ] && echo present || echo missing)"
  done
  [ "$(fsize "$MODEL_DIR/$GGUF")" = "$GGUF_BYTES" ] || need=$((need + GGUF_BYTES + PART_BASE + 1))
  if ! engram_ok; then
    for i in 0 1; do
      [ "$(fsize "$ENGRAM/${SHARDS[$i]}")" = "${SHARD_BYTES[$i]}" ] || need=$((need + SHARD_BYTES[i]))
    done
  fi
  mkdir -p "$DIR"
  free="$(df_avail "$DIR")"
  say ""
  say "still to fetch: $(human $need); free on target $(df_target "$DIR"): $(human ${free:-0})"
  if [ -n "$SIDECAR" ]; then
    say ""
    say "the command start would run:"
    say "  $SERVER_BIN --cuda -m $MODEL_DIR/$GGUF \\"
    say "      --zchain $MODEL_DIR/$SIDECAR \\"
    say "      --engram-dir $ENGRAM \\"
    say "      --mem-budget-mb $BUDGET_MB --host $HOST_ADDR --port $PORT"
    say "  (no --ctx: this engine rejects it; the bound comes from the model metadata)"
  fi
  say ""
  say "ports 8000, 8003, 8004 and 8005 are used by the other launchers in this home; this one defaults to $PORT"
}

# --------------------------------------------------------------------------
# sidecars
# --------------------------------------------------------------------------
sidecar_layers() {  # $1 = sidecar dir name
  local d="$MODEL_DIR/$1" g r
  g="$(find "$d" -maxdepth 1 -name 'gr_L*.bin' 2>/dev/null | wc -l)"
  r="$(find "$d" -maxdepth 1 -name 'rb_L*.bin' 2>/dev/null | wc -l)"
  printf 'gains on %s layers, router bias on %s layers' "$g" "$r"
}

sidecar_count() {
  local n=0 d
  for d in finance code law medicine science; do
    [ -s "$MODEL_DIR/$(sidecar_of "$d")/manifest.txt" ] && n=$((n + 1))
  done
  printf '%s of 5' "$n"
}

cmd_domains() {
  local d dir
  for d in finance code law medicine science; do
    dir="$(sidecar_of "$d")"
    if [ -s "$MODEL_DIR/$dir/manifest.txt" ]; then
      say "$(printf '%-9s %s' "$d" "$(sidecar_layers "$dir")")"
    else
      say "$(printf '%-9s %s' "$d" "not downloaded")"
    fi
  done
  say ""
  say "none      bare base: no sidecar loaded"
  say "switch:   bash $0 stop && bash $0 start <domain>"
  [ "$POSTTRAIN" = 1 ] && say "posttrain: loading $POSTTRAIN_DIR (stacks on $POSTTRAIN_DOMAIN only)"
  return 0
}

# --------------------------------------------------------------------------
# start / stop / status / logs / test
# --------------------------------------------------------------------------
watchdog() {   # $1 pid, $2 floor MB, $3 log file
  local pid="$1" bad=0 a
  while kill -0 "$pid" 2>/dev/null; do
    a="$(avail_mb)"
    if [ "$a" -lt "$2" ]; then bad=$((bad + 1)); else bad=0; fi
    if [ "$bad" -ge 2 ]; then
      echo "$(date) MemAvailable ${a} MB below $2 MB twice; stopping the server" >>"$3"
      kill "$pid" 2>/dev/null
      return
    fi
    sleep 5
  done
}

cmd_start() {
  if alive; then
    say "already running (pid $(cat "$PIDF"), $(cat "$INFOF" 2>/dev/null)); stop it first to switch domain"
    return 0
  fi
  [ -f "$MODEL_DIR/$GGUF" ] || die "the base is not assembled; run: bash $0 download"
  [ -x "$SERVER_BIN" ] || die "no executable $SERVER_BIN; run: bash $0 build"
  engram_ok || die "the n-gram tables are not in $ENGRAM (203 GB); run: bash $0 download"
  if [ -n "$SIDECAR" ]; then
    [ -s "$MODEL_DIR/$SIDECAR/manifest.txt" ] || die "the $DOMAIN sidecar is not in $MODEL_DIR; run: bash $0 download"
  fi
  if [ "$POSTTRAIN" = 1 ]; then
    [ "$DOMAIN" = "$POSTTRAIN_DOMAIN" ] \
      || die "the post-training file was solved on the $POSTTRAIN_DOMAIN sidecar; the engine refuses any other pairing"
    [ -d "$MODEL_DIR/$POSTTRAIN_DIR" ] || die "$POSTTRAIN_DIR is not in $MODEL_DIR; run: bash $0 download"
  fi

  local a; a="$(avail_mb)"
  [ "$a" -ge "$START_FLOOR_MB" ] \
    || die "only $a MB of memory available, need $START_FLOOR_MB: stop another model first (this box serves 8000/8003/8004/8005)"

  mkdir -p "$LOGS"
  local zc=() pt=()
  [ -n "$SIDECAR" ] && zc=(--zchain "$MODEL_DIR/$SIDECAR")
  [ "$POSTTRAIN" = 1 ] && pt=(--posttrain "$MODEL_DIR/$POSTTRAIN_DIR")

  say "starting ds4-server on $HOST_ADDR:$PORT, domain $DOMAIN (loading takes about two minutes)"
  say "  log: $SERVER_LOG"
  nohup "$SERVER_BIN" --cuda -m "$MODEL_DIR/$GGUF" "${zc[@]}" --engram-dir "$ENGRAM" "${pt[@]}" \
    --mem-budget-mb "$BUDGET_MB" --host "$HOST_ADDR" --port "$PORT" \
    >"$SERVER_LOG" 2>&1 </dev/null &
  echo $! >"$PIDF"
  echo "domain $DOMAIN on $HOST_ADDR:$PORT" >"$INFOF"

  if [ "$WATCHDOG" = 1 ]; then
    export -f watchdog avail_mb
    nohup bash -c "watchdog $(cat "$PIDF") $KILL_MB '$WATCHDOG_LOG'" >/dev/null 2>&1 </dev/null &
    echo $! >"$WDPIDF"
    say "  watchdog pid $(cat "$WDPIDF"), stops the server if MemAvailable stays below ${KILL_MB} MB"
  fi

  local waited=0 code=0
  say "waiting for the endpoint (up to ${READY_TIMEOUT}s; the port binds only after the weights are in)"
  while [ "$waited" -lt "$READY_TIMEOUT" ]; do
    if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then code=0; break; fi
    if ! alive; then
      tail -12 "$SERVER_LOG" 2>/dev/null | sed 's/^/  /'
      cmd_stop >/dev/null
      die "the server exited while loading (full log: $SERVER_LOG)"
    fi
    sleep 5; waited=$((waited + 5))
    [ $((waited % 60)) -eq 0 ] && say "  still loading (${waited}s)"
  done
  if [ "$waited" -ge "$READY_TIMEOUT" ]; then
    tail -12 "$SERVER_LOG" 2>/dev/null | sed 's/^/  /'
    cmd_stop >/dev/null
    die "the server did not answer /v1/models within ${READY_TIMEOUT}s (log: $SERVER_LOG)"
  fi

  a="$(avail_mb)"
  if [ "$a" -lt "$RUN_FLOOR_MB" ]; then
    cmd_stop >/dev/null
    die "only $a MB left after loading (need $RUN_FLOOR_MB); the machine would start swapping"
  fi

  say "loaded; ${a} MB of memory left"
  cmd_test || warn "the smoke request failed; the server is up, so check $SERVER_LOG"
  say ""
  say "install dir: $DIR"
  say "ready: http://$HOST_ADDR:$PORT/v1   (OpenAI: /v1/chat/completions /v1/completions /v1/responses, Anthropic: /v1/messages)"
  say "domain: $DOMAIN   stop with: bash $0 stop   switch: bash $0 stop && bash $0 start <domain>"
  say "requests default to non-thinking mode and to the card's sampling recipe; send \"temperature\": 0 for reproducible output"
}

cmd_stop() {
  [ -s "$WDPIDF" ] && kill "$(cat "$WDPIDF")" 2>/dev/null || true
  if alive; then
    local pid; pid="$(cat "$PIDF")"
    say "stopping ds4-server (pid $pid)"
    kill "$pid" 2>/dev/null || true
    local i=0
    while [ "$i" -lt 20 ] && kill -0 "$pid" 2>/dev/null; do sleep 1; i=$((i + 1)); done
    kill -0 "$pid" 2>/dev/null && { say "  still alive after 20s, sending KILL"; kill -9 "$pid" 2>/dev/null || true; }
  else
    say "ds4-server: not running"
  fi
  rm -f "$PIDF" "$WDPIDF" "$INFOF"
}

cmd_status() {
  if alive; then
    say "ds4-server:    running (pid $(cat "$PIDF")), $(cat "$INFOF" 2>/dev/null)"
    ps -o pid=,etime=,rss= -p "$(cat "$PIDF")" 2>/dev/null | sed 's/^/  /'
  else
    say "ds4-server:    not running"
  fi
  if [ -s "$WDPIDF" ] && kill -0 "$(cat "$WDPIDF")" 2>/dev/null; then
    say "watchdog:      running (pid $(cat "$WDPIDF"))"
  else
    say "watchdog:      not running"
  fi
  say "memory:        $(avail_mb) MB available of $(total_mb) MB"
  say "domain:        $DOMAIN${SIDECAR:+ ($(sidecar_layers "$SIDECAR"))}"
  say "engram dir:    $ENGRAM $([ "$(engram_ok && echo ok)" = ok ] && echo "(both shards present)" || echo "(INCOMPLETE)")"
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | head -c 400; then
    say ""
    say "endpoint:      answering on port $PORT"
  else
    say "endpoint:      no answer on port $PORT"
  fi
  if [ -f "$SERVER_LOG" ]; then
    say ""
    say "server log tail:"
    tail -n 6 "$SERVER_LOG" | sed 's/^/  /'
  fi
}

cmd_logs() {
  say "server log: $SERVER_LOG"
  [ -f "$SERVER_LOG" ] && tail -n 20 "$SERVER_LOG" | sed 's/^/  /'
  say ""
  say "watchdog log: $WATCHDOG_LOG"
  [ -f "$WATCHDOG_LOG" ] && tail -n 10 "$WATCHDOG_LOG" | sed 's/^/  /'
  return 0
}

cmd_test() {
  local q
  case "$DOMAIN" in
    code)     q="In one sentence, what does git rebase do?";;
    law)      q="In one sentence, what is a statute of limitations?";;
    medicine) q="In one sentence, what is hypertension?";;
    science)  q="In one sentence, what is entropy in thermodynamics?";;
    *)        q="In one sentence, what is a price-to-earnings ratio?";;
  esac
  command -v curl >/dev/null 2>&1 || die "curl is required for the smoke test"
  local out
  out="$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"deepseek-chat\",\"temperature\":0,\"max_tokens\":48,\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}]}")"
  case "$out" in
    *'"choices"'*) printf '%s\n' "$out" | sed -n 's/.*"content":"\([^"]*\)".*/  -> \1/p' | head -1;;
    *) warn "the request failed: $(printf '%s' "$out" | head -c 300)"; return 1;;
  esac
}

# --------------------------------------------------------------------------
usage() {
  cat <<EOF
YoungAi DeepSeek-V4.1-Flash (VQ-8 base + one domain sidecar) on the DGX Spark

  bash $0 preflight   check GPU, CUDA 13, memory, disk and ports
  bash $0 plan        what would be fetched and started, with sizes (touches nothing)
  bash $0 download    sidecars, shipped binaries, 40 base parts, 203 GB n-gram shards
  bash $0 merge       assemble the base from the 40 parts (download does this too)
  bash $0 verify      re-check the parts, the assembled base and the n-gram shards
  bash $0 build       chmod +x the shipped binaries and check the flags they must know
  bash $0 start       serve with the selected sidecar, under a memory watchdog
  bash $0 stop        stop the server and the watchdog
  bash $0 status      pids, port, free memory, sidecar and engram state
  bash $0 logs        tail the server and watchdog logs
  bash $0 test        one chat request against the running server
  bash $0 domains     the sidecars on disk, with their layer counts
  bash $0 all         preflight, download, verify, build, start

Domains (start takes one as its argument; the card's default is finance)
  finance    the sidecar the card's installer serves by default
  code       programming text
  law        statutes, case facts, exam questions
  medicine   encyclopedia Q&A, consultations, exam questions
  science    journal abstracts, arXiv, STEM questions
  none       bare base: a complete model, no sidecar, -10 to -13 points of Same top-1
  bash $0 start code            same as DOMAIN=code bash $0 start

Paths
  install dir   $DIR
  model dir     $MODEL_DIR
  engram dir    $ENGRAM
  logs          $LOGS

Overrides (environment)
  DIR, DOMAIN, PORT (default 8006), HOST_ADDR (default 0.0.0.0)
  ENGRAM_DIR      use the two official shards where they already are (skips 203 GB)
  POSTTRAIN=1     also load the experimental post-training file (finance only)
  BUDGET_MB=110000, START_FLOOR_MB, RUN_FLOOR_MB, KILL_MB, WATCHDOG=0
  READY_TIMEOUT=900, HF_BIN, HF_MAX_WORKERS
  ENDPOINT=https://hf-mirror.com, NO_XET=1, REPO_ID, OFFICIAL_REPO

What the card's shape means here
  The base is 113.6 GB resident; the sidecar adds about 40 MB and roughly 0.6 MB
  of reads per decoded token. The 203 GB of n-gram tables stay on the SSD and are
  read, untouched, through --engram-dir. There is no context flag: the engine
  rejects --ctx and takes the bound from the model metadata (1M tokens).
  Ports 8000, 8003, 8004 and 8005 answer for the other launchers in this home, so
  stop one before starting this one.
EOF
}

case "${1:-}" in
  preflight) shift; preflight "$@";;
  plan)      shift; cmd_plan "$@";;
  download)  shift; cmd_download "$@";;
  merge)     shift; merge_parts "$@";;
  verify)    shift; cmd_verify "$@";;
  build)     shift; cmd_build "$@";;
  start)     shift; [ $# -gt 0 ] && { DOMAIN="$1"; SIDECAR="$(sidecar_of "$DOMAIN")" || { echo "unknown domain '$DOMAIN' (choose $DOMAINS)" >&2; exit 2; }; shift; }; cmd_start "$@";;
  stop)      shift; cmd_stop "$@";;
  status)    shift; cmd_status "$@";;
  logs)      shift; cmd_logs "$@";;
  test)      shift; cmd_test "$@";;
  domains)   shift; cmd_domains "$@";;
  all)       shift; preflight; cmd_download; cmd_verify; cmd_build; cmd_start "$@";;
  ""|-h|--help|help) usage;;
  *) echo "unknown command: $1 (see help)" >&2; usage >&2; exit 2;;
esac
