#!/usr/bin/env bash
#
# GLM-5.3-Flash VQ (Q4_K nonexpert v3) on the DGX Spark: 56 GGUF parts, 72 GiB,
# served by the ds4-dfm-rs Rust host at the artifact's native 1,048,576 context.
#
# WHERE THIS RUNS
#   On the Spark itself (GB10, sm_121, aarch64, 121.6 GiB unified memory), as the
#   user who owns the model directory. The download is 72 GiB, so it happens on
#   the box that serves it. Nothing needs sudo.
#
# WHAT IT DOES
#   preflight  GPU, CUDA 13 runtime, memory, disk, ports, and a second-owner probe
#   plan       what would be fetched and started, with sizes; touches nothing
#   download   the release, resumable at the file level
#   verify     SHA256SUMS over every payload byte (slow: it reads all 72 GiB)
#   check      ds4-server --check-config: the plan quote, opens no weights
#   start      serve, under a memory watchdog; waits for /v1/models then smokes it
#   stop       stop the server and the watchdog
#   status     pid, port, free memory, release completeness, log tail
#   logs       the server and watchdog logs
#   test       one chat request against the running server
#   all        preflight, download, check, start
#
# THE RELEASE IS A DIRECTORY, NOT A FILE
#   Unlike a single-file GGUF, this artifact is a GVQ1 release: the 56 parts sit
#   beside the tables that describe them (files.json, tensor-manifest.json,
#   tokenizer.json, tokenizer_config.json, chat_template.jinja). The host reads
#   them from the directory it is given, so -m points at the DIRECTORY and the
#   tables must stay with the parts. Flattening the release into a shared models
#   folder breaks it.
#
# THE TWO FLAGS THAT ARE NOT OPTIONAL
#   --mtp-mode on is required. The family reports mtp_support unqualified, and an
#   automatic MTP mode resolves to off in that case, so --mtp-draft 2 alone is
#   silently discarded and the plan still reads mtp=off
#   (crates/ds4-core/src/serving.rs:2410 resolves Auto -> Off, and
#   open_draft_tokens returns None when the mode is Off, serving.rs:2517).
#   --mtp-draft 2 then loads the embedded predictor, which this artifact ships as
#   four parts of its own. There is no separate --mtp file: the predictor is part
#   of the release, so a path never appears on this command line.
#
# THE SPECULATOR, AND WHAT WAS MEASURED
#   The continuous ("batched") decode loop keeps its own speculation controls
#   instead of reading the plan: mode 0=off, 1=non-accepting probe, 2=verify+
#   accept, with the depth in DS4_CONT_MTP_DEPTH (ds4.c:62014-62031). The C oracle
#   server arms mode 2 by itself whenever an MTP source is loaded
#   (ds4_server.c:23970); the Rust host does not, so this launcher exports both
#   variables when MTP is on.
#   Measured on the reference Spark, 2026-10-17, engine checkout optimize4:
#   with --mtp-mode on --mtp-draft 2 AND DS4_CONT_MTP_MODE=2 DS4_CONT_MTP_DEPTH=2
#   confirmed inside the running process's environment, the plan reports mtp=on
#   weights=true and the log still warns "MTP execution is present but not
#   qualified", yet every request reports tok_per_step 1.0. Drafts are loaded and
#   offered; the served lane does not commit them on this build. The artifact
#   card's 34.6-40.6 tok/s figures come from the single-stream greedy bench path
#   (ds4-bench / the ds4 CLI, whose use_glm_spec gate lives in ds4-cli), not from
#   this HTTP lane. Set MTP_MODE=off to stop paying for the predictor state.
#
# THROUGHPUT, HONESTLY
#   Measured on the reference Spark at temperature 0, 256-token completions:
#   one stream 24.2-24.3 tok/s; two concurrent streams 12.1 and 12.0 tok/s each,
#   with the same wall time for the pair (22.67 s) as running them back to back
#   (22.65 s). Two banks buy a second client not waiting, not extra tokens per
#   second: decode is bytes per token against the weight stream and batching two
#   rows does not halve those bytes. Time to first token also rose, 222 ms alone
#   against 1592 ms when the two prompts arrived together.
#
# SAMPLING AND THINKING (card behaviour that surprises people)
#   The template opens a thinking channel on every reply, so a small max_tokens is
#   consumed entirely by reasoning_content and content comes back empty: give it
#   room, or send reasoning_effort low. temperature 0 is reproducible; the default
#   when a request omits sampling fields is the artifact's own recipe.
#
# USAGE
#   bash start-glm53-flash-vq.sh preflight
#   bash start-glm53-flash-vq.sh download
#   bash start-glm53-flash-vq.sh check
#   bash start-glm53-flash-vq.sh start
#   bash start-glm53-flash-vq.sh all
#   bash start-glm53-flash-vq.sh stop | status | logs | test | verify
#
#   PORT=8008 bash start-glm53-flash-vq.sh start       another port
#   CTX=65536 bash start-glm53-flash-vq.sh start       shallower context
#   MTP_MODE=off bash start-glm53-flash-vq.sh start    no predictor state
#   MAX_SEQS=1 bash start-glm53-flash-vq.sh start      one sequence, no second bank
#   GLMVQ_ALLOW_SHARED=1 ... start                     overrule the single-owner probe
#   HF_TOKEN=hf_... bash ... download                  authenticated fetch, higher rate limit
#
# Notes
#   - The repository is public: no token is needed, and one is never written into
#     this file. A token does raise the CDN rate limit; the CLI picks up HF_TOKEN
#     from the environment on its own.
#   - Every part is checked against the published SHA256SUMS by the engine at load
#     (about 20 s for this artifact), so verify is a second, independent opinion
#     rather than a precondition.
#   - The engine holds a single global lock, so one ds4 model runs at a time on
#     the box: a second process refuses to start by design.

set -Eeuo pipefail

# --------------------------------------------------------------------------
# Configuration (every value is overridable from the environment)
# --------------------------------------------------------------------------
REPO_ID="${REPO_ID:-Baekpica/GLM-5.3-Flash-VQ-GGUF}"
REVISION="${REVISION:-}"                  # empty tracks main; a commit pins the bytes
RELEASE_NAME="GLM-5.3-Flash-VQ-GGUF"
DIR="${DIR:-$HOME/models}"
MODEL_DIR="${MODEL_DIR:-$DIR/$RELEASE_NAME}"

SERVER_BIN="${SERVER_BIN:-$HOME/ds4-dfm-rs/ds4-server}"
HOST_ADDR="${HOST_ADDR:-0.0.0.0}"
# 8000 is DeepSeek V4.1 Flash, 8003 Qwen3.8, 8005 Bonsai and 8006 YoungAi in this
# home, so this artifact takes the next free port.
PORT="${PORT:-8007}"
MODEL_ID="${MODEL_ID:-glm-5.3-flash-vq}"
CTX="${CTX:-}"                            # empty keeps the native 1,048,576
MAX_SEQS="${MAX_SEQS:-}"                  # empty leaves the plan's auto width
MTP_MODE="${MTP_MODE:-on}"                # see the header: auto resolves to off
DRAFT="${DRAFT:-2}"                       # the card's measured draft count

LOGS="$DIR/logs"
PIDF="$DIR/glm53flash.pid"
WDPIDF="$DIR/glm53flash-watchdog.pid"
INFOF="$DIR/glm53flash.info"
SERVER_LOG="$LOGS/glm53flash-server.log"
WATCHDOG_LOG="$LOGS/glm53flash-watchdog.log"

GGUF_PARTS=56                             # parts listed in files.json at revision 4da527cd
PAYLOAD_BYTES=77272028416                 # the 56 GGUF parts, 71.96 GiB
NEED_GB=80                                # payload plus slack for partial files
READY_TIMEOUT="${READY_TIMEOUT:-900}"     # loading hashes 72 GiB before the port binds
START_FLOOR_MB="${START_FLOOR_MB:-100000}"  # before loading
RUN_FLOOR_MB="${RUN_FLOOR_MB:-10000}"     # after loading, before the box starts swapping
KILL_MB="${KILL_MB:-2500}"                # watchdog: two samples below this and the server stops
WATCHDOG="${WATCHDOG:-1}"
HF_BIN="${HF_BIN:-}"
HF="${HF:-}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-8}"
ENDPOINT="${ENDPOINT:-}"
NO_XET="${NO_XET:-0}"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
say()  { echo "[glm53 $(date +%H:%M:%S)] $*"; }
warn() { echo "[glm53 $(date +%H:%M:%S)] warning: $*" >&2; }
die()  { echo "[glm53] ERROR: $*" >&2; exit 1; }

avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
total_mb() { awk '/MemTotal/{print int($2/1024)}' /proc/meminfo; }
fsize()    { stat -L -c %s "$1" 2>/dev/null || echo 0; }
df_avail() { df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' \n' || true; }
df_target(){ df --output=target "$1" 2>/dev/null | tail -1 || true; }
alive()    { [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
human()    { awk -v b="$1" 'BEGIN{printf "%.1f GB", b/1000000000}'; }

# A complete release has every part, its tables, and no partial file left behind.
# Sizes and hashes themselves are the loader's business: it verifies all of them.
parts_present() { find "$MODEL_DIR" -maxdepth 1 -name '*.gguf' 2>/dev/null | wc -l; }
parts_partial() { find "$MODEL_DIR" -maxdepth 1 -name '*.incomplete' 2>/dev/null | wc -l; }
release_ok() {
  [ -f "$MODEL_DIR/files.json" ] || return 1
  [ -f "$MODEL_DIR/tensor-manifest.json" ] || return 1
  [ -f "$MODEL_DIR/tokenizer_config.json" ] || return 1
  [ -f "$MODEL_DIR/chat_template.jinja" ] || return 1
  [ "$(parts_present)" -eq "$GGUF_PARTS" ] || return 1
  [ "$(parts_partial)" -eq 0 ]
}

ensure_hf() {
  if [ -n "$HF_BIN" ]; then HF="$HF_BIN"; return; fi
  if command -v hf >/dev/null 2>&1 && hf download --help >/dev/null 2>&1; then HF=hf; return; fi
  if command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli; return; fi
  HF="$DIR/.hfenv/bin/hf"
  [ -x "$HF" ] && return
  say "installing the Hugging Face CLI into $DIR/.hfenv (the system Python is left alone)"
  mkdir -p "$DIR"
  python3 -m venv "$DIR/.hfenv" || die "python3 venv failed; install it with: sudo apt install python3-venv"
  "$DIR/.hfenv/bin/pip" install -q -U "huggingface_hub[hf_xet]" \
    || die "pip install huggingface_hub failed (behind a firewall? try ENDPOINT= and NO_XET=1)"
  HF="$DIR/.hfenv/bin/hf"
}

hf_env() {
  [ -n "$ENDPOINT" ] && export HF_ENDPOINT="$ENDPOINT"
  [ "$NO_XET" = 1 ] && export HF_HUB_DISABLE_XET=1
  export HF_HUB_MAX_WORKERS="$HF_MAX_WORKERS"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
  return 0
}

# hf download resumes by itself: a re-run continues from the files already there.
hf_download() {
  if [ -n "$REVISION" ]; then
    "$HF" download "$REPO_ID" --revision "$REVISION" --local-dir "$MODEL_DIR" "$@"
  else
    "$HF" download "$REPO_ID" --local-dir "$MODEL_DIR" "$@"
  fi
}

# --------------------------------------------------------------------------
# preflight: the machine checks, plus this box's one-model rule
# --------------------------------------------------------------------------
preflight() {
  local fatal=0

  if [ "$(uname -s)/$(uname -m)" = "Linux/aarch64" ]; then
    say "arch ok: Linux/aarch64"
  else
    warn "this is $(uname -s)/$(uname -m); the ds4 CUDA host and its kernels target the DGX Spark (sm_121)"; fatal=1
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu; gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    case "$gpu" in
      *GB10*) say "gpu ok: $gpu";;
      *) warn "GPU is '$gpu'; the CUDA kernels are built for GB10 (sm_121)"; fatal=1;;
    esac
  else
    warn "nvidia-smi not found: no NVIDIA driver on this machine"; fatal=1
  fi

  local ldc miss=() lib
  ldc="$(command -v ldconfig || echo /sbin/ldconfig)"
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

  local t; t="$(total_mb)"
  if [ "$t" -ge 115000 ]; then
    say "memory ok: ${t} MB total"
  else
    warn "only ${t} MB of memory; the plan quotes 118.4 GB for two banks at the native context"; fatal=1
  fi
  say "memory available now: $(avail_mb) MB (start refuses below ${START_FLOOR_MB} MB)"

  mkdir -p "$DIR" 2>/dev/null || { warn "cannot create $DIR"; fatal=1; }
  local need=0 free
  if ! release_ok; then
    need="$PAYLOAD_BYTES"
  fi
  free="$(df_avail "$DIR")"
  if [ -n "$free" ] && [ "$free" -gt $((need + 5000000000)) ]; then
    say "disk ok: $(human $need) still to fetch, $(human $free) free on $(df_target "$DIR")"
  else
    warn "not enough room on $(df_target "$DIR"): want $(human $need) plus 5 GB working room, have $(human ${free:-0})"
    warn "set DIR= to a volume with room (the release needs about ${NEED_GB} GB in total)"; fatal=1
  fi

  # One resident model. The engine takes a single global lock by design, and this
  # box already answers on other ports; the probe catches the case the lock does
  # not, an owner that is still loading.
  local busy=() p
  if command -v curl >/dev/null 2>&1; then
    for p in 8000 8003 8004 8005 8006; do
      [ "$p" = "$PORT" ] && continue
      curl -fsS --max-time 2 "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && busy+=("$p")
    done
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      warn "port $PORT already answers /v1/models; pick another with PORT="; fatal=1
    fi
  fi
  if [ "${#busy[@]}" -gt 0 ]; then
    warn "these ports already serve a model: ${busy[*]} - stop that one first, or expect the engine's lock to refuse this start"
  fi
  if pgrep -af 'ds4-server' | grep -qv "$$"; then
    warn "a ds4-server process is already running: $(pgrep -af 'ds4-server' | head -2 | tr '\n' ' ')"
  fi

  [ -x "$SERVER_BIN" ] && say "runtime ok: $SERVER_BIN" \
    || warn "no executable at $SERVER_BIN; build the host with: cd ~/ds4-dfm-rs && make cuda-spark"

  [ "$fatal" = 0 ] && say "preflight: this machine can run the artifact" || return 1
}

# --------------------------------------------------------------------------
# plan: what download and start would do, touching nothing
# --------------------------------------------------------------------------
cmd_plan() {
  local free
  say "install dir:   $DIR"
  say "release dir:   $MODEL_DIR"
  say "runtime:       $SERVER_BIN"
  say "serve:         $HOST_ADDR:$PORT, model id $MODEL_ID, two banks, watchdog below ${KILL_MB} MB"
  say ""
  say "artifacts"
  say "  parts        $GGUF_PARTS GGUF files, $(human $PAYLOAD_BYTES)   $([ "$(parts_present)" -eq "$GGUF_PARTS" ] && echo present || echo "$(parts_present) on disk")"
  say "  tables       files.json, tensor-manifest.json, tokenizer*, chat_template.jinja"
  say "  predictor    the 4 embedded MTP parts (no --mtp path: they ship inside the release)"
  say ""
  say "the command start would run:"
  say "  $SERVER_BIN --cuda -m $MODEL_DIR \\"
  say "      --mtp-mode $MTP_MODE --mtp-draft $DRAFT --host $HOST_ADDR --port $PORT \\"
  say "      --model-id $MODEL_ID --no-update-check${CTX:+ -c $CTX}"
  if [ "$MTP_MODE" != "off" ]; then
    say "  with DS4_CONT_MTP_MODE=2 DS4_CONT_MTP_DEPTH=$DRAFT in the environment (see the header)"
  fi
  say "  (no -c unless CTX is set: the family defaults to the native 1048576)"
  mkdir -p "$DIR"
  free="$(df_avail "$DIR")"
  say ""
  say "still to fetch: $(human $([ "$(parts_present)" -eq "$GGUF_PARTS" ] && echo 0 || echo $PAYLOAD_BYTES)); free on target $(df_target "$DIR"): $(human ${free:-0})"
  say "ports 8000, 8003, 8004, 8005, 8006, 8008 and 8009 answer for the other launchers in this home; this one defaults to $PORT"
}

# --------------------------------------------------------------------------
# download / verify / check
# --------------------------------------------------------------------------
cmd_download() {
  mkdir -p "$MODEL_DIR"
  if release_ok; then
    say "release already complete: $(parts_present) parts in $MODEL_DIR"
    return 0
  fi
  ensure_hf; hf_env
  local free; free="$(df_avail "$DIR")"
  [ -z "$free" ] || [ "$free" -gt $((PAYLOAD_BYTES + 5000000000)) ] \
    || die "$MODEL_DIR has $(human ${free:-0}) free, needs $(human $((PAYLOAD_BYTES + 5000000000)))"
  say "fetching $REPO_ID${REVISION:+ @ $REVISION} into $MODEL_DIR"
  say "an interrupted run resumes from where it stopped; run the same command again"
  hf_download || die "download failed; run the same command again to resume"
  release_ok || die "the fetch finished but the release is incomplete: $(parts_present)/$GGUF_PARTS parts, $(parts_partial) partial"
  say "done: $GGUF_PARTS parts, $(parts_present) present, $(human "$(du -sb "$MODEL_DIR" | cut -f1)") on disk"
  say "next: bash $0 check && bash $0 start"
}

cmd_verify() {
  release_ok || die "the release is incomplete in $MODEL_DIR; run download first"
  command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
  say "hashing $(human "$(du -sb "$MODEL_DIR" | cut -f1)"); this reads every byte"
  ( cd "$MODEL_DIR" && sha256sum -c SHA256SUMS ) || die "a payload hash does not match SHA256SUMS"
  say "verify: every payload hash matches SHA256SUMS"
}

# --check-config prints the plan as JSON and opens no weights, so this is the
# cheapest way to see the family, the context and the memory quote before a load.
cmd_check() {
  release_ok || die "the release is incomplete in $MODEL_DIR; run download first"
  [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN (build it: cd ~/ds4-dfm-rs && make cuda-spark)"
  say "release: $(( $(parts_present) ))/$GGUF_PARTS parts, $(human "$(du -sb "$MODEL_DIR" | cut -f1)") on disk"
  local args=(--check-config -m "$MODEL_DIR")
  [ -n "$CTX" ] && args+=(-c "$CTX")
  [ "$MTP_MODE" = "off" ] || args+=(--mtp-mode "$MTP_MODE" --mtp-draft "$DRAFT")
  "$SERVER_BIN" "${args[@]}"
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
    say "already running (pid $(cat "$PIDF"), $(cat "$INFOF" 2>/dev/null)); stop it first"
    return 0
  fi
  release_ok || die "the release is incomplete in $MODEL_DIR; run: bash $0 download"
  [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN; build the host first"
  command -v curl >/dev/null 2>&1 || die "curl is required"

  local owners
  owners="$(pgrep -af 'ds4-server' | grep -v "$$" || true)"
  if [ -n "$owners" ] && [ "${GLMVQ_ALLOW_SHARED:-0}" != "1" ]; then
    printf '[glm53] another model owner is running:\n%s\n' "$owners" >&2
    die "stop it first, or set GLMVQ_ALLOW_SHARED=1 to overrule (one ds4 model fits on this box)"
  fi

  local a; a="$(avail_mb)"
  [ "$a" -ge "$START_FLOOR_MB" ] \
    || die "only $a MB of memory available, need $START_FLOOR_MB: stop another model first"

  mkdir -p "$LOGS"
  local args=(--cuda -m "$MODEL_DIR" --host "$HOST_ADDR" --port "$PORT"
              --model-id "$MODEL_ID" --no-update-check)
  local spec=()
  if [ "$MTP_MODE" = "off" ]; then
    args+=(--mtp-mode off)
  else
    args+=(--mtp-mode "$MTP_MODE" --mtp-draft "$DRAFT")
    spec=(DS4_CONT_MTP_MODE=2 DS4_CONT_MTP_DEPTH="$DRAFT")
  fi
  [ -n "$CTX" ] && args+=(-c "$CTX")
  [ -n "$MAX_SEQS" ] && args+=(--max-seqs "$MAX_SEQS")

  say "starting ds4-server on $HOST_ADDR:$PORT (hashes 72 GiB and maps the weights: about a minute)"
  say "  log: $SERVER_LOG"
  nohup env "${spec[@]}" "$SERVER_BIN" "${args[@]}" >"$SERVER_LOG" 2>&1 </dev/null &
  echo $! >"$PIDF"
  echo "$MODEL_ID on $HOST_ADDR:$PORT${MTP_MODE:+ mtp=$MTP_MODE/$DRAFT}" >"$INFOF"

  if [ "$WATCHDOG" = 1 ]; then
    export -f watchdog avail_mb
    nohup bash -c "watchdog $(cat "$PIDF") $KILL_MB '$WATCHDOG_LOG'" >/dev/null 2>&1 </dev/null &
    echo $! >"$WDPIDF"
    say "  watchdog pid $(cat "$WDPIDF"), stops the server if MemAvailable stays below ${KILL_MB} MB"
  fi

  local waited=0
  say "waiting for the endpoint (up to ${READY_TIMEOUT}s; the port binds only after the release is verified)"
  while [ "$waited" -lt "$READY_TIMEOUT" ]; do
    if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then break; fi
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
  # The plan line and its warnings are the artifact's own account of what it will
  # and will not execute; quote them rather than paraphrasing.
  grep -aE 'effective:|^warn:' "$SERVER_LOG" | tail -3 | sed 's/^/  /' || true
  cmd_test || warn "the smoke request failed; the server is up, so check $SERVER_LOG"
  say ""
  say "ready: http://$HOST_ADDR:$PORT/v1   (OpenAI: /v1/chat/completions, /v1/responses; Anthropic: /v1/messages)"
  say "stop with: bash $0 stop   context: ${CTX:-native 1048576}   banks: ${MAX_SEQS:-auto (2)}"
}

cmd_stop() {
  [ -s "$WDPIDF" ] && kill "$(cat "$WDPIDF")" 2>/dev/null || true
  if alive; then
    local pid; pid="$(cat "$PIDF")"
    say "stopping ds4-server (pid $pid)"
    kill "$pid" 2>/dev/null || true
    local i=0
    while [ "$i" -lt 30 ] && kill -0 "$pid" 2>/dev/null; do sleep 1; i=$((i + 1)); done
    kill -0 "$pid" 2>/dev/null && { say "  still alive after 30s, sending KILL"; kill -9 "$pid" 2>/dev/null || true; }
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
  say "release:       $(( $(parts_present) ))/$GGUF_PARTS parts in $MODEL_DIR $([ "$(parts_partial)" = 0 ] && echo '(no partial files)' || echo "($(parts_partial) PARTIAL)")"
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | head -c 400; then
    say ""
    say "endpoint:      answering on port $PORT"
  else
    say "endpoint:      no answer on port $PORT"
  fi
  if [ -f "$SERVER_LOG" ]; then
    say ""
    say "server log tail:"
    tail -n 5 "$SERVER_LOG" | sed 's/^/  /'
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
  command -v curl >/dev/null 2>&1 || die "curl is required for the smoke test"
  local out
  # 64 tokens: the template opens a thinking channel first, so a shorter cap would
  # return an empty content field and look like a failure.
  out="$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"temperature\":0,\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, what does a vector-quantized expert layer store?\"}]}")"
  case "$out" in
    *'"choices"'*) printf '%s\n' "$out" | sed -n 's/.*"timings":{\(.*\)}.*/  timings: \1/p' | head -1
                   printf '%s\n' "$out" | sed -n 's/.*"reasoning_content":"\([^"]\{0,90\}\)".*/  thinking: \1.../p' | head -1
                   printf '%s\n' "$out" | sed -n 's/.*"content":"\([^"]\{0,90\}\)".*/  answer: \1/p' | head -1;;
    *) warn "the request failed: $(printf '%s' "$out" | head -c 300)"; return 1;;
  esac
}

# --------------------------------------------------------------------------
usage() {
  sed -n '3,75p' "$0" | sed 's/^# \{0,1\}//'
}

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
  *) echo "unknown command: $1 (see help)" >&2; usage >&2; exit 2;;
esac
