#!/usr/bin/env bash
#
# DeepSeek-V4.1-Flash-1017 VQ on the DGX Spark: one 113.56 GB base built from 40
# byte-range parts, one 971 MB vision sidecar, and the 1017 Engram tables read in
# place from disk (2 x 101.5 GB).
#
# DeepSeek's October 17 V4.1 Flash checkpoint run through YoungAi's
# vq8sh14-q4k-mtpnative recipe, unchanged, and served by ds4-dfm-rs at the
# artifact's native 1,048,576-token context. The model's own three-tower DSpark
# drafter is inside the base GGUF.
#
# WHERE THIS RUNS
#   On the Spark itself (GB10, sm_121, aarch64, 121.6 GiB unified memory), as the
#   user who owns the model directory. The download is 318 GB, so it happens on
#   the box that serves it. Nothing needs sudo. Engines live in ~/ds4-dfm-rs and
#   weights in ~/models on the reference box.
#
# WHAT IT DOES
#   preflight  GPU, CUDA 13 runtime, memory, both disks, ports, second-owner probe
#   plan       what would be fetched, assembled and started, with sizes
#   download   base parts, vision sidecar and the 1017 Engram tables, resumable
#   assemble   append the 40 parts into the base and delete each part as it lands
#   verify     SHA256SUMS over the base, the vision sidecar and every part
#   build      check the ds4-dfm-rs binaries and the flags they must advertise
#   start      serve under a memory watchdog; waits for /v1/models, then smokes it
#   stop       stop the server and the watchdog
#   status     pid, port, free memory, base and Engram completeness, log tail
#   logs       the server and watchdog logs
#   test       one chat request against the running server
#   all        preflight, download, assemble, verify, build, start
#
# THE THREE MISTAKES THIS SCRIPT EXISTS TO PREVENT
#   1. Using V4.1's Engram tables with 1017 weights. 1017 changed all 48 shards,
#      including the two that hold the tables. The names and sizes are identical,
#      so an engine pointed at the older tables loads them without a word and
#      runs a mismatched model. Fetch them from THIS repository (deepseek-engram/)
#      or from your own 1017 checkpoint; the launcher checks the two published
#      SHA-256 digests and refuses to start on a mismatch.
#   2. Copying an older V4.1 vision sidecar in. Same size, same 306 tensors,
#      532,623,100 of its 971,066,016 bytes different. Same silent failure.
#   3. Mounting YoungAi's V4.1 domain sidecars (zchain / GR-RB). They were solved
#      against the V4.1 base and do not fit these weights; this artifact ships no
#      sidecar and this launcher passes none.
#
# DISK
#   The download is 318 GB and the naive assembly needs another 114 GB free while
#   it runs. The append-and-delete loop in `assemble` holds only ONE extra part
#   (2.8 GB): each part is appended and removed before the next is read.
#   BASE_DIR and ENGRAM_DIR are separate on purpose, because the two halves have
#   opposite read patterns. The base is read once at load and then stays resident
#   (~106 GiB), so it belongs on the roomy volume. The Engram tables are read in
#   rows on the tokens that match an n-gram, so they belong on the fastest device
#   you have; the artifact card asks for a local SSD for exactly that reason.
#   On the reference Spark: base under /mnt/models (953 GB), tables on NVMe.
#
# THINKING AND SAMPLING
#   YoungAi reports that this weight recipe under-weights stop decisions
#   (response, EOS), so a long thinking segment can fail to close; its server
#   therefore defaults to non-thinking mode, and ds4-dfm-rs does the same for
#   this family (thinking is opt-in per request). DeepSeek recommends temperature
#   1.0 and top_p 1.0 for 1017 and says to send them explicitly, because sampling
#   follows the request on this path. The launcher's smoke test sends both.
#
# WHAT HAS NOT BEEN MEASURED
#   No 1017 speed or fidelity figure exists yet, here or upstream. The numbers in
#   the runbook are V4.1 Flash's, for the same recipe and the same runtime, and
#   are labelled as such. Drafter acceptance depends on generated text and can
#   differ between the two checkpoints.
#
# USAGE
#   bash start-ds41-1017.sh preflight
#   bash start-ds41-1017.sh download && bash start-ds41-1017.sh assemble
#   bash start-ds41-1017.sh verify && bash start-ds41-1017.sh build
#   bash start-ds41-1017.sh start
#   bash start-ds41-1017.sh all
#   bash start-ds41-1017.sh stop | status | logs | test
#
#   BASE_DIR=/mnt/models/ds41-1017 bash start-ds41-1017.sh download
#   ENGRAM_DIR=/mnt/big/ds41-engram bash start-ds41-1017.sh download
#     put the 114 GB base and the 203 GB of tables on different devices
#   PORT=8009 MTP_MODE=off bash start-ds41-1017.sh start
#   NO_VISION=1 bash start-ds41-1017.sh start        text only, no sidecar mounted
#   HF_TOKEN=hf_... bash start-ds41-1017.sh download authenticated fetch
#   ALLOW_SHARED=1 bash start-ds41-1017.sh start     overrule the single-owner probe
#
# Notes
#   - The repository is public: no token is required, and one is never written
#     into this file. A token only raises the CDN rate limit.
#   - Re-running download resumes wherever it stopped; re-running assemble skips
#     the parts already appended because it records progress as it goes.
#   - One ds4 model runs at a time: the engine takes a single global lock.

set -Eeuo pipefail

# --------------------------------------------------------------------------
# Configuration (every value is overridable from the environment)
# --------------------------------------------------------------------------
REPO_ID="${REPO_ID:-Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF}"
REVISION="${REVISION:-f81ee2e9801fcc58fd51ae75dd827467ffc0b9e5}"

DIR="${DIR:-$HOME/models}"                       # logs, pid files, the hf venv
# The base is read once at load and then resident, so it goes on the roomy volume.
# The Engram tables are read in rows at run time, so they go on the fastest device.
# On the reference Spark: /mnt/models (953 GB) for the base, NVMe for the tables.
BASE_DIR="${BASE_DIR:-/mnt/models/ds41-1017}"
ENGRAM_DIR="${ENGRAM_DIR:-$HOME/models/ds41-1017-engram}"

GGUF="DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf"
GGUF_BYTES=113556639424
GGUF_SHA256=9f3afe6a46ff8de711911dfad5994519a4e5dd57b163f88a5bf4111c308d0e6f
NPART=40
PART_LONG=2838915986          # bytes in parts 1..24
PART_SHORT=2838915985         # bytes in parts 25..40
PART_LONG_N=24

VISION="DeepSeek-V4.1-Flash.vision.gguf"
VISION_BYTES=971066016
VISION_SHA256=3a8b194a011e630e8e49a51b5f4e9821e48a9e5674e678d52c0302d6b164698c

# The 1017 n-gram tables, copied unmodified from the official 1017 checkpoint.
# The V4.1 shards carry the same names and sizes and MUST NOT be used here.
ENGRAM_SUBDIR="deepseek-engram"
SHARDS=(model-00047-of-00048.safetensors model-00048-of-00048.safetensors)
SHARD_BYTES=(101535150936 101537926640)
SHARD_SHA256=(76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5 \
              9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd)

SERVER_BIN="${SERVER_BIN:-$HOME/ds4-dfm-rs/ds4-server}"
CLI_BIN="${CLI_BIN:-$HOME/ds4-dfm-rs/ds4}"
BENCH_BIN="${BENCH_BIN:-$HOME/ds4-dfm-rs/ds4-bench}"

HOST_ADDR="${HOST_ADDR:-0.0.0.0}"
# 8000 is the deepseek-v4.1-flash-ds4 entry, 8003 Qwen3.8, 8005 Bonsai,
# 8006 the YoungAi V4.1 launcher, 8007 GLM-5.3 VQ and 8009 GLM-5.3 Uncensored
# VQ in this home.
PORT="${PORT:-8008}"
MTP_MODE="${MTP_MODE:-auto}"            # off | auto | on: the embedded DSpark drafter
BUDGET_MB="${BUDGET_MB:-110000}"        # the artifact card's engine budget
NO_VISION="${NO_VISION:-0}"
CTX="${CTX:-}"                          # empty takes the native 1,048,576

LOGS="$DIR/logs"
PIDF="$DIR/ds41-1017.pid"
WDPIDF="$DIR/ds41-1017-watchdog.pid"
INFOF="$DIR/ds41-1017.info"
SERVER_LOG="$LOGS/ds41-1017-server.log"
WATCHDOG_LOG="$LOGS/ds41-1017-watchdog.log"
PROGRESS="$BASE_DIR/.assemble-progress"

NEED_GB=325                              # 318 GB payload plus assembly slack
START_FLOOR_MB="${START_FLOOR_MB:-100000}"
RUN_FLOOR_MB="${RUN_FLOOR_MB:-8000}"
KILL_MB="${KILL_MB:-2500}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"    # loading maps 113 GB; the card reports ~2 min
WATCHDOG="${WATCHDOG:-1}"
HF_BIN="${HF_BIN:-}"
HF="${HF:-}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-8}"
ENDPOINT="${ENDPOINT:-}"
NO_XET="${NO_XET:-0}"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
say()  { echo "[ds41-1017 $(date +%H:%M:%S)] $*"; }
warn() { echo "[ds41-1017 $(date +%H:%M:%S)] warning: $*" >&2; }
die()  { echo "[ds41-1017] ERROR: $*" >&2; exit 1; }

avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
total_mb() { awk '/MemTotal/{print int($2/1024)}' /proc/meminfo; }
fsize()    { stat -L -c %s "$1" 2>/dev/null || echo 0; }
df_avail() { df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' \n' || true; }
df_target(){ df --output=target "$1" 2>/dev/null | tail -1 || true; }
alive()    { [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
human()    { awk -v b="$1" 'BEGIN{printf "%.1f GB", b/1000000000}'; }
sha_of()   { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }

part_name() { printf '%s.part%02d-of-%d' "$GGUF" "$1" "$NPART"; }
part_bytes() { if [ "$1" -le "$PART_LONG_N" ]; then echo "$PART_LONG"; else echo "$PART_SHORT"; fi; }
# Bytes held by parts 1..k, so a resumed assembly can be truncated to a known-good length.
parts_len() { local k="$1" n; if [ "$k" -le "$PART_LONG_N" ]; then echo $(( k * PART_LONG )); else n=$(( k - PART_LONG_N )); echo $(( PART_LONG_N * PART_LONG + n * PART_SHORT )); fi; }

# SHA256SUMS is the artifact's own manifest and wins where it lists a file; the
# constants above are the fallback and the cross-check for the two Engram shards,
# which matter most because a V4.1 copy would load silently.
sum_of() {
  local f="$1" manifest="$2"
  [ -s "$manifest" ] || return 1
  awk -v f="$f" '$2 == f {print $1; found=1} END{exit !found}' "$manifest"
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

hf_get() {   # $1 = local dir, rest = paths/patterns
  local dir="$1"; shift
  "$HF" download "$REPO_ID" --revision "$REVISION" --local-dir "$dir" "$@"
}

base_ok()   { [ "$(fsize "$BASE_DIR/$GGUF")" = "$GGUF_BYTES" ]; }
vision_ok() { [ "$(fsize "$BASE_DIR/$VISION")" = "$VISION_BYTES" ]; }
engram_ok() {
  [ "$(fsize "$ENGRAM_DIR/${SHARDS[0]}")" = "${SHARD_BYTES[0]}" ] &&
  [ "$(fsize "$ENGRAM_DIR/${SHARDS[1]}")" = "${SHARD_BYTES[1]}" ]
}

# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------
preflight() {
  local fatal=0

  if [ "$(uname -s)/$(uname -m)" = "Linux/aarch64" ]; then
    say "arch ok: Linux/aarch64"
  else
    warn "this is $(uname -s)/$(uname -m); the ds4 CUDA host targets the DGX Spark (sm_121)"; fatal=1
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu; gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    case "$gpu" in
      *GB10*) say "gpu ok: $gpu";;
      *) warn "GPU is '$gpu'; the kernels are built for GB10 (sm_121)"; fatal=1;;
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
  fi

  local t; t="$(total_mb)"
  if [ "$t" -ge 115000 ]; then
    say "memory ok: ${t} MB total (budget ${BUDGET_MB} MB, floor ${START_FLOOR_MB} MB before loading)"
  else
    warn "only ${t} MB of memory; a 113.6 GB base needs a 128 GB DGX Spark"; fatal=1
  fi

  mkdir -p "$BASE_DIR" "$ENGRAM_DIR" 2>/dev/null || { warn "cannot create $BASE_DIR"; fatal=1; }

  # Base and Engram may sit on different devices, so each is checked on its own.
  local need_base=0 free_base need_engram=0 free_engram
  base_ok || need_base=$(( GGUF_BYTES + PART_LONG ))
  vision_ok || need_base=$(( need_base + VISION_BYTES ))
  engram_ok || need_engram=$(( SHARD_BYTES[0] + SHARD_BYTES[1] ))
  free_base="$(df_avail "$BASE_DIR")"; free_engram="$(df_avail "$ENGRAM_DIR")"
  say "base dir:   $BASE_DIR on $(df_target "$BASE_DIR"), $(human "${free_base:-0}") free, $(human $need_base) to fetch"
  say "engram dir: $ENGRAM_DIR on $(df_target "$ENGRAM_DIR"), $(human "${free_engram:-0}") free, $(human $need_engram) to fetch"
  if [ -n "$free_base" ] && [ "$free_base" -lt $((need_base + 5000000000)) ]; then
    warn "not enough room for the base and the assembly slack (one part, $(human "$PART_LONG"))"; fatal=1
  fi
  if [ "$ENGRAM_DIR" != "$BASE_DIR" ] && [ -n "$free_engram" ] && [ "$free_engram" -lt $((need_engram + 2000000000)) ]; then
    warn "not enough room for the Engram tables in $ENGRAM_DIR"; fatal=1
  fi

  local busy=() p
  if command -v curl >/dev/null 2>&1; then
    for p in 8000 8003 8005 8006 8007; do
      [ "$p" = "$PORT" ] && continue
      curl -fsS --max-time 2 "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && busy+=("$p")
    done
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      warn "port $PORT already answers /v1/models; pick another with PORT="; fatal=1
    fi
  fi
  if [ "${#busy[@]}" -gt 0 ]; then
    warn "these ports already serve a model: ${busy[*]} - one ds4 model runs at a time here"
  fi

  [ "$fatal" = 0 ] && say "preflight: this machine can hold and run the artifact" || return 1
}

# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
cmd_plan() {
  say "base dir:      $BASE_DIR   ($(df_target "$BASE_DIR"), $(human "$(df_avail "$BASE_DIR")") free)"
  say "engram dir:    $ENGRAM_DIR ($(df_target "$ENGRAM_DIR"), $(human "$(df_avail "$ENGRAM_DIR")") free)"
  say "runtime:       $SERVER_BIN"
  say "serve:         $HOST_ADDR:$PORT, --mem-budget-mb $BUDGET_MB, watchdog below ${KILL_MB} MB"
  say ""
  say "artifacts"
  say "  base         $NPART parts -> $(human $GGUF_BYTES)   $(base_ok && echo assembled || echo "not assembled ($(ls "$BASE_DIR"/$GGUF.part*-of-$NPART 2>/dev/null | wc -l)/$NPART parts on disk)")"
  say "  vision       $VISION ($(human $VISION_BYTES))   $(vision_ok && echo present || echo missing)"
  say "  engram       ${SHARDS[0]}, ${SHARDS[1]}   $(engram_ok && echo present || echo missing)"
  say ""
  say "the command start would run:"
  say "  $SERVER_BIN --cuda -m $BASE_DIR/$GGUF \\"
  say "      --engram-dir $ENGRAM_DIR \\"
  [ "$NO_VISION" = 1 ] || say "      --vision-gguf $BASE_DIR/$VISION \\"
  say "      --mtp-mode $MTP_MODE --mem-budget-mb $BUDGET_MB --host $HOST_ADDR --port $PORT${CTX:+ -c $CTX}"
  say ""
  say "still to fetch: base $(human $([ "$(base_ok && echo 1)" = 1 ] && echo 0 || echo $(( GGUF_BYTES + PART_LONG )))) + vision $(human $([ "$(vision_ok && echo 1)" = 1 ] && echo 0 || echo $VISION_BYTES)) + engram $(human $([ "$(engram_ok && echo 1)" = 1 ] && echo 0 || echo $(( SHARD_BYTES[0] + SHARD_BYTES[1] ))))"
  say "the Engram tables must be the 1017 copies; a V4.1 table under the same name loads silently and runs the wrong model"
}

# --------------------------------------------------------------------------
# download / assemble / verify / build
# --------------------------------------------------------------------------
cmd_download() {
  ensure_hf; hf_env
  mkdir -p "$BASE_DIR" "$ENGRAM_DIR"

  if base_ok && vision_ok; then
    say "base and vision already present in $BASE_DIR"
  else
    say "fetching base parts, vision sidecar and the release manifests into $BASE_DIR"
    say "an interrupted run resumes from where it stopped; run the same command again"
    hf_get "$BASE_DIR" --exclude "$ENGRAM_SUBDIR/*" \
      || die "download failed; run the same command again to resume"
    [ -s "$BASE_DIR/SHA256SUMS" ] || die "SHA256SUMS is missing from the download"
    say "manifests and sidecar in place; $(ls "$BASE_DIR"/$GGUF.part*-of-$NPART 2>/dev/null | wc -l)/$NPART parts on disk"
  fi

  if engram_ok; then
    say "Engram tables already present in $ENGRAM_DIR"
  else
    say "fetching the 1017 Engram tables into $ENGRAM_DIR (203.1 GB)"
    say "  these are read in rows at run time and never modified"
    hf_get "$ENGRAM_DIR" "$ENGRAM_SUBDIR/${SHARDS[0]}" "$ENGRAM_SUBDIR/${SHARDS[1]}" \
      || die "Engram download failed; run the same command again to resume"
    # hf keeps the repository path, so flatten the two files into ENGRAM_DIR.
    local s
    for s in "${SHARDS[@]}"; do
      [ -f "$ENGRAM_DIR/$ENGRAM_SUBDIR/$s" ] || continue
      mv -f "$ENGRAM_DIR/$ENGRAM_SUBDIR/$s" "$ENGRAM_DIR/$s"
    done
    rmdir "$ENGRAM_DIR/$ENGRAM_SUBDIR" 2>/dev/null || true
  fi

  say "next: bash $0 assemble"
}

# The card's loop, with a progress file so an interrupted assembly can resume
# instead of re-appending: truncate to the recorded length first, then continue.
cmd_assemble() {
  local out="$BASE_DIR/$GGUF.assembling" k=0 i p want got
  if base_ok; then
    say "base already assembled: $(human $GGUF_BYTES)"
    rm -f "$BASE_DIR"/$GGUF.part*-of-$NPART "$PROGRESS" "$out"
    return 0
  fi
  mkdir -p "$BASE_DIR"
  [ -s "$PROGRESS" ] && k="$(cat "$PROGRESS")"
  case "$k" in ''|*[!0-9]*) die "$PROGRESS does not hold a part count";; esac
  if [ "$k" -gt 0 ] && [ "$(fsize "$out")" -lt "$(parts_len "$k")" ]; then
    die "$out is shorter than the $k parts recorded in $PROGRESS; delete both and run download then assemble again"
  fi
  truncate -s "$(parts_len "$k")" "$out"

  say "appending $NPART parts into $GGUF (one extra part of disk at a time)"
  for ((i = k + 1; i <= NPART; i++)); do
    p="$(part_name "$i")"
    [ "$(fsize "$BASE_DIR/$p")" = "$(part_bytes "$i")" ] \
      || die "$p is missing or short; run download first"
    want="$(sum_of "$p" "$BASE_DIR/SHA256SUMS" || true)"
    if [ -n "$want" ]; then
      got="$(sha_of "$BASE_DIR/$p")"
      [ "$got" = "$want" ] || { rm -f "$BASE_DIR/$p"; die "$p failed its SHA-256 check and was deleted; run download again"; }
    else
      warn "$p is not listed in SHA256SUMS; appending it unchecked"
    fi
    cat "$BASE_DIR/$p" >>"$out" || die "cannot append $p (disk full?)"
    echo "$i" >"$PROGRESS"
    rm -f "$BASE_DIR/$p"
    [ $((i % 10)) -eq 0 ] && say "  $i/$NPART appended"
  done
  [ "$(fsize "$out")" = "$GGUF_BYTES" ] || die "assembled file is $(fsize "$out") bytes, expected $GGUF_BYTES"
  local got; got="$(sha_of "$out")"
  [ "$got" = "$GGUF_SHA256" ] || die "assembled base hashes to $got, expected $GGUF_SHA256"
  mv "$out" "$BASE_DIR/$GGUF" && rm -f "$PROGRESS"
  say "base assembled and verified: $(human $GGUF_BYTES)"
  say "next: bash $0 verify && bash $0 build && bash $0 start"
}

cmd_verify() {
  local bad=0 s got want
  [ -s "$BASE_DIR/SHA256SUMS" ] || die "SHA256SUMS is missing; run download first"
  say "verifying base and vision against $BASE_DIR/SHA256SUMS"
  if [ -f "$BASE_DIR/$GGUF" ]; then
    got="$(sha_of "$BASE_DIR/$GGUF")"
    if [ "$got" = "$GGUF_SHA256" ]; then say "  base ok ($(human "$(fsize "$BASE_DIR/$GGUF")"))"; else warn "  base MISMATCH"; bad=1; fi
  else
    say "  base not assembled yet"
  fi
  if [ -f "$BASE_DIR/$VISION" ]; then
    got="$(sha_of "$BASE_DIR/$VISION")"
    if [ "$got" = "$VISION_SHA256" ]; then say "  vision ok ($(human $VISION_BYTES))"; else warn "  vision MISMATCH (a V4.1 sidecar under this name would look like this)"; bad=1; fi
  else
    say "  vision sidecar not present"
  fi
  # The Engram digests are the ones worth checking hardest: the V4.1 files have
  # identical names and sizes, so only the hash separates them.
  s=0
  while [ "$s" -lt 2 ]; do
    if [ "$(fsize "$ENGRAM_DIR/${SHARDS[$s]}")" = "${SHARD_BYTES[$s]}" ]; then
      say "  hashing ${SHARDS[$s]} (101.5 GB; this takes a few minutes)"
      got="$(sha_of "$ENGRAM_DIR/${SHARDS[$s]}")"
      if [ "$got" = "${SHARD_SHA256[$s]}" ]; then
        say "  ${SHARDS[$s]} ok (1017 tables)"
      else
        warn "  ${SHARDS[$s]} is NOT the 1017 table (got $got)"; bad=1
      fi
    else
      say "  ${SHARDS[$s]} missing or short"
      bad=1
    fi
    s=$((s + 1))
  done
  [ "$bad" = 0 ] && say "verify: everything checked out" || die "verify: something above is wrong"
}

cmd_build() {
  local f help
  for f in "$SERVER_BIN" "$CLI_BIN" "$BENCH_BIN"; do
    [ -f "$f" ] && chmod +x "$f"
  done
  [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN; build it with: cd ~/ds4-dfm-rs && make cuda-spark"
  # The flags this launcher passes must exist in the binary that will run.
  help="$("$SERVER_BIN" --help 2>&1 || true)"
  [ -n "$help" ] || die "$SERVER_BIN printed nothing for --help; is it the aarch64 build for this machine?"
  local ok=1
  for f in --engram-dir --vision-gguf --mem-budget-mb --mtp-mode; do
    case "$help" in *"$f"*) ;; *) warn "$SERVER_BIN does not advertise $f"; ok=0;; esac
  done
  [ "$ok" = 1 ] || die "$SERVER_BIN is not the binary this launcher expects"
  say "runtime ok: $SERVER_BIN advertises --engram-dir, --vision-gguf, --mem-budget-mb and --mtp-mode"
  say "nothing was compiled by this script; the engine is ds4-dfm-rs (make cuda-spark)"
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
  base_ok || die "the base is not assembled in $BASE_DIR; run: bash $0 download && bash $0 assemble"
  engram_ok || die "the Engram tables are not in $ENGRAM_DIR (203 GB); run: bash $0 download"
  if [ "$NO_VISION" != 1 ]; then
    vision_ok || die "the vision sidecar is not present; run download, or set NO_VISION=1 to serve text only"
  fi
  [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN; run: bash $0 build"
  command -v curl >/dev/null 2>&1 || die "curl is required"

  local owners
  owners="$(pgrep -af 'ds4-server' | grep -v "$$" || true)"
  if [ -n "$owners" ] && [ "${ALLOW_SHARED:-0}" != "1" ]; then
    printf '[ds41-1017] another model owner is running:\n%s\n' "$owners" >&2
    die "stop it first, or set ALLOW_SHARED=1 to overrule (one ds4 model fits on this box)"
  fi

  local a; a="$(avail_mb)"
  [ "$a" -ge "$START_FLOOR_MB" ] || die "only $a MB of memory available, need $START_FLOOR_MB: stop another model first"

  mkdir -p "$LOGS"
  local args=(--cuda -m "$BASE_DIR/$GGUF" --engram-dir "$ENGRAM_DIR"
              --mtp-mode "$MTP_MODE" --mem-budget-mb "$BUDGET_MB"
              --host "$HOST_ADDR" --port "$PORT")
  [ "$NO_VISION" = 1 ] || args+=(--vision-gguf "$BASE_DIR/$VISION")
  [ -n "$CTX" ] && args+=(-c "$CTX")

  say "starting ds4-server on $HOST_ADDR:$PORT (mapping 113.6 GB; the card reports about two minutes)"
  say "  log: $SERVER_LOG"
  nohup "$SERVER_BIN" "${args[@]}" >"$SERVER_LOG" 2>&1 </dev/null &
  echo $! >"$PIDF"
  echo "ds41-1017 on $HOST_ADDR:$PORT mtp=$MTP_MODE vision=$([ "$NO_VISION" = 1 ] && echo off || echo on)" >"$INFOF"

  if [ "$WATCHDOG" = 1 ]; then
    export -f watchdog avail_mb
    nohup bash -c "watchdog $(cat "$PIDF") $KILL_MB '$WATCHDOG_LOG'" >/dev/null 2>&1 </dev/null &
    echo $! >"$WDPIDF"
    say "  watchdog pid $(cat "$WDPIDF"), stops the server if MemAvailable stays below ${KILL_MB} MB"
  fi

  local waited=0
  say "waiting for the endpoint (up to ${READY_TIMEOUT}s)"
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
  cmd_test || warn "the smoke request failed; the server is up, so check $SERVER_LOG"
  say ""
  say "ready: http://$HOST_ADDR:$PORT/v1   (OpenAI: /v1/chat/completions, /v1/responses; Anthropic: /v1/messages)"
  say "stop with: bash $0 stop   drafter: $MTP_MODE   Engram: $ENGRAM_DIR"
  say "this family defaults to non-thinking mode; send temperature and top_p explicitly"
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
  say "memory:        $(avail_mb) MB available of $(total_mb) MB"
  say "base:          $([ "$(base_ok && echo 1)" = 1 ] && echo "assembled ($(human $GGUF_BYTES))" || echo "NOT assembled ($(ls "$BASE_DIR"/$GGUF.part*-of-$NPART 2>/dev/null | wc -l)/$NPART parts)")"
  say "vision:        $([ "$(vision_ok && echo 1)" = 1 ] && echo present || echo missing)   engram: $([ "$(engram_ok && echo 1)" = 1 ] && echo present || echo missing) in $ENGRAM_DIR"
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | head -c 400; then
    say ""
    say "endpoint:      answering on port $PORT"
  else
    say "endpoint:      no answer on port $PORT"
  fi
  [ -f "$SERVER_LOG" ] && { say ""; say "server log tail:"; tail -n 5 "$SERVER_LOG" | sed 's/^/  /'; }
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
  # Temperature and top_p are sent explicitly: this path follows the request, and
  # the artifact's own recommendation is 1.0 / 1.0.
  local out
  out="$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4.1-flash-1017","temperature":1.0,"top_p":1.0,"max_tokens":64,"messages":[{"role":"user","content":"In one sentence, what does a vector-quantized expert layer store?"}]}')"
  case "$out" in
    *'"choices"'*) printf '%s\n' "$out" | sed -n 's/.*"timings":{\(.*\)}.*/  timings: \1/p' | head -1
                   printf '%s\n' "$out" | sed -n 's/.*"content":"\([^"]\{0,90\}\)".*/  answer: \1/p' | head -1;;
    *) warn "the request failed: $(printf '%s' "$out" | head -c 300)"; return 1;;
  esac
}

# --------------------------------------------------------------------------
usage() { sed -n '3,72p' "$0" | sed 's/^# \{0,1\}//'; }

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
  all)       shift; preflight; cmd_download; cmd_assemble; cmd_verify; cmd_build; cmd_start "$@";;
  ""|-h|--help|help) usage;;
  *) echo "unknown command: $1 (see help)" >&2; usage >&2; exit 2;;
esac
