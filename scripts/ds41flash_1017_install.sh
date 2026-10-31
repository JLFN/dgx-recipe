#!/usr/bin/env bash
#
# ds41flash_1017_install.sh - deploy and serve YoungAi's DeepSeek-V4.1-Flash-1017
# VQ artifact on the DGX Spark with the ds4-dfm-rs engine.
#
# WHAT THIS ARTIFACT IS
#   DeepSeek's October 17 V4.1 Flash checkpoint, converted with YoungAi's
#   vq8sh14-q4k-mtpnative recipe: one 113.56 GB base GGUF assembled from 40
#   byte-range parts (the three-tower DSpark drafter is inside the base), one
#   971 MB vision sidecar, and five domain sidecars of about 42 MB each. The
#   1017 Engram tables are NOT in this repository; they are the two official
#   DeepSeek 1017 shards, fetched here from deepseek-ai/DeepSeek-V4.1-Flash-1017
#   and read in place from disk.
#
# WHERE EACH PIECE GOES, AND WHY
#   The base is read exactly once at load and then stays resident in unified
#   memory, so disk speed stops mattering after the first minutes: it belongs on
#   the roomy volume. Default /mnt/models/ds41flash-1017.
#   Everything read while the server runs stays on the fast device:
#     - the five domain sidecars, applied per layer on every token
#     - the vision tower, read from disk when an image arrives
#     - the Engram tables, read in rows on the tokens that hit an n-gram
#   Default $HOME/models/ds41flash-1017 (the NVMe home on the reference Spark).
#
# WHAT IT DOES
#   preflight  GPU, CUDA 13, memory, disk space on both volumes, ports
#   plan       what would be fetched, assembled and started, with sizes
#   download   base parts + vision + five sidecars, then the two official 1017
#              Engram shards; every fetch resumes
#   assemble   append the 40 parts into the base and delete each part as it lands
#   verify     SHA256SUMS over the base, the vision digest, the two Engram
#              digests and a SHA-256 aggregate over every sidecar directory
#   build      check (or build) the engine the launcher will run
#   start      serve on 0.0.0.0:8008 under a memory watchdog, then smoke it
#   stop | status | logs | test
#   install    preflight, download, assemble, verify, build, start
#
# THE MISTAKES THIS SCRIPT EXISTS TO PREVENT
#   1. Mixing the two 1017 packagings. The older Baekpica base has the same
#      113,556,639,424-byte size and a different digest (9f3afe6a... against
#      0da4ae2d... here). It is a different file; do not reuse it or its sibling
#      files.
#   2. Using V4.1's Engram tables. 1017 changed all 48 shards, including the two
#      that hold the tables. The names and sizes are identical, so an engine
#      pointed at the older tables loads them without a word. `verify` hashes
#      both against the official 1017 digests and refuses a mismatch.
#   3. Using the older vision sidecar under its V4.1 name. The published 1017
#      digest is 3a8b194a..., the same bytes the older packaging shipped, but
#      the name carries the date; the script accepts only the digest.
#   4. Pointing --zchain at a V4.1 sidecar. The five directories here carry the
#      1017 base name in their own manifests. The engine does not fingerprint a
#      zchain against the base, so this script checks the directory aggregate.
#
# WHAT HAS NOT BEEN MEASURED
#   Nothing in this recipe has been loaded or requested on the reference Spark at
#   the time it was committed. The speed and fidelity figures in the runbook are
#   V4.1 Flash's, for the same recipe and runtime, and are labelled as such.
#
# USAGE
#   bash ds41flash_1017_install.sh preflight
#   bash ds41flash_1017_install.sh download     # 317.8 GB, resumable
#   bash ds41flash_1017_install.sh assemble     # 40 parts -> the base
#   bash ds41flash_1017_install.sh verify
#   bash ds41flash_1017_install.sh build
#   bash ds41flash_1017_install.sh start        # 0.0.0.0:8008
#   bash ds41flash_1017_install.sh install      # all of the above, then start
#   bash ds41flash_1017_install.sh stop | status | logs | test | plan
#
# OPTIONS (environment variables use uppercase names; see Configuration below)
#   --dir DIR          fast-device root: the five sidecars, the vision file, the
#                      logs and the pid files (default $HOME/models/ds41flash-1017)
#   --gguf-dir DIR     where the base GGUF alone lives (default
#                      /mnt/models/ds41flash-1017). It is read once at load and
#                      then stays resident, so it belongs on the roomy volume.
#   --engram-dir DIR   the two 101.5 GB Engram tables (default
#                      <--dir>/deepseek-engram); read in rows, so keep it fast
#   --domain NAME      finance | code | law | medicine | science | none
#                      (default finance). One sidecar is mounted at a time
#                      through --zchain; none serves the bare base.
#   --host ADDR        listen address (default 0.0.0.0)
#   --port N           listen port (default 8008, this artifact's port)
#   --mtp-mode MODE    off | auto | on, the embedded DSpark drafter (default auto)
#   --mem-budget-mb N  the engine's load budget (default 110000)
#   --kv-disk-dir DIR  enable disk KV checkpoints (default off; present, not qualified)
#   --kv-disk-space-mb N  disk KV budget in MiB (engine default 4096)
#   --kv-cache-min-tokens N  minimum checkpoint length (engine default 512)
#   --prefix-reuse MODE  auto | exact | off (partial is unsupported)
#   --dspark-verify N  verification width, 0 for automatic
#   --tokens N        default output token limit
#   --mem-floor-gb N  engine admission memory floor in GiB
#   --model-id ID     advertised model identifier
#   --tool-memory-max-ids N  maximum exact-DSML replay entries
#   --disable-exact-dsml-tool-replay  disable exact DSML replay
#   --no-kv-reuse     disable live and disk prefix reuse
#   --no-graph        disable the V4.1 decode graph (diagnostic)
#   --emit-trace      log token IDs (diagnostic; may expose prompt content)
#   --v41-prof        enable V4.1 profiling (diagnostic)
#   --trace FILE      write session trace (contains prompts and outputs)
#   --cors            enable CORS
#   --min-p F          exported as DS4_SERVER_DEFAULT_MIN_P, for parity with the
#                      V4.1 installer. On the deployed engine this variable feeds
#                      the GLM family (parse.rs:132, 380-384); the V4.1 route
#                      takes its request-omitted min_p from the engine defaults,
#                      which are 0.0 (parse.rs:582-587). It is recorded and
#                      passed, but it does not change this model's sampling.
#   --engine-dir DIR   the ds4-dfm-rs tree (default $HOME/ds4-dfm-rs)
#   --server-bin PATH  the binary to run (default <--engine-dir>/ds4-server)
#   --hf-bin PATH      the Hugging Face CLI to use
#   --revision SHA     pin another revision of the model repository
#   --watchdog 0|1     memory watchdog (default 1)
#   --no-vision        serve text only; do not mount the vision tower
#   --no-build         never run make, even when the binary is missing
#   --no-start         everything except starting the server
#   --force-build      run make cuda-spark even when a binary already exists
#   --allow-shared     overrule the single-owner probe before starting
#   --posttrain        recognised and refused: this packaging ships no
#                      post-training directory (the V4.1 one is finance-only)
#   -h, --help         this text
#
#   Examples
#   DOMAIN=code bash ds41flash_1017_install.sh start
#   bash ds41flash_1017_install.sh --domain medicine --port 8008 start
#   bash ds41flash_1017_install.sh --gguf-dir /mnt/big --dir $HOME/models/ds41flash-1017 plan
#   HF_TOKEN=hf_... bash ds41flash_1017_install.sh download
#   ALLOW_SHARED=1 bash ds41flash_1017_install.sh start  overrule the owner probe
#
# Notes
#   - The repositories are public: no token is required, and one is never written
#     into this file. A token only raises the CDN rate limit.
#   - start checks file sizes; verify checks digests. Run install (which includes
#     verify) or verify itself at least once after a fetch.
#   - The engine takes a single global lock: only one ds4 model can be resident.
#   - The context is not a flag on this family. The GGUF metadata carries
#     1,048,576 and the engine takes it from there; -c/--ctx is not passed.
#   - Thinking is off by default for this weight recipe (under-weighted stop
#     decisions). The launcher sets no temperature: the smoke test sends 0, as
#     the older V4.1 installer did, and real requests should carry DeepSeek's
#     recommendation of temperature 1.0 with top_p 1.0, sent explicitly.
#   - --min-p exists for parity with the older V4.1 installer. The deployed
#     engine resolves the V4.1 route's request-omitted min_p to 0.0 from the
#     engine defaults, so the value is recorded but does not change this model.

set -Eeuo pipefail
export LC_ALL=C

# --------------------------------------------------------------------------
# Configuration (every value is overridable from the environment)
# --------------------------------------------------------------------------
REPO_ID="${REPO_ID:-wenzhouwu/YoungAi-DeepSeek-V4.1-Flash-1017}"
REVISION="${REVISION:-993a24e1ea816c6796c2e6c79bc905383baf1829}"

OFFICIAL_REPO_ID="${OFFICIAL_REPO_ID:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"
OFFICIAL_REVISION="${OFFICIAL_REVISION:-1bae0e1088993b9b82de08f59de5be5a34e42170}"

# The base is read once at load and then resident: roomy volume.
BASE_DIR="${BASE_DIR:-/mnt/models/ds41flash-1017}"
# The fast device: sidecars, vision tower, Engram tables, logs, pid files.
AUX_DIR="${AUX_DIR:-$HOME/models/ds41flash-1017}"
# Filled in derive_paths() once the command line has been read.
ENGRAM_DIR="${ENGRAM_DIR:-}"
MODEL_DIR=""

GGUF="DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf"
GGUF_BYTES=113556639424
GGUF_SHA256=0da4ae2d7528ae6a60153ee072ddb4680b9d36f1c8f0b5915cdd950e89de5070
NPART=40
PART_LONG=2838915986          # bytes in parts 1..24
PART_SHORT=2838915985         # bytes in parts 25..40
PART_LONG_N=24

VISION="DeepSeek-V4.1-Flash-1017.vision.gguf"
VISION_BYTES=971066016
VISION_SHA256=3a8b194a011e630e8e49a51b5f4e9821e48a9e5674e678d52c0302d6b164698c

# The five domain sidecars, from this repository. The directory names carry the
# 1017 base name; their manifests name the 1017 GGUF they were solved against.
SIDECAR_PREFIX="DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb"
SIDECAR_FINANCE="$SIDECAR_PREFIX-vqfin41_vqhalf_a_n8192-engine"
SIDECAR_CODE="$SIDECAR_PREFIX-code_fit_n15360-engine"
SIDECAR_LAW="$SIDECAR_PREFIX-law_fit_n15360-engine"
SIDECAR_MED="$SIDECAR_PREFIX-med_fit_n15360-engine"
SIDECAR_SCI="$SIDECAR_PREFIX-sci_fit_n15360-engine"
# SHA-256 over the sorted "digest  ./name" list of every gr_L*.bin and rb_L*.bin
# in a directory. Computed from the repository's LFS metadata at revision
# 993a24e1; verify recomputes it from disk.
SIDECAR_FINANCE_SHA=aacdecd2bd52369a1550850c0aa86b2e6f9355fee9c7112145e7ae1a9858f976
SIDECAR_CODE_SHA=dba83d0950f8414417c55b4ec2ba2cffeebf83bdc20671ab936065291b227133
SIDECAR_LAW_SHA=7f9b451d3880fd53f720de01b1a73fd46f2958fb597d19cc04faf48abf492001
SIDECAR_MED_SHA=52e3cdb30f433b8ce7778dbbf40a97e426f7f175a9ed465f3a05f005adc5a53c
SIDECAR_SCI_SHA=b4fedb1abe58a48b49e23bf64346e0c82255f2cd73754e8d0282154a99335ab2
SIDECAR_BYTES_TOTAL=209154581   # all five directories together

# The official 1017 Engram tables, copied unmodified from DeepSeek's repository.
# The V4.1 shards carry the same names and sizes and MUST NOT be used.
SHARDS=(model-00047-of-00048.safetensors model-00048-of-00048.safetensors)
SHARD_BYTES=(101535150936 101537926640)
SHARD_SHA256=(76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5 \
              9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd)

ENGINE_DIR="${ENGINE_DIR:-$HOME/ds4-dfm-rs}"
SERVER_BIN="${SERVER_BIN:-$ENGINE_DIR/ds4-server}"

HOST_ADDR="${HOST_ADDR:-0.0.0.0}"
# 8008 is this artifact; 8000 the V4.1 port deployment, 8003 Qwen3.8, 8005
# Bonsai, 8006 YoungAi V4.1, 8007 GLM-5.3 VQ and 8009 GLM-5.3 Uncensored VQ.
PORT="${PORT:-8008}"
DOMAIN="${DOMAIN:-finance}"             # finance | code | law | medicine | science | none
MTP_MODE="${MTP_MODE:-auto}"            # off | auto | on: the embedded DSpark drafter
BUDGET_MB="${BUDGET_MB:-110000}"        # the artifact card's engine budget
KV_DISK_DIR="${KV_DISK_DIR:-}"
KV_DISK_SPACE_MB="${KV_DISK_SPACE_MB:-}"
KV_CACHE_MIN_TOKENS="${KV_CACHE_MIN_TOKENS:-}"
PREFIX_REUSE="${PREFIX_REUSE:-}"
DSPARK_VERIFY="${DSPARK_VERIFY:-}"
TOKENS="${TOKENS:-}"
MEM_FLOOR_GB="${MEM_FLOOR_GB:-}"
MODEL_ID="${MODEL_ID:-}"
TOOL_MEMORY_MAX_IDS="${TOOL_MEMORY_MAX_IDS:-}"
DISABLE_EXACT_DSML_TOOL_REPLAY="${DISABLE_EXACT_DSML_TOOL_REPLAY:-0}"
NO_KV_REUSE="${NO_KV_REUSE:-0}"
NO_GRAPH="${NO_GRAPH:-0}"
EMIT_TRACE="${EMIT_TRACE:-0}"
V41_PROF="${V41_PROF:-0}"
TRACE="${TRACE:-}"
CORS="${CORS:-0}"
NO_VISION="${NO_VISION:-0}"
ALLOW_SHARED="${ALLOW_SHARED:-0}"
FORCE_BUILD="${FORCE_BUILD:-0}"
NO_BUILD="${NO_BUILD:-0}"
NO_START="${NO_START:-0}"
MIN_P="${MIN_P:-}"

# Derived paths; resolve_paths() recomputes them after the command line is read.
LOGS=""; PIDF=""; WDPIDF=""; INFOF=""; SERVER_LOG=""; WATCHDOG_LOG=""; PROGRESS=""
resolve_paths() {
  MODEL_DIR="$AUX_DIR/model"
  [ -n "$ENGRAM_DIR" ] || ENGRAM_DIR="$AUX_DIR/deepseek-engram"
  LOGS="$AUX_DIR/logs"
  PIDF="$AUX_DIR/ds41flash-1017.pid"
  WDPIDF="$AUX_DIR/ds41flash-1017-watchdog.pid"
  INFOF="$AUX_DIR/ds41flash-1017.info"
  SERVER_LOG="$LOGS/ds41flash-1017-server.log"
  WATCHDOG_LOG="$LOGS/ds41flash-1017-watchdog.log"
  PROGRESS="$BASE_DIR/.assemble-progress"
}

START_FLOOR_MB="${START_FLOOR_MB:-100000}"
RUN_FLOOR_MB="${RUN_FLOOR_MB:-10000}"
KILL_MB="${KILL_MB:-2500}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"   # mapping 113.6 GB; the card reports ~2 min
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

part_name()  { printf '%s.part%02d-of-%d' "$GGUF" "$1" "$NPART"; }
part_bytes() { if [ "$1" -le "$PART_LONG_N" ]; then echo "$PART_LONG"; else echo "$PART_SHORT"; fi; }
parts_len()  { local k="$1" n; if [ "$k" -le "$PART_LONG_N" ]; then echo $(( k * PART_LONG )); else n=$(( k - PART_LONG_N )); echo $(( PART_LONG_N * PART_LONG + n * PART_SHORT )); fi; }

sidecar_of() {
  case "$1" in
    finance)  echo "$SIDECAR_FINANCE";;
    code)     echo "$SIDECAR_CODE";;
    law)      echo "$SIDECAR_LAW";;
    medicine) echo "$SIDECAR_MED";;
    science)  echo "$SIDECAR_SCI";;
    none)     echo "";;
    *)        return 1;;
  esac
}
all_sidecars() { printf '%s\n' "$SIDECAR_FINANCE" "$SIDECAR_CODE" "$SIDECAR_LAW" "$SIDECAR_MED" "$SIDECAR_SCI"; }

# The aggregate is over the sorted "hash  ./name" lines of the gr/rb files only,
# so it matches the values derived from the repository's LFS metadata.
dir_sha() {
  local d="$1"
  ( cd "$d" && find . -maxdepth 1 -type f \( -name 'gr_L*.bin' -o -name 'rb_L*.bin' \) -print0 \
      | sort -z | xargs -0 -r sha256sum ) | sha256sum | cut -d' ' -f1
}

# SHA256SUMS is the artifact's own manifest and wins where it lists a file.
sum_of() {
  local f="$1" manifest="$2"
  [ -s "$manifest" ] || return 1
  awk -v f="$f" '$2 == f {print $1; found=1} END{exit !found}' "$manifest"
}

ensure_hf() {
  if [ -n "$HF_BIN" ]; then HF="$HF_BIN"; return; fi
  if command -v hf >/dev/null 2>&1 && hf download --help >/dev/null 2>&1; then HF=hf; return; fi
  if command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli; return; fi
  HF="$AUX_DIR/.hfenv/bin/hf"
  [ -x "$HF" ] && return
  say "installing the Hugging Face CLI into $AUX_DIR/.hfenv (the system Python is left alone)"
  mkdir -p "$AUX_DIR"
  python3 -m venv "$AUX_DIR/.hfenv" || die "python3 venv failed; install it with: sudo apt install python3-venv"
  "$AUX_DIR/.hfenv/bin/pip" install -q -U "huggingface_hub[hf_xet]" \
    || die "pip install huggingface_hub failed (behind a firewall? try ENDPOINT= and NO_XET=1)"
  HF="$AUX_DIR/.hfenv/bin/hf"
}

hf_env() {
  [ -n "$ENDPOINT" ] && export HF_ENDPOINT="$ENDPOINT"
  [ "$NO_XET" = 1 ] && export HF_HUB_DISABLE_XET=1
  export HF_HUB_MAX_WORKERS="$HF_MAX_WORKERS"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
  return 0
}

hf_fetch() {   # $1 repo, $2 revision, $3 local dir, rest paths/patterns
  local repo="$1" rev="$2" dir="$3"; shift 3
  "$HF" download "$repo" --revision "$rev" --local-dir "$dir" "$@"
}

base_ok()   { [ "$(fsize "$BASE_DIR/$GGUF")" = "$GGUF_BYTES" ]; }
vision_ok() { [ "$(fsize "$MODEL_DIR/$VISION")" = "$VISION_BYTES" ]; }
engram_ok() {
  [ "$(fsize "$ENGRAM_DIR/${SHARDS[0]}")" = "${SHARD_BYTES[0]}" ] &&
  [ "$(fsize "$ENGRAM_DIR/${SHARDS[1]}")" = "${SHARD_BYTES[1]}" ]
}
sidecars_ok() {
  local d
  for d in $(all_sidecars); do
    [ -s "$MODEL_DIR/$d/manifest.txt" ] || return 1
  done
}

# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
CMD=""
parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      preflight|plan|download|assemble|verify|build|start|stop|status|logs|test|install|all)
        CMD="$1"; shift;;
      --dir|--aux-dir)     [ $# -ge 2 ] || die "$1 needs a directory"; AUX_DIR="$2"; shift 2;;
      --gguf-dir|--base-dir) [ $# -ge 2 ] || die "$1 needs a directory"; BASE_DIR="$2"; shift 2;;
      --engram-dir)        [ $# -ge 2 ] || die "$1 needs a directory"; ENGRAM_DIR="$2"; shift 2;;
      --domain)            [ $# -ge 2 ] || die "$1 needs finance | code | law | medicine | science | none"; DOMAIN="$2"; shift 2;;
      --host)              [ $# -ge 2 ] || die "$1 needs an address"; HOST_ADDR="$2"; shift 2;;
      --port)              [ $# -ge 2 ] || die "$1 needs a port number"; PORT="$2"; shift 2;;
      --mtp-mode)          [ $# -ge 2 ] || die "$1 needs off | auto | on"; MTP_MODE="$2"; shift 2;;
      --mem-budget-mb)     [ $# -ge 2 ] || die "$1 needs a number"; BUDGET_MB="$2"; shift 2;;
      --kv-disk-dir)       [ $# -ge 2 ] || die "$1 needs a directory"; KV_DISK_DIR="$2"; shift 2;;
      --kv-disk-space-mb)  [ $# -ge 2 ] || die "$1 needs a number"; KV_DISK_SPACE_MB="$2"; shift 2;;
      --kv-cache-min-tokens) [ $# -ge 2 ] || die "$1 needs a number"; KV_CACHE_MIN_TOKENS="$2"; shift 2;;
      --prefix-reuse)      [ $# -ge 2 ] || die "$1 needs auto | exact | off"; PREFIX_REUSE="$2"; shift 2;;
      --dspark-verify)     [ $# -ge 2 ] || die "$1 needs a number"; DSPARK_VERIFY="$2"; shift 2;;
      --tokens)            [ $# -ge 2 ] || die "$1 needs a number"; TOKENS="$2"; shift 2;;
      --mem-floor-gb)      [ $# -ge 2 ] || die "$1 needs a number"; MEM_FLOOR_GB="$2"; shift 2;;
      --model-id)          [ $# -ge 2 ] || die "$1 needs an identifier"; MODEL_ID="$2"; shift 2;;
      --tool-memory-max-ids) [ $# -ge 2 ] || die "$1 needs a number"; TOOL_MEMORY_MAX_IDS="$2"; shift 2;;
      --disable-exact-dsml-tool-replay) DISABLE_EXACT_DSML_TOOL_REPLAY=1; shift;;
      --no-kv-reuse)       NO_KV_REUSE=1; shift;;
      --no-graph)          NO_GRAPH=1; shift;;
      --emit-trace)        EMIT_TRACE=1; shift;;
      --v41-prof)          V41_PROF=1; shift;;
      --trace)             [ $# -ge 2 ] || die "$1 needs a path"; TRACE="$2"; shift 2;;
      --cors)              CORS=1; shift;;
      --min-p)             [ $# -ge 2 ] || die "$1 needs a value"; MIN_P="$2"; shift 2;;
      --engine-dir)        [ $# -ge 2 ] || die "$1 needs a directory"; ENGINE_DIR="$2"; SERVER_BIN="$ENGINE_DIR/ds4-server"; shift 2;;
      --server-bin)        [ $# -ge 2 ] || die "$1 needs a path"; SERVER_BIN="$2"; shift 2;;
      --hf-bin)            [ $# -ge 2 ] || die "$1 needs a path"; HF_BIN="$2"; shift 2;;
      --revision)          [ $# -ge 2 ] || die "$1 needs a commit"; REVISION="$2"; shift 2;;
      --watchdog)          [ $# -ge 2 ] || die "$1 needs 0 or 1"; WATCHDOG="$2"; shift 2;;
      --no-vision)         NO_VISION=1; shift;;
      --no-build)          NO_BUILD=1; shift;;
      --no-start)          NO_START=1; shift;;
      --force-build)       FORCE_BUILD=1; shift;;
      --allow-shared)      ALLOW_SHARED=1; shift;;
      --posttrain)         die "this packaging ships no post-training directory: YoungAi's posttrain file is bound to the V4.1 finance sidecar and is absent from the 1017 repository";;
      -h|--help|help)      usage; exit 0;;
      -*)                  die "unknown option: $1 (see --help)";;
      *)                   die "unknown command: $1 (see --help)";;
    esac
  done
  sidecar_of "$DOMAIN" >/dev/null || die "unknown --domain '$DOMAIN' (choose finance | code | law | medicine | science | none)"
  case "$PREFIX_REUSE" in ''|auto|exact|off) ;; *) die "--prefix-reuse must be auto, exact or off";; esac
  if [ -n "$KV_DISK_SPACE_MB$KV_CACHE_MIN_TOKENS" ] && [ -z "$KV_DISK_DIR" ]; then
    die "disk KV tuning requires --kv-disk-dir"
  fi
  if [ -n "$KV_DISK_DIR" ] && { [ "$PREFIX_REUSE" = off ] || [ "$NO_KV_REUSE" = 1 ]; }; then
    die "disk KV requires prefix reuse; remove --no-kv-reuse or --prefix-reuse off"
  fi
  if [ "$NO_KV_REUSE" = 1 ] && [ -n "$PREFIX_REUSE" ] && [ "$PREFIX_REUSE" != auto ] && [ "$PREFIX_REUSE" != off ]; then
    die "--no-kv-reuse conflicts with --prefix-reuse $PREFIX_REUSE"
  fi
  resolve_paths
}

server_args() {
  local sc; sc="$(sidecar_of "$DOMAIN")"
  SERVER_ARGS=(--cuda -m "$BASE_DIR/$GGUF" --engram-dir "$ENGRAM_DIR"
               --mtp-mode "$MTP_MODE" --mem-budget-mb "$BUDGET_MB"
               --host "$HOST_ADDR" --port "$PORT")
  [ -z "$sc" ] || SERVER_ARGS+=(--zchain "$MODEL_DIR/$sc")
  [ "$NO_VISION" = 1 ] || SERVER_ARGS+=(--vision-gguf "$MODEL_DIR/$VISION")
  [ -z "$KV_DISK_DIR" ] || SERVER_ARGS+=(--kv-disk-dir "$KV_DISK_DIR")
  [ -z "$KV_DISK_SPACE_MB" ] || SERVER_ARGS+=(--kv-disk-space-mb "$KV_DISK_SPACE_MB")
  [ -z "$KV_CACHE_MIN_TOKENS" ] || SERVER_ARGS+=(--kv-cache-min-tokens "$KV_CACHE_MIN_TOKENS")
  [ -z "$PREFIX_REUSE" ] || SERVER_ARGS+=(--prefix-reuse "$PREFIX_REUSE")
  [ -z "$DSPARK_VERIFY" ] || SERVER_ARGS+=(--dspark-verify "$DSPARK_VERIFY")
  [ -z "$TOKENS" ] || SERVER_ARGS+=(--tokens "$TOKENS")
  [ -z "$MEM_FLOOR_GB" ] || SERVER_ARGS+=(--mem-floor-gb "$MEM_FLOOR_GB")
  [ -z "$MODEL_ID" ] || SERVER_ARGS+=(--model-id "$MODEL_ID")
  [ -z "$TOOL_MEMORY_MAX_IDS" ] || SERVER_ARGS+=(--tool-memory-max-ids "$TOOL_MEMORY_MAX_IDS")
  [ "$DISABLE_EXACT_DSML_TOOL_REPLAY" != 1 ] || SERVER_ARGS+=(--disable-exact-dsml-tool-replay)
  [ "$NO_KV_REUSE" != 1 ] || SERVER_ARGS+=(--no-kv-reuse)
  [ "$NO_GRAPH" != 1 ] || SERVER_ARGS+=(--no-graph)
  [ "$EMIT_TRACE" != 1 ] || SERVER_ARGS+=(--emit-trace)
  [ "$V41_PROF" != 1 ] || SERVER_ARGS+=(--v41-prof)
  [ -z "$TRACE" ] || SERVER_ARGS+=(--trace "$TRACE")
  [ "$CORS" != 1 ] || SERVER_ARGS+=(--cors)
  return 0
}

check_server_flags() {
  local help flag expect_value=0
  help="$("$SERVER_BIN" --help 2>&1 || true)"
  [ -n "$help" ] || die "$SERVER_BIN printed nothing for --help"
  server_args
  for flag in "${SERVER_ARGS[@]}"; do
    if [ "$expect_value" = 1 ]; then
      expect_value=0
      continue
    fi
    case "$flag" in
      --*)
        printf '%s\n' "$help" | grep -Eq -- "(^|[^[:alnum:]_-])${flag}([^[:alnum:]_-]|$)" || die "$SERVER_BIN does not advertise $flag"
        case "$flag" in
          --cuda|--disable-exact-dsml-tool-replay|--no-kv-reuse|--no-graph|--emit-trace|--v41-prof|--cors) ;;
          *) expect_value=1;;
        esac;;
      -m) expect_value=1;;
    esac
  done
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

  mkdir -p "$BASE_DIR" "$MODEL_DIR" "$ENGRAM_DIR" 2>/dev/null || { warn "cannot create the model directories"; fatal=1; }
  if [ -n "$KV_DISK_DIR" ]; then
    if mkdir -p "$KV_DISK_DIR" 2>/dev/null && [ -w "$KV_DISK_DIR" ]; then
      say "kv disk dir: $KV_DISK_DIR on $(df_target "$KV_DISK_DIR"), $(human "$(df_avail "$KV_DISK_DIR")") free; budget ${KV_DISK_SPACE_MB:-4096} MiB (not reserved)"
    else
      warn "cannot create or write disk KV directory $KV_DISK_DIR"; fatal=1
    fi
  fi

  # The reference layout expects /mnt/models to be the roomy volume. If it is not
  # mounted, the base would land on the root filesystem instead, so refuse that
  # rather than fill the system disk silently.
  case "$BASE_DIR" in
    /mnt/*)
      if [ "$(df_target "$BASE_DIR")" = "/" ]; then
        warn "$BASE_DIR resolves to the root filesystem; /mnt/models is not a mounted volume"; fatal=1
      fi;;
  esac

  # The base and everything read at run time may sit on different devices.
  local need_base=0 need_aux=0 need_engram=0 free_base free_aux free_engram
  base_ok || need_base=$(( GGUF_BYTES + PART_LONG ))
  vision_ok || need_aux=$(( need_aux + VISION_BYTES ))
  sidecars_ok || need_aux=$(( need_aux + SIDECAR_BYTES_TOTAL ))
  engram_ok || need_engram=$(( SHARD_BYTES[0] + SHARD_BYTES[1] ))
  free_base="$(df_avail "$BASE_DIR")"; free_aux="$(df_avail "$MODEL_DIR")"; free_engram="$(df_avail "$ENGRAM_DIR")"
  say "base dir:   $BASE_DIR on $(df_target "$BASE_DIR"), $(human "${free_base:-0}") free, $(human $need_base) to fetch"
  say "fast dir:   $MODEL_DIR on $(df_target "$MODEL_DIR"), $(human "${free_aux:-0}") free, $(human $need_aux) to fetch (sidecars + vision)"
  say "engram dir: $ENGRAM_DIR on $(df_target "$ENGRAM_DIR"), $(human "${free_engram:-0}") free, $(human $need_engram) to fetch"
  if [ "$need_base" -gt 0 ] && [ -n "$free_base" ] && [ "$free_base" -lt $((need_base + 5000000000)) ]; then
    warn "not enough room for the base and the assembly slack (one part, $(human "$PART_LONG"))"; fatal=1
  fi
  if [ "$need_aux" -gt 0 ] && [ -n "$free_aux" ] && [ "$free_aux" -lt $((need_aux + 2000000000)) ]; then
    warn "not enough room in $MODEL_DIR for the vision sidecar and the five domain sidecars"; fatal=1
  fi
  if [ "$need_engram" -gt 0 ] && [ -n "$free_engram" ] && [ "$free_engram" -lt $((need_engram + 2000000000)) ]; then
    warn "not enough room for the Engram tables in $ENGRAM_DIR"; fatal=1
  fi

  local busy=() p
  if command -v curl >/dev/null 2>&1; then
    for p in 8000 8003 8005 8006 8007 8009; do
      [ "$p" = "$PORT" ] && continue
      curl -fsS --max-time 2 "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && busy+=("$p")
    done
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      warn "port $PORT already answers /v1/models; stop the old 1017 launcher first, or pick another with PORT="; fatal=1
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
  local d
  mkdir -p "$BASE_DIR" "$MODEL_DIR" "$ENGRAM_DIR"
  say "base dir:      $BASE_DIR   ($(df_target "$BASE_DIR"), $(human "$(df_avail "$BASE_DIR")") free)"
  say "fast dir:      $MODEL_DIR  ($(df_target "$MODEL_DIR"), $(human "$(df_avail "$MODEL_DIR")") free)"
  say "engram dir:    $ENGRAM_DIR ($(df_target "$ENGRAM_DIR"), $(human "$(df_avail "$ENGRAM_DIR")") free)"
  say "runtime:       $SERVER_BIN"
  say "serve:         $HOST_ADDR:$PORT domain=$DOMAIN mtp=$MTP_MODE budget=${BUDGET_MB} MB watchdog below ${KILL_MB} MB"
  say ""
  say "artifacts"
  say "  base         $NPART parts -> $(human $GGUF_BYTES)   $(base_ok && echo assembled || echo "not assembled ($(ls "$BASE_DIR"/$GGUF.part*-of-$NPART 2>/dev/null | wc -l)/$NPART parts on disk)")"
  say "  vision       $VISION ($(human $VISION_BYTES))   $(vision_ok && echo present || echo missing)"
  for d in $(all_sidecars); do
    say "  sidecar      $d   $([ -s "$MODEL_DIR/$d/manifest.txt" ] && echo present || echo missing)"
  done
  say "  engram       ${SHARDS[0]}, ${SHARDS[1]}   $(engram_ok && echo present || echo missing)"
  say ""
  say "the command start would run:"
  server_args
  printf '  %q' "$SERVER_BIN" "${SERVER_ARGS[@]}"
  printf '\n'
}

# --------------------------------------------------------------------------
# download / assemble / verify / build
# --------------------------------------------------------------------------
cmd_download() {
  ensure_hf; hf_env
  mkdir -p "$BASE_DIR" "$MODEL_DIR" "$ENGRAM_DIR"

  if base_ok; then
    say "base already assembled in $BASE_DIR"
  else
    say "fetching the 40 base parts and SHA256SUMS into $BASE_DIR"
    say "an interrupted run resumes from where it stopped; run the same command again"
    hf_fetch "$REPO_ID" "$REVISION" "$BASE_DIR" --exclude "*-engine/*" --exclude "$VISION" \
      || die "download failed; run the same command again to resume"
    [ -s "$BASE_DIR/SHA256SUMS" ] || die "SHA256SUMS is missing from the download"
    say "base payload in place; $(ls "$BASE_DIR"/$GGUF.part*-of-$NPART 2>/dev/null | wc -l)/$NPART parts on disk"
  fi
  # An assembled base does not imply the manifest survived: verify needs it.
  if [ ! -s "$BASE_DIR/SHA256SUMS" ]; then
    say "fetching SHA256SUMS into $BASE_DIR"
    hf_fetch "$REPO_ID" "$REVISION" "$BASE_DIR" "SHA256SUMS" \
      || die "SHA256SUMS download failed; run the same command again to resume"
  fi

  if vision_ok; then
    say "vision sidecar already present in $MODEL_DIR"
  else
    say "fetching the vision sidecar into $MODEL_DIR (read from disk when an image arrives)"
    hf_fetch "$REPO_ID" "$REVISION" "$MODEL_DIR" "$VISION" \
      || die "vision download failed; run the same command again to resume"
  fi

  if sidecars_ok; then
    say "the five domain sidecars are already in $MODEL_DIR"
  else
    say "fetching the five domain sidecars into $MODEL_DIR (about $(human $SIDECAR_BYTES_TOTAL))"
    hf_fetch "$REPO_ID" "$REVISION" "$MODEL_DIR" --include "*-engine/*" \
      || die "sidecar download failed; run the same command again to resume"
  fi

  if engram_ok; then
    say "Engram tables already present in $ENGRAM_DIR"
  else
    say "fetching the official 1017 Engram tables into $ENGRAM_DIR (203.1 GB)"
    say "  repository: $OFFICIAL_REPO_ID at $OFFICIAL_REVISION"
    hf_fetch "$OFFICIAL_REPO_ID" "$OFFICIAL_REVISION" "$ENGRAM_DIR" \
      "${SHARDS[0]}" "${SHARDS[1]}" \
      || die "Engram download failed; run the same command again to resume"
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
  got="$(sha_of "$out")"
  [ "$got" = "$GGUF_SHA256" ] || die "assembled base hashes to $got, expected $GGUF_SHA256"
  mv "$out" "$BASE_DIR/$GGUF" && rm -f "$PROGRESS"
  say "base assembled and verified: $(human $GGUF_BYTES)"
  say "next: bash $0 verify && bash $0 build && bash $0 start"
}

cmd_verify() {
  local bad=0 s got d expect
  [ -s "$BASE_DIR/SHA256SUMS" ] || die "SHA256SUMS is missing; run download first"
  say "verifying the base against its own SHA256SUMS"
  if [ -f "$BASE_DIR/$GGUF" ]; then
    got="$(sha_of "$BASE_DIR/$GGUF")"
    if [ "$got" = "$GGUF_SHA256" ]; then say "  base ok ($(human "$(fsize "$BASE_DIR/$GGUF")"))"; else warn "  base MISMATCH"; bad=1; fi
  else
    say "  base not assembled yet"
  fi
  # SHA256SUMS does not list the vision file or the sidecars; their digests are
  # carried by this script and were read from the repository's LFS metadata.
  if [ -f "$MODEL_DIR/$VISION" ]; then
    got="$(sha_of "$MODEL_DIR/$VISION")"
    if [ "$got" = "$VISION_SHA256" ]; then say "  vision ok ($(human $VISION_BYTES))"; else warn "  vision MISMATCH (a V4.1 sidecar under this name would look like this)"; bad=1; fi
  else
    say "  vision sidecar not present"
  fi
  for d in $(all_sidecars); do
    if [ -d "$MODEL_DIR/$d" ]; then
      case "$d" in
        "$SIDECAR_FINANCE") expect="$SIDECAR_FINANCE_SHA";;
        "$SIDECAR_CODE")    expect="$SIDECAR_CODE_SHA";;
        "$SIDECAR_LAW")     expect="$SIDECAR_LAW_SHA";;
        "$SIDECAR_MED")     expect="$SIDECAR_MED_SHA";;
        "$SIDECAR_SCI")     expect="$SIDECAR_SCI_SHA";;
      esac
      got="$(dir_sha "$MODEL_DIR/$d")"
      if [ "$got" = "$expect" ]; then
        say "  sidecar $d ok ($(ls "$MODEL_DIR/$d"/gr_L*.bin 2>/dev/null | wc -l) gr, $(ls "$MODEL_DIR/$d"/rb_L*.bin 2>/dev/null | wc -l) rb)"
      else
        warn "  sidecar $d MISMATCH (got $got, want $expect)"; bad=1
      fi
    else
      say "  sidecar $d missing"
      bad=1
    fi
  done
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
  if [ "$NO_BUILD" = 1 ]; then
    [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN and --no-build was given"
  elif [ ! -x "$SERVER_BIN" ] || [ "$FORCE_BUILD" = 1 ]; then
    [ -d "$ENGINE_DIR" ] || die "no engine tree at $ENGINE_DIR; clone Baekpica/ds4-dfm-rs there or set ENGINE_DIR="
    say "building the engine in $ENGINE_DIR: make cuda-spark (about 12 minutes)"
    ( cd "$ENGINE_DIR" && make cuda-spark ) || die "make cuda-spark failed"
  fi
  [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN"
  check_server_flags
  say "runtime ok: $SERVER_BIN advertises the selected flags"
  say "the engine does not need a rebuild for this artifact when it already serves a 1017 model; the artifact is data"
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
  local sc; sc="$(sidecar_of "$DOMAIN")" || die "unknown DOMAIN '$DOMAIN' (choose finance | code | law | medicine | science | none)"
  if alive; then
    say "already running (pid $(cat "$PIDF"), $(cat "$INFOF" 2>/dev/null)); stop it first"
    return 0
  fi
  base_ok || die "the base is not assembled in $BASE_DIR; run: bash $0 download && bash $0 assemble"
  engram_ok || die "the Engram tables are not in $ENGRAM_DIR (203 GB); run: bash $0 download"
  if [ -n "$sc" ]; then
    [ -s "$MODEL_DIR/$sc/manifest.txt" ] || die "the $DOMAIN sidecar is not in $MODEL_DIR; run: bash $0 download"
  fi
  if [ "$NO_VISION" != 1 ]; then
    vision_ok || die "the vision sidecar is not present; run download, or set NO_VISION=1 to serve text only"
  fi
  [ -x "$SERVER_BIN" ] || die "no executable at $SERVER_BIN; run: bash $0 build"
  check_server_flags
  command -v curl >/dev/null 2>&1 || die "curl is required"

  local owners
  owners="$(pgrep -af 'ds4-server' | grep -v "$$" || true)"
  if [ -n "$owners" ] && [ "$ALLOW_SHARED" != "1" ]; then
    printf '[ds41-1017] another model owner is running:\n%s\n' "$owners" >&2
    die "stop it first (the engine takes a single global lock), or set ALLOW_SHARED=1 to overrule"
  fi

  local a; a="$(avail_mb)"
  [ "$a" -ge "$START_FLOOR_MB" ] || die "only $a MB of memory available, need $START_FLOOR_MB: stop another model first"

  mkdir -p "$LOGS"
  if [ -n "$KV_DISK_DIR" ]; then
    mkdir -p "$KV_DISK_DIR" || die "cannot create disk KV directory $KV_DISK_DIR"
    [ -d "$KV_DISK_DIR" ] && [ -w "$KV_DISK_DIR" ] || die "disk KV directory is not writable: $KV_DISK_DIR"
  fi
  if [ -n "$MIN_P" ]; then
    export DS4_SERVER_DEFAULT_MIN_P="$MIN_P"
    say "exported DS4_SERVER_DEFAULT_MIN_P=$MIN_P (on this engine build the V4.1 route takes 0.0 from the engine defaults, so this is recorded but does not change this model)"
  fi

  say "starting ds4-server on $HOST_ADDR:$PORT (mapping 113.6 GB; the card reports about two minutes)"
  say "  base:    $BASE_DIR/$GGUF"
  [ -n "$sc" ] && say "  sidecar: $MODEL_DIR/$sc"
  say "  engram:  $ENGRAM_DIR"
  say "  kv disk: ${KV_DISK_DIR:-off}"
  say "  vision:  $([ "$NO_VISION" = 1 ] && echo off || echo "$MODEL_DIR/$VISION")"
  say "  log:     $SERVER_LOG"
  nohup "$SERVER_BIN" "${SERVER_ARGS[@]}" >"$SERVER_LOG" 2>&1 </dev/null &
  echo $! >"$PIDF"
  echo "ds41flash-1017 on $HOST_ADDR:$PORT domain=$DOMAIN mtp=$MTP_MODE vision=$([ "$NO_VISION" = 1 ] && echo off || echo on)" >"$INFOF"

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
  say "stop with: bash $0 stop   drafter: $MTP_MODE   domain: $DOMAIN"
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
  say "kv disk:       ${KV_DISK_DIR:-off} (configured for this invocation; running server may differ)"
  say "base:          $([ "$(base_ok && echo 1)" = 1 ] && echo "assembled ($(human $GGUF_BYTES))" || echo "NOT assembled ($(ls "$BASE_DIR"/$GGUF.part*-of-$NPART 2>/dev/null | wc -l)/$NPART parts)")"
  say "vision:        $([ "$(vision_ok && echo 1)" = 1 ] && echo present || echo missing)   sidecars: $(sidecars_ok && echo present || echo incomplete)   engram: $(engram_ok && echo present || echo missing)"
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | head -c 400; then
    say ""
    say "endpoint:      answering on port $PORT"
  else
    say "endpoint:      no answer on port $PORT"
  fi
  [ -f "$SERVER_LOG" ] && { say ""; say "server log tail:"; tail -n 5 "$SERVER_LOG" | sed 's/^/  /'; }
  return 0
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
  # Temperature 0, the older V4.1 installer's own smoke setting: the check wants a
  # deterministic reply, not a sample. The artifact's recommendation for real
  # requests is temperature 1.0 with top_p 1.0 sent explicitly, and that is left
  # to the client; this launcher does not set a temperature anywhere.
  local out
  out="$(curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4.1-flash-1017","temperature":0,"max_tokens":64,"messages":[{"role":"user","content":"In one sentence, what does a vector-quantized expert layer store?"}]}')"
  case "$out" in
    *'"choices"'*) printf '%s\n' "$out" | sed -n 's/.*"timings":{\(.*\)}.*/  timings: \1/p' | head -1
                   printf '%s\n' "$out" | sed -n 's/.*"content":"\([^"]\{0,90\}\)".*/  answer: \1/p' | head -1;;
    *) warn "the request failed: $(printf '%s' "$out" | head -c 300)"; return 1;;
  esac
}

# --------------------------------------------------------------------------
usage() { awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"; }

CMD=""
parse_args "$@"

case "$CMD" in
  "")        usage; exit 0;;
  preflight) preflight;;
  plan)      cmd_plan;;
  download)  cmd_download;;
  assemble)  cmd_assemble;;
  verify)    cmd_verify;;
  build)     cmd_build;;
  start)     cmd_start;;
  stop)      cmd_stop;;
  status)    cmd_status;;
  logs)      cmd_logs;;
  test)      cmd_test;;
  install|all)
    preflight
    cmd_download
    cmd_assemble
    cmd_verify
    cmd_build
    if [ "$NO_START" = 1 ]; then
      say "not starting the server (--no-start); start it with: bash $0 start"
    else
      cmd_start
    fi;;
esac
