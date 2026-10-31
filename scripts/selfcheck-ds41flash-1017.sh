#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCHER="$ROOT/scripts/ds41flash_1017_install.sh"
TEMP="$(mktemp -d)"
cleanup() {
  if [ -s "$TEMP/aux/ds41flash-1017.pid" ]; then
    kill "$(cat "$TEMP/aux/ds41flash-1017.pid")" 2>/dev/null || true
  fi
  rm -rf "$TEMP"
}
trap cleanup EXIT

mkdir -p "$TEMP/bin" "$TEMP/base" "$TEMP/aux/model" "$TEMP/engram"
cat >"$TEMP/bin/server" <<'EOF'
#!/usr/bin/env bash
if [ "${1:-}" = --help ]; then
  echo '[-m GGUF] [--cuda] [--engram-dir DIR] [--mtp-mode MODE] [--mem-budget-mb N] [--host HOST] [--port N] [--kv-disk-dir DIR] [--kv-disk-space-mb N] [--kv-cache-min-tokens N] [--prefix-reuse MODE] [--dspark-verify N] [--tokens N] [--mem-floor-gb N] [--model-id ID] [--tool-memory-max-ids N] [--disable-exact-dsml-tool-replay] [--no-kv-reuse] [--no-graph] [--emit-trace] [--v41-prof] [--cors]'
  [ "${OMIT_TRACE:-0}" = 1 ] || echo '[--trace FILE]'
  exit 0
fi
printf '%s\n' "$@" >"$CAPTURE"
exec sleep 30
EOF
cat >"$TEMP/bin/curl" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' '{"choices":[{"message":{"content":"ok"}}]}'
EOF
chmod +x "$TEMP/bin/server" "$TEMP/bin/curl"
truncate -s 113556639424 "$TEMP/base/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf"
truncate -s 101535150936 "$TEMP/engram/model-00047-of-00048.safetensors"
truncate -s 101537926640 "$TEMP/engram/model-00048-of-00048.safetensors"
export PATH="$TEMP/bin:$PATH" CAPTURE="$TEMP/args" START_FLOOR_MB=0 RUN_FLOOR_MB=0
common=(--dir "$TEMP/aux" --gguf-dir "$TEMP/base" --engram-dir "$TEMP/engram"
        --server-bin "$TEMP/bin/server" --domain none --no-vision --watchdog 0 --allow-shared)
options=(--kv-disk-dir "$TEMP/kv with spaces" --kv-disk-space-mb 8192
         --kv-cache-min-tokens 1024 --prefix-reuse exact --dspark-verify 2
         --tokens 99 --mem-floor-gb 2 --model-id my-model --tool-memory-max-ids 42
         --disable-exact-dsml-tool-replay --no-graph --emit-trace --v41-prof
         --trace "$TEMP/session trace" --cors)

bash -n "$LAUNCHER"
plan="$(bash "$LAUNCHER" "${common[@]}" "${options[@]}" plan)"
[[ "$plan" == *'--kv-disk-dir'* && "$plan" == *'kv\ with\ spaces'* && "$plan" == *'--trace'* ]]
if bash "$LAUNCHER" "${common[@]}" --kv-disk-dir "$TEMP/kv" --prefix-reuse off plan >/dev/null 2>&1; then
  echo 'FAIL: disabled reuse with disk KV was accepted' >&2; exit 1
fi
if bash "$LAUNCHER" "${common[@]}" --kv-disk-space-mb 100 plan >/dev/null 2>&1; then
  echo 'FAIL: disk budget without a directory was accepted' >&2; exit 1
fi
if OMIT_TRACE=1 bash "$LAUNCHER" "${common[@]}" --emit-trace --trace "$TEMP/trace" --no-build build >/dev/null 2>&1; then
  echo 'FAIL: --emit-trace incorrectly satisfied the --trace capability check' >&2; exit 1
fi
bash "$LAUNCHER" "${common[@]}" "${options[@]}" start >/dev/null
for attempt in {1..50}; do
  [ -s "$CAPTURE" ] && break
  sleep 0.1
done
[ -s "$CAPTURE" ] || { echo 'FAIL: mock server did not start' >&2; exit 1; }
for value in "${options[@]}"; do
  grep -Fqx -- "$value" "$CAPTURE" || { echo "FAIL: missing server argument: $value" >&2; exit 1; }
done
echo 'PASS: syntax, plan, validation, capability check, and mock start arguments'
