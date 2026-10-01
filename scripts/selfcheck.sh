#!/usr/bin/env bash
#
# Regression checks for the Bonsai launcher and the budget report.
#
# WHY THIS EXISTS
#   The launcher prints its help from an unquoted heredoc, so that $0 expands
#   into it. The price of that choice is that a backtick in the help text is a
#   command substitution: a backticked word does not print, it runs. That has now
#   happened twice - "budget: command not found" and later "fits: command not
#   found" appearing in the middle of the launcher's own help - and reading the
#   file did not catch either one. Running the help did. So these checks run the
#   things that can break, and they exit non-zero when one of them does.
#
# USAGE
#   bash scripts/selfcheck.sh
#
#   The checks that need the engine and the model take those paths from the
#   environment (ENGINE, MODEL, REPO_DIR, MODEL_ROOT) exactly as the launcher
#   does, and skip themselves with a stated reason when either is missing. That
#   way the same command works on the Spark, on a workstation, and in CI.
#
# EXIT
#   0 when every check passed or skipped, 1 when any check failed.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCHER="$ROOT/scripts/start-bonsai-spark.sh"
BUDGET="$ROOT/scripts/bank-budget.sh"
ENGINE="${ENGINE:-${REPO_DIR:-$HOME/ds4-dfm-rs}/ds4-server}"
MODEL="${MODEL:-${MODEL_ROOT:-$HOME/models}/Ternary-Bonsai-2-27B-PQ2_0.gguf}"
FAILED=0

pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; [ -n "${2:-}" ] && printf '%s\n' "$2" | sed 's/^/      /' | head -8; FAILED=1; }
skip() { printf 'SKIP  %s (%s)\n' "$1" "${2:-}"; }
note() { printf '\n%s\n' "$1"; }

# probe <ctx> <field>: the engine's own quote at that context, no weights opened.
probe() {
  "$ENGINE" -m "$MODEL" --backend cuda -c "$1" --mem-floor-gb 1 --check-config 2>/dev/null \
    | grep -o "\"$2\":[0-9][0-9]*" | head -1 | cut -d: -f2 || true
}

note "selfcheck: $ROOT"

note "-- syntax"
if bash -n "$LAUNCHER" 2>/tmp/selfcheck-n.$$; then pass "launcher: bash -n"; else fail "launcher: bash -n" "$(cat /tmp/selfcheck-n.$$)"; fi
if bash -n "$BUDGET"   2>/tmp/selfcheck-n.$$; then pass "budget: bash -n";   else fail "budget: bash -n"   "$(cat /tmp/selfcheck-n.$$)"; fi
rm -f /tmp/selfcheck-n.$$

note "-- help text runs nothing of itself"
HELP="$(bash "$LAUNCHER" help 2>&1 || true)"
case "$HELP" in
  *"command not found"*) fail "help executes no commands" "$(printf '%s' "$HELP" | grep -n 'command not found' | head -3)" ;;
  *"unbound variable"*)  fail "help executes no commands" "$(printf '%s' "$HELP" | grep -n 'unbound variable' | head -3)" ;;
  *) pass "help executes no commands" ;;
esac
if [ -n "${HELP//[[:space:]]/}" ]; then pass "help prints text"; else fail "help prints text" "empty output"; fi
case "$HELP" in
  *"both halves of that sentence"*) pass "help keeps its prose" ;;
  *) fail "help keeps its prose" "a backtick ate a word again: look for a \` in the usage heredoc" ;;
esac

note "-- dispatch"
rc=0; OUT="$(bash "$LAUNCHER" bogus 2>&1)" || rc=$?
[ "$rc" = "2" ] && pass "unknown command exits 2" || fail "unknown command exits 2" "exit $rc"
case "$OUT" in *"unknown command"*) pass "unknown command says so" ;; *) fail "unknown command says so" "$(printf '%s' "$OUT" | head -3)" ;; esac

rc=0; OUT="$(bash "$LAUNCHER" 131k 2>&1)" || rc=$?
[ "$rc" = "2" ] && pass "bare shape exits 2" || fail "bare shape exits 2" "exit $rc"
case "$OUT" in
  *"did you mean"*"start 131k"*) pass "bare shape names the command it wanted" ;;
  *) fail "bare shape names the command it wanted" "$(printf '%s' "$OUT" | head -3)" ;;
esac

note "-- engine and model"
if [ ! -x "$ENGINE" ]; then
  skip "plan, fits, budget" "no engine at $ENGINE"
elif [ ! -f "$MODEL" ]; then
  skip "plan, fits, budget" "no model at $MODEL"
else
  say() { :; }   # the launcher's own say is not used here
  pf_ok=1; pf_out="$(bash "$LAUNCHER" plan 45k 2>&1)" || pf_ok=0
  if [ "$pf_ok" = "1" ]; then pass "plan 45k runs to a verdict"; else fail "plan 45k runs to a verdict" "$(printf '%s' "$pf_out" | head -5)"; fi
  case "$pf_out" in
    *"did not reach a verdict"*) fail "plan reports a verdict, not a failed run" "$(printf '%s' "$pf_out" | tail -4)" ;;
    *) pass "plan distinguishes a verdict from a failed run" ;;
  esac

  if [ -n "$(probe 32768 shared_weights)" ]; then
    w="$(probe 32768 shared_weights)"; f="$(probe 32768 floor)"
    pb1="$(probe 32768 per_bank)"; pb2="$(probe 65536 per_bank)"; t1="$(probe 32768 total)"
    # the identity the recipe documents: total = weights + banks x per_bank + floor
    if [ "$(( w + pb1 + f ))" = "$t1" ]; then
      pass "the engine's quote satisfies total = weights + per_bank + floor"
    else
      fail "the engine's quote satisfies total = weights + per_bank + floor" \
           "weights=$w per_bank=$pb1 floor=$f total=$t1"
    fi
    # the slope the report prints must be the slope its own quotes imply
    derived=$(( (pb2 - pb1) / (65536 - 32768) ))
    printed="$(SERVER="$ENGINE" MODEL="$MODEL" CTX_CEILING=262144 CTX_REF=32768 \
               bash "$BUDGET" 2>/dev/null | grep -o '[0-9][0-9]* bytes per token' | awk '{print $1}')"
    if [ "$derived" = "$printed" ]; then
      pass "the budget report's slope ($printed bytes/token) matches direct quotes"
    else
      fail "the budget report's slope matches direct quotes" "direct=$derived printed=${printed:-none}"
    fi
  else
    skip "quote identity and slope" "the engine produced no quote"
  fi

  fits_out="$(bash "$LAUNCHER" fits 2>&1)" && pass "fits runs" || fail "fits runs" "$(printf '%s' "$fits_out" | tail -4)"
  case "$fits_out" in
    *"the right step here is"*|*"no shape fits the memory free right now"*) pass "fits reaches a recommendation" ;;
    *) fail "fits reaches a recommendation" "$(printf '%s' "$fits_out" | tail -4)" ;;
  esac
fi

note ""
if [ "$FAILED" = "0" ]; then
  printf 'selfcheck: all checks passed\n'
else
  printf 'selfcheck: FAILURES above\n'
fi
exit "$FAILED"
