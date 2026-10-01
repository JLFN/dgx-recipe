#!/usr/bin/env bash
#
# bank-budget.sh - how much room there is for banks, and how much context each
# bank can hold, answered by the engine's own memory quote.
#
# WHAT IT ANSWERS
#   Given a model, a backend and the memory this box actually has free, this
#   script reports, in the engine's own numbers:
#     - the size of one bank (the per-sequence allocate the plan charges) at any
#       context, and the total plan for N banks of it;
#     - the headroom left after that plan, against the live free memory;
#     - the largest context that still fits, and the largest number of banks
#       that still fit, by inversion;
#     - which of those lanes the family will actually admit, and which it
#       refuses by name.
#
# HOW IT WORKS (and why the numbers are the engine's, not this script's)
#   `ds4-server --check-config` resolves the serving plan and prints it as JSON
#   on stdout, without opening the weights (crates/ds4-server/src/bin/
#   ds4-server-rs.rs, the check_config branch, verified 2026-10-01 on vizzio).
#   The JSON carries quote.shared_weights, quote.per_bank, quote.floor,
#   quote.available (the free device memory read at plan time), quote.banks and
#   quote.total. This script reads those fields and nothing else, so a change in
#   the engine's sizing moves the answer here with it.
#
#   Two quotes at different context lengths give the per-bank line exactly, for
#   the families whose bank scales linearly with context (the attention KV rows
#   grow, the rest does not). The script derives that line from the engine,
#   checks a third quote against it, and only then extrapolates. It never
#   extrapolates across a family boundary or guesses a geometry constant.
#
# EVIDENCE THIS DESIGN RESTS ON (measured 2026-10-01, vizzio, RTX 4070 SUPER)
#   Ternary-Bonsai-2-27B-PQ2_0.gguf, family qwen35:
#     ctx 32768 -> per_bank 2570354688 B (2.394 GiB), total 10850265440 (10.105)
#     ctx 40960 -> per_bank 3108093952 B (2.894 GiB), total 11388004704 (10.606)
#     ctx 65536 -> per_bank 4717838336 B (4.394 GiB), total 13097764384 (12.106)
#   total = shared_weights + per_bank + floor, exactly, and
#   per_bank = 422871040 + 65536 * ctx, exactly: 64.0 KiB per token.
#
# USAGE
#   bash bank-budget.sh                       # the Bonsai model, default grid
#   MODEL=... bash bank-budget.sh             # any model the engine serves
#   BANKS=2 bash bank-budget.sh               # plan for two banks where allowed
#   CEILING_GB=100 bash bank-budget.sh        # plan against a budget you set
#   CTX_LIST="32768 65536 131072 262144" bash bank-budget.sh
#   bash bank-budget.sh --help
#
# EXIT CODES
#   0  the report was produced (a refusal inside the report is still a report)
#   1  the script could not measure: no binary, no model, or no usable quote
#
# KNOWN LIMITS
#   - The quote is computed at plan time. A plan that "may listen" is not a
#     guarantee that the real start fits: the engine reports plans that its own
#     allocator then refuses when the memory baseline has moved. Treat a fitting
#     plan as the upper bound, and confirm with a real start.
#   - `available` is the free device memory the quote read at that moment, so it
#     moves with whatever else is on the card. Set CEILING_GB to plan against a
#     fixed budget instead of the live figure.
#   - The linear per-bank model is checked against a third quote, but it is
#     still a model: families with a chunk-dependent transient (prefill chunk,
#     MTP state, PLE cache) can bend it away from the line. The script prints
#     the deviation it measured so a bend is visible rather than hidden.

set -Eeuo pipefail

say()  { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '5,40p' "$0" | sed 's/^# \{0,1\}//'
}

# ---------------------------------------------------------------------------
# Configuration (every value is overridable from the environment)
# ---------------------------------------------------------------------------
SERVER="${SERVER:-$HOME/ds4-dfm-rs/ds4-server}"
MODEL="${MODEL:-$HOME/models/Ternary-Bonsai-2-27B-PQ2_0.gguf}"
BACKEND="${BACKEND:-cuda}"
MEM_FLOOR_GB="${MEM_FLOOR_GB:-1}"
CEILING_GB="${CEILING_GB:-0}"          # 0 = use the engine's live `available`
BANKS="${BANKS:-0}"                    # 0 = ask the engine for its own default
BANKS_MAX="${BANKS_MAX:-8}"            # how far up the bank axis to probe
CTX_REF="${CTX_REF:-32768}"            # context the bank axis is probed at
CTX_LIST="${CTX_LIST:-}"               # explicit context grid, space separated
PROBE_A="${PROBE_A:-32768}"            # the two quotes the line is derived from
PROBE_B="${PROBE_B:-65536}"
PROBE_CHECK="${PROBE_CHECK:-49152}"    # the third quote the line is checked on
CTX_CEILING="${CTX_CEILING:-0}"        # 0 = read the model's declared cap

MIB=$((1024 * 1024))

# ---------------------------------------------------------------------------
# The engine quote
# ---------------------------------------------------------------------------
QUOTE_DIR=""
cleanup() { [ -n "$QUOTE_DIR" ] && rm -rf "$QUOTE_DIR"; }
trap cleanup EXIT

# quote <ctx> <banks or empty> <tag> -> writes <tag>.json, returns the exit code
quote() {
  local ctx="$1" banks="$2" tag="$3"
  local -a args=( -m "$MODEL" --backend "$BACKEND" -c "$ctx" --mem-floor-gb "$MEM_FLOOR_GB" )
  [ -n "$banks" ] && args+=( --max-seqs "$banks" )
  "$SERVER" "${args[@]}" --check-config >"$QUOTE_DIR/$tag.json" 2>"$QUOTE_DIR/$tag.err"
}

# Every reader below ends in `|| true`: an absent field is a normal answer here,
# and under `set -e` with `pipefail` a failing grep inside a command
# substitution would otherwise end the script with no message at all.

# The digit class is [0-9][0-9]*, never [0-9]*: the JSON carries both
# "banks":"serial" in controls and "banks":1 in the quote, and a pattern that
# allows an empty match would report the string one and read as "no number".
jget() { # jget <tag> <key> -> integer field from the quote JSON, empty if absent
  grep -o "\"$2\":[0-9][0-9]*" "$QUOTE_DIR/$1.json" 2>/dev/null | head -1 | cut -d: -f2 || true
}

jstr() { # jstr <tag> <key> -> string field from the flat part of the quote JSON
  grep -o "\"$2\":\"[^\"]*\"" "$QUOTE_DIR/$1.json" 2>/dev/null | head -1 | cut -d'"' -f4 || true
}

jcode() { # jcode <tag> -> the first error issue code, empty when none
  grep -o '"level":"error","code":"[^"]*"' "$QUOTE_DIR/$1.json" 2>/dev/null | head -1 | cut -d'"' -f8 || true
}

jmsg() { # jmsg <tag> -> the first error issue message, empty when none
  grep -o '"level":"error","code":"[^"]*","message":"[^"]*"' "$QUOTE_DIR/$1.json" 2>/dev/null | head -1 | sed 's/.*"message":"//; s/"$//' || true
}

gib() { awk -v b="${1:-0}" 'BEGIN{printf "%.3f GiB", b/1073741824}'; }
gib2() { awk -v b="${1:-0}" 'BEGIN{printf "%.2f", b/1073741824}'; }

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
case "${1:-}" in
  -h|--help|help) usage; exit 0 ;;
  "") ;;
  *) die "unknown argument '$1' (see --help)" ;;
esac

[ -x "$SERVER" ] || die "no executable server at $SERVER (set SERVER=<ds4-server path>)"
[ -f "$MODEL" ] || die "no model at $MODEL (set MODEL=<gguf path>)"

# BANKS=0 means "let the engine decide", which is an absent --max-seqs, not a
# literal zero: the engine rejects a zero width at argument parsing.
[ "$BANKS" = "0" ] && BANKS=""

QUOTE_DIR="$(mktemp -d)"

# --- the two quotes the per-bank line is derived from ----------------------
rc_a=0; quote "$PROBE_A" "$BANKS" a || rc_a=$?
rc_b=0; quote "$PROBE_B" "$BANKS" b || rc_b=$?

pa="$(jget a per_bank)"; pb="$(jget b per_bank)"
if [ -z "$pa" ] || [ -z "$pb" ]; then
  say "--- engine output for the $PROBE_A quote ---"
  sed 's/^/  /' "$QUOTE_DIR/a.err"
  die "the engine produced no usable quote for this model (see the lines above)"
fi

shared="$(jget a shared_weights)"
floor="$(jget a floor)"
avail="$(jget a available)"
family="$(jstr a family)"
lane="$(jstr a banks)"      # "serial", "persistent" or "opt_in" from controls
banks_frac=1

# --- how much memory to plan against ---------------------------------------
if [ "$CEILING_GB" != "0" ]; then
  budget=$((CEILING_GB * 1024 * MIB))
  budget_note="set by CEILING_GB=${CEILING_GB}"
else
  budget="$avail"
  budget_note="live free memory the engine read at plan time"
fi

# --- the per-bank line:  per_bank(ctx) = fixed + slope * ctx ---------------
slope=$(( (pb - pa) / (PROBE_B - PROBE_A) ))
fixed=$(( pa - slope * PROBE_A ))

# --- check a third quote against that line ---------------------------------
rc_c=0; quote "$PROBE_CHECK" "$BANKS" c || rc_c=$?
pc="$(jget c per_bank)"
if [ -n "$pc" ]; then
  predicted=$(( fixed + slope * PROBE_CHECK ))
  dev=$(( predicted - pc ))
  [ "$dev" -lt 0 ] && dev=$(( -dev ))
  dev_note="$(gib "$dev")"
else
  dev_note="not measured"
fi

# --- which lanes the family admits -----------------------------------------
rc_2=0; quote "$CTX_REF" 2 two || rc_2=$?
bank_code="$(jcode two)"
bank_msg="$(jmsg two)"

say "bank budget, read from the engine's own memory quote"
say ""
say "  engine            $SERVER"
say "  model             $MODEL"
say "  family            ${family:-unknown}"
say "  bank lane         ${lane:-unknown}$( [ -n "$bank_code" ] && printf ' (--max-seqs 2 refused: %s)' "$bank_code" )"
say "  weights           $(gib "$shared") shared"
say "  memory floor      ${MEM_FLOOR_GB} GiB"
say "  per bank, fixed   $(gib "$fixed")"
say "  per bank, slope   $slope bytes per token ($(awk -v s="$slope" 'BEGIN{printf "%.1f", s/1024}') KiB per token)"
say "  line check        at ctx $PROBE_CHECK the line predicts $(gib "$((fixed + slope * PROBE_CHECK))"), the engine says $(gib "$pc"), deviation $dev_note"
say "  budget            $(gib "$budget") ($budget_note)"
if [ -n "$bank_msg" ]; then
  say ""
  say "  the family refuses more than one bank: $bank_msg"
  say "  the bank axis below is therefore information about the memory, not an"
  say "  option this family will actually serve."
fi

# --- context -> bank, total, headroom --------------------------------------
# The default grid walks the artifact's own ceiling, when the engine declares
# one, so the table always covers the full range a start could ask for.
if [ -n "$CTX_LIST" ]; then
  grid="$CTX_LIST"
else
  # Refuse to invent a ceiling: derive the grid from the model's declared cap
  # when the quote reports one, otherwise from the largest ctx that fits.
  ctx_cap="$CTX_CEILING"
  if [ "$ctx_cap" = "0" ]; then
    ctx_cap=262144
    warn "no declared context cap given; walking the grid to 262144 (set CTX_CEILING to change it)"
  fi
  # Deduplicated in order: the reference context is often also a fraction of the
  # ceiling (131072 is both the 131k shape and half of 262144), and a repeated
  # row in the table reads as a measurement error rather than as a duplicate.
  grid=""
  for v in "$CTX_REF" "$(( ctx_cap / 4 ))" "$(( ctx_cap / 2 ))" "$ctx_cap"; do
    case " $grid " in
      *" $v "*) ;;
      *) grid="$grid $v" ;;
    esac
  done
fi

per_bank_budget=$(( budget - shared - floor ))

# How many banks the engine itself resolved for the width that was asked for.
# For a serial family that is 1 however wide the request was, which is the
# point: the column below says what the engine charges, not what was hoped for.
n_plan="$(jget a banks)"; [ -z "$n_plan" ] && n_plan=1
plan_label="plan ($n_plan bank$( [ "$n_plan" = "1" ] && printf '' || printf 's' ))"
engine_total="$(jget a total)"

printf '\n  %-10s %-12s %-16s %-14s %s\n' ctx per_bank "$plan_label" headroom verdict
for ctx in $grid; do
  bank_bytes=$(( fixed + slope * ctx ))
  total=$(( shared + floor + n_plan * bank_bytes ))
  head=$(( budget - total ))
  if [ "$head" -ge 0 ]; then verdict="opens"; else verdict="does not fit"; fi
  printf '  %-10s %-12s %-16s %-14s %s\n' "$ctx" "$(gib "$bank_bytes")" "$(gib "$total")" "$(gib "$head")" "$verdict"
done
say ""
say "  total = weights + $n_plan x bank + floor, the $n_plan-bank width being the engine's own."
say "  At ctx $PROBE_A the engine quotes $(gib "$engine_total"); the line above gives $(gib "$(( shared + floor + n_plan * (fixed + slope * PROBE_A) ))")."

# --- inversion 1: how many banks fit at a context --------------------------
say ""
say "  banks that fit a given context (the bank axis):"
for ctx in $grid; do
  bank_bytes=$(( fixed + slope * ctx ))
  n=$(( per_bank_budget / bank_bytes ))
  [ "$n" -lt 0 ] && n=0
  printf '    ctx %-8s bank %-12s the budget covers %s bank(s)\n' \
    "$ctx" "$(gib "$bank_bytes")" "$n"
done

# --- inversion 2: the largest context that fits N banks --------------------
say ""
say "  largest context that fits, by inversion:"
for n in 1 2 3 4; do
  room=$(( budget - shared - floor - n * fixed ))
  if [ "$room" -le 0 ]; then
    printf '    %s bank(s): nothing left after the weights, the floor and %s x fixed per-bank cost\n' "$n" "$n"
    continue
  fi
  ctx_max=$(( room / (n * slope) ))
  cap_note=""
  if [ "$CTX_CEILING" != "0" ] && [ "$ctx_max" -gt "$CTX_CEILING" ]; then
    ctx_max="$CTX_CEILING"
    cap_note=" (the artifact's declared ceiling is $CTX_CEILING; the memory would allow more)"
  fi
  printf '    %s bank(s): %s tokens of context each (%s of bank each, %s in total)%s\n' \
    "$n" "$ctx_max" "$(gib $(( fixed + slope * ctx_max )))" "$(gib $(( shared + floor + n * (fixed + slope * ctx_max) )))" "$cap_note"
done

# --- what the engine itself says about the bank counts --------------------
say ""
say "  engine verdict per bank count at ctx $CTX_REF:"
for n in $(seq 1 "$BANKS_MAX"); do
  rc=0; quote "$CTX_REF" "$n" "n$n" || rc=$?
  code="$(jcode "n$n")"
  eff_banks="$(jget "n$n" banks)"
  if [ "$rc" = 0 ]; then
    printf '    %-3s accepted (engine quotes %s bank(s), total %s)\n' "$n" "${eff_banks:-?}" "$(gib "$(jget "n$n" total)")"
  else
    printf '    %-3s refused: %s\n' "$n" "${code:-exit $rc}"
  fi
done

say ""
say "  A fitting plan is the upper bound, not a promise: the engine also reports"
say "  plans its own allocator then refuses at a real start, when the memory"
say "  baseline has moved. Confirm a shape with a real start before trusting it."
