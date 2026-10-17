# QA report: Prism Bonsai 2 27B (qwen35) on the Spark

**Unit:** commits `a3b5fdb` through `a0e2794` on `feat/shape-profiles`: the Bonsai
recipe, the launcher, the bank/context budget tool, the runbook and the README.
**Revision under test:** `a0e2794` (the revision being pushed).
**Date:** 2026-10-01.
**Tester:** the session model, which the operator named as the QA model for this
project (rule 19 allows the operator to choose, and there is no recorded QA
guardrail for this project). A fresh-context subagent on the same model was
launched twice as an independent reviewer; both runs were cancelled by the
environment before finishing, so the pass below is the in-session one. Its
limitation is stated at the end.
**Method:** every claim in the deliverables was treated as a hypothesis and run
from the outside: the scripts were executed, the engine's own quotes were read
back independently, and the numbers printed by the tools were re-derived.

**Surfaces covered:** ALL NEW SURFACES - `start-bonsai-spark.sh` (download,
verify, build, test, shapes, plan, fits, budget, start, stop, status, logs, all,
plus the unknown-command and bare-shape paths), `bank-budget.sh`, the recipe
contract, the runbook's factual claims, and `selfcheck.sh`.

---

## 1. Static checks

```
$ bash -n scripts/start-bonsai-spark.sh   -> PASS
$ bash -n scripts/bank-budget.sh          -> PASS
$ bash -n scripts/selfcheck.sh            -> PASS
```

No shell static analyser is installed on this machine (`shellcheck`, `shfmt`,
`bashate`, `checkbashisms` all absent; `shellcheck` 0.10.0 is available from apt
if you want it). Static analysis here is therefore `bash -n` plus the executable
checks below, which is why the help-text defect in section 6 was caught by
running the help rather than by reading it.

## 2. Dispatch and help

```
$ ./scripts/start-bonsai-spark.sh bogus ; echo $?
unknown command: bogus
2

$ ./scripts/start-bonsai-spark.sh 131k ; echo $?
unknown command: 131k
did you mean:  bash ./scripts/start-bonsai-spark.sh start 131k
or to size it: bash ./scripts/start-bonsai-spark.sh budget 131k
2

$ ./scripts/start-bonsai-spark.sh help | grep -c "command not found"
0            (after the fix in section 6; it was 3 before)
```

## 3. The measurement tools, against the engine's own quotes

`fits` on the 12 GiB reference box:

```
memory on this box right now
  MemTotal              31.2 GiB   (the operating system)
  MemAvailable          19.8 GiB
  shape   ctx       plan         headroom      verdict
  262k    262144    24.109 GiB   -             refused (quote_overflow)
  131k    131072    16.107 GiB   -             refused (quote_overflow)
  64k     65536     12.106 GiB   -             refused (quote_overflow)
  45k     45056     10.855 GiB   0.5 GiB       opens
  the engine reads 11.3 GiB free when it checks a plan
this launcher's lock    /tmp/ds4.lock   (passed to the engine as DS4_LOCK_FILE)
the engine slot        free (nothing holds this lock file)
the right step here is 45k ...
```

`plan auto` resolved to that same shape and said so: `auto: chose 45k, the deepest
shape the engine accepted just now`.

Independent re-derivation, not trusting the tool's arithmetic: `ds4-server
--check-config` run directly at 32768 and 65536 gives per_bank 2570354688 and
4718362624, a slope of 65552 bytes per token, which is exactly the slope the
budget report prints; and `shared_weights + per_bank + floor = total` holds on
the raw quote (7206168928 + 2570354688 + 1073741824 = 10850265440). Both are now
checked on every run by `selfcheck.sh`.

## 4. The plan verdict versus a failed run

With a runtime directory that did not exist yet (a box where no start has run):

```
$ RUNTIME=/tmp/fresh ./scripts/start-bonsai-spark.sh plan 45k    -> "check-config exit 0: this plan may listen"
$ RUNTIME=/tmp/fresh ./scripts/start-bonsai-spark.sh plan 262k   -> "check-config exit 2: the engine refused this plan"
```

These are separate from the third case, a check that fails to run at all, which
now says so instead of blaming the shape (this was defect D2, section 6).

## 5. The lock, the sharing route, and the neighbour

```
CONTROL, same lock file:  ds4: another ds4 process is already running (pid 117121); refusing to start
TEST, own lock file:      CUDA backend initialized on NVIDIA GeForce RTX 4070 SUPER (sm_89)
                          model catalog base: 851 tensors, ...        (own lock pid 117145)
```

A launcher start with `LOCK_FILE=/tmp/bonsai-alt.lock` came up and wrote its pid
into that file; its engine took that file rather than the default. With a
neighbour running, `fits` printed `the engine slot busy: ds4-server (pid 117482)`,
the engine's refusal text, and the exact command to share the box. A start from a
second runtime directory and port was refused quoting that pid. After the server
stopped, the same lock file, still containing a dead pid, read as free in both
`fits` and `status`: the staleness check works.

## 6. Defects found, and their state

| # | Defect | Severity | State |
|---|---|---|---|
| D1 | Help text printed from an unquoted heredoc executed three backticked words (`fits` twice, `plan` once), printing `line 885: fits: command not found` inside its own help and eating those words from two sentences. Found by this QA pass. | Medium: it is the first thing a user sees when their invocation is wrong, and it makes the tool look broken. | **Fixed** in `a0e2794`; now guarded by `selfcheck.sh`, which fails if the help executes anything or if the canary sentence loses its words. |
| D2 | `cmd_plan` wrote its JSON into `$RUNTIME` without creating the directory, so on a box where no start had run the redirect failed and the failure was reported as "this plan was rejected", sending the reader to lower a shape that was never the problem. Found by my own pass during this unit. | High: a false diagnosis on a healthy configuration. | **Fixed** in `2a9bf3d`; the three outcomes are now distinct, and `selfcheck.sh` checks that a verdict is not confused with a failed run. |
| D3 | `cmd_test` invoked the engine's make targets without passing the model path, so they used the engine's own default `/data/models/...`, which does not exist on the Spark (`cannot open model ... No such file or directory`). Reported by the operator. | High on the Spark: the documented test command failed for a path reason. | **Fixed** in `96b42d1`; the launcher now exports its own model path into every make invocation. |
| D4 | The launcher recorded the wrong pid for the server (`setsid` forks, so `$!` named an intermediate that exited), which left `stop` unable to find it and the start waiting on a child. Found during this unit. | High: `stop` would have reported "not running" while the server held the card. | **Fixed** in `9248634`; verified by a start, status, request, stop cycle. |
| D5 | The help text executed its backticks once before, in the same file (`budget: command not found`). Found by my own pass earlier in this unit. | Medium. | **Fixed** in `c769d1f`; recurred as D1, which is why the check now exists rather than another fix. |

## 7. Lifecycle, end to end, on the reference box

```
$ bash start-bonsai-spark.sh start          (ctx 32768, port 8011)
  listening on 0.0.0.0:8011 model_id=Ternary-Bonsai-2-27B-PQ2_0 engine=open ...
  id:           Ternary-Bonsai-2-27B-PQ2_0 (aliases prism-bonsai-2-27b*)

$ curl .../chat/completions -d '{"model":"prism-bonsai-2-27b", ... "max_tokens":64}'
  finish: stop
  content: 'Paris'
  usage: prompt 49, completion 27

$ bash start-bonsai-spark.sh status
  device memory: 9984 MiB, 12282 MiB
  engine slot:   busy: ./ds4-server (pid 121438, holding /tmp/ds4.lock)

$ bash start-bonsai-spark.sh stop
  -> no ds4-server left, port 8011 free, engine slot free
```

The alias in that request is the one the open-grok entry sends
(`model = "prism-bonsai-2-27b"`), and it is accepted: the Rust host lists it in
`model_alias_known` (crates/ds4-server/src/models.rs). The 64-token cap also
confirms the documented reasoning floor: at 32 tokens the same prompt returns
reasoning with empty content.

## 8. The artifact, and the Spark's copies

```
7206168928 bytes
3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1
```

Both match the published values checked against the sibling tree's
`download_model.sh`. On the Spark, `bank-budget.sh` and the runbook were
checksum-identical to this revision at the time of writing, and the launcher copy
was behind by the D1 fix; it is copied again with this report.

## 9. What this pass does not establish

- **Nothing on the Spark has been executed by this pass.** This session has no
  shell access there, so the Spark paths (`fits`, `auto`, the lock override, the
  262k shape) are verified only by code reading, by local runs of the same
  scripts, and by HTTP probes against the server the operator started there. The
  first `fits` run on the Spark is still the confirmation.
- **The interactive prompt** was exercised only through a pty (`script -q`), not
  on a real terminal.
- **The 262k and 131k shapes** have never been started anywhere; only their plans
  have been quoted.
- **No shell static analyser** ran (not installed).
- **The tester is the author.** That is the operator's choice, and it is a real
  limitation: a self-review shares the author's blind spots. It is mitigated by
  running everything from the outside, by re-deriving the engine's numbers
  independently, and by `selfcheck.sh`, which now makes the most recurrent defect
  class (D1, D5) fail loudly. It is not the same as an independent reviewer.

## Verdict

verdict: overall PASS
