#!/usr/bin/env bash
# Run every test file on its own.
#
# The falsifier registry cannot catch order-dependent tests: it runs one test
# at a time in a subprocess, which inherits the same import state, so a test
# that only passes because another module was collected first looks healthy.
# That is a real blind spot, and it has produced two bugs so far - both in the
# same test, which measured "which test modules happened to be collected"
# rather than what the package implements.
#
# A full `pytest tests/` run cannot find these either, by definition. This can.
# Run it before a freeze, and after adding any module that registers an action.
#
#   ./tools/check_isolation.sh
#
#
# Failure detail is written to a log rather than only printed. A previous run
# reported a failure whose per-file lines were lost to a `| tail -3` in the
# calling command, leaving nothing to act on. A check you cannot read the
# output of is not a check.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG="${TMPDIR:-/tmp}/drf-isolation.log"
: > "$LOG"

failed=0
for file in tests/test_*.py; do
    full=$(python3 -m pytest "$file" -q -p no:cacheprovider 2>&1)
    summary=$(printf '%s\n' "$full" | tail -1)
    if [[ "$summary" == *"failed"* || "$summary" == *"error"* ]]; then
        printf '  FAIL  %-28s %s\n' "$(basename "$file")" "$summary"
        { echo "=== $file ==="; printf '%s\n' "$full"; } >> "$LOG"
        failed=1
    else
        printf '  ok    %-28s %s\n' "$(basename "$file")" "$summary"
    fi
done

if [[ $failed -eq 1 ]]; then
    echo
    echo "At least one file fails in isolation but passes in a full run."
    echo "That is an order-dependent test: it measures collection order, not"
    echo "behaviour. Fix the test, not the ordering."
    echo
    echo "Full output: $LOG"
    exit 1
fi
echo
echo "All files pass in isolation."
echo "(run serially - do not run concurrently with another pytest invocation)"
