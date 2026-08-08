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
set -uo pipefail
cd "$(dirname "$0")/.."

failed=0
for file in tests/test_*.py; do
    output=$(python3 -m pytest "$file" -q -p no:cacheprovider 2>&1 | tail -1)
    if [[ "$output" == *"failed"* || "$output" == *"error"* ]]; then
        printf '  FAIL  %-28s %s\n' "$(basename "$file")" "$output"
        failed=1
    else
        printf '  ok    %-28s %s\n' "$(basename "$file")" "$output"
    fi
done

if [[ $failed -eq 1 ]]; then
    echo
    echo "At least one file fails in isolation but passes in a full run."
    echo "That is an order-dependent test: it measures collection order, not"
    echo "behaviour. Fix the test, not the ordering."
    exit 1
fi
echo
echo "All files pass in isolation."
