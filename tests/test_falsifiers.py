"""Every checkpoint test must be capable of failing.

A test that cannot fail proves nothing, and nothing in a normal test run
distinguishes the two - both are green. This module removes that blind spot:
for each entry in `spec/invariants.json` it re-runs the named test in a
subprocess with its falsifier active, and asserts the test **fails**.

Why this exists, concretely. The M1.2 checkpoint originally included
"shuffling posting order changes nothing". After the commutativity rule was
adopted at M1.1, that assertion became true by construction - `fsum` returns
the unique correctly-rounded total and integer addition is commutative, so no
permutation can change a score. The test would have passed on day one and been
logged as evidence. A governing rule adopted mid-project had silently
downgraded a pending checkpoint, and the suite would still have been green.

The general lesson, and the reason this is mechanical rather than a review
habit: **accumulated knowledge from step n changes what step n+1's tests are
worth.** Some of that change is constructive (measurements resolve open design
questions) and some is destructive (a new rule makes an old assertion
vacuous). The constructive half needs judgement. The destructive half does
not, so it should not depend on anyone remembering to look.

Subprocess isolation is deliberate. The mutations in `conftest.py` patch
module-level state, and running them in-process would leak into every
subsequent test in unpredictable order.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conftest import FALSIFIERS  # noqa: E402


def _spec() -> dict:
    with open(ROOT / "spec" / "invariants.json") as f:
        return json.load(f)


INVARIANTS = _spec()["invariants"]


def _run_target(test_id: str, falsifier: str | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if falsifier:
        env["DRF_FALSIFY"] = falsifier
    else:
        env.pop("DRF_FALSIFY", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", test_id, "-q", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )


@pytest.mark.parametrize(
    "invariant", INVARIANTS, ids=[i["id"] for i in INVARIANTS]
)
def test_target_passes_without_its_falsifier(invariant):
    """Control for the control: the test must pass on the real implementation.

    Without this, a target that is simply broken would look like a successful
    falsification below.
    """
    result = _run_target(invariant["test"], None)
    assert result.returncode == 0, (
        f"{invariant['id']}: target test does not pass unfalsified\n"
        f"{result.stdout[-2000:]}"
    )


@pytest.mark.parametrize(
    "invariant", INVARIANTS, ids=[i["id"] for i in INVARIANTS]
)
def test_target_fails_under_its_falsifier(invariant):
    """The point of the module: the mutation must break the test.

    A non-zero exit is accepted whether the target failed an assertion or
    errored during setup. Both mean the mutation was detected, which is the
    property under test; requiring a specific failure mode would make the
    registry brittle without making it stricter.
    """
    result = _run_target(invariant["test"], invariant["id"])
    assert result.returncode != 0, (
        f"{invariant['id']}: test SURVIVED its falsifier and is therefore "
        f"vacuous.\nFalsifier: {invariant['falsifier']}\n"
        f"{result.stdout[-2000:]}"
    )


def test_spec_and_conftest_do_not_drift():
    """Bijection between declared invariants and implemented mutations."""
    declared = {i["id"] for i in INVARIANTS}
    implemented = set(FALSIFIERS)
    assert declared == implemented, (
        f"declared but not implemented: {sorted(declared - implemented)}; "
        f"implemented but not declared: {sorted(implemented - declared)}"
    )


def test_every_invariant_is_fully_documented():
    """A falsifier without a stated rationale is a mutation, not an argument."""
    for invariant in INVARIANTS:
        for key in ("id", "test", "asserts", "falsifier", "why_this_falsifier"):
            assert invariant.get(key), f"{invariant.get('id')}: missing {key}"
        assert "::" in invariant["test"], invariant["id"]


def test_exemptions_carry_reasons():
    """Checkpoints deliberately left unfalsified must say why.

    An empty exemption list would be the suspicious outcome, not a clean one -
    it would mean nobody distinguished test-guarded properties from
    runtime-guarded ones.
    """
    exemptions = _spec()["not_falsified"]
    assert exemptions
    for entry in exemptions:
        assert entry.get("test") and entry.get("reason")
