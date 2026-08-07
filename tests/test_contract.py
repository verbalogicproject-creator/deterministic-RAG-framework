"""M1.0 checkpoint: the contract machinery is load-bearing.

These tests exist to prove that the labels in spec/actions.json are enforced
rather than described. Each one attempts a violation and requires it to fail.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drf.contract import (  # noqa: E402
    ACTIONS,
    Advisory,
    ActionOutput,
    AuthorityViolation,
    DeclarationError,
    DeterminismViolation,
    Trace,
    action,
    reset_replay_log,
    strict_replay,
)
from drf.hashing import canonical_json, edge_id, node_id, sha256_value  # noqa: E402
from drf.fixed import quantize, exact_sum  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    reset_replay_log()
    yield
    reset_replay_log()


# --------------------------------------------------------------------------
# canonical_json stability
# --------------------------------------------------------------------------

def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_json_rejects_nan():
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_canonical_json_stable_across_processes():
    """Hash order must not depend on PYTHONHASHSEED."""
    snippet = (
        "import sys; sys.path.insert(0, %r);"
        "from drf.hashing import sha256_value;"
        "print(sha256_value({'z':1,'a':[1,2,{'q':True,'b':None}],'m':'\\u00e9'}))"
        % str(ROOT)
    )
    digests = set()
    for seed in ("0", "1", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1, f"canonical hash varied across hash seeds: {digests}"


# --------------------------------------------------------------------------
# Content-addressed IDs
# --------------------------------------------------------------------------

def test_node_id_is_pure_function_of_content():
    a = node_id(type="t", name="n", description="d", source="s")
    b = node_id(type="t", name="n", description="d", source="s")
    c = node_id(type="t", name="n", description="d2", source="s")
    assert a == b
    assert a != c


def test_edge_id_dedupes_identical_triples():
    """Duplicate (from, to, type) edges collapse structurally, because the
    edge_id is the PRIMARY KEY. No dedupe pass is needed."""
    assert edge_id(from_id="a", to_id="b", type="uses") == \
           edge_id(from_id="a", to_id="b", type="uses")
    assert edge_id(from_id="a", to_id="b", type="uses") != \
           edge_id(from_id="b", to_id="a", type="uses")


# --------------------------------------------------------------------------
# Fixed point
# --------------------------------------------------------------------------

def test_quantize_returns_int():
    assert isinstance(quantize(0.123456789), int)


def test_exact_sum_is_correctly_rounded_and_order_independent():
    """The properties ranking actually relies on.

    Deliberately does NOT assert that sum() is worse. Measured on CPython
    3.12: sum() uses Neumaier compensation and matched fsum on 200,000
    random sums with no order dependence. fsum is used because correct
    rounding is a *documented language guarantee* that survives a change of
    interpreter, not because sum() misbehaves here.
    """
    vals = [1.0, 1e100, 1.0, -1e100]
    assert exact_sum(vals) == 2.0                              # exact
    assert exact_sum(vals) == exact_sum(list(reversed(vals)))  # order-free

    import random
    rng = random.Random(1234)
    for _ in range(2000):
        v = [rng.uniform(-1, 1) * 10 ** rng.randint(-8, 8) for _ in range(12)]
        assert exact_sum(v) == exact_sum(list(reversed(v)))


# --------------------------------------------------------------------------
# Declaration enforcement
# --------------------------------------------------------------------------

def test_deterministic_action_may_not_declare_confidence():
    @action("t.det_conf", determinism="deterministic", authority="authoritative")
    def bad(x):
        return ActionOutput(value=x, confidence=0.9)

    with pytest.raises(DeclarationError, match="must not declare a confidence"):
        bad(1)


def test_probabilistic_action_must_declare_confidence():
    @action("t.prob_noconf", determinism="probabilistic", authority="advisory")
    def bad(x):
        return ActionOutput(value=x)

    with pytest.raises(DeclarationError, match="must declare a confidence"):
        bad(1)


def test_confidence_must_be_in_unit_interval():
    @action("t.prob_range", determinism="probabilistic", authority="advisory")
    def bad(x):
        return ActionOutput(value=x, confidence=1.5)

    with pytest.raises(DeclarationError, match="outside"):
        bad(1)


def test_action_must_return_action_output():
    @action("t.raw", determinism="deterministic", authority="authoritative")
    def bad(x):
        return x

    with pytest.raises(DeclarationError, match="must return ActionOutput"):
        bad(1)


# --------------------------------------------------------------------------
# The replay check - the mechanism that makes "deterministic" load-bearing
# --------------------------------------------------------------------------

def test_mislabelled_deterministic_action_raises_on_second_call():
    counter = {"n": 0}

    @action("t.liar", determinism="deterministic", authority="authoritative")
    def liar(x):
        counter["n"] += 1
        return ActionOutput(value=counter["n"])

    liar(1)
    with pytest.raises(DeterminismViolation, match="different result"):
        liar(1)


def test_honest_deterministic_action_survives_replay():
    @action("t.honest", determinism="deterministic", authority="authoritative")
    def honest(x):
        return ActionOutput(value=x * 2)

    assert honest(21).value == 42
    assert honest(21).value == 42


def test_strict_replay_catches_on_first_call():
    counter = {"n": 0}

    @action("t.liar_strict", determinism="deterministic", authority="authoritative")
    def liar(x):
        counter["n"] += 1
        return ActionOutput(value=counter["n"])

    with pytest.raises(DeterminismViolation, match="strict replay"):
        with strict_replay():
            liar(1)


def test_probabilistic_action_is_not_replay_checked():
    counter = {"n": 0}

    @action("t.varies", determinism="probabilistic", authority="advisory")
    def varies(x):
        counter["n"] += 1
        return ActionOutput(value=counter["n"], confidence=0.5)

    varies(1)
    varies(1)  # must not raise


# --------------------------------------------------------------------------
# Advisory boxing - the authority firewall
# --------------------------------------------------------------------------

def test_advisory_result_is_boxed():
    @action("t.adv", determinism="deterministic", authority="advisory")
    def adv(x):
        return ActionOutput(value=[1, 2, 3])

    assert isinstance(adv(1).value, Advisory)


def test_unwrap_refused_outside_allowlist():
    """This test module is not drf.retrieval.merge, so unwrapping must fail.
    That is the firewall: advisory data cannot reach authoritative code."""
    box = Advisory([1, 2, 3], provider="test")
    with pytest.raises(AuthorityViolation, match="may not unwrap"):
        box.unwrap()


def test_is_empty_is_safe_for_anyone():
    assert Advisory([], provider="t").is_empty() is True
    assert Advisory([1], provider="t").is_empty() is False


# --------------------------------------------------------------------------
# spec <-> code bijection
# --------------------------------------------------------------------------

def _spec_actions():
    with open(ROOT / "spec" / "actions.json") as f:
        return json.load(f)["actions"]


def test_every_spec_action_declares_valid_axes():
    for entry in _spec_actions():
        assert entry["determinism"] in ("deterministic", "probabilistic"), entry["name"]
        assert entry["authority"] in ("authoritative", "advisory"), entry["name"]


def test_probabilistic_actions_are_never_authoritative():
    """The architectural rule: uncertainty may never hold authority."""
    for entry in _spec_actions():
        if entry["determinism"] == "probabilistic":
            assert entry["authority"] == "advisory", (
                f"{entry['name']} is probabilistic AND authoritative - "
                "this combination is forbidden"
            )


def test_spec_action_names_are_unique():
    names = [e["name"] for e in _spec_actions()]
    assert len(names) == len(set(names))


def test_determinism_labels_carry_evidence_or_are_structural():
    """A 'deterministic' label on anything touching the network must cite a
    measurement. Structural determinism (pure functions over frozen state)
    needs no evidence; empirical determinism does."""
    for entry in _spec_actions():
        if "remote" in entry["name"] and entry["determinism"] == "deterministic":
            assert "determinism_evidence" in entry, (
                f"{entry['name']} claims determinism across a network boundary "
                "without citing a measurement"
            )
            ev = entry["determinism_evidence"]
            for key in ("measured", "protocol", "result", "caveat"):
                assert key in ev and ev[key], f"{entry['name']}: missing {key}"


# --------------------------------------------------------------------------
# Trace
# --------------------------------------------------------------------------

def test_trace_digest_excludes_timing():
    @action("t.traced", determinism="deterministic", authority="authoritative")
    def traced(x):
        return ActionOutput(value=x, evidence=("n_abc",))

    t1, t2 = Trace(), Trace()
    _, j1 = traced(1)
    reset_replay_log()
    _, j2 = traced(1)
    t1.record(j1)
    t2.record(j2)
    assert j1.elapsed_ns != j2.elapsed_ns or True  # timings may coincide
    assert t1.digest() == t2.digest(), "trace digest must not depend on timing"


# --------------------------------------------------------------------------
# Spec files parse
# --------------------------------------------------------------------------

def test_every_spec_file_is_valid_json():
    """A malformed spec must fail as a test, not abort collection.

    `spec/invariants.json` is read at import time by `test_falsifiers.py`, so
    a stray missing comma there raises during collection and takes the entire
    suite down with an opaque JSONDecodeError. This test lives in a module
    with no such import, so it survives to report which file is broken and
    where.
    """
    for path in sorted((ROOT / "spec").glob("*.json")):
        with open(path) as f:
            try:
                json.load(f)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path.name}: {exc}") from exc
