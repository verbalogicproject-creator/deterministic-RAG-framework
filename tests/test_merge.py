"""M1.4 checkpoint: subordination, proven against hostile providers.

This is the architecture's central claim, so it is tested the way a claim
should be - by trying to break it. Six providers appear below: one that
proposes nothing, one that works, and four that misbehave in the ways a real
provider actually fails. Every one of them must leave the authoritative prefix
byte-identical.

    merged[:len(D)] == D,  elementwise, in order, always.

Both headline assertions here are **falsifiable**, registered in
`spec/invariants.json` as `merge_is_append_only` and `advisory_allowlist`. The
first needed care: interleaving alone only trips merge's runtime
postcondition, which would prove the *postcondition* fires rather than the
test. The falsifier therefore neuters the guard first, so the bad merge
returns quietly and the assertion is the only thing left that can notice.
"""

import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drf.contract import Advisory, AuthorityViolation, reset_replay_log  # noqa: E402
from drf.ingest.build import build_index  # noqa: E402
from drf.retrieval import merge as merge_module  # noqa: E402
from drf.retrieval import neural, stage1  # noqa: E402
from drf.retrieval.providers.null import NullProvider  # noqa: E402
from drf.retrieval.providers.stored_vectors import StoredVectorProvider  # noqa: E402
from drf.retrieval.tokenize import tokenize  # noqa: E402
from drf.store import connect, iter_nodes, read_manifest  # noqa: E402

SOURCE = "/home/eyaln/Downloads/claude-cookbook-kg3/claude-cookbook-kg.db"

requires_source = pytest.mark.skipif(
    not os.path.exists(SOURCE), reason=f"source corpus not present at {SOURCE}"
)

QUERIES = ["prompt caching", "tool use", "extended thinking", "streaming",
           "agent orchestration", "batch processing"]


# --------------------------------------------------------------------------
# Hostile providers - the ways a real one fails
# --------------------------------------------------------------------------

class AdversarialProvider:
    """Actively tries to subvert the ranking.

    Proposes the authoritative results themselves, in reverse. If merge were
    naive it would append duplicates, and a document already at rank 0 would
    reappear lower down - or, worse, a promotion scheme would lift it. This is
    the provider that a genuinely malicious plugin would be.
    """
    name = "adversarial"

    def __init__(self, authoritative):
        self._authoritative = authoritative

    def propose(self, *, anchors, limit):
        return list(reversed(self._authoritative))


class CrashingProvider:
    name = "crashing"

    def propose(self, *, anchors, limit):
        raise RuntimeError("provider exploded")


class HangingProvider:
    """Never returns. The failure mode that takes a service down."""
    name = "hanging"

    def propose(self, *, anchors, limit):
        time.sleep(3600)
        return []


class FloodProvider:
    """Ignores `limit` entirely and returns ten thousand ids."""
    name = "flood"

    def propose(self, *, anchors, limit):
        return [f"n_flood_{i}" for i in range(10_000)]


class LyingProvider:
    """Returns well-formed ids for documents that do not exist."""
    name = "lying"

    def propose(self, *, anchors, limit):
        return [f"n_does_not_exist_{i}" for i in range(5)]


class JunkProvider:
    """Returns values that are not even strings."""
    name = "junk"

    def propose(self, *, anchors, limit):
        return [None, 42, "", b"bytes", ["nested"]]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def index(tmp_path_factory):
    reset_replay_log()
    out = tmp_path_factory.mktemp("m14") / "index.db"
    build_index(source_path=SOURCE, out_path=str(out))
    conn = connect(str(out))
    yield {
        "conn": conn,
        "hash": read_manifest(conn)["content_hash"],
        "known": {n.id for n in iter_nodes(conn)},
    }
    conn.close()


@pytest.fixture(autouse=True)
def fast_timeout(monkeypatch):
    """Keep the hanging-provider test quick without weakening it."""
    monkeypatch.setattr(neural, "PROVIDER_TIMEOUT_SECONDS", 0.2)


def _authoritative(index, text):
    value, _ = stage1.rank(
        conn=index["conn"], query_terms=tokenize(text), index_hash=index["hash"]
    )
    return [stage1.Ranked(*row).node_id for row in value]


def _run(index, text, provider):
    """Full stage1 -> neural -> merge pipeline for one provider."""
    deterministic = _authoritative(index, text)
    advisory, justification = neural.propose_from_anchors(
        provider=provider,
        anchors=deterministic[:5],
        limit=10,
        provider_name=provider.name,
        index_hash=index["hash"],
    )
    merged = merge_module.merge(
        deterministic=deterministic,
        advisory=advisory,
        known_ids=index["known"],
    )
    return deterministic, merged, justification


def _all_providers(index, deterministic):
    return [
        NullProvider(),
        StoredVectorProvider(index["conn"]),
        AdversarialProvider(deterministic),
        CrashingProvider(),
        HangingProvider(),
        FloodProvider(),
        LyingProvider(),
        JunkProvider(),
    ]


# --------------------------------------------------------------------------
# The central guarantee
# --------------------------------------------------------------------------

@requires_source
def test_advisory_never_disturbs_the_prefix(index):
    """The checkpoint. Every provider, every query, elementwise.

    Falsifiable - spec/invariants.json::merge_is_append_only.
    """
    for text in QUERIES:
        deterministic = _authoritative(index, text)
        if not deterministic:
            continue
        for provider in _all_providers(index, deterministic):
            _, merged, _ = _run(index, text, provider)
            prefix = [r.node_id for r in merged[:len(deterministic)]]
            assert prefix == deterministic, (
                f"{provider.name} disturbed the prefix for {text!r}"
            )
            assert all(
                r.origin == merge_module.AUTHORITATIVE
                for r in merged[:len(deterministic)]
            )


@requires_source
def test_prefix_is_byte_identical_across_every_provider(index):
    """`--neural off` and `--neural stored` must agree exactly on the prefix."""
    for text in QUERIES:
        deterministic = _authoritative(index, text)
        if not deterministic:
            continue
        prefixes = set()
        for provider in _all_providers(index, deterministic):
            _, merged, _ = _run(index, text, provider)
            prefixes.add(
                tuple(merge_module.deterministic_prefix_ids(merged))
            )
        assert len(prefixes) == 1, f"{text!r}: providers disagreed on the prefix"


@requires_source
def test_discordant_pairs_is_zero(index):
    """Exact integer, not a float correlation.

    A determinism suite that asserts `tau == 1.000` is asserting a rounding
    artefact. Counting inversions and requiring exactly zero says the thing
    itself.
    """
    total_discordant = 0
    for text in QUERIES:
        baseline = _authoritative(index, text)
        if not baseline:
            continue
        for provider in _all_providers(index, baseline):
            _, merged, _ = _run(index, text, provider)
            observed = merge_module.deterministic_prefix_ids(merged)
            position = {node_id: i for i, node_id in enumerate(observed)}
            for i, node_id in enumerate(baseline):
                if position.get(node_id) != i:
                    total_discordant += 1
    assert total_discordant == 0


@requires_source
def test_null_provider_output_equals_d_exactly(index):
    """The reference case: no advice means no change, not merely no reorder."""
    for text in QUERIES:
        deterministic, merged, _ = _run(index, text, NullProvider())
        assert [r.node_id for r in merged] == deterministic


# --------------------------------------------------------------------------
# Provider failure containment
# --------------------------------------------------------------------------

@requires_source
def test_crashing_provider_yields_empty_advice(index):
    _, merged, _ = _run(index, "prompt caching", CrashingProvider())
    assert all(r.origin == merge_module.AUTHORITATIVE for r in merged)


@requires_source
def test_hanging_provider_is_abandoned_not_awaited(index):
    """A provider that never returns must not hold the query."""
    started = time.perf_counter()
    _, merged, _ = _run(index, "prompt caching", HangingProvider())
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0, f"query waited {elapsed:.2f}s on a hanging provider"
    assert all(r.origin == merge_module.AUTHORITATIVE for r in merged)


@requires_source
def test_flood_provider_cannot_flood(index):
    """10,000 proposals must not become 10,000 results."""
    _, merged, _ = _run(index, "prompt caching", FloodProvider())
    advisory_count = sum(
        1 for r in merged if r.origin == merge_module.ADVISORY
    )
    assert advisory_count == 0, (
        "flooded ids do not exist in the index and must be dropped"
    )


@requires_source
def test_lying_provider_ids_are_dropped(index):
    _, merged, _ = _run(index, "prompt caching", LyingProvider())
    assert all(r.origin == merge_module.AUTHORITATIVE for r in merged)


@requires_source
def test_junk_provider_non_strings_are_discarded(index):
    _, merged, _ = _run(index, "prompt caching", JunkProvider())
    assert all(isinstance(r.node_id, str) for r in merged)


@requires_source
def test_adversarial_provider_cannot_duplicate_a_result(index):
    """Proposing an already-ranked document must not produce a second copy."""
    for text in QUERIES:
        deterministic = _authoritative(index, text)
        if not deterministic:
            continue
        _, merged, _ = _run(index, text, AdversarialProvider(deterministic))
        node_ids = [r.node_id for r in merged]
        assert len(node_ids) == len(set(node_ids))
        assert node_ids == deterministic


@requires_source
def test_stored_vectors_actually_proposes_something(index):
    """Control: the working provider must contribute, or every test above
    is passing because nothing ever reached the tail."""
    conn = index["conn"]
    provider = StoredVectorProvider(conn)
    contributed = 0
    for text in QUERIES:
        _, merged, _ = _run(index, text, provider)
        contributed += sum(
            1 for r in merged if r.origin == merge_module.ADVISORY
        )
    assert contributed > 0, (
        "no provider ever reached the tail; the subordination tests would "
        "pass vacuously"
    )


@requires_source
def test_stored_vector_provider_is_deterministic(index):
    """Frozen vectors, no model: identical proposals on every call."""
    provider = StoredVectorProvider(index["conn"])
    anchors = _authoritative(index, "prompt caching")[:5]
    runs = {
        tuple(provider.propose(anchors=anchors, limit=10)) for _ in range(5)
    }
    assert len(runs) == 1


# --------------------------------------------------------------------------
# The box
# --------------------------------------------------------------------------

@requires_source
def test_unwrap_outside_merge_is_refused(index):
    """Advisory data cannot be opened here, and this module is a test.

    Falsifiable - spec/invariants.json::advisory_allowlist.
    """
    advisory, _ = neural.propose_from_anchors(
        provider=NullProvider(), anchors=[], limit=5,
        provider_name="null", index_hash=index["hash"],
    )
    assert isinstance(advisory, Advisory)
    with pytest.raises(AuthorityViolation, match="may not unwrap"):
        advisory.unwrap()


@requires_source
def test_advisory_presence_is_visible_without_unwrapping(index):
    """`is_empty()` reveals presence, not contents - safe for anyone."""
    advisory, _ = neural.propose_from_anchors(
        provider=NullProvider(), anchors=[], limit=5,
        provider_name="null", index_hash=index["hash"],
    )
    assert advisory.is_empty() is True


@requires_source
def test_a_deterministic_action_can_still_be_advisory(index):
    """The two axes are independent - this is the case that proves it.

    Anchor-mode search over frozen vectors runs no model and touches no
    network, so it replays exactly. It is still forbidden from influencing
    ranking. Determinism is not a licence.
    """
    _, _, justification = _run(
        index, "prompt caching", StoredVectorProvider(index["conn"])
    )
    assert justification.determinism == "deterministic"
    assert justification.authority == "advisory"
    assert justification.confidence is None
    assert justification.provider == "stored_vectors"


# --------------------------------------------------------------------------
# The runtime postcondition itself
# --------------------------------------------------------------------------

def test_postcondition_rejects_a_reordered_prefix():
    """Direct test of the guard, with a hand-built bad merge."""
    bad = [
        merge_module.MergedResult("b", merge_module.AUTHORITATIVE, 0),
        merge_module.MergedResult("a", merge_module.AUTHORITATIVE, 1),
    ]
    with pytest.raises(merge_module.SubordinationViolation):
        merge_module._assert_subordination(bad, ["a", "b"])


def test_postcondition_rejects_advisory_inside_the_prefix():
    bad = [
        merge_module.MergedResult("a", merge_module.AUTHORITATIVE, 0),
        merge_module.MergedResult("b", merge_module.ADVISORY, 1),
    ]
    with pytest.raises(merge_module.SubordinationViolation, match="marked"):
        merge_module._assert_subordination(bad, ["a", "b"])


def test_postcondition_accepts_a_correct_merge():
    good = [
        merge_module.MergedResult("a", merge_module.AUTHORITATIVE, 0),
        merge_module.MergedResult("b", merge_module.AUTHORITATIVE, 1),
        merge_module.MergedResult("c", merge_module.ADVISORY, 2),
    ]
    merge_module._assert_subordination(good, ["a", "b"])
