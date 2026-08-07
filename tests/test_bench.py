"""M1.6 checkpoint: the reproducibility suite, and proof that it can fail.

Every metric here scores perfectly on this pipeline. That is precisely why
perfect scores are not the evidence - a harness that compared nothing would
report the same. The evidence is the **chaos control**: the identical
measurement applied to the defect this framework replaced (rank by score
alone, no tiebreak, unordered candidates - `python_apps_hybrid_query.py:304`),
which must score strictly worse.

The control also produced a result worth keeping, measured on this corpus:

    metric                  real     chaos     separates?
    distinct_digests           1         5     yes
    mismatched_positions       0       551     yes
    discordant_pairs           0     3,931     yes
    exact_match_rate      1.0000    0.6196     yes
    kendall_tau           1.0000    0.9761     barely
    rbo                   1.0000    0.9942     barely
    jaccard               1.0000    1.0000     NO
    overlap_coefficient   1.0000    1.0000     NO

Set-based metrics are **blind** to ordering non-determinism: the chaos
pipeline returns five different orderings in five runs and Jaccard reports
1.0000 for all of them. And a Kendall's Tau of 0.976 - which rounds to 0.98 -
describes a pipeline that is provably non-deterministic. This is the
measured justification for the project's rule that assertions use exact
integers and floats are for humans only.
"""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drf.bench import metrics, repro  # noqa: E402
from drf.contract import reset_replay_log  # noqa: E402
from drf.ingest.build import build_index  # noqa: E402

SOURCE = "/home/eyaln/Downloads/claude-cookbook-kg3/claude-cookbook-kg.db"

requires_source = pytest.mark.skipif(
    not os.path.exists(SOURCE), reason=f"source corpus not present at {SOURCE}"
)


@pytest.fixture(scope="module")
def two_builds(tmp_path_factory):
    """Two independently built indexes from the same source."""
    reset_replay_log()
    directory = tmp_path_factory.mktemp("m16")
    paths = []
    for name in ("a.db", "b.db"):
        out = directory / name
        build_index(source_path=SOURCE, out_path=str(out))
        paths.append(str(out))
    return paths


# --------------------------------------------------------------------------
# Metric unit behaviour
# --------------------------------------------------------------------------

def test_identical_lists_score_perfectly_on_every_metric():
    comparison = metrics.compare(list("abcdef"), list("abcdef"))
    assert comparison.identical
    assert comparison.mismatched_positions == 0
    assert comparison.discordant_pairs == 0
    assert comparison.exact_match == 1.0
    assert comparison.jaccard == 1.0
    assert comparison.kendall_tau == 1.0
    assert round(comparison.rbo, 9) == 1.0


def test_rbo_reaches_one_for_identical_lists_of_any_length():
    """Regression for a real defect, and a lesson about asserting on floats.

    Un-normalised RBO is bounded by `1 - p**k`, so identical short lists
    scored far below 1 - 0.716 on the real pipeline against 0.712 for the
    broken control. The metric could not separate perfect from broken, which
    is worse than not having it. Normalising by `1 - p**k` fixed that.

    The assertion is `round(..., 9) == 1.0`, not `== 1.0`, and the reason is
    the point of the whole discipline: the normalised division returns
    1.0000000000000002 at some lengths and 0.9999999999999998 at others. An
    exact float assertion here fails in *both* directions for inputs that are
    byte-identical. Floats are for reading; the integer surface below is what
    actually gets asserted.
    """
    for length in (1, 2, 3, 5, 10, 40):
        items = [f"n{i}" for i in range(length)]
        comparison = metrics.compare(items, items)
        # The assertion surface: exact, and exactly zero.
        assert comparison.identical
        assert comparison.discordant_pairs == 0
        assert comparison.mismatched_positions == 0
        # The human-facing number: correct to display precision, no further.
        assert round(comparison.rbo, 9) == 1.0
        assert comparison.rbo <= 1.0, "RBO is bounded by 1 by definition"


def test_reordering_is_detected_by_order_sensitive_metrics():
    comparison = metrics.compare(list("abcde"), list("edcba"))
    assert comparison.discordant_pairs > 0
    assert comparison.mismatched_positions > 0
    assert comparison.rbo < 1.0
    # ...and invisible to the set-based ones.
    assert comparison.jaccard == 1.0
    assert comparison.overlap_coefficient == 1.0


def test_score_stability_returns_an_int():
    assert metrics.score_stability([1, 2, 3], [1, 2, 3]) == 0
    assert type(metrics.score_stability([1, 2, 3], [1, 2, 3])) is int
    assert metrics.score_stability([1, 2, 3], [1, 2, 9]) == 6


def test_length_difference_is_counted():
    comparison = metrics.compare(["a", "b"], ["a"])
    assert comparison.length_delta == 1
    assert comparison.symmetric_difference == 1
    assert not comparison.identical


# --------------------------------------------------------------------------
# The query set
# --------------------------------------------------------------------------

def test_query_file_parses_and_covers_edge_cases():
    queries = repro.load_queries()
    assert len(queries) >= 20
    ids = {q["id"] for q in queries}
    assert len(ids) == len(queries), "duplicate query ids"
    edge = [q for q in queries if q["id"].startswith("e")]
    assert len(edge) >= 6, "edge cases are the point of a fixed query set"
    texts = {q["text"] for q in queries}
    assert "" in texts, "empty query must be exercised"
    assert any(len(t.split()) > 10 for t in texts), "long query must be exercised"


# --------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------

@requires_source
def test_reproducibility_matrix_is_perfect(two_builds):
    """Every axis, every cell, exact integers.

    Reduced repeat counts keep the suite fast; `drf bench repro` runs the
    full 5 x 3 x 3 x 2 matrix.
    """
    summary = repro.run_matrix(
        two_builds, in_process_repeats=3, subprocess_repeats=1
    )
    assert summary["distinct_digests"] == 1
    assert summary["mismatched_positions"] == 0
    assert summary["discordant_pairs"] == 0
    assert summary["length_delta"] == 0
    assert summary["symmetric_difference"] == 0
    assert summary["identical"] == summary["comparisons"]
    assert summary["cells"] >= 12


@requires_source
def test_two_independent_builds_agree(two_builds):
    """The rebuild axis on its own, so a failure localises."""
    first, second = two_builds
    queries = repro.load_queries()
    digests = set()
    for path in (first, second):
        from drf.store import connect, read_manifest
        conn = connect(path)
        digests.add(
            repro.digest(
                repro.run_all(conn, read_manifest(conn)["content_hash"], queries)
            )
        )
        conn.close()
    assert len(digests) == 1


# --------------------------------------------------------------------------
# The control that makes the perfect scores mean something
# --------------------------------------------------------------------------

@requires_source
def test_chaos_control_is_detected(two_builds):
    """The harness must be able to report failure.

    Falsifiable - spec/invariants.json::bench_detects_nondeterminism.
    """
    chaos = repro.run_chaos_control(two_builds[0], runs=5)
    assert chaos["distinct_digests"] > 1, (
        "the chaos control produced identical output; the harness cannot "
        "distinguish a broken pipeline from a correct one"
    )
    assert chaos["discordant_pairs"] > 0
    assert chaos["mismatched_positions"] > 0
    assert chaos["exact_match_rate"] < 1.0


@requires_source
def test_set_metrics_are_blind_to_ordering_nondeterminism(two_builds):
    """A recorded finding, pinned so it cannot be quietly forgotten.

    Jaccard and the overlap coefficient report 1.0000 for a pipeline that
    returns five different orderings in five runs. Anyone citing them as
    evidence of reproducibility would be citing nothing. Pinned here so that
    the limitation is discovered by reading the suite rather than by trusting
    a number.
    """
    chaos = repro.run_chaos_control(two_builds[0], runs=5)
    assert chaos["distinct_digests"] > 1
    assert chaos["jaccard"] == 1.0
    assert chaos["overlap_coefficient"] == 1.0


@requires_source
def test_float_metrics_look_nearly_perfect_when_the_pipeline_is_broken(two_builds):
    """The measured case for exact-integer assertions.

    Kendall's Tau above 0.97 and RBO above 0.99 for a pipeline that is
    provably non-deterministic. Rounded for a report they read as 0.98 and
    0.99 - indistinguishable from correct. The integers are not.
    """
    chaos = repro.run_chaos_control(two_builds[0], runs=5)
    assert chaos["kendall_tau"] > 0.9
    assert chaos["rbo"] > 0.9
    assert chaos["discordant_pairs"] > 100, (
        "the integers must show clearly what the floats obscure"
    )


# --------------------------------------------------------------------------
# Sensitivity
# --------------------------------------------------------------------------

@requires_source
def test_every_ranking_setting_is_live(two_builds):
    """Each advertised knob reorders at least one query, using its spec probe."""
    report = repro.run_sensitivity(two_builds[0])
    assert report["settings"]
    dead = [k for k, v in report["settings"].items() if not v["live"]]
    assert dead == [], f"settings declared to affect ranking but inert: {dead}"


@requires_source
def test_sensitivity_records_how_weak_the_graph_settings_are(two_builds):
    """Honest reporting: graph parameters reorder far less than lexical ones.

    Not a failure - a measurement. It says graph expansion contributes little
    to *ordering* on a 266-node corpus, which is worth knowing before anyone
    concludes the graph layer is carrying the results.
    """
    report = repro.run_sensitivity(two_builds[0])
    lexical = report["settings"]["ranking.b"]["queries_reordered"]
    graph = report["settings"]["graph.seed_count"]["queries_reordered"]
    assert lexical > graph, (
        "graph settings now reorder more than lexical ones; the recorded "
        "finding in spec/config_schema.json is stale"
    )
