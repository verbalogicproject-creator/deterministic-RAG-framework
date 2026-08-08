"""M2.0 checkpoint: the quality harness, tested before any label exists.

The circularity this milestone exists to break: an evaluation harness is
normally validated by running it on labelled data, which makes "is the harness
right?" and "is the system good?" one experiment - so when the answer
disappoints there is no way to tell which failed. Building the instrument
first is only worth doing if the instrument can be checked without the data.

It can, three ways, and all three are here:

1. **A hand-computed reference.** nDCG derived on paper for a five-document
   example, compared against a literal. The M1.2 BM25 fixture, again.
2. **Properties true of any label set.** The oracle sorts the system's own
   candidates by grade, so it cannot score below the system - for *every*
   label set, including synthetic ones. A reversal is a permutation, so it
   cannot change a retrieved set. Both catch real arithmetic bugs today.
3. **The structural bound.** `advisory_horizon` needs no labels at all.

What is deliberately absent: any assertion that this system retrieves well.
There are no relevance judgements yet, so there is no such fact to assert, and
inventing one is exactly the drift this project exists to prevent.
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drf.bench import controls, evaluate, labels as labels_module, quality  # noqa: E402
from drf.bench.repro import load_queries  # noqa: E402
from drf.contract import reset_replay_log  # noqa: E402
from drf.ingest.build import build_index  # noqa: E402

SOURCE = os.environ.get(
    "DRF_SOURCE_DB",
    str(Path.home() / "Downloads/claude-cookbook-kg3/claude-cookbook-kg.db"),
)
requires_source = pytest.mark.skipif(
    not os.path.exists(SOURCE), reason=f"source corpus not present at {SOURCE}"
)


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    reset_replay_log()
    out = tmp_path_factory.mktemp("m20") / "index.db"
    build_index(source_path=SOURCE, out_path=str(out))
    return str(out)


# --------------------------------------------------------------------------
# The hand-computed reference
# --------------------------------------------------------------------------
#
# Five documents, graded:  d1=3  d2=0  d3=2  d4=1  d5=0
# Gains (2**g - 1):        d1=7  d2=0  d3=3  d4=1  d5=0
# Ranking under test:      [d3, d1, d5, d4, d2]
#
#   DCG@5  = 3/log2(2) + 7/log2(3) + 0/log2(4) + 1/log2(5) + 0/log2(6)
#          = 3.0000000 + 4.4165085 + 0 + 0.4306766 + 0
#          = 7.8471851
#   ideal  = [d1, d3, d4, d2, d5]      (by -grade, then id)
#   IDCG@5 = 7/log2(2) + 3/log2(3) + 1/log2(4) + 0 + 0
#          = 7.0000000 + 1.8927892 + 0.5000000
#          = 9.3927892
#   nDCG@5 = 7.8471851 / 9.3927892 = 0.8354478
#
# Reversed: [d2, d4, d5, d1, d3]
#   DCG@5  = 0 + 1/log2(3) + 0 + 7/log2(5) + 3/log2(6)
#          = 0.6309298 + 3.0147360 + 1.1605580 = 4.8062238
#   nDCG@5 = 4.8062238 / 9.3927892 = 0.5116930
#
# Derived by hand first, then checked. The literals below are the hand values.

REFERENCE_LABELS = {"d1": 3, "d2": 0, "d3": 2, "d4": 1, "d5": 0}
REFERENCE_RANKING = ["d3", "d1", "d5", "d4", "d2"]
REFERENCE_NDCG = 0.8354478
REFERENCE_NDCG_REVERSED = 0.5116930


def test_gain_is_exponential_and_integral():
    assert [quality.gain(g) for g in (0, 1, 2, 3)] == [0, 1, 3, 7]
    assert all(isinstance(quality.gain(g), int) for g in (0, 1, 2, 3))


def test_gain_rejects_a_grade_outside_the_scale():
    with pytest.raises(ValueError):
        quality.gain(4)


def test_ndcg_matches_the_hand_computed_reference():
    judged = quality.judge(REFERENCE_RANKING, REFERENCE_LABELS, depth=5)
    assert round(judged.ndcg, 7) == REFERENCE_NDCG


def test_ndcg_of_the_reversed_reference_matches_the_hand_computation():
    reversed_ranking = controls.reverse(REFERENCE_RANKING, REFERENCE_LABELS)
    judged = quality.judge(reversed_ranking, REFERENCE_LABELS, depth=5)
    assert round(judged.ndcg, 7) == REFERENCE_NDCG_REVERSED


def test_the_integer_surface_of_the_reference_is_exact():
    """No rounding, no tolerance. This is what assertions are made of."""
    judged = quality.judge(REFERENCE_RANKING, REFERENCE_LABELS, depth=5)
    assert judged.relevant_total == 2          # d1 (3) and d3 (2), threshold 2
    assert judged.relevant_retrieved == 2
    assert judged.ranks_of_relevant == (1, 2)  # d3 first, d1 second
    assert judged.first_relevant_rank == 1
    assert judged.gain_total == 11             # 3 + 7 + 0 + 1 + 0


def test_ideal_order_is_injective_not_grade_only():
    """Grade alone is not a total order, and IDCG is every nDCG's denominator."""
    ties = {"z": 2, "a": 2, "m": 2}
    assert quality.ideal_order(ties) == ["a", "m", "z"]


# --------------------------------------------------------------------------
# Ordering: the blindness M1 measured, caught here in integers
# --------------------------------------------------------------------------

def test_recall_cannot_tell_a_reversal_from_the_real_ranking():
    """The finding, made mechanical rather than remembered.

    Milestone 1 measured that Jaccard reports 1.0000 for five different
    orderings. Recall@k inherits the blindness exactly. This test asserts the
    blindness *exists* so that no future reader mistakes recall for an
    ordering metric.
    """
    system = quality.judge(REFERENCE_RANKING, REFERENCE_LABELS, depth=5)
    reversed_ranking = controls.reverse(REFERENCE_RANKING, REFERENCE_LABELS)
    control = quality.judge(reversed_ranking, REFERENCE_LABELS, depth=5)
    assert control.recall == system.recall
    assert control.relevant_retrieved == system.relevant_retrieved


def test_ranks_of_relevant_does_tell_them_apart_and_does_it_in_integers():
    """The M2 analogue of `discordant_pairs`."""
    system = quality.judge(REFERENCE_RANKING, REFERENCE_LABELS, depth=5)
    reversed_ranking = controls.reverse(REFERENCE_RANKING, REFERENCE_LABELS)
    control = quality.judge(reversed_ranking, REFERENCE_LABELS, depth=5)
    assert system.ranks_of_relevant == (1, 2)
    assert control.ranks_of_relevant == (4, 5)
    assert system.ranks_of_relevant != control.ranks_of_relevant


def test_ndcg_ranks_the_controls_in_the_expected_order():
    """oracle >= system > reverse. The middle inequality is what M2 must earn."""
    ordered = {
        name: quality.judge(
            control(REFERENCE_RANKING, REFERENCE_LABELS),
            REFERENCE_LABELS, depth=5,
        ).ndcg
        for name, control in controls.CONTROLS.items()
    }
    system = quality.judge(REFERENCE_RANKING, REFERENCE_LABELS, depth=5).ndcg
    assert ordered["oracle"] >= system > ordered["reverse"]


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------

def test_oracle_is_optimal_over_the_same_candidates():
    ranking = list(REFERENCE_LABELS)
    ideal = controls.oracle(ranking, REFERENCE_LABELS)
    assert ideal[0] == "d1" and ideal[1] == "d3"
    assert quality.judge(ideal, REFERENCE_LABELS, depth=5).ndcg == 1.0


def test_id_order_is_deterministic_and_relevance_blind():
    """The counter-example: perfectly reproducible, entirely meaningless."""
    a = controls.id_order(REFERENCE_RANKING, REFERENCE_LABELS)
    b = controls.id_order(list(reversed(REFERENCE_RANKING)), REFERENCE_LABELS)
    assert a == b == ["d1", "d2", "d3", "d4", "d5"]


def test_every_control_returns_a_permutation_of_its_input():
    for name, control in controls.all_controls().items():
        out = control(REFERENCE_RANKING, REFERENCE_LABELS)
        assert sorted(out) == sorted(REFERENCE_RANKING), name


def test_shuffle_is_seeded_so_a_control_result_is_reproducible():
    first = controls.shuffle(REFERENCE_RANKING, REFERENCE_LABELS, seed=3)
    second = controls.shuffle(REFERENCE_RANKING, REFERENCE_LABELS, seed=3)
    assert first == second
    assert controls.shuffle(REFERENCE_RANKING, REFERENCE_LABELS, seed=4) != first


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

def test_divergent_judgements_for_the_same_pair_are_an_error():
    """Identical is fine, divergent is an error - the ingest collapse rule."""
    parsed = labels_module.parse([
        json.dumps({"query_id": "q01", "node_id": "abc", "grade": 3}),
        json.dumps({"query_id": "q01", "node_id": "abc", "grade": 1}),
    ])
    with pytest.raises(labels_module.LabelError, match="conflicting"):
        labels_module.collate(parsed)


def test_a_repeated_identical_judgement_is_accepted():
    parsed = labels_module.parse([
        json.dumps({"query_id": "q01", "node_id": "abc", "grade": 3}),
        json.dumps({"query_id": "q01", "node_id": "abc", "grade": 3}),
    ])
    assert labels_module.collate(parsed).count == 1


def test_an_unknown_node_id_is_an_error_not_a_dropped_row():
    """A skipped label shrinks the denominator, which RAISES measured recall."""
    parsed = labels_module.parse([
        json.dumps({"query_id": "q01", "node_id": "typo", "grade": 3}),
    ])
    with pytest.raises(labels_module.LabelError, match="not in the index"):
        labels_module.collate(parsed, known_node_ids={"real"})


def test_an_out_of_scale_grade_is_rejected_with_its_line_number():
    with pytest.raises(labels_module.LabelError, match="line 1"):
        labels_module.parse([json.dumps(
            {"query_id": "q01", "node_id": "abc", "grade": 9}
        )])


def test_a_boolean_is_not_accepted_as_a_grade():
    """`True == 1` in Python, so a bool would silently become a valid grade."""
    with pytest.raises(labels_module.LabelError, match="must be an int"):
        labels_module.parse([json.dumps(
            {"query_id": "q01", "node_id": "abc", "grade": True}
        )])


def test_labels_hash_ignores_file_order_but_not_content():
    """Commutativity again: identity depends on content, never on arrival order."""
    a = json.dumps({"query_id": "q01", "node_id": "n1", "grade": 3})
    b = json.dumps({"query_id": "q01", "node_id": "n2", "grade": 1})
    forward = labels_module.collate(labels_module.parse([a, b]))
    backward = labels_module.collate(labels_module.parse([b, a]))
    assert forward.labels_hash == backward.labels_hash

    changed = labels_module.collate(labels_module.parse(
        [a, json.dumps({"query_id": "q01", "node_id": "n2", "grade": 2})]
    ))
    assert changed.labels_hash != forward.labels_hash


# --------------------------------------------------------------------------
# Harness self-checks against the real index, on synthetic labels
# --------------------------------------------------------------------------

def _synthetic_labels(ranking, *, seed):
    """Grades assigned by position hash - deliberately unrelated to relevance.

    Nonsense labels are the right input here. These properties must hold for
    *any* label set, so labels that correlate with the ranking would make the
    test easier to pass and prove less.
    """
    return {node_id: (i * 7 + seed) % 4 for i, node_id in enumerate(ranking)}


@requires_source
def test_oracle_never_scores_below_the_system_on_real_rankings(index):
    """Holds for every label set. A violation is an arithmetic bug."""
    from drf.store import connect, read_manifest

    conn = connect(index)
    index_hash = read_manifest(conn)["content_hash"]
    for seed, query in enumerate(load_queries()):
        ranking = evaluate.rank_ids(conn, index_hash, query["text"])
        if not ranking:
            continue
        labels = _synthetic_labels(ranking, seed=seed)
        # `evaluate_query` runs `_self_check`, which raises HarnessError.
        evaluate.evaluate_query(ranking, labels, depths=[1, 5, 10, len(ranking)])
    conn.close()


@requires_source
def test_the_self_check_can_actually_fire(index):
    """The control for the control. An assertion that never fails is decorative.

    Milestone 1's lesson stated in one test: every "X does not happen" needs a
    sibling proving X *can* happen. Here the oracle is sabotaged so that it is
    no longer optimal, and the harness must notice.
    """
    from drf.store import connect, read_manifest

    conn = connect(index)
    index_hash = read_manifest(conn)["content_hash"]
    ranking = evaluate.rank_ids(conn, index_hash, "prompt caching")
    conn.close()
    labels = _synthetic_labels(ranking, seed=1)

    original = controls.CONTROLS["oracle"]
    controls.CONTROLS["oracle"] = lambda r, l: list(reversed(controls.oracle(r, l)))
    try:
        with pytest.raises(evaluate.HarnessError):
            evaluate.evaluate_query(ranking, labels, depths=[10])
    finally:
        controls.CONTROLS["oracle"] = original


# --------------------------------------------------------------------------
# The structural bound - needs no labels at all
# --------------------------------------------------------------------------

@requires_source
def test_the_advisory_layer_cannot_reach_below_the_horizon(index):
    """Subordination observed from outside, per query, as an integer.

    This is the number that stops M2.1 reporting "the neural layer did not
    improve nDCG@5" as a finding when it is a structural impossibility.
    """
    report = evaluate.run_advisory_invariance(index, load_queries())
    assert report["prefixes_differing"] == 0
    assert report["prefixes_identical"] == report["queries"]


@requires_source
def test_advisory_reach_is_inverse_to_lexical_success(index):
    """The bound is per query, and that turns out to be the interesting part.

    Measured on this corpus, |D| runs from 0 to 147. Where stage 1 found 20 or
    more documents the advisory layer is structurally silent at every
    evaluated depth; where it found one or two, the advisory layer can act
    from depth 5 down.

    So the neural layer can only speak **where lexical retrieval did badly**,
    and is provably mute where it did well. That is not a limitation of the
    provider - it is what append-only subordination *means*, stated as a
    measurement. It also fixes what M2.1 may ask: a quality comparison at any
    depth <= min(|D|) is guaranteed to show zero difference, and reporting
    that as a finding about neural retrieval would be a category error.
    """
    depths = [1, 5, 10, 20]
    report = evaluate.run_advisory_invariance(index, load_queries(), depths=depths)
    by_horizon = {r["query_id"]: r for r in report["rows"]}

    for row in report["rows"]:
        expected = [d for d in depths if d > row["horizon"]]
        assert row["depths_advisory_can_reach"] == expected, row["query_id"]

    well_served = [r for r in report["rows"] if r["horizon"] >= max(depths)]
    assert well_served, "no query has |D| >= 20; the bound cannot be shown"
    for row in well_served:
        assert row["depths_advisory_can_reach"] == [], (
            f"{row['query_id']} has |D|={row['horizon']} yet reports a "
            f"reachable depth - the horizon is computed wrongly"
        )
    assert by_horizon["q02"]["horizon"] > by_horizon["q08"]["horizon"]


@requires_source
def test_a_reachable_horizon_does_not_mean_the_advisory_layer_can_help(index):
    """Necessary but not sufficient - and the gap is measurable.

    The three out-of-vocabulary queries have |D| = 0, so by the horizon alone
    the advisory layer could occupy every position. It proposes nothing, because
    anchor-mode search takes its anchors *from D*, and D is empty.

    This is the M1 scope limit ("an all-OOV query yields empty D and therefore
    no proposals") observed rather than asserted. It matters for M2 because a
    recall figure on those queries measures anchor starvation, not the
    provider - and the fix, if one is wanted, is a stage 1 fix such as
    character n-grams, never a neural one.
    """
    report = evaluate.run_advisory_invariance(index, load_queries())
    starved = [
        r for r in report["rows"]
        if r["horizon"] == 0 and r["tail_lengths"]["stored_vectors"] == 0
    ]
    assert starved, "expected the OOV queries to produce no proposals"
    for row in starved:
        assert row["depths_advisory_can_reach"], (
            "these queries are unreachable by horizon AND unreachable in "
            "practice; the test would then prove nothing"
        )


# --------------------------------------------------------------------------
# The spec is the source of truth for the thresholds
# --------------------------------------------------------------------------

def test_evaluation_spec_declares_its_margin_and_says_it_was_declared_first():
    spec = json.loads((ROOT / "spec" / "evaluation.json").read_text())
    assert isinstance(spec["min_ndcg_margin"], float)
    assert spec["declared"] == "2026-08-08"
    # This asserted `"no labels yet" in spec["status"]` until 2026-08-08. When
    # labels arrived the sentence became false - and the test *held it in
    # place*, which is the worst shape available: a green suite enforcing a
    # claim that had stopped being true. Found only by an audit re-reading spec
    # prose against the filesystem. What is durable is that the margin was
    # declared before any judgement existed, so that is what is asserted now.
    assert "declared 2026-08-08" in spec["min_ndcg_margin_note"].lower()
    assert "before any label existed" in spec["min_ndcg_margin_note"].lower()


def test_spec_grades_match_the_implementation():
    """Spec and code must not drift - the same bijection rule as actions."""
    spec = json.loads((ROOT / "spec" / "evaluation.json").read_text())
    assert {int(k) for k in spec["relevance_scale"] if k.isdigit()} == set(quality.GRADES)
    assert spec["relevance_scale"]["threshold_for_recall_and_precision"] == \
        quality.RELEVANCE_THRESHOLD
    assert {int(k): v for k, v in spec["gain_function"]["values"].items()} == \
        {g: quality.gain(g) for g in quality.GRADES}


#: A key that *is* a quality metric, optionally at a depth. `min_ndcg_margin`
#: deliberately does not match: it is a declared threshold, not a result.
QUALITY_METRIC_KEY = re.compile(r"^(ndcg|recall|precision|mrr|map)(@\d+)?$")


def _numeric_metric_keys(node, path="") -> list[str]:
    """Walk a JSON document for a metric name bound to a number."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if QUALITY_METRIC_KEY.match(str(key).lower()) and isinstance(
                value, (int, float)
            ) and not isinstance(value, bool):
                found.append(here)
            found.extend(_numeric_metric_keys(value, here))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            found.extend(_numeric_metric_keys(item, f"{path}[{i}]"))
    return found


def test_every_published_quality_figure_carries_its_provenance():
    """A quality number may exist - but only bound to the labels that made it.

    Guards the exact drift this project was recovered from: a number appearing
    in documentation without a producer.

    **The premise changed on 2026-08-08.** This test used to assert that *no*
    quality figure existed anywhere, which was true until labels did. A guard
    whose premise expires must be re-aimed, not deleted, so it now enforces the
    rule that was always the real one: a quality figure must sit in a block
    carrying both a `producer` command and a `labels_hash`. Edit one grade and
    every nDCG changes with no commit touching any code - so a figure without a
    label hash is a figure without a producer.

    Scans **structure, not prose**. An earlier version matched the substring
    `ndcg@` and failed on the spec's own worked example, a sentence explaining
    why a bare figure is not evidence.
    """
    for path in sorted((ROOT / "spec").glob("*.json")):
        document = json.loads(path.read_text())
        for offender in _numeric_metric_keys(document):
            block = document[offender.split(".")[0]]
            assert isinstance(block, dict), f"{path.name}: {offender} is unanchored"
            assert block.get("producer"), (
                f"{path.name}: {offender} published with no producer command"
            )
            assert block.get("labels_hash"), (
                f"{path.name}: {offender} is a quality figure with no "
                f"labels_hash; it cannot be reproduced or falsified"
            )


def test_the_declared_margin_was_not_revised_to_convert_a_failure():
    """The single most corrosive edit possible here, blocked mechanically.

    `spec/evaluation.json` fixed `min_ndcg_margin` at 0.05 on 2026-08-08 before
    any judgement existed. The recorded rows are checked against the **live**
    declared value rather than a number copied beside them, so lowering the
    threshold to turn the recorded FAIL into a PASS fails this test instead.
    """
    evaluation = json.loads((ROOT / "spec" / "evaluation.json").read_text())
    benchmarks = json.loads((ROOT / "spec" / "benchmarks.json").read_text())
    block = benchmarks.get("retrieval_quality")
    if block is None:
        pytest.skip("no quality measurement recorded yet")

    declared = evaluation["min_ndcg_margin"]
    assert declared == 0.05, (
        f"min_ndcg_margin is {declared}, not the 0.05 declared on 2026-08-08. "
        f"If raised deliberately, record it as a change of standard."
    )
    for row in block["rows"]:
        assert row["required"] == declared
        assert row["clears"] == (row["margin"] >= declared), (
            f"depth {row['depth']} records clears={row['clears']} for margin "
            f"{row['margin']} against a declared {declared}"
        )


def test_the_quality_result_records_that_its_labels_are_model_generated():
    """Provenance is part of the result, not a footnote.

    The annotator also authored the retrieval system. Blind grading removes
    position bias; it does not remove authorship bias, and a reader must be
    told before reading any figure.
    """
    benchmarks = json.loads((ROOT / "spec" / "benchmarks.json").read_text())
    block = benchmarks.get("retrieval_quality")
    if block is None:
        pytest.skip("no quality measurement recorded yet")
    assert "MODEL-GENERATED" in block["annotator"]
    assert "NOT human" in block["annotator"]


# --------------------------------------------------------------------------
# The labelling worksheet - its selection must stay true, not just present
# --------------------------------------------------------------------------

def _strata():
    sys.path.insert(0, str(ROOT / "tools"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_labelling_worksheet", ROOT / "tools" / "make_labelling_worksheet.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.STRATA


def test_every_stratum_names_queries_that_exist():
    known = {q["id"] for q in load_queries()}
    for stratum in _strata():
        missing = set(stratum["queries"]) - known
        assert not missing, f"{stratum['name']} names unknown queries {missing}"


@requires_source
def test_stratum_a_queries_are_actually_reachable_by_the_advisory_layer(index):
    """Asserts the *reason* for the selection, not merely that it exists.

    Stratum A exists because those queries have a small enough |D| that the
    evaluated depths fall beyond the horizon. A hand-written list of query ids
    goes stale silently when the query set changes - the same class of bug as
    the order-dependent registry test found in M1 - so the property that
    justified the selection is what gets checked, not the list.
    """
    from drf.store import connect, read_manifest

    stratum = next(s for s in _strata() if s["name"] == "A_advisory")
    conn = connect(index)
    index_hash = read_manifest(conn)["content_hash"]
    queries = {q["id"]: q for q in load_queries()}
    for query_id in stratum["queries"]:
        horizon = len(evaluate.rank_ids(conn, index_hash, queries[query_id]["text"]))
        assert 0 < horizon < 10, (
            f"{query_id} has |D|={horizon}; stratum A requires a horizon the "
            f"evaluated depths can exceed, and a non-empty D to anchor from"
        )
    conn.close()


@requires_source
def test_stratum_c_is_marked_as_unable_to_settle_its_question(index):
    """The graph decision has a real sample size of one, and says so.

    Measured: only q02 and e06 reorder under any graph setting, and e06 is a
    synthetic edge case. A stratum that cannot answer its question must not be
    presented as though it can.
    """
    stratum = next(s for s in _strata() if s["name"] == "C_graph")
    assert stratum["complete_on_its_own"] is False
    assert len(stratum["queries"]) == 1
    assert "cannot settle" in stratum["why_these"].lower() or \
        "sample size of ONE" in stratum["why_these"]


# --------------------------------------------------------------------------
# Structural integrity audit, 2026-08-08. Each test below closes a gap found
# by re-reading claims against the filesystem rather than against memory.
# --------------------------------------------------------------------------

def test_a_duplicate_in_a_ranking_is_rejected_not_counted_twice():
    """recall 2.0 and nDCG 1.63 were reachable before this guard.

    `judge(["a", "a"], {"a": 3}, depth=2)` counted the same document twice and
    returned metrics above their own bound, silently, in the direction that
    flatters the system. Stage 1 cannot emit a duplicate today - which is why
    this raises rather than de-duplicating. A metric is the last place that
    should paper over a malformed input.
    """
    with pytest.raises(ValueError, match="duplicate"):
        quality.judge(["a", "a"], {"a": 3}, depth=2)

    # The control: the same call without the duplicate must work, or the guard
    # could be rejecting everything and the test would still pass.
    assert quality.judge(["a"], {"a": 3}, depth=2).recall == 1.0


def test_no_metric_can_exceed_its_own_bound_on_any_stratum_a_query():
    """Swept over the real label set rather than argued about."""
    label_set = labels_module.load(ROOT / "queries" / "labels.jsonl")
    for query_id, labels in label_set.by_query.items():
        for depth in (1, 5, 10, 20):
            judged = quality.judge(sorted(labels), labels, depth=depth)
            assert 0.0 <= judged.recall <= 1.0, (query_id, depth)
            assert 0.0 <= judged.precision <= 1.0, (query_id, depth)
            assert 0.0 <= judged.ndcg <= 1.0, (query_id, depth)
            assert judged.relevant_retrieved <= judged.relevant_total


@requires_source
def test_recorded_quality_figures_match_a_live_run(index):
    """The figures in `spec/benchmarks.json` were transcribed by hand.

    Design rule 1 is *no number without a producer*, and the producer command
    is recorded - but the values were copied into the spec by a human hand,
    which is exactly the step at which the recovered project's numbers went
    wrong. This re-runs the producer and compares.

    Rounded to 4 decimals because that is the precision the spec records; the
    comparison is against what the tool emits, not against a remembered value.
    """
    benchmarks = json.loads((ROOT / "spec" / "benchmarks.json").read_text())
    block = benchmarks.get("retrieval_quality")
    if block is None:
        pytest.skip("no quality measurement recorded yet")

    from drf.store import connect, iter_nodes
    conn = connect(index)
    known = {n.id for n in iter_nodes(conn)}
    conn.close()
    label_set = labels_module.load(
        ROOT / "queries" / "labels.jsonl", known_node_ids=known
    )
    assert label_set.labels_hash == block["labels_hash"], (
        "the recorded figures were computed from a different label set"
    )
    assert label_set.count == block["judgements"]

    live = evaluate.run_quality(index, label_set, load_queries())
    for row in block["rows"]:
        computed = live["depths"][row["depth"]]
        assert round(computed["aggregate"]["system"]["ndcg"], 4) == row["system_ndcg"]
        assert round(computed["ndcg_margin_over_best_blind"], 4) == row["margin"]
        assert computed["best_blind_control"] == row["best_blind"]
        assert computed["clears_required_margin"] == row["clears"]


def test_the_worksheet_and_the_collected_labels_cannot_drift():
    """Two files hold the same judgements. Nothing forced them to agree.

    `labels_collect.py` derives `labels.jsonl` from the worksheet, but both are
    committed, so editing a grade in the worksheet and forgetting to re-collect
    leaves a stale label file - and every published figure is bound to the
    stale one's hash. A second source of truth with no reconciliation is the
    shape of defect this project was recovered from.
    """
    worksheet = [
        json.loads(line)
        for line in (ROOT / "queries" / "labels.worksheet.jsonl").read_text().splitlines()
        if line.strip()
    ]
    collected = {
        (r["query_id"], r["node_id"]): r["grade"]
        for r in (
            json.loads(line)
            for line in (ROOT / "queries" / "labels.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    graded = {
        (r["query_id"], r["node_id"]): r["grade"]
        for r in worksheet if r.get("grade") is not None
    }
    stale = {k: (graded[k], collected.get(k)) for k in graded if collected.get(k) != graded[k]}
    assert not stale, (
        f"{len(stale)} judgement(s) differ between the worksheet and "
        f"labels.jsonl. Re-run tools/labels_collect.py. Examples: "
        f"{dict(list(stale.items())[:3])}"
    )


def test_the_release_binds_the_labels_that_produced_its_figures():
    """A tag names an exact spec, index and result set - and now the labels.

    Without `labels_hash` in the freeze, a release publishes nDCG figures while
    naming nothing that fixes the judgements behind them. Editing one grade
    changes every figure with no commit touching any code, so the labels are
    as much a part of a reproducible release as the index is.
    """
    frozen = json.loads((ROOT / "spec" / "frozen.json").read_text())
    benchmarks = json.loads((ROOT / "spec" / "benchmarks.json").read_text())
    if "retrieval_quality" not in benchmarks:
        pytest.skip("no quality measurement recorded yet")
    assert "labels_hash" in frozen, (
        "spec/frozen.json does not bind the label set that produced the "
        "published quality figures"
    )
    assert frozen["labels_hash"] == benchmarks["retrieval_quality"]["labels_hash"]
