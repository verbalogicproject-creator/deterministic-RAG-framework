"""M1.3 checkpoint: a strict total order, and graph expansion that earns its place.

The headline assertion - that every sort key is distinct - was **audited before
it was written**, because injectivity is guaranteed by construction once the
key ends in a content-addressed primary key, and an assertion that cannot fail
is worth nothing. Measured first: dropping `node_id` collides 66 candidates
across 7 of 15 queries. The falsifier is registered in `spec/invariants.json`.

Measured facts this module pins:

    266 nodes / 553 edges, directed cycles present, 5 nodes with no edges
    depth-2 reach   forward-only mean 7.2   bidirectional mean 23.3  (3.22x)
    nodes reaching nothing:  81/266 forward-only,  5/266 bidirectional
"""

import os
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drf.contract import reset_replay_log  # noqa: E402
from drf.ingest.build import build_index  # noqa: E402
from drf.retrieval import graph, stage1  # noqa: E402
from drf.retrieval.tokenize import tokenize  # noqa: E402
from drf.store import (  # noqa: E402
    connect,
    iter_edges,
    iter_nodes,
    neighbours,
    read_manifest,
)

# Override with DRF_SOURCE_DB. Hardcoding an absolute path would publish a
# username and make every test skip for anyone else who clones this.
SOURCE = os.environ.get(
    "DRF_SOURCE_DB",
    str(Path.home() / "Downloads/claude-cookbook-kg3/claude-cookbook-kg.db"),
)

requires_source = pytest.mark.skipif(
    not os.path.exists(SOURCE), reason=f"source corpus not present at {SOURCE}"
)

QUERIES = [
    "prompt caching", "tool use", "extended thinking", "rag retrieval",
    "agent orchestration", "streaming", "pdf vision", "citations",
    "batch processing", "embeddings semantic search",
    "json mode structured output", "summarization", "classification",
    "sub agents", "memory",
]

# Measured. Producer: the audit run recorded in spec/invariants.json.
MEASURED_ISOLATED_NODES = 5
MEASURED_FORWARD_ONLY_DEAD = 81


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    reset_replay_log()
    out = tmp_path_factory.mktemp("m13") / "index.db"
    build_index(source_path=SOURCE, out_path=str(out))
    conn = connect(str(out))
    yield {"conn": conn, "hash": read_manifest(conn)["content_hash"]}
    conn.close()


def _rank(index, text, **kwargs):
    value, justification = stage1.rank(
        conn=index["conn"], query_terms=tokenize(text),
        index_hash=index["hash"], **kwargs
    )
    return [stage1.Ranked(*row) for row in value], justification


# --------------------------------------------------------------------------
# The strict total order
# --------------------------------------------------------------------------

@requires_source
def test_sort_key_is_injective_on_every_query(index):
    """The checkpoint: len(set(sort_keys)) == len(D), for every query.

    Falsifiable - see spec/invariants.json::strict_total_order. Dropping the
    node_id component collides 66 candidates across 7 of these 15 queries.
    """
    for text in QUERIES:
        ranked, _ = _rank(index, text)
        if not ranked:
            continue
        keys = [stage1.sort_key(r) for r in ranked]
        assert len(set(keys)) == len(keys), f"non-injective sort key for {text!r}"
        assert stage1.is_strict_total_order(ranked)


@requires_source
def test_output_is_identical_across_input_shuffles(index):
    """A strict total order makes enumeration order irrelevant.

    Shuffling the scored candidates before ranking must not change one byte of
    the result. This is the property that sort *stability* would otherwise be
    silently providing - and stability is not guaranteed across
    implementations, only across CPython versions.
    """
    from drf.retrieval import bm25
    from drf.store import (
        corpus_totals, df_for_terms, doc_lengths, postings_for_terms,
    )

    conn = index["conn"]
    n_docs, total_len = corpus_totals(conn)
    doc_lens = doc_lengths(conn)

    for text in ("prompt caching", "tool use", "agent orchestration"):
        query = tokenize(text)
        scored = bm25.score_documents(
            postings=postings_for_terms(conn, query),
            dfs=df_for_terms(conn, query),
            doc_lens=doc_lens, n_docs=n_docs, total_len=total_len,
        )
        if not scored:
            continue
        seeds = stage1.select_seeds(scored, stage1.DEFAULT_SEED_COUNT)
        depths = graph.expand(conn, seeds, stage1.DEFAULT_MAX_DEPTH)
        baseline = stage1.rank_candidates(scored, depths)

        rng = random.Random(20260808)
        for _ in range(50):
            shuffled = scored[:]
            rng.shuffle(shuffled)
            assert stage1.rank_candidates(shuffled, depths) == baseline


@requires_source
def test_truncation_is_safe_at_every_boundary(index):
    """No two adjacent results are tied, so top-k cannot cascade.

    The prior engine's `[:15]` boundary reordered globally when one tie
    straddled it. With a strict order the k-th and (k+1)-th elements are
    always distinguishable, at every k.
    """
    for text in QUERIES:
        ranked, _ = _rank(index, text)
        keys = [stage1.sort_key(r) for r in ranked]
        for i in range(len(keys) - 1):
            assert keys[i] < keys[i + 1], f"{text!r}: tie at boundary {i}"


@requires_source
def test_sort_key_carries_no_floats(index):
    """Floats structurally cannot reach a comparison."""
    ranked, _ = _rank(index, "prompt caching")
    assert ranked
    for result in ranked:
        key = stage1.sort_key(result)
        assert len(key) == 5
        for component in key[:-1]:
            assert type(component) is int
        assert type(key[-1]) is str


# --------------------------------------------------------------------------
# Graph expansion
# --------------------------------------------------------------------------

@requires_source
def test_expansion_is_bidirectional(index):
    """Following outgoing edges only would strand a quarter of the corpus.

    Falsifiable - see spec/invariants.json::bidirectional_expansion.
    """
    conn = index["conn"]
    node_ids = [n.id for n in iter_nodes(conn)]

    forward_only = {e.from_id: set() for e in iter_edges(conn)}
    for edge in iter_edges(conn):
        forward_only.setdefault(edge.from_id, set()).add(edge.to_id)

    dead_forward = sum(
        1 for n in node_ids if not forward_only.get(n)
    )
    dead_both = sum(1 for n in node_ids if not neighbours(conn, n))

    assert dead_both == MEASURED_ISOLATED_NODES
    assert dead_forward == MEASURED_FORWARD_ONLY_DEAD
    assert dead_forward > dead_both, (
        "bidirectional expansion reaches no more than forward-only; the "
        "measured justification for it no longer holds"
    )


@requires_source
def test_expansion_is_independent_of_seed_order(index):
    """BFS assigns minimum depth, so seed order cannot matter."""
    conn = index["conn"]
    ranked, _ = _rank(index, "prompt caching")
    seeds = [r.node_id for r in ranked[:10]]
    baseline = graph.expand(conn, seeds, 2)

    rng = random.Random(20260808)
    for _ in range(20):
        shuffled = seeds[:]
        rng.shuffle(shuffled)
        assert graph.expand(conn, shuffled, 2) == baseline


@requires_source
def test_expansion_terminates_despite_cycles(index):
    """The graph has directed cycles; the visited set is load-bearing."""
    conn = index["conn"]
    node_ids = sorted(n.id for n in iter_nodes(conn))
    depths = graph.expand(conn, node_ids[:20], 6)
    assert depths
    assert all(isinstance(d, int) and d >= 0 for d in depths.values())


@requires_source
def test_depth_zero_reaches_only_the_seeds(index):
    conn = index["conn"]
    seeds = sorted(n.id for n in iter_nodes(conn))[:5]
    assert graph.expand(conn, seeds, 0) == {s: 0 for s in seeds}


@requires_source
def test_depth_is_monotone_in_max_depth(index):
    """Increasing depth may only add nodes, never change an existing depth."""
    conn = index["conn"]
    seeds = sorted(n.id for n in iter_nodes(conn))[:8]
    shallow = graph.expand(conn, seeds, 1)
    deep = graph.expand(conn, seeds, 3)
    assert set(shallow) <= set(deep)
    for node_id, depth in shallow.items():
        assert deep[node_id] == depth


def test_negative_depth_is_rejected(index=None):
    with pytest.raises(ValueError, match="max_depth"):
        graph.expand(None, ["x"], -1)


# --------------------------------------------------------------------------
# Parameters actually do something
# --------------------------------------------------------------------------

@requires_source
def test_seed_count_change_alters_internal_state_for_at_least_one_query(index):
    """`seed_count` 10 -> 11 must be observable in the ranked state.

    **Scope, corrected at M1.5.** This compares full `Ranked` tuples, so it
    passes on a change to `best_depth` alone. Measured: 10 -> 11 alters
    `best_depth` for 7 of 15 queries and reorders **zero** of them. The
    stronger claim - that the knob changes what a user sees - is tested in
    `test_config.py::test_every_ranking_setting_changes_real_query_output`,
    which compares node_id order and needs a 10 -> 20 probe to hold.

    The two tests disagreeing is what surfaced the distinction. Renamed so the
    name states which question it answers; the original read as though it
    proved visible sensitivity, which it does not.
    """
    changed = [
        text for text in QUERIES
        if _rank(index, text, seed_count=10)[0]
        != _rank(index, text, seed_count=11)[0]
    ]
    assert changed, "seed_count 10 -> 11 changed nothing for any query"


@requires_source
def test_max_depth_change_alters_output_for_at_least_one_query(index):
    changed = [
        text for text in QUERIES
        if _rank(index, text, max_depth=1)[0]
        != _rank(index, text, max_depth=3)[0]
    ]
    assert changed, "max_depth 1 -> 3 changed nothing for any query"


# --------------------------------------------------------------------------
# Graph signal is subordinate to lexical signal
# --------------------------------------------------------------------------

@requires_source
def test_graph_never_overrides_a_strictly_better_lexical_match(index):
    """No rank fusion: proximity breaks ties, it does not outrank scores.

    Reciprocal Rank Fusion - recommended in 5 of 9 playbooks in the recovered
    corpus - would blend the two into one score and allow exactly this
    inversion. Lexicographic composition makes it structurally impossible,
    which is why there is no weight to tune.
    """
    for text in QUERIES:
        ranked, _ = _rank(index, text)
        for earlier, later in zip(ranked, ranked[1:]):
            assert earlier.bm25_q >= later.bm25_q, (
                f"{text!r}: {later.node_id} outranked {earlier.node_id} "
                "despite a strictly lower BM25 score"
            )


@requires_source
def test_graph_resolves_ties_that_bm25_leaves_undefined(index):
    """The positive case: expansion does real work where scoring is silent."""
    resolved = 0
    for text in QUERIES:
        ranked, _ = _rank(index, text)
        for earlier, later in zip(ranked, ranked[1:]):
            if (earlier.bm25_q == later.bm25_q
                    and earlier.matched_terms == later.matched_terms
                    and earlier.best_depth != later.best_depth):
                resolved += 1
    assert resolved > 0, (
        "graph depth never decided an ordering; it contributes nothing "
        "measurable on this corpus"
    )


@requires_source
def test_d_contains_only_lexical_candidates(index):
    """Expansion supplies depth; it does not inject unmatched documents.

    Admitting graph-reached documents with no query term would be a relevance
    claim, and M1 carries no labels with which to justify one. Recorded as an
    M2 decision rather than taken silently here.
    """
    from drf.store import postings_for_terms

    conn = index["conn"]
    for text in ("prompt caching", "tool use"):
        query = tokenize(text)
        union: set[str] = set()
        for term_postings in postings_for_terms(conn, query).values():
            union.update(term_postings)
        ranked, _ = _rank(index, text)
        assert {r.node_id for r in ranked} == union


@requires_source
def test_rank_justification_declares_the_right_axes(index):
    _, justification = _rank(index, "prompt caching")
    assert justification.action == "stage1.rank"
    assert justification.determinism == "deterministic"
    assert justification.authority == "authoritative"
    assert justification.confidence is None
    assert justification.evidence
