"""M1.2 checkpoint: BM25 with real length normalisation, and no silent caps.

Two things this module is careful about.

**Every "X does not happen" test has a control proving X *can* happen.** The
padded-document test would pass trivially if the padded document simply never
outranked anything, so a sibling test sets `b=0` and requires the padded
document to *win*. Without that, the assertion would be describing an accident
rather than length normalisation.

**The reference values are exact integers, computed independently.** The 5-doc
figures below were derived from the formula written out longhand, not by
calling `score_documents`, then pinned as literals. An assertion computed by
the code under test cannot contradict it.
"""

import itertools
import json
import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drf.contract import reset_replay_log  # noqa: E402
from drf.fixed import QUANTUM_EXP  # noqa: E402
from drf.ingest.build import build_index  # noqa: E402
from drf.retrieval import bm25, lexical  # noqa: E402
from drf.retrieval.tokenize import (  # noqa: E402
    TOKEN_PATTERN,
    document_text,
    terms,
    tokenize,
)
from drf.store import (  # noqa: E402
    connect,
    corpus_totals,
    df_for_terms,
    doc_lengths,
    iter_nodes,
    iter_postings,
    postings_for_terms,
    read_manifest,
    table_count,
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

# Measured on the real corpus, b=0.75 vs b=0.0, 15 corpus-vocabulary queries.
# Producer: tools/measure_length_norm.py
#
# All of these are **tiebreak-free**. An earlier set of figures ("top-1 changed
# 9/15", "26.2% discordant pairs") turned out to depend on an unstated
# tiebreak: 7 of 15 queries have exact ties in a top-set, and breaking them by
# lowest node id gives 9 while highest node id gives 4. Those numbers described
# BM25 *plus a convention*, not BM25. The metrics below are defined only over
# strictly-ordered comparisons, so no convention can move them.
MEASURED_QUERIES = 15
MEASURED_TIED_TOPSET_QUERIES = 7    # queries where a top-set has >1 member
MEASURED_DISJOINT_TOPSETS = 3       # top-set changed unambiguously
MEASURED_DIRECTION_LONGER = 9       # b=0 top-set is longer on average
MEASURED_DIRECTION_SHORTER = 0      # the directional claim: never shorter


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    reset_replay_log()
    out = tmp_path_factory.mktemp("m12") / "index.db"
    build_index(source_path=SOURCE, out_path=str(out))
    conn = connect(str(out))
    yield {"conn": conn, "hash": read_manifest(conn)["content_hash"]}
    conn.close()


# --------------------------------------------------------------------------
# The 5-document reference
# --------------------------------------------------------------------------

REFERENCE_DOCS = {
    "d1": "alpha",
    "d2": "alpha beta",
    "d3": "alpha beta gamma",
    "d4": "beta gamma delta epsilon",
    "d5": "alpha alpha beta gamma delta",
}
# N=5, total_len=15, avgdl=3.0
# df: alpha 4, beta 4, gamma 3, delta 2, epsilon 1
REFERENCE_N = 5
REFERENCE_TOTAL = 15

# Quantised scores, derived longhand from
#   idf(t)   = ln((N - df + 0.5)/(df + 0.5) + 1)
#   score    = idf * tf * (k1+1) / (tf + k1*(1 - b + b*dl/avgdl))
# with k1=1.2, b=0.75.
REFERENCE_SCORES = {
    ("alpha",): {
        "d1": 395562850,   # dl=1 tf=1
        "d2": 333105558,   # dl=2 tf=1
        "d3": 287682072,   # dl=3 tf=1
        "d5": 333105558,   # dl=5 tf=2
    },
    ("alpha", "gamma"): {
        "d1": 395562850,
        "d2": 333105558,
        "d3": 826678573,
        "d4": 474316921,
        "d5": 756602808,
    },
}


def _reference_tables(docs: dict[str, str]):
    tokens = {k: v.split() for k, v in docs.items()}
    doc_lens = {k: len(v) for k, v in tokens.items()}
    dfs: dict[str, int] = {}
    for token_list in tokens.values():
        for term in set(token_list):
            dfs[term] = dfs.get(term, 0) + 1
    return tokens, doc_lens, dfs


def _reference_postings(tokens, query):
    postings: dict[str, dict[str, int]] = {}
    for term in query:
        hits = {k: v.count(term) for k, v in tokens.items() if term in v}
        if hits:
            postings[term] = hits
    return postings


@pytest.mark.parametrize("query", list(REFERENCE_SCORES))
def test_bm25_matches_hand_computed_reference(query):
    """Exact integer equality against independently derived values."""
    tokens, doc_lens, dfs = _reference_tables(REFERENCE_DOCS)
    scored = bm25.score_documents(
        postings=_reference_postings(tokens, query),
        dfs=dfs,
        doc_lens=doc_lens,
        n_docs=REFERENCE_N,
        total_len=REFERENCE_TOTAL,
    )
    assert {s.node_id: s.score_q for s in scored} == REFERENCE_SCORES[query]


def test_reference_corpus_shape_is_what_the_literals_assume():
    """Guard the guard: if the reference corpus changes, the literals are stale."""
    _, doc_lens, dfs = _reference_tables(REFERENCE_DOCS)
    assert len(doc_lens) == REFERENCE_N
    assert sum(doc_lens.values()) == REFERENCE_TOTAL
    assert dfs == {"alpha": 4, "beta": 4, "gamma": 3, "delta": 2, "epsilon": 1}


def test_exact_bm25_ties_occur_which_is_why_stage1_needs_a_tiebreak():
    """d2 and d5 score identically on 'alpha' despite differing dl and tf.

    Evidence that exact ties in BM25 are real rather than hypothetical, and
    therefore that M1.3's tiebreak chain ending in the content-addressed node
    id is load-bearing rather than defensive.
    """
    scores = REFERENCE_SCORES[("alpha",)]
    assert scores["d2"] == scores["d5"]
    assert len(set(scores.values())) < len(scores)


# --------------------------------------------------------------------------
# Length normalisation, with its control
# --------------------------------------------------------------------------

def _padded_corpus():
    """A short exact match against a long document that repeats the term."""
    corpus = {
        "short": ["caching"],
        "padded": ["caching", "caching"] + [f"filler{i}" for i in range(30)],
    }
    for j in range(8):
        corpus[f"other{j}"] = [f"other{j}x{i}" for i in range(5)]
    doc_lens = {k: len(v) for k, v in corpus.items()}
    dfs: dict[str, int] = {}
    for tokens in corpus.values():
        for term in set(tokens):
            dfs[term] = dfs.get(term, 0) + 1
    postings = {"caching": {k: v.count("caching")
                            for k, v in corpus.items() if "caching" in v}}
    return postings, dfs, doc_lens, len(corpus), sum(doc_lens.values())


def _score_padded(b: float) -> dict[str, int]:
    postings, dfs, doc_lens, n_docs, total = _padded_corpus()
    scored = bm25.score_documents(
        postings=postings, dfs=dfs, doc_lens=doc_lens,
        n_docs=n_docs, total_len=total, b=b,
    )
    return {s.node_id: s.score_q for s in scored}


def test_padded_document_does_not_outrank_short_exact_match():
    """The checkpoint assertion: length normalisation is real.

    `padded` carries tf=2 against `short`'s tf=1, so it wins on raw term
    frequency. It must still lose, because dl=32 against avgdl=7.3.
    """
    scores = _score_padded(b=bm25.B)
    assert scores["short"] > scores["padded"]


def test_control_padded_document_wins_when_b_is_zero():
    """Proof the test above can fail.

    With length normalisation disabled - which is exactly what the prior
    engine did - the padded document wins on tf alone. Without this control,
    the assertion above would be indistinguishable from one that happened to
    hold for an unrelated reason.
    """
    scores = _score_padded(b=0.0)
    assert scores["padded"] > scores["short"]


CORPUS_QUERIES = [
    "prompt caching", "tool use", "extended thinking", "rag retrieval",
    "agent orchestration", "streaming", "pdf vision", "citations",
    "batch processing", "embeddings semantic search",
    "json mode structured output", "summarization", "classification",
    "sub agents", "memory",
]


def _top_set(scored) -> set[str]:
    """Every document sharing the maximum score.

    Returning a *set* rather than one document is what makes the metrics
    below tiebreak-free. Picking a single top-1 would require a convention,
    and on this corpus the convention decides the answer for 7 of 15 queries.
    """
    best = max(s.score_q for s in scored)
    return {s.node_id for s in scored if s.score_q == best}


@requires_source
def test_length_normalisation_changes_the_real_corpus_ranking(index):
    """`b` is load-bearing on real data, not only on a constructed case.

    Every assertion is an exact integer over a tiebreak-free quantity. The
    directional claim - disabling normalisation never promotes a *shorter*
    result - is the load-bearing one, and it is asserted as `shorter == 0`
    rather than as a mean, because a mean can hide counterexamples.
    """
    assert len(CORPUS_QUERIES) == MEASURED_QUERIES

    conn = index["conn"]
    n_docs, total = corpus_totals(conn)
    doc_lens = doc_lengths(conn)
    tied = disjoint = longer = shorter = 0

    for text in CORPUS_QUERIES:
        query = tokenize(text)
        postings = postings_for_terms(conn, query)
        if not postings:
            continue
        common = dict(postings=postings, dfs=df_for_terms(conn, query),
                      doc_lens=doc_lens, n_docs=n_docs, total_len=total)
        top_norm = _top_set(bm25.score_documents(**common, b=0.75))
        top_flat = _top_set(bm25.score_documents(**common, b=0.0))

        if len(top_norm) > 1 or len(top_flat) > 1:
            tied += 1
        if top_norm.isdisjoint(top_flat):
            disjoint += 1

        mean_norm = sum(doc_lens[x] for x in top_norm) / len(top_norm)
        mean_flat = sum(doc_lens[x] for x in top_flat) / len(top_flat)
        if mean_flat > mean_norm:
            longer += 1
        elif mean_flat < mean_norm:
            shorter += 1

    assert tied == MEASURED_TIED_TOPSET_QUERIES
    assert disjoint == MEASURED_DISJOINT_TOPSETS
    assert longer == MEASURED_DIRECTION_LONGER
    assert shorter == MEASURED_DIRECTION_SHORTER, (
        "disabling length normalisation promoted a shorter result; the "
        "directional claim in spec/ranking.json no longer holds"
    )


@requires_source
def test_exact_ties_are_pervasive_on_the_real_corpus(index):
    """Nearly half the queries have a tie for top place.

    This is the empirical case for M1.3's injective sort key. Ties are not a
    rare edge case to be handled defensively - without a total order, the
    identity of "the best result" is undefined for 7 of 15 ordinary queries,
    and any answer is an artefact of an arbitrary convention.
    """
    conn = index["conn"]
    n_docs, total = corpus_totals(conn)
    doc_lens = doc_lengths(conn)
    with_ties = 0
    for text in CORPUS_QUERIES:
        query = tokenize(text)
        postings = postings_for_terms(conn, query)
        if not postings:
            continue
        scored = bm25.score_documents(
            postings=postings, dfs=df_for_terms(conn, query),
            doc_lens=doc_lens, n_docs=n_docs, total_len=total,
        )
        if len(_top_set(scored)) > 1:
            with_ties += 1
    assert with_ties > 0, (
        "no ties found; M1.3's tiebreak chain would be untested by this corpus"
    )


# --------------------------------------------------------------------------
# Determinism properties
# --------------------------------------------------------------------------

def test_all_scores_are_int():
    """No float may reach a comparison."""
    tokens, doc_lens, dfs = _reference_tables(REFERENCE_DOCS)
    scored = bm25.score_documents(
        postings=_reference_postings(tokens, ("alpha", "gamma")),
        dfs=dfs, doc_lens=doc_lens,
        n_docs=REFERENCE_N, total_len=REFERENCE_TOTAL,
    )
    assert scored
    for s in scored:
        assert type(s.score_q) is int
        assert type(s.matched_terms) is int
        assert type(s.doc_len) is int


def test_scoring_is_commutative_over_posting_order():
    """Permuting the postings cannot change any score.

    Regression guard rather than a discovery: `fsum` returns the unique
    correctly-rounded total, so this holds by construction. It is kept because
    it would catch a regression to naive accumulation with early rounding -
    which `test_quantisation_point_is_a_real_choice` shows produces different
    numbers.
    """
    tokens, doc_lens, dfs = _reference_tables(REFERENCE_DOCS)
    query = ("alpha", "beta", "gamma")
    base = _reference_postings(tokens, query)
    results = set()
    for order in itertools.permutations(base):
        shuffled = {k: base[k] for k in order}
        scored = bm25.score_documents(
            postings=shuffled, dfs=dfs, doc_lens=doc_lens,
            n_docs=REFERENCE_N, total_len=REFERENCE_TOTAL,
        )
        results.add(json.dumps({s.node_id: s.score_q for s in scored},
                               sort_keys=True))
    assert len(results) == 1


def test_quantisation_point_is_a_real_choice():
    """The two orders of operation genuinely differ.

    `spec/ranking.json` records that quantisation happens once after
    summation, not per contribution. If both produced identical output the
    spec would be asserting a decision no test could distinguish from its
    opposite - unfalsifiable, and therefore worthless.
    """
    tokens, doc_lens, dfs = _reference_tables(REFERENCE_DOCS)
    postings = _reference_postings(tokens, ("alpha", "beta", "gamma"))
    common = dict(postings=postings, dfs=dfs, doc_lens=doc_lens,
                  n_docs=REFERENCE_N, total_len=REFERENCE_TOTAL)
    after = {s.node_id: s.score_q for s in bm25.score_documents(**common)}
    per_term = bm25.score_documents_quantise_per_term(**common)
    assert after != per_term, (
        "quantise-after-summation and quantise-per-term agree on this input, "
        "so the spec's stated choice is untested here"
    )


def test_idf_is_never_negative():
    """A term in most of the corpus must not subtract from a score."""
    for n_docs in (1, 5, 266, 10_000):
        for df in range(1, n_docs + 1):
            assert bm25.idf(df, n_docs) >= 0.0


def test_code_and_ranking_spec_agree():
    """Drift between spec/ranking.json and the implementation is a failure."""
    with open(ROOT / "spec" / "ranking.json") as f:
        spec = json.load(f)
    assert spec["bm25"]["k1"] == bm25.K1
    assert spec["bm25"]["b"] == bm25.B
    assert spec["quantisation"]["exponent"] == QUANTUM_EXP
    assert spec["quantisation"]["point"] == "after_summation"
    assert spec["document"]["fields"] == ["name", "description"]


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------

def test_index_and_query_paths_share_one_tokenizer():
    """The spec's guarantee, made true rather than merely stated."""
    text = "PDF-vision & 100k-token Context: Real-Time!"
    from_query, _ = terms(text=text)
    assert from_query == tokenize(text)
    assert from_query == ["pdf", "vision", "100k", "token", "context",
                          "real", "time"]


def test_tokenizer_is_ascii_only_by_construction():
    """`\\w` would depend on the interpreter's Unicode database version."""
    assert TOKEN_PATTERN.pattern == r"[a-z0-9]+"
    assert tokenize("naïve × café → x") == ["na", "ve", "caf", "x"]


def test_document_text_excludes_type_and_source_ref():
    assert document_text("Name", "Desc") == "Name Desc"


# --------------------------------------------------------------------------
# Candidates: complete, and never truncated
# --------------------------------------------------------------------------

@requires_source
def test_candidates_equal_the_posting_union(index):
    """The checkpoint assertion: no candidate with a nonzero score is lost."""
    conn, index_hash = index["conn"], index["hash"]
    for text in ("prompt caching", "agent tool use streaming", "memory"):
        query = tokenize(text)
        found, _ = lexical.candidates(
            conn=conn, query_terms=query, index_hash=index_hash
        )
        union: set[str] = set()
        for term_postings in postings_for_terms(conn, query).values():
            union.update(term_postings)
        assert len(found) == len(union)
        assert set(found) == union


@requires_source
def test_no_query_length_truncation(index):
    """A query longer than 10 terms uses all of them.

    The prior engine silently kept only the first ten
    (`python_apps_hybrid_query.py:339`). Here, adding a term that matches
    something must be able to grow the candidate set.
    """
    conn, index_hash = index["conn"], index["hash"]
    long_query = tokenize(
        "prompt caching tool use streaming citations batch vision pdf "
        "embeddings memory classification summarization agent thinking"
    )
    assert len(long_query) > 10

    head, _ = lexical.candidates(
        conn=conn, query_terms=long_query[:10], index_hash=index_hash
    )
    full, _ = lexical.candidates(
        conn=conn, query_terms=long_query, index_hash=index_hash
    )
    assert set(head).issubset(set(full))
    assert len(full) > len(head), (
        "terms beyond the tenth contributed no candidates; this test cannot "
        "detect truncation on this query"
    )


@requires_source
def test_out_of_vocabulary_query_returns_empty_not_error(index):
    """Correct under subordination: no authoritative results to append below."""
    found, _ = lexical.candidates(
        conn=index["conn"],
        query_terms=tokenize("zzzqqq wwwvvv"),
        index_hash=index["hash"],
    )
    assert found == []


@requires_source
def test_scoring_the_real_corpus_yields_only_ints(index):
    scored, justification = lexical.bm25_score(
        conn=index["conn"],
        query_terms=tokenize("prompt caching"),
        index_hash=index["hash"],
    )
    assert scored
    for _node_id, score_q, matched, doc_len in scored:
        assert type(score_q) is int and type(matched) is int
        assert type(doc_len) is int
    assert justification.determinism == "deterministic"
    assert justification.authority == "authoritative"
    assert justification.confidence is None


# --------------------------------------------------------------------------
# Index consistency
# --------------------------------------------------------------------------

@requires_source
def test_lexical_index_agrees_with_the_manifest(index):
    conn = index["conn"]
    manifest = read_manifest(conn)
    counts = manifest["content"]["lexical"]["counts"]
    assert counts["documents"] == table_count(conn, "doc_stats")
    assert counts["terms"] == table_count(conn, "terms")
    assert counts["postings"] == table_count(conn, "postings")
    n_docs, total = corpus_totals(conn)
    assert n_docs == counts["documents"]
    assert total == manifest["content"]["lexical"]["total_length"]


@requires_source
def test_every_node_has_doc_stats_and_postings_reference_real_nodes(index):
    conn = index["conn"]
    node_ids = {n.id for n in iter_nodes(conn)}
    assert set(doc_lengths(conn)) == node_ids
    posted = {p.node_id for p in iter_postings(conn)}
    assert posted <= node_ids


@requires_source
def test_df_equals_the_number_of_documents_posting_each_term(index):
    """df is derived, so it must agree with the postings it summarises."""
    conn = index["conn"]
    counted: dict[str, int] = {}
    for posting in iter_postings(conn):
        counted[posting.term] = counted.get(posting.term, 0) + 1
    stored = {
        r[0]: r[1] for r in conn.execute("SELECT term, df FROM terms")
    }
    assert stored == counted


def test_inconsistent_index_raises_rather_than_scoring_partially():
    """A missing df or doc_stats row must stop scoring, not skip a term."""
    with pytest.raises(ValueError, match="no document-frequency row"):
        bm25.score_documents(
            postings={"alpha": {"d1": 1}}, dfs={}, doc_lens={"d1": 1},
            n_docs=1, total_len=1,
        )
    with pytest.raises(ValueError, match="no doc_stats row"):
        bm25.score_documents(
            postings={"alpha": {"d1": 1}}, dfs={"alpha": 1}, doc_lens={},
            n_docs=1, total_len=1,
        )
