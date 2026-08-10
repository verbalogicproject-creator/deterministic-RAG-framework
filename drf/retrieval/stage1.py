"""Stage 1: the authoritative total order D.

    sort key = (-bm25_q, -matched_terms, best_depth, doc_len, node_id)

Every component is `int` or `str`. No float is ever compared. The final
component is the content-addressed node id, which is **injective** over the
candidate set, and a tuple ending in an injective component is itself
injective - so the induced order is *strict* and *total*. Two consequences
follow, and they are the whole point of this module:

* `sorted()` returns the same sequence for every permutation of its input, so
  sort stability is irrelevant and candidate enumeration order cannot leak in.
* Truncation to top-k is safe, because the k-th and (k+1)-th elements are
  never tied. The prior engine's `[:15]` boundary cascaded a single tie into a
  global reordering precisely because its order was not strict.

That ties are real here is measured, not assumed: 7 of 15 ordinary queries
have exact BM25 ties in their top set, and the 5-document reference corpus
reproduces one (`d2` and `d5` both score 333105558 on "alpha"). Without the
tiebreak chain, "the best result" would be undefined for nearly half of
ordinary queries.

---

**Two deliberate deviations from the original build plan.**

*1. There is no `s1_q` component.* The plan specified
`(-s1_q, -bm25_q, ...)` where `s1_q` blended lexical and graph signal. Any
such blend needs a weight, and **milestone 1 carries no relevance labels** -
so a weight chosen now could not be validated, only asserted. Introducing an
unfalsifiable tuning parameter into the one layer that must be defensible is
exactly the failure this project exists to remove. Lexicographic composition
is parameter-free and does real work: graph proximity orders precisely the
documents lexical scoring cannot distinguish. A weighted combination is an
M2 question, to be settled when labels exist to settle it.

This also keeps a clear distance from Reciprocal Rank Fusion, which appears
in 5 of 9 playbooks in the recovered corpus and is recommended there for
"production queries". RRF blends rankings into one score; here graph signal
can never override a strictly better lexical match, only break its ties.

*2. D contains lexical candidates only.* Graph expansion supplies
`best_depth`; it does not inject documents that contain no query term.
Admitting unmatched documents would be a relevance claim, and relevance is
exactly what M1 cannot measure. The alternative - appending graph-reached
documents below the lexical ones - is recorded as an M2 decision, not
silently taken here.

Stdlib only.
"""

from typing import NamedTuple

from ..contract import ActionOutput, action
from ..store import corpus_totals, df_for_terms, doc_lengths, postings_for_terms
from . import bm25, graph

# How many top lexical hits seed graph expansion, and how far it runs.
# Both are declared inputs of `stage1.rank`, so changing either changes the
# identity of the computation rather than silently changing its result.
DEFAULT_SEED_COUNT = 10
DEFAULT_MAX_DEPTH = 2

# Whether graph-reached documents that contain no query term enter D at all.
# See `graph_only_candidates`. The default is a policy choice with a measured
# basis, recorded in spec/benchmarks.json:graph_candidate_admission - not a
# guess, and not a value tuned against an outcome.
DEFAULT_ADMIT_GRAPH_CANDIDATES = False


class Ranked(NamedTuple):
    """One result in the authoritative order."""
    node_id: str
    bm25_q: int
    matched_terms: int
    best_depth: int
    doc_len: int


def sort_key(result: Ranked) -> tuple:
    """The injective ordering. Negation gives descending on the score fields.

    Component order is lexical-before-structural on purpose: score first, then
    breadth of term coverage, and only then graph proximity. A document that
    matches the query better is never displaced by one that merely sits closer
    to a seed.
    """
    return (
        -result.bm25_q,
        -result.matched_terms,
        result.best_depth,
        result.doc_len,
        result.node_id,
    )


def select_seeds(scored: list[bm25.Scored], seed_count: int) -> list[str]:
    """The top `seed_count` lexical hits, chosen by a total order.

    Seeds are picked with `(-score_q, node_id)` rather than by score alone.
    Score alone is *not* a total order on this corpus - ties are pervasive -
    so a plain `nlargest` would return whichever tied document the sort
    happened to visit first, and the seed set would depend on enumeration
    order. Including the node id makes the choice injective and therefore
    reproducible.
    """
    ordered = sorted(scored, key=lambda s: (-s.score_q, s.node_id))
    return [s.node_id for s in ordered[:seed_count]]


def graph_only_candidates(
    scored: list[bm25.Scored],
    depths: dict[str, int],
    doc_lens: dict[str, int],
) -> list[bm25.Scored]:
    """Documents the graph reached that contain no query term at all.

    Scored **exactly zero**, with zero matched terms. That is not a placeholder
    standing in for an unknown value - it is the true BM25 score of a document
    with no query term in it, so nothing is being invented. The number is what
    the existing scorer would compute; it is simply cheaper to state than to
    derive.

    **Why zero is safe, and why that is not an accident.** The sort key opens
    with `-bm25_q`, so a document at zero sorts strictly below every document
    with a positive score. That holds only because `bm25.idf` uses
    `log(... + 1.0)`, which floors idf at zero and makes every real match score
    positive - measured across the corpus query set, the minimum non-zero score
    is 1,017,536,991 and there are no negative or zero scores at all. That `+1`
    was added in M1.2 for a different reason (a negative idf would let a
    document matching a common term rank below one matching nothing, which
    cannot be explained to a user). It turns out to be the precondition for
    admitting unmatched documents safely. Remove it and this function silently
    starts promoting unmatched documents above matched ones.

    **The ordering consequence, stated plainly.** Within the admitted tail
    every document has `bm25_q = 0` and `matched_terms = 0`, so the third
    component - `best_depth` - becomes the *primary* discriminator. This is the
    one place in the design where graph proximity actually decides an order,
    and it decides it only among documents that lexical retrieval scored
    identically at zero. No weight, no blend: the same lexicographic key,
    reaching its third component for the first time in earnest.

    Measured through M2.2: as a *tiebreak* the graph signal is inert above rank
    19 across the whole query set. This is the re-scope that gives it a job
    where it can act without ever displacing a lexical match.
    """
    already = {s.node_id for s in scored}
    return [
        bm25.Scored(
            node_id=node_id,
            score_q=0,
            matched_terms=0,
            doc_len=doc_lens.get(node_id, 0),
        )
        for node_id in sorted(depths)
        if node_id not in already
    ]


def rank_candidates(
    scored: list[bm25.Scored], depths: dict[str, int]
) -> list[Ranked]:
    """Impose the strict total order on the scored candidates."""
    results = [
        Ranked(
            node_id=s.node_id,
            bm25_q=s.score_q,
            matched_terms=s.matched_terms,
            best_depth=depths.get(s.node_id, graph.UNREACHED_DEPTH),
            doc_len=s.doc_len,
        )
        for s in scored
    ]
    return sorted(results, key=sort_key)


@action(
    "stage1.rank",
    determinism="deterministic",
    authority="authoritative",
    inputs=("query_terms", "index_hash", "seed_count", "max_depth", "k1", "b",
            "admit_graph_candidates"),
)
def rank(
    *,
    conn,
    query_terms: list[str],
    index_hash: str,
    seed_count: int = DEFAULT_SEED_COUNT,
    max_depth: int = DEFAULT_MAX_DEPTH,
    k1: float = bm25.K1,
    b: float = bm25.B,
    admit_graph_candidates: bool = DEFAULT_ADMIT_GRAPH_CANDIDATES,
) -> ActionOutput:
    """Produce D, the authoritative total order.

    Every parameter that can change the output is a declared input, so a
    parameter change produces a different replay key rather than a spurious
    determinism violation.
    """
    postings = postings_for_terms(conn, query_terms)
    n_docs, total_len = corpus_totals(conn)
    doc_lens = doc_lengths(conn)
    scored = bm25.score_documents(
        postings=postings,
        dfs=df_for_terms(conn, query_terms),
        doc_lens=doc_lens,
        n_docs=n_docs,
        total_len=total_len,
        k1=k1,
        b=b,
    )
    seeds = select_seeds(scored, seed_count)
    depths = graph.expand(conn, seeds, max_depth) if seeds else {}

    # Seeds come from the lexical hits, so an all-out-of-vocabulary query has
    # no seeds, reaches nothing, and admits nothing. Graph admission therefore
    # cannot rescue an OOV query - the fix for that remains a Stage 1 lexical
    # fix (character n-grams), exactly as recorded in M1.
    admitted = (
        graph_only_candidates(scored, depths, doc_lens)
        if admit_graph_candidates else []
    )
    ordered = rank_candidates(scored + admitted, depths)

    return ActionOutput(
        value=[list(r) for r in ordered],
        evidence=(
            f"candidates={len(ordered)}",
            f"lexical={len(scored)}",
            f"graph_admitted={len(admitted)}",
            f"seeds={len(seeds)}",
            f"max_depth={max_depth}",
            f"graph_reached={len(depths)}",
        ),
    )


def is_strict_total_order(ordered: list[Ranked]) -> bool:
    """Every sort key distinct - the property the whole design rests on.

    Exposed as a function rather than left inside a test so that it can also
    be asserted at query time if a caller wants belt and braces.
    """
    keys = [sort_key(r) for r in ordered]
    return len(set(keys)) == len(keys)
