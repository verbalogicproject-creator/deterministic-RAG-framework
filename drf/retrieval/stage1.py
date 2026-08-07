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
    inputs=("query_terms", "index_hash", "seed_count", "max_depth", "k1", "b"),
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
) -> ActionOutput:
    """Produce D, the authoritative total order.

    Every parameter that can change the output is a declared input, so a
    parameter change produces a different replay key rather than a spurious
    determinism violation.
    """
    postings = postings_for_terms(conn, query_terms)
    n_docs, total_len = corpus_totals(conn)
    scored = bm25.score_documents(
        postings=postings,
        dfs=df_for_terms(conn, query_terms),
        doc_lens=doc_lengths(conn),
        n_docs=n_docs,
        total_len=total_len,
        k1=k1,
        b=b,
    )
    seeds = select_seeds(scored, seed_count)
    depths = graph.expand(conn, seeds, max_depth) if seeds else {}
    ordered = rank_candidates(scored, depths)

    return ActionOutput(
        value=[list(r) for r in ordered],
        evidence=(
            f"candidates={len(ordered)}",
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
