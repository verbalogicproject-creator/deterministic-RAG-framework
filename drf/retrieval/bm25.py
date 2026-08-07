"""Okapi BM25, with the length normalisation the prior engine omitted.

    score(t,d) = idf(t) * tf(t,d) * (k1 + 1)
                 / ( tf(t,d) + k1 * (1 - b + b * dl(d)/avgdl) )

    idf(t)     = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

The `+ 1` inside the logarithm keeps idf non-negative. Without it, a term
occurring in more than half the corpus contributes a *negative* score, so a
document matching a common term can rank below one matching nothing - a result
that cannot be explained to a user.

**Why `b` is here.** The engine this replaces scored
`idf * (tf*k1)/(tf+k1)` with `k1=2.5` and no `b` or `avgdl` at all. Measured
against this implementation over 15 corpus queries, using tiebreak-free
metrics: 17.0% of strictly-ordered pairs at depth 10 flip (60 of 352), top
sets are disjoint for 3 of 15 queries, and mean top-set length rises from 15.9
to 24.9 tokens against a corpus mean of 18.9 - **9 longer, 0 shorter**, so the
direction holds monotonically. Producer: `tools/measure_length_norm.py`.

A note on those numbers, because the first version of them was wrong in an
instructive way: 7 of 15 queries have *exact ties* in a top set, so any
"top-1" metric silently depends on how ties are broken - lowest node id gives
9 changed, highest gives 4. The metrics above are defined only over strictly
ordered comparisons and are therefore convention-free. Ties in BM25 alone are
pervasive, which is precisely why stage1's sort key must end in an injective
component.

`k1`, by contrast, barely matters on this corpus: tf == 1 for 82.7% of
(document, term) pairs, so the saturation term has almost nothing to act on.
Moving k1 from 1.2 to 2.5 shifts ranking discordance by one percentage point.

**Float discipline.** Contributions are accumulated with `math.fsum` and the
total is quantised to `int` exactly once, after summation. Because the
correctly-rounded sum of a multiset is unique, accumulation is *commutative* -
the result does not depend on the order contributions are visited, so no
upstream sort is load-bearing. The alternative (quantise each contribution,
sum integers) is equally deterministic but admits up to half a quantum of
error per term rather than half a quantum in total; `spec/ranking.json`
records the choice and `tests/test_retrieval.py` proves the two differ.

This module is pure arithmetic. It imports nothing from the contract layer, so
it can be tested and reasoned about without any framework machinery.

Stdlib only.
"""

import math
from typing import NamedTuple

from ..fixed import exact_sum, quantize

# Mirrored in spec/ranking.json. A mismatch is a failing test, not a runtime
# concern - see tests/test_retrieval.py::test_code_and_ranking_spec_agree.
K1 = 1.2
B = 0.75


class Scored(NamedTuple):
    """One scored candidate.

    `matched_terms` and `doc_len` are carried because M1.3's sort key needs
    them as tiebreak components. Computing them here avoids a second pass that
    could disagree with the pass that produced the score.
    """
    node_id: str
    score_q: int          # fixed-point int; never a float
    matched_terms: int
    doc_len: int


def idf(df: int, n_docs: int) -> float:
    """Inverse document frequency, floored at zero by the +1 inside the log.

    `df` is assumed to satisfy 1 <= df <= n_docs, which the index guarantees:
    a term exists in `terms` only because at least one document posted it.
    """
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


def score_documents(
    *,
    postings: dict[str, dict[str, int]],
    dfs: dict[str, int],
    doc_lens: dict[str, int],
    n_docs: int,
    total_len: int,
    k1: float = K1,
    b: float = B,
) -> list[Scored]:
    """Score every document appearing in `postings`.

    `postings` is {term: {node_id: tf}} - already restricted to the query's
    terms by the caller, so the set of documents mentioned here is exactly the
    support of BM25. Documents with no query term are not scored because their
    score is exactly zero, not because they were filtered.

    `avgdl` is computed here from two exact integers rather than read from a
    stored float, so it is reproduced identically rather than round-tripped
    through a decimal representation.

    Returns results ordered by node_id. That is *not* the ranking - it is a
    stable enumeration order. Ranking happens in stage1 (M1.3), which applies
    the injective sort key.
    """
    if n_docs <= 0 or not postings:
        return []

    avgdl = total_len / n_docs
    if avgdl <= 0:
        raise ValueError(
            f"avgdl must be positive, got {avgdl} from total_len={total_len}, "
            f"n_docs={n_docs}; an index of empty documents cannot be scored"
        )

    contributions: dict[str, list[float]] = {}
    matched: dict[str, int] = {}

    # Sorted for a reproducible walk; fsum makes the *result* independent of
    # this order, so the sort is a convenience rather than a correctness
    # dependency. See spec/ranking.json "commutativity".
    for term in sorted(postings):
        df = dfs.get(term)
        if not df:
            # A term with postings but no df row means the index is
            # inconsistent. Skipping silently would produce a plausible score
            # from an incomplete index.
            raise ValueError(
                f"term {term!r} has postings but no document-frequency row; "
                "the lexical index is inconsistent"
            )
        term_idf = idf(df, n_docs)
        for node_id, tf in postings[term].items():
            doc_len = doc_lens.get(node_id)
            if doc_len is None:
                raise ValueError(
                    f"node {node_id!r} is posted under {term!r} but has no "
                    "doc_stats row; the lexical index is inconsistent"
                )
            denom = tf + k1 * (1.0 - b + b * doc_len / avgdl)
            contributions.setdefault(node_id, []).append(
                term_idf * tf * (k1 + 1.0) / denom
            )
            matched[node_id] = matched.get(node_id, 0) + 1

    return [
        Scored(
            node_id=node_id,
            score_q=quantize(exact_sum(contributions[node_id])),
            matched_terms=matched[node_id],
            doc_len=doc_lens[node_id],
        )
        for node_id in sorted(contributions)
    ]


def score_documents_quantise_per_term(
    *,
    postings: dict[str, dict[str, int]],
    dfs: dict[str, int],
    doc_lens: dict[str, int],
    n_docs: int,
    total_len: int,
    k1: float = K1,
    b: float = B,
) -> dict[str, int]:
    """The rejected alternative: quantise each contribution, then sum ints.

    Kept in the codebase, and only used by the test that proves the two orders
    of operation genuinely differ. Without it, `spec/ranking.json`'s claim
    that the quantisation point is a real decision would be unfalsifiable -
    the spec would assert a choice that no test could distinguish from its
    opposite.
    """
    avgdl = total_len / n_docs
    out: dict[str, int] = {}
    for term in sorted(postings):
        term_idf = idf(dfs[term], n_docs)
        for node_id, tf in postings[term].items():
            denom = tf + k1 * (1.0 - b + b * doc_lens[node_id] / avgdl)
            out[node_id] = out.get(node_id, 0) + quantize(
                term_idf * tf * (k1 + 1.0) / denom
            )
    return out
