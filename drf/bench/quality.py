"""Retrieval-quality metrics, and the discipline that replaces exact integers.

Milestone 1's assertion rule was *assert integers, never floats*, and it was
load-bearing: the chaos control scored Kendall's Tau **0.9761** while being
provably non-deterministic, so a float near 1.0 proved nothing. That rule
**cannot transfer here**. nDCG and recall are ratios; they are floats by
definition and pretending otherwise would be worse than admitting it.

So this module keeps the *shape* of the rule rather than the letter. Every
`Judged` result carries two surfaces:

* **Integers, which is what tests assert.** How many relevant documents came
  back, at which 1-based ranks, and how much graded gain was collected. These
  are exact - a rank is a position, not a measurement.
* **Floats, which are for people.** nDCG, recall, precision, reciprocal rank.
  Reported, never asserted bare.

`ranks_of_relevant` is the M2 analogue of `discordant_pairs`. Milestone 1
measured that set metrics are blind to ordering - Jaccard reported 1.0000 for
a pipeline returning five different orderings in five runs - and recall@k has
exactly the same blindness. A tuple of ranks is a *set metric's* information
plus the ordering it discards, in integers, so a reversal is visible in the
assertion surface rather than only in the third decimal of a float.

**No float here may be asserted against a bare comparison.** A quality number
is only evidence relative to a control; see `drf/bench/controls.py`.

Stdlib only.
"""

import math
from typing import Mapping, NamedTuple, Sequence

# Graded relevance, TREC-style. Four levels rather than binary because nDCG
# needs to distinguish "on topic" from "answers the question" - and because
# binary judgements on a 266-node corpus of closely related documents would
# collapse to almost everything being a 1.
GRADES = {
    0: "irrelevant",
    1: "marginal",      # touches the topic, would not satisfy the query
    2: "relevant",      # a reasonable answer
    3: "vital",         # the document the query is really asking for
}

# A document counts as relevant for recall/precision at this grade or above.
# Stated once, here, because a recall figure computed at a different threshold
# is a different number wearing the same name.
RELEVANCE_THRESHOLD = 2


def gain(grade: int) -> int:
    """Exponential gain, `2**g - 1`. Integer by construction, and used as one.

    The exponential form is what makes nDCG discriminate between a vital
    document and a merely relevant one (7 against 3) rather than treating the
    scale as linear. Its integrality is the reason `gain_total` can live on
    the assertion surface while `ndcg` cannot.
    """
    if grade not in GRADES:
        raise ValueError(f"grade {grade!r} is not one of {sorted(GRADES)}")
    return 2 ** grade - 1


class Judged(NamedTuple):
    """One ranking judged against labels at one depth.

    Integers first: that ordering is not cosmetic. It is the same convention
    as `bench.metrics.Comparison`, and it exists so that reading a result left
    to right shows the assertable facts before the reportable ones.
    """

    # --- Exact integers: the assertion surface ---------------------------
    depth: int                       # the k this was computed at
    relevant_total: int              # relevant docs in the labels, retrieved or not
    relevant_retrieved: int          # of those, how many appear within depth
    first_relevant_rank: int         # 1-based; 0 means none was found
    ranks_of_relevant: tuple[int, ...]   # 1-based ranks, ascending
    gain_total: int                  # sum of graded gains within depth

    # --- Floats: reporting only, never asserted bare ---------------------
    ndcg: float
    recall: float
    precision: float
    reciprocal_rank: float

    @property
    def found_nothing(self) -> bool:
        return self.relevant_retrieved == 0


def _dcg(order: Sequence[str], labels: Mapping[str, int]) -> float:
    """Discounted cumulative gain over an already-truncated ranking.

    `math.fsum` over the terms in rank order: correctly rounded, so the value
    does not depend on summation order or platform. That matters less than it
    did in stage 1 - nothing here feeds a comparison that decides an ordering -
    but a benchmark that reported different figures on two machines would
    undermine the claim the benchmark exists to support.
    """
    return math.fsum(
        gain(labels.get(node_id, 0)) / math.log2(rank + 1)
        for rank, node_id in enumerate(order, start=1)
    )


def ideal_order(labels: Mapping[str, int]) -> list[str]:
    """The best possible ranking of the labelled documents.

    Sorted by `(-grade, node_id)`, not by grade alone. Grade alone is not a
    total order - most documents share a grade - so the ideal ranking would
    otherwise depend on dictionary iteration order, and IDCG is the
    *denominator* of every nDCG figure this project will publish. The same
    injectivity argument as `stage1.sort_key`, applied to the yardstick
    instead of to the results.
    """
    return sorted(labels, key=lambda node_id: (-labels[node_id], node_id))


def judge(
    ranking: Sequence[str],
    labels: Mapping[str, int],
    *,
    depth: int,
    threshold: int = RELEVANCE_THRESHOLD,
) -> Judged:
    """Judge one ranking against one query's labels at one depth.

    `labels` maps node id to grade and is expected to include explicit zeros
    for documents that were judged and found irrelevant. Anything absent is
    treated as grade 0 - unjudged and non-relevant - which is the standard
    pooled-assessment assumption and is recorded here because it silently
    lowers every recall figure if the pool was shallow.

    `relevant_total` counts relevant documents **in the labels**, not in the
    ranking, so a document the retrieval path can never reach still lands in
    the denominator. That is deliberate: the advisory layer's only honest
    question is whether it finds relevant documents lexical search missed
    entirely, and a denominator that quietly excluded them could not express
    the answer.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    truncated = list(ranking[:depth])
    relevant = {node_id for node_id, g in labels.items() if g >= threshold}

    ranks = tuple(
        rank for rank, node_id in enumerate(truncated, start=1)
        if node_id in relevant
    )
    ideal = ideal_order(labels)[:depth]
    idcg = _dcg(ideal, labels)
    dcg = _dcg(truncated, labels)

    return Judged(
        depth=depth,
        relevant_total=len(relevant),
        relevant_retrieved=len(ranks),
        first_relevant_rank=ranks[0] if ranks else 0,
        ranks_of_relevant=ranks,
        gain_total=sum(gain(labels.get(node_id, 0)) for node_id in truncated),
        # nDCG is 1.0 when there is nothing to get wrong: no labelled document
        # carries any gain, so every ranking is equally ideal. Returning 0.0
        # would penalise a ranking for a property of the labels.
        ndcg=(dcg / idcg) if idcg > 0 else 1.0,
        recall=(len(ranks) / len(relevant)) if relevant else 1.0,
        precision=len(ranks) / len(truncated) if truncated else 0.0,
        reciprocal_rank=(1.0 / ranks[0]) if ranks else 0.0,
    )


def advisory_horizon(deterministic: Sequence[str]) -> int:
    """`|D|` - the depth below which the advisory layer provably cannot act.

    This is the single most important number in M2's evaluation, and it is not
    a metric but a *bound*. Merge is append-only, so positions `0 .. |D|-1` of
    any merged result are byte-identical whatever the provider does or fails
    to do. Every metric computed at `k <= |D|` is therefore **structurally
    invariant** to the neural layer.

    Evaluating precision@1 or nDCG@5 with the provider on and off will show
    exactly zero difference on this corpus, where `|D|` is typically 10-86.
    That is the guarantee working, not a negative result about neural
    retrieval - and the distinction is impossible to see without this number,
    which is why the harness computes it rather than leaving it to prose.
    """
    return len(deterministic)


def can_advisory_affect(deterministic: Sequence[str], depth: int) -> bool:
    """Whether a metric at this depth is even capable of moving. See above."""
    return depth > advisory_horizon(deterministic)


def aggregate(judgements: Sequence[Judged]) -> dict:
    """Summarise per-query judgements. Integers summed, floats averaged.

    Averaging over queries rather than pooling documents (macro, not micro) so
    a single broad query cannot dominate the figure - `tool use` returns 86
    candidates against `rag retrieval`'s 10.
    """
    if not judgements:
        return {"queries": 0}
    count = len(judgements)
    return {
        "queries": count,
        "depth": judgements[0].depth,
        # Assertion surface.
        "relevant_total": sum(j.relevant_total for j in judgements),
        "relevant_retrieved": sum(j.relevant_retrieved for j in judgements),
        "gain_total": sum(j.gain_total for j in judgements),
        "queries_with_no_relevant_hit": sum(1 for j in judgements if j.found_nothing),
        # Reporting only.
        "ndcg": math.fsum(j.ndcg for j in judgements) / count,
        "recall": math.fsum(j.recall for j in judgements) / count,
        "precision": math.fsum(j.precision for j in judgements) / count,
        "mrr": math.fsum(j.reciprocal_rank for j in judgements) / count,
    }
