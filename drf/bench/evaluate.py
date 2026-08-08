"""The quality harness: system against controls, with self-checks that hold
before any real judgement exists.

M2.0 is built *before* M2.1's labels on purpose. An evaluation harness is
normally validated by running it on labelled data, which makes "is the harness
right?" and "is the system good?" the same experiment - and when the answer
disappoints, there is no way to tell which one failed. Building the harness
first breaks that circularity, but it raises an obvious objection: what can be
tested with no ground truth?

Three things, and they are enough.

**1. Properties that are true of any labels at all.** `oracle` sorts the
system's own candidates by grade, so `oracle.ndcg >= system.ndcg` is a
theorem, not a finding - it holds for every label set including nonsense ones.
If it is ever violated, nDCG is implemented wrongly. Likewise `reverse` holds
the retrieved set exactly constant, so its recall at full depth must equal the
system's; if a metric separates them, that metric is not the set metric it
claims to be. These run on synthetic labels and catch real arithmetic bugs.

**2. A hand-computed reference.** As in M1.2's BM25 fixture: nDCG derived on
paper for a five-document example, compared against a literal. Independent of
the implementation by construction.

**3. The structural bound.** `advisory_horizon` is measurable with no labels
whatsoever, and the assertion that the advisory layer cannot move a metric
below it is checkable by running the same evaluation twice.

What *cannot* be tested yet is whether the system retrieves well. That is
M2.1, and it stays out of this module.

**On margins.** `MIN_NDCG_MARGIN` is a declared requirement, not a
measurement. It says how much better than a relevance-blind ranking the system
must be before a quality claim may be published. Naming it here, in advance of
seeing any result, is the point - a threshold chosen after looking at the
number it must pass is not a threshold.

Stdlib only.
"""

import json
from pathlib import Path

from ..retrieval import stage1
from ..retrieval.tokenize import tokenize
from ..store import connect, read_manifest
from . import controls as controls_module
from . import quality
from .labels import LabelSet

ROOT = Path(__file__).resolve().parent.parent.parent
SPEC = ROOT / "spec" / "evaluation.json"


def _spec() -> dict:
    with open(SPEC) as handle:
        return json.load(handle)


class HarnessError(AssertionError):
    """A property that must hold for *any* labels was violated.

    Distinct from a poor result. This means the measurement is wrong, and no
    figure computed alongside it may be reported.
    """


def evaluate_query(
    ranking: list[str],
    labels: dict[str, int],
    *,
    depths: list[int],
    seeds: tuple[int, ...] = controls_module.SHUFFLE_SEEDS,
) -> dict:
    """Judge one ranking and every control over the same candidates.

    Controls reorder *the system's own candidate list*. That is deliberate and
    it narrows the question usefully: holding the retrieved set fixed, the
    comparison isolates the ordering, which is the only thing stage 1's sort
    key decides. Candidate generation is judged separately, by the oracle's
    own score - a low oracle means the documents were never retrieved and no
    reordering could have helped.
    """
    rankings = {"system": list(ranking)}
    for name, control in controls_module.all_controls(seeds).items():
        rankings[name] = control(ranking, labels)

    out: dict = {"horizon": quality.advisory_horizon(ranking), "depths": {}}
    for depth in depths:
        judged = {
            name: quality.judge(order, labels, depth=depth)
            for name, order in rankings.items()
        }
        _self_check(judged, ranking, depth)
        out["depths"][depth] = judged
    return out


def _self_check(judged: dict[str, quality.Judged], ranking: list[str], depth: int) -> None:
    """Properties true of every label set. A violation means the metric is wrong.

    These are the reason the harness can be trusted before ground truth
    exists. Neither depends on the labels being sensible - only on nDCG and
    recall meaning what they are supposed to mean.
    """
    system, oracle = judged["system"], judged["oracle"]

    # The oracle sorts the system's own candidates by grade, so no ordering of
    # that same set can score higher. Compared on the integer surface first,
    # because gain collected within a depth is exact; nDCG is checked with a
    # tolerance because it is a ratio of sums of floats.
    if oracle.gain_total < system.gain_total:
        raise HarnessError(
            f"depth {depth}: oracle collected less graded gain "
            f"({oracle.gain_total}) than the system ({system.gain_total}). "
            f"The oracle is by construction optimal over the same candidates, "
            f"so this is an implementation error, not a result."
        )
    if oracle.ndcg + 1e-12 < system.ndcg:
        raise HarnessError(
            f"depth {depth}: system nDCG {system.ndcg!r} exceeds the oracle's "
            f"{oracle.ndcg!r}. nDCG is computed wrongly."
        )

    # `reverse` is a permutation of the same list, so at full depth it must
    # retrieve exactly the same relevant documents. If it does not, the
    # metric is not counting what it claims to count.
    if depth >= len(ranking):
        reverse = judged["reverse"]
        if reverse.relevant_retrieved != system.relevant_retrieved:
            raise HarnessError(
                f"depth {depth}: reversing the ranking changed how many "
                f"relevant documents were retrieved "
                f"({system.relevant_retrieved} -> {reverse.relevant_retrieved}). "
                f"A permutation cannot change a set."
            )


def rank_ids(conn, index_hash: str, text: str, **kwargs) -> list[str]:
    value, _ = stage1.rank(
        conn=conn, query_terms=tokenize(text), index_hash=index_hash, **kwargs
    )
    return [stage1.Ranked(*row).node_id for row in value]


def run_quality(
    index_path: str,
    label_set: LabelSet,
    queries: list[dict],
    *,
    depths: list[int] | None = None,
) -> dict:
    """Evaluate every judged query, aggregate, and compare against controls.

    Only queries carrying judgements are evaluated. An unjudged query
    contributes nothing but a denominator, and averaging it in as a zero would
    understate quality for a reason that has nothing to do with retrieval.
    """
    spec = _spec()
    depths = depths or spec["depths"]
    conn = connect(index_path)
    index_hash = read_manifest(conn)["content_hash"]

    per_query: dict[str, dict] = {}
    for query in queries:
        labels = label_set.for_query(query["id"])
        if not labels:
            continue
        ranking = rank_ids(conn, index_hash, query["text"])
        per_query[query["id"]] = evaluate_query(ranking, labels, depths=depths)
    conn.close()

    if not per_query:
        return {
            "queries_evaluated": 0,
            "labels_hash": label_set.labels_hash,
            "index_hash": index_hash,
            "note": "no judged queries - nothing to report, which is not a "
                    "score of zero",
        }

    names = ["system", *controls_module.all_controls()]
    report: dict = {
        "queries_evaluated": len(per_query),
        "labels_hash": label_set.labels_hash,
        "index_hash": index_hash,
        "horizons": {qid: r["horizon"] for qid, r in per_query.items()},
        "depths": {},
    }

    for depth in depths:
        aggregated = {
            name: quality.aggregate(
                [r["depths"][depth][name] for r in per_query.values()]
            )
            for name in names
        }
        blind = [
            name for name in names
            if name not in ("system", "oracle")
        ]
        best_blind = max(aggregated[name]["ndcg"] for name in blind)
        margin = aggregated["system"]["ndcg"] - best_blind
        report["depths"][depth] = {
            "aggregate": aggregated,
            "best_blind_control": max(blind, key=lambda n: aggregated[n]["ndcg"]),
            "ndcg_margin_over_best_blind": margin,
            "required_margin": spec["min_ndcg_margin"],
            "clears_required_margin": margin >= spec["min_ndcg_margin"],
            "headroom_to_oracle": aggregated["oracle"]["ndcg"] - aggregated["system"]["ndcg"],
            # Recorded per depth because it is the difference between "the
            # ranking is poor" and "the documents were never retrieved".
            "structurally_reachable_by_advisory": [
                qid for qid, r in per_query.items() if depth > r["horizon"]
            ],
        }
    return report


def run_advisory_invariance(
    index_path: str,
    queries: list[dict],
    *,
    depths: list[int] | None = None,
) -> dict:
    """Prove, per query, the depth below which the advisory layer cannot act.

    Runs the full pipeline with the null provider and with stored vectors, and
    compares the merged output position by position. Everything above
    `|D|` must be byte-identical; that is `merge()`'s postcondition observed
    from outside rather than asserted from inside.

    Needs no labels, so it runs today. Its output is what stops M2.1 from
    reporting "the neural layer did not improve nDCG@5" as a finding when it
    is a structural impossibility.
    """
    from ..retrieval import merge as merge_module
    from ..retrieval import neural
    from ..retrieval.providers.null import NullProvider
    from ..retrieval.providers.stored_vectors import StoredVectorProvider
    from ..store import iter_nodes

    spec = _spec()
    depths = depths or spec["depths"]
    conn = connect(index_path)
    index_hash = read_manifest(conn)["content_hash"]
    known = {n.id for n in iter_nodes(conn)}

    rows = []
    for query in queries:
        deterministic = rank_ids(conn, index_hash, query["text"])
        merged_by_provider = {}
        for provider in (NullProvider(), StoredVectorProvider(conn)):
            advisory, _ = neural.propose_from_anchors(
                provider=provider, anchors=deterministic[:5], limit=100,
                provider_name=provider.name, index_hash=index_hash,
            )
            merged = merge_module.merge(
                deterministic=deterministic, advisory=advisory, known_ids=known,
            )
            merged_by_provider[provider.name] = [r.node_id for r in merged]

        orders = list(merged_by_provider.values())
        horizon = quality.advisory_horizon(deterministic)
        rows.append({
            "query_id": query["id"],
            "horizon": horizon,
            "prefix_identical": orders[0][:horizon] == orders[1][:horizon],
            "tail_lengths": {n: len(o) - horizon for n, o in merged_by_provider.items()},
            "depths_below_horizon": [d for d in depths if d <= horizon],
            "depths_advisory_can_reach": [d for d in depths if d > horizon],
        })
    conn.close()

    return {
        "queries": len(rows),
        "rows": rows,
        # The assertion surface: integers, and both must hold.
        "prefixes_identical": sum(1 for r in rows if r["prefix_identical"]),
        "prefixes_differing": sum(1 for r in rows if not r["prefix_identical"]),
        "queries_where_advisory_can_reach_any_depth": sum(
            1 for r in rows if r["depths_advisory_can_reach"]
        ),
    }
