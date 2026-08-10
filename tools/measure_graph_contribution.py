#!/usr/bin/env python3
"""Does the graph layer earn its place in the sort key?

The question was deferred through all of M1 because it looked like it needed
relevance labels. It does not. Two measurements settle it, and neither reads a
judgement:

**1. Ablation, not parameter sensitivity.** M1.6 measured that nudging
`graph.max_depth` 2->3 reorders 2 of 23 queries and concluded the layer was
"live but weak". That measured a *nudge*. Removing the signal entirely is the
question actually being asked, and it is a different experiment - a layer can
be insensitive to its parameters while still doing a great deal of work.

**2. Nuisance correlation** (mud-detection layer 3). The tool this borrows from
exists because a `parent-boost` signal - propagate a matched child's score up to
its parent - passed an LLM compatibility check and then regressed held-out
recall, having learned to promote *well-connected* items rather than relevant
ones. `best_depth` from BFS expansion is structurally the same shape of signal,
so the same pathology is worth ruling in or out before trusting it.

Run:

    python3 tools/measure_graph_contribution.py --index index.db

Requires `mud_detection` on the path for layers 2-3; the ablation runs without
it. Nothing here is vendored - drf takes no dependency on that package, and
this tool degrades to the ablation alone if it is absent.

Stdlib only (plus the optional import above).
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drf.bench.repro import load_queries                       # noqa: E402
from drf.retrieval import bm25, graph, stage1                   # noqa: E402
from drf.retrieval.stage1 import (graph_only_candidates,        # noqa: E402
                                  rank_candidates)
from drf.retrieval.tokenize import tokenize                     # noqa: E402
from drf.store import (connect, corpus_totals, df_for_terms,    # noqa: E402
                       doc_lengths, iter_edges, iter_nodes,
                       postings_for_terms, read_manifest)


def _parts(conn, text: str):
    """BM25 scores, the depth map, and document lengths for one query."""
    terms = tokenize(text)
    n_docs, total_len = corpus_totals(conn)
    doc_lens = doc_lengths(conn)
    scored = bm25.score_documents(
        postings=postings_for_terms(conn, terms),
        dfs=df_for_terms(conn, terms), doc_lens=doc_lens,
        n_docs=n_docs, total_len=total_len, k1=bm25.K1, b=bm25.B,
    )
    seeds = stage1.select_seeds(scored, stage1.DEFAULT_SEED_COUNT)
    depths = graph.expand(conn, seeds, stage1.DEFAULT_MAX_DEPTH) if seeds else {}
    return scored, depths, doc_lens


def rank(conn, text: str, *, with_graph: bool) -> list[str]:
    """Stage 1, with the graph contribution present or ablated.

    Ablation is done by withholding the depth map rather than by editing the
    sort key, so the comparison isolates the *signal* and leaves the ordering
    machinery - including the injective tiebreak chain - byte-identical. Change
    the key instead and you measure two things at once.
    """
    terms = tokenize(text)
    n_docs, total_len = corpus_totals(conn)
    scored = bm25.score_documents(
        postings=postings_for_terms(conn, terms),
        dfs=df_for_terms(conn, terms),
        doc_lens=doc_lengths(conn),
        n_docs=n_docs, total_len=total_len, k1=bm25.K1, b=bm25.B,
    )
    depths: dict[str, int] = {}
    if with_graph:
        seeds = stage1.select_seeds(scored, stage1.DEFAULT_SEED_COUNT)
        depths = graph.expand(conn, seeds, stage1.DEFAULT_MAX_DEPTH) if seeds else {}
    return [r.node_id for r in stage1.rank_candidates(scored, depths)]


def unsafe_degree_ordered(conn, text: str, degrees: dict[str, int]) -> list[str]:
    """The **positive control**: a deliberately unsafe graph signal.

    A screen that only ever returns CLEAN proves nothing - the same lesson the
    chaos control taught in M1.6, where every reproducibility metric scored 1.0
    and only a deliberately broken pipeline showed the harness could report
    failure. Layer 3 returned CLEAN for drf's graph layer twice. Without a
    signal it is *known* to reject, that verdict is a claim about the screen.

    So: identical architecture, identical admission, identical sort key. The
    only change is that the admitted tail is ordered by **degree** instead of
    by `best_depth`. That is the parent-boost pathology reproduced inside drf's
    own ordering - promote the well-connected rather than the relevant - and it
    is the narrowest possible edit that makes the signal unsafe.

    Note what this construction also demonstrates: the pathology can only be
    built *inside the admitted tail*. The sort key opens with `-bm25_q`, so no
    amount of connectivity can lift an unmatched document above a matched one.
    drf's architecture confines the failure mode to the one region where every
    document already scored zero.
    """
    scored, depths, doc_lens = _parts(conn, text)
    admitted = graph_only_candidates(scored, depths, doc_lens)
    # High degree -> low pseudo-depth -> sorts earlier, since best_depth is the
    # third key component and ascends.
    rigged = {**depths,
              **{s.node_id: 1000 - degrees.get(s.node_id, 0) for s in admitted}}
    return [r.node_id for r in rank_candidates(scored + admitted, rigged)]


def degree_map(conn) -> dict[str, int]:
    """Incident-edge count per node. Undirected, matching `store.neighbours`."""
    counts: Counter = Counter()
    for edge in iter_edges(conn):
        counts[edge.from_id] += 1
        counts[edge.to_id] += 1
    return {node.id: counts.get(node.id, 0) for node in iter_nodes(conn)}


def ablate(conn, queries: list[dict]) -> dict:
    """Where, if anywhere, does the graph layer change what a user sees?"""
    rows = []
    for query in queries:
        without = rank(conn, query["text"], with_graph=False)
        with_it = rank(conn, query["text"], with_graph=True)
        first = next(
            (i for i, (a, b) in enumerate(zip(without, with_it)) if a != b), None
        )
        rows.append({
            "query_id": query["id"],
            "horizon": len(with_it),
            "changed": without != with_it,
            # 1-based, so it reads against a result list. None when nothing moved.
            "first_changed_rank": None if first is None else first + 1,
            "same_item_set": set(without) == set(with_it),
        })
    changed = [r for r in rows if r["changed"]]
    depths_affected = [
        r["first_changed_rank"] for r in changed if r["first_changed_rank"]
    ]
    return {
        "queries": len(rows),
        "queries_changed": len(changed),
        "shallowest_change": min(depths_affected) if depths_affected else None,
        "item_set_ever_changes": any(not r["same_item_set"] for r in rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", default="index.db")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    conn = connect(args.index)
    read_manifest(conn)
    queries = load_queries()
    base = {q["id"]: rank(conn, q["text"], with_graph=False) for q in queries}
    candidate = {q["id"]: rank(conn, q["text"], with_graph=True) for q in queries}
    report = {"ablation": ablate(conn, queries)}

    try:
        from mud_detection import correlate, interference
    except ImportError:
        report["layers"] = "mud_detection not importable - ablation only"
    else:
        degrees = degree_map(conn)
        redundancy = correlate.assess(candidate, base)
        nuisance = interference.assess(candidate, base, degrees, name="degree")
        per_query = correlate.per_query_rho(candidate, base)
        non_empty = {k: v for k, v in per_query.items() if base[k]}
        control = {q["id"]: unsafe_degree_ordered(conn, q["text"], degrees)
                   for q in queries}
        admitted = {
            q["id"]: [r.node_id for r in rank_candidates(
                (lambda p: p[0] + graph_only_candidates(*p))(_parts(conn, q["text"])),
                _parts(conn, q["text"])[1])]
            for q in queries
        }
        control_report = interference.assess(control, base, degrees, name="degree")
        admitted_report = interference.assess(admitted, base, degrees, name="degree")
        report["layers"] = {
            "positive_control_verdict": control_report.verdict,
            "positive_control_rho": control_report.rho,
            "admitted_tail_verdict": admitted_report.verdict,
            "admitted_tail_rho": admitted_report.rho,
            "screen_separates_safe_from_unsafe":
                control_report.verdict != admitted_report.verdict,
            "redundancy_verdict": "REDUNDANT" if redundancy.redundant else "COMPLEMENTARY",
            "redundancy_why": redundancy.why,
            "rho_mean_all": redundancy.rho_mean,
            "rho_mean_non_empty": (
                sum(non_empty.values()) / len(non_empty) if non_empty else None
            ),
            "queries_rho_exactly_1": sum(1 for v in non_empty.values() if v == 1.0),
            "nuisance_verdict": nuisance.verdict,
            "nuisance_rho": nuisance.rho,
            "nuisance_why": nuisance.why,
            "promoted_degree_ratio": nuisance.top_promoted_nuisance_ratio,
        }
    conn.close()

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    ablation = report["ablation"]
    print("=== ablation: the graph signal withheld entirely ===")
    print(f"  queries whose output changes   {ablation['queries_changed']}"
          f"/{ablation['queries']}")
    print(f"  shallowest rank ever affected  {ablation['shallowest_change']}")
    print(f"  candidate set ever changes     {ablation['item_set_ever_changes']}"
          "   (a tiebreak cannot add documents)")
    for row in ablation["rows"]:
        if row["changed"]:
            print(f"    {row['query_id']:5s} |D|={row['horizon']:3d}  "
                  f"first difference at rank {row['first_changed_rank']}")

    layers = report.get("layers")
    if isinstance(layers, dict):
        print("\n=== mud-detection layers 2 and 3 ===")
        print(f"  redundancy   {layers['redundancy_verdict']}")
        print(f"               {layers['redundancy_why']}")
        print(f"               rho mean {layers['rho_mean_all']:.4f} over all queries, "
              f"{layers['rho_mean_non_empty']:.4f} over non-empty")
        print(f"               {layers['queries_rho_exactly_1']} queries rank-identical")
        print(f"  nuisance     {layers['nuisance_verdict']}  (degree)")
        print(f"               {layers['nuisance_why']}")
        print("\n  positive control - the same signal made deliberately unsafe")
        print("  (admitted tail ordered by degree instead of depth):")
        print(f"    drf tail      rho {layers['admitted_tail_rho']:+.4f}  "
              f"{layers['admitted_tail_verdict']}")
        print(f"    UNSAFE tail   rho {layers['positive_control_rho']:+.4f}  "
              f"{layers['positive_control_verdict']}")
        print(f"    screen separates them: "
              f"{layers['screen_separates_safe_from_unsafe']}")
        print("    Without this, a CLEAN verdict is a claim about the screen.")
        print("\n  Read the redundancy verdict with care: this signal is a TIEBREAK, so")
        print("  it cannot introduce a document and complementarity is 0 by construction.")
        print("  The ablation above is the load-bearing evidence, not layer 2.")
    else:
        print(f"\n  {layers}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
