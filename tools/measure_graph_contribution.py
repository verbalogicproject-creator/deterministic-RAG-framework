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
from drf.retrieval.tokenize import tokenize                     # noqa: E402
from drf.store import (connect, corpus_totals, df_for_terms,    # noqa: E402
                       doc_lengths, iter_edges, iter_nodes,
                       postings_for_terms, read_manifest)


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
        report["layers"] = {
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
        print("\n  Read the redundancy verdict with care: this signal is a TIEBREAK, so")
        print("  it cannot introduce a document and complementarity is 0 by construction.")
        print("  The ablation above is the load-bearing evidence, not layer 2.")
    else:
        print(f"\n  {layers}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
