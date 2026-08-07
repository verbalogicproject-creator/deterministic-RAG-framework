#!/usr/bin/env python3
"""Producer for the length-normalisation figures quoted in STATE.md.

The engine this framework replaces scored `idf*(tf*k1)/(tf+k1)` with `k1=2.5`
and no `b` or `avgdl` - BM25 with length normalisation switched off. STATE.md
asserted that this let "long padded entities win by default". That was
inherited from reading the code; this script measures it.

    python3 tools/measure_length_norm.py --index index.db

**Every metric here is tiebreak-free, and that correction matters.** The first
version of this script ranked with `sorted(key=(-score, node_id))` and reported
"top-1 changed for 9 of 15 queries" and "26.2% discordant pairs". Both figures
turned out to depend on the tiebreak: 7 of 15 queries have exact ties in a top
set, and breaking them by *lowest* node id gives 9 while *highest* gives 4.
Those numbers described BM25 plus an arbitrary convention, not BM25 - the same
class of defect this project catalogued in the old engine ("no tiebreak on any
sort"), reproduced in its own measurement.

What replaced them:

  * **top sets, not top-1** - every document sharing the maximum score. A set
    needs no convention to identify.
  * **strictly-ordered pairs only** - a pair tied under either setting has no
    defined order without a tiebreak, so it is counted separately as
    tie-affected rather than silently resolved.
  * **direction as a count, not a mean** - `shorter == 0` is the claim. A mean
    can hide counterexamples; an exact zero cannot.

Deliberately standalone: it reimplements BM25 in ~15 lines rather than
importing `drf.retrieval.bm25`, so it stays an *independent* check. A
measurement that calls the code under test cannot contradict it.

Stdlib only.
"""

import argparse
import collections
import itertools
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drf.store import connect, iter_nodes  # noqa: E402

# Drawn from the corpus vocabulary: topics a user of this knowledge graph
# would plausibly search for. Not tuned to produce a particular result.
QUERIES = [
    "prompt caching", "tool use", "extended thinking", "rag retrieval",
    "agent orchestration", "streaming", "pdf vision", "citations",
    "batch processing", "embeddings semantic search",
    "json mode structured output", "summarization", "classification",
    "sub agents", "memory",
]

K1 = 1.2
B_NORMALISED = 0.75
B_OLD_ENGINE = 0.0


def tokenize(text: str) -> list[str]:
    """Deliberately naive - this is a reference, not the indexer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def score_all(docs, df, avgdl, n_docs, query, b):
    scores = {}
    for node_id, tokens in docs.items():
        tf = collections.Counter(tokens)
        total = 0.0
        for term in query:
            if tf[term] == 0:
                continue
            idf = math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1)
            denom = tf[term] + K1 * (1 - b + b * len(tokens) / avgdl)
            total += idf * tf[term] * (K1 + 1) / denom
        if total > 0:
            scores[node_id] = total
    return scores


def top_set(scores: dict) -> set:
    """Every document sharing the maximum score. Needs no tiebreak."""
    best = max(scores.values())
    return {k for k, v in scores.items() if v == best}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", default="index.db")
    parser.add_argument("--depth", type=int, default=10)
    args = parser.parse_args()

    conn = connect(args.index)
    nodes = list(iter_nodes(conn))
    conn.close()

    docs = {n.id: tokenize(f"{n.name} {n.description}") for n in nodes}
    n_docs = len(docs)
    avgdl = sum(len(d) for d in docs.values()) / n_docs
    df = collections.Counter()
    for tokens in docs.values():
        df.update(set(tokens))

    lengths = sorted(len(d) for d in docs.values())
    print(f"corpus  N={n_docs}  avgdl={avgdl:.1f}  min={lengths[0]}  "
          f"max={lengths[-1]}  median={statistics.median(lengths):.0f}")
    print()
    print(f"{'query':30s} {'|top| .75':>10s} {'|top| 0':>8s} "
          f"{'disjoint':>9s} {'mean len .75 -> 0':>20s}")

    tied = disjoint = longer = shorter = equal = 0
    strict_pairs = strict_disc = tie_affected = 0
    means_norm, means_flat = [], []

    for text in QUERIES:
        query = tokenize(text)
        norm = score_all(docs, df, avgdl, n_docs, query, B_NORMALISED)
        flat = score_all(docs, df, avgdl, n_docs, query, B_OLD_ENGINE)
        if not norm:
            print(f"{text:30s}   (no hits)")
            continue

        top_norm, top_flat = top_set(norm), top_set(flat)
        if len(top_norm) > 1 or len(top_flat) > 1:
            tied += 1
        is_disjoint = top_norm.isdisjoint(top_flat)
        disjoint += is_disjoint

        mean_norm = statistics.mean(len(docs[x]) for x in top_norm)
        mean_flat = statistics.mean(len(docs[x]) for x in top_flat)
        means_norm.append(mean_norm)
        means_flat.append(mean_flat)
        if mean_flat > mean_norm:
            longer += 1
        elif mean_flat < mean_norm:
            shorter += 1
        else:
            equal += 1

        # Depth cut by score, including everything tied at the boundary, so
        # the head itself is not chosen by a convention.
        ranked = sorted(norm, key=lambda k: -norm[k])
        if len(ranked) > args.depth:
            cut = norm[ranked[args.depth - 1]]
            head = [k for k in ranked if norm[k] >= cut]
        else:
            head = ranked
        for x, y in itertools.combinations(sorted(head), 2):
            if norm[x] == norm[y] or flat[x] == flat[y]:
                tie_affected += 1
                continue
            strict_pairs += 1
            if (norm[x] > norm[y]) != (flat[x] > flat[y]):
                strict_disc += 1

        print(f"{text:30s} {len(top_norm):10d} {len(top_flat):8d} "
              f"{str(is_disjoint):>9s} {mean_norm:9.1f} -> {mean_flat:6.1f}")

    total_pairs = strict_pairs + tie_affected
    print()
    print(f"queries with a tie in a top set : {tied}/{len(QUERIES)}")
    print(f"top sets disjoint               : {disjoint}/{len(QUERIES)}")
    print(f"strictly-ordered pairs @{args.depth}      : {strict_pairs}")
    print(f"  discordant                    : {strict_disc} "
          f"= {100 * strict_disc / strict_pairs:.1f}%")
    print(f"pairs tie-affected              : {tie_affected} "
          f"= {100 * tie_affected / total_pairs:.1f}% of all pairs "
          f"(order undefined without a tiebreak)")
    print(f"mean top-set length  b={B_NORMALISED}      : "
          f"{statistics.mean(means_norm):.1f} tokens")
    print(f"mean top-set length  b={B_OLD_ENGINE}       : "
          f"{statistics.mean(means_flat):.1f} tokens")
    print(f"direction longer/shorter/equal  : {longer}/{shorter}/{equal}")
    print()
    print("The claim under test is the *direction*: disabling length "
          "normalisation should favour longer results. It is supported only "
          f"if 'shorter' is 0 - measured {shorter}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
