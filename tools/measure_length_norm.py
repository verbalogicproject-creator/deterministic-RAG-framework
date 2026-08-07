#!/usr/bin/env python3
"""Producer for the length-normalisation figures quoted in STATE.md.

The old retrieval engine's scorer had no `b` and no `avgdl`, so it was BM25
with length normalisation switched off. STATE.md asserted that this let "long
padded entities win by default". That was inherited from reading the code, not
measured - this script measures it, so the claim has a producer.

    python3 tools/measure_length_norm.py --index index.db

What it reports, on the real corpus rather than a synthetic reference:

  * discordant pairs @10 between b=0.75 and b=0.0 rankings - how much of the
    ranking the parameter actually moves
  * how often top-1 changes
  * the mean document length of the top-1 result under each setting, and
    whether b=0 favours longer documents *monotonically* (the specific claim)

Deliberately standalone and self-contained: it reimplements Okapi BM25 in ~15
lines rather than importing `drf.retrieval.bm25`. That keeps it usable as an
*independent* check of the implementation once M1.2 lands - a measurement that
calls the code under test cannot contradict it.

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

# Drawn from the corpus vocabulary: topics a user of this knowledge graph would
# plausibly search for. Not tuned to produce a particular result.
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
    print(f"corpus  N={n_docs}  avgdl={avgdl:.1f}  "
          f"min={lengths[0]}  max={lengths[-1]}  "
          f"median={statistics.median(lengths):.0f}")
    print()
    print(f"{'query':32s} {'hits':>5s} {'disc@' + str(args.depth):>9s} {'top1 same':>10s}")

    discordant = pairs_total = 0
    top1_lengths = {B_NORMALISED: [], B_OLD_ENGINE: []}
    longer = shorter = same = 0

    for text in QUERIES:
        query = tokenize(text)
        norm = score_all(docs, df, avgdl, n_docs, query, B_NORMALISED)
        old = score_all(docs, df, avgdl, n_docs, query, B_OLD_ENGINE)
        if not norm:
            print(f"{text:32s} {0:5d}   (no hits)")
            continue

        rank_norm = sorted(norm, key=lambda k: (-norm[k], k))
        rank_old = sorted(old, key=lambda k: (-old[k], k))
        pos_norm = {x: i for i, x in enumerate(rank_norm)}
        pos_old = {x: i for i, x in enumerate(rank_old)}

        head = rank_norm[:min(args.depth, len(rank_norm))]
        disc = sum(
            1 for x, y in itertools.combinations(head, 2)
            if (pos_norm[x] < pos_norm[y]) != (pos_old[x] < pos_old[y])
        )
        pairs = len(head) * (len(head) - 1) // 2
        discordant += disc
        pairs_total += pairs

        top1_lengths[B_NORMALISED].append(len(docs[rank_norm[0]]))
        top1_lengths[B_OLD_ENGINE].append(len(docs[rank_old[0]]))
        if len(docs[rank_old[0]]) > len(docs[rank_norm[0]]):
            longer += 1
        elif len(docs[rank_old[0]]) < len(docs[rank_norm[0]]):
            shorter += 1
        else:
            same += 1

        print(f"{text:32s} {len(norm):5d} {disc:4d}/{pairs:<4d} "
              f"{str(rank_norm[0] == rank_old[0]):>10s}")

    print()
    print(f"discordant pairs @{args.depth}: {discordant}/{pairs_total} "
          f"= {100 * discordant / pairs_total:.1f}%")
    print(f"mean top-1 length  b={B_NORMALISED}: "
          f"{statistics.mean(top1_lengths[B_NORMALISED]):.1f} tokens")
    print(f"mean top-1 length  b={B_OLD_ENGINE}: "
          f"{statistics.mean(top1_lengths[B_OLD_ENGINE]):.1f} tokens")
    print(f"b=0 top-1 longer/shorter/same: {longer}/{shorter}/{same}")
    print()
    print("The claim under test is the *direction*: disabling length "
          "normalisation should favour longer documents. It is supported only "
          f"if 'shorter' is 0 - measured {shorter}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
