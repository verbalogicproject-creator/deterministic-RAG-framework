#!/usr/bin/env python3
"""Turn a completed worksheet into `queries/labels.jsonl`.

Refuses to write a partial file, and says exactly which rows are missing. The
tempting alternative - write the graded rows and skip the rest - is the
failure mode `drf/bench/labels.py` already guards against from the other
direction: a missing judgement shrinks the recall denominator, which *raises*
measured recall. A tool whose partial output flatters the results is worse
than one that refuses.

Helper keys (`_query`, `_name`, ...) are stripped on the way out. They exist to
make the worksheet gradeable by hand; they are not judgements, and leaving
them in `labels.jsonl` would put unversioned prose into the file whose hash
every quality figure is bound to.

Stdlib only.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drf.bench import labels as labels_module   # noqa: E402
from drf.bench.repro import load_queries        # noqa: E402
from drf.store import connect, iter_nodes       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP = ("query_id", "node_id", "grade", "note")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worksheet", default=os.path.join(
        ROOT, "queries", "labels.worksheet.jsonl"))
    parser.add_argument("--out", default=os.path.join(
        ROOT, "queries", "labels.jsonl"))
    parser.add_argument("--index", default="index.db")
    parser.add_argument("--stratum", help="collect only one stratum, e.g. A_advisory")
    args = parser.parse_args()

    with open(args.worksheet) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.stratum:
        rows = [r for r in rows if r.get("_stratum") == args.stratum]

    ungraded = [r for r in rows if r.get("grade") is None]
    if ungraded:
        print(f"{len(ungraded)} of {len(rows)} rows are still ungraded. "
              f"Nothing written.")
        by_query: dict[str, int] = {}
        for row in ungraded:
            by_query[row["query_id"]] = by_query.get(row["query_id"], 0) + 1
        for query_id, count in sorted(by_query.items()):
            print(f"  {query_id}  {count} remaining")
        print("\nA partial file would shrink the recall denominator and raise "
              "the measured score. Grade them, or pass --stratum to collect a "
              "completed stratum on its own.")
        return 1

    conn = connect(args.index)
    known_nodes = {n.id for n in iter_nodes(conn)}
    conn.close()

    parsed = labels_module.parse(
        json.dumps({k: r[k] for k in KEEP if k in r}) for r in rows
    )
    label_set = labels_module.collate(
        parsed,
        known_query_ids={q["id"] for q in load_queries()},
        known_node_ids=known_nodes,
    )

    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps({k: row[k] for k in KEEP if k in row}) + "\n")

    print(f"wrote {label_set.count} judgements over "
          f"{len(label_set.judged_queries)} queries to {args.out}")
    print(f"  labels_hash  {label_set.labels_hash}")
    distribution: dict[int, int] = {}
    for grades in label_set.by_query.values():
        for grade in grades.values():
            distribution[grade] = distribution.get(grade, 0) + 1
    for grade in sorted(distribution):
        print(f"  grade {grade}  {distribution[grade]:4d}")
    print("\nNow: python3 tools/drf eval quality --index index.db")
    return 0


if __name__ == "__main__":
    sys.exit(main())
