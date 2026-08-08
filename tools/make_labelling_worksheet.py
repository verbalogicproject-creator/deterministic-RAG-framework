#!/usr/bin/env python3
"""Generate a stratified labelling worksheet for M2.1.

**The selection is the experiment.** Judging fifty arbitrary pairs produces
fifty judgements and settles nothing; judging the right fifty settles a stated
question. So this script does not sample - it stratifies, and each stratum is
independently complete so that stopping after any one of them still leaves a
coherent measurement rather than a partial one.

The strata come from a measurement, not from taste. Running every
`affects_ranking` setting against its probe over the 23-query set shows which
queries each decision actually depends on:

    ranking.b         13 queries    length normalisation
    ranking.k1         9 queries    term saturation
    graph.max_depth    2 queries    q02 and e06
    graph.seed_count   2 queries    q02 and e06

and the advisory horizon shows which queries the neural layer can reach at all.

**The corpus does not let one query serve both purposes.** Low-`|D|` queries
are the only ones the advisory layer can act on, and they are inert to every
ranking parameter - too few candidates for a weight to reorder. High-`|D|`
queries are parameter-sensitive and advisory-mute. The strata are therefore
disjoint by construction, which is a fact about this corpus worth knowing
before spending an afternoon on judgements.

**Pooling.** Candidates come from D *and* from the advisory provider, which is
standard pooled assessment and is load-bearing here: a pool drawn only from D
could never contain a document that lexical retrieval missed, so recall would
be measured against a denominator that excluded the exact thing the advisory
layer exists to find. The pool depth is recorded per stratum, because
unjudged documents count as grade 0 and a shallow pool silently flatters
recall.

Ungraded rows are written as `"grade": null`, which is **invalid** to the
label parser. That is deliberate: a template pre-filled with 0 would turn a
forgotten row into a confident judgement of "irrelevant".

Stdlib only.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drf.bench.evaluate import rank_ids                      # noqa: E402
from drf.bench.repro import load_queries                     # noqa: E402
from drf.retrieval import neural                             # noqa: E402
from drf.retrieval.providers.stored_vectors import StoredVectorProvider  # noqa: E402
from drf.store import connect, iter_nodes, read_manifest     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Each stratum: the queries, how deep to pool from D, how deep from the
# advisory provider, and the question the judgements are meant to settle.
STRATA = [
    {
        "name": "A_advisory",
        "question": "Does the advisory layer retrieve relevant documents that "
                    "lexical search missed entirely? This is the ONLY question "
                    "the neural layer can be asked - everything at k <= |D| is "
                    "structurally identical with it on and off.",
        "queries": ["q08", "q12", "q06", "q13", "q15", "q07", "q14"],
        "why_these": "|D| <= 7, so evaluated depths 10 and 20 fall beyond the "
                     "horizon and the advisory layer can actually occupy them.",
        "pool_deterministic": 20,
        "pool_advisory": 3,
        "complete_on_its_own": True,
    },
    {
        "name": "B_length_normalisation",
        "question": "Does BM25 length normalisation improve relevance, or only "
                    "change it? M1 measured 17.0% discordance against the flat "
                    "scorer and that longer documents win 9/0 without it - but "
                    "'different' is not 'better' without labels.",
        "queries": ["q04", "q10", "e07", "e08"],
        "why_these": "These four reorder under ranking.b and NOT under "
                     "ranking.k1, so they isolate length normalisation from "
                     "term saturation. The other nine move under both and "
                     "cannot separate the two effects.",
        "pool_deterministic": 10,
        "pool_advisory": 0,
        "complete_on_its_own": True,
    },
    {
        "name": "C_graph",
        "question": "Does the graph layer earn its place in the sort key?",
        "queries": ["q02"],
        "why_these": "ONLY q02 and e06 reorder under either graph setting, and "
                     "e06 is a synthetic 13-term edge case. So this stratum has "
                     "a real sample size of ONE. Judge it for the qualitative "
                     "read, but see LABELLING.md - fifty labels cannot settle "
                     "this decision, and pretending otherwise would be the "
                     "failure mode this project exists to remove.",
        "pool_deterministic": 20,
        "pool_advisory": 0,
        "complete_on_its_own": False,
    },
]


def build_rows(index_path: str) -> list[dict]:
    conn = connect(index_path)
    index_hash = read_manifest(conn)["content_hash"]
    nodes = {n.id: n for n in iter_nodes(conn)}
    queries = {q["id"]: q for q in load_queries()}
    provider = StoredVectorProvider(conn)

    rows: list[dict] = []
    for stratum in STRATA:
        for query_id in stratum["queries"]:
            query = queries[query_id]
            deterministic = rank_ids(conn, index_hash, query["text"])

            pooled = [
                (node_id, f"D{rank}")
                for rank, node_id in enumerate(
                    deterministic[:stratum["pool_deterministic"]], start=1
                )
            ]
            if stratum["pool_advisory"]:
                advisory, _ = neural.propose_from_anchors(
                    provider=provider, anchors=deterministic[:5],
                    limit=stratum["pool_advisory"] + len(deterministic),
                    provider_name=provider.name, index_hash=index_hash,
                )
                # `unwrap` is forbidden outside merge, and rightly so. Re-deriving
                # the proposals here would need the same allowlist exemption, so
                # instead the provider is asked directly - this is a labelling
                # tool, not the retrieval path, and it has no authority to leak.
                proposals = [
                    node_id
                    for node_id in provider.propose(
                        anchors=deterministic[:5], limit=200
                    )
                    if node_id not in set(deterministic)
                ][:stratum["pool_advisory"]]
                pooled += [
                    (node_id, f"A{rank}")
                    for rank, node_id in enumerate(proposals, start=1)
                ]

            for node_id, origin in pooled:
                node = nodes[node_id]
                rows.append({
                    "query_id": query_id,
                    "node_id": node_id,
                    "grade": None,
                    "note": "",
                    # Helper keys. The label parser ignores anything it does not
                    # need, so these travel with the row and make it gradeable
                    # without cross-referencing another file.
                    "_query": query["text"],
                    "_stratum": stratum["name"],
                    "_origin": origin,
                    "_horizon": len(deterministic),
                    "_name": node.name,
                    "_type": node.type,
                    "_description": (node.description or "")[:220],
                })
    conn.close()
    return rows


def render_markdown(rows: list[dict]) -> str:
    """A readable companion to the JSONL. Grading happens in the JSONL.

    Two views of one artefact rather than two artefacts: the markdown is
    regenerated from the same rows, so it cannot drift into being a second,
    disagreeing worksheet. Nothing reads it back - it is for deciding, not for
    recording.
    """
    out = [
        "# Labelling worksheet",
        "",
        "Read here, grade in `queries/labels.worksheet.jsonl`, then run",
        "`python3 tools/labels_collect.py`.",
        "",
        "Grades: **0** irrelevant · **1** marginal · **2** relevant · "
        "**3** vital. Recall counts 2 and above.",
        "",
        "`D`*n* = authoritative rank *n*.  `A`*n* = advisory proposal, "
        "absent from D entirely.",
        "",
    ]
    for stratum in STRATA:
        subset = [r for r in rows if r["_stratum"] == stratum["name"]]
        if not subset:
            continue
        out += [
            f"## Stratum {stratum['name']} — {len(subset)} judgements",
            "",
            f"**Question.** {stratum['question']}",
            "",
            f"**Why these queries.** {stratum['why_these']}",
            "",
        ]
        if not stratum["complete_on_its_own"]:
            out += ["> ⚠️ Not complete on its own — see `LABELLING.md`.", ""]
        current = None
        for row in subset:
            if row["query_id"] != current:
                current = row["query_id"]
                out += [
                    "",
                    f"### {row['query_id']} — `{row['_query']}`  "
                    f"(|D| = {row['_horizon']})",
                    "",
                    "| | grade | name | type | description |",
                    "|---|---|---|---|---|",
                ]
            description = row["_description"].replace("|", "\\|")
            out.append(
                f"| `{row['_origin']}` |  | **{row['_name']}** | "
                f"{row['_type']} | {description} |"
            )
        out.append("")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", default="index.db")
    parser.add_argument("--out", default=os.path.join(
        ROOT, "queries", "labels.worksheet.jsonl"))
    parser.add_argument("--markdown", default=os.path.join(
        ROOT, "queries", "labels.worksheet.md"))
    args = parser.parse_args()

    rows = build_rows(args.index)
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with open(args.markdown, "w") as handle:
        handle.write(render_markdown(rows))

    print(f"wrote {len(rows)} rows to {args.out}")
    for stratum in STRATA:
        count = sum(1 for r in rows if r["_stratum"] == stratum["name"])
        flag = "" if stratum["complete_on_its_own"] else "   [n=1, see LABELLING.md]"
        print(f"  {stratum['name']:24s} {count:4d} judgements  "
              f"{len(stratum['queries'])} queries{flag}")
    print("\nGrade in place: set \"grade\" to 0/1/2/3. A null grade is invalid,")
    print("so a forgotten row fails loudly instead of becoming a silent 0.")
    print("Then: python3 tools/labels_collect.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
