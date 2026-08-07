"""The build manifest: what this index contains, and what it dropped.

A manifest has two halves, and the split is the whole point.

**`content`** - a pure function of the source data and the code version.
Hashed into `content_hash`. Two builds from the same source on different
machines, at different times, into different directories, produce the same
`content_hash`. This is the value that reproducibility is asserted on.

**`provenance`** - build time, source path, interpreter version. Useful for
debugging, deliberately *excluded* from the hash. The moment a wall clock or
an absolute path enters the hash, reproducibility becomes untestable. The old
project's databases carry `DEFAULT CURRENT_TIMESTAMP` on six tables for
exactly this reason: the information was worth keeping, and nobody separated
it from the data.

Counts are always derived with `len()` over the actual records. No count in
this file is ever written as a literal, and none is passed in from a caller
that "knows" the answer - `tests/test_ingest.py` cross-checks every one of
them against `SELECT count(*)` on the built database.

Stdlib only.
"""

from ..hashing import canonical_json, sha256_value
from ..version import (
    ID_SCHEMA_VERSION,
    MANIFEST_VERSION,
    PARSER_VERSION,
    RANKER_VERSION,
)
from .normalize import Normalized


def _node_digest(node) -> list:
    return [node.id, node.type, node.name, node.description,
            node.source, node.source_ref, canonical_json(node.metadata)]


def _edge_digest(edge) -> list:
    return [edge.id, edge.from_id, edge.to_id, edge.type,
            edge.weight_q, canonical_json(edge.metadata)]


def _embedding_digest(emb) -> list:
    # Vectors are hashed by content, not by reference; hex keeps the manifest
    # JSON-serialisable without changing what is covered.
    return [emb.node_id, emb.model, emb.dim, sha256_value(emb.vector.hex())]


def build_content(normalized: Normalized, corpus: str) -> dict:
    """The reproducible half of the manifest.

    Full records are hashed, not just their ids. Node content is already
    covered by `node_id` (which is a hash of type, name, description, source),
    but `edge_id` covers only (from, to, type) - edge weight and metadata are
    *not* in the id. Hashing ids alone would let a change in the collapse rule
    silently produce a different index with an identical `content_hash`.
    """
    nodes = list(normalized.nodes)
    edges = list(normalized.edges)
    embeddings = list(normalized.embeddings)
    dropped = [d._asdict() for d in normalized.dropped]

    content = {
        "manifest_version": MANIFEST_VERSION,
        "corpus": corpus,
        "versions": {
            "parser": PARSER_VERSION,
            "ranker": RANKER_VERSION,
            "id_schema": ID_SCHEMA_VERSION,
        },
        "counts": {
            "nodes": len(nodes),
            "edges": len(edges),
            "embeddings": len(embeddings),
            "dropped": len(dropped),
            "collapsed_groups": len(normalized.collapsed),
        },
        "digests": {
            "nodes": sha256_value([_node_digest(n) for n in nodes]),
            "edges": sha256_value([_edge_digest(e) for e in edges]),
            "embeddings": sha256_value([_embedding_digest(e) for e in embeddings]),
        },
        # Recorded in full, not merely counted. A dropped record that is only
        # tallied cannot be investigated or fixed; four dangling edges are
        # small enough to name, and naming them is what makes the drop
        # auditable rather than merely disclosed.
        "dropped": sorted(
            dropped, key=lambda d: canonical_json([d["kind"], d["reason"], d["detail"]])
        ),
        "collapsed": sorted(normalized.collapsed, key=lambda c: c["edge_id"]),
        "embedding_models": sorted({e.model for e in embeddings}),
        "embedding_dims": sorted({e.dim for e in embeddings}),
    }
    return content


def lexical_content(*, doc_lens: dict, postings: list, dfs: dict) -> dict:
    """The lexical index's contribution to `content_hash`.

    The postings digest is included even though postings are *derived* from
    nodes, which are already hashed. The redundancy is the point: it makes the
    tokenizer part of the index identity, so an accidental change to
    tokenisation alters `content_hash` even if nobody remembered to bump
    `PARSER_VERSION`. A version constant records intent; a digest records fact.

    `total_length` is stored as an exact integer rather than `avgdl` as a
    float. `avgdl` is derived at use time as `total/n`, so the float that
    scoring sees is produced by one division of two exact integers on every
    run, instead of being round-tripped through a stored decimal.
    """
    return {
        "counts": {
            "documents": len(doc_lens),
            "terms": len(dfs),
            "postings": len(postings),
        },
        "total_length": sum(doc_lens.values()),
        "digests": {
            "doc_stats": sha256_value(sorted(doc_lens.items())),
            "terms": sha256_value(sorted(dfs.items())),
            "postings": sha256_value(
                [[p.term, p.node_id, p.tf] for p in sorted(postings)]
            ),
        },
    }


def build_manifest(
    normalized: Normalized,
    *,
    corpus: str,
    source_fingerprint: dict,
    lexical: dict | None = None,
    provenance: dict | None = None,
) -> dict:
    """Assemble the full manifest with its `content_hash`.

    `source_fingerprint` describes what was read and lives inside `content`:
    it is a property of the input data, so it belongs in the hash. The source
    *path* does not, and is kept in `provenance`.
    """
    content = build_content(normalized, corpus)
    content["source"] = source_fingerprint
    if lexical is not None:
        content["lexical"] = lexical_content(**lexical)
    return {
        "content": content,
        "content_hash": sha256_value(content),
        "provenance": provenance or {},
    }


def reconcile(normalized: Normalized, source_fingerprint: dict) -> dict:
    """Account for every source row exactly once.

    This is the arithmetic that makes the drop and collapse counts trustworthy.
    Because orphan-drop happens before collapse, and because the two defect
    sets were measured not to intersect, the following identity holds exactly:

        edges_read = edges_written + collapsed_variants_removed + edges_dropped

    If it ever fails, some row was counted twice or lost silently, and the
    build must not be trusted. `build.py` asserts it on every run rather than
    only in tests, because a reconciliation that is only checked in CI is not
    checking the build that shipped.
    """
    edges_dropped = len([d for d in normalized.dropped if d.kind == "edge"])
    variants_removed = sum(c["variants"] - 1 for c in normalized.collapsed)
    accounted = len(normalized.edges) + variants_removed + edges_dropped
    return {
        "edges_read": source_fingerprint["edges_read"],
        "edges_written": len(normalized.edges),
        "collapsed_variants_removed": variants_removed,
        "edges_dropped": edges_dropped,
        "accounted": accounted,
        "balanced": accounted == source_fingerprint["edges_read"],
    }
