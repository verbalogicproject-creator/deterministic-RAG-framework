"""Read the source knowledge graph.

This module is deliberately dumb: it reads rows and hands them on. It applies
no repairs, no filtering, and no interpretation. All of that lives in
`normalize.py`, where it can be tested and where every dropped record is
accounted for in the manifest.

Two rules govern every query here:

1. **Explicit ORDER BY on a column the source guarantees unique.** The source
   tables carry an implicit rowid and the builder that produced them inserted
   in an order we do not control. Reading without ORDER BY would make our
   build depend on their insertion history.

2. **Never propagate the source's integer ids into the index.** They are
   AUTOINCREMENT artefacts. They are carried only as `source_ref` provenance
   for nodes (where they are the human-meaningful slug) and are discarded
   entirely for edges. In particular they must never break a tie - if a
   collapse rule needed the source rowid to decide, the rule would be
   inheriting exactly the non-determinism we are removing.

Stdlib only.
"""

import sqlite3
from typing import NamedTuple


class SourceNode(NamedTuple):
    src_id: str          # source slug, e.g. "claude_technique_prompt_caching"
    type: str
    name: str
    description: str


class SourceEdge(NamedTuple):
    from_src: str
    to_src: str
    type: str
    weight: float
    metadata: str | None  # raw JSON text or NULL, exactly as stored


class SourceEmbedding(NamedTuple):
    src_id: str
    model: str
    dim: int
    vector: bytes


def read_nodes(conn: sqlite3.Connection) -> list[SourceNode]:
    """All nodes, ordered by their source slug (unique - verified at build)."""
    rows = conn.execute(
        "SELECT id, type, name, COALESCE(description,'')"
        " FROM nodes ORDER BY id"
    ).fetchall()
    return [SourceNode(r[0], r[1], r[2], r[3]) for r in rows]


def read_edges(conn: sqlite3.Connection) -> list[SourceEdge]:
    """All edges in a content-determined order.

    Ordered by `(from,to,type,metadata)` rather than by the source's integer
    id. That tuple is not unique - the 48 duplicate groups are precisely the
    rows that collide on the first three components - but ordering by content
    means the sequence handed to `normalize` is a function of the *data*, so a
    source rebuilt with different rowids yields the same sequence here.

    `metadata` is included as the final sort component so that even the four
    payload-divergent groups arrive in a stable order.
    """
    rows = conn.execute(
        "SELECT from_node, to_node, type, COALESCE(weight,1.0), metadata"
        " FROM edges ORDER BY from_node, to_node, type, metadata"
    ).fetchall()
    return [SourceEdge(r[0], r[1], r[2], float(r[3]), r[4]) for r in rows]


def read_embeddings(conn: sqlite3.Connection) -> list[SourceEmbedding]:
    """Frozen vectors carried forward from the source.

    These are *data*, not a model: the vectors already exist and are copied
    verbatim. Nothing here runs an embedder. Re-keying them onto the new
    content-addressed node ids happens in `normalize.py`.
    """
    rows = conn.execute(
        "SELECT node_id, embedding_model, dimension, embedding"
        " FROM node_embeddings ORDER BY node_id"
    ).fetchall()
    return [SourceEmbedding(r[0], r[1], int(r[2]), r[3]) for r in rows]


def source_fingerprint(conn: sqlite3.Connection) -> dict:
    """A content fingerprint of what was read, for the manifest.

    Records the shape of the input so that a manifest can be traced back to
    the exact source state. Deliberately not a hash of the source *file*: the
    file carries SQLite header counters and unrelated tables (audit logs, UI
    positions) that change without the graph changing.
    """
    from ..hashing import sha256_value

    nodes = read_nodes(conn)
    edges = read_edges(conn)
    embeddings = read_embeddings(conn)
    return {
        "nodes_read": len(nodes),
        "edges_read": len(edges),
        "embeddings_read": len(embeddings),
        "nodes_sha": sha256_value([list(n) for n in nodes]),
        "edges_sha": sha256_value([list(e) for e in edges]),
        "embeddings_sha": sha256_value(
            [[e.src_id, e.model, e.dim, e.vector.hex()] for e in embeddings]
        ),
    }
