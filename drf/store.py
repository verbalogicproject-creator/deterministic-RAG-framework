"""The index store: schema, and the only sanctioned way to read or write it.

Three properties of this schema are load-bearing, not stylistic.

**No AUTOINCREMENT.** Every primary key is a content-addressed TEXT id from
`drf.hashing`. An id therefore depends on what a record *is*, never on when it
was inserted. The source knowledge graph uses `INTEGER PRIMARY KEY
AUTOINCREMENT` on both nodes and edges; re-ingesting the same data in a
different order would renumber everything. Here it cannot.

**No CURRENT_TIMESTAMP, no timestamps at all.** A default of
`CURRENT_TIMESTAMP` makes a row's bytes depend on the wall clock, which
defeats build reproducibility at the first column. Build time is recorded once
in the manifest as *provenance*, explicitly excluded from `content_hash`.
`tests/test_ingest.py` greps the DDL for both keywords and requires zero hits.

**WITHOUT ROWID.** An ordinary SQLite table keeps a hidden integer `rowid`
assigned in insertion order, and an unordered `SELECT` scans in that order -
so insertion history leaks into query results even when the declared primary
key is a content hash. `WITHOUT ROWID` removes the hidden column and keys the
b-tree on the primary key itself. Every read below still specifies `ORDER BY`
explicitly; this is the second line of defence, so that a future forgotten
`ORDER BY` degrades to *content* order rather than *insertion* order.

Stdlib only.
"""

import sqlite3
from typing import Any, Iterable, Iterator, NamedTuple

from .hashing import canonical_json

# ---------------------------------------------------------------- schema

SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE nodes (
        id          TEXT NOT NULL PRIMARY KEY,
        type        TEXT NOT NULL,
        name        TEXT NOT NULL,
        description TEXT NOT NULL,
        source      TEXT NOT NULL,
        source_ref  TEXT NOT NULL,
        metadata    TEXT NOT NULL
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE edges (
        id       TEXT NOT NULL PRIMARY KEY,
        from_id  TEXT NOT NULL REFERENCES nodes(id),
        to_id    TEXT NOT NULL REFERENCES nodes(id),
        type     TEXT NOT NULL,
        weight_q INTEGER NOT NULL,
        metadata TEXT NOT NULL
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE embeddings (
        node_id TEXT NOT NULL PRIMARY KEY REFERENCES nodes(id),
        model   TEXT NOT NULL,
        dim     INTEGER NOT NULL,
        vector  BLOB NOT NULL
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE manifest (
        key   TEXT NOT NULL PRIMARY KEY,
        value TEXT NOT NULL
    ) WITHOUT ROWID
    """,
    # ---- lexical index (M1.2) ----
    # Document length in terms, needed for BM25 length normalisation. Stored
    # per document rather than recomputed, so scoring never re-tokenises.
    """
    CREATE TABLE doc_stats (
        node_id TEXT NOT NULL PRIMARY KEY REFERENCES nodes(id),
        doc_len INTEGER NOT NULL
    ) WITHOUT ROWID
    """,
    # Document frequency per term. Denormalised from postings deliberately:
    # df is read for every query term on every query, and deriving it with
    # count(*) would make idf depend on a scan.
    """
    CREATE TABLE terms (
        term TEXT NOT NULL PRIMARY KEY,
        df   INTEGER NOT NULL
    ) WITHOUT ROWID
    """,
    # The inverted index. (term, node_id) is the primary key, so a term
    # cannot be posted twice against one document - duplicate postings
    # collapse structurally, exactly as duplicate edges do.
    """
    CREATE TABLE postings (
        term    TEXT NOT NULL,
        node_id TEXT NOT NULL REFERENCES nodes(id),
        tf      INTEGER NOT NULL,
        PRIMARY KEY (term, node_id)
    ) WITHOUT ROWID
    """,
    # Traversal index. Covering, so graph expansion never scans the table and
    # never depends on physical row placement.
    "CREATE INDEX idx_edges_from ON edges(from_id, type, to_id)",
    "CREATE INDEX idx_edges_to   ON edges(to_id, type, from_id)",
)

# Table names declared above, derived rather than listed by hand so that
# adding a table cannot silently escape the WITHOUT ROWID and count checks.
TABLES: tuple[str, ...] = tuple(
    sorted(
        stmt.split("CREATE TABLE", 1)[1].split("(", 1)[0].strip()
        for stmt in SCHEMA if "CREATE TABLE" in stmt
    )
)

# Keywords whose presence anywhere in the DDL is a determinism defect.
FORBIDDEN_DDL = ("AUTOINCREMENT", "CURRENT_TIMESTAMP", "DEFAULT (datetime", "random()")


# ---------------------------------------------------------------- records

class NodeRecord(NamedTuple):
    id: str
    type: str
    name: str
    description: str
    source: str
    source_ref: str
    metadata: dict


class EdgeRecord(NamedTuple):
    id: str
    from_id: str
    to_id: str
    type: str
    weight_q: int
    metadata: dict


class EmbeddingRecord(NamedTuple):
    node_id: str
    model: str
    dim: int
    vector: bytes


class PostingRecord(NamedTuple):
    term: str
    node_id: str
    tf: int


# ---------------------------------------------------------------- open/create

def connect(path: str) -> sqlite3.Connection:
    """Open an index with settings pinned for reproducibility.

    `foreign_keys` is ON so a dangling edge is a hard error rather than silent
    corruption - the source graph carries 4 such edges, and they must be
    dropped explicitly and recorded, never inserted and forgotten.
    """
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    # Not a determinism control, but keeps builds honest under interruption.
    conn.execute("PRAGMA journal_mode = DELETE")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    for statement in SCHEMA:
        conn.execute(statement)
    conn.commit()


def ddl_text(conn: sqlite3.Connection) -> str:
    """Every CREATE statement in the database, in name order.

    Read back from `sqlite_master` rather than from `SCHEMA` so the test
    inspects what SQLite actually stored, not what we intended to write.
    """
    rows = conn.execute(
        "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY name"
    ).fetchall()
    return "\n".join(r[0] for r in rows)


# ---------------------------------------------------------------- writes

def insert_nodes(conn: sqlite3.Connection, nodes: Iterable[NodeRecord]) -> int:
    """Insert nodes; returns the count actually written.

    Plain INSERT, not INSERT OR IGNORE: by this point ids are unique by
    construction and a collision means the id recipe is wrong. Surfacing that
    as an IntegrityError is the correct outcome - silently ignoring it would
    drop a node and quietly shrink the corpus.
    """
    rows = [
        (n.id, n.type, n.name, n.description, n.source, n.source_ref,
         canonical_json(n.metadata))
        for n in nodes
    ]
    conn.executemany(
        "INSERT INTO nodes (id,type,name,description,source,source_ref,metadata)"
        " VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def insert_edges(conn: sqlite3.Connection, edges: Iterable[EdgeRecord]) -> int:
    rows = [
        (e.id, e.from_id, e.to_id, e.type, e.weight_q, canonical_json(e.metadata))
        for e in edges
    ]
    conn.executemany(
        "INSERT INTO edges (id,from_id,to_id,type,weight_q,metadata)"
        " VALUES (?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def insert_embeddings(conn: sqlite3.Connection,
                      embeddings: Iterable[EmbeddingRecord]) -> int:
    rows = [(e.node_id, e.model, e.dim, e.vector) for e in embeddings]
    conn.executemany(
        "INSERT INTO embeddings (node_id,model,dim,vector) VALUES (?,?,?,?)", rows
    )
    return len(rows)


def write_manifest(conn: sqlite3.Connection, manifest: dict) -> None:
    """Store the manifest as one canonical-JSON blob under a single key.

    Stored whole rather than shredded into key/value rows so that what is read
    back is byte-identical to what was hashed.
    """
    conn.execute(
        "INSERT OR REPLACE INTO manifest (key,value) VALUES (?,?)",
        ("manifest", canonical_json(manifest)),
    )
    conn.commit()


# ---------------------------------------------------------------- reads

def read_manifest(conn: sqlite3.Connection) -> dict:
    import json
    row = conn.execute(
        "SELECT value FROM manifest WHERE key = 'manifest'"
    ).fetchone()
    if row is None:
        raise ValueError("index has no manifest")
    return json.loads(row[0])


def iter_nodes(conn: sqlite3.Connection) -> Iterator[NodeRecord]:
    """All nodes in content-id order. The ORDER BY is mandatory, not defensive."""
    import json
    for r in conn.execute(
        "SELECT id,type,name,description,source,source_ref,metadata"
        " FROM nodes ORDER BY id"
    ):
        yield NodeRecord(r[0], r[1], r[2], r[3], r[4], r[5], json.loads(r[6]))


def iter_edges(conn: sqlite3.Connection) -> Iterator[EdgeRecord]:
    import json
    for r in conn.execute(
        "SELECT id,from_id,to_id,type,weight_q,metadata FROM edges ORDER BY id"
    ):
        yield EdgeRecord(r[0], r[1], r[2], r[3], r[4], json.loads(r[5]))


def iter_embeddings(conn: sqlite3.Connection) -> Iterator[EmbeddingRecord]:
    for r in conn.execute(
        "SELECT node_id,model,dim,vector FROM embeddings ORDER BY node_id"
    ):
        yield EmbeddingRecord(r[0], r[1], r[2], r[3])


def table_count(conn: sqlite3.Connection, table: str) -> int:
    """Row count for a table. `table` is checked against the declared set
    rather than interpolated blindly."""
    if table not in TABLES:
        raise ValueError(f"unknown table: {table!r}; known: {list(TABLES)}")
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


# ---------------------------------------------------------------- lexical index

def insert_lexical(
    conn: sqlite3.Connection,
    *,
    doc_lens: dict[str, int],
    postings: list[PostingRecord],
    dfs: dict[str, int],
) -> dict[str, int]:
    """Write the inverted index. Returns counts derived by `len()`."""
    conn.executemany(
        "INSERT INTO doc_stats (node_id, doc_len) VALUES (?,?)",
        sorted(doc_lens.items()),
    )
    conn.executemany(
        "INSERT INTO terms (term, df) VALUES (?,?)",
        sorted(dfs.items()),
    )
    conn.executemany(
        "INSERT INTO postings (term, node_id, tf) VALUES (?,?,?)",
        [(p.term, p.node_id, p.tf) for p in sorted(postings)],
    )
    return {
        "doc_stats": len(doc_lens),
        "terms": len(dfs),
        "postings": len(postings),
    }


def iter_postings(conn: sqlite3.Connection) -> Iterator[PostingRecord]:
    for r in conn.execute(
        "SELECT term, node_id, tf FROM postings ORDER BY term, node_id"
    ):
        yield PostingRecord(r[0], r[1], r[2])


def postings_for_terms(
    conn: sqlite3.Connection, terms: list[str]
) -> dict[str, dict[str, int]]:
    """Postings for the given terms, as {term: {node_id: tf}}.

    Terms are queried in sorted order and each term's rows come back ordered,
    so the result is built identically on every run. A term with no postings
    is simply absent - it contributes nothing to the union and nothing to any
    score.
    """
    out: dict[str, dict[str, int]] = {}
    for term in sorted(set(terms)):
        rows = conn.execute(
            "SELECT node_id, tf FROM postings WHERE term = ? ORDER BY node_id",
            (term,),
        ).fetchall()
        if rows:
            out[term] = {r[0]: r[1] for r in rows}
    return out


def df_for_terms(conn: sqlite3.Connection, terms: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for term in sorted(set(terms)):
        row = conn.execute(
            "SELECT df FROM terms WHERE term = ?", (term,)
        ).fetchone()
        if row is not None:
            out[term] = row[0]
    return out


def doc_lengths(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT node_id, doc_len FROM doc_stats ORDER BY node_id"
        )
    }


def neighbours(conn: sqlite3.Connection, node_id: str) -> list[str]:
    """Every node adjacent to `node_id`, in **both** directions, sorted.

    Bidirectional by design, and that is a measured decision rather than a
    convenience. On this corpus, following only outgoing edges leaves 81 of
    266 nodes reaching nothing at depth 2, against 5 when both directions are
    followed - and those 5 are exactly the nodes with no edges at all. Direction
    is a property of how the source author happened to phrase a relation
    ("A enables B" versus "B requires A"), not of whether two nodes are
    related, so honouring it would drop 76 nodes' worth of structure for a
    reason that carries no meaning.

    Both queries are covered by `idx_edges_from` / `idx_edges_to`, so neither
    scans the table, and the result never depends on physical row placement.
    """
    out = {
        r[0] for r in conn.execute(
            "SELECT to_id FROM edges WHERE from_id = ?", (node_id,)
        )
    }
    out |= {
        r[0] for r in conn.execute(
            "SELECT from_id FROM edges WHERE to_id = ?", (node_id,)
        )
    }
    out.discard(node_id)
    return sorted(out)


def corpus_totals(conn: sqlite3.Connection) -> tuple[int, int]:
    """(n_docs, total_length) as exact integers.

    `avgdl` is deliberately *not* stored. It is derived as `total / n` at use
    time, so the float is produced by one division of two exact integers -
    identical on every run - rather than persisted as a rounded decimal that
    could differ from what produced it.
    """
    row = conn.execute(
        "SELECT count(*), COALESCE(sum(doc_len), 0) FROM doc_stats"
    ).fetchone()
    return int(row[0]), int(row[1])
