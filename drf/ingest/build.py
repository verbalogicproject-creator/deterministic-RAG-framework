"""Build an index from a source knowledge graph.

The build is registered as `ingest.build_index`, declared deterministic and
authoritative in `spec/actions.json`. That declaration is not a comment: the
`@action` replay check memoises `inputs_hash -> sha256(value)`, and `inputs`
below names only `source_path` and `corpus`. So building the *same source*
into *two different directories* produces the same inputs hash, and the
contract compares the two results automatically.

The consequence is worth stating plainly: **the M1.1 reproducibility
requirement is enforced at runtime by the contract, not only by a test.** A
change that made the build depend on the output path, the wall clock, or dict
iteration order would raise `DeterminismViolation` on the second build, in
production, without anyone having remembered to write an assertion.

Stdlib only.
"""

import os
import sqlite3
import sys

from ..contract import ActionOutput, action
from ..store import (
    connect,
    create_schema,
    insert_edges,
    insert_embeddings,
    insert_nodes,
    write_manifest,
)
from . import source_kg
from .manifest import build_manifest, reconcile
from .normalize import CORPUS, normalize_all


class BuildError(Exception):
    """The build could not produce a trustworthy index."""


@action(
    "ingest.build_index",
    determinism="deterministic",
    authority="authoritative",
    inputs=("source_path", "corpus"),
)
def build_index(*, source_path: str, out_path: str, corpus: str = CORPUS) -> ActionOutput:
    """Read a source KG, normalise it, and write a content-addressed index.

    Returns only the reproducible half of the manifest. Provenance - build
    time, absolute paths, interpreter version - is attached to the manifest
    written to disk but deliberately excluded from the returned value, because
    the returned value is what the replay check compares.
    """
    if not os.path.exists(source_path):
        raise BuildError(f"source not found: {source_path}")

    src = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    try:
        fingerprint = source_kg.source_fingerprint(src)
        normalized = normalize_all(
            source_kg.read_nodes(src),
            source_kg.read_edges(src),
            source_kg.read_embeddings(src),
        )
    finally:
        src.close()

    # Every source row accounted for exactly once. Asserted on every build,
    # not only under test - a reconciliation checked solely in CI is not
    # checking the build that shipped.
    balance = reconcile(normalized, fingerprint)
    if not balance["balanced"]:
        raise BuildError(
            f"edge reconciliation failed: read {balance['edges_read']}, "
            f"accounted {balance['accounted']} "
            f"(written {balance['edges_written']}, "
            f"collapsed {balance['collapsed_variants_removed']}, "
            f"dropped {balance['edges_dropped']}). "
            "A row was double-counted or lost; the index is not trustworthy."
        )

    manifest = build_manifest(
        normalized,
        corpus=corpus,
        source_fingerprint=fingerprint,
        provenance={
            "source_path": os.path.abspath(source_path),
            "out_path": os.path.abspath(out_path),
            "python": sys.version.split()[0],
            "reconciliation": balance,
        },
    )

    if os.path.exists(out_path):
        os.remove(out_path)
    conn = connect(out_path)
    try:
        create_schema(conn)
        written_nodes = insert_nodes(conn, normalized.nodes)
        written_edges = insert_edges(conn, normalized.edges)
        written_embeddings = insert_embeddings(conn, normalized.embeddings)
        write_manifest(conn, manifest)
        conn.commit()
    finally:
        conn.close()

    # The manifest claims counts derived by len(); confirm the database agrees
    # before the build is allowed to succeed. This is the one place where a
    # silent partial write would otherwise go unnoticed.
    counts = manifest["content"]["counts"]
    for label, claimed, written in (
        ("nodes", counts["nodes"], written_nodes),
        ("edges", counts["edges"], written_edges),
        ("embeddings", counts["embeddings"], written_embeddings),
    ):
        if claimed != written:
            raise BuildError(
                f"{label}: manifest claims {claimed}, wrote {written}"
            )

    return ActionOutput(
        value={
            "content_hash": manifest["content_hash"],
            "content": manifest["content"],
        },
        evidence=(
            f"source={fingerprint['nodes_read']}n/{fingerprint['edges_read']}e",
            f"written={written_nodes}n/{written_edges}e",
            f"dropped={len(normalized.dropped)}",
            f"collapsed={len(normalized.collapsed)}",
        ),
    )
