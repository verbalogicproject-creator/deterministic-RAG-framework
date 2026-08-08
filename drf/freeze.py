"""Freeze: bind a release to an exact spec, index and result set.

A git tag names a commit. That is not enough to reproduce a result, because a
result depends on three things and a commit pins only one of them:

    spec_sha        every spec/*.json, hashed together
    manifest_hash   the index the results came from
    bench_digest    the actual answers to the fixed query set

`drf freeze` records all three, and `tests/test_freeze.py` rebuilds from source
and checks that every one still matches. So a release is not "the code at this
commit" but "these answers, from this index, under this spec" - and the claim
is verified rather than asserted.

The bench digest is the load-bearing one. Spec and index hashes prove the
*inputs* are unchanged; only the digest proves the *outputs* are. A refactor
that alters ranking while leaving both inputs untouched is exactly what would
otherwise slip through a release.

Stdlib only.
"""

import json
from pathlib import Path

from .hashing import sha256_value
from .version import (
    ID_SCHEMA_VERSION,
    MANIFEST_VERSION,
    PARSER_VERSION,
    RANKER_VERSION,
    RELEASE_VERSION,
)

ROOT = Path(__file__).resolve().parent.parent
FROZEN_PATH = ROOT / "spec" / "frozen.json"


def spec_sha() -> str:
    """One hash over every spec file."""
    return sha256_value({
        path.name: json.loads(path.read_text())
        for path in sorted((ROOT / "spec").glob("*.json"))
        if path.name != "frozen.json"   # a freeze cannot include its own hash
    })


def compute(index_path: str) -> dict:
    """The three hashes plus the versions that produced them."""
    from .bench import repro
    from .store import connect, read_manifest

    conn = connect(index_path)
    manifest_hash = read_manifest(conn)["content_hash"]
    queries = repro.load_queries()
    results = repro.run_all(conn, manifest_hash, queries)
    conn.close()

    return {
        "release": RELEASE_VERSION,
        "spec_sha": spec_sha(),
        "manifest_hash": manifest_hash,
        "bench_digest": repro.digest(results),
        "query_count": len(queries),
        "versions": {
            "parser": PARSER_VERSION,
            "ranker": RANKER_VERSION,
            "id_schema": ID_SCHEMA_VERSION,
            "manifest": MANIFEST_VERSION,
        },
    }


def write(index_path: str) -> dict:
    frozen = compute(index_path)
    FROZEN_PATH.write_text(json.dumps(frozen, indent=2) + "\n")
    return frozen


def read() -> dict:
    if not FROZEN_PATH.exists():
        raise FileNotFoundError(
            f"{FROZEN_PATH} not found; run `drf freeze --index index.db`"
        )
    return json.loads(FROZEN_PATH.read_text())


def verify(index_path: str) -> tuple[bool, list[str]]:
    """Compare a live rebuild against the recorded freeze.

    Returns `(ok, differences)` rather than raising, so a caller can report
    every mismatch at once. A release check that stops at the first difference
    makes the second one look like it appeared later.
    """
    recorded = read()
    live = compute(index_path)
    differences = [
        f"{key}: frozen {recorded.get(key)!r} != live {live.get(key)!r}"
        for key in ("spec_sha", "manifest_hash", "bench_digest", "versions")
        if recorded.get(key) != live.get(key)
    ]
    return not differences, differences
