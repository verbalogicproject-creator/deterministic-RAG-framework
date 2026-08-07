"""Canonical serialisation and content-addressed identifiers.

Everything the framework hashes goes through `canonical_json` first. The
encoding is pinned so that the same logical value produces the same bytes on
any machine, any Python build, any day:

    sort_keys=True      - dict order cannot leak in
    separators tight    - no incidental whitespace
    ensure_ascii=False  - one representation per string, UTF-8 encoded
    allow_nan=False     - NaN/Infinity are not JSON and are not silently emitted

Stdlib only. This module must never import anything from the retrieval path.
"""

import hashlib
import json
from typing import Any

from .version import ID_SCHEMA_VERSION


def canonical_json(value: Any) -> str:
    """Serialise `value` to its one canonical JSON representation."""
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    """SHA-256 of a string, hashed as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_value(value: Any) -> str:
    """SHA-256 of any JSON-serialisable value, via its canonical form."""
    return sha256_text(canonical_json(value))


def content_id(prefix: str, payload: dict, length: int = 32) -> str:
    """A stable, content-addressed identifier.

    The ID is a pure function of `payload`, so it survives rebuilds, dedupes
    identical records for free, and - because it is unique - serves as the
    final tiebreak component of the ranking sort key. That last property is
    what makes the total order strict; see retrieval/stage1.py.

    `ID_SCHEMA_VERSION` is folded in so that changing the recipe cannot
    silently produce colliding IDs across schema generations.
    """
    body = dict(payload)
    body["_id_schema"] = ID_SCHEMA_VERSION
    return f"{prefix}_{sha256_value(body)[:length]}"


def node_id(*, type: str, name: str, description: str, source: str) -> str:
    """Content-addressed node identifier.

    `source` must be a stable logical reference, never an absolute filesystem
    path - otherwise the same content ingested from a different directory
    would produce a different ID.
    """
    return content_id("n", {
        "type": type,
        "name": name,
        "description": description or "",
        "source": source,
    })


def edge_id(*, from_id: str, to_id: str, type: str) -> str:
    """Content-addressed edge identifier.

    Because this is the edges table's PRIMARY KEY, duplicate (from, to, type)
    triples collapse structurally on insert - no dedupe pass is needed.
    """
    return content_id("e", {"from": from_id, "to": to_id, "type": type})
