"""Falsifier injection.

When `DRF_FALSIFY=<invariant id>` is set, this conftest mutates the drf
package *before* any test module imports it, so the named test runs against a
deliberately broken implementation. `tests/test_falsifiers.py` sets the
variable in a subprocess and asserts the target test fails.

Each mutation here corresponds to one entry in `spec/invariants.json`. The two
must not drift, and `test_falsifiers.py` asserts a bijection between them.

Nothing in this file has any effect unless the environment variable is set,
so a normal `pytest` run is entirely unaffected.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------
# The mutations
# --------------------------------------------------------------------------

def _falsify_ddl_forbidden_constructs() -> None:
    """Add a plausible future table that reintroduces AUTOINCREMENT."""
    import drf.store as store
    store.SCHEMA = store.SCHEMA + (
        "CREATE TABLE build_log ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " note TEXT)",
    )


def _falsify_collapse_preserves_metadata() -> None:
    """The rejected alternative: discard edge metadata rather than union it."""
    import drf.ingest.normalize as normalize
    normalize.collapse_edge_group = (
        lambda variants: variants[0]._replace(metadata={})
    )


def _falsify_dangling_edges_not_repaired() -> None:
    """The tempting heuristic: resolve a missing slug by substring match."""
    import drf.ingest.normalize as normalize

    def fuzzy_resolve(src_ref: str, id_map: dict) -> str | None:
        exact = id_map.get(src_ref)
        if exact is not None:
            return exact
        for slug in sorted(id_map):
            if src_ref in slug:
                return id_map[slug]
        return None

    normalize.resolve_endpoint = fuzzy_resolve


def _falsify_isolated_nodes_kept() -> None:
    """Drop nodes with no incident edge - flattering to every graph metric."""
    import drf.ingest.normalize as normalize

    original = normalize.normalize_all

    def pruning_normalize_all(source_nodes, source_edges, source_embeddings):
        result = original(source_nodes, source_edges, source_embeddings)
        linked = set()
        for edge in result.edges:
            linked.add(edge.from_id)
            linked.add(edge.to_id)
        kept = [n for n in result.nodes if n.id in linked]
        kept_ids = {n.id for n in kept}
        return result._replace(
            nodes=kept,
            embeddings=[e for e in result.embeddings if e.node_id in kept_ids],
        )

    normalize.normalize_all = pruning_normalize_all
    import drf.ingest.build as build
    build.normalize_all = pruning_normalize_all


def _falsify_content_hash_ignores_hash_seed() -> None:
    """Emit a set without sorting it - hash-seed dependence, the classic.

    Uses node *types* (25 distinct values) rather than embedding models. The
    first version of this falsifier mutated `embedding_models`, and the
    registry caught that it was a no-op: this corpus has exactly one model and
    one dimension, so `list(set)` and `sorted(set)` are identical for both.

    That is a finding about the corpus, not just the falsifier. No
    multi-element set currently reaches `content_hash`, so the hash-seed test
    protects against a class of bug the present data cannot exhibit. This
    mutation adds the field a future contributor would plausibly add, so the
    protection is exercised rather than merely asserted.
    """
    import drf.ingest.manifest as manifest

    original = manifest.build_content

    def unsorted_build_content(normalized, corpus):
        content = original(normalized, corpus)
        content["node_types"] = list({n.type for n in normalized.nodes})
        return content

    manifest.build_content = unsorted_build_content


FALSIFIERS = {
    "ddl_forbidden_constructs": _falsify_ddl_forbidden_constructs,
    "collapse_preserves_metadata": _falsify_collapse_preserves_metadata,
    "dangling_edges_not_repaired": _falsify_dangling_edges_not_repaired,
    "isolated_nodes_kept": _falsify_isolated_nodes_kept,
    "content_hash_ignores_hash_seed": _falsify_content_hash_ignores_hash_seed,
}


_requested = os.environ.get("DRF_FALSIFY")
if _requested:
    if _requested not in FALSIFIERS:
        raise SystemExit(
            f"unknown falsifier {_requested!r}; "
            f"known: {sorted(FALSIFIERS)}"
        )
    FALSIFIERS[_requested]()
