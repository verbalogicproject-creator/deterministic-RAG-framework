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


def _falsify_bm25_length_normalisation() -> None:
    """Disable length normalisation - literally the prior engine's defect."""
    import drf.retrieval.bm25 as bm25

    real = bm25.score_documents

    def flat(**kwargs):
        kwargs["b"] = 0.0
        return real(**kwargs)

    bm25.score_documents = flat


def _falsify_candidates_not_truncated() -> None:
    """Cap the query at ten terms, as python_apps_hybrid_query.py:339 did."""
    import drf.retrieval.lexical as lexical

    real = lexical.postings_for_terms
    lexical.postings_for_terms = lambda conn, terms: real(conn, terms[:10])


def _falsify_scores_are_int() -> None:
    """Skip quantisation so floats reach the sort key."""
    import drf.retrieval.bm25 as bm25
    from drf.fixed import QUANTUM

    bm25.quantize = lambda x: x * QUANTUM


def _falsify_one_tokenizer() -> None:
    """Let the query path filter short tokens while indexing does not.

    The classic index/query divergence: a rule applied on one side only. Terms
    present in a document become unfindable, and no ranking test can see it.
    """
    import drf.retrieval.tokenize as tokenize_module

    real = tokenize_module.terms

    def divergent(*, text):
        value, justification = real(text=text)
        return type(value)([v for v in value if len(v) > 3]), justification

    tokenize_module.terms = divergent


def _falsify_strict_total_order() -> None:
    """Drop the injective component, leaving the sort key non-injective.

    Audited before the test was written: on this corpus the truncated key
    collides for 66 candidates across 7 of 15 queries, so the mutation is
    genuinely detectable rather than theoretically so.
    """
    import drf.retrieval.stage1 as stage1

    real = stage1.sort_key
    stage1.sort_key = lambda result: real(result)[:-1]


def _falsify_bidirectional_expansion() -> None:
    """Follow outgoing edges only - the reading that honours edge direction."""
    import drf.store as store

    store.neighbours = lambda conn, node_id: sorted(
        {r[0] for r in conn.execute(
            "SELECT to_id FROM edges WHERE from_id = ?", (node_id,)
        )} - {node_id}
    )
    import drf.retrieval.graph as graph
    graph.neighbours = store.neighbours


def _falsify_merge_is_append_only() -> None:
    """Interleave advisory results into the prefix, AND neuter the guard.

    Interleaving alone would only trip merge's runtime postcondition, which
    would prove the *postcondition* fires - a different claim. To falsify the
    *test*, the bad merge must be allowed to return quietly, so the assertion
    is the only thing left that can notice.
    """
    import drf.retrieval.merge as merge_module

    merge_module._assert_subordination = lambda merged, deterministic: None
    real = merge_module.merge

    def interleaving_merge(*, deterministic, advisory, known_ids=None):
        merged = real(
            deterministic=deterministic, advisory=advisory, known_ids=known_ids
        )
        tail = [r for r in merged if r.origin == merge_module.ADVISORY]
        head = [r for r in merged if r.origin == merge_module.AUTHORITATIVE]
        if not tail:
            return merged
        # Promote one advisory result into second place.
        woven = head[:1] + tail[:1] + head[1:] + tail[1:]
        return [r._replace(rank=i) for i, r in enumerate(woven)]

    merge_module.merge = interleaving_merge


def _falsify_advisory_allowlist() -> None:
    """Widen the unwrap allowlist so any module can open the box."""
    import drf.contract as contract

    contract.ADVISORY_CONSUMERS = frozenset(
        set(contract.ADVISORY_CONSUMERS) | {"tests.test_merge", "test_merge"}
    )


def _falsify_config_hash_ignores_display() -> None:
    """Hash every setting - the obvious, more-thorough-looking implementation."""
    import drf.config.manager as manager

    manager.Config.content_hash = lambda self: manager.sha256_value(
        {"schema_version": 1, "ranking": self.as_dict()}
    )


def _falsify_config_hash_covers_ranking() -> None:
    """Silently omit one ranking key from the *hashed* set only.

    `ranking_keys()` is deliberately left intact. A first attempt patched it
    too, which removed `ranking.b` from the list the test iterates as well as
    from the hash - so the test never checked the key that had been broken and
    passed happily. The falsifier must damage the thing under test without
    also narrowing what the test looks at.
    """
    import drf.config.manager as manager

    manager.Config.ranking_settings = lambda self: {
        k: self._settings[k] for k in manager.ranking_keys() if k != "ranking.b"
    }


def _falsify_config_rejects_unknown_keys() -> None:
    """Accept anything, as the recovered project's config_manager.set() did."""
    import drf.config.manager as manager

    manager.validate_one = lambda key, value: None
    manager.Config.set = lambda self, key, value: self._settings.__setitem__(key, value)


def _falsify_ranking_params_are_live() -> None:
    """Thread max_depth through the call chain but never read it."""
    import drf.retrieval.stage1 as stage1

    real = stage1.rank

    def deaf_rank(**kwargs):
        kwargs["max_depth"] = stage1.DEFAULT_MAX_DEPTH
        return real(**kwargs)

    stage1.rank = deaf_rank


FALSIFIERS = {
    "ddl_forbidden_constructs": _falsify_ddl_forbidden_constructs,
    "collapse_preserves_metadata": _falsify_collapse_preserves_metadata,
    "dangling_edges_not_repaired": _falsify_dangling_edges_not_repaired,
    "isolated_nodes_kept": _falsify_isolated_nodes_kept,
    "content_hash_ignores_hash_seed": _falsify_content_hash_ignores_hash_seed,
    "bm25_length_normalisation": _falsify_bm25_length_normalisation,
    "candidates_not_truncated": _falsify_candidates_not_truncated,
    "scores_are_int": _falsify_scores_are_int,
    "one_tokenizer": _falsify_one_tokenizer,
    "strict_total_order": _falsify_strict_total_order,
    "bidirectional_expansion": _falsify_bidirectional_expansion,
    "merge_is_append_only": _falsify_merge_is_append_only,
    "advisory_allowlist": _falsify_advisory_allowlist,
    "config_hash_ignores_display": _falsify_config_hash_ignores_display,
    "config_hash_covers_ranking": _falsify_config_hash_covers_ranking,
    "config_rejects_unknown_keys": _falsify_config_rejects_unknown_keys,
    "ranking_params_are_live": _falsify_ranking_params_are_live,
}


_requested = os.environ.get("DRF_FALSIFY")
if _requested:
    if _requested not in FALSIFIERS:
        raise SystemExit(
            f"unknown falsifier {_requested!r}; "
            f"known: {sorted(FALSIFIERS)}"
        )
    FALSIFIERS[_requested]()


def _falsify_bench_detects_nondeterminism() -> None:
    """Give the chaos control its tiebreak back, making it deterministic.

    If the control cannot fail, the entire benchmark becomes decorative while
    continuing to report 1.0 on every metric.
    """
    import drf.bench.repro as repro
    from drf.retrieval import stage1

    def deterministic_chaos(conn, index_hash, queries, seed):
        return {
            q["id"]: repro.rank_ids(conn, index_hash, q["text"]) for q in queries
        }

    repro.chaos_run = deterministic_chaos


FALSIFIERS["bench_detects_nondeterminism"] = _falsify_bench_detects_nondeterminism


def _falsify_docs_are_generated() -> None:
    """Make a fresh render diverge from what is committed.

    Changes the renderer rather than editing a file on disk: the effect on the
    test is identical, and it leaves no dirty working tree behind.
    """
    import drf.docs.render as render

    real = render.render_document
    render.render_document = lambda audience, context: (
        real(audience, context) + "\n<!-- injected by falsifier -->\n"
    )
    render.render_all = lambda context: {
        audience: render.render_document(audience, context)
        for audience in render.AUDIENCES
    }


def _falsify_docs_fail_on_missing_placeholder() -> None:
    """safe_substitute: the forgiving call that leaves holes in the prose."""
    import string

    import drf.docs.render as render

    class Forgiving(string.Template):
        def substitute(self, *args, **kwargs):
            return self.safe_substitute(*args, **kwargs)

    render.Template = Forgiving


FALSIFIERS["docs_are_generated"] = _falsify_docs_are_generated
FALSIFIERS["docs_fail_on_missing_placeholder"] = _falsify_docs_fail_on_missing_placeholder
