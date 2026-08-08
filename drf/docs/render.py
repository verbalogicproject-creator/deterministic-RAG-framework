"""Generate the four audience documents from `spec/` and a built index.

Documentation drift is the failure this project was recovered from: the source
project claimed 147 dimensions against 118 real, documented 36 flags that did
not exist while omitting 35 that did, and printed `80% vs 60% accuracy` with no
evaluation anywhere in the repository. Every one of those numbers was written
by hand into prose that nothing checked.

So nothing here is written by hand. Every figure comes from `spec/*.json`, from
`drf/version.py`, or from an index built at render time - and every figure in
`spec/benchmarks.json` is stored beside the command that produces it.

**`Template.substitute`, never `safe_substitute`.** `safe_substitute` leaves an
unresolved `$placeholder` sitting in the output looking like prose. `substitute`
raises `KeyError`. A template referring to something the context does not supply
is a build failure, not a document with a hole in it.

The rendered files are committed, and `tests/test_docs.py` re-renders and
compares. A hand-edit therefore fails the suite. That is the point of
committing them: an ignored file cannot be checked.

Stdlib only.
"""

import json
from pathlib import Path
from string import Template

from ..version import (
    ID_SCHEMA_VERSION,
    MANIFEST_VERSION,
    PARSER_VERSION,
    RANKER_VERSION,
    RELEASE_VERSION,
)

ROOT = Path(__file__).resolve().parent.parent.parent
SPEC_DIR = ROOT / "spec"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
OUTPUT_DIR = ROOT / "docs"

AUDIENCES = ("peer", "agent", "operator", "plain", "readme")

# `readme` lands at the repository root, because that is the page a visitor
# sees first and therefore the most likely place for a stale number to sit
# unchallenged. Generating it means it cannot carry one.
OUTPUT_PATHS = {"readme": "README.md"}

BANNER = (
    "<!-- GENERATED FILE - DO NOT EDIT.\n"
    "     Produced by `drf docs build` from spec/*.json and a built index.\n"
    "     Hand edits are detected by tests/test_docs.py, which re-renders\n"
    "     and compares. Change the spec, then regenerate. -->"
)


def load_spec(name: str) -> dict:
    with open(SPEC_DIR / f"{name}.json") as f:
        return json.load(f)


def _md_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    """Render a markdown table from dicts. Columns are (key, header) pairs."""
    header = "| " + " | ".join(h for _, h in columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider] + body)


def build_context(index_path: str | None = None) -> dict:
    """Assemble every value the templates may reference.

    Values derived from files are computed with `len()` and comprehensions, so
    a count in the documentation is a count of the thing itself rather than a
    number someone remembered to update.
    """
    actions = load_spec("actions")["actions"]
    invariants = load_spec("invariants")
    ranking = load_spec("ranking")
    benchmarks = load_spec("benchmarks")
    config_schema = load_spec("config_schema")["settings"]

    context: dict[str, object] = {
        "banner": BANNER,
        "parser_version": PARSER_VERSION,
        "ranker_version": RANKER_VERSION,
        "id_schema_version": ID_SCHEMA_VERSION,
        "manifest_version": MANIFEST_VERSION,

        "action_count": len(actions),
        "action_table": _md_table(
            sorted(actions, key=lambda a: a["name"]),
            [("name", "action"), ("determinism", "determinism"),
             ("authority", "authority"), ("summary", "what it does")],
        ),
        "deterministic_count": sum(
            1 for a in actions if a["determinism"] == "deterministic"
        ),
        "advisory_count": sum(1 for a in actions if a["authority"] == "advisory"),

        "invariant_count": len(invariants["invariants"]),
        "exemption_count": len(invariants["not_falsified"]),
        "invariant_table": _md_table(
            invariants["invariants"],
            [("id", "invariant"), ("asserts", "asserts"),
             ("falsifier", "falsifier")],
        ),

        "k1": ranking["bm25"]["k1"],
        "b": ranking["bm25"]["b"],
        "quantum_exp": ranking["quantisation"]["exponent"],
        "quantisation_point": ranking["quantisation"]["point"],
        "bm25_formula": ranking["bm25"]["formula"],
        "idf_formula": ranking["bm25"]["idf"],

        "setting_count": len(config_schema),
        "ranking_setting_count": sum(
            1 for s in config_schema.values() if s.get("affects_ranking")
        ),
        "setting_table": _md_table(
            [
                {
                    "name": name,
                    "default": spec["default"],
                    "affects_ranking": "yes" if spec.get("affects_ranking") else "no",
                    "summary": spec["summary"],
                }
                for name, spec in sorted(config_schema.items())
            ],
            [("name", "setting"), ("default", "default"),
             ("affects_ranking", "affects ranking"), ("summary", "what it is")],
        ),

        "repro_cells": benchmarks["reproducibility"]["cells"],
        "repro_digests": benchmarks["reproducibility"]["distinct_digests"],
        "repro_discordant": benchmarks["reproducibility"]["discordant_pairs"],
        "repro_producer": benchmarks["reproducibility"]["producer"],
        "repro_axes": json.dumps(benchmarks["reproducibility"]["axes"]),

        "chaos_producer": benchmarks["chaos_control"]["producer"],
        "chaos_what": benchmarks["chaos_control"]["what_it_is"],
        "chaos_why": benchmarks["chaos_control"]["why_it_exists"],
        "chaos_table": _md_table(
            benchmarks["chaos_control"]["rows"],
            [("metric", "metric"), ("real", "real"), ("chaos", "chaos"),
             ("separates", "separates?")],
        ),
        "chaos_findings": "\n".join(
            f"- {f}" for f in benchmarks["chaos_control"]["findings"]
        ),

        "sensitivity_producer": benchmarks["sensitivity"]["producer"],
        "sensitivity_queries": benchmarks["sensitivity"]["queries"],
        "sensitivity_table": _md_table(
            benchmarks["sensitivity"]["rows"],
            [("setting", "setting"), ("default", "default"),
             ("probe", "probe"), ("reordered", "queries reordered")],
        ),
        "sensitivity_finding": benchmarks["sensitivity"]["finding"],

        "lengthnorm_producer": benchmarks["length_normalisation"]["producer"],
        "lengthnorm_discordant_pct": benchmarks["length_normalisation"]["discordant_pct"],
        "lengthnorm_pairs": benchmarks["length_normalisation"]["strictly_ordered_pairs"],
        "lengthnorm_direction": benchmarks["length_normalisation"]["direction_longer_shorter_equal"],
        "lengthnorm_ties": benchmarks["length_normalisation"]["queries_with_tied_top_set"],
        "lengthnorm_correction": benchmarks["length_normalisation"]["correction"],

        "graph_bidirectional_reach": benchmarks["graph"]["bidirectional_mean_reach"],
        "graph_forward_reach": benchmarks["graph"]["forward_only_mean_reach"],
        "graph_forward_dead": benchmarks["graph"]["forward_only_dead_nodes"],
        "graph_bidirectional_dead": benchmarks["graph"]["bidirectional_dead_nodes"],
        "graph_bfs_ms": benchmarks["graph"]["bfs_depth2_10_seeds_ms"],

        "horizon_producer": benchmarks["advisory_horizon"]["producer"],
        "horizon_queries": benchmarks["advisory_horizon"]["queries"],
        "horizon_identical": benchmarks["advisory_horizon"]["prefixes_identical"],
        "horizon_differing": benchmarks["advisory_horizon"]["prefixes_differing"],
        "horizon_min": benchmarks["advisory_horizon"]["horizon_min"],
        "horizon_max": benchmarks["advisory_horizon"]["horizon_max"],
        "horizon_silent": benchmarks["advisory_horizon"][
            "queries_with_no_reachable_evaluated_depth"],
        "horizon_finding": benchmarks["advisory_horizon"]["finding"],
        "horizon_second_finding": benchmarks["advisory_horizon"]["second_finding"],
        "horizon_why_m2": benchmarks["advisory_horizon"]["why_it_matters_for_M2"],

        "quality_producer": benchmarks["retrieval_quality"]["producer"],
        "quality_verdict": benchmarks["retrieval_quality"]["verdict"],
        "quality_annotator": benchmarks["retrieval_quality"]["annotator"],
        "quality_labels_hash": benchmarks["retrieval_quality"]["labels_hash"],
        "quality_judgements": benchmarks["retrieval_quality"]["judgements"],
        "quality_queries": benchmarks["retrieval_quality"]["queries"],
        "quality_table": _md_table(
            benchmarks["retrieval_quality"]["rows"],
            [("depth", "depth"), ("system_ndcg", "system nDCG"),
             ("best_blind", "best blind control"),
             ("best_blind_ndcg", "its nDCG"), ("oracle_ndcg", "oracle"),
             ("margin", "margin"), ("required", "required"),
             ("per_query", "per-query W/L/T")],
        ),
        "quality_findings": "\n".join(
            f"- {f}" for f in benchmarks["retrieval_quality"]["findings"]
        ),
        "quality_not_shown": benchmarks["retrieval_quality"]["what_this_does_not_show"],

        "scope_no_labels": benchmarks["scope_limits"]["no_relevance_labels"],
        "scope_underpowered": benchmarks["scope_limits"]["quality_evidence_is_underpowered"],
        "scope_no_quality": benchmarks["scope_limits"]["no_quality_measurement"],
        "scope_advisory_reach": benchmarks["scope_limits"]["advisory_reach_bounds_evaluation"],
        "scope_graph_underpowered": benchmarks["scope_limits"]["graph_decision_is_underpowered"],
        "scope_toy": benchmarks["scope_limits"]["toy_corpus"],
        "scope_scale": benchmarks["scope_limits"]["posting_union_does_not_scale"],
        "scope_oov": benchmarks["scope_limits"]["oov_yields_nothing"],
    }

    context["spec_sha"] = _spec_sha()
    context["spec_sha_short"] = str(context["spec_sha"])[:12]
    context["release"] = RELEASE_VERSION

    if index_path:
        context.update(_index_facts(index_path))
    else:
        context.update({
            "content_hash": "(no index supplied at render time)",
            "content_hash_short": "(none)",
            "node_count": "?", "edge_count": "?", "embedding_count": "?",
            "term_count": "?", "posting_count": "?", "avgdl": "?",
        })
    return context


def _spec_sha() -> str:
    """One hash over every spec file, so a doc can name the spec it came from.

    Delegates to `drf.freeze.spec_sha` rather than recomputing. A second
    implementation of the same hash immediately disagreed with the first: this
    one globbed every spec/*.json including `frozen.json`, which `freeze` must
    exclude because that file *contains* the hash. One concept, one definition.
    """
    from ..freeze import spec_sha

    return spec_sha()


def _index_facts(index_path: str) -> dict:
    from ..store import connect, corpus_totals, read_manifest, table_count

    conn = connect(index_path)
    manifest = read_manifest(conn)
    n_docs, total_len = corpus_totals(conn)
    facts = {
        "content_hash": manifest["content_hash"],
        "content_hash_short": manifest["content_hash"][:12],
        "node_count": table_count(conn, "nodes"),
        "edge_count": table_count(conn, "edges"),
        "embedding_count": table_count(conn, "embeddings"),
        "term_count": table_count(conn, "terms"),
        "posting_count": table_count(conn, "postings"),
        "avgdl": f"{total_len / n_docs:.4f}" if n_docs else "0",
    }
    conn.close()
    return facts


def render_document(audience: str, context: dict) -> str:
    """Render one template. Raises on any placeholder the context lacks."""
    if audience not in AUDIENCES:
        raise ValueError(f"unknown audience {audience!r}; expected {AUDIENCES}")
    template_text = (TEMPLATE_DIR / f"{audience}.md.tmpl").read_text()
    # substitute, never safe_substitute: a missing key must be an error, not a
    # `$placeholder` left in the prose looking like it belongs there.
    return Template(template_text).substitute(context)


def render_all(context: dict) -> dict[str, str]:
    return {audience: render_document(audience, context) for audience in AUDIENCES}


def write_all(context: dict, output_dir: Path | None = None) -> list[Path]:
    directory = output_dir or OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for audience, text in render_all(context).items():
        relative = OUTPUT_PATHS.get(audience)
        path = (ROOT / relative) if relative else directory / f"{audience}.md"
        path.write_text(text)
        written.append(path)
    return written
