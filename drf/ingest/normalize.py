"""Map source rows onto content-addressed index records.

This is where the source graph's three known data defects are handled. All
three are handled *explicitly and countably* - every record that does not make
it into the index is returned in a `dropped` list with a machine-readable
reason, and those counts land in the manifest. Silent repair is the failure
mode this module exists to prevent.

Measured against `claude-cookbook-kg.db` on 2026-08-07:

    266 nodes, 605 edges, 228 embeddings
    48 duplicate (from,to,type) groups, 96 rows -> 48 survivors
       44 groups byte-identical in payload
        4 groups divergent: one variant carries semantic-analysis metadata,
          the other carries NULL
     4 orphan edges (all dangling on `to`, none inside a duplicate group)
     5 isolated nodes (no edge in either direction) - kept, see below
     0 self-loops
     0 node collisions on (type, name, description)

Stdlib only.
"""

from typing import NamedTuple

from ..fixed import quantize
from ..hashing import canonical_json, edge_id, node_id
from ..store import EdgeRecord, EmbeddingRecord, NodeRecord
from .source_kg import SourceEdge, SourceEmbedding, SourceNode

# Logical corpus name. Part of every node id, so it must be a stable label -
# never a filesystem path, or the same content ingested from a different
# directory would produce different ids.
CORPUS = "claude-cookbook-kg"


class Dropped(NamedTuple):
    """One record that did not enter the index, and why."""
    kind: str      # "edge" | "node" | "embedding"
    reason: str    # machine-readable code
    detail: dict   # enough to locate it in the source


class Normalized(NamedTuple):
    nodes: list[NodeRecord]
    edges: list[EdgeRecord]
    embeddings: list[EmbeddingRecord]
    dropped: list[Dropped]
    collapsed: list[dict]   # duplicate groups that merged, with variant counts


def _parse_metadata(raw: str | None) -> dict:
    """Source metadata is a JSON text column that is sometimes NULL.

    A malformed value is *not* swallowed - it becomes an explicit marker so it
    can be counted, rather than vanishing into an empty dict.
    """
    import json
    if raw is None or raw == "":
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_unparsed": raw}
    return value if isinstance(value, dict) else {"_value": value}


def normalize_nodes(source_nodes: list[SourceNode]) -> tuple[list[NodeRecord], dict]:
    """Assign content-addressed ids and build the source-slug -> new-id map.

    Raises if two source nodes map to the same content id. That would mean two
    records with identical (type, name, description) - indistinguishable to
    this schema, so keeping both is impossible and dropping one silently is
    unacceptable. Measured as 0 for this corpus, but asserted rather than
    assumed, because a future corpus may differ.
    """
    records: list[NodeRecord] = []
    id_map: dict[str, str] = {}
    for sn in source_nodes:
        nid = node_id(
            type=sn.type, name=sn.name, description=sn.description, source=CORPUS
        )
        if nid in id_map.values():
            clash = next(k for k, v in id_map.items() if v == nid)
            raise ValueError(
                f"node id collision: {sn.src_id!r} and {clash!r} both map to {nid}. "
                "Two nodes share (type, name, description); the id recipe cannot "
                "distinguish them."
            )
        id_map[sn.src_id] = nid
        records.append(NodeRecord(
            id=nid,
            type=sn.type,
            name=sn.name,
            description=sn.description,
            source=CORPUS,
            source_ref=sn.src_id,
            metadata={},
        ))
    records.sort(key=lambda r: r.id)
    return records, id_map


def collapse_edge_group(variants: list[EdgeRecord]) -> EdgeRecord:
    """Reduce several source rows sharing one content-addressed edge id to one.

    `variants` arrives sorted by canonical payload, so the sequence is a pure
    function of the data - it does not depend on the source's rowids, on
    insertion history, or on the order rows came back from SQLite. Whatever
    this function does must preserve that property: reading `variants[0]` is
    fine because position 0 is content-determined, but any appeal to a source
    integer id would reintroduce the non-determinism this framework removes.

    In this corpus 44 of the 48 groups are byte-identical, so any rule returns
    the same answer for them. The 4 divergent groups all have the same shape:

        variant A  metadata = {"confidence": 0.95, "reasoning": "...",
                               "created_by": "claude_code_semantic_analysis"}
        variant B  metadata = {}

    Weight never diverges (measured: 0 groups), but the rule is still total
    over the inputs it might see.

    **The rule: conflict-free union.** Variants are merged key by key; if two
    variants ever assign *different* values to the *same* key, that is a real
    contradiction in the source and the build stops rather than picking a
    winner. Measured on this corpus: 44 groups byte-identical, 4 divergent but
    conflict-free (one variant is `{}`), **0 genuinely conflicting**. So no
    information is lost and nothing is invented - the four rich payloads
    survive intact and the empty variants contribute nothing.

    The property that makes this the right rule is not that it is
    deterministic - all the alternatives were - but that it is *commutative*.
    Union over a conflict-free key set gives the same answer regardless of the
    order the variants are visited, so this function's output does not depend
    on the caller's sort at all. The alternatives ("richest wins", "last write
    wins") are order-independent only because the caller sorts first; their
    correctness rests on that sort staying exactly as written. Here there is
    no ordering dependency to preserve in the first place.

    The count of merged variants is deliberately *not* written into the
    returned metadata. A record in the index should say what the source said;
    build-time bookkeeping belongs in the manifest's `collapsed` list, where
    `normalize_edges` puts it.
    """
    if len(variants) == 1:
        return variants[0]

    weights = {v.weight_q for v in variants}
    if len(weights) > 1:
        raise ValueError(
            f"divergent weight on duplicate edge {variants[0].id}: "
            f"{sorted(weights)}. Content-identical edges disagree on weight; "
            "the source must be corrected."
        )

    merged: dict = {}
    for variant in variants:
        for key, value in variant.metadata.items():
            if key in merged and merged[key] != value:
                raise ValueError(
                    f"conflicting metadata on duplicate edge {variants[0].id}: "
                    f"key {key!r} is both {merged[key]!r} and {value!r}. "
                    "No content-derived rule can choose between them; the "
                    "source must be corrected."
                )
            merged[key] = value

    return variants[0]._replace(metadata=merged)


def resolve_endpoint(src_ref: str, id_map: dict[str, str]) -> str | None:
    """Map a source slug to a content id, or None if it does not exist.

    The single place endpoint resolution policy lives. It is - and must remain
    - an exact lookup. Anything cleverer here (prefix matching, edit distance,
    "did you mean") would be a heuristic guess inside a build that claims
    determinism, and it would be invisible: a wrongly-resolved edge produces a
    perfectly valid-looking index that silently corrupts every traversal
    crossing it.

    Isolated as a named function so that `spec/invariants.json` can falsify it
    - the fuzzy-repair falsifier replaces exactly this, and
    `test_dangling_edges_were_not_silently_repaired` must fail when it does.
    """
    return id_map.get(src_ref)


def normalize_edges(
    source_edges: list[SourceEdge],
    id_map: dict[str, str],
) -> tuple[list[EdgeRecord], list[Dropped], list[dict]]:
    """Re-key edges onto content ids, drop danglers, collapse duplicates.

    Order of operations matters and is deliberate: **drop orphans first, then
    collapse.** An orphan cannot be re-keyed at all (its endpoint has no
    content id), so it can never reach a group. Measured: the two sets do not
    intersect - 0 of the 4 orphans sit in a duplicate group - so the manifest's
    `dropped` and `collapsed` counts partition the 605 source rows cleanly and
    cannot double-count.

    Orphans are dropped, never repaired. Three of the four dangling targets
    have plausible near-matches in the corpus (`prompt_caching` ->
    `claude_technique_prompt_caching`), and resolving them by string similarity
    would be a heuristic - a guess with no ground truth, injected into an index
    that claims to be deterministic. A dropped edge is visible in the manifest
    and can be fixed at the source; a wrongly-repaired edge is invisible and
    corrupts every traversal that crosses it.
    """
    dropped: list[Dropped] = []
    groups: dict[str, list[EdgeRecord]] = {}

    for se in source_edges:
        from_id = resolve_endpoint(se.from_src, id_map)
        to_id = resolve_endpoint(se.to_src, id_map)
        if from_id is None or to_id is None:
            dropped.append(Dropped(
                kind="edge",
                reason="dangling_endpoint",
                detail={
                    "from": se.from_src,
                    "to": se.to_src,
                    "type": se.type,
                    "missing": "from" if from_id is None else "to",
                },
            ))
            continue

        eid = edge_id(from_id=from_id, to_id=to_id, type=se.type)
        groups.setdefault(eid, []).append(EdgeRecord(
            id=eid,
            from_id=from_id,
            to_id=to_id,
            type=se.type,
            weight_q=quantize(se.weight),
            metadata=_parse_metadata(se.metadata),
        ))

    records: list[EdgeRecord] = []
    collapsed: list[dict] = []
    for eid in sorted(groups):
        variants = sorted(groups[eid], key=lambda e: canonical_json(e.metadata))
        if len(variants) > 1:
            payloads = {canonical_json([v.weight_q, v.metadata]) for v in variants}
            collapsed.append({
                "edge_id": eid,
                "variants": len(variants),
                "payload_divergent": len(payloads) > 1,
            })
        records.append(collapse_edge_group(variants))

    records.sort(key=lambda r: r.id)
    return records, dropped, collapsed


def normalize_embeddings(
    source_embeddings: list[SourceEmbedding],
    id_map: dict[str, str],
) -> tuple[list[EmbeddingRecord], list[Dropped]]:
    """Re-key frozen vectors onto content ids.

    The vectors are copied byte-for-byte. Nothing here runs a model, and the
    dimension is recorded per row so that mixing incompatible spaces - the
    384-d MiniLM vectors stored here versus 1024-d BGE - is detectable rather
    than silently averaged.
    """
    records: list[EmbeddingRecord] = []
    dropped: list[Dropped] = []
    for se in source_embeddings:
        nid = id_map.get(se.src_id)
        if nid is None:
            dropped.append(Dropped(
                kind="embedding",
                reason="unknown_node",
                detail={"node": se.src_id, "model": se.model},
            ))
            continue
        records.append(EmbeddingRecord(
            node_id=nid, model=se.model, dim=se.dim, vector=se.vector
        ))
    records.sort(key=lambda r: r.node_id)
    return records, dropped


def normalize_all(
    source_nodes: list[SourceNode],
    source_edges: list[SourceEdge],
    source_embeddings: list[SourceEmbedding],
) -> Normalized:
    """Full normalisation pass.

    Isolated nodes are kept deliberately. A node with no edges is still a
    legitimate BM25 hit - it simply contributes nothing to graph expansion.
    Dropping it would silently shrink the searchable corpus to improve a
    connectivity statistic, which is the wrong trade in a retrieval system.
    Measured: 5 such nodes, 4 atomic capabilities and 1 use case.
    """
    nodes, id_map = normalize_nodes(source_nodes)
    edges, edge_dropped, collapsed = normalize_edges(source_edges, id_map)
    embeddings, emb_dropped = normalize_embeddings(source_embeddings, id_map)
    return Normalized(
        nodes=nodes,
        edges=edges,
        embeddings=embeddings,
        dropped=edge_dropped + emb_dropped,
        collapsed=collapsed,
    )
