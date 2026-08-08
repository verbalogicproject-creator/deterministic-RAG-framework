"""M1.1 checkpoint: the build is reproducible and every source row is accounted for.

Every count asserted here is derived - by `len()`, by `SELECT count(*)`, or by
a query against the source - and then cross-checked against a second,
independent derivation. No expected value is written as a literal unless it
was measured and is named as such, because a test that asserts a number it
also computed proves nothing.

The three source defects are pinned as *regression* expectations, taken from
direct measurement of `claude-cookbook-kg.db` on 2026-08-07:

    48 duplicate (from,to,type) groups, all pairs -> 48 variants removed
     4 dangling edges, all on `to`, none inside a duplicate group
     5 isolated nodes, kept deliberately

If the source is ever cleaned, these tests fail loudly rather than silently
passing over different data - which is the intent.
"""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drf.contract import (  # noqa: E402
    ACTIONS,
    DeterminismViolation,
    reset_replay_log,
    strict_replay,
)
from drf.hashing import sha256_value  # noqa: E402
from drf.ingest import source_kg  # noqa: E402
from drf.ingest.build import build_index  # noqa: E402
from drf.ingest.manifest import reconcile  # noqa: E402
from drf.ingest.normalize import (  # noqa: E402
    CORPUS,
    collapse_edge_group,
    normalize_all,
    normalize_nodes,
)
from drf.store import (  # noqa: E402
    EdgeRecord,
    FORBIDDEN_DDL,
    TABLES,
    connect,
    ddl_text,
    iter_edges,
    iter_nodes,
    read_manifest,
    table_count,
)

# Override with DRF_SOURCE_DB. Hardcoding an absolute path would publish a
# username and make every test skip for anyone else who clones this.
SOURCE = os.environ.get(
    "DRF_SOURCE_DB",
    str(Path.home() / "Downloads/claude-cookbook-kg3/claude-cookbook-kg.db"),
)

# Measured, not assumed. See module docstring.
MEASURED_DUPLICATE_GROUPS = 48
MEASURED_DANGLING_EDGES = 4
MEASURED_ISOLATED_NODES = 5

requires_source = pytest.mark.skipif(
    not os.path.exists(SOURCE), reason=f"source corpus not present at {SOURCE}"
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def source_conn():
    conn = sqlite3.connect(f"file:{SOURCE}?mode=ro", uri=True)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One index, built once, shared by the read-only assertions below."""
    reset_replay_log()
    out = tmp_path_factory.mktemp("index") / "index.db"
    value, justification = build_index(source_path=SOURCE, out_path=str(out))
    conn = connect(str(out))
    yield {"path": str(out), "value": value, "just": justification, "conn": conn}
    conn.close()


# --------------------------------------------------------------------------
# Schema: the determinism defects of the source must not be reproduced
# --------------------------------------------------------------------------

@requires_source
def test_ddl_contains_no_forbidden_constructs(built):
    """Zero AUTOINCREMENT, zero CURRENT_TIMESTAMP - read back from sqlite_master.

    Checked against what SQLite actually stored rather than against the SCHEMA
    constant, so the test inspects the built artefact, not our intent.
    """
    ddl = ddl_text(built["conn"]).upper()
    for keyword in FORBIDDEN_DDL:
        assert keyword.upper() not in ddl, f"DDL contains {keyword!r}"


@requires_source
def test_source_does_contain_what_we_forbid(source_conn):
    """Control: prove the forbidden-keyword check can actually fail.

    A test asserting the absence of a string is worthless unless something
    demonstrates the string would have been found. The source schema contains
    both keywords, so the same check applied to it must fail.
    """
    rows = source_conn.execute(
        "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"
    ).fetchall()
    source_ddl = "\n".join(r[0] for r in rows).upper()
    assert "AUTOINCREMENT" in source_ddl
    assert "CURRENT_TIMESTAMP" in source_ddl


@requires_source
def test_every_table_is_without_rowid(built):
    """No hidden insertion-order column anywhere in the index.

    The expected table count is derived from `store.TABLES`, which is itself
    parsed from `SCHEMA`, rather than written as a literal. An earlier version
    asserted `== 4` and broke the moment M1.2 added the lexical tables - it
    would have been equally happy to pass while a new table silently escaped
    the WITHOUT ROWID requirement, since a literal cannot notice what it does
    not mention.
    """
    ddl = ddl_text(built["conn"]).upper()
    creates = ddl.split("CREATE TABLE")[1:]
    assert len(creates) == len(TABLES), (
        f"sqlite_master has {len(creates)} tables, SCHEMA declares "
        f"{len(TABLES)}: {list(TABLES)}"
    )
    for statement in creates:
        assert "WITHOUT ROWID" in statement


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

@requires_source
def test_two_builds_same_process_agree_via_the_contract(tmp_path):
    """Two builds to *different* paths must produce identical results.

    `ingest.build_index` declares `inputs=("source_path", "corpus")`, so the
    output path is excluded from the inputs hash and both calls land on the
    same key in the replay log. The contract therefore compares the two
    results itself: if the build depended on the output path, the wall clock,
    or dict iteration order, this raises DeterminismViolation before the
    assertion below is even reached.
    """
    reset_replay_log()
    first, _ = build_index(source_path=SOURCE, out_path=str(tmp_path / "a.db"))
    second, _ = build_index(source_path=SOURCE, out_path=str(tmp_path / "b.db"))
    assert first["content_hash"] == second["content_hash"]


@requires_source
def test_build_survives_strict_replay(tmp_path):
    """Under strict_replay the action is invoked twice and its results compared."""
    reset_replay_log()
    with strict_replay():
        value, _ = build_index(source_path=SOURCE, out_path=str(tmp_path / "s.db"))
    assert value["content_hash"]


@requires_source
@pytest.mark.parametrize("seed", ["0", "1", "12345"])
def test_content_hash_is_stable_across_hash_seeds(tmp_path, built, seed):
    """PYTHONHASHSEED must not reach the manifest.

    Set iteration order and dict ordering vary with the seed. Any place the
    build enumerates a set without sorting would show up here and nowhere
    else, which is why this runs in a subprocess rather than in-process.
    """
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from drf.ingest.build import build_index\n"
        "v, _ = build_index(source_path=%r, out_path=%r)\n"
        "print(v['content_hash'])\n"
    ) % (str(ROOT), SOURCE, str(tmp_path / f"seed{seed}.db"))
    env = dict(os.environ, PYTHONHASHSEED=seed)
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == built["value"]["content_hash"]


@requires_source
def test_manifest_hash_covers_content_and_excludes_provenance(built):
    """content_hash must be recomputable from `content` alone."""
    manifest = read_manifest(built["conn"])
    assert sha256_value(manifest["content"]) == manifest["content_hash"]
    # Provenance exists, carries the absolute path, and is outside the hash.
    assert manifest["provenance"]["source_path"] == os.path.abspath(SOURCE)
    assert "provenance" not in manifest["content"]


# --------------------------------------------------------------------------
# Counts: manifest vs database vs source
# --------------------------------------------------------------------------

@requires_source
def test_manifest_counts_match_database(built):
    """Every count in the manifest is confirmed by SELECT count(*)."""
    manifest = read_manifest(built["conn"])
    for table in ("nodes", "edges", "embeddings"):
        assert manifest["content"]["counts"][table] == table_count(built["conn"], table)


@requires_source
def test_all_source_nodes_survive(built, source_conn):
    """No node is lost. Both sides derived, neither written as a literal."""
    source_count = source_conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
    assert table_count(built["conn"], "nodes") == source_count


@requires_source
def test_node_ids_are_unique_and_content_addressed(built):
    ids = [n.id for n in iter_nodes(built["conn"])]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("n_") for i in ids)


@requires_source
def test_edge_reconciliation_balances(built, source_conn):
    """605 = 553 written + 48 collapsed + 4 dropped, all derived.

    This is the identity that makes the drop and collapse counts trustworthy:
    it proves no source row was double-counted or silently lost.
    """
    normalized = normalize_all(
        source_kg.read_nodes(source_conn),
        source_kg.read_edges(source_conn),
        source_kg.read_embeddings(source_conn),
    )
    fingerprint = source_kg.source_fingerprint(source_conn)
    balance = reconcile(normalized, fingerprint)
    assert balance["balanced"]
    assert (
        balance["edges_written"]
        + balance["collapsed_variants_removed"]
        + balance["edges_dropped"]
        == balance["edges_read"]
    )


# --------------------------------------------------------------------------
# The three source defects
# --------------------------------------------------------------------------

@requires_source
def test_duplicate_groups_collapse_structurally(built, source_conn):
    """48 duplicate groups collapse, and the count is derived from the source."""
    source_dup_groups = source_conn.execute(
        "SELECT count(*) FROM (SELECT 1 FROM edges"
        " GROUP BY from_node, to_node, type HAVING count(*) > 1)"
    ).fetchone()[0]
    assert source_dup_groups == MEASURED_DUPLICATE_GROUPS

    manifest = read_manifest(built["conn"])
    assert manifest["content"]["counts"]["collapsed_groups"] == source_dup_groups

    # Structural, not a dedupe pass: edge ids are unique by construction.
    edge_ids = [e.id for e in iter_edges(built["conn"])]
    assert len(set(edge_ids)) == len(edge_ids)


@requires_source
def test_the_four_divergent_groups_kept_their_metadata(built):
    """The conflict-free union preserved the semantic-analysis provenance.

    Measured: 4 of the 48 groups had one variant with metadata and one with
    NULL. Union means the rich payload survives; "strip metadata" or an
    unlucky "pick one" rule would lose it, and this test is what tells them
    apart.
    """
    manifest = read_manifest(built["conn"])
    divergent = [c for c in manifest["content"]["collapsed"] if c["payload_divergent"]]
    assert len(divergent) == 4

    by_id = {e.id: e for e in iter_edges(built["conn"])}
    for group in divergent:
        metadata = by_id[group["edge_id"]].metadata
        assert metadata, f"{group['edge_id']} lost its payload in the collapse"
        assert "confidence" in metadata and "reasoning" in metadata


@requires_source
def test_dangling_edges_are_dropped_and_named(built, source_conn):
    """4 orphan edges appear in manifest.dropped with enough detail to fix them."""
    source_orphans = source_conn.execute(
        "SELECT count(*) FROM edges e WHERE NOT EXISTS"
        " (SELECT 1 FROM nodes n WHERE n.id = e.from_node)"
        " OR NOT EXISTS (SELECT 1 FROM nodes n WHERE n.id = e.to_node)"
    ).fetchone()[0]
    assert source_orphans == MEASURED_DANGLING_EDGES

    manifest = read_manifest(built["conn"])
    dangling = [
        d for d in manifest["content"]["dropped"]
        if d["kind"] == "edge" and d["reason"] == "dangling_endpoint"
    ]
    assert len(dangling) == source_orphans
    for record in dangling:
        assert record["detail"]["from"] and record["detail"]["to"]
        assert record["detail"]["missing"] in ("from", "to")


@requires_source
def test_dangling_edges_were_not_silently_repaired(built, source_conn):
    """No dangling edge reappears in the index under a fuzzy-matched target.

    Three of the four dangling targets have plausible near-matches in the
    corpus. Resolving them by string similarity would be a guess inside a
    build that claims determinism, so the index must contain no edge whose
    source slug pair matches a dangling one.

    The dangling set is derived from the **source**, deliberately. An earlier
    version of this test read it from `manifest.dropped`, which made it
    vacuous in exactly the case it was written for: if the build repaired the
    orphans, nothing would be dropped, the set would be empty, and the empty
    set is disjoint from everything. The falsifier registry caught that -
    the fuzzy-repair mutation left this test green.

    Verified against the source: all four dangling `(from, type)` pairs have
    zero *valid* edges, so a live match can only mean a repair happened.
    """
    dangling_pairs = {
        (row[0], row[1]) for row in source_conn.execute(
            "SELECT from_node, type FROM edges e WHERE NOT EXISTS"
            " (SELECT 1 FROM nodes n WHERE n.id = e.to_node)"
            " OR NOT EXISTS (SELECT 1 FROM nodes n WHERE n.id = e.from_node)"
        )
    }
    assert len(dangling_pairs) == MEASURED_DANGLING_EDGES

    by_ref = {n.id: n.source_ref for n in iter_nodes(built["conn"])}
    live_pairs = {
        (by_ref[e.from_id], e.type) for e in iter_edges(built["conn"])
    }
    assert dangling_pairs.isdisjoint(live_pairs), (
        f"a dangling edge was repaired rather than dropped: "
        f"{sorted(dangling_pairs & live_pairs)}"
    )


@requires_source
def test_isolated_nodes_are_kept(built):
    """5 nodes have no edges and remain searchable."""
    linked = set()
    for edge in iter_edges(built["conn"]):
        linked.add(edge.from_id)
        linked.add(edge.to_id)
    all_ids = {n.id for n in iter_nodes(built["conn"])}
    assert len(all_ids - linked) == MEASURED_ISOLATED_NODES


@requires_source
def test_embeddings_are_rekeyed_and_single_space(built, source_conn):
    """228 vectors carried over, re-keyed, all in one embedding space."""
    source_count = source_conn.execute(
        "SELECT count(*) FROM node_embeddings"
    ).fetchone()[0]
    assert table_count(built["conn"], "embeddings") == source_count

    manifest = read_manifest(built["conn"])
    assert manifest["content"]["embedding_dims"] == [384]
    assert len(manifest["content"]["embedding_models"]) == 1


# --------------------------------------------------------------------------
# The collapse rule itself
# --------------------------------------------------------------------------

def _edge(metadata: dict, weight_q: int = 1_000_000_000) -> EdgeRecord:
    return EdgeRecord(
        id="e_test", from_id="n_a", to_id="n_b", type="enables",
        weight_q=weight_q, metadata=metadata,
    )


def test_collapse_is_commutative():
    """The rule's output must not depend on the order variants arrive in.

    This is the property that distinguishes conflict-free union from the
    alternatives considered: "richest wins" and "last write wins" are
    order-independent only because the caller sorts first. Here there is no
    ordering dependency to preserve, so the caller's sort is a convenience.
    """
    rich = _edge({"confidence": 0.98, "reasoning": "because"})
    bare = _edge({})
    forward = collapse_edge_group([rich, bare])
    reverse = collapse_edge_group([bare, rich])
    assert forward.metadata == reverse.metadata == rich.metadata


def test_collapse_is_commutative_over_three_disjoint_variants():
    """Union keeps every key regardless of visit order; "pick one" would not."""
    import itertools
    variants = [_edge({"a": 1}), _edge({"b": 2}), _edge({"c": 3})]
    results = {
        json.dumps(collapse_edge_group(list(p)).metadata, sort_keys=True)
        for p in itertools.permutations(variants)
    }
    assert len(results) == 1
    assert json.loads(results.pop()) == {"a": 1, "b": 2, "c": 3}


def test_collapse_raises_on_genuine_metadata_conflict():
    """A real contradiction stops the build instead of picking a winner."""
    with pytest.raises(ValueError, match="conflicting metadata"):
        collapse_edge_group([_edge({"confidence": 0.9}), _edge({"confidence": 0.5})])


def test_collapse_raises_on_divergent_weight():
    with pytest.raises(ValueError, match="divergent weight"):
        collapse_edge_group([_edge({}, weight_q=1), _edge({}, weight_q=2)])


def test_collapse_of_a_single_variant_is_identity():
    only = _edge({"x": 1})
    assert collapse_edge_group([only]) is only


@requires_source
def test_source_has_no_genuine_metadata_conflicts(source_conn):
    """Measured claim, re-derived: 0 of the 48 groups genuinely conflict.

    The collapse rule is only lossless because this holds. If a future source
    revision introduces a real conflict, this fails here with a clear reason
    rather than surfacing as an opaque build error.
    """
    groups = source_conn.execute(
        "SELECT from_node, to_node, type FROM edges"
        " GROUP BY 1,2,3 HAVING count(*) > 1"
    ).fetchall()
    conflicts = 0
    for from_node, to_node, edge_type in groups:
        rows = source_conn.execute(
            "SELECT metadata FROM edges WHERE from_node=? AND to_node=? AND type=?",
            (from_node, to_node, edge_type),
        ).fetchall()
        seen: dict = {}
        for (raw,) in rows:
            for key, value in (json.loads(raw) if raw else {}).items():
                if key in seen and seen[key] != value:
                    conflicts += 1
                seen[key] = value
    assert conflicts == 0
    assert len(groups) == MEASURED_DUPLICATE_GROUPS


# --------------------------------------------------------------------------
# Node id injectivity
# --------------------------------------------------------------------------

def test_colliding_nodes_raise_rather_than_silently_merge():
    """Two nodes indistinguishable under the id recipe must stop the build."""
    from drf.ingest.source_kg import SourceNode

    duplicated = [
        SourceNode("slug_a", "pattern", "Same Name", "Same description"),
        SourceNode("slug_b", "pattern", "Same Name", "Same description"),
    ]
    with pytest.raises(ValueError, match="collision"):
        normalize_nodes(duplicated)


@requires_source
def test_source_has_no_node_collisions(source_conn):
    collisions = source_conn.execute(
        "SELECT count(*) FROM (SELECT 1 FROM nodes"
        " GROUP BY type, name, description HAVING count(*) > 1)"
    ).fetchone()[0]
    assert collisions == 0


# --------------------------------------------------------------------------
# spec <-> code bijection, code direction
# --------------------------------------------------------------------------

def _import_all_action_modules() -> None:
    """Import every module under `drf/`, so ACTIONS is fully populated.

    Registration is an import side effect, so without this the two tests below
    would measure *which test modules happened to be collected* rather than
    what the package implements - passing in a full run and failing when
    `test_ingest.py` is run alone.

    **Walks the package rather than listing modules.** The first version of
    this helper hand-listed five modules and then silently went stale: M1.4
    added `drf.retrieval.neural` and nobody updated the list, so
    `neural.propose_from_anchors` looked unimplemented whenever this file ran
    alone. A hand-maintained list drifting is precisely the failure the helper
    exists to prevent, so it no longer keeps one.
    """
    import importlib
    import pkgutil

    import drf

    for module in pkgutil.walk_packages(drf.__path__, prefix="drf."):
        # providers/base.py defines a Protocol only; importing everything is
        # still correct - a module with no @action simply registers nothing.
        importlib.import_module(module.name)


def test_registered_actions_appear_in_spec_with_matching_axes():
    """Now that real actions exist, the code -> spec direction can be asserted.

    Filtered to actions *defined in the drf package*. `test_contract.py`
    registers throwaway actions from inside test bodies, so the global
    registry is polluted by whatever has run before this. Discriminating on
    the defining module is robust to that, and to test-ordering changes, in a
    way that a name-prefix convention would not be.
    """
    _import_all_action_modules()
    with open(ROOT / "spec" / "actions.json") as f:
        spec = {e["name"]: e for e in json.load(f)["actions"]}

    implemented = {
        name: registered for name, registered in ACTIONS.items()
        if registered.func.__module__.startswith("drf.")
    }
    assert implemented, "no drf actions registered - import drift"
    for name, registered in implemented.items():
        assert name in spec, f"action {name!r} is implemented but not in spec"
        assert registered.determinism == spec[name]["determinism"], name
        assert registered.authority == spec[name]["authority"], name


def test_spec_actions_not_yet_implemented_are_declared_not_forgotten():
    """The other direction, scoped honestly to the current milestone.

    The full bijection cannot hold until M1.8. Rather than skip the check,
    this pins *which* actions are still unimplemented, so landing one without
    a spec entry - or deleting a spec entry - fails here.
    """
    _import_all_action_modules()
    with open(ROOT / "spec" / "actions.json") as f:
        spec_names = {e["name"] for e in json.load(f)["actions"]}
    implemented = {
        name for name, registered in ACTIONS.items()
        if registered.func.__module__.startswith("drf.")
    }
    pending = spec_names - implemented
    assert pending == {
        # Deliberately unregistered, with reasons - not oversights.
        #
        # merge.append_advisory: merge() is deterministic given (D, proposals),
        #   but the proposals arrive boxed in Advisory[T] and the box may only
        #   be opened inside merge itself. Declaring `inputs` for the replay
        #   check would therefore require either unwrapping outside the
        #   allowlist or hashing something that is not the real input. The
        #   guarantee is enforced instead by merge's runtime postcondition,
        #   which runs on every query - a stronger check than replay, since it
        #   verifies the property rather than the repeatability.
        "merge.append_advisory",
        # neural.encode_query_remote: RemoteHTTPProvider was cut from M1 by the
        #   build plan. BGE is 1024-d and the corpus vectors are 384-d MiniLM -
        #   different spaces that can never be compared - so a remote provider
        #   needs the whole corpus re-embedded first. Deferred to M2.
        "neural.encode_query_remote",
        # store.load_manifest: the manifest is read directly by
        #   store.read_manifest; there is no second reader to keep honest yet.
        "store.load_manifest",
    }, f"milestone drift: pending set changed to {sorted(pending)}"


@requires_source
def test_build_index_justification_declares_deterministic_and_authoritative(built):
    justification = built["just"]
    assert justification.action == "ingest.build_index"
    assert justification.determinism == "deterministic"
    assert justification.authority == "authoritative"
    assert justification.confidence is None
    assert justification.evidence
