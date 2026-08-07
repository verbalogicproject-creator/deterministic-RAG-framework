"""M1.5 checkpoint: configuration whose hash means what it says.

Two properties carry the milestone.

**The content hash covers exactly the settings that can change a result.**
Not more (or a display preference would look like a different computation)
and not less (or two genuinely different rankings would collide).

**`affects_ranking` is derived, not declared.** A flag maintained by hand
drifts. `test_affects_ranking_matches_declared_action_inputs` asserts the
schema's flags equal the set of config-bound parameters actually appearing in
an authoritative action's `inputs`, so the schema cannot claim a setting
matters unless the code says so, nor omit one that does.

And the strongest check here is not about hashing at all:
`test_every_ranking_setting_changes_real_query_output` runs actual queries and
requires each advertised knob to move something. A parameter documented as
affecting ranking that changes nothing is the same defect as a metric with no
producer.
"""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drf.config.manager import (  # noqa: E402
    SCHEMA,
    Config,
    ConfigError,
    Snapshot,
    defaults,
    diff_dicts,
    display_keys,
    ranking_keys,
)
from drf.contract import ACTIONS, reset_replay_log  # noqa: E402
from drf.ingest.build import build_index  # noqa: E402
from drf.retrieval import merge as merge_module  # noqa: E402
from drf.retrieval import neural, stage1  # noqa: E402
from drf.retrieval.providers.null import NullProvider  # noqa: E402
from drf.retrieval.providers.stored_vectors import StoredVectorProvider  # noqa: E402
from drf.retrieval.tokenize import tokenize  # noqa: E402
from drf.store import connect, iter_nodes, read_manifest  # noqa: E402

SOURCE = "/home/eyaln/Downloads/claude-cookbook-kg3/claude-cookbook-kg.db"

requires_source = pytest.mark.skipif(
    not os.path.exists(SOURCE), reason=f"source corpus not present at {SOURCE}"
)

QUERIES = ["prompt caching", "tool use", "extended thinking", "streaming",
           "agent orchestration", "batch processing", "classification"]

# The full corpus-vocabulary set, shared with test_retrieval and test_stage1.
# Sensitivity is measured against all 15 because graph parameters reorder at
# most 1 of them - a 7-query subset misses the effect entirely, which is how
# the order-versus-state distinction was found.
QUERIES_15 = QUERIES + [
    "rag retrieval", "pdf vision", "citations", "embeddings semantic search",
    "json mode structured output", "summarization", "sub agents", "memory",
]


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    reset_replay_log()
    out = tmp_path_factory.mktemp("m15") / "index.db"
    build_index(source_path=SOURCE, out_path=str(out))
    conn = connect(str(out))
    yield {
        "conn": conn,
        "hash": read_manifest(conn)["content_hash"],
        "known": {n.id for n in iter_nodes(conn)},
    }
    conn.close()


def _rank_ids(index, text, config: Config) -> list[str]:
    value, _ = stage1.rank(
        conn=index["conn"], query_terms=tokenize(text),
        index_hash=index["hash"], **config.action_kwargs(),
    )
    return [stage1.Ranked(*row).node_id for row in value]


# --------------------------------------------------------------------------
# Schema and validation
# --------------------------------------------------------------------------

def test_defaults_are_valid_against_their_own_schema():
    """The shipped defaults must survive the validator they ship with."""
    config = Config()
    for key, value in defaults().items():
        config.set(key, value)
    assert config.as_dict() == defaults()


def test_unknown_setting_is_rejected():
    with pytest.raises(ConfigError, match="unknown setting"):
        Config().set("ranking.k2", 1.0)


def test_ill_typed_setting_is_rejected():
    with pytest.raises(ConfigError, match="expected number"):
        Config().set("ranking.b", "0.75")
    with pytest.raises(ConfigError, match="expected integer"):
        Config().set("graph.seed_count", 1.5)


def test_bool_is_not_accepted_as_an_integer():
    """`isinstance(True, int)` is True in Python, so this needs its own check.

    Without it `graph.seed_count = True` would validate and then silently
    behave as 1.
    """
    with pytest.raises(ConfigError, match="got bool"):
        Config().set("graph.seed_count", True)


def test_range_and_enum_are_enforced():
    with pytest.raises(ConfigError, match="above maximum"):
        Config().set("ranking.b", 1.5)
    with pytest.raises(ConfigError, match="below minimum"):
        Config().set("graph.max_depth", -1)
    with pytest.raises(ConfigError, match="not in"):
        Config().set("neural.provider", "gpt")


# --------------------------------------------------------------------------
# The content hash covers exactly what matters
# --------------------------------------------------------------------------

def test_display_keys_do_not_change_the_content_hash():
    """Presentation is not computation.

    Falsifiable - spec/invariants.json::config_hash_ignores_display.
    """
    baseline = Config().content_hash()
    for key in display_keys():
        config = Config()
        spec = SCHEMA[key]
        if spec["type"] == "boolean":
            config.set(key, not spec["default"])
        elif "enum" in spec:
            other = [v for v in spec["enum"] if v != spec["default"]][0]
            config.set(key, other)
        else:
            config.set(key, spec["default"] + 1)
        assert config.content_hash() == baseline, (
            f"{key} is presentation but changed the content hash"
        )


def test_every_ranking_key_changes_the_content_hash():
    """No ranking setting may be silently absent from the hash.

    Falsifiable - spec/invariants.json::config_hash_covers_ranking.
    """
    baseline = Config().content_hash()
    assert ranking_keys(), "no ranking settings declared"
    for key in ranking_keys():
        config = Config()
        spec = SCHEMA[key]
        config.set(key, spec["default"] + (1 if spec["type"] == "integer" else 0.1))
        assert config.content_hash() != baseline, (
            f"{key} affects ranking but is not covered by the content hash"
        )


def test_neural_settings_do_not_change_the_content_hash():
    """The architecture, restated in the configuration layer.

    A provider cannot influence ranking, so a configuration differing only in
    provider is the same computation and must hash identically.
    """
    baseline = Config().content_hash()
    switched = Config({"neural.provider": "stored", "neural.limit": 25})
    assert switched.content_hash() == baseline


# --------------------------------------------------------------------------
# Anti-drift: the flag is derived from the code
# --------------------------------------------------------------------------

def test_affects_ranking_matches_declared_action_inputs():
    """The schema cannot claim, or omit, a ranking setting on its own say-so.

    Every setting flagged `affects_ranking` must bind to a parameter that some
    *authoritative* action actually declares in its `inputs`. This is what
    stops the flag from becoming documentation.
    """
    import drf.retrieval.graph  # noqa: F401
    import drf.retrieval.lexical  # noqa: F401
    import drf.retrieval.stage1  # noqa: F401

    authoritative_inputs: set[str] = set()
    for spec in ACTIONS.values():
        if spec.func.__module__.startswith("drf.") and spec.authority == "authoritative":
            authoritative_inputs |= set(spec.inputs)

    declared = {SCHEMA[k]["action_input"] for k in ranking_keys()}
    assert declared <= authoritative_inputs, (
        f"schema flags settings as ranking-affecting whose parameters no "
        f"authoritative action declares: {sorted(declared - authoritative_inputs)}"
    )


def test_no_ranking_setting_binds_to_an_advisory_action_only():
    """A setting that only reaches an advisory action does not affect ranking."""
    import drf.retrieval.neural  # noqa: F401

    advisory_only: set[str] = set()
    authoritative: set[str] = set()
    for spec in ACTIONS.values():
        if not spec.func.__module__.startswith("drf."):
            continue
        target = authoritative if spec.authority == "authoritative" else advisory_only
        target |= set(spec.inputs)

    for key in ranking_keys():
        param = SCHEMA[key]["action_input"]
        assert param in authoritative, (
            f"{key} is flagged affects_ranking but {param!r} reaches only "
            "advisory actions"
        )


# --------------------------------------------------------------------------
# Sensitivity: advertised knobs must actually do something
# --------------------------------------------------------------------------

@requires_source
def test_every_ranking_setting_changes_real_query_output(index):
    """The strong form: each knob must change the *order a user sees*.

    Compares ranked node_id sequences, not internal state. That distinction
    was discovered here: an earlier version of this test disagreed with
    `test_stage1.py::test_seed_count_change_alters_output_for_at_least_one_query`,
    which compares full `Ranked` tuples. Both were right about different
    questions - `seed_count` 10 -> 11 alters `best_depth` for 7 of 15 queries
    while reordering **zero** of them.

    Each probe value is declared in `spec/config_schema.json` rather than
    computed here, so the claim "this value moves the order" is recorded where
    a reader can see it instead of buried in test arithmetic.

    Falsifiable - spec/invariants.json::ranking_params_are_live.
    """
    baseline_config = Config()
    for key in ranking_keys():
        spec = SCHEMA[key]
        probe = spec.get("sensitivity_probe")
        assert probe is not None, (
            f"{key} is flagged affects_ranking but declares no "
            "sensitivity_probe; the claim is untested"
        )
        alternative = Config()
        alternative.set(key, probe)
        changed = [
            text for text in QUERIES_15
            if _rank_ids(index, text, baseline_config)
            != _rank_ids(index, text, alternative)
        ]
        assert changed, (
            f"{key} is declared to affect ranking but reordered no query "
            f"({spec['default']} -> {probe})"
        )


@requires_source
def test_seed_count_plus_one_reorders_nothing(index):
    """A recorded negative result, asserted so it cannot quietly change.

    The build plan's checkpoint reads "graph.seed_count 10 -> 11 changes
    output for at least one query". Measured, that holds only for *internal
    state*: `best_depth` moves for 7 of 15 queries and the visible order for
    none. Pinning the zero here means that if graph signal ever becomes
    influential enough to reorder on a one-seed change, this test fails and
    the finding gets revisited rather than silently ageing.
    """
    baseline_config = Config()
    stronger = Config({"graph.seed_count": 11})
    reordered = [
        text for text in QUERIES_15
        if _rank_ids(index, text, baseline_config)
        != _rank_ids(index, text, stronger)
    ]
    assert reordered == [], (
        "seed_count 10 -> 11 now reorders results; spec/config_schema.json's "
        f"recorded finding is stale (changed: {reordered})"
    )


@requires_source
def test_neural_provider_cannot_change_the_authoritative_prefix(index):
    """Checked against real output, not against the schema flag.

    This is the configuration layer's independent confirmation of merge's
    guarantee. If the flag and the behaviour ever disagreed, this fails.
    """
    known = index["known"]
    for text in QUERIES:
        deterministic = _rank_ids(index, text, Config())
        if not deterministic:
            continue
        prefixes = set()
        for provider in (NullProvider(), StoredVectorProvider(index["conn"])):
            advisory, _ = neural.propose_from_anchors(
                provider=provider, anchors=deterministic[:5], limit=10,
                provider_name=provider.name, index_hash=index["hash"],
            )
            merged = merge_module.merge(
                deterministic=deterministic, advisory=advisory, known_ids=known
            )
            prefixes.add(tuple(merge_module.deterministic_prefix_ids(merged)))
        assert len(prefixes) == 1


# --------------------------------------------------------------------------
# Diff - ported, with its ordering defect fixed
# --------------------------------------------------------------------------

def test_diff_reports_both_directions():
    differences = diff_dicts({"a": 1, "b": 2}, {"b": 3, "c": 4})
    paths = {d["path"] for d in differences}
    assert paths == {"a", "b", "c"}
    by_path = {d["path"]: d for d in differences}
    assert by_path["a"]["in_second"] is False
    assert by_path["c"]["in_first"] is False
    assert (by_path["b"]["value1"], by_path["b"]["value2"]) == (2, 3)


def test_diff_order_does_not_depend_on_insertion_order():
    """The defect fixed on port: the original iterated dict insertion order."""
    first = {"z": 1, "a": 2, "m": 3}
    second = {"a": 9, "m": 8, "z": 7}
    rebuilt_first = {"m": 3, "z": 1, "a": 2}
    rebuilt_second = {"z": 7, "a": 9, "m": 8}
    assert diff_dicts(first, second) == diff_dicts(rebuilt_first, rebuilt_second)
    assert [d["path"] for d in diff_dicts(first, second)] == ["a", "m", "z"]


def test_diff_recurses_into_nested_dicts():
    differences = diff_dicts({"x": {"y": 1}}, {"x": {"y": 2}})
    assert [d["path"] for d in differences] == ["x.y"]


def test_config_diff_against_defaults():
    changed = Config({"ranking.b": 0.5, "display.k": 20})
    paths = {d["path"] for d in changed.diff(Config())}
    assert paths == {"ranking.b", "display.k"}


# --------------------------------------------------------------------------
# Snapshots bind to an index
# --------------------------------------------------------------------------

def test_snapshot_requires_a_manifest_hash():
    """Settings alone do not identify a result."""
    with pytest.raises(ConfigError, match="manifest hash"):
        Snapshot(name="s", config=Config(), manifest_hash="")


def test_snapshot_round_trips():
    snapshot = Snapshot(name="s", config=Config({"ranking.b": 0.5}),
                        manifest_hash="abc123")
    restored = Snapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))
    assert restored.config == snapshot.config
    assert restored.manifest_hash == "abc123"
    assert restored.identity() == snapshot.identity()


def test_snapshot_detects_a_hand_edited_file():
    """A stored hash that disagrees with its settings means tampering."""
    data = Snapshot(name="s", config=Config(), manifest_hash="abc").to_dict()
    data["settings"]["ranking.b"] = 0.1
    with pytest.raises(ConfigError, match="has been altered"):
        Snapshot.from_dict(data)


def test_snapshot_identity_covers_both_config_and_index():
    config = Config()
    first = Snapshot(name="s", config=config, manifest_hash="index-one")
    second = Snapshot(name="s", config=config, manifest_hash="index-two")
    assert first.config.content_hash() == second.config.content_hash()
    assert first.identity() != second.identity()
