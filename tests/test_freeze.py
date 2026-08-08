"""M1.8 checkpoint: a release names results, not just a commit.

A git tag pins the code. It does not pin the answers, because an answer depends
on three things:

    spec_sha        every spec/*.json hashed together
    manifest_hash   the index the results came from
    bench_digest    the actual answers to the fixed query set

`spec/frozen.json` records all three, and these tests rebuild from source and
check each one still matches.

The **bench digest is the load-bearing one**. Spec and index hashes prove the
*inputs* are unchanged; only the digest proves the *outputs* are. A refactor
that alters ranking while leaving both inputs untouched is exactly what a
release check made of input hashes alone would wave through - so
`test_bench_digest_is_the_component_that_catches_a_ranking_change` constructs
that case and requires it to be caught.
"""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drf import freeze  # noqa: E402
from drf.contract import reset_replay_log  # noqa: E402
from drf.ingest.build import build_index  # noqa: E402
from drf.version import RELEASE_VERSION  # noqa: E402

# Override with DRF_SOURCE_DB. Hardcoding an absolute path would publish a
# username and make every test skip for anyone else who clones this.
SOURCE = os.environ.get(
    "DRF_SOURCE_DB",
    str(Path.home() / "Downloads/claude-cookbook-kg3/claude-cookbook-kg.db"),
)

requires_source = pytest.mark.skipif(
    not os.path.exists(SOURCE), reason=f"source corpus not present at {SOURCE}"
)


@pytest.fixture(scope="module")
def rebuilt(tmp_path_factory):
    """A fresh index from source - not the one the freeze was written from."""
    reset_replay_log()
    out = tmp_path_factory.mktemp("m18") / "index.db"
    build_index(source_path=SOURCE, out_path=str(out))
    return str(out)


@requires_source
def test_freeze_matches_a_live_rebuild(rebuilt):
    """The checkpoint: everything recorded still holds after rebuilding."""
    ok, differences = freeze.verify(rebuilt)
    assert ok, "freeze does not match a live rebuild:\n" + "\n".join(differences)


@requires_source
def test_all_three_hashes_are_recorded_and_non_empty():
    recorded = freeze.read()
    for key in ("spec_sha", "manifest_hash", "bench_digest"):
        assert recorded.get(key), f"freeze is missing {key}"
        assert len(recorded[key]) == 64, f"{key} is not a sha256 hex digest"


def test_release_in_freeze_matches_the_code():
    """A freeze written under an older version must not survive a bump."""
    assert freeze.read()["release"] == RELEASE_VERSION, (
        "spec/frozen.json records a different release than drf/version.py; "
        "run `drf freeze write --index index.db` after bumping"
    )


def test_freeze_does_not_hash_itself():
    """`frozen.json` lives in spec/ but cannot be part of spec_sha.

    Including it would make the hash depend on its own value - a fixed point
    that could never be computed, and a mistake that is easy to make because
    the file sits with the others.
    """
    recorded = freeze.read()
    assert freeze.spec_sha() == recorded["spec_sha"]
    assert (ROOT / "spec" / "frozen.json").exists()


@requires_source
def test_bench_digest_is_the_component_that_catches_a_ranking_change(rebuilt):
    """Input hashes alone would miss a ranking regression.

    Constructed directly: change how results are produced without touching the
    spec or the index, and confirm that spec_sha and manifest_hash both still
    match while bench_digest does not. This is why the digest is recorded.
    """
    from drf.bench import repro
    from drf.store import connect, read_manifest

    conn = connect(rebuilt)
    manifest_hash = read_manifest(conn)["content_hash"]
    queries = repro.load_queries()
    honest = repro.run_all(conn, manifest_hash, queries)
    # A ranking change with identical inputs: reverse each result list.
    altered = {qid: list(reversed(ids)) for qid, ids in honest.items()}
    conn.close()

    recorded = freeze.read()
    assert freeze.spec_sha() == recorded["spec_sha"]         # input unchanged
    assert manifest_hash == recorded["manifest_hash"]        # input unchanged
    assert repro.digest(honest) == recorded["bench_digest"]
    assert repro.digest(altered) != recorded["bench_digest"], (
        "the bench digest did not change when the results did; it would not "
        "catch a ranking regression"
    )


@requires_source
def test_verify_reports_every_difference_not_just_the_first(rebuilt, monkeypatch):
    """A release check that stops at the first mismatch hides the rest."""
    monkeypatch.setattr(
        freeze, "read",
        lambda: {"release": "x", "spec_sha": "a", "manifest_hash": "b",
                 "bench_digest": "c", "versions": {}},
    )
    ok, differences = freeze.verify(rebuilt)
    assert not ok
    assert len(differences) == 4, (
        f"expected all four components reported, got {differences}"
    )


def test_frozen_file_is_valid_json_and_committed():
    path = ROOT / "spec" / "frozen.json"
    assert path.exists(), "spec/frozen.json must be committed - it is the release record"
    json.loads(path.read_text())
