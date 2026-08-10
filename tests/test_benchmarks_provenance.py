"""Every recorded figure must stay a *measurement*, not become a transcription.

Design rule 1 of this project is "no number without a producer", and every
section of `spec/benchmarks.json` records the command that produced it. That
rule turns out to be weaker than it reads, and this module exists because of
how it failed.

**What happened on 2026-08-10.** `graph_candidate_admission.positive_control`
recorded `drf_tail_rho: -0.0288`. The producer emits `+0.0288`. The sign had
been inverted in transcription. The full suite was green, the figure had a
producer recorded beside it, and the error had shipped through a structural
audit - because *nothing ever ran the producer*. The `mud_detection` checkout
the producer needs had been lost from `/tmp`, so the figure had silently
stopped being a measurement some time earlier. No event marked the transition.

**Why the existing guard did not catch it.**
`test_recorded_quality_figures_match_a_live_run` does exactly the right thing,
and its docstring names exactly the right risk - "the values were copied into
the spec by a human hand, which is exactly the step at which the recovered
project's numbers went wrong". It was applied to one section. Nine others had
no equivalent, and nothing anywhere said so.

That is the third instance of one defect in this project:

    M1.6 registry test              invariants listed literally in the test
    test_verify_reports_every_...   `== 4`, stale when labels_hash joined
    here                            *which* sections get a live-run check

In each case a collection was maintained by hand, drifted, and the suite
stayed green while it drifted. The first two were fixed by deriving the
collection - `freeze.VERIFIED_KEYS`, `config_schema.sensitivity_probe`. This
module applies the same fix to the third: the set of sections is *derived from
the spec*, so a new section cannot be silently unverified. It must declare
`verified_by` or state a `verification_exempt` reason, and the exemption is
visible in the file rather than absent from it.

The exemption pattern is borrowed from the falsifier registry, which already
carries deliberately-exempt checkpoints each with a recorded reason. An
exemption that must be written down is not the same as a gap nobody noticed.
"""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drf.bench.repro import load_queries  # noqa: E402

SOURCE = os.environ.get(
    "DRF_SOURCE_DB",
    str(Path.home() / "Downloads/claude-cookbook-kg3/claude-cookbook-kg.db"),
)
requires_source = pytest.mark.skipif(
    not os.path.exists(SOURCE), reason=f"source corpus not present at {SOURCE}"
)

BENCHMARKS = json.loads((ROOT / "spec" / "benchmarks.json").read_text())


def measured_sections() -> dict:
    """Sections carrying a `producer` - i.e. claiming to report a measurement.

    Derived, never listed. Adding a section to the spec adds it here, which is
    the entire point: the failure mode being closed is a section that exists
    and is checked by nothing.
    """
    return {
        name: block
        for name, block in BENCHMARKS.items()
        if isinstance(block, dict) and "producer" in block
    }


def test_every_measured_section_declares_how_it_is_verified():
    """A figure with a producer must say whether anything re-runs it.

    This does not require that every section be live-checked - some producers
    take minutes and re-running them in the unit suite would be its own kind
    of dishonesty, trading a real guarantee for a slow one. It requires that
    the choice be *stated*. An undeclared section is the only failure here.
    """
    undeclared = sorted(
        name for name, block in measured_sections().items()
        if "verified_by" not in block and "verification_exempt" not in block
    )
    assert not undeclared, (
        f"{undeclared} record measured figures but declare no verification. "
        "Add `verified_by` naming a test that re-runs the producer, or "
        "`verification_exempt` stating why re-running it in the suite is not "
        "worth what it costs. A section checked by nothing, saying nothing, "
        "is how drf_tail_rho shipped with an inverted sign."
    )


def test_no_exemption_is_left_blank():
    """An exemption has to carry a reason, or it is just a silencer."""
    for name, block in measured_sections().items():
        reason = block.get("verification_exempt")
        if reason is None:
            continue
        assert isinstance(reason, str) and len(reason) > 40, (
            f"{name}: verification_exempt must explain itself, not merely "
            f"exist. Got {reason!r}"
        )


def test_named_verifiers_exist():
    """`verified_by` must name a test that is actually present.

    Otherwise the declaration is worth less than the gap it replaced - it
    reads as coverage while pointing at nothing.
    """
    for name, block in measured_sections().items():
        ref = block.get("verified_by")
        if ref is None:
            continue
        path, _, test_name = ref.partition("::")
        source = ROOT / path
        assert source.exists(), f"{name}: verified_by names a missing file {path}"
        assert f"def {test_name}(" in source.read_text(), (
            f"{name}: verified_by names {test_name}, which is not defined in "
            f"{path}"
        )


@requires_source
def test_recorded_ablation_figures_match_a_live_run(tmp_path_factory):
    """The load-bearing half of the graph question, re-measured.

    `graph_contribution.ablation` is the evidence that actually settled whether
    the graph layer earns its place - layer 2's REDUNDANT verdict is partly
    structural, as the spec itself records. It is also the one graph figure
    whose producer needs no optional dependency, so unlike the positive
    control it can be re-run here unconditionally.

    That asymmetry is worth naming rather than smoothing over: the check that
    can always run covers the evidence, and the check that cannot run covers
    the control that makes the evidence trustworthy.
    """
    import importlib.util

    from drf.contract import reset_replay_log
    from drf.ingest.build import build_index
    from drf.store import connect, read_manifest

    block = BENCHMARKS["graph_contribution"]["ablation"]

    reset_replay_log()
    out = tmp_path_factory.mktemp("prov") / "index.db"
    build_index(source_path=SOURCE, out_path=str(out))

    spec = importlib.util.spec_from_file_location(
        "mgc", ROOT / "tools" / "measure_graph_contribution.py"
    )
    mgc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mgc)

    conn = connect(str(out))
    read_manifest(conn)
    live = mgc.ablate(conn, load_queries())
    conn.close()

    assert live["queries"] == block["queries"]
    assert live["queries_changed"] == block["queries_whose_output_changes"]
    assert live["shallowest_change"] == block["shallowest_rank_ever_affected"]
    assert live["item_set_ever_changes"] == block["candidate_set_ever_changes"]


@requires_source
def test_recorded_horizon_figures_match_a_live_run(tmp_path_factory):
    """The advisory horizon, re-measured.

    These are the figures the whole M2 evaluation is conditioned on - `|D|`
    bounds where the neural layer can act at all, and every quality number is
    read against it. They are also label-free and cheap, so there is no excuse
    for them to be transcription rather than measurement.

    Compared as integers, which is the project's standing rule: a determinism
    figure asserted as a float means nothing.
    """
    from drf.bench import evaluate
    from drf.contract import reset_replay_log
    from drf.ingest.build import build_index

    block = BENCHMARKS["advisory_horizon"]

    reset_replay_log()
    out = tmp_path_factory.mktemp("horizon") / "index.db"
    build_index(source_path=SOURCE, out_path=str(out))

    live = evaluate.run_advisory_invariance(str(out), load_queries())
    horizons = [row["horizon"] for row in live["rows"]]

    assert live["queries"] == block["queries"]
    assert live["prefixes_identical"] == block["prefixes_identical"]
    assert live["prefixes_differing"] == block["prefixes_differing"]
    assert min(horizons) == block["horizon_min"]
    assert max(horizons) == block["horizon_max"]
    assert (
        live["queries"] - live["queries_where_advisory_can_reach_any_depth"]
        == block["queries_with_no_reachable_evaluated_depth"]
    )
