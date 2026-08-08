"""M1.7 checkpoint: documentation that cannot drift from the system.

The failure this guards against is the one the source project actually
suffered: 147 claimed dimensions against 118 real, 36 documented flags that did
not exist alongside 35 real ones omitted, and `80% vs 60% accuracy` printed in
prose with no evaluation anywhere in the repository. Every one of those was
hand-written into a document that nothing checked.

**The audit, run before these tests were written.** "A hand edit fails a test"
is vacuous if the test regenerates the files before comparing - it would then
compare a fresh render against itself and pass forever. So
`test_committed_docs_match_a_fresh_render` reads what is on disk and never
writes, and the registered falsifier changes what the renderer produces so the
two must diverge.

The rendered files are committed rather than ignored. An ignored file cannot be
checked, which would leave the guarantee unenforceable.
"""

import os
import re
import sys
from pathlib import Path
from string import Template

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drf.contract import reset_replay_log  # noqa: E402
from drf.docs.render import (  # noqa: E402
    AUDIENCES,
    BANNER,
    OUTPUT_DIR,
    TEMPLATE_DIR,
    build_context,
    render_all,
    render_document,
)
from drf.ingest.build import build_index  # noqa: E402

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
def context(tmp_path_factory):
    """A render context from a freshly built index.

    The build is reproducible, so this index carries the same content hash as
    the one the committed documents were rendered from. If it did not, that
    would itself be a reproducibility failure and M1.1 would have caught it.
    """
    reset_replay_log()
    out = tmp_path_factory.mktemp("m17") / "index.db"
    build_index(source_path=SOURCE, out_path=str(out))
    return build_context(str(out))


# --------------------------------------------------------------------------
# The hand-edit guarantee
# --------------------------------------------------------------------------

@requires_source
def test_committed_docs_match_a_fresh_render(context):
    """The checkpoint. Reads from disk; never writes.

    Falsifiable - spec/invariants.json::docs_are_generated. If this test ever
    regenerated the files first, it would compare a fresh render against
    itself and could never fail.
    """
    from drf.docs.render import OUTPUT_PATHS

    rendered = render_all(context)
    for audience, expected in rendered.items():
        relative = OUTPUT_PATHS.get(audience)
        path = (ROOT / relative) if relative else OUTPUT_DIR / f"{audience}.md"
        assert path.exists(), (
            f"{path} is missing; run `drf docs build --index index.db`"
        )
        on_disk = path.read_text()
        assert on_disk == expected, (
            f"docs/{audience}.md differs from a fresh render. Either it was "
            "edited by hand, or spec/ changed and the docs were not "
            "regenerated. Run `drf docs build`."
        )


def test_every_audience_has_a_template_and_a_rendered_file():
    """Five audiences. README is generated too - the landing page is the most
    likely place for a stale number to sit unchallenged."""
    from drf.docs.render import OUTPUT_PATHS

    assert len(AUDIENCES) == 5
    for audience in AUDIENCES:
        assert (TEMPLATE_DIR / f"{audience}.md.tmpl").exists()
        relative = OUTPUT_PATHS.get(audience)
        path = (ROOT / relative) if relative else OUTPUT_DIR / f"{audience}.md"
        assert path.exists(), f"{path} not rendered"


def test_rendered_files_carry_the_do_not_edit_banner():
    from drf.docs.render import OUTPUT_PATHS

    for audience in AUDIENCES:
        relative = OUTPUT_PATHS.get(audience)
        path = (ROOT / relative) if relative else OUTPUT_DIR / f"{audience}.md"
        text = path.read_text()
        assert text.startswith(BANNER.split("\n")[0])
        assert "DO NOT EDIT" in text


# --------------------------------------------------------------------------
# Unresolved placeholders are a hard error
# --------------------------------------------------------------------------

def test_a_missing_placeholder_raises_rather_than_rendering():
    """`substitute`, never `safe_substitute`.

    Falsifiable - spec/invariants.json::docs_fail_on_missing_placeholder.
    `safe_substitute` would leave `$whatever` sitting in the prose looking
    like it belonged there - a document with a hole in it that reads as
    finished.
    """
    with pytest.raises(KeyError):
        Template("value is $definitely_not_in_context").substitute({"a": 1})

    with pytest.raises(KeyError):
        render_document("peer", {"banner": "x"})


@requires_source
def test_no_unresolved_placeholders_survive_into_the_output(context):
    """Belt and braces: scan the rendered text for template syntax."""
    for audience, text in render_all(context).items():
        leftovers = re.findall(r"\$\{?[a-z_][a-z0-9_]*\}?", text)
        assert leftovers == [], (
            f"{audience}.md contains unresolved placeholders: {leftovers}"
        )


def test_unknown_audience_is_rejected():
    with pytest.raises(ValueError, match="unknown audience"):
        render_document("marketing", {})


# --------------------------------------------------------------------------
# The numbers came from somewhere
# --------------------------------------------------------------------------

@requires_source
def test_key_figures_appear_and_come_from_the_index(context):
    """Figures in the docs must equal what the index actually contains."""
    rendered = render_all(context)
    for audience in ("peer", "agent", "operator"):
        text = rendered[audience]
        assert str(context["content_hash"]) in text
        assert str(context["node_count"]) in text


@requires_source
def test_every_benchmark_figure_has_a_recorded_producer(context):
    """`spec/benchmarks.json` stores each measurement beside its command."""
    from drf.docs.render import load_spec

    benchmarks = load_spec("benchmarks")
    for section in ("reproducibility", "chaos_control", "sensitivity",
                    "length_normalisation", "graph"):
        assert benchmarks[section].get("producer"), (
            f"{section} records a measurement with no producer command"
        )


@requires_source
def test_peer_and_plain_docs_state_the_scope_limit(context):
    """No claim about retrieval quality, stated where it cannot be missed.

    M1 carries no relevance labels. An unqualified "all metrics 1.0" would be
    exactly the drift this framework exists to prevent, so the limit is
    rendered into the documents rather than left to a reader's inference.
    """
    rendered = render_all(context)
    assert "no relevance" in rendered["peer"].lower()
    assert "NO claim about retrieval quality" in rendered["peer"]

    # Markdown emphasis can split a phrase ("does **not** claim"), so match on
    # words rather than on a contiguous sentence - the first version of this
    # assertion failed for that reason and would have forced the prose to be
    # written around the test.
    plain = rendered["plain"].lower()
    assert "better answers" in plain
    assert "correct answers to compare against" in plain
    assert "80% versus 60%" in plain, (
        "the plain document should name the concrete drift this project was "
        "recovered from, not just gesture at honesty"
    )


@requires_source
def test_plain_document_avoids_jargon(context):
    """The fourth audience is the one that gets skipped. Check it stayed plain."""
    plain = render_all(context)["plain"]
    for term in ("BM25", "idf", "quantis", "injective", "postings",
                 "Advisory[T]", "sha256"):
        assert term not in plain, (
            f"plain-English document contains {term!r}; it is written for a "
            "reader who does not have the vocabulary"
        )


@requires_source
def test_agent_document_records_the_traps_already_hit(context):
    """The agent doc exists to stop a future session repeating known mistakes."""
    agent = render_all(context)["agent"]
    for marker in ("falsifier", "control", "RRF", "exact integers"):
        assert marker in agent
