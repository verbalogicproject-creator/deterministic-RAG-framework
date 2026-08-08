"""Relevance judgements: loading, validation, and binding them to a result.

A quality figure is a statement about *a specific set of labels*. Edit one
grade and every nDCG in the project changes, silently, with no commit touching
any code. That is the same class of failure this framework was recovered from -
published metrics whose producer could not be found - so labels are hashed and
the hash travels with any result computed from them.

**Format.** One judgement per line, flat rather than nested:

    {"query_id": "q01", "node_id": "<sha256>", "grade": 2, "note": "..."}

Flat because labels are written by hand over many sittings. A nested
`{"query_id": ..., "labels": {...}}` form makes every addition a modification
of an existing line, and hand-editing a growing JSON object is how a judgement
gets silently overwritten. One line per judgement means additions are appends
and a diff shows exactly what a labelling session decided.

**Duplicates are resolved by the rule used everywhere else in this project:
identical is fine, divergent is an error.** The same conflict-free union that
governs edge collapse in ingest. A repeated judgement at the same grade is
just a duplicate line; a repeated judgement at a *different* grade means two
sittings disagreed, and averaging them or taking the last would destroy the
only evidence that the query is ambiguous.

Stdlib only.
"""

import json
from pathlib import Path
from typing import Iterable, NamedTuple

from ..hashing import sha256_value
from .quality import GRADES


class Label(NamedTuple):
    query_id: str
    node_id: str
    grade: int
    note: str = ""


class LabelSet(NamedTuple):
    """Judgements grouped by query, plus the hash that identifies them."""

    by_query: dict[str, dict[str, int]]
    labels_hash: str
    count: int

    def for_query(self, query_id: str) -> dict[str, int]:
        return self.by_query.get(query_id, {})

    @property
    def judged_queries(self) -> list[str]:
        return sorted(self.by_query)


class LabelError(ValueError):
    """A malformed, conflicting, or unresolvable judgement.

    Raised rather than warned. A skipped label is a silently smaller
    denominator, which moves every recall figure downward without appearing
    anywhere in the output.
    """


def parse(lines: Iterable[str]) -> list[Label]:
    """Parse JSONL, reporting the line number of whatever is wrong with it."""
    labels: list[Label] = []
    for number, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LabelError(f"line {number}: not valid JSON: {exc}") from exc
        missing = {"query_id", "node_id", "grade"} - set(record)
        if missing:
            raise LabelError(f"line {number}: missing {sorted(missing)}")
        grade = record["grade"]
        if not isinstance(grade, int) or isinstance(grade, bool):
            raise LabelError(f"line {number}: grade must be an int, got {grade!r}")
        if grade not in GRADES:
            raise LabelError(
                f"line {number}: grade {grade} is not one of {sorted(GRADES)}"
            )
        labels.append(Label(
            query_id=str(record["query_id"]),
            node_id=str(record["node_id"]),
            grade=grade,
            note=str(record.get("note", "")),
        ))
    return labels


def collate(
    labels: list[Label],
    *,
    known_query_ids: set[str] | None = None,
    known_node_ids: set[str] | None = None,
) -> LabelSet:
    """Group by query, rejecting conflicts and unresolvable references.

    An unknown `node_id` is an error rather than a dropped row. Node ids are
    sha256 digests transcribed by hand or by script; a typo produces an id
    that matches nothing, and silently ignoring it removes a relevant document
    from the denominator - which *raises* the measured recall. A validation
    step whose failure mode flatters the results is worse than none.
    """
    grouped: dict[str, dict[str, int]] = {}
    for label in labels:
        if known_query_ids is not None and label.query_id not in known_query_ids:
            raise LabelError(
                f"judgement references unknown query {label.query_id!r}"
            )
        if known_node_ids is not None and label.node_id not in known_node_ids:
            raise LabelError(
                f"judgement for {label.query_id} references node "
                f"{label.node_id[:16]}... which is not in the index"
            )
        existing = grouped.setdefault(label.query_id, {})
        if label.node_id in existing and existing[label.node_id] != label.grade:
            raise LabelError(
                f"conflicting judgements for {label.query_id} / "
                f"{label.node_id[:16]}...: grade {existing[label.node_id]} "
                f"and grade {label.grade}. Two sittings disagreed; resolve it "
                f"deliberately rather than letting the file order decide."
            )
        existing[label.node_id] = label.grade

    return LabelSet(
        by_query=grouped,
        labels_hash=hash_labels(grouped),
        count=sum(len(v) for v in grouped.values()),
    )


def hash_labels(by_query: dict[str, dict[str, int]]) -> str:
    """Content hash of a judgement set, over canonical JSON.

    Order-independent by construction, because `canonical_json` sorts keys -
    so appending a judgement changes the hash but reordering the file does
    not. The same commutativity rule the ingest layer follows: identity should
    depend on content, never on the order the content arrived in.
    """
    return sha256_value({"labels": by_query})


def load(
    path: str | Path,
    *,
    known_query_ids: set[str] | None = None,
    known_node_ids: set[str] | None = None,
) -> LabelSet:
    with open(path) as handle:
        return collate(
            parse(handle),
            known_query_ids=known_query_ids,
            known_node_ids=known_node_ids,
        )
