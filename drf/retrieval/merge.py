"""Stage 3: append-only merge. The one module permitted to open the box.

This module's name is hard-coded in `contract.ADVISORY_CONSUMERS`. Every other
module in the framework raises `AuthorityViolation` on `Advisory.unwrap()`, so
advisory data physically cannot reach authoritative arithmetic anywhere else -
not by convention, by exception.

The guarantee, stated exactly:

    merged[:len(D)] == D,  elementwise, in order, always.

Advisory results may only occupy positions `len(D)` and beyond. They cannot
displace, reorder, interleave with, or remove an authoritative result. The
worst a broken, hostile, or brilliant provider can do is contribute nothing -
which is why adding a neural layer here is not a gamble.

**The postcondition runs on every query, in production**, not only under test.
A guarantee checked solely in CI is a guarantee about CI. It is cheap - one
list comparison over the prefix - and it converts the architecture's central
claim from a design intention into a runtime fact.

Stdlib only.
"""

from typing import NamedTuple

from ..contract import Advisory, ContractError

AUTHORITATIVE = "authoritative"
ADVISORY = "advisory"


class SubordinationViolation(ContractError):
    """Advisory data reached or disturbed the authoritative prefix.

    Unreachable by construction. If it is ever raised, the architecture has
    been broken and the result must not be returned - so it raises rather than
    logging, and no caller is offered a way to continue.
    """


class MergedResult(NamedTuple):
    node_id: str
    origin: str      # AUTHORITATIVE | ADVISORY
    rank: int        # 0-based position in the merged output


def merge(
    *,
    deterministic: list[str],
    advisory: Advisory[list[str]],
    known_ids: set[str] | None = None,
) -> list[MergedResult]:
    """Append advisory proposals below the authoritative order.

    `deterministic` is D as a list of node ids, already in strict total order
    from stage 1. `advisory` is the boxed provider output. `known_ids`, when
    given, is the set of ids that exist in the index; proposals outside it are
    dropped, because a provider naming a nonexistent document is proposing
    nothing, and appending an id that resolves to no content would surface as
    an empty result rather than as the provider error it is.

    Three filters are applied to proposals, in order: unknown ids, ids already
    present in D, and duplicates within the proposal list itself. The second
    matters most - without it a provider could "promote" a document by
    proposing something already ranked, producing a duplicate whose second
    appearance looks like a lower-ranked distinct result.
    """
    merged = [
        MergedResult(node_id=node_id, origin=AUTHORITATIVE, rank=position)
        for position, node_id in enumerate(deterministic)
    ]

    # The one sanctioned unwrap in the entire framework.
    proposals = advisory.unwrap() if advisory is not None else []

    already = set(deterministic)
    seen: set[str] = set()
    for node_id in proposals:
        if known_ids is not None and node_id not in known_ids:
            continue
        if node_id in already or node_id in seen:
            continue
        seen.add(node_id)
        merged.append(
            MergedResult(node_id=node_id, origin=ADVISORY, rank=len(merged))
        )

    _assert_subordination(merged, deterministic)
    return merged


def _assert_subordination(
    merged: list[MergedResult], deterministic: list[str]
) -> None:
    """The runtime postcondition. Cheap, and checked on every query.

    Compares the prefix elementwise rather than by length or by set equality.
    A length check would pass if two authoritative results swapped places; a
    set check would pass if the entire prefix were reordered. Only elementwise
    order-sensitive comparison actually says what the guarantee says.
    """
    prefix = [result.node_id for result in merged[:len(deterministic)]]
    if prefix != deterministic:
        raise SubordinationViolation(
            "advisory data disturbed the authoritative prefix: "
            f"expected {deterministic[:5]}..., got {prefix[:5]}..."
        )
    for position, result in enumerate(merged[:len(deterministic)]):
        if result.origin != AUTHORITATIVE:
            raise SubordinationViolation(
                f"position {position} is inside the authoritative prefix but "
                f"is marked {result.origin!r}"
            )


def deterministic_prefix_ids(merged: list[MergedResult]) -> list[str]:
    """The authoritative portion, for prefix-equality comparison across runs.

    Used by the benchmark to assert that `--neural off`, `--neural stored`,
    and every hostile provider double produce a byte-identical prefix.
    """
    return [r.node_id for r in merged if r.origin == AUTHORITATIVE]
