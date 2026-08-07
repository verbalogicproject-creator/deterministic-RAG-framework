"""What a neural provider is, and what it is not permitted to be.

A provider proposes node ids. That is the whole interface. It cannot score,
cannot reorder, cannot see the authoritative order, and cannot be consulted
about anything already in it. Everything it returns is boxed as `Advisory[T]`
before it leaves `neural.py`, and only `drf.retrieval.merge` can open the box.

The narrowness is the point. A provider that returned *scores* would invite a
caller to combine them with BM25, and that combination is exactly what the
architecture forbids. By returning bare ids, the type system leaves no
arithmetic for a future edit to reach for.

**Providers may fail in any way at all.** They may raise, hang, return
nonsense, return ten thousand duplicates, or return ids that do not exist.
None of that can affect the authoritative prefix, and
`tests/test_merge.py` proves it against deliberately hostile doubles rather
than asserting it. A provider is untrusted input, not a component.

Stdlib only.
"""

from typing import Protocol


class Provider(Protocol):
    """Proposes node ids that might be worth appending below D."""

    name: str

    def propose(self, *, anchors: list[str], limit: int) -> list[str]:
        """Return candidate node ids given anchor nodes from D.

        `anchors` are ids the authoritative stage already ranked highly.
        Passing ids rather than text is deliberate: it keeps the provider on
        the *formal identifier* side of the determinism boundary described in
        SAT-Graph RAG, where operations over resolved ids are reproducible.

        The return value is advice. It may be empty, and returning nothing is
        always a valid answer.
        """
        ...
