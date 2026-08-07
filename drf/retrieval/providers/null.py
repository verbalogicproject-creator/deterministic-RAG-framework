"""The provider that proposes nothing.

Not a placeholder. It is the reference implementation of the guarantee: with
`NullProvider` installed, the merged output must equal D exactly, and every
other provider's output must share that same prefix. It is what
`--neural off` uses, and what the byte-identical-prefix comparison in
`tests/test_merge.py` compares against.

Stdlib only.
"""


class NullProvider:
    """Proposes nothing, always."""

    name = "null"

    def propose(self, *, anchors: list[str], limit: int) -> list[str]:
        return []
