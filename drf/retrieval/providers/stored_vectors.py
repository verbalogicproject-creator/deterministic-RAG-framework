"""Anchor-mode nearest neighbours over frozen vectors.

This provider is the clearest evidence that **determinism and authority are
independent axes**. It is fully deterministic - it runs no model, calls no
network, and reads vectors that were computed once and frozen into the index -
and it is still strictly *advisory*. Nothing about being reproducible earns a
signal the right to influence ranking.

The vectors are 384-dimensional MiniLM, stored as little-endian float32 and
**already L2-normalised** (measured: norm 1.000000), so cosine similarity is a
plain dot product. That removes the normalisation step and with it a division
whose behaviour near zero would need defining.

**No numpy.** 228 vectors x 384 dimensions is 87,552 multiply-adds per anchor;
`array('f')` and `math.fsum` handle it without adding a dependency to the core
path, and `fsum` keeps the accumulation order-independent.

Byte order is handled explicitly rather than assumed. `array.frombytes` uses
*native* order, so on a big-endian machine the same blob would decode to
different numbers - a silent cross-platform divergence in a framework whose
purpose is the opposite.

Stdlib only.
"""

import array
import math
import sys

from ...fixed import quantize
from ...store import iter_embeddings


def decode_vector(blob: bytes, dim: int) -> array.array:
    """Little-endian float32 bytes to a float array, byte order made explicit."""
    if len(blob) != dim * 4:
        raise ValueError(
            f"embedding blob is {len(blob)} bytes, expected {dim * 4} for "
            f"{dim} float32 values"
        )
    values = array.array("f")
    values.frombytes(blob)
    if sys.byteorder == "big":
        values.byteswap()
    return values


def dot(a: array.array, b: array.array) -> float:
    """Dot product, which is cosine here because both vectors are unit length."""
    return math.fsum(x * y for x, y in zip(a, b))


class StoredVectorProvider:
    """Proposes nodes whose frozen vectors are closest to the anchors.

    Similarity is quantised to `int` before it is used for ordering, for the
    same reason scores are: a float comparison is a place where two platforms
    can disagree. Advisory output does not influence ranking, but it should
    still be *reproducible* advice - otherwise `--neural stored` would return
    a different tail on different machines and the trace would be unauditable.
    """

    name = "stored_vectors"

    def __init__(self, conn):
        self._vectors: dict[str, array.array] = {}
        self._dims: set[int] = set()
        for record in iter_embeddings(conn):
            self._vectors[record.node_id] = decode_vector(
                record.vector, record.dim
            )
            self._dims.add(record.dim)
        if len(self._dims) > 1:
            raise ValueError(
                f"index mixes embedding dimensions {sorted(self._dims)}; "
                "vectors from different spaces are not comparable and must "
                "never be scored against each other"
            )

    def propose(self, *, anchors: list[str], limit: int) -> list[str]:
        """Nodes most similar to any anchor, excluding the anchors themselves.

        Scored by the *maximum* similarity to any anchor rather than the mean.
        Mean similarity rewards documents that are mediocre matches for
        everything, which is the opposite of what an anchor set is for.

        Ties are broken by node id, making the ordering injective and so
        reproducible - the same discipline the authoritative stage uses, for
        the same reason.
        """
        anchor_vectors = [
            self._vectors[a] for a in sorted(set(anchors)) if a in self._vectors
        ]
        if not anchor_vectors or limit <= 0:
            return []

        excluded = set(anchors)
        scored: list[tuple[int, str]] = []
        for node_id in sorted(self._vectors):
            if node_id in excluded:
                continue
            best = max(dot(self._vectors[node_id], a) for a in anchor_vectors)
            scored.append((quantize(best), node_id))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [node_id for _score, node_id in scored[:limit]]
