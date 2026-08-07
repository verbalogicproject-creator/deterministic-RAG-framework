"""Reproducibility metrics, after ReproRAG (arXiv 2509.18869).

Each metric is reported twice: as an **exact integer** for assertions and as a
float for humans. Assertions use the integers. `kendall_tau == 1.000` is a
statement about rounding; `discordant_pairs == 0` is a statement about the
thing itself, and it cannot be satisfied by a value that merely rounds well.

**A determinism suite that only ever reports 1.0 proves nothing.** On a fully
deterministic pipeline every metric here is perfect by construction, so the
numbers alone are not evidence - they are compatible with a harness that
computes nothing. `drf/bench/repro.py` therefore runs a **chaos control**: the
same metrics against a deliberately non-deterministic pipeline, which must
score strictly worse. Without that, "all metrics 1.0" is a claim about the
harness, not about the system.

Stdlib only.
"""

import math
from typing import NamedTuple, Sequence


class Comparison(NamedTuple):
    """Two result lists compared. Integers first - those are what get asserted."""

    # Exact integers: the assertion surface.
    mismatched_positions: int   # positions where the two lists differ
    discordant_pairs: int       # pairs ordered oppositely in the two lists
    length_delta: int           # |len(a) - len(b)|
    symmetric_difference: int   # items in exactly one list

    # Floats: for reporting to people, never for assertions.
    exact_match: float
    jaccard: float
    overlap_coefficient: float
    kendall_tau: float
    rbo: float

    @property
    def identical(self) -> bool:
        """The only judgement that matters. Every integer must be zero."""
        return (
            self.mismatched_positions == 0
            and self.discordant_pairs == 0
            and self.length_delta == 0
            and self.symmetric_difference == 0
        )


def _kendall_tau(a: Sequence[str], b: Sequence[str]) -> tuple[int, float]:
    """Discordant pair count, and tau over the items both lists contain.

    Restricted to the shared items because rank correlation is undefined for
    an item that appears in only one list. The items dropped are already
    counted by `symmetric_difference`, so nothing goes unreported - it is
    reported by the metric that can express it.
    """
    shared = [x for x in a if x in set(b)]
    if len(shared) < 2:
        return 0, 1.0
    rank_b = {x: i for i, x in enumerate(b)}
    discordant = 0
    total = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            total += 1
            if rank_b[shared[i]] > rank_b[shared[j]]:
                discordant += 1
    concordant = total - discordant
    tau = (concordant - discordant) / total if total else 1.0
    return discordant, tau


def _rbo(a: Sequence[str], b: Sequence[str], p: float = 0.9) -> float:
    """Rank-Biased Overlap: top-weighted, so early disagreement costs more.

    The right shape for retrieval, where position 1 matters far more than
    position 40 - unlike Jaccard, which treats every position alike.

    **Normalised by the maximum achievable value at this depth.** RBO is
    defined over infinite rankings; truncated at depth k the raw sum can only
    reach `1 - p**k`, so two *identical* short lists score far below 1. The
    first version of this function omitted the normalisation and reported
    0.716 for byte-identical results - against 0.712 for the deliberately
    broken control. A metric that cannot separate perfect agreement from a
    pipeline with 3,931 discordant pairs is not a weak metric, it is a
    misleading one, and it would have been quoted as evidence.

    Dividing by `1 - p**k` makes identical lists score exactly 1.0 and keeps
    the top-weighting that makes RBO worth having.
    """
    if not a and not b:
        return 1.0
    depth = max(len(a), len(b))
    if depth == 0:
        return 1.0
    seen_a: set[str] = set()
    seen_b: set[str] = set()
    total = 0.0
    for d in range(1, depth + 1):
        if d <= len(a):
            seen_a.add(a[d - 1])
        if d <= len(b):
            seen_b.add(b[d - 1])
        total += (len(seen_a & seen_b) / d) * (p ** (d - 1))
    raw = (1 - p) * total
    maximum = 1 - p ** depth
    if maximum <= 0:
        return 1.0
    # Clamped because RBO is bounded by 1 by definition; the division can
    # otherwise return 1.0000000000000002 for identical lists. This is exactly
    # why the assertion surface is integers - a float metric needs a caveat
    # like this one, and a caveat is a place for a bug to live.
    return min(1.0, raw / maximum)


def compare(a: Sequence[str], b: Sequence[str]) -> Comparison:
    """Compare two ordered result lists."""
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    intersection = set_a & set_b

    mismatched = sum(
        1 for i in range(max(len(a), len(b)))
        if (a[i] if i < len(a) else None) != (b[i] if i < len(b) else None)
    )
    discordant, tau = _kendall_tau(a, b)

    return Comparison(
        mismatched_positions=mismatched,
        discordant_pairs=discordant,
        length_delta=abs(len(a) - len(b)),
        symmetric_difference=len(union - intersection),
        exact_match=1.0 if list(a) == list(b) else 0.0,
        jaccard=len(intersection) / len(union) if union else 1.0,
        overlap_coefficient=(
            len(intersection) / min(len(set_a), len(set_b))
            if set_a and set_b else 1.0
        ),
        kendall_tau=tau,
        rbo=_rbo(a, b),
    )


def score_stability(a: Sequence[int], b: Sequence[int]) -> int:
    """Maximum absolute difference between two score vectors.

    Returns an `int` because scores *are* ints - fixed-point, quantised at
    build time. A float score stability of 1e-9 would be indistinguishable
    from zero at a glance; an integer 0 is unambiguous, and any non-zero value
    is a real difference rather than a rounding artefact.
    """
    if len(a) != len(b):
        return max(max(a, default=0), max(b, default=0))
    return max((abs(x - y) for x, y in zip(a, b)), default=0)


def aggregate(comparisons: list[Comparison]) -> dict:
    """Summarise many comparisons. Integers summed, floats averaged."""
    if not comparisons:
        return {"comparisons": 0}
    count = len(comparisons)
    return {
        "comparisons": count,
        # Assertion surface.
        "mismatched_positions": sum(c.mismatched_positions for c in comparisons),
        "discordant_pairs": sum(c.discordant_pairs for c in comparisons),
        "length_delta": sum(c.length_delta for c in comparisons),
        "symmetric_difference": sum(c.symmetric_difference for c in comparisons),
        "identical": sum(1 for c in comparisons if c.identical),
        # Reporting only.
        "exact_match_rate": math.fsum(c.exact_match for c in comparisons) / count,
        "jaccard": math.fsum(c.jaccard for c in comparisons) / count,
        "overlap_coefficient": math.fsum(
            c.overlap_coefficient for c in comparisons
        ) / count,
        "kendall_tau": math.fsum(c.kendall_tau for c in comparisons) / count,
        "rbo": math.fsum(c.rbo for c in comparisons) / count,
    }
