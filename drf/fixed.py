"""Exact accumulation and fixed-point quantisation.

Two rules make float behaviour irrelevant to ranking:

1. Sums are computed with `math.fsum` over a deterministically ordered list.
   `fsum` is *contractually* correctly rounded - a documented guarantee of
   the language, so it holds on any conforming implementation and version.

   The deeper reason, and the one that matters architecturally: a correctly
   rounded sum of a multiset is *unique*. There is exactly one representable
   float nearest the true total, so `fsum` returns the same value whatever
   order the addends arrive in. That makes accumulation **commutative**, not
   merely pinned - the caller's ordering becomes a convenience rather than a
   load-bearing invariant, and a future refactor that reorders contributions
   cannot silently change a score. Framework-wide rule, see STATE.md:
   prefer operations that are order-independent by construction over
   operations made deterministic by sorting first.

   Note, measured rather than assumed: on CPython 3.12 `sum()` uses Neumaier
   compensated summation and agreed with `fsum` on all 200,000 random
   12-element sums tested, with neither showing order dependence. So the
   common claim that `sum()` is order-dependent is outdated for this
   interpreter. `fsum` is used anyway because a determinism framework needs
   a *guarantee* that travels - CPython's compensation is an implementation
   detail, not a promise, and does not necessarily hold on PyPy, on older
   CPython, or on future builds. The distinction is between "observed to be
   accurate here" and "specified to be accurate everywhere".

2. Every value that could ever reach a comparison is quantised to `int`
   immediately. Floats are therefore structurally incapable of participating
   in a sort key, which removes an entire class of ranking instability
   rather than merely making it unlikely.

Stdlib only.
"""

import math
from typing import Iterable

# 10**QUANTUM_EXP fixed-point units per 1.0. Mirrored in spec/ranking.json;
# a mismatch between the two is a spec-drift failure, not a runtime concern.
QUANTUM_EXP = 9
QUANTUM = 10 ** QUANTUM_EXP


def exact_sum(values: Iterable[float]) -> float:
    """Correctly-rounded sum over an already-ordered iterable.

    The caller is responsible for the ordering; this function does not sort,
    because the meaningful order is domain-specific (e.g. BM25 contributions
    are summed in sorted-term order).
    """
    return math.fsum(values)


def quantize(x: float) -> int:
    """Convert a float to fixed-point `int` with round-half-up.

    `math.floor(x * QUANTUM + 0.5)` is used rather than `round()` because
    `round()` implements banker's rounding, whose behaviour at exact .5
    depends on the neighbouring digit - deterministic, but surprising, and
    harder to state in a spec.
    """
    return math.floor(x * QUANTUM + 0.5)


def unquantize(q: int) -> float:
    """Fixed-point back to float. For display only - never for comparison."""
    return q / QUANTUM


def qmul(a_q: int, b_q: int) -> int:
    """Multiply two fixed-point values, returning fixed-point.

    Integer arithmetic throughout; the floor division truncates toward
    negative infinity, which is deterministic for the non-negative values
    used in ranking.
    """
    return (a_q * b_q) // QUANTUM
