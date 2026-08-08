"""Control rankers. Without these, a quality number is not evidence.

Milestone 1 established this the hard way. Every reproducibility metric scored
1.0, which was compatible with a harness that computed nothing; only the chaos
control - the prior engine's defect, reproduced exactly - proved the harness
could report failure at all. Quality measurement has the same hole in a worse
form, because a quality figure has no natural perfect score to look suspicious
against. `nDCG@10 = 0.71` looks like a result. It is not one. It is a result
only against a stated alternative.

So every control here answers a specific question that a bare number cannot:

| control | the question it answers |
|---|---|
| `oracle` | What is the ceiling *given these candidates*? Separates a ranking failure from a retrieval failure. |
| `id_order` | Does relevance ranking beat sorting by hash? **A deterministic, reproducible, entirely meaningless ranking.** |
| `shuffle` | Does the ordering carry information at all? |
| `reverse` | Does the metric notice ordering? A set metric scores this identically to the real ranking. |

`id_order` matters most to *this* project specifically. Node ids are sha256
digests, so sorting by id is arbitrary with respect to relevance while being
perfectly deterministic and perfectly reproducible - it would score 1.0 on
every metric in `bench/metrics.py`. It is the counter-example to the reading
this framework most invites: that determinism is a quality property. It is
not, and the control makes the distinction measurable rather than asserted.

`reverse` is the sharpest diagnostic of the harness itself. It holds the
retrieved set exactly constant and inverts only the order, so recall@k and
Jaccard cannot distinguish it from the real ranking by construction. Any
metric that scores `reverse` equal to the system is order-blind, and this is
how M2 detects that mechanically instead of by remembering it.

Stdlib only.
"""

import random
from typing import Callable, Mapping, Sequence

# Seeds are fixed rather than drawn, so a control result is reproducible and a
# surprising figure can be re-run rather than argued about. Several, because a
# single shuffle can be lucky - on a 10-candidate list with 3 relevant
# documents, one permutation in a few dozen beats a real ranker.
SHUFFLE_SEEDS = (0, 1, 2, 3, 4)


def oracle(ranking: Sequence[str], labels: Mapping[str, int]) -> list[str]:
    """The best ordering achievable *from the candidates actually retrieved*.

    The ceiling, and a diagnostic in its own right: if the oracle's nDCG is
    itself low, the ranking is not the problem - the candidate set never
    contained the relevant documents, and no amount of reordering will help.
    That is precisely the failure the advisory layer exists to address, so
    distinguishing the two cases is not a nicety.

    Ties broken by node id for the same reason as everywhere else: grade alone
    is not a total order, and a ceiling that moved with dictionary iteration
    order would be a poor yardstick.
    """
    return sorted(ranking, key=lambda node_id: (-labels.get(node_id, 0), node_id))


def id_order(ranking: Sequence[str], labels: Mapping[str, int]) -> list[str]:
    """Sort by node id: fully deterministic, entirely relevance-blind.

    `labels` is accepted and ignored so every control shares one signature.
    The system must beat this by a stated margin. If it does not, the ranking
    logic is contributing nothing that a sha256 digest does not.
    """
    return sorted(ranking)


def reverse(ranking: Sequence[str], labels: Mapping[str, int]) -> list[str]:
    """Exact reversal. Same set, worst plausible order.

    Its recall@k for k >= len(ranking) is identical to the system's by
    construction - that identity is the point, and a test asserts it.
    """
    return list(reversed(ranking))


def shuffle(ranking: Sequence[str], labels: Mapping[str, int], *, seed: int) -> list[str]:
    """A seeded permutation. Random in content, reproducible in behaviour."""
    out = list(ranking)
    random.Random(seed).shuffle(out)
    return out


#: Controls with the uniform `(ranking, labels) -> ranking` signature. The
#: shuffle family is handled separately because it needs a seed.
CONTROLS: dict[str, Callable[[Sequence[str], Mapping[str, int]], list[str]]] = {
    "oracle": oracle,
    "id_order": id_order,
    "reverse": reverse,
}


def shuffle_controls(seeds: Sequence[int] = SHUFFLE_SEEDS) -> dict[str, Callable]:
    """One control per seed, so a lucky permutation is visible as an outlier."""
    return {
        f"shuffle_{seed}": (lambda r, l, s=seed: shuffle(r, l, seed=s))
        for seed in seeds
    }


def all_controls(seeds: Sequence[int] = SHUFFLE_SEEDS) -> dict[str, Callable]:
    return {**CONTROLS, **shuffle_controls(seeds)}
