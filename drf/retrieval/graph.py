"""Bounded graph expansion from the lexical seed set.

Breadth-first, bidirectional, depth-bounded. Returns `{node_id: best_depth}` -
the *minimum* number of hops from any seed. Depth is the only structural
signal produced here; nothing in this module computes a score, and nothing
weights one signal against another.

Three properties, each following from a measurement rather than a preference:

**Bidirectional.** Following only outgoing edges leaves 81 of 266 nodes
reaching nothing at depth 2, versus 5 when both directions are followed - and
those 5 are exactly the nodes that have no edges at all. Edge direction here
records how a relation was phrased, not whether two nodes are related.

**A visited set is mandatory, not defensive.** The graph contains directed
cycles (measured). Without it, expansion does not terminate.

**No precomputed path index.** A path index over this graph would cost 2.7 MB
and seconds to build; measured on-the-fly cost is 0.149 ms for 10 seeds at
depth 2, and 5 ms for full depth-4 reachability across all 266 nodes. The
index would be pure overhead at this scale. (It becomes a real question at
M3 scale, and the schema in
`playbooks-v2/KNOWLEDGE-bidirectional-path-indexing.pb` is a reasonable
starting point there - its query results reproduce exactly against
`claude-code-tools-kg.db`, though several of its other figures cite graphs
that do not exist on this system.)

**BFS, not DFS**, because we want *minimum* depth per node. Breadth-first
reaches every node by a shortest path first, so `best_depth` is correct on
first visit and never needs revising - which is also what makes the visited
set safe.

Stdlib only.
"""

from collections import deque

from ..contract import ActionOutput, action
from ..store import neighbours

# Depth recorded for a candidate the graph never reached. Not a magic number
# used in arithmetic - it only has to sort *after* every real depth, and it is
# an int so it can sit in the sort key alongside the rest.
UNREACHED_DEPTH = 1_000_000


def expand(conn, seeds: list[str], max_depth: int) -> dict[str, int]:
    """Minimum hop distance from any seed, out to `max_depth`.

    Seeds themselves are depth 0. The traversal is a pure function of the
    frozen index and the seed list: `neighbours()` returns sorted results, the
    frontier is processed in sorted order, and BFS assigns each node its
    minimum depth regardless, so the output does not depend on seed order.
    That last property is checked directly by
    `tests/test_stage1.py::test_expansion_is_independent_of_seed_order`.
    """
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}")

    depths: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()
    for seed in sorted(set(seeds)):
        depths[seed] = 0
        queue.append((seed, 0))

    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbour in neighbours(conn, node):
            if neighbour not in depths:
                depths[neighbour] = depth + 1
                queue.append((neighbour, depth + 1))
    return depths


@action(
    "graph.expand",
    determinism="deterministic",
    authority="authoritative",
    inputs=("seeds", "max_depth", "index_hash"),
)
def graph_expand(
    *, conn, seeds: list[str], max_depth: int, index_hash: str
) -> ActionOutput:
    """Audited entry point for graph expansion.

    `index_hash` is a declared input for the same reason it is on the lexical
    actions: without an index identity, two different indexes expanded from
    the same seeds would collide on one replay-log key and the second would
    raise DeterminismViolation for being correctly different.
    """
    depths = expand(conn, seeds, max_depth)
    reached = len(depths) - len(set(seeds))
    return ActionOutput(
        value=sorted(depths.items()),
        evidence=(
            f"seeds={len(set(seeds))}",
            f"max_depth={max_depth}",
            f"reached={reached}",
        ),
    )
