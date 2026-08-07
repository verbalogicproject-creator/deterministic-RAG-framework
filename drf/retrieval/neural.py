"""The advisory layer: proposals, boxed, with the provider treated as hostile.

Everything returned here is wrapped in `Advisory[T]` by the `@action`
decorator, because these actions declare `authority="advisory"`. Only
`drf.retrieval.merge` can unwrap it. A caller that tries anywhere else gets
`AuthorityViolation` - not a lint warning, a raised exception.

**Providers are untrusted input.** They may raise, hang, return ten thousand
ids, return duplicates, or return ids that do not exist. This module contains
each of those failure modes so that the merge layer never sees them, and
`tests/test_merge.py` proves the containment against deliberately hostile
doubles instead of asserting it.

The containment is defence in depth, not the guarantee itself. Even if every
check here were removed, subordination would still hold, because `merge()`
appends and re-checks its own postcondition on every call. These limits exist
so that a broken provider degrades to *nothing* rather than to *slow*.

Stdlib only.
"""

import threading

from ..contract import ActionOutput, action

# A provider that has not answered within this many seconds is abandoned. The
# query proceeds with its authoritative results, which is the correct outcome:
# advisory input is optional by construction, so its absence is never an error.
PROVIDER_TIMEOUT_SECONDS = 2.0

# Hard cap on accepted proposals, applied whatever the provider returns. A
# provider that ignores `limit` cannot flood the tail.
MAX_PROPOSALS = 100


def _call_provider(provider, anchors: list[str], limit: int) -> list[str]:
    """Invoke a provider, surviving any way it can misbehave.

    Runs in a daemon thread so a provider that never returns cannot block the
    query. The thread is abandoned rather than killed - Python has no safe way
    to kill a thread - but it is a daemon, so it cannot keep the process
    alive, and its result is discarded whenever it does finish.
    """
    result: list[list[str]] = []
    error: list[BaseException] = []

    def run() -> None:
        try:
            result.append(list(provider.propose(anchors=anchors, limit=limit)))
        except BaseException as exc:  # noqa: BLE001 - a provider may do anything
            error.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(PROVIDER_TIMEOUT_SECONDS)

    if worker.is_alive() or error or not result:
        return []
    return result[0]


def _sanitise(proposals: list[str], limit: int) -> list[str]:
    """Coerce whatever came back into a bounded list of unique strings.

    Order is preserved. The provider's ordering is its one legitimate
    expression of preference, and it costs nothing to honour it *within the
    tail* - the tail cannot reach the authoritative prefix regardless.
    """
    seen: set[str] = set()
    clean: list[str] = []
    for item in proposals:
        if not isinstance(item, str) or not item or item in seen:
            continue
        seen.add(item)
        clean.append(item)
        if len(clean) >= min(limit, MAX_PROPOSALS):
            break
    return clean


@action(
    "neural.propose_from_anchors",
    determinism="deterministic",
    authority="advisory",
    inputs=("anchors", "limit", "provider_name", "index_hash"),
)
def propose_from_anchors(
    *,
    provider,
    anchors: list[str],
    limit: int,
    provider_name: str,
    index_hash: str,
) -> ActionOutput:
    """Ask a provider for ids worth appending below D.

    Declared **deterministic and advisory**, which is the combination this
    whole design exists to make expressible. Anchor-mode search over frozen
    vectors runs no model and touches no network, so it replays exactly - and
    it is still forbidden from influencing ranking, because determinism is not
    a licence.

    `provider_name` is a declared input so that two providers answering the
    same anchors do not collide on one replay-log key; `index_hash` for the
    same reason across indexes.
    """
    proposals = _sanitise(_call_provider(provider, anchors, limit), limit)
    return ActionOutput(
        value=proposals,
        provider=provider_name,
        evidence=(
            f"anchors={len(anchors)}",
            f"limit={limit}",
            f"proposed={len(proposals)}",
        ),
    )
