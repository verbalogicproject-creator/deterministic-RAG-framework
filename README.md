<!-- GENERATED FILE - DO NOT EDIT.
     Produced by `drf docs build` from spec/*.json and a built index.
     Hand edits are detected by tests/test_docs.py, which re-renders
     and compares. Change the spec, then regenerate. -->

# Deterministic RAG Framework

**A retrieval system whose ranking path contains no model.**

Ask the same question twice and you get exactly the same answer — same results,
same order, on any machine, on any day. A neural layer can be attached, but it
is mechanically prevented from changing an authoritative result. It can only
append below one.

```
release 0.0.3 · spec 2576b96ccc5b · index 90ab5db96958
Python 3.12+ · standard library only · no dependencies
```

---

## The guarantee

```
Stage 1  AUTHORITATIVE  BM25 + graph traversal -> total order D
Stage 2  ADVISORY       neural proposes candidates not in D
Stage 3  MERGE          append-only; order(D) preserved exactly
```

Checked at runtime on **every query**, not only in tests:

```
merged[:len(D)] == D,  elementwise, in order, always
```

So the worst a broken, hostile, or brilliant provider can do is contribute
nothing. There is no weighted blend of lexical and neural signal — any weight
grants authority. Rank fusion (RRF) is rejected, not merely unused.

## Try it

```bash
./tools/drf build --source /path/to/knowledge-graph.db --out index.db
./tools/drf query "prompt caching" --explain
./tools/drf query "prompt caching" --neural stored
```

Diff the last two. The authoritative block is identical, character for
character. That is the promise, and it takes about ten seconds to check.

## What is proven

| claim | evidence |
|---|---|
| Same question, same answer | 28 cells — repeats, subprocesses, hash seeds, two independent builds — **1** distinct digest, **0** discordant pairs |
| The tests can detect failure | The same suite against the engine this replaced: **5** different answers to one question in 5 runs |
| The neural layer cannot interfere | 8 providers including crashing, hanging, flooding, and one actively trying to promote itself |
| Documentation cannot drift | Every doc is generated from `spec/`; a hand edit fails the suite |

24 checkpoint tests each carry a **falsifier** — a mutation under
which that test must fail. A test that survives its own falsifier cannot fail,
and a test that cannot fail proves nothing.

## What is NOT proven

Milestone 1 carries no relevance judgements and makes NO claim about retrieval quality. It proves reproducibility and subordination. Recall, nDCG and MRR require labels and belong to milestone 2. An unqualified 'all metrics 1.0' would be exactly the drift this framework exists to prevent.

266 nodes is small. Every determinism metric is perfect here, which demonstrates determinism, not scalability.

Read `docs/peer.md` for the full list of limits — they are stated before the
results, not after.

## Documentation

Four audiences, all generated from `spec/*.json` and a built index:

| file | for |
|---|---|
| `docs/plain.md` | a reader with no background — start here |
| `docs/operator.md` | running it |
| `docs/peer.md` | architecture, measurements, field anchors |
| `docs/agent.md` | a future AI session: verified facts, what raises, traps already hit |

## Design rules

1. **No number without a producer.** Counts come from `len()`, never memory.
   Every figure quoted lives in `spec/benchmarks.json` beside the command that
   emits it.
2. **Assert exact integers, never floats.** Measured reason: the broken control
   scores Kendall's Tau 0.9761 and RBO 0.9942 while being provably
   non-deterministic. Rounded, those read as correct.
3. **Prefer commutativity to pinned order.** If something is reproducible only
   because it was sorted first, find the formulation where order cannot matter.
4. **Every checkpoint needs a falsifier**, registered before the test is
   trusted.

## Origin

Formalisation of research recovered in August 2026. The engine it replaces had
an embedding term that never ran (`return 0.5  # Placeholder`), a "BM25" with no
length normalisation, three stacked silent truncations, and documentation
claiming `80% vs 60% accuracy` against an evaluation that did not exist.

Those are the failures this framework is built to make impossible.
