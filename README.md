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
release 0.0.5 · spec c747755c9fde · index 90ab5db96958
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

```bash
./tools/drf bench all --index index.db     # reproducibility, control, sensitivity
./tools/drf eval invariance --index index.db   # where the neural layer can act
./tools/drf verify --index index.db
```

## What is proven

| claim | evidence |
|---|---|
| Same question, same answer | 28 cells — repeats, subprocesses, hash seeds, two independent builds — **1** distinct digest, **0** discordant pairs |
| The tests can detect failure | The same suite against the engine this replaced: **5** different answers to one question in 5 runs |
| The neural layer cannot interfere | 8 providers including crashing, hanging, flooding, and one actively trying to promote itself |
| The guarantee holds on real queries | Authoritative prefix identical with the provider on and off, **23** of 23 queries |
| Documentation cannot drift | Every doc is generated from `spec/`; a hand edit fails the suite |

24 checkpoint tests each carry a **falsifier** — a mutation under
which that test must fail. A test that survives its own falsifier cannot fail,
and a test that cannot fail proves nothing.

## Where the neural layer can act — and where it provably cannot

`drf eval invariance --index index.db`

Merge is append-only, so everything above `|D|` is identical whatever the
provider does. Measured across 23 queries, `|D|` runs from
**0 to 147** — the bound is a property of the *query*.

The advisory layer's reach is inverse to lexical success. Where stage 1 returned 20 or more documents it is structurally silent at every evaluated depth; where it returned one or two, the advisory layer can act from depth 5 down. The neural layer can only speak where lexical retrieval did badly, and is provably mute where it did well - which is what append-only subordination means, stated as a measurement rather than a design intention.

A reachable horizon is necessary but not sufficient. The three out-of-vocabulary queries have |D| = 0, so by horizon alone the advisory layer could occupy every position - and it proposes nothing, because anchor-mode search takes its anchors from D. Anchor starvation is a separate bound from the horizon, and a recall figure on those queries would measure the starvation rather than the provider.

This is the guarantee working, not a weakness. It is stated here because it
also bounds what any future evaluation can show: A quality comparison at any depth <= |D| is guaranteed to show zero difference between provider on and off. Reporting that as a finding about neural retrieval would be a category error, and measuring the horizon first is what makes the error visible in advance.

## What is NOT proven

There are no relevance judgements in this repository, so NO claim is made about retrieval quality. Milestone 2.0 built the instrument that would measure it - metrics, controls, a label format and self-checks - and deliberately built it before the labels, so that 'is the harness right?' and 'is the system good?' are not the same experiment. M2.1 supplies the judgements. `drf eval quality` prints that no labels exist rather than printing a zero, and a test walks every spec file to keep it true.

What M1 proves is reproducibility and subordination. Recall, nDCG and MRR need labels. An unqualified 'all metrics 1.0' would be exactly the drift this framework exists to prevent.

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
2. **Assert exact integers, report floats.** Measured reason: the broken control
   scores Kendall's Tau 0.9761 and RBO 0.9942 while being provably
   non-deterministic. Rounded, those read as correct. Where a metric is a ratio
   and *cannot* be an integer — nDCG, recall — the rule keeps its shape rather
   than its letter: rank positions and counts are asserted, the ratio is
   reported, and no improvement may be claimed except against a control by a
   margin stated in advance.
3. **Prefer commutativity to pinned order.** If something is reproducible only
   because it was sorted first, find the formulation where order cannot matter.
4. **Every checkpoint needs a falsifier**, registered before the test is
   trusted.

## Status

Milestone 1 is complete: the deterministic path, the subordination guarantee,
the reproducibility suite and its control, generated documentation, and a
freeze that binds a release to an exact spec, index and result set.

Milestone 2 is measuring quality, and its first step was the **instrument**.
`spec/evaluation.json` declares the metrics, the relevance scale, the controls
and the required margin — all dated before any judgement existed, because a
threshold chosen after seeing the number it must pass is not a threshold. The
next step is the judgements themselves; see `queries/LABELLING.md`.

## Origin

Formalisation of research recovered in August 2026. The engine it replaces had
an embedding term that never ran (`return 0.5  # Placeholder`), a "BM25" with no
length normalisation, three stacked silent truncations, and documentation
claiming `80% vs 60% accuracy` against an evaluation that did not exist.

Those are the failures this framework is built to make impossible.
