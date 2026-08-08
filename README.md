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
release 0.0.6 · spec 56f56a620b55 · index 90ab5db96958
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

## The first quality measurement — and it fails

`python3 tools/drf eval quality --index index.db`

FAIL at every depth against the margin declared on 2026-08-08, before any label existed.

| depth | system nDCG | best blind control | its nDCG | oracle | margin | required | per-query W/L/T |
|---|---|---|---|---|---|---|---|
| 1 | 0.8367 | shuffle_3 | 0.7959 | 1.0 | 0.0408 | 0.05 | 2W 1L 4T |
| 5 | 0.8787 | id_order | 0.8412 | 0.9811 | 0.0376 | 0.05 | 3W 3L 1T |
| 10 | 0.9064 | id_order | 0.8761 | 0.9787 | 0.0303 | 0.05 | 3W 3L 1T |
| 20 | 0.9064 | id_order | 0.8761 | 0.9787 | 0.0303 | 0.05 | 3W 3L 1T |

- The system fails the declared margin at every depth. The margin was fixed at 0.05 on 2026-08-08 before any judgement existed and has NOT been revised; a threshold changed after seeing the number it must pass is not a threshold.
- The mean understated the problem. Per query at depth 10 the record against id_order is 3 wins, 3 losses, 1 tie - a coin flip. The positive mean comes from one query winning by +0.3569 while three losses were smaller. On this label set there is NO evidence the ranking beats sorting by sha256 hash.
- id_order - sorting by content hash, relevance-blind and perfectly deterministic - scores 0.8761 nDCG@10. It is the counter-example the control was chosen to provide: determinism is not a quality property, and here it is nearly as good as the ranker on the metric.
- The advisory layer found exactly ONE relevant document that lexical retrieval missed, across 7 queries. Of 21 stored-vector proposals, 18 were graded irrelevant, 2 marginal, 1 relevant. Recall of D alone is already 0.9714 at depth 10, so there was almost no room to contribute.
- Small-sample saturation is severe. Candidate sets run 1 to 7 documents; q08 has |D| = 1, where every ranker ties trivially. nDCG's log discount is gentle enough that even a poor ordering of a mostly-relevant set scores above 0.77.

**What this does not show.** That the ranking is bad. 7 queries and a 3-3 split is absence of evidence, not evidence of absence - the same data is consistent with a real but small effect. What it does show is that the current evidence CANNOT support a quality claim, which is the outcome the harness was built to make unavoidable.

**Provenance.** MODEL-GENERATED (Claude Opus 5), graded blind to rank. NOT human judgements. The annotator also authored the retrieval system, which is a real confound: relevance calls may correlate with what this ranking already surfaces. Blind grading removes position bias but not authorship bias.

The margin was fixed at 0.05 in `spec/evaluation.json` on the day the harness
was built, before any judgement existed, and **it has not been revised**. A test
checks the recorded rows against the live declared value, so lowering the
threshold to convert this failure into a pass fails the suite instead.

## What is NOT proven

49 model-generated judgements over 7 queries now exist (stratum A). They are NOT human judgements, and the annotator authored the retrieval system - see retrieval_quality.annotator. The one quality measurement made from them is a FAIL against a margin declared in advance. No positive claim about retrieval quality is made anywhere in this repository.

7 queries with 1-7 candidates each. The per-query record against a relevance-blind control is 3 wins / 3 losses / 1 tie. Any aggregate quality figure from this label set is dominated by single-query variation and must not be quoted as a system property.

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
