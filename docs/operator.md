<!-- GENERATED FILE - DO NOT EDIT.
     Produced by `drf docs build` from spec/*.json and a built index.
     Hand edits are detected by tests/test_docs.py, which re-renders
     and compares. Change the spec, then regenerate. -->

# Deterministic RAG — running it

**Index** `90ab5db969588b5a2a41beddce996cd3bf25d27b28d9791f984416d8b33cf72a`
266 nodes · 553 edges · 1303 terms · 4185 postings · 228 vectors

Python 3.12+, standard library only. Nothing to install.

## Build an index

```bash
./tools/drf build --source /path/to/claude-cookbook-kg.db --out index.db
```

Prints the content hash and what was dropped. Two builds from the same source
produce the same hash, on any machine, at any time.

## Check an index

```bash
./tools/drf verify --index index.db
./tools/drf verify --index index.db --compare other.db
```

Exits non-zero if the manifest was altered, if counts disagree with the
database, or if the two indexes differ. `inspect` prints the full manifest.

## Query

```bash
./tools/drf query "prompt caching"
./tools/drf query "prompt caching" --neural stored --explain
./tools/drf query "prompt caching" -k 20 --seeds 10 --depth 2
```

Results are labelled `authoritative` or `advisory`. **The authoritative block is
identical whether the neural layer is on or off** — verify it yourself by
diffing the two outputs. Advisory results are only ever appended below.

`--explain` prints, for each step, whether it was deterministic or
probabilistic and whether it was allowed to influence the ranking.

## Benchmarks

```bash
./tools/drf bench repro --index a.db --compare b.db
./tools/drf bench chaos --index a.db
./tools/drf bench sensitivity --index a.db
```

`repro` runs 28 cells across four axes — repeats, subprocesses, hash
seeds, independent builds — and should report **1** distinct digest and **0**
discordant pairs.

`chaos` is the one worth understanding. Perfect scores prove nothing on their
own, because a benchmark that measures nothing also scores perfectly. `chaos`
runs the same measurement against the older, broken ranking so you can see the
suite reporting failure:

| metric | real | chaos | separates? |
|---|---|---|---|
| distinct_digests | 1 | 5 | yes |
| mismatched_positions | 0 | 551 | yes |
| discordant_pairs | 0 | 3931 | yes |
| exact_match_rate | 1.0000 | 0.6196 | yes |
| kendall_tau | 1.0000 | 0.9761 | barely |
| rbo | 1.0000 | 0.9942 | barely |
| jaccard | 1.0000 | 1.0000 | NO |
| overlap_coefficient | 1.0000 | 1.0000 | NO |

`sensitivity` checks that every setting advertised as affecting ranking actually
changes results:

| setting | default | probe | queries reordered |
|---|---|---|---|
| ranking.b | 0.75 | 0.375 | 13 |
| ranking.k1 | 1.2 | 0.6 | 9 |
| graph.max_depth | 2 | 3 | 2 |
| graph.seed_count | 10 | 20 | 2 |

## Settings

9 settings, 4 of which affect ranking. Only
those are part of a configuration's identity — changing how many results you
display does not make it a different computation.

| setting | default | affects ranking | what it is |
|---|---|---|---|
| display.explain | False | no | Print the justification trace. |
| display.format | text | no | Output rendering. |
| display.k | 10 | no | How many results to print. Presentation only. |
| graph.max_depth | 2 | yes | Hop limit for bidirectional expansion. |
| graph.seed_count | 10 | yes | How many top lexical hits seed graph expansion. |
| neural.limit | 10 | no | Maximum advisory proposals appended below D. |
| neural.provider | off | no | Advisory provider. Cannot affect the authoritative prefix - see merge.py. |
| ranking.b | 0.75 | yes | BM25 document-length normalisation. The load-bearing parameter; the prior engine omitted it entirely. |
| ranking.k1 | 1.2 | yes | BM25 term-frequency saturation. Weak on this corpus - tf == 1 for 82.7% of (document, term) pairs. |

## What it will not do

49 model-generated judgements over 7 queries now exist (stratum A). They are NOT human judgements, and the annotator authored the retrieval system - see retrieval_quality.annotator. The one quality measurement made from them is a FAIL against a margin declared in advance. No positive claim about retrieval quality is made anywhere in this repository.

- 266 nodes is small. Every determinism metric is perfect here, which demonstrates determinism, not scalability.
- An all-out-of-vocabulary query produces empty D and therefore no proposals. Correct under subordination; the fix, if wanted, is a Stage 1 fix such as character n-grams, never a neural one.

## Regenerating documentation

```bash
./tools/drf docs build --index index.db
```

These files are generated from `spec/`. Editing them directly fails the test
suite — change the spec and regenerate.
