<!-- GENERATED FILE - DO NOT EDIT.
     Produced by `drf docs build` from spec/*.json and a built index.
     Hand edits are detected by tests/test_docs.py, which re-renders
     and compares. Change the spec, then regenerate. -->

# Deterministic RAG — technical reference

A retrieval framework whose ranking path contains no model. A neural layer may
be attached under a mechanically enforced guarantee that it can never change an
authoritative result — only append below it.

**Spec hash** `fe40d6b09a675ff68f95959c4936837d126c7819d6093ee23a78335ee8862602`
**Index** `90ab5db969588b5a2a41beddce996cd3bf25d27b28d9791f984416d8b33cf72a`
**Versions** parser `1.1.0`, ranker `1.0.0`, id-schema `1`, manifest `1`

---

## Scope, stated before the results

49 model-generated judgements over 7 queries now exist (stratum A). They are NOT human judgements, and the annotator authored the retrieval system - see retrieval_quality.annotator. The one quality measurement made from them is a FAIL against a margin declared in advance. No positive claim about retrieval quality is made anywhere in this repository.

7 queries with 1-7 candidates each. The per-query record against a relevance-blind control is 3 wins / 3 losses / 1 tie. Any aggregate quality figure from this label set is dominated by single-query variation and must not be quoted as a system property.

What M1 proves is reproducibility and subordination. Recall, nDCG and MRR need labels. An unqualified 'all metrics 1.0' would be exactly the drift this framework exists to prevent.

266 nodes is small. Every determinism metric is perfect here, which demonstrates determinism, not scalability.

Four further limits, recorded rather than discovered later:

- Candidate generation has no truncation and degrades at scale. FTS5 with a mandatory ORDER BY rank plus deterministic pagination is the milestone 3 path.
- An all-out-of-vocabulary query produces empty D and therefore no proposals. Correct under subordination; the fix, if wanted, is a Stage 1 fix such as character n-grams, never a neural one.
- Measured: |D| runs 0 to 147 across the 23-query set, so the depth at which the advisory layer can act at all is a property of the query, not of the corpus. A quality comparison at any depth <= |D| is guaranteed to show zero difference between provider on and off.
- Only 2 of 23 queries reorder under any graph setting, and one of those is a synthetic 13-term edge case. 'Does the graph layer earn its place?' therefore has a real sample size of one and cannot be settled by labelling; it needs more queries, not more labels.

---

## Two independent axes

Determinism (same input, same output) and authority (may influence ranking) are
orthogonal. Conflating them is the mistake the contract exists to prevent:
anchor-mode search over frozen vectors is fully *deterministic* and strictly
*advisory*, because reproducibility is not a licence to decide.

10 actions are declared, 10 deterministic and
2 advisory.

| action | determinism | authority | what it does |
|---|---|---|---|
| graph.expand | deterministic | authoritative | Bounded breadth-first expansion from the top-ranked seeds. |
| ingest.build_index | deterministic | authoritative | Build the content-addressed index from a source knowledge graph. |
| lexical.bm25_score | deterministic | authoritative | Okapi BM25 with k1, b and document-length normalisation. |
| lexical.candidates | deterministic | authoritative | Union of postings over the sorted query terms. |
| merge.append_advisory | deterministic | authoritative | Append advisory proposals below the authoritative order. |
| neural.encode_query_remote | deterministic | advisory | Encode query text via a remote embedding server. |
| neural.propose_from_anchors | deterministic | advisory | Propose nodes similar to the top-ranked members of D, using frozen vectors. |
| stage1.rank | deterministic | authoritative | Produce the authoritative total order D. |
| store.load_manifest | deterministic | authoritative | Read and validate the index manifest. |
| tokenize.terms | deterministic | authoritative | Normalise text to an ordered term list. Shared by index and query paths. |

Each label is enforced, not documented. Deterministic actions are replay-checked
— calling one twice with the same inputs and getting different results raises.
Probabilistic actions must declare a confidence; deterministic ones must not.
Advisory results are boxed in `Advisory[T]`, whose `unwrap()` refuses any caller
outside a one-module allowlist.

## The three stages

```
Stage 1  AUTHORITATIVE  BM25 + graph traversal -> total order D
Stage 2  ADVISORY       neural proposes candidates not in D
Stage 3  MERGE          append-only; order(D) preserved exactly
```

The guarantee, checked at runtime on every query rather than only in CI:

```
merged[:len(D)] == D,  elementwise, in order, always
```

There is no weighted combination of lexical and neural signal. Any weight would
grant authority; as a product it would grant veto. Rank fusion (RRF) is
therefore rejected rather than merely unused.

## Why the order is unambiguous

```
sort key = (-bm25_q, -matched_terms, best_depth, doc_len, node_id)
```

Every component is `int` or `str`; no float is ever compared. The final
component is the content-addressed node id, which is injective over the
candidate set, so the induced order is strict and total. `sorted()` therefore
returns the same sequence for every input permutation, sort stability becomes
irrelevant, and truncation to top-k cannot cascade.

This is load-bearing rather than defensive: 7/15 queries have exact
BM25 ties in their top set, so without the tiebreak chain "the best result" is
undefined for nearly half of ordinary queries.

## Scoring

```
idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * dl(d) / avgdl))
idf(t) = ln((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
```

`k1 = 1.2`, `b = 0.75`. Contributions are accumulated with `math.fsum` and
quantised to `int` exactly once, after_summation, with
`QUANTUM_EXP = 9`. Because the correctly-rounded sum of a multiset is
unique, accumulation is commutative — the result does not depend on the order
contributions are visited, so no upstream sort is load-bearing.

## Measurements

Every figure below is stored in `spec/benchmarks.json` beside the command that
produces it.

### Reproducibility — `drf bench repro --index a.db --compare b.db`

{"in_process_repeats": 5, "subprocess_repeats": 3, "hash_seeds": ["0", "1", "12345"], "independent_builds": 2}

28 cells, 1 distinct digest, 0 discordant pairs.

### The control that makes those numbers mean something — `drf bench chaos --index a.db`

Every metric scores 1.0 on the real pipeline, so perfect scores are compatible with a harness that measures nothing. This control is the only evidence the harness can report failure.

The prior engine's ranking reproduced exactly: sort by score alone with no tiebreak, over candidates in arbitrary order. python_apps_hybrid_query.py:304 applied to the unordered LIMIT 100 at :342. Not synthetic noise - the implementation this framework replaced.

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

- Set-based metrics are blind to ordering non-determinism. Jaccard reports 1.0000 for a pipeline returning five different orderings in five runs.
- Kendall's Tau of 0.9761 and RBO of 0.9942 describe a provably non-deterministic pipeline. Rounded for a report they read as 0.98 and 0.99. This is the measured reason assertions in this project use exact integers.

### Length normalisation — `python3 tools/measure_length_norm.py --index index.db`

The prior engine had no `b` and no `avgdl`. Measured against a correct
implementation over strictly-ordered comparisons only:
17.0% of 352 pairs discordant, direction
longer/shorter/equal 9/0/6.

**Correction on record.** Earlier figures of 26.2% discordance and 'top-1 changed 9 of 15' were tiebreak-dependent and are withdrawn. 7 of 15 queries have exact ties in a top set; breaking them by lowest node id gives 9, by highest gives 4. Those numbers described BM25 plus an arbitrary convention. All figures here are defined only over strictly-ordered comparisons. The directional claim survived unchanged.

### Graph expansion

Bidirectional, on measurement rather than preference: mean depth-2 reach
23.3 against 7.2 forward-only, and
81/266 nodes reach nothing forward-only against
5/266 bidirectionally. No precomputed path index — BFS costs
0.149 ms for 10 seeds at depth 2.

### The advisory horizon — `drf eval invariance --index index.db`

`|D|` per query: the depth below which the advisory layer provably cannot act,
because merge is append-only. Across 23 queries `|D|` runs
**0 to 147**, and the authoritative prefix is identical with
the provider on and off in 23 of them
(0 differing).

The advisory layer's reach is inverse to lexical success. Where stage 1 returned 20 or more documents it is structurally silent at every evaluated depth; where it returned one or two, the advisory layer can act from depth 5 down. The neural layer can only speak where lexical retrieval did badly, and is provably mute where it did well - which is what append-only subordination means, stated as a measurement rather than a design intention.

A reachable horizon is necessary but not sufficient. The three out-of-vocabulary queries have |D| = 0, so by horizon alone the advisory layer could occupy every position - and it proposes nothing, because anchor-mode search takes its anchors from D. Anchor starvation is a separate bound from the horizon, and a recall figure on those queries would measure the starvation rather than the provider.

A quality comparison at any depth <= |D| is guaranteed to show zero difference between provider on and off. Reporting that as a finding about neural retrieval would be a category error, and measuring the horizon first is what makes the error visible in advance.

### Retrieval quality — `python3 tools/drf eval quality --index index.db`

49 judgements over 7 queries,
`labels_hash 33d8e1beb4302bbc5c6ada98d97b9047ff84581b6552387676a8648e96c7a439`.

**FAIL at every depth against the margin declared on 2026-08-08, before any label existed.**

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

### Sensitivity — `drf bench sensitivity --index a.db`

Over 23 queries, each setting probed with the value recorded
in `spec/config_schema.json`:

| setting | default | probe | queries reordered |
|---|---|---|---|
| ranking.b | 0.75 | 0.375 | 13 |
| ranking.k1 | 1.2 | 0.6 | 9 |
| graph.max_depth | 2 | 3 | 2 |
| graph.seed_count | 10 | 20 | 2 |

Lexical parameters dominate; graph parameters are live but weak. seed_count 10 -> 11 reorders zero of 15 corpus queries - it moves best_depth for 7 of them but never enough to change what a user sees.

## Falsifiers

24 checkpoint invariants each carry a falsifier: a mutation under
which the named test *must* fail. A test that survives its own falsifier cannot
fail, and a test that cannot fail proves nothing. 3 checkpoints
are deliberately exempt, each with a recorded reason — they are guarded at
runtime, so falsifying them would prove the guard rather than the test.

| invariant | asserts | falsifier |
|---|---|---|
| ddl_forbidden_constructs | The built index DDL contains zero AUTOINCREMENT and zero CURRENT_TIMESTAMP. | Append a table carrying INTEGER PRIMARY KEY AUTOINCREMENT to store.SCHEMA - the shape a plausible future 'build_log' table would take. |
| collapse_preserves_metadata | The conflict-free union keeps the semantic-analysis confidence/reasoning on the 4 payload-divergent duplicate groups. | Replace collapse_edge_group with the strip-metadata rule - the exact alternative rejected when the collapse rule was chosen. |
| dangling_edges_not_repaired | No dropped edge reappears in the index under a fuzzy-matched endpoint. | Replace normalize.resolve_endpoint with a substring-matching resolver that maps a missing slug onto the first node whose source_ref contains it. |
| isolated_nodes_kept | The 5 nodes with no edges remain in the index and stay searchable. | Filter nodes with no incident edge out of normalize_all. |
| content_hash_ignores_hash_seed | The manifest content_hash is identical under PYTHONHASHSEED 0, 1 and 12345. | Add node_types as list(set(...)) rather than sorted(set(...)) in manifest.build_content - 25 distinct values, so iteration order genuinely varies. |
| bm25_length_normalisation | A long document repeating the query term (tf=2, dl=32) must not outrank a short exact match (tf=1, dl=1). | Force b=0.0 in bm25.score_documents. |
| candidates_not_truncated | Query terms beyond the tenth still contribute candidates. | Truncate the term list to its first ten before looking up postings. |
| scores_are_int | Every value that can reach a sort key is int, never float. | Replace quantize with a float-preserving multiplication. |
| one_tokenizer | The query path and the index path produce identical terms. | Make the query path drop tokens of three characters or fewer, a filter indexing does not apply. |
| strict_total_order | Every candidate in D has a distinct sort key, so the order is strict and total. | Drop the trailing node_id component from stage1.sort_key. |
| bidirectional_expansion | Graph expansion follows edges in both directions. | Restrict store.neighbours to outgoing edges only. |
| merge_is_append_only | merged[:len(D)] equals D elementwise, for every provider including hostile ones. | Promote one advisory result into second place AND neuter merge's runtime postcondition. |
| advisory_allowlist | Advisory.unwrap() raises AuthorityViolation outside drf.retrieval.merge. | Add the test module to contract.ADVISORY_CONSUMERS. |
| config_hash_ignores_display | Changing a presentation setting does not change a configuration's content_hash. | Fold every setting into the hash rather than only the ranking ones. |
| config_hash_covers_ranking | Every setting flagged affects_ranking is folded into content_hash. | Drop one ranking key from Config.ranking_settings(). |
| config_rejects_unknown_keys | An unknown setting name raises rather than being stored. | Make validate_one accept unknown keys silently. |
| ranking_params_are_live | Every setting flagged affects_ranking reorders results for at least one query, using its declared sensitivity_probe. | Make stage1.rank ignore max_depth and use the default instead. |
| bench_detects_nondeterminism | The chaos control - the prior engine's ranking, score-only with no tiebreak over unordered candidates - produces more than one distinct digest. | Give chaos_run the injective tiebreak back, making it deterministic. |
| docs_are_generated | Each committed docs/*.md is byte-identical to a fresh render from spec/ plus a built index. | Append a line to whatever render_document produces, so a fresh render diverges from what is on disk. |
| docs_fail_on_missing_placeholder | Rendering a template whose placeholder the context does not supply raises KeyError. | Use Template.safe_substitute instead of substitute. |
| quality_ranks_detect_reordering | ranks_of_relevant distinguishes a ranking from its exact reversal, in integers, where recall cannot. | Return the relevant documents' ranks sorted by node id rather than by rank - the shape the tuple takes if someone 'simplifies' it toward a set. |
| quality_labels_reject_conflicts | Two judgements of the same query/document pair at different grades raise rather than resolving. | Take the last judgement seen, the ordinary dict-assignment behaviour. |
| quality_unknown_label_rejected | A judgement naming a node id absent from the index raises rather than being skipped. | Skip unresolvable judgements, the tolerant behaviour that looks like robustness. |
| advisory_horizon_is_live | advisory_horizon returns |D| per query, so the depths the advisory layer can reach are computed rather than assumed. | Return a large constant, making every depth look structurally unreachable. |

## Configuration

9 settings, of which 4 can influence ranking.
A configuration's content hash covers exactly those, so two configurations
differing only in presentation are provably the same computation. Neural
settings are excluded on the same grounds as display settings: they cannot
influence the authoritative prefix.

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

## Field anchors

- **SAT-Graph RAG** (arXiv 2510.06002) — the determinism boundary as a typed
  action contract. Has no evaluation, and says so.
- **ReproRAG** (arXiv 2509.18869) — metric suite and a measured hierarchy of
  uncertainty; embedding-model choice dominates, while ANN index algorithms
  scored a perfect 1.000, falsifying the common assumption.

A fixed vector pipeline *is* reproducible. The defensible claim is narrower:
reproducible conditional on a model choice that is itself arbitrary and drives
much of the result variance.
