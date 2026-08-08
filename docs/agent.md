<!-- GENERATED FILE - DO NOT EDIT.
     Produced by `drf docs build` from spec/*.json and a built index.
     Hand edits are detected by tests/test_docs.py, which re-renders
     and compares. Change the spec, then regenerate. -->

# Deterministic RAG — operating notes for an agent

Read this before changing anything. It states what is enforced, what will raise,
and which assumptions are already measured so you do not re-derive them.

**Spec hash** `77302e076be0b25173227381b0b7426e35180abd11ddd8301ca325e273e96b97` · **Index** `90ab5db969588b5a2a41beddce996cd3bf25d27b28d9791f984416d8b33cf72a`
**Versions** parser `1.1.0`, ranker `1.0.0`, id-schema `1`

## Verified facts — do not re-measure, do not assume otherwise

| fact | value |
|---|---|
| nodes / edges / embeddings | 266 / 553 / 228 |
| terms / postings | 1303 / 4185 |
| avgdl | 18.8684 |
| BM25 k1 / b | 1.2 / 0.75 |
| quantisation | 10^9, after_summation |
| queries with tied top set | 7/15 |
| reproducibility cells / digests | 28 / 1 |

## What raises, and why

- **`DeterminismViolation`** — an action declared deterministic returned two
  different results for one inputs hash. The label is load-bearing; if you add a
  parameter that changes output, add it to that action's `inputs` or this fires
  spuriously.
- **`AuthorityViolation`** — something outside `drf.retrieval.merge` called
  `Advisory.unwrap()`. Do not widen the allowlist. That is how the guarantee
  erodes: one locally-reasonable module at a time.
- **`SubordinationViolation`** — a merge disturbed the authoritative prefix.
  Unreachable by construction; if you see it, the architecture is broken and the
  result must not be returned.
- **`ConfigError`** — unknown or ill-typed setting. Typos are errors, never
  silently ignored keys.
- **`BuildError`** — manifest counts disagree with rows written, or edge
  reconciliation did not balance.

## Rules that are not negotiable

1. **No number without a producer.** Counts come from `len()` or
   `SELECT count(*)`, never a literal you remembered. Figures quoted in docs
   live in `spec/benchmarks.json` beside the command that emits them.
2. **Assert exact integers, never floats.** Measured reason: the chaos control
   scores Kendall's Tau 0.9761 and RBO 0.9942 while being provably
   non-deterministic. Rounded, those read as correct.
3. **Prefer commutativity to pinned order.** If an operation is reproducible
   only because something sorted first, look for a formulation where order
   cannot matter. `fsum`, conflict-free union, and the injective sort key are
   the same principle three times.
4. **Docs are generated.** Edit `spec/`, run `drf docs build`. A hand edit fails
   `tests/test_docs.py`.
5. **Every checkpoint needs a falsifier.** Before trusting a new assertion,
   register a mutation that makes it fail. If it survives, it is decoration.

## The axes you must keep separate

10 actions, 10 deterministic, 2
advisory. Deterministic does **not** imply authoritative. Anchor-mode vector
search runs no model and replays exactly, and it is still forbidden from
influencing ranking.

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

## Known-weak signals — do not overstate these

Lexical parameters dominate; graph parameters are live but weak. seed_count 10 -> 11 reorders zero of 15 corpus queries - it moves best_depth for 7 of them but never enough to change what a user sees.

| setting | default | probe | queries reordered |
|---|---|---|---|
| ranking.b | 0.75 | 0.375 | 13 |
| ranking.k1 | 1.2 | 0.6 | 9 |
| graph.max_depth | 2 | 3 | 2 |
| graph.seed_count | 10 | 20 | 2 |

## Traps already hit, so you do not repeat them

- A test asserting the *absence* of something is worthless without a control
  proving the check can fire.
- A falsifier must damage the thing under test **without narrowing what the test
  looks at**. Patching shared machinery can remove the broken case from the
  test's own iteration set.
- Action registration is an import side effect, so registry tests must import
  the modules themselves or they measure collection order. **Walk the package;
  never hand-list modules.** A hand-written list drifted once already — M1.4
  added `drf.retrieval.neural` and nothing noticed until a file was run alone.
- **Falsifiers cannot catch order-dependent tests.** Each one runs a single
  test in a subprocess that inherits the same import state, so a test passing
  only because another module was collected first looks healthy. Run
  `./tools/check_isolation.sh` — it is the only thing that finds these, and a
  full `pytest tests/` run cannot, by definition.
- Rank fusion (RRF) is the ecosystem's default advice and appears throughout the
  recovered playbooks. It is rejected here: it grants authority to the advisory
  side.

## Scope

Milestone 1 carries no relevance judgements and makes NO claim about retrieval quality. It proves reproducibility and subordination. Recall, nDCG and MRR require labels and belong to milestone 2. An unqualified 'all metrics 1.0' would be exactly the drift this framework exists to prevent.
