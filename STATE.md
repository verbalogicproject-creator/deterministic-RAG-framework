# Project State — resume here

**Last checkpoint:** M2.2 **closed.** The graph re-scope was built, measured, and **rejected on its own evidence** — implemented, tested, default OFF. The nuisance screen now has a positive control, so `CLEAN` finally means something. 260 tests, isolation clean (2026-08-10). Released `v0.0.9`.
**Next:** expand the query set beyond 23 (the graph decision was underpowered at n=1 labelled query), then M2.3 BGE re-embedding.
**Full build plan:** `~/.claude/plans/plan-step-1-out-crystalline-firefly.md` — read this first, it has the complete M1.0–M1.8 sequence, architecture, and verification steps.

---

## Goal

A Deterministic RAG framework whose retrieval path contains no model. A neural layer may be added under a machine-checked guarantee that **it can never change an authoritative result — only append below it.**

Field anchors (both read in full, both directly relevant):
- **SAT-Graph RAG** (arXiv 2510.06002) — the determinism boundary as a typed action contract. *"Once a formal id is acquired, all subsequent actions that operate on it are guaranteed to be deterministic."* Has **no evaluation** — states this as a limitation.
- **ReproRAG** (arXiv 2509.18869) — metric suite (Exact Match Rate, Jaccard, Kendall's Tau, RBO, Overlap Coefficient, Score Stability) and a measured *hierarchy of uncertainty*. Key result: **embedding-model choice is the dominant non-determinism source** (cross-model Overlap 0.43–0.54, Kendall's Tau 0.32–0.38, p > 0.05), while ANN index algorithms scored a perfect 1.000 — falsifying the common assumption.
- **Claim discipline:** never say "vector RAG is non-reproducible." A fixed pipeline *is* perfectly reproducible (8/8 configs, L2 = 0.00). The defensible claim: *reproducible conditional on a model choice that is itself arbitrary and drives ~50% of result variance.*

---

## Verified ground truth (expensive to re-establish — trust this, not the old docs)

The recovered source project's documentation is **systematically unreliable**. Verified by direct inspection:

| Item | Reality |
|---|---|
| `claude-cookbook-kg.db` | **266 nodes / 605 edges — claim TRUE.** 228 real 384-d MiniLM embeddings. **This is our corpus.** |
| Its data quality | 48 duplicate `(from,to,type)` edge groups, 4 orphan edges, 5 isolated nodes |
| `python_apps_kg.db` | **CORRUPT** — 12,154 rows = exactly 2× 6,077 distinct. Builder appends without DELETE. Do not use. |
| `unified-kg.db` | 2,086 nodes (not the claimed 2,057); canonical copy has only **17/2,086 embeddings**. A 99%-populated copy exists at `~/Downloads/llm_web_brain/cost optimization/unified-kg.db` that no tool points at. |
| Old retrieval engine | `python_apps_hybrid_query.py:212` = `return 0.5  # Placeholder`. **The α=0.40 embedding term never contributed anything.** All published recall numbers describe `0.85·BM25 + 0.15·same-file-popularity`. |
| Its BM25 | Not BM25 — no `b`, no `avgdl`, no length normalisation. |
| Its determinism | Unordered `LIMIT 100` (`:342`); no tiebreak on any sort (`:304`, `:317`); hard `[:15]` boundary cascade (`:307`). |
| KG tools | **The strongest part of the old project** — 9/15 FUNCTIONAL, 12 run today from `~/Downloads/claude-cookbook-kg3/`. |
| `config_manager.py` | Best code in the old repo. Port `diff`/`_diff_dicts` (lines 314–397) as-is. |

## BGE endpoint — measured, not assumed

`http://<bge-host>:8080  (private LAN address, redacted)` — llama.cpp serving **BAAI/bge-large-en-v1.5 Q4_0 GGUF** from an Android path.

- `POST /v1/embeddings` `{"input": ..., "model":"bge"}` → **dim 1024, already L2-normalised** (cosine = dot product, no numpy needed), ~0.1–0.3 s
- **Measured deterministic:** 5 sequential identical requests → byte-identical; single-input vs in-batch → byte-identical
- Therefore labelled `deterministic` in `spec/actions.json` **with the evidence recorded inline**, and the replay check stays armed — this is a property of *this server build*, not of remote embedders generally
- **Still `advisory`.** Determinism and authority are independent axes; this is the clearest case proving it
- ⚠️ 1024-d BGE ≠ 384-d MiniLM. Different spaces, never comparable. Using BGE requires `drf embed --provider bge` re-embedding all 266 nodes (~30 s at measured latency)

---

## Architecture (settled)

```
Stage 1  AUTHORITATIVE : BM25 + graph traversal → total order D
Stage 2  ADVISORY      : neural proposes candidates NOT in D
Stage 3  MERGE         : append-only. order(D) preserved exactly.
```

- **No α term. Deleted, not zeroed** — any weight implies authority.
- Sort key: `(-s1_q, -bm25_q, -matched_terms, best_depth, doc_len, node_id)`. Component 6 is the content-addressed sha256 PK → **injective** → strict total order → output independent of input permutation and of sort stability. Exact ties within D **cannot occur**, so neural tie-breaking is unreachable and was removed.
- Candidates = `⋃ postings(t)` — the exact support of BM25. No LIMIT, no ordering dependency.
- All ranking values are `int` (fixed-point, `QUANTUM_EXP = 9`). Floats structurally cannot reach a comparison.

**Enforcement, not documentation:** runtime postcondition in `merge()` on every query; `Advisory[T]` box with a one-module allowlist (`drf.retrieval.merge`); static AST check that Stage 1 cannot import neural; `@action` replay check making the determinism label load-bearing.

### Governing design rule: commutativity beats pinned order

When an operation can be made reproducible either by **pinning an input order** or by **being order-independent by construction**, choose the latter. A pinned order is deterministic only as long as the pinning survives — its correctness lives in a `sorted()` call upstream, so a refactor that reorders inputs silently changes the output. An order-independent operation has no such dependency to preserve. *"There is no nondeterminism to pin"* beats *"we pinned the nondeterminism."*

Review lens for every new operation: **would this produce the same output if the inputs arrived in a different order?** If the answer is "only because we sorted first," look for a formulation where it is "yes, always."

Three existing decisions are the same principle in different clothes:

| Decision | Where | Why it is order-independent |
|---|---|---|
| Conflict-free union of duplicate edge variants | `ingest/normalize.py:collapse_edge_group` | Union over a non-conflicting key set is commutative. Raises on genuine conflict rather than picking a winner. Measured: 0 conflicts in 48 groups. |
| `math.fsum` for score accumulation | `fixed.py:exact_sum` | A correctly-rounded sum of a multiset is *unique*, so the result does not depend on summation order. Stronger than the portability argument in that docstring. |
| Injective 6-tuple sort key | `retrieval/stage1.py` (M1.3) | Ending the key with the content-addressed node id makes it injective → strict total order → `sorted()` agrees on every input permutation, and sort stability becomes irrelevant. |

### Milestone gate: harvest, then audit

**Accumulated knowledge from step *n* changes what step *n+1* is worth.** Run both halves before writing any code for a milestone:

- **Harvest** *(constructive, needs judgement)* — what did step *n* measure that resolves an open question in step *n+1*? M1.1 → M1.2 produced six: `avgdl`=18.9 with a 4–61 spread (so `b` is load-bearing), 0 empty descriptions (so `avgdl` needs no guard), `source_ref` 71% duplicate tokens (excluded from the document), `type` crushed by IDF (excluded), the commutativity rule (killed a checkpoint), and the DDL-grep control test (generalised into the falsifier registry).
- **Audit** *(destructive, mechanised)* — did step *n*'s new rule make a pending checkpoint vacuous? This half does **not** need judgement, so it must not depend on anyone remembering: see `spec/invariants.json`.

---

## Built so far (M1.0 ✅)

```
drf/version.py    PARSER/RANKER/ID_SCHEMA versions (participate in hashing)
drf/hashing.py    canonical_json, sha256_value, content_id, node_id, edge_id
drf/fixed.py      exact_sum (fsum), quantize/unquantize, qmul
drf/contract.py   @action, Justification, ActionOutput/ActionResult,
                  Advisory[T], ACTIONS registry, replay check, strict_replay, Trace
spec/actions.json 10 actions with determinism × authority + evidence
tests/test_contract.py   23 tests, all green
```

Proven at this checkpoint: canonical hash stable across `PYTHONHASHSEED`; content IDs are pure functions of content; edge IDs dedupe structurally; mislabelled deterministic action raises on second call; `strict_replay` catches on first call; advisory unwrap refused outside allowlist; probabilistic actions can never be authoritative; remote determinism claims must cite a measurement.

## Built so far (M1.1 ✅)

```
drf/store.py             schema + the only sanctioned reads/writes
drf/ingest/source_kg.py  read source KG, content-ordered, no rowid reliance
drf/ingest/normalize.py  content-addressed mapping, drop, collapse
drf/ingest/manifest.py   content vs provenance split, reconcile()
drf/ingest/build.py      @action("ingest.build_index")
tools/drf                build | verify | inspect
tests/test_ingest.py     30 tests
```

**Index built from `claude-cookbook-kg.db`:**

```
content_hash  27679065a72e473dd57420dbc866adf740e59f4e56eb4262643748fabccd4bbe
nodes 266   edges 553   embeddings 228 (all 384-d, one model)
```

Edge reconciliation, asserted on **every build** (not only in tests):
`605 read = 553 written + 48 collapsed variants + 4 dropped`.

⚠️ **Correction to an earlier figure.** The number 557 appears in some notes as the expected edge count. That was the count of distinct `(from,to,type)` triples *including* the 4 orphan triples; subtracting orphans separately double-counted them. **553 is correct.**

Proven at this checkpoint:
- Two builds to different paths → identical `content_hash`. Enforced by the **contract**, not just a test: `inputs=("source_path","corpus")` excludes the output path, so both calls share an `inputs_hash` and the replay check compares results itself.
- `content_hash` stable across `PYTHONHASHSEED` ∈ {0, 1, 12345} in subprocesses.
- Every manifest count confirmed by `SELECT count(*)`; every expected value re-derived from the source, never a literal.
- 48 duplicate groups collapse structurally (edge id is the PK); the 4 payload-divergent groups **keep** their `confidence`/`reasoning` provenance.
- 4 dangling edges named in `manifest.dropped`, and proven **not** silently repaired despite 3 having plausible near-matches.
- 5 isolated nodes kept deliberately — a node with no edges is still a valid BM25 hit.
- DDL: **0** `AUTOINCREMENT`, **0** `CURRENT_TIMESTAMP`, all 4 tables `WITHOUT ROWID`. Control test proves the check can fail (source schema has 6 `AUTOINCREMENT`).
- `drf verify` exits 1 on a tampered manifest.

**Collapse rule decision (M1.1):** conflict-free union — merge duplicate variants key by key, **raise** if two assign different values to one key. Measured: 44 groups byte-identical, 4 divergent but conflict-free, **0 genuinely conflicting**. Chosen over "richest wins" / "strip metadata" / "blind union" / "always refuse" because it is the only one that is *commutative* — see the governing design rule above.

## M1.2.0 ✅ — falsifier registry

`spec/invariants.json` + `tests/conftest.py` + `tests/test_falsifiers.py`. Each checkpoint invariant names a **falsifier**: a mutation under which the named test *must* fail. Run in subprocesses; a surviving test is reported as vacuous. Three exemptions are recorded with reasons (properties already guarded at runtime by `build.py` or the contract, where a falsifier would prove the guard, not the test).

**It caught two real problems on its first run:**

1. **`test_dangling_edges_were_not_silently_repaired` was broken.** It derived the dropped set from `manifest.dropped`. Under repair, nothing is dropped → empty set → disjoint from everything → green. It could never detect the thing it was named for. Now derives the dangling pairs from the **source**. (Verified safe: all 4 dangling `(from,type)` pairs have 0 valid edges.)
2. **The hash-seed falsifier was a no-op.** The corpus has exactly 1 embedding model and 1 dimension, so `list(set)` == `sorted(set)`. Retargeted at node types (25 values). Finding about the corpus: **no multi-element set currently reaches `content_hash`**, so that test guards a bug class this data cannot exhibit unaided.

## Prior art reviewed: `~/Downloads/synthesis-rules/python_apps_*`

| File | Verdict |
|---|---|
| `python_apps_hybrid_query.py` (449) | Negative reference — see below |
| `python_apps_kg_builder.py` (317) | FTS5 pattern, **but the triggers are buggy** |
| `python_apps_kg.db` (4.7 MB) | Corrupt (2× duplication). Unusable. |
| `python_apps_embeddings.py` (605) | 384-d MiniLM path; not used in M1.2 |

**⚠️ Plan correction.** The build plan says lift `python_apps_kg_builder.py:68-93` (FTS5 external-content + 3 sync triggers) **verbatim**. Do not. Measured: for an external-content FTS5 table, `DELETE FROM fts WHERE rowid=…` and `UPDATE fts SET …` do **not** remove old terms from the term index. After an UPDATE the stale term still matches, and `INSERT INTO fts(fts) VALUES('integrity-check')` **passes**, so the corruption is silent. The documented pattern is `INSERT INTO fts(fts, rowid, …) VALUES('delete', old.rowid, …)` followed by a fresh insert. Affects M1.3/M3, not M1.2.

**Old engine scorer, exactly:** `idf * (tf*k1)/(tf+k1)` with `k1=2.5`, no `b`, no `avgdl`.
- Measured vs proper BM25: **27.2%** discordant pairs @10, top-1 changed 9/15, mean top-1 length 28.1 vs 15.9 tokens.
- I predicted `k1=2.5` would compound the length bias. **It does not** — 27.2% vs 26.2% at `k1=1.2`. Reason: **tf==1 for 82.7%** of (doc, term) pairs in a ~19-token corpus, so `k1` (which controls tf saturation) has almost nothing to act on. `b` is the entire defect; `k1` is a safe tunable here.
- The numerator `tf*k1` instead of `tf*(k1+1)` is a monotone rescale — **not** a ranking defect. Do not list it as one.

**Third truncation found:** `python_apps_hybrid_query.py:339` is `' OR '.join(keywords[:10])` — queries silently truncated to their first 10 terms, on top of the unordered `LIMIT 100` (`:347`) and the `[:15]` boundary (`:307`). M1.2's `candidates = ⋃ postings(t)` must carry no such cap; `len(candidates) == len(posting_union)` is exactly that guard.

## M1.2 ✅ — `tokenize.py`, `bm25.py`, `lexical.py`

```
drf/retrieval/tokenize.py   one tokenizer, ASCII by construction
drf/retrieval/bm25.py       Okapi BM25, pure arithmetic, no contract import
drf/retrieval/lexical.py    candidates + scoring actions, index build tables
spec/ranking.json           k1, b, quantisation point, document definition
tests/test_retrieval.py     24 tests
```

Index now carries `doc_stats`, `terms`, `postings` (7 tables, all `WITHOUT ROWID`):

```
content_hash  90ab5db969588b5a2a41beddce996cd3bf25d27b28d9791f984416d8b33cf72a
266 docs   1,303 terms   4,185 postings   total_len 5,019   avgdl 18.8684
```

`PARSER_VERSION` **1.0.0 → 1.1.0** — the build now indexes text, so the hash necessarily moved from `27679065…`. The old value in earlier commits is correct *for an index without postings*.

**98 tests green.** Proven: exact-integer match against an independently derived 5-doc reference; padded long document loses to a short exact match **with a control proving `b=0` reverses it**; `len(candidates) == len(posting_union)`; all scores `int`; scoring invariant across all 6 permutations of posting order; the two quantisation points genuinely differ (so the spec's choice is falsifiable); index and query paths share one tokenizer; `df` agrees with the postings it summarises; OOV query returns empty rather than raising.

**9 falsifiers now registered**, 4 of them new and 2 reproducing defects that actually shipped in the old engine (`b=0`, and the ten-term query cap from `hybrid_query.py:339`).

### Gaps found and fixed while implementing

| Gap | Fix |
|---|---|
| Replay-log key had no index identity — two indexes, same query → spurious `DeterminismViolation` | `index_hash` is a declared input of `lexical.candidates` / `lexical.bm25_score`; `k1`/`b` too |
| `test_every_table_is_without_rowid` asserted a literal `4` | Derives from `store.TABLES`, itself parsed from `SCHEMA` |
| `spec/actions.json` said scores are "summed in sorted-term order" | Corrected — the sort is a convenience, not load-bearing |
| A silent tokenizer hazard: `\w` depends on the interpreter's Unicode DB version | `[a-z0-9]+`; measured cost is 2 symbol chars (`×`, `→`) in 5/266 docs |
| `avgdl` stored as a float would not round-trip | Store `total_len` and `n` as ints; derive `avgdl` at use time |
| A term with postings but no `df` row would score silently | Raises `ValueError` — an inconsistent index must not produce a plausible score |

## M1.3 ✅ — `graph.py`, `stage1.py`

```
drf/retrieval/graph.py    bidirectional bounded BFS -> {node_id: best_depth}
drf/retrieval/stage1.py   the strict total order D
tests/test_stage1.py      16 tests
```

**119 tests green. 11 falsifiers registered.**

### Sort key — two deliberate plan deviations

```
(-bm25_q, -matched_terms, best_depth, doc_len, node_id)
```

**No `s1_q`.** The plan had `(-s1_q, -bm25_q, …)` blending lexical and graph signal. Any blend needs a weight, and **M1 carries no relevance labels** — a weight chosen now could be asserted but never validated. Lexicographic composition is parameter-free and does real work: graph proximity orders exactly the documents BM25 cannot distinguish. Weighted combination is an **M2** question, to be settled when labels exist. This also keeps structural distance from RRF, which can invert a strictly better lexical match; here that is impossible by construction (`test_graph_never_overrides_a_strictly_better_lexical_match`).

**D contains lexical candidates only.** Expansion supplies `best_depth`; it does not inject documents containing no query term. Admitting them would be a relevance claim M1 cannot justify. Recorded as an M2 decision rather than taken silently.

### The audit fired — and passed

`len(set(sort_keys)) == len(D)` was flagged as *probably vacuous*, since injectivity is guaranteed once the key ends in a content-addressed PK. **Measured before writing the test:** dropping `node_id` collides **66 candidates across 7 of 15 queries** (`tool use`: 86 candidates → 29 distinct keys). The test can fail, so it is worth having. Falsifier registered as `strict_total_order`.

Proven: injectivity on every query; **50 input shuffles → identical output**; no tie at any truncation boundary (killing the `[:15]` cascade at the root); `seed_count` 10→11 and `max_depth` 1→3 each change output for ≥1 query; expansion is independent of seed order; terminates despite directed cycles; depth is monotone in `max_depth`; graph depth *does* resolve ties BM25 leaves undefined.

### Gaps found and fixed while implementing

| Gap | Fix |
|---|---|
| Registry tests were **order-dependent** — passed in the full suite, failed running `test_ingest.py` alone, because action registration is an import side effect | `_import_all_action_modules()` populates the registry from within the test |
| A malformed `spec/*.json` aborted collection with an opaque error (hit **twice**) | `test_every_spec_file_is_valid_json` in a module that doesn't import it, so it reports which file and where |

## M1.4 ✅ — `neural.py`, `providers/*`, `merge.py` — **the centerpiece**

```
drf/retrieval/neural.py                    advisory actions, providers untrusted
drf/retrieval/merge.py                     append-only; the ONLY unwrap site
drf/retrieval/providers/{base,null,stored_vectors}.py
tools/drf query --neural off|stored --explain
tests/test_merge.py                        18 tests
```

**141 tests green. 13 falsifiers.** The guarantee is now a runtime fact, not a design intention:

```
merged[:len(D)] == D,  elementwise, in order, always
```

Proven against **eight providers** — null, stored-vectors, and six hostile: adversarial (proposes D reversed, trying to duplicate/promote), crashing, hanging, flooding (10,000 ids), lying (nonexistent ids), junk (non-strings). Every one leaves the prefix byte-identical. `discordant_pairs == 0` as an exact integer across all providers × all queries.

Verified from the CLI too — authoritative prefix identical under `--neural off` and `--neural stored`: `prompt caching` 22 rows, `tool use` 40, `streaming` 3.

**Determinism × authority, demonstrated.** `StoredVectorProvider` runs no model and touches no network — 384-d MiniLM vectors frozen in the index, already L2-normalised so cosine is a dot product, no numpy. It is fully `deterministic` and strictly `advisory`. Determinism is not a licence.

**Vacuity guarded both ways.** `test_stored_vectors_actually_proposes_something` is the control: without it, every subordination test could pass because nothing ever reached the tail. Measured — the tail is populated (3 advisory results at k=25 for `prompt caching`).

### The audit, again — and it needed care

`discordant_pairs == 0` was flagged as possibly unfalsifiable. It is *not*, but the obvious falsifier would have been wrong: interleaving alone merely trips `merge()`'s runtime postcondition, which proves the **postcondition** fires — a different claim. The registered falsifier therefore **neuters the guard first**, so the bad merge returns quietly and the test assertion is the only thing left that can notice. Second falsifier widens `ADVISORY_CONSUMERS`, the way the guarantee would actually erode in practice: one locally-reasonable module at a time.

### Deliberately unregistered actions, with reasons

| Action | Why |
|---|---|
| `merge.append_advisory` | `merge()` is deterministic given (D, proposals), but proposals arrive **boxed** and the box may only open inside merge. Declaring `inputs` would require unwrapping outside the allowlist or hashing something that isn't the real input. The runtime postcondition is the stronger check — it verifies the *property*, not the repeatability. |
| `neural.encode_query_remote` | `RemoteHTTPProvider` cut from M1 by the plan. BGE is 1024-d, corpus vectors are 384-d MiniLM — incomparable spaces, so it needs the whole corpus re-embedded. M2. |
| `store.load_manifest` | No second reader to keep honest yet. |

## M1.6 ✅ — `bench/*`, `queries/milestone1.jsonl`

```
drf/bench/metrics.py    ReproRAG metrics: integers to assert, floats to read
drf/bench/repro.py      the matrix + the chaos control + sensitivity
queries/milestone1.jsonl  23 queries: 15 corpus + 8 edge cases
tools/drf bench repro|chaos|sensitivity|all
tests/test_bench.py     13 tests
```

**185 tests green. 18 falsifiers.** Full matrix — **28 cells, 1 distinct digest**:

```
5 in-process x 3 subprocess x 3 PYTHONHASHSEED x 2 independent builds
mismatched_positions 0   discordant_pairs 0   length_delta 0   symmetric_difference 0
```

### The chaos control — why the perfect scores mean anything

Every metric scores 1.0 here, which is exactly why the numbers alone are *not* evidence: a harness that compared nothing would report the same. So `bench chaos` runs the identical measurement against **the defect this framework replaced** — rank by score alone, no tiebreak, unordered candidates (`python_apps_hybrid_query.py:304`). Not synthetic noise; the previous implementation.

| metric | real | chaos | separates? |
|---|---|---|---|
| `distinct_digests` | **1** | **5** | ✅ |
| `mismatched_positions` | **0** | **551** | ✅ |
| `discordant_pairs` | **0** | **3,931** | ✅ |
| exact_match_rate | 1.0000 | 0.6196 | ✅ |
| kendall_tau | 1.0000 | **0.9761** | barely |
| rbo | 1.0000 | **0.9942** | barely |
| jaccard | 1.0000 | **1.0000** | ❌ blind |
| overlap_coefficient | 1.0000 | **1.0000** | ❌ blind |

**Two findings, both pinned as tests:**

1. **Set metrics are blind to ordering non-determinism.** Jaccard reports 1.0000 for a pipeline returning five different orderings in five runs. Citing it as reproducibility evidence would be citing nothing.
2. **Kendall's Tau 0.976 and RBO 0.994 describe a provably non-deterministic pipeline.** Rounded for a report they read as 0.98 and 0.99 — indistinguishable from correct. This is the *measured* justification for the project's "assert exact integers, never floats" rule, which until now was asserted on principle.

### Gaps found and fixed while implementing

| Gap | Fix |
|---|---|
| **My RBO was broken** — un-normalised RBO is bounded by `1 - p^k`, so it scored **0.716 for byte-identical lists** against 0.712 for the broken control. A metric that cannot separate perfect from broken is worse than absent. | Normalise by `1 - p^k` |
| My own test then asserted `rbo == 1.0` — an **exact float assertion**, the very thing the project forbids. It failed in *both* directions (`1.0000000000000002` and `0.9999999999999998`) for identical inputs. | Assert the integer surface; `round(rbo, 9)` for the float |

Sensitivity (23 queries): `ranking.b` reorders 13, `ranking.k1` 9, `graph.max_depth` 2, `graph.seed_count` 2. Lexical parameters dominate; graph parameters are live but weak, consistent with M1.5.

## M1.7 ✅ — `docs/render.py`, templates

```
drf/docs/render.py                  context from spec/ + a built index
drf/docs/templates/{peer,agent,operator,plain}.md.tmpl
spec/benchmarks.json                every measurement beside its producer command
docs/{peer,agent,operator,plain}.md GENERATED, and committed
tools/drf docs build --index index.db
tests/test_docs.py                  11 tests
```

**200 tests green. 20 falsifiers.**

`Template.substitute`, never `safe_substitute` — a placeholder the context lacks raises `KeyError` rather than leaving `$whatever` sitting in the prose looking like it belongs.

### The hand-edit guarantee, demonstrated not asserted

```
$ printf 'This system is 40%% more accurate than alternatives.\n' >> docs/plain.md
$ pytest tests/test_docs.py::test_committed_docs_match_a_fresh_render
E  + This system is 40% more accurate than alternatives.
FAILED
```

That is the exact failure mode the source project shipped (`80% vs 60% accuracy`, no evaluation anywhere). It now cannot survive a test run.

**`.gitignore` reversed for `docs/*.md`.** They were ignored as "generated output". But an ignored file cannot be checked, which would leave the guarantee unenforceable. They are committed, and the test re-renders and compares.

### The audit paid off again

"A hand edit fails a test" is vacuous **if the test regenerates before comparing** — it would compare a fresh render against itself and pass forever. `test_committed_docs_match_a_fresh_render` therefore only reads. The falsifier mutates the *renderer* rather than editing a file, producing the same divergence without leaving a dirty working tree.

It also fired for real during development: adding entries to `spec/invariants.json` changed `spec_sha` and `invariant_count`, so the committed docs no longer matched. Drift detected within seconds of being introduced.

### Four audiences

| doc | for | contains |
|---|---|---|
| `peer.md` | technical reader | architecture, the two axes, all measurements with producers, field anchors, scope limits first |
| `agent.md` | a future Claude session | verified facts not to re-measure, what raises and why, non-negotiable rules, **traps already hit** |
| `operator.md` | you | every command, what each proves, how to check the prefix guarantee yourself |
| `plain.md` | non-technical reader | no jargon (asserted: no "BM25", "idf", "injective", "postings", "sha256") |

## M1.8 ✅ — freeze

```
drf/freeze.py           spec_sha + manifest_hash + bench_digest
spec/frozen.json        the release record, committed
drf/docs/templates/readme.md.tmpl   README is a fifth generated audience
tools/drf freeze write|verify
tests/test_freeze.py    8 tests
```

```
release        0.0.1
spec_sha       77302e076be0b25173227381b0b7426e35180abd11ddd8301ca325e273e96b97
manifest_hash  90ab5db969588b5a2a41beddce996cd3bf25d27b28d9791f984416d8b33cf72a
bench_digest   5053fe6600b21201756c5046995c063ffa4566c97002692c8d5566ea112fec88
queries        23
```

**A git tag pins the code; it does not pin the answers.** A result depends on three things, and a commit pins one. `drf freeze verify` rebuilds from source and checks all three.

**The bench digest is the load-bearing component.** Spec and index hashes prove the *inputs* are unchanged; only the digest proves the *outputs* are. A refactor that alters ranking while leaving both inputs untouched would pass an input-hash-only check — `test_bench_digest_is_the_component_that_catches_a_ranking_change` constructs exactly that case and requires it caught.

**README is now generated** (fifth audience, rendered to repo root). The landing page is the most likely place for a stale number to sit unchallenged.

### Gaps found and fixed while implementing

| Gap | Fix |
|---|---|
| **Two implementations of `spec_sha` disagreed.** `render.py` globbed every `spec/*.json` including `frozen.json`; `freeze.py` must exclude it, because that file *contains* the hash. | `render.py` delegates to `freeze.spec_sha()`. One concept, one definition. |
| `verify()` stopping at the first mismatch would hide later ones | Returns all differences; a test asserts all four are reported |

## Release process

```bash
./tools/drf build --source <src> --out index.db     # rebuild
./tools/drf docs build --index index.db             # regenerate all 5 docs
./tools/drf freeze write --index index.db           # record the three hashes
python3 -m pytest tests/ -q                         # 208 tests
./tools/check_isolation.sh                          # order-dependence guard
git tag v<version> && git push --tags
```

Bump `RELEASE_VERSION` in `drf/version.py` **before** `freeze write` — a test asserts the frozen release matches the code.

## M2.0 ✅ — the quality harness, built before the labels

`drf/bench/quality.py`, `controls.py`, `labels.py`, `evaluate.py`, `spec/evaluation.json`, `tests/test_quality.py` (27 tests), `drf eval`.

**Why the harness comes before the labels.** Validating a harness on labelled data makes "is the harness right?" and "is the system good?" one experiment, so a disappointing answer names no culprit. Three things can be checked with no ground truth, and all three are implemented:

1. **Hand-computed reference** — nDCG derived on paper for a 5-document example (`0.8354478`; reversed `0.5116930`), compared against literals. The M1.2 BM25 fixture again.
2. **Properties true of any label set** — the oracle sorts the system's *own* candidates by grade, so `oracle.ndcg >= system.ndcg` is a theorem, not a finding. A reversal is a permutation, so it cannot change a retrieved set. Both run on deliberately nonsensical synthetic labels and catch real arithmetic bugs.
3. **The structural bound** — `advisory_horizon` needs no labels whatsoever.

**The M1 assertion rule, adapted rather than abandoned.** nDCG and recall are ratios; they cannot be integers. Kept the shape: `ranks_of_relevant` is a tuple of 1-based integer ranks and is the M2 analogue of `discordant_pairs`. Recall reports the same value for a ranking and its exact reversal (test asserts this blindness *exists*); the rank tuple reports `(1, 2)` against `(4, 5)`.

**Controls** — `oracle` (ceiling; a low oracle means the documents were never retrieved, so reordering could not have helped), `reverse` (same set, inverted order — any metric scoring it equal to the system is order-blind), `shuffle_0..4`, and **`id_order`**: sort by sha256 node id. Perfectly deterministic, perfectly reproducible, entirely relevance-blind — it would score 1.0 on every metric in `bench/metrics.py`. It is the counter-example to the reading this framework most invites, that determinism is a quality property. `min_ndcg_margin = 0.05` was declared in `spec/evaluation.json` on 2026-08-08, before any label existed.

### Measured: the advisory horizon (needs no labels, ran today)

`|D|` per query across the 23-query set: **min 0, max 147**. Prefixes identical **23/23** with the provider on and off.

| `|D|` | queries | depths the advisory layer can reach (of 1/5/10/20) |
|---|---|---|
| 0 | 3 (all OOV) | all — but it proposes nothing (see below) |
| 1–4 | 4 | 5, 10, 20 |
| 5–7 | 3 | 10, 20 |
| 10–19 | 6 | 20 |
| ≥ 20 | 7 | **none** |

**Finding: the advisory layer's reach is inverse to lexical success.** Where stage 1 returned ≥ 20 documents it is structurally silent at every evaluated depth; where it returned one or two, it can act from depth 5 down. The neural layer can only speak where lexical retrieval did badly, and is provably mute where it did well. That is what append-only subordination *means*, now stated as a measurement rather than a design intention.

**Second finding: a reachable horizon is necessary but not sufficient.** The three OOV queries have `|D| = 0`, so by horizon alone the advisory layer could occupy every position — and it proposes nothing, because anchor-mode search takes its anchors *from D*. **Anchor starvation is a second bound, independent of the horizon.** A recall figure on those queries would measure starvation, not the provider. This is the M1 scope limit observed rather than asserted, and the fix if wanted is a Stage 1 fix (character n-grams), never a neural one.

### Two bugs found and fixed while implementing

- **`test_shallow_depths_are_structurally_unreachable` asserted the wrong thing.** It assumed a corpus-wide bound; `|D|` is per query and 7 of 23 fall below depth 5. Replaced with the real structure above — the failure produced a better finding than the test would have.
- **`test_no_quality_figure_is_published` was a substring scan and matched the spec's own worked example** (a sentence explaining why `nDCG@10 = 0.71` is not evidence). Now walks the JSON for a **metric name bound to a number** — prose is where explanation lives, a key bound to a number is where a claim lives, and only the second is a publication. `min_ndcg_margin` deliberately does not match: it is a declared threshold, not a result.

### Freeze discrimination worth noting

`spec_sha` moved; `manifest_hash` and `bench_digest` did **not**. Correct, and a useful demonstration: M2.0 added an instrument and changed nothing about what the system returns.

### Scope, stated

**No quality figure exists for this system and none appears in this repository.** `drf eval quality` prints that fact rather than a zero. A test walks every spec file to keep it true.

---

## M2.1 (planning record) — how the worksheet was stratified

`tools/make_labelling_worksheet.py`, `tools/labels_collect.py`, `queries/LABELLING.md`, `queries/labels.worksheet.{jsonl,md}`.

**The selection is the experiment.** Fifty arbitrary judgements settle nothing. The strata come from a measurement — every `affects_ranking` setting probed across all 23 queries:

| setting | queries reordered | which |
|---|---|---|
| `ranking.b` | 13 | q01 q02 q03 q04 q05 q09 q10 q11 e04 e05 e06 e07 e08 |
| `ranking.k1` | 9 | q01 q02 q03 q05 q09 q11 e04 e05 e06 |
| `graph.max_depth` | **2** | q02, e06 |
| `graph.seed_count` | **2** | q02, e06 |

| stratum | judgements | queries | settles |
|---|---|---|---|
| **A_advisory** | 49 | 7 (`|D| ≤ 7`) | Does the neural layer find what lexical search missed? |
| **B_length_normalisation** | 40 | 4 (move on `b`, **not** `k1`) | Does `b` improve relevance, or only change it? |
| **C_graph** | 20 | 1 | ⚠️ cannot be settled |

Stratum A is complete on its own and is almost exactly the ~50 the M2 plan budgeted. B isolates length normalisation from term saturation by taking only the four queries that move under `b` and not under `k1` — the other nine move under both and cannot separate the effects.

### ⚠️ Finding that changes the roadmap: the graph decision is underpowered

Only `q02` and `e06` reorder under any graph setting, and `e06` is a synthetic 13-term edge case. **"Does the graph layer earn its place?" has a real sample size of one.** Labelling cannot settle it — it needs **more queries**, chosen to stress graph structure (lexically thin, well connected). That work moves to **M2.2**. Stratum C is retained for the qualitative read only.

### Finding: the strata are disjoint, and that is a property of the corpus

Low-`|D|` queries are the only ones the advisory layer can reach — and they are inert to *every* ranking parameter, because a handful of candidates leaves a weight nothing to reorder. High-`|D|` queries are parameter-sensitive and advisory-mute. **No single query serves both purposes here.**

### Pooling

Candidates come from D **and** from the advisory provider. A pool drawn only from D could never contain a document lexical retrieval missed, so recall would be measured against a denominator that excluded the exact thing the advisory layer exists to find. Stratum A pools the top **3** advisory proposals per query — shallow, since the provider returns up to 100. Unjudged documents count as grade 0, so **a shallow pool silently flatters recall**; if stratum A shows the advisory layer contributing little, check pool depth before concluding anything. One edit to `pool_advisory`.

### Guardrails

Ungraded rows are written as `"grade": null`, which is **invalid** to the parser; `labels_collect.py` exits 1 rather than writing a partial file. A template pre-filled with `0` would turn a forgotten row into a confident "irrelevant", and a partial file shrinks the recall denominator — which *raises* the measured score.

### Smoke-tested end to end

Ran the full chain on throwaway grades (assigned by rank position, so not judgements and not committed). Three things confirmed: `reverse` scores **recall 1.0000 at depth 10, identical to the system** — the ordering blindness visible in live output; nDCG separates them 0.8830 against 0.5887; and `shuffle_3` scored 0.7826 against `shuffle_4`'s 0.6021, so **a single shuffle seed would have moved the reported margin by ~0.18**. That is the measured justification for using five.

### Staleness guard

`STRATA` hand-lists query ids, which is the same rot as the order-dependent registry bug from M1. The test does not check the list exists — it re-derives `|D|` and asserts stratum A's queries **actually have** the small horizon that justified selecting them. Assert the reason, not the artefact.

---

## M2.1 ✅ — stratum A graded and measured. **Result: FAIL.**

`queries/labels.jsonl`, 49 judgements over 7 queries, `labels_hash 33d8e1beb4302bbc…`.

> ⚠️ **Provenance.** The labels are **model-generated (Claude Opus 5)**, graded **blind to rank** — candidates were presented in sha256 order with names and descriptions only, no D-rank, no origin. Blind grading removes position bias. It does **not** remove authorship bias: the annotator also wrote the retrieval system, so relevance calls may correlate with what this ranking already surfaces. The user asked for this after I flagged the circularity; it is recorded, not hidden.

### The measurement

| depth | system nDCG | best blind control | its nDCG | oracle | margin | required | per-query |
|---|---|---|---|---|---|---|---|
| 1 | 0.8367 | shuffle_3 | 0.7959 | 1.0000 | +0.0408 | 0.05 | 2W 1L 4T |
| 5 | 0.8787 | id_order | 0.8412 | 0.9811 | +0.0376 | 0.05 | 3W 3L 1T |
| 10 | 0.9064 | id_order | 0.8761 | 0.9787 | +0.0303 | 0.05 | **3W 3L 1T** |
| 20 | 0.9064 | id_order | 0.8761 | 0.9787 | +0.0303 | 0.05 | **3W 3L 1T** |

**The margin was NOT revised.** It was fixed at 0.05 on 2026-08-08 before any judgement existed. `tests/test_quality.py::test_the_declared_margin_was_not_revised_to_convert_a_failure` checks the recorded rows against the *live* declared value, so lowering it to manufacture a pass fails the suite.

### ⚠️ The mean hid the real problem — and that is a defect in M2.0

The aggregate said "+0.0303, a near miss". The **per-query record at depth 10 is 3 wins, 3 losses, 1 tie** — a coin flip. The mean is positive only because q07 won by +0.3569 while three losses were smaller:

```
q06 |D|=3  system 0.8452  id_order 1.0000  LOSS -0.1548
q07 |D|=6  system 0.9884  id_order 0.6315  win  +0.3569
q08 |D|=1  system 1.0000  id_order 1.0000  tie
q12 |D|=2  system 0.9468  id_order 0.7896  win  +0.1572
q13 |D|=4  system 0.7871  id_order 0.9655  LOSS -0.1784
q14 |D|=7  system 0.7775  id_order 0.8957  LOSS -0.1182
q15 |D|=5  system 1.0000  id_order 0.8503  win  +0.1497
```

**This is the project's own rule broken by the harness that enforces it.** *Assert integers, report floats* — and the margin test gated on an **averaged float**. A win count cannot be rescued by one lucky query. `_sign_test()` added to `evaluate.py`; the declared criterion is unchanged, this is an additional diagnostic reported beside it.

### What the result does and does not say

- **Does say:** on this label set there is **no evidence** the ranking beats sorting by sha256 hash. `id_order` — relevance-blind, perfectly deterministic — scores 0.8761 nDCG@10 against the system's 0.9064.
- **Does not say:** that the ranking is bad. 7 queries and a 3–3 split is *absence of evidence*, equally consistent with a real but small effect.
- **Does say, clearly:** the current evidence **cannot support a quality claim**, which is exactly the outcome the harness was built to force.

### The advisory layer: 1 relevant document in 21 proposals

Of 21 stored-vector proposals pooled into stratum A, **18 graded 0, 2 graded 1, 1 graded 2**. Exactly one relevant document across 7 queries lay outside D (`Orchestrator-Workers` for `sub agents`). Recall of D alone is already **0.9714** at depth 10 — there was almost nowhere to contribute. Preliminary and underpowered, but it is the first evidence on the neural layer's actual value here, and it is not encouraging.

### Small-sample saturation is severe

Candidate sets run **1 to 7** documents. `q08` has |D| = 1, where every ranker ties trivially. nDCG's log discount is gentle enough that even a poor ordering of a mostly-relevant set scores above 0.77. This is the corpus-size warning from the M2 gate, now measured rather than predicted.

### Two guards re-aimed, not deleted

Both had premises that expired the moment labels existed:
- `test_no_quality_figure_is_published_anywhere_yet` → **`test_every_published_quality_figure_carries_its_provenance`**. The durable rule was never "no numbers"; it is "no number without a producer *and* a `labels_hash`".
- `test_peer_and_plain_docs_state_the_scope_limit` asserted `"no relevance"` in peer.md. Now asserts the document states the failure.

### ➡️ M2.2 is re-scoped again: expand the query set first

More labels on these 7 queries will not resolve a 3–3 split; the variance is between queries, not within them. M2.2 leads with **query-set expansion** — which also unblocks the graph decision, still stuck at a real sample size of one.

---

## M2.2 ✅ (first question) — does the graph layer earn its place? **No.**

Producer: `python3 tools/measure_graph_contribution.py --index index.db`

Deferred through all of M1 as *"needs relevance labels"*. **It never did.** Two label-free measurements settle it.

### 1. Ablation ≠ parameter sensitivity — and M1.6 measured the wrong one

M1.6 nudged `graph.max_depth` 2→3, saw 2/23 queries reorder, and recorded *"live but weak"*. **That measured a nudge, not the layer.** Withholding the signal entirely is the experiment the question actually asked:

```
queries whose output changes    2/23
shallowest rank ever affected   20
candidate set ever changes      False   (a tiebreak cannot add documents)
  q02  |D|= 86  first difference at rank 20
  e06  |D|=147  first difference at rank 61   (synthetic 13-term edge case)
```

**The graph layer is inert above rank 19 across the entire query set.** At evaluation depths 1, 5 and 10 its contribution is *exactly zero* on all 23 queries. At depth 20 it moves one real query. A signal can be insensitive to its parameters while doing a great deal of work — or, as here, none.

### 2. Nuisance screen (mud-detection layer 3): **CLEAN**

The feared pathology — the one that motivated mud-detection — is a signal that propagates score along edges and learns to promote **well-connected** items rather than relevant ones. `best_depth` is structurally that shape.

```
rho vs node degree        -0.0417   over 61 moved items
promoted/corpus degree     0.51x    (promoted items are LESS connected than average)
verdict                    CLEAN
```

**Ruled out, with no evaluation run and no label.** The graph layer is not degree-riding. It is simply silent.

### Why this is structural, not a tuning failure

`best_depth` is the **fourth** component of the sort key, after `bm25_q` and `matched_terms`. It breaks only ties surviving both, and those are rare above rank 19 here. Moving it up the key would let graph proximity override a strictly better lexical match — **which the architecture forbids by design**. So the layer cannot be made influential without abandoning the ordering principle. Its inertness is a *consequence of the guarantee*.

### ⚠️ Layer 2 (REDUNDANT) is partly tautological — do not lean on it

`correlate.assess` returned REDUNDANT with `complementarity = 0.0`. But the graph layer is a **tiebreak**: it annotates depth on BM25 candidates and *cannot* introduce a document, so complementarity is 0 **by construction**. Verified directly (`set(base) == set(candidate)` on every query). Also `rho_mean = 0.826` is contaminated by three empty-|D| queries scoring 0.0; over non-empty queries it is **0.9441**, with **17 of 20 rank-identical**. Directionally right, but the ablation is the load-bearing evidence.

### The decision this opens

Three options, none of which I have taken:

| option | cost | argument |
|---|---|---|
| **Keep, documented as inert** | BFS on every query, 0.149 ms | Cheap, and a larger corpus may make it matter |
| **Remove** | Simplifies the sort key to 4 components | It buys nothing measurable and `graph.expand` is real code with real tests |
| **Keep and re-scope** | New work | Use graph signal for *candidate generation* (documents with no matching term) rather than tie-breaking — an M2 decision already on record |

Removal changes `bench_digest` and the architecture, so it is not mine to take.

### On adopting mud-detection

`mud_detection` used for layers 2–3; **nothing vendored, no dependency added.** `tools/measure_graph_contribution.py` degrades to the ablation alone if the package is absent (verified). `declared_core`/`sqlite3_sag` deliberately **not** used — `declared_core/retrieval/fusion.py` blends signals into one score, which drf rejects outright.

---

## AUDIT — structural integrity cycle, 2026-08-08 (before M2.2)

Ran deliberately: re-read every claim **against the filesystem**, not against memory. Nine findings; all fixed.

### 🔴 Severe

**1. `spec/evaluation.json` asserted something false — and a test held it in place.** Its `status` read *"no labels yet. No quality figure exists and none may be quoted."* Labels existed; a figure was published. Worse, `tests/test_quality.py:398` asserted the literal string `"no labels yet"`, so the suite was **green while enforcing a claim that had stopped being true**. Nothing edited that sentence; the world moved under it.
> This is the drift mechanism the whole project targets, found *inside* the project. The lesson is not "check the spec" — it is that **a test can pin an expired premise**, and only re-reading the claim against reality catches it. Re-aimed at the durable fact (the margin was declared before any judgement existed).

**2. `judge()` let metrics exceed their own bound.** `judge(["a","a"], {"a":3}, depth=2)` returned **recall 2.0 and nDCG 1.63** — a duplicate counted twice, silently, in the direction that flatters the system. Same shape as the un-normalised RBO defect from M1.6. Now raises; stage 1 cannot emit duplicates today, which is precisely why the metric should not paper over one arriving from a future caller (merged results, a fused ranking).

### 🟠 Gaps closed

**3. Hand-transcribed figures had no verifier.** The nDCG values in `spec/benchmarks.json` were typed by hand from CLI output. They happened to match — now `test_recorded_quality_figures_match_a_live_run` re-runs the producer and compares, including the `labels_hash`.

**4. Two sources of truth for the judgements.** `labels.worksheet.jsonl` and `labels.jsonl` are both committed and could drift silently; a stale collect would leave every published figure bound to the wrong hash. Now reconciled by a test.

**5. The release did not bind its labels.** `frozen.json` named spec + index + bench digest, but not the judgements behind the published quality figures. `labels_hash` added to the freeze and to `verify()`.

**6. `test_verify_reports_every_difference` hard-coded `== 4`.** Went stale the instant `labels_hash` joined the freeze — the *same hand-listed-collection defect* as the M1.6 registry test. Fixed structurally: `freeze.VERIFIED_KEYS` is named once and the test iterates it.

### 🟡 Stale prose (fixed)

**7–9.** SOT status line, its M2.1 row (`⬜ NEXT`), its discipline rule 6 (*"there are still no relevance labels"*); STATE.md had **two** M2.1 sections; `LABELLING.md` framed stratum A as to-do.

### Edge-case sweep — clean apart from finding 2

Empty ranking, empty labels, all-grade-0, depth 0, depth > len, unlabelled documents, negative grades, float/string/bool/null grades, blank and comment lines, unknown query and node ids, oracle/reverse/shuffle on empty and single-item lists. All behave, all documented. One asymmetry noted and left: with no relevant documents, recall gets the 1.0 convention and precision does not.

---

## ➡️ M2.2 input: adopt `mud_detection`

`https://github.com/verbalogicproject-creator/NLKE-mud-detection` — user-authored, Apache-2.0, **`dependencies = []`**, stdlib only, "no model in the decision path", "determinism is the audit property", float-free journal hashing. Those are drf's own invariants, independently arrived at.

**Run against M2.1's actual result, it settles the question I answered by eye:**

```
power_report(n=7)   min_detectable_flips = 6   resolution = 0.143
discordant          3 system-only / 3 control-only
McNemar exact       p = 1.0000
Wilson 95%          system 3/7 [0.158, 0.750]  control 3/7 [0.158, 0.750]   overlap
paired delta        +0.0000  CI [-0.7143, +0.7143]  includes zero
verdict             NOT_ESTABLISHED
```

**And it corrects me.** I published nDCG to four decimals (0.9064, margin +0.0303) on a sample whose resolution is **0.143**. Spurious precision — the same error mud-detection's own README calls out about its author at n=19. Recorded in `benchmarks.retrieval_quality.statistical_verdict`.

**`NOT_ESTABLISHED` and the margin FAIL say different things**, and both must be reported: the margin says *the system did not clear the bar*; the statistical verdict says *the sample could not have told us either way*.

### ⚠️ Adopt `mud_detection` only — not `declared_core`

The repo vendors `declared_core/retrieval/` (bm25, **fusion**, structural, intent) and `sqlite3_sag/`. Two reasons to take none of it:
1. **`fusion.py` blends signals into one score.** drf *rejects* fusion — any weight grants authority. Vendoring it would contradict the central guarantee.
2. A second BM25 in-tree is architectural mud in the literal sense. drf's own is proven against a hand-computed reference.

### The strongest lead: Layer 3 against the graph question

mud-detection exists because a **`parent-boost`** signal — propagate a matched child's score up to its parent — passed an LLM compatibility check and then regressed held-out recall. It had learned to promote **well-connected** items rather than relevant ones.

drf's graph layer contributes `best_depth` from BFS expansion. **That is structurally the same pathology risk**, and it is the open question M2.2 inherits ("does the graph layer earn its place?"). Layer 3 tests exactly this — nuisance-correlated signal against a `degree` map — and needs **no evaluation run**. It should be the first thing M2.2 does, before any new labelling.

Its `fixtures/calibration/` also holds 49 queries over a 150-item KG with gold labels and a degree map — measured, not synthetic. A different corpus, so not a drop-in eval set, but it can calibrate the harness against a case with a known answer.

---

## M2.2 CLOSED — 2026-08-10, `v0.0.9`

Three questions were inherited. All three are now answered, and **none of them needed a label.**

### 1. Does the graph layer earn its place as a tiebreak? — **No. It is inert.**

Ablation (withhold the depth map entirely; sort key untouched, so the *signal* is isolated rather than the machinery) changes **2 of 23** queries and never above **rank 20**. At evaluation depths 1, 5 and 10 its contribution is exactly zero on every query.

**Ablation ≠ parameter sensitivity.** M1.6 nudged `graph.max_depth` 2→3, saw 2/23 reorder, and concluded "live but weak". That measured a nudge. Removing the signal is a different experiment, and it is the one that was actually being asked. A layer can be insensitive to its parameters and still do enormous work — or, as here, none.

The inertness is **structural, not a tuning failure**: `best_depth` is the third key component, after `bm25_q` and `matched_terms`. Promoting it would let graph proximity override a strictly better lexical match, which the architecture forbids. It cannot be made influential without abandoning the ordering principle.

### 2. Should graph-reached documents enter D? — **Built, measured, rejected.**

Re-scoped from tie-breaking to **candidate generation**: unmatched documents admitted at score exactly 0, ordered strictly below every lexical match, with `best_depth` becoming the primary discriminator inside that tail — the one place graph proximity genuinely decides an order. Safe only because `bm25.idf` uses `log(…+1.0)`, flooring idf at zero (minimum observed real score 1,017,536,991). That `+1` was added in M1.2 for an unrelated reason and turns out to be the precondition for admitting unmatched documents at all.

| | Measured |
|---|---|
| Candidate growth | 457 → 2,267 over 23 queries (**4.96×**) |
| Advisory horizon | 16 of 23 queries with a reachable depth → **1** |
| Relevant documents gained | **Zero** |
| Invariant | Holds — 0 matched documents displaced, lexical prefix preserved elementwise |
| mud-detection | REDUNDANT (complementarity 0.0000, rescue_rate 0.0) |
| Sensitivity | Reorders **20 of 23** — the largest blast radius of any setting, exceeding `ranking.b`'s 13 |

**Decision: implemented, tested, invariant-checked, `DEFAULT OFF`.** Maximum disruption, zero measured gain — and it would additionally silence the advisory layer, since authoritative tail positions and advisory positions are in direct competition for the same space. This is exactly the failure mud-detection exists to prevent: an architecturally reasonable change that wins nothing. The evidence decides the default, not the reasoning.

### 3. Is `best_depth` the parent-boost pathology? — **No, and now that verdict is worth something.**

Layer 3 returned CLEAN for every real drf configuration. **A screen that only ever returns CLEAN is a claim about the screen**, not about the signal — the chaos-control lesson from M1.6 applied to a borrowed tool.

So: a **positive control**. Identical architecture, admission and sort key; the admitted tail ordered by node **degree** instead of `best_depth` — the parent-boost pathology reproduced inside drf's own ordering, the narrowest edit that makes the signal unsafe.

| | rho | verdict |
|---|---|---|
| drf admitted tail | **+0.0288** | CLEAN |
| Deliberately unsafe tail | **+0.6394** | NUISANCE_CORRELATED |

The screen separates them. Note what the construction also shows: the pathology can **only** be built inside the admitted tail, because the sort key opens with `-bm25_q` and no amount of connectivity lifts an unmatched document above a matched one. The architecture confines the failure mode to the region where everything already scored zero.

### The release-day finding: a measurement had decayed into a transcription

`drf_tail_rho` was recorded as **−0.0288**. The producer emits **+0.0288**. Sign inverted. Caught only by re-running the producer before tagging.

It had shipped through a full structural audit with a green suite, because **nothing ever ran the producer** — the `mud_detection` checkout it needs had been lost from `/tmp`, so the figure had silently stopped being a measurement. No event marked the transition. Verdict and magnitude were unaffected, but the sign reverses the story: the tail leans very slightly *toward* degree, not away from it.

**This is the third instance of one defect.**

| Where | Hand-maintained collection | How it drifted |
|---|---|---|
| M1.6 registry test | invariants listed literally | new invariant → still green |
| `test_verify_reports_every_difference` | `== 4` | `labels_hash` joined the freeze |
| M2.2 | *which* benchmark sections get a live-run check | 1 of 10 checked, 9 undeclared |

Fixed the same way as the first two — **derive the collection**. `tests/test_benchmarks_provenance.py` walks `spec/benchmarks.json`, and every section carrying a `producer` must declare `verified_by` (naming a test that re-runs it, whose existence is checked) or `verification_exempt` (a reason longer than a shrug). A new section cannot be silently unverified. Two new live-run checks landed with it: the graph ablation and the advisory horizon — both label-free, both falsified to confirm they can fail.

**Standing rule added:** re-run every producer before tagging a release. A recorded `reproduce` command is now part of any section whose producer needs an optional dependency.

**Known residual:** the positive-control test skips without `mud_detection`, so the default environment does not exercise it. That is honest but it is environmental vacuity — the strongest evidence is the most fragile check. Mitigated by the recorded reproduce command, not solved.

---

## M2 GATE — run 2026-08-08, after M1 completed

**Yes, M1 changes the M2 roadmap. Four structural findings, not just refinements.**

### ⚠️ 1. Subordination *bounds what M2 can measure*. This is the big one.

The neural layer can only **append below D**. It therefore **cannot improve precision@1, nDCG@5, or MRR — ever, by construction.** Those numbers will be byte-identical with the provider on and off, because that is the guarantee working.

Anyone evaluating at shallow depth will measure exactly zero benefit and conclude the neural layer is useless. That would be a **measurement artefact, not a finding.**

M2 must evaluate the advisory layer only where it can act:

| metric | can neural affect it? |
|---|---|
| precision@1, nDCG@5, MRR | **No — structurally impossible** |
| recall@k for k > \|D\| | **Yes — the only place it can help** |
| coverage of relevant docs absent from D | **Yes — the honest question** |

The real question for the advisory layer is *"does it find relevant documents that lexical retrieval missed entirely?"* Nothing else.

### ⚠️ 2. The "exact integers" rule does not transfer — a new discipline is needed

nDCG and recall are inherently floats; they cannot be asserted as exact integers. M1's central assertion discipline **does not carry into M2**, and pretending otherwise would be worse than admitting it.

M1 measured why this matters: the chaos control scored Kendall's Tau **0.9761** while being provably broken. A quality metric that moves 0.97 → 0.98 says nothing on its own.

**Replacement discipline for M2:** assert on **rank positions (integers)** and **counts of relevant documents retrieved (integers)**; report nDCG as a float for humans; and require any improvement claim to clear a **stated margin against a control**, never a bare comparison.

### ⚠️ 3. Recall@k is a set metric — M1 proved set metrics are blind to ordering

Jaccard reported **1.0000** for a pipeline returning five different orderings in five runs. Recall@k has the same blindness. M2 must never report recall alone; always pair it with a rank-sensitive metric, or it will silently discard everything M1 established about ordering.

### ⚠️ 4. The falsifier pattern transfers, but needs restating

You cannot falsify "nDCG improved" by mutating code. The M2 analogue is the **chaos control generalised**: score a deliberately-bad ranker (random order, or the chaos ranker already implemented) and require the real one to beat it by a stated margin. If it doesn't, the evaluation is measuring noise. Build this **before** the first quality number is quoted.

### Reordered M2 plan

| step | why this order |
|---|---|
| **M2.0** Evaluation harness + control | Before any labels. If a random ranker scores well on the harness, the harness is wrong. |
| **M2.1** Relevance labels | ~50 graded query/document pairs. Unblocks all three deferred decisions. |
| **M2.2** Answer the three deferred questions | weighted `s1_q`; graph-only candidates in D; **does the graph layer earn its place** (reorders 2/23 vs `ranking.b` 13/23) |
| **M2.3** Re-embed with BGE, then compare | ReproRAG found embedding choice dominates variance — so switch *after* labels exist, or the change cannot be evaluated |
| **M2.4** Remote provider | Needs 2.3; least urgent despite being most interesting |

### Corpus-size warning for labelling

266 documents with ~20 candidates per query means IR metrics **saturate easily and are noisy**. Expect wide error bars. Either accept them explicitly, or plan a larger evaluation corpus — and note that a larger corpus hits M3's un-truncated posting-union sooner than planned.

### Carried forward unchanged

- RRF stays rejected (grants authority to the advisory side)
- FTS5 triggers from the old project must **not** be lifted verbatim
- `./tools/check_isolation.sh` after any new test file — falsifiers cannot catch order-dependence
- Every checkpoint needs a falsifier registered **before** it is trusted

---

## Milestone 1 complete

**Recommended reordering, from the M1.5/M1.6 findings.** Start M2 with **relevance labels**, not features. Three decisions are currently blocked on having no ground truth:

1. whether to add a weighted lexical+graph score (deferred deliberately — a weight chosen now is unfalsifiable)
2. whether graph-reached documents with no matching term should enter D at all
3. **whether the graph layer earns its place** — it reorders only 2 of 23 queries against `ranking.b`'s 13

Building more retrieval before labels means settling those by taste. Even ~50 hand-judged pairs converts three guesses into three measurements. BGE re-embedding and the remote provider follow *after* — they cannot be evaluated without labels either.

## M1.7 planning record

Tag; record `spec_sha`, manifest hash and bench digest in `docs/peer.md`. Most of this already renders — `spec_sha` and `content_hash` are in the context. The remaining work is a `drf freeze` that captures the triple and a test that the recorded values match a live rebuild.

## M1.6 planning record

Four audiences (peer, agent, operator, plain) generated from `spec/`. `Template.substitute`, never `safe_substitute`, so an unresolved placeholder is a hard error and no number can enter docs without a producer. A hand-edit must **fail a test**.

**Harvest from M1.6.** The docs now have real numbers to render, all producer-backed: 28 cells / 1 digest, the chaos comparison table, the sensitivity counts, `content_hash 90ab5db9…`. `docs/peer.md` must also carry the scope honesty already agreed: milestone 1 has **no relevance labels** and makes **no claim about retrieval quality**.

**Audit before writing.** "A hand-edit fails a test" is the vacuity risk — if the test regenerates the file before comparing, it can never fail. The falsifier must edit a rendered file and require the test to notice.

## M1.5 planning record

**The centerpiece.** Subordination must hold against adversarial, crashing, hanging and flooding providers; the deterministic prefix sha must be identical in every case; `discordant_pairs == 0`.

**Harvest from M1.3.** `Advisory[T]`'s allowlist is `drf.retrieval.merge` — that module does not exist yet, so the boxing has never actually been exercised end-to-end. Stage 1 now produces a real D to append below, so M1.4 is the first point where the central guarantee is testable rather than declared.

**Audit before writing.** `discordant_pairs == 0` risks the same vacuity: if `merge()` appends by construction, the postcondition may be unable to fail. **Register a falsifier first** — make `merge()` interleave advisory results into D — and require the test to fail under it. Also falsify the `Advisory.unwrap()` allowlist by adding a second module.

**Carry forward:** RRF remains actively rejected. M1.4 is where it would be most tempting, since that is exactly where a "combine the two rankings" instinct arrives.

---

### M1.3 planning record (kept for provenance)

Milestone gate, run before writing code.

**Harvest from M1.2.** The injective sort key is now known to be *load-bearing, not defensive*: **7 of 15** ordinary queries have exact ties in their top set, and the reference corpus reproduces one exactly (`d2` and `d5` both score `333105558` on `alpha`). Without a total order, "the best result" is undefined for nearly half of real queries — and we proved the cost of not noticing, by publishing tiebreak-dependent figures ourselves. `tests/test_retrieval.py::test_exact_ties_are_pervasive_on_the_real_corpus` already pins this.

**Audit — the planned checkpoint is probably vacuous.** `len(set(sort_keys)) == len(D)` is guaranteed *by construction* once the key ends in a content-addressed PK, so it may be unable to fail, exactly like the shuffle test M1.2 retired. **Register a falsifier before trusting it:** drop the `node_id` component from the key and require the test to fail. If it survives, the assertion is decoration.

**Carry forward:** the FTS5 trigger correction (see above) applies the moment `graph.py` or M3 touches FTS5 — the plan's "lift verbatim" instruction is wrong.

### Measured graph facts (drive `graph.py`)

```
266 nodes, 553 edges     max degree 27, p90 9, median 3, 5 nodes with 0 edges
directed cycles present  -> visited-set is mandatory, not defensive
BFS depth-2, 10 seeds    0.149 ms      full depth-4 all-pairs: 5 ms / 7,194 pairs
```

**Traverse bidirectionally.** Measured on our graph:

| depth-2 expansion | forward-only | bidirectional |
|---|---|---|
| mean reach | 7.2 | **23.3** (3.22×) |
| nodes reaching nothing | **81/266** | 5/266 |

Forward-only would leave **76 nodes (28.6%) with zero graph signal purely because of edge direction**, not because they are unconnected; the residual 5 are exactly the known isolated nodes. No precomputed path index — at 0.149 ms, on-the-fly BFS makes one pure overhead.

### `~/Downloads/playbooks/playbooks-v2/KNOWLEDGE-*.pb` — reviewed, 1 of 9 admitted

Criterion: bears on an *open* M1.3–M1.8 decision **and** contains testable specifics **and** those are checkable. Default is exclusion.

- **Admitted: `KNOWLEDGE-bidirectional-path-indexing.pb`** — for its bidirectional claim only, and only because it was re-verified on our graph (table above). Its path-index artifact is excluded on measurement.
- **Excluded: the other 8.** No open decision + testable specific.
- **Provenance, checked rather than assumed.** Its headline result reproduces *exactly* — `bash_execute` → **459** capabilities, per-depth **50/263/146** — against `python_apps/claude_kg_truth/claude-code-tools-kg.db` (511 nodes / 631 edges). The method was really run. But its `context_caching` example belongs to a *different* graph, its "789 nodes / 28,292 paths" index statistics match no graph on this system, and Pattern 4's code opens `unified-kg-optimized.db`, **which does not exist anywhere**. Verified method, real numbers, unreliable labelling.
- **Zero citations across all 9 files (240 KB).**
- **⚠️ RRF is actively rejected, not merely unused.** Reciprocal Rank Fusion appears in **5 of 9** files, and `KNOWLEDGE-graph-traversal.pb` Strategy 4 recommends it for "production queries". RRF blends rankings into one score, which *grants authority to the neural side* — exactly what the three-stage architecture forbids. It is the ecosystem's default advice and the most tempting wrong turn available.

---

### M1.2 planning record (kept for provenance)

**Document definition (decided, binds `PARSER_VERSION`): `name + description`.** `source_ref` excluded — 71% of its tokens already present, 84/266 docs gain zero, and duplicates inflate `tf` on title terms. `type` excluded — 29/30 tokens already in vocabulary and 101 docs share `use_case`, so IDF crushes it; type belongs in a structured filter.

### Corpus facts, measured (`tools/measure_length_norm.py`)

```
N=266  avgdl=18.9 tokens  min=4  median=17  max=61   (name + description)
0 empty descriptions -> avgdl well-defined
5 isolated nodes -> score on text alone
```

The old engine's missing `b`/`avgdl` was **not** a harmless omission — measured against `b=0.75` over 15 corpus-vocabulary queries, **tiebreak-free**:

| | |
|---|---|
| strictly-ordered pairs @10 discordant | **60/352 = 17.0%** |
| pairs tie-affected (order undefined without a convention) | **95 = 21.3%** |
| queries with a tie in a top set | **7/15** |
| top sets disjoint (unambiguously changed) | **3/15** |
| mean top-set length, `b=0.75` → `b=0.0` | **15.9 → 24.9** tokens (corpus 18.9) |
| direction: longer / shorter / equal | **9 / 0 / 6** |

The direction claim holds *monotonically* — zero counterexamples.

> **⚠️ Correction — figures withdrawn.** An earlier revision of this file reported **26.2% / 27.2% discordance** and **"top-1 changed 9/15"**. Both were **tiebreak-dependent**: 7 of 15 queries have exact ties in a top set, and breaking them by *lowest* node id gives 9 while *highest* gives 4. Those numbers described *BM25 plus an arbitrary convention*, not BM25 — the same defect catalogued in the old engine ("no tiebreak on any sort"), reproduced in our own measurement. The discordance figure was inflated because tied pairs were counted as ordered. All metrics above are now defined only over strictly-ordered comparisons. **The load-bearing directional claim survived unchanged.**

That 7/15 tie rate is the empirical case for M1.3: without a total order, "the best result" is **undefined** for nearly half of ordinary queries.

### 1. Replace "shuffling posting order changes nothing" with a control test

Under the commutativity rule, that test now **passes by construction**: `fsum` returns the unique correctly-rounded total, and integer addition is commutative, so no permutation can change a score. A test that cannot fail proves nothing. Keep the shuffle as a cheap regression guard, but the load-bearing test is a **control** — a naive `sum()`-with-early-rounding formulation must be shown to *differ*, exactly as `test_source_does_contain_what_we_forbid` proves the DDL grep can fire.

### 2. Pin *where* quantisation happens — currently unspecified

`quantize(fsum(contributions))` and `sum(quantize(c) for c in contributions)` are both deterministic and both commutative, but they **give different answers** at the last quantum. The original plan said "fsum over *sorted* contributions"; sorting is now known to be unnecessary (fsum's result is order-free), so that phrasing should not survive into the code. Record the chosen order of operations in `spec/ranking.json` and assert the alternative differs, or the spec is unfalsifiable.

### 3. Decide the document definition — it is consequential and unrecorded

M1.1 stores `type`, `name`, `description`, `source_ref`, `metadata`. Which of these *is* the indexed document is undecided, and it moves `avgdl` by 31%:

| Document | avgdl |
|---|---|
| `name + description` | 18.9 |
| `+ type` | 20.8 |
| `+ source_ref` (slug, underscores split) | 24.8 |

`source_ref` slugs (`claude_technique_prompt_caching`) are high-signal but largely duplicate `name` tokens, inflating `tf`. Whatever is chosen binds to `PARSER_VERSION`.

### 4. Test length normalisation on the real corpus, not only the 5-doc reference

The synthetic reference validates the *formula*; it cannot show `b` matters *here*. The 26.2% / 9-of-15 / 0-counterexample figures above are a regression test with teeth. `tools/measure_length_norm.py` reimplements BM25 independently rather than importing `drf.retrieval.bm25`, so it stays a genuine cross-check once the real implementation lands — a measurement that calls the code under test cannot contradict it.

Do **not** assert byte-identical `.db` files — SQLite's header change counter makes that fragile. Assert manifest `content_hash` + `drf export --canonical` equality.

---

## Working discipline (this is the point of the project)

1. **No number without a producer.** Every metric maps to the script that emits it. Counts via `len()`, never asserted literals.
2. **Assert exact integers, never floats.** `discordant_pairs == 0`, not `kendall_tau == 1.000`.
3. **Docs are generated from `spec/`, never hand-written.** `Template.substitute` (strict) so an unresolved placeholder is a hard error. A hand-edit must fail a test.
4. **Verify before claiming — including my own claims.** Two corrections already caught this way:
   - The design agent confabulated a detail ("BGE on your phone") that the user had not stated. It later turned out true, but was unfounded when written.
   - I justified `math.fsum` by claiming `sum()` is order-dependent. **Measured false on CPython 3.12** — Neumaier compensation, 0 disagreements in 200,000 random sums. `fsum` is kept for the *documented language guarantee* that survives a change of interpreter, and the docstring now says so.

## Deferred, deliberately

- 3 unused dimension templates (`document_analysis`, `product_catalog`, `research_paper`) — 49 dimensions, zero consumers, no ground truth
- The 36 "pending" transferred dimensions — nothing needs them
- Prior work that *does* carry forward: `~/Downloads/synthesis-rules/templates/embeddings/` v2 schema (`code_analysis` = 28 dimensions, 0 pending, 11 authored with numeric endpoint anchors) plus `validation/` (entropy discrimination harness). Empirical rules from it: **endpoint anchors are mandatory** (one prose endpoint lost 40.3% of a corpus to unbanded values), and **discriminating power is a property of (rubric × unit)** (entropy 0.33 per-function → 0.94 per-module with identical bands).
