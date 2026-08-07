# Deterministic RAG — Progress Report & Source of Truth

**Date:** 2026-08-07
**Status:** M1.0 complete (23/23 tests green). Framework scaffolded at `/home/eyaln/Documents/Deterministic-RAG-Framework/`.
**Purpose:** Persistent system atlas, plan, and context. This document supersedes any claim made in the recovered source project's own documentation.

> **Reading order for a fresh session:** this file → `STATE.md` (resume point) → `/home/eyaln/.claude/plans/plan-step-1-out-crystalline-firefly.md` (full build plan).

---

## 0. Executive summary

A body of research from Nov–Dec 2025, believed lost, was recovered and audited on 2026-08-07. The audit found:

- **The architecture is sound and was independently validated by the field.** Two papers published Sept–Oct 2025 arrive at the same core principle. This is convergence, not lag.
- **The implementation is roughly half hollow**, and its headline retrieval feature — semantic embedding similarity — **never executed a single time**.
- **The documentation is systematically unreliable.** Not sloppy: actively false in ways that survived three passes.

The response is a clean framework that keeps the architecture and the knowledge graph, discards the code, and — critically — encodes the anti-drift discipline into machinery rather than intent.

**Current position:** foundational contract layer built and proven. The enforcement is real: a mislabelled "deterministic" action raises on its second call; advisory data physically cannot be unwrapped outside one allowlisted module.

---

# PART I — SYSTEM ATLAS

Everything below was verified by direct inspection (opening databases, reading code, running tools). **Nothing here is repeated from a README.**

## 1.1 Environment

| | |
|---|---|
| Python | 3.12.3 |
| SQLite | 3.45.1 (FTS5 available) |
| Present | `numpy 1.26.4`, `scipy 1.11.4`, `matplotlib` |
| **Missing** | `networkx`, `sklearn`, `anthropic`, `graphviz`, `sentence-transformers` |

No `kg-*.py` tool imports networkx or sklearn, so only `kg-pipeline.py` (needs `anthropic`) and `kg-architecture-viz.py`'s render path (needs `graphviz`) are blocked. **`sentence-transformers` being absent means there is no local query encoder** — which is why the neural design defaults to anchor mode over frozen vectors.

## 1.2 Tree topology

Four near-duplicate trees. All 15 `kg-*.py` files are md5-identical across them.

| Tree | Size | Role |
|---|---|---|
| `~/Downloads/claude-cookbook-kg3/` | 8.5 G | **Canonical superset** — DBs + tools colocated. Run tools from here. |
| `~/Downloads/full-original-system/` | 8.3 G | Byte-identical twin for all KG assets |
| `~/Downloads/synthesis-rules/` | 413 M | Different project (python_apps KG / unified_memory) |
| `~/Downloads/2/` | 34 M | Fragment; embeddings only, no usable DB |

Also holding divergent copies: `~/Downloads/kg-truth/Knowledge-Graphs/` and `~/Downloads/llm_web_brain/cost optimization/`.

## 1.3 Databases — claimed vs actual

| Database | Claimed | **Actual** | Verdict |
|---|---|---|---|
| `claude-cookbook-kg.db` | 266 / 605 | **266 nodes / 605 edges** | ✅ TRUE |
| `kg-extractor-v2-complete.db` | 1,791 / 283,166 | **1,791 / 283,166** | ✅ TRUE |
| `unified-kg.db` | 2,057 nodes | **2,086** (275 manual + 1,811 auto) | ❌ FALSE — 2,057 is arithmetic on two source DBs, not a count |
| `unified-kg.db` edges | "284K+" | **284,202** (283,190 auto + 605 manual + 407 cross-ref) | ✅ TRUE |
| `python_apps_kg.db` | 12,154 entities | **12,154 rows = 2× 6,077 distinct** | ⚠️ **CORRUPT** |
| `unified_memory.db` | "unified memory" | 14,141 KG entities but **19 memory entries** | ⚠️ overstated ~3 orders of magnitude |

**29 further `.db` files are 0 bytes.** Any document citing them as data sources is wrong.

### The chosen corpus — `claude-cookbook-kg.db`

`~/Downloads/claude-cookbook-kg3/claude-cookbook-kg.db` (1,064,960 B)

- `nodes` 266, `edges` 605, `node_embeddings` **228**, `node_positions` 42, `collaboration_context` 136, `kg_metadata` 7, `schema_metadata` 12, `schema_versions` 4
- Embeddings are **genuine**: `sentence-transformers/all-MiniLM-L6-v2`, dim 384, BLOB 1536 B (= 384 × float32), non-degenerate. 38 nodes lack vectors.
- **Data quality issues to handle at ingest:** 48 duplicate `(from,to,type)` edge groups, 4 orphan edges (`to_node` absent from `nodes`), 5 isolated nodes.
- ⚠️ Name collision: `~/Downloads/synthesis-rules/python_apps/claude-cookbook-kg/claude-cookbook-kg.db` is an **older, smaller** graph (232/562). Different file, same name.

### Rejected corpora

- **`python_apps_kg.db`** — every entity duplicated exactly twice. The builder `INSERT`s with no `DELETE`. This corrupts IDF (every `df` doubles), doubles context-graph cliques, and lets duplicates crowd each other in top-k. The REPL banner advertises 6,077; the ranker sees 12,154.
- **`unified-kg.db` (canonical copy)** — only **17 embeddings for 2,086 nodes (0.8%)**. A far more complete copy exists at `~/Downloads/llm_web_brain/cost optimization/unified-kg.db` (2,164 nodes, **2,152 embeddings ≈ 99%**, plus `tfidf_index` 2,148) — **and no tool points at it.** Worth revisiting at scale-up.

## 1.4 Embedding assets

| Artifact | Shape | Notes |
|---|---|---|
| `claude-cookbook-kg.db → node_embeddings` | 228 × 384 | MiniLM-L6-v2, real. **Our starting vectors.** |
| `~/Downloads/2/data/embedding_matrix.npy` | 259 × 50 | Largest interpretable-dimension matrix that exists |
| `.../auto_kg_embeddings_production/` | **17** × 50 | Named "production", smaller than the one named "test" (20) |
| `.../claude_kg_truth/embeddings.pkl` | 496 × 256 | Loads clean; most substantive trained artifact found |
| `.../gemini-kg/embeddings.pkl` | — | **Dead** — `ModuleNotFoundError: contextual_embeddings` |
| `dimension_names.npy` | (50,) | The 50 interpretable axis labels, consistent across files |

**Ceiling of the 50-d interpretable work: 259 nodes.** Against a 2,086-node graph.

## 1.5 The BGE endpoint — measured

`http://10.161.19.239:8080` — llama.cpp serving **BAAI/bge-large-en-v1.5 Q4_0 GGUF** from an Android path.

| Property | Measured value |
|---|---|
| API | `POST /v1/embeddings` `{"input":..., "model":"bge"}` (OpenAI-shaped) |
| Dimension | **1024** |
| Normalisation | **Already L2-normalised** → cosine = dot product, no numpy needed |
| Latency | 0.08 – 0.33 s |
| Determinism | **5 sequential identical requests → byte-identical.** Single vs in-batch → **byte-identical** |
| Health | `GET /health` → `{"status":"ok"}`; no `/info`, no `/docs` |

⚠️ **1024-d BGE vectors can never be compared with 384-d MiniLM vectors.** Using BGE requires re-embedding all 266 nodes (~30 s at measured latency). The `vectors` table is keyed `(model_id, node_id)` so spaces coexist and cross-space comparison is impossible by construction.

> The determinism result is a property of *this server build under light load*, not of remote embedders generally. The `@action` replay check stays armed so drift fails loudly.

## 1.6 Tools inventory

**Claimed: "12 core tools, 8,680 lines." Actual: 15 files, 9,335 lines.** (The top 13 files sum to exactly 8,681 — a stale count mislabelled.)

**The tools are the strongest part of the recovered project.**

| Verdict | Count | Tools |
|---|---|---|
| **FUNCTIONAL** | 9 | `kg-validate-rules`, `kg-ask`, `kg-query-unified`, `kg-evolution-tracker`, `kg-merger`, `kg-importer`, `kg-validation-corpus-builder`, `kg-impact-analyzer`, `kg-critical-components` |
| **PARTIAL** | 5 | `kg-architecture-viz` (DOT ok, render needs graphviz), `kg-pattern-mapper` (leans on the 17-node matrix), `kg-self-evolve` (2 of 3 DBs don't exist), `kg-ask-frontend-enhancement`, `kg-mine-github` (needs network) |
| **STUB** | 1 | `kg-pipeline` — 84 prints, **zero** DB access; tier modules unimportable |

**12 of 15 run today** from `~/Downloads/claude-cookbook-kg3/`. All use *relative* default DB paths, so they only work from a directory containing the DBs.

`query-cookbook.py` (3,507 lines, 983 prints, 62 flags): **functional query core, hollow reasoning shell.** Primitives like `find_nodes`, `get_edges_from/to`, `explore_node`, `trace_relationship`, `get_provenance` are real. But **25 of 48 user-facing handlers (52%) print constants** and perform ≤1 data lookup:

```python
# query-cookbook.py:1881, 1913 — literal constants, not measurements
print(f"  Latency: 200-400ms")
print(f"  Lines of code: ~50")
```

## 1.7 The old retrieval engine — why it is replaced

`~/Downloads/synthesis-rules/python_apps_hybrid_query.py`

```python
# line 198-212
def _compute_embedding_similarity(self, query_text, entity_id) -> float:
    ...
    # For now, return 0.5 as placeholder
    return 0.5  # Placeholder
```

**Consequence traced:** every candidate scores 0.5 → `min == max` → normalisation yields 0.0 for all → **the α = 0.40 embedding term contributes exactly nothing.**

| State | Effective formula |
|---|---|
| Embeddings present | `0.45·bm25 + 0.15·boost` |
| Embeddings absent | `0.85·bm25 + 0.15·boost` |

**Loading embeddings makes retrieval worse** — it halves BM25's weight and substitutes nothing. Every published recall figure (88.5%, 95.6%) describes keyword search plus a same-file popularity prior.

Further defects:
1. **BM25 is not BM25** — no `b`, no `avgdl`, no length normalisation. Long padded entities win structurally.
2. **Unordered `LIMIT 100`** (`:342`) — arbitrary candidate truncation.
3. **No tiebreak on any sort** (`:304`, `:317`) — stable sort over arbitrary input = arbitrary output.
4. **Hard `[:15]` boundary** (`:307`) — one tie at the cut cascades into global reordering.
5. **α/β/γ hardcoded** (`:46-48`), configurable nowhere.
6. **Non-deterministic build** — unsorted `Path.glob`, `AUTOINCREMENT` ids, `created_at DEFAULT CURRENT_TIMESTAMP`.
7. **`api_server.py` queries tables that exist in neither database** → `/api/query` returns HTTP 500.

**Worth keeping:** `config_manager.py:314-397` (`diff`/`_diff_dicts` — genuinely good), its DB-backed named snapshots, and the FTS5 external-content + trigger pattern at `python_apps_kg_builder.py:68-93`.

---

# PART II — CONTEXT

## 2.1 The two halves

`KG-QUERY-FLAGS-REFERENCE.md:32-45` specifies the system as two cooperating dimension types:

> **Interface Dimensions** (CLI flags): *how you ask the question* · **Data Dimensions** (semantic): *what aspects get measured* — "Both work together"

Only the interface half was built. The reason nothing visibly broke:

```python
def _compute_graph_similarity(...):
    """... No embeddings required (Phase 1)"""
```

`--similar-to` was **deliberately built to avoid the semantic layer**, so no shipped flag ever depended on a dimension. The halves drifted apart silently.

The deferral is dated and explicit — `WORKFLOW-SYNTHESIS-SESSION-COMPLETE.md:209-213`:

> **Lazy Re-Analysis (DEFERRED)** — 259 nodes marked for re-analysis with 56 dimensions. **Current 50-dim embeddings still functional.**

## 2.2 Documentation drift — the catalogue

| Claim | Reality |
|---|---|
| 147 semantic dimensions | 118 present, only **70 actually defined**; 48 were name-pointers with no rubric, no category, no examples |
| Flags reference documents 63 flags, "all active" | **36 documented flags do not exist**; **35 real flags are undocumented** — including `--fastest`, `--simplest`, `--budget`, the three most obviously needing dimensions |
| `Todos-flag-arsenal.md` header `✅ COMPLETE` | ~970 checkboxes, **all unchecked**; footer still reads `Ready for Review` |
| "80%+ accuracy vs 60% graph-based" | **No accuracy evaluation exists anywhere in the corpus.** No labels, no gold set, no metric definition |
| "150x faster than estimate" | Measured against an estimate for a *different, larger* deliverable that was scoped down mid-document |
| POC "0.94 vs 0.77, beats Voyage" | n=20 nodes, **one query**, no control. The 0.77 has no provenance; the same table says "Not tested yet". Top-10 spanning 0.949→0.909 is **weak discrimination**, reported as a win |
| `similarity: 1.00 — PERFECT!` "BREAKTHROUGH" | Under Jaccard-over-neighbours, 1.00 between two differently-named nodes means **identical neighbour sets** — a duplicate-node bug signature |
| "15-25x ROI", "65% context efficiency" | Bare assertions; the 65% is rounded up from its own document's 51-63% |

**Structural irony:** `semantic-dimentions.txt` is a session transcript whose line 3-4 reads `⎿ User rejected write to KG-QUERY-FLAGS-REFERENCE.md`. **The rejected draft documented the real flags.** The inaccurate version shipped.

## 2.3 Four incompatible vocabularies

1. **Canonical JSON** — 56 dimensions (what SET 4 derives from)
2. **Production generator** — 50 dimensions, a clean subset of the 56 ✅
3. **`SEMANTIC-DIMENSIONS-REFERENCE.md`** — a *different* 50, only **10 names shared**. Header claims 56 and promises a "Category 9 below" that **does not exist in the file**
4. **`semantic-dimentions.txt`** — a *third* 56-variant; 50 match canonical, 6 differ

Plus `DOMAIN-SPECIFIC-DIMENSIONS-DESIGN.md` proposing ~120 more (healthcare/fintech/gaming) — names and one-line descriptions only, **2 of ~120 filled in**.

**No file anywhere contains a per-dimension 5-tier scoring rubric.** The only reusable artifacts are a single global 5-band scale (`SEMANTIC-DIMENSIONS-REFERENCE.md:468-474`) and consistency heuristics at `:476-488` (inverse relationships, logical implications, exclusivity, sparsity).

## 2.4 The shape mismatch — why wiring alone wouldn't work

The 56 dimensions almost all measure *"how much does X matter to this node"* — **sensitivity weights**. The flags need *"what IS this node's X"* — **absolute magnitudes**.

| Flag | Needs | Catalog offers | Verdict |
|---|---|---|---|
| `--fastest` | observed latency (ms) | `latency_sensitivity` = *how much latency matters* | a weight, not a value |
| `--budget cost 50` | absolute $/month | `cost_optimization` = *how much this saves* | a delta, not a level |
| `--simplest` | LOC / dependency footprint | `implementation_complexity` | not a size measure |

Ranking by `latency_sensitivity` returns things that **care about** speed, not things that **are** fast. The hardcoding is the honest downstream consequence of a data-model mismatch. **~14 net-new magnitude dimensions would be required.** Of the 62 flags, ~40 need no dimensions at all (correctly built graph operations); ~22 do.

## 2.5 Field anchors

### SAT-Graph RAG — arXiv 2510.06002 (Oct 2025)

> **Maximal Determinism:** isolate this non-determinism at the API's entry points… Once a formal id (URI/URN) is acquired, all subsequent actions that operate on it are **guaranteed to be deterministic.**

28 canonical actions — *atomic, composable into DAGs, auditable*. Every action returns justification metadata, so **auditability does not require determinism, only declared confidence**. Temporal policy is an explicit query parameter. Structural entities (tree) are strictly separated from classification ontology (DAG).

> **Evaluation Gap** — "No formal evaluation metrics or benchmarks have been implemented."

### ReproRAG — arXiv 2509.18869 (Sept 2025)

Metric suite: Exact Match Rate, Jaccard, Kendall's Tau, RBO, Overlap Coefficient, Score Stability.

**Measured hierarchy of uncertainty** — and it falsified the field's folk wisdom:

| Source | Finding |
|---|---|
| **Embedding model choice** | **Dominant.** BGE↔E5 Overlap 0.540 / RBO 0.570 / τ 0.384; BGE↔Qwen 0.454/0.486/0.338; E5↔Qwen 0.432/0.474/0.322 — **p > 0.05, not statistically significant** |
| Data insertion | Overlap ~0.80, RBO ~0.73, but **Kendall τ = 1.000, zero std-dev** → displacement, *not* re-ranking |
| Precision (FP32/16/BF16/TF32) | Real but small; BF16 the outlier at L2 6.31e-03 |
| **ANN algorithms** (HNSW/IVF/LSH) | **Perfect 1.000 on every metric** — the common hypothesis is false |
| `cudnn.deterministic` flags | **No observable effect**; non-det mode 10–15% *faster* |
| Distributed execution | Perfect 1.000 across all sharding strategies |

**Claim discipline (load-bearing):** vector RAG is **not** non-reproducible — a fixed pipeline is perfectly reproducible (8/8 configs, L2 = 0.00, cosine = 1.0). The defensible claim is narrower: *reproducible conditional on a model choice that is itself arbitrary and drives ~50% of result variance.*

### What the field teaches us to adopt

Typed action contracts over ad-hoc flags · justification on every return · temporal policy as a parameter · structural/classification separation · measured (not assumed) uncertainty ranking · standard metrics so results are legible · **plainly stated limitations sections** — both papers state theirs and it costs them nothing.

---

# PART III — ARCHITECTURE

## 3.1 Two independent axes

`determinism` (same input → same output) and `authority` (may influence ranking) are **orthogonal**. Conflating them is the mistake the contract layer exists to prevent.

| Action | determinism | authority |
|---|---|---|
| `lexical.candidates`, `bm25_score`, `graph.expand`, `stage1.rank` | deterministic | **authoritative** |
| `neural.propose_from_anchors` | **deterministic** | **advisory** |
| `neural.encode_query_remote` | **deterministic** *(measured)* | **advisory** |
| `merge.append_advisory` | deterministic | authoritative |

Anchor-mode search over frozen vectors is fully deterministic **and** strictly advisory. The BGE endpoint is measured deterministic **and** still advisory. Authority is architectural; determinism is empirical.

## 3.2 The central guarantee

```
Stage 1  AUTHORITATIVE : BM25 + graph traversal → total order D
Stage 2  ADVISORY      : neural proposes candidates NOT in D
Stage 3  MERGE         : append-only. order(D) preserved exactly.
```

**No α term. Deleted, not zeroed** — any weight, in a sum *or* a product, implies authority. (A product is worse: a low vector score would *veto* a perfect deterministic match.)

Machine-checkable invariant: `discordant_pairs(rank(D) | neural on, rank(D) | neural off) == 0`.

## 3.3 Why ordering is unambiguous

Sort key, all `int`/`str`, no float ever compared:

```
(-s1_q, -bm25_q, -matched_terms, best_depth, doc_len, node_id)
```

Component 6 is the content-addressed sha256 PRIMARY KEY — **injective** over the candidate set. A tuple ending in an injective component is itself injective, which induces a **strict total order**. Therefore `sorted()` yields the same output for *every* permutation of its input and sort stability is irrelevant.

Two consequences: exact ties within D **cannot occur** (so neural tie-breaking is unreachable and was removed from the design), and the `[:15]` cascade defect is a *symptom* of the missing tiebreak that disappears with it.

**Candidate generation:** `candidates = ⋃ postings(t)` over sorted terms — the **exact support** of BM25. Zero-overlap nodes score exactly 0, so nothing is lost. No `LIMIT`, no `ORDER BY` to forget. Truncation happens only after a total order exists.

**Float discipline:** `math.fsum` over deterministically ordered contributions, then immediate fixed-point quantisation `q(x) = floor(x · 10⁹ + 0.5)`.

## 3.4 Enforcement mechanisms

| Mechanism | What it prevents |
|---|---|
| Runtime postcondition in `merge()`, **on every query in production** | Silent subordination breakage |
| `Advisory[T]` box, `unwrap()` allowlisted to `drf.retrieval.merge` only | Advisory data reaching authoritative arithmetic anywhere |
| Static AST check — Stage 1 modules may not import neural, transitively | Accidental coupling |
| `@action` replay check: `inputs_hash → result_sha`, mismatch raises | A mislabelled "deterministic" action |
| Confidence coupled to label (deterministic → must be `None`) | Undeclared uncertainty |
| spec↔code bijection test, **both directions** | Spec drift |
| `Template.substitute` (never `safe_substitute`) | A number entering docs without a producer |

---

# PART IV — PLAN

| Step | Deliverable | Proves | Status |
|---|---|---|---|
| **M1.0** | `hashing` `fixed` `contract` `spec/actions.json` | Hash stable across `PYTHONHASHSEED`; mislabelled action raises; advisory firewall holds; spec↔code bijection | ✅ **DONE** 23/23 |
| **M1.1** | `store.py`, `ingest/*`, `drf build` | Two builds → identical `content_hash`; counts via `len()` match `SELECT count(*)`; 48 dup edge groups collapse; 4 orphans in `manifest.dropped`; zero `AUTOINCREMENT`, zero `CURRENT_TIMESTAMP` | ⬜ **NEXT** |
| **M1.2** | `tokenize` `bm25` `lexical` | BM25 matches hand-computed reference **incl. length normalisation**; `len(candidates) == len(posting_union)`; all scores `int` | ⬜ |
| **M1.3** | `graph.py`, `stage1.py` | **`len(set(sort_keys)) == len(D)`** (injectivity); 50 shuffles → byte-identical; `seed_count` change is live | ⬜ |
| **M1.4** | `neural` `providers` `merge` | **Centerpiece.** Subordination vs adversarial/crashing/hanging/flood; `discordant_pairs == 0` | ⬜ |
| **M1.5** | `config/*` | `content_hash` ignores display keys; ill-typed key rejected; snapshot binds to manifest | ⬜ |
| **M1.6** | `bench/*`, query set | ReproRAG suite × 5 runs × 3 subprocesses × 3 hash seeds × 2 builds; every knob proven live | ⬜ |
| **M1.7** | `docs/render.py` | Zero unresolved placeholders; **hand-edit fails the test** | ⬜ |
| **M1.8** | Freeze | Tag with `spec_sha` + manifest hash + bench sha | ⬜ |

**M1.3 must precede M1.4** — subordination is only meaningful against an already-total order.

## Built so far

```
drf/version.py    version constants participating in hashing
drf/hashing.py    canonical_json, sha256_value, content_id, node_id, edge_id
drf/fixed.py      exact_sum, quantize/unquantize, qmul  (QUANTUM_EXP = 9)
drf/contract.py   @action, Justification, ActionOutput/Result, Advisory[T],
                  ACTIONS registry, replay check, strict_replay, Trace
spec/actions.json 10 actions × determinism × authority, with evidence
tests/test_contract.py   23 tests green
```

## Scope decisions

**Reversed after measurement:** `RemoteHTTPProvider` was slated to be cut. The endpoint proved live, fast, and deterministic, and re-embedding 266 nodes costs ~30 s — so `drf embed --provider bge` is **in** for M1. It remains advisory.

**Never cut:** content-addressed node IDs (they *are* the tiebreak key), the injective 6-tuple sort key, `merge()`'s runtime postcondition, `AdversarialProvider` and its test. Those four are milestone 1.

**Known limits, recorded not hidden:** 266 nodes is a toy — every metric will be 1.0, proving determinism, not scalability. Posting-union has no truncation and degrades at scale (FTS5 with mandatory `ORDER BY rank` + deterministic pagination is the M3 path). An all-OOV query yields empty D and therefore no proposals — correct under subordination; the fix if wanted is a **Stage 1** fix (character n-grams), never a neural one. **That pressure is the most likely future threat to the invariant.**

---

# PART V — WORKING DISCIPLINE

1. **No number without a producer.** Every metric maps to the script that emits it; CI regenerates. If no script can produce it, it is a hypothesis and is labelled one.
2. **Counts derived, never asserted.** `len()`, not a hand-written integer.
3. **Assert exact integers, never floats.** `discordant_pairs == 0`, not `tau == 1.000`. Report floats for humans; assert on integers.
4. **Docs generated from `spec/`, never hand-written.** A hand-edit must fail a test.
5. **Don't assert byte-identical `.db` files** — SQLite's header change counter makes that fragile. Assert manifest `content_hash` + canonical export.
6. **Scope honesty in every claim.** M1 has **no relevance labels** and therefore makes **no claim about retrieval quality**. An unqualified "all metrics 1.0" would be exactly the drift this project exists to prevent.
7. **Verify before claiming — including my own claims.** See the corrections log.

---

# APPENDIX — Corrections log

Claims that failed verification during this session. Kept because the failure mode is the subject matter.

| Claim | Source | Outcome |
|---|---|---|
| "`sum()` is order-dependent, so we need `fsum`" | **Me**, in `fixed.py` | **False on CPython 3.12** — Neumaier compensation; **0 disagreements in 200,000 random 12-element sums**, no order dependence in either. `fsum` kept for the *documented language guarantee* that survives a change of interpreter. Docstring corrected. |
| "BGE-large served from your phone" | A design subagent | **Confabulated** — the user had not said it. Later turned out true, which is the dangerous kind of wrong. |
| "125-character limit on quotations" | A WebFetch sub-model | **Invented.** No such limit exists. It was persuasive enough to change behaviour — the same mechanism that let `80% vs 60%` survive three documentation passes. |
| "Only 5.7% of rubrics are machine-readable" | **Me**, first audit pass | **My regex was broken** (`"30%+"` unmatched). Fixed → 7.1%, then 21% after authoring. |
| "`SEMANTIC-DIMENSIONS-REFERENCE.md`'s 50 dims may be an extraction artifact" | **Me**, hedging | **Verified correct** — it really is a different vocabulary sharing only 10 names. |
| "The production generator uses the `fresh/` 50-dim set" | **Me**, hypothesis | **False** — it's a 100% subset of the canonical 56. Checking paid off. |
| Corpus is 1051 modules / 13,946 functions | **Me**, first measurement | **Contaminated** — 856 of 1023 unique files were `venv/` third-party code. Re-scoped to 167 project modules; kept venv as a *separate* second corpus, which strengthened the result. |

---

*Numbers in this document were produced by direct inspection on 2026-08-07. Where a figure is quoted from a paper, the paper is named. Where a figure was measured here, the protocol is stated.*
