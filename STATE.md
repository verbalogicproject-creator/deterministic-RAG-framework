# Project State — resume here

**Last checkpoint:** M1.0 complete, 23/23 tests green (2026-08-07)
**Full build plan:** `/home/eyaln/.claude/plans/plan-step-1-out-crystalline-firefly.md` — read this first, it has the complete M1.0–M1.8 sequence, architecture, and verification steps.

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

`http://10.161.19.239:8080` — llama.cpp serving **BAAI/bge-large-en-v1.5 Q4_0 GGUF** from an Android path.

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

## Next: M1.2.1–M1.2.4 — `tokenize.py`, `bm25.py`, `lexical.py`

**Document definition (decided, binds `PARSER_VERSION`): `name + description`.** `source_ref` excluded — 71% of its tokens already present, 84/266 docs gain zero, and duplicates inflate `tf` on title terms. `type` excluded — 29/30 tokens already in vocabulary and 101 docs share `use_case`, so IDF crushes it; type belongs in a structured filter.

Unchanged from the original plan: BM25 must match a hand-computed 5-doc reference; `len(candidates) == len(posting_union)`; all scores are `int`. Four changes follow.

### Corpus facts, measured (`tools/measure_length_norm.py`)

```
N=266  avgdl=18.9 tokens  min=4  median=17  max=61   (name + description)
0 empty descriptions -> avgdl well-defined
5 isolated nodes -> score on text alone
```

The old engine's missing `b`/`avgdl` was **not** a harmless omission — measured against `b=0.75` over 15 corpus-vocabulary queries:

| | |
|---|---|
| discordant pairs @10 | **109/416 = 26.2%** |
| queries whose top-1 changed | **9/15** |
| mean top-1 doc length, `b=0.75` | 15.9 tokens |
| mean top-1 doc length, `b=0.0` | 28.1 tokens |
| `b=0` top-1 longer / shorter / same | **9 / 0 / 6** |

The direction claim holds *monotonically* — zero counterexamples. This replaces the previously-asserted "long padded entities win by default" with a producer-backed figure.

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
