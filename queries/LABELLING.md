# M2.1 — labelling

## The short version

```bash
python3 tools/drf build --source <source.db> --out index.db     # if not built
python3 tools/make_labelling_worksheet.py --index index.db      # regenerate
$EDITOR queries/labels.worksheet.jsonl                          # set "grade"
python3 tools/labels_collect.py --stratum A_advisory            # collect
python3 tools/drf eval quality --index index.db                 # measure
```

Read `queries/labels.worksheet.md` while deciding — same rows, laid out for
humans. Grade in the **JSONL**; the markdown is regenerated and is not read
back.

**Grades.** `0` irrelevant · `1` marginal, touches the topic but would not
satisfy the query · `2` relevant, a reasonable answer · `3` vital, the document
the query is really asking for. Recall and precision count **2 and above**.

`D`*n* is authoritative rank *n*. `A`*n* is an advisory proposal that is
**absent from D entirely** — grading those is the whole point of stratum A.

A `null` grade is *invalid*, and `labels_collect.py` refuses to write a partial
file. A template pre-filled with `0` would turn a forgotten row into a
confident judgement of "irrelevant", and a partial file would shrink the recall
denominator — which *raises* the measured score.

---

## What each stratum can settle, and what it cannot

The selection is the experiment. Fifty arbitrary judgements settle nothing;
fifty chosen ones settle a stated question. The strata come from a measurement:
running every `affects_ranking` setting against its probe across the 23-query
set shows which queries each decision actually depends on.

| stratum | judgements | queries | settles |
|---|---|---|---|
| **A_advisory** | 49 | 7 | Does the neural layer find what lexical search missed? |
| **B_length_normalisation** | 40 | 4 | Does BM25's `b` improve relevance, or only change it? |
| **C_graph** | 20 | 1 | ⚠️ **Cannot be settled — see below** |

**Stratum A is complete on its own** and is the one to do first. It is also
almost exactly the ~50 judgements the M2 plan budgeted.

### ⚠️ Stratum C cannot answer its question, and more labels will not help

The open decision is *"does the graph layer earn its place in the sort key?"*
Measured across all 23 queries, the graph settings reorder results for exactly
two of them — `q02` and `e06` — and `e06` is a synthetic 13-term edge case
built to stress the tokenizer, not a query anyone would type.

**So the real sample size is one.** Judging `q02` exhaustively produces one
data point. That is an anecdote, and reporting it as a finding about the graph
layer would be exactly the drift this project exists to remove.

The fix is **more queries, not more labels per query** — queries chosen to
stress graph structure (documents that are lexically thin but well connected).
That is a change to `queries/`, and it belongs to M2.2. Stratum C is included
for the qualitative read only: if `q02`'s graph-driven reordering is visibly
wrong, that is worth knowing early even though it proves nothing.

### The strata are disjoint, and that is a fact about the corpus

Low-`|D|` queries are the only ones the advisory layer can act on — and they
are inert to *every* ranking parameter, because a handful of candidates leaves
a weight nothing to reorder. High-`|D|` queries are parameter-sensitive and
advisory-mute.

**No single query can serve both purposes here.** Worth knowing before
assuming one afternoon of judgements answers everything.

---

## Pooling, and how it can mislead

Candidates are pooled from D **and** from the advisory provider. This is
standard pooled assessment, and it is load-bearing: a pool drawn only from D
could never contain a document that lexical retrieval missed, so recall would
be measured against a denominator that excluded the exact thing the advisory
layer exists to find.

Pool depths are recorded per stratum in `tools/make_labelling_worksheet.py`.
Stratum A pools the top **3** advisory proposals per query, which is shallow —
the provider returns up to 100. Anything unjudged counts as grade 0, so:

> **A shallow pool silently flatters recall.** If stratum A shows the advisory
> layer contributing little, check whether the pool was deep enough before
> concluding anything about the provider.

Deepening it is one edit to `pool_advisory` and a re-run.

---

## Grading advice

- **Judge the document against the query, not against the other candidates.**
  Relevance is not a ranking. If four documents are all vital, grade all four
  `3` — the metric handles it, and forcing a spread invents information.
- **Use `1` freely.** On a corpus of closely related Claude API documents,
  "touches the topic but would not satisfy the query" is the honest answer
  often, and it is the grade that stops everything collapsing to relevant.
- **A disagreement with yourself is data.** If a second sitting grades a pair
  differently, `labels_collect.py` raises rather than picking a winner. That is
  deliberate: it means the query is ambiguous, and averaging destroys the only
  evidence of it. Resolve it by deciding, then edit the row.
- **Write the `note` when the call is close.** It costs a sentence now and is
  the only thing that will explain a surprising number later.

## What happens to the labels

`labels_collect.py` strips the `_`-prefixed helper keys and writes
`queries/labels.jsonl`, reporting a `labels_hash`. Every quality figure the
project ever publishes carries that hash, because a quality number is a
statement about a specific set of judgements — edit one grade and every nDCG
changes with no commit touching any code.
