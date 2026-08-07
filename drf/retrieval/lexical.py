"""Candidate generation and scoring: the audited lexical layer.

Candidates are the **union of postings** over the distinct query terms - the
exact support of BM25. A document containing no query term scores exactly
zero, so nothing with a nonzero score can lie outside the union. There is no
`LIMIT` and no `ORDER BY` on candidate generation, which means there is no
ordering dependency to get wrong.

That is a deliberate contrast with the engine this replaces, which carried
three independent silent caps: an unordered `LIMIT 100`
(`python_apps_hybrid_query.py:347`), a query truncated to its first ten terms
(`:339`), and a `[:15]` boundary where a single tie cascaded into global
reordering (`:307`). Truncation is legitimate only *after* a strict total
order exists - which is M1.3, not here.

Stdlib only.
"""

import collections

from ..contract import ActionOutput, action
from ..store import (
    PostingRecord,
    corpus_totals,
    df_for_terms,
    doc_lengths,
    postings_for_terms,
)
from . import bm25
from .tokenize import document_text, tokenize


def build_index_tables(
    nodes,
) -> tuple[dict[str, int], list[PostingRecord], dict[str, int]]:
    """Tokenise every document and build the inverted index.

    Returns `(doc_lens, postings, dfs)`. All three are derived from one pass
    over one tokenisation of each document, so `doc_len`, `tf` and `df` cannot
    disagree with each other - a second pass could.

    `df` counts *documents*, not occurrences: a term appearing three times in
    one document contributes 1 to df and 3 to that document's tf.
    """
    doc_lens: dict[str, int] = {}
    postings: list[PostingRecord] = []
    dfs: collections.Counter = collections.Counter()

    for node in nodes:
        terms = tokenize(document_text(node.name, node.description))
        doc_lens[node.id] = len(terms)
        frequencies = collections.Counter(terms)
        for term in sorted(frequencies):
            postings.append(
                PostingRecord(term=term, node_id=node.id, tf=frequencies[term])
            )
        dfs.update(frequencies.keys())

    return doc_lens, sorted(postings), dict(dfs)


@action(
    "lexical.candidates",
    determinism="deterministic",
    authority="authoritative",
    inputs=("query_terms", "index_hash"),
)
def candidates(*, conn, query_terms: list[str], index_hash: str) -> ActionOutput:
    """The union of postings over the distinct query terms.

    `index_hash` is part of the declared inputs and is not decorative. The
    contract's replay check keys on the hash of declared inputs, so without an
    index identity two *different* indexes queried with the same terms would
    collide on one replay-log entry and the second would raise
    DeterminismViolation for being correctly different. Including it makes the
    key say what the result is actually a function of: (query, index).

    An empty result is returned for an all-out-of-vocabulary query rather than
    an error. That is the correct behaviour under subordination: no
    authoritative candidates means nothing for a neural layer to be appended
    below, and the fix - if one is ever wanted - is a Stage 1 fix such as
    character n-grams, never a neural one.
    """
    posting_lists = postings_for_terms(conn, query_terms)
    union: set[str] = set()
    for term_postings in posting_lists.values():
        union.update(term_postings)
    return ActionOutput(
        value=sorted(union),
        evidence=(
            f"terms={len(set(query_terms))}",
            f"matched_terms={len(posting_lists)}",
            f"candidates={len(union)}",
        ),
    )


@action(
    "lexical.bm25_score",
    determinism="deterministic",
    authority="authoritative",
    inputs=("query_terms", "index_hash", "k1", "b"),
)
def bm25_score(
    *,
    conn,
    query_terms: list[str],
    index_hash: str,
    k1: float = bm25.K1,
    b: float = bm25.B,
) -> ActionOutput:
    """Score the candidate set with Okapi BM25.

    `k1` and `b` are declared inputs so that changing a ranking parameter
    produces a different replay key rather than a spurious determinism
    violation. A parameter that changes the output must change the identity of
    the computation; otherwise the contract would be asserting that two
    genuinely different calculations should agree.

    Returns a list of `Scored` tuples as plain lists, ordered by node_id. This
    is an enumeration order, not a ranking - ranking is M1.3.
    """
    posting_lists = postings_for_terms(conn, query_terms)
    n_docs, total_len = corpus_totals(conn)
    scored = bm25.score_documents(
        postings=posting_lists,
        dfs=df_for_terms(conn, query_terms),
        doc_lens=doc_lengths(conn),
        n_docs=n_docs,
        total_len=total_len,
        k1=k1,
        b=b,
    )
    return ActionOutput(
        value=[list(s) for s in scored],
        evidence=(
            f"scored={len(scored)}",
            f"k1={k1}",
            f"b={b}",
            f"avgdl={total_len}/{n_docs}",
        ),
    )
