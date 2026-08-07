"""The single tokenizer definition, shared by the index and query paths.

If indexing and querying could tokenize differently, a term present in a
document would be unfindable by the query that contains it - and nothing in
the ranking layer could detect the problem. `tokenize()` is therefore the only
way text becomes terms anywhere in the framework, and
`tests/test_retrieval.py` asserts the two paths agree.

**ASCII by construction.** The pattern is `[a-z0-9]+` over `text.lower()`,
not `\\w+`. `\\w` under Unicode matches according to character properties from
the Unicode database bundled with the interpreter, and that database changes
between Python releases (this build carries 15.0.0). A tokenizer whose output
depends on the interpreter's Unicode version is not portable, and portability
is the entire point of the surrounding machinery.

Measured cost of that choice on this corpus: the only non-ASCII characters
present are `×` and `→`, in 5 of 266 documents. Both are symbols carrying no
search value, so nothing findable is lost. A corpus with genuine non-ASCII
*words* would need a different decision and a `PARSER_VERSION` bump.

**Deliberately absent:**

* *Stemming* - it would improve recall, but every implementation is a library
  whose behaviour changes across versions, which is precisely the dependency
  this module refuses. It is a quality knob, not a correctness one, and it
  belongs to a milestone that measures quality.
* *A stopword list* - IDF already suppresses ubiquitous terms, and by a
  measured property of the corpus rather than by a hand-written list whose
  contents are arbitrary.
* *A minimum token length* - "ai", "go" and "k" are meaningful here.

Compound words split: `real-time` becomes `real` and `time` (148 distinct
hyphen or underscore compounds exist in the corpus). This is the standard
choice and it makes the query "real time" match; keeping compounds whole would
require the query to reproduce the punctuation exactly.

Stdlib only.
"""

import re

from ..contract import ActionOutput, action

# Compiled once. Matches runs of ASCII lowercase letters and digits, applied
# after lowering, so `PDF-vision` yields ["pdf", "vision"].
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Text to an ordered term list. The one definition.

    Order is preserved because callers count term frequency; nothing here
    sorts, and nothing downstream depends on the order beyond counting.
    """
    return TOKEN_PATTERN.findall(text.lower())


def document_text(name: str, description: str) -> str:
    """The indexed document for a node.

    Decided at M1.2 from measurement, and bound to `PARSER_VERSION`:

    * `source_ref` is **excluded**. 71% of its tokens already appear in
      name+description and 84 of 266 documents gain nothing from it, so it
      mostly duplicates title terms - and a duplicate raises `tf`, silently
      double-weighting titles.
    * `type` is **excluded**. 29 of its 30 tokens are already in the corpus
      vocabulary, and 101 documents share `use_case`, so its document
      frequency is high and IDF crushes whatever it contributes. It shifts
      `avgdl` without buying discrimination. Type is a structured filter,
      not text.
    """
    return f"{name} {description}"


@action(
    "tokenize.terms",
    determinism="deterministic",
    authority="authoritative",
    inputs=("text",),
)
def terms(*, text: str) -> ActionOutput:
    """Audited entry point for the query path.

    Indexing calls `tokenize()` directly - 266 documents through the contract
    would pay input-hashing costs for no audit value, since the build is
    already a single audited action. Queries go through here because a query
    is a user-facing operation whose justification is worth recording.

    Both paths call the same `tokenize()`, which is what makes the spec's
    guarantee true rather than merely stated.
    """
    return ActionOutput(
        value=tokenize(text),
        evidence=(f"pattern={TOKEN_PATTERN.pattern}",),
    )
