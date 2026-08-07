"""The reproducibility matrix, and the control that stops it being vacuous.

Results are compared across four independent axes:

    5 in-process repeats      does state leak between calls?
    3 subprocess runs         does anything depend on interpreter state?
    3 PYTHONHASHSEED values   does set or dict iteration order reach output?
    2 independent builds      is the index itself reproducible?

Every cell must produce the same digest. On this pipeline they do, and that
is exactly why the numbers alone are not evidence: a harness that compared
nothing would report the same perfect score.

**The chaos control.** `chaos_run()` reproduces the defect this framework was
built to remove - rank by score alone, with no tiebreak, over a candidate list
in arbitrary order. That is precisely
`python_apps_hybrid_query.py:304` (`combined_scores.sort(key=lambda x: -x[1])`)
applied to a set of candidates that arrived unordered. It is not artificial
noise; it is the previous implementation. If the harness cannot tell that
apart from the real pipeline, the harness is broken, and `bench repro`
reports both so the comparison is visible rather than asserted.

Stdlib only.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

from ..hashing import sha256_value
from ..retrieval import stage1
from ..retrieval.tokenize import tokenize
from ..store import connect, read_manifest
from . import metrics

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_QUERY_FILE = ROOT / "queries" / "milestone1.jsonl"

# PYTHONHASHSEED values used for the subprocess axis. Fixed rather than random
# so a failure is reproducible by re-running the same command.
HASH_SEEDS = ("0", "1", "12345")


def load_queries(path: Path | None = None) -> list[dict]:
    with open(path or DEFAULT_QUERY_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def rank_ids(conn, index_hash: str, text: str, **kwargs) -> list[str]:
    value, _ = stage1.rank(
        conn=conn, query_terms=tokenize(text), index_hash=index_hash, **kwargs
    )
    return [stage1.Ranked(*row).node_id for row in value]


def rank_scores(conn, index_hash: str, text: str, **kwargs) -> list[int]:
    value, _ = stage1.rank(
        conn=conn, query_terms=tokenize(text), index_hash=index_hash, **kwargs
    )
    return [stage1.Ranked(*row).bm25_q for row in value]


def run_all(conn, index_hash: str, queries: list[dict], **kwargs) -> dict[str, list[str]]:
    return {q["id"]: rank_ids(conn, index_hash, q["text"], **kwargs) for q in queries}


def digest(results: dict[str, list[str]]) -> str:
    return sha256_value(results)


def chaos_run(conn, index_hash: str, queries: list[dict], seed: int) -> dict[str, list[str]]:
    """The prior engine's ranking: score only, no tiebreak, unordered input.

    Reproduces `python_apps_hybrid_query.py:304`. Candidates are shuffled
    first because that engine's candidate list came from an unordered
    `LIMIT 100` (`:342`), so its input order was not controlled either.

    Python's `sorted` is stable, so shuffling plus a non-injective key is
    exactly what makes the output vary - which is the defect. This is not a
    strawman: it is what the measured recall numbers in the recovered project
    were computed on.
    """
    rng = random.Random(seed)
    out: dict[str, list[str]] = {}
    for query in queries:
        value, _ = stage1.rank(
            conn=conn, query_terms=tokenize(query["text"]), index_hash=index_hash
        )
        rows = [stage1.Ranked(*row) for row in value]
        rng.shuffle(rows)
        rows.sort(key=lambda r: -r.bm25_q)   # score only - no tiebreak
        out[query["id"]] = [r.node_id for r in rows]
    return out


_SUBPROCESS_SCRIPT = """
import json, sys
sys.path.insert(0, {root!r})
from drf.bench.repro import run_all, digest, load_queries
from drf.store import connect, read_manifest
conn = connect({index!r})
index_hash = read_manifest(conn)["content_hash"]
print(digest(run_all(conn, index_hash, load_queries())))
"""


def subprocess_digest(index_path: str, hash_seed: str) -> str:
    """Run the whole query set in a fresh interpreter under a given hash seed."""
    script = _SUBPROCESS_SCRIPT.format(root=str(ROOT), index=index_path)
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, cwd=str(ROOT),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"subprocess failed:\n{completed.stderr}")
    return completed.stdout.strip()


def run_matrix(
    index_paths: list[str],
    *,
    queries: list[dict] | None = None,
    in_process_repeats: int = 5,
    subprocess_repeats: int = 3,
    hash_seeds: tuple[str, ...] = HASH_SEEDS,
) -> dict:
    """Every axis, every cell. Returns integers to assert and floats to read."""
    queries = queries or load_queries()
    digests: list[str] = []
    baseline: dict[str, list[str]] | None = None
    comparisons: list[metrics.Comparison] = []
    cells = 0

    for index_path in index_paths:
        conn = connect(index_path)
        index_hash = read_manifest(conn)["content_hash"]

        for _ in range(in_process_repeats):
            results = run_all(conn, index_hash, queries)
            if baseline is None:
                baseline = results
            for query_id, ids in results.items():
                comparisons.append(metrics.compare(baseline[query_id], ids))
            digests.append(digest(results))
            cells += 1
        conn.close()

        for seed in hash_seeds:
            for _ in range(subprocess_repeats):
                digests.append(subprocess_digest(index_path, seed))
                cells += 1

    summary = metrics.aggregate(comparisons)
    summary.update({
        "cells": cells,
        "indexes": len(index_paths),
        "queries": len(queries),
        "distinct_digests": len(set(digests)),
        "digest": digests[0] if digests else None,
        "axes": {
            "in_process_repeats": in_process_repeats,
            "subprocess_repeats": subprocess_repeats,
            "hash_seeds": list(hash_seeds),
            "independent_builds": len(index_paths),
        },
    })
    return summary


def run_chaos_control(index_path: str, *, queries: list[dict] | None = None,
                      runs: int = 5) -> dict:
    """The same measurement against the defect, so the harness proves itself."""
    queries = queries or load_queries()
    conn = connect(index_path)
    index_hash = read_manifest(conn)["content_hash"]

    runs_out = [chaos_run(conn, index_hash, queries, seed) for seed in range(runs)]
    conn.close()

    baseline = runs_out[0]
    comparisons = [
        metrics.compare(baseline[query_id], other[query_id])
        for other in runs_out[1:]
        for query_id in baseline
    ]
    summary = metrics.aggregate(comparisons)
    summary["distinct_digests"] = len({digest(r) for r in runs_out})
    summary["runs"] = runs
    return summary


def run_sensitivity(index_path: str, *, queries: list[dict] | None = None) -> dict:
    """Every setting flagged `affects_ranking` must reorder something.

    Probe values come from `spec/config_schema.json`, so the claim that a
    given value moves the order is recorded where a reader can find it rather
    than encoded in test arithmetic.
    """
    from ..config.manager import SCHEMA, Config, ranking_keys

    queries = queries or load_queries()
    conn = connect(index_path)
    index_hash = read_manifest(conn)["content_hash"]
    baseline_config = Config()
    baseline = {
        q["id"]: rank_ids(conn, index_hash, q["text"],
                          **baseline_config.action_kwargs())
        for q in queries
    }

    findings = {}
    for key in ranking_keys():
        probe = SCHEMA[key].get("sensitivity_probe")
        alternative = Config()
        alternative.set(key, probe)
        changed_order = 0
        for query in queries:
            ids = rank_ids(conn, index_hash, query["text"],
                           **alternative.action_kwargs())
            if ids != baseline[query["id"]]:
                changed_order += 1
        findings[key] = {
            "default": SCHEMA[key]["default"],
            "probe": probe,
            "queries_reordered": changed_order,
            "live": changed_order > 0,
        }
    conn.close()
    return {"queries": len(queries), "settings": findings}
