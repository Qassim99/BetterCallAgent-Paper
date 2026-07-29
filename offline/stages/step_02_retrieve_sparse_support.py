"""Stage 2: aggregate citation support from saved SIRA/BM25 trace files.

This stage deliberately does not claim to recreate the unpublished SIRA retriever.
It turns its saved, ranked evidence into a deterministic support signal that can be
audited and replayed independently of the dense candidate retrieval.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from bettercallagent.citations.extract import CitationExtractor
from bettercallagent.citations.vocabulary import (
    CitationVocabulary,
)
from offline.io import JsonObject, atomic_write_jsonl, load_queries, normalize_text
from offline.vocabulary import load_targeted_vocabulary


def balanced_hits(trace: Mapping[str, Any], limit: int) -> list[JsonObject]:
    """Interleave one law hit and two court hits, matching the reported run."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    raw_laws = trace.get("candidate_law_hits") or []
    raw_courts = trace.get("candidate_court_hits") or []
    if not isinstance(raw_laws, list) or not isinstance(raw_courts, list):
        raise ValueError("Trace hit collections must be JSON arrays.")
    laws = [dict(hit) for hit in raw_laws if isinstance(hit, dict)]
    courts = [dict(hit) for hit in raw_courts if isinstance(hit, dict)]
    if len(laws) != len(raw_laws) or len(courts) != len(raw_courts):
        raise ValueError("Every sparse hit must be a JSON object.")

    output: list[JsonObject] = []
    law_index = 0
    court_index = 0
    while len(output) < limit and (law_index < len(laws) or court_index < len(courts)):
        if law_index < len(laws):
            output.append(laws[law_index])
            law_index += 1
        for _ in range(2):
            if len(output) >= limit or court_index >= len(courts):
                break
            output.append(courts[court_index])
            court_index += 1
    return output[:limit]


def _hit_text(hit: Mapping[str, Any]) -> str:
    return "\n".join(
        value
        for value in (
            normalize_text(hit.get("citation")),
            normalize_text(hit.get("title")),
            normalize_text(hit.get("source")),
            str(hit.get("text") or ""),
        )
        if value
    )


def citations_for_hit(
    hit: Mapping[str, Any],
    *,
    vocabulary: CitationVocabulary,
    extractor: CitationExtractor,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return accepted and out-of-vocabulary extracted citations separately."""

    extracted = set(extractor.extract(_hit_text(hit)))
    exact = normalize_text(hit.get("citation"))
    if exact:
        extracted.add(exact)
    accepted = tuple(sorted(citation for citation in extracted if citation in vocabulary))
    rejected = tuple(sorted(set(extracted) - set(accepted)))
    return accepted, rejected


def _trace_paths(trace_root: Path) -> list[Path]:
    paths = sorted(trace_root.glob("*/trace.json"))
    if not paths:
        raise FileNotFoundError(f"No <query_id>/trace.json files under {trace_root}")
    return paths


def discover_trace_citations(
    *,
    trace_root: Path,
    query_ids: Iterable[str],
    top_k: int,
) -> set[str]:
    """Extract the small candidate set needed for targeted vocabulary scanning."""

    expected = set(query_ids)
    discovered_queries: set[str] = set()
    extractor = CitationExtractor()
    candidates: set[str] = set()
    for path in _trace_paths(trace_root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected a JSON object")
        query_id = str(payload.get("query_id") or "").strip()
        if not query_id or query_id in discovered_queries:
            raise ValueError(f"Duplicate or empty sparse trace query_id {query_id!r}.")
        discovered_queries.add(query_id)
        for hit in balanced_hits(payload, top_k):
            candidates.update(extractor.extract(_hit_text(hit)))
            if exact := normalize_text(hit.get("citation")):
                candidates.add(exact)
    if discovered_queries != expected:
        missing = sorted(expected - discovered_queries)
        extra = sorted(discovered_queries - expected)
        raise ValueError(f"Sparse trace query IDs differ: missing={missing}, extra={extra}")
    return candidates


def aggregate_saved_traces(
    *,
    trace_root: Path,
    query_ids: Iterable[str],
    vocabulary: CitationVocabulary,
    top_k: int,
) -> list[JsonObject]:
    """Aggregate exact per-hit support counts for every expected query."""

    expected = tuple(query_ids)
    if len(expected) != len(set(expected)):
        raise ValueError("Expected query IDs must be unique.")
    traces: dict[str, Mapping[str, Any]] = {}
    for path in _trace_paths(trace_root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected a JSON object")
        query_id = str(payload.get("query_id") or "").strip()
        if not query_id:
            raise ValueError(f"{path}: missing query_id")
        if query_id in traces:
            raise ValueError(f"Duplicate sparse trace for query {query_id!r}")
        traces[query_id] = payload
    missing = sorted(set(expected) - set(traces))
    extra = sorted(set(traces) - set(expected))
    if missing or extra:
        raise ValueError(f"Sparse trace query IDs differ: missing={missing}, extra={extra}")

    extractor = CitationExtractor()
    output: list[JsonObject] = []
    for query_id in expected:
        counts: Counter[str] = Counter()
        serialized_hits: list[JsonObject] = []
        for rank, hit in enumerate(balanced_hits(traces[query_id], top_k), start=1):
            accepted, rejected = citations_for_hit(
                hit,
                vocabulary=vocabulary,
                extractor=extractor,
            )
            counts.update(accepted)
            serialized_hits.append(
                {
                    "rank": rank,
                    "source": normalize_text(hit.get("source")),
                    "score": hit.get("score"),
                    "declared_citation": normalize_text(hit.get("citation")),
                    "citations": list(accepted),
                    "out_of_vocabulary": list(rejected),
                }
            )
        output.append(
            {
                "query_id": query_id,
                "method": "saved_sira_bm25_balanced_1_law_2_court",
                "top_k": top_k,
                "support_counts": dict(sorted(counts.items())),
                "hits": serialized_hits,
            }
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--laws", type=Path, required=True)
    parser.add_argument("--courts", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    queries = load_queries(args.queries)
    candidates = discover_trace_citations(
        trace_root=args.trace_root,
        query_ids=queries,
        top_k=args.top_k,
    )
    vocabulary = load_targeted_vocabulary(
        laws_path=args.laws,
        courts_path=args.courts,
        candidates=candidates,
    )
    rows = aggregate_saved_traces(
        trace_root=args.trace_root,
        query_ids=queries,
        vocabulary=vocabulary,
        top_k=args.top_k,
    )
    atomic_write_jsonl(args.output, rows)
    print(f"Aggregated sparse support for {len(rows)} queries -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
