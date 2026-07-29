"""Stage 4: join queries, dense candidates, and full document text."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from offline.identity import candidate_id
from offline.io import (
    JsonObject,
    atomic_write_jsonl,
    load_queries,
    read_jsonl,
    require_file,
)
from offline.stages.step_03_materialize_documents import document_reference


def load_summary(path: Path | None) -> dict[str, JsonObject]:
    if path is None:
        return {}
    resolved = require_file(path, description="retrieval summary")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{resolved}: expected a JSON array")
    output: dict[str, JsonObject] = {}
    for record in payload:
        if not isinstance(record, dict) or not record.get("query_id"):
            raise ValueError(f"{resolved}: invalid summary record")
        output[str(record["query_id"])] = record
    return output


def load_documents(path: Path) -> dict[str, JsonObject]:
    """Index a materialized cache by immutable document reference only."""

    by_reference: dict[str, JsonObject] = {}
    for document in read_jsonl(path):
        reference = str(document.get("doc_id") or "")
        if not reference:
            raise ValueError("Every materialized document requires a doc_id.")
        if reference in by_reference:
            raise ValueError(f"Duplicate materialized doc_id {reference!r}")
        by_reference[reference] = document
    if not by_reference:
        raise ValueError("Document cache is empty.")
    return by_reference


def prepare(
    *,
    candidates_path: Path,
    queries_path: Path,
    documents_path: Path,
    summary_path: Path | None,
    top_n: int,
) -> list[JsonObject]:
    """Create a complete, ordered reranker input."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    queries = load_queries(queries_path)
    summary = load_summary(summary_path)
    by_reference = load_documents(documents_path)

    grouped: dict[str, list[JsonObject]] = defaultdict(list)
    for candidate in read_jsonl(candidates_path):
        query_id = str(candidate.get("query_id") or "")
        if query_id not in queries:
            raise ValueError(f"Candidate references unknown query_id {query_id!r}")
        grouped[query_id].append(candidate)
    if set(grouped) != set(queries):
        missing = sorted(set(queries) - set(grouped))
        raise ValueError(f"No dense candidates for queries: {', '.join(missing)}")
    for candidates in grouped.values():
        candidates.sort(key=lambda row: int(row.get("rank") or 10**9))

    output: list[JsonObject] = []
    for query_id in queries:
        query_summary = summary.get(query_id, {})
        query = str(queries[query_id]["query"])
        summary_query = str(query_summary.get("query") or query)
        if summary_query != query:
            raise ValueError(f"Retrieval summary query text differs for query_id {query_id!r}.")
        for candidate in grouped[query_id][:top_n]:
            reference = document_reference(candidate)
            document = by_reference.get(reference)
            global_index = candidate.get("global_idx")
            text = str((document or {}).get("full_document_text") or "")
            if not text:
                raise ValueError(
                    f"Missing full text for query={query_id}, doc={reference}, "
                    f"global_idx={global_index}"
                )
            prepared = {
                "query_id": query_id,
                "query": query,
                "meta_query": str(query_summary.get("meta_query") or ""),
                "keywords_query": str(query_summary.get("keywords_query") or ""),
                "rank": int(candidate.get("rank") or 10**9),
                "global_idx": global_index,
                "doc_id": reference,
                "fusion_score": candidate.get("fusion_score"),
                "hits": candidate.get("hits") or [],
                "metadata": candidate.get("metadata") or {},
                "document_text": text,
                "document_text_length": len(text),
            }
            prepared["candidate_id"] = candidate_id(prepared)
            output.append(prepared)
        if len(grouped[query_id][:top_n]) != min(top_n, len(grouped[query_id])):
            raise AssertionError("Unexpected candidate slicing result.")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = prepare(
        candidates_path=args.candidates,
        queries_path=args.queries,
        documents_path=args.documents,
        summary_path=args.summary,
        top_n=args.top_n,
    )
    atomic_write_jsonl(args.output, rows)
    print(f"Prepared {len(rows)} reranker records -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
