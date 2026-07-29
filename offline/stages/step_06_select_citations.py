"""Stage 6: apply the fixed dense-vote gate plus sparse citation support."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from bettercallagent.citations.extract import CitationExtractor
from bettercallagent.citations.policy import (
    FixedVotePolicy,
    order_candidates,
    select_citations,
)
from bettercallagent.citations.vocabulary import (
    CitationVocabulary,
    CsvCitationVocabulary,
)
from bettercallagent.schemas import RankedCandidate
from offline.identity import candidate_id
from offline.io import JsonObject, atomic_write_json, load_queries, read_jsonl


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    query_id = str(row.get("query_id") or "").strip()
    return query_id, candidate_id(row)


def _index_unique(rows: Iterable[JsonObject], *, label: str) -> dict[tuple[str, str], JsonObject]:
    output: dict[tuple[str, str], JsonObject] = {}
    for row in rows:
        key = _key(row)
        if key in output:
            raise ValueError(f"Duplicate {label} record for {key}.")
        output[key] = row
    if not output:
        raise ValueError(f"{label} input is empty.")
    return output


def _support_by_query(
    rows: Iterable[JsonObject],
    *,
    query_ids: set[str],
    vocabulary: CitationVocabulary,
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for row in rows:
        query_id = str(row.get("query_id") or "").strip()
        if not query_id or query_id in output:
            raise ValueError(f"Duplicate or empty sparse support query_id {query_id!r}.")
        raw_counts = row.get("support_counts")
        if not isinstance(raw_counts, dict):
            raise ValueError(f"{query_id}: support_counts must be an object.")
        counts: dict[str, int] = {}
        for raw_citation, raw_count in raw_counts.items():
            citation = str(raw_citation)
            count = int(raw_count)
            if citation not in vocabulary:
                raise ValueError(f"{query_id}: sparse support is out of vocabulary: {citation}")
            if count < 0:
                raise ValueError(f"{query_id}: sparse support count cannot be negative.")
            counts[citation] = count
        output[query_id] = counts
    missing = sorted(query_ids - set(output))
    extra = sorted(set(output) - query_ids)
    if missing or extra:
        raise ValueError(f"Sparse support query IDs differ: missing={missing}, extra={extra}")
    return output


def select_for_queries(
    *,
    query_ids: Iterable[str],
    prepared_rows: Iterable[JsonObject],
    score_rows: Iterable[JsonObject],
    support_rows: Iterable[JsonObject],
    vocabulary: CitationVocabulary,
    policy: FixedVotePolicy,
) -> tuple[dict[str, tuple[str, ...]], list[JsonObject]]:
    """Join by document identity and return predictions plus a complete audit."""

    ordered_query_ids = tuple(query_ids)
    if len(ordered_query_ids) != len(set(ordered_query_ids)):
        raise ValueError("Query IDs must be unique.")
    query_set = set(ordered_query_ids)
    prepared = _index_unique(prepared_rows, label="prepared candidate")
    scores = _index_unique(score_rows, label="reranker score")
    if set(prepared) != set(scores):
        missing = sorted(set(prepared) - set(scores))
        extra = sorted(set(scores) - set(prepared))
        raise ValueError(f"Reranker identities differ: missing={missing}, extra={extra}")
    if {query_id for query_id, _ in prepared} != query_set:
        raise ValueError("Prepared candidate query IDs do not match the query file.")
    support = _support_by_query(
        support_rows,
        query_ids=query_set,
        vocabulary=vocabulary,
    )

    extractor = CitationExtractor()
    candidates_by_query: dict[str, list[RankedCandidate]] = defaultdict(list)
    unknown_by_query: dict[str, dict[str, tuple[str, ...]]] = defaultdict(dict)
    for key, row in prepared.items():
        query_id, identity = key
        doc_id = str(row["doc_id"])
        score_row = scores[key]
        if int(row["rank"]) != int(score_row["rank"]):
            raise ValueError(f"{key}: retrieval rank changed between stages 4 and 5.")
        extracted = extractor.extract(str(row.get("document_text") or ""))
        citations = tuple(citation for citation in extracted if citation in vocabulary)
        unknown = tuple(citation for citation in extracted if citation not in vocabulary)
        if unknown:
            unknown_by_query[query_id][identity] = unknown
        candidates_by_query[query_id].append(
            RankedCandidate(
                doc_ref=doc_id,
                rank=int(row["rank"]),
                score=float(score_row["score"]),
                confidence=float(score_row.get("confidence") or 0.0),
                rationale=str(score_row.get("rationale") or ""),
                citations=tuple(sorted(set(citations))),
                text=str(row.get("document_text") or ""),
                court=(row.get("metadata") or {}).get("court"),
                decision_id=(row.get("metadata") or {}).get("decision_id"),
                docket_number=(row.get("metadata") or {}).get("docket_number"),
                decision_date=(row.get("metadata") or {}).get("decision_date"),
            )
        )

    predictions: dict[str, tuple[str, ...]] = {}
    audit: list[JsonObject] = []
    for query_id in ordered_query_ids:
        selection = select_citations(
            candidates_by_query[query_id],
            support[query_id],
            policy,
        )
        predictions[query_id] = selection.predicted_citations
        audit.append(
            {
                "query_id": query_id,
                "policy": policy.description,
                "predicted_citations": list(selection.predicted_citations),
                "dense_vote_counts": dict(selection.dense_vote_counts),
                "sparse_support_counts": dict(selection.bm25_counts),
                "accepted": [
                    {
                        "citation": decision.citation,
                        "kind": decision.kind.value,
                        "reason": decision.reason,
                        "evidence": decision.votes,
                    }
                    for decision in selection.accepted
                ],
                "rejected": [
                    {
                        "citation": decision.citation,
                        "kind": decision.kind.value,
                        "reason": decision.reason,
                        "evidence": decision.votes,
                    }
                    for decision in selection.rejected
                ],
                "out_of_vocabulary_extractions": {
                    doc_id: list(citations)
                    for doc_id, citations in sorted(unknown_by_query[query_id].items())
                },
                "ranked_candidates": [
                    {
                        "doc_id": candidate.doc_ref,
                        "retrieval_rank": candidate.rank,
                        "verifier_score": candidate.score,
                        "confidence": candidate.confidence,
                        "citations": list(candidate.citations),
                    }
                    for candidate in order_candidates(candidates_by_query[query_id])
                ],
            }
        )
    return predictions, audit


def atomic_write_submission(
    path: Path,
    *,
    query_ids: Iterable[str],
    predictions: Mapping[str, Iterable[str]],
) -> None:
    """Write the competition CSV atomically and in query-file order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["query_id", "predicted_citations"],
            )
            writer.writeheader()
            for query_id in query_ids:
                writer.writerow(
                    {
                        "query_id": query_id,
                        "predicted_citations": ";".join(sorted(set(predictions.get(query_id, ())))),
                    }
                )
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--sparse-support", type=Path, required=True)
    parser.add_argument("--laws", type=Path, required=True)
    parser.add_argument("--courts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--minimum-dense-votes", type=int, default=4)
    parser.add_argument("--anchor-top-k", type=int, default=3)
    parser.add_argument("--anchor-minimum-score", type=float, default=8.5)
    parser.add_argument("--sparse-top-k", type=int, default=5)
    parser.add_argument("--minimum-sparse-votes", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    queries = load_queries(args.queries)
    vocabulary = CsvCitationVocabulary.from_paths(
        laws_path=args.laws,
        courts_path=args.courts,
    )
    policy = FixedVotePolicy(
        candidate_top_k=args.candidate_top_k,
        minimum_dense_votes=args.minimum_dense_votes,
        anchor_top_k=args.anchor_top_k,
        anchor_minimum_score=args.anchor_minimum_score,
        bm25_top_k=args.sparse_top_k,
        minimum_bm25_votes=args.minimum_sparse_votes,
    )
    predictions, audit = select_for_queries(
        query_ids=queries,
        prepared_rows=read_jsonl(args.prepared),
        score_rows=read_jsonl(args.scores),
        support_rows=read_jsonl(args.sparse_support),
        vocabulary=vocabulary,
        policy=policy,
    )
    atomic_write_submission(
        args.output,
        query_ids=queries,
        predictions=predictions,
    )
    atomic_write_json(args.audit, audit)
    print(f"Selected citations for {len(predictions)} queries -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
