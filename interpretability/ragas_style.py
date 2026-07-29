"""Citation metrics inspired by common RAG evaluation questions.

This module is named ``ragas_style`` deliberately. It is a transparent local
metric implementation and is not the official RAGAS package.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .io import index_unique, require_matching_query_ids
from .metrics import arithmetic_mean, harmonic_f1, set_precision, set_recall
from .schemas import PredictionRecord, QueryGoldRecord, RetrievalRecord

SCHEMA_VERSION = "bettercallagent.interpretability.ragas_style.v1"


def _context_precision(gold: tuple[str, ...], context: tuple[str, ...]) -> float:
    if not context:
        return 1.0 if not gold else 0.0
    return len(set(gold) & set(context)) / len(context)


def _citation_faithfulness(context: tuple[str, ...], predictions: tuple[str, ...]) -> float:
    if not predictions:
        return 1.0
    return len(set(context) & set(predictions)) / len(predictions)


def evaluate_ragas_style(
    queries: Sequence[QueryGoldRecord],
    retrieval: Sequence[RetrievalRecord],
    predictions: Sequence[PredictionRecord],
    *,
    context_k: int = 10,
) -> dict[str, Any]:
    """Compute deterministic citation-level, RAGAS-style diagnostics."""

    if context_k <= 0:
        raise ValueError("context_k must be positive")

    query_index = index_unique(queries, "queries")
    retrieval_index = index_unique(retrieval, "retrieval")
    prediction_index = index_unique(predictions, "predictions")
    if not query_index:
        raise ValueError("at least one query is required")
    require_matching_query_ids(query_index, retrieval_index, prediction_index)

    query_reports: list[dict[str, Any]] = []
    for query_id in sorted(query_index):
        gold = query_index[query_id].gold_citations
        context = retrieval_index[query_id].retrieved_citations[:context_k]
        predictions_for_query = prediction_index[query_id].predicted_citations

        answer_precision = set_precision(gold, predictions_for_query)
        answer_recall = set_recall(gold, predictions_for_query)
        query_reports.append(
            {
                "query_id": query_id,
                "context_size": len(context),
                "metrics": {
                    "context_precision_at_k": _context_precision(gold, context),
                    "gold_context_recall_at_k": set_recall(gold, context),
                    "citation_faithfulness": _citation_faithfulness(context, predictions_for_query),
                    "answer_precision": answer_precision,
                    "answer_recall": answer_recall,
                    "answer_f1": harmonic_f1(answer_precision, answer_recall),
                },
            }
        )

    metric_names = tuple(query_reports[0]["metrics"])
    macro = {
        metric_name: arithmetic_mean(report["metrics"][metric_name] for report in query_reports)
        for metric_name in metric_names
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "ragas_style_citation_metrics",
        "official_ragas_library": False,
        "parameters": {"context_k": context_k},
        "definitions": {
            "context_precision_at_k": (
                "Fraction of the first k retrieved citations that occur in gold."
            ),
            "gold_context_recall_at_k": (
                "Fraction of gold citations found in the first k retrieval results."
            ),
            "citation_faithfulness": (
                "Fraction of predicted citations supported by the first k retrieval "
                "results; this measures retrieval support, not legal correctness."
            ),
            "answer_f1": "Set F1 between predicted and gold citation identifiers.",
            "empty_set_policy": (
                "An empty prediction/context scores 1 only when its reference is "
                "also empty; otherwise its precision or recall contribution is 0."
            ),
        },
        "macro": macro,
        "queries": query_reports,
    }
