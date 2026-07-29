"""Attribute citation errors to observable offline pipeline stages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .io import index_unique, require_matching_query_ids
from .metrics import arithmetic_mean, harmonic_f1, set_precision, set_recall
from .schemas import PredictionRecord, QueryGoldRecord, RetrievalRecord

SCHEMA_VERSION = "bettercallagent.interpretability.error_attribution.v1"


def attribute_citation_errors(
    queries: Sequence[QueryGoldRecord],
    retrieval: Sequence[RetrievalRecord],
    predictions: Sequence[PredictionRecord],
) -> dict[str, Any]:
    """Return conservative, evidence-based citation error categories.

    A retrieved gold citation that is absent from the final prediction cannot be
    assigned more narrowly without a saved selector/gate trace. It is therefore
    reported as ``selection_or_gate_miss`` rather than guessing a cause.
    """

    query_index = index_unique(queries, "queries")
    retrieval_index = index_unique(retrieval, "retrieval")
    prediction_index = index_unique(predictions, "predictions")
    if not query_index:
        raise ValueError("at least one query is required")
    require_matching_query_ids(query_index, retrieval_index, prediction_index)

    query_reports: list[dict[str, Any]] = []
    for query_id in sorted(query_index):
        query_record = query_index[query_id]
        retrieval_record = retrieval_index[query_id]
        prediction_record = prediction_index[query_id]

        gold = query_record.gold_citations
        retrieved = retrieval_record.retrieved_citations
        predicted = prediction_record.predicted_citations
        gold_set = set(gold)
        retrieved_set = set(retrieved)
        predicted_set = set(predicted)

        precision = set_precision(gold, predicted)
        recall = set_recall(gold, predicted)
        report = {
            "query_id": query_id,
            "counts": {
                "gold": len(gold),
                "retrieved": len(retrieved),
                "predicted": len(predicted),
            },
            "metrics": {
                "retrieval_recall": set_recall(gold, retrieved),
                "prediction_precision": precision,
                "prediction_recall": recall,
                "prediction_f1": harmonic_f1(precision, recall),
            },
            "correct_citations": [citation for citation in gold if citation in predicted_set],
            "retrieval_misses": [citation for citation in gold if citation not in retrieved_set],
            "selection_or_gate_misses": [
                citation
                for citation in gold
                if citation in retrieved_set and citation not in predicted_set
            ],
            "selected_false_positives": [
                citation
                for citation in predicted
                if citation in retrieved_set and citation not in gold_set
            ],
            "unsupported_false_positives": [
                citation
                for citation in predicted
                if citation not in retrieved_set and citation not in gold_set
            ],
            "unsupported_correct_predictions": [
                citation
                for citation in predicted
                if citation not in retrieved_set and citation in gold_set
            ],
        }
        query_reports.append(report)

    count_fields = (
        "correct_citations",
        "retrieval_misses",
        "selection_or_gate_misses",
        "selected_false_positives",
        "unsupported_false_positives",
        "unsupported_correct_predictions",
    )
    summary_counts = {
        field: sum(len(report[field]) for report in query_reports) for field in count_fields
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "method": "citation_error_stage_attribution",
        "definitions": {
            "retrieval_miss": "Gold citation absent from the saved retrieval list.",
            "selection_or_gate_miss": (
                "Gold citation retrieved but absent from the final prediction; "
                "the supplied artifacts do not distinguish selector from gate."
            ),
            "selected_false_positive": (
                "Non-gold citation present in both retrieval and final prediction."
            ),
            "unsupported_false_positive": (
                "Non-gold prediction absent from the saved retrieval list."
            ),
            "unsupported_correct_prediction": (
                "Gold prediction absent from the saved retrieval list; this signals "
                "an artifact or pipeline-path mismatch."
            ),
        },
        "summary": {
            "query_count": len(query_reports),
            "counts": summary_counts,
            "macro_retrieval_recall": arithmetic_mean(
                report["metrics"]["retrieval_recall"] for report in query_reports
            ),
            "macro_prediction_f1": arithmetic_mean(
                report["metrics"]["prediction_f1"] for report in query_reports
            ),
        },
        "queries": query_reports,
    }
