"""Pure citation-level precision, recall, and Macro-F1 computations."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """Metrics for one query."""

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True, slots=True)
class MacroMetrics:
    """Macro averages and aggregate confusion counts over a query set."""

    macro_precision: float
    macro_recall: float
    macro_f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    per_query: Mapping[str, QueryMetrics]


def citation_f1(
    predicted: Collection[str],
    gold: Collection[str],
) -> QueryMetrics:
    """Compute citation-set metrics for one query."""
    predicted_set = set(predicted)
    gold_set = set(gold)
    true_positives = len(predicted_set & gold_set)
    false_positives = len(predicted_set - gold_set)
    false_negatives = len(gold_set - predicted_set)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return QueryMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def macro_f1(
    predictions: Mapping[str, Collection[str]],
    gold: Mapping[str, Collection[str]],
) -> MacroMetrics:
    """Compute Macro-F1 over exactly the query IDs in the gold mapping."""
    unexpected = set(predictions) - set(gold)
    if unexpected:
        raise ValueError(f"Predictions contain unexpected query IDs: {sorted(unexpected)}")
    per_query = {
        query_id: citation_f1(predictions.get(query_id, ()), citations)
        for query_id, citations in sorted(gold.items())
    }
    count = len(per_query)
    return MacroMetrics(
        macro_precision=(
            sum(metrics.precision for metrics in per_query.values()) / count if count else 0.0
        ),
        macro_recall=(
            sum(metrics.recall for metrics in per_query.values()) / count if count else 0.0
        ),
        macro_f1=(sum(metrics.f1 for metrics in per_query.values()) / count if count else 0.0),
        true_positives=sum(metrics.true_positives for metrics in per_query.values()),
        false_positives=sum(metrics.false_positives for metrics in per_query.values()),
        false_negatives=sum(metrics.false_negatives for metrics in per_query.values()),
        per_query=per_query,
    )
