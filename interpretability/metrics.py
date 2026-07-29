"""Small citation-set metric helpers with explicit empty-set behavior."""

from __future__ import annotations

from collections.abc import Collection, Iterable


def arithmetic_mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("cannot average an empty sequence")
    return sum(materialized) / len(materialized)


def set_precision(reference: Collection[str], prediction: Collection[str]) -> float:
    """Precision, treating two empty sets as a perfect match."""

    if not prediction:
        return 1.0 if not reference else 0.0
    return len(set(reference) & set(prediction)) / len(set(prediction))


def set_recall(reference: Collection[str], prediction: Collection[str]) -> float:
    """Recall, treating an empty reference set as fully covered."""

    if not reference:
        return 1.0
    return len(set(reference) & set(prediction)) / len(set(reference))


def harmonic_f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
