"""Typed input records shared by the offline analyses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite


def _require_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a non-empty, trimmed string")


def _require_unique(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicate citations")
    for value in values:
        _require_identifier(value, field_name)


@dataclass(frozen=True)
class QueryGoldRecord:
    """A research query and its citation-level reference labels."""

    query_id: str
    query: str
    gold_citations: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.query_id, "query_id")
        if not isinstance(self.query, str) or not self.query or self.query.strip() != self.query:
            raise ValueError("query must be a non-empty, trimmed string")
        _require_unique(self.gold_citations, "gold_citations")


@dataclass(frozen=True)
class RetrievalRecord:
    """Ranked canonical citation identifiers returned by retrieval."""

    query_id: str
    retrieved_citations: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.query_id, "query_id")
        _require_unique(self.retrieved_citations, "retrieved_citations")


@dataclass(frozen=True)
class PredictionRecord:
    """Canonical citation identifiers emitted by the pipeline."""

    query_id: str
    predicted_citations: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.query_id, "query_id")
        _require_unique(self.predicted_citations, "predicted_citations")


@dataclass(frozen=True)
class GateCandidate:
    """One citation-gate decision with observable numeric features."""

    query_id: str
    citation: str
    accepted: bool
    features: Mapping[str, float]

    def __post_init__(self) -> None:
        _require_identifier(self.query_id, "query_id")
        _require_identifier(self.citation, "citation")
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted must be a boolean")
        if not self.features:
            raise ValueError("features must contain at least one numeric feature")
        for name, value in self.features.items():
            _require_identifier(name, "feature name")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"feature {name!r} must be numeric")
            if not isfinite(value):
                raise ValueError(f"feature {name!r} must be finite")


@dataclass(frozen=True)
class AblationObservation:
    """A supplied black-box score after removing one named feature."""

    feature: str
    score_without_feature: float
    is_evidence: bool

    def __post_init__(self) -> None:
        _require_identifier(self.feature, "feature")
        if not isinstance(self.is_evidence, bool):
            raise ValueError("is_evidence must be a boolean")
        if isinstance(self.score_without_feature, bool) or not isinstance(
            self.score_without_feature, (int, float)
        ):
            raise ValueError("score_without_feature must be numeric")
        if not isfinite(self.score_without_feature):
            raise ValueError("score_without_feature must be finite")


@dataclass(frozen=True)
class PerturbationCase:
    """Full and feature-ablated scores for one query."""

    query_id: str
    full_score: float
    ablations: tuple[AblationObservation, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.query_id, "query_id")
        if isinstance(self.full_score, bool) or not isinstance(self.full_score, (int, float)):
            raise ValueError("full_score must be numeric")
        if not isfinite(self.full_score):
            raise ValueError("full_score must be finite")
        if not self.ablations:
            raise ValueError("ablations must not be empty")
        names = tuple(item.feature for item in self.ablations)
        if len(names) != len(set(names)):
            raise ValueError("ablations must use unique feature names per query")
