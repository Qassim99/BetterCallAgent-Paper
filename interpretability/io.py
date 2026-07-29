"""Strict JSON/JSONL readers for reproducible offline analysis."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from .schemas import (
    AblationObservation,
    GateCandidate,
    PerturbationCase,
    PredictionRecord,
    QueryGoldRecord,
    RetrievalRecord,
)

RecordT = TypeVar("RecordT")


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    """Read JSON objects from a non-empty JSONL file."""

    if not path.is_file():
        raise FileNotFoundError(f"JSONL artifact does not exist: {path}")

    records: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)

    if not records:
        raise ValueError(f"JSONL artifact contains no records: {path}")
    return records


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON artifact does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _require_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _require_bool(record: Mapping[str, Any], field: str) -> bool:
    value = record.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _require_number(record: Mapping[str, Any], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    return float(value)


def _require_string_list(record: Mapping[str, Any], field: str) -> tuple[str, ...]:
    value = record.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a JSON array of strings")
    return tuple(value)


def load_query_gold(path: Path) -> list[QueryGoldRecord]:
    return [
        QueryGoldRecord(
            query_id=_require_string(record, "query_id"),
            query=_require_string(record, "query"),
            gold_citations=_require_string_list(record, "gold_citations"),
        )
        for record in read_jsonl(path)
    ]


def load_retrieval(path: Path) -> list[RetrievalRecord]:
    return [
        RetrievalRecord(
            query_id=_require_string(record, "query_id"),
            retrieved_citations=_require_string_list(record, "retrieved_citations"),
        )
        for record in read_jsonl(path)
    ]


def load_predictions(path: Path) -> list[PredictionRecord]:
    return [
        PredictionRecord(
            query_id=_require_string(record, "query_id"),
            predicted_citations=_require_string_list(record, "predicted_citations"),
        )
        for record in read_jsonl(path)
    ]


def load_gate_candidates(path: Path) -> list[GateCandidate]:
    candidates: list[GateCandidate] = []
    for record in read_jsonl(path):
        raw_features = record.get("features")
        if not isinstance(raw_features, dict) or not raw_features:
            raise ValueError("features must be a non-empty JSON object")
        features: dict[str, float] = {}
        for name, value in raw_features.items():
            if not isinstance(name, str):
                raise ValueError("feature names must be strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"feature {name!r} must be numeric")
            features[name] = float(value)
        candidates.append(
            GateCandidate(
                query_id=_require_string(record, "query_id"),
                citation=_require_string(record, "citation"),
                accepted=_require_bool(record, "accepted"),
                features=features,
            )
        )
    return candidates


def load_perturbation_cases(path: Path) -> list[PerturbationCase]:
    cases: list[PerturbationCase] = []
    for record in read_jsonl(path):
        raw_ablations = record.get("ablations")
        if not isinstance(raw_ablations, list) or not raw_ablations:
            raise ValueError("ablations must be a non-empty JSON array")
        ablations: list[AblationObservation] = []
        for raw_ablation in raw_ablations:
            if not isinstance(raw_ablation, dict):
                raise ValueError("each ablation must be a JSON object")
            ablations.append(
                AblationObservation(
                    feature=_require_string(raw_ablation, "feature"),
                    score_without_feature=_require_number(raw_ablation, "score_without_feature"),
                    is_evidence=_require_bool(raw_ablation, "is_evidence"),
                )
            )
        cases.append(
            PerturbationCase(
                query_id=_require_string(record, "query_id"),
                full_score=_require_number(record, "full_score"),
                ablations=tuple(ablations),
            )
        )
    return cases


def index_unique(records: Iterable[RecordT], artifact_name: str) -> dict[str, RecordT]:
    """Index dataclass records by query_id and reject duplicates."""

    indexed: dict[str, RecordT] = {}
    for record in records:
        query_id = getattr(record, "query_id", None)
        if not isinstance(query_id, str):
            raise TypeError(f"{artifact_name} record has no string query_id")
        if query_id in indexed:
            raise ValueError(f"duplicate query_id {query_id!r} in {artifact_name}")
        indexed[query_id] = record
    return indexed


def require_matching_query_ids(
    queries: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    predictions: Mapping[str, Any],
) -> None:
    """Require exact query coverage across the three analysis artifacts."""

    expected = set(queries)
    mismatches: list[str] = []
    for name, artifact in (("retrieval", retrieval), ("predictions", predictions)):
        missing = sorted(expected - set(artifact))
        extra = sorted(set(artifact) - expected)
        if missing or extra:
            mismatches.append(f"{name}: missing={missing}, extra={extra}")
    if mismatches:
        raise ValueError("query_id coverage mismatch; " + "; ".join(mismatches))
