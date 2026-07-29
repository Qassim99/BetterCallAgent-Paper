"""Deterministic logistic/additive surrogate for saved gate decisions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .metrics import arithmetic_mean
from .schemas import GateCandidate

SCHEMA_VERSION = "bettercallagent.interpretability.logistic_additive_surrogate.v1"


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        factor = math.exp(-value)
        return 1.0 / (1.0 + factor)
    factor = math.exp(value)
    return factor / (1.0 + factor)


def _validate_candidates(
    candidates: Sequence[GateCandidate],
) -> tuple[str, ...]:
    if len(candidates) < 2:
        raise ValueError("at least two gate candidates are required")
    labels = {candidate.accepted for candidate in candidates}
    if labels != {False, True}:
        raise ValueError("gate candidates must contain accepted and rejected examples")

    candidate_ids = [(candidate.query_id, candidate.citation) for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("gate candidates must use unique query_id/citation pairs")

    feature_names = tuple(sorted(candidates[0].features))
    expected = set(feature_names)
    for candidate in candidates:
        observed = set(candidate.features)
        if observed != expected:
            raise ValueError(
                "all gate candidates must use the same feature names; "
                f"expected={sorted(expected)}, observed={sorted(observed)}"
            )
    return feature_names


def fit_logistic_additive_surrogate(
    candidates: Sequence[GateCandidate],
    *,
    iterations: int = 2_000,
    learning_rate: float = 0.1,
    l2_strength: float = 0.01,
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    """Fit and explain a deterministic linear logistic surrogate.

    Contributions are coefficient-times-standardized-feature terms. They sum
    exactly to the surrogate log-odds and describe the fitted surrogate only.
    """

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if l2_strength < 0.0:
        raise ValueError("l2_strength must be non-negative")
    if not 0.0 < decision_threshold < 1.0:
        raise ValueError("decision_threshold must be between zero and one")

    feature_names = _validate_candidates(candidates)
    row_count = len(candidates)

    means = {
        name: arithmetic_mean(candidate.features[name] for candidate in candidates)
        for name in feature_names
    }
    scales: dict[str, float] = {}
    for name in feature_names:
        variance = arithmetic_mean(
            (candidate.features[name] - means[name]) ** 2 for candidate in candidates
        )
        scales[name] = math.sqrt(variance) if variance > 0.0 else 1.0

    rows = [
        tuple((candidate.features[name] - means[name]) / scales[name] for name in feature_names)
        for candidate in candidates
    ]
    labels = [1.0 if candidate.accepted else 0.0 for candidate in candidates]

    intercept = 0.0
    coefficients = [0.0] * len(feature_names)
    for _ in range(iterations):
        probabilities = [
            _sigmoid(
                intercept
                + sum(weight * value for weight, value in zip(coefficients, row, strict=True))
            )
            for row in rows
        ]
        residuals = [
            probability - label for probability, label in zip(probabilities, labels, strict=True)
        ]
        intercept_update = arithmetic_mean(residuals)
        coefficient_updates = [
            (
                sum(residual * row[index] for residual, row in zip(residuals, rows, strict=True))
                / row_count
            )
            + l2_strength * coefficients[index]
            for index in range(len(feature_names))
        ]
        intercept -= learning_rate * intercept_update
        coefficients = [
            weight - learning_rate * update
            for weight, update in zip(
                coefficients,
                coefficient_updates,
                strict=True,
            )
        ]

    explanations: list[dict[str, Any]] = []
    probabilities: list[float] = []
    correct = 0
    for candidate, row in zip(candidates, rows, strict=True):
        contributions = {
            name: weight * value
            for name, weight, value in zip(
                feature_names,
                coefficients,
                row,
                strict=True,
            )
        }
        log_odds = intercept + sum(contributions.values())
        probability = _sigmoid(log_odds)
        prediction = probability >= decision_threshold
        probabilities.append(probability)
        correct += int(prediction == candidate.accepted)
        explanations.append(
            {
                "query_id": candidate.query_id,
                "citation": candidate.citation,
                "observed_accepted": candidate.accepted,
                "surrogate_probability": probability,
                "surrogate_prediction": prediction,
                "additive_log_odds": {
                    "intercept": intercept,
                    "feature_contributions": contributions,
                    "sum": log_odds,
                },
            }
        )

    epsilon = 1e-15
    log_loss = -arithmetic_mean(
        label * math.log(max(probability, epsilon))
        + (1.0 - label) * math.log(max(1.0 - probability, epsilon))
        for label, probability in zip(labels, probabilities, strict=True)
    )
    brier_score = arithmetic_mean(
        (probability - label) ** 2 for label, probability in zip(labels, probabilities, strict=True)
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "method": "logistic_additive_gate_surrogate",
        "scope": (
            "Observable feature contributions to a fitted linear surrogate; "
            "not an explanation of model internals or a causal effect."
        ),
        "parameters": {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "l2_strength": l2_strength,
            "decision_threshold": decision_threshold,
        },
        "fit": {
            "row_count": row_count,
            "fidelity_accuracy": correct / row_count,
            "brier_score": brier_score,
            "log_loss": log_loss,
            "intercept": intercept,
            "features": [
                {
                    "name": name,
                    "mean": means[name],
                    "scale": scales[name],
                    "standardized_coefficient": weight,
                    "raw_unit_coefficient": weight / scales[name],
                }
                for name, weight in zip(
                    feature_names,
                    coefficients,
                    strict=True,
                )
            ],
        },
        "explanations": explanations,
    }
