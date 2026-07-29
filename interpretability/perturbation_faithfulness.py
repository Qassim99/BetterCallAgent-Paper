"""Seeded perturbation faithfulness proxy over supplied black-box scores."""

from __future__ import annotations

import hashlib
import random
import statistics
from collections.abc import Sequence
from typing import Any

from .metrics import arithmetic_mean
from .schemas import PerturbationCase

SCHEMA_VERSION = "bettercallagent.interpretability.perturbation_proxy.v1"


def _query_seed(seed: int, query_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{query_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def evaluate_perturbation_faithfulness(
    cases: Sequence[PerturbationCase],
    *,
    seed: int = 17,
    random_trials: int = 1_000,
) -> dict[str, Any]:
    """Compare evidence ablations with a size-matched seeded random baseline."""

    if not cases:
        raise ValueError("at least one perturbation case is required")
    if random_trials <= 0:
        raise ValueError("random_trials must be positive")
    query_ids = [case.query_id for case in cases]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("perturbation cases must use unique query_id values")

    query_reports: list[dict[str, Any]] = []
    for case in sorted(cases, key=lambda item: item.query_id):
        evidence = [item for item in case.ablations if item.is_evidence]
        controls = [item for item in case.ablations if not item.is_evidence]
        if not evidence:
            raise ValueError(f"{case.query_id}: at least one evidence ablation is required")
        if len(controls) < len(evidence):
            raise ValueError(
                f"{case.query_id}: need at least {len(evidence)} non-evidence "
                "ablations for a size-matched random baseline"
            )

        evidence_effects = [case.full_score - item.score_without_feature for item in evidence]
        control_effects = [case.full_score - item.score_without_feature for item in controls]
        targeted_aopc = arithmetic_mean(evidence_effects)

        generator = random.Random(_query_seed(seed, case.query_id))
        random_aopc_values = [
            arithmetic_mean(generator.sample(control_effects, k=len(evidence)))
            for _ in range(random_trials)
        ]
        random_baseline_aopc = arithmetic_mean(random_aopc_values)
        query_reports.append(
            {
                "query_id": case.query_id,
                "full_score": case.full_score,
                "evidence_feature_count": len(evidence),
                "control_feature_count": len(controls),
                "targeted_aopc": targeted_aopc,
                "random_baseline_aopc": random_baseline_aopc,
                "random_baseline_std": statistics.pstdev(random_aopc_values),
                "faithfulness_gap": targeted_aopc - random_baseline_aopc,
                "evidence_effects": {
                    item.feature: effect
                    for item, effect in zip(
                        evidence,
                        evidence_effects,
                        strict=True,
                    )
                },
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "method": "seeded_perturbation_faithfulness_proxy",
        "parameters": {"seed": seed, "random_trials": random_trials},
        "definitions": {
            "targeted_aopc": (
                "Mean decrease from the supplied full score after removing each "
                "feature marked as evidence."
            ),
            "random_baseline_aopc": (
                "Mean decrease for equally sized samples of non-evidence feature "
                "ablations, repeated with a deterministic per-query random seed."
            ),
            "faithfulness_gap": "targeted_aopc minus random_baseline_aopc.",
        },
        "limitations": (
            "This is a perturbation proxy over supplied black-box scores. It does "
            "not access verifier logits or gradients, and it does not establish "
            "causal faithfulness."
        ),
        "macro": {
            "query_count": len(query_reports),
            "targeted_aopc": arithmetic_mean(report["targeted_aopc"] for report in query_reports),
            "random_baseline_aopc": arithmetic_mean(
                report["random_baseline_aopc"] for report in query_reports
            ),
            "faithfulness_gap": arithmetic_mean(
                report["faithfulness_gap"] for report in query_reports
            ),
        },
        "queries": query_reports,
    }
