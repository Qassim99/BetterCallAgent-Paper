from __future__ import annotations

import unittest

from interpretability.perturbation_faithfulness import (
    evaluate_perturbation_faithfulness,
)
from interpretability.schemas import AblationObservation, PerturbationCase


def _case() -> PerturbationCase:
    return PerturbationCase(
        query_id="q1",
        full_score=1.0,
        ablations=(
            AblationObservation("evidence_a", 0.4, True),
            AblationObservation("evidence_b", 0.8, True),
            AblationObservation("control_a", 0.95, False),
            AblationObservation("control_b", 0.9, False),
            AblationObservation("control_c", 1.0, False),
            AblationObservation("control_d", 0.8, False),
        ),
    )


class PerturbationFaithfulnessTests(unittest.TestCase):
    def test_seeded_baseline_is_deterministic(self) -> None:
        first = evaluate_perturbation_faithfulness([_case()], seed=41, random_trials=200)
        second = evaluate_perturbation_faithfulness([_case()], seed=41, random_trials=200)

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["queries"][0]["targeted_aopc"], 0.4)
        self.assertAlmostEqual(
            first["macro"]["faithfulness_gap"],
            first["macro"]["targeted_aopc"] - first["macro"]["random_baseline_aopc"],
        )
        self.assertIn("does not access verifier logits or gradients", first["limitations"])

    def test_random_baseline_requires_enough_controls(self) -> None:
        insufficient = PerturbationCase(
            query_id="q1",
            full_score=1.0,
            ablations=(
                AblationObservation("evidence_a", 0.4, True),
                AblationObservation("evidence_b", 0.8, True),
                AblationObservation("control_a", 0.95, False),
            ),
        )
        with self.assertRaisesRegex(ValueError, "size-matched"):
            evaluate_perturbation_faithfulness([insufficient])
