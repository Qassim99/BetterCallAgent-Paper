from __future__ import annotations

import unittest

from interpretability.gate_surrogate import fit_logistic_additive_surrogate
from interpretability.schemas import GateCandidate


def _candidates() -> list[GateCandidate]:
    return [
        GateCandidate("q1", "A", False, {"score": 0.0, "votes": 0.0}),
        GateCandidate("q1", "B", False, {"score": 1.0, "votes": 0.0}),
        GateCandidate("q1", "C", False, {"score": 2.0, "votes": 1.0}),
        GateCandidate("q2", "D", True, {"score": 8.0, "votes": 4.0}),
        GateCandidate("q2", "E", True, {"score": 9.0, "votes": 5.0}),
        GateCandidate("q2", "F", True, {"score": 10.0, "votes": 6.0}),
    ]


class GateSurrogateTests(unittest.TestCase):
    def test_surrogate_is_deterministic_and_additive(self) -> None:
        first = fit_logistic_additive_surrogate(_candidates(), iterations=600)
        second = fit_logistic_additive_surrogate(_candidates(), iterations=600)

        self.assertEqual(first, second)
        self.assertEqual(first["method"], "logistic_additive_gate_surrogate")
        self.assertAlmostEqual(first["fit"]["fidelity_accuracy"], 1.0)
        for explanation in first["explanations"]:
            additive = explanation["additive_log_odds"]
            reconstructed = additive["intercept"] + sum(additive["feature_contributions"].values())
            self.assertAlmostEqual(additive["sum"], reconstructed)

    def test_surrogate_requires_both_gate_outcomes(self) -> None:
        with self.assertRaisesRegex(ValueError, "accepted and rejected"):
            fit_logistic_additive_surrogate(
                [
                    GateCandidate("q1", "A", True, {"score": 1.0}),
                    GateCandidate("q1", "B", True, {"score": 2.0}),
                ]
            )
