from __future__ import annotations

import unittest

from interpretability.error_attribution import attribute_citation_errors
from interpretability.schemas import (
    PredictionRecord,
    QueryGoldRecord,
    RetrievalRecord,
)


class ErrorAttributionTests(unittest.TestCase):
    def test_error_categories_are_mutually_observable(self) -> None:
        result = attribute_citation_errors(
            queries=[
                QueryGoldRecord("q1", "Question one", ("A", "B")),
                QueryGoldRecord("q2", "Question two", ("X",)),
            ],
            retrieval=[
                RetrievalRecord("q1", ("A", "C")),
                RetrievalRecord("q2", ("X",)),
            ],
            predictions=[
                PredictionRecord("q1", ("A", "C", "D")),
                PredictionRecord("q2", ()),
            ],
        )

        first, second = result["queries"]
        self.assertEqual(first["correct_citations"], ["A"])
        self.assertEqual(first["retrieval_misses"], ["B"])
        self.assertEqual(first["selected_false_positives"], ["C"])
        self.assertEqual(first["unsupported_false_positives"], ["D"])
        self.assertEqual(second["selection_or_gate_misses"], ["X"])
        self.assertAlmostEqual(result["summary"]["macro_retrieval_recall"], 0.75)
        self.assertAlmostEqual(result["summary"]["macro_prediction_f1"], 0.2)

    def test_query_coverage_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            attribute_citation_errors(
                [QueryGoldRecord("q1", "Question", ("A",))],
                [RetrievalRecord("q2", ("A",))],
                [PredictionRecord("q1", ("A",))],
            )
