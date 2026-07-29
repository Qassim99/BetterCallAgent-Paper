from __future__ import annotations

import unittest

from interpretability.ragas_style import evaluate_ragas_style
from interpretability.schemas import (
    PredictionRecord,
    QueryGoldRecord,
    RetrievalRecord,
)


class RagasStyleTests(unittest.TestCase):
    def test_formulas_are_explicit_citation_set_metrics(self) -> None:
        result = evaluate_ragas_style(
            [QueryGoldRecord("q1", "Question", ("A", "B"))],
            [RetrievalRecord("q1", ("A", "C", "B"))],
            [PredictionRecord("q1", ("A", "C"))],
            context_k=2,
        )

        metrics = result["queries"][0]["metrics"]
        self.assertFalse(result["official_ragas_library"])
        self.assertAlmostEqual(metrics["context_precision_at_k"], 0.5)
        self.assertAlmostEqual(metrics["gold_context_recall_at_k"], 0.5)
        self.assertAlmostEqual(metrics["citation_faithfulness"], 1.0)
        self.assertAlmostEqual(metrics["answer_precision"], 0.5)
        self.assertAlmostEqual(metrics["answer_recall"], 0.5)
        self.assertAlmostEqual(metrics["answer_f1"], 0.5)
        self.assertAlmostEqual(result["macro"]["answer_f1"], 0.5)

    def test_context_k_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "context_k"):
            evaluate_ragas_style(
                [QueryGoldRecord("q1", "Question", ("A",))],
                [RetrievalRecord("q1", ("A",))],
                [PredictionRecord("q1", ("A",))],
                context_k=0,
            )
