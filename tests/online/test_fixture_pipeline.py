"""End-to-end tests for the network-free six-stage fixture."""

from __future__ import annotations

import unittest

from online.context import RunContext
from online.pipeline import stream_pipeline
from tests.online.helpers import FIXTURE_MODEL, fixture_dependencies


class FixturePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixture_event_order_and_grounded_answer(self) -> None:
        dependencies = fixture_dependencies()
        self.addAsyncCleanup(dependencies.close)
        record = next(iter(dependencies.repository.queries.values()))
        context = RunContext(
            run_id="stable-test-run",
            record=record,
            model=FIXTURE_MODEL,
        )

        events = [event async for event in stream_pipeline(context, dependencies)]

        self.assertEqual(
            [event["type"] for event in events],
            [
                "run_start",
                "step_start",
                "step_complete",
                "step_start",
                "step_complete",
                "step_start",
                "step_complete",
                "step_start",
                "step_complete",
                "step_start",
                "step_complete",
                "step_start",
                "step_complete",
                "final_answer",
                "run_complete",
                "stream_end",
            ],
        )
        self.assertEqual(
            [event["step"] for event in events if event["type"] == "step_complete"],
            [1, 2, 3, 4, 5, 6],
        )
        for event in events:
            self.assertIsInstance(event["ts"], float)
            self.assertEqual(event["run_id"], "stable-test-run")

        completed = {event["step"]: event for event in events if event["type"] == "step_complete"}
        self.assertEqual(completed[1]["data"]["kind"], "understanding")
        self.assertEqual(
            len(completed[2]["data"]["search_queries"]),
            5,
        )
        self.assertEqual(
            context.query_views,
            dependencies.repository.retrieval_views[record.query_id],
        )
        self.assertEqual(completed[3]["data"]["kind"], "retrieval")
        self.assertEqual(
            completed[3]["name"],
            "Artifact-backed ranking replay",
        )
        self.assertTrue(completed[3]["data"]["dense_available"])
        for source in ("dense", "bm25", "hybrid"):
            for document in completed[3]["data"][source]:
                self.assertTrue({"doc_ref", "snippet", "score", "score_kind"} <= document.keys())
        self.assertEqual(completed[4]["data"]["kind"], "reranking")
        self.assertEqual(completed[5]["data"]["kind"], "citation_validation")
        self.assertEqual(completed[5]["data"]["bm25_support"]["top_k"], 5)
        self.assertEqual(completed[6]["data"]["kind"], "final_answer_step")

        final = next(event for event in events if event["type"] == "final_answer")
        self.assertEqual(
            final["grounded_on"],
            [
                "Art. 102 Abs. 1 OR",
                "Art. 107 Abs. 2 OR",
                "Art. 97 Abs. 1 OR",
            ],
        )
        self.assertNotIn("Art. 41 Abs. 1 OR", final["markdown"])
        run_complete = next(event for event in events if event["type"] == "run_complete")
        self.assertIsInstance(run_complete["elapsed_s"], float)
        self.assertIsInstance(run_complete["usage_total_tokens"], int)
