"""Offline pipeline tests use only the repository's synthetic fixture."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from bettercallagent.citations.policy import FixedVotePolicy
from bettercallagent.citations.vocabulary import InMemoryCitationVocabulary
from bettercallagent.providers.openai_compatible import ProviderError
from bettercallagent.retrieval.query_views import QueryViews
from bettercallagent.schemas import (
    CitationKind,
    LLMResponse,
    RetrievalHit,
)
from offline.identity import candidate_id
from offline.io import read_jsonl
from offline.run import REPOSITORY_ROOT, run_pipeline
from offline.stages.step_01_retrieve_dense import FIELDS, _serialize_query
from offline.stages.step_02_retrieve_sparse_support import balanced_hits
from offline.stages.step_05_rerank import rerank_from_replay, rerank_live
from offline.stages.step_06_select_citations import select_for_queries

FIXTURES = REPOSITORY_ROOT / "offline" / "fixtures"


class OfflinePipelineTests(unittest.TestCase):
    def test_fixture_runs_end_to_end_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            manifest = asyncio.run(
                run_pipeline(
                    FIXTURES / "config.toml",
                    output_override=output,
                )
            )

            self.assertEqual(manifest["mode"], "fixture")
            self.assertEqual(manifest["reranking"]["source"], "fingerprint_bound_replay")
            self.assertEqual(manifest["metrics"]["macro_f1"], 1.0)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "01_dense_candidates.jsonl",
                    "01_dense_summary.json",
                    "02_sparse_support.jsonl",
                    "03_documents.jsonl",
                    "04_reranker_input.jsonl",
                    "05_reranker_scores.jsonl",
                    "06_submission.csv",
                    "06_selection_audit.json",
                    "07_metrics.json",
                    "run_manifest.json",
                },
            )
            audit = json.loads((output / "06_selection_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(
                audit[0]["predicted_citations"],
                ["Art. 41 Abs. 1 OR", "Art. 42 Abs. 1 OR", "BGE 145 III 1"],
            )

    def test_replay_rejects_changed_document_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            asyncio.run(run_pipeline(FIXTURES / "config.toml", output_override=output))
            prepared = list(read_jsonl(output / "04_reranker_input.jsonl"))
            prepared[0]["document_text"] += " changed"

            with self.assertRaisesRegex(ValueError, "stale replay fingerprint"):
                rerank_from_replay(
                    prepared,
                    replay_path=FIXTURES / "reranker_replay.jsonl",
                    model="fixture-reranker-v1",
                )

    def test_replay_binds_rank_truncation_and_batch_composition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            asyncio.run(run_pipeline(FIXTURES / "config.toml", output_override=output))
            prepared = list(read_jsonl(output / "04_reranker_input.jsonl"))
            replay = FIXTURES / "reranker_replay.jsonl"

            changed_rank = [dict(row) for row in prepared]
            changed_rank[0]["rank"] = 6
            with self.assertRaisesRegex(ValueError, "stale replay fingerprint"):
                rerank_from_replay(
                    changed_rank,
                    replay_path=replay,
                    model="fixture-reranker-v1",
                )
            with self.assertRaisesRegex(ValueError, "stale replay fingerprint"):
                rerank_from_replay(
                    prepared,
                    replay_path=replay,
                    model="fixture-reranker-v1",
                    document_char_limit=20,
                )
            with self.assertRaisesRegex(ValueError, "batch fingerprint"):
                rerank_from_replay(
                    prepared,
                    replay_path=replay,
                    model="fixture-reranker-v1",
                    batch_size=4,
                )

    def test_fixture_mode_cannot_fall_through_to_network(self) -> None:
        original = (FIXTURES / "config.toml").read_text(encoding="utf-8")
        without_replay = "\n".join(
            line for line in original.splitlines() if "reranker_replay" not in line
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.toml"
            config.write_text(without_replay, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires paths.reranker_replay"):
                asyncio.run(run_pipeline(config, output_override=root / "run"))

    def test_nonempty_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            (output / "stale.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                asyncio.run(run_pipeline(FIXTURES / "config.toml", output_override=output))

    def test_sparse_balancing_is_one_law_then_two_courts(self) -> None:
        trace = {
            "candidate_law_hits": [{"citation": "L1"}, {"citation": "L2"}],
            "candidate_court_hits": [
                {"citation": "C1"},
                {"citation": "C2"},
                {"citation": "C3"},
            ],
        }
        self.assertEqual(
            [hit["citation"] for hit in balanced_hits(trace, 5)],
            ["L1", "C1", "C2", "L2", "C3"],
        )

    def test_duplicate_physical_document_keeps_distinct_index_rows(self) -> None:
        rankings = {
            field: [
                RetrievalHit(
                    doc_ref=identity,
                    score=1.0 / rank,
                    score_kind="cosine",
                    rank=rank,
                )
                for rank, identity in enumerate(("7", "11"), start=1)
            ]
            for field in FIELDS
        }
        fused = [
            RetrievalHit(
                doc_ref=identity,
                score=1.0 / rank,
                score_kind="weighted_rrf",
                rank=rank,
            )
            for rank, identity in enumerate(("7", "11"), start=1)
        ]
        candidates, _ = _serialize_query(
            query_id="q",
            query="question",
            views=QueryViews("question", "meta", "keywords", "question", "question"),
            rankings=rankings,
            fused=fused,
            metadata={
                "7": {"global_idx": 7},
                "11": {"global_idx": 11},
            },
            document_ids={
                "7": "same.parquet:rg0:row3",
                "11": "same.parquet:rg0:row3",
            },
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            [candidate["doc_id"] for candidate in candidates],
            ["same.parquet:rg0:row3", "same.parquet:rg0:row3"],
        )
        self.assertEqual(
            [candidate_id(candidate) for candidate in candidates],
            ["q::gi7", "q::gi11"],
        )

    def test_selection_joins_duplicate_documents_by_candidate_identity(self) -> None:
        prepared = [
            {
                "query_id": "q",
                "doc_id": "same-document",
                "global_idx": global_index,
                "candidate_id": f"q::gi{global_index}",
                "rank": rank,
                "document_text": "Art. 41 OR",
                "metadata": {},
            }
            for rank, global_index in enumerate((7, 11), start=1)
        ]
        scores = [
            {
                "query_id": "q",
                "doc_id": "same-document",
                "candidate_id": row["candidate_id"],
                "rank": row["rank"],
                "score": 9.0,
                "confidence": 1.0,
            }
            for row in prepared
        ]
        predictions, _ = select_for_queries(
            query_ids=["q"],
            prepared_rows=prepared,
            score_rows=scores,
            support_rows=[{"query_id": "q", "support_counts": {}}],
            vocabulary=InMemoryCitationVocabulary(entries={"Art. 41 OR": CitationKind.LAW}),
            policy=FixedVotePolicy(
                candidate_top_k=2,
                minimum_dense_votes=2,
                anchor_top_k=1,
                anchor_minimum_score=10.0,
                bm25_top_k=1,
                minimum_bm25_votes=1,
            ),
        )
        self.assertEqual(predictions["q"], ("Art. 41 OR",))

    def test_live_reranker_retries_and_resumes_from_raw_checkpoint(self) -> None:
        row = {
            "query_id": "q",
            "query": "question",
            "meta_query": "meta",
            "keywords_query": "keywords",
            "doc_id": "doc",
            "global_idx": 7,
            "rank": 1,
            "fusion_score": 0.5,
            "hits": [],
            "document_text": "document",
        }

        class FlakyProvider:
            def __init__(self) -> None:
                self.calls = 0

            async def complete(self, *args: object, **kwargs: object) -> LLMResponse:
                self.calls += 1
                if self.calls == 1:
                    raise ProviderError("temporary")
                content = json.dumps(
                    {
                        "scores": [
                            {
                                "candidate_id": "q::gi7",
                                "score": 9.0,
                                "confidence": 0.9,
                                "rationale_de": "match",
                            }
                        ]
                    }
                )
                return LLMResponse(
                    content=content,
                    model="provider-resolved-model",
                    usage_total_tokens=17,
                    raw={"model": "provider-resolved-model", "content": content},
                )

            async def close(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "batches.jsonl"
            provider = FlakyProvider()
            first = asyncio.run(
                rerank_live(
                    [row],
                    provider=provider,
                    model="requested-model",
                    batch_size=1,
                    concurrency=1,
                    document_char_limit=0,
                    checkpoint_path=checkpoint,
                    max_attempts=2,
                    retry_delay_seconds=0,
                )
            )
            self.assertEqual(provider.calls, 2)
            checkpoint_rows = list(read_jsonl(checkpoint))
            self.assertEqual(checkpoint_rows[0]["attempts"], 2)
            self.assertEqual(
                checkpoint_rows[0]["provider_model"],
                "provider-resolved-model",
            )
            self.assertIn("raw_response", checkpoint_rows[0])

            resumed_provider = FlakyProvider()
            resumed_provider.calls = 1
            second = asyncio.run(
                rerank_live(
                    [row],
                    provider=resumed_provider,
                    model="requested-model",
                    batch_size=1,
                    concurrency=1,
                    document_char_limit=0,
                    checkpoint_path=checkpoint,
                    max_attempts=2,
                    retry_delay_seconds=0,
                )
            )
            self.assertEqual(resumed_provider.calls, 1)
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
