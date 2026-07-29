"""Versioned online-asset and retrieval-view binding tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from online.context import RunContext
from online.dependencies import OnlineDependencies
from online.repository import AssetError, OnlineAssetRepository
from online.stages.stage_02_generate_queries import run as generate_queries
from tests.online.helpers import (
    FIXTURE_MODEL,
    FIXTURE_PATH,
    FIXTURE_QUERY_ID,
    fixture_settings,
)


def _fixture_payload() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _write_asset(directory: str, payload: dict[str, Any]) -> Path:
    path = Path(directory) / "asset.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


class OnlineAssetRepositoryTests(unittest.TestCase):
    def test_asset_version_and_view_query_coverage_are_strict(self) -> None:
        legacy = _fixture_payload()
        legacy["version"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = _write_asset(directory, legacy)
            with self.assertRaisesRegex(AssetError, "version must be exactly 2"):
                OnlineAssetRepository.from_json(path)

        incomplete = _fixture_payload()
        incomplete["retrieval_views"]["unexpected-query"] = incomplete["retrieval_views"].pop(
            FIXTURE_QUERY_ID
        )
        with tempfile.TemporaryDirectory() as directory:
            path = _write_asset(directory, incomplete)
            with self.assertRaisesRegex(
                AssetError,
                "keys must exactly match configured query IDs",
            ):
                OnlineAssetRepository.from_json(path)

    def test_retrieval_views_are_required_for_every_query(self) -> None:
        payload = _fixture_payload()
        del payload["retrieval_views"]
        with tempfile.TemporaryDirectory() as directory:
            path = _write_asset(directory, payload)
            with self.assertRaisesRegex(AssetError, "retrieval_views"):
                OnlineAssetRepository.from_json(path)

    def test_retrieval_views_require_exactly_five_fields(self) -> None:
        payload = _fixture_payload()
        views = payload["retrieval_views"][FIXTURE_QUERY_ID]
        del views["citations"]
        views["unexpected"] = "not part of the ranking contract"
        with tempfile.TemporaryDirectory() as directory:
            path = _write_asset(directory, payload)
            with self.assertRaisesRegex(
                AssetError,
                r"missing=\['citations'\].*unknown=\['unexpected'\]",
            ):
                OnlineAssetRepository.from_json(path)


class RetrievalViewBindingTests(unittest.IsolatedAsyncioTestCase):
    async def test_stage_two_rejects_one_byte_of_asset_view_drift(self) -> None:
        payload = _fixture_payload()
        payload["retrieval_views"][FIXTURE_QUERY_ID]["keywords"] += " "
        with tempfile.TemporaryDirectory() as directory:
            asset_path = _write_asset(directory, payload)
            settings = replace(
                fixture_settings(),
                asset_path=asset_path,
            )
            dependencies = OnlineDependencies.build(settings)
            try:
                record = dependencies.repository.queries[FIXTURE_QUERY_ID]
                context = RunContext(
                    run_id="view-drift-test",
                    record=record,
                    model=FIXTURE_MODEL,
                    understanding={
                        "summary_de": "fixture",
                        "legal_issues": [],
                    },
                )
                with self.assertRaisesRegex(
                    AssetError,
                    r"mismatched fields=\['keywords'\]",
                ):
                    await generate_queries(context, dependencies)
                self.assertIsNone(context.query_views)
            finally:
                await dependencies.close()
