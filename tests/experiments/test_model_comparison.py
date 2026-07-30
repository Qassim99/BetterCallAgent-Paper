"""Tests for the fingerprinted hosted-model comparison runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.model_comparison import run_soofi
from offline.io import atomic_write_jsonl, sha256_file


def validation_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for query_number in range(1, 11):
        query_id = f"val_{query_number:03d}"
        for rank in range(1, 11):
            global_index = query_number * 100 + rank
            rows.append(
                {
                    "query_id": query_id,
                    "query": f"Frage {query_number}",
                    "meta_query": f"Meta {query_number}",
                    "keywords_query": f"Stichwort {query_number}",
                    "doc_id": f"doc-{global_index}",
                    "global_idx": global_index,
                    "rank": rank,
                    "fusion_score": 1.0 / rank,
                    "hits": [],
                    "document_text": f"Entscheid {global_index}",
                }
            )
    return rows


class PinnedInputTests(unittest.TestCase):
    def test_validation_preserves_two_five_candidate_batches_per_query(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "input.jsonl"
            atomic_write_jsonl(input_path, validation_rows())
            with patch.object(run_soofi, "EXPECTED_INPUT_SHA256", sha256_file(input_path)):
                rows, batches = run_soofi._validate_pinned_input(input_path)

        self.assertEqual(len(rows), 100)
        self.assertEqual(len(batches), 20)
        self.assertTrue(all(len(batch) == 5 for batch in batches))
        self.assertEqual([row["rank"] for row in batches[0]], [1, 2, 3, 4, 5])
        self.assertEqual([row["rank"] for row in batches[1]], [6, 7, 8, 9, 10])

    def test_dry_run_needs_no_credential_and_writes_no_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.jsonl"
            output_dir = root / "output"
            atomic_write_jsonl(input_path, validation_rows())
            arguments = [
                "--input",
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--base-url",
                "https://example.invalid/api/",
                "--model",
                "test-model",
                "--dry-run",
            ]
            with (
                patch.object(run_soofi, "EXPECTED_INPUT_SHA256", sha256_file(input_path)),
                patch.object(run_soofi, "_reference_hashes", return_value={}) as preflight,
                patch.dict("os.environ", {}, clear=True),
            ):
                exit_code = run_soofi.main(arguments)
            preflight.assert_called_once()
            manifest_text = (output_dir / "manifest.json").read_text(encoding="utf-8")
            status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(status["state"], "dry_run")
        self.assertNotIn("SOOFI_API_KEY=", manifest_text)
        self.assertIn('"base_url": "https://example.invalid/api"', manifest_text)

    def test_configuration_rejects_unverified_http_endpoint(self) -> None:
        args = run_soofi.build_parser().parse_args(
            ["--base-url", "http://example.invalid/api", "--model", "test-model"]
        )
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            run_soofi._validated_config(args)


class ProviderModelIdentityTests(unittest.TestCase):
    def test_accepts_one_consistent_served_model(self) -> None:
        scores = [
            {"provider_model": "served-soofi"},
            {"provider_model": "served-soofi"},
        ]

        self.assertEqual(run_soofi._single_provider_model(scores), "served-soofi")

    def test_rejects_mixed_served_models(self) -> None:
        scores = [
            {"provider_model": "served-soofi-v1"},
            {"provider_model": "served-soofi-v2"},
        ]

        with self.assertRaisesRegex(ValueError, "exactly one served model"):
            run_soofi._single_provider_model(scores)

    def test_rejects_missing_served_model(self) -> None:
        invalid_values: tuple[object, ...] = (None, "", "   ", 7)
        for value in invalid_values:
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "non-empty provider_model"),
            ):
                run_soofi._single_provider_model([{"provider_model": value}])


if __name__ == "__main__":
    unittest.main()
