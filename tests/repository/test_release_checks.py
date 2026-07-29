"""Unit tests for conservative release scanning."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.check_release import check_content, check_path, credential_problem


class ReleaseCheckTests(unittest.TestCase):
    def test_placeholders_are_allowed_but_real_assignments_are_not(self) -> None:
        self.assertFalse(credential_problem("API_KEY=replace-at-runtime"))
        self.assertFalse(credential_problem("auth_token = 'fixture-token-placeholder'"))
        self.assertTrue(credential_problem("API_KEY=" + "AbCdEf1234567890GhIjKlMnOpQrStUv"))
        self.assertFalse(
            credential_problem(
                "token = authorization.partition(' ')[2]\napi_key = settings.llm_api_key"
            )
        )

    def test_private_keys_and_cluster_paths_are_rejected(self) -> None:
        private_key = "-----BEGIN " + "PRIVATE KEY-----\nnot-a-real-key"
        self.assertTrue(credential_problem(private_key))
        self.assertEqual(
            check_content(
                Path("example.md"),
                "path=" + "/mnt/" + "vast-kisski/projects/private",
            ),
            ["example.md: machine-specific cluster path"],
        )

    def test_runtime_paths_and_large_files_are_rejected(self) -> None:
        self.assertTrue(check_path(Path("runs/output.json"), size=10))
        self.assertTrue(check_path(Path("artifact.bin"), size=11 * 1024 * 1024))


if __name__ == "__main__":
    unittest.main()
