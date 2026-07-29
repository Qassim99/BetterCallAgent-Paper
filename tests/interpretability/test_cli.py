from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPOSITORY_ROOT / "interpretability" / "examples"


def _run(*arguments: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "interpretability", *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)


class CliSmokeTests(unittest.TestCase):
    def test_complete_cli_smoke_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            self._run_workflow(Path(temporary_directory))

    def _run_workflow(self, output_directory: Path) -> None:
        error_output = output_directory / "error.json"
        ragas_output = output_directory / "ragas.json"
        surrogate_output = output_directory / "surrogate.json"
        faithfulness_output = output_directory / "faithfulness.json"
        report_json = output_directory / "report.json"
        report_markdown = output_directory / "report.md"

        shared = (
            "--queries",
            str(EXAMPLES / "queries.jsonl"),
            "--retrieval",
            str(EXAMPLES / "retrieval.jsonl"),
            "--predictions",
            str(EXAMPLES / "predictions.jsonl"),
        )
        _run("attribute-errors", *shared, "--output", str(error_output))
        _run(
            "ragas-style",
            *shared,
            "--context-k",
            "10",
            "--output",
            str(ragas_output),
        )
        _run(
            "gate-surrogate",
            "--input",
            str(EXAMPLES / "gate_candidates.jsonl"),
            "--iterations",
            "300",
            "--output",
            str(surrogate_output),
        )
        _run(
            "faithfulness-proxy",
            "--input",
            str(EXAMPLES / "perturbations.jsonl"),
            "--seed",
            "17",
            "--random-trials",
            "50",
            "--output",
            str(faithfulness_output),
        )
        _run(
            "render-report",
            "--input",
            str(error_output),
            "--input",
            str(ragas_output),
            "--input",
            str(surrogate_output),
            "--input",
            str(faithfulness_output),
            "--json-output",
            str(report_json),
            "--markdown-output",
            str(report_markdown),
        )

        report = json.loads(report_json.read_text(encoding="utf-8"))
        markdown = report_markdown.read_text(encoding="utf-8")
        self.assertEqual(
            report["schema_version"],
            "bettercallagent.interpretability.report.v1",
        )
        self.assertEqual(len(report["components"]), 4)
        self.assertIn("Custom `ragas_style` citation metrics", markdown)
        self.assertIn("Logistic/additive citation-gate surrogate", markdown)
