"""Command-line interface for the offline interpretability package."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .error_attribution import attribute_citation_errors
from .gate_surrogate import fit_logistic_additive_surrogate
from .io import (
    load_gate_candidates,
    load_perturbation_cases,
    load_predictions,
    load_query_gold,
    load_retrieval,
    read_json,
    write_json,
)
from .perturbation_faithfulness import evaluate_perturbation_faithfulness
from .ragas_style import evaluate_ragas_style
from .report import build_report


def _add_shared_citation_artifacts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m interpretability",
        description=("Deterministic offline diagnostics for saved BetterCallAgent artifacts."),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    attribution = subparsers.add_parser(
        "attribute-errors", help="attribute citation errors to observable stages"
    )
    _add_shared_citation_artifacts(attribution)

    ragas = subparsers.add_parser(
        "ragas-style",
        help="compute custom citation metrics (not the official RAGAS library)",
    )
    _add_shared_citation_artifacts(ragas)
    ragas.add_argument("--context-k", type=int, default=10)

    surrogate = subparsers.add_parser(
        "gate-surrogate", help="fit a logistic/additive gate surrogate"
    )
    surrogate.add_argument("--input", required=True, type=Path)
    surrogate.add_argument("--output", required=True, type=Path)
    surrogate.add_argument("--iterations", type=int, default=2_000)
    surrogate.add_argument("--learning-rate", type=float, default=0.1)
    surrogate.add_argument("--l2-strength", type=float, default=0.01)
    surrogate.add_argument("--decision-threshold", type=float, default=0.5)

    faithfulness = subparsers.add_parser(
        "faithfulness-proxy",
        help="evaluate seeded evidence and random-control perturbations",
    )
    faithfulness.add_argument("--input", required=True, type=Path)
    faithfulness.add_argument("--output", required=True, type=Path)
    faithfulness.add_argument("--seed", type=int, default=17)
    faithfulness.add_argument("--random-trials", type=int, default=1_000)

    report = subparsers.add_parser("render-report", help="render known component JSON files")
    report.add_argument("--input", required=True, action="append", type=Path)
    report.add_argument("--json-output", required=True, type=Path)
    report.add_argument("--markdown-output", required=True, type=Path)
    report.add_argument("--title", default="BetterCallAgent interpretability report")
    return parser


def _run(args: argparse.Namespace) -> None:
    if args.command in {"attribute-errors", "ragas-style"}:
        queries = load_query_gold(args.queries)
        retrieval = load_retrieval(args.retrieval)
        predictions = load_predictions(args.predictions)
        if args.command == "attribute-errors":
            payload = attribute_citation_errors(queries, retrieval, predictions)
        else:
            payload = evaluate_ragas_style(
                queries, retrieval, predictions, context_k=args.context_k
            )
        write_json(args.output, payload)
        return

    if args.command == "gate-surrogate":
        payload = fit_logistic_additive_surrogate(
            load_gate_candidates(args.input),
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            l2_strength=args.l2_strength,
            decision_threshold=args.decision_threshold,
        )
        write_json(args.output, payload)
        return

    if args.command == "faithfulness-proxy":
        payload = evaluate_perturbation_faithfulness(
            load_perturbation_cases(args.input),
            seed=args.seed,
            random_trials=args.random_trials,
        )
        write_json(args.output, payload)
        return

    if args.command == "render-report":
        json_report, markdown = build_report(
            [read_json(path) for path in args.input], title=args.title
        )
        write_json(args.json_output, json_report)
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
        return

    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _run(args)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
