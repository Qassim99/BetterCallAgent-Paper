"""Stage 7: evaluate a submission with citation-level macro F1."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from bettercallagent.evaluation.metrics import macro_f1
from offline.io import atomic_write_json, load_queries, split_citations


def load_submission(path: Path) -> dict[str, set[str]]:
    """Load and validate exactly one prediction row per query."""

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = {"query_id", "predicted_citations"}
        missing = expected_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        predictions: dict[str, set[str]] = {}
        for row_number, row in enumerate(reader, start=2):
            query_id = str(row.get("query_id") or "").strip()
            if not query_id:
                raise ValueError(f"{path}:{row_number}: empty query_id")
            if query_id in predictions:
                raise ValueError(f"{path}:{row_number}: duplicate query_id {query_id!r}")
            predictions[query_id] = split_citations(row.get("predicted_citations"))
    return predictions


def evaluate(submission_path: Path, gold_path: Path) -> dict[str, object]:
    gold_rows = load_queries(gold_path, require_gold=True)
    gold = {query_id: split_citations(row["gold_citations"]) for query_id, row in gold_rows.items()}
    predictions = load_submission(submission_path)
    if set(predictions) != set(gold):
        missing = sorted(set(gold) - set(predictions))
        extra = sorted(set(predictions) - set(gold))
        raise ValueError(f"Submission query IDs differ: missing={missing}, extra={extra}")
    return asdict(macro_f1(predictions, gold))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = evaluate(args.submission, args.gold)
    atomic_write_json(args.output, metrics)
    print(f"macro_f1={float(metrics['macro_f1']):.6f} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
