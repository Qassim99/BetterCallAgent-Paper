"""Reproduce the reported metric from the incomplete historical score artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from bettercallagent.citations.extract import CitationExtractor
from bettercallagent.citations.policy import FixedVotePolicy
from offline.identity import candidate_id
from offline.io import (
    JsonObject,
    atomic_write_json,
    load_queries,
    read_jsonl,
    sha256_file,
    sha256_tree,
)
from offline.stages.step_02_retrieve_sparse_support import (
    aggregate_saved_traces,
    discover_trace_citations,
)
from offline.stages.step_06_select_citations import (
    atomic_write_submission,
    select_for_queries,
)
from offline.stages.step_07_evaluate import evaluate
from offline.vocabulary import load_targeted_vocabulary

EXPECTED_MACRO_F1 = 0.4806246255438345
EXPECTED_MODEL_KEY = "qwen_ac"
EXPECTED_INPUT_HASHES = {
    "queries": "41862ef772801995cae80cf5ea947b08e197603aa37ef93aefb632e0d9de5f7f",
    "reranker_input": "a2624d1c15bf922a542f03e941f5e92219d75176e2e47daac7ca90d8ae5dc374",
    "legacy_scores": "aac9b173d6cb6241b0cd1dd22e0409a5470f1b6a5ddf7575eff926e83c42859c",
    "sparse_traces": "a80e96e9ff4d8f10f4d33ec07502de72fbdd54456f08deb40eea3c881aa26ae6",
    "laws": "6602c06fbfe83ee9942f05083402fe5265e43df7be6b48748c7be4f52650609e",
    "courts": "a5adb6ceea4bb057b816800460bcb5b6fded7beb5c4e9e29e3dee5a777077017",
}
EXPECTED_MISSING_IDS = {
    "val_002::rank6::gi103007",
    "val_002::rank7::gi491948",
    "val_002::rank8::gi767176",
    "val_002::rank9::gi225060",
    "val_002::rank10::gi811960",
}


def legacy_candidate_id(row: Mapping[str, Any]) -> str:
    """Reconstruct the identifier emitted by the historical batch prompt."""

    return f"{row['query_id']}::rank{int(row['rank'])}::gi{int(row['global_idx'])}"


def load_historical_scores(
    *,
    prepared_rows: Iterable[JsonObject],
    results_path: Path,
    model_key: str,
) -> tuple[list[JsonObject], list[JsonObject], list[str], int]:
    """Join legacy batch scores to documents and expose every missing identity."""

    prepared = list(prepared_rows)
    by_legacy_id: dict[str, JsonObject] = {}
    for row in prepared:
        identifier = legacy_candidate_id(row)
        if identifier in by_legacy_id:
            raise ValueError(f"Duplicate historical candidate ID: {identifier}")
        by_legacy_id[identifier] = row

    scores_by_identity: dict[tuple[str, str], JsonObject] = {}
    batch_count = 0
    for result in read_jsonl(results_path):
        if result.get("model_key") != model_key:
            continue
        if not result.get("ok", False):
            raise ValueError(f"Historical batch is marked failed: {result.get('batch_id')}")
        batch_count += 1
        raw_scores = result.get("scores")
        if not isinstance(raw_scores, list):
            raise ValueError("Historical batch scores must be a list.")
        for raw_score in raw_scores:
            if not isinstance(raw_score, dict):
                raise ValueError("Historical score entries must be objects.")
            identifier = str(raw_score.get("candidate_id") or "")
            if identifier not in by_legacy_id:
                raise ValueError(f"Score references unknown candidate: {identifier}")
            row = by_legacy_id[identifier]
            identity = (str(row["query_id"]), str(row["doc_id"]))
            if identity in scores_by_identity:
                raise ValueError(f"Duplicate historical score for {identity}.")
            scores_by_identity[identity] = {
                "query_id": row["query_id"],
                "doc_id": row["doc_id"],
                "global_idx": row.get("global_idx"),
                "candidate_id": candidate_id(row),
                "rank": int(row["rank"]),
                "score": float(raw_score["score"]),
                "confidence": float(raw_score.get("confidence") or 0.0),
                "rationale": str(raw_score.get("rationale_de") or "historical replay"),
            }

    filtered = [
        row for row in prepared if (str(row["query_id"]), str(row["doc_id"])) in scores_by_identity
    ]
    missing = sorted(
        identifier
        for identifier, row in by_legacy_id.items()
        if (str(row["query_id"]), str(row["doc_id"])) not in scores_by_identity
    )
    return filtered, list(scores_by_identity.values()), missing, batch_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--reranker-input", type=Path, required=True)
    parser.add_argument("--legacy-scores", type=Path, required=True)
    parser.add_argument("--sparse-traces", type=Path, required=True)
    parser.add_argument("--laws", type=Path, required=True)
    parser.add_argument("--courts", type=Path, required=True)
    parser.add_argument("--model-key", default=EXPECTED_MODEL_KEY)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.model_key != EXPECTED_MODEL_KEY:
        raise ValueError(f"Historical replay requires model key {EXPECTED_MODEL_KEY!r}.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    actual_input_hashes = {
        "queries": sha256_file(args.queries),
        "reranker_input": sha256_file(args.reranker_input),
        "legacy_scores": sha256_file(args.legacy_scores),
        "sparse_traces": sha256_tree(args.sparse_traces)[0],
        "laws": sha256_file(args.laws),
        "courts": sha256_file(args.courts),
    }
    if actual_input_hashes != EXPECTED_INPUT_HASHES:
        differences = {
            name: {
                "expected": EXPECTED_INPUT_HASHES[name],
                "actual": actual_input_hashes[name],
            }
            for name in EXPECTED_INPUT_HASHES
            if actual_input_hashes[name] != EXPECTED_INPUT_HASHES[name]
        }
        raise ValueError(f"Historical artifact checksum mismatch: {differences}")

    queries = load_queries(args.queries, require_gold=True)
    all_prepared = list(read_jsonl(args.reranker_input))
    prepared, scores, missing, batch_count = load_historical_scores(
        prepared_rows=all_prepared,
        results_path=args.legacy_scores,
        model_key=args.model_key,
    )
    if (
        len(all_prepared) != 100
        or len(scores) != 95
        or batch_count != 19
        or set(missing) != EXPECTED_MISSING_IDS
    ):
        raise ValueError("Historical artifact shape differs from the audited 19/20-batch run.")

    extractor = CitationExtractor()
    candidates = discover_trace_citations(
        trace_root=args.sparse_traces,
        query_ids=queries,
        top_k=5,
    )
    for row in prepared:
        candidates.update(extractor.extract(str(row.get("document_text") or "")))
    vocabulary = load_targeted_vocabulary(
        laws_path=args.laws,
        courts_path=args.courts,
        candidates=candidates,
    )
    support = aggregate_saved_traces(
        trace_root=args.sparse_traces,
        query_ids=queries,
        vocabulary=vocabulary,
        top_k=5,
    )
    predictions, audit = select_for_queries(
        query_ids=queries,
        prepared_rows=prepared,
        score_rows=scores,
        support_rows=support,
        vocabulary=vocabulary,
        policy=FixedVotePolicy(),
    )

    submission_path = args.output / "submission.csv"
    audit_path = args.output / "selection_audit.json"
    metrics_path = args.output / "metrics.json"
    manifest_path = args.output / "historical_replay_manifest.json"
    atomic_write_submission(
        submission_path,
        query_ids=queries,
        predictions=predictions,
    )
    atomic_write_json(audit_path, audit)
    metrics = evaluate(submission_path, args.queries)
    atomic_write_json(metrics_path, metrics)
    if abs(float(metrics["macro_f1"]) - EXPECTED_MACRO_F1) > 1e-12:
        raise AssertionError(
            f"Historical replay drifted: {metrics['macro_f1']} != {EXPECTED_MACRO_F1}"
        )

    manifest = {
        "schema_version": 1,
        "scope": "historical_incomplete_validation_replay",
        "model_key": args.model_key,
        "completed_batches": batch_count,
        "prepared_candidates": len(all_prepared),
        "scored_candidates": len(scores),
        "missing_score_count": len(missing),
        "missing_score_ids": missing,
        "inputs": {
            "queries_sha256": actual_input_hashes["queries"],
            "reranker_input_sha256": actual_input_hashes["reranker_input"],
            "legacy_scores_sha256": actual_input_hashes["legacy_scores"],
            "sparse_traces_tree_sha256": actual_input_hashes["sparse_traces"],
            "laws_sha256": actual_input_hashes["laws"],
            "courts_sha256": actual_input_hashes["courts"],
        },
        "metrics": metrics,
        "limitations": [
            "Only 19 of 20 verifier batches completed.",
            "val_002 ranks 6-10 have no verifier scores.",
            "Thresholds were selected on the same ten validation queries.",
        ],
    }
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "macro_f1": metrics["macro_f1"],
                "scored_candidates": len(scores),
                "missing_scores": len(missing),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
