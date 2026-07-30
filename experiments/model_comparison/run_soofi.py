"""Run a restartable SOOFI verifier comparison on the pinned validation input."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bettercallagent.citations.extract import CitationExtractor
from bettercallagent.citations.policy import FixedVotePolicy
from bettercallagent.providers.openai_compatible import OpenAICompatibleProvider
from offline.io import (
    JsonObject,
    atomic_write_json,
    atomic_write_jsonl,
    load_queries,
    read_jsonl,
    sha256_file,
    sha256_tree,
)
from offline.stages.step_02_retrieve_sparse_support import (
    aggregate_saved_traces,
    discover_trace_citations,
)
from offline.stages.step_05_rerank import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    prepare_batches,
    rerank_live,
)
from offline.stages.step_06_select_citations import (
    atomic_write_submission,
    select_for_queries,
)
from offline.stages.step_07_evaluate import evaluate
from offline.vocabulary import load_targeted_vocabulary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_INPUT_SHA256 = "a2624d1c15bf922a542f03e941f5e92219d75176e2e47daac7ca90d8ae5dc374"
EXPECTED_CANDIDATES = 100
EXPECTED_BATCHES = 20
BATCH_SIZE = 5
REFERENCE_HASHES = {
    "queries": "41862ef772801995cae80cf5ea947b08e197603aa37ef93aefb632e0d9de5f7f",
    "sparse_traces": "a80e96e9ff4d8f10f4d33ec07502de72fbdd54456f08deb40eea3c881aa26ae6",
    "laws": "6602c06fbfe83ee9942f05083402fe5265e43df7be6b48748c7be4f52650609e",
    "courts": "a5adb6ceea4bb057b816800460bcb5b6fded7beb5c4e9e29e3dee5a777077017",
}


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Semantic settings that identify one comparable verifier run."""

    base_url: str
    model: str
    api_key_env: str
    max_tokens: int
    temperature: float
    timeout_seconds: int
    max_attempts: int
    retry_delay_seconds: float
    document_char_limit: int
    json_response: bool
    enable_thinking: bool


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def _validate_pinned_input(path: Path) -> tuple[list[JsonObject], list[list[JsonObject]]]:
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_INPUT_SHA256:
        raise ValueError(
            "Reranker input checksum differs from the audited validation artifact: "
            f"expected={EXPECTED_INPUT_SHA256}, actual={actual_hash}"
        )
    rows = list(read_jsonl(path))
    batches = prepare_batches(rows, batch_size=BATCH_SIZE)
    if len(rows) != EXPECTED_CANDIDATES or len(batches) != EXPECTED_BATCHES:
        raise ValueError("Pinned reranker input must contain exactly 100 candidates in 20 batches.")
    if any(len(batch) != BATCH_SIZE for batch in batches):
        raise ValueError("Every historical-comparison batch must contain exactly five candidates.")
    for batch_index in range(0, len(batches), 2):
        first, second = batches[batch_index : batch_index + 2]
        query_ids = {str(row["query_id"]) for row in (*first, *second)}
        ranks = {int(row["rank"]) for row in (*first, *second)}
        if len(query_ids) != 1 or ranks != set(range(1, 11)):
            raise ValueError("Each query must contribute ranks 1-10 in two ordered batches.")
    return rows, batches


def _check_resume_manifest(
    path: Path,
    *,
    config_sha256: str,
    input_sha256: str,
) -> None:
    if not path.exists():
        return
    previous = _load_json(path)
    if previous.get("config_sha256") != config_sha256:
        raise ValueError("Output directory belongs to a different experiment configuration.")
    inputs = previous.get("inputs")
    if not isinstance(inputs, dict) or inputs.get("reranker_input_sha256") != input_sha256:
        raise ValueError("Output directory belongs to a different reranker input.")


def _reference_hashes(args: argparse.Namespace) -> dict[str, str]:
    actual = {
        "queries": sha256_file(args.queries),
        "sparse_traces": sha256_tree(args.sparse_traces)[0],
        "laws": sha256_file(args.laws),
        "courts": sha256_file(args.courts),
    }
    if actual != REFERENCE_HASHES:
        differences = {
            name: {"expected": REFERENCE_HASHES[name], "actual": actual[name]}
            for name in REFERENCE_HASHES
            if actual[name] != REFERENCE_HASHES[name]
        }
        raise ValueError(f"Evaluation artifact checksum mismatch: {differences}")
    return actual


def _evaluate_complete_scores(
    *,
    args: argparse.Namespace,
    prepared: list[JsonObject],
    scores: list[JsonObject],
) -> tuple[dict[str, object], dict[str, str]]:
    """Apply the existing sparse-support gate and macro-F1 evaluator."""

    reference_hashes = _reference_hashes(args)
    queries = load_queries(args.queries, require_gold=True)
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

    output_dir = args.output_dir
    support_path = output_dir / "sparse_support.jsonl"
    submission_path = output_dir / "submission.csv"
    audit_path = output_dir / "selection_audit.json"
    metrics_path = output_dir / "metrics.json"
    atomic_write_jsonl(support_path, support)
    atomic_write_submission(submission_path, query_ids=queries, predictions=predictions)
    atomic_write_json(audit_path, audit)
    metrics = evaluate(submission_path, args.queries)
    atomic_write_json(metrics_path, metrics)
    output_hashes = {
        "scores_sha256": sha256_file(output_dir / "scores.jsonl"),
        "sparse_support_sha256": sha256_file(support_path),
        "submission_sha256": sha256_file(submission_path),
        "selection_audit_sha256": sha256_file(audit_path),
        "metrics_sha256": sha256_file(metrics_path),
    }
    return metrics, {**reference_hashes, **output_hashes}


def _single_provider_model(scores: list[JsonObject]) -> str:
    """Require one non-empty served-model identity across every scored candidate."""

    provider_models: set[str] = set()
    for row in scores:
        raw_model = row.get("provider_model")
        if not isinstance(raw_model, str) or not raw_model.strip():
            raise ValueError("Every verifier score must record a non-empty provider_model.")
        provider_models.add(raw_model.strip())
    if len(provider_models) != 1:
        raise ValueError(
            "A comparable verifier run must use exactly one served model; "
            f"provider returned {sorted(provider_models)}."
        )
    return next(iter(provider_models))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "artifacts/downloads/derived/val_qwen_top10_fulltext.jsonl",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=PROJECT_ROOT / "artifacts/downloads/eval/val.csv",
    )
    parser.add_argument(
        "--sparse-traces",
        type=Path,
        default=PROJECT_ROOT / "artifacts/downloads/derived/sira_bm25_traces",
    )
    parser.add_argument(
        "--laws",
        type=Path,
        default=PROJECT_ROOT / "artifacts/downloads/eval/laws_de.csv",
    )
    parser.add_argument(
        "--courts",
        type=Path,
        default=PROJECT_ROOT / "artifacts/downloads/eval/court_considerations.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs/model_comparison/soofi",
    )
    parser.add_argument("--base-url", default=os.environ.get("SOOFI_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("SOOFI_MODEL"))
    parser.add_argument("--api-key-env", default="SOOFI_API_KEY")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=16_384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--document-char-limit", type=int, default=0)
    parser.add_argument(
        "--json-response",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request provider-side JSON mode; disabled for the hosted SOOFI endpoint by default.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward chat_template_kwargs.enable_thinking (default: false).",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        help="Run only the first N ordered batches; useful for a smoke test.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and plan without an API call."
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Stop after producing a complete 100-score artifact.",
    )
    return parser


def _validated_config(args: argparse.Namespace) -> RunConfig:
    if not args.base_url:
        raise ValueError("Provide --base-url or SOOFI_BASE_URL.")
    if not args.model:
        raise ValueError("Provide --model or SOOFI_MODEL.")
    model = str(args.model).strip()
    if not model:
        raise ValueError("The configured model name must not be blank.")
    api_key_env = str(args.api_key_env).strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
        raise ValueError("--api-key-env must be a valid environment-variable name.")
    url_validator = OpenAICompatibleProvider(
        base_url=str(args.base_url),
        api_key="placeholder",
    )
    if args.limit_batches is not None and not 1 <= args.limit_batches <= EXPECTED_BATCHES:
        raise ValueError(f"--limit-batches must be in [1, {EXPECTED_BATCHES}].")
    if min(args.timeout, args.max_attempts, args.max_tokens) <= 0:
        raise ValueError("timeout, max-attempts, and max-tokens must be positive.")
    if args.retry_delay_seconds < 0 or args.document_char_limit < 0:
        raise ValueError("retry delay and document character limit cannot be negative.")
    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError("temperature must be in [0, 2].")
    return RunConfig(
        base_url=url_validator.base_url,
        model=model,
        api_key_env=api_key_env,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
        max_attempts=args.max_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
        document_char_limit=args.document_char_limit,
        json_response=args.json_response,
        enable_thinking=args.enable_thinking,
    )


async def _run(args: argparse.Namespace) -> int:
    config = _validated_config(args)
    rows, batches = _validate_pinned_input(args.input)
    if not args.skip_evaluation:
        _reference_hashes(args)
    selected_batches = batches[: args.limit_batches] if args.limit_batches else batches
    selected_rows = [row for batch in selected_batches for row in batch]
    input_hash = sha256_file(args.input)
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    config_payload = {
        **asdict(config),
        "batch_size": BATCH_SIZE,
        "concurrency": 1,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": prompt_hash,
        "reranker_input_sha256": input_hash,
    }
    config_hash = _canonical_sha256(config_payload)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    status_path = args.output_dir / "status.json"
    checkpoint_path = args.output_dir / "batches.jsonl"
    scores_path = args.output_dir / "scores.jsonl"
    _check_resume_manifest(
        manifest_path,
        config_sha256=config_hash,
        input_sha256=input_hash,
    )
    manifest: JsonObject = {
        "schema_version": 1,
        "scope": "fresh_soofi_verifier_comparison",
        "config_sha256": config_hash,
        "config": config_payload,
        "inputs": {
            "reranker_input": str(args.input.resolve()),
            "reranker_input_sha256": input_hash,
        },
        "prompt": {
            "version": PROMPT_VERSION,
            "system_prompt_sha256": prompt_hash,
        },
        "expected_candidates": EXPECTED_CANDIDATES,
        "expected_batches": EXPECTED_BATCHES,
    }
    atomic_write_json(manifest_path, manifest)
    planned_status = {
        "schema_version": 1,
        "state": "dry_run" if args.dry_run else "running",
        "updated_at": _utc_now(),
        "planned_batches": len(selected_batches),
        "planned_candidates": len(selected_rows),
        "config_sha256": config_hash,
    }
    atomic_write_json(status_path, planned_status)
    if args.dry_run:
        print(json.dumps(planned_status, indent=2, sort_keys=True))
        return 0

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing required environment variable: {config.api_key_env}")
    provider = OpenAICompatibleProvider(
        base_url=config.base_url,
        api_key=api_key,
        timeout_seconds=config.timeout_seconds,
        chat_template_kwargs={"enable_thinking": config.enable_thinking},
    )
    try:
        scores = await rerank_live(
            selected_rows,
            provider=provider,
            model=config.model,
            batch_size=BATCH_SIZE,
            concurrency=1,
            document_char_limit=config.document_char_limit,
            checkpoint_path=checkpoint_path,
            max_attempts=config.max_attempts,
            retry_delay_seconds=config.retry_delay_seconds,
            max_tokens=config.max_tokens,
            json_response=config.json_response,
            temperature=config.temperature,
            chat_template_kwargs={"enable_thinking": config.enable_thinking},
        )
    finally:
        await provider.close()
    provider_model = _single_provider_model(scores)
    atomic_write_jsonl(scores_path, scores)

    complete = len(selected_batches) == EXPECTED_BATCHES
    metrics: dict[str, object] | None = None
    output_hashes: dict[str, str] = {"scores_sha256": sha256_file(scores_path)}
    state = "scores_complete" if complete else "partial"
    if complete and not args.skip_evaluation:
        metrics, evaluated_hashes = _evaluate_complete_scores(
            args=args,
            prepared=rows,
            scores=scores,
        )
        output_hashes.update(evaluated_hashes)
        state = "complete"
    manifest.update(
        {
            "state": state,
            "completed_candidates": len(scores),
            "completed_batches": len(selected_batches),
            "provider_models": [provider_model],
            "outputs": output_hashes,
            "metrics": metrics,
        }
    )
    atomic_write_json(manifest_path, manifest)
    final_status = {
        "schema_version": 1,
        "state": state,
        "updated_at": _utc_now(),
        "completed_batches": len(selected_batches),
        "completed_candidates": len(scores),
        "config_sha256": config_hash,
    }
    atomic_write_json(status_path, final_status)
    print(json.dumps(final_status, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except BaseException as exc:
        if args.output_dir.exists():
            atomic_write_json(
                args.output_dir / "status.json",
                {
                    "schema_version": 1,
                    "state": "failed",
                    "updated_at": _utc_now(),
                    "error_type": type(exc).__name__,
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
