"""Run the seven-stage offline pipeline from one explicit TOML configuration."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.metadata
import os
import platform
import random
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import partial
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
    require_file,
    sha256_file,
    sha256_tree,
)
from offline.stages.step_01_retrieve_dense import (
    DEFAULT_WEIGHTS,
    DenseIndexManifest,
    QwenEmbeddingEncoder,
    retrieve_from_index,
    retrieve_from_saved_rankings,
)
from offline.stages.step_02_retrieve_sparse_support import (
    aggregate_saved_traces,
    discover_trace_citations,
)
from offline.stages.step_03_materialize_documents import materialize
from offline.stages.step_04_prepare_rerank_input import prepare
from offline.stages.step_05_rerank import (
    PROMPT_VERSION,
    rerank_from_replay,
    rerank_live,
)
from offline.stages.step_06_select_citations import (
    atomic_write_submission,
    select_for_queries,
)
from offline.stages.step_07_evaluate import evaluate
from offline.vocabulary import load_targeted_vocabulary

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _table(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration requires a [{name}] table.")
    return value


def _path(value: Any, *, name: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Configuration path {name!r} is required.")
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def _has_gold(path: Path) -> bool:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return "gold_citations" in set(csv.DictReader(handle).fieldnames or [])


def _git_state() -> dict[str, Any]:
    """Describe the exact Git state and make dirty runs explicit."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        digest = hashlib.sha256()
        digest.update(status.encode("utf-8"))
        digest.update(diff)
        for raw_path in sorted(item for item in untracked if item):
            path = REPOSITORY_ROOT / os.fsdecode(raw_path)
            if path.is_file():
                digest.update(raw_path)
                digest.update(b"\0")
                digest.update(bytes.fromhex(sha256_file(path)))
        return {
            "commit": commit,
            "dirty": bool(status),
            "working_tree_sha256": digest.hexdigest() if status else None,
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": "unavailable",
            "dirty": None,
            "working_tree_sha256": None,
        }


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("fastapi", "numpy", "pyarrow", "pydantic", "torch", "transformers"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _validate_document_replay(path: Path) -> list[JsonObject]:
    rows = list(read_jsonl(path))
    seen: set[str] = set()
    for row in rows:
        doc_id = str(row.get("doc_id") or "").strip()
        text = str(row.get("full_document_text") or "").strip()
        if not doc_id or not text or doc_id in seen:
            raise ValueError(
                "Document replay requires unique doc_id values and non-empty full text."
            )
        seen.add(doc_id)
    if not rows:
        raise ValueError("Document replay is empty.")
    return rows


def _weights(config: Mapping[str, Any]) -> dict[str, float]:
    raw = config.get("weights") or DEFAULT_WEIGHTS
    if not isinstance(raw, dict):
        raise ValueError("retrieval.weights must be a TOML inline table.")
    weights = {str(key): float(value) for key, value in raw.items()}
    if set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError(f"retrieval.weights must contain exactly {sorted(DEFAULT_WEIGHTS)}.")
    return weights


def _input_provenance(
    *,
    files: Mapping[str, Path],
    directories: Mapping[str, Path],
) -> dict[str, JsonObject]:
    output: dict[str, JsonObject] = {}
    for name, path in sorted(files.items()):
        resolved = require_file(path, description=name)
        output[name] = {
            "path": _relative(resolved),
            "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size,
        }
    for name, path in sorted(directories.items()):
        tree_digest, file_count, total_bytes = sha256_tree(path)
        output[name] = {
            "path": _relative(path),
            "tree_sha256": tree_digest,
            "files": file_count,
            "bytes": total_bytes,
        }
    return output


def _verify_integrity(
    *,
    mode: str,
    config: Mapping[str, Any],
    provenance: Mapping[str, JsonObject],
) -> None:
    """Bind full-paper inputs to checksums committed in the run configuration."""

    raw = config.get("integrity")
    if mode == "fixture" and raw is None:
        return
    if not isinstance(raw, dict):
        raise ValueError("Full mode requires an [integrity] checksum table.")
    required = {
        "queries": ("queries_sha256", "sha256"),
        "laws": ("laws_sha256", "sha256"),
        "courts": ("courts_sha256", "sha256"),
        "sparse_traces": ("sparse_traces_tree_sha256", "tree_sha256"),
    }
    if mode == "full":
        required.update(
            {
                "dense_index": ("dense_index_tree_sha256", "tree_sha256"),
                "document_store": ("document_store_tree_sha256", "tree_sha256"),
            }
        )
    if "reranker_replay" in provenance:
        required["reranker_replay"] = (
            "reranker_replay_sha256",
            "sha256",
        )
    for input_name, (config_name, provenance_field) in required.items():
        expected = str(raw.get(config_name) or "").lower()
        actual = str(provenance[input_name][provenance_field]).lower()
        if len(expected) != 64:
            raise ValueError(f"integrity.{config_name} must be a SHA-256 digest.")
        if actual != expected:
            raise ValueError(f"Input integrity mismatch for {input_name}: {actual} != {expected}")


async def run_pipeline(
    config_path: Path,
    *,
    output_override: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute all applicable stages and return the run manifest."""

    resolved_config = require_file(config_path, description="pipeline configuration")
    with resolved_config.open("rb") as handle:
        config = tomllib.load(handle)
    if int(config.get("schema_version") or 0) != 1:
        raise ValueError("Only offline configuration schema_version = 1 is supported.")
    mode = str(config.get("mode") or "full")
    if mode not in {"fixture", "full"}:
        raise ValueError("mode must be either 'fixture' or 'full'.")
    seed = int(config.get("seed") or 0)
    random.seed(seed)

    paths = _table(config, "paths")
    retrieval = _table(config, "retrieval")
    reranking = _table(config, "reranking")
    gate = _table(config, "citation_gate")
    weights = _weights(retrieval)
    configured_fields = tuple(retrieval.get("fields") or ())
    if configured_fields and configured_fields != tuple(DEFAULT_WEIGHTS):
        raise ValueError(
            "retrieval.fields must preserve the five configured index fields in order."
        )

    retrieval_model = str(retrieval.get("model") or "")
    retrieval_revision = str(retrieval.get("model_revision") or "")
    retrieval_device = str(retrieval.get("device") or "cuda:0")
    retrieval_batch_size = int(retrieval.get("batch_size") or 16)
    retrieval_max_length = int(retrieval.get("max_length") or 1_024)
    retrieval_chunk_size = int(retrieval.get("chunk_size") or 30_000)
    retrieval_field_top_k = int(retrieval.get("field_top_k") or 1_000)
    retrieval_fusion_top_k = int(retrieval.get("fusion_top_k") or 1_000)
    retrieval_rrf_k = int(retrieval.get("rrf_k") or 60)
    retrieval_allow_download = bool(retrieval.get("allow_model_download", False))

    reranker_model = str(reranking.get("model") or "").strip()
    if not reranker_model:
        raise ValueError("reranking.model is required.")
    reranker_top_n = int(reranking.get("top_n") or 10)
    reranker_batch_size = int(reranking.get("batch_size") or 5)
    reranker_concurrency = int(reranking.get("concurrency") or 1)
    reranker_timeout = int(reranking.get("timeout_seconds") or 300)
    reranker_document_limit = int(reranking.get("document_char_limit") or 0)
    reranker_max_attempts = int(reranking.get("max_attempts") or 3)
    reranker_retry_delay = float(reranking.get("retry_delay_seconds", 1.0))
    reranker_max_tokens = int(reranking.get("max_tokens") or 4_096)
    positive_values = {
        "retrieval.batch_size": retrieval_batch_size,
        "retrieval.max_length": retrieval_max_length,
        "retrieval.chunk_size": retrieval_chunk_size,
        "retrieval.field_top_k": retrieval_field_top_k,
        "retrieval.fusion_top_k": retrieval_fusion_top_k,
        "retrieval.rrf_k": retrieval_rrf_k,
        "reranking.top_n": reranker_top_n,
        "reranking.batch_size": reranker_batch_size,
        "reranking.concurrency": reranker_concurrency,
        "reranking.timeout_seconds": reranker_timeout,
        "reranking.max_attempts": reranker_max_attempts,
        "reranking.max_tokens": reranker_max_tokens,
    }
    invalid_positive = sorted(name for name, value in positive_values.items() if value <= 0)
    if invalid_positive:
        raise ValueError("Configuration values must be positive: " + ", ".join(invalid_positive))
    if reranker_document_limit < 0 or reranker_retry_delay < 0:
        raise ValueError(
            "reranking.document_char_limit and retry_delay_seconds cannot be negative."
        )
    if mode == "full" and (not retrieval_model or not retrieval_revision):
        raise ValueError("Full mode requires retrieval.model and an immutable model_revision.")

    policy = FixedVotePolicy(
        candidate_top_k=int(gate.get("candidate_top_k") or gate.get("top_k") or 10),
        minimum_dense_votes=int(gate.get("minimum_dense_votes") or gate.get("min_votes") or 4),
        anchor_top_k=int(gate.get("anchor_top_k") or 3),
        anchor_minimum_score=float(
            gate.get("anchor_minimum_score") or gate.get("anchor_score") or 8.5
        ),
        bm25_top_k=int(gate.get("sparse_top_k") or gate.get("bm25_top_k") or 5),
        minimum_bm25_votes=int(gate.get("minimum_sparse_votes") or gate.get("bm25_min_votes") or 2),
    )

    queries_path = _path(paths.get("queries"), name="paths.queries")
    laws_path = _path(paths.get("laws"), name="paths.laws")
    courts_path = _path(paths.get("courts"), name="paths.courts")
    sparse_trace_root = _path(
        paths.get("sparse_traces"),
        name="paths.sparse_traces",
    )
    output_dir = (
        output_override.expanduser().resolve()
        if output_override is not None
        else _path(paths.get("output"), name="paths.output")
    )
    stage_paths = {
        "dense_candidates": output_dir / "01_dense_candidates.jsonl",
        "dense_summary": output_dir / "01_dense_summary.json",
        "sparse_support": output_dir / "02_sparse_support.jsonl",
        "documents": output_dir / "03_documents.jsonl",
        "reranker_input": output_dir / "04_reranker_input.jsonl",
        "reranker_batches": output_dir / "05_reranker_batches.jsonl",
        "reranker_scores": output_dir / "05_reranker_scores.jsonl",
        "submission": output_dir / "06_submission.csv",
        "selection_audit": output_dir / "06_selection_audit.json",
        "metrics": output_dir / "07_metrics.json",
        "manifest": output_dir / "run_manifest.json",
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(
                f"Output directory is not empty; choose a new run directory: {output_dir}"
            )
        unknown = sorted(
            item.name
            for item in output_dir.iterdir()
            if item.name not in {path.name for path in stage_paths.values()}
        )
        if unknown:
            raise ValueError("Resume directory contains unknown files: " + ", ".join(unknown))
    output_dir.mkdir(parents=True, exist_ok=True)
    input_files: dict[str, Path] = {
        "config": resolved_config,
        "queries": queries_path,
        "laws": laws_path,
        "courts": courts_path,
    }
    input_directories = {"sparse_traces": sparse_trace_root}
    saved_rankings: Path | None = None
    document_replay: Path | None = None
    index_dir: Path | None = None
    document_root: Path | None = None
    reranker_replay: Path | None = None
    if mode == "fixture":
        saved_rankings = _path(
            paths.get("saved_rankings"),
            name="paths.saved_rankings",
        )
        document_replay = _path(
            paths.get("documents_replay"),
            name="paths.documents_replay",
        )
        if not paths.get("reranker_replay"):
            raise ValueError("Fixture mode requires paths.reranker_replay; network is forbidden.")
        input_files["saved_rankings"] = saved_rankings
        input_files["documents_replay"] = document_replay
    else:
        index_dir = _path(paths.get("index"), name="paths.index")
        document_root = _path(
            paths.get("document_store"),
            name="paths.document_store",
        )
        input_directories["dense_index"] = index_dir
        input_directories["document_store"] = document_root
    if paths.get("reranker_replay"):
        reranker_replay = _path(
            paths.get("reranker_replay"),
            name="paths.reranker_replay",
        )
        input_files["reranker_replay"] = reranker_replay

    provider_base_url: str | None = None
    provider: OpenAICompatibleProvider | None = None
    if reranker_replay is None:
        base_url_env = str(reranking.get("base_url_env") or "BCA_RERANK_BASE_URL")
        api_key_env = str(reranking.get("api_key_env") or "BCA_RERANK_API_KEY")
        base_url = os.environ.get(base_url_env)
        api_key = os.environ.get(api_key_env)
        if not base_url or not api_key:
            raise RuntimeError(f"Live reranking requires {base_url_env} and {api_key_env}.")
        provider_base_url = base_url
        provider = OpenAICompatibleProvider(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=reranker_timeout,
        )

    # Snapshot all inputs before model or retrieval work starts.
    provenance = _input_provenance(
        files=input_files,
        directories=input_directories,
    )
    _verify_integrity(mode=mode, config=config, provenance=provenance)

    if mode == "fixture":
        assert saved_rankings is not None
        dense_candidates, dense_summary = retrieve_from_saved_rankings(
            queries_path=queries_path,
            rankings_path=saved_rankings,
            fusion_top_k=retrieval_fusion_top_k,
            rrf_k=retrieval_rrf_k,
            weights=weights,
        )
    else:
        assert index_dir is not None
        index_manifest = DenseIndexManifest.load(
            index_dir,
            expected_model=retrieval_model,
            expected_revision=retrieval_revision,
        )
        if (
            retrieval_model != index_manifest.model
            or retrieval_revision != index_manifest.model_revision
        ):
            raise ValueError(
                "Configured embedding model/revision must exactly match the index manifest."
            )
        encoder_factory = partial(
            QwenEmbeddingEncoder,
            model=retrieval_model,
            revision=retrieval_revision,
            device=retrieval_device,
            batch_size=retrieval_batch_size,
            max_length=retrieval_max_length,
            local_files_only=not retrieval_allow_download,
        )
        dense_candidates, dense_summary = retrieve_from_index(
            queries_path=queries_path,
            index_dir=index_dir,
            encoder_factory=encoder_factory,
            model=retrieval_model,
            model_revision=retrieval_revision,
            top_k=retrieval_field_top_k,
            fusion_top_k=retrieval_fusion_top_k,
            rrf_k=retrieval_rrf_k,
            chunk_size=retrieval_chunk_size,
            device=retrieval_device,
            weights=weights,
        )
    atomic_write_jsonl(stage_paths["dense_candidates"], dense_candidates)
    atomic_write_json(stage_paths["dense_summary"], dense_summary)
    del dense_candidates, dense_summary

    queries = load_queries(queries_path)

    if mode == "fixture":
        assert document_replay is not None
        documents = _validate_document_replay(document_replay)
    else:
        assert document_root is not None
        documents = materialize(
            stage_paths["dense_candidates"],
            document_root,
            top_n_per_query=reranker_top_n,
        )
    atomic_write_jsonl(stage_paths["documents"], documents)
    del documents

    reranker_input = prepare(
        candidates_path=stage_paths["dense_candidates"],
        queries_path=queries_path,
        documents_path=stage_paths["documents"],
        summary_path=stage_paths["dense_summary"],
        top_n=reranker_top_n,
    )
    atomic_write_jsonl(stage_paths["reranker_input"], reranker_input)

    sparse_top_k = policy.bm25_top_k
    citation_candidates = discover_trace_citations(
        trace_root=sparse_trace_root,
        query_ids=queries,
        top_k=sparse_top_k,
    )
    extractor = CitationExtractor()
    for row in reranker_input:
        citation_candidates.update(extractor.extract(str(row.get("document_text") or "")))
    vocabulary = load_targeted_vocabulary(
        laws_path=laws_path,
        courts_path=courts_path,
        candidates=citation_candidates,
    )
    sparse_rows = aggregate_saved_traces(
        trace_root=sparse_trace_root,
        query_ids=queries,
        vocabulary=vocabulary,
        top_k=sparse_top_k,
    )
    atomic_write_jsonl(stage_paths["sparse_support"], sparse_rows)

    if reranker_replay is not None:
        score_rows = rerank_from_replay(
            reranker_input,
            replay_path=reranker_replay,
            model=reranker_model,
            batch_size=reranker_batch_size,
            document_char_limit=reranker_document_limit,
            max_tokens=reranker_max_tokens,
        )
        reranking_source = "fingerprint_bound_replay"
    else:
        assert provider is not None
        try:
            score_rows = await rerank_live(
                reranker_input,
                provider=provider,
                model=reranker_model,
                batch_size=reranker_batch_size,
                concurrency=reranker_concurrency,
                document_char_limit=reranker_document_limit,
                checkpoint_path=stage_paths["reranker_batches"],
                max_attempts=reranker_max_attempts,
                retry_delay_seconds=reranker_retry_delay,
                max_tokens=reranker_max_tokens,
            )
        finally:
            await provider.close()
        reranking_source = "live_https_provider"
    atomic_write_jsonl(stage_paths["reranker_scores"], score_rows)

    predictions, selection_audit = select_for_queries(
        query_ids=queries,
        prepared_rows=reranker_input,
        score_rows=score_rows,
        support_rows=sparse_rows,
        vocabulary=vocabulary,
        policy=policy,
    )
    atomic_write_submission(
        stage_paths["submission"],
        query_ids=queries,
        predictions=predictions,
    )
    atomic_write_json(stage_paths["selection_audit"], selection_audit)

    metrics: dict[str, Any] | None = None
    if _has_gold(queries_path):
        metrics = evaluate(stage_paths["submission"], queries_path)
        atomic_write_json(stage_paths["metrics"], metrics)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_name": str(config.get("run_name") or output_dir.name),
        "mode": mode,
        "seed": seed,
        "resume_requested": resume,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git": _git_state(),
        "command_line": list(sys.argv),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependency_versions": _dependency_versions(),
        "source_config_sha256": sha256_file(resolved_config),
        "effective_config": {
            "paths": {
                "queries": _relative(queries_path),
                "laws": _relative(laws_path),
                "courts": _relative(courts_path),
                "sparse_traces": _relative(sparse_trace_root),
                "index": _relative(index_dir) if index_dir is not None else None,
                "document_store": (_relative(document_root) if document_root is not None else None),
                "saved_rankings": (
                    _relative(saved_rankings) if saved_rankings is not None else None
                ),
                "documents_replay": (
                    _relative(document_replay) if document_replay is not None else None
                ),
                "reranker_replay": (
                    _relative(reranker_replay) if reranker_replay is not None else None
                ),
                "output": _relative(output_dir),
            },
            "retrieval": {
                "model": retrieval_model,
                "model_revision": retrieval_revision,
                "device": retrieval_device,
                "batch_size": retrieval_batch_size,
                "max_length": retrieval_max_length,
                "chunk_size": retrieval_chunk_size,
                "allow_model_download": retrieval_allow_download,
                "fields": list(DEFAULT_WEIGHTS),
                "field_top_k": retrieval_field_top_k,
                "fusion_top_k": retrieval_fusion_top_k,
                "rrf_k": retrieval_rrf_k,
                "weights": weights,
                "attention_implementation": "sdpa",
                "encoder_lifecycle": ("fresh_per_field" if mode == "full" else "saved_rankings"),
                "model_placement": "single_device_map" if mode == "full" else "saved_rankings",
            },
            "reranking": {
                "model": reranker_model,
                "top_n": reranker_top_n,
                "batch_size": reranker_batch_size,
                "concurrency": reranker_concurrency,
                "timeout_seconds": reranker_timeout,
                "document_char_limit": reranker_document_limit,
                "max_attempts": reranker_max_attempts,
                "retry_delay_seconds": reranker_retry_delay,
                "max_tokens": reranker_max_tokens,
                "json_response": True,
                "temperature": 0.0,
            },
            "citation_gate": {
                "candidate_top_k": policy.candidate_top_k,
                "minimum_dense_votes": policy.minimum_dense_votes,
                "anchor_top_k": policy.anchor_top_k,
                "anchor_minimum_score": policy.anchor_minimum_score,
                "sparse_top_k": policy.bm25_top_k,
                "minimum_sparse_votes": policy.minimum_bm25_votes,
            },
        },
        "retrieval": {
            "fields": list(DEFAULT_WEIGHTS),
            "weights": weights,
            "rrf_k": retrieval_rrf_k,
            "field_top_k": retrieval_field_top_k,
            "fusion_top_k": retrieval_fusion_top_k,
            "embedding_model": retrieval_model,
            "embedding_revision": retrieval_revision,
            "index_model_binding": (index_manifest.model_binding if mode == "full" else "fixture"),
            "attention_implementation": "sdpa",
            "encoder_lifecycle": "fresh_per_field" if mode == "full" else "saved_rankings",
            "model_placement": "single_device_map" if mode == "full" else "saved_rankings",
            "pooling": "last_token",
            "normalization": "l2",
            "matrix_dtype": "float16",
        },
        "reranking": {
            "model": reranker_model,
            "source": reranking_source,
            "prompt_version": PROMPT_VERSION,
            "batch_size": reranker_batch_size,
            "concurrency": reranker_concurrency,
            "timeout_seconds": reranker_timeout,
            "max_attempts": reranker_max_attempts,
            "retry_delay_seconds": reranker_retry_delay,
            "max_tokens": reranker_max_tokens,
            "json_response": True,
            "temperature": 0.0,
            "document_char_limit": reranker_document_limit,
            "provider_base_url": provider_base_url,
            "provider_models": sorted(
                {str(row["provider_model"]) for row in score_rows if row.get("provider_model")}
            ),
        },
        "citation_policy": policy.description,
        "inputs": provenance,
        "stages": [
            {"stage": 1, "name": "dense_retrieval", "status": "completed"},
            {"stage": 2, "name": "sparse_support", "status": "completed"},
            {"stage": 3, "name": "document_materialization", "status": "completed"},
            {"stage": 4, "name": "reranker_input", "status": "completed"},
            {"stage": 5, "name": "origin_verification", "status": "completed"},
            {"stage": 6, "name": "citation_selection", "status": "completed"},
            *(
                [{"stage": 7, "name": "evaluation", "status": "completed"}]
                if metrics is not None
                else []
            ),
        ],
        "outputs": {
            name: {
                "path": _relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in stage_paths.items()
            if name != "manifest" and path.exists()
        },
        "metrics": metrics,
    }
    atomic_write_json(stage_paths["manifest"], manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Recompute deterministic stages and reuse exact Stage-5 checkpoints.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = asyncio.run(
        run_pipeline(
            args.config,
            output_override=args.output,
            resume=args.resume,
        )
    )
    metrics = manifest.get("metrics") or {}
    suffix = f", macro_f1={float(metrics['macro_f1']):.6f}" if "macro_f1" in metrics else ""
    print(f"Completed {manifest['run_name']} ({manifest['mode']}){suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
