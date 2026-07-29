from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.verify_artifacts import Artifact, load_manifest, sha256_tree, verify


def test_versioned_manifest_is_well_formed() -> None:
    artifacts = load_manifest(Path("artifacts/manifest.toml"))
    assert {artifact.artifact_id for artifact in artifacts} >= {
        "validation_queries",
        "law_vocabulary",
        "validation_rerank_scores",
        "validation_sparse_traces",
        "validation_support_metrics",
    }
    assert len({artifact.artifact_id for artifact in artifacts}) == len(artifacts)
    assert len({artifact.relative_path for artifact in artifacts}) == len(artifacts)
    assert all(artifact.size_bytes > 0 for artifact in artifacts)


def test_verifier_accepts_matching_file(tmp_path: Path) -> None:
    payload = b"synthetic fixture\n"
    path = tmp_path / "fixture.txt"
    path.write_bytes(payload)
    artifact = Artifact(
        artifact_id="fixture",
        relative_path=Path("fixture.txt"),
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert verify([artifact], tmp_path) == []


def test_verifier_reports_missing_file(tmp_path: Path) -> None:
    artifact = Artifact("missing", Path("missing.txt"), 1, "0" * 64)
    assert verify([artifact], tmp_path) == ["missing: missing missing.txt"]


def test_verifier_accepts_matching_tree(tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    (directory / "q1").mkdir(parents=True)
    (directory / "q1" / "trace.json").write_text("{}\n", encoding="utf-8")
    checksum, count, size = sha256_tree(directory)
    artifact = Artifact(
        "tree",
        Path("traces"),
        size,
        checksum,
        kind="tree",
        file_count=count,
    )
    assert verify([artifact], tmp_path) == []


def test_manifest_rejects_duplicates_and_non_positive_sizes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        """
schema_version = 1
[[artifact]]
id = "duplicate"
path = "one"
size_bytes = 1
sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
[[artifact]]
id = "duplicate"
path = "two"
size_bytes = 0
sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate artifact ID"):
        load_manifest(manifest)
