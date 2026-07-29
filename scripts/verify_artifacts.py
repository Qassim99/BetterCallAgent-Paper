#!/usr/bin/env python3
"""Verify external research artifacts against the versioned manifest."""

from __future__ import annotations

import argparse
import hashlib
import sys
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    relative_path: Path
    size_bytes: int
    sha256: str
    kind: str = "file"
    file_count: int | None = None


def load_manifest(path: Path) -> list[Artifact]:
    """Parse and validate the artifact records in a TOML manifest."""

    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError("Only artifact manifest schema_version=1 is supported.")

    artifacts: list[Artifact] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for record in payload.get("artifact", []):
        relative_path = Path(str(record["path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Artifact path must stay below the artifact root: {relative_path}")
        artifact_id = str(record["id"])
        if artifact_id in seen_ids:
            raise ValueError(f"Duplicate artifact ID: {artifact_id}")
        if relative_path in seen_paths:
            raise ValueError(f"Duplicate artifact path: {relative_path}")
        seen_ids.add(artifact_id)
        seen_paths.add(relative_path)
        checksum = str(record["sha256"]).lower()
        if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
            raise ValueError(f"Invalid SHA-256 for artifact {record.get('id')!r}.")
        size_bytes = int(record["size_bytes"])
        if size_bytes <= 0:
            raise ValueError(f"Artifact {artifact_id!r} must have a positive size.")
        kind = str(record.get("kind") or "file")
        if kind not in {"file", "tree"}:
            raise ValueError(f"Artifact {artifact_id!r} has unsupported kind {kind!r}.")
        file_count = int(record["file_count"]) if "file_count" in record else None
        if kind == "tree" and (file_count is None or file_count <= 0):
            raise ValueError(f"Tree artifact {artifact_id!r} requires file_count > 0.")
        if kind == "file" and file_count is not None:
            raise ValueError(f"File artifact {artifact_id!r} cannot set file_count.")
        artifacts.append(
            Artifact(
                artifact_id=artifact_id,
                relative_path=relative_path,
                size_bytes=size_bytes,
                sha256=checksum,
                kind=kind,
                file_count=file_count,
            )
        )
    if not artifacts:
        raise ValueError("The manifest contains no [[artifact]] records.")
    return artifacts


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> tuple[str, int, int]:
    """Hash a directory from sorted relative paths and per-file SHA-256 values."""

    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
        total_bytes += item.stat().st_size
    return digest.hexdigest(), len(files), total_bytes


def verify(artifacts: Iterable[Artifact], root: Path) -> list[str]:
    """Return human-readable verification failures."""

    failures: list[str] = []
    resolved_root = root.resolve()
    for artifact in artifacts:
        path = (resolved_root / artifact.relative_path).resolve()
        if not path.is_relative_to(resolved_root):
            failures.append(f"{artifact.artifact_id}: path escapes artifact root")
            continue
        if artifact.kind == "tree":
            if not path.is_dir():
                failures.append(
                    f"{artifact.artifact_id}: missing directory {artifact.relative_path}"
                )
                continue
            actual_hash, actual_count, actual_size = sha256_tree(path)
            if actual_count != artifact.file_count:
                failures.append(
                    f"{artifact.artifact_id}: file count {actual_count}, "
                    f"expected {artifact.file_count}"
                )
                continue
        else:
            if not path.is_file():
                failures.append(f"{artifact.artifact_id}: missing {artifact.relative_path}")
                continue
            actual_size = path.stat().st_size
            actual_hash = sha256_file(path)
        if actual_size != artifact.size_bytes:
            failures.append(
                f"{artifact.artifact_id}: size {actual_size}, expected {artifact.size_bytes}"
            )
            continue
        if actual_hash != artifact.sha256:
            failures.append(
                f"{artifact.artifact_id}: sha256 {actual_hash}, expected {artifact.sha256}"
            )
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifest.toml"),
        help="Versioned TOML artifact manifest.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/downloads"),
        help="Directory containing downloaded artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = load_manifest(args.manifest)
    failures = verify(artifacts, args.root)
    if failures:
        for failure in failures:
            print(f"FAIL  {failure}", file=sys.stderr)
        print(
            f"{len(failures)} of {len(artifacts)} artifacts failed verification.", file=sys.stderr
        )
        return 1
    print(f"Verified {len(artifacts)} artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
