#!/usr/bin/env python3
"""Reject secrets, machine paths, generated files, and large release artifacts."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

MAX_TRACKED_BYTES = 10 * 1024 * 1024
FORBIDDEN_NAMES = {
    ".env",
    ".env.local",
    ".llm_env",
    "id_rsa",
    "id_ed25519",
}
FORBIDDEN_SUFFIXES = {".secret"}
FORBIDDEN_PARTS = {
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".runtime",
    ".venv",
    "__pycache__",
    "checkpoints",
    "dist",
    "htmlcov",
    "indexes",
    "logs",
    "models",
    "node_modules",
    "qdrant_storage",
    "runs",
    "snapshots",
}
TEXT_SUFFIXES = {
    ".cfg",
    ".cff",
    ".css",
    ".csv",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".lock",
    ".md",
    ".mjs",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
QUOTED_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(?:access[_-]?key|api[_-]?key|auth[_-]?token|password|private[_-]?key|"
    r"secret(?:[_-]?access[_-]?key)?|token)\s*[:=]\s*['\"]"
    r"(?P<value>[A-Za-z0-9_./+=:@-]{20,})['\"]"
)
ENV_ASSIGNMENT_PATTERN = re.compile(
    r"(?m)^\s*(?:export\s+)?"
    r"(?:[A-Z0-9_]*(?:ACCESS_KEY|API_KEY|AUTH_TOKEN|PASSWORD|PRIVATE_KEY|"
    r"SECRET_ACCESS_KEY|TOKEN))\s*=\s*"
    r"(?P<value>[A-Za-z0-9_./+=:@-]{20,})\s*(?:#.*)?$"
)
PRIVATE_KEY_PATTERN = re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")
PLACEHOLDER_MARKERS = {
    "changeme",
    "example",
    "fixture",
    "placeholder",
    "replace",
    "your-",
}
SCANNER_SOURCE = Path("scripts/check_release.py")
ABSOLUTE_CLUSTER_PATH = re.compile("/" + "mnt/" + "vast-kisski/" + r"(?:home|projects)/")


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def candidate_files(root: Path) -> list[Path]:
    """Return tracked plus untracked, non-ignored files in the working tree."""

    output = _git(
        root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    )
    return sorted(root / item.decode("utf-8") for item in output.split(b"\0") if item)


def history_entries(root: Path) -> list[tuple[str, Path]]:
    """List every named object reachable from a local Git ref."""

    output = _git(root, "rev-list", "--objects", "--all")
    entries: list[tuple[str, Path]] = []
    for raw_line in output.decode("utf-8", errors="replace").splitlines():
        object_id, separator, raw_path = raw_line.partition(" ")
        if separator and raw_path:
            entries.append((object_id, Path(raw_path)))
    return entries


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def credential_problem(text: str) -> bool:
    """Return true only for private-key blocks or non-placeholder assignments."""

    if PRIVATE_KEY_PATTERN.search(text):
        return True
    return any(
        not _looks_like_placeholder(match.group("value"))
        for pattern in (QUOTED_ASSIGNMENT_PATTERN, ENV_ASSIGNMENT_PATTERN)
        for match in pattern.finditer(text)
    )


def check_content(relative: Path, text: str) -> list[str]:
    problems: list[str] = []
    if relative != SCANNER_SOURCE and credential_problem(text):
        problems.append(f"{relative}: possible embedded credential")
    if ABSOLUTE_CLUSTER_PATH.search(text):
        problems.append(f"{relative}: machine-specific cluster path")
    return problems


def check_path(relative: Path, *, size: int | None) -> list[str]:
    problems: list[str] = []
    if relative.name in FORBIDDEN_NAMES or relative.suffix in FORBIDDEN_SUFFIXES:
        problems.append(f"{relative}: forbidden secret filename")
    if FORBIDDEN_PARTS.intersection(relative.parts):
        problems.append(f"{relative}: generated/runtime directory is present")
    if size is not None and size > MAX_TRACKED_BYTES:
        problems.append(f"{relative}: file is {size} bytes (limit {MAX_TRACKED_BYTES})")
    return problems


def check_worktree_file(root: Path, path: Path) -> list[str]:
    relative = path.relative_to(root)
    if not path.is_file():
        return check_path(relative, size=None)
    problems = check_path(relative, size=path.stat().st_size)
    if path.suffix.lower() not in TEXT_SUFFIXES and not path.name.endswith(
        (".env.example", ".env.sample")
    ):
        return problems
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return problems
    return [*problems, *check_content(relative, text)]


def check_history(root: Path, entries: Iterable[tuple[str, Path]]) -> list[str]:
    """Scan reachable historical blobs without printing their contents."""

    problems: list[str] = []
    scanned_objects: set[str] = set()
    for object_id, relative in entries:
        path_problems = check_path(relative, size=None)
        problems.extend(f"history: {problem}" for problem in path_problems)
        if (
            object_id in scanned_objects
            or relative == SCANNER_SOURCE
            or (
                relative.suffix.lower() not in TEXT_SUFFIXES
                and not relative.name.endswith((".env.example", ".env.sample"))
            )
        ):
            continue
        scanned_objects.add(object_id)
        try:
            size = int(_git(root, "cat-file", "-s", object_id))
        except (ValueError, subprocess.CalledProcessError):
            continue
        if size > MAX_TRACKED_BYTES:
            problems.append(
                f"history: {relative}: blob is {size} bytes (limit {MAX_TRACKED_BYTES})"
            )
            continue
        try:
            text = _git(root, "cat-file", "-p", object_id).decode("utf-8")
        except (UnicodeDecodeError, subprocess.CalledProcessError):
            continue
        problems.extend(f"history: {problem}" for problem in check_content(relative, text))
    return sorted(set(problems))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Only for an uninitialized tree; release CI must scan history.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    problems = [
        problem for path in candidate_files(root) for problem in check_worktree_file(root, path)
    ]
    if not args.skip_history:
        problems.extend(check_history(root, history_entries(root)))
    problems = sorted(set(problems))
    if problems:
        for problem in problems:
            print(f"FAIL  {problem}", file=sys.stderr)
        return 1
    print("Release checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
