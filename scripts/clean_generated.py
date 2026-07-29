#!/usr/bin/env python3
"""Remove only known, reproducible build and test outputs below this repository."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = (
    ROOT / ".pytest_cache",
    ROOT / ".ruff_cache",
    ROOT / ".venv",
    ROOT / "bettercallagent.egg-info",
    ROOT / "build",
    ROOT / "dist",
    ROOT / "frontend" / "coverage",
    ROOT / "frontend" / "dist",
    ROOT / "frontend" / "node_modules",
)


def main() -> int:
    removed: list[Path] = []
    for path in DIRECTORIES:
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(path.relative_to(ROOT))
    for path in ROOT.rglob("__pycache__"):
        if ".git" not in path.parts and path.is_dir():
            shutil.rmtree(path)
            removed.append(path.relative_to(ROOT))
    print(
        "Removed generated paths: "
        + (", ".join(str(path) for path in sorted(removed)) if removed else "none")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
