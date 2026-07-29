#!/usr/bin/env python3
"""Check that built wheels contain every intended Python package and no research data."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REQUIRED_MEMBERS = {
    "bettercallagent/__init__.py",
    "experiments/historical_validation/__init__.py",
    "interpretability/__init__.py",
    "interpretability/examples/gate_candidates.jsonl",
    "interpretability/examples/perturbations.jsonl",
    "interpretability/examples/predictions.jsonl",
    "interpretability/examples/queries.jsonl",
    "interpretability/examples/retrieval.jsonl",
    "offline/__init__.py",
    "offline/fixtures/config.toml",
    "offline/fixtures/courts.csv",
    "offline/fixtures/documents.jsonl",
    "offline/fixtures/laws.csv",
    "offline/fixtures/queries.csv",
    "offline/fixtures/reranker_replay.jsonl",
    "offline/fixtures/retrieval_rankings.jsonl",
    "offline/fixtures/sparse_traces/q_demo_001/trace.json",
    "offline/fixtures/sparse_traces/q_demo_002/trace.json",
    "online/__init__.py",
    "online/fixtures/demo.json",
}
FORBIDDEN_PARTS = {
    ".env",
    "__pycache__",
    "law_data",
    "node_modules",
    "tests",
}


def check_wheel(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    problems = [f"{path.name}: missing {member}" for member in sorted(REQUIRED_MEMBERS - names)]
    for name in sorted(names):
        if FORBIDDEN_PARTS.intersection(Path(name).parts):
            problems.append(f"{path.name}: forbidden member {name}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    wheels = sorted(args.directory.glob("bettercallagent-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"Expected one BetterCallAgent wheel, found {len(wheels)}.")
    problems = check_wheel(wheels[0])
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"Distribution check passed: {wheels[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
